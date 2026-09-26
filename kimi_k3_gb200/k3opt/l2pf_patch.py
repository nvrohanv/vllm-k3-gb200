"""vLLM integration of k3pf.prefetch_l2: cross-layer L2 prefetch of the next layer's attention weights.

Call ``patch_l2pf(load_ext)`` once per process at plugin-registration time (before model
construction). ``load_ext()`` must build/load l2pf.cu so that ``torch.ops.k3pf.prefetch_l2``
exists. ``K3PF_DISABLE=1`` turns the whole thing off (nothing is patched).
Other knobs: K3PF_MAX_TOKENS (16), K3PF_MODE=prefetch|tma (prefetch), K3PF_GRID (32 / 64 for tma),
K3PF_CHUNK (16384 / 32768 for tma), K3PF_POLICY (0 = plain L2 prefetch; 1 evict_last, 2 evict_first,
3 evict_normal cache hints -- measured no benefit).

What it does
------------
For every MoE layer ``i`` of ``KimiLinearModel`` whose MoE runs through ``LatentMoERunner``, right
after the routed experts (FC2) and the shared-expert join -- i.e. at the start of the MoE tail
(all-reduce+RMSNorm -> multicast up-proj -> Lamport copy -> next layer's AttnRes, ~14 us at M=1,
~15 us at M=8, all latency/NVLink bound) -- it forks a dedicated side stream and launches one
``k3pf.prefetch_l2`` kernel that pulls layer ``i+1``'s attention weights into L2, in consumption
order:

* KDA layer:  in_proj_qkvgfab (3216x7168, 46 MB) -> f_b_proj -> o_proj (7168x768, 11 MB)
* MLA layer:  fused_qkv_a_g_proj (qkv_a 2112 + gate 768 rows, 41 MB) -> q_b_proj -> W_UK_T ->
              W_UV -> o_proj

The side stream is joined once, at the end of ``KimiLinearModel.forward`` (the same pattern
vLLM's weight offloader uses), so the prefetch never sits on the critical path: its kernel
lives ~8-9 us (it stalls on the TMA prefetch queue), which is fully hidden behind the tail.

Why there (measured, see RESULTS.md): prefetched lines survive ~30 MB of competing traffic but
are mostly evicted by ~60-100 MB. Before FC1/FC2 (e.g. next to the shared experts) the prefetch
only works at M=1 (22 MB of expert weights); at M=8 the ~100 MB of expert weights evict it and
the net effect is negative. After FC2 it works for M=1..16.

K3PF_MOE (l2pf 2026-09-26, default off)
---------------------------------------
``K3PF_MOE=1`` adds a SECOND k3pf kernel (plain cp.async.bulk.prefetch.L2, evict_normal) on the same side stream
that pulls MoE-side dense weights into L2 for decode batches M in [K3PF_MOE_MIN_M, K3PF_MOE_MAX_M] (default 2..4):
  down   routed_expert_down_proj rows of this rank (K3OPT_DOWNSHARD shard, 3.2 MB): read cold by the vLLM
         down-shard GEMV on the aux stream (no PDL), on the M = 2..4 critical path into the H2 hop
  sdown  shared_experts.down_proj (7168 x 384, 5.5 MB): read by the routing-only block kernel after route_shared
  router gate.weight (896 x 7168, 12.8 MB) and shared  shared_experts.gate_up_proj (768 x 7168, 11 MB):
         route_shared stages both into smem before its PDL wait, during the o_proj / H1 section, with slack
Issue point K3PF_MOE_AT:
  oproj  (default) a fork at this layer's o_proj call (o_proj.forward is wrapped; with oproj_patch that call only
         returns the marker, right after the attention core is launched): the prefetch of THIS layer's MoE
         weights starts when kda_split / attn_out completes, i.e. during the fused o_proj + H1 hop, ~7 us before
         the down-shard GEMV, and off in_proj (where the attention prefetch's value is: in-server B2 profile)
  attn   appended after the attention prefetch in the post-FC2 fork (prefetches the NEXT layer's MoE weights;
         runs on top of that layer's in_proj -- measured worse)
1-GPU chain (agents/l2pf/RESULTS.md "K3PF_MOE"): "down" at the o_proj point -0.3..-1.0 us/layer (M = 2..4, KDA and
MLA); "sdown" adds nothing, "router"/"shared" cost (they compete with route_shared's own staging), so opt-in only.
Knobs: K3PF_MOE_W (comma set, table order = the given order; default "down"), K3PF_MOE_AT (oproj | attn),
K3PF_MOE_MIN_M / K3PF_MOE_MAX_M (default 2..4; M = 1 would overlap the front kernel's pre-wait staging),
K3PF_MOE_GRID (default K3PF_GRID). The attention fork is unchanged. The tables are built with the attention ones
(lazily, at the first eligible eager MoE), so K3PF_MOE is inactive wherever the attention fork never runs
(K3PF_DISABLE=1, K3T1_PF=off); with K3PF_MOE_AT=attn it is also skipped wherever launch_after_moe is skipped.

Safety
------
* Prefill / large batches: skipped when num_tokens > K3PF_MAX_TOKENS (default 16).
* CUDA graphs: FULL capture and eager are supported (static addresses: the (ptr, bytes) table is
  a device tensor built once, lazily, at the first eligible forward that is not being captured --
  vLLM always runs an eager warmup before each capture -- so the k3snap in-place restore and any
  post-load re-layout have already happened). Under breakable piecewise capture
  (``BreakableCUDAGraphCapture.is_active()``) and under torch.compile tracing the prefetch is
  skipped, so no stream is ever left unjoined at a segment boundary.
* Last layer: no prefetch (the lm_head runs outside the graph and is larger than L2).
* The prefetch only reads weights; it never changes results.
"""

from __future__ import annotations

import os
import weakref

import torch

_STATE = {
    "patched": False,
    "stream": None,
    "pending": False,
    "orig_forward_impl": None,
    "orig_model_forward": None,
    "orig_model_init": None,
    # id(runner) -> (ranges tensor, n, bytes, weakref(runner))
    "tables": {},
    "built": False,
    "events": {},
    "stats": {"launched": 0, "skipped_tokens": 0, "skipped_capture_unbuilt": 0,
              "skipped_breakable": 0, "joins": 0},
}
_MODELS: "weakref.WeakSet" = weakref.WeakSet()

MAX_TOKENS = int(os.environ.get("K3PF_MAX_TOKENS", "16"))
MOE_ON = os.environ.get("K3PF_MOE", "0") == "1"
MOE_SET = [k.strip() for k in os.environ.get("K3PF_MOE_W", "down").split(",") if k.strip()]
MOE_MIN_M = int(os.environ.get("K3PF_MOE_MIN_M", "2"))
MOE_MAX_M = int(os.environ.get("K3PF_MOE_MAX_M", "4"))
# K3PF_MOE_AT: issue point of the MoE prefetch. "attn" = appended after the attention prefetch in the post-FC2 fork
# (prefetches layer i+1's MoE weights; runs during layer i+1's in_proj). "oproj" = a fork at the o_proj call of the
# same layer (after the attention core, i.e. during the o_proj / H1 hop section; prefetches that layer's own MoE
# weights), same side stream, joined with it at the end of the forward.
MOE_AT = os.environ.get("K3PF_MOE_AT", "oproj")
# K3PF_MODE=prefetch (default): cp.async.bulk.prefetch.L2, 32 one-warp CTAs, no smem.
# K3PF_MODE=tma (opt-in): TMA bulk-load fill (policy 4), 64 CTAs x 128 KB smem ring for ~9 us; ~1 us/layer
#   faster GEMMs in the emulation but occupies smem on 64 SMs during the MoE tail -- validate in the real chain.
MODE = os.environ.get("K3PF_MODE", "prefetch")
PDL = int(os.environ.get("K3PF_PDL", "0"))
GRID = int(os.environ.get("K3PF_GRID", "64" if MODE == "tma" else "32"))
CHUNK = int(os.environ.get("K3PF_CHUNK", "32768" if MODE == "tma" else "16384"))
POLICY = 4 if MODE == "tma" else int(os.environ.get("K3PF_POLICY", "0"))
MOE_GRID = int(os.environ.get("K3PF_MOE_GRID", str(GRID)))


def _enabled() -> bool:
    return os.environ.get("K3PF_DISABLE", "0") != "1"


def _span(t: torch.Tensor) -> tuple[int, int] | None:
    """(ptr, bytes) covering a (possibly strided, non-negative-stride) CUDA tensor view."""
    if not isinstance(t, torch.Tensor) or not t.is_cuda or t.numel() == 0:
        return None
    if any(s < 0 for s in t.stride()):
        return None
    last = sum((sz - 1) * st for sz, st in zip(t.shape, t.stride()))
    return int(t.data_ptr()), int((last + 1) * t.element_size())


def _weight(mod, name):
    m = getattr(mod, name, None)
    if m is None:
        return None
    w = getattr(m, "weight", None)
    return w if isinstance(w, torch.Tensor) else None


def _attn_tensors(layer) -> list[torch.Tensor]:
    """Next layer's attention-block weights, in the order its decode path reads them."""
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        return []
    out = []
    for name in ("in_proj_qkvgfab", "fused_qkv_a_g_proj", "fused_qkv_a_proj", "q_proj",
                 "kv_a_proj_with_mqa", "f_b_proj", "q_b_proj", "g_proj"):
        w = _weight(attn, name)
        if w is not None:
            out.append(w)
    for name in ("W_UK_T", "W_UV"):
        w = getattr(attn, name, None)
        if isinstance(w, torch.Tensor):
            out.append(w)
    w = _weight(attn, "o_proj")
    if w is not None:
        out.append(w)
    return out


def _moe_tensors(layer) -> list[torch.Tensor]:
    """K3PF_MOE: the layer's MoE-side dense weights in K3PF_MOE_W order (see the module docstring)."""
    moe = getattr(layer, "mlp", None)
    if moe is None:
        return []
    sh = getattr(moe, "shared_experts", None)
    out = []
    for k in MOE_SET:
        w = None
        if k == "down":
            down = getattr(moe, "routed_expert_down_proj", None)
            dw = getattr(down, "weight", None)
            if isinstance(dw, torch.Tensor) and getattr(down, "_k3opt_sharded", False) and dw.dim() == 2:
                from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

                tp, rank = get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()
                if dw.shape[0] % tp == 0:
                    shard = dw.shape[0] // tp
                    w = dw.narrow(0, rank * shard, shard)
        elif k == "sdown":
            w = _weight(sh, "down_proj") if sh is not None else None
        elif k == "router":
            w = _weight(moe, "gate")
        elif k == "shared":
            w = _weight(sh, "gate_up_proj") if sh is not None else None
        if isinstance(w, torch.Tensor):
            out.append(w)
    return out


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Drop duplicates / overlaps while keeping first-use order."""
    out: list[tuple[int, int]] = []
    for p, b in spans:
        if b <= 0:
            continue
        dup = False
        for q, c in out:
            if q <= p and p + b <= q + c:
                dup = True
                break
        if not dup:
            out.append((p, b))
    return out


def _runner_of(layer):
    from vllm.models.kimi_k3.nvidia.latent_moe_runner import LatentMoERunner

    runner = getattr(getattr(layer, "mlp", None), "experts", None)
    return runner if isinstance(runner, LatentMoERunner) else None


def ms_dev(layer) -> torch.device:
    ts = _moe_tensors(layer)
    return ts[0].device if ts else torch.device("cuda")


def launch_moe_at_oproj(o_proj, num_tokens: int) -> bool:
    """K3PF_MOE_AT=oproj: fork the side stream at this o_proj call and prefetch this layer's MoE weights."""
    if not (MOE_MIN_M <= num_tokens <= MOE_MAX_M) or num_tokens > MAX_TOKENS:
        return False
    if torch.compiler.is_compiling() or not _STATE["built"]:
        return False  # tables are built lazily at the first eligible MoE of an eager forward, never under capture
    me = _STATE.get("moe_tables", {}).get(id(o_proj))
    if me is None or me[3]() is not o_proj:
        return False
    if torch.cuda.is_current_stream_capturing() and _breakable_active():
        _STATE["stats"]["skipped_breakable"] += 1
        return False
    cur = torch.cuda.current_stream()
    key = ("moe", id(o_proj))
    ev = _STATE["events"].get(key)
    if ev is None:
        ev = torch.cuda.Event()
        _STATE["events"][key] = ev
    ev.record(cur)
    s = _stream(me[0].device)
    s.wait_event(ev)
    _STATE["pending"] = True  # joined with the attention prefetch at the end of the model forward
    with torch.cuda.stream(s):
        torch.ops.k3pf.prefetch_l2_cfg(me[0], me[1], MOE_GRID, CHUNK, 0, 0)
    _STATE["stats"]["moe_launched"] = _STATE["stats"].get("moe_launched", 0) + 1
    return True


def _wrap_oproj(o_proj) -> None:
    """Wrap o_proj.forward (the instance attribute set by oproj_patch, or the bound method) once."""
    if getattr(o_proj, "_k3pf_moe", False):
        return
    inner = o_proj.forward

    def forward(*args, **kwargs):
        x = args[0] if args else kwargs.get("input_")
        try:
            if isinstance(x, torch.Tensor) and x.dim() >= 1:
                launch_moe_at_oproj(o_proj, int(x.shape[0]))
        except Exception as e:  # noqa: BLE001  (never break the forward because of a hint)
            print(f"[k3opt] K3PF_MOE: o_proj-time prefetch failed ({e!r}); disabling", flush=True)
            _STATE["moe_tables"] = {}
        return inner(*args, **kwargs)

    o_proj.forward = forward
    o_proj._k3pf_moe = True


def build_tables(model=None) -> int:
    """(Re)build the per-runner (ptr, bytes) tables. Must not run under stream capture."""
    assert not torch.cuda.is_current_stream_capturing()
    models = [model] if model is not None else list(_MODELS)
    tables = {}
    moe_tables = {}
    total = 0
    for m in models:
        start, end = getattr(m, "start_layer", 0), getattr(m, "end_layer", len(m.layers))
        for idx in range(start, end):
            runner = _runner_of(m.layers[idx])
            if runner is None:
                continue
            nxt = idx + 1
            if nxt >= end:
                continue  # last layer: nothing useful to prefetch inside the graph
            tensors = _attn_tensors(m.layers[nxt])
            spans = _merge([s for s in (_span(t) for t in tensors) if s is not None])
            if not spans:
                continue
            dev = tensors[0].device
            rng = torch.tensor(spans, dtype=torch.int64).to(dev)
            nbytes = sum(b for _, b in spans)
            tables[id(runner)] = (rng, len(spans), nbytes, weakref.ref(runner))
            total += 1
            if MOE_ON and MOE_AT == "attn":
                ms = _merge([s for s in (_span(t) for t in _moe_tensors(m.layers[nxt])) if s is not None])
                if ms:
                    moe_tables[id(runner)] = (torch.tensor(ms, dtype=torch.int64).to(dev), len(ms),
                                              sum(b for _, b in ms))
        if MOE_ON and MOE_AT == "oproj":
            for idx in range(start, end):
                layer = m.layers[idx]
                o_proj = getattr(getattr(layer, "self_attn", None), "o_proj", None)
                if o_proj is None or _runner_of(layer) is None:
                    continue
                ms = _merge([s for s in (_span(t) for t in _moe_tensors(layer)) if s is not None])
                if ms:
                    moe_tables[id(o_proj)] = (torch.tensor(ms, dtype=torch.int64).to(ms_dev(layer)), len(ms),
                                              sum(b for _, b in ms), weakref.ref(o_proj))
                    _wrap_oproj(o_proj)
    _STATE["tables"] = tables
    _STATE["moe_tables"] = moe_tables
    _STATE["built"] = True
    if total:
        mb = sum(t[2] for t in tables.values()) / total / 2**20
        print(f"[k3opt] K3PF: {total} MoE layers prefetch the next layer's attention weights "
              f"(avg {mb:.1f} MB, mode {MODE}, grid {GRID}, chunk {CHUNK >> 10} KB)", flush=True)
    if MOE_ON:
        n = len(moe_tables)
        mb2 = (sum(t[2] for t in moe_tables.values()) / n / 2**20) if n else 0.0
        where = ("the next layer's MoE weights, after the attention prefetch" if MOE_AT == "attn" else
                 "their own MoE weights, forked at the o_proj call")
        print(f"[k3opt] K3PF_MOE: {n} MoE layers prefetch {where} {MOE_SET} "
              f"(avg {mb2:.1f} MB, M {MOE_MIN_M}..{MOE_MAX_M}, grid {MOE_GRID})", flush=True)
    return total


def _stream(device) -> torch.cuda.Stream:
    s = _STATE["stream"]
    if s is None:
        s = torch.cuda.Stream(device=device)
        _STATE["stream"] = s
    return s


def _breakable_active() -> bool:
    try:
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    except Exception:  # noqa: BLE001
        return False
    return bool(BreakableCUDAGraphCapture.is_active())


def launch_after_moe(runner, num_tokens: int) -> bool:
    """Fork the prefetch stream at this point of the current stream (call after FC2)."""
    if num_tokens <= 0 or num_tokens > MAX_TOKENS:
        _STATE["stats"]["skipped_tokens"] += 1
        return False
    if torch.compiler.is_compiling():
        return False
    capturing = torch.cuda.is_current_stream_capturing()
    if not _STATE["built"]:
        if capturing:
            _STATE["stats"]["skipped_capture_unbuilt"] += 1
            return False
        build_tables()
    entry = _STATE["tables"].get(id(runner))
    if entry is None or entry[3]() is not runner:
        return False
    if capturing and _breakable_active():
        _STATE["stats"]["skipped_breakable"] += 1
        return False
    rng, n, _, _ = entry
    cur = torch.cuda.current_stream()
    ev = _STATE["events"].get(id(runner))
    if ev is None:
        ev = torch.cuda.Event()
        _STATE["events"][id(runner)] = ev
    ev.record(cur)
    s = _stream(rng.device)
    s.wait_event(ev)
    _STATE["pending"] = True  # from here on the side stream must be joined (capture validity)
    with torch.cuda.stream(s):
        # K3PF_PDL=1 launches with programmatic serialization. Off by default: the side stream is off
        # the critical path, and under PDL the first prefetch of a step (layer 1) could start early
        # and linger ~200 us, holding SMs that layer 2's persistent MoE kernel needs (agents/probe).
        torch.ops.k3pf.prefetch_l2_cfg(rng, n, GRID, CHUNK, POLICY, PDL)
        if MOE_ON and MOE_AT == "attn" and MOE_MIN_M <= num_tokens <= MOE_MAX_M:
            me = _STATE.get("moe_tables", {}).get(id(runner))
            if me is not None:  # K3PF_MOE: after the attention prefetch, same stream, plain (evict_normal) policy
                torch.ops.k3pf.prefetch_l2_cfg(me[0], me[1], MOE_GRID, CHUNK, 0, 0)
                _STATE["stats"]["moe_launched"] = _STATE["stats"].get("moe_launched", 0) + 1
    _STATE["stats"]["launched"] += 1
    return True


def join() -> None:
    """Join the prefetch stream into the current stream (end of the model forward)."""
    if not _STATE["pending"]:
        return
    s = _STATE["stream"]
    if s is not None:
        torch.cuda.current_stream().wait_stream(s)
    _STATE["pending"] = False
    _STATE["stats"]["joins"] += 1


def _forward_impl(self, hidden_states, *args, **kwargs):
    """LatentMoERunner._forward_impl: routed experts + shared join, then fork the prefetch."""
    out = _STATE["orig_forward_impl"](self, hidden_states, *args, **kwargs)
    try:
        launch_after_moe(self, int(hidden_states.shape[0]))
    except Exception as e:  # noqa: BLE001  (never break the forward because of a hint)
        print(f"[k3opt] K3PF: prefetch launch failed ({e!r}); disabling", flush=True)
        _STATE["tables"] = {}
        _STATE["built"] = True
    return out


def patch_l2pf(load_ext) -> None:
    """Install the integration (idempotent). Call before model construction in every worker."""
    if _STATE["patched"] or not _enabled():
        return
    load_ext()
    if not hasattr(torch.ops.k3pf, "prefetch_l2"):
        raise RuntimeError("k3pf.prefetch_l2 not loaded (build l2pf.cu)")

    from vllm.models.kimi_k3.nvidia import latent_moe_runner as lmr
    from vllm.models.kimi_k3.nvidia import model as k3_model

    Runner = lmr.LatentMoERunner
    Model = k3_model.KimiLinearModel
    _STATE["orig_forward_impl"] = Runner._forward_impl
    Runner._forward_impl = _forward_impl

    orig_forward = Model.forward
    orig_init = Model.__init__
    _STATE["orig_model_forward"] = orig_forward
    _STATE["orig_model_init"] = orig_init

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        _MODELS.add(self)
        _STATE["built"] = False  # (re)build lazily once weights are final

    def forward(self, *args, **kwargs):
        try:
            return _STATE["orig_model_forward"](self, *args, **kwargs)
        finally:
            join()

    Model.__init__ = __init__
    Model.forward = forward
    _STATE["patched"] = True
    print(f"[k3opt] K3PF: cross-layer L2 prefetch of next-layer attention weights "
          f"(k3pf.prefetch_l2, M<={MAX_TOKENS}, grid {GRID})", flush=True)


def stats() -> dict:
    """Python-side counters (eager/capture-time calls only; graph replays don't run Python)."""
    return dict(_STATE["stats"])
