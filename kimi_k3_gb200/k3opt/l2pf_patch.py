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
# K3PF_MODE=prefetch (default): cp.async.bulk.prefetch.L2, 32 one-warp CTAs, no smem.
# K3PF_MODE=tma (opt-in): TMA bulk-load fill (policy 4), 64 CTAs x 128 KB smem ring for ~9 us; ~1 us/layer
#   faster GEMMs in the emulation but occupies smem on 64 SMs during the MoE tail -- validate in the real chain.
MODE = os.environ.get("K3PF_MODE", "prefetch")
GRID = int(os.environ.get("K3PF_GRID", "64" if MODE == "tma" else "32"))
CHUNK = int(os.environ.get("K3PF_CHUNK", "32768" if MODE == "tma" else "16384"))
POLICY = 4 if MODE == "tma" else int(os.environ.get("K3PF_POLICY", "0"))


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


def build_tables(model=None) -> int:
    """(Re)build the per-runner (ptr, bytes) tables. Must not run under stream capture."""
    assert not torch.cuda.is_current_stream_capturing()
    models = [model] if model is not None else list(_MODELS)
    tables = {}
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
    _STATE["tables"] = tables
    _STATE["built"] = True
    if total:
        mb = sum(t[2] for t in tables.values()) / total / 2**20
        print(f"[k3opt] K3PF: {total} MoE layers prefetch the next layer's attention weights "
              f"(avg {mb:.1f} MB, mode {MODE}, grid {GRID}, chunk {CHUNK >> 10} KB)", flush=True)
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
        if GRID == 32 and CHUNK == 16384 and POLICY == 0:
            torch.ops.k3pf.prefetch_l2(rng, n)
        else:
            torch.ops.k3pf.prefetch_l2_cfg(rng, n, GRID, CHUNK, POLICY, 1)
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
