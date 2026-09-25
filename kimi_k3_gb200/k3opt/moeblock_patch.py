"""K3 MoE block restructure for decode batches M <= 2 (on top of K3OPT_DOWNSHARD + K3OPT_MOEFUSED).

Call ``patch_moeblock(load_ext)`` from ``k3opt.register()`` AFTER the K3OPT_DOWNSHARD /
K3OPT_MOEFUSED / K3OPT_TAILATTN patches. ``load_ext()`` must build csrc/moe_small.cu (the k3moe
library with ``route_shared`` and ``moe_block_lamport``). ``K3MOEBLOCK_DISABLE=1`` turns it off;
``K3MOEBLOCK_GRID=<n>`` runs the MoE kernel on at most n CTAs (default: all SMs).

What ``KimiMoE.forward`` does for an eligible layer and 1 <= M <= 2 (static per CUDA graph):

    main:  ev0.record()
    aux :  ev0.wait(); down-shard producer (k3opt's AdaptiveUpProjectionKernel, the existing
           multi-rank multicast op; returns the Lamport mailbox, NO LamportCopy); ev1.record()
    main:  k3moe.route_shared   router GEMV (+bias, sigmoid) -> scores [M,896,2] fp32, and the
                                shared expert's gate_up GEMV + SiTU -> h_shared [M,384] bf16
           ev1.wait()           (the producer finishes at about the same time as route_shared)
           k3moe.moe_block_lamport  per-CTA top-16 (== vLLM fused_grouped_topk: sigmoid,
                                1 group, renormalize, routed_scaling_factor), latent polled from
                                the mailbox (re-armed), FC1+SiTU+FC2 unfinalized [M*16,3584],
                                shared down_proj partial [M,7168], ids/weights [M,16] for the tail
           runner._small_batch_tail(UnfinalizedMoEOutput(...), shared_partial, None)
                                (the existing MoE-tail path; tailattn's wrapper included)

This replaces: router GEMV, fused_grouped_topk, 2 copy kernels, LamportCopy, the event-join
launch gap, shared gate_up / situ_and_mul / down_proj (3 kernels, one of them blocked behind the
MoE kernel) and moe_fused_unfinalized: 9 kernels -> 3 (+ the unchanged producer and tail).

Everything else (M > 2, prefill, MegaMoE, sequence parallel, non-latent MoE, other quant
layouts, missing tail fusion, zero-expert routers, unexpected shapes/dtypes, breakable-graph
capture for the stream split) falls back to the original forward, or to a single-stream variant.

FC1/FC2 via k3moebody (agents/moe8/body, W1-4; K3MOEBLOCK_MOEBODY=1, for M in K3MOEBLOCK_MOEBODY_MS, default "2"):
    the routing-only block kernel (latent_out mode: top-16, shared expert, latent hand-off + mailbox re-arm), then
    k3moebody.expert_body (tcgen05 MXFP4 x MXFP8, cluster groups with DSMEM hand-offs, no grid barrier), mode 0
    = unfinalized gemm2 [M*16, 3584] for the unchanged tail; bit-identical to k3moe8's gemm2 (same MXFP8 numerics).
    It takes precedence over k3moe8 for those M. It reads latent / ids only after griddepcontrol.wait (plain PDL,
    no flag protocol needed). Its workspace is one process-wide 0xFF-armed buffer (per-CTA call counters inside;
    calls are ordered by the stream). One-GPU chain (block_chain.py): M=2 34.77 us vs 36.06 (routing-only + moe8)
    and 34.87 (full block); M=1 29.78 vs 28.69 (full block), so M=1 keeps the full CUDA-core block by default.

MoE-tail replacement (experimental, default off; K3MOEBLOCK_TAIL=fused|kernel, TP16, M <= 4):
    kernel  after the MoE kernel (moe_block_lamport or k3moe8): k3moe.moe_tail, ONE 7x8-CTA
            cluster kernel doing vLLM's CollectiveKernel + AdaptiveUpProjectionKernel (top-16
            finalize, NVLS latent all-reduce, RMSNorm, shared reduce-scatter, up-proj, multicast into
            vLLM's up-proj mailbox) with the same rounding points
    fused   (only where the block kernel runs FC1/FC2 itself, i.e. not the k3moe8 path)
            k3moe.moe_block_tail: the same tail inside the persistent MoE kernel
    Either writes vLLM's up-proj mailbox; the result is handed to tailattn's deferred consumer
    (mailbox view) when the runner is flagged for it, else copied out with vLLM's LamportCopy.
    Needs two extra symmetric mailboxes (allocated collectively in KimiMoE.__init__).
    k3mk    (DESIGN_PLAN W1-2 "tail2", agents/moefused/tail2.cu, namespace k3mk; UNTESTED in the server)
            the block kernel's PUBLISH epilogue (k3mk.moe_block_lamport(..., lat_mb, ...): finalized partial
            latent multicast into lat_mb[par][rank], shared down rows into rs_mb[par][rank] of their owner)
            + k3mk.tail (clusters of 8, K-split; variant via K3MK_TAIL_VARIANT = 14 (default) | 7 | 114 (14,
            W_up staged after the RMS exchange) | 4 (8 x 4-CTA clusters; needs K3MOEBLOCK_GRID=112) | 1 (flat));
            on the k3moe8 path k3mk.tail runs in front mode (finalizes gemm2 itself) only with K3MK_FRONT=1
            (else those layers keep moe8 + vLLM's tail). par = layer_idx % 3.

MoE FRONT END (DESIGN_PLAN W1-3 "moefront", agents/oproj/front; K3MOEFRONT=1, default off; TP16, M <= 2):
    Needs agents/oproj/front/moe_front.cu (namespace k3mf) + oproj_front.cu (namespace k3oprojf) built, and
    agents/oproj/front/oproj_patch.py instead of k3opt/oproj_patch.py (its post-attention consume / fused
    kernel publishes x_moe for the MoE front kernel). For an eligible layer and 1 <= M <= K3MOEFRONT_MAX_M
    (default 2) ONE kernel, k3mf.moe_block_front, replaces [aux-stream down-shard kernel, route_shared,
    moe_block_lamport]: router GEMV + exact top-16, the latent down-shard GEMV + NVLS multicast into the
    down-shard mailbox, shared gate_up + SiTU + down_proj, FC1/FC2. It writes gemm2 / ids / wts / shared_out
    like moe_block_lamport, so every tail mode can follow it:
      K3MOEBLOCK_TAIL unset  -> vLLM's tail (runner._small_batch_tail)
      K3MOEBLOCK_TAIL=kernel -> k3moe.moe_tail
      K3MOEBLOCK_TAIL=k3mk   -> k3mk.tail in FRONT mode (finalizes gemm2 / wts / shared_out itself); this is
                                independent of K3MK_FRONT, which only concerns the k3moe8 path
    K3MOEFRONT_FLAGS=1 stages the shared gate_up rows after the router step (less HBM traffic before x_moe).
    Protocol: the post-attention op publishes x_moe for layer L iff layer L's MoE runs the front kernel (every
    publish must be consumed, else the next front layer of the same parity reads stale data): the publish
    decision (static per-layer eligibility front_layer_par + M <= K3MOEFRONT_MAX_M) is handed to the MoE
    forward of the same call (moe._k3front_expect), which runs the front kernel exactly then. All MoE layers
    must be eligible (K3's are uniform): if the first layer checked is not, the front end is switched off
    for the whole model; a later non-eligible layer raises (mixed layers would break the parity protocol).

VALID ENV COMBINATIONS (K3MOEBLOCK on, TP16). Per decode batch size M the first matching row of this list
decides the MoE path; the tail mode (K3MOEBLOCK_TAIL) then applies to whatever produced gemm2:
    1. K3MOEFRONT=1 and M <= K3MOEFRONT_MAX_M (default 2, <= K3MOEBLOCK_MAX_M):
         k3mf.moe_block_front (routing + down-shard + shared expert + its own CUDA-core FC1/FC2 in one kernel)
         tail: unset -> vLLM tail | kernel -> k3moe.moe_tail | k3mk -> k3mk.tail FRONT mode (always, whatever
         K3MK_FRONT says) | fused -> not combinable: K3MOEFRONT switches itself off at start-up (logged)
         Precedence: the front kernel wins over K3MOEBLOCK_MOEBODY / K3MOEBLOCK_MOE8 for these M. To keep an M on
         k3moebody / k3moe8, lower K3MOEFRONT_MAX_M (e.g. K3MOEFRONT_MAX_M=1 keeps M=2 on k3moebody).
    2. K3MOEBLOCK_MOEBODY=1 and M in K3MOEBLOCK_MOEBODY_MS (default "2"): routing-only block kernel + k3moebody
         tail: unset -> vLLM | k3mk -> k3mk.tail front mode only with K3MK_FRONT=1, else vLLM | kernel -> moe_tail
    3. K3MOEBLOCK_MOE8=1 (M >= K3MOEBLOCK_MOE8_MIN_M): routing-only block kernel + k3moe8; tails as in row 2
    4. otherwise (M <= K3MOEBLOCK_MAX_M): route_shared + moe_block_lamport (CUDA-core block)
         tail: unset -> vLLM | k3mk -> PUBLISH-mode block kernel + k3mk.tail | kernel -> moe_tail |
         fused -> moe_block_tail
    Row 1 was measured with the vLLM tail on 16 ranks (test_dist_front.py chain: -4.3 us/layer at M=1, -1.1 at
    M=2 vs row 4 + vLLM tail); row 1 with the k3mk / kernel tails and with MOEBODY/MOE8 set is written but not
    yet run. Rows 2-4: see agents/moe8/body and agents/moefused.
    Every rank must use the same env (eligibility decisions and mailbox parities must agree across ranks).
    Needs, in k3opt/csrc + SOURCES: moe_front.cu (k3mf) and oproj_front.cu (k3oprojf) for K3MOEFRONT; tail2.cu
    (k3mk) for K3MOEBLOCK_TAIL=k3mk; moebody.cu (k3moebody) for K3MOEBLOCK_MOEBODY. K3MOEFRONT also needs
    agents/oproj/front/oproj_patch.py as k3opt/oproj_patch.py (it publishes x_moe).

Kernel-level merges still to do:
  * W1-3 x W1-2: in K3MOEFRONT + k3mk mode the front kernel writes gemm2 / shared_out and k3mk.tail (front mode)
    re-reads and finalizes them. Better: the front kernel with tail2.cu's PUBLISH epilogue: FC2 results kept in
    smem as `fin` [unit][16][8] bf16 and handed to tail2's moe_publish_latent(fin, wsm, ...) with the front
    kernel's own bf16 routing weights, shared down rows sent as 16-byte fragments straight to the owner rank's
    rs_mb, no gemm2_out / shared_out; then k3mk.tail in publish mode (W1-2 measured publish 0.5-1 us/layer better
    than front mode at M=2, about equal at M=1).
  * W1-3 x W1-4: a routing-only mode of the front kernel (no FC1/FC2: its latent poll already copies the polled
    latent rows out (lat_dbg); every token row must then be polled by some CTA and CTA 0 re-arms after all
    have read it), followed by k3moebody.expert_body on that latent, gives routing-first + the barrier-free
    tcgen05 body in two kernels; the full merge puts the body after the front end in one kernel.
"""

from __future__ import annotations

import os

import torch

_STATE = {"patched": False, "orig_forward": None, "barriers": {}, "stats": {"fused": 0, "fallback": 0, "front": 0}}
# Largest M routed through the block path (kernels support 1..4); K3MOEBLOCK_MAX_M overrides
# for A/B in the server. Default = best measured (see RESULTS.md).
_DEFAULT_MAX_M = 3
_MAX_M = max(0, min(4, int(os.environ.get("K3MOEBLOCK_MAX_M", str(_DEFAULT_MAX_M)))))
_TOPK = 16
_E = 896
_H = 7168
_LAT = 3584
_SH = 384


def _enabled() -> bool:
    return os.environ.get("K3MOEBLOCK_DISABLE", "0") != "1"


_TAIL_MODE = os.environ.get("K3MOEBLOCK_TAIL", "0").strip().lower()
if _TAIL_MODE in ("0", "", "off", "none"):
    _TAIL_MODE = ""
_TAIL = {}  # symmetric mailboxes + scratch for the tail replacement (one set per process)
_KEEPALIVE: list = []  # inputs of the last tail call (the PDL consumer may start early; see tailattn)


def _tail_buffers():
    """Collective (all TP ranks, at model construction): mailboxes of k3moe.moe_tail / moe_block_tail."""
    if "lat" in _TAIL:
        return _TAIL
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem
    from vllm.distributed import get_tp_group

    group = get_tp_group().device_group
    tp = dist.get_world_size(group)
    if tp != 16:
        _TAIL["lat"] = None
        return _TAIL
    dev = torch.device("cuda", torch.cuda.current_device())

    def pair():
        lat = symm_mem.empty((2, 4, 16, _LAT), dtype=torch.bfloat16, device=dev)
        lat_h = symm_mem.rendezvous(lat, group.group_name)
        rs = symm_mem.empty((2, 4, 16, 448), dtype=torch.bfloat16, device=dev)
        rs_h = symm_mem.rendezvous(rs, group.group_name)
        lat.view(torch.int32).fill_(-0x80000000)
        rs.view(torch.int32).fill_(-0x80000000)
        mc = int(lat_h.multicast_ptr or 0)
        peers = [rs_h.get_buffer(d, tuple(rs.shape), torch.bfloat16).data_ptr() for d in range(tp)]
        return lat, mc, rs, peers

    # moe_tail (deferred re-arm, buffer = device call counter parity) and moe_block_tail (immediate
    # re-arm, buffer = layer parity) keep separate mailboxes: their protocols must not mix.
    lat, mc, rs, peers = pair()
    lat_f, mc_f, rs_f, peers_f = pair() if _TAIL_MODE == "fused" else (None, 1, None, [1])
    torch.cuda.synchronize()
    if mc == 0 or mc_f == 0 or any(p == 0 for p in peers + peers_f):
        _TAIL["lat"] = None
        return _TAIL
    _TAIL.update(lat=lat, lat_mc=mc, rs=rs, rs_peers=peers, rank=dist.get_rank(group),
                 counter=torch.zeros(1, dtype=torch.int64, device=dev),
                 lat_f=lat_f, lat_f_mc=mc_f, rs_f=rs_f, rs_f_peers=peers_f,
                 fused_ws=torch.zeros(1 << 20, dtype=torch.uint8, device=dev))
    return _TAIL


# ------------------------------------------------------------------------------------------------
# W1-3 moefront (see the module docstring). Buffers: one set per process/device, shared by all layers
# (layer parity alternates; the front kernel re-arms the other parity's buffers at its start):
#   xmoe  bf16 [2 parities][1 replica][4 rows][7168], every 32-bit word 0x80000000 initially (Lamport)
#   epoch int32 [256] hint counters, sc int64 [2*8*2*896] all ones, hg int32 [2*(2*768 + 2*16*192)] zeros
# ------------------------------------------------------------------------------------------------
_FRONT = os.environ.get("K3MOEFRONT", "0") == "1"
_FRONT_MAX_M = max(0, min(2, int(os.environ.get("K3MOEFRONT_MAX_M", "2"))))
_FRONT_FLAGS = int(os.environ.get("K3MOEFRONT_FLAGS", "0"))  # bit 0: shared gate_up rows staged after the router
_FRONT_BUF: dict = {}


def front_buffers(dev) -> dict:
    b = _FRONT_BUF.get(str(dev))
    if b is None:
        xmoe = torch.empty(2, 1, 4, _H, dtype=torch.bfloat16, device=dev)
        xmoe.view(torch.int32).fill_(-0x80000000)
        b = dict(xmoe=xmoe, epoch=torch.zeros(256, dtype=torch.int32, device=dev),
                 sc=torch.full((2 * 8 * 2 * _E,), -1, dtype=torch.int64, device=dev),
                 hg=torch.zeros(2 * (2 * 2 * _SH + 2 * _TOPK * 192), dtype=torch.int32, device=dev),
                 barrier=torch.zeros(8, dtype=torch.int64, device=dev))
        _FRONT_BUF[str(dev)] = b
    return b


def front_enabled() -> bool:
    """Static part of the front-end eligibility (identical on every rank)."""
    return (_FRONT and _STATE["patched"] and 1 <= _FRONT_MAX_M <= _MAX_M and _TAIL_MODE != "fused"
            and not _FRONT_BUF.get("off") and hasattr(torch.ops, "k3mf")
            and hasattr(torch.ops.k3mf, "moe_block_front"))


def front_layer_par(moe) -> int:
    """This MoE layer's parity (0/1) if it runs the front kernel for M <= _FRONT_MAX_M, else -1. Static per
    layer; oproj_patch's post-attention op publishes x_moe for the layer iff this is >= 0."""
    if not front_enabled():
        return -1
    ok = moe.__dict__.get("_k3front")
    if ok is None:
        s = _layer_state(moe)
        cfg = s["runner"].moe_config if s is not None else None
        ok = bool(s is not None and all(cfg.should_defer_moe_finalize(m) for m in range(1, _FRONT_MAX_M + 1)))
        moe._k3front = ok
        if ok:
            _FRONT_BUF["any_on"] = True
        else:
            if _FRONT_BUF.get("any_on"):
                # a mix of front / non-front MoE layers would break the per-parity buffer protocol
                raise RuntimeError(f"K3MOEFRONT: MoE layer {getattr(moe, 'layer_idx', '?')} is not front-eligible "
                                   f"but earlier layers are; run with K3MOEFRONT=0")
            _FRONT_BUF["off"] = True
            print(f"[k3opt] K3MOEFRONT: layer {getattr(moe, 'layer_idx', '?')} is not front-eligible: front end "
                  f"disabled for the whole model", flush=True)
            return -1
    return (int(getattr(moe, "layer_idx", 0) or 0) & 1) if ok else -1


def _front_expected(moe) -> bool:
    """Per-call hand-off from oproj_patch._post_attn_norm (same forward call, just before this MoE forward):
    True iff x_moe was published for this call, so the front kernel MUST run (every publish needs its
    consumer). Consumed (reset) here."""
    exp = moe.__dict__.get("_k3front_expect", False)
    moe._k3front_expect = False
    if exp and moe.__dict__.get("_k3front") is not True:
        raise RuntimeError(f"K3MOEFRONT: layer {getattr(moe, 'layer_idx', '?')} published x_moe but is not "
                           f"front-eligible; run with K3MOEFRONT=0")
    return bool(exp)


def _k3mk_buffers():
    """Collective (all TP ranks, at model construction): mailboxes of k3mk.tail (DESIGN_PLAN W1-2)."""
    if "k3mk" in _TAIL:
        return _TAIL["k3mk"]
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem
    from vllm.distributed import get_tp_group

    group = get_tp_group().device_group
    tp = dist.get_world_size(group)
    if tp != 16:
        _TAIL["k3mk"] = None
        return None
    dev = torch.device("cuda", torch.cuda.current_device())
    lat = symm_mem.empty((3, 16, 8, _LAT), dtype=torch.bfloat16, device=dev)
    lat_h = symm_mem.rendezvous(lat, group.group_name)
    rs = symm_mem.empty((3, 16, 8, 448), dtype=torch.bfloat16, device=dev)
    rs_h = symm_mem.rendezvous(rs, group.group_name)
    lat.view(torch.int32).fill_(-0x80000000)
    rs.view(torch.int32).fill_(-0x80000000)
    mc = int(lat_h.multicast_ptr or 0)
    peers = [rs_h.get_buffer(d, tuple(rs.shape), torch.bfloat16).data_ptr() for d in range(tp)]
    torch.cuda.synchronize()
    if mc == 0 or any(p == 0 for p in peers):
        _TAIL["k3mk"] = None
        return None
    _TAIL["k3mk"] = dict(lat=lat, lat_mc=mc, rs=rs, rs_peers=peers, rank=dist.get_rank(group),
                         epoch=torch.zeros(64, dtype=torch.int64, device=dev),
                         variant=int(os.environ.get("K3MK_TAIL_VARIANT", "14")))
    return _TAIL["k3mk"]


def _grid() -> int:
    return int(os.environ.get("K3MOEBLOCK_GRID", "0"))


def _moe8_enabled(m: int) -> bool:
    # K3MOEBLOCK_MOE8_MIN_M: at M=1 the extra kernel boundary (routing-only block kernel +
    # separate tcgen05 kernel) outweighs the FC speedup, so keep the fused CUDA-core block there.
    return (os.environ.get("K3MOEBLOCK_MOE8", "0") == "1"
            and m >= int(os.environ.get("K3MOEBLOCK_MOE8_MIN_M", "1"))
            and hasattr(torch.ops, "k3moe8") and hasattr(torch.ops.k3moe8, "moe_fused_unfinalized"))


def _moe8_workspace(dev) -> torch.Tensor:
    # One 0xFF-armed workspace for every k3moe8 call in the process (k3opt's M 5..8 path too):
    # the kernel keeps its own call parity in it, so all calls must share it in stream order.
    import k3opt
    ws = k3opt._fused_state.get("moe8_ws")
    if ws is None:
        ws = torch.full((torch.ops.k3moe8.workspace_bytes(),), 0xFF, dtype=torch.uint8, device=dev)
        k3opt._fused_state["moe8_ws"] = ws
    return ws


_MOEBODY_MS = {int(v) for v in os.environ.get("K3MOEBLOCK_MOEBODY_MS", "2").split(",") if v.strip()}


def _moebody_enabled(m: int) -> bool:
    # k3moebody (agents/moe8/body): barrier-free tcgen05 expert body after the routing-only block kernel.
    # One-GPU chain: M=2 -1.3 us vs routing-only + k3moe8; M=1 +1.1 us vs the fused CUDA-core block (not default).
    return (os.environ.get("K3MOEBLOCK_MOEBODY", "0") == "1" and m in _MOEBODY_MS
            and hasattr(torch.ops, "k3moebody") and hasattr(torch.ops.k3moebody, "expert_body"))


def _moebody_workspace(dev) -> torch.Tensor:
    # One 0xFF-armed workspace for every k3moebody call in the process (never cleared again): it holds the
    # per-CTA call counters and parity-buffered staging; the calls must stay ordered by the stream (they are:
    # each call waits for its predecessor chain with griddepcontrol.wait before touching the workspace).
    import k3opt
    ws = k3opt._fused_state.get("moebody_ws")
    if ws is None:
        ws = torch.full((torch.ops.k3moebody.workspace_bytes(),), 0xFF, dtype=torch.uint8, device=dev)
        k3opt._fused_state["moebody_ws"] = ws
    return ws


def _barrier(dev) -> torch.Tensor:
    # Dedicated to moe_block_lamport (its grid can differ from the other fused MoE ops').
    key = (str(dev), _grid())
    b = _STATE["barriers"].get(key)
    if b is None:
        b = torch.zeros(8, dtype=torch.int64, device=dev)
        _STATE["barriers"][key] = b
    return b


def _build_layer_state(moe):
    """Static eligibility + cached weight views for one KimiMoE, or None (use the original)."""
    try:
        import k3opt as k3  # the plugin module (K3OPT_DOWNSHARD objects)
        from vllm.model_executor.layers.fused_moe.experts import trtllm_mxfp4_moe as fi_moe
        from vllm.models.kimi_k3.nvidia.latent_moe_runner import LatentMoERunner
    except Exception as ex:  # noqa: BLE001
        print(f"[k3opt] K3MOEBLOCK: disabled ({ex})", flush=True)
        return None

    def no(reason):
        if not getattr(moe, "_k3mb_reason_printed", False):
            print(f"[k3opt] K3MOEBLOCK: layer {getattr(moe, 'layer_idx', '?')} uses the original path "
                  f"({reason})", flush=True)
            moe._k3mb_reason_printed = True
        return None

    if getattr(moe, "use_mega_moe", False) or not getattr(moe, "use_latent_moe", False):
        return no("not latent / MegaMoE")
    down = moe.routed_expert_down_proj
    if down is None or not getattr(down, "_k3opt_sharded", False):
        return no("K3OPT_DOWNSHARD not active")
    runner = moe.experts
    if not isinstance(runner, LatentMoERunner):
        return no("runner is not LatentMoERunner")
    if not (runner._use_fused_path() and getattr(runner, "enable_k3_latent_moe_tail_fusion", False)
            and runner.moe_config.use_deferred_moe_finalize):
        return no("latent tail fusion / deferred finalize off")
    if getattr(runner, "gate", None) is not None or getattr(moe, "use_sequence_parallel", False):
        return no("runner-held gate / sequence parallel")
    router = getattr(runner, "router", None)
    if router is not None and type(router).__name__ == "ZeroExpertRouter":
        return no("zero-expert router")
    if not (moe.use_grouped_topk and moe.num_expert_group in (None, 0, 1) and moe.topk_group in (None, 0, 1)
            and moe.moe_router_activation_func == "sigmoid"):
        return no("routing config")
    if getattr(moe.experts, "top_k", _TOPK) != _TOPK:
        return no("top_k != 16")
    gate = moe.gate
    gw, gb = gate.weight, getattr(gate, "e_score_correction_bias", None)
    if not (gw.dtype == torch.bfloat16 and tuple(gw.shape) == (_E, _H) and gw.is_contiguous()
            and gb is not None and gb.dtype == torch.float32 and gb.numel() == _E):
        return no("router weight layout")
    sh = moe.shared_experts
    if sh is None or getattr(sh, "shard_sequence_parallel", False) or getattr(sh, "gemm_rs_ar", None) is not None:
        return no("shared expert layout")
    w13s, wds = sh.gate_up_proj.weight, sh.down_proj.weight
    if not (w13s.dtype == torch.bfloat16 and tuple(w13s.shape) == (2 * _SH, _H) and w13s.is_contiguous()
            and wds.dtype == torch.bfloat16 and tuple(wds.shape) == (_H, _SH) and wds.is_contiguous()):
        return no(f"shared weights {tuple(w13s.shape)} {tuple(wds.shape)} {w13s.dtype}")
    act = sh.act_fn
    if type(act).__name__ != "SituAndMul":
        return no("shared activation")
    qm = runner._quant_method
    mk = getattr(qm, "moe_kernel", None)
    impl = getattr(mk, "impl", None) if mk is not None else None
    experts = getattr(impl, "fused_experts", None)
    if not isinstance(experts, fi_moe.TrtLlmMxfp4ExpertsMonolithic):
        return no("routed experts are not TRT-LLM MXFP4 monolithic")
    routed = runner.routed_experts
    try:
        routed._ensure_moe_quant_config_init()
    except Exception:  # noqa: BLE001
        pass
    w1, w2 = routed.w13_weight, routed.w2_weight
    if not (w1.shape[1:] == (512, 1792) and w2.shape[1:] == (_LAT, 128)):
        return no("routed expert layout")
    op, _copy = k3._down_shard_ops()
    st = k3._down_state
    shard = _LAT // st["tp"]
    tail = None
    if _TAIL_MODE in ("fused", "kernel", "k3mk"):
        tb = _k3mk_buffers() if _TAIL_MODE == "k3mk" else _tail_buffers()
        if tb is None:
            tb = {"lat": None}
        top = getattr(runner, "_k3_latent_moe_tail_op", None)
        transform = getattr(runner, "routed_output_transform", None)
        if tb.get("lat") is None or top is None or transform is None or getattr(transform, "norm", None) is None:
            print(f"[k3opt] K3MOEBLOCK_TAIL: layer {getattr(moe, 'layer_idx', '?')} keeps vLLM's tail "
                  f"(no TP16 multicast / tail op)", flush=True)
        elif top.contract.tp_size != 16 or top.rank != tb["rank"]:
            print("[k3opt] K3MOEBLOCK_TAIL: tail op group mismatch, keeping vLLM's tail", flush=True)
        else:
            up = transform.up_proj.weight
            gamma = transform.norm.weight
            if (up.dtype == torch.bfloat16 and tuple(up.shape) == (_H, _LAT) and up.is_contiguous()
                    and gamma.dtype == torch.bfloat16 and gamma.numel() == _LAT and gamma.is_contiguous()):
                upp = top._up_projection
                tail = dict(op=top, w_up=up.narrow(0, top.rank * 448, 448).contiguous(), gamma=gamma,
                            eps=float(top.contract.rms_eps), mb=upp._mailbox, mb_mc=int(upp._mailbox_multicast_ptr),
                            parity=int(getattr(moe, "layer_idx", 0) or 0) & 1,
                            par3=int(getattr(moe, "layer_idx", 0) or 0) % 3)
            else:
                print("[k3opt] K3MOEBLOCK_TAIL: up_proj / norm layout, keeping vLLM's tail", flush=True)
    return dict(
        tail=tail,
        runner=runner,
        gate_w=gw, bias=gb.data.contiguous(),
        sh_w13=w13s, sh_down=wds, sh_beta=float(act.beta),
        sh_lbeta=float(act.linear_beta) if act.linear_beta is not None else -1.0,
        w=(w1.view(torch.uint8), experts.w1_scale.view(torch.uint8), w2.view(torch.uint8),
           experts.w2_scale.view(torch.uint8)),
        betas=(float(experts.gemm1_alpha[0]) if experts.gemm1_alpha is not None else 4.0,
               float(experts.gemm1_beta[0]) if experts.gemm1_beta is not None else 25.0),
        renorm=bool(moe.moe_renormalize),
        scale=float(moe.routed_scaling_factor) if moe.routed_scaling_factor is not None else 1.0,
        down_op=op,
        down_w=down.weight.narrow(0, st["rank"] * shard, shard),
        zero_shard=st["zeros"].narrow(1, st["rank"] * shard, shard),
        ident={},
    )


def _layer_state(moe):
    s = getattr(moe, "_k3mb_state", "unset")
    if s == "unset":
        s = _build_layer_state(moe)
        moe._k3mb_state = s
    return s


def _forward_small(moe, s, hidden_states):
    from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    K = torch.ops.k3moe
    num_tokens, hidden = hidden_states.shape
    x = hidden_states.view(-1, hidden).contiguous()
    m = x.shape[0]
    dev = x.device
    aux = getattr(moe, "_down_proj_stream", None)
    if aux is not None and BreakableCUDAGraphCapture.is_active():
        aux = None
    # 1) latent down-projection shard -> NVLS multicast into every rank's Lamport mailbox (aux).
    if aux is not None:
        ev0, ev1 = moe._down_proj_events
        ev0.record()
        with torch.cuda.stream(aux):
            ev0.wait()
            mailbox = s["down_op"](x, s["down_w"], s["zero_shard"])
            ev1.record()
    else:
        mailbox = s["down_op"](x, s["down_w"], s["zero_shard"])
    # 2) router GEMV (scores) + shared gate_up/SiTU (main stream, overlaps the producer).
    scores = torch.empty(8, m, _E, 2, dtype=torch.float32, device=dev)  # 8 replicas
    h_sh = torch.empty(m, 2 * _SH, dtype=torch.bfloat16, device=dev)  # shared gate_up (bf16)
    none_i = torch.empty(0, dtype=torch.int32, device=dev)
    none_w = torch.empty(0, dtype=torch.bfloat16, device=dev)
    K.route_shared(x, s["gate_w"], s["bias"], s["sh_w13"], scores, none_i, none_w, h_sh, s["sh_beta"],
                   s["sh_lbeta"], s["renorm"], s["scale"])
    if aux is not None:
        ev1.wait()  # also orders the mailbox use after the producer launch (no starvation)
    # 3) top-k + routed FC1/FC2 (unfinalized) + shared down_proj, one persistent kernel.
    ids = torch.empty(m, _TOPK, dtype=torch.int32, device=dev)
    wts = torch.empty(m, _TOPK, dtype=torch.bfloat16, device=dev)
    gemm2 = torch.empty(m * _TOPK, _LAT, dtype=torch.bfloat16, device=dev)
    shared_out = torch.empty(m, _H, dtype=torch.bfloat16, device=dev)
    workspace = torch.empty(m * _TOPK * 192, dtype=torch.float16, device=dev)
    tail = s.get("tail")
    use_body = _moebody_enabled(m)
    use_fused_tail = tail is not None and _TAIL_MODE == "fused" and not _moe8_enabled(m) and not use_body
    # k3mk on the moe8 path needs k3mk.tail's front mode (it finalizes gemm2 itself); it measured neutral to
    # -0.8 us vs vLLM's tail on 16 ranks, so it is opt-in (K3MK_FRONT=1); otherwise such layers keep moe8 +
    # vLLM's tail.
    use_k3mk = tail is not None and _TAIL_MODE == "k3mk" and (
        not (_moe8_enabled(m) or use_body) or os.environ.get("K3MK_FRONT", "0") == "1")
    if use_k3mk:
        KM = torch.ops.k3mk
        kb = _TAIL["k3mk"]
        par = tail["par3"]
        empty = torch.empty(0, dtype=torch.bfloat16, device=dev)
        if use_body or _moe8_enabled(m):  # tcgen05 FC1/FC2 -> k3mk.tail front mode (finalizes gemm2, publishes)
            latent = torch.empty(m, _LAT, dtype=torch.bfloat16, device=dev)
            K.moe_block_lamport(mailbox, scores, *s["w"], workspace, _barrier(dev), gemm2, ids, wts, h_sh,
                                s["sh_down"], shared_out, s["betas"][0], s["betas"][1], s["renorm"],
                                s["scale"], _grid(), s["sh_beta"], s["sh_lbeta"], None, latent)
            if use_body:
                torch.ops.k3moebody.expert_body(latent, ids, None, *s["w"], _moebody_workspace(dev), None, gemm2,
                                                0, s["betas"][0], s["betas"][1])
            else:
                torch.ops.k3moe8.moe_fused_unfinalized(latent, ids, *s["w"], _moe8_workspace(dev),
                                                       _barrier(dev), gemm2, s["betas"][0], s["betas"][1])
            front = (gemm2, wts, shared_out, kb["lat_mc"], kb["rs_peers"])
        else:  # block kernel PUBLISH epilogue (no gemm2_out / shared_out)
            KM.moe_block_lamport(mailbox, scores, *s["w"], workspace, _barrier(dev), empty, ids, wts, h_sh,
                                 s["sh_down"], empty, s["betas"][0], s["betas"][1], s["renorm"], s["scale"],
                                 _grid(), s["sh_beta"], s["sh_lbeta"], None, None, kb["lat"], kb["lat_mc"],
                                 kb["rs_peers"], par, kb["rank"], True)
            front = (None, None, None, 0, None)
        runner = s["runner"]
        try:  # K3OPT_L2PF: same fork point as below (after FC2, before the tail)
            import l2pf_patch
            if l2pf_patch._STATE["patched"]:
                l2pf_patch.launch_after_moe(runner, m)
        except ImportError:
            pass
        KM.tail(kb["lat"], kb["rs"], par, tail["w_up"], tail["gamma"], tail["eps"], tail["mb"], tail["mb_mc"],
                kb["epoch"], kb["rank"], m, None, front[0], front[1], front[2], front[3], front[4], True,
                kb["variant"])
        _KEEPALIVE[:] = [x, gemm2, wts, shared_out, ids, h_sh, scores, workspace]
        result = _tail_result(runner, tail, m)
        _STATE["stats"]["tail_k3mk"] = _STATE["stats"].get("tail_k3mk", 0) + 1
        _STATE["stats"]["fused"] += 1
        return result.view(num_tokens, hidden)
    if use_fused_tail:
        tb = _TAIL
        buf = tail["parity"]
        K.moe_block_tail(mailbox, scores, *s["w"], workspace, _barrier(dev), ids, wts, h_sh, s["sh_down"],
                         tb["lat_f"][buf], tb["lat_f_mc"] + buf * tb["lat_f"][0].numel() * 2, tb["rs_f"][buf],
                         [p + buf * tb["rs_f"][0].numel() * 2 for p in tb["rs_f_peers"]], tail["mb"], tail["mb_mc"],
                         tail["w_up"], tail["gamma"], tb["fused_ws"], tail["eps"], tb["rank"], tb.get("multicast", True),
                         s["betas"][0], s["betas"][1], s["renorm"], s["scale"], _grid(), s["sh_beta"], s["sh_lbeta"])
    elif use_body:
        # Routing-only block kernel (top-k, shared expert, latent hand-off + mailbox re-arm), then the
        # barrier-free tcgen05 expert body (agents/moe8/body) for FC1/FC2 -> unfinalized gemm2.
        latent = torch.empty(m, _LAT, dtype=torch.bfloat16, device=dev)
        K.moe_block_lamport(mailbox, scores, *s["w"], workspace, _barrier(dev), gemm2, ids, wts, h_sh,
                            s["sh_down"], shared_out, s["betas"][0], s["betas"][1], s["renorm"],
                            s["scale"], _grid(), s["sh_beta"], s["sh_lbeta"], None, latent)
        torch.ops.k3moebody.expert_body(latent, ids, None, *s["w"], _moebody_workspace(dev), None, gemm2, 0,
                                        s["betas"][0], s["betas"][1])
    elif _moe8_enabled(m):
        # Routing-only block kernel (top-k, shared expert, latent hand-off + mailbox re-arm),
        # then the tcgen05 MXFP4 MoE (agents/moe8) for FC1/FC2.
        latent = torch.empty(m, _LAT, dtype=torch.bfloat16, device=dev)
        K.moe_block_lamport(mailbox, scores, *s["w"], workspace, _barrier(dev), gemm2, ids, wts, h_sh,
                            s["sh_down"], shared_out, s["betas"][0], s["betas"][1], s["renorm"],
                            s["scale"], _grid(), s["sh_beta"], s["sh_lbeta"], None, latent)
        torch.ops.k3moe8.moe_fused_unfinalized(latent, ids, *s["w"], _moe8_workspace(dev),
                                               _barrier(dev), gemm2, s["betas"][0], s["betas"][1])
    else:
        K.moe_block_lamport(mailbox, scores, *s["w"], workspace, _barrier(dev), gemm2, ids, wts, h_sh,
                            s["sh_down"], shared_out, s["betas"][0], s["betas"][1], s["renorm"],
                            s["scale"], _grid(), s["sh_beta"], s["sh_lbeta"])
    runner = s["runner"]
    if tail is not None and _TAIL_MODE in ("fused", "kernel"):
        # K3OPT_L2PF: same fork point as below (after FC2, before the tail)
        try:
            import l2pf_patch
            if l2pf_patch._STATE["patched"]:
                l2pf_patch.launch_after_moe(runner, m)
        except ImportError:
            pass
        if not use_fused_tail:
            tb = _TAIL
            K.moe_tail(gemm2, wts, shared_out, tail["w_up"], tail["gamma"], tb["lat"], tb["lat_mc"], tb["rs"],
                       tb["rs_peers"], tail["mb"], tail["mb_mc"], tb["counter"], tail["eps"], tb["rank"],
                       tb.get("multicast", True))
        _KEEPALIVE[:] = [x, gemm2, wts, shared_out, ids, h_sh, scores, workspace]
        result = _tail_result(runner, tail, m)
        _STATE["stats"]["tail_" + ("fused" if use_fused_tail else "kernel")] = \
            _STATE["stats"].get("tail_" + ("fused" if use_fused_tail else "kernel"), 0) + 1
        _STATE["stats"]["fused"] += 1
        return result.view(num_tokens, hidden)
    ident = s["ident"].get(m)
    if ident is None:
        ident = torch.arange(m * _TOPK, dtype=torch.int32, device=dev).view(m, _TOPK)
        s["ident"][m] = ident
    fused = UnfinalizedMoEOutput(gemm2_permuted=gemm2, expert_weights=wts, expanded_idx_to_permuted_idx=ident)
    # 4) existing MoE tail (latent reduce + RMSNorm + shared reduce-scatter + up-proj multicast).
    # K3OPT_L2PF hooks LatentMoERunner._forward_impl, which this path bypasses: fork the
    # next layer's L2 prefetch here instead (same point: after FC2, before the tail).
    try:
        import l2pf_patch
        if l2pf_patch._STATE["patched"]:
            l2pf_patch.launch_after_moe(runner, m)
    except ImportError:
        pass
    result = runner._small_batch_tail(fused, shared_out, None)
    result = runner._maybe_add_zero_expert_output(result)
    _STATE["stats"]["fused"] += 1
    return result.view(num_tokens, hidden)


def _forward_front(moe, s, x, m, par, num_tokens, hidden):
    """W1-3: steps 1-3 of _forward_small in ONE kernel (k3mf.moe_block_front). x_moe comes from the
    post-attention op's publish (same values as x: KimiDecoderLayer feeds post-attn-norm's output to the MoE),
    then the configured tail."""
    from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput

    import k3opt as k3

    dev = x.device
    fb = front_buffers(dev)
    op = s["down_op"]
    st = k3._down_state
    ids = torch.empty(m, _TOPK, dtype=torch.int32, device=dev)
    wts = torch.empty(m, _TOPK, dtype=torch.bfloat16, device=dev)
    gemm2 = torch.empty(m * _TOPK, _LAT, dtype=torch.bfloat16, device=dev)
    shared_out = torch.empty(m, _H, dtype=torch.bfloat16, device=dev)
    workspace = torch.empty(m * _TOPK * 192, dtype=torch.float16, device=dev)
    torch.ops.k3mf.moe_block_front(
        fb["xmoe"], fb["epoch"], par, s["gate_w"], s["bias"], s["down_w"], op._mailbox,
        int(op._mailbox_multicast_ptr), st["rank"], s["sh_w13"], fb["sc"], *s["w"], workspace, fb["barrier"], gemm2,
        ids, wts, fb["hg"], s["sh_down"], shared_out, s["betas"][0], s["betas"][1], s["renorm"], s["scale"], 0,
        s["sh_beta"], s["sh_lbeta"], 0, None, None, _FRONT_FLAGS)
    if os.environ.get("K3MOEFRONT_CHECK", "0") == "1" and not torch.cuda.is_current_stream_capturing():
        # debug (eager only): the x_moe the kernel read == the MoE input
        torch.cuda.synchronize()
        xc = x.clone()
        xc.view(torch.int16)[xc.view(torch.int16) == -32768] = 0
        if not torch.equal(fb["xmoe"][par, 0, :m].view(torch.int16), xc.view(torch.int16)):
            raise RuntimeError(f"K3MOEFRONT: x_moe published for layer {getattr(moe, 'layer_idx', '?')} != MoE input")
    _STATE["stats"]["front"] += 1
    runner = s["runner"]
    tail = s.get("tail")
    try:  # K3OPT_L2PF: same fork point as the other paths (after FC2, before the tail)
        import l2pf_patch
        if l2pf_patch._STATE["patched"]:
            l2pf_patch.launch_after_moe(runner, m)
    except ImportError:
        pass
    if tail is not None and _TAIL_MODE == "k3mk" and _TAIL.get("k3mk"):
        # k3mk.tail FRONT mode: finalizes gemm2 / wts / shared_out (griddepcontrol.wait on the front kernel)
        kb = _TAIL["k3mk"]
        torch.ops.k3mk.tail(kb["lat"], kb["rs"], tail["par3"], tail["w_up"], tail["gamma"], tail["eps"], tail["mb"],
                            tail["mb_mc"], kb["epoch"], kb["rank"], m, None, gemm2, wts, shared_out, kb["lat_mc"],
                            kb["rs_peers"], True, kb["variant"])
        _KEEPALIVE[:] = [x, gemm2, wts, shared_out, ids, workspace]
        _STATE["stats"]["tail_k3mk"] = _STATE["stats"].get("tail_k3mk", 0) + 1
        return _tail_result(runner, tail, m).view(num_tokens, hidden)
    if tail is not None and _TAIL_MODE == "kernel":
        tb = _TAIL
        torch.ops.k3moe.moe_tail(gemm2, wts, shared_out, tail["w_up"], tail["gamma"], tb["lat"], tb["lat_mc"],
                                 tb["rs"], tb["rs_peers"], tail["mb"], tail["mb_mc"], tb["counter"], tail["eps"],
                                 tb["rank"], tb.get("multicast", True))
        _KEEPALIVE[:] = [x, gemm2, wts, shared_out, ids, workspace]
        return _tail_result(runner, tail, m).view(num_tokens, hidden)
    ident = s["ident"].get(m)
    if ident is None:
        ident = torch.arange(m * _TOPK, dtype=torch.int32, device=dev).view(m, _TOPK)
        s["ident"][m] = ident
    fused = UnfinalizedMoEOutput(gemm2_permuted=gemm2, expert_weights=wts, expanded_idx_to_permuted_idx=ident)
    result = runner._small_batch_tail(fused, shared_out, None)
    result = runner._maybe_add_zero_expert_output(result)
    return result.view(num_tokens, hidden)


def _tail_result(runner, tail, m):
    """The [m, 7168] MoE output from vLLM's up-proj mailbox (written by our tail kernel)."""
    mb = tail["mb"]
    if getattr(runner, "_k3tail_defer", False):
        try:  # tailattn: its fused attn_res consumer polls the mailbox view directly
            import tailattn_patch

            tailattn_patch._register_mailbox(mb, tail["op"]._lamport_copy)
            tailattn_patch._count("deferred")
            return runner._maybe_reduce_final_output(mb[0, :m], None, output_is_reduced=True)
        except ImportError:
            pass
    out = tail["op"]._lamport_copy(mb, m=m).squeeze(0)
    return runner._maybe_reduce_final_output(out, None, output_is_reduced=True)


def _register_fakes() -> None:
    # The new ops only mutate their arguments; give torch.compile/functionalization a no-op fake.
    for name in ("k3moe::route_shared", "k3moe::moe_block_lamport", "k3moe::moe_block_tail", "k3moe::moe_tail",
                 "k3mk::moe_block_lamport", "k3mk::tail", "k3moebody::expert_body", "k3mf::moe_block_front"):
        try:
            torch.library.register_fake(name)(lambda *args, **kwargs: None)
        except Exception:  # noqa: BLE001  (already registered, or no tracing support)
            pass


def patch_moeblock(load_ext) -> None:
    """Install (idempotent). Call before model construction, after the other k3opt patches."""
    if _STATE["patched"] or not _enabled():
        return
    load_ext()
    if not (hasattr(torch.ops.k3moe, "route_shared") and hasattr(torch.ops.k3moe, "moe_block_lamport")):
        raise RuntimeError("K3MOEBLOCK needs the k3moe build with route_shared / moe_block_lamport "
                           "(agents/moefused/moe_small.cu -> csrc/moe_small.cu)")
    _register_fakes()
    from vllm.models.kimi_k3.nvidia import model as k3_model

    KimiMoE = k3_model.KimiMoE
    orig = KimiMoE.forward
    _STATE["orig_forward"] = orig
    if _TAIL_MODE in ("fused", "kernel", "k3mk"):
        if _TAIL_MODE == "k3mk":
            if not (hasattr(torch.ops, "k3mk") and hasattr(torch.ops.k3mk, "tail")):
                raise RuntimeError("K3MOEBLOCK_TAIL=k3mk needs agents/moefused/tail2.cu built (namespace k3mk)")
        elif not (hasattr(torch.ops.k3moe, "moe_tail") and hasattr(torch.ops.k3moe, "moe_block_tail")):
            raise RuntimeError("K3MOEBLOCK_TAIL needs the k3moe build with moe_tail / moe_block_tail")
        orig_init = KimiMoE.__init__

        def __init__(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            # collective symmetric-memory setup at model init (every rank, same order)
            _k3mk_buffers() if _TAIL_MODE == "k3mk" else _tail_buffers()

        KimiMoE.__init__ = __init__

    def forward(self, hidden_states):
        m = hidden_states.shape[0]
        if _front_expected(self):
            # x_moe was published for this call: the front kernel MUST consume it (protocol invariant).
            num_tokens, hidden = hidden_states.shape
            x = hidden_states.view(-1, hidden).contiguous()
            return _forward_front(self, _layer_state(self), x, x.shape[0],
                                  int(getattr(self, "layer_idx", 0) or 0) & 1, num_tokens, hidden)
        if 1 <= m <= _MAX_M and hidden_states.dim() == 2 and hidden_states.dtype == torch.bfloat16:
            s = _layer_state(self)
            if s is not None and s["runner"].moe_config.should_defer_moe_finalize(m):
                return _forward_small(self, s, hidden_states)
        _STATE["stats"]["fallback"] += 1
        return orig(self, hidden_states)

    KimiMoE.forward = forward
    _STATE["patched"] = True
    if _FRONT:
        if front_enabled():
            print(f"[k3opt] K3MOEFRONT: routing-first MoE front end (k3mf.moe_block_front) for M<={_FRONT_MAX_M} "
                  f"(flags={_FRONT_FLAGS}, tail: {_TAIL_MODE or 'vLLM'}); needs agents/oproj/front/oproj_patch.py",
                  flush=True)
        else:
            print(f"[k3opt] K3MOEFRONT: NOT enabled (needs k3mf.moe_block_front built, 1 <= K3MOEFRONT_MAX_M <= "
                  f"K3MOEBLOCK_MAX_M, K3MOEBLOCK_TAIL != fused)", flush=True)
    print(f"[k3opt] K3MOEBLOCK: fused router/top-k + MoE + shared expert for M<={_MAX_M} "
          f"{'[FC1/FC2 via k3moe8 tcgen05] ' if os.environ.get('K3MOEBLOCK_MOE8', '0') == '1' else ''}"
          f"{'[M in ' + str(sorted(_MOEBODY_MS)) + ': FC1/FC2 via k3moebody] ' if os.environ.get('K3MOEBLOCK_MOEBODY', '0') == '1' else ''}"
          f"{('[tail: ' + _TAIL_MODE + '] ') if _TAIL_MODE else ''}"
          f"(grid={_grid() or 'all SMs'})", flush=True)


def stats() -> dict:
    return dict(_STATE["stats"])
