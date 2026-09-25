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
"""

from __future__ import annotations

import os

import torch

_STATE = {"patched": False, "orig_forward": None, "barriers": {}, "stats": {"fused": 0, "fallback": 0}}
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
    if _TAIL_MODE in ("fused", "kernel"):
        tb = _tail_buffers()
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
                tail = dict(op=top, w_up=up.narrow(0, top.rank * 448, 448), gamma=gamma,
                            eps=float(top.contract.rms_eps), mb=upp._mailbox, mb_mc=int(upp._mailbox_multicast_ptr),
                            parity=int(getattr(moe, "layer_idx", 0) or 0) & 1)
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
    use_fused_tail = tail is not None and _TAIL_MODE == "fused" and not _moe8_enabled(m)
    if use_fused_tail:
        tb = _TAIL
        buf = tail["parity"]
        K.moe_block_tail(mailbox, scores, *s["w"], workspace, _barrier(dev), ids, wts, h_sh, s["sh_down"],
                         tb["lat_f"][buf], tb["lat_f_mc"] + buf * tb["lat_f"][0].numel() * 2, tb["rs_f"][buf],
                         [p + buf * tb["rs_f"][0].numel() * 2 for p in tb["rs_f_peers"]], tail["mb"], tail["mb_mc"],
                         tail["w_up"], tail["gamma"], tb["fused_ws"], tail["eps"], tb["rank"], tb.get("multicast", True),
                         s["betas"][0], s["betas"][1], s["renorm"], s["scale"], _grid(), s["sh_beta"], s["sh_lbeta"])
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
    if tail is not None:
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
    for name in ("k3moe::route_shared", "k3moe::moe_block_lamport", "k3moe::moe_block_tail", "k3moe::moe_tail"):
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
    if _TAIL_MODE in ("fused", "kernel"):
        if not (hasattr(torch.ops.k3moe, "moe_tail") and hasattr(torch.ops.k3moe, "moe_block_tail")):
            raise RuntimeError("K3MOEBLOCK_TAIL needs the k3moe build with moe_tail / moe_block_tail")
        orig_init = KimiMoE.__init__

        def __init__(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            _tail_buffers()  # collective symmetric-memory setup at model init (every rank, same order)

        KimiMoE.__init__ = __init__

    def forward(self, hidden_states):
        m = hidden_states.shape[0]
        if 1 <= m <= _MAX_M and hidden_states.dim() == 2 and hidden_states.dtype == torch.bfloat16:
            s = _layer_state(self)
            if s is not None and s["runner"].moe_config.should_defer_moe_finalize(m):
                return _forward_small(self, s, hidden_states)
        _STATE["stats"]["fallback"] += 1
        return orig(self, hidden_states)

    KimiMoE.forward = forward
    _STATE["patched"] = True
    print(f"[k3opt] K3MOEBLOCK: fused router/top-k + MoE + shared expert for M<={_MAX_M} "
          f"{'[FC1/FC2 via k3moe8 tcgen05] ' if os.environ.get('K3MOEBLOCK_MOE8', '0') == '1' else ''}"
          f"{('[tail: ' + _TAIL_MODE + '] ') if _TAIL_MODE else ''}"
          f"(grid={_grid() or 'all SMs'})", flush=True)


def stats() -> dict:
    return dict(_STATE["stats"])
