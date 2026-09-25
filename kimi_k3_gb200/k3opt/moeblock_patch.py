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


def _grid() -> int:
    return int(os.environ.get("K3MOEBLOCK_GRID", "0"))


def _moe8_enabled() -> bool:
    return os.environ.get("K3MOEBLOCK_MOE8", "0") == "1" and hasattr(torch.ops, "k3moe8") \
        and hasattr(torch.ops.k3moe8, "moe_fused_unfinalized")


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
    return dict(
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
    if _moe8_enabled():
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
    ident = s["ident"].get(m)
    if ident is None:
        ident = torch.arange(m * _TOPK, dtype=torch.int32, device=dev).view(m, _TOPK)
        s["ident"][m] = ident
    fused = UnfinalizedMoEOutput(gemm2_permuted=gemm2, expert_weights=wts, expanded_idx_to_permuted_idx=ident)
    # 4) existing MoE tail (latent reduce + RMSNorm + shared reduce-scatter + up-proj multicast).
    runner = s["runner"]
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


def _register_fakes() -> None:
    # The new ops only mutate their arguments; give torch.compile/functionalization a no-op fake.
    for name in ("k3moe::route_shared", "k3moe::moe_block_lamport"):
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
          f"(grid={_grid() or 'all SMs'})", flush=True)


def stats() -> dict:
    return dict(_STATE["stats"])
