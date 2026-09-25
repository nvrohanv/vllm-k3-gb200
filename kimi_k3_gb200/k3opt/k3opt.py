"""vLLM plugin with Kimi K3 decode optimizations, each behind an env flag.

K3OPT_KDA6=1   Use the fused KDA decode kernel (conv + recurrent update + gated
               RMSNorm in one launch) at 6 heads per rank, i.e. TP16. vLLM's
               kernel only instantiates 12/24/48/96 heads and falls back to
               three Triton kernels plus a copy at TP16.
K3OPT_DOWNSHARD=1
               Shard the replicated latent down-projection (7168->3584) across
               TP ranks for decode batches (M<=8): each rank computes its 224
               latent outputs and multicasts them into every rank's Lamport
               mailbox (vLLM's MoE-tail multicast skinny GEMM, reused), then a
               Lamport copy collects the full latent. Reads 3.2 MB instead of
               51 MB per rank per layer.
K3OPT_MOEFUSED=1
               With K3OPT_ROUTE, run decode batches of <= K3OPT_MOEFUSED_MAX_M
               (default 2) tokens through k3moe.moe_fused: one persistent kernel
               doing FC1 + SiTU-GLU + FC2 + finalize on the BF16 latent, reading
               TRT-LLM's MXFP4 layout in place and skipping its 192->256 padding.
               Replaces MXFP8 quantize + routing + two GEMM launches.
K3OPT_ARRES=1  Fuse the attention o_proj all-reduce into the post-attention
               AttnRes (k3ar.ar_attn_res): o_proj emits its TP partial, which is
               multicast over NVLS into a Lamport mailbox, reduced in rank order
               and fed straight into AttnRes, for decode batches <= 16 tokens.
               Requires VLLM_KIMI_K3_GEMM_AR=0 (o_proj must always emit partials).
K3OPT_TAILATTN=1
               MoE-tail Lamport mailbox consumed directly by the next layer's
               pre-attention AttnRes (k3tail.lamport_attn_res; see
               agents/tailattn/INTEGRATION.md), removing the Lamport copy kernel.
K3OPT_ROUTE=1  Compute MoE top-k in the router branch (which already runs in
               parallel with the latent down-projection on the aux stream) and
               call FlashInfer's pre-routed TRT-LLM MXFP4 MoE, so the routing
               kernel leaves the MoE critical path. Top-k (ids, fp32 weights)
               travels to the expert call packed in one int32 tensor.
"""

import os

import torch

_TOPK = 16


def _flag(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


def _patch_route_precompute():
    from vllm.model_executor.layers.fused_moe.experts import trtllm_mxfp4_moe as fi_moe
    from vllm.models.kimi_k3.nvidia import model as k3_model

    KimiMoE = k3_model.KimiMoE

    def _maybe_overlap_router_and_down_proj(self, hidden_states):
        def _router(hidden_states):
            router_logits, _ = self.gate(hidden_states)
            topk_weights, topk_ids = k3_model.fused_grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=getattr(self.experts, "top_k", None)
                or getattr(getattr(self.experts, "moe_config", None), "experts_per_token", _TOPK),
                renormalize=self.moe_renormalize,
                e_score_correction_bias=self.gate.e_score_correction_bias.data,
                num_expert_group=self.num_expert_group,
                topk_group=self.topk_group,
                scoring_func=self.moe_router_activation_func,
                routed_scaling_factor=self.routed_scaling_factor,
            )
            if self.use_mega_moe:
                return topk_weights, topk_ids
            packed = torch.cat(
                [topk_ids.to(torch.int32), topk_weights.to(torch.float32).view(torch.int32)], dim=-1
            )
            return packed, None

        down_proj = self.routed_expert_down_proj
        if down_proj is None:
            router_output, topk_ids = _router(hidden_states)
            return hidden_states, router_output, topk_ids
        num_tokens = hidden_states.shape[0]
        (router_output, topk_ids), (routed_hidden_states, _) = k3_model.maybe_execute_in_parallel(
            lambda: _router(hidden_states),
            lambda: down_proj(hidden_states),
            self._down_proj_events[0],
            self._down_proj_events[1],
            self._down_proj_stream
            if num_tokens <= k3_model._ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD
            else None,
        )
        return routed_hidden_states, router_output, topk_ids

    KimiMoE._maybe_overlap_router_and_down_proj = _maybe_overlap_router_and_down_proj

    Mono = fi_moe.TrtLlmMxfp4ExpertsMonolithic
    orig_apply = Mono.apply

    def apply(self, hidden_states, w1, w2, router_logits, activation, global_num_experts,
              expert_map, a1q_scale, apply_router_weight_on_input, num_expert_group=None,
              e_score_correction_bias=None, routed_scaling_factor=None, topk_group=None):
        if router_logits.dtype != torch.int32:
            return orig_apply(self, hidden_states, w1, w2, router_logits, activation,
                              global_num_experts, expert_map, a1q_scale,
                              apply_router_weight_on_input, num_expert_group,
                              e_score_correction_bias, routed_scaling_factor, topk_group)
        from flashinfer import trtllm_fp4_block_scale_routed_moe

        topk = router_logits.shape[-1] // 2
        topk_ids = router_logits[:, :topk].contiguous()
        # FlashInfer's own routing emits BF16 expert weights, and the K3 MoE-tail
        # (deferred finalize) requires them in BF16.
        topk_weights = router_logits[:, topk:].contiguous().view(torch.float32).to(torch.bfloat16)
        x_scale = a1q_scale.view(torch.float8_e4m3fn) if a1q_scale is not None else None
        num_tokens = hidden_states.shape[0]
        defer = self.moe_config.should_defer_moe_finalize(num_tokens)
        finalized_output = None
        if not defer:
            finalized_output = torch.empty(*hidden_states.shape[:-1], self.hidden_dim_unpadded,
                                           dtype=torch.bfloat16, device=hidden_states.device)
        out = trtllm_fp4_block_scale_routed_moe(
            topk_ids=(topk_ids, topk_weights),
            routing_bias=None,
            hidden_states=hidden_states,
            hidden_states_scale=x_scale,
            gemm1_weights=w1,
            gemm1_weights_scale=self.w1_scale,
            gemm1_bias=self.w1_bias,
            gemm1_alpha=self.gemm1_alpha,
            gemm1_beta=self.gemm1_beta,
            gemm1_clamp_limit=self.gemm1_clamp_limit,
            gemm2_weights=w2,
            gemm2_weights_scale=self.w2_scale,
            gemm2_bias=self.w2_bias,
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            num_experts=global_num_experts,
            top_k=topk,
            n_group=None,
            topk_group=None,
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.ep_rank * self.local_num_experts,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=None,
            routing_method_type=fi_moe.RoutingMethodType.Renormalize,
            do_finalize=not defer,
            enable_pdl=True,
            activation_type=self._flashinfer_activation_type(activation),
            output=finalized_output,
            tune_max_num_tokens=fi_moe.fi_moe_largest_bucket(self.moe_config),
        )
        return fi_moe.convert_flashinfer_moe_output(
            out, do_finalize=not defer, num_tokens=num_tokens, top_k=topk,
            finalized_output=finalized_output,
        )

    Mono.apply = apply
    print("[k3opt] K3OPT_ROUTE: top-k precomputed in router branch; pre-routed TRT-LLM MoE", flush=True)


def _load_ext():
    import sys
    sys.path.insert(0, os.environ.get("K3OPT_SRC", "/tmp/k3opt/csrc") + "/..")
    from build import build  # noqa: E402  (shipped next to the CUDA sources)

    build()


def _kda_op():
    # K3OPT_KDASPLIT=1: value-dim split cluster kernel (agents/kda), bit-identical results.
    if _flag("K3OPT_KDASPLIT"):
        return torch.ops.k3kdas.fused_kda_decode
    return torch.ops.k3kda.fused_kda_decode


def _patch_kda6():
    import vllm._custom_ops as vops
    from vllm.models.kimi_k3.nvidia import kda as k3_kda

    _load_ext()
    orig_supported = k3_kda.is_fused_kda_decode_supported

    def is_fused_kda_decode_supported(num_heads, *args, **kwargs):
        return orig_supported(12 if num_heads == 6 else num_heads, *args, **kwargs)

    k3_kda.is_fused_kda_decode_supported = is_fused_kda_decode_supported

    def fused_kda_decode(x, weight, bias, conv_state, raw_g, raw_beta, A_log, dt_bias,
                         state_indices, state, out=None, lower_bound=None, output_gate=None,
                         norm_weight=None, norm_eps=1e-5):
        # Mirrors vllm._custom_ops.fused_kda_decode, dispatching to the 6-head build.
        if out is None:
            out = torch.empty(1, x.shape[0], raw_g.shape[2], raw_g.shape[3],
                              dtype=x.dtype, device=x.device)
        _kda_op()(x, weight, bias, conv_state, raw_g, raw_beta, A_log,
                                         dt_bias, state_indices, state, out, lower_bound,
                                         output_gate, norm_weight, norm_eps)
        return out

    vops.fused_kda_decode = fused_kda_decode
    print("[k3opt] K3OPT_KDA6: native fused KDA decode enabled for 6 heads/rank", flush=True)


_DOWN_MAX_M = 8
_down_state = {}


def _down_shard_ops():
    """Build (once per process) the multicast down-projection and its Lamport copy."""
    if "op" in _down_state:
        return _down_state["op"], _down_state["copy"]
    from vllm.distributed import get_tp_group
    from vllm.models.kimi_k3.nvidia.ops.cute_dsl.latent_moe_tail import (
        AdaptiveUpProjectionKernel,
        LamportCopyKernel,
    )

    group = get_tp_group().device_group
    rank = torch.distributed.get_rank(group)
    tp = torch.distributed.get_world_size(group)
    op = AdaptiveUpProjectionKernel(
        group=group, rank=rank, tp_size=tp, latent_dim=7168, hidden_dim=3584,
        max_m=_DOWN_MAX_M, skinny_max_m=_DOWN_MAX_M, mma_tiler_mn=(64, 32),
        cluster_shape_mn=(1, 1), b_prime_stages=2,
    )
    for m in range(1, _DOWN_MAX_M + 1):
        op.compile_skinny(m)
    copy = LamportCopyKernel(hidden_dim=3584, max_m=_DOWN_MAX_M, ctas=32, threads=128)
    zeros = torch.zeros(_DOWN_MAX_M, 3584, dtype=torch.bfloat16, device="cuda")
    _down_state.update(op=op, copy=copy, rank=rank, tp=tp, zeros=zeros)
    return op, copy


@torch.library.custom_op("k3opt::down_shard", mutates_args=())
def _down_shard(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # The size branch lives inside this opaque op: vLLM traces the model once
    # for all batch sizes, so a Python branch outside would be baked in.
    m = x.shape[0]
    if m > _DOWN_MAX_M:
        return torch.nn.functional.linear(x, weight)
    op, copy = _down_shard_ops()
    shard = 3584 // _down_state["tp"]
    local_w = weight.narrow(0, _down_state["rank"] * shard, shard)
    # The multicast kernel sizes its shared-add operand to the mailbox (max_m rows).
    zero_shard = _down_state["zeros"].narrow(1, _down_state["rank"] * shard, shard)
    mailbox = op(x.contiguous(), local_w, zero_shard)
    return copy(mailbox, m=m).squeeze(0)


@_down_shard.register_fake
def _(x, weight):
    return x.new_empty(x.shape[0], weight.shape[0])


def _patch_down_shard():
    from vllm.models.kimi_k3.nvidia import model as k3_model

    KimiMoE = k3_model.KimiMoE
    orig_init = KimiMoE.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        down = self.routed_expert_down_proj
        if down is None or getattr(down, "_k3opt_sharded", False):
            return
        _down_shard_ops()  # collective symmetric-memory setup at model init

        def forward(x):
            return torch.ops.k3opt.down_shard(x, down.weight), None

        down.forward = forward
        down._k3opt_sharded = True

    KimiMoE.__init__ = __init__
    print("[k3opt] K3OPT_DOWNSHARD: latent down-projection sharded + multicast for M<=8", flush=True)


_fused_state = {}


def _patch_moe_fused():
    from vllm.model_executor.layers.fused_moe import modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.experts import trtllm_mxfp4_moe as fi_moe
    from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput

    _load_ext()
    max_m = int(os.environ.get("K3OPT_MOEFUSED_MAX_M", "2"))
    use_moe8 = _flag("K3OPT_MOE8")
    seen = set()

    def _count_moefused(m):
        if m not in seen:
            seen.add(m)
            print(f"[k3opt] K3OPT_MOEFUSED: first unfinalized fused MoE call at M={m}"
                  f"{' (k3moe8 tensor-core kernel)' if use_moe8 else ''}", flush=True)
    Impl = mk.FusedMoEKernelMonolithicImpl
    orig_apply = Impl.apply

    def apply(self, hidden_states, w1, w2, router_logits, activation, global_num_experts,
              expert_map, apply_router_weight_on_input, num_expert_group=None,
              e_score_correction_bias=None, routed_scaling_factor=None, topk_group=None):
        experts = self.fused_experts
        m = hidden_states.shape[0]
        stash = _fused_state.pop("route", None)
        if stash is not None and stash[0].shape[0] == m and router_logits.dtype != torch.int32:
            routed = stash
        elif router_logits.dtype == torch.int32:
            topk = router_logits.shape[-1] // 2
            # Explicit dense [m, topk] buffers: at m=1 .contiguous() would keep the
            # packed row stride, which the MoE-tail CuTe kernels reject.
            ids = torch.empty((m, topk), dtype=torch.int32, device=hidden_states.device)
            ids.copy_(router_logits[:, :topk])
            wts = torch.empty((m, topk), dtype=torch.bfloat16, device=hidden_states.device)
            wts.copy_(router_logits[:, topk:].view(torch.float32))
            routed = (ids, wts)
        else:
            routed = None
        if (routed is None or not 1 <= m <= max_m
                or not isinstance(experts, fi_moe.TrtLlmMxfp4ExpertsMonolithic)
                or w1.shape[1:] != (512, 1792) or w2.shape[1:] != (3584, 128)):
            return orig_apply(self, hidden_states, w1, w2, router_logits, activation,
                              global_num_experts, expert_map, apply_router_weight_on_input,
                              num_expert_group, e_score_correction_bias,
                              routed_scaling_factor, topk_group)
        if "barrier" not in _fused_state:
            _fused_state["barrier"] = torch.zeros(8, dtype=torch.int64, device=hidden_states.device)
        # SiTU betas live in per-expert GPU tensors; read them once (eager warmup),
        # never during CUDA-graph capture.
        betas = _fused_state.get(id(experts))
        if betas is None:
            betas = (float(experts.gemm1_alpha[0]) if experts.gemm1_alpha is not None else 4.0,
                     float(experts.gemm1_beta[0]) if experts.gemm1_beta is not None else 25.0)
            _fused_state[id(experts)] = betas
        topk_ids, topk_weights_bf16 = routed
        topk = topk_ids.shape[-1]
        dev = hidden_states.device
        workspace = torch.empty(m * topk * 192, dtype=torch.float16, device=dev)
        w = (w1.view(torch.uint8), experts.w1_scale.view(torch.uint8), w2.view(torch.uint8),
             experts.w2_scale.view(torch.uint8))
        if not experts.moe_config.should_defer_moe_finalize(m):
            out = torch.empty(m, 3584, dtype=torch.bfloat16, device=dev)
            torch.ops.k3moe.moe_fused(hidden_states.contiguous(), topk_ids, topk_weights_bf16.float(), *w,
                                      workspace, _fused_state["barrier"], out, betas[0], betas[1])
            return out
        # Deferred finalize (the K3 MoE-tail does the top-k reduction): hand back the
        # unweighted per-(token, expert) rows with an identity permute map.
        ident = _fused_state.get(("ident", m, topk))
        if ident is None:
            ident = torch.arange(m * topk, dtype=torch.int32, device=dev).view(m, topk)
            _fused_state[("ident", m, topk)] = ident
        gemm2_out = torch.empty(m * topk, 3584, dtype=torch.bfloat16, device=dev)
        if use_moe8:
            # K3OPT_MOE8: tcgen05 block-scaled MXFP4 MoE (agents/moe8). Its workspace is armed
            # with 0xFF once and re-armed by the kernel itself; shared by all layers (in-order).
            ws8 = _fused_state.get("moe8_ws")
            if ws8 is None:
                ws8 = torch.full((torch.ops.k3moe8.workspace_bytes(),), 0xFF, dtype=torch.uint8, device=dev)
                _fused_state["moe8_ws"] = ws8
            torch.ops.k3moe8.moe_fused_unfinalized(hidden_states.contiguous(), topk_ids, *w, ws8,
                                                   _fused_state["barrier"], gemm2_out, betas[0], betas[1])
        else:
            torch.ops.k3moe.moe_fused_unfinalized(hidden_states.contiguous(), topk_ids, *w, workspace,
                                                  _fused_state["barrier"], gemm2_out, betas[0], betas[1])
        _count_moefused(m)
        return UnfinalizedMoEOutput(gemm2_permuted=gemm2_out,
                                    expert_weights=topk_weights_bf16,
                                    expanded_idx_to_permuted_idx=ident)

    Impl.apply = apply

    if not _flag("K3OPT_ROUTE"):
        # Top-k for small batches only, computed next to the gate GEMM while the
        # latent down-projection runs on the aux stream, and handed to apply()
        # through _fused_state (same layer, same Python call order under capture).
        from vllm.models.kimi_k3.nvidia import model as k3_model

        KimiMoE = k3_model.KimiMoE
        orig_overlap = KimiMoE._maybe_overlap_router_and_down_proj

        def _maybe_overlap_router_and_down_proj(self, hidden_states):
            if self.use_mega_moe or not 1 <= hidden_states.shape[0] <= max_m:
                return orig_overlap(self, hidden_states)
            gate = self.gate

            def gate_and_topk(x):
                logits = gate.__class__.forward(gate, x)
                topk_weights, topk_ids = k3_model.fused_grouped_topk(
                    hidden_states=x, gating_output=logits[0],
                    topk=getattr(self.experts, "top_k", None) or _TOPK,
                    renormalize=self.moe_renormalize,
                    e_score_correction_bias=gate.e_score_correction_bias.data,
                    num_expert_group=self.num_expert_group, topk_group=self.topk_group,
                    scoring_func=self.moe_router_activation_func,
                    routed_scaling_factor=self.routed_scaling_factor)
                ids = torch.empty(topk_ids.shape, dtype=torch.int32, device=x.device)
                wts = torch.empty(topk_weights.shape, dtype=torch.bfloat16, device=x.device)
                ids.copy_(topk_ids)
                wts.copy_(topk_weights)
                _fused_state["route"] = (ids, wts)
                return logits

            self.gate.forward = gate_and_topk
            try:
                return orig_overlap(self, hidden_states)
            finally:
                del self.gate.forward

        KimiMoE._maybe_overlap_router_and_down_proj = _maybe_overlap_router_and_down_proj
    print(f"[k3opt] K3OPT_MOEFUSED: fused MXFP4 MoE kernel for M<={max_m}"
          f"{'' if _flag('K3OPT_ROUTE') else ' (small-M routing in the router branch)'}", flush=True)


_ar_state = {}


def _ar_mailbox():
    if "mailbox" in _ar_state:
        return _ar_state
    import torch.distributed._symmetric_memory as symm_mem
    from vllm.distributed import get_tp_group

    group = get_tp_group().device_group
    tp = torch.distributed.get_world_size(group)
    mailbox = symm_mem.empty((2, tp, 16, 7168), dtype=torch.bfloat16, device="cuda")
    handle = symm_mem.rendezvous(mailbox, group.group_name)
    mailbox.view(torch.int16).fill_(-32768)  # bf16 -0.0 Lamport sentinels
    torch.cuda.synchronize()
    _ar_state.update(mailbox=mailbox, mc=int(handle.multicast_ptr), tp=tp,
                     rank=torch.distributed.get_rank(group))
    return _ar_state


@torch.library.custom_op("k3opt::ar_attn_res", mutates_args=("prefix",))
def _ar_attn_res(partial: torch.Tensor, prefix: torch.Tensor, has_delta: bool, parity: int,
                 blocks: torch.Tensor, norm_w: torch.Tensor, qk_w: torch.Tensor,
                 out_w: torch.Tensor, num_blocks: int, eps: float, out_eps: float) -> torch.Tensor:
    out = torch.empty_like(partial)
    if partial.shape[0] <= 16:
        st = _ar_mailbox()
        torch.ops.k3ar.ar_attn_res(partial, st["mailbox"], st["mc"], parity, st["rank"], st["tp"], prefix,
                                   has_delta, blocks, norm_w, qk_w, out_w, out, num_blocks, -1,
                                   eps, out_eps)
        return out
    from vllm.distributed import tensor_model_parallel_all_reduce
    from vllm.models.kimi_k3.nvidia.ops.attn_res import attn_res

    reduced = tensor_model_parallel_all_reduce(partial)
    if not has_delta:
        prefix.copy_(reduced)
    return attn_res(prefix, reduced if has_delta else None, blocks, norm_w, qk_w, out_w,
                    num_blocks=num_blocks, block_write_idx=-1, eps=eps, output_norm_eps=out_eps)


@_ar_attn_res.register_fake
def _(partial, prefix, has_delta, parity, blocks, norm_w, qk_w, out_w, num_blocks, eps, out_eps):
    return torch.empty_like(partial)


def _patch_ar_attn_res():
    import vllm.envs as envs
    from vllm.models.kimi_k3.nvidia import model as k3_model

    if envs.VLLM_KIMI_K3_GEMM_AR:
        raise RuntimeError("K3OPT_ARRES needs VLLM_KIMI_K3_GEMM_AR=0")
    _load_ext()
    Layer = k3_model.KimiDecoderLayer
    orig_init = Layer.__init__
    orig_post = Layer._post_attn_norm

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self._k3opt_arres = bool(self.use_attn_res and not self.use_sequence_parallel)
        if self._k3opt_arres:
            self.self_attn.o_proj.reduce_results = False
            _ar_mailbox()  # collective symmetric-memory setup at model init

    def _post_attn_norm(self, hidden_states, residual, prefix_sum):
        if not getattr(self, "_k3opt_arres", False):
            return orig_post(self, hidden_states, residual, prefix_sum)
        if self.is_block_write_layer:
            prefix_buf, has_delta = torch.empty_like(hidden_states), False
        else:
            prefix_buf, has_delta = prefix_sum, True
        out = torch.ops.k3opt.ar_attn_res(
            hidden_states, prefix_buf, has_delta, self.layer_idx % 2, residual, self.mlp_res_norm.weight,
            self.mlp_res_proj.weight.squeeze(0), self.post_attention_layernorm.weight,
            self.prev_valid_blocks + self.is_block_write_layer,
            self.mlp_res_norm.variance_epsilon, self.post_attention_layernorm.variance_epsilon)
        return out, prefix_buf, residual

    Layer.__init__ = __init__
    Layer._post_attn_norm = _post_attn_norm
    print("[k3opt] K3OPT_ARRES: o_proj all-reduce fused into post-attention AttnRes (M<=16)", flush=True)


def register():
    if _flag("K3OPT_DOWNSHARD"):
        _patch_down_shard()
    if _flag("K3OPT_KDA6"):
        _patch_kda6()
    if _flag("K3OPT_ROUTE"):
        _patch_route_precompute()
    if _flag("K3OPT_MOEFUSED"):
        _patch_moe_fused()
    if _flag("K3OPT_ARRES"):
        _patch_ar_attn_res()
    if _flag("K3OPT_TAILATTN"):
        from tailattn_patch import patch_tailattn
        patch_tailattn(_load_ext)
    if _flag("K3OPT_KDA6") and _flag("K3OPT_KDAFB"):
        from kda_fb_patch import patch_kda_fb
        patch_kda_fb(_load_ext)
    if _flag("K3OPT_MLA"):
        from mla_patch import patch_mla
        patch_mla(_load_ext)
    if _flag("K3OPT_MOEBLOCK"):
        from moeblock_patch import patch_moeblock
        patch_moeblock(_load_ext)
    if _flag("K3OPT_OPROJ"):
        from oproj_patch import patch_oproj
        patch_oproj(_load_ext)
    if _flag("K3OPT_L2PF"):
        from l2pf_patch import patch_l2pf
        patch_l2pf(_load_ext)
    if _flag("K3OPT_PLANS"):
        from plans_patch import patch_plans
        patch_plans()
    if _flag("K3OPT_GEMV"):
        from gemv_patch import patch_gemv
        patch_gemv(_load_ext)
