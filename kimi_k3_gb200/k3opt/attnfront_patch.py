"""vLLM integration of k3sgt.attnres_inproj (agents/l2pf/tcstage.cu): fused Lamport mailbox consume +
pre-attention AttnRes + KDA in_proj (tcgen05 GEMV over staged weights) for decode.

Call ``patch_attnfront(load_ext)`` once per process at plugin time, AFTER ``tailattn_patch.patch_tailattn`` (this
module reuses its mailbox registry and its fallbacks) and after ``kda_fb_patch.patch_kda_fb`` if that is used.
``load_ext()`` must build/load tcstage.cu so that ``torch.ops.k3sgt.attnres_inproj`` exists.
A/B switch: ``K3AF_DISABLE=1`` (no patching).

What changes (``KimiDecoderLayer`` of KDA layers with a full-rank gate, statically eligible only):
  * ``forward``: ``_pre_attn_norm`` + ``self_attn.in_proj_qkvgfab`` are replaced by one opaque custom op
    ``k3af_vllm::attnres_inproj(delta, prefix, blocks, ..., in_proj_weight) -> projected`` and the rest of
    ``KimiK3DeltaAttention.forward`` runs from ``projected`` (split, f_b_proj / fused KDA decode via kda_fb_patch's
    dispatcher when that patch is active, o_proj). ``_post_attn_norm`` and the MLP are unchanged.
  * Inside the op (every decision is on static properties, so capture == replay):
      - delta is a registered tail mailbox view (tailattn_patch deferred the Lamport copy) and 1 <= M <= 8:
        one launch of k3sgt.attnres_inproj -> prefix/blocks updated exactly as lamport_attn_res does (bit-exact),
        projected = AttnRes(x) @ W_in^T (fp32 accumulate, one bf16 rounding; within 1 ulp of CuTe).
      - otherwise: the tailattn consumer (lamport_attn_res / Lamport copy + vLLM attn_res) followed by the layer's
        own in_proj_qkvgfab module call, i.e. exactly today's path.
Statically eligible layer: use_attn_res, no sequence parallel, self_attn is KimiK3DeltaAttention, in_proj weight is
an unquantized bias-free contiguous bf16 [N, 7168] tensor with 1920 <= N <= 3240 (3216 at TP16), no projection-
overlap path (``_projection_overlap_max_tokens <= 0``).
Constraints of the kernel: 120 CTAs in 8-CTA clusters, 512 threads, ~190-196 KB smem, 1 CTA/SM; PDL trigger at
entry, no griddepcontrol.wait (the mailbox poll is the readiness signal, as in lamport_attn_res); prefix/blocks/
weights must be produced >= 2 kernels earlier (true in the model). Do not run k3pf next to it: prefetch the in_proj
weights from a resident kernel if needed (a concurrently launched k3pf grid can hold one 8-cluster's SMs).
"""

from __future__ import annotations

import os
from typing import Optional

import torch

HIDDEN = 7168
MAX_M = int(os.environ.get("K3AF_MAX_M", "4"))  # M = 5..8 loses to lamport_attn_res + GEMM (see RESULTS)
# K3AF_PF: what to do with l2pf's k3pf prefetch (forked right after the previous layer's MoE FC2) for the batches
# this op fuses (M <= MAX_M, eligible layer):
#   keep (default):  k3pf as today. REQUIRED in the current server: the fused grid only becomes resident when the
#                    vLLM tail kernels (allreduce_rmsnorm 512 CTAs x 214 regs, fused_add_multicast 224 CTAs) leave
#                    the SMs, ~2.5 us before landing, so it stages in_proj mostly after landing and needs it L2-hot
#                    (af_skip profile: 12.6 us per stage_kernel vs 9.7 us for lamport_attn_res + skinny GEMM).
#   inkernel:        no k3pf launch; the kernel L2-prefetches f_b_proj + o_proj from 30 of its CTAs after staging
#                    (only sensible once the grid is resident early, i.e. with a persistent-kernel tail);
#   skip:            no k3pf launch, no in-kernel prefetch.
# Batches the op does not fuse (M > MAX_M) always get k3pf as today.
PF_MODE = os.environ.get("K3AF_PF", "keep")
# K3AF_POLL = mode + 10 * backoff (units of 32 ns): 0 = full-slice polling (default), 1 = watch one word per source
# rank first (less L2 pressure on the lines the NVLS stores land in), e.g. 31 = mode 1 with 96 ns backoff.
POLL = int(os.environ.get("K3AF_POLL", "0"))
# K3AF_MLA=1 (default 0, opt-in; kernel tested at N=2880, model path not yet run in a server): also fuse AttnRes +
# fused_qkv_a_g_proj (2112 qkv_a + 768 gate rows) for the q-LoRA MLA layers with an output gate.
MLA_ON = os.environ.get("K3AF_MLA", "0") == "1"
MLA_QKV_ROWS = 1536 + 512 + 64  # q_lora_rank + kv_lora_rank + qk_rope_head_dim
_STATE = {"patched": False, "orig_forward": None, "orig_launch_after_moe": None, "orig_model_init": None, "kfb": None,
          "mla": None, "gate_mul": None,
          "stats": {"fused": 0, "fallback": 0, "k3pf_skipped": 0}}
_PF: dict[int, Optional[torch.Tensor]] = {}  # key -> int64 [n, 2] (ptr, bytes) of f_b_proj / o_proj weights
_NEXT: dict[int, object] = {}  # id(LatentMoERunner) -> weakref of the next decoder layer
_LAYERS: dict[int, object] = {}  # key -> projection module (KDA in_proj_qkvgfab / MLA fused_qkv_a_g_proj); fallback call
_ATTN: dict[int, object] = {}    # key -> self_attn module (in-kernel prefetch table)
_SCHED: dict[int, torch.Tensor] = {}      # key -> int32[4] reader / epoch counters of the fused kernel


def _enabled() -> bool:
    return os.environ.get("K3AF_DISABLE", "0") != "1"


def _sched(key: int, device) -> torch.Tensor:
    t = _SCHED.get(key)
    if t is None:  # first (eager warm-up) call per layer; the counters are self-resetting / monotonic
        t = _SCHED[key] = torch.zeros(4, dtype=torch.int32, device=device)
    return t


def _ta():
    """tailattn_patch (same package in k3opt; plain module in tests)."""
    try:
        from . import tailattn_patch as ta
    except ImportError:
        import tailattn_patch as ta  # type: ignore
    return ta


def _pf_table(key: int) -> Optional[torch.Tensor]:
    """(ptr, bytes) of this layer's f_b_proj / o_proj weights for the in-kernel L2 prefetch. Built at the first
    call outside stream capture (vLLM always runs an eager warm-up first), when the weights are final."""
    if PF_MODE != "inkernel":
        return None
    if key in _PF:
        return _PF[key]
    if torch.cuda.is_current_stream_capturing():
        return None
    attn = _ATTN[key]
    spans = []
    dev = None
    for name in ("f_b_proj", "q_b_proj", "kv_b_proj", "o_proj"):  # the layer's later weights, in use order
        w = getattr(getattr(attn, name, None), "weight", None)
        if isinstance(w, torch.Tensor) and w.is_cuda and w.is_contiguous() and w.numel() > 0:
            spans.append((w.data_ptr(), w.numel() * w.element_size()))
            dev = w.device
    _PF[key] = torch.tensor(spans, dtype=torch.int64).to(dev) if spans else None
    return _PF[key]


def _fused_ok(mailbox, delta, prefix, blocks, w, num_blocks, block_write_idx) -> bool:
    ta = _ta()

    M = delta.shape[0]
    if not (1 <= M <= MAX_M and M <= mailbox.shape[1]):
        return False
    if not ta._fused_eligible(mailbox, delta, prefix, blocks, None, None, None, num_blocks, block_write_idx):
        return False
    return (w.dtype == torch.bfloat16 and w.dim() == 2 and w.shape[1] == HIDDEN and w.is_contiguous()
            and 1920 <= w.shape[0] <= 3240 and w.is_cuda and w.data_ptr() % 16 == 0)


@torch.library.custom_op("k3af_vllm::attnres_inproj", mutates_args=("delta", "prefix", "blocks"))
def _attnres_inproj_op(
    delta: torch.Tensor,
    prefix: torch.Tensor,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor,
    in_proj_weight: torch.Tensor,
    num_blocks: int,
    block_write_idx: int,
    eps: float,
    output_norm_eps: float,
    key: int,
) -> torch.Tensor:
    """projected = in_proj(attn_res(prefix, delta, blocks, ...)); delta may be a pending Lamport mailbox view."""
    ta = _ta()

    entry = ta._MAILBOXES.get(delta.data_ptr())
    if entry is not None and _fused_ok(entry[0], delta, prefix, blocks, in_proj_weight, num_blocks,
                                        block_write_idx):
        mailbox = entry[0]
        M = delta.shape[0]
        y = torch.empty((M, in_proj_weight.shape[0]), dtype=torch.bfloat16, device=delta.device)
        torch.ops.k3sgt.attnres_inproj(
            mailbox, prefix, blocks, norm_weight, qk_weight, output_norm_weight, None, in_proj_weight, y,
            _sched(key, delta.device), num_blocks, block_write_idx, eps, output_norm_eps, None, _pf_table(key), POLL)
        _STATE["stats"]["fused"] += 1
        return y
    # Today's path: tailattn's consumer (fused Lamport + AttnRes, or copy + vLLM attn_res), then the in_proj module.
    out = torch.ops.k3tail_vllm.attn_res(delta, prefix, blocks, norm_weight, qk_weight, output_norm_weight,
                                         num_blocks, block_write_idx, eps, output_norm_eps)
    _STATE["stats"]["fallback"] += 1
    return _LAYERS[key](out)[0]


@_attnres_inproj_op.register_fake
def _(delta, prefix, blocks, norm_weight, qk_weight, output_norm_weight, in_proj_weight, num_blocks,
      block_write_idx, eps, output_norm_eps, key):
    return torch.empty((delta.shape[0], in_proj_weight.shape[0]), dtype=torch.bfloat16, device=delta.device)


def _kda_forward_projected(attn, projected: torch.Tensor, hidden_dtype) -> torch.Tensor:
    """KimiK3DeltaAttention.forward from the in_proj output on (non-overlap path). Traced by torch.compile: no
    imports / exception handling here (the modules are resolved once, at patch time)."""
    from einops import rearrange  # (as in KimiK3DeltaAttention.forward and kda_fb_patch._forward)

    num_tokens = projected.size(0)
    split_sizes = [3 * attn.local_projection_size, attn.local_projection_size, attn.head_dim,
                   attn.local_num_heads]
    if attn.in_proj_padding:
        split_sizes.append(attn.in_proj_padding)
    mixed_qkv, g_proj_states, f_a, beta = projected.split(split_sizes, dim=-1)[:4]
    beta = beta.unsqueeze(0)
    g2 = rearrange(g_proj_states, "... (h d) -> ... h d", d=attn.head_dim)
    core_attn_out = torch.empty((1, num_tokens, attn.local_num_heads, attn.head_dim), dtype=hidden_dtype,
                                device=projected.device)
    kfb = _STATE["kfb"]
    if kfb is not None and kfb._STATE["patched"] and kfb._static_ok(attn):
        kfb._dispatch_fn()(attn, mixed_qkv, f_a, g2, beta, core_attn_out)
    else:
        g1 = rearrange(attn.f_b_proj(f_a)[0], "n (h d) -> 1 n h d", d=attn.head_dim)
        attn._forward(mixed_qkv=mixed_qkv, g1=g1, g2=g2, beta=beta, core_attn_out=core_attn_out)
    core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
    if attn.gemm_rs_ar is not None and attn.gemm_rs_ar.should_run(core_attn_out):
        return attn.gemm_rs_ar(core_attn_out, attn.o_proj.weight)
    return attn.o_proj(core_attn_out)[0]


def _is_mla(attn) -> bool:
    try:
        from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention
    except Exception:  # noqa: BLE001
        return False
    return bool(isinstance(attn, MultiHeadLatentAttention) and attn.q_lora_rank is not None
                and getattr(attn, "use_output_gate", False) and attn.fused_qkv_a_g_proj is not None
                and attn.q_lora_rank + attn.kv_lora_rank + attn.qk_rope_head_dim == MLA_QKV_ROWS)


def _mla_forward_projected(attn, positions: torch.Tensor, projected: torch.Tensor) -> torch.Tensor:
    """MultiHeadLatentAttention.forward from the fused_qkv_a_g_proj output on (mla_patch's fused decode when it is
    active and eligible, else vLLM's q-LoRA attention), with oproj_patch's gate hand-off when installed."""
    M = projected.shape[0]
    qkv_lora, gate = projected.split([MLA_QKV_ROWS, projected.shape[1] - MLA_QKV_ROWS], dim=-1)
    mp = _STATE["mla"]
    if mp is not None and mp._STATE["patched"] and mp._enabled() and 1 <= M <= mp._max_m() and mp._static_ok(attn):
        attn_out = torch.empty((M, gate.shape[1]), dtype=projected.dtype, device=projected.device)
        mp._dispatch_fn()(attn, positions, qkv_lora, gate, attn_out)  # writes the gated output
        gate = None
    else:
        attn_out = attn._apply_q_lora_attention(positions, qkv_lora, qkv_lora)
    if attn.gemm_rs_ar is not None and attn.gemm_rs_ar.should_run(attn_out):
        if gate is not None:
            attn_out = _STATE["gate_mul"](attn_out, gate)
        return attn.gemm_rs_ar(attn_out, attn.o_proj.weight)
    if gate is None:
        return attn.o_proj(attn_out)[0]
    if getattr(attn.o_proj, "_k3oproj", False):
        return attn.o_proj(attn_out, _k3_gate=gate)[0]
    return attn.o_proj(_STATE["gate_mul"](attn_out, gate))[0]


def _kda_fb():
    try:
        from . import kda_fb_patch as kfb
    except ImportError:
        try:
            import kda_fb_patch as kfb  # type: ignore
        except Exception:  # noqa: BLE001
            return None
    return kfb


def _layer_ok(layer) -> bool:
    ok = getattr(layer, "_k3af_ok", None)
    if ok is not None:
        return ok
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention

    ta = _ta()

    attn = getattr(layer, "self_attn", None)
    is_kda = isinstance(attn, KimiK3DeltaAttention)
    is_mla = MLA_ON and _is_mla(attn)
    ok = bool(
        _enabled()
        and getattr(layer, "use_attn_res", False)
        and not getattr(layer, "use_sequence_parallel", False)
        and (is_kda or is_mla)
        and not getattr(layer, "_self_attn_writes_output", True)
        and type(layer)._pre_attn_norm is ta._pre_attn_norm
        and (not is_kda or getattr(attn, "_projection_overlap_max_tokens", 0) <= 0)
    )
    if ok:
        ip = attn.in_proj_qkvgfab if is_kda else attn.fused_qkv_a_g_proj
        w = getattr(ip, "weight", None)
        ok = bool(
            getattr(ip, "bias", None) is None
            and isinstance(getattr(ip, "quant_method", None), UnquantizedLinearMethod)
            and isinstance(w, torch.Tensor) and w.dtype == torch.bfloat16 and w.is_contiguous()
            and w.dim() == 2 and w.shape[1] == HIDDEN and 1920 <= w.shape[0] <= 3240
        )  # device / alignment are checked per call (the op falls back) and by the kernel
    if ok:
        key = id(attn)
        _LAYERS[key] = ip
        _ATTN[key] = attn
        layer._k3af_key = key
        layer._k3af_mla = bool(is_mla)
    layer._k3af_ok = ok
    return ok


def _flag_layers(model) -> int:
    """At model construction: static eligibility of every layer, and the MoE runner -> next layer map (k3pf)."""
    import weakref

    n = 0
    start, end = getattr(model, "start_layer", 0), getattr(model, "end_layer", len(model.layers))
    for idx in range(start, end):
        layer = model.layers[idx]
        layer.__dict__.pop("_k3af_ok", None)
        n += bool(_layer_ok(layer))
        runner = getattr(getattr(layer, "mlp", None), "experts", None)
        if runner is not None and idx + 1 < end:
            _NEXT[id(runner)] = weakref.ref(model.layers[idx + 1])
    return n


def _launch_after_moe(runner, num_tokens: int) -> bool:
    """l2pf_patch.launch_after_moe: no k3pf for a batch whose next layer runs the fused op (see K3AF_PF)."""
    if PF_MODE != "keep" and 1 <= num_tokens <= MAX_M:
        ref = _NEXT.get(id(runner))
        nxt = ref() if ref is not None else None
        if nxt is not None and getattr(nxt, "_k3af_ok", False):
            _STATE["stats"]["k3pf_skipped"] += 1
            return False
    return _STATE["orig_launch_after_moe"](runner, num_tokens)


def _layer_forward(self, positions, hidden_states, residual, prefix_sum=None, **kwargs):
    """KimiDecoderLayer.forward with AttnRes + in_proj fused for eligible KDA layers."""
    if hidden_states is None or residual is None or prefix_sum is None or not getattr(self, "_k3af_ok", False):
        return _STATE["orig_forward"](self, positions, hidden_states, residual, prefix_sum, **kwargs)
    attn = self.self_attn
    projected = torch.ops.k3af_vllm.attnres_inproj(
        hidden_states,
        prefix_sum,
        residual,
        self.self_attention_res_norm.weight,
        self.self_attention_res_proj.weight.squeeze(0),
        self.input_layernorm.weight,
        (attn.fused_qkv_a_g_proj if self._k3af_mla else attn.in_proj_qkvgfab).weight,
        self.prev_valid_blocks,
        self.block_write_idx if self.is_block_write_layer else -1,
        self.self_attention_res_norm.variance_epsilon,
        self.input_layernorm.variance_epsilon,
        self._k3af_key,
    )
    if self._k3af_mla:
        hidden_states = _mla_forward_projected(attn, positions, projected)
    else:
        hidden_states = _kda_forward_projected(attn, projected, prefix_sum.dtype)
    hidden_states, prefix_sum, residual = self._post_attn_norm(hidden_states, residual, prefix_sum)
    hidden_states = self.mlp(hidden_states)
    return hidden_states, prefix_sum, residual


def patch_attnfront(load_ext) -> bool:
    """Install (idempotent). Returns True if patched."""
    if not _enabled():
        print("[k3af] K3AF_DISABLE=1: AttnRes + in_proj fusion off", flush=True)
        return False
    if _STATE["patched"]:
        return True
    ta = _ta()

    if not ta._STATE["patched"]:
        raise RuntimeError("k3af: install tailattn_patch.patch_tailattn first (mailbox registry + fallback)")
    load_ext()
    if not hasattr(torch.ops.k3sgt, "attnres_inproj"):
        raise RuntimeError("k3af: torch.ops.k3sgt.attnres_inproj missing (load_ext must build tcstage.cu)")
    from vllm.models.kimi_k3.nvidia import model as k3_model

    _STATE["kfb"] = _kda_fb()  # None if kda_fb_patch is not shipped; its _STATE says whether it is active
    try:
        import mla_patch as _mp  # noqa: F401
    except ImportError:
        _mp = None
    _STATE["mla"] = _mp
    try:
        from vllm.models.kimi_k3.nvidia.mla import _gate_sigmoid_mul
    except Exception:  # noqa: BLE001
        _gate_sigmoid_mul = None
    _STATE["gate_mul"] = _gate_sigmoid_mul
    Layer, Model = k3_model.KimiDecoderLayer, k3_model.KimiLinearModel
    _STATE["orig_forward"] = Layer.forward
    Layer.forward = _layer_forward
    orig_init = Model.__init__
    _STATE["orig_model_init"] = orig_init

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        n = _flag_layers(self)
        print(f"[k3af] {n} layers fuse AttnRes + in_proj (KDA{' + MLA' if MLA_ON else ''}, M<={MAX_M}, "
              f"K3AF_PF={PF_MODE})", flush=True)

    Model.__init__ = __init__
    try:
        import l2pf_patch as l2pf  # k3opt ships the patches as top-level modules
    except ImportError:
        l2pf = None
    if l2pf is not None and PF_MODE != "keep" and hasattr(l2pf, "launch_after_moe"):
        _STATE["orig_launch_after_moe"] = l2pf.launch_after_moe
        l2pf.launch_after_moe = _launch_after_moe  # both l2pf's own hook and moeblock look it up by module attr
    _STATE["patched"] = True
    print(f"[k3af] fused mailbox consume + AttnRes + KDA in_proj (k3sgt.attnres_inproj, tcgen05) for decode M<={MAX_M}",
          flush=True)
    return True


def stats() -> dict:
    return dict(_STATE["stats"])
