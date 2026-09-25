"""vLLM integration of k3oproj: o_proj GEMV + TP all-reduce + post-attention AttnRes (Kimi K3 decode).

Call ``patch_oproj(load_ext)`` once per process at plugin-registration time, before model
construction; ``load_ext()`` must build/load oproj_ar.cu so that ``torch.ops.k3oproj`` exists.

What changes (per KimiDecoderLayer with AttnRes, no sequence parallelism, a KimiK3DeltaAttention or
MultiHeadLatentAttention self_attn, and a bf16 [7168, 768] o_proj; i.e. all 93 K3 layers at TP16):

* ``self_attn.o_proj.forward`` (instance attribute): for decode batches 1 <= M <= MAX_M it does NOT
  run the GEMM + all-reduce. It records ``layer._k3oproj_pending = (x, gate)`` and returns the
  o_proj INPUT x ([M, 768]) as a marker. Anything other than our _post_attn_norm that consumed the
  marker would fail loudly on the 768 vs 7168 shape. Otherwise it runs the original forward.
* ``MultiHeadLatentAttention.forward`` (K3OPROJ_MLA_GATE=1, default): hands the sigmoid output gate
  to o_proj instead of launching the gate-multiply kernel; the producer applies x * sigmoid(gate).
* ``KimiDecoderLayer._post_attn_norm``: if hidden_states is this layer's pending marker, calls the
  opaque custom op ``torch.ops.k3oproj_vllm.post_attn`` (GEMV -> NVLS Lamport all-reduce -> AttnRes,
  one or two kernels); else the original. For block-write layers (layer_idx % 12 == 0) the new
  prefix_sum (= the all-reduced o_proj output, which the original returns as hidden_states) is a
  fresh buffer that the kernel writes.
* ``KimiDecoderLayer.__init__``: decides eligibility and allocates, once per process, the symmetric
  Lamport mailbox [2, tp, MAX_M, 7168] bf16 (symm_mem.empty + rendezvous on the TP group, NVLS
  multicast required, 7.3 MB at TP16), filled with the 0x80000000 sentinel. Collective and
  identical on all TP ranks (all ranks construct the same layers in the same order).

Mailbox reuse safety (buffer = layer_idx % 2):
  1. Local: a consumer reads a buffer only after the previous call on this GPU completed (the fused
     kernel polls after its own griddepcontrol.wait; in the split path the producer triggers the
     consumer only after its wait), so the previous consumer of that buffer has re-armed it.
  2. Cross-rank: rank s writes buffer b again (layer L+2) only after its own layer-L+1 call
     completed, which needed rank r's layer-L+1 data, which r published only after its layer-L call
     (the previous user of b) completed, i.e. after r re-armed b. So a fast rank never overwrites a
     slot that a slow rank has not consumed.
  3. Same buffer back to back (layer 92 -> next step's layer 0; MTP layers): separated by TP
     collectives (MoE-tail all-reduce, logits gather) that a rank joins only after its previous
     call completed.

Env (read at call time unless noted):
  K3OPROJ_DISABLE=1        patch stays installed, every call takes the original path (A/B switch)
  K3OPROJ_MAX_M=16         largest batch that takes the k3oproj path (<= 16; read at patch time)
  K3OPROJ_FUSED_MIN_M=2    single fused kernel for FUSED_MIN_M <= M <= FUSED_MAX_M, else produce +
  K3OPROJ_FUSED_MAX_M=8    consume (capped by k3oproj.fused_max_m(), the co-residency limit)
  K3OPROJ_MLA_FUSED_MIN_M=5  the same threshold for MLA layers (2026-09-25: with k3mla.attn_out
                           triggering its dependents early for B <= 4, produce + consume is 1-2 us
                           faster than the fused kernel at M = 2..4 in the MLA layer chain)
  K3OPROJ_MLA_GATE=1       fold MLA's output gate into the producer (read at patch time)
"""

from __future__ import annotations

import os
import weakref
from typing import Optional

import torch

HIDDEN, K_ATTN, NBUF, MAX_NB = 7168, 768, 2, 8
SENTINEL = -0x80000000  # int32 view of the empty marker (bf16 pair 0x8000, 0x0000)

_STATE = {
    "patched": False,
    "max_m": 16,
    "mailbox": None,   # [NBUF, tp, max_m, 7168] bf16 symmetric (or local in tests)
    "mc": 0,           # multicast pointer of the mailbox
    "mode": 0,         # 0: multimem.st (production); 1: local stores (single-GPU tests, tp == 1)
    "rank": 0,
    "tp": 1,
    "init_failed": False,
    "orig_post_attn_norm": None,
    "orig_mla_forward": None,
    "orig_attn_res": None,
    "all_reduce": None,
    "stats": {"fused": 0, "split": 0, "fallback": 0, "deferred": 0},
}


def _disabled() -> bool:
    return os.environ.get("K3OPROJ_DISABLE", "0") == "1"


def _count(key: str) -> None:
    _STATE["stats"][key] = _STATE["stats"].get(key, 0) + 1


def stats() -> dict:
    """Python-side call counters (eager/capture-time calls only; graph replays don't run Python)."""
    return dict(_STATE["stats"])


# --------------------------------------------------------------------------------------------
# Symmetric mailbox (collective, once per process)
# --------------------------------------------------------------------------------------------
def _ensure_mailbox() -> bool:
    if _STATE["mailbox"] is not None:
        return True
    if _STATE["init_failed"]:
        return False
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem
    from vllm.distributed import get_tp_group

    tpg = get_tp_group()
    tp, rank = tpg.world_size, tpg.rank_in_group
    ok = 1 < tp <= 16
    mailbox, mc = None, 0
    if ok:
        try:
            mailbox = symm_mem.empty((NBUF, tp, _STATE["max_m"], HIDDEN), dtype=torch.bfloat16,
                                     device=torch.device("cuda", torch.cuda.current_device()))
            handle = symm_mem.rendezvous(mailbox, tpg.device_group.group_name)
            mc = int(handle.multicast_ptr or 0)
        except Exception as e:  # noqa: BLE001
            print(f"[k3opt] K3OPROJ: symmetric mailbox setup failed: {e}", flush=True)
            mailbox, mc = None, 0
    # All ranks must agree (a rank without multicast would never publish).
    flag = torch.tensor([1 if (ok and mailbox is not None and mc != 0) else 0], dtype=torch.int32, device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=tpg.device_group)
    if int(flag.item()) == 0:
        _STATE["init_failed"] = True
        print("[k3opt] K3OPROJ: NVLS multicast mailbox unavailable -> original o_proj/all-reduce path",
              flush=True)
        return False
    mailbox.view(torch.int32).fill_(SENTINEL)
    torch.cuda.synchronize()
    tpg.barrier()
    _STATE.update(mailbox=mailbox, mc=mc, mode=0, rank=rank, tp=tp)
    return True


def install_local_mailbox_for_tests(max_m: int = 16) -> torch.Tensor:
    """Single-GPU tests (tp == 1): a plain local mailbox and local stores instead of multicast."""
    mailbox = torch.empty((NBUF, 1, max_m, HIDDEN), dtype=torch.bfloat16, device="cuda")
    mailbox.view(torch.int32).fill_(SENTINEL)
    _STATE.update(mailbox=mailbox, mc=0, mode=1, rank=0, tp=1, max_m=max_m, init_failed=False)
    return mailbox


# --------------------------------------------------------------------------------------------
# The opaque custom op: every size / eligibility branch lives inside it.
# --------------------------------------------------------------------------------------------
def _k3_paths(M: int, mla: bool = False) -> Optional[str]:
    if mla:  # MLA layers: k3mla.attn_out triggers its dependents early for B <= 4 -> produce+consume wins
        fmin = int(os.environ.get("K3OPROJ_MLA_FUSED_MIN_M", "5"))
    else:
        fmin = int(os.environ.get("K3OPROJ_FUSED_MIN_M", "2"))
    fmax = min(int(os.environ.get("K3OPROJ_FUSED_MAX_M", "8")), int(torch.ops.k3oproj.fused_max_m()))
    return "fused" if fmin <= M <= fmax else "split"


def _kernel_ok(x, gate, weight, prefix, blocks, nw, qw, ow, num_blocks) -> bool:
    mb = _STATE["mailbox"]
    if mb is None or _disabled():
        return False
    M = x.shape[0]
    bf = torch.bfloat16
    if not (1 <= M <= mb.shape[2] and x.dim() == 2 and x.shape[1] == K_ATTN and x.dtype == bf
            and x.stride(1) == 1 and x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0):
        return False
    if gate is not None and not (gate.shape == x.shape and gate.dtype == bf and gate.stride(1) == 1
                                 and gate.stride(0) % 8 == 0 and gate.data_ptr() % 16 == 0):
        return False
    if not (weight.shape == (HIDDEN, K_ATTN) and weight.dtype == bf and weight.is_contiguous()
            and weight.data_ptr() % 16 == 0):
        return False
    if not (prefix.dim() == 2 and prefix.shape == (M, HIDDEN) and prefix.dtype == bf and prefix.stride(1) == 1
            and prefix.stride(0) % 8 == 0 and prefix.data_ptr() % 16 == 0):
        return False
    if not (blocks.dim() == 3 and blocks.shape[0] >= M and blocks.shape[2] == HIDDEN and blocks.dtype == bf
            and blocks.stride(2) == 1 and blocks.stride(0) % 8 == 0 and blocks.stride(1) % 8 == 0
            and blocks.data_ptr() % 16 == 0 and 0 <= num_blocks <= min(MAX_NB, blocks.shape[1])):
        return False
    for w in (nw, qw, ow):
        if not (w.dtype == bf and w.numel() == HIDDEN and w.is_contiguous() and w.data_ptr() % 16 == 0):
            return False
    return True


@torch.library.custom_op("k3oproj_vllm::post_attn", mutates_args=("prefix",))
def _post_attn_op(
    x: torch.Tensor,
    gate: Optional[torch.Tensor],
    weight: torch.Tensor,
    prefix: torch.Tensor,
    has_delta: bool,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor,
    num_blocks: int,
    buf: int,
    eps: float,
    output_norm_eps: float,
    mla: bool = False,
    xmoe_par: int = -1,
) -> torch.Tensor:
    """out = AttnRes(prefix (+)= all_reduce(o_proj(x [* sigmoid(gate)])), blocks, ...) (post-attention).
    mla: the layer is an MLA layer (only selects the fused / produce+consume dispatch threshold).
    xmoe_par (W1-3 moefront): >= 0 -> this layer's MoE runs k3mf.moe_block_front for small M; then (and
    only then, M <= moeblock_patch._FRONT_MAX_M) the consumer also publishes x_moe for it (parity xmoe_par)."""
    M = x.shape[0]
    out = torch.empty((M, HIDDEN), dtype=torch.bfloat16, device=x.device)
    pub = ()
    if xmoe_par >= 0:
        import moeblock_patch

        if M <= moeblock_patch._FRONT_MAX_M:
            fb = moeblock_patch.front_buffers(x.device)
            pub = (fb["xmoe"], fb["epoch"], xmoe_par)
    if _kernel_ok(x, gate, weight, prefix, blocks, norm_weight, qk_weight, output_norm_weight, num_blocks):
        mb, mc, mode, rank = _STATE["mailbox"], _STATE["mc"], _STATE["mode"], _STATE["rank"]
        ops = torch.ops.k3oprojf if pub else torch.ops.k3oproj
        if _k3_paths(M, mla) == "fused":
            ops.fused(x, weight, gate, mb, mc, buf, rank, mode, prefix, has_delta, blocks, norm_weight, qk_weight,
                      output_norm_weight, out, num_blocks, eps, output_norm_eps, None, *pub)
            _count("fused")
        else:
            torch.ops.k3oproj.produce(x, weight, gate, mb, mc, buf, rank, mode, None, 44, True, None, None)
            ops.consume(mb, buf, prefix, has_delta, blocks, norm_weight, qk_weight, output_norm_weight, out, None,
                        num_blocks, -1, eps, output_norm_eps, *pub)
            _count("split")
        if pub:
            _count("xmoe_publish")
        return out
    if pub:  # the front kernel of this layer waits for x_moe: never skip the publish
        raise RuntimeError("K3MOEFRONT: post-attention fallback path on a front layer (x_moe not published)")
    # Fallback: the original math (gate multiply, o_proj GEMM, TP all-reduce, vLLM attn_res).
    _count("fallback")
    from vllm.models.kimi_k3.nvidia.low_latency_gemm import try_low_latency_gemm

    if gate is not None:
        from vllm.models.kimi_k3.nvidia.mla import _gate_sigmoid_mul

        x = _gate_sigmoid_mul(x, gate)
    y = try_low_latency_gemm(x, weight)
    if y is None:
        y = torch.nn.functional.linear(x, weight)
    y = _STATE["all_reduce"](y)
    if has_delta:
        res = _STATE["orig_attn_res"](prefix, y, blocks, norm_weight, qk_weight, output_norm_weight,
                                      num_blocks=num_blocks, block_write_idx=-1, eps=eps,
                                      output_norm_eps=output_norm_eps)
    else:
        prefix.copy_(y)
        res = _STATE["orig_attn_res"](prefix, None, blocks, norm_weight, qk_weight, output_norm_weight,
                                      num_blocks=num_blocks, block_write_idx=-1, eps=eps,
                                      output_norm_eps=output_norm_eps)
    out.copy_(res)
    return out


@_post_attn_op.register_fake
def _(x, gate, weight, prefix, has_delta, blocks, norm_weight, qk_weight, output_norm_weight, num_blocks, buf,
      eps, output_norm_eps, mla=False, xmoe_par=-1):
    return x.new_empty((x.shape[0], HIDDEN))


# --------------------------------------------------------------------------------------------
# Producer side: o_proj defers, MLA hands over its output gate.
# --------------------------------------------------------------------------------------------
def _defer_ok(x: torch.Tensor) -> bool:
    return (not _disabled() and _STATE["mailbox"] is not None and x.dim() == 2 and x.shape[1] == K_ATTN
            and 1 <= x.shape[0] <= _STATE["max_m"] and x.dtype == torch.bfloat16 and x.is_cuda)


def _wrap_o_proj(layer, o_proj) -> None:
    orig_forward = o_proj.forward  # bound method of RowParallelLinear
    layer_ref = weakref.ref(layer)
    return_bias = getattr(o_proj, "return_bias", True)

    def forward(input_, _k3_gate=None):
        lyr = layer_ref()
        if lyr is not None and _defer_ok(input_):
            lyr._k3oproj_pending = (input_, _k3_gate)
            _count("deferred")
            return (input_, None) if return_bias else input_  # marker: the [M, 768] o_proj input
        if _k3_gate is not None:
            from vllm.models.kimi_k3.nvidia.mla import _gate_sigmoid_mul

            input_ = _gate_sigmoid_mul(input_, _k3_gate)
        return orig_forward(input_)

    o_proj.forward = forward
    o_proj._k3oproj = True


def _mla_forward(self, positions, hidden_states):
    """MultiHeadLatentAttention.forward, passing the output gate to our o_proj instead of multiplying."""
    if not getattr(self.o_proj, "_k3oproj", False):
        return _STATE["orig_mla_forward"](self, positions, hidden_states)
    if self.q_lora_rank is None:
        attn_out, gate = self._forward_full_rank_q(positions, hidden_states)
    else:
        attn_out, gate = self._forward_q_lora(positions, hidden_states)
    if self.gemm_rs_ar is not None and self.gemm_rs_ar.should_run(attn_out):
        if gate is not None:
            from vllm.models.kimi_k3.nvidia.mla import _gate_sigmoid_mul

            attn_out = _gate_sigmoid_mul(attn_out, gate)
        return self.gemm_rs_ar(attn_out, self.o_proj.weight)
    if gate is None:
        return self.o_proj(attn_out)[0]
    return self.o_proj(attn_out, _k3_gate=gate)[0]


def install_on_layer(layer) -> bool:
    """Eligibility + o_proj wrapper for one KimiDecoderLayer (called after its __init__)."""
    from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention
    from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention

    attn = getattr(layer, "self_attn", None)
    o_proj = getattr(attn, "o_proj", None)
    w = getattr(o_proj, "weight", None)
    ok = (bool(getattr(layer, "use_attn_res", False)) and not getattr(layer, "use_sequence_parallel", False)
          and isinstance(attn, (KimiK3DeltaAttention, MultiHeadLatentAttention)) and w is not None
          and tuple(w.shape) == (HIDDEN, K_ATTN) and w.dtype == torch.bfloat16
          and getattr(o_proj, "reduce_results", True) and getattr(o_proj, "bias", None) is None)
    layer._k3oproj = False
    if not ok or not _ensure_mailbox():
        return False
    _wrap_o_proj(layer, o_proj)
    layer._k3oproj = True
    layer._k3oproj_mla = isinstance(attn, MultiHeadLatentAttention)
    return True


# --------------------------------------------------------------------------------------------
# Consumer side: KimiDecoderLayer._post_attn_norm
# --------------------------------------------------------------------------------------------
def _post_attn_norm(self, hidden_states, residual, prefix_sum):
    pend = self.__dict__.get("_k3oproj_pending")
    moe = getattr(self, "block_sparse_moe", None)
    if pend is None or pend[0] is not hidden_states:
        if moe is not None:
            moe._k3front_expect = False  # W1-3: no x_moe published for this call
        return _STATE["orig_post_attn_norm"](self, hidden_states, residual, prefix_sum)
    self._k3oproj_pending = None
    x, gate = pend
    xmoe_par = _xmoe_par(self)
    if moe is not None:
        # W1-3 per-call hand-off: the MoE forward of THIS call runs the front kernel iff x_moe is published now
        # (same rule as _post_attn_op: xmoe_par >= 0 and M <= K3MOEFRONT_MAX_M; the op raises if it cannot).
        pub = False
        if xmoe_par >= 0:
            import moeblock_patch

            pub = x.shape[0] <= moeblock_patch._FRONT_MAX_M
        moe._k3front_expect = pub
    if self.is_block_write_layer:
        # The original makes the all-reduced o_proj output the new prefix_sum (no delta).
        prefix_buf, has_delta = torch.empty((x.shape[0], HIDDEN), dtype=torch.bfloat16, device=x.device), False
    else:
        prefix_buf, has_delta = prefix_sum, True
    out = torch.ops.k3oproj_vllm.post_attn(
        x,
        gate,
        self.self_attn.o_proj.weight,
        prefix_buf,
        has_delta,
        residual,
        self.mlp_res_norm.weight,
        self.mlp_res_proj.weight.squeeze(0),
        self.post_attention_layernorm.weight,
        self.prev_valid_blocks + int(self.is_block_write_layer),
        self.layer_idx % NBUF,
        self.mlp_res_norm.variance_epsilon,
        self.post_attention_layernorm.variance_epsilon,
        bool(getattr(self, "_k3oproj_mla", False)),
        xmoe_par,
    )
    return out, prefix_buf, residual


def _xmoe_par(layer) -> int:
    """W1-3: this layer's parity if its MoE runs the front kernel (moeblock_patch.front_layer_par: K3MOEFRONT=1
    and the layer is front-eligible), else -1. Static per layer, decided the first time the layer runs (the
    same point where moeblock_patch builds its per-layer state). _post_attn_norm hands the per-call decision
    to the MoE forward (moe._k3front_expect), which runs the front kernel exactly when x_moe was published.
    Deploy with agents/oproj/front/moeblock_patch_merged.py as k3opt/moeblock_patch.py."""
    p = layer.__dict__.get("_k3xmoe_par")
    if p is None:
        p = -1
        moe = getattr(layer, "block_sparse_moe", None)
        if moe is not None and getattr(layer, "mlp", None) is moe:
            try:
                import moeblock_patch

                if getattr(moeblock_patch, "front_enabled", lambda: False)():
                    p = moeblock_patch.front_layer_par(moe)
            except ImportError:
                p = -1
        layer._k3xmoe_par = p
    return p


def patch_oproj(load_ext) -> None:
    """Install the integration (idempotent). Call before model construction in every worker."""
    if _STATE["patched"]:
        return
    if os.environ.get("K3OPT_ARRES", "0") == "1":
        raise RuntimeError("K3OPROJ replaces K3OPT_ARRES (both rewrite _post_attn_norm); enable only one")
    load_ext()
    if not hasattr(torch.ops.k3oproj, "fused"):
        raise RuntimeError("k3oproj ops not loaded (build oproj_ar.cu)")

    import importlib

    from vllm.models.kimi_k3.nvidia import mla as k3_mla
    from vllm.models.kimi_k3.nvidia import model as k3_model

    _STATE["max_m"] = max(1, min(16, int(os.environ.get("K3OPROJ_MAX_M", "16"))))
    _STATE["orig_attn_res"] = importlib.import_module("vllm.models.kimi_k3.nvidia.ops.attn_res").attn_res
    if _STATE["all_reduce"] is None:
        from vllm.distributed import tensor_model_parallel_all_reduce

        _STATE["all_reduce"] = tensor_model_parallel_all_reduce
    Layer = k3_model.KimiDecoderLayer
    _STATE["orig_post_attn_norm"] = Layer._post_attn_norm
    Layer._post_attn_norm = _post_attn_norm
    if os.environ.get("K3OPROJ_MLA_GATE", "1") == "1":
        _STATE["orig_mla_forward"] = k3_mla.MultiHeadLatentAttention.forward
        k3_mla.MultiHeadLatentAttention.forward = _mla_forward

    orig_init = Layer.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        install_on_layer(self)

    Layer.__init__ = __init__
    _STATE["patched"] = True
    print(f"[k3opt] K3OPROJ: o_proj GEMV + NVLS all-reduce + post-attention AttnRes fused "
          f"(M <= {_STATE['max_m']}; k3oproj.fused / produce+consume)", flush=True)
