"""vLLM integration of k3tail.lamport_attn_res (fused MoE-tail Lamport consume + next AttnRes).

Call ``patch_tailattn(load_ext)`` once per process at plugin-registration time (before model
construction). ``load_ext()`` must build/load lamport_attn_res.cu so that
``torch.ops.k3tail.lamport_attn_res`` exists.

What it changes (see INTEGRATION.md for details and risks):

* Producer: ``LatentMoERunner._small_batch_tail`` (tier-0 latent-MoE tail, M <= 128) asks
  ``KimiK3LatentMoETailOp.__call__(..., return_mailbox=True)`` to skip the Lamport copy and return
  the ``[M, 7168]`` view of the symmetric mailbox instead -- but only on runners whose output is
  provably consumed by one of the patched AttnRes calls below (flag ``runner._k3tail_defer``,
  computed from the static model structure at construction and whenever the EAGLE aux layers
  change).
* Consumers: ``KimiDecoderLayer._pre_attn_norm`` and the model-level final AttnRes (through the
  module-global ``attn_res`` of vllm.models.kimi_k3.nvidia.model) call the opaque custom op
  ``torch.ops.k3tail_vllm.attn_res``. Inside it, the delta is recognised as a mailbox view by its
  data pointer; then the fused kernel polls it, re-arms it, and runs AttnRes. A delta that is not
  a mailbox view goes to vLLM's attn_res unchanged, and a mailbox view that the fused kernel
  cannot take gets a real Lamport copy first. Every size/pointer branch lives inside the custom
  op (register_fake provided), so the traced code has no data-dependent Python branches.
"""

from __future__ import annotations

import os
import weakref
from typing import Optional

import torch

HIDDEN = 7168
MAX_NB = 8

# data_ptr of a registered symmetric mailbox -> (mailbox [1, max_m, H], LamportCopyKernel)
_MAILBOXES: dict[int, tuple[torch.Tensor, object]] = {}
# Producer inputs of the most recent deferred tail call. Holding them keeps the caching
# allocator from handing their memory to the consumer's `out` while the producer may still be
# reading them (only possible when the producer triggers PDL early; harmless otherwise).
_KEEPALIVE: list = []
_MODELS: "weakref.WeakSet" = weakref.WeakSet()
_STATE = {"patched": False, "orig_attn_res": None, "stats": {"deferred": 0, "fused": 0,
                                                              "copied": 0, "native": 0}}


def _enabled() -> bool:
    return os.environ.get("K3TAIL_DISABLE", "0") != "1"


def _count(key: str) -> None:
    _STATE["stats"][key] += 1


# --------------------------------------------------------------------------------------------
# Consumer: one opaque custom op for "attn_res with a delta" (pre-attn and final AttnRes).
# --------------------------------------------------------------------------------------------
def _fused_eligible(mailbox, delta, prefix, blocks, norm_weight, qk_weight, output_norm_weight,
                    num_blocks, block_write_idx) -> bool:
    bf = torch.bfloat16
    M = delta.shape[0]
    if not (delta.dim() == 2 and delta.shape[1] == HIDDEN and delta.stride(1) == 1
            and delta.stride(0) == HIDDEN and 1 <= M <= mailbox.shape[1]):
        return False
    if not (prefix.dim() == 2 and prefix.shape[0] == M and prefix.shape[1] == HIDDEN
            and prefix.dtype == bf and prefix.stride(1) == 1 and prefix.stride(0) % 8 == 0
            and prefix.data_ptr() % 16 == 0 and prefix.is_cuda):
        return False
    if not (blocks.dim() == 3 and blocks.shape[0] >= M and blocks.shape[2] == HIDDEN
            and blocks.dtype == bf and blocks.stride(2) == 1 and blocks.stride(0) % 8 == 0
            and blocks.stride(1) % 8 == 0 and blocks.data_ptr() % 16 == 0):
        return False
    if not (0 <= num_blocks <= MAX_NB and num_blocks <= blocks.shape[1]):
        return False
    if not (block_write_idx == -1 or num_blocks <= block_write_idx < blocks.shape[1]):
        return False
    for w in (norm_weight, qk_weight, output_norm_weight):
        if w is None:
            continue
        if not (w.dtype == bf and w.numel() == HIDDEN and w.is_contiguous() and w.data_ptr() % 16 == 0):
            return False
    return True


@torch.library.custom_op("k3tail_vllm::attn_res", mutates_args=("delta", "prefix", "blocks"))
def _attn_res_op(
    delta: torch.Tensor,
    prefix: torch.Tensor,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: Optional[torch.Tensor],
    num_blocks: int,
    block_write_idx: int,
    eps: float,
    output_norm_eps: float,
) -> torch.Tensor:
    """attn_res(prefix, delta, blocks, ...) where delta may be a pending Lamport mailbox view."""
    entry = _MAILBOXES.get(delta.data_ptr())
    if entry is not None:
        mailbox, lamport_copy = entry
        if _fused_eligible(mailbox, delta, prefix, blocks, norm_weight, qk_weight,
                           output_norm_weight, num_blocks, block_write_idx):
            out = torch.empty(prefix.shape, dtype=prefix.dtype, device=prefix.device)
            torch.ops.k3tail.lamport_attn_res(
                mailbox, prefix, blocks, norm_weight, qk_weight, output_norm_weight, out, None,
                num_blocks, block_write_idx, eps, output_norm_eps)
            _count("fused")
            return out
        # A pending mailbox the fused kernel cannot take: consume it the original way.
        delta = lamport_copy(mailbox, m=delta.shape[0]).squeeze(0)
        _count("copied")
    else:
        _count("native")
    return _STATE["orig_attn_res"](
        prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight,
        num_blocks=num_blocks, block_write_idx=block_write_idx, eps=eps,
        output_norm_eps=output_norm_eps)


@_attn_res_op.register_fake
def _(delta, prefix, blocks, norm_weight, qk_weight, output_norm_weight, num_blocks,
      block_write_idx, eps, output_norm_eps):
    return torch.empty_like(prefix)


def _attn_res_wrapper(prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight,
                      num_blocks, block_write_idx, eps, output_norm_eps):
    """Drop-in for vllm.models.kimi_k3.nvidia.model.attn_res (module global)."""
    if delta is None:  # static (argument structure), not data-dependent
        return _STATE["orig_attn_res"](
            prefix, None, blocks, norm_weight, qk_weight, output_norm_weight,
            num_blocks=num_blocks, block_write_idx=block_write_idx, eps=eps,
            output_norm_eps=output_norm_eps)
    return torch.ops.k3tail_vllm.attn_res(
        delta, prefix, blocks, norm_weight, qk_weight, output_norm_weight, num_blocks,
        block_write_idx, eps, output_norm_eps)


def _pre_attn_norm(self, hidden_states, residual, prefix_sum):
    """KimiDecoderLayer._pre_attn_norm; hidden_states may be the previous layer's mailbox view."""
    if not self.use_attn_res or hidden_states is None or prefix_sum is None or residual is None:
        return _STATE["orig_pre_attn_norm"](self, hidden_states, residual, prefix_sum)
    out = torch.ops.k3tail_vllm.attn_res(
        hidden_states,
        prefix_sum,
        residual,
        self.self_attention_res_norm.weight,
        self.self_attention_res_proj.weight.squeeze(0),
        self.input_layernorm.weight,
        self.prev_valid_blocks,
        self.block_write_idx if self.is_block_write_layer else -1,
        self.self_attention_res_norm.variance_epsilon,
        self.input_layernorm.variance_epsilon,
    )
    return out, prefix_sum, residual


# --------------------------------------------------------------------------------------------
# Producer: return the mailbox view instead of the Lamport copy (flagged runners only).
# --------------------------------------------------------------------------------------------
def _register_mailbox(mailbox: torch.Tensor, lamport_copy) -> None:
    ptr = mailbox.data_ptr()
    if ptr not in _MAILBOXES:
        assert mailbox.dim() == 3 and mailbox.shape[0] == 1 and mailbox.shape[2] == HIDDEN
        assert mailbox.dtype == torch.bfloat16 and mailbox.is_contiguous()
        _MAILBOXES[ptr] = (mailbox, lamport_copy)


def _tail_call(self, routed_output, shared_output, rms_weight, up_weight, return_mailbox=False):
    """KimiK3LatentMoETailOp.__call__ with an opt-in mailbox-view return."""
    if not return_mailbox:
        return _STATE["orig_tail_call"](self, routed_output, shared_output, rms_weight, up_weight)
    from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput

    self._validate_inputs(routed_output, shared_output, rms_weight, up_weight)
    num_tokens = (
        routed_output.expanded_idx_to_permuted_idx.shape[0]
        if isinstance(routed_output, UnfinalizedMoEOutput)
        else routed_output.shape[0]
    )
    self._up_projection.ensure_compiled(num_tokens)
    latent, shared_shard = self._collective(routed_output, shared_output, rms_weight)
    local_hidden_size = self.contract.hidden_size // self.contract.tp_size
    local_up_weight = up_weight.narrow(0, self.rank * local_hidden_size, local_hidden_size)
    mailbox = self._up_projection(latent, local_up_weight, shared_shard)
    _register_mailbox(mailbox, self._lamport_copy)
    _KEEPALIVE[:] = [routed_output, shared_output, latent, shared_shard]
    _count("deferred")
    # Pending data: only torch.ops.k3tail_vllm.attn_res may read this view.
    return mailbox[0, :num_tokens]


def _small_batch_tail(self, fused_output, shared_output, trunc_size):
    """LatentMoERunner._small_batch_tail (tier 0)."""
    if not (getattr(self, "_k3tail_defer", False) and trunc_size is None
            and not _has_zero_expert_router(self)):
        return _STATE["orig_small_batch_tail"](self, fused_output, shared_output, trunc_size)
    transform = self.routed_output_transform
    result = self._k3_latent_moe_tail_op(
        fused_output, shared_output, transform.norm.weight, transform.up_proj.weight,
        return_mailbox=True)
    return self._maybe_reduce_final_output(result, trunc_size, output_is_reduced=True)


def _has_zero_expert_router(runner) -> bool:
    try:
        from vllm.model_executor.layers.fused_moe.router.zero_expert_router import ZeroExpertRouter
    except Exception:  # noqa: BLE001
        ZeroExpertRouter = None
    router = getattr(runner, "router", None)
    if ZeroExpertRouter is not None:
        return isinstance(router, ZeroExpertRouter)
    return type(router).__name__ == "ZeroExpertRouter"


# --------------------------------------------------------------------------------------------
# Which runners may defer: decided once from the static model structure.
# --------------------------------------------------------------------------------------------
def _compute_defer_flags(model) -> int:
    from vllm.models.kimi_k3.nvidia import model as k3_model
    from vllm.models.kimi_k3.nvidia.latent_moe_runner import LatentMoERunner

    try:
        from vllm.distributed import get_pp_group

        last_rank = get_pp_group().is_last_rank
    except Exception:  # noqa: BLE001  (no PP group, e.g. unit tests)
        last_rank = True
    aux = set(getattr(model, "aux_hidden_state_layers", ()) or ())
    start, end = model.start_layer, model.end_layer
    n = 0
    for idx in range(start, end):
        runner = getattr(getattr(model.layers[idx], "mlp", None), "experts", None)
        if not isinstance(runner, LatentMoERunner):
            continue
        nxt = idx + 1
        ok = _enabled() and bool(getattr(model, "use_attn_res", False)) and nxt not in aux
        if ok and nxt < end:
            nl = model.layers[nxt]
            # The consumer must be our _pre_attn_norm (not replaced by another patch).
            ok = (bool(getattr(nl, "use_attn_res", False))
                  and not getattr(nl, "use_sequence_parallel", False)
                  and type(nl)._pre_attn_norm is _pre_attn_norm)
        elif ok:
            # Last local layer: consumed by the final AttnRes (last PP rank only).
            ok = (last_rank and not getattr(model, "use_sequence_parallel", False)
                  and k3_model.attn_res is _attn_res_wrapper)
        runner._k3tail_defer = bool(ok)
        n += bool(ok)
    model._k3tail_num_deferred = n
    return n


def patch_tailattn(load_ext) -> None:
    """Install the integration (idempotent). Call before model construction in every worker."""
    if _STATE["patched"]:
        return
    load_ext()
    if not hasattr(torch.ops.k3tail, "lamport_attn_res"):
        raise RuntimeError("k3tail.lamport_attn_res not loaded (build lamport_attn_res.cu)")

    import importlib

    from vllm.models.kimi_k3.nvidia import latent_moe_runner as lmr
    from vllm.models.kimi_k3.nvidia import model as k3_model
    from vllm.models.kimi_k3.nvidia.ops import latent_moe_tail as lmt

    # (the ops package re-exports the function under the submodule's name)
    _STATE["orig_attn_res"] = importlib.import_module(
        "vllm.models.kimi_k3.nvidia.ops.attn_res").attn_res
    Layer, Model = k3_model.KimiDecoderLayer, k3_model.KimiLinearModel
    _STATE["orig_pre_attn_norm"] = Layer._pre_attn_norm
    _STATE["orig_tail_call"] = lmt.KimiK3LatentMoETailOp.__call__
    _STATE["orig_small_batch_tail"] = lmr.LatentMoERunner._small_batch_tail

    # Consumers.
    Layer._pre_attn_norm = _pre_attn_norm
    k3_model.attn_res = _attn_res_wrapper  # final AttnRes (+ other delta calls, unchanged math)
    # Producer.
    lmt.KimiK3LatentMoETailOp.__call__ = _tail_call
    lmr.LatentMoERunner._small_batch_tail = _small_batch_tail

    orig_init = Model.__init__
    orig_set_aux = Model._set_aux_hidden_state_layers

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        _MODELS.add(self)
        n = _compute_defer_flags(self)
        print(f"[k3opt] K3TAIL: {n} MoE tails hand their Lamport mailbox to the next AttnRes",
              flush=True)

    def _set_aux_hidden_state_layers(self, layers):
        orig_set_aux(self, layers)
        _compute_defer_flags(self)

    Model.__init__ = __init__
    Model._set_aux_hidden_state_layers = _set_aux_hidden_state_layers
    _STATE["patched"] = True
    print("[k3opt] K3TAIL: fused Lamport consume + pre-attention AttnRes (k3tail.lamport_attn_res)",
          flush=True)


def stats() -> dict:
    """Python-side call counters (eager/capture-time calls only; graph replays don't run Python)."""
    return dict(_STATE["stats"])
