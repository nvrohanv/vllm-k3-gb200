"""vLLM integration of k3mla (fused Kimi K3 MLA decode: k3mla.q_prep + k3mla.attn_out).

Call ``patch_mla(load_ext)`` once per process at plugin-registration time (before model
construction). ``load_ext()`` must build/load ``k3mla.cu`` so that ``torch.ops.k3mla.q_prep`` and
``torch.ops.k3mla.attn_out`` exist.  A/B switch: ``K3MLA_DISABLE=1`` (no patching at all).
Optional ``K3MLA_MAX_M`` (default 8): largest token count routed to the fused path.

What it changes (vllm/models/kimi_k3/nvidia/mla.py, ``MultiHeadLatentAttention``):

* ``_forward_q_lora`` -- for layers that are statically eligible (q-LoRA front-end with the output
  gate in ``fused_qkv_a_g_proj``, unquantized bf16 projections, 6 local heads, kv_lora 512,
  rope 64 (unrotated / NoPE layers), nope 128, v 128, plain fp8-e4m3 KV cache, TOKENSPEED_MLA
  backend, no DCP) and a forward of M <= K3MLA_MAX_M tokens: the qkv_a and gate GEMMs run exactly
  as before (gate on the aux stream via ``maybe_execute_in_parallel``, joined right after the
  GEMMs), then ONE eager-break function ``_k3mla_attention_impl(self, positions, qkv_lora, gate,
  attn_out)`` writes the *gated* attention output into ``attn_out`` and ``_forward_q_lora`` returns
  ``(attn_out, None)`` so ``forward()`` does not gate again. Other layers / larger M use the
  original ``_forward_q_lora`` unchanged (M is fixed per captured graph, so that choice is static).
* ``_k3mla_attention_impl`` is wrapped by ``@eager_break_during_capture`` (lazily, at the first
  forward, so a VLLM_USE_BREAKABLE_CUDAGRAPH auto-enabled by VllmConfig is honoured): in FULL
  cudagraph mode it is captured
  (the batch is uniform decode, padded to the capture size); in breakable/piecewise capture it is
  an eager segment that re-runs every replay, so the fused-vs-original decision below is taken
  with the current step's metadata. Decision (all must hold, else the original path):
    decode-only batch (num_prefills == 0, num_decode_tokens == num_decodes == num_actual_tokens,
    i.e. one query token per request), 1 <= num_decodes <= K3MLA_MAX_M, no DCP seq-lens,
    int32 block table / seq lens, a dense [pages, page_size, 576] fp8 cache view.
  Fused path: q_prep(qkv_lora[:B]) inserts the new latent rows into the cache at
  ``forward_context.slot_mapping[layer]`` (PAD_SLOT_ID rows skipped) and builds the fp8 query;
  attn_out(...) runs attention over ``attn_metadata.decode.{block_table, seq_lens}`` (seq_len 0
  padded rows give zeros), W_UV and the sigmoid gate, writing attn_out[:B].
  Original path (mixed prefill+decode, spec-decode q_len > 1, ...): fused_q_kv_rmsnorm ->
  q_b_proj -> self._attention (unchanged, incl. prefill) -> attn_out = _gate_sigmoid_mul(...).
* Weights are used in place: ``W_UK_T`` / ``W_UV`` are read through their existing (strided)
  views of kv_b_proj, so no new tensors/attributes are created (works with in-place snapshot
  restores of the parameters). Scales: ``_q_scale_inv``, ``_k_scale_inv`` (fp8 query / cache
  quantization, as fused_mla_decode_q_concat_kv_cache_insert) and softmax_scale =
  scale * _q_scale_float * _k_scale_float, output_scale = _k_scale_float (as TokenspeedMLAImpl).
"""

from __future__ import annotations

import os
import weakref

import torch

H, QL, KVL, ROPE, NOPE, VD = 6, 1536, 512, 64, 128, 128
D = KVL + ROPE

_MLAS: "weakref.WeakSet" = weakref.WeakSet()
_STATE = {
    "patched": False,
    "orig_forward_q_lora": None,
    "fused_layers": set(),
    "reported": False,
    "stats": {"fused": 0, "fallback": 0, "no_metadata": 0},
}


def _enabled() -> bool:
    return os.environ.get("K3MLA_DISABLE", "0") != "1"


def _max_m() -> int:
    return max(1, min(8, int(os.environ.get("K3MLA_MAX_M", "8"))))


# --------------------------------------------------------------------------------------------
# Static (per-module) eligibility, computed once on first use (after weight loading).
# --------------------------------------------------------------------------------------------
def _static_ok(self) -> bool:
    ok = self.__dict__.get("_k3mla_static")
    if ok is not None:
        return ok
    ok = False
    try:
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
        from vllm.platforms import current_platform

        bf = torch.bfloat16
        g = self.fused_qkv_a_g_proj
        uk, uv = self.W_UK_T, self.W_UV
        ok = (
            self.q_lora_rank == QL and self.kv_lora_rank == KVL and self.qk_rope_head_dim == ROPE
            and self.qk_nope_head_dim == NOPE and self.v_head_dim == VD
            and self.num_local_heads == H and self.use_output_gate and g is not None
            and self.rotary_emb is None
            and isinstance(g.quant_method, UnquantizedLinearMethod) and g.weight.dtype == bf
            and isinstance(self.q_b_proj.quant_method, UnquantizedLinearMethod)
            and self.q_b_proj.weight.dtype == bf and self.q_b_proj.weight.is_contiguous()
            and tuple(self.q_b_proj.weight.shape) == (H * (NOPE + ROPE), QL)
            and self.kv_cache_dtype in ("fp8", "fp8_e4m3")
            and current_platform.fp8_dtype() == torch.float8_e4m3fn
            and (getattr(self, "dcp_world_size", 1) or 1) <= 1
            and self.attn_backend.get_name() == "TOKENSPEED_MLA"
            and bool(getattr(self.impl, "supports_quant_query_input", False))
            and tuple(uk.shape) == (H, NOPE, KVL) and uk.dtype == bf and uk.stride(2) == 1
            and uk.stride(1) % 8 == 0 and uk.stride(0) % 8 == 0 and uk.data_ptr() % 16 == 0
            and tuple(uv.shape) == (H, KVL, VD) and uv.dtype == bf and uv.stride(1) == 1
            and uv.stride(2) % 8 == 0 and uv.stride(0) % 8 == 0 and uv.data_ptr() % 16 == 0
            and self.q_a_layernorm.weight.is_contiguous() and self.kv_a_layernorm.weight.is_contiguous()
            and hasattr(self, "_q_scale_inv") and hasattr(self, "_k_scale_inv")
        )
    except AttributeError:
        ok = False
    self._k3mla_static = bool(ok)
    return self._k3mla_static


def _cache_view(self):
    """The latent cache exactly as the TOKENSPEED decode path reads it: [pages, page, 576]."""
    cache = self._attn_read_kv_cache()
    if cache.dim() == 4:  # [pages, 1, page, D] (tokenspeed_mla_decode does the same squeeze)
        cache = cache.squeeze(1)
    if not (cache.dim() == 3 and cache.shape[-1] == D and cache.element_size() == 1
            and cache.stride(2) == 1 and cache.stride(1) == D and cache.shape[1] >= 16
            and cache.stride(0) % 16 == 0 and cache.data_ptr() % 16 == 0):
        return None
    return cache


def _runtime_ok(self, md, n_rows: int):
    """Decode-only, one token per request, B <= K3MLA_MAX_M; returns (B, cache) or None."""
    dec = getattr(md, "decode", None)
    if dec is None or getattr(md, "num_prefills", 1) != 0:
        return None
    B = md.num_decodes
    if not (md.num_decode_tokens == B == md.num_actual_tokens and 1 <= B <= min(_max_m(), n_rows)):
        return None
    if getattr(dec, "dcp_tot_seq_lens", None) is not None:
        return None
    bt, sl = dec.block_table, dec.seq_lens
    if not (bt.dtype == torch.int32 and sl.dtype == torch.int32 and bt.dim() == 2
            and bt.shape[0] >= B and sl.shape[0] >= B and (bt.stride(1) == 1 or bt.shape[1] == 1)):
        return None
    cache = _cache_view(self)
    if cache is None:
        return None
    return B, cache


def _report(layer_name: str) -> None:
    fused = _STATE["fused_layers"]
    if layer_name in fused:
        if not _STATE["reported"]:
            _STATE["reported"] = True
            print(f"[k3opt] K3MLA: {len(fused)} of {len(_MLAS)} MLA layers took the fused path",
                  flush=True)
        return
    fused.add(layer_name)


# --------------------------------------------------------------------------------------------
# The single eager-break function (writes the GATED output into attn_out).
# --------------------------------------------------------------------------------------------
def _k3mla_attention_impl(self, positions: torch.Tensor, qkv_lora: torch.Tensor,
                          gate: torch.Tensor, attn_out: torch.Tensor) -> None:
    from vllm.forward_context import get_forward_context

    fc = get_forward_context()
    md_by_layer = fc.attn_metadata
    if md_by_layer is None:  # profile / dummy run: same result as the original (0 * gate)
        attn_out.zero_()
        _STATE["stats"]["no_metadata"] += 1
        return
    md = md_by_layer[self.layer_name]
    rt = _runtime_ok(self, md, qkv_lora.shape[0])
    if rt is None:
        # Original path: rmsnorm -> q_b_proj -> _attention (prefill and/or decode) -> gate.
        from vllm.models.kimi_k3.nvidia import mla as mla_mod

        tmp = self._apply_q_lora_attention(positions, qkv_lora, qkv_lora)
        attn_out.copy_(mla_mod._gate_sigmoid_mul(tmp, gate))
        _STATE["stats"]["fallback"] += 1
        return
    B, cache = rt
    cache_u8 = cache.view(torch.uint8)
    slot = fc.slot_mapping[self.layer_name]
    mqa_q = torch.empty((B, H, D), dtype=torch.uint8, device=qkv_lora.device)
    torch.ops.k3mla.q_prep(
        qkv_lora[:B], self.q_a_layernorm.weight, self.kv_a_layernorm.weight, self.q_b_proj.weight,
        self.W_UK_T, cache_u8, slot[:B], self._q_scale_inv, self._k_scale_inv,
        float(self.rms_norm_eps), mqa_q, 1)
    torch.ops.k3mla.attn_out(
        mqa_q, cache_u8, md.decode.block_table, md.decode.seq_lens, self.W_UV, gate[:B],
        float(self.scale * self._q_scale_float * self._k_scale_float), float(self._k_scale_float),
        attn_out[:B], 1)
    _STATE["stats"]["fused"] += 1
    _report(self.layer_name)


def _dispatch_fn():
    """_k3mla_attention_impl wrapped by @eager_break_during_capture, created on first use: the
    decorator reads VLLM_USE_BREAKABLE_CUDAGRAPH when applied, and VllmConfig may auto-enable it
    after plugins are loaded, so decorating lazily (first forward) sees the final setting."""
    fn = _STATE.get("dispatch")
    if fn is None:
        from vllm.compilation.breakable_cudagraph import eager_break_during_capture

        fn = _STATE["dispatch"] = eager_break_during_capture(_k3mla_attention_impl)
    return fn


def _forward_q_lora(self, positions: torch.Tensor, hidden_states: torch.Tensor):
    """MultiHeadLatentAttention._forward_q_lora; returns an already-gated output + gate=None."""
    M = hidden_states.shape[0]
    if not (1 <= M <= _max_m() and _static_ok(self)):
        return _STATE["orig_forward_q_lora"](self, positions, hidden_states)
    from vllm.utils.multi_stream_utils import maybe_execute_in_parallel

    qkv_rows = QL + KVL + ROPE
    w = self.fused_qkv_a_g_proj.weight
    if self._gate_events is not None:
        # Same aux-stream overlap as vLLM, but joined right after the two GEMMs.
        start_event, end_event = self._gate_events
        qkv_lora, gate = maybe_execute_in_parallel(
            lambda: self._unquantized_gemm(hidden_states, w[:qkv_rows]),
            lambda: self._unquantized_gemm(hidden_states, w[qkv_rows:]),
            start_event,
            end_event,
            self.aux_stream,
        )
    else:
        qkv_lora, gate = self.fused_qkv_a_g_proj(hidden_states)[0].split([qkv_rows, H * VD], dim=-1)
    attn_out = torch.empty((M, H * VD), dtype=hidden_states.dtype, device=hidden_states.device)
    _dispatch_fn()(self, positions, qkv_lora, gate, attn_out)
    return attn_out, None


def patch_mla(load_ext) -> None:
    """Install the integration (idempotent). Call before model construction in every worker."""
    if _STATE["patched"]:
        return
    if not _enabled():
        print("[k3opt] K3MLA: disabled (K3MLA_DISABLE=1)", flush=True)
        _STATE["patched"] = True
        return
    load_ext()
    if not (hasattr(torch.ops, "k3mla") and hasattr(torch.ops.k3mla, "q_prep")
            and hasattr(torch.ops.k3mla, "attn_out")):
        raise RuntimeError("k3mla ops not loaded (build k3mla.cu)")

    from vllm.models.kimi_k3.nvidia import mla as mla_mod

    Cls = mla_mod.MultiHeadLatentAttention
    _STATE["orig_forward_q_lora"] = Cls._forward_q_lora
    Cls._forward_q_lora = _forward_q_lora

    orig_init = Cls.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        _MLAS.add(self)

    Cls.__init__ = __init__
    _STATE["patched"] = True
    print(f"[k3opt] K3MLA: fused MLA decode (k3mla.q_prep + k3mla.attn_out) for decode-only "
          f"batches with M <= {_max_m()} (fp8 KV, TOKENSPEED_MLA, NoPE, 6 heads/rank)", flush=True)


def stats() -> dict:
    """Python-side call counters (eager/capture-time calls; FULL-graph replays don't run Python)."""
    return dict(_STATE["stats"], fused_layers=len(_STATE["fused_layers"]), mla_layers=len(_MLAS))
