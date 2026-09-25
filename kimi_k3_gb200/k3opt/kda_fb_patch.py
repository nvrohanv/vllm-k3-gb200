"""vLLM integration of the fused KDA gate up-projection + decode kernel
(torch.ops.k3kdas.fused_kda_decode_fb, agents/kda/kda_split.cu).

Call ``patch_kda_fb(load_ext)`` once per process at plugin time (before model construction), together
with / after k3opt's ``_patch_kda6`` (K3OPT_KDA6=1): that patch is what makes the native fused decode
backend eligible at 6 heads/rank, and this patch only takes over layers that selected it.
``load_ext()`` must build/load a ``kda_split.cu`` that defines ``fused_kda_decode_fb``.
A/B switch: ``K3KDAFB_DISABLE=1`` (no patching).  ``K3KDAFB_MAX_M`` (default 16) caps the fused M.

What changes (``KimiK3DeltaAttention``, statically eligible layers only):
  * ``forward``: in_proj_qkvgfab and the split are unchanged, but f_b_proj is no longer run there;
    the [M, 128] f_a slice is handed to one eager-break function instead of g1.
  * that function (``_k3kdafb_impl``, wrapped by ``@eager_break_during_capture`` on first use, like
    the original ``_forward``) reads the forward context exactly as the original ``_forward`` does:
      - decode-only batch (no prefill, no spec decode) with M <= K3KDAFB_MAX_M: one launch of
        fused_kda_decode_fb(f_a, f_b_proj.weight, ...) -- f_b_proj + conv + gates + delta rule +
        gated RMSNorm, bit-identical to f_b_proj(cuBLAS) -> fused_kda_decode;
      - anything else (prefill / mixed / spec decode / larger M / no metadata): g1 = f_b_proj(f_a)
        (the original module call) and the original ``_forward``, unchanged.
CUDA graphs: FULL (uniform decode) captures the fused kernel; with breakable (piecewise) capture the
function is an eager segment, so the fused/unfused decision is re-made from the live metadata at every
replay (mixed batches take the original path, f_b_proj then runs eagerly in that segment).  The
original ``_forward`` is itself an eager break; calling it from inside our eager segment runs it
directly (the capture is paused there), so there is no nesting issue.

Statically eligible layer: 6 local heads x 128 (TP16), native decode backend, decode conv / norm
buffers present, f_b_proj a bias-free unquantized bf16 [768, 128] linear, and the TP8 projection-
overlap path off.  f_b_proj must be UnquantizedLinearMethod (cuBLAS; on SM100 768x128 has no
low-latency plan) for bit-identical results; with a KimiK3LowLatencyLinearMethod plan (SM103 table,
dsv3 at M=1) g can differ from the unfused path by 1 bf16 ulp in ~1e-4 of the elements.
"""

from __future__ import annotations

import os

import torch

_STATE = {
    "patched": False,
    "orig_forward": None,
    "orig__forward": None,  # unwrapped KimiK3DeltaAttention._forward
    "dispatch": None,
    "stats": {"fused": 0, "fallback": 0, "no_metadata": 0},
    "reported": False,
}


def _enabled() -> bool:
    return os.environ.get("K3KDAFB_DISABLE", "0") != "1"


def _max_m() -> int:
    return int(os.environ.get("K3KDAFB_MAX_M", "16"))


def _static_ok(self) -> bool:
    ok = getattr(self, "_k3kdafb_ok", None)
    if ok is not None:
        return ok
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    fb = self.f_b_proj
    w = getattr(fb, "weight", None)
    ok = bool(
        hasattr(torch.ops.k3kdas, "fused_kda_decode_fb")
        and self.local_num_heads == 6
        and self.head_dim == 128
        and getattr(self, "_projection_overlap_max_tokens", 0) <= 0
        and self.kda_decode_backend == "native"
        and self.decode_conv1d_weight is not None
        and self.decode_norm_weight is not None
        and getattr(fb, "bias", None) is None
        and isinstance(fb.quant_method, UnquantizedLinearMethod)
        and isinstance(w, torch.Tensor)
        and w.dtype == torch.bfloat16
        and w.is_cuda
        and tuple(w.shape) == (768, 128)
        and w.is_contiguous()
        and w.data_ptr() % 16 == 0
    )
    self._k3kdafb_ok = ok
    return ok


def _k3kdafb_impl(self, mixed_qkv: torch.Tensor, f_a: torch.Tensor, g2: torch.Tensor,
                  beta: torch.Tensor, core_attn_out: torch.Tensor) -> None:
    """Eager-break body: fused f_b_proj + KDA decode, or f_b_proj + the original _forward."""
    from einops import rearrange

    from vllm.forward_context import get_forward_context
    from vllm.models.kimi_k3.nvidia import kda as k3_kda

    md_raw = get_forward_context().attn_metadata
    if md_raw is None:  # profile / dummy run: the original _forward returns here too
        _STATE["stats"]["no_metadata"] += 1
        return
    m = md_raw.get(self.prefix) if isinstance(md_raw, dict) else None
    n = m.num_actual_tokens if m is not None else 0
    fused = (
        m is not None
        and m.num_spec_decodes == 0
        and m.num_prefills == 0
        and m.num_decodes > 0
        and 0 < n <= _max_m()
        and m.non_spec_state_indices_tensor is not None
    )
    if fused:
        fa = f_a[:n]
        fused = (fa.stride(1) == 1 and (n == 1 or fa.stride(0) % 4 == 0)
                 and fa.data_ptr() % 8 == 0)
    if not fused:
        g1 = rearrange(self.f_b_proj(f_a)[0], "n (h d) -> 1 n h d", d=self.head_dim)
        _STATE["orig__forward"](self, mixed_qkv=mixed_qkv, g1=g1, g2=g2, beta=beta,
                                core_attn_out=core_attn_out)
        _STATE["stats"]["fallback"] += 1
        return
    conv_state, recurrent_state = self.kv_cache[0], self.kv_cache[1]
    if not k3_kda.is_conv_state_dim_first():  # the kernels consume (..., dim, width - 1)
        conv_state = conv_state.transpose(-1, -2)
    state_indices = m.non_spec_state_indices_tensor[:n]
    if not state_indices.is_contiguous():
        state_indices = state_indices.contiguous()
    torch.ops.k3kdas.fused_kda_decode_fb(
        mixed_qkv[:n], self.decode_conv1d_weight, self.conv1d.bias, conv_state, fa,
        self.f_b_proj.weight, beta[:, :n], self.A_log, self.dt_bias, state_indices,
        recurrent_state, core_attn_out[:, :n], self.gate_lower_bound, g2[:n],
        self.decode_norm_weight, self.o_norm.eps, 0)
    _STATE["stats"]["fused"] += 1
    if not _STATE["reported"]:
        _STATE["reported"] = True
        print(f"[k3kdafb] fused f_b_proj + KDA decode active ({self.prefix}, M={n})", flush=True)


def _dispatch_fn():
    """_k3kdafb_impl wrapped by @eager_break_during_capture, created on first use: the decorator
    reads VLLM_USE_BREAKABLE_CUDAGRAPH when applied, and VllmConfig may auto-enable it after
    plugins are loaded, so decorating lazily (first forward) sees the final setting."""
    fn = _STATE["dispatch"]
    if fn is None:
        from vllm.compilation.breakable_cudagraph import eager_break_during_capture

        fn = _STATE["dispatch"] = eager_break_during_capture(_k3kdafb_impl)
    return fn


def _forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """KimiK3DeltaAttention.forward with f_b_proj moved into the (fused) decode call."""
    if not _static_ok(self):
        return _STATE["orig_forward"](self, hidden_states, positions)
    from einops import rearrange

    num_tokens = hidden_states.size(0)
    projected_qkvgfab = self.in_proj_qkvgfab(hidden_states)[0]
    split_sizes = [3 * self.local_projection_size, self.local_projection_size, self.head_dim,
                   self.local_num_heads]
    if self.in_proj_padding:
        split_sizes.append(self.in_proj_padding)
    mixed_qkv, g_proj_states, f_a, beta = projected_qkvgfab.split(split_sizes, dim=-1)[:4]
    beta = beta.unsqueeze(0)
    g2 = rearrange(g_proj_states, "... (h d) -> ... h d", d=self.head_dim)
    core_attn_out = torch.empty((1, num_tokens, self.local_num_heads, self.head_dim),
                                dtype=hidden_states.dtype, device=hidden_states.device)
    _dispatch_fn()(self, mixed_qkv, f_a, g2, beta, core_attn_out)
    core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
    if self.gemm_rs_ar is not None and self.gemm_rs_ar.should_run(core_attn_out):
        return self.gemm_rs_ar(core_attn_out, self.o_proj.weight)
    return self.o_proj(core_attn_out)[0]


def patch_kda_fb(load_ext) -> bool:
    """Install the patch (idempotent).  Returns True if patched."""
    if not _enabled():
        print("[k3kdafb] K3KDAFB_DISABLE=1: f_b_proj fusion off", flush=True)
        return False
    if _STATE["patched"]:
        return True
    load_ext()
    if not hasattr(torch.ops.k3kdas, "fused_kda_decode_fb"):
        raise RuntimeError("k3kdafb: torch.ops.k3kdas.fused_kda_decode_fb missing "
                           "(load_ext must build agents/kda/kda_split.cu)")
    from vllm.models.kimi_k3.nvidia import kda as k3_kda

    Cls = k3_kda.KimiK3DeltaAttention
    _STATE["orig_forward"] = Cls.forward
    _STATE["orig__forward"] = getattr(Cls._forward, "__wrapped__", Cls._forward)
    Cls.forward = _forward
    _STATE["patched"] = True
    print(f"[k3kdafb] KDA f_b_proj folded into the fused decode kernel (decode M<={_max_m()})",
          flush=True)
    return True
