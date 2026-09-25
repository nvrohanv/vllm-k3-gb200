"""vLLM integration of the PDL weight-prefetch GEMV (torch.ops.k3gemv.gemv, agents/prefetch/k3gemv.cu).

Call ``patch_gemv(load_ext)`` once per process at plugin time (before model construction).
``load_ext()`` must build/load ``k3gemv.cu`` so that ``torch.ops.k3gemv.gemv`` exists.
A/B switch: ``K3GEMV_DISABLE=1`` (no patching). ``K3GEMV_EARLY=1`` additionally enables the cells
that only win when the GEMM's predecessor triggers griddepcontrol.launch_dependents early.

Routing (decided only from static shapes -> safe under FULL and breakable/piecewise CUDA graphs):
a local GEMM y[M,N] = x[M,K] @ W[N,K]^T with bf16 x/W, no bias, contiguous 2-D x and W, whose
(N, K, M) is in CELLS goes to k3gemv (fp32 accumulate, one bf16 rounding -- same class as
cuBLAS); everything else is untouched. Hooks:
  * ``low_latency_gemm._run_plan`` -- used by KimiK3LowLatencyLinearMethod.apply (in_proj_qkvgfab,
    shared_experts.gate_up_proj, o_proj, routed_expert_down_proj, ...) and by try_low_latency_gemm
    (MultiHeadLatentAttention._unquantized_gemm: the qkv_a and output-gate slices of
    fused_qkv_a_g_proj). Only cells where the measured plan has no entry (vLLM falls back to
    cuBLAS F.linear) are listed, so tuned CuTe / dsv3 plan entries keep running.
  * ``UnquantizedLinearMethod.apply`` -- modules without a plan (shared_experts.down_proj
    7168x384); only used for K3GEMV_EARLY cells.
The linear-method level replaces only the local GEMM: RowParallelLinear's reduce_results
all-reduce (and the MoE runner's fused shared-expert reduction) still run as before.
"""

from __future__ import annotations

import os

import torch

# (N, K) -> {M: stage}. Measured in bench_cells.py (chain with a ~4.5 us latency-bound
# predecessor, weights from HBM); these win even when the predecessor never triggers early.
CELLS_DEFAULT: dict[tuple[int, int], dict[int, int]] = {
    (3216, 7168): {m: 2 for m in (3, 4, 5, 6, 7, 8)},  # KDA in_proj_qkvgfab
    (2112, 7168): {m: 2 for m in (5, 6, 7, 8)},        # MLA qkv_a slice of fused_qkv_a_g_proj
    (768, 7168): {m: 2 for m in (5, 6, 7, 8)},         # MLA gate slice / shared gate_up_proj
    (3584, 7168): {m: 2 for m in (4, 5, 6, 7, 8)},     # routed_expert_down_proj (if not sharded)
}
# Win only if the predecessor calls launch_dependents early (e.g. k3tail lamport_attn_res before
# qkv_a, the KDA decode kernel before KDA o_proj); roughly neutral otherwise.
CELLS_EARLY: dict[tuple[int, int], dict[int, int]] = {
    (2112, 7168): {3: 4},
    (7168, 768): {m: 1 << 20 for m in (3, 4, 5, 6, 7, 8)},  # o_proj (KDA + MLA)
    (7168, 384): {m: 1 << 20 for m in (3, 4, 5, 6, 7, 8)},  # shared_experts.down_proj
}
NW = {7168: 8, 768: 8, 384: 4}

_STATE = {"patched": False, "cells": {}, "orig_run_plan": None, "orig_apply": None,
          "stats": {"k3gemv": 0}}


def _enabled() -> bool:
    return os.environ.get("K3GEMV_DISABLE", "0") != "1"


def _build_cells() -> dict:
    cells = {k: dict(v) for k, v in CELLS_DEFAULT.items()}
    # K3GEMV_ONLY="3216x7168,2112x7168": restrict routing to these (N x K) weights.
    only = os.environ.get("K3GEMV_ONLY", "")
    if only:
        keep = {tuple(int(v) for v in item.split("x")) for item in only.split(",") if item}
        cells = {k: v for k, v in cells.items() if k in keep}
    if os.environ.get("K3GEMV_EARLY", "0") == "1":
        for k, v in CELLS_EARLY.items():
            cells.setdefault(k, {}).update(v)
    return cells


def k3gemv_or_none(x: torch.Tensor, weight: torch.Tensor):
    """k3gemv result for a routed (N, K, M) cell, else None (caller runs its original path)."""
    if x.dim() != 2 or weight.dim() != 2:
        return None
    N, K = weight.shape
    stage = _STATE["cells"].get((N, K), {}).get(x.shape[0])
    if stage is None:
        return None
    if not (x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16 and x.shape[1] == K
            and x.is_cuda and weight.device == x.device and x.is_contiguous()
            and weight.is_contiguous() and x.data_ptr() % 16 == 0 and weight.data_ptr() % 16 == 0):
        return None
    y = torch.empty((x.shape[0], N), dtype=x.dtype, device=x.device)
    torch.ops.k3gemv.gemv(x, weight, y, stage, 0, 0, NW[K], 1, 0, None)
    _STATE["stats"]["k3gemv"] += 1
    return y


def _run_plan(plan, x, weight):
    """low_latency_gemm._run_plan with k3gemv cells first."""
    y = k3gemv_or_none(x, weight)
    return y if y is not None else _STATE["orig_run_plan"](plan, x, weight)


def _apply(self, layer, x, bias=None):
    """UnquantizedLinearMethod.apply with k3gemv cells first (local GEMM only)."""
    if bias is None:
        y = k3gemv_or_none(x, layer.weight)
        if y is not None:
            return y
    return _STATE["orig_apply"](self, layer, x, bias)


def patch_gemv(load_ext) -> None:
    """Install the routing (idempotent). Call before model construction in every worker."""
    if _STATE["patched"]:
        return
    _STATE["patched"] = True
    if not _enabled():
        print("[k3opt] K3GEMV: disabled (K3GEMV_DISABLE=1)", flush=True)
        return
    load_ext()
    if not (hasattr(torch.ops, "k3gemv") and hasattr(torch.ops.k3gemv, "gemv")):
        raise RuntimeError("k3gemv.gemv not loaded (build k3gemv.cu)")
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.models.kimi_k3.nvidia import low_latency_gemm as llg

    _STATE["cells"] = _build_cells()
    _STATE["orig_run_plan"] = llg._run_plan
    _STATE["orig_apply"] = UnquantizedLinearMethod.apply
    llg._run_plan = _run_plan
    UnquantizedLinearMethod.apply = _apply
    desc = ", ".join(f"{n}x{k}:M{min(ms)}-{max(ms)}" for (n, k), ms in sorted(_STATE["cells"].items()))
    print(f"[k3opt] K3GEMV: PDL weight-prefetch GEMV (k3gemv.gemv) for {desc}"
          f"{' (+early-trigger cells)' if os.environ.get('K3GEMV_EARLY', '0') == '1' else ''}",
          flush=True)


def stats() -> dict:
    """Python-side call count (eager / capture-time calls; graph replays don't run Python)."""
    return dict(_STATE["stats"])
