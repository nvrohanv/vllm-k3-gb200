"""Extra Kimi K3 low-latency GEMM plan entries for decode batches M = 3..16 (SM100, TP16 shapes).

Call ``patch_plans()`` once per process at plugin-registration time, before model construction
(``enable_kimi_k3_low_latency_gemm`` builds the per-module plans at model init). A/B switch:
``K3PLANS_DISABLE=1`` (nothing is patched). ``K3PLANS_ONLY="3216x7168,2112x7168"`` restricts the extra
entries to those (N x K) weights; ``K3PLANS_MAX_M`` (default 16) caps the token count.

What it changes (vllm/models/kimi_k3/nvidia/low_latency_gemm.py):
  * ``_build_plan(spec)``: after the original plan is built, the measured winners in ``EXTRA[(N, K)]``
    are added for their token counts. Every other (shape, M) keeps vLLM's entry (CuTe skinny / dsv3)
    or its cuBLAS fallback. Used both by ``KimiK3LowLatencyLinearMethod`` (KDA in_proj_qkvgfab,
    shared_experts.gate_up_proj) and by ``try_low_latency_gemm`` (MLA qkv_a / output-gate slices of
    fused_qkv_a_g_proj).
  * ``MultiHeadLatentAttention._unquantized_gemm``: the ``EXTRA_MLA`` cells (the 768x7168 output-gate
    slice) are served there, so the identically shaped shared-expert gate_up keeps vLLM's plan.
  * ``_run_plan(plan, x, weight)``: entries with backend ``"fi_splitk"`` run FlashInfer's CuTeDSL low-M
    split-K GEMM (``flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk.run_splitk_dense``: swap-AB
    tcgen05 MMA, K split across a 2/4-CTA cluster and reduced in-kernel through DSMEM, one bf16
    store; no separate split-K reduce kernel) with PDL. Its weight TMA warp starts streaming weights
    before ``griddepcontrol.wait`` (only the activation loads wait), so an early-triggering predecessor
    (k3tail.lamport_attn_res) hides the first ~200 KB/CTA of weight traffic. All other backends go
    to the original ``_run_plan``.
  * Runtime guard: x and weight must be 2-D, contiguous, bf16, on the same CUDA device, with matching
    K; otherwise ``None`` is returned and the caller falls back to cuBLAS exactly as before.
CUDA graphs: the dispatch depends only on (N, K, M) (static per captured graph). FlashInfer compiles a
tactic on its first call, which happens in vLLM's eager warmup forward before each capture; replay has
no Python. Numerics: fp32 accumulation, one bf16 rounding (same class as cuBLAS; not bit-identical).
"""

from __future__ import annotations

import os

import torch

K = 7168
_S1 = (64, 8, 2, 12)   # mma_m (public N tile), mma_n (public M tile), split_k (cluster), ab_stages
_S2 = (64, 16, 2, 11)
_S4 = (64, 8, 4, 12)
_S4W = (64, 16, 4, 10)

# (N, K) -> {M: (backend, config)}; only cells measured to win in the model-like chain (RESULTS.md).
# Installed into vLLM's plan table (KimiK3LowLatencyLinearMethod + try_low_latency_gemm).
EXTRA: dict[tuple[int, int], dict[int, tuple[str, tuple]]] = {
    # KDA in_proj_qkvgfab (69 layers): cuBLAS splitK + splitKreduce today for M >= 3
    (3216, K): {**{m: ("fi_splitk", _S1) for m in range(3, 9)},
                **{m: ("fi_splitk", _S2) for m in range(9, 17)}},
    # MLA qkv_a slice of fused_qkv_a_g_proj (24 layers): cuBLAS, or dsv3 at M = 4 / 16, today
    (2112, K): {**{m: ("fi_splitk", _S4) for m in range(3, 9)},
                **{m: ("fi_splitk", _S4W) for m in range(9, 17)}},
}
# Same format, applied ONLY to MultiHeadLatentAttention._unquantized_gemm (the MLA output-gate slice,
# which runs on the aux stream next to qkv_a). 768x7168 is also the shared-expert gate_up shape, which
# runs next to the routed experts, off the critical path, and measured no gain -> left on vLLM's plan.
EXTRA_MLA: dict[tuple[int, int], dict[int, tuple[str, tuple]]] = {
    (768, K): {m: ("fi_splitk", _S4) for m in range(3, 17)},
}

_STATE = {"patched": False, "orig_build_plan": None, "orig_run_plan": None, "orig_mla_gemm": None,
          "tactics": {},
          "stats": {"fi_splitk": 0, "fallback": 0}}


def _enabled() -> bool:
    return os.environ.get("K3PLANS_DISABLE", "0") != "1"


def _extra_table(src=None) -> dict:
    max_m = int(os.environ.get("K3PLANS_MAX_M", "16"))
    only = os.environ.get("K3PLANS_ONLY", "")
    keep = None
    if only:
        keep = {tuple(int(v) for v in item.split("x")) for item in only.split(",") if item}
    out = {}
    for nk, cells in (EXTRA if src is None else src).items():
        if keep is not None and nk not in keep:
            continue
        out[nk] = {m: e for m, e in cells.items() if m <= max_m}
    return out


def _tactic(cfg):
    t = _STATE["tactics"].get(cfg)
    if t is None:
        from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import SplitKTactic

        t = SplitKTactic(*cfg)
        _STATE["tactics"][cfg] = t
    return t


def run_fi_splitk(x: torch.Tensor, weight: torch.Tensor, cfg) -> torch.Tensor | None:
    """y = x @ weight.T with FlashInfer's CuTeDSL low-M split-K GEMM (PDL), or None if not eligible."""
    if not (x.dim() == 2 and weight.dim() == 2 and x.is_contiguous() and weight.is_contiguous()
            and x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16 and x.is_cuda
            and x.device == weight.device and x.shape[1] == weight.shape[1] and 1 <= x.shape[0] <= 32):
        _STATE["stats"]["fallback"] += 1
        return None
    from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import run_splitk_dense

    out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    run_splitk_dense(x, weight.t(), None, out, True, _tactic(cfg))
    _STATE["stats"]["fi_splitk"] += 1
    return out


def patch_plans() -> None:
    """Install the extra plan entries (idempotent). Call before model construction."""
    if _STATE["patched"] or not _enabled():
        return
    from vllm.models.kimi_k3.nvidia import low_latency_gemm as llg
    from vllm.platforms import current_platform

    if not current_platform.is_device_capability((10, 0)):
        print("[k3opt] K3PLANS: not SM100, extra GEMM plans not installed", flush=True)
        return
    try:
        from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import run_splitk_dense  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"[k3opt] K3PLANS: FlashInfer CuTeDSL split-K GEMM unavailable ({e!r})", flush=True)
        return
    table = _extra_table()
    mla_table = _extra_table(EXTRA_MLA)
    orig_build = llg._build_plan
    orig_run = llg._run_plan
    _STATE["orig_build_plan"] = orig_build
    _STATE["orig_run_plan"] = orig_run

    def _build_plan(spec):
        plan = orig_build(spec)
        extra = table.get((spec.n, spec.k))
        if extra:
            plan = dict(plan)
            plan.update(extra)
        return plan

    def _run_plan(plan, x, weight):
        entry = plan.get(x.shape[0])
        if entry is not None and entry[0] == "fi_splitk":
            return run_fi_splitk(x, weight, entry[1])
        return orig_run(plan, x, weight)

    llg._build_plan = _build_plan
    llg._run_plan = _run_plan

    if mla_table:
        from vllm.models.kimi_k3.nvidia import mla as k3_mla

        Cls = k3_mla.MultiHeadLatentAttention
        orig_gemm = Cls._unquantized_gemm
        _STATE["orig_mla_gemm"] = orig_gemm

        def _unquantized_gemm(self, hidden_states, weight):
            cells = mla_table.get((weight.shape[0], weight.shape[1])) if weight.dim() == 2 else None
            entry = cells.get(hidden_states.shape[0]) if cells else None
            if entry is not None and entry[0] == "fi_splitk":
                out = run_fi_splitk(hidden_states, weight, entry[1])
                if out is not None:
                    return out
            return orig_gemm(self, hidden_states, weight)

        Cls._unquantized_gemm = _unquantized_gemm
    _STATE["patched"] = True
    cells = sum(len(v) for v in table.values()) + sum(len(v) for v in mla_table.values())
    print(f"[k3opt] K3PLANS: {cells} extra low-M GEMM plan cells (FlashInfer CuTeDSL split-K) for "
          + ", ".join(f"{n}x{k}" for n, k in table)
          + (" + MLA-only " + ", ".join(f"{n}x{k}" for n, k in mla_table) if mla_table else ""), flush=True)


def stats() -> dict:
    """Python-side counters (eager/capture-time calls only; graph replays don't run Python)."""
    return dict(_STATE["stats"])
