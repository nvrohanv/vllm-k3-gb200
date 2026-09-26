"""Extra Kimi K3 low-latency GEMM plan entries for decode batches M = 3..16 (SM100, TP16 shapes).

Call ``patch_plans()`` once per process at plugin-registration time, before model construction
(``enable_kimi_k3_low_latency_gemm`` builds the per-module plans at model init). A/B switch:
``K3PLANS_DISABLE=1`` (nothing is patched). ``K3PLANS_ONLY="3216x7168,2112x7168"`` restricts the extra
entries to those (N x K) weights; ``K3PLANS_MIN_M`` (default 3) / ``K3PLANS_MAX_M`` (default 16) bound the
token counts. ``K3PLANS_MIN_M=1`` additionally enables the M = 1..2 cells (FlashInfer direct GEMM for the
KDA in_proj and the MLA gate; ~0.2 us/call in the chain, i.e. marginal -- A/B before adopting).

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
    (k3tail.lamport_attn_res) hides the first ~200 KB/CTA of weight traffic. ``"fi_direct"`` entries
    (opt-in M = 1..2 only) run FlashInfer's CuTeDSL direct CUDA-core GEMM (``run_direct_dense``) with
    PDL. All other backends go to the original ``_run_plan``.
  * Runtime guard: x and weight must be 2-D, contiguous, bf16, on the same CUDA device, with matching
    K; otherwise ``None`` is returned and the caller falls back to cuBLAS exactly as before.
A7 (l2pf, 2026-09-26; env ``K3PLANS_A7``, default off): same-bytes CuTe skinny cells with ``static_k`` for the
M = 1..2 decode GEMMs. vLLM's ``CuteSkinnyGemm`` with ``static_k`` loads the first 2 K-tiles of each CTA's weight
rows into registers BEFORE ``griddepcontrol.wait``, i.e. while the early-triggering ``lamport_attn_res`` still
runs, and fully unrolls K. vLLM's shipped SM100 configs leave ``static_k`` unset. ``K3PLANS_A7`` is a comma list:
  * ``m2``  : M=2 cells (``A7_M2``), * ``m1`` : M=1 cells (``A7_M1``), * ``up`` : MoE-tail up-projection
    (fused_add_multicast) configs (``A7_UP``), * ``ef4`` : every FlashInfer split-K cell (M = 3..16) runs with its
    weight TMA at L2::evict_first (``fi_splitk_ef``, see ``fsk_ef_source``); ``1`` / ``all`` = every group.
Backend ``cute_ef`` runs the same kernel built with the pre-PDL prefetch loaded STREAMING (.cs, like every later
K-tile) instead of ALWAYS, so the dead weights never sit in L2 at evict_normal (RESULTS.md Task 6: dead
evict_normal weights slow the next MoE); ``cute_sk`` runs vLLM's own kernel (ALWAYS prefetch).

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
# FlashInfer CuTeDSL direct (CUDA-core) GEMM tactics (block_size, outputs_per_block, rows_per_block),
# used only for the opt-in M = 1..2 cells (K3PLANS_MIN_M=1).
_D11 = (128, 1, 1)
_D21 = (128, 2, 1)
_D22 = (128, 2, 2)

# (N, K) -> {M: (backend, config)}; only cells measured to win in the model-like chain (RESULTS.md).
# Installed into vLLM's plan table (KimiK3LowLatencyLinearMethod + try_low_latency_gemm).
EXTRA: dict[tuple[int, int], dict[int, tuple[str, tuple]]] = {
    # KDA in_proj_qkvgfab (69 layers): cuBLAS splitK + splitKreduce today for M >= 3
    # M = 1..2 (opt-in, K3PLANS_MIN_M=1): measured -0.17 / -0.19 us vs vLLM's CuTe skinny (noise ~0.1)
    (3216, K): {1: ("fi_direct", _D11), 2: ("fi_direct", _D22),
                **{m: ("fi_splitk", _S1) for m in range(3, 9)},
                **{m: ("fi_splitk", _S2) for m in range(9, 17)}},
    # MLA qkv_a slice of fused_qkv_a_g_proj (24 layers): cuBLAS, or dsv3 at M = 4 / 16, today
    (2112, K): {**{m: ("fi_splitk", _S4) for m in range(3, 9)},
                **{m: ("fi_splitk", _S4W) for m in range(9, 17)}},
}
# Same format, applied ONLY to MultiHeadLatentAttention._unquantized_gemm (the MLA output-gate slice,
# which runs on the aux stream next to qkv_a). 768x7168 is also the shared-expert gate_up shape, which
# runs next to the routed experts, off the critical path, and measured no gain -> left on vLLM's plan.
EXTRA_MLA: dict[tuple[int, int], dict[int, tuple[str, tuple]]] = {
    # M = 1..2 (opt-in, K3PLANS_MIN_M=1): measured -0.20 / -0.23 us vs vLLM's CuTe skinny gate
    (768, K): {1: ("fi_direct", _D21), 2: ("fi_direct", _D22),
               **{m: ("fi_splitk", _S4) for m in range(3, 17)}},
}

# ---------------------------------------------------------------------------------------------------- A7
# (N, K) -> {M: (backend, (block_size, outputs_per_block, k_unroll, vector_width))}; static_k = K always.
# Measured in the model-like chain (agents/l2pf/a7/, chain.py with k3pf on): see RESULTS.md "A7".
A7_M2: dict[tuple[int, int], dict[int, tuple[str, tuple]]] = {
    # KDA in_proj: vLLM (128,4,2) -> static_k (64,4,2) STREAMING prefetch: -0.98 us/call (chain, pf), -1.2 us/layer
    # with the downstream MoE in the chain (pf_chain M=2)
    (3216, K): {2: ("cute_ef", (64, 4, 2, 8))},
    # MLA qkv_a: vLLM (128,2,2) -> static_k (64,4,2) cute_ef; only pays together with the gate cell below
    (2112, K): {2: ("cute_ef", (64, 4, 2, 8))},
}
A7_M2_MLA: dict[tuple[int, int], dict[int, tuple[str, tuple]]] = {
    # MLA output gate (aux stream, joined before q_prep = the M=2 MLA critical path): vLLM skinny (224,2,2) ->
    # FlashInfer direct (128,2,2) (STREAMING weight loads). qkv_a cute_ef + this gate: -0.62 us/layer (pf)
    (768, K): {2: ("fi_direct", _D22)},
}
A7_M1: dict[tuple[int, int], dict[int, tuple[str, tuple]]] = {
    # KDA in_proj M=1: vLLM (224,3,4) -> static_k (128,4,1) cute_ef: -0.74 us/call (chain, pf) and -0.46..-0.76 us per
    # layer in the full-layer chain with the MoE (pf_chain --tgate 1, 4 runs). NOT (224,3,1): faster alone (-1.12) but
    # it prefetches half of in_proj (23 MB) before its PDL wait, inside the H4 hop window, and nets +0.1..+0.4 per
    # layer in the full chain; NOT the ALWAYS (vLLM) prefetch: its dead evict_normal lines slow the next MoE +1.0 us.
    (3216, K): {1: ("cute_ef", (128, 4, 1, 8))},
    # MLA qkv_a M=1: not shipped (GEMM-only chain -1.03 with (224,3,1), but no full-layer measurement).
}
A7_M1_MLA: dict[tuple[int, int], dict[int, tuple[str, tuple]]] = {}
# MoE-tail up-projection (vLLM fused_add_multicast_skinny_gemm, shard 448): M -> (block, opb, k_unroll, vw).
# Sweep at M = 1..4 (a7/fam_sweep.py): vLLM's config_for_m (224,2,1,16) is already the fastest -> left empty.
A7_UP: dict[int, tuple[int, int, int, int]] = {}


def _a7_groups() -> set:
    v = os.environ.get("K3PLANS_A7", "0").strip().lower()
    if v in ("", "0", "off", "none"):
        return set()
    if v in ("1", "all", "on"):
        return {"m1", "m2", "up", "ef4"}
    return {g.strip() for g in v.split(",") if g.strip()}


def _merge(dst: dict, src: dict) -> None:
    for nk, cells in src.items():
        if cells:
            dst.setdefault(nk, {}).update(cells)


_SKINNY_EF = None


def _skinny_ef():
    """vLLM's shape-dynamic CuTe skinny GEMM, rebuilt with the static_k pre-PDL prefetch loaded STREAMING."""
    global _SKINNY_EF
    if _SKINNY_EF is None:
        import sys

        from vllm.model_executor.kernels.linear.cute_dsl import _skinny_gemm as vsg
        from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import ShapeDynamicSkinnyGemm

        old = ("        copy_b_prefetch = cute.make_copy_atom(\n"
               "            cute.nvgpu.CopyG2ROp(),\n"
               "            self.element_type,\n"
               "            num_bits_per_copy=self.vector_width * self.element_type.width,\n"
               "            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,\n"
               "        )")
        src = open(vsg.__file__).read()
        if src.count(old) != 1:
            raise RuntimeError("K3PLANS_A7: vLLM _skinny_gemm.py changed (prefetch copy atom not found once)")
        # CuTe DSL re-reads @cute.jit sources from their file: write the patched kernel to a real file
        # (content-addressed, next to this module or in $K3PLANS_EF_DIR) and import it from there.
        import hashlib
        import importlib.util

        new = src.replace(old, old.replace("LoadCacheMode.ALWAYS", "LoadCacheMode.STREAMING"))
        d = os.environ.get("K3PLANS_EF_DIR") or os.path.dirname(os.path.abspath(__file__))
        if not os.access(d, os.W_OK):
            import tempfile

            d = tempfile.gettempdir()
        path = os.path.join(d, f"_k3plans_skinny_ef_{hashlib.md5(new.encode()).hexdigest()[:8]}.py")
        if not os.path.exists(path):
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                f.write(new)
            os.replace(tmp, path)
        spec = importlib.util.spec_from_file_location("k3plans_skinny_ef_kernel", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)

        class _EF(ShapeDynamicSkinnyGemm):
            def _compile(self, dtype, config, has_residual):
                import cutlass.cute as cute
                from quack.compile_utils import make_fake_tensor

                et = self._cutlass_dtype(dtype)
                n = cute.sym_int(divisibility=config.outputs_per_block)
                k = (config.static_k if config.static_k is not None
                     else cute.sym_int(divisibility=config.block_size * config.vector_width))
                a = make_fake_tensor(et, (config.num_rows, k), divisibility=config.vector_width)
                b = make_fake_tensor(et, (n, k), divisibility=config.vector_width)
                c = make_fake_tensor(et, (config.num_rows, n), divisibility=1)
                r = make_fake_tensor(et, (config.num_rows, n), divisibility=1)
                kern = mod.CuteSkinnyGemm(element_type=et, num_rows=config.num_rows, block_size=config.block_size,
                                          outputs_per_block=config.outputs_per_block,
                                          vector_width=config.vector_width, k_unroll=config.k_unroll,
                                          has_residual=has_residual, use_pdl=self._use_pdl(),
                                          static_k=config.static_k)
                self._compiled[(dtype, config, has_residual)] = cute.compile(
                    kern, a, b, r, c, self._stream(), options="--enable-tvm-ffi --ptxas-options -maxrregcount=64")

        _SKINNY_EF = _EF()
    return _SKINNY_EF


# ---------------------------------------------------------------------------------------------------- EF4
# K3PLANS_A7 group "ef4": every fi_splitk cell (M = 3..16: KDA in_proj, MLA qkv_a, MLA gate) runs FlashInfer's
# split-K kernel rebuilt so the weight TMA carries L2::evict_first (fi_splitk_ef). Bit-identical outputs.
_FSK_EF_EDITS: list = []


def _e(old: str, new: str, count: int = 1) -> None:
    _FSK_EF_EDITS.append((old, new, count))


# Text edits of FlashInfer's dense_bf16_gemm_sm100_splitk.py (flashinfer 0.6.18.post1); verbatim from
# agents/l2pf/ef4/make_fsk_ef.py.
# 1) kernel object: flag + tensor-map argument plumbing
_e("""        use_pdl: bool,
        has_bias: bool,
    ) -> None:
        self.acc_dtype = cutlass.Float32""", """        use_pdl: bool,
        has_bias: bool,
        ef_a: bool = True,
    ) -> None:
        self.ef_a = ef_a
        self.acc_dtype = cutlass.Float32""")
_e("""        bias: cute.Tensor,
        stream: _cuda.CUstream,
    ):
        # Grid-y packs output-N tile and cluster rank.
        self.kernel(a, b, c, bias).launch(""", """        bias: cute.Tensor,
        a_tmap: cutlass.Int64,
        stream: _cuda.CUstream,
    ):
        # Grid-y packs output-N tile and cluster rank.
        self.kernel(a, b, c, bias, a_tmap).launch(""")
_e("""        mBias: cute.Tensor,  # Broadcast bias; dead when has_bias=False
    ):
        \"\"\"Allocate storage and dispatch the specialized warps.\"\"\"""", """        mBias: cute.Tensor,  # Broadcast bias; dead when has_bias=False
        a_tmap: cutlass.Int64,  # device tensor map of mA (weights), box {64, cta_m}, SW128 (k3 EF4)
    ):
        \"\"\"Allocate storage and dispatch the specialized warps.\"\"\"""")
_e("""                cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A"),
                k_tile_start,
                k_tile_count,
                True,
            )""", """                cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A"),
                k_tile_start,
                k_tile_count,
                True,
                a_tmap,
                bidx * self.cta_m,
            )""")
_e("""                cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B"),
                k_tile_start,
                k_tile_count,
                False,
            )""", """                cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B"),
                k_tile_start,
                k_tile_count,
                False,
                a_tmap,
                Int32(0),
            )""")
# 2) dma_warp: hand-issued evict_first TMA for A
_e("""        k_tile_count: cutlass.Int32,
        is_a: cutlass.Constexpr,
    ):
        stages = self.num_ab_stage
        if cutlass.const_expr(not is_a and self.use_pdl):""", """        k_tile_count: cutlass.Int32,
        is_a: cutlass.Constexpr,
        a_tmap: cutlass.Int64,
        row0: cutlass.Int32,
    ):
        stages = self.num_ab_stage
        if cutlass.const_expr(not is_a and self.use_pdl):""")
_e("""            cute_ext.tma_load(
                g_tile[None, None, k_tile_start + k_tile],
                s_tile[None, None, None, stage],
                (bar_full + stage).value,
                cta_v_map=cta_v_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
            )
            if stage == stages - 1:""", """            if cutlass.const_expr(is_a and self.ef_a):
                with cute.arch.elect_one():
                    _tma_a_evict_first(
                        s_tile[None, None, None, stage].iterator,
                        bar_full + stage,
                        a_tmap,
                        (k_tile_start + k_tile) * Int32(self.cta_k),
                        row0,
                        Int32(self.cta_m * 128),
                    )
            else:
                cute_ext.tma_load(
                    g_tile[None, None, k_tile_start + k_tile],
                    s_tile[None, None, None, stage],
                    (bar_full + stage).value,
                    cta_v_map=cta_v_map,
                    tma_operation_type=self.tma_op,
                    update_expect_tx=False,
                )
            if stage == stages - 1:""")
# 3) the TMA helper (next to FlashInfer's own inline-asm helpers)
_e("""#: Rank that gathers partials and stores the output.
OWNER_RANK = 0""", """@dsl_user_op
def _tma_a_evict_first(
    smem_ptr: "cute.Pointer",
    mbar_ptr: "cute.Pointer",
    tmap: "cutlass.Int64",
    k0: "Int32",
    row0: "Int32",
    atom_bytes: "Int32",
    *,
    loc=None,
    ip=None,
) -> None:
    \"\"\"k3 EF4: one K-tile (2 x 64-element SW128 atoms, stacked cta_m rows apart) of A, L2 evict_first.\"\"\"
    llvm.inline_asm(
        None,
        [
            smem_ptr.toint(loc=loc, ip=ip).ir_value(),
            tmap.ir_value(loc=loc, ip=ip),
            k0.ir_value(loc=loc, ip=ip),
            row0.ir_value(loc=loc, ip=ip),
            mbar_ptr.toint(loc=loc, ip=ip).ir_value(),
            atom_bytes.ir_value(loc=loc, ip=ip),
        ],
        "{\\n\\t.reg .b64 pol;\\n\\t.reg .b32 d1, k1;\\n\\t"
        "createpolicy.fractional.L2::evict_first.b64 pol, 1.0;\\n\\t"
        "add.u32 d1, $0, $5;\\n\\tadd.u32 k1, $2, 64;\\n\\t"
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[$0], [$1, {$2, $3}], [$4], pol;\\n\\t"
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[d1], [$1, {k1, $3}], [$4], pol;\\n\\t}",
        "r,l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


#: Rank that gathers partials and stores the output.
OWNER_RANK = 0""")
# 4) host side: thread the tensor-map pointer through the bmm wrappers, compile and run
_e("""    c: cute.Tensor,
    stream: _cuda.CUstream,
):
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        c,
        cute.make_tensor(c.iterator, cute.select(c.layout, mode=[0, 1, 2])),
        stream,
    )""", """    c: cute.Tensor,
    a_tmap: cutlass.Int64,
    stream: _cuda.CUstream,
):
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        c,
        cute.make_tensor(c.iterator, cute.select(c.layout, mode=[0, 1, 2])),
        a_tmap,
        stream,
    )""")
_e("""    bias: cute.Tensor,
    stream: _cuda.CUstream,
):
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0])),
        cute.make_tensor(bias.iterator, cute.select(bias.layout, mode=[1, 2, 0])),
        stream,
    )""", """    bias: cute.Tensor,
    a_tmap: cutlass.Int64,
    stream: _cuda.CUstream,
):
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0])),
        cute.make_tensor(bias.iterator, cute.select(bias.layout, mode=[1, 2, 0])),
        a_tmap,
        stream,
    )""")
_e("""    if has_bias:
        compiled = cute_ext.compile(_bmm_bias, kernel, *compile_tensors, stream)
    else:
        compiled = cute_ext.compile(_bmm_no_bias, kernel, *compile_tensors[:3], stream)
    return compiled""", """    if has_bias:
        compiled = cute_ext.compile(_bmm_bias, kernel, *compile_tensors, cutlass.Int64(0), stream)
    else:
        compiled = cute_ext.compile(_bmm_no_bias, kernel, *compile_tensors[:3], cutlass.Int64(0), stream)
    return compiled""")
_e("""def run_splitk_dense(
    a,
    b,
    bias,
    out,
    pdl: bool,
    tactic: SplitKTactic,
):
    \"\"\"Run ``A[M,K] @ B[K,N]`` with the ``mm_bf16`` layouts.\"\"\"""", """def run_splitk_dense(
    a,
    b,
    bias,
    out,
    pdl: bool,
    tactic: SplitKTactic,
    a_tmap: int = 0,
):
    \"\"\"Run ``A[M,K] @ B[K,N]`` with the ``mm_bf16`` layouts. k3 EF4: ``a_tmap`` = device address of the tensor
    map of the weight ``b.t()`` ([N, K] row-major; ef4/tmap.cu, box rows = tactic.mma_m); required.\"\"\"
    if not a_tmap:
        raise ValueError("fsk_ef: a_tmap (device tensor map of the weight) is required")""")
_e("""    if has_bias:
        compiled(*cute_tensors[:4], stream)
    else:
        compiled(*cute_tensors[:3], stream)
    return out""", """    if has_bias:
        compiled(*cute_tensors[:4], cutlass.Int64(a_tmap), stream)
    else:
        compiled(*cute_tensors[:3], cutlass.Int64(a_tmap), stream)
    return out""")


def fsk_ef_source(src: str) -> str:
    """FlashInfer split-K source -> the same kernel with the weight (kernel-A) TMA hand-issued with
    L2::evict_first through a device tensor map (Int64 kernel arg a_tmap). Every edit must match exactly once,
    else RuntimeError (no silent fallback to a different kernel)."""
    for old, new, count in _FSK_EF_EDITS:
        n = src.count(old)
        if n != count:
            raise RuntimeError(f"K3PLANS ef4: FlashInfer split-K source changed ({n} matches for: {old[:80]!r})")
        src = src.replace(old, new)
    return ("# GENERATED by k3opt plans_patch.fsk_ef_source (k3 EF4: weight TMA with L2::evict_first). "
            "Do not edit by hand.\n") + src


_FSK_EF = {}


def _fsk_ef():
    """(module, tmap_for): the evict_first split-K kernel module (written content-addressed next to this file or to
    $K3PLANS_EF_DIR / the temp dir, CuTe DSL re-parses sources) and a (weight, box rows) -> device tensor map cache."""
    if "mod" not in _FSK_EF:
        import ctypes
        import hashlib
        import importlib.util
        import sys

        from flashinfer.gemm.kernels import dense_bf16_gemm_sm100_splitk as ref

        new = fsk_ef_source(open(ref.__file__).read())
        d = os.environ.get("K3PLANS_EF_DIR") or os.path.dirname(os.path.abspath(__file__))
        if not os.access(d, os.W_OK):
            import tempfile

            d = tempfile.gettempdir()
        path = os.path.join(d, f"_k3plans_fsk_ef_{hashlib.md5(new.encode()).hexdigest()[:8]}.py")
        if not os.path.exists(path):
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                f.write(new)
            os.replace(tmp, path)
        spec = importlib.util.spec_from_file_location("k3plans_fsk_ef_kernel", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _FSK_EF["mod"], _FSK_EF["maps"], _FSK_EF["ctypes"] = mod, {}, ctypes
    return _FSK_EF["mod"]


def _tmap_for(w: torch.Tensor, rows: int) -> int:
    """Device address of a TMA tensor map of w ([N, K] row-major bf16/fp16): box {64, rows}, SWIZZLE_128B.
    Created once per (weight address, shape, rows), outside graph capture (0 if it would be created in capture)."""
    key = (w.data_ptr(), tuple(w.shape), rows, w.dtype)
    t = _FSK_EF["maps"].get(key)
    if t is None:
        if torch.cuda.is_current_stream_capturing():
            return 0
        from cuda.bindings import driver as cu

        dt = (cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 if w.dtype == torch.bfloat16
              else cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT16)
        n, k = w.shape
        err, tm = cu.cuTensorMapEncodeTiled(
            dt, cu.cuuint32_t(2), w.data_ptr(), [cu.cuuint64_t(k), cu.cuuint64_t(n)],
            [cu.cuuint64_t(w.stride(0) * w.element_size())], [cu.cuuint32_t(64), cu.cuuint32_t(rows)],
            [cu.cuuint32_t(1), cu.cuuint32_t(1)], cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
            cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
        if err != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"K3PLANS ef4: cuTensorMapEncodeTiled failed ({err})")
        ctypes = _FSK_EF["ctypes"]
        raw = bytes((ctypes.c_char * 128).from_address(tm.getPtr()))
        t = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(w.device)
        _FSK_EF["maps"][key] = t
    return t.data_ptr()


def run_fi_splitk_ef(x: torch.Tensor, weight: torch.Tensor, cfg) -> torch.Tensor | None:
    """fi_splitk with the weight TMA at L2::evict_first; falls back to fi_splitk if no tensor map is available
    (never created during graph capture) or the weight is not a plain row-major matrix."""
    if not _eligible(x, weight):
        _STATE["stats"]["fallback"] += 1
        return None
    mod = _fsk_ef()
    tm = _tmap_for(weight, cfg[0]) if weight.stride(1) == 1 and weight.data_ptr() % 16 == 0 else 0
    if not tm:
        _STATE["stats"]["fi_splitk_ef_fallback"] += 1
        return run_fi_splitk(x, weight, cfg)
    t = _STATE["tactics"].get(("ef",) + tuple(cfg))
    if t is None:
        t = mod.SplitKTactic(*cfg)
        _STATE["tactics"][("ef",) + tuple(cfg)] = t
    out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    mod.run_splitk_dense(x, weight.t(), None, out, True, t, tm)
    _STATE["stats"]["fi_splitk_ef"] += 1
    return out


def _to_ef(table: dict) -> None:
    for cells in table.values():
        for m, (be, cfg) in list(cells.items()):
            if be == "fi_splitk":
                cells[m] = ("fi_splitk_ef", cfg)


def _sk_config(m: int, k: int, cfg):
    from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import SkinnyGemmConfig

    bs, opb, ku, vw = cfg
    return SkinnyGemmConfig(m, bs, opb, ku, vw, static_k=k)


def _sk_ok(x: torch.Tensor, weight: torch.Tensor, cfg) -> bool:
    bs, opb, _, vw = cfg
    k = weight.shape[1]
    return (_eligible(x, weight) and x.shape[0] <= 16 and weight.shape[0] % opb == 0 and k % (bs * vw) == 0
            and k >= 2 * bs * vw)


def run_cute_sk(x: torch.Tensor, weight: torch.Tensor, cfg, ef: bool = False) -> torch.Tensor | None:
    """y = x @ weight.T with vLLM's CuTe skinny GEMM at static_k = K (ef: STREAMING pre-PDL prefetch), or None."""
    if not _sk_ok(x, weight, cfg):
        _STATE["stats"]["fallback"] += 1
        return None
    config = _sk_config(x.shape[0], weight.shape[1], cfg)
    if ef:
        out = _skinny_ef()(x, weight, config, None)
    else:
        from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import shape_dynamic_skinny_gemm

        out = shape_dynamic_skinny_gemm(x, weight, config, None)
    _STATE["stats"]["cute_ef" if ef else "cute_sk"] += 1
    return out


def _patch_up_proj(cells: dict) -> None:
    """MoE-tail up-projection (shard 448) configs for the listed M; every other (M, shard) keeps vLLM's."""
    from vllm.models.kimi_k3.nvidia.ops.cute_dsl.latent_moe_tail import fused_add_multicast_skinny_gemm as fam

    orig = fam.config_for_m

    def config_for_m(num_rows: int, shard_dim: int = 896):
        c = cells.get(num_rows) if shard_dim == 448 else None
        if c is None:
            return orig(num_rows, shard_dim)
        bs, opb, ku, vw = c
        return fam.SkinnyConfig(block_size=bs, outputs_per_block=opb, k_unroll=ku, vector_width=vw)

    fam.config_for_m = config_for_m
    _STATE["orig_up_config"] = orig


_STATE = {"patched": False, "orig_build_plan": None, "orig_run_plan": None, "orig_mla_gemm": None,
          "tactics": {},
          "stats": {"fi_splitk": 0, "fi_direct": 0, "cute_sk": 0, "cute_ef": 0, "fi_splitk_ef": 0,
                    "fi_splitk_ef_fallback": 0, "fallback": 0},
          "a7": []}


def _enabled() -> bool:
    return os.environ.get("K3PLANS_DISABLE", "0") != "1"


def _extra_table(src=None) -> dict:
    max_m = int(os.environ.get("K3PLANS_MAX_M", "16"))
    min_m = int(os.environ.get("K3PLANS_MIN_M", "3"))  # M = 1..2 stay on vLLM's CuTe plan by default
    only = os.environ.get("K3PLANS_ONLY", "")
    keep = None
    if only:
        keep = {tuple(int(v) for v in item.split("x")) for item in only.split(",") if item}
    out = {}
    for nk, cells in (EXTRA if src is None else src).items():
        if keep is not None and nk not in keep:
            continue
        out[nk] = {m: e for m, e in cells.items() if min_m <= m <= max_m}
    return out


def _tactic(cfg):
    t = _STATE["tactics"].get(cfg)
    if t is None:
        from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import SplitKTactic

        t = SplitKTactic(*cfg)
        _STATE["tactics"][cfg] = t
    return t


def _eligible(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (x.dim() == 2 and weight.dim() == 2 and x.is_contiguous() and weight.is_contiguous()
            and x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16 and x.is_cuda
            and x.device == weight.device and x.shape[1] == weight.shape[1] and 1 <= x.shape[0] <= 32)


def run_fi_splitk(x: torch.Tensor, weight: torch.Tensor, cfg) -> torch.Tensor | None:
    """y = x @ weight.T with FlashInfer's CuTeDSL low-M split-K GEMM (PDL), or None if not eligible."""
    if not _eligible(x, weight):
        _STATE["stats"]["fallback"] += 1
        return None
    from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import run_splitk_dense

    out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    run_splitk_dense(x, weight.t(), None, out, True, _tactic(cfg))
    _STATE["stats"]["fi_splitk"] += 1
    return out


def run_fi_direct(x: torch.Tensor, weight: torch.Tensor, cfg) -> torch.Tensor | None:
    """y = x @ weight.T with FlashInfer's CuTeDSL direct (CUDA-core) GEMM (PDL), or None."""
    if not (_eligible(x, weight) and x.shape[0] % cfg[2] == 0 and weight.shape[0] % cfg[1] == 0):
        _STATE["stats"]["fallback"] += 1
        return None
    from flashinfer.gemm.kernels.dense_bf16_gemm_direct import DirectTactic, run_direct_dense

    t = _STATE["tactics"].get(("direct",) + tuple(cfg))
    if t is None:
        t = DirectTactic(*cfg)
        _STATE["tactics"][("direct",) + tuple(cfg)] = t
    out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    run_direct_dense(x, weight.t(), out, True, t)
    _STATE["stats"]["fi_direct"] += 1
    return out


_RUNNERS = {"fi_splitk": run_fi_splitk, "fi_direct": run_fi_direct,
            "cute_sk": lambda x, w, c: run_cute_sk(x, w, c, False),
            "cute_ef": lambda x, w, c: run_cute_sk(x, w, c, True),
            "fi_splitk_ef": run_fi_splitk_ef}


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
    groups = _a7_groups()
    if "m2" in groups:
        _merge(table, A7_M2)
        _merge(mla_table, A7_M2_MLA)
    if "m1" in groups:
        _merge(table, A7_M1)
        _merge(mla_table, A7_M1_MLA)
    if "up" in groups and A7_UP:
        _patch_up_proj(A7_UP)
    if "ef4" in groups:
        _to_ef(table)
        _to_ef(mla_table)
    _STATE["a7"] = sorted(groups)
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
        if entry is not None and entry[0] in _RUNNERS:
            return _RUNNERS[entry[0]](x, weight, entry[1])
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
            if entry is not None and entry[0] in _RUNNERS:
                out = _RUNNERS[entry[0]](hidden_states, weight, entry[1])
                if out is not None:
                    return out
            return orig_gemm(self, hidden_states, weight)

        Cls._unquantized_gemm = _unquantized_gemm
    _STATE["patched"] = True
    cells = sum(len(v) for v in table.values()) + sum(len(v) for v in mla_table.values())
    if groups:
        print(f"[k3opt] K3PLANS_A7={','.join(sorted(groups))}: static_k CuTe skinny cells M2 "
              f"{ {nk: sorted(v) for nk, v in A7_M2.items() if v} } MLA {A7_M2_MLA if 'm2' in groups else {}} "
              f"M1 {A7_M1 if 'm1' in groups else {}}; up-proj {A7_UP if 'up' in groups else {}}; "
              f"split-K weight TMA evict_first: {'ef4' in groups}", flush=True)
    print(f"[k3opt] K3PLANS: {cells} extra low-M GEMM plan cells (FlashInfer CuTeDSL split-K) for "
          + ", ".join(f"{n}x{k}" for n, k in table)
          + (" + MLA-only " + ", ".join(f"{n}x{k}" for n, k in mla_table) if mla_table else ""), flush=True)


def stats() -> dict:
    """Python-side counters (eager/capture-time calls only; graph replays don't run Python)."""
    return dict(_STATE["stats"])
