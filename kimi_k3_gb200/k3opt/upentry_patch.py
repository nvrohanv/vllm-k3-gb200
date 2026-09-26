"""K3OPT_UPENTRY (PLAN A3 "upentry"): vLLM's latent-MoE tail skinny up-projection (M <= 5) with its PDL trigger at
kernel entry.

What it changes: ``AdaptiveUpProjectionKernel.compile_skinny`` (vllm/.../latent_moe_tail/fused_add_multicast_gemm.py)
builds its per-M kernels from the module global ``FusedAddMulticastSkinnyGemmKernel``; the patch points that global at
``upentry_skinny_patch.FusedAddMulticastSkinnyGemmKernel``, a copy of fused_add_multicast_skinny_gemm.py (v0.30.0, md5
86833e39) whose only change is ``griddepcontrol.launch_dependents`` moved from after the epilogue stores (line 294) to
kernel entry (bit-exact outputs: upentry/test_upentry.py part 1).  Must run before the CuTe warm-up compiles the skinny
kernels (plugin registration time).  Only for M <= K3UPENTRY_MAX_M (default 2, read at patch time; the 1-GPU chain
stand-in gains at M = 1, is neutral at 2 and loses ~0.2 us at 4 -- early spinning consumer CTAs slow the up-proj);
larger M keep vLLM's kernel (with K3OPT_PTRIGGER's edit).  The dynamic (M > 5) GEMM is untouched.
K3UPENTRY_DISABLE=1 at patch time: off.  Coexists with K3OPT_PTRIGGER=1 (which edits vLLM's installed file).

Safety argument (PDL: a dependent grid may launch once every CTA of its predecessor has executed launch_dependents or
exited; its griddepcontrol.wait returns only after the predecessor grid completed and its writes are visible).
Chain:  AR = allreduce_rmsnorm_reduce_scatter_early_exit (every CTA waits first; token CTAs trigger after their
multicast stores; the others exit after the wait)  ->  UP = skinny up-proj (griddepcontrol.wait before reading latent /
shared; the pre-wait B prefetch reads only the static weight)  ->  C = Lamport consumer (vLLM lamport_copy,
k3tail.lamport_attn_res or k3sgt.attnres_inproj: trigger at entry, no griddepcontrol.wait, readiness = mailbox words
!= 0x80000000).
 1. UP's own accesses are unchanged (all after its wait): RAW AR -> UP (latent, shared) is as today.
 2. The only change: C may launch once all UP CTAs have STARTED instead of after they finished their stores.  UP CTAs
    start only after every AR CTA triggered or exited, and each does so after its own griddepcontrol.wait, so C still
    launches after AR's predecessor grid completed.  Today's guarantee for every older kernel (e.g. the post-attention
    AttnRes writing prefix / blocks, read in C's prologue) already rests on AR's wait alone (AR's completion adds
    nothing about older grids), so it is unchanged.  New overlaps are with AR and UP only:
    (a) C's pre-poll reads (prefix, blocks, AttnRes weights, attnres_inproj's in_proj weight): not written by AR / UP.
    (b) C's mailbox reads: Lamport; UP never publishes 0x80000000 (sanitize_negative_zero).  The slots were re-armed by
        the previous layer's C, which completed before this C launched (trigger-after-wait kernels such as kda_split /
        k3oproj.produce lie between two tails); remote ranks write only after their AR completed, which needs this
        rank's AR contribution, which follows this rank's previous C.  As today.
    (c) C's writes (prefix, blocks, out, re-arm) vs AR / UP reads, incl. caching-allocator aliasing of C's fresh `out`
        with dead producer inputs: C writes token row t only after observing row t of this rank's shard, which every UP
        CTA of this rank writes (each owns 2..8 columns of all M rows) after its whole K loop (all latent reads) and
        after reading shared[t]; UP's first store follows its wait, i.e. AR completed.  So no C write can overtake an AR
        read or an UP read of latent / shared rows <= t; only UP's shared reads of rows > t may still be in flight, and
        shared_output is a persistent CollectiveKernel buffer (tailattn / moeblock also keep the tail inputs alive,
        _KEEPALIVE).
    (d) C and UP write disjoint data (UP: mailbox slots; C re-arms only slots it consumed).
 3. Progress: C launches only when all UP CTAs are resident, so C's spinning CTAs cannot starve UP; triggered AR CTAs
    are resident too.  Side streams are joined by events (full dependencies): unaffected.
Evidence (1 GPU, test_upentry.py part 2): chain restore -> AR stand-in -> UP -> consumer stand-in (junk-writes the AR
input and UP's latent after each row) -> marker, 16 x 10 graph iterations per M: 0 mismatches, mailbox re-armed.
"""

from __future__ import annotations

import os

_STATE = {"patched": False, "orig": None, "max_m": 0}


def patch_upentry() -> bool:
    """Install (idempotent). Returns True if patched."""
    if _STATE["patched"]:
        return True
    if os.environ.get("K3UPENTRY_DISABLE", "0") == "1":
        print("[k3opt] K3UPENTRY_DISABLE=1: vLLM's skinny up-projection (trigger at the end)", flush=True)
        return False
    import vllm.models.kimi_k3.nvidia.ops.cute_dsl.latent_moe_tail.fused_add_multicast_gemm as fam

    try:
        from . import upentry_skinny_patch as ue
    except ImportError:
        import upentry_skinny_patch as ue  # type: ignore
    orig = _STATE["orig"] = fam.FusedAddMulticastSkinnyGemmKernel
    max_m = int(os.environ.get("K3UPENTRY_MAX_M", "2"))
    _STATE["max_m"] = max_m

    def select(*, num_rows: int, **kwargs):
        """compile_skinny's constructor: the upentry kernel for M <= K3UPENTRY_MAX_M, vLLM's otherwise."""
        return (ue.FusedAddMulticastSkinnyGemmKernel if num_rows <= max_m else orig)(num_rows=num_rows, **kwargs)

    fam.FusedAddMulticastSkinnyGemmKernel = select
    _STATE["patched"] = True
    print(f"[k3opt] K3OPT_UPENTRY: latent-MoE tail skinny up-projection triggers its PDL dependents at entry "
          f"for M <= {max_m}", flush=True)
    return True
