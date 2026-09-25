"""Cut the per-step GPU work outside the decoder layers (Kimi K3 decode, TP16, vLLM V2 runner).

W1-6 ``stepov``.  Successor of k3opt/step_patch.py (it contains all of step_patch's features;
enable one or the other, not both -- if step_patch is already installed this patch restores its
originals first).  Call ``patch_stepov(load_ext)`` once per process at plugin time, before model
construction; ``load_ext()`` must build/load k3step.cu (torch.ops.k3step), and l2pf.cu if the
lm_head prefetch is wanted.

Switches (read at call time unless noted):
  K3STEPOV_DISABLE=1       nothing is patched (read at patch time)
  K3STEPOV_SAMPLE=0        no TP-distributed sampling (1.)
  K3STEPOV_FUSED_POST=0    distributed sampling without the fused state update (2.)
  K3STEPOV_FAST_GATHER=0   no one-hop logits all-gather for the other small batches
  K3STEPOV_PREP=0          no input-prep caching / fusion (3.)
  K3STEPOV_DECODE_PREP=0   keep vLLM's two position / input-id kernels (3.; the fused one is used
                           only if k3step_prep.cu is built, i.e. torch.ops.k3step.decode_prep exists)
  K3STEPOV_EMBED=0         no embedding broadcast (4.)  (checked at graph-capture time)
  K3STEPOV_LMHEAD_PF_MB=N  lm_head L2 prefetch size in MB (default 48; 0 = off) (5.)  (read at
                           l2pf table build time; _GRID / _CHUNK / _POLICY at graph capture)
  K3STEPOV_PREP_ATTN=0     keep vLLM's gather_block_tables + compute_slot_mappings kernels (3.;
                           default: folded into k3step.decode_prep_attn when k3step_prep.cu is built)
  K3STEPOV_FINAL_NORM=0    final RMSNorm stays in compute_logits (6.; read at model construction)
  K3STEPOV_L0FUSE=1        fuse the layer-0 MLP all-reduce into layer 1's AttnRes (7.; default OFF:
                           the fused k3ar kernel measured 10.5-12 us per call on 16 ranks, about what
                           flashinfer's all-reduce + vLLM's AttnRes take in the trace -- A/B only)
  K3STEPOV_STATE_ALIAS=0   KDA builders keep their own state-index buffers (8.; decided once,
                           before the first graph capture)

1. TP-distributed Gumbel-max sampling (as step_patch): each rank runs vLLM's own
   ``gumbel_noised_argmax`` over its [M, vocab/16] logits shard with global token ids as noise
   keys, and k3step.sample_finish exchanges one (value, id) packet per token and rank over an NVLS
   Lamport mailbox.  Bit-identical to vLLM's full-vocab path.  The index / length metadata of the
   V2 runner may be int32 or int64 and strided: the kernel reads them as they are (no conversion
   kernel).  Launched with PDL.
2. The same kernel also does vLLM's ``post_update`` (last_sampled_tokens, total_len,
   all_token_ids, output_bin_counts, num_computed_tokens) and the mamba-hybrid model state's
   num_accepted scatter for its rows, so ``postprocess_sampled`` skips both launches.  Only inside
   ``sample_tokens`` of a batch without speculative decoding / prompt logprobs (nothing between
   sampling and postprocess_sampled reads the request state then).
3. ``GPUModelRunner.prepare_inputs`` (source-patched, exact-text; falls back to the original if
   vLLM's text differs): the is_padding fills and the query_start_loc host->device copy are skipped
   when the buffer already holds exactly those values (tracked with the tensor version counter,
   which every writer -- make_dummy, prepare_inputs_to_capture, pcp -- bumps); idx_mapping reuses
   the previous step's device tensor when the host array is unchanged (nothing writes it in place
   without a speculator); cu_num_logits = arange and expanded_local_pos = zeros are cached per
   num_reqs; if k3step_prep.cu is built and there are no draft tokens, vLLM's
   _prepare_pos_seq_lens + _combine_sampled_and_draft_tokens kernels become one launch.  Disabled with a speculator / adaptive
   verification / PCP.  The inputs are bit-identical; every rank decides from replicated
   scheduler state.
4. ``KimiLinearModel.embed_input_ids`` for 1 <= M <= 16 tokens (TP > 1, unquantized bf16 shard,
   no added vocab): k3step.embed_bcast -- the owner rank multicasts the embedding row into every
   rank's mailbox -- replaces the masked gather + flashinfer one-shot all-reduce (two kernels and
   the ~10 us gap between them at the start of every step's graph).  Bit-identical (the all-reduce
   of one row and 15 zero rows returns that row, with -0.0 turned into +0.0; the kernel does the
   same).  CUDA-graph safe; the mailbox is allocated collectively on the first eager call.
5. lm_head L2 prefetch: extends l2pf_patch (K3OPT_L2PF) so that the last MoE layer, which had no
   prefetch, pulls the first K3STEPOV_LMHEAD_PF_MB MB of this rank's lm_head rows into L2 right
   after its FC2 (same side stream / join as l2pf); the CuTe skinny GEMM of the lm_head reads rows
   in block order, so its first third then comes from L2 (~2.5 us of 25 us measured on 1 GPU).
   Nothing changes numerically.

6. Final RMSNorm: applied at the end of KimiLinearModel.forward (inside the CUDA graph, before the
   l2pf join) and skipped in compute_logits; bit-identical (row-wise).
7. (opt-in, K3STEPOV_L0FUSE=1) Layer 0 (the dense MLP): its down_proj returns the TP partial and
   layer 1's pre-attention AttnRes reduces it in k3ar.ar_attn_res (NVLS Lamport all-reduce + AttnRes
   in one kernel), replacing flashinfer's one-shot all-reduce + vLLM's AttnRes kernel.  Not
   bit-identical (fp32 rank-order sum, like K3OPT_ARRES: 99.99% of outputs identical, worst rel.
   error 1.6e-3 on 16 ranks).
8. KDA metadata builders: their persistent state-index buffers alias the align context's aligned
   state indices, so the per-step staging copy (one device-to-device copy per KDA KV-cache group,
   3 at TP16) is a no-op view copy.  Identical memory contents.

Everything that decides between the fast and the original path depends only on state replicated on
all TP ranks (scheduler output, request sampling params, batch shape), so all ranks always take the
same path.  A rank that never publishes into a mailbox makes the others trap after 20 s.
"""

from __future__ import annotations

import functools
import inspect
import os
import sys
import textwrap
import weakref

import numpy as np
import torch

MAX_M = 16
NBUF = 2
BLOCK = 1024  # vLLM's gumbel_sample block (noise keys and tie-break are per global token id)
SENT_G = -0x80000000  # int32 view of the bf16-mailbox empty marker
SENT_A = -1           # int32 view of the argmax-packet empty marker

_STATE = {
    "patched": False,
    # sampling mailboxes
    "init": None,       # None: not tried; False: unavailable; True: ready
    "tp": 1,
    "rank": 0,
    "shard": 0,
    "gather_mb": None, "gather_mc": 0, "gather_call": 0,
    "arg_mb": None, "arg_mc": 0, "arg_call": 0,
    "mode": 0,          # 0: multicast (production); 1: local stores (single-GPU tests)
    "want_local": False,
    "got_local": False,
    "kernel": None,
    # fused post-update bookkeeping
    "in_sample_tokens": False,
    "fused": None,      # (sampled_token_ids tensor, scatter_fused) of the current step
    "skip_scatter_ns": None,
    "mamba_cls": None,
    "mamba_postprocess": None,
    # originals
    "orig": {},
    "stats": {"dist_sample": 0, "sample_fallback": 0, "fused_post": 0, "fast_gather": 0,
              "gather_fallback": 0, "embed_bcast": 0, "embed_fallback": 0},
    "reported": set(),
    "capture_seen": False,  # a CUDA-graph capture of the forward has started (set by the hooks)
}
_EMB = {"init": None, "mb": None, "mc": 0, "rank": 0, "tp": 1, "mode": 0}
_PREP = {"pad": None, "idx": None, "qsl": None, "meta": {}, "installed": False, "attn_ready": None,
         "stats": {"pad_hit": 0, "pad_miss": 0, "idx_hit": 0, "idx_miss": 0, "qsl_hit": 0,
                   "qsl_miss": 0, "meta_hit": 0, "meta_miss": 0, "decode_prep": 0,
                   "decode_prep_attn": 0}}
_LMH: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()  # KimiLinearModel -> lm_head ref


def _on(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) == "1"


def _report(key: str, msg: str) -> None:
    if key not in _STATE["reported"]:
        _STATE["reported"].add(key)
        print(f"[k3stepov] {msg}", flush=True)


def _capturing() -> bool:
    return torch.cuda.is_current_stream_capturing()


# ==========================================================================================
# 1./2. Sampling
# ==========================================================================================
def _get_kernel():
    k = _STATE["kernel"]
    if k is not None:
        return k
    from vllm.triton_utils import tl, triton
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax

    @triton.jit
    def _k3_local_gumbel_kernel(local_max_ptr, local_arg_ptr, nb, logits_ptr, logits_stride,
                                shard_off, shard, vocab_size, expanded_idx_mapping_ptr,
                                temp_ptr, seeds_ptr, positions_ptr, logits_indices_ptr,
                                BLOCK_SIZE: tl.constexpr):
        # = vLLM _gumbel_sample_kernel / gumbel_block_argmax for global vocab block
        #   (shard_off / BLOCK_SIZE + program_id(1)), reading this rank's logits shard.
        token_idx = tl.program_id(0).to(tl.int64)
        block_idx = tl.program_id(1)
        offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        keys = shard_off + offs  # global token ids (int32, like vLLM's `block`)
        mask = (offs < shard) & (keys < vocab_size)
        logits = tl.load(logits_ptr + token_idx * logits_stride + offs, mask=mask,
                         other=float("-inf"))
        logits = logits.to(tl.float32)
        req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx).to(tl.int64)
        is_valid_req = req_state_idx >= 0
        temp = tl.load(temp_ptr + req_state_idx, mask=is_valid_req, other=0.0).to(tl.float32)
        seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid_req, other=0)
        pos = tl.load(positions_ptr + tl.load(logits_indices_ptr + token_idx))
        value, idx = gumbel_noised_argmax(logits, keys, mask, seed, pos, temp, IS_DRAFTING=False,
                                          USE_FP64=False, APPLY_TEMPERATURE=False)
        tl.store(local_arg_ptr + token_idx * nb + block_idx,
                 (shard_off + block_idx * BLOCK_SIZE + idx).to(tl.int32))
        tl.store(local_max_ptr + token_idx * nb + block_idx, value)

    _STATE["kernel"] = _k3_local_gumbel_kernel
    return _k3_local_gumbel_kernel


def local_gumbel(local_logits, shard_off, vocab_size, expanded_idx_mapping, temperature, seeds,
                 positions, logits_indices):
    """Per-block (value, global id) Gumbel maxima of this rank's shard: f32 / i32 [M, NB]."""
    M, S = local_logits.shape
    nb = (S + BLOCK - 1) // BLOCK
    local_max = torch.empty((M, nb), dtype=torch.float32, device=local_logits.device)
    local_arg = torch.empty((M, nb), dtype=torch.int32, device=local_logits.device)
    _get_kernel()[(M, nb)](local_max, local_arg, nb, local_logits, local_logits.stride(0),
                           shard_off, S, vocab_size, expanded_idx_mapping, temperature, seeds,
                           positions, logits_indices, BLOCK_SIZE=BLOCK)
    return local_max, local_arg


def _init(shard: int) -> bool:
    """Collective (first small logits call): sampling mailboxes."""
    if _STATE["init"] is not None:
        return bool(_STATE["init"]) and shard == _STATE["shard"]
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem
    from vllm.distributed import get_tp_group

    tpg = get_tp_group()
    tp, rank = tpg.world_size, tpg.rank_in_group
    dev = torch.device("cuda", torch.cuda.current_device())
    ok = 1 < tp <= 16 and shard % 8 == 0
    gmb = amb = None
    gmc = amc = 0
    if ok:
        try:
            gmb = symm_mem.empty((NBUF, tp, MAX_M, shard), dtype=torch.bfloat16, device=dev)
            gmc = int(symm_mem.rendezvous(gmb, tpg.device_group.group_name).multicast_ptr or 0)
            amb = symm_mem.empty((NBUF, MAX_M, tp, 2), dtype=torch.int32, device=dev)
            amc = int(symm_mem.rendezvous(amb, tpg.device_group.group_name).multicast_ptr or 0)
        except Exception as e:  # noqa: BLE001
            print(f"[k3stepov] sampling mailbox setup failed: {e}", flush=True)
            gmc = amc = 0
    flag = torch.tensor([1 if (ok and gmc != 0 and amc != 0) else 0], dtype=torch.int32,
                        device=dev)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=tpg.device_group)
    if int(flag.item()) == 0:
        _STATE["init"] = False
        print("[k3stepov] NVLS multicast unavailable -> original sampling / all-gather",
              flush=True)
        return False
    gmb.view(torch.int32).fill_(SENT_G)
    amb.fill_(SENT_A)
    torch.cuda.synchronize()
    tpg.barrier()
    _STATE.update(init=True, tp=tp, rank=rank, shard=shard, gather_mb=gmb, gather_mc=gmc,
                  arg_mb=amb, arg_mc=amc, mode=0)
    print(f"[k3stepov] NVLS sampling mailboxes ready (tp={tp}, vocab shard={shard})", flush=True)
    return True


def install_local_mailboxes_for_tests(shard: int, world: int = 1, rank: int = 0):
    """Single-GPU tests: local sampling mailboxes, local stores (mode 1) into slot `rank`."""
    dev = torch.device("cuda", torch.cuda.current_device())
    gmb = torch.empty((NBUF, world, MAX_M, shard), dtype=torch.bfloat16, device=dev)
    gmb.view(torch.int32).fill_(SENT_G)
    amb = torch.full((NBUF, MAX_M, world, 2), SENT_A, dtype=torch.int32, device=dev)
    _STATE.update(init=True, tp=world, rank=rank, shard=shard, gather_mb=gmb, gather_mc=0,
                  arg_mb=amb, arg_mc=0, mode=1, gather_call=0, arg_call=0)
    return gmb, amb


def fast_allgather(local: torch.Tensor) -> torch.Tensor:
    """[M, S] bf16 -> [M, tp * S] bf16 via the NVLS mailbox."""
    M, S = local.shape
    out = torch.empty((M, _STATE["tp"] * S), dtype=local.dtype, device=local.device)
    buf = _STATE["gather_call"] % NBUF
    _STATE["gather_call"] += 1
    torch.ops.k3step.allgather(local, _STATE["gather_mb"], _STATE["gather_mc"], buf,
                               _STATE["rank"], _STATE["mode"], out)
    return out


def sample_finish(local_max, local_arg, idx_mapping, seq_lens, cu_num_logits, prefill_len,
                  post: dict | None = None):
    """Cross-rank argmax + num_sampled / num_rejected (+ fused post-update if `post`)."""
    M = local_max.shape[0]
    dev = local_max.device
    sampled = torch.empty(M, dtype=torch.int64, device=dev)
    num_sampled = torch.empty(M, dtype=torch.int32, device=dev)
    num_rejected = torch.empty(M, dtype=torch.int32, device=dev)
    buf = _STATE["arg_call"] % NBUF
    _STATE["arg_call"] += 1
    p = post or {}
    torch.ops.k3step.sample_finish(
        local_max, local_arg, _STATE["arg_mb"], _STATE["arg_mc"], buf, _STATE["rank"],
        _STATE["mode"], idx_mapping, seq_lens, cu_num_logits, prefill_len, sampled, num_sampled,
        num_rejected, p.get("query_start_loc"), p.get("num_computed_tokens"),
        p.get("last_sampled_tokens"), p.get("total_len"), p.get("all_token_ids"),
        p.get("output_bin_counts"), p.get("num_accepted"), None)
    return sampled, num_sampled, num_rejected


def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
    """LogitsProcessor._gather_logits (only reached for TP > 1)."""
    orig = _STATE["orig"]["gather"]
    if _capturing() or logits.dim() != 2:
        return orig(self, logits)
    want_local = _STATE["want_local"]
    small = (logits.dtype == torch.bfloat16 and 1 <= logits.shape[0] <= MAX_M
             and logits.stride(1) == 1)
    fast = small and _on("K3STEPOV_FAST_GATHER")
    if (want_local or fast) and not _init(logits.shape[1]):
        return orig(self, logits)
    if want_local:
        _STATE["got_local"] = True
        return logits
    if fast and logits.data_ptr() % 16 == 0 and (logits.stride(0) * 2) % 16 == 0:
        _STATE["stats"]["fast_gather"] += 1
        _report("gather", "one-hop NVLS logits all-gather active")
        return fast_allgather(logits)
    _STATE["stats"]["gather_fallback"] += 1
    return orig(self, logits)


def _dist_sample_ok(runner, input_batch, grammar_output) -> bool:
    from vllm.v1.worker.gpu.sample.sampler import Sampler

    if not _on("K3STEPOV_SAMPLE") or _STATE["init"] is False or _capturing():
        return False
    s = runner.sampler
    if (type(s) is not Sampler or grammar_output is not None
            or getattr(runner, "batch_sharder", None) is not None
            or getattr(runner, "pp_handler", None) is not None
            or getattr(runner, "pcp_manager", None) is not None
            or not getattr(runner, "is_last_pp_rank", True)):
        return False
    n = input_batch.num_reqs
    if not (1 <= n <= MAX_M) or input_batch.num_draft_tokens != 0:
        return False
    if s.compute_nans or s.return_sampling_mask or s.trace_replay_state is not None \
            or s.use_fp64_gumbel:
        return False
    idx_np = input_batch.idx_mapping_np
    if s.get_logprobs_dims(idx_np) is not None:
        return False
    if np.any(s.needs_logits_processing[idx_np]):
        return False
    return True


def _post_args(runner, input_batch) -> tuple[dict | None, bool]:
    """Targets of vLLM's post_update (+ mamba num_accepted) if they can be fused, else None."""
    if not _on("K3STEPOV_FUSED_POST") or not _STATE["in_sample_tokens"]:
        return None, False
    if getattr(runner, "speculator", None) is not None:
        return None, False
    if type(runner).postprocess_sampled is not _postprocess_sampled:
        return None, False
    pw = getattr(runner, "prompt_logprobs_worker", None)
    uses = getattr(pw, "uses_prompt_logprobs", None)
    if pw is None or uses is None or np.any(uses[input_batch.idx_mapping_np]):
        return None, False
    rs = runner.req_states
    nct = rs.num_computed_tokens.gpu
    lst = rs.last_sampled_tokens
    tl_ = rs.total_len.gpu
    ati = rs.all_token_ids.gpu
    obc = runner.sampler.penalties_state.output_bin_counts
    ok = (nct.dtype == torch.int32 and nct.is_contiguous() and tl_.dtype == torch.int32
          and tl_.is_contiguous() and lst.dtype == torch.int64 and lst.dim() == 2
          and ati.dtype == torch.int32 and ati.dim() == 2 and ati.stride(1) == 1
          and (obc is None or (obc.dtype == torch.int32 and obc.dim() == 2
                               and obc.stride(1) == 1)))
    if not ok:
        return None, False
    post = dict(query_start_loc=input_batch.query_start_loc, num_computed_tokens=nct,
                last_sampled_tokens=lst, total_len=tl_, all_token_ids=ati,
                output_bin_counts=obc, num_accepted=None)
    ms = runner.model_state
    fuse_scatter = False
    na = getattr(ms, "num_accepted_tokens_gpu", None)
    if (na is not None and _STATE["mamba_cls"] is not None
            and type(ms).postprocess_state is _STATE["mamba_postprocess"]
            and na.dtype == torch.int32 and na.is_contiguous()):
        post["num_accepted"] = na
        fuse_scatter = True
    return post, fuse_scatter


def _sample(self, hidden_states, input_batch, grammar_output):
    """GPUModelRunner.sample with TP-distributed Gumbel-max for plain sampling batches."""
    orig = _STATE["orig"]["sample"]
    _STATE["fused"] = None
    if not _dist_sample_ok(self, input_batch, grammar_output):
        _STATE["stats"]["sample_fallback"] += 1
        return orig(self, hidden_states, input_batch, grammar_output)
    n = input_batch.num_reqs
    if input_batch.num_tokens == n:  # one token per request: logits_indices == arange(n)
        sample_hidden_states = hidden_states[:n]
    else:
        sample_hidden_states = hidden_states[input_batch.logits_indices]
    _STATE["want_local"], _STATE["got_local"] = True, False
    try:
        logits = self.model.compute_logits(sample_hidden_states)
    finally:
        _STATE["want_local"] = False
    if not _STATE["got_local"]:
        _STATE["stats"]["sample_fallback"] += 1
        so = self.sampler(logits, input_batch)
        return so, so.num_sampled, so.num_rejected
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    states = self.sampler.sampling_states
    local_max, local_arg = local_gumbel(
        logits, _STATE["rank"] * logits.shape[1], self.vocab_size,
        input_batch.expanded_idx_mapping, states.temperature.gpu, states.seeds.gpu,
        input_batch.positions, input_batch.logits_indices)
    post, fuse_scatter = _post_args(self, input_batch)
    sampled, num_sampled, num_rejected = sample_finish(
        local_max, local_arg, input_batch.idx_mapping, input_batch.seq_lens,
        input_batch.cu_num_logits, self.req_states.prefill_len.gpu, post)
    sampled_2d = sampled.view(-1, 1)
    if post is not None:
        _STATE["fused"] = (sampled_2d, fuse_scatter)
        _STATE["stats"]["fused_post"] += 1
        _report("post", "post_update" + (" + num_accepted scatter" if fuse_scatter else "")
                + " fused into k3step.sample_finish")
    _STATE["stats"]["dist_sample"] += 1
    _report("sample", f"TP-distributed Gumbel-max sampling active (M={n})")
    so = SamplerOutput(sampled_token_ids=sampled_2d, logprobs_tensors=None, num_nans=None,
                       num_sampled=num_sampled, num_rejected=num_rejected)
    return so, num_sampled, num_rejected


def _postprocess_sampled(self, idx_mapping, sampled_tokens, num_sampled, num_rejected,
                         query_start_loc=None):
    fused = _STATE["fused"]
    _STATE["fused"] = None
    if fused is None or fused[0] is not sampled_tokens:
        return _STATE["orig"]["postprocess_sampled"](self, idx_mapping, sampled_tokens,
                                                     num_sampled, num_rejected, query_start_loc)
    # post_update already done by k3step.sample_finish; the model state's own postprocess still
    # runs (recoverssm commit, mamba align), minus the num_accepted scatter if it was fused.
    _STATE["skip_scatter_ns"] = num_sampled if fused[1] else None
    try:
        self.model_state.postprocess_state(idx_mapping, num_sampled,
                                           self.req_states.num_computed_tokens.gpu)
    finally:
        _STATE["skip_scatter_ns"] = None


def _sample_tokens(self, grammar_output):
    _STATE["in_sample_tokens"] = True
    try:
        return _STATE["orig"]["sample_tokens"](self, grammar_output)
    finally:
        _STATE["in_sample_tokens"] = False
        _STATE["fused"] = None


class _ScatterProxy:
    """Stands in for mamba_hybrid._scatter_num_accepted_kernel: skips the launch whose
    num_sampled tensor was already scattered by k3step.sample_finish."""

    def __init__(self, kernel):
        self._k = kernel

    def __getitem__(self, grid):
        launch = self._k[grid]

        def run(*args, **kwargs):
            ns = _STATE["skip_scatter_ns"]
            if ns is not None and len(args) >= 2 and args[1] is ns:
                return None
            return launch(*args, **kwargs)

        return run

    def __getattr__(self, name):
        return getattr(self._k, name)


def _install_sampling() -> None:
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    sp = sys.modules.get("step_patch")
    if sp is not None and getattr(sp, "_STATE", {}).get("patched"):
        # step_patch (K3OPT_STEP) is a subset of this patch: undo it.
        LogitsProcessor._gather_logits = sp._STATE["orig_gather"]
        GPUModelRunner.sample = sp._STATE["orig_sample"]
        sp._STATE["patched"] = False
        print("[k3stepov] step_patch was installed; replaced by stepov_patch", flush=True)
    o = _STATE["orig"]
    o["gather"] = LogitsProcessor._gather_logits
    o["sample"] = GPUModelRunner.sample
    o["postprocess_sampled"] = GPUModelRunner.postprocess_sampled
    o["sample_tokens"] = GPUModelRunner.sample_tokens
    LogitsProcessor._gather_logits = _gather_logits
    GPUModelRunner.sample = _sample
    GPUModelRunner.postprocess_sampled = _postprocess_sampled
    GPUModelRunner.sample_tokens = _sample_tokens
    try:
        from vllm.v1.worker.gpu.model_states import mamba_hybrid as mh

        cls = getattr(mh, "MambaHybridModelState", None)
        if cls is not None and "postprocess_state" not in vars(cls):
            cls = None
        k = getattr(mh, "_scatter_num_accepted_kernel", None)
        if cls is not None and k is not None:
            if not isinstance(k, _ScatterProxy):
                mh._scatter_num_accepted_kernel = _ScatterProxy(k)
            _STATE["mamba_cls"] = cls
            _STATE["mamba_postprocess"] = vars(cls)["postprocess_state"]
    except Exception as e:  # noqa: BLE001
        print(f"[k3stepov] mamba num_accepted fusion unavailable ({e!r})", flush=True)


# ==========================================================================================
# 3. prepare_inputs caching
# ==========================================================================================
def _prep_ok(runner) -> bool:
    return (_on("K3STEPOV_PREP") and getattr(runner, "speculator", None) is None
            and getattr(runner, "adaptive_verification", None) is None
            and getattr(runner, "pcp_manager", None) is None and not _capturing())


def _version(t: torch.Tensor):
    try:
        return t._version
    except Exception:  # noqa: BLE001  (inference tensors have no version counter)
        return None


def _k3sov_padding(runner, buf, num_tokens, num_tokens_after_padding):
    ok = _prep_ok(runner)
    key = (buf.data_ptr(), int(num_tokens), int(num_tokens_after_padding))
    if ok:
        c = _PREP["pad"]
        v = _version(buf)
        if v is not None and c is not None and c[0] == key and c[1] == v:
            _PREP["stats"]["pad_hit"] += 1
            return
    buf[:num_tokens].fill_(False)
    buf[num_tokens:num_tokens_after_padding].fill_(True)
    _PREP["pad"] = (key, _version(buf)) if ok else None
    _PREP["stats"]["pad_miss"] += 1


def _k3sov_idx_mapping(runner, idx_mapping_np):
    from vllm.utils.torch_utils import async_tensor_h2d

    ok = _prep_ok(runner)
    if ok:
        c = _PREP["idx"]
        if (c is not None and c[2] == runner.device and c[0].shape == idx_mapping_np.shape
                and c[0].dtype == idx_mapping_np.dtype and np.array_equal(c[0], idx_mapping_np)):
            _PREP["stats"]["idx_hit"] += 1
            return c[1]
    t = async_tensor_h2d(idx_mapping_np, device=runner.device)
    _PREP["idx"] = (np.array(idx_mapping_np, copy=True), t, runner.device) if ok else None
    _PREP["stats"]["idx_miss"] += 1
    return t


def _k3sov_logits_meta(runner, num_reqs):
    ok = _prep_ok(runner)
    if ok:
        c = _PREP["meta"].get((num_reqs, str(runner.device)))
        if c is not None and _version(c[0]) == c[2] and _version(c[1]) == c[3]:
            _PREP["stats"]["meta_hit"] += 1
            return c[0], c[1]
    with torch.inference_mode(False):  # normal tensors: version counters detect writers
        cu = torch.arange(num_reqs + 1, device=runner.device, dtype=torch.int32)
        zl = torch.zeros(num_reqs, dtype=torch.int32, device=runner.device)
    if ok:
        _PREP["meta"][(num_reqs, str(runner.device))] = (cu, zl, _version(cu), _version(zl))
    _PREP["stats"]["meta_miss"] += 1
    return cu, zl


def _k3sov_qsl(runner, query_start_loc_np, buf):
    from vllm.utils.torch_utils import async_tensor_h2d

    ok = _prep_ok(runner)
    if ok:
        c = _PREP["qsl"]
        v = _version(buf)
        if v is None:
            _report("prep_inf", "input buffers are inference tensors: prepare_inputs caching inactive")
        if (v is not None and c is not None and c[1] == buf.data_ptr() and c[2] == v
                and c[0].shape == query_start_loc_np.shape
                and np.array_equal(c[0], query_start_loc_np)):
            _PREP["stats"]["qsl_hit"] += 1
            return
    async_tensor_h2d(query_start_loc_np, out=buf)
    _PREP["qsl"] = ((np.array(query_start_loc_np, copy=True), buf.data_ptr(), _version(buf))
                    if ok else None)
    _PREP["stats"]["qsl_miss"] += 1


def _bt_layout_ok(runner, num_tokens_after_padding) -> bool:
    """BlockTables of this runner can be served by k3step.decode_prep_attn."""
    bt = getattr(runner, "block_tables", None)
    if bt is None:
        return False
    ok = getattr(bt, "_k3sov_ok", None)
    if ok is None:
        try:
            g = bt.num_kv_cache_groups
            ok = (1 <= g <= 8 and bt.cp_size == 1 and bt.slot_mappings.dtype == torch.int64
                  and bt.slot_mappings.dim() == 2 and bt.slot_mappings.stride(1) == 1
                  and bt.num_blocks.gpu.dtype == torch.int32 and bt.num_blocks.gpu.stride(1) == 1
                  and all(t.dtype == torch.int32 and t.gpu.stride(1) == 1 for t in bt.block_tables)
                  and all(t.dtype == torch.int32 and t.stride(1) == 1 for t in bt.input_block_tables)
                  and all(a.gpu.stride(0) == b.stride(0) for a, b in zip(bt.block_tables,
                                                                         bt.input_block_tables)))
        except Exception:  # noqa: BLE001
            ok = False
        bt._k3sov_ok = ok
    return bool(ok) and num_tokens_after_padding <= bt.slot_mappings.shape[1]


def _k3sov_decode_prep(runner, idx_mapping, query_start_loc, cu_num_logits, total_num_logits,
                       total_num_draft_tokens, num_reqs_padded=None, num_tokens_after_padding=None):
    """Without draft tokens: k3step.decode_prep_attn (= prepare_pos_seq_lens +
    combine_sampled_and_draft_tokens + BlockTables.gather_block_tables + compute_slot_mappings; the
    following prepare_attn then launches nothing) or k3step.decode_prep (the first two only).
    Returns logits_indices, or None to run vLLM's kernels."""
    _PREP["attn_ready"] = None
    if not (_on("K3STEPOV_DECODE_PREP") and hasattr(torch.ops.k3step, "decode_prep")
            and _prep_ok(runner) and total_num_draft_tokens == 0
            and getattr(runner.model_state, "num_new_sampled_tokens_per_step", None) == 1
            and total_num_logits == idx_mapping.shape[0]):
        return None
    rs, ib = runner.req_states, runner.input_buffers
    li = torch.empty(total_num_logits, dtype=torch.int64, device=idx_mapping.device)
    if (_on("K3STEPOV_PREP_ATTN") and hasattr(torch.ops.k3step, "decode_prep_attn")
            and num_reqs_padded is not None and num_tokens_after_padding is not None
            and num_reqs_padded >= idx_mapping.shape[0] and num_reqs_padded <= ib.seq_lens.shape[0]
            and _bt_layout_ok(runner, num_tokens_after_padding)):
        from vllm.v1.attention.backends.utils import PAD_SLOT_ID

        bt = runner.block_tables
        torch.ops.k3step.decode_prep_attn(
            idx_mapping, query_start_loc, rs.num_computed_tokens.gpu, rs.last_sampled_tokens,
            rs.prefill_len.gpu, cu_num_logits, ib.positions, ib.seq_lens, ib.input_ids, li,
            int(num_reqs_padded), bt.block_table_ptrs, bt.input_block_table_ptrs,
            bt.block_table_strides, bt.num_blocks.gpu, bt.kernel_block_sizes_tensor,
            bt.slot_mapping_enabled, bt.slot_mappings, int(PAD_SLOT_ID))
        _PREP["attn_ready"] = (idx_mapping, int(num_reqs_padded), int(num_tokens_after_padding))
        _PREP["stats"]["decode_prep_attn"] += 1
        return li
    torch.ops.k3step.decode_prep(idx_mapping, query_start_loc, rs.num_computed_tokens.gpu,
                                 rs.last_sampled_tokens, rs.prefill_len.gpu, cu_num_logits,
                                 ib.positions, ib.seq_lens, ib.input_ids, li)
    _PREP["stats"]["decode_prep"] += 1
    return li


def _prepare_attn(self, input_batch):
    """GPUModelRunner.prepare_attn: block tables and slot mappings were already produced by
    k3step.decode_prep_attn for exactly this batch -> return vLLM's views without launching."""
    r = _PREP.get("attn_ready")
    _PREP["attn_ready"] = None
    if (r is not None and r[0] is input_batch.idx_mapping
            and r[1] == input_batch.num_reqs_after_padding
            and r[2] == input_batch.num_tokens_after_padding
            and getattr(self, "pcp_manager", None) is None):
        bt = self.block_tables
        return (tuple(t[: r[1]] for t in bt.input_block_tables),
                bt.slot_mappings[:, : r[2]])
    return _STATE["orig"]["prepare_attn"](self, input_batch)


# (old lines, new lines): matched line by line ignoring indentation; the replacement takes the
# indentation of the first matched line.
_PREP_EDITS = [
    (["is_padding[:num_tokens].fill_(False)",
      "is_padding[num_tokens:num_tokens_after_padding].fill_(True)"],
     ["_k3sov_padding(self, is_padding, num_tokens, num_tokens_after_padding)"]),
    (["idx_mapping = async_tensor_h2d(idx_mapping_np, device=self.device)"],
     ["idx_mapping = _k3sov_idx_mapping(self, idx_mapping_np)"]),
    (["cu_num_logits = torch.arange(",
      "num_reqs + 1, device=self.device, dtype=torch.int32",
      ")",
      "expanded_idx_mapping = idx_mapping",
      "expanded_local_pos = torch.zeros(",
      "num_reqs, dtype=torch.int32, device=self.device",
      ")"],
     ["cu_num_logits, expanded_local_pos = _k3sov_logits_meta(self, num_reqs)",
      "expanded_idx_mapping = idx_mapping"]),
    (["async_tensor_h2d(query_start_loc_np, out=query_start_loc)"],
     ["_k3sov_qsl(self, query_start_loc_np, query_start_loc)"]),
    (["prepare_pos_seq_lens("],
     ["_k3sov_li = _k3sov_decode_prep(self, idx_mapping, query_start_loc, cu_num_logits, "
      "total_num_logits, total_num_draft_tokens, num_reqs_padded, num_tokens_after_padding)",
      "if _k3sov_li is None: prepare_pos_seq_lens("]),
    (["logits_indices = combine_sampled_and_draft_tokens("],
     ["logits_indices = _k3sov_li if _k3sov_li is not None else combine_sampled_and_draft_tokens("]),
]


def _replace_lines(src: str, old: list[str], new: list[str]) -> tuple[str, bool]:
    lines = src.split("\n")
    n = len(old)
    hits = [i for i in range(len(lines) - n + 1)
            if all(lines[i + j].strip() == old[j] for j in range(n))]
    if len(hits) != 1:
        return src, False
    i = hits[0]
    indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    lines[i:i + n] = [indent + x for x in new]
    return "\n".join(lines), True


def transform_prepare_inputs_source(src: str) -> tuple[str | None, list[int]]:
    """Apply _PREP_EDITS to the (dedented) source; returns (new source or None, missing edits)."""
    missing = []
    for k, (old, new) in enumerate(_PREP_EDITS):
        src, ok = _replace_lines(src, old, new)
        if not ok:
            missing.append(k)
    return (None if missing else src), missing


def _install_prepare_inputs() -> bool:
    from vllm.v1.worker.gpu import model_runner as mr

    fn = mr.GPUModelRunner.prepare_inputs
    if getattr(fn, "_k3stepov", False):
        return True
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except Exception as e:  # noqa: BLE001
        print(f"[k3stepov] prepare_inputs source unavailable ({e!r}); not patched", flush=True)
        return False
    new, missing = transform_prepare_inputs_source(src)
    if new is None:
        print(f"[k3stepov] prepare_inputs text differs from the expected vLLM version (edits "
              f"{missing} not found); input-prep caching off", flush=True)
        return False
    g = mr.__dict__
    g.update(_k3sov_padding=_k3sov_padding, _k3sov_idx_mapping=_k3sov_idx_mapping,
             _k3sov_logits_meta=_k3sov_logits_meta, _k3sov_qsl=_k3sov_qsl,
             _k3sov_decode_prep=_k3sov_decode_prep)
    ns: dict = {}
    exec(compile(new, "<k3stepov:GPUModelRunner.prepare_inputs>", "exec"), g, ns)  # noqa: S102
    newfn = ns[fn.__name__]
    newfn._k3stepov = True
    newfn.__qualname__ = fn.__qualname__
    newfn.__doc__ = fn.__doc__
    _STATE["orig"]["prepare_inputs"] = fn
    mr.GPUModelRunner.prepare_inputs = newfn
    _STATE["orig"]["prepare_attn"] = mr.GPUModelRunner.prepare_attn
    mr.GPUModelRunner.prepare_attn = _prepare_attn
    _PREP["installed"] = True
    return True


# ==========================================================================================
# 4. Embedding broadcast
# ==========================================================================================
def _emb_init(et) -> bool:
    """Collective (first eager embed call on every rank): the [MAX_M, H] mailbox."""
    if _EMB["init"] is not None:
        return bool(_EMB["init"])
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem
    from vllm.distributed import get_tp_group

    tpg = get_tp_group()
    tp, rank = tpg.world_size, tpg.rank_in_group
    dev = et.weight.device
    H = et.weight.shape[1]
    ok = (1 < tp <= 16 and et.tp_size == tp and et.tp_rank == rank and H % 8 == 0
          and hasattr(torch.ops, "k3step") and hasattr(torch.ops.k3step, "embed_bcast"))
    mb, mc = None, 0
    if ok:
        try:
            mb = symm_mem.empty((MAX_M, H), dtype=torch.bfloat16, device=dev)
            mc = int(symm_mem.rendezvous(mb, tpg.device_group.group_name).multicast_ptr or 0)
        except Exception as e:  # noqa: BLE001
            print(f"[k3stepov] embedding mailbox setup failed: {e}", flush=True)
            mc = 0
    flag = torch.tensor([1 if (ok and mc != 0) else 0], dtype=torch.int32, device=dev)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=tpg.device_group)
    if int(flag.item()) == 0:
        _EMB["init"] = False
        print("[k3stepov] embedding broadcast unavailable -> original embedding", flush=True)
        return False
    mb.view(torch.int32).fill_(SENT_G)
    torch.cuda.synchronize()
    tpg.barrier()
    _EMB.update(init=True, mb=mb, mc=mc, rank=rank, tp=tp, mode=0)
    print(f"[k3stepov] NVLS embedding broadcast ready (tp={tp}, H={H})", flush=True)
    return True


def install_local_embed_mailbox_for_tests(hidden: int, rank: int = 0):
    dev = torch.device("cuda", torch.cuda.current_device())
    mb = torch.empty((MAX_M, hidden), dtype=torch.bfloat16, device=dev)
    mb.view(torch.int32).fill_(SENT_G)
    _EMB.update(init=True, mb=mb, mc=0, rank=rank, tp=0, mode=1)
    return mb


def _embed_static_ok(et) -> bool:
    ok = getattr(et, "_k3sov_embed_ok", None)
    if ok is not None:
        return ok
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

    si = getattr(et, "shard_indices", None)
    w = getattr(et, "weight", None)
    ok = (type(et) is VocabParallelEmbedding and getattr(et, "tp_size", 1) > 1
          and bool(getattr(et, "use_fused_embedding", False))
          and getattr(et, "parallel_group", None) is None
          and isinstance(w, torch.Tensor) and w.is_cuda and w.dtype == torch.bfloat16
          and w.dim() == 2 and w.stride(1) == 1 and (w.stride(0) * 2) % 16 == 0
          and w.data_ptr() % 16 == 0 and w.shape[1] % 8 == 0
          and si is not None and getattr(et, "num_added_embeddings_per_partition", 1) == 0
          and si.org_vocab_start_index == et.tp_rank * w.shape[0]
          and si.org_vocab_end_index == min((et.tp_rank + 1) * w.shape[0], et.org_vocab_size))
    et._k3sov_embed_ok = ok
    return ok


def _embed_input_ids(self, input_ids):
    orig = _STATE["orig"]["embed_input_ids"]
    if _capturing():
        _STATE["capture_seen"] = True
    et = getattr(self, "embed_tokens", None)
    if (not _on("K3STEPOV_EMBED") or et is None or not isinstance(input_ids, torch.Tensor)
            or input_ids.dim() != 1 or not (1 <= input_ids.shape[0] <= MAX_M)
            or input_ids.dtype not in (torch.int32, torch.int64) or not input_ids.is_cuda
            or torch.compiler.is_compiling() or not _embed_static_ok(et)):
        _STATE["stats"]["embed_fallback"] += 1
        return orig(self, input_ids)
    if _EMB["init"] is None:
        if _capturing():
            _STATE["stats"]["embed_fallback"] += 1
            return orig(self, input_ids)
        _emb_init(et)
    if not _EMB["init"]:
        return orig(self, input_ids)
    w = et.weight
    out = torch.empty((input_ids.shape[0], w.shape[1]), dtype=w.dtype, device=w.device)
    torch.ops.k3step.embed_bcast(input_ids, w, et.org_vocab_size, _EMB["rank"], _EMB["mb"],
                                 _EMB["mc"], _EMB["mode"], out)
    _STATE["stats"]["embed_bcast"] += 1
    _report("embed", "NVLS embedding broadcast active (M<=16)")
    return out


def _install_embed() -> None:
    from vllm.models.kimi_k3.nvidia import model as k3_model

    Model = k3_model.KimiLinearModel
    _STATE["orig"]["embed_input_ids"] = Model.embed_input_ids
    Model.embed_input_ids = _embed_input_ids


# ==========================================================================================
# 5. lm_head L2 prefetch (extends l2pf_patch)
# ==========================================================================================
_LMPF: dict = {}  # id(last MoE runner) -> (ranges tensor, bytes, weakref(runner))


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _lmhead_pf_bytes() -> int:
    return max(0, _env_int("K3STEPOV_LMHEAD_PF_MB", 48)) * 2**20


def _install_lmhead_prefetch() -> None:
    """Measured on 1 GPU (bench_lmhead_pf.py): prefetching the first 45-60 MB of the 147 MB lm_head
    shard saves ~2.5 us of its 25 us; the plain prefetch at grid 64 issues 48 MB in ~10.5 us,
    which fits the ~10.8 us between the last FC2 and the end of the model forward (where l2pf
    joins its side stream).  Knobs: K3STEPOV_LMHEAD_PF_MB (48; 0 = off), _GRID (64), _CHUNK
    (16384), _POLICY (0; 4 = TMA fill: ~2 us more gain at 60 MB but ~16 us long)."""
    from vllm.models.kimi_k3.nvidia import model as k3_model

    CausalLM = k3_model.KimiLinearForCausalLM
    if "causal_init" not in _STATE["orig"]:
        orig_init = CausalLM.__init__

        # functools.wraps: vLLM's initialize_model() inspects the constructor signature to decide
        # whether to pass vllm_config; a bare (*args, **kwargs) wrapper hides it.
        @functools.wraps(orig_init)
        def __init__(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            lm = getattr(self, "lm_head", None)
            if isinstance(getattr(lm, "weight", None), torch.Tensor):
                _LMH[self.model] = weakref.ref(lm)

        CausalLM.__init__ = __init__
        _STATE["orig"]["causal_init"] = orig_init

    pf = sys.modules.get("l2pf_patch")
    if pf is None or not pf._STATE.get("patched"):
        print("[k3stepov] l2pf_patch not installed (K3OPT_L2PF off): no lm_head prefetch",
              flush=True)
        return
    if getattr(pf.build_tables, "_k3stepov", False):
        return
    orig_build = pf.build_tables
    orig_launch = pf.launch_after_moe

    def build_tables(model=None) -> int:
        total = orig_build(model)
        _LMPF.clear()
        nbytes = _lmhead_pf_bytes()
        if nbytes <= 0:
            return total
        for m in ([model] if model is not None else list(pf._MODELS)):
            ref = _LMH.get(m)
            lm = ref() if ref is not None else None
            w = getattr(lm, "weight", None)
            if not isinstance(w, torch.Tensor) or not w.is_cuda or w.dim() != 2 \
                    or w.stride(1) != 1:
                continue
            start, end = getattr(m, "start_layer", 0), getattr(m, "end_layer", len(m.layers))
            runner = pf._runner_of(m.layers[end - 1]) if end > start else None
            if runner is None:
                continue  # last layer is not an MoE layer: nothing forks right before the tail
            row_bytes = w.stride(0) * w.element_size()
            rows = min(w.shape[0], max(1, nbytes // row_bytes))
            span = (int(w.data_ptr()), int(rows * row_bytes))
            rng = torch.tensor([span], dtype=torch.int64).to(w.device)
            _LMPF[id(runner)] = (rng, span[1], weakref.ref(runner))
            print(f"[k3stepov] lm_head L2 prefetch: the last MoE layer pulls lm_head rows 0..{rows} "
                  f"of {w.shape[0]} ({span[1] / 2**20:.0f} MB) after its FC2", flush=True)
        return total

    def launch_after_moe(runner, num_tokens: int) -> bool:
        if not pf._STATE["built"] and not _capturing():
            pf.build_tables()
        ent = _LMPF.get(id(runner))
        if ent is None or ent[2]() is not runner:
            return orig_launch(runner, num_tokens)
        if num_tokens <= 0 or num_tokens > pf.MAX_TOKENS or torch.compiler.is_compiling():
            return False
        if _capturing() and pf._breakable_active():
            return False
        rng = ent[0]
        cur = torch.cuda.current_stream()
        ev = pf._STATE["events"].get(id(runner))
        if ev is None:
            ev = torch.cuda.Event()
            pf._STATE["events"][id(runner)] = ev
        ev.record(cur)
        s = pf._stream(rng.device)
        s.wait_event(ev)
        pf._STATE["pending"] = True  # joined at the end of KimiLinearModel.forward by l2pf
        with torch.cuda.stream(s):
            torch.ops.k3pf.prefetch_l2_cfg(rng, 1, _env_int("K3STEPOV_LMHEAD_PF_GRID", 64),
                                           _env_int("K3STEPOV_LMHEAD_PF_CHUNK", 16384),
                                           _env_int("K3STEPOV_LMHEAD_PF_POLICY", 0), 1)
        pf._STATE["stats"]["launched"] += 1
        return True

    build_tables._k3stepov = True
    pf.build_tables = build_tables
    pf.launch_after_moe = launch_after_moe
    pf._STATE["built"] = False


# ==========================================================================================
# 6. Final RMSNorm inside the model graph
# ==========================================================================================
def _install_final_norm() -> None:
    """KimiLinearModel.forward returns model.norm(hidden_states) and KimiLinearForCausalLM.compute_logits
    skips its own model.norm -- the same kernel on the same rows (row-wise, so applying it before
    the logits-row selection changes nothing: bit-identical), but inside the CUDA graph (and inside
    the lm_head prefetch window) instead of as an eager launch.  Only when the pre-norm hidden states
    have no other consumer: no speculative decoding (MTP drafts read them), no aux hidden-state
    layers, last PP rank.  Decided once per model at construction (K3STEPOV_FINAL_NORM=0: off)."""
    from vllm.models.kimi_k3.nvidia import model as k3_model

    CausalLM, Model = k3_model.KimiLinearForCausalLM, k3_model.KimiLinearModel
    if "final_norm_init" in _STATE["orig"]:
        return
    orig_init = CausalLM.__init__

    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        m = getattr(self, "model", None)
        try:
            from vllm.distributed import get_pp_group

            vc = getattr(self, "vllm_config", None)
            ok = (_on("K3STEPOV_FINAL_NORM") and m is not None and vc is not None
                  and getattr(vc, "speculative_config", None) is None
                  and get_pp_group().is_last_rank
                  and not getattr(m, "aux_hidden_state_layers", ())
                  and isinstance(getattr(m, "norm", None), torch.nn.Module))
        except Exception:  # noqa: BLE001
            ok = False
        if m is not None:
            m._k3sov_final_norm = bool(ok)
        if ok:
            _report("final_norm", "final RMSNorm moved into the model forward (CUDA graph)")

    CausalLM.__init__ = __init__
    _STATE["orig"]["final_norm_init"] = orig_init

    orig_logits = CausalLM.compute_logits

    @functools.wraps(orig_logits)
    def compute_logits(self, hidden_states, *args, **kwargs):
        if getattr(getattr(self, "model", None), "_k3sov_final_norm", False):
            return self.logits_processor(self.lm_head, hidden_states)
        return orig_logits(self, hidden_states, *args, **kwargs)

    CausalLM.compute_logits = compute_logits
    _STATE["orig"]["compute_logits"] = orig_logits

    pf = sys.modules.get("l2pf_patch")
    if pf is not None and pf._STATE.get("patched") and pf._STATE.get("orig_model_forward"):
        # inside l2pf's wrapper, i.e. before its side-stream join: the norm runs in the prefetch
        # window
        pf._STATE["orig_model_forward"] = _final_norm_wrap(pf._STATE["orig_model_forward"])
    else:
        Model.forward = _final_norm_wrap(Model.forward)


def _final_norm_wrap(inner):
    @functools.wraps(inner)
    def forward(self, *args, **kwargs):
        out = inner(self, *args, **kwargs)
        if getattr(self, "_k3sov_final_norm", False) and isinstance(out, torch.Tensor):
            out = self.norm(out, None)
        return out

    return forward


# ==========================================================================================
# 7. Layer 0 (dense MLP) -> layer 1: all-reduce fused into layer 1's pre-attention AttnRes
# ==========================================================================================
_L0 = {"init": None, "mb": None, "mc": 0, "rank": 0, "tp": 1, "pending": None,
       "stats": {"fused": 0, "fallback": 0}}


def _l0_init(dev) -> bool:
    """Collective (first eager eligible call on every rank): k3ar mailbox [2, tp, 16, 7168]."""
    if _L0["init"] is not None:
        return bool(_L0["init"])
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem
    from vllm.distributed import get_tp_group

    tpg = get_tp_group()
    tp, rank = tpg.world_size, tpg.rank_in_group
    ok = 1 < tp <= 16 and hasattr(torch.ops, "k3ar") and hasattr(torch.ops.k3ar, "ar_attn_res")
    mb, mc = None, 0
    if ok:
        try:
            mb = symm_mem.empty((2, tp, MAX_M, 7168), dtype=torch.bfloat16, device=dev)
            mc = int(symm_mem.rendezvous(mb, tpg.device_group.group_name).multicast_ptr or 0)
        except Exception as e:  # noqa: BLE001
            print(f"[k3stepov] layer-0 mailbox setup failed: {e}", flush=True)
            mc = 0
    flag = torch.tensor([1 if (ok and mc != 0) else 0], dtype=torch.int32, device=dev)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=tpg.device_group)
    if int(flag.item()) == 0:
        _L0["init"] = False
        print("[k3stepov] layer-0 AR+AttnRes fusion unavailable -> original path", flush=True)
        return False
    mb.view(torch.int16).fill_(-32768)  # bf16 -0.0 Lamport sentinels (k3ar protocol)
    torch.cuda.synchronize()
    tpg.barrier()
    _L0.update(init=True, mb=mb, mc=mc, rank=rank, tp=tp)
    print(f"[k3stepov] layer-0 AR+AttnRes fusion ready (tp={tp})", flush=True)
    return True


def install_local_l0_for_tests(mb, mc, rank, tp):
    _L0.update(init=True, mb=mb, mc=mc, rank=rank, tp=tp, pending=None)


def _l0_mlp_forward(self, x):
    """KimiMLP.forward of the flagged dense layer: return the down_proj partial (no all-reduce);
    the next layer's _pre_attn_norm reduces it inside k3ar.ar_attn_res."""
    orig = _STATE["orig"]["mlp_forward"]
    _L0["pending"] = None
    if not (getattr(self, "_k3sov_l0", False) and _on("K3STEPOV_L0FUSE", "0")
            and isinstance(x, torch.Tensor) and x.dim() == 2 and 1 <= x.shape[0] <= MAX_M
            and x.dtype == torch.bfloat16 and not torch.compiler.is_compiling()):
        return orig(self, x)
    if _L0["init"] is None:
        if _capturing():
            return orig(self, x)
        _l0_init(x.device)
    if not _L0["init"]:
        return orig(self, x)
    gate_up, _ = self.gate_up_proj(x)
    y = self.act_fn(gate_up)
    dp = self.down_proj
    dp.reduce_results = False
    try:
        partial, _ = dp(y)
    finally:
        dp.reduce_results = True
    if not (partial.is_contiguous() and partial.shape[1] == 7168):
        from vllm.distributed import tensor_model_parallel_all_reduce

        _L0["stats"]["fallback"] += 1
        return tensor_model_parallel_all_reduce(partial)
    _L0["pending"] = partial
    return partial


def _l1_pre_attn_norm(self, hidden_states, residual, prefix_sum):
    p = _L0["pending"]
    if p is None or hidden_states is not p:
        return _STATE["orig"]["pre_attn_norm"](self, hidden_states, residual, prefix_sum)
    _L0["pending"] = None
    if not (getattr(self, "_k3sov_l1", False) and prefix_sum is not None and residual is not None
            and prefix_sum.stride(0) == 7168 and residual.stride(-1) == 1
            and 0 <= self.prev_valid_blocks <= 8):
        from vllm.distributed import tensor_model_parallel_all_reduce

        _L0["stats"]["fallback"] += 1
        return _STATE["orig"]["pre_attn_norm"](self, tensor_model_parallel_all_reduce(p), residual,
                                               prefix_sum)
    out = torch.empty_like(p)
    torch.ops.k3ar.ar_attn_res(
        p, _L0["mb"], _L0["mc"], 0, _L0["rank"], _L0["tp"], prefix_sum, True, residual,
        self.self_attention_res_norm.weight, self.self_attention_res_proj.weight.squeeze(0),
        self.input_layernorm.weight, out, self.prev_valid_blocks,
        self.block_write_idx if self.is_block_write_layer else -1,
        self.self_attention_res_norm.variance_epsilon, self.input_layernorm.variance_epsilon)
    _L0["stats"]["fused"] += 1
    _report("l0fuse", "layer-0 MLP all-reduce fused into layer 1's AttnRes (k3ar.ar_attn_res)")
    return out, prefix_sum, residual


def _install_l0_fuse() -> None:
    """Flags (at model construction) the dense layer 0 -> layer 1 pair when: TP > 1, AttnRes, layer 0
    is a plain KimiMLP (TP-sharded down_proj with reduce_results, no sequence parallel, no GEMM-RS-AR)
    and nothing reads layer 0's output except layer 1 (no aux hidden-state layers).  One k3ar call per
    forward uses mailbox buffer 0 only: consecutive calls are separated by the forward's other
    collectives.  Numerics: fp32 sum of the 16 partials in rank order, one bf16 rounding, then
    AttnRes (the K3OPT_ARRES kernel); not bit-identical to flashinfer's all-reduce + vLLM's AttnRes
    kernel.  K3STEPOV_L0FUSE=0: original path."""
    from vllm.models.kimi_k3.nvidia import model as k3_model

    if "mlp_forward" in _STATE["orig"]:
        return
    Model, Layer, MLP = k3_model.KimiLinearModel, k3_model.KimiDecoderLayer, k3_model.KimiMLP
    orig_init = Model.__init__

    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            layers = self.layers
            start = getattr(self, "start_layer", 0)
            end = getattr(self, "end_layer", len(layers))
            if end - start < 2 or getattr(self, "aux_hidden_state_layers", ()):
                return
            l0, l1 = layers[start], layers[start + 1]
            mlp = getattr(l0, "mlp", None)
            dp = getattr(mlp, "down_proj", None)
            ok = (type(mlp) is MLP and dp is not None and getattr(dp, "reduce_results", False)
                  and getattr(dp, "tp_size", 1) > 1 and mlp.gemm_rs_ar is None
                  and not mlp.shard_sequence_parallel
                  and not getattr(l0, "use_sequence_parallel", False)
                  and not getattr(l1, "use_sequence_parallel", False)
                  and getattr(l1, "use_attn_res", False) and getattr(l0, "use_attn_res", False)
                  and getattr(dp, "output_size", 7168) == 7168)
            if ok:
                mlp._k3sov_l0 = True
                l1._k3sov_l1 = True
        except Exception as e:  # noqa: BLE001
            print(f"[k3stepov] layer-0 fusion not installed: {e!r}", flush=True)

    Model.__init__ = __init__
    _STATE["orig"]["l0_model_init"] = orig_init
    _STATE["orig"]["mlp_forward"] = MLP.forward
    MLP.forward = _l0_mlp_forward
    _STATE["orig"]["pre_attn_norm"] = Layer._pre_attn_norm
    Layer._pre_attn_norm = _l1_pre_attn_norm


# ==========================================================================================
# 8. KDA state-index staging: alias the builders' persistent buffers to the aligned indices
# ==========================================================================================
def _alias_state_indices(ms, attn_groups, kv_cache_config, block_tables) -> None:
    """Once per model state, before any CUDA graph is captured: point every KDA metadata builder's
    persistent ``non_spec_state_indices_tensor`` at its row of the align context's
    ``aligned_state_indices`` [groups, max_reqs, 1] (which get_aligned_state_indices_multi_group
    writes every step).  The builder's per-step ``copy_`` from that row into its buffer then copies
    a view onto itself, which PyTorch skips (no kernel): one device-to-device copy per KDA group and
    step disappears.  Same memory contents at every point the forward reads them."""
    if getattr(ms, "_k3sov_alias", None) is not None:
        return
    ok = False
    try:
        if (_on("K3STEPOV_STATE_ALIAS") and not _STATE["capture_seen"] and not _capturing()
                and getattr(ms, "_align_mode", False) and block_tables is not None):
            group_ids, _ = ms._get_mamba_group_info(kv_cache_config)
            ctx = ms._ensure_align_ctx(kv_cache_config, group_ids, block_tables)
            al = getattr(ctx, "aligned_state_indices", None)
            if isinstance(al, torch.Tensor) and al.dim() == 3 and al.shape[2] == 1 and al.is_contiguous():
                n = 0
                for gi, gid in enumerate(group_ids):
                    for group in attn_groups[gid]:
                        b = group.get_metadata_builder(0)
                        buf = getattr(b, "non_spec_state_indices_tensor", None)
                        if (not hasattr(b, "mamba_aligned_state_indices") or not isinstance(buf, torch.Tensor)
                                or buf.dim() != 1 or buf.dtype != al.dtype or buf.device != al.device
                                or buf.shape[0] > al.shape[1]):
                            continue
                        b.non_spec_state_indices_tensor = al[gi, : buf.shape[0], 0]
                        n += 1
                ok = n > 0
                if ok:
                    print(f"[k3stepov] KDA state-index staging aliased for {n} group builder(s): "
                          f"no per-step device-to-device copies", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[k3stepov] state-index alias not installed: {e!r}", flush=True)
        ok = False
    ms._k3sov_alias = ok


def _install_state_alias() -> None:
    try:
        from vllm.v1.worker.gpu.model_states import mamba_hybrid as mh
    except Exception:  # noqa: BLE001
        return
    cls = getattr(mh, "MambaHybridModelState", None)
    if cls is None or "state_alias_prepare_attn" in _STATE["orig"]:
        return
    orig = cls.prepare_attn

    @functools.wraps(orig)
    def prepare_attn(self, input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups,
                     kv_cache_config, *args, **kwargs):
        _alias_state_indices(self, attn_groups, kv_cache_config, block_tables)
        return orig(self, input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups,
                    kv_cache_config, *args, **kwargs)

    _STATE["orig"]["state_alias_prepare_attn"] = orig
    cls.prepare_attn = prepare_attn


# ==========================================================================================
def patch_stepov(load_ext) -> bool:
    if _on("K3STEPOV_DISABLE", "0"):
        print("[k3stepov] K3STEPOV_DISABLE=1: stepov patch off", flush=True)
        return False
    if _STATE["patched"]:
        return True
    load_ext()
    if not hasattr(torch.ops, "k3step") or not hasattr(torch.ops.k3step, "sample_finish"):
        raise RuntimeError("k3stepov: torch.ops.k3step missing (load_ext must build k3step.cu)")
    _install_sampling()
    prep = _install_prepare_inputs()
    _install_embed()
    _install_lmhead_prefetch()
    _install_final_norm()
    # Install only when enabled: the wrapper replaces KimiDecoderLayer._pre_attn_norm, and tailattn
    # defers a MoE tail only if that method is still its own (identity check) -- an always-installed
    # pass-through wrapper silently disabled 91 of 92 tail hand-offs (+0.17-0.25 ms/step).
    if _on("K3STEPOV_L0FUSE", "0"):
        _install_l0_fuse()
    _install_state_alias()
    _STATE["patched"] = True
    print("[k3stepov] patched: distributed sampling + fused post-update, "
          f"input-prep caching {'on' if prep else 'OFF'}, embedding broadcast, lm_head prefetch "
          f"({_lmhead_pf_bytes() / 2**20:.0f} MB)", flush=True)
    return True


def stats() -> dict:
    return {**_STATE["stats"], **_PREP["stats"], **{"l0_" + k: v for k, v in _L0["stats"].items()}}
