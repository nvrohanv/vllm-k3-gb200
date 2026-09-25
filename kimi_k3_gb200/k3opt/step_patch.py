"""Shrink the per-step GPU work outside the CUDA graph (Kimi K3 decode, TP16, vLLM V2 runner).

Call ``patch_step(load_ext)`` once per process at plugin time; ``load_ext()`` must build/load
agents/kda/k3samp.cu (torch.ops.k3samp).  A/B switch: ``K3STEP_DISABLE=1`` (nothing patched).
Per-feature switches (read at call time): ``K3STEP_DIST_SAMPLE=0``, ``K3STEP_FAST_GATHER=0``.

1. TP-distributed Gumbel-max sampling (``GPUModelRunner.sample``), for batches whose sampling is
   exactly vLLM's plain Gumbel-max path: every request has temperature 0 or 1 and no penalties /
   logit bias / bad words / min_p / top-k / top-p / thinking budget (``needs_logits_processing``
   false), no logprobs, no structured output, no spec-decode drafts, no trace replay / NaN
   counting / sampling mask / fp64 Gumbel, a plain ``Sampler``, no batch-sharded sampling, no PP,
   1 <= num_reqs <= 16.  This is what ``vllm bench serve`` requests carry (no sampling params ->
   temperature 1.0).  Instead of all-gathering [M, vocab] logits over TP16 (NCCL ring, ~34 us)
   and running the full-vocab Gumbel kernel + argmax + gather (~15 us, 6 launches), every rank
   runs vLLM's own ``gumbel_noised_argmax`` on its [M, vocab/16] shard with global token ids as
   noise keys (same seeds / positions / temperature), and k3samp.dist_argmax exchanges one
   (value, id) packet per token and rank over an NVLS Lamport mailbox and reduces them with
   vLLM's tie-breaking (max value, smallest id).  Sampled ids are bit-identical to the unpatched
   path and identical on all ranks.  The hidden-state row gather is skipped when it is an
   identity (one token per request), and num_sampled / num_rejected come out of the same kernel.
   2 launches instead of 9.
2. One-hop logits all-gather (``LogitsProcessor._gather_logits``) for everything else with
   1 <= M <= 16 bf16 rows (logprobs, top-p, ...): k3samp.allgather over the NVLS mailbox instead
   of NCCL.  Bit-identical except that bf16 -0.0 logits become +0.0.

All eligibility decisions depend only on state that is replicated on every TP rank (scheduler
output, per-request sampling params, the same numpy seed stream), so all ranks always take the
same path; the mailboxes are allocated collectively on the first logits call (which every rank
makes at the same point) and both features turn themselves off on all ranks together if NVLS
multicast is unavailable.  Nothing runs inside CUDA graph capture (the hooks fall back then).
A rank that never publishes makes the others abort after 20 s (trap) instead of hanging.
"""

from __future__ import annotations

import os

import numpy as np
import torch

MAX_M = 16
NBUF = 2
BLOCK = 1024  # vLLM's gumbel_sample block (noise keys and tie-break are per global token id)
SENT_G = -0x80000000  # int32 view of the all-gather empty marker
SENT_A = -1           # int32 view of the argmax-packet empty marker

_STATE = {
    "patched": False,
    "init": None,       # None: not tried; False: failed/unavailable; True: ready
    "tp": 1,
    "rank": 0,
    "shard": 0,         # per-rank vocab shard S
    "gather_mb": None, "gather_mc": 0, "gather_call": 0,
    "arg_mb": None, "arg_mc": 0, "arg_call": 0,
    "mode": 0,          # 0: multicast (production); 1: local stores (single-GPU tests)
    "want_local": False,
    "got_local": False,
    "orig_gather": None,
    "orig_sample": None,
    "kernel": None,
    "stats": {"dist_sample": 0, "sample_fallback": 0, "fast_gather": 0, "gather_fallback": 0},
    "reported": set(),
}


def _on(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) == "1"


def _report(key: str, msg: str) -> None:
    if key not in _STATE["reported"]:
        _STATE["reported"].add(key)
        print(f"[k3step] {msg}", flush=True)


# ------------------------------------------------------------------------------------------
# Triton: per-rank Gumbel-max over the local vocab shard, vLLM's own noise function
# ------------------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------------------
# Symmetric mailboxes (collective, first logits call)
# ------------------------------------------------------------------------------------------
def _init(shard: int) -> bool:
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
            print(f"[k3step] symmetric mailbox setup failed: {e}", flush=True)
            gmc = amc = 0
    flag = torch.tensor([1 if (ok and gmc != 0 and amc != 0) else 0], dtype=torch.int32,
                        device=dev)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=tpg.device_group)
    if int(flag.item()) == 0:
        _STATE["init"] = False
        print("[k3step] NVLS multicast mailbox unavailable -> original sampling / all-gather",
              flush=True)
        return False
    gmb.view(torch.int32).fill_(SENT_G)
    amb.fill_(SENT_A)
    torch.cuda.synchronize()
    tpg.barrier()
    _STATE.update(init=True, tp=tp, rank=rank, shard=shard, gather_mb=gmb, gather_mc=gmc,
                  arg_mb=amb, arg_mc=amc, mode=0)
    print(f"[k3step] NVLS mailboxes ready (tp={tp}, vocab shard={shard})", flush=True)
    return True


def install_local_mailboxes_for_tests(shard: int, world: int = 1, rank: int = 0):
    """Single-GPU tests: local mailboxes, local stores (mode 1) into slot `rank`."""
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
    torch.ops.k3samp.allgather(local, _STATE["gather_mb"], _STATE["gather_mc"], buf,
                               _STATE["rank"], _STATE["mode"], out)
    return out


def dist_argmax(local_max, local_arg, idx_mapping, seq_lens, cu_num_logits, prefill_len):
    M = local_max.shape[0]
    dev = local_max.device
    sampled = torch.empty(M, dtype=torch.int64, device=dev)
    num_sampled = torch.empty(M, dtype=torch.int32, device=dev)
    num_rejected = torch.empty(M, dtype=torch.int32, device=dev)
    buf = _STATE["arg_call"] % NBUF
    _STATE["arg_call"] += 1
    # The V2 runner's metadata may be int64 or strided views; the kernel wants dense int32 [M].
    idx_mapping, seq_lens, cu_num_logits, prefill_len = (
        t if (t.dtype == torch.int32 and t.is_contiguous()) else t.to(torch.int32).contiguous()
        for t in (idx_mapping, seq_lens, cu_num_logits, prefill_len))
    torch.ops.k3samp.dist_argmax(local_max, local_arg, _STATE["arg_mb"], _STATE["arg_mc"], buf,
                                 _STATE["rank"], _STATE["mode"], idx_mapping, seq_lens,
                                 cu_num_logits, prefill_len, sampled, num_sampled, num_rejected)
    return sampled, num_sampled, num_rejected


# ------------------------------------------------------------------------------------------
# Hooks
# ------------------------------------------------------------------------------------------
def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
    """LogitsProcessor._gather_logits (only reached for TP > 1)."""
    orig = _STATE["orig_gather"]
    if torch.cuda.is_current_stream_capturing() or logits.dim() != 2:
        return orig(self, logits)
    want_local = _STATE["want_local"]
    small = (logits.dtype == torch.bfloat16 and 1 <= logits.shape[0] <= MAX_M
             and logits.stride(1) == 1)
    if (want_local or (small and _on("K3STEP_FAST_GATHER"))) and not _init(logits.shape[1]):
        return orig(self, logits)
    if want_local:
        _STATE["got_local"] = True
        return logits
    if small and _on("K3STEP_FAST_GATHER") and logits.data_ptr() % 16 == 0 \
            and (logits.stride(0) * 2) % 16 == 0:
        _STATE["stats"]["fast_gather"] += 1
        _report("gather", "one-hop NVLS logits all-gather active")
        return fast_allgather(logits)
    _STATE["stats"]["gather_fallback"] += 1
    return orig(self, logits)


def _dist_sample_ok(runner, input_batch, grammar_output) -> bool:
    from vllm.v1.worker.gpu.sample.sampler import Sampler

    if not _on("K3STEP_DIST_SAMPLE") or _STATE["init"] is False:
        return False
    if torch.cuda.is_current_stream_capturing():
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


def _sample(self, hidden_states, input_batch, grammar_output):
    """GPUModelRunner.sample with TP-distributed Gumbel-max for plain sampling batches."""
    orig = _STATE["orig_sample"]
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
        # the logits hook was not reached / mailboxes unavailable: `logits` are full logits
        _STATE["stats"]["sample_fallback"] += 1
        so = self.sampler(logits, input_batch)
        return so, so.num_sampled, so.num_rejected
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    s = self.sampler
    states = s.sampling_states
    local_max, local_arg = local_gumbel(
        logits, _STATE["rank"] * logits.shape[1], self.vocab_size,
        input_batch.expanded_idx_mapping, states.temperature.gpu, states.seeds.gpu,
        input_batch.positions, input_batch.logits_indices)
    sampled, num_sampled, num_rejected = dist_argmax(
        local_max, local_arg, input_batch.idx_mapping, input_batch.seq_lens,
        input_batch.cu_num_logits, self.req_states.prefill_len.gpu)
    _STATE["stats"]["dist_sample"] += 1
    _report("sample", f"TP-distributed Gumbel-max sampling active (M={n})")
    so = SamplerOutput(sampled_token_ids=sampled.view(-1, 1), logprobs_tensors=None,
                       num_nans=None, num_sampled=num_sampled, num_rejected=num_rejected)
    return so, num_sampled, num_rejected


def patch_step(load_ext) -> bool:
    if _on("K3STEP_DISABLE", "0"):
        print("[k3step] K3STEP_DISABLE=1: step patch off", flush=True)
        return False
    if _STATE["patched"]:
        return True
    load_ext()
    if not hasattr(torch.ops, "k3samp") or not hasattr(torch.ops.k3samp, "dist_argmax"):
        raise RuntimeError("k3step: torch.ops.k3samp missing (load_ext must build k3samp.cu)")
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    _STATE["orig_gather"] = LogitsProcessor._gather_logits
    _STATE["orig_sample"] = GPUModelRunner.sample
    LogitsProcessor._gather_logits = _gather_logits
    GPUModelRunner.sample = _sample
    _STATE["patched"] = True
    print("[k3step] patched: TP-distributed sampling + one-hop logits all-gather "
          "(K3STEP_DIST_SAMPLE / K3STEP_FAST_GATHER)", flush=True)
    return True
