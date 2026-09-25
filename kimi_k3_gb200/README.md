# Kimi K3 decode on 16× GB200: beating the TPU v7 megakernel numbers

Work in progress. The goal is to make vLLM v0.30.0's Kimi K3 decode on 16× GB200 (TP16, 4 NVL4 trays)
beat the TPU v7 numbers from <https://inferact.ai/blog/tpu-megakernels> by 25%. The benchmark is no
spec-dec, 1 input / 1024 output tokens, concurrency B, and the metric is decode-only aggregate tok/s
= B / TPOT.

| B | blog GB200 vLLM | our repro of stock vLLM | **current best** | TPU v7 (blog) | TPU +25% target |
|---|---|---|---|---|---|
| 1 | 127 | 127.2 (7.86 ms) | **193.5 (5.168 ms)** | 249 | 311 (3.2 ms) |
| 2 | 227 | 229.4 (8.72 ms) | **344.0 (5.814 ms)** | 392 | 490 |
| 4 | 373 | 384.6 (10.40 ms) | **571.1 (7.003 ms)** | 515 | 644 |
| 8 | 636 | 663.3 (12.06 ms) | **906.4 (8.826 ms)** | 865 | 1081 (7.4 ms) |

## How it works

Everything is a vLLM **plugin** applied on top of the stock `vllm/vllm-openai:v0.30.0` image. No vLLM
source files are edited; the patches are installed at plugin registration.

- `k3snap/`: `--load-format k3snap`. On first start, each rank saves its weights *after*
  `process_weights_after_loading` to local NVMe. Later starts rebuild the model and copy the saved
  tensors in place. Model load drops from ~400 s (fastsafetensors from Lustre) to ~60–100 s.
- `k3opt/`: decode kernels and model patches, each turned on by an env flag:
  - `K3OPT_KDA6=1`: vLLM's fused KDA decode kernel (`csrc/kda_decode6.cu`), extended to the 6 KDA
    heads per rank that TP16 produces. Without this, the model falls back to 4 unfused kernels.
  - `K3OPT_DOWNSHARD=1`: the latent MoE down-projection (7168→3584, replicated on every rank) is
    sharded 16 ways for M ≤ 8. It reuses vLLM's CuTe skinny GEMM and Lamport multicast kernels.
  - `K3OPT_TAILATTN=1` (`csrc/lamport_attn_res.cu`, `tailattn_patch.py`): the MoE tail's Lamport
    mailbox is consumed directly by the next layer's pre-attention AttnRes, in one PDL cluster kernel.
    This replaces the `LamportCopyKernel` + `kimi_k3_attn_res` pair, and runs 92× per token.
  - `K3OPT_MOEFUSED=1 K3OPT_MOEFUSED_MAX_M=2` (`csrc/moe_small.cu`): for decode batches of ≤ 2
    tokens, one persistent kernel does FC1 + SiTU-GLU + FC2 on the BF16 latent. It reads TRT-LLM's
    MXFP4 shuffled layout in place and returns the *unfinalized* per-(token, expert) rows the K3
    MoE-tail collective expects. Top-k is computed next to the gate GEMM, while the down-projection
    runs on the aux stream. This replaces MXFP8 quantize + routing + FC1 + FC2 (~27 µs → ~13 µs at M=1).
  - `K3OPT_KDASPLIT=1` (`csrc/kda_split.cu`): KDA decode with the value dimension split over a
    thread-block cluster (8 CTAs per head at M=1), with state prefetched before `griddepcontrol.wait`
    and bit-identical results. ~8.4 → ~2 µs at M=1, ~8.6 → ~3.1 µs at M=8, on 69 layers.
  - `K3OPT_KDAFB=1` (`kda_fb_patch.py`): the KDA gate low-rank up-projection (`f_b_proj`, a separate
    cuBLAS call) is computed inside the KDA split kernel by a dedicated warp, matching cuBLAS's
    accumulation order bit for bit.
  - `K3OPT_OPROJ=1` (`csrc/oproj_ar.cu`, `oproj_patch.py`): the o_proj GEMV epilogue
    multicasts each rank's partial into a Lamport mailbox on all 16 GPUs, and the consumer sums the 16
    slots in a fixed order and runs the post-attention AttnRes. o_proj + all-reduce + AttnRes become
    one PDL kernel.
  - `K3OPT_MOEBLOCK=1` (`moeblock_patch.py`, kernels in `csrc/moe_small.cu`): for M ≤ 2 the MoE
    block becomes two main-stream kernels, replacing nine:
    - `route_shared`: router GEMV + scores + shared-expert gate_up/SiTU;
    - `moe_block_lamport`: in-kernel top-16, latent polled straight from the down-shard Lamport
      mailbox, FC1/FC2, and shared down.
    The side stream only runs the down-shard multicast producer.
  - `K3OPT_PLANS=1` (`plans_patch.py`): adds FlashInfer's CuTe-DSL split-K dense GEMM
    (`run_splitk_dense`: cluster split-K, reduction in-kernel, PDL weight prefetch) to vLLM's
    low-latency GEMM plan table for KDA in_proj and MLA qkv_a/gate at M = 3..16, replacing
    cuBLAS split-K + reduce.
  - `K3OPT_MOE8=1` (`csrc/moe8.cu`, `tc_utils.cuh`, `mma_issue.cuh`): a tensor-core MXFP4 MoE in one
    persistent launch:
    - MXFP8 quantization of x, then FC1, SiTU and MXFP8 quantization of h, then FC2 with bf16
      unfinalized output;
    - tcgen05 block-scaled MMA (swap-AB M=128 N=8), TMA and TMEM;
    - reads TRT-LLM's shuffled layout in place.
    Standalone it takes 12.2 / 14.5 / 19.4 / 30.2 µs at M = 1 / 2 / 4 / 8, against TRT-LLM's quant +
    routing + FC1 + FC2 at 17.9 / 21.3 / 31.1 / 55.0 µs. Used for M = 5..8 through the fused-MoE path;
    top-k runs in the router branch. With `K3MOEBLOCK_MOE8=1 K3MOEBLOCK_MOE8_MIN_M=2` it also
    replaces the CUDA-core FC1/FC2 inside the M ≤ 4 MoE block. The block kernel then runs in
    routing-only mode: top-k, shared expert, latent hand-off. At M=1 the extra kernel boundary
    costs more than the FC gain, so M=1 keeps the fused CUDA-core block.
  - `K3OPT_STEP=1` (`csrc/k3samp.cu`, `step_patch.py`): distributed sampling. For plain
    temperature-0/1 batches (no top-p/k, penalties or logprobs), each rank runs vLLM's own
    Gumbel-max over its vocab shard, and one NVLS mailbox exchange of (value, id) picks the token.
    Sampled ids are bit-identical to vLLM's on all 16 ranks. Otherwise, a one-hop NVLS all-gather
    replaces NCCL's ring for the logits. This removes ~60–100 µs of per-step eager work.
  - `K3OPT_STEPOV=1` (`csrc/k3step.cu`, `k3step_prep.cu`, `stepov_patch.py`; supersedes `K3OPT_STEP`),
    which trims the per-step work outside the CUDA graph:
    - distributed sampling with vLLM's post-update and mamba scatter fused into the finish kernel;
    - an NVLS embedding broadcast that replaces the masked gather + all-reduce;
    - input-prep caching;
    - an lm_head L2 prefetch.
  - `K3OPT_MLA=1` (`csrc/k3mla.cu`, `mla_patch.py`): two fused cluster kernels replace 8 in the MLA
    decode chain (q-prep + cache insert; split-KV attention + combine + W_UV + gate). The dispatch is
    decided inside an eager-break function, so it is correct under breakable piecewise graphs.
  - `K3OPT_L2PF=1` (`csrc/l2pf.cu`, `l2pf_patch.py`): right after each layer's routed-expert FC2, a
    side-stream kernel bulk-prefetches the next layer's attention weights (~55 MB) into L2, so the
    next in_proj / fused_a / o_proj read from L2 instead of HBM. The stream joins at the end of the
    forward.
  - `K3OPT_GEMV=1` (`csrc/k3gemv.cu`, `gemv_patch.py`): PDL weight-prefetch GEMV for cuBLAS-fallback
    cells at M=3..8. **Not used**: in the model it costs +0.58 ms at B=8, probably because it
    competes for SMs with the concurrent routed-MoE stream.
  - Experimental and not a win: `K3OPT_ROUTE`, `K3OPT_ARRES` (`csrc/ar_attn_res.cu`).
- `k3opt/ptrigger.sh` (`K3OPT_PTRIGGER=1`): the only edit to vLLM source. It moves the early PDL
  `launch_dependents` into the two MoE-tail producer kernels. At startup `serve.sh` applies or
  reverts it according to the flag, then `verify_pristine.py` checks the installed vLLM/FlashInfer
  against pip RECORD hashes.
- `deploy/`: `serve.sh` plus the server args and env used, and `exps/*.env` flag sets
  (`best.env` = current best). Restart-to-restart noise is about ±2%.
- `bench/`: `bench.sh` (mirrors the TPU harness: `vllm bench serve`, random 1/1024,
  `--ignore-eos`, 4·B prompts), `summarize.py`, `gsm8k_eval.py` (8-shot, first 300 questions),
  and `critpath.py` (per-kernel critical-path attribution from torch profiler traces; with PDL, raw
  kernel durations overlap their predecessors).

## Accuracy checks for each change
- Full GSM8K 8-shot (all 1319 questions, greedy):
  - current best: 1235/1319 = 93.6%;
  - stock vLLM v0.30.0 recipe on the same harness: 1237/1319 = 93.8%, within noise.
- GSM8K 8-shot, first 300 questions: 95.0–97.0% for every intermediate stack.
- Non-target workloads, current best vs stock recipe:

  | workload | ours | stock |
  |---|---|---|
  | B=16 decode | 1282 tok/s | 1125 tok/s |
  | prefill-heavy (ISL 2048 / OSL 128, 8 concurrent): total throughput | 6761 tok/s | 6691 tok/s |
  | same: TPOT | 11.4 ms | 13.4 ms |
  | same: TTFT | 1129 ms | 895 ms |

- The patches allocate memory after vLLM's memory profiling. At the recipe's `--gpu-memory-utilization 0.95`, a
  14k-token prefill step ran out of memory, so the deploy config uses 0.90 plus
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (see `deploy/base.args`, `deploy/base.env`). Decode speed
  is unchanged.
- Teacher-forced logprobs vs stock, and greedy token match on fixed prompts.

## History (TPOT ms, B=1 / B=8)
| stack | B=1 | B=8 |
|---|---|---|
| stock vLLM v0.30.0 recipe | 7.86 | 12.06 |
| + KDA6 + DOWNSHARD | 7.613 | 11.629 |
| + TAILATTN | 7.394 | 11.441 |
| + MOEFUSED (M ≤ 2) | 6.989 | 11.390 |
| + KDASPLIT | 6.701 | 11.020 |
| + MLA + L2PF (`combo5g`) | 6.404 | 10.822 |
| + KDAFB | 6.283 | 10.615 |
| + OPROJ (fused o_proj + NVLS all-reduce + AttnRes) | 5.769 | 10.155 |
| + MOEBLOCK (M ≤ 2) | 5.383 | 10.159 |
| + PLANS (split-K GEMM, M = 3..16) | 5.50* | 9.873 |
| + MOEBLOCK up to M = 4, tensor-core MoE (moe8) for M = 5..8 | 5.53 | 9.072 (B=4 7.772) |
| + moe8 FC1/FC2 inside the MoE block for M = 2..4 (`moe8c`) | 5.511 | 9.068 (B=4 7.346) |
| + distributed sampling / NVLS logits gather (`stepB`) | 5.454 | 8.970 (B=4 7.257) |
| + faster k3mla (early PDL release, bulk loads, fewer cluster syncs) + split o_proj on MLA layers (`mla2`) | 5.370 | 8.870 (B=4 7.123) |
| + MoE block kernel shared-memory addressing fix (`smemfix`) | 5.329 | 8.848 (B=4 7.118) |
| + L2 prefetch launched without PDL (layer 1's could linger ~200 us and block layer 2's MoE kernel) | 5.301 | 8.874 (B=4 7.091) |
| + step-overhead trims (`stepov`) | 5.275 | 8.833 (B=2 5.821, B=4 7.046) |
| + MoE block "slow mode" fix: routing warps no longer poll the latent, so the top-16 doesn't wait for the last rank's down-shard (`nopoll`) | **5.168** | **8.826** (B=4 **7.003**) |

Next: per-layer persistent kernels (ATTN / MOE / TAIL). See `DESIGN_PLAN.md` for the plan to get B=1/B=2 past the TPU.

Measured primitives on 16 GB200s (wave-1 probe), which update the plan's cost model:

| primitive | measured |
|---|---|
| Lamport/NVLS hop, 14 KB, p50 incl. skew | 2.84-3.11 us |
| in-GPU grid handoff | 1.15 us fence-free (2.08 us with a release counter) |
| 8-CTA clusters co-resident at <=220 KB smem | 15 |
| per-boundary residue with PDL | 0.16-0.22 us |
| DSMEM all-gather, 14 KB | 0.7-0.8 us |

This projects B=1 at about 3.2 ms (range 3.07-3.38; TPU v7 is 4.02, TPU+25% is 3.21) and B=2 at about 3.7 ms
(TPU+25% is 4.08).

\* B=1 doesn't use the PLANS path; 5.38 vs 5.50 is restart-to-restart noise.
