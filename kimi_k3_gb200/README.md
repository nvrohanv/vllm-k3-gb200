# Kimi K3 decode on 16× GB200: beating the TPU v7 megakernel numbers

Work in progress. The goal is to make vLLM v0.30.0's Kimi K3 decode on 16× GB200 (TP16, 4 NVL4 trays)
beat the TPU v7 numbers from <https://inferact.ai/blog/tpu-megakernels> by 25%. The benchmark is no
spec-dec, 1 input / 1024 output tokens, concurrency B, and the metric is decode-only aggregate tok/s
= B / TPOT.

| B | blog GB200 vLLM | our repro of stock vLLM | **current best** | TPU v7 (blog) | TPU +25% target |
|---|---|---|---|---|---|
| 1 | 127 | 127.2 (7.86 ms) | **135.2 (7.394 ms)** | 249 | 311 (3.2 ms) |
| 8 | 636 | 663.3 (12.06 ms) | **699.2 (11.441 ms)** | 865 | 1081 (7.4 ms) |

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
  - Experimental and not yet a win: `K3OPT_ROUTE`, `K3OPT_MOEFUSED` (`csrc/moe_small.cu`),
    `K3OPT_ARRES` (`csrc/ar_attn_res.cu`).
- `deploy/`: `serve.sh` plus the server args and env used, and `exps/*.env` flag sets
  (`best.env` = current best).
- `bench/`: `bench.sh` (mirrors the TPU harness: `vllm bench serve`, random 1/1024,
  `--ignore-eos`, 4·B prompts), `summarize.py`, `gsm8k_eval.py` (8-shot, first 300 questions),
  and `critpath.py` (per-kernel critical-path attribution from torch profiler traces; with PDL, raw
  kernel durations overlap their predecessors).

## Accuracy checks for each change
- GSM8K 8-shot, first 300 questions: 96.0–97.0% for every stack listed here.
- Teacher-forced logprobs vs stock, and greedy token match on fixed prompts.

## History (TPOT ms, B=1 / B=8)
| stack | B=1 | B=8 |
|---|---|---|
| stock vLLM v0.30.0 recipe | 7.86 | 12.06 |
| + KDA6 + DOWNSHARD | 7.613 | 11.629 |
| + TAILATTN | 7.394 | 11.441 |
