# Kimi K3 decode on 16× GB200: beating the TPU v7 megakernel numbers

Work in progress. The goal is to make vLLM v0.30.0's Kimi K3 decode on 16× GB200 (TP16, 4 NVL4 trays)
beat the TPU v7 numbers from <https://inferact.ai/blog/tpu-megakernels> by 25%. The benchmark is no
spec-dec, 1 input / 1024 output tokens, concurrency B, and the metric is decode-only aggregate tok/s
= B / TPOT.

| B | blog GB200 vLLM | our repro of stock vLLM | **current best** | TPU v7 (blog) | TPU +25% target |
|---|---|---|---|---|---|
| 1 | 127 | 127.2 (7.86 ms) | **159.2 (6.283 ms)** | 249 | 311 (3.2 ms) |
| 2 | 227 | 229.4 (8.72 ms) | **276.2 (7.240 ms)** | 392 | 490 |
| 4 | 373 | 384.6 (10.40 ms) | **448.9 (8.910 ms)** | 515 | 644 |
| 8 | 636 | 663.3 (12.06 ms) | **753.6 (10.615 ms)** | 865 | 1081 (7.4 ms) |

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
  - `K3OPT_OPROJ=1` (`csrc/oproj_ar.cu`, `oproj_patch.py`, being validated): the o_proj GEMV epilogue
    multicasts each rank's partial into a Lamport mailbox on all 16 GPUs, and the consumer sums the 16
    slots in a fixed order and runs the post-attention AttnRes. o_proj + all-reduce + AttnRes become
    one PDL kernel.
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
- `deploy/`: `serve.sh` plus the server args and env used, and `exps/*.env` flag sets
  (`best.env` = current best). Restart-to-restart noise is about ±2%.
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
| + MOEFUSED (M ≤ 2) | 6.989 | 11.390 |
| + KDASPLIT | 6.701 | 11.020 |
| + MLA + L2PF (`combo5g`) | 6.404 | 10.822 |
| + KDAFB | 6.283 | 10.615 |
