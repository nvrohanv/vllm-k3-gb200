# Design brief: Kimi K3 decode on 16x GB200 — beat TPU v7 at B=1/B=2, ideally +25% at every B

## Goal (decode-only aggregate tok/s = B / TPOT; 1 input token, 1024 output tokens, no spec-dec / MTP)
| B | TPU v7 megakernel (blog) | TPU +25% (stretch) | our current best | per-layer budget to beat TPU / +25% |
|---|---|---|---|---|
| 1 | 249 tok/s = 4.02 ms | 311 = 3.21 ms | 186.2 = **5.370 ms** | ~43 us / ~34.5 us (93 layers + ~0.1 ms/step outside layers) |
| 2 | 392 = 5.10 ms | 490 = 4.08 ms | 339.2 = **5.896 ms** | ~54 us / ~43 us |
| 4 | 515 = 7.77 ms | 644 = 6.21 ms | 561.6 = **7.123 ms** (already beats TPU) | / ~66 us |
| 8 | 865 = 9.25 ms | 1081 = 7.40 ms | 901.9 = **8.870 ms** (already beats TPU) | / ~79 us |
Blog: https://inferact.ai/blog/tpu-megakernels ("700 TPS on Kimi K3", a TPU v7 megakernel). Accuracy must be
preserved (GSM8K 8-shot first-300 ~96%, teacher-forced logprobs vs stock; no weight-precision changes unless
explicitly justified and validated).

## Deployment
vLLM v0.30.0 (image vllm/vllm-openai:v0.30.0, torch 2.13, CUDA 13.0, CuTe DSL 4.7.1, FlashInfer 0.6.18), TP16 across
4 GB200 NVL4 trays (one NVLink domain, NVLS multicast available, MNNVL). Decode runs in FULL CUDA graphs captured at
batch 1,2,4,8,16 (V2 model runner, compile mode NONE, breakable piecewise graphs for mixed batches). All our work is a
vLLM *plugin* (`gpu-opt/k3opt/`: monkeypatches + custom CUDA ops); we may replace arbitrarily large parts of the
decode forward (a custom op can implement an entire layer or the entire model) as long as it runs inside vLLM's
serving loop, keeps KV/state caches compatible, and falls back for prefill/mixed batches.

## Hardware (GB200, per GPU)
152 SMs @ 2062 MHz (at max clock, no throttling), ~228 KB smem/SM, 64K regs/SM, TMEM + tcgen05 (block-scaled FP4/FP8
MMA), TMA, clusters/DSMEM, PDL. HBM ~6.9 TB/s (measured copy). L2 129 MB split across 2 dies (local hit ~128 ns,
remote-die hit ~377 ns, HBM ~425-530 ns; prefetch lands at the home slice only; see agents/l2pf/RESULTS.md).
NVLink5 1.8 TB/s bidirectional per GPU; measured one-way NVLS multicast store -> remote poll visible ~1.5-2.5 us incl.
kernel overheads. Empty PDL kernel boundary ~0.6-1 us. Internal NVIDIA docs available via the Perforce MCP tools
(`maas_p4_hw` //hw/doc/gpu/..., `maas_p4_sw` //sw/compiler/gpgpu/...; see agents/CONTEXT.md).

## Model per GPU (TP16), decode
hidden 7168, 93 layers = 69 KDA (gated delta-rule linear attention, 6 heads x 128 per rank) + 24 MLA (6 heads per
rank, q-LoRA, fp8 paged KV, NoPE, sigmoid output gate); layer 0 dense MLP, layers 1..92 MoE.
AttnRes block residuals (vLLM `attn_res` op: norm over up to 8 block summaries + mixing) before attention and before
MoE. MoE = latent MoE: hidden 7168 -> latent 3584 (down-proj, replicated weight, we shard it 16-way + multicast),
896 experts, top-16 (sigmoid + bias, renormalize, scaling), per-rank expert slice 192 (padded 256) MXFP4 weights,
SiTU-GLU; shared expert (gate_up 768x7168, down 7168x384 per rank); MoE tail = weighted top-k reduce + TP all-reduce
of latent + RMSNorm + reduce-scatter + sharded up-projection 3584->7168 + multicast to all ranks (Lamport mailbox),
consumed by the next layer's pre-attention AttnRes.
Approx weight bytes read per GPU per layer at B=1: KDA in_proj 46 MB, o_proj 11 MB, router 12.8 MB, shared expert
16.5 MB, down-shard 3.2 MB, up-proj shard ~3.2 MB, routed experts ~17-22 MB (16 experts x (FC1+FC2 slices), MXFP4)
=> ~110-115 MB/layer => ~16.5 us/layer at HBM BW => ~1.55 ms/token weight floor at B=1 (MLA layers similar:
fused_qkv_a_g ~41 MB instead of in_proj). At B=8 routed experts grow to up to 128 distinct experts (~140-176 MB).
lm_head 147 MB per step.

## Where the time goes now (B=1, rank 0, real profile of the current best; full dumps in agents/timeline_b1_mla2.txt)
KDA layer = 53.0 us (from previous MoE-tail end):
  up-proj multicast end 1.7 -> pre-attn AttnRes (Lamport consumer, k3tail) end 5.4 -> in_proj CuTe skinny GEMM end 11.4
  -> KDA split+f_b kernel end 15.3 -> o_proj produce (GEMV + NVLS multicast) end 17.9 -> o_proj consume (16-slot sum +
  post-attn AttnRes) end 22.4 -> route_shared (router GEMV + scores + shared gate_up/SiTU) end 25.9 ; side stream:
  down-shard GEMV+multicast 22.7->28.4 -> MoE block kernel (poll latent mailbox, in-kernel top-16, FC1/FC2 CUDA-core
  MXFP4, shared down) 26.4 -> 46.2 (19.8 us) -> vLLM MoE-tail all-reduce+RMSNorm+reduce-scatter end 53.0
  (+ next layer's up-proj multicast). An L2-prefetch side kernel prefetches the next layer's attention weights.
MLA layer = 60.4 us (qkv_a GEMM 2.6->9.8, q_prep ->16.2, attn_out ->21.6, o_proj ->27.9, route ->32.5, MoE block ->51.0,
tail ->60.4).
Cross-GPU hop costs (median, identical on all 16 ranks => NO straggler; this is intrinsic latency incl. consumer
work): MoE block end -> tail end 8.1 us; o_proj produce end -> consume end 5.3 us; tail up-proj end -> AttnRes end
3.6 us; plus the down-shard hop (~3-5 us). => ~17-21 us/layer of the ~53-60 us is cross-GPU hop latency.
Outside the layers: ~130 us/step (lm_head 22 us, sampling ~10 us after our distributed-sampling patch, ~90 us of
V2-runner input-prep kernels outside the graph).
B=8: current best 8.87 ms (~95 us/layer): MoE side stream (down-shard + our tcgen05 MXFP4 MoE kernel 30 us) dominates,
dense GEMMs use FlashInfer split-K.

## Existing building blocks (all working in the server; sources in gpu-opt/k3opt/csrc, notes in gpu-opt/agents/*/)
- lamport_attn_res.cu (k3tail): Lamport mailbox consume + AttnRes, cluster per token, ~0.8-2.6 us.
- oproj_ar.cu (k3oproj): o_proj GEMV with multimem.st epilogue into 16-slot mailbox + consumer (fixed-order sum +
  post-attn AttnRes); fused (one kernel) or produce+consume.
- kda_split.cu (k3kdas): KDA decode split over a cluster, f_b_proj folded, ~2-3 us.
- k3mla.cu: q_prep (+cache insert) and attn_out (split-KV + combine + W_UV + gate) cluster kernels.
- moe_small.cu (k3moe): route_shared, moe_block_lamport (persistent grid-barrier FC1/FC2 CUDA-core MXFP4, in-kernel
  top-16, shared down, routing-only mode), moe_fused_unfinalized(_lamport).
- moe8.cu (k3moe8): tcgen05 block-scaled MXFP4 MoE (MXFP8 activations), 12.2/14.5/19.4/30.2 us at M=1/2/4/8 standalone.
- l2pf.cu: cross-layer L2 bulk prefetch side kernel. k3samp.cu: distributed Gumbel-max sampling + NVLS all-gather.
- vLLM's own CuTe DSL latent-MoE-tail kernels (allreduce_rmsnorm_reduce_scatter_early_exit, fused_add_multicast_*
  up-projection, LamportCopy) and CuTe skinny GEMM / FlashInfer split-K dense GEMMs.

## Measurement / validation harness
gpu-opt/exp.sh + queue.sh (server A/B with events), dist_test.sh (16-rank tests on the server pods), per-kernel
single-GPU chain benchmarks in agents/*/. Restart-to-restart noise ~2%.
