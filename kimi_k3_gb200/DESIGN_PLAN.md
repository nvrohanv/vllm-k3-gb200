# Kimi K3 decode on 16x GB200: plan to beat TPU v7 at B=1/B=2 and reach +25% at every batch size

## 1. Verdict

**Both targets are reachable, and a whole-model megakernel is not needed.**

- **TPU parity at B=1 (4.02 ms) and B=2 (5.10 ms): high confidence, about 85-90%.** This needs the MoE segment and the MoE tail rebuilt as persistent kernels whose boundaries sit on NVLink hops. Small wins on today's one-kernel-per-op stack stop at about 5.0 ms (B=1) and 5.5 ms (B=2).
- **+25% over TPU:**

  | B | target | odds |
  |---|---|---|
  | 4 | 6.21 ms | about 90% |
  | 2 | 4.08 ms | about 80% |
  | 8 | 7.40 ms | about 75% |
  | 1 | 3.21 ms | about 50-60% |

  B=1 depends on three primitive costs that have not been measured inside a kernel yet (see the gate after the first task, W1-1).
- **Architecture: three persistent kernels per decoder layer (ATTN, MOE, TAIL).** Each kernel boundary falls on one of the three NVLink hops that cannot be removed. Each successor launches early (PDL trigger at the start of its predecessor), loads its weights into shared memory while the hop is in flight, and polls the Lamport mailbox instead of calling `griddepcontrol.wait`. That hides the boundary under the hop.
- **A single persistent kernel for all 93 layers is optional (stage 2) and worth about 0.05-0.15 ms.** Three reasons:
  - Its main benefit, cross-layer weight staging, already happens in the per-layer design: MOE thread blocks exit when FC2 finishes, so the next layer's ATTN kernel is resident about 6 us before its input arrives.
  - A megakernel does not make handoffs inside one GPU cheaper. They measure about 1 us under load, and Hazy's B200 megakernel spends about 40% of its time on this kind of activation sync.
  - It adds instruction-cache pressure, a union of register needs across all phases, and a hang that takes down all 16 GPUs.

### Why the targets are physically reasonable

**Our floor is lower than TPU's.**
- At B=1 a layer reads about 111 MB per GPU. At 7.0 TB/s that is 15.9 us/layer, or 1.50 ms per step.
- We currently run at 54.9 us/layer, about 28% of HBM roofline.
- TPU v7 runs at about 41.6 us/layer, about 58% of its own roofline. Its floor is about 24 us/layer: it reads about 90 MB per TensorCore per layer, including gate_up re-encoded from MXFP4 to FP8 (2x bytes), and has about 8 serial communication phases.
- +25% at B=1 needs 32.5 us/layer or less (3.21 ms minus about 0.19 ms outside the layers, over 93 layers). That is about 49% of GB200 roofline: TPU-level efficiency, not better.

**Most of today's 53.0 us KDA layer is software, not physics.**
- NVLink physics accounts for about 3 x 0.75 us.
- The rest:
  - about 9 kernel boundaries on the critical path;
  - consumer work serialized after each boundary;
  - the MoE block's internal chain: x_moe to MoE end is 23.8 us, of which the 17.5 MB expert stream is only about 2.7 us;
  - random cross-rank skew of about 3.7 us/layer.

### Cost model used everywhere below (calibrated to our own measurements)

A KDA layer at B=1 costs about **3h + 4g + W**:
- **h** is one NVLink hop including skew: H1 2.55 us (includes 0.25 us of ingress), H3 2.6 us, H4 2.3 us. Derivation: the no-skew straggler wait in the server is 2.82 us, minus about 0.8 us of consumer work, giving about 2.0 us from last store to remote visibility. Residual skew of about 0.3-0.6 us is added on top.
- **g** is a grid-wide handoff inside one GPU while HBM is loaded: 1.1 us. Measured L2 round trip under load is 0.9-1.1 us; a release counter polled by all blocks costs 0.9 us.
- **W** is weight streaming and compute: about 13.6 us raw, times 1.15 for cold code and slack, about 15.6 us.

That gives about 28 us for a KDA layer and 29.5 us for an MLA layer. Adding 0.19 ms outside the layers gives a **modeled B=1 of 2.85 ms**.

Sensitivities per step at B=1:

| change | cost |
|---|---|
| +1 us per hop (h) | +0.28 ms |
| +1 us per handoff (g) | +0.37 ms |
| +1 us in the expert phase | +0.09 ms |
| +1 us of in_proj | +0.06 ms |

The **risk-adjusted** figures below assume about 75-80% of the modeled saving is realized. Past single-GPU wins have transferred to the server at 50-100%.

### Approaches considered

| Approach | Core idea | Claimed B1/B2 (ms) | Realistic after review (ms) | Decision |
|---|---|---|---|---|
| A. Per-layer persistent kernels | 3 kernels per layer, boundaries on hops, weights staged ahead, Lamport handoffs | 2.6 / 2.9 | 2.9-3.2 / 3.25-3.6 | **Adopt as the skeleton** |
| B. Whole-model megakernel | One launch per step, static per-SM instruction streams | 2.6 / 2.8 | 3.0-3.3 / 3.3-3.7 | Optional stage 2 only. Its cross-layer gain is ≤0.15 ms |
| C. Fewer hops | Fold router and down-proj into the o_proj all-reduce (3 hops), MLA K-split, early shared all-gather | 3.35 / 3.85 | 3.25-3.7 / 3.65-4.2 | Take the FC2-epilogue publish and early shared-expert sends. MLA K-split only behind an accuracy gate. **Reject the fold** (it precomposes weights and can flip routing near-ties) |
| D. Routing-first compute | Staged router, routing known about 2.6 us after the hop, barrier-free FC, staged GEMVs | 2.6 / 3.0 | 2.95-3.3 / 3.25-3.7 | **Adopt its MoE mechanics** |
| E. Floor accounting | 2 kernels per layer, skew control, outside-layer cleanup | 2.7 / 3.2 | 2.95-3.3 / 3.3-3.7 | **Adopt** its accounting, skew work and outside-layer cleanup |
| F. Port of the TPU recipe | Phase-ahead staging including registers, die-aware placement | 3.0 / 3.4 | 3.0-3.4 / 3.4-3.9 | Take the insight. **Reject** register-file staging and `red.global.add` reductions |

Also rejected, with the numbers:
- **Expert parallelism at B≤8.** At B=1 the most-loaded rank reads about 52-70 MB versus 17.5 MB with the current TP16 expert slicing.
- **Replicating the up- or down-projection** to remove a hop: +48 MB/layer, about +6.9 us, to save about 1.3 us.
- **multimem.red and NVLS two-shot reductions.** They are nondeterministic across ranks and cost at least 3 half-trips versus 1 for a Lamport one-shot.
- **Enabling moe8 at M=1 as it stands.** Already A/B-tested: B=1 went from 5.526 to 5.664 ms.

### Projection by stage (ms; central value with range; TPU / TPU+25% shown for reference)

| Stage | B=1 | B=2 | B=4 | B=8 |
|---|---|---|---|---|
| Today (mla2) | 5.37 | 5.90 | 7.12 | 8.87 |
| TPU v7 / +25% | 4.02 / 3.21 | 5.10 / 4.08 | 7.77 / 6.21 | 9.25 / 7.40 |
| Stage 0: incremental wins (1-2 weeks) | 5.05 (4.95-5.15) | 5.55 (5.45-5.65) | 6.85 (6.7-6.95) | 8.65 (8.5-8.8) |
| Stage 1a: TAIL + MOE persistent kernels | 3.6 (3.4-3.9) ✓parity | 4.05 (3.9-4.4) ✓parity, borderline +25% | 5.1 (4.8-5.6) ✓+25% | 7.0 (6.6-7.5) borderline +25% |
| Stage 1b: ATTN persistent kernels (KDA, MLA) | 3.2 (2.9-3.45) | 3.6 (3.3-3.9) ✓+25% | 4.6 (4.2-5.0) | 6.3 (5.8-6.9) ✓+25% |
| Stage 1c: tuning and gated options | 3.05 (2.75-3.35) | 3.45 (3.2-3.8) | 4.4 (4.0-4.9) | 6.0 (5.5-6.7) |
| Stage 2: whole-model kernel (optional) | -0.05 to -0.15 | same | same | same |
| HBM byte floor | 1.50 | 1.73 | 2.19 | 3.07 |

---

## 2. Staged roadmap

### Stage 0: incremental wins on today's stack (1-2 weeks, about 6-8 agent-days)

**What changes**
- **Outside-layer cleanup (W1-6).** 0.263 ms per step sits outside the layers: 5.370 minus 69 x 53.0 minus 24 x 60.4. That is about 90 us of V2-runner input preparation outside the graph, 22 us of lm_head, about 10 us of sampling, and about 133 us not attributed (layer 0 dense MLP, final AttnRes, variance). Fuse or capture the input preparation, and L2-prefetch about 60 MB of lm_head during layer 92's tail. Worth -0.06 to -0.09 ms at every B.
- **Run `agents/moefused/test_dist_tail.py` on 16 ranks**, which has not been run yet. It compares `moe_tail` and `moe_block_tail` against vLLM's tail. Enable whichever wins in the server.
- **If W1-3 (the MoE front) is more than a week out,** add two quick changes:
  - cap the `route_shared` grid so the aux-stream down-shard kernel can co-reside with it;
  - have the last-arriving `route_shared` CTA publish the top-16 ids with a flag, so the MoE producer warp issues FC1 loads without `griddepcontrol.wait`.

  Worth -1.5 to -3 us/layer at M≤4.
- **Skew attribution (inside W1-1).** Find which pre-hop kernel's duration variance creates the 1.8 us p50 MoE-to-tail straggler wait, and apply any cheap fix, such as moving k3pf or changing its grid.

**Projected:** 5.05 / 5.55 / 6.85 / 8.65 ms. This is not enough for parity at B=1 or B=2.

**Risks:** small. The input-prep kernels can only be captured if every rank makes the same decision from replicated state; `step_patch.py` already follows that rule.

**Go/no-go:** none. These ship on A/B plus GSM8K and `check_lp`.

### Stage 1: per-layer persistent kernels ("layer megakernels")

#### Structure per layer L (for uniform decode with M in {1, 2, 4, 8}, specialized per captured graph)

```
TAIL(L-1) ──[H4 up-proj all-gather]──> ATTN(L) ──[H1 o_proj one-shot all-reduce]──> MOE(L) ──[H3 latent all-reduce]──> TAIL(L) ──[H4]──> ATTN(L+1)
                                                                   (H2 down-shard all-gather happens inside MOE, hidden behind router + top-k)
```

**ATTN-KDA** (about 10-11 us at B=1), H4 landing to H1 store:
- The prologue runs while H3, TAIL and H4 are in flight:
  - in_proj rows are claimed dynamically and TMA-staged into smem, so CTAs that land early take more rows;
  - KDA state and conv state are pre-staged;
  - the AttnRes block mix is precomputed.
- Clusters poll the H4 mailbox redundantly. The last reader re-arms it, tracked by a readers-done counter.
- Cluster AttnRes uses a 4-scalar DSMEM exchange, then all-gathers x over DSMEM.
- in_proj: a pure LDS+FFMA2 loop over the staged rows, then the remaining rows from L2 or HBM.
- g handoff to the 6 head clusters, which run kda_split's arithmetic unchanged.
- g handoff to all blocks, then o_proj from staged smem (72 KB/SM), then `multimem.st` into slot[rank] of the H1 mailbox. Blocks exit.

**ATTN-MLA** (about 11-13 us):
- Same front end, then fused_qkv_a_g (41 MB).
- Head clusters, reworked from 16 to 8 blocks per cluster:
  - q_a/kv_a RMSNorm computed redundantly, q_b from staged smem, W_UK absorb;
  - fp8 KV-cache insert. Attention must read the dequantized fp8 own-token entry, with `fence.proxy.async` before any TMA read of it;
  - split-KV attention with a DSMEM combine, then W_UV and the output gate.
- g handoff, then o_proj and H1, as in the KDA kernel.

**MOE** (about 12-13 us at B=1), H1 landing to H3 store:
- **Prologue:** stage router rows (84 KB/SM), down-shard rows (21 KB/SM) and shared gate_up rows (69 KB/SM). They were L2-prefetched during ATTN.
- **H1 consume and routing:**
  - clusters poll H1 redundantly and do a fixed-order 16-slot sum plus post-attention AttnRes;
  - router GEMV, then a logits exchange through an epoch counter (g);
  - an exact top-16 on every CTA (lane-threshold or radix select, about 0.5 us; today it takes 3.6 us under contention);
  - expert TMA loads are issued immediately.
- **Down-shard:** its GEMV and `multimem.st` (H2) run right after the router. The latent lands before the first expert bytes (about +2.6 us versus +2.9 us from x_moe).
- **Experts:** the barrier-free body (W1-4), then a fixed-order expert reduction (g) and a finalized partial latent sent with `multimem.st` into H3.
- **Shared expert:** its down-projection shards are stored directly into each peer's reduce-scatter mailbox, off the critical path.
- **Exit:** blocks exit after FC2, which frees SMs for ATTN(L+1) to stage in_proj.

**TAIL** (32 CTAs as 4 clusters of 8, about 110 KB smem; about 3.9 us including H3):
- Poll the 16 latent slots, sum them in fixed order, and reduce the RMSNorm scalar over DSMEM.
- DSMEM all-gather of the normalized latent.
- Up-proj from staged rows (100 KB/CTA), plus the 16-slot shared reduce-scatter shard.
- `multimem.st` into the H4 mailbox, which is vLLM's up-proj mailbox and is consumed unchanged by `k3tail`.

#### Per-layer budget, KDA at B=1 (us)

| Segment | Today | Stage 1 (raw) | Derivation |
|---|---|---|---|
| MoE end → H3 visible → sum + RMSNorm | 6.84 | 2.6 + 0.7 | hop incl. skew; RMSNorm scalar over DSMEM |
| up-proj + H4 store | 1.70 | 0.6 | 14 rows x 3584 staged in smem |
| H4 → AttnRes output | 3.68 | 2.3 + 0.6 | tailattn post-poll measured at 0.55 |
| in_proj, 46 MB | 6.01 | 3.0 | 20-29 MB staged in smem (0.7 us of LDS); rest at about 11 TB/s from L2/HBM |
| → KDA | 3.94 | 1.1 + 1.2 | g, then math with state already staged |
| → o_proj + H1 store | 2.59 | 1.1 + 0.6 | g; 72 KB/SM from smem |
| H1 → x_moe | 4.45 | 2.55 + 1.0 | 0.25 us ingress (229 KB at 900 GB/s); consumer measured at 0.83 on one GPU |
| router → top-16 → first expert TMA | 3.52, plus about 5.3 inside the MoE kernel | 0.5 + 1.1 + 0.5 | staged GEMV, counter exchange, contention-free select |
| experts, FC1/FC2 | about 14.5 | 4.5 | 0.8 to first byte + 17.5 MB / 6.5 TB/s + compute tail |
| expert reduce + finalize + publish | (inside the tail) | 1.4 | g + finalize |
| **Total** | **53.0** | **25.4 raw → 28.1 planning** | hops 7.45; non-hop 17.9 x 1.15 |

The MLA layer has the same MOE and TAIL, and ATTN rises to 11.6 us raw, giving **29.4 us** (60.4 today). B=1 comes to 69 x 28.1 + 24 x 29.4 + 190 = **2.84 ms modeled**.

B=2 adds, per layer:
- 17.3 MB of experts: +2.5 us;
- M=2 FC compute on tcgen05: +0.5-1.0 us;
- M=2 GEMVs and AttnRes: +0.5 us;
- KDA for 2 tokens: +0.3 us;
- larger hop payloads: +0.3 us.

That is about +5 us/layer after the 1.15 factor, **3.3 ms modeled**.

B=4 and B=8 use a tcgen05 swap-AB dense path and moe8's stream-K expert core:
- B=4 adds about 11.5 us/layer raw: +7.8 experts, +1 dense, +1.15 ingress, +1.5 other. Modeled **4.1 ms**.
- B=8 adds about 26 us/layer: +17.6 experts, +2.8 ingress, +2 dense, +3.5 other. Modeled **5.5 ms**, with HBM at about 4.1 TB/s (60% duty).

#### Design rules (these fix every error the judges raised)

1. **Grid geometry comes from measurement.** Use cudaOccupancyMaxActiveClusters at the real smem and register footprint, taking the minimum over all 16 ranks. Never assume 16-19 clusters of 8. Measured: 7-CTA clusters, 15 of 16 fit; 8-CTA clusters, at least 14 (112 SMs); 16-CTA clusters, at most one per GPC. Assign roles over the CTAs that actually exist. Use dynamic tile claiming so that blocks which land late do not become stragglers.
2. **No kernel may depend on another kernel for progress.** No kernel waits on its successor or on a side stream; PDL concurrency is opportunistic. Move the k3pf L2 prefetch inside the kernels, with at most 32 issuing blocks (at 48 or more, prefetches are silently dropped).
3. **Mailbox discipline:**
   - one reading cluster per fragment, or a readers-done counter where the last reader re-arms;
   - buffers rotate over 3 slots by layer;
   - a release fence between each re-arm and the next publish that lets a peer reuse the buffer;
   - consumers check every 32-bit word, because a 16 B `multimem.st` can be observed torn;
   - producers canonicalize -0.0 and NaN (moe8's 0xFF sentinel collides with NaN).
4. **Same-GPU state.** Because every kernel triggers its successor at entry, kernel boundaries no longer order prefix, blocks, KDA/conv state or KV. Use:
   - 64-bit epoch flags;
   - two-slot buffers for prefix and blocks;
   - writes to in-place state only after the readers-done counter clears.
5. **Numerics:**
   - keep vLLM's bf16 rounding points;
   - fixed-order reductions everywhere;
   - no atomics and no `multimem.red` on replicated data or outputs;
   - top-16 must be bit-identical on all ranks, with a debug-build audit that hashes the ids.
6. **HBM schedule by window:**
   - during ATTN, L2-prefetch MOE's static weights (32.5 MB), throttled to about 4 TB/s;
   - during the expert phase, only expert bytes, loaded with `evict_first`;
   - after FC2 (H3, TAIL, H4, about 6 us, about 41 MB), the next ATTN's static weights;
   - no bulk prefetch just before a hop, because it adds skew.
7. **Safety tooling:**
   - a `%globaltimer` watchdog that traps after at least 2 s in debug builds only (a 20 ms clock64 trap would fire falsely on host launch skew);
   - in production, an error word in host-mapped memory;
   - a trace ring;
   - a split mode that runs each phase as its own kernel, for bisecting;
   - a mailbox audit at every step boundary, since the mailboxes are shared with vLLM's fallback path.
8. **Code layout:** loops, not full unrolls; one kernel per role. moefused measured +1 us from dead tail code in a shared template.
9. **Integration:** per-layer patches controlled by env flags, chained through the existing mailboxes (k3oproj H1, down-shard H2, k3tail H4). Eligibility is a deterministic function of replicated scheduler state. Prefill, mixed batches and M=16 fall back to today's path.

#### Stage 1a: MOE + TAIL (about 30 agent-days, 3-4 weeks with 3-4 agents)

- **Delivered by:** W1-2 (TAIL), W1-3 (MoE front end), W1-4 (barrier-free expert body). Each ships into today's stack on its own; they are then merged into a single MOE kernel. Add M=4/8 variants using the moe8 core.
- **Projected:** 3.6 / 4.05 / 5.1 / 7.0 ms.
- **Risks:**
  - the expert phase stays above 6 us (moe8 has a measured 9.7 us fixed intercept; CUDA-core FP4 decode reaches only 24-32 elements/clk/SM);
  - the tail repeats the fused-tail failure mode, where each phase cost 3-5x its estimate;
  - L2 flooding from redundant polling.
- **Gate G2:**
  - in the server trace, o_proj-produce end to H3 store ≤15 us at B=1 (28.3 today);
  - MoE end to H4 store ≤4.5 us (8.6 today);
  - B=1 ≤3.8 ms, B=2 ≤4.3 ms.

  If the MOE segment is over 18 us, diagnose it before starting 1b.

#### Stage 1b: ATTN-KDA, then ATTN-MLA (about 20 agent-days)

- **Delivered by:** W1-5 (fused H4 consumer + staged in_proj), then head clusters (kda_split / k3mla math), staged o_proj and the H1 publish (oproj_produce code).
- **Projected:** 3.2 / 3.6 / 4.6 / 6.3 ms.
- **Risks:**
  - staged GEMVs have lost to CuTe before (k3gemv -0.24 us at M=1; the gemvin server run at B=4 was 9.33 vs 8.91 ms);
  - the MLA cluster rework from 16 to 8 CTAs;
  - smem budget: 197 KB ring + 48 KB state + 14 KB x does not fit, so the ring must shrink to about 150 KB.
- **Gate G3:** W1-5's chain benchmark shows H4 landing to in_proj output ≤3.0 us. Otherwise keep CuTe in_proj as its own PDL kernel inside ATTN, which costs about +2 us per KDA layer.

#### Stage 1c: tuning and gated options (about 8-10 agent-days)

Items, with their expected effect:
- **Skew control:** fixed work partitions, `evict_first`, and sending FC2 partials block by block. Bringing H3 skew from 1.8 to 0.5 us is worth about -1.3 us/layer.
- **TMEM as in_proj staging:** about -1 us per KDA layer.
- **MLA K-split of the replicated qkv_a** (30.3 MB down to 1.9 MB per rank), behind an accuracy gate: about -2.5 us per MLA layer and -28 MB of HBM traffic.
- **Speculative expert L2 prefetch** from router(x_attn), only if the offline overlap is ≥12 of 16: -1 to -1.5 us/layer at B=1, up to -4 us at B=8.
- **Router logits carried on H2 at B≥4:** -12.8 MB/layer, about -1.8 us when HBM-bound.
- **Two-shot H1 at B=8:** one-shot ingress there is 1.84 MB, about 2 us.

**Projected:** 3.05 / 3.45 / 4.4 / 6.0 ms. At B=1 this is the margin that makes +25% likely rather than a coin flip.

### Stage 2: whole-model persistent kernel (optional, +20-30 agent-days)

- **What it adds:** removes CTA relaunch and prologue per kernel (already hidden under hops, residue about 0.3 us each), keeps next-layer weights across boundaries, and folds in lm_head and sampling. Total about 0.5-1.5 us/layer, 0.05-0.15 ms.
- **Costs:**
  - about 15 handlers competing for instruction cache (0.2-0.5 us per miss chain);
  - the register union across all phases;
  - cooperative launch + clusters + PDL + graph capture is unverified;
  - one protocol bug hangs all 16 GPUs for the whole step.
- **Go only if** after stage 1c B=1 is above 3.2 ms **and** traces show boundary residue plus lost staging ≥1.5 us/layer.

**Total effort through 1c:** about 55-65 agent-days (1.3-1.5x the proposals' estimates), about 6-8 weeks of wall time with 4 parallel agents. The first measurable wins arrive in about 1 week.

---

## 3. Wave-1 tasks (start now, in parallel)

Common rules for all six:
- Each agent works in `gpu-opt/agents/<name>/` on copies of the kernel files with renamed namespaces. The integration owner merges.
- W1-2, W1-3 and W1-4 take over the in-flight moefused work (merging route_shared, fusing the tail epilogue). Assign them to that agent or hand its state over; do not run it in parallel with them.

### W1-1 `probe`: measure the primitives and decide go/no-go (2-3 agent-days)

**Goal:** replace every assumed unit cost with a measured one before any grid shape is fixed.

**Measure, on 16 ranks with `dist_test.sh` plus one GPU:**
- **(a) NVLink hop inside a persistent kernel.** `multimem.st` of 16 B fragments (with -0.0 canonicalized) into a 16-slot mailbox, payloads of 0.9, 7, 14, 28 and 115 KB per rank.
  - Report last-store to poll-visible per rank (p50/p99), with cross-node clock offsets calibrated by ping-pong.
  - Also run a chain of 2000 dependent hops to get us/hop including skew.
  - Vary: 1, 7 or 16 consumer CTAs; polling only invalid fragments versus all; with and without a concurrent 6 TB/s TMA reader.
- **(b) Handoff inside one GPU (g).** 152 producers to 152 consumers.
  - Compare a counter (`red.release.gpu` plus a one-lane `ld.acquire`) against data-as-flag.
  - Idle versus 4 and 6 TB/s of background TMA; same die versus cross die.
- **(c) DSMEM.** `st.async` + mbarrier all-gather of 7 and 14 KB at cluster sizes 4, 8 and 16.
- **(d) Co-residency.** cudaOccupancyMaxActiveClusters for cluster sizes 1/2/4/6/7/8/16 at 1 CTA per SM, 110/200/220 KB smem and 384/512 threads, on all 16 GPUs. Report the minimum.
- **(e) Chaining.** Three dummy persistent kernels (200/200/110 KB) x 93 in a CUDA graph, triggering at entry and polling mailboxes across GPUs.
  - Measure the boundary residue.
  - Check whether cooperative launch + cluster dims + PDL + graph capture is accepted.
- **(f) Instruction cache.** Extend `agents/moe8/icache_probe.py` to measure cold-versus-warm cost per phase for 64-256 KB of SASS interleaved with other kernels.
- **(g) Pending tail test.** Run `agents/moefused/test_dist_tail.py`.
- **(h) Skew attribution.** Correlate the straggler rank with the duration of each pre-hop kernel in `results/mla2/profile-b1` (`skew16.py`).

**Output:** a table of primitives, and the verdict of gate **G1**:

| Quantity | Pass if |
|---|---|
| h (14 KB, p50 incl. skew) | ≤2.6 us |
| g (under 6 TB/s load) | ≤1.3 us |
| 8-CTA clusters co-resident | ≥14 on every rank (otherwise use cluster size 4) |
| boundary residue | ≤0.3 us |

If h is above 3.0 us, B=1 +25% is not reachable with this design: each +0.5 us on three hops costs +0.14 ms. Keep parity at B=1 and +25% at B=2/4/8 as the goals.

**Start from:** `agents/oproj/test_dist.py`, `oproj_ar.cu` (multicast epilogue and consumer), `lamport_attn_res.cu` (`st.async` helpers), `moefused/gather_probe2.py` and `lat_probe.py`, `moe8/icache_probe.py`, `skew16.py`.

### W1-2 `tail2`: TAIL kernel plus the MoE publish epilogue (4-5 agent-days)

**Goal:** in the server trace, p50 from MoE end (last FC2 store) to the up-proj mailbox store ≤4.5 us. Today it is 8.6 us (46.18 → 54.78). There must be no grid-wide sync; the earlier `moe_tail` design lost 4.7 us to a grid gather plus barrier.

**MoE side (publish mode of `moe_block_lamport`):**
- Each CTA already holds all 16 experts' FC2 output for its latent rows. It computes the finalized partial with stock rounding: `bf16(Σ_j fp32 fma(bf16 row_j, bf16 w_j))`, summed in fixed j order.
- It sends that with `multimem.st` into `lat_mb[par][rank]`.
- It stores the shared-down shards into each peer's `rs_mb[par][rank]` as soon as shared down finishes.
- For the moe8 path, TAIL's front end finalizes `gemm2_out` locally instead.

**TAIL kernel:**
- 4 clusters of 8, 110 KB smem or less, PDL trigger at entry.
- Prologue stages 14 up-proj rows per CTA.
- Poll all words, sum the 16 slots in fixed order, reduce sum-of-squares over DSMEM, `rsqrt`, all-gather over DSMEM.
- GEMV from smem, add the 16-slot reduce-scatter shard, sanitize -0.0, `multimem.st` into vLLM's `AdaptiveUpProjectionKernel._mailbox`.
- Readers-done counter across the 4 clusters for re-arm.

**Op signatures:**
- `k3mk::tail(Tensor(a!) lat_mb /*[3,16,Mmax,3584] bf16 symm*/, Tensor(b!) rs_mb /*[3,16,Mmax,448]*/, int par, Tensor w_up /*[448,3584]*/, Tensor gamma, float eps, Tensor(c!) up_mb, int up_mc_ptr, Tensor(d!) epoch /*i64*/, int rank, int M, Tensor(e!)? trace=None) -> ()`
- `moe_block_lamport(..., Tensor(x!)? lat_mb=None, int lat_mc_ptr=0, int[]? rs_peers=None, int par=0)`

**Plugs in:** `moeblock_patch.py` with `K3MOEBLOCK_TAIL=k3mk`. Replaces `allreduce_rmsnorm_reduce_scatter_early_exit` + `fused_add_multicast_*` for M≤8.

**Tests:**
- one GPU emulating rank 5 of 16: 0 ulp against the torch model of vLLM's tail (≤2 ulp allowed from summation order);
- 16 ranks: identical outputs on every rank, mailbox fully re-armed after each call and each graph replay, 10k-iteration soak;
- server A/B with GSM8K-300 and `check_lp`.

**Success:** ≤4.5 us in the server; B=1 -0.3 to -0.4 ms, similar at B=2/4/8.

**Start from:** `moe_small.cu` (`moe_tail`, `moe_block_tail`), `INTEGRATION_TAIL.md`, `test_tail.py`, `test_moe_tail.py`, `test_dist_tail.py`, vLLM `latent_moe_tail/*.py`.

### W1-3 `moefront`: routing-first MoE front end (6-8 agent-days)

**Goal:** x_moe visible to first FC1 TMA issued ≤3.5 us at M=1 and ≤4.0 us at M=2. Today it is about 10.1 us: 22.37 → route_shared → boundary → 3.6 us of top-16 → 1.7 us of issue lag, about 32.5. Also delete the aux-stream down-shard kernel and `route_shared`.

**Design:**
- `k3oproj.consume` stays as a co-resident 7-CTA hop-cluster kernel. It publishes x_moe into a two-slot intra-GPU buffer and releases an epoch counter.
- `moe_block_lamport` gains a front mode with grid 145, so the two co-reside. Its prologue stages router rows, down-shard rows and shared gate_up rows (13 rows/CTA, about 183 KB) into the ring.
- After polling the counter:
  - router GEMV;
  - logits exchange through a counter;
  - exact top-16 on every CTA (a 2-pass 16-bit radix select over 896 scores, ties to the lower id);
  - the producer issues FC1 TMA immediately.
- Down-shard GEMV (224 rows, about 1.5 per CTA) with `multimem.st` into the existing H2 mailbox that `moe_block_lamport` already polls.
- Shared gate_up and SiTU into `h_shared` (existing code).

**Op signature:** `k3moe::moe_block_front(Tensor xmoe_buf, Tensor(a!) epoch, int par, Tensor gate_w, Tensor bias, Tensor down_w_shard /*[224,7168]*/, Tensor(b!) lat_mb, int lat_mc_ptr, Tensor sh_w13, <existing moe_block_lamport args>)`, plus `consume(..., Tensor(x!)? xmoe_buf=None, Tensor(y!)? epoch=None, int par=0)`.

**Tests:**
- `test_topk_mismatch.py`: 0 of 4000 mismatches on real router weights;
- gemm2 rows bit-identical;
- down-shard latent within 1 bf16 ulp of the CuTe kernel;
- the chain harness (`test_block.py`/`bench_lamport.py`) with an emulated H1 producer and a `%globaltimer` phase trace;
- server A/B.

**Success:** ≤3.5 us from x_moe to FC1 issue; B=1 -0.4 to -0.55 ms; B=2 about the same.

**Start from:** `moe_small.cu` (`route_shared`, `moe_block_lamport`), `oproj_ar.cu` (consume), `INTEGRATION_MOEBLOCK.md`, `moeblock_patch.py`.

### W1-4 `moebody`: expert body with no grid barrier (6-8 agent-days)

**Goal:** from routing known (ids in smem) to finalized partial latent published (or gemm2 rows written):

| M | target | today |
|---|---|---|
| 1 | ≤5.5 us | about 12.1 (chain harness) / 13.7 (server) |
| 2 | ≤7.5 us | about 17 |
| 4 | ≤13 us | 19.4 (moe8 standalone) |
| 8 | ≤24 us | 30.2 (moe8 standalone) |

Today's post-FC1 overhead is pure sync: FC1 done 7.63 → barrier 9.68 → h staged 10.40 → end 13.26, i.e. 5.6 us.

**Design options (choose by measurement):**
- **(A) CUDA-core neuron blocks, for M=1.** Each CTA owns (expert, 16- or 32-neuron) units: FC1 gate and up rows plus the matching columns `W2[:, neurons]`, aligned to the scale blocks. h never leaves the CTA. Partials are reduced over DSMEM within the cluster, then a fixed-order reducer applies per-expert bf16 rounding and the weighted sum.
- **(B) tcgen05 (moe8 core), for M≥2 and tried at M=1.** 128-row FC1 tiles with K split across a 2-4 CTA cluster and a DSMEM reduction. SiTU and MXFP8 quantization on the cluster leader, h image broadcast over DSMEM, FC2 row tiles inside the cluster.

Both must avoid moe8's intercept:
- quantize x as soon as the latent lands, not after routing;
- issue all FC1 tiles at routing-known;
- no dedupe pass at M≤2.

**Op signature:** `k3moe::expert_body(Tensor x_or_mailbox, Tensor ids, Tensor wts, Tensor w13, Tensor w13_scale, Tensor w2, Tensor w2_scale, Tensor(a!) ws, Tensor(b!) epoch, Tensor(c!) out, int mode /*gemm2|finalized|publish*/, float beta, float linear_beta, Tensor(d!)? trace) -> ()`. Also provide it as a device function, so it can replace the FC body of `moe_block_lamport` (the `latent_out` routing-only hook already exists) and replace moe8 for M≥2.

**Tests:**
- relative L2 error ≤2e-3 against the fp32 reference;
- routing identical to the current path, gemm2 within bf16 rounding;
- the MXFP8 path matches `trtllm_fp4_block_scale_moe`;
- persistent harness where routing arrives through a flag (moefused `trace.py`, `bench_lamport.py`);
- server A/B.

**Success:** the targets above; B=1 -0.4 ms, B=2 -0.5 ms, B=4/8 -0.4 to -0.6 ms.

**Start from:** `moe_small.cu` (`moe_fused_kernel`), `moe8.cu`, `mma_issue.cuh`, `tc_utils.cuh`, `moefused/bw_tma.py`, `moe8/trace.py`, `test_moe8.py`.

### W1-5 `attnfront`: fused H4 consumer + pre-staged in_proj (5-6 agent-days)

**Goal:** H4 mailbox landing to in_proj output (3216 x M bf16) ≤3.0 us at M=1 and ≤3.5 us at M=2. Today it is about 6.6 us (about 4.8 → 11.39). An MLA variant then covers fused_qkv_a_g.

**Design:**
- One persistent PDL kernel, launched after the up-proj producer (which already triggers early via PTRIGGER), triggering at entry.
- Prologue:
  - AttnRes block-mix precompute;
  - TMA-stage up to about 150-190 KB/SM of in_proj rows, sourced from the lines k3pf already put in L2;
  - dynamic tile claiming.
- Clusters poll the up-proj mailbox and run AttnRes redundantly (4-scalar DSMEM exchange), then all-gather x over DSMEM.
- Pure LDS+FFMA2 over the staged rows, then the rows streamed from L2.
- One bf16 rounding of the fp32 accumulator, as CuTe does.
- One cluster writes prefix/blocks into two-slot buffers.
- kda_split consumes the output unchanged through PDL.

**Op signature:** `k3attn::attnres_inproj(Tensor(a!) up_mailbox, Tensor(b!) prefix, Tensor(c!) blocks, Tensor norm_weight, Tensor qk_weight, Tensor? output_norm_weight, Tensor w_in, Tensor(d!) y, Tensor(e!)? x_out, int num_blocks, int block_write_idx, float eps, float output_norm_eps, Tensor(f!)? trace) -> ()`

**Plugs in:** `tailattn_patch.py`, KDA layers first (replaces `lamport_attn_res` + `CuteSkinnyGemm`), then MLA fused_qkv_a_g.

**Tests:**
- AttnRes output bit-identical to `lamport_attn_res`;
- y within 1 ulp of the CuTe result (summation order);
- chain benchmark (`tailattn/bench_chain.py`, early-trigger producer, weights rotated beyond 129 MB, with and without k3pf);
- then server A/B.

**Gate G3:** must beat `lamport_attn_res` + CuTe in the same chain by at least 2 us. Otherwise stop: this is the ninth staged-GEMV attempt, and the known weak spot is efficiency after the wait.

**Success:** B=1 -0.2 to -0.25 ms; the MLA variant another -0.05 ms.

**Start from:** `lamport_attn_res.cu`, `k3gemv.cu`, `prefetch/timeline.py`, `tailattn/bench_chain.py`, `l2pf/chain.py`.

### W1-6 `stepov`: outside-layer time and expert-predictor data (2-3 agent-days)

**Goal:** cut the time outside the layers from 0.263 ms to ≤0.17 ms at every B, and collect the data that decides speculative expert prefetch.

**Items:**
1. Attribute every piece of GPU work outside the layers in one B=1 step (`critpath.py`, `analyze_trace.py`). That covers the 133 us that is not attributed today.
2. Fuse the V2-runner input-prep kernels into one, or capture them in the FULL graph when every rank makes the same decision from replicated state (about 90 us today).
3. Extend k3pf to L2-prefetch the first 60 MB of lm_head during layer 92's tail.
4. Check that layer 0's dense MLP and the final AttnRes use the fast kernels.
5. Add a debug patch that dumps the pre-attention and post-attention normalized hidden states for 256 decode tokens on rank 0. Report the top-16 overlap of router(x_attn) against router(x_moe), and against the previous token's routing, for every MoE layer.

**Tests:** greedy tokens bit-identical (`check.sh`), then server A/B.

**Success:** -60 to -90 us per step at every B (B=1 ≤5.30 ms on its own), plus a yes/no on speculative prefetch (yes if overlap ≥12 of 16).

**Start from:** `k3opt/step_patch.py`, `k3samp.cu`, `l2pf_patch.py`, `critpath.py`.

**Expected effect of wave 1, shipped into today's stack:** B=1 about 3.7-4.0 ms (TPU parity) and B=2 about 4.1-4.4 ms. Overlap between W1-3 and W1-4 is small: the front end and the body are consecutive segments.

---

## 4. Key facts and corrections

**Floors**
- Per-layer bytes: 111 / 129 / 164 / 231 MB at B = 1 / 2 / 4 / 8, i.e. 15.9 / 18.5 / 23.5 / 33.0 us at 7.0 TB/s. Per step, including lm_head (147 MB): 1.50 / 1.73 / 2.19 / 3.07 ms.
- Distinct experts per layer: 16.0 / 31.7 / 62.3 / 120.3, i.e. 17.5 / 34.8 / 68.3 / **132 MB** of routed weights. The brief's 140-176 MB at B=8 is too high: expert padding is never read.
- The expert stream cannot be prefetched unless experts are predicted. It is a serial term of 2.5 / 5.0 / 9.8 / 18.8 us per layer (at B=8 that alone is 1.75 ms per step).
- Budgets per layer:

  | B | TPU parity | +25% | today |
  |---|---|---|---|
  | 1 | 41.2 us | 32.5 us | 54.9 us |
  | 2 | 52.7 us | 41.7 us | 60.5 us |
  | 4 | — | 64.6 us | 73.6 us |
  | 8 | — | 77.2 us | 92.2 us |

- Time outside the layers is **0.263 ms**, not about 0.1: 5.370 − (69 x 53.0 + 24 x 60.4 = 5.107).

**Hops**
- Three hops per layer are structural and on the critical path: H1 (o_proj reduce before the nonlinear AttnRes), H3 (latent reduce before RMSNorm), H4 (up-proj gather). H2 can be hidden behind router + top-k.
- Removing a hop by replicating a projection costs +48 MB/layer, about +6.9 us, to save about 1.3 us. Expert parallelism does not reduce the hop count.
- **Correction to "17-21 us/layer is hop latency":**
  - raw NVLink one-way latency is about 0.72-0.83 us (GB200 point-to-point round trip 1.60-1.78 us, bimodal by uGPU; the two NVSwitch crossings are about 0.8 us of it);
  - measured in the server, last store to remote visibility is about 2.0 us;
  - the rest of the 17-21 us is kernel boundaries plus consumer work serialized after them.
- **Correction to "no straggler":** the straggler is random and spread uniformly over ranks.
  - Extra wait for the average rank, p50 (p90): moe→tail 1.80 (3.25) us, o_proj 1.00 (1.49) us, upproj→AttnRes 0.86 (1.72) us.
  - That is about 3.7 us/layer, 0.34 ms per step.
- **Protocols:**
  - Lamport one-shot costs 1 half-trip; flag + `membar.sys` costs 3; NVLS `ld_reduce` costs 3-6.
  - `multimem.red` sums in arrival order, so ranks are not bit-identical. It is forbidden for anything that feeds routing.
  - One-shot ingress is 16 x payload: H1 is 0.25 us at B=1 and about 2.0 us at B=8 (900 GB/s per direction).

**Synchronization inside one GPU (measured)**

| Operation | Cost |
|---|---|
| L2 round trip under HBM load | 0.9-1.1 us (unloaded: 150-180 ns local, 377 ns cross-die) |
| Counter-based "all 152 blocks arrived" | 0.9 us |
| Tagged-flag gather | about 10 us |
| Grid barrier | about 2 us |
| `st.async` + mbarrier | 60-100 ns |
| `barrier.cluster` | about 420 ns (compiles to `MEMBAR.ALL.GPU`, about 1000 cycles) |
| `griddepcontrol.wait` release after predecessor exit | about 0.65 us |

- An early PDL trigger saved 1.4-2.4 us per boundary (tailattn: 3.67 → 0.83 us with an early trigger, 5.25 → 2.17 us with a late one).
- **A megakernel does not make these handoffs cheaper.**

**HBM and L2**
- Read-only HBM bandwidth is 7.0-7.6 TB/s; the brief's 6.9 is a copy figure. Pure TMA streaming costs 2 us fixed plus 7.66 TB/s.
- HBM-saturating streams double L2-hit latency for latency-bound kernels running next to them: the predecessor's end moved from 3.4 us to 4.7-7.6 us. Streaming must be scheduled by window.
- L2 prefetch behaviour:
  - it fills only the home slice (250 ns versus 155 ns for a near copy);
  - issue rate is about 250 GB/s per SM;
  - it is silently dropped when 48 or more blocks issue it;
  - `evict_last` is ineffective;
  - after 60 MB of competing traffic, 43% survives;
  - in_proj runs 8.29 us cold, 6.54 prefetched, 5.61 hot.
- Total smem is 34.7 MB; about 26-29 MB of weights could actually be staged before the wait at M≤2.

**Compute**
- CUDA-core FP4 decode peaks at 64 elements/clk/SM and achieves 24-32.
- tcgen05 `kind::mxf8f6f4` reaches about 105 elements/clk/SM (a 128x8x32 MMA in about 39 cycles, bound by the TMA stream).
- Legacy `mma.sync` runs at about 0.15 MMA/clk/SM on sm_100.
- At M=1, CUDA-core FP4 roughly keeps pace with the stream. At M≥2 it is consumer-bound (1.2 us per group versus 0.8 us of data).

**Clusters:** 15 of 16 7-CTA clusters fit; at least 14 8-CTA clusters fit (112 SMs); at most one 16-CTA cluster per GPC. Plans that assumed 16-19 clusters of 8 would deadlock or lose 20-25% of bandwidth.

**Negative results to respect**
- moe8 at M=1 regressed in the server (5.526 → 5.664 ms). Its intercept is about 9.7 us of 12.2 us; it is not limited by the MMA.
- The k3gemv staged GEMV lost to CuTe at M=1 and regressed B=4 (9.33 vs 8.91 ms). The TMA-mode L2 fill regressed (5.437 vs 5.370 ms).
- The fused-tail attempts cost +8.9 us (`moe_block_tail`) and +12.7 us (`moe_tail`) on one GPU. Each phase cost 3-5x its estimate from cold instruction fetch, grid syncs and low warp parallelism.
- Parity-in-data polling cost +4 us and was unsafe.
- **Planning consequence:** apply a 1.15x factor on non-hop work and a 75-80% realization factor on savings.

**TPU v7, verified in `tpu-megakernels/kimi`**
- One Pallas call runs layers 1..92.
- TP32 attention; experts EP8 x TP4.
- MXFP4 gate_up is losslessly re-encoded to FP8 (2x bytes).
- About 8 serial communication phases per layer.
- It wins through no kernel boundaries, staging each phase's weights ahead of time, and computing norms redundantly on every core, not through fewer bytes or fewer hops.

**Correctness hazards the design must close**
- multi-reader re-arm;
- fences between re-arm and publish;
- write-after-read on prefix, blocks and in-place state;
- 32-bit epoch wrap (about 20 hours at 385 steps/s);
- NaN colliding with sentinels;
- torn 16 B multicast loads;
- MLA attention on the own-token fp8 KV;
- a 20 ms `__trap` watchdog firing falsely on host skew;
- ranks disagreeing on geometry or eligibility.