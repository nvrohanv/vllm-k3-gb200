// tail2 (agents/moefused, DESIGN_PLAN W1-2): copy of moe_small.cu under TORCH_LIBRARY namespace k3mk with
//   * moe_block_lamport(..., lat_mb, lat_mc_ptr, rs_peers, par): optional PUBLISH epilogue -- the block
//     kernel finalizes its FC2 columns (top-16 weighted sum, vLLM rounding) and multicasts them into
//     lat_mb[par][rank], stores the shared-expert down rows into every owner's rs_mb[par][rank], and
//     exits (no gemm2_out / shared_out);
//   * tail(...): the TAIL kernel (4 clusters x 8 CTAs): polls the 16 latent slots, fixed-order sum,
//     RMSNorm with a DSMEM reduction, DSMEM all-gather, up-proj from staged rows, reduce-scattered
//     shared add, multimem.st into vLLM's up-proj mailbox; readers-done counter for the re-arm.
// The production kernels (moe_fused_kernel etc.) are byte-identical to agents/moefused/moe_small.cu.
// moefused v2 (agents/moefused): k3moe.moe_fused / k3moe.moe_fused_unfinalized rewritten as a
// one-CTA-per-SM TMA-pipelined kernel (M <= 8); k3moe.moe_small unchanged. See RESULTS.md.
// Small-batch (decode) MXFP4 MoE for Kimi K3 at TP16, reading vLLM's TRT-LLM
// weight layout in place and skipping its zero padding.
//
// Per rank, each routed expert holds a 192-wide slice of the 3072 intermediate
// dim, padded by vLLM to 256 for TRT-LLM's tiles. Layout per expert (see
// vllm/model_executor/layers/fused_moe/oracle/mxfp4.py,
// convert_weight_to_mxfp4_moe_kernel_format, TRTLLM branch):
//   w13 [512, 1792] u8: logical rows interleaved [up_0, gate_0, up_1, gate_1,
//       ...], then permuted within 32-row blocks: physical p holds logical
//       32*(p/32) + 4*(p%8) + (p%32)/8. Logical rows >= 384 (j >= 192) are
//       padding and live in physical rows 384..511.
//   w13 scales: UE8M0 per 32 elements, [512, 112], rows permuted like w13,
//       then 128x4-swizzled.
//   w2  [3584, 128] u8 (rows permuted the same way; K=256 padded, only the
//       first 96 bytes / 6 scale columns of each row are real), scales
//       [3584, 8] 128x4-swizzled.
// FP4 is E2M1 with the even element in the low nibble.
//
// Kernel 1: h[t, j, i] = situ(gate_i . x_t, up_i . x_t) for the expert
//           routed at (t, j); one warp per (t, j, i).
// Kernel 2: out[t, n] = sum_j w[t, j] * (W2_e[n, :192] . h[t, j, :]).
#include <algorithm>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

namespace {

constexpr int kHidden = 3584;         // latent dim (FC1 K, FC2 N)
constexpr int kInter = 192;           // real intermediate slice per rank
constexpr int kInterPad = 256;        // vLLM-padded slice
constexpr int kW13Rows = 2 * kInterPad;
constexpr int kW13RowBytes = kHidden / 2;       // 1792
constexpr int kW13ScaleCols = kHidden / 32;     // 112
constexpr int kW2RowBytes = kInterPad / 2;      // 128
constexpr int kW2ScaleCols = kInterPad / 32;    // 8
constexpr int kTopK = 16;

__device__ __forceinline__ int physical_row(int logical) {
  const int q = logical & 31;
  return (logical & ~31) + (q & 3) * 8 + (q >> 2);
}

// Offset of scale (row r [physical], col c) in a 128x4-swizzled [R, C] matrix.
__device__ __forceinline__ int swizzled_scale(int r, int c, int col_tiles) {
  return ((r >> 7) * col_tiles + (c >> 2)) * 512 + (r & 31) * 16 + ((r & 127) >> 5) * 4 + (c & 3);
}

__device__ __forceinline__ float ue8m0(uint8_t e) {
  return __uint_as_float(static_cast<uint32_t>(e) << 23);
}

// FP16 dot of one 32-element block: 16 bytes of FP4 weights against 16 half2
// activations in registers. Products are accumulated in half2 within the
// block (native e2m1x2->f16x2 cvt + HFMA2) and returned as fp32; blocks are
// combined in fp32 by the caller after applying the UE8M0 scale. Rounding here
// is far below the MXFP8 activation/intermediate rounding TRT-LLM applies.
// Four e2m1x2 -> f16x2 conversions of one 32-bit word. Splitting the word with
// mov.b32/mov.b16 lets ptxas feed the byte lanes to the cvt directly instead of
// materialising every byte with shifts (which made the int pipe the bottleneck).
__device__ __forceinline__ void cvt_e2m1_word(uint32_t w, uint32_t (&r)[4]) {
  asm("{\n\t.reg .b16 lo, hi;\n\t.reg .b8 b0, b1, b2, b3;\n\t"
      "mov.b32 {lo, hi}, %4;\n\tmov.b16 {b0, b1}, lo;\n\tmov.b16 {b2, b3}, hi;\n\t"
      "cvt.rn.f16x2.e2m1x2 %0, b0;\n\tcvt.rn.f16x2.e2m1x2 %1, b1;\n\t"
      "cvt.rn.f16x2.e2m1x2 %2, b2;\n\tcvt.rn.f16x2.e2m1x2 %3, b3;\n\t}"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(w));
}

__device__ __forceinline__ float dot32_h(uint4 w, const __half2 (&x)[16]) {
  const uint32_t words[4] = {w.x, w.y, w.z, w.w};
  __half2 acc0 = __float2half2_rn(0.f), acc1 = acc0;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    uint32_t r[4];
    cvt_e2m1_word(words[k], r);
#pragma unroll
    for (int b = 0; b < 4; b += 2) {
      acc0 = __hfma2(*reinterpret_cast<const __half2*>(&r[b]), x[k * 4 + b], acc0);
      acc1 = __hfma2(*reinterpret_cast<const __half2*>(&r[b + 1]), x[k * 4 + b + 1], acc1);
    }
  }
  const float2 f = __half22float2(__hadd2(acc0, acc1));
  return f.x + f.y;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

// Load a lane's 32-element block of bf16 activations from shared memory as half2.
__device__ __forceinline__ void load_block_h(const __nv_bfloat16* src, __half2 (&x)[16]) {
#pragma unroll
  for (int v = 0; v < 4; ++v) {
    const uint4 raw = reinterpret_cast<const uint4*>(src)[v];
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int q = 0; q < 4; ++q) x[v * 4 + q] = __float22half2_rn(__bfloat1622float2(p[q]));
  }
}

constexpr int kWarps = 8;
constexpr int kFc1Chunks = (kW13ScaleCols + 31) / 32;  // 4 (lanes >= 16 do 3)
constexpr int kFc1Tasks = kTopK * kInter;               // (expert slot j, intermediate i) per token

struct Fc1Load {
  uint4 wu[kFc1Chunks], wg[kFc1Chunks];
  uint8_t su[kFc1Chunks], sg[kFc1Chunks];
};

__device__ __forceinline__ void fc1_issue(Fc1Load& L, const uint8_t* w13, const uint8_t* w13_scale,
                                          int e, int i, int lane) {
  const uint8_t* w = w13 + static_cast<long>(e) * kW13Rows * kW13RowBytes;
  const uint8_t* s = w13_scale + static_cast<long>(e) * kW13Rows * kW13ScaleCols;
  const int up_row = physical_row(2 * i), gate_row = physical_row(2 * i + 1);
#pragma unroll
  for (int k = 0; k < kFc1Chunks; ++k) {
    const int c = lane + 32 * k;
    if (c < kW13ScaleCols) {
      L.wu[k] = __ldg(reinterpret_cast<const uint4*>(w + up_row * kW13RowBytes + c * 16));
      L.wg[k] = __ldg(reinterpret_cast<const uint4*>(w + gate_row * kW13RowBytes + c * 16));
      L.su[k] = __ldg(s + swizzled_scale(up_row, c, kW13ScaleCols / 4));
      L.sg[k] = __ldg(s + swizzled_scale(gate_row, c, kW13ScaleCols / 4));
    }
  }
}

// grid: (G, M). Warps of token t grid-stride over its 16*192 (j, i) tasks; each
// task computes one up/gate row pair and writes h[t, j, i] = situ(gate, up) (fp16).
__global__ void __launch_bounds__(kWarps * 32)
fc1_situ_kernel(const __nv_bfloat16* __restrict__ x, const int* __restrict__ topk_ids,
                const uint8_t* __restrict__ w13, const uint8_t* __restrict__ w13_scale,
                __half* __restrict__ h, float beta, float linear_beta) {
  __shared__ __align__(16) __nv_bfloat16 xs[kHidden];
  __shared__ int es[kTopK];
  const int t = blockIdx.y;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");
  for (int v = threadIdx.x; v < kHidden / 8; v += blockDim.x)
    reinterpret_cast<uint4*>(xs)[v] = reinterpret_cast<const uint4*>(x + t * kHidden)[v];
  if (threadIdx.x < kTopK) es[threadIdx.x] = topk_ids[t * kTopK + threadIdx.x];
  __syncthreads();

  __half2 xr[kFc1Chunks][16];
#pragma unroll
  for (int k = 0; k < kFc1Chunks; ++k) {
    const int c = lane + 32 * k;
    if (c < kW13ScaleCols) load_block_h(xs + c * 32, xr[k]);
  }

  const int stride = gridDim.x * kWarps;
  int task = blockIdx.x * kWarps + warp;
  if (task >= kFc1Tasks) return;
  Fc1Load cur;
  fc1_issue(cur, w13, w13_scale, es[task / kInter], task % kInter, lane);
  while (true) {
    const int next = task + stride;
    Fc1Load nxt;
    if (next < kFc1Tasks) fc1_issue(nxt, w13, w13_scale, es[next / kInter], next % kInter, lane);
    float up = 0.f, gate = 0.f;
#pragma unroll
    for (int k = 0; k < kFc1Chunks; ++k) {
      if (lane + 32 * k < kW13ScaleCols) {
        up = fmaf(ue8m0(cur.su[k]), dot32_h(cur.wu[k], xr[k]), up);
        gate = fmaf(ue8m0(cur.sg[k]), dot32_h(cur.wg[k], xr[k]), gate);
      }
    }
    up = warp_sum(up);
    gate = warp_sum(gate);
    if (lane == 0) {
      const float sig = 1.f / (1.f + __expf(-gate));
      const float g = beta * tanhf(gate / beta) * sig;
      const float u = linear_beta > 0.f ? linear_beta * tanhf(up / linear_beta) : up;
      h[static_cast<long>(t) * kFc1Tasks + task] = __float2half_rn(g * u);
    }
    if (next >= kFc1Tasks) break;
    task = next;
    cur = nxt;
  }
}

constexpr int kPairs = kTopK * (kInter / 32);  // 96 (expert, K-block) pairs per output row
constexpr int kFc2PerLane = kPairs / 32;       // 3

// grid: (G, M). Warps of token t grid-stride over the 3584 output rows; each
// lane owns 3 (expert, K-block) pairs and keeps their h activations in registers.
__global__ void __launch_bounds__(kWarps * 32)
fc2_finalize_kernel(const __half* __restrict__ h, const int* __restrict__ topk_ids,
                    const float* __restrict__ topk_w, const uint8_t* __restrict__ w2,
                    const uint8_t* __restrict__ w2_scale, __nv_bfloat16* __restrict__ out) {
  __shared__ __align__(16) __half hs[kFc1Tasks];
  __shared__ long bases[kTopK];
  __shared__ float wsc[kTopK];
  const int t = blockIdx.y;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");
  for (int v = threadIdx.x; v < kFc1Tasks / 8; v += blockDim.x)
    reinterpret_cast<uint4*>(hs)[v] = reinterpret_cast<const uint4*>(h + static_cast<long>(t) * kFc1Tasks)[v];
  if (threadIdx.x < kTopK) {
    bases[threadIdx.x] = static_cast<long>(topk_ids[t * kTopK + threadIdx.x]) * kHidden;
    wsc[threadIdx.x] = topk_w[t * kTopK + threadIdx.x];
  }
  __syncthreads();

  __half2 hr[kFc2PerLane][16];
  long base[kFc2PerLane];
  float wj[kFc2PerLane];
  int cblk[kFc2PerLane];
#pragma unroll
  for (int k = 0; k < kFc2PerLane; ++k) {
    const int q = lane + 32 * k, j = q / (kInter / 32), c = q % (kInter / 32);
    const uint4* src = reinterpret_cast<const uint4*>(hs + j * kInter + c * 32);
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      const uint4 raw = src[v];
      const __half2* p = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
      for (int u = 0; u < 4; ++u) hr[k][v * 4 + u] = p[u];
    }
    base[k] = bases[j];
    wj[k] = wsc[j];
    cblk[k] = c;
  }

  for (int n = blockIdx.x * kWarps + warp; n < kHidden; n += gridDim.x * kWarps) {
    const int p = physical_row(n);
    uint4 wv[kFc2PerLane];
    uint8_t sc[kFc2PerLane];
#pragma unroll
    for (int k = 0; k < kFc2PerLane; ++k) {
      wv[k] = __ldg(reinterpret_cast<const uint4*>(w2 + (base[k] + p) * kW2RowBytes + cblk[k] * 16));
      sc[k] = __ldg(w2_scale + base[k] * kW2ScaleCols + swizzled_scale(p, cblk[k], kW2ScaleCols / 4));
    }
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < kFc2PerLane; ++k) acc = fmaf(wj[k] * ue8m0(sc[k]), dot32_h(wv[k], hr[k]), acc);
    acc = warp_sum(acc);
    if (lane == 0) out[t * kHidden + n] = __float2bfloat16(acc);
  }
}

// ---------------------------------------------------------------------------
// Fused persistent MoE (k3moe.moe_fused / k3moe.moe_fused_unfinalized), one CTA
// per SM, M <= 8 tokens. All weight/scale bytes move with TMA into a smem ring.
//
// Work split (CTA s of G):
//   FC1: the M*16*192 (token, expert slot, intermediate) row pairs are cut into
//        G contiguous ranges. A range is consumed as "groups" = intersections
//        with 8-pair blocks; a group of n pairs is n up rows + n gate rows, each
//        run contiguous in memory (physical rows 16b+r0.. and 16b+8+r0..) ->
//        two 1D bulk copies, plus one 4D tensor copy of the 128x4-swizzled
//        scales of physical rows 16b..16b+15 (all 4 row groups, 7 KB).
//   FC2: the 448 physical-row octets are cut into G contiguous ranges; per
//        token an octet is a "unit": 16 experts x (8 rows x 96 B weights via a
//        2D tensor copy + 256 B of swizzled scales via a 4D tensor copy).
//        Two units per ring stage.
// Warp 14 is the producer: after griddepcontrol.wait it reads the routing and
// issues every stage (FC1 groups, then FC2 units) as ring slots free up, so FC2
// loads overlap FC1 compute and the grid barrier. Warps 0..13 consume.
//   FC1: warp w owns K-chunks 8w..8w+7 (lane&7), a lane owns every 4th row of a
//        group (lane>>3, up to 4 rows at once), x in registers; row partials are
//        transpose-reduced over the 8 lanes and added into smem with atomics.
//   FC2: a warp task is (stage, expert j): lane = (unit, row, half) with 3 of
//        the 6 K-chunks; h is read from smem as warp-broadcasts.
// Between them a grid barrier (monotonic counter in barrier[M-1], never reset:
// the tensor must be zero-initialised once and only used by these ops).

constexpr int kFc1Warps = 14;                          // FC1: 14 warps x 8 K-chunks
constexpr int kConsumerWarps = 16;                     // FC2: one warp per expert slot
constexpr int kFusedThreads = (kConsumerWarps + 1) * 32;
constexpr int kConsumerThreads = kConsumerWarps * 32;
constexpr int kPairsPerTok = kTopK * kInter;           // 3072
constexpr int kOctets = kHidden / 8;                   // 448
constexpr int kStageW = 16 * kW13RowBytes;             // 28672: 16 FC1 rows
constexpr int kStageBytes = kStageW + 28 * 16 * 16;    // + 7168 scale bytes = 35840
constexpr int kF2W = kTopK * 8 * 96;                   // 12288 weight bytes per unit
constexpr int kF2Unit = kF2W + kTopK * 256;            // + 4096 scale bytes
constexpr int kMaxFusedM = 8;
constexpr int kFlagWaitPrior = 2;  // Lamport variant: producer warp waits for the PDL predecessor
static_assert(kConsumerWarps == kTopK, "FC2 maps warp -> expert slot");
static_assert(kFc1Warps * 8 == kW13ScaleCols, "consumer warps x 8 chunks must cover K");
static_assert(2 * kF2Unit <= kStageBytes, "FC2 stage overflow");

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_arrive_cnt(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred P1;\n\tLAB_WAIT:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
      "@P1 bra DONE;\n\tbra LAB_WAIT;\n\tDONE:\n\t}" ::"r"(bar),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void tma_1d(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
      "l"(src), "r"(bytes), "r"(bar)
      : "memory");
}
__device__ __forceinline__ void tma_2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], "
      "[%4];" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(bar)
      : "memory");
}
__device__ __forceinline__ void tma_3d(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2,
                                       uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], "
      "[%5];" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(bar)
      : "memory");
}
__device__ __forceinline__ void tma_4d(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2, int c3,
                                       uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, "
      "%5}], [%6];" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(bar)
      : "memory");
}
__device__ __forceinline__ void consumer_sync() {
  asm volatile("bar.sync 1, %0;" ::"n"(kConsumerThreads) : "memory");
}

// Per-CTA work description (identical in every thread).
struct FusedWork {
  int qa, qb;      // FC1 pair range [qa, qb) over (t, j, u)
  int ngroups;     // FC1 groups (8-pair-block intersections)
  int oa, nocts;   // FC2 octet range [oa, oa + nocts)
  int nunits;      // M * nocts
  int nf2;         // FC2 stages (2 units each)
};

__device__ __forceinline__ FusedWork fused_work(int M) {
  FusedWork w;
  const int G = gridDim.x, s = blockIdx.x;
  const int P = M * kPairsPerTok;
  w.qa = static_cast<int>(static_cast<long>(P) * s / G);
  w.qb = static_cast<int>(static_cast<long>(P) * (s + 1) / G);
  w.ngroups = w.qb > w.qa ? ((w.qb - 1) >> 3) - (w.qa >> 3) + 1 : 0;
  w.oa = kOctets * s / G;
  w.nocts = kOctets * (s + 1) / G - w.oa;
  w.nunits = M * w.nocts;
  w.nf2 = (w.nunits + 1) / 2;
  return w;
}

// SiTU-GLU on one (up, gate) pair.
__device__ __forceinline__ float situ(float up, float gate, float beta, float linear_beta) {
  const float sig = 1.f / (1.f + __expf(-gate));
  const float g = beta * tanhf(gate / beta) * sig;
  const float u = linear_beta > 0.f ? linear_beta * tanhf(up / linear_beta) : up;
  return g * u;
}

// Order-preserving float -> uint32 (larger float -> larger key).
__device__ __forceinline__ uint32_t order_key(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// ---- MoE publish epilogue + TAIL (DESIGN_PLAN W1-2) -----------------------------------------
// Mailboxes (bf16, Lamport: an empty 32-bit word is 0x80000000, producers never write bf16 -0.0):
//   lat_mb [3 par][16 src][Mmax][3584]  finalized routed partial latent of rank src (multicast by its MoE)
//   rs_mb  [3 par][16 src][Mmax][448]   rank src's shared-expert partial, this rank's 448 columns (peer st)
//   up_mb  vLLM's AdaptiveUpProjectionKernel._mailbox [1, >=M, 7168] (consumed by k3tail / LamportCopy)
// par rotates over 3 buffers (by layer). lat_mb fragments are read by the 4 TAIL clusters; the last one
// (readers-done counter) re-arms them. rs_mb words have one reader, which re-arms right after reading.
constexpr int kTp = 16;                 // TP ranks
constexpr int kUpShard = 7168 / kTp;    // 448 hidden rows per rank
constexpr int kUpRowBytes = kHidden * 2;  // 7168
constexpr int kPubMaxM = 4;             // publish mode of the block kernel (M <= 4)
constexpr int kPollBackoffNs = 64;      // __nanosleep between failed polls

struct TailArgs {                       // publish-mode arguments of moe_block_lamport
  unsigned long long lat_st;            // multicast (or plain) address of lat_mb[0][0][0][0]
  const __nv_bfloat16* lat_mb;          // non-null: publish mode
  unsigned long long rs_peer[kTp];      // rank d's rs_mb base address
  int par, mmax, rank, mc;              // mc: 1 = multimem.st (NVLS) for the latent, 0 = plain st (tests)
};

// Shared-memory map of the publish extras (bytes from a 16-aligned base), identical on host and device.
struct TailLayout {
  int nu, sdr;                          // max FC2 units, shared-down rows per CTA
  int fin, sdo, args, bytes;
};
__host__ __device__ inline TailLayout tail_layout(int M, int G) {
  TailLayout L;
  L.nu = M * ((kHidden / 8 + G - 1) / G);
  L.sdr = 8 * ((kHidden * 2 / 8 + G - 1) / G);
  int o = 0;
  L.fin = o;    o += L.nu * kTopK * 16;          // bf16 [unit][16 experts][8]
  L.sdo = o;    o += (M * L.sdr * 2 + 15) & ~15; // bf16 shared-down rows [t][row]
  L.args = o;   o += (sizeof(TailArgs) + 15) & ~15;
  L.bytes = o;
  return L;
}

__device__ __forceinline__ uint32_t no_neg_zero(uint32_t w) {  // bf16 -0.0 -> +0.0 (both halves)
  if ((w & 0xffffu) == 0x8000u) w &= 0xffff0000u;
  if ((w >> 16) == 0x8000u) w &= 0x0000ffffu;
  return w;
}
__device__ __forceinline__ uint4 no_neg_zero4(uint4 v) {
  return make_uint4(no_neg_zero(v.x), no_neg_zero(v.y), no_neg_zero(v.z), no_neg_zero(v.w));
}
__device__ __forceinline__ bool lamport_dirty(const uint4& v) {
  return v.x == 0x80000000u || v.y == 0x80000000u || v.z == 0x80000000u || v.w == 0x80000000u;
}
__device__ __forceinline__ uint4 ld_volatile16(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint4 ld_relaxed16(const void* p) {  // gpu-scope poll (intra-GPU data)
  uint4 v;
  asm volatile("ld.relaxed.gpu.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_sentinel16(const void* p) {
  asm volatile("st.global.v4.u32 [%0], {%1, %1, %1, %1};" ::"l"(p), "r"(0x80000000u) : "memory");
}
__device__ __forceinline__ void st_mb16(unsigned long long addr, const uint4& v, int mc) {
  if (mc)
    asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(addr), "r"(v.x), "r"(v.y),
                 "r"(v.z), "r"(v.w) : "memory");
  else
    asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}
__device__ __forceinline__ void st_mb4(unsigned long long addr, uint32_t w, int mc) {
  if (mc)
    asm volatile("multimem.st.relaxed.sys.global.b32 [%0], %1;" ::"l"(addr), "r"(w) : "memory");
  else
    asm volatile("st.global.b32 [%0], %1;" ::"l"(addr), "r"(w) : "memory");
}
__device__ __forceinline__ float bf16_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

// Extra inputs/outputs of the whole-MoE-block variant (moe_block_lamport).
struct BlockArgs {
  const uint2* scores;            // [M][896] (sigmoid(logit), sigmoid(logit) + bias) from route_shared
  int* ids_out;                   // [M][16] routing written by CTA 0 (for the MoE tail)
  __nv_bfloat16* wts_out;         // [M][16]
  const __nv_bfloat16* h_sh;      // [M][768] shared-expert gate_up output (route_shared), SiTU applied here
  const __nv_bfloat16* w_sd;      // [7168][384] shared down_proj shard
  __nv_bfloat16* sh_out;          // [M][7168] shared-expert partial output
  float rscale;                   // routed_scaling_factor
  int renorm;
  float sh_beta, sh_lbeta;        // shared-expert SiTU betas (lbeta <= 0: none)
  __nv_bfloat16* latent_out;      // non-null: routing-only mode (no FC1/FC2): CTA 0 copies the latent
                                  // [M][3584] out of the mailbox (and re-arms it) for another MoE kernel
  TailArgs tail;                  // publish mode only (moe_fused_pub_kernel)
};
constexpr int kBlockMaxM = 4;
constexpr int kTopkWarp0 = 16 - kBlockMaxM;  // warps 12..15 route tokens 0..3
constexpr int kSdRows = kHidden * 2;  // 7168 shared down rows (hidden)
constexpr int kSdK = 384;             // shared intermediate per rank
constexpr int kSdRowBytes = kSdK * 2; // 768
constexpr int kTopkScratch = 3 * 64 + 1;  // candidates (32 + per-lane dump) x (id, key, score) + counter

// Exact top-16 of 896 (score + bias) for one token, one warp. Values -> 1024 bins over [min,max];
// the bins from the one holding the 16th largest upward are the candidates (ranked exactly,
// ties -> lower id); > 32 candidates falls back to 16 rounds of warp argmax.
// Writes sel_i[0..15] (descending) and sel_s[0..15] (sigmoid scores).
__device__ __forceinline__ void warp_top16(const uint2* __restrict__ row, unsigned* hist, int* sel_i,
                                           float* sel_s, int lane, unsigned long long* tr = nullptr) {
#define TK_MARK(slot)                                                             \
  if (tr && lane == 0) {                                                          \
    unsigned long long ts;                                                        \
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts));                       \
    tr[slot] = ts;                                                                \
  }
  TK_MARK(0)
  float sb[28], sc[28];
#pragma unroll
  for (int j = 0; j < 28; ++j) {
    uint2 v;
    asm volatile("ld.global.cg.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(row + lane + 32 * j));
    sc[j] = __uint_as_float(v.x);
    sb[j] = __uint_as_float(v.y);
  }
  // Candidate threshold T = 16th largest of the 32 lane maxima: >= 16 lanes hold a value >= T,
  // so the 16th largest overall is >= T and every top-16 element is a candidate (v >= T).
  float lm = sb[0];
#pragma unroll
  for (int j = 1; j < 28; ++j) lm = fmaxf(lm, sb[j]);
  TK_MARK(1)
  int lrank = 0;
#pragma unroll
  for (int d = 0; d < 32; ++d) {
    const float o = __shfl_sync(0xffffffffu, lm, d);
    lrank += (o > lm) || (o == lm && d < lane);
  }
  const int src = __ffs(__ballot_sync(0xffffffffu, lrank == 15)) - 1;
  const float T = __shfl_sync(0xffffffffu, lm, src < 0 ? 0 : src);
  unsigned m = 0;
#pragma unroll
  for (int j = 0; j < 28; ++j) m |= (sb[j] >= T ? 1u : 0u) << j;
  const unsigned cntl = __popc(m);
  unsigned excl = cntl;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const unsigned y = __shfl_up_sync(0xffffffffu, excl, o);
    if (lane >= o) excl += y;
  }
  const unsigned ncand = __shfl_sync(0xffffffffu, excl, 31);
  TK_MARK(2)
  int* ci = reinterpret_cast<int*>(hist);
  uint32_t* ck = reinterpret_cast<uint32_t*>(ci + 64);
  float* cs = reinterpret_cast<float*>(ck + 64);
  if (tr && lane == 0) tr[5] = ncand;
  if (src >= 0 && ncand <= 32) {
    // Branch-free compaction: non-candidates go to a per-lane dump slot (divergent branches cost
    // ~150 cycles each here).
    unsigned pos = excl - cntl;
#pragma unroll
    for (int j = 0; j < 28; ++j) {
      const unsigned bit = m >> j & 1u;
      const unsigned dst = bit ? pos : 32u + lane;
      ci[dst] = lane + 32 * j;
      ck[dst] = order_key(sb[j]);
      cs[dst] = sc[j];
      pos += bit;
    }
    __syncwarp();
    TK_MARK(3)
    const int i = lane < static_cast<int>(ncand) ? ci[lane] : 0x7fffffff;
    const uint32_t k = lane < static_cast<int>(ncand) ? ck[lane] : 0u;
    int rank = 0;
    for (int d = 0; d < static_cast<int>(ncand); ++d) {
      const uint32_t kd = __shfl_sync(0xffffffffu, k, d);
      const int id = __shfl_sync(0xffffffffu, i, d);
      rank += (kd > k) || (kd == k && id < i);
    }
    if (lane < static_cast<int>(ncand) && rank < 16) {
      sel_i[rank] = i;
      sel_s[rank] = cs[lane];
    }
  } else {
    // Exact slow path (many near-equal values): 16 rounds of warp argmax (key desc, id asc).
    unsigned taken = 0;
    for (int r = 0; r < 16; ++r) {
      uint32_t bk = 0;
      int bj = -1;
#pragma unroll
      for (int j = 0; j < 28; ++j)
        if (!(taken >> j & 1u)) {
          const uint32_t k = order_key(sb[j]);
          if (bj < 0 || k > bk) {
            bk = k;
            bj = j;
          }
        }
      int bi = bj < 0 ? 0x7fffffff : lane + 32 * bj;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const uint32_t ok2 = __shfl_xor_sync(0xffffffffu, bk, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ok2 > bk || (ok2 == bk && oi < bi)) {
          bk = ok2;
          bi = oi;
        }
      }
      if ((bi & 31) == lane) {
        const int jb = bi >> 5;
        taken |= 1u << jb;
        float sv = 0.f;
#pragma unroll
        for (int j = 0; j < 28; ++j)
          if (j == jb) sv = sc[j];
        sel_i[r] = bi;
        sel_s[r] = sv;
      }
    }
  }
  __syncwarp();
  TK_MARK(4)
#undef TK_MARK
}

// Debug/test harness: one warp runs warp_top16 on scores [896] (timing in globaltimer ns).
__global__ void topk_debug_kernel(const uint2* scores, int* ids, float* sc, unsigned long long* tr) {
  __shared__ unsigned scr[kTopkScratch];
  __shared__ int si[16];
  __shared__ float ss[16];
  warp_top16(scores, scr, si, ss, threadIdx.x, tr);
  if (threadIdx.x < 16) {
    ids[threadIdx.x] = si[threadIdx.x];
    sc[threadIdx.x] = ss[threadIdx.x];
  }
}

template <bool kFinal, bool kLamport, bool kBlock = false>
__global__ void __launch_bounds__(kFusedThreads, 1)
moe_fused_kernel(const __grid_constant__ CUtensorMap tm_w13s, const __grid_constant__ CUtensorMap tm_w2,
                 const __grid_constant__ CUtensorMap tm_w2s, const __nv_bfloat16* __restrict__ x,
                 const int* __restrict__ topk_ids, const float* __restrict__ topk_w,
                 const uint8_t* __restrict__ w13, __half* __restrict__ h,
                 unsigned long long* __restrict__ barrier, __nv_bfloat16* __restrict__ out, int M,
                 int nstages, float beta, float linear_beta, int flags,
                 unsigned long long* __restrict__ trace, BlockArgs ba) {
  // kLamport: x is the local view of a Lamport mailbox (rows [0, M) of 3584 bf16;
  //        an empty 32-bit word is 0x80000000). CTAs poll their rows instead of TMA-loading x,
  //        and CTA 0 re-arms rows [0, M) after the grid barrier (all CTAs have read x by then).
  //        kFlagWaitPrior -> the producer warp still executes griddepcontrol.wait before reading
  //        topk_ids (needed when routing comes from the PDL predecessor).
#define K3_MARK(slot)                                                                 \
  if (trace) {                                                                        \
    unsigned long long ts;                                                            \
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts));                           \
    trace[blockIdx.x * 32 + (slot)] = ts;                                             \
  }
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  FusedWork W = fused_work(M);
  const bool routed_only = kBlock && ba.latent_out != nullptr;
  if (routed_only) {  // routing + shared expert only: no routed FC1/FC2 work in this kernel
    W.qa = W.qb = 0;
    W.ngroups = 0;
    W.nunits = 0;
    W.nf2 = 0;
  }
  // Shared memory map: x (bf16, later fp16 h) | red | acc2 | wsm | barriers | ring.
  const int x_bytes = M * kHidden * 2;
  uint8_t* xs = smem;
  const int max_slots = 2 * ((M * kPairsPerTok + gridDim.x - 1) / gridDim.x) + 16;
  const int max_rows2 = M * 8 * ((kOctets + gridDim.x - 1) / gridDim.x);
  int so = x_bytes;
  float* red = reinterpret_cast<float*>(smem + so);
  // Deterministic reductions: red = per-FC1-warp slot partials [kFc1Warps][max_slots] (summed in warp order
  // in the FC1 epilogue); acc2 (kFinal) = per-expert-slot rows [kTopK][max_rows2] (summed in slot order).
  float* acc2 = red + kFc1Warps * max_slots;
  float* wsm = acc2 + (kFinal ? kTopK : 1) * max_rows2;
  so += (kFc1Warps * max_slots + (kFinal ? kTopK : 1) * max_rows2 + M * kTopK) * 4;
  int* tks = reinterpret_cast<int*>(smem + so);
  so += (kBlock ? kBlockMaxM * kTopK : 0) * 4;
  unsigned* tkscr = reinterpret_cast<unsigned*>(smem + so);
  so += (kBlock ? kBlockMaxM * kTopkScratch : 0) * 4;
  float* sel_sm = reinterpret_cast<float*>(smem + so);
  so += (kBlock ? kBlockMaxM * kTopK : 0) * 4;
  so = (so + 15) & ~15;
  float* hsh = reinterpret_cast<float*>(smem + so);
  so += (kBlock ? M * kSdK : 0) * 4;
  so = (so + 15) & ~15;
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + so);
  so += (2 + 2 * nstages) * 8;
  so = (so + 127) & ~127;
  uint8_t* stages = smem + so;
  const uint32_t route_bar = smem_u32(bars + 1 + 2 * nstages);
  // Shared down_proj rows of this CTA, in <= 2 ring stages issued before griddepcontrol.wait.
  const int sd_a = kBlock ? kSdRows * static_cast<int>(blockIdx.x) / static_cast<int>(gridDim.x) : 0;
  const int sd_b = kBlock ? kSdRows * (static_cast<int>(blockIdx.x) + 1) / static_cast<int>(gridDim.x) : 0;
  const int nsd = kBlock ? (sd_b - sd_a + kStageBytes / kSdRowBytes - 1) / (kStageBytes / kSdRowBytes) : 0;
  const int sd_per = nsd ? (sd_b - sd_a + nsd - 1) / nsd : 0;
  const uint32_t xbar = smem_u32(bars);
  auto full_bar = [&](int s) { return smem_u32(bars + 1 + s); };
  auto empty_bar = [&](int s) { return smem_u32(bars + 1 + nstages + s); };

  if (tid == 0) K3_MARK(0)
  if (tid == 0) {
    mbar_init(xbar, kLamport ? kConsumerThreads + (kFinal ? 1 : 0) : (kFinal ? 2 : 1));
    for (int s = 0; s < nstages; ++s) {
      mbar_init(full_bar(s), 1);
      mbar_init(empty_bar(s), kConsumerWarps);
    }
    if (kBlock) mbar_init(route_bar, kBlockMaxM);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  // (no zero-fill: every per-warp slot that is read is written exactly once)
  __syncthreads();
  if (kBlock && tid == kConsumerThreads) {
    // Static shared down_proj weights: stream them before waiting on the predecessor.
    for (int k = 0; k < min(nsd, nstages); ++k) {
      const int r0 = sd_a + k * sd_per, r1 = min(sd_b, r0 + sd_per);
      mbar_arrive_expect_tx(full_bar(k), (r1 - r0) * kSdRowBytes);
      tma_1d(smem_u32(stages + k * kStageBytes), ba.w_sd + static_cast<long>(r0) * kSdK,
             (r1 - r0) * kSdRowBytes, full_bar(k));
    }
  }

  // Lamport mode: readiness of x is carried by the mailbox data, so only the routing reader
  // (producer warp) waits for the PDL predecessor, and only if asked to. The block variant
  // reads route_shared's outputs everywhere, so every thread waits.
  if (!kLamport || kBlock || ((flags & kFlagWaitPrior) && warp == kConsumerWarps))
    asm volatile("griddepcontrol.wait;" ::: "memory");
  if (tid == 0) K3_MARK(1)

  const int total_seq = nsd + W.ngroups + W.nf2;
  if (warp == kConsumerWarps) {
    // ------------------------------ producer ------------------------------
    if (lane == 0) {
      if constexpr (!kLamport) {
        mbar_arrive_expect_tx(xbar, x_bytes);
        tma_1d(smem_u32(xs), x, x_bytes, xbar);
      }
      asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&tm_w13s)) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&tm_w2)) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&tm_w2s)) : "memory");
    }
    if (kFinal) {
      for (int i = lane; i < M * kTopK; i += 32) wsm[i] = __ldcg(topk_w + i);
      __syncwarp();
      if (lane == 0) mbar_arrive(xbar);
    }
    const CUtensorMap* pm13s = &tm_w13s;
    const CUtensorMap* pm2 = &tm_w2;
    const CUtensorMap* pm2s = &tm_w2s;
    auto issue = [&](int kseq) {
      const int s = kseq % nstages;
      uint8_t* dst = stages + s * kStageBytes;
      const uint32_t bar = full_bar(s);
      if (kseq < nsd) {
        if (lane == 0) {
          const int r0 = sd_a + kseq * sd_per, r1 = min(sd_b, r0 + sd_per);
          mbar_arrive_expect_tx(bar, (r1 - r0) * kSdRowBytes);
          tma_1d(smem_u32(dst), ba.w_sd + static_cast<long>(r0) * kSdK, (r1 - r0) * kSdRowBytes, bar);
        }
        return;
      }
      const int k = kseq - nsd;
      if (k < W.ngroups) {
        if (lane == 0) {
          const int qs = k == 0 ? W.qa : ((W.qa >> 3) + k) << 3;
          const int n = min(W.qb, (qs & ~7) + 8) - qs;
          const int t = qs / kPairsPerTok, j = (qs / kInter) % kTopK, u = qs % kInter;
          const int b = u >> 3, r0 = u & 7;
          const int e = kBlock ? tks[t * kTopK + j] : __ldcg(topk_ids + t * kTopK + j);
          const uint8_t* base = w13 + static_cast<long>(e) * kW13Rows * kW13RowBytes;
          mbar_arrive_expect_tx(bar, 2 * n * kW13RowBytes + 28 * 16 * 16);
          tma_1d(smem_u32(dst), base + (16 * b + r0) * kW13RowBytes, n * kW13RowBytes, bar);
          tma_1d(smem_u32(dst + n * kW13RowBytes), base + (16 * b + 8 + r0) * kW13RowBytes,
                 n * kW13RowBytes, bar);
          tma_3d(smem_u32(dst + kStageW), pm13s, 256 * (b & 1), 0, e * 4 + (b >> 3), bar);
        }
      } else {
        const int f = k - W.ngroups;
        const int nu = min(2, W.nunits - 2 * f);
        if (lane == 0) mbar_arrive_expect_tx(bar, nu * kTopK * (8 * 96 + 256));
        const int ui = lane >> 4, j = lane & 15, unit = 2 * f + ui;
        if (unit < W.nunits) {
          const int t = unit / W.nocts, o = W.oa + unit % W.nocts;
          const int e = kBlock ? tks[t * kTopK + j] : __ldcg(topk_ids + t * kTopK + j);
          uint8_t* d = dst + ui * kF2Unit;
          tma_2d(smem_u32(d + j * 8 * 96), pm2, 0, e * kHidden + 8 * o, bar);
          tma_3d(smem_u32(d + kF2W + j * 256), pm2s, 128 * (o & 3), 0, e * 28 + (o >> 4), bar);
        }
      }
    };
    const int first = min(total_seq, nstages);
    int k0 = 0;
    if (kBlock) {
      k0 = min(nsd, nstages);   // shared down_proj stages were issued before the wait
      mbar_wait(route_bar, 0);  // routing computed in-kernel (consumer warps 12..15)
    }
    if (lane == 0) K3_MARK(27)
    for (int k = k0; k < first; ++k) issue(k);
    if (lane == 0) K3_MARK(8)
    for (int k = first; k < total_seq; ++k) {
      mbar_wait(empty_bar(k % nstages), ((k / nstages) - 1) & 1);
      issue(k);
    }
    if (lane == 0) K3_MARK(9)
    if (trace && lane == 0 && total_seq <= nstages) {
      for (int k = 0; k < total_seq; ++k) {
        mbar_wait(full_bar(k), 0);
        if (k == nsd + W.ngroups - 1) K3_MARK(12)
      }
      K3_MARK(13)
    }
    return;
  }

  // ------------------------------ consumers ------------------------------
  {
    unsigned long long* counter = barrier + (M - 1);
    unsigned long long bar_target = 0;
    if (tid == 0) {
      unsigned long long c0;
      asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(c0) : "l"(counter) : "memory");
      bar_target = (c0 / gridDim.x + 1) * gridDim.x;
    }
    if (routed_only) {
      // CTA 0 hands the latent to the next kernel: poll, copy out, re-arm (like LamportCopy).
      if (blockIdx.x == 0) {
        // Routing warps (12..15) do not poll (see the Lamport poll below); warps 0..11 keep all their fragments
        // (<= 5 for M <= 4) in flight at once.
        constexpr int kPT = kTopkWarp0 * 32;
        constexpr int kMaxF = (kBlockMaxM * (kHidden / 8) + kPT - 1) / kPT;  // 5
        if (tid < kPT) {
          const int nf = M * (kHidden / 8);
          uint4 v[kMaxF];
          unsigned pend = 0;
#pragma unroll
          for (int i = 0; i < kMaxF; ++i)
            if (tid + i * kPT < nf) pend |= 1u << i;
          while (pend) {
#pragma unroll
            for (int i = 0; i < kMaxF; ++i)
              if (pend >> i & 1u)
                asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                             : "=r"(v[i].x), "=r"(v[i].y), "=r"(v[i].z), "=r"(v[i].w)
                             : "l"(reinterpret_cast<const uint4*>(x) + tid + i * kPT)
                             : "memory");
#pragma unroll
            for (int i = 0; i < kMaxF; ++i)
              if ((pend >> i & 1u) && v[i].x != 0x80000000u && v[i].y != 0x80000000u && v[i].z != 0x80000000u &&
                  v[i].w != 0x80000000u) {
                const int f = tid + i * kPT;
                reinterpret_cast<uint4*>(ba.latent_out)[f] = v[i];
                asm volatile("st.global.v4.u32 [%0], {%1, %1, %1, %1};" ::"l"(reinterpret_cast<const uint4*>(x) + f),
                             "r"(0x80000000u)
                             : "memory");
                pend &= ~(1u << i);
              }
          }
        }
      }
      mbar_arrive(xbar);
    } else if constexpr (kLamport) {
      // Poll this CTA's x rows (<= 2 tokens) out of the mailbox, 16 B per fragment.
      // Block variant: the routing warps (12..15) do NOT poll x - routing depends only on the
      // route_shared scores, so it must not wait for the latent. Warps 0..11 poll every fragment, all of a
      // thread's fragments in flight at once (<= 3 per thread for M <= 2 tokens in a CTA).
      const int t_lo = W.qa / kPairsPerTok, t_hi = (W.qb - 1) / kPairsPerTok;
      constexpr int kFrags = kHidden / 8;  // 448
      constexpr int kPollThreads = kBlock ? kTopkWarp0 * 32 : kConsumerThreads;
      if (kPollThreads == kConsumerThreads) {
        for (int f = t_lo * kFrags + tid; f < (t_hi + 1) * kFrags; f += kConsumerThreads) {
          const uint4* src = reinterpret_cast<const uint4*>(x) + f;
          uint4 v;
          do {
            asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(src) : "memory");
          } while (v.x == 0x80000000u || v.y == 0x80000000u || v.z == 0x80000000u || v.w == 0x80000000u);
          reinterpret_cast<uint4*>(xs)[f] = v;
        }
      } else if (tid < kPollThreads) {
        constexpr int kMaxF = (2 * kFrags + kPollThreads - 1) / kPollThreads;  // 3
        const int f0 = t_lo * kFrags + tid, f1 = (t_hi + 1) * kFrags;
        uint4 v[kMaxF];
        unsigned pend = 0;
#pragma unroll
        for (int i = 0; i < kMaxF; ++i)
          if (f0 + i * kPollThreads < f1) pend |= 1u << i;
        while (pend) {
#pragma unroll
          for (int i = 0; i < kMaxF; ++i)
            if (pend >> i & 1u)
              asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                           : "=r"(v[i].x), "=r"(v[i].y), "=r"(v[i].z), "=r"(v[i].w)
                           : "l"(reinterpret_cast<const uint4*>(x) + f0 + i * kPollThreads)
                           : "memory");
#pragma unroll
          for (int i = 0; i < kMaxF; ++i)
            if ((pend >> i & 1u) && v[i].x != 0x80000000u && v[i].y != 0x80000000u && v[i].z != 0x80000000u &&
                v[i].w != 0x80000000u) {
              reinterpret_cast<uint4*>(xs)[f0 + i * kPollThreads] = v[i];
              pend &= ~(1u << i);
            }
        }
      }
      mbar_arrive(xbar);
      if (tid == 0) K3_MARK(2)
      if (tid == 0) K3_MARK(24)
    }
    if constexpr (kBlock) {
      if (warp >= kTopkWarp0) {
        // Routing: warp 12 + t -> token t (M <= 4), one warp per token in parallel.
        const int t = warp - kTopkWarp0;
        if (t < M && (!routed_only || blockIdx.x == 0)) {
          int* sel_i = tks + t * kTopK;
          float* sel_s = sel_sm + t * kTopK;
          warp_top16(ba.scores + ((blockIdx.x % 8) * M + t) * 896, tkscr + t * kTopkScratch, sel_i, sel_s, lane,
                     nullptr);
          float sv = lane < kTopK ? sel_s[lane] : 0.f;
          float sum = sv;
#pragma unroll
          for (int o = 8; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
          if (blockIdx.x == 0 && lane < kTopK) {
            ba.ids_out[t * kTopK + lane] = sel_i[lane];
            ba.wts_out[t * kTopK + lane] = __float2bfloat16((ba.renorm ? sv / sum : sv) * ba.rscale);
          }
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(route_bar);
      } else {
        // Shared expert: h = bf16(SiTU(gate, up)) from route_shared's bf16 gate_up output
        // (same rounding points as vLLM's GEMM + situ_and_mul), kept as fp32 for the down GEMV.
        for (int i = tid; i < M * kSdK; i += kTopkWarp0 * 32) {
          const int t = i / kSdK, c = i - t * kSdK;
          const float gv = __bfloat162float(ba.h_sh[t * 2 * kSdK + c]);
          const float uv = __bfloat162float(ba.h_sh[t * 2 * kSdK + kSdK + c]);
          const float ga = ba.sh_beta * tanhf(gv / ba.sh_beta) * (1.f / (1.f + expf(-gv)));
          const float ua = ba.sh_lbeta > 0.f ? ba.sh_lbeta * tanhf(uv / ba.sh_lbeta) : uv;
          hsh[i] = __bfloat162float(__float2bfloat16(ga * ua));
        }
        asm volatile("bar.sync 2, %0;" ::"n"(kTopkWarp0 * 32) : "memory");
      }
      if (tid == 0) K3_MARK(3)
      // Shared-expert down_proj partial: out[t][n] = W_sd[n, :384] . h[t, :]. Row per warp
      // (warps 0..11, while the routing warps finish), 3 x 8 B per lane per token; warps 12..15
      // only keep the ring in step. Runs under the FC1 weight latency.
      for (int k = 0; k < nsd; ++k) {
        const int s = k % nstages;
        mbar_wait(full_bar(s), (k / nstages) & 1);
        const int r0 = sd_a + k * sd_per, r1 = min(sd_b, r0 + sd_per);
        const uint8_t* sw = stages + s * kStageBytes;
        if (warp < kTopkWarp0) {
          for (int r = warp; r < r1 - r0; r += kTopkWarp0) {
            float a[kBlockMaxM] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int q = 0; q < 3; ++q) {
              const int e0 = 4 * (lane + 32 * q);
              const uint2 wv = *reinterpret_cast<const uint2*>(sw + r * kSdRowBytes + e0 * 2);
              const float2 w01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wv.x));
              const float2 w23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wv.y));
#pragma unroll
              for (int tt = 0; tt < kBlockMaxM; ++tt)
                if (tt < M) {
                  const float4 hv = *reinterpret_cast<const float4*>(hsh + tt * kSdK + e0);
                  a[tt] = fmaf(w01.x, hv.x, fmaf(w01.y, hv.y, fmaf(w23.x, hv.z, fmaf(w23.y, hv.w, a[tt]))));
                }
            }
#pragma unroll
            for (int tt = 0; tt < kBlockMaxM; ++tt)
              if (tt < M) {
                const float v = warp_sum(a[tt]);
                if (lane == 0) ba.sh_out[static_cast<long>(tt) * kSdRows + r0 + r] = __float2bfloat16(v);
              }
          }
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(empty_bar(s));
      }
      if (tid == 0) K3_MARK(25)
    }
    const bool fc1_warp = warp < kFc1Warps;
    const int c = (fc1_warp ? warp : 0) * 8 + (lane & 7);  // K chunk (32 elements)
    const int rs = lane >> 3;             // row sub-index
    const int i8 = lane & 7;
    __half2 xr[16];
    int cur_t = -1;
    for (int k = 0; k < W.ngroups; ++k) {
      const int kseq = nsd + k;
      const int s = kseq % nstages;
      const int qs = k == 0 ? W.qa : ((W.qa >> 3) + k) << 3;
      const int n = min(W.qb, (qs & ~7) + 8) - qs;
      const int t = qs / kPairsPerTok, u = qs % kInter;
      const int b = u >> 3, r0 = u & 7, rg = (b & 7) >> 1;
      if (!fc1_warp) {
        mbar_wait(full_bar(s), (kseq / nstages) & 1);
        __syncwarp();
        if (lane == 0) mbar_arrive(empty_bar(s));
        continue;
      }
      if (t != cur_t) {
        if (cur_t < 0) mbar_wait(xbar, 0);
        load_block_h(reinterpret_cast<const __nv_bfloat16*>(xs) + t * kHidden + c * 32, xr);
        cur_t = t;
        if (tid == 0) K3_MARK(2)
      }
#ifdef K3_FC1TRACE
      if (trace && tid == 0 && k < 16) {
        unsigned long long ts;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts));
        trace[(gridDim.x + blockIdx.x) * 32 + k] = ts;
      }
#endif
      mbar_wait(full_bar(s), (kseq / nstages) & 1);
#ifdef K3_FC1TRACE
      if (trace && tid == 0 && k < 16) {
        unsigned long long ts;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts));
        trace[(gridDim.x + blockIdx.x) * 32 + 16 + k] = ts;
      }
#endif
      if (k == 0 && tid == 0 && !kBlock) K3_MARK(3)
      const uint8_t* sw = stages + s * kStageBytes;
      const uint8_t* ssc = sw + kStageW + (c >> 2) * 256 + rg * 4 + (c & 3);
      const int sigma0 = 2 * (qs - W.qa);
      const int s0 = (rs - sigma0) & 3;
      float v[4];
#ifdef FC1_SERIAL
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const int sl = s0 + 4 * m;
        v[m] = 0.f;
        if (sl < 2 * n) {
          const uint4 wv = *reinterpret_cast<const uint4*>(sw + sl * kW13RowBytes + c * 16);
          const int rib = sl < n ? r0 + sl : 8 + r0 + sl - n;
          v[m] = ue8m0(ssc[rib * 16]) * dot32_h(wv, xr);
        }
      }
#else
      {
        // Four rows at once, branch-free (rows past the group read a valid row and are zeroed):
        // unconditional loads + 8 independent HFMA2 chains instead of 4 serialized dots.
        uint4 wv[4];
        float sc4[4];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
          const int sl = s0 + 4 * m;
          const int slc = min(sl, 2 * n - 1);
          wv[m] = *reinterpret_cast<const uint4*>(sw + slc * kW13RowBytes + c * 16);
          const int rib = slc < n ? r0 + slc : 8 + r0 + slc - n;
          sc4[m] = sl < 2 * n ? ue8m0(ssc[rib * 16]) : 0.f;
        }
        __half2 a0[4], a1[4];
#pragma unroll
        for (int m = 0; m < 4; ++m) a0[m] = a1[m] = __float2half2_rn(0.f);
#ifndef NO_MATH
#pragma unroll
        for (int kw = 0; kw < 4; ++kw) {
          uint32_t r[4][4];
#pragma unroll
          for (int m = 0; m < 4; ++m) {
            const uint32_t w = kw == 0 ? wv[m].x : kw == 1 ? wv[m].y : kw == 2 ? wv[m].z : wv[m].w;
            cvt_e2m1_word(w, r[m]);
          }
#pragma unroll
          for (int b = 0; b < 4; b += 2)
#pragma unroll
            for (int m = 0; m < 4; ++m) {
              a0[m] = __hfma2(*reinterpret_cast<const __half2*>(&r[m][b]), xr[kw * 4 + b], a0[m]);
              a1[m] = __hfma2(*reinterpret_cast<const __half2*>(&r[m][b + 1]), xr[kw * 4 + b + 1], a1[m]);
            }
        }
#endif
#pragma unroll
        for (int m = 0; m < 4; ++m) {
          const float2 f = __half22float2(__hadd2(a0[m], a1[m]));
          v[m] = sc4[m] * (f.x + f.y);
#ifdef NO_MATH
          v[m] = sc4[m] * __int_as_float((wv[m].x ^ wv[m].w) & 0x3fffffffu) + __half2float(xr[m].x);
#endif
        }
      }
#endif
      // Transpose-reduce 4 row partials over the 8 lanes sharing rs.
      const bool b4 = i8 & 4, b2 = i8 & 2;
      float k0 = b4 ? v[2] : v[0], k1 = b4 ? v[3] : v[1];
      k0 += __shfl_xor_sync(0xffffffffu, b4 ? v[0] : v[2], 4);
      k1 += __shfl_xor_sync(0xffffffffu, b4 ? v[1] : v[3], 4);
      float kp = b2 ? k1 : k0;
      kp += __shfl_xor_sync(0xffffffffu, b2 ? k0 : k1, 2);
      kp += __shfl_xor_sync(0xffffffffu, kp, 1);
      const int sl = s0 + 4 * ((b4 ? 2 : 0) + (b2 ? 1 : 0));
      if (!(i8 & 1) && sl < 2 * n) red[warp * max_slots + sigma0 + sl] = kp;  // one writer per (warp, slot)
      __syncwarp();
      if (lane == 0) mbar_arrive(empty_bar(s));
    }
    if (tid == 0) K3_MARK(11)
    // Before the first global write (h, then gemm2_out): the outputs may reuse memory the PDL
    // predecessor was still reading. Free here: the producer finished long ago.
    if constexpr (kLamport) asm volatile("griddepcontrol.wait;" ::: "memory");
    consumer_sync();
    // FC1 epilogue: h = SiTU(gate, up) for this CTA's pairs.
    for (int kk = tid; kk < W.qb - W.qa; kk += kConsumerThreads) {
      const int q = W.qa + kk;
      const int gs = max(W.qa, q & ~7), ge = min(W.qb, (q & ~7) + 8);
      const int su = 2 * (gs - W.qa) + (q - gs);
      const int sg = su + (ge - gs);
      float up = 0.f, gate = 0.f;
#pragma unroll
      for (int w = 0; w < kFc1Warps; ++w) {  // fixed warp (K-range) order
        up += red[w * max_slots + su];
        gate += red[w * max_slots + sg];
      }
      const int t = q / kPairsPerTok, j = (q / kInter) % kTopK, u = q % kInter;
      const int b = u >> 3, r = u & 7;
      const int i = 16 * (b >> 1) + 2 * r + (b & 1);
      h[t * kPairsPerTok + j * kInter + i] = __float2half_rn(situ(up, gate, beta, linear_beta));
    }
    consumer_sync();
    if (tid == 0 && !routed_only) {
      K3_MARK(4)
      // Every call adds exactly G to the counter, so this call's target is the next
      // multiple of G above the value read at start (our own add is still pending).
      asm volatile("fence.acq_rel.gpu;\n\tred.relaxed.gpu.global.add.u64 [%0], 1;" ::"l"(counter) : "memory");
      unsigned long long cur;
      do {
        asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(cur) : "l"(counter) : "memory");
      } while (cur < bar_target);
      K3_MARK(5)
    }
    consumer_sync();
    if (kLamport && blockIdx.x == 0 && !routed_only) {
      // Every CTA polled its x rows before the barrier: re-arm mailbox rows [0, M).
      const uint4 empty = make_uint4(0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u);
      for (int f = tid; f < M * (kHidden / 8); f += kConsumerThreads)
        asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(reinterpret_cast<const uint4*>(x) + f),
                     "r"(empty.x), "r"(empty.y), "r"(empty.z), "r"(empty.w)
                     : "memory");
    }
    asm volatile("griddepcontrol.launch_dependents;");
    // Stage h (all tokens) into shared memory (over x).
    __half* hs = reinterpret_cast<__half*>(xs);
    if (!routed_only)
      for (int v = tid; v < M * kPairsPerTok / 8; v += kConsumerThreads)
        reinterpret_cast<uint4*>(hs)[v] = __ldcg(reinterpret_cast<const uint4*>(h) + v);
    consumer_sync();
    if (tid == 0) K3_MARK(6)

    // FC2: warp = expert slot j; lane = (unit ui, row rho, K-half hf) with 3 of the 6 K-chunks;
    // h[t, j, K-half] stays in registers while t is unchanged.
    const int ui = lane >> 4, rho = (lane >> 1) & 7, hf = lane & 1;
    const int j = warp;
    if (tid == 0 && trace) trace[blockIdx.x * 32 + 20] = clock64();
    __half2 hr[3][16];
    int h_t = -1;
    for (int f = 0; f < W.nf2; ++f) {
      const int k = nsd + W.ngroups + f;
      const int s = k % nstages;
      const int unit = 2 * f + ui;
      const bool valid = unit < W.nunits;
      const int t = valid ? unit / W.nocts : 0;
      const int o = W.oa + (valid ? unit % W.nocts : 0);
      const int p = 8 * o + rho;
      const int rg = (p & 127) >> 5;
      if (valid && t != h_t) {
        const uint4* src = reinterpret_cast<const uint4*>(hs + (t * kTopK + j) * kInter + 3 * hf * 32);
#pragma unroll
        for (int m = 0; m < 3; ++m)
#pragma unroll
          for (int vv = 0; vv < 4; ++vv) {
            const uint4 raw = src[m * 4 + vv];
            const __half2* pp = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
            for (int uu = 0; uu < 4; ++uu) hr[m][vv * 4 + uu] = pp[uu];
          }
        h_t = t;
      }
      mbar_wait(full_bar(s), (k / nstages) & 1);
      if (f == 0 && tid == 0) K3_MARK(10)
      if (f == 0 && tid == 0 && trace) trace[blockIdx.x * 32 + 21] = clock64();
      const uint8_t* base = stages + s * kStageBytes + ui * kF2Unit;
      float acc = 0.f;
      if (valid) {
        uint4 wv[3];
        uint8_t sc[3];
#pragma unroll
        for (int m = 0; m < 3; ++m) {
          const int cc = 3 * hf + m;
          wv[m] = *reinterpret_cast<const uint4*>(base + (j * 8 + rho) * 96 + cc * 16);
          sc[m] = base[kF2W + j * 256 + (cc >> 2) * 128 + rho * 16 + rg * 4 + (cc & 3)];
        }
#pragma unroll
        for (int m = 0; m < 3; ++m) {
#ifdef NO_MATH2
          acc += __int_as_float((wv[m].x ^ wv[m].z) & 0x3fffffff) * __half2float(hr[m][m].x) + sc[m];
#else
          acc = fmaf(ue8m0(sc[m]), dot32_h(wv[m], hr[m]), acc);
#endif
        }
      }
      if (f == 0 && tid == 0 && trace) trace[blockIdx.x * 32 + 22] = clock64() + (acc == 1.2345f);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty_bar(s));
      acc += __shfl_xor_sync(0xffffffffu, acc, 1);
      if (valid && hf == 0) {
        if (kFinal) {
          acc2[j * max_rows2 + unit * 8 + rho] = wsm[t * kTopK + j] * acc;  // one writer per (slot, row)
        } else {
          const int n_out = 32 * (p >> 5) + 4 * (p & 7) + ((p & 31) >> 3);
          out[static_cast<long>(t * kTopK + j) * kHidden + n_out] = __float2bfloat16(acc);
        }
      }
      if (f == 0 && tid == 0) K3_MARK(18)
      if (f == 0 && tid == 0 && trace) trace[blockIdx.x * 32 + 23] = clock64();
    }
    if (kFinal) {
      consumer_sync();
      for (int i = tid; i < W.nunits * 8; i += kConsumerThreads) {
        const int unit = i >> 3, r = i & 7;
        const int t = unit / W.nocts, o = W.oa + unit % W.nocts;
        const int p = 8 * o + r;
        const int n_out = 32 * (p >> 5) + 4 * (p & 7) + ((p & 31) >> 3);
        float v = 0.f;
#pragma unroll
        for (int jj = 0; jj < kTopK; ++jj) v += acc2[jj * max_rows2 + i];  // fixed expert-slot order
        out[t * kHidden + n_out] = __float2bfloat16(v);
      }
    }
  }
  if (trace) {
    if (tid == 0) K3_MARK(15)
    consumer_sync();
    if (tid == 0) K3_MARK(7)
    if (tid == 0) trace[blockIdx.x * 32 + 17] = clock64();
  }
#undef K3_MARK
}

// ---------------------------------------------------------------------------------------------
// MoE publish epilogue (separable device function; composes with other expert bodies): this CTA's
// FC2 units u = (token t, 8-column fragment f = oa + u % nocts) sit in smem `fin` as bf16 [u][16 j][8]
// (slot order j = routing order); `wsm` [M][16] holds the bf16-rounded routing weights. Finalize like
// vLLM's finalize_top16_bf16 (fp32 FMA over j = 0..15, one bf16 rounding), canonicalize -0.0, and
// store 16 bytes into lat_mb[par][rank][t][8f..8f+8) of every rank (multimem.st). Threads 0..nunits*8-1
// (rounded up to whole warps) participate.
__device__ __forceinline__ void moe_publish_latent(const __nv_bfloat16* fin, const float* wsm, int nunits,
                                                   int nocts, int oa, const TailArgs& ta, int tid) {
  if (tid >= ((nunits * 8 + 31) & ~31)) return;
  const bool ok = tid < nunits * 8;
  const int u = ok ? tid >> 3 : 0, e = tid & 7, t = u / nocts;
  float a = 0.f;
  if (ok)
#pragma unroll 4
    for (int jj = 0; jj < kTopK; ++jj) a = fmaf(__bfloat162float(fin[(u * kTopK + jj) * 8 + e]), wsm[t * kTopK + jj], a);
  const __nv_bfloat16 b16 = __float2bfloat16(a);
  uint32_t v = *reinterpret_cast<const uint16_t*>(&b16);
  v |= __shfl_down_sync(0xffffffffu, v, 1) << 16;
  const uint32_t w1 = __shfl_down_sync(0xffffffffu, v, 2);
  const uint32_t w2 = __shfl_down_sync(0xffffffffu, v, 4);
  const uint32_t w3 = __shfl_down_sync(0xffffffffu, v, 6);
  if (ok && e == 0) {
    const int f = oa + u % nocts;
    const unsigned long long off =
        static_cast<unsigned long long>(((ta.par * kTp + ta.rank) * ta.mmax + t) * kHidden + 8 * f) * 2;
    st_mb16(ta.lat_st + off, no_neg_zero4(make_uint4(v, w1, w2, w3)), ta.mc);
  }
}

// moe_block_lamport's PUBLISH-mode kernel: a separate copy of moe_fused_kernel (kTail = publish), so the
// production kernel above keeps its exact code (code layout changes alone moved it by ~1 us).
template <bool kFinal, bool kLamport, bool kBlock = false, bool kTail = false>
__global__ void __launch_bounds__(kFusedThreads, 1)
moe_fused_pub_kernel(const __grid_constant__ CUtensorMap tm_w13s, const __grid_constant__ CUtensorMap tm_w2,
                 const __grid_constant__ CUtensorMap tm_w2s, const __nv_bfloat16* __restrict__ x,
                 const int* __restrict__ topk_ids, const float* __restrict__ topk_w,
                 const uint8_t* __restrict__ w13, __half* __restrict__ h,
                 unsigned long long* __restrict__ barrier, __nv_bfloat16* __restrict__ out, int M,
                 int nstages, float beta, float linear_beta, int flags,
                 unsigned long long* __restrict__ trace,
#ifdef K3_NO_GRIDCONST
                 BlockArgs ba) {
#else
                 const __grid_constant__ BlockArgs ba) {
#endif
  // kLamport: x is the local view of a Lamport mailbox (rows [0, M) of 3584 bf16;
  //        an empty 32-bit word is 0x80000000). CTAs poll their rows instead of TMA-loading x,
  //        and CTA 0 re-arms rows [0, M) after the grid barrier (all CTAs have read x by then).
  //        kFlagWaitPrior -> the producer warp still executes griddepcontrol.wait before reading
  //        topk_ids (needed when routing comes from the PDL predecessor).
#if defined(K3_TRACE_CLOCK) || defined(K3_TRACE_SMEM)  // marks buffered in smem, copied out at the end
  __shared__ unsigned long long trsm[32];
  if (threadIdx.x < 32) trsm[threadIdx.x] = 0;
#ifdef K3_TRACE_CLOCK  // SM cycles
#define K3_MARK(slot)                                                                 \
  if (trace) trsm[slot] = clock64();
#else                  // %globaltimer (32 ns resolution here)
#define K3_MARK(slot)                                                                 \
  if (trace) {                                                                        \
    unsigned long long ts;                                                            \
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts));                           \
    trsm[slot] = ts;                                                                  \
  }
#endif
#else
#define K3_MARK(slot)                                                                 \
  if (trace) {                                                                        \
    unsigned long long ts;                                                            \
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts));                           \
    trace[blockIdx.x * 32 + (slot)] = ts;                                             \
  }
#endif
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  FusedWork W = fused_work(M);
  const bool routed_only = kBlock && ba.latent_out != nullptr;
  if (routed_only) {  // routing + shared expert only: no routed FC1/FC2 work in this kernel
    W.qa = W.qb = 0;
    W.ngroups = 0;
    W.nunits = 0;
    W.nf2 = 0;
  }
  // Shared memory map: x (bf16, later fp16 h) | red | acc2 | wsm | barriers | ring.
  const int x_bytes = M * kHidden * 2;
  uint8_t* xs = smem;
  const int max_slots = 2 * ((M * kPairsPerTok + gridDim.x - 1) / gridDim.x) + 16;
  const int max_rows2 = M * 8 * ((kOctets + gridDim.x - 1) / gridDim.x);
  // All smem pointers are smem + integer offset, so the compiler keeps the shared address space
  // (LDS/STS instead of generic LD/ST; an int->pointer cast for alignment used to lose it).
  int so = x_bytes;
  float* red = reinterpret_cast<float*>(smem + so);
  // Deterministic reductions: red = per-FC1-warp slot partials [kFc1Warps][max_slots] (summed in warp order
  // in the FC1 epilogue); acc2 (kFinal) = per-expert-slot rows [kTopK][max_rows2] (summed in slot order).
  float* acc2 = red + kFc1Warps * max_slots;
  float* wsm = acc2 + (kFinal ? kTopK : 1) * max_rows2;
  so += (kFc1Warps * max_slots + (kFinal ? kTopK : 1) * max_rows2 + M * kTopK) * 4;
  // Block variant: routing ids, top-k scratch (2 warps), shared-expert activations (fp32).
  int* tks = reinterpret_cast<int*>(smem + so);
  so += (kBlock ? kBlockMaxM * kTopK : 0) * 4;
  unsigned* tkscr = reinterpret_cast<unsigned*>(smem + so);
  so += (kBlock ? kBlockMaxM * kTopkScratch : 0) * 4;
  float* sel_sm = reinterpret_cast<float*>(smem + so);
  so += (kBlock ? kBlockMaxM * kTopK : 0) * 4;
  so = (so + 15) & ~15;
  float* hsh = reinterpret_cast<float*>(smem + so);
  so += (kBlock ? M * kSdK : 0) * 4;
  // Publish epilogue (kTail): extra smem after hsh.
  constexpr bool tail = kBlock && kTail;
  const TailLayout TL = tail_layout(M, gridDim.x);
  so = (so + 15) & ~15;
  uint8_t* tsm = smem + so;
  so += tail ? TL.bytes : 0;
  so = (so + 15) & ~15;
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + so);
  so += (2 + 2 * nstages) * 8;
  so = (so + 127) & ~127;
#ifdef K3_OLD_SMEMPTR
  uint8_t* stages = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem) + so + 127) & ~static_cast<uintptr_t>(127));
#else
  uint8_t* stages = smem + so;
#endif
  const uint32_t route_bar = smem_u32(bars + 1 + 2 * nstages);
  // Shared down_proj rows of this CTA (whole 8-row fragments), in <= 2 ring stages issued before
  // griddepcontrol.wait.
  // (tail: whole 8-row fragments, for the shared reduce-scatter stores)
  const int sd_a = !kBlock ? 0
                   : kTail ? 8 * ((kSdRows / 8) * static_cast<int>(blockIdx.x) / static_cast<int>(gridDim.x))
                           : kSdRows * static_cast<int>(blockIdx.x) / static_cast<int>(gridDim.x);
  const int sd_b = !kBlock ? 0
                   : kTail ? 8 * ((kSdRows / 8) * (static_cast<int>(blockIdx.x) + 1) / static_cast<int>(gridDim.x))
                           : kSdRows * (static_cast<int>(blockIdx.x) + 1) / static_cast<int>(gridDim.x);
  const int nsd = kBlock ? (sd_b - sd_a + kStageBytes / kSdRowBytes - 1) / (kStageBytes / kSdRowBytes) : 0;
  const int sd_per = nsd ? (sd_b - sd_a + nsd - 1) / nsd : 0;
  const uint32_t xbar = smem_u32(bars);
  auto full_bar = [&](int s) { return smem_u32(bars + 1 + s); };
  auto empty_bar = [&](int s) { return smem_u32(bars + 1 + nstages + s); };

  if (tid == 0) K3_MARK(0)
  if (tid == 0) {
    mbar_init(xbar, kLamport ? kConsumerThreads + (kFinal ? 1 : 0) : (kFinal ? 2 : 1));
    for (int s = 0; s < nstages; ++s) {
      mbar_init(full_bar(s), 1);
      mbar_init(empty_bar(s), kConsumerWarps);
    }
    if (kBlock) mbar_init(route_bar, kBlockMaxM);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  // (no zero-fill: every per-warp slot that is read is written exactly once)
#ifndef K3TAIL_PARAM_CONST
  if (tail && tid < static_cast<int>(sizeof(TailArgs) / 4))
    reinterpret_cast<uint32_t*>(tsm + TL.args)[tid] = reinterpret_cast<const uint32_t*>(&ba.tail)[tid];
#endif
  __syncthreads();
  if (kBlock && tid == kConsumerThreads) {
    // Static shared down_proj weights: stream them before waiting on the predecessor.
    for (int k = 0; k < min(nsd, nstages); ++k) {
      const int r0 = sd_a + k * sd_per, r1 = min(sd_b, r0 + sd_per);
      mbar_arrive_expect_tx(full_bar(k), (r1 - r0) * kSdRowBytes);
      tma_1d(smem_u32(stages + k * kStageBytes), ba.w_sd + static_cast<long>(r0) * kSdK,
             (r1 - r0) * kSdRowBytes, full_bar(k));
    }
  }

  // Lamport mode: readiness of x is carried by the mailbox data, so only the routing reader
  // (producer warp) waits for the PDL predecessor, and only if asked to. The block variant
  // reads route_shared's outputs everywhere, so every thread waits.
  if (!kLamport || kBlock || ((flags & kFlagWaitPrior) && warp == kConsumerWarps))
    asm volatile("griddepcontrol.wait;" ::: "memory");
  if (tid == 0) K3_MARK(1)

  const int total_seq = nsd + W.ngroups + W.nf2;
  if (warp == kConsumerWarps) {
    // ------------------------------ producer ------------------------------
    if (lane == 0) {
      if constexpr (!kLamport) {
        mbar_arrive_expect_tx(xbar, x_bytes);
        tma_1d(smem_u32(xs), x, x_bytes, xbar);
      }
      asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&tm_w13s)) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&tm_w2)) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&tm_w2s)) : "memory");
    }
    if (kFinal) {
      for (int i = lane; i < M * kTopK; i += 32) wsm[i] = __ldcg(topk_w + i);
      __syncwarp();
      if (lane == 0) mbar_arrive(xbar);
    }
    const CUtensorMap* pm13s = &tm_w13s;
    const CUtensorMap* pm2 = &tm_w2;
    const CUtensorMap* pm2s = &tm_w2s;
    auto issue = [&](int kseq) {
      const int s = kseq % nstages;
      uint8_t* dst = stages + s * kStageBytes;
      const uint32_t bar = full_bar(s);
      if (kseq < nsd) {
        if (lane == 0) {
          const int r0 = sd_a + kseq * sd_per, r1 = min(sd_b, r0 + sd_per);
          mbar_arrive_expect_tx(bar, (r1 - r0) * kSdRowBytes);
          tma_1d(smem_u32(dst), ba.w_sd + static_cast<long>(r0) * kSdK, (r1 - r0) * kSdRowBytes, bar);
        }
        return;
      }
      const int k = kseq - nsd;
      if (k < W.ngroups) {
        if (lane == 0) {
          const int qs = k == 0 ? W.qa : ((W.qa >> 3) + k) << 3;
          const int n = min(W.qb, (qs & ~7) + 8) - qs;
          const int t = qs / kPairsPerTok, j = (qs / kInter) % kTopK, u = qs % kInter;
          const int b = u >> 3, r0 = u & 7;
          const int e = kBlock ? tks[t * kTopK + j] : __ldcg(topk_ids + t * kTopK + j);
          const uint8_t* base = w13 + static_cast<long>(e) * kW13Rows * kW13RowBytes;
          mbar_arrive_expect_tx(bar, 2 * n * kW13RowBytes + 28 * 16 * 16);
          tma_1d(smem_u32(dst), base + (16 * b + r0) * kW13RowBytes, n * kW13RowBytes, bar);
          tma_1d(smem_u32(dst + n * kW13RowBytes), base + (16 * b + 8 + r0) * kW13RowBytes,
                 n * kW13RowBytes, bar);
          tma_3d(smem_u32(dst + kStageW), pm13s, 256 * (b & 1), 0, e * 4 + (b >> 3), bar);
        }
      } else if (!tail || k - W.ngroups < W.nf2) {
        const int f = k - W.ngroups;
        const int nu = min(2, W.nunits - 2 * f);
        if (lane == 0) mbar_arrive_expect_tx(bar, nu * kTopK * (8 * 96 + 256));
        const int ui = lane >> 4, j = lane & 15, unit = 2 * f + ui;
        if (unit < W.nunits) {
          const int t = unit / W.nocts, o = W.oa + unit % W.nocts;
          const int e = kBlock ? tks[t * kTopK + j] : __ldcg(topk_ids + t * kTopK + j);
          uint8_t* d = dst + ui * kF2Unit;
          if (tail) {
            // Unit o = output columns [8o, 8o+8): physical rows 32(o/4) + 8b' + 2(o%4) + {0,1}, b' = 0..3
            // (4D boxes over [blk][b'][r][128 B] and the swizzled scales [tile][cb][b'][128 B]).
            tma_4d(smem_u32(d + j * 8 * 96), pm2, 0, 2 * (o & 3), 0, e * (kHidden / 32) + (o >> 2), bar);
            tma_4d(smem_u32(d + kF2W + j * 256), pm2s, 32 * (o & 3), 0, 0, e * 28 + (o >> 4), bar);
          } else {
            tma_2d(smem_u32(d + j * 8 * 96), pm2, 0, e * kHidden + 8 * o, bar);
            tma_3d(smem_u32(d + kF2W + j * 256), pm2s, 128 * (o & 3), 0, e * 28 + (o >> 4), bar);
          }
        }
      }
    };
    const int first = min(total_seq, nstages);
    int k0 = 0;
    if (kBlock) {
      k0 = min(nsd, nstages);   // shared down_proj stages were issued before the wait
      mbar_wait(route_bar, 0);  // routing computed in-kernel (consumer warps 12..15)
    }
    if (lane == 0) K3_MARK(27)
    for (int k = k0; k < first; ++k) issue(k);
    if (lane == 0) K3_MARK(8)
    for (int k = first; k < total_seq; ++k) {
      mbar_wait(empty_bar(k % nstages), ((k / nstages) - 1) & 1);
      issue(k);
    }
    if (lane == 0) K3_MARK(9)
    if (trace && lane == 0 && total_seq <= nstages) {
      for (int k = 0; k < total_seq; ++k) {
        mbar_wait(full_bar(k), 0);
        if (k == nsd + W.ngroups - 1) K3_MARK(12)
      }
      K3_MARK(13)
    }
    return;
  }

  // ------------------------------ consumers ------------------------------
  {
    unsigned long long* counter = barrier + (M - 1);
    unsigned long long bar_target = 0;
    if (tid == 0) {
      unsigned long long c0;
      asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(c0) : "l"(counter) : "memory");
      bar_target = (c0 / gridDim.x + 1) * gridDim.x;
    }
    if (routed_only) {
      // CTA 0 hands the latent to the next kernel: poll, copy out, re-arm (like LamportCopy).
      if (blockIdx.x == 0) {
        // Routing warps (12..15) do not poll (see the Lamport poll below); warps 0..11 keep all their fragments
        // (<= 5 for M <= 4) in flight at once.
        constexpr int kPT = kTopkWarp0 * 32;
        constexpr int kMaxF = (kBlockMaxM * (kHidden / 8) + kPT - 1) / kPT;  // 5
        if (tid < kPT) {
          const int nf = M * (kHidden / 8);
          uint4 v[kMaxF];
          unsigned pend = 0;
#pragma unroll
          for (int i = 0; i < kMaxF; ++i)
            if (tid + i * kPT < nf) pend |= 1u << i;
          while (pend) {
#pragma unroll
            for (int i = 0; i < kMaxF; ++i)
              if (pend >> i & 1u)
                asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                             : "=r"(v[i].x), "=r"(v[i].y), "=r"(v[i].z), "=r"(v[i].w)
                             : "l"(reinterpret_cast<const uint4*>(x) + tid + i * kPT)
                             : "memory");
#pragma unroll
            for (int i = 0; i < kMaxF; ++i)
              if ((pend >> i & 1u) && v[i].x != 0x80000000u && v[i].y != 0x80000000u && v[i].z != 0x80000000u &&
                  v[i].w != 0x80000000u) {
                const int f = tid + i * kPT;
                reinterpret_cast<uint4*>(ba.latent_out)[f] = v[i];
                asm volatile("st.global.v4.u32 [%0], {%1, %1, %1, %1};" ::"l"(reinterpret_cast<const uint4*>(x) + f),
                             "r"(0x80000000u)
                             : "memory");
                pend &= ~(1u << i);
              }
          }
        }
      }
      mbar_arrive(xbar);
    } else if constexpr (kLamport) {
      // Poll this CTA's x rows (<= 2 tokens) out of the mailbox, 16 B per fragment.
      // Block variant: the routing warps (12..15) do NOT poll x - routing depends only on the
      // route_shared scores, so it must not wait for the latent. Warps 0..11 poll every fragment, all of a
      // thread's fragments in flight at once (<= 3 per thread for M <= 2 tokens in a CTA).
      const int t_lo = W.qa / kPairsPerTok, t_hi = (W.qb - 1) / kPairsPerTok;
      constexpr int kFrags = kHidden / 8;  // 448
      constexpr int kPollThreads = kBlock ? kTopkWarp0 * 32 : kConsumerThreads;
      if (kPollThreads == kConsumerThreads) {
        for (int f = t_lo * kFrags + tid; f < (t_hi + 1) * kFrags; f += kConsumerThreads) {
          const uint4* src = reinterpret_cast<const uint4*>(x) + f;
          uint4 v;
          do {
            asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(src) : "memory");
          } while (v.x == 0x80000000u || v.y == 0x80000000u || v.z == 0x80000000u || v.w == 0x80000000u);
          reinterpret_cast<uint4*>(xs)[f] = v;
        }
      } else if (tid < kPollThreads) {
        constexpr int kMaxF = (2 * kFrags + kPollThreads - 1) / kPollThreads;  // 3
        const int f0 = t_lo * kFrags + tid, f1 = (t_hi + 1) * kFrags;
        uint4 v[kMaxF];
        unsigned pend = 0;
#pragma unroll
        for (int i = 0; i < kMaxF; ++i)
          if (f0 + i * kPollThreads < f1) pend |= 1u << i;
        while (pend) {
#pragma unroll
          for (int i = 0; i < kMaxF; ++i)
            if (pend >> i & 1u)
              asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                           : "=r"(v[i].x), "=r"(v[i].y), "=r"(v[i].z), "=r"(v[i].w)
                           : "l"(reinterpret_cast<const uint4*>(x) + f0 + i * kPollThreads)
                           : "memory");
#pragma unroll
          for (int i = 0; i < kMaxF; ++i)
            if ((pend >> i & 1u) && v[i].x != 0x80000000u && v[i].y != 0x80000000u && v[i].z != 0x80000000u &&
                v[i].w != 0x80000000u) {
              reinterpret_cast<uint4*>(xs)[f0 + i * kPollThreads] = v[i];
              pend &= ~(1u << i);
            }
        }
      }
      mbar_arrive(xbar);
      if (tid == 0) K3_MARK(2)
      if (tid == 0) K3_MARK(24)
    }
    if constexpr (kBlock) {
      if (warp >= kTopkWarp0) {
        // Routing: warp 12 + t -> token t (M <= 4), one warp per token in parallel.
        const int t = warp - kTopkWarp0;
        if (t < M && (!routed_only || blockIdx.x == 0)) {
          int* sel_i = tks + t * kTopK;
          float* sel_s = sel_sm + t * kTopK;
          warp_top16(ba.scores + ((blockIdx.x % 8) * M + t) * 896, tkscr + t * kTopkScratch, sel_i, sel_s, lane,
                     nullptr);
          float sv = lane < kTopK ? sel_s[lane] : 0.f;
          float sum = sv;
#pragma unroll
          for (int o = 8; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
          const __nv_bfloat16 wb = __float2bfloat16((ba.renorm ? sv / sum : sv) * ba.rscale);
          if (kTail && lane < kTopK) wsm[t * kTopK + lane] = __bfloat162float(wb);  // tail finalize weights
          if (blockIdx.x == 0 && lane < kTopK) {
            ba.ids_out[t * kTopK + lane] = sel_i[lane];
            ba.wts_out[t * kTopK + lane] = wb;
          }
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(route_bar);
      } else {
        // Shared expert: h = bf16(SiTU(gate, up)) from route_shared's bf16 gate_up output
        // (same rounding points as vLLM's GEMM + situ_and_mul), kept as fp32 for the down GEMV.
        for (int i = tid; i < M * kSdK; i += kTopkWarp0 * 32) {
          const int t = i / kSdK, c = i - t * kSdK;
          const float gv = __bfloat162float(ba.h_sh[t * 2 * kSdK + c]);
          const float uv = __bfloat162float(ba.h_sh[t * 2 * kSdK + kSdK + c]);
          const float ga = ba.sh_beta * tanhf(gv / ba.sh_beta) * (1.f / (1.f + expf(-gv)));
          const float ua = ba.sh_lbeta > 0.f ? ba.sh_lbeta * tanhf(uv / ba.sh_lbeta) : uv;
          hsh[i] = __bfloat162float(__float2bfloat16(ga * ua));
        }
        asm volatile("bar.sync 2, %0;" ::"n"(kTopkWarp0 * 32) : "memory");
      }
      if (tid == 0) K3_MARK(3)
      // Shared-expert down_proj partial: out[t][n] = W_sd[n, :384] . h[t, :]. Row per warp
      // (warps 0..11, while the routing warps finish), 3 x 8 B per lane per token; warps 12..15
      // only keep the ring in step. Runs under the FC1 weight latency.
      for (int k = 0; k < nsd; ++k) {
        const int s = k % nstages;
        mbar_wait(full_bar(s), (k / nstages) & 1);
        const int r0 = sd_a + k * sd_per, r1 = min(sd_b, r0 + sd_per);
        const uint8_t* sw = stages + s * kStageBytes;
        if (warp < kTopkWarp0) {
          for (int r = warp; r < r1 - r0; r += kTopkWarp0) {
            float a[kBlockMaxM] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int q = 0; q < 3; ++q) {
              const int e0 = 4 * (lane + 32 * q);
              const uint2 wv = *reinterpret_cast<const uint2*>(sw + r * kSdRowBytes + e0 * 2);
              const float2 w01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wv.x));
              const float2 w23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wv.y));
#pragma unroll
              for (int tt = 0; tt < kBlockMaxM; ++tt)
                if (tt < M) {
                  const float4 hv = *reinterpret_cast<const float4*>(hsh + tt * kSdK + e0);
                  a[tt] = fmaf(w01.x, hv.x, fmaf(w01.y, hv.y, fmaf(w23.x, hv.z, fmaf(w23.y, hv.w, a[tt]))));
                }
            }
#pragma unroll
            for (int tt = 0; tt < kBlockMaxM; ++tt)
              if (tt < M) {
                const float v = warp_sum(a[tt]);
                if (lane == 0) {
                  if (tail)
                    reinterpret_cast<__nv_bfloat16*>(tsm + TL.sdo)[tt * TL.sdr + r0 + r - sd_a] = __float2bfloat16(v);
                  else
                    ba.sh_out[static_cast<long>(tt) * kSdRows + r0 + r] = __float2bfloat16(v);
                }
              }
          }
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(empty_bar(s));
      }
      if (tail && warp < kTopkWarp0) {
        // Shared reduce-scatter, producer side: 8-row fragments of the shared partial go straight into
        // the owner rank's RS mailbox, slot [t][rank] (vLLM: shared_peer_ptrs stores).
        asm volatile("bar.sync 2, %0;" ::"n"(kTopkWarp0 * 32) : "memory");
        const int nfr = (sd_b - sd_a) >> 3;
        for (int i = tid; i < M * nfr; i += kTopkWarp0 * 32) {
          const int t = i / nfr, q = i - t * nfr, row = sd_a + 8 * q;
          const int d = row / kUpShard, col = row - d * kUpShard;
          const uint4 v = no_neg_zero4(*reinterpret_cast<const uint4*>(tsm + TL.sdo + (t * TL.sdr + 8 * q) * 2));
          const TailArgs& tas = *reinterpret_cast<const TailArgs*>(tsm + TL.args);
          st_mb16(tas.rs_peer[d] +
                      static_cast<unsigned long long>(((tas.par * kTp + tas.rank) * tas.mmax + t) * kUpShard + col) * 2,
                  v, 0);
        }
      }
      if (tid == 0) K3_MARK(25)
    }
    const bool fc1_warp = warp < kFc1Warps;
    const int c = (fc1_warp ? warp : 0) * 8 + (lane & 7);  // K chunk (32 elements)
    const int rs = lane >> 3;             // row sub-index
    const int i8 = lane & 7;
    __half2 xr[16];
    int cur_t = -1;
    for (int k = 0; k < W.ngroups; ++k) {
      const int kseq = nsd + k;
      const int s = kseq % nstages;
      const int qs = k == 0 ? W.qa : ((W.qa >> 3) + k) << 3;
      const int n = min(W.qb, (qs & ~7) + 8) - qs;
      const int t = qs / kPairsPerTok, u = qs % kInter;
      const int b = u >> 3, r0 = u & 7, rg = (b & 7) >> 1;
      if (!fc1_warp) {
        mbar_wait(full_bar(s), (kseq / nstages) & 1);
        __syncwarp();
        if (lane == 0) mbar_arrive(empty_bar(s));
        continue;
      }
      if (t != cur_t) {
        if (cur_t < 0) mbar_wait(xbar, 0);
        load_block_h(reinterpret_cast<const __nv_bfloat16*>(xs) + t * kHidden + c * 32, xr);
        cur_t = t;
        if (tid == 0) K3_MARK(2)
      }
#ifdef K3_FC1TRACE
      if (trace && tid == 0 && k < 16) {
        unsigned long long ts;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts));
        trace[(gridDim.x + blockIdx.x) * 32 + k] = ts;
      }
#endif
      mbar_wait(full_bar(s), (kseq / nstages) & 1);
#ifdef K3_FC1TRACE
      if (trace && tid == 0 && k < 16) {
        unsigned long long ts;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts));
        trace[(gridDim.x + blockIdx.x) * 32 + 16 + k] = ts;
      }
#endif
      if (k == 0 && tid == 0 && !kBlock) K3_MARK(3)
      const uint8_t* sw = stages + s * kStageBytes;
      const uint8_t* ssc = sw + kStageW + (c >> 2) * 256 + rg * 4 + (c & 3);
      const int sigma0 = 2 * (qs - W.qa);
      const int s0 = (rs - sigma0) & 3;
      float v[4];
#ifdef FC1_SERIAL
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const int sl = s0 + 4 * m;
        v[m] = 0.f;
        if (sl < 2 * n) {
          const uint4 wv = *reinterpret_cast<const uint4*>(sw + sl * kW13RowBytes + c * 16);
          const int rib = sl < n ? r0 + sl : 8 + r0 + sl - n;
          v[m] = ue8m0(ssc[rib * 16]) * dot32_h(wv, xr);
        }
      }
#else
      {
        // Four rows at once, branch-free (rows past the group read a valid row and are zeroed):
        // unconditional loads + 8 independent HFMA2 chains instead of 4 serialized dots.
        uint4 wv[4];
        float sc4[4];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
          const int sl = s0 + 4 * m;
          const int slc = min(sl, 2 * n - 1);
          wv[m] = *reinterpret_cast<const uint4*>(sw + slc * kW13RowBytes + c * 16);
          const int rib = slc < n ? r0 + slc : 8 + r0 + slc - n;
          sc4[m] = sl < 2 * n ? ue8m0(ssc[rib * 16]) : 0.f;
        }
        __half2 a0[4], a1[4];
#pragma unroll
        for (int m = 0; m < 4; ++m) a0[m] = a1[m] = __float2half2_rn(0.f);
#ifndef NO_MATH
#pragma unroll
        for (int kw = 0; kw < 4; ++kw) {
          uint32_t r[4][4];
#pragma unroll
          for (int m = 0; m < 4; ++m) {
            const uint32_t w = kw == 0 ? wv[m].x : kw == 1 ? wv[m].y : kw == 2 ? wv[m].z : wv[m].w;
            cvt_e2m1_word(w, r[m]);
          }
#pragma unroll
          for (int b = 0; b < 4; b += 2)
#pragma unroll
            for (int m = 0; m < 4; ++m) {
              a0[m] = __hfma2(*reinterpret_cast<const __half2*>(&r[m][b]), xr[kw * 4 + b], a0[m]);
              a1[m] = __hfma2(*reinterpret_cast<const __half2*>(&r[m][b + 1]), xr[kw * 4 + b + 1], a1[m]);
            }
        }
#endif
#pragma unroll
        for (int m = 0; m < 4; ++m) {
          const float2 f = __half22float2(__hadd2(a0[m], a1[m]));
          v[m] = sc4[m] * (f.x + f.y);
#ifdef NO_MATH
          v[m] = sc4[m] * __int_as_float((wv[m].x ^ wv[m].w) & 0x3fffffffu) + __half2float(xr[m].x);
#endif
        }
      }
#endif
      // Transpose-reduce 4 row partials over the 8 lanes sharing rs.
      const bool b4 = i8 & 4, b2 = i8 & 2;
      float k0 = b4 ? v[2] : v[0], k1 = b4 ? v[3] : v[1];
      k0 += __shfl_xor_sync(0xffffffffu, b4 ? v[0] : v[2], 4);
      k1 += __shfl_xor_sync(0xffffffffu, b4 ? v[1] : v[3], 4);
      float kp = b2 ? k1 : k0;
      kp += __shfl_xor_sync(0xffffffffu, b2 ? k0 : k1, 2);
      kp += __shfl_xor_sync(0xffffffffu, kp, 1);
      const int sl = s0 + 4 * ((b4 ? 2 : 0) + (b2 ? 1 : 0));
      if (!(i8 & 1) && sl < 2 * n) red[warp * max_slots + sigma0 + sl] = kp;  // one writer per (warp, slot)
      __syncwarp();
      if (lane == 0) mbar_arrive(empty_bar(s));
    }
    if (tid == 0) K3_MARK(11)
    // Before the first global write (h, then gemm2_out): the outputs may reuse memory the PDL
    // predecessor was still reading. Free here: the producer finished long ago.
    if constexpr (kLamport) asm volatile("griddepcontrol.wait;" ::: "memory");
    consumer_sync();
    // FC1 epilogue: h = SiTU(gate, up) for this CTA's pairs.
    for (int kk = tid; kk < W.qb - W.qa; kk += kConsumerThreads) {
      const int q = W.qa + kk;
      const int gs = max(W.qa, q & ~7), ge = min(W.qb, (q & ~7) + 8);
      const int su = 2 * (gs - W.qa) + (q - gs);
      const int sg = su + (ge - gs);
      float up = 0.f, gate = 0.f;
#pragma unroll
      for (int w = 0; w < kFc1Warps; ++w) {  // fixed warp (K-range) order
        up += red[w * max_slots + su];
        gate += red[w * max_slots + sg];
      }
      const int t = q / kPairsPerTok, j = (q / kInter) % kTopK, u = q % kInter;
      const int b = u >> 3, r = u & 7;
      const int i = 16 * (b >> 1) + 2 * r + (b & 1);
      h[t * kPairsPerTok + j * kInter + i] = __float2half_rn(situ(up, gate, beta, linear_beta));
    }
    consumer_sync();
    if (tid == 0 && !routed_only) {
      K3_MARK(4)
      // Every call adds exactly G to the counter, so this call's target is the next
      // multiple of G above the value read at start (our own add is still pending).
      asm volatile("fence.acq_rel.gpu;\n\tred.relaxed.gpu.global.add.u64 [%0], 1;" ::"l"(counter) : "memory");
      unsigned long long cur;
      do {
        asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(cur) : "l"(counter) : "memory");
      } while (cur < bar_target);
      K3_MARK(5)
    }
    consumer_sync();
    if (kLamport && blockIdx.x == 0 && !routed_only) {
      // Every CTA polled its x rows before the barrier: re-arm mailbox rows [0, M).
      const uint4 empty = make_uint4(0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u);
      for (int f = tid; f < M * (kHidden / 8); f += kConsumerThreads)
        asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(reinterpret_cast<const uint4*>(x) + f),
                     "r"(empty.x), "r"(empty.y), "r"(empty.z), "r"(empty.w)
                     : "memory");
    }
    asm volatile("griddepcontrol.launch_dependents;");
    // Stage h (all tokens) into shared memory (over x).
    __half* hs = reinterpret_cast<__half*>(xs);
    if (!routed_only)
      for (int v = tid; v < M * kPairsPerTok / 8; v += kConsumerThreads)
        reinterpret_cast<uint4*>(hs)[v] = __ldcg(reinterpret_cast<const uint4*>(h) + v);
    consumer_sync();
    if (tid == 0) K3_MARK(6)

    // FC2: warp = expert slot j; lane = (unit ui, row rho, K-half hf) with 3 of the 6 K-chunks;
    // h[t, j, K-half] stays in registers while t is unchanged.
    const int ui = lane >> 4, rho = (lane >> 1) & 7, hf = lane & 1;
    const int j = warp;
    if (tid == 0 && trace) trace[blockIdx.x * 32 + 20] = clock64();
    __half2 hr[3][16];
    int h_t = -1;
    for (int f = 0; f < W.nf2; ++f) {
      const int k = nsd + W.ngroups + f;
      const int s = k % nstages;
      const int unit = 2 * f + ui;
      const bool valid = unit < W.nunits;
      const int t = valid ? unit / W.nocts : 0;
      const int o = W.oa + (valid ? unit % W.nocts : 0);
      const int p = 8 * o + rho;
      const int rg = (p & 127) >> 5;
      if (valid && t != h_t) {
        const uint4* src = reinterpret_cast<const uint4*>(hs + (t * kTopK + j) * kInter + 3 * hf * 32);
#pragma unroll
        for (int m = 0; m < 3; ++m)
#pragma unroll
          for (int vv = 0; vv < 4; ++vv) {
            const uint4 raw = src[m * 4 + vv];
            const __half2* pp = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
            for (int uu = 0; uu < 4; ++uu) hr[m][vv * 4 + uu] = pp[uu];
          }
        h_t = t;
      }
      mbar_wait(full_bar(s), (k / nstages) & 1);
      if (f == 0 && tid == 0) K3_MARK(10)
      if (f == 0 && tid == 0 && trace) trace[blockIdx.x * 32 + 21] = clock64();
      const uint8_t* base = stages + s * kStageBytes + ui * kF2Unit;
      float acc = 0.f;
      if (valid) {
        uint4 wv[3];
        uint8_t sc[3];
#pragma unroll
        for (int m = 0; m < 3; ++m) {
          const int cc = 3 * hf + m;
          wv[m] = *reinterpret_cast<const uint4*>(base + (j * 8 + rho) * 96 + cc * 16);
          sc[m] = base[kF2W + j * 256 + (cc >> 2) * 128 + rho * 16 + rg * 4 + (cc & 3)];
        }
#pragma unroll
        for (int m = 0; m < 3; ++m) {
#ifdef NO_MATH2
          acc += __int_as_float((wv[m].x ^ wv[m].z) & 0x3fffffff) * __half2float(hr[m][m].x) + sc[m];
#else
          acc = fmaf(ue8m0(sc[m]), dot32_h(wv[m], hr[m]), acc);
#endif
        }
      }
      if (f == 0 && tid == 0 && trace) trace[blockIdx.x * 32 + 22] = clock64() + (acc == 1.2345f);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty_bar(s));
      acc += __shfl_xor_sync(0xffffffffu, acc, 1);
      if (valid && hf == 0) {
        if (kFinal) {
          acc2[j * max_rows2 + unit * 8 + rho] = wsm[t * kTopK + j] * acc;  // one writer per (slot, row)
        } else if (tail) {
          // lane row rho = 2b' + r -> output column 8o + 4r + b' (see the producer's 4D boxes)
          reinterpret_cast<__nv_bfloat16*>(tsm + TL.fin)[(unit * kTopK + j) * 8 + 4 * (rho & 1) + (rho >> 1)] =
              __float2bfloat16(acc);
        } else {
          const int n_out = 32 * (p >> 5) + 4 * (p & 7) + ((p & 31) >> 3);
          out[static_cast<long>(t * kTopK + j) * kHidden + n_out] = __float2bfloat16(acc);
        }
      }
      if (f == 0 && tid == 0) K3_MARK(18)
      if (f == 0 && tid == 0 && trace) trace[blockIdx.x * 32 + 23] = clock64();
    }
    if (kFinal) {
      consumer_sync();
      for (int i = tid; i < W.nunits * 8; i += kConsumerThreads) {
        const int unit = i >> 3, r = i & 7;
        const int t = unit / W.nocts, o = W.oa + unit % W.nocts;
        const int p = 8 * o + r;
        const int n_out = 32 * (p >> 5) + 4 * (p & 7) + ((p & 31) >> 3);
        float v = 0.f;
#pragma unroll
        for (int jj = 0; jj < kTopK; ++jj) v += acc2[jj * max_rows2 + i];  // fixed expert-slot order
        out[t * kHidden + n_out] = __float2bfloat16(v);
      }
    }
    if (tail) {
      consumer_sync();  // every FC2 warp's rows are in `fin`
      moe_publish_latent(reinterpret_cast<const __nv_bfloat16*>(tsm + TL.fin), wsm, W.nunits, W.nocts, W.oa,
                         *reinterpret_cast<const TailArgs*>(tsm + TL.args), tid);
      if (tid == 0) K3_MARK(26)
    }
  }
  if (trace) {
#if defined(K3_TRACE_CLOCK) || defined(K3_TRACE_SMEM)
    consumer_sync();
    if (tid < 32) trace[blockIdx.x * 32 + tid] = trsm[tid];
#endif
    if (tid == 0) K3_MARK(15)
    consumer_sync();
    if (tid == 0) K3_MARK(7)
    if (tid == 0) trace[blockIdx.x * 32 + 17] = clock64();
  }
#undef K3_MARK
}

template <typename Kernel, typename... Args>
void launch_pdl(Kernel kernel, dim3 grid, dim3 block, cudaStream_t stream, Args... args) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, kernel, args...);
}

}  // namespace

void moe_small(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_weights,
               torch::Tensor w13, torch::Tensor w13_scale, torch::Tensor w2,
               torch::Tensor w2_scale, torch::Tensor workspace, torch::Tensor out, double beta,
               double linear_beta, int64_t stages) {
  const int M = x.size(0);
  TORCH_CHECK(x.size(1) == kHidden && x.is_contiguous() && x.scalar_type() == at::kBFloat16);
  TORCH_CHECK(topk_ids.size(1) == kTopK && topk_ids.scalar_type() == at::kInt && topk_ids.is_contiguous());
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat && topk_weights.is_contiguous());
  TORCH_CHECK(w13.size(1) == kW13Rows && w13.size(2) == kW13RowBytes);
  TORCH_CHECK(w2.size(1) == kHidden && w2.size(2) == kW2RowBytes);
  TORCH_CHECK(w13_scale.numel() == w13.size(0) * kW13Rows * kW13ScaleCols);
  TORCH_CHECK(w2_scale.numel() == w2.size(0) * kHidden * kW2ScaleCols);
  TORCH_CHECK(workspace.numel() * workspace.element_size() >= M * kTopK * kInter * 2);
  TORCH_CHECK(out.size(0) == M && out.size(1) == kHidden && out.scalar_type() == at::kBFloat16);
  if (M == 0) return;
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  auto u8 = [](const torch::Tensor& t) { return reinterpret_cast<const uint8_t*>(t.data_ptr()); };
  static int num_sms = 0;
  if (num_sms == 0) cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, x.get_device());
  const int grid_x = std::max(1, (2 * num_sms + M - 1) / M);
  if (stages & 1) launch_pdl(fc1_situ_kernel, dim3(grid_x, M), dim3(kWarps * 32), stream,
             reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), topk_ids.data_ptr<int>(),
             u8(w13), u8(w13_scale), reinterpret_cast<__half*>(workspace.data_ptr()), static_cast<float>(beta),
             static_cast<float>(linear_beta));
  if (stages & 2) launch_pdl(fc2_finalize_kernel, dim3(grid_x, M), dim3(kWarps * 32), stream,
             reinterpret_cast<const __half*>(workspace.data_ptr()), topk_ids.data_ptr<int>(),
             static_cast<const float*>(topk_weights.data_ptr<float>()), u8(w2), u8(w2_scale),
             reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {

PFN_cuTensorMapEncodeTiled_v12000 tensor_map_encoder() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (fn == nullptr) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q));
    TORCH_CHECK(p != nullptr && q == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled unavailable");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}

// uint8 tensor map; dims/box innermost first, strides in bytes for dims 1..rank-1.
CUtensorMap make_u8_map(const void* base, int rank, const cuuint64_t* dims, const cuuint64_t* strides,
                        const cuuint32_t* box) {
  CUtensorMap m;
  const cuuint32_t es[5] = {1, 1, 1, 1, 1};
  const CUresult r = tensor_map_encoder()(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, rank, const_cast<void*>(base), dims,
                                          strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                          CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
  return m;
}

template <bool kFinal, bool kLamport = false, bool kBlock = false>
void launch_fused(const torch::Tensor& x, const torch::Tensor& topk_ids, const float* topk_w,
                  const torch::Tensor& w13, const torch::Tensor& w13_scale, const torch::Tensor& w2,
                  const torch::Tensor& w2_scale, const torch::Tensor& workspace, const torch::Tensor& barrier,
                  __nv_bfloat16* out, double beta, double linear_beta,
                  const std::optional<torch::Tensor>& trace, int flags = 0, BlockArgs ba = BlockArgs{},
                  int grid = 0) {
  // x: [M, 3584] bf16, or with kLamport the mailbox (>= M rows of 3584 bf16).
  const int M = topk_ids.size(0);
  TORCH_CHECK(M >= 1 && M <= kMaxFusedM, "moe_fused supports 1..8 tokens");
  TORCH_CHECK(x.size(-1) == kHidden && x.is_contiguous() && x.scalar_type() == at::kBFloat16 &&
              x.numel() >= static_cast<long>(M) * kHidden && x.is_cuda());
  TORCH_CHECK(kLamport || x.numel() == static_cast<long>(M) * kHidden);
  TORCH_CHECK(topk_ids.size(0) == M && topk_ids.size(1) == kTopK && topk_ids.scalar_type() == at::kInt &&
              topk_ids.is_contiguous());
  TORCH_CHECK(w13.dim() == 3 && w13.size(1) == kW13Rows && w13.size(2) == kW13RowBytes && w13.is_contiguous());
  TORCH_CHECK(w2.dim() == 3 && w2.size(1) == kHidden && w2.size(2) == kW2RowBytes && w2.is_contiguous());
  const long E = w13.size(0);
  TORCH_CHECK(w2.size(0) == E && w13_scale.is_contiguous() && w2_scale.is_contiguous());
  TORCH_CHECK(w13_scale.numel() == E * kW13Rows * kW13ScaleCols);
  TORCH_CHECK(w2_scale.numel() == E * kHidden * kW2ScaleCols);
  TORCH_CHECK(workspace.numel() * workspace.element_size() >= M * kTopK * kInter * 2);
  TORCH_CHECK(barrier.scalar_type() == at::kLong && barrier.numel() >= kMaxFusedM);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  static int sms = 0;
  if (sms == 0) {
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, x.get_device());
    cudaFuncSetAttribute(moe_fused_kernel<true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448);
    cudaFuncSetAttribute(moe_fused_kernel<false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448);
    cudaFuncSetAttribute(moe_fused_kernel<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448);
    cudaFuncSetAttribute(moe_fused_kernel<false, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448);
  }
  const int G = grid > 0 ? std::min(grid, sms) : sms;
  // Tensor maps (host-side encode, a few hundred ns; captured by value in CUDA graphs).
  // FC1 scales: per expert 4 bands x 28 col-tiles x 512 B; box = 16 rows (256 B) x 28 tiles.
  const cuuint64_t d13[3] = {512, 28, static_cast<cuuint64_t>(4 * E)};
  const cuuint64_t s13[2] = {512, 28 * 512};
  const cuuint32_t b13[3] = {256, 28, 1};
  const CUtensorMap tm_w13s = make_u8_map(w13_scale.data_ptr(), 3, d13, s13, b13);
  // FC2 weights: [E*3584 rows, 128 B]; box = 8 rows x 96 B (skips the K padding).
  const cuuint64_t d2[2] = {static_cast<cuuint64_t>(kW2RowBytes), static_cast<cuuint64_t>(E * kHidden)};
  const cuuint64_t s2[1] = {static_cast<cuuint64_t>(kW2RowBytes)};
  const cuuint32_t b2[2] = {96, 8};
  const CUtensorMap tm_w2 = make_u8_map(w2.data_ptr(), 2, d2, s2, b2);
  // FC2 scales: per expert 28 bands x 2 col-tiles x 512 B; box = 8 rows (128 B) x 2 tiles.
  const cuuint64_t d2s[3] = {512, 2, static_cast<cuuint64_t>(28 * E)};
  const cuuint64_t s2s[2] = {512, 1024};
  const cuuint32_t b2s[3] = {128, 2, 1};
  const CUtensorMap tm_w2s = make_u8_map(w2_scale.data_ptr(), 3, d2s, s2s, b2s);

  const int max_slots = 2 * ((M * kPairsPerTok + G - 1) / G) + 16;
  const int max_rows2 = M * 8 * ((kOctets + G - 1) / G);
  const int block_extra =
      kBlock ? (kBlockMaxM * kTopK + kBlockMaxM * kTopkScratch + kBlockMaxM * kTopK) * 4 + 16 + M * kSdK * 4 : 0;
  const int fixed = M * kHidden * 2 + (kFc1Warps * max_slots + (kFinal ? kTopK : 1) * max_rows2 + M * kTopK) * 4 + block_extra + 16 +
                    (2 + 2 * 8) * 8 + 128;
  const int nstages = std::min(8, (232448 - fixed) / kStageBytes);
  TORCH_CHECK(nstages >= 2);
  const size_t smem = fixed + static_cast<size_t>(nstages) * kStageBytes;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(G);
  cfg.blockDim = dim3(kFusedThreads);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(
      &cfg, moe_fused_kernel<kFinal, kLamport, kBlock>, tm_w13s, tm_w2, tm_w2s,
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), static_cast<const int*>(topk_ids.data_ptr<int>()),
      topk_w, reinterpret_cast<const uint8_t*>(w13.data_ptr()), reinterpret_cast<__half*>(workspace.data_ptr()),
      reinterpret_cast<unsigned long long*>(barrier.data_ptr()), out, M, nstages, static_cast<float>(beta),
      static_cast<float>(linear_beta), flags,
      trace ? reinterpret_cast<unsigned long long*>(trace->data_ptr()) : nullptr, ba));
}

#if defined(K3_TRACE_CLOCK) || defined(K3_TRACE_SMEM)
constexpr int kMaxDynSmem = 232448 - 256;
#else
constexpr int kMaxDynSmem = 232448;
#endif
template <bool kFinal, bool kLamport = false, bool kBlock = false, bool kTail = false>
void launch_fused_pub(const torch::Tensor& x, const torch::Tensor& topk_ids, const float* topk_w,
                  const torch::Tensor& w13, const torch::Tensor& w13_scale, const torch::Tensor& w2,
                  const torch::Tensor& w2_scale, const torch::Tensor& workspace, const torch::Tensor& barrier,
                  __nv_bfloat16* out, double beta, double linear_beta,
                  const std::optional<torch::Tensor>& trace, int flags = 0, BlockArgs ba = BlockArgs{},
                  int grid = 0) {
  // x: [M, 3584] bf16, or with kLamport the mailbox (>= M rows of 3584 bf16).
  const int M = topk_ids.size(0);
  TORCH_CHECK(M >= 1 && M <= kMaxFusedM, "moe_fused supports 1..8 tokens");
  TORCH_CHECK(x.size(-1) == kHidden && x.is_contiguous() && x.scalar_type() == at::kBFloat16 &&
              x.numel() >= static_cast<long>(M) * kHidden && x.is_cuda());
  TORCH_CHECK(kLamport || x.numel() == static_cast<long>(M) * kHidden);
  TORCH_CHECK(topk_ids.size(0) == M && topk_ids.size(1) == kTopK && topk_ids.scalar_type() == at::kInt &&
              topk_ids.is_contiguous());
  TORCH_CHECK(w13.dim() == 3 && w13.size(1) == kW13Rows && w13.size(2) == kW13RowBytes && w13.is_contiguous());
  TORCH_CHECK(w2.dim() == 3 && w2.size(1) == kHidden && w2.size(2) == kW2RowBytes && w2.is_contiguous());
  const long E = w13.size(0);
  TORCH_CHECK(w2.size(0) == E && w13_scale.is_contiguous() && w2_scale.is_contiguous());
  TORCH_CHECK(w13_scale.numel() == E * kW13Rows * kW13ScaleCols);
  TORCH_CHECK(w2_scale.numel() == E * kHidden * kW2ScaleCols);
  TORCH_CHECK(workspace.numel() * workspace.element_size() >= M * kTopK * kInter * 2);
  TORCH_CHECK(barrier.scalar_type() == at::kLong && barrier.numel() >= kMaxFusedM);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  static int sms = 0;
  if (sms == 0) {
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, x.get_device());
#ifndef K3_NO_TAIL_MODULE
    cudaFuncSetAttribute(moe_fused_pub_kernel<false, true, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kMaxDynSmem);
#endif
  }
  const int G = grid > 0 ? std::min(grid, sms) : sms;
  constexpr bool tail = kBlock && kTail;
  TORCH_CHECK(!tail || ba.tail.lat_mb != nullptr);
  TORCH_CHECK(!tail || M <= kPubMaxM, "moe_block_lamport publish mode supports M <= 4");
  // Tensor maps (host-side encode, a few hundred ns; captured by value in CUDA graphs).
  // FC1 scales: per expert 4 bands x 28 col-tiles x 512 B; box = 16 rows (256 B) x 28 tiles.
  const cuuint64_t d13[3] = {512, 28, static_cast<cuuint64_t>(4 * E)};
  const cuuint64_t s13[2] = {512, 28 * 512};
  const cuuint32_t b13[3] = {256, 28, 1};
  const CUtensorMap tm_w13s = make_u8_map(w13_scale.data_ptr(), 3, d13, s13, b13);
  CUtensorMap tm_w2, tm_w2s;
  if (!tail) {
    // FC2 weights: [E*3584 rows, 128 B]; box = 8 rows x 96 B (skips the K padding).
    const cuuint64_t d2[2] = {static_cast<cuuint64_t>(kW2RowBytes), static_cast<cuuint64_t>(E * kHidden)};
    const cuuint64_t s2[1] = {static_cast<cuuint64_t>(kW2RowBytes)};
    const cuuint32_t b2[2] = {96, 8};
    tm_w2 = make_u8_map(w2.data_ptr(), 2, d2, s2, b2);
    // FC2 scales: per expert 28 bands x 2 col-tiles x 512 B; box = 8 rows (128 B) x 2 tiles.
    const cuuint64_t d2s[3] = {512, 2, static_cast<cuuint64_t>(28 * E)};
    const cuuint64_t s2s[2] = {512, 1024};
    const cuuint32_t b2s[3] = {128, 2, 1};
    tm_w2s = make_u8_map(w2_scale.data_ptr(), 3, d2s, s2s, b2s);
  } else {
    // Tail units = 8 consecutive OUTPUT columns: physical rows 32 blk + 8 b' + 2 g + {0, 1}.
    // Weights viewed as [E*112 blk][4 b'][8 r][128 B]; box = 96 B x 2 r x 4 b' x 1.
    const cuuint64_t d2[4] = {static_cast<cuuint64_t>(kW2RowBytes), 8, 4, static_cast<cuuint64_t>(E * (kHidden / 32))};
    const cuuint64_t s2[3] = {128, 1024, 4096};
    const cuuint32_t b2[4] = {96, 2, 4, 1};
    tm_w2 = make_u8_map(w2.data_ptr(), 4, d2, s2, b2);
    // Scales viewed as [E*28 tiles][2 cb][4 b'][128 B]; box = 32 B (the 2 rows of g) x 4 b' x 2 cb.
    const cuuint64_t d2s[4] = {128, 4, 2, static_cast<cuuint64_t>(28 * E)};
    const cuuint64_t s2s[3] = {128, 512, 1024};
    const cuuint32_t b2s[4] = {32, 4, 2, 1};
    tm_w2s = make_u8_map(w2_scale.data_ptr(), 4, d2s, s2s, b2s);
  }

  const int max_slots = 2 * ((M * kPairsPerTok + G - 1) / G) + 16;
  const int max_rows2 = M * 8 * ((kOctets + G - 1) / G);
  const int block_extra =
      (kBlock ? (kBlockMaxM * kTopK + kBlockMaxM * kTopkScratch + kBlockMaxM * kTopK) * 4 + 16 + M * kSdK * 4 : 0) +
      (tail ? tail_layout(M, G).bytes + 16 : 0);
#if defined(K3_TRACE_CLOCK) || defined(K3_TRACE_SMEM)
  constexpr int kTraceSmem = 256;  // static smem of the buffered trace marks
#else
  constexpr int kTraceSmem = 0;
#endif
  const int fixed = M * kHidden * 2 + (kFc1Warps * max_slots + (kFinal ? kTopK : 1) * max_rows2 + M * kTopK) * 4 + block_extra + (tail ? 64 : 16) +
                    kTraceSmem + (2 + 2 * 8) * 8 + 128;
  const int nstages = std::min(8, (kMaxDynSmem - fixed) / kStageBytes);
  TORCH_CHECK(nstages >= 2);
  const size_t smem = fixed + static_cast<size_t>(nstages) * kStageBytes;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(G);
  cfg.blockDim = dim3(kFusedThreads);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  // Placement only (the kernel uses no cluster features): K3MK_MOE_CLUSTER=8 with grid=120 launches 15
  // clusters of 8 CTAs, which leaves the 32 SMs that 8-clusters cannot use free for a co-resident TAIL.
  static const int moe_cl = getenv("K3MK_MOE_CLUSTER") ? atoi(getenv("K3MK_MOE_CLUSTER")) : 0;
  if (moe_cl > 1 && G % moe_cl == 0) {
    attr[1].id = cudaLaunchAttributeClusterDimension;
    attr[1].val.clusterDim.x = moe_cl;
    attr[1].val.clusterDim.y = 1;
    attr[1].val.clusterDim.z = 1;
    cfg.numAttrs = 2;
  }
  C10_CUDA_CHECK(cudaLaunchKernelEx(
      &cfg, moe_fused_pub_kernel<kFinal, kLamport, kBlock, kTail>, tm_w13s, tm_w2, tm_w2s,
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), static_cast<const int*>(topk_ids.data_ptr<int>()),
      topk_w, reinterpret_cast<const uint8_t*>(w13.data_ptr()), reinterpret_cast<__half*>(workspace.data_ptr()),
      reinterpret_cast<unsigned long long*>(barrier.data_ptr()), out, M, nstages, static_cast<float>(beta),
      static_cast<float>(linear_beta), flags,
      trace ? reinterpret_cast<unsigned long long*>(trace->data_ptr()) : nullptr, ba));
}


// ===========================================================================
// k3moe.route_shared: Kimi K3 router gate GEMV + grouped top-k (sigmoid, 1 group) +
// shared-expert gate_up GEMV + SiTU, one PDL kernel for decode batches M <= 2.
//
// CTA s owns router rows [896 s/G, 896 (s+1)/G) and shared pairs [384 s/G, 384 (s+1)/G)
// (gate row i, up row 384+i of this rank's [768, 7168] gate_up weight): <= 12 rows of
// 7168 bf16, TMA-prefetched into smem BEFORE griddepcontrol.wait (weights are static).
// After the wait: x -> smem, one 16-row x 8-token tile per CTA on the tensor cores
// (mma.m16n8k16 bf16, fp32 accumulate; K split over the 16 warps, reduced in smem).
// Router rows publish (sigmoid(logit), sigmoid(logit) + bias) into a Lamport score
// buffer (empty = 0xFFFFFFFF words); CTA 0 polls all 896 per token, re-arms them, runs an
// exact radix select for the top-16 of score + bias (ties -> lower expert id), and writes
// ids int32 [M,16] (descending score + bias) and weights bf16 [M,16] = score / sum(score)
// * routed_scaling_factor. Shared pairs write h = bf16(situ(bf16(g), bf16(u))) [M,384].
// ===========================================================================
constexpr int kRtE = 896;             // routed experts
constexpr int kRtH = 7168;            // hidden
constexpr int kRtSh = 384;            // shared intermediate per rank (6144 / TP16)
constexpr int kRtMaxM = 4;             // scores mode (select mode: <= 2)
constexpr int kRtSelMaxM = 2;
constexpr int kRtRows = 11;           // flat rows (896 router + 768 shared gate_up) / 152 CTAs
constexpr int kRtRowStride = kRtH * 2 + 64;  // bank-conflict padding
constexpr int kRtThreads = 512;
constexpr int kRtWarps = kRtThreads / 32;
constexpr int kRtKPerWarp = kRtH / kRtWarps;  // 448
constexpr unsigned kRtEmpty = 0xffffffffu;
constexpr int kRtReplicas = 8;  // scores-only mode: 8 copies, so 152 readers don't hot-spot 56 L2 lines

__device__ __forceinline__ void mma_bf16_16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                               uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}


__global__ void __launch_bounds__(kRtThreads, 1)
route_shared_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ gate_w,
                    const float* __restrict__ bias, const __nv_bfloat16* __restrict__ sh_w13,
                    uint2* __restrict__ sbuf, int* __restrict__ ids, __nv_bfloat16* __restrict__ wts,
                    __nv_bfloat16* __restrict__ h_sh, int M, float beta, float linear_beta, int renorm,
                    float scale, int select, unsigned long long* __restrict__ trace) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* rows = smem;                                                     // [12][kRtRowStride]
  // smem: rows [11][stride] | x [M][7168] | part [16 warps][16][M] | rowv [16][M] |
  //       (select mode) keys [M][896] | scs [M][896] | hist [M][256] | misc [M][4] | lst [M][16] | bar
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem + kRtRows * kRtRowStride);
  float* part = reinterpret_cast<float*>(xs + M * kRtH);
  float* rowv = part + kRtWarps * 16 * M;
  uint32_t* keys = reinterpret_cast<uint32_t*>(rowv + 16 * M);
  const int selM = select ? M : 0;
  float* scs = reinterpret_cast<float*>(keys + selM * kRtE);
  unsigned* hist = reinterpret_cast<unsigned*>(scs + selM * kRtE);
  unsigned* misc = hist + selM * 256;
  int* lst_i = reinterpret_cast<int*>(misc + selM * 4);
  uint64_t* bar = reinterpret_cast<uint64_t*>(
      (reinterpret_cast<uintptr_t>(lst_i + selM * 16) + 15) & ~static_cast<uintptr_t>(15));

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int G = gridDim.x, s = blockIdx.x;
  // Flat rows: 0..895 router, 896..1663 shared gate_up (gate 0..383, up 384..767).
  const int total_rows = sh_w13 ? kRtE + 2 * kRtSh : kRtE;
  const int fa = total_rows * s / G, fb = total_rows * (s + 1) / G;
  const int nrows = fb - fa;
#define RT_CK(slot)                                                             \
  if (trace && tid == 0 && s == 0) trace[G * 8 + (slot)] = clock64();
#define RT_MARK(slot)                                                           \
  if (trace && tid == 0) {                                                      \
    unsigned long long ts;                                                      \
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts));                     \
    trace[s * 8 + (slot)] = ts;                                                 \
  }
  RT_MARK(0)
  const uint32_t mb = smem_u32(bar);
  if (tid == 0) {
    mbar_init(mb, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    // Static weights: prefetch before waiting on the predecessor.
    mbar_arrive_expect_tx(mb, nrows * kRtH * 2);
    for (int r = 0; r < nrows; ++r) {
      const int gr = fa + r;
      const __nv_bfloat16* src = gr < kRtE ? gate_w + static_cast<long>(gr) * kRtH
                                           : sh_w13 + static_cast<long>(gr - kRtE) * kRtH;
      tma_1d(smem_u32(rows + r * kRtRowStride), src, kRtH * 2, mb);
    }
  }
  // Static per-row bias, loaded before the wait as well.
  const float my_bias = tid < 16 * M && tid / M < nrows && fa + tid / M < kRtE ? __ldg(bias + fa + tid / M) : 0.f;
  __syncthreads();
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");
  RT_MARK(1)
  for (int v = tid; v < M * kRtH / 8; v += kRtThreads)
    reinterpret_cast<uint4*>(xs)[v] = __ldcg(reinterpret_cast<const uint4*>(x) + v);
  __syncthreads();
  mbar_wait(mb, 0);
  RT_MARK(2)

  // ---- 16-row x 8-token tile, K split across warps -------------------------------------
  RT_CK(8)
  {
    const int g = lane >> 2, tig = lane & 3;
    // Rows >= nrows and token columns >= M contribute nothing: skip their smem loads (the
    // GEMV is smem-bandwidth bound).
    const bool v0 = g < nrows, v1 = g + 8 < nrows, vx = g < M;
    const int r0 = v0 ? g : 0, r1 = v1 ? g + 8 : 0;
    const int xt = vx ? g : 0;
    const uint8_t* pa0 = rows + r0 * kRtRowStride;
    const uint8_t* pa1 = rows + r1 * kRtRowStride;
    const uint8_t* pb = reinterpret_cast<const uint8_t*>(xs + xt * kRtH);
    // Legacy mma.sync on sm_100 has ~130-cycle latency: keep 8 independent accumulators.
    float acc[8][4];
#pragma unroll
    for (int a = 0; a < 8; ++a)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[a][q] = 0.f;
#pragma unroll
    for (int it = 0; it < kRtKPerWarp / 32; ++it) {
      const int kb = (warp * kRtKPerWarp + it * 32 + 8 * tig) * 2;
      const uint4 z = make_uint4(0u, 0u, 0u, 0u);
      const uint4 a_lo = v0 ? *reinterpret_cast<const uint4*>(pa0 + kb) : z;
      const uint4 a_hi = v1 ? *reinterpret_cast<const uint4*>(pa1 + kb) : z;
      const uint4 bv = vx ? *reinterpret_cast<const uint4*>(pb + kb) : z;
      mma_bf16_16816(acc[(2 * it) & 7], a_lo.x, a_hi.x, a_lo.y, a_hi.y, bv.x, bv.y);
      mma_bf16_16816(acc[(2 * it + 1) & 7], a_lo.z, a_hi.z, a_lo.w, a_hi.w, bv.z, bv.w);
    }
    float c[4];
#pragma unroll
    for (int q = 0; q < 4; ++q)
      c[q] = ((acc[0][q] + acc[1][q]) + (acc[2][q] + acc[3][q])) + ((acc[4][q] + acc[5][q]) + (acc[6][q] + acc[7][q]));
    if (tid == 0 && s == 0 && trace) trace[G * 8 + 9] = clock64() + (c[0] == 1.2345f);
    // c0 = (row g, token 2tig), c1 = (g, 2tig+1), c2 = (g+8, 2tig), c3 = (g+8, 2tig+1)
    float* pw = part + warp * 16 * M;
    for (int q = 0; q < 2; ++q) {
      const int n = 2 * tig + q;
      if (n < M) {
        pw[g * M + n] = c[q];
        pw[(g + 8) * M + n] = c[2 + q];
      }
    }
  }
  __syncthreads();
  RT_CK(10)
  if (tid < 16 * M) {
    const int r = tid / M, n = tid % M;
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < kRtWarps; ++w) v += part[(w * 16 + r) * M + n];
    rowv[r * M + n] = v;
    const int gr = fa + r;
    if (r < nrows && gr >= kRtE) h_sh[n * (2 * kRtSh) + gr - kRtE] = __float2bfloat16(v);  // gate_up (bf16)
    if (r < nrows && gr < kRtE) {
      const float sc = 1.f / (1.f + expf(-v));
      float sb = sc + my_bias;
      uint2 pk = make_uint2(__float_as_uint(sc), __float_as_uint(sb));
      if (pk.x == kRtEmpty || pk.y == kRtEmpty || isnan(sc) || isnan(sb))  // never publish the marker
        pk = make_uint2(0u, __float_as_uint(-INFINITY));
      if (select) {
        asm volatile("st.global.v2.u32 [%0], {%1, %2};" ::"l"(sbuf + n * kRtE + gr), "r"(pk.x), "r"(pk.y)
                     : "memory");
      } else {
#pragma unroll
        for (int c = 0; c < kRtReplicas; ++c)
          asm volatile("st.global.v2.u32 [%0], {%1, %2};" ::"l"(sbuf + (c * M + n) * kRtE + gr), "r"(pk.x),
                       "r"(pk.y)
                       : "memory");
      }
    }
  }
  __syncthreads();
  RT_CK(11)
  RT_MARK(3)
  if (s != 0 || !select) return;

  // ---- CTA 0: gather the Lamport scores and select the top-16 per token -------------------
  const int t = tid >> 8, lt = tid & 255;  // token group of 256 threads
  const bool active = t < M;
  uint32_t k4[4];
  float v4[4];
  if (active) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int i = lt + 256 * j;
      k4[j] = 0u;
      v4[j] = -INFINITY;
      if (i < kRtE) {
        const uint2* src = sbuf + t * kRtE + i;
        uint2 v;
        do {
          asm volatile("ld.volatile.global.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(src) : "memory");
        } while (v.x == kRtEmpty || v.y == kRtEmpty);
        asm volatile("st.global.v2.u32 [%0], {%1, %1};" ::"l"(src), "r"(kRtEmpty) : "memory");  // re-arm
        v4[j] = __uint_as_float(v.y);
        k4[j] = order_key(v4[j]);
        keys[t * kRtE + i] = k4[j];
        scs[t * kRtE + i] = __uint_as_float(v.x);
      }
    }
  }
  RT_MARK(4)
  RT_CK(0)
  // Fast path: one 1024-bin histogram of score+bias over [min, max]; the bins at or above the
  // one holding the 16th largest give <= ~20 candidates, ranked exactly (ties -> lower id).
  // Falls back to an exact MSB-first radix select if the candidate set is too large.
  float* red = reinterpret_cast<float*>(hist);            // [2][8][2] (min/max per warp)
  unsigned* hist2 = reinterpret_cast<unsigned*>(xs);      // [2][1024] (x is no longer needed)
  __shared__ int s_ci[kRtSelMaxM][64];
  __shared__ uint32_t s_ck[kRtSelMaxM][64];
  __shared__ int s_fast;
  const int wg = lt >> 5;
  float mx = -INFINITY, mn = INFINITY;
  if (active) {
#pragma unroll
    for (int j = 0; j < 4; ++j)
      if (lt + 256 * j < kRtE) {
        mx = fmaxf(mx, v4[j]);
        mn = fminf(mn, v4[j]);
      }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
      mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, o));
    }
    if (lane == 0) {
      red[(t * 8 + wg) * 2 + 0] = mx;
      red[(t * 8 + wg) * 2 + 1] = mn;
    }
    for (int b = lt; b < 1056; b += 256) hist2[t * 1056 + b] = 0;
  }
  if (tid == 0) s_fast = 1;
  __syncthreads();
  RT_CK(1)
  float inv = 0.f;
  if (active) {
    mx = red[(t * 8) * 2];
    mn = red[(t * 8) * 2 + 1];
    for (int w = 1; w < 8; ++w) {
      mx = fmaxf(mx, red[(t * 8 + w) * 2]);
      mn = fminf(mn, red[(t * 8 + w) * 2 + 1]);
    }
    const bool ok = mx > mn && isfinite(mx) && isfinite(mn);
    if (!ok && lt == 0) s_fast = 0;
    inv = ok ? 1023.f / (mx - mn) : 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const bool in = ok && lt + 256 * j < kRtE;
      if (in) {  // bins are spread out here: plain shared atomics (bank-skewed index)
        const unsigned bin = min(1023u, static_cast<unsigned>((v4[j] - mn) * inv));
        atomicAdd(&hist2[t * 1056 + bin + (bin >> 5)], 1u);
      }
    }
  }
  __syncthreads();
  RT_CK(2)
  if (active && lt < 32) {
    const unsigned* hb = hist2 + t * 1056;
    auto hbin = [&](int bin) { return hb[bin + (bin >> 5)]; };
    unsigned sl = 0;
#pragma unroll 8
    for (int b = 0; b < 32; ++b) sl += hbin(1023 - 32 * lt - b);
    unsigned incl = sl;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const unsigned y = __shfl_up_sync(0xffffffffu, incl, o);
      if (lt >= o) incl += y;
    }
    const int L = __ffs(__ballot_sync(0xffffffffu, incl >= 16u)) - 1;
    if (L >= 0) {
      // Walk chunk L in parallel: lane b holds bin 1023 - 32L - b (descending).
      const unsigned before = __shfl_sync(0xffffffffu, incl - sl, L);
      const int bin = 1023 - 32 * L - lt;
      const unsigned hv = hbin(bin);
      unsigned c2 = hv;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const unsigned y = __shfl_up_sync(0xffffffffu, c2, o);
        if (lt >= o) c2 += y;
      }
      const int bl = __ffs(__ballot_sync(0xffffffffu, before + c2 >= 16u)) - 1;
      if (lt == bl) {
        misc[t * 4 + 0] = bin;
        misc[t * 4 + 1] = before + c2;
        if (before + c2 > 32u) s_fast = 0;
      }
    }
    if (L < 0 && lt == 0) s_fast = 0;
    if (lt == 0) misc[t * 4 + 2] = 0;
  }
  __syncthreads();
  RT_CK(3)
  if (s_fast) {
    if (active) {
      const unsigned B = misc[t * 4 + 0];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int i = lt + 256 * j;
        if (i < kRtE && min(1023u, static_cast<unsigned>((v4[j] - mn) * inv)) >= B) {  // same bin math
          const int slot = atomicAdd(&misc[t * 4 + 2], 1u);
          s_ci[t][slot] = i;
          s_ck[t][slot] = k4[j];
        }
      }
    }
    __syncthreads();
    RT_CK(4)
    if (active && lt < 32) {
      const int nc = misc[t * 4 + 2];  // <= 32
      const int i = lt < nc ? s_ci[t][lt] : 0x7fffffff;
      const uint32_t k = lt < nc ? s_ck[t][lt] : 0u;
      int rank = 0;
      for (int d = 0; d < nc; ++d) {
        const uint32_t kd = __shfl_sync(0xffffffffu, k, d);
        const int id = __shfl_sync(0xffffffffu, i, d);
        rank += (kd > k) || (kd == k && id < i);
      }
      if (lt < nc && rank < 16) lst_i[t * 16 + rank] = i;
    }
  } else {
    // Exact radix select (8 bits per pass, MSB first) of the 16th largest key.
    uint32_t prefix = 0, pmask = 0, need = 16;
    for (int shift = 24; shift >= 0; shift -= 8) {
      if (active) hist[t * 256 + lt] = 0;
      __syncthreads();
      if (active) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const bool in = lt + 256 * j < kRtE && (k4[j] & pmask) == prefix;
          const unsigned bin = in ? ((k4[j] >> shift) & 255) : 256u;
          const unsigned peers = __match_any_sync(0xffffffffu, bin);
          if (in && (__ffs(peers) - 1) == lane) atomicAdd(&hist[t * 256 + bin], static_cast<unsigned>(__popc(peers)));
        }
      }
      __syncthreads();
      if (active && lt < 32) {
        const unsigned* hb = hist + t * 256;
        unsigned sl = 0;
#pragma unroll
        for (int b = 0; b < 8; ++b) sl += hb[255 - 8 * lt - b];
        unsigned incl = sl;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
          const unsigned y = __shfl_up_sync(0xffffffffu, incl, o);
          if (lt >= o) incl += y;
        }
        const int L = __ffs(__ballot_sync(0xffffffffu, incl >= need)) - 1;
        if (lt == L) {
          unsigned cum = incl - sl;
          for (int b = 0; b < 8; ++b) {
            const int bin = 255 - 8 * lt - b;
            if (cum + hb[bin] >= need) {
              misc[t * 4 + 0] = prefix | (static_cast<uint32_t>(bin) << shift);
              misc[t * 4 + 1] = need - cum;
              break;
            }
            cum += hb[bin];
          }
        }
      }
      __syncthreads();
      if (active) {
        prefix = misc[t * 4 + 0];
        need = misc[t * 4 + 1];
        pmask |= 0xffu << shift;
      }
    }
    // prefix = exact key of the 16th largest; `need` of the elements equal to it are taken.
    if (active && lt == 0) {
      misc[t * 4 + 2] = 0;
      misc[t * 4 + 3] = 0;
    }
    __syncthreads();
    if (active) {
      unsigned eq = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) eq += (lt + 256 * j < kRtE && k4[j] == prefix);
      if (eq) atomicAdd(&misc[t * 4 + 3], eq);
    }
    __syncthreads();
    if (active) {
      const unsigned need_eq = need, eq_total = misc[t * 4 + 3];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int i = lt + 256 * j;
        if (i >= kRtE) continue;
        bool take = k4[j] > prefix;
        if (k4[j] == prefix) {
          if (eq_total == need_eq) {
            take = true;
          } else {  // rare exact ties: lower expert ids first
            unsigned before = 0;
            for (int q = 0; q < i; ++q) before += keys[t * kRtE + q] == prefix;
            take = before < need_eq;
          }
        }
        if (take) lst_i[t * 16 + atomicAdd(&misc[t * 4 + 2], 1u)] = i;
      }
    }
  }
  __syncthreads();
  RT_CK(5)
  RT_MARK(5)
  if (active && lt < 32) {
    const int i = lt < 16 ? lst_i[t * 16 + lt] : 0;
    const uint32_t ki = lt < 16 ? keys[t * kRtE + i] : 0u;
    const float sc = lt < 16 ? scs[t * kRtE + i] : 0.f;
    int rank = 0;
    for (int q = 0; q < 16; ++q) {
      const int iq = __shfl_sync(0xffffffffu, i, q);
      const uint32_t kq = __shfl_sync(0xffffffffu, ki, q);
      rank += (kq > ki) || (kq == ki && iq < i);
    }
    float sum = sc;
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lt < 16) {
      const float w = (renorm ? sc / sum : sc) * scale;
      ids[t * 16 + rank] = i;
      wts[t * 16 + rank] = __float2bfloat16(w);
    }
  }
  RT_MARK(6)
#undef RT_MARK
#undef RT_CK
}

}  // namespace

// k3moe.route_shared: see the kernel comment. sbuf: persistent int32 [>= 2*896*2] Lamport score
// buffer, every word 0xFFFFFFFF initially (the kernel re-arms it). sh_w13 optional ([768, 7168]
// bf16: shared gate rows 0..383, up rows 384..767); h_shared [M, 384] bf16 is then written.
void route_shared(torch::Tensor x, torch::Tensor gate_w, torch::Tensor bias,
                  std::optional<torch::Tensor> sh_w13, torch::Tensor sbuf, torch::Tensor ids,
                  torch::Tensor wts, std::optional<torch::Tensor> h_shared, double beta, double linear_beta,
                  bool renormalize, double routed_scaling_factor, std::optional<torch::Tensor> trace) {
  const int M = x.size(0);
  TORCH_CHECK(M >= 1 && M <= kRtMaxM, "route_shared supports 1..4 tokens");
  TORCH_CHECK(x.dim() == 2 && x.size(1) == kRtH && x.scalar_type() == at::kBFloat16 && x.is_contiguous());
  TORCH_CHECK(gate_w.size(0) == kRtE && gate_w.size(1) == kRtH && gate_w.scalar_type() == at::kBFloat16 &&
              gate_w.is_contiguous());
  TORCH_CHECK(bias.numel() == kRtE && bias.scalar_type() == at::kFloat && bias.is_contiguous());
  TORCH_CHECK(sbuf.numel() * sbuf.element_size() >= (ids.numel() > 0 ? 1 : kRtReplicas) * M * kRtE * 8 &&
              sbuf.is_contiguous());
  // ids/wts empty -> scores-only mode: sbuf receives plain (score, score + bias) pairs [M][896]
  // for moe_block_lamport (no Lamport markers, no selection here).
  const bool select = ids.numel() > 0;
  TORCH_CHECK(!select || M <= kRtSelMaxM, "route_shared select mode supports 1..2 tokens");
  TORCH_CHECK(!select || (ids.numel() == M * 16 && ids.scalar_type() == at::kInt && ids.is_contiguous()));
  TORCH_CHECK(!select || (wts.numel() == M * 16 && wts.scalar_type() == at::kBFloat16 && wts.is_contiguous()));
  const __nv_bfloat16* w13p = nullptr;
  __nv_bfloat16* hp = nullptr;
  if (sh_w13) {
    TORCH_CHECK(h_shared.has_value());
    TORCH_CHECK(sh_w13->size(0) == 2 * kRtSh && sh_w13->size(1) == kRtH &&
                sh_w13->scalar_type() == at::kBFloat16 && sh_w13->is_contiguous());
    TORCH_CHECK(h_shared->numel() == M * 2 * kRtSh && h_shared->scalar_type() == at::kBFloat16 &&
                h_shared->is_contiguous(), "h_shared: shared gate_up output [M, 768] bf16");
    w13p = reinterpret_cast<const __nv_bfloat16*>(sh_w13->data_ptr());
    hp = reinterpret_cast<__nv_bfloat16*>(h_shared->data_ptr());
  }
  static int sms = 0;
  auto smem_for = [](int m, int sel_m) {
    return static_cast<size_t>(kRtRows * kRtRowStride + m * kRtH * 2 + (kRtWarps * 16 * m + 16 * m) * 4 +
                               sel_m * (kRtE * 8 + 256 * 4 + 4 * 4 + 16 * 4) + 16 + 16);
  };
  const size_t smem = smem_for(M, select ? M : 0);
  if (sms == 0) {
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, x.get_device());
    C10_CUDA_CHECK(cudaFuncSetAttribute(route_shared_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(std::max(smem_for(kRtMaxM, 0),
                                                                  smem_for(kRtSelMaxM, kRtSelMaxM)))));
  }
  const int G = std::min(sms, 152);
  TORCH_CHECK(G >= 152, "route_shared needs >= 152 SMs (<= 11 rows per CTA)");
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(G);
  cfg.blockDim = dim3(kRtThreads);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(
      &cfg, route_shared_kernel, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(gate_w.data_ptr()), bias.data_ptr<float>(), w13p,
      reinterpret_cast<uint2*>(sbuf.data_ptr()), ids.data_ptr<int>(),
      reinterpret_cast<__nv_bfloat16*>(wts.data_ptr()), hp, M, static_cast<float>(beta),
      static_cast<float>(linear_beta), renormalize ? 1 : 0, static_cast<float>(routed_scaling_factor),
      select ? 1 : 0, trace ? reinterpret_cast<unsigned long long*>(trace->data_ptr()) : nullptr));
}

namespace {
// ===========================================================================
// k3mk.tail: the MoE tail as one small kernel, K-split over thread-block clusters of 8.
// Semantics = vLLM KimiK3LatentMoETailOp at the same rounding points:
//   x   = bf16(sum_{src=0..15} fp32 partial_src)                          (routed all-reduce)
//   inv = rsqrt.approx(sum fp32(bf16(x*x)) / 3584 + eps); xn = bf16(x * inv * gamma)
//   sh  = bf16(sum_{src=0..15} fp32 shared_src[:, shard])                  (shared reduce-scatter)
//   out = bf16(fp32(bf16(xn . W_up_row)) + fp32(sh))  -> up-proj mailbox columns rank*448 + row
// Geometry (NC clusters x 8 CTAs; NC = 7 default, 14 optional): cluster c owns up-proj rows
// [c R, c R + R) (R = 448/NC); CTA q of a cluster owns the latent columns [448 q, 448 q + 448) and the
// output rows [c R + q R/8, + R/8) ("owner" rows).
//   entry    griddepcontrol.launch_dependents; one TMA of W_up[cluster rows][CTA columns] (R x 448 bf16,
//            57 KB at NC=7) and gamma; no griddepcontrol.wait (mailbox data = readiness; front mode excepted)
//   poll     thread (t, k), k < 56: fragment 56 q + k of token t from all 16 sources (every 32-bit word
//            checked), fixed-order fp32 sum -> bf16; k = 56 + w: the owner rows' reduce-scattered shared
//            word w (one reader: re-armed at once)
//   rms      per-CTA sum of squares -> the 8 CTAs of the cluster (st.async + mbarrier), fixed-order total,
//            xn of the CTA's own 448 columns only
//   gemv     partial up-proj: R rows x 448 columns (smem weights), fp32 partials -> the owner CTAs
//   owner    fixed-order sum of the 8 partials, bf16, + shared, bf16, multimem.st (R/8 rows per token)
//   re-arm   readers-done counter epoch[par][q] (+1 per cluster); the NC-th reader re-arms
//            latent columns [448 q, +448) of lat_mb[par] (off the critical path, after the publish)
// Only scalars cross DSMEM (no latent all-gather); every CTA does 1/8 of the RMSNorm and R x 448 MACs.
// ===========================================================================
constexpr int kT2Cl = 8;                         // CTAs per cluster
constexpr int kT2Threads = 512;
constexpr int kT2Warps = kT2Threads / 32;
constexpr int kT2MaxM = 8;
constexpr int kT2Cols = kHidden / kT2Cl;         // 448 latent columns per CTA
constexpr int kT2Frag = kT2Cols / 8;             // 56 fragments per token per CTA
static_assert(64 * kT2MaxM <= kT2Threads, "one 64-thread group per token");

struct T2Args {  // fields the prologue and the poll need first share one 64-byte constant-cache line
  int M, par, mmax, rank;
  unsigned long long* trace;    // [G][16] marks or nullptr
  __nv_bfloat16* lat_mb;        // local [3][16][Mmax][3584]
  __nv_bfloat16* rs_mb;         // local [3][16][Mmax][448]
  const __nv_bfloat16* gamma;   // [3584]
  const __nv_bfloat16* gemm2;   // front mode: [M*16][3584] unfinalized rows (nullptr: off)
  float eps;
  int mc;
  int defer_w;                  // 1: stage W_up after the RMS exchange instead of at CTA start (variant + 100)
  unsigned long long up_st;     // multicast (or plain) address of the up-proj mailbox [1][>=M][7168]
  unsigned long long* cnt;      // [3][8] readers-done counters (monotonic, zero-initialised once)
  const __nv_bfloat16* w_up_flat;  // flat mode: [448][3584] (the cluster mode uses a tensor map)
  const __nv_bfloat16* wts;     // front mode: [M][16] bf16 routing weights
  const __nv_bfloat16* sh;      // front mode: [M][7168] shared-expert partial
  unsigned long long lat_st;    // front mode: multicast (or plain) address of lat_mb
  unsigned long long rs_peer[kTp];  // front mode: rank d's rs_mb
};

__device__ __forceinline__ uint32_t mapa_u32(uint32_t a, uint32_t cta) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(cta));
  return r;
}
__device__ __forceinline__ void st_async4(uint32_t raddr, float v, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(raddr),
               "r"(__float_as_uint(v)), "r"(rbar)
               : "memory");
}
__device__ __forceinline__ void st_mb8(unsigned long long addr, uint32_t w0, uint32_t w1, int mc) {
  if (mc)
    asm volatile("multimem.st.relaxed.sys.global.v2.f32 [%0], {%1, %2};" ::"l"(addr), "r"(w0), "r"(w1) : "memory");
  else
    asm volatile("st.global.v2.u32 [%0], {%1, %2};" ::"l"(addr), "r"(w0), "r"(w1) : "memory");
}

// Poll one 16-byte (poll16x16) or 4-byte (poll16x4) word of each of the 16 sources until no 32-bit word is
// the Lamport sentinel. Every round re-issues ALL still-empty loads back-to-back: one round trip per round
// rather than one per late source (a per-source spin pays a round trip for each source that landed after
// the first load).
#ifdef K3_T2_POLL_GPU
#define POLL_LD16 ld_relaxed16
#else
#define POLL_LD16 ld_volatile16
#endif
__device__ __forceinline__ int poll16x16(const uint4* base, long sstride, uint4 (&v)[kTp]) {
#pragma unroll
  for (int s = 0; s < kTp; ++s) v[s] = POLL_LD16(base + s * sstride);
  uint32_t pend = 0;
  int rounds = 0;
#pragma unroll
  for (int s = 0; s < kTp; ++s) pend |= lamport_dirty(v[s]) ? (1u << s) : 0u;
  while (pend) {
    ++rounds;
    __nanosleep(kPollBackoffNs);
#pragma unroll
    for (int s = 0; s < kTp; ++s)
      if (pend & (1u << s)) v[s] = POLL_LD16(base + s * sstride);
    uint32_t np = 0;
#pragma unroll
    for (int s = 0; s < kTp; ++s) np |= ((pend >> s) & 1u) && lamport_dirty(v[s]) ? (1u << s) : 0u;
    pend = np;
  }
  return rounds;
}
__device__ __forceinline__ int poll16x4(const uint32_t* base, long sstride, uint32_t (&v)[kTp]) {
#pragma unroll
  for (int s = 0; s < kTp; ++s)
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v[s]) : "l"(base + s * sstride) : "memory");
  uint32_t pend = 0;
  int rounds = 0;
#pragma unroll
  for (int s = 0; s < kTp; ++s) pend |= v[s] == 0x80000000u ? (1u << s) : 0u;
  while (pend) {
    ++rounds;
    __nanosleep(kPollBackoffNs);
#pragma unroll
    for (int s = 0; s < kTp; ++s)
      if (pend & (1u << s))
        asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v[s]) : "l"(base + s * sstride) : "memory");
    uint32_t np = 0;
#pragma unroll
    for (int s = 0; s < kTp; ++s) np |= ((pend >> s) & 1u) && v[s] == 0x80000000u ? (1u << s) : 0u;
    pend = np;
  }
  return rounds;
}

// Sum N values (power of 2) across the warp at once, by value halving: step o = 16, 8, ... sends the half a
// lane does not keep. Each result is the same addition tree as warp_sum's butterfly (own + partner at every
// step), so it is bit-identical; lane L ends with value index `row` (built from its high lane bits).
template <int N>
__device__ __forceinline__ float halving_sum(float (&v)[N], int lane, int& row) {
  static_assert(N >= 1 && N <= 32 && (N & (N - 1)) == 0, "N power of 2");
  constexpr int kSteps = N == 1 ? 0 : N == 2 ? 1 : N == 4 ? 2 : N == 8 ? 3 : N == 16 ? 4 : 5;
  row = 0;
#pragma unroll
  for (int st = 0; st < kSteps; ++st) {
    const int o = 16 >> st, h = N >> (st + 1);
    const bool hi = (lane & o) != 0;
#pragma unroll
    for (int k = 0; k < h; ++k) {
      const float send = hi ? v[k] : v[k + h];
      const float keep = hi ? v[k + h] : v[k];
      v[k] = keep + __shfl_xor_sync(0xffffffffu, send, o);
    }
    if (hi) row += h;
  }
  float r = v[0];
#pragma unroll
  for (int o = 16 >> kSteps; o > 0; o >>= 1) r += __shfl_xor_sync(0xffffffffu, r, o);
  return r;
}

// Partial up-proj of kRw rows (warp) x 448 columns for MT tokens at once (W chunk read once for all tokens):
// part[(warp kRw + r) M + t]. Same per-accumulator FMA order (chunk lane, then lane + 28) and reduction tree
// as the rolled per-token loop.
template <int kRw, int MT>
__device__ __forceinline__ void t2_gemv_mt(const uint8_t* sm, const __nv_bfloat16* xs, float* part, int warp,
                                           int lane) {
  constexpr int kN = kRw * MT;
  constexpr int kNp = kN <= 1 ? 1 : kN <= 2 ? 2 : kN <= 4 ? 4 : kN <= 8 ? 8 : kN <= 16 ? 16 : 32;
  float acc[kNp];
#pragma unroll
  for (int i = 0; i < kNp; ++i) acc[i] = 0.f;
  if (lane < kT2Frag / 2) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int ch = lane + (kT2Frag / 2) * h;
      float xv[MT][8];
#pragma unroll
      for (int t = 0; t < MT; ++t) {
        const uint4 u = reinterpret_cast<const uint4*>(xs + t * kT2Cols)[ch];
        xv[t][0] = bf16_lo(u.x);
        xv[t][1] = bf16_hi(u.x);
        xv[t][2] = bf16_lo(u.y);
        xv[t][3] = bf16_hi(u.y);
        xv[t][4] = bf16_lo(u.z);
        xv[t][5] = bf16_hi(u.z);
        xv[t][6] = bf16_lo(u.w);
        xv[t][7] = bf16_hi(u.w);
      }
#pragma unroll
      for (int r = 0; r < kRw; ++r) {
        const uint4 wv = reinterpret_cast<const uint4*>(sm + (warp * kRw + r) * kT2Cols * 2)[ch];
        const float w0 = bf16_lo(wv.x), w1 = bf16_hi(wv.x), w2 = bf16_lo(wv.y), w3 = bf16_hi(wv.y);
        const float w4 = bf16_lo(wv.z), w5 = bf16_hi(wv.z), w6 = bf16_lo(wv.w), w7 = bf16_hi(wv.w);
#pragma unroll
        for (int t = 0; t < MT; ++t) {
          float s0 = acc[r * MT + t];
          s0 = fmaf(w0, xv[t][0], s0);
          s0 = fmaf(w1, xv[t][1], s0);
          s0 = fmaf(w2, xv[t][2], s0);
          s0 = fmaf(w3, xv[t][3], s0);
          s0 = fmaf(w4, xv[t][4], s0);
          s0 = fmaf(w5, xv[t][5], s0);
          s0 = fmaf(w6, xv[t][6], s0);
          s0 = fmaf(w7, xv[t][7], s0);
          acc[r * MT + t] = s0;
        }
      }
    }
  }
  int i = 0;
  const float v = halving_sum<kNp>(acc, lane, i);
  constexpr int kSteps = kNp == 1 ? 0 : kNp == 2 ? 1 : kNp == 4 ? 2 : kNp == 8 ? 3 : kNp == 16 ? 4 : 5;
  if ((lane & ((32 >> kSteps) - 1)) == 0 && i < kN) {
    const int r = i / MT, t = i - r * MT;
    part[(warp * kRw + r) * MT + t] = v;
  }
}

// kMinB = 2 (variant + 1000): <= 64 registers per thread, so a TAIL CTA holds half an SM's register file instead of
// all of it and concurrent kernels (side streams, PDL-early successors) can co-reside (costs ~1 us standalone).
template <int kNC, bool kMT = false, int kMinB = 1>  // kMT: M = 3 / 4 instantiation with the all-tokens GEMV
__global__ void __launch_bounds__(kT2Threads, kMinB)
tail2_kernel(const __grid_constant__ CUtensorMap tm_w, const __grid_constant__ T2Args a) {
  constexpr int kR = kUpShard / kNC;   // up-proj rows per cluster (64 or 32)
  constexpr int kRo = kR / kT2Cl;      // owner rows per CTA (8 or 4)
  constexpr int kRw = kR / kT2Warps;   // gemv rows per warp (4 or 2)
  static_assert(kR * kNC == kUpShard && kRo * kT2Cl == kR && kRw * kT2Warps == kR && kRo % 2 == 0, "geometry");
  extern __shared__ __align__(128) uint8_t sm[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int G = gridDim.x, gb = blockIdx.x, c = gb / kT2Cl, M = a.M;
  uint32_t q;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(q));
  // smem: W [R][448] | gamma [448] | xs [M][448] bf16 (xn) | part [R][M] f32 | pr [8 q][kRo][M] f32 |
  //       ssr [8 q][2 w][M] f32 |
  //       shs [M][kRo] f32 | outv [M][kRo] bf16 | bars [4] | flag | trace [16]
  int so = kR * kT2Cols * 2;
  const __nv_bfloat16* gs = reinterpret_cast<const __nv_bfloat16*>(sm + so);  // gamma [448 q, +448) (bulk copy)
  so += kT2Cols * 2;
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(sm + so);
  so += M * kT2Cols * 2;
  float* part = reinterpret_cast<float*>(sm + so);
  so += kR * M * 4;
  float* pr = reinterpret_cast<float*>(sm + so);
  so += kT2Cl * kRo * M * 4;
  float* ssr = reinterpret_cast<float*>(sm + so);  // [8 q][2 w][M]: per-warp sums of squares
  so += kT2Cl * 2 * M * 4;
  float* shs = reinterpret_cast<float*>(sm + so);
  so += M * kRo * 4;
  __nv_bfloat16* outv = reinterpret_cast<__nv_bfloat16*>(sm + so);
  so += M * kRo * 2;
  so = (so + 7) & ~7;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + so);  // [0] W TMA, [1] sums of squares, [2] partials, [3] gamma
  int* flag = reinterpret_cast<int*>(bars + 4);
  unsigned long long* trs = reinterpret_cast<unsigned long long*>(bars + 5);
  const uint32_t bw = smem_u32(bars), bs = smem_u32(bars + 1), bp = smem_u32(bars + 2), bg = smem_u32(bars + 3);
#ifdef K3_TRACE_CLOCK
#define T2_MARK(slot)                                                             \
  if (a.trace && tid == 0) trs[slot] = clock64();
#else
#define T2_MARK(slot)                                                             \
  if (a.trace && tid == 0) {                                                      \
    unsigned long long ts_;                                                       \
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts_));                      \
    trs[slot] = ts_;                                                              \
  }
#endif
#ifdef K3_T2_FINE
  if (a.trace && tid == 0) trs[10] = trs[13] = trs[14] = trs[15] = 0;
#endif
  T2_MARK(0)
  // Dedicated staging thread (tid 511: never a poller for M <= 8): a thread that issues bulk TMA
  // staging must not also poll (probe e7: detection +1-1.3 us).
  if (tid == kT2Threads - 1) {
    mbar_init(bw, 1);
    mbar_init(bs, 1);
    mbar_init(bp, 1);
    mbar_init(bg, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    // gamma slice first (small, needed at xn), then W. A plain __ldg of gamma by the pollers stalled them
    // ~640 cycles before their first poll load.
    mbar_arrive_expect_tx(bg, kT2Cols * 2);
    tma_1d(smem_u32(gs), a.gamma + kT2Cols * q, kT2Cols * 2, bg);
    mbar_arrive_expect_tx(bs, kT2Cl * 2 * M * 4);      // the 8 CTAs' per-warp sums of squares (2 warps/token)
    mbar_arrive_expect_tx(bp, kT2Cl * kRo * M * 4);    // the 8 CTAs' partials of our owner rows
#ifdef K3_T2_NOTMA  // timing experiment only: no W staging (the GEMV reads garbage)
    mbar_arrive_expect_tx(bw, 0);
    if (false)
#else
    mbar_arrive_expect_tx(bw, kR * kT2Cols * 2);
    if (!a.defer_w)
#endif
      asm volatile(
          "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], "
          "[%5];" ::"r"(smem_u32(sm)),
          "l"(reinterpret_cast<uint64_t>(&tm_w)), "r"(0), "r"(2 * static_cast<int>(q)), "r"(c * kR), "r"(bw)
          : "memory");
#ifdef K3_T2_FINE
    if (a.trace) {  // when the staging thread gets past the bulk-tensor issue
      unsigned long long ts_;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts_));
      trs[12] = ts_;
    }
#endif
  }
  const int pt = tid >> 6, pk = tid & 63;
  const bool lat_th = pt < M && pk < kT2Frag;
  // Reduce-scattered shared words: for M < 8 their readers sit in warps of their own (tid 64 M + j), so they
  // poll in parallel with the latent warps instead of after them (a warp runs divergent branches serially).
  int rs_t = -1, rs_w = 0;
  if (M < kT2MaxM) {
    const int j = tid - 64 * M;
    if (j >= 0 && j < M * (kRo / 2)) rs_t = j / (kRo / 2), rs_w = j % (kRo / 2);
  } else if (pt < M && pk >= kT2Frag && pk < kT2Frag + kRo / 2) {
    rs_t = pt, rs_w = pk - kT2Frag;
  }
  const bool rs_th = rs_t >= 0;
#ifdef K3_T2_PRO
  T2_MARK(9)
#endif
  static_assert(64 * (kT2MaxM - 1) + (kT2MaxM - 1) * 4 < kT2Threads - 1, "RS readers never on the staging thread");
  const int col8 = kT2Cols * static_cast<int>(q) + 8 * pk;  // first latent column of this thread's fragment
#ifdef K3_T2_PRO
  T2_MARK(10)
#endif
  asm volatile("griddepcontrol.launch_dependents;");
#ifdef K3_T2_PRO
  T2_MARK(11)
#endif
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");  // mbarriers initialised (fence above)
#ifdef K3_T2_PRO
  T2_MARK(12)
#endif
  const long lat_par = static_cast<long>(a.par) * kTp * a.mmax;  // [par] offset in units of [3584] rows
  const int orow = c * kR + static_cast<int>(q) * kRo;           // first owner row (of this rank's 448)
  T2_MARK(1)
  // ---------------------------------------------------------------- front (unfinalized MoE output)
  if (a.gemm2 != nullptr) {
    asm volatile("griddepcontrol.wait;" ::: "memory");  // gemm2 / wts / shared_out come from the predecessor
    const int nfin = (kHidden / 8 + G - 1) / G;  // fragments per token finalized by this CTA
    for (int i0 = 0; i0 < M * nfin * 8; i0 += kT2Threads) {  // warp-uniform loop (8 threads per fragment)
      const int i = i0 + tid;
      const int it = i >> 3, e = i & 7, t = it / nfin, f = gb + G * (it % nfin);
      const bool ok = i < M * nfin * 8 && f < kHidden / 8;
      float acc = 0.f;
      if (ok) {
        const __nv_bfloat16* src = a.gemm2 + static_cast<long>(t * kTopK) * kHidden + 8 * f + e;
        float vj[kTopK];
#pragma unroll
        for (int j = 0; j < kTopK; ++j) vj[j] = __bfloat162float(src[static_cast<long>(j) * kHidden]);
#pragma unroll
        for (int j = 0; j < kTopK; ++j) acc = fmaf(vj[j], __bfloat162float(a.wts[t * kTopK + j]), acc);
      }
      const __nv_bfloat16 b16 = __float2bfloat16(acc);
      uint32_t v = *reinterpret_cast<const uint16_t*>(&b16);
      v |= __shfl_down_sync(0xffffffffu, v, 1) << 16;
      const uint32_t w1 = __shfl_down_sync(0xffffffffu, v, 2);
      const uint32_t w2 = __shfl_down_sync(0xffffffffu, v, 4);
      const uint32_t w3 = __shfl_down_sync(0xffffffffu, v, 6);
      if (ok && e == 0)
        st_mb16(a.lat_st + static_cast<unsigned long long>(((lat_par + a.rank * a.mmax + t) * kHidden) + 8 * f) * 2,
                no_neg_zero4(make_uint4(v, w1, w2, w3)), a.mc);
    }
    const int nsd = (2 * kHidden / 8 + G - 1) / G;
    for (int i = tid; i < M * nsd; i += kT2Threads) {
      const int t = i / nsd, cf = gb + G * (i % nsd);
      if (cf >= 2 * kHidden / 8) continue;
      const int d = 8 * cf / kUpShard, col = 8 * cf - d * kUpShard;
      const uint4 v = no_neg_zero4(*reinterpret_cast<const uint4*>(a.sh + static_cast<long>(t) * 2 * kHidden + 8 * cf));
      st_mb16(a.rs_peer[d] +
                  static_cast<unsigned long long>(((static_cast<long>(a.par) * kTp + a.rank) * a.mmax + t) * kUpShard + col) * 2,
              v, 0);
    }
  }
  // ---------------------------------------------------------------- poll
  float s2 = 0.f;
  uint32_t xw[4] = {0u, 0u, 0u, 0u};
  const uint32_t* rs_base = nullptr;                         // RS reader: its word of source 0
  const long rs_ss = static_cast<long>(a.mmax) * kUpShard / 2;  // words between sources
  if (lat_th) {
    const uint4* base = reinterpret_cast<const uint4*>(a.lat_mb + (lat_par + pt) * kHidden + col8);
    const long sstride = static_cast<long>(a.mmax) * kHidden / 8;  // uint4 between sources
    uint4 v[kTp];
#ifdef K3_T2_PRO
    T2_MARK(13)
#endif
    const int rounds = poll16x16(base, sstride, v);
#ifdef K3_T2_PRO
    T2_MARK(14)
#endif
#ifdef K3_T2_FINE
    if (a.trace) {
      unsigned long long ts_;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts_));
      atomicMax(&trs[13], ts_);
      atomicMax(&trs[10], static_cast<unsigned long long>(rounds));
    }
    T2_MARK(9)
#endif
    (void)rounds;
    float acc[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = 0.f;
#pragma unroll
    for (int s = 0; s < kTp; ++s) {  // fixed source order 0..15
      acc[0] += bf16_lo(v[s].x);
      acc[1] += bf16_hi(v[s].x);
      acc[2] += bf16_lo(v[s].y);
      acc[3] += bf16_hi(v[s].y);
      acc[4] += bf16_lo(v[s].z);
      acc[5] += bf16_hi(v[s].z);
      acc[6] += bf16_lo(v[s].w);
      acc[7] += bf16_hi(v[s].w);
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const __nv_bfloat162 b2 = __floats2bfloat162_rn(acc[2 * k], acc[2 * k + 1]);
      xw[k] = *reinterpret_cast<const uint32_t*>(&b2);
      const __nv_bfloat162 sq2 = __hmul2(b2, b2);  // bf16 squares (upstream-compatible mode)
      s2 += __low2float(sq2);
      s2 += __high2float(sq2);
    }
  } else if (rs_th) {
    rs_base = reinterpret_cast<const uint32_t*>(a.rs_mb) +
              ((static_cast<long>(a.par) * kTp * a.mmax + rs_t) * kUpShard + orow + 2 * rs_w) / 2;
    uint32_t v[kTp];
    const int rrounds = poll16x4(rs_base, rs_ss, v);
#ifdef K3_T2_FINE
    if (a.trace) {
      unsigned long long ts_;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts_));
      atomicMax(&trs[14], ts_);
      atomicMax(&trs[15], static_cast<unsigned long long>(rrounds));
    }
#endif
    (void)rrounds;
    float lo = 0.f, hi = 0.f;
#pragma unroll
    for (int s = 0; s < kTp; ++s) {
      lo += bf16_lo(v[s]);
      hi += bf16_hi(v[s]);
    }
    shs[rs_t * kRo + 2 * rs_w] = __bfloat162float(__float2bfloat16(lo));
    shs[rs_t * kRo + 2 * rs_w + 1] = __bfloat162float(__float2bfloat16(hi));
  }
  T2_MARK(2)
  // ---------------------------------------------------------------- RMSNorm statistics over the cluster
  // Same reduction tree as vLLM's routed AllReduce/RMSNorm (TP16: 8-CTA clusters x 56 threads x 8 columns):
  // per-thread sequential sum of the 8 bf16 squares, per-warp butterfly (warp 1's lanes 24-31 hold 0, which
  // leaves lanes 0-7 bit-identical to vLLM's masked 24-lane butterfly), and, per peer CTA d in order,
  // full = (full + warp0_d) + warp1_d. Every warp pushes its sum straight to the 8 CTAs (lanes 0-7):
  // no CTA barrier between the poll and the exchange.
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) s2 += __shfl_xor_sync(0xffffffffu, s2, o);  // fixed-order butterfly
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");  // peers' mbarriers are initialised
  if ((warp >> 1) < M && lane < kT2Cl) {  // latent warps 2t, 2t + 1 of token t
    const uint32_t la = smem_u32(ssr + (static_cast<int>(q) * 2 + (warp & 1)) * M + (warp >> 1));
    st_async4(mapa_u32(la, lane), s2, mapa_u32(bs, lane));
  }
  mbar_wait(bs, 0);
  if (a.defer_w && tid == kT2Threads - 1)  // deferred staging: after this CTA's poll and the RMS exchange
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], "
        "[%5];" ::"r"(smem_u32(sm)),
        "l"(reinterpret_cast<uint64_t>(&tm_w)), "r"(0), "r"(2 * static_cast<int>(q)), "r"(c * kR), "r"(bw)
        : "memory");
  T2_MARK(3)
  if (lat_th) {  // xn of this CTA's own columns
    float tot = 0.f;
#pragma unroll
    for (int d = 0; d < kT2Cl; ++d) tot = (tot + ssr[(2 * d) * M + pt]) + ssr[(2 * d + 1) * M + pt];  // vLLM order
    float r;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(__fdiv_rn(tot, static_cast<float>(kHidden)) + a.eps));
    mbar_wait(bg, 0);  // landed long ago
    const uint4 gam = reinterpret_cast<const uint4*>(gs)[pk];
    const uint32_t gw[4] = {gam.x, gam.y, gam.z, gam.w};
    uint32_t ow[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const __nv_bfloat162 r2 = __floats2bfloat162_rn(bf16_lo(xw[k]) * r * bf16_lo(gw[k]),
                                                      bf16_hi(xw[k]) * r * bf16_hi(gw[k]));
      ow[k] = *reinterpret_cast<const uint32_t*>(&r2);
    }
    reinterpret_cast<uint4*>(xs + pt * kT2Cols)[pk] = make_uint4(ow[0], ow[1], ow[2], ow[3]);
  }
  mbar_wait(bw, 0);  // W landed (long ago)
  __syncthreads();
#ifdef K3_T2_DUMPXN  // debug: cluster 0 writes token 0's xn (and x) into the trace buffer
  if (a.trace && c == 0) {
    __nv_bfloat16* dump = reinterpret_cast<__nv_bfloat16*>(a.trace);
    for (int i = tid; i < kT2Cols; i += kT2Threads) dump[kT2Cols * q + i] = xs[i];
    if (lat_th && pt == 0) reinterpret_cast<uint4*>(dump + kHidden)[kT2Frag * q + pk] = make_uint4(xw[0], xw[1], xw[2], xw[3]);
  }
#endif
  T2_MARK(4)
  // ---------------------------------------------------------------- partial up-proj: warp = kRw rows
  // M = 3 / 4: all tokens per W chunk, one value-halving reduction (t2_gemv_mt). Otherwise: token loop
  // rolled with the runtime M (an unrolled kT2MaxM loop issues every predicated-off FFMA).
  // (a separate kernel instantiation, kMT, launched for M = 3 / 4 only: inlined into the M <= 2 kernel it cost
  // spills and ~0.1 us there)
  const bool mt = kMT && (M == 3 || M == 4);
  if constexpr (kMT) {
    if (M == 3) t2_gemv_mt<kRw, 3>(sm, xs, part, warp, lane);
    if (M == 4) t2_gemv_mt<kRw, 4>(sm, xs, part, warp, lane);
  }
#pragma unroll 1
  for (int t = 0; t < (mt ? 0 : M); ++t) {
    float acc[kRw];
#pragma unroll
    for (int r = 0; r < kRw; ++r) acc[r] = 0.f;
    if (lane < kT2Frag / 2) {  // lanes 0..27: 16-byte chunks lane and lane + 28 of the 448 columns
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int ch = lane + (kT2Frag / 2) * h;
        const uint4 xv = reinterpret_cast<const uint4*>(xs + t * kT2Cols)[ch];
        const float x0 = bf16_lo(xv.x), x1 = bf16_hi(xv.x), x2 = bf16_lo(xv.y), x3 = bf16_hi(xv.y);
        const float x4 = bf16_lo(xv.z), x5 = bf16_hi(xv.z), x6 = bf16_lo(xv.w), x7 = bf16_hi(xv.w);
#pragma unroll
        for (int r = 0; r < kRw; ++r) {
          const uint4 wv = reinterpret_cast<const uint4*>(sm + (warp * kRw + r) * kT2Cols * 2)[ch];
          float s0 = acc[r];
          s0 = fmaf(bf16_lo(wv.x), x0, s0);
          s0 = fmaf(bf16_hi(wv.x), x1, s0);
          s0 = fmaf(bf16_lo(wv.y), x2, s0);
          s0 = fmaf(bf16_hi(wv.y), x3, s0);
          s0 = fmaf(bf16_lo(wv.z), x4, s0);
          s0 = fmaf(bf16_hi(wv.z), x5, s0);
          s0 = fmaf(bf16_lo(wv.w), x6, s0);
          s0 = fmaf(bf16_hi(wv.w), x7, s0);
          acc[r] = s0;
        }
      }
    }
#pragma unroll
    for (int r = 0; r < kRw; ++r) {
      const float v = warp_sum(acc[r]);
      if (lane == 0) part[(warp * kRw + r) * M + t] = v;
    }
  }
  __syncthreads();
  // partials of owner rows -> their owner CTA: pr[our q][row][t] of CTA q' (rows q' kRo .. q' kRo + kRo), one
  // st.async per thread (a lane that issues several in a row serialises their issue latency)
  if (tid < kT2Cl * kRo * M) {
    const int d = tid / (kRo * M), rt = tid - d * (kRo * M);
    const uint32_t la = smem_u32(pr + static_cast<int>(q) * kRo * M + rt);
    st_async4(mapa_u32(la, d), part[d * kRo * M + rt], mapa_u32(bp, d));
  }
  T2_MARK(5)
  mbar_wait(bp, 0);
  T2_MARK(6)
  static_assert(kRo * kT2MaxM <= 64, "owner threads in warps 0-1");
  if (tid < kRo * M) {  // owner: fixed-order sum of the 8 partials, bf16, + shared, bf16
    const int r = tid / M, t = tid - r * M;
    float v = 0.f;
#pragma unroll
    for (int d = 0; d < kT2Cl; ++d) v += pr[(d * kRo + r) * M + t];
    const float gv = __bfloat162float(__float2bfloat16(v));
    outv[t * kRo + r] = __float2bfloat16(gv + shs[t * kRo + r]);
  }
  if (tid < 64) asm volatile("bar.sync 1, 64;" ::: "memory");  // warps 0-1 (owner rows and the publishers)
  if (tid < M) {  // publish kRo rows of token tid (16 or 8 bytes)
    const uint32_t* ow = reinterpret_cast<const uint32_t*>(outv + tid * kRo);
    const unsigned long long addr =
        a.up_st + static_cast<unsigned long long>(tid * (2 * kHidden) + a.rank * kUpShard + orow) * 2;
    if (kRo == 8)
      st_mb16(addr, no_neg_zero4(make_uint4(ow[0], ow[1], ow[2], ow[3])), a.mc);
    else
      st_mb8(addr, no_neg_zero(ow[0]), no_neg_zero(ow[1]), a.mc);
  }
  T2_MARK(7)
  if (rs_th) {  // one reader per RS word: re-arm it (after the publish, off the critical path)
#pragma unroll 1
    for (int s = 0; s < kTp; ++s)
      asm volatile("st.global.u32 [%0], %1;" ::"l"(rs_base + s * rs_ss), "r"(0x80000000u) : "memory");
  }
  // ---------------------------------------------------------------- readers-done re-arm (off the critical path)
  if (tid == 0) {
    unsigned long long old;
    asm volatile("atom.add.relaxed.gpu.global.u64 %0, [%1], 1;" : "=l"(old) : "l"(a.cnt + a.par * kT2Cl + q)
                 : "memory");
    *flag = (old % kNC) == kNC - 1;
  }
  __syncthreads();
  if (*flag) {  // the NC-th cluster to read columns [448 q, +448) of lat_mb[par] re-arms them
    for (int i = tid; i < kTp * M * kT2Frag; i += kT2Threads) {
      const int s = i / (M * kT2Frag), rr = i - s * (M * kT2Frag), t = rr / kT2Frag, k = rr - t * kT2Frag;
      st_sentinel16(a.lat_mb + ((lat_par + static_cast<long>(s) * a.mmax + t) * kHidden) + kT2Cols * q + 8 * k);
    }
  }
  // No CTA may exit while a cluster peer can still write into its smem: all pushes into this CTA have
  // landed (bs, bp completed); one cluster barrier covers our pushes into the peers.
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
  T2_MARK(8)
#ifndef K3_T2_DUMPXN
  if (a.trace && tid < 16) a.trace[gb * 16 + tid] = trs[tid];
#endif
#undef T2_MARK
}

// ---------------------------------------------------------------------------------------------
// "c4" TAIL (variant 4): 8 clusters x 4 CTAs = 32 CTAs at <= 111 KB smem. Sized for the 32 SMs that a
// 120-CTA MoE grid leaves free (probe RESULTS (d), 4.1): launched by the MoE's PDL trigger while the MoE
// still runs, it stages W_up and starts polling before the MoE ends, so only poll detection + RMS +
// GEMV + publish remain after the last MoE store. Cluster c owns up-proj rows [56 c, +56); CTA q owns
// latent columns [896 q, +896) and owner rows [56 c + 16 q, +16) (q = 3: +8; 16-byte aligned publish).
// Same rounding points and protocol as tail2_kernel; publish mode only; M <= 4.
// ---------------------------------------------------------------------------------------------
constexpr int kC4Cl = 4;
constexpr int kC4NC = 8;
constexpr int kC4R = kUpShard / kC4NC;   // 56 rows per cluster
constexpr int kC4Cols = kHidden / kC4Cl;  // 896 latent columns per CTA
constexpr int kC4Frag = kC4Cols / 8;      // 112 fragments per token per CTA
constexpr int kC4MaxM = 4;
constexpr int kC4RoMax = 16;
static_assert(128 * kC4MaxM <= kT2Threads && kC4Frag + kC4RoMax / 2 <= 127, "one 128-thread group per token");

__device__ __forceinline__ int c4_ro(int q) { return q < 3 ? 16 : 8; }

// Partial up-proj of the CTA: 56 rows x 896 columns x MT tokens, W read from smem once.
// Warp w: rows 7 (w & 7) .. +7, chunk half w >> 3 (16-byte chunks 56 (w >> 3) .. +56); lane < 28 does chunks
// lane and lane + 28 of that half. ph[half][row][t] (fp32).
template <int MT>
__device__ __forceinline__ void c4_gemv(const uint8_t* sm, const __nv_bfloat16* xs, float* ph, int warp, int lane) {
  const int rg = warp & 7, kh = warp >> 3;
  float acc[7][MT];
#pragma unroll
  for (int r = 0; r < 7; ++r)
#pragma unroll
    for (int t = 0; t < MT; ++t) acc[r][t] = 0.f;
  if (lane < 28) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int ch = 56 * kh + lane + 28 * h;
      float xv[MT][8];
#pragma unroll
      for (int t = 0; t < MT; ++t) {
        const uint4 u = reinterpret_cast<const uint4*>(xs + t * kC4Cols)[ch];
        xv[t][0] = bf16_lo(u.x);
        xv[t][1] = bf16_hi(u.x);
        xv[t][2] = bf16_lo(u.y);
        xv[t][3] = bf16_hi(u.y);
        xv[t][4] = bf16_lo(u.z);
        xv[t][5] = bf16_hi(u.z);
        xv[t][6] = bf16_lo(u.w);
        xv[t][7] = bf16_hi(u.w);
      }
#pragma unroll
      for (int r = 0; r < 7; ++r) {
        const uint4 wv = reinterpret_cast<const uint4*>(sm + (7 * rg + r) * kC4Cols * 2)[ch];
        const float w0 = bf16_lo(wv.x), w1 = bf16_hi(wv.x), w2 = bf16_lo(wv.y), w3 = bf16_hi(wv.y);
        const float w4 = bf16_lo(wv.z), w5 = bf16_hi(wv.z), w6 = bf16_lo(wv.w), w7 = bf16_hi(wv.w);
#pragma unroll
        for (int t = 0; t < MT; ++t) {
          float s0 = acc[r][t];
          s0 = fmaf(w0, xv[t][0], s0);
          s0 = fmaf(w1, xv[t][1], s0);
          s0 = fmaf(w2, xv[t][2], s0);
          s0 = fmaf(w3, xv[t][3], s0);
          s0 = fmaf(w4, xv[t][4], s0);
          s0 = fmaf(w5, xv[t][5], s0);
          s0 = fmaf(w6, xv[t][6], s0);
          s0 = fmaf(w7, xv[t][7], s0);
          acc[r][t] = s0;
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < 7; ++r)
#pragma unroll
    for (int t = 0; t < MT; ++t) {
      const float v = warp_sum(acc[r][t]);
      if (lane == 0) ph[(kh * kC4R + 7 * rg + r) * MT + t] = v;
    }
}

__global__ void __launch_bounds__(kT2Threads, 1)
tail_c4_kernel(const __grid_constant__ CUtensorMap tm_w, const __grid_constant__ T2Args a) {
  extern __shared__ __align__(128) uint8_t sm[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int gb = blockIdx.x, c = gb / kC4Cl, M = a.M;
  uint32_t q;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(q));
  const int ro = c4_ro(static_cast<int>(q));
  // smem: W [56][896] | xs [M][896] bf16 (xn) | ph [2][56][M] f32 | pr [4 q][16][M] f32 | ssr [4 q][M] f32 |
  //       wp [16] f32 | shs [M][16] f32 | outv [M][16] bf16 | bars [3] | flag | trace [16]
  int so = kC4R * kC4Cols * 2;
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(sm + so);
  so += M * kC4Cols * 2;
  float* ph = reinterpret_cast<float*>(sm + so);
  so += 2 * kC4R * M * 4;
  float* pr = reinterpret_cast<float*>(sm + so);
  so += kC4Cl * kC4RoMax * M * 4;
  float* ssr = reinterpret_cast<float*>(sm + so);
  so += kC4Cl * M * 4;
  float* wp = reinterpret_cast<float*>(sm + so);
  so += kT2Warps * 4;
  float* shs = reinterpret_cast<float*>(sm + so);
  so += M * kC4RoMax * 4;
  __nv_bfloat16* outv = reinterpret_cast<__nv_bfloat16*>(sm + so);
  so += M * kC4RoMax * 2;
  so = (so + 7) & ~7;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + so);  // [0] W TMA, [1] sums of squares, [2] partials
  int* flag = reinterpret_cast<int*>(bars + 3);
  unsigned long long* trs = reinterpret_cast<unsigned long long*>(bars + 4);
  const uint32_t bw = smem_u32(bars), bs = smem_u32(bars + 1), bp = smem_u32(bars + 2);
#ifdef K3_TRACE_CLOCK
#define C4_MARK(slot)                                                             \
  if (a.trace && tid == 0) trs[slot] = clock64();
#else
#define C4_MARK(slot)                                                             \
  if (a.trace && tid == 0) {                                                      \
    unsigned long long ts_;                                                       \
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts_));                      \
    trs[slot] = ts_;                                                              \
  }
#endif
  C4_MARK(0)
  if (tid == kT2Threads - 1) {  // dedicated staging thread (tid 511: never a poller)
    mbar_init(bw, 1);
    mbar_init(bs, 1);
    mbar_init(bp, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_arrive_expect_tx(bs, kC4Cl * M * 4);
    mbar_arrive_expect_tx(bp, kC4Cl * ro * M * 4);
    mbar_arrive_expect_tx(bw, kC4R * kC4Cols * 2);
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], "
        "[%5];" ::"r"(smem_u32(sm)),
        "l"(reinterpret_cast<uint64_t>(&tm_w)), "r"(0), "r"(4 * static_cast<int>(q)), "r"(c * kC4R), "r"(bw)
        : "memory");
  }
  const int pt = tid >> 7, pk = tid & 127;
  const bool lat_th = pt < M && pk < kC4Frag;
  const bool rs_th = pt < M && pk >= kC4Frag && pk < kC4Frag + ro / 2;
  const int col8 = kC4Cols * static_cast<int>(q) + 8 * pk;
  const uint4 gam = lat_th ? __ldg(reinterpret_cast<const uint4*>(a.gamma + col8)) : make_uint4(0u, 0u, 0u, 0u);
  asm volatile("griddepcontrol.launch_dependents;");
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  const long lat_par = static_cast<long>(a.par) * kTp * a.mmax;
  const int orow = c * kC4R + kC4RoMax * static_cast<int>(q);
  C4_MARK(1)
  // ---------------------------------------------------------------- poll
  float s2 = 0.f;
  uint32_t xw[4] = {0u, 0u, 0u, 0u};
  if (lat_th) {
    const uint4* base = reinterpret_cast<const uint4*>(a.lat_mb + (lat_par + pt) * kHidden + col8);
    const long sstride = static_cast<long>(a.mmax) * kHidden / 8;
    uint4 v[kTp];
    poll16x16(base, sstride, v);
    float acc[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = 0.f;
#pragma unroll
    for (int s = 0; s < kTp; ++s) {
      acc[0] += bf16_lo(v[s].x);
      acc[1] += bf16_hi(v[s].x);
      acc[2] += bf16_lo(v[s].y);
      acc[3] += bf16_hi(v[s].y);
      acc[4] += bf16_lo(v[s].z);
      acc[5] += bf16_hi(v[s].z);
      acc[6] += bf16_lo(v[s].w);
      acc[7] += bf16_hi(v[s].w);
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const __nv_bfloat162 b2 = __floats2bfloat162_rn(acc[2 * k], acc[2 * k + 1]);
      xw[k] = *reinterpret_cast<const uint32_t*>(&b2);
      const __nv_bfloat162 sq2 = __hmul2(b2, b2);
      s2 += __low2float(sq2);
      s2 += __high2float(sq2);
    }
  } else if (rs_th) {
    const int w = pk - kC4Frag;
    const uint32_t* base = reinterpret_cast<const uint32_t*>(a.rs_mb) +
                           ((static_cast<long>(a.par) * kTp * a.mmax + pt) * kUpShard + orow + 2 * w) / 2;
    const long sstride = static_cast<long>(a.mmax) * kUpShard / 2;
    uint32_t v[kTp];
    poll16x4(base, sstride, v);
    float lo = 0.f, hi = 0.f;
#pragma unroll
    for (int s = 0; s < kTp; ++s) {
      lo += bf16_lo(v[s]);
      hi += bf16_hi(v[s]);
    }
#pragma unroll 1
    for (int s = 0; s < kTp; ++s)
      asm volatile("st.global.u32 [%0], %1;" ::"l"(base + s * sstride), "r"(0x80000000u) : "memory");
    shs[pt * kC4RoMax + 2 * w] = __bfloat162float(__float2bfloat16(lo));
    shs[pt * kC4RoMax + 2 * w + 1] = __bfloat162float(__float2bfloat16(hi));
  }
  C4_MARK(2)
  // ---------------------------------------------------------------- RMSNorm statistics over the cluster
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) s2 += __shfl_xor_sync(0xffffffffu, s2, o);
  if (lane == 0) wp[warp] = s2;
  __syncthreads();
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
  if (tid < M) {  // token tid = warps 4 tid .. 4 tid + 3 (fixed order) -> all 4 CTAs
    const float v = ((wp[4 * tid] + wp[4 * tid + 1]) + wp[4 * tid + 2]) + wp[4 * tid + 3];
    const uint32_t la = smem_u32(ssr + static_cast<int>(q) * M + tid);
#pragma unroll
    for (int d = 0; d < kC4Cl; ++d) st_async4(mapa_u32(la, d), v, mapa_u32(bs, d));
  }
  mbar_wait(bs, 0);
  C4_MARK(3)
  if (lat_th) {
    float tot = 0.f;
#pragma unroll
    for (int d = 0; d < kC4Cl; ++d) tot += ssr[d * M + pt];
    float r;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(__fdiv_rn(tot, static_cast<float>(kHidden)) + a.eps));
    const uint32_t gw[4] = {gam.x, gam.y, gam.z, gam.w};
    uint32_t ow[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const __nv_bfloat162 r2 = __floats2bfloat162_rn(bf16_lo(xw[k]) * r * bf16_lo(gw[k]),
                                                      bf16_hi(xw[k]) * r * bf16_hi(gw[k]));
      ow[k] = *reinterpret_cast<const uint32_t*>(&r2);
    }
    reinterpret_cast<uint4*>(xs + pt * kC4Cols)[pk] = make_uint4(ow[0], ow[1], ow[2], ow[3]);
  }
  mbar_wait(bw, 0);
  __syncthreads();
  C4_MARK(4)
  // ---------------------------------------------------------------- partial up-proj
  switch (M) {
    case 1: c4_gemv<1>(sm, xs, ph, warp, lane); break;
    case 2: c4_gemv<2>(sm, xs, ph, warp, lane); break;
    case 3: c4_gemv<3>(sm, xs, ph, warp, lane); break;
    default: c4_gemv<4>(sm, xs, ph, warp, lane); break;
  }
  __syncthreads();
  C4_MARK(5)
  // partials (fixed order: half 0 + half 1) of row 16 d + r -> pr[our q][r][t] of CTA d
  if (tid < kC4R * M) {
    const int row = tid / M, t = tid - row * M, d = row >> 4, r = row & 15;
    const float v = ph[row * M + t] + ph[(kC4R + row) * M + t];
    const uint32_t la = smem_u32(pr + (static_cast<int>(q) * kC4RoMax + r) * M + t);
    st_async4(mapa_u32(la, d), v, mapa_u32(bp, d));
  }
  mbar_wait(bp, 0);
  C4_MARK(6)
  static_assert(kC4RoMax * kC4MaxM <= 64, "owner threads in warps 0-1");
  if (tid < ro * M) {  // owner: fixed-order sum of the 4 partials, bf16, + shared, bf16
    const int r = tid / M, t = tid - r * M;
    float v = 0.f;
#pragma unroll
    for (int d = 0; d < kC4Cl; ++d) v += pr[(d * kC4RoMax + r) * M + t];
    const float gv = __bfloat162float(__float2bfloat16(v));
    outv[t * kC4RoMax + r] = __float2bfloat16(gv + shs[t * kC4RoMax + r]);
  }
  if (tid < 64) asm volatile("bar.sync 1, 64;" ::: "memory");
  if (tid < M * (ro / 8)) {  // 16 bytes = 8 rows per store
    const int t = tid / (ro / 8), hh = tid - t * (ro / 8);
    const uint32_t* ow = reinterpret_cast<const uint32_t*>(outv + t * kC4RoMax + 8 * hh);
    const unsigned long long addr =
        a.up_st + static_cast<unsigned long long>(t * (2 * kHidden) + a.rank * kUpShard + orow + 8 * hh) * 2;
    st_mb16(addr, no_neg_zero4(make_uint4(ow[0], ow[1], ow[2], ow[3])), a.mc);
  }
  C4_MARK(7)
  // ---------------------------------------------------------------- readers-done re-arm (off the critical path)
  if (tid == 0) {
    unsigned long long old;
    asm volatile("atom.add.relaxed.gpu.global.u64 %0, [%1], 1;" : "=l"(old) : "l"(a.cnt + a.par * kT2Cl + q)
                 : "memory");
    *flag = (old % kC4NC) == kC4NC - 1;
  }
  __syncthreads();
  if (*flag) {
    for (int i = tid; i < kTp * M * kC4Frag; i += kT2Threads) {
      const int s = i / (M * kC4Frag), rr = i - s * (M * kC4Frag), t = rr / kC4Frag, k = rr - t * kC4Frag;
      st_sentinel16(a.lat_mb + ((lat_par + static_cast<long>(s) * a.mmax + t) * kHidden) + kC4Cols * q + 8 * k);
    }
  }
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
  C4_MARK(8)
  if (a.trace && tid < 16) a.trace[gb * 16 + tid] = trs[tid];
#undef C4_MARK
}

size_t tail_c4_smem(int M) {
  return kC4R * kC4Cols * 2 + M * kC4Cols * 2 + 2 * kC4R * M * 4 + kC4Cl * kC4RoMax * M * 4 + kC4Cl * M * 4 +
         kT2Warps * 4 + M * kC4RoMax * 6 + 64 + 256;
}

// ---------------------------------------------------------------------------------------------
// "Flat" TAIL (K3MK_TAIL_MODE=flat): 56 independent CTAs, no clusters, no cross-CTA exchange on the
// critical path. CTA g owns up-proj rows [8 g, 8 g + 8) (one 57 KB TMA). Every CTA polls the WHOLE
// latent (thread c < 448: 8-column chunk c of all 16 sources), sums in fixed order, computes the
// RMSNorm locally (block reduction), then 8 rows x 3584 on 16 warps (warp = row, K half), + the
// owner-row RS fragment, one 16-byte multimem.st per token. Readers-done: epoch[par] counts 56 CTAs;
// the last one re-arms all of lat_mb[par].
// ---------------------------------------------------------------------------------------------
constexpr int kTfCtas = 56;
constexpr int kTfRows = kUpShard / kTfCtas;   // 8
constexpr int kTfPollM = 4;                   // tokens polled per round (registers)

__global__ void __launch_bounds__(kT2Threads, 1) tail_flat_kernel(const __grid_constant__ T2Args a) {
  extern __shared__ __align__(128) uint8_t sm[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = blockIdx.x, M = a.M;
  // smem: W [8][3584] | xs [M][3584] (x -> xn) | wsum [16][M] | invs [M] | part [8 rows][2][M] | shs [M][8] |
  //       bars | flag | trace
  int so = kTfRows * kUpRowBytes;
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(sm + so);
  so += M * kUpRowBytes;
  float* wsum = reinterpret_cast<float*>(sm + so);
  so += kT2Warps * M * 4;
  float* invs = reinterpret_cast<float*>(sm + so);
  so += M * 4;
  float* part = reinterpret_cast<float*>(sm + so);
  so += kTfRows * 2 * M * 4;
  float* shs = reinterpret_cast<float*>(sm + so);
  so += M * kTfRows * 4;
  so = (so + 7) & ~7;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + so);
  int* flag = reinterpret_cast<int*>(bars + 1);
  unsigned long long* trs = reinterpret_cast<unsigned long long*>(bars + 2);
  const uint32_t bw = smem_u32(bars);
#ifdef K3_TRACE_CLOCK
#define TF_MARK(slot)                                                             \
  if (a.trace && tid == 0) trs[slot] = clock64();
#else
#define TF_MARK(slot)                                                             \
  if (a.trace && tid == 0) {                                                      \
    unsigned long long ts_;                                                       \
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts_));                      \
    trs[slot] = ts_;                                                              \
  }
#endif
  TF_MARK(0)
  if (tid == kT2Threads - 1) {  // dedicated staging thread (never a poller)
    mbar_init(bw, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_arrive_expect_tx(bw, kTfRows * kUpRowBytes);
    tma_1d(smem_u32(sm), a.w_up_flat + static_cast<long>(kTfRows * g) * kHidden, kTfRows * kUpRowBytes, bw);
  }
  const bool cth = tid < kHidden / 8;  // chunk thread
  const uint4 gam = cth ? __ldg(reinterpret_cast<const uint4*>(a.gamma) + tid) : make_uint4(0u, 0u, 0u, 0u);
  asm volatile("griddepcontrol.launch_dependents;");
  const long lat_par = static_cast<long>(a.par) * kTp * a.mmax;
  TF_MARK(1)
  // ---------------------------------------------------------------- front (unfinalized MoE output)
  if (a.gemm2 != nullptr) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    const int nfin = (kHidden / 8 + kTfCtas - 1) / kTfCtas;
    for (int i0 = 0; i0 < M * nfin * 8; i0 += kT2Threads) {
      const int i = i0 + tid;
      const int it = i >> 3, e = i & 7, t = it / nfin, f = g + kTfCtas * (it % nfin);
      const bool ok = i < M * nfin * 8 && f < kHidden / 8;
      float acc = 0.f;
      if (ok) {
        const __nv_bfloat16* src = a.gemm2 + static_cast<long>(t * kTopK) * kHidden + 8 * f + e;
        float vj[kTopK];
#pragma unroll
        for (int j = 0; j < kTopK; ++j) vj[j] = __bfloat162float(src[static_cast<long>(j) * kHidden]);
#pragma unroll
        for (int j = 0; j < kTopK; ++j) acc = fmaf(vj[j], __bfloat162float(a.wts[t * kTopK + j]), acc);
      }
      const __nv_bfloat16 b16 = __float2bfloat16(acc);
      uint32_t v = *reinterpret_cast<const uint16_t*>(&b16);
      v |= __shfl_down_sync(0xffffffffu, v, 1) << 16;
      const uint32_t w1 = __shfl_down_sync(0xffffffffu, v, 2);
      const uint32_t w2 = __shfl_down_sync(0xffffffffu, v, 4);
      const uint32_t w3 = __shfl_down_sync(0xffffffffu, v, 6);
      if (ok && e == 0)
        st_mb16(a.lat_st + static_cast<unsigned long long>(((lat_par + a.rank * a.mmax + t) * kHidden) + 8 * f) * 2,
                no_neg_zero4(make_uint4(v, w1, w2, w3)), a.mc);
    }
    const int nsd = (2 * kHidden / 8 + kTfCtas - 1) / kTfCtas;
    for (int i = tid; i < M * nsd; i += kT2Threads) {
      const int t = i / nsd, cf = g + kTfCtas * (i % nsd);
      if (cf >= 2 * kHidden / 8) continue;
      const int d = 8 * cf / kUpShard, col = 8 * cf - d * kUpShard;
      const uint4 v = no_neg_zero4(*reinterpret_cast<const uint4*>(a.sh + static_cast<long>(t) * 2 * kHidden + 8 * cf));
      st_mb16(a.rs_peer[d] +
                  static_cast<unsigned long long>(((static_cast<long>(a.par) * kTp + a.rank) * a.mmax + t) * kUpShard + col) * 2,
              v, 0);
    }
  }
  // ---------------------------------------------------------------- poll the whole latent + the RS fragment
  float s2[kTfPollM];
  uint32_t xw[kTfPollM][4];
#pragma unroll
  for (int k = 0; k < kTfPollM; ++k) s2[k] = 0.f;
#pragma unroll 1
  for (int t0 = 0; t0 < M; t0 += kTfPollM) {
    if (cth) {
      const long sstride = static_cast<long>(a.mmax) * kHidden / 8;
#pragma unroll
      for (int k = 0; k < kTfPollM; ++k) {
        const int t = t0 + k;
        if (t < M) {
          const uint4* base = reinterpret_cast<const uint4*>(a.lat_mb + (lat_par + t) * kHidden) + tid;
          uint4 v[kTp];
          poll16x16(base, sstride, v);
          float acc[8];
#pragma unroll
          for (int e = 0; e < 8; ++e) acc[e] = 0.f;
#pragma unroll
          for (int s = 0; s < kTp; ++s) {
            acc[0] += bf16_lo(v[s].x);
            acc[1] += bf16_hi(v[s].x);
            acc[2] += bf16_lo(v[s].y);
            acc[3] += bf16_hi(v[s].y);
            acc[4] += bf16_lo(v[s].z);
            acc[5] += bf16_hi(v[s].z);
            acc[6] += bf16_lo(v[s].w);
            acc[7] += bf16_hi(v[s].w);
          }
          float q2 = 0.f;
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) {
            const __nv_bfloat162 b2 = __floats2bfloat162_rn(acc[2 * kk], acc[2 * kk + 1]);
            xw[k][kk] = *reinterpret_cast<const uint32_t*>(&b2);
            const __nv_bfloat162 sq2 = __hmul2(b2, b2);
            q2 += __low2float(sq2);
            q2 += __high2float(sq2);
          }
          s2[k] = q2;
        }
      }
    } else if (tid - kHidden / 8 < M && t0 == 0) {  // RS: owner rows [8 g, 8 g + 8) of token t, 16 sources
      const int t = tid - kHidden / 8;
      const uint4* base = reinterpret_cast<const uint4*>(a.rs_mb + ((static_cast<long>(a.par) * kTp * a.mmax + t) *
                                                                    kUpShard) + kTfRows * g);
      const long sstride = static_cast<long>(a.mmax) * kUpShard / 8;
      uint4 v[kTp];
      poll16x16(base, sstride, v);
      float acc[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[e] = 0.f;
#pragma unroll
      for (int s = 0; s < kTp; ++s) {
        acc[0] += bf16_lo(v[s].x);
        acc[1] += bf16_hi(v[s].x);
        acc[2] += bf16_lo(v[s].y);
        acc[3] += bf16_hi(v[s].y);
        acc[4] += bf16_lo(v[s].z);
        acc[5] += bf16_hi(v[s].z);
        acc[6] += bf16_lo(v[s].w);
        acc[7] += bf16_hi(v[s].w);
      }
#pragma unroll 1
      for (int s = 0; s < kTp; ++s) st_sentinel16(base + s * sstride);  // one reader: re-arm now
#pragma unroll
      for (int e = 0; e < 8; ++e) shs[t * kTfRows + e] = __bfloat162float(__float2bfloat16(acc[e]));
    }
    // per-warp partial sums of squares (fixed-order butterflies)
#pragma unroll
    for (int k = 0; k < kTfPollM; ++k) {
      if (t0 + k < M) {
        float v = s2[k];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) wsum[warp * M + t0 + k] = v;
      }
    }
    if (M > kTfPollM && cth) {  // more tokens than the register set: park x in smem (xn pass reads it back)
#pragma unroll
      for (int k = 0; k < kTfPollM; ++k)
        if (t0 + k < M)
          reinterpret_cast<uint4*>(xs + (t0 + k) * kHidden)[tid] = make_uint4(xw[k][0], xw[k][1], xw[k][2], xw[k][3]);
    }
  }
  __syncthreads();
  TF_MARK(2)
  if (warp < M) {  // inv_rms of token `warp`: 14 warp partials, fixed-order butterfly over lanes 0..15
    float v = lane < kHidden / 8 / 32 ? wsum[lane * M + warp] : 0.f;
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    float r;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(__fdiv_rn(v, static_cast<float>(kHidden)) + a.eps));
    if (lane == 0) invs[warp] = r;
  }
  __syncthreads();
  TF_MARK(3)
  if (cth) {  // xn = bf16(x * inv * gamma), in place
    const uint32_t gw[4] = {gam.x, gam.y, gam.z, gam.w};
    if (M > kTfPollM) {  // x parked in smem
#pragma unroll 1
      for (int t = 0; t < M; ++t) {
        const uint4 xv = reinterpret_cast<const uint4*>(xs + t * kHidden)[tid];
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) xw[0][kk] = kk == 0 ? xv.x : kk == 1 ? xv.y : kk == 2 ? xv.z : xv.w;
        const float r = invs[t];
        uint32_t ow[4];
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          const __nv_bfloat162 r2 = __floats2bfloat162_rn(bf16_lo(xw[0][kk]) * r * bf16_lo(gw[kk]),
                                                          bf16_hi(xw[0][kk]) * r * bf16_hi(gw[kk]));
          ow[kk] = *reinterpret_cast<const uint32_t*>(&r2);
        }
        reinterpret_cast<uint4*>(xs + t * kHidden)[tid] = make_uint4(ow[0], ow[1], ow[2], ow[3]);
      }
    } else {
#pragma unroll
    for (int t = 0; t < kTfPollM; ++t) {
      if (t >= M) break;
      uint32_t x4[4];
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) x4[kk] = xw[t][kk];  // registers
      const float r = invs[t];
      uint32_t ow[4];
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        const __nv_bfloat162 r2 = __floats2bfloat162_rn(bf16_lo(x4[kk]) * r * bf16_lo(gw[kk]),
                                                        bf16_hi(x4[kk]) * r * bf16_hi(gw[kk]));
        ow[kk] = *reinterpret_cast<const uint32_t*>(&r2);
      }
      reinterpret_cast<uint4*>(xs + t * kHidden)[tid] = make_uint4(ow[0], ow[1], ow[2], ow[3]);
    }
    }
  }
  mbar_wait(bw, 0);  // W landed (long ago)
  __syncthreads();
  TF_MARK(4)
  // ---------------------------------------------------------------- GEMV: warp = (row, K half)
  {
    const int row = warp >> 1, hk = warp & 1;
    const uint4* wr = reinterpret_cast<const uint4*>(sm + row * kUpRowBytes) + hk * (kHidden / 16);
#pragma unroll 1
    for (int t = 0; t < M; ++t) {
      const uint4* xr = reinterpret_cast<const uint4*>(xs + t * kHidden) + hk * (kHidden / 16);
      float a0 = 0.f, a1 = 0.f;
#pragma unroll
      for (int z = 0; z < kHidden / 16 / 32; ++z) {  // 7
        const uint4 wv = wr[lane + 32 * z], xv = xr[lane + 32 * z];
        a0 = fmaf(bf16_lo(wv.x), bf16_lo(xv.x), a0);
        a1 = fmaf(bf16_hi(wv.x), bf16_hi(xv.x), a1);
        a0 = fmaf(bf16_lo(wv.y), bf16_lo(xv.y), a0);
        a1 = fmaf(bf16_hi(wv.y), bf16_hi(xv.y), a1);
        a0 = fmaf(bf16_lo(wv.z), bf16_lo(xv.z), a0);
        a1 = fmaf(bf16_hi(wv.z), bf16_hi(xv.z), a1);
        a0 = fmaf(bf16_lo(wv.w), bf16_lo(xv.w), a0);
        a1 = fmaf(bf16_hi(wv.w), bf16_hi(xv.w), a1);
      }
      const float v = warp_sum(a0 + a1);
      if (lane == 0) part[(row * 2 + hk) * M + t] = v;
    }
  }
  __syncthreads();
  TF_MARK(5)
  if (warp == 0) {  // lanes = (row, token pair): bf16(bf16(dot) + shared) -> 16 bytes per token
    for (int t = 0; t < M; ++t) {
      uint32_t word = 0;
      if (lane < kTfRows) {
        const float v = part[(lane * 2) * M + t] + part[(lane * 2 + 1) * M + t];  // fixed order
        const float gv = __bfloat162float(__float2bfloat16(v));
        const __nv_bfloat16 o = __float2bfloat16(gv + shs[t * kTfRows + lane]);
        word = *reinterpret_cast<const uint16_t*>(&o);
      }
      word |= __shfl_down_sync(0xffffffffu, word, 1) << 16;  // even lanes: rows (lane, lane + 1)
      const uint32_t w1 = __shfl_sync(0xffffffffu, word, 2), w2 = __shfl_sync(0xffffffffu, word, 4),
                     w3 = __shfl_sync(0xffffffffu, word, 6);
      if (lane == 0)
        st_mb16(a.up_st + static_cast<unsigned long long>(t * (2 * kHidden) + a.rank * kUpShard + kTfRows * g) * 2,
                no_neg_zero4(make_uint4(word, w1, w2, w3)), a.mc);
    }
  }
  TF_MARK(6)
  // ---------------------------------------------------------------- readers-done re-arm (off the critical path)
  if (tid == 0) {
    unsigned long long old;
    asm volatile("atom.add.relaxed.gpu.global.u64 %0, [%1], 1;" : "=l"(old) : "l"(a.cnt + a.par * kT2Cl) : "memory");
    *flag = (old % kTfCtas) == kTfCtas - 1;
  }
  __syncthreads();
  if (*flag)
    for (int i = tid; i < kTp * M * (kHidden / 8); i += kT2Threads) {
      const int s = i / (M * (kHidden / 8)), rr = i - s * (M * (kHidden / 8)), t = rr / (kHidden / 8),
                f = rr - t * (kHidden / 8);
      st_sentinel16(a.lat_mb + ((lat_par + static_cast<long>(s) * a.mmax + t) * kHidden) + 8 * f);
    }
  TF_MARK(7)
  if (a.trace && tid < 16) a.trace[g * 16 + tid] = trs[tid];
#undef TF_MARK
}

size_t tail_flat_smem(int M) {
  return kTfRows * kUpRowBytes + M * kUpRowBytes + kT2Warps * M * 4 + M * 4 + kTfRows * 2 * M * 4 + M * kTfRows * 4 +
         64 + 256;
}

// bf16 [448, 3584] up_proj shard as [448 rows][16 halves][224 cols] (2-byte elements): box {224, 2, R}
CUtensorMap make_wup_map(const void* base, int rows_box, int chunks = 2) {
  CUtensorMap m;
  const cuuint64_t dims[3] = {224, 16, static_cast<cuuint64_t>(kUpShard)};
  const cuuint64_t strides[2] = {448, static_cast<cuuint64_t>(kUpRowBytes)};
  const cuuint32_t box[3] = {224, static_cast<cuuint32_t>(chunks), static_cast<cuuint32_t>(rows_box)};
  const cuuint32_t es[3] = {1, 1, 1};
  const CUresult r = tensor_map_encoder()(&m, CU_TENSOR_MAP_DATA_TYPE_UINT16, 3, const_cast<void*>(base), dims, strides,
                                          box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                          CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (w_up) failed: ", static_cast<int>(r));
  return m;
}

template <int kNC>
size_t tail2_smem(int M) {
  constexpr int kR = kUpShard / kNC, kRo = kR / kT2Cl;
  return kR * kT2Cols * 2 + kT2Cols * 2 + M * kT2Cols * 2 + kR * M * 4 + kT2Cl * kRo * M * 4 + kT2Cl * 2 * M * 4 +
         M * kRo * 6 + 8 + 5 * 8 + 16 * 8 + 64;
}
}  // namespace

// k3mk.tail: see tail2_kernel. lat_mb [3, 16, Mmax, 3584] / rs_mb [3, 16, Mmax, 448] bf16 symmetric
// mailboxes (all words 0x80000000 initially), par in {0,1,2} (the buffer the MoE published into),
// w_up [448, 3584] = up_proj.weight[rank*448 : rank*448+448], gamma [3584], up_mb / up_mc_ptr: vLLM's
// up-proj mailbox [1, >=M, 7168] and its multicast address (or its own address with multicast=false),
// epoch: int64 [>= 24] zero-initialised once (readers-done counters; one tensor per mailbox set AND per
// variant -- the counters count modulo the variant's cluster count, so never switch variants on one epoch).
// Front mode (gemm2 given): the unfinalized MoE output (e.g. moe8) is finalized and published here.
// variant (0 = env K3MK_TAIL_VARIANT, default 14):
//   7 / 14  NC clusters x 8 CTAs (tail2_kernel), 56 / 112 CTAs, 57 / 29 KB of W_up per CTA
//   4       8 clusters x 4 CTAs (tail_c4_kernel), 32 CTAs <= 111 KB: pair with moe_block_lamport(grid=120) so
//           the TAIL is resident (stages W_up, polls) while the MoE runs. Publish mode only, M <= 4.
//   1       flat, 56 CTAs, no clusters (tail_flat_kernel)
void tail(torch::Tensor lat_mb, torch::Tensor rs_mb, int64_t par, torch::Tensor w_up, torch::Tensor gamma, double eps,
          torch::Tensor up_mb, int64_t up_mc_ptr, torch::Tensor epoch, int64_t rank, int64_t M,
          std::optional<torch::Tensor> trace, std::optional<torch::Tensor> gemm2, std::optional<torch::Tensor> wts,
          std::optional<torch::Tensor> shared_out, int64_t lat_mc_ptr, std::optional<std::vector<int64_t>> rs_peers,
          bool multicast, int64_t variant) {
  TORCH_CHECK(M >= 1 && M <= kT2MaxM, "k3mk.tail supports 1..8 tokens");
  TORCH_CHECK(par >= 0 && par < 3 && rank >= 0 && rank < kTp);
  auto bf = [](const torch::Tensor& t) { return t.scalar_type() == at::kBFloat16 && t.is_contiguous() && t.is_cuda(); };
  TORCH_CHECK(bf(lat_mb) && lat_mb.dim() == 4 && lat_mb.size(0) == 3 && lat_mb.size(1) == kTp && lat_mb.size(3) == kHidden,
              "lat_mb: bf16 [3, 16, Mmax, 3584]");
  const int mmax = lat_mb.size(2);
  TORCH_CHECK(M <= mmax);
  TORCH_CHECK(bf(rs_mb) && rs_mb.dim() == 4 && rs_mb.size(0) == 3 && rs_mb.size(1) == kTp && rs_mb.size(2) == mmax &&
              rs_mb.size(3) == kUpShard, "rs_mb: bf16 [3, 16, Mmax, 448]");
  TORCH_CHECK(bf(w_up) && w_up.size(0) == kUpShard && w_up.size(1) == kHidden &&
              reinterpret_cast<uintptr_t>(w_up.data_ptr()) % 16 == 0, "w_up: bf16 [448, 3584]");
  TORCH_CHECK(bf(gamma) && gamma.numel() == kHidden && reinterpret_cast<uintptr_t>(gamma.data_ptr()) % 16 == 0);
  TORCH_CHECK(bf(up_mb) && up_mb.size(-1) == 2 * kHidden && up_mb.numel() >= M * 2 * kHidden);
  TORCH_CHECK(up_mc_ptr != 0 && up_mc_ptr % 16 == 0);
  TORCH_CHECK(epoch.scalar_type() == at::kLong && epoch.is_cuda() && epoch.numel() >= 3 * kT2Cl);
  // variant: 0 = default (env K3MK_TAIL_VARIANT, else 14 clusters), 7 / 14 = clusters x 8 CTAs, 4 = 8 x 4, 1 = flat
  static const int env_variant = getenv("K3MK_TAIL_VARIANT") ? atoi(getenv("K3MK_TAIL_VARIANT")) : 14;
  int v = variant ? static_cast<int>(variant) : env_variant;
  const bool lowreg = v >= 1000;  // variant + 1000: the <= 64-register instantiation (kMinB = 2)
  v %= 1000;
  const bool defer_w = v >= 100;  // variant + 100: 8-CTA kernels stage W_up after the RMS exchange
  v %= 100;
  TORCH_CHECK(v == 7 || v == 14 || v == 1 || v == 4,
              "k3mk.tail variant must be 7, 14 (8-CTA clusters), 4 (8 x 4-CTA clusters) or 1 (flat)");
  TORCH_CHECK(v != 4 || (M <= kC4MaxM && !gemm2), "k3mk.tail variant 4: publish mode, M <= 4");
  const bool flat = v == 1;
  const int nc = flat ? 7 : v;
  T2Args a{};
  a.lat_mb = reinterpret_cast<__nv_bfloat16*>(lat_mb.data_ptr());
  a.rs_mb = reinterpret_cast<__nv_bfloat16*>(rs_mb.data_ptr());
  a.gamma = reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr());
  a.up_st = static_cast<unsigned long long>(up_mc_ptr);
  a.cnt = reinterpret_cast<unsigned long long*>(epoch.data_ptr());
  a.trace = trace ? reinterpret_cast<unsigned long long*>(trace->data_ptr()) : nullptr;
  if (gemm2) {
    TORCH_CHECK(bf(*gemm2) && gemm2->dim() == 2 && gemm2->size(0) == M * kTopK && gemm2->size(1) == kHidden);
    TORCH_CHECK(wts && bf(*wts) && wts->numel() == M * kTopK, "front mode: wts [M, 16] bf16");
    TORCH_CHECK(shared_out && bf(*shared_out) && shared_out->numel() == M * 2 * kHidden, "front mode: shared_out");
    TORCH_CHECK(lat_mc_ptr != 0 && lat_mc_ptr % 16 == 0 && rs_peers && static_cast<int>(rs_peers->size()) == kTp);
    a.gemm2 = reinterpret_cast<const __nv_bfloat16*>(gemm2->data_ptr());
    a.wts = reinterpret_cast<const __nv_bfloat16*>(wts->data_ptr());
    a.sh = reinterpret_cast<const __nv_bfloat16*>(shared_out->data_ptr());
    a.lat_st = static_cast<unsigned long long>(lat_mc_ptr);
    for (int d = 0; d < kTp; ++d) {
      TORCH_CHECK((*rs_peers)[d] != 0 && (*rs_peers)[d] % 16 == 0);
      a.rs_peer[d] = static_cast<unsigned long long>((*rs_peers)[d]);
    }
  }
  a.eps = static_cast<float>(eps);
  a.par = static_cast<int>(par);
  a.mmax = mmax;
  a.rank = static_cast<int>(rank);
  a.M = static_cast<int>(M);
  a.mc = multicast ? 1 : 0;
  a.defer_w = defer_w ? 1 : 0;
  a.w_up_flat = reinterpret_cast<const __nv_bfloat16*>(w_up.data_ptr());
  if (flat) {
    static bool finit = false;
    if (!finit) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(tail_flat_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          static_cast<int>(tail_flat_smem(kT2MaxM))));
      finit = true;
    }
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(kTfCtas);
    cfg.blockDim = dim3(kT2Threads);
    cfg.dynamicSmemBytes = tail_flat_smem(M);
    cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, tail_flat_kernel, a));
    return;
  }
  // tensor map of the up_proj shard, cached per (pointer, rows per cluster); captured by value in graphs
  static std::vector<std::pair<std::pair<const void*, int>, CUtensorMap>> maps;
  const int rows = v == 4 ? kC4R : kUpShard / nc;
  const CUtensorMap* tm = nullptr;
  for (auto& e : maps)
    if (e.first.first == w_up.data_ptr() && e.first.second == rows) tm = &e.second;
  if (!tm) {
    maps.push_back({{w_up.data_ptr(), rows}, make_wup_map(w_up.data_ptr(), rows, v == 4 ? 4 : 2)});
    tm = &maps.back().second;
  }
  static bool init = false;
  if (!init) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(tail2_kernel<7>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(tail2_smem<7>(kT2MaxM))));
    C10_CUDA_CHECK(cudaFuncSetAttribute(tail2_kernel<14>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(tail2_smem<14>(kT2MaxM))));
    C10_CUDA_CHECK(cudaFuncSetAttribute(tail2_kernel<14, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(tail2_smem<14>(kT2MaxM))));
    C10_CUDA_CHECK(cudaFuncSetAttribute(tail2_kernel<7, false, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(tail2_smem<7>(kT2MaxM))));
    C10_CUDA_CHECK(cudaFuncSetAttribute(tail2_kernel<14, false, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(tail2_smem<14>(kT2MaxM))));
    C10_CUDA_CHECK(cudaFuncSetAttribute(tail2_kernel<14, true, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(tail2_smem<14>(kT2MaxM))));
    C10_CUDA_CHECK(cudaFuncSetAttribute(tail_c4_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(tail_c4_smem(kC4MaxM))));
    init = true;
  }
  if (v == 4) {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(kC4Cl * kC4NC);
    cfg.blockDim = dim3(kT2Threads);
    cfg.dynamicSmemBytes = tail_c4_smem(M);
    cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[2];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    attr[1].id = cudaLaunchAttributeClusterDimension;
    attr[1].val.clusterDim.x = kC4Cl;
    attr[1].val.clusterDim.y = 1;
    attr[1].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 2;
    const CUtensorMap tmv = *tm;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, tail_c4_kernel, tmv, a));
    return;
  }
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(kT2Cl * nc);
  cfg.blockDim = dim3(kT2Threads);
  cfg.dynamicSmemBytes = nc == 7 ? tail2_smem<7>(M) : tail2_smem<14>(M);
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  attr[1].id = cudaLaunchAttributeClusterDimension;
  attr[1].val.clusterDim.x = kT2Cl;
  attr[1].val.clusterDim.y = 1;
  attr[1].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 2;
  const CUtensorMap tmv = *tm;
  if (lowreg) {
    if (nc == 7)
      C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, tail2_kernel<7, false, 2>, tmv, a));
    else if (M == 3 || M == 4)
      C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, tail2_kernel<14, true, 2>, tmv, a));
    else
      C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, tail2_kernel<14, false, 2>, tmv, a));
  } else if (nc == 7)
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, tail2_kernel<7>, tmv, a));
  else if (M == 3 || M == 4)
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, tail2_kernel<14, true>, tmv, a));
  else
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, tail2_kernel<14>, tmv, a));
}

// out[t, n] = sum_j topk_weights[t, j] * (W2_e[n, :192] . h[t, j]); finalized bf16 [M, 3584].
void moe_fused(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_weights,
               torch::Tensor w13, torch::Tensor w13_scale, torch::Tensor w2,
               torch::Tensor w2_scale, torch::Tensor workspace, torch::Tensor barrier,
               torch::Tensor out, double beta, double linear_beta,
               std::optional<torch::Tensor> trace) {
  const int M = x.size(0);
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat && topk_weights.is_contiguous() &&
              topk_weights.numel() == M * kTopK);
  TORCH_CHECK(out.size(0) == M && out.size(1) == kHidden && out.scalar_type() == at::kBFloat16 &&
              out.is_contiguous());
  launch_fused<true>(x, topk_ids, topk_weights.data_ptr<float>(), w13, w13_scale, w2, w2_scale, workspace,
                     barrier, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), beta, linear_beta, trace);
}

// gemm2_out[t*16 + j, n] = W2_{e_tj}[n, :192] . h[t, j] (unweighted, not summed over j).
void moe_fused_unfinalized(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor w13,
                           torch::Tensor w13_scale, torch::Tensor w2, torch::Tensor w2_scale,
                           torch::Tensor workspace, torch::Tensor barrier, torch::Tensor gemm2_out,
                           double beta, double linear_beta, std::optional<torch::Tensor> trace) {
  const int M = x.size(0);
  TORCH_CHECK(gemm2_out.size(0) == M * kTopK && gemm2_out.size(1) == kHidden &&
              gemm2_out.scalar_type() == at::kBFloat16 && gemm2_out.is_contiguous());
  launch_fused<false>(x, topk_ids, nullptr, w13, w13_scale, w2, w2_scale, workspace, barrier,
                      reinterpret_cast<__nv_bfloat16*>(gemm2_out.data_ptr()), beta, linear_beta, trace);
}

// Same as moe_fused_unfinalized, but x comes straight out of the latent down-projection's
// Lamport mailbox (local view of the NVLS-multicast symmetric buffer, >= M rows of 3584 bf16,
// empty word = 0x80000000). Replaces LamportCopyKernel + moe_fused_unfinalized: rows are polled
// per CTA and re-armed (rows [0, M) only) after every CTA has read them. M = topk_ids.size(0).
void moe_fused_unfinalized_lamport(torch::Tensor mailbox, torch::Tensor topk_ids, torch::Tensor w13,
                                   torch::Tensor w13_scale, torch::Tensor w2, torch::Tensor w2_scale,
                                   torch::Tensor workspace, torch::Tensor barrier, torch::Tensor gemm2_out,
                                   double beta, double linear_beta, bool wait_prior,
                                   std::optional<torch::Tensor> trace) {
  const int M = topk_ids.size(0);
  TORCH_CHECK(gemm2_out.size(0) == M * kTopK && gemm2_out.size(1) == kHidden &&
              gemm2_out.scalar_type() == at::kBFloat16 && gemm2_out.is_contiguous());
  TORCH_CHECK(reinterpret_cast<uintptr_t>(mailbox.data_ptr()) % 16 == 0);
  launch_fused<false, true>(mailbox, topk_ids, nullptr, w13, w13_scale, w2, w2_scale, workspace, barrier,
                            reinterpret_cast<__nv_bfloat16*>(gemm2_out.data_ptr()), beta, linear_beta, trace,
                            wait_prior ? kFlagWaitPrior : 0);
}

// Whole routed MoE block for M <= 2 (after route_shared): top-k from route_shared's scores
// (in-kernel, every CTA), latent x polled from the down-shard Lamport mailbox (re-armed), FC1 +
// SiTU + FC2 unfinalized rows [M*16, 3584], plus the shared expert's down_proj partial
// shared_out [M, 7168] = h_shared @ w_sd^T. ids_out / wts_out ([M, 16] int32 / bf16) are written
// for the MoE tail. grid <= 0: all SMs; otherwise at most `grid` CTAs (barrier slot per grid!).
void moe_block_lamport(torch::Tensor mailbox, torch::Tensor scores, torch::Tensor w13, torch::Tensor w13_scale,
                       torch::Tensor w2, torch::Tensor w2_scale, torch::Tensor workspace, torch::Tensor barrier,
                       torch::Tensor gemm2_out, torch::Tensor ids_out, torch::Tensor wts_out,
                       torch::Tensor h_shared, torch::Tensor w_sd, torch::Tensor shared_out, double beta,
                       double linear_beta, bool renormalize, double routed_scaling_factor, int64_t grid,
                       double shared_beta, double shared_linear_beta, std::optional<torch::Tensor> trace,
                       std::optional<torch::Tensor> latent_out, std::optional<torch::Tensor> lat_mb,
                       int64_t lat_mc_ptr, std::optional<std::vector<int64_t>> rs_peers, int64_t par, int64_t rank,
                       bool multicast) {
  const int M = ids_out.size(0);
  TORCH_CHECK(M >= 1 && M <= kBlockMaxM, "moe_block_lamport supports 1..4 tokens");
  if (lat_mb) {
    // PUBLISH mode: finalized partial latent -> lat_mb[par][rank] (multicast), shared-expert down rows ->
    // rs_mb[par][rank] of the owning rank; no gemm2_out / shared_out. Consumed by k3mk.tail.
    TORCH_CHECK(!latent_out, "publish mode excludes latent_out");
    TORCH_CHECK(M <= kPubMaxM);
    TORCH_CHECK(lat_mb->scalar_type() == at::kBFloat16 && lat_mb->is_contiguous() && lat_mb->dim() == 4 &&
                    lat_mb->size(0) == 3 && lat_mb->size(1) == kTp && lat_mb->size(2) >= M && lat_mb->size(3) == kHidden,
                "lat_mb: bf16 [3, 16, Mmax, 3584]");
    TORCH_CHECK(lat_mc_ptr != 0 && lat_mc_ptr % 16 == 0 && par >= 0 && par < 3 && rank >= 0 && rank < kTp);
    TORCH_CHECK(rs_peers && static_cast<int>(rs_peers->size()) == kTp, "rs_peers: 16 addresses of rs_mb");
    TORCH_CHECK(scores.numel() * scores.element_size() >= 8 * M * 896 * 8 && scores.is_contiguous());
    TORCH_CHECK(h_shared.numel() == M * 2 * kSdK && h_shared.scalar_type() == at::kBFloat16 && h_shared.is_contiguous());
    TORCH_CHECK(w_sd.size(0) == kSdRows && w_sd.size(1) == kSdK && w_sd.scalar_type() == at::kBFloat16 &&
                w_sd.is_contiguous());
    TORCH_CHECK(ids_out.dim() == 2 && ids_out.size(1) == kTopK && ids_out.scalar_type() == at::kInt &&
                ids_out.is_contiguous() && wts_out.numel() == M * kTopK && wts_out.scalar_type() == at::kBFloat16);
    BlockArgs ba{};
    ba.scores = reinterpret_cast<const uint2*>(scores.data_ptr());
    ba.ids_out = ids_out.data_ptr<int>();
    ba.wts_out = reinterpret_cast<__nv_bfloat16*>(wts_out.data_ptr());
    ba.h_sh = reinterpret_cast<const __nv_bfloat16*>(h_shared.data_ptr());
    ba.w_sd = reinterpret_cast<const __nv_bfloat16*>(w_sd.data_ptr());
    ba.sh_out = nullptr;
    ba.rscale = static_cast<float>(routed_scaling_factor);
    ba.renorm = renormalize ? 1 : 0;
    ba.sh_beta = static_cast<float>(shared_beta);
    ba.sh_lbeta = static_cast<float>(shared_linear_beta);
    ba.latent_out = nullptr;
    TailArgs& ta = ba.tail;
    ta.lat_st = static_cast<unsigned long long>(lat_mc_ptr);
    ta.lat_mb = reinterpret_cast<const __nv_bfloat16*>(lat_mb->data_ptr());
    for (int d = 0; d < kTp; ++d) {
      TORCH_CHECK((*rs_peers)[d] != 0 && (*rs_peers)[d] % 16 == 0);
      ta.rs_peer[d] = static_cast<unsigned long long>((*rs_peers)[d]);
    }
    ta.par = static_cast<int>(par);
    ta.mmax = static_cast<int>(lat_mb->size(2));
    ta.rank = static_cast<int>(rank);
    ta.mc = multicast ? 1 : 0;
    launch_fused_pub<false, true, true, true>(mailbox, ids_out, nullptr, w13, w13_scale, w2, w2_scale, workspace,
                                              barrier, nullptr, beta, linear_beta, trace, 0, ba, static_cast<int>(grid));
    return;
  }
  TORCH_CHECK(ids_out.dim() == 2 && ids_out.size(1) == kTopK && ids_out.scalar_type() == at::kInt &&
              ids_out.is_contiguous());
  TORCH_CHECK(wts_out.numel() == M * kTopK && wts_out.scalar_type() == at::kBFloat16 && wts_out.is_contiguous());
  TORCH_CHECK(scores.numel() * scores.element_size() >= 8 * M * 896 * 8 && scores.is_contiguous(),
              "scores: route_shared scores-only output [8 replicas, M, 896, 2] fp32");
  TORCH_CHECK(h_shared.numel() == M * 2 * kSdK && h_shared.scalar_type() == at::kBFloat16 && h_shared.is_contiguous(),
              "h_shared: route_shared's shared gate_up output [M, 768] bf16");
  TORCH_CHECK(w_sd.size(0) == kSdRows && w_sd.size(1) == kSdK && w_sd.scalar_type() == at::kBFloat16 &&
              w_sd.is_contiguous());
  TORCH_CHECK(shared_out.numel() == M * kSdRows && shared_out.scalar_type() == at::kBFloat16 &&
              shared_out.is_contiguous());
  TORCH_CHECK(latent_out.has_value() ||
              (gemm2_out.dim() == 2 && gemm2_out.size(0) == M * kTopK && gemm2_out.size(1) == kHidden &&
               gemm2_out.scalar_type() == at::kBFloat16 && gemm2_out.is_contiguous()));
  if (latent_out)
    TORCH_CHECK(latent_out->numel() == M * kHidden && latent_out->scalar_type() == at::kBFloat16 &&
                latent_out->is_contiguous(), "latent_out: [M, 3584] bf16");
  BlockArgs ba{};
  ba.latent_out = latent_out ? reinterpret_cast<__nv_bfloat16*>(latent_out->data_ptr()) : nullptr;
  ba.scores = reinterpret_cast<const uint2*>(scores.data_ptr());
  ba.ids_out = ids_out.data_ptr<int>();
  ba.wts_out = reinterpret_cast<__nv_bfloat16*>(wts_out.data_ptr());
  ba.h_sh = reinterpret_cast<const __nv_bfloat16*>(h_shared.data_ptr());
  ba.w_sd = reinterpret_cast<const __nv_bfloat16*>(w_sd.data_ptr());
  ba.sh_out = reinterpret_cast<__nv_bfloat16*>(shared_out.data_ptr());
  ba.rscale = static_cast<float>(routed_scaling_factor);
  ba.renorm = renormalize ? 1 : 0;
  ba.sh_beta = static_cast<float>(shared_beta);
  ba.sh_lbeta = static_cast<float>(shared_linear_beta);
  launch_fused<false, true, true>(mailbox, ids_out, nullptr, w13, w13_scale, w2, w2_scale, workspace, barrier,
                                  reinterpret_cast<__nv_bfloat16*>(gemm2_out.data_ptr()), beta, linear_beta,
                                  trace, 0, ba, static_cast<int>(grid));
}

void topk_debug(torch::Tensor scores, torch::Tensor ids, torch::Tensor sc, torch::Tensor tr) {
  topk_debug_kernel<<<1, 32, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint2*>(scores.data_ptr()), ids.data_ptr<int>(), sc.data_ptr<float>(),
      reinterpret_cast<unsigned long long*>(tr.data_ptr()));
}

namespace {
__global__ void rsqrt_approx_kernel(const float* x, float* y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float r;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x[i]));
    y[i] = r;
  }
}
}  // namespace
// Test helper: the rsqrt.approx.ftz.f32 the tail (and vLLM's fastmath RMSNorm) uses, elementwise on fp32.
torch::Tensor rsqrt_approx(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat && x.is_contiguous());
  auto y = torch::empty_like(x);
  const int n = x.numel();
  if (n > 0)
    rsqrt_approx_kernel<<<(n + 255) / 256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),
                                                                                         y.data_ptr<float>(), n);
  return y;
}

TORCH_LIBRARY(k3mk, m) {
  m.def("rsqrt_approx(Tensor x) -> Tensor");
  m.def("moe_small(Tensor x, Tensor topk_ids, Tensor topk_weights, Tensor w13, Tensor w13_scale, "
        "Tensor w2, Tensor w2_scale, Tensor(a!) workspace, Tensor(b!) out, float beta, "
        "float linear_beta, int stages=3) -> ()");
  m.def("moe_fused(Tensor x, Tensor topk_ids, Tensor topk_weights, Tensor w13, Tensor w13_scale, "
        "Tensor w2, Tensor w2_scale, Tensor(a!) workspace, Tensor(b!) barrier, Tensor(c!) out, "
        "float beta, float linear_beta, Tensor(d!)? trace=None) -> ()");
  m.def("moe_fused_unfinalized(Tensor x, Tensor topk_ids, Tensor w13, Tensor w13_scale, "
        "Tensor w2, Tensor w2_scale, Tensor(a!) workspace, Tensor(b!) barrier, Tensor(c!) gemm2_out, "
        "float beta, float linear_beta, Tensor(d!)? trace=None) -> ()");
  m.def("route_shared(Tensor x, Tensor gate_w, Tensor bias, Tensor? sh_w13, Tensor(a!) sbuf, Tensor(b!) ids, "
        "Tensor(c!) wts, Tensor(d!)? h_shared, float beta, float linear_beta, bool renormalize, "
        "float routed_scaling_factor, Tensor(e!)? trace=None) -> ()");
  m.def("moe_block_lamport(Tensor(a!) mailbox, Tensor scores, Tensor w13, Tensor w13_scale, Tensor w2, "
        "Tensor w2_scale, Tensor(b!) workspace, Tensor(c!) barrier, Tensor(d!) gemm2_out, Tensor(e!) ids_out, "
        "Tensor(f!) wts_out, Tensor h_shared, Tensor w_sd, Tensor(g!) shared_out, float beta, float linear_beta, "
        "bool renormalize, float routed_scaling_factor, int grid=0, float shared_beta=4.0, "
        "float shared_linear_beta=25.0, Tensor(h!)? trace=None, Tensor(i!)? latent_out=None, "
        "Tensor(j!)? lat_mb=None, int lat_mc_ptr=0, int[]? rs_peers=None, int par=0, int rank=0, "
        "bool multicast=True) -> ()");
  m.def("tail(Tensor(a!) lat_mb, Tensor(b!) rs_mb, int par, Tensor w_up, Tensor gamma, float eps, Tensor(c!) up_mb, "
        "int up_mc_ptr, Tensor(d!) epoch, int rank, int M, Tensor(e!)? trace=None, Tensor? gemm2=None, "
        "Tensor? wts=None, Tensor? shared_out=None, int lat_mc_ptr=0, int[]? rs_peers=None, "
        "bool multicast=True, int variant=0) -> ()");
  m.def("topk_debug(Tensor scores, Tensor(a!) ids, Tensor(b!) sc, Tensor(c!) tr) -> ()");
  m.def("moe_fused_unfinalized_lamport(Tensor(a!) mailbox, Tensor topk_ids, Tensor w13, Tensor w13_scale, "
        "Tensor w2, Tensor w2_scale, Tensor(b!) workspace, Tensor(c!) barrier, Tensor(d!) gemm2_out, "
        "float beta, float linear_beta, bool wait_prior=True, Tensor(e!)? trace=None) -> ()");
}
TORCH_LIBRARY_IMPL(k3mk, CUDA, m) {
  m.impl("moe_small", &moe_small);
  m.impl("moe_fused", &moe_fused);
  m.impl("moe_fused_unfinalized", &moe_fused_unfinalized);
  m.impl("moe_fused_unfinalized_lamport", &moe_fused_unfinalized_lamport);
  m.impl("route_shared", &route_shared);
  m.impl("moe_block_lamport", &moe_block_lamport);
  m.impl("rsqrt_approx", &rsqrt_approx);
  m.impl("tail", &tail);
  m.impl("topk_debug", &topk_debug);
}
