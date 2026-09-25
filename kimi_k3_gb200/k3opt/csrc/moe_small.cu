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

// ---- MoE tail fused into the block kernel (k3moe.moe_block_tail) -------------------------------
// Replaces vLLM's KimiK3LatentMoETailOp (CollectiveKernel: top-16 finalize + Lamport/NVLS latent
// all-reduce + RMSNorm + shared reduce-scatter; AdaptiveUpProjectionKernel: up-proj GEMV + shared
// add + multicast into the up-proj mailbox) with the same rounding points:
//   partial_r = bf16(sum_j fp32 fma(bf16 row_j, bf16 w_j))            (finalize_top16_bf16)
//   x = bf16(sum_{r=0..15} fp32 partial_r)  [fixed rank order]         (routed all-reduce)
//   inv = rsqrt(sum fp32(bf16(x*x)) / 3584 + eps); xn = bf16(x * inv * gamma)
//   sh = bf16(sum_{r=0..15} fp32 shared_partial_r[:, shard])            (shared reduce-scatter)
//   out[:, shard] = bf16(fp32(bf16(xn . W_up_row)) + fp32(sh))          (skinny up-proj epilogue)
// Mailboxes (bf16, Lamport: an empty 32-bit word is 0x80000000, producers never write bf16 -0.0):
//   latent AR  [4 tokens][16 src ranks][3584]: CTA s multicasts (multimem.st) the finalized 16-byte
//              fragments of ITS FC2 columns into slot [t][rank] of every rank, then polls the same
//              fragments from all 16 slots (its own copy), sums, and re-arms them.
//   shared RS  [4][16][448]: the shared-expert down rows (split in 8-row fragments) are stored
//              straight into the owner rank's buffer (peer pointer), slot [t][rank].
//   up mailbox vLLM's AdaptiveUpProjectionKernel._mailbox [1, 128, 7168] (consumed by tailattn's
//              lamport_attn_res / LamportCopy): rank r writes columns [448 r, 448 r + 448).
// Single buffers are safe for the same reason vLLM's up-proj mailbox is: between two uses of a slot
// every rank passes a later all-reduce (o_proj) that the re-arming consumer precedes.
constexpr int kTp = 16;                 // TP ranks
constexpr int kUpShard = 7168 / kTp;    // 448 hidden rows per rank
constexpr int kRsFrags = kUpShard / 8;  // 56 16-byte fragments per rank shard
constexpr int kUpPairs = kUpShard / 2;  // 224 row pairs (one 32-bit mailbox word per token)
constexpr int kUpRowBytes = kHidden * 2;  // 7168
constexpr int kTailMinGrid = 112;       // <= 4 up-proj rows per CTA: rows + gamma fit one ring stage
constexpr int kTailMaxM = 4;
constexpr int kMaxGrid = 160;           // tail: G <= 160 CTAs (per-lane partial-sum registers)
constexpr int kGatherRep = 8;           // replicas of the intra-GPU gather buffers (<= 19 readers per line)
#ifndef K3TAIL_BACKOFF
#define K3TAIL_BACKOFF 64
#endif
constexpr int kPollBackoffNs = K3TAIL_BACKOFF;  // __nanosleep between failed polls (0: spin)

struct TailArgs {
  unsigned long long lat_st;        // store base of the latent AR mailbox (multicast or local address)
  const __nv_bfloat16* lat_mb;      // local latent AR mailbox (poll + re-arm); nullptr = no tail
  unsigned long long rs_peer[kTp];  // every rank's shared RS mailbox base (peer-mapped)
  const __nv_bfloat16* rs_mb;       // local shared RS mailbox
  unsigned long long up_st;         // store base of the up-proj mailbox (multicast or local address)
  const __nv_bfloat16* w_up;        // [448][3584] this rank's up_proj rows
  const __nv_bfloat16* gamma;       // [3584] RMSNorm weight
  // Intra-GPU gather buffers (tail_ws), read after grid barrier #2.
  __nv_bfloat16* g_lat;             // [8 rep][M][3584] all-reduced latent (bf16)
  __nv_bfloat16* g_shs;             // [M][448] reduce-scattered shared shard (bf16)
  float* g_ss;                      // [8 rep][M][G] per-CTA sums of squares
  float eps;
  int rank;
  int mc;                           // 1: multimem.st (NVLS) latent/up stores, 0: plain st (tests)
  int dbg;                          // profiling only (env K3TAIL_DBG): 1/2/3 = stop after FC2/publish/gather-write
};

// Shared-memory map of the tail extras (bytes from a 16-aligned base), identical on host and device.
struct TailLayout {
  int nu, nq, sdr;                  // max FC2 units, RS fragments, shared-down rows per CTA
  int fin, finout, pol, sdo, ssq, upart, invs, args, bytes;
};
__host__ __device__ inline TailLayout tail_layout(int M, int G) {
  TailLayout L;
  L.nu = M * ((kHidden / 8 + G - 1) / G);
  L.nq = (kRsFrags * M + G - 1) / G;
  L.sdr = 8 * ((kHidden * 2 / 8 + G - 1) / G);
  int o = 0;
  L.fin = o;    o += L.nu * kTopK * 16;          // bf16 [unit][16 experts][8]
  L.finout = o; o += L.nu * 16;                  // bf16 [unit][8] finalized
  L.pol = o;    o += (L.nu + L.nq) * kTp * 16;   // polled fragments [item][src]
  L.sdo = o;    o += (M * L.sdr * 2 + 15) & ~15; // bf16 shared-down rows [t][row]
  L.ssq = o;    o += L.nu * 8 * 4;               // fp32 squares [unit][8]
  L.upart = o;  o += 16 * 4 * kTailMaxM * 4;     // fp32 GEMV partials [warp][row][t]
  L.invs = o;   o += 96;                         // inv_rms [4], shared values [4][4], tag
  L.args = o;   o += (sizeof(TailArgs) + 15) & ~15;  // copy of the TailArgs (no late constant-cache misses)
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
  TailArgs tail;                  // moe_block_tail only (moe_fused_tail_kernel)
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
  float* acc2 = red + max_slots;
  float* wsm = acc2 + max_rows2;
  so += (max_slots + max_rows2 + M * kTopK) * 4;
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
  for (int i = tid; i < max_slots + max_rows2; i += kFusedThreads) red[i] = 0.f;
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
        for (int f = tid; f < M * (kHidden / 8); f += kConsumerThreads) {
          const uint4* src = reinterpret_cast<const uint4*>(x) + f;
          uint4 v;
          do {
            asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(src) : "memory");
          } while (v.x == 0x80000000u || v.y == 0x80000000u || v.z == 0x80000000u || v.w == 0x80000000u);
          reinterpret_cast<uint4*>(ba.latent_out)[f] = v;
          asm volatile("st.global.v4.u32 [%0], {%1, %1, %1, %1};" ::"l"(src), "r"(0x80000000u) : "memory");
        }
      }
      mbar_arrive(xbar);
    } else if constexpr (kLamport) {
      // Poll this CTA's x rows (<= 2 tokens) out of the mailbox, 16 B per thread per token.
      const int t_lo = W.qa / kPairsPerTok, t_hi = (W.qb - 1) / kPairsPerTok;
      constexpr int kFrags = kHidden / 8;  // 448
      for (int f = t_lo * kFrags + tid; f < (t_hi + 1) * kFrags; f += kConsumerThreads) {
        const uint4* src = reinterpret_cast<const uint4*>(x) + f;
        uint4 v;
        do {
          asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(src) : "memory");
        } while (v.x == 0x80000000u || v.y == 0x80000000u || v.z == 0x80000000u || v.w == 0x80000000u);
        reinterpret_cast<uint4*>(xs)[f] = v;
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
      if (!(i8 & 1) && sl < 2 * n) atomicAdd(red + sigma0 + sl, kp);
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
      const float up = red[su], gate = red[su + (ge - gs)];
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
          atomicAdd(acc2 + unit * 8 + rho, wsm[t * kTopK + j] * acc);
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
        out[t * kHidden + n_out] = __float2bfloat16(acc2[i]);
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
// Fused MoE tail body (moe_block_tail), run by the 16 consumer warps after FC2. (A `dry` warm-up
// run by the FC1-idle warps 14/15 -- to pull this code into the instruction caches early -- was
// tried and made the kernel slower overall; dry is always 0 now and folds away.)
struct TailCtx {
  const TailArgs* ta;             // smem copy
  uint8_t* tsm;                   // tail smem base
  TailLayout TL;
  const float* wsm;               // [M][16] bf16-rounded routing weights
  const uint8_t* stages;          // smem ring
  uint32_t full_bar0;             // smem address of full_bar(0)
  int nstages, M, nunits, nocts, oa, k_up, up_a, up_b;
  unsigned long long* counter;    // grid-barrier counter (barrier[M-1])
  unsigned long long bar_target;  // barrier #1 target (tid 0 only)
  unsigned long long* trsm;       // trace marks (smem) or nullptr
};

__device__ __forceinline__ void tail_run(const TailCtx& c, int tid, int dry) {
  const TailArgs& ta = *c.ta;
  const int warp = tid >> 5, lane = tid & 31;
  const int G = gridDim.x, sb = blockIdx.x, M = c.M;
  const __nv_bfloat16* fin = reinterpret_cast<const __nv_bfloat16*>(c.tsm + c.TL.fin);
  uint4* pol = reinterpret_cast<uint4*>(c.tsm + c.TL.pol);
  float* ssq = reinterpret_cast<float*>(c.tsm + c.TL.ssq);
  float* upart = reinterpret_cast<float*>(c.tsm + c.TL.upart);
  float* invs = reinterpret_cast<float*>(c.tsm + c.TL.invs);  // [4] inv_rms, [4][4] shared shard values
  float* shv = invs + 4;
  const bool live = dry == 0;
#if defined(K3_TRACE_CLOCK) || defined(K3_TRACE_SMEM)
#ifdef K3_TRACE_CLOCK
#define TL_MARK(slot) if (c.trsm && live && tid == 0) c.trsm[slot] = clock64();
#else
#define TL_MARK(slot)                                                   \
  if (c.trsm && live && tid == 0) {                                     \
    unsigned long long ts_;                                             \
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts_));             \
    c.trsm[slot] = ts_;                                                 \
  }
#endif
#else
#define TL_MARK(slot)
#endif
  if (live) consumer_sync();
  TL_MARK(19)
  if (live && ta.dbg == 1) return;
  // (1) Top-16 finalize of this CTA's output columns (fp32 FMA over slots 0..15 of bf16 row x bf16
  //     weight) and (2) publish the 16-byte fragments into slot [t][rank] of every rank's latent
  //     all-reduce mailbox.
  if (tid < ((c.nunits * 8 + 31) & ~31)) {
    const bool ok = tid < c.nunits * 8;
    const int u = ok ? tid >> 3 : 0, e = tid & 7, t = u / c.nocts;
    float a = 0.f;
#pragma unroll 1
    for (int jj = 0; jj < kTopK; ++jj) a = fmaf(__bfloat162float(fin[(u * kTopK + jj) * 8 + e]), c.wsm[t * kTopK + jj], a);
    const __nv_bfloat16 b16 = __float2bfloat16(a);
    uint32_t v = *reinterpret_cast<const uint16_t*>(&b16);
    v |= __shfl_down_sync(0xffffffffu, v, 1) << 16;
    const uint32_t w1 = __shfl_down_sync(0xffffffffu, v, 2);
    const uint32_t w2 = __shfl_down_sync(0xffffffffu, v, 4);
    const uint32_t w3 = __shfl_down_sync(0xffffffffu, v, 6);
    if (live && ok && e == 0) {
      const int f = c.oa + u % c.nocts;
      st_mb16(ta.lat_st + static_cast<unsigned long long>((t * kTp + ta.rank) * kHidden + 8 * f) * 2,
              no_neg_zero4(make_uint4(v, w1, w2, w3)), ta.mc);
    }
  }
  TL_MARK(26)
  if (live && ta.dbg == 2) return;
  // (3) Poll + re-arm the same fragments in all 16 source slots, and this CTA's shared RS fragments
  //     q = sb, sb + G, ... of this rank's shard (16 sources each).
  const int nq = sb < kRsFrags * M ? (kRsFrags * M - 1 - sb) / G + 1 : 0;
  for (int i = tid; i < (c.nunits + nq) * kTp; i += kConsumerThreads) {
    const int u = i >> 4, src = i & 15;
    const __nv_bfloat16* p;
    if (u < c.nunits) {
      const int t = u / c.nocts, f = c.oa + u % c.nocts;
      p = ta.lat_mb + (t * kTp + src) * kHidden + 8 * f;
    } else {
      const int q = sb + (u - c.nunits) * G, t = q / kRsFrags, cf = q - t * kRsFrags;
      p = ta.rs_mb + (t * kTp + src) * kUpShard + 8 * cf;
    }
    uint4 v = ld_volatile16(p);
    while (live && lamport_dirty(v)) {
      if (kPollBackoffNs) __nanosleep(kPollBackoffNs);
      v = ld_volatile16(p);
    }
    pol[i] = v;
    if (live) st_sentinel16(p);  // this CTA is the fragment's only reader on this rank
  }
  if (live) consumer_sync();
  TL_MARK(28)
  // (4) Rank sums in fixed order 0..15 (fp32) -> bf16; squares in bf16 (upstream-compatible mode).
  //     Results go to the intra-GPU gather buffers (latent and partial sums of squares replicated
  //     kGatherRep times so that <= 19 CTAs read any line).
  for (int i = tid; i < (c.nunits + nq) * 8; i += kConsumerThreads) {
    const int u = i >> 3, e = i & 7;
    float a = 0.f;
#pragma unroll 1
    for (int src = 0; src < kTp; ++src) {
      const uint32_t w = reinterpret_cast<const uint32_t*>(pol + u * kTp + src)[e >> 1];
      a += (e & 1) ? bf16_hi(w) : bf16_lo(w);
    }
    const __nv_bfloat16 nb = __float2bfloat16(a);
    if (u < c.nunits) {
      const int t = u / c.nocts, f = c.oa + u % c.nocts;
      if (live)
#pragma unroll
        for (int rp = 0; rp < kGatherRep; ++rp) ta.g_lat[(rp * M + t) * kHidden + 8 * f + e] = nb;
      ssq[i] = __bfloat162float(__hmul(nb, nb));
    } else {
      const int q = sb + (u - c.nunits) * G, t = q / kRsFrags, cf = q - t * kRsFrags;
      if (live) ta.g_shs[t * kUpShard + 8 * cf + e] = nb;
    }
  }
  if (live) consumer_sync();
  if (tid < M) {  // this CTA's partial sum of squares per token (fixed order)
    float s2 = 0.f;
    for (int i = tid * c.nocts * 8; i < (tid + 1) * c.nocts * 8; ++i) s2 += ssq[i];
    if (live)
#pragma unroll
      for (int rp = 0; rp < kGatherRep; ++rp) ta.g_ss[(rp * M + tid) * G + sb] = s2;
  }
  if (live && ta.dbg == 3) return;
  // (5) Grid barrier #2 on the same monotonic counter (this call adds 2G): release this CTA's gather
  //     writes, acquire everyone's. (One counter polled by one thread per CTA: ~0.9 us here. Polling
  //     per-CTA flags from every CTA instead takes ~10 us on GB200 -- see gather_probe2.py.)
  if (live) {
    consumer_sync();
    if (tid == 0) {
      asm volatile("red.release.gpu.global.add.u64 [%0], 1;" ::"l"(c.counter) : "memory");
      unsigned long long cur;
      do {
        asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(cur) : "l"(c.counter) : "memory");
      } while (cur < c.bar_target + static_cast<unsigned long long>(G));
    }
    consumer_sync();
  }
  TL_MARK(29)
  if (live && ta.dbg == 8) return;
  // (6) RMSNorm + up-proj GEMV. Warp 15 loads the G partial sums (-> inv_rms) and this CTA's shared
  //     shard values; warps 0..13 (thread c: K elements [8c, 8c+8)) load their latent slice.
  const int nr = c.up_b - c.up_a;
  if (warp == kConsumerWarps - 1) {
    constexpr int kSsPer = (kMaxGrid + 31) / 32;
    float v[kTailMaxM][kSsPer];
    const float* pss = ta.g_ss + (sb % kGatherRep) * M * G;
#pragma unroll
    for (int t = 0; t < kTailMaxM; ++t)
#pragma unroll
      for (int k = 0; k < kSsPer; ++k) {
        v[t][k] = 0.f;
        if (t < M && lane + 32 * k < G)
          asm volatile("ld.relaxed.gpu.global.f32 %0, [%1];" : "=f"(v[t][k]) : "l"(pss + t * G + lane + 32 * k)
                       : "memory");
      }
    if (lane < nr * M) {
      unsigned short h;
      asm volatile("ld.relaxed.gpu.global.u16 %0, [%1];"
                   : "=h"(h) : "l"(ta.g_shs + (lane / nr) * kUpShard + c.up_a + lane % nr) : "memory");
      shv[(lane / nr) * 4 + lane % nr] = __uint_as_float(static_cast<uint32_t>(h) << 16);
    }
#pragma unroll
    for (int t = 0; t < kTailMaxM; ++t) {
      if (t < M) {
        float s2 = 0.f;
#pragma unroll
        for (int k = 0; k < kSsPer; ++k) s2 += v[t][k];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s2 += __shfl_xor_sync(0xffffffffu, s2, o);
        float inv;
        asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(s2 / static_cast<float>(kHidden) + ta.eps));
        if (lane == 0) invs[t] = inv;
      }
    }
    if (live) asm volatile("bar.sync 3, %0;" ::"n"(15 * 32) : "memory");
  } else if (warp < kHidden / 8 / 32) {
    uint32_t xw[kTailMaxM][4];  // this thread's 8 latent values per token, packed bf16x2
    {
      const __nv_bfloat16* p = ta.g_lat + (sb % kGatherRep) * M * kHidden + 8 * tid;
#pragma unroll
      for (int t = 0; t < kTailMaxM; ++t) {
        uint4 q = make_uint4(0u, 0u, 0u, 0u);
        if (t < M) q = ld_relaxed16(p + t * kHidden);
        xw[t][0] = q.x;
        xw[t][1] = q.y;
        xw[t][2] = q.z;
        xw[t][3] = q.w;
      }
    }
    if (live) mbar_wait(c.full_bar0 + 8 * (c.k_up % c.nstages), (c.k_up / c.nstages) & 1);
    const uint8_t* ust = c.stages + (c.k_up % c.nstages) * kStageBytes;
    const uint4 gv = reinterpret_cast<const uint4*>(ust + 4 * kUpRowBytes)[tid];
    const uint32_t gw[4] = {gv.x, gv.y, gv.z, gv.w};
    if (live) asm volatile("bar.sync 3, %0;" ::"n"(15 * 32) : "memory");
    // xn = bf16(x * inv * gamma)
#pragma unroll
    for (int t = 0; t < kTailMaxM; ++t) {
      if (t < M) {
        const float inv = invs[t];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const __nv_bfloat162 r2 = __floats2bfloat162_rn(bf16_lo(xw[t][q]) * inv * bf16_lo(gw[q]),
                                                          bf16_hi(xw[t][q]) * inv * bf16_hi(gw[q]));
          xw[t][q] = *reinterpret_cast<const uint32_t*>(&r2);
        }
      }
    }
#pragma unroll 1
    for (int rr = 0; rr < nr; ++rr) {
      const uint4 wv = *reinterpret_cast<const uint4*>(ust + rr * kUpRowBytes + tid * 16);
      const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
      for (int t = 0; t < kTailMaxM; ++t) {
        if (t < M) {
          float a = 0.f;
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            a = fmaf(bf16_lo(ww[q]), bf16_lo(xw[t][q]), a);
            a = fmaf(bf16_hi(ww[q]), bf16_hi(xw[t][q]), a);
          }
          a = warp_sum(a);
          if (lane == 0) upart[(warp * 4 + rr) * kTailMaxM + t] = a;
        }
      }
    }
  }
  if (live) consumer_sync();
  TL_MARK(16)
  // (7) Epilogue: bf16(bf16(dot) + shared shard) -> 32-bit words (2 rows) -> up-proj mailbox.
  if (tid < (nr / 2) * M) {
    const int pp = tid / M, t = tid - pp * M;
    uint32_t word = 0;
#pragma unroll
    for (int h2 = 0; h2 < 2; ++h2) {
      const int row = 2 * pp + h2;
      float tot = 0.f;
      for (int w = 0; w < kHidden / 8 / 32; ++w) tot += upart[(w * 4 + row) * kTailMaxM + t];
      const float gvv = __bfloat162float(__float2bfloat16(tot));
      const __nv_bfloat16 o = __float2bfloat16(gvv + shv[t * 4 + row]);
      word |= static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&o)) << (16 * h2);
    }
    if (live)
      st_mb4(ta.up_st + static_cast<unsigned long long>(t * (kHidden * 2) + ta.rank * kUpShard + c.up_a + 2 * pp) * 2,
             no_neg_zero(word), ta.mc);
  }
  TL_MARK(30)
#undef TL_MARK
}

// moe_block_tail's kernel: a separate copy of moe_fused_kernel with the fused MoE tail (kTail), so the
// production kernel above keeps its exact code (code layout changes alone moved it by ~1 us).
template <bool kFinal, bool kLamport, bool kBlock = false, bool kTail = false>
__global__ void __launch_bounds__(kFusedThreads, 1)
moe_fused_tail_kernel(const __grid_constant__ CUtensorMap tm_w13s, const __grid_constant__ CUtensorMap tm_w2,
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
  float* acc2 = red + max_slots;
  float* wsm = acc2 + max_rows2;
  so += (max_slots + max_rows2 + M * kTopK) * 4;
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
  // Fused MoE tail (moe_block_tail): extra smem after hsh.
  // The tail is a separate instantiation: its code (executed once per call) would otherwise sit in the
  // non-tail kernel's instruction stream (~1 us slower there, measured: cold instruction fetch).
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
  for (int i = tid; i < max_slots + max_rows2; i += kFusedThreads) red[i] = 0.f;
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

  const int total_seq = nsd + W.ngroups + W.nf2 + (tail ? 1 : 0);  // tail: + up_proj rows & gamma stage
  // Tail: this CTA's up_proj rows [up_a, up_b) (whole row pairs, 2..4 rows for G >= 112).
  const int up_a = 2 * (kUpPairs * static_cast<int>(blockIdx.x) / static_cast<int>(gridDim.x));
  const int up_b = 2 * (kUpPairs * (static_cast<int>(blockIdx.x) + 1) / static_cast<int>(gridDim.x));
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
      } else if (tail && lane == 0) {  // tail: up_proj rows + RMSNorm gamma (static; streamed under FC2)
        mbar_arrive_expect_tx(bar, (up_b - up_a + 1) * kUpRowBytes);
        tma_1d(smem_u32(dst), ba.tail.w_up + static_cast<long>(up_a) * kHidden, (up_b - up_a) * kUpRowBytes, bar);
        tma_1d(smem_u32(dst + 4 * kUpRowBytes), ba.tail.gamma, kUpRowBytes, bar);
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
    auto fill_tail_ctx = [&](TailCtx& tc) {
      tc.ta = reinterpret_cast<const TailArgs*>(tsm + TL.args);
      tc.tsm = tsm;
      tc.TL = TL;
      tc.wsm = wsm;
      tc.stages = stages;
      tc.full_bar0 = full_bar(0);
      tc.nstages = nstages;
      tc.M = M;
      tc.nunits = W.nunits;
      tc.nocts = W.nocts;
      tc.oa = W.oa;
      tc.k_up = nsd + W.ngroups + W.nf2;
      tc.up_a = up_a;
      tc.up_b = up_b;
      tc.counter = counter;
      tc.bar_target = bar_target;
#if defined(K3_TRACE_CLOCK) || defined(K3_TRACE_SMEM)
      tc.trsm = trace ? trsm : nullptr;
#else
      tc.trsm = nullptr;
#endif
    };
    if (routed_only) {
      // CTA 0 hands the latent to the next kernel: poll, copy out, re-arm (like LamportCopy).
      if (blockIdx.x == 0) {
        for (int f = tid; f < M * (kHidden / 8); f += kConsumerThreads) {
          const uint4* src = reinterpret_cast<const uint4*>(x) + f;
          uint4 v;
          do {
            asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(src) : "memory");
          } while (v.x == 0x80000000u || v.y == 0x80000000u || v.z == 0x80000000u || v.w == 0x80000000u);
          reinterpret_cast<uint4*>(ba.latent_out)[f] = v;
          asm volatile("st.global.v4.u32 [%0], {%1, %1, %1, %1};" ::"l"(src), "r"(0x80000000u) : "memory");
        }
      }
      mbar_arrive(xbar);
    } else if constexpr (kLamport) {
      // Poll this CTA's x rows (<= 2 tokens) out of the mailbox, 16 B per thread per token.
      const int t_lo = W.qa / kPairsPerTok, t_hi = (W.qb - 1) / kPairsPerTok;
      constexpr int kFrags = kHidden / 8;  // 448
      for (int f = t_lo * kFrags + tid; f < (t_hi + 1) * kFrags; f += kConsumerThreads) {
        const uint4* src = reinterpret_cast<const uint4*>(x) + f;
        uint4 v;
        do {
          asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(src) : "memory");
        } while (v.x == 0x80000000u || v.y == 0x80000000u || v.z == 0x80000000u || v.w == 0x80000000u);
        reinterpret_cast<uint4*>(xs)[f] = v;
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
          st_mb16(tas.rs_peer[d] + static_cast<unsigned long long>((t * kTp + tas.rank) * kUpShard + col) * 2, v, 0);
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
      if (!(i8 & 1) && sl < 2 * n) atomicAdd(red + sigma0 + sl, kp);
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
      const float up = red[su], gate = red[su + (ge - gs)];
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
          atomicAdd(acc2 + unit * 8 + rho, wsm[t * kTopK + j] * acc);
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
        out[t * kHidden + n_out] = __float2bfloat16(acc2[i]);
      }
    }
    if (tail) {
      TailCtx tc;
      fill_tail_ctx(tc);
      tail_run(tc, tid, 0);
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
  const int fixed = M * kHidden * 2 + (max_slots + max_rows2 + M * kTopK) * 4 + block_extra + 16 +
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
void launch_fused_tail(const torch::Tensor& x, const torch::Tensor& topk_ids, const float* topk_w,
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
    cudaFuncSetAttribute(moe_fused_tail_kernel<false, true, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kMaxDynSmem);
#endif
  }
  const int G = grid > 0 ? std::min(grid, sms) : sms;
  constexpr bool tail = kBlock && kTail;
  TORCH_CHECK(!tail || ba.tail.lat_mb != nullptr);
  TORCH_CHECK(!tail || (G >= kTailMinGrid && G <= kMaxGrid && M <= kTailMaxM),
              "moe_block_tail needs 112..160 CTAs and M <= 4");
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
  const int fixed = M * kHidden * 2 + (max_slots + max_rows2 + M * kTopK) * 4 + block_extra + (tail ? 64 : 16) +
                    kTraceSmem + (2 + 2 * 8) * 8 + 128;
  const int nstages = std::min(8, (kMaxDynSmem - fixed) / kStageBytes);
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
      &cfg, moe_fused_tail_kernel<kFinal, kLamport, kBlock, kTail>, tm_w13s, tm_w2, tm_w2s,
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
// k3moe.moe_tail: the whole latent-MoE tail as ONE kernel -- 7 thread-block clusters x 8 CTAs
// (56 CTAs x 256 threads) -- replacing vLLM's CollectiveKernel + AdaptiveUpProjectionKernel with the
// same rounding points (see TailArgs). Works on the unfinalized MoE output of either
// moe_block_lamport (CUDA cores) or agents/moe8 (tcgen05): gemm2 rows t*16+j, bf16 weights [M,16],
// shared-expert partial [M,7168].
//   CTA g (cluster c = g/8, rank q in cluster) owns up-proj rows [8g, 8g+8) of this rank's 448.
//   prologue  TMA its 8 up_proj rows (57 KB) + gamma into smem (overlaps the PDL predecessor)
//   finalize  fragments f = g + 56 i (8 output columns each): fp32 FMA over slots 0..15, bf16,
//             multimem.st into slot [t][rank] of the latent all-reduce mailbox (2 buffers, see below)
//   shared    RS: 16-byte fragments of the shared partial -> owner rank's RS mailbox (peer stores)
//   gather    every cluster rebuilds the full all-reduced latent: CTA q polls fragments f = q mod 8
//             (all 16 sources), sums in rank order, bf16, and writes them into all 8 CTAs' smem
//             (DSMEM); per-CTA sums of squares likewise; one cluster barrier.
//   RMSNorm   inv = rsqrt(sum/3584 + eps); xn = bf16(x * inv * gamma) (bf16 squares, fp32 sums)
//   up-proj   warp r = row 8g+r; bf16(bf16(dot) + reduce-scattered shared) -> 16-byte multimem.st
//             into vLLM's up-proj mailbox (columns rank*448 + 8g).
// Latent mailbox: [2 buffers][4][16][3584]; the buffer alternates per call (call index from a device
// counter, identical on every rank). Each fragment is read by 7 clusters, so it is re-armed one call
// later: a call first re-arms the other buffer (all its readers finished with the previous kernel),
// then publishes. RS mailbox [2][4][16][448]: one reader per fragment, re-armed right after reading.
// ===========================================================================
constexpr int kMtCtas = 56;
constexpr int kMtCluster = 8;
constexpr int kMtThreads = 256;
constexpr int kMtFin = kHidden / 8 / kMtCtas;          // 8 finalize fragments per token per CTA
constexpr int kMtShf = 2 * kHidden / 8 / kMtCtas;      // 16 shared RS fragments per token per CTA
constexpr int kMtPoll = kHidden / 8 / kMtCluster;      // 56 latent fragments polled per token per CTA
constexpr unsigned long long kMtLatBuf = 4ull * kTp * kHidden * 2;
constexpr unsigned long long kMtRsBuf = 4ull * kTp * kUpShard * 2;
constexpr int kMtW = 8 * kUpRowBytes;                  // 57344
static_assert(kMtCtas * 8 == kUpShard && kMtCtas % kMtCluster == 0, "moe_tail geometry");

struct MtArgs {
  const __nv_bfloat16* gemm2;    // [M*16][3584]
  const __nv_bfloat16* wts;      // [M][16]
  const __nv_bfloat16* sh;       // [M][7168]
  const __nv_bfloat16* w_up;     // [448][3584]
  const __nv_bfloat16* gamma;    // [3584]
  __nv_bfloat16* lat_mb;         // local [2][4][16][3584]
  unsigned long long lat_st;     // multicast (or plain) address of lat_mb buffer 0
  __nv_bfloat16* rs_mb;          // local [2][4][16][448]
  unsigned long long rs_peer[kTp];
  unsigned long long up_st;      // multicast (or plain) address of the up-proj mailbox
  unsigned long long* counter;   // call counter (+56 per call)
  unsigned long long* trace;     // [56][16] %globaltimer marks or nullptr
  float eps;
  int rank, mc, M;
  int dbg;                       // profiling only (env K3TAIL_DBG=21: skip the latent publish)
};

__device__ __forceinline__ void st_cluster16(uint32_t local_addr, uint32_t cta, const uint4& v) {
  uint32_t ra;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(local_addr), "r"(cta));
  asm volatile("st.shared::cluster.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(ra), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w) : "memory");
}
__device__ __forceinline__ void st_cluster_f32(uint32_t local_addr, uint32_t cta, float v) {
  uint32_t ra;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(local_addr), "r"(cta));
  asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(ra), "f"(v) : "memory");
}

__global__ void __launch_bounds__(kMtThreads, 1) moe_tail_kernel(const __grid_constant__ MtArgs a) {
  extern __shared__ __align__(128) uint8_t sm[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = blockIdx.x, M = a.M, rank = a.rank;
  uint32_t q;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(q));
  // smem: W rows | gamma | x/xn [M][3584] bf16 | pol [M*56][16] | shp [M][16] | ssp [8][4] | misc
  uint8_t* wsm = sm;
  const uint4* gsm = reinterpret_cast<const uint4*>(sm + kMtW);
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(sm + kMtW + kUpRowBytes);
  uint4* pol = reinterpret_cast<uint4*>(sm + kMtW + kUpRowBytes + M * kUpRowBytes);
  uint4* shp = pol + M * kMtPoll * kTp;
  float* ssp = reinterpret_cast<float*>(shp + M * kTp);  // [8][4] (written by every CTA of the cluster)
  float* wtf = ssp + 32;                                 // [4][16] routing weights
  float* spart = wtf + 64;                               // [4][14] per-warp partial sums of squares
  float* shs = spart + 56;                               // [4][8] reduce-scattered shared values
  float* inv = shs + 32;                                 // [4]
  __nv_bfloat16* outv = reinterpret_cast<__nv_bfloat16*>(inv + 4);  // [4][8]
  uint64_t* bar = reinterpret_cast<uint64_t*>(outv + 32);
  unsigned long long* s_call = reinterpret_cast<unsigned long long*>(bar + 1);
#ifdef K3_TRACE_CLOCK
#define MT_MARK(slot)                                                             \
  if (a.trace && tid == 0) a.trace[g * 16 + (slot)] = clock64();
#else
#define MT_MARK(slot)                                                             \
  if (a.trace && tid == 0) {                                                      \
    unsigned long long ts_;                                                       \
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts_));                      \
    a.trace[g * 16 + (slot)] = ts_;                                               \
  }
#endif
  MT_MARK(0)
  if (tid == 0) {
    mbar_init(smem_u32(bar), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_arrive_expect_tx(smem_u32(bar), kMtW + kUpRowBytes);
    tma_1d(smem_u32(wsm), a.w_up + static_cast<long>(8 * g) * kHidden, kMtW, smem_u32(bar));
    tma_1d(smem_u32(gsm), a.gamma, kUpRowBytes, smem_u32(bar));
  }
  // Cluster peers must be running before any DSMEM store: arrive now, wait before the first store.
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  __syncthreads();
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");
  MT_MARK(1)
  // All independent loads of this phase are issued together (one round trip): call counter, routing
  // weights, the 16 gemm2 values of this thread's finalize element, its shared-partial fragment.
  const bool fin_ok = tid < M * kMtFin * 8;
  const int fit = fin_ok ? tid >> 3 : 0, fe = tid & 7, ft = fit / kMtFin, ff = g + kMtCtas * (fit % kMtFin);
  float vj[kTopK];
  {
    const __nv_bfloat16* src = a.gemm2 + static_cast<long>(ft * kTopK) * kHidden + 8 * ff + fe;
#pragma unroll
    for (int j = 0; j < kTopK; ++j) vj[j] = fin_ok ? __bfloat162float(src[static_cast<long>(j) * kHidden]) : 0.f;
  }
  const bool sh_ok = tid < M * kMtShf;
  const int st_ = sh_ok ? tid / kMtShf : 0, scol = 8 * (g + kMtCtas * (tid % kMtShf));
  uint4 shv4 = make_uint4(0u, 0u, 0u, 0u);
  if (sh_ok) shv4 = *reinterpret_cast<const uint4*>(a.sh + static_cast<long>(st_) * 2 * kHidden + scol);
  if (tid == 0) {
    unsigned long long v0;
    asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v0) : "l"(a.counter) : "memory");
    *s_call = v0 / kMtCtas;  // every CTA of this call reads a value in [56 k, 56 k + 55]
  }
  if (tid < M * kTopK) wtf[tid] = __bfloat162float(a.wts[tid]);
  __syncthreads();
  const int par = static_cast<int>(*s_call & 1);
  // (2) finalize this CTA's fragments and publish them (multimem) into slot [t][rank]
  if (tid < ((M * kMtFin * 8 + 31) & ~31)) {
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < kTopK; ++j) acc = fmaf(vj[j], wtf[ft * kTopK + j], acc);
    const __nv_bfloat16 b16 = __float2bfloat16(acc);
    uint32_t v = *reinterpret_cast<const uint16_t*>(&b16);
    v |= __shfl_down_sync(0xffffffffu, v, 1) << 16;
    const uint32_t w1 = __shfl_down_sync(0xffffffffu, v, 2);
    const uint32_t w2 = __shfl_down_sync(0xffffffffu, v, 4);
    const uint32_t w3 = __shfl_down_sync(0xffffffffu, v, 6);
    if (fin_ok && fe == 0 && a.dbg != 21)
      st_mb16(a.lat_st + par * kMtLatBuf + static_cast<unsigned long long>((ft * kTp + rank) * kHidden + 8 * ff) * 2,
              no_neg_zero4(make_uint4(v, w1, w2, w3)), a.mc);
  }
  // (3) shared reduce-scatter: this CTA's fragments of the shared partial -> owner's RS mailbox
  if (sh_ok) {
    const int d = scol / kUpShard, col = scol - d * kUpShard;
    st_mb16(a.rs_peer[d] + par * kMtRsBuf + static_cast<unsigned long long>((st_ * kTp + rank) * kUpShard + col) * 2,
            no_neg_zero4(shv4), 0);
  }
  MT_MARK(2)
  // (4) poll the latent fragments f = q + 8 k of every token from all 16 sources (no re-arm here)
  const __nv_bfloat16* lat = a.lat_mb + par * (kMtLatBuf / 2);
  {
    constexpr int kPer = kTailMaxM * kMtPoll * kTp / kMtThreads;  // 14 items per thread at M = 4
    uint4 v[kPer];
    auto addr = [&](int i) {
      const int src = i & 15, tk = i >> 4, t = tk / kMtPoll, f = q + kMtCluster * (tk % kMtPoll);
      return lat + (t * kTp + src) * kHidden + 8 * f;
    };
    const int n = M * kMtPoll * kTp;
#pragma unroll
    for (int k = 0; k < kPer; ++k)
      if (tid + kMtThreads * k < n) v[k] = ld_volatile16(addr(tid + kMtThreads * k));
    MT_MARK(7)
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
      const int i = tid + kMtThreads * k;
      if (i < n) {
        while (lamport_dirty(v[k])) {
          if (kPollBackoffNs) __nanosleep(kPollBackoffNs);
          v[k] = ld_volatile16(addr(i));
        }
        pol[i] = v[k];
      }
    }
    MT_MARK(8)
  }
  // (5) this CTA's shared RS fragment (columns 8g..8g+7 = its up-proj rows), one reader: re-arm
  if (tid < M * kTp) {
    const int t = tid >> 4, src = tid & 15;
    const __nv_bfloat16* p = a.rs_mb + par * (kMtRsBuf / 2) + (t * kTp + src) * kUpShard + 8 * g;
    uint4 v = ld_volatile16(p);
    while (lamport_dirty(v)) {
      if (kPollBackoffNs) __nanosleep(kPollBackoffNs);
      v = ld_volatile16(p);
    }
    shp[tid] = v;
    st_sentinel16(p);
  }
  __syncthreads();
  MT_MARK(3)
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");  // peers are running (phase 0)
  MT_MARK(9)
  // (6) rank sums (fixed order 0..15, fp32) -> bf16 -> every cluster CTA's smem; bf16 squares
  for (int i = tid; i < M * kMtPoll * 8; i += kMtThreads) {  // warp-uniform token (448 = 14 warps)
    const int e = i & 7, tk = i >> 3, t = tk / kMtPoll, f = q + kMtCluster * (tk % kMtPoll);
    float acc = 0.f;
#pragma unroll 4
    for (int src = 0; src < kTp; ++src) {
      const uint32_t w = reinterpret_cast<const uint32_t*>(pol + tk * kTp + src)[e >> 1];
      acc += (e & 1) ? bf16_hi(w) : bf16_lo(w);
    }
    const __nv_bfloat16 nb = __float2bfloat16(acc);
    float sq = __bfloat162float(__hmul(nb, nb));
    uint32_t v = *reinterpret_cast<const uint16_t*>(&nb);
    v |= __shfl_down_sync(0xffffffffu, v, 1) << 16;
    const uint32_t w1 = __shfl_down_sync(0xffffffffu, v, 2);
    const uint32_t w2 = __shfl_down_sync(0xffffffffu, v, 4);
    const uint32_t w3 = __shfl_down_sync(0xffffffffu, v, 6);
    if (e == 0) {
      const uint4 frag = make_uint4(v, w1, w2, w3);
      const uint32_t la = smem_u32(xs + t * kHidden + 8 * f);
#pragma unroll
      for (int dst = 0; dst < kMtCluster; ++dst) st_cluster16(la, dst, frag);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
    if (lane == 0) spart[t * 14 + (i % (kMtPoll * 8)) / 32] = sq;
  }
  if (tid < M * 8) {  // shared shard values of this CTA's rows (rank order sum)
    const int t = tid >> 3, e = tid & 7;
    float acc = 0.f;
    for (int src = 0; src < kTp; ++src) {
      const uint32_t w = reinterpret_cast<const uint32_t*>(shp + t * kTp + src)[e >> 1];
      acc += (e & 1) ? bf16_hi(w) : bf16_lo(w);
    }
    shs[tid] = __bfloat162float(__float2bfloat16(acc));
  }
  MT_MARK(10)
  __syncthreads();
  MT_MARK(11)
  if (tid < M) {
    float s2 = 0.f;
    for (int w = 0; w < 14; ++w) s2 += spart[tid * 14 + w];
    const uint32_t la = smem_u32(ssp + q * 4 + tid);
    for (int dst = 0; dst < kMtCluster; ++dst) st_cluster_f32(la, dst, s2);
  }
  MT_MARK(12)
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  MT_MARK(13)
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
  MT_MARK(4)
  // (7) RMSNorm
  if (tid < M) {
    float s2 = 0.f;
    for (int c = 0; c < kMtCluster; ++c) s2 += ssp[c * 4 + tid];
    float r;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(s2 / static_cast<float>(kHidden) + a.eps));
    inv[tid] = r;
  }
  mbar_wait(smem_u32(bar), 0);  // up_proj rows + gamma landed
  MT_MARK(14)
  __syncthreads();
  for (int v8 = tid; v8 < M * (kHidden / 8); v8 += kMtThreads) {
    const int t = v8 / (kHidden / 8), c8 = v8 - t * (kHidden / 8);
    uint4* px = reinterpret_cast<uint4*>(xs) + v8;
    const uint4 xv = *px, gv = gsm[c8];
    const float r = inv[t];
    const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w}, gw[4] = {gv.x, gv.y, gv.z, gv.w};
    uint32_t ow[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const __nv_bfloat162 r2 = __floats2bfloat162_rn(bf16_lo(xw[k]) * r * bf16_lo(gw[k]),
                                                      bf16_hi(xw[k]) * r * bf16_hi(gw[k]));
      ow[k] = *reinterpret_cast<const uint32_t*>(&r2);
    }
    *px = make_uint4(ow[0], ow[1], ow[2], ow[3]);
  }
  __syncthreads();
  MT_MARK(5)
  // (8) up-proj GEMV: warp r = row 8g + r; lane: 16-byte K chunks lane + 32 z (14 per lane)
  {
    const uint4* wr = reinterpret_cast<const uint4*>(wsm + warp * kUpRowBytes);
    float acc[kTailMaxM][2] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
#pragma unroll 2
    for (int z = 0; z < kHidden / 8 / 32; ++z) {
      const int c8 = lane + 32 * z;
      const uint4 wv = wr[c8];
      const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
      for (int t = 0; t < kTailMaxM; ++t) {
        if (t < M) {
          const uint4 xv = reinterpret_cast<const uint4*>(xs)[t * (kHidden / 8) + c8];
          const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            acc[t][0] = fmaf(bf16_lo(ww[k]), bf16_lo(xw[k]), acc[t][0]);
            acc[t][1] = fmaf(bf16_hi(ww[k]), bf16_hi(xw[k]), acc[t][1]);
          }
        }
      }
    }
#pragma unroll
    for (int t = 0; t < kTailMaxM; ++t) {
      if (t < M) {
        const float tot = warp_sum(acc[t][0] + acc[t][1]);
        if (lane == 0) {
          const float gvv = __bfloat162float(__float2bfloat16(tot));
          outv[t * 8 + warp] = __float2bfloat16(gvv + shs[t * 8 + warp]);
        }
      }
    }
  }
  MT_MARK(15)
  __syncthreads();
  // (9) publish 8 rows x M tokens into the up-proj mailbox (16 bytes per token)
  if (tid < M) {
    const uint4 v = no_neg_zero4(reinterpret_cast<const uint4*>(outv)[tid]);
    st_mb16(a.up_st + static_cast<unsigned long long>(tid * (2 * kHidden) + rank * kUpShard + 8 * g) * 2, v, a.mc);
  }
  // (10) re-arm this CTA's share of the OTHER latent buffer: it was read by the previous call (finished
  //      before this kernel's griddepcontrol.wait) and the next call publishes into it (after this kernel
  //      completes), so remote writes into it are ordered after these stores.
  {
    uint4* base = reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(a.lat_mb) + (par ^ 1) * kMtLatBuf);
    constexpr int kChunk = static_cast<int>(kMtLatBuf / 16 / kMtCtas);  // 512 uint4 per CTA
    for (int i = tid; i < kChunk; i += kMtThreads) st_sentinel16(base + g * kChunk + i);
  }
  if (tid == 0) asm volatile("red.relaxed.gpu.global.add.u64 [%0], 1;" ::"l"(a.counter) : "memory");
  MT_MARK(6)
#undef MT_MARK
}
}  // namespace

// k3moe.moe_tail: see moe_tail_kernel. gemm2 [M*16, 3584] bf16 (row t*16+j = slot j of token t),
// wts [M, 16] bf16, shared_out [M, 7168] bf16 (this rank's shared-expert partial), w_up [448, 3584]
// (this rank's rows of up_proj.weight), gamma [3584]; lat_mb / rs_mb: symmetric [2, 4, 16, 3584] /
// [2, 4, 16, 448] bf16 (every word 0x80000000 initially); lat_st = multicast address of lat_mb (or
// its own address with multicast=false); rs_peers[d] = rank d's rs_mb; up_mb / up_st: vLLM's
// up-proj mailbox [1, >=M, 7168] and its multicast address; counter: int64 [>=1], zero-initialised
// once, used only by this op (must be the same sequence of calls on every rank).
void moe_tail(torch::Tensor gemm2, torch::Tensor wts, torch::Tensor shared_out, torch::Tensor w_up,
              torch::Tensor gamma, torch::Tensor lat_mb, int64_t lat_st, torch::Tensor rs_mb,
              std::vector<int64_t> rs_peers, torch::Tensor up_mb, int64_t up_st, torch::Tensor counter,
              double eps, int64_t rank, bool multicast, std::optional<torch::Tensor> trace) {
  const int M = wts.size(0);
  TORCH_CHECK(M >= 1 && M <= kTailMaxM, "moe_tail supports 1..4 tokens");
  auto bf = [](const torch::Tensor& t) { return t.scalar_type() == at::kBFloat16 && t.is_contiguous() && t.is_cuda(); };
  TORCH_CHECK(bf(gemm2) && gemm2.dim() == 2 && gemm2.size(0) == M * kTopK && gemm2.size(1) == kHidden);
  TORCH_CHECK(bf(wts) && wts.numel() == M * kTopK);
  TORCH_CHECK(bf(shared_out) && shared_out.numel() == M * 2 * kHidden);
  TORCH_CHECK(bf(w_up) && w_up.size(0) == kUpShard && w_up.size(1) == kHidden && w_up.data_ptr() != nullptr &&
              reinterpret_cast<uintptr_t>(w_up.data_ptr()) % 16 == 0);
  TORCH_CHECK(bf(gamma) && gamma.numel() == kHidden && reinterpret_cast<uintptr_t>(gamma.data_ptr()) % 16 == 0);
  TORCH_CHECK(bf(lat_mb) && lat_mb.numel() == 2L * 4 * kTp * kHidden, "lat_mb: bf16 [2, 4, 16, 3584]");
  TORCH_CHECK(bf(rs_mb) && rs_mb.numel() == 2L * 4 * kTp * kUpShard, "rs_mb: bf16 [2, 4, 16, 448]");
  TORCH_CHECK(bf(up_mb) && up_mb.size(-1) == 2 * kHidden && up_mb.numel() >= static_cast<long>(M) * 2 * kHidden);
  TORCH_CHECK(counter.scalar_type() == at::kLong && counter.is_cuda() && counter.numel() >= 1);
  TORCH_CHECK(static_cast<int>(rs_peers.size()) == kTp && rank >= 0 && rank < kTp);
  TORCH_CHECK(lat_st != 0 && lat_st % 16 == 0 && up_st != 0 && up_st % 16 == 0);
  MtArgs a{};
  a.gemm2 = reinterpret_cast<const __nv_bfloat16*>(gemm2.data_ptr());
  a.wts = reinterpret_cast<const __nv_bfloat16*>(wts.data_ptr());
  a.sh = reinterpret_cast<const __nv_bfloat16*>(shared_out.data_ptr());
  a.w_up = reinterpret_cast<const __nv_bfloat16*>(w_up.data_ptr());
  a.gamma = reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr());
  a.lat_mb = reinterpret_cast<__nv_bfloat16*>(lat_mb.data_ptr());
  a.lat_st = static_cast<unsigned long long>(lat_st);
  a.rs_mb = reinterpret_cast<__nv_bfloat16*>(rs_mb.data_ptr());
  for (int d = 0; d < kTp; ++d) {
    TORCH_CHECK(rs_peers[d] != 0 && rs_peers[d] % 16 == 0);
    a.rs_peer[d] = static_cast<unsigned long long>(rs_peers[d]);
  }
  a.up_st = static_cast<unsigned long long>(up_st);
  a.counter = reinterpret_cast<unsigned long long*>(counter.data_ptr());
  a.trace = trace ? reinterpret_cast<unsigned long long*>(trace->data_ptr()) : nullptr;
  a.eps = static_cast<float>(eps);
  a.rank = static_cast<int>(rank);
  a.mc = multicast ? 1 : 0;
  a.M = M;
  static const int dbg = getenv("K3TAIL_DBG") ? atoi(getenv("K3TAIL_DBG")) : 0;
  a.dbg = dbg;
  const size_t smem = kMtW + kUpRowBytes + static_cast<size_t>(M) * kUpRowBytes +
                      static_cast<size_t>(M) * (kMtPoll + 1) * kTp * 16 + 1024;
  static bool init = false;
  if (!init) {
#ifndef K3_NO_TAIL_MODULE
    C10_CUDA_CHECK(cudaFuncSetAttribute(moe_tail_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(kMtW + kUpRowBytes + 4 * kUpRowBytes +
                                                         4 * (kMtPoll + 1) * kTp * 16 + 1024)));
#endif
    init = true;
  }
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(kMtCtas);
  cfg.blockDim = dim3(kMtThreads);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  attr[1].id = cudaLaunchAttributeClusterDimension;
  attr[1].val.clusterDim.x = kMtCluster;
  attr[1].val.clusterDim.y = 1;
  attr[1].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 2;
#ifndef K3_NO_TAIL_MODULE
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, moe_tail_kernel, a));
#endif
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
                       std::optional<torch::Tensor> latent_out) {
  const int M = ids_out.size(0);
  TORCH_CHECK(M >= 1 && M <= kBlockMaxM, "moe_block_lamport supports 1..4 tokens");
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

// moe_block_lamport + the whole latent-MoE tail (see TailArgs): instead of gemm2_out / shared_out
// the kernel publishes into the tail's mailboxes and writes this rank's [M, 448] slice of the final
// MoE output into the up-proj mailbox (columns rank*448..; vLLM's AdaptiveUpProjectionKernel mailbox).
//   lat_mb   local latent all-reduce mailbox, bf16 [>= M*16*3584] (symmetric; every 32-bit word
//            0x80000000 initially; re-armed by the kernel); lat_st = its NVLS multicast address
//            (multicast=true) or any address of the same layout for plain stores (tests)
//   rs_mb    local shared reduce-scatter mailbox, bf16 [>= M*16*448]; rs_peers[d] = address of rank
//            d's rs_mb (16 entries; symm_mem buffer_ptrs)
//   up_mb    local up-proj mailbox [1, >= M, 7168] bf16; up_st = its multicast (or plain) address
//   w_up     [448, 3584] bf16 this rank's rows of up_proj.weight; gamma [3584] bf16 (RMSNorm weight)
//   tail_ws  scratch, >= M*(8*7168 + 896 + 32*G) bytes
void moe_block_tail(torch::Tensor mailbox, torch::Tensor scores, torch::Tensor w13, torch::Tensor w13_scale,
                    torch::Tensor w2, torch::Tensor w2_scale, torch::Tensor workspace, torch::Tensor barrier,
                    torch::Tensor ids_out, torch::Tensor wts_out, torch::Tensor h_shared, torch::Tensor w_sd,
                    torch::Tensor lat_mb, int64_t lat_st, torch::Tensor rs_mb, std::vector<int64_t> rs_peers,
                    torch::Tensor up_mb, int64_t up_st, torch::Tensor w_up, torch::Tensor gamma,
                    torch::Tensor tail_ws, double eps, int64_t rank, bool multicast, double beta,
                    double linear_beta, bool renormalize, double routed_scaling_factor, int64_t grid,
                    double shared_beta, double shared_linear_beta, std::optional<torch::Tensor> trace) {
  const int M = ids_out.size(0);
  TORCH_CHECK(M >= 1 && M <= kTailMaxM, "moe_block_tail supports 1..4 tokens");
  TORCH_CHECK(ids_out.dim() == 2 && ids_out.size(1) == kTopK && ids_out.scalar_type() == at::kInt &&
              ids_out.is_contiguous());
  TORCH_CHECK(wts_out.numel() == M * kTopK && wts_out.scalar_type() == at::kBFloat16 && wts_out.is_contiguous());
  TORCH_CHECK(scores.numel() * scores.element_size() >= 8 * M * 896 * 8 && scores.is_contiguous(),
              "scores: route_shared scores-only output [8 replicas, M, 896, 2] fp32");
  TORCH_CHECK(h_shared.numel() == M * 2 * kSdK && h_shared.scalar_type() == at::kBFloat16 && h_shared.is_contiguous());
  TORCH_CHECK(w_sd.size(0) == kSdRows && w_sd.size(1) == kSdK && w_sd.scalar_type() == at::kBFloat16 &&
              w_sd.is_contiguous());
  auto bf = [](const torch::Tensor& t) { return t.scalar_type() == at::kBFloat16 && t.is_contiguous() && t.is_cuda(); };
  TORCH_CHECK(bf(lat_mb) && lat_mb.numel() >= static_cast<long>(M) * kTp * kHidden, "lat_mb: bf16 [>= M*16*3584]");
  TORCH_CHECK(bf(rs_mb) && rs_mb.numel() >= static_cast<long>(M) * kTp * kUpShard, "rs_mb: bf16 [>= M*16*448]");
  TORCH_CHECK(bf(up_mb) && up_mb.size(-1) == 2 * kHidden && up_mb.numel() >= static_cast<long>(M) * 2 * kHidden,
              "up_mb: bf16 [1, >= M, 7168]");
  TORCH_CHECK(bf(w_up) && w_up.size(0) == kUpShard && w_up.size(1) == kHidden, "w_up: bf16 [448, 3584]");
  TORCH_CHECK(bf(gamma) && gamma.numel() == kHidden, "gamma: bf16 [3584]");
  TORCH_CHECK(static_cast<int>(rs_peers.size()) == kTp, "rs_peers: 16 addresses");
  TORCH_CHECK(rank >= 0 && rank < kTp);
  for (auto p : rs_peers) TORCH_CHECK(p != 0 && p % 16 == 0);
  TORCH_CHECK(lat_st % 16 == 0 && up_st % 16 == 0 && lat_st != 0 && up_st != 0);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(w_up.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(gamma.data_ptr()) % 16 == 0);
  static int sms = 0;
  if (sms == 0) cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, ids_out.get_device());
  const int G = grid > 0 ? std::min(static_cast<int>(grid), sms) : sms;
  TORCH_CHECK(tail_ws.is_cuda() && tail_ws.is_contiguous() &&
              tail_ws.numel() * tail_ws.element_size() >=
                  static_cast<long>(M) * (kGatherRep * 2 * kHidden + 2 * kUpShard + kGatherRep * 4 * G) &&
              reinterpret_cast<uintptr_t>(tail_ws.data_ptr()) % 16 == 0, "tail_ws too small");
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
  ta.lat_st = static_cast<unsigned long long>(lat_st);
  ta.lat_mb = reinterpret_cast<const __nv_bfloat16*>(lat_mb.data_ptr());
  for (int d = 0; d < kTp; ++d) ta.rs_peer[d] = static_cast<unsigned long long>(rs_peers[d]);
  ta.rs_mb = reinterpret_cast<const __nv_bfloat16*>(rs_mb.data_ptr());
  ta.up_st = static_cast<unsigned long long>(up_st);
  ta.w_up = reinterpret_cast<const __nv_bfloat16*>(w_up.data_ptr());
  ta.gamma = reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr());
  uint8_t* ws = reinterpret_cast<uint8_t*>(tail_ws.data_ptr());
  ta.g_lat = reinterpret_cast<__nv_bfloat16*>(ws);
  ta.g_shs = reinterpret_cast<__nv_bfloat16*>(ws + kGatherRep * M * 2 * kHidden);
  ta.g_ss = reinterpret_cast<float*>(ws + kGatherRep * M * 2 * kHidden + M * 2 * kUpShard);
  ta.eps = static_cast<float>(eps);
  ta.rank = static_cast<int>(rank);
  ta.mc = multicast ? 1 : 0;
  static const int dbg = getenv("K3TAIL_DBG") ? atoi(getenv("K3TAIL_DBG")) : 0;
  ta.dbg = dbg;
#ifndef K3_NO_TAIL_MODULE
  launch_fused_tail<false, true, true, true>(mailbox, ids_out, nullptr, w13, w13_scale, w2, w2_scale, workspace,
                                        barrier, nullptr, beta, linear_beta, trace, 0, ba, static_cast<int>(grid));
#endif
}

void topk_debug(torch::Tensor scores, torch::Tensor ids, torch::Tensor sc, torch::Tensor tr) {
  topk_debug_kernel<<<1, 32, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint2*>(scores.data_ptr()), ids.data_ptr<int>(), sc.data_ptr<float>(),
      reinterpret_cast<unsigned long long*>(tr.data_ptr()));
}

TORCH_LIBRARY(k3moe, m) {
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
        "float shared_linear_beta=25.0, Tensor(h!)? trace=None, Tensor(i!)? latent_out=None) -> ()");
  m.def("moe_block_tail(Tensor(a!) mailbox, Tensor scores, Tensor w13, Tensor w13_scale, Tensor w2, "
        "Tensor w2_scale, Tensor(b!) workspace, Tensor(c!) barrier, Tensor(d!) ids_out, Tensor(e!) wts_out, "
        "Tensor h_shared, Tensor w_sd, Tensor(f!) lat_mb, int lat_st, Tensor(g!) rs_mb, int[] rs_peers, "
        "Tensor(h!) up_mb, int up_st, Tensor w_up, Tensor gamma, Tensor(i!) tail_ws, float eps, int rank, "
        "bool multicast, float beta, float linear_beta, bool renormalize, float routed_scaling_factor, "
        "int grid=0, float shared_beta=4.0, float shared_linear_beta=25.0, Tensor(j!)? trace=None) -> ()");
  m.def("moe_tail(Tensor gemm2, Tensor wts, Tensor shared_out, Tensor w_up, Tensor gamma, Tensor(a!) lat_mb, "
        "int lat_st, Tensor(b!) rs_mb, int[] rs_peers, Tensor(c!) up_mb, int up_st, Tensor(d!) counter, "
        "float eps, int rank, bool multicast, Tensor(e!)? trace=None) -> ()");
  m.def("topk_debug(Tensor scores, Tensor(a!) ids, Tensor(b!) sc, Tensor(c!) tr) -> ()");
  m.def("moe_fused_unfinalized_lamport(Tensor(a!) mailbox, Tensor topk_ids, Tensor w13, Tensor w13_scale, "
        "Tensor w2, Tensor w2_scale, Tensor(b!) workspace, Tensor(c!) barrier, Tensor(d!) gemm2_out, "
        "float beta, float linear_beta, bool wait_prior=True, Tensor(e!)? trace=None) -> ()");
}
TORCH_LIBRARY_IMPL(k3moe, CUDA, m) {
  m.impl("moe_small", &moe_small);
  m.impl("moe_fused", &moe_fused);
  m.impl("moe_fused_unfinalized", &moe_fused_unfinalized);
  m.impl("moe_fused_unfinalized_lamport", &moe_fused_unfinalized_lamport);
  m.impl("route_shared", &route_shared);
  m.impl("moe_block_lamport", &moe_block_lamport);
  m.impl("moe_block_tail", &moe_block_tail);
  m.impl("moe_tail", &moe_tail);
  m.impl("topk_debug", &topk_debug);
}
