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

template <bool kFinal>
__global__ void __launch_bounds__(kFusedThreads, 1)
moe_fused_kernel(const __grid_constant__ CUtensorMap tm_w13s, const __grid_constant__ CUtensorMap tm_w2,
                 const __grid_constant__ CUtensorMap tm_w2s, const __nv_bfloat16* __restrict__ x,
                 const int* __restrict__ topk_ids, const float* __restrict__ topk_w,
                 const uint8_t* __restrict__ w13, __half* __restrict__ h,
                 unsigned long long* __restrict__ barrier, __nv_bfloat16* __restrict__ out, int M,
                 int nstages, float beta, float linear_beta, unsigned long long* __restrict__ trace) {
#define K3_MARK(slot)                                                                 \
  if (trace) {                                                                        \
    unsigned long long ts;                                                            \
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts));                           \
    trace[blockIdx.x * 32 + (slot)] = ts;                                             \
  }
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const FusedWork W = fused_work(M);
  // Shared memory map: x (bf16, later fp16 h) | red | acc2 | wsm | barriers | ring.
  const int x_bytes = M * kHidden * 2;
  uint8_t* xs = smem;
  const int max_slots = 2 * ((M * kPairsPerTok + gridDim.x - 1) / gridDim.x) + 16;
  const int max_rows2 = M * 8 * ((kOctets + gridDim.x - 1) / gridDim.x);
  float* red = reinterpret_cast<float*>(smem + x_bytes);
  float* acc2 = red + max_slots;
  float* wsm = acc2 + max_rows2;
  uint64_t* bars = reinterpret_cast<uint64_t*>(
      (reinterpret_cast<uintptr_t>(wsm + M * kTopK) + 15) & ~static_cast<uintptr_t>(15));
  uint8_t* stages = reinterpret_cast<uint8_t*>(
      (reinterpret_cast<uintptr_t>(bars + 1 + 2 * nstages) + 127) & ~static_cast<uintptr_t>(127));
  const uint32_t xbar = smem_u32(bars);
  auto full_bar = [&](int s) { return smem_u32(bars + 1 + s); };
  auto empty_bar = [&](int s) { return smem_u32(bars + 1 + nstages + s); };

  if (tid == 0) K3_MARK(0)
  if (tid == 0 && trace) trace[blockIdx.x * 32 + 16] = clock64();
  if (tid == 0) {
    mbar_init(xbar, kFinal ? 2 : 1);
    for (int s = 0; s < nstages; ++s) {
      mbar_init(full_bar(s), 1);
      mbar_init(empty_bar(s), kConsumerWarps);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  for (int i = tid; i < max_slots + max_rows2; i += kFusedThreads) red[i] = 0.f;
  __syncthreads();

  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (tid == 0) K3_MARK(1)

  const int total_seq = W.ngroups + W.nf2;
  if (warp == kConsumerWarps) {
    // ------------------------------ producer ------------------------------
    if (lane == 0) {
      mbar_arrive_expect_tx(xbar, x_bytes);
      tma_1d(smem_u32(xs), x, x_bytes, xbar);
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
    auto issue = [&](int k) {
      const int s = k % nstages;
      uint8_t* dst = stages + s * kStageBytes;
      const uint32_t bar = full_bar(s);
      if (k < W.ngroups) {
        if (lane == 0) {
          const int qs = k == 0 ? W.qa : ((W.qa >> 3) + k) << 3;
          const int n = min(W.qb, (qs & ~7) + 8) - qs;
          const int t = qs / kPairsPerTok, j = (qs / kInter) % kTopK, u = qs % kInter;
          const int b = u >> 3, r0 = u & 7;
          const int e = __ldcg(topk_ids + t * kTopK + j);
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
          const int e = __ldcg(topk_ids + t * kTopK + j);
          uint8_t* d = dst + ui * kF2Unit;
          tma_2d(smem_u32(d + j * 8 * 96), pm2, 0, e * kHidden + 8 * o, bar);
          tma_3d(smem_u32(d + kF2W + j * 256), pm2s, 128 * (o & 3), 0, e * 28 + (o >> 4), bar);
        }
      }
    };
    const int first = min(total_seq, nstages);
    const int first1 = min(W.ngroups, first);
    for (int k = 0; k < first1; ++k) issue(k);
    if (lane == 0) K3_MARK(8)
#ifdef FC2_WAIT
    if (first1 > 0) mbar_wait(full_bar((first1 - 1) % nstages), 0);
#endif
    for (int k = first1; k < first; ++k) issue(k);
    for (int k = first; k < total_seq; ++k) {
      mbar_wait(empty_bar(k % nstages), ((k / nstages) - 1) & 1);
      issue(k);
    }
    if (lane == 0) K3_MARK(9)
    if (trace && lane == 0 && total_seq <= nstages) {
      for (int k = 0; k < total_seq; ++k) {
        mbar_wait(full_bar(k), 0);
        if (k == W.ngroups - 1) K3_MARK(12)
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
    const bool fc1_warp = warp < kFc1Warps;
    const int c = (fc1_warp ? warp : 0) * 8 + (lane & 7);  // K chunk (32 elements)
    const int rs = lane >> 3;             // row sub-index
    const int i8 = lane & 7;
    __half2 xr[16];
    int cur_t = -1;
    for (int k = 0; k < W.ngroups; ++k) {
      const int s = k % nstages;
      const int qs = k == 0 ? W.qa : ((W.qa >> 3) + k) << 3;
      const int n = min(W.qb, (qs & ~7) + 8) - qs;
      const int t = qs / kPairsPerTok, u = qs % kInter;
      const int b = u >> 3, r0 = u & 7, rg = (b & 7) >> 1;
      if (!fc1_warp) {
        mbar_wait(full_bar(s), (k / nstages) & 1);
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
      mbar_wait(full_bar(s), (k / nstages) & 1);
      if (k == 0 && tid == 0) K3_MARK(3)
      const uint8_t* sw = stages + s * kStageBytes;
      const uint8_t* ssc = sw + kStageW + (c >> 2) * 256 + rg * 4 + (c & 3);
      const int sigma0 = 2 * (qs - W.qa);
      const int s0 = (rs - sigma0) & 3;
      float v[4];
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const int sl = s0 + 4 * m;
        v[m] = 0.f;
        if (sl < 2 * n) {
          const uint4 wv = *reinterpret_cast<const uint4*>(sw + sl * kW13RowBytes + c * 16);
          const int rib = sl < n ? r0 + sl : 8 + r0 + sl - n;
#ifdef NO_MATH
          v[m] = __int_as_float((wv.x ^ wv.w) & 0x3fffffff) + __half2float(xr[m].x) + ssc[rib * 16];
#else
          v[m] = ue8m0(ssc[rib * 16]) * dot32_h(wv, xr);
#endif
        }
      }
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
    if (tid == 0) {
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
    asm volatile("griddepcontrol.launch_dependents;");
    // Stage h (all tokens) into shared memory (over x).
    __half* hs = reinterpret_cast<__half*>(xs);
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
      const int k = W.ngroups + f;
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

template <bool kFinal>
void launch_fused(const torch::Tensor& x, const torch::Tensor& topk_ids, const float* topk_w,
                  const torch::Tensor& w13, const torch::Tensor& w13_scale, const torch::Tensor& w2,
                  const torch::Tensor& w2_scale, const torch::Tensor& workspace, const torch::Tensor& barrier,
                  __nv_bfloat16* out, double beta, double linear_beta,
                  const std::optional<torch::Tensor>& trace) {
  const int M = x.size(0);
  TORCH_CHECK(M >= 1 && M <= kMaxFusedM, "moe_fused supports 1..8 tokens");
  TORCH_CHECK(x.size(1) == kHidden && x.is_contiguous() && x.scalar_type() == at::kBFloat16);
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
    cudaFuncSetAttribute(moe_fused_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448);
    cudaFuncSetAttribute(moe_fused_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448);
  }
  const int G = sms;
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
  const int fixed = M * kHidden * 2 + (max_slots + max_rows2 + M * kTopK) * 4 + 16 +
                    (1 + 2 * 8) * 8 + 128;
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
      &cfg, moe_fused_kernel<kFinal>, tm_w13s, tm_w2, tm_w2s,
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), static_cast<const int*>(topk_ids.data_ptr<int>()),
      topk_w, reinterpret_cast<const uint8_t*>(w13.data_ptr()), reinterpret_cast<__half*>(workspace.data_ptr()),
      reinterpret_cast<unsigned long long*>(barrier.data_ptr()), out, M, nstages, static_cast<float>(beta),
      static_cast<float>(linear_beta),
      trace ? reinterpret_cast<unsigned long long*>(trace->data_ptr()) : nullptr));
}

}  // namespace

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
}
TORCH_LIBRARY_IMPL(k3moe, CUDA, m) {
  m.impl("moe_small", &moe_small);
  m.impl("moe_fused", &moe_fused);
  m.impl("moe_fused_unfinalized", &moe_fused_unfinalized);
}
