// moe8: fused routed-MoE (x -> MXFP8 quant, FC1 -> SiTU -> FC2, unfinalized) for Kimi K3 decode,
// M <= 8 tokens, on sm_100a tensor cores (tcgen05.mma kind::mxf8f6f4: MXFP4 weights x MXFP8 activations).
// One kernel launch; routing (top-k ids) is an input. See the final report of agents/moe8 for numbers.
//
// Weights are read in place in vLLM's TRT-LLM MXFP4 layout (no re-layout; see agents/moefused/moe_small.cu):
//   w13 [E, 512, 1792] u8: physical row p of an expert holds logical row
//       L = 32*(p/32) + 4*(p%8) + (p%32)/8 of the interleaved [up_0, gate_0, up_1, ...] matrix;
//       physical rows 384..511 are padding and are never read.
//   w13_scale [E, 512, 112] UE8M0, 128x4-swizzled: the 512 B chunk for (128-row tile r, 4-column
//       group c) sits at ((r*28 + c) * 512) and is exactly the smem image that
//       tcgen05.cp.32x128b.warpx4 needs for the A scale factors of an M = 128 MMA.
//   w2 [E, 3584, 128] u8 (rows permuted the same way; K = 192 real of 256), w2_scale [E, 3584, 8].
//
// Swap-AB: D[128 weight rows x 8 tokens] (fp32, TMEM) = A (weights, e2m1) x B (tokens, e4m3), K-major,
// 128B swizzle. A is TMA-loaded with CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B (fp4 unpacked to one
// byte/element in smem, as kind::mxf8f6f4 requires; complete_tx counts the packed global bytes), one
// 3D box {128 elements, 128 rows, 2 K-chunks} = 16 KB per stage (FC2's second chunk starts at k = 64
// so the K padding is never read). B (x or h in e4m3) is written into smem by threads. A stage's
// 2 scale copies + 8 MMAs + commits are issued from one asm block (mma_issue.cuh): a 128x8x32
// block-scaled MMA then costs ~39 cycles and the kernel is bound by the TMA weight stream.
//
// Schedule (static, one persistent CTA per SM, G CTAs; experts deduplicated across tokens):
//   Experts split in two groups when D > 40: FC1(A), FC1(B), FC2(A), FC2(B) per CTA, so group A's
//   h is ready long before FC2(A) and FC2(A) hides the latency of group B's h hand-off.
//   FC1: 42*D_g stages (experts x 3 row tiles x 14 K-stages of 256) in G contiguous ranges
//        (stream-K). A tile split across CTAs is reduced by its owner (the CTA holding its first
//        K-stage); the other CTAs write fp32 partials that the owner polls (0xFFFFFFFF sentinels,
//        re-armed by the owner).
//   FC2: 28*D_g (expert, 128-row output tile) units in G contiguous ranges.
//   h hand-off: owners write SiTU(h) quantized to MXFP8 directly as the FC2 B-operand smem image into
//        a global buffer pre-filled with 0xFF; FC2 consumers poll the data itself. h buffers are
//        double-buffered by call parity (per-CTA call counters) and the idle parity is re-armed
//        during the call: no grid-wide synchronisation, no counter resets.
//   x -> MXFP8 is done in every CTA (loads issued during the dedupe); xready[ks] barriers release
//   the MMA warp per K-stage.
//
// Warp roles (384 threads): w0 TMA producer, w1 MMA issuer, w2 h loader, w3 TMEM owner + h re-arm,
// w4..7 and w8..11 two epilogue groups (alternate accumulators; TMEM lane quadrant = warp % 4).
// Warps 2..11 quantize x first.
#include <algorithm>
#include <vector>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#include "tc_utils.cuh"
#include "mma_issue.cuh"

namespace moe8 {

constexpr int kHidden = 3584;
constexpr int kTopK = 16;
constexpr int kMaxM = 8;
constexpr int kMaxPairs = kMaxM * kTopK;  // 128
constexpr int kW13Rows = 512;
constexpr int kW13RowBytes = 1792;
constexpr int kW13ScaleBytes = kW13Rows * 112;  // 57344 per expert
constexpr int kW2ScaleBytes = kHidden * 8;      // 28672 per expert
constexpr int kFc1Stages = 14;                  // K = 3584 = 14 x 256
constexpr int kFc1Tiles = 3;                    // 384 real rows = 3 x 128
constexpr int kFc2Tiles = 28;                   // 3584 = 28 x 128
constexpr int kMaxE = 4096;

#ifndef MOE8_SLOTS
#define MOE8_SLOTS 5
#endif
constexpr int kSlots = MOE8_SLOTS;
#ifndef MOE8_ACC
#define MOE8_ACC 16
#endif
constexpr int kAcc = MOE8_ACC;
constexpr int kThreads = 384;  // 12 warps; two epilogue groups (warps 4..7, 8..11)
constexpr int kMaxG = 160;

// smem layout (offsets from a 1024-aligned base)
constexpr int kOffA = 0;                             // kSlots x 32 KB weights (unpacked fp4)
constexpr int kOffSFA = kOffA + kSlots * 32768;      // kSlots x 1 KB weight scales
constexpr int kOffXq = kOffSFA + kSlots * 1024;      // 28 KB x (e4m3), B operand of FC1
constexpr int kOffHq = kOffXq + 28672;               // 2 x 2 KB h (e4m3), B operand of FC2
constexpr int kOffXsf = kOffHq + 2 * 2048;           // 28 x 128 (+384) SFB images of x
constexpr int kOffHsf = kOffXsf + 28 * 128 + 384;    // 2 x (2 x 128 + 384) SFB images of h
constexpr int kOffMisc = kOffHsf + 2 * 640;
constexpr int kSmemBytes = kOffMisc + 5120;
static_assert(kSmemBytes <= 232448, "smem");

// TMEM columns (all scale-factor bases are multiples of 4)
constexpr uint32_t kTmemCols = 512;
constexpr uint32_t kColAcc = 0;                        // kAcc x 8
constexpr uint32_t kColSFA = 128;                      // kSlots x 8
constexpr uint32_t kColSFBh = kColSFA + kSlots * 8;    // 2 x 8
constexpr uint32_t kColSFBx = 256;                     // 28 x 4
static_assert(kAcc * 8 <= 128 && kColSFBh + 16 <= kColSFBx, "tmem");

// global workspace layout (bytes); everything starts as 0xFF.
constexpr int kHImg = 2048 + 128;                          // per expert: B image + compact scales
constexpr long kWsH = 0;                                   // [2 parities][128][kHImg]
constexpr long kWsP = kWsH + 2L * kMaxPairs * kHImg;           // [2 groups][kMaxG][8][128] fp32 partials
constexpr long kWsCtr = kWsP + 2L * kMaxG * 8 * 128 * 4;       // [kMaxG] int per-CTA call counters
constexpr long kWsBytes = kWsCtr + kMaxG * 4;

enum : int { kFC1 = 0, kFC2 = 1, kEnd = 2 };

struct Meta {
  int type, a, b, c;  // FC1: a = tile T, b = ks, c = flags (1 first, 2 last); FC2: a = u, b = mt
};

struct Misc {
  uint64_t full[kSlots], empty[kSlots];
  uint64_t accfull[kAcc], accempty[kAcc];
  uint64_t hfull[2], hempty[2];
  uint64_t xready[kFc1Stages];
  Meta meta[kSlots];
  Meta accmeta[kAcc];
  uint32_t tmem_base;
  int D, parity;
  int wcnt[4];
  float amax[2][4][8];
  int expert[kMaxPairs];
  int uof[kMaxPairs];
  alignas(16) signed char tokslot[kMaxPairs][kMaxM];
  alignas(16) uint8_t hst[2][4][8][16];
};
static_assert(sizeof(Misc) <= 5120, "misc too big");

struct Params {
  const __nv_bfloat16* x;
  const int* topk_ids;
  const uint8_t* w13s;
  const uint8_t* w2s;
  const uint8_t* w2;
  uint8_t* ws;
  __nv_bfloat16* out;
  int M, E;
  float beta_lb, inv_beta, inv_lb;  // SiTU = beta*lb * tanh(g/beta) * sigmoid(g) * tanh(u/lb)
  unsigned long long* trace;
};

// ---------------------------------------------------------------- small helpers
__device__ __forceinline__ float tanh_fast(float x) {
  float y;
  asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float sigmoid_fast(float x) {
  float e, r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(-1.4426950408889634f * x));
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(1.f + e));
  return r;
}
// MXFP8 block scale exponent: smallest s with 2^(s-127) * 448 >= amax (= ceil(log2(amax/448)) + 127).
__device__ __forceinline__ int mx_exp(float amax) {
  const uint32_t b = __float_as_uint(amax);
  const int e = static_cast<int>(b >> 23);  // amax >= 0
  if (e == 0) return 0;
  const int s = e - 8 + ((b & 0x7fffff) > 0x600000 ? 1 : 0);
  return min(max(s, 0), 254);
}
__device__ __forceinline__ float mx_rescale(int sb) {  // 2^(127 - sb)
  const int e = 254 - sb;
  return e >= 1 ? __uint_as_float(static_cast<uint32_t>(e) << 23) : 0.f;
}
__device__ __forceinline__ uint32_t e4m3x4(float a, float b, float c, float d) {
  const uint32_t lo = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
  const uint32_t hi = __nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
  return lo | (hi << 16);
}
__device__ __forceinline__ bool has_ff_byte(uint32_t w) {  // any byte == 0xFF
  const uint32_t x = ~w;
  return ((x - 0x01010101u) & ~x & 0x80808080u) != 0;
}
__device__ __forceinline__ uint4 ld_volatile_v4(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}
__device__ __forceinline__ uint32_t ld_volatile_u32(const void* p) {
  uint32_t v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ uint64_t desc_sf(uint32_t saddr) { return tc::smem_desc(saddr, 0, 128, 0); }

#ifdef MOE8_TRACE
constexpr bool kTraceOn = true;
#else
constexpr bool kTraceOn = false;
#endif
// Trace points (%globaltimer + clock64 per event) only exist in -DMOE8_TRACE builds (trace.py).
__device__ __forceinline__ void trace_ev(const Params& p, int ev) {
#ifndef MOE8_TRACE
  return;
#endif
  if (p.trace != nullptr) {
    p.trace[blockIdx.x * 128 + ev] = tc::globaltimer();
    p.trace[blockIdx.x * 128 + 32 + ev] = clock64();
  }
}
#ifdef MOE8_PROF
#define PCLK() clock64()
#else
#define PCLK() 0ll
#endif
__device__ __forceinline__ void trace_ctr(const Params& p, int i, long long v) {
#ifndef MOE8_TRACE
  return;
#endif
  if (p.trace != nullptr) p.trace[blockIdx.x * 128 + 64 + i] = v;
}

__device__ __forceinline__ int split_lo(int c, int total, int G) {
  return static_cast<int>(static_cast<unsigned>(c * total) / static_cast<unsigned>(G));  // < 2^31
}

__global__ void __launch_bounds__(kThreads, 1)
moe8_kernel(const __grid_constant__ CUtensorMap tmA1, const __grid_constant__ CUtensorMap tmA2, const Params p) {
  // No static smem in this kernel, so the dynamic smem window starts 1024-aligned; using the array
  // directly (no integer round trip) keeps every access in the shared state space (LDS/STS).
  extern __shared__ __align__(1024) uint8_t sm[];
  Misc& ms = *reinterpret_cast<Misc*>(sm + kOffMisc);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int M = p.M;
  const int G = gridDim.x, cta = blockIdx.x;

  if (threadIdx.x == 0) {
    if ((tc::su32(sm) & 1023) != 0) asm volatile("trap;");
    trace_ev(p, 0);
    for (int s = 0; s < kSlots; ++s) {
      tc::mbar_init(&ms.full[s], 1);
      tc::mbar_init(&ms.empty[s], 1);
    }
    for (int a = 0; a < kAcc; ++a) {
      tc::mbar_init(&ms.accfull[a], 2);
      tc::mbar_init(&ms.accempty[a], 4);
    }
    for (int b = 0; b < 2; ++b) {
      tc::mbar_init(&ms.hfull[b], 1);
      tc::mbar_init(&ms.hempty[b], 1);
    }
    for (int k = 0; k < kFc1Stages; ++k) tc::mbar_init(&ms.xready[k], 8 * p.M);
    tc::fence_mbar_init();
    tc::prefetch_tmap(&tmA1);
    tc::prefetch_tmap(&tmA2);
  }
  if (warp == 3) tc::tmem_alloc(&ms.tmem_base, kTmemCols);
  int* table = reinterpret_cast<int*>(sm + kOffA);  // dedupe scratch (ring is still unused)
  for (int e = threadIdx.x; e < p.E; e += kThreads) table[e] = 0x7fffffff;
  // rows t >= M of the x B image and of its scale images stay zero
  for (int i = threadIdx.x; i < (kMaxM - M) * 224; i += kThreads) {
    const int t = M + i / 224, kc = i % 224;
    *reinterpret_cast<uint4*>(sm + kOffXq + (kc / 8) * 1024 + t * 128 + (((kc % 8) ^ t) * 16)) = make_uint4(0, 0, 0, 0);
  }
  for (int i = threadIdx.x; i < 28 * 8; i += kThreads)
    *reinterpret_cast<uint4*>(sm + kOffXsf + (i >> 3) * 128 + (i & 7) * 16) = make_uint4(0, 0, 0, 0);
  tc::fence_proxy_async_smem();

  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (threadIdx.x == 0) trace_ev(p, 1);



  // ---- routing ids + per-CTA call parity (one round trip); dedupe experts in first-occurrence order
  const int npairs = M * kTopK;
  int myid = -1;
  if (threadIdx.x < kMaxPairs) {
    if (threadIdx.x < npairs) myid = p.topk_ids[threadIdx.x];
    *reinterpret_cast<uint2*>(ms.tokslot[threadIdx.x]) = make_uint2(~0u, ~0u);
  }
  int* ctr = reinterpret_cast<int*>(p.ws + kWsCtr);
  if (threadIdx.x == 160) ms.parity = ctr[cta];
  __syncthreads();
  // x does not depend on the routing: warps 2..11 issue their x loads once the routing ids are in (so the ids load is not queued behind them), overlapping the rest of the dedupe.
  // Thread xtid, round r: block j = xtid % 64 (token j/8, 32-block j%8) of K-stage 5r + xtid/64.
  const int xtid = static_cast<int>(threadIdx.x) - 64;
  uint4 xraw[3][4];
  if (warp >= 2) {
#pragma unroll
    for (int r = 0; r < 3; ++r) {
      const int ks = 5 * r + (xtid >> 6), j = xtid & 63, t = j >> 3, kbl = j & 7;
      const bool ok = ks < kFc1Stages && t < M;
      const uint4* src = reinterpret_cast<const uint4*>(p.x + t * kHidden + (ok ? ks : 0) * 256 + kbl * 32);
#pragma unroll
      for (int v = 0; v < 4; ++v) xraw[r][v] = ok ? src[v] : make_uint4(0, 0, 0, 0);
    }
  }
  if (threadIdx.x < npairs) atomicMin(&table[myid], static_cast<int>(threadIdx.x));
  __syncthreads();
  if (threadIdx.x < kMaxPairs) {
    const int i = threadIdx.x;
    const int firstk = i < npairs ? table[myid] : -1;
    const bool first = (i < npairs) && firstk == i;
    const unsigned bal = __ballot_sync(0xffffffffu, first);
    if (lane == 0) ms.wcnt[warp] = __popc(bal);
    asm volatile("bar.sync 1, 128;" ::: "memory");
    const int base = (warp > 0 ? ms.wcnt[0] : 0) + (warp > 1 ? ms.wcnt[1] : 0) + (warp > 2 ? ms.wcnt[2] : 0);
    const int rank = base + __popc(bal & ((1u << lane) - 1));
    if (first) {
      ms.uof[i] = rank;
      ms.expert[rank] = myid;
    }
    if (i == 0) ms.D = ms.wcnt[0] + ms.wcnt[1] + ms.wcnt[2] + ms.wcnt[3];
    asm volatile("bar.sync 1, 128;" ::: "memory");
    if (i < npairs) ms.tokslot[ms.uof[firstk]][i / kTopK] = static_cast<signed char>(i % kTopK);
  }
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;");
  if (threadIdx.x == 0) trace_ev(p, 2);
  const int D = ms.D;
  const int P = ms.parity & 1;
  // Two expert groups: FC1(A), FC1(B), FC2(A), FC2(B). Group A's h is ready long before FC2(A)
  // starts, and FC2(A) covers the latency of group B's FC1 -> h hand-off.
#ifndef MOE8_GROUP_MIN_D
#define MOE8_GROUP_MIN_D 40
#endif
  const int ub[3] = {0, D > MOE8_GROUP_MIN_D ? (D + 1) / 2 : D, D};
  int S1g[2], f1lo[2], f1hi[2], f2lo[2], f2hi[2];
#pragma unroll
  for (int g = 0; g < 2; ++g) {
    const int De = ub[g + 1] - ub[g];
    S1g[g] = kFc1Tiles * kFc1Stages * De;
    f1lo[g] = split_lo(cta, S1g[g], G);
    f1hi[g] = split_lo(cta + 1, S1g[g], G);
    f2lo[g] = kFc2Tiles * ub[g] + split_lo(cta, kFc2Tiles * De, G);
    f2hi[g] = kFc2Tiles * ub[g] + split_lo(cta + 1, kFc2Tiles * De, G);
  }
  uint8_t* hset = p.ws + kWsH + static_cast<long>(P) * kMaxPairs * kHImg;


  if (warp == 0) {
    // ================================================================ TMA producer
    if (lane == 0) {
      int slot = 0, phase = 0;
      long long wait_empty = 0;
      // This CTA's stage sequence: FC1 group 0, FC1 group 1, FC2 group 0, FC2 group 1.
      const int n1a = f1hi[0] - f1lo[0], n1b = f1hi[1] - f1lo[1], n2a = f2hi[0] - f2lo[0], n2b = f2hi[1] - f2lo[1];
      const int total = n1a + n1b + n2a + n2b;
      struct SInfo {
        Meta m;
        int row0, kc;           // TMA coordinates (weights)
        const uint8_t* sf;      // global address of the stage's 1 KB of weight scales
      };
      auto info = [&](int n) -> SInfo {
        SInfo si;
        if (n < n1a + n1b) {
          const int g = n >= n1a ? 1 : 0;
          const int s2 = f1lo[g] + (g ? n - n1a : n);
          const int T = kFc1Tiles * ub[g] + s2 / kFc1Stages, ks = s2 % kFc1Stages;
          const int e = ms.expert[T / 3], r = T % 3;
          const int flags =
              ((s2 == f1lo[g] || ks == 0) ? 1 : 0) | ((s2 == f1hi[g] - 1 || ks == kFc1Stages - 1) ? 2 : 0);
          si.m = Meta{kFC1, T, ks, flags};
          si.row0 = e * kW13Rows + r * 128;
          si.kc = 2 * ks;
          si.sf = p.w13s + static_cast<long>(e) * kW13ScaleBytes + r * 14336 + ks * 1024;
        } else {
          const int n2 = n - n1a - n1b;
          const int g = n2 >= n2a ? 1 : 0;
          const int j = f2lo[g] + (g ? n2 - n2a : n2);
          const int u = j / kFc2Tiles, mt = j % kFc2Tiles, e = ms.expert[u];
          si.m = Meta{kFC2, u, mt, 0};
          si.row0 = e * kHidden + mt * 128;
          si.kc = 0;
          si.sf = p.w2s + static_cast<long>(e) * kW2ScaleBytes + mt * 1024;
        }
        return si;
      };
      for (int n = 0; n < total; ++n) {
        const SInfo si = info(n);
        if (n >= kSlots) {
          const long long c0 = PCLK();
          tc::mbar_wait(&ms.empty[slot], phase ^ 1);
          wait_empty += PCLK() - c0;
        }
        if (n == 0) trace_ev(p, 5);
        if (n == n1a + n1b) trace_ev(p, 6);
        ms.meta[slot] = si.m;
        tc::mbar_expect_tx(&ms.full[slot], 16384 + 1024);
        if (si.m.type == kFC1)
          tc::tma_load_3d(sm + kOffA + slot * 32768, &tmA1, &ms.full[slot], 0, si.row0, si.kc);
        else
          tc::tma_load_3d(sm + kOffA + slot * 32768, &tmA2, &ms.full[slot], 0, si.row0, si.kc);
        tc::bulk_load(sm + kOffSFA + slot * 1024, si.sf, 1024, &ms.full[slot]);
        if (++slot == kSlots) {
          slot = 0;
          phase ^= 1;
        }
      }
      const int uses = total;
      trace_ev(p, 8);
      trace_ctr(p, 0, wait_empty);
      if (uses >= kSlots) tc::mbar_wait(&ms.empty[slot], phase ^ 1);
      ms.meta[slot] = Meta{kEnd, 0, 0, 0};
      tc::mbar_arrive(&ms.full[slot]);
    }
  } else {
    tc::tc_fence_after();
    const uint32_t tmem = ms.tmem_base;
    if (warp >= 2) {
      // ================================================================ x -> MXFP8 B operand (warps 2..11)
      // All 14 K-stages, natural order (loads were issued before the dedupe). Each block owner writes
      // its two 16-B chunks and its scale byte, fences, and arrives on xready[ks] (count 8*M).
      const int j = xtid & 63, t = j >> 3, kbl = j & 7;
#pragma unroll
      for (int r = 0; r < 3; ++r) {
        const int ks = 5 * r + (xtid >> 6);
        if (ks < kFc1Stages && t < M) {
          float f[32];
#pragma unroll
          for (int v = 0; v < 4; ++v) {
            const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&xraw[r][v]);
#pragma unroll
            for (int w = 0; w < 4; ++w) {
              const float2 ff = __bfloat1622float2(b2[w]);
              f[8 * v + 2 * w] = ff.x;
              f[8 * v + 2 * w + 1] = ff.y;
            }
          }
          float m16[16];
#pragma unroll
          for (int i = 0; i < 16; ++i) m16[i] = fmaxf(fabsf(f[i]), fabsf(f[i + 16]));
#pragma unroll
          for (int w = 8; w > 0; w >>= 1)
#pragma unroll
            for (int i = 0; i < w; ++i) m16[i] = fmaxf(m16[i], m16[i + w]);
          const int sb = mx_exp(m16[0]);
          const float rs = mx_rescale(sb);
          uint32_t wv[8];
#pragma unroll
          for (int w = 0; w < 8; ++w)
            wv[w] = e4m3x4(f[4 * w] * rs, f[4 * w + 1] * rs, f[4 * w + 2] * rs, f[4 * w + 3] * rs);
          const int kc0 = 16 * ks + 2 * kbl, kc1 = kc0 + 1;
          uint8_t* xq = sm + kOffXq;
          *reinterpret_cast<uint4*>(xq + (kc0 / 8) * 1024 + t * 128 + (((kc0 % 8) ^ t) * 16)) =
              make_uint4(wv[0], wv[1], wv[2], wv[3]);
          *reinterpret_cast<uint4*>(xq + (kc1 / 8) * 1024 + t * 128 + (((kc1 % 8) ^ t) * 16)) =
              make_uint4(wv[4], wv[5], wv[6], wv[7]);
          // SFB image (k-group g = kb/4): row t byte kb%4 (N = 8: only bytes 0..3 of each row are read)
          const int kb = 8 * ks + kbl;
          sm[kOffXsf + (kb >> 2) * 128 + t * 16 + (kb & 3)] = static_cast<uint8_t>(sb);
          tc::fence_proxy_async_smem();
          tc::mbar_arrive(&ms.xready[ks]);
          if (r == 0 && xtid == 0) trace_ev(p, 3);
        }
      }
      if (xtid == 0) trace_ev(p, 4);
    }

    if (warp == 1) {
      // ================================================================ MMA issuer (whole warp; lane 0 issues)
      const uint32_t idesc = tc::idesc_mxf8f6f4(128, 8, 5, 0);
      int phase = 0;
      int acc = 0, accph = 0, accuses = 0;
      int cur_u = -1, hord = -1;
      uint32_t xmask = 0;
      long long w_full = 0, w_acc = 0, w_h = 0, w_x = 0, t_issue = 0, n_st = 0;
      // Loop-invariant bases; per-slot / per-ks / per-acc operands are base + small offsets.
      const uint64_t adesc0 = tc::desc_sw128(tc::su32(sm + kOffA));          // + slot * 32 KB / 16
      const uint64_t sfa_src0 = desc_sf(tc::su32(sm + kOffSFA));             // + slot * 1 KB / 16
      const uint64_t xdesc0 = tc::desc_sw128(tc::su32(sm + kOffXq));         // + ks * 2 KB / 16
      const uint64_t xsf_src0 = desc_sf(tc::su32(sm + kOffXsf));             // + ks * 256 B / 16
      const uint64_t hdesc0 = tc::desc_sw128(tc::su32(sm + kOffHq));         // + b * 2 KB / 16
      const uint64_t hsf_src0 = desc_sf(tc::su32(sm + kOffHsf));             // + b * 640 B / 16
      const uint32_t bar_empty0 = tc::su32(&ms.empty[0]), bar_acc0 = tc::su32(&ms.accfull[0]);
      const uint32_t bar_hempty0 = tc::su32(&ms.hempty[0]);
      int slot = 0;
      // Wait until accumulator buffer `acc` is free (first use of a tile).
      auto acc_acquire = [&]() {
        if (accuses >= kAcc) {
          tc::mbar_wait(&ms.accempty[acc], accph ^ 1);
          tc::tc_fence_after();
        }
      };
      // Publish the finished accumulator (meta + the plain arrive; the MMA commit is the other arrival).
      auto acc_release = [&](const Meta& mm) {
        if (lane == 0) {
          ms.accmeta[acc] = mm;
          tc::mbar_arrive(&ms.accfull[acc]);
        }
        ++accuses;
        if (++acc == kAcc) {
          acc = 0;
          accph ^= 1;
        }
      };
      auto x_ready = [&](int ks) -> uint32_t {
        if ((xmask >> ks) & 1) return 0u;
        const long long c1 = PCLK();
        tc::mbar_wait(&ms.xready[ks], 0);
        tc::tc_fence_after();
        w_x += PCLK() - c1;
        xmask |= 1u << ks;
        return 1u;
      };
#pragma unroll 1
      for (;;) {
        long long c0 = PCLK();
        tc::mbar_wait(&ms.full[slot], phase);
        tc::tc_fence_after();
        w_full += PCLK() - c0;
        const Meta m = ms.meta[slot];
        if (m.type == kEnd) {
          if (lane == 0) {
            trace_ev(p, 19);
            trace_ctr(p, 1, w_full);
            trace_ctr(p, 2, w_acc);
            trace_ctr(p, 3, w_h);
            trace_ctr(p, 4, w_x);
            trace_ctr(p, 5, t_issue);
            trace_ctr(p, 6, n_st);
            if (cur_u >= 0) tc::mma_commit(&ms.hempty[hord & 1]);
          }
          for (int q = 0; q < 2; ++q) {  // one END per epilogue group
            if (accuses >= kAcc) tc::mbar_wait(&ms.accempty[acc], accph ^ 1);
            if (lane == 0) {
              ms.accmeta[acc] = m;
              tc::mbar_arrive(&ms.accfull[acc]);
              tc::mbar_arrive(&ms.accfull[acc]);
            }
            ++accuses;
            if (++acc == kAcc) {
              acc = 0;
              accph ^= 1;
            }
          }
          break;
        }
        c0 = PCLK();
        if (m.type == kFC1) {
          const int ks = m.b;
          const bool first = m.c & 1, last = m.c & 2;
          if (first) acc_acquire();
          const uint32_t copy_sfb = x_ready(ks);
          tc::fc1_stage(tmem + kColAcc + acc * 8, adesc0 + slot * 2048, xdesc0 + ks * 128, idesc,
                        tmem + kColSFA + slot * 8, sfa_src0 + slot * 64, tmem + kColSFBx + 8 * ks, xsf_src0 + ks * 16,
                        copy_sfb, first ? 0u : 1u, bar_empty0 + slot * 8, last ? bar_acc0 + acc * 8 : 0u);
          __syncwarp();
          if (last) acc_release(m);
        } else {
          uint32_t copy_sfb = 0, bar_hrel = 0;
          if (m.a != cur_u) {  // new expert: release the previous h buffer, wait for this one
            if (cur_u >= 0) bar_hrel = bar_hempty0 + (hord & 1) * 8;
            cur_u = m.a;
            ++hord;
            const long long c1 = PCLK();
            tc::mbar_wait(&ms.hfull[hord & 1], (hord >> 1) & 1);
            tc::tc_fence_after();
            w_h += PCLK() - c1;
            if (hord == 0 && lane == 0) trace_ev(p, 11);
            copy_sfb = 1;
          }
          const int hb = hord & 1;
          acc_acquire();
          tc::fc2_stage(tmem + kColAcc + acc * 8, adesc0 + slot * 2048, hdesc0 + hb * 128, idesc,
                        tmem + kColSFA + slot * 8, sfa_src0 + slot * 64, tmem + kColSFBh + hb * 8, hsf_src0 + hb * 40,
                        copy_sfb, bar_hrel, bar_empty0 + slot * 8, bar_acc0 + acc * 8);
          __syncwarp();
          acc_release(m);
        }
        t_issue += PCLK() - c0;
        ++n_st;
        if (++slot == kSlots) {
          slot = 0;
          phase ^= 1;
        }
      }
    } else if (warp == 2) {
      // ================================================================ h loader
      int k = -1;
      for (int g = 0; g < 2; ++g) {
        if (f2hi[g] <= f2lo[g]) continue;
        const int u0 = f2lo[g] / kFc2Tiles, u1 = (f2hi[g] - 1) / kFc2Tiles;
        for (int u = u0; u <= u1; ++u) {
          ++k;
          const int b = k & 1;
          if (k >= 2) tc::mbar_wait(&ms.hempty[b], ((k >> 1) - 1) & 1);
          const uint8_t* src = hset + static_cast<long>(u) * kHImg;
          uint4 q[4];
          uint32_t s4;
          for (;;) {
            bool bad = false;
#pragma unroll
            for (int v = 0; v < 4; ++v) {
              const int ci = lane + 32 * v;  // 16-B chunk of the 2 KB image
              q[v] = ld_volatile_v4(src + ci * 16);
              const int atom = ci >> 6, t = (ci >> 3) & 7, pos = ci & 7;
              if (atom == 0 || (pos ^ t) < 4)
                bad |= has_ff_byte(q[v].x) | has_ff_byte(q[v].y) | has_ff_byte(q[v].z) | has_ff_byte(q[v].w);
            }
            // compact scales: row t = lane/4 at bytes 16*t + 4*(lane%4); k-blocks 0..5 are written.
            s4 = ld_volatile_u32(src + 2048 + lane * 4);
            if ((lane & 3) < 2) bad |= has_ff_byte(s4 & ((lane & 3) == 0 ? 0xffffffffu : 0x0000ffffu));
            if (!__any_sync(0xffffffffu, bad)) break;
            __nanosleep(20);
          }
          uint4* dq = reinterpret_cast<uint4*>(sm + kOffHq + b * 2048);
#pragma unroll
          for (int v = 0; v < 4; ++v) dq[lane + 32 * v] = q[v];
          // SFB image g: row t bytes 0..3 = h scales of token t for k-blocks 4g..4g+3 (held by lane 4t+g)
          const uint32_t w = __shfl_sync(0xffffffffu, s4, 4 * (lane & 7) + (lane >> 3));
          if (lane < 16)
            *reinterpret_cast<uint4*>(sm + kOffHsf + b * 640 + (lane >> 3) * 128 + (lane & 7) * 16) =
                make_uint4(w, w, w, w);
          tc::fence_proxy_async_smem();
          __syncwarp();
          if (lane == 0) {
            if (k == 0) trace_ev(p, 12);
            tc::mbar_arrive(&ms.hfull[b]);
          }
        }
      }
    } else if (warp == 3) {
      // ================================================================ re-arm the idle h parity
      uint4* other = reinterpret_cast<uint4*>(p.ws + kWsH + static_cast<long>(P ^ 1) * kMaxPairs * kHImg);
      const int n16 = kMaxPairs * kHImg / 16;
      const int lo = split_lo(cta, n16, G), hi = split_lo(cta + 1, n16, G);
      const uint4 ff = make_uint4(~0u, ~0u, ~0u, ~0u);
      for (int i = lo + lane; i < hi; i += 32) other[i] = ff;
    } else if (warp >= 4) {
      // ================================================================ epilogue (two groups of 4 warps;
      // group eg handles accumulators eg, eg + 2, ...; ew = TMEM lane quadrant)
      const int ew = warp & 3, eg = (warp - 4) >> 2;
      const int ebar = eg == 0 ? 1 : 4;
      const uint32_t tl = tmem + (static_cast<uint32_t>(32 * ew) << 16);
      const int row = 32 * ew + lane;  // physical row within the 128-row tile
      const bool up_lane = (lane & 8) == 0;
      const int i_local = 2 * (lane % 8) + lane / 16;
      float* pbase = reinterpret_cast<float*>(p.ws + kWsP);
      int acc = eg, accph = 0;
      long long e_wait = 0, e_fc1 = 0, e_fc2 = 0, e_n2 = 0;
      for (;;) {
        long long ec0 = PCLK();
        tc::mbar_wait(&ms.accfull[acc], accph);
        tc::tc_fence_after();
        e_wait += PCLK() - ec0;
        ec0 = PCLK();
        const Meta m = ms.accmeta[acc];
        if (m.type == kEnd) break;
        float v[8];
        tc::tmem_ld8(tl + kColAcc + acc * 8, v);
        tc::tc_fence_before();
        __syncwarp();
        if (lane == 0) tc::mbar_arrive(&ms.accempty[acc]);
        if ((acc += 2) >= kAcc) {
          acc -= kAcc;
          accph ^= 1;
        }
        if (m.type == kFC1) {
          const int T = m.a;
          const int g = (T >= kFc1Tiles * ub[1]) ? 1 : 0;
          const int s0 = (T - kFc1Tiles * ub[g]) * kFc1Stages;  // group-local first stage of the tile
          if (s0 < f1lo[g]) {  // not the owner: publish the partial
            float* dst = pbase + (static_cast<long>(g) * kMaxG + cta) * 1024 + row;
#pragma unroll
            for (int t = 0; t < kMaxM; ++t)
              if (t < M) dst[t * 128] = v[t];
            continue;
          }
          if (eg == 0 && ew == 0 && lane == 0 && kTraceOn && p.trace != nullptr && p.trace[cta * 128 + 17] == 0) trace_ev(p, 17);
          // Other pieces of this tile: CTAs cta+1.. whose ranges start inside the tile.
          for (int c2 = cta + 1; c2 < G && split_lo(c2, S1g[g], G) < s0 + kFc1Stages; ++c2) {
            if (split_lo(c2 + 1, S1g[g], G) == split_lo(c2, S1g[g], G)) continue;  // empty range: no piece
            float* src = pbase + (static_cast<long>(g) * kMaxG + c2) * 1024 + row;
            float pv[8];
            for (;;) {
              bool bad = false;
#pragma unroll
              for (int t = 0; t < kMaxM; ++t) {
                pv[t] = 0.f;
                if (t < M) {
                  const uint32_t w = ld_volatile_u32(src + t * 128);
                  bad |= (w == 0xffffffffu);
                  pv[t] = __uint_as_float(w);
                }
              }
              if (!__any_sync(0xffffffffu, bad)) break;
            }
#pragma unroll
            for (int t = 0; t < kMaxM; ++t)
              if (t < M) {
                v[t] += pv[t];
                src[t * 128] = __uint_as_float(0xffffffffu);
              }
          }
          if (eg == 0 && ew == 0 && lane == 0) trace_ev(p, 24);
          const int u = T / 3, r = T % 3;
          float h[8], am[8];
#pragma unroll
          for (int t = 0; t < 8; ++t) {
            const float o = __shfl_xor_sync(0xffffffffu, v[t], 8);  // the gate row sits 8 lanes up
            const float hv = p.beta_lb * tanh_fast(o * p.inv_beta) * sigmoid_fast(o) * tanh_fast(v[t] * p.inv_lb);
            h[t] = up_lane ? hv : 0.f;
            am[t] = fabsf(h[t]);
          }
#pragma unroll
          for (int t = 0; t < 8; ++t) {
#pragma unroll
            for (int off = 1; off < 32; off <<= 1) am[t] = fmaxf(am[t], __shfl_xor_sync(0xffffffffu, am[t], off));
          }
          if (lane == 0) {
#pragma unroll
            for (int t = 0; t < 8; ++t) ms.amax[eg][ew][t] = am[t];
          }
          asm volatile("bar.sync %0, 128;" ::"r"(ebar) : "memory");
          int sbt = 0;
#pragma unroll
          for (int t = 0; t < 8; ++t) {
            const int sb = mx_exp(fmaxf(am[t], ms.amax[eg][ew ^ 1][t]));
            if (lane == t) sbt = sb;
            if (up_lane)
              ms.hst[eg][ew][t][i_local] =
                  static_cast<uint8_t>(__nv_cvt_float_to_fp8(h[t] * mx_rescale(sb), __NV_SATFINITE, __NV_E4M3));
          }
          __syncwarp();
          uint8_t* himg = hset + static_cast<long>(u) * kHImg;
          if (lane < 8) {
            const int t = lane, kc = 4 * r + ew;
            const uint4 val = *reinterpret_cast<const uint4*>(ms.hst[eg][ew][t]);
            *reinterpret_cast<uint4*>(himg + (kc / 8) * 1024 + t * 128 + (((kc % 8) ^ t) * 16)) = val;
            if ((ew & 1) == 0) himg[2048 + t * 16 + 2 * r + ew / 2] = static_cast<uint8_t>(sbt);
          }
          asm volatile("bar.sync %0, 128;" ::"r"(ebar) : "memory");  // ms.amax / hst reuse
          if (eg == 0 && ew == 0 && lane == 0) trace_ev(p, 18);
        } else {
          const int u = m.a, mt = m.b;
          // physical row p -> logical output row 32*(p/32) + 4*(p%8) + (p%32)/8; lanes l and l+8 hold
          // consecutive logical rows, stored as one bf16x2.
          const int n = mt * 128 + 32 * ew + 4 * (lane % 8) + lane / 8;
          const uint2 ts = *reinterpret_cast<const uint2*>(ms.tokslot[u]);
#pragma unroll
          for (int t = 0; t < kMaxM; ++t) {
            const int j = static_cast<signed char>(((t < 4 ? ts.x : ts.y) >> (8 * (t & 3))) & 0xff);
            if (j >= 0) {  // warp-uniform
              const float o = __shfl_down_sync(0xffffffffu, v[t], 8);
              if ((lane & 8) == 0) {
                const __nv_bfloat162 b2 = __floats2bfloat162_rn(v[t], o);
                *reinterpret_cast<__nv_bfloat162*>(p.out + static_cast<long>(t * kTopK + j) * kHidden + n) = b2;
              }
            }
          }
          e_fc2 += PCLK() - ec0;
          ++e_n2;
        }
      }
      if (eg == 0 && ew == 0 && lane == 0) {
        trace_ev(p, 13);
        trace_ctr(p, 13, e_wait);
        trace_ctr(p, 14, e_fc1);
        trace_ctr(p, 15, e_fc2);
        trace_ctr(p, 7, e_n2);
      }
    }
  }
  tc::tc_fence_before();
  __syncthreads();
  if (warp == 3) {
    tc::tc_fence_after();
    tc::tmem_dealloc(ms.tmem_base, kTmemCols);
  }
  if (threadIdx.x == 0) {
    ctr[cta] = ms.parity + 1;
    trace_ev(p, 20);
  }
}

// ---------------------------------------------------------------- host
PFN_cuTensorMapEncodeTiled_v12000 encoder() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    void* q = nullptr;
    cudaDriverEntryPointQueryResult r;
    C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &q, 12000, cudaEnableDefault, &r));
    TORCH_CHECK(q != nullptr && r == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled unavailable");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(q);
  }
  return fn;
}

// 3D fp4 view: {128 elements (64 B) of K, rows, K chunks of 128}; box {128, 128, 2} -> smem [2][128][128 B].
CUtensorMap fp4_map(const void* base, uint64_t rows, uint64_t row_bytes, uint64_t kchunks, uint64_t kstride,
                    CUtensorMapL2promotion promo) {
  CUtensorMap m;
  const cuuint64_t dims[3] = {128, rows, kchunks};
  const cuuint64_t strides[2] = {row_bytes, kstride};
  const cuuint32_t box[3] = {128, 128, 2};
  const cuuint32_t es[3] = {1, 1, 1};
  const CUresult r = encoder()(&m, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, 3, const_cast<void*>(base), dims, strides,
                               box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, promo,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
  return m;
}

struct MapEntry {
  const void* w13;
  const void* w2;
  long E;
  CUtensorMap a1, a2;
};

void launch(const torch::Tensor& x, const torch::Tensor& topk_ids, const torch::Tensor& w13,
            const torch::Tensor& w13_scale, const torch::Tensor& w2, const torch::Tensor& w2_scale,
            const torch::Tensor& workspace, const torch::Tensor& out, double beta, double linear_beta,
            const std::optional<torch::Tensor>& trace) {
  const int M = x.size(0);
  TORCH_CHECK(M >= 1 && M <= kMaxM, "moe8 supports 1..8 tokens");
  TORCH_CHECK(x.size(1) == kHidden && x.is_contiguous() && x.scalar_type() == at::kBFloat16);
  TORCH_CHECK(topk_ids.size(0) == M && topk_ids.size(1) == kTopK && topk_ids.scalar_type() == at::kInt &&
              topk_ids.is_contiguous());
  TORCH_CHECK(w13.dim() == 3 && w13.size(1) == kW13Rows && w13.size(2) == kW13RowBytes && w13.is_contiguous());
  TORCH_CHECK(w2.dim() == 3 && w2.size(1) == kHidden && w2.size(2) == 128 && w2.is_contiguous());
  const long E = w13.size(0);
  TORCH_CHECK(E <= kMaxE && w2.size(0) == E && w13_scale.is_contiguous() && w2_scale.is_contiguous());
  TORCH_CHECK(w13_scale.numel() * w13_scale.element_size() == E * kW13ScaleBytes);
  TORCH_CHECK(w2_scale.numel() * w2_scale.element_size() == E * kW2ScaleBytes);
  TORCH_CHECK(workspace.is_contiguous() && workspace.numel() * workspace.element_size() >= kWsBytes,
              "workspace must hold >= ", kWsBytes, " bytes, filled with 0xFF once");
  TORCH_CHECK(out.size(0) == M * kTopK && out.size(1) == kHidden && out.scalar_type() == at::kBFloat16 &&
              out.is_contiguous());
  TORCH_CHECK(linear_beta > 0.0 && beta > 0.0);
  const int dev = x.get_device();
  static int sms = 0;
  if (sms == 0) {
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    C10_CUDA_CHECK(cudaFuncSetAttribute(moe8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  }
  TORCH_CHECK(sms <= kMaxG);
  // Tensor maps per weight pair (one entry per MoE layer), encoded once; passed by value as kernel
  // parameters, so CUDA-graph capture bakes them in.
  static std::vector<MapEntry> cache;
  const MapEntry* mc = nullptr;
  for (const MapEntry& me : cache)
    if (me.w13 == w13.data_ptr() && me.w2 == w2.data_ptr() && me.E == E) mc = &me;
  if (mc == nullptr) {
    MapEntry me;
    me.a1 = fp4_map(w13.data_ptr(), E * kW13Rows, kW13RowBytes, 28, 64, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    // W2: K-chunks at k = 0 and k = 64 (32-B chunk stride), so bytes 96..127 (K padding) are never read.
    me.a2 = fp4_map(w2.data_ptr(), E * kHidden, 128, 2, 32, CU_TENSOR_MAP_L2_PROMOTION_NONE);
    me.w13 = w13.data_ptr();
    me.w2 = w2.data_ptr();
    me.E = E;
    cache.push_back(me);
    mc = &cache.back();
  }
  Params prm;
  prm.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  prm.topk_ids = topk_ids.data_ptr<int>();
  prm.w13s = static_cast<const uint8_t*>(w13_scale.data_ptr());
  prm.w2s = static_cast<const uint8_t*>(w2_scale.data_ptr());
  prm.w2 = static_cast<const uint8_t*>(w2.data_ptr());
  prm.ws = static_cast<uint8_t*>(workspace.data_ptr());
  prm.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  prm.M = M;
  prm.E = static_cast<int>(E);
  prm.beta_lb = static_cast<float>(beta * linear_beta);
  prm.inv_beta = static_cast<float>(1.0 / beta);
  prm.inv_lb = static_cast<float>(1.0 / linear_beta);
  prm.trace = trace ? reinterpret_cast<unsigned long long*>(trace->data_ptr()) : nullptr;

  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(sms);
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = kSmemBytes;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, moe8_kernel, mc->a1, mc->a2, prm));
}

}  // namespace moe8

// gemm2_out[t*16 + j, n] = W2_{e_tj}[n, :192] . h_tj (unweighted, not summed over j), bf16 [M*16, 3584].
// `barrier` is accepted for signature compatibility with k3moe.moe_fused_unfinalized and unused.
void moe_fused_unfinalized(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor w13, torch::Tensor w13_scale,
                           torch::Tensor w2, torch::Tensor w2_scale, torch::Tensor workspace, torch::Tensor barrier,
                           torch::Tensor gemm2_out, double beta, double linear_beta,
                           std::optional<torch::Tensor> trace) {
  (void)barrier;
  moe8::launch(x, topk_ids, w13, w13_scale, w2, w2_scale, workspace, gemm2_out, beta, linear_beta, trace);
}

int64_t workspace_bytes() { return moe8::kWsBytes; }

TORCH_LIBRARY(k3moe8, m) {
  m.def("moe_fused_unfinalized(Tensor x, Tensor topk_ids, Tensor w13, Tensor w13_scale, "
        "Tensor w2, Tensor w2_scale, Tensor(a!) workspace, Tensor(b!) barrier, Tensor(c!) gemm2_out, "
        "float beta, float linear_beta, Tensor(d!)? trace=None) -> ()");
  m.def("workspace_bytes() -> int");
}
TORCH_LIBRARY_IMPL(k3moe8, CUDA, m) { m.impl("moe_fused_unfinalized", &moe_fused_unfinalized); }
TORCH_LIBRARY_IMPL(k3moe8, CompositeExplicitAutograd, m) { m.impl("workspace_bytes", &workspace_bytes); }
