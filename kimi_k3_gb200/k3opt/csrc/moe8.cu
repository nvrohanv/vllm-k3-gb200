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
// v11: the kernel is instantiated per token capacity MT (1, 2, 4, 8): the owner's SiTU / MXFP8 path of the
// FC1 -> FC2 hand-off (on the critical path at small M) and the per-token loops run MT columns only, and at
// MT <= 2 the expert dedupe is skipped (the pairs are the experts). v10 -> v11: M = 1 11.56 -> 10.13 us,
// M = 2 13.51 -> 12.45, M = 4 18.1 -> 17.5, M = 8 27.86 -> 27.78.
// Weight loads use an L2 evict-first cache policy (single-use stream; keeps L2 for x / partials / h images and
// other layers' data): measured M = 8 29.8 -> 27.9 us, M = 4 18.8 -> 18.0 us (-DMOE8_EVICT_NORMAL to disable).
// x loads are issued after the expert dedupe (-DMOE8_XLOAD_EARLY for the old order).
//
// v12 (cluster mode, MT = 4 i.e. M = 3..4; -DMOE8_CLUSTER=0 restores v11's grid): the grid is 36 clusters of 4
// CTAs (144 of 152 SMs: the same HBM rate as 152, measured). The last 36 L experts in dedupe order (group B,
// L = floor(D * MOE8_CL_B_PCT / 100 / 36), 1 at M = 4) are cluster-local: cluster k owns L whole experts, split
// over its 4 CTAs. Their FC1 partials go to the tile owner and the owners' h chunks (+ one scale word per token)
// go straight into every consumer's smem receive buffer with st.async + mbarrier complete_tx (DSMEM; dedicated
// single-use receive buffers, so no flow control). The FC2 MMA warp waits on the receive barrier itself and builds
// the SFB images from the pushed scale words. Group A (the other experts) keeps v11's global path; the order per
// CTA stays FC1(A), FC1(B), FC2(A), FC2(B). Cluster barrier: every thread arrives (relaxed, after
// fence.mbarrier_init) at the start; only the epilogue warps wait, right before their first remote store.
// The per-CTA call counters of the SMs a 144-CTA grid does not use are advanced by the launched CTAs (h-image
// parity stays in lockstep across calls with different grids). M <= 2 (no clusters) also runs on the 144-CTA grid
// (-DMOE8_SMALLM_GRID=0: all SMs); M = 5..8 keeps v11's 152-CTA grid and code path.
// v11 -> v12, same session, graph of 32 calls (us): M = 1 10.13 -> 9.67-9.71, M = 2 12.42-12.46 -> 12.29-12.34,
// M = 3 15.42-15.45 -> 15.33-15.40, M = 4 17.47 -> 16.73-16.78 (other sessions: 17.52-17.60 -> 16.81-16.97),
// M = 5..8 unchanged (within +-0.1). Numerics identical to v11 (same accumulation order, same quantization).
// With a concurrent 32-CTA side kernel the 144-CTA grids co-reside far better: + 2 / 5 us side kernel costs
// M = 4 +0.28 / +0.37 us (v11 +1.80 / +4.53), M = 2 +0.27 / +0.46 (v11 +2.08 / +4.80), M = 1 +0.27 / +3.45
// (v11 +2.01 / +4.81); M = 8 unchanged (+0.8 / +3.7).
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
constexpr int kThreads = 384;
#ifndef MOE8_GROUP_MIN_D
#define MOE8_GROUP_MIN_D 40
#endif
#ifndef MOE8_GROUP_A_PCT
#define MOE8_GROUP_A_PCT 50
#endif  // 12 warps; two epilogue groups (warps 4..7, 8..11)
constexpr int kMaxG = 160;

// smem layout (offsets from a 1024-aligned base)
constexpr int kOffA = 0;                             // kSlots x 32 KB weights (unpacked fp4)
constexpr int kOffSFA = kOffA + kSlots * 32768;      // kSlots x 1 KB weight scales
constexpr int kOffXq = kOffSFA + kSlots * 1024;      // 28 KB x (e4m3), B operand of FC1
constexpr int kOffHq = kOffXq + 28672;               // 2 x 2 KB h (e4m3), B operand of FC2
constexpr int kOffXsf = kOffHq + 2 * 2048;           // 28 x 128 (+384) SFB images of x
constexpr int kOffHsf = kOffXsf + 28 * 128 + 384;    // 2 x (2 x 128 + 384) SFB images of h
constexpr int kOffMisc = kOffHsf + 2 * 640;
// MOE8_PF_PARTIAL (opt-in): warp 3 prefetches the remote FC1 partials of this CTA's split owner tile into smem.
// Measured: owner reduce median 2.05 -> 0.77 us and MMA h-stall p90 2.05 -> 0.77 us at M = 8, but no end-to-end
// gain (M = 4/8 within noise) and +0.2..0.4 us at M = 1/2 from the extra 16 KB of smem (less L1), so it is off.
constexpr int kOffPf = kOffMisc + 5120;              // 2 groups x 2 prefetched remote partials (fp32 [8][128])
constexpr int kPfSlots = 2;
#ifdef MOE8_PF_PARTIAL
constexpr int kSmemBytes = kOffPf + 2 * kPfSlots * 4096;
#else
constexpr int kSmemBytes = kOffPf;
#endif
static_assert(kSmemBytes <= 232448, "smem");

// ---- v12 cluster mode (MT = 4; MT = 8 with -DMOE8_CLUSTER_MT8=1): the grid is launched as clusters of kCl CTAs.
// The last 36 L experts (group B) are cluster-local: cluster k owns L whole experts; their FC1 partials and h chunks
// move over DSMEM (st.async + mbarrier complete_tx) into the receive buffers below. Group A = the other experts,
// v11's global path.
#ifndef MOE8_CLUSTER
#define MOE8_CLUSTER 1
#endif
#ifndef MOE8_HPOLL_NS
#define MOE8_HPOLL_NS 20  // h-loader: sleep between polls of a global h image
#endif
#ifndef MOE8_PPOLL_NS
#define MOE8_PPOLL_NS 0  // owner: sleep between polls of a remote FC1 partial
#endif
#ifndef MOE8_PLOOK
#define MOE8_PLOOK 0  // owner (MT = 8): load the next remote partial with the current one's first poll
#endif
#ifndef MOE8_CLUSTER_MT8
#define MOE8_CLUSTER_MT8 0
#endif
#ifndef MOE8_CL_B_PCT
#define MOE8_CL_B_PCT 60  // at most this % of the experts are cluster-local (L = floor(D * pct / 100 / #clusters))
#endif
#ifndef MOE8_CL_MAX_L
#define MOE8_CL_MAX_L 4
#endif
#ifndef MOE8_CL_BFIRST
#define MOE8_CL_BFIRST 0
#endif

constexpr int kCl = 4;
template <int MT>
__host__ __device__ constexpr bool cl_mode() {
  return MOE8_CLUSTER != 0 && (MT == 4 || (MT == 8 && MOE8_CLUSTER_MT8 != 0));
}
#if defined(MOE8_PF_PARTIAL) && MOE8_CLUSTER
#error "MOE8_PF_PARTIAL is not supported with the cluster mode"
#endif
constexpr int kOffHr = (kSmemBytes + 1023) / 1024 * 1024;  // 2 x 2 KB h receive images (B operand, SW128)
constexpr int kOffHrsf = kOffHr + 2 * 2048;                // 2 x (2 x 128 + 384) SFB images of them
constexpr int kOffHsc = kOffHrsf + 2 * 640;                // 2 x [3 tiles][8 tokens] u32 scale words (sb0 | sb1 << 8)
constexpr int kOffPrecv = kOffHsc + 2 * 128;               // 2 remote pieces x [128 rows][MT] fp32 partials
template <int MT>
__host__ __device__ constexpr int smem_bytes() { return cl_mode<MT>() ? kOffPrecv + 2 * 128 * MT * 4 : kSmemBytes; }
static_assert(kOffPrecv + 2 * 128 * 8 * 4 <= 232448, "smem (cluster mode)");

// TMEM columns (all scale-factor bases are multiples of 4)
constexpr uint32_t kTmemCols = 512;
constexpr uint32_t kColAcc = 0;                        // kAcc x 8
constexpr uint32_t kColSFA = 128;                      // kSlots x 8
constexpr uint32_t kColSFBh = kColSFA + kSlots * 8;    // 4 x 8 (2 global-path buffers, 2 cluster receive buffers)
constexpr uint32_t kColSFBx = 256;                     // 28 x 4
static_assert(kAcc * 8 <= 128 && kColSFBh + 32 <= kColSFBx, "tmem");

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
  uint64_t xgo;  // MOE8_XAFTER: the producer has issued its first ring of stages (x loads may start)
  uint64_t pfbar[2][2];  // remote FC1 partials of this CTA's split owner tile, prefetched into smem by warp 3
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
  // v12 cluster mode (after v11's fields, so the other instantiations keep v11's layout)
  uint64_t hrecv[2];   // DSMEM h chunks + scale words received (complete_tx from the owners)
  uint64_t precv;      // DSMEM FC1 partials received
  uint64_t hfullR[2];  // -DMOE8_CL_HLOADER only: the h loader has built receive buffer j's SFB images
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

// ---------------------------------------------------------------- cluster / DSMEM (cluster mode only)
#ifndef MOE8_CL_WAIT
#define MOE8_CL_WAIT 0  // DSMEM receive waits: 0 test_wait spin, 1 try_wait + suspend hint, 2 test_wait + nanosleep
#endif
#ifndef MOE8_CL_WAIT_NS
#define MOE8_CL_WAIT_NS 1000
#endif
namespace clx {
// shared::cta address -> shared::cluster address of the same offset in CTA `rank` of this cluster
__device__ __forceinline__ uint32_t mapa(uint32_t saddr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(saddr), "r"(rank));
  return r;
}
__device__ __forceinline__ void st_async_v4(uint32_t caddr, uint4 v, uint32_t cbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(caddr),
               "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "r"(cbar)
               : "memory");
}
__device__ __forceinline__ void st_async_b32(uint32_t caddr, uint32_t v, uint32_t cbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(caddr), "r"(v), "r"(cbar)
               : "memory");
}
// Wait on a local mbarrier completed by remote st.async (acquire at cluster scope). Spins on test_wait: a thread
// suspended in try_wait is not necessarily woken by a remote complete_tx (agents/moe8/body, agents/probe).
__device__ __forceinline__ void wait(uint64_t* bar, uint32_t parity) {
#ifdef TC_WAIT_TIMEOUT
  {
    const uint64_t t0 = tc::globaltimer();
    while (!tc::mbar_test(bar, parity)) {
      if (tc::globaltimer() - t0 > 2000000000ull) {
        printf("clx::wait timeout: block %d thread %d parity %u\n", blockIdx.x, threadIdx.x, parity);
        asm volatile("trap;");
      }
    }
  }
#endif
#if MOE8_CL_WAIT == 0
  asm volatile(
      "{\n\t.reg .pred P1;\n\tWAIT_%=:\n\t"
      "mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\n\t"
      "@!P1 bra WAIT_%=;\n\t}" ::"r"(tc::su32(bar)),
      "r"(parity)
      : "memory");
#elif MOE8_CL_WAIT == 1  // suspend (bounded by the hint) instead of spinning
  asm volatile(
      "{\n\t.reg .pred P1;\n\tWAIT_%=:\n\t"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1, %2;\n\t"
      "@!P1 bra WAIT_%=;\n\t}" ::"r"(tc::su32(bar)),
      "r"(parity), "n"(MOE8_CL_WAIT_NS)
      : "memory");
#else  // test + nanosleep backoff
  for (;;) {
    uint32_t ok;
    asm volatile(
        "{\n\t.reg .pred P1;\n\tmbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t}"
        : "=r"(ok)
        : "r"(tc::su32(bar)), "r"(parity)
        : "memory");
    if (ok) break;
    __nanosleep(MOE8_CL_WAIT_NS);
  }
#endif
}
__device__ __forceinline__ void arrive_release() { asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory"); }
// Relaxed arrive: enough to publish mbarrier inits when preceded by fence.mbarrier_init.release.cluster (the
// release arrive measured +0.4 us per call at M = 4).
__device__ __forceinline__ void arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
__device__ __forceinline__ void wait_acquire() { asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
}  // namespace clx

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

// Per-tile / per-expert hand-off timestamps (MOE8_TRACE builds): region at trace[20480 ...].
__device__ __forceinline__ void trace_at(const Params& p, int idx, long long v) {
#ifdef MOE8_TRACE
  if (p.trace != nullptr) p.trace[20480 + idx] = v;
#endif
}
// Per-warp epilogue timestamps of FC1 owner tiles (MOE8_TRACE builds): trace[25600 + (T * 4 + ew) * 6 + k].
__device__ __forceinline__ void trace_w(const Params& p, int T, int ew, int k, long long v) {
#ifdef MOE8_TRACE
  if (p.trace != nullptr && T < 400) p.trace[25600 + (T * 4 + ew) * 6 + k] = v;
#endif
}
__device__ __forceinline__ int split_lo(int c, int total, int G) {
  return static_cast<int>(static_cast<unsigned>(c * total) / static_cast<unsigned>(G));  // < 2^31
}

// MT: token capacity of the instantiation (1, 2, 4, 8 >= M): the owner's SiTU / MXFP8 path and the per-token
// loops only run MT columns (rows t >= M of the accumulators are zero, their h rows and scales are written as 0).
template <int MT>
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
    if constexpr (cl_mode<MT>()) {
      tc::mbar_init(&ms.hrecv[0], 1);
      tc::mbar_init(&ms.hrecv[1], 1);
      tc::mbar_init(&ms.precv, 1);
      tc::mbar_init(&ms.hfullR[0], 1);
      tc::mbar_init(&ms.hfullR[1], 1);
    }
    for (int k = 0; k < kFc1Stages; ++k) tc::mbar_init(&ms.xready[k], 8 * p.M);
    tc::mbar_init(&ms.xgo, 1);
    for (int g = 0; g < 2; ++g)
      for (int i = 0; i < kPfSlots; ++i) tc::mbar_init(&ms.pfbar[g][i], 1);
    tc::fence_mbar_init();
    tc::prefetch_tmap(&tmA1);
    tc::prefetch_tmap(&tmA2);
  }
  if (warp == 3) tc::tmem_alloc(&ms.tmem_base, kTmemCols);
  // cluster mode: the mbarrier inits (thread 0, fence.mbarrier_init above) must be visible cluster-wide before any
  // remote st.async. Every thread arrives (relaxed; thread 0's fence.mbarrier_init.release.cluster publishes the
  // inits); only the epilogue warps wait, lazily, right before their first remote store (barrier.cluster.wait only
  // waits for non-exited threads, so threads that never wait are fine). Measured: waiting in all warps at their
  // start or end costs 0.1-0.25 us per call at M = 4; a release arrive another ~0.1.
#ifdef MOE8_CL_ARRIVE_RELEASE
  if constexpr (cl_mode<MT>()) clx::arrive_release();
#else
  if constexpr (cl_mode<MT>()) clx::arrive_relaxed();
#endif
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
#ifndef MOE8_DEDUPE_ALL_M
  // M <= 2 (MT <= 2 instantiations): no dedupe pass; the 16M (token, slot) pairs are the experts (a routed
  // expert shared by the two tokens is streamed twice: ~1.5% of the bytes at M = 2 on random routing).
  constexpr bool nodedup = MT <= 2;
#else
  constexpr bool nodedup = false;
#endif
  int myid = -1;
  if (threadIdx.x < kMaxPairs) {
    uint2 ts = make_uint2(~0u, ~0u);
    if (threadIdx.x < npairs) {
      myid = p.topk_ids[threadIdx.x];
      if (nodedup) {
        ms.expert[threadIdx.x] = myid;
        const uint32_t byte = static_cast<uint32_t>(threadIdx.x % kTopK);
        if (threadIdx.x < kTopK) ts.x = (ts.x & ~0xffu) | byte;
        else ts.x = (ts.x & ~0xff00u) | (byte << 8);
      }
    }
    *reinterpret_cast<uint2*>(ms.tokslot[threadIdx.x]) = ts;
  }
  int* ctr = reinterpret_cast<int*>(p.ws + kWsCtr);
  if (threadIdx.x == 160) ms.parity = ctr[cta];
  if (nodedup && threadIdx.x == 0) ms.D = npairs;
  __syncthreads();
  // x does not depend on the routing: warps 2..11 issue their x loads once the routing ids are in (so the ids load is not queued behind them), overlapping the rest of the dedupe.
  // Thread xtid, round r: block j = xtid % 64 (token j/8, 32-block j%8) of K-stage 5r + xtid/64.
  const int xtid = static_cast<int>(threadIdx.x) - 64;
  uint4 xraw[3][4];
  auto load_x = [&]() {
#pragma unroll
    for (int r = 0; r < 3; ++r) {
      const int ks = 5 * r + (xtid >> 6), j = xtid & 63, t = j >> 3, kbl = j & 7;
      const bool ok = ks < kFc1Stages && t < M;
      const uint4* src = reinterpret_cast<const uint4*>(p.x + t * kHidden + (ok ? ks : 0) * 256 + kbl * 32);
#pragma unroll
#ifdef MOE8_DBG_NO_XLOAD  // timing experiment only (wrong results): no x loads, quantization kept
      for (int v = 0; v < 4; ++v) xraw[r][v] = make_uint4(ok ? 0x3f803f80u : 0u, 0, 0, 0);
      (void)src;
#else
      for (int v = 0; v < 4; ++v) xraw[r][v] = ok ? src[v] : make_uint4(0, 0, 0, 0);
#endif
    }
  };
#ifdef MOE8_XLOAD_EARLY
  if (warp >= 2) load_x();
#endif
  if (!nodedup) {
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
  }  // !nodedup
#if !defined(MOE8_XLOAD_EARLY) && !defined(MOE8_XAFTER)
  // x loads after the dedupe: at M = 8 the 60 KB/CTA of x requests otherwise stall the dedupe's barriers in
  // the LSU queue; the MMA is gated per K-stage by xready[] anyway.
  if (warp >= 2) load_x();
#endif
  asm volatile("griddepcontrol.launch_dependents;");
  if (threadIdx.x == 0) trace_ev(p, 2);
  const int D = ms.D;
  const int P = ms.parity & 1;
  constexpr bool kClu = cl_mode<MT>();
  // cluster mode: L cluster-local experts per cluster (group 1 = experts [D - L * ncl, D)); L = 0: v11 groups
  const int ncl = kClu ? G / kCl : 1, ck = kClu ? cta / kCl : 0, crank = kClu ? cta % kCl : 0;
  const int L = kClu ? min(MOE8_CL_MAX_L, (D * MOE8_CL_B_PCT / 100) / ncl) : 0;
  const bool loc1 = kClu && L > 0;
  // group-1 FC1 stage / FC2 unit start of cluster CTA q (cluster mode, loc1)
  auto lo1c = [&](int q) { return kFc1Tiles * kFc1Stages * L * ck + split_lo(q, kFc1Tiles * kFc1Stages * L, kCl); };
  // Two expert groups: FC1(A), FC1(B), FC2(A), FC2(B). Group A's h is ready long before FC2(A)
  // starts, and FC2(A) covers the latency of group B's FC1 -> h hand-off.
  // group A gets MOE8_GROUP_A_PCT % of the experts: a smaller group B leaves h(B) more slack behind FC2(A)
  const int ub[3] = {0, loc1 ? D - L * ncl : (D > MOE8_GROUP_MIN_D ? (D * MOE8_GROUP_A_PCT + 99) / 100 : D), D};
  auto lo2c = [&](int q) { return kFc2Tiles * (ub[1] + L * ck) + split_lo(q, kFc2Tiles * L, kCl); };
  // remote-partial prefetch (warp 3) only with two expert groups: at M <= 2 it measured slower
#ifdef MOE8_PF_PARTIAL
#ifdef MOE8_PF_ALL_M
  const bool pf_on = true;
#else
  const bool pf_on = D > MOE8_GROUP_MIN_D;
#endif
#else
  const bool pf_on = false;
#endif
  int S1g[2], f1lo[2], f1hi[2], f2lo[2], f2hi[2];
#pragma unroll
  for (int g = 0; g < 2; ++g) {
    const int De = ub[g + 1] - ub[g];
    S1g[g] = kFc1Tiles * kFc1Stages * De;
    if (g == 1 && loc1) {
      f1lo[g] = lo1c(crank);
      f1hi[g] = lo1c(crank + 1);
      f2lo[g] = lo2c(crank);
      f2hi[g] = lo2c(crank + 1);
    } else {
      f1lo[g] = split_lo(cta, S1g[g], G);
      f1hi[g] = split_lo(cta + 1, S1g[g], G);
      f2lo[g] = kFc2Tiles * ub[g] + split_lo(cta, kFc2Tiles * De, G);
      f2hi[g] = kFc2Tiles * ub[g] + split_lo(cta + 1, kFc2Tiles * De, G);
    }
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
      // stage order FC1(first group), FC1(other), FC2(first), FC2(other); MOE8_CL_BFIRST: the cluster-local group
      // first (its DSMEM hand-off then hides behind FC1(A), the global one behind FC2(B))
      const int gf = (kClu && loc1 && MOE8_CL_BFIRST) ? 1 : 0;
      const int nf1 = gf ? n1b : n1a, nf2 = gf ? n2b : n2a;
      struct SInfo {
        Meta m;
        int row0, kc;           // TMA coordinates (weights)
        const uint8_t* sf;      // global address of the stage's 1 KB of weight scales
      };
      auto info = [&](int n) -> SInfo {
        SInfo si;
        if (n < n1a + n1b) {
          const int g = n >= nf1 ? gf ^ 1 : gf;
          const int s2 = f1lo[g] + (n >= nf1 ? n - nf1 : n);
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
          const int g = n2 >= nf2 ? gf ^ 1 : gf;
          const int j = f2lo[g] + (n2 >= nf2 ? n2 - nf2 : n2);
          const int u = j / kFc2Tiles, mt = j % kFc2Tiles, e = ms.expert[u];
          // c: h buffer of a cluster-local expert (2 + its index among this CTA's group-1 experts), else 0
          si.m = Meta{kFC2, u, mt, (kClu && g == 1 && loc1) ? 2 + u - f2lo[1] / kFc2Tiles : 0};
          si.row0 = e * kHidden + mt * 128;
          si.kc = 0;
          si.sf = p.w2s + static_cast<long>(e) * kW2ScaleBytes + mt * 1024;
        }
        return si;
      };
#ifndef MOE8_EVICT_NORMAL
      const uint64_t wpol = tc::policy_evict_first();
#endif
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
#ifndef MOE8_EVICT_NORMAL
        // single-use weight stream: evict-first keeps L2 for x / partials / h images
        if (si.m.type == kFC1)
          tc::tma_load_3d_hint(sm + kOffA + slot * 32768, &tmA1, &ms.full[slot], 0, si.row0, si.kc, wpol);
        else
          tc::tma_load_3d_hint(sm + kOffA + slot * 32768, &tmA2, &ms.full[slot], 0, si.row0, si.kc, wpol);
        tc::bulk_load_hint(sm + kOffSFA + slot * 1024, si.sf, 1024, &ms.full[slot], wpol);
#else
        if (si.m.type == kFC1)
          tc::tma_load_3d(sm + kOffA + slot * 32768, &tmA1, &ms.full[slot], 0, si.row0, si.kc);
        else
          tc::tma_load_3d(sm + kOffA + slot * 32768, &tmA2, &ms.full[slot], 0, si.row0, si.kc);
        tc::bulk_load(sm + kOffSFA + slot * 1024, si.sf, 1024, &ms.full[slot]);
#endif
#ifdef MOE8_XAFTER
        if (n == min(total, MOE8_XAFTER) - 1) tc::mbar_arrive(&ms.xgo);
#endif
        if (++slot == kSlots) {
          slot = 0;
          phase ^= 1;
        }
      }
      const int uses = total;
#ifdef MOE8_XAFTER
      if (total == 0) tc::mbar_arrive(&ms.xgo);
#endif
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
#ifdef MOE8_XAFTER
      // x loads only after the producer's first TMAs are out (they would otherwise queue behind 60 KB/CTA of x)
      tc::mbar_wait(&ms.xgo, 0);
      load_x();
#endif
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
      int cur_hb = 0;  // cluster mode: h buffer of the current FC2 expert (0, 1: global path; 2, 3: receive)
      uint32_t xmask = 0;
      long long w_full = 0, w_acc = 0, w_h = 0, w_x = 0, t_issue = 0, n_st = 0;
      // Loop-invariant bases; per-slot / per-ks / per-acc operands are base + small offsets.
      const uint64_t adesc0 = tc::desc_sw128(tc::su32(sm + kOffA));          // + slot * 32 KB / 16
      const uint64_t sfa_src0 = desc_sf(tc::su32(sm + kOffSFA));             // + slot * 1 KB / 16
      const uint64_t xdesc0 = tc::desc_sw128(tc::su32(sm + kOffXq));         // + ks * 2 KB / 16
      const uint64_t xsf_src0 = desc_sf(tc::su32(sm + kOffXsf));             // + ks * 256 B / 16
      const uint64_t hdesc0 = tc::desc_sw128(tc::su32(sm + kOffHq));         // + b * 2 KB / 16
      const uint64_t hsf_src0 = desc_sf(tc::su32(sm + kOffHsf));             // + b * 640 B / 16
      const uint64_t hdescR = tc::desc_sw128(tc::su32(sm + kOffHr));         // cluster receive buffers
      const uint64_t hsfR = desc_sf(tc::su32(sm + kOffHrsf));
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
        if (kTraceOn && n_st == 0 && lane == 0) trace_ev(p, 9);
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
            if constexpr (kClu) {
              if (cur_u >= 0 && cur_hb < 2) tc::mma_commit(&ms.hempty[cur_hb]);
            } else {
              if (cur_u >= 0) tc::mma_commit(&ms.hempty[hord & 1]);
            }
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
          if (kTraceOn && n_st == 0 && lane == 0) trace_ev(p, 10);
          if (last) acc_release(m);
        } else if constexpr (kClu) {
          uint32_t copy_sfb = 0, bar_hrel = 0;
          if (m.a != cur_u) {  // new expert: release the previous h buffer, wait for this one
            if (cur_u >= 0 && cur_hb < 2) bar_hrel = bar_hempty0 + cur_hb * 8;  // receive buffers: single use
            cur_u = m.a;
            const long long c1 = PCLK();
            if (m.c >= 2) {  // cluster-local expert: receive buffer j = m.c - 2 (single use per call)
              cur_hb = m.c;
              const int j = m.c - 2;
              if (kTraceOn && lane == 0) trace_ctr(p, 56 + j, static_cast<long long>(tc::globaltimer()));
#ifdef MOE8_CL_HLOADER  // the h loader waits for the DSMEM chunks and builds the SFB images (one more hop)
              tc::mbar_wait(&ms.hfullR[j], 0);
#else
              // wait for the owners' DSMEM chunks here and build the SFB images (rows t: scales of token t for
              // k-blocks 4i..4i+3, words replicated) from the pushed scale words
              clx::wait(&ms.hrecv[j], 0);
              if (lane < 8) {
                const uint32_t* sc = reinterpret_cast<const uint32_t*>(sm + kOffHsc + j * 128);
                const uint32_t w0 = (sc[lane] & 0xffffu) | (sc[8 + lane] << 16), w1 = sc[16 + lane] & 0xffffu;
                *reinterpret_cast<uint4*>(sm + kOffHrsf + j * 640 + lane * 16) = make_uint4(w0, w0, w0, w0);
                *reinterpret_cast<uint4*>(sm + kOffHrsf + j * 640 + 128 + lane * 16) = make_uint4(w1, w1, w1, w1);
              }
              __syncwarp();
#endif
              if (kTraceOn && lane == 0) {
                trace_ctr(p, 58 + j, static_cast<long long>(tc::globaltimer()));
                trace_ctr(p, 60 + j, m.a);
              }
              tc::fence_proxy_async_smem();  // h image written by remote st.async, SFB images by this warp
            } else {
              ++hord;
              cur_hb = hord & 1;
              if (lane == 0 && hord < 8) trace_ctr(p, 32 + hord, static_cast<long long>(tc::globaltimer()));
              tc::mbar_wait(&ms.hfull[cur_hb], (hord >> 1) & 1);
              if (lane == 0 && hord < 8) {
                trace_ctr(p, 40 + hord, static_cast<long long>(tc::globaltimer()));
                trace_ctr(p, 48 + hord, m.a);
              }
            }
            tc::tc_fence_after();
            w_h += PCLK() - c1;
            if (hord == 0 && lane == 0) trace_ev(p, 11);
            copy_sfb = 1;
          }
          const bool rb = cur_hb >= 2;
          const uint64_t hd = rb ? hdescR + (cur_hb - 2) * 128 : hdesc0 + cur_hb * 128;
          const uint64_t hs = rb ? hsfR + (cur_hb - 2) * 40 : hsf_src0 + cur_hb * 40;
          acc_acquire();
          tc::fc2_stage(tmem + kColAcc + acc * 8, adesc0 + slot * 2048, hd, idesc, tmem + kColSFA + slot * 8,
                        sfa_src0 + slot * 64, tmem + kColSFBh + cur_hb * 8, hs, copy_sfb, bar_hrel,
                        bar_empty0 + slot * 8, bar_acc0 + acc * 8);
          __syncwarp();
          acc_release(m);
        } else {
          uint32_t copy_sfb = 0, bar_hrel = 0;
          if (m.a != cur_u) {  // new expert: release the previous h buffer, wait for this one
            if (cur_u >= 0) bar_hrel = bar_hempty0 + (hord & 1) * 8;
            cur_u = m.a;
            ++hord;
            const long long c1 = PCLK();
            if (lane == 0 && hord < 8) trace_ctr(p, 32 + hord, static_cast<long long>(tc::globaltimer()));
            tc::mbar_wait(&ms.hfull[hord & 1], (hord >> 1) & 1);
            if (lane == 0 && hord < 8) {
              trace_ctr(p, 40 + hord, static_cast<long long>(tc::globaltimer()));
              trace_ctr(p, 48 + hord, m.a);
            }
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
      if constexpr (kClu) {
        if (loc1 && lane == 0 && f2hi[1] > f2lo[1]) {  // arm the DSMEM h receive barriers (bytes may be arriving)
          const int u0 = f2lo[1] / kFc2Tiles, u1 = (f2hi[1] - 1) / kFc2Tiles;
          if (u1 - u0 > 1) asm volatile("trap;");
          for (int u = u0; u <= u1; ++u) tc::mbar_expect_tx(&ms.hrecv[u - u0], 3 * M * (4 * 16 + 4));
        }
      }
      for (int g = 0; g < 2; ++g) {
        if (f2hi[g] <= f2lo[g]) continue;
        const int u0 = f2lo[g] / kFc2Tiles, u1 = (f2hi[g] - 1) / kFc2Tiles;
        for (int u = u0; u <= u1; ++u) {
          if constexpr (kClu) {
            if (g == 1 && loc1) {  // cluster-local expert: owners pushed the image + scale words (DSMEM)
#ifndef MOE8_CL_HLOADER
              continue;  // the MMA warp waits for it itself
#endif
              const int j = u - u0;
#ifndef MOE8_DBG_NO_BWAIT  // timing bound only (wrong results, and remote stores may land after exit)
              clx::wait(&ms.hrecv[j], 0);
#endif
              if (lane < 8) {  // SFB images: row t = scales of token t for k-blocks 4i..4i+3 (words replicated)
                const uint32_t* sc = reinterpret_cast<const uint32_t*>(sm + kOffHsc + j * 128);
                const uint32_t w0 = (sc[lane] & 0xffffu) | (sc[8 + lane] << 16), w1 = sc[16 + lane] & 0xffffu;
                *reinterpret_cast<uint4*>(sm + kOffHrsf + j * 640 + lane * 16) = make_uint4(w0, w0, w0, w0);
                *reinterpret_cast<uint4*>(sm + kOffHrsf + j * 640 + 128 + lane * 16) = make_uint4(w1, w1, w1, w1);
              }
              tc::fence_proxy_async_smem();
              __syncwarp();
              if (lane == 0) tc::mbar_arrive(&ms.hfullR[j]);
              continue;
            }
          }
          ++k;
          const int b = k & 1;
          if (k >= 2) tc::mbar_wait(&ms.hempty[b], ((k >> 1) - 1) & 1);
          const uint8_t* src = hset + static_cast<long>(u) * kHImg;
          if (lane == 0 && k < 8) trace_ctr(p, 16 + k, static_cast<long long>(tc::globaltimer()));
          uint4 q[4];
          uint32_t s4;
#ifdef MOE8_DBG_NO_HPOLL  // timing bound only (wrong results): h is "ready" as soon as a buffer is free
          for (int v = 0; v < 4; ++v) q[v] = make_uint4(0, 0, 0, 0);
          s4 = 0x7f7f7f7fu;
          if (false)
#endif
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
            __nanosleep(MOE8_HPOLL_NS);
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
            if (k < 8) trace_ctr(p, 24 + k, static_cast<long long>(tc::globaltimer()));
            tc::mbar_arrive(&ms.hfull[b]);
          }
        }
      }
    } else if (warp == 3) {
      // ================================================================ re-arm the idle h parity
      if constexpr (kClu) {
        // arm the DSMEM partial receive barrier: the tile holding my last group-1 stage, if I own it and it
        // continues into the next cluster CTAs' ranges (at most 2 remote pieces)
        if (loc1 && lane == 0 && f1hi[1] > f1lo[1]) {
          const int s0 = ((f1hi[1] - 1) / kFc1Stages) * kFc1Stages;
          if (s0 >= f1lo[1] && s0 + kFc1Stages > f1hi[1]) {
            int np = 0;
            for (int q = crank + 1; q < kCl; ++q) np += lo1c(q) < s0 + kFc1Stages ? 1 : 0;
            if (np > 2) asm volatile("trap;");
            if (np > 0) tc::mbar_expect_tx(&ms.precv, np * 128 * MT * 4);
          }
        }
      }
      uint4* other = reinterpret_cast<uint4*>(p.ws + kWsH + static_cast<long>(P ^ 1) * kMaxPairs * kHImg);
      const int n16 = kMaxPairs * kHImg / 16;
      const int lo = split_lo(cta, n16, G), hi = split_lo(cta + 1, n16, G);
      const uint4 ff = make_uint4(~0u, ~0u, ~0u, ~0u);
      for (int i = lo + lane; i < hi; i += 32) other[i] = ff;
#ifdef MOE8_PF_PARTIAL
      // ================================================================ prefetch remote FC1 partials
      // This CTA owns at most one tile per group that continues into the next CTAs' ranges (its last piece).
      // Those CTAs publish their pieces at the START of their ranges, long before this CTA's owner piece is
      // done; fetch them into smem now so the owner epilogue does not pay 1-2 loaded L2 round trips.
      float* pbase_pf = reinterpret_cast<float*>(p.ws + kWsP);
      for (int g = 0; g < (pf_on ? 2 : 0); ++g) {
        if (f1hi[g] <= f1lo[g]) continue;
        const int s0 = ((f1hi[g] - 1) / kFc1Stages) * kFc1Stages;  // first stage of the tile holding my last stage
        if (s0 < f1lo[g] || s0 + kFc1Stages <= f1hi[g]) continue;   // not an owner, or the tile ends in my range
        int i = 0;
        for (int c2 = cta + 1; c2 < G && i < kPfSlots && split_lo(c2, S1g[g], G) < s0 + kFc1Stages; ++c2) {
          if (split_lo(c2 + 1, S1g[g], G) == split_lo(c2, S1g[g], G)) continue;  // empty range: no piece
          uint4* src = reinterpret_cast<uint4*>(pbase_pf + (static_cast<long>(g) * kMaxG + c2) * 1024);
          uint4 v[8];
          for (;;) {
            bool bad = false;
#pragma unroll
            for (int q = 0; q < 8; ++q) {
              const int f = lane + 32 * q;  // float4 index: token f / 32, rows 4 (f % 32) .. +3
              const int t = f >> 5;
              v[q] = t < M ? ld_volatile_v4(src + f) : make_uint4(0, 0, 0, 0);
              bad |= (v[q].x == 0xffffffffu) | (v[q].y == 0xffffffffu) | (v[q].z == 0xffffffffu) |
                     (v[q].w == 0xffffffffu);
            }
            if (!__any_sync(0xffffffffu, bad)) break;
            __nanosleep(64);
          }
          uint4* dst = reinterpret_cast<uint4*>(sm + kOffPf + (g * kPfSlots + i) * (M * 512));
          const uint4 ff4 = make_uint4(~0u, ~0u, ~0u, ~0u);
#pragma unroll
          for (int q = 0; q < 8; ++q) {
            const int f = lane + 32 * q;
            if ((f >> 5) < M) {
              dst[f] = v[q];
              src[f] = ff4;  // re-arm (the owner no longer touches this slot)
            }
          }
          __syncwarp();
          if (lane == 0) tc::mbar_arrive(&ms.pfbar[g][i]);
          ++i;
        }
      }
#endif
    } else if (warp >= 4) {
      // ================================================================ epilogue (two groups of 4 warps;
      // group eg handles accumulators eg, eg + 2, ...; ew = TMEM lane quadrant)
      const int ew = warp & 3, eg = (warp - 4) >> 2;
      // cluster mode: only the epilogue warps store to peers; each waits once on the cluster barrier (peers'
      // mbarrier inits visible) before its first remote store
      bool clw = false;
      auto cl_ready = [&]() {
        if (!clw) {
          clx::wait_acquire();
          clw = true;
        }
      };
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
        const long long t_accw = kTraceOn ? static_cast<long long>(tc::globaltimer()) : 0ll;
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
          const bool locg = kClu && g == 1 && loc1;  // cluster-local expert (warp-uniform)

          if (s0 < f1lo[g]) {  // not the owner: publish the partial
            if constexpr (kClu) {
              if (locg) {  // DSMEM: [row][MT] fp32 into the owner's receive slot (cluster CTAs are consecutive)
                cl_ready();
                int qo = crank - 1;
                while (qo > 0 && lo1c(qo) > s0) --qo;
                const uint32_t la = tc::su32(sm + kOffPrecv + (crank - qo - 1) * (128 * MT * 4) + row * (MT * 4));
                const uint32_t ra = clx::mapa(la, qo), rbar = clx::mapa(tc::su32(&ms.precv), qo);
#pragma unroll
                for (int q4 = 0; q4 < MT / 4; ++q4)
                  clx::st_async_v4(ra + 16 * q4,
                                   make_uint4(__float_as_uint(v[4 * q4]), __float_as_uint(v[4 * q4 + 1]),
                                              __float_as_uint(v[4 * q4 + 2]), __float_as_uint(v[4 * q4 + 3])),
                                   rbar);
                continue;
              }
            }
            float* dst = pbase + (static_cast<long>(g) * kMaxG + cta) * 1024 + row;
#pragma unroll
            for (int t = 0; t < MT; ++t)
              if (t < M) dst[t * 128] = v[t];
            continue;
          }
          if (eg == 0 && ew == 0 && lane == 0 && kTraceOn && p.trace != nullptr && p.trace[cta * 128 + 17] == 0) trace_ev(p, 17);
          if (ew == 0 && lane == 0 && T < 400) trace_at(p, T, static_cast<long long>(tc::globaltimer()));
          if (kTraceOn && lane == 0) {
            trace_w(p, T, ew, 5, t_accw);
            trace_w(p, T, ew, 0, static_cast<long long>(tc::globaltimer()));
          }
          // Other pieces of this tile: CTAs cta+1.. whose ranges start inside the tile.
          int ipf = 0;
          bool gpoll = true;
          if constexpr (kClu) {
            if (locg) {  // DSMEM pieces (at most 2: cluster CTAs crank+1.. whose ranges start inside the tile)
              gpoll = false;
              int np = 0;
              for (int q = crank + 1; q < kCl; ++q) np += lo1c(q) < s0 + kFc1Stages ? 1 : 0;
              if (np > 0) {
#ifndef MOE8_DBG_NO_BWAIT
                clx::wait(&ms.precv, 0);
#endif
#pragma unroll 1
                for (int i = 0; i < np; ++i) {  // (np <= 2; unrolled, this loop was 1.9 KB of SASS)
                  const float* pr = reinterpret_cast<const float*>(sm + kOffPrecv + i * (128 * MT * 4)) + row * MT;
#pragma unroll
                  for (int t = 0; t < MT; ++t) v[t] += pr[t];
                }
              }
            }
          }
#ifdef MOE8_DBG_NO_PPOLL  // timing bound only (wrong results): the owner does not wait for remote partials
          gpoll = false;
#endif
          // opt-in, MT = 8 only: measured M = 8 27.9 -> 27.7 us but M = 5..7 +0.2..0.4 (even when gated off at run
          // time: code layout), and +0.05..0.15 us at M <= 4
          constexpr bool kPlook = MOE8_PLOOK != 0 && MT == 8;
          float nv[MT];
          int nc = -1;
          if (gpoll)
          for (int c2 = cta + 1; c2 < G && split_lo(c2, S1g[g], G) < s0 + kFc1Stages; ++c2) {
            if (split_lo(c2 + 1, S1g[g], G) == split_lo(c2, S1g[g], G)) continue;  // empty range: no piece
#ifdef MOE8_PF_PARTIAL
            if (pf_on && ipf < kPfSlots) {  // prefetched by warp 3 (and already re-armed there)
              tc::mbar_wait(&ms.pfbar[g][ipf], 0);
              const float* pf = reinterpret_cast<const float*>(sm + kOffPf + (g * kPfSlots + ipf) * (M * 512)) + row;
#pragma unroll
              for (int t = 0; t < MT; ++t)
                if (t < M) v[t] += pf[t * 128];
              ++ipf;
              continue;
            }
#endif
            float* src = pbase + (static_cast<long>(g) * kMaxG + c2) * 1024 + row;
            float pv[8];
            bool okp = false;
            // look-ahead: the next CTA's slot was loaded together with this piece's first poll (a CTA publishes
            // its piece of this tile at the start of its range, long before the owner gets here)
            if (kPlook && nc == c2) {
              bool bad = false;
#pragma unroll
              for (int t = 0; t < MT; ++t) bad |= t < M && __float_as_uint(nv[t]) == 0xffffffffu;
              okp = !__any_sync(0xffffffffu, bad);
              if (okp) {
#pragma unroll
                for (int t = 0; t < MT; ++t) pv[t] = nv[t];
              }
            }
            if (kPlook && c2 + 1 < G) {
              nc = c2 + 1;
              const float* ns = pbase + (static_cast<long>(g) * kMaxG + nc) * 1024 + row;
#pragma unroll
              for (int t = 0; t < MT; ++t) nv[t] = t < M ? __uint_as_float(ld_volatile_u32(ns + t * 128)) : 0.f;
            }
            if (!okp)
            for (;;) {
              bool bad = false;
#pragma unroll
              for (int t = 0; t < MT; ++t) {
                pv[t] = 0.f;
                if (t < M) {
                  const uint32_t w = ld_volatile_u32(src + t * 128);
                  bad |= (w == 0xffffffffu);
                  pv[t] = __uint_as_float(w);
                }
              }
              if (!__any_sync(0xffffffffu, bad)) break;
              if (MOE8_PPOLL_NS > 0) __nanosleep(MOE8_PPOLL_NS);
            }
#pragma unroll
            for (int t = 0; t < MT; ++t)
              if (t < M) {
                v[t] += pv[t];
                src[t * 128] = __uint_as_float(0xffffffffu);
              }
          }
          if (eg == 0 && ew == 0 && lane == 0) trace_ev(p, 24);
          if (ew == 0 && lane == 0 && T < 400) trace_at(p, 400 + T, static_cast<long long>(tc::globaltimer()));
          if (kTraceOn && lane == 0) trace_w(p, T, ew, 1, static_cast<long long>(tc::globaltimer()));
          const int u = T / 3, r = T % 3;
          float h[MT], am[MT];
#pragma unroll
          for (int t = 0; t < MT; ++t) {
            const float o = __shfl_xor_sync(0xffffffffu, v[t], 8);  // the gate row sits 8 lanes up
            const float hv = p.beta_lb * tanh_fast(o * p.inv_beta) * sigmoid_fast(o) * tanh_fast(v[t] * p.inv_lb);
            h[t] = up_lane ? hv : 0.f;
            am[t] = fabsf(h[t]);
          }
#pragma unroll
          for (int t = 0; t < MT; ++t) {
#pragma unroll
            for (int off = 1; off < 32; off <<= 1) am[t] = fmaxf(am[t], __shfl_xor_sync(0xffffffffu, am[t], off));
          }
          if (lane == 0) {
#pragma unroll
            for (int t = 0; t < MT; ++t) ms.amax[eg][ew][t] = am[t];
          }
          asm volatile("bar.sync %0, 128;" ::"r"(ebar) : "memory");
          if (kTraceOn && lane == 0) trace_w(p, T, ew, 2, static_cast<long long>(tc::globaltimer()));
          int sbt = 0;
#pragma unroll
          for (int t = 0; t < MT; ++t) {
            const int sb = mx_exp(fmaxf(am[t], ms.amax[eg][ew ^ 1][t]));
            if (lane == t) sbt = sb;
            if (up_lane)
              ms.hst[eg][ew][t][i_local] =
                  static_cast<uint8_t>(__nv_cvt_float_to_fp8(h[t] * mx_rescale(sb), __NV_SATFINITE, __NV_E4M3));
          }
          __syncwarp();
          if constexpr (kClu) {
            if (locg) {  // push this 64-neuron chunk (+ its 2 scale bytes per token) to every consumer in the cluster
              cl_ready();
              if (lane < M) {
                const int t = lane, kc = 4 * r + ew;
                const uint4 val = *reinterpret_cast<const uint4*>(ms.hst[eg][ew][t]);
                uint32_t sw = 0;
                if (ew == 0)
                  sw = static_cast<uint32_t>(mx_exp(fmaxf(ms.amax[eg][0][t], ms.amax[eg][1][t]))) |
                       (static_cast<uint32_t>(mx_exp(fmaxf(ms.amax[eg][2][t], ms.amax[eg][3][t]))) << 8);
                const uint32_t la = tc::su32(sm + kOffHr + (kc / 8) * 1024 + t * 128 + (((kc % 8) ^ t) * 16));
                const uint32_t ls = tc::su32(sm + kOffHsc + r * 32 + t * 4);
                const uint32_t lb = tc::su32(&ms.hrecv[0]);
#pragma unroll 1
                for (int q = 0; q < kCl; ++q) {
                  const int lo = lo2c(q), hi = lo2c(q + 1);
                  if (lo < kFc2Tiles * (u + 1) && hi > kFc2Tiles * u) {
                    const int jq = u - lo / kFc2Tiles;
                    const uint32_t rbar = clx::mapa(lb + 8 * jq, q);
                    clx::st_async_v4(clx::mapa(la + jq * 2048, q), val, rbar);
                    if (ew == 0) clx::st_async_b32(clx::mapa(ls + jq * 128, q), sw, rbar);
                  }
                }
              }
              if (kTraceOn && lane == 0) trace_w(p, T, ew, 3, static_cast<long long>(tc::globaltimer()));
              asm volatile("bar.sync %0, 128;" ::"r"(ebar) : "memory");  // ms.amax / hst reuse
              if (kTraceOn && lane == 0) trace_w(p, T, ew, 4, static_cast<long long>(tc::globaltimer()));
              if (ew == 0 && lane == 0 && T < 400) trace_at(p, 800 + T, static_cast<long long>(tc::globaltimer()));
              continue;
            }
          }
          uint8_t* himg = hset + static_cast<long>(u) * kHImg;
          if (lane < 8) {  // all 8 rows are written (the FC2 side validates every row); rows t >= MT are 0
            const int t = lane, kc = 4 * r + ew;
            const uint4 val = t < MT ? *reinterpret_cast<const uint4*>(ms.hst[eg][ew][t]) : make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(himg + (kc / 8) * 1024 + t * 128 + (((kc % 8) ^ t) * 16)) = val;
            if ((ew & 1) == 0) himg[2048 + t * 16 + 2 * r + ew / 2] = static_cast<uint8_t>(sbt);
          }
          if (kTraceOn && lane == 0) trace_w(p, T, ew, 3, static_cast<long long>(tc::globaltimer()));
          asm volatile("bar.sync %0, 128;" ::"r"(ebar) : "memory");  // ms.amax / hst reuse
          if (kTraceOn && lane == 0) trace_w(p, T, ew, 4, static_cast<long long>(tc::globaltimer()));
          if (eg == 0 && ew == 0 && lane == 0) trace_ev(p, 18);
          if (ew == 0 && lane == 0 && T < 400) trace_at(p, 800 + T, static_cast<long long>(tc::globaltimer()));
        } else {
          const int u = m.a, mt = m.b;
          // physical row p -> logical output row 32*(p/32) + 4*(p%8) + (p%32)/8; lanes l and l+8 hold
          // consecutive logical rows, stored as one bf16x2.
          const int n = mt * 128 + 32 * ew + 4 * (lane % 8) + lane / 8;
          const uint2 ts = *reinterpret_cast<const uint2*>(ms.tokslot[u]);
#pragma unroll
          for (int t = 0; t < MT; ++t) {
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
    // A smaller grid (cluster mode: 144 of 152 SMs) also advances the counters of the CTAs it does not launch,
    // so all per-CTA call counters (the h-image parity) stay in lockstep across calls with different grids.
    if (cta < kMaxG - G) ctr[G + cta] = ms.parity + 1;  // (G >= kMaxG / 2)
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
  static int sms = 0, gcl = 0;
  if (sms == 0) {
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    C10_CUDA_CHECK(cudaFuncSetAttribute(moe8_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes<1>()));
    C10_CUDA_CHECK(cudaFuncSetAttribute(moe8_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes<2>()));
    C10_CUDA_CHECK(cudaFuncSetAttribute(moe8_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes<4>()));
    C10_CUDA_CHECK(cudaFuncSetAttribute(moe8_kernel<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes<8>()));
    // cluster-mode grid: as many kCl-CTA clusters as fit co-resident (36 on GB200 at ~220 KB smem: 144 CTAs)
    cudaLaunchConfig_t oc{};
    oc.gridDim = dim3(kCl * (sms / kCl));
    oc.blockDim = dim3(kThreads);
    oc.dynamicSmemBytes = smem_bytes<4>();
    cudaLaunchAttribute oa[1];
    oa[0].id = cudaLaunchAttributeClusterDimension;
    oa[0].val.clusterDim.x = kCl;
    oa[0].val.clusterDim.y = 1;
    oa[0].val.clusterDim.z = 1;
    oc.attrs = oa;
    oc.numAttrs = 1;
    int ncl = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&ncl, moe8_kernel<4>, &oc));
#ifdef MOE8_CL_NCL  // experiment: fewer clusters
    ncl = std::min(ncl, MOE8_CL_NCL);
#endif
    gcl = kCl * std::min(ncl, sms / kCl);
    TORCH_CHECK(gcl >= kCl && 2 * gcl >= kMaxG, "moe8: too few co-resident clusters");
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
#ifndef MOE8_FC2_PROMO
#define MOE8_FC2_PROMO CU_TENSOR_MAP_L2_PROMOTION_NONE
#endif
#ifndef MOE8_FC1_PROMO
#define MOE8_FC1_PROMO CU_TENSOR_MAP_L2_PROMOTION_L2_256B
#endif
    me.a1 = fp4_map(w13.data_ptr(), E * kW13Rows, kW13RowBytes, 28, 64, MOE8_FC1_PROMO);
    // W2: K-chunks at k = 0 and k = 64 (32-B chunk stride), so bytes 96..127 (K padding) are never read.
#ifndef MOE8_FC2_PROMO
#define MOE8_FC2_PROMO CU_TENSOR_MAP_L2_PROMOTION_NONE
#endif
#ifndef MOE8_FC1_PROMO
#define MOE8_FC1_PROMO CU_TENSOR_MAP_L2_PROMOTION_L2_256B
#endif
    me.a2 = fp4_map(w2.data_ptr(), E * kHidden, 128, 2, 32, MOE8_FC2_PROMO);
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
#ifdef MOE8_PF_PARTIAL
  // prefetch buffers sized by M (fp32 [M][128] per slot): a smaller smem request leaves more L1 at small M
  cfg.dynamicSmemBytes = kOffPf + 2 * kPfSlots * M * 512;
#else
  cfg.dynamicSmemBytes = kSmemBytes;
#endif
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
#ifdef MOE8_NO_PDL  // experiment: plain stream order
  attr[0].val.programmaticStreamSerializationAllowed = 0;
#else
  attr[0].val.programmaticStreamSerializationAllowed = 1;
#endif
  attr[1].id = cudaLaunchAttributeClusterDimension;
  attr[1].val.clusterDim.x = kCl;
  attr[1].val.clusterDim.y = 1;
  attr[1].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
#ifdef MOE8_GRID_CAP  // experiment: cap the non-cluster grid
  cfg.gridDim = dim3(std::min(sms, MOE8_GRID_CAP));
#endif
  const int MT = M == 1 ? 1 : M == 2 ? 2 : M <= 4 ? 4 : 8;
  const bool clu = MT == 4 ? cl_mode<4>() : MT == 8 ? cl_mode<8>() : false;
#ifndef MOE8_SMALLM_GRID
#define MOE8_SMALLM_GRID 1
#endif
  // M <= 2: the same 144-CTA grid without clusters (measured M = 1 10.2 -> 9.7 us, M = 2 12.47 -> 12.35; M = 8 would
  // lose 0.2 us, so it keeps all SMs). The 8 idle SMs also let a small co-running kernel start at once.
  if (MOE8_SMALLM_GRID && MT <= 2) cfg.gridDim = dim3(std::min(sms, gcl));
  if (clu) {
    cfg.gridDim = dim3(gcl);
    cfg.numAttrs = 2;
    cfg.dynamicSmemBytes = MT == 4 ? smem_bytes<4>() : smem_bytes<8>();
  }
  if (MT == 1) C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, moe8_kernel<1>, mc->a1, mc->a2, prm));
  else if (MT == 2) C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, moe8_kernel<2>, mc->a1, mc->a2, prm));
  else if (MT == 4) C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, moe8_kernel<4>, mc->a1, mc->a2, prm));
  else C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, moe8_kernel<8>, mc->a1, mc->a2, prm));
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
