// moebody_dev.cuh (W1-4): device code of the barrier-free routed-expert body for Kimi K3 decode, M <= 8 tokens, sm_100a.
// x (bf16) -> MXFP8, FC1 -> SiTU -> MXFP8 h -> FC2 on tcgen05 (kind::mxf8f6f4: MXFP4 weights x MXFP8
// activations, swap-AB 128x8x32 MMAs, TMA, TMEM), weights read in place in vLLM's TRT-LLM MXFP4 layout
// (see agents/moe8/moe8.cu for the layout and the MMA / scale-factor conventions reused here).
//
// Expert-per-group schedule (no grid barrier, no global hand-off):
//   The grid is launched as clusters of L CTAs (L = 6 / 4 / 2 for M = 1 / 2 / >= 3). Once routing is
//   known every CTA derives the same D (distinct experts; no dedupe pass at M <= 2: the 16M (token, slot)
//   pairs are the experts) and picks the group size C (largest divisor of L, <= 6, with #groups >= D).
//   Group g (C CTAs of one cluster) owns expert g: FC1's 42 (tile, K-stage) units and FC2's 28 row
//   tiles are split contiguously over the C CTAs, so every CTA streams ~1/C of one expert.
//   FC1 tile r (128 rows = neurons 64r..64r+63, gate+up) is split over at most two CTAs. The CTA that
//   holds the tile's first K-stage owns it; the other one sends its fp32 partial (128 x 8) over DSMEM
//   (st.async + mbarrier complete_tx). The owner adds it, applies SiTU, quantizes h to MXFP8 and
//   broadcasts the 64-neuron chunk (FC2 B-operand image + per-tile scale word) over DSMEM to all C
//   CTAs of the group. FC2 starts when a CTA's h barrier has received all three chunks.
//   All weight TMAs of a CTA are issued as soon as ring slots allow (5 x 32 KB); FC2 weights stream
//   while FC1 finishes.
//   C = 1 (large D) degenerates to one expert per CTA: h never leaves the CTA.
// Warp roles (384 threads): w0 TMA producer, w1 MMA issuer, w3 TMEM owner, w4..7 / w8..11 two epilogue
// groups (alternate accumulators). All warps quantize x first.
//
// Routing input: `ids` [M,16] int32 in global memory, read after griddepcontrol.wait, or - flag mode
// (persistent harness) - after an epoch counter reaches this CTA's call count.
// Device-side API (no torch dependency): moebody::group_body<MT>(), moebody::sk_body(), moebody::Params.
// moebody.cu wraps them as kernels and registers the torch op k3moebody::expert_body.
#pragma once
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "tc_utils.cuh"
#include "body_mma.cuh"
#include "cluster_util.cuh"

namespace moebody {

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
constexpr int kFc1Seq = kFc1Stages * kFc1Tiles;  // 42
constexpr int kFc2Tiles = 28;                   // 3584 = 28 x 128
constexpr int kMaxE = 4096;
constexpr int kMaxC = 6;
constexpr int kMaxG = 160;
// Workspace (bytes; everything 0xFF-initialized once, never cleared again):
//   [kMaxG] inverted per-CTA call counters | 4 KB scratch (warm-up stores) |
//   stream-K: h images [2 parities][128][kHImg], partials [kSkMaxNG][kMaxG][8][128] fp32 |
//   modes 1/2: tagged gemm2 staging [2 parities][8][16][3584] bf16, arrival counters [8][28][4] int
constexpr int kSkMaxNG = 3;
constexpr int kHImg = 2048 + 128;  // per expert: FC2 B image (8 rows x 192 e4m3, SW128) + compact scales
constexpr long kWsScratch = kMaxG * 4;
constexpr long kWsSkH = 4864;
constexpr long kWsSkP = kWsSkH + 2L * kMaxPairs * kHImg;
constexpr long kWsG2 = kWsSkP + static_cast<long>(kSkMaxNG) * kMaxG * 8 * 128 * 4;
constexpr long kWsG2Parity = static_cast<long>(kMaxPairs) * kHidden * 2;
constexpr long kWsCnt = kWsG2 + 2 * kWsG2Parity;
constexpr long kWsAll = kWsCnt + kMaxM * 28 * 4 * 4;

#ifndef BODY_SLOTS
#define BODY_SLOTS 5
#endif
constexpr int kSlots = BODY_SLOTS;
constexpr int kAcc = 16;
constexpr int kThreads = 384;

// smem layout (offsets from a 1024-aligned base)
constexpr int kOffA = 0;                          // kSlots x 32 KB weights (unpacked fp4)
constexpr int kOffSFA = kOffA + kSlots * 32768;   // kSlots x 1 KB weight scales
constexpr int kOffXq = kOffSFA + kSlots * 1024;   // 28 KB x (e4m3), FC1 B operand
constexpr int kOffHq = kOffXq + 28672;            // 2 x 2 KB h (e4m3), FC2 B operand
constexpr int kOffXsf = kOffHq + 2 * 2048;        // 28 x 128 (+384) SFB images of x
constexpr int kOffHsf = kOffXsf + 28 * 128 + 384;  // 2 x (3 x 128 + 384) SFB images of h (one per FC1 tile)
constexpr int kOffPrecv = kOffHsf + 2 * 768;      // 3 x 128 rows x 8 fp32 partials received
constexpr int kOffMisc = kOffPrecv + kFc1Tiles * 128 * 8 * 4;  // (sized for 8 tokens)
constexpr int kSmemBytes = kOffMisc + 5120;
static_assert(kSmemBytes <= 232448, "smem");
static_assert(kOffXq % 1024 == 0 && kOffHq % 1024 == 0, "SW128 operands must be 1024-aligned");

// TMEM columns (scale-factor bases are multiples of 4)
constexpr uint32_t kTmemCols = 512;
constexpr uint32_t kColAcc = 0;                       // kAcc x 8
constexpr uint32_t kColSFA = 128;                     // kSlots x 8
constexpr uint32_t kColSFBh = kColSFA + kSlots * 8;   // 2 buffers x 3 tiles x 4
constexpr uint32_t kColSFBx = 256;                    // 28 x 4
static_assert(kAcc * 8 <= 128 && kColSFBh + 24 <= kColSFBx && kColSFBh % 4 == 0, "tmem");

enum : int { kFC1 = 0, kFC2 = 1, kEnd = 2 };
// FC1 flags (Meta::c); bits 8..15: cluster rank of the tile owner (non-owner pieces); bits 16..23: K-stage
enum : int { fFirst = 1, fLast = 2, fOwner = 4, fRemote = 8 };

struct Meta {
  int type, a, b, c;  // FC1: a = local expert i, b = tile r, c = flags; FC2: a = i, b = mt, c = expert slot u
};

struct Misc {
  uint64_t full[kSlots], empty[kSlots];
  uint64_t accfull[kAcc], accempty[kAcc];
  uint64_t hfull[2], hempty[2];
  uint64_t precv[kFc1Tiles];
  uint64_t warmbar;  // sink for warm-up DSMEM stores (never waited on)
  Meta meta[kSlots];
  Meta accmeta[kAcc];
  uint32_t tmem_base;
  int D, C, ngroups, grp, grank, rank0, nloc;
  int wcnt[4];
  float amax[2][4][8];
  int expert[kMaxPairs];
  int uof[kMaxPairs];
  alignas(16) signed char tokslot[kMaxPairs][kMaxM];
  alignas(16) uint8_t hst[2][4][8][16];
  int call;        // this CTA's call index
  long long epi_poll[2];  // trace builds: epilogue cycles spent polling partials (stream-K)
  int skgeo[24];   // SkGeo (stream-K kernel)
};
static_assert(sizeof(Misc) <= 5120, "misc too big");

struct Params {
  const __nv_bfloat16* x;
  const int* ids;
  const uint8_t* w13s;
  const uint8_t* w2s;
  const uint8_t* w2;
  __nv_bfloat16* out;
  int M, E;
  float beta_lb, inv_beta, inv_lb;  // SiTU = beta*lb * tanh(g/beta) * sigmoid(g) * tanh(u/lb)
  const unsigned long long* flag;   // flag mode: wait until *flag >= ctr[cta] + 1, then read ids
  int* ctr;                         // [kMaxG] per-CTA call counters, stored inverted (workspace starts all-0xFF)
  void* scratch;                    // >= 256 B of global scratch (warm-up stores)
  unsigned long long* endlog;       // flag mode, optional: endlog[call % endlog_n] = max over CTAs of end time
  int endlog_n;
  unsigned long long* trace;
  int force_c;
  int sk_groups;  // stream-K kernel: expert groups (0 = auto)
  int mode;       // 0 gemm2, 1 finalized, 2 publish
  const int* ids_smem;  // routing source (c): generic pointer to int[M*16] in this CTA's shared memory
  uint32_t ids_bar;     // shared::cta address of the mbarrier the caller completes when the ids are written
  int ids_phase;        // parity of that completion (0 for the first use after mbarrier.init)
  int no_griddep;       // device-function use: the caller handles PDL
  int x_late;           // flag mode: x is published together with the ids (quantized after the flag wait)
  const void* wts;
  int wts_bf16;
  __nv_bfloat16* fout;
  long fstride;
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
__device__ __forceinline__ uint64_t desc_sf(uint32_t saddr) { return tc::smem_desc(saddr, 0, 128, 0); }
__device__ __forceinline__ int ld_relaxed_i32(const int* p) {
  int v;
  asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned long long ld_acquire_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

#ifdef BODY_TRACE
constexpr bool kTraceOn = true;
#else
constexpr bool kTraceOn = false;
#endif
// Trace points (%globaltimer + clock64) only exist in -DBODY_TRACE builds.
__device__ __forceinline__ void trace_ev(const Params& p, int ev) {
  if constexpr (kTraceOn) {
    if (p.trace != nullptr) {
      p.trace[blockIdx.x * 128 + ev] = tc::globaltimer();
      p.trace[blockIdx.x * 128 + 32 + ev] = clock64();
    }
  }
}
__device__ __forceinline__ void trace_ctr(const Params& p, int i, long long v) {
  if constexpr (kTraceOn) {
    if (p.trace != nullptr) p.trace[blockIdx.x * 128 + 64 + i] = v;
  }
}

#ifdef BODY_TRACE
#define PCLK() clock64()
#else
#define PCLK() 0ll
#endif

// Per-token-count sizes (MT = token capacity of the instantiation: 1, 2, 4 or 8).
template <int MT>
struct Sz {
  static constexpr uint32_t kHChunk = 4 * MT * 16 + MT * 4;  // one FC1 tile's h per destination CTA
  static constexpr uint32_t kH = kFc1Tiles * kHChunk;         // per expert and CTA
  static constexpr uint32_t kPart = 128 * MT * 4;             // one tile's fp32 partial
};

__device__ __forceinline__ void mbar_arrive_u32(uint32_t addr) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(addr) : "memory");
}
template <int MT>
__device__ __forceinline__ void tmem_ldn(uint32_t taddr, float (&v)[MT]) {
  uint32_t r[MT];
  if constexpr (MT == 1) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(r[0]) : "r"(taddr) : "memory");
  } else if constexpr (MT == 2) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(taddr) : "memory");
  } else if constexpr (MT == 4) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(taddr)
                 : "memory");
  } else {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
                 : "r"(taddr)
                 : "memory");
  }
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
  for (int i = 0; i < MT; ++i) v[i] = __uint_as_float(r[i]);
}

#ifdef BODY_TRACE
#define PCLK() clock64()
#else
#define PCLK() 0ll
#endif

// One (token, 32-element block) of x -> e4m3 into the FC1 B image and its UE8M0 scale into the SFB image.
__device__ __forceinline__ void quant_x_block(uint8_t* sm, const __nv_bfloat16* x, int t, int kb) {
  const uint4* src = reinterpret_cast<const uint4*>(x + t * kHidden + kb * 32);
  float f[32];
#pragma unroll
  for (int v = 0; v < 4; ++v) {
    const uint4 raw = src[v];
    const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
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
  for (int w = 0; w < 8; ++w) wv[w] = e4m3x4(f[4 * w] * rs, f[4 * w + 1] * rs, f[4 * w + 2] * rs, f[4 * w + 3] * rs);
  const int kc0 = 2 * kb, kc1 = kc0 + 1;
  uint8_t* xq = sm + kOffXq;
  *reinterpret_cast<uint4*>(xq + (kc0 / 8) * 1024 + t * 128 + (((kc0 % 8) ^ t) * 16)) = make_uint4(wv[0], wv[1], wv[2], wv[3]);
  *reinterpret_cast<uint4*>(xq + (kc1 / 8) * 1024 + t * 128 + (((kc1 % 8) ^ t) * 16)) = make_uint4(wv[4], wv[5], wv[6], wv[7]);
  // SFB image (k-group g = kb/4): row t byte kb%4
  sm[kOffXsf + (kb >> 2) * 128 + t * 16 + (kb & 3)] = static_cast<uint8_t>(sb);
}

// ---------------------------------------------------------------- code executed after the routing is known.
// Each role's per-item work is a __noinline__ function, executed once in a dry / warm-up mode before the
// routing arrives: its first real execution then hits a warm instruction cache (a cold first execution
// costs several us per role on GB200, the code having been evicted from L2 by the weight stream).

// Hot per-item functions: noinline (so a dry run warms exactly the code the real run executes) unless
// BODY_INL_* asks for inlining (A/B of the call overhead).
#ifndef BODY_NOINL_MMA  // measured: an out-of-line MMA stage call costs 0.8-2.8 us per call
#define MMA_FN __forceinline__
#else
#define MMA_FN __noinline__
#endif
#ifdef BODY_INL_TMA
#define TMA_FN __forceinline__
#else
#define TMA_FN __noinline__
#endif
#ifdef BODY_INL_EPI
#define EPI_FN __forceinline__
#else
#define EPI_FN __noinline__
#endif

__device__ __forceinline__ bool has_ff_byte(uint32_t w) {  // any byte == 0xFF
  const uint32_t x = ~w;
  return ((x - 0x01010101u) & ~x & 0x80808080u) != 0;
}
__device__ __forceinline__ uint4 ld_volatile_v4(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint32_t ld_volatile_u32(const void* p) {
  uint32_t v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ int split_lo(int c, int total, int G) {
  return static_cast<int>(static_cast<unsigned>(c * total) / static_cast<unsigned>(G));  // < 2^31
}

struct EpiConst {
  float beta_lb, inv_beta, inv_lb;
  __nv_bfloat16* out;         // mode 0: gemm2 rows [M*16, 3584]
  unsigned long long* trace;
  int mode;                   // 0 gemm2, 1 finalized, 2 publish
  uint32_t* g2;               // modes 1/2: tagged gemm2 staging [8][16][3584] bf16 (this call's parity), 0xFFFF = empty
  int* cnt;                   // modes 1/2: arrival counters [8][28][4]
  const void* wts;            // [M, 16] routing weights (bf16 or fp32)
  int wts_bf16;
  __nv_bfloat16* fout;        // modes 1/2: finalized rows, token t at fout + t * fstride
  long fstride;
};

// CTA-wide barrier of the body's 384 threads (named barrier 6; the caller may run extra warps).
__device__ __forceinline__ void body_sync() { asm volatile("bar.sync 6, 384;" ::: "memory"); }
// Barrier of the threads that are not the early producer (warps 1..11, named barrier 7) when `early`.
__device__ __forceinline__ void rest_sync(bool early) {
  if (early) asm volatile("bar.sync 7, 352;" ::: "memory");
  else body_sync();
}

// End of a call: advance the call counter of every possible CTA slot (not only the G that ran), so all
// counters stay equal whatever grid the next call uses (the grid depends on M and the kernel).
__device__ __forceinline__ void bump_counters(int* ctr, int cta, int G, int call) {
  for (int c2 = cta; c2 < kMaxG; c2 += G) ctr[c2] = ~(call + 1);
}

// Epilogue constants of one call (call index -> staging parity).
__device__ __forceinline__ EpiConst make_epi(const Params& p, int call) {
  EpiConst e;
  e.beta_lb = p.beta_lb;
  e.inv_beta = p.inv_beta;
  e.inv_lb = p.inv_lb;
  e.out = p.out;
  e.trace = p.trace;
  e.mode = p.mode;
  uint8_t* ws = reinterpret_cast<uint8_t*>(p.ctr);
  e.g2 = reinterpret_cast<uint32_t*>(ws + kWsG2 + (call & 1) * kWsG2Parity);
  e.cnt = reinterpret_cast<int*>(ws + kWsCnt);
  e.wts = p.wts;
  e.wts_bf16 = p.wts_bf16;
  e.fout = p.fout;
  e.fstride = p.fstride;
  return e;
}

// FC2 output of one epilogue item (this warp's 32 rows; lanes l, l+8 hold consecutive logical rows).
// mode 0: bf16 gemm2 rows. modes 1/2: the rows go to the tagged staging buffer (NaN canonicalized so 0xFFFF
// never occurs in data), then a relaxed arrival counter per (token, row tile, warp quadrant); the 16th
// arrival (one per routed slot j) reduces in fixed j order: out = bf16(sum_j fma(bf16 row_j, bf16 w_j)),
// validating the other rows by their tags (no fences: this CTA has TMA loads in flight), and re-arms them.
// Mode 2 also maps -0.0 to +0.0 (Lamport mailbox sentinel).
template <int MT>
__device__ __forceinline__ void fc2_emit(const float (&v)[MT], uint2 ts, int mt, int ew, int lane, const EpiConst& ec) {
  const int n = mt * 128 + 32 * ew + 4 * (lane % 8) + lane / 8;
#pragma unroll
  for (int t = 0; t < MT; ++t) {
    const int j = static_cast<signed char>(((t < 4 ? ts.x : ts.y) >> (8 * (t & 3))) & 0xff);
    if (j < 0) continue;  // warp-uniform
    const float o = __shfl_down_sync(0xffffffffu, v[t], 8);
    const long off = static_cast<long>(t * kTopK + j) * kHidden + n;
    if (ec.mode == 0) {  // gemm2 rows (warm-up: into scratch)
      if ((lane & 8) == 0)
        *reinterpret_cast<__nv_bfloat162*>(ec.out + off) = __floats2bfloat162_rn(v[t], o);
      continue;
    }
    if ((lane & 8) == 0) {
      __nv_bfloat162 b2 = __floats2bfloat162_rn(v[t], o);
      uint32_t w = *reinterpret_cast<uint32_t*>(&b2);
      if ((w & 0x7fffu) > 0x7f80u) w = (w & 0xffff0000u) | 0x7fc0u;  // NaN -> canonical NaN
      if ((w & 0x7fff0000u) > 0x7f800000u) w = (w & 0xffffu) | 0x7fc00000u;
      ec.g2[off >> 1] = w;
    }
    __syncwarp();
    int old = 0;
    if (lane == 0) old = atomicAdd(ec.cnt + (t * kFc2Tiles + mt) * 4 + ew, 1);
    old = __shfl_sync(0xffffffffu, old, 0);
    if (((old + 2) & (kTopK - 1)) != 0) continue;  // not the last of the 16 slots (counters start at -1)
    // ---- last arrival: fixed-order reduction of the 16 slots of token t over these 32 rows
    uint32_t wv[kTopK];
    const uint32_t* src = ec.g2 + ((static_cast<long>(t * kTopK) * kHidden + n) >> 1);
    for (;;) {
      bool bad = false;
      if ((lane & 8) == 0) {
#pragma unroll
        for (int q = 0; q < kTopK; ++q) {
          wv[q] = ld_volatile_u32(src + q * (kHidden / 2));
          bad |= (wv[q] & 0xffffu) == 0xffffu || (wv[q] >> 16) == 0xffffu;
        }
      }
      if (!__any_sync(0xffffffffu, bad)) break;
    }
    if ((lane & 8) == 0) {
      float a0 = 0.f, a1 = 0.f;
#pragma unroll
      for (int q = 0; q < kTopK; ++q) {
        float wq;
        if (ec.wts_bf16) wq = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(ec.wts)[t * kTopK + q]);
        else wq = __bfloat162float(__float2bfloat16(reinterpret_cast<const float*>(ec.wts)[t * kTopK + q]));
        a0 = fmaf(__uint_as_float(wv[q] << 16), wq, a0);
        a1 = fmaf(__uint_as_float(wv[q] & 0xffff0000u), wq, a1);
      }
      __nv_bfloat162 r2 = __floats2bfloat162_rn(a0, a1);
      uint32_t rw = *reinterpret_cast<uint32_t*>(&r2);
      if (ec.mode == 2) {
        if ((rw & 0xffffu) == 0x8000u) rw &= 0xffff0000u;
        if ((rw >> 16) == 0x8000u) rw &= 0x0000ffffu;
      }
      *reinterpret_cast<uint32_t*>(ec.fout + t * ec.fstride + n) = rw;
#pragma unroll
      for (int q = 0; q < kTopK; ++q) const_cast<uint32_t*>(src)[q * (kHidden / 2)] = 0xffffffffu;
    }
  }
}

// One epilogue item: one 128-row accumulator (this warp's 32 rows): load it from TMEM, release it, then
//   FC1 non-owner piece: send the fp32 partial to the tile owner (DSMEM);
//   FC1 owner: add the remote partial, SiTU, MXFP8, broadcast the 64-neuron h chunk to the group (DSMEM);
//   FC2: bf16 stores of the routed (token, slot) rows.
// warm_bar != 0: warm-up call (no TMEM release, all DSMEM stores go to this CTA on warm_bar, FC2 stores
// go to out = scratch row 0).
template <int MT>
__device__ EPI_FN void epi_item(uint8_t* sm, uint32_t tacc, uint32_t accempty, int mtype, int ma, int mb, int mc,
                                      int eg, int ew, int lane, int C, int rank0, uint32_t warm_bar, EpiConst ec) {
  Misc& ms = *reinterpret_cast<Misc*>(sm + kOffMisc);
  float v[MT];
  tmem_ldn<MT>(tacc, v);
  tc::tc_fence_before();
  __syncwarp();
  if (lane == 0 && accempty != 0) mbar_arrive_u32(accempty);
  const int row = 32 * ew + lane;  // physical row within the 128-row tile
  auto tev = [&](int ev) {
    if constexpr (kTraceOn) {
      if (ec.trace != nullptr && warm_bar == 0) ec.trace[blockIdx.x * 128 + ev] = tc::globaltimer();
    }
  };
  if (mtype == kFC1) {
    const int i = ma, r = mb, f = mc;
    if (ew == 0 && lane == 0) tev((f & fOwner) ? 12 : 14);
    // warm-up stores go to ring slot 1 (unused until this CTA's own post-routing TMAs)
    const int o_part = warm_bar ? kOffA + 32768 : kOffPrecv;
    const int o_h = warm_bar ? kOffA + 32768 + 8192 : kOffHq;
    const int o_hsf = warm_bar ? kOffA + 32768 + 12288 : kOffHsf;
    float* pbuf = reinterpret_cast<float*>(sm + o_part) + (r * 128 + row) * MT;
    if (!(f & fOwner)) {
      // partial of a tile owned by the previous CTA of the group: MT fp32 per row over DSMEM
      const uint32_t dst = static_cast<uint32_t>((f >> 8) & 0xff);
      const uint32_t caddr = cl::mapa(tc::su32(pbuf), dst);
      const uint32_t cbar = warm_bar ? warm_bar : cl::mapa(tc::su32(&ms.precv[r]), dst);
      if constexpr (MT == 1) {
        cl::st_async_b32(caddr, __float_as_uint(v[0]), cbar);
      } else if constexpr (MT == 2) {
        cl::st_async_v2(caddr, __float_as_uint(v[0]), __float_as_uint(v[1]), cbar);
      } else {
#pragma unroll
        for (int q = 0; q < MT / 4; ++q)
          cl::st_async_v4(caddr + 16 * q, make_uint4(__float_as_uint(v[4 * q]), __float_as_uint(v[4 * q + 1]),
                                                     __float_as_uint(v[4 * q + 2]), __float_as_uint(v[4 * q + 3])),
                          cbar);
      }
      return;
    }
    if (f & fRemote) {
      if (!warm_bar) cl::mbar_wait_cluster(tc::su32(&ms.precv[r]), 0);
      if (ew == 0 && lane == 0) tev(13);
#pragma unroll
      for (int t = 0; t < MT; ++t) v[t] += pbuf[t];
    }
    const int ebar = eg == 0 ? 1 : 4;
    const bool up_lane = (lane & 8) == 0;
    const int i_local = 2 * (lane % 8) + lane / 16;
    float h[MT], am[MT];
#pragma unroll
    for (int t = 0; t < MT; ++t) {
      const float o = __shfl_xor_sync(0xffffffffu, v[t], 8);  // the gate row sits 8 lanes up
      const float hv = ec.beta_lb * tanh_fast(o * ec.inv_beta) * sigmoid_fast(o) * tanh_fast(v[t] * ec.inv_lb);
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
#pragma unroll
    for (int t = 0; t < MT; ++t) {
      const int sb = mx_exp(fmaxf(am[t], ms.amax[eg][ew ^ 1][t]));
      if (up_lane)
        ms.hst[eg][ew][t][i_local] =
            static_cast<uint8_t>(__nv_cvt_float_to_fp8(h[t] * mx_rescale(sb), __NV_SATFINITE, __NV_E4M3));
    }
    __syncwarp();
    const int hb = i & 1;
    if (i >= 2) tc::mbar_wait(&ms.hempty[hb], ((i >> 1) - 1) & 1);  // C == 1 only: FC2 of expert i-2 done
    if (lane < MT) {
      const int t = lane, kc = 4 * r + ew;
      const uint4 val = *reinterpret_cast<const uint4*>(ms.hst[eg][ew][t]);
      const uint32_t la = tc::su32(sm + o_h + hb * 2048 + (kc / 8) * 1024 + t * 128 + (((kc % 8) ^ t) * 16));
      const uint32_t lb = tc::su32(&ms.hfull[hb]);
      const uint32_t ls = tc::su32(sm + o_hsf + hb * 768 + r * 128 + t * 16);
      uint32_t sw = 0;
      if (ew == 0) {
        const int sb0 = mx_exp(fmaxf(ms.amax[eg][0][t], ms.amax[eg][1][t]));
        const int sb1 = mx_exp(fmaxf(ms.amax[eg][2][t], ms.amax[eg][3][t]));
        sw = static_cast<uint32_t>(sb0) | (static_cast<uint32_t>(sb1) << 8);
      }
#pragma unroll 1
      for (int q = 0; q < C; ++q) {
        const uint32_t rk = static_cast<uint32_t>(rank0 + q);
        const uint32_t cb = warm_bar ? warm_bar : cl::mapa(lb, rk);
        cl::st_async_v4(cl::mapa(la, rk), val, cb);
        if (ew == 0) cl::st_async_b32(cl::mapa(ls, rk), sw, cb);
      }
    }
    asm volatile("bar.sync %0, 128;" ::"r"(ebar) : "memory");  // ms.amax / hst reuse
    if (ew == 0 && lane == 0) tev(6);
  } else {
    const int u = mc, mt = mb;
    // physical row p -> logical output row 32*(p/32) + 4*(p%8) + (p%32)/8
    const uint2 ts = warm_bar ? make_uint2(0xffffff00u, ~0u) : *reinterpret_cast<const uint2*>(ms.tokslot[u]);
    fc2_emit<MT>(v, ts, mt, ew, lane, ec);
  }
}

// MMA stage issue (whole warp; one elected lane issues; live == 0: dry run).
__device__ MMA_FN void mma_fc1(uint32_t d, uint64_t adesc, uint64_t bdesc, uint32_t idesc, uint32_t sfa,
                                     uint64_t sfa_src, uint32_t sfb, uint64_t sfb_src, uint32_t copy_sfb, uint32_t accum,
                                     uint32_t bar_empty, uint32_t bar_acc, uint32_t live) {
  tc::fc1_stage(d, adesc, bdesc, idesc, sfa, sfa_src, sfb, sfb_src, copy_sfb, accum, bar_empty, bar_acc, live);
}
__device__ MMA_FN void mma_fc2(uint32_t d, uint64_t adesc, uint64_t bdesc, uint32_t idesc, uint32_t sfa,
                                     uint64_t sfa_src, uint32_t sfb, uint64_t sfb_src, uint32_t copy_sfb, uint32_t bar_hrel,
                                     uint32_t bar_empty, uint32_t bar_acc, uint32_t live) {
  tc::fc2_stage(d, adesc, bdesc, idesc, sfa, sfa_src, sfb, sfb_src, copy_sfb, bar_hrel, bar_empty, bar_acc, live);
}

// One weight stage: 16 KB (unpacked to 32 KB) of fp4 rows + 1 KB of scales into ring slot `slot`.
__device__ TMA_FN void tma_stage(uint8_t* sm, int slot, const CUtensorMap* map, int row0, int kc,
                                       const uint8_t* sf, uint32_t live) {
  Misc& ms = *reinterpret_cast<Misc*>(sm + kOffMisc);
  if (live) {
    tc::mbar_expect_tx(&ms.full[slot], 16384 + 1024);
#ifndef BODY_EVICT_NORMAL  // L2 evict-first on the single-use weight stream (as agents/moe8/moe8.cu v10)
    const uint64_t pol = tc::policy_evict_first();
    tc::tma_load_3d_hint(sm + kOffA + slot * 32768, map, &ms.full[slot], 0, row0, kc, pol);
    tc::bulk_load_hint(sm + kOffSFA + slot * 1024, sf, 1024, &ms.full[slot], pol);
#else
    tc::tma_load_3d(sm + kOffA + slot * 32768, map, &ms.full[slot], 0, row0, kc);
    tc::bulk_load(sm + kOffSFA + slot * 1024, sf, 1024, &ms.full[slot]);
#endif
  }
}

// Dedupe the routed experts in first-occurrence order (threads 0..127; M > 2). npairs == 0: dry run.
__device__ __noinline__ void dedupe(uint8_t* sm, int myid, int npairs) {
  Misc& ms = *reinterpret_cast<Misc*>(sm + kOffMisc);
  int* table = reinterpret_cast<int*>(sm + kOffA);
  const int i = threadIdx.x, warp = i / 32, lane = i % 32;
  *reinterpret_cast<uint2*>(ms.tokslot[i]) = make_uint2(~0u, ~0u);
  if (i < npairs) atomicMin(&table[myid], i);
  asm volatile("bar.sync 3, 128;" ::: "memory");  // (1, 4: epilogue groups)
  const int firstk = i < npairs ? table[myid] : -1;
  const bool first = (i < npairs) && firstk == i;
  const unsigned bal = __ballot_sync(0xffffffffu, first);
  if (lane == 0) ms.wcnt[warp] = __popc(bal);
  asm volatile("bar.sync 3, 128;" ::: "memory");  // (1, 4: epilogue groups)
  const int base = (warp > 0 ? ms.wcnt[0] : 0) + (warp > 1 ? ms.wcnt[1] : 0) + (warp > 2 ? ms.wcnt[2] : 0);
  const int rank = base + __popc(bal & ((1u << lane) - 1));
  if (first) {
    ms.uof[i] = rank;
    ms.expert[rank] = myid;
  }
  if (i == 0) ms.D = ms.wcnt[0] + ms.wcnt[1] + ms.wcnt[2] + ms.wcnt[3];
  asm volatile("bar.sync 3, 128;" ::: "memory");  // (1, 4: epilogue groups)
  if (i < npairs) ms.tokslot[ms.uof[firstk]][i / kTopK] = static_cast<signed char>(i % kTopK);
}

// Group size C (largest divisor of L, <= 6, giving every expert its own group) and this CTA's place; arms
// this call's receive barriers (remote chunks may already be in flight: the tx-count may go negative).
__device__ __noinline__ void setup_groups(uint8_t* sm, int D, int L, int crank, int ncl, int cid, int force_c,
                                          uint32_t hbytes, uint32_t pbytes, int arm) {
  Misc& ms = *reinterpret_cast<Misc*>(sm + kOffMisc);
  int C = 1;
  if (force_c > 0) {
    C = force_c;
  } else {
    for (int c = kMaxC; c >= 1; --c)
      if (L % c == 0 && ncl * (L / c) >= D) {
        C = c;
        break;
      }
  }
  const int per = L / C, ngroups = ncl * per;
  const int grp = cid * per + crank / C, grank = crank % C;
  const int nloc = grp < D ? (D - grp + ngroups - 1) / ngroups : 0;
  ms.D = D;
  ms.C = C;
  ms.ngroups = ngroups;
  ms.grp = grp;
  ms.grank = grank;
  ms.rank0 = crank - grank;
  ms.nloc = nloc;
  if (arm) {
    if (nloc > 0) tc::mbar_expect_tx(&ms.hfull[0], hbytes);
    if (nloc > 1) tc::mbar_expect_tx(&ms.hfull[1], hbytes);
    if (nloc > 0) {
      const int lo1 = kFc1Seq * grank / C, hi1 = kFc1Seq * (grank + 1) / C;
      for (int r = 0; r < kFc1Tiles; ++r)
        if (lo1 <= kFc1Stages * r && kFc1Stages * r < hi1 && hi1 < kFc1Stages * (r + 1))
          tc::mbar_expect_tx(&ms.precv[r], pbytes);
    }
  }
}


// ================================================================ TMA producer of the group kernel (one thread).
// dry != 0: a pass over one fake FC1 and one FC2 stage that executes the same instructions without waiting,
// loading or publishing (warms the instruction cache before the routing is known).
__device__ __noinline__ void group_producer(uint8_t* sm, const CUtensorMap* tmA1, const CUtensorMap* tmA2,
                                            const Params& p, int dry) {
  Misc& ms = *reinterpret_cast<Misc*>(sm + kOffMisc);
  const int C = dry ? 2 : ms.C, grank = dry ? 1 : ms.grank, D = dry ? 1 : ms.D, ngroups = dry ? 1 : ms.ngroups;
  const int grp = dry ? 0 : ms.grp, rank0 = dry ? 0 : ms.rank0;
  const int lo1 = kFc1Seq * grank / C, hi1 = dry ? lo1 + 1 : kFc1Seq * (grank + 1) / C;
  const int lo2 = kFc2Tiles * grank / C, hi2 = dry ? lo2 + 1 : kFc2Tiles * (grank + 1) / C;
  const int n1 = hi1 - lo1, n2 = hi2 - lo2;
  const int owner_prev = (rank0 + grank - 1) << 8;
  const uint32_t live = dry ? 0u : 1u;
  int slot = 0, phase = 0, n = 0;
  long long w_empty = 0;
  int i = 0;

#pragma unroll 1
  for (int u = grp; u < D; u += ngroups, ++i) {
    const int e = dry ? 0 : ms.expert[u];
#pragma unroll 1
    for (int q = 0; q < n1 + n2; ++q) {
      Meta m;
      const CUtensorMap* map;
      int row0, kc;
      const uint8_t* sf;
      if (q < n1) {
        const int s = lo1 + q, r = s / kFc1Stages, ks = s % kFc1Stages;
        const int ps = max(lo1, kFc1Stages * r), pe = min(hi1, kFc1Stages * (r + 1));  // this piece
        const bool owner = ps == kFc1Stages * r;
        int f = (s == ps ? fFirst : 0) | (s == pe - 1 ? fLast : 0) | (ks << 16);
        f |= owner ? (fOwner | (pe < kFc1Stages * (r + 1) ? fRemote : 0)) : owner_prev;
        m = Meta{kFC1, i, r, f};
        map = tmA1;
        row0 = e * kW13Rows + r * 128;
        kc = 2 * ks;
        sf = p.w13s + static_cast<long>(e) * kW13ScaleBytes + r * 14336 + ks * 1024;
      } else {
        const int mt = lo2 + q - n1;
        if (q == n1 && i == 0 && live) trace_ev(p, 15);
        m = Meta{kFC2, i, mt, u};
        map = tmA2;
        row0 = e * kHidden + mt * 128;
        kc = 0;
        sf = p.w2s + static_cast<long>(e) * kW2ScaleBytes + mt * 1024;
      }
      if (n >= kSlots && live) {
        const long long c0 = PCLK();
        tc::mbar_wait(&ms.empty[slot], phase ^ 1);
        w_empty += PCLK() - c0;
      }
      if (n == 0 && live) trace_ev(p, 4);
      if (live) ms.meta[slot] = m;
      tma_stage(sm, slot, map, row0, kc, sf, live);
      ++n;
      if (++slot == kSlots) {
        slot = 0;
        phase ^= 1;
      }
    }
  }
  if (!live) return;
  trace_ev(p, 5);
  trace_ctr(p, 0, w_empty);
  if (n >= kSlots) tc::mbar_wait(&ms.empty[slot], phase ^ 1);
  ms.meta[slot] = Meta{kEnd, 0, 0, 0};
  tc::mbar_arrive(&ms.full[slot]);
}

// ================================================================ MMA issuer of the group kernel (whole warp; one
// lane issues). dry != 0: one fake FC1 and one fake FC2 stage, nothing issued or waited on.
template <int MT>
__device__ __noinline__ void group_mma(uint8_t* sm, const Params& p, int dry) {
  Misc& ms = *reinterpret_cast<Misc*>(sm + kOffMisc);
  const int lane = threadIdx.x % 32;
  const uint32_t live = dry ? 0u : 1u;
  tc::tc_fence_after();  // tmem_base was written by tcgen05.alloc (warp 3) before a CTA barrier
  const uint32_t tmem = ms.tmem_base;
  const uint32_t idesc = tc::idesc_mxf8f6f4(128, 8, 5, 0);
  const int C = dry ? 2 : ms.C, nloc = dry ? 1 : ms.nloc;
  int phase = 0, slot = 0;
  int acc = 0, accph = 0, accuses = 0;
  int cur_i = -1;
  uint32_t xmask = 0;
  long long w_full = 0, w_acc = 0, w_h = 0, n_st = 0;
  const uint64_t adesc0 = tc::desc_sw128(tc::su32(sm + kOffA));    // + slot * 32 KB / 16
  const uint64_t sfa_src0 = desc_sf(tc::su32(sm + kOffSFA));       // + slot * 1 KB / 16
  const uint64_t xdesc0 = tc::desc_sw128(tc::su32(sm + kOffXq));   // + ks * 2 KB / 16
  const uint64_t xsf_src0 = desc_sf(tc::su32(sm + kOffXsf));       // + ks * 256 B / 16
  const uint64_t hdesc0 = tc::desc_sw128(tc::su32(sm + kOffHq));   // + b * 2 KB / 16
  const uint64_t hsf_src0 = desc_sf(tc::su32(sm + kOffHsf));       // + b * 768 B / 16
  const uint32_t bar_empty0 = tc::su32(&ms.empty[0]), bar_acc0 = tc::su32(&ms.accfull[0]);
  const uint32_t bar_hempty0 = tc::su32(&ms.hempty[0]);
#pragma unroll 1
  for (;;) {
    long long c0 = PCLK();
    if (live) {
      tc::mbar_wait(&ms.full[slot], phase);
      tc::tc_fence_after();
    }
    w_full += PCLK() - c0;
    Meta m;
    if (live) m = ms.meta[slot];
    else m = n_st == 0 ? Meta{kFC1, 0, 0, fFirst | fLast} : n_st == 1 ? Meta{kFC2, 0, 0, 0} : Meta{kEnd, 0, 0, 0};
    ++n_st;
    if (m.type == kEnd) break;
    const bool fc1 = m.type == kFC1;
    const bool first = fc1 ? (m.c & fFirst) != 0 : true;
    const bool last = fc1 ? (m.c & fLast) != 0 : true;
    if (first && accuses >= kAcc && live) {
      c0 = PCLK();
      tc::mbar_wait(&ms.accempty[acc], accph ^ 1);
      tc::tc_fence_after();
      w_acc += PCLK() - c0;
    }
    if (fc1) {
      const int kstage = (m.c >> 16) & 0xff;  // K-stage index (flags bits 16..23)
      const uint32_t copy_sfb = ((xmask >> kstage) & 1) ? 0u : 1u;
      xmask |= 1u << kstage;
      mma_fc1(tmem + kColAcc + acc * 8, adesc0 + slot * 2048, xdesc0 + kstage * 128, idesc, tmem + kColSFA + slot * 8,
              sfa_src0 + slot * 64, tmem + kColSFBx + 8 * kstage, xsf_src0 + kstage * 16, copy_sfb,
              first ? 0u : 1u, bar_empty0 + slot * 8, last ? bar_acc0 + acc * 8 : 0u, live);
    } else {
      uint32_t copy_sfb = 0, bar_hrel = 0;
      if (m.a != cur_i) {  // new expert: release the previous h buffer, wait for this one
        if (cur_i >= 0) bar_hrel = bar_hempty0 + (cur_i & 1) * 8;
        cur_i = m.a;
        c0 = PCLK();
        if (live) cl::mbar_wait_cluster(tc::su32(&ms.hfull[cur_i & 1]), (cur_i >> 1) & 1);
        w_h += PCLK() - c0;
        tc::fence_proxy_async_smem();
        tc::tc_fence_after();
        if (lane == 0 && live) {
          if (C == 1 && cur_i + 2 < nloc) tc::mbar_expect_tx(&ms.hfull[cur_i & 1], Sz<MT>::kH);
          if (cur_i == 0) trace_ev(p, 7);
        }
        copy_sfb = 1;
      }
      const int hb = cur_i & 1;
      mma_fc2(tmem + kColAcc + acc * 8, adesc0 + slot * 2048, hdesc0 + hb * 128, idesc, tmem + kColSFA + slot * 8,
              sfa_src0 + slot * 64, tmem + kColSFBh + hb * 12, hsf_src0 + hb * 48, copy_sfb, bar_hrel,
              bar_empty0 + slot * 8, bar_acc0 + acc * 8, live);
    }
    __syncwarp();
    if (last) {
      if (lane == 0 && live) {
        ms.accmeta[acc] = m;
        tc::mbar_arrive(&ms.accfull[acc]);
        if (fc1) trace_ev(p, 11);
      }
      ++accuses;
      if (++acc == kAcc) {
        acc = 0;
        accph ^= 1;
      }
    }
    if (++slot == kSlots) {
      slot = 0;
      phase ^= 1;
    }
  }
  if (!live) return;
  if (lane == 0) {
    trace_ev(p, 8);
    trace_ctr(p, 1, w_full);
    trace_ctr(p, 2, w_acc);
    trace_ctr(p, 3, w_h);
    trace_ctr(p, 6, n_st);
  }
  const Meta mend{kEnd, 0, 0, 0};
  for (int q = 0; q < 2; ++q) {  // one END per epilogue group
    if (accuses >= kAcc) tc::mbar_wait(&ms.accempty[acc], accph ^ 1);
    if (lane == 0) {
      ms.accmeta[acc] = mend;
      tc::mbar_arrive(&ms.accfull[acc]);
      tc::mbar_arrive(&ms.accfull[acc]);
    }
    ++accuses;
    if (++acc == kAcc) {
      acc = 0;
      accph ^= 1;
    }
  }
}

// ================================================================ group body (M <= 4 by default), device entry point.
// Requirements: threads 0..383 of the CTA execute it (extra warps may exist and run other code); `sm` is
// the 1024-aligned base of >= kSmemBytes of dynamic shared memory owned by the body; the grid is launched
// as clusters of L CTAs (L = 6 / 4 / 2 for M = 1 / 2 / >= 3), all CTAs co-resident; TMEM: 512 columns are
// allocated (the caller must not hold TMEM); named barriers 1, 3, 4, 5, 6 are used.
template <int MT>
__device__ __forceinline__ void group_body(uint8_t* sm, const CUtensorMap& tmA1, const CUtensorMap& tmA2,
                                           const Params& p) {
  Misc& ms = *reinterpret_cast<Misc*>(sm + kOffMisc);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int M = p.M;
  const int cta = blockIdx.x;
  const int L = static_cast<int>(cl::nctarank()), crank = static_cast<int>(cl::ctarank());
  const int ncl = static_cast<int>(cl::nclusterid()), cid = static_cast<int>(cl::clusterid());
  constexpr uint32_t kHB = Sz<MT>::kH, kPB = Sz<MT>::kPart;

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
    for (int r = 0; r < kFc1Tiles; ++r) tc::mbar_init(&ms.precv[r], 1);
    tc::mbar_init(&ms.warmbar, 1);
    tc::fence_mbar_init();
    tc::prefetch_tmap(&tmA1);
    tc::prefetch_tmap(&tmA2);
  }
  if (warp == 3) {
    tc::tmem_alloc(&ms.tmem_base, kTmemCols);
    tc::tc_fence_before();
  }
  if (M > 2) {
    int* table = reinterpret_cast<int*>(sm + kOffA);  // dedupe scratch (the ring is still unused)
    for (int e = threadIdx.x; e < p.E; e += kThreads) table[e] = 0x7fffffff;
  }
  // rows t >= M of the x B image and of its scale images stay zero
  for (int i = threadIdx.x; i < (kMaxM - M) * 224; i += kThreads) {
    const int t = M + i / 224, kc = i % 224;
    *reinterpret_cast<uint4*>(sm + kOffXq + (kc / 8) * 1024 + t * 128 + (((kc % 8) ^ t) * 16)) = make_uint4(0, 0, 0, 0);
  }
  for (int i = threadIdx.x; i < 28 * 8; i += kThreads)
    *reinterpret_cast<uint4*>(sm + kOffXsf + (i >> 3) * 128 + (i & 7) * 16) = make_uint4(0, 0, 0, 0);
  body_sync();
  if (L > 1) cl::arrive_release();  // mbarrier inits visible cluster-wide before any remote st.async
  // ---- warm every role's post-routing code: needs no input, so it runs before griddepcontrol.wait and can
  // overlap the predecessor's tail
  const EpiConst ec_warm = make_epi(p, 0);
#ifdef BODY_WARM_ALWAYS
  const bool late_routing = true;
#else
  const bool late_routing = p.flag != nullptr || p.ids_smem != nullptr;
#endif
#ifndef BODY_NO_WARM
  if (!late_routing) {
  } else if (warp == 0) {
#ifdef BODY_GROUP_DRY  // measured: the dry pass does not pay off in the group kernel (it does in stream-K)
    if (lane == 0) group_producer(sm, &tmA1, &tmA2, p, 1);
#endif
  } else if (warp == 1) {
#ifdef BODY_GROUP_DRY
    tc::tc_fence_after();
    group_mma<MT>(sm, p, 1);
#endif
  } else if (warp >= 4) {
    tc::tc_fence_after();
    const int ew = warp & 3, eg = (warp - 4) >> 2;
    const uint32_t wb = tc::su32(&ms.warmbar);
    const uint32_t tl = ms.tmem_base + (static_cast<uint32_t>(32 * ew) << 16);
    EpiConst ew_c = ec_warm;
    ew_c.out = reinterpret_cast<__nv_bfloat16*>(p.scratch);
    ew_c.mode = 0;
    epi_item<MT>(sm, tl, 0u, kFC1, 0, 0, crank << 8, eg, ew, lane, 1, crank, wb, ew_c);
    epi_item<MT>(sm, tl, 0u, kFC1, 0, 0, fOwner | fRemote, eg, ew, lane, 1, crank, wb, ew_c);
    epi_item<MT>(sm, tl, 0u, kFC2, 0, 0, 0, eg, ew, lane, 1, crank, wb, ew_c);
  }
  if (late_routing && M > 2 && warp < 4) dedupe(sm, -1, 0);
#endif

  const int npairs = M * kTopK;
  // M <= 2: the (token, slot) pairs are the experts, so the group geometry and the slot table do not depend on
  // the routing: set them up (and arm this call's receive barriers) before waiting for the predecessor.
  if (threadIdx.x == 0) {
    if (M <= 2) {
      setup_groups(sm, npairs, L, crank, ncl, cid, p.force_c, kHB, kPB, 1);
    } else {
#ifndef BODY_NO_WARM
      setup_groups(sm, npairs, L, crank, ncl, cid, p.force_c, kHB, kPB, 0);
#endif
    }
  }
  if (M <= 2 && threadIdx.x < npairs) {
    uint2 ts = make_uint2(~0u, ~0u);
    const uint32_t byte = static_cast<uint32_t>(threadIdx.x % kTopK);
    if (threadIdx.x < kTopK) ts.x = (ts.x & ~0xffu) | byte;
    else ts.x = (ts.x & ~0xff00u) | (byte << 8);
    *reinterpret_cast<uint2*>(ms.tokslot[threadIdx.x]) = ts;
  }
  // Routing source: (a) p.ids in global memory, after griddepcontrol.wait; (b) flag mode: x is ready, the
  // routing arrives through the epoch counter (waiting for the predecessor grid would serialize with it);
  // (c) p.ids_smem: ids written into shared memory by the caller's own warps, signalled on p.ids_bar.
  const bool flagmode = p.flag != nullptr;
  const bool smemids = p.ids_smem != nullptr;
  if (!flagmode && !smemids && !p.no_griddep) asm volatile("griddepcontrol.wait;" ::: "memory");
  if (threadIdx.x == 0) trace_ev(p, 1);
  // M <= 2 with the routing in global memory: the producer thread reads its own group's expert id and starts
  // streaming at once; warps 1..11 quantize x and meet on a 352-thread barrier (the MMA warp is among them, so
  // x is complete before its first FC1 stage).
#ifdef BODY_EARLY  // opt-in: ~1.5 us faster in the block chain, but a timing-dependent fault is not yet understood
  const bool early = M <= 2 && !flagmode && !smemids;
#else
  const bool early = false;
#endif
  if (early && warp == 0) {
    if (L > 1) cl::wait_acquire();  // every thread that arrived on the cluster barrier also waits on it
    if (lane == 0) {
      for (int u = ms.grp; u < ms.D; u += ms.ngroups) ms.expert[u] = p.ids[u];
      trace_ev(p, 2);
      group_producer(sm, &tmA1, &tmA2, p, 0);
    }
  } else {
    // the call index is read after griddepcontrol.wait: a previous call on this workspace has finished
    if (threadIdx.x == 32) ms.call = ~p.ctr[cta];  // counters are stored inverted (the workspace starts all-0xFF)
    int myid = -1;
    if (!flagmode && !smemids && !early && threadIdx.x < npairs) myid = p.ids[threadIdx.x];
    // x -> MXFP8 (does not depend on the routing); flag mode with x_late: after the flag wait (below)
    const bool xlate = flagmode && p.x_late;
    const int q0 = early ? 32 : 0, nq = early ? kThreads - 32 : kThreads;
    if (!xlate)
      for (int it = static_cast<int>(threadIdx.x) - q0; it < M * 112; it += nq) quant_x_block(sm, p.x, it / 112, it % 112);
    tc::fence_proxy_async_smem();
    rest_sync(early);
    if (L > 1) cl::wait_acquire();

    const int call = ms.call;
    const EpiConst ec = make_epi(p, call);
  if (flagmode) {
    body_sync();
    if (threadIdx.x == 0) {
      trace_ev(p, 16);
      const unsigned long long target = static_cast<unsigned long long>(call) + 1ull;
      while (ld_acquire_u64(p.flag) < target) {
      }
    }
    body_sync();
    if (threadIdx.x < npairs) myid = ld_relaxed_i32(p.ids + threadIdx.x);
    if (xlate) {  // x published with the ids (acquire by thread 0's ld.acquire + the CTA barrier above)
      for (int it = threadIdx.x; it < M * 112; it += kThreads) quant_x_block(sm, p.x, it / 112, it % 112);
      tc::fence_proxy_async_smem();
    }
  } else if (smemids) {
    if (threadIdx.x == 0) trace_ev(p, 16);
    if (threadIdx.x < npairs) {
      tc::mbar_wait(reinterpret_cast<uint64_t*>(__cvta_shared_to_generic(p.ids_bar)), p.ids_phase);
      myid = p.ids_smem[threadIdx.x];
    }
  }
  if (threadIdx.x == 0) trace_ev(p, 2);

  // ---- routing known: expert list (M <= 2: the pairs themselves; else dedupe), then the groups
  if (early) {
  } else if (M <= 2) {
    if (threadIdx.x < npairs) ms.expert[threadIdx.x] = myid;
  } else if (warp < 4) {
    dedupe(sm, myid, npairs);
    if (threadIdx.x == 0) setup_groups(sm, ms.D, L, crank, ncl, cid, p.force_c, kHB, kPB, 1);
  }
  rest_sync(early);
  asm volatile("griddepcontrol.launch_dependents;");
  if (threadIdx.x == 0) trace_ev(p, 3);

  if (warp == 0) {
    if (lane == 0) group_producer(sm, &tmA1, &tmA2, p, 0);
  } else if (warp == 1) {
    group_mma<MT>(sm, p, 0);
#ifdef BODY_PF_FC2
  } else if (warp == 2) {
    // L2 prefetch of the group's FC2 weights (the smem ring cannot hold them while FC1 streams): CTAs with
    // grank < BODY_PF_FC2 prefetch the row tiles of granks grank, grank + BODY_PF_FC2, ..., in stage order.
    const int C = ms.C, grank = ms.grank, D = ms.D, ngroups = ms.ngroups;
    if (lane == 0 && grank < BODY_PF_FC2 && ms.grp < D) {
      const int e = ms.expert[ms.grp];
      const uint8_t* w2 = p.w2 + static_cast<long>(e) * kHidden * 128;
      const uint8_t* w2s = p.w2s + static_cast<long>(e) * kW2ScaleBytes;
      uint64_t pol;
      asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
      int lo[kMaxC], hi[kMaxC], nq = 0;
      for (int g2 = grank; g2 < C; g2 += BODY_PF_FC2, ++nq) {
        lo[nq] = kFc2Tiles * g2 / C;
        hi[nq] = kFc2Tiles * (g2 + 1) / C;
      }
      for (int step = 0; step < kFc2Tiles; ++step)
        for (int q = 0; q < nq; ++q) {
          const int mt = lo[q] + step;
          if (mt < hi[q]) {
            // evict_last prefetches are never dropped by L2 (Blackwell TMA programming guide, opUBLKPF)
            asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" ::"l"(w2 + static_cast<long>(mt) * 128 * 128),
                         "r"(128 * 128), "l"(pol) : "memory");
            asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" ::"l"(w2s + mt * 1024), "r"(1024),
                         "l"(pol) : "memory");
          }
        }
    }
    (void)ngroups;
#endif
  } else if (warp >= 4) {
    // ================================================================ epilogue (two groups of 4 warps;
    // group eg handles accumulators eg, eg + 2, ...; ew = TMEM lane quadrant)
    tc::tc_fence_after();
    const uint32_t tmem = ms.tmem_base;
    const int ew = warp & 3, eg = (warp - 4) >> 2;
    const uint32_t tl = tmem + (static_cast<uint32_t>(32 * ew) << 16);
    const int C = ms.C, rank0 = ms.rank0;
    int acc = eg, accph = 0;
    long long e_wait = 0, e_fc1 = 0, e_fc2 = 0;
    for (;;) {
      long long ec0 = PCLK();
      tc::mbar_wait(&ms.accfull[acc], accph);
      tc::tc_fence_after();
      e_wait += PCLK() - ec0;
      ec0 = PCLK();
      const Meta m = ms.accmeta[acc];
      if (m.type == kEnd) break;
      epi_item<MT>(sm, tl + kColAcc + acc * 8, tc::su32(&ms.accempty[acc]), m.type, m.a, m.b, m.c, eg, ew, lane, C,
                   rank0, 0u, ec);
      if (m.type == kFC1) e_fc1 += PCLK() - ec0;
      else e_fc2 += PCLK() - ec0;
      if ((acc += 2) >= kAcc) {
        acc -= kAcc;
        accph ^= 1;
      }
    }
    if (eg == 0 && ew == 0 && lane == 0) {
      trace_ev(p, 9);
      trace_ctr(p, 7, e_wait);
      trace_ctr(p, 8, e_fc1);
      trace_ctr(p, 9, e_fc2);
    }
  }
  }  // (early producer thread / everyone else)
#ifdef BODY_EARLY_SYNCWARP
  if (early && warp == 0) __syncwarp();
#endif
  tc::tc_fence_before();
  body_sync();
  if (warp == 3) {
    tc::tc_fence_after();
    tc::tmem_dealloc(ms.tmem_base, kTmemCols);
  }
  // Flag mode never waited for the predecessor grid: do it before exiting, so that this grid's completion
  // still implies the predecessor's (successors' griddepcontrol.wait only covers the immediate predecessor).
  if (flagmode) asm volatile("griddepcontrol.wait;" ::: "memory");
  if (threadIdx.x == 0) {
    const int call_end = ms.call;
    if (p.endlog != nullptr)
      atomicMax(p.endlog + (call_end % p.endlog_n), static_cast<unsigned long long>(tc::globaltimer()));
    bump_counters(p.ctr, cta, static_cast<int>(gridDim.x), call_end);
    trace_ev(p, 10);
  }
}

#include "sk_kernel.cuh"  // stream-K body (inside namespace moebody)

}  // namespace moebody
