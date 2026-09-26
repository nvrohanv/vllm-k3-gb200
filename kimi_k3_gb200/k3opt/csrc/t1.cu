// T1: the LIGHT MoE tail (design_persistent.md s2.2 "T", Stage 1). Agent moefused. Namespace k3t1. v2.
// Protocol / mailbox contract: agents/moefused/T1_INTERFACE.md (identical to k3mk.tail's, tail2_det.cu).
//
// Semantics = vLLM KimiK3LatentMoETailOp at the same rounding points (and bit-identical to k3mk.tail v14):
//   x   = bf16(sum_{src=0..15} fp32 lat_mb[par][src][t])                   (routed all-reduce, H3)
//   inv = rsqrt.approx(div.rn(S, 3584) + eps), S = vLLM's TP16 cluster tree over bf16(x*x)
//   xn  = bf16(x * inv * gamma)
//   sh  = bf16(sum_{src=0..15} fp32 rs_mb[par][src][t][own rows])          (shared reduce-scatter)
//   out = bf16(fp32(bf16(xn . W_up_row)) + fp32(sh))  -> up-proj mailbox columns 448 rank + row (multimem, H4)
//
// LIGHT resource contract (design_persistent s2.1): 128 threads, __maxnreg__(120) (15,360 regs/CTA), <= 40 KB smem,
// so a BIG CTA (<= 189 KB, <= 49,152 regs) plus a 1-warp side CTA (k3pf) can share the SM. The kernel requests the
// maximum shared-memory carveout (a LIGHT kernel with the default preference gets a small carveout on its SMs and a
// BIG successor cannot land there until they drain). PDL trigger at entry by default (K3T1_TRIG=1: after the poll);
// no griddepcontrol.wait in publish mode (the mailbox data is the readiness); no exit cluster barrier. PDL trigger after
// the H4 publish by default (K3T1_TRIG; in-server, triggering at entry launched the next layer's cascade early).
//
// Geometry: NC clusters x 8 CTAs (NC = 14 (default): 112 CTAs, or 15: 120). The 448 up-proj rows of this rank are split
// in 8-row groups: cluster c owns rows [row0(c), +nrows(c)) (NC=14: 32 each; NC=15: 11 x 32 + 4 x 24). CTA q of a
// cluster owns latent columns [448 q, +448) (K-split); the cluster LEADER (q = 0) owns all nrows output rows.
// SINGLE WRITER PER 128-B LINE (v4; agents/probe RESULTS s6.6): every line T1 stores is written by one warp.
//   entry   thread 127 (never a poller) inits mbarriers, bulk-copies gamma[448 q..] and TMAs W_up[row0 .. +32]
//           [448 q .. +448] (28 KB) into smem; optional next-layer L2 prefetch issue (pf_mode 1).
//   front   (front mode only) griddepcontrol.wait; tasks = whole 128-B lines: finalize a 64-column latent line (vLLM
//           finalize_top16_bf16 order, lane = 2 columns) or copy a 64-column shared-out line to its owner rank's RS
//           mailbox; one warp per line, stored as 8 lanes x 16 B.
//   poll    rounds of 2 tokens: thread (t = t0 + tid/64, k = tid%64): k < 56 polls latent fragment 56 q + k of all
//           16 sources (16-byte loads, every 32-bit word checked, branch-free); the leader's RS readers (4 x 16 B =
//           32 rows per token) run the same code; fixed source order.
//   rms     per-thread sequential sum of 8 bf16 squares, per-warp xor butterfly, lanes 0-7 push the warp sum to
//           the 8 CTAs of the cluster (st.async + mbarrier); total per peer d in order (tot + w0_d) + w1_d.
//   gemv    warp w: rows [8 w, +8) x the CTA's 448 columns (value-halving reduction, bit-identical to the per-row
//           butterfly); fp32 partials of all rows -> the leader over DSMEM.
//   owner   leader, warp per token, lane per row: fixed-order sum of the 8 partials, bf16, + sh, bf16; the cluster's
//           rows are ONE contiguous 64-B run (NC=14: 2 writers per H4 line), stored as 4 lanes x 16 B multimem.st.
//   re-arm  RS chunks by their reader after the publish; latent columns by the last of the NC readers
//           (readers-done counter cnt[par*8 + q], relaxed atom); optional next-layer prefetch issue (pf_mode 2).
// Kernel instantiations t1_kernel<NC, KM>: KM = 1 / 2 compile-time token count (the hot paths: no loops, no runtime
// division; cold instruction fetch of branchy code dominated v1's time), 0 = runtime M 3..8.
// Build knobs (experiments only): T1_DIAG (extra trace marks), T1_NOTMA, T1_POLL_NS, T1_POLL_LD, T1_FRONT_FENCE,
// T1_EXIT_BARRIER, T1_NT (the kernel assumes 128). Runtime env: K3T1_NC, K3T1_TRIG, K3T1_POLL_DELAY_NS, K3T1_CARVEOUT.
#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cudaTypedefs.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {
constexpr int kLat = 3584;             // latent dim
constexpr int kHid = 7168;             // model hidden
constexpr int kTp = 16;
constexpr int kShard = kHid / kTp;     // 448 up-proj rows per rank
constexpr int kTopK = 16;
constexpr int kMmax = 8;
constexpr int kCl = 8;                 // CTAs per cluster
#ifndef T1_NT
#define T1_NT 128
#endif
constexpr int kNT = T1_NT;             // threads per CTA (128; other values for experiments only)
constexpr int kCols = kLat / kCl;      // 448 latent columns per CTA
constexpr int kFrag = kCols / 8;       // 56 16-byte fragments per token per CTA
constexpr int kBoxR = 32;              // W_up rows staged per CTA (max rows per cluster)
constexpr int kStage = kNT - 1;        // staging thread (never a poller: pollers are k <= 56 of each half)
#ifndef T1_POLL_NS
#define T1_POLL_NS 64
#endif
constexpr int kPollNs = T1_POLL_NS;  // __nanosleep between poll rounds
constexpr uint32_t kSent = 0x80000000u;

// Kernel parameters are read through the constant cache; every 64-byte line touched before the poll is a cold miss
// on the critical path. Everything the prologue, the poll and xn need is in the FIRST 64 bytes.
struct T1Args {
  int M, par, mmax, rank;        //  0
  int nc;                        // 16: clusters (14 or 15)
  int mc;                        // 20: 1 = multimem.st, 0 = plain stores (1-GPU tests)
  unsigned long long* trace;     // 24: [G][16] globaltimer marks or nullptr
  __nv_bfloat16* lat_mb;         // 32: local [3][16][mmax][3584]
  __nv_bfloat16* rs_mb;          // 40: local [3][16][mmax][448]
  const __nv_bfloat16* gemm2;    // 48: front mode: [M*16][3584] (nullptr: publish mode)
  float eps;                     // 56
  int flags;                     // 60: bits 0-1 prefetch mode (0 off, 1 at entry before polling, 2 after the H4 publish);
                                 //     bits 4-5 PDL trigger point (0 entry, 1 after the poll, 2 after the H4 publish)
  // ---- second line: staging thread (gamma, prefetch), after the GEMV (up_st, cnt), front mode
  const __nv_bfloat16* gamma;    // [3584]
  unsigned long long up_st;      // multicast (or plain) address of the up-proj mailbox [1][>=M][7168]
  unsigned long long* cnt;       // [3][8] readers-done counters (monotonic, zero once; one tensor per nc)
  const long long* pf_rng;       // [pf_n][2] (ptr, bytes) device table of the next layer's weights
  int pf_n;                      // ranges
  int pf_stride;                 // CTA gb issues iff gb % pf_stride == 0 (issuer index gb / pf_stride)
  int pf_nis;                    // number of issuing CTAs
  int pf_chunk;                  // bytes per bulk prefetch (multiple of 16)
  int poll_delay_ns;             // first poll read not before this long after entry (publish) / own front publish
  int pad1;
  const __nv_bfloat16* wts;      // front mode: [M][16] bf16
  const __nv_bfloat16* sh;       // front mode: [M][7168] shared-expert partial
  unsigned long long lat_st;     // front mode: multicast (or plain) address of lat_mb
  unsigned long long rs_peer[kTp];  // front mode: rank d's rs_mb
};
static_assert(offsetof(T1Args, eps) + sizeof(float) <= 64 && offsetof(T1Args, gemm2) < 64, "poll-path params in line 0");

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred P1;\n\tT1_WAIT:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
      "@P1 bra T1_DONE;\n\tbra T1_WAIT;\n\tT1_DONE:\n\t}" ::"r"(bar),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void tma_1d(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(bar)
               : "memory");
}
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
__device__ __forceinline__ uint32_t no_neg_zero(uint32_t w) {  // bf16 -0.0 -> +0.0 (both halves)
  if ((w & 0xffffu) == 0x8000u) w &= 0xffff0000u;
  if ((w >> 16) == 0x8000u) w &= 0x0000ffffu;
  return w;
}
__device__ __forceinline__ uint4 no_neg_zero4(uint4 v) {
  return make_uint4(no_neg_zero(v.x), no_neg_zero(v.y), no_neg_zero(v.z), no_neg_zero(v.w));
}
// Poll load. T1_POLL_LD (experiments): 0 ld.volatile (default), 1 ld.relaxed.gpu, 2 ld.global.cv, 3 ld.acquire.sys,
// 4 ld.relaxed.sys
#ifndef T1_POLL_LD
#define T1_POLL_LD 0
#endif
__device__ __forceinline__ uint4 ld_volatile16(const void* p) {
  uint4 v;
#if T1_POLL_LD == 1
  asm volatile("ld.relaxed.gpu.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
#elif T1_POLL_LD == 2
  asm volatile("ld.global.cv.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
#elif T1_POLL_LD == 3
  asm volatile("ld.acquire.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
#elif T1_POLL_LD == 4
  asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
#else
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
#endif
  return v;
}
__device__ __forceinline__ void st_sentinel16(const void* p) {
  asm volatile("st.global.v4.u32 [%0], {%1, %1, %1, %1};" ::"l"(p), "r"(kSent) : "memory");
}
__device__ __forceinline__ void st_mb16(unsigned long long addr, const uint4& v, int mc) {
  if (mc)
    asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(addr), "r"(v.x), "r"(v.y),
                 "r"(v.z), "r"(v.w) : "memory");
  else
    asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}
__device__ __forceinline__ void st_mb8(unsigned long long addr, uint32_t w0, uint32_t w1, int mc) {
  if (mc)
    asm volatile("multimem.st.relaxed.sys.global.v2.f32 [%0], {%1, %2};" ::"l"(addr), "r"(w0), "r"(w1) : "memory");
  else
    asm volatile("st.global.v2.u32 [%0], {%1, %2};" ::"l"(addr), "r"(w0), "r"(w1) : "memory");
}
__device__ __forceinline__ float bf16_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// Any checked 32-bit word still the Lamport sentinel? Branch-free (compact code: cold instruction fetch is the
// dominant cost of this kernel's once-per-call code): unchecked words are OR-ed with 1, so they never equal it.
// um = per-word "unchecked" bits: {x, y, z, w} each 0 (checked) or 1 (ignored).
__device__ __forceinline__ bool dirty_m(const uint4& v, const uint4& um) {
  return ((v.x | um.x) == kSent) | ((v.y | um.y) == kSent) | ((v.z | um.z) == kSent) | ((v.w | um.w) == kSent);
}
// Poll one 16-byte word of each of the 16 sources until no checked 32-bit word is the sentinel. Every round
// re-issues ALL still-empty loads back to back (one round trip per round, not per late source).
__device__ __forceinline__ void poll16m(const uint4* base, long ss, const uint4 m, uint4 (&v)[kTp]) {
#pragma unroll
  for (int s = 0; s < kTp; ++s) v[s] = ld_volatile16(base + s * ss);
  uint32_t pend = 0;
#pragma unroll
  for (int s = 0; s < kTp; ++s) pend |= dirty_m(v[s], m) ? (1u << s) : 0u;
  while (pend) {
    __nanosleep(kPollNs);
#pragma unroll
    for (int s = 0; s < kTp; ++s)
      if (pend & (1u << s)) v[s] = ld_volatile16(base + s * ss);
    uint32_t np = 0;
#pragma unroll
    for (int s = 0; s < kTp; ++s) np |= ((pend >> s) & 1u) && dirty_m(v[s], m) ? (1u << s) : 0u;
    pend = np;
  }
}
// Value-halving warp sum of N accumulators (bit-identical to a full xor butterfly per accumulator; lane L ends
// with index `row`, built from its high lane bits).
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
// Partial up-proj of 8 rows (warp) x 448 columns for MT tokens t0 .. t0 + MT - 1: part[row * M + t].
// Per accumulator: FMA over 16-byte chunk `lane`, then `lane + 28` (lanes 0..27), then the warp butterfly --
// the same order as k3mk.tail (bit-identical partials).
template <int MT>
__device__ __forceinline__ void gemv8(const uint8_t* wsm, const __nv_bfloat16* xs, float* part, int warp, int lane,
                                      int t0, int M) {
  constexpr int kN = 8 * MT;
  float acc[kN];
#pragma unroll
  for (int i = 0; i < kN; ++i) acc[i] = 0.f;
  if (lane < kFrag / 2) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int ch = lane + (kFrag / 2) * h;
      float xv[MT][8];
#pragma unroll
      for (int t = 0; t < MT; ++t) {
        const uint4 u = reinterpret_cast<const uint4*>(xs + (t0 + t) * kCols)[ch];
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
      for (int r = 0; r < 8; ++r) {
        const uint4 wv = reinterpret_cast<const uint4*>(wsm + (warp * 8 + r) * kCols * 2)[ch];
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
  const float v = halving_sum<kN>(acc, lane, i);
  constexpr int kSteps = kN == 8 ? 3 : 4;
  if ((lane & ((32 >> kSteps) - 1)) == 0) {
    const int r = i / MT, t = i - r * MT;
    part[(warp * 8 + r) * M + t0 + t] = v;
  }
}

// Issue this CTA's share of the next layer's L2 prefetch (one thread): chunk c of the concatenated ranges goes to
// issuer c % nis (k3pf's split). cp.async.bulk.prefetch.L2 is a hint issued by the SM's TMA unit (~250 GB/s per SM,
// the thread stalls when the queue is full; too many issuers make the L2 drop prefetches, k3pf RESULTS).
__device__ __noinline__ void t1_prefetch(const T1Args& a, int issuer) {
  long long phase = 0;
  const long long chunk = a.pf_chunk, nis = a.pf_nis;
  for (int r = 0; r < a.pf_n; ++r) {
    const unsigned long long base = static_cast<unsigned long long>(__ldg(a.pf_rng + 2 * r));
    const long long bytes = __ldg(a.pf_rng + 2 * r + 1);
    if (base == 0 || bytes <= 0) continue;
    const long long nch = (bytes + chunk - 1) / chunk;
    long long c = issuer - (phase % nis);
    if (c < 0) c += nis;
    for (; c < nch; c += nis) {
      const long long off = c * chunk;
      long long sz = bytes - off < chunk ? bytes - off : chunk;
      sz &= ~15ll;
      if (sz <= 0) continue;
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"((base + off) & ~15ull), "r"(static_cast<uint32_t>(sz))
                   : "memory");
    }
    phase += nch;
  }
}

// smem: W [32][448] bf16 | gamma [448] | xs [M][448] bf16 | part [32][M] f32 | pr [8 q][32 rows][M] f32 (leader) |
//       ssr [8 q][2 w][M] f32 | shs [M][32] f32 (leader) | bars [4] | flag | trace [16]
__host__ __device__ constexpr int t1_smem(int M) {
  return kBoxR * kCols * 2 + kCols * 2 + M * kCols * 2 + kBoxR * M * 4 + kCl * kBoxR * M * 4 + kCl * 2 * M * 4 +
         M * kBoxR * 4 + 4 * 8 + 8 + 16 * 8;
}
static_assert(t1_smem(4) <= 40 * 1024, "LIGHT contract: <= 40 KB smem for the M <= 4 the server uses");
static_assert(t1_smem(kMmax) <= 48 * 1024, "M = 5..8: <= 48 KB");
constexpr int kLpt = kLat / 64;  // 128-B lines per token of a latent partial (56)
constexpr int kRpt = kHid / 64;  // 128-B lines per token of the shared-expert output (112; 7 per destination rank)

// NC: clusters (14 / 15). KM: tokens, compile-time for the hot M = 1 / 2 paths (no loops, no runtime M in the
// critical path: cold instruction fetch of branchy code dominated v1), 0 = runtime M (3..8, rounds of 2 tokens).
// SINGLE WRITER PER 128-B LINE (v4; agents/probe RESULTS s6.6: a line with several partial writers that SMs are
// polling takes the stores ~4 us late): front-mode latent / RS lines are each stored by ONE warp as 8 lanes x 16 B;
// the H4 rows of a cluster are summed and stored by the cluster leader (CTA 0) as one contiguous 64-B run (NC=14:
// two writers per H4 line instead of 16).
template <int NC, int KM>
__global__ void __maxnreg__(120) t1_kernel(const __grid_constant__ CUtensorMap tm_w, const __grid_constant__ T1Args a) {
  const int trig = (a.flags >> 4) & 3, pf_mode = a.flags & 3;
  if (trig == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // trigger at ENTRY
  const unsigned long long t_entry = gtimer();
  extern __shared__ __align__(128) uint8_t sm[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int gb = blockIdx.x, c = gb / kCl;
  const int M = KM ? KM : a.M;
  uint32_t q;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(q));
  // rows of this cluster: 56 8-row groups over NC clusters (NC=14: 32 rows each; NC=15: 11 x 32 + 4 x 24)
  constexpr int kBase = 56 / NC, kExtra = 56 % NC;
  const int nrows = 8 * (kBase + (c < kExtra ? 1 : 0));
  const int row0 = 8 * (c * kBase + min(c, kExtra));
  const bool leader = q == 0;  // owns (sums, adds the shared part to, publishes) all rows of the cluster
  const int nrs = nrows >> 3;  // 16-B RS chunks (8 rows each) per token
  int so = kBoxR * kCols * 2;
  const __nv_bfloat16* gs = reinterpret_cast<const __nv_bfloat16*>(sm + so);
  so += kCols * 2;
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(sm + so);
  so += M * kCols * 2;
  float* part = reinterpret_cast<float*>(sm + so);
  so += kBoxR * M * 4;
  float* pr = reinterpret_cast<float*>(sm + so);  // [8 q][32 rows][M] (leader)
  so += kCl * kBoxR * M * 4;
  float* ssr = reinterpret_cast<float*>(sm + so);
  so += kCl * 2 * M * 4;
  float* shs = reinterpret_cast<float*>(sm + so);  // [M][32] (leader)
  so += M * kBoxR * 4;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + so);  // [0] W, [1] sums of squares, [2] partials, [3] gamma
  int* flag = reinterpret_cast<int*>(bars + 4);
  unsigned long long* trs = reinterpret_cast<unsigned long long*>(bars + 5);
  const uint32_t bw = smem_u32(bars), bs = smem_u32(bars + 1), bp = smem_u32(bars + 2), bg = smem_u32(bars + 3);
  const bool tr = a.trace != nullptr;
#define T1_MARK(slot) \
  if (tr && tid == 0) trs[slot] = gtimer();
  if (tr && tid == 0) {
    trs[0] = t_entry;
    trs[6] = trs[7] = trs[9] = trs[10] = 0ull;
#ifdef T1_DIAG
    trs[11] = trs[12] = trs[13] = trs[14] = trs[15] = 0ull;
#endif
  }
#ifdef T1_DIAG
  __syncthreads();
#endif
  if (tid == kStage) {
    mbar_init(bw, 1);
    mbar_init(bs, 1);
    mbar_init(bp, 1);
    mbar_init(bg, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_arrive_expect_tx(bg, kCols * 2);
    tma_1d(smem_u32(gs), a.gamma + kCols * q, kCols * 2, bg);
    mbar_arrive_expect_tx(bs, kCl * 2 * M * 4);                     // the 8 CTAs' per-warp sums of squares
    mbar_arrive_expect_tx(bp, leader ? kCl * nrows * M * 4 : 0);   // the 8 CTAs' partials of the cluster's rows
#ifdef T1_NOTMA  // timing experiment only: no W_up staging (the GEMV reads garbage)
    mbar_arrive_expect_tx(bw, 0);
    if (false)
#else
    mbar_arrive_expect_tx(bw, kBoxR * kCols * 2);  // rows beyond 448 are zero-filled and still counted
#endif
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], "
        "[%5];" ::"r"(smem_u32(sm)),
        "l"(reinterpret_cast<uint64_t>(&tm_w)), "r"(0), "r"(2 * static_cast<int>(q)), "r"(row0), "r"(bw)
        : "memory");
  }
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");  // mbarriers initialised (fence above)
  if (pf_mode == 1 && tid == kStage && gb % a.pf_stride == 0) t1_prefetch(a, gb / a.pf_stride);  // before polling
  const long lat_par = static_cast<long>(a.par) * kTp * a.mmax;          // [par] in units of 3584-rows
  T1_MARK(1)
  // ---------------------------------------------------------------- front mode (unfinalized MoE output)
  if (a.gemm2 != nullptr) {
    asm volatile("griddepcontrol.wait;" ::: "memory");  // gemm2 / wts / shared_out come from the predecessor
    T1_MARK(9)
    constexpr int G = kCl * NC;
    // Tasks = whole 128-B lines: M * 56 latent lines (finalize) then M * 112 shared-out lines (RS copies); task
    // -> CTA task % G, warp task / G (warp-uniform). Each line is written by ONE warp: 8 lanes x 16 B.
    const int ntask = M * (kLpt + kRpt);
#pragma unroll 1
    for (int task = gb + G * warp; task < ntask; task += G * (kNT / 32)) {
      if (task < M * kLpt) {
        const int t = task / kLpt, L = task - t * kLpt;
        // lane: columns 64 L + 2 lane, +1; vLLM finalize_top16_bf16 per column (fp32 FMA over slots 0..15 in order,
        // one bf16 rounding), identical to the per-column arithmetic of every earlier version
        const __nv_bfloat16* src = a.gemm2 + static_cast<long>(t * kTopK) * kLat + 64 * L + 2 * lane;
        const uint4* wp = reinterpret_cast<const uint4*>(a.wts + t * kTopK);
        const uint4 w01 = wp[0], w23 = wp[1];
        const uint32_t ww[8] = {w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w};
        uint32_t vj[kTopK];
#pragma unroll
        for (int j = 0; j < kTopK; ++j) vj[j] = *reinterpret_cast<const uint32_t*>(src + static_cast<long>(j) * kLat);
        float a0 = 0.f, a1 = 0.f;
#pragma unroll
        for (int j = 0; j < kTopK; ++j) {
          const float wv = (j & 1) ? bf16_hi(ww[j >> 1]) : bf16_lo(ww[j >> 1]);
          a0 = fmaf(bf16_lo(vj[j]), wv, a0);
          a1 = fmaf(bf16_hi(vj[j]), wv, a1);
        }
        const __nv_bfloat162 b2 = __floats2bfloat162_rn(a0, a1);
        const uint32_t wd = no_neg_zero(*reinterpret_cast<const uint32_t*>(&b2));
        const uint32_t g0 = __shfl_sync(0xffffffffu, wd, (4 * lane) & 31);
        const uint32_t g1 = __shfl_sync(0xffffffffu, wd, (4 * lane + 1) & 31);
        const uint32_t g2 = __shfl_sync(0xffffffffu, wd, (4 * lane + 2) & 31);
        const uint32_t g3 = __shfl_sync(0xffffffffu, wd, (4 * lane + 3) & 31);
        if (lane < 8)
          st_mb16(a.lat_st + static_cast<unsigned long long>(((lat_par + a.rank * a.mmax + t) * kLat) + 64 * L + 8 * lane) * 2,
                  make_uint4(g0, g1, g2, g3), a.mc);
#ifdef T1_DIAG
        if (tr && tid == 0) {  // timestamp that cannot issue before the finalized value exists
          unsigned long long tv;
          asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tv) : "r"(wd));
          trs[14] = tv;
        }
#endif
      } else if (lane < 8) {  // shared-out line -> the owner rank's RS mailbox (7 whole lines per 448-column shard)
        const int u = task - M * kLpt, t = u / kRpt, Ls = u - t * kRpt;
        const int d = Ls / 7, cl = Ls - 7 * d;
        const uint4 v = no_neg_zero4(*reinterpret_cast<const uint4*>(a.sh + static_cast<long>(t) * kHid + 64 * Ls + 8 * lane));
        st_mb16(a.rs_peer[d] + static_cast<unsigned long long>(
                                   ((static_cast<long>(a.par) * kTp + a.rank) * a.mmax + t) * kShard + 64 * cl + 8 * lane) * 2,
                v, 0);
      }
    }
#ifdef T1_FRONT_FENCE  // A/B knob (v3 finding; the single-writer lines should make it unnecessary)
    asm volatile("fence.acq_rel.sys;" ::: "memory");
#endif
    T1_MARK(10)
  }
  // K3T1_POLL_DELAY_NS holds the first poll read back (front mode: from our own publish; publish mode: from entry).
  if (a.poll_delay_ns > 0) {
    const unsigned long long t_d = (a.gemm2 != nullptr ? gtimer() : t_entry) + static_cast<unsigned>(a.poll_delay_ns);
    while (gtimer() < t_d) {
    }
  }
  // ---------------------------------------------------------------- poll (rounds of 2 tokens)
  const int pt = tid >> 6, pk = tid & 63;
  const long lss = static_cast<long>(a.mmax) * kLat / 8;    // uint4 between latent sources
  const long rss = static_cast<long>(a.mmax) * kShard / 8;  // uint4 between RS sources
  // RS readers (leader only), chunk j = rows [row0 + 8 j, +8) of the token: M = 1 -> tid 64 + j (warp 2, idle at M = 1,
  // so a late RS word cannot hold back a latent warp's RMS push); M >= 2 -> lanes 56 + j of the token's odd warp.
  int rs_j = -1;
  if (leader) {
    const int jj = KM == 1 ? tid - 64 : pk - kFrag;
    if (jj >= 0 && jj < nrs) rs_j = jj;
  }
#pragma unroll 1
  for (int t0 = 0; t0 < M; t0 += 2) {
    const int t_ = t0 + pt;
    const bool lat_th = t_ < M && pk < kFrag;
    const bool rs_th = rs_j >= 0 && (KM == 1 ? t0 == 0 : t_ < M);
    const int t = (KM == 1 && rs_th) ? 0 : t_;
    float s2 = 0.f;
    if (lat_th || rs_th) {
      // one code path for latent and RS readers (no divergence in the poll)
      const uint4* base = lat_th ? reinterpret_cast<const uint4*>(a.lat_mb + (lat_par + t) * kLat + kCols * q + 8 * pk)
                                 : reinterpret_cast<const uint4*>(a.rs_mb + (static_cast<long>(a.par) * kTp * a.mmax + t) *
                                                                                kShard + row0 + 8 * rs_j);
      const long ss = lat_th ? lss : rss;
      uint4 v[kTp];
      poll16m(base, ss, make_uint4(0u, 0u, 0u, 0u), v);
#ifdef T1_DIAG
      if (tr && t0 == 0) {
        if (tid == 0) trs[12] = gtimer();
        atomicMax(&trs[lat_th ? 13 : 11], gtimer());  // 13: latent pollers, 11: RS readers
      }
#endif
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
      if (lat_th) {
        uint32_t xw[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const __nv_bfloat162 b2 = __floats2bfloat162_rn(acc[2 * k], acc[2 * k + 1]);
          xw[k] = *reinterpret_cast<const uint32_t*>(&b2);
          const __nv_bfloat162 sq2 = __hmul2(b2, b2);  // bf16 squares (vLLM-compatible)
          s2 += __low2float(sq2);
          s2 += __high2float(sq2);
        }
        reinterpret_cast<uint4*>(xs + t * kCols)[pk] = make_uint4(xw[0], xw[1], xw[2], xw[3]);
      } else {
#pragma unroll
        for (int e = 0; e < 8; ++e) shs[t * kBoxR + 8 * rs_j + e] = __bfloat162float(__float2bfloat16(acc[e]));
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s2 += __shfl_xor_sync(0xffffffffu, s2, o);  // fixed-order butterfly
    if (t0 == 0) asm volatile("barrier.cluster.wait.aligned;" ::: "memory");  // peers' mbarriers initialised
    const int tw = t0 + (warp >> 1);  // token of this warp in this round
    if (tw < M && lane < kCl) {
      const uint32_t la = smem_u32(ssr + (static_cast<int>(q) * 2 + (warp & 1)) * M + tw);
      st_async4(mapa_u32(la, lane), s2, mapa_u32(bs, lane));
    }
  }
  if (trig == 1) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // after the poll
  T1_MARK(2)
  mbar_wait(bs, 0);
  T1_MARK(3)
  mbar_wait(bg, 0);  // landed long ago
#pragma unroll 1
  for (int t0 = 0; t0 < M; t0 += 2) {  // xn of this CTA's own columns, in place
    const int t = t0 + pt;
    if (t < M && pk < kFrag) {
      float tot = 0.f;
#pragma unroll
      for (int d = 0; d < kCl; ++d) tot = (tot + ssr[(2 * d) * M + t]) + ssr[(2 * d + 1) * M + t];  // vLLM order
      float r;
      asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(__fdiv_rn(tot, static_cast<float>(kLat)) + a.eps));
      const uint4 gam = reinterpret_cast<const uint4*>(gs)[pk];
      const uint4 xv = reinterpret_cast<const uint4*>(xs + t * kCols)[pk];
      const uint32_t gw[4] = {gam.x, gam.y, gam.z, gam.w};
      const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
      uint32_t ow[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const __nv_bfloat162 r2 = __floats2bfloat162_rn(bf16_lo(xw[k]) * r * bf16_lo(gw[k]),
                                                        bf16_hi(xw[k]) * r * bf16_hi(gw[k]));
        ow[k] = *reinterpret_cast<const uint32_t*>(&r2);
      }
      reinterpret_cast<uint4*>(xs + t * kCols)[pk] = make_uint4(ow[0], ow[1], ow[2], ow[3]);
    }
  }
  mbar_wait(bw, 0);  // W landed (long ago)
  __syncthreads();  // every thread of this CTA has finished reading lat_mb / rs_mb
  T1_MARK(4)
  // ---------------------------------------------------------------- partial up-proj (warp = 8 rows)
  if (warp * 8 < nrows) {
    if constexpr (KM == 1) {
      gemv8<1>(sm, xs, part, warp, lane, 0, 1);
    } else if constexpr (KM == 2) {
      gemv8<2>(sm, xs, part, warp, lane, 0, 2);
    } else {
#pragma unroll 1
      for (int t0 = 0; t0 < M; t0 += 2) {
        if (M - t0 >= 2)
          gemv8<2>(sm, xs, part, warp, lane, t0, M);
        else
          gemv8<1>(sm, xs, part, warp, lane, t0, M);
      }
    }
  }
  __syncthreads();
  // partials of the cluster's rows -> the leader: pr[our q][row][t] of CTA 0
#pragma unroll 1
  for (int i = tid; i < nrows * M; i += kNT) {
    const int r = i / M, t = i - r * M;
    const uint32_t la = smem_u32(pr + (static_cast<int>(q) * kBoxR + r) * M + t);
    st_async4(mapa_u32(la, 0), part[i], mapa_u32(bp, 0));
  }
  T1_MARK(5)
  if (leader) {  // warp w: token w (+4): lane = row; fixed-order sum of the 8 partials, bf16, + shared, bf16
#pragma unroll 1
    for (int t = warp; t < M; t += kNT / 32) {
      mbar_wait(bp, 0);
      T1_MARK(6)
      uint32_t hb = 0;
      if (lane < nrows) {
        float v = 0.f;
#pragma unroll
        for (int d = 0; d < kCl; ++d) v += pr[(d * kBoxR + lane) * M + t];
        const float gv = __bfloat162float(__float2bfloat16(v));
        const __nv_bfloat16 ob = __float2bfloat16(gv + shs[t * kBoxR + lane]);
        hb = *reinterpret_cast<const uint16_t*>(&ob);
      }
      const uint32_t wd = no_neg_zero(hb | (__shfl_down_sync(0xffffffffu, hb, 1) << 16));  // rows 2k, 2k+1 at lane 2k
      const uint32_t g0 = __shfl_sync(0xffffffffu, wd, (8 * lane) & 31);
      const uint32_t g1 = __shfl_sync(0xffffffffu, wd, (8 * lane + 2) & 31);
      const uint32_t g2 = __shfl_sync(0xffffffffu, wd, (8 * lane + 4) & 31);
      const uint32_t g3 = __shfl_sync(0xffffffffu, wd, (8 * lane + 6) & 31);
      if (lane < nrs)  // the cluster's rows as ONE contiguous run: nrs lanes x 16 B (64 B at NC=14)
        st_mb16(a.up_st + static_cast<unsigned long long>(t * kHid + a.rank * kShard + row0 + 8 * lane) * 2,
                make_uint4(g0, g1, g2, g3), a.mc);
    }
    T1_MARK(7)
  }
  if (trig == 2) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // default: after the H4 publish
  // ---------------------------------------------------------------- re-arm (after the publish)
  if (rs_j >= 0) {  // RS readers: one reader per 16-B RS chunk
#pragma unroll 1
    for (int t = KM == 1 ? 0 : pt; t < M; t += 2) {
      const __nv_bfloat16* rb = a.rs_mb + (static_cast<long>(a.par) * kTp * a.mmax + t) * kShard + row0 + 8 * rs_j;
#pragma unroll 1
      for (int s = 0; s < kTp; ++s) st_sentinel16(rb + static_cast<long>(s) * rss * 8);
    }
  }
  if (tid == 0) {
    unsigned long long old;
    asm volatile("atom.add.relaxed.gpu.global.u64 %0, [%1], 1;" : "=l"(old) : "l"(a.cnt + a.par * kCl + q) : "memory");
    *flag = (old % static_cast<unsigned long long>(NC)) == static_cast<unsigned long long>(NC - 1);
  }
  __syncthreads();
  if (*flag) {  // the NC-th cluster to read columns [448 q, +448) of lat_mb[par] re-arms them
#pragma unroll 1
    for (int i = tid; i < kTp * M * kFrag; i += kNT) {
      const int s = i / (M * kFrag), rr = i - s * (M * kFrag), t = rr / kFrag, k = rr - t * kFrag;
      st_sentinel16(a.lat_mb + ((lat_par + static_cast<long>(s) * a.mmax + t) * kLat) + kCols * q + 8 * k);
    }
  }
  if (pf_mode == 2 && tid == kStage && gb % a.pf_stride == 0) t1_prefetch(a, gb / a.pf_stride);  // after the H4 publish
  // No exit barrier: a CTA may exit once no peer can still write into its smem, and every DSMEM push INTO this CTA
  // has landed (all threads waited on bs; the leader's owner warps waited on bp, which counts every partial).
#ifdef T1_EXIT_BARRIER
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
#endif
  T1_MARK(8)
#ifdef T1_DIAG
  if (tr && tid < 16) a.trace[gb * 16 + tid] = trs[tid];
#else
  if (tr && tid < 16) a.trace[gb * 16 + tid] = (tid <= 10) ? trs[tid] : 0ull;
#endif
#undef T1_MARK
}

PFN_cuTensorMapEncodeTiled_v12000 tensor_map_encoder() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (fn == nullptr) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult qr;
    C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &qr));
    TORCH_CHECK(p != nullptr && qr == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled unavailable");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}
// W_up shard [448][3584] bf16 as [row][16 blocks][224]; box = 32 rows x 2 blocks x 224 = 32 x 448 columns.
CUtensorMap make_wup_map(const void* base) {
  CUtensorMap m;
  const cuuint64_t dims[3] = {224, 16, static_cast<cuuint64_t>(kShard)};
  const cuuint64_t strides[2] = {448, static_cast<cuuint64_t>(kLat * 2)};
  const cuuint32_t box[3] = {224, 2, static_cast<cuuint32_t>(kBoxR)};
  const cuuint32_t es[3] = {1, 1, 1};
  const CUresult r = tensor_map_encoder()(&m, CU_TENSOR_MAP_DATA_TYPE_UINT16, 3, const_cast<void*>(base), dims, strides,
                                          box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                          CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (w_up) failed: ", static_cast<int>(r));
  return m;
}
}  // namespace

using T1Kern = void (*)(CUtensorMap, T1Args);
template <int NC>
T1Kern pick_km(int M) {
  return M == 1 ? t1_kernel<NC, 1> : M == 2 ? t1_kernel<NC, 2> : t1_kernel<NC, 0>;
}
T1Kern pick(int nc, int M) { return nc == 14 ? pick_km<14>(M) : pick_km<15>(M); }

// k3t1.tail: see t1_kernel and T1_INTERFACE.md. Same leading arguments as k3mk.tail (variant = cluster count), plus
// the optional in-kernel next-layer L2 prefetch: pf_ranges int64 [n, 2] (ptr, bytes) device table, pf_mode 1 = issue
// at entry (before polling) / 2 = after the H4 publish, pf_issuers issuing CTAs (0 = default: 120 for mode 1,
// 32 for mode 2), pf_chunk bytes per prefetch (0 = default: 64 KB mode 1, 16 KB mode 2).
void tail(torch::Tensor lat_mb, torch::Tensor rs_mb, int64_t par, torch::Tensor w_up, torch::Tensor gamma, double eps,
          torch::Tensor up_mb, int64_t up_mc_ptr, torch::Tensor cnt, int64_t rank, int64_t M,
          std::optional<torch::Tensor> trace, std::optional<torch::Tensor> gemm2, std::optional<torch::Tensor> wts,
          std::optional<torch::Tensor> shared_out, int64_t lat_mc_ptr, std::optional<std::vector<int64_t>> rs_peers,
          bool multicast, int64_t variant, std::optional<torch::Tensor> pf_ranges, int64_t pf_mode, int64_t pf_issuers,
          int64_t pf_chunk) {
  TORCH_CHECK(M >= 1 && M <= kMmax, "k3t1.tail supports 1..8 tokens");
  TORCH_CHECK(par >= 0 && par < 3 && rank >= 0 && rank < kTp);
  auto bf = [](const torch::Tensor& t) { return t.scalar_type() == at::kBFloat16 && t.is_contiguous() && t.is_cuda(); };
  TORCH_CHECK(bf(lat_mb) && lat_mb.dim() == 4 && lat_mb.size(0) == 3 && lat_mb.size(1) == kTp && lat_mb.size(3) == kLat,
              "lat_mb: bf16 [3, 16, Mmax, 3584]");
  const int mmax = lat_mb.size(2);
  TORCH_CHECK(M <= mmax && mmax <= kMmax);
  TORCH_CHECK(bf(rs_mb) && rs_mb.dim() == 4 && rs_mb.size(0) == 3 && rs_mb.size(1) == kTp && rs_mb.size(2) == mmax &&
              rs_mb.size(3) == kShard, "rs_mb: bf16 [3, 16, Mmax, 448]");
  TORCH_CHECK(bf(w_up) && w_up.size(0) == kShard && w_up.size(1) == kLat &&
              reinterpret_cast<uintptr_t>(w_up.data_ptr()) % 16 == 0, "w_up: bf16 [448, 3584]");
  TORCH_CHECK(bf(gamma) && gamma.numel() == kLat && reinterpret_cast<uintptr_t>(gamma.data_ptr()) % 16 == 0);
  TORCH_CHECK(bf(up_mb) && up_mb.size(-1) == kHid && up_mb.numel() >= M * kHid);
  TORCH_CHECK(up_mc_ptr != 0 && up_mc_ptr % 16 == 0);
  TORCH_CHECK(cnt.scalar_type() == at::kLong && cnt.is_cuda() && cnt.numel() >= 3 * kCl);
  // NC = 14 (default since v4): 32-row clusters align with the 64-row H4 lines (two writers per line)
  static const int env_nc = getenv("K3T1_NC") ? atoi(getenv("K3T1_NC")) : 14;
  const int nc = variant ? static_cast<int>(variant) : env_nc;
  TORCH_CHECK(nc == 14 || nc == 15, "k3t1.tail: variant (clusters) must be 14 or 15");
  const int G = kCl * nc;
  T1Args a{};
  a.M = static_cast<int>(M);
  a.par = static_cast<int>(par);
  a.mmax = mmax;
  a.rank = static_cast<int>(rank);
  a.trace = trace ? reinterpret_cast<unsigned long long*>(trace->data_ptr()) : nullptr;
  if (trace) TORCH_CHECK(trace->numel() * trace->element_size() >= 8 * 16 * G, "trace: int64 [>= G*16]");
  a.lat_mb = reinterpret_cast<__nv_bfloat16*>(lat_mb.data_ptr());
  a.rs_mb = reinterpret_cast<__nv_bfloat16*>(rs_mb.data_ptr());
  a.gamma = reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr());
  a.eps = static_cast<float>(eps);
  a.mc = multicast ? 1 : 0;
  a.nc = nc;
  a.up_st = static_cast<unsigned long long>(up_mc_ptr);
  a.cnt = reinterpret_cast<unsigned long long*>(cnt.data_ptr());
  if (gemm2) {
    TORCH_CHECK(bf(*gemm2) && gemm2->dim() == 2 && gemm2->size(0) == M * kTopK && gemm2->size(1) == kLat);
    TORCH_CHECK(wts && bf(*wts) && wts->numel() == M * kTopK, "front mode: wts [M, 16] bf16");
    TORCH_CHECK(shared_out && bf(*shared_out) && shared_out->numel() == M * kHid, "front mode: shared_out [M, 7168]");
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
  // PDL trigger point: env K3T1_TRIG = 2 after the H4 publish (default since the in-server A/B t1tr2: B1 4.826 vs
  // stack1 4.895; at entry (0) the next layer's LAR + A7 in_proj cascade landed ~8 us early: +0.73 ms) | 0 entry |
  // 1 after the poll (better with a BIG polling successor, stage 1b)
  static const int trig = getenv("K3T1_TRIG") ? atoi(getenv("K3T1_TRIG")) : 2;
  static const int pdelay = getenv("K3T1_POLL_DELAY_NS") ? atoi(getenv("K3T1_POLL_DELAY_NS")) : 0;
  a.poll_delay_ns = pdelay;
  TORCH_CHECK(trig >= 0 && trig <= 2, "K3T1_TRIG must be 0, 1 or 2");
  a.flags = trig << 4;
  if (pf_mode != 0 && pf_ranges && pf_ranges->numel() > 0) {
    TORCH_CHECK(pf_mode == 1 || pf_mode == 2, "pf_mode: 0 off, 1 entry, 2 after the publish");
    TORCH_CHECK(pf_ranges->is_cuda() && pf_ranges->scalar_type() == at::kLong && pf_ranges->is_contiguous() &&
                pf_ranges->numel() % 2 == 0, "pf_ranges: int64 CUDA [n, 2] (ptr, bytes)");
    const int nis = pf_issuers > 0 ? static_cast<int>(std::min<int64_t>(pf_issuers, G)) : (pf_mode == 1 ? G : 32);
    a.flags |= static_cast<int>(pf_mode);
    a.pf_rng = reinterpret_cast<const long long*>(pf_ranges->data_ptr<int64_t>());
    a.pf_n = static_cast<int>(pf_ranges->numel() / 2);
    a.pf_stride = std::max(1, G / nis);
    a.pf_nis = (G + a.pf_stride - 1) / a.pf_stride;
    const int64_t ch = pf_chunk > 0 ? pf_chunk : (pf_mode == 1 ? 65536 : 16384);
    TORCH_CHECK(ch % 16 == 0 && ch <= (1 << 20), "pf_chunk: multiple of 16, <= 1 MiB");
    a.pf_chunk = static_cast<int>(ch);
  } else {
    a.pf_stride = 1;
  }
  // tensor map of the up_proj shard, cached per pointer; passed by value (captured in CUDA graphs)
  static std::vector<std::pair<const void*, CUtensorMap>> maps;
  const CUtensorMap* tm = nullptr;
  for (auto& e : maps)
    if (e.first == w_up.data_ptr()) tm = &e.second;
  if (!tm) {
    maps.push_back({w_up.data_ptr(), make_wup_map(w_up.data_ptr())});
    tm = &maps.back().second;
  }
  static bool init = false;
  if (!init) {
    // The SM's L1/shared split is configured by the kernels resident on it: a LIGHT kernel with the default
    // preference gets a small shared carveout, and a BIG CTA (~189 KB) then cannot land on that SM until it drains
    // (measured: BIG successor last CTA start 12 us -> 4.3 us after MoE end). Ask for the maximum shared carveout so
    // a BIG successor can co-reside (K3T1_CARVEOUT=-1 restores the default).
    static const int carve = getenv("K3T1_CARVEOUT") ? atoi(getenv("K3T1_CARVEOUT")) : 100;
    for (int n2 : {14, 15})
      for (int m2 : {1, 2, 3}) {
        const T1Kern k = pick(n2, m2);
        C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, t1_smem(kMmax)));
        if (carve >= 0) C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, carve));
      }
    init = true;
  }
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(G);
  cfg.blockDim = dim3(kNT);
  cfg.dynamicSmemBytes = t1_smem(static_cast<int>(M));
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  attr[1].id = cudaLaunchAttributeClusterDimension;
  attr[1].val.clusterDim.x = kCl;
  attr[1].val.clusterDim.y = 1;
  attr[1].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 2;
  const CUtensorMap tmv = *tm;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, pick(nc, static_cast<int>(M)), tmv, a));
}

// k3t1.info(): per instantiation (NC 15, M = 1 / 2 / 3..8): [numRegs, localSizeBytes, static smem], then smem(M=1),
// smem(M=8)
std::vector<int64_t> info() {
  std::vector<int64_t> r;
  for (int m2 : {1, 2, 3}) {
    cudaFuncAttributes fa{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, pick(15, m2)));
    r.push_back(fa.numRegs);
    r.push_back(static_cast<int64_t>(fa.localSizeBytes));
    r.push_back(static_cast<int64_t>(fa.sharedSizeBytes));
  }
  r.push_back(t1_smem(1));
  r.push_back(t1_smem(kMmax));
  return r;
}

// op namespace (a second build with -DK3T1_NS=<name> can coexist in one process, e.g. for A/B tests)
#ifndef K3T1_NS
#define K3T1_NS k3t1
#endif
#define K3T1_LIBRARY(ns, m) TORCH_LIBRARY(ns, m)
#define K3T1_LIBRARY_IMPL(ns, k, m) TORCH_LIBRARY_IMPL(ns, k, m)
K3T1_LIBRARY(K3T1_NS, m) {
  m.def("tail(Tensor(a!) lat_mb, Tensor(b!) rs_mb, int par, Tensor w_up, Tensor gamma, float eps, Tensor(c!) up_mb, "
        "int up_mc_ptr, Tensor(d!) cnt, int rank, int M, Tensor(e!)? trace=None, Tensor? gemm2=None, Tensor? wts=None, "
        "Tensor? shared_out=None, int lat_mc_ptr=0, int[]? rs_peers=None, bool multicast=True, int variant=0, "
        "Tensor? pf_ranges=None, int pf_mode=0, int pf_issuers=0, int pf_chunk=0) -> ()");
  m.def("info() -> int[]", &info);
}
K3T1_LIBRARY_IMPL(K3T1_NS, CUDA, m) { m.impl("tail", &tail); }
