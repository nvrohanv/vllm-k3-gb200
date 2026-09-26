// tcgen05 GEMV over pre-staged weights for the ATTN front (in_proj 3216 x 7168 bf16, M = 1..8 tokens). Agent l2pf.
//
// Grid: 15 clusters x 8 CTAs (120 CTAs, 1 per SM, <= 200 KB smem). Cluster c owns a set of output rows; CTA j of
// the cluster owns the K slice [896 j, 896 j + 896) of all of them (split-K 8, reduced through DSMEM).
// Per CTA, for its K slice:
//   * TMEM tile: 128 rows x 896 K (448 TMEM columns), staged TMA -> smem ring (SW128) -> tcgen05.cp -> TMEM.
//     After landing: swap-AB tcgen05.mma kind::f16, A = weights from TMEM (M = 128), B = x (N = 8, SW128 smem).
//     4 issuing warps, each into its own 8-column accumulator (one issuer tops out at ~250 B/clk; 2-3 reach
//     the 512 B/clk TMEM-A rate).
//   * smem tile: cfg 0: 64 rows x 896 K, M = 64 MMAs;  cfg 1: 88-row boxes x 896 K, M = 128 MMAs on an
//     overlapped layout (chunk stride 88 rows; lanes >= rows are ignored).
//   * CUDA-core rows (cfg 0 only, <= 24 rows): fma.rn.f32.bf16 dot products from smem, overlapped with the tensor
//     pipe (M <= 4).
// Warps: 0 TMA producer (never polls), 1 TMEM alloc + tcgen05.cp, 2-3 mailbox pollers (3 then issues the smem-tile
// MMAs after the TMEM-tile MMAs), 4-7 TMEM-tile MMA issuers, then CUDA rows, then TMEM epilogue (warp w reads lane
// quadrant w % 4), 8-15 CUDA-core rows.
// Reduction: every lane that finishes a partial row pushes it straight to the row's owner CTA (27 rows per rank)
// with st.async + mbarrier complete_tx (no CTA or cluster barrier on the critical path); the owner sums the 8 K
// slices in rank order and rounds once to bf16.
// Mailbox: the last of the 120 readers re-arms it right after reading (relaxed counter, no fence). No release fence
// anywhere in the kernel, so nothing waits for in-flight TMA loads.
// attnres_inproj (AR = true): the same kernel with the pre-attention AttnRes of lamport_attn_res in front of the
// GEMV, run by warps 8..15 on this CTA's K slice (block statistics and B = sum_s e_s v_s before landing; after
// landing P = bf16(prefix + delta), 4 statistics per token exchanged across the cluster with st.async, then
// x = (ca B + cp P) * out_w written into the B tile). prefix / blocks[write_idx] (and optionally out) are written by
// cluster 0 once every CTA has read the old prefix (monotonic epoch counter, relaxed).
#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cudaTypedefs.h>
#include <map>
#include <tuple>
#include "tc_utils.cuh"

namespace {
constexpr int K = 7168, KS = 896, KCH = 14, CL = 8, NCL = 15, NT = 512;
constexpr int ROWB = K * 2, SLICEB = KS * 2;  // 14336, 1792
constexpr int OWN = 27, PROWS = OWN * CL;     // 216 partial rows per cluster
constexpr int MAXCU = 24;
constexpr int TA = 64;                        // TMEM A tile columns [64, 512)
constexpr int RING = 3, SLOTB = 32768;        // ring: 3 x (2 chunks x 128 rows x 128 B)
constexpr int XBAR = 1, SBAR = 3;             // named barriers

struct Params {
  const __nv_bfloat16* w;
  const uint32_t* mailbox;       // [M][K] bf16 viewed as 32-bit words; 0x80000000 = empty
  __nv_bfloat16* y;              // [M][N]
  int* done;                     // mailbox reader count (the last reader re-arms and resets it)
  unsigned long long* gate;      // optional emulated landing time (0 = not yet published)
  unsigned long long* tl;        // optional [grid][16] timeline
  float* dbg;                    // optional [grid][216][8] partials (local row, token)
  int N, sm_m, sm_box, r_bytes, c_off, b_off, i_off, bar_off, detect, poll_ns, poll_mode, dbgf;
  int ef;  // 1: in_proj weight TMA with .L2::cache_hint evict_first (read-once stream; op arg poll += 100000)
  int t_row0[NCL], s_row0[NCL], s_rows[NCL], c_row0[NCL], c_rows[NCL];
  int mv;  // valid tokens (<= M); rows >= mv of x are zero and their outputs are not stored
  // AttnRes front (attnres_inproj only)
  __nv_bfloat16* prefix;
  __nv_bfloat16* blocks;
  const __nv_bfloat16* norm_w;
  const __nv_bfloat16* qk_w;
  const __nv_bfloat16* out_w;
  __nv_bfloat16* out;  // optional
  long long prefix_sm, blocks_sm, blocks_sr, out_sm;
  int nb, widx, ar_off;
  float eps, out_eps;
  unsigned long long* pref_cnt;  // monotonic count of CTAs that have read the old prefix
  const long long* pf_rng;       // optional [pf_n][2] (ptr, bytes): L2-prefetched once the in_proj is staged
  int pf_n;
};
constexpr int ARB = 5, DBAR = 6;  // named barriers of the AttnRes warps
constexpr int ar_bytes(int M) { return 6912 + M * 28 * 20 * 4 + M * 256 + M * 40; }  // AttnRes smem scratch
constexpr float kInvH = 1.0f / K;

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)::"memory");  // "memory": keep probes in program order
  return t;
}
// Timer read ordered after `dep` is available (ptxas may otherwise hoist a %globaltimer read above a bar.sync).
__device__ __forceinline__ unsigned long long gtimer_after(float dep) {
  unsigned long long t;
  asm volatile("{ .reg .f32 d; mov.f32 d, %1; mov.u64 %0, %globaltimer; }" : "=l"(t) : "f"(dep) : "memory");
  return t;
}
__host__ __device__ constexpr uint32_t idesc_bf16(int M, int N) {
  return (1u << 4) | (1u << 7) | (1u << 10) | (static_cast<uint32_t>(N >> 3) << 17) |
         (static_cast<uint32_t>(M >> 4) << 24);
}
// 4 MMAs (one 64-element K chunk), whole warp, one elected lane. A from TMEM columns a, a+8, a+16, a+24.
__device__ __forceinline__ void mma4_ts(uint32_t d, uint32_t a, uint64_t b, uint32_t idesc, uint32_t acc0) {
  asm volatile(
      "{\n\t.reg .pred e, p, pt;\n\t.reg .b32 a1, a2, a3;\n\t.reg .b64 b1, b2, b3;\n\t"
      "elect.sync _|e, 0xffffffff;\n\tsetp.ne.b32 p, %4, 0;\n\tsetp.eq.b32 pt, 0, 0;\n\t"
      "add.u32 a1, %1, 8;\n\tadd.u32 a2, %1, 16;\n\tadd.u32 a3, %1, 24;\n\t"
      "add.s64 b1, %2, 2;\n\tadd.s64 b2, %2, 4;\n\tadd.s64 b3, %2, 6;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a1], b1, %3, pt;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a2], b2, %3, pt;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a3], b3, %3, pt;\n\t}" ::"r"(d),
      "r"(a), "l"(b), "r"(idesc), "r"(acc0)
      : "memory");
}
// All MMAs of one issuing warp (NCH K chunks = 4 NCH MMAs) in one asm block: one elect, no per-MMA loop overhead.
__device__ __forceinline__ void mma_ts_chunks4(uint32_t d, uint32_t a, uint64_t b, uint32_t idesc) {
  asm volatile(
      "{\n\t.reg .pred e, p0, pt;\n\t.reg .b32 a<16>;\n\t.reg .b64 b<16>;\n\t"
      "elect.sync _|e, 0xffffffff;\n\tsetp.eq.b32 pt, 0, 0;\n\tsetp.ne.b32 p0, 0, 0;\n\t"
      "add.u32 a0, %1, 0;\n\tadd.s64 b0, %2, 0;\n\t"
      "add.u32 a1, %1, 8;\n\tadd.s64 b1, %2, 2;\n\t"
      "add.u32 a2, %1, 16;\n\tadd.s64 b2, %2, 4;\n\t"
      "add.u32 a3, %1, 24;\n\tadd.s64 b3, %2, 6;\n\t"
      "add.u32 a4, %1, 32;\n\tadd.s64 b4, %2, 64;\n\t"
      "add.u32 a5, %1, 40;\n\tadd.s64 b5, %2, 66;\n\t"
      "add.u32 a6, %1, 48;\n\tadd.s64 b6, %2, 68;\n\t"
      "add.u32 a7, %1, 56;\n\tadd.s64 b7, %2, 70;\n\t"
      "add.u32 a8, %1, 64;\n\tadd.s64 b8, %2, 128;\n\t"
      "add.u32 a9, %1, 72;\n\tadd.s64 b9, %2, 130;\n\t"
      "add.u32 a10, %1, 80;\n\tadd.s64 b10, %2, 132;\n\t"
      "add.u32 a11, %1, 88;\n\tadd.s64 b11, %2, 134;\n\t"
      "add.u32 a12, %1, 96;\n\tadd.s64 b12, %2, 192;\n\t"
      "add.u32 a13, %1, 104;\n\tadd.s64 b13, %2, 194;\n\t"
      "add.u32 a14, %1, 112;\n\tadd.s64 b14, %2, 196;\n\t"
      "add.u32 a15, %1, 120;\n\tadd.s64 b15, %2, 198;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a0], b0, %3, p0;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a1], b1, %3, pt;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a2], b2, %3, pt;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a3], b3, %3, pt;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a4], b4, %3, pt;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a5], b5, %3, pt;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a6], b6, %3, pt;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a7], b7, %3, pt;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a8], b8, %3, pt;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a9], b9, %3, pt;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a10], b10, %3, pt;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a11], b11, %3, pt;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a12], b12, %3, pt;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a13], b13, %3, pt;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a14], b14, %3, pt;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a15], b15, %3, pt;\n\t"
      "}" ::"r"(d), "r"(a), "l"(b), "r"(idesc) : "memory");
}
__device__ __forceinline__ void mma_ts_chunks3(uint32_t d, uint32_t a, uint64_t b, uint32_t idesc) {
  asm volatile(
      "{\n\t.reg .pred e, p0, pt;\n\t.reg .b32 a<12>;\n\t.reg .b64 b<12>;\n\t"
      "elect.sync _|e, 0xffffffff;\n\tsetp.eq.b32 pt, 0, 0;\n\tsetp.ne.b32 p0, 0, 0;\n\t"
      "add.u32 a0, %1, 0;\n\tadd.s64 b0, %2, 0;\n\t"
      "add.u32 a1, %1, 8;\n\tadd.s64 b1, %2, 2;\n\t"
      "add.u32 a2, %1, 16;\n\tadd.s64 b2, %2, 4;\n\t"
      "add.u32 a3, %1, 24;\n\tadd.s64 b3, %2, 6;\n\t"
      "add.u32 a4, %1, 32;\n\tadd.s64 b4, %2, 64;\n\t"
      "add.u32 a5, %1, 40;\n\tadd.s64 b5, %2, 66;\n\t"
      "add.u32 a6, %1, 48;\n\tadd.s64 b6, %2, 68;\n\t"
      "add.u32 a7, %1, 56;\n\tadd.s64 b7, %2, 70;\n\t"
      "add.u32 a8, %1, 64;\n\tadd.s64 b8, %2, 128;\n\t"
      "add.u32 a9, %1, 72;\n\tadd.s64 b9, %2, 130;\n\t"
      "add.u32 a10, %1, 80;\n\tadd.s64 b10, %2, 132;\n\t"
      "add.u32 a11, %1, 88;\n\tadd.s64 b11, %2, 134;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a0], b0, %3, p0;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a1], b1, %3, pt;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a2], b2, %3, pt;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a3], b3, %3, pt;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a4], b4, %3, pt;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a5], b5, %3, pt;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a6], b6, %3, pt;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a7], b7, %3, pt;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a8], b8, %3, pt;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a9], b9, %3, pt;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a10], b10, %3, pt;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], [a11], b11, %3, pt;\n\t"
      "}" ::"r"(d), "r"(a), "l"(b), "r"(idesc) : "memory");
}
// 4 MMAs, A from smem (SW128 K-major descriptor, +32 B per k-step).
__device__ __forceinline__ void mma4_ss(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t acc0) {
  asm volatile(
      "{\n\t.reg .pred e, p, pt;\n\t.reg .b64 a1, a2, a3, b1, b2, b3;\n\t"
      "elect.sync _|e, 0xffffffff;\n\tsetp.ne.b32 p, %4, 0;\n\tsetp.eq.b32 pt, 0, 0;\n\t"
      "add.s64 a1, %1, 2;\n\tadd.s64 a2, %1, 4;\n\tadd.s64 a3, %1, 6;\n\t"
      "add.s64 b1, %2, 2;\n\tadd.s64 b2, %2, 4;\n\tadd.s64 b3, %2, 6;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], a1, b1, %3, pt;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], a2, b2, %3, pt;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::f16 [%0], a3, b3, %3, pt;\n\t}" ::"r"(d),
      "l"(a), "l"(b), "r"(idesc), "r"(acc0)
      : "memory");
}
// 4 x tcgen05.cp.128x256b: one 64-element K chunk of a 128-row SW128 tile -> TMEM columns t .. t+31.
__device__ __forceinline__ void cp4(uint32_t t, uint64_t s) {
  asm volatile(
      "{\n\t.reg .pred e;\n\t.reg .b32 t1, t2, t3;\n\t.reg .b64 s1, s2, s3;\n\t"
      "elect.sync _|e, 0xffffffff;\n\t"
      "add.u32 t1, %0, 8;\n\tadd.u32 t2, %0, 16;\n\tadd.u32 t3, %0, 24;\n\t"
      "add.s64 s1, %1, 2;\n\tadd.s64 s2, %1, 4;\n\tadd.s64 s3, %1, 6;\n\t"
      "@e tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t"
      "@e tcgen05.cp.cta_group::1.128x256b [t1], s1;\n\t"
      "@e tcgen05.cp.cta_group::1.128x256b [t2], s2;\n\t"
      "@e tcgen05.cp.cta_group::1.128x256b [t3], s3;\n\t}" ::"r"(t),
      "l"(s)
      : "memory");
}
__device__ __forceinline__ void commit_e(uint32_t bar) {
  asm volatile(
      "{\n\t.reg .pred e;\n\telect.sync _|e, 0xffffffff;\n\t"
      "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n\t}" ::"r"(bar)
      : "memory");
}
__device__ __forceinline__ bool try_wait(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
      : "=r"(ok)
      : "r"(bar), "r"(parity)
      : "memory");
  return ok != 0;
}
__device__ __forceinline__ void wait_bar(uint32_t bar, uint32_t parity) {
#ifdef K3SGT_DEBUG_WAIT
  long long n = 0;
  while (!try_wait(bar, parity)) {
    if (++n == (1ll << 22)) {
      printf("wait_bar timeout: block %d thread %d bar off %u parity %u\n", blockIdx.x, threadIdx.x, bar & 0xffff,
             parity);
    }
  }
#else
  while (!try_wait(bar, parity)) {
  }
#endif
}
__device__ __forceinline__ void expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
// ef (l2pf 2026-09-26): the staged in_proj is read once per layer; at evict_normal its 46 MB of dead lines slow
// the next MoE by ~2-3 us (RESULTS.md Task 6); ef = 1 loads it with an L2 evict_first cache hint instead.
__device__ __forceinline__ void tma3(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, int c2,
                                     int ef = 0) {
  if (ef) {
    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], "
        "[%1, {%3, %4, %5}], [%2], %6;" ::"r"(dst),
        "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2), "l"(pol)
        : "memory");
  } else {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], "
        "[%2];" ::"r"(dst),
        "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2)
        : "memory");
  }
}
__device__ __forceinline__ void bulk(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(bar)
               : "memory");
}
__device__ __forceinline__ uint4 ld_relaxed16(const void* p) {
  uint4 v;
  asm volatile("ld.relaxed.gpu.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ unsigned long long ld_relaxed64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ bool ok4(const uint4& v) {
  return v.x != 0x80000000u && v.y != 0x80000000u && v.z != 0x80000000u && v.w != 0x80000000u;
}
__device__ __forceinline__ void sts16(uint32_t a, const uint4& v) {
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ uint4 lds16(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
  return v;
}
__device__ __forceinline__ void mfma2(uint32_t w, uint32_t x, float& a0, float& a1) {
  asm("{ .reg .b16 wl, wh, xl, xh;\n\tmov.b32 {wl, wh}, %2;\n\tmov.b32 {xl, xh}, %3;\n\t"
      "fma.rn.f32.bf16 %0, wl, xl, %0;\n\tfma.rn.f32.bf16 %1, wh, xh, %1; }"
      : "+f"(a0), "+f"(a1)
      : "r"(w), "r"(x));
}
__device__ __forceinline__ uint32_t mapa(uint32_t a, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  return r;
}
__device__ __forceinline__ void st_async(uint32_t raddr, float v, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(raddr),
               "r"(__float_as_uint(v)), "r"(rbar)
               : "memory");
}
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %cluster_ctarank;" : "=r"(r));
  return r;
}

// Mailbox poll of this CTA's K slice for tokens m = m0, m0 + 2, ... (< M), then write into the SW128 B tile.
// Mailbox poll of this CTA's K slice for tokens m = m0, m0 + 2, ... (< mv), written into the SW128 B tile. With
// AR, lanes 16..31 of slot 3 (pieces 112..127) also fetch the half lamport chunk next to the K slice that the
// AttnRes statistics need (even rank: pieces 112..127, odd rank: pieces -16..-1) into xb[m][16].
template <int M, bool AR = false>
__device__ __forceinline__ void poll_x(const Params& p, int j, int m0, int lane, uint32_t bt, uint32_t xb = 0) {
  constexpr int MT = (M + 1) / 2;
  if (p.poll_mode == 1 && lane < 2 * MT && m0 + 2 * (lane >> 1) < p.mv) {
    // low-pressure phase: each lane watches the last 16 B of one 448-element (one source rank's) segment
    const uint32_t* f = p.mailbox + (size_t)(m0 + 2 * (lane >> 1)) * (K / 2) + j * (KS / 2) + (56 * (lane & 1) + 55) * 4;
    while (!ok4(ld_relaxed16(f))) {
      if (p.poll_ns) __nanosleep(p.poll_ns);
    }
  }
  __syncwarp();
  uint4 v[MT][4];
  bool good[MT][4];
  const int xoff = (j & 1) ? -128 : 0;  // extra pieces: kc in [112, 128) -> kc + xoff
#pragma unroll
  for (int t = 0; t < MT; ++t)
#pragma unroll
    for (int i = 0; i < 4; ++i) good[t][i] = !(m0 + 2 * t < p.mv && lane + 32 * i < (AR ? 128 : 112));
  for (;;) {
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
      for (int i = 0; i < 4; ++i)
        if (!good[t][i]) {
          const int kc = lane + 32 * i, kp = kc >= 112 ? kc + xoff : kc;
          v[t][i] = ld_relaxed16(p.mailbox + (size_t)(m0 + 2 * t) * (K / 2) + j * (KS / 2) + kp * 4);
        }
    bool all = true;
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
      for (int i = 0; i < 4; ++i)
        if (!good[t][i]) {
          good[t][i] = ok4(v[t][i]);
          if (good[t][i]) {
            const int kc = lane + 32 * i, m = m0 + 2 * t;
            if (kc < 112) sts16(bt + (kc >> 3) * 1024 + m * 128 + (((kc & 7) ^ m) << 4), v[t][i]);
            else sts16(xb + (m * 16 + kc - 112) * 16, v[t][i]);
          }
          all = all && good[t][i];
        }
    if (__all_sync(0xffffffffu, all)) break;
    if (p.poll_ns) __nanosleep(p.poll_ns);
  }
  tc::fence_proxy_async_smem();
  __syncwarp();
}

// CUDA-core share (fma.rn.f32.bf16 from smem, one warp per row, fixed-order shuffle reduction), pool of 12 warps:
//   jobs [0, 64): M64-tile row r, K chunks [ct, 14) (read in place from the SW128 tile) -> pc[r]      (M <= 2)
//   jobs [64, 64 + c_rows): CUDA row r, whole K slice -> pl[128 + s_rows + r]

// Push M fp32 values of local partial row r (0..215) from CTA rank j to its owner's inbox (st.async, counted by the
// owner's mbarrier). Inbox layout at the owner: [src rank][27 rows][M].
template <int M>
__device__ __forceinline__ void push_row(uint32_t IB, uint32_t inbox, int j, int r, const float* v) {
  const int d = r / OWN, rr = r - d * OWN;
  const uint32_t ra = mapa(IB + (uint32_t)((j * OWN + rr) * M) * 4, d), rb = mapa(inbox, d);
  if constexpr (M == 1) {
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(ra),
                 "r"(__float_as_uint(v[0])), "r"(rb) : "memory");
  } else if constexpr (M == 2) {
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];" ::"r"(ra),
                 "r"(__float_as_uint(v[0])), "r"(__float_as_uint(v[1])), "r"(rb) : "memory");
  } else {
#pragma unroll
    for (int h = 0; h < M / 4; ++h)
      asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(
                       ra + 16 * h),
                   "r"(__float_as_uint(v[4 * h])), "r"(__float_as_uint(v[4 * h + 1])),
                   "r"(__float_as_uint(v[4 * h + 2])), "r"(__float_as_uint(v[4 * h + 3])), "r"(rb)
                   : "memory");
  }
}

// CUDA-core rows (cfg 0): fma.rn.f32.bf16 from smem, 3 rows per warp job (latencies of the rows overlap),
// fixed-order shuffle reduction, lane 0 pushes the row. Pool = warps 8..15 (one round for <= 24 rows); warps 4..7
// stay free for the TMEM epilogue.
template <int M>
__device__ __forceinline__ void cuda_rows(const Params& p, int pi, int lane, int j, int c_rows, int s_rows,
                                          uint32_t C, uint32_t BT, uint32_t IB, uint32_t inbox, uint32_t cfull,
                                          float* dbg) {
  if constexpr (M <= 4) {
    if (c_rows == 0 || (p.dbgf & 8)) return;
    constexpr int RB = 3, NPOOL = 8;
    const int nj = (c_rows + RB - 1) / RB;
    if (pi >= nj) return;
    wait_bar(cfull, 0);
    uint4 xb[M][4];
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int kc = lane + 32 * i;
        xb[m][i] = kc < 112 ? lds16(BT + (kc >> 3) * 1024 + m * 128 + (((kc & 7) ^ m) << 4)) : make_uint4(0, 0, 0, 0);
      }
    for (int job = pi; job < nj; job += NPOOL) {
      const int r0 = job * RB;
      float a[RB][M][2];
#pragma unroll
      for (int rr = 0; rr < RB; ++rr)
#pragma unroll
        for (int m = 0; m < M; ++m) a[rr][m][0] = a[rr][m][1] = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int kc = lane + 32 * i;
        if (kc < 112) {
          uint4 wv[RB];
#pragma unroll
          for (int rr = 0; rr < RB; ++rr)
            wv[rr] = r0 + rr < c_rows ? lds16(C + (r0 + rr) * SLICEB + kc * 16) : make_uint4(0, 0, 0, 0);
#pragma unroll
          for (int rr = 0; rr < RB; ++rr)
#pragma unroll
            for (int m = 0; m < M; ++m) {
              mfma2(wv[rr].x, xb[m][i].x, a[rr][m][0], a[rr][m][1]);
              mfma2(wv[rr].y, xb[m][i].y, a[rr][m][0], a[rr][m][1]);
              mfma2(wv[rr].z, xb[m][i].z, a[rr][m][0], a[rr][m][1]);
              mfma2(wv[rr].w, xb[m][i].w, a[rr][m][0], a[rr][m][1]);
            }
        }
      }
      float sm[RB][M];
#pragma unroll
      for (int rr = 0; rr < RB; ++rr)
#pragma unroll
        for (int m = 0; m < M; ++m) sm[rr][m] = a[rr][m][0] + a[rr][m][1];
#pragma unroll
      for (int o = 16; o; o >>= 1)
#pragma unroll
        for (int rr = 0; rr < RB; ++rr)
#pragma unroll
          for (int m = 0; m < M; ++m) sm[rr][m] += __shfl_xor_sync(0xffffffffu, sm[rr][m], o);
      if (lane == 0) {
#pragma unroll
        for (int rr = 0; rr < RB; ++rr)
          if (r0 + rr < c_rows) {
            const int r = 128 + s_rows + r0 + rr;
            push_row<M>(IB, inbox, j, r, sm[rr]);
            if (dbg)
#pragma unroll
              for (int m = 0; m < M; ++m) dbg[r * 8 + m] = sm[rr][m];
          }
      }
    }
  }
}


// ---------------------------------------------------------------- AttnRes front (lamport_attn_res semantics)
__device__ __forceinline__ void unpack8(const uint4& u, float2 (&f)[4]) {
  f[0] = make_float2(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u));
  f[1] = make_float2(__uint_as_float(u.y << 16), __uint_as_float(u.y & 0xffff0000u));
  f[2] = make_float2(__uint_as_float(u.z << 16), __uint_as_float(u.z & 0xffff0000u));
  f[3] = make_float2(__uint_as_float(u.w << 16), __uint_as_float(u.w & 0xffff0000u));
}
__device__ __forceinline__ uint32_t pack2(float2 f) {
  __nv_bfloat162 b = __float22bfloat162_rn(f);
  return *reinterpret_cast<uint32_t*>(&b);
}
__device__ __forceinline__ uint4 pack8(const float2 (&f)[4]) {
  return make_uint4(pack2(f[0]), pack2(f[1]), pack2(f[2]), pack2(f[3]));
}
__device__ __forceinline__ float dot8(const float2 (&a)[4], const float2 (&b)[4]) {
  float2 s = __fmul2_rn(a[0], b[0]);
  float2 s1 = __fmul2_rn(a[1], b[1]);
  s = __ffma2_rn(a[2], b[2], s);
  s1 = __ffma2_rn(a[3], b[3], s1);
  s = __fadd2_rn(s, s1);
  return s.x + s.y;
}
__device__ __forceinline__ uint4 ldg16nc(const void* p) {
  uint4 u;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(u.x), "=r"(u.y), "=r"(u.z), "=r"(u.w) : "l"(p));
  return u;
}
__device__ __forceinline__ void stg16(void* p, const uint4& u) {
  asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(u.x), "r"(u.y), "r"(u.z), "r"(u.w) : "memory");
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float tree_max8(const float* x, int n) {
  float m[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) m[i] = i < n ? x[i] : -INFINITY;
#pragma unroll
  for (int w = 1; w < 8; w <<= 1)
#pragma unroll
    for (int i = 0; i + w < 8; i += 2 * w) m[i] = fmaxf(m[i], m[i + w]);
  return m[0];
}
__device__ __forceinline__ float tree_sum8(const float* x, int n) {
  float m[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) m[i] = i < n ? x[i] : 0.f;
#pragma unroll
  for (int w = 1; w < 8; w <<= 1)
#pragma unroll
    for (int i = 0; i + w < 8; i += 2 * w) m[i] += m[i + w];
  return m[0];
}
__device__ __forceinline__ void st_async_f(uint32_t local, uint32_t bar, int rank, float v) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(mapa(local, rank)),
               "r"(__float_as_uint(v)), "r"(mapa(bar, rank)) : "memory");
}

// AttnRes for this CTA's K slice, run by warps 8..15. Warp v handles token v / (8 / M) and 112 / (8 / M) of the
// slice's 16-byte pieces. Statistics are reduced warp -> CTA (fixed order) -> cluster (st.async pushes to all 8
// ranks, summed in rank order). The x pieces (bf16 AttnRes output) overwrite the mailbox delta in the SW128 B
// tile. Pre-landing: block logits, B = sum_s e_s v_s, |B|^2, old prefix. Post-landing: P = bf16(prefix + delta),
// <B,P>, |P|^2, <P,w>, then out = (ca B + cp P) * out_w as in lamport_attn_res (OUT_NORM = true).
// Reduction helpers copied from lamport_attn_res.cu (k3opt/csrc) so that the AttnRes statistics are summed in exactly
// the same order: dot8 per lane, butterfly reduce-scatter across the warp (RS), per-warp partials summed per
// lamport "rank" (4 warps x 256 elements = 1024 elements), then ranks summed as (even ranks) + (odd ranks) for the
// block statistics, and round-robin into 4 accumulators for the post-landing statistics.
template <int N, int O>
struct RS {
  static __device__ __forceinline__ void run(float* v, int lane) {
    if constexpr (O > 0) {
      if constexpr (N == 1) {
        v[0] += __shfl_xor_sync(0xffffffffu, v[0], O);
        RS<1, O / 2>::run(v, lane);
      } else {
        constexpr int H = N / 2;
        const bool upper = lane & O;
#pragma unroll
        for (int j = 0; j < H; ++j) {
          const float send = upper ? v[j] : v[j + H];
          const float keep = upper ? v[j + H] : v[j];
          v[j] = keep + __shfl_xor_sync(0xffffffffu, send, O);
        }
        RS<H, O / 2>::run(v, lane);
      }
    }
  }
};
template <int KP>
__device__ __forceinline__ int rs_index(int j, int lane) {
  int idx = j, n = KP;
#pragma unroll
  for (int o = 16; o >= 1; o >>= 1) {
    if (n > 1) {
      n >>= 1;
      if (lane & o) idx += n;
    }
  }
  return idx;
}
template <int KP>
__host__ __device__ constexpr int rs_dup() { return KP >= 32 ? 1 : 32 / KP; }
__device__ __forceinline__ float tree_max_l(const float* x, int n) {
  float m[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) m[i] = i < n ? x[i] : 0.f;
#pragma unroll
  for (int w = 1; w < 16; w <<= 1)
#pragma unroll
    for (int i = 0; i + w < 16; i += 2 * w)
      if (i + w < n) m[i] = fmaxf(m[i], m[i + w]);
  return m[0];
}
__device__ __forceinline__ float tree_sum_l(const float* x, int n) {
  float m[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) m[i] = i < n ? x[i] : 0.f;
#pragma unroll
  for (int w = 1; w < 16; w <<= 1)
#pragma unroll
    for (int i = 0; i + w < 16; i += 2 * w)
      if (i + w < n) m[i] += m[i + w];
  return m[0];
}
// RS of KP values; lanes holding a finished statistic (index < nst) return it via (*val, *idx).
template <int KP>
__device__ __forceinline__ bool rs_hold(float* a, int lane, int nst, float& val, int& idx) {
  RS<KP, 16>::run(a, lane);
  idx = rs_index<KP>(0, lane);
  val = a[0];
  return (lane & (rs_dup<KP>() - 1)) == 0 && idx < nst;
}

// AttnRes of lamport_attn_res (OUT_NORM = true), bit-exact. Statistics domain of CTA rank j (pair q = j / 2): the
// four lamport warp chunks (256 elements) covering its K slice: even j: chunks 7q .. 7q+3 = elements
// [1792 q, 1792 q + 1024); odd j: chunks 7q+3 .. 7q+6 = [1792 q + 768, 1792 q + 1792). The shared chunk 7q+3 is
// computed by both CTAs of the pair (identical values) and contributed by the even one. Every chunk's per-warp
// partials go to all 8 CTAs (st.async), which then sum them in lamport's order. Items = (token, chunk) pairs, one
// warp each (lane l = elements 8l .. 8l+7 of the chunk, as lamport's thread layout). The x pieces of the K slice are
// written into the SW128 B tile over the mailbox delta the pollers put there; the half chunk outside the K slice
// is polled here directly.
template <int M>
struct ARFront {
  static constexpr int NIT = 4 * M, IPW = (NIT + 7) / 8;
  static constexpr bool KEEP = IPW <= 2;  // keep the block rows in registers between the two passes (M <= 4)
  float2 B[IPW][4];
  uint4 Pp[IPW];  // old prefix (pre), then P = bf16(old + delta) (post)
  float q[IPW];
  unsigned long long target;

  __device__ __forceinline__ void geom(int j, int it, int lane, int& m, int& g, int& dofs, bool& contrib,
                                       int& pc) const {
    m = it >> 2;
    const int cl = it & 3, pr = j >> 1, odd = j & 1;
    g = 7 * pr + (odd ? 3 : 0) + cl;
    dofs = 256 * cl + 8 * lane;                               // offset in the CTA's statistics domain
    const int e0 = 1792 * pr + (odd ? 768 : 0) + dofs;        // hidden index of the lane's first element
    contrib = !(odd && cl == 0);
    const int k = e0 - KS * j;
    pc = (k >= 0 && k < KS) ? (k >> 3) : -1;                  // 16-B piece in the K slice (x / P outputs)
  }

  __device__ __forceinline__ void pre(const Params& p, int j, int v, int lane, unsigned char* smem, uint32_t sb,
                                      uint32_t pre_bar, uint32_t ar_ready, unsigned long long* tl) {
    float* w_s = reinterpret_cast<float*>(smem + p.ar_off);
    uint4* ow_s = reinterpret_cast<uint4*>(smem + p.ar_off + 4096);
    float* tpre = reinterpret_cast<float*>(smem + p.ar_off + 6144);
    const float* pre_in = reinterpret_cast<const float*>(smem + p.ar_off + 6912);
    const int nb = p.nb, npre = 2 * nb;
    const int kpre = npre <= 1 ? 1 : npre <= 2 ? 2 : npre <= 4 ? 4 : npre <= 8 ? 8 : 16;
    const int dstart = 1792 * (j >> 1) + ((j & 1) ? 768 : 0);
    uint4 vb[KEEP ? IPW : 1][8];
    unsigned int chk = 0;
    // ---- pass 1: weights, block rows, old prefix; block norms / dots -> RS -> push the chunk partials
#pragma unroll
    for (int ii = 0; ii < IPW; ++ii) {
      const int it = v + 8 * ii;
      Pp[ii] = make_uint4(0, 0, 0, 0);
      q[ii] = 0.f;
      if (it >= NIT) continue;
      int m, g, dofs, pc;
      bool contrib;
      geom(j, it, lane, m, g, dofs, contrib, pc);
      if (m >= p.mv) continue;
      const int e0 = dstart + dofs;
      float2 w[4];
      {
        float2 a0[4], b0[4];
        unpack8(ldg16nc(p.norm_w + e0), a0);
        unpack8(ldg16nc(p.qk_w + e0), b0);
#pragma unroll
        for (int c = 0; c < 4; ++c) w[c] = __fmul2_rn(a0[c], b0[c]);
        if (m == 0) {
          reinterpret_cast<float4*>(w_s)[dofs / 4] = make_float4(w[0].x, w[0].y, w[1].x, w[1].y);
          reinterpret_cast<float4*>(w_s)[dofs / 4 + 1] = make_float4(w[2].x, w[2].y, w[3].x, w[3].y);
          ow_s[dofs / 8] = ldg16nc(p.out_w + e0);
        }
      }
      const __nv_bfloat16* brow = p.blocks + (long long)m * p.blocks_sm + e0;
      uint4 vv[8];
#pragma unroll
      for (int s = 0; s < 8; ++s) vv[s] = s < nb ? ldg16nc(brow + s * p.blocks_sr) : make_uint4(0, 0, 0, 0);
      Pp[ii] = ld_relaxed16(p.prefix + (long long)m * p.prefix_sm + e0);  // coherent: this kernel writes prefix
      if constexpr (KEEP) {
#pragma unroll
        for (int s = 0; s < 8; ++s) vb[ii][s] = vv[s];
      }
#pragma unroll
      for (int s = 0; s < 8; ++s) chk |= vv[s].x ^ vv[s].w;
      chk |= (Pp[ii].x ^ Pp[ii].w) | __float_as_uint(w[0].x);
      if (nb > 0) {
        float a[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) a[k] = 0.f;
#pragma unroll
        for (int s = 0; s < 8; ++s)
          if (s < nb) {
            float2 x[4];
            unpack8(vv[s], x);
            a[s] = dot8(x, x);
            a[nb + s] = dot8(x, w);
          }
        float val = 0.f;
        int idx = 0;
        bool hold;
        if (kpre == 2) hold = rs_hold<2>(a, lane, npre, val, idx);
        else if (kpre == 4) hold = rs_hold<4>(a, lane, npre, val, idx);
        else if (kpre == 8) hold = rs_hold<8>(a, lane, npre, val, idx);
        else hold = rs_hold<16>(a, lane, npre, val, idx);
        if (hold && contrib) {
          const uint32_t dst = sb + p.ar_off + 6912 + (uint32_t)((m * 28 + g) * 16 + idx) * 4;
#pragma unroll
          for (int r = 0; r < CL; ++r) st_async_f(dst, pre_bar, r, val);
        }
      }
    }
    // Wait until our global loads have returned: the st.shared below consumes a value built from every loaded
    // word (old prefix, block rows), so it cannot issue before all of them completed; the mbarrier arrive
    // (release.cta) and the named barrier order it before the TMA warp's staging and before t == 0 counts this CTA
    // as a reader of the old prefix. That count is a relaxed atomic on purpose: a release fence here would wait for
    // the in_proj TMA loads this SM may already have in flight (probe rule). Sufficient because a load that has
    // returned its value cannot observe a later store: cluster 0 stores prefix' only after its relaxed poll sees
    // all 120 counts (control dependency, stores are not speculated).
    asm volatile("st.shared.u32 [%0], %1;" ::"r"(sb + p.ar_off + 6800), "r"(chk) : "memory");
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(ar_ready) : "memory");
    tc::named_bar(ARB, 256);
    if (v * 32 + lane == 0) {
      const unsigned long long c0 = atomicAdd(p.pref_cnt, 1ull);
      reinterpret_cast<unsigned long long*>(smem + p.ar_off + 6784)[0] = (c0 / gridDim.x + 1) * gridDim.x;
    }
    if (tl && v == 0 && lane == 0) tl[16] = gtimer();
    const int t = v * 32 + lane;
    if (nb > 0) {
      wait_bar(pre_bar, 0);
      if (t < p.mv * npre) {  // totals in lamport's order: per rank 4 warps in order, then even + odd ranks
        const int m = t / npre, idx = t - m * npre;
        float s0 = 0.f, s1 = 0.f;
#pragma unroll
        for (int r = 0; r < 7; ++r) {
          float s = 0.f;
#pragma unroll
          for (int k = 0; k < 4; ++k) s += pre_in[(m * 28 + 4 * r + k) * 16 + idx];
          if (r & 1) s1 += s;
          else s0 += s;
        }
        tpre[m * 16 + idx] = s0 + s1;
      }
    }
    tc::named_bar(ARB, 256);
    float* esm = reinterpret_cast<float*>(smem + p.ar_off + 6912 + M * 28 * 20 * 4 + M * 256);  // [M][10]
    if (nb > 0 && t < p.mv) {  // block logits and softmax weights of token t (lamport formulas), once per token
      float lg[8], e[8];
#pragma unroll
      for (int s = 0; s < 8; ++s)
        lg[s] = s < nb ? tpre[t * 16 + nb + s] * rsqrtf(fmaf(tpre[t * 16 + s], kInvH, p.eps)) : 0.f;
      const float m0 = tree_max_l(lg, nb);
#pragma unroll
      for (int s = 0; s < 8; ++s) e[s] = s < nb ? __expf(lg[s] - m0) : 0.f;
      const float Eb = tree_sum_l(e, nb);
#pragma unroll
      for (int s = 0; s < 8; ++s) esm[t * 10 + s] = e[s];
      esm[t * 10 + 8] = m0;
      esm[t * 10 + 9] = Eb;
    }
    tc::named_bar(ARB, 256);
    if (tl && v == 0 && lane == 0) tl[18] = gtimer();
    // ---- pass 2: B = sum_s e_s v_s, |B|^2 lane partial
#pragma unroll
    for (int ii = 0; ii < IPW; ++ii) {
      const int it = v + 8 * ii;
#pragma unroll
      for (int c = 0; c < 4; ++c) B[ii][c] = make_float2(0.f, 0.f);
      if (it >= NIT) continue;
      int m, g, dofs, pc;
      bool contrib;
      geom(j, it, lane, m, g, dofs, contrib, pc);
      if (m >= p.mv) continue;
      if (nb == 0) continue;
      float e[8];
#pragma unroll
      for (int s = 0; s < 8; ++s) e[s] = esm[m * 10 + s];
      const __nv_bfloat16* brow = p.blocks + (long long)m * p.blocks_sm + dstart + dofs;
      float2 x[4];
      if constexpr (KEEP) unpack8(vb[ii][0], x);
      else unpack8(ldg16nc(brow), x);
#pragma unroll
      for (int c = 0; c < 4; ++c) B[ii][c] = __fmul2_rn(make_float2(e[0], e[0]), x[c]);
#pragma unroll
      for (int s = 1; s < 8; ++s)
        if (s < nb) {
          if constexpr (KEEP) unpack8(vb[ii][s], x);
          else unpack8(ldg16nc(brow + s * p.blocks_sr), x);
#pragma unroll
          for (int c = 0; c < 4; ++c) B[ii][c] = __ffma2_rn(make_float2(e[s], e[s]), x[c], B[ii][c]);
        }
      q[ii] = dot8(B[ii], B[ii]);
    }
    if (tl && v == 0 && lane == 0) tl[19] = gtimer();
  }

  // after the pollers have written the mailbox delta of the K slice into the B tile
  __device__ __forceinline__ void post(const Params& p, int j, int v, int lane, unsigned char* smem, uint32_t sb,
                                       uint32_t BT, uint32_t post_bar, bool writer, unsigned long long* tl) {
    const float* w_s = reinterpret_cast<const float*>(smem + p.ar_off);
    const uint4* ow_s = reinterpret_cast<const uint4*>(smem + p.ar_off + 4096);
    float* tpost = reinterpret_cast<float*>(smem + p.ar_off + 6656);
    const float* tpre = reinterpret_cast<const float*>(smem + p.ar_off + 6144);
    const float* post_in = reinterpret_cast<const float*>(smem + p.ar_off + 6912 + M * 28 * 16 * 4);
    const int nb = p.nb, npost = nb > 0 ? 4 : 1;
    const int dstart = 1792 * (j >> 1) + ((j & 1) ? 768 : 0);
    const int t = v * 32 + lane;
    if (tl && v == 0 && lane == 0) tl[23] = gtimer_after(__uint_as_float(lds16(BT).x));  // after DBAR
#pragma unroll
    for (int ii = 0; ii < IPW; ++ii) {
      const int it = v + 8 * ii;
      if (it >= NIT) continue;
      int m, g, dofs, pc;
      bool contrib;
      geom(j, it, lane, m, g, dofs, contrib, pc);
      if (m >= p.mv) continue;
      uint4 du;
      if (pc >= 0) {
        du = lds16(BT + (pc >> 3) * 1024 + m * 128 + (((pc & 7) ^ m) << 4));
      } else {  // half chunk outside the K slice (fetched by the pollers)
        const int xi = (j & 1) ? lane : lane - 16;
        du = lds16(sb + p.ar_off + 6912 + M * 28 * 20 * 4 + (m * 16 + xi) * 16);
      }
      float2 d[4], o[4], P[4], w[4];
      unpack8(du, d);
      unpack8(Pp[ii], o);
#pragma unroll
      for (int c = 0; c < 4; ++c) P[c] = __bfloat1622float2(__float22bfloat162_rn(__fadd2_rn(o[c], d[c])));
      Pp[ii] = pack8(P);
      const float4 w0 = reinterpret_cast<const float4*>(w_s)[dofs / 4], w1 = reinterpret_cast<const float4*>(w_s)[dofs / 4 + 1];
      w[0] = make_float2(w0.x, w0.y); w[1] = make_float2(w0.z, w0.w);
      w[2] = make_float2(w1.x, w1.y); w[3] = make_float2(w1.z, w1.w);
      float b[4];
      float val = 0.f;
      int idx = 0;
      bool hold;
      if (nb > 0) {
        b[0] = dot8(B[ii], P);
        b[1] = q[ii];
        b[2] = dot8(P, P);
        b[3] = dot8(P, w);
        hold = rs_hold<4>(b, lane, 4, val, idx);
      } else {
        b[0] = dot8(P, P);
        hold = rs_hold<1>(b, lane, 1, val, idx);
      }
      if (hold && contrib) {
        const uint32_t dst = sb + p.ar_off + 6912 + M * 28 * 16 * 4 + (uint32_t)((m * 28 + g) * 4 + idx) * 4;
#pragma unroll
        for (int r = 0; r < CL; ++r) st_async_f(dst, post_bar, r, val);
      }
    }
    if (tl && v == 0 && lane == 0) tl[20] = gtimer();  // statistics pushed (after the post-loop, no barrier)
    wait_bar(post_bar, 0);
    if (tl && v == 0 && lane == 0) tl[21] = gtimer_after(post_in[0]);
    // warp v < mv: totals of token v in lamport's order (lane 4 * idx + a accumulates chunks c = a (mod 4) in
    // order; then (s0 + s1) + (s2 + s3) by shuffles), then lamport's output scale (ca, cp) on lane 0
    float* csm = reinterpret_cast<float*>(smem + p.ar_off + 6656);  // [M][2] (ca, cp)
    const float* esm = reinterpret_cast<const float*>(smem + p.ar_off + 6912 + M * 28 * 20 * 4 + M * 256);
    if (v < p.mv) {
      const int idx = lane >> 2, a = lane & 3;
      float acc = 0.f;
      if (idx < npost)
#pragma unroll
        for (int c = 0; c < 7; ++c) acc += post_in[(v * 28 + 4 * c + a) * 4 + idx];
      const float o1 = __shfl_xor_sync(0xffffffffu, acc, 1);       // pairs (s0, s1) and (s2, s3)
      const float pr = (a & 1) ? o1 + acc : acc + o1;               // s0 + s1 on lanes a = 0, 1; s2 + s3 on 2, 3
      const float o2 = __shfl_xor_sync(0xffffffffu, pr, 2);
      const float tot = (a & 2) ? o2 + pr : pr + o2;                // (s0 + s1) + (s2 + s3)
      const float X = __shfl_sync(0xffffffffu, tot, 0), Q = __shfl_sync(0xffffffffu, tot, 4);
      const float SP = __shfl_sync(0xffffffffu, tot, nb > 0 ? 8 : 0), DP = __shfl_sync(0xffffffffu, tot, 12);
      if (lane == 0) {
        float cav = 0.f, cpv;
        if (nb > 0) {
          const float m0 = esm[v * 10 + 8], Eb = esm[v * 10 + 9];
          const float lp = DP * rsqrtf(fmaf(SP, kInvH, p.eps));
          const float mx = fmaxf(m0, lp);
          const float alpha = __expf(m0 - mx), ep = __expf(lp - mx);
          const float den = fmaf(alpha, Eb, ep);
          const float quad = fmaf(alpha * alpha, Q, fmaf(2.f * alpha * ep, X, ep * ep * SP));
          const float scale = rsqrtf(fmaf(quad, kInvH, p.out_eps * den * den));
          cav = alpha * scale;
          cpv = ep * scale;
        } else {
          cpv = rsqrtf(fmaf(SP, kInvH, p.out_eps));
        }
        csm[v * 2] = cav;
        csm[v * 2 + 1] = cpv;
      }
    }
    tc::named_bar(ARB, 256);
    uint4 xo[IPW];
#pragma unroll
    for (int ii = 0; ii < IPW; ++ii) {
      const int it = v + 8 * ii;
      xo[ii] = make_uint4(0, 0, 0, 0);
      if (it >= NIT) continue;
      int m, g, dofs, pc;
      bool contrib;
      geom(j, it, lane, m, g, dofs, contrib, pc);
      if (m >= p.mv || pc < 0) continue;
      const float ca = csm[m * 2], cp = csm[m * 2 + 1];
      float2 P[4], ow[4], o[4];
      unpack8(Pp[ii], P);
      unpack8(ow_s[dofs / 8], ow);
      if (nb > 0) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const float2 acc = __ffma2_rn(make_float2(ca, ca), B[ii][c], __fmul2_rn(make_float2(cp, cp), P[c]));
          o[c] = __fmul2_rn(acc, ow[c]);
        }
      } else {
#pragma unroll
        for (int c = 0; c < 4; ++c) o[c] = __fmul2_rn(__fmul2_rn(make_float2(cp, cp), P[c]), ow[c]);
      }
      xo[ii] = pack8(o);
      sts16(BT + (pc >> 3) * 1024 + m * 128 + (((pc & 7) ^ m) << 4), xo[ii]);
    }
    tc::fence_proxy_async_smem();
    if (tl && v == 0 && lane == 0) tl[22] = gtimer_after(__uint_as_float(xo[0].x));
    tc::named_bar(XBAR, 416);  // x ready: release the MMA issuers (warps 3..7) and the CUDA-row warps
    // mailbox reader count (all of this CTA's mailbox reads, incl. the half chunk above, are done); the last of
    // the 120 readers re-arms the mailbox
    if (v == 0) {
      int last = 0;
      if (lane == 0) last = atomicAdd(p.done, 1) == (int)gridDim.x - 1;
      last = __shfl_sync(0xffffffffu, last, 0);
      if (last) {
        for (int i = lane; i < p.mv * K / 8; i += 32)
          reinterpret_cast<uint4*>(const_cast<uint32_t*>(p.mailbox))[i] =
              make_uint4(0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u);
        if (lane == 0) {
          *p.done = 0;
          if (p.gate) *p.gate = 0;
        }
      }
    }
    if (writer) {
      // global side effects of AttnRes (cluster 0 only), after every CTA of the grid has read the old prefix
      target = reinterpret_cast<const unsigned long long*>(smem + p.ar_off + 6784)[0];
      if (t == 0)
        while (ld_relaxed64(p.pref_cnt) < target) {
        }
      tc::named_bar(ARB, 256);
#pragma unroll
      for (int ii = 0; ii < IPW; ++ii) {
        const int it = v + 8 * ii;
        if (it >= NIT) continue;
        int m, g, dofs, pc;
        bool contrib;
        geom(j, it, lane, m, g, dofs, contrib, pc);
        if (m >= p.mv || pc < 0) continue;
        const long long off = KS * j + pc * 8;
        stg16(p.prefix + (long long)m * p.prefix_sm + off, Pp[ii]);
        if (p.widx >= 0) stg16(p.blocks + (long long)m * p.blocks_sm + p.widx * p.blocks_sr + off, Pp[ii]);
        if (p.out) stg16(p.out + (long long)m * p.out_sm + off, xo[ii]);
      }
    }
  }
};

// __maxnreg__(120): 512 x 120 = 61440 of the SM's 65536 registers, so one small side kernel CTA (e.g. k3pf's 32-thread
// prefetch CTAs) can still share an SM with this CTA instead of blocking a whole 8-CTA cluster's placement.
#ifndef K3SGT_MAXNREG
#define K3SGT_MAXNREG 120
#endif
template <int M, bool AR>
__global__ void __cluster_dims__(CL, 1, 1) __maxnreg__(K3SGT_MAXNREG)
    stage_kernel(const __grid_constant__ CUtensorMap map_t, const __grid_constant__ CUtensorMap map_s,
                 const Params p) {
  extern __shared__ __align__(1024) unsigned char smem[];
  const unsigned long long t_start = gtimer();
  const long long k_start = clock64();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int j = (int)cluster_rank(), c = blockIdx.x / CL;
  const uint32_t sb = tc::su32(smem);
  const uint32_t R = sb, C = sb + p.c_off, BT = sb + p.b_off, IB = sb + p.i_off;
  const uint32_t bars = sb + p.bar_off;
  const uint32_t full0 = bars, empty0 = bars + 8 * RING, tmem_ready = bars + 16 * RING, sfull0 = tmem_ready + 8,
                 cfull = sfull0 + 16, acc_done = cfull + 8, inbox = acc_done + 8;
  float* ib = reinterpret_cast<float*>(smem + p.i_off);
  uint32_t* tptr = reinterpret_cast<uint32_t*>(smem + p.bar_off + 16 * RING + 48);
  unsigned long long* cdone = reinterpret_cast<unsigned long long*>(smem + p.bar_off + 128);  // [12]
  const int t_row0 = p.t_row0[c], s_row0 = p.s_row0[c], s_rows = p.s_rows[c], c_row0 = p.c_row0[c],
            c_rows = p.c_rows[c];
  const int nrows = 128 + s_rows + c_rows;
  const int nv = min(OWN, max(0, nrows - j * OWN));  // rows this CTA owns
  unsigned long long* tl = p.tl ? p.tl + blockIdx.x * 32 : nullptr;
  float* dbg = p.dbg ? p.dbg + (size_t)blockIdx.x * PROWS * 8 : nullptr;

  if (threadIdx.x == 0) {
    if ((sb & 1023) != 0) asm volatile("trap;");
    // full[3], empty[3], tmem_ready, sfull[2], cfull (count 1); acc_done (5 commits); inbox (count 1)
    for (int i = 0; i < 2 * RING + 4; ++i) tc::mbar_init(reinterpret_cast<uint64_t*>(smem + p.bar_off + 8 * i), 1);
    tc::mbar_init(reinterpret_cast<uint64_t*>(smem + p.bar_off + 16 * RING + 32), 5);
    tc::mbar_init(reinterpret_cast<uint64_t*>(smem + p.bar_off + 16 * RING + 40), 1);
    if constexpr (AR) {  // AttnRes statistic exchanges: pre (block norms / dots), post (4 scalars per token)
      tc::mbar_init(reinterpret_cast<uint64_t*>(smem + p.bar_off + 16 * RING + 64), 1);
      tc::mbar_init(reinterpret_cast<uint64_t*>(smem + p.bar_off + 16 * RING + 72), 1);
      tc::mbar_init(reinterpret_cast<uint64_t*>(smem + p.bar_off + 16 * RING + 80), 256);  // AttnRes loads returned
    }
    tc::fence_mbar_init();
    if (!p.detect) expect_tx(inbox, (uint32_t)(CL * nv * M * 4));  // 8 pushes per owned row, incl. our own
    if constexpr (AR) {  // 28 lamport chunks x tokens x statistics, each pushed to every CTA of the cluster
      if (p.nb > 0) expect_tx(bars + 16 * RING + 64, (uint32_t)(28 * p.mv * 2 * p.nb * 4));
      expect_tx(bars + 16 * RING + 72, (uint32_t)(28 * p.mv * (p.nb > 0 ? 4 : 1) * 4));
    }
  }
  if (warp == 1) tc::tmem_alloc(tptr, 512);
  if (threadIdx.x == 0) *reinterpret_cast<volatile int*>(smem + p.bar_off + 16 * RING + 56) = 0;
  for (int i = threadIdx.x; i < 1024 * KCH / 16; i += NT) sts16(BT + i * 16, make_uint4(0, 0, 0, 0));
  tc::fence_proxy_async_smem();
  tc::tc_fence_before();
  __syncthreads();
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n\tbarrier.cluster.wait.aligned;" ::: "memory");
  tc::tc_fence_after();
  const uint32_t tb = *reinterpret_cast<volatile uint32_t*>(tptr);

  if (warp == 2 || warp == 3) {
    // ---------------- mailbox pollers (warp 2: tokens 0, 2, ..; warp 3: tokens 1, 3, ..)
    if (p.gate) {
      if (lane == 0) {
        unsigned long long g;
        while ((g = ld_relaxed64(p.gate)) == 0) {
        }
        while (gtimer() < g) {
        }
      }
      __syncwarp();
    }
    poll_x<M, AR>(p, j, warp - 2, lane, BT, sb + p.ar_off + 6912 + M * 28 * 20 * 4);
    if (AR) {
      tc::named_bar(4, 64);  // both pollers done (warp 3 counts the CTA as a reader only after that)
      if (warp == 2 && tl && lane == 0) tl[3] = gtimer();
      asm volatile("bar.arrive %0, %1;" ::"n"(DBAR), "n"(320) : "memory");  // delta in the B tile
    } else if (warp == 2) {
      if (lane == 0) *reinterpret_cast<volatile int*>(smem + p.bar_off + 16 * RING + 56) = 1;
      if (tl && lane == 0) tl[3] = gtimer();
      if (!p.detect) asm volatile("bar.arrive %0, %1;" ::"n"(XBAR), "n"(448) : "memory");
      else asm volatile("bar.arrive 4, 64;" ::: "memory");
    } else if (!p.detect) {
      tc::named_bar(XBAR, 448);
    } else {
      tc::named_bar(4, 64);
    }
    // warp 3 (both pollers done) counts this CTA as a reader; the last reader re-arms the mailbox right away
    // (relaxed: the reads have returned their values, nothing to order; no fence while TMA loads may be in flight)
    if (warp == 3 && !AR) {
      int last = 0;
      if (lane == 0) last = atomicAdd(p.done, 1) == (int)gridDim.x - 1;
      last = __shfl_sync(0xffffffffu, last, 0);
      if (last) {
        for (int i = lane; i < p.mv * K / 8; i += 32)
          reinterpret_cast<uint4*>(const_cast<uint32_t*>(p.mailbox))[i] =
              make_uint4(0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u);
        if (lane == 0) {
          *p.done = 0;
          if (p.gate) *p.gate = 0;
        }
      }
    }
    if (warp == 3 && !p.detect) {
      if (AR) tc::named_bar(XBAR, 416);  // x (AttnRes output) ready
      // smem-tile MMAs, issued after the TMEM-tile MMAs (in-order tensor pipe) unless dbgf bit 4
      if (!(p.dbgf & 16)) tc::named_bar(SBAR, 160);
      if (p.sm_m && !(p.dbgf & 2)) {
        const uint32_t idesc = idesc_bf16(p.sm_m, 8);
        const uint32_t cstride = p.sm_box * 128;
        for (int h = 0; h < 2; ++h) {
          wait_bar(sfull0 + 8 * h, 0);
          tc::tc_fence_after();
          for (int k = 7 * h; k < 7 * h + 7; ++k)
            mma4_ss(tb + 32, tc::desc_sw128(R + k * cstride), tc::desc_sw128(BT + k * 1024), idesc, k != 0);
        }
      }
      if (tl && lane == 0) tl[11] = gtimer();
      commit_e(acc_done);
    }
  } else if (p.detect) {
    // detection-latency probe: pollers only
  } else if (warp == 0) {
    // ---------------- TMA producer (never polls)
    if (lane == 0) {
      if constexpr (AR) wait_bar(bars + 16 * RING + 80, 0);  // AttnRes operands first: its loads are latency-bound
      if (c_rows > 0) {
        expect_tx(cfull, c_rows * SLICEB);
        for (int r = 0; r < c_rows; ++r)
          bulk(C + r * SLICEB, p.w + (size_t)(c_row0 + r) * K + j * KS, SLICEB, cfull);
      }
      for (int q = 0; q < KCH / 2; ++q) {
        const int s = q % RING;
        if (q >= RING) wait_bar(empty0 + 8 * s, ((q / RING) - 1) & 1);
        expect_tx(full0 + 8 * s, SLOTB);
        tma3(R + s * SLOTB, &map_t, full0 + 8 * s, 0, t_row0, j * KCH + 2 * q, p.ef);
      }
      if (p.sm_m) {
        wait_bar(tmem_ready, 0);  // ring free
        if (tl) tl[1] = gtimer();
        const uint32_t hb = 7 * p.sm_box * 128;
        for (int h = 0; h < 2; ++h) {
          expect_tx(sfull0 + 8 * h, hb);
          tma3(R + h * hb, &map_s, sfull0 + 8 * h, 0, s_row0, j * KCH + 7 * h, p.ef);
        }
        if (tl || p.pf_n) {  // staging complete (this warp never polls, so waiting here is free)
          wait_bar(sfull0 + 8, 0);
          if (c_rows > 0) wait_bar(cfull, 0);
          if (tl) tl[2] = gtimer();
        }
        // in-kernel L2 prefetch of the layer's later weights (f_b_proj, o_proj): 30 issuing CTAs (>= 48
        // concurrent bulk-prefetch issuers make the L2 drop prefetches), 16 KB requests, round robin
        if (p.pf_n && (blockIdx.x & 3) == 0) {
          // each issuer takes every nis-th 16 KB chunk of every range (~23 requests per CTA for o_proj + f_b_proj)
          const int nis = gridDim.x / 4, me = blockIdx.x / 4;
          for (int r = 0; r < p.pf_n; ++r) {
            const char* base = reinterpret_cast<const char*>(p.pf_rng[2 * r]);
            const long long bytes = p.pf_rng[2 * r + 1];
            for (long long b = 16384ll * me; b < bytes; b += 16384ll * nis) {
              const uint32_t n = (uint32_t)(bytes - b < 16384 ? bytes - b : 16384);
              asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(base + b), "r"(n) : "memory");
            }
          }
        }
      }
    }
  } else if (warp == 1) {
    // ---------------- smem ring -> TMEM (tcgen05.cp)
    for (int q = 0; q < KCH / 2; ++q) {
      const int s = q % RING;
      wait_bar(full0 + 8 * s, (q / RING) & 1);
      tc::tc_fence_after();
      const uint64_t d0 = tc::desc_sw128(R + s * SLOTB);
      cp4(tb + TA + 64 * q, d0);
      cp4(tb + TA + 64 * q + 32, d0 + (16384 >> 4));
      commit_e(empty0 + 8 * s);
    }
    commit_e(tmem_ready);
  } else if (warp < 8) {
    // ---------------- TMEM-tile MMAs (chunks {0-3, 4-7, 8-10, 11-13}), CUDA rows, then the TMEM epilogue
    const int i = warp - 4;
    const int k0 = i < 2 ? 4 * i : 8 + 3 * (i - 2), k1 = i < 2 ? k0 + 4 : k0 + 3;
    wait_bar(tmem_ready, 0);
    tc::named_bar(XBAR, AR ? 416 : 448);
    if (!AR && tl && i == 0 && lane == 0) tl[15] = *reinterpret_cast<volatile int*>(smem + p.bar_off + 16 * RING + 56);
    tc::tc_fence_after();
    const uint32_t idesc = idesc_bf16(128, 8);
    if (!(p.dbgf & 1)) {
      if (k1 - k0 == 4) mma_ts_chunks4(tb + 8 * i, tb + TA + 32 * k0, tc::desc_sw128(BT + k0 * 1024), idesc);
      else mma_ts_chunks3(tb + 8 * i, tb + TA + 32 * k0, tc::desc_sw128(BT + k0 * 1024), idesc);
    }
    if (!(p.dbgf & 16)) asm volatile("bar.arrive %0, %1;" ::"n"(SBAR), "n"(160) : "memory");
    if (tl && i == 0 && lane == 0) tl[9] = gtimer();
    commit_e(acc_done);
    // epilogue: warp w reads TMEM lane quadrant w % 4 (rows 32q .. 32q + 31)
    wait_bar(acc_done, 0);
    if (tl && i == 0 && lane == 0) tl[4] = gtimer();
    tc::tc_fence_after();
    const int q = warp & 3;
    const uint32_t ta = tb + ((32 * q) << 16);
    float d[5][8];
    tc::tmem_ld8(ta + 0, d[0]);
    tc::tmem_ld8(ta + 8, d[1]);
    tc::tmem_ld8(ta + 16, d[2]);
    tc::tmem_ld8(ta + 24, d[3]);
    if (p.sm_m) tc::tmem_ld8(ta + 32, d[4]);
    const int r = 32 * q + lane;
    float v[M];
#pragma unroll
    for (int m = 0; m < M; ++m) v[m] = ((d[0][m] + d[1][m]) + d[2][m]) + d[3][m];
    push_row<M>(IB, inbox, j, r, v);
    if (dbg)
#pragma unroll
      for (int m = 0; m < M; ++m) dbg[r * 8 + m] = v[m];
    if (p.sm_m) {
      // M = 128: lane = row. M = 64: rows 16q .. 16q + 15 sit in lanes 32q .. 32q + 15.
      const int rs = p.sm_m == 128 ? r : (lane < 16 ? 16 * q + lane : -1);
      if (rs >= 0 && rs < s_rows) {
        push_row<M>(IB, inbox, j, 128 + rs, d[4]);
        if (dbg)
#pragma unroll
          for (int m = 0; m < M; ++m) dbg[(128 + rs) * 8 + m] = d[4][m];
      }
    }
    if (tl && i == 0 && lane == 0) tl[5] = gtimer();
  } else {
    // ---------------- AttnRes front (AR), then CUDA-core rows (pool slots 0..7)
    if constexpr (AR) {
      ARFront<M> f;
      f.pre(p, j, warp - 8, lane, smem, sb, bars + 16 * RING + 64, bars + 16 * RING + 80, tl);
      if (tl && warp == 8 && lane == 0) tl[15] = gtimer();
      tc::named_bar(DBAR, 320);
      if (tl && warp == 8 && lane == 0) tl[12] = gtimer();
      f.post(p, j, warp - 8, lane, smem, sb, BT, bars + 16 * RING + 72, c == 0, tl);
      if (tl && warp == 8 && lane == 0) tl[10] = gtimer();
    } else {
      tc::named_bar(XBAR, 448);
    }
    cuda_rows<M>(p, warp - 8, lane, j, c_rows, s_rows, C, BT, IB, inbox, cfull, dbg);
    if (lane == 0) cdone[warp - 8] = gtimer();
  }

  // ---------------- owner: wait for the 8 slices of its <= 27 rows, sum in rank order, one bf16 rounding
  if (warp >= 4 && !p.detect) {
    wait_bar(inbox, 0);
    if (tl && threadIdx.x == 128) tl[13] = gtimer();
    for (int e = threadIdx.x - 128; e < nv * M; e += 384) {
      const int rr = e / M, m = e - rr * M, r = j * OWN + rr;
      float s = 0.f;
#pragma unroll
      for (int src = 0; src < CL; ++src) s += ib[(src * OWN + rr) * M + m];
      const int g = r < 128 ? t_row0 + r : (r < 128 + s_rows ? s_row0 + (r - 128) : c_row0 + (r - 128 - s_rows));
      if (g < p.N && m < p.mv) p.y[(size_t)m * p.N + g] = __float2bfloat16_rn(s);
    }
    if (tl && threadIdx.x == 128) tl[6] = gtimer();
  }

  // ---------------- teardown
  tc::tc_fence_before();
  __syncthreads();
  if (warp == 1) {
    tc::tc_fence_after();
    tc::tmem_dealloc(tb, 512);
  }
  if (tl && threadIdx.x == 0) {
    unsigned long long mx = 0;
    for (int k = 0; k < 8; ++k) mx = cdone[k] > mx ? cdone[k] : mx;
    tl[8] = mx;
    tl[0] = t_start;
    tl[7] = gtimer();
    tl[14] = clock64() - k_start;  // SM clocks over [start, end] -> effective SM frequency
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
// 3D view of W [N][7168] bf16: {64 elements, rows, 112 K chunks}; box {64, rows_box, chunks_box}, SW128.
CUtensorMap wmap(const void* base, int N, int rows_box, int chunks_box) {
  CUtensorMap m;
  const cuuint64_t dims[3] = {64, (cuuint64_t)N, 112};
  const cuuint64_t strides[2] = {(cuuint64_t)ROWB, 128};
  const cuuint32_t box[3] = {64, (cuuint32_t)rows_box, (cuuint32_t)chunks_box};
  const cuuint32_t es[3] = {1, 1, 1};
  const CUresult r = encoder()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(base), dims, strides, box,
                               es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
  return m;
}

template <int M, bool AR = false>
void launch_m(const CUtensorMap& mt, const CUtensorMap& ms, const Params& p, int smem) {
  static bool init = false;
  if (!init) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(stage_kernel<M, AR>, cudaFuncAttributeMaxDynamicSharedMemorySize, 220 * 1024));
    init = true;
  }
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(CL * NCL);
  cfg.blockDim = dim3(NT);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, stage_kernel<M, AR>, mt, ms, p));
}

// cfg 0: TMEM tile 128 + smem tile 64 (M=64 MMA) + <= 24 CUDA-core rows per cluster (M <= 4).
// cfg 1: TMEM tile 128 + smem tile <= 88 rows (M=128 MMA, overlapped 88-row chunk stride), no CUDA rows.
// Row split and smem layout. cfg 0: TMEM tile 128 + smem tile 64 (M=64 MMA) + <= 24 CUDA-core rows per cluster
// (M <= 4); cfg 1: TMEM tile 128 + smem tile <= 88 rows (M=128 MMA, overlapped 88-row chunk stride), no CUDA rows.
int plan(Params& p, int N, int cfg, bool ar, int Mt = 8) {
  TORCH_CHECK(N >= 128 * NCL, "N must be >= 1920");
  p.N = N;
  const int rem = N - 128 * NCL;
  int cu_total = 0;
  if (cfg == 0) {
    p.sm_m = 64;
    p.sm_box = 64;
    cu_total = rem > 64 * NCL ? rem - 64 * NCL : 0;
    TORCH_CHECK(cu_total <= MAXCU * NCL, "N too large for cfg 0");
  } else {
    p.sm_m = 128;
    p.sm_box = 88;
    TORCH_CHECK(rem <= 88 * NCL, "N too large for cfg 1");
  }
  int s_next = 128 * NCL, c_next = 128 * NCL + (cfg == 0 ? 64 * NCL : rem);
  for (int c = 0; c < NCL; ++c) {
    p.t_row0[c] = 128 * c;
    int sr;
    if (cfg == 0) sr = std::max(0, std::min(64, rem - 64 * c));
    else sr = rem / NCL + (c < rem % NCL ? 1 : 0);
    p.s_row0[c] = cfg == 0 ? 128 * NCL + 64 * c : s_next;
    s_next += sr;
    p.s_rows[c] = sr;
    const int cr = cu_total / NCL + (c < cu_total % NCL ? 1 : 0);
    p.c_row0[c] = c_next;
    c_next += cr;
    p.c_rows[c] = cr;
  }
  if (rem == 0) p.sm_m = 0;
  p.r_bytes = cfg == 0 ? std::max(RING * SLOTB, 14 * 64 * 128) : 14 * 88 * 128 + 40 * 128;
  p.c_off = p.r_bytes;
  const int c_bytes = cu_total ? MAXCU * SLICEB : 0;
  p.b_off = (p.c_off + c_bytes + 1023) & ~1023;
  p.i_off = p.b_off + 1024 * KCH;
  p.ar_off = p.i_off + CL * OWN * 8 * 4;
  p.bar_off = p.ar_off + (ar ? ((ar_bytes(Mt) + 127) & ~127) : 0);
  const int smem = p.bar_off + 512;
  TORCH_CHECK(smem <= 220 * 1024, "smem ", smem);
  return smem;
}
const std::pair<CUtensorMap, CUtensorMap>& maps(const torch::Tensor& w, int N, int cfg, int sm_box) {
  static std::map<std::tuple<const void*, int, int>, std::pair<CUtensorMap, CUtensorMap>> cache;
  auto key = std::make_tuple(w.data_ptr(), N, cfg);
  auto it = cache.find(key);
  if (it == cache.end())
    it = cache.emplace(key, std::make_pair(wmap(w.data_ptr(), N, 128, 2), wmap(w.data_ptr(), N, sm_box, 7))).first;
  return it->second;
}

void stage_gemv(torch::Tensor mailbox, torch::Tensor w, torch::Tensor y, torch::Tensor done, int64_t cfgi,
                std::optional<torch::Tensor> gate, std::optional<torch::Tensor> tl, std::optional<torch::Tensor> dbg,
                int64_t ct) {
  const int M = y.size(0), N = y.size(1);
  TORCH_CHECK(w.size(0) == N && w.size(1) == K && w.scalar_type() == at::kBFloat16 && w.is_contiguous());
  TORCH_CHECK(mailbox.size(1) == K && mailbox.size(0) >= M && mailbox.is_contiguous());
  TORCH_CHECK(M == 1 || M == 2 || M == 4 || M == 8, "M must be 1, 2, 4 or 8");
  const int cfg = (int)(cfgi % 10);
  TORCH_CHECK(cfg == 0 || cfg == 1);
  // cfgi = cfg + 10 * detect_only + 100 * poll_mode + 1000 * (poll backoff in units of 32 ns) + 100000 * dbg flags
  TORCH_CHECK(!(cfg == 0 && M > 4), "cfg 0 (CUDA-core rows) supports M <= 4");
  TORCH_CHECK(ct == -1 || ct == KCH, "ct (split of the smem tile between tensor and CUDA cores) was removed");
  Params p{};
  p.w = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  p.mailbox = reinterpret_cast<const uint32_t*>(mailbox.data_ptr());
  p.y = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
  p.done = done.data_ptr<int>();
  p.gate = gate ? reinterpret_cast<unsigned long long*>(gate->data_ptr()) : nullptr;
  p.tl = tl ? reinterpret_cast<unsigned long long*>(tl->data_ptr()) : nullptr;
  p.dbg = dbg ? dbg->data_ptr<float>() : nullptr;
  p.mv = M;
  p.detect = (cfgi / 10) % 10;
  p.poll_mode = (cfgi / 100) % 10;
  p.poll_ns = (int)((cfgi / 1000) % 100) * 32;
  p.ef = 0;
  p.dbgf = (int)(cfgi / 100000);  // timing only: bit 0 skip TMEM MMAs, 1 skip smem MMAs, 3 skip CUDA-core share
  if (tl) TORCH_CHECK(tl->numel() >= CL * NCL * 32, "tl needs [120][32] int64");
  const int smem = plan(p, N, cfg, false);
  const auto& mp = maps(w, N, cfg, p.sm_box);
  switch (M) {
    case 1: launch_m<1>(mp.first, mp.second, p, smem); break;
    case 2: launch_m<2>(mp.first, mp.second, p, smem); break;
    case 4: launch_m<4>(mp.first, mp.second, p, smem); break;
    default: launch_m<8>(mp.first, mp.second, p, smem); break;
  }
}

// Fused ATTN front for decode: Lamport mailbox consume + pre-attention AttnRes (lamport_attn_res semantics,
// output norm required) + in_proj GEMV (tcgen05, staged weights). Writes prefix (and blocks[write_idx]) like
// lamport_attn_res, optionally the AttnRes output `out`, and y = out @ w^T. sched: int32[4] zero-initialised once
// ([0] mailbox reader count, [2:4] u64 prefix-reader epoch counter); never reset by the caller.
void attnres_inproj(torch::Tensor mailbox, torch::Tensor prefix, torch::Tensor blocks, torch::Tensor norm_weight,
                    torch::Tensor qk_weight, torch::Tensor output_norm_weight, std::optional<torch::Tensor> out,
                    torch::Tensor w, torch::Tensor y, torch::Tensor sched, int64_t num_blocks,
                    int64_t block_write_idx, double eps, double output_norm_eps, std::optional<torch::Tensor> tl,
                    std::optional<torch::Tensor> pf_ranges, int64_t poll) {
  const int mv = prefix.size(0), N = w.size(0);
  TORCH_CHECK(mv >= 1 && mv <= 8, "attnres_inproj: 1..8 tokens");
  TORCH_CHECK(w.size(1) == K && w.scalar_type() == at::kBFloat16 && w.is_contiguous());
  TORCH_CHECK(y.size(0) == mv && y.size(1) == N && y.is_contiguous() && y.scalar_type() == at::kBFloat16);
  TORCH_CHECK(mailbox.is_contiguous() && mailbox.scalar_type() == at::kBFloat16 && mailbox.size(-1) == K &&
              mailbox.numel() >= (int64_t)mv * K);
  auto rowok = [](const torch::Tensor& x) {
    return x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.stride(-1) == 1 && x.size(-1) == K &&
           reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0;
  };
  TORCH_CHECK(rowok(prefix) && prefix.dim() == 2 && prefix.stride(0) % 8 == 0);
  TORCH_CHECK(rowok(blocks) && blocks.dim() == 3 && blocks.size(0) >= mv && blocks.stride(0) % 8 == 0 &&
              blocks.stride(1) % 8 == 0);
  for (auto* t : {&norm_weight, &qk_weight, &output_norm_weight})
    TORCH_CHECK(rowok(*t) && t->is_contiguous() && t->numel() == K);
  TORCH_CHECK(num_blocks >= 0 && num_blocks <= 8 && num_blocks <= blocks.size(1));
  TORCH_CHECK(block_write_idx == -1 || (block_write_idx >= num_blocks && block_write_idx < blocks.size(1)));
  if (out) TORCH_CHECK(rowok(*out) && out->size(0) == mv && out->stride(0) % 8 == 0);
  TORCH_CHECK(sched.is_cuda() && sched.scalar_type() == at::kInt && sched.numel() >= 4 &&
              reinterpret_cast<uintptr_t>(sched.data_ptr()) % 8 == 0);
  const int M = mv <= 1 ? 1 : mv <= 2 ? 2 : mv <= 4 ? 4 : 8;
  const int cfg = M <= 2 ? 0 : 1;  // M = 3..8: no CUDA-core rows (at M = 4 they finished after the tensor pipe)
  Params p{};
  p.w = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  p.mailbox = reinterpret_cast<const uint32_t*>(mailbox.data_ptr());
  p.y = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
  p.done = sched.data_ptr<int>();
  p.pref_cnt = reinterpret_cast<unsigned long long*>(sched.data_ptr<int>() + 2);
  p.tl = tl ? reinterpret_cast<unsigned long long*>(tl->data_ptr()) : nullptr;
  if (tl) TORCH_CHECK(tl->numel() >= CL * NCL * 32, "tl needs [120][32] int64");
  p.mv = mv;
  auto bf = [](const torch::Tensor& x) { return reinterpret_cast<__nv_bfloat16*>(x.data_ptr()); };
  p.prefix = bf(prefix);
  p.blocks = bf(blocks);
  p.norm_w = bf(norm_weight);
  p.qk_w = bf(qk_weight);
  p.out_w = bf(output_norm_weight);
  p.out = out ? bf(*out) : nullptr;
  p.prefix_sm = prefix.stride(0);
  p.blocks_sm = blocks.stride(0);
  p.blocks_sr = blocks.stride(1);
  p.out_sm = out ? out->stride(0) : 0;
  p.nb = (int)num_blocks;
  p.widx = (int)block_write_idx;
  p.eps = (float)eps;
  p.out_eps = (float)output_norm_eps;
  // poll = mode + 10 * backoff: mode 0 = every lane re-polls its whole slice, 1 = first watch one 16-B word per
  // source-rank segment (low L2 pressure while the NVLS stores land), backoff in units of 32 ns between rounds
  p.poll_mode = (int)(poll % 10);
  p.poll_ns = (int)((poll / 10) % 100) * 32;
  p.ef = (int)((poll / 100000) % 10);  // poll += 100000: in_proj TMA with L2 evict_first (K3AF_EF)
  if (pf_ranges && pf_ranges->numel()) {
    TORCH_CHECK(pf_ranges->is_cuda() && pf_ranges->scalar_type() == at::kLong && pf_ranges->is_contiguous() &&
                pf_ranges->numel() % 2 == 0, "pf_ranges: int64 CUDA [n, 2] (ptr, bytes)");
    p.pf_rng = reinterpret_cast<const long long*>(pf_ranges->data_ptr<int64_t>());
    p.pf_n = (int)(pf_ranges->numel() / 2);
  }
  const int smem = plan(p, N, cfg, true, M);
  const auto& mp = maps(w, N, cfg, p.sm_box);
  switch (M) {
    case 1: launch_m<1, true>(mp.first, mp.second, p, smem); break;
    case 2: launch_m<2, true>(mp.first, mp.second, p, smem); break;
    case 4: launch_m<4, true>(mp.first, mp.second, p, smem); break;
    default: launch_m<8, true>(mp.first, mp.second, p, smem); break;
  }
}

// ---------------------------------------------------------------- chain stand-ins
// MOE stand-in: 120 CTAs in 8-clusters with 200 KB smem, PDL trigger at entry, wait, spin, exit.
__global__ void __cluster_dims__(CL, 1, 1) __launch_bounds__(128, 1) moe_emu_kernel(long long spin_ns,
                                                                                     unsigned long long* ts,
                                                                                     const char* pf, long long pf_bytes) {
  extern __shared__ unsigned char dummy[];
  const unsigned long long ta = gtimer();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const unsigned long long t0 = gtimer();
  // next layer's in_proj -> L2 (k3pf's job, issued from 30 of the resident MOE CTAs: >= 48 concurrent bulk-prefetch
  // issuers make the L2 drop prefetches, see Task 1)
  if (pf && threadIdx.x == 0 && (blockIdx.x & 3) == 0) {
    const int nis = (gridDim.x + 3) / 4, me = blockIdx.x / 4;
    const long long per = ((pf_bytes / nis) + 16383) & ~16383ll;
    const long long b0 = per * me, b1 = b0 + per < pf_bytes ? b0 + per : pf_bytes;
    for (long long b = b0; b < b1; b += 16384) {
      const uint32_t n = (uint32_t)(b1 - b < 16384 ? b1 - b : 16384);
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(pf + b), "r"(n) : "memory");
    }
  }
  while ((long long)(gtimer() - t0) < spin_ns) {
  }
  if (threadIdx.x == 0) {
    dummy[0] = 1;
    if (ts) {
      uint32_t smid;
      asm volatile("mov.u32 %0, %smid;" : "=r"(smid));
      ts[blockIdx.x * 4 + 0] = ta;
      ts[blockIdx.x * 4 + 1] = t0;
      ts[blockIdx.x * 4 + 2] = gtimer();
      ts[blockIdx.x * 4 + 3] = smid;
    }
  }
}
// TAIL stand-in: all CTAs trigger at entry, wait, spin; CTA 0 waits for the others, then either
//   gate == nullptr: spins hop ns (H4 hop, holding one SM) and stores the M canonicalised mailbox rows = landing;
//   gate != nullptr: stores the rows immediately and publishes gate = now + hop (consumers treat that as landing).
// ts[0] = landing; tts (optional, [ctas][4]): entry, wait returned, spin done, CTA 0: rows stored.
__global__ void tail_gate_kernel(long long spin_ns, long long hop_ns, int* counter, uint32_t* mailbox,
                                 const uint32_t* src, int M, unsigned long long* gate, unsigned long long* ts,
                                 unsigned long long* tts) {
  extern __shared__ unsigned char dummy[];
  __shared__ unsigned long long t_seen;
  const unsigned long long ta = gtimer();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const unsigned long long t0 = gtimer();
  while ((long long)(gtimer() - t0) < spin_ns) {
  }
  if (tts && threadIdx.x == 0) {
    tts[blockIdx.x * 4 + 0] = ta;
    tts[blockIdx.x * 4 + 1] = t0;
    tts[blockIdx.x * 4 + 2] = gtimer();
  }
  if (blockIdx.x != 0) {
    if (threadIdx.x == 0) {
      dummy[0] = 1;
      atomicAdd(counter, 1);
    }
    return;
  }
  // stage the canonicalised rows in smem first, so the landing itself is one fast burst of 16-B stores
  uint4* st = reinterpret_cast<uint4*>(dummy);
  for (int f = threadIdx.x; f < M * K / 8; f += blockDim.x) {
    uint4 u = reinterpret_cast<const uint4*>(src)[f];
    uint32_t* w = reinterpret_cast<uint32_t*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if ((w[i] & 0xffffu) == 0x8000u) w[i] &= 0xffff0000u;
      if ((w[i] >> 16) == 0x8000u) w[i] &= 0x0000ffffu;
    }
    st[f] = u;
  }
  if (threadIdx.x == 0) {
    while (*(volatile int*)counter < (int)gridDim.x - 1) {
    }
    *(volatile int*)counter = 0;
    t_seen = gtimer();
  }
  __syncthreads();
  if (!gate) {
    while ((long long)(gtimer() - t_seen) < hop_ns) {
    }
  }
  const unsigned long long t_hop = gtimer();
  for (int f = threadIdx.x; f < M * K / 8; f += blockDim.x) reinterpret_cast<uint4*>(mailbox)[f] = st[f];
  const unsigned long long t_st = gtimer();
  if (gate) __threadfence();
  __syncthreads();
  if (tts && threadIdx.x == 0) {
    tts[gridDim.x * 4 + 0] = t_seen;
    tts[gridDim.x * 4 + 1] = t_hop;
    tts[gridDim.x * 4 + 2] = t_st;
  }
  if (threadIdx.x == 0) {
    const unsigned long long now = gtimer();
    if (gate) {
      *(volatile unsigned long long*)gate = t_seen + hop_ns;
      if (ts) *ts = t_seen + hop_ns;
    } else if (ts) {
      *ts = now;
    }
    if (tts) tts[3] = now;
  }
}
void launch_pdl(const void* k, dim3 grid, dim3 block, int smem, void** args) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelExC(&cfg, k, args));
}
void moe_emu(int64_t ctas, int64_t smem_kb, int64_t spin_ns, std::optional<torch::Tensor> ts,
             std::optional<torch::Tensor> pf) {
  static int last = -1;
  const int smem = (int)smem_kb * 1024;
  if (last != smem) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(moe_emu_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    last = smem;
  }
  long long s = spin_ns;
  unsigned long long* t = ts ? reinterpret_cast<unsigned long long*>(ts->data_ptr()) : nullptr;
  const char* pp = pf ? reinterpret_cast<const char*>(pf->data_ptr()) : nullptr;
  long long pb = pf ? (long long)(pf->numel() * pf->element_size()) : 0;
  void* args[] = {&s, &t, &pp, &pb};
  launch_pdl((const void*)moe_emu_kernel, dim3(ctas), dim3(128), smem, args);
}
void tail_gate(int64_t ctas, int64_t smem_kb, int64_t spin_ns, int64_t hop_ns, torch::Tensor counter,
               torch::Tensor mailbox, torch::Tensor src, std::optional<torch::Tensor> gate, torch::Tensor ts,
               std::optional<torch::Tensor> tts) {
  static int last = -1;
  const int smem = (int)smem_kb * 1024;
  TORCH_CHECK(src.numel() * 2 <= smem, "tail_gate: smem must hold the M mailbox rows");
  if (last != smem) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(tail_gate_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    last = smem;
  }
  long long s = spin_ns, h = hop_ns;
  int* cp = counter.data_ptr<int>();
  uint32_t* mb = reinterpret_cast<uint32_t*>(mailbox.data_ptr());
  const uint32_t* sp = reinterpret_cast<const uint32_t*>(src.data_ptr());
  int M = src.size(0);
  unsigned long long* g = gate ? reinterpret_cast<unsigned long long*>(gate->data_ptr()) : nullptr;
  unsigned long long* t = reinterpret_cast<unsigned long long*>(ts.data_ptr());
  unsigned long long* tt = tts ? reinterpret_cast<unsigned long long*>(tts->data_ptr()) : nullptr;
  void* args[] = {&s, &h, &cp, &mb, &sp, &M, &g, &t, &tt};
  launch_pdl((const void*)tail_gate_kernel, dim3(ctas), dim3(256), smem, args);
}
}  // namespace

TORCH_LIBRARY(k3sgt, m) {
  m.def(
      "stage_gemv(Tensor(a!) mailbox, Tensor w, Tensor(b!) y, Tensor(c!) done, int cfg, Tensor(d!)? gate=None, "
      "Tensor(e!)? tl=None, Tensor(f!)? dbg=None, int ct=-1) -> ()");
  m.impl("stage_gemv", c10::DispatchKey::CUDA, &stage_gemv);
  m.def(
      "attnres_inproj(Tensor(a!) mailbox, Tensor(b!) prefix, Tensor(c!) blocks, Tensor norm_weight, "
      "Tensor qk_weight, Tensor output_norm_weight, Tensor(d!)? out, Tensor w, Tensor(e!) y, Tensor(f!) sched, "
      "int num_blocks, int block_write_idx, float eps, float output_norm_eps, Tensor(g!)? tl=None, "
      "Tensor? pf_ranges=None, int poll=0) -> ()");
  m.impl("attnres_inproj", c10::DispatchKey::CUDA, &attnres_inproj);
  m.def("moe_emu(int ctas, int smem_kb, int spin_ns, Tensor(a!)? ts=None, Tensor? pf=None) -> ()", &moe_emu);
  m.def(
      "tail_gate(int ctas, int smem_kb, int spin_ns, int hop_ns, Tensor(a!) counter, Tensor(b!) mailbox, Tensor src, "
      "Tensor(c!)? gate, Tensor(d!) ts, Tensor(e!)? tts=None) -> ()",
      &tail_gate);
}
