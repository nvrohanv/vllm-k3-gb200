// Minimal sm_100a primitives: mbarrier, TMA / bulk copies, tcgen05 (alloc, mma, cp, ld, commit).
#pragma once
#include <cstdint>
#include <cuda.h>

namespace tc {

__device__ __forceinline__ uint32_t su32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---------------------------------------------------------------- mbarrier
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(su32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void fence_mbar_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(su32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(su32(bar)), "r"(bytes)
               : "memory");
}
// complete `bytes` of pending tx without a copy (debug)
__device__ __forceinline__ void mbar_arrive_tx_dummy(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.complete_tx.shared::cta.b64 [%0], %1;" ::"r"(su32(bar)), "r"(bytes) : "memory");
}
#ifndef TC_SUSPEND_NS
#define TC_SUSPEND_NS 0x989680
#endif
// With a suspend-time hint the waiting thread sleeps until the phase completes (or the hint expires)
// instead of spinning and stealing issue slots from the other warps of its SM sub-partition.
__device__ __forceinline__ bool mbar_try_wait(uint32_t addr, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n\t"
      "selp.u32 %0, 1, 0, p;\n\t}"
      : "=r"(ok)
      : "r"(addr), "r"(parity), "n"(TC_SUSPEND_NS)
      : "memory");
  return ok != 0;
}
// Non-blocking phase test.
__device__ __forceinline__ bool mbar_test(uint64_t* bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
      "selp.u32 %0, 1, 0, p;\n\t}"
      : "=r"(ok)
      : "r"(su32(bar)), "r"(parity)
      : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  const uint32_t a = su32(bar);
#ifdef TC_WAIT_TIMEOUT
  // time-based (the suspend hint makes an iteration count meaningless): trap after 2 s
  uint64_t t0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  while (!mbar_try_wait(a, parity)) {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    if (t - t0 > 2000000000ull) {
      printf("mbar_wait timeout: block %d thread %d bar %p parity %u\n", blockIdx.x, threadIdx.x, bar, parity);
      asm volatile("trap;");
    }
  }
#else
  while (!mbar_try_wait(a, parity)) {
  }
#endif
}

// ---------------------------------------------------------------- TMA / bulk
__device__ __forceinline__ void prefetch_tmap(const void* map) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(map) : "memory");
}
__device__ __forceinline__ void tma_load_2d(void* dst, const void* map, uint64_t* bar, int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4}], [%2];" ::"r"(su32(dst)),
      "l"(map), "r"(su32(bar)), "r"(c0), "r"(c1)
      : "memory");
}
__device__ __forceinline__ void tma_load_3d(void* dst, const void* map, uint64_t* bar, int c0, int c1, int c2) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5}], [%2];" ::"r"(su32(dst)),
      "l"(map), "r"(su32(bar)), "r"(c0), "r"(c1), "r"(c2)
      : "memory");
}
__device__ __forceinline__ void tma_load_3d_hint(void* dst, const void* map, uint64_t* bar, int c0, int c1, int c2,
                                                 uint64_t policy) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4, %5}], [%2], %6;" ::"r"(su32(dst)),
      "l"(map), "r"(su32(bar)), "r"(c0), "r"(c1), "r"(c2), "l"(policy)
      : "memory");
}
__device__ __forceinline__ void bulk_load_hint(void* dst, const void* src, uint32_t bytes, uint64_t* bar,
                                               uint64_t policy) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(
          su32(dst)),
      "l"(src), "r"(bytes), "r"(su32(bar)), "l"(policy)
      : "memory");
}
__device__ __forceinline__ void tma_load_2d_hint(void* dst, const void* map, uint64_t* bar, int c0, int c1,
                                                 uint64_t policy) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(su32(dst)),
      "l"(map), "r"(su32(bar)), "r"(c0), "r"(c1), "l"(policy)
      : "memory");
}
__device__ __forceinline__ void bulk_load(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
          su32(dst)),
      "l"(src), "r"(bytes), "r"(su32(bar))
      : "memory");
}
__device__ __forceinline__ void prefetch_l2(const void* src, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}

// ---------------------------------------------------------------- fences
__device__ __forceinline__ void fence_proxy_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void tc_fence_before() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tc_fence_after() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

// ---------------------------------------------------------------- TMEM
// Warp-wide.
__device__ __forceinline__ void tmem_alloc(uint32_t* dst_smem, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(su32(dst_smem)),
               "r"(ncols)
               : "memory");
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols) : "memory");
}

// Shared-memory matrix descriptor (sm_100 format, version 1).
// layout: 0 = none, 2 = 128B swizzle, 4 = 64B, 6 = 32B.
__device__ __forceinline__ uint64_t smem_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo, uint32_t layout) {
  return static_cast<uint64_t>((saddr & 0x3FFFF) >> 4) | (static_cast<uint64_t>((lbo & 0x3FFFF) >> 4) << 16) |
         (static_cast<uint64_t>((sbo & 0x3FFFF) >> 4) << 32) | (1ull << 46) |
         (static_cast<uint64_t>(layout) << 61);
}
__device__ __forceinline__ uint64_t desc_sw128(uint32_t saddr) { return smem_desc(saddr, 16, 1024, 2); }

// Block-scaled instruction descriptor, kind::mxf8f6f4, K-major A and B, UE8M0 scales.
// fmt: 0 = E4M3, 1 = E5M2, 3 = E2M3, 4 = E3M2, 5 = E2M1.
__host__ __device__ constexpr uint32_t idesc_mxf8f6f4(int M, int N, uint32_t afmt, uint32_t bfmt) {
  return (afmt << 7) | (bfmt << 10) | (static_cast<uint32_t>(N >> 3) << 17) | (1u << 23) |
         (static_cast<uint32_t>(M >> 4) << 24);
}
__device__ __forceinline__ uint32_t idesc_sf(uint32_t base, uint32_t a_sf_id, uint32_t b_sf_id) {
  return base | (b_sf_id << 4) | (a_sf_id << 29);
}

__device__ __forceinline__ void mma_mxf8f6f4(uint32_t d_tmem, uint64_t adesc, uint64_t bdesc, uint32_t idesc,
                                             uint32_t sfa_tmem, uint32_t sfb_tmem, uint32_t accum) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p;\n\t}" ::"r"(d_tmem),
      "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accum), "r"(sfa_tmem), "r"(sfb_tmem)
      : "memory");
}
// Same, but executed by a full warp: one elected lane issues (operands stay warp-uniform).
__device__ __forceinline__ void mma_mxf8f6f4_elect(uint32_t d_tmem, uint64_t adesc, uint64_t bdesc, uint32_t idesc,
                                                   uint32_t sfa_tmem, uint32_t sfb_tmem, uint32_t accum) {
  asm volatile(
      "{\n\t.reg .pred p, e;\n\tsetp.ne.b32 p, %4, 0;\n\telect.sync _|e, 0xffffffff;\n\t"
      "@e tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p;\n\t}" ::"r"(d_tmem),
      "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accum), "r"(sfa_tmem), "r"(sfb_tmem)
      : "memory");
}
// Arrive (once) on an mbarrier when all prior tcgen05 async ops of this thread complete.
__device__ __forceinline__ void mma_commit(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(su32(bar))
               : "memory");
}
// 32 lanes x 128 bit from smem, replicated into all four 32-lane subpartitions.
__device__ __forceinline__ void tmem_cp_32x128b(uint32_t taddr, uint64_t sdesc) {
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(taddr), "l"(sdesc) : "memory");
}
// Two 32x128b copies (desc, desc + 512 B -> taddr, taddr + 4), issued by one elected lane.
__device__ __forceinline__ void cp2_elect(uint32_t taddr, uint64_t sdesc) {
  asm volatile(
      "{\n\t.reg .pred e;\n\t.reg .b64 d1;\n\t.reg .b32 t1;\n\t"
      "elect.sync _|e, 0xffffffff;\n\t"
      "add.s64 d1, %1, 32;\n\tadd.u32 t1, %0, 4;\n\t"
      "@e tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;\n\t"
      "@e tcgen05.cp.cta_group::1.32x128b.warpx4 [t1], d1;\n\t}" ::"r"(taddr),
      "l"(sdesc)
      : "memory");
}
__device__ __forceinline__ void tmem_ld16(uint32_t taddr, float (&v)[16]) {
  uint32_t r[16];
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
      : "r"(taddr)
      : "memory");
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
  for (int i = 0; i < 16; ++i) v[i] = __uint_as_float(r[i]);
}
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tma_prefetch_3d(const void* map, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" ::"l"(map), "r"(c0), "r"(c1),
               "r"(c2)
               : "memory");
}
__device__ __forceinline__ void tmem_ld8(uint32_t taddr, float (&v)[8]) {
  uint32_t r[8];
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
               : "r"(taddr)
               : "memory");
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
  for (int i = 0; i < 8; ++i) v[i] = __uint_as_float(r[i]);
}

// ---------------------------------------------------------------- misc
__device__ __forceinline__ bool elect_one() {
  uint32_t pred = 0;
  asm volatile(
      "{\n\t.reg .pred P1;\n\telect.sync _|P1, 0xffffffff;\n\tselp.u32 %0, 1, 0, P1;\n\t}"
      : "=r"(pred));
  return pred != 0;
}
__device__ __forceinline__ void named_bar(int id, int nthreads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(nthreads) : "memory");
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void red_release_add(int* p, int v) {
  asm volatile("red.release.gpu.global.add.s32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}

}  // namespace tc
