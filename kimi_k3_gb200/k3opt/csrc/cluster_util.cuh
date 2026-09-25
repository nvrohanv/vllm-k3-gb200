// Thread-block-cluster / DSMEM helpers for moebody (agents/moe8/body).
#pragma once
#include <cstdint>

namespace cl {

__device__ __forceinline__ uint32_t ctarank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ uint32_t nctarank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ uint32_t clusterid() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%clusterid.x;" : "=r"(r));
  return r;
}
__device__ __forceinline__ uint32_t nclusterid() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%nclusterid.x;" : "=r"(r));
  return r;
}
// shared::cta address -> shared::cluster address of the same offset in CTA `rank` of this cluster
__device__ __forceinline__ uint32_t mapa(uint32_t saddr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(saddr), "r"(rank));
  return r;
}
// Remote (or local) async store with complete_tx on an mbarrier of the same destination CTA.
__device__ __forceinline__ void st_async_v4(uint32_t caddr, uint4 v, uint32_t cbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(caddr),
               "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "r"(cbar)
               : "memory");
}
__device__ __forceinline__ void st_async_v2(uint32_t caddr, uint32_t a, uint32_t b, uint32_t cbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];" ::"r"(caddr), "r"(a),
               "r"(b), "r"(cbar)
               : "memory");
}
__device__ __forceinline__ void st_async_b32(uint32_t caddr, uint32_t v, uint32_t cbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(caddr), "r"(v), "r"(cbar)
               : "memory");
}
// Wait on a local mbarrier whose phase is completed by remote st.async (acquire at cluster scope).
// Spins on test_wait: a thread suspended in try_wait is not necessarily woken by a remote complete_tx.
#ifndef CL_WAIT_MODE
#define CL_WAIT_MODE 0
#endif
__device__ __forceinline__ void mbar_wait_cluster(uint32_t addr, uint32_t parity) {
#ifdef TC_WAIT_TIMEOUT
  {
    uint64_t t0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    for (;;) {
      uint32_t ok;
      asm volatile(
          "{\n\t.reg .pred P1;\n\tmbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 P1, [%1], %2;\n\t"
          "selp.u32 %0, 1, 0, P1;\n\t}"
          : "=r"(ok)
          : "r"(addr), "r"(parity)
          : "memory");
      if (ok) return;
      uint64_t t;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
      if (t - t0 > 2000000000ull) {
        printf("mbar_wait_cluster timeout: block %d thread %d bar %x parity %u\n", blockIdx.x, threadIdx.x, addr, parity);
        asm volatile("trap;");
      }
    }
  }
#endif
#if CL_WAIT_MODE == 0
  asm volatile(
      "{\n\t.reg .pred P1;\n\tWAIT_%=:\n\t"
      "mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\n\t"
      "@!P1 bra WAIT_%=;\n\t}" ::"r"(addr),
      "r"(parity)
      : "memory");
#elif CL_WAIT_MODE == 1
  asm volatile(
      "{\n\t.reg .pred P1;\n\tWAIT_%=:\n\t"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\n\t"
      "@!P1 bra WAIT_%=;\n\t}" ::"r"(addr),
      "r"(parity)
      : "memory");
#else
  asm volatile(
      "{\n\t.reg .pred P1;\n\tWAIT_%=:\n\t"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1, %2;\n\t"
      "@!P1 bra WAIT_%=;\n\t}" ::"r"(addr),
      "r"(parity), "r"(0x989680)
      : "memory");
#endif
}
__device__ __forceinline__ void arrive_release() { asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory"); }
__device__ __forceinline__ void wait_acquire() { asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); }

}  // namespace cl
