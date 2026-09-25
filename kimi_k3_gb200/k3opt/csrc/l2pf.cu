// k3pf: cross-layer L2 weight prefetch for Kimi K3 decode on GB200 (agent l2pf).
//
//   torch.ops.k3pf.prefetch_l2(Tensor ranges, int n) -> ()
//   torch.ops.k3pf.prefetch_l2_cfg(Tensor ranges, int n, int grid, int chunk, int policy, int pdl) -> ()
//
// `ranges` is a device int64 tensor holding n (ptr, bytes) pairs (row-major [n, 2] or flat [2n]),
// built once at init. Because the kernel reads the table from device memory, the launch has
// only static arguments and is CUDA-graph safe (FULL capture and eager).
//
// Each CTA runs one warp; every thread issues `cp.async.bulk.prefetch.L2.global` for 16 KB chunks
// (chunk c of the concatenated ranges goes to global thread c % (grid*32)), range by range, so the
// first range lands first. Measured on GB200 (see RESULTS.md):
//   * one SM's TMA issues bulk L2 prefetches at ~250 GB/s; the issuing thread stalls when its queue
//     is full, so the kernel lives ~bytes / min(grid*250 GB/s, HBM) (55 MB, grid 32: ~8-9 us);
//   * too many issuers (152 CTAs x 32 thr) oversubscribe the memory system and the hardware DROPS
//     most prefetches (hint semantics) -> grid 24..48 is the sweet spot, default 32;
//   * chunks of 8..64 KB behave the same; >=256 KB chunks are partially / fully ignored;
//   * L2::evict_last cache-hint policy gives no useful protection on GB200 (no persisting carve-out),
//     so the default policy is the plain prefetch (policy 0).
//   * policy 4 (opt-in, prefetch_l2_cfg only) replaces the prefetch by a TMA bulk-load fill; see below.
// Launched with programmatic stream serialization (PDL); it calls griddepcontrol.launch_dependents
// first and never griddepcontrol.wait (it reads nothing produced upstream).
#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kThreads = 32;
constexpr int kDefaultGrid = 32;
constexpr int64_t kDefaultChunk = 16384;

__global__ void __launch_bounds__(kThreads) k3pf_prefetch_kernel(const int64_t* __restrict__ ranges, int n,
                                                                  int64_t chunk, int policy) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  uint64_t pol = 0;
  if (policy == 1) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  else if (policy == 2) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  else if (policy == 3) asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pol));
  const int64_t tid = (int64_t)blockIdx.x * kThreads + threadIdx.x;
  const int64_t nth = (int64_t)gridDim.x * kThreads;
  int64_t phase = 0;  // global chunk index of this range's first chunk (keeps work balanced across ranges)
  for (int r = 0; r < n; ++r) {
    const uint64_t base = (uint64_t)__ldg(ranges + 2 * r);
    const int64_t bytes = __ldg(ranges + 2 * r + 1);
    if (base == 0 || bytes <= 0) continue;
    const int64_t nch = (bytes + chunk - 1) / chunk;
    // first local chunk handled by this thread: smallest c >= 0 with (phase + c) % nth == tid
    int64_t c = tid - (phase % nth);
    if (c < 0) c += nth;
    for (; c < nch; c += nth) {
      const int64_t off = c * chunk;
      int64_t sz = bytes - off < chunk ? bytes - off : chunk;
      sz &= ~int64_t(15);
      if (sz <= 0) continue;
      const uint64_t p = (base + off) & ~uint64_t(15);
      if (policy == 0) {
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"((uint32_t)sz) : "memory");
      } else {
        asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" ::"l"(p), "r"((uint32_t)sz),
                     "l"(pol)
                     : "memory");
      }
    }
    phase += nch;
  }
}

// policy 4: TMA bulk-LOAD fill. Thread 0 of each CTA streams its chunks (chunk c -> CTA c % grid) through a
// 4-slot shared-memory ring with cp.async.bulk + mbarriers and discards them. Unlike the L2 prefetch, the
// lines are also cached in the loading SM's near L2 (GB200 caches reads on the requester's die), which
// makes the consuming GEMM ~1 us faster for 30-46 MB weights, at the price of 4*chunk bytes of smem per
// CTA while it runs (~9 us for 55 MB at grid 64, chunk 32 KB).
constexpr int kSlots = 4;

__global__ void __launch_bounds__(kThreads) k3pf_tmaload_kernel(const int64_t* __restrict__ ranges, int n,
                                                                 int chunk) {
  extern __shared__ __align__(128) uint8_t ring[];
  __shared__ __align__(8) uint64_t bar[kSlots];
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (threadIdx.x != 0) return;
  for (int i = 0; i < kSlots; ++i) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[i]);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(a));
  }
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  auto wait = [&](int64_t j) {
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[j % kSlots]);
    uint32_t par = (uint32_t)((j / kSlots) & 1);
    uint32_t done = 0;
    while (!done) {
      asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                   : "=r"(done)
                   : "r"(ba), "r"(par)
                   : "memory");
    }
  };
  int64_t k = 0, phase = 0;
  for (int r = 0; r < n; ++r) {
    const uint64_t base = (uint64_t)__ldg(ranges + 2 * r);
    const int64_t bytes = __ldg(ranges + 2 * r + 1);
    if (base == 0 || bytes <= 0) continue;
    const int64_t nch = (bytes + chunk - 1) / chunk;
    int64_t c = (int64_t)blockIdx.x - (phase % gridDim.x);
    if (c < 0) c += gridDim.x;
    for (; c < nch; c += gridDim.x) {
      const int64_t off = c * chunk;
      int64_t sz = bytes - off < chunk ? bytes - off : chunk;
      sz &= ~int64_t(15);
      const uint64_t src = (base + off) & ~uint64_t(15);
      if (sz <= 0) continue;
      if (k >= kSlots) wait(k - kSlots);
      const int slot = (int)(k % kSlots);
      uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[slot]);
      uint32_t dst = (uint32_t)__cvta_generic_to_shared(ring + (size_t)slot * chunk);
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(ba), "r"((uint32_t)sz) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
                   "l"(src), "r"((uint32_t)sz), "r"(ba)
                   : "memory");
      ++k;
    }
    phase += nch;
  }
  for (int64_t j = (k > kSlots ? k - kSlots : 0); j < k; ++j) wait(j);  // never exit with copies in flight
}

void launch_prefetch(const at::Tensor& ranges, int64_t n, int64_t grid, int64_t chunk, int64_t policy, bool pdl) {
  TORCH_CHECK(ranges.is_cuda(), "k3pf: ranges must be a CUDA tensor");
  TORCH_CHECK(ranges.scalar_type() == at::kLong, "k3pf: ranges must be int64");
  TORCH_CHECK(ranges.is_contiguous(), "k3pf: ranges must be contiguous");
  TORCH_CHECK(n >= 0 && 2 * n <= ranges.numel(), "k3pf: n out of range for ranges");
  TORCH_CHECK(grid >= 1 && grid <= 1024, "k3pf: grid must be in [1, 1024]");
  TORCH_CHECK(chunk >= 16 && chunk % 16 == 0 && chunk <= (1 << 20), "k3pf: chunk must be a multiple of 16 <= 1 MiB");
  TORCH_CHECK(policy >= 0 && policy <= 4, "k3pf: policy must be 0..4");
  TORCH_CHECK(policy != 4 || chunk <= 32768, "k3pf: policy 4 (TMA load) needs chunk <= 32 KB");
  if (n == 0) return;
  c10::cuda::CUDAGuard guard(ranges.device());
  const int64_t* r = ranges.data_ptr<int64_t>();
  int ni = (int)n, pi = (int)policy;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)grid);
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  if (policy == 4) {
    static bool smem_attr_set = false;
    if (!smem_attr_set) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(k3pf_tmaload_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          kSlots * 32768));
      smem_attr_set = true;
    }
    cfg.dynamicSmemBytes = (size_t)kSlots * chunk;
    int ci = (int)chunk;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3pf_tmaload_kernel, r, ni, ci));
    return;
  }
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3pf_prefetch_kernel, r, ni, chunk, pi));
}

void prefetch_l2(const at::Tensor& ranges, int64_t n) {
  launch_prefetch(ranges, n, kDefaultGrid, kDefaultChunk, 0, true);
}

void prefetch_l2_cfg(const at::Tensor& ranges, int64_t n, int64_t grid, int64_t chunk, int64_t policy, int64_t pdl) {
  launch_prefetch(ranges, n, grid, chunk, policy, pdl != 0);
}

}  // namespace

TORCH_LIBRARY(k3pf, m) {
  m.def("prefetch_l2(Tensor ranges, int n) -> ()");
  m.def("prefetch_l2_cfg(Tensor ranges, int n, int grid, int chunk, int policy, int pdl) -> ()");
}

TORCH_LIBRARY_IMPL(k3pf, CUDA, m) {
  m.impl("prefetch_l2", &prefetch_l2);
  m.impl("prefetch_l2_cfg", &prefetch_l2_cfg);
}
