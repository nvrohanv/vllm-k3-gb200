// PDL weight-prefetching skinny GEMV for Kimi K3 decode (GB200 / sm_100a).
//
//   y[M, N] = x[M, K] @ W[N, K]^T      bf16 in, fp32 accumulate, bf16 out, 1 <= M <= 8.
//
// Idea (GPU analog of the TPU megakernel's cross-op weight prefetch into VMEM): the kernel
// is launched with cudaLaunchAttributeProgrammaticStreamSerialization. As soon as the
// predecessor executes `griddepcontrol.launch_dependents`, our CTAs become resident and,
// BEFORE `griddepcontrol.wait`, pull the first P "batches" of their (static) weight slice
// into shared memory with cp.async while the latency-bound predecessor is still running.
// After the wait (x now valid) the same smem ring is used as a deep cp.async pipeline for
// the rest of the weights, so HBM stays busy. Optionally (l2_rest) the part that does not fit
// in smem is prefetched into L2 (cp.async.bulk.prefetch.L2) before the wait as well.
//
// Work split
//   * CTA b owns whole rows [b*N/G, (b+1)*N/G) (<= 8*MAXG rows -> MAXG mma n-tiles of 8 rows).
//   * K is cut in 32-element blocks; block j is handled by warp j % NW. Batch bt = blocks
//     [bt*D*NW, (bt+1)*D*NW) = a contiguous column range of every row of the CTA.
//   * Lane (g = lane/4, t = lane%4) of a warp, for block j: 16 B of x[token g] and 16 B of
//     W[row 8*rg + g] at element 32*j + 8*t -> two mma.sync.m16n8k16 (A = x, rows 8..15 zero;
//     B = 8 weight rows). The k permutation inside the fragment is identical for A and B.
//   * Every lane cp.async's exactly the 16-byte fragments it will later consume into a
//     lane-private smem slot (warp reads are 512 B contiguous -> conflict free), so the
//     pipeline needs no barriers: cp.async.wait_group per thread is enough.
//   * Warp partial tiles are reduced through smem; bf16 result stored by the CTA.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

constexpr int kMaxSmem = 232448;  // sm_100 opt-in max dynamic smem per block
constexpr int kMaxStages = 16;
constexpr int kXChunks = 8;  // x is copied in up to 8 chunks, each with its own mbarrier

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ long long gtimer() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes,
                                           uint64_t pol) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;" ::"r"(dst),
               "l"(src), "r"(src_bytes), "l"(pol)
               : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}
// wait until at most n groups of this thread are pending (n clamped: stricter is still correct)
__device__ __forceinline__ void cp_async_wait_pending(int n) {
  if (n >= 15) { cp_async_wait<15>(); return; }
  switch (n) {
    case 0: cp_async_wait<0>(); break;
    case 1: cp_async_wait<1>(); break;
    case 2: cp_async_wait<2>(); break;
    case 3: cp_async_wait<3>(); break;
    case 4: cp_async_wait<4>(); break;
    case 5: cp_async_wait<5>(); break;
    case 6: cp_async_wait<6>(); break;
    case 7: cp_async_wait<7>(); break;
    case 8: cp_async_wait<8>(); break;
    case 9: cp_async_wait<9>(); break;
    case 10: cp_async_wait<10>(); break;
    case 11: cp_async_wait<11>(); break;
    case 12: cp_async_wait<12>(); break;
    case 13: cp_async_wait<13>(); break;
    case 14: cp_async_wait<14>(); break;
    default: cp_async_wait<15>(); break;
  }
}
__device__ __forceinline__ uint4 lds128(uint32_t addr) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(addr));
  return v;
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void fence_mbar_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)),
               "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  const uint32_t a = smem_u32(bar);
  uint32_t ok = 0;
  while (!ok) {
    asm volatile(
        "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        " selp.u32 %0, 1, 0, p;\n}\n"
        : "=r"(ok)
        : "r"(a), "r"(parity)
        : "memory");
  }
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes,
                                         uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::
          "r"(dst),
      "l"(src), "r"(bytes), "r"(smem_u32(bar))
      : "memory");
}
__device__ __forceinline__ void bulk_prefetch_l2(const void* src, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}
// D += A(16x16; rows 8..15 zero) * B(16x8)
__device__ __forceinline__ void mma_x_w(float (&c)[4], uint32_t a0, uint32_t a2, uint32_t b0,
                                        uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}

// S = smem ring depth (batches), P = batches issued before griddepcontrol.wait (<= S).
// ts (optional, int64) per CTA: [start, after griddepcontrol.wait, first batch ready, end].
__device__ __forceinline__ void store_out(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }
__device__ __forceinline__ void store_out(float* p, float v) { *p = v; }

// OutT = __nv_bfloat16 (bf16 output, fp32 accumulate) or float (fp32 logits, e.g. the MoE router)
template <int MAXG, int NW, int D, typename OutT>
__global__ void __launch_bounds__(NW * 32, 1)
    k3gemv_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                  OutT* __restrict__ y, int M, int N, int K, int S, int P, int l2_rest,
                  long long* __restrict__ ts) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int kLaneBytes = D * MAXG * 16;
  constexpr int kWarpBytes = 32 * kLaneBytes;
  constexpr int kSlotBytes = NW * kWarpBytes;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int G = gridDim.x;
  const int r0 = static_cast<int>((static_cast<long long>(blockIdx.x) * N) / G);
  const int r1 = static_cast<int>((static_cast<long long>(blockIdx.x + 1) * N) / G);
  const int R = r1 - r0;
  const size_t row_bytes = static_cast<size_t>(K) * 2;
  const uint32_t xstride = static_cast<uint32_t>(row_bytes) + 64;  // == 64 mod 128: no conflicts
  // smem: [8 x-chunk mbarriers | pad 128][red NW*MAXG*256][x: M rows, stride 2K+64][W ring]
  uint64_t* xbar = reinterpret_cast<uint64_t*>(smem);
  float* red = reinterpret_cast<float*>(smem + 128);
  uint8_t* xs = smem + 128 + NW * MAXG * 256;
  const uint32_t xs_bytes = (static_cast<uint32_t>(M) * xstride + 127u) & ~127u;
  const uint32_t my_base = smem_u32(xs + xs_bytes) + warp * kWarpBytes + lane * 16;
  const char* wcta = reinterpret_cast<const char*>(w) + static_cast<size_t>(r0) * row_bytes;
  const int NBT = K / (32 * NW * D);            // batches
  const int XB = (NBT + kXChunks - 1) / kXChunks;  // batches per x chunk / mbarrier
  const int NXC = (NBT + XB - 1) / XB;           // x chunks

  if (ts != nullptr && tid == 0) ts[blockIdx.x * 32 + 0] = gtimer();
  if (tid == 0) {
    for (int c = 0; c < NXC; ++c) mbar_init(&xbar[c], 1);
    fence_mbar_init();
  }
  __syncthreads();

  const char* wrow[MAXG];
  int wsz[MAXG];
#pragma unroll
  for (int rg = 0; rg < MAXG; ++rg) {
    const int lr = rg * 8 + g;
    wrow[rg] = wcta + static_cast<size_t>(min(lr, R - 1)) * row_bytes + t * 16;
    wsz[rg] = lr < R ? 16 : 0;  // zero-fill padding rows
  }
  const uint64_t pol = policy_evict_first();
  // (macros rather than lambdas: lambdas capturing the per-lane arrays by reference forced them
  //  to local memory, which put an L2 round trip on every batch)
  // (no runtime div/mod in the loop: with 1 CTA/SM the integer-division latency chains were
  //  ~500 cycles per batch; ring slots and x chunks are tracked incrementally instead)
  const uint32_t ring_end = my_base + static_cast<uint32_t>(S) * kSlotBytes;
  uint32_t isb = my_base;  // next slot to fill
  uint32_t lsb = my_base;  // next slot to consume
#define K3_ISSUE(bt_)                                                                     \
  {                                                                                       \
    const int bi_ = (bt_);                                                                \
    const uint32_t sb_ = isb;                                                             \
    isb += kSlotBytes;                                                                    \
    if (isb == ring_end) isb = my_base;                                                   \
    _Pragma("unroll") for (int d = 0; d < D; ++d) {                                       \
      const size_t col_ = static_cast<size_t>((bi_ * D + d) * NW + warp) * 64;            \
      _Pragma("unroll") for (int rg = 0; rg < MAXG; ++rg)                                 \
          cp_async16(sb_ + (d * MAXG + rg) * 512, wrow[rg] + col_, wsz[rg], pol);          \
    }                                                                                     \
    cp_async_commit();                                                                    \
  }

  // ---- pre-wait: stage the first P batches of static weights while the predecessor runs ----
  const int Sn = min(S, NBT);
  const int Pn = min(P, Sn);
  int issued = 0;
  for (; issued < Pn; ++issued) K3_ISSUE(issued);
  if (l2_rest && tid == 0 && Pn < NBT) {
    const size_t c0 = static_cast<size_t>(Pn) * D * NW * 64;
    for (int r = 0; r < R; ++r)
      bulk_prefetch_l2(wcta + static_cast<size_t>(r) * row_bytes + c0,
                       static_cast<uint32_t>(row_bytes - c0));
  }

  pdl_wait();  // predecessor grid complete, x visible
  if (ts != nullptr && tid == 0) ts[blockIdx.x * 32 + 1] = gtimer();

  // x -> smem: TMA bulk copies, one per (x chunk, token row), spread over warp 0's lanes; chunk c
  // (= batches [c*XB, (c+1)*XB)) completes xbar[c] so compute starts as soon as chunk 0 lands.
  const uint32_t xcb = static_cast<uint32_t>(XB) * D * NW * 64;  // bytes per chunk per row
  if (warp == 0) {
    if (lane < NXC) {
      const uint32_t c0 = lane * xcb;
      const uint32_t cb = min(xcb, static_cast<uint32_t>(row_bytes) - c0);
      mbar_expect_tx(&xbar[lane], cb * M);
    }
    for (int i = lane; i < NXC * M; i += 32) {
      const int c = i / M, m = i - c * M;
      const uint32_t c0 = c * xcb;
      const uint32_t cb = min(xcb, static_cast<uint32_t>(row_bytes) - c0);
      bulk_g2s(smem_u32(xs) + m * xstride + c0,
               reinterpret_cast<const char*>(x) + m * row_bytes + c0, cb, &xbar[c]);
    }
  }
  for (; issued < Sn; ++issued) K3_ISSUE(issued);

  // tokens >= M read row M-1: their C rows are garbage and discarded
  const uint32_t xlane = smem_u32(xs) + static_cast<uint32_t>(min(g, M - 1)) * xstride + t * 16;

  constexpr int NACC = MAXG <= 3 ? D : 1;  // independent accumulator chains
  float acc[NACC][MAXG][4];
#pragma unroll
  for (int a = 0; a < NACC; ++a)
#pragma unroll
    for (int rg = 0; rg < MAXG; ++rg) acc[a][rg][0] = acc[a][rg][1] = acc[a][rg][2] = acc[a][rg][3] = 0.f;

#define K3_LOAD(bt_, XR, WR)                                                              \
  {                                                                                       \
    const int bl_ = (bt_);                                                                \
    if (bl_ == xnext) {                                                                   \
      mbar_wait(&xbar[xci], 0);                                                           \
      ++xci;                                                                              \
      xnext += XB;                                                                        \
    }                                                                                     \
    if (!all_in_ring) cp_async_wait_pending(issued - bl_ - 1);                            \
    const uint32_t sb_ = lsb;                                                             \
    lsb += kSlotBytes;                                                                    \
    if (lsb == ring_end) lsb = my_base;                                                   \
    _Pragma("unroll") for (int d = 0; d < D; ++d) {                                       \
      XR[d] = lds128(xlane + ((bl_ * D + d) * NW + warp) * 64);                           \
      _Pragma("unroll") for (int rg = 0; rg < MAXG; ++rg) WR[d][rg] =                     \
          lds128(sb_ + (d * MAXG + rg) * 512);                                            \
    }                                                                                     \
  }
  int xci = 0, xnext = 0;
  // whole slice fits in the ring (small weights): one wait instead of per-batch waits
  const bool all_in_ring = Sn == NBT;
  if (all_in_ring) cp_async_wait<0>();
  uint4 xa[D], wa[D][MAXG];
  K3_LOAD(0, xa, wa);
  if (ts != nullptr && tid == 0) ts[blockIdx.x * 32 + 2] = gtimer();
  for (int bt = 0; bt < NBT; ++bt) {
    uint4 xb[D], wb[D][MAXG];
    if (bt + 1 < NBT) K3_LOAD(bt + 1, xb, wb);  // software pipeline: next batch's smem loads
#pragma unroll
    for (int d = 0; d < D; ++d)
#pragma unroll
      for (int rg = 0; rg < MAXG; ++rg) {
        mma_x_w(acc[d % NACC][rg], xa[d].x, xa[d].y, wa[d][rg].x, wa[d][rg].y);
        mma_x_w(acc[d % NACC][rg], xa[d].z, xa[d].w, wa[d][rg].z, wa[d][rg].w);
      }
    // slot bt % S was consumed (its registers fed the mma above) -> refill with batch bt + S
    if (issued < NBT) {
      K3_ISSUE(issued);
      ++issued;
    }
#pragma unroll
    for (int d = 0; d < D; ++d) {
      xa[d] = xb[d];
#pragma unroll
      for (int rg = 0; rg < MAXG; ++rg) wa[d][rg] = wb[d][rg];
    }
  }
#pragma unroll
  for (int a = 1; a < NACC; ++a)
#pragma unroll
    for (int rg = 0; rg < MAXG; ++rg) {
      acc[0][rg][0] += acc[a][rg][0];
      acc[0][rg][1] += acc[a][rg][1];
    }

  if (ts != nullptr && tid == 0) ts[blockIdx.x * 32 + 4] = gtimer();
#undef K3_LOAD
#undef K3_ISSUE
  // ---- cross-warp reduction; c0,c1 = y[token g][row 8*rg + 2t, +1] ----
#pragma unroll
  for (int rg = 0; rg < MAXG; ++rg)
    *reinterpret_cast<float2*>(&red[((warp * MAXG + rg) * 8 + g) * 8 + 2 * t]) =
        make_float2(acc[0][rg][0], acc[0][rg][1]);
  __syncthreads();
  for (int idx = tid; idx < MAXG * 64; idx += NW * 32) {
    const int rg = idx >> 6, tok = (idx >> 3) & 7, j = idx & 7;
    const int lr = rg * 8 + j;
    if (tok < M && lr < R) {
      float s = 0.f;
#pragma unroll
      for (int ww = 0; ww < NW; ++ww) s += red[((ww * MAXG + rg) * 8 + tok) * 8 + j];
      store_out(&y[static_cast<size_t>(tok) * N + r0 + lr], s);
    }
  }
  pdl_trigger();
  if (ts != nullptr && tid == 0) ts[blockIdx.x * 32 + 3] = gtimer();
}

// Synthetic latency-bound predecessor (emulates AttnRes / all-reduce): `nctas` CTAs, triggers
// launch_dependents immediately, then `iters` dependent L2 loads per thread (pointer chase),
// then spins until `spin_ns` have elapsed since its start, then writes x_out = x_src (made
// data-dependent on the chase). ts (optional) per CTA: [start, end].
template <bool kMaxShared>
__global__ void __launch_bounds__(256) pred_kernel(const __nv_bfloat16* __restrict__ xsrc,
                                                   __nv_bfloat16* __restrict__ xout, int n_elems,
                                                   const int* __restrict__ chase, int chase_mask,
                                                   int iters, long long spin_ns, int trigger,
                                                   long long* __restrict__ ts) {
  // trigger: 0 = never (dependents launch when this grid exits), 1 = at entry,
  //          > 1 = after `trigger` ns (e.g. a kernel that triggers after its main wait)
  const long long t0 = gtimer();
  if (ts != nullptr && threadIdx.x == 0) ts[blockIdx.x * 2 + 0] = t0;
  if (trigger == 1) pdl_trigger();
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = (gtid * 97) & chase_mask;
  for (int i = 0; i < iters; ++i) idx = __ldcg(chase + idx);
  bool fired = trigger <= 1;
  if (spin_ns > 0)
    while (gtimer() - t0 < spin_ns) {
      if (!fired && gtimer() - t0 >= trigger) {
        pdl_trigger();
        fired = true;
      }
    }
  if (!fired) pdl_trigger();
  const int stride = gridDim.x * blockDim.x * 8;
  for (int e = gtid * 8; e < n_elems; e += stride) {
    uint4 v = *reinterpret_cast<const uint4*>(xsrc + e);
    if (idx == -1) v.x = 0;  // never true; keeps the chase on the critical path
    *reinterpret_cast<uint4*>(xout + e) = v;
  }
  __syncthreads();
  if (ts != nullptr && threadIdx.x == 0) ts[blockIdx.x * 2 + 1] = gtimer();
}

template <int MAXG, int NW, int D, typename OutT>
void launch_gemv(const __nv_bfloat16* x, const __nv_bfloat16* w, OutT* y, int M, int N,
                 int K, int S, int P, int l2_rest, int grid, bool pdl, long long* ts, size_t smem,
                 cudaStream_t stream) {
  auto kern = k3gemv_kernel<MAXG, NW, D, OutT>;
  static bool attr_set = false;
  if (!attr_set) {
    C10_CUDA_CHECK(
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxSmem));
    attr_set = true;
  }
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(NW * 32);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, x, w, y, M, N, K, S, P, l2_rest, ts));
}

int num_sms() {
  static int n = 0;
  if (n == 0) {
    int dev;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
  }
  return n;
}

}  // namespace

// plan -> [G, rows/CTA max, MAXG, D, S (ring batches), P (pre-wait batches), NBT (batches),
//          smem bytes]
std::vector<int64_t> gemv_plan(int64_t N, int64_t K, int64_t stage, int64_t grid, int64_t nw,
                               int64_t max_smem, int64_t M) {
  const int64_t G = grid > 0 ? grid : num_sms();
  const int64_t rmax = (N + G - 1) / G;
  const int64_t ng = (rmax + 7) / 8;
  TORCH_CHECK(ng <= 8, "k3gemv: too many rows per CTA (", rmax, "); raise grid");
  const int64_t maxg = ng <= 4 ? ng : (ng <= 6 ? 6 : 8);
  TORCH_CHECK(K % (32 * nw) == 0, "K must be a multiple of 32*nw");
  const int64_t NB = K / (32 * nw);
  int64_t D = 1;
  for (int64_t d : {4, 3, 2}) {
    if (NB % d == 0 && d * maxg <= 6) {
      D = d;
      break;
    }
  }
  const int64_t nbt = NB / D;
  const int64_t slot = nw * 32 * D * maxg * 16;
  const int64_t red = 128 + nw * maxg * 256 + ((M * (2 * K + 64) + 127) / 128) * 128;
  const int64_t cap = max_smem > 0 ? std::min<int64_t>(max_smem, kMaxSmem) : kMaxSmem;
  const int64_t S = std::min<int64_t>({(int64_t)kMaxStages, nbt, (cap - red) / slot});
  // the consumer loads batch b+1 before refilling slot b, so it needs 2 slots when nbt > 1
  TORCH_CHECK(S >= std::min<int64_t>(2, nbt), "k3gemv: smem budget too small (", cap, " B)");
  const int64_t P = std::max<int64_t>(0, std::min<int64_t>(stage, S));
  return {G, rmax, maxg, D, S, P, nbt, red + S * slot};
}

void gemv(torch::Tensor x, torch::Tensor w, torch::Tensor y, int64_t stage, int64_t l2_rest,
          int64_t grid, int64_t nw, int64_t pdl, int64_t max_smem,
          std::optional<torch::Tensor> ts) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16 &&
              (y.scalar_type() == at::kBFloat16 || y.scalar_type() == at::kFloat));
  const bool f32 = y.scalar_type() == at::kFloat;
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && y.is_contiguous());
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(w.size(1) == K && y.size(0) == M && y.size(1) == N);
  TORCH_CHECK(M >= 1 && M <= 8, "M must be 1..8");
  auto plan = gemv_plan(N, K, stage, grid, nw, max_smem, M);
  const int G = plan[0], maxg = plan[2], D = plan[3], S = plan[4], P = plan[5];
  const size_t smem = plan[7];
  long long* tsp = ts ? reinterpret_cast<long long*>(ts->data_ptr<int64_t>()) : nullptr;
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  auto xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto wp = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  auto yp = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
  auto yf = reinterpret_cast<float*>(y.data_ptr());
  const bool p = pdl != 0;
  const int l2 = static_cast<int>(l2_rest);
#define K3_GEMV(MG, NWV, DV)                                                                    \
  if (!f32 && maxg == MG && nw == NWV && D == DV) {                                             \
    launch_gemv<MG, NWV, DV>(xp, wp, yp, M, N, K, S, P, l2, G, p, tsp, smem, stream);           \
    return;                                                                                     \
  }
#define K3_GEMV_F32(MG, NWV, DV)                                                                \
  if (f32 && maxg == MG && nw == NWV && D == DV) {                                              \
    launch_gemv<MG, NWV, DV>(xp, wp, yf, M, N, K, S, P, l2, G, p, tsp, smem, stream);           \
    return;                                                                                     \
  }
  // fp32 output (MoE router gate 896 x 7168: <= 6 rows per CTA)
  K3_GEMV_F32(1, 8, 4) K3_GEMV_F32(1, 16, 2)
  // K = 7168 shapes
  K3_GEMV(1, 8, 4) K3_GEMV(2, 8, 2) K3_GEMV(3, 8, 2) K3_GEMV(4, 8, 1)
  K3_GEMV(1, 16, 2) K3_GEMV(2, 16, 2) K3_GEMV(3, 16, 2) K3_GEMV(4, 16, 1)
  // K = 768 / 384 shapes (N = 7168 -> 48 rows per CTA -> 6 groups)
  K3_GEMV(6, 4, 1) K3_GEMV(6, 8, 1) K3_GEMV(6, 12, 1) K3_GEMV(6, 24, 1)
  K3_GEMV(8, 4, 1) K3_GEMV(8, 8, 1) K3_GEMV(8, 12, 1) K3_GEMV(8, 24, 1)
#undef K3_GEMV
#undef K3_GEMV_F32
  TORCH_CHECK(false, "k3gemv: no instantiation for maxg=", maxg, " nw=", nw, " D=", D);
}

void pred(torch::Tensor xsrc, torch::Tensor xout, torch::Tensor chase, int64_t nctas,
          int64_t iters, int64_t spin_ns, int64_t trigger, int64_t max_shared_carveout,
          std::optional<torch::Tensor> ts) {
  TORCH_CHECK(xsrc.numel() == xout.numel() && xsrc.numel() % 8 == 0);
  const int64_t n = chase.numel();
  TORCH_CHECK((n & (n - 1)) == 0, "chase size must be a power of two");
  long long* tsp = ts ? reinterpret_cast<long long*>(ts->data_ptr<int64_t>()) : nullptr;
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  // max_shared_carveout: configure the SM for max shared memory while the predecessor runs, so
  // the prefetching GEMV's big-smem CTAs can co-reside with it (the carveout of an SM can only
  // change when it is idle).
  static bool attr_set = false;
  if (!attr_set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(pred_kernel<true>,
                                        cudaFuncAttributePreferredSharedMemoryCarveout,
                                        cudaSharedmemCarveoutMaxShared));
    attr_set = true;
  }
  auto kern = max_shared_carveout ? pred_kernel<true> : pred_kernel<false>;
  kern<<<static_cast<int>(nctas), 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(xsrc.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(xout.data_ptr()), static_cast<int>(xsrc.numel()),
      chase.data_ptr<int>(), static_cast<int>(n - 1), static_cast<int>(iters), spin_ns,
      static_cast<int>(trigger), tsp);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

TORCH_LIBRARY(k3gemv, m) {
  m.def(
      "gemv(Tensor x, Tensor w, Tensor(a!) y, int stage, int l2_rest, int grid, int nw, "
      "int pdl, int max_smem, Tensor? ts) -> ()");
  m.def(
      "pred(Tensor xsrc, Tensor(a!) xout, Tensor chase, int nctas, int iters, int spin_ns, "
      "int trigger, int max_shared_carveout, Tensor? ts) -> ()");
  m.def("plan(int N, int K, int stage, int grid, int nw, int max_smem, int M) -> int[]",
        &gemv_plan);
}
TORCH_LIBRARY_IMPL(k3gemv, CUDA, m) {
  m.impl("gemv", &gemv);
  m.impl("pred", &pred);
}
