// Fused Kimi K3 MLA decode (TP16 rank: 6 heads, q_lora 1536, kv_lora 512, rope 64, nope 128,
// v 128, fp8 e4m3 paged KV cache with 64-token kernel pages, sigmoid output gate). sm_100a.
//
// Replaces, in MultiHeadLatentAttention decode (vllm/models/kimi_k3/nvidia/mla.py):
//   k3mla.q_prep   = fused_q_kv_rmsnorm + q_b_proj GEMM + bmm(W_UK_T) +
//                    fused_mla_decode_q_concat_kv_cache_insert (fp8 query + fp8 cache insert)
//   k3mla.attn_out = tokenspeed_mla_decode (split_kv + reduction) + bmm(W_UV) + attn*sigmoid(gate)
// Both kernels use 16-CTA thread-block clusters and are PDL-launched: everything that only depends
// on static weights / old KV entries is prefetched into smem before griddepcontrol.wait.
//
// Revision (agents/oproj, 2026-09-25) -- same ops / signatures / rounding points, lower latency:
//  * q_prep: the q_b weight slice (36 KB, contiguous) is one TMA bulk copy; the q_nope rows are
//    pushed to all 16 CTAs with st.async + mbarrier (one exchange) instead of cluster.sync +
//    DSMEM gather + cluster.sync.
//  * attn_out: for B <= 4, griddepcontrol.launch_dependents right after griddepcontrol.wait (was at
//    the very end), so the o_proj GEMV (k3oproj) launches early and prefetches its 11 MB weight
//    while the attention runs (B > 4: still at the end, see K3MLA_TRIGGER_POS). The first KV tile
//    is TMA bulk copies (one per 576-byte row, completing on an mbarrier) and no longer shares a
//    cp.async group with the W_UV prefetch; only the 16-token k-steps that hold tokens are computed. The (m, l, O) combine and W_UV are two exchange rounds (TMA bulk
//    DSMEM copy of one 816-byte message per destination; st.async for the W_UV partials) instead of
//    3 cluster.sync + 3 DSMEM pull rounds: CTA q finalizes latent dims [32q, 32q+32) of all heads,
//    computes the W_UV partial sums of all 768 outputs over those dims and pushes them to the owner
//    of each output. Output = bf16(bf16(latent) @ W_UV) * sigmoid(gate), same rounding points; only
//    the fp32 summation order of W_UV changed (outputs within 2 bf16 ulp of the previous kernels,
//    mqa_q / KV cache bit-identical).
//  * %globaltimer probes (ts != nullptr) are staged in shared memory (a global store per probe
//    distorted the timing by up to ~1 us per phase).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>
#include <mutex>
#include <unordered_set>

namespace {

constexpr int kH = 6, kQL = 1536, kKVL = 512, kRope = 64, kNope = 128, kV = 128;
constexpr int kD = kKVL + kRope;          // 576 latent+rope per cache row
constexpr int kQKV = kQL + kKVL + kRope;  // 2112
constexpr int kQHead = kNope + kRope;     // 192 q rows per head
constexpr int kCl = 16;                   // cluster size (both kernels)
constexpr int kThreads = 256;
constexpr int kMaxB = 8;
constexpr int kMaxTilePages = 8;  // pages touched by one 64-token tile (page size >= 16)

__device__ __forceinline__ long long gtimer() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// optional phase timestamps (thread 0 of every CTA), for timeline debugging
// (staged in shared memory and written at the end: a global store per probe distorts the timing)
#define TS(i) \
  if (ts != nullptr && threadIdx.x == 0) s_ts[(i)] = gtimer();
#define TS_FLUSH()                                                                               \
  if (ts != nullptr && threadIdx.x == 0) {                                                       \
    for (int k_ = 0; k_ < 8; ++k_) ts[(blockIdx.y * gridDim.x + blockIdx.x) * 8 + k_] = s_ts[k_]; \
  }

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("{ .reg .b64 st; mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1; }" ::"r"(smem_u32(bar)),
               "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred P1;\n\t"
      "LAB_WAIT:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
      "@!P1 bra LAB_WAIT;\n\t}" ::"r"(smem_u32(bar)), "r"(parity) : "memory");
}
__device__ __forceinline__ void fence_mbar_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_wait() {
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ uint32_t mapa(const void* local, int rank) {
  uint32_t ra;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(smem_u32(local)), "r"(rank));
  return ra;
}
// Asynchronous DSMEM stores into CTA `rank`'s copy of `local`, completing their bytes on that CTA's
// copy of mbarrier `bar` (no cluster barrier / fence needed on the critical path).
__device__ __forceinline__ void st_async_f32(const void* local, const uint64_t* bar, int rank, float v) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(mapa(local, rank)),
               "r"(__float_as_uint(v)), "r"(mapa(bar, rank)) : "memory");
}
__device__ __forceinline__ void st_async_v4(const void* local, const uint64_t* bar, int rank, float a, float b,
                                            float c, float d) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(
                   mapa(local, rank)),
               "r"(__float_as_uint(a)), "r"(__float_as_uint(b)), "r"(__float_as_uint(c)), "r"(__float_as_uint(d)),
               "r"(mapa(bar, rank)) : "memory");
}
// TMA 1D bulk copy global -> own smem, completing on mbarrier `bar`.
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                   smem_u32(dst)),
               "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
// TMA bulk copy own smem -> CTA `rank`'s smem (same offset as `dst_local`), completing on that CTA's
// copy of mbarrier `bar`.
__device__ __forceinline__ void bulk_s2cluster(const void* dst_local, const void* src, uint32_t bytes, int rank,
                                               const uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                   mapa(dst_local, rank)),
               "r"(smem_u32(src)), "r"(bytes), "r"(mapa(bar, rank)) : "memory");
}
__device__ __forceinline__ void fence_proxy_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ uint4 ldcg16(const void* p) {
  uint4 u;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(u.x), "=r"(u.y), "=r"(u.z), "=r"(u.w) : "l"(p));
  return u;
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src)
               : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ uint8_t to_fp8(float x) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ void bf16x8_to_f32(const uint4& u, float* f) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 t = __bfloat1622float2(p[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}

// ======================================================================================
// Kernel A: q_prep.  grid (16, 6) = one 16-CTA cluster per head, 256 threads, B = #tokens.
//   CTA (r, h): q rows h*192 + [12r, 12r+12) (q_b GEMV), then ql_nope[:, h, 32r : 32r+32]
//   (W_UK absorb, needs all 128 q_nope rows of the head: every CTA pushes its q_nope rows to all
//   16 CTAs with st.async), fp8 quant, and (h == 0, r < B) the cache insert of token r.
// ======================================================================================
constexpr int kRowsA = kQHead / kCl;  // 12
constexpr int kLA = kKVL / kCl;       // 32

struct SmemA {
  __nv_bfloat16 wqb[kRowsA][kQL];      // 36 KB  (TMA bulk copy, pre-wait)
  __nv_bfloat16 wuk[kNope][kLA];       // 8 KB   (prefetched pre-wait)
  __nv_bfloat16 wqa[kQL];              // 3 KB
  __nv_bfloat16 xq[kMaxB][kQL];        // 24 KB  normalized q_c (bf16)
  float qrow[kRowsA][kMaxB];           // this CTA's q rows (bf16-rounded values)
  float qn[kMaxB][kNope];              // q_nope of the head (pushed by the owning CTAs)
  float red[8][kMaxB][kLA];            // W_UK partial sums
  float ss[8][kMaxB];                  // rmsnorm partial sums
  alignas(8) uint64_t barw;            // W_qb bulk copy
  alignas(8) uint64_t barq;            // q_nope exchange
};

template <int B>
__global__ void __launch_bounds__(kThreads, 1)
    q_prep_kernel(const __nv_bfloat16* __restrict__ qkv, int qkv_stride,
                  const __nv_bfloat16* __restrict__ w_qa, const __nv_bfloat16* __restrict__ w_kva,
                  const __nv_bfloat16* __restrict__ W_qb, const __nv_bfloat16* __restrict__ W_UK_T,
                  long long uk_sh, long long uk_sp, uint8_t* __restrict__ cache, int page_size, long long page_stride,
                  const int64_t* __restrict__ slot,
                  const float* __restrict__ q_scale_inv, const float* __restrict__ k_scale_inv,
                  float eps, uint8_t* __restrict__ mqa_q, long long* __restrict__ ts) {
  extern __shared__ __align__(128) uint8_t smem_raw[];
  SmemA& s = *reinterpret_cast<SmemA*>(smem_raw);
  __shared__ long long s_ts[8];
  TS(0)
  const int r = blockIdx.x;  // == %cluster_ctarank for cluster dims (16, 1, 1)
  const int h = blockIdx.y;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

  // ---- pre-wait: mbarriers, static weights -> smem ----
  if (tid == 0) {
    mbar_init(&s.barw, 1);
    mbar_init(&s.barq, 1);
    fence_mbar_init();
    mbar_expect_tx(&s.barw, kRowsA * kQL * 2);
    // this CTA's 12 q_b rows are one contiguous 36 KB range
    bulk_g2s(&s.wqb[0][0], W_qb + (static_cast<size_t>(h) * kQHead + kRowsA * r) * kQL, kRowsA * kQL * 2, &s.barw);
    mbar_expect_tx(&s.barq, kNope * B * 4);  // q_nope rows x B tokens, from CTAs 0..10
  }
  cluster_arrive_relaxed();  // "running, mbarriers initialized" (waited on before the first push)
  {
    for (int c = tid; c < kNope * kLA / 8; c += kThreads) {
      const int p = c / (kLA / 8), j = c % (kLA / 8);
      cp_async16(&s.wuk[p][j * 8], W_UK_T + h * uk_sh + p * uk_sp + kLA * r + j * 8);
    }
    for (int c = tid; c < kQL / 8; c += kThreads) cp_async16(&s.wqa[c * 8], w_qa + c * 8);
    cp_async_commit();
  }
  TS(1)
  pdl_wait();
  pdl_trigger();  // let the attention kernel start prefetching its KV
  TS(2)
  const float qsi = *q_scale_inv;

  // ---- rmsnorm(q_c) for all B tokens (each CTA recomputes; 1536 x B) ----
  {
    float ssq[B];
    float xv[B][8];
#pragma unroll
    for (int b = 0; b < B; ++b) ssq[b] = 0.f;
    if (tid < kQL / 8) {  // 192 chunks of 8
#pragma unroll
      for (int b = 0; b < B; ++b) {
        const uint4 u = *reinterpret_cast<const uint4*>(qkv + static_cast<size_t>(b) * qkv_stride + tid * 8);
        bf16x8_to_f32(u, xv[b]);
#pragma unroll
        for (int i = 0; i < 8; ++i) ssq[b] += xv[b][i] * xv[b][i];
      }
    }
#pragma unroll
    for (int b = 0; b < B; ++b) {
      const float v = warp_sum(ssq[b]);
      if (lane == 0) s.ss[warp][b] = v;
    }
    cp_async_wait_all();  // wqa, wuk
    __syncthreads();
    if (tid < kQL / 8) {
      float w[8];
      bf16x8_to_f32(*reinterpret_cast<const uint4*>(&s.wqa[tid * 8]), w);
#pragma unroll
      for (int b = 0; b < B; ++b) {
        float tot = 0.f;
#pragma unroll
        for (int ww = 0; ww < 8; ++ww) tot += s.ss[ww][b];
        const float rr = rsqrtf(tot / kQL + eps);
#pragma unroll
        for (int i = 0; i < 8; ++i) s.xq[b][tid * 8 + i] = __float2bfloat16(xv[b][i] * rr * w[i]);
      }
    }
    __syncthreads();
  }
  TS(3)

  // ---- q_b GEMV: 12 rows x B tokens; warp w handles rows w and w + 8 (< 12) ----
  mbar_wait(&s.barw, 0);
  for (int row = warp; row < kRowsA; row += 8) {
    float acc[B][2];
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b][0] = acc[b][1] = 0.f;
#pragma unroll
    for (int c = 0; c < kQL / 8 / 32; ++c) {  // 6 chunks of 8 per lane
      const int k = (c * 32 + lane) * 8;
      float w[8];
      bf16x8_to_f32(*reinterpret_cast<const uint4*>(&s.wqb[row][k]), w);
#pragma unroll
      for (int b = 0; b < B; ++b) {
        float x[8];
        bf16x8_to_f32(*reinterpret_cast<const uint4*>(&s.xq[b][k]), x);
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[b][i & 1] += w[i] * x[i];
      }
    }
#pragma unroll
    for (int b = 0; b < B; ++b) {
      const float v = warp_sum(acc[b][0] + acc[b][1]);
      if (lane == 0) s.qrow[row][b] = bf16r(v);
    }
  }
  __syncthreads();
  TS(4)

  // ---- q_nope rows (p < 128) -> all 16 CTAs of the head (st.async); q_pe rows -> fp8 query ----
  cluster_wait();  // every CTA of the cluster is running with initialized mbarriers
  {
    const int nrows = min(kRowsA, max(0, kNope - kRowsA * r));  // this CTA's q_nope rows
    for (int i = tid; i < nrows * B * kCl; i += kThreads) {
      const int dst = i % kCl, jb = i / kCl, j = jb % nrows, b = jb / nrows;
      st_async_f32(&s.qn[b][kRowsA * r + j], &s.barq, dst, s.qrow[j][b]);
    }
  }
  for (int i = tid; i < kRowsA * B; i += kThreads) {
    const int j = i % kRowsA, b = i / kRowsA;
    const int p = kRowsA * r + j;
    if (p >= kNope)
      mqa_q[(static_cast<size_t>(b) * kH + h) * kD + kKVL + (p - kNope)] = to_fp8(s.qrow[j][b] * qsi);
  }
  mbar_wait(&s.barq, 0);
  TS(5)

  // ---- W_UK absorb: ql_nope[b][h][32r + l] = sum_p qn[b][p] * W_UK_T[h][p][32r + l] ----
  {
    const int l = tid & 31, pg = tid >> 5;  // 8 groups x 16 p
    float acc[B][2];
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b][0] = acc[b][1] = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int p = pg * 16 + i;
      const float w = __bfloat162float(s.wuk[p][l]);
#pragma unroll
      for (int b = 0; b < B; ++b) acc[b][i & 1] += s.qn[b][p] * w;
    }
#pragma unroll
    for (int b = 0; b < B; ++b) s.red[pg][b][l] = acc[b][0] + acc[b][1];
    __syncthreads();
    if (tid < kLA * B) {
      const int b = tid / kLA, ll = tid % kLA;
      float v = 0.f;
#pragma unroll
      for (int gg = 0; gg < 8; ++gg) v += s.red[gg][b][ll];
      mqa_q[(static_cast<size_t>(b) * kH + h) * kD + kLA * r + ll] = to_fp8(bf16r(v) * qsi);
    }
  }
  TS(6)

  // ---- cache insert of token r (head-0 cluster): fp8([rmsnorm(kv_c) | k_pe] * k_scale_inv) ----
  // Padded rows (CUDA-graph padding) carry slot = PAD_SLOT_ID (-1): no write.
  const long long sl = (h == 0 && r < B) ? slot[r] : -1;
  if (sl >= 0) {
    const int b = r;
    const float ksi = *k_scale_inv;
    const __nv_bfloat16* row = qkv + static_cast<size_t>(b) * qkv_stride;
    uint8_t* dst = cache + (sl / page_size) * page_stride + (sl % page_size) * kD;
    float x[8];
    float ssq = 0.f;
    if (tid < kKVL / 8) {  // 64 threads x 8 values
      bf16x8_to_f32(*reinterpret_cast<const uint4*>(row + kQL + tid * 8), x);
#pragma unroll
      for (int i = 0; i < 8; ++i) ssq += x[i] * x[i];
    }
    ssq = warp_sum(ssq);
    __syncthreads();
    if (lane == 0 && warp < 2) s.ss[warp][0] = ssq;
    __syncthreads();
    if (tid < kKVL / 8) {
      const float rr = rsqrtf((s.ss[0][0] + s.ss[1][0]) / kKVL + eps);
      float w[8];
      bf16x8_to_f32(*reinterpret_cast<const uint4*>(w_kva + tid * 8), w);
      uint32_t lo = 0, hi = 0;
#pragma unroll
      for (int i = 0; i < 4; ++i) lo |= static_cast<uint32_t>(to_fp8(bf16r(x[i] * rr * w[i]) * ksi)) << (8 * i);
#pragma unroll
      for (int i = 0; i < 4; ++i) hi |= static_cast<uint32_t>(to_fp8(bf16r(x[4 + i] * rr * w[4 + i]) * ksi)) << (8 * i);
      *reinterpret_cast<uint2*>(dst + tid * 8) = make_uint2(lo, hi);
    } else if (tid >= 64 && tid < 64 + kRope / 8) {
      const int c = tid - 64;
      float e[8];
      bf16x8_to_f32(*reinterpret_cast<const uint4*>(row + kQL + kKVL + c * 8), e);
      uint32_t lo = 0, hi = 0;
#pragma unroll
      for (int i = 0; i < 4; ++i) lo |= static_cast<uint32_t>(to_fp8(e[i] * ksi)) << (8 * i);
#pragma unroll
      for (int i = 0; i < 4; ++i) hi |= static_cast<uint32_t>(to_fp8(e[4 + i] * ksi)) << (8 * i);
      *reinterpret_cast<uint2*>(dst + kKVL + c * 8) = make_uint2(lo, hi);
    }
  }
  TS(7)
  TS_FLUSH()
}

// ======================================================================================
// Kernel B: attention + W_UV + gate.  grid (16, B): one 16-CTA cluster per request.
//   CTA r: KV tokens [r*per, (r+1)*per) of its request (per = ceil(L/16)), 64-token tiles in
//   smem (fp8 rows, stride 592 B; one TMA bulk copy per 576-byte row). QK^T: mma m16n8k32 e4m3
//   (rows = 6 heads, cols = 8 tokens per warp). Online softmax over tiles. PV: mma m16n8k16 f16
//   (P f16 x V fp8->f16), warp w owns latent dims [64w, 64w+64).
//   Round 1 (one TMA bulk DSMEM copy per destination): CTA r sends CTA q its unnormalized
//   O[h][32q .. 32q+32) of all 6 heads and its (m, l); CTA q finalizes latent[h][32q .. 32q+32)
//   (bf16). Round 2 (st.async): CTA q computes the W_UV partial sums of all 768 outputs over its 32
//   latent dims and sends each output owner (CTA o/48) its 48 partials; the owner sums the 16 partials
//   in rank order, rounds to bf16 and applies the sigmoid gate.
//   ~102 KB smem -> 2 CTAs/SM, so 14 clusters fit (B = 8 in one wave).
// PDL: griddepcontrol.launch_dependents right after griddepcontrol.wait for B <= 4, at the end for
// B > 4 (K3MLA_TRIGGER_POS); the dependent o_proj kernel waits for this grid's completion before
// reading the output either way.
// ======================================================================================
// attn_out's griddepcontrol.launch_dependents: 0 = right after griddepcontrol.wait, 1 = after the KV
// tiles, 2 = at the end (previous behaviour), 3 = after the wait for B <= 4, at the end for B > 4.
// Measured (single-GPU CUDA-graph MLA layer chain, dependent = k3oproj, L = 512): early is faster
// by 1.2 us at B = 1 and by 0.5 / 1.1 us at B = 2 / 4 when o_proj runs as produce + consume (the
// producer prefetches its 11 MB weight during the attention); with the fused o_proj kernel as the
// dependent (B >= 2 by default) early vs late is within +-0.6 us, and at B = 8 early is ~0.3 us
// slower (112 full-SM dependent CTAs compete with 128 attention CTAs).
#ifndef K3MLA_TRIGGER_POS
#define K3MLA_TRIGGER_POS 3
#endif
constexpr int kTile = 64;
constexpr int kRowB = kD + 16;              // 592 B smem row stride
constexpr int kNOut = kH * kV;              // 768 outputs per token
constexpr int kOutPerCta = kNOut / kCl;     // 48 outputs finalized per CTA
constexpr int kLatPerCta = kKVL / kCl;      // 32 latent dims finalized per CTA
constexpr int kX1 = kH * kLatPerCta + 2 * kH;  // 204 floats per round-1 message (O slice + (m, l))
static_assert((kX1 * 4) % 16 == 0 && (kOutPerCta * 4) % 16 == 0, "bulk copy sizes");
static_assert(kCl * kX1 * 4 <= kTile * kRowB, "round-1 staging must fit in the KV tile");

struct SmemB {
  alignas(128) uint8_t kv[kTile][kRowB];       // 37.9 KB KV tile (reused as the round 1/2 staging buffers)
  alignas(128) __nv_bfloat16 wuv[kNOut][kLatPerCta];  // 48 KB W_UV[h][32q + d][v] at row h*128+v
  float S[kH][kTile];                          // scores (log2 domain)
  __half P[8][kTile];                          // probabilities (rows 6,7 zero)
  float alpha[8];
  float m[kH], l[kH];
  alignas(16) float rX[kCl][kX1];              // 12.8 KB received round-1 messages
  alignas(16) float lat[kH][kLatPerCta];       // this CTA's finalized latent slice (bf16 values)
  alignas(16) float rP[kCl][kOutPerCta];       // 3 KB    received W_UV partial sums (round 2)
  int pages[kMaxTilePages];                    // block-table entries of the current tile (tiles >= 1)
  alignas(8) uint64_t bar[3];                  // [0] round 1, [1] round 2, [2] KV tile 0
};

__device__ __forceinline__ void mma_fp8(float (&c)[4], uint32_t a0, uint32_t a2, uint32_t b0,
                                        uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f16(float (&c)[4], uint32_t a0, uint32_t a2, uint32_t b0,
                                        uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t fp8x2_to_f16x2(uint32_t two_bytes) {
  __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(two_bytes & 0xffffu), __NV_E4M3);
  return static_cast<uint32_t>(hr.x) | (static_cast<uint32_t>(hr.y) << 16);
}

__global__ void __launch_bounds__(kThreads, 2)  // <= 128 regs -> 2 CTAs/SM (14 clusters resident)
    attn_out_kernel(const uint8_t* __restrict__ mqa_q, const uint8_t* __restrict__ cache,
                    int page_size, long long page_stride,
                    const int* __restrict__ block_table, int bt_stride,
                    const int* __restrict__ seq_lens, const __nv_bfloat16* __restrict__ W_UV,
                    long long uv_sh, long long uv_sv,
                    const __nv_bfloat16* __restrict__ gate, int gate_stride, float scale_log2,
                    float output_scale, __nv_bfloat16* __restrict__ out, long long* __restrict__ ts) {
  extern __shared__ __align__(128) uint8_t smem_raw[];
  SmemB& s = *reinterpret_cast<SmemB*>(smem_raw);
  __shared__ long long s_ts[8];
  TS(0)
  const int r = blockIdx.x;  // == %cluster_ctarank for cluster dims (16, 1, 1)
  const int b = blockIdx.y;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t4 = lane & 3;

  const int L = max(seq_lens[b], 0);  // padded rows have seq_len 0 -> no KV work, zero output
  const int* btrow = block_table + static_cast<size_t>(b) * bt_stride;
  if (tid == 0) {
    for (int i = 0; i < 3; ++i) mbar_init(&s.bar[i], 1);
    fence_mbar_init();
    mbar_expect_tx(&s.bar[0], kCl * kX1 * 4);
    mbar_expect_tx(&s.bar[1], kCl * kOutPerCta * 4);

  }
  if (tid < kH) {
    s.m[tid] = -INFINITY;
    s.l[tid] = 0.f;
  }
  if (tid < 8) s.alpha[tid] = 0.f;
  if (tid < 2 * kTile) s.P[6 + tid / kTile][tid % kTile] = __float2half(0.f);
  const int per = (L + kCl - 1) / kCl;
  const int t0 = min(L, r * per), t1 = min(L, t0 + per);
  const int ntok = t1 - t0;
  const int ntiles = (ntok + kTile - 1) / kTile;
  __syncthreads();  // mbarriers initialized

  // KV tile 0 -> smem (pre-wait): warp 0 lane i issues one 576-byte TMA bulk copy per token row
  // i, i + 32 (the newest token of the request is skipped: q_prep inserts it; it is loaded after the
  // wait). Tiles >= 1 (> 1024 tokens) are loaded after the previous one with cp.async by all threads.
  // Rows [n, roundup16(n)) are zeroed (PV reads whole 16-row k-steps; stale fp8 NaN * 0 = NaN).
  auto zero_rows = [&](int n) {
    const int nz = ((n + 15) & ~15) - n;
    for (int c = tid; c < nz * (kRowB / 16); c += kThreads)
      *reinterpret_cast<uint4*>(&s.kv[n + c / (kRowB / 16)][(c % (kRowB / 16)) * 16]) = make_uint4(0, 0, 0, 0);
  };
  auto issue_tile0 = [&]() {
    const int n = min(kTile, t1 - t0);
    if (warp == 0) {
      const bool has_new = L - 1 >= t0 && L - 1 < t0 + n;  // skipped here, loaded after the wait
      if (lane == 0) mbar_expect_tx(&s.bar[2], kD * (n - (has_new ? 1 : 0)));  // before any copy
      __syncwarp();
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const int i = lane + 32 * k, tok = t0 + i;
        if (i < n && tok != L - 1) {
          const long long pg = btrow[tok / page_size];
          bulk_g2s(&s.kv[i][0], cache + pg * page_stride + static_cast<long long>(tok % page_size) * kD, kD,
                   &s.bar[2]);
        }
      }
    }
    zero_rows(n);
  };
  auto load_tile_sync = [&](int tile) {  // tiles >= 1: all threads, cp.async (as the previous kernel)
    const int base = t0 + tile * kTile;
    const int n = min(kTile, t1 - base);
    const int tpg0 = base / page_size;
    const int np = (base + n - 1) / page_size - tpg0 + 1;
    if (tid < np) s.pages[tid] = btrow[tpg0 + tid];
    __syncthreads();
    for (int c = tid; c < n * (kD / 16); c += kThreads) {
      const int t = c / (kD / 16), j = c % (kD / 16), tok = base + t;
      cp_async16(&s.kv[t][j * 16], cache + static_cast<long long>(s.pages[tok / page_size - tpg0]) * page_stride +
                                       static_cast<long long>(tok % page_size) * kD + j * 16);
    }
    zero_rows(n);
    cp_async_commit();
    cp_async_wait_all();
  };
  // ---- pre-wait prefetch: old KV rows of the first tile (TMA), then this CTA's W_UV slice
  // W_UV[:, 32r .. 32r+32, :] (768 rows of 64 contiguous bytes, only needed after the attention) ----
  if (ntiles > 0) issue_tile0();
  // (cp.async, 3072 x 16 B: measured faster than 768 64-byte TMA bulk copies, whose issue is slow)
  for (int c = tid; c < kNOut * (kLatPerCta / 8); c += kThreads) {
    const int o = c >> 2, ch = c & 3, hh = o / kV, v = o % kV;
    cp_async16(&s.wuv[o][8 * ch], W_UV + hh * uv_sh + v * uv_sv + kLatPerCta * r + 8 * ch);
  }
  cp_async_commit();
  cluster_arrive_relaxed();  // "running, mbarriers initialized" (waited on before the first remote copy)
  TS(1)
  pdl_wait();
#if K3MLA_TRIGGER_POS == 0
  pdl_trigger();  // the o_proj GEMV may launch now and prefetch its weights while we run
#elif K3MLA_TRIGGER_POS == 3
  if (gridDim.y <= 4) pdl_trigger();  // B <= 4: see above
#endif
  TS(2)
  // Post-wait global loads, all independent (one L2 round trip): Q fragments (mqa_q, written by
  // q_prep), the gate values of the outputs this CTA finalizes, the newest KV row (q_prep's insert).
  // Q fragments: lane (g, t4): head g (< 6), bytes [32kk + 8t4, +8)
  uint32_t qa0[kD / 32], qa2[kD / 32];
  {
    const uint8_t* qrow = mqa_q + (static_cast<size_t>(b) * kH + min(g, kH - 1)) * kD + 8 * t4;
#pragma unroll
    for (int kk = 0; kk < kD / 32; ++kk) {
      const uint2 v = *reinterpret_cast<const uint2*>(qrow + 32 * kk);
      qa0[kk] = g < kH ? v.x : 0u;
      qa2[kk] = g < kH ? v.y : 0u;
    }
  }
  float gv = 0.f;
  if (tid < kOutPerCta) gv = __bfloat162float(__ldcg(gate + static_cast<size_t>(b) * gate_stride + kOutPerCta * r + tid));
  if (L - 1 >= t0 && L - 1 < min(t1, t0 + kTile) && tid < kD / 16) {
    const int tok = L - 1;
    const long long pg = btrow[tok / page_size];
    *reinterpret_cast<uint4*>(&s.kv[tok - t0][tid * 16]) =
        ldcg16(cache + pg * page_stride + static_cast<long long>(tok % page_size) * kD + tid * 16);
  }

  float oacc[8][4];
#pragma unroll
  for (int j = 0; j < 8; ++j) oacc[j][0] = oacc[j][1] = oacc[j][2] = oacc[j][3] = 0.f;

  for (int tile = 0; tile < ntiles; ++tile) {
    if (tile == 0) mbar_wait(&s.bar[2], 0);
    else load_tile_sync(tile);  // long context (> 1024 tokens): synchronous tiles
    __syncthreads();  // newest row / zeroed rows / tile visible
    const int n = min(kTile, ntok - tile * kTile);
    // ---- S = Q K^T for tokens [8w, 8w+8) of the tile ----
    {
      float c[4] = {0.f, 0.f, 0.f, 0.f}, c2[4] = {0.f, 0.f, 0.f, 0.f};
      const uint8_t* krow = &s.kv[8 * warp + g][8 * t4];
      if (8 * warp < n) {
#pragma unroll
        for (int kk = 0; kk < kD / 32; kk += 2) {
          const uint2 kv = *reinterpret_cast<const uint2*>(krow + 32 * kk);
          const uint2 kv2 = *reinterpret_cast<const uint2*>(krow + 32 * kk + 32);
          mma_fp8(c, qa0[kk], qa2[kk], kv.x, kv.y);
          mma_fp8(c2, qa0[kk + 1], qa2[kk + 1], kv2.x, kv2.y);
        }
      }
      if (g < kH) {
        const int tk = 8 * warp + 2 * t4;
        s.S[g][tk] = tk < n ? (c[0] + c2[0]) * scale_log2 : -INFINITY;
        s.S[g][tk + 1] = tk + 1 < n ? (c[1] + c2[1]) * scale_log2 : -INFINITY;
      }
    }
    __syncthreads();
    // ---- online softmax: warp h owns head h ----
    if (warp < kH) {
      const float x0 = s.S[warp][lane], x1 = s.S[warp][lane + 32];
      const float mt = warp_max(fmaxf(x0, x1));
      const float mo = s.m[warp];
      const float mn = fmaxf(mo, mt);
      const float p0 = exp2f(x0 - mn), p1 = exp2f(x1 - mn);
      const float sum = warp_sum(p0 + p1);
      s.P[warp][lane] = __float2half(p0);
      s.P[warp][lane + 32] = __float2half(p1);
      if (lane == 0) {
        const float a = exp2f(mo - mn);  // 0 on the first tile (mo = -inf)
        s.alpha[warp] = a;
        s.l[warp] = s.l[warp] * a + sum;
        s.m[warp] = mn;
      }
    }
    __syncthreads();
    // ---- O = alpha * O + P V for dims [64w, 64w+64) (only the 16-token k-steps that hold tokens) ----
    {
      const float al = s.alpha[g];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        oacc[j][0] *= al;
        oacc[j][1] *= al;
      }
      const int nks = (n + 15) / 16;
#pragma unroll
      for (int ks = 0; ks < kTile / 16; ++ks) {
        if (ks < nks) {
          const int ta = 16 * ks + 2 * t4;
          const uint32_t pa0 = *reinterpret_cast<const uint32_t*>(&s.P[g][ta]);
          const uint32_t pa2 = *reinterpret_cast<const uint32_t*>(&s.P[g][ta + 8]);
          const uint8_t* vb = &s.kv[0][64 * warp + 8 * g];
          const uint2 vA = *reinterpret_cast<const uint2*>(vb + (ta) * kRowB);
          const uint2 vB = *reinterpret_cast<const uint2*>(vb + (ta + 1) * kRowB);
          const uint2 vC = *reinterpret_cast<const uint2*>(vb + (ta + 8) * kRowB);
          const uint2 vD = *reinterpret_cast<const uint2*>(vb + (ta + 9) * kRowB);
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            const uint32_t a = half ? vA.y : vA.x, bb = half ? vB.y : vB.x;
            const uint32_t cc = half ? vC.y : vC.x, d = half ? vD.y : vD.x;
#pragma unroll
            for (int q = 0; q < 2; ++q) {  // bytes (a_j, b_j, a_{j+1}, b_{j+1}), j = 2q
              const uint32_t sel = (2 * q) | ((4 + 2 * q) << 4) | ((2 * q + 1) << 8) | ((5 + 2 * q) << 12);
              const uint32_t ab = __byte_perm(a, bb, sel);
              const uint32_t cd = __byte_perm(cc, d, sel);
              const int j = 4 * half + 2 * q;
              mma_f16(oacc[j], pa0, pa2, fp8x2_to_f16x2(ab), fp8x2_to_f16x2(cd));
              mma_f16(oacc[j + 1], pa0, pa2, fp8x2_to_f16x2(ab >> 16), fp8x2_to_f16x2(cd >> 16));
            }
          }
        }
      }
    }
    __syncthreads();  // tile / P / S reuse; also publishes s.m / s.l of the last tile
  }
  __syncthreads();  // s.m / s.l final (also when this CTA had no tokens); KV tile buffer free
#if K3MLA_TRIGGER_POS == 1
  pdl_trigger();
#endif
  TS(3)

  // ---- round 1: stage [q][h*32 + d] = O slices, [q][192 + 2h + {0,1}] = (m, l); one bulk copy per q ----
  // Lane (g < 6, t4) of warp w holds head g, dims [64w + 16t4, +16) = slice q = 2w + t4/2,
  // offset 16*(t4&1): oacc[j][0] -> dim +j, oacc[j][1] -> dim +8+j.
  float* stage1 = reinterpret_cast<float*>(&s.kv[0][0]);             // [kCl][kX1]
  if (g < kH) {
    float* dst = stage1 + (2 * warp + (t4 >> 1)) * kX1 + g * kLatPerCta + 16 * (t4 & 1);
    *reinterpret_cast<float4*>(dst) = make_float4(oacc[0][0], oacc[1][0], oacc[2][0], oacc[3][0]);
    *reinterpret_cast<float4*>(dst + 4) = make_float4(oacc[4][0], oacc[5][0], oacc[6][0], oacc[7][0]);
    *reinterpret_cast<float4*>(dst + 8) = make_float4(oacc[0][1], oacc[1][1], oacc[2][1], oacc[3][1]);
    *reinterpret_cast<float4*>(dst + 12) = make_float4(oacc[4][1], oacc[5][1], oacc[6][1], oacc[7][1]);
  }
  if (tid < kH * kCl) {
    const int hh = tid % kH, q = tid / kH;
    *reinterpret_cast<float2*>(stage1 + q * kX1 + kH * kLatPerCta + 2 * hh) = make_float2(s.m[hh], s.l[hh]);
  }
  fence_proxy_async_smem();  // staging writes -> async proxy (bulk copies)
  __syncthreads();
  cluster_wait();  // every CTA of the cluster is running with initialized mbarriers
  if (tid < kCl) bulk_s2cluster(&s.rX[r][0], stage1 + tid * kX1, kX1 * 4, tid, &s.bar[0]);
  mbar_wait(&s.bar[0], 0);
  TS(4)
  if (tid < kH * kLatPerCta) {  // latent[h][32r + d] = bf16(sum_q w_q O_q / sum_q w_q l_q * output_scale)
    const int hh = tid / kLatPerCta, d = tid % kLatPerCta;
    float M = -INFINITY;
#pragma unroll
    for (int rr = 0; rr < kCl; ++rr) M = fmaxf(M, s.rX[rr][kH * kLatPerCta + 2 * hh]);
    float Ls = 0.f, o = 0.f;
#pragma unroll
    for (int rr = 0; rr < kCl; ++rr) {
      const float mr = s.rX[rr][kH * kLatPerCta + 2 * hh];
      const float w = mr == -INFINITY ? 0.f : exp2f(mr - M);
      Ls += w * s.rX[rr][kH * kLatPerCta + 2 * hh + 1];
      o += s.rX[rr][hh * kLatPerCta + d] * w;
    }
    const float Lsum = Ls > 0.f ? output_scale / Ls : 0.f;  // seq_len 0 (padding) -> zeros
    s.lat[hh][d] = bf16r(o * Lsum);
  }
  cp_async_wait_all();  // W_UV slice
  __syncthreads();

  // ---- round 2: W_UV partial sums over dims [32r, 32r+32); thread t < 192 computes outputs
  // 4t .. 4t+3 (same head, same owner 4t/48) and pushes them with one 16-byte st.async (register
  // source: no smem-lifetime constraint on the sender). Chunk order rotated by t: 2-way instead of
  // 8-way smem bank conflicts on the 64-byte W_UV rows.
  if (tid < kNOut / 4) {
    const int o0 = 4 * tid, hh = o0 / kV;
    const float* lt = s.lat[hh];
    float acc[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float a0 = 0.f, a1 = 0.f;
#pragma unroll
      for (int c4 = 0; c4 < 4; ++c4) {
        const int ch = (c4 + tid) & 3;
        float w[8];
        bf16x8_to_f32(*reinterpret_cast<const uint4*>(&s.wuv[o0 + i][8 * ch]), w);
        const float4 x0 = *reinterpret_cast<const float4*>(lt + 8 * ch);
        const float4 x1 = *reinterpret_cast<const float4*>(lt + 8 * ch + 4);
        a0 += x0.x * w[0] + x0.y * w[1] + x0.z * w[2] + x0.w * w[3];
        a1 += x1.x * w[4] + x1.y * w[5] + x1.z * w[6] + x1.w * w[7];
      }
      acc[i] = a0 + a1;
    }
    st_async_v4(&s.rP[r][o0 % kOutPerCta], &s.bar[1], o0 / kOutPerCta, acc[0], acc[1], acc[2], acc[3]);
  }
  TS(5)
  if (tid < kOutPerCta) {
    mbar_wait(&s.bar[1], 0);
    float a = 0.f;
#pragma unroll
    for (int rr = 0; rr < kCl; ++rr) a += s.rP[rr][tid];
    a = bf16r(a);
    out[static_cast<size_t>(b) * kNOut + kOutPerCta * r + tid] = __float2bfloat16(a * (1.f / (1.f + __expf(-gv))));
  }
  // (Smem lifetime: this CTA's round-1 bulk copies have been read once every destination q finished
  // round 1; q sends its round-2 partials only after that, and we exit only after receiving all of
  // them, so no outgoing copy can still be reading our staging buffer.)
  TS(6)
#if K3MLA_TRIGGER_POS == 2
  pdl_trigger();
#elif K3MLA_TRIGGER_POS == 3
  if (gridDim.y > 4) pdl_trigger();
#endif
  TS(7)
  TS_FLUSH()
}

template <typename K, typename... Args>
void launch(K kern, dim3 grid, size_t smem, cudaStream_t stream, bool pdl, Args... args) {
  // Per-kernel (not per-signature: all q_prep_kernel<B> share one function-pointer type)
  // one-time attribute setup.
  static std::mutex mu;
  static std::unordered_set<const void*> done;
  {
    std::lock_guard<std::mutex> lock(mu);
    const void* key = reinterpret_cast<const void*>(kern);
    if (!done.count(key)) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
      C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
      done.insert(key);
    }
  }
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = kCl;
  at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = pdl ? 2 : 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, args...));
}

// Paged latent cache [num_pages, page_size, 576] (1-byte fp8/uint8); pages may be padded
// (page stride >= page_size * 576), rows must be dense.
void check_cache(const torch::Tensor& c) {
  TORCH_CHECK(c.dim() == 3 && c.size(2) == kD && c.element_size() == 1, "kv_cache must be [pages, page, 576] fp8");
  TORCH_CHECK(c.stride(2) == 1 && c.stride(1) == kD && c.stride(0) >= c.size(1) * kD, "kv_cache rows must be dense");
  TORCH_CHECK(c.size(1) >= 16 && c.stride(0) % 16 == 0 && c.data_ptr() != nullptr &&
              reinterpret_cast<uintptr_t>(c.data_ptr()) % 16 == 0, "kv_cache page size >= 16, 16B aligned");
}

long long* ts_ptr(const std::optional<torch::Tensor>& ts) {
  return ts ? reinterpret_cast<long long*>(ts->data_ptr<int64_t>()) : nullptr;
}

}  // namespace

void q_prep(torch::Tensor qkv, torch::Tensor w_qa, torch::Tensor w_kva, torch::Tensor W_qb,
            torch::Tensor W_UK_T, torch::Tensor kv_cache, torch::Tensor slot_mapping,
            torch::Tensor q_scale_inv, torch::Tensor k_scale_inv, double eps, torch::Tensor mqa_q,
            int64_t pdl, std::optional<torch::Tensor> ts) {
  const int B = qkv.size(0);
  TORCH_CHECK(B >= 1 && B <= kMaxB, "q_prep: 1 <= B <= 8");
  TORCH_CHECK(qkv.size(1) >= kQKV && qkv.stride(1) == 1 && qkv.scalar_type() == at::kBFloat16);
  TORCH_CHECK(qkv.stride(0) % 8 == 0, "qkv row stride must be a multiple of 8");
  TORCH_CHECK(W_qb.is_contiguous() && W_qb.size(0) == kH * kQHead && W_qb.size(1) == kQL &&
              W_qb.scalar_type() == at::kBFloat16 && reinterpret_cast<uintptr_t>(W_qb.data_ptr()) % 16 == 0);
  // W_UK_T [6, 128, 512] with the latent dim contiguous (vLLM's permuted view of kv_b_proj is fine)
  TORCH_CHECK(W_UK_T.dim() == 3 && W_UK_T.size(0) == kH && W_UK_T.size(1) == kNope && W_UK_T.size(2) == kKVL &&
              W_UK_T.stride(2) == 1 && W_UK_T.stride(1) % 8 == 0 && W_UK_T.stride(0) % 8 == 0 &&
              W_UK_T.scalar_type() == at::kBFloat16 && reinterpret_cast<uintptr_t>(W_UK_T.data_ptr()) % 16 == 0,
              "W_UK_T must be bf16 [6,128,512] with a contiguous last dim");
  TORCH_CHECK(w_qa.is_contiguous() && w_kva.is_contiguous());
  check_cache(kv_cache);
  TORCH_CHECK(mqa_q.is_contiguous() && mqa_q.numel() == B * kH * kD && mqa_q.element_size() == 1);
  TORCH_CHECK(slot_mapping.scalar_type() == at::kLong && slot_mapping.numel() >= B);
  auto bf = [](const torch::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()); };
  cudaStream_t st = c10::cuda::getCurrentCUDAStream();
#define K3_QP(NB)                                                                                   \
  case NB:                                                                                          \
    launch(q_prep_kernel<NB>, dim3(kCl, kH), sizeof(SmemA), st, pdl != 0, bf(qkv),                  \
           (int)qkv.stride(0), bf(w_qa), bf(w_kva), bf(W_qb), bf(W_UK_T),                           \
           (long long)W_UK_T.stride(0), (long long)W_UK_T.stride(1),                                \
           reinterpret_cast<uint8_t*>(kv_cache.data_ptr()), (int)kv_cache.size(1),                  \
           (long long)kv_cache.stride(0), slot_mapping.data_ptr<int64_t>(),                         \
           q_scale_inv.data_ptr<float>(), k_scale_inv.data_ptr<float>(), (float)eps,                \
           reinterpret_cast<uint8_t*>(mqa_q.data_ptr()), ts_ptr(ts));                               \
    break;
  switch (B) { K3_QP(1) K3_QP(2) K3_QP(3) K3_QP(4) K3_QP(5) K3_QP(6) K3_QP(7) K3_QP(8) }
#undef K3_QP
}

void attn_out(torch::Tensor mqa_q, torch::Tensor kv_cache, torch::Tensor block_table,
              torch::Tensor seq_lens, torch::Tensor W_UV, torch::Tensor gate, double softmax_scale,
              double output_scale, torch::Tensor out, int64_t pdl, std::optional<torch::Tensor> ts) {
  const int B = seq_lens.size(0);
  TORCH_CHECK(mqa_q.is_contiguous() && mqa_q.numel() == B * kH * kD && mqa_q.element_size() == 1);
  check_cache(kv_cache);
  TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) >= B);
  TORCH_CHECK(block_table.scalar_type() == at::kInt && (block_table.stride(1) == 1 || block_table.size(1) == 1));
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt);
  // W_UV [6, 512, 128] with the latent dim (dim 1) contiguous -- exactly vLLM's
  // kv_b_proj.weight[...].T view (a fully contiguous [6,512,128] tensor must be re-laid out)
  TORCH_CHECK(W_UV.dim() == 3 && W_UV.size(0) == kH && W_UV.size(1) == kKVL && W_UV.size(2) == kV &&
              W_UV.stride(1) == 1 && W_UV.stride(2) % 8 == 0 && W_UV.stride(0) % 8 == 0 &&
              W_UV.scalar_type() == at::kBFloat16 && reinterpret_cast<uintptr_t>(W_UV.data_ptr()) % 16 == 0,
              "W_UV must be bf16 [6,512,128] with a contiguous latent (dim 1)");
  TORCH_CHECK(gate.stride(1) == 1 && gate.size(1) == kH * kV && out.is_contiguous());
  launch(attn_out_kernel, dim3(kCl, B), sizeof(SmemB), c10::cuda::getCurrentCUDAStream(), pdl != 0,
         reinterpret_cast<const uint8_t*>(mqa_q.data_ptr()),
         reinterpret_cast<const uint8_t*>(kv_cache.data_ptr()), (int)kv_cache.size(1),
         (long long)kv_cache.stride(0), block_table.data_ptr<int>(),
         (int)block_table.stride(0), seq_lens.data_ptr<int>(),
         reinterpret_cast<const __nv_bfloat16*>(W_UV.data_ptr()), (long long)W_UV.stride(0),
         (long long)W_UV.stride(2),
         reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr()), (int)gate.stride(0),
         (float)(softmax_scale * 1.4426950408889634), (float)output_scale,
         reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), ts_ptr(ts));
}

#ifndef K3MLA_NS
#define K3MLA_NS k3mla  // variants (e.g. -DK3MLA_TRIGGER_POS=2 -DK3MLA_NS=k3mla_late) can coexist
#endif
#define K3MLA_LIBRARY(ns, m) TORCH_LIBRARY(ns, m)
#define K3MLA_LIBRARY_IMPL(ns, k, m) TORCH_LIBRARY_IMPL(ns, k, m)

K3MLA_LIBRARY(K3MLA_NS, m) {
  m.def(
      "q_prep(Tensor qkv, Tensor w_qa, Tensor w_kva, Tensor W_qb, Tensor W_UK_T, Tensor(a!) kv_cache, "
      "Tensor slot_mapping, Tensor q_scale_inv, Tensor k_scale_inv, float eps, Tensor(b!) mqa_q, "
      "int pdl, Tensor? ts=None) -> ()");
  m.def(
      "attn_out(Tensor mqa_q, Tensor kv_cache, Tensor block_table, Tensor seq_lens, Tensor W_UV, "
      "Tensor gate, float softmax_scale, float output_scale, Tensor(a!) out, int pdl, "
      "Tensor? ts=None) -> ()");
}
K3MLA_LIBRARY_IMPL(K3MLA_NS, CUDA, m) {
  m.impl("q_prep", &q_prep);
  m.impl("attn_out", &attn_out);
}
