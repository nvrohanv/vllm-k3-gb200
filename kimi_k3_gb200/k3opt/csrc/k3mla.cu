// Fused Kimi K3 MLA decode (TP16 rank: 6 heads, q_lora 1536, kv_lora 512, rope 64, nope 128,
// v 128, fp8 e4m3 paged KV cache with 64-token kernel pages, sigmoid output gate). sm_100a.
//
// Replaces, in MultiHeadLatentAttention decode (vllm/models/kimi_k3/nvidia/mla.py):
//   k3mla.q_prep   = fused_q_kv_rmsnorm + q_b_proj GEMM + bmm(W_UK_T) +
//                    fused_mla_decode_q_concat_kv_cache_insert (fp8 query + fp8 cache insert)
//   k3mla.attn_out = tokenspeed_mla_decode (split_kv + reduction) + bmm(W_UV) + attn*sigmoid(gate)
// Both kernels use 16-CTA thread-block clusters with DSMEM reductions and are PDL-launched:
// everything that only depends on static weights / old KV entries is prefetched into smem before
// griddepcontrol.wait.
//
// Rounding points match vLLM: rmsnorm out bf16, q (q_b out) bf16, ql_nope (bmm out) bf16,
// mqa_q = fp8(x * q_scale_inv), cache = fp8(x * k_scale_inv), latent (attention out) bf16,
// W_UV out bf16, final = bf16(float(attn) * sigmoid(float(gate))).
// Differences vs tokenspeed: P is kept in f16 for the PV MMA; summation orders differ.
#include <cooperative_groups.h>
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

namespace cg = cooperative_groups;

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
#define TS(i) \
  if (ts != nullptr && threadIdx.x == 0) ts[(blockIdx.y * gridDim.x + blockIdx.x) * 8 + (i)] = clock64();

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
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
//   (W_UK absorb, needs all 128 q_nope rows of the head -> DSMEM gather), fp8 quant, and
//   (h == 0, r < B) the cache insert of token r.
// ======================================================================================
constexpr int kRowsA = kQHead / kCl;  // 12
constexpr int kLA = kKVL / kCl;       // 32

struct SmemA {
  __nv_bfloat16 wqb[kRowsA][kQL];      // 36 KB  (prefetched pre-wait)
  __nv_bfloat16 wuk[kNope][kLA];       // 8 KB   (prefetched pre-wait)
  __nv_bfloat16 wqa[kQL];              // 3 KB
  __nv_bfloat16 xq[kMaxB][kQL];        // 24 KB  normalized q_c (bf16)
  float qrow[kRowsA][kMaxB];           // this CTA's q rows (bf16-rounded values)
  float qn[kMaxB][kNope];              // gathered q_nope of the head
  float red[8][kMaxB][kLA];            // W_UK partial sums
  float ss[8][kMaxB];                  // rmsnorm partial sums
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
  TS(0)
  cg::cluster_group cluster = cg::this_cluster();
  const int r = cluster.block_rank();
  const int h = blockIdx.y;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

  // ---- pre-wait: static weights -> smem ----
  {
    const __nv_bfloat16* src = W_qb + (static_cast<size_t>(h) * kQHead + kRowsA * r) * kQL;
    for (int c = tid; c < kRowsA * kQL / 8; c += kThreads) cp_async16(&s.wqb[0][0] + c * 8, src + c * 8);
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
    cp_async_wait_all();
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
  TS(4)
  cluster.sync();
  TS(5)

  // ---- gather q_nope of head h from the cluster (rows 0..127 live in ranks 0..10) ----
  for (int i = tid; i < kNope * B; i += kThreads) {
    const int p = i % kNope, b = i / kNope;
    const float* remote = cluster.map_shared_rank(&s.qrow[0][0], p / kRowsA);
    s.qn[b][p] = remote[(p % kRowsA) * kMaxB + b];
  }
  // q_pe rows held by this CTA -> fp8 query (rope part, no rotation for the NoPE model)
  for (int i = tid; i < kRowsA * B; i += kThreads) {
    const int j = i % kRowsA, b = i / kRowsA;
    const int p = kRowsA * r + j;
    if (p >= kNope)
      mqa_q[(static_cast<size_t>(b) * kH + h) * kD + kKVL + (p - kNope)] = to_fp8(s.qrow[j][b] * qsi);
  }
  cluster.sync();  // all DSMEM reads done (smem of every CTA may be released after this)

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
}

// ======================================================================================
// Kernel B: attention + W_UV + gate.  grid (16, B): one 16-CTA cluster per request.
//   CTA r: KV tokens [r*per, (r+1)*per) of its request (per = ceil(L/16)), 64-token tiles in
//   smem (fp8 rows, stride 592 B). QK^T: mma m16n8k32 e4m3 (rows = 6 heads, cols = 8 tokens
//   per warp). Online softmax over tiles. PV: mma m16n8k16 f16 (P f16 x V fp8->f16), warp w
//   owns latent dims [64w, 64w+64). Cluster combine of (m, l, O) through DSMEM: CTA r
//   finalizes latent dims [32r, 32r+32) (bf16), then W_UV outputs [48r, 48r+48) (+ gate).
//   ~108 KB smem -> 2 CTAs/SM, so 14 clusters fit (B = 8 in one wave).
// ======================================================================================
constexpr int kTile = 64;
constexpr int kRowB = kD + 16;              // 592 B smem row stride
constexpr int kOutPerCta = kH * kV / kCl;  // 48
constexpr int kLatPerCta = kKVL / kCl;     // 32

struct SmemB {
  uint8_t kv[kTile][kRowB];                    // 37.9 KB (first tile prefetched pre-wait)
  __nv_bfloat16 wuv[kOutPerCta][kKVL + 8];     // 48.8 KB (W_UV slice [out][l], prefetched pre-wait)
  float S[kH][kTile];                          // scores (log2 domain)
  __half P[8][kTile];                          // probabilities (rows 6,7 zero)
  float alpha[8];
  float m[kH], l[kH];
  float O[kH][kKVL];                           // 12 KB unnormalized partial output
  float lat[kH][kLatPerCta];                   // finalized latent slice (bf16-rounded)
  float latg[2][kKVL];                         // gathered latent of the (<=2) heads needed
  float red[8][kOutPerCta];
  float wc[kH][kCl];                           // combine weights exp2(m_r - M)
  float lsc[kH * kCl];
  float Lsum[kH];
  int pages[kMaxTilePages];                    // block-table entries of the current tile
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
  TS(0)
  cg::cluster_group cluster = cg::this_cluster();
  const int r = cluster.block_rank();
  const int b = blockIdx.y;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t4 = lane & 3;

  const int L = max(seq_lens[b], 0);  // padded rows have seq_len 0 -> no KV work, zero output
  const int per = (L + kCl - 1) / kCl;
  const int t0 = min(L, r * per), t1 = min(L, t0 + per);
  const int ntok = t1 - t0;
  const int ntiles = (ntok + kTile - 1) / kTile;

  // ---- pre-wait: W_UV slice -> smem (static weight) ----
  const int o0 = kOutPerCta * r;
  for (int c = tid; c < kOutPerCta * (kKVL / 8); c += kThreads) {
    const int oo = c / (kKVL / 8), j = c % (kKVL / 8);
    const int o = o0 + oo, hh = o / kV, v = o % kV;
    cp_async16(&s.wuv[oo][j * 8], W_UV + hh * uv_sh + v * uv_sv + j * 8);
  }
  cp_async_commit();
  const int* btrow = block_table + static_cast<size_t>(b) * bt_stride;
  if (tid < kH) {
    s.m[tid] = -INFINITY;
    s.l[tid] = 0.f;
  }
  if (tid < 8) s.alpha[tid] = 0.f;
  if (tid < 2 * kTile) s.P[6 + tid / kTile][tid % kTile] = __float2half(0.f);
  __syncthreads();

  // block-table entries of a tile -> smem first (no dependent global load per cp.async)
  int tpg0 = 0;
  auto tile_pages = [&](int tile) {
    const int base = t0 + tile * kTile;
    const int n = min(kTile, t1 - base);
    tpg0 = base / page_size;
    const int np = (base + n - 1) / page_size - tpg0 + 1;
    __syncthreads();  // previous users of s.pages are done
    if (tid < np) s.pages[tid] = btrow[tpg0 + tid];
    __syncthreads();
  };
  auto row_ptr = [&](int tok) {
    return cache + static_cast<long long>(s.pages[tok / page_size - tpg0]) * page_stride +
           static_cast<long long>(tok % page_size) * kD;
  };
  auto load_tile = [&](int tile, bool skip_new) {
    const int base = t0 + tile * kTile;
    const int n = min(kTile, t1 - base);
    tile_pages(tile);
    for (int c = tid; c < n * (kD / 16); c += kThreads) {
      const int t = c / (kD / 16), j = c % (kD / 16);
      const int tok = base + t;
      if (skip_new && tok == L - 1) continue;  // written by q_prep; loaded after the wait
      cp_async16(&s.kv[t][j * 16], row_ptr(tok) + j * 16);
    }
    // rows past the end of the range: zero (stale smem could hold fp8 NaN patterns, 0*NaN = NaN)
    for (int c = tid; c < (kTile - n) * (kRowB / 16); c += kThreads)
      *reinterpret_cast<uint4*>(&s.kv[n + c / (kRowB / 16)][(c % (kRowB / 16)) * 16]) = make_uint4(0, 0, 0, 0);
    cp_async_commit();
  };
  if (ntiles > 0) load_tile(0, true);  // old KV rows of the first tile, before the wait
  TS(1)
  pdl_wait();
  TS(2)
  // newest token (inserted by q_prep) if it is in the first tile
  if (L - 1 >= t0 && L - 1 < min(t1, t0 + kTile) && tid < kD / 16) {
    const int tok = L - 1;  // s.pages still holds tile 0's pages
    cp_async16(&s.kv[tok - t0][tid * 16], row_ptr(tok) + tid * 16);
  }
  cp_async_commit();
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

  float oacc[8][4];
#pragma unroll
  for (int j = 0; j < 8; ++j) oacc[j][0] = oacc[j][1] = oacc[j][2] = oacc[j][3] = 0.f;

  for (int tile = 0; tile < ntiles; ++tile) {
    if (tile >= 1) load_tile(tile, false);  // long context (> 1024 tokens): synchronous tiles
    cp_async_wait_all();
    __syncthreads();
    const int n = min(kTile, ntok - tile * kTile);
    // ---- S = Q K^T for tokens [8w, 8w+8) of the tile ----
    {
      float c[4] = {0.f, 0.f, 0.f, 0.f}, c2[4] = {0.f, 0.f, 0.f, 0.f};
      const uint8_t* krow = &s.kv[8 * warp + g][8 * t4];
#pragma unroll
      for (int kk = 0; kk < kD / 32; kk += 2) {
        const uint2 kv = *reinterpret_cast<const uint2*>(krow + 32 * kk);
        const uint2 kv2 = *reinterpret_cast<const uint2*>(krow + 32 * kk + 32);
        mma_fp8(c, qa0[kk], qa2[kk], kv.x, kv.y);
        mma_fp8(c2, qa0[kk + 1], qa2[kk + 1], kv2.x, kv2.y);
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
    // ---- O = alpha * O + P V for dims [64w, 64w+64): lane group g <-> dims 64w + 8g + j ----
    {
      const float al = s.alpha[g];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        oacc[j][0] *= al;
        oacc[j][1] *= al;
      }
#pragma unroll
      for (int ks = 0; ks < kTile / 16; ++ks) {
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
    __syncthreads();  // tile / P / S reuse
  }
  TS(3)

  // ---- partial O (rows g < 6): C col n = 2t4 (+1) of n-tile j -> dim 64w + 8n + j ----
  if (g < kH) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      s.O[g][64 * warp + 8 * (2 * t4) + j] = oacc[j][0];
      s.O[g][64 * warp + 8 * (2 * t4 + 1) + j] = oacc[j][1];
    }
  }
  cluster.sync();
  TS(4)

  // ---- combine: CTA r finalizes latent dims [32r, 32r+32) for all heads ----
  if (tid < kH * kCl) {  // m, l of every rank (parallel DSMEM loads)
    const int hh = tid / kCl, rr = tid % kCl;
    const SmemB* rs = cluster.map_shared_rank(&s, rr);
    s.wc[hh][rr] = rs->m[hh];
    s.lsc[tid] = rs->l[hh];
  }
  __syncthreads();
  if (tid < kH) {
    float M = -INFINITY;
#pragma unroll
    for (int rr = 0; rr < kCl; ++rr) M = fmaxf(M, s.wc[tid][rr]);
    float Ls = 0.f;
#pragma unroll
    for (int rr = 0; rr < kCl; ++rr) {
      const float mr = s.wc[tid][rr];
      const float w = mr == -INFINITY ? 0.f : exp2f(mr - M);
      s.wc[tid][rr] = w;
      Ls += w * s.lsc[tid * kCl + rr];
    }
    s.Lsum[tid] = Ls > 0.f ? output_scale / Ls : 0.f;  // seq_len 0 (padding) -> zeros
  }
  __syncthreads();
  if (tid < kH * kLatPerCta) {
    const int hh = tid / kLatPerCta, d = kLatPerCta * r + tid % kLatPerCta;
    float ov[kCl];
#pragma unroll
    for (int rr = 0; rr < kCl; ++rr) ov[rr] = cluster.map_shared_rank(&s.O[hh][d], rr)[0];
    float o = 0.f;
#pragma unroll
    for (int rr = 0; rr < kCl; ++rr) o += ov[rr] * s.wc[hh][rr];
    s.lat[hh][tid % kLatPerCta] = bf16r(o * s.Lsum[hh]);
  }
  cluster.sync();
  TS(5)

  // ---- W_UV for outputs [48r, 48r+48) (+ sigmoid gate) ----
  {
    const int h0 = o0 / kV, h1 = (o0 + kOutPerCta - 1) / kV;
    for (int i = tid; i < (h1 - h0 + 1) * kKVL; i += kThreads) {
      const int hi = i / kKVL, l = i % kKVL;
      s.latg[hi][l] = cluster.map_shared_rank(&s.lat[0][0], l / kLatPerCta)[(h0 + hi) * kLatPerCta + l % kLatPerCta];
    }
    cp_async_wait_all();  // W_UV slice
    cluster.sync();       // gathers done (also keeps remote smem alive until everyone has read)
    if (tid < 4 * kOutPerCta) {  // 48 outputs x 4 l-parts of 128
      const int oo = tid % kOutPerCta, part = tid / kOutPerCta;
      const int hi = (o0 + oo) / kV - h0;
      float a[4] = {0.f, 0.f, 0.f, 0.f};
      const float* lg = &s.latg[hi][part * (kKVL / 4)];
      const __nv_bfloat16* wr = &s.wuv[oo][part * (kKVL / 4)];
#pragma unroll 4
      for (int l = 0; l < kKVL / 4; l += 8) {
        float w[8];
        bf16x8_to_f32(*reinterpret_cast<const uint4*>(wr + l), w);
        const float4 x0 = *reinterpret_cast<const float4*>(lg + l);
        const float4 x1 = *reinterpret_cast<const float4*>(lg + l + 4);
        a[0] += x0.x * w[0] + x0.y * w[1];
        a[1] += x0.z * w[2] + x0.w * w[3];
        a[2] += x1.x * w[4] + x1.y * w[5];
        a[3] += x1.z * w[6] + x1.w * w[7];
      }
      s.red[part][oo] = (a[0] + a[1]) + (a[2] + a[3]);
    }
    __syncthreads();
    if (tid < kOutPerCta) {
      float a = 0.f;
#pragma unroll
      for (int part = 0; part < 4; ++part) a += s.red[part][tid];
      a = bf16r(a);
      const float gt = __bfloat162float(gate[static_cast<size_t>(b) * gate_stride + o0 + tid]);
      out[static_cast<size_t>(b) * (kH * kV) + o0 + tid] = __float2bfloat16(a * (1.f / (1.f + __expf(-gt))));
    }
  }
  TS(6)
  pdl_trigger();
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
              W_qb.scalar_type() == at::kBFloat16);
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

TORCH_LIBRARY(k3mla, m) {
  m.def(
      "q_prep(Tensor qkv, Tensor w_qa, Tensor w_kva, Tensor W_qb, Tensor W_UK_T, Tensor(a!) kv_cache, "
      "Tensor slot_mapping, Tensor q_scale_inv, Tensor k_scale_inv, float eps, Tensor(b!) mqa_q, "
      "int pdl, Tensor? ts=None) -> ()");
  m.def(
      "attn_out(Tensor mqa_q, Tensor kv_cache, Tensor block_table, Tensor seq_lens, Tensor W_UV, "
      "Tensor gate, float softmax_scale, float output_scale, Tensor(a!) out, int pdl, "
      "Tensor? ts=None) -> ()");
}
TORCH_LIBRARY_IMPL(k3mla, CUDA, m) {
  m.impl("q_prep", &q_prep);
  m.impl("attn_out", &attn_out);
}
