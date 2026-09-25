// Fused Kimi K3 "MoE-tail Lamport consume + next layer's pre-attention AttnRes" for decode.
//
// Replaces two kernels on the per-layer critical path:
//   LamportCopyKernel (poll symmetric mailbox -> fresh tensor, re-arm sentinels)
//   torch.ops._C.kimi_k3_attn_res(prefix, delta=<copy>, blocks, ...)   (KimiDecoderLayer._pre_attn_norm)
//
// Semantics (vllm/models/kimi_k3/nvidia/ops/attn_res.py, HAS_DELTA=True):
//   delta    = mailbox[t]                      (polled until no 32-bit word equals 0x80000000,
//                                               then every word is re-armed to 0x80000000)
//   prefix'  = bf16(prefix + delta)            (stored back into prefix)
//   blocks[t, write_idx] = prefix'             (when write_idx >= 0; must be >= num_blocks)
//   sources  = blocks[t, 0..NB-1], prefix'
//   logit_s  = dot(v_s, norm_w*qk_w) * rsqrt(mean(v_s^2) + eps);  a = softmax(logit)
//   mixed    = sum_s a_s v_s
//   out      = mixed * rsqrt(mean(mixed^2) + out_eps) * out_w     (if out_w given, else mixed)
//
// Parallelization: one C-CTA thread-block cluster per token (C=7: 128 threads x 8 bf16 per CTA,
// one 16-byte load per source row per thread).
//  * Before polling (overlaps the producer when it triggers PDL early): load weights, stored
//    block rows, old prefix and issue the first poll load; reduce the block norms/dots, exchange
//    them across the cluster (st.async + mbarrier tx-count), form the block logits and the
//    block-only softmax mix B = sum_s exp(l_s - m0) v_s per element.
//  * After polling: P = prefix'; only 4 scalars cross the cluster: <B,P>, |B|^2, |P|^2, <P,w>.
//    With alpha = exp(m0 - mx), e_P = exp(l_P - mx), den = alpha*sum_s exp(l_s-m0) + e_P:
//      mixed = (alpha B + e_P P) / den,  |mixed|^2 = (alpha^2|B|^2 + 2 alpha e_P <B,P> + e_P^2|P|^2)/den^2
//    so the output RMS needs no second reduction round.
//  * DSMEM exchanges use st.async with mbarrier complete_tx (no barrier.cluster release on the
//    critical path); global side effects (prefix, block write, sentinel re-arm) are issued after
//    the pushes.
//
// PDL: the kernel issues griddepcontrol.launch_dependents at entry and (like LamportCopy) never
// executes griddepcontrol.wait unless wait_prior=1 (debug op): the mailbox poll is the readiness
// signal. prefix/blocks/weights must have been produced >= 2 kernels earlier (true in the model).
#include <cuda_bf16.h>
#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#ifndef K3TAIL_C
#define K3TAIL_C 7
#endif

namespace {

constexpr int kHidden = 7168;
constexpr int kVec = 8;                       // bf16 per 16-byte fragment
constexpr int kV2 = kVec / 2;                 // float2 pairs per fragment
constexpr int kC = K3TAIL_C;                  // CTAs (cluster size) per token
constexpr int kThreads = kHidden / (kC * kVec);
constexpr int kWarps = kThreads / 32;
static_assert(kC * kThreads * kVec == kHidden && kThreads % 32 == 0, "bad cluster split");
static_assert(kThreads >= 54, "need one thread per exchanged scalar");
constexpr int kMaxNB = 8;
constexpr uint32_t kSentinel = 0x80000000u;   // upstream NEG_ZERO_F32_BITS, one per 32-bit word
constexpr int kMaxPre = kMaxNB * (kMaxNB + 1) / 2 + kMaxNB;  // 44
constexpr int kMaxPost = kMaxNB + 2;                          // 10
constexpr float kInvH = 1.0f / kHidden;
constexpr int kProbes = 16;

__host__ __device__ constexpr int pow2ceil(int x) { int p = 1; while (p < x) p <<= 1; return p; }

struct Params {
  __nv_bfloat16* mailbox;
  __nv_bfloat16* prefix;
  __nv_bfloat16* blocks;
  const __nv_bfloat16* norm_w;
  const __nv_bfloat16* qk_w;
  const __nv_bfloat16* out_w;
  __nv_bfloat16* out;
  __nv_bfloat16* delta_out;   // optional copy of the MoE output (e.g. aux hidden-state capture)
  long long mailbox_sm, prefix_sm, out_sm, delta_sm, blocks_sm, blocks_sr;
  int write_idx;
  int wait_prior;
  float eps, out_eps;
  unsigned long long* ts;     // optional %globaltimer probes [M][C][kProbes] (profiling only)
};

struct alignas(16) U4 { uint32_t w[4]; };

__device__ __forceinline__ U4 ldg16(const void* p) {
  U4 u;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(u.w[0]), "=r"(u.w[1]), "=r"(u.w[2]), "=r"(u.w[3]) : "l"(p));
  return u;
}
__device__ __forceinline__ U4 ld_volatile16(const void* p) {
  U4 u;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(u.w[0]), "=r"(u.w[1]), "=r"(u.w[2]), "=r"(u.w[3]) : "l"(p) : "memory");
  return u;
}
__device__ __forceinline__ void st16(void* p, const U4& u) {
  asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(u.w[0]), "r"(u.w[1]),
               "r"(u.w[2]), "r"(u.w[3]) : "memory");
}
__device__ __forceinline__ bool dirty(const U4& u) {
  return (u.w[0] == kSentinel) | (u.w[1] == kSentinel) | (u.w[2] == kSentinel) |
         (u.w[3] == kSentinel);
}
__device__ __forceinline__ void unpack(const U4& u, float2 (&f)[kV2]) {
#pragma unroll
  for (int i = 0; i < kV2; ++i)
    f[i] = make_float2(__uint_as_float(u.w[i] << 16), __uint_as_float(u.w[i] & 0xffff0000u));
}
__device__ __forceinline__ U4 pack(const float2 (&f)[kV2]) {
  U4 u;
#pragma unroll
  for (int i = 0; i < kV2; ++i) {
    __nv_bfloat162 b = __float22bfloat162_rn(f[i]);
    u.w[i] = *reinterpret_cast<uint32_t*>(&b);
  }
  return u;
}
__device__ __forceinline__ float dot8(const float2 (&a)[kV2], const float2 (&b)[kV2]) {
  float2 s = __fmul2_rn(a[0], b[0]);
  float2 s1 = __fmul2_rn(a[1], b[1]);
  s = __ffma2_rn(a[2], b[2], s);
  s1 = __ffma2_rn(a[3], b[3], s1);
  s = __fadd2_rn(s, s1);
  return s.x + s.y;
}

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
// Timer read that is ordered after `dep` becomes available (for "loads landed" probes).
__device__ __forceinline__ unsigned long long gtimer_after(float dep) {
  unsigned long long t;
  asm volatile("{ .reg .f32 d; mov.f32 d, %1; mov.u64 %0, %globaltimer; }" : "=l"(t) : "f"(dep));
  return t;
}

__device__ __forceinline__ void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_arrive_release() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_wait() {
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ void st_cluster(const float* local, int rank, float v) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(local));
  uint32_t ra;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(a), "r"(rank));
  asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(ra), "f"(v) : "memory");
}

// Butterfly reduce-scatter of N (power of two) per-lane values across the warp with xor
// offsets O, O/2, ..., 1. Each halving step costs N/2 shuffles; once one value is left the
// remaining offsets are a plain all-reduce. See rs_index for where each sum ends up.
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
// Original index of the value lane `lane` holds in slot j after RS<KP,16>.
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
// Number of lanes holding a copy of each reduced value.
template <int KP>
__host__ __device__ constexpr int rs_dup() { return KP >= 32 ? 1 : 32 / KP; }

__device__ __forceinline__ float tree_max(const float* x, int n) {
  float m[16];
#pragma unroll
  for (int i = 0; i < n; ++i) m[i] = x[i];
#pragma unroll
  for (int w = 1; w < n; w <<= 1)
#pragma unroll
    for (int i = 0; i + w < n; i += 2 * w) m[i] = fmaxf(m[i], m[i + w]);
  return m[0];
}
__device__ __forceinline__ float tree_sum(const float* x, int n) {
  float m[16];
#pragma unroll
  for (int i = 0; i < n; ++i) m[i] = x[i];
#pragma unroll
  for (int w = 1; w < n; w <<= 1)
#pragma unroll
    for (int i = 0; i + w < n; i += 2 * w) m[i] += m[i + w];
  return m[0];
}

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_addr(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("{ .reg .b64 st; mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1; }" ::"r"(smem_addr(bar)),
               "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred P1;\n\t"
      "LAB_WAIT:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
      "@!P1 bra LAB_WAIT;\n\t}" ::"r"(smem_addr(bar)), "r"(parity) : "memory");
}
// Asynchronous DSMEM store of one fp32 into CTA `rank`'s copy of `local`, completing `bytes`
// on that CTA's copy of mbarrier `bar` (no release fence / cluster barrier needed).
__device__ __forceinline__ void st_async(const float* local, const uint64_t* bar, int rank, float v) {
  uint32_t ra, rb;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(smem_addr(local)), "r"(rank));
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rb) : "r"(smem_addr(bar)), "r"(rank));
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(ra),
               "r"(__float_as_uint(v)), "r"(rb) : "memory");
}

template <int NB, bool OUT_NORM>
__global__ void __launch_bounds__(kThreads)
lamport_attn_res_kernel(const Params p) {
  constexpr int NS = NB + 1;
  // Pre-poll exchange: block norms S_s and dots D_s (s < NB).
  constexpr int NPRE = 2 * NB;
  constexpr int KPRE = pow2ceil(NPRE > 0 ? NPRE : 1);
  // Post-poll exchange: [X=<B,P>, Q=|B|^2,] S_P, D_P   (B = block-only softmax mix)
  //                     NB == 0: [S_P] (output norm only).
  constexpr int NPOST = NB > 0 ? (OUT_NORM ? 4 : 2) : (OUT_NORM ? 1 : 0);
  constexpr int KPOST = pow2ceil(NPOST > 0 ? NPOST : 1);
  constexpr bool EXCH = NPOST > 0;
  constexpr int iS = NB > 0 && OUT_NORM ? 2 : 0;  // index of S_P in the post layout

  const int rank = blockIdx.x;  // == %cluster_ctarank for cluster dims (kC,1,1)
  const int t = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int h = (rank * kThreads + tid) * kVec;
  const bool prof = p.ts != nullptr;
  unsigned long long tp[kProbes] = {};

  __shared__ float s_wpre[kWarps][2 * kMaxNB];
  __shared__ float s_pre[kC][2 * kMaxNB];
  __shared__ __align__(16) float s_blk[2 * kMaxNB];
  __shared__ float s_post[kC][kWarps][4];
  __shared__ __align__(16) float s_tot[4];
  __shared__ __align__(8) uint64_t s_bar[2];  // [0]: pre exchange, [1]: post exchange

  if (prof) tp[0] = gtimer();
  if constexpr (EXCH) {
    if (tid == 0) {
      if constexpr (NB > 0) {
        mbar_init(&s_bar[0], 1);
        mbar_arrive_expect_tx(&s_bar[0], kC * NPRE * 4);
      }
      mbar_init(&s_bar[1], 1);
      mbar_arrive_expect_tx(&s_bar[1], kC * kWarps * NPOST * 4);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    cluster_arrive_relaxed();  // "this CTA is running and its mbarriers are initialized"
  }
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  // ---- Issue every load at once: weights, stored blocks, old prefix and the first poll. ----
  float2 v[NS][kV2];
  float2 w[kV2], ow[kV2], old[kV2];
  U4 wn, wq, wo;
  if constexpr (NB > 0) {
    wn = ldg16(p.norm_w + h);
    wq = ldg16(p.qk_w + h);
  }
  if constexpr (OUT_NORM) wo = ldg16(p.out_w + h);
  const long long brow = static_cast<long long>(t) * p.blocks_sm + h;
  U4 bu[NS];
#pragma unroll
  for (int s = 0; s < NB; ++s) bu[s] = ldg16(p.blocks + brow + s * p.blocks_sr);
  const U4 pu = ldg16(p.prefix + static_cast<long long>(t) * p.prefix_sm + h);
  __nv_bfloat16* mb = p.mailbox + static_cast<long long>(t) * p.mailbox_sm + h;
  if (p.wait_prior) asm volatile("griddepcontrol.wait;" ::: "memory");
  U4 u = ld_volatile16(mb);

  if constexpr (NB > 0) {
    float2 a2[kV2], b2[kV2];
    unpack(wn, a2);
    unpack(wq, b2);
#pragma unroll
    for (int i = 0; i < kV2; ++i) w[i] = __fmul2_rn(a2[i], b2[i]);
  }
  if constexpr (OUT_NORM) unpack(wo, ow);
#pragma unroll
  for (int s = 0; s < NB; ++s) unpack(bu[s], v[s]);
  unpack(pu, old);
  if (prof) {
    float dep = old[0].x;
    if constexpr (NB > 0) dep += v[NB - 1][0].x;
    tp[1] = gtimer_after(dep);
  }

  // ---- Pre-poll: block logits -> block-only softmax mix B = sum_s e_s v_s (e_s = exp(l_s - m0)).
  float2 B[kV2];
  float m0 = 0.f, Eb = 0.f, qpart = 0.f;
  if constexpr (NB > 0) {
    float a[KPRE];
#pragma unroll
    for (int k = 0; k < KPRE; ++k) a[k] = 0.f;
#pragma unroll
    for (int s = 0; s < NB; ++s) {
      a[s] = dot8(v[s], v[s]);
      a[NB + s] = dot8(v[s], w);
    }
    RS<KPRE, 16>::run(a, lane);
    if (prof) tp[8] = gtimer_after(a[0]);
    {
      constexpr int kHeld = KPRE >= 32 ? KPRE / 32 : 1;
      if ((lane & (rs_dup<KPRE>() - 1)) == 0) {
#pragma unroll
        for (int j = 0; j < kHeld; ++j) {
          const int idx = rs_index<KPRE>(j, lane);
          if (idx < NPRE) s_wpre[warp][idx] = a[j];
        }
      }
    }
    __syncthreads();
    cluster_wait();  // every CTA of the cluster is resident with initialized mbarriers
    if (prof) tp[9] = gtimer();
    if (tid < NPRE) {
      float s = 0.f;
#pragma unroll
      for (int k = 0; k < kWarps; ++k) s += s_wpre[k][tid];
#pragma unroll
      for (int r = 0; r < kC; ++r) st_async(&s_pre[rank][tid], &s_bar[0], r, s);
    }
    mbar_wait(&s_bar[0], 0);
    if (prof) tp[10] = gtimer();
    if (tid < NPRE) {
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int r = 0; r < kC; ++r) {
        if (r & 1) s1 += s_pre[r][tid];
        else s0 += s_pre[r][tid];
      }
      s_blk[tid] = s0 + s1;
    }
    __syncthreads();
    float T[2 * kMaxNB];
#pragma unroll
    for (int k = 0; k < NPRE; k += 4) {
      const float4 x = *reinterpret_cast<const float4*>(&s_blk[k]);
      T[k] = x.x;
      T[k + 1] = x.y;
      T[k + 2] = x.z;
      T[k + 3] = x.w;
    }
    float lg[NB], e[NB];
#pragma unroll
    for (int s = 0; s < NB; ++s) lg[s] = T[NB + s] * rsqrtf(fmaf(T[s], kInvH, p.eps));
    m0 = tree_max(lg, NB);
#pragma unroll
    for (int s = 0; s < NB; ++s) e[s] = __expf(lg[s] - m0);
    Eb = tree_sum(e, NB);
#pragma unroll
    for (int i = 0; i < kV2; ++i) {
      float2 acc = __fmul2_rn(make_float2(e[0], e[0]), v[0][i]);
#pragma unroll
      for (int s = 1; s < NB; ++s) acc = __ffma2_rn(make_float2(e[s], e[s]), v[s][i], acc);
      B[i] = acc;
    }
    if constexpr (OUT_NORM) qpart = dot8(B, B);
  } else if constexpr (EXCH) {
    cluster_wait();
  }
  if (prof) {
    float dep = qpart;
    if constexpr (NB > 0) dep += B[0].x;
    tp[2] = gtimer_after(dep);
  }

  // ---- Poll the mailbox fragment; the data itself is the readiness signal. ----
  while (dirty(u)) u = ld_volatile16(mb);
  float2 P[kV2];
  {
    float2 d[kV2];
    unpack(u, d);
#pragma unroll
    for (int i = 0; i < kV2; ++i)
      P[i] = __bfloat1622float2(__float22bfloat162_rn(__fadd2_rn(old[i], d[i])));
  }
  if (prof) tp[3] = gtimer_after(P[0].x);
  const U4 pp = pack(P);
  const long long prow = static_cast<long long>(t) * p.prefix_sm + h;
  __nv_bfloat16* outp = p.out + static_cast<long long>(t) * p.out_sm + h;

  if constexpr (EXCH) {
    float b[KPOST];
#pragma unroll
    for (int k = 0; k < KPOST; ++k) b[k] = 0.f;
    if constexpr (NB > 0) {
      if constexpr (OUT_NORM) {
        b[0] = dot8(B, P);
        b[1] = qpart;
      }
      b[iS] = dot8(P, P);
      b[iS + 1] = dot8(P, w);
    } else {
      b[0] = dot8(P, P);
    }
    RS<KPOST, 16>::run(b, lane);
    {
      constexpr int dup = rs_dup<KPOST>();
      const int idx = rs_index<KPOST>(0, lane);
      if (idx < NPOST) {
#pragma unroll
        for (int r = lane & (dup - 1); r < kC; r += dup) st_async(&s_post[rank][warp][idx], &s_bar[1], r, b[0]);
      }
    }
    if (prof) tp[4] = gtimer();
  }
  // Global side effects (off the exchange's critical path).
  st16(p.prefix + prow, pp);
  if (p.write_idx >= 0) st16(p.blocks + brow + p.write_idx * p.blocks_sr, pp);
  {
    U4 s;
    s.w[0] = s.w[1] = s.w[2] = s.w[3] = kSentinel;
    st16(mb, s);  // re-arm
  }
  if (p.delta_out != nullptr) st16(p.delta_out + static_cast<long long>(t) * p.delta_sm + h, u);

  if constexpr (!EXCH) {
    st16(outp, pp);  // single source, no norm: out = prefix'
    if (prof) tp[7] = gtimer();
  } else {
    mbar_wait(&s_bar[1], 0);
    if (prof) tp[5] = gtimer();
    if (tid < NPOST) {
      float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int r = 0; r < kC; ++r)
#pragma unroll
        for (int q = 0; q < kWarps; ++q) s[(r * kWarps + q) & 3] += s_post[r][q][tid];
      s_tot[tid] = (s[0] + s[1]) + (s[2] + s[3]);
    }
    __syncthreads();
    if (prof) tp[6] = gtimer();
    const float4 T = *reinterpret_cast<const float4*>(s_tot);
    float2 o[kV2];
    if constexpr (NB > 0) {
      const float X = T.x, Q = T.y;
      const float SP = iS == 2 ? T.z : T.x, DP = iS == 2 ? T.w : T.y;
      const float lp = DP * rsqrtf(fmaf(SP, kInvH, p.eps));
      const float mx = fmaxf(m0, lp);
      const float alpha = __expf(m0 - mx), ep = __expf(lp - mx);
      const float den = fmaf(alpha, Eb, ep);
      float ca, cp;  // out = ca * B + cp * P  (times out_w)
      if constexpr (OUT_NORM) {
        // |alpha B + ep P|^2 = alpha^2 Q + 2 alpha ep X + ep^2 S_P; rstd/den folded into one rsqrt.
        const float quad = fmaf(alpha * alpha, Q, fmaf(2.f * alpha * ep, X, ep * ep * SP));
        const float scale = rsqrtf(fmaf(quad, kInvH, p.out_eps * den * den));
        ca = alpha * scale;
        cp = ep * scale;
      } else {
        const float inv = __fdividef(1.0f, den);
        ca = alpha * inv;
        cp = ep * inv;
      }
#pragma unroll
      for (int i = 0; i < kV2; ++i) {
        const float2 acc = __ffma2_rn(make_float2(ca, ca), B[i], __fmul2_rn(make_float2(cp, cp), P[i]));
        o[i] = OUT_NORM ? __fmul2_rn(acc, ow[i]) : acc;
      }
    } else {
      const float scale = rsqrtf(fmaf(T.x, kInvH, p.out_eps));
#pragma unroll
      for (int i = 0; i < kV2; ++i) o[i] = __fmul2_rn(__fmul2_rn(make_float2(scale, scale), P[i]), ow[i]);
    }
    st16(outp, pack(o));
    if (prof) tp[7] = gtimer_after(o[0].x);
  }
  if (prof && tid == 0) {
    unsigned long long* dst = p.ts + (static_cast<long long>(t) * kC + rank) * kProbes;
#pragma unroll
    for (int k = 0; k < kProbes; ++k) dst[k] = tp[k];
  }
}


template <int NB, bool OUT_NORM>
void launch(const Params& prm, int M, cudaStream_t stream) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(kC, M);
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = kC;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 2;
  if constexpr (kC > 8) {
    static bool once = [] {
      cudaFuncSetAttribute(lamport_attn_res_kernel<NB, OUT_NORM>,
                           cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      return true;
    }();
    (void)once;
  }
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, lamport_attn_res_kernel<NB, OUT_NORM>, prm));
}

template <bool OUT_NORM>
void dispatch_nb(int nb, const Params& prm, int M, cudaStream_t s) {
  switch (nb) {
    case 0: launch<0, OUT_NORM>(prm, M, s); break;
    case 1: launch<1, OUT_NORM>(prm, M, s); break;
    case 2: launch<2, OUT_NORM>(prm, M, s); break;
    case 3: launch<3, OUT_NORM>(prm, M, s); break;
    case 4: launch<4, OUT_NORM>(prm, M, s); break;
    case 5: launch<5, OUT_NORM>(prm, M, s); break;
    case 6: launch<6, OUT_NORM>(prm, M, s); break;
    case 7: launch<7, OUT_NORM>(prm, M, s); break;
    case 8: launch<8, OUT_NORM>(prm, M, s); break;
    default: TORCH_CHECK(false, "num_blocks must be in [0, 8]");
  }
}

void check_row_major(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16, name, " must be CUDA bf16");
  TORCH_CHECK(x.stride(-1) == 1 && x.size(-1) == kHidden, name, " must be [..., 7168] with unit stride");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, name, " must be 16-byte aligned");
}

}  // namespace

void lamport_attn_res_impl(torch::Tensor mailbox, torch::Tensor prefix, torch::Tensor blocks,
                           torch::Tensor norm_weight, torch::Tensor qk_weight,
                           std::optional<torch::Tensor> output_norm_weight, torch::Tensor out,
                           std::optional<torch::Tensor> delta_out, int64_t num_blocks,
                           int64_t block_write_idx, double eps, double output_norm_eps,
                           bool wait_prior, std::optional<torch::Tensor> timestamps) {
  TORCH_CHECK(prefix.dim() == 2 && out.dim() == 2 && blocks.dim() == 3);
  const int M = prefix.size(0);
  TORCH_CHECK(out.size(0) == M && blocks.size(0) >= M, "row counts disagree");
  check_row_major(prefix, "prefix");
  check_row_major(out, "out");
  check_row_major(blocks, "blocks");
  check_row_major(norm_weight, "norm_weight");
  check_row_major(qk_weight, "qk_weight");
  TORCH_CHECK(norm_weight.is_contiguous() && qk_weight.is_contiguous());
  TORCH_CHECK(blocks.stride(0) % 8 == 0 && blocks.stride(1) % 8 == 0 && prefix.stride(0) % 8 == 0 &&
              out.stride(0) % 8 == 0, "row strides must be multiples of 8 elements");
  // The mailbox is the fixed-capacity [1, max_m, H] (or [max_m, H]) symmetric buffer.
  TORCH_CHECK(mailbox.is_contiguous() && mailbox.scalar_type() == at::kBFloat16 &&
              mailbox.size(-1) == kHidden && mailbox.numel() >= static_cast<int64_t>(M) * kHidden,
              "mailbox must be contiguous bf16 [.., max_m >= M, 7168]");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(mailbox.data_ptr()) % 16 == 0);
  TORCH_CHECK(num_blocks >= 0 && num_blocks <= kMaxNB && num_blocks <= blocks.size(1));
  TORCH_CHECK(block_write_idx == -1 || (block_write_idx >= num_blocks && block_write_idx < blocks.size(1)),
              "block_write_idx must be -1 or in [num_blocks, blocks.size(1))");
  if (output_norm_weight) {
    check_row_major(*output_norm_weight, "output_norm_weight");
    TORCH_CHECK(output_norm_weight->is_contiguous());
  }
  if (delta_out) {
    check_row_major(*delta_out, "delta_out");
    TORCH_CHECK(delta_out->size(0) == M && delta_out->stride(0) % 8 == 0);
  }
  if (timestamps) {
    TORCH_CHECK(timestamps->is_cuda() && timestamps->scalar_type() == at::kLong &&
                timestamps->is_contiguous() && timestamps->numel() >= static_cast<int64_t>(M) * kC * kProbes);
  }
  if (M == 0) return;
  auto bf = [](const torch::Tensor& x) { return reinterpret_cast<__nv_bfloat16*>(x.data_ptr()); };
  Params prm;
  prm.mailbox = bf(mailbox);
  prm.prefix = bf(prefix);
  prm.blocks = bf(blocks);
  prm.norm_w = bf(norm_weight);
  prm.qk_w = bf(qk_weight);
  prm.out_w = output_norm_weight ? bf(*output_norm_weight) : nullptr;
  prm.out = bf(out);
  prm.delta_out = delta_out ? bf(*delta_out) : nullptr;
  prm.mailbox_sm = kHidden;
  prm.prefix_sm = prefix.stride(0);
  prm.out_sm = out.stride(0);
  prm.delta_sm = delta_out ? delta_out->stride(0) : 0;
  prm.blocks_sm = blocks.stride(0);
  prm.blocks_sr = blocks.stride(1);
  prm.write_idx = static_cast<int>(block_write_idx);
  prm.wait_prior = wait_prior ? 1 : 0;
  prm.eps = static_cast<float>(eps);
  prm.out_eps = static_cast<float>(output_norm_eps);
  prm.ts = timestamps ? reinterpret_cast<unsigned long long*>(timestamps->data_ptr()) : nullptr;
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  if (output_norm_weight) dispatch_nb<true>(static_cast<int>(num_blocks), prm, M, stream);
  else dispatch_nb<false>(static_cast<int>(num_blocks), prm, M, stream);
}

void lamport_attn_res(torch::Tensor mailbox, torch::Tensor prefix, torch::Tensor blocks,
                      torch::Tensor norm_weight, torch::Tensor qk_weight,
                      std::optional<torch::Tensor> output_norm_weight, torch::Tensor out,
                      std::optional<torch::Tensor> delta_out, int64_t num_blocks,
                      int64_t block_write_idx, double eps, double output_norm_eps) {
  lamport_attn_res_impl(mailbox, prefix, blocks, norm_weight, qk_weight, output_norm_weight, out,
                        delta_out, num_blocks, block_write_idx, eps, output_norm_eps, false,
                        std::nullopt);
}

// Test/profiling entry point: optional griddepcontrol.wait before polling + %globaltimer probes.
void lamport_attn_res_debug(torch::Tensor mailbox, torch::Tensor prefix, torch::Tensor blocks,
                            torch::Tensor norm_weight, torch::Tensor qk_weight,
                            std::optional<torch::Tensor> output_norm_weight, torch::Tensor out,
                            std::optional<torch::Tensor> delta_out, int64_t num_blocks,
                            int64_t block_write_idx, double eps, double output_norm_eps,
                            bool wait_prior, std::optional<torch::Tensor> timestamps) {
  lamport_attn_res_impl(mailbox, prefix, blocks, norm_weight, qk_weight, output_norm_weight, out,
                        delta_out, num_blocks, block_write_idx, eps, output_norm_eps, wait_prior,
                        timestamps);
}

#ifndef K3TAIL_NS
#define K3TAIL_NS k3tail  // variants (e.g. -DK3TAIL_C=14 -DK3TAIL_NS=k3tail_c14) can coexist
#endif
#define K3TAIL_LIBRARY(ns, m) TORCH_LIBRARY(ns, m)
#define K3TAIL_LIBRARY_IMPL(ns, k, m) TORCH_LIBRARY_IMPL(ns, k, m)

K3TAIL_LIBRARY(K3TAIL_NS, m) {
  m.def("lamport_attn_res(Tensor(a!) mailbox, Tensor(b!) prefix, Tensor(c!) blocks, "
        "Tensor norm_weight, Tensor qk_weight, Tensor? output_norm_weight, Tensor(d!) out, "
        "Tensor(e!)? delta_out, int num_blocks, int block_write_idx, float eps, "
        "float output_norm_eps) -> ()");
  m.def("lamport_attn_res_debug(Tensor(a!) mailbox, Tensor(b!) prefix, Tensor(c!) blocks, "
        "Tensor norm_weight, Tensor qk_weight, Tensor? output_norm_weight, Tensor(d!) out, "
        "Tensor(e!)? delta_out, int num_blocks, int block_write_idx, float eps, "
        "float output_norm_eps, bool wait_prior, Tensor(f!)? timestamps) -> ()");
}
K3TAIL_LIBRARY_IMPL(K3TAIL_NS, CUDA, m) {
  m.impl("lamport_attn_res", &lamport_attn_res);
  m.impl("lamport_attn_res_debug", &lamport_attn_res_debug);
}
