// Fused TP all-reduce + Kimi K3 AttnRes for decode batches (M <= kMaxM).
//
// Replaces [o_proj all-reduce] -> [attn_res] after attention. Each rank
// multicasts its bf16 o_proj partial through NVLS (multimem.st) into slot
// `rank` of every rank's Lamport mailbox [tp][kMaxM][7168]; consumers poll
// their local mailbox until all tp slots hold data (no -0.0 sentinels), sum the
// slots in rank order (identical on every rank), reset the sentinels, and run
// AttnRes on the result. Semantics match vllm's KimiDecoderLayer._post_attn_norm:
//   attn     = bf16(sum_r partial_r)
//   prefix'  = bf16(prefix + attn) if HAS_DELTA else attn     (stored to prefix)
//   sources  = blocks[t, 0..num_blocks-1], prefix'
//   out      = RMSNorm_w(sum_s softmax(logit_s) v_s),  logit_s = dot(v_s, nw*qw) / rms(v_s)
// One 7-CTA cluster per token, 128 threads x 8 elements per CTA.
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

namespace cg = cooperative_groups;

namespace {

constexpr int kHidden = 7168;
constexpr int kCluster = 7;
constexpr int kThreads = 128;
constexpr int kVec = 8;
constexpr int kWarps = kThreads / 32;
constexpr int kMaxSources = 9;
constexpr int kMaxM = 16;
constexpr uint32_t kNegZero2 = 0x80008000u;  // two bf16 -0.0

struct alignas(16) U4 {
  uint32_t v[4];
};

__device__ __forceinline__ void unpack(const U4& u, float (&f)[kVec]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&u.v[i]);
    const float2 x = __bfloat1622float2(b);
    f[2 * i] = x.x;
    f[2 * i + 1] = x.y;
  }
}

__device__ __forceinline__ U4 pack(const float (&f)[kVec]) {
  U4 u;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __nv_bfloat162 b = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
    u.v[i] = *reinterpret_cast<const uint32_t*>(&b);
  }
  return u;
}

__device__ __forceinline__ void load8(const __nv_bfloat16* p, float (&f)[kVec]) {
  unpack(*reinterpret_cast<const U4*>(p), f);
}

__device__ __forceinline__ bool has_sentinel(const U4& u) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if ((u.v[i] & 0xffffu) == 0x8000u || (u.v[i] >> 16) == 0x8000u) return true;
  }
  return false;
}

__device__ __forceinline__ U4 ld_volatile(const void* p) {
  U4 u;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(u.v[0]), "=r"(u.v[1]), "=r"(u.v[2]), "=r"(u.v[3])
               : "l"(p));
  return u;
}

__device__ __forceinline__ void multimem_st(uint64_t addr, const U4& u) {
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(addr),
               "r"(u.v[0]), "r"(u.v[1]), "r"(u.v[2]), "r"(u.v[3])
               : "memory");
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o);
  return x;
}

template <int NS, bool HAS_DELTA, bool WRITE_BLOCK, bool OUT_NORM>
__global__ void __cluster_dims__(kCluster, 1, 1) __launch_bounds__(kThreads)
ar_attn_res_kernel(const __nv_bfloat16* __restrict__ partial, __nv_bfloat16* mailbox,
                   uint64_t mailbox_mc, int rank, int tp, __nv_bfloat16* __restrict__ prefix,
                   __nv_bfloat16* __restrict__ blocks, const __nv_bfloat16* __restrict__ norm_w,
                   const __nv_bfloat16* __restrict__ qk_w, const __nv_bfloat16* __restrict__ out_w,
                   __nv_bfloat16* __restrict__ out, long block_stride_m, long block_stride_r,
                   int write_idx, float eps, float out_eps, unsigned long long* __restrict__ trace) {
#define K3_MARK(slot)                                                                  \
  if (trace && threadIdx.x == 0 && cg::this_cluster().block_rank() == 0 && blockIdx.y == 0) { \
    unsigned long long ts;                                                             \
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts));                            \
    trace[slot] = ts;                                                                  \
  }
  K3_MARK(0)
  constexpr int NB = NS - 1;
  cg::cluster_group cluster = cg::this_cluster();
  const int crank = cluster.block_rank();
  const int t = blockIdx.y;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int h = (crank * kThreads + threadIdx.x) * kVec;
  __shared__ float warp_part[kWarps][2 * kMaxSources + 2];
  __shared__ float cta_part[2 * kMaxSources + 2];

  // Independent of the preceding kernel: weights and stored block rows.
  float w[kVec], v[NS][kVec];
  {
    float a[kVec], b[kVec];
    load8(norm_w + h, a);
    load8(qk_w + h, b);
#pragma unroll
    for (int i = 0; i < kVec; ++i) w[i] = a[i] * b[i];
  }
  const long row = static_cast<long>(t) * block_stride_m;
#pragma unroll
  for (int s = 0; s < NB; ++s) load8(blocks + row + s * block_stride_r + h, v[s]);
  float ow[kVec];
  if constexpr (OUT_NORM) load8(out_w + h, ow);

  asm volatile("griddepcontrol.wait;" ::: "memory");
  K3_MARK(1)

  // Multicast this rank's partial into slot `rank` of every rank's mailbox.
  {
    U4 p = *reinterpret_cast<const U4*>(partial + static_cast<long>(t) * kHidden + h);
#pragma unroll
    for (int i = 0; i < 4; ++i) {  // -0.0 is the Lamport sentinel: send +0.0 instead
      if ((p.v[i] & 0xffffu) == 0x8000u) p.v[i] &= 0xffff0000u;
      if ((p.v[i] >> 16) == 0x8000u) p.v[i] &= 0x0000ffffu;
    }
    const long off = (static_cast<long>(rank) * kMaxM + t) * kHidden + h;
    multimem_st(mailbox_mc + off * sizeof(__nv_bfloat16), p);
  }
  K3_MARK(2)
  // Poll all tp slots at once (one round trip instead of tp dependent ones),
  // re-polling only slots still holding sentinels; then reduce in rank order
  // (identical on every rank) and re-arm the slots.
  constexpr int kMaxTp = 16;
  U4 u[kMaxTp];
  unsigned pending = 0;
#pragma unroll
  for (int r = 0; r < kMaxTp; ++r) {
    if (r < tp) {
      u[r] = ld_volatile(mailbox + (static_cast<long>(r) * kMaxM + t) * kHidden + h);
      pending |= (has_sentinel(u[r]) ? 1u : 0u) << r;
    }
  }
  while (pending) {
#pragma unroll
    for (int r = 0; r < kMaxTp; ++r) {
      if (pending & (1u << r)) {
        u[r] = ld_volatile(mailbox + (static_cast<long>(r) * kMaxM + t) * kHidden + h);
        if (!has_sentinel(u[r])) pending &= ~(1u << r);
      }
    }
  }
  K3_MARK(3)
  float acc[kVec] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  U4 sentinel;
#pragma unroll
  for (int i = 0; i < 4; ++i) sentinel.v[i] = kNegZero2;
#pragma unroll
  for (int r = 0; r < kMaxTp; ++r) {
    if (r < tp) {
      float f[kVec];
      unpack(u[r], f);
#pragma unroll
      for (int i = 0; i < kVec; ++i) acc[i] += f[i];
      *reinterpret_cast<U4*>(mailbox + (static_cast<long>(r) * kMaxM + t) * kHidden + h) = sentinel;
    }
  }
  asm volatile("griddepcontrol.launch_dependents;");

  // attn = bf16(sum); prefix' = bf16(prefix + attn) or attn.
  const long prow = static_cast<long>(t) * kHidden + h;
  {
    const U4 attn_b = pack(acc);
    float attn[kVec];
    unpack(attn_b, attn);
    if constexpr (HAS_DELTA) {
      float pre[kVec];
      load8(prefix + prow, pre);
#pragma unroll
      for (int i = 0; i < kVec; ++i) v[NB][i] = __bfloat162float(__float2bfloat16_rn(pre[i] + attn[i]));
    } else {
#pragma unroll
      for (int i = 0; i < kVec; ++i) v[NB][i] = attn[i];
    }
    *reinterpret_cast<U4*>(prefix + prow) = pack(v[NB]);
  }
  if constexpr (WRITE_BLOCK)
    *reinterpret_cast<U4*>(blocks + row + write_idx * block_stride_r + h) = pack(v[NB]);

  float logit[NS];
  if constexpr (NS > 1) {
#pragma unroll
    for (int s = 0; s < NS; ++s) {
      float sq = 0.f, dt = 0.f;
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        sq += v[s][i] * v[s][i];
        dt += v[s][i] * w[i];
      }
      sq = warp_sum(sq);
      dt = warp_sum(dt);
      if (lane == 0) {
        warp_part[warp][s] = sq;
        warp_part[warp][NS + s] = dt;
      }
    }
    __syncthreads();
    if (threadIdx.x < 2 * NS) {
      float a = 0.f;
#pragma unroll
      for (int k = 0; k < kWarps; ++k) a += warp_part[k][threadIdx.x];
      cta_part[threadIdx.x] = a;
    }
    cluster.sync();
#pragma unroll
    for (int s = 0; s < NS; ++s) {
      float sq = 0.f, dt = 0.f;
#pragma unroll
      for (int r = 0; r < kCluster; ++r) {
        const float* remote = cluster.map_shared_rank(cta_part, r);
        sq += remote[s];
        dt += remote[NS + s];
      }
      logit[s] = dt * rsqrtf(sq * (1.0f / kHidden) + eps);
    }
  }

  float mixed[kVec];
  if constexpr (NS == 1) {
#pragma unroll
    for (int i = 0; i < kVec; ++i) mixed[i] = v[0][i];
  } else {
    float mx = logit[0];
#pragma unroll
    for (int s = 1; s < NS; ++s) mx = fmaxf(mx, logit[s]);
    float p[NS], denom = 0.f;
#pragma unroll
    for (int s = 0; s < NS; ++s) {
      p[s] = __expf(logit[s] - mx);
      denom += p[s];
    }
    const float inv = 1.0f / denom;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      float a = 0.f;
#pragma unroll
      for (int s = 0; s < NS; ++s) a += p[s] * v[s][i];
      mixed[i] = a * inv;
    }
  }
  if constexpr (OUT_NORM) {
    float sq = 0.f;
#pragma unroll
    for (int i = 0; i < kVec; ++i) sq += mixed[i] * mixed[i];
    sq = warp_sum(sq);
    if (lane == 0) warp_part[warp][2 * kMaxSources + 1] = sq;
    __syncthreads();
    if (threadIdx.x == 0) {
      float a = 0.f;
#pragma unroll
      for (int k = 0; k < kWarps; ++k) a += warp_part[k][2 * kMaxSources + 1];
      cta_part[2 * kMaxSources + 1] = a;
    }
    cluster.sync();
    float total = 0.f;
#pragma unroll
    for (int r = 0; r < kCluster; ++r) total += cluster.map_shared_rank(cta_part, r)[2 * kMaxSources + 1];
    const float rstd = rsqrtf(total * (1.0f / kHidden) + out_eps);
#pragma unroll
    for (int i = 0; i < kVec; ++i) mixed[i] = mixed[i] * rstd * ow[i];
  }
  *reinterpret_cast<U4*>(out + prow) = pack(mixed);
  K3_MARK(4)
  cluster.sync();
#undef K3_MARK
}

template <int NS>
void launch_ns(bool has_delta, bool write_block, bool out_norm, dim3 grid, cudaStream_t stream,
               const __nv_bfloat16* partial, __nv_bfloat16* mailbox, uint64_t mc, int rank, int tp,
               __nv_bfloat16* prefix, __nv_bfloat16* blocks, const __nv_bfloat16* nw,
               const __nv_bfloat16* qw, const __nv_bfloat16* ow, __nv_bfloat16* out, long bsm,
               long bsr, int widx, float eps, float oeps, unsigned long long* trace) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(kThreads);
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
#define K3_L(D, W, O)                                                                        \
  cudaLaunchKernelEx(&cfg, ar_attn_res_kernel<NS, D, W, O>, partial, mailbox, mc, rank, tp, \
                     prefix, blocks, nw, qw, ow, out, bsm, bsr, widx, eps, oeps, trace)
  if (has_delta) {
    if (write_block) { out_norm ? K3_L(true, true, true) : K3_L(true, true, false); }
    else { out_norm ? K3_L(true, false, true) : K3_L(true, false, false); }
  } else {
    if (write_block) { out_norm ? K3_L(false, true, true) : K3_L(false, true, false); }
    else { out_norm ? K3_L(false, false, true) : K3_L(false, false, false); }
  }
#undef K3_L
}

}  // namespace

void ar_attn_res(torch::Tensor partial, torch::Tensor mailbox, int64_t mailbox_mc, int64_t parity,
                 int64_t rank, int64_t tp, torch::Tensor prefix, bool has_delta, torch::Tensor blocks,
                 torch::Tensor norm_weight, torch::Tensor qk_weight,
                 std::optional<torch::Tensor> output_norm_weight, torch::Tensor out,
                 int64_t num_blocks, int64_t block_write_idx, double eps, double output_norm_eps,
                 std::optional<torch::Tensor> trace) {
  unsigned long long* trace_ptr = trace ? reinterpret_cast<unsigned long long*>(trace->data_ptr()) : nullptr;
  const int M = partial.size(0);
  TORCH_CHECK(M >= 1 && M <= kMaxM && partial.size(1) == kHidden && partial.is_contiguous());
  TORCH_CHECK(prefix.stride(0) == kHidden && out.stride(0) == kHidden && blocks.stride(-1) == 1);
  TORCH_CHECK(num_blocks >= 0 && num_blocks < kMaxSources && tp >= 1 && tp <= 16);
  // Two buffers, alternated by the caller, so a rank can never overwrite a slot
  // that a slower rank has not consumed yet, even with back-to-back calls.
  TORCH_CHECK(mailbox.numel() >= 2 * tp * kMaxM * kHidden && (parity == 0 || parity == 1));
  const long buf_elems = parity * tp * kMaxM * kHidden;
  auto bf = [](const torch::Tensor& x) { return reinterpret_cast<__nv_bfloat16*>(x.data_ptr()); };
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  dim3 grid(kCluster, M);
#define K3_NS(N)                                                                                \
  case N - 1:                                                                                   \
    launch_ns<N>(has_delta, block_write_idx >= 0, output_norm_weight.has_value(), grid, stream, \
                 bf(partial), bf(mailbox) + buf_elems,                                          \
                 static_cast<uint64_t>(mailbox_mc) + buf_elems * sizeof(__nv_bfloat16), (int)rank, (int)tp, \
                 bf(prefix), bf(blocks), bf(norm_weight), bf(qk_weight),                         \
                 output_norm_weight ? bf(*output_norm_weight) : nullptr, bf(out),                \
                 blocks.stride(0), blocks.stride(1), (int)block_write_idx, (float)eps,           \
                 (float)output_norm_eps, trace_ptr);                                             \
    break;
  switch (num_blocks) {
    K3_NS(1) K3_NS(2) K3_NS(3) K3_NS(4) K3_NS(5) K3_NS(6) K3_NS(7) K3_NS(8) K3_NS(9)
    default: TORCH_CHECK(false, "unsupported num_blocks");
  }
#undef K3_NS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

TORCH_LIBRARY(k3ar, m) {
  m.def("ar_attn_res(Tensor partial, Tensor(a!) mailbox, int mailbox_mc, int parity, int rank, int tp, "
        "Tensor(b!) prefix, bool has_delta, Tensor(c!) blocks, Tensor norm_weight, Tensor qk_weight, "
        "Tensor? output_norm_weight, Tensor(d!) out, int num_blocks, int block_write_idx, "
        "float eps, float output_norm_eps, Tensor(e!)? trace=None) -> ()");
}
TORCH_LIBRARY_IMPL(k3ar, CUDA, m) { m.impl("ar_attn_res", &ar_attn_res); }
