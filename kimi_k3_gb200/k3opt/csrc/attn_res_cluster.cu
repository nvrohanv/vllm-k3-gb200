// Kimi K3 AttnRes for decode-sized batches: one 7-CTA thread-block cluster per
// token. Each CTA owns 1024 of the 7168 hidden elements (128 threads x 8 bf16),
// so every source row is read with a single 16-byte load per thread, and the
// per-source norm/dot partials and the output RMSNorm are combined across the
// cluster through distributed shared memory.
//
// Semantics match vllm/models/kimi_k3/nvidia/ops/attn_res.py (_attn_res_kernel):
//   prefix' = bf16(prefix + delta)            (stored back when delta is given)
//   blocks[t, write_idx] = prefix'            (when write_idx >= 0)
//   sources = blocks[t, 0..num_blocks-1], prefix'
//   logit_s = dot(v_s, norm_w * qk_w) * rsqrt(mean(v_s^2) + eps)
//   mixed   = sum_s softmax(logit)_s * v_s
//   out     = mixed * rsqrt(mean(mixed^2) + out_eps) * out_w   (if out_w given)
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

namespace cg = cooperative_groups;

namespace {

constexpr int kHidden = 7168;
constexpr int kClusterSize = 7;
constexpr int kThreads = 128;
constexpr int kVec = 8;  // bf16 per 16-byte load
static_assert(kClusterSize * kThreads * kVec == kHidden, "tiling must cover hidden");
constexpr int kWarps = kThreads / 32;
constexpr int kMaxSources = 9;

struct alignas(16) Bf16x8 {
  __nv_bfloat162 h[4];
};

__device__ __forceinline__ void load8(const __nv_bfloat16* p, float (&v)[kVec]) {
  Bf16x8 x = *reinterpret_cast<const Bf16x8*>(p);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 f = __bfloat1622float2(x.h[i]);
    v[2 * i] = f.x;
    v[2 * i + 1] = f.y;
  }
}

__device__ __forceinline__ void store8(__nv_bfloat16* p, const float (&v)[kVec]) {
  Bf16x8 x;
#pragma unroll
  for (int i = 0; i < 4; ++i) x.h[i] = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
  *reinterpret_cast<Bf16x8*>(p) = x;
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o);
  return x;
}

template <int NS, bool HAS_DELTA, bool WRITE_BLOCK, bool OUT_NORM>
__global__ void __cluster_dims__(kClusterSize, 1, 1) __launch_bounds__(kThreads)
attn_res_cluster_kernel(__nv_bfloat16* __restrict__ prefix,
                        const __nv_bfloat16* __restrict__ delta,
                        __nv_bfloat16* __restrict__ blocks,
                        const __nv_bfloat16* __restrict__ norm_w,
                        const __nv_bfloat16* __restrict__ qk_w,
                        const __nv_bfloat16* __restrict__ out_w,
                        __nv_bfloat16* __restrict__ out, long block_stride_m,
                        long block_stride_r, int write_idx, float eps, float out_eps) {
  constexpr int NB = NS - 1;  // stored block sources; the last source is prefix'
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = cluster.block_rank();
  const int token = blockIdx.y;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int h = (rank * kThreads + threadIdx.x) * kVec;

  // [0, NS): partial sum of squares, [NS, 2NS): partial dot, [2NS]: mixed^2.
  __shared__ float warp_part[kWarps][2 * kMaxSources + 2];
  __shared__ float cta_part[2 * kMaxSources + 2];

  // Weights and previously written block rows do not depend on the preceding
  // kernel, so load them before waiting on it (PDL).
  float w[kVec], v[NS][kVec];
  {
    float a[kVec], b[kVec];
    load8(norm_w + h, a);
    load8(qk_w + h, b);
#pragma unroll
    for (int i = 0; i < kVec; ++i) w[i] = a[i] * b[i];
  }
  const long row = static_cast<long>(token) * block_stride_m;
#pragma unroll
  for (int s = 0; s < NB; ++s) load8(blocks + row + s * block_stride_r + h, v[s]);
  float ow[kVec];
  if constexpr (OUT_NORM) load8(out_w + h, ow);

  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");

  const long prow = static_cast<long>(token) * kHidden + h;
  load8(prefix + prow, v[NB]);
  if constexpr (HAS_DELTA) {
    float d[kVec];
    load8(delta + prow, d);
#pragma unroll
    for (int i = 0; i < kVec; ++i)
      v[NB][i] = __bfloat162float(__float2bfloat16_rn(v[NB][i] + d[i]));
    store8(prefix + prow, v[NB]);
  }
  if constexpr (WRITE_BLOCK) store8(blocks + row + write_idx * block_stride_r + h, v[NB]);

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
      float acc = 0.f;
#pragma unroll
      for (int k = 0; k < kWarps; ++k) acc += warp_part[k][threadIdx.x];
      cta_part[threadIdx.x] = acc;
    }
    cluster.sync();
    // Every CTA sums the cluster partials in the same order, so all CTAs
    // compute bitwise-identical logits.
#pragma unroll
    for (int s = 0; s < NS; ++s) {
      float sq = 0.f, dt = 0.f;
#pragma unroll
      for (int r = 0; r < kClusterSize; ++r) {
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
      float acc = 0.f;
#pragma unroll
      for (int s = 0; s < NS; ++s) acc += p[s] * v[s][i];
      mixed[i] = acc * inv;
    }
  }

  if constexpr (OUT_NORM) {
    float sq = 0.f;
#pragma unroll
    for (int i = 0; i < kVec; ++i) sq += mixed[i] * mixed[i];
    sq = warp_sum(sq);
    // A second exchange slot keeps this reduction independent of the first.
    if (lane == 0) warp_part[warp][2 * kMaxSources + 1] = sq;
    __syncthreads();
    if (threadIdx.x == 0) {
      float acc = 0.f;
#pragma unroll
      for (int k = 0; k < kWarps; ++k) acc += warp_part[k][2 * kMaxSources + 1];
      cta_part[2 * kMaxSources + 1] = acc;
    }
    cluster.sync();
    float total = 0.f;
#pragma unroll
    for (int r = 0; r < kClusterSize; ++r)
      total += cluster.map_shared_rank(cta_part, r)[2 * kMaxSources + 1];
    const float rstd = rsqrtf(total * (1.0f / kHidden) + out_eps);
#pragma unroll
    for (int i = 0; i < kVec; ++i) mixed[i] = mixed[i] * rstd * ow[i];
  }
  store8(out + prow, mixed);
  // Keep shared memory alive until every CTA in the cluster has read it.
  cluster.sync();
}

template <int NS>
void dispatch_flags(bool has_delta, bool write_block, bool out_norm, dim3 grid,
                    cudaStream_t stream, __nv_bfloat16* prefix, const __nv_bfloat16* delta,
                    __nv_bfloat16* blocks, const __nv_bfloat16* norm_w, const __nv_bfloat16* qk_w,
                    const __nv_bfloat16* out_w, __nv_bfloat16* out, long bsm, long bsr,
                    int write_idx, float eps, float out_eps) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(kThreads);
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
#define K3_LAUNCH(D, W, O)                                                                   \
  cudaLaunchKernelEx(&cfg, attn_res_cluster_kernel<NS, D, W, O>, prefix, delta, blocks,      \
                     norm_w, qk_w, out_w, out, bsm, bsr, write_idx, eps, out_eps)
  if (has_delta) {
    if (write_block) { out_norm ? K3_LAUNCH(true, true, true) : K3_LAUNCH(true, true, false); }
    else { out_norm ? K3_LAUNCH(true, false, true) : K3_LAUNCH(true, false, false); }
  } else {
    if (write_block) { out_norm ? K3_LAUNCH(false, true, true) : K3_LAUNCH(false, true, false); }
    else { out_norm ? K3_LAUNCH(false, false, true) : K3_LAUNCH(false, false, false); }
  }
#undef K3_LAUNCH
}

}  // namespace

void attn_res_cluster(torch::Tensor prefix, std::optional<torch::Tensor> delta,
                      torch::Tensor blocks, torch::Tensor norm_weight, torch::Tensor qk_weight,
                      std::optional<torch::Tensor> output_norm_weight, torch::Tensor output,
                      int64_t num_blocks, int64_t block_write_idx, double eps,
                      double output_norm_eps) {
  TORCH_CHECK(prefix.size(1) == kHidden && prefix.stride(0) == kHidden && output.stride(0) == kHidden);
  TORCH_CHECK(!delta || delta->stride(0) == kHidden);
  TORCH_CHECK(blocks.stride(-1) == 1 && num_blocks >= 0 && num_blocks < kMaxSources);
  const int T = prefix.size(0);
  if (T == 0) return;
  auto bf = [](const torch::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr()); };
  dim3 grid(kClusterSize, T);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
#define K3_NS(N)                                                                              \
  case N - 1:                                                                                 \
    dispatch_flags<N>(delta.has_value(), block_write_idx >= 0, output_norm_weight.has_value(), \
                      grid, stream, bf(prefix), delta ? bf(*delta) : nullptr, bf(blocks),      \
                      bf(norm_weight), bf(qk_weight),                                          \
                      output_norm_weight ? bf(*output_norm_weight) : nullptr, bf(output),      \
                      blocks.stride(0), blocks.stride(1), (int)block_write_idx, (float)eps,    \
                      (float)output_norm_eps);                                                 \
    break;
  switch (num_blocks) {
    K3_NS(1) K3_NS(2) K3_NS(3) K3_NS(4) K3_NS(5) K3_NS(6) K3_NS(7) K3_NS(8) K3_NS(9)
    default: TORCH_CHECK(false, "unsupported num_blocks");
  }
#undef K3_NS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

TORCH_LIBRARY(k3opt, m) {
  m.def("attn_res_cluster(Tensor(a!) prefix, Tensor? delta, Tensor(b!) blocks, Tensor norm_weight, "
        "Tensor qk_weight, Tensor? output_norm_weight, Tensor(c!) output, int num_blocks, "
        "int block_write_idx, float eps, float output_norm_eps) -> ()");
}
TORCH_LIBRARY_IMPL(k3opt, CUDA, m) { m.impl("attn_res_cluster", &attn_res_cluster); }
