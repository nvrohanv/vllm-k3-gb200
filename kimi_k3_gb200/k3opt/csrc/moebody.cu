// moebody (W1-4): torch op k3moebody::expert_body = kernels wrapping the device code in moebody_dev.cuh.
// See moebody_dev.cuh for the design, and the integration notes in agents/moe8/body (final report).
#include <algorithm>
#include <vector>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#include "moebody_dev.cuh"

namespace moebody {

// ---------------------------------------------------------------- kernels (thin wrappers of the device entry points)
template <int MT>
__global__ void __launch_bounds__(kThreads, 1)
body_kernel(const __grid_constant__ CUtensorMap tmA1, const __grid_constant__ CUtensorMap tmA2,
            const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) uint8_t sm[];
  group_body<MT>(sm, tmA1, tmA2, p);
}
__global__ void __launch_bounds__(kThreads, 1)
sk_kernel(const __grid_constant__ CUtensorMap tmA1, const __grid_constant__ CUtensorMap tmA2,
          const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) uint8_t sm[];
  sk_body(sm, tmA1, tmA2, p);
}

// ---------------------------------------------------------------- device-function integration test (op arg harness=1)
// A 13th warp plays the MoE front end: it waits for the routing (epoch flag, or griddepcontrol.wait), copies
// the ids into shared memory after the body's region and completes an mbarrier; the body runs on threads
// 0..383 with routing source (c). Checks the device-function contract (foreign warps, named barriers, smem ids).
constexpr int kFrontThreads = kThreads + 32;
constexpr int kFrontSmem = kSmemBytes + 1024;
__device__ __forceinline__ void front_warp(uint8_t* sm, const Params& p, const unsigned long long* flag, const int* ids_g,
                                           int npairs) {
  const int lane = threadIdx.x % 32;
  int* ids_s = reinterpret_cast<int*>(sm + kSmemBytes);
  uint64_t* bar = reinterpret_cast<uint64_t*>(sm + kSmemBytes + 512);
  if (flag != nullptr) {
    if (lane == 0) {
      const unsigned long long target = static_cast<unsigned long long>(~p.ctr[blockIdx.x]) + 1ull;
      while (ld_acquire_u64(flag) < target) {
      }
    }
    __syncwarp();
  } else {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
  for (int i = lane; i < npairs; i += 32) ids_s[i] = ld_relaxed_i32(ids_g + i);
  __syncwarp();
  if (lane == 0) tc::mbar_arrive(bar);  // release (CTA scope): the ids stores are visible to the waiters
}
__device__ __forceinline__ Params front_params(uint8_t* sm, const Params& p) {
  if (threadIdx.x == 0) tc::mbar_init(reinterpret_cast<uint64_t*>(sm + kSmemBytes + 512), 1);
  __syncthreads();  // all 416 threads: the ids barrier is initialized before anyone waits on it
  Params q = p;
  q.ids_smem = reinterpret_cast<const int*>(sm + kSmemBytes);
  q.ids_bar = tc::su32(sm + kSmemBytes + 512);
  q.ids_phase = 0;
  q.flag = nullptr;
  q.no_griddep = 1;
  return q;
}
template <int MT>
__global__ void __launch_bounds__(kFrontThreads, 1)
front_group_kernel(const __grid_constant__ CUtensorMap tmA1, const __grid_constant__ CUtensorMap tmA2,
                   const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const Params q = front_params(sm, p);
  if (threadIdx.x >= kThreads) front_warp(sm, p, p.flag, p.ids, p.M * kTopK);
  else group_body<MT>(sm, tmA1, tmA2, q);
}
__global__ void __launch_bounds__(kFrontThreads, 1)
front_sk_kernel(const __grid_constant__ CUtensorMap tmA1, const __grid_constant__ CUtensorMap tmA2,
                const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const Params q = front_params(sm, p);
  if (threadIdx.x >= kThreads) front_warp(sm, p, p.flag, p.ids, p.M * kTopK);
  else sk_body(sm, tmA1, tmA2, q);
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

// 3D fp4 view: {128 elements (64 B) of K, rows, K chunks}; box {128, 128, 2} -> smem [2][128][128 B].
CUtensorMap fp4_map(const void* base, uint64_t rows, uint64_t row_bytes, uint64_t kchunks, uint64_t kstride,
                    CUtensorMapL2promotion promo) {
  CUtensorMap m;
  const cuuint64_t dims[3] = {128, rows, kchunks};
  const cuuint64_t strides[2] = {row_bytes, kstride};
  const cuuint32_t box[3] = {128, 128, 2};
  const cuuint32_t es[3] = {1, 1, 1};
  const CUresult r = encoder()(&m, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, 3, const_cast<void*>(base), dims, strides,
                               box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, promo,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
  return m;
}

struct MapEntry {
  const void* w13;
  const void* w2;
  long E;
  CUtensorMap a1, a2;
};

template <int MT>
void set_attrs() {
  C10_CUDA_CHECK(cudaFuncSetAttribute(body_kernel<MT>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  C10_CUDA_CHECK(cudaFuncSetAttribute(body_kernel<MT>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  C10_CUDA_CHECK(cudaFuncSetAttribute(front_group_kernel<MT>, cudaFuncAttributeMaxDynamicSharedMemorySize, kFrontSmem));
  C10_CUDA_CHECK(cudaFuncSetAttribute(front_group_kernel<MT>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
}

int max_clusters(int L) {
  static int cache[17] = {0};
  if (cache[L] == 0) {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(L * 64);
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = kSmemBytes;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = L;
    at[0].val.clusterDim.y = 1;
    at[0].val.clusterDim.z = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    int n = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&n, body_kernel<8>, &cfg));
    TORCH_CHECK(n > 0, "no co-resident clusters of size ", L);
    cache[L] = n;
  }
  return cache[L];
}

void launch(const torch::Tensor& x, const torch::Tensor& ids, const std::optional<torch::Tensor>& wts,
            const torch::Tensor& w13, const torch::Tensor& w13_scale,
            const torch::Tensor& w2, const torch::Tensor& w2_scale, const torch::Tensor& ws,
            const std::optional<torch::Tensor>& epoch, const torch::Tensor& out, int64_t mode, double beta, double linear_beta,
            int64_t cluster, int64_t force_c, int64_t sk_min_m, int64_t sk_groups, int64_t max_ctas, int64_t harness,
            int64_t x_late, const std::optional<torch::Tensor>& trace, const std::optional<torch::Tensor>& endlog) {
  const int M = x.size(0);
  TORCH_CHECK(M >= 1 && M <= kMaxM, "moebody supports 1..8 tokens");
  TORCH_CHECK(x.size(1) == kHidden && x.is_contiguous() && x.scalar_type() == at::kBFloat16);
  TORCH_CHECK(ids.size(0) == M && ids.size(1) == kTopK && ids.scalar_type() == at::kInt && ids.is_contiguous());
  TORCH_CHECK(w13.dim() == 3 && w13.size(1) == kW13Rows && w13.size(2) == kW13RowBytes && w13.is_contiguous());
  TORCH_CHECK(w2.dim() == 3 && w2.size(1) == kHidden && w2.size(2) == 128 && w2.is_contiguous());
  const long E = w13.size(0);
  TORCH_CHECK(E <= kMaxE && w2.size(0) == E && w13_scale.is_contiguous() && w2_scale.is_contiguous());
  TORCH_CHECK(w13_scale.numel() * w13_scale.element_size() == E * kW13ScaleBytes);
  TORCH_CHECK(w2_scale.numel() * w2_scale.element_size() == E * kW2ScaleBytes);
  TORCH_CHECK(mode >= 0 && mode <= 2, "mode: 0 gemm2, 1 finalized, 2 publish");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.dim() == 2 && out.size(1) == kHidden && out.stride(1) == 1);
  if (mode == 0) {
    TORCH_CHECK(out.size(0) == M * kTopK && out.is_contiguous(), "mode 0: out is gemm2 [M*16, 3584]");
  } else {
    TORCH_CHECK(out.size(0) == M, "modes 1/2: out is [M, 3584] (rows may be strided, e.g. a mailbox slot)");
    TORCH_CHECK(wts.has_value() && wts->size(0) == M && wts->size(1) == kTopK && wts->is_contiguous() &&
                    (wts->scalar_type() == at::kBFloat16 || wts->scalar_type() == at::kFloat),
                "modes 1/2 need wts [M, 16] bf16 or fp32");
  }
  TORCH_CHECK(linear_beta > 0.0 && beta > 0.0);
  static bool init = false;
  if (!init) {
    set_attrs<1>();
    set_attrs<2>();
    set_attrs<4>();
    set_attrs<8>();
    C10_CUDA_CHECK(cudaFuncSetAttribute(sk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    C10_CUDA_CHECK(cudaFuncSetAttribute(front_sk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kFrontSmem));
    init = true;
  }
  const bool sk = M >= sk_min_m;
  static int sms = 0;
  if (sms == 0) C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, x.get_device()));
  int L = static_cast<int>(cluster);
  if (L <= 0) L = M == 1 ? 6 : M == 2 ? 4 : 2;
  TORCH_CHECK(L >= 1 && L <= 16);
  TORCH_CHECK(force_c == 0 || (force_c >= 1 && force_c <= kMaxC && L % force_c == 0), "bad force_c");
  int G = sk ? sms : L * max_clusters(L);
  if (max_ctas > 0 && G > max_ctas) G = sk ? static_cast<int>(max_ctas) : L * static_cast<int>(max_ctas / L);
  TORCH_CHECK(G <= kMaxG);
  TORCH_CHECK(sk_groups >= 0 && sk_groups <= kSkMaxNG);
  static std::vector<MapEntry> cache;
  const MapEntry* mc = nullptr;
  for (const MapEntry& me : cache)
    if (me.w13 == w13.data_ptr() && me.w2 == w2.data_ptr() && me.E == E) mc = &me;
  if (mc == nullptr) {
    MapEntry me;
    me.a1 = fp4_map(w13.data_ptr(), E * kW13Rows, kW13RowBytes, 28, 64, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    // W2: K-chunks at k = 0 and k = 64 (32-B chunk stride), so bytes 96..127 (K padding) are never read.
    me.a2 = fp4_map(w2.data_ptr(), E * kHidden, 128, 2, 32, CU_TENSOR_MAP_L2_PROMOTION_NONE);
    me.w13 = w13.data_ptr();
    me.w2 = w2.data_ptr();
    me.E = E;
    cache.push_back(me);
    mc = &cache.back();
  }
  Params prm;
  prm.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  prm.ids = ids.data_ptr<int>();
  prm.w13s = static_cast<const uint8_t*>(w13_scale.data_ptr());
  prm.w2s = static_cast<const uint8_t*>(w2_scale.data_ptr());
  prm.w2 = static_cast<const uint8_t*>(w2.data_ptr());
  prm.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  prm.M = M;
  prm.E = static_cast<int>(E);
  prm.beta_lb = static_cast<float>(beta * linear_beta);
  prm.inv_beta = static_cast<float>(1.0 / beta);
  prm.inv_lb = static_cast<float>(1.0 / linear_beta);
  TORCH_CHECK(ws.is_contiguous() && ws.numel() * ws.element_size() >= kWsAll,
              "workspace must hold workspace_bytes() bytes, filled with 0xFF once");
  prm.ctr = reinterpret_cast<int*>(ws.data_ptr());
  prm.scratch = static_cast<uint8_t*>(ws.data_ptr()) + kWsScratch;
  prm.flag = nullptr;
  if (epoch.has_value() && epoch->numel() > 0) {
    TORCH_CHECK(epoch->scalar_type() == at::kLong);
    prm.flag = reinterpret_cast<const unsigned long long*>(epoch->data_ptr());
  }
  prm.trace = trace ? reinterpret_cast<unsigned long long*>(trace->data_ptr()) : nullptr;
  prm.force_c = static_cast<int>(force_c);
  prm.sk_groups = static_cast<int>(sk_groups);
  prm.mode = static_cast<int>(mode);
  prm.wts = (mode != 0) ? wts->data_ptr() : nullptr;
  prm.wts_bf16 = (mode != 0 && wts->scalar_type() == at::kBFloat16) ? 1 : 0;
  prm.fout = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  prm.ids_smem = nullptr;
  prm.ids_bar = 0;
  prm.ids_phase = 0;
  prm.no_griddep = 0;
  prm.x_late = static_cast<int>(x_late);
  prm.fstride = out.stride(0);
  prm.endlog = endlog ? reinterpret_cast<unsigned long long*>(endlog->data_ptr()) : nullptr;
  prm.endlog_n = endlog ? static_cast<int>(endlog->numel()) : 1;

  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(G);
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = kSmemBytes;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  attr[1].id = cudaLaunchAttributeClusterDimension;
  attr[1].val.clusterDim.x = L;
  attr[1].val.clusterDim.y = 1;
  attr[1].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = sk ? 1 : 2;
  if (harness == 1) {
    cfg.blockDim = dim3(kFrontThreads);
    cfg.dynamicSmemBytes = kFrontSmem;
    if (sk) C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, front_sk_kernel, mc->a1, mc->a2, prm));
    else if (M == 1) C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, front_group_kernel<1>, mc->a1, mc->a2, prm));
    else if (M == 2) C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, front_group_kernel<2>, mc->a1, mc->a2, prm));
    else if (M <= 4) C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, front_group_kernel<4>, mc->a1, mc->a2, prm));
    else C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, front_group_kernel<8>, mc->a1, mc->a2, prm));
    return;
  }
  if (sk) C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, sk_kernel, mc->a1, mc->a2, prm));
  else if (M == 1) C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, body_kernel<1>, mc->a1, mc->a2, prm));
  else if (M == 2) C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, body_kernel<2>, mc->a1, mc->a2, prm));
  else if (M <= 4) C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, body_kernel<4>, mc->a1, mc->a2, prm));
  else C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, body_kernel<8>, mc->a1, mc->a2, prm));
}

}  // namespace moebody

// mode 0 (gemm2): out[t*16 + j, n] = W2_{e_tj}[n, :192] . h_tj (unweighted), bf16 [M*16, 3584].
void expert_body(torch::Tensor x, torch::Tensor ids, std::optional<torch::Tensor> wts, torch::Tensor w13,
                 torch::Tensor w13_scale, torch::Tensor w2, torch::Tensor w2_scale, torch::Tensor ws,
                 std::optional<torch::Tensor> epoch, torch::Tensor out, int64_t mode, double beta, double linear_beta,
                 int64_t cluster, int64_t force_c, int64_t sk_min_m, int64_t sk_groups, int64_t max_ctas,
                 int64_t harness, int64_t x_late, std::optional<torch::Tensor> trace,
                 std::optional<torch::Tensor> endlog) {
  moebody::launch(x, ids, wts, w13, w13_scale, w2, w2_scale, ws, epoch, out, mode, beta, linear_beta, cluster, force_c,
                  sk_min_m, sk_groups, max_ctas, harness, x_late, trace, endlog);
}

int64_t workspace_bytes() { return moebody::kWsAll; }

TORCH_LIBRARY(k3moebody, m) {
  m.def("expert_body(Tensor x, Tensor ids, Tensor? wts, Tensor w13, Tensor w13_scale, Tensor w2, Tensor w2_scale, "
        "Tensor(a!) ws, Tensor(b!)? epoch, Tensor(c!) out, int mode, float beta, float linear_beta, int cluster=0, "
        "int force_c=0, int sk_min_m=3, int sk_groups=0, int max_ctas=0, int harness=0, int x_late=1, "
        "Tensor(d!)? trace=None, Tensor(e!)? endlog=None) -> ()");
  m.def("workspace_bytes() -> int");
}
TORCH_LIBRARY_IMPL(k3moebody, CUDA, m) { m.impl("expert_body", &expert_body); }
TORCH_LIBRARY_IMPL(k3moebody, CompositeExplicitAutograd, m) { m.impl("workspace_bytes", &workspace_bytes); }
