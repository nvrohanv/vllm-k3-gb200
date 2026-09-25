// k3step.decode_prep (W1-6 stepov, OPTIONAL, separate library fragment so that k3step.cu stays as tested):
// vLLM's _prepare_pos_seq_lens_kernel + _combine_sampled_and_draft_tokens_kernel for batches without
// draft tokens (one new sampled token per request) in one launch.  Block b < num_reqs handles request b;
// block num_reqs zero-pads seq_lens[num_reqs:] (for FULL CUDA graphs), exactly like vLLM.
// Used by stepov_patch.py's prepare_inputs patch whenever this library is loaded (K3STEPOV_DECODE_PREP=0
// disables it).  Tested: test_k3step.py test_decode_prep (40 cases) and test_stepov_hooks.py
// test_decode_prep_hook, bit-identical to vLLM.
#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

namespace {

struct IView {
  const void* p;
  int64_t stride;
  int is64;
};
__device__ __forceinline__ int64_t rd(const IView& v, int64_t i) {
  return v.is64 ? static_cast<const int64_t*>(v.p)[i * v.stride]
                : static_cast<int64_t>(static_cast<const int32_t*>(v.p)[i * v.stride]);
}
IView iview(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.dim() >= 1 &&
                  (t.scalar_type() == at::kInt || t.scalar_type() == at::kLong),
              name, " must be a CUDA int32 / int64 tensor");
  return IView{t.data_ptr(), t.stride(0), t.scalar_type() == at::kLong ? 1 : 0};
}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

template <typename... KArgs, typename... Args>
void launch_pdl(void (*kernel)(KArgs...), dim3 grid, dim3 block, cudaStream_t stream, Args&&... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...));
}

// decode_prep: vLLM's _prepare_pos_seq_lens_kernel + _combine_sampled_and_draft_tokens_kernel
// (no draft tokens, one new sampled token per request) in one launch.  Block b < num_reqs handles
// request b; block num_reqs zero-pads seq_lens[num_reqs:] (for FULL CUDA graphs), like vLLM.
// ------------------------------------------------------------------------------------------
constexpr int kDpThreads = 128;

__global__ void __launch_bounds__(kDpThreads)
    decode_prep_kernel(IView idx, IView qsl, const int32_t* __restrict__ num_computed,
                       const int64_t* __restrict__ last_sampled, int64_t ls_stride, IView prefill_len,
                       IView cu, int64_t* __restrict__ positions, int32_t* __restrict__ seq_lens,
                       int seq_lens_len, int32_t* __restrict__ input_ids,
                       int64_t* __restrict__ logits_indices, int num_reqs) {
  pdl_wait();
  const int b = blockIdx.x;
  if (b == num_reqs) {
    for (int i = num_reqs + threadIdx.x; i < seq_lens_len; i += kDpThreads) seq_lens[i] = 0;
    return;
  }
  const int64_t req = rd(idx, b);
  const int32_t nc = num_computed[req];
  const int32_t start = static_cast<int32_t>(rd(qsl, b));
  const int32_t end = static_cast<int32_t>(rd(qsl, b + 1));
  const int32_t query_len = end - start;
  const int32_t seq_len = nc + query_len;
  for (int32_t i = threadIdx.x; i < query_len; i += kDpThreads)
    positions[start + i] = static_cast<int64_t>(nc + i);
  if (threadIdx.x != 0) return;
  seq_lens[b] = seq_len;
  const int32_t c0 = static_cast<int32_t>(rd(cu, b));
  const int32_t num_logits = static_cast<int32_t>(rd(cu, b + 1)) - c0;
  const int32_t logits_start = end - num_logits;
  for (int32_t j = 0; j < num_logits; ++j) logits_indices[c0 + j] = static_cast<int64_t>(logits_start + j);
  const int32_t pl = static_cast<int32_t>(rd(prefill_len, req));
  if (seq_len <= pl) return;  // prefill tokens: no sampled token to insert
  if (seq_len - num_logits >= pl)
    input_ids[logits_start] = static_cast<int32_t>(last_sampled[req * ls_stride]);
}

void decode_prep(const torch::Tensor& idx_mapping, const torch::Tensor& query_start_loc,
                 const torch::Tensor& num_computed_tokens, const torch::Tensor& last_sampled_tokens,
                 const torch::Tensor& prefill_len, const torch::Tensor& cu_num_logits,
                 torch::Tensor& positions, torch::Tensor& seq_lens, torch::Tensor& input_ids,
                 torch::Tensor& logits_indices) {
  const int num_reqs = static_cast<int>(idx_mapping.numel());
  TORCH_CHECK(num_reqs >= 1 && idx_mapping.dim() == 1, "idx_mapping must be 1-D, non-empty");
  TORCH_CHECK(query_start_loc.numel() >= num_reqs + 1 && cu_num_logits.numel() >= num_reqs + 1,
              "query_start_loc / cu_num_logits too short");
  TORCH_CHECK(num_computed_tokens.is_cuda() && num_computed_tokens.scalar_type() == at::kInt &&
                  num_computed_tokens.is_contiguous(),
              "num_computed_tokens must be contiguous int32");
  TORCH_CHECK(last_sampled_tokens.is_cuda() && last_sampled_tokens.scalar_type() == at::kLong &&
                  last_sampled_tokens.dim() == 2,
              "last_sampled_tokens must be int64 [R, 1]");
  TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == at::kLong && positions.is_contiguous(),
              "positions must be contiguous int64");
  TORCH_CHECK(seq_lens.is_cuda() && seq_lens.scalar_type() == at::kInt && seq_lens.is_contiguous() &&
                  seq_lens.numel() >= num_reqs,
              "seq_lens must be contiguous int32 [>= num_reqs]");
  TORCH_CHECK(input_ids.is_cuda() && input_ids.scalar_type() == at::kInt && input_ids.is_contiguous(),
              "input_ids must be contiguous int32");
  TORCH_CHECK(logits_indices.is_cuda() && logits_indices.scalar_type() == at::kLong &&
                  logits_indices.is_contiguous(),
              "logits_indices must be contiguous int64");
  launch_pdl(decode_prep_kernel, dim3(num_reqs + 1), dim3(kDpThreads), at::cuda::getCurrentCUDAStream().stream(),
             iview(idx_mapping, "idx_mapping"), iview(query_start_loc, "query_start_loc"),
             static_cast<const int32_t*>(num_computed_tokens.data_ptr<int32_t>()),
             static_cast<const int64_t*>(last_sampled_tokens.data_ptr<int64_t>()),
             static_cast<int64_t>(last_sampled_tokens.stride(0)), iview(prefill_len, "prefill_len"),
             iview(cu_num_logits, "cu_num_logits"), positions.data_ptr<int64_t>(),
             seq_lens.data_ptr<int32_t>(), static_cast<int>(seq_lens.numel()),
             input_ids.data_ptr<int32_t>(), logits_indices.data_ptr<int64_t>(), num_reqs);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(k3step, m) {
  m.def("decode_prep(Tensor idx_mapping, Tensor query_start_loc, Tensor num_computed_tokens, "
        "Tensor last_sampled_tokens, Tensor prefill_len, Tensor cu_num_logits, Tensor(a!) positions, "
        "Tensor(b!) seq_lens, Tensor(c!) input_ids, Tensor(d!) logits_indices) -> ()");
}

TORCH_LIBRARY_IMPL(k3step, CUDA, m) {
  m.impl("decode_prep", &decode_prep);
}
