// k3step.decode_prep_attn additionally does BlockTables.gather_block_tables + compute_slot_mappings.
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

// ------------------------------------------------------------------------------------------
// decode_prep_attn: decode_prep + vLLM's _gather_block_tables_kernel + _compute_slot_mappings_kernel
// (BlockTables.gather_block_tables / compute_slot_mappings, context parallelism off) in one launch.
// grid = (num_reqs_padded + 1, KV-cache groups); block (b, g):
//   b <  num_reqs         request b, group g: copy the request's block-table row (num_blocks entries)
//                         into row b of the forward block table, and the slot ids of its query tokens
//                         (from the request's source block table, like vLLM); g == 0 also writes
//                         positions / seq_lens / logits_indices / input_ids (as decode_prep)
//   num_reqs <= b < num_reqs_padded   zero row b of group g's forward block table (graph padding)
//   b == num_reqs_padded  (g == 0) zero seq_lens[num_reqs:]
//   every block           a grid-strided share of slot_mappings[g][qsl[num_reqs]:max_tokens] = PAD
// ------------------------------------------------------------------------------------------
constexpr int kDaThreads = 256;
constexpr int kMaxGroups = 8;

struct BTArgs {
  int groups;
  const uint64_t* src_ptrs;          // [G] request block tables (int32 [max_reqs, stride])
  const uint64_t* dst_ptrs;          // [G] forward block tables (same stride)
  const int64_t* strides;            // [G]
  const int32_t* num_blocks;         // [G, max_reqs] (host-mapped)
  int64_t num_blocks_stride;
  const int32_t* kernel_block_sizes; // [G]
  const bool* slot_enabled;          // [G]
  int64_t* slot_mappings;            // [G, max_tokens]
  int64_t sm_stride;
  int max_tokens;
  int64_t pad_id;
  int num_reqs_padded;
};

__global__ void __launch_bounds__(kDaThreads)
    decode_prep_attn_kernel(IView idx, IView qsl, const int32_t* __restrict__ num_computed,
                            const int64_t* __restrict__ last_sampled, int64_t ls_stride, IView prefill_len,
                            IView cu, int64_t* __restrict__ positions, int32_t* __restrict__ seq_lens,
                            int seq_lens_len, int32_t* __restrict__ input_ids,
                            int64_t* __restrict__ logits_indices, int num_reqs, const BTArgs bt) {
  // grid = (num_reqs_padded + 1, groups): block (b, g) does request b's work for KV-cache group g
  // (g == 0 also does the group-independent decode_prep part)
  pdl_wait();
  const int b = blockIdx.x, g = blockIdx.y;
  const int64_t stride = bt.strides[g];
  int32_t* const dst_base = reinterpret_cast<int32_t*>(bt.dst_ptrs[g]);
  int64_t* const sm = bt.slot_mappings + g * bt.sm_stride;
  {  // slot-mapping padding of group g, shared by the blocks of group g
    const int32_t actual = static_cast<int32_t>(rd(qsl, num_reqs));
    const int nthr = gridDim.x * kDaThreads;
    for (int i = actual + b * kDaThreads + threadIdx.x; i < bt.max_tokens; i += nthr) sm[i] = bt.pad_id;
  }
  if (b == bt.num_reqs_padded) {
    if (g == 0)
      for (int i = num_reqs + threadIdx.x; i < seq_lens_len; i += kDaThreads) seq_lens[i] = 0;
    return;
  }
  if (b >= num_reqs) {  // padded request rows of the forward block tables
    int32_t* dst = dst_base + b * stride;
    for (int64_t i = threadIdx.x; i < stride; i += kDaThreads) dst[i] = 0;
    return;
  }
  const int64_t req = rd(idx, b);
  const int32_t nbk = bt.num_blocks[g * bt.num_blocks_stride + req];  // host-mapped: issue early
  const int32_t nc = num_computed[req];
  const int32_t start = static_cast<int32_t>(rd(qsl, b));
  const int32_t end = static_cast<int32_t>(rd(qsl, b + 1));
  const int32_t query_len = end - start;
  const int32_t* src = reinterpret_cast<const int32_t*>(bt.src_ptrs[g]) + req * stride;
  const int64_t kbs = bt.kernel_block_sizes[g];
  const bool en = bt.slot_enabled[g];
  for (int32_t i = threadIdx.x; i < query_len; i += kDaThreads) {
    const int64_t pos = static_cast<int64_t>(nc + i);
    if (g == 0) positions[start + i] = pos;
    int64_t slot = bt.pad_id;
    if (en) {
      const int32_t bn = src[pos / kbs];
      // Triton: int32 block number * int32 kernel block size (wrapping), then + int64 offset
      const int32_t prod = static_cast<int32_t>(static_cast<uint32_t>(bn) * static_cast<uint32_t>(kbs));
      slot = static_cast<int64_t>(prod) + pos % kbs;
    }
    sm[start + i] = slot;
  }
  int32_t* dst = dst_base + b * stride;
  for (int32_t i = threadIdx.x; i < nbk; i += kDaThreads) dst[i] = src[i];
  if (g != 0 || threadIdx.x != 0) return;
  const int32_t seq_len = nc + query_len;
  seq_lens[b] = seq_len;
  const int32_t c0 = static_cast<int32_t>(rd(cu, b));
  const int32_t num_logits = static_cast<int32_t>(rd(cu, b + 1)) - c0;
  const int32_t logits_start = end - num_logits;
  for (int32_t j = 0; j < num_logits; ++j) logits_indices[c0 + j] = static_cast<int64_t>(logits_start + j);
  const int32_t pl = static_cast<int32_t>(rd(prefill_len, req));
  if (seq_len <= pl) return;
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

void decode_prep_attn(const torch::Tensor& idx_mapping, const torch::Tensor& query_start_loc,
                      const torch::Tensor& num_computed_tokens, const torch::Tensor& last_sampled_tokens,
                      const torch::Tensor& prefill_len, const torch::Tensor& cu_num_logits,
                      torch::Tensor& positions, torch::Tensor& seq_lens, torch::Tensor& input_ids,
                      torch::Tensor& logits_indices, int64_t num_reqs_padded,
                      const torch::Tensor& src_ptrs, const torch::Tensor& dst_ptrs,
                      const torch::Tensor& strides, const torch::Tensor& num_blocks,
                      const torch::Tensor& kernel_block_sizes, const torch::Tensor& slot_enabled,
                      torch::Tensor& slot_mappings, int64_t pad_id) {
  const int num_reqs = static_cast<int>(idx_mapping.numel());
  TORCH_CHECK(num_reqs >= 1 && idx_mapping.dim() == 1 && num_reqs_padded >= num_reqs,
              "idx_mapping must be 1-D, non-empty; num_reqs_padded >= num_reqs");
  TORCH_CHECK(query_start_loc.numel() >= num_reqs + 1 && cu_num_logits.numel() >= num_reqs + 1,
              "query_start_loc / cu_num_logits too short");
  TORCH_CHECK(num_computed_tokens.scalar_type() == at::kInt && num_computed_tokens.is_contiguous(),
              "num_computed_tokens must be contiguous int32");
  TORCH_CHECK(last_sampled_tokens.scalar_type() == at::kLong && last_sampled_tokens.dim() == 2,
              "last_sampled_tokens must be int64 [R, 1]");
  TORCH_CHECK(positions.scalar_type() == at::kLong && positions.is_contiguous() &&
                  seq_lens.scalar_type() == at::kInt && seq_lens.is_contiguous() &&
                  seq_lens.numel() >= num_reqs_padded && input_ids.scalar_type() == at::kInt &&
                  input_ids.is_contiguous() && logits_indices.scalar_type() == at::kLong &&
                  logits_indices.is_contiguous(),
              "positions i64 / seq_lens i32 / input_ids i32 / logits_indices i64, contiguous");
  const int G = static_cast<int>(src_ptrs.numel());
  TORCH_CHECK(G >= 1 && G <= kMaxGroups && dst_ptrs.numel() == G && strides.numel() == G &&
                  kernel_block_sizes.numel() == G && slot_enabled.numel() == G,
              "block-table layout tensors must all have num_kv_cache_groups entries");
  TORCH_CHECK(src_ptrs.element_size() == 8 && dst_ptrs.element_size() == 8 &&
                  strides.scalar_type() == at::kLong && kernel_block_sizes.scalar_type() == at::kInt &&
                  slot_enabled.scalar_type() == at::kBool && src_ptrs.is_cuda() && dst_ptrs.is_cuda() &&
                  strides.is_cuda() && kernel_block_sizes.is_cuda() && slot_enabled.is_cuda(),
              "block-table layout tensor dtypes");
  TORCH_CHECK(num_blocks.scalar_type() == at::kInt && num_blocks.dim() == 2 && num_blocks.size(0) == G &&
                  num_blocks.stride(1) == 1,
              "num_blocks must be int32 [G, max_reqs]");
  TORCH_CHECK(slot_mappings.scalar_type() == at::kLong && slot_mappings.dim() == 2 &&
                  slot_mappings.size(0) == G && slot_mappings.stride(1) == 1,
              "slot_mappings must be int64 [G, max_tokens]");
  BTArgs bt;
  bt.groups = G;
  bt.src_ptrs = static_cast<const uint64_t*>(src_ptrs.data_ptr());
  bt.dst_ptrs = static_cast<const uint64_t*>(dst_ptrs.data_ptr());
  bt.strides = strides.data_ptr<int64_t>();
  bt.num_blocks = num_blocks.data_ptr<int32_t>();
  bt.num_blocks_stride = num_blocks.stride(0);
  bt.kernel_block_sizes = kernel_block_sizes.data_ptr<int32_t>();
  bt.slot_enabled = slot_enabled.data_ptr<bool>();
  bt.slot_mappings = slot_mappings.data_ptr<int64_t>();
  bt.sm_stride = slot_mappings.stride(0);
  bt.max_tokens = static_cast<int>(slot_mappings.size(1));
  bt.pad_id = pad_id;
  bt.num_reqs_padded = static_cast<int>(num_reqs_padded);
  launch_pdl(decode_prep_attn_kernel, dim3(num_reqs_padded + 1, G), dim3(kDaThreads),
             at::cuda::getCurrentCUDAStream().stream(), iview(idx_mapping, "idx_mapping"),
             iview(query_start_loc, "query_start_loc"),
             static_cast<const int32_t*>(num_computed_tokens.data_ptr<int32_t>()),
             static_cast<const int64_t*>(last_sampled_tokens.data_ptr<int64_t>()),
             static_cast<int64_t>(last_sampled_tokens.stride(0)), iview(prefill_len, "prefill_len"),
             iview(cu_num_logits, "cu_num_logits"), positions.data_ptr<int64_t>(),
             seq_lens.data_ptr<int32_t>(), static_cast<int>(seq_lens.numel()),
             input_ids.data_ptr<int32_t>(), logits_indices.data_ptr<int64_t>(), num_reqs, bt);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(k3step, m) {
  m.def("decode_prep(Tensor idx_mapping, Tensor query_start_loc, Tensor num_computed_tokens, "
        "Tensor last_sampled_tokens, Tensor prefill_len, Tensor cu_num_logits, Tensor(a!) positions, "
        "Tensor(b!) seq_lens, Tensor(c!) input_ids, Tensor(d!) logits_indices) -> ()");
  m.def("decode_prep_attn(Tensor idx_mapping, Tensor query_start_loc, Tensor num_computed_tokens, "
        "Tensor last_sampled_tokens, Tensor prefill_len, Tensor cu_num_logits, Tensor(a!) positions, "
        "Tensor(b!) seq_lens, Tensor(c!) input_ids, Tensor(d!) logits_indices, int num_reqs_padded, "
        "Tensor src_ptrs, Tensor dst_ptrs, Tensor strides, Tensor num_blocks, Tensor kernel_block_sizes, "
        "Tensor slot_enabled, Tensor(e!) slot_mappings, int pad_id) -> ()");
}

TORCH_LIBRARY_IMPL(k3step, CUDA, m) {
  m.impl("decode_prep", &decode_prep);
  m.impl("decode_prep_attn", &decode_prep_attn);
}
