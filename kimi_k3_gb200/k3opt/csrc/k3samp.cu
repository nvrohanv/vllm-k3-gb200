// Kimi K3 decode, post-graph step work: faster TP logits all-gather and TP-distributed sampling.
// Registered as torch.ops.k3samp.*; used by agents/kda/step_patch.py.
//
// Both ops exchange data through a symmetric-memory Lamport mailbox with NVLS multicast stores
// (the pattern of agents/oproj/oproj_ar.cu): every 32-bit word of an empty mailbox entry holds a
// sentinel, producers write their entry into slot [rank] of EVERY rank's mailbox with one
// multimem.st, consumers poll their local mailbox until no sentinel word is left, consume, and
// re-arm (write the sentinel back).  One kernel launch per call, no NCCL.
//
// k3samp.allgather(local[M, S] bf16, mailbox[NBUF, W, MAXM, S] bf16, mc, buf, rank, mode,
//                  out[M, W*S] bf16)
//   Replaces tensor_model_parallel_all_gather(logits) for the lm_head logits (idea 1).
//   Sentinel 0x80000000 (bf16 pair (+0.0, -0.0)); producers turn bf16 -0.0 into +0.0, which no
//   sampler / logprob computation can tell apart.  Every CTA first publishes its share of the
//   local rows, then polls / copies / re-arms its share of the W slots.  The grid is small
//   enough to be co-resident, so a polling CTA never blocks a CTA that still has to publish.
//
// k3samp.dist_argmax(local_max[M, NB] f32, local_arg[M, NB] i32, mailbox[NBUF, MAXM, W, 2] u32,
//                    mc, buf, rank, world, mode, idx_mapping, seq_lens, cu_num_logits,
//                    prefill_len, sampled[M] i64, num_sampled[M] i32, num_rejected[M] i32,
//                    my_packet?)
//   Finishes TP-distributed Gumbel-max sampling (idea 2): local_max / local_arg are the per-block
//   (1024 vocab ids) noised maxima of THIS rank's vocab shard, keyed by global token id (computed
//   by step_patch's Triton kernel with vLLM's own gumbel_noised_argmax).  Warp m reduces token m's
//   blocks, publishes (value, id) as one 8-byte multimem.st, polls the W packets, and reduces them
//   with the total order "NaN first, then larger value, then smaller id" -- the order vLLM's
//   full-vocab path realises (tl.max leftmost index within a 1024-block, torch.argmax first block
//   across blocks), and associative, so every rank gets the same, identical token.  It also writes
//   num_sampled / num_rejected exactly as vLLM's _get_num_sampled_and_rejected_kernel does for one
//   logit per request.  Sentinel 0xFFFFFFFF in both words (never a real id; arithmetic NaNs are
//   0x7fffffff / 0x7fc00000).
//
// Mailbox reuse (callers alternate NBUF = 2 buffers per call, on every rank in the same order):
// a rank writes buffer b again two calls later, after its intermediate call completed, which
// needed every rank's packet of that call, which each rank published only after its previous call
// (the one that consumed and re-armed buffer b) completed.  Calls are plain stream-ordered launches.
//
// mode 0: multimem.st to the multicast address (production); 1: local store into slot [rank] of the
// local mailbox (single-GPU tests; other slots pre-filled by the test); 2 (dist_argmax only): only
// compute this rank's packet into my_packet (tests).
#include <cuda_bf16.h>
#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

namespace {

constexpr uint32_t kSentG = 0x80000000u;    // all-gather: empty marker, one per 32-bit word
constexpr uint32_t kSentA = 0xffffffffu;    // argmax packets: empty marker
constexpr int kMaxWorld = 16;
constexpr int kMaxM = 16;
constexpr unsigned long long kTimeoutNs = 20ull * 1000 * 1000 * 1000;  // 20 s -> trap, not hang

struct U4 {
  uint32_t w[4];
};

__device__ __forceinline__ unsigned long long globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ U4 ld16(const void* p) {
  U4 u;
  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(u.w[0]), "=r"(u.w[1]), "=r"(u.w[2]), "=r"(u.w[3]) : "l"(p));
  return u;
}
__device__ __forceinline__ U4 ld_volatile16(const void* p) {
  U4 u;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(u.w[0]), "=r"(u.w[1]), "=r"(u.w[2]), "=r"(u.w[3]) : "l"(p) : "memory");
  return u;
}
__device__ __forceinline__ uint2 ld_volatile8(const void* p) {
  uint2 u;
  asm volatile("ld.volatile.global.v2.u32 {%0,%1}, [%2];" : "=r"(u.x), "=r"(u.y) : "l"(p) : "memory");
  return u;
}
__device__ __forceinline__ void st16(void* p, const U4& u) {
  asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(u.w[0]), "r"(u.w[1]),
               "r"(u.w[2]), "r"(u.w[3]) : "memory");
}
__device__ __forceinline__ void st8(void* p, uint2 u) {
  asm volatile("st.global.v2.u32 [%0], {%1,%2};" ::"l"(p), "r"(u.x), "r"(u.y) : "memory");
}
__device__ __forceinline__ void multimem_st16(unsigned long long addr, const U4& u) {
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(addr),
               "r"(u.w[0]), "r"(u.w[1]), "r"(u.w[2]), "r"(u.w[3]) : "memory");
}
__device__ __forceinline__ void multimem_st8(unsigned long long addr, uint2 u) {
  asm volatile("multimem.st.relaxed.sys.global.v2.f32 [%0], {%1,%2};" ::"l"(addr), "r"(u.x),
               "r"(u.y) : "memory");
}
__device__ __forceinline__ bool dirty_g(const U4& u) {
  return (u.w[0] == kSentG) | (u.w[1] == kSentG) | (u.w[2] == kSentG) | (u.w[3] == kSentG);
}
__device__ __forceinline__ uint32_t no_neg_zero(uint32_t w) {  // bf16 -0.0 -> +0.0 (both halves)
  if ((w & 0xffffu) == 0x8000u) w &= 0xffff0000u;
  if ((w >> 16) == 0x8000u) w &= 0x0000ffffu;
  return w;
}
__device__ __forceinline__ void check_timeout(unsigned long long t0) {
  if (globaltimer() - t0 > kTimeoutNs) {
    printf("k3samp: mailbox poll timed out (a TP rank did not publish) -- aborting\n");
    __trap();
  }
}

// ------------------------------------------------------------------------------------------
// all-gather
// ------------------------------------------------------------------------------------------
constexpr int kAgThreads = 256;
constexpr int kAgBatch = 8;  // fragments polled per thread per round

__global__ void __launch_bounds__(kAgThreads)
    allgather_kernel(const uint8_t* __restrict__ local, int64_t local_row_bytes,
                     uint8_t* __restrict__ mailbox, unsigned long long mc, int buf, int rank,
                     int world, int max_m, int frags, int M, int mode,
                     uint8_t* __restrict__ out, int64_t out_row_bytes) {
  const int nthr = gridDim.x * kAgThreads;
  const int tid = blockIdx.x * kAgThreads + threadIdx.x;
  const int64_t slot_bytes = static_cast<int64_t>(max_m) * frags * 16;  // one rank's rows
  const int64_t buf_bytes = slot_bytes * world;
  // publish my rows into slot [rank]
  for (int i = tid; i < M * frags; i += nthr) {
    const int m = i / frags, f = i - m * frags;
    U4 u = ld16(local + m * local_row_bytes + static_cast<int64_t>(f) * 16);
#pragma unroll
    for (int k = 0; k < 4; ++k) u.w[k] = no_neg_zero(u.w[k]);
    const int64_t off = buf * buf_bytes + rank * slot_bytes + (static_cast<int64_t>(m) * frags + f) * 16;
    if (mode == 0) {
      multimem_st16(mc + off, u);
    } else {
      st16(mailbox + off, u);
    }
  }
  // consume every slot: poll, copy into out[m, w*S : (w+1)*S], re-arm
  const U4 sentinel = {{kSentG, kSentG, kSentG, kSentG}};
  const int total = world * M * frags;
  const unsigned long long t0 = globaltimer();
  for (int base = tid; base < total; base += nthr * kAgBatch) {
    U4 v[kAgBatch];
    int64_t src[kAgBatch], dst[kAgBatch];
    uint32_t pending = 0;
#pragma unroll
    for (int r = 0; r < kAgBatch; ++r) {
      const int i = base + r * nthr;
      if (i < total) {
        const int w = i / (M * frags);
        const int rem = i - w * M * frags;
        const int m = rem / frags, f = rem - m * frags;
        src[r] = buf * buf_bytes + w * slot_bytes + (static_cast<int64_t>(m) * frags + f) * 16;
        dst[r] = m * out_row_bytes + (static_cast<int64_t>(w) * frags + f) * 16;
        v[r] = ld_volatile16(mailbox + src[r]);
        pending |= 1u << r;
      }
    }
    while (true) {
      uint32_t still = 0;
#pragma unroll
      for (int r = 0; r < kAgBatch; ++r) {
        if (!(pending & (1u << r))) continue;
        if (dirty_g(v[r])) {
          v[r] = ld_volatile16(mailbox + src[r]);
          still |= 1u << r;
        } else {
          st16(out + dst[r], v[r]);
          st16(mailbox + src[r], sentinel);
        }
      }
      if (!still) break;
      pending = still;
      check_timeout(t0);
    }
  }
}

// ------------------------------------------------------------------------------------------
// distributed argmax
// ------------------------------------------------------------------------------------------
struct Cand {
  float v;
  int id;
};
// "a before b" in the order NaN first, larger value, smaller id.
__device__ __forceinline__ bool better(const Cand& a, const Cand& b) {
  const bool an = isnan(a.v), bn = isnan(b.v);
  if (an != bn) return an;
  if (!an && a.v != b.v) return a.v > b.v;
  return a.id < b.id;
}
__device__ __forceinline__ Cand warp_best(Cand c) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    Cand o;
    o.v = __shfl_xor_sync(0xffffffffu, c.v, off);
    o.id = __shfl_xor_sync(0xffffffffu, c.id, off);
    if (better(o, c)) c = o;
  }
  return c;
}

__global__ void __launch_bounds__(32 * kMaxM)
    dist_argmax_kernel(const float* __restrict__ local_max, const int* __restrict__ local_arg,
                       int nb, uint32_t* __restrict__ mailbox, unsigned long long mc, int buf,
                       int rank, int world, int max_m, int M, int mode,
                       const int* __restrict__ idx_mapping, const int* __restrict__ seq_lens,
                       const int* __restrict__ cu_num_logits, const int* __restrict__ prefill_len,
                       int64_t* __restrict__ sampled, int* __restrict__ num_sampled,
                       int* __restrict__ num_rejected, uint32_t* __restrict__ my_packet) {
  const int m = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (m >= M) return;
  const Cand none = {-__int_as_float(0x7f800000), 0x7fffffff};  // (-inf, INT_MAX): neutral
  Cand c = none;
  if (lane < nb) {
    c.v = local_max[m * nb + lane];
    c.id = local_arg[m * nb + lane];
  }
  c = warp_best(c);
  const uint2 packet = make_uint2(__float_as_uint(c.v), static_cast<uint32_t>(c.id));
  if (mode == 2) {
    if (lane == 0) {
      my_packet[2 * m] = packet.x;
      my_packet[2 * m + 1] = packet.y;
    }
    return;
  }
  // entry (buf, m, w) at ((buf * max_m + m) * world + w) * 8 bytes
  const int64_t row = (static_cast<int64_t>(buf) * max_m + m) * world;
  if (lane == 0) {
    if (mode == 0) {
      multimem_st8(mc + (row + rank) * 8, packet);
    } else {
      st8(mailbox + (row + rank) * 2, packet);
    }
  }
  Cand g = none;
  uint32_t* const src = mailbox + (row + lane) * 2;
  if (lane < world) {
    const unsigned long long t0 = globaltimer();
    uint2 p = ld_volatile8(src);
    while (p.x == kSentA || p.y == kSentA) {
      check_timeout(t0);
      p = ld_volatile8(src);
    }
    g.v = __uint_as_float(p.x);
    g.id = static_cast<int>(p.y);
  }
  g = warp_best(g);
  if (lane < world) st8(src, make_uint2(kSentA, kSentA));  // re-arm
  if (lane == 0) {
    sampled[m] = g.id;
    // vLLM _get_num_sampled_and_rejected_kernel with num_sampled initialised to 1
    const int req = idx_mapping[m];
    const bool chunked = seq_lens[m] < prefill_len[req];
    const int num_logits = cu_num_logits[m + 1] - cu_num_logits[m];
    num_sampled[m] = chunked ? 0 : 1;
    num_rejected[m] = chunked ? 0 : num_logits - 1;
  }
}

cudaStream_t stream() { return at::cuda::getCurrentCUDAStream().stream(); }

void allgather(const torch::Tensor& local, torch::Tensor& mailbox, int64_t mc, int64_t buf,
               int64_t rank, int64_t mode, torch::Tensor& out) {
  TORCH_CHECK(local.is_cuda() && local.scalar_type() == at::kBFloat16 && local.dim() == 2 &&
                  local.stride(1) == 1,
              "local must be bf16 [M, S] with unit inner stride");
  TORCH_CHECK(mailbox.is_cuda() && mailbox.scalar_type() == at::kBFloat16 && mailbox.dim() == 4 &&
                  mailbox.is_contiguous(),
              "mailbox must be contiguous bf16 [NBUF, W, MAXM, S]");
  const int M = local.size(0), S = local.size(1);
  const int nbuf = mailbox.size(0), world = mailbox.size(1), max_m = mailbox.size(2);
  TORCH_CHECK(mailbox.size(3) == S && S % 8 == 0, "shard width mismatch / not a multiple of 8");
  TORCH_CHECK(M >= 1 && M <= max_m && 0 <= buf && buf < nbuf && 0 <= rank && rank < world &&
                  world <= kMaxWorld,
              "bad M / buf / rank");
  TORCH_CHECK((local.stride(0) * 2) % 16 == 0 && reinterpret_cast<uintptr_t>(local.data_ptr()) % 16 == 0,
              "local rows must be 16-byte aligned");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.dim() == 2 &&
                  out.size(0) == M && out.size(1) == static_cast<int64_t>(world) * S &&
                  out.stride(1) == 1 && (out.stride(0) * 2) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
              "out must be bf16 [M, W*S], 16-byte aligned rows");
  TORCH_CHECK(mode == 0 ? mc != 0 : mode == 1, "mode 0 needs a multicast address; mode 1 local");
  const int frags = S / 8;
  const int work = world * M * frags;
  int grid = (work + kAgThreads * 2 - 1) / (kAgThreads * 2);  // ~2 fragments per thread
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, local.get_device());
  grid = std::max(1, std::min(grid, sms));  // co-resident (1 CTA of 256 threads per SM)
  allgather_kernel<<<grid, kAgThreads, 0, stream()>>>(
      static_cast<const uint8_t*>(local.data_ptr()), local.stride(0) * 2,
      static_cast<uint8_t*>(mailbox.data_ptr()), static_cast<unsigned long long>(mc),
      static_cast<int>(buf), static_cast<int>(rank), world, max_m, frags, M,
      static_cast<int>(mode), static_cast<uint8_t*>(out.data_ptr()), out.stride(0) * 2);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dist_argmax(const torch::Tensor& local_max, const torch::Tensor& local_arg,
                 torch::Tensor& mailbox, int64_t mc, int64_t buf, int64_t rank, int64_t mode,
                 const torch::Tensor& idx_mapping, const torch::Tensor& seq_lens,
                 const torch::Tensor& cu_num_logits, const torch::Tensor& prefill_len,
                 torch::Tensor& sampled, torch::Tensor& num_sampled, torch::Tensor& num_rejected,
                 const std::optional<torch::Tensor>& my_packet) {
  TORCH_CHECK(local_max.is_cuda() && local_max.scalar_type() == at::kFloat &&
                  local_max.dim() == 2 && local_max.is_contiguous(),
              "local_max must be contiguous f32 [M, NB]");
  TORCH_CHECK(local_arg.scalar_type() == at::kInt && local_arg.sizes() == local_max.sizes() &&
                  local_arg.is_contiguous(),
              "local_arg must be contiguous i32 [M, NB]");
  const int M = local_max.size(0), nb = local_max.size(1);
  TORCH_CHECK(M >= 1 && M <= kMaxM && nb >= 1 && nb <= 32, "M in [1, 16], NB in [1, 32]");
  TORCH_CHECK(mailbox.is_cuda() && mailbox.scalar_type() == at::kInt && mailbox.dim() == 4 &&
                  mailbox.size(3) == 2 && mailbox.is_contiguous(),
              "mailbox must be contiguous i32 [NBUF, MAXM, W, 2]");
  const int nbuf = mailbox.size(0), max_m = mailbox.size(1), world = mailbox.size(2);
  TORCH_CHECK(M <= max_m && 0 <= buf && buf < nbuf && 0 <= rank && rank < world && world <= 32,
              "bad M / buf / rank");
  TORCH_CHECK(mode == 0 ? mc != 0 : (mode == 1 || mode == 2), "bad mode / multicast address");
  TORCH_CHECK(mode != 2 || (my_packet.has_value() && my_packet->scalar_type() == at::kInt &&
                            my_packet->numel() >= 2 * M && my_packet->is_contiguous()),
              "mode 2 needs my_packet i32 [M, 2]");
  for (const auto* t : {&idx_mapping, &seq_lens, &cu_num_logits, &prefill_len}) {
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kInt && t->is_contiguous(),
                "idx_mapping / seq_lens / cu_num_logits / prefill_len must be contiguous i32");
  }
  TORCH_CHECK(idx_mapping.numel() >= M && seq_lens.numel() >= M && cu_num_logits.numel() >= M + 1,
              "index tensors too short");
  TORCH_CHECK(sampled.scalar_type() == at::kLong && sampled.numel() >= M && sampled.is_contiguous() &&
                  num_sampled.scalar_type() == at::kInt && num_sampled.numel() >= M &&
                  num_rejected.scalar_type() == at::kInt && num_rejected.numel() >= M,
              "outputs: sampled i64 [M], num_sampled / num_rejected i32 [M]");
  dist_argmax_kernel<<<1, 32 * M, 0, stream()>>>(
      local_max.data_ptr<float>(), local_arg.data_ptr<int>(), nb,
      reinterpret_cast<uint32_t*>(mailbox.data_ptr<int>()), static_cast<unsigned long long>(mc),
      static_cast<int>(buf), static_cast<int>(rank), world, max_m, M, static_cast<int>(mode),
      idx_mapping.data_ptr<int>(), seq_lens.data_ptr<int>(), cu_num_logits.data_ptr<int>(),
      prefill_len.data_ptr<int>(), sampled.data_ptr<int64_t>(), num_sampled.data_ptr<int>(),
      num_rejected.data_ptr<int>(),
      my_packet.has_value() ? reinterpret_cast<uint32_t*>(my_packet->data_ptr<int>()) : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

TORCH_LIBRARY(k3samp, m) {
  m.def("allgather(Tensor local, Tensor(a!) mailbox, int mc, int buf, int rank, int mode, "
        "Tensor(b!) out) -> ()");
  m.def("dist_argmax(Tensor local_max, Tensor local_arg, Tensor(a!) mailbox, int mc, int buf, "
        "int rank, int mode, Tensor idx_mapping, Tensor seq_lens, Tensor cu_num_logits, "
        "Tensor prefill_len, Tensor(b!) sampled, Tensor(c!) num_sampled, Tensor(d!) num_rejected, "
        "Tensor? my_packet=None) -> ()");
}

TORCH_LIBRARY_IMPL(k3samp, CUDA, m) {
  m.impl("allgather", &allgather);
  m.impl("dist_argmax", &dist_argmax);
}
