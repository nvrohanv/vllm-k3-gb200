// Kimi K3 decode, work outside the layers (W1-6 stepov).  torch.ops.k3step.*, used by
// agents/stepov/stepov_patch.py.  Successor of k3opt/csrc/k3samp.cu (same protocols).
//
// All three ops exchange data through symmetric-memory Lamport mailboxes written with NVLS
// multicast (the pattern of agents/oproj/oproj_ar.cu): every 32-bit word of an empty entry holds a
// sentinel; a producer writes its entry into EVERY rank's mailbox with one multimem.st; consumers
// poll their local mailbox until no sentinel word is left, consume, and re-arm.  No NCCL, one
// launch per op.  A consumer that waits > 20 s traps (a rank did not publish) instead of hanging.
//
// k3step.allgather                  (as k3samp.allgather) one-hop all-gather of [M, S] bf16 logits.
//
// k3step.sample_finish              (k3samp.dist_argmax + vLLM's _post_update + mamba
//                                    _scatter_num_accepted, fused)
//   Finishes TP-distributed Gumbel-max sampling: warp m reduces token m's local block maxima,
//   publishes one (value, id) packet, polls the W packets, reduces them (NaN first, larger value,
//   smaller id: the order vLLM's full-vocab path realises), writes sampled[m], num_sampled[m],
//   num_rejected[m] exactly like _get_num_sampled_and_rejected_kernel (one logit per request),
//   and, if `post` is given, applies vLLM's _post_update_kernel for that row
//   (last_sampled_tokens, total_len, all_token_ids, output_bin_counts, num_computed_tokens) and
//   the mamba-hybrid model state's _scatter_num_accepted_kernel (num_accepted = max(ns, 1)).
//   A4 (optional): also writes sampled / num_sampled into pinned host mirrors (the V2 runner's
//   AsyncOutput then needs no DtoH copies) and, after every row's scatter, the mamba align
//   postprocess snapshot of the whole num_accepted buffer (the 64 B DtoD copy of
//   MambaSpecDecodeGPUContext.run_fused_postprocess_align).
//   Index / length inputs may be int32 or int64 and strided (no host-side conversion kernels).
//
// k3step.embed_bcast                (replaces vocab_parallel_embedding + TP all-reduce)
//   Every token has exactly one owner rank (id in [rank*S, rank*S + S)).  The owner reads its
//   embedding row, turns bf16 -0.0 into +0.0 (what the all-reduce of one row plus 15 +0.0 rows
//   returns), and multicasts it into slot [m] of every rank's mailbox [MAXM, H]; every rank polls,
//   copies to `out` and re-arms.  Ids outside [0, vocab) produce a zero row on every rank without
//   any exchange (the all-reduce of 16 masked rows).  One mailbox buffer suffices: a rank publishes
//   for step t+1 only after its step-t forward, which needed every rank's layer-1 publish, which
//   each rank issues after its step-t embedding kernel (and its re-arm) completed.  The kernel is
//   CUDA-graph safe (static buffers, no host state).
#include <cuda_bf16.h>
#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

namespace {

constexpr uint32_t kSentG = 0x80000000u;  // bf16 mailboxes: empty marker per 32-bit word
constexpr uint32_t kSentA = 0xffffffffu;  // argmax packets: empty marker
constexpr int kMaxWorld = 16;
constexpr int kMaxM = 16;
constexpr unsigned long long kTimeoutNs = 20ull * 1000 * 1000 * 1000;

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
__device__ __forceinline__ void check_timeout(unsigned long long t0, const char* what) {
  if (globaltimer() - t0 > kTimeoutNs) {
    printf("k3step.%s: mailbox poll timed out (a TP rank did not publish) -- aborting\n", what);
    __trap();
  }
}

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

// Launch with programmatic stream serialization: the kernel may start while its predecessor drains;
// it executes griddepcontrol.wait before reading anything the predecessor produced.
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

// Integer tensor view: int32 or int64, any stride.
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

// ------------------------------------------------------------------------------------------
// all-gather (unchanged from k3samp)
// ------------------------------------------------------------------------------------------
constexpr int kAgThreads = 256;
constexpr int kAgBatch = 8;

__global__ void __launch_bounds__(kAgThreads)
    allgather_kernel(const uint8_t* __restrict__ local, int64_t local_row_bytes,
                     uint8_t* __restrict__ mailbox, unsigned long long mc, int buf, int rank,
                     int world, int max_m, int frags, int M, int mode,
                     uint8_t* __restrict__ out, int64_t out_row_bytes) {
  const int nthr = gridDim.x * kAgThreads;
  const int tid = blockIdx.x * kAgThreads + threadIdx.x;
  const int64_t slot_bytes = static_cast<int64_t>(max_m) * frags * 16;
  const int64_t buf_bytes = slot_bytes * world;
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
      check_timeout(t0, "allgather");
    }
  }
}

// ------------------------------------------------------------------------------------------
// sample_finish
// ------------------------------------------------------------------------------------------
struct Cand {
  float v;
  int id;
};
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

struct PostArgs {  // vLLM _post_update_kernel + mamba _scatter_num_accepted_kernel targets
  int enabled;
  IView qsl;                     // query_start_loc [M + 1]
  int32_t* num_computed;         // [R]
  int64_t* last_sampled;         // [R, *], row stride last_stride
  int64_t last_stride;
  int32_t* total_len;            // [R]
  int32_t* all_token_ids;        // [R, L], row stride tok_stride (may be UVA)
  int64_t tok_stride;
  int32_t* bin_counts;           // [R, V] or null
  int64_t bin_stride;
  int32_t* num_accepted;         // [R] or null
};

struct SampleParams {
  const float* local_max;
  const int* local_arg;
  int nb;
  uint32_t* mailbox;
  unsigned long long mc;
  int buf, rank, world, max_m, M, mode;
  IView idx_mapping, seq_lens, cu_num_logits, prefill_len;
  int64_t* sampled;
  int* num_sampled;
  int* num_rejected;
  uint32_t* my_packet;
  PostArgs post;
  // A4 (stepov item L4): optional folds of the eager copies that follow sampling
  int64_t* h_sampled;      // host-mapped (pinned, UVA) mirror of sampled [M]: replaces AsyncOutput's DtoH copy
  int* h_num_sampled;      // host-mapped mirror of num_sampled [M]
  int32_t* snap;           // mamba align postprocess snapshot of post.num_accepted (whole buffer), or null
  int snap_n;
};

__global__ void __launch_bounds__(32 * kMaxM) sample_finish_kernel(const SampleParams p) {
  pdl_wait();
  const int m = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (m >= p.M) return;
  const Cand none = {-__int_as_float(0x7f800000), 0x7fffffff};
  Cand c = none;
  if (lane < p.nb) {
    c.v = p.local_max[m * p.nb + lane];
    c.id = p.local_arg[m * p.nb + lane];
  }
  c = warp_best(c);
  const uint2 packet = make_uint2(__float_as_uint(c.v), static_cast<uint32_t>(c.id));
  if (p.mode == 2) {
    if (lane == 0) {
      p.my_packet[2 * m] = packet.x;
      p.my_packet[2 * m + 1] = packet.y;
    }
    return;
  }
  // Everything the tail needs that does not depend on the sampled token is loaded now, so its
  // latency (prefill_len may live in host memory) hides behind the exchange.
  int64_t req = -1;
  int ns = 0, nr = 0, total_len = 0, new_computed = 0, delta = 0;
  if (lane == 0) {
    req = rd(p.idx_mapping, m);
    const int64_t sl = rd(p.seq_lens, m);
    const int64_t pl = req >= 0 ? rd(p.prefill_len, req) : 0;
    const int num_logits = static_cast<int>(rd(p.cu_num_logits, m + 1) - rd(p.cu_num_logits, m));
    // vLLM _get_num_sampled_and_rejected_kernel with num_sampled initialised to 1
    const bool chunked = req >= 0 && sl < pl;
    ns = chunked ? 0 : 1;
    nr = chunked ? 0 : num_logits - 1;
    if (p.post.enabled && req >= 0) {
      total_len = p.post.total_len[req];
      delta = static_cast<int>(rd(p.post.qsl, m + 1) - rd(p.post.qsl, m)) - nr;
      new_computed = p.post.num_computed[req] + delta;
    }
  }
  const int64_t row = (static_cast<int64_t>(p.buf) * p.max_m + m) * p.world;
  if (lane == 0) {
    if (p.mode == 0) {
      multimem_st8(p.mc + (row + p.rank) * 8, packet);
    } else {
      st8(p.mailbox + (row + p.rank) * 2, packet);
    }
  }
  Cand g = none;
  uint32_t* const src = p.mailbox + (row + lane) * 2;
  if (lane < p.world) {
    const unsigned long long t0 = globaltimer();
    uint2 v = ld_volatile8(src);
    while (v.x == kSentA || v.y == kSentA) {
      check_timeout(t0, "sample_finish");
      v = ld_volatile8(src);
    }
    g.v = __uint_as_float(v.x);
    g.id = static_cast<int>(v.y);
  }
  g = warp_best(g);
  if (lane < p.world) st8(src, make_uint2(kSentA, kSentA));  // re-arm
  if (lane == 0) {
    const int64_t token = g.id;
    p.sampled[m] = token;
    p.num_sampled[m] = ns;
    p.num_rejected[m] = nr;
    if (p.h_sampled != nullptr) p.h_sampled[m] = token;  // host mirrors (A4): no DtoH copies afterwards
    if (p.h_num_sampled != nullptr) p.h_num_sampled[m] = ns;
    if (p.post.enabled && req >= 0) {
      // vLLM _post_update_kernel for this row (sampled_tokens = [M, 1]); the (possibly host-resident)
      // all_token_ids store goes first so its latency overlaps the rest.
      const PostArgs& q = p.post;
      if (ns > 0) {
        q.all_token_ids[req * q.tok_stride + total_len] = static_cast<int32_t>(token);
        q.last_sampled[req * q.last_stride] = token;
        q.total_len[req] = total_len + ns;
        if (q.bin_counts != nullptr) atomicAdd(q.bin_counts + req * q.bin_stride + token, 1);
      }
      if (delta != 0) q.num_computed[req] = new_computed;
      // mamba hybrid model state: num_accepted_tokens[req] = max(num_sampled, 1)
      if (q.num_accepted != nullptr) q.num_accepted[req] = ns > 1 ? ns : 1;
    }
  }
  // A4: the mamba align postprocess snapshot (MambaSpecDecodeGPUContext.run_fused_postprocess_align's
  // 64 B device-to-device copy of the whole num_accepted buffer), taken after every row's scatter.
  if (p.snap != nullptr) {  // uniform over the block
    __syncthreads();
    for (int i = threadIdx.x; i < p.snap_n; i += blockDim.x) p.snap[i] = p.post.num_accepted[i];
  }
}

// ------------------------------------------------------------------------------------------
// embedding broadcast
// ------------------------------------------------------------------------------------------
constexpr int kEbThreads = 128;

__device__ __forceinline__ void embed_bcast_body(IView ids, const uint8_t* __restrict__ weight, int64_t w_row_bytes,
                                                 int64_t vocab, int64_t shard, int rank, uint8_t* __restrict__ mailbox,
                                                 unsigned long long mc, int frags, int mode, uint8_t* __restrict__ out,
                                                 int64_t out_row_bytes);

__global__ void __launch_bounds__(kEbThreads)
    embed_bcast_kernel(IView ids, const uint8_t* __restrict__ weight, int64_t w_row_bytes,
                       int64_t vocab, int64_t shard, int rank, uint8_t* __restrict__ mailbox,
                       unsigned long long mc, int frags, int mode, uint8_t* __restrict__ out,
                       int64_t out_row_bytes, int M, unsigned long long* gs) {
  // grid = (ceil(frags / kEbThreads), M): thread handles one 16-byte fragment of token m's row
  pdl_wait();
  // A2c graph-start probe (optional): gs[0] = min CTA start, gs[1] = max CTA end (reset by gs_stamp)
  if (gs != nullptr && threadIdx.x == 0) atomicMin(gs, globaltimer());
  embed_bcast_body(ids, weight, w_row_bytes, vocab, shard, rank, mailbox, mc, frags, mode, out, out_row_bytes);
  if (gs != nullptr) {  // uniform: every thread returns from the body
    __syncthreads();
    if (threadIdx.x == 0) atomicMax(gs + 1, globaltimer());
  }
}

__device__ __forceinline__ void embed_bcast_body(IView ids, const uint8_t* __restrict__ weight, int64_t w_row_bytes,
                                                 int64_t vocab, int64_t shard, int rank, uint8_t* __restrict__ mailbox,
                                                 unsigned long long mc, int frags, int mode, uint8_t* __restrict__ out,
                                                 int64_t out_row_bytes) {
  const int m = blockIdx.y;
  const int f = blockIdx.x * kEbThreads + threadIdx.x;
  if (f >= frags) return;
  const int64_t id = rd(ids, m);
  uint8_t* const dst = out + m * out_row_bytes + static_cast<int64_t>(f) * 16;
  if (id < 0 || id >= vocab) {  // masked on every rank: the all-reduce returns zeros
    st16(dst, U4{{0u, 0u, 0u, 0u}});
    return;
  }
  const int64_t off = (static_cast<int64_t>(m) * frags + f) * 16;
  if (id / shard == rank) {
    U4 u = ld16(weight + (id - static_cast<int64_t>(rank) * shard) * w_row_bytes +
                static_cast<int64_t>(f) * 16);
#pragma unroll
    for (int k = 0; k < 4; ++k) u.w[k] = no_neg_zero(u.w[k]);
    if (mode == 0) {
      multimem_st16(mc + off, u);
    } else {
      st16(mailbox + off, u);
    }
  }
  const unsigned long long t0 = globaltimer();
  U4 v = ld_volatile16(mailbox + off);
  while (dirty_g(v)) {
    check_timeout(t0, "embed_bcast");
    v = ld_volatile16(mailbox + off);
  }
  st16(dst, v);
  st16(mailbox + off, U4{{kSentG, kSentG, kSentG, kSentG}});  // re-arm
}

// A2c graph-start probe: launched (no PDL) right after embed_bcast inside the graph.  ring[3 s .. 3 s + 2] = embed start,
// embed end, this kernel's start for slot s = step % nslots (ring[3 nslots] = step counter); resets gs.
__global__ void gs_stamp_kernel(unsigned long long* gs, unsigned long long* ring, int nslots) {
  const unsigned long long t = globaltimer();
  if (threadIdx.x != 0) return;
  const unsigned long long step = atomicAdd(ring + 3 * nslots, 1ull);
  unsigned long long* r = ring + 3 * (step % nslots);
  r[0] = gs[0];
  r[1] = gs[1];
  r[2] = t;
  gs[0] = ~0ull;
  gs[1] = 0ull;
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
  int grid = (work + kAgThreads * 2 - 1) / (kAgThreads * 2);
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, local.get_device());
  grid = std::max(1, std::min(grid, sms));
  allgather_kernel<<<grid, kAgThreads, 0, stream()>>>(
      static_cast<const uint8_t*>(local.data_ptr()), local.stride(0) * 2,
      static_cast<uint8_t*>(mailbox.data_ptr()), static_cast<unsigned long long>(mc),
      static_cast<int>(buf), static_cast<int>(rank), world, max_m, frags, M,
      static_cast<int>(mode), static_cast<uint8_t*>(out.data_ptr()), out.stride(0) * 2);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
T* opt_ptr(const std::optional<torch::Tensor>& t, at::ScalarType st, const char* name) {
  if (!t.has_value()) return nullptr;
  TORCH_CHECK(t->is_cuda() && t->scalar_type() == st, name, " has an unexpected dtype / device");
  return static_cast<T*>(t->data_ptr());
}

void sample_finish(const torch::Tensor& local_max, const torch::Tensor& local_arg,
                   torch::Tensor& mailbox, int64_t mc, int64_t buf, int64_t rank, int64_t mode,
                   const torch::Tensor& idx_mapping, const torch::Tensor& seq_lens,
                   const torch::Tensor& cu_num_logits, const torch::Tensor& prefill_len,
                   torch::Tensor& sampled, torch::Tensor& num_sampled, torch::Tensor& num_rejected,
                   const std::optional<torch::Tensor>& query_start_loc,
                   const std::optional<torch::Tensor>& num_computed_tokens,
                   const std::optional<torch::Tensor>& last_sampled_tokens,
                   const std::optional<torch::Tensor>& total_len,
                   const std::optional<torch::Tensor>& all_token_ids,
                   const std::optional<torch::Tensor>& output_bin_counts,
                   const std::optional<torch::Tensor>& num_accepted,
                   const std::optional<torch::Tensor>& my_packet,
                   const std::optional<torch::Tensor>& num_accepted_snapshot,
                   const std::optional<torch::Tensor>& host_sampled,
                   const std::optional<torch::Tensor>& host_num_sampled) {
  TORCH_CHECK(local_max.is_cuda() && local_max.scalar_type() == at::kFloat &&
                  local_max.dim() == 2 && local_max.is_contiguous(),
              "local_max must be contiguous f32 [M, NB]");
  TORCH_CHECK(local_arg.scalar_type() == at::kInt && local_arg.sizes() == local_max.sizes() &&
                  local_arg.is_contiguous(),
              "local_arg must be contiguous i32 [M, NB]");
  SampleParams p{};
  p.M = local_max.size(0);
  p.nb = local_max.size(1);
  TORCH_CHECK(p.M >= 1 && p.M <= kMaxM && p.nb >= 1 && p.nb <= 32, "M in [1, 16], NB in [1, 32]");
  TORCH_CHECK(mailbox.is_cuda() && mailbox.scalar_type() == at::kInt && mailbox.dim() == 4 &&
                  mailbox.size(3) == 2 && mailbox.is_contiguous(),
              "mailbox must be contiguous i32 [NBUF, MAXM, W, 2]");
  const int nbuf = mailbox.size(0);
  p.max_m = mailbox.size(1);
  p.world = mailbox.size(2);
  TORCH_CHECK(p.M <= p.max_m && 0 <= buf && buf < nbuf && 0 <= rank && rank < p.world && p.world <= 32,
              "bad M / buf / rank");
  TORCH_CHECK(mode == 0 ? mc != 0 : (mode == 1 || mode == 2), "bad mode / multicast address");
  TORCH_CHECK(mode != 2 || (my_packet.has_value() && my_packet->scalar_type() == at::kInt &&
                            my_packet->numel() >= 2 * p.M && my_packet->is_contiguous()),
              "mode 2 needs my_packet i32 [M, 2]");
  p.local_max = local_max.data_ptr<float>();
  p.local_arg = local_arg.data_ptr<int>();
  p.mailbox = reinterpret_cast<uint32_t*>(mailbox.data_ptr<int>());
  p.mc = static_cast<unsigned long long>(mc);
  p.buf = static_cast<int>(buf);
  p.rank = static_cast<int>(rank);
  p.mode = static_cast<int>(mode);
  p.idx_mapping = iview(idx_mapping, "idx_mapping");
  p.seq_lens = iview(seq_lens, "seq_lens");
  p.cu_num_logits = iview(cu_num_logits, "cu_num_logits");
  p.prefill_len = iview(prefill_len, "prefill_len");
  TORCH_CHECK(idx_mapping.numel() >= p.M && seq_lens.numel() >= p.M &&
                  cu_num_logits.numel() >= p.M + 1,
              "index tensors too short");
  TORCH_CHECK(sampled.scalar_type() == at::kLong && sampled.numel() >= p.M && sampled.is_contiguous() &&
                  num_sampled.scalar_type() == at::kInt && num_sampled.numel() >= p.M &&
                  num_sampled.is_contiguous() && num_rejected.scalar_type() == at::kInt &&
                  num_rejected.numel() >= p.M && num_rejected.is_contiguous(),
              "outputs: sampled i64 [M], num_sampled / num_rejected i32 [M]");
  p.sampled = sampled.data_ptr<int64_t>();
  p.num_sampled = num_sampled.data_ptr<int>();
  p.num_rejected = num_rejected.data_ptr<int>();
  p.my_packet = my_packet.has_value() ? reinterpret_cast<uint32_t*>(my_packet->data_ptr<int>()) : nullptr;
  p.post.enabled = query_start_loc.has_value() ? 1 : 0;
  if (p.post.enabled) {
    TORCH_CHECK(num_computed_tokens.has_value() && last_sampled_tokens.has_value() &&
                    total_len.has_value() && all_token_ids.has_value(),
                "post-update needs query_start_loc, num_computed_tokens, last_sampled_tokens, "
                "total_len, all_token_ids");
    p.post.qsl = iview(*query_start_loc, "query_start_loc");
    p.post.num_computed = opt_ptr<int32_t>(num_computed_tokens, at::kInt, "num_computed_tokens");
    p.post.last_sampled = opt_ptr<int64_t>(last_sampled_tokens, at::kLong, "last_sampled_tokens");
    p.post.last_stride = last_sampled_tokens->stride(0);
    p.post.total_len = opt_ptr<int32_t>(total_len, at::kInt, "total_len");
    p.post.all_token_ids = opt_ptr<int32_t>(all_token_ids, at::kInt, "all_token_ids");
    TORCH_CHECK(all_token_ids->dim() == 2 && all_token_ids->stride(1) == 1, "all_token_ids [R, L]");
    p.post.tok_stride = all_token_ids->stride(0);
    p.post.bin_counts = opt_ptr<int32_t>(output_bin_counts, at::kInt, "output_bin_counts");
    if (output_bin_counts.has_value()) {
      TORCH_CHECK(output_bin_counts->dim() == 2 && output_bin_counts->stride(1) == 1, "bin counts [R, V]");
      p.post.bin_stride = output_bin_counts->stride(0);
    }
    p.post.num_accepted = opt_ptr<int32_t>(num_accepted, at::kInt, "num_accepted");
    TORCH_CHECK(num_computed_tokens->is_contiguous() && total_len->is_contiguous() &&
                    (!num_accepted.has_value() || num_accepted->is_contiguous()),
                "per-request state tensors must be contiguous");
  }
  if (num_accepted_snapshot.has_value()) {
    TORCH_CHECK(p.post.enabled && p.post.num_accepted != nullptr && num_accepted.has_value(),
                "num_accepted_snapshot needs the fused post-update with num_accepted");
    TORCH_CHECK(num_accepted_snapshot->is_cuda() && num_accepted_snapshot->scalar_type() == at::kInt &&
                    num_accepted_snapshot->is_contiguous() &&
                    num_accepted_snapshot->numel() == num_accepted->numel(),
                "num_accepted_snapshot: contiguous i32 with num_accepted's size");
    p.snap = num_accepted_snapshot->data_ptr<int32_t>();
    p.snap_n = static_cast<int>(num_accepted_snapshot->numel());
  }
  if (host_sampled.has_value()) {
    TORCH_CHECK(host_sampled->is_pinned() && host_sampled->scalar_type() == at::kLong &&
                    host_sampled->is_contiguous() && host_sampled->numel() >= p.M,
                "host_sampled: pinned contiguous i64 [>= M]");
    p.h_sampled = host_sampled->data_ptr<int64_t>();  // UVA: the host address is valid on the device
  }
  if (host_num_sampled.has_value()) {
    TORCH_CHECK(host_num_sampled->is_pinned() && host_num_sampled->scalar_type() == at::kInt &&
                    host_num_sampled->is_contiguous() && host_num_sampled->numel() >= p.M,
                "host_num_sampled: pinned contiguous i32 [>= M]");
    p.h_num_sampled = host_num_sampled->data_ptr<int32_t>();
  }
  launch_pdl(sample_finish_kernel, dim3(1), dim3(32 * p.M), stream(), p);
}

void embed_bcast(const torch::Tensor& ids, const torch::Tensor& weight, int64_t vocab,
                 int64_t rank, torch::Tensor& mailbox, int64_t mc, int64_t mode, torch::Tensor& out,
                 const std::optional<torch::Tensor>& gs) {
  TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == at::kBFloat16 && weight.dim() == 2 &&
                  weight.stride(1) == 1 && (weight.stride(0) * 2) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0,
              "weight must be bf16 [S, H] with 16-byte aligned rows");
  const int64_t shard = weight.size(0), H = weight.size(1);
  TORCH_CHECK(H % 8 == 0, "hidden size must be a multiple of 8");
  const int M = ids.numel();
  TORCH_CHECK(M >= 1 && ids.dim() == 1, "ids must be 1-D, non-empty");
  TORCH_CHECK(mailbox.is_cuda() && mailbox.scalar_type() == at::kBFloat16 && mailbox.dim() == 2 &&
                  mailbox.size(1) == H && mailbox.size(0) >= M && mailbox.is_contiguous(),
              "mailbox must be contiguous bf16 [MAXM >= M, H]");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.dim() == 2 &&
                  out.size(0) == M && out.size(1) == H && out.stride(1) == 1 &&
                  (out.stride(0) * 2) % 16 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
              "out must be bf16 [M, H], 16-byte aligned rows");
  TORCH_CHECK(mode == 0 ? mc != 0 : mode == 1, "mode 0 needs a multicast address; mode 1 local");
  TORCH_CHECK(rank >= 0 && rank * shard < vocab + shard && vocab > 0, "bad rank / shard / vocab");
  const int frags = static_cast<int>(H / 8);
  dim3 grid((frags + kEbThreads - 1) / kEbThreads, M);
  launch_pdl(embed_bcast_kernel, grid, dim3(kEbThreads), stream(), iview(ids, "ids"),
             static_cast<const uint8_t*>(weight.data_ptr()), static_cast<int64_t>(weight.stride(0) * 2),
             vocab, shard, static_cast<int>(rank), static_cast<uint8_t*>(mailbox.data_ptr()),
             static_cast<unsigned long long>(mc), frags, static_cast<int>(mode),
             static_cast<uint8_t*>(out.data_ptr()), static_cast<int64_t>(out.stride(0) * 2), M,
             gs.has_value() ? reinterpret_cast<unsigned long long*>(gs->data_ptr()) : nullptr);
}

void gs_stamp(torch::Tensor& gs, torch::Tensor& ring) {
  TORCH_CHECK(gs.is_cuda() && gs.scalar_type() == at::kLong && gs.numel() >= 2 && ring.is_cuda() &&
                  ring.scalar_type() == at::kLong && ring.numel() >= 4 && (ring.numel() - 1) % 3 == 0,
              "gs i64 [2], ring i64 [3 n + 1]");
  gs_stamp_kernel<<<1, 32, 0, stream()>>>(reinterpret_cast<unsigned long long*>(gs.data_ptr()),
                                         reinterpret_cast<unsigned long long*>(ring.data_ptr()),
                                         static_cast<int>((ring.numel() - 1) / 3));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

TORCH_LIBRARY(k3step, m) {
  m.def("allgather(Tensor local, Tensor(a!) mailbox, int mc, int buf, int rank, int mode, "
        "Tensor(b!) out) -> ()");
  m.def("sample_finish(Tensor local_max, Tensor local_arg, Tensor(a!) mailbox, int mc, int buf, "
        "int rank, int mode, Tensor idx_mapping, Tensor seq_lens, Tensor cu_num_logits, "
        "Tensor prefill_len, Tensor(b!) sampled, Tensor(c!) num_sampled, Tensor(d!) num_rejected, "
        "Tensor? query_start_loc=None, Tensor(e!)? num_computed_tokens=None, "
        "Tensor(f!)? last_sampled_tokens=None, Tensor(g!)? total_len=None, "
        "Tensor(h!)? all_token_ids=None, Tensor(i!)? output_bin_counts=None, "
        "Tensor(j!)? num_accepted=None, Tensor(k!)? my_packet=None, "
        "Tensor(l!)? num_accepted_snapshot=None, Tensor(m!)? host_sampled=None, "
        "Tensor(n!)? host_num_sampled=None) -> ()");
  m.def("embed_bcast(Tensor ids, Tensor weight, int vocab, int rank, Tensor(a!) mailbox, int mc, "
        "int mode, Tensor(b!) out, Tensor(c!)? gs=None) -> ()");
  m.def("gs_stamp(Tensor(a!) gs, Tensor(b!) ring) -> ()");
}

TORCH_LIBRARY_IMPL(k3step, CUDA, m) {
  m.impl("allgather", &allgather);
  m.impl("sample_finish", &sample_finish);
  m.impl("embed_bcast", &embed_bcast);
  m.impl("gs_stamp", &gs_stamp);
}
