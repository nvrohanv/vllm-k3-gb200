// Kimi K3 post-attention chain for decode (GB200, TP <= 16), fused:
//     o_proj GEMV  ->  TP all-reduce  ->  post-attention AttnRes (KimiDecoderLayer._post_attn_norm)
//
// k3oproj::fused (production path): ONE PDL kernel replaces three (o_proj GEMM, flashinfer one-shot
// all-reduce, kimi_k3_attn_res). Grid = 16 clusters x 7 CTAs = 112 CTAs x 512 threads:
//   * Producer phase (all 112 CTAs): CTA owns 4 row tiles of 16 rows (64 rows of W[7168, 768]);
//     warp = (row tile, K quarter). Its 96 KB W slice is loaded into REGISTERS before
//     griddepcontrol.wait (weights do not depend on the predecessor). After the wait: x (optionally
//     gated for MLA: x * sigmoid(gate)), mma.sync.m16n8k16 (A = 16 weight rows, B = x^T with tokens
//     in N; identical k permutation for A and B so every lane loads 16 contiguous bytes), K-quarter
//     reduction through smem, bf16 round, -0.0 -> +0.0, and one 16-byte multimem.st (NVLS) per
//     fragment into slot [rank] of the symmetric Lamport mailbox [nbuf][world][max_m][7168] of
//     every GPU. Test modes store into the local mailbox instead.
//   * Consumer phase (cluster c < M handles token c; other clusters exit): the AttnRes inputs
//     (weights, prefix, block rows) were cp.async-prefetched into smem at kernel start. Block
//     norms/dots, cluster exchange (st.async + mbarrier), block-only softmax mix B; then 4 thread
//     groups of 128 each poll + sum + re-arm 4 of the 16 slots of every fragment (the per-SM L2 read
//     of 16 x 16 B per fragment is the critical step; 4x more threads makes it ~2x faster), group
//     sums combine in a fixed order (bit-identical on every rank), bf16 round (= all-reduce output),
//     prefix' = bf16(prefix + attn) (HAS_DELTA) or attn (block-write layers), 4-scalar post
//     exchange, RMS-normed output (tailattn's algebra, agents/tailattn/lamport_attn_res.cu).
//   * PDL: launch_dependents at entry. The mailbox data is the readiness signal (Lamport): every
//     32-bit word of an empty fragment is 0x80000000, producers never publish bf16 -0.0. The
//     consumer phase polls only after this kernel's griddepcontrol.wait (in the producer phase), so
//     the previous call on this GPU has completed (and re-armed its slots) before it reads.
//
// k3oproj::produce / k3oproj::consume: the same two phases as separate kernels (M = 1 in the
// integration, M > fused_max_m, tests). The consumer never executes griddepcontrol.wait; the
// producer triggers its dependents only AFTER its own griddepcontrol.wait, which orders every
// consumer after the previous call on this GPU (else a consumer could poll a buffer that the
// previous consumer of that buffer has not re-armed yet).
//
// Cross-rank mailbox reuse: callers alternate two buffers (layer parity). Rank s writes buffer b
// again (two calls later) only after its own intermediate call completed, which needed rank r's
// data of that call, which r published only after its call on buffer b completed and re-armed it.
//
// Constraint (as for tailattn): prefix / blocks / AttnRes weights are read at kernel start
// (cp.async, before griddepcontrol.wait). They must be complete when this kernel launches, i.e.
// some kernel between their writer (the pre-attention AttnRes) and this one must trigger its
// dependents only after its own griddepcontrol.wait (true in K3: the q/in-proj GEMMs).
//
// AttnRes semantics (vllm/models/kimi_k3/nvidia/ops/attn_res.py):
//   attn     = bf16(sum over ranks of the o_proj partials)       (fixed order, see consume_post)
//   prefix'  = HAS_DELTA ? bf16(prefix + attn) : attn            (stored to `prefix`)
//   blocks[t, write_idx] = prefix'                               (write_idx >= 0; post-attn: -1)
//   sources  = blocks[t, 0..NB-1], prefix';  logit_s = dot(v_s, nw*qw) * rsqrt(mean(v_s^2)+eps)
//   out      = RMSNorm_ow(softmax mix)  (or the plain mix without output norm weight)
#include <cuda_bf16.h>
#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

namespace {

constexpr int kHidden = 7168;   // o_proj N and AttnRes width
constexpr int kK = 768;         // o_proj K per rank (6 heads x 128 for KDA and MLA)
constexpr int kTiles = kHidden / 16;
constexpr int kKBlocks = kK / 32;
constexpr int kMaxWorld = 16;
constexpr int kMaxM = 16;
constexpr uint32_t kSentinel = 0x80000000u;  // empty marker, one per 32-bit word

// Consumer geometry (as tailattn): 7-CTA cluster per token, 128 fragments of 8 bf16 per CTA.
constexpr int kVec = 8;
constexpr int kV2 = kVec / 2;
constexpr int kC = 7;
constexpr int kThreads = kHidden / (kC * kVec);  // 128
constexpr int kWarps = kThreads / 32;
constexpr int kMaxNB = 8;
constexpr float kInvH = 1.0f / kHidden;
constexpr int kProbes = 16;

// Fused kernel geometry: 112 CTAs (each 4 row tiles x 4 K quarters = 16 warps) in clusters of
// kFusedCS CTAs. With kFusedCS = 8 (14 clusters) all clusters are co-resident on GB200 (7-CTA
// clusters: only 15 of 16 fit); CTA rank 7 only produces, ranks 0..6 also run the consumer.
#ifndef K3OPROJ_FUSED_CS
#define K3OPROJ_FUSED_CS 8
#endif
constexpr int kFusedKS = 4, kFusedRT = 4, kFusedG = 4;
constexpr int kFusedCS = K3OPROJ_FUSED_CS;
static_assert(kFusedCS >= kC && kFusedCS <= 8, "fused cluster must hold the 7 consumer CTAs");
constexpr int kFusedCTAs = kTiles / kFusedRT;             // 112
constexpr int kFusedClusters = kFusedCTAs / kFusedCS;     // 14 (CS=8) or 16 (CS=7)
static_assert(kFusedClusters * kFusedCS == kFusedCTAs, "fused grid must cover all row tiles");
static_assert(kFusedKS * kFusedRT * 32 == kThreads * kFusedG, "fused phases must use the same block");

// Probe slots in the per-CTA timestamp record (profiling only).
enum Probe : int {
  kPStart = 0, kPLoads = 1, kPB = 2, kPAttn = 3, kPPush = 4, kPRecv = 5, kPTot = 6, kPOut = 7,
  kPPoll = 8, kPSum0 = 9, kPRounds = 10,
  kQStart = 11, kQWait = 12, kQX = 13, kQRed = 14, kQStored = 15,
};

__host__ __device__ constexpr int pow2ceil(int x) { int p = 1; while (p < x) p <<= 1; return p; }

struct alignas(16) U4 { uint32_t w[4]; };

// ------------------------------------------------------------------------------------------
// memory / math helpers
// ------------------------------------------------------------------------------------------
__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ U4 ldg_w(const void* ptr, uint64_t pol) {  // static weights, streamed once
  U4 u;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(u.w[0]), "=r"(u.w[1]), "=r"(u.w[2]), "=r"(u.w[3]) : "l"(ptr), "l"(pol));
  return u;
}
__device__ __forceinline__ U4 ldg_nc(const void* p) {  // read-only for the kernel's lifetime
  U4 u;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(u.w[0]), "=r"(u.w[1]), "=r"(u.w[2]), "=r"(u.w[3]) : "l"(p));
  return u;
}
__device__ __forceinline__ U4 ldg_cg(const void* p) {  // written by an earlier kernel (L2 only)
  U4 u;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(u.w[0]), "=r"(u.w[1]), "=r"(u.w[2]), "=r"(u.w[3]) : "l"(p));
  return u;
}
// L1-cached load of data written by the predecessor grid (valid after griddepcontrol.wait, which
// makes the prerequisite grid's writes visible to this grid). All 16 warps of a CTA read the same
// x, so L1 turns ~16 L2 requests per line and CTA into one.
__device__ __forceinline__ U4 ldg_ca(const void* p) {
  U4 u;
  asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
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
// W1-3 moefront: publish x_moe as Lamport words (bf16 -0.0 -> +0.0 so no 32-bit word equals the
// 0x80000000 sentinel) into every replica of the MoE front kernel's x_moe slot, then one relaxed hint
// add per warp (no fence: k3mf.moe_block_front validates the data).
__device__ __forceinline__ uint32_t xmoe_canon(uint32_t w) {
  if ((w & 0xffffu) == 0x8000u) w &= 0xffff0000u;
  if ((w >> 16) == 0x8000u) w &= 0x0000ffffu;
  return w;
}
__device__ __forceinline__ void xmoe_publish(__nv_bfloat16* dst, int reps, long long rep_stride, unsigned* ctr,
                                             const U4& u) {
  const uint32_t a = xmoe_canon(u.w[0]), b = xmoe_canon(u.w[1]), c = xmoe_canon(u.w[2]), d = xmoe_canon(u.w[3]);
  for (int rp = 0; rp < reps; ++rp)
    asm volatile("st.relaxed.gpu.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(dst + rp * rep_stride), "r"(a), "r"(b),
                 "r"(c), "r"(d) : "memory");
  __syncwarp();
  if ((threadIdx.x & 31) == 0) asm volatile("red.relaxed.gpu.global.add.u32 [%0], 1;" ::"l"(ctr) : "memory");
}
__device__ __forceinline__ void multimem_st16(unsigned long long addr, const U4& u) {
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(addr),
               "r"(u.w[0]), "r"(u.w[1]), "r"(u.w[2]), "r"(u.w[3]) : "memory");
}
__device__ __forceinline__ uint64_t policy_evict_last() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ void prefetch_l2_evict_last(const void* ptr, uint32_t bytes, uint64_t pol) {
  asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" ::"l"(ptr), "r"(bytes), "l"(pol)
               : "memory");
}
__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* smem_dst, const void* gsrc) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_addr(smem_dst)), "l"(gsrc) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ bool dirty(const U4& u) {
  return (u.w[0] == kSentinel) | (u.w[1] == kSentinel) | (u.w[2] == kSentinel) |
         (u.w[3] == kSentinel);
}
__device__ __forceinline__ uint32_t no_neg_zero(uint32_t w) {  // bf16 -0.0 -> +0.0 (both halves)
  if ((w & 0xffffu) == 0x8000u) w &= 0xffff0000u;
  if ((w >> 16) == 0x8000u) w &= 0x0000ffffu;
  return w;
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
__device__ __forceinline__ float2 round_bf16(float2 x) {
  return __bfloat1622float2(__float22bfloat162_rn(x));
}
__device__ __forceinline__ float dot8(const float2 (&a)[kV2], const float2 (&b)[kV2]) {
  float2 s = __fmul2_rn(a[0], b[0]);
  float2 s1 = __fmul2_rn(a[1], b[1]);
  s = __ffma2_rn(a[2], b[2], s);
  s1 = __ffma2_rn(a[3], b[3], s1);
  s = __fadd2_rn(s, s1);
  return s.x + s.y;
}
__device__ __forceinline__ float sigmoid_fast(float v) { return __fdividef(1.0f, 1.0f + __expf(-v)); }
constexpr int kXsStride = kK / 8 + 1;  // gated-x smem row stride in 16-byte chunks (97: conflict-free)
constexpr int kXsBytes = kMaxM * kXsStride * 16;
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ unsigned long long gtimer_after(float dep) {
  unsigned long long t;
  asm volatile("{ .reg .f32 d; mov.f32 d, %1; mov.u64 %0, %globaltimer; }" : "=l"(t) : "f"(dep));
  return t;
}
__device__ __forceinline__ unsigned long long gtimer_after_u(uint32_t dep) {
  unsigned long long t;
  asm volatile("{ .reg .b32 d; mov.b32 d, %1; mov.u64 %0, %globaltimer; }" : "=l"(t) : "r"(dep));
  return t;
}
// D(16x8) += A(16x16 row-major) * B(16x8 col-major), bf16 in, fp32 accumulate.
__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                         uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
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
// Asynchronous DSMEM store of one fp32 into CTA `rank`'s copy of `local`, completing 4 bytes on
// that CTA's copy of mbarrier `bar`.
__device__ __forceinline__ void st_async(const float* local, const uint64_t* bar, int rank, float v) {
  uint32_t ra, rb;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(smem_addr(local)), "r"(rank));
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rb) : "r"(smem_addr(bar)), "r"(rank));
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(ra),
               "r"(__float_as_uint(v)), "r"(rb) : "memory");
}
__device__ __forceinline__ void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_wait() {
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ void named_sync(int id, int n) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void named_arrive(int id, int n) {
  asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory");
}

// Butterfly reduce-scatter of N (power of two) per-lane values across the warp (xor O, O/2, .., 1).
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

// ==========================================================================================
// PRODUCER PHASE
// ==========================================================================================
enum StoreMode : int {
  kModeMulticast = 0,  // multimem.st into slot [slot] of every rank's mailbox (production)
  kModeLocal = 1,      // plain store into slot [slot] of the local mailbox (single-GPU tests)
  kModeLocalAll = 2,   // plain stores of this partial into all `world` local slots (emulation)
  kModePlain = 3,      // plain [M, 7168] output y (GEMV test, no mailbox)
};

struct ProdParams {
  const __nv_bfloat16* x;
  const __nv_bfloat16* gate;
  const __nv_bfloat16* w;
  const __nv_bfloat16* w_next;  // optional: next layer's o_proj weight, L2-prefetched (evict_last)
  __nv_bfloat16* mbox;       // local mailbox, start of the selected buffer: [world][max_m][7168]
  unsigned long long mc;     // multicast VA of the same location (mode 0)
  __nv_bfloat16* y;
  long long x_sm, g_sm, y_sm;
  int M, slot, world, max_m, mode;
  unsigned long long* ts;    // optional per-CTA %globaltimer probes [grid][kProbes]
};

template <int KS, int RT, int NT>
struct ProdSmem {
  static constexpr int kPad = 20;  // floats per token row: conflict-free writes, 16-B aligned reads
  float red[NT][RT][KS][8][kPad];  // [token group][row tile][K part][token][row + pad]
};

// W slice of one warp: 16 rows (g, g+8) x KB 32-wide k blocks, 16 bytes per lane and row.
template <int KB>
struct WRegs {
  U4 a[KB], b[KB];
};

// Issue the (predecessor-independent) weight loads of this warp's (row tile, K part) into registers.
template <int KS, int RT>
__device__ __forceinline__ WRegs<kKBlocks / KS> produce_load_w(const ProdParams& p, int cta) {
  constexpr int KB = kKBlocks / KS;
  static_assert(KB * KS == kKBlocks, "KS must divide 24");
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int rt = warp / KS, kq = warp - (warp / KS) * KS;
  const int tile = cta * RT + rt;
  WRegs<KB> W;
  if (tile < kTiles) {
    const uint64_t pol = policy_evict_first();
    const __nv_bfloat16* w0 = p.w + static_cast<size_t>(tile * 16 + g) * kK + kq * KB * 32 + t * 8;
#pragma unroll
    for (int j = 0; j < KB; ++j) {
      W.a[j] = ldg_w(w0 + j * 32, pol);
      W.b[j] = ldg_w(w0 + 8 * kK + j * 32, pol);
    }
  } else {
#pragma unroll
    for (int j = 0; j < KB; ++j)
#pragma unroll
      for (int i = 0; i < 4; ++i) W.a[j].w[i] = W.b[j].w[i] = 0u;
  }
  return W;
}

// griddepcontrol.wait, x, mma, K-part reduction, epilogue store. All KS*RT*32 threads of the block
// must call this (it contains a __syncthreads).
// xs: GATE only, >= kXsBytes of shared memory for the gated x (x * sigmoid(gate), bf16).
// TRIGGER_AFTER_WAIT: issue griddepcontrol.launch_dependents right after griddepcontrol.wait
// (stand-alone producer, see oproj_produce_kernel).
template <int KS, int RT, int NT, bool GATE, bool TRIGGER_AFTER_WAIT = false>
__device__ __forceinline__ void produce_compute(const ProdParams& p, int cta, ProdSmem<KS, RT, NT>& sm,
                                                const WRegs<kKBlocks / KS>& W, U4* xs, unsigned long long* tp) {
  constexpr int KB = kKBlocks / KS;
  constexpr int kPad = ProdSmem<KS, RT, NT>::kPad;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int rt = warp / KS, kq = warp - (warp / KS) * KS;
  const int tile = cta * RT + rt;
  const bool active = tile < kTiles;
  const int kofs = kq * KB * 32 + t * 8;

  asm volatile("griddepcontrol.wait;" ::: "memory");
  if constexpr (TRIGGER_AFTER_WAIT) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (tp) tp[kQWait] = gtimer();

  // ---- x (tokens in the mma N dimension). MLA: x * sigmoid(gate) is computed once per CTA by all
  // threads (fp32 math, one bf16 rounding, like vLLM's compiled _gate_sigmoid_mul) into smem. ----
  U4 xr[NT][KB];
  if constexpr (GATE) {
    const int nchunk = p.M * (kK / 8);
    for (int c = tid; c < nchunk; c += KS * RT * 32) {
      const int tok = c / (kK / 8), kc = c - tok * (kK / 8);
      const U4 xv = ldg_ca(p.x + static_cast<long long>(tok) * p.x_sm + kc * 8);
      const U4 gv = ldg_ca(p.gate + static_cast<long long>(tok) * p.g_sm + kc * 8);
      float2 a[kV2], b[kV2];
      unpack(xv, a);
      unpack(gv, b);
#pragma unroll
      for (int i = 0; i < kV2; ++i) a[i] = make_float2(a[i].x * sigmoid_fast(b[i].x), a[i].y * sigmoid_fast(b[i].y));
      xs[tok * kXsStride + kc] = pack(a);
    }
    __syncthreads();
  }
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int tok = nt * 8 + g;
    if (active && tok < p.M) {
      if constexpr (GATE) {
#pragma unroll
        for (int j = 0; j < KB; ++j) xr[nt][j] = xs[tok * kXsStride + (kofs >> 3) + j * 4];
      } else {
        const __nv_bfloat16* xp = p.x + static_cast<long long>(tok) * p.x_sm + kofs;
#pragma unroll
        for (int j = 0; j < KB; ++j) xr[nt][j] = ldg_ca(xp + j * 32);
      }
    } else {
#pragma unroll
      for (int j = 0; j < KB; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) xr[nt][j].w[i] = 0u;
    }
  }
  if (tp) tp[kQX] = gtimer_after_u(xr[0][KB - 1].w[3] ^ W.b[KB - 1].w[3]);

  // ---- mma: four accumulator chains per token group. ----
  float acc[NT][4][4];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt)
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[nt][c][i] = 0.f;
#pragma unroll
  for (int j = 0; j < KB; ++j) {
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      // logical k (2t,2t+1) <-> physical 8t..8t+1, (2t+8,2t+9) <-> 8t+2..8t+3 (first mma),
      // then 8t+4..8t+7 (second mma); identical for A (weights) and B (x).
      mma16816(acc[nt][(2 * j) & 3], W.a[j].w[0], W.b[j].w[0], W.a[j].w[1], W.b[j].w[1], xr[nt][j].w[0],
               xr[nt][j].w[1]);
      mma16816(acc[nt][(2 * j + 1) & 3], W.a[j].w[2], W.b[j].w[2], W.a[j].w[3], W.b[j].w[3], xr[nt][j].w[2],
               xr[nt][j].w[3]);
    }
  }

  // ---- K-part reduction through smem. D[row][token]: c0 (g,2t) c1 (g,2t+1) c2 (g+8,2t) c3 (g+8,2t+1).
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    float* r = &sm.red[nt][rt][kq][0][0];
    float d[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) d[i] = (acc[nt][0][i] + acc[nt][1][i]) + (acc[nt][2][i] + acc[nt][3][i]);
    r[(2 * t) * kPad + g] = d[0];
    r[(2 * t + 1) * kPad + g] = d[1];
    r[(2 * t) * kPad + g + 8] = d[2];
    r[(2 * t + 1) * kPad + g + 8] = d[3];
  }
  __syncthreads();
  if (tp) tp[kQRed] = gtimer();

  // ---- Epilogue: thread f = (m, rt2, h) owns one 16-byte fragment (8 rows of token m). ----
  if (tid < 2 * RT * p.M) {
    const int h = tid & 1;
    const int rt2 = (tid >> 1) % RT;
    const int m = tid / (2 * RT);
    const int tile2 = cta * RT + rt2;
    if (tile2 < kTiles) {
      const int nt = m >> 3, mm = m & 7;
      float4 pa[KS], pb[KS];
#pragma unroll
      for (int q = 0; q < KS; ++q) {
        pa[q] = *reinterpret_cast<const float4*>(&sm.red[nt][rt2][q][mm][h * 8]);
        pb[q] = *reinterpret_cast<const float4*>(&sm.red[nt][rt2][q][mm][h * 8 + 4]);
      }
#pragma unroll
      for (int q = 1; q < KS; ++q) {
        pa[0].x += pa[q].x; pa[0].y += pa[q].y; pa[0].z += pa[q].z; pa[0].w += pa[q].w;
        pb[0].x += pb[q].x; pb[0].y += pb[q].y; pb[0].z += pb[q].z; pb[0].w += pb[q].w;
      }
      const float2 s4[kV2] = {make_float2(pa[0].x, pa[0].y), make_float2(pa[0].z, pa[0].w),
                              make_float2(pb[0].x, pb[0].y), make_float2(pb[0].z, pb[0].w)};
      U4 o = pack(s4);
      const int col = tile2 * 16 + h * 8;
      const long long base = static_cast<long long>(m) * kHidden + col;
      const long long row_elems = static_cast<long long>(p.max_m) * kHidden;
      if (p.mode == kModePlain) {
        st16(p.y + static_cast<long long>(m) * p.y_sm + col, o);
      } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) o.w[i] = no_neg_zero(o.w[i]);
        if (p.mode == kModeMulticast) {
          multimem_st16(p.mc + static_cast<unsigned long long>(p.slot * row_elems + base) * 2ull, o);
        } else if (p.mode == kModeLocal) {
          st16(p.mbox + p.slot * row_elems + base, o);
        } else {  // kModeLocalAll (emulation): this partial into every slot
          for (int r = 0; r < p.world; ++r) st16(p.mbox + r * row_elems + base, o);
        }
      }
    }
  }
  if (tp) tp[kQStored] = gtimer();
  // Cross-layer prefetch: this CTA's rows of the NEXT layer's o_proj weight (a contiguous
  // RT*16*1536-byte range) into L2 with evict_last priority, so that the next call's weight loads
  // hit L2 even if its predecessor gives no PDL prefetch window. Off the critical path (after the
  // stores); the next call reads them with evict_first, which demotes them again.
  if (p.w_next != nullptr && warp == 0 && lane < RT * 2) {
    const int tile_pf = cta * RT + (lane >> 1);
    if (tile_pf < kTiles) {
      const char* base = reinterpret_cast<const char*>(p.w_next) + static_cast<size_t>(tile_pf) * 16 * kK * 2;
      prefetch_l2_evict_last(base + (lane & 1) * 8 * kK * 2, 8 * kK * 2, policy_evict_last());
    }
  }
}

template <int KS, int RT, int NT, bool GATE>
__global__ void __launch_bounds__(KS * RT * 32, 1)
oproj_produce_kernel(const ProdParams p) {
  __shared__ ProdSmem<KS, RT, NT> sm;
  __shared__ U4 xs[GATE ? kMaxM * kXsStride : 1];
  const bool prof = p.ts != nullptr;
  unsigned long long tp[kProbes];  // profiling only
  if (prof) {
    for (int k = 0; k < kProbes; ++k) tp[k] = 0;
    tp[kQStart] = gtimer();
  }
  // The consumer is launched only once every producer CTA has passed griddepcontrol.wait, i.e. once
  // the producer's predecessor (in a chain of layers: the previous consumer) has completed. The
  // consumer never waits itself, so this is what orders a consumer after the previous consumer of
  // the same mailbox buffer (two layers back), even in chains without other waiting kernels.
  const WRegs<kKBlocks / KS> W = produce_load_w<KS, RT>(p, blockIdx.x);
  produce_compute<KS, RT, NT, GATE, true>(p, blockIdx.x, sm, W, xs, prof ? tp : nullptr);
  if (prof && threadIdx.x == 0) {
    unsigned long long* dst = p.ts + static_cast<long long>(blockIdx.x) * kProbes;
#pragma unroll
    for (int k = 0; k < kProbes; ++k) dst[k] = tp[k];
  }
}

// ==========================================================================================
// CONSUMER PHASE
// ==========================================================================================
struct ConsParams {
  __nv_bfloat16* mbox;       // local mailbox, start of the selected buffer: [world][max_m][7168]
  __nv_bfloat16* prefix;     // HAS_DELTA: read + written in place; else: receives prefix' = attn
  __nv_bfloat16* blocks;
  const __nv_bfloat16* norm_w;
  const __nv_bfloat16* qk_w;
  const __nv_bfloat16* out_w;
  __nv_bfloat16* out;
  __nv_bfloat16* attn_out;   // optional: the all-reduced o_proj output (bf16)
  __nv_bfloat16* xmoe;       // optional (W1-3): x_moe slot of this layer's parity, replica 0 ([rows][7168])
  long long xmoe_rs;         // elements between replicas
  unsigned* xmoe_ctr;        // x_moe hint counter of this parity
  int xmoe_reps;
  long long slot_stride;     // elements between consecutive slots (max_m * 7168)
  long long prefix_sm, out_sm, attn_sm, blocks_sm, blocks_sr;
  int M, world, write_idx, wait_prior, poll;
  float eps, out_eps;
  unsigned long long* ts;    // optional [M][C][kProbes]
};

template <int G>
struct ConsSmem {
  float wpre[kWarps][2 * kMaxNB];
  float pre[kC][2 * kMaxNB];
  alignas(16) float blk[2 * kMaxNB];
  float post[kC][kWarps][4];
  alignas(16) float tot[4];
  alignas(8) uint64_t bar[2];  // [0]: pre exchange, [1]: post exchange
  alignas(16) float4 part[G > 1 ? G - 1 : 1][2][kThreads];  // loader groups' partial sums
};
// AttnRes inputs, cp.async-prefetched at kernel start by the group-0 threads:
// [0] nw, [1] qw, [2] ow, [3] prefix, [4 + s] block s  (one 16-byte fragment per thread).
struct PrefetchSmem {
  U4 frag[4 + kMaxNB][kThreads];
};
// Group-0 state carried from consume_pre (block logits) to consume_post (after the poll).
struct ConsState {
  float2 B[kV2];  // block-only softmax mix sum_s exp(l_s - m0) v_s
  float m0, Eb, qpart;
};

template <int NB, bool OUT_NORM, int G>
__device__ __forceinline__ void consume_setup(ConsSmem<G>& cs) {
  constexpr int NPRE = 2 * NB;
  constexpr int NPOST = NB > 0 ? (OUT_NORM ? 4 : 2) : (OUT_NORM ? 1 : 0);
  if constexpr (NPOST > 0) {
    if (threadIdx.x == 0) {
      if constexpr (NB > 0) {
        mbar_init(&cs.bar[0], 1);
        mbar_arrive_expect_tx(&cs.bar[0], kC * NPRE * 4);
      }
      mbar_init(&cs.bar[1], 1);
      mbar_arrive_expect_tx(&cs.bar[1], kC * kWarps * NPOST * 4);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    cluster_arrive_relaxed();  // "this CTA is running and its mbarriers are initialized"
  }
}

// Group-0 threads (thread = fragment): cp.async prefetch of this fragment's AttnRes inputs.
template <int NB, bool OUT_NORM, bool HAS_DELTA>
__device__ __forceinline__ void consume_prefetch(const ConsParams& p, int rank, int t, PrefetchSmem& pf) {
  const int tid = threadIdx.x;
  const int h = (rank * kThreads + tid) * kVec;
  if constexpr (NB > 0) {
    cp_async16(&pf.frag[0][tid], p.norm_w + h);
    cp_async16(&pf.frag[1][tid], p.qk_w + h);
  }
  if constexpr (OUT_NORM) cp_async16(&pf.frag[2][tid], p.out_w + h);
  if constexpr (HAS_DELTA) cp_async16(&pf.frag[3][tid], p.prefix + static_cast<long long>(t) * p.prefix_sm + h);
  const long long brow = static_cast<long long>(t) * p.blocks_sm + h;
#pragma unroll
  for (int s = 0; s < NB; ++s) cp_async16(&pf.frag[4 + s][tid], p.blocks + brow + s * p.blocks_sr);
  cp_async_commit();
}

__device__ __forceinline__ void load_w(const PrefetchSmem& pf, int tid, float2 (&w)[kV2]) {
  float2 a2[kV2], b2[kV2];
  unpack(pf.frag[0][tid], a2);
  unpack(pf.frag[1][tid], b2);
#pragma unroll
  for (int i = 0; i < kV2; ++i) w[i] = __fmul2_rn(a2[i], b2[i]);
}

// Group 0, before the poll: block norms/dots -> cluster exchange -> block logits -> B.
// The block rows are streamed from the prefetch buffer (few live registers, so the fused kernel
// can run this while its weight registers are in flight).
template <int NB, bool OUT_NORM, int G>
__device__ __forceinline__ ConsState consume_pre(const ConsParams& p, int rank, ConsSmem<G>& cs,
                                                 const PrefetchSmem& pf, unsigned long long* tp) {
  constexpr int NPRE = 2 * NB;
  constexpr int KPRE = pow2ceil(NPRE > 0 ? NPRE : 1);
  constexpr int NPOST = NB > 0 ? (OUT_NORM ? 4 : 2) : (OUT_NORM ? 1 : 0);
  constexpr bool EXCH = NPOST > 0;
  constexpr int kBarMain = 1;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  ConsState st;
#pragma unroll
  for (int i = 0; i < kV2; ++i) st.B[i] = make_float2(0.f, 0.f);
  st.m0 = st.Eb = st.qpart = 0.f;
  asm volatile("cp.async.wait_group 0;" ::: "memory");  // this thread's own prefetch
  if (p.wait_prior) asm volatile("griddepcontrol.wait;" ::: "memory");
  if (tp) tp[kPLoads] = gtimer();
  if constexpr (NB > 0) {
    float2 w[kV2];
    load_w(pf, tid, w);
    float a[KPRE];
#pragma unroll
    for (int k = 0; k < KPRE; ++k) a[k] = 0.f;
#pragma unroll
    for (int s = 0; s < NB; ++s) {
      float2 v[kV2];
      unpack(pf.frag[4 + s][tid], v);
      a[s] = dot8(v, v);
      a[NB + s] = dot8(v, w);
    }
    RS<KPRE, 16>::run(a, lane);
    {
      constexpr int kHeld = KPRE >= 32 ? KPRE / 32 : 1;
      if ((lane & (rs_dup<KPRE>() - 1)) == 0) {
#pragma unroll
        for (int j = 0; j < kHeld; ++j) {
          const int idx = rs_index<KPRE>(j, lane);
          if (idx < NPRE) cs.wpre[warp][idx] = a[j];
        }
      }
    }
    named_sync(kBarMain, kThreads);
    cluster_wait();  // every CTA of the cluster is resident with initialized mbarriers
    if (tid < NPRE) {
      float s = 0.f;
#pragma unroll
      for (int k = 0; k < kWarps; ++k) s += cs.wpre[k][tid];
#pragma unroll
      for (int r = 0; r < kC; ++r) st_async(&cs.pre[rank][tid], &cs.bar[0], r, s);
    }
    mbar_wait(&cs.bar[0], 0);
    if (tid < NPRE) {
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int r = 0; r < kC; ++r) {
        if (r & 1) s1 += cs.pre[r][tid];
        else s0 += cs.pre[r][tid];
      }
      cs.blk[tid] = s0 + s1;
    }
    named_sync(kBarMain, kThreads);
    float T[2 * kMaxNB];
#pragma unroll
    for (int k = 0; k < NPRE; k += 4) {
      const float4 x = *reinterpret_cast<const float4*>(&cs.blk[k]);
      T[k] = x.x;
      T[k + 1] = x.y;
      T[k + 2] = x.z;
      T[k + 3] = x.w;
    }
    float lg[NB], e[NB];
#pragma unroll
    for (int s = 0; s < NB; ++s) lg[s] = T[NB + s] * rsqrtf(fmaf(T[s], kInvH, p.eps));
    st.m0 = tree_max(lg, NB);
#pragma unroll
    for (int s = 0; s < NB; ++s) e[s] = __expf(lg[s] - st.m0);
    st.Eb = tree_sum(e, NB);
#pragma unroll
    for (int s = 0; s < NB; ++s) {
      float2 v[kV2];
      unpack(pf.frag[4 + s][tid], v);
#pragma unroll
      for (int i = 0; i < kV2; ++i) st.B[i] = __ffma2_rn(make_float2(e[s], e[s]), v[i], st.B[i]);
    }
    if constexpr (OUT_NORM) st.qpart = dot8(st.B, st.B);
  } else if constexpr (EXCH) {
    cluster_wait();
  }
  if (tp) tp[kPB] = gtimer_after(st.B[0].x + st.qpart);
  return st;
}

// Poll + sum slots [r0, r0 + S) (S <= SMAX) of one 16-byte fragment, in ascending slot order.
//  poll=0: every round reloads all still-pending slots.
//  poll=1: phase 1, lane l spins on ONE slot (r0 + l % S) until the whole warp has seen every slot
//          of the group (1 load per thread per round), then one round of all S slots.
// Fragments still carrying a sentinel word are re-polled individually (16-byte fragments of one
// slot may land out of order).
template <int SMAX>
__device__ __forceinline__ void poll_sum(const __nv_bfloat16* mb, long long slot_stride, int r0, int S,
                                         int poll, int lane, float2 (&A)[kV2], unsigned& rounds) {
  if (poll == 1) {
    const __nv_bfloat16* pp1 = mb + (r0 + lane % S) * slot_stride;
    bool ready = !dirty(ld_volatile16(pp1));
    while (!__all_sync(0xffffffffu, ready)) {
      if (!ready) ready = !dirty(ld_volatile16(pp1));
      ++rounds;
    }
  }
  U4 u[SMAX];
  unsigned pending = 0;
#pragma unroll
  for (int r = 0; r < SMAX; ++r)
    if (r < S) u[r] = ld_volatile16(mb + (r0 + r) * slot_stride);
#pragma unroll
  for (int r = 0; r < SMAX; ++r)
    if (r < S && dirty(u[r])) pending |= 1u << r;
  while (pending) {
#pragma unroll
    for (int r = 0; r < SMAX; ++r)
      if (pending & (1u << r)) u[r] = ld_volatile16(mb + (r0 + r) * slot_stride);
#pragma unroll
    for (int r = 0; r < SMAX; ++r)
      if ((pending & (1u << r)) && !dirty(u[r])) pending &= ~(1u << r);
    rounds += 1000;
  }
  unpack(u[0], A);
#pragma unroll
  for (int r = 1; r < SMAX; ++r) {
    if (r < S) {
      float2 d[kV2];
      unpack(u[r], d);
#pragma unroll
      for (int i = 0; i < kV2; ++i) A[i] = __fadd2_rn(A[i], d[i]);
    }
  }
}

// After consume_pre. G thread groups of 128 (threadIdx.x / 128): group q polls, sums and re-arms
// slots [q*S, (q+1)*S), S = ceil(world/G); group 0 then combines and runs the rest of AttnRes.
// The all-reduce value is
//   attn = bf16( ((g_0 + g_1) + g_2) + g_3 ),  g_q = ((s_{qS} + s_{qS+1}) + ...) in fp32,
// a fixed order, so every rank computes bit-identical results.
template <int NB, bool OUT_NORM, bool HAS_DELTA, int G>
__device__ __forceinline__ void consume_post(const ConsParams& p, int rank, int t, ConsSmem<G>& cs,
                                             const PrefetchSmem& pf, const ConsState& st,
                                             unsigned long long* tp) {
  constexpr int NPOST = NB > 0 ? (OUT_NORM ? 4 : 2) : (OUT_NORM ? 1 : 0);
  constexpr int KPOST = pow2ceil(NPOST > 0 ? NPOST : 1);
  constexpr bool EXCH = NPOST > 0;
  constexpr int iS = NB > 0 && OUT_NORM ? 2 : 0;
  constexpr int SMAX = kMaxWorld / G;
  constexpr int kBarMain = 1, kBarAll = 2;  // named barriers (0 is __syncthreads)

  const int grp = threadIdx.x / kThreads;
  const int tid = threadIdx.x - grp * kThreads;  // fragment index within the CTA
  const int warp = tid >> 5, lane = tid & 31;
  const int h = (rank * kThreads + tid) * kVec;
  const int S = (p.world + G - 1) / G;
  __nv_bfloat16* mb = p.mbox + static_cast<long long>(t) * kHidden + h;  // slot 0, row t
  const U4 sentinel = {{kSentinel, kSentinel, kSentinel, kSentinel}};

  if constexpr (G > 1) {
    if (grp > 0) {  // ---- loader groups: poll + sum + re-arm their slots, hand the sum over. ----
      if (p.wait_prior) asm volatile("griddepcontrol.wait;" ::: "memory");
      const int r0 = grp * S;
      const int n = min(S, p.world - r0);
      float2 A[kV2];
      unsigned rounds = 0;
      if (n > 0) {
        poll_sum<SMAX>(mb, p.slot_stride, r0, n, p.poll, lane, A, rounds);
      } else {
#pragma unroll
        for (int i = 0; i < kV2; ++i) A[i] = make_float2(0.f, 0.f);
      }
      cs.part[grp - 1][0][tid] = make_float4(A[0].x, A[0].y, A[1].x, A[1].y);
      cs.part[grp - 1][1][tid] = make_float4(A[2].x, A[2].y, A[3].x, A[3].y);
      named_arrive(kBarAll, kThreads * G);
#pragma unroll
      for (int r = 0; r < SMAX; ++r)
        if (r < n) st16(mb + (r0 + r) * p.slot_stride, sentinel);  // re-arm
      return;
    }
  }

  // ---- Group 0: its own slots, then the loader groups' partial sums (fixed order). ----
  unsigned rounds = 0;
  if (tp) tp[kPPoll] = gtimer();
  float2 A[kV2];
  const int n0 = min(S, p.world);
  poll_sum<SMAX>(mb, p.slot_stride, 0, n0, p.poll, lane, A, rounds);
  if (tp) {
    tp[kPSum0] = gtimer_after(A[0].x);
    tp[kPRounds] = rounds;
  }
  if constexpr (G > 1) {
    named_sync(kBarAll, kThreads * G);
#pragma unroll
    for (int q = 0; q < G - 1; ++q) {
      const float4 x0 = cs.part[q][0][tid], x1 = cs.part[q][1][tid];
      A[0] = __fadd2_rn(A[0], make_float2(x0.x, x0.y));
      A[1] = __fadd2_rn(A[1], make_float2(x0.z, x0.w));
      A[2] = __fadd2_rn(A[2], make_float2(x1.x, x1.y));
      A[3] = __fadd2_rn(A[3], make_float2(x1.z, x1.w));
    }
  }
  float2 P[kV2];
  {
    float2 old[kV2];
    if constexpr (HAS_DELTA) unpack(pf.frag[3][tid], old);
#pragma unroll
    for (int i = 0; i < kV2; ++i) {
      A[i] = round_bf16(A[i]);
      if constexpr (HAS_DELTA) P[i] = round_bf16(__fadd2_rn(old[i], A[i]));
      else P[i] = A[i];
    }
  }
  if (tp) tp[kPAttn] = gtimer_after(P[0].x);
  const U4 pp = pack(P);
  const long long prow = static_cast<long long>(t) * p.prefix_sm + h;
  const long long brow = static_cast<long long>(t) * p.blocks_sm + h;
  __nv_bfloat16* outp = p.out + static_cast<long long>(t) * p.out_sm + h;

  float2 w[kV2];
  if constexpr (NB > 0) load_w(pf, tid, w);
  if constexpr (EXCH) {
    // Only 4 scalars cross the cluster: <B,P>, |B|^2, |P|^2, <P,w> (see tailattn).
    float b[KPOST];
#pragma unroll
    for (int k = 0; k < KPOST; ++k) b[k] = 0.f;
    if constexpr (NB > 0) {
      if constexpr (OUT_NORM) {
        b[0] = dot8(st.B, P);
        b[1] = st.qpart;
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
        for (int r = lane & (dup - 1); r < kC; r += dup) st_async(&cs.post[rank][warp][idx], &cs.bar[1], r, b[0]);
      }
    }
    if (tp) tp[kPPush] = gtimer();
  }
  // Global side effects, off the exchange's critical path.
  st16(p.prefix + prow, pp);
  if (p.write_idx >= 0) st16(p.blocks + brow + p.write_idx * p.blocks_sr, pp);
#pragma unroll
  for (int r = 0; r < SMAX; ++r)
    if (r < n0) st16(mb + r * p.slot_stride, sentinel);  // re-arm group 0's slots
  if (p.attn_out != nullptr) st16(p.attn_out + static_cast<long long>(t) * p.attn_sm + h, pack(A));

  if constexpr (!EXCH) {
    st16(outp, pp);  // single source, no norm: out = prefix'
    if (p.xmoe) xmoe_publish(p.xmoe + static_cast<long long>(t) * kHidden + h, p.xmoe_reps, p.xmoe_rs, p.xmoe_ctr, pp);
    if (tp) tp[kPOut] = gtimer();
  } else {
    mbar_wait(&cs.bar[1], 0);
    if (tp) tp[kPRecv] = gtimer();
    if (tid < NPOST) {
      float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int r = 0; r < kC; ++r)
#pragma unroll
        for (int q = 0; q < kWarps; ++q) s[(r * kWarps + q) & 3] += cs.post[r][q][tid];
      cs.tot[tid] = (s[0] + s[1]) + (s[2] + s[3]);
    }
    named_sync(kBarMain, kThreads);
    if (tp) tp[kPTot] = gtimer();
    const float4 T = *reinterpret_cast<const float4*>(cs.tot);
    float2 ow[kV2];
    if constexpr (OUT_NORM) unpack(pf.frag[2][tid], ow);
    float2 o[kV2];
    if constexpr (NB > 0) {
      const float X = T.x, Q = T.y;
      const float SP = iS == 2 ? T.z : T.x, DP = iS == 2 ? T.w : T.y;
      const float lp = DP * rsqrtf(fmaf(SP, kInvH, p.eps));
      const float mx = fmaxf(st.m0, lp);
      const float alpha = __expf(st.m0 - mx), ep = __expf(lp - mx);
      const float den = fmaf(alpha, st.Eb, ep);
      float ca, cp;  // out = ca * B + cp * P  (times ow)
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
        const float2 acc = __ffma2_rn(make_float2(ca, ca), st.B[i], __fmul2_rn(make_float2(cp, cp), P[i]));
        o[i] = OUT_NORM ? __fmul2_rn(acc, ow[i]) : acc;
      }
    } else {
      const float scale = rsqrtf(fmaf(T.x, kInvH, p.out_eps));
#pragma unroll
      for (int i = 0; i < kV2; ++i) o[i] = __fmul2_rn(__fmul2_rn(make_float2(scale, scale), P[i]), ow[i]);
    }
    const U4 po = pack(o);
    st16(outp, po);
    if (p.xmoe) xmoe_publish(p.xmoe + static_cast<long long>(t) * kHidden + h, p.xmoe_reps, p.xmoe_rs, p.xmoe_ctr, po);
    if (tp) tp[kPOut] = gtimer_after(o[0].x);
  }
}

// Stand-alone consumer (one 7-CTA cluster per token, 128*G threads per CTA).
template <int NB, bool OUT_NORM, bool HAS_DELTA, int G>
__global__ void __launch_bounds__(kThreads * G)
oproj_consume_kernel(const ConsParams p) {
  __shared__ ConsSmem<G> cs;
  __shared__ PrefetchSmem pf;
  const bool prof = p.ts != nullptr;
  unsigned long long tp[kProbes];  // profiling only
  if (prof) {
    for (int k = 0; k < kProbes; ++k) tp[k] = 0;
    tp[kPStart] = gtimer();
  }
  consume_setup<NB, OUT_NORM, G>(cs);
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const int rank = blockIdx.x, t = blockIdx.y;
  ConsState st;
  if (threadIdx.x < kThreads) {
    consume_prefetch<NB, OUT_NORM, HAS_DELTA>(p, rank, t, pf);
    st = consume_pre<NB, OUT_NORM, G>(p, rank, cs, pf, prof ? tp : nullptr);
  }
  consume_post<NB, OUT_NORM, HAS_DELTA, G>(p, rank, t, cs, pf, st, prof ? tp : nullptr);
  if (prof && threadIdx.x == 0) {
    unsigned long long* dst = p.ts + (static_cast<long long>(t) * kC + rank) * kProbes;
#pragma unroll
    for (int k = 0; k < kProbes; ++k) dst[k] = tp[k];
  }
}

// ==========================================================================================
// FUSED: producer phase on all 112 CTAs, consumer phase on clusters c < M (CTA ranks 0..6).
// Consumer CTAs run the block-logit part of the consumer (consume_pre) while their weight loads
// are in flight / the predecessor finishes, then the GEMV, then poll + the rest of AttnRes.
// ==========================================================================================
template <int NT, bool GATE, int NB, bool OUT_NORM, bool HAS_DELTA>
__global__ void __launch_bounds__(kThreads * kFusedG, 1)
oproj_fused_kernel(const ProdParams pp, const ConsParams cp) {
  __shared__ ProdSmem<kFusedKS, kFusedRT, NT> ps;
  __shared__ ConsSmem<kFusedG> cs;
  extern __shared__ __align__(16) uint8_t dyn_smem[];  // PrefetchSmem [+ gated x] (static smem <= 48 KB)
  PrefetchSmem& pf = *reinterpret_cast<PrefetchSmem*>(dyn_smem);
  U4* xs = reinterpret_cast<U4*>(dyn_smem + sizeof(PrefetchSmem));
  const int rank = blockIdx.x;  // == %cluster_ctarank
  const int c = blockIdx.y;     // cluster index == token handled by the consumer phase
  const bool consumer = c < pp.M && rank < kC;
  const bool prof = cp.ts != nullptr && consumer;
  unsigned long long tp[kProbes];  // profiling only
  if (prof) {
    for (int k = 0; k < kProbes; ++k) tp[k] = 0;
    tp[kPStart] = gtimer();
    tp[kQStart] = tp[kPStart];
  }
  consume_setup<NB, OUT_NORM, kFusedG>(cs);  // (non-consumer CTAs: harmless, they exit)
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const bool g0 = consumer && threadIdx.x < kThreads;
  if (g0) consume_prefetch<NB, OUT_NORM, HAS_DELTA>(cp, rank, c, pf);
  const WRegs<kKBlocks / kFusedKS> W = produce_load_w<kFusedKS, kFusedRT>(pp, c * kFusedCS + rank);
  ConsState st;
  if (g0) st = consume_pre<NB, OUT_NORM, kFusedG>(cp, rank, cs, pf, prof ? tp : nullptr);
  produce_compute<kFusedKS, kFusedRT, NT, GATE>(pp, c * kFusedCS + rank, ps, W, xs, prof ? tp : nullptr);
  if (!consumer) return;
  consume_post<NB, OUT_NORM, HAS_DELTA, kFusedG>(cp, rank, c, cs, pf, st, prof ? tp : nullptr);
  if (prof && threadIdx.x == 0) {
    unsigned long long* dst = cp.ts + (static_cast<long long>(c) * kC + rank) * kProbes;
#pragma unroll
    for (int k = 0; k < kProbes; ++k) dst[k] = tp[k];
  }
}

// ------------------------------------------------------------------------------------------
// launchers
// ------------------------------------------------------------------------------------------
cudaLaunchAttribute pdl_attr(bool on) {
  cudaLaunchAttribute a;
  a.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  a.val.programmaticStreamSerializationAllowed = on ? 1 : 0;
  return a;
}
cudaLaunchAttribute cluster_attr(int size = kC) {
  cudaLaunchAttribute a;
  a.id = cudaLaunchAttributeClusterDimension;
  a.val.clusterDim.x = size;
  a.val.clusterDim.y = 1;
  a.val.clusterDim.z = 1;
  return a;
}

template <int KS, int RT, int NT, bool GATE>
void launch_produce(const ProdParams& prm, cudaStream_t stream, bool pdl) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3((kTiles + RT - 1) / RT);
  cfg.blockDim = dim3(KS * RT * 32);
  cfg.stream = stream;
  cudaLaunchAttribute attr[1] = {pdl_attr(pdl)};
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, oproj_produce_kernel<KS, RT, NT, GATE>, prm));
}

// variant = 10*KS + RT
template <int NT, bool GATE>
void dispatch_variant(int variant, const ProdParams& prm, cudaStream_t s, bool pdl) {
  switch (variant) {
    case 44: launch_produce<4, 4, NT, GATE>(prm, s, pdl); break;
    case 34: launch_produce<3, 4, NT, GATE>(prm, s, pdl); break;
    case 64: launch_produce<6, 4, NT, GATE>(prm, s, pdl); break;
    case 43: launch_produce<4, 3, NT, GATE>(prm, s, pdl); break;
    case 63: launch_produce<6, 3, NT, GATE>(prm, s, pdl); break;
    default: TORCH_CHECK(false, "unsupported producer variant ", variant, " (44, 34, 64, 43, 63)");
  }
}

template <int NB, bool OUT_NORM, bool HAS_DELTA, int G>
void launch_consume(const ConsParams& prm, int M, cudaStream_t stream) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(kC, M);
  cfg.blockDim = dim3(kThreads * G);
  cfg.stream = stream;
  cudaLaunchAttribute attr[2] = {cluster_attr(), pdl_attr(true)};
  cfg.attrs = attr;
  cfg.numAttrs = 2;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, oproj_consume_kernel<NB, OUT_NORM, HAS_DELTA, G>, prm));
}

template <int NB, bool OUT_NORM, bool HAS_DELTA>
void dispatch_groups(int groups, const ConsParams& prm, int M, cudaStream_t s) {
  switch (groups) {
    case 1: launch_consume<NB, OUT_NORM, HAS_DELTA, 1>(prm, M, s); break;
    case 2: launch_consume<NB, OUT_NORM, HAS_DELTA, 2>(prm, M, s); break;
    case 4: launch_consume<NB, OUT_NORM, HAS_DELTA, 4>(prm, M, s); break;
    default: TORCH_CHECK(false, "groups must be 1, 2 or 4");
  }
}

template <bool OUT_NORM, bool HAS_DELTA>
void dispatch_consume(int nb, int groups, const ConsParams& prm, int M, cudaStream_t s) {
  switch (nb) {
    case 0: dispatch_groups<0, OUT_NORM, HAS_DELTA>(groups, prm, M, s); break;
    case 1: dispatch_groups<1, OUT_NORM, HAS_DELTA>(groups, prm, M, s); break;
    case 2: dispatch_groups<2, OUT_NORM, HAS_DELTA>(groups, prm, M, s); break;
    case 3: dispatch_groups<3, OUT_NORM, HAS_DELTA>(groups, prm, M, s); break;
    case 4: dispatch_groups<4, OUT_NORM, HAS_DELTA>(groups, prm, M, s); break;
    case 5: dispatch_groups<5, OUT_NORM, HAS_DELTA>(groups, prm, M, s); break;
    case 6: dispatch_groups<6, OUT_NORM, HAS_DELTA>(groups, prm, M, s); break;
    case 7: dispatch_groups<7, OUT_NORM, HAS_DELTA>(groups, prm, M, s); break;
    case 8: dispatch_groups<8, OUT_NORM, HAS_DELTA>(groups, prm, M, s); break;
    default: TORCH_CHECK(false, "num_blocks must be in [0, 8]");
  }
}

template <int NT, bool GATE, int NB, bool HAS_DELTA>
void launch_fused(const ProdParams& pp, const ConsParams& cp, cudaStream_t stream, bool pdl) {
  auto* kernel = oproj_fused_kernel<NT, GATE, NB, true, HAS_DELTA>;
  constexpr int dyn = static_cast<int>(sizeof(PrefetchSmem)) + (GATE ? kXsBytes : 0);
  static const bool attr_set = [&] {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn));
    return true;
  }();
  (void)attr_set;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(kFusedCS, kFusedClusters);
  cfg.blockDim = dim3(kThreads * kFusedG);
  cfg.dynamicSmemBytes = dyn;
  cfg.stream = stream;
  cudaLaunchAttribute attr[2] = {cluster_attr(kFusedCS), pdl_attr(pdl)};
  cfg.attrs = attr;
  cfg.numAttrs = 2;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, pp, cp));
}

// Max number of the fused kernel's 7-CTA clusters that can be co-resident on an idle GPU.
// Consumer clusters (c < M) spin until every producer CTA has stored its fragment, so the fused
// kernel needs M < this number (the remaining producers run on SMs freed by the exiting
// non-consumer clusters); larger M must use produce + consume.
template <int NT>
int fused_max_active_clusters() {
  static const int value = [] {
    auto* kernel = oproj_fused_kernel<NT, true, 8, true, true>;  // gated: the larger smem footprint
    const int dyn = static_cast<int>(sizeof(PrefetchSmem)) + kXsBytes;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn));
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(kFusedCS, kFusedClusters);
    cfg.blockDim = dim3(kThreads * kFusedG);
    cfg.dynamicSmemBytes = dyn;
    cudaLaunchAttribute attr[1] = {cluster_attr(kFusedCS)};
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    int n = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&n, kernel, &cfg));
    return n;
  }();
  return value;
}

template <int NT, bool GATE, bool HAS_DELTA>
void dispatch_fused_nb(int nb, const ProdParams& pp, const ConsParams& cp, cudaStream_t s, bool pdl) {
  switch (nb) {
    case 0: launch_fused<NT, GATE, 0, HAS_DELTA>(pp, cp, s, pdl); break;
    case 1: launch_fused<NT, GATE, 1, HAS_DELTA>(pp, cp, s, pdl); break;
    case 2: launch_fused<NT, GATE, 2, HAS_DELTA>(pp, cp, s, pdl); break;
    case 3: launch_fused<NT, GATE, 3, HAS_DELTA>(pp, cp, s, pdl); break;
    case 4: launch_fused<NT, GATE, 4, HAS_DELTA>(pp, cp, s, pdl); break;
    case 5: launch_fused<NT, GATE, 5, HAS_DELTA>(pp, cp, s, pdl); break;
    case 6: launch_fused<NT, GATE, 6, HAS_DELTA>(pp, cp, s, pdl); break;
    case 7: launch_fused<NT, GATE, 7, HAS_DELTA>(pp, cp, s, pdl); break;
    case 8: launch_fused<NT, GATE, 8, HAS_DELTA>(pp, cp, s, pdl); break;
    default: TORCH_CHECK(false, "num_blocks must be in [0, 8]");
  }
}

// ------------------------------------------------------------------------------------------
// argument checks / parameter packing
// ------------------------------------------------------------------------------------------
void check_bf16_rows(const torch::Tensor& x, int64_t width, const char* name) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16, name, " must be CUDA bf16");
  TORCH_CHECK(x.size(-1) == width && x.stride(-1) == 1, name, " must be [..., ", width, "] with unit stride");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, name, " must be 16-byte aligned");
}

void check_mailbox(const torch::Tensor& mailbox, int64_t buf) {
  TORCH_CHECK(mailbox.is_cuda() && mailbox.scalar_type() == at::kBFloat16 && mailbox.is_contiguous() &&
                  mailbox.dim() == 4 && mailbox.size(3) == kHidden,
              "mailbox must be contiguous CUDA bf16 [nbuf, world, max_m, 7168]");
  TORCH_CHECK(mailbox.size(1) >= 1 && mailbox.size(1) <= kMaxWorld, "world must be in [1, 16]");
  TORCH_CHECK(buf >= 0 && buf < mailbox.size(0), "buffer index out of range");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(mailbox.data_ptr()) % 16 == 0);
}

__nv_bfloat16* bf(const torch::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr()); }

ProdParams make_prod(const torch::Tensor& x, const torch::Tensor& weight, const std::optional<torch::Tensor>& gate,
                     const torch::Tensor& mailbox, int64_t mc_ptr, int64_t buf, int64_t slot, int64_t mode,
                     const std::optional<torch::Tensor>& y,
                     const std::optional<torch::Tensor>& next_weight = std::nullopt) {
  check_mailbox(mailbox, buf);
  TORCH_CHECK(x.dim() == 2, "x must be [M, 768]");
  check_bf16_rows(x, kK, "x");
  const int M = x.size(0);
  TORCH_CHECK(x.stride(0) % 8 == 0, "x row stride must be a multiple of 8");
  TORCH_CHECK(weight.is_contiguous() && weight.dim() == 2 && weight.size(0) == kHidden, "weight must be [7168, 768]");
  check_bf16_rows(weight, kK, "weight");
  const int world = mailbox.size(1), max_m = mailbox.size(2);
  TORCH_CHECK(M >= 1 && M <= kMaxM && M <= max_m, "M must be in [1, min(16, max_m)]");
  TORCH_CHECK(slot >= 0 && slot < world, "slot out of range");
  TORCH_CHECK(mode >= 0 && mode <= 3, "bad mode");
  TORCH_CHECK(mode != kModeMulticast || mc_ptr != 0, "multicast mode needs a multicast pointer");
  if (gate) {
    TORCH_CHECK(gate->dim() == 2 && gate->size(0) == M && gate->stride(0) % 8 == 0);
    check_bf16_rows(*gate, kK, "gate");
  }
  if (mode == kModePlain) {
    TORCH_CHECK(y.has_value() && y->dim() == 2 && y->size(0) == M && y->stride(0) % 8 == 0);
    check_bf16_rows(*y, kHidden, "y");
  }
  if (next_weight) {
    TORCH_CHECK(next_weight->is_contiguous() && next_weight->sizes() == weight.sizes() &&
                    next_weight->scalar_type() == at::kBFloat16 && next_weight->is_cuda(),
                "next_weight must be a contiguous bf16 [7168, 768] tensor");
  }
  ProdParams prm{};
  prm.x = bf(x);
  prm.gate = gate ? bf(*gate) : nullptr;
  prm.w = bf(weight);
  prm.w_next = next_weight ? bf(*next_weight) : nullptr;
  const long long buf_elems = static_cast<long long>(world) * max_m * kHidden;
  prm.mbox = bf(mailbox) + buf * buf_elems;
  prm.mc = mc_ptr ? static_cast<unsigned long long>(mc_ptr) + static_cast<unsigned long long>(buf * buf_elems) * 2ull : 0ull;
  prm.y = y ? bf(*y) : nullptr;
  prm.x_sm = x.stride(0);
  prm.g_sm = gate ? gate->stride(0) : 0;
  prm.y_sm = y ? y->stride(0) : 0;
  prm.M = M;
  prm.slot = static_cast<int>(slot);
  prm.world = world;
  prm.max_m = max_m;
  prm.mode = static_cast<int>(mode);
  return prm;
}

ConsParams make_cons(const torch::Tensor& mailbox, int64_t buf, const torch::Tensor& prefix,
                     const torch::Tensor& blocks, const torch::Tensor& norm_weight,
                     const torch::Tensor& qk_weight, const std::optional<torch::Tensor>& output_norm_weight,
                     const torch::Tensor& out, const std::optional<torch::Tensor>& attn_out,
                     int64_t num_blocks, int64_t block_write_idx, double eps, double output_norm_eps,
                     bool wait_prior, int64_t poll,
                     const std::optional<torch::Tensor>& xmoe_buf = std::nullopt,
                     const std::optional<torch::Tensor>& epoch = std::nullopt, int64_t par = 0) {
  check_mailbox(mailbox, buf);
  TORCH_CHECK(prefix.dim() == 2 && out.dim() == 2 && blocks.dim() == 3);
  const int M = prefix.size(0);
  const int world = mailbox.size(1), max_m = mailbox.size(2);
  TORCH_CHECK(M >= 1 && M <= max_m, "M must be in [1, max_m]");
  TORCH_CHECK(out.size(0) == M && blocks.size(0) >= M, "row counts disagree");
  check_bf16_rows(prefix, kHidden, "prefix");
  check_bf16_rows(out, kHidden, "out");
  check_bf16_rows(blocks, kHidden, "blocks");
  check_bf16_rows(norm_weight, kHidden, "norm_weight");
  check_bf16_rows(qk_weight, kHidden, "qk_weight");
  TORCH_CHECK(norm_weight.is_contiguous() && qk_weight.is_contiguous());
  TORCH_CHECK(blocks.stride(0) % 8 == 0 && blocks.stride(1) % 8 == 0 && prefix.stride(0) % 8 == 0 &&
              out.stride(0) % 8 == 0, "row strides must be multiples of 8 elements");
  TORCH_CHECK(num_blocks >= 0 && num_blocks <= kMaxNB && num_blocks <= blocks.size(1));
  TORCH_CHECK(block_write_idx == -1 || (block_write_idx >= num_blocks && block_write_idx < blocks.size(1)),
              "block_write_idx must be -1 or in [num_blocks, blocks.size(1))");
  TORCH_CHECK(poll == 0 || poll == 1, "poll must be 0 or 1");
  if (output_norm_weight) {
    check_bf16_rows(*output_norm_weight, kHidden, "output_norm_weight");
    TORCH_CHECK(output_norm_weight->is_contiguous());
  }
  if (attn_out) {
    check_bf16_rows(*attn_out, kHidden, "attn_out");
    TORCH_CHECK(attn_out->size(0) == M && attn_out->stride(0) % 8 == 0);
  }
  ConsParams prm{};
  const long long slot_stride = static_cast<long long>(max_m) * kHidden;
  prm.mbox = bf(mailbox) + buf * world * slot_stride;
  prm.prefix = bf(prefix);
  prm.blocks = bf(blocks);
  prm.norm_w = bf(norm_weight);
  prm.qk_w = bf(qk_weight);
  prm.out_w = output_norm_weight ? bf(*output_norm_weight) : nullptr;
  prm.out = bf(out);
  prm.attn_out = attn_out ? bf(*attn_out) : nullptr;
  prm.slot_stride = slot_stride;
  prm.prefix_sm = prefix.stride(0);
  prm.out_sm = out.stride(0);
  prm.attn_sm = attn_out ? attn_out->stride(0) : 0;
  prm.blocks_sm = blocks.stride(0);
  prm.blocks_sr = blocks.stride(1);
  prm.M = M;
  prm.world = world;
  prm.write_idx = static_cast<int>(block_write_idx);
  prm.wait_prior = wait_prior ? 1 : 0;
  prm.poll = static_cast<int>(poll);
  prm.eps = static_cast<float>(eps);
  prm.out_eps = static_cast<float>(output_norm_eps);
  if (xmoe_buf) {
    // [2 parities][reps][rows >= M][7168] bf16 (Lamport-armed; the MoE front kernel re-arms it)
    TORCH_CHECK(epoch.has_value() && epoch->scalar_type() == at::kInt && epoch->numel() >= 192,
                "xmoe_buf needs the int32 epoch (hint counter) tensor");
    TORCH_CHECK(xmoe_buf->dim() == 4 && xmoe_buf->size(0) == 2 && xmoe_buf->size(2) >= M && xmoe_buf->size(3) == kHidden &&
                    xmoe_buf->scalar_type() == at::kBFloat16 && xmoe_buf->is_contiguous(),
                "xmoe_buf: bf16 [2, reps, >= M, 7168]");
    TORCH_CHECK(par == 0 || par == 1);
    const long long rows = xmoe_buf->size(2);
    prm.xmoe = bf(*xmoe_buf) + par * xmoe_buf->size(1) * rows * kHidden;
    prm.xmoe_rs = rows * kHidden;
    prm.xmoe_reps = static_cast<int>(xmoe_buf->size(1));
    prm.xmoe_ctr = reinterpret_cast<unsigned*>(epoch->data_ptr()) + par * 32;  // kind 0 (x_moe), parity par
  }
  return prm;
}

unsigned long long* ts_ptr(const std::optional<torch::Tensor>& ts, int64_t need) {
  if (!ts) return nullptr;
  TORCH_CHECK(ts->is_cuda() && ts->scalar_type() == at::kLong && ts->is_contiguous() && ts->numel() >= need,
              "timestamps must be a contiguous int64 CUDA tensor with >= ", need, " elements");
  return reinterpret_cast<unsigned long long*>(ts->data_ptr());
}

}  // namespace

// ------------------------------------------------------------------------------------------
// ops
// ------------------------------------------------------------------------------------------
void oproj_produce(torch::Tensor x, torch::Tensor weight, std::optional<torch::Tensor> gate,
                   torch::Tensor mailbox, int64_t mc_ptr, int64_t buf, int64_t slot, int64_t mode,
                   std::optional<torch::Tensor> y, int64_t variant, bool pdl,
                   std::optional<torch::Tensor> timestamps, std::optional<torch::Tensor> next_weight) {
  ProdParams prm = make_prod(x, weight, gate, mailbox, mc_ptr, buf, slot, mode, y, next_weight);
  const int rt = static_cast<int>(variant % 10);
  prm.ts = ts_ptr(timestamps, static_cast<int64_t>((kTiles + rt - 1) / (rt > 0 ? rt : 1)) * kProbes);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const int v = static_cast<int>(variant);
  if (prm.M <= 8) {
    if (gate) dispatch_variant<1, true>(v, prm, stream, pdl);
    else dispatch_variant<1, false>(v, prm, stream, pdl);
  } else {
    if (gate) dispatch_variant<2, true>(v, prm, stream, pdl);
    else dispatch_variant<2, false>(v, prm, stream, pdl);
  }
}

void oproj_consume_debug(torch::Tensor mailbox, int64_t buf, torch::Tensor prefix, bool has_delta,
                         torch::Tensor blocks, torch::Tensor norm_weight, torch::Tensor qk_weight,
                         std::optional<torch::Tensor> output_norm_weight, torch::Tensor out,
                         std::optional<torch::Tensor> attn_out, int64_t num_blocks, int64_t block_write_idx,
                         double eps, double output_norm_eps, bool wait_prior,
                         std::optional<torch::Tensor> timestamps, int64_t poll, int64_t groups,
                         std::optional<torch::Tensor> xmoe_buf, std::optional<torch::Tensor> epoch, int64_t par) {
  ConsParams prm = make_cons(mailbox, buf, prefix, blocks, norm_weight, qk_weight, output_norm_weight, out,
                             attn_out, num_blocks, block_write_idx, eps, output_norm_eps, wait_prior, poll,
                             xmoe_buf, epoch, par);
  prm.ts = ts_ptr(timestamps, static_cast<int64_t>(prm.M) * kC * kProbes);
  const int gr = static_cast<int>(groups);
  TORCH_CHECK(gr == 1 || gr == 2 || gr == 4, "groups must be 1, 2 or 4");
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const int nb = static_cast<int>(num_blocks);
  if (output_norm_weight) {
    if (has_delta) dispatch_consume<true, true>(nb, gr, prm, prm.M, stream);
    else dispatch_consume<true, false>(nb, gr, prm, prm.M, stream);
  } else {
    if (has_delta) dispatch_consume<false, true>(nb, gr, prm, prm.M, stream);
    else dispatch_consume<false, false>(nb, gr, prm, prm.M, stream);
  }
}

void oproj_consume(torch::Tensor mailbox, int64_t buf, torch::Tensor prefix, bool has_delta,
                   torch::Tensor blocks, torch::Tensor norm_weight, torch::Tensor qk_weight,
                   std::optional<torch::Tensor> output_norm_weight, torch::Tensor out,
                   std::optional<torch::Tensor> attn_out, int64_t num_blocks, int64_t block_write_idx,
                   double eps, double output_norm_eps, std::optional<torch::Tensor> xmoe_buf,
                   std::optional<torch::Tensor> epoch, int64_t par) {
  oproj_consume_debug(mailbox, buf, prefix, has_delta, blocks, norm_weight, qk_weight, output_norm_weight, out,
                      attn_out, num_blocks, block_write_idx, eps, output_norm_eps, false, std::nullopt, 0, 4,
                      xmoe_buf, epoch, par);
}

void oproj_fused_debug(torch::Tensor x, torch::Tensor weight, std::optional<torch::Tensor> gate,
                       torch::Tensor mailbox, int64_t mc_ptr, int64_t buf, int64_t slot, int64_t mode,
                       torch::Tensor prefix, bool has_delta, torch::Tensor blocks, torch::Tensor norm_weight,
                       torch::Tensor qk_weight, torch::Tensor output_norm_weight, torch::Tensor out,
                       std::optional<torch::Tensor> attn_out, int64_t num_blocks, double eps,
                       double output_norm_eps, bool pdl, int64_t poll, std::optional<torch::Tensor> timestamps,
                       std::optional<torch::Tensor> next_weight, std::optional<torch::Tensor> xmoe_buf,
                       std::optional<torch::Tensor> epoch, int64_t par) {
  TORCH_CHECK(mode != kModePlain, "the fused op needs a mailbox mode");
  ProdParams pp = make_prod(x, weight, gate, mailbox, mc_ptr, buf, slot, mode, std::nullopt, next_weight);
  ConsParams cp = make_cons(mailbox, buf, prefix, blocks, norm_weight, qk_weight, output_norm_weight, out,
                            attn_out, num_blocks, -1, eps, output_norm_eps, false, poll, xmoe_buf, epoch, par);
  TORCH_CHECK(cp.M == pp.M, "x and prefix row counts disagree");
  const int max_clusters = pp.M <= 8 ? fused_max_active_clusters<1>() : fused_max_active_clusters<2>();
  TORCH_CHECK(pp.M < max_clusters && pp.M <= kFusedClusters, "fused op needs M < ", max_clusters,
              " and M <= ", kFusedClusters, " (co-resident clusters on this GPU); use produce + consume");
  cp.ts = ts_ptr(timestamps, static_cast<int64_t>(pp.M) * kC * kProbes);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const int nb = static_cast<int>(num_blocks);
  if (pp.M <= 8) {
    if (gate) {
      if (has_delta) dispatch_fused_nb<1, true, true>(nb, pp, cp, stream, pdl);
      else dispatch_fused_nb<1, true, false>(nb, pp, cp, stream, pdl);
    } else {
      if (has_delta) dispatch_fused_nb<1, false, true>(nb, pp, cp, stream, pdl);
      else dispatch_fused_nb<1, false, false>(nb, pp, cp, stream, pdl);
    }
  } else {
    if (gate) {
      if (has_delta) dispatch_fused_nb<2, true, true>(nb, pp, cp, stream, pdl);
      else dispatch_fused_nb<2, true, false>(nb, pp, cp, stream, pdl);
    } else {
      if (has_delta) dispatch_fused_nb<2, false, true>(nb, pp, cp, stream, pdl);
      else dispatch_fused_nb<2, false, false>(nb, pp, cp, stream, pdl);
    }
  }
}

void oproj_fused(torch::Tensor x, torch::Tensor weight, std::optional<torch::Tensor> gate, torch::Tensor mailbox,
                 int64_t mc_ptr, int64_t buf, int64_t slot, int64_t mode, torch::Tensor prefix, bool has_delta,
                 torch::Tensor blocks, torch::Tensor norm_weight, torch::Tensor qk_weight,
                 torch::Tensor output_norm_weight, torch::Tensor out, int64_t num_blocks, double eps,
                 double output_norm_eps, std::optional<torch::Tensor> next_weight,
                 std::optional<torch::Tensor> xmoe_buf, std::optional<torch::Tensor> epoch, int64_t par) {
  oproj_fused_debug(x, weight, gate, mailbox, mc_ptr, buf, slot, mode, prefix, has_delta, blocks, norm_weight,
                    qk_weight, output_norm_weight, out, std::nullopt, num_blocks, eps, output_norm_eps, true, 0,
                    std::nullopt, next_weight, xmoe_buf, epoch, par);
}


// ------------------------------------------------------------------------------------------
// test/benchmark helpers (not used by the integration)
// ------------------------------------------------------------------------------------------
namespace {
// Stand-in for the o_proj predecessor (KDA decode / MLA gate multiply): griddepcontrol.wait, spin
// `spin_ns`, trigger dependents `lead_ns` before the end (lead >= spin: right after the wait;
// 0: after the last store), then write x = src.
__global__ void bench_pred_kernel(uint4* __restrict__ x, const uint4* __restrict__ src, int n16, int spin_ns,
                                  int lead_ns, unsigned long long* ts) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const unsigned long long t0 = gtimer();
  bool trig = false;
  if (lead_ns >= spin_ns) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    trig = true;
  }
  const unsigned long long t_trig = static_cast<unsigned long long>(spin_ns - lead_ns);
  while (true) {
    const unsigned long long dt = gtimer() - t0;
    if (!trig && dt >= t_trig) {
      asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
      trig = true;
    }
    if (dt >= static_cast<unsigned long long>(spin_ns)) break;
  }
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += gridDim.x * blockDim.x) x[i] = src[i];
  if (ts != nullptr) {
    __syncthreads();
    if (threadIdx.x == 0) {
      ts[2 * blockIdx.x] = t0;
      ts[2 * blockIdx.x + 1] = gtimer();
    }
  }
  if (!trig) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

__global__ void __launch_bounds__(256) bench_evict_kernel(const uint4* __restrict__ buf, long long n16, float* sink) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  uint32_t acc = 0;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n16; i += (long long)gridDim.x * blockDim.x) {
    uint4 u = __ldcs(buf + i);
    acc ^= u.x ^ u.y ^ u.z ^ u.w;
  }
  if (acc == 0x9e3779b9u) sink[0] = 1.f;
}

template <typename Kern, typename... Args>
void launch_simple(bool pdl, Kern kernel, dim3 grid, dim3 block, Args... args) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1] = {pdl_attr(pdl)};
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, args...));
}
}  // namespace

void oproj_bench_pred(torch::Tensor x, torch::Tensor src, int64_t ctas, int64_t threads, int64_t spin_ns,
                      int64_t lead_ns, bool pdl, std::optional<torch::Tensor> ts) {
  TORCH_CHECK(x.is_contiguous() && src.is_contiguous() && x.nbytes() == src.nbytes() && x.nbytes() % 16 == 0);
  launch_simple(pdl, bench_pred_kernel, dim3(ctas), dim3(threads), reinterpret_cast<uint4*>(x.data_ptr()),
                reinterpret_cast<const uint4*>(src.data_ptr()), static_cast<int>(x.nbytes() / 16),
                static_cast<int>(spin_ns), static_cast<int>(lead_ns),
                ts ? reinterpret_cast<unsigned long long*>(ts->data_ptr()) : nullptr);
}

void oproj_bench_l2_evict(torch::Tensor buf, int64_t offset_bytes, int64_t nbytes, torch::Tensor sink) {
  TORCH_CHECK(offset_bytes % 16 == 0 && nbytes % 16 == 0 && offset_bytes + nbytes <= static_cast<int64_t>(buf.nbytes()));
  const auto* p = reinterpret_cast<const uint4*>(static_cast<const char*>(buf.data_ptr()) + offset_bytes);
  launch_simple(true, bench_evict_kernel, dim3(148 * 4), dim3(256), p, static_cast<long long>(nbytes / 16),
                sink.data_ptr<float>());
}

int64_t oproj_fused_max_m() {
  const int m1 = std::min(fused_max_active_clusters<1>() - 1, std::min(8, kFusedClusters));
  if (m1 < 8) return m1;
  return std::min(fused_max_active_clusters<2>() - 1, std::min(kMaxM, kFusedClusters));
}
int64_t oproj_fused_max_clusters(int64_t nt) {
  return nt <= 1 ? fused_max_active_clusters<1>() : fused_max_active_clusters<2>();
}

TORCH_LIBRARY(k3oprojf, m) {
  m.def("fused_max_m() -> int");
  m.def("bench_pred(Tensor(a!) x, Tensor src, int ctas, int threads, int spin_ns, int lead_ns, bool pdl, "
        "Tensor(b!)? ts=None) -> ()");
  m.def("bench_l2_evict(Tensor buf, int offset_bytes, int nbytes, Tensor(a!) sink) -> ()");
  m.def("fused_max_clusters(int nt) -> int");
  m.def("fused(Tensor x, Tensor weight, Tensor? gate, Tensor(a!) mailbox, int mc_ptr, int buf, int slot, "
        "int mode, Tensor(b!) prefix, bool has_delta, Tensor blocks, Tensor norm_weight, Tensor qk_weight, "
        "Tensor output_norm_weight, Tensor(c!) out, int num_blocks, float eps, float output_norm_eps, "
        "Tensor? next_weight=None, Tensor(x!)? xmoe_buf=None, Tensor(y!)? epoch=None, int par=0) -> ()");
  m.def("fused_debug(Tensor x, Tensor weight, Tensor? gate, Tensor(a!) mailbox, int mc_ptr, int buf, int slot, "
        "int mode, Tensor(b!) prefix, bool has_delta, Tensor blocks, Tensor norm_weight, Tensor qk_weight, "
        "Tensor output_norm_weight, Tensor(c!) out, Tensor(d!)? attn_out, int num_blocks, float eps, "
        "float output_norm_eps, bool pdl=True, int poll=0, Tensor(e!)? timestamps=None, "
        "Tensor? next_weight=None, Tensor(x!)? xmoe_buf=None, Tensor(y!)? epoch=None, int par=0) -> ()");
  m.def("produce(Tensor x, Tensor weight, Tensor? gate, Tensor(a!) mailbox, int mc_ptr, int buf, "
        "int slot, int mode, Tensor(b!)? y=None, int variant=44, bool pdl=True, "
        "Tensor(c!)? timestamps=None, Tensor? next_weight=None) -> ()");
  m.def("consume(Tensor(a!) mailbox, int buf, Tensor(b!) prefix, bool has_delta, Tensor(c!) blocks, "
        "Tensor norm_weight, Tensor qk_weight, Tensor? output_norm_weight, Tensor(d!) out, "
        "Tensor(e!)? attn_out, int num_blocks, int block_write_idx, float eps, "
        "float output_norm_eps, Tensor(x!)? xmoe_buf=None, Tensor(y!)? epoch=None, int par=0) -> ()");
  m.def("consume_debug(Tensor(a!) mailbox, int buf, Tensor(b!) prefix, bool has_delta, Tensor(c!) blocks, "
        "Tensor norm_weight, Tensor qk_weight, Tensor? output_norm_weight, Tensor(d!) out, "
        "Tensor(e!)? attn_out, int num_blocks, int block_write_idx, float eps, "
        "float output_norm_eps, bool wait_prior, Tensor(f!)? timestamps, int poll=0, int groups=4, "
        "Tensor(x!)? xmoe_buf=None, Tensor(y!)? epoch=None, int par=0) -> ()");
}
TORCH_LIBRARY_IMPL(k3oprojf, CompositeExplicitAutograd, m) {
  m.impl("fused_max_m", &oproj_fused_max_m);
  m.impl("fused_max_clusters", &oproj_fused_max_clusters);
}
TORCH_LIBRARY_IMPL(k3oprojf, CUDA, m) {
  m.impl("bench_pred", &oproj_bench_pred);
  m.impl("bench_l2_evict", &oproj_bench_l2_evict);
  m.impl("fused", &oproj_fused);
  m.impl("fused_debug", &oproj_fused_debug);
  m.impl("produce", &oproj_produce);
  m.impl("consume", &oproj_consume);
  m.impl("consume_debug", &oproj_consume_debug);
}
