// Kimi K3 fused KDA decode at 6 heads/rank (TP16), value-split across a
// thread-block cluster.  Drop-in replacement for torch.ops.k3kda.fused_kda_decode
// (gpu-opt/k3opt/csrc/kda_decode6.cu): same schema, same math, same in-place
// state / conv_state updates, bit-identical results.  Registered as
// torch.ops.k3kdas.fused_kda_decode (+ fused_kda_decode_split for tuning).
//
// Why the old kernel is slow: one 256-thread CTA per (token, head) walks all 128
// value rows of the 64 KB fp32 state in 4 cp.async chunks, with two 5-level warp
// reductions per row pair, and only starts loading state after
// griddepcontrol.wait.  At M=1 that is 6 CTAs on 152 SMs, all latency.
//
// Layout here: grid = (C, 6 heads, B tokens), cluster = (C, 1, 1), NT threads.
// CTA c of a cluster owns value rows [c*R, c*R+R) of its head's state, R = 128/C.
// Each row is handled by LPR lanes; lane l holds the old kernel's lanes l + LPR*j
// (k = 4(l + LPR*j) .. +3), so the old reduction tree is kept exactly: the high
// lane bits are summed locally in the old pairing order, the low ones by shuffles
// (LPR 8: 3 shuffle levels instead of 5).
//
//   1. Before griddepcontrol.wait (nothing here depends on the predecessor):
//      leader mbarrier init + cluster "started" arrive, slot index, this
//      thread's state rows straight into registers, conv taps, conv weights,
//      dt_bias, A_log, norm weight.  Loads only -- no arithmetic on loaded values,
//      so a slow (HBM) load can never delay the PDL release.
//   2. After the wait: x (q/k/v), raw gate, beta, output gate (L2-hot); trigger
//      dependents; conv sums in the old FMA order (bias, 3 taps, x), SiLU,
//      sigmoids, decay.  The IEEE reciprocal inside each sigmoid is issued
//      straight-line on the compiler's own fast path (rcp.approx + one Newton FMA,
//      correctly rounded in that exponent range -- checked exhaustively over all
//      fp32 inputs) with one exact fallback, instead of six serialised BSSY/CALL
//      regions.  Every CTA recomputes the per-head q/k work redundantly (cheaper
//      than another cross-CTA round trip).
//   3. One __syncthreads (+ the cluster "started" wait): per-warp q^2 / k^2 sums
//      and raw q/k/decay go through smem; each lane normalises its own channels.
//   4. Delta-rule update of the thread's rows in registers, o = S_new q.
//   5. o -> leader CTA with st.async + mbarrier complete_tx (a release/acquire
//      cluster barrier here compiles to MEMBAR.ALL.GPU and cost ~1000 cycles),
//      then the state rows and the v conv-state taps are stored; non-leader CTAs
//      exit.
//   6. Leader (CTA 0): waits on its mbarrier, gated RMSNorm over the head's 128 o
//      values (old reduction tree), bf16 output, and only then the q/k conv-state
//      shift: once all o values have arrived every CTA of the cluster has consumed
//      the old taps, so the in-place update is race-free.
//
// All arithmetic mirrors kda_decode6.cu expression by expression (FMA contraction
// included), so state, conv state and output are bit-identical (test_kda.py).
//
// PDL caveat (same as any pre-wait prefetch): state, conv_state and state_indices
// are read before griddepcontrol.wait.  They are written by the previous decode
// step's KDA of this layer (or by the scheduler's metadata copies), which completes
// long before; this is only unsafe if a kernel that writes the same slots is our
// *immediate* PDL predecessor (or a chain of predecessors that all trigger their
// dependents before their own griddepcontrol.wait).  Set K3KDAS_NO_PREFETCH=1 to
// move all loads after the wait.

#include <cstdint>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <optional>

#include "torch_utils.h"

namespace {

constexpr int kD = 128;              // head dim, K == V
constexpr int kH = 6;                // heads per rank at TP16
constexpr int kDim = kH * kD;        // 768 channels per q/k/v section

struct KdaSplitParams {
  const __nv_bfloat16* x;       // [B, >= 3*kDim], row stride x_row
  const float* w;               // [3, 4, kDim]
  const float* bias;            // [3*kDim] or nullptr
  __nv_bfloat16* cs;            // conv state base
  const float* a_log;           // [kH]
  const __nv_bfloat16* g;       // [B, kH, kD] contiguous
  const float* dt_bias;         // [kDim]
  const __nv_bfloat16* beta;    // [B, kH], row stride beta_row
  const __nv_bfloat16* gate;    // [B, kH, kD], row stride gate_row (may be null)
  const float* norm_w;          // [kD] (may be null)
  const int* idx;               // [B]
  float* state;                 // [slots, kH, kD, kD] with slot stride state_slot
  __nv_bfloat16* out;           // [B, kH, kD] contiguous
  int64_t x_row;
  int64_t beta_row;
  int64_t gate_row;
  int64_t cs_slot;
  int64_t state_slot;
  float lower_bound;
  float scale;
  float eps;
  unsigned long long* ts;       // K3KDAS_TIMING builds only: [grid CTAs, 32] stamps
};

#ifdef K3KDAS_TIMING
__device__ __forceinline__ unsigned long long globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// ts[cta][0..15] = %globaltimer, ts[cta][16..31] = clock64 at stamp k
#define K3KDAS_TS(k)                                                                   \
  do {                                                                                 \
    if (threadIdx.x == 0 && p.ts != nullptr) {                                         \
      unsigned long long* t_ =                                                         \
          p.ts + ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * 32; \
      t_[(k)] = globaltimer();                                                         \
      t_[16 + (k)] = static_cast<unsigned long long>(clock64());                       \
    }                                                                                  \
  } while (0)
#define K3KDAS_NS k3kdas_t
#else
#define K3KDAS_TS(k) \
  do {               \
  } while (0)
#define K3KDAS_NS k3kdas
#endif

// ---- math helpers -----------------------------------------------------------
// softplus_fast is identical to kda_decode6.cu (only used without lower_bound).
__device__ __forceinline__ float softplus_fast(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}

// kda_decode6.cu: sigmoid_fast(x) = 1.0f / (1.0f + __expf(-x)); silu = x * sigmoid.
// The IEEE division 1.0f / y compiles to: if ((bits(y) + 0x01800000) & 0x7f800000) >
// 0x01ffffff (y normal, |y| < 2^126) { r = rcp.approx(y); r = fma(r, -fma(y, r, -1), r) }
// else call the slow path.  Both branches return the correctly rounded 1/y, which is
// unique, so issuing the fast path unconditionally and redoing the plain division only
// when some operand is outside the fast range gives bit-identical results while
// letting the compiler interleave the independent sigmoid chains.
__device__ __forceinline__ float sig_den(float x) { return 1.0f + __expf(-x); }
__device__ __forceinline__ bool rcp_fast_ok(float y) {
  return ((__float_as_uint(y) + 0x01800000u) & 0x7f800000u) > 0x01ffffffu;
}
__device__ __forceinline__ float rcp_fast(float y) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(y));
  const float e = fmaf(y, r, -1.0f);
  return fmaf(r, -e, r);
}

// ---- PDL / cluster / mbarrier primitives -------------------------------------
__device__ __forceinline__ void pdl_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
__device__ __forceinline__ void pdl_trigger() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
__device__ __forceinline__ void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_wait() {
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}
__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ uint32_t map_rank0(uint32_t a) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(r) : "r"(a));
  return r;
}
__device__ __forceinline__ void mbar_init_expect(uint32_t bar, uint32_t tx_bytes) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar) : "memory");
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar),
               "r"(tx_bytes)
               : "memory");
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
// 4-byte remote store into CTA 0's smem that completes 4 tx-bytes on its mbarrier.
__device__ __forceinline__ void st_async_rank0(uint32_t remote_addr, float v,
                                               uint32_t remote_bar) {
  asm volatile(
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(
          remote_addr),
      "r"(__float_as_uint(v)), "r"(remote_bar)
      : "memory");
}
__device__ __forceinline__ void mbar_wait_parity0(uint32_t bar) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], 0;\n"
      "@!p bra WAIT_%=;\n"
      "}\n" ::"r"(bar)
      : "memory");
}

// Position of value row v (= 32g + l + 8j, l < 8, j < 4) in the leader's s_o, so that
// lane l of the leader's RMSNorm reduction reads its 16 values as 4 float4.
__device__ __forceinline__ int opos(int v) {
  const int g = v >> 5, L = v & 31;
  return ((L & 7) << 4) | (g << 2) | (L >> 3);
}

// Row dot product with the old kernel's reduction tree.  The old kernel has lane L
// (0..31) hold k = 4L..4L+3, forms ((a0 + a1) + a2) + a3 per lane (FMA-contracted) and
// butterflies xor 16, 8, 4, 2, 1.  Here a row is spread over LPR lanes; lane l holds the
// "old lanes" L = l + LPR*j, j < 32/LPR.  The high bits of L (j) are reduced locally in
// the same pairing order (xor 16 first), the low bits by shuffles.
template <int LPR>
__device__ __forceinline__ float row_dot(const float (&h)[32 / LPR][4],
                                         const float (&x)[32 / LPR][4]) {
  constexpr int NJ = 32 / LPR;
  float part[NJ];
#pragma unroll
  for (int j = 0; j < NJ; ++j) {
    part[j] = h[j][0] * x[j][0] + h[j][1] * x[j][1] + h[j][2] * x[j][2] + h[j][3] * x[j][3];
  }
#pragma unroll
  for (int m = NJ / 2; m >= 1; m >>= 1) {
#pragma unroll
    for (int j = 0; j < m; ++j) part[j] = part[j] + part[j + m];
  }
  return part[0];
}

template <int LPR, int N>
__device__ __forceinline__ void shfl_tree(float (&v)[N]) {
#pragma unroll
  for (int off = LPR / 2; off > 0; off >>= 1) {
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] += __shfl_xor_sync(0xffffffffu, v[i], off);
  }
}

// NT  = threads per CTA.  128: every thread owns q and k channel `tid`.  256: threads
//       [0,128) own q channel tid (+ its decay), threads [128,256) own k channel
//       tid-128, the v channels and (leader) the output rows; both halves run the same
//       straight-line code on selected inputs.
// C   = CTAs per (token, head) (cluster size); each owns R = 128 / C value rows.
// LPR = lanes per state row in the update phase (8: 3 shuffle levels, 16: 4, 32: 5).
template <int NT, int C, int LPR, bool kSD, bool kLowerBound, bool kOnorm, bool kPrefetch>
__global__ void __launch_bounds__(NT)
    kda_split_kernel(const KdaSplitParams p) {
  constexpr bool kHalves = NT == 2 * kD;
  static_assert(NT == kD || kHalves, "NT must be 128 or 256");
  constexpr int R = kD / C;                  // value rows per CTA
  constexpr int NJ = 32 / LPR;               // float4 k-blocks per lane per row
  constexpr int GROUPS = NT / LPR;           // rows per pass
  constexpr int PASSES = (R + GROUPS - 1) / GROUPS;
  static_assert(R % GROUPS == 0 || GROUPS % R == 0, "bad split");
  static_assert(R <= 32, "v channels live in one warp");
  // conv_state element (slot, channel, tap) = slot*cs_slot + channel*kChS + tap*kTapS
  constexpr int kChS = kSD ? 1 : 3;
  constexpr int kTapS = kSD ? 3 * kDim : 1;
  constexpr int kHi = kHalves ? kD : 0;      // first thread of the k / v / output half

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int c = blockIdx.x;        // == rank in the (C,1,1) cluster
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int v_base = c * R;
  const bool leader = c == 0;
  const int ch = tid & (kD - 1);   // q/k channel of this thread
  const bool hi = tid >= kHi;      // k / v / output half (all threads if NT == 128)
  const bool lo = !hi || !kHalves; // q / decay half (all threads if NT == 128)
  const int sec = kHalves && hi ? 1 : 0;  // NT == 256: own conv section (0 = q, 1 = k)
  const int vi = tid - kHi;        // v channel v_base + vi if has_v
  const bool has_v = vi >= 0 && vi < R;
  const int grp = tid / LPR;       // update phase: row group
  const int gl = tid % LPR;        // update phase: lane within the row group

  __shared__ __align__(16) float s_q[kD];      // raw SiLU(q conv)
  __shared__ __align__(16) float s_k[kD];      // raw SiLU(k conv)
  __shared__ __align__(16) float s_decay[kD];
  __shared__ __align__(16) float s_red[8];     // per-warp q^2 sums [0,4), k^2 sums [4,8)
  __shared__ float s_v[R];
  __shared__ __align__(16) float s_o[kD];      // leader: the head's 128 outputs, opos order
  __shared__ __align__(8) unsigned long long s_bar;

  K3KDAS_TS(0);
  if (leader && tid == 0) mbar_init_expect(smem_addr(&s_bar), kD * 4);
  cluster_arrive_relaxed();  // "started" (+ mbarrier init visible); waited on at sync 1
  if constexpr (!kPrefetch) pdl_wait();

  // ------------------ independent of the predecessor: loads only ---------------
  // (no arithmetic on loaded values before the wait, so a slow load never delays
  // the PDL release; only the slot index is consumed, to form addresses)
  const int slot = p.idx[b];
  float* const st_head = p.state + static_cast<int64_t>(slot) * p.state_slot +
                         static_cast<int64_t>(h * kD + v_base) * kD;
  float4 hraw[PASSES][NJ];
#pragma unroll
  for (int ps = 0; ps < PASSES; ++ps) {
    const int row = ps * GROUPS + grp;
    if (row < R) {
#pragma unroll
      for (int j = 0; j < NJ; ++j) {
        hraw[ps][j] = __ldcg(
            reinterpret_cast<const float4*>(st_head + row * kD + 4 * (gl + LPR * j)));
      }
    }
  }

  __nv_bfloat16* const cs = p.cs + static_cast<int64_t>(slot) * p.cs_slot;
  const int hk = h * kD + ch;                  // q/k channel of this thread
  const int hv = h * kD + v_base + vi;         // v channel (valid if has_v)
  // "a" = q channel (NT 128) or the thread's own section (NT 256); "k" = k channel (NT 128).
  __nv_bfloat16 ca[3], ck[3], cv[3];
  float wa[4], wk[4], wv[4];
  const int sa = sec * kDim + hk;              // conv-section channel of "a"
#pragma unroll
  for (int j = 0; j < 3; ++j) ca[j] = cs[sa * kChS + j * kTapS];
#pragma unroll
  for (int j = 0; j < 4; ++j) wa[j] = p.w[sec * 4 * kDim + j * kDim + hk];
  const float ba = p.bias == nullptr ? 0.0f : p.bias[sa];
  float bk = 0.0f;
  if constexpr (!kHalves) {
#pragma unroll
    for (int j = 0; j < 3; ++j) ck[j] = cs[(kDim + hk) * kChS + j * kTapS];
#pragma unroll
    for (int j = 0; j < 4; ++j) wk[j] = p.w[4 * kDim + j * kDim + hk];
    bk = p.bias == nullptr ? 0.0f : p.bias[kDim + hk];
  }
  float bv = 0.0f;
  if (has_v) {
#pragma unroll
    for (int j = 0; j < 3; ++j) cv[j] = cs[(2 * kDim + hv) * kChS + j * kTapS];
#pragma unroll
    for (int j = 0; j < 4; ++j) wv[j] = p.w[8 * kDim + j * kDim + hv];
    bv = p.bias == nullptr ? 0.0f : p.bias[2 * kDim + hv];
  } else {
    cv[0] = cv[1] = cv[2] = __float2bfloat16(0.0f);
    wv[0] = wv[1] = wv[2] = wv[3] = 0.0f;
  }
  const float dtb = p.dt_bias[hk];
  const float a_log = p.a_log[h];
  float nw = 0.0f;
  if constexpr (kOnorm) {
    if (leader && hi) nw = p.norm_w[ch];
  }
  // Addresses of the dependent inputs, formed before the wait.
  const __nv_bfloat16* const xr = p.x + b * p.x_row;
  const __nv_bfloat16* const gp = p.g + b * kDim + hk;
  const __nv_bfloat16* const bp = p.beta + b * p.beta_row + h;
  const __nv_bfloat16* const gatep =
      kOnorm ? p.gate + b * p.gate_row + h * kD + ch : nullptr;
  const float lower_bound = p.lower_bound, scale = p.scale, eps = p.eps;

  // ------------------ dependent inputs ---------------------------------------
  if constexpr (kPrefetch) pdl_wait();
  K3KDAS_TS(1);
  const __nv_bfloat16 xa = xr[sa];
  __nv_bfloat16 xk = __float2bfloat16(0.0f);
  if constexpr (!kHalves) xk = xr[kDim + hk];
  const float graw = __bfloat162float(*gp);
  __nv_bfloat16 xv = __float2bfloat16(0.0f);
  if (has_v) xv = xr[2 * kDim + hv];
  const float beta_raw = __bfloat162float(*bp);
  float gate_raw = 0.0f;
  if constexpr (kOnorm) {
    if (leader && hi) gate_raw = __bfloat162float(*gatep);
  }
  pdl_trigger();
#ifdef K3KDAS_TIMING
  if (threadIdx.x == 0 && p.ts != nullptr) {  // stamp 7: dependent loads landed
    float sink = __bfloat162float(xa) + __bfloat162float(xk) + graw + beta_raw + gate_raw +
                 __bfloat162float(xv);
    asm volatile("" ::"f"(sink));
  }
  K3KDAS_TS(7);
#endif

  // ------------------ conv + SiLU + gates (per head, redundant per CTA) -------
  // kda_decode6.cu order: bias, 3 conv-state taps, then the new input.
  float a_acc = ba, k_acc = bk, v_acc = bv;
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    a_acc += __bfloat162float(ca[j]) * wa[j];
    if constexpr (!kHalves) k_acc += __bfloat162float(ck[j]) * wk[j];
    v_acc += __bfloat162float(cv[j]) * wv[j];
  }
  a_acc += __bfloat162float(xa) * wa[3];
  if constexpr (!kHalves) k_acc += __bfloat162float(xk) * wk[3];
  v_acc += __bfloat162float(xv) * wv[3];
  const float exp_a = __expf(a_log);
  const float g_arg = exp_a * (graw + dtb);
  // NT 256: the hi half runs the decay sigmoid chain on the output gate instead.
  const float x_arg = (kHalves && hi) || !kLowerBound ? gate_raw : g_arg;
#ifdef K3KDAS_TIMING
  if (threadIdx.x == 0 && p.ts != nullptr) asm volatile("" ::"f"(a_acc), "f"(k_acc), "f"(g_arg));
  K3KDAS_TS(10);  // conv sums done
#endif
  float as, ks = 0.0f, vs, decay, beta, gate;
  {
    const float ya = sig_den(a_acc);
    const float yk = sig_den(k_acc);
    const float yv = sig_den(v_acc);
    const float yb = sig_den(beta_raw);
    const float yx = sig_den(x_arg);
    float ra = rcp_fast(ya), rk = rcp_fast(yk), rv = rcp_fast(yv), rb = rcp_fast(yb),
          rx = rcp_fast(yx);
    const bool ok = rcp_fast_ok(ya) & rcp_fast_ok(yk) & rcp_fast_ok(yv) &
                    rcp_fast_ok(yb) & rcp_fast_ok(yx);
    if (!ok) {  // some operand outside the rcp fast range: exact IEEE division
      ra = 1.0f / ya;
      rk = 1.0f / yk;
      rv = 1.0f / yv;
      rb = 1.0f / yb;
      rx = 1.0f / yx;
    }
    as = a_acc * ra;
    if constexpr (!kHalves) ks = k_acc * rk;
    vs = v_acc * rv;
    beta = rb;
    if constexpr (kLowerBound) {
      decay = __expf(lower_bound * rx);  // meaningful where x_arg == g_arg
      gate = kHalves ? rx : 0.0f;
      if constexpr (!kHalves) {
        const float yg = sig_den(gate_raw);
        float rg = rcp_fast(yg);
        if (!rcp_fast_ok(yg)) rg = 1.0f / yg;
        gate = rg;
      }
    } else {
      decay = __expf(-exp_a * softplus_fast(graw + dtb));
      gate = rx;
    }
  }
#ifdef K3KDAS_TIMING
  if (threadIdx.x == 0 && p.ts != nullptr) asm volatile("" ::"f"(as), "f"(ks), "f"(decay));
  K3KDAS_TS(11);  // sigmoids done
#endif
  if constexpr (kHalves) {
    (hi ? s_k : s_q)[ch] = as;
    if (lo) s_decay[ch] = decay;
  } else {
    s_q[ch] = as;
    s_k[ch] = ks;
    s_decay[ch] = decay;
  }
  if (has_v) s_v[vi] = vs;

  // q/k L2 norms: warp butterfly, then (w0 + w2) + (w1 + w3) -- the tree the
  // old kernel's 8-warp block reduction produces (warps 4..7 contribute 0).
  if constexpr (kHalves) {
    float a2[1] = {as * as};
    shfl_tree<32>(a2);
    if (lane == 0) s_red[warp] = a2[0];
  } else {
    float qk2[2] = {as * as, ks * ks};
    shfl_tree<32>(qk2);
    if (lane == 0) {
      s_red[warp] = qk2[0];
      s_red[4 + warp] = qk2[1];
    }
  }
  K3KDAS_TS(8);
  cluster_wait();  // all CTAs of the cluster started, leader mbarrier initialised
  K3KDAS_TS(9);
  __syncthreads();
  K3KDAS_TS(2);

  // ------------------ delta-rule update of this thread's rows ------------------
  float r_q[NJ][4], r_k[NJ][4], r_d[NJ][4];
  {
    const float4 red0 = *reinterpret_cast<const float4*>(s_red);
    const float4 red1 = *reinterpret_cast<const float4*>(s_red + 4);
    const float qn = (red0.x + red0.z) + (red0.y + red0.w);
    const float kn = (red1.x + red1.z) + (red1.y + red1.w);
    const float fq = rsqrtf(qn + 1.0e-6f) * scale;  // old: s_q *= rsqrtf(..) * scale
    const float fk = rsqrtf(kn + 1.0e-6f);
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      const int kb = 4 * (gl + LPR * j);
      const float4 q4 = *reinterpret_cast<const float4*>(s_q + kb);
      const float4 k4 = *reinterpret_cast<const float4*>(s_k + kb);
      const float4 d4 = *reinterpret_cast<const float4*>(s_decay + kb);
      r_q[j][0] = q4.x * fq; r_q[j][1] = q4.y * fq; r_q[j][2] = q4.z * fq; r_q[j][3] = q4.w * fq;
      r_k[j][0] = k4.x * fk; r_k[j][1] = k4.y * fk; r_k[j][2] = k4.z * fk; r_k[j][3] = k4.w * fk;
      r_d[j][0] = d4.x; r_d[j][1] = d4.y; r_d[j][2] = d4.z; r_d[j][3] = d4.w;
    }
  }
#ifdef K3KDAS_TIMING
  if (threadIdx.x == 0 && p.ts != nullptr) asm volatile("" ::"f"(r_q[0][0]), "f"(r_k[NJ - 1][3]), "f"(r_d[NJ - 1][3]), "f"(hraw[0][0].x), "f"(hraw[PASSES - 1][NJ - 1].w));
  K3KDAS_TS(12);  // q/k normalised, state landed
#endif
  float hn[PASSES][NJ][4];
  float dot[PASSES];
#pragma unroll
  for (int ps = 0; ps < PASSES; ++ps) {
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      hn[ps][j][0] = hraw[ps][j].x * r_d[j][0];
      hn[ps][j][1] = hraw[ps][j].y * r_d[j][1];
      hn[ps][j][2] = hraw[ps][j].z * r_d[j][2];
      hn[ps][j][3] = hraw[ps][j].w * r_d[j][3];
    }
    dot[ps] = row_dot<LPR>(hn[ps], r_k);
  }
  shfl_tree<LPR>(dot);
#ifdef K3KDAS_TIMING
  if (threadIdx.x == 0 && p.ts != nullptr) asm volatile("" ::"f"(dot[0]));
  K3KDAS_TS(13);  // first row reduction done
#endif
#pragma unroll
  for (int ps = 0; ps < PASSES; ++ps) {
    const int row = ps * GROUPS + grp;
    const float v_new = (s_v[row < R ? row : 0] - dot[ps]) * beta;
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e) hn[ps][j][e] = hn[ps][j][e] + r_k[j][e] * v_new;
    }
    dot[ps] = row_dot<LPR>(hn[ps], r_q);
  }
  shfl_tree<LPR>(dot);
  K3KDAS_TS(3);

  // ------------------ o -> leader (st.async + mbarrier), stores ---------------
  const uint32_t bar0 = map_rank0(smem_addr(&s_bar));
#pragma unroll
  for (int ps = 0; ps < PASSES; ++ps) {
    const int row = ps * GROUPS + grp;
    if (row < R && gl == 0) {
      st_async_rank0(map_rank0(smem_addr(&s_o[opos(v_base + row)])), dot[ps], bar0);
    }
  }
#pragma unroll
  for (int ps = 0; ps < PASSES; ++ps) {
    const int row = ps * GROUPS + grp;
    if (row < R) {
#pragma unroll
      for (int j = 0; j < NJ; ++j) {
        *reinterpret_cast<float4*>(st_head + row * kD + 4 * (gl + LPR * j)) =
            make_float4(hn[ps][j][0], hn[ps][j][1], hn[ps][j][2], hn[ps][j][3]);
      }
    }
  }
  if (has_v) {  // v channels are read by this CTA only
    const int base = (2 * kDim + hv) * kChS;
    cs[base] = cv[1];
    cs[base + kTapS] = cv[2];
    cs[base + 2 * kTapS] = xv;
  }
  K3KDAS_TS(4);
  if (!leader) return;

  mbar_wait_parity0(smem_addr(&s_bar));  // all 128 o values of the head are in s_o
  K3KDAS_TS(5);

  // q/k conv-state shift, after every CTA of the cluster has consumed the old taps
  // (their o values, computed from them, have all arrived).
  auto conv_shift = [&]() {
    cs[sa * kChS] = ca[1];
    cs[sa * kChS + kTapS] = ca[2];
    cs[sa * kChS + 2 * kTapS] = xa;
    if constexpr (!kHalves) {
      const int bk_ = (kDim + hk) * kChS;
      cs[bk_] = ck[1];
      cs[bk_ + kTapS] = ck[2];
      cs[bk_ + 2 * kTapS] = xk;
    }
  };
  if (!hi) {  // NT 256, q half: nothing left but the q conv shift
    conv_shift();
    return;
  }

  const float raw_o = s_o[opos(ch)];
#ifdef K3KDAS_TIMING
  if (threadIdx.x == kHi && p.ts != nullptr) asm volatile("" ::"f"(raw_o));
  K3KDAS_TS(14);  // leader: first LDS of s_o returned
#endif
  float y;
  if constexpr (kOnorm) {
    // Sum of squares with the old kernel's tree: butterfly (xor 16..1) inside each
    // 32-row group, then (g0 + g2) + (g1 + g3).  Lane l8 = lane & 7 holds rows
    // 32g + l8 + 8j; xor 16 / 8 (j bits) are local, xor 4 / 2 / 1 are shuffles.
    const int l8 = lane & 7;
    float a[4];
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      const float4 o4 = *reinterpret_cast<const float4*>(&s_o[(l8 << 4) | (g << 2)]);
      // squares rounded on their own (no FMA contraction), as in the old kernel
      const float s0 = __fmul_rn(o4.x, o4.x), s1 = __fmul_rn(o4.y, o4.y),
                  s2 = __fmul_rn(o4.z, o4.z), s3 = __fmul_rn(o4.w, o4.w);
      a[g] = (s0 + s2) + (s1 + s3);
    }
    shfl_tree<8>(a);
    const float sumsq = (a[0] + a[2]) + (a[1] + a[3]);
    const float rstd = rsqrtf(sumsq / static_cast<float>(kD) + eps);
    y = raw_o * rstd * nw * gate;
  } else {
    y = raw_o;
  }
  p.out[(b * kH + h) * kD + ch] = __float2bfloat16(y);
  conv_shift();
  K3KDAS_TS(6);
}

template <int NT, int C, int LPR, bool kSD, bool kLowerBound, bool kOnorm, bool kPrefetch>
cudaError_t launch_split(const KdaSplitParams& p, int B, cudaStream_t stream) {
  auto kernel = &kda_split_kernel<NT, C, LPR, kSD, kLowerBound, kOnorm, kPrefetch>;
  if constexpr (C > 8) {
    static const cudaError_t attr_err = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    if (attr_err != cudaSuccess) return attr_err;
  }
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(C, kH, B);
  config.blockDim = dim3(NT);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = C;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = 2;
  return cudaLaunchKernelEx(&config, kernel, p);
}

template <int NT, int C, int LPR>
cudaError_t dispatch_c(const KdaSplitParams& p, int B, bool sd, bool lb,
                       bool onorm, bool prefetch, cudaStream_t s) {
#define K3KDAS_L(SD, LB, ON)                                              \
  return prefetch ? launch_split<NT, C, LPR, SD, LB, ON, true>(p, B, s)   \
                  : launch_split<NT, C, LPR, SD, LB, ON, false>(p, B, s)
  if (sd) {
    if (lb) {
      if (onorm) { K3KDAS_L(true, true, true); } else { K3KDAS_L(true, true, false); }
    } else {
      if (onorm) { K3KDAS_L(true, false, true); } else { K3KDAS_L(true, false, false); }
    }
  } else {
    if (lb) {
      if (onorm) { K3KDAS_L(false, true, true); } else { K3KDAS_L(false, true, false); }
    } else {
      if (onorm) { K3KDAS_L(false, false, true); } else { K3KDAS_L(false, false, false); }
    }
  }
#undef K3KDAS_L
}

#ifdef K3KDAS_TIMING
// Exhaustive check of the reciprocal fast path: for every fp32 bit pattern y with
// rcp_fast_ok(y), rcp_fast(y) must equal the IEEE round-to-nearest 1/y bit for bit.
__global__ void check_rcp_kernel(unsigned long long* counts) {
  unsigned long long bad = 0, checked = 0;
  for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < (1ull << 32);
       i += (uint64_t)gridDim.x * blockDim.x) {
    const float y = __uint_as_float(static_cast<uint32_t>(i));
    if (!rcp_fast_ok(y)) continue;
    ++checked;
    bad += __float_as_uint(rcp_fast(y)) != __float_as_uint(__frcp_rn(y));
  }
  atomicAdd(&counts[0], checked);
  atomicAdd(&counts[1], bad);
}
void check_rcp(torch::stable::Tensor& counts) {
  check_rcp_kernel<<<1024, 256>>>(static_cast<unsigned long long*>(counts.data_ptr()));
}

unsigned long long* g_host_ts = nullptr;
void set_ts(torch::stable::Tensor& ts) {
  g_host_ts = ts.numel() ? static_cast<unsigned long long*>(ts.data_ptr()) : nullptr;
}
#endif

// (CTAs per head C, lanes per state row LPR) by batch size.  Tuned on GB200 with
// bench_kda.py; see the report for the sweep.
struct SplitCfg {
  int NT;   // threads per CTA (128 or 256)
  int C;    // CTAs per (token, head)
  int LPR;  // lanes per state row in the update phase
};
// Tuned on GB200 with bench_kda.py (kda-only chain and after each predecessor):
// B=1: C=8 (C=16 is equal within noise but needs a non-portable cluster size);
// B>=2: C=4, i.e. fewer CTAs and less redundant q/k work; LPR 16 from B=5 on (also the
// most robust choice at B=16 under every predecessor).  NT=256 variants were never
// better by more than noise.
SplitCfg auto_split(int B) {
  if (B <= 1) return {128, 8, 8};
  if (B <= 4) return {128, 4, 8};
  return {128, 4, 16};
}
constexpr int kDefaultLPR = 8;

// Kimi K3 decode configuration (SD conv layout, lower_bound, gated RMSNorm, prefetch):
// every (NT, C, LPR) tuning variant.  Other configurations use NT 128, kDefaultLPR.
cudaError_t dispatch(const KdaSplitParams& p, int B, SplitCfg cfg, bool sd, bool lb,
                     bool onorm, bool prefetch, cudaStream_t s) {
  if (sd && lb && onorm && prefetch) {
    switch (cfg.NT * 10000 + cfg.LPR * 100 + cfg.C) {
#define K3KDAS_K3(T_, C_, L_) \
  case T_ * 10000 + L_ * 100 + C_: \
    return launch_split<T_, C_, L_, true, true, true, true>(p, B, s);
      K3KDAS_K3(128, 4, 8) K3KDAS_K3(128, 8, 8) K3KDAS_K3(128, 16, 8)
      K3KDAS_K3(128, 4, 16) K3KDAS_K3(128, 8, 16) K3KDAS_K3(128, 16, 16)
      K3KDAS_K3(128, 4, 32) K3KDAS_K3(128, 8, 32) K3KDAS_K3(128, 16, 32)
      K3KDAS_K3(256, 4, 8) K3KDAS_K3(256, 8, 8) K3KDAS_K3(256, 16, 8)
      K3KDAS_K3(256, 4, 16) K3KDAS_K3(256, 8, 16) K3KDAS_K3(256, 16, 16)
      K3KDAS_K3(256, 4, 32) K3KDAS_K3(256, 8, 32) K3KDAS_K3(256, 16, 32)
#undef K3KDAS_K3
      default:
        break;
    }
  }
  if (cfg.NT == 256) {
    switch (cfg.C) {
      case 4:
        return dispatch_c<256, 4, kDefaultLPR>(p, B, sd, lb, onorm, prefetch, s);
      case 8:
        return dispatch_c<256, 8, kDefaultLPR>(p, B, sd, lb, onorm, prefetch, s);
      case 16:
        return dispatch_c<256, 16, kDefaultLPR>(p, B, sd, lb, onorm, prefetch, s);
      default:
        return cudaErrorInvalidValue;
    }
  }
  switch (cfg.C) {
    case 4:
      return dispatch_c<128, 4, kDefaultLPR>(p, B, sd, lb, onorm, prefetch, s);
    case 8:
      return dispatch_c<128, 8, kDefaultLPR>(p, B, sd, lb, onorm, prefetch, s);
    case 16:
      return dispatch_c<128, 16, kDefaultLPR>(p, B, sd, lb, onorm, prefetch, s);
    default:
      return cudaErrorInvalidValue;
  }
}

bool prefetch_enabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("K3KDAS_NO_PREFETCH");
    return !(v != nullptr && v[0] == '1');
  }();
  return enabled;
}

void fused_kda_decode_impl(
    torch::stable::Tensor const& x, torch::stable::Tensor const& weight,
    std::optional<torch::stable::Tensor> const& bias,
    torch::stable::Tensor& conv_state, torch::stable::Tensor const& raw_g,
    torch::stable::Tensor const& raw_beta, torch::stable::Tensor const& a_log,
    torch::stable::Tensor const& dt_bias,
    torch::stable::Tensor const& state_indices, torch::stable::Tensor& state,
    torch::stable::Tensor& out, std::optional<double> lower_bound,
    std::optional<torch::stable::Tensor> const& output_gate,
    std::optional<torch::stable::Tensor> const& norm_weight, double norm_eps,
    int64_t split) {
  using torch::headeronly::ScalarType;
  constexpr int kConvWidth = 4;

  STD_TORCH_CHECK(x.is_cuda() && x.scalar_type() == ScalarType::BFloat16,
                  "x must be a CUDA bfloat16 tensor");
  STD_TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == ScalarType::Float,
                  "weight must be a CUDA float32 tensor");
  STD_TORCH_CHECK(
      conv_state.is_cuda() && conv_state.scalar_type() == ScalarType::BFloat16,
      "conv_state must be a CUDA bfloat16 tensor");
  STD_TORCH_CHECK(raw_g.is_cuda() && raw_g.scalar_type() == ScalarType::BFloat16,
                  "raw_g must be a CUDA bfloat16 tensor");
  STD_TORCH_CHECK(
      raw_beta.is_cuda() && raw_beta.scalar_type() == ScalarType::BFloat16,
      "raw_beta must be a CUDA bfloat16 tensor");
  STD_TORCH_CHECK(a_log.is_cuda() && a_log.scalar_type() == ScalarType::Float,
                  "A_log must be a CUDA float32 tensor");
  STD_TORCH_CHECK(dt_bias.is_cuda() && dt_bias.scalar_type() == ScalarType::Float,
                  "dt_bias must be a CUDA float32 tensor");
  STD_TORCH_CHECK(state.is_cuda() && state.scalar_type() == ScalarType::Float,
                  "state must be a CUDA float32 tensor");
  STD_TORCH_CHECK(out.is_cuda() && out.scalar_type() == ScalarType::BFloat16,
                  "out must be a CUDA bfloat16 tensor");
  STD_TORCH_CHECK(
      state_indices.is_cuda() && state_indices.scalar_type() == ScalarType::Int,
      "state_indices must be a CUDA int32 tensor");

  STD_TORCH_CHECK(x.dim() == 2 && x.size(1) == 3 * kDim,
                  "k3kdas: x must have shape [B, 3 * 6 * 128] (6 heads/rank)");
  int const batch_size = static_cast<int>(x.size(0));
  STD_TORCH_CHECK(batch_size > 0, "KDA decode fusion requires at least one row");
  STD_TORCH_CHECK(batch_size <= 65535, "k3kdas: batch too large");

  STD_TORCH_CHECK(weight.dim() == 3 && weight.is_contiguous() &&
                      weight.size(0) == 3 && weight.size(1) == kConvWidth &&
                      weight.size(2) == kDim,
                  "weight must have shape [3, 4, H * 128]");
  STD_TORCH_CHECK(conv_state.dim() == 3 && conv_state.size(1) == 3 * kDim &&
                      conv_state.size(2) == kConvWidth - 1,
                  "conv_state must have shape [slots, 3 * H * 128, 3]");
  STD_TORCH_CHECK(raw_g.dim() == 4 && raw_g.size(0) == 1 &&
                      raw_g.size(1) == batch_size && raw_g.size(2) == kH &&
                      raw_g.size(3) == kD,
                  "raw_g must have shape [1, B, H, 128]");
  STD_TORCH_CHECK(raw_beta.dim() == 3 && raw_beta.size(0) == 1 &&
                      raw_beta.size(1) == batch_size && raw_beta.size(2) == kH,
                  "raw_beta must have shape [1, B, H]");
  STD_TORCH_CHECK(a_log.is_contiguous() && a_log.numel() == kH,
                  "A_log must be contiguous with H elements");
  STD_TORCH_CHECK(dt_bias.is_contiguous() && dt_bias.numel() == kDim,
                  "dt_bias must be contiguous with H * 128 elements");
  STD_TORCH_CHECK(
      state_indices.is_contiguous() && state_indices.numel() == batch_size,
      "state_indices must be contiguous with B elements");
  STD_TORCH_CHECK(state.dim() == 4 && state.size(1) == kH && state.size(2) == kD &&
                      state.size(3) == kD,
                  "state must have shape [slots, H, 128, 128]");
  STD_TORCH_CHECK(out.dim() == 4 && out.size(0) == 1 &&
                      out.size(1) == batch_size && out.size(2) == kH &&
                      out.size(3) == kD,
                  "out must have shape [1, B, H, 128]");
  STD_TORCH_CHECK(x.stride(1) == 1, "x must be contiguous in its channel dimension");
  STD_TORCH_CHECK(
      conv_state.stride(0) >= 3 * kDim * (kConvWidth - 1) &&
          ((conv_state.stride(1) == 1 && conv_state.stride(2) == 3 * kDim) ||
           (conv_state.stride(1) == kConvWidth - 1 && conv_state.stride(2) == 1)),
      "conv_state must use the SD or DS cache layout");
  STD_TORCH_CHECK(state.stride(0) >= kH * kD * kD && state.stride(1) == kD * kD &&
                      state.stride(2) == kD && state.stride(3) == 1,
                  "state must have contiguous [H, 128, 128] slot contents");
  STD_TORCH_CHECK(state.stride(0) % 4 == 0 &&
                      reinterpret_cast<uintptr_t>(state.data_ptr()) % 16 == 0,
                  "k3kdas: state slots must be 16-byte aligned");
  STD_TORCH_CHECK(raw_g.is_contiguous(), "raw_g must be contiguous");
  STD_TORCH_CHECK(raw_beta.stride(2) == 1,
                  "raw_beta must be contiguous in its head dimension");
  STD_TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

  bool const apply_onorm = output_gate.has_value();
  STD_TORCH_CHECK(apply_onorm == norm_weight.has_value(),
                  "output_gate and norm_weight must be provided together");
  void const* output_gate_ptr = nullptr;
  float const* norm_weight_ptr = nullptr;
  int64_t output_gate_row_stride = 0;
  if (apply_onorm) {
    STD_TORCH_CHECK(output_gate->is_cuda() &&
                        output_gate->scalar_type() == ScalarType::BFloat16,
                    "output_gate must be a CUDA bfloat16 tensor");
    bool const gate_is_3d = output_gate->dim() == 3 &&
                            output_gate->size(0) == batch_size &&
                            output_gate->size(1) == kH && output_gate->size(2) == kD;
    bool const gate_is_4d = output_gate->dim() == 4 && output_gate->size(0) == 1 &&
                            output_gate->size(1) == batch_size &&
                            output_gate->size(2) == kH && output_gate->size(3) == kD;
    STD_TORCH_CHECK(gate_is_3d || gate_is_4d,
                    "output_gate must have shape [B, H, 128] or [1, B, H, 128]");
    int const row_dim = gate_is_3d ? 0 : 1;
    STD_TORCH_CHECK(output_gate->stride(output_gate->dim() - 1) == 1,
                    "output_gate must be contiguous in its last dimension");
    STD_TORCH_CHECK(output_gate->stride(row_dim + 1) == kD,
                    "output_gate must have contiguous head rows");
    STD_TORCH_CHECK(norm_weight->is_cuda() &&
                        norm_weight->scalar_type() == ScalarType::Float,
                    "norm_weight must be a CUDA float32 tensor");
    STD_TORCH_CHECK(norm_weight->is_contiguous() && norm_weight->numel() == kD,
                    "norm_weight must be contiguous with 128 elements");
    STD_TORCH_CHECK(norm_eps >= 0.0, "norm_eps must be non-negative");
    output_gate_ptr = output_gate->data_ptr();
    norm_weight_ptr = static_cast<float const*>(norm_weight->data_ptr());
    output_gate_row_stride = output_gate->stride(row_dim);
  }

  float const* bias_ptr = nullptr;
  if (bias.has_value()) {
    STD_TORCH_CHECK(bias->is_cuda() && bias->scalar_type() == ScalarType::Float,
                    "bias must be a CUDA float32 tensor");
    STD_TORCH_CHECK(bias->is_contiguous() && bias->numel() == 3 * kDim,
                    "bias must be contiguous with 3 * H * 128 elements");
    bias_ptr = static_cast<float const*>(bias->data_ptr());
  }

  KdaSplitParams p{};
  p.x = static_cast<const __nv_bfloat16*>(x.data_ptr());
  p.w = static_cast<const float*>(weight.data_ptr());
  p.bias = bias_ptr;
  p.cs = static_cast<__nv_bfloat16*>(conv_state.data_ptr());
  p.a_log = static_cast<const float*>(a_log.data_ptr());
  p.g = static_cast<const __nv_bfloat16*>(raw_g.data_ptr());
  p.dt_bias = static_cast<const float*>(dt_bias.data_ptr());
  p.beta = static_cast<const __nv_bfloat16*>(raw_beta.data_ptr());
  p.gate = static_cast<const __nv_bfloat16*>(output_gate_ptr);
  p.norm_w = norm_weight_ptr;
  p.idx = static_cast<const int*>(state_indices.data_ptr());
  p.state = static_cast<float*>(state.data_ptr());
  p.out = static_cast<__nv_bfloat16*>(out.data_ptr());
  p.x_row = x.stride(0);
  p.beta_row = raw_beta.stride(1);
  p.gate_row = output_gate_row_stride;
  p.cs_slot = conv_state.stride(0);
  p.state_slot = state.stride(0);
  p.lower_bound = lower_bound.has_value() ? static_cast<float>(*lower_bound) : 0.0f;
  p.scale = 0.08838834764831845f;
  p.eps = static_cast<float>(norm_eps);
#ifdef K3KDAS_TIMING
  p.ts = g_host_ts;
#else
  p.ts = nullptr;
#endif

  bool const sd = conv_state.stride(1) == 1;
  // split: 0 = auto; otherwise T * 10000 + LPR * 100 + C with T = 0 or 1 (128 threads)
  // or 2 (256 threads), LPR in {8, 16, 32} (0 = default), C in {4, 8, 16}.
  SplitCfg cfg = auto_split(batch_size);
  if (split > 0) {
    const int t = static_cast<int>(split / 10000);
    const int l = static_cast<int>((split / 100) % 100);
    cfg = SplitCfg{t == 2 ? 256 : 128, static_cast<int>(split % 100), l ? l : kDefaultLPR};
  }
  STD_TORCH_CHECK((cfg.C == 4 || cfg.C == 8 || cfg.C == 16) &&
                      (cfg.LPR == 8 || cfg.LPR == 16 || cfg.LPR == 32) && split / 10000 <= 2,
                  "k3kdas: unsupported split ", split);
  bool const prefetch = prefetch_enabled();

  torch::stable::accelerator::DeviceGuard const device_guard(x.get_device_index());
  cudaStream_t const stream = get_current_cuda_stream(x.get_device_index());
  cudaError_t err = dispatch(p, batch_size, cfg, sd, lower_bound.has_value(), apply_onorm,
                             prefetch, stream);
  if (err == cudaSuccess) err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess,
                  "k3kdas KDA decode launch failed: ", cudaGetErrorString(err));
}

void fused_kda_decode(
    torch::stable::Tensor const& x, torch::stable::Tensor const& weight,
    std::optional<torch::stable::Tensor> bias,
    torch::stable::Tensor& conv_state, torch::stable::Tensor const& raw_g,
    torch::stable::Tensor const& raw_beta, torch::stable::Tensor const& a_log,
    torch::stable::Tensor const& dt_bias,
    torch::stable::Tensor const& state_indices, torch::stable::Tensor& state,
    torch::stable::Tensor& out, std::optional<double> lower_bound,
    std::optional<torch::stable::Tensor> output_gate,
    std::optional<torch::stable::Tensor> norm_weight, double norm_eps) {
  fused_kda_decode_impl(x, weight, bias, conv_state, raw_g, raw_beta, a_log,
                        dt_bias, state_indices, state, out, lower_bound,
                        output_gate, norm_weight, norm_eps, 0);
}

// Same as fused_kda_decode with an explicit variant (see fused_kda_decode_impl for
// the split encoding); 0 = auto.  For tuning / benchmarking only.
void fused_kda_decode_split(
    torch::stable::Tensor const& x, torch::stable::Tensor const& weight,
    std::optional<torch::stable::Tensor> bias,
    torch::stable::Tensor& conv_state, torch::stable::Tensor const& raw_g,
    torch::stable::Tensor const& raw_beta, torch::stable::Tensor const& a_log,
    torch::stable::Tensor const& dt_bias,
    torch::stable::Tensor const& state_indices, torch::stable::Tensor& state,
    torch::stable::Tensor& out, std::optional<double> lower_bound,
    std::optional<torch::stable::Tensor> output_gate,
    std::optional<torch::stable::Tensor> norm_weight, double norm_eps,
    int64_t split) {
  fused_kda_decode_impl(x, weight, bias, conv_state, raw_g, raw_beta, a_log,
                        dt_bias, state_indices, state, out, lower_bound,
                        output_gate, norm_weight, norm_eps, split);
}

}  // namespace

#include <torch/csrc/stable/library.h>

#define K3KDAS_LIB(ns, m) STABLE_TORCH_LIBRARY(ns, m)
#define K3KDAS_LIB_IMPL(ns, k, m) STABLE_TORCH_LIBRARY_IMPL(ns, k, m)

K3KDAS_LIB(K3KDAS_NS, m) {
  m.def(
      "fused_kda_decode("
      "Tensor x, Tensor weight, Tensor? bias, Tensor! conv_state, "
      "Tensor raw_g, Tensor raw_beta, Tensor A_log, Tensor dt_bias, "
      "Tensor state_indices, Tensor! state, Tensor! out, "
      "float? lower_bound=None, Tensor? output_gate=None, "
      "Tensor? norm_weight=None, float norm_eps=1e-5) -> ()");
  m.def(
      "fused_kda_decode_split("
      "Tensor x, Tensor weight, Tensor? bias, Tensor! conv_state, "
      "Tensor raw_g, Tensor raw_beta, Tensor A_log, Tensor dt_bias, "
      "Tensor state_indices, Tensor! state, Tensor! out, "
      "float? lower_bound=None, Tensor? output_gate=None, "
      "Tensor? norm_weight=None, float norm_eps=1e-5, int split=0) -> ()");
#ifdef K3KDAS_TIMING
  m.def("set_ts(Tensor! ts) -> ()");
  m.def("check_rcp(Tensor! counts) -> ()");
#endif
}

K3KDAS_LIB_IMPL(K3KDAS_NS, CUDA, m) {
  m.impl("fused_kda_decode", TORCH_BOX(&fused_kda_decode));
  m.impl("fused_kda_decode_split", TORCH_BOX(&fused_kda_decode_split));
#ifdef K3KDAS_TIMING
  m.impl("set_ts", TORCH_BOX(&set_ts));
  m.impl("check_rcp", TORCH_BOX(&check_rcp));
#endif
}
