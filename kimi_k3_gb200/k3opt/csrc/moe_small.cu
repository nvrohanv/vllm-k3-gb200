// Small-batch (decode) MXFP4 MoE for Kimi K3 at TP16, reading vLLM's TRT-LLM
// weight layout in place and skipping its zero padding.
//
// Per rank, each routed expert holds a 192-wide slice of the 3072 intermediate
// dim, padded by vLLM to 256 for TRT-LLM's tiles. Layout per expert (see
// vllm/model_executor/layers/fused_moe/oracle/mxfp4.py,
// convert_weight_to_mxfp4_moe_kernel_format, TRTLLM branch):
//   w13 [512, 1792] u8: logical rows interleaved [up_0, gate_0, up_1, gate_1,
//       ...], then permuted within 32-row blocks: physical p holds logical
//       32*(p/32) + 4*(p%8) + (p%32)/8. Logical rows >= 384 (j >= 192) are
//       padding and live in physical rows 384..511.
//   w13 scales: UE8M0 per 32 elements, [512, 112], rows permuted like w13,
//       then 128x4-swizzled.
//   w2  [3584, 128] u8 (rows permuted the same way; K=256 padded, only the
//       first 96 bytes / 6 scale columns of each row are real), scales
//       [3584, 8] 128x4-swizzled.
// FP4 is E2M1 with the even element in the low nibble.
//
// Kernel 1: h[t, j, i] = situ(gate_i . x_t, up_i . x_t) for the expert
//           routed at (t, j); one warp per (t, j, i).
// Kernel 2: out[t, n] = sum_j w[t, j] * (W2_e[n, :192] . h[t, j, :]).
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

namespace {

constexpr int kHidden = 3584;         // latent dim (FC1 K, FC2 N)
constexpr int kInter = 192;           // real intermediate slice per rank
constexpr int kInterPad = 256;        // vLLM-padded slice
constexpr int kW13Rows = 2 * kInterPad;
constexpr int kW13RowBytes = kHidden / 2;       // 1792
constexpr int kW13ScaleCols = kHidden / 32;     // 112
constexpr int kW2RowBytes = kInterPad / 2;      // 128
constexpr int kW2ScaleCols = kInterPad / 32;    // 8
constexpr int kTopK = 16;

__device__ __forceinline__ int physical_row(int logical) {
  const int q = logical & 31;
  return (logical & ~31) + (q & 3) * 8 + (q >> 2);
}

// Offset of scale (row r [physical], col c) in a 128x4-swizzled [R, C] matrix.
__device__ __forceinline__ int swizzled_scale(int r, int c, int col_tiles) {
  return ((r >> 7) * col_tiles + (c >> 2)) * 512 + (r & 31) * 16 + ((r & 127) >> 5) * 4 + (c & 3);
}

__device__ __forceinline__ float ue8m0(uint8_t e) {
  return __uint_as_float(static_cast<uint32_t>(e) << 23);
}

// FP16 dot of one 32-element block: 16 bytes of FP4 weights against 16 half2
// activations in registers. Products are accumulated in half2 within the
// block (native e2m1x2->f16x2 cvt + HFMA2) and returned as fp32; blocks are
// combined in fp32 by the caller after applying the UE8M0 scale. Rounding here
// is far below the MXFP8 activation/intermediate rounding TRT-LLM applies.
__device__ __forceinline__ float dot32_h(uint4 w, const __half2 (&x)[16]) {
  const uint32_t words[4] = {w.x, w.y, w.z, w.w};
  __half2 acc0 = __float2half2_rn(0.f), acc1 = acc0;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
#pragma unroll
    for (int b = 0; b < 4; b += 2) {
      const __half2_raw w0 = __nv_cvt_fp4x2_to_halfraw2(
          static_cast<__nv_fp4x2_storage_t>((words[k] >> (8 * b)) & 0xff), __NV_E2M1);
      const __half2_raw w1 = __nv_cvt_fp4x2_to_halfraw2(
          static_cast<__nv_fp4x2_storage_t>((words[k] >> (8 * b + 8)) & 0xff), __NV_E2M1);
      acc0 = __hfma2(*reinterpret_cast<const __half2*>(&w0), x[k * 4 + b], acc0);
      acc1 = __hfma2(*reinterpret_cast<const __half2*>(&w1), x[k * 4 + b + 1], acc1);
    }
  }
  const float2 f = __half22float2(__hadd2(acc0, acc1));
  return f.x + f.y;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

// Load a lane's 32-element block of bf16 activations from shared memory as half2.
__device__ __forceinline__ void load_block_h(const __nv_bfloat16* src, __half2 (&x)[16]) {
#pragma unroll
  for (int v = 0; v < 4; ++v) {
    const uint4 raw = reinterpret_cast<const uint4*>(src)[v];
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int q = 0; q < 4; ++q) x[v * 4 + q] = __float22half2_rn(__bfloat1622float2(p[q]));
  }
}

constexpr int kWarps = 8;
constexpr int kFc1Chunks = (kW13ScaleCols + 31) / 32;  // 4 (lanes >= 16 do 3)
constexpr int kFc1Tasks = kTopK * kInter;               // (expert slot j, intermediate i) per token

struct Fc1Load {
  uint4 wu[kFc1Chunks], wg[kFc1Chunks];
  uint8_t su[kFc1Chunks], sg[kFc1Chunks];
};

__device__ __forceinline__ void fc1_issue(Fc1Load& L, const uint8_t* w13, const uint8_t* w13_scale,
                                          int e, int i, int lane) {
  const uint8_t* w = w13 + static_cast<long>(e) * kW13Rows * kW13RowBytes;
  const uint8_t* s = w13_scale + static_cast<long>(e) * kW13Rows * kW13ScaleCols;
  const int up_row = physical_row(2 * i), gate_row = physical_row(2 * i + 1);
#pragma unroll
  for (int k = 0; k < kFc1Chunks; ++k) {
    const int c = lane + 32 * k;
    if (c < kW13ScaleCols) {
      L.wu[k] = __ldg(reinterpret_cast<const uint4*>(w + up_row * kW13RowBytes + c * 16));
      L.wg[k] = __ldg(reinterpret_cast<const uint4*>(w + gate_row * kW13RowBytes + c * 16));
      L.su[k] = __ldg(s + swizzled_scale(up_row, c, kW13ScaleCols / 4));
      L.sg[k] = __ldg(s + swizzled_scale(gate_row, c, kW13ScaleCols / 4));
    }
  }
}

// grid: (G, M). Warps of token t grid-stride over its 16*192 (j, i) tasks; each
// task computes one up/gate row pair and writes h[t, j, i] = situ(gate, up) (fp16).
__global__ void __launch_bounds__(kWarps * 32)
fc1_situ_kernel(const __nv_bfloat16* __restrict__ x, const int* __restrict__ topk_ids,
                const uint8_t* __restrict__ w13, const uint8_t* __restrict__ w13_scale,
                __half* __restrict__ h, float beta, float linear_beta) {
  __shared__ __align__(16) __nv_bfloat16 xs[kHidden];
  __shared__ int es[kTopK];
  const int t = blockIdx.y;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");
  for (int v = threadIdx.x; v < kHidden / 8; v += blockDim.x)
    reinterpret_cast<uint4*>(xs)[v] = reinterpret_cast<const uint4*>(x + t * kHidden)[v];
  if (threadIdx.x < kTopK) es[threadIdx.x] = topk_ids[t * kTopK + threadIdx.x];
  __syncthreads();

  __half2 xr[kFc1Chunks][16];
#pragma unroll
  for (int k = 0; k < kFc1Chunks; ++k) {
    const int c = lane + 32 * k;
    if (c < kW13ScaleCols) load_block_h(xs + c * 32, xr[k]);
  }

  const int stride = gridDim.x * kWarps;
  int task = blockIdx.x * kWarps + warp;
  if (task >= kFc1Tasks) return;
  Fc1Load cur;
  fc1_issue(cur, w13, w13_scale, es[task / kInter], task % kInter, lane);
  while (true) {
    const int next = task + stride;
    Fc1Load nxt;
    if (next < kFc1Tasks) fc1_issue(nxt, w13, w13_scale, es[next / kInter], next % kInter, lane);
    float up = 0.f, gate = 0.f;
#pragma unroll
    for (int k = 0; k < kFc1Chunks; ++k) {
      if (lane + 32 * k < kW13ScaleCols) {
        up = fmaf(ue8m0(cur.su[k]), dot32_h(cur.wu[k], xr[k]), up);
        gate = fmaf(ue8m0(cur.sg[k]), dot32_h(cur.wg[k], xr[k]), gate);
      }
    }
    up = warp_sum(up);
    gate = warp_sum(gate);
    if (lane == 0) {
      const float sig = 1.f / (1.f + __expf(-gate));
      const float g = beta * tanhf(gate / beta) * sig;
      const float u = linear_beta > 0.f ? linear_beta * tanhf(up / linear_beta) : up;
      h[static_cast<long>(t) * kFc1Tasks + task] = __float2half_rn(g * u);
    }
    if (next >= kFc1Tasks) break;
    task = next;
    cur = nxt;
  }
}

constexpr int kPairs = kTopK * (kInter / 32);  // 96 (expert, K-block) pairs per output row
constexpr int kFc2PerLane = kPairs / 32;       // 3

// grid: (G, M). Warps of token t grid-stride over the 3584 output rows; each
// lane owns 3 (expert, K-block) pairs and keeps their h activations in registers.
__global__ void __launch_bounds__(kWarps * 32)
fc2_finalize_kernel(const __half* __restrict__ h, const int* __restrict__ topk_ids,
                    const float* __restrict__ topk_w, const uint8_t* __restrict__ w2,
                    const uint8_t* __restrict__ w2_scale, __nv_bfloat16* __restrict__ out) {
  __shared__ __align__(16) __half hs[kFc1Tasks];
  __shared__ long bases[kTopK];
  __shared__ float wsc[kTopK];
  const int t = blockIdx.y;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");
  for (int v = threadIdx.x; v < kFc1Tasks / 8; v += blockDim.x)
    reinterpret_cast<uint4*>(hs)[v] = reinterpret_cast<const uint4*>(h + static_cast<long>(t) * kFc1Tasks)[v];
  if (threadIdx.x < kTopK) {
    bases[threadIdx.x] = static_cast<long>(topk_ids[t * kTopK + threadIdx.x]) * kHidden;
    wsc[threadIdx.x] = topk_w[t * kTopK + threadIdx.x];
  }
  __syncthreads();

  __half2 hr[kFc2PerLane][16];
  long base[kFc2PerLane];
  float wj[kFc2PerLane];
  int cblk[kFc2PerLane];
#pragma unroll
  for (int k = 0; k < kFc2PerLane; ++k) {
    const int q = lane + 32 * k, j = q / (kInter / 32), c = q % (kInter / 32);
    const uint4* src = reinterpret_cast<const uint4*>(hs + j * kInter + c * 32);
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      const uint4 raw = src[v];
      const __half2* p = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
      for (int u = 0; u < 4; ++u) hr[k][v * 4 + u] = p[u];
    }
    base[k] = bases[j];
    wj[k] = wsc[j];
    cblk[k] = c;
  }

  for (int n = blockIdx.x * kWarps + warp; n < kHidden; n += gridDim.x * kWarps) {
    const int p = physical_row(n);
    uint4 wv[kFc2PerLane];
    uint8_t sc[kFc2PerLane];
#pragma unroll
    for (int k = 0; k < kFc2PerLane; ++k) {
      wv[k] = __ldg(reinterpret_cast<const uint4*>(w2 + (base[k] + p) * kW2RowBytes + cblk[k] * 16));
      sc[k] = __ldg(w2_scale + base[k] * kW2ScaleCols + swizzled_scale(p, cblk[k], kW2ScaleCols / 4));
    }
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < kFc2PerLane; ++k) acc = fmaf(wj[k] * ue8m0(sc[k]), dot32_h(wv[k], hr[k]), acc);
    acc = warp_sum(acc);
    if (lane == 0) out[t * kHidden + n] = __float2bfloat16(acc);
  }
}

// ---------------------------------------------------------------------------
// Fused persistent variant for the smallest batches. FC2 weight addresses only
// depend on the routing, so every warp prefetches its FC2 rows into registers
// before doing FC1; a grid-wide barrier then separates FC1 (writes h) from FC2
// (reads h from L2). All CTAs must be co-resident: the grid is sized from the
// occupancy calculator, and dependents are only released after the barrier so
// early-launched PDL kernels cannot occupy SMs a straggling CTA still needs.

// Dot of one 32-element FP4 block with half2 activations in shared memory at
// x2[w * kStride] (w = 0..15), fp16 accumulation within the block.
template <int kStride>
__device__ __forceinline__ float dot32_hs(uint4 w, const __half2* x2) {
  const uint32_t words[4] = {w.x, w.y, w.z, w.w};
  __half2 acc0 = __float2half2_rn(0.f), acc1 = acc0;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
#pragma unroll
    for (int b = 0; b < 4; b += 2) {
      const __half2_raw w0 = __nv_cvt_fp4x2_to_halfraw2(
          static_cast<__nv_fp4x2_storage_t>((words[k] >> (8 * b)) & 0xff), __NV_E2M1);
      const __half2_raw w1 = __nv_cvt_fp4x2_to_halfraw2(
          static_cast<__nv_fp4x2_storage_t>((words[k] >> (8 * b + 8)) & 0xff), __NV_E2M1);
      acc0 = __hfma2(*reinterpret_cast<const __half2*>(&w0), x2[(k * 4 + b) * kStride], acc0);
      acc1 = __hfma2(*reinterpret_cast<const __half2*>(&w1), x2[(k * 4 + b + 1) * kStride], acc1);
    }
  }
  const float2 f = __half22float2(__hadd2(acc0, acc1));
  return f.x + f.y;
}

__device__ __forceinline__ void grid_barrier(unsigned long long* counter) {
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned long long n = gridDim.x * gridDim.y;
    __threadfence();
    const unsigned long long old = atomicAdd(counter, 1ull);
    const unsigned long long target = (old / n + 1) * n;
    unsigned long long cur;
    do {
      asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(cur) : "l"(counter));
    } while (cur < target);
  }
  __syncthreads();
}

constexpr int kFusedWarps = 16;

// grid: (G) CTAs, all co-resident; M tokens handled by every CTA.
__global__ void __launch_bounds__(kFusedWarps * 32)
moe_fused_kernel(const __nv_bfloat16* __restrict__ x, const int* __restrict__ topk_ids,
                 const float* __restrict__ topk_w, const uint8_t* __restrict__ w13,
                 const uint8_t* __restrict__ w13_scale, const uint8_t* __restrict__ w2,
                 const uint8_t* __restrict__ w2_scale, __half* __restrict__ h,
                 unsigned long long* __restrict__ barrier, __nv_bfloat16* __restrict__ out,
                 int M, float beta, float linear_beta, unsigned long long* __restrict__ trace) {
#define K3_MARK(slot)                                                                 \
  if (trace && threadIdx.x == 0) {                                                    \
    unsigned long long ts;                                                            \
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts));                           \
    trace[blockIdx.x * 8 + (slot)] = ts;                                              \
  }
  K3_MARK(0)
  // xs[t][w * kW13ScaleCols + c] = (x[t, c*32 + 2w], x[t, c*32 + 2w + 1])
  extern __shared__ __half2 xs[];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int gwarp = blockIdx.x * kFusedWarps + warp;
  const int nwarps = gridDim.x * kFusedWarps;
  constexpr int kBlocks = kInter / 32;  // 6

  asm volatile("griddepcontrol.wait;" ::: "memory");
  K3_MARK(1)

  // 1) Prefetch this warp's first FC2 row (per token) into registers.
  const int fc2_tasks = M * kHidden;
  int f2 = gwarp;
  uint4 pw[kFc2PerLane];
  uint8_t ps[kFc2PerLane];
  if (f2 < fc2_tasks) {
    const int t = f2 / kHidden, n = f2 % kHidden, p = physical_row(n);
#pragma unroll
    for (int k = 0; k < kFc2PerLane; ++k) {
      const int q = lane + 32 * k, j = q / kBlocks, c = q % kBlocks;
      const long base = static_cast<long>(__ldg(topk_ids + t * kTopK + j)) * kHidden;
      pw[k] = __ldg(reinterpret_cast<const uint4*>(w2 + (base + p) * kW2RowBytes + c * 16));
      ps[k] = __ldg(w2_scale + base * kW2ScaleCols + swizzled_scale(p, c, kW2ScaleCols / 4));
    }
  }

  // 2) Stage x (all tokens) transposed as half2.
  for (int v = threadIdx.x; v < M * kHidden / 8; v += blockDim.x) {
    const int t = v / (kHidden / 8), vv = v % (kHidden / 8);
    const uint4 raw = reinterpret_cast<const uint4*>(x + t * kHidden)[vv];
    const __nv_bfloat162* pr = reinterpret_cast<const __nv_bfloat162*>(&raw);
    const int c = vv / 4, w0 = (vv % 4) * 4;
#pragma unroll
    for (int q = 0; q < 4; ++q)
      xs[t * 16 * kW13ScaleCols + (w0 + q) * kW13ScaleCols + c] =
          __float22half2_rn(__bfloat1622float2(pr[q]));
  }
  __syncthreads();

  K3_MARK(2)
  // 3) FC1 + SiTU over all (t, j, i) tasks.
  for (int task = gwarp; task < M * kFc1Tasks; task += nwarps) {
    const int t = task / kFc1Tasks, r = task % kFc1Tasks;
    const int e = __ldg(topk_ids + t * kTopK + r / kInter), i = r % kInter;
    Fc1Load L;
    fc1_issue(L, w13, w13_scale, e, i, lane);
    const __half2* xt = xs + t * 16 * kW13ScaleCols;
    float up = 0.f, gate = 0.f;
#pragma unroll
    for (int k = 0; k < kFc1Chunks; ++k) {
      const int c = lane + 32 * k;
      if (c < kW13ScaleCols) {
        up = fmaf(ue8m0(L.su[k]), dot32_hs<kW13ScaleCols>(L.wu[k], xt + c), up);
        gate = fmaf(ue8m0(L.sg[k]), dot32_hs<kW13ScaleCols>(L.wg[k], xt + c), gate);
      }
    }
    up = warp_sum(up);
    gate = warp_sum(gate);
    if (lane == 0) {
      const float sig = 1.f / (1.f + __expf(-gate));
      const float g = beta * tanhf(gate / beta) * sig;
      const float u = linear_beta > 0.f ? linear_beta * tanhf(up / linear_beta) : up;
      h[static_cast<long>(t) * kFc1Tasks + r] = __float2half_rn(g * u);
    }
  }

  K3_MARK(3)
  grid_barrier(barrier + M);
  K3_MARK(4)
  asm volatile("griddepcontrol.launch_dependents;");

  // 4) FC2 + finalize. Stage h (all tokens) into shared memory once per CTA
  // (L2 reads bypass L1: h was written by other CTAs of this launch); every
  // warp reading it from L2 directly hot-spots the few L2 lines holding it.
  __half* hsm = reinterpret_cast<__half*>(xs);  // x is no longer needed
  for (int v = threadIdx.x; v < M * kFc1Tasks / 8; v += blockDim.x)
    reinterpret_cast<uint4*>(hsm)[v] = __ldcg(reinterpret_cast<const uint4*>(h) + v);
  __syncthreads();
  bool prefetched = true;
  for (; f2 < fc2_tasks; f2 += nwarps) {
    const int t = f2 / kHidden, n = f2 % kHidden, p = physical_row(n);
    uint4 wv[kFc2PerLane];
    uint8_t sc[kFc2PerLane];
#pragma unroll
    for (int k = 0; k < kFc2PerLane; ++k) {
      const int q = lane + 32 * k, j = q / kBlocks, c = q % kBlocks;
      if (prefetched) {
        wv[k] = pw[k];
        sc[k] = ps[k];
      } else {
        const long base = static_cast<long>(__ldg(topk_ids + t * kTopK + j)) * kHidden;
        wv[k] = __ldg(reinterpret_cast<const uint4*>(w2 + (base + p) * kW2RowBytes + c * 16));
        sc[k] = __ldg(w2_scale + base * kW2ScaleCols + swizzled_scale(p, c, kW2ScaleCols / 4));
      }
    }
    prefetched = false;
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < kFc2PerLane; ++k) {
      const int q = lane + 32 * k, j = q / kBlocks, c = q % kBlocks;
      const uint4* src = reinterpret_cast<const uint4*>(hsm + t * kFc1Tasks + j * kInter + c * 32);
      __half2 hr[16];
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const uint4 raw = src[v];
        const __half2* pp = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
        for (int u = 0; u < 4; ++u) hr[v * 4 + u] = pp[u];
      }
      acc = fmaf(__ldg(topk_w + t * kTopK + j) * ue8m0(sc[k]), dot32_h(wv[k], hr), acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) out[t * kHidden + n] = __float2bfloat16(acc);
  }
  __syncthreads();
  K3_MARK(5)
#undef K3_MARK
}

template <typename Kernel, typename... Args>
void launch_pdl(Kernel kernel, dim3 grid, dim3 block, cudaStream_t stream, Args... args) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, kernel, args...);
}

}  // namespace

void moe_small(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_weights,
               torch::Tensor w13, torch::Tensor w13_scale, torch::Tensor w2,
               torch::Tensor w2_scale, torch::Tensor workspace, torch::Tensor out, double beta,
               double linear_beta, int64_t stages) {
  const int M = x.size(0);
  TORCH_CHECK(x.size(1) == kHidden && x.is_contiguous() && x.scalar_type() == at::kBFloat16);
  TORCH_CHECK(topk_ids.size(1) == kTopK && topk_ids.scalar_type() == at::kInt && topk_ids.is_contiguous());
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat && topk_weights.is_contiguous());
  TORCH_CHECK(w13.size(1) == kW13Rows && w13.size(2) == kW13RowBytes);
  TORCH_CHECK(w2.size(1) == kHidden && w2.size(2) == kW2RowBytes);
  TORCH_CHECK(w13_scale.numel() == w13.size(0) * kW13Rows * kW13ScaleCols);
  TORCH_CHECK(w2_scale.numel() == w2.size(0) * kHidden * kW2ScaleCols);
  TORCH_CHECK(workspace.numel() * workspace.element_size() >= M * kTopK * kInter * 2);
  TORCH_CHECK(out.size(0) == M && out.size(1) == kHidden && out.scalar_type() == at::kBFloat16);
  if (M == 0) return;
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  auto u8 = [](const torch::Tensor& t) { return reinterpret_cast<const uint8_t*>(t.data_ptr()); };
  static int num_sms = 0;
  if (num_sms == 0) cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, x.get_device());
  const int grid_x = std::max(1, (2 * num_sms + M - 1) / M);
  if (stages & 1) launch_pdl(fc1_situ_kernel, dim3(grid_x, M), dim3(kWarps * 32), stream,
             reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), topk_ids.data_ptr<int>(),
             u8(w13), u8(w13_scale), reinterpret_cast<__half*>(workspace.data_ptr()), static_cast<float>(beta),
             static_cast<float>(linear_beta));
  if (stages & 2) launch_pdl(fc2_finalize_kernel, dim3(grid_x, M), dim3(kWarps * 32), stream,
             reinterpret_cast<const __half*>(workspace.data_ptr()), topk_ids.data_ptr<int>(),
             static_cast<const float*>(topk_weights.data_ptr<float>()), u8(w2), u8(w2_scale),
             reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void moe_fused(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_weights,
               torch::Tensor w13, torch::Tensor w13_scale, torch::Tensor w2,
               torch::Tensor w2_scale, torch::Tensor workspace, torch::Tensor barrier,
               torch::Tensor out, double beta, double linear_beta,
               std::optional<torch::Tensor> trace) {
  const int M = x.size(0);
  TORCH_CHECK(M >= 1 && M <= 4, "moe_fused supports 1..4 tokens");
  TORCH_CHECK(x.size(1) == kHidden && x.is_contiguous() && x.scalar_type() == at::kBFloat16);
  TORCH_CHECK(topk_ids.size(1) == kTopK && topk_ids.scalar_type() == at::kInt && topk_ids.is_contiguous());
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat && topk_weights.is_contiguous());
  TORCH_CHECK(workspace.numel() * workspace.element_size() >= M * kTopK * kInter * 2);
  TORCH_CHECK(barrier.scalar_type() == at::kLong && barrier.numel() >= 8);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const size_t smem = static_cast<size_t>(M) * 16 * kW13ScaleCols * sizeof(__half2);
  static int grid_for_m[5] = {0, 0, 0, 0, 0};
  if (grid_for_m[M] == 0) {
    cudaFuncSetAttribute(moe_fused_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * 16 * kW13ScaleCols * 4);
    int per_sm = 0, sms = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, moe_fused_kernel, kFusedWarps * 32, smem);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, x.get_device());
    grid_for_m[M] = std::max(1, std::min(per_sm, 2)) * sms;
  }
  auto u8 = [](const torch::Tensor& t) { return reinterpret_cast<const uint8_t*>(t.data_ptr()); };
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(grid_for_m[M]);
  cfg.blockDim = dim3(kFusedWarps * 32);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, moe_fused_kernel, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
                     static_cast<const int*>(topk_ids.data_ptr<int>()),
                     static_cast<const float*>(topk_weights.data_ptr<float>()), u8(w13), u8(w13_scale),
                     u8(w2), u8(w2_scale), reinterpret_cast<__half*>(workspace.data_ptr()),
                     reinterpret_cast<unsigned long long*>(barrier.data_ptr()),
                     reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), M, static_cast<float>(beta),
                     static_cast<float>(linear_beta),
                     trace ? reinterpret_cast<unsigned long long*>(trace->data_ptr()) : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

TORCH_LIBRARY(k3moe, m) {
  m.def("moe_small(Tensor x, Tensor topk_ids, Tensor topk_weights, Tensor w13, Tensor w13_scale, "
        "Tensor w2, Tensor w2_scale, Tensor(a!) workspace, Tensor(b!) out, float beta, "
        "float linear_beta, int stages=3) -> ()");
  m.def("moe_fused(Tensor x, Tensor topk_ids, Tensor topk_weights, Tensor w13, Tensor w13_scale, "
        "Tensor w2, Tensor w2_scale, Tensor(a!) workspace, Tensor(b!) barrier, Tensor(c!) out, "
        "float beta, float linear_beta, Tensor(d!)? trace=None) -> ()");
}
TORCH_LIBRARY_IMPL(k3moe, CUDA, m) {
  m.impl("moe_small", &moe_small);
  m.impl("moe_fused", &moe_fused);
}
