"""Validate k3moe.moe_small against an fp32 reference and TRT-LLM on vLLM's layout."""

import sys

import torch

sys.path.insert(0, "/work/k3opt")
from build import build  # noqa: E402

build()
from flashinfer.fp4_quantization import nvfp4_block_scale_interleave  # noqa: E402
from flashinfer.fused_moe.core import get_w2_permute_indices_with_cache  # noqa: E402

E, H, I, IP, TOPK = 896, 3584, 192, 256, 16
dev = "cuda"
torch.manual_seed(0)
LUT = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=dev)


def rand_fp4(rows, cols, real_rows=None, real_cols=None):
    """Packed FP4 [rows, cols/2] and UE8M0 scales [rows, cols/32]; zero outside real region."""
    q = torch.randint(0, 256, (E, rows, cols // 2), dtype=torch.uint8, device=dev)
    s = torch.randint(118, 124, (E, rows, cols // 32), dtype=torch.uint8, device=dev)
    if real_rows is not None:
        q[:, real_rows:] = 0
    if real_cols is not None:
        q[:, :, real_cols // 2:] = 0
    return q, s


def dequant(q, s):
    lo, hi = (q & 15).long(), (q >> 4).long()
    vals = torch.stack([LUT[lo], LUT[hi]], dim=-1).flatten(-2)
    scale = torch.pow(2.0, s.float() - 127).repeat_interleave(32, dim=-1)
    return vals * scale


# Logical (vLLM-loaded) layout: w13 = [gate(IP rows); up(IP rows)], rows >= I zero.
w1q, w1s = rand_fp4(IP, H, real_rows=I)
w3q, w3s = rand_fp4(IP, H, real_rows=I)
w2q, w2s = rand_fp4(H, IP, real_cols=I)
w13q = torch.cat([w1q, w3q], 1)
w13s = torch.cat([w1s, w3s], 1)

# vLLM TRTLLM preprocessing (oracle/mxfp4.py): interleave [up, gate], permute, swizzle.
cache = {}
il_q = torch.stack([w13q[:, IP:], w13q[:, :IP]], dim=2).reshape(w13q.shape)
il_s = torch.stack([w13s[:, IP:], w13s[:, :IP]], dim=2).reshape(w13s.shape)
perm = get_w2_permute_indices_with_cache(cache, il_q[0], 128).to(dev)
sperm = get_w2_permute_indices_with_cache(cache, il_s[0], 128, num_elts_per_sf=16).to(dev)
p13q = il_q[:, perm].contiguous()
p13s = nvfp4_block_scale_interleave(il_s[:, sperm].contiguous().reshape(E * 2 * IP, -1)).reshape(E, 2 * IP, H // 32)
perm2 = get_w2_permute_indices_with_cache(cache, w2q[0], 128).to(dev)
sperm2 = get_w2_permute_indices_with_cache(cache, w2s[0], 128, num_elts_per_sf=16).to(dev)
p2q = w2q[:, perm2].contiguous()
p2s = nvfp4_block_scale_interleave(w2s[:, sperm2].contiguous().reshape(E * H, -1)).reshape(E, H, IP // 32)


def reference(x, ids, wts, beta=4.0, lb=25.0):
    out = torch.zeros(x.shape[0], H, device=dev)
    for t in range(x.shape[0]):
        for j in range(TOPK):
            e = int(ids[t, j])
            g = dequant(w1q[e, :I], w1s[e, :I]) @ x[t].float()
            u = dequant(w3q[e, :I], w3s[e, :I]) @ x[t].float()
            h = beta * torch.tanh(g / beta) * torch.sigmoid(g) * lb * torch.tanh(u / lb)
            out[t] += wts[t, j] * (dequant(w2q[e], w2s[e])[:, :I] @ h)
    return out


def ours(x, ids, wts):
    ws = torch.empty(x.shape[0] * 16 * 96 * 2, device=dev)
    out = torch.empty(x.shape[0], H, device=dev, dtype=torch.bfloat16)
    torch.ops.k3moe.moe_small(x, ids, wts, p13q, p13s, p2q, p2s, ws, out, 4.0, 25.0)
    return out


def trtllm(x, ids, wts):
    from flashinfer import trtllm_fp4_block_scale_routed_moe
    from flashinfer.fused_moe import RoutingMethodType
    out = torch.empty(x.shape[0], H, device=dev, dtype=torch.bfloat16)
    alpha = torch.full((E,), 4.0, device=dev)
    beta = torch.full((E,), 25.0, device=dev)
    trtllm_fp4_block_scale_routed_moe(
        topk_ids=(ids, wts), routing_bias=None, hidden_states=x, hidden_states_scale=None,
        gemm1_weights=p13q, gemm1_weights_scale=p13s.view(torch.float8_e4m3fn), gemm1_bias=None,
        gemm1_alpha=alpha, gemm1_beta=beta, gemm1_clamp_limit=None, gemm2_weights=p2q,
        gemm2_weights_scale=p2s.view(torch.float8_e4m3fn), gemm2_bias=None,
        output1_scale_scalar=None, output1_scale_gate_scalar=None, output2_scale_scalar=None,
        num_experts=E, top_k=TOPK, n_group=None, topk_group=None, intermediate_size=IP,
        local_expert_offset=0, local_num_experts=E, routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize, do_finalize=True, enable_pdl=True,
        activation_type=4, output=out, tune_max_num_tokens=16)
    return out


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


for M in (1, 8):
    x = (torch.randn(M, H, device=dev) * 0.5).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
    wts = torch.softmax(torch.randn(M, TOPK, device=dev), -1)
    ref = reference(x, ids, wts)
    o = ours(x, ids, wts)
    msg = f"M={M}: ours vs fp32 ref rel err {rel(o, ref):.2e}"
    try:
        tr = trtllm(x, ids, wts)
        msg += f" | trtllm vs ref {rel(tr, ref):.2e} | ours vs trtllm {rel(o, tr):.2e}"
    except Exception as ex:  # activation id / API differences
        msg += f" | trtllm failed: {str(ex)[:160]}"
    print(msg, flush=True)


def graph_us(fn, iters=32):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    g.replay(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(5):
        g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / (5 * iters)


for M in (1, 2, 4, 8):
    # Different experts per call so weights stream from HBM (E*1.4 MB >> L2).
    calls = []
    for _ in range(32):
        x = (torch.randn(M, H, device=dev) * 0.5).to(torch.bfloat16)
        ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
        wts = torch.softmax(torch.randn(M, TOPK, device=dev), -1)
        calls.append((x, ids, wts))
    it = iter(range(10**9))
    ws = torch.empty(M * 16 * 96 * 2, device=dev)
    out = torch.empty(M, H, device=dev, dtype=torch.bfloat16)
    k = [0]
    def run_ours():
        x, ids, wts = calls[k[0] % 32]; k[0] += 1
        torch.ops.k3moe.moe_small(x, ids, wts, p13q, p13s, p2q, p2s, ws, out, 4.0, 25.0)
    t_ours = graph_us(run_ours)
    k[0] = 0
    def run_tr():
        x, ids, wts = calls[k[0] % 32]; k[0] += 1
        trtllm(x, ids, wts)
    try:
        t_tr = graph_us(run_tr)
    except Exception as ex:
        t_tr = float("nan")
    print(f"M={M}: moe_small {t_ours:6.2f} us   trtllm routed (bf16 act) {t_tr:6.2f} us", flush=True)


# Fused persistent kernel (M <= 4).
barrier = torch.zeros(8, dtype=torch.int64, device=dev)
for M in (1, 2, 4):
    x = (torch.randn(M, H, device=dev) * 0.5).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
    wts = torch.softmax(torch.randn(M, TOPK, device=dev), -1)
    ws = torch.zeros(M * TOPK * I, device=dev, dtype=torch.float16)
    out = torch.empty(M, H, device=dev, dtype=torch.bfloat16)
    torch.ops.k3moe.moe_fused(x, ids, wts, p13q, p13s, p2q, p2s, ws, barrier, out, 4.0, 25.0)
    err = rel(out, reference(x, ids, wts))
    calls = []
    for _ in range(32):
        xx = (torch.randn(M, H, device=dev) * 0.5).to(torch.bfloat16)
        ii = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
        calls.append((xx, ii, torch.softmax(torch.randn(M, TOPK, device=dev), -1)))
    k = [0]
    def run_fused():
        xx, ii, ww = calls[k[0] % 32]; k[0] += 1
        torch.ops.k3moe.moe_fused(xx, ii, ww, p13q, p13s, p2q, p2s, ws, barrier, out, 4.0, 25.0)
    print(f"M={M}: moe_fused rel err {err:.2e}  {graph_us(run_fused):6.2f} us", flush=True)
