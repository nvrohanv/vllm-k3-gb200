"""Check the 6-head fused KDA decode against vLLM's native 12-head kernel.

Heads are independent, so a 12-head problem whose heads 6..11 duplicate heads
0..5 must reproduce the 6-head result on heads 0..5 (outputs, recurrent state,
conv state). Also times the fused kernel at H=6.
"""

import sys

import torch

sys.path.insert(0, "/work/k3opt")
import vllm._custom_ops  # noqa: E402,F401
from build import build  # noqa: E402

build()
D, W, SLOTS = 128, 4, 16
dev = "cuda"
torch.manual_seed(0)


def dup_sections(t, H, dim):
    """[..., 3*H*D] (q|k|v sections) -> same with each section's heads doubled."""
    secs = t.split(H * D, dim=dim)
    return torch.cat([torch.cat([s, s], dim=dim) for s in secs], dim=dim)


def make(B, H):
    x = torch.randn(B, 3 * H * D, device=dev, dtype=torch.bfloat16)
    weight = torch.randn(3, W, H * D, device=dev, dtype=torch.float32) * 0.3
    conv_state = torch.randn(SLOTS, 3 * H * D, W - 1, device=dev, dtype=torch.bfloat16)
    raw_g = torch.randn(1, B, H, D, device=dev, dtype=torch.bfloat16)
    raw_beta = torch.randn(1, B, H, device=dev, dtype=torch.bfloat16)
    a_log = torch.randn(H, device=dev, dtype=torch.float32) * 0.5
    dt_bias = torch.randn(H * D, device=dev, dtype=torch.float32) * 0.1
    idx = torch.randperm(SLOTS, device=dev)[:B].to(torch.int32)
    state = torch.randn(SLOTS, H, D, D, device=dev, dtype=torch.float32) * 0.05
    gate = torch.randn(1, B, H, D, device=dev, dtype=torch.bfloat16)
    norm_w = torch.rand(D, device=dev, dtype=torch.float32) + 0.5
    return [x, weight, conv_state, raw_g, raw_beta, a_log, dt_bias, idx, state, gate, norm_w]


def to12(a):
    x, weight, conv_state, raw_g, raw_beta, a_log, dt_bias, idx, state, gate, norm_w = a
    return [dup_sections(x, 6, 1), torch.cat([weight, weight], dim=2).contiguous(),
            dup_sections(conv_state, 6, 1).contiguous(), torch.cat([raw_g, raw_g], 2).contiguous(),
            torch.cat([raw_beta, raw_beta], 2).contiguous(), torch.cat([a_log, a_log]).contiguous(),
            torch.cat([dt_bias, dt_bias]).contiguous(), idx, torch.cat([state, state], 1).contiguous(),
            torch.cat([gate, gate], 2).contiguous(), norm_w]


def run(op, a):
    x, weight, conv_state, raw_g, raw_beta, a_log, dt_bias, idx, state, gate, norm_w = [t.clone() for t in a]
    B, H = raw_g.shape[1], raw_g.shape[2]
    out = torch.empty(1, B, H, D, device=dev, dtype=torch.bfloat16)
    op(x, weight, None, conv_state, raw_g, raw_beta, a_log, dt_bias, idx, state, out, -5.0, gate, norm_w, 1e-5)
    return out, state, conv_state


for B in (1, 4, 8, 16):
    a6 = make(B, 6)
    o6, s6, c6 = run(torch.ops.k3kda.fused_kda_decode, a6)
    o12, s12, c12 = run(torch.ops._C.fused_kda_decode, to12(a6))
    idx = a6[7].long()
    errs = {
        "out": (o6.float() - o12[:, :, :6].float()).abs().max().item(),
        "state": (s6[idx] - s12[idx][:, :6]).abs().max().item(),
        "conv": (c6[idx].float() - torch.cat([s[:, :6 * D] for s in c12[idx].split(12 * D, 1)], 1).float()).abs().max().item(),
    }
    print(f"B={B:2}: max |diff| vs native-12 " + " ".join(f"{k}={v:.2e}" for k, v in errs.items()))


def bench(op, a, iters=64):
    def body():
        for _ in range(iters):
            run(op, a)
    torch.cuda.synchronize()
    x, weight, conv_state, raw_g, raw_beta, a_log, dt_bias, idx, state, gate, norm_w = a
    out = torch.empty(1, raw_g.shape[1], raw_g.shape[2], D, device=dev, dtype=torch.bfloat16)
    def call():
        for _ in range(iters):
            op(x, weight, None, conv_state, raw_g, raw_beta, a_log, dt_bias, idx, state, out, -5.0, gate, norm_w, 1e-5)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        call()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    g.replay(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(10):
        g.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000 / (10 * iters)


for B in (1, 8):
    print(f"B={B}: fused KDA decode H=6 {bench(torch.ops.k3kda.fused_kda_decode, make(B, 6)):.2f} us"
          f"  (native H=12 {bench(torch.ops._C.fused_kda_decode, make(B, 12)):.2f} us)")
