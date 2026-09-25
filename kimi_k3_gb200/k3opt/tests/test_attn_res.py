"""Correctness and latency of k3opt.attn_res_cluster vs vLLM's native kernel."""

import itertools
import sys

import torch

sys.path.insert(0, "/work/k3opt")
from build import build  # noqa: E402

import vllm._custom_ops  # noqa: E402,F401  (registers torch.ops._C)

k3 = build(verbose="-v" in sys.argv)
native = torch.ops._C.kimi_k3_attn_res
H, MAXB = 7168, 10
dev = "cuda"
torch.manual_seed(0)


def make(T, nb):
    prefix = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
    delta = torch.randn(T, H, device=dev, dtype=torch.bfloat16) * 0.5
    blocks = torch.randn(T, MAXB, H, device=dev, dtype=torch.bfloat16) * torch.rand(T, MAXB, 1, device=dev, dtype=torch.bfloat16)
    nw = torch.rand(H, device=dev, dtype=torch.bfloat16) + 0.5
    qw = torch.randn(H, device=dev, dtype=torch.bfloat16) * 0.05
    ow = torch.rand(H, device=dev, dtype=torch.bfloat16) + 0.5
    return prefix, delta, blocks, nw, qw, ow


def run(fn, args, T, nb, has_delta, widx, out_norm, eps=1e-5):
    prefix, delta, blocks, nw, qw, ow = [a.clone() for a in args]
    out = torch.empty_like(prefix)
    fn(prefix, delta if has_delta else None, blocks, nw, qw, ow if out_norm else None, out, nb, widx, eps, eps)
    return out, prefix, blocks


worst = 0.0
for T, nb, has_delta, write, out_norm in itertools.product([1, 3, 8, 16], range(0, 9), [False, True], [False, True], [False, True]):
    widx = nb if write else -1
    args = make(T, nb)
    ref = run(native, args, T, nb, has_delta, widx, out_norm)
    got = run(k3.attn_res_cluster, args, T, nb, has_delta, widx, out_norm)
    for name, a, b in zip(("out", "prefix", "blocks"), ref, got):
        err = (a.float() - b.float()).abs().max().item()
        scale = a.float().abs().max().item() + 1e-6
        worst = max(worst, err / scale)
        if err / scale > 2e-2:
            print(f"MISMATCH T={T} nb={nb} delta={has_delta} write={write} norm={out_norm} {name}: {err:.3e} (scale {scale:.2e})")
print(f"correctness: worst relative max-abs error {worst:.2e}")


def bench(fn, T, nb, has_delta, out_norm, iters=64, copies=48):
    # Rotate through independent buffers so each call streams from HBM like the model.
    sets = [make(T, nb) for _ in range(copies)]
    outs = [torch.empty_like(s[0]) for s in sets]
    def body():
        for i in range(iters):
            p, d, b, nw, qw, ow = sets[i % copies]
            fn(p, d if has_delta else None, b, nw, qw, ow if out_norm else None, outs[i % copies], nb, -1, 1e-5, 1e-5)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        body()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(10):
        g.replay()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) * 1000 / (10 * iters)


print(f"{'T':>3} {'nb':>3} {'delta':>5} {'norm':>5} {'native us':>10} {'cluster us':>11}")
for T, nb, has_delta, out_norm in [(1, 8, True, True), (1, 8, False, True), (8, 8, True, True), (8, 8, False, True), (8, 3, True, True), (16, 8, True, True), (1, 1, True, True)]:
    print(f"{T:>3} {nb:>3} {str(has_delta):>5} {str(out_norm):>5} {bench(native, T, nb, has_delta, out_norm):10.2f} {bench(k3.attn_res_cluster, T, nb, has_delta, out_norm):11.2f}")
