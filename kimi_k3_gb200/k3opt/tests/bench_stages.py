"""Time moe_small FC1-only, FC2-only and both (CUDA graphs, HBM-streaming weights)."""
import sys
import torch
sys.argv = [sys.argv[0]]
exec(open("/work/k3opt/test_moe_small.py").read().split("def rel(")[0])
exec("def graph_us" + open("/work/k3opt/test_moe_small.py").read().split("def graph_us")[1].split("for M in (1, 2, 4, 8):")[0])
for M in (1, 8):
    calls = []
    for _ in range(32):
        x = (torch.randn(M, H, device=dev) * 0.5).to(torch.bfloat16)
        ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
        calls.append((x, ids, torch.softmax(torch.randn(M, TOPK, device=dev), -1)))
    ws = torch.zeros(M * 16 * 96 * 2, device=dev)
    out = torch.empty(M, H, device=dev, dtype=torch.bfloat16)
    res = {}
    for stages in (1, 2, 3):
        k = [0]
        def run():
            x, ids, wts = calls[k[0] % 32]; k[0] += 1
            torch.ops.k3moe.moe_small(x, ids, wts, p13q, p13s, p2q, p2s, ws, out, 4.0, 25.0, stages)
        res[stages] = graph_us(run)
    mb1 = M * 16 * 384 * 1792 / 1e6; mb2 = M * 16 * 3584 * 96 / 1e6
    print(f"M={M}: FC1 {res[1]:6.2f} us ({mb1/res[1]:5.2f} TB/s)  FC2 {res[2]:6.2f} us ({mb2/res[2]:5.2f} TB/s)  both {res[3]:6.2f} us")
