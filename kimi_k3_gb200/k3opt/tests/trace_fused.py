"""Phase timeline of moe_fused per CTA (globaltimer)."""
import sys
import torch
sys.argv = [sys.argv[0]]
exec(open("/work/k3opt/test_moe_small.py").read().split("def reference(")[0])
barrier = torch.zeros(8, dtype=torch.int64, device=dev)
for M in (1,):
    x = (torch.randn(M, H, device=dev) * 0.5).to(torch.bfloat16)
    ws = torch.zeros(M * TOPK * I, device=dev, dtype=torch.float16)
    out = torch.empty(M, H, device=dev, dtype=torch.bfloat16)
    trace = torch.zeros(4096 * 8, dtype=torch.int64, device=dev)
    for it in range(4):
        ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
        wts = torch.softmax(torch.randn(M, TOPK, device=dev), -1)
        torch.cuda.synchronize()
        torch.ops.k3moe.moe_fused(x, ids, wts, p13q, p13s, p2q, p2s, ws, barrier, out, 4.0, 25.0, trace)
        torch.cuda.synchronize()
    t = trace.view(-1, 8).cpu()
    t = t[t[:, 0] > 0]
    t0 = t[:, 0].min()
    names = ["start", "after_wait", "staged", "fc1_done", "barrier_out", "end"]
    print(f"M={M}: {len(t)} CTAs")
    for k, nm in enumerate(names):
        col = (t[:, k] - t0).double() / 1000
        print(f"  {nm:12} min {col.min():7.2f}  median {col.median():7.2f}  max {col.max():7.2f} us")
