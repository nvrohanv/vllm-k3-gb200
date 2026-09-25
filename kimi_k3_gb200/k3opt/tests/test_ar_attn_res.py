"""16-rank test of k3ar.ar_attn_res (fused NVLS all-reduce + AttnRes).

Reference: exact fp32 sum of every rank's partial (all_gather), rounded to bf16,
then vLLM's native kimi_k3_attn_res. Also times the fused op inside a CUDA
graph of back-to-back calls against NCCL all_reduce + native attn_res.
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

sys.path.insert(0, "/work/dt")
rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
local = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
dist.init_process_group("nccl", device_id=torch.device("cuda", local))
import vllm._custom_ops  # noqa: E402,F401  (native attn_res op)
from build import build  # noqa: E402

build()
H, MAXM, MAXB = 7168, 16, 10
dev = torch.device("cuda", local)

mailbox = symm_mem.empty((2, world, MAXM, H), dtype=torch.bfloat16, device=dev)
handle = symm_mem.rendezvous(mailbox, dist.group.WORLD.group_name)
mailbox.view(torch.int16).fill_(-32768)  # bf16 -0.0 sentinels
torch.cuda.synchronize()
dist.barrier()
mc = int(handle.multicast_ptr)
assert mc != 0, "no NVLS multicast"


def log(*a):
    if rank == 0:
        print(*a, flush=True)


def inputs(M, nb, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    blocks = torch.randn(M, MAXB, H, device=dev, dtype=torch.bfloat16, generator=g)
    prefix = torch.randn(M, H, device=dev, dtype=torch.bfloat16, generator=g)
    nw = torch.rand(H, device=dev, dtype=torch.bfloat16, generator=g) + 0.5
    qw = torch.randn(H, device=dev, dtype=torch.bfloat16, generator=g) * 0.05
    ow = torch.rand(H, device=dev, dtype=torch.bfloat16, generator=g) + 0.5
    return blocks, prefix, nw, qw, ow


worst = 0.0
calls = [0]
for M in (1, 3, 8, 16):
    for nb in (0, 1, 4, 8):
        for has_delta in (True, False):
            for widx in (-1, nb):
                blocks, prefix, nw, qw, ow = inputs(M, nb, 1234 + M * 100 + nb)  # same on all ranks
                partial = torch.randn(M, H, device=dev, dtype=torch.bfloat16,
                                      generator=torch.Generator(device=dev).manual_seed(rank * 7 + M)) * 0.3
                allp = [torch.empty_like(partial) for _ in range(world)]
                dist.all_gather(allp, partial)
                attn = torch.stack(allp).float().sum(0).to(torch.bfloat16)
                # Reference via native op.
                rb, rp = blocks.clone(), prefix.clone()
                if not has_delta:
                    rp = attn.clone()
                rout = torch.empty_like(prefix)
                torch.ops._C.kimi_k3_attn_res(rp, attn if has_delta else None, rb, nw, qw, ow, rout, nb, widx, 1e-5, 1e-5)
                # Fused.
                fb, fp = blocks.clone(), prefix.clone()
                fout = torch.empty_like(prefix)
                torch.ops.k3ar.ar_attn_res(partial, mailbox, mc, calls[0] % 2, rank, world, fp, has_delta, fb, nw, qw, ow, fout,
                                           nb, widx, 1e-5, 1e-5)
                calls[0] += 1
                torch.cuda.synchronize()
                for name, a, b in (("out", fout, rout), ("prefix", fp, rp), ("blocks", fb, rb)):
                    err = ((a.float() - b.float()).abs().max() / (b.float().abs().max() + 1e-6)).item()
                    worst = max(worst, err)
                    if err > 2e-2:
                        log(f"MISMATCH M={M} nb={nb} delta={has_delta} widx={widx} {name} rel {err:.3e}")
log(f"correctness: worst relative max error {worst:.2e}")


def graph_us(fn, iters=94):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn(iters)
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn(iters)
    torch.cuda.synchronize()
    dist.barrier()
    g.replay()
    torch.cuda.synchronize()
    dist.barrier()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(5):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / (5 * iters)


for M in (1, 8):
    blocks, prefix, nw, qw, ow = inputs(M, 8, 99)
    partial = torch.randn(M, H, device=dev, dtype=torch.bfloat16) * 0.3
    out = torch.empty_like(prefix)

    def fused(n):
        for i in range(n):
            torch.ops.k3ar.ar_attn_res(partial, mailbox, mc, i % 2, rank, world, prefix, True, blocks, nw, qw,
                                       ow, out, 8, -1, 1e-5, 1e-5)

    t_fused = graph_us(fused)

    def unfused(n):
        for _ in range(n):
            red = partial.clone()
            dist.all_reduce(red)
            torch.ops._C.kimi_k3_attn_res(prefix, red, blocks, nw, qw, ow, out, 8, -1, 1e-5, 1e-5)

    t_unfused = graph_us(unfused)
    log(f"M={M}: fused NVLS AR + attn_res {t_fused:6.2f} us/call   NCCL AR + native attn_res {t_unfused:6.2f} us/call")

# Phase timeline (globaltimer, ns) of one call in steady state, M=1.
trace = torch.zeros(8, dtype=torch.int64, device=dev)
blocks, prefix, nw, qw, ow = inputs(1, 8, 7)
partial = torch.randn(1, H, device=dev, dtype=torch.bfloat16) * 0.3
out = torch.empty_like(prefix)
rows = []
for i in range(40):
    dist.barrier()
    torch.cuda.synchronize()
    torch.ops.k3ar.ar_attn_res(partial, mailbox, mc, i % 2, rank, world, prefix, True, blocks, nw, qw, ow, out,
                               8, -1, 1e-5, 1e-5, trace)
    torch.cuda.synchronize()
    t = trace.cpu().tolist()
    rows.append([(t[k] - t[0]) / 1000 for k in range(1, 5)])
med = [sorted(r[k] for r in rows[5:])[len(rows[5:]) // 2] for k in range(4)]
log(f"M=1 single-call phases (us from start): after_wait {med[0]:.2f}  after_store {med[1]:.2f}  "
    f"all_arrived {med[2]:.2f}  end {med[3]:.2f}   (eager, barrier-aligned; rank 0)")

dist.barrier()
dist.destroy_process_group()
