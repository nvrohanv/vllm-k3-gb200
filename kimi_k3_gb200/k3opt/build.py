"""JIT-build the k3opt CUDA extension (cached under TORCH_EXTENSIONS_DIR)."""

import os

import torch
from torch.utils.cpp_extension import load

SRC_DIR = os.environ.get("K3OPT_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc"))
# One extension per kernel file, so editing one kernel only rebuilds that one.
SOURCES = ["kda_decode6.cu", "moe_small.cu", "ar_attn_res.cu", "lamport_attn_res.cu", "k3mla.cu", "kda_split.cu", "l2pf.cu", "k3gemv.cu", "oproj_ar.cu"]


def build(verbose: bool = False, only=None):
    for src in SOURCES:
        if only and src not in only:
            continue
        load(
            name="k3opt_" + src.split(".")[0],
            sources=[os.path.join(SRC_DIR, src)],
            extra_include_paths=[SRC_DIR],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "-std=c++17", "-DUSE_CUDA"],
            extra_cflags=["-O3", "-std=c++17", "-DUSE_CUDA"],
            is_python_module=False,
            verbose=verbose,
        )
    return torch.ops


if __name__ == "__main__":
    build(verbose=True)
    print("k3opt_ext built")
