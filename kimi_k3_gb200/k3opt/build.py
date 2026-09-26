"""JIT-build the k3opt CUDA extension (cached under TORCH_EXTENSIONS_DIR)."""

import os

import torch
from torch.utils.cpp_extension import load

SRC_DIR = os.environ.get("K3OPT_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc"))
# One extension per kernel file, so editing one kernel only rebuilds that one.
SOURCES = ["kda_decode6.cu", "moe_small.cu", "ar_attn_res.cu", "lamport_attn_res.cu", "k3mla.cu", "kda_split.cu", "l2pf.cu", "k3gemv.cu", "oproj_ar.cu", "moe8.cu", "k3samp.cu", "k3step.cu", "k3step_prep.cu", "tail2.cu", "moebody.cu", "moe_front.cu", "oproj_front.cu", "tcstage.cu", "t1.cu"]


# Evict-first audit (l2pf 2026-09-26): K3EF=<comma list> compiles the listed kernel families with their weight
# loads at L2::evict_first (read-once streams). Families: route (moe_small.cu, -DK3EF_ROUTE), mla (k3mla.cu,
# -DK3EF_MLA). Unset / empty = the deployed code, bit for bit. Separate extension names per variant.
_EF_DEFS = {"route": ("moe_small.cu", "-DK3EF_ROUTE"), "mla": ("k3mla.cu", "-DK3EF_MLA")}


def _ef_defs(src: str) -> list:
    fams = {f.strip() for f in os.environ.get("K3EF", "").split(",") if f.strip()}
    return [d for fam, (s, d) in _EF_DEFS.items() if fam in fams and s == src]


def build(verbose: bool = False, only=None):
    for src in SOURCES:
        if only and src not in only:
            continue
        defs = _ef_defs(src)
        load(
            name="k3opt_" + src.split(".")[0] + ("_ef" if defs else ""),
            sources=[os.path.join(SRC_DIR, src)],
            extra_include_paths=[SRC_DIR],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "-std=c++17", "-DUSE_CUDA"] + defs,
            extra_cflags=["-O3", "-std=c++17", "-DUSE_CUDA"],
            is_python_module=False,
            verbose=verbose,
        )
    return torch.ops


if __name__ == "__main__":
    build(verbose=True)
    print("k3opt_ext built")
