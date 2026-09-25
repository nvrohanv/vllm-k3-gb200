#!/usr/bin/env bash
# Earlier PDL trigger in vLLM's K3 MoE-tail up-projection kernels (Lamport mailbox producers).
#
# Adds `cute.arch.griddepcontrol_launch_dependents()` right after the producer's own
# `griddepcontrol_wait()`:
#   * fused_add_multicast_skinny_gemm.py (M <= 5): after the single wait at kernel entry
#     (before the K loop).
#   * fused_add_multicast_gemm.py (M >= 6): in the TMA-producer warp, after the wait that
#     precedes the first A load. The PTX ISA says the first launch_dependents from any thread
#     of a CTA triggers that CTA, so one warp is enough.
# The original end-of-kernel triggers stay in place; repeating the instruction is a no-op.
#
# Effect: the Lamport consumer (k3tail.lamport_attn_res, or LamportCopy) becomes resident while
# the up-projection is still running. It loads weights and blocks, runs its block-softmax cluster
# exchange, and starts polling, instead of starting only after the producer's last store. The
# consumer's correctness never depended on the trigger point: the mailbox poll is its
# readiness signal. See INTEGRATION.md "Producer early trigger" for the WAR-safety argument. It
# relies on tailattn_patch.py keeping the producer inputs alive (_KEEPALIVE).
#
# Usage: producer_early_trigger.patch.sh [--revert] [TARGET_DIR]
#   TARGET_DIR defaults to the installed vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail.
# Idempotent: already-patched files are left alone. Originals are kept as *.py.k3tail_orig.
# CuTe DSL kernels are JIT-compiled per process, so the change applies from the next process.
# The on-disk CuTe DSL cache ($CUTE_DSL_CACHE_DIR, default /tmp/cute_dsl_python_cache_*) is keyed
# on the generated IR, so stale entries are not reused; delete it anyway if in doubt.
set -euo pipefail

REVERT=0
if [[ "${1:-}" == "--revert" ]]; then REVERT=1; shift; fi
TARGET="${1:-$(python3 -c 'import os, vllm; print(os.path.join(os.path.dirname(vllm.__file__), "models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail"))')}"

python3 - "$TARGET" "$REVERT" <<'PY'
import os, shutil, sys

target, revert = sys.argv[1], sys.argv[2] == "1"
MARK = "# k3tail-early-trigger"
EDITS = {
    "fused_add_multicast_skinny_gemm.py": (
        "        cute.arch.griddepcontrol_wait()\n",
        "        cute.arch.griddepcontrol_wait()\n"
        f"        cute.arch.griddepcontrol_launch_dependents()  {MARK}\n",
    ),
    "fused_add_multicast_gemm.py": (
        "                cute.arch.griddepcontrol_wait()\n\n"
        "                # Supply A to the very same stages after the producer AR has\n",
        "                cute.arch.griddepcontrol_wait()\n"
        f"                cute.arch.griddepcontrol_launch_dependents()  {MARK}\n\n"
        "                # Supply A to the very same stages after the producer AR has\n",
    ),
}
for name, (old, new) in EDITS.items():
    path = os.path.join(target, name)
    orig = path + ".k3tail_orig"
    src = open(path).read()
    if revert:
        if os.path.exists(orig):
            shutil.copy2(orig, path)
            os.remove(orig)
            print(f"[k3tail] reverted {path}")
        elif MARK in src:
            open(path, "w").write(src.replace(new, old))
            print(f"[k3tail] reverted {path} (no backup; removed marker lines)")
        else:
            print(f"[k3tail] {path} not patched")
        continue
    if MARK in src:
        print(f"[k3tail] already patched: {path}")
        continue
    if src.count(old) != 1:
        sys.exit(f"[k3tail] anchor not found exactly once in {path} (found {src.count(old)}); "
                 "vLLM version mismatch -- not patching")
    shutil.copy2(path, orig)
    open(path, "w").write(src.replace(old, new))
    print(f"[k3tail] patched {path}")
PY
find "$TARGET" -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
