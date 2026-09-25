"""Fail server startup if the installed vLLM / FlashInfer differ from their pip RECORD hashes
in any way the current config does not declare (e.g. a source edit left over from an earlier
experiment on these persistent pods)."""
import base64
import glob
import hashlib
import os
import sys

SITE = "/usr/local/lib/python3.12/dist-packages"
ALLOWED = {"build_backend.py"}  # flashinfer_python RECORD mismatch shipped in the image itself
if os.environ.get("K3OPT_PTRIGGER", "0") == "1":
    ALLOWED |= {f"vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/{f}"
                for f in ("fused_add_multicast_gemm.py", "fused_add_multicast_skinny_gemm.py")}
bad = []
for dist in glob.glob(SITE + "/vllm-*.dist-info") + glob.glob(SITE + "/flashinfer*.dist-info"):
    for line in open(os.path.join(dist, "RECORD")):
        rel, digest, _ = (line.rstrip("\n").rsplit(",", 2) + ["", ""])[:3]
        path = os.path.normpath(os.path.join(SITE, rel))
        if not digest.startswith("sha256=") or not path.endswith(".py") or not os.path.exists(path):
            continue
        h = base64.urlsafe_b64encode(hashlib.sha256(open(path, "rb").read()).digest()).rstrip(b"=").decode()
        if "sha256=" + h != digest and rel not in ALLOWED:
            bad.append(rel)
if bad:
    sys.exit(f"[k3opt] undeclared modifications to installed packages: {bad}")
print("[k3opt] installed vLLM/FlashInfer match pip RECORD (declared patches: "
      f"{'ptrigger' if os.environ.get('K3OPT_PTRIGGER', '0') == '1' else 'none'})")
