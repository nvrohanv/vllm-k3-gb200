"""Summarize vllm bench results against the Inferact blog's GB200 and TPU numbers.

Usage: python3 summarize.py results/<run-id>
Decode-only aggregate = B * 1000 / mean TPOT (ms): the GPU counterpart of the
TPU harness metric (decode tokens / device decode time, prefill token excluded).
"""

import glob
import json
import os
import re
import statistics
import sys

BLOG_GPU = {1: 127, 2: 227, 4: 373, 8: 636}
BLOG_TPU = {1: 249, 2: 392, 4: 515, 8: 865}

run_dir = sys.argv[1]
rows = {}
for path in sorted(glob.glob(os.path.join(run_dir, "b*_r*.json"))):
    batch = int(re.match(r"b(\d+)_r\d+\.json", os.path.basename(path)).group(1))
    d = json.load(open(path))
    rows.setdefault(batch, []).append({
        "decode_agg": batch * 1000.0 / d["mean_tpot_ms"],
        "output_tput": d["output_throughput"],
        "tpot": d["mean_tpot_ms"],
        "ttft": d["mean_ttft_ms"],
        "p99_itl": d.get("p99_itl_ms", float("nan")),
        "out_tokens": d["total_output_tokens"] / d["completed"],
    })


def fmt(values):
    mean = statistics.mean(values)
    spread = (max(values) - min(values)) / mean * 100 if len(values) > 1 else 0.0
    return mean, spread


print(f"{'B':>2} {'n':>2} {'decode agg tok/s':>17} {'spread':>7} {'blog GPU':>8} {'delta':>7} "
      f"{'e2e out tok/s':>13} {'per-user':>8} {'TPOT ms':>8} {'TTFT ms':>8} {'p99 ITL':>8} {'TPU':>5}")
for batch in sorted(rows):
    r = rows[batch]
    agg, spread = fmt([x["decode_agg"] for x in r])
    tpot = statistics.mean(x["tpot"] for x in r)
    delta = (agg / BLOG_GPU[batch] - 1) * 100 if batch in BLOG_GPU else float("nan")
    print(f"{batch:>2} {len(r):>2} {agg:>17.1f} {spread:>6.1f}% {BLOG_GPU.get(batch, 0):>8} {delta:>+6.1f}% "
          f"{statistics.mean(x['output_tput'] for x in r):>13.1f} {1000 / tpot:>8.1f} {tpot:>8.3f} "
          f"{statistics.mean(x['ttft'] for x in r):>8.1f} {statistics.mean(x['p99_itl'] for x in r):>8.3f} "
          f"{BLOG_TPU.get(batch, 0):>5}")
    if any(abs(x["out_tokens"] - 1024) > 1 for x in r):
        print(f"   warning: B={batch} mean output tokens/request != 1024")
