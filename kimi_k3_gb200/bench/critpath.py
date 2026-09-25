"""Attribute decode-step time to kernels by their marginal cost on the main stream.

For consecutive main-stream kernels, cost_i = end_i - end_{i-1}: the time kernel i
adds to the step, net of PDL pre-launch waiting. Aux-stream work only shows up
through the main-stream kernel that joins on it.
Usage: python3 critpath.py <trace.json.gz> [--top N]
"""

import collections
import gzip
import json
import re
import sys

path = sys.argv[1]
top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 30
events = json.load(gzip.open(path, "rt"))["traceEvents"]
kernels = sorted((e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy")),
                 key=lambda e: e["ts"])
main_tid = collections.Counter(e["tid"] for e in kernels).most_common(1)[0][0]
main = [e for e in kernels if e["tid"] == main_tid]
marks = [i for i, e in enumerate(main) if "vocab_parallel_embedding" in e["name"]]


def norm(name):
    name = re.sub(r"^void ", "", name)
    name = re.sub(r"kernel_cutlass_kernel_vllm(model_executorkernelslinearcute_dsl_|modelskimi_k3nvidiaopscute_dsl)", "", name)
    name = re.sub(r"_object_at_.*", "", name)
    name = re.sub(r"<.*", "", name)
    name = re.sub(r"\(.*", "", name)
    return name[:80]


# Keep steady-state decode steps only (prefill / first decode are ~10x longer).
periods = sorted(main[b]["ts"] - main[a]["ts"] for a, b in zip(marks[1:-1], marks[2:]))
median = periods[len(periods) // 2]
cost = collections.defaultdict(float)
count = collections.Counter()
steps = 0
totals = []
for a, b in zip(marks[1:-1], marks[2:]):
    if abs((main[b]["ts"] - main[a]["ts"]) - median) > 0.1 * median:
        continue
    seg = main[a - 1:b]  # include previous step's last kernel as the anchor
    total = 0.0
    for prev, cur in zip(seg, seg[1:]):
        c = (cur["ts"] + cur["dur"]) - (prev["ts"] + prev["dur"])
        cost[norm(cur["name"])] += c
        count[norm(cur["name"])] += 1
        total += c
    totals.append(total)
    steps += 1
step = sum(totals) / steps
print(f"{steps} steps, main stream {main_tid}, step {step:.1f} us")
print(f"{'us/step':>9} {'%':>6} {'n/step':>7} {'us/call':>8}  kernel")
for name, t in sorted(cost.items(), key=lambda kv: -kv[1])[:top]:
    print(f"{t / steps:9.1f} {100 * t / steps / step:5.1f}% {count[name] / steps:7.1f} {t / count[name]:8.2f}  {name}")
