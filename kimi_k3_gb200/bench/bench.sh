#!/usr/bin/env bash
# Runs inside the leader pod:  kubectl exec -i <leader-pod> -- bash -s <out-dir> < bench.sh
# Mirrors tpu-megakernels/scripts/benchmark_kimi.sh: 1 input token, 1024 output
# tokens, ignore EOS, concurrency = B, 4*B requests per run.
set -euo pipefail
OUT="${1:-/tmp/results}"
REPEATS="${REPEATS:-3}"
BATCHES="${BATCHES:-1 2 4 8}"
TOKENIZER=/shared-model-cache/hub/models--moonshotai--Kimi-K3/snapshots/f831ab66814297da540d832a5235f8e904f29d06
common=(
  --backend openai --base-url http://127.0.0.1:8000
  --model moonshotai/Kimi-K3 --tokenizer "${TOKENIZER}" --trust-remote-code
  # vllm bench samples lengths from [X*(1-r), X*(1+r)], so r=0.0 gives the exact
  # 1/1024 lengths the TPU harness gets from its own tool's --random-range-ratio 1.0.
  --dataset-name random --random-input-len 1 --random-output-len 1024 --random-range-ratio 0.0
  --ignore-eos --percentile-metrics ttft,tpot,itl,e2el --save-result --save-detailed
)
# Match the TPU harness flags where this vLLM build supports them.
help="$(vllm bench serve --help=all 2>/dev/null || true)"
for flag in --prompt-token-ids --no-steady-state; do
  grep -q -- "${flag}" <<<"${help}" && common+=("${flag}")
done
grep -q -- --num-warmups <<<"${help}" && common+=(--num-warmups 0)
echo "bench flags: ${common[*]}"

run() {  # run <batch> <num-prompts> <dir> <name>
  mkdir -p "$3"
  vllm bench serve "${common[@]}" --num-prompts "$2" --max-concurrency "$1" \
    --result-dir "$3" --result-filename "$4.json" --label "$4" >"$3/$4.log" 2>&1
  echo "$4: $(grep -E 'Output token throughput|Mean TPOT|Mean TTFT' "$3/$4.log" | tr -s ' ' | paste -sd ';' -)"
}

# Throwaway warm-up so first-run JIT/cudagraph effects don't land in measured runs.
for B in ${BATCHES}; do run "${B}" "${B}" "${OUT}/warmup" "warmup_b${B}"; done
for r in $(seq 1 "${REPEATS}"); do
  for B in ${BATCHES}; do run "${B}" "$((4 * B))" "${OUT}" "b${B}_r${r}"; done
done
