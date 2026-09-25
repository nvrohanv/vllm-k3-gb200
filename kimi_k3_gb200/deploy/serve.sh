#!/usr/bin/env bash
# Starts vllm serve in a k3-vllm pod from the config dir $CFG (copied in by exp.sh).
# Node rank is derived from the node we run on (NODE_RANKS), not the LWS index, so
# each rank always lands on the same tray and reuses that tray's page cache.
set -euo pipefail
CFG="${CFG:-/work/cfg}"
while IFS='=' read -r key value; do
  [[ -z "${key}" || "${key}" == \#* ]] && continue
  export "${key}=${value}"
done < ${CFG}/env.txt
# Installed-package hygiene: these pods persist across experiments, so undo/apply the only
# vLLM source edit we use (early PDL trigger in the MoE-tail producers) per the config, then
# verify everything else is pristine against pip's RECORD hashes.
if [[ "${K3OPT_PTRIGGER:-0}" == "1" ]]; then bash ${CFG}/ptrigger.sh; else bash ${CFG}/ptrigger.sh --revert; fi
python3 ${CFG}/verify_pristine.py
# Snapshot weight loader plugin (--load-format k3snap).
for plugin in k3snap k3opt; do
  if [[ -f "${CFG}/${plugin}.py" ]]; then
    mkdir -p "/tmp/${plugin}" && cp "${CFG}/${plugin}.py" "/tmp/${plugin}/"
    [[ "${plugin}" == k3opt ]] && cp "${CFG}"/*_patch.py "/tmp/${plugin}/"
    cp "${CFG}/${plugin}_setup.py" "/tmp/${plugin}/setup.py"
    pip install -q --no-deps --no-build-isolation --root-user-action=ignore "/tmp/${plugin}"
  fi
done
# k3opt CUDA extension: stage sources and build once per tray (cached on local NVMe).
if [[ -f ${CFG}/k3opt_build.py ]]; then
  mkdir -p /tmp/k3opt/csrc
  cp -p ${CFG}/*.cu ${CFG}/*.h ${CFG}/*.cuh /tmp/k3opt/csrc/  # -p: unchanged sources keep mtimes -> no rebuild
  cp ${CFG}/k3opt_build.py /tmp/k3opt/build.py
  # A build killed mid-way leaves torch's file lock behind and every later
  # build waits on it forever; this is the only builder in the pod.
  rm -f "${TORCH_EXTENSIONS_DIR:-/root/.cache/torch_extensions}"/*/*/lock
  if env | grep -qE '^K3OPT_(KDA6|ATTN_RES|MOEFUSED|ARRES|TAILATTN|MLA|L2PF|GEMV|KDASPLIT|KDAFB|OPROJ|MOEBLOCK|MOE8|STEP|STEPOV)=1'; then
    K3OPT_SRC=/tmp/k3opt/csrc python3 /tmp/k3opt/build.py 2>&1 | tail -2
  fi
fi
# Experiment-specific source patches (runs in every pod before the server starts).
[[ -s ${CFG}/patch.sh ]] && bash -euo pipefail ${CFG}/patch.sh
MODEL="${MODEL:-/shared-model-cache/hub/models--moonshotai--Kimi-K3/snapshots/f831ab66814297da540d832a5235f8e904f29d06}"
RANK=""
for entry in ${NODE_RANKS}; do
  [[ "${NODE_NAME}" == "${entry%%:*}" ]] && RANK="${entry##*:}"
done
[[ -n "${RANK}" ]] || { echo "node ${NODE_NAME} not in NODE_RANKS=${NODE_RANKS}" >&2; exit 1; }
until HEAD_IP="$(getent hosts "${LWS_LEADER_ADDRESS}" | awk '{print $1; exit}')" && [[ -n "${HEAD_IP}" ]]; do
  sleep 2
done
mapfile -t ARGS < <(grep -v '^[[:space:]]*\(#\|$\)' ${CFG}/args.txt)
[[ "${RANK}" -gt 0 ]] && ARGS+=(--headless)
echo "node ${NODE_NAME} rank ${RANK} master ${HEAD_IP}"
echo "args: ${ARGS[*]}"
env | grep -E '^(VLLM|NCCL|TRTLLM|FLASHINFER|CUDA|TORCH)' | sort
exec vllm serve "${MODEL}" --served-model-name moonshotai/Kimi-K3 \
  --nnodes 4 --node-rank "${RANK}" --master-addr "${HEAD_IP}" "${ARGS[@]}"
