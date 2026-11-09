#!/usr/bin/env bash
# Example host-side driver for the exact benchmark environment:
#   host -> ssh rack10os2 -> docker exec j00934364_step5_v2
#
# Usage:
#   bash scripts/run_from_host.sh [LOCAL_OUTPUT_DIR]
#
# Env overrides:
#   REMOTE=rack10os2
#   CONTAINER=j00934364_step5_v2
#   REMOTE_DIR=/root/gbsa_range_benchmark
#   SEEDS=12 WARMUP=10 ITERS=50
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_OUT="${1:-${ROOT_DIR}/results_from_host_$(date +%Y%m%d_%H%M%S)}"
REMOTE="${REMOTE:-rack10os2}"
CONTAINER="${CONTAINER:-j00934364_step5_v2}"
REMOTE_DIR="${REMOTE_DIR:-/root/gbsa_range_benchmark}"
SEEDS="${SEEDS:-12}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-50}"

echo "[run_from_host] remote=${REMOTE} container=${CONTAINER}"
ssh "${REMOTE}" "mkdir -p '${REMOTE_DIR}/scripts'"
scp "${SCRIPT_DIR}/run_bench.py" "${REMOTE}:/tmp/gbsa_run_bench.py"
ssh "${REMOTE}" "docker cp /tmp/gbsa_run_bench.py '${CONTAINER}:${REMOTE_DIR}/scripts/run_bench.py'"
ssh "${REMOTE}" "docker exec '${CONTAINER}' bash -lc 'cd \"${REMOTE_DIR}\" && source /opt/upgrade/runtime-env.sh && python3 scripts/run_bench.py --output-dir outputs/from_host --seeds ${SEEDS} --warmup ${WARMUP} --iters ${ITERS}'"
ssh "${REMOTE}" "rm -rf /tmp/gbsa_from_host && docker cp '${CONTAINER}:${REMOTE_DIR}/outputs/from_host' /tmp/gbsa_from_host"
rm -rf "${LOCAL_OUT}"
scp -r "${REMOTE}:/tmp/gbsa_from_host" "${LOCAL_OUT}"
echo "[run_from_host] copied to ${LOCAL_OUT}"