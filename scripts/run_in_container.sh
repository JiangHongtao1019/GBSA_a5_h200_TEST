#!/usr/bin/env bash
# Run the GBSA range benchmark from inside the target container.
#
# Usage:
#   bash scripts/run_in_container.sh [OUTPUT_DIR]
#
# Env overrides:
#   RUNTIME_ENV=/opt/upgrade/runtime-env.sh
#   PYTHON=python3
#   SEEDS=12 WARMUP=10 ITERS=50
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

OUTPUT_DIR="${1:-outputs/run_$(date +%Y%m%d_%H%M%S)}"
RUNTIME_ENV="${RUNTIME_ENV:-/opt/upgrade/runtime-env.sh}"
PYTHON="${PYTHON:-python3}"
SEEDS="${SEEDS:-12}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-50}"

if [[ -f "${RUNTIME_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${RUNTIME_ENV}"
fi

mkdir -p "$(dirname "${OUTPUT_DIR}")"
echo "[run_in_container] output: ${OUTPUT_DIR}"
"${PYTHON}" scripts/run_bench.py \
  --output-dir "${OUTPUT_DIR}" \
  --seeds "${SEEDS}" \
  --warmup "${WARMUP}" \
  --iters "${ITERS}"