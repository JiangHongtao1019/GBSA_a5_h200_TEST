#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
PY
python h200_vllm/run_bench.py \
  --output-dir "${1:-outputs_h200_vllm/full_12seed}" \
  --seeds "${SEEDS:-12}" \
  --warmup "${WARMUP:-10}" \
  --iters "${ITERS:-50}" \
  --require-vllm