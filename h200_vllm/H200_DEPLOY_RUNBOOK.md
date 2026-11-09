# H200 容器部署与运行手册

本文说明如何把 `GBSA_a5_h200_TEST` 部署到 H200 容器、运行 vLLM 兼容的 GBSA 等效 benchmark，并回收结果。

> 当前脚本是 **vLLM 自定义 attention backend 的算子级 benchmark**，不是完整 Step5 在线服务。它用受控 top-k ids 复现 GBSA 的输入语义；真实 Step5 推理仍需 IndexR 实际 ids 和完整的模型/后端接入。

## 1. 把代码放进 H200 容器

容器可联网时，直接 clone：

```bash
docker exec -it <容器名> bash

mkdir -p /workspace
cd /workspace
git clone https://github.com/JiangHongtao1019/GBSA_a5_h200_TEST.git
cd GBSA_a5_h200_TEST
```

容器不能联网时，从宿主机拷入：

```bash
docker cp GBSA_a5_h200_TEST <容器名>:/workspace/GBSA_a5_h200_TEST
docker exec -it <容器名> bash
cd /workspace/GBSA_a5_h200_TEST
```

如果是新建容器，使用支持 H200 的 vLLM 镜像：

```bash
docker run --gpus all --ipc=host --shm-size=16g -it \
  --name gbsa-h200 \
  -v /path/to/GBSA_a5_h200_TEST:/workspace/GBSA_a5_h200_TEST \
  <你的 H200 vLLM 镜像> bash
```
## 2. 确认容器环境

```bash
nvidia-smi
python -c "import torch, vllm; print(torch.__version__, torch.version.cuda, vllm.__version__)"
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

应看到 H200 和 `sm_90`。

如果容器已有 vLLM，不要直接安装 `requirements.txt`，避免把已有 torch/vLLM 降级。只有缺依赖时才安装：

```bash
python -m pip install -r h200_vllm/requirements.txt
```

## 3. 先做自检

```bash
cd /workspace/GBSA_a5_h200_TEST
python h200_vllm/run_bench.py --check-only --require-vllm
```

该步骤只跑 B=1、W=1k 的一次 forward，确认 CUDA、vLLM、BF16 SDPA 的 `enable_gqa` 路径和 region/page 映射正常。

如果提示 `enable_gqa` 不支持，说明 PyTorch 低于 2.5，需要换匹配的 H200 环境。

## 4. 跑一个小 smoke test

```bash
python h200_vllm/run_bench.py \
  --output-dir outputs_h200_vllm/smoke \
  --batches 1 \
  --scenes uniform \
  --ranges 1024 \
  --modes gather_sdpa \
  --seeds 1 \
  --warmup 2 \
  --iters 5 \
  --require-vllm
```

应生成：

```text
outputs_h200_vllm/smoke/
  raw_runs.csv
  raw_timings.csv
  summary_by_scene_range_batch.csv
  manifest.json
  environment.json
```
## 5. 跑完整 benchmark

```bash
bash h200_vllm/run_in_container.sh outputs_h200_vllm/full_12seed
```

等价命令：

```bash
python h200_vllm/run_bench.py \
  --output-dir outputs_h200_vllm/full_12seed \
  --batches 1,8,32,128,256 \
  --scenes recent_contiguous,multi_sink,uniform \
  --ranges 1024,4096,8192,16384 \
  --modes gather_sdpa,sdpa_only \
  --seeds 12 \
  --warmup 10 \
  --iters 50 \
  --kv-mode shared \
  --require-vllm
```

默认口径：

- `gather_sdpa`：gather 选中 K/V + FlashAttention，作为 GBSA 主对标数据；
- `sdpa_only`：只做 attention，作为计算控制组；
- `--kv-mode shared`：与 Ascend 的 batch-level working-set 口径一致。

显存紧张时保持 `shared`，不要加 `per_request`。`per_request` 在 B=128 时 KV 约 48 GiB，B=256 约 96 GiB。

## 6. 验证结果

```bash
python h200_vllm/validate_results.py outputs_h200_vllm/full_12seed
```

正常会输出 `PASS`，并给出 run 数、timed sample 数、每 run 迭代数和中位数一致性检查。
## 7. 和 Ascend 结果对照

仓库已包含 Ascend 汇总数据：

```bash
python h200_vllm/compare_with_ascend.py \
  --ascend-dir results \
  --h200-dir outputs_h200_vllm/full_12seed
```

生成：

```text
outputs_h200_vllm/full_12seed/comparison_vs_ascend/
  h200_vs_ascend.csv
  h200_vs_ascend_B128.md
```

重点看 B=128、`gather_sdpa` 下的：

- `latency_median_ms`
- `union_pages_median`
- `working_set_mib_median`
- `speedup_x_vs_w1k`
- `delta_pct_vs_w1k`

## 8. 快速查看 B=128 主结果

```bash
python - <<'PY'
import pandas as pd
p = "outputs_h200_vllm/full_12seed/summary_by_scene_range_batch.csv"
df = pd.read_csv(p)
cols = [
    "scene", "range_regions_requested", "latency_median_ms",
    "union_pages_median", "working_set_mib_median",
    "speedup_x_vs_w1k", "delta_pct_vs_w1k",
]
print(df[(df.batch_size == 128) & (df.timing_mode == "gather_sdpa")][cols].to_string(index=False))
PY
```
## 9. 判断是否复现 Ascend 的波动

重点看两组：

1. `gather_sdpa` 在 `recent_contiguous` / `uniform` 上是否随候选范围上升；
2. `multi_sink` 是否基本平直。

如果 `sdpa_only` 基本平直，而 `gather_sdpa` 出现同样的场景差异，说明 H200 上也是页 fan-out / KV 工作集在驱动波动，而不是 attention 计算量。

如果 `gather_sdpa` 也不明显，说明 H200 的 HBM 带宽、缓存和 gather 实现把这个问题掩盖了。

## 10. 把结果拷出容器

```bash
docker cp <容器名>:/workspace/GBSA_a5_h200_TEST/outputs_h200_vllm/full_12seed ./h200_full_12seed
```

主要文件：

- `summary_by_scene_range_batch.csv`：聚合结果；
- `raw_runs.csv`：每个 run 的中位数和页指标；
- `raw_timings.csv`：每个 timed sample；
- `manifest.json`：run 数、样本数和 SHA256；
- `environment.json`：GPU、torch、vLLM 环境。
## 11. 常见问题

**CUDA 不可用**

- 容器启动时是否加了 `--gpus all`；
- `nvidia-smi` 是否正常；
- `torch.cuda.is_available()` 是否为 True。

**提示 vLLM 不可用**

- 先 `python -c "import vllm; print(vllm.__version__)"`；
- 确认当前 shell 在正确的 venv/conda 环境中；
- 如果暂时不想强制 vLLM，可去掉 `--require-vllm`，脚本会走纯 PyTorch 参考路径，但这就不是 vLLM 环境下的结果。

**显存不足**

- 保持 `--kv-mode shared`；
- 减少 `--batches`，例如先跑 `1,8,32,128`；
- 不要同时跑 `per_request` 和 B=256。

**结果和 Ascend 不完全同趋势**

这是可能的。H200 的 HBM 带宽、缓存层次、gather 实现和 FlashAttention 版本都会改变斜率；重点看 `recent/uniform` 是否随 union pages 上升、`multi_sink` 是否平直。

## 12. 重要边界

- `topk_ids` 是受控 IndexR-like 构造，不是实际 IndexR 输出；
- KV 是确定性结构化生成的 BF16 cache，不是真实模型推理缓存；
- `gather_sdpa` 是 PyTorch gather + FlashAttention 组合，不是单融合 GBSA kernel；
- 要做真实 Step5 推理对比，需要 dump Ascend 服务的实际 topk ids，并替换脚本中的受控 ids。