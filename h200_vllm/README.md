# H200 + vLLM: GBSA 128K / top-512 平行 benchmark

这套脚本把 Ascend GBSA 实验的工作负载、受控 top-512 构造、统计口径和输出结构搬到 NVIDIA H200，并以 vLLM AttentionImpl 风格的自定义 attention backend 作为执行入口。

> 它不是把 Ascend 的 `aclnnGenericBlockSparseAttention` 直接搬到 GPU，也不是完整的 vLLM 在线服务压测。当前 vLLM 没有原生 GBSA region-sparse attention，所以这里提供的是一个可复现的 operator-level 参考实现：
>
> `selected region ids -> gather K/V -> SDPA（H200 上走 FlashAttention）`
>
> 后续可以把 gather+SDPA 替换成融合 Triton kernel，benchmark 的输入、场景和计时边界保持不变。

## 1. 文件说明

| 文件 | 作用 |
|---|---|
| `requirements.txt` | H200 侧依赖，目标 vLLM 0.8.x / PyTorch 2.5 |
| `workload.py` | Step5 形状 KV、三种 top-512 构造、region→token→physical page 映射、页足迹统计 |
| `vllm_backend.py` | `GBSAH200Impl`（vLLM `AttentionImpl.forward` 风格）和 `GBSAH200Backend` 描述符 |
| `run_bench.py` | 12 seeds × 1k/4k/8k/16k × 3 场景 × 5 个 batch 的 CUDA Event 计时 |
| `validate_results.py` | 不依赖 GPU，检查 run 数、样本数、中位数一致性、SHA256 |
| `compare_with_ascend.py` | 与 Ascend `results/summary_by_scene_range_batch.csv` 做工作量级对照 |
| `run_in_container.sh` | 容器内一键运行 |
| `H200_DEPLOY_RUNBOOK.md` | H200 容器部署、自检、smoke、全量运行、结果回收和排障手册 |

## 2. 环境

目标环境：

- NVIDIA H200（sm_90，141 GB HBM3e）
- CUDA 12.4+
- PyTorch 2.5.x
- vLLM 0.8.x
- Triton 3.1+（vLLM 依赖；当前参考路径不强制使用 Triton）

```bash
cd /path/to/gbsa_range_benchmark
python -m pip install -r h200_vllm/requirements.txt
```

若容器已有 vLLM/PyTorch，不要重复安装，先确认：

```bash
python - <<'PY'
import torch, vllm
print(torch.__version__, torch.version.cuda)
print(vllm.__version__)
print(torch.cuda.get_device_name(0))
PY
```

## 3. 快速自检

```bash
python h200_vllm/run_bench.py --check-only
```

构造 B=1、W=1k 的 KV 和 ids，并跑一次 forward，确认 CUDA、BF16 SDPA 的 `enable_gqa` 路径和 region/page 映射正常。

## 4. 完整运行

```bash
bash h200_vllm/run_in_container.sh outputs_h200_vllm/full_12seed
```

等价于：

```bash
python h200_vllm/run_bench.py \
  --output-dir outputs_h200_vllm/full_12seed \
  --batches 1,8,32,128,256 \
  --scenes recent_contiguous,multi_sink,uniform \
  --ranges 1024,4096,8192,16384 \
  --modes gather_sdpa,sdpa_only \
  --seeds 12 --warmup 10 --iters 50 \
  --kv-mode shared \
  --require-vllm
```

默认矩阵：5 个 batch（1/8/32/128/256）× 3 个场景 × 4 个范围 × 12 个 selection seeds × 2 个计时口径，共 1440 个 run、72,000 个 timed samples（另加 warmup）。

## 5. 两种计时口径

`gather_sdpa`：按 region ids gather K/V，再做 SDPA，包含 gather/repack 成本。

`sdpa_only`：只对预 gather 的 K/V 做 SDPA，不包含稀疏访存成本。

不要把 `sdpa_only` 当成 GBSA 的等价实现；它只是计算参考，用于判断瓶颈在 attention 还是 page gather。

## 6. KV 模式

默认 `--kv-mode shared`，与 Ascend 实验的 batch-level working-set 口径一致：所有请求共享同一个 128K KV pool，`block_table` 为 identity mapping。

`--kv-mode per_request` 会为每个请求分配一份完整 128K KV：单请求 K+V 约 384 MiB，B=128 约 48 GiB，B=256 约 96 GiB。H200 141 GB 可以放下，但还会叠加 gather buffer 和 PyTorch allocator 余量，正式跑 B=256 前请先确认显存。

## 7. 输出

```text
outputs_h200_vllm/full_12seed/
  config.json
  environment.json
  raw_runs.csv
  raw_timings.csv
  summary_by_scene_range_batch.csv
  raw_runs_sorted.csv
  manifest.json
```

`summary_by_scene_range_batch.csv` 的关键列：`latency_median_ms`、`latency_q25_ms`/`latency_q75_ms`、`union_pages_median`、`working_set_mib_median`、`speedup_x_vs_w1k`、`delta_pct_vs_w1k`。

## 8. 验证与对照

```bash
python h200_vllm/validate_results.py outputs_h200_vllm/full_12seed
python h200_vllm/compare_with_ascend.py \
  --ascend-dir results \
  --h200-dir outputs_h200_vllm/full_12seed
```

对照结果只是工作量级比较，不是算子等价性证明。Ascend 侧是原生 GBSA，H200 侧是 gather+SDPA 的 vLLM 兼容参考实现，二者 kernel、访存路径和编译优化不同。

## 9. vLLM 集成边界

`vllm_backend.py` 中的 `GBSAH200Impl.forward` 使用与 vLLM `AttentionImpl` 相同的调用形状：

```python
forward(layer, query, key, value, kv_cache, attn_metadata, output=None)
```

当前 benchmark 直接驱动这个 forward，并把 K/V 和 top-k token indices 放进 `GBSAMetadata`。好处是计时边界干净，不受 vLLM scheduler、sampling、detokenization 干扰。

限制：这不是完整 `LLM` 引擎路径。要接入完整 vLLM 服务，还需要：

1. 在模型侧产出 IndexR-like top-k region ids；
2. 实现版本匹配的 `AttentionMetadataBuilder`；
3. 把 ids 写入 backend metadata，并处理 CUDA Graph / 多步 decode 的静态 shape；
4. 用 `GBSAH200Backend` 替换 vLLM 的 paged attention backend。

`register_backend()` 提供 best-effort 注册，但不同 vLLM 小版本的 backend registry 接口可能变化，完整引擎接入前需要按实际 vLLM 版本适配。

## 10. 结果解释注意事项

- `topk_ids` 是受控 IndexR-like 构造，不是实际 IndexR 输出；
- KV 是确定性结构化生成的 BF16 cache，不是真实模型推理缓存；
- `gather_sdpa` 的 gather 是显式 PyTorch index/gather，不代表 Ascend GBSA 的片上稀疏访存；
- 不要把 H200 绝对延迟直接换算成 Ascend 的加速比；应分别报告 `gather_sdpa` 和 `sdpa_only`，并注明计时边界。

正式 H200 运行时建议加 `--require-vllm`，确保脚本不会在缺失 vLLM 的机器上静默走纯 PyTorch fallback。
## 11. 与 Ascend GBSA 语义对齐

- `topk=512` 是历史 region 数；GBSA 还会追加当前 region，因此 attention 实际读取 `513` 个 region、`4104` 个 token；
- 候选窗口取最近 `min(W, current_region)` 个历史 region，`current_region = 16383`；
- `block_table` 使用固定随机 permutation，并重复到每个 request，与 Ascend 测试一致；
- `selection_metrics` 只统计 512 个历史 ids，和 Ascend 的页足迹口径一致；实际 gather/attention 使用 513 个 ids。