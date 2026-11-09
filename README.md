# GBSA 128K / top-512 候选范围实验包

本目录是一个可重复执行、可跨机复现的 GBSA 单算子 benchmark，用于回答：

> 在 128K 输入、topk=512 固定的条件下，把 IndexR 的候选范围限制在 1k / 4k / 8k / 16k 个 region 内，GBSA 的耗时如何变化？

## 核心结论

- 在 B=128 时，`recent_contiguous` 场景从 1k 到 16k 的耗时从 **0.3785 ms** 增至 **0.6153 ms（+62.6%）**；`uniform` 从 **0.5315 ms** 增至 **1.0427 ms（+96.2%）**。
- 同样在 B=128、16k 范围下，`multi_sink` 只有 **0.3741 ms**，`uniform` 为 **1.0427 ms**，相差 **2.79×**，尽管二者选中 span 都接近 16k。
- 真正的驱动量是 **per-request physical page fan-out** 和 **batch-level union KV working set**，不是 topk span 本身。详细分析见 `GBSA_128K_TOP512_Range_Report.md`。
- B=1 / B=8 时 launch overhead 占主导，范围效应 ≤ 2.3%，工程上可忽略；B≥32 后效应才明显。

## 目录

| 路径 | 内容 |
|---|---|
| `scripts/run_bench.py` | 完整 benchmark：生成 128K BF16 KV、受控 top-512 ids、计时核心 GBSA、断点续跑 |
| `scripts/run_in_container.sh` | 在容器内一键运行 |
| `scripts/run_from_host.sh` | 从 host 经 ssh + docker 运行并回收结果 |
| `results/` | 本次 720-run 原始数据、配置、环境、汇总、manifest |
| `figures/` | 7 张 PNG 图表（含 top-512 构造示意） |
| `report/` | 报告生成代码与 Markdown 模板 |
| `GBSA_128K_TOP512_Range_Report.md` | 图文报告 |
| `GBSA_128K_TOP512_Range_Report.xlsx` | 原始数据 + 透视汇总 + 条件格式 + 原生 Excel 图 + Dashboard |
| `GBSA_128K_B128_Dashboard.xlsx` | B=128 单页耗时对比透视：KPI、耗时表、主折线图、场景说明与结论；`_clean.xlsx` 为同一文件的备份 |
| `GBSA_128K_B128_Overview.png` | **B=128 一张图**（3000×1650，300 dpi）：12 个耗时值 + 三种 top-512 空间分布 + 物理页足迹，一句话结论；由 `report/b128_overview.py` 生成 |
| `GBSA_128K_Core_Brief_2p.pdf` | **两页核心数据速览**（A4 竖版）：4 个 KPI、B=128 主结果表、两次受控对照、batch 规模表、Spearman ρ、结论；由 `report/build_brief_pdf.py` 生成 |
| `h200_vllm/` | **NVIDIA H200 + vLLM 平行 benchmark**：vLLM AttentionImpl 风格 backend、gather+SDPA / SDPA-only 两种口径、验证和 Ascend 对照脚本；详见 `h200_vllm/README.md` |

## 环境

本次实测环境：

- host: `177.45.22.1` (alias `rack10os2`)
- container: `j00934364_step5_v2`
- NPU: Ascend950DT_9582 × 8，单卡 device 0
- CANN: 9.2.0
- Python: 3.12.13
- torch: 2.10.0+cpu
- torch_npu: 2.10.0.post4
- vllm_ascend: 0.30.0rc2+g293e9084d4c7bba14916b916e4a2c8c4a4d1ac7d
- cann_ops_transformer: 1.0.0
- runtime env: `source /opt/upgrade/runtime-env.sh`
- 运行前建议确认 NPU 没有其他进程；测试已获得授权可清理占卡进程。

## 快速运行

### 容器内

```bash
cd /root/gbsa_range_benchmark
source /opt/upgrade/runtime-env.sh
python3 scripts/run_bench.py --output-dir outputs/full_12seed --seeds 12 --warmup 10 --iters 50
```

或：

```bash
bash scripts/run_in_container.sh outputs/full_12seed
```

快速 smoke：

```bash
python3 scripts/run_bench.py --output-dir outputs/quick --quick
```

断点续跑：

```bash
python3 scripts/run_bench.py --output-dir outputs/full_12seed --seeds 12 --warmup 10 --iters 50 --resume
```

### 从 host

```bash
REMOTE=rack10os2 CONTAINER=j00934364_step5_v2 bash scripts/run_from_host.sh ./results_from_host
```

## 关键配置

- context = 131072 tokens
- region = 8 tokens → 16384 logical regions
- page = 128 tokens → 1024 physical pages
- topk = 512；当前 region 自动追加 → counts = 513
- KV shape = `[1024, 128, 4, 192]`，BF16，K+V 384 MiB
- heads = 64，kv groups = 4，head_dim = 192
- `layout_q="TND"`, `layout_kv="PA_BBND"`, `layout_sparse_pattern=4`, `mask_mode=1`, `quant_mode=0`, `softmax_precision=1`, `is_consistent_topk=False`

## 三种 top-512 空间分布

| scene | 说明 |
|---|---|
| `recent_contiguous` | 每个 `(group, request)` 在候选窗口内选择一个 512-region 连续带；单请求 span=512，但不同请求的带不同 |
| `multi_sink` | 8 个 64-region 热点带横跨窗口；所有请求共享同一组带 |
| `uniform` | 平坦 score，top-512 在窗口内近似均匀分散 |

## 计时口径

主指标只计 `generic_block_sparse_attention` 核心 op 的 NPU event 时间：

- `10` 次 warmup + `50` 次 timed iterations；
- 每个 (batch, scene, range) 跑 `12` 个 selection seed；
- 先取每个 run 50 个样本的 median，再取 12 个 seed median 的 median；
- NPU event 对象预分配复用，每个 run 后 `torch.npu.synchronize()`；
- **不包含** IndexR、metadata、pack/append-current-region、host 调度。

metadata 单独测量结果在 `results/metadata_bench.csv`，B=128 约 0.08 ms。

## 数据与复现

- `results/raw_runs.csv`：720 行 run-level 记录；
- `results/raw_timings.csv`：36,000 行逐次计时；
- `results/environment.json`：含 `npu-smi info`、包版本、CANN 环境；
- `results/config.json`：完整参数；
- `results/manifest.json`：完成时间、run 数与原始文件 SHA-256；
- `results/summary_by_scene_range_batch.csv`：主汇总；
- `results/contrast_w1k_to_w16k.csv`：1k → 16k 对照；
- `results/correlations_by_batch.csv`：Spearman 相关。

重新生成图表、Excel、Markdown（本地安装 pandas / numpy / Pillow / XlsxWriter）：

```bash
python report/build_report.py --data results --out .
```

## 结果自检

不依赖 NPU，验证原始行数、每 run 50 个样本、重算 median、manifest SHA-256：

`ash
python scripts/validate_results.py
`

## 重要局限

1. KV 是 **完整 128K、BF16、Step5 shape、结构化高斯** 的确定性 cache，不是某条真实请求的模型推理中间结果；但形状、大小、dtype、统计量贴近真实，且不是 zeros。
2. topk ids 是 **IndexR-like score + 受控 topk**，不是实际 IndexR 算子输出；这是为了单独测 GBSA、控制变量。
3. 只测核心 op，不测端到端 layer / scheduler / continuous batching。
4. 容器内 `vllm_ascend` import 受 `libcrypto.so.3 OPENSSL_3.3.0` 缺失影响，因此直接调用官方低层 `cann_ops_transformer` wrapper；核心 aclnn op 相同。
5. 单机单卡单 session；换机器/换卡后建议先 `--quick` 再全量。
## H200 + vLLM 平行 benchmark

`h200_vllm/` 提供同一套 128K / top-512 工作负载的 NVIDIA H200 版本，执行入口是 vLLM `AttentionImpl` 风格的自定义 attention backend：

- `gather_sdpa`：按 top-512 region ids gather K/V，再做 SDPA（包含 gather 成本）；
- `sdpa_only`：对预 gather 的 K/V 只做 SDPA（计算参考）。

详见 `h200_vllm/README.md`。这不是 Ascend GBSA 算子的直接移植，也不是完整 vLLM 在线服务压测；H200 侧当前是 vLLM 兼容的参考实现，后续可替换成融合 Triton kernel。