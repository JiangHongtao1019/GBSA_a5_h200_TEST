# GBSA 单算子测试：128K KV / top-512 / 候选范围 1k–16k regions

> **结论先行**：在 B=128、topk=512 固定不变时，把候选范围从 1k 扩大到 16k，GBSA 核心算子中位耗时可能从 **@@R128_1K@@ ms 增至 @@R128_16K@@ ms（+@@R128_16_PCT@@%）**；但如果 512 个 topk 始终落在跨请求共享的多 sink 区域，则耗时几乎不变（**@@M128_1K@@ → @@M128_16K@@ ms，@@M128_16_PCT@@%**）。
>
> 真正的驱动量不是「选中区域的时间跨度」，而是 **每个请求实际触及的物理页数（request-level page fan-out）** 与 **整个 batch 的物理页并集工作集（batch-level union working set）**。候选范围只是放大了这两个量的上限。

- 测试环境：`177.45.22.1` / `j00934364_step5_v2` / Ascend950DT_9582 × 8 / CANN 9.2.0 / torch 2.10.0 / torch_npu 2.10.0.post4。
- 测试对象：`aclnnGenericBlockSparseAttention` 核心算子；计时范围 **不包含** IndexR、metadata、Triton pack/append。
- 数据规模：完整 128K BF16 KV cache，形状 `[1024, 128, 4, 192]`，K+V 共 384 MiB；topk=512，region=8 tokens，page=128 tokens。
- 数据矩阵：5 个 batch × 3 种空间分布 × 4 个候选范围 × 12 个 selection seed × 50 次 NPU-event 计时 = **720 个 run / 36,000 个计时样本**；所有输出均有限，@@FINITE@@/720 校验通过。
- 复现：`scripts/run_bench.py`；结果与图表：`results/` + `figures/`；Excel：`GBSA_128K_TOP512_Range_Report.xlsx`。
---

## 1. 问题定义

线上场景是 128K context 的 decode / GBSA：

- context `S = 131072` tokens；
- region `R = 8` tokens，因此逻辑 region 总数 `16384`；
- topk `512` 个历史 region，当前 region 由 GBSA 自动追加，所以每个请求实际 `counts = 513`；
- IndexR 在 128K 范围（16384 个 region）中选出 top-512，候选范围越大，top-512 的空间离散程度可能越大。

本测试只改变 **topk 的候选范围 W**，保持 topk=512、counts=513、KV cache 形状、page size、block table、dtype 不变；只改变候选窗口内 top-512 的空间分布。

| 候选范围 W（regions） | 对应 token 范围 | 说明 |
|---:|---:|---|
| 1k | 8K | 最近 1024 个 region 中选 512 |
| 4k | 32K | 最近 4096 个 region 中选 512 |
| 8k | 64K | 最近 8192 个 region 中选 512 |
| 16k | 128K | 最近 16383 个历史 region（当前 region 不计入候选），约 128K |

> 16k 请求实际可用历史 region 是 `16383`（当前 region `16383` 由算子追加），因此表中 effective candidate 为 16383；这是 128K 下全范围的真实上限。

---

## 2. 测试对象与计时边界

### 2.1 调用栈

使用容器内已安装的官方低层算子 wrapper：

```python
from cann_ops_transformer.ops.attention.generic_block_sparse_attention import (
    generic_block_sparse_attention,
    generic_block_sparse_attention_metadata,
)
```

核心调用参数：

```python
out, lse = generic_block_sparse_attention(
    q, k, v, wide, counts, [1, region_size],
    metadata=metadata,
    cu_seqlens_q=cu_seqlens_q,
    seqused_kv=seqused_kv,
    block_table=block_table,
    layout_q="TND",
    layout_kv="PA_BBND",
    layout_sparse_pattern=4,
    softmax_scale=head_dim ** -0.5,
    mask_mode=1,
    quant_mode=0,
    softmax_precision=1,
    return_softmax_lse=False,
    is_consistent_topk=False,
)
```

其中：

- `q`: `[B, 64, 192]` BF16；
- `k`, `v`: `[1024, 128, 4, 192]` BF16；
- `wide`: `[4, B, 513]` int32，前 512 个为 topk region id，最后一个为当前 region `16383`；
- `counts`: `[4, B]` int32，全部为 513；
- `block_table`: `[B, 1024]` int32，固定随机 page 映射；
- `is_consistent_topk=False`（与当前 Step5 配置一致）。

### 2.2 计时范围

**计入**：`generic_block_sparse_attention` 核心 aclnn 调用的设备执行时间。

**不计入**：IndexR / region score / topk 选择；metadata 生成（单独测量，见 §6.7）；topk id 的 append-current-region / Triton pack；Python / host 调度开销。

因此下文所有数字是 **GBSA 单算子核心耗时**，不是完整 attention layer、也不是端到端 decode step。
---

## 3. 128K KV cache 的生成方式

### 3.1 形状与精度

| 项目 | 值 |
|---|---|
| context tokens | 131,072 |
| page size | 128 tokens |
| pages | 1,024 |
| KV groups | 4 |
| head dim | 192 |
| K shape | `[1024, 128, 4, 192]` |
| V shape | `[1024, 128, 4, 192]` |
| dtype | BF16 |
| K + V bytes | 384 MiB |
| region size | 8 tokens |
| regions / page | 16 |
| total regions | 16,384 |

### 3.2 数值生成

使用 **确定性结构化高斯**：

- 每个 token 的 K/V 元素 = 独立高斯噪声 + 所属 region 的共享 latent 向量；
- latent std = 0.35，noise std = 1.0；
- K/V 元素整体 RMS 接近 1，接近 post-RMSNorm/RoPE 后 K/V 的量级；
- 使用 `numpy.random.default_rng` + CPU 生成，固定 `kv_seed=20261105`，可跨机重复；
- K 和 V 使用不同 seed（`kv_seed` / `kv_seed+1`）。

> **重要说明（局限）**：这不是某一条真实请求经过完整 Step5 前向产生的 K/V 中间结果，而是**形状、精度、大小、统计量贴近真实、完整 128K、确定性可复现**的 BF16 KV cache。这样做的目的是控制变量、避免模型推理随机性，并让其他机器也能复现同一输入。它不是 zeros / 占位 buffer。

### 3.3 物理页映射

- `block_table` 用固定 seed `20261106` 生成一个 0..1023 的随机排列；
- 所有请求共享同一 mapping（batch 内重复），保证不同场景之间的 page 映射公平；
- 逻辑 region `r` 对应逻辑 page `r // 16`，物理 page 为 `block_table[logical_page]`。

---

## 4. 受控 top-512 选择场景

为了把候选范围和空间离散度解耦，使用确定性 IndexR-like relevance score：

1. 在每个请求、每个 KV group 内生成一个候选窗口内的 score；
2. 按场景叠加矩形 pulse；
3. 在候选窗口内取 `argpartition` top-512；
4. 排序、去重、映射为绝对 region id。

> 这不是实际调用 IndexR 算子得到的 ids，而是 **IndexR-like score + 受控 topk**。本测试目标是 GBSA 单算子，因此把 IndexR 的延迟排除在外，同时保留 top-512 within candidate range 的语义。真实线上 ids 可以直接替换本脚本的 `make_scene_ids()`，不影响计时框架。

| 场景 | 生成方式 | request 内形态 | batch 并集形态 |
|---|---|---|---|
| `recent_contiguous` | 每个 `(group, request)` 在候选窗口内随机选一个 512-region 连续带，score +12 | 512 连续 region，span=512 | 各请求的带不同，batch 并集随 W 增大而铺满窗口 |
| `multi_sink` | 8 个 64-region 带，横跨候选窗口；所有请求共享同一组带 | span≈W，8 个热点 | 只有约 40 个物理页，跨请求高度复用 |
| `uniform` | 平坦 score，top-512 近似均匀分布 | span≈W，分散 | 几乎覆盖整个候选窗口的物理页 |

> 三个场景的 topk 都是 512，`counts` 都是 513；区别只在 **空间分布**。

### 4.1 同一 topk=512，三种选择模式长什么样

![fig7](figures/fig7_scene_topk_patterns.png)

图中横轴是候选窗口内的 logical region 位置（示意图为 `W=16k regions`），每一行代表一个请求（`group 0`，其余 3 个 group 规则相同）。三个场景的构造细节如下：

| 场景 | score 构造 | 512 个选中 region 的形态 | 对 GBSA 的影响 |
|---|---|---|---|
| `recent_contiguous` | 高斯 score；每个 `(group, request)` 随机选一个起点，在 `[start, start+512)` 上叠加 `+12` pulse | 每个请求是一段 **512 连续 region**；不同请求的带位置不同 | 单请求约 32 页，但 batch 并集随 W 增大而增大 |
| `multi_sink` | 高斯 score；在窗口内均匀放 8 个 `64-region` 带（`8×64=512`），每个带叠加 `+12` pulse；所有请求使用同一组带 | 每个请求都是相同的 **8 段 64-region 热点带** | 跨请求高度复用，并集始终约 39 页 |
| `uniform` | 不加 pulse，直接对平坦的高斯 score 取 top-512 | 512 个 region 在窗口内 **近似均匀分散** | 单请求页数和 batch 并集都接近窗口上限 |

关键点不是「选了多少 token」，三者都严格是 512；关键是这 512 个 id 映射到多少 **物理页**，以及这些页能否在 batch 内复用。`512` 个连续 region 约等于 `32` 页；而 16k 窗口内分散的 512 个 region 可能覆盖约 `408` 页/请求。

> 该图是受控 IndexR-like ids 示意，不是实际 IndexR 算子输出；构造代码位于 `scripts/run_bench.py::make_scene_ids()`。
---

## 5. 计时方法

- 每个 run：`10` 次 warmup + `50` 次 NPU-event 计时；
- 每个场景/范围/batch：`12` 个 selection seed；
- 主指标：先对每个 run 的 50 个样本取 median，再对 12 个 seed 的 run median 取 median；
- 误差带：12 个 run median 的 IQR（图 1 / 图 2）；
- 事件对象预分配并复用，避免高频创建 `torch.npu.Event` 造成设备错误；
- 每次 run 后 `torch.npu.synchronize()`，确保记录完整；
- 输出 `out` 做 `isfinite` 校验，@@FINITE@@/720 全部通过，`out_abs_mean` 中位数约 `@@OUT_ABS_MED@@`。

> 计时只包含核心算子执行本身；host 侧调用、id 生成、metadata 构造不在计时区间内。

---

## 6. 结果

### 6.1 主结果：B=128

![fig1](figures/fig1_latency_vs_range_B128.png)

**表 1：B=128 核心算子中位耗时与 page footprint（12 seeds 中位）**

@@TABLE_B128@@

关键观察：

- `recent_contiguous`：每个请求只选 512 个连续 region（per-request pages 约 33，**不随 W 变**），但 16k 时 batch 并集 page 数从 65 增至 1022，耗时从 **@@R128_1K@@ ms 增至 @@R128_16K@@ ms（+@@R128_16_PCT@@%）**。
- `multi_sink`：选中 span 从 1024 增至 16383，但 batch 并集始终只有约 40 个物理页；耗时 **@@M128_1K@@ → @@M128_16K@@ ms（@@M128_16_PCT@@%）**，基本不变。
- `uniform`：per-request pages 与 batch 并集同时增大；耗时 **@@U128_1K@@ → @@U128_16K@@ ms（+@@U128_16_PCT@@%）**。

### 6.2 同一个 16k span，不同 latency

在 B=128、W=16k 时：

| 场景 | 选中 span (regions) | 每请求页数 | batch 并集页数 | 中位耗时 (ms) |
|---|---:|---:|---:|---:|
| multi_sink | @@SPAN_M@@ | @@PERREQ_M@@ | @@UNION_M@@ | @@M128_16K@@ |
| uniform | @@SPAN_U@@ | @@PERREQ_U@@ | @@UNION_U@@ | @@U128_16K@@ |

两者的选中 span 都接近 16k，但 `uniform` 比 `multi_sink` 慢 **@@U_M_RATIO@@×**。这直接说明：**候选范围/选中 span 本身不是充分统计量**。

### 6.3 同一 batch 并集页数，不同 per-request fan-out

B=128、W=8k 时：

| 场景 | 每请求页数 | batch 并集页数 | 中位耗时 (ms) |
|---|---:|---:|---:|
| recent_contiguous | @@PERREQ_R8@@ | @@SAME_UNION_R@@ | @@LAT_R8@@ |
| uniform | @@PERREQ_U8@@ | @@SAME_UNION_U@@ | @@LAT_U8@@ |

batch 并集几乎相同（@@SAME_UNION_R@@ vs @@SAME_UNION_U@@ pages），但 `uniform` 的 per-request 页数高约 10×，耗时高 **@@LAT_U8_DIFF_PCT@@%**。所以 request 内的 page fan-out 也是独立成本项。

同一 W=1k 下也有相同对照：

- `recent_contiguous`: per-request pages @@PERREQ_BASE_R@@, union @@BASE_UNION_R@@, latency @@BASELINE_R@@ ms；
- `uniform`: per-request pages @@PERREQ_BASE_U@@, union @@BASE_UNION_U@@, latency @@BASELINE_U@@ ms；
- union 相同，per-request pages 约 2×，耗时高 **@@BASELINE_U_DIFF_PCT@@%**。

### 6.4 batch size 的影响

![fig2](figures/fig2_latency_vs_range_facets.png)

**表 2：各 batch 下 W=1k → W=16k 的中位耗时**

@@TABLE_BATCH@@

- B=1 / B=8：launch overhead 主导，候选范围影响 ≤ 2.3%，工程上可视为无差异；
- B=32 起：range 效应显现，`uniform` 约 +@@B32_U_PCT@@%，`recent_contiguous` 约 +@@B32_R_PCT@@%；
- B=128：`uniform` +@@B128_U_PCT@@%，`recent_contiguous` +@@B128_R_PCT@@%；
- B=256：`uniform` +@@B256_U_PCT@@%，`recent_contiguous` +@@B256_R_PCT@@%；
- `multi_sink` 在所有 batch 下都接近 0%，说明只要跨请求并集小，大 batch 也不会因范围大而变慢。

**大 batch 对比表（B=32/128/256）**

@@TABLE_CONTRAST@@
### 6.5 机制拆解：per-request fan-out vs batch union

![fig3](figures/fig3_page_footprint_mechanism.png)

- 左图：`uniform` 的 per-request pages 随 W 快速增大（64 → 408），而 `recent_contiguous` 几乎恒定（33），`multi_sink` 恒定（40）。
- 右图：`recent_contiguous` 的 batch 并集随 W 从约 65 增至 1022 页，即使每个请求只碰 33 页；原因是每个请求选的是不同的连续带。
- `multi_sink` 的并集始终只有 40 页，因为所有请求共享同样的 8 个 sink 带。

![fig4](figures/fig4_latency_vs_union_pages.png)

下图把 B=128 的 144 个 run 全部画在 batch 并集页数 vs latency 平面上：union pages 与 latency 的 Spearman ρ = **@@RHO128@@**。

**表 3：batch 级 Spearman 相关（all scenes）**

@@TABLE_CORR@@

> 注意：相关性用于描述关联，不单独证明因果；真正的因果证据来自上面的三个受控对照（同 span、同 union、同 per-request pages 的交叉比较）。

### 6.6 16k/1k 延迟比总览

![fig5](figures/fig5_speedup_heatmap.png)

![fig6](figures/fig6_w1k_vs_w16k_bars.png)

> 延迟比 = median(W=16k) / median(W=1k)；大于 1 表示 16k 更慢。

- `multi_sink` 在所有 batch 下都是绿色（约 1.00×），即范围大但热点集中不会拖慢 GBSA。
- `recent_contiguous` 在 B≥32 后明显变红（约 1.57–1.64×），尽管单请求 span 始终只有 512。
- `uniform` 最红（约 1.91–1.97×），因为 request 内 fan-out 和 batch union 同时增大。

### 6.7 metadata 单独测量

`generic_block_sparse_attention_metadata` 不在主计时内，但单独测了：

| Batch | metadata median (ms) |
|---:|---:|
| 1 | @@META_1@@ |
| 8 | @@META_8@@ |
| 32 | @@META_32@@ |
| 128 | @@META_128@@ |
| 256 | @@META_256@@ |

metadata 只依赖 shape/counts，不依赖具体 topk ids，因此可以被缓存；若每步都重新生成，它会额外增加约 **@@META_128@@ ms** 的固定成本（B=128 时与核心算子 0.37–1.04 ms 同量级，不能忽略）。

---

## 7. 解释

一个合理的机制解释是：

1. **per-request page fan-out**：`uniform` 的 512 个 region 在逻辑页上高度分散，单请求就要触碰 64–408 个物理页；`recent_contiguous` 只触碰约 33 页。页级 gather / TLB / DMA 事务数增加。
2. **batch-level union working set**：即使单请求只碰 33 页，128 个请求各选不同的 512-region 带时，并集可覆盖 1022/1024 页，即接近整个 384 MiB KV cache；L2 复用下降，HBM 带宽压力上升。
3. **launch overhead 的掩盖**：B=1/8 时核心算子只有约 0.074–0.084 ms，主要是 dispatch/固定开销，page 工作集差异被掩盖；B≥32 后计算/访存占比上升，差异才显现。
4. **跨请求复用是关键**：`multi_sink` 的 40 页被 128 个请求复用，因此即使 span=16k，latency 仍与 1k 相同。这说明共享 sink / 热点对 GBSA 非常友好。

---

## 8. 对 IndexR / 线上选择的启示

1. **不要只用候选范围预测 GBSA 耗时**。同样 16k 范围，`multi_sink` 与 `uniform` 可以差 **@@U_M_RATIO@@×**。
2. **建议监控/限制 batch 级 union page 数**。可以从 `block_table` + 每步 topk ids 直接算：`unique(block_table[id // 16])`。它比 topk span 更接近真实 GBSA 成本。
3. **限制范围要配合限制分散度**：只把候选窗口缩到 1k 并不自动等于低耗时；若 1k 内仍均匀分散，per-request pages 仍可能到 64 页（本测试中 uniform W=1k 比 recent W=1k 慢约 @@BASELINE_U_DIFF_PCT@@%）。反过来，16k 范围但选择共享 sink，耗时可保持最低。
4. **跨请求聚类/批调度**：把选择模式相近的请求放在同一 batch，或把 hotspot/sink 页做共享缓存，可显著降低 batch union；这在 B≥32 时收益最大。
5. **物理页分配与 block table 布局**：在 page 粒度做连续分配、hot page 预留、避免同一请求跨过多 HBM channel/page，都可能降低 per-request fan-out 成本。本测试固定了随机 block table；真实分配策略的影响值得继续测。
6. **对大 batch 做自适应 chunk**：当 `batch_union_pages` 或估算 working set 超过阈值时，把 batch 拆小可以减少 cache thrash；但会牺牲吞吐，需要结合端到端吞吐评估。
---

## 9. 局限与边界

1. **KV 不是模型推理中间结果**：是完整 128K、BF16、Step5 shape、结构化高斯的真实尺寸 cache；统计量接近真实，但不是某条真实请求的模型激活。
2. **ids 不是实际 IndexR 输出**：是受控 IndexR-like score + topk；本测试目标是 GBSA 单算子，故排除了 IndexR 延迟。线上真实 ids 可能具有不同的 cluster 结构。
3. **只测核心算子**：不包含 IndexR、metadata、pack/append-current-region、vLLM scheduler、continuous batching；端到端收益需要另测。
4. **vLLM wrapper 未直接用于计时**：容器内 `vllm_ascend` 的 import 受 `libcrypto.so.3 OPENSSL_3.3.0` 缺失影响；因此直接调用官方 `cann_ops_transformer` 低层 wrapper，避免引入上层依赖。核心 aclnn 算子相同。
5. **单机单卡、单次 session**：未跨卡/跨机做方差分析；未做频率/温度/功耗监控。跨机复现时建议先跑本脚本的 `--quick` 再跑全量。
6. **block_table 固定随机排列**：真实 page allocator 可能更连续或更碎片；本测试只说明在给定随机映射下的效应。
7. **未做数值正确性/精度对比**：只校验输出有限；未与 dense/reference attention 做逐元素比对。

---

## 10. 复现

### 10.1 容器内直接运行

```bash
# 在 177.45.22.1 的 j00934364_step5_v2 容器内
cd /root/gbsa_range_benchmark
source /opt/upgrade/runtime-env.sh
python3 scripts/run_bench.py \
  --output-dir outputs/full_12seed \
  --seeds 12 --warmup 10 --iters 50
```

快速 smoke（2 seeds / 10 iters / B=1,128）：

```bash
python3 scripts/run_bench.py --output-dir outputs/quick --quick
```

中断后可以续跑：

```bash
python3 scripts/run_bench.py --output-dir outputs/full_12seed \
  --seeds 12 --warmup 10 --iters 50 --resume
```

### 10.2 从宿主机复制

```bash
CONTAINER=j00934364_step5_v2
REMOTE=rack10os2
REMOTE_DIR=/root/gbsa_range_benchmark

ssh $REMOTE "mkdir -p $REMOTE_DIR/scripts"
scp scripts/run_bench.py $REMOTE:$REMOTE_DIR/scripts/run_bench.py
ssh $REMOTE "docker cp $REMOTE_DIR/scripts/run_bench.py $CONTAINER:$REMOTE_DIR/scripts/run_bench.py"
ssh $REMOTE "docker exec $CONTAINER bash -lc 'cd $REMOTE_DIR && source /opt/upgrade/runtime-env.sh && python3 scripts/run_bench.py --output-dir outputs/full_12seed --seeds 12 --warmup 10 --iters 50'"
ssh $REMOTE "rm -rf /tmp/gbsa_full; docker cp $CONTAINER:$REMOTE_DIR/outputs/full_12seed /tmp/gbsa_full"
scp -r $REMOTE:/tmp/gbsa_full ./results
```

### 10.3 重新生成报告 / Excel

本地（Windows/Linux 均可；需要 `pandas`, `numpy`, `Pillow`, `XlsxWriter`）：

```bash
python report/build_report.py --data results --out .
```

### 10.4 复现清单

- [x] 固定 `kv_seed=20261105`、`block_table_seed=20261106`、`selection_seed=7000..7011`
- [x] 固定 topk=512、region=8、page=128、heads=64、kv_groups=4、head_dim=192
- [x] 每个 run 保存 50 个原始计时样本（`results/raw_timings.csv`）
- [x] 环境快照（`results/environment.json`，含 `npu-smi info`）
- [x] 数据文件 SHA-256（`results/manifest.json`）
- [x] @@FINITE@@/720 输出 finite
- [x] 所有图 / Excel 可由 `report/build_report.py` 重新生成

---

## 11. 文件清单

| 文件 | 内容 |
|---|---|
| `scripts/run_bench.py` | 完整 benchmark（KV 生成、受控 topk、GBSA 计时、checkpoint/resume） |
| `scripts/run_in_container.sh` | 容器内一键运行 |
| `scripts/run_from_host.sh` | 从宿主机通过 ssh + docker 运行并回收结果 |
| `scripts/validate_results.py` | 不依赖 NPU 的结果完整性校验（行数、每 run 50 样本、median、SHA-256） |
| `report/build_report.py` | 从 `results/` 生成图表、Excel、Markdown |
| `results/README.md` | 数据文件与关键列说明 |
| `results/raw_runs.csv` | 720 个 run-level 记录（含选择 metrics、latency、种子） |
| `results/raw_timings.csv` | 36,000 个 per-iteration NPU-event 计时样本 |
| `results/metadata_bench.csv` | metadata 单独计时 |
| `results/config.json` | 本次运行的完整配置 |
| `results/environment.json` | 硬件/OS/Python/CANN/torch-npu/package 与 `npu-smi` |
| `results/manifest.json` | 完成时间、run 数、原始数据 SHA-256 |
| `results/summary_by_scene_range_batch.csv` | 汇总表（中位数/IQR/page/latency ratio） |
| `results/contrast_w1k_to_w16k.csv` | 1k → 16k 对照表 |
| `results/correlations_by_batch.csv` | 分 batch / scene 的 Spearman 相关 |
| `figures/fig1...` ~ `figures/fig7...` | 报告与 Excel 使用的 PNG（含 top-512 场景构造示意） |
| `GBSA_128K_TOP512_Range_Report.xlsx` | 原始数据 + 透视汇总 + 条件格式 + 原生 Excel 图 + Dashboard |
| `GBSA_128K_TOP512_Range_Report.md` | 本报告 |

---

## 12. 一句话总结

> **topk=512 不变时，GBSA 的范围效应本质上是页级工作集效应：同样 16k span，共享 sink 时约 @@M128_16K@@ ms，均匀分散时约 @@U128_16K@@ ms（B=128，@@U_M_RATIO@@×）。限制候选范围是必要条件，但还不够；必须同时约束 per-request page fan-out 和 batch-level union pages。**