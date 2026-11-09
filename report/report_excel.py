#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the Excel workbook (raw data + pivot-style summaries + charts)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xlsxwriter

from report_common import COLORS, SCENE_ORDER


def write_df(ws: Any, df: pd.DataFrame, start_row: int = 0, start_col: int = 0,
             hdr_fmt: Any = None, body_fmts: dict[str, Any] | None = None) -> tuple[int, int]:
    body_fmts = body_fmts or {}
    for j, c in enumerate(df.columns):
        ws.write(start_row, start_col + j, c, hdr_fmt)
    for i, row in enumerate(df.itertuples(index=False), start=1):
        for j, c in enumerate(df.columns):
            v = getattr(row, c)
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = float(v)
            elif isinstance(v, (np.bool_,)):
                v = bool(v)
            ws.write(start_row + i, start_col + j, v, body_fmts.get(c))
    return start_row + len(df), start_col + len(df.columns) - 1


def make_excel(summary: pd.DataFrame, contrast: pd.DataFrame, corr: pd.DataFrame,
               raw: pd.DataFrame, timings: pd.DataFrame, meta: pd.DataFrame,
               config: dict[str, Any], env: dict[str, Any], manifest: dict[str, Any],
               figs: dict[str, Path], out_path: Path) -> None:
    wb = xlsxwriter.Workbook(str(out_path), {"nan_inf_to_errors": True})
    f_title = wb.add_format({"bold": True, "font_size": 20, "font_color": "#111827"})
    f_sub = wb.add_format({"font_size": 11, "font_color": "#4B5563", "text_wrap": True, "valign": "top"})
    f_h = wb.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#1F2937",
                         "border": 1, "border_color": "#374151", "align": "center",
                         "valign": "vcenter", "text_wrap": True})
    f_num = wb.add_format({"num_format": "0.0000", "border": 1, "border_color": "#E5E7EB"})
    f_int = wb.add_format({"num_format": "0", "border": 1, "border_color": "#E5E7EB"})
    f_txt = wb.add_format({"border": 1, "border_color": "#E5E7EB", "valign": "top"})
    f_x = wb.add_format({"num_format": "0.00\"×\"", "border": 1, "border_color": "#E5E7EB"})
    f_note = wb.add_format({"font_size": 11, "font_color": "#374151", "text_wrap": True, "valign": "top"})

    # README
    ws = wb.add_worksheet("README")
    ws.set_column("A:A", 3); ws.set_column("B:B", 34); ws.set_column("C:C", 95)
    ws.write("B2", "GBSA 128K / top-512 candidate-range benchmark", f_title)
    ws.write("B4", "Scope", f_h)
    ws.write("C4", "Ascend aclnn GenericBlockSparseAttention core op only; metadata / pack / IndexR excluded.", f_note)
    ws.write("B5", "Run matrix", f_h)
    ws.write("C5", "5 batch sizes × 3 selection scenes × 4 candidate ranges × 12 seeds × 50 timed iterations = 720 runs / 36,000 timed samples.", f_note)
    ws.write("B6", "Primary metric", f_h)
    ws.write("C6", "NPU-event elapsed time of the core op; run median across 50 iterations; group median across 12 selection seeds (IQR across seeds).", f_note)
    ws.write("B7", "Candidate range", f_h)
    ws.write("C7", "1k / 4k / 8k / 16k logical 8-token regions = 8K / 32K / 64K / 128K candidate tokens. At 16k requested, 16,383 historical regions exist.", f_note)
    ws.write("B8", "Key finding", f_h)
    ws.write("C8", "Latency is driven by per-request physical-page fan-out and batch-level union KV working set, not by candidate span alone. See 透视_延迟比 and 透视图_Dashboard.", f_note)
    ws.write("B9", "Raw data", f_h)
    ws.write("C9", "原始数据_Runs (one row per run), 原始数据_Timings (all 50 samples per run). SHA-256 is in manifest.json.", f_note)
    ws.write("B10", "Reproduce", f_h)
    ws.write("C10", "cd /root/gbsa_range_benchmark && source /opt/upgrade/runtime-env.sh && python3 scripts/run_bench.py --output-dir outputs/full_12seed --seeds 12 --warmup 10 --iters 50", f_note)
    ws.write("B11", "Pivot note", f_h)
    ws.write("C11", "The 透视_* sheets are materialized pivot-style crosstabs (not native Excel PivotTable objects). The Dashboard contains native Excel line/scatter charts plus the PNGs.", f_note)
    ws.write("B13", "Sheet map", f_h)
    sheet_map = [
        ("原始数据_Runs", "All 720 run-level records and selection metrics."),
        ("原始数据_Timings", "36,000 per-iteration NPU-event timings."),
        ("汇总_Summary", "Aggregated medians / IQR / page footprint / latency ratio."),
        ("透视_延迟", "Pivot-style crosstabs: scene × candidate range, one block per batch size."),
        ("透视_离散度", "Per-request pages, batch-union pages and working set at B=128."),
        ("透视_延迟比", "W=16k / W=1k latency ratio heat table (>1 = slower at 16k)."),
        ("场景构造_示意", "Fig 7: how the three controlled top-512 selection patterns are constructed."),
        ("透视图_Dashboard", "Embedded PNG figures + native Excel line/scatter charts."),
        ("环境_Environment", "Hardware, OS, Python, CANN, torch-npu, package versions."),
        ("配置_Config", "Exact benchmark configuration."),
    ]
    rr = 14
    for name, desc in sheet_map:
        ws.write(rr, 1, name, f_txt); ws.write(rr, 2, desc, f_txt); rr += 1
    ws.set_row(14, 30)
    ws.freeze_panes(2, 2)

    # raw runs
    ws = wb.add_worksheet("原始数据_Runs")
    ws.freeze_panes(1, 1)
    ws.set_column(0, 0, 32); ws.set_column(1, len(raw.columns) - 1, 14)
    body = {}
    for c in raw.columns:
        if raw[c].dtype.kind in "iu":
            body[c] = f_int
        elif raw[c].dtype.kind == "f":
            body[c] = f_num
        else:
            body[c] = f_txt
    write_df(ws, raw, 0, 0, f_h, body)
    ws.autofilter(0, 0, len(raw), len(raw.columns) - 1)

    # raw timings
    ws = wb.add_worksheet("原始数据_Timings")
    ws.freeze_panes(1, 1)
    ws.set_column("A:A", 38); ws.set_column("B:B", 10); ws.set_column("C:C", 14)
    for j, c in enumerate(timings.columns):
        ws.write(0, j, c, f_h)
    for i, row in enumerate(timings.itertuples(index=False), start=1):
        ws.write(i, 0, row.run_id, f_txt)
        ws.write_number(i, 1, int(row.iter), f_int)
        ws.write_number(i, 2, float(row.latency_ms), f_num)
    ws.autofilter(0, 0, len(timings), len(timings.columns) - 1)

    # summary
    ws = wb.add_worksheet("汇总_Summary")
    ws.freeze_panes(1, 1)
    ws.set_column(0, len(summary.columns) - 1, 16)
    body = {}
    for c in summary.columns:
        if summary[c].dtype.kind in "iu":
            body[c] = f_int
        elif summary[c].dtype.kind == "f":
            body[c] = f_num
        else:
            body[c] = f_txt
    write_df(ws, summary, 0, 0, f_h, body)
    ws.autofilter(0, 0, len(summary), len(summary.columns) - 1)
    sp_col = list(summary.columns).index("speedup_x_vs_w1k")
    ws.conditional_format(1, sp_col, len(summary), sp_col, {
        "type": "3_color_scale", "min_color": "#DCFCE7", "mid_color": "#FEF3C7",
        "max_color": "#FECACA", "min_type": "num", "min_value": 0.98,
        "mid_type": "num", "mid_value": 1.3, "max_type": "num", "max_value": 2.1,
    })

    # pivot latency
    ws = wb.add_worksheet("透视_延迟")
    ws.set_column("A:A", 28); ws.set_column("B:G", 16)
    row = 0
    ws.write(row, 0, "Pivot-style view: median latency (ms), scene × range, one block per batch size", f_title)
    row = 2
    block_title = wb.add_format({"bold": True, "font_size": 14, "bg_color": "#E0E7FF",
                                 "font_color": "#1E3A8A", "border": 1})
    for b in sorted(summary.batch_size.unique()):
        ws.merge_range(row, 0, row, 6, f"B = {b}", block_title)
        row += 1
        ws.write_row(row, 0, ["Scene", "1k", "4k", "8k", "16k", "Δ 16k-1k (ms)", "Latency ratio 16k/1k"], f_h)
        row += 1
        first_data = row
        for sc in SCENE_ORDER:
            ws.write(row, 0, sc, f_txt)
            for j, w in enumerate([1024, 4096, 8192, 16384], start=1):
                v = float(summary[(summary.batch_size == b) & (summary.scene == sc) &
                                  (summary.range_regions_requested == w)]["latency_median_ms"].iloc[0])
                ws.write_number(row, j, v, f_num)
            row16 = summary[(summary.batch_size == b) & (summary.scene == sc) &
                            (summary.range_regions_requested == 16384)].iloc[0]
            ws.write_number(row, 5, float(row16["delta_ms_vs_w1k"]), f_num)
            ws.write_number(row, 6, float(row16["speedup_x_vs_w1k"]), f_x)
            row += 1
        ws.conditional_format(first_data, 1, row - 1, 4, {
            "type": "3_color_scale", "min_color": "#DCFCE7", "mid_color": "#FEF9C3", "max_color": "#FECACA"})
        ws.conditional_format(first_data, 6, row - 1, 6, {"type": "data_bar", "bar_color": "#F97316"})
        row += 1

    # pivot dispersion
    ws = wb.add_worksheet("透视_离散度")
    ws.set_column("A:A", 28); ws.set_column("B:G", 18)
    ws.write(0, 0, "Pivot-style view: page footprint at B=128", f_title)
    row = 2
    block_title_g = wb.add_format({"bold": True, "font_size": 13, "bg_color": "#ECFDF5",
                                   "font_color": "#065F46", "border": 1})
    for label, col_name in [
        ("Per-request unique physical pages (median)", "per_request_pages_median"),
        ("Batch-union unique physical pages (median)", "union_pages_median"),
        ("Batch-union KV working set (MiB, median)", "working_set_mib_median"),
    ]:
        ws.merge_range(row, 0, row, 4, label, block_title_g)
        row += 1
        ws.write_row(row, 0, ["Scene", "1k", "4k", "8k", "16k"], f_h)
        row += 1
        first = row
        for sc in SCENE_ORDER:
            ws.write(row, 0, sc, f_txt)
            for j, w in enumerate([1024, 4096, 8192, 16384], start=1):
                v = float(summary[(summary.batch_size == 128) & (summary.scene == sc) &
                                  (summary.range_regions_requested == w)][col_name].iloc[0])
                ws.write_number(row, j, v, f_num)
            row += 1
        ws.conditional_format(first, 1, row - 1, 4, {
            "type": "3_color_scale", "min_color": "#DBEAFE", "mid_color": "#FEF9C3", "max_color": "#FECACA"})
        row += 1

    # pivot latency ratio
    ws = wb.add_worksheet("透视_延迟比")
    ws.set_column("A:A", 28); ws.set_column("B:F", 16)
    ws.write(0, 0, "Pivot-style view: W=16k / W=1k median-latency ratio", f_title)
    ws.write(2, 0, "Scene", f_h)
    for j, b in enumerate(sorted(summary.batch_size.unique()), start=1):
        ws.write(2, j, f"B={b}", f_h)
    for i, sc in enumerate(SCENE_ORDER, start=3):
        ws.write(i, 0, sc, f_txt)
        for j, b in enumerate(sorted(summary.batch_size.unique()), start=1):
            v = float(summary[(summary.batch_size == b) & (summary.scene == sc) &
                              (summary.range_regions_requested == 16384)]["speedup_x_vs_w1k"].iloc[0])
            ws.write_number(i, j, v, f_x)
    ws.conditional_format(3, 1, 5, len(summary.batch_size.unique()), {
        "type": "3_color_scale", "min_color": "#DCFCE7", "mid_color": "#FEF9C3",
        "max_color": "#FECACA", "min_type": "num", "min_value": 0.98,
        "mid_type": "num", "mid_value": 1.3, "max_type": "num", "max_value": 2.1,
    })

    # scene-pattern explanation sheet
    ws = wb.add_worksheet("场景构造_示意")
    ws.hide_gridlines(2)
    ws.set_column("A:A", 2)
    ws.set_column("B:B", 12)
    ws.set_column("C:J", 15)
    ws.write("B2", "三种 top-512 选择模式：是怎么构造的", f_title)
    ws.merge_range("B3:J3",
                   "受控的 IndexR-like 构造：候选窗口内每个 region 先取高斯 score，按场景叠加矩形脉冲，再取 top-512。"
                   "图中展示 group 0、请求 0..2；其余 3 个 group 规则相同。",
                   f_note)
    fig7 = figs.get("fig7")
    if fig7 and Path(fig7).exists():
        ws.insert_image("B5", str(fig7), {"x_scale": 0.52, "y_scale": 0.52, "object_position": 1})
    ws.merge_range("B42:J42", "读图", f_h)
    lines = [
        "recent_contiguous：每个请求在候选窗口内随机选一段连续 512-region 带；单请求局部连续，但请求之间位置不同。",
        "multi_sink：在候选窗口内均匀放 8 个 64-region 热点带，8 × 64 = 512；所有请求共享同一组 region。",
        "uniform：不加脉冲，直接对平坦的高斯 score 取 top-512；512 个 region 在窗口内近似均匀分散。",
        "共同点：topk 都是 512、counts 都是 513、selected token 数相同；区别只在 512 个 region 的空间位置。",
        "注意：这是受控 IndexR-like ids 示意，不是实际 IndexR 算子输出；本测试只计时 GBSA 核心 op。",
    ]
    rr = 43
    for txt in lines:
        ws.merge_range(rr, 1, rr, 9, "• " + txt, f_note)
        ws.set_row(rr, 30)
        rr += 1

    # hidden chart data
    ws_cd = wb.add_worksheet("_chart_data")
    ws_cd.hide()
    ws_cd.write_row(0, 0, ["union_recent", "lat_recent", "union_multi", "lat_multi",
                           "union_uniform", "lat_uniform"], f_h)
    for i, w in enumerate([1024, 4096, 8192, 16384], start=1):
        vals = {}
        for sc in SCENE_ORDER:
            row = summary[(summary.batch_size == 128) & (summary.scene == sc) &
                          (summary.range_regions_requested == w)].iloc[0]
            vals[sc] = (float(row.union_pages_median), float(row.latency_median_ms))
        for j, sc in enumerate(SCENE_ORDER):
            ws_cd.write_number(i, j * 2, vals[sc][0], f_num)
            ws_cd.write_number(i, j * 2 + 1, vals[sc][1], f_num)

    # dashboard
    ws = wb.add_worksheet("透视图_Dashboard")
    ws.hide_gridlines(2)
    ws.set_column("A:A", 2); ws.set_column("B:H", 18)
    ws.write("B2", "GBSA 128K / top-512 — Dashboard", f_title)
    ws.write("B3", "Below: static high-resolution figures (same as report) and native interactive Excel charts at the bottom.", f_sub)
    img_positions = [
        ("fig1", "B5", 0.46), ("fig2", "B28", 0.46), ("fig3", "B52", 0.46),
        ("fig4", "B74", 0.46), ("fig5", "B96", 0.46), ("fig6", "B116", 0.46),
    ]
    for key, cell, scale in img_positions:
        p = figs.get(key)
        if p and p.exists():
            ws.insert_image(cell, str(p), {"x_scale": scale, "y_scale": scale, "object_position": 1})

    cd_row = 20
    ws_cd.write(cd_row, 0, "Native line data B=128", f_h)
    ws_cd.write_row(cd_row + 1, 0, ["Scene", "1k", "4k", "8k", "16k"], f_h)
    for i, sc in enumerate(SCENE_ORDER, start=cd_row + 2):
        ws_cd.write(i, 0, sc)
        for j, w in enumerate([1024, 4096, 8192, 16384], start=1):
            v = float(summary[(summary.batch_size == 128) & (summary.scene == sc) &
                              (summary.range_regions_requested == w)]["latency_median_ms"].iloc[0])
            ws_cd.write_number(i, j, v, f_num)

    chart = wb.add_chart({"type": "line"})
    for i, sc in enumerate(SCENE_ORDER, start=cd_row + 2):
        chart.add_series({
            "name": ["_chart_data", i, 0],
            "categories": ["_chart_data", cd_row + 1, 1, cd_row + 1, 4],
            "values": ["_chart_data", i, 1, i, 4],
            "line": {"color": COLORS[sc], "width": 2.5},
            "marker": {"type": "circle", "size": 7, "border": {"color": COLORS[sc]},
                       "fill": {"color": COLORS[sc]}},
        })
    chart.set_title({"name": "Native Excel line chart — latency vs candidate range (B=128)"})
    chart.set_x_axis({"name": "Candidate range (logical regions)"})
    chart.set_y_axis({"name": "Latency (ms)", "major_gridlines": {"visible": True,
                      "line": {"color": "#E5E7EB"}}})
    chart.set_legend({"position": "bottom"})
    chart.set_size({"width": 920, "height": 480})
    ws.insert_chart("B140", chart, {"x_scale": 1.0, "y_scale": 1.0})

    scat = wb.add_chart({"type": "scatter"})
    for j, sc in enumerate(SCENE_ORDER):
        xcol = j * 2
        ycol = xcol + 1
        scat.add_series({
            "name": sc,
            "categories": ["_chart_data", 1, xcol, 4, xcol],
            "values": ["_chart_data", 1, ycol, 4, ycol],
            "marker": {"type": "circle", "size": 9, "border": {"color": COLORS[sc]},
                       "fill": {"color": COLORS[sc]}},
            "line": {"color": COLORS[sc], "width": 2.0},
        })
    scat.set_title({"name": "Native Excel scatter — latency vs batch-union pages (B=128)"})
    scat.set_x_axis({"name": "Unique physical KV pages in batch union"})
    scat.set_y_axis({"name": "Latency (ms)", "major_gridlines": {"visible": True,
                     "line": {"color": "#E5E7EB"}}})
    scat.set_legend({"position": "bottom"})
    scat.set_size({"width": 920, "height": 480})
    ws.insert_chart("B166", scat, {"x_scale": 1.0, "y_scale": 1.0})

    # environment
    ws = wb.add_worksheet("环境_Environment")
    ws.set_column("A:A", 38); ws.set_column("B:B", 110)
    ws.write(0, 0, "Environment / reproducibility snapshot", f_title)
    flat = {
        "timestamp": env.get("timestamp"), "hostname": env.get("hostname"),
        "platform": env.get("platform"), "machine": env.get("machine"),
        "python": env.get("python"), "npu_device": env.get("npu_device"),
        "torch": env.get("torch", {}).get("__version__"),
        "torch_npu": env.get("python_packages", {}).get("torch-npu"),
        "vllm": env.get("python_packages", {}).get("vllm"),
        "vllm_ascend": env.get("python_packages", {}).get("vllm-ascend"),
        "cann_ops_transformer": env.get("python_packages", {}).get("cann-ops-transformer"),
        "triton": env.get("python_packages", {}).get("triton"),
        "npu_device_name_0": env.get("torch", {}).get("npu_device_name_0"),
        "ASCEND_HOME_PATH": env.get("env", {}).get("ASCEND_HOME_PATH"),
        "ASCEND_CUSTOM_OPP_PATH": env.get("env", {}).get("ASCEND_CUSTOM_OPP_PATH"),
        "command_line": env.get("command_line"),
        "raw_runs_sha256": manifest.get("raw_runs_sha256"),
        "raw_timings_sha256": manifest.get("raw_timings_sha256"),
        "completed_at": manifest.get("completed_at"),
    }
    row = 2
    for k, v in flat.items():
        ws.write(row, 0, k, f_h); ws.write(row, 1, str(v), f_txt); row += 1
    ws.write(row + 1, 0, "npu-smi info", f_h)
    ws.merge_range(row + 2, 0, row + 20, 1, env.get("npu_smi", ""),
                   wb.add_format({"font_name": "Consolas", "font_size": 9,
                                  "text_wrap": True, "valign": "top"}))

    # config
    ws = wb.add_worksheet("配置_Config")
    ws.set_column("A:A", 40); ws.set_column("B:B", 110)
    ws.write(0, 0, "Benchmark config", f_title)
    cfg_text = json.dumps(config, indent=2, ensure_ascii=False)
    ws.merge_range(2, 0, 60, 1, cfg_text,
                   wb.add_format({"font_name": "Consolas", "font_size": 9,
                                  "text_wrap": True, "valign": "top"}))
    wb.close()