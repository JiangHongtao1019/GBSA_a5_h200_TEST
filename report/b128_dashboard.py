#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a clean one-page B=128 latency comparison workbook."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import xlsxwriter

from report_common import COLORS, SCENE_ORDER

SCENE_DESC = {
    "recent_contiguous": "每个请求随机选一段连续 512-region；请求之间位置不同，单请求约 33 页，16k 时 batch 并集约 1022 页",
    "multi_sink": "8 个 64-region 热点带，所有请求共享同一组 512 个 region；单请求和 batch 并集都约 39 页",
    "uniform": "平坦分数，top-512 在候选窗口内均匀散开；16k 时单请求约 408 页，batch 并集 1024 页",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results")
    ap.add_argument("--out", default="GBSA_128K_B128_Dashboard_clean.xlsx")
    args = ap.parse_args()
    summary = pd.read_csv(Path(args.data) / "summary_by_scene_range_batch.csv")
    r = summary[summary.batch_size == 128].set_index(["scene", "range_regions_requested"])

    def v(sc: str, w: int, col: str) -> float:
        return float(r.loc[(sc, w), col])

    wb = xlsxwriter.Workbook(args.out, {"nan_inf_to_errors": True})
    ws = wb.add_worksheet("B128耗时对比")
    ws.hide_gridlines(2)
    ws.set_zoom(70)
    ws.set_default_row(14)
    ws.set_column("A:A", 30)
    ws.set_column("B:E", 15)
    ws.set_column("F:F", 18)
    ws.set_column("G:J", 16)
    ws.set_column("K:M", 16)

    f_title = wb.add_format({"bold": True, "font_size": 24, "font_color": "#0F172A"})
    f_sub = wb.add_format({"font_size": 11, "font_color": "#475569", "text_wrap": True, "valign": "vcenter"})
    f_section = wb.add_format({"bold": True, "font_size": 13, "font_color": "#FFFFFF", "bg_color": "#1E3A8A", "valign": "vcenter", "border": 1, "border_color": "#1E3A8A"})
    f_h = wb.add_format({"bold": True, "font_size": 11, "font_color": "#FFFFFF", "bg_color": "#334155", "border": 1, "border_color": "#475569", "align": "center", "valign": "vcenter", "text_wrap": True})
    f_txt = wb.add_format({"font_size": 11, "border": 1, "border_color": "#CBD5E1", "valign": "vcenter", "text_wrap": True})
    f_num = wb.add_format({"font_size": 12, "num_format": "0.0000", "border": 1, "border_color": "#CBD5E1", "align": "center", "valign": "vcenter"})
    f_ratio = wb.add_format({"font_size": 12, "num_format": "0.00\"×\"", "border": 1, "border_color": "#CBD5E1", "align": "center", "valign": "vcenter"})
    f_pct = wb.add_format({"font_size": 12, "num_format": "+0.0%;-0.0%", "border": 1, "border_color": "#CBD5E1", "align": "center", "valign": "vcenter"})
    f_kpi_l = wb.add_format({"font_size": 11, "font_color": "#475569", "bg_color": "#F1F5F9", "border": 1, "border_color": "#CBD5E1", "align": "center", "valign": "vcenter", "text_wrap": True})
    f_kpi_red = wb.add_format({"font_size": 19, "bold": True, "font_color": "#B91C1C", "bg_color": "#FEF2F2", "border": 1, "border_color": "#FECACA", "align": "center", "valign": "vcenter"})
    f_kpi_green = wb.add_format({"font_size": 19, "bold": True, "font_color": "#047857", "bg_color": "#ECFDF5", "border": 1, "border_color": "#A7F3D0", "align": "center", "valign": "vcenter"})
    f_note = wb.add_format({"font_size": 11, "font_color": "#334155", "text_wrap": True, "valign": "vcenter", "border": 1, "border_color": "#E2E8F0", "bg_color": "#F8FAFC"})
    f_concl = wb.add_format({"font_size": 11, "font_color": "#7C2D12", "text_wrap": True, "valign": "vcenter", "border": 1, "border_color": "#FDBA74", "bg_color": "#FFF7ED"})
    f_footer = wb.add_format({"font_size": 8, "font_color": "#64748B", "italic": True})

    ws.merge_range("A1:M1", "GBSA 128K / top-512 — B=128 耗时对比", f_title)
    ws.set_row(0, 38)
    ws.merge_range("A2:M2", "同一个 batch 内 128 个 request；topk=512 固定；完整 128K BF16 KV；12 个 selection seed × 50 次 NPU-event 计时；只计 GBSA 核心 op。", f_sub)
    ws.set_row(1, 30)

    kpis = [
        ("recent_contiguous\n1k → 16k", f"{v('recent_contiguous',1024,'latency_median_ms'):.4f} → {v('recent_contiguous',16384,'latency_median_ms'):.4f} ms\n+62.6%", f_kpi_red, 0, 2),
        ("uniform\n1k → 16k", f"{v('uniform',1024,'latency_median_ms'):.4f} → {v('uniform',16384,'latency_median_ms'):.4f} ms\n+96.2%", f_kpi_red, 3, 5),
        ("multi_sink\n1k → 16k", f"{v('multi_sink',1024,'latency_median_ms'):.4f} → {v('multi_sink',16384,'latency_median_ms'):.4f} ms\n-0.3%", f_kpi_green, 6, 9),
    ]
    for label, value, fmt, c0, c1 in kpis:
        ws.merge_range(3, c0, 3, c1, label, f_kpi_l)
        ws.merge_range(4, c0, 4, c1, value, fmt)
    ws.set_row(3, 38)
    ws.set_row(4, 58)

    ws.merge_range("A7:M7", "B=128 各候选范围中位耗时", f_section)
    ws.set_row(6, 26)
    heads = ["场景", "1k (ms)", "4k (ms)", "8k (ms)", "16k (ms)", "16k/1k\n延迟比", "16k vs 1k\n变化"]
    for j, h in enumerate(heads):
        ws.write(7, j, h, f_h)
    ws.merge_range(7, 6, 7, 9, heads[6], f_h)
    ws.set_row(7, 38)
    for i, sc in enumerate(SCENE_ORDER):
        row = 8 + i
        ws.write(row, 0, sc, f_txt)
        for j, w in enumerate([1024, 4096, 8192, 16384], start=1):
            ws.write_number(row, j, v(sc, w, "latency_median_ms"), f_num)
        ws.write_number(row, 5, v(sc, 16384, "speedup_x_vs_w1k"), f_ratio)
        ws.merge_range(row, 6, row, 9, "", f_pct)
        ws.write_number(row, 6, v(sc, 16384, "speedup_x_vs_w1k") - 1.0, f_pct)
        ws.set_row(row, 36)
    ws.conditional_format("B9:E11", {"type": "3_color_scale", "min_color": "#DCFCE7", "mid_color": "#FEF9C3", "max_color": "#FECACA"})
    ws.merge_range("A12:M12", "读表：multi_sink 从 1k 到 16k 基本不变；recent_contiguous +62.6%；uniform +96.2%。同样 topk=512，区别在 512 个 region 的空间分布和物理页复用。", f_note)
    ws.set_row(11, 36)
    ws.merge_range("A14:M14", "延迟随候选范围的变化", f_section)
    ws.set_row(13, 26)

    cd = wb.add_worksheet("_data")
    cd.hide()
    cd.write_row(0, 0, ["range", "recent_contiguous", "multi_sink", "uniform"], f_h)
    for i, w in enumerate([1024, 4096, 8192, 16384], start=1):
        cd.write(i, 0, f"{w//1024}k")
        for j, sc in enumerate(SCENE_ORDER, start=1):
            cd.write_number(i, j, v(sc, w, "latency_median_ms"), f_num)

    ch = wb.add_chart({"type": "line"})
    for j, sc in enumerate(SCENE_ORDER, start=1):
        ch.add_series({
            "name": ["_data", 0, j],
            "categories": ["_data", 1, 0, 4, 0],
            "values": ["_data", 1, j, 4, j],
            "line": {"color": COLORS[sc], "width": 3.25},
            "marker": {"type": "circle", "size": 9, "border": {"color": COLORS[sc]}, "fill": {"color": COLORS[sc]}},
            "data_labels": {"value": True, "num_format": "0.000",
                            "position": ("below" if sc == "multi_sink" else "above"),
                            "font": {"size": 9, "bold": True, "color": COLORS[sc]}},
        })
    ch.set_title({"name": "B=128：候选范围 1k → 16k 的 GBSA 核心耗时", "name_font": {"size": 14}})
    ch.set_x_axis({"name": "候选范围", "name_font": {"size": 11}})
    ch.set_y_axis({"name": "Latency (ms)", "name_font": {"size": 11}, "min": 0.34, "max": 1.10, "major_unit": 0.1, "major_gridlines": {"visible": True, "line": {"color": "#E5E7EB"}}})
    ch.set_legend({"position": "bottom", "font": {"size": 11}})
    ch.set_size({"width": 1300, "height": 620})
    ws.insert_chart("A15", ch, {"x_scale": 1.0, "y_scale": 1.0})

    ws.merge_range("A48:M48", "同样的 512 个 region，为什么空间分布不同？", f_section)
    ws.set_row(47, 26)
    fig7 = Path(__file__).resolve().parents[1] / "figures" / "fig7_scene_topk_patterns.png"
    if fig7.exists():
        ws.insert_image("A49", str(fig7), {"x_scale": 0.80, "y_scale": 0.80, "object_position": 1})

    ws.merge_range("A77:M79", "结论：B=128 时，候选范围从 1k 增到 16k，recent_contiguous +62.6%，uniform +96.2%，multi_sink -0.3%。耗时差异主要不是计算量，而是 top-512 落到多少物理页、这些页能否在 batch 内复用。", f_concl)
    ws.set_row(76, 24)
    ws.set_row(77, 24)
    ws.set_row(78, 24)

    ws.merge_range("A81:M81", "数据源：results/raw_runs.csv（720 runs / 36,000 个 timed samples）；表中为 12 个 selection seed 的中位数。复现：python report/b128_dashboard.py --data results --out GBSA_128K_B128_Dashboard_clean.xlsx", f_footer)
    ws.set_row(80, 18)

    ws.print_area("A1:M82")
    ws.set_landscape()
    ws.set_paper(66)
    ws.set_margins(left=0.25, right=0.25, top=0.3, bottom=0.3)
    ws.center_horizontally()
    ws.fit_to_pages(1, 1)
    ws.set_header("&C&\"Arial,Bold\"GBSA 128K / top-512 — B=128 latency")
    ws.set_footer("&LData: results/raw_runs.csv&CGenerated by report/b128_dashboard.py&RPage &P / &N")

    # explanatory sheet: keeps the main B=128 page clean while documenting the three id patterns
    ws2 = wb.add_worksheet("场景构造_示意")
    ws2.hide_gridlines(2)
    ws2.set_column("A:A", 2)
    ws2.set_column("B:B", 12)
    ws2.set_column("C:J", 15)
    ws2.write("B2", "三种 top-512 选择模式：为什么同样的 topk 耗时不同", f_title)
    ws2.merge_range("B3:J3", "受控 IndexR-like 构造：候选窗口内每个 region 先取高斯 score，按场景叠加矩形脉冲，再取 top-512。图中展示 group 0、请求 0..2。", f_sub)
    ws2.set_row(2, 30)
    img = Path(__file__).resolve().parents[1] / "figures" / "fig7_scene_topk_patterns.png"
    if img.exists():
        ws2.insert_image("B5", str(img), {"x_scale": 0.50, "y_scale": 0.50, "object_position": 1})
    ws2.merge_range("B39:J39", "读图", f_h)
    notes = [
        "recent_contiguous：每个请求在候选窗口内随机选一段连续 512-region 带；请求之间位置不同。",
        "multi_sink：窗口内均匀放 8 个 64-region 热点带，8 × 64 = 512；所有请求共享同一组 region。",
        "uniform：不加脉冲，直接对平坦的高斯 score 取 top-512；512 个 region 近似均匀分散。",
        "共同点：topk 都是 512、counts 都是 513；区别只在 512 个 region 的空间位置。",
        "关键：512 个 id 映射到多少物理页、这些页能否在 batch 内复用，才决定 GBSA 耗时。",
    ]
    rr = 40
    for txt in notes:
        ws2.merge_range(rr, 1, rr, 9, "• " + txt, f_sub)
        ws2.set_row(rr, 28)
        rr += 1
    ws2.print_area("A1:J46")
    ws2.set_landscape()
    ws2.set_paper(8)
    ws2.fit_to_pages(1, 1)

    wb.close()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())