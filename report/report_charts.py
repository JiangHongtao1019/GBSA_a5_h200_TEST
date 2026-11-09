#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate PNG figures for the GBSA report using Pillow only."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import ImageDraw

from report_common import (
    AXIS, BG, COLORS, COLORS_RGB, GRID, INK, MUTED, PANEL_BG, SCENE_ORDER,
    SCENE_SHORT, F, P, box, draw_legend, draw_line_panel, draw_rounded_box,
    draw_text, human_range, new_canvas, save,
)


def make_figures(summary: pd.DataFrame, raw: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    figs: dict[str, Path] = {}
    pv = summary.set_index(["batch_size", "scene", "range_regions_requested"])
    ranges = [1024, 4096, 8192, 16384]
    xlabels = [human_range(w) for w in ranges]

    def series_for(b: int, metric: str = "latency_median_ms",
                   q1: str | None = "latency_q25_ms",
                   q3: str | None = "latency_q75_ms") -> list[dict[str, Any]]:
        out = []
        for sc in SCENE_ORDER:
            vals, a, b_ = [], [], []
            for w in ranges:
                row = pv.loc[(b, sc, w)]
                vals.append(float(row[metric]))
                if q1:
                    a.append(float(row[q1]))
                if q3:
                    b_.append(float(row[q3]))
            out.append({"name": sc, "color": COLORS[sc], "values": vals,
                        "q1": a if q1 else [], "q3": b_ if q3 else []})
        return out

    # --- Figure 1: B=128 line chart -----------------------------------------
    img = new_canvas(1800, 980)
    draw = ImageDraw.Draw(img)
    draw_text(draw, (80, 38), "GBSA core-op latency vs candidate range — B = 128",
              F(28, bold=True), fill=INK)
    draw_text(draw, (80, 80),
              "topk = 512 fixed; full 128K BF16 KV cache; line = median of 12 selection seeds; band = IQR across seeds",
              F(15), fill=MUTED)
    draw_legend(img, [(SCENE_SHORT[s], COLORS[s]) for s in SCENE_ORDER], 80, 112, font_size=15)
    draw_line_panel(img, (70, 150, 1730, 890), xlabels, series_for(128),
                    title="B = 128", y_unit="Latency (ms)", show_xlabel=True,
                    annotate_last=True)
    draw_rounded_box(draw, (1145, 665, 1675, 850), fill="#FFF7ED", outline="#FDBA74", radius=12)
    draw_text(draw, (1165, 682), "Same topk, different physical layout", F(16, bold=True), fill="#9A3412")
    draw_text(draw, (1165, 712), "16k-span multi_sink: 0.374 ms", F(13), fill=INK)
    draw_text(draw, (1165, 736), "16k-span uniform:     1.043 ms", F(13), fill=INK)
    draw_text(draw, (1165, 766), "Selected span alone is not the driver;", F(13), fill=INK)
    draw_text(draw, (1165, 790), "batch-level KV page working set is.", F(13), fill=INK)
    p = out_dir / "fig1_latency_vs_range_B128.png"
    save(img, p); figs["fig1"] = p

    # --- Figure 2: small multiples over batch sizes --------------------------
    bs = sorted(raw.batch_size.unique())
    img = new_canvas(1800, 1220)
    draw = ImageDraw.Draw(img)
    draw_text(draw, (70, 34), "Latency vs candidate range across batch sizes",
              F(28, bold=True), fill=INK)
    draw_text(draw, (70, 76),
              "Each panel has its own y scale. Percentages are the W=16k vs W=1k median-latency change.",
              F(15), fill=MUTED)
    draw_legend(img, [(SCENE_SHORT[s], COLORS[s]) for s in SCENE_ORDER], 70, 106, font_size=14)
    cols, rows = 3, 2
    x_margin, x_gap = 70, 36
    panel_w = (1800 - 2 * x_margin - (cols - 1) * x_gap) / cols
    y_top, y_bottom = 155, 1155
    panel_h = (y_bottom - y_top - 36) / rows
    for idx, b in enumerate(bs):
        row_i, col_i = idx // cols, idx % cols
        x0 = x_margin + col_i * (panel_w + x_gap)
        y0 = y_top + row_i * (panel_h + 36)
        pct = float(pv.loc[(b, "recent_contiguous", 16384), "delta_pct_vs_w1k"])
        title = f"B = {b}   (recent_contiguous Δ16k = {pct:+.1f}%)"
        draw_line_panel(img, (x0, y0, x0 + panel_w, y0 + panel_h), xlabels,
                        series_for(b), title=title, y_unit="Latency (ms)",
                        show_xlabel=True)
    x0 = x_margin + 2 * (panel_w + x_gap)
    y0 = y_top + (panel_h + 36)
    draw_rounded_box(draw, (x0, y0, x0 + panel_w, y0 + panel_h),
                     fill=PANEL_BG, outline=GRID, radius=12)
    draw_text(draw, (x0 + 18, y0 + 16), "How to read", F(18, bold=True), fill=INK)
    lines = [
        "B = 1 / 8: launch overhead dominates;",
        "range effect is ≤ 2.3% and not material.",
        "",
        "B ≥ 32: the batch-level union KV",
        "working set becomes visible.",
        "",
        "At B = 128 / 256 the same top-512",
        "can differ by ~1.6–2.8× depending",
        "on how ids are spatially distributed.",
    ]
    yy = y0 + 52
    for ln in lines:
        draw_text(draw, (x0 + 18, yy), ln, F(13), fill=MUTED)
        yy += 24
    p = out_dir / "fig2_latency_vs_range_facets.png"
    save(img, p); figs["fig2"] = p

    # --- Figure 3: per-request pages vs union pages --------------------------
    img = new_canvas(1800, 900)
    draw = ImageDraw.Draw(img)
    draw_text(draw, (70, 34), "Two different page-footprint metrics", F(28, bold=True), fill=INK)
    draw_text(draw, (70, 76),
              "B = 128. Per-request pages describe intra-request dispersion; union pages describe the whole-batch HBM working set.",
              F(15), fill=MUTED)
    draw_legend(img, [(SCENE_SHORT[s], COLORS[s]) for s in SCENE_ORDER], 70, 108, font_size=14)
    s_pr, s_un = [], []
    for sc in SCENE_ORDER:
        vals_pr, vals_un = [], []
        for w in ranges:
            vals_pr.append(float(pv.loc[(128, sc, w), "per_request_pages_median"]))
            vals_un.append(float(pv.loc[(128, sc, w), "union_pages_median"]))
        s_pr.append({"name": sc, "color": COLORS[sc], "values": vals_pr})
        s_un.append({"name": sc, "color": COLORS[sc], "values": vals_un})
    draw_line_panel(img, (60, 150, 880, 820), xlabels, s_pr,
                    title="Per-request unique physical pages", y_unit="Pages / request",
                    show_xlabel=True, annotate_last=True)
    draw_line_panel(img, (920, 150, 1740, 820), xlabels, s_un,
                    title="Union of physical pages across batch", y_unit="Unique pages",
                    show_xlabel=True, annotate_last=True,
                    hline=(1024, "#9CA3AF", "1024 = entire cache"))
    p = out_dir / "fig3_page_footprint_mechanism.png"
    save(img, p); figs["fig3"] = p

    # --- Figure 4: latency vs union pages scatter ---------------------------
    img = new_canvas(1800, 1000)
    draw = ImageDraw.Draw(img)
    draw_text(draw, (70, 34), "Latency vs batch-level KV page working set — B = 128",
              F(28, bold=True), fill=INK)
    rho = float(raw[raw.batch_size == 128][["touched_pages_union", "latency_ms_median"]]
                .corr(method="spearman").iloc[0, 1])
    draw_text(draw, (70, 76),
              f"All 144 runs (12 seeds × 4 ranges × 3 scenes). Spearman ρ = {rho:.3f}.",
              F(15), fill=MUTED)
    draw_legend(img, [(SCENE_SHORT[s], COLORS[s]) for s in SCENE_ORDER], 70, 108, font_size=14)
    x0, y0, x1, y1 = 100, 155, 1720, 885
    left, right, top, bottom = 95, 30, 35, 85
    px0, py0, px1, py1 = x0 + left, y0 + top, x1 - right, y1 - bottom
    sub = raw[raw.batch_size == 128]
    xmin, xmax = math.log2(30), math.log2(1200)
    ymin, ymax = 0.33, 1.10

    def xp(v: float) -> float:
        return px0 + (math.log2(max(v, 30.0)) - xmin) * (px1 - px0) / (xmax - xmin)

    def yp(v: float) -> float:
        return py1 - (v - ymin) * (py1 - py0) / (ymax - ymin)

    for tick in [32, 64, 128, 256, 512, 1024]:
        xx = xp(tick)
        draw.line([P(xx, py0), P(xx, py1)], fill=GRID, width=1 * 2)
        draw_text(draw, (xx, py1 + 16), str(tick), F(13), fill=MUTED, anchor="ma")
    for k in range(5):
        v = ymin + (ymax - ymin) * k / 4
        yy = yp(v)
        draw.line([P(px0, yy), P(px1, yy)], fill=GRID, width=1 * 2)
        draw_text(draw, (px0 - 12, yy), f"{v:.2f}", F(13), fill=MUTED, anchor="rm")
    draw.line([P(px0, py0), P(px0, py1)], fill=AXIS, width=1 * 2)
    draw.line([P(px0, py1), P(px1, py1)], fill=AXIS, width=1 * 2)
    draw_text(draw, ((px0 + px1) / 2, y1 - 22),
              "Unique physical KV pages in batch union (log scale)",
              F(13), fill=MUTED, anchor="ma")
    draw_text(draw, (x0 + 16, y0 + 10), "Latency (ms)", F(12), fill=MUTED)
    for sc in SCENE_ORDER:
        s = sub[sub.scene == sc]
        col = COLORS_RGB[sc]
        for _, r in s.iterrows():
            xx, yy = xp(float(r.touched_pages_union)), yp(float(r.latency_ms_median))
            draw.ellipse(box(xx - 4.2, yy - 4.2, xx + 4.2, yy + 4.2), fill=(*col, 130))
    for sc in SCENE_ORDER:
        med = summary[(summary.batch_size == 128) & (summary.scene == sc)].sort_values("union_pages_median")
        pts = [(xp(float(r.union_pages_median)), yp(float(r.latency_median_ms))) for _, r in med.iterrows()]
        draw.line([P(x, y) for x, y in pts], fill=COLORS[sc], width=int(3 * 2), joint="curve")
        for x, y in pts:
            draw.ellipse(box(x - 7, y - 7, x + 7, y + 7), fill=COLORS[sc],
                         outline="#FFFFFF", width=2 * 2)
    draw_rounded_box(draw, (1125, 190, 1685, 310), fill="#FFF7ED", outline="#FDBA74", radius=12)
    draw_text(draw, (1145, 204), "At ~16k selected span:", F(16, bold=True), fill="#9A3412")
    draw_text(draw, (1145, 234), "multi_sink  0.374 ms  (39 pages)", F(13), fill=INK)
    draw_text(draw, (1145, 258), "uniform     1.043 ms  (1024 pages)", F(13), fill=INK)
    draw_text(draw, (1145, 282), "→ 2.79× latency difference", F(13, bold=True), fill="#B91C1C")
    p = out_dir / "fig4_latency_vs_union_pages.png"
    save(img, p); figs["fig4"] = p

    # --- Figure 5: heatmap of W16k/W1k speedup -------------------------------
    img = new_canvas(1800, 850)
    draw = ImageDraw.Draw(img, "RGBA")
    draw_text(draw, (70, 34), "W = 16k vs W = 1k latency ratio", F(28, bold=True), fill=INK)
    draw_text(draw, (70, 76),
              "Ratio > 1 means the wider candidate range makes GBSA slower. Values are median-latency ratios.",
              F(15), fill=MUTED)
    bs_disp = sorted(summary.batch_size.unique())
    left, top = 260, 190
    cw, ch = 250, 118

    def speed_color(v: float) -> tuple[int, int, int]:
        if v <= 1.0:
            t = min(max((1.0 - v) / 0.10, 0.0), 1.0)
            a, b = (249, 250, 251), (220, 252, 231)
            return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))
        t = min(max((v - 1.0) / 1.0, 0.0), 1.0)
        if t < 0.55:
            u = t / 0.55
            a, b = (249, 250, 251), (254, 215, 170)
            return tuple(int(a[i] + (b[i] - a[i]) * u) for i in range(3))
        u = (t - 0.55) / 0.45
        a, b = (254, 215, 170), (220, 38, 38)
        return tuple(int(a[i] + (b[i] - a[i]) * u) for i in range(3))

    for j, b in enumerate(bs_disp):
        draw_text(draw, (left + j * cw + cw / 2, top - 20), f"B={b}",
                  F(15, bold=True), fill=INK, anchor="ma")
    for i, sc in enumerate(SCENE_ORDER):
        draw_text(draw, (left - 20, top + i * ch + ch / 2), SCENE_SHORT[sc],
                  F(14), fill=INK, anchor="rm")
        for j, b in enumerate(bs_disp):
            v = float(pv.loc[(b, sc, 16384), "speedup_x_vs_w1k"])
            col = speed_color(v)
            xx, yy = left + j * cw, top + i * ch
            draw.rectangle(box(xx, yy, xx + cw, yy + ch), fill=col, outline="#FFFFFF", width=2 * 2)
            lum = (0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]) / 255
            tcol = "#FFFFFF" if lum < 0.55 else INK
            draw_text(draw, (xx + cw / 2, yy + ch / 2 - 10), f"{v:.2f}×",
                      F(21, bold=True), fill=tcol, anchor="mm")
            draw_text(draw, (xx + cw / 2, yy + ch / 2 + 20), f"{100*(v-1):+.1f}%",
                      F(12), fill=tcol, anchor="mm")
    cb_x, cb_y, cb_w, cb_h = 260, 610, 1250, 24
    for k in range(cb_w):
        v = 0.95 + (2.1 - 0.95) * k / cb_w
        draw.line([P(cb_x + k, cb_y), P(cb_x + k, cb_y + cb_h)], fill=speed_color(v), width=1 * 2)
    draw.rectangle(box(cb_x, cb_y, cb_x + cb_w, cb_y + cb_h), outline=AXIS, width=1 * 2)
    for v, lab in [(0.95, "0.95×"), (1.0, "1.00×"), (1.5, "1.50×"), (2.0, "2.00×")]:
        xx = cb_x + (v - 0.95) / (2.1 - 0.95) * cb_w
        draw_text(draw, (xx, cb_y + cb_h + 8), lab, F(12), fill=MUTED, anchor="ma")
    draw_text(draw, (cb_x, cb_y - 22), "green = faster / flat; red = slower at 16k", F(12), fill=MUTED)
    draw_text(draw, (70, 775),
              "B=1/8 ratios stay near 1.00×; B=32–256 show the range effect clearly. multi_sink remains flat at every batch size.",
              F(14), fill=MUTED)
    p = out_dir / "fig5_speedup_heatmap.png"
    save(img, p); figs["fig5"] = p

    # --- Figure 6: bars W1k vs W16k for B=128 and B=256 ----------------------
    img = new_canvas(1800, 900)
    draw = ImageDraw.Draw(img)
    draw_text(draw, (70, 34), "The 1k → 16k contrast at large batch", F(28, bold=True), fill=INK)
    draw_text(draw, (70, 76),
              "Bars are median latency over 12 seeds; labels show the W=16k vs W=1k change.",
              F(15), fill=MUTED)
    draw.rectangle(box(70, 108, 94, 126), fill="#9CA3AF", outline="#9CA3AF")
    draw_text(draw, (104, 111), "W = 1k", F(13), fill=INK)
    draw.rectangle(box(210, 108, 234, 126), fill="#EF4444", outline="#EF4444")
    draw_text(draw, (244, 111), "W = 16k", F(13), fill=INK)
    for panel_i, b in enumerate([128, 256]):
        bx0 = 60 + panel_i * 880
        bx1 = bx0 + 820
        by0, by1 = 165, 805
        left, right, top, bottom = 90, 30, 45, 95
        px0, py0, px1, py1 = bx0 + left, by0 + top, bx1 - right, by1 - bottom
        vals1 = [float(pv.loc[(b, sc, 1024), "latency_median_ms"]) for sc in SCENE_ORDER]
        vals2 = [float(pv.loc[(b, sc, 16384), "latency_median_ms"]) for sc in SCENE_ORDER]
        ymax = max(vals1 + vals2) * 1.18
        for k in range(5):
            v = ymax * k / 4
            yy = py1 - v / ymax * (py1 - py0)
            draw.line([P(px0, yy), P(px1, yy)], fill=GRID, width=1 * 2)
            draw_text(draw, (px0 - 12, yy), f"{v:.2f}", F(12), fill=MUTED, anchor="rm")
        draw.line([P(px0, py0), P(px0, py1)], fill=AXIS, width=1 * 2)
        draw.line([P(px0, py1), P(px1, py1)], fill=AXIS, width=1 * 2)
        group_w = (px1 - px0) / 3.0
        bw = group_w * 0.27
        for i, sc in enumerate(SCENE_ORDER):
            gx = px0 + (i + 0.5) * group_w
            for off, val, col in [(-bw * 0.55, vals1[i], "#9CA3AF"),
                                  (bw * 0.55, vals2[i], "#EF4444")]:
                xx = gx + off
                yy = py1 - val / ymax * (py1 - py0)
                draw.rectangle(box(xx - bw / 2, yy, xx + bw / 2, py1), fill=col, outline=col)
                draw_text(draw, (xx, yy - 8), f"{val:.3f}", F(11, bold=True),
                          fill=INK, anchor="mb")
            pct = 100 * (vals2[i] / vals1[i] - 1)
            draw_text(draw, (gx, py1 + 16), SCENE_SHORT[sc], F(12), fill=MUTED, anchor="ma")
            draw_text(draw, (gx, py1 + 38), f"{pct:+.1f}%", F(12, bold=True),
                      fill=("#B91C1C" if pct > 5 else "#047857"), anchor="ma")
        draw_text(draw, (bx0 + 16, by0 + 12), f"B = {b}", F(17, bold=True), fill=INK)
        draw_text(draw, (bx0 + 16, by0 + 34), "Latency (ms)", F(11), fill=MUTED)
        draw_text(draw, ((px0 + px1) / 2, by1 - 20), "Selection scene",
                  F(12), fill=MUTED, anchor="ma")
    p = out_dir / "fig6_w1k_vs_w16k_bars.png"
    save(img, p); figs["fig6"] = p
    return figs