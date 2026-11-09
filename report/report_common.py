#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared data, palette, font and drawing helpers for the GBSA report."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results"
OUT = ROOT
FIG = OUT / "figures"

SCENE_ORDER = ["recent_contiguous", "multi_sink", "uniform"]
SCENE_ZH = {
    "recent_contiguous": "recent contiguous",
    "multi_sink": "multi sink",
    "uniform": "uniform",
}
SCENE_SHORT = {
    "recent_contiguous": "recent_contiguous",
    "multi_sink": "multi_sink",
    "uniform": "uniform",
}
COLORS = {
    "recent_contiguous": "#2563EB",
    "multi_sink": "#16A34A",
    "uniform": "#DC2626",
}
COLORS_RGB = {k: tuple(int(v[i:i + 2], 16) for i in (1, 3, 5)) for k, v in COLORS.items()}
INK = "#111827"
MUTED = "#6B7280"
GRID = "#E5E7EB"
AXIS = "#9CA3AF"
BG = "#FFFFFF"
PANEL_BG = "#F9FAFB"

S = 2

FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/Deng.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
BOLD_CANDIDATES = [
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/Dengb.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

def _first_existing(paths: list[str]) -> str | None:
    for p in paths:
        if Path(p).exists():
            return p
    return None


FONT_REG_PATH = _first_existing(FONT_CANDIDATES)
FONT_BOLD_PATH = _first_existing(BOLD_CANDIDATES) or FONT_REG_PATH


def F(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = FONT_BOLD_PATH if bold else FONT_REG_PATH
    try:
        if path:
            return ImageFont.truetype(path, size * S)
    except Exception:
        pass
    try:
        return ImageFont.load_default(size=size * S)
    except Exception:
        return ImageFont.load_default()


def P(x: float, y: float) -> tuple[int, int]:
    return (int(round(x * S)), int(round(y * S)))


def box(x0: float, y0: float, x1: float, y1: float) -> tuple[int, int, int, int]:
    return (int(round(x0 * S)), int(round(y0 * S)), int(round(x1 * S)), int(round(y1 * S)))


def new_canvas(w: int = 1800, h: int = 1050) -> Image.Image:
    return Image.new("RGB", (w * S, h * S), BG)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[float, float]:
    b = draw.textbbox((0, 0), text, font=font)
    return ((b[2] - b[0]) / S, (b[3] - b[1]) / S)


def human_range(w: int) -> str:
    return f"{w // 1024}k"


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str,
              font: ImageFont.ImageFont, fill: str = INK, anchor: str = "la") -> None:
    draw.text(P(*xy), text, font=font, fill=fill, anchor=anchor)


def draw_rounded_box(draw: ImageDraw.ImageDraw, xy: tuple[float, float, float, float],
                     fill: str = "#FFFFFF", outline: str = GRID, radius: int = 10,
                     width: int = 1) -> None:
    draw.rounded_rectangle(box(*xy), radius=radius * S, fill=fill, outline=outline, width=width * S)


def draw_legend(img: Image.Image, items: list[tuple[str, str]], x: float, y: float,
                font_size: int = 15, gap: float = 34) -> None:
    draw = ImageDraw.Draw(img, "RGBA")
    cx = x
    font = F(font_size)
    for name, color in items:
        draw.line([P(cx, y + 8), P(cx + 30, y + 8)], fill=color, width=int(3 * S))
        draw.ellipse(box(cx + 11, y + 4, cx + 19, y + 12), fill=color, outline=BG, width=1 * S)
        draw_text(draw, (cx + 38, y), name, font, fill=INK)
        w, _ = text_size(draw, name, font)
        cx += 38 + w + gap


def draw_line_panel(img: Image.Image, bbox: tuple[float, float, float, float],
                    xlabels: list[str], series: list[dict[str, Any]],
                    title: str = "", y_unit: str = "", show_xlabel: bool = True,
                    y_limits: tuple[float, float] | None = None,
                    hline: tuple[float, str, str] | None = None,
                    annotate_last: bool = False) -> None:
    draw = ImageDraw.Draw(img, "RGBA")
    x0, y0, x1, y1 = bbox
    left, right, top, bottom = 80, 25, 42, 60
    px0, py0, px1, py1 = x0 + left, y0 + top, x1 - right, y1 - bottom
    draw_rounded_box(draw, (x0, y0, x1, y1), fill="#FFFFFF", outline="#E5E7EB", radius=12)

    all_vals = []
    for s in series:
        all_vals.extend(s["values"])
        all_vals.extend(s.get("q1", []))
        all_vals.extend(s.get("q3", []))
    all_vals = [float(v) for v in all_vals
                if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not all_vals:
        return
    if y_limits is None:
        vmin, vmax = min(all_vals), max(all_vals)
        span = max(vmax - vmin, vmax * 0.03, 1e-4)
        ymin = max(0.0, vmin - span * 0.12)
        ymax = vmax + span * 0.14
    else:
        ymin, ymax = y_limits

    def xpos(i: int) -> float:
        if len(xlabels) <= 1:
            return (px0 + px1) / 2
        pad = 35
        return px0 + pad + i * (px1 - px0 - 2 * pad) / (len(xlabels) - 1)

    def ypos(v: float) -> float:
        if ymax <= ymin:
            return (py0 + py1) / 2
        return py1 - (v - ymin) * (py1 - py0) / (ymax - ymin)

    for k in range(5):
        v = ymin + (ymax - ymin) * k / 4.0
        y = ypos(v)
        draw.line([P(px0, y), P(px1, y)], fill=GRID, width=1 * S)
        txt = f"{v:.3f}" if ymax - ymin < 0.1 else f"{v:.2f}"
        draw_text(draw, (px0 - 12, y), txt, F(13), fill=MUTED, anchor="rm")

    draw.line([P(px0, py0), P(px0, py1)], fill=AXIS, width=1 * S)
    draw.line([P(px0, py1), P(px1, py1)], fill=AXIS, width=1 * S)

    if hline is not None:
        hv, hcolor, htext = hline
        if ymin <= hv <= ymax:
            yy = ypos(hv)
            for xd in range(int(px0), int(px1), 14):
                draw.line([P(xd, yy), P(min(xd + 7, px1), yy)], fill=hcolor, width=2 * S)
            draw_text(draw, (px1 - 4, yy - 6), htext, F(12), fill=hcolor, anchor="rs")

    for s in series:
        color = tuple(int(s["color"][i:i + 2], 16) for i in (1, 3, 5))
        if "q1" in s and "q3" in s and s["q1"] and s["q3"]:
            pts = [(xpos(i), ypos(float(s["q3"][i]))) for i in range(len(xlabels))]
            pts += [(xpos(i), ypos(float(s["q1"][i]))) for i in range(len(xlabels) - 1, -1, -1)]
            draw.polygon([P(x, y) for x, y in pts], fill=(*color, 38))

    for s in series:
        color = s["color"]
        pts = [(xpos(i), ypos(float(s["values"][i]))) for i in range(len(xlabels))]
        draw.line([P(x, y) for x, y in pts], fill=color, width=int(3 * S), joint="curve")
        for x, y in pts:
            draw.ellipse(box(x - 5.5, y - 5.5, x + 5.5, y + 5.5),
                         fill=color, outline="#FFFFFF", width=1 * S)
        if annotate_last:
            x, y = pts[-1]
            draw_text(draw, (x - 8, y - 18), f"{float(s['values'][-1]):.3f}",
                      F(12, bold=True), fill=color, anchor="rm")

    if show_xlabel:
        for i, lab in enumerate(xlabels):
            draw_text(draw, (xpos(i), py1 + 14), lab, F(13), fill=MUTED, anchor="ma")
        draw_text(draw, ((px0 + px1) / 2, y1 - 22),
                  "Candidate range (logical regions; 1 region = 8 tokens)",
                  F(12), fill=MUTED, anchor="ma")

    if title:
        draw_text(draw, (x0 + 16, y0 + 12), title, F(16, bold=True), fill=INK)
    if y_unit:
        draw_text(draw, (x0 + 16, y0 + 30), y_unit, F(11), fill=MUTED)


def save(img: Image.Image, path: Path) -> None:
    img = img.resize((img.width // S, img.height // S), Image.Resampling.LANCZOS)
    img.save(path, format="PNG", optimize=True)


def load_data(data_dir: Path) -> dict[str, Any]:
    raw = pd.read_csv(data_dir / "raw_runs.csv")
    timings = pd.read_csv(data_dir / "raw_timings.csv")
    meta = pd.read_csv(data_dir / "metadata_bench.csv")
    config = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
    env = json.loads((data_dir / "environment.json").read_text(encoding="utf-8"))
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    raw["out_finite"] = raw["out_finite"].astype(str).str.lower().isin(["true", "1", "yes"])
    raw["range_label"] = raw["range_regions_requested"].map(human_range)
    raw["scene_zh"] = raw["scene"].map(SCENE_ZH)
    return {"raw": raw, "timings": timings, "meta": meta, "config": config,
            "env": env, "manifest": manifest}


def make_summaries(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    g = raw.groupby(["batch_size", "scene", "range_regions_requested"], as_index=False)
    summary = g.agg(
        n_runs=("run_id", "count"),
        latency_median_ms=("latency_ms_median", "median"),
        latency_q25_ms=("latency_ms_median", lambda x: x.quantile(0.25)),
        latency_q75_ms=("latency_ms_median", lambda x: x.quantile(0.75)),
        latency_mean_ms=("latency_ms_median", "mean"),
        latency_std_ms=("latency_ms_median", "std"),
        latency_p90_sample_ms=("latency_ms_p90", "median"),
        span_median_regions=("selected_span_mean_regions", "median"),
        per_request_pages_median=("touched_pages_mean", "median"),
        union_pages_median=("touched_pages_union", "median"),
        working_set_mib_median=("kv_working_set_mib", "median"),
        logical_union_span_median=("logical_union_span", "median"),
        out_abs_mean_median=("out_abs_mean", "median"),
    )
    base = summary[summary.range_regions_requested == 1024][
        ["batch_size", "scene", "latency_median_ms", "union_pages_median",
         "per_request_pages_median", "working_set_mib_median"]
    ].rename(columns={
        "latency_median_ms": "latency_w1k_ms",
        "union_pages_median": "union_w1k",
        "per_request_pages_median": "per_request_pages_w1k",
        "working_set_mib_median": "working_set_w1k_mib",
    })
    summary = summary.merge(base, on=["batch_size", "scene"], how="left")
    summary["delta_ms_vs_w1k"] = summary["latency_median_ms"] - summary["latency_w1k_ms"]
    summary["speedup_x_vs_w1k"] = summary["latency_median_ms"] / summary["latency_w1k_ms"]
    summary["delta_pct_vs_w1k"] = 100.0 * (summary["speedup_x_vs_w1k"] - 1.0)
    summary["scene_zh"] = summary["scene"].map(SCENE_ZH)
    summary["range_label"] = summary["range_regions_requested"].map(human_range)

    contrast = summary[summary.range_regions_requested == 16384].copy()
    contrast = contrast[[
        "batch_size", "scene", "scene_zh", "latency_w1k_ms", "latency_median_ms",
        "delta_ms_vs_w1k", "speedup_x_vs_w1k", "delta_pct_vs_w1k",
        "per_request_pages_w1k", "per_request_pages_median",
        "union_w1k", "union_pages_median", "working_set_w1k_mib",
        "working_set_mib_median",
    ]].rename(columns={
        "latency_median_ms": "latency_w16k_ms",
        "per_request_pages_median": "per_request_pages_w16k",
        "union_pages_median": "union_w16k",
        "working_set_mib_median": "working_set_w16k_mib",
    })

    rows = []
    for b in sorted(raw.batch_size.unique()):
        sub_b = raw[raw.batch_size == b]
        subsets = [("all_scenes", sub_b)] + [(s, sub_b[sub_b.scene == s]) for s in SCENE_ORDER]
        for label, sub in subsets:
            if len(sub) < 3:
                continue

            def sp(a: str) -> float | None:
                if sub[a].nunique() < 2:
                    return None
                try:
                    return float(sub[[a, "latency_ms_median"]].corr(method="spearman").iloc[0, 1])
                except Exception:
                    return None

            rows.append({
                "batch_size": int(b),
                "subset": label,
                "n_runs": int(len(sub)),
                "spearman_span_vs_latency": sp("selected_span_mean_regions"),
                "spearman_per_request_pages_vs_latency": sp("touched_pages_mean"),
                "spearman_union_pages_vs_latency": sp("touched_pages_union"),
                "spearman_working_set_vs_latency": sp("kv_working_set_mib"),
            })
    corr = pd.DataFrame(rows)
    return summary, contrast, corr


def md_table(df: pd.DataFrame, cols: list[str], headers: list[str],
             fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if c in fmt:
                vals.append(fmt[c].format(v))
            elif isinstance(v, (float, np.floating)):
                vals.append(f"{float(v):.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)