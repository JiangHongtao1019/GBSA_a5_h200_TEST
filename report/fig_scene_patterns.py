#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw a polished compact figure for the three controlled top-512 patterns."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bench import make_scene_ids  # noqa: E402

from report_common import (  # noqa: E402
    COLORS, COLORS_RGB, F, GRID, INK, MUTED, P, box, draw_rounded_box,
    draw_text, new_canvas, save,
)

SCENES = ["recent_contiguous", "multi_sink", "uniform"]
TITLES = {
    "recent_contiguous": "recent_contiguous",
    "multi_sink": "multi_sink",
    "uniform": "uniform",
}
SUBTITLES = {
    "recent_contiguous": "1 段连续带，请求间位置不同",
    "multi_sink": "8 个共享热点带",
    "uniform": "512 点均匀散开",
}
TAGS = {
    "recent_contiguous": "512 连续",
    "multi_sink": "8×64 共享",
    "uniform": "512 分散",
}

NOTES = {
    "recent_contiguous": "请求之间位置不同",
    "multi_sink": "所有请求共用同一组热点",
    "uniform": "512 个点铺满整个窗口",
}


def _rgb_light(rgb: tuple[int, int, int], amount: float = 0.90) -> tuple[int, int, int]:
    return tuple(int(255 - (255 - c) * amount) for c in rgb)


def _xmap(rid: float, n: int, x0: float, x1: float) -> float:
    return x0 + float(rid) / float(n) * (x1 - x0)


def make_scene_pattern_figure(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 16384
    topk = 512
    ids = {
        sc: make_scene_ids(
            scene=sc, candidate_start=0, candidate_regions=n,
            batch_size=3, groups=4, topk=topk, seed=7000,
        )
        for sc in SCENES
    }

    img = new_canvas(1800, 620)
    draw = ImageDraw.Draw(img, "RGBA")

    # header
    draw_text(draw, (46, 20), "同样的 512 个 region，空间分布不同", F(36, bold=True), fill=INK)
    draw_text(draw, (48, 64),
              "受控 IndexR-like score → top-512；每行一个请求（group 0）；横轴为 16k region 候选窗口",
              F(17), fill=MUTED)
    draw_rounded_box(draw, (1424, 22, 1754, 64), fill="#0F172A", outline="#0F172A", radius=14)
    draw_text(draw, (1589, 43), "topk = 512   |   W = 16k", F(15, bold=True), fill="#FFFFFF", anchor="mm")

    card_w = 544
    card_x = [40, 628, 1216]
    card_y0, card_y1 = 108, 548

    for si, sc in enumerate(SCENES):
        x0 = card_x[si]
        x1 = x0 + card_w
        rgb = COLORS_RGB[sc]
        light = _rgb_light(rgb, 0.93)
        mid = _rgb_light(rgb, 0.84)

        # soft shadow + card
        draw_rounded_box(draw, (x0 + 3, card_y0 + 4, x1 + 3, card_y1 + 4), fill="#E2E8F0", outline="#E2E8F0", radius=16)
        draw_rounded_box(draw, (x0, card_y0, x1, card_y1), fill="#FFFFFF", outline="#E2E8F0", radius=16)

        # accent and title
        draw_rounded_box(draw, (x0 + 20, card_y0 + 20, x0 + 26, card_y0 + 52), fill=COLORS[sc], outline=COLORS[sc], radius=3)
        draw_text(draw, (x0 + 38, card_y0 + 17), TITLES[sc], F(24, bold=True), fill=INK)
        draw_text(draw, (x0 + 38, card_y0 + 47), SUBTITLES[sc], F(15), fill=MUTED)

        # tag pill
        tag_w = 132 if sc != "uniform" else 146
        draw_rounded_box(draw, (x1 - tag_w - 20, card_y0 + 17, x1 - 20, card_y0 + 43), fill=COLORS[sc], outline=COLORS[sc], radius=13)
        draw_text(draw, (x1 - tag_w / 2 - 20, card_y0 + 30), TAGS[sc], F(14, bold=True), fill="#FFFFFF", anchor="mm")

        # plot area
        px0, px1 = x0 + 36, x1 - 28
        py0, py1 = card_y0 + 76, card_y0 + 360
        draw_rounded_box(draw, (px0, py0, px1, py1), fill="#F8FAFC", outline="#E2E8F0", radius=8)
        for tick in [0, 4096, 8192, 12288, 16384]:
            xx = _xmap(tick, n, px0 + 22, px1 - 8)
            draw.line([P(xx, py0 + 8), P(xx, py1 - 2)], fill="#E2E8F0", width=1 * 2)
        for tick, lab in [(0, "0"), (4096, "4k"), (8192, "8k"), (12288, "12k"), (16384, "16k")]:
            xx = _xmap(tick, n, px0 + 22, px1 - 8)
            draw_text(draw, (xx, py1 + 8), lab, F(12), fill=MUTED, anchor="ma")

        arr = ids[sc][0]
        centers = [card_y0 + 125, card_y0 + 195, card_y0 + 265]
        for ri, cy in enumerate(centers):
            draw_text(draw, (x0 + 20, cy - 5), f"R{ri}", F(11), fill=MUTED, anchor="lm")
            draw.line([P(px0 + 22, cy), P(px1 - 8, cy)], fill="#CBD5E1", width=1 * 2)
            row = arr[ri]
            if sc == "recent_contiguous":
                lo, hi = int(row.min()), int(row.max())
                xa = _xmap(lo, n, px0 + 22, px1 - 8)
                xb = _xmap(hi + 1, n, px0 + 22, px1 - 8)
                draw.rounded_rectangle(box(xa, cy - 10, max(xb, xa + 3), cy + 10), radius=3 * 2,
                                       fill=(*rgb, 215), outline=COLORS[sc], width=1 * 2)
            elif sc == "multi_sink":
                cuts = np.where(np.diff(row) > 1)[0] + 1
                runs = np.split(row, cuts)
                for run in runs:
                    xa = _xmap(int(run[0]), n, px0 + 22, px1 - 8)
                    xb = _xmap(int(run[-1]) + 1, n, px0 + 22, px1 - 8)
                    draw.rounded_rectangle(box(xa, cy - 12, max(xb, xa + 2.4), cy + 12), radius=2 * 2,
                                           fill=(*rgb, 230), outline=COLORS[sc], width=1 * 2)
            else:
                for rid in row:
                    xx = _xmap(int(rid), n, px0 + 22, px1 - 8)
                    jitter = ((int(rid) * 29) % 13 - 6) * 0.85
                    draw.ellipse(box(xx - 1.4, cy + jitter - 1.4, xx + 1.4, cy + jitter + 1.4),
                                 fill=(*rgb, 95))

        draw_text(draw, (x0 + 22, card_y1 - 30), NOTES[sc], F(13), fill=MUTED)

    draw_text(draw, (48, 576),
              "每个色块 / 点 = 1 个选中的 region；三者 topk 都是 512，区别只在 512 个 region 的空间位置。",
              F(14, bold=True), fill=INK)
    draw_text(draw, (48, 598),
              "构造：Gaussian score + 场景 pulse → top-512；受控 IndexR-like ids 示意，不是实际 IndexR 输出。",
              F(13), fill=MUTED)

    p = out_dir / "fig7_scene_topk_patterns.png"
    save(img, p)
    return p


if __name__ == "__main__":
    print(make_scene_pattern_figure(ROOT / "figures"))