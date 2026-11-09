# -*- coding: utf-8 -*-
"""One large figure for the B=128 dashboard (first sheet of GBSA_128K_B128_Dashboard.xlsx).

Left: the latency-vs-range lines with the 12 table values.
Right: the three spatial-distribution patterns (why the same 512 regions cost
different amounts of time) with page-footprint numbers.
Bottom: one-sentence conclusion.

Everything is measured before it is drawn; text boxes are checked for overlap.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import openpyxl
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "GBSA_128K_B128_Dashboard.xlsx"
OUT = ROOT / "GBSA_128K_B128_Overview.png"

FONT_R = r"C:\Windows\Fonts\msyh.ttc"
FONT_B = r"C:\Windows\Fonts\msyhbd.ttc"

W, H = 3000, 1650
INK = (17, 24, 39)
INK2 = (31, 41, 55)
MUTED = (107, 114, 128)
MUTED2 = (156, 163, 175)
GRID = (232, 235, 239)
TRACK = (244, 246, 248)
WHITE = (255, 255, 255)
BLUE = (37, 99, 235)
GREEN = (22, 163, 74)
RED = (220, 38, 38)
COL = {"recent_contiguous": BLUE, "multi_sink": GREEN, "uniform": RED}

_fc: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}


def F(px: float, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = (int(round(px)), bold)
    if key not in _fc:
        _fc[key] = ImageFont.truetype(FONT_B if bold else FONT_R, key[0])
    return _fc[key]


BOXES: list[tuple[tuple[float, float, float, float], str]] = []
SEGS: list[tuple[tuple[float, float], tuple[float, float]]] = []
MARKS: list[tuple[float, float, float]] = []


def T(d, xy, s, px, fill, anchor="la", bold=False, halo=0):
    f = F(px, bold)
    if halo:
        d.text(xy, s, font=f, fill=fill, anchor=anchor,
               stroke_width=int(halo), stroke_fill=WHITE)
    d.text(xy, s, font=f, fill=fill, anchor=anchor)
    bb = d.textbbox(xy, s, font=f, anchor=anchor)
    BOXES.append((bb, s))


def rrect(d, box, r, fill=None, outline=None, width=3):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


# --------------------------------------------------------------------------- #
# data from the workbook
# --------------------------------------------------------------------------- #
def load():
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    ws = wb["_data"]
    rows = {}
    for r in ws.iter_rows(min_row=2, values_only=True):
        rows[r[0]] = {"recent_contiguous": r[1], "multi_sink": r[2], "uniform": r[3]}
    lat = {}
    for k, v in rows.items():
        for sc, val in v.items():
            lat[(sc, k)] = float(val)
    dash = wb["B128耗时对比"]
    # page footprint comes from the same run set, carried in the dashboard's own table region
    t = {}
    for r in range(9, 12):
        t[dash.cell(r, 1).value] = [dash.cell(r, c).value for c in range(2, 6)]
    return lat, t


LAT, _ = load()
RANGES = ["1k", "4k", "8k", "16k"]
SCENES = ["recent_contiguous", "multi_sink", "uniform"]
PAGES = {  # per-request pages, batch union pages (from the same 720-run dataset)
    "recent_contiguous": ((32.9, 32.9), (65, 1022)),
    "multi_sink": ((40.0, 39.0), (40, 39)),
    "uniform": ((64.5, 408.1), (65, 1024)),
}


def stats(sc):
    a, b = LAT[(sc, "1k")], LAT[(sc, "16k")]
    return a, b, (b / a - 1) * 100, b / a


# --------------------------------------------------------------------------- #
# pattern strips
# --------------------------------------------------------------------------- #
N_REGION = 16384
TOPK = 512


def draws_positions(sc: str, seed: int, rows: int = 8):
    rng = np.random.default_rng(seed)
    out = []
    if sc == "recent_contiguous":
        for _ in range(rows):
            s = int(rng.integers(0, N_REGION - TOPK))
            out.append([(s, TOPK)])
    elif sc == "multi_sink":
        bands, bw = 8, TOPK // 8
        starts = [b * (N_REGION // bands) for b in range(bands)]
        for _ in range(rows):
            out.append([(s, bw) for s in starts])
    else:
        for _ in range(rows):
            pos = rng.integers(0, N_REGION, TOPK)
            out.append([(int(p), 1) for p in pos])
    return out


def strip(d, box, sc, seed, rows=8):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    rrect(d, box, 12, fill=TRACK, outline=(231, 234, 239), width=2)
    gap = 4
    rh = (h - 2 * 8 - gap * (rows - 1)) / rows
    lay = Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 0))
    ld = ImageDraw.Draw(lay)
    col = COL[sc] + ((88 if sc == "uniform" else 205),)
    for i, spans in enumerate(draws_positions(sc, seed, rows)):
        ry0 = 8 + i * (rh + gap)
        ry1 = ry0 + rh
        for st, ln in spans:
            px0 = st / N_REGION * w
            px1 = (st + ln) / N_REGION * w
            if sc == "uniform":
                ld.rectangle([px0, ry0, px0 + 1.1, ry1], fill=col)
            else:
                ld.rectangle([px0, ry0, max(px1, px0 + 3), ry1], fill=col)
    base = Image.new("RGBA", (int(w), int(h)), TRACK + (255,))
    img = Image.alpha_composite(base, lay)
    return img


# --------------------------------------------------------------------------- #
def build():
    img = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(img)

    # ---------------- title ----------------
    T(d, (100, 58), "GBSA 128K / top-512 · B = 128：候选范围 1k → 16k 的耗时", 58, INK, "la", True)
    T(d, (100, 138),
      "同一 batch 内 128 个 request；topk = 512、counts = 513 与 KV 形状全部固定；完整 128K BF16 KV（K+V 384 MiB）；"
      "12 selection seeds × 50 次 NPU-event 计时；只计 GBSA 核心算子", 26, MUTED)

    # ---------------- chart ----------------
    px0, px1 = 250, 1600
    py0, py1 = 405, 1275
    xs = [px0 + 55 + i * (px1 - px0 - 110) / 3 for i in range(4)]
    vmin, vmax = 0.0, 1.2

    def sy(v):
        return py1 - (v - vmin) / (vmax - vmin) * (py1 - py0)

    T(d, (100, 236), "B = 128 各候选范围中位耗时", 30, INK2, "la", True)
    # legend
    lx = 100
    for sc in SCENES:
        d.ellipse([lx, 286, lx + 17, 303], fill=COL[sc])
        name = sc
        T(d, (lx + 26, 282), name, 25, MUTED)
        lx += 26 + d.textlength(name, font=F(25)) + 40

    # grid + y ticks
    v = vmin
    while v <= vmax + 1e-9:
        yy = sy(v)
        d.line([(px0, yy), (px1, yy)], fill=GRID, width=2)
        T(d, (px0 - 16, yy), f"{v:.1f}", 23, MUTED, "rm")
        v = round(v + 0.2, 4)
    d.line([(px0, py1), (px1, py1)], fill=(203, 209, 217), width=3)

    # x labels
    for i, rg in enumerate(RANGES):
        T(d, (xs[i], py1 + 20), rg, 34, INK2, "ma", True)
        T(d, (xs[i], py1 + 66), {"1k": "8K", "4k": "32K", "8k": "64K", "16k": "128K"}[rg] + " tokens",
          22, MUTED2, "ma")

    # y axis title
    yt = Image.new("RGBA", (700, 44), (0, 0, 0, 0))
    ImageDraw.Draw(yt).text((0, 0), "中位耗时（ms）", font=F(24), fill=MUTED)
    yt = yt.crop(yt.getbbox()).rotate(90, expand=True)
    img.paste(yt, (118, int((py0 + py1) / 2 - yt.height / 2)), yt)

    # series
    order = ["multi_sink", "recent_contiguous", "uniform"]
    for sc in order:
        vals = [LAT[(sc, rg)] for rg in RANGES]
        pts = [(xs[i], sy(v)) for i, v in enumerate(vals)]
        d.line(pts, fill=COL[sc], width=9, joint="curve")
        for (x_, y_) in pts:
            d.ellipse([x_ - 12, y_ - 12, x_ + 12, y_ + 12], fill=WHITE, outline=COL[sc], width=5)
        SEGS.extend([(pts[i], pts[i + 1]) for i in range(len(pts) - 1)])
        for (x_, y_) in pts:
            MARKS.append((x_, y_, 15))

    # point labels (white halo so a steep segment behind them stays legible)
    def lab(sc, rg, dx=14, dy=-22, anchor="ls", size=23):
        x_ = xs[RANGES.index(rg)]
        s = f"{LAT[(sc, rg)]:.3f}"
        T(d, (x_ + dx, sy(LAT[(sc, rg)]) + dy), s, size, COL[sc], anchor, True, halo=6)

    lab("uniform", "1k", dx=16, dy=26, anchor="la")
    lab("uniform", "4k")
    lab("uniform", "8k")
    lab("uniform", "16k", dx=-16, dy=-24, anchor="rs")
    lab("recent_contiguous", "4k")
    lab("recent_contiguous", "8k")
    lab("recent_contiguous", "16k", dx=-16, dy=-24, anchor="rs")
    lab("recent_contiguous", "1k", dx=-8, dy=28, anchor="ma")
    T(d, (xs[2], sy(LAT[("multi_sink", "8k")]) + 34), "0.375 全程持平", 22, GREEN, "ma", True, halo=6)

    # endpoint deltas
    for sc, dy in [("uniform", 0), ("recent_contiguous", 2), ("multi_sink", 0)]:
        _, _, pct, ratio = stats(sc)
        txt = f"{pct:+.1f}%"
        T(d, (px1 + 42, sy(LAT[(sc, "16k")]) + dy), txt.replace("-", "\u2212"), 34, COL[sc], "lm", True)

    # 1k overlap note
    T(d, (xs[0] + 4, sy(0.30) + 26), "1k 处两场景几乎重合", 19, MUTED2, "ma")

    # ---------------- right panel ----------------
    RX0, RX1 = 1860, 2900
    T(d, (RX0, 236), "同样 512 个 region，落在多少物理页？", 29, INK2, "la", True)
    T(d, (RX0, 280), "每行 = 1 个请求的 top-512 落点（16,384 region 窗口内，示意 8 行）", 20, MUTED)

    card_y = [330, 686, 1042]
    ch = 316
    for sc, cy in zip(SCENES, card_y):
        a, b, pct, ratio = stats(sc)
        rrect(d, (RX0, cy, RX1, cy + ch), 16, fill=WHITE, outline=(229, 231, 235), width=3)
        d.ellipse([RX0 + 26, cy + 30, RX0 + 46, cy + 50], fill=COL[sc])
        T(d, (RX0 + 58, cy + 22), sc, 27, INK, "la", True)
        T(d, (RX1 - 26, cy + 20), f"{pct:+.1f}%".replace("-", "\u2212"), 40, COL[sc], "ra", True)
        T(d, (RX0 + 26, cy + 66), f"{a:.4f} → {b:.4f} ms   ·   延迟比 {ratio:.2f}×", 22, MUTED)
        st = strip(d, (RX0 + 26, cy + 108, RX1 - 26, cy + 222), sc, seed=7000)
        img.paste(st, (RX0 + 26, cy + 108), st)
        pr, un = PAGES[sc]
        T(d, (RX0 + 26, cy + 240),
          f"页/请求 {pr[0]:.1f} → {pr[1]:.1f}      batch 并集 {un[0]} → {un[1]} 页", 21, INK2)

    T(d, (RX0, 1370), "1,024 页 = 全部 384 MiB KV cache；越接近它，L2 复用越差", 20, MUTED2)

    # ---------------- bottom conclusion ----------------
    rrect(d, (100, 1424, 2900, 1548), 14, fill=(248, 250, 252), outline=(219, 234, 254), width=3)
    d.rectangle([100, 1424, 106, 1548], fill=BLUE)
    T(d, (140, 1442),
      "只把候选范围从 1k 放宽到 16k、topk 仍是 512：recent_contiguous +62.6%，uniform +96.2%，multi_sink −0.3%。",
      26, INK2)
    T(d, (140, 1492),
      "真正的成本是物理页工作集：同样 512 个 region，落在 39 页 0.3741 ms、落在 1,024 页 1.0427 ms（2.79×）——"
      "限范围只是必要条件，还必须同时约束分散度。", 26, INK2)

    T(d, (100, 1580),
      "数据源：GBSA_128K_B128_Dashboard.xlsx / results/raw_runs.csv（B=128，每场景 4 个范围 × 12 seeds；取 12 个 seed 的中位数）",
      19, MUTED2)

    img = img.convert("RGB")
    return img


def check(img):
    print("canvas", img.size)
    print("text blocks:", len(BOXES))
    bad = 0
    for i in range(len(BOXES)):
        b1, s1 = BOXES[i]
        for j in range(i + 1, len(BOXES)):
            b2, s2 = BOXES[j]
            ox = min(b1[2], b2[2]) - max(b1[0], b2[0])
            oy = min(b1[3], b2[3]) - max(b1[1], b2[1])
            if ox > 1 and oy > 1:
                print(f"  text overlap {ox*oy:6.0f}px2  {s1!r} <> {s2!r}")
                bad += 1

    def seg_hits(bb, p, q, n=400):
        hits = 0
        for k in range(n + 1):
            t = k / n
            x = p[0] + (q[0] - p[0]) * t
            y = p[1] + (q[1] - p[1]) * t
            if bb[0] - 1 <= x <= bb[2] + 1 and bb[1] - 1 <= y <= bb[3] + 1:
                hits += 1
        return hits

    for bb, s in BOXES:
        if bb[0] < 60 or bb[2] > W - 60 or bb[3] > H - 30:
            print(f"  margin  {bb}  {s!r}")
            bad += 1
        for p, q in SEGS:
            h = seg_hits(bb, p, q)
            if h:
                print(f"  label-on-line {h:4d} samples  {s!r}")
                bad += 1
        for cx, cy, r in MARKS:
            if bb[0] - r < cx < bb[2] + r and bb[1] - r < cy < bb[3] + r:
                print(f"  label-on-marker  {s!r}")
                bad += 1
    print("issues:", bad)


if __name__ == "__main__":
    im = build()
    im.save(OUT, dpi=(300, 300))
    check(im)
    print("saved", OUT, im.size)
