# -*- coding: utf-8 -*-
"""Build the 2-page GBSA core-data PDF brief.

Pipeline: pandas -> Pillow charts (300 dpi) -> python-pptx A4 portrait
-> LibreOffice Kit PDF export.

Text is pre-wrapped with real font metrics so nothing overflows the fixed
layout; every text call reports a warning if it does not fit.
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pptx.oxml.ns import qn
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Emu, Mm, Pt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ASSETS = ROOT / "report" / "brief_assets"
OUT_PPTX = ROOT / "report" / "GBSA_128K_Core_Brief_2p.pptx"
OUT_PDF = ROOT / "GBSA_128K_Core_Brief_2p.pdf"

# --------------------------------------------------------------------------- #
# design tokens
# --------------------------------------------------------------------------- #
INK = "111827"
INK2 = "1F2937"
MUTED = "6B7280"
MUTED2 = "9CA3AF"
RULE = "E5E7EB"
PANEL = "F9FAFB"
NAVY = "0F172A"
NAVY2 = "1E293B"
BLUE = "2563EB"
GREEN = "16A34A"
RED = "DC2626"
AMBER = "B45309"
SLATE_L = "CBD5E1"
C_RECENT = "2563EB"
C_MULTI = "16A34A"
C_UNIFORM = "DC2626"

FONT = "Microsoft YaHei"
FONT_R = r"C:\Windows\Fonts\msyh.ttc"
FONT_B = r"C:\Windows\Fonts\msyhbd.ttc"

PAGE_W, PAGE_H = 210.0, 297.0
ML, MR = 14.0, 14.0
CONTENT_W = PAGE_W - ML - MR

WARN: list[str] = []

_font_cache: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}


def _pil_font(size_px: int, bold: bool) -> ImageFont.FreeTypeFont:
    key = (size_px, bold)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(FONT_B if bold else FONT_R, size_px)
    return _font_cache[key]


def tw_mm(text: str, size_pt: float, bold: bool = False) -> float:
    """Measured single-line width in mm (PIL size unit == px at 72 dpi == pt)."""
    f = _pil_font(max(8, int(round(size_pt * 4))), bold)
    return f.getlength(text) / 4.0 * 0.3527778


NO_LEAD = "，。、；：？！）】」』”’%,.;:?!)]}"
NO_TAIL = "（【「『“‘([{"
TOKEN_CHARS = set("._%+-,:×→/()[]=<>|~≈≥≤")


def _tokens(text: str) -> list[str]:
    """Split into atomic units: latin/number runs stay whole, CJK char by char."""
    out: list[str] = []
    buf = ""
    for ch in text:
        if (ch.isascii() and (ch.isalnum() or ch in TOKEN_CHARS)) or ch in "×→≈≥≤":
            buf += ch
        else:
            if buf:
                out.append(buf)
                buf = ""
            out.append(ch)
    if buf:
        out.append(buf)
    return out


def wrap_mm(text: str, max_mm: float, size_pt: float, bold: bool = False) -> list[str]:
    if not text:
        return [""]
    lines: list[str] = []
    cur = ""
    for tok in _tokens(text):
        cand = cur + tok
        if tw_mm(cand, size_pt, bold) <= max_mm or not cur.strip():
            cur = cand
            continue
        if tok.strip() and tok.strip()[0] in NO_LEAD:
            cur = cand  # never start a line with closing punctuation
            continue
        line = cur.rstrip()
        carry = ""
        if line and line[-1] in NO_TAIL:  # never end a line with an opening bracket
            carry = line[-1]
            line = line[:-1].rstrip()
        lines.append(line)
        cur = carry + tok.lstrip()
        while cur and tw_mm(cur, size_pt, bold) > max_mm and len(cur) > 1:
            k = len(cur)
            while k > 1 and tw_mm(cur[:k], size_pt, bold) > max_mm:
                k -= 1
            lines.append(cur[:k])
            cur = cur[k:]
    if cur.strip():
        lines.append(cur.rstrip())
    return lines or [""]


def num(s: str) -> str:
    """Typographic minus for negative numeric labels."""
    return s.replace("-", "\u2212")


def rgb(h: str) -> RGBColor:
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


# --------------------------------------------------------------------------- #
# pptx primitives
# --------------------------------------------------------------------------- #
def _style_run(run, size: float, bold: bool, color: str) -> None:
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = FONT
    run.font.color.rgb = rgb(color)
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.find("{http://schemas.openxmlformats.org/drawingml/2006/main}" + tag.split(":")[1])
        if el is None:
            el = rPr.makeelement(
                "{http://schemas.openxmlformats.org/drawingml/2006/main}" + tag.split(":")[1], {}
            )
            rPr.append(el)
        el.set("typeface", FONT)


def text(
    slide,
    x: float,
    y: float,
    w: float,
    s: str,
    size: float,
    color: str = INK,
    bold: bool = False,
    align: str = "l",
    leading: float | None = None,
    max_lines: int | None = None,
) -> float:
    """Place a single pre-wrapped text block; returns bottom y in mm."""
    lead = leading if leading is not None else size * 0.3527778 * 1.42
    lines = wrap_mm(s, w, size, bold)
    if max_lines and len(lines) > max_lines:
        WARN.append(f"too many lines ({len(lines)}>{max_lines}) at y={y}: {s[:40]}")
    for ln in lines:
        if tw_mm(ln, size, bold) > w + 0.25:
            WARN.append(f"overflow w={w} y={y} '{ln}'")
    box = slide.shapes.add_textbox(Mm(x), Mm(y), Mm(w), Mm(lead * len(lines) + 0.6))
    tf = box.text_frame
    tf.word_wrap = False
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.TOP
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = Pt(lead / 0.3527778)
        p.space_before = Pt(0)
        p.space_after = Pt(0)
        p.alignment = {"l": PP_ALIGN.LEFT, "r": PP_ALIGN.RIGHT, "c": PP_ALIGN.CENTER}[align]
        r = p.add_run()
        r.text = ln
        _style_run(r, size, bold, color)
    if y + lead * len(lines) > 288.5:
        WARN.append(f"past footer: y={y} lines={len(lines)}")
    return y + lead * len(lines)


def rect(slide, x, y, w, h, fill: str | None = None, line: str | None = None,
         line_w: float = 0.6, radius: float | None = None):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(shape_type, Mm(x), Mm(y), Mm(w), Mm(h))
    # drop the theme style reference so LibreOffice does not paint a shadow/effect
    style = shp._element.find(qn("p:style"))
    if style is not None:
        shp._element.remove(style)
    if radius:
        shp.adjustments[0] = radius
    if fill:
        shp.fill.solid()
        shp.fill.fore_color.rgb = rgb(fill)
    else:
        shp.fill.background()
    if line:
        shp.line.color.rgb = rgb(line)
        shp.line.width = Pt(line_w)
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    shp.text_frame.word_wrap = False
    return shp


def hline(slide, x, y, w, color: str = RULE, thick: float = 0.35):
    rect(slide, x, y, w, thick, fill=color)


def section(slide, x, y, w, idx: str, title: str, size: float = 9.5, x_end: float | None = None) -> float:
    """Section header: blue index + title + hairline to the right."""
    if idx:
        text(slide, x, y, 4 if idx.isdigit() else 9, idx, size, BLUE, True)
        x2 = x + tw_mm(idx, size, True) + 2.4
    else:
        x2 = x
    w_title = tw_mm(title, size, True)
    yb = text(slide, x2, y, w_title + 3, title, size, INK, True)
    rule_end = x_end if x_end is not None else x + w
    hline(slide, x2 + w_title + 4, y + size * 0.3527778 * 0.62, rule_end - (x2 + w_title + 4), RULE, 0.3)
    return yb


# --------------------------------------------------------------------------- #
# table renderer
# --------------------------------------------------------------------------- #
def table(
    slide,
    x: float,
    y: float,
    col_w: list[float],
    rows: list[list[str]],
    *,
    header: bool = True,
    header_fill: str = "F3F4F6",
    row_h: float = 7.2,
    header_h: float = 6.4,
    font: float = 8.0,
    header_font: float = 7.2,
    aligns: list[str] | None = None,
    zebra: str | None = PANEL,
    styles: dict[tuple[int, int], dict] | None = None,
    pad: float = 1.8,
    rule_color: str = RULE,
    expect_w: float | None = CONTENT_W,
) -> float:
    styles = styles or {}
    total_w = sum(col_w)
    if expect_w is not None and abs(total_w - expect_w) > 0.6:
        WARN.append(f"table width {total_w:.1f} != {expect_w:.1f}")
    aligns = aligns or ["l"] * len(col_w)
    cy = y
    for ri, row in enumerate(rows):
        is_head = header and ri == 0
        size = header_font if is_head else font
        lead = size * 0.3527778 * 1.34
        wrapped = [wrap_mm(c, col_w[ci] - 2 * pad, size, is_head or styles.get((ri, ci), {}).get("bold", False))
                   for ci, c in enumerate(row)]
        n_lines = max(len(w) for w in wrapped)
        h = max(header_h if is_head else row_h, n_lines * lead + 2.0)
        if is_head and header_fill:
            rect(slide, x, cy, total_w, h, fill=header_fill)
        elif not is_head and zebra and (ri % 2 == 0):
            rect(slide, x, cy, total_w, h, fill=zebra)
        for ci, cell_lines in enumerate(wrapped):
            st = styles.get((ri, ci), {})
            color = st.get("color", MUTED if is_head else INK2)
            bold = st.get("bold", is_head)
            cx = x + sum(col_w[:ci])
            for li, ln in enumerate(cell_lines):
                ty = cy + (h - len(cell_lines) * lead) / 2 + li * lead
                for ln in [ln]:
                    if tw_mm(ln, size, bold) > col_w[ci] - 2 * pad + 0.3:
                        WARN.append(f"cell overflow r{ri}c{ci}: '{ln}' ({tw_mm(ln, size, bold):.1f}>{col_w[ci]-2*pad:.1f})")
                tb = slide.shapes.add_textbox(Mm(cx + pad), Mm(ty), Mm(col_w[ci] - 2 * pad), Mm(lead + 0.4))
                tf = tb.text_frame
                tf.word_wrap = False
                tf.auto_size = MSO_AUTO_SIZE.NONE
                tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
                p = tf.paragraphs[0]
                p.line_spacing = Pt(lead / 0.3527778)
                p.space_before = Pt(0)
                p.space_after = Pt(0)
                p.alignment = {"l": PP_ALIGN.LEFT, "r": PP_ALIGN.RIGHT, "c": PP_ALIGN.CENTER}[
                    st.get("align", aligns[ci])
                ]
                r = p.add_run()
                r.text = ln
                _style_run(r, size, bold, str(color))
        if is_head:
            hline(slide, x, cy + h, total_w, "D1D5DB", 0.5)
        elif ri < len(rows) - 1:
            hline(slide, x, cy + h, total_w, rule_color, 0.3)
        cy += h
    return cy


def kpi_card(slide, x, y, w, h, label: str, value: str, sub: str, accent: str, soft: str):
    rect(slide, x, y, w, h, fill="FFFFFF", line=RULE, line_w=0.6, radius=0.05)
    rect(slide, x, y, w, 1.1, fill=accent)
    ty = y + 3.0
    ty = text(slide, x + 3.2, ty, w - 6.4, label, 7.0, MUTED, False, max_lines=1)
    ty = text(slide, x + 3.2, ty + 1.0, w - 6.4, value, 16.0, accent, True, leading=8.0, max_lines=1)
    text(slide, x + 3.2, ty + 0.8, w - 6.4, sub, 6.9, MUTED, False, leading=3.4, max_lines=2)


# --------------------------------------------------------------------------- #
# charts (Pillow, 300 dpi)
# --------------------------------------------------------------------------- #
DPI = 300.0
PXMM = DPI / 25.4

CH_INK = (17, 24, 39)
CH_MUTED = (107, 114, 128)
CH_MUTED2 = (156, 163, 175)
CH_GRID = (238, 240, 243)
CH_AXIS = (209, 213, 219)
CH_WHITE = (255, 255, 255)
CH_COL = {"recent_contiguous": (37, 99, 235), "multi_sink": (22, 163, 74), "uniform": (220, 38, 38)}
CH_NAME = {"recent_contiguous": "recent_contiguous", "multi_sink": "multi_sink", "uniform": "uniform"}


def _f(px: float, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_B if bold else FONT_R, int(round(px)))


def _vtext(img: Image.Image, xy, s: str, px: float, color, bold: bool = False):
    f = _f(px, bold)
    tmp = Image.new("RGBA", (int(f.getlength(s)) + 8, int(px * 1.6)), (255, 255, 255, 0))
    ImageDraw.Draw(tmp).text((4, 0), s, font=f, fill=color)
    tmp = tmp.rotate(90, expand=True)
    img.paste(tmp, (int(xy[0]), int(xy[1])), tmp)


def chart_line(path: Path, summary: pd.DataFrame):
    W, H = int(182 * PXMM), int(92.0 * PXMM)
    img = Image.new("RGB", (W, H), CH_WHITE)
    d = ImageDraw.Draw(img)
    xs_cat = [1024, 4096, 8192, 16384]
    xlabels = ["1k", "4k", "8k", "16k"]
    xsub = ["8K tokens", "32K tokens", "64K tokens", "128K tokens"]
    ymin, ymax, ystep = 0.0, 1.20, 0.20

    left, right, top, bot = 100, 190, 130, 124
    px0, px1 = left, W - right
    py0, py1 = top, H - bot

    def sx(i):
        return px0 + (px1 - px0) * (i / (len(xs_cat) - 1))

    def sy(v):
        return py1 - (v - ymin) / (ymax - ymin) * (py1 - py0)

    # grid + y ticks
    v = ymin
    while v <= ymax + 1e-9:
        yy = sy(v)
        d.line([(px0, yy), (px1, yy)], fill=CH_GRID, width=3)
        d.text((px0 - 18, yy - 15), f"{v:.1f}", font=_f(30), fill=CH_MUTED, anchor="ra")
        v = round(v + ystep, 4)
    d.line([(px0, py1), (px1, py1)], fill=CH_AXIS, width=3)

    # x labels
    for i, (lb, sb) in enumerate(zip(xlabels, xsub)):
        d.text((sx(i), py1 + 16), lb, font=_f(36, True), fill=CH_INK, anchor="ma")
        d.text((sx(i), py1 + 58), sb, font=_f(26), fill=CH_MUTED2, anchor="ma")

    # legend
    lx = px0
    for sc in ["recent_contiguous", "multi_sink", "uniform"]:
        d.ellipse([lx, 34, lx + 22, 56], fill=CH_COL[sc])
        d.text((lx + 32, 33), CH_NAME[sc], font=_f(30), fill=CH_MUTED)
        lx += 32 + d.textlength(CH_NAME[sc], font=_f(30)) + 46

    # series
    pv = summary.set_index(["batch_size", "scene", "range_regions_requested"])
    series = {}
    for sc in ["recent_contiguous", "multi_sink", "uniform"]:
        vals = [float(pv.loc[(128, sc, w), "latency_median_ms"]) for w in xs_cat]
        series[sc] = vals
        pts = [(sx(i), sy(val)) for i, val in enumerate(vals)]
        d.line(pts, fill=CH_COL[sc], width=8, joint="curve")
        for (x_, y_) in pts:
            d.ellipse([x_ - 11, y_ - 11, x_ + 11, y_ + 11], fill=CH_WHITE, outline=CH_COL[sc], width=5)
        # endpoint value
        lab = f"{vals[-1]:.3f}"
        d.text((pts[-1][0] + 24, pts[-1][1] - 16), lab, font=_f(32, True), fill=CH_COL[sc])

    # y axis title
    _vtext(img, (24, py0 + 8), "核心算子中位耗时（ms）", 29, CH_MUTED)

    img.save(path, dpi=(DPI, DPI))
    return series


def chart_scatter(path: Path, raw: pd.DataFrame):
    sub = raw[raw.batch_size == 128]
    W, H = int(94 * PXMM), int(64 * PXMM)
    img = Image.new("RGB", (W, H), CH_WHITE)
    d = ImageDraw.Draw(img)
    left, right, top, bot = 96, 26, 84, 104
    px0, px1 = left, W - right
    py0, py1 = top, H - bot
    xmax, ymax = 1120.0, 1.20

    def sx(v):
        return px0 + v / xmax * (px1 - px0)

    def sy(v):
        return py1 - v / ymax * (py1 - py0)

    for v in [0.0, 0.4, 0.8, 1.2]:
        yy = sy(v)
        d.line([(px0, yy), (px1, yy)], fill=CH_GRID, width=2)
        d.text((px0 - 14, yy - 13), f"{v:.1f}", font=_f(26), fill=CH_MUTED, anchor="ra")
    for v in [0, 256, 512, 768, 1024]:
        xx = sx(v)
        d.line([(xx, py0), (xx, py1)], fill=(246, 247, 249), width=2)
        d.text((xx, py1 + 12), f"{v}", font=_f(26), fill=CH_MUTED, anchor="ma")
    d.line([(px0, py1), (px1, py1)], fill=CH_AXIS, width=2)

    # 1024 reference line (entire KV cache)
    xx = sx(1024)
    yy = py0
    while yy < py1:
        d.line([(xx, yy), (xx, min(yy + 10, py1))], fill=(148, 163, 184), width=3)
        yy += 20

    # points, alpha blended
    overlay = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    od = ImageDraw.Draw(overlay)
    for sc in ["recent_contiguous", "multi_sink", "uniform"]:
        r_, g_, b_ = CH_COL[sc]
        for _, row in sub[sub.scene == sc].iterrows():
            x_, y_ = sx(float(row.touched_pages_union)), sy(float(row.latency_ms_median))
            od.ellipse([x_ - 8, y_ - 8, x_ + 8, y_ + 8], fill=(r_, g_, b_, 120))
    img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"), (0, 0))
    d = ImageDraw.Draw(img)

    d.text((px1, py0 - 44), "ρ = +0.939", font=_f(32, True), fill=CH_INK, anchor="ra")
    d.text((px1, py1 + 46), "batch 并集物理页数（pages）", font=_f(25), fill=CH_MUTED, anchor="ra")
    _vtext(img, (22, py0 + 34), "中位耗时（ms）", 26, CH_MUTED)

    # legend (top-left, above the plot)
    lx, ly = px0, 14
    for sc in ["recent_contiguous", "multi_sink", "uniform"]:
        d.ellipse([lx, ly, lx + 15, ly + 15], fill=CH_COL[sc])
        name = CH_NAME[sc]
        d.text((lx + 21, ly - 4), name, font=_f(23), fill=CH_MUTED)
        lx += 21 + d.textlength(name, font=_f(23)) + 26
    img.save(path, dpi=(DPI, DPI))


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load():
    summary = pd.read_csv(RESULTS / "summary_by_scene_range_batch.csv")
    raw = pd.read_csv(RESULTS / "raw_runs.csv")
    meta = pd.read_csv(RESULTS / "metadata_bench.csv")
    return summary, raw, meta


def v(summary, b, sc, w, col="latency_median_ms"):
    q = summary[(summary.batch_size == b) & (summary.scene == sc) & (summary.range_regions_requested == w)]
    return float(q[col].iloc[0])


# --------------------------------------------------------------------------- #
# pages
# --------------------------------------------------------------------------- #
def build(summary: pd.DataFrame, raw: pd.DataFrame, meta: pd.DataFrame):
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig1 = ASSETS / "brief_line_b128.png"
    fig2 = ASSETS / "brief_scatter_b128.png"
    series = chart_line(fig1, summary)
    chart_scatter(fig2, raw)

    prs = Presentation()
    prs.slide_width = Mm(PAGE_W)
    prs.slide_height = Mm(PAGE_H)
    cp = prs.core_properties
    cp.title = "GBSA 128K / top-512 候选范围基准 · 核心数据速览"
    cp.subject = "Ascend950DT_9582 单卡；128K BF16 KV；topk=512；候选范围 1k/4k/8k/16k"
    cp.author = "GBSA benchmark"
    cp.comments = "720 runs / 36,000 计时样本；数据源 results/summary_by_scene_range_batch.csv"

    w1, w4, w8, w16 = 1024, 4096, 8192, 16384
    R = "recent_contiguous"
    M = "multi_sink"
    U = "uniform"

    # ------------------------------------------------------------------ page 1
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, PAGE_W, 34.0, fill=NAVY)
    text(s, ML, 7.4, CONTENT_W - 60, "GBSA · 128K KV · topk 512 · 候选范围 1k–16k 消融实验", 7.4, "93C5FD", True)
    text(s, ML, 12.6, 130, "GBSA 128K / top-512 候选范围基准", 19.5, "FFFFFF", True, leading=8.6)
    text(s, ML, 22.4, 130, "Ascend950DT_9582 单卡 · CANN 9.2.0 · 128K BF16 KV（K+V 384 MiB）· 仅计 GBSA 核心算子",
         7.2, SLATE_L, False)
    text(s, PAGE_W - MR - 46, 8.4, 46, "720 runs", 10.5, "FFFFFF", True, align="r")
    text(s, PAGE_W - MR - 46, 14.4, 46, "36,000 计时样本", 7.0, SLATE_L, False, align="r")
    text(s, PAGE_W - MR - 46, 18.6, 46, "12 seeds × 50 iters", 7.0, SLATE_L, False, align="r")
    rect(s, 0, 34.0, PAGE_W, 1.1, fill=BLUE)

    # headline
    rect(s, ML, 40.4, 1.1, 13.6, fill=BLUE)
    y = text(
        s, ML + 4.4, 40.0, CONTENT_W - 4.4,
        "结论先行｜topk=512、counts=513 与 KV 形状全部固定，只把候选范围从 1k 放宽到 16k：uniform 场景核心耗时 +96.2%，"
        "recent_contiguous +62.6%，而命中跨请求共享 sink 的 multi_sink 为 −0.3%。",
        9.2, INK2, False, leading=5.1,
    )
    y = text(
        s, ML + 4.4, y + 0.6, CONTENT_W - 4.4,
        "真正的成本不是选中区域的时间跨度，而是 per-request 物理页扇出（page fan-out）与整个 batch 的物理页并集（union working set）。",
        9.2, INK2, False, leading=5.1,
    )

    # KPI cards
    gap = 5.2
    cw = (CONTENT_W - 3 * gap) / 4
    ky, kh = 64.0, 25.6
    kpi = [
        ("uniform · B=128", "+96.2%", f"{v(summary,128,U,w1):.4f}→{v(summary,128,U,w16):.4f} ms（{v(summary,128,U,w16,'speedup_x_vs_w1k'):.2f}×）", C_UNIFORM, "white"),
        ("recent_contiguous · B=128", "+62.6%", f"{v(summary,128,R,w1):.4f}→{v(summary,128,R,w16):.4f} ms（{v(summary,128,R,w16,'speedup_x_vs_w1k'):.2f}×）", C_RECENT, "white"),
        ("multi_sink · B=128 · span 16k", "−0.3%", f"{v(summary,128,M,w1):.4f}→{v(summary,128,M,w16):.4f} ms（{v(summary,128,M,w16,'speedup_x_vs_w1k'):.2f}×）", C_MULTI, "white"),
        ("batch 并集页 vs 耗时", "ρ = +0.939", "而 span vs 耗时仅 −0.115", BLUE, "white"),
    ]
    for i, (lab, val, sub, acc, soft) in enumerate(kpi):
        kpi_card(s, ML + i * (cw + gap), ky, cw, kh, lab, val, sub, acc, soft)

    # table 1
    ty = 94.6
    section(s, ML, ty, CONTENT_W, "表 1", "B = 128 中位耗时与页足迹（12 seeds median，单位 ms）", 9.0)
    rows = [["场景（top-512 分布）", "1k", "4k", "8k", "16k", "Δ 16k−1k", "延迟比", "页/请求", "并集页"]]
    for sc, tag in [(R, "recent_contiguous"), (M, "multi_sink"), (U, "uniform")]:
        rows.append([
            tag,
            f"{v(summary,128,sc,w1):.4f}", f"{v(summary,128,sc,w4):.4f}",
            f"{v(summary,128,sc,w8):.4f}", f"{v(summary,128,sc,w16):.4f}",
            num(f"{v(summary,128,sc,w16,'delta_pct_vs_w1k'):+.1f}%"),
            f"{v(summary,128,sc,w16,'speedup_x_vs_w1k'):.2f}×",
            f"{v(summary,128,sc,w1,'per_request_pages_median'):.1f} → {v(summary,128,sc,w16,'per_request_pages_median'):.1f}",
            f"{v(summary,128,sc,w1,'union_pages_median'):.0f} → {v(summary,128,sc,w16,'union_pages_median'):.0f}",
        ])
    styles = {}
    for ri, sc in enumerate([R, M, U], start=1):
        col = {R: C_RECENT, M: C_MULTI, U: C_UNIFORM}[sc]
        styles[(ri, 0)] = {"bold": True, "color": col}
        d16 = v(summary, 128, sc, w16, "delta_pct_vs_w1k")
        styles[(ri, 5)] = {"bold": True, "color": RED if d16 > 1 else (GREEN if d16 < 0 else MUTED)}
        styles[(ri, 4)] = {"bold": True, "color": INK}
        styles[(ri, 8)] = {"bold": True, "color": INK}
    col_w = [33.8, 17.4, 17.4, 17.4, 18.6, 19.8, 15.4, 22.4, 19.6]
    ty2 = table(s, ML, ty + 5.6, col_w, rows, row_h=7.4, header_h=6.6, font=8.0,
                aligns=["l", "r", "r", "r", "r", "r", "r", "c", "c"], styles=styles)
    text(s, ML, ty2 + 1.6, CONTENT_W,
         "仅计 generic_block_sparse_attention 核心 aclnn 调用，不含 IndexR / metadata / pack / host 调度；16k 的可用历史 region 为 16383（当前 region 由算子追加）；"
         "末两列为 W=1k → W=16k 的变化。",
         6.7, MUTED, False, leading=3.4, max_lines=2)

    # figure 1
    fy = 141.6
    text(s, ML, fy, CONTENT_W, "图 1｜B = 128 三种 top-512 空间分布下的核心算子中位耗时曲线（误差带为 12 seeds IQR）",
         7.6, MUTED, False, max_lines=1)
    s.shapes.add_picture(str(fig1), Mm(ML), Mm(fy + 4.4), width=Mm(182))

    # takeaway box
    by = 244.0
    rect(s, ML, by, CONTENT_W, 22.4, fill="F8FAFC", line="DBEAFE", line_w=0.7, radius=0.06)
    rect(s, ML, by, 1.2, 22.4, fill=BLUE)
    text(s, ML + 5.2, by + 3.2, 46, "一句话结论", 7.6, BLUE, True)
    text(s, ML + 5.2, by + 7.6, CONTENT_W - 10.4,
         "同样的 ≈16k 选中 span：multi_sink 0.3741 ms、uniform 1.0427 ms —— 2.79× 的差距只来自物理页布局。"
         "限制候选范围是必要条件，但不是充分条件。",
         8.8, INK2, False, leading=4.7)

    # footer
    hline(s, ML, 283.2, CONTENT_W, RULE, 0.3)
    text(s, ML, 285.0, 140, "复现：scripts/run_bench.py ｜ 原始数据：results/ ｜ 图表：figures/", 6.6, MUTED2)
    text(s, PAGE_W - MR - 20, 285.0, 20, "1 / 2", 6.6, MUTED2, False, align="r")

    # ------------------------------------------------------------------ page 2
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, PAGE_W, 21.0, fill=NAVY)
    text(s, ML, 6.2, 100, "机制拆解与工程启示", 14.0, "FFFFFF", True, leading=7.2)
    text(s, ML, 14.4, 120, "同样是 topk=512，为什么耗时差 2.79×", 7.4, SLATE_L, False)
    text(s, PAGE_W - MR - 62, 8.6, 62, "B=128 全部 144 runs · 720/720 output finite", 7.0, SLATE_L, False, align="r")
    rect(s, 0, 21.0, PAGE_W, 1.1, fill=BLUE)

    # 01 controlled contrasts
    section(s, ML, 26.4, CONTENT_W, "01", "两次受控对照：topk 相同，只有物理页足迹不同", 9.4)
    cy0, chh = 32.2, 43.4
    cw2 = (CONTENT_W - 5.2) / 2
    _pr = lambda b, sc, w_, c: v(summary, b, sc, w_, c)
    spanA_m, spanA_u = _pr(128, M, w16, "span_median_regions"), _pr(128, U, w16, "span_median_regions")
    spanB_r, spanB_u = _pr(128, R, w8, "span_median_regions"), _pr(128, U, w8, "span_median_regions")
    cards = [
        (ML, "对照 A｜选中 span 都 ≈16k（B=128，W=16k）", "FEF2F2", RED,
         [["指标", "multi_sink", "uniform"],
          ["选中 span（reg.）", f"{spanA_m:.0f}", f"{spanA_u:.0f}"],
          ["页 / 请求", f"{_pr(128,M,w16,'per_request_pages_median'):.1f}", f"{_pr(128,U,w16,'per_request_pages_median'):.1f}"],
          ["batch 并集页", f"{_pr(128,M,w16,'union_pages_median'):.0f}", f"{_pr(128,U,w16,'union_pages_median'):.0f}"],
          ["中位耗时（ms）", f"{_pr(128,M,w16,'latency_median_ms'):.4f}", f"{_pr(128,U,w16,'latency_median_ms'):.4f}"]],
         f"→ uniform 慢 {_pr(128,U,w16,'latency_median_ms')/_pr(128,M,w16,'latency_median_ms'):.2f}×：span 几乎相同，并集页 39 vs 1024"),
        (ML + cw2 + 5.2, "对照 B｜batch 并集都 ≈512 页（B=128，W=8k）", "EFF6FF", BLUE,
         [["指标", "recent_contiguous", "uniform"],
          ["页 / 请求", f"{_pr(128,R,w8,'per_request_pages_median'):.1f}", f"{_pr(128,U,w8,'per_request_pages_median'):.1f}"],
          ["batch 并集页", f"{_pr(128,R,w8,'union_pages_median'):.0f}", f"{_pr(128,U,w8,'union_pages_median'):.0f}"],
          ["选中 span（reg.）", f"{spanB_r:.0f}", f"{spanB_u:.0f}"],
          ["中位耗时（ms）", f"{_pr(128,R,w8,'latency_median_ms'):.4f}", f"{_pr(128,U,w8,'latency_median_ms'):.4f}"]],
         f"→ uniform 慢 {_pr(128,U,w8,'latency_median_ms')/_pr(128,R,w8,'latency_median_ms'):.2f}×：并集相同，请求内扇出发散 10×"),
    ]
    for cx, title, soft, acc, rows_c, foot in cards:
        rect(s, cx, cy0, cw2, chh, fill="FFFFFF", line=RULE, line_w=0.6, radius=0.04)
        rect(s, cx, cy0, cw2, 6.2, fill=soft)
        text(s, cx + 3.0, cy0 + 1.5, cw2 - 6, title, 7.6, INK, True, max_lines=1)
        cwl = [26.0, 28.0, 28.4]
        table(s, cx + 3.0, cy0 + 8.2, cwl, rows_c, row_h=5.6, header_h=4.8, font=7.4, header_font=6.8,
              aligns=["l", "r", "r"], zebra=None, pad=1.0, rule_color="F1F3F5", expect_w=None)
        text(s, cx + 3.0, cy0 + 35.8, cw2 - 6, foot, 7.2, acc, True, max_lines=1)

    text(s, ML, 77.8, CONTENT_W,
         f"对照 C（W=1k，并集同为 65 页）：{v(summary,128,R,w1):.4f} vs {v(summary,128,U,w1):.4f} ms，"
         f"+{(v(summary,128,U,w1)/v(summary,128,R,w1)-1)*100:.1f}% —— per-request fan-out 是独立于并集的成本项。",
         7.4, MUTED, False, max_lines=1)

    # 02 batch table
    section(s, ML, 86.4, CONTENT_W, "02", "batch 规模决定效应是否显现：B≥32 才被计算 / 访存放大", 9.4)
    brows = [["Batch", "recent_contiguous  W=1k → 16k", "Δ", "uniform  W=1k → 16k", "Δ", "multi_sink  Δ"]]
    for b in [1, 8, 32, 128, 256]:
        brows.append([
            str(b),
            f"{v(summary,b,R,w1):.4f} → {v(summary,b,R,w16):.4f}",
            num(f"{v(summary,b,R,w16,'delta_pct_vs_w1k'):+.1f}%"),
            f"{v(summary,b,U,w1):.4f} → {v(summary,b,U,w16):.4f}",
            num(f"{v(summary,b,U,w16,'delta_pct_vs_w1k'):+.1f}%"),
            num(f"{v(summary,b,M,w16,'delta_pct_vs_w1k'):+.1f}%"),
        ])
    bstyles = {}
    for ri, b in enumerate([1, 8, 32, 128, 256], start=1):
        hot = b >= 32
        for ci in (2, 4):
            bstyles[(ri, ci)] = {"bold": True, "color": RED if hot else MUTED}
        bstyles[(ri, 5)] = {"bold": True, "color": GREEN}
        bstyles[(ri, 0)] = {"bold": True, "color": INK}
    ty2 = table(s, ML, 91.8, [16.0, 52.0, 19.0, 52.0, 19.0, 24.0],
                brows, row_h=6.4, header_h=6.2, font=7.8,
                aligns=["c", "r", "r", "r", "r", "r"], styles=bstyles)
    text(s, ML, ty2 + 1.2, CONTENT_W,
         "B=1/8 时 launch overhead 主导，范围效应 ≤2.3%；multi_sink 在所有 batch 下延迟比均为 1.00× —— 只要跨请求并集小，大 batch 也不会因范围大而变慢。",
         6.8, MUTED, False, leading=3.4, max_lines=1)

    # 03 predictor
    section(s, ML, 142.4, CONTENT_W, "03", "预测因子：batch 并集页，而不是 span", 9.4, x_end=100.0)
    corr = pd.read_csv(RESULTS / "correlations_by_batch.csv")
    corr = corr[corr.subset == "all_scenes"].set_index("batch_size")

    def cval(b, col):
        return float(corr.loc[b, col])

    crows = [["Spearman ρ（144 runs）", "B=32", "B=128", "B=256"]]
    for label, col in [
        ("选中 span vs 耗时", "spearman_span_vs_latency"),
        ("页 / 请求 vs 耗时", "spearman_per_request_pages_vs_latency"),
        ("batch 并集页 vs 耗时", "spearman_union_pages_vs_latency"),
    ]:
        crows.append([label] + [num(f"{cval(b, col):+.3f}") for b in (32, 128, 256)])
    cstyles = {(3, 1): {"bold": True, "color": BLUE}, (3, 2): {"bold": True, "color": BLUE},
               (3, 3): {"bold": True, "color": BLUE}, (1, 1): {"color": MUTED2},
               (1, 2): {"color": MUTED2}, (1, 3): {"color": MUTED2},
               (2, 1): {"color": MUTED}, (2, 2): {"color": MUTED}, (2, 3): {"color": MUTED}}
    ty3 = table(s, ML, 147.8, [42.0, 14.0, 14.0, 14.0], crows,
                row_h=6.2, header_h=6.0, font=7.6, header_font=6.8,
                aligns=["l", "r", "r", "r"], styles=cstyles, expect_w=None)
    text(s, ML, ty3 + 2.4, 84,
         "span 与耗时几乎无关（ρ≈−0.12），并集页与耗时的 ρ 稳定在 +0.93 以上，且不随 batch 漂移。",
         7.2, INK2, False, leading=3.9, max_lines=2)

    # monitoring snippet panel
    rect(s, ML, 184.6, 84.0, 21.6, fill=PANEL, line=RULE, line_w=0.6, radius=0.05)
    text(s, ML + 3.4, 186.8, 78, "线上可直接算出来的成本代理量", 7.0, MUTED, True, max_lines=1)
    text(s, ML + 3.4, 191.0, 78, "union_pages = len(set(block_table[ids // 16]))", 7.6, INK, True, max_lines=1)
    text(s, ML + 3.4, 196.0, 78,
         "1024 页 = 全部 384 MiB KV cache；越接近该上限，L2 复用越差。",
         6.8, MUTED, False, leading=3.3, max_lines=1)

    s.shapes.add_picture(str(fig2), Mm(102.0), Mm(142.4), width=Mm(94))
    text(s, 102.0, 208.2, 94,
         "图 2｜B=128 全部 144 个 run：并集页数 vs 中位耗时（虚线 = 1,024 页，即全部 KV cache）",
         6.6, MUTED2, False, leading=3.1, max_lines=2)

    # 04 conclusions
    section(s, ML, 219.0, CONTENT_W, "04", "结论与可执行项", 9.4)
    rect(s, ML, 224.4, CONTENT_W, 37.6, fill="F8FAFC", line=RULE, line_w=0.6, radius=0.03)
    items = [
        ("不要用候选范围或 span 预测耗时", "同 16k 范围可差 2.79×；span 与耗时的 ρ 仅 −0.115。"),
        ("把 batch 并集页作为线上指标", "拆 batch、限分散度、hot page 预留都围绕它做取舍。"),
        ("缩窗口必须同时约束分散度", "1k 内均匀分散仍比连续带慢 40.4%，仅缩范围不够。"),
        ("跨请求聚类 / 共享 sink 页", "选择模式相近的请求同批，B≥32 时收益最大。"),
    ]
    colx = [ML + 6.0, ML + 95.0]
    for i, (head, body) in enumerate(items):
        cx = colx[i % 2]
        cy = 229.2 + (i // 2) * 14.6
        text(s, cx, cy + 0.3, 4, "0%d" % (i + 1), 8.6, BLUE, True, max_lines=1)
        text(s, cx + 5.2, cy, 76, head, 8.2, INK, True, max_lines=1)
        text(s, cx + 5.2, cy + 4.9, 76, body, 7.2, MUTED, False, leading=3.6, max_lines=2)

    hline(s, ML, 265.8, CONTENT_W, RULE, 0.3)
    text(s, ML, 267.6, 160,
         "环境：host 177.45.22.1 · 容器 j00934364_step5_v2 · Ascend950DT_9582 ×8（device 0）· CANN 9.2.0 · torch 2.10.0+cpu · torch_npu 2.10.0.post4",
         6.5, MUTED2, False, leading=3.2, max_lines=1)
    text(s, ML, 272.0, CONTENT_W - 36,
         "局限：KV 为确定性结构化高斯 cache（非真实推理中间结果）；topk ids 为 IndexR-like 受控构造，非实际算子输出；仅测核心算子，不含端到端 layer / scheduler / metadata（B=128 约 0.081 ms）。",
         6.5, MUTED2, False, leading=3.2, max_lines=2)
    text(s, 180.0, 267.6, 16, "2 / 2", 6.6, MUTED2, False, align="r")

    prs.save(str(OUT_PPTX))
    return OUT_PPTX


def check_glyphs(prs) -> list[str]:
    """Every character used must exist in the CJK font, else it prints as a box."""
    f = ImageFont.truetype(FONT_R, 48)
    box = Image.new("L", (70, 80), 255)
    ImageDraw.Draw(box).text((4, 4), "\ue000", font=f, fill=0)
    ref = np.asarray(box)

    def bitmap(ch: str):
        im = Image.new("L", (70, 80), 255)
        ImageDraw.Draw(im).text((4, 4), ch, font=f, fill=0)
        return np.asarray(im)

    chars: set[str] = set()
    for slide in prs.slides:
        for shp in slide.shapes:
            if not shp.has_text_frame:
                continue
            for p in shp.text_frame.paragraphs:
                for r in p.runs:
                    chars.update(r.text)
    missing = []
    for ch in sorted(chars):
        if ch.isspace():
            continue
        if np.array_equal(bitmap(ch), ref):
            missing.append(ch)
    return missing


def convert(pptx: Path) -> Path:
    node = r"D:\DeepSeek harness\resources\runtime\primary-runtime\dependencies\node\bin\node.exe"
    cli = (r"D:\DeepSeek harness\resources\app.asar.unpacked\dsh\node_modules"
           r"\@deepseek-ai\libreoffice-kit\lib\cli.js")
    if OUT_PDF.exists():
        OUT_PDF.unlink()
    res = subprocess.run([node, cli, "convert", "--input", str(pptx), "--output", str(OUT_PDF)],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("convert:", res.stdout.strip() or res.stderr.strip())
    return OUT_PDF


if __name__ == "__main__":
    summary, raw, meta = load()
    build(summary, raw, meta)
    if WARN:
        print(f"--- {len(WARN)} layout warnings ---")
        for w in WARN:
            print(" !", w)
    else:
        print("layout: no warnings")
    prs = Presentation(str(OUT_PPTX))
    miss = check_glyphs(prs)
    print("missing glyphs:", miss if miss else "none")
    pdf = convert(OUT_PPTX)
    data = pdf.read_bytes()
    n_page = data.count(b"/Type/Page") - data.count(b"/Type/Pages") - data.count(b"/Type /Pages")
    print("pdf:", len(data), "bytes, pages:", n_page, "| sig:", data[:5])
