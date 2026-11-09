#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fill the Markdown report template with benchmark values."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from report_common import SCENE_ORDER, md_table


def make_markdown(summary: pd.DataFrame, contrast: pd.DataFrame, corr: pd.DataFrame,
                  raw: pd.DataFrame, meta: pd.DataFrame, config: dict,
                  env: dict, manifest: dict, figs: dict[str, Path], out_path: Path) -> None:
    pv = summary.set_index(["batch_size", "scene", "range_regions_requested"])

    def lat(b: int, sc: str, w: int) -> float:
        return float(pv.loc[(b, sc, w), "latency_median_ms"])

    def ratio(b: int, sc: str) -> float:
        return lat(b, sc, 16384) / lat(b, sc, 1024)

    def val(b: int, sc: str, w: int, col: str) -> float:
        return float(pv.loc[(b, sc, w), col])

    r128_1k = lat(128, "recent_contiguous", 1024)
    r128_16k = lat(128, "recent_contiguous", 16384)
    m128_1k = lat(128, "multi_sink", 1024)
    m128_16k = lat(128, "multi_sink", 16384)
    u128_1k = lat(128, "uniform", 1024)
    u128_16k = lat(128, "uniform", 16384)
    r128_16_pct = 100 * (r128_16k / r128_1k - 1)
    u128_16_pct = 100 * (u128_16k / u128_1k - 1)
    m128_16_pct = 100 * (m128_16k / m128_1k - 1)
    span_m = val(128, "multi_sink", 16384, "span_median_regions")
    span_u = val(128, "uniform", 16384, "span_median_regions")
    union_m = val(128, "multi_sink", 16384, "union_pages_median")
    union_u = val(128, "uniform", 16384, "union_pages_median")
    perreq_m = val(128, "multi_sink", 16384, "per_request_pages_median")
    perreq_u = val(128, "uniform", 16384, "per_request_pages_median")
    same_union_r = val(128, "recent_contiguous", 8192, "union_pages_median")
    same_union_u = val(128, "uniform", 8192, "union_pages_median")
    lat_r8 = lat(128, "recent_contiguous", 8192)
    lat_u8 = lat(128, "uniform", 8192)
    baseline_r = lat(128, "recent_contiguous", 1024)
    baseline_u = lat(128, "uniform", 1024)
    perreq_base_r = val(128, "recent_contiguous", 1024, "per_request_pages_median")
    perreq_base_u = val(128, "uniform", 1024, "per_request_pages_median")
    base_union_r = val(128, "recent_contiguous", 1024, "union_pages_median")
    base_union_u = val(128, "uniform", 1024, "union_pages_median")
    finite = int(raw.out_finite.sum())
    meta_med = float(meta[meta.batch_size == 128]["metadata_ms_median"].iloc[0])
    rho128 = float(raw[raw.batch_size == 128][["touched_pages_union", "latency_ms_median"]]
                   .corr(method="spearman").iloc[0, 1])

    rows = []
    for sc in SCENE_ORDER:
        rows.append({
            "Scene": sc,
            "1k": lat(128, sc, 1024),
            "4k": lat(128, sc, 4096),
            "8k": lat(128, sc, 8192),
            "16k": lat(128, sc, 16384),
            "Δ16k-1k (ms)": lat(128, sc, 16384) - lat(128, sc, 1024),
            "Latency ratio": lat(128, sc, 16384) / lat(128, sc, 1024),
            "Per-req pages 1k→16k": f"{val(128, sc, 1024, 'per_request_pages_median'):.1f} → {val(128, sc, 16384, 'per_request_pages_median'):.1f}",
            "Union pages 1k→16k": f"{val(128, sc, 1024, 'union_pages_median'):.0f} → {val(128, sc, 16384, 'union_pages_median'):.0f}",
        })
    t128 = pd.DataFrame(rows)
    table_b128 = md_table(t128, list(t128.columns), list(t128.columns), {
        "1k": "{:.4f}", "4k": "{:.4f}", "8k": "{:.4f}", "16k": "{:.4f}",
        "Δ16k-1k (ms)": "{:+.4f}", "Latency ratio": "{:.2f}×",
    })

    rows = []
    for b in [32, 128, 256]:
        for sc in SCENE_ORDER:
            rows.append({
                "Batch": b,
                "Scene": sc,
                "1k (ms)": lat(b, sc, 1024),
                "16k (ms)": lat(b, sc, 16384),
                "Δ (%)": 100 * (ratio(b, sc) - 1),
                "Ratio": ratio(b, sc),
                "Union pages 1k→16k": f"{val(b, sc, 1024, 'union_pages_median'):.0f} → {val(b, sc, 16384, 'union_pages_median'):.0f}",
                "Per-req pages 1k→16k": f"{val(b, sc, 1024, 'per_request_pages_median'):.1f} → {val(b, sc, 16384, 'per_request_pages_median'):.1f}",
            })
    tcontrast = pd.DataFrame(rows)
    table_contrast = md_table(tcontrast, list(tcontrast.columns), list(tcontrast.columns), {
        "Batch": "{:.0f}", "1k (ms)": "{:.4f}", "16k (ms)": "{:.4f}", "Δ (%)": "{:+.1f}%", "Ratio": "{:.2f}×",
    })

    rows = []
    for b in sorted(summary.batch_size.unique()):
        rows.append({
            "Batch": b,
            "recent_contiguous": f"{lat(b, 'recent_contiguous', 1024):.4f} → {lat(b, 'recent_contiguous', 16384):.4f} ({100*(ratio(b,'recent_contiguous')-1):+.1f}%)",
            "multi_sink": f"{lat(b, 'multi_sink', 1024):.4f} → {lat(b, 'multi_sink', 16384):.4f} ({100*(ratio(b,'multi_sink')-1):+.1f}%)",
            "uniform": f"{lat(b, 'uniform', 1024):.4f} → {lat(b, 'uniform', 16384):.4f} ({100*(ratio(b,'uniform')-1):+.1f}%)",
        })
    tbatch = pd.DataFrame(rows)
    table_batch = md_table(tbatch, list(tbatch.columns), list(tbatch.columns), {"Batch": "{:.0f}"})

    crows = corr[(corr.subset == "all_scenes") & (corr.batch_size >= 32)].copy()
    crows = crows[["batch_size", "n_runs", "spearman_span_vs_latency",
                   "spearman_per_request_pages_vs_latency", "spearman_union_pages_vs_latency"]]
    crows = crows.rename(columns={
        "batch_size": "Batch", "n_runs": "N",
        "spearman_span_vs_latency": "ρ(span, latency)",
        "spearman_per_request_pages_vs_latency": "ρ(per-req pages, latency)",
        "spearman_union_pages_vs_latency": "ρ(union pages, latency)",
    })
    table_corr = md_table(crows, list(crows.columns), list(crows.columns), {
        "Batch": "{:.0f}", "N": "{:.0f}",
        "ρ(span, latency)": "{:+.3f}",
        "ρ(per-req pages, latency)": "{:+.3f}",
        "ρ(union pages, latency)": "{:+.3f}",
    })

    repl = {
        "R128_1K": f"{r128_1k:.4f}",
        "R128_16K": f"{r128_16k:.4f}",
        "R128_16_PCT": f"{r128_16_pct:.1f}",
        "M128_1K": f"{m128_1k:.4f}",
        "M128_16K": f"{m128_16k:.4f}",
        "M128_16_PCT": f"{m128_16_pct:+.1f}",
        "U128_1K": f"{u128_1k:.4f}",
        "U128_16K": f"{u128_16k:.4f}",
        "U128_16_PCT": f"{u128_16_pct:.1f}",
        "FINITE": str(finite),
        "OUT_ABS_MED": f"{raw.out_abs_mean.median():.4f}",
        "TABLE_B128": table_b128,
        "TABLE_BATCH": table_batch,
        "TABLE_CONTRAST": table_contrast,
        "TABLE_CORR": table_corr,
        "SPAN_M": f"{span_m:.0f}",
        "SPAN_U": f"{span_u:.0f}",
        "UNION_M": f"{union_m:.0f}",
        "UNION_U": f"{union_u:.0f}",
        "PERREQ_M": f"{perreq_m:.1f}",
        "PERREQ_U": f"{perreq_u:.1f}",
        "U_M_RATIO": f"{u128_16k / m128_16k:.2f}",
        "SAME_UNION_R": f"{same_union_r:.0f}",
        "SAME_UNION_U": f"{same_union_u:.0f}",
        "LAT_R8": f"{lat_r8:.4f}",
        "LAT_U8": f"{lat_u8:.4f}",
        "LAT_U8_DIFF_PCT": f"{100*(lat_u8/lat_r8-1):.1f}",
        "PERREQ_R8": f"{val(128, 'recent_contiguous', 8192, 'per_request_pages_median'):.1f}",
        "PERREQ_U8": f"{val(128, 'uniform', 8192, 'per_request_pages_median'):.1f}",
        "PERREQ_BASE_R": f"{perreq_base_r:.1f}",
        "PERREQ_BASE_U": f"{perreq_base_u:.1f}",
        "BASE_UNION_R": f"{base_union_r:.0f}",
        "BASE_UNION_U": f"{base_union_u:.0f}",
        "BASELINE_R": f"{baseline_r:.4f}",
        "BASELINE_U": f"{baseline_u:.4f}",
        "BASELINE_U_DIFF_PCT": f"{100*(baseline_u/baseline_r-1):.1f}",
        "RHO128": f"{rho128:.3f}",
        "META_1": f"{meta[meta.batch_size==1]['metadata_ms_median'].iloc[0]:.4f}",
        "META_8": f"{meta[meta.batch_size==8]['metadata_ms_median'].iloc[0]:.4f}",
        "META_32": f"{meta[meta.batch_size==32]['metadata_ms_median'].iloc[0]:.4f}",
        "META_128": f"{meta_med:.4f}",
        "META_256": f"{meta[meta.batch_size==256]['metadata_ms_median'].iloc[0]:.4f}",
        "B32_R_PCT": f"{100*(ratio(32,'recent_contiguous')-1):.1f}",
        "B32_U_PCT": f"{100*(ratio(32,'uniform')-1):.1f}",
        "B128_R_PCT": f"{100*(ratio(128,'recent_contiguous')-1):.1f}",
        "B128_U_PCT": f"{100*(ratio(128,'uniform')-1):.1f}",
        "B256_R_PCT": f"{100*(ratio(256,'recent_contiguous')-1):.1f}",
        "B256_U_PCT": f"{100*(ratio(256,'uniform')-1):.1f}",
    }
    template = (Path(__file__).resolve().parent / "report_template.md").read_text(encoding="utf-8")
    for k, v in repl.items():
        template = template.replace(f"@@{k}@@", str(v))
    template = template.replace("\n---\n", "\n\n---\n")
    out_path.write_text(template, encoding="utf-8")