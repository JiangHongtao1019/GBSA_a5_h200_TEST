#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Side-by-side H200/vLLM reference numbers vs the Ascend GBSA run.

WARNING: this is a workload-level comparison, not proof that the two kernels
are equivalent.  The H200 path is gather+SDPA or SDPA-only through the
vLLM-compatible backend, while Ascend uses aclnnGenericBlockSparseAttention.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEYS = ["batch_size", "scene", "range_regions_requested"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ascend-dir", default="results")
    ap.add_argument("--h200-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    ad = Path(args.ascend_dir)
    hd = Path(args.h200_dir)
    out = Path(args.out_dir) if args.out_dir else hd / "comparison_vs_ascend"
    out.mkdir(parents=True, exist_ok=True)

    a = pd.read_csv(ad / "summary_by_scene_range_batch.csv")
    h = pd.read_csv(hd / "summary_by_scene_range_batch.csv")
    a = a[KEYS + ["latency_median_ms", "union_pages_median", "working_set_mib_median"]].rename(
        columns={
            "latency_median_ms": "ascend_latency_ms",
            "union_pages_median": "ascend_union_pages",
            "working_set_mib_median": "ascend_working_set_mib",
        }
    )
    h = h.rename(columns={
        "latency_median_ms": "h200_latency_ms",
        "union_pages_median": "h200_union_pages",
        "working_set_mib_median": "h200_working_set_mib",
    })
    merged = h.merge(a, on=KEYS, how="left")
    merged["h200_over_ascend_latency_ratio"] = merged["h200_latency_ms"] / merged["ascend_latency_ms"]
    merged["h200_speedup_vs_ascend"] = merged["ascend_latency_ms"] / merged["h200_latency_ms"]
    merged = merged.sort_values(["timing_mode", "batch_size", "scene", "range_regions_requested"])
    merged.to_csv(out / "h200_vs_ascend.csv", index=False)

    b128 = merged[merged.batch_size == 128].copy()
    if not b128.empty:
        cols = ["timing_mode", "scene", "range_regions_requested", "h200_latency_ms",
                "ascend_latency_ms", "h200_speedup_vs_ascend", "h200_union_pages"]
        md = ["# H200 vs Ascend (B=128)", "",
              "| " + " | ".join(cols) + " |",
              "|" + "---|" * len(cols)]
        for _, row in b128[cols].iterrows():
            vals = []
            for c in cols:
                v = row[c]
                vals.append(f"{v:.6g}" if isinstance(v, float) else str(v))
            md.append("| " + " | ".join(vals) + " |")
        (out / "h200_vs_ascend_B128.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out / 'h200_vs_ascend.csv'}")
    if not b128.empty:
        print(f"wrote {out / 'h200_vs_ascend_B128.md'}")
    print("NOTE: workload-level comparison only; kernels/timing boundaries differ.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())