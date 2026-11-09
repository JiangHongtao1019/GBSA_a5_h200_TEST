#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Side-by-side H200/vLLM reference numbers vs the Ascend GBSA run.

WARNING: workload-level comparison only; kernels/timing boundaries differ.
Uses only the Python standard library (no pandas/numpy).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

KEYS = ["batch_size", "scene", "range_regions_requested"]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(v) -> str:
    if v is None or v == "":
        return ""
    try:
        return f"{float(v):.6g}"
    except (TypeError, ValueError):
        return str(v)


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

    ascend_rows = read_csv(ad / "summary_by_scene_range_batch.csv")
    h200_rows = read_csv(hd / "summary_by_scene_range_batch.csv")
    ascend = {}
    for r in ascend_rows:
        key = (r["batch_size"], r["scene"], r["range_regions_requested"])
        ascend[key] = r

    merged: list[dict] = []
    for r in h200_rows:
        key = (r["batch_size"], r["scene"], r["range_regions_requested"])
        a = ascend.get(key)
        row = {
            "batch_size": r["batch_size"],
            "scene": r["scene"],
            "range_regions_requested": r["range_regions_requested"],
            "timing_mode": r["timing_mode"],
            "h200_latency_ms": r.get("latency_median_ms"),
            "h200_union_pages": r.get("union_pages_median"),
            "h200_working_set_mib": r.get("working_set_mib_median"),
            "ascend_latency_ms": a.get("latency_median_ms") if a else None,
            "ascend_union_pages": a.get("union_pages_median") if a else None,
            "ascend_working_set_mib": a.get("working_set_mib_median") if a else None,
        }
        h200_lat = float(row["h200_latency_ms"]) if row["h200_latency_ms"] else None
        asc_lat = float(row["ascend_latency_ms"]) if row["ascend_latency_ms"] else None
        row["h200_over_ascend_latency_ratio"] = (h200_lat / asc_lat) if h200_lat and asc_lat else None
        row["h200_speedup_vs_ascend"] = (asc_lat / h200_lat) if h200_lat and asc_lat else None
        merged.append(row)

    merged.sort(key=lambda x: (x["timing_mode"], int(x["batch_size"]), x["scene"], int(x["range_regions_requested"])))
    fields = [
        "batch_size", "scene", "range_regions_requested", "timing_mode",
        "h200_latency_ms", "ascend_latency_ms",
        "h200_over_ascend_latency_ratio", "h200_speedup_vs_ascend",
        "h200_union_pages", "ascend_union_pages",
        "h200_working_set_mib", "ascend_working_set_mib",
    ]
    write_csv(out / "h200_vs_ascend.csv", merged, fields)

    b128 = [r for r in merged if r["batch_size"] == "128"]
    if b128:
        cols = ["timing_mode", "scene", "range_regions_requested", "h200_latency_ms",
                "ascend_latency_ms", "h200_speedup_vs_ascend", "h200_union_pages"]
        md = ["# H200 vs Ascend (B=128)", "",
              "| " + " | ".join(cols) + " |",
              "|" + "---|" * len(cols)]
        for r in b128:
            md.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
        (out / "h200_vs_ascend_B128.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"wrote {out / 'h200_vs_ascend.csv'}")
    if b128:
        print(f"wrote {out / 'h200_vs_ascend_B128.md'}")
    print("NOTE: workload-level comparison only; kernels/timing boundaries differ.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())