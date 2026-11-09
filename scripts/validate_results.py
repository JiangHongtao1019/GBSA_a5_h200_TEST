#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate the packaged benchmark results without requiring an NPU."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    raw = pd.read_csv(RESULTS / "raw_runs.csv")
    timings = pd.read_csv(RESULTS / "raw_timings.csv")
    meta = pd.read_csv(RESULTS / "metadata_bench.csv")
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))

    assert len(raw) == 720, f"raw_runs rows={len(raw)}, expected 720"
    assert len(timings) == 36000, f"raw_timings rows={len(timings)}, expected 36000"
    assert raw.out_finite.astype(str).str.lower().isin(["true", "1", "yes"]).all()
    assert set(raw.batch_size) == {1, 8, 32, 128, 256}
    assert set(raw.scene) == {"recent_contiguous", "multi_sink", "uniform"}
    assert set(raw.range_regions_requested) == {1024, 4096, 8192, 16384}
    assert set(raw.seed) == set(range(7000, 7012))
    assert (raw.topk == 512).all()
    assert (raw.region_size == 8).all()
    assert (raw.page_size == 128).all()
    assert (raw.context_tokens == 131072).all()

    counts = timings.groupby("run_id").size()
    assert (counts == 50).all(), f"unexpected timing counts: {counts.value_counts().to_dict()}"
    assert set(counts.index) == set(raw.run_id)

    calc = timings.groupby("run_id")["latency_ms"].median()
    merged = raw.set_index("run_id").join(calc.rename("calc_median"))
    max_diff = (merged.latency_ms_median - merged.calc_median).abs().max()
    assert max_diff < 1e-9, f"median mismatch: {max_diff}"

    assert len(meta) == 5
    assert set(meta.batch_size) == {1, 8, 32, 128, 256}
    assert sha256(RESULTS / "raw_runs.csv") == manifest["raw_runs_sha256"]
    assert sha256(RESULTS / "raw_timings.csv") == manifest["raw_timings_sha256"]

    summary = pd.read_csv(RESULTS / "summary_by_scene_range_batch.csv")
    assert len(summary) == 5 * 3 * 4
    assert summary.latency_median_ms.notna().all()

    print("PASS")
    print(f"  runs={len(raw)} timed_samples={len(timings)} metadata_rows={len(meta)}")
    print(f"  max |raw median - recomputed median| = {max_diff:.3e}")
    print(f"  raw_runs sha256={manifest['raw_runs_sha256']}")
    print(f"  raw_timings sha256={manifest['raw_timings_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())