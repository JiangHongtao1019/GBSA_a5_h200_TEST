#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""No-GPU validation for the H200/vLLM benchmark result directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="outputs_h200_vllm/full_12seed")
    args = ap.parse_args()
    d = Path(args.results)
    runs = pd.read_csv(d / "raw_runs.csv")
    timings = pd.read_csv(d / "raw_timings.csv")
    summary = pd.read_csv(d / "summary_by_scene_range_batch.csv")
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))

    assert not runs.empty and not timings.empty and not summary.empty
    assert runs[["latency_median_ms", "latency_mean_ms", "latency_min_ms", "latency_max_ms"]].notna().all().all()
    assert np.isfinite(runs.select_dtypes(include=[np.number]).to_numpy()).all()
    assert np.isfinite(timings["latency_ms"].to_numpy()).all()
    assert (runs["latency_median_ms"] > 0).all()

    expected_runs = manifest["runs"]
    assert len(runs) == expected_runs, (len(runs), expected_runs)
    assert len(timings) == manifest["timed_samples"], (len(timings), manifest["timed_samples"])
    assert len(summary) == runs.groupby(
        ["batch_size", "scene", "range_regions_requested", "timing_mode"]
    ).ngroups

    # Recompute each run median from the per-iteration timings.
    calc = timings.groupby("run_id")["latency_ms"].median()
    merged = runs.set_index("run_id")["latency_median_ms"].to_frame().join(calc, how="left")
    max_diff = float((merged["latency_median_ms"] - merged["latency_ms"]).abs().max())
    assert max_diff < 1e-9, max_diff
    assert timings.groupby("run_id").size().nunique() == 1

    assert sha256(d / "raw_runs.csv") == manifest["raw_runs_sha256"]
    assert sha256(d / "raw_timings.csv") == manifest["raw_timings_sha256"]

    print("PASS")
    print(f"  runs={len(runs)} timed_samples={len(timings)} summary_rows={len(summary)}")
    print(f"  iterations_per_run={int(timings.groupby('run_id').size().iloc[0])}")
    print(f"  max |raw median - recomputed median| = {max_diff:.3e}")
    print(f"  raw_runs sha256={manifest['raw_runs_sha256']}")
    print(f"  raw_timings sha256={manifest['raw_timings_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())