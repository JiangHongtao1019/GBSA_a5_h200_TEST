#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""No-GPU validation for the H200/vLLM benchmark result directory.

Uses only the Python standard library (no pandas/numpy).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
    runs = read_csv(d / "raw_runs.csv")
    timings = read_csv(d / "raw_timings.csv")
    summary = read_csv(d / "summary_by_scene_range_batch.csv")
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))

    assert runs and timings and summary
    for row in runs:
        for key in ("latency_median_ms", "latency_mean_ms", "latency_min_ms", "latency_max_ms"):
            v = float(row[key])
            assert math.isfinite(v) and v > 0, (key, v)
    for row in timings:
        v = float(row["latency_ms"])
        assert math.isfinite(v) and v > 0, v

    assert len(runs) == int(manifest["runs"]), (len(runs), manifest["runs"])
    assert len(timings) == int(manifest["timed_samples"]), (len(timings), manifest["timed_samples"])
    keys = {(r["batch_size"], r["scene"], r["range_regions_requested"], r["timing_mode"]) for r in runs}
    assert len(summary) == len(keys), (len(summary), len(keys))

    by_run: dict[str, list[float]] = defaultdict(list)
    for row in timings:
        by_run[row["run_id"]].append(float(row["latency_ms"]))
    run_medians = {run_id: statistics.median(xs) for run_id, xs in by_run.items()}
    max_diff = max(abs(float(r["latency_median_ms"]) - run_medians[r["run_id"]]) for r in runs)
    assert max_diff < 1e-9, max_diff
    assert len(set(len(xs) for xs in by_run.values())) == 1

    assert sha256(d / "raw_runs.csv") == manifest["raw_runs_sha256"]
    assert sha256(d / "raw_timings.csv") == manifest["raw_timings_sha256"]

    print("PASS")
    print(f"  runs={len(runs)} timed_samples={len(timings)} summary_rows={len(summary)}")
    print(f"  iterations_per_run={len(next(iter(by_run.values())))}")
    print(f"  max |raw median - recomputed median| = {max_diff:.3e}")
    print(f"  raw_runs sha256={manifest['raw_runs_sha256']}")
    print(f"  raw_timings sha256={manifest['raw_timings_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())