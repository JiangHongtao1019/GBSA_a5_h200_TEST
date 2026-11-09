#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H200 + vLLM-compatible GBSA sparse-attention benchmark.

This is an operator microbenchmark driven through a vLLM AttentionImpl-style
forward (``GBSAH200Impl``).  It deliberately does not time IndexR / scheduler /
detokenization, matching the Ascend GBSA timing boundary.

Two timing modes are produced:
  gather_sdpa: selected-region gather + SDPA (FlashAttention on H200)
  sdpa_only:   SDPA on pre-gathered K/V (compute-only reference)

Run:
  python h200_vllm/run_bench.py --output-dir outputs_h200_vllm/full_12seed \
      --seeds 12 --warmup 10 --iters 50
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Callable

import numpy as np
import torch

try:
    from workload import (
        SCENES, WorkloadConfig, build_kv_cache, gather_kv, fixed_permuted_block_table,
        make_scene_ids, selected_token_indices, selection_metrics,
    )
    from vllm_backend import GBSAH200Impl, GBSAMetadata, VLLM_AVAILABLE
except ImportError:  # package import: python -m h200_vllm.run_bench
    from .workload import (
        SCENES, WorkloadConfig, build_kv_cache, gather_kv, fixed_permuted_block_table,
        make_scene_ids, selected_token_indices, selection_metrics,
    )
    from .vllm_backend import GBSAH200Impl, GBSAMetadata, VLLM_AVAILABLE


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_strs(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def make_query(batch: int, heads: int, head_dim: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(20250901 + batch)
    return torch.randn((batch, heads, head_dim), dtype=torch.bfloat16, device=device, generator=gen)


def time_call(fn: Callable[[], torch.Tensor], warmup: int, iters: int,
              device: torch.device) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    with torch.inference_mode():
        for s, e in zip(starts, ends):
            s.record()
            fn()
            e.record()
        torch.cuda.synchronize(device)
    return [float(s.elapsed_time(e)) for s, e in zip(starts, ends)]


def env_dict(device: torch.device, cfg: WorkloadConfig, args: argparse.Namespace) -> dict:
    props = torch.cuda.get_device_properties(device)
    versions = {"torch": torch.__version__, "cuda": torch.version.cuda}
    try:
        import vllm
        versions["vllm"] = getattr(vllm, "__version__", "unknown")
    except Exception:
        versions["vllm"] = "not-installed"
    try:
        import triton
        versions["triton"] = getattr(triton, "__version__", "unknown")
    except Exception:
        versions["triton"] = "not-installed"
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], text=True, timeout=10).strip()
    except Exception:
        smi = "unavailable"
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "device": str(device),
        "device_name": props.name,
        "capability": list(torch.cuda.get_device_capability(device)),
        "total_memory_GiB": round(props.total_memory / (1024 ** 3), 2),
        "nvidia_smi": smi,
        "versions": versions,
        "workload": cfg.to_dict(),
        "args": vars(args),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="outputs_h200_vllm/full_12seed")
    ap.add_argument("--batches", default="1,8,32,128,256")
    ap.add_argument("--scenes", default="recent_contiguous,multi_sink,uniform")
    ap.add_argument("--ranges", default="1024,4096,8192,16384")
    ap.add_argument("--modes", default="gather_sdpa,sdpa_only")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--kv-mode", choices=["shared", "per_request"], default="shared")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed-base", type=int, default=7000,
                    help="base seed; run seed is seed-base + selection_seed")
    ap.add_argument("--check-only", action="store_true",
                    help="only validate imports and one tiny forward")
    ap.add_argument("--require-vllm", action="store_true",
                    help="fail if vLLM is not importable")
    args = ap.parse_args()

    if args.require_vllm and not VLLM_AVAILABLE:
        raise RuntimeError("vLLM is required but could not be imported")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this runner is for NVIDIA H200")
    device = torch.device(args.device)
    batches = parse_ints(args.batches)
    scenes = parse_strs(args.scenes)
    ranges = parse_ints(args.ranges)
    modes = parse_strs(args.modes)
    for name in scenes:
        if name not in SCENES:
            raise ValueError(f"unknown scene {name!r}")

    if args.check_only:
        cfg = WorkloadConfig(batch_size=1, candidate_regions=1024, kv_mode=args.kv_mode)
        k, v = build_kv_cache(cfg, device=device, seed=args.seed_base)
        bt = fixed_permuted_block_table(cfg, device)
        candidate_regions = min(1024, cfg.current_region)
        candidate_start = cfg.current_region - candidate_regions
        ids = make_scene_ids(scene="uniform", candidate_start=candidate_start,
                             candidate_regions=candidate_regions,
                             batch_size=1, kv_groups=cfg.kv_groups, topk=cfg.topk,
                             seed=args.seed_base)
        wide = np.concatenate(
            [ids, np.full((cfg.kv_groups, 1, 1), cfg.current_region, dtype=np.int32)],
            axis=2,
        )
        idx = selected_token_indices(wide, bt, cfg)
        print("check:", k.shape, v.shape, wide.shape, idx.shape,
              selection_metrics(ids, bt, cfg))
        return 0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_runs: list[dict] = []
    raw_timings: list[dict] = []
    base_cfg = WorkloadConfig(batch_size=max(batches), kv_mode=args.kv_mode)
    (out_dir / "config.json").write_text(json.dumps(base_cfg.to_dict(), indent=2), encoding="utf-8")
    (out_dir / "environment.json").write_text(
        json.dumps(env_dict(device, base_cfg, args), indent=2), encoding="utf-8")

    # Shared KV is built once; per-request KV is built per batch size.
    shared_cache = None
    if args.kv_mode == "shared":
        print("[gbsa-h200] building shared KV cache ...", flush=True)
        shared_cache = build_kv_cache(base_cfg, device=device, seed=2025)

    t0 = time.time()
    for batch in batches:
        cfg = WorkloadConfig(batch_size=batch, kv_mode=args.kv_mode)
        if args.kv_mode == "shared":
            assert shared_cache is not None
            cache_k, cache_v = shared_cache
        else:
            print(f"[gbsa-h200] building per-request KV B={batch} ...", flush=True)
            cache_k, cache_v = build_kv_cache(cfg, device=device, seed=2025)
        block_table = fixed_permuted_block_table(cfg, device, seed=4242)
        q = make_query(batch, cfg.heads, cfg.head_dim, device)
        impl = GBSAH200Impl(num_heads=cfg.heads, head_size=cfg.head_dim,
                            num_kv_heads=cfg.kv_groups).to(device)

        for selected_range in ranges:
            for scene in scenes:
                for selection_seed in range(args.seeds):
                    seed = args.seed_base + selection_seed
                    candidate_regions = min(selected_range, cfg.current_region)
                    candidate_start = cfg.current_region - candidate_regions
                    ids_np = make_scene_ids(
                        scene=scene, candidate_start=candidate_start,
                        candidate_regions=candidate_regions,
                        batch_size=batch, kv_groups=cfg.kv_groups, topk=cfg.topk, seed=seed,
                    )
                    # GBSA always appends the current region to the top-k ids.
                    wide_np = np.concatenate(
                        [ids_np, np.full((cfg.kv_groups, batch, 1), cfg.current_region, dtype=np.int32)],
                        axis=2,
                    )
                    metrics = selection_metrics(ids_np, block_table, cfg)
                    token_idx = selected_token_indices(wide_np, block_table, cfg)
                    pre_k = None
                    pre_v = None
                    if "sdpa_only" in modes:
                        pre_k = gather_kv(cache_k, token_idx, cfg)
                        pre_v = gather_kv(cache_v, token_idx, cfg)

                    for mode in modes:
                        run_id = f"B{batch}_W{selected_range}_{scene}_s{selection_seed}_{mode}"
                        meta = GBSAMetadata(
                            cache_k=cache_k, cache_v=cache_v, block_table=block_table,
                            token_idx=token_idx, cfg=cfg,
                            precomputed_kv=(pre_k, pre_v) if mode == "sdpa_only" else None,
                        )

                        def call(meta=meta) -> torch.Tensor:
                            return impl.forward(None, q, None, None, None, meta)

                        samples = time_call(call, args.warmup, args.iters, device)
                        run = {
                            "run_id": run_id,
                            "batch_size": batch,
                            "scene": scene,
                            "range_regions_requested": selected_range,
                            "candidate_regions_effective": candidate_regions,
                            "candidate_start": candidate_start,
                            "timing_mode": mode,
                            "selection_seed": selection_seed,
                            "seed": seed,
                            "latency_median_ms": statistics.median(samples),
                            "latency_mean_ms": statistics.fmean(samples),
                            "latency_min_ms": min(samples),
                            "latency_max_ms": max(samples),
                            "latency_std_ms": statistics.pstdev(samples),
                            **metrics,
                        }
                        raw_runs.append(run)
                        for i, x in enumerate(samples):
                            raw_timings.append({
                                "run_id": run_id,
                                "batch_size": batch,
                                "scene": scene,
                                "range_regions_requested": selected_range,
                                "timing_mode": mode,
                                "selection_seed": selection_seed,
                                "iteration": i,
                                "latency_ms": x,
                            })
                print(f"[gbsa-h200] B={batch} W={selected_range} {scene} done", flush=True)
        if args.kv_mode == "per_request":
            del cache_k, cache_v
            torch.cuda.empty_cache()

    write_csv(out_dir / "raw_runs.csv", raw_runs)
    write_csv(out_dir / "raw_timings.csv", raw_timings)

    # Per-run medians are the unit of aggregation across selection seeds.
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in raw_runs:
        key = (row["batch_size"], row["scene"], row["range_regions_requested"], row["timing_mode"])
        groups[key].append(row)

    summary: list[dict] = []
    for (batch, scene, selected_range, mode), vals in groups.items():
        lat = [float(v["latency_median_ms"]) for v in vals]
        summary.append({
            "batch_size": batch,
            "scene": scene,
            "range_regions_requested": selected_range,
            "timing_mode": mode,
            "n_runs": len({int(v["selection_seed"]) for v in vals}),
            "latency_median_ms": float(statistics.median(lat)),
            "latency_mean_ms": float(statistics.fmean(lat)),
            "latency_q25_ms": float(np.percentile(lat, 25)),
            "latency_q75_ms": float(np.percentile(lat, 75)),
            "per_request_pages_mean": float(statistics.median([float(v["per_request_pages_mean"]) for v in vals])),
            "per_request_pages_median": float(statistics.median([float(v["per_request_pages_median"]) for v in vals])),
            "union_pages_median": float(statistics.median([float(v["union_pages"]) for v in vals])),
            "working_set_mib_median": float(statistics.median([float(v["working_set_MiB"]) for v in vals])),
        })
    for row in summary:
        row["latency_iqr_ms"] = row["latency_q75_ms"] - row["latency_q25_ms"]

    # Align the summary schema with the Ascend report where possible.
    base = {(r["batch_size"], r["scene"], r["timing_mode"]): r["latency_median_ms"]
            for r in summary if r["range_regions_requested"] == 1024}
    for row in summary:
        b0 = base.get((row["batch_size"], row["scene"], row["timing_mode"]))
        row["latency_w1k_ms"] = b0
        if b0 is not None:
            row["delta_ms_vs_w1k"] = row["latency_median_ms"] - b0
            row["speedup_x_vs_w1k"] = row["latency_median_ms"] / b0
            row["delta_pct_vs_w1k"] = row["speedup_x_vs_w1k"] * 100.0 - 100.0
        else:
            row["delta_ms_vs_w1k"] = None
            row["speedup_x_vs_w1k"] = None
            row["delta_pct_vs_w1k"] = None
    summary.sort(key=lambda r: (r["timing_mode"], r["batch_size"], r["scene"], r["range_regions_requested"]))
    write_csv(out_dir / "summary_by_scene_range_batch.csv", summary)
    raw_sorted = sorted(raw_runs, key=lambda r: (r["timing_mode"], r["batch_size"], r["scene"],
                                                r["range_regions_requested"], r["selection_seed"]))
    write_csv(out_dir / "raw_runs_sorted.csv", raw_sorted)

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": round(time.time() - t0, 3),
        "runs": len(raw_runs),
        "timed_samples": len(raw_timings),
        "seeds": args.seeds,
        "warmup": args.warmup,
        "iters": args.iters,
        "batches": batches,
        "scenes": scenes,
        "ranges": ranges,
        "modes": modes,
        "kv_mode": args.kv_mode,
        "raw_runs_sha256": hashlib_sha256(out_dir / "raw_runs.csv"),
        "raw_timings_sha256": hashlib_sha256(out_dir / "raw_timings.csv"),
        "note": "H200/vLLM-compatible operator microbenchmark; not a full LLM serving benchmark.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(raw_runs)} runs / {len(raw_timings)} timed samples -> {out_dir}")
    return 0


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def hashlib_sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())