#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GBSA (GenericBlockSparseAttention) single-op benchmark.

The benchmark measures only the Ascend aclnn GenericBlockSparseAttention core
operator.  IndexR / selection, metadata scheduling and the vLLM Triton
``append-current-region`` pack are deliberately kept outside the timed region.

The input K/V cache is a full 128 Ki-token Step-5-shaped BF16 cache:
    [num_pages, page_size=128, num_kv_groups=4, head_dim=192]
A deterministic structured-Gaussian generator is used so the cache is not a
zero/placeholder buffer but still reproducible on another machine.

For every candidate range (1k/4k/8k/16k regions) we generate top-512 logical
region ids by taking top-k on a reproducible, IndexR-like relevance score
inside the candidate window.  Three score patterns cover different top-k
dispersion modes:
  - recent_contiguous: one 512-region contiguous band near the recent end
  - multi_sink:        8 x 64-region bands spread across the candidate window
  - uniform:           diffuse/flat scores (top-k almost uniform over window)

The output is resumable and machine-portable:
  python3 run_bench.py --output-dir ./outputs/run_001
If the process is interrupted, re-run the same command with --resume.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as md
import json
import os
import platform
import random
import shutil
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# CLI / constants
# ---------------------------------------------------------------------------

def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_strs(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GBSA range / top-k dispersion benchmark")
    p.add_argument("--output-dir", default="outputs/gbsa_range_run")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--context-tokens", type=int, default=131072)
    p.add_argument("--region-size", type=int, default=8)
    p.add_argument("--topk", type=int, default=512)
    p.add_argument("--page-size", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=64)
    p.add_argument("--num-kv-groups", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=192)
    p.add_argument("--batch-sizes", default="1,8,32,128,256")
    p.add_argument("--ranges", default="1024,4096,8192,16384",
                   help="candidate window measured in logical 8-token regions")
    p.add_argument("--scenes", default="recent_contiguous,multi_sink,uniform")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--kv-seed", type=int, default=20261105)
    p.add_argument("--block-table-seed", type=int, default=20261106)
    p.add_argument("--selection-seed", type=int, default=7000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="small smoke matrix: batches 1,128; 2 seeds; 10 iters")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def safe_version(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except Exception:
        return None


def run_capture(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=30)
        return r.stdout.strip()
    except Exception as exc:
        return f"<failed: {exc}>"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_environment(device: str) -> dict[str, Any]:
    env = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        "npu_device": device,
        "env": {k: os.environ.get(k) for k in [
            "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH",
            "ASCEND_CUSTOM_OPP_PATH", "LD_LIBRARY_PATH", "PYTHONPATH",
            "ASCEND_RT_VISIBLE_DEVICES",
        ]},
        "python_packages": {p: safe_version(p) for p in [
            "torch", "torch-npu", "numpy", "pandas", "vllm", "vllm-ascend",
            "cann-ops-transformer", "triton",
        ]},
        "npu_smi": run_capture(["npu-smi", "info"]),
        "lscpu": run_capture(["lscpu"]),
    }
    try:
        import torch
        import torch_npu  # noqa: F401
        env["torch"] = {
            "__version__": torch.__version__,
            "npu_available": bool(torch.npu.is_available()),
            "npu_device_count": int(torch.npu.device_count()),
            "npu_device_name_0": torch.npu.get_device_name(0),
            "npu_snapshot": str(torch.npu.memory_snapshot()) if hasattr(torch.npu, "memory_snapshot") else None,
        }
    except Exception as exc:
        env["torch_error"] = repr(exc)
    return env


# ---------------------------------------------------------------------------
# Full-size deterministic K/V cache
# ---------------------------------------------------------------------------

def build_structured_cache(
    *,
    pages: int,
    page_size: int,
    groups: int,
    dim: int,
    regions_per_page: int,
    num_regions: int,
    seed: int,
    latent_std: float = 0.35,
    noise_std: float = 1.0,
    dtype: Any,
) -> Any:
    """Build a deterministic BF16 cache with token noise + per-region latent.

    Values are normalised around unit per-head RMS, which is representative of
    post-RMSNorm/RoPE K and V caches.  The function is CPU-side and returns a
    CPU tensor; callers move it to the accelerator.
    """
    import torch

    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((num_regions, groups, dim), dtype=np.float32)
    latent *= latent_std
    out = torch.empty((pages, page_size, groups, dim), dtype=torch.bfloat16)
    chunk_pages = 8
    for p0 in range(0, pages, chunk_pages):
        p1 = min(pages, p0 + chunk_pages)
        n = p1 - p0
        noise = rng.standard_normal((n, page_size, groups, dim), dtype=np.float32)
        noise *= noise_std
        regs = np.arange(p0 * regions_per_page, p1 * regions_per_page, dtype=np.int64)
        tokens_per_region = page_size // regions_per_page
        regs = np.repeat(regs, tokens_per_region)
        # noise: [n, page_size, groups, dim]; latent[regs]: [n*page_size, groups, dim]
        arr = noise + latent[regs].reshape(n, page_size, groups, dim)
        out[p0:p1] = torch.from_numpy(arr.astype(np.float32)).to(torch.bfloat16)
    return out


# ---------------------------------------------------------------------------
# IndexR-like selection score and top-k inside a candidate window
# ---------------------------------------------------------------------------

def make_scene_ids(
    *,
    scene: str,
    candidate_start: int,
    candidate_regions: int,
    batch_size: int,
    groups: int,
    topk: int,
    seed: int,
    pulse_db: float = 12.0,
) -> np.ndarray:
    """Return int32 ids [G, B, topk], sorted, unique and inside the window.

    Scores are Gaussian noise plus scene-specific rectangular pulses.  This is
    an IndexR-like surrogate with no selection-op timing mixed into GBSA.
    """
    rng = np.random.default_rng(seed)
    ids = np.empty((groups, batch_size, topk), dtype=np.int32)
    n = int(candidate_regions)
    if topk > n:
        raise ValueError(f"topk={topk} exceeds candidate_regions={n}")

    for g in range(groups):
        for b in range(batch_size):
            scores = rng.standard_normal(n, dtype=np.float32)
            if scene == "recent_contiguous":
                start = int(rng.integers(0, n - topk + 1))
                scores[start:start + topk] += pulse_db
            elif scene == "multi_sink":
                n_clusters = 8
                band = topk // n_clusters
                starts = np.rint(np.linspace(0, n - band, n_clusters)).astype(int)
                for s in starts:
                    scores[int(s):int(s) + band] += pulse_db
            elif scene == "uniform":
                pass
            else:
                raise ValueError(f"unknown scene {scene!r}")

            cand = np.argpartition(scores, -topk)[-topk:]
            cand.sort()
            ids[g, b] = (cand.astype(np.int32) + int(candidate_start))
    return ids


# ---------------------------------------------------------------------------
# Selection metrics
# ---------------------------------------------------------------------------

def compute_selection_metrics(
    *,
    ids: np.ndarray,
    candidate_regions: int,
    block_table_row: np.ndarray,
    regions_per_page: int,
) -> dict[str, float]:
    """Aggregate dispersion metrics across [G, B, topk] ids."""
    g, b, topk = ids.shape
    spans, norm_spans, mean_gaps, page_counts, page_spans = [], [], [], [], []
    group_union_pages = []
    for gi in range(g):
        group_ids = ids[gi].reshape(-1)
        group_logical_pages = np.unique(group_ids // regions_per_page)
        group_union_pages.append(int(np.unique(block_table_row[group_logical_pages]).size))
        for bi in range(b):
            row = ids[gi, bi]
            span = int(row[-1] - row[0] + 1)
            spans.append(span)
            norm_spans.append(span / float(candidate_regions))
            diffs = np.diff(row.astype(np.int64))
            mean_gaps.append(float(diffs.mean()) if diffs.size else 0.0)
            logical_pages = row // regions_per_page
            phys_pages = block_table_row[logical_pages]
            uniq = np.unique(phys_pages)
            page_counts.append(int(uniq.size))
            if uniq.size > 1:
                page_spans.append(int(uniq.max() - uniq.min() + 1))
            else:
                page_spans.append(1)

    all_ids = ids.reshape(-1)
    logical_pages_union = np.unique(all_ids // regions_per_page)
    phys_pages_union = np.unique(block_table_row[logical_pages_union])
    logical_regions_union = np.unique(all_ids)
    union_span = int(phys_pages_union.max() - phys_pages_union.min() + 1) if phys_pages_union.size else 0
    logical_union_span = int(logical_regions_union.max() - logical_regions_union.min() + 1) if logical_regions_union.size else 0

    def q(v: list[float], p: float) -> float:
        return float(np.quantile(np.asarray(v, dtype=np.float64), p))

    return {
        "selected_span_mean_regions": float(np.mean(spans)),
        "selected_span_p50_regions": q(spans, 0.50),
        "selected_span_p90_regions": q(spans, 0.90),
        "selected_span_norm_mean": float(np.mean(norm_spans)),
        "mean_nn_gap_regions": float(np.mean(mean_gaps)),
        "touched_pages_mean": float(np.mean(page_counts)),
        "touched_pages_p50": q(page_counts, 0.50),
        "touched_pages_p90": q(page_counts, 0.90),
        "touched_page_span_mean": float(np.mean(page_spans)),
        "logical_regions_union": int(logical_regions_union.size),
        "logical_union_span": int(logical_union_span),
        "logical_pages_union": int(logical_pages_union.size),
        "touched_pages_union": int(phys_pages_union.size),
        "touched_pages_union_page_span": int(union_span),
        "touched_pages_union_group_mean": float(np.mean(group_union_pages)),
        "touched_pages_union_group_max": int(np.max(group_union_pages)),
        "topk": int(topk),
    }


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------

def summarize_samples(samples: list[float]) -> dict[str, float]:
    a = np.asarray(samples, dtype=np.float64)
    return {
        "latency_ms_mean": float(a.mean()),
        "latency_ms_median": float(np.median(a)),
        "latency_ms_p10": float(np.quantile(a, 0.10)),
        "latency_ms_p90": float(np.quantile(a, 0.90)),
        "latency_ms_min": float(a.min()),
        "latency_ms_max": float(a.max()),
        "latency_ms_std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
    }


def benchmark_metadata(
    *,
    generic_metadata: Any,
    wide: Any,
    counts: Any,
    num_heads: int,
    num_kv_groups: int,
    head_dim: int,
    region_size: int,
    cu: Any,
    seqused: Any,
    context_tokens: int,
    iters: int,
) -> dict[str, float]:
    import torch
    # One untimed call so all JIT/ACL setup is complete.
    generic_metadata(
        wide, counts, num_heads, num_kv_groups, head_dim, [1, region_size],
        cu_seqlens_q=cu, seqused_kv=seqused, max_seqlen_q=1,
        max_seqlen_kv=context_tokens, layout_q="TND", layout_kv="PA_BBND",
        layout_sparse_pattern=4, mask_mode=1, quant_mode=0,
        softmax_precision=1, is_consistent_topk=False,
    )
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        generic_metadata(
            wide, counts, num_heads, num_kv_groups, head_dim, [1, region_size],
            cu_seqlens_q=cu, seqused_kv=seqused, max_seqlen_q=1,
            max_seqlen_kv=context_tokens, layout_q="TND", layout_kv="PA_BBND",
            layout_sparse_pattern=4, mask_mode=1, quant_mode=0,
            softmax_precision=1, is_consistent_topk=False,
        )
        ends[i].record()
    torch.npu.synchronize()
    vals = [float(s.elapsed_time(e)) for s, e in zip(starts, ends)]
    base = summarize_samples(vals)
    return {k.replace('latency_ms_', 'metadata_ms_'): v for k, v in base.items()}


def main() -> int:
    args = parse_args()
    if args.quick:
        args.batch_sizes = "1,128"
        args.seeds = 2
        args.iters = 10
        args.warmup = 3

    import torch
    import torch_npu  # noqa: F401

    from cann_ops_transformer.ops.attention.generic_block_sparse_attention import (
        generic_block_sparse_attention,
        generic_block_sparse_attention_metadata,
    )

    batches = parse_csv_ints(args.batch_sizes)
    ranges = parse_csv_ints(args.ranges)
    scenes = parse_csv_strs(args.scenes)

    if not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available")
    if not args.device.startswith("npu"):
        raise RuntimeError("this benchmark expects an Ascend NPU device, e.g. --device npu:0")
    device = torch.device(args.device)
    torch.npu.set_device(device)

    ctx = int(args.context_tokens)
    region_size = int(args.region_size)
    topk = int(args.topk)
    page_size = int(args.page_size)
    num_heads = int(args.num_heads)
    groups = int(args.num_kv_groups)
    head_dim = int(args.head_dim)
    if ctx % region_size or ctx % page_size:
        raise ValueError("context-tokens must be divisible by region-size and page-size")
    if page_size % region_size:
        raise ValueError("page-size must be a multiple of region-size")
    num_regions = ctx // region_size
    pages = ctx // page_size
    regions_per_page = page_size // region_size
    current_region = num_regions - 1
    softmax_scale = float(head_dim) ** -0.5
    page_kv_bytes = page_size * groups * head_dim * 2 * 2  # K+V, BF16
    max_batch = max(batches)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    config = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "args": vars(args),
        "shape": {
            "context_tokens": ctx,
            "num_regions_total": num_regions,
            "current_region": current_region,
            "page_size": page_size,
            "num_pages": pages,
            "regions_per_page": regions_per_page,
            "num_heads": num_heads,
            "num_kv_groups": groups,
            "head_dim": head_dim,
            "topk": topk,
            "region_size": region_size,
            "kv_dtype": "bfloat16",
            "k_cache_shape": [pages, page_size, groups, head_dim],
            "v_cache_shape": [pages, page_size, groups, head_dim],
            "q_shape_per_token": [num_heads, head_dim],
            "softmax_scale": softmax_scale,
        },
        "selection_model": {
            "source": "deterministic IndexR-score surrogate: Gaussian score + scene pulse, top-k inside candidate window",
            "candidate_window": "last min(W, num_regions_total-1) historical regions, current region appended by GBSA contract",
            "pulse_db": 12.0,
            "scenes": {
                "recent_contiguous": "one contiguous top-k region band near the recent end",
                "multi_sink": "8 contiguous bands of topk/8 regions evenly spread over the candidate window",
                "uniform": "flat/diffuse score, top-k approximately uniform over the candidate window",
            },
        },
        "timed_scope": "aclnn GenericBlockSparseAttention core op only; metadata/prepack/indexer excluded",
        "is_consistent_topk": False,
        "mask_mode": 1,
        "layout_q": "TND",
        "layout_kv": "PA_BBND",
    }

    env = collect_environment(args.device)
    env["command_line"] = " ".join([sys.executable] + sys.argv)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")

    # Deterministic checkpoint files.
    runs_path = out_dir / "raw_runs.csv"
    timings_path = out_dir / "raw_timings.csv"
    meta_path = out_dir / "metadata_bench.csv"
    existing_ids: set[str] = set()
    if args.resume and runs_path.exists():
        with runs_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_ids.add(row["run_id"])
        print(f"[resume] {len(existing_ids)} completed runs found", flush=True)
    else:
        if runs_path.exists():
            runs_path.unlink()
        if timings_path.exists():
            timings_path.unlink()
        if meta_path.exists():
            meta_path.unlink()

    run_cols = [
        "run_id", "timestamp", "scene", "range_regions_requested", "range_label",
        "candidate_regions_effective", "candidate_tokens_effective", "batch_size",
        "seed", "kv_seed", "block_table_seed", "topk", "region_size", "page_size",
        "context_tokens", "num_regions_total", "num_pages", "num_heads",
        "num_kv_groups", "head_dim",
        "selected_span_mean_regions", "selected_span_p50_regions",
        "selected_span_p90_regions", "selected_span_norm_mean",
        "mean_nn_gap_regions", "touched_pages_mean", "touched_pages_p50",
        "touched_pages_p90", "touched_page_span_mean",
        "logical_regions_union", "logical_union_span", "logical_pages_union",
        "touched_pages_union", "touched_pages_union_page_span",
        "touched_pages_union_group_mean", "touched_pages_union_group_max",
        "kv_working_set_mib", "kv_working_set_mib_per_request",
        "latency_ms_mean", "latency_ms_median", "latency_ms_p10", "latency_ms_p90",
        "latency_ms_min", "latency_ms_max", "latency_ms_std",
        "latency_ms_per_request", "out_abs_mean", "out_finite",
    ]
    run_fh = runs_path.open("a", newline="", encoding="utf-8")
    run_writer = csv.DictWriter(run_fh, fieldnames=run_cols)
    if run_fh.tell() == 0:
        run_writer.writeheader()
        run_fh.flush()

    timings_fh = timings_path.open("a", newline="", encoding="utf-8")
    timings_writer = csv.DictWriter(timings_fh, fieldnames=["run_id", "iter", "latency_ms"])
    if timings_fh.tell() == 0:
        timings_writer.writeheader()
        timings_fh.flush()

    meta_fh = meta_path.open("a", newline="", encoding="utf-8")
    meta_writer = csv.DictWriter(meta_fh, fieldnames=[
        "batch_size", "metadata_ms_mean", "metadata_ms_median",
        "metadata_ms_p10", "metadata_ms_p90", "metadata_ms_min", "metadata_ms_max", "metadata_ms_std",
    ])
    if meta_fh.tell() == 0:
        meta_writer.writeheader()
        meta_fh.flush()

    print("[1/4] building full-size KV cache ...", flush=True)
    t0 = time.time()
    k_cpu = build_structured_cache(
        pages=pages, page_size=page_size, groups=groups, dim=head_dim,
        regions_per_page=regions_per_page, num_regions=num_regions,
        seed=args.kv_seed, dtype=torch.bfloat16,
    )
    k = k_cpu.to(device=device, non_blocking=False)
    del k_cpu
    v_cpu = build_structured_cache(
        pages=pages, page_size=page_size, groups=groups, dim=head_dim,
        regions_per_page=regions_per_page, num_regions=num_regions,
        seed=args.kv_seed + 1, dtype=torch.bfloat16,
    )
    v = v_cpu.to(device=device, non_blocking=False)
    del v_cpu
    torch.npu.synchronize()
    kv_seconds = time.time() - t0
    print(f"      cache ready in {kv_seconds:.2f}s, "
          f"K+V host bytes={(k.numel() + v.numel()) * 2 / 2**20:.1f} MiB", flush=True)

    # A fixed pseudo-random physical-page mapping, repeated per request.
    base_perm = np.random.default_rng(args.block_table_seed).permutation(pages).astype(np.int32)
    block_table_np = np.tile(base_perm[None, :], (max_batch, 1))
    block_table = torch.from_numpy(block_table_np).to(device=device, dtype=torch.int32)
    block_table_cpu = block_table_np

    # Q and positions are fixed across all runs of a given seed/batch.
    # Positions stay in the final 8-token region so every row has count=513.
    q_cache: dict[int, Any] = {}
    pos_cache: dict[int, Any] = {}
    cu_cache: dict[int, Any] = {}
    seq_cache: dict[int, Any] = {}
    count_cache: dict[int, Any] = {}
    meta_cache: dict[int, Any] = {}
    for bsz in batches:
        qgen = torch.Generator(device="cpu").manual_seed(args.selection_seed + bsz)
        q_cpu = torch.randn((bsz, num_heads, head_dim), generator=qgen, dtype=torch.bfloat16)
        q_cache[bsz] = q_cpu.to(device=device)
        pos_vals = np.arange(ctx - region_size, ctx, dtype=np.int64)
        pos_vals = np.resize(pos_vals, bsz)
        pos_cache[bsz] = torch.from_numpy(pos_vals).to(device=device, dtype=torch.int64)
        cu_cache[bsz] = torch.arange(bsz + 1, dtype=torch.int64, device=device)
        seq_cache[bsz] = torch.full((bsz,), ctx, dtype=torch.int32, device=device)
        count_cache[bsz] = torch.full((groups, bsz), topk + 1, dtype=torch.int32, device=device)

    print("[2/4] compiling / warming the core op ...", flush=True)
    # Build one dummy wide tensor and run one call before the measured matrix.
    dummy_ids = make_scene_ids(
        scene="uniform", candidate_start=max(0, current_region - 1024),
        candidate_regions=min(1024, current_region), batch_size=1, groups=groups,
        topk=topk, seed=1,
    )
    dummy_wide = torch.empty((groups, 1, topk + 1), dtype=torch.int32, device=device)
    dummy_wide[:, :, :topk] = torch.from_numpy(dummy_ids[:, 0, :]).unsqueeze(1).to(device=device)
    dummy_wide[:, :, topk] = current_region
    dummy_counts = torch.full((groups, 1), topk + 1, dtype=torch.int32, device=device)
    dummy_cu = torch.tensor([0, 1], dtype=torch.int64, device=device)
    dummy_seq = torch.tensor([ctx], dtype=torch.int32, device=device)
    dummy_q = torch.randn((1, num_heads, head_dim), dtype=torch.bfloat16, device=device)
    dummy_bt = block_table[:1].clone()
    dummy_meta = generic_block_sparse_attention_metadata(
        dummy_wide, dummy_counts, num_heads, groups, head_dim, [1, region_size],
        cu_seqlens_q=dummy_cu, seqused_kv=dummy_seq,
        max_seqlen_q=1, max_seqlen_kv=ctx, layout_q="TND", layout_kv="PA_BBND",
        layout_sparse_pattern=4, mask_mode=1, quant_mode=0,
        softmax_precision=1, is_consistent_topk=False,
    )
    _ = generic_block_sparse_attention(
        dummy_q, k, v, dummy_wide, dummy_counts, [1, region_size],
        metadata=dummy_meta, cu_seqlens_q=dummy_cu,
        seqused_kv=dummy_seq, block_table=dummy_bt,
        layout_q="TND", layout_kv="PA_BBND", layout_sparse_pattern=4,
        softmax_scale=softmax_scale, mask_mode=1, quant_mode=0,
        softmax_precision=1, return_softmax_lse=False, is_consistent_topk=False,
    )
    torch.npu.synchronize()
    print("      core op warmup complete", flush=True)

    # Pre-build metadata per batch.  Values depend only on shape/counts here.
    print("[3/4] building metadata tensors ...", flush=True)
    for bsz in batches:
        dummy_wide_b = torch.zeros((groups, bsz, topk + 1), dtype=torch.int32, device=device)
        meta_cache[bsz] = generic_block_sparse_attention_metadata(
            dummy_wide_b, count_cache[bsz], num_heads, groups, head_dim, [1, region_size],
            cu_seqlens_q=cu_cache[bsz], seqused_kv=seq_cache[bsz], max_seqlen_q=1,
            max_seqlen_kv=ctx, layout_q="TND", layout_kv="PA_BBND",
            layout_sparse_pattern=4, mask_mode=1, quant_mode=0,
            softmax_precision=1, is_consistent_topk=False,
        )
    torch.npu.synchronize()

    # Metadata micro-benchmark (not part of headline latency).
    for bsz in batches:
        dummy_wide_b = torch.zeros((groups, bsz, topk + 1), dtype=torch.int32, device=device)
        mt = benchmark_metadata(
            generic_metadata=generic_block_sparse_attention_metadata,
            wide=dummy_wide_b, counts=count_cache[bsz], num_heads=num_heads,
            num_kv_groups=groups, head_dim=head_dim, region_size=region_size,
            cu=cu_cache[bsz], seqused=seq_cache[bsz], context_tokens=ctx,
            iters=max(10, min(30, args.iters)),
        )
        meta_writer.writerow({"batch_size": bsz, **mt})
        meta_fh.flush()

    # Reusable event arrays eliminate per-iteration NPU event allocation.
    starts = [torch.npu.Event(enable_timing=True) for _ in range(args.iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(args.iters)]

    scene_order = list(scenes)
    total_specs = len(batches) * len(scene_order) * len(ranges) * args.seeds
    done_this_process = 0
    print(f"[4/4] running {total_specs} measured scenarios "
          f"({args.warmup} warmup + {args.iters} timed each) ...", flush=True)

    for bsz in batches:
        q = q_cache[bsz]
        positions = pos_cache[bsz]
        cu = cu_cache[bsz]
        seqused = seq_cache[bsz]
        counts_full = count_cache[bsz]
        metadata = meta_cache[bsz]
        bt = block_table[:bsz]
        for scene in scene_order:
            for range_requested in ranges:
                candidate_regions = min(int(range_requested), current_region)
                if candidate_regions < topk:
                    raise ValueError(
                        f"range {range_requested} has only {candidate_regions} candidates, < topk={topk}"
                    )
                candidate_start = current_region - candidate_regions
                for seed_i in range(int(args.seeds)):
                    run_id = (
                        f"{scene}__r{range_requested}__b{bsz}__s{args.selection_seed + seed_i}"
                    )
                    if run_id in existing_ids:
                        continue
                    ids_np = make_scene_ids(
                        scene=scene, candidate_start=candidate_start,
                        candidate_regions=candidate_regions,
                        batch_size=bsz, groups=groups, topk=topk,
                        seed=args.selection_seed + seed_i,
                    )
                    if ids_np.min() < 0 or ids_np.max() >= current_region:
                        raise AssertionError("selected id out of legal historical range")
                    sel = compute_selection_metrics(
                        ids=ids_np, candidate_regions=candidate_regions,
                        block_table_row=block_table_cpu[0], regions_per_page=regions_per_page,
                    )
                    wide = torch.empty((groups, bsz, topk + 1),
                                       dtype=torch.int32, device=device)
                    wide[:, :, :topk] = torch.from_numpy(ids_np).to(device=device)
                    wide[:, :, topk] = current_region

                    # Warmup.  The op allocates its own output, matching the
                    # production aclnn adapter call.
                    for _ in range(args.warmup):
                        out, _lse = generic_block_sparse_attention(
                            q, k, v, wide, counts_full, [1, region_size],
                            metadata=metadata, cu_seqlens_q=cu, seqused_kv=seqused,
                            block_table=bt, layout_q="TND", layout_kv="PA_BBND",
                            layout_sparse_pattern=4, softmax_scale=softmax_scale,
                            mask_mode=1, quant_mode=0, softmax_precision=1,
                            return_softmax_lse=False, is_consistent_topk=False,
                        )
                    torch.npu.synchronize()

                    for i in range(args.iters):
                        starts[i].record()
                        out, _lse = generic_block_sparse_attention(
                            q, k, v, wide, counts_full, [1, region_size],
                            metadata=metadata, cu_seqlens_q=cu, seqused_kv=seqused,
                            block_table=bt, layout_q="TND", layout_kv="PA_BBND",
                            layout_sparse_pattern=4, softmax_scale=softmax_scale,
                            mask_mode=1, quant_mode=0, softmax_precision=1,
                            return_softmax_lse=False, is_consistent_topk=False,
                        )
                        ends[i].record()
                    torch.npu.synchronize()
                    samples = [float(s.elapsed_time(e)) for s, e in zip(starts, ends)]
                    st = summarize_samples(samples)
                    finite = bool(torch.isfinite(out).all().item())
                    out_abs = float(out.float().abs().mean().item())
                    if not finite:
                        raise RuntimeError(f"non-finite output for {run_id}")

                    rec = {
                        "run_id": run_id,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "scene": scene,
                        "range_regions_requested": int(range_requested),
                        "range_label": (f"{range_requested // 1024}k"
                                        if range_requested % 1024 == 0 else str(range_requested)),
                        "candidate_regions_effective": int(candidate_regions),
                        "candidate_tokens_effective": int(candidate_regions * region_size),
                        "batch_size": int(bsz),
                        "seed": int(args.selection_seed + seed_i),
                        "kv_seed": int(args.kv_seed),
                        "block_table_seed": int(args.block_table_seed),
                        "topk": int(topk),
                        "region_size": int(region_size),
                        "page_size": int(page_size),
                        "context_tokens": int(ctx),
                        "num_regions_total": int(num_regions),
                        "num_pages": int(pages),
                        "num_heads": int(num_heads),
                        "num_kv_groups": int(groups),
                        "head_dim": int(head_dim),
                        **sel,
                        **st,
                        "kv_working_set_mib": float(sel["touched_pages_union"] * page_kv_bytes / 2**20),
                        "kv_working_set_mib_per_request": float(sel["touched_pages_mean"] * page_kv_bytes / 2**20),
                        "latency_ms_per_request": float(st["latency_ms_median"] / bsz),
                        "out_abs_mean": out_abs,
                        "out_finite": finite,
                    }
                    run_writer.writerow(rec)
                    run_fh.flush()
                    os.fsync(run_fh.fileno())
                    for i, vms in enumerate(samples):
                        timings_writer.writerow({"run_id": run_id, "iter": i, "latency_ms": vms})
                    timings_fh.flush()
                    done_this_process += 1
                    if done_this_process % 20 == 0 or done_this_process == 1:
                        print(
                            f"  [{done_this_process}] {run_id} "
                            f"span={sel['selected_span_mean_regions']:.0f} "
                            f"pages={sel['touched_pages_mean']:.1f} "
                            f"median={st['latency_ms_median']:.4f} ms",
                            flush=True,
                        )
        print(f"  batch {bsz} complete", flush=True)

    run_fh.close()
    timings_fh.close()
    meta_fh.close()

    manifest = {
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_runs": len(existing_ids) + done_this_process,
        "runs_this_process": done_this_process,
        "kv_build_seconds": kv_seconds,
        "raw_runs_sha256": sha256_file(runs_path),
        "raw_timings_sha256": sha256_file(timings_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"done: {manifest['total_runs']} runs -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())