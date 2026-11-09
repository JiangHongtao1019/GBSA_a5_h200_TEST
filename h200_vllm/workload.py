#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H200/vLLM workload generation: Step5-shaped KV cache and controlled top-512 ids.

The KV layout mirrors the Ascend GBSA test:
    cache_k/cache_v: [num_pages, page_size, kv_groups, head_dim]
    topk_ids:        [kv_groups, batch, topk]   (historical logical regions)
    wide_ids:        [kv_groups, batch, topk+1] (topk historical + current region)
    block_table:     [batch, num_pages]

The selection patterns are deliberately controlled IndexR-like surrogates, not
actual IndexR output.  This keeps the GBSA timing isolated from selection cost.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np
import torch

SCENES = ("recent_contiguous", "multi_sink", "uniform")


@dataclass(frozen=True)
class WorkloadConfig:
    batch_size: int = 128
    kv_groups: int = 4
    heads: int = 64
    head_dim: int = 192
    page_size: int = 128
    region_size: int = 8
    topk: int = 512
    num_pages: int = 1024
    candidate_regions: int = 16384
    kv_mode: Literal["shared", "per_request"] = "shared"
    dtype: str = "bfloat16"

    @property
    def regions_per_page(self) -> int:
        return self.page_size // self.region_size

    @property
    def total_selected_regions(self) -> int:
        # GBSA contract: topk historical regions + the current region.
        return self.topk + 1

    @property
    def historical_selected_tokens(self) -> int:
        return self.topk * self.region_size

    @property
    def selected_tokens(self) -> int:
        return self.total_selected_regions * self.region_size

    @property
    def num_regions_total(self) -> int:
        return self.num_pages * self.page_size // self.region_size

    @property
    def current_region(self) -> int:
        return self.num_regions_total - 1

    @property
    def page_bytes(self) -> int:
        # K + V, BF16
        return 2 * self.page_size * self.kv_groups * self.head_dim * 2

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(
            regions_per_page=self.regions_per_page,
            total_selected_regions=self.total_selected_regions,
            historical_selected_tokens=self.historical_selected_tokens,
            selected_tokens=self.selected_tokens,
            num_regions_total=self.num_regions_total,
            current_region=self.current_region,
            page_bytes=self.page_bytes,
        )
        return d


def make_scene_ids(
    *,
    scene: str,
    candidate_start: int,
    candidate_regions: int,
    batch_size: int,
    kv_groups: int,
    topk: int,
    seed: int,
    pulse: float = 12.0,
) -> np.ndarray:
    """Return int32 ids [G, B, topk], sorted, unique, absolute region ids.

    This is an exact port of the Ascend benchmark's controlled construction:
      - recent_contiguous: one random 512-region contiguous band per (g,b)
      - multi_sink: 8 x 64-region bands at fixed positions, shared by all (g,b)
      - uniform: top-512 of a flat Gaussian score
    """
    rng = np.random.default_rng(seed)
    n = int(candidate_regions)
    if topk > n:
        raise ValueError(f"topk={topk} exceeds candidate_regions={n}")
    out = np.empty((kv_groups, batch_size, topk), dtype=np.int32)
    for g in range(kv_groups):
        for b in range(batch_size):
            scores = rng.standard_normal(n, dtype=np.float32)
            if scene == "recent_contiguous":
                start = int(rng.integers(0, n - topk + 1))
                scores[start:start + topk] += pulse
            elif scene == "multi_sink":
                n_clusters = 8
                band = topk // n_clusters
                starts = np.rint(np.linspace(0, n - band, n_clusters)).astype(int)
                for s in starts:
                    scores[int(s):int(s) + band] += pulse
            elif scene == "uniform":
                pass
            else:
                raise ValueError(f"unknown scene {scene!r}")
            cand = np.argpartition(scores, -topk)[-topk:]
            cand.sort()
            out[g, b] = cand.astype(np.int32) + int(candidate_start)
    return out


def build_kv_cache(cfg: WorkloadConfig, *, device: str | torch.device, seed: int = 2025):
    """Build deterministic structured BF16 K/V caches.

    shared:      one [P, page, G, D] K and V, shared by all requests
    per_request: [B, P, page, G, D] K and V, one full 128K cache per request

    The per-request mode is memory heavy: B=128 x 384 MiB ~= 48 GiB for K+V.
    """
    dev = torch.device(device)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    shape = (cfg.num_pages, cfg.page_size, cfg.kv_groups, cfg.head_dim)
    if cfg.kv_mode == "per_request":
        shape = (cfg.batch_size,) + shape
    dtype = getattr(torch, cfg.dtype)

    # Page-level latent + token-level noise gives non-trivial but reproducible KV.
    noise_k = torch.randn(shape, dtype=dtype, device=dev, generator=gen)
    noise_v = torch.randn(shape, dtype=dtype, device=dev, generator=gen)
    latent_shape = (cfg.num_pages, cfg.kv_groups, cfg.head_dim)
    if cfg.kv_mode == "per_request":
        latent_shape = (cfg.batch_size,) + latent_shape
    latent_k = torch.randn(latent_shape, dtype=dtype, device=dev, generator=gen)
    latent_v = torch.randn(latent_shape, dtype=dtype, device=dev, generator=gen)

    if cfg.kv_mode == "shared":
        k = noise_k * 0.55 + latent_k[:, None, :, :] * 0.45
        v = noise_v * 0.55 + latent_v[:, None, :, :] * 0.45
    else:
        k = noise_k * 0.55 + latent_k[:, :, None, :, :] * 0.45
        v = noise_v * 0.55 + latent_v[:, :, None, :, :] * 0.45
    return k.contiguous(), v.contiguous()


def fixed_permuted_block_table(
    cfg: WorkloadConfig,
    device: str | torch.device,
    seed: int = 4242,
) -> torch.Tensor:
    """[B, num_pages] fixed random physical-page permutation, shared by requests.

    The Ascend benchmark uses one permutation repeated for every request; this
    keeps physical-page fan-out identical while making the mapping non-identity.
    """
    dev = torch.device(device)
    perm = np.random.default_rng(seed).permutation(cfg.num_pages).astype(np.int64)
    row = torch.as_tensor(perm, dtype=torch.int64, device=dev)
    return row[None, :].repeat(cfg.batch_size, 1).contiguous()


def selected_token_indices(
    topk_ids_np: np.ndarray,
    block_table: torch.Tensor,
    cfg: WorkloadConfig,
) -> torch.Tensor:
    """Map region ids -> flattened token indices in cache_k/cache_v.

    Return [B, G, n_ids*region_size].  The cache flattening convention is:
      shared:      [num_pages * page_size, G, D]
      per_request: [B * num_pages * page_size, G, D]
    """
    dev = block_table.device
    ids = torch.as_tensor(topk_ids_np, dtype=torch.int64, device=dev)  # [G,B,K]
    bsz, groups = cfg.batch_size, cfg.kv_groups
    ids = ids.permute(1, 0, 2).contiguous()  # [B,G,K]
    n_ids = ids.shape[2]

    rpp = cfg.regions_per_page
    logical_page = torch.div(ids, rpp, rounding_mode="floor")
    intra = (ids % rpp) * cfg.region_size
    intra = intra.unsqueeze(-1) + torch.arange(cfg.region_size, dtype=torch.int64, device=dev)
    intra = intra.reshape(bsz, groups, n_ids * cfg.region_size)

    flat_lp = logical_page.reshape(bsz, -1)
    phys_page = torch.gather(block_table, 1, flat_lp).reshape(bsz, groups, n_ids)
    token_idx = phys_page.unsqueeze(-1) * cfg.page_size + intra.reshape(bsz, groups, n_ids, cfg.region_size)
    token_idx = token_idx.reshape(bsz, groups, n_ids * cfg.region_size)
    if cfg.kv_mode == "per_request":
        offset = torch.arange(bsz, dtype=torch.int64, device=dev)[:, None, None] * cfg.num_pages * cfg.page_size
        token_idx = token_idx + offset
    return token_idx.contiguous()


def gather_kv(
    cache: torch.Tensor,
    token_idx: torch.Tensor,
    cfg: WorkloadConfig,
) -> torch.Tensor:
    """Gather selected K or V -> [B, G, selected_tokens, D]."""
    flat = cache.view(-1, cfg.kv_groups, cfg.head_dim)
    parts = [flat[token_idx[:, g, :], g, :] for g in range(cfg.kv_groups)]
    return torch.stack(parts, dim=1).contiguous()


def selection_metrics(
    topk_ids_np: np.ndarray,
    block_table: torch.Tensor,
    cfg: WorkloadConfig,
) -> dict[str, float]:
    """Compute per-request page fan-out and batch-level union pages."""
    ids = np.asarray(topk_ids_np, dtype=np.int64)  # [G,B,K]
    bt = block_table.detach().cpu().numpy()
    rpp = cfg.regions_per_page
    counts: list[int] = []
    union: set[int] = set()
    for g in range(cfg.kv_groups):
        for b in range(cfg.batch_size):
            logical_pages = np.unique(ids[g, b] // rpp)
            phys = np.unique(bt[b, logical_pages])
            counts.append(int(phys.size))
            union.update(int(x) for x in phys)
    union_pages = len(union)
    return {
        "per_request_pages_mean": float(np.mean(counts)),
        "per_request_pages_median": float(np.median(counts)),
        "union_pages": float(union_pages),
        "working_set_MiB": float(union_pages * cfg.page_bytes / (1024.0 * 1024.0)),
    }