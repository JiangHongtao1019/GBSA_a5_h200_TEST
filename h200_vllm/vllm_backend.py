#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reference vLLM-compatible GBSA forward for H200.

The timed path is deliberately simple and semantically exact:
  selected top-k regions -> gather K/V tokens -> SDPA (FlashAttention on H200).

This is a vLLM custom-backend reference path, not a claim that vLLM has a
native GBSA kernel.  A fused Triton kernel can replace gather+SDPA later while
keeping the same benchmark contract.
"""
from __future__ import annotations

import math
import traceback
from dataclasses import dataclass

import torch
import torch.nn.functional as F

try:
    from workload import WorkloadConfig, gather_kv
except ImportError:  # package import: python -m h200_vllm.run_bench
    from .workload import WorkloadConfig, gather_kv


def gbsa_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    heads: int,
    kv_groups: int,
    head_dim: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """q [B,H,D] or [B,H,1,D] + k/v [B,G,S,D] -> [B,H*D]."""
    if q.dim() == 2:
        bsz = q.shape[0]
        q = q.view(bsz, heads, 1, head_dim)
    elif q.dim() == 3:
        q = q.unsqueeze(2)
    if q.shape[1] != heads or q.shape[-1] != head_dim:
        raise ValueError(f"unexpected q shape {tuple(q.shape)}")
    if k.shape[1] != kv_groups or v.shape[1] != kv_groups:
        raise ValueError(f"unexpected k/v shape {tuple(k.shape)}")

    # PyTorch >=2.5 dispatches to the CUDA FlashAttention backend for BF16.
    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=1.0 / math.sqrt(head_dim),
        enable_gqa=(heads != kv_groups),
    )
    out = out.squeeze(2).reshape(q.shape[0], heads * head_dim)
    if output is not None:
        output.copy_(out)
        return output
    return out


@dataclass
class GBSAMetadata:
    """Minimal metadata accepted by the vLLM-style backend adapter."""
    cache_k: torch.Tensor
    cache_v: torch.Tensor
    block_table: torch.Tensor
    token_idx: torch.Tensor  # [B,G,n_ids*region_size], int64, flat cache indices
    cfg: WorkloadConfig
    # Optional pre-gathered KV for the sdpa_only timing mode.
    precomputed_kv: tuple[torch.Tensor, torch.Tensor] | None = None


# vLLM is optional at import time so the reference path can be tested alone.
# Keep the actual import error so users see the real cause instead of only
# "vLLM is required but could not be imported".
_VLLM_IMPORT_ERROR: str | None = None
_VLLM_ATTENTION_API_ERROR: str | None = None
try:
    import vllm as _vllm
    VLLM_AVAILABLE = True
    VLLM_VERSION = getattr(_vllm, "__version__", "unknown")
except Exception:
    _vllm = None
    VLLM_AVAILABLE = False
    VLLM_VERSION = "not-importable"
    _VLLM_IMPORT_ERROR = traceback.format_exc()

try:
    from vllm.attention.backends.abstract import AttentionBackend
    try:
        from vllm.attention.backends.abstract import AttentionImpl as _VllmAttentionImpl
    except Exception as exc:
        _VllmAttentionImpl = None
        _VLLM_ATTENTION_API_ERROR = f"AttentionImpl unavailable: {exc}"
except Exception as exc:  # pragma: no cover - only used on non-vLLM hosts
    class AttentionBackend:  # type: ignore
        pass
    _VllmAttentionImpl = None
    if _VLLM_ATTENTION_API_ERROR is None:
        _VLLM_ATTENTION_API_ERROR = f"AttentionBackend unavailable: {exc}"

VLLM_ATTENTION_API_AVAILABLE = _VllmAttentionImpl is not None


class GBSAH200Impl(torch.nn.Module):
    """AttentionImpl-compatible forward used by the H200 microbenchmark.

    Signature intentionally follows vLLM's ``AttentionImpl.forward``:
        forward(layer, query, key, value, kv_cache, attn_metadata, output=None)
    The benchmark passes ``layer=None`` and keeps K/V in ``attn_metadata``.
    """

    def __init__(self, num_heads: int = 64, head_size: int = 192,
                 num_kv_heads: int = 4, **kwargs):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_size = int(head_size)
        self.num_kv_heads = int(num_kv_heads)
        self.scale = 1.0 / math.sqrt(self.head_size)

    def forward(self, layer, query, key, value, kv_cache, attn_metadata: GBSAMetadata, output=None):
        meta = attn_metadata
        if meta.precomputed_kv is not None:
            k_sel, v_sel = meta.precomputed_kv
        else:
            k_sel = gather_kv(meta.cache_k, meta.token_idx, meta.cfg)
            v_sel = gather_kv(meta.cache_v, meta.token_idx, meta.cfg)
        return gbsa_forward(
            query, k_sel, v_sel,
            heads=self.num_heads,
            kv_groups=self.num_kv_heads,
            head_dim=self.head_size,
            output=output,
        )


if _VllmAttentionImpl is not None:
    class VLLMGBSAH200Impl(_VllmAttentionImpl):  # type: ignore[misc]
        """Thin adapter for the full vLLM engine; benchmark uses GBSAH200Impl."""

        def __init__(self, *args, **kwargs):
            torch.nn.Module.__init__(self)
            num_heads = kwargs.pop("num_heads", 64)
            head_size = kwargs.pop("head_size", 192)
            num_kv_heads = kwargs.pop("num_kv_heads", 4)
            self.num_heads = int(num_heads)
            self.head_size = int(head_size)
            self.num_kv_heads = int(num_kv_heads)
            self.scale = 1.0 / math.sqrt(self.head_size)
            self._impl = GBSAH200Impl(num_heads=self.num_heads, head_size=self.head_size,
                                      num_kv_heads=self.num_kv_heads)

        def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None):
            return self._impl.forward(layer, query, key, value, kv_cache,
                                      attn_metadata, output=output)
else:
    VLLMGBSAH200Impl = GBSAH200Impl


class GBSAH200Backend(AttentionBackend):
    """vLLM backend descriptor; registration is best-effort/version-dependent."""

    @staticmethod
    def get_name() -> str:
        return "GBSA_H200"

    @staticmethod
    def get_impl_cls():
        return VLLMGBSAH200Impl

    @staticmethod
    def get_metadata_cls():
        return GBSAMetadata

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int,
                           num_kv_heads: int, head_size: int):
        # Combined K/V cache, compatible with the reference layout.
        return (2, num_blocks, block_size, num_kv_heads, head_size)


def register_backend() -> bool:
    """Best-effort registration for a full vLLM engine setup.

    The microbenchmark does not require this call; it drives GBSAH200Impl
    directly through the same forward signature.  A full engine additionally
    needs a metadata builder wired to the model's top-k ids.
    """
    try:
        from vllm.attention.backends.registry import register_backend as _reg
        _reg(GBSAH200Backend)
        return True
    except Exception as exc:  # pragma: no cover
        print(f"[gbsa-h200] backend registration skipped: {exc}")
        return False