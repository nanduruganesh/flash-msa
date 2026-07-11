"""Compatibility hooks for FlashAttention 3 and FlashAttention 4."""

from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _fa4_varlen_func():
    try:
        from flash_attn.cute.interface import flash_attn_varlen_func
    except ModuleNotFoundError as exc:
        if exc.name == "flash_attn" or (
            exc.name is not None and exc.name.startswith("flash_attn.cute")
        ):
            return None
        raise
    return flash_attn_varlen_func


@lru_cache(maxsize=1)
def _fa3_varlen_func():
    try:
        from flash_attn_interface import flash_attn_varlen_func
    except ModuleNotFoundError as exc:
        if exc.name == "flash_attn_interface":
            return None
        raise
    return flash_attn_varlen_func


def flash_attn_supports_narrow_value_dim() -> bool:
    """Whether the active backend accepts the 8-wide dummy V used for LSE-only calls."""

    return _fa4_varlen_func() is not None


def flash_attn_varlen_paged_forward(
    *,
    q: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    page_table: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Use FA4 paged varlen forward, or return ``None`` on FA3 installs."""

    flash_attn_varlen_func = _fa4_varlen_func()
    if flash_attn_varlen_func is None:
        return None

    out, lse = flash_attn_varlen_func(
        q,
        k_pages,
        v_pages,
        cu_seqlens_q=cu_seqlens_q,
        page_table=page_table,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        return_lse=True,
    )
    return out, lse


def flash_attn_varlen_forward(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(out, lse)`` for FlashAttention 3 or 4."""

    flash_attn_varlen_func = _fa3_varlen_func()
    if flash_attn_varlen_func is not None:
        out, lse = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            return_attn_probs=True,
        )
        return out, lse

    flash_attn_varlen_func = _fa4_varlen_func()
    if flash_attn_varlen_func is None:
        raise ModuleNotFoundError(
            "Flash-MSA requires FlashAttention 3 or 4 varlen support"
        )

    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        return_lse=True,
    )
    return out, lse
