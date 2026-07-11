"""Varlen FlashAttention selected-block forward path for MSA main attention."""

from __future__ import annotations

import torch

from flash_msa.reverse_index_cuda import SparseAttentionMetadata
from flash_msa.sparse_flash_varlen import sparse_flash_varlen_forward

BLOCK_SIZE = 128


def _validate_head_tiling(
    n_heads: int,
    n_kv_heads: int,
    n_proxy_heads: int,
) -> None:
    """Validate that selected attention supports this head configuration."""

    if n_heads % n_proxy_heads != 0:
        raise NotImplementedError("n_heads must be divisible by n_proxy_heads")
    if n_heads % n_kv_heads != 0:
        raise NotImplementedError("n_heads must be divisible by n_kv_heads")
    if n_proxy_heads < n_kv_heads or n_proxy_heads % n_kv_heads != 0:
        raise NotImplementedError(
            "MSA forward requires n_proxy_heads >= n_kv_heads and divisibility"
        )


def run_main_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    metadata: SparseAttentionMetadata,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run varlen selected attention and return its forward outputs."""

    if q.device.type != "cuda":
        raise ValueError("MSA forward requires CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"MSA forward supports fp16/bf16, got {q.dtype}")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [B, H, S, D]")

    _, n_heads, seq_len, head_dim = map(int, q.shape)
    n_kv_heads = int(k.shape[1])
    n_proxy_heads = metadata.n_proxy_heads
    if seq_len % BLOCK_SIZE:
        raise ValueError(f"sequence length must be divisible by {BLOCK_SIZE}")
    if head_dim != 128:
        raise NotImplementedError(f"MSA forward requires D=128, got {head_dim}")
    _validate_head_tiling(n_heads, n_kv_heads, n_proxy_heads)

    q_c = q.detach().contiguous()
    k_c = k.detach().contiguous()
    v_c = v.detach().contiguous()

    o_main, lse_main = sparse_flash_varlen_forward(
        q_c,
        k_c,
        v_c,
        metadata=metadata,
        scale=float(scale),
        return_output=True,
    )
    assert o_main is not None
    kl_loss = torch.zeros((), dtype=torch.float32, device=q.device)
    return o_main, lse_main, kl_loss
