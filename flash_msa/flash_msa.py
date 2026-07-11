"""Block-sparse top-k GQA attention training entrypoint.

This module owns the Python boundary for the native fused MSA kernels.
"""

import torch

from flash_msa.msa_select_cutedsl import compute_proxy_lse, select_blocks
from flash_msa.msa_backward_cutedsl import run_fused_backward
from flash_msa.msa_forward_cutedsl import run_main_forward
from flash_msa.reverse_index_cuda import (
    SparseAttentionMetadata,
    build_sparse_attention_metadata_cuda,
)

BLOCK_SIZE = 128
NATIVE_MMA_ROWS_PER_TASK = 128

def _validate_inputs(
    q_proxy: torch.Tensor,
    k_proxy: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    top_k: int,
) -> tuple[int, int]:
    if q_proxy.ndim != 4 or k_proxy.ndim != 4 or q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("all attention tensors must have shape (B, H, S, D)")

    b, n_proxy_heads, s, head_dim = q_proxy.shape
    bk, n_proxy_kv_heads, sk, dk = k_proxy.shape
    bq, n_heads, sq, dq = q.shape
    bm, n_kv_heads, sm, dm = k.shape
    bv, n_v_heads, sv, dv = v.shape

    if not (
        b == bk == bq == bm == bv
        and s == sk == sq == sm == sv
        and head_dim == dk == dq == dm == dv
        and n_kv_heads == n_v_heads
    ):
        raise ValueError("q_proxy, k_proxy, q, k, and v must agree on batch, sequence, head dimension, and KV head count")

    if s % BLOCK_SIZE != 0:
        raise ValueError(f"sequence length must be divisible by {BLOCK_SIZE}, got {s}")
    if top_k % BLOCK_SIZE != 0:
        raise ValueError(f"top_k must be divisible by {BLOCK_SIZE}, got {top_k}")
    if n_proxy_heads % n_proxy_kv_heads != 0:
        raise ValueError("n_proxy_heads must be divisible by n_proxy_kv_heads")
    if n_heads % n_kv_heads != 0:
        raise ValueError("n_heads must be divisible by n_kv_heads")
    if n_heads % n_proxy_heads != 0:
        raise ValueError("n_heads must be divisible by n_proxy_heads")
    if q_proxy.shape[-1] != 128:
        raise NotImplementedError("Headdim must be 128")

    num_blocks = s // BLOCK_SIZE
    top_k_blocks = int(top_k) // BLOCK_SIZE
    if not 1 <= top_k_blocks <= num_blocks:
        raise ValueError(f"top_k selects {top_k_blocks} blocks, but sequence has {num_blocks} blocks")
    if top_k_blocks > 32:
        raise NotImplementedError("No more than 32 blocks / topk=4096 supported")

    return num_blocks, top_k_blocks

def _run_fused_selected_edge_backward(
    q_proxy: torch.Tensor,
    k_proxy: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lse_main: torch.Tensor,
    o_main: torch.Tensor,
    grad_o_main: torch.Tensor,
    grad_kl: torch.Tensor | None,
    metadata: SparseAttentionMetadata,
    *,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the native reverse-index fused selected-edge backward."""

    bsz, n_heads, seq_len, _ = q.shape
    n_proxy_heads = q_proxy.shape[1]

    main_per_proxy = int(n_heads) // int(n_proxy_heads)
    if NATIVE_MMA_ROWS_PER_TASK % main_per_proxy != 0:
        raise ValueError(
            "Main q heads / proxy q heads ratio must divide "
            "NATIVE_MMA_ROWS_PER_TASK evenly"
        )

    lse_proxy = compute_proxy_lse(
        q_proxy,
        k_proxy,
        scale=float(scale),
        metadata=metadata,
    )

    delta_main = (o_main.float() * grad_o_main.float()).sum(dim=-1)
    # Proxy KL gradients are linear in the upstream scalar.  Run the native
    # kernel at the static normalization scale, then apply the CUDA scalar to
    # only the proxy gradients.  This avoids the synchronizing ``.item()`` that
    # would otherwise be needed for a by-value CuTeDSL kernel argument.
    proxy_grad_scale = (
        0.0 if grad_kl is None else 1.0 / float(bsz * n_proxy_heads * seq_len)
    )
    dq_proxy, dk_proxy, dq, dk, dv = run_fused_backward(
        q_proxy,
        k_proxy,
        q,
        k,
        v,
        grad_o_main,
        lse_main,
        lse_proxy,
        delta_main,
        metadata.task_meta,
        metadata.task_qids,
        scale=float(scale),
        grad_kl_scale=proxy_grad_scale,
    )
    if grad_kl is not None:
        proxy_multiplier = grad_kl.detach().to(device=q.device, dtype=dq_proxy.dtype)
        dq_proxy = dq_proxy * proxy_multiplier
        dk_proxy = dk_proxy * proxy_multiplier.to(dtype=dk_proxy.dtype)
    return dq_proxy, dk_proxy, dq, dk, dv


class _SparseAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_proxy: torch.Tensor,
        k_proxy: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        top_k: int,
        scale: float,
    ):
        b, n_proxy_heads, s, head_dim = q_proxy.shape
        n_heads = q.shape[1]
        n_kv_heads = k.shape[1]
        num_blocks, top_k_blocks = _validate_inputs(q_proxy, k_proxy, q, k, v, int(top_k))
        block_indices = select_blocks(
            q_proxy,
            k_proxy,
            scale=float(scale),
            num_blocks=num_blocks,
            top_k_blocks=top_k_blocks,
        )

        main_per_proxy = int(n_heads) // int(n_proxy_heads)
        if NATIVE_MMA_ROWS_PER_TASK % main_per_proxy != 0:
            raise ValueError(
                "Main q heads / proxy q heads ratio must divide "
                "NATIVE_MMA_ROWS_PER_TASK evenly"
            )
        metadata = build_sparse_attention_metadata_cuda(
            block_indices,
            backward_query_chunk=NATIVE_MMA_ROWS_PER_TASK // main_per_proxy,
        )

        o_main, lse_main, kl_loss = run_main_forward(
            q,
            k,
            v,
            scale=float(scale),
            metadata=metadata,
        )

        out = o_main.transpose(1, 2).reshape(b, s, -1)

        save_tensors = (
            q_proxy,
            k_proxy,
            q,
            k,
            v,
            lse_main,
            o_main,
            metadata.task_meta,
            metadata.task_qids,
            metadata.remote_task_meta,
            metadata.remote_task_offsets,
            metadata.packed_qids,
            metadata.destinations,
            metadata.edge_positions,
        )
        ctx.save_for_backward(*save_tensors)
        ctx.scale = float(scale)
        ctx.num_remote_tasks = metadata.num_remote_tasks
        ctx.remote_task_meta_cpu = metadata.remote_task_meta_cpu
        ctx.metadata_shape = (
            metadata.batch,
            metadata.n_proxy_heads,
            metadata.seq_len,
            metadata.top_k_blocks,
            metadata.remote_query_chunk,
        )
        return out, kl_loss

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor | None, grad_kl: torch.Tensor | None):
        (
            q_proxy,
            k_proxy,
            q,
            k,
            v,
            lse_main,
            o_main,
            task_meta,
            task_qids,
            remote_task_meta,
            remote_task_offsets,
            packed_qids,
            destinations,
            edge_positions,
        ) = ctx.saved_tensors
        batch, n_proxy_heads, seq_len, top_k_blocks, remote_query_chunk = (
            ctx.metadata_shape
        )
        metadata = SparseAttentionMetadata(
            task_meta=task_meta,
            task_qids=task_qids,
            remote_task_meta=remote_task_meta,
            remote_task_offsets=remote_task_offsets,
            packed_qids=packed_qids,
            destinations=destinations,
            edge_positions=edge_positions,
            num_remote_tasks=ctx.num_remote_tasks,
            remote_task_meta_cpu=ctx.remote_task_meta_cpu,
            batch=batch,
            n_proxy_heads=n_proxy_heads,
            seq_len=seq_len,
            top_k_blocks=top_k_blocks,
            remote_query_chunk=remote_query_chunk,
        )

        if grad_out is None:
            grad_o_main = torch.zeros_like(o_main)
        else:
            b, s, _ = grad_out.shape
            grad_o_main = (
                grad_out.reshape(b, s, q.shape[1], q.shape[3]).transpose(1, 2).contiguous()
            )

        dq_proxy, dk_proxy, dq, dk, dv = _run_fused_selected_edge_backward(
            q_proxy,
            k_proxy,
            q,
            k,
            v,
            lse_main,
            o_main,
            grad_o_main,
            grad_kl,
            metadata,
            scale=ctx.scale,
        )
        return dq_proxy, dk_proxy, dq, dk, dv, None, None


def sparse_attention(
    q_proxy: torch.Tensor,
    k_proxy: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    top_k: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (attn_out, kl_loss placeholder)
    Saves reverse-index metadata and main attention state for backward.
    """
    return _SparseAttentionFunction.apply(q_proxy, k_proxy, q, k, v, int(top_k), float(scale))
