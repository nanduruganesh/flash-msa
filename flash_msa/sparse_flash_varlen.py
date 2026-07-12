"""Memory-bounded selected-block attention using varlen FlashAttention.

The local causal block is handled densely.  Strictly earlier selected blocks
use compact reverse-index metadata, bounded varlen calls, and an online CUDA
merge into O(S)-sized FP32 output/LSE accumulators.
"""

from __future__ import annotations

import os

import torch

from flash_msa._flash_attn_compat import (
    flash_attn_supports_narrow_value_dim,
    flash_attn_varlen_forward,
    flash_attn_varlen_paged_forward,
)
from flash_msa.reverse_index_cuda import (
    SparseAttentionMetadata,
    merge_attention_chunk_cuda,
    merge_lse_chunk_cuda,
)


BLOCK_SIZE = 128
TASKS_PER_FLASH_CHUNK = int(os.environ.get("MSA_FLASH_TASKS_PER_CHUNK", "256"))

# Had to write this section to set the narrow V head for Proxy LSE varlen flash call
# to the minimum dim supported on each backend. FA4 on B200 fails with V_dim=8
LSE_VALUE_DIM_H100 = 8
LSE_VALUE_DIM_B200 = 16
def _lse_value_dim(device: torch.device) -> int:
    instruction_set = torch.cuda.get_device_capability(device)
    if instruction_set[0] == 9:
        return LSE_VALUE_DIM_H100
    if instruction_set[0] == 10:
        return LSE_VALUE_DIM_B200
    raise NotImplementedError(
        f"Proxy LSE dummy V is not configured for SM{instruction_set[0]}{instruction_set[1]}"
    )


def _local_block_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run causal attention independently inside every local 128-token block."""

    batch, n_heads, seq_len, head_dim = map(int, q.shape)
    n_kv_heads = int(k.shape[1])
    value_dim = int(v.shape[-1])
    total_tokens = batch * seq_len
    num_sequences = total_tokens // BLOCK_SIZE
    cu_seqlens = torch.arange(
        num_sequences + 1,
        device=q.device,
        dtype=torch.int32,
    ) * BLOCK_SIZE

    q_tokens = q.transpose(1, 2).contiguous().view(total_tokens, n_heads, head_dim)
    k_tokens = k.transpose(1, 2).contiguous().view(total_tokens, n_kv_heads, head_dim)
    v_tokens = v.transpose(1, 2).contiguous().view(total_tokens, n_kv_heads, value_dim)
    return flash_attn_varlen_forward(
        q=q_tokens,
        k=k_tokens,
        v=v_tokens,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=BLOCK_SIZE,
        max_seqlen_k=BLOCK_SIZE,
        softmax_scale=float(scale),
        causal=True,
    )


def _chunk_bounds(
    metadata: SparseAttentionMetadata,
    task_start: int,
    task_end: int,
) -> tuple[int, int]:
    meta = metadata.remote_task_meta_cpu
    edge_start = int(meta[task_start, 4])
    last = task_end - 1
    edge_end = int(meta[last, 4]) + int(meta[last, 3])
    return edge_start, edge_end


def sparse_flash_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    metadata: SparseAttentionMetadata,
    scale: float,
    return_output: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Return selected-block output (optionally) and LSE using bounded chunks."""

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must be rank-4 tensors")
    if q.device.type != "cuda":
        raise ValueError("sparse varlen FlashAttention requires CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"sparse varlen FlashAttention supports fp16/bf16, got {q.dtype}")

    batch, n_heads, seq_len, head_dim = map(int, q.shape)
    n_kv_heads = int(k.shape[1])
    n_proxy_heads = metadata.n_proxy_heads
    num_blocks = seq_len // BLOCK_SIZE
    if seq_len % BLOCK_SIZE:
        raise ValueError(f"sequence length must be divisible by {BLOCK_SIZE}")
    if n_heads % n_proxy_heads:
        raise ValueError("query heads must be divisible by proxy heads")
    if n_proxy_heads < n_kv_heads or n_proxy_heads % n_kv_heads:
        raise ValueError("proxy heads must be >= and divisible by KV heads")
    if (metadata.batch, metadata.seq_len) != (batch, seq_len):
        raise ValueError("metadata has an incompatible batch or sequence length")
    if metadata.task_meta.device != q.device:
        raise ValueError("metadata must be on the same device as q")
    if TASKS_PER_FLASH_CHUNK < 1:
        raise ValueError("MSA_FLASH_TASKS_PER_CHUNK must be positive")

    main_per_proxy = n_heads // n_proxy_heads

    # FA4 accepts a narrow dummy V for LSE-only proxy attention. SM100 requires
    # 16 elements because its packed-GQA epilogue rejects an 8-element V.
    if return_output or not flash_attn_supports_narrow_value_dim():
        attention_v = v
    else:
        attention_v = torch.zeros(
            (*v.shape[:-1], _lse_value_dim(v.device)),
            device=v.device,
            dtype=v.dtype,
        )
    local_out, local_lse_hs = _local_block_attention(
        q, k, attention_v, scale=float(scale)
    )
    lse_accum = (
        local_lse_hs.transpose(0, 1)
        .reshape(batch, seq_len, n_proxy_heads, main_per_proxy)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(batch * n_proxy_heads * seq_len, main_per_proxy)
    )
    if return_output:
        output_accum = (
            local_out.reshape(batch, seq_len, n_proxy_heads, main_per_proxy, head_dim)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(batch * n_proxy_heads * seq_len, main_per_proxy, head_dim)
            .float()
        )
    else:
        output_accum = None

    q_grouped = q.reshape(
        batch, n_proxy_heads, main_per_proxy, seq_len, head_dim
    ).permute(0, 1, 3, 2, 4)
    proxy_per_kv = n_proxy_heads // n_kv_heads
    proxy_to_kv = torch.arange(n_proxy_heads, device=q.device) // proxy_per_kv
    value_dim = int(attention_v.shape[-1])
    k_pages = k.view(-1, BLOCK_SIZE, 1, head_dim)
    v_pages = attention_v.view(-1, BLOCK_SIZE, 1, value_dim)

    for task_start in range(0, metadata.num_remote_tasks, TASKS_PER_FLASH_CHUNK):
        task_end = min(task_start + TASKS_PER_FLASH_CHUNK, metadata.num_remote_tasks)
        edge_start, edge_end = _chunk_bounds(metadata, task_start, task_end)
        edge_count = edge_end - edge_start
        if edge_count == 0:
            continue

        task_meta = metadata.remote_task_meta[task_start:task_end]
        task_offsets = metadata.remote_task_offsets[task_start : task_end + 1]
        cu_seqlens_q = (task_offsets - int(edge_start)).contiguous()
        destination = metadata.destinations[edge_start:edge_end]
        packed_qids = metadata.packed_qids[edge_start:edge_end]
        group = destination.div(seq_len, rounding_mode="floor")
        packed_batch = group.div(n_proxy_heads, rounding_mode="floor").long()
        packed_proxy = (group - packed_batch * n_proxy_heads).long()
        packed_q = q_grouped[
            packed_batch,
            packed_proxy,
            packed_qids.long(),
        ].contiguous()

        task_batch = task_meta[:, 0].long()
        task_proxy = task_meta[:, 1].long()
        task_block = task_meta[:, 2].long()
        physical_page = (
            (task_batch * n_kv_heads + proxy_to_kv[task_proxy]) * num_blocks
            + task_block
        ).to(torch.int32).view(-1, 1)
        paged_result = flash_attn_varlen_paged_forward(
            q=packed_q,
            k_pages=k_pages,
            v_pages=v_pages,
            cu_seqlens_q=cu_seqlens_q,
            page_table=physical_page,
            max_seqlen_q=metadata.remote_query_chunk,
            max_seqlen_k=BLOCK_SIZE,
            softmax_scale=float(scale),
            causal=False,
        )
        if paged_result is None:
            packed_k = k_pages[physical_page[:, 0].long()].reshape(-1, 1, head_dim)
            packed_v = v_pages[physical_page[:, 0].long()].reshape(-1, 1, value_dim)
            cu_seqlens_k = torch.arange(
                task_end - task_start + 1,
                device=q.device,
                dtype=torch.int32,
            ) * BLOCK_SIZE
            paged_result = flash_attn_varlen_forward(
                q=packed_q,
                k=packed_k,
                v=packed_v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=metadata.remote_query_chunk,
                max_seqlen_k=BLOCK_SIZE,
                softmax_scale=float(scale),
                causal=False,
            )
        remote_out, remote_lse_hs = paged_result
        remote_lse = remote_lse_hs.transpose(0, 1).contiguous()
        if output_accum is None:
            merge_lse_chunk_cuda(
                lse_accum,
                remote_lse,
                metadata,
                edge_start=edge_start,
            )
        else:
            merge_attention_chunk_cuda(
                output_accum,
                lse_accum,
                remote_out,
                remote_lse,
                metadata,
                edge_start=edge_start,
            )

    lse = (
        lse_accum.view(batch, n_proxy_heads, seq_len, main_per_proxy)
        .permute(0, 1, 3, 2)
        .contiguous()
        .view(batch, n_heads, seq_len)
    )
    if output_accum is None:
        return None, lse
    output = (
        output_accum.to(q.dtype)
        .view(batch, n_proxy_heads, seq_len, main_per_proxy, head_dim)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(batch, n_heads, seq_len, head_dim)
    )
    return output, lse


__all__ = ["sparse_flash_varlen_forward"]
