"""Warmup MSA tiled CuTeDSL backward.

Warmup attention uses the full causal block mask.  This module builds that
schedule and reuses the fused selected-edge CuTeDSL backward over every causal
block edge.  Proxy forward is not run during forward; proxy LSE is computed in
backward before launching the tiled gradient kernel.
"""

import torch

BLOCK_SIZE = 128
_SCHEDULE_CACHE: dict[tuple[int, int, int, int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _device_index(device: torch.device) -> int:
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def _dense_causal_schedule(
    *,
    batch: int,
    n_proxy_heads: int,
    seq_len: int,
    query_chunk: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return reverse-index tasks for all causal block edges."""

    if seq_len % BLOCK_SIZE != 0:
        raise ValueError(f"sequence length must be divisible by {BLOCK_SIZE}")
    key = (
        _device_index(device),
        int(batch),
        int(n_proxy_heads),
        int(seq_len),
        int(query_chunk),
        int(BLOCK_SIZE),
    )
    if key in _SCHEDULE_CACHE:
        return _SCHEDULE_CACHE[key]

    num_blocks = int(seq_len) // BLOCK_SIZE
    meta_rows: list[list[int]] = []
    qid_rows: list[torch.Tensor] = []

    for bidx in range(int(batch)):
        for proxy_head in range(int(n_proxy_heads)):
            for key_block in range(num_blocks):
                query_start = key_block * BLOCK_SIZE
                qids = torch.arange(query_start, int(seq_len), dtype=torch.int32)
                for start in range(0, int(qids.numel()), int(query_chunk)):
                    chunk = qids[start : start + int(query_chunk)]
                    valid = int(chunk.numel())
                    qid_row = torch.full((int(query_chunk),), -1, dtype=torch.int32)
                    qid_row[:valid] = chunk
                    meta_rows.append([bidx, proxy_head, key_block, valid])
                    qid_rows.append(qid_row)

    task_meta = torch.tensor(meta_rows, dtype=torch.int32, device=device)
    task_qids = torch.stack(qid_rows, dim=0).to(device=device, non_blocking=True)
    _SCHEDULE_CACHE[key] = (task_meta, task_qids)
    return task_meta, task_qids


def _lse_from_flash(
    lse: torch.Tensor,
    *,
    batch: int,
    n_heads: int,
    seq_len: int,
) -> torch.Tensor:
    if lse.shape == (n_heads, batch * seq_len):
        return lse.transpose(0, 1).contiguous().view(batch, seq_len, n_heads).permute(0, 2, 1)
    if lse.shape == (batch, n_heads, seq_len):
        return lse.contiguous()
    if lse.shape == (n_heads, batch, seq_len):
        return lse.permute(1, 0, 2).contiguous()
    raise RuntimeError(f"unexpected FlashAttention LSE shape: {tuple(lse.shape)}")


def _run_proxy_lse_flash(
    q_proxy: torch.Tensor,
    k_proxy: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Compute dense causal proxy LSE for the KL-gradient branch."""

    from flash_msa._flash_attn_compat import flash_attn_varlen_forward

    batch, n_proxy_heads, seq_len, head_dim = q_proxy.shape
    n_proxy_kv_heads = k_proxy.shape[1]
    q_pack = q_proxy.transpose(1, 2).contiguous().view(
        batch * seq_len, n_proxy_heads, head_dim
    )
    k_pack = k_proxy.transpose(1, 2).contiguous().view(
        batch * seq_len, n_proxy_kv_heads, head_dim
    )
    cu_seqlens = torch.arange(batch + 1, device=q_proxy.device, dtype=torch.int32) * int(seq_len)

    _out, lse = flash_attn_varlen_forward(
        q=q_pack,
        k=k_pack,
        v=k_pack,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=int(seq_len),
        max_seqlen_k=int(seq_len),
        softmax_scale=float(scale),
        causal=True,
    )
    return _lse_from_flash(lse, batch=batch, n_heads=n_proxy_heads, seq_len=seq_len)


def run_warmup_backward(
    q_proxy: torch.Tensor,
    k_proxy: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lse_main: torch.Tensor,
    o_main: torch.Tensor,
    grad_out: torch.Tensor | None,
    grad_kl: torch.Tensor | None,
    *,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run tiled CuTeDSL backward over the full causal block schedule."""

    batch, n_heads, seq_len, head_dim = q.shape
    n_proxy_heads = q_proxy.shape[1]
    n_kv_heads = k.shape[1]

    if q.device.type != "cuda":
        raise ValueError("warmup CuTeDSL backward requires CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"warmup CuTeDSL backward supports fp16/bf16, got {q.dtype}")
    if head_dim != 128:
        raise NotImplementedError("warmup CuTeDSL backward supports only D=128")

    if grad_out is None:
        grad_o_main = torch.zeros_like(o_main)
    else:
        grad_o_main = (
            grad_out.reshape(batch, seq_len, n_heads, head_dim).transpose(1, 2).contiguous()
        )

    # Proxy KL gradients are linear in the upstream scalar.  Keep the kernel
    # argument static, then apply the device-resident scalar to the proxy
    # gradients below instead of synchronizing through ``grad_kl.item()``.
    proxy_grad_scale = (
        0.0 if grad_kl is None else 1.0 / float(batch * n_proxy_heads * seq_len)
    )

    from flash_msa.msa_backward_cutedsl import _derive_head_tiling, run_fused_backward

    _main_per_proxy, query_chunk, _rows_per_task, _proxy_query_rows = _derive_head_tiling(
        n_heads, n_kv_heads, n_proxy_heads
    )
    task_meta, task_qids = _dense_causal_schedule(
        batch=batch,
        n_proxy_heads=n_proxy_heads,
        seq_len=seq_len,
        query_chunk=query_chunk,
        device=q.device,
    )
    lse_proxy = _run_proxy_lse_flash(q_proxy, k_proxy, scale=float(scale))
    delta_main = (o_main.float() * grad_o_main.float()).sum(dim=-1)

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
        task_meta,
        task_qids,
        scale=float(scale),
        grad_kl_scale=proxy_grad_scale,
    )
    if grad_kl is not None:
        proxy_multiplier = grad_kl.detach().to(device=q.device, dtype=dq_proxy.dtype)
        dq_proxy = dq_proxy * proxy_multiplier
        dk_proxy = dk_proxy * proxy_multiplier.to(dtype=dk_proxy.dtype)
    return dq_proxy, dk_proxy, dq, dk, dv
