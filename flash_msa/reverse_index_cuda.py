"""CUDA reverse-index builder for selected-block MSA backward."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import torch
from torch.utils.cpp_extension import load


BLOCK_SIZE = 128
QUERY_CHUNK = 32
REMOTE_QUERY_CHUNK = 1024

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_THIS_DIR, "csrc", "reverse_index_cuda.cu")
_EXT = None


def _cuda_arch_flag() -> str:
    major, minor = torch.cuda.get_device_capability()
    return f"-arch=sm_{major}{minor}"


def _load_ext():
    global _EXT
    if _EXT is None:
        python_bin = os.path.dirname(sys.executable)
        ninja_path = os.path.join(python_bin, "ninja")
        if os.path.exists(ninja_path):
            path_entries = os.environ.get("PATH", "").split(os.pathsep)
            if python_bin not in path_entries:
                os.environ["PATH"] = os.pathsep.join(
                    [python_bin, os.environ.get("PATH", "")]
                )
        _EXT = load(
            name="msa_reverse_index_ext",
            sources=[_SRC],
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "-lineinfo",
                _cuda_arch_flag(),
            ],
            verbose=False,
        )
    return _EXT


@dataclass
class ReverseIndexWorkspace:
    cache: dict[tuple[int, int, int, int, int, int, int], dict[str, torch.Tensor]] = field(
        default_factory=dict
    )

    def get(
        self,
        batch: int,
        n_proxy_heads: int,
        seq_len: int,
        top_k_blocks: int,
        query_chunk: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        num_blocks = seq_len // BLOCK_SIZE
        padded_tasks = batch * n_proxy_heads * (
            ((seq_len * top_k_blocks + query_chunk - 1) // query_chunk)
            + num_blocks
        )
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        key = (
            int(device_index),
            int(batch),
            int(n_proxy_heads),
            int(seq_len),
            int(top_k_blocks),
            int(query_chunk),
            int(padded_tasks),
        )
        if key not in self.cache:
            self.cache[key] = {
                "counts": torch.empty(
                    (batch * n_proxy_heads * num_blocks,),
                    device=device,
                    dtype=torch.int32,
                ),
                "write_counts": torch.empty(
                    (batch * n_proxy_heads * num_blocks,),
                    device=device,
                    dtype=torch.int32,
                ),
                "bucket_offsets": torch.empty(
                    (batch * n_proxy_heads * num_blocks,),
                    device=device,
                    dtype=torch.int32,
                ),
                "task_meta": torch.empty(
                    (padded_tasks, 4),
                    device=device,
                    dtype=torch.int32,
                ),
                "task_qids": torch.empty(
                    (padded_tasks, query_chunk),
                    device=device,
                    dtype=torch.int32,
                ),
                "num_tasks": torch.empty(1, device=device, dtype=torch.int32),
            }
        return self.cache[key]


_DEFAULT_WORKSPACE = ReverseIndexWorkspace()


@dataclass
class SparseAttentionMetadata:
    """Persistent reverse-index and compact varlen metadata for one forward."""

    task_meta: torch.Tensor
    task_qids: torch.Tensor
    remote_task_meta: torch.Tensor
    remote_task_offsets: torch.Tensor
    packed_qids: torch.Tensor
    destinations: torch.Tensor
    edge_positions: torch.Tensor
    num_remote_tasks: int
    remote_task_meta_cpu: torch.Tensor
    batch: int
    n_proxy_heads: int
    seq_len: int
    top_k_blocks: int
    remote_query_chunk: int


def build_reverse_index_cuda(
    block_indices: torch.Tensor,
    *,
    query_chunk: int = QUERY_CHUNK,
    workspace: ReverseIndexWorkspace | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build padded ``(task_meta, task_qids)`` tensors fully on CUDA.

    ``block_indices`` has shape ``[B, Hp, S, top_k_blocks]``. The returned
    tensors have fixed padded task count:
    ``B * Hp * (ceil(S * top_k_blocks / query_chunk) + S / 128)``.
    """

    if block_indices.device.type != "cuda":
        raise ValueError("CUDA reverse-index builder requires a CUDA tensor")
    if block_indices.ndim != 4:
        raise ValueError("block_indices must have shape [B, Hp, S, top_k_blocks]")

    block_indices_c = block_indices.detach().to(torch.int32).contiguous()
    batch, n_proxy_heads, seq_len, top_k_blocks = map(int, block_indices_c.shape)
    if seq_len % BLOCK_SIZE != 0:
        raise ValueError(f"sequence length must be divisible by {BLOCK_SIZE}")
    if top_k_blocks < 1:
        raise ValueError("top_k_blocks must be positive")
    query_chunk = int(query_chunk)
    if query_chunk < 1:
        raise ValueError("query_chunk must be positive")

    ws = (workspace or _DEFAULT_WORKSPACE).get(
        batch,
        n_proxy_heads,
        seq_len,
        top_k_blocks,
        query_chunk,
        block_indices_c.device,
    )
    _load_ext().run_build_reverse_index(
        block_indices_c,
        ws["counts"],
        ws["write_counts"],
        ws["bucket_offsets"],
        ws["task_meta"],
        ws["task_qids"],
        ws["num_tasks"],
        int(BLOCK_SIZE),
        int(query_chunk),
    )
    num_tasks = int(ws["num_tasks"].cpu()[0])
    return ws["task_meta"][:num_tasks], ws["task_qids"][:num_tasks]


def build_sparse_attention_metadata_cuda(
    block_indices: torch.Tensor,
    *,
    backward_query_chunk: int,
    remote_query_chunk: int = REMOTE_QUERY_CHUNK,
) -> SparseAttentionMetadata:
    """Build persistent backward tasks and compact remote-edge varlen metadata."""

    if block_indices.device.type != "cuda" or block_indices.ndim != 4:
        raise ValueError("block_indices must be a CUDA tensor shaped [B, Hp, S, Kb]")
    block_indices_c = block_indices.detach().to(torch.int32).contiguous()
    batch, n_proxy_heads, seq_len, top_k_blocks = map(int, block_indices_c.shape)
    if seq_len % BLOCK_SIZE:
        raise ValueError(f"sequence length must be divisible by {BLOCK_SIZE}")
    if remote_query_chunk < 1:
        raise ValueError("remote_query_chunk must be positive")

    # Use a per-forward workspace: these tensors are saved by autograd and must
    # not be overwritten by a later forward before its corresponding backward.
    backward_workspace = ReverseIndexWorkspace()
    task_meta, task_qids = build_reverse_index_cuda(
        block_indices_c,
        query_chunk=int(backward_query_chunk),
        workspace=backward_workspace,
    )

    num_blocks = seq_len // BLOCK_SIZE
    buckets = batch * n_proxy_heads * num_blocks
    max_edges = batch * n_proxy_heads * seq_len * top_k_blocks
    padded_remote_tasks = batch * n_proxy_heads * (
        ((seq_len * top_k_blocks + remote_query_chunk - 1) // remote_query_chunk)
        + num_blocks
    )
    device = block_indices_c.device
    remote_counts = torch.empty(buckets, device=device, dtype=torch.int32)
    remote_write_counts = torch.empty_like(remote_counts)
    remote_bucket_offsets = torch.empty(buckets + 1, device=device, dtype=torch.int32)
    remote_task_meta = torch.empty(
        (padded_remote_tasks, 5), device=device, dtype=torch.int32
    )
    remote_task_offsets = torch.empty(
        padded_remote_tasks + 1, device=device, dtype=torch.int32
    )
    packed_qids = torch.empty(max_edges, device=device, dtype=torch.int32)
    destinations = torch.empty(max_edges, device=device, dtype=torch.int32)
    edge_positions = torch.empty(max_edges, device=device, dtype=torch.int32)
    sizes = torch.empty(2, device=device, dtype=torch.int32)

    _load_ext().run_build_remote_metadata(
        block_indices_c,
        remote_counts,
        remote_write_counts,
        remote_bucket_offsets,
        remote_task_meta,
        remote_task_offsets,
        packed_qids,
        destinations,
        edge_positions,
        sizes,
        int(BLOCK_SIZE),
        int(remote_query_chunk),
    )
    sizes_cpu = sizes.cpu()
    num_remote_tasks = int(sizes_cpu[0])
    remote_task_meta_cpu = remote_task_meta[:num_remote_tasks].cpu()
    return SparseAttentionMetadata(
        task_meta=task_meta,
        task_qids=task_qids,
        remote_task_meta=remote_task_meta,
        remote_task_offsets=remote_task_offsets,
        packed_qids=packed_qids,
        destinations=destinations,
        edge_positions=edge_positions,
        num_remote_tasks=num_remote_tasks,
        remote_task_meta_cpu=remote_task_meta_cpu,
        batch=batch,
        n_proxy_heads=n_proxy_heads,
        seq_len=seq_len,
        top_k_blocks=top_k_blocks,
        remote_query_chunk=int(remote_query_chunk),
    )


def merge_attention_chunk_cuda(
    output_accum: torch.Tensor,
    lse_accum: torch.Tensor,
    remote_output: torch.Tensor,
    remote_lse: torch.Tensor,
    metadata: SparseAttentionMetadata,
    *,
    edge_start: int,
) -> None:
    _load_ext().merge_attention_chunk(
        output_accum,
        lse_accum,
        remote_output,
        remote_lse,
        metadata.destinations,
        metadata.edge_positions,
        int(edge_start),
        metadata.top_k_blocks,
    )


def merge_lse_chunk_cuda(
    lse_accum: torch.Tensor,
    remote_lse: torch.Tensor,
    metadata: SparseAttentionMetadata,
    *,
    edge_start: int,
) -> None:
    _load_ext().merge_lse_chunk(
        lse_accum,
        remote_lse,
        metadata.destinations,
        metadata.edge_positions,
        int(edge_start),
        metadata.top_k_blocks,
    )


__all__ = [
    "ReverseIndexWorkspace",
    "SparseAttentionMetadata",
    "build_reverse_index_cuda",
    "build_sparse_attention_metadata_cuda",
    "merge_attention_chunk_cuda",
    "merge_lse_chunk_cuda",
]
