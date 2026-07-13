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


@dataclass
class DocumentSegmentMetadata:
    """Compact intersections of contiguous documents and physical MSA blocks."""

    starts: torch.Tensor
    lengths: torch.Tensor
    batches: torch.Tensor
    doc_first_segment: torch.Tensor
    token_segment_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    full_segments: torch.Tensor | None = None
    full_segments_cpu: torch.Tensor | None = None

    @property
    def num_segments(self) -> int:
        return int(self.starts.shape[0])


def _build_document_segment_metadata_from_starts(
    document_starts: torch.Tensor,
    *,
    block_size: int,
) -> DocumentSegmentMetadata:
    """Build physical segments from a boolean ``[B, S]`` document-start mask."""

    batch, seq_len = map(int, document_starts.shape)
    block_starts = torch.zeros(
        seq_len, dtype=torch.bool, device=document_starts.device
    )
    block_starts[:: int(block_size)] = True
    segment_starts_mask = document_starts | block_starts.unsqueeze(0)
    flat_segment_starts = torch.nonzero(
        segment_starts_mask.reshape(-1), as_tuple=False
    ).flatten()

    flat_end = torch.tensor(
        [batch * seq_len], dtype=torch.int64, device=document_starts.device
    )
    flat_boundaries = torch.cat((flat_segment_starts, flat_end))
    lengths = (flat_boundaries[1:] - flat_boundaries[:-1]).to(torch.int32)
    batches = torch.div(flat_segment_starts, seq_len, rounding_mode="floor").to(
        torch.int32
    )
    starts = (
        flat_segment_starts - batches.to(torch.int64) * seq_len
    ).to(torch.int32)

    segment_is_doc_start = document_starts.reshape(-1)[flat_segment_starts]
    segment_indices = torch.arange(
        flat_segment_starts.numel(),
        dtype=torch.int32,
        device=document_starts.device,
    )
    doc_first_segment = torch.where(
        segment_is_doc_start, segment_indices, 0
    ).cummax(dim=0).values
    token_segment_ids = (
        segment_starts_mask.reshape(-1).cumsum(dim=0, dtype=torch.int32) - 1
    ).reshape(batch, seq_len)
    segment_cu_seqlens = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=document_starts.device),
            lengths.cumsum(dim=0, dtype=torch.int32),
        )
    )
    full_segments = (
        (starts.remainder(int(block_size)) == 0) & (lengths == int(block_size))
    ).to(torch.int32)
    full_segments_cpu = full_segments.to(torch.bool).cpu()
    return DocumentSegmentMetadata(
        starts=starts,
        lengths=lengths,
        batches=batches,
        doc_first_segment=doc_first_segment,
        token_segment_ids=token_segment_ids,
        cu_seqlens=segment_cu_seqlens,
        full_segments=full_segments,
        full_segments_cpu=full_segments_cpu,
    )


def build_document_segment_metadata(
    document_list: torch.Tensor,
    *,
    block_size: int = BLOCK_SIZE,
) -> DocumentSegmentMetadata:
    """Split packed documents at both document and physical block boundaries."""

    if document_list.ndim != 2:
        raise ValueError("document_list must have shape [B, S]")
    if document_list.device.type != "cuda":
        raise ValueError("document_list must be a CUDA tensor")
    if document_list.dtype not in (torch.int32, torch.int64):
        raise TypeError("document_list must contain int32 or int64 document IDs")
    batch, seq_len = map(int, document_list.shape)
    if batch < 1 or seq_len < 1:
        raise ValueError("document_list batch and sequence dimensions must be positive")
    if block_size < 1:
        raise ValueError("block_size must be positive")

    docs = document_list.detach().contiguous()
    document_starts = torch.zeros_like(docs, dtype=torch.bool)
    document_starts[:, 0] = True
    document_starts[:, 1:] = docs[:, 1:] != docs[:, :-1]
    return _build_document_segment_metadata_from_starts(
        document_starts, block_size=int(block_size)
    )


def build_document_segment_metadata_from_cu_seqlens(
    cu_seqlens: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    block_size: int = BLOCK_SIZE,
) -> DocumentSegmentMetadata:
    """Build document segments directly from FA4-style cumulative offsets.

    Offsets index the flattened ``B * S`` token dimension. Empty documents are
    accepted, but every batch-row boundary must occur in ``cu_seqlens`` because
    Flash-MSA inputs retain a separate batch dimension.
    """

    if cu_seqlens.ndim != 1:
        raise ValueError("cu_seqlens must have shape [num_documents + 1]")
    if cu_seqlens.device.type != "cuda":
        raise ValueError("cu_seqlens must be a CUDA tensor")
    if cu_seqlens.dtype != torch.int32:
        raise TypeError("cu_seqlens must have dtype torch.int32")
    if cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must contain at least two offsets")
    if batch_size < 1 or seq_len < 1:
        raise ValueError("batch_size and seq_len must be positive")
    if block_size < 1:
        raise ValueError("block_size must be positive")

    offsets = cu_seqlens.detach().contiguous()
    offsets_cpu = offsets.cpu()
    total_tokens = int(batch_size) * int(seq_len)
    if int(offsets_cpu[0]) != 0:
        raise ValueError("cu_seqlens must start at 0")
    if int(offsets_cpu[-1]) != total_tokens:
        raise ValueError(
            f"cu_seqlens must end at B * S ({total_tokens}), got "
            f"{int(offsets_cpu[-1])}"
        )
    if bool((offsets_cpu[1:] < offsets_cpu[:-1]).any()):
        raise ValueError("cu_seqlens must be nondecreasing")

    required_boundaries = torch.arange(
        0, total_tokens + 1, int(seq_len), dtype=torch.int32
    )
    boundary_positions = torch.searchsorted(offsets_cpu, required_boundaries)
    boundary_positions.clamp_max_(offsets_cpu.numel() - 1)
    if not torch.equal(offsets_cpu[boundary_positions], required_boundaries):
        raise ValueError(
            "cu_seqlens must contain every batch-row boundary "
            "(0, S, 2*S, ..., B*S)"
        )

    document_starts = torch.zeros(
        total_tokens, dtype=torch.bool, device=offsets.device
    )
    document_offsets = offsets[:-1]
    document_offsets = document_offsets[document_offsets < total_tokens]
    document_starts[document_offsets.to(torch.int64)] = True
    document_starts = document_starts.reshape(int(batch_size), int(seq_len))
    return _build_document_segment_metadata_from_starts(
        document_starts, block_size=int(block_size)
    )


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
    document_segments: DocumentSegmentMetadata | None = None


def build_reverse_index_cuda(
    block_indices: torch.Tensor,
    *,
    query_chunk: int = QUERY_CHUNK,
    workspace: ReverseIndexWorkspace | None = None,
    document_segments: DocumentSegmentMetadata | None = None,
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

    if document_segments is None:
        ws = (workspace or _DEFAULT_WORKSPACE).get(
            batch,
            n_proxy_heads,
            seq_len,
            top_k_blocks,
            query_chunk,
            block_indices_c.device,
        )
        build = _load_ext().run_build_reverse_index
        extra_args = (int(BLOCK_SIZE), int(query_chunk))
    else:
        num_segments = document_segments.num_segments
        if document_segments.full_segments is None:
            document_segments.full_segments = (
                (document_segments.starts.remainder(BLOCK_SIZE) == 0)
                & (document_segments.lengths == BLOCK_SIZE)
            ).to(torch.int32)
        padded_tasks = n_proxy_heads * (
            ((batch * seq_len * top_k_blocks + query_chunk - 1) // query_chunk)
            + num_segments
        )
        buckets = n_proxy_heads * num_segments
        device = block_indices_c.device
        ws = {
            "counts": torch.empty(buckets, device=device, dtype=torch.int32),
            "write_counts": torch.empty(buckets, device=device, dtype=torch.int32),
            "bucket_offsets": torch.empty(buckets, device=device, dtype=torch.int32),
            "task_meta": torch.empty((padded_tasks, 4), device=device, dtype=torch.int32),
            "task_qids": torch.empty(
                (padded_tasks, query_chunk), device=device, dtype=torch.int32
            ),
            "num_tasks": torch.empty(1, device=device, dtype=torch.int32),
        }
        build = _load_ext().run_build_reverse_index_segments
        extra_args = (
            document_segments.token_segment_ids.contiguous(),
            document_segments.batches.contiguous(),
            document_segments.full_segments.contiguous(),
            int(num_segments),
            int(query_chunk),
        )
    build(
        block_indices_c,
        ws["counts"],
        ws["write_counts"],
        ws["bucket_offsets"],
        ws["task_meta"],
        ws["task_qids"],
        ws["num_tasks"],
        *extra_args,
    )
    num_tasks = int(ws["num_tasks"].cpu()[0])
    return ws["task_meta"][:num_tasks], ws["task_qids"][:num_tasks]


def build_sparse_attention_metadata_cuda(
    block_indices: torch.Tensor,
    *,
    backward_query_chunk: int,
    remote_query_chunk: int = REMOTE_QUERY_CHUNK,
    document_segments: DocumentSegmentMetadata | None = None,
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
        document_segments=document_segments,
    )

    num_key_units = (
        seq_len // BLOCK_SIZE
        if document_segments is None
        else document_segments.num_segments
    )
    buckets = (
        batch * n_proxy_heads * num_key_units
        if document_segments is None
        else n_proxy_heads * num_key_units
    )
    max_edges = batch * n_proxy_heads * seq_len * top_k_blocks
    bucket_units_per_head = (
        batch * num_key_units if document_segments is None else num_key_units
    )
    padded_remote_tasks = n_proxy_heads * (
        ((batch * seq_len * top_k_blocks + remote_query_chunk - 1) // remote_query_chunk)
        + bucket_units_per_head
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

    build_remote = _load_ext().run_build_remote_metadata
    remote_extra_args = (int(BLOCK_SIZE), int(remote_query_chunk))
    if document_segments is not None:
        if document_segments.full_segments is None:
            document_segments.full_segments = (
                (document_segments.starts.remainder(BLOCK_SIZE) == 0)
                & (document_segments.lengths == BLOCK_SIZE)
            ).to(torch.int32)
        build_remote = _load_ext().run_build_remote_metadata_segments
        remote_extra_args = (
            document_segments.token_segment_ids.contiguous(),
            document_segments.batches.contiguous(),
            document_segments.full_segments.contiguous(),
            int(document_segments.num_segments),
            int(remote_query_chunk),
        )
    build_remote(
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
        *remote_extra_args,
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
        document_segments=document_segments,
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
    "DocumentSegmentMetadata",
    "ReverseIndexWorkspace",
    "SparseAttentionMetadata",
    "build_reverse_index_cuda",
    "build_document_segment_metadata",
    "build_document_segment_metadata_from_cu_seqlens",
    "build_sparse_attention_metadata_cuda",
    "merge_attention_chunk_cuda",
    "merge_lse_chunk_cuda",
]
