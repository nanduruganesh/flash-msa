"""CuTeDSL kernel for MSA proxy block selection.

The selection pass streams proxy QK tiles like flash attention. For each
``(batch, proxy head, SELECT_M-query tile)`` CTA it scans all 128-token key
blocks and maintains only max-pooled top-k scores and block indices without
materializing either the full ``S x S`` score matrix or the compact
``S x num_blocks`` block-score matrix.
"""

import math

import cutlass
import torch
from cuda.bindings import driver as cuda
from cutlass import Int32, cute
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack

from flash_msa.reverse_index_cuda import SparseAttentionMetadata

BLOCK_SIZE = 128
SELECT_M = 128
KEY_SLICE_SIZE = 64
KEY_SLICES = BLOCK_SIZE // KEY_SLICE_SIZE
_COMPILE_CACHE = {}


def _to_cute_tensor(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor.detach(), assumed_align=16)


class _MSASelectBlocksKernel:
    def __init__(
        self,
        *,
        batch: int,
        n_proxy_heads: int,
        n_proxy_kv_heads: int,
        seq_len: int,
        top_k_blocks: int,
        head_dim: int,
        num_threads: int = 128,
    ) -> None:
        if head_dim != 128:
            raise NotImplementedError(f"MSA selector requires D=128, got {head_dim}")
        if seq_len % BLOCK_SIZE != 0:
            raise ValueError(f"sequence length must be divisible by {BLOCK_SIZE}")
        if n_proxy_heads % n_proxy_kv_heads != 0:
            raise ValueError("n_proxy_heads must be divisible by n_proxy_kv_heads")

        self.batch = int(batch)
        self.n_proxy_heads = int(n_proxy_heads)
        self.proxy_groups = int(n_proxy_heads) // int(n_proxy_kv_heads)
        self.seq_len = int(seq_len)
        self.num_blocks = int(seq_len) // BLOCK_SIZE
        self.num_query_tiles = (int(seq_len) + SELECT_M - 1) // SELECT_M
        self.top_k_blocks = int(top_k_blocks)
        self.head_dim = int(head_dim)
        self.head_dim_padded = (int(head_dim) + 31) // 32 * 32
        self.rows_per_task = SELECT_M
        self.num_threads = int(num_threads)

    @cute.jit
    def __call__(
        self,
        q_proxy: cute.Tensor,
        k_proxy: cute.Tensor,
        block_indices: cute.Tensor,
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(q_proxy.element_type != k_proxy.element_type):
            raise TypeError("q_proxy and k_proxy must have the same dtype")
        if cutlass.const_expr(
            not (
                q_proxy.element_type == cutlass.Float16
                or q_proxy.element_type == cutlass.BFloat16
            )
        ):
            raise TypeError("MSA selector supports only Float16 and BFloat16")
        if cutlass.const_expr(block_indices.element_type != cutlass.Int32):
            raise TypeError("block_indices must be Int32")
        self._dtype = q_proxy.element_type

        smem_k_block_size = 64
        s_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size), stride=(smem_k_block_size, 1)),
        )
        sQ_layout = cute.tile_to_shape(
            s_layout_atom,
            (self.rows_per_task, self.head_dim_padded),
            (0, 1),
        )
        sK_layout = cute.tile_to_shape(
            s_layout_atom,
            (KEY_SLICE_SIZE, self.head_dim_padded),
            (0, 1),
        )

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sK_layout)], 1024
            ]

        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, 16, 16),
        )
        copy_elements = 128 // self._dtype.width
        threads_per_row = self.head_dim // copy_elements
        gmem_copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            self._dtype,
            num_bits_per_copy=128,
        )
        gmem_thread_layout = cute.make_ordered_layout(
            (self.num_threads // threads_per_row, threads_per_row), order=(1, 0)
        )
        gmem_value_layout = cute.make_layout((1, copy_elements))
        gmem_tiled_copy = cute.make_tiled_copy_tv(
            gmem_copy_atom, gmem_thread_layout, gmem_value_layout
        )

        LOG2_E = 1.4426950408889634074
        softmax_scale_log2 = softmax_scale * LOG2_E
        self.kernel(
            q_proxy,
            k_proxy,
            block_indices,
            softmax_scale_log2,
            sQ_layout,
            sK_layout,
            gmem_tiled_copy,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=[self.num_query_tiles, self.n_proxy_heads, self.batch],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_proxy: cute.Tensor,
        k_proxy: cute.Tensor,
        block_indices: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        gmem_tiled_copy: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        query_tile, proxy_head, batch = cute.arch.block_idx()
        query_start = query_tile * Int32(SELECT_M)
        query_count = cutlass.min(Int32(SELECT_M), Int32(self.seq_len) - query_start)
        proxy_kv_head = proxy_head // Int32(self.proxy_groups)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sK_layout)

        zero = cutlass.Float32(0.0).to(self._dtype)
        softmax_scale = softmax_scale_log2 * cutlass.Float32(math.log(2.0))
        q_linear = tidx
        while q_linear < Int32(self.rows_per_task * self.head_dim):
            row = q_linear // Int32(self.head_dim)
            dim = q_linear - row * Int32(self.head_dim)
            q_pos = query_start + row
            if row < query_count:
                sQ[row, dim] = q_proxy[batch, proxy_head, q_pos, dim]
            else:
                sQ[row, dim] = zero
            q_linear += Int32(self.num_threads)

        cute.arch.sync_threads()

        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))

        smem_copy_atom_Q = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self._dtype,
        )
        smem_copy_atom_K = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self._dtype,
        )
        smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_Q, tiled_mma)
        smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_K, tiled_mma)
        smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
        smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tSrK_copy_view = smem_thr_copy_K.retile(tSrK)
        gmem_thr_copy = gmem_tiled_copy.get_slice(tidx)
        tKsK_gmem = gmem_thr_copy.partition_D(sK)

        cS = cute.make_identity_tensor((self.rows_per_task, KEY_SLICE_SIZE))
        tScS = thr_mma.partition_C(cS)
        tScS_mn = self._make_acc_tensor_mn_view(tScS)
        acc_shape_S = thr_mma.partition_shape_C((self.rows_per_task, KEY_SLICE_SIZE))
        num_acc_rows = cute.size(tScS_mn.shape[0])

        top_vals = cute.make_rmem_tensor((num_acc_rows, self.top_k_blocks), cutlass.Float32)
        top_idx = cute.make_rmem_tensor((num_acc_rows, self.top_k_blocks), Int32)
        top_vals.fill(-cutlass.Float32.inf)
        for rr in cutlass.range_constexpr(num_acc_rows):
            for slot in cutlass.range_constexpr(self.top_k_blocks):
                # Empty top-k slots must use an out-of-range sentinel.  Seeding
                # them with real block IDs creates duplicates when those blocks
                # are subsequently inserted with their computed scores.
                top_idx[rr, slot] = Int32(self.num_blocks)

        cute.copy(
            smem_tiled_copy_Q,
            tSsQ[None, None, 0],
            tSrQ_copy_view[None, None, 0],
        )

        # SELECT_M == BLOCK_SIZE, so every valid query in this CTA belongs to
        # the same causal block.  Future blocks can never enter the selected
        # attention set; leave their top-k slots at the out-of-range sentinel.
        key_block_end = query_start // Int32(BLOCK_SIZE) + Int32(1)
        key_block = Int32(0)
        while key_block < key_block_end:
            block_max = cute.make_rmem_tensor((num_acc_rows,), cutlass.Float32)
            block_max.fill(-cutlass.Float32.inf)

            for key_slice in cutlass.range_constexpr(KEY_SLICES):
                key_tile = key_block * Int32(KEY_SLICES) + Int32(key_slice)
                gK = cute.local_tile(
                    k_proxy[batch, proxy_kv_head, None, None],
                    (KEY_SLICE_SIZE, self.head_dim),
                    (key_tile, 0),
                )
                # The dynamic causal loop bound currently makes CuTe lose the
                # statically valid 16-byte alignment of this tile offset.
                # Every tile starts at 64 * D bf16 elements, so reassert it.
                gK = cute.make_tensor(
                    cute.make_ptr(
                        self._dtype,
                        gK.iterator.llvm_ptr,
                        gK.iterator.memspace,
                        assumed_align=16,
                    ),
                    gK.layout,
                )
                tKgK = gmem_thr_copy.partition_S(gK)
                cute.copy(gmem_tiled_copy, tKgK, tKsK_gmem)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()

                acc_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)
                acc_S.fill(0.0)
                self._gemm_qk(
                    tiled_mma,
                    smem_tiled_copy_Q,
                    smem_tiled_copy_K,
                    tSsQ,
                    tSsK,
                    tSrQ,
                    tSrK,
                    tSrQ_copy_view,
                    tSrK_copy_view,
                    acc_S,
                )

                acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
                for rr in cutlass.range_constexpr(num_acc_rows):
                    row_m = tScS_mn[rr, 0][0]
                    q_pos = query_start + row_m
                    row_is_valid = row_m < query_count
                    for cc in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
                        col_n = tScS_mn[rr, cc][1]
                        k_pos = (
                            key_block * Int32(BLOCK_SIZE)
                            + Int32(key_slice * KEY_SLICE_SIZE)
                            + col_n
                        )
                        if (not row_is_valid) or k_pos > q_pos:
                            acc_S_mn[rr, cc] = -cutlass.Float32.inf

                    row_scores = (acc_S_mn[rr, None].load() * softmax_scale).to(
                        self._dtype
                    ).to(cutlass.Float32)
                    row_max = row_scores.reduce(
                        cute.ReductionOp.MAX,
                        -cutlass.Float32.inf,
                        0,
                    )
                    row_max = self._threadquad_reduce_max(row_max)
                    block_max[rr] = cute.arch.fmax(block_max[rr], row_max)

                cute.arch.sync_threads()

            for rr in cutlass.range_constexpr(num_acc_rows):
                row_m = tScS_mn[rr, 0][0]
                q_pos = query_start + row_m
                row_is_valid = row_m < query_count
                score = block_max[rr]
                local_block = q_pos // Int32(BLOCK_SIZE)
                if row_is_valid and key_block == local_block:
                    score = cutlass.Float32.inf
                self._insert_topk(top_vals, top_idx, rr, score, key_block)

            key_block += Int32(1)

        for rr in cutlass.range_constexpr(num_acc_rows):
            row_m = tScS_mn[rr, 0][0]
            col_n = tScS_mn[rr, 0][1]
            q_pos = query_start + row_m
            row_is_valid = row_m < query_count
            if col_n == Int32(0) and row_is_valid:
                for slot in cutlass.range_constexpr(self.top_k_blocks):
                    block_indices[batch, proxy_head, q_pos, slot] = top_idx[rr, slot]

    @cute.jit
    def _gemm_qk(
        self,
        tiled_mma: cute.TiledMma,
        smem_tiled_copy_Q: cute.TiledCopy,
        smem_tiled_copy_K: cute.TiledCopy,
        tSsQ: cute.Tensor,
        tSsK: cute.Tensor,
        tSrQ: cute.Tensor,
        tSrK: cute.Tensor,
        tSrQ_copy_view: cute.Tensor,
        tSrK_copy_view: cute.Tensor,
        acc_S: cute.Tensor,
    ):
        cute.copy(
            smem_tiled_copy_K,
            tSsK[None, None, 0],
            tSrK_copy_view[None, None, 0],
        )
        for kk in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
            kk_next = (kk + 1) % cute.size(tSsQ.shape[2])
            cute.copy(
                smem_tiled_copy_Q,
                tSsQ[None, None, kk_next],
                tSrQ_copy_view[None, None, kk_next],
            )
            cute.copy(
                smem_tiled_copy_K,
                tSsK[None, None, kk_next],
                tSrK_copy_view[None, None, kk_next],
            )
            cute.gemm(
                tiled_mma,
                acc_S,
                tSrQ[None, None, kk],
                tSrK[None, None, kk],
                acc_S,
            )

    @cute.jit
    def _insert_topk(
        self,
        top_vals: cute.Tensor,
        top_idx: cute.Tensor,
        rr: Int32,
        score: cutlass.Float32,
        block_idx: Int32,
    ):
        insert_val = score
        insert_idx = block_idx
        for slot in cutlass.range_constexpr(self.top_k_blocks):
            old_val = top_vals[rr, slot]
            old_idx = top_idx[rr, slot]
            if insert_val > old_val or (insert_val == old_val and insert_idx < old_idx):
                top_vals[rr, slot] = insert_val
                top_idx[rr, slot] = insert_idx
                if old_idx == insert_idx:
                    insert_val = -cutlass.Float32.inf
                    insert_idx = Int32(self.num_blocks)
                else:
                    insert_val = old_val
                    insert_idx = old_idx

    def _make_acc_tensor_mn_view(self, acc: cute.Tensor) -> cute.Tensor:
        acc_layout_col_major = cute.make_layout(acc.layout.shape)
        acc_layout_mn = cute.make_layout(
            (
                (
                    acc_layout_col_major.shape[0][1],
                    acc_layout_col_major.shape[1],
                ),
                (
                    acc_layout_col_major.shape[0][0],
                    acc_layout_col_major.shape[2],
                ),
            ),
            stride=(
                (
                    acc_layout_col_major.stride[0][1],
                    acc_layout_col_major.stride[1],
                ),
                (
                    acc_layout_col_major.stride[0][0],
                    acc_layout_col_major.stride[2],
                ),
            ),
        )
        acc_layout_mn = cute.composition(acc.layout, acc_layout_mn)
        return cute.make_tensor(acc.iterator, acc_layout_mn)

    def _threadquad_reduce(self, val: cutlass.Float32, op) -> cutlass.Float32:
        val = op(
            val,
            cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31),
        )
        val = op(
            val,
            cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31),
        )
        return val

    def _threadquad_reduce_max(self, val: cutlass.Float32) -> cutlass.Float32:
        return self._threadquad_reduce(val, lambda x, y: cute.arch.fmax(x, y))


def _compile_select_kernel(
    batch: int,
    n_proxy_heads: int,
    n_proxy_kv_heads: int,
    seq_len: int,
    top_k_blocks: int,
    head_dim: int,
    q_proxy: cute.Tensor,
    k_proxy: cute.Tensor,
    block_indices: cute.Tensor,
    softmax_scale: float,
    stream: cuda.CUstream,
):
    key = (
        "select_blocks",
        int(batch),
        int(n_proxy_heads),
        int(n_proxy_kv_heads),
        int(seq_len),
        int(top_k_blocks),
        int(head_dim),
        q_proxy.element_type,
        k_proxy.element_type,
        block_indices.element_type,
    )
    if key not in _COMPILE_CACHE:
        kernel = _MSASelectBlocksKernel(
            batch=batch,
            n_proxy_heads=n_proxy_heads,
            n_proxy_kv_heads=n_proxy_kv_heads,
            seq_len=seq_len,
            top_k_blocks=top_k_blocks,
            head_dim=head_dim,
        )
        _COMPILE_CACHE[key] = cute.compile(
            kernel,
            q_proxy,
            k_proxy,
            block_indices,
            float(softmax_scale),
            stream,
        )
    return _COMPILE_CACHE[key]


def select_blocks(
    q_proxy: torch.Tensor,
    k_proxy: torch.Tensor,
    *,
    scale: float,
    num_blocks: int,
    top_k_blocks: int,
) -> torch.Tensor:
    """Select proxy blocks with a CuTeDSL MMA kernel."""

    if q_proxy.device.type != "cuda" or k_proxy.device.type != "cuda":
        raise ValueError("CuTeDSL selector requires CUDA tensors")
    if q_proxy.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"CuTeDSL selector supports fp16/bf16, got {q_proxy.dtype}")
    if q_proxy.ndim != 4 or k_proxy.ndim != 4:
        raise ValueError("q_proxy and k_proxy must have shape (B, H, S, D)")

    batch, n_proxy_heads, seq_len, head_dim = q_proxy.shape
    if int(num_blocks) != seq_len // BLOCK_SIZE:
        raise ValueError("num_blocks does not match q_proxy sequence length")
    if q_proxy.shape[0] != k_proxy.shape[0] or q_proxy.shape[2:] != k_proxy.shape[2:]:
        raise ValueError("q_proxy and k_proxy must agree on batch, sequence, and head_dim")
    n_proxy_kv_heads = int(k_proxy.shape[1])

    q_c = q_proxy.detach().contiguous()
    k_c = k_proxy.detach().contiguous()
    block_indices = torch.empty(
        (batch, n_proxy_heads, seq_len, int(top_k_blocks)),
        dtype=torch.int32,
        device=q_proxy.device,
    )

    q_t = _to_cute_tensor(q_c)
    k_t = _to_cute_tensor(k_c)
    block_indices_t = _to_cute_tensor(block_indices)
    stream = cuda.CUstream(torch.cuda.current_stream(q_proxy.device).cuda_stream)

    compiled_select = _compile_select_kernel(
        batch,
        n_proxy_heads,
        n_proxy_kv_heads,
        seq_len,
        int(top_k_blocks),
        head_dim,
        q_t,
        k_t,
        block_indices_t,
        float(scale),
        stream,
    )
    compiled_select(
        q_t,
        k_t,
        block_indices_t,
        float(scale),
        stream,
    )
    return block_indices


def compute_proxy_lse(
    q_proxy: torch.Tensor,
    k_proxy: torch.Tensor,
    *,
    scale: float,
    metadata: SparseAttentionMetadata,
) -> torch.Tensor:
    """Compute selected-token proxy LSE with varlen FlashAttention."""

    if q_proxy.device.type != "cuda" or k_proxy.device.type != "cuda":
        raise ValueError("varlen FlashAttention proxy LSE requires CUDA tensors")
    if q_proxy.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            f"varlen FlashAttention proxy LSE supports fp16/bf16, got {q_proxy.dtype}"
        )
    if q_proxy.ndim != 4 or k_proxy.ndim != 4:
        raise ValueError("q_proxy and k_proxy must have shape (B, H, S, D)")

    batch, n_proxy_heads, seq_len, head_dim = q_proxy.shape
    if q_proxy.shape[0] != k_proxy.shape[0] or q_proxy.shape[2:] != k_proxy.shape[2:]:
        raise ValueError("q_proxy and k_proxy must agree on batch, sequence, and head_dim")
    if (metadata.batch, metadata.n_proxy_heads, metadata.seq_len) != (
        batch,
        n_proxy_heads,
        seq_len,
    ):
        raise ValueError("metadata has an incompatible shape")

    q_c = q_proxy.detach().contiguous()
    k_c = k_proxy.detach().contiguous()
    from flash_msa.sparse_flash_varlen import sparse_flash_varlen_forward

    _unused_out, lse_proxy = sparse_flash_varlen_forward(
        q_c,
        k_c,
        k_c,
        metadata=metadata,
        scale=float(scale),
        return_output=False,
    )
    return lse_proxy
