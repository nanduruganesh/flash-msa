"""CuTeDSL backward kernels for MSA selected-edge attention.

Unlike the forward, requires two different paths for n_kv_heads <= n_proxy_heads vs. n_kv_heads > n_proxy_heads

The fused MMA kernel recomputes per-tile probabilities, forms main/proxy
``dS`` tiles, issues tensor-core products for ``dQ/dK/dV`` and proxy
``dQ/dK``, then accumulates dense FP32 gradient buffers with native atomics.
"""

import inspect
import os

import cutlass
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, cute
from cutlass._mlir.dialects import nvvm
from cutlass.cute.nvgpu import warp
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

BLOCK_SIZE = 128
KEY_SLICE_SIZE = 64
KEY_SLICES = BLOCK_SIZE // KEY_SLICE_SIZE
MMA_ROWS_PER_TASK = int(os.environ.get("MSA_NATIVE_BWD_MMA_ROWS", "64"))
_COMPILE_CACHE = {}
_NVVM_ATOMICRMW_HAS_RES = "res" in inspect.signature(nvvm.atomicrmw).parameters


def _to_cute_tensor(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor.detach(), assumed_align=16)


def _derive_head_tiling(
    n_heads: int, n_kv_heads: int, n_proxy_heads: int
) -> tuple[int, int, int, int]:
    """Return ``(main_per_proxy, query_chunk, rows_per_task, proxy_query_rows)``."""

    if n_heads % n_proxy_heads != 0:
        raise NotImplementedError("n_heads must be divisible by n_proxy_heads")
    if n_heads % n_kv_heads != 0:
        raise NotImplementedError("n_heads must be divisible by n_kv_heads")
    if n_proxy_heads < n_kv_heads or n_proxy_heads % n_kv_heads != 0:
        raise NotImplementedError(
            "native backward base kernel requires n_proxy_heads >= n_kv_heads and divisibility"
        )

    main_per_proxy = int(n_heads) // int(n_proxy_heads)
    if main_per_proxy <= 0 or MMA_ROWS_PER_TASK % main_per_proxy != 0:
        raise NotImplementedError(
            f"main heads per proxy must divide {MMA_ROWS_PER_TASK}, got {main_per_proxy}"
        )

    query_chunk = MMA_ROWS_PER_TASK // main_per_proxy
    proxy_query_rows = max(32, query_chunk)
    return main_per_proxy, query_chunk, MMA_ROWS_PER_TASK, proxy_query_rows


@dsl_user_op
def _atomic_add_fp32(value: float | Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    if _NVVM_ATOMICRMW_HAS_RES:
        nvvm.atomicrmw(
            T.f32(), nvvm.AtomicOpKind.FADD, gmem_ptr.llvm_ptr, Float32(value).ir_value()
        )
    else:
        nvvm.atomicrmw(nvvm.AtomicOpKind.FADD, gmem_ptr.llvm_ptr, Float32(value).ir_value())


@dsl_user_op
def _elem_pointer(tensor: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
    return tensor.iterator + cute.crd2idx(coord, tensor.layout, loc=loc, ip=ip)


def _transpose_view(tensor: cute.Tensor) -> cute.Tensor:
    """Transpose the first two dimensions of a shared-memory tensor."""

    shape = (tensor.shape[1], tensor.shape[0], *tensor.shape[2:])
    order = (1, 0, *range(2, cute.rank(tensor)))
    return cute.composition(tensor, cute.make_ordered_layout(shape, order=order))


def _make_acc_tensor_mn_view(acc: cute.Tensor) -> cute.Tensor:
    acc_layout_col_major = cute.make_layout(acc.layout.shape)
    acc_layout_mn = cute.make_layout(
        (
            (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),
            (
                acc_layout_col_major.shape[0][0],
                *acc_layout_col_major.shape[0][2:],
                acc_layout_col_major.shape[2],
            ),
            *acc_layout_col_major.shape[3:],
        ),
        stride=(
            (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),
            (
                acc_layout_col_major.stride[0][0],
                *acc_layout_col_major.stride[0][2:],
                acc_layout_col_major.stride[2],
            ),
            *acc_layout_col_major.stride[3:],
        ),
    )
    return cute.make_tensor(acc.iterator, cute.composition(acc.layout, acc_layout_mn))


class _MSAFusedBackwardMMAKernel:
    def __init__(
        self,
        *,
        batch: int,
        n_heads: int,
        n_kv_heads: int,
        n_proxy_heads: int,
        n_proxy_kv_heads: int,
        seq_len: int,
        head_dim: int,
        num_tasks: int,
        input_query_chunk: int,
        num_threads: int = 256,
    ) -> None:
        if head_dim != 128:
            raise NotImplementedError(f"fused backward requires D=128, got {head_dim}")
        main_per_proxy, query_chunk, rows_per_task, proxy_query_rows = _derive_head_tiling(
            int(n_heads), int(n_kv_heads), int(n_proxy_heads)
        )

        self.head_dim = int(head_dim)
        self.head_dim_padded = (int(head_dim) + 31) // 32 * 32
        self.main_per_proxy = main_per_proxy
        self.query_chunk = query_chunk
        self.rows_per_task = rows_per_task
        self.proxy_query_rows = proxy_query_rows
        self.proxy_heads_per_kv = int(n_proxy_heads) // int(n_kv_heads)
        self.proxy_groups = int(n_proxy_heads) // int(n_proxy_kv_heads)
        self.input_query_chunk = int(input_query_chunk)
        if self.input_query_chunk % query_chunk != 0:
            raise NotImplementedError(
                "fused backward input task query width must be divisible by "
                f"backward query_chunk={query_chunk}, got {self.input_query_chunk}"
            )
        self.chunks_per_input_task = self.input_query_chunk // query_chunk
        self.num_tasks = int(num_tasks) * self.chunks_per_input_task
        self.num_threads = int(num_threads)

    @cute.jit
    def __call__(
        self,
        q_proxy: cute.Tensor,
        k_proxy: cute.Tensor,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        grad_o_main: cute.Tensor,
        lse_main: cute.Tensor,
        lse_proxy: cute.Tensor,
        delta_main: cute.Tensor,
        task_meta: cute.Tensor,
        task_qids: cute.Tensor,
        dq_proxy: cute.Tensor,
        dk_proxy: cute.Tensor,
        dq: cute.Tensor,
        dk: cute.Tensor,
        dv: cute.Tensor,
        softmax_scale: cutlass.Float32,
        grad_kl_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(
            not (
                q.element_type
                == k.element_type
                == v.element_type
                == q_proxy.element_type
                == k_proxy.element_type
                == grad_o_main.element_type
            )
        ):
            raise TypeError("fused backward inputs must have the same fp16/bf16 dtype")
        if cutlass.const_expr(
            not (q.element_type == cutlass.Float16 or q.element_type == cutlass.BFloat16)
        ):
            raise TypeError("fused backward supports only Float16 and BFloat16")

        self._dtype = q.element_type

        smem_k_block_size = 64
        s_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size), stride=(smem_k_block_size, 1)),
        )
        sQ_layout = cute.tile_to_shape(
            s_layout_atom, (self.rows_per_task, self.head_dim_padded), (0, 1)
        )
        sKV_layout = cute.tile_to_shape(
            s_layout_atom, (KEY_SLICE_SIZE, self.head_dim_padded), (0, 1)
        )
        sPdS_layout = cute.tile_to_shape(
            s_layout_atom, (self.rows_per_task, KEY_SLICE_SIZE), (0, 1)
        )
        sQpx_layout = cute.tile_to_shape(
            s_layout_atom, (self.proxy_query_rows, self.head_dim_padded), (0, 1)
        )
        sPdSpx_layout = cute.tile_to_shape(
            s_layout_atom, (self.proxy_query_rows, KEY_SLICE_SIZE), (0, 1)
        )

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024]
            sdO: cute.struct.Align[cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024]
            sP: cute.struct.Align[cute.struct.MemRange[self._dtype, cute.cosize(sPdS_layout)], 1024]
            sdS: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sPdS_layout)], 1024
            ]
            sQpx: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sQpx_layout)], 1024
            ]
            sKpx: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]

        mma_warps = self.num_threads // 32
        mma_m_warps = min(mma_warps, max(1, self.rows_per_task // 16))
        mma_n_warps = max(1, mma_warps // mma_m_warps)
        tiled_mma_sdp = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (mma_m_warps, mma_n_warps, 1),
            permutation_mnk=(mma_m_warps * 16, mma_n_warps * 16, 16),
        )
        tiled_mma_dq = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (mma_m_warps, mma_n_warps, 1),
            permutation_mnk=(mma_m_warps * 16, mma_n_warps * 16, 16),
        )
        tiled_mma_dkv = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (2, self.num_threads // 64, 1),
            permutation_mnk=(32, self.num_threads // 64 * 16, 16),
        )
        tiled_mma_px = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (2, self.num_threads // 64, 1),
            permutation_mnk=(32, self.num_threads // 64 * 16, 16),
        )

        LOG2_E = 1.4426950408889634074
        self.kernel(
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
            dq_proxy,
            dk_proxy,
            dq,
            dk,
            dv,
            softmax_scale,
            softmax_scale * Float32(LOG2_E),
            Float32(LOG2_E),
            grad_kl_scale,
            sQ_layout,
            sKV_layout,
            sPdS_layout,
            sQpx_layout,
            sPdSpx_layout,
            tiled_mma_sdp,
            tiled_mma_dq,
            tiled_mma_dkv,
            tiled_mma_px,
            SharedStorage,
        ).launch(grid=[self.num_tasks, 1, 1], block=[self.num_threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        q_proxy: cute.Tensor,
        k_proxy: cute.Tensor,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        grad_o_main: cute.Tensor,
        lse_main: cute.Tensor,
        lse_proxy: cute.Tensor,
        delta_main: cute.Tensor,
        task_meta: cute.Tensor,
        task_qids: cute.Tensor,
        dq_proxy: cute.Tensor,
        dk_proxy: cute.Tensor,
        dq: cute.Tensor,
        dk: cute.Tensor,
        dv: cute.Tensor,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        log2_e: cutlass.Float32,
        grad_kl_scale: cutlass.Float32,
        sQ_layout: cute.ComposedLayout,
        sKV_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        sQpx_layout: cute.ComposedLayout,
        sPdSpx_layout: cute.ComposedLayout,
        tiled_mma_sdp: cute.TiledMma,
        tiled_mma_dq: cute.TiledMma,
        tiled_mma_dkv: cute.TiledMma,
        tiled_mma_px: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        launch_task_idx, _, _ = cute.arch.block_idx()
        task_idx = launch_task_idx // Int32(self.chunks_per_input_task)
        task_chunk = launch_task_idx - task_idx * Int32(self.chunks_per_input_task)
        qid_base = task_chunk * Int32(self.query_chunk)

        batch = task_meta[task_idx, 0]
        proxy_head = task_meta[task_idx, 1]
        key_block = task_meta[task_idx, 2]
        query_count = cutlass.min(Int32(self.query_chunk), task_meta[task_idx, 3] - qid_base)
        kv_head = proxy_head // Int32(self.proxy_heads_per_kv)
        proxy_kv_head = proxy_head // Int32(self.proxy_groups)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sdO = storage.sdO.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sKV_layout)
        sV = storage.sV.get_tensor(sKV_layout)
        sP = storage.sP.get_tensor(sPdS_layout)
        sdS = storage.sdS.get_tensor(sPdS_layout)
        sQpx = storage.sQpx.get_tensor(sQpx_layout)
        sKpx = storage.sKpx.get_tensor(sKV_layout)

        sQt = _transpose_view(sQ)
        sdOt = _transpose_view(sdO)
        sKt = _transpose_view(sK)
        sPt = _transpose_view(sP)
        sdSt = _transpose_view(sdS)

        zero = Float32(0.0).to(self._dtype)

        copy_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype
        )
        copy_atom_t = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self._dtype
        )
        r2s_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self._dtype, num_bits_per_copy=2 * self._dtype.width
        )

        thr_sdp = tiled_mma_sdp.get_slice(tidx)
        thr_dq = tiled_mma_dq.get_slice(tidx)
        thr_dkv = tiled_mma_dkv.get_slice(tidx)

        copy_A_sdp = cute.make_tiled_copy_A(copy_atom, tiled_mma_sdp).get_slice(tidx)
        copy_B_sdp = cute.make_tiled_copy_B(copy_atom, tiled_mma_sdp).get_slice(tidx)
        copy_A_dq = cute.make_tiled_copy_A(copy_atom, tiled_mma_dq).get_slice(tidx)
        copy_B_dq = cute.make_tiled_copy_B(copy_atom_t, tiled_mma_dq).get_slice(tidx)
        copy_A_dkv = cute.make_tiled_copy_A(copy_atom_t, tiled_mma_dkv).get_slice(tidx)
        copy_B_dkv = cute.make_tiled_copy_B(copy_atom_t, tiled_mma_dkv).get_slice(tidx)
        r2s_sdp = cute.make_tiled_copy_C(r2s_atom, tiled_mma_sdp).get_slice(tidx)

        tSrQ = thr_sdp.make_fragment_A(thr_sdp.partition_A(sQ))
        tSrK = thr_sdp.make_fragment_B(thr_sdp.partition_B(sK))
        tdPrdO = thr_sdp.make_fragment_A(thr_sdp.partition_A(sdO))
        tdPrV = thr_sdp.make_fragment_B(thr_sdp.partition_B(sV))
        tSsQ = copy_A_sdp.partition_S(sQ)
        tSsK = copy_B_sdp.partition_S(sK)
        tdPsdO = copy_A_sdp.partition_S(sdO)
        tdPsV = copy_B_sdp.partition_S(sV)
        tSrQ_copy = copy_A_sdp.retile(tSrQ)
        tSrK_copy = copy_B_sdp.retile(tSrK)
        tdPrdO_copy = copy_A_sdp.retile(tdPrdO)
        tdPrV_copy = copy_B_sdp.retile(tdPrV)

        tdQrdS = thr_dq.make_fragment_A(thr_dq.partition_A(sdS))
        tdQrK = thr_dq.make_fragment_B(thr_dq.partition_B(sKt))
        tdQsdS = copy_A_dq.partition_S(sdS)
        tdQsKt = copy_B_dq.partition_S(sKt)
        tdQrdS_copy = copy_A_dq.retile(tdQrdS)
        tdQrK_copy = copy_B_dq.retile(tdQrK)

        tdVrP = thr_dkv.make_fragment_A(thr_dkv.partition_A(sPt))
        tdVrdO = thr_dkv.make_fragment_B(thr_dkv.partition_B(sdOt))
        tdKrdS = thr_dkv.make_fragment_A(thr_dkv.partition_A(sdSt))
        tdKrQ = thr_dkv.make_fragment_B(thr_dkv.partition_B(sQt))
        tdVsPt = copy_A_dkv.partition_S(sPt)
        tdVsdOt = copy_B_dkv.partition_S(sdOt)
        tdKsdSt = copy_A_dkv.partition_S(sdSt)
        tdKsQt = copy_B_dkv.partition_S(sQt)
        tdVrP_copy = copy_A_dkv.retile(tdVrP)
        tdVrdO_copy = copy_B_dkv.retile(tdVrdO)
        tdKrdS_copy = copy_A_dkv.retile(tdKrdS)
        tdKrQ_copy = copy_B_dkv.retile(tdKrQ)

        tPsP = r2s_sdp.partition_D(sP)
        tdSsdS = r2s_sdp.partition_D(sdS)

        cS = cute.make_identity_tensor((self.rows_per_task, KEY_SLICE_SIZE))
        tScS_mn = _make_acc_tensor_mn_view(thr_sdp.partition_C(cS))
        cDq = cute.make_identity_tensor((self.rows_per_task, self.head_dim_padded))
        tDqcDq_mn = _make_acc_tensor_mn_view(thr_dq.partition_C(cDq))
        cKV = cute.make_identity_tensor((KEY_SLICE_SIZE, self.head_dim_padded))
        tKVcKV_mn = _make_acc_tensor_mn_view(thr_dkv.partition_C(cKV))

        for key_slice in cutlass.range_constexpr(KEY_SLICES):
            kv_linear = tidx
            while kv_linear < Int32(KEY_SLICE_SIZE * self.head_dim):
                row = kv_linear // Int32(self.head_dim)
                dim = kv_linear - row * Int32(self.head_dim)
                key_pos = key_block * Int32(BLOCK_SIZE) + Int32(key_slice * KEY_SLICE_SIZE) + row
                sK[row, dim] = k[batch, kv_head, key_pos, dim]
                sV[row, dim] = v[batch, kv_head, key_pos, dim]
                kv_linear += Int32(self.num_threads)

            if grad_kl_scale != Float32(0.0):
                px_k_linear = tidx
                while px_k_linear < Int32(KEY_SLICE_SIZE * self.head_dim):
                    row = px_k_linear // Int32(self.head_dim)
                    dim = px_k_linear - row * Int32(self.head_dim)
                    key_pos = (
                        key_block * Int32(BLOCK_SIZE) + Int32(key_slice * KEY_SLICE_SIZE) + row
                    )
                    sKpx[row, dim] = k_proxy[batch, proxy_kv_head, key_pos, dim]
                    px_k_linear += Int32(self.num_threads)

            cute.arch.sync_threads()

            q_linear = tidx
            while q_linear < Int32(self.rows_per_task * self.head_dim):
                row = q_linear // Int32(self.head_dim)
                dim = q_linear - row * Int32(self.head_dim)
                q_slot = row // Int32(self.main_per_proxy)
                head_offset = row - q_slot * Int32(self.main_per_proxy)
                if q_slot < query_count:
                    q_pos = task_qids[task_idx, qid_base + q_slot]
                    head = proxy_head * Int32(self.main_per_proxy) + head_offset
                    sQ[row, dim] = q[batch, head, q_pos, dim]
                    sdO[row, dim] = grad_o_main[batch, head, q_pos, dim]
                else:
                    sQ[row, dim] = zero
                    sdO[row, dim] = zero
                q_linear += Int32(self.num_threads)

            cute.arch.sync_threads()

            acc_shape_S = thr_sdp.partition_shape_C((self.rows_per_task, KEY_SLICE_SIZE))
            acc_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)
            acc_S.fill(0.0)
            for kk in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
                cute.copy(copy_A_sdp, tSsQ[None, None, kk], tSrQ_copy[None, None, kk])
                cute.copy(copy_B_sdp, tSsK[None, None, kk], tSrK_copy[None, None, kk])
                cute.gemm(tiled_mma_sdp, acc_S, tSrQ[None, None, kk], tSrK[None, None, kk], acc_S)

            acc_dP = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)
            acc_dP.fill(0.0)
            for kk in cutlass.range_constexpr(cute.size(tdPsdO.shape[2])):
                cute.copy(copy_A_sdp, tdPsdO[None, None, kk], tdPrdO_copy[None, None, kk])
                cute.copy(copy_B_sdp, tdPsV[None, None, kk], tdPrV_copy[None, None, kk])
                cute.gemm(
                    tiled_mma_sdp, acc_dP, tdPrdO[None, None, kk], tdPrV[None, None, kk], acc_dP
                )

            acc_S_mn = _make_acc_tensor_mn_view(acc_S)
            acc_dP_mn = _make_acc_tensor_mn_view(acc_dP)
            for rr in cutlass.range_constexpr(cute.size(acc_S_mn.shape[0])):
                row_m = tScS_mn[rr, 0][0]
                q_slot = row_m // Int32(self.main_per_proxy)
                head_offset = row_m - q_slot * Int32(self.main_per_proxy)
                row_is_valid = q_slot < query_count
                q_pos = Int32(0)
                lse = Float32(0.0)
                delta = Float32(0.0)
                if row_is_valid:
                    q_pos = task_qids[task_idx, qid_base + q_slot]
                    head = proxy_head * Int32(self.main_per_proxy) + head_offset
                    lse = lse_main[batch, head, q_pos]
                    delta = delta_main[batch, head, q_pos]

                for cc in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
                    col_n = tScS_mn[rr, cc][1]
                    key_pos = (
                        key_block * Int32(BLOCK_SIZE) + Int32(key_slice * KEY_SLICE_SIZE) + col_n
                    )
                    p = Float32(0.0)
                    if row_is_valid and key_pos <= q_pos:
                        p = cute.math.exp2(
                            acc_S_mn[rr, cc] * softmax_scale_log2 - lse * log2_e, fastmath=True
                        )
                    acc_S_mn[rr, cc] = p
                    acc_dP_mn[rr, cc] = p * (acc_dP_mn[rr, cc] - delta)

            rP = cute.make_fragment_like(acc_S, self._dtype)
            rdS = cute.make_fragment_like(acc_dP, self._dtype)
            rP.store(acc_S.load().to(self._dtype))
            rdS.store(acc_dP.load().to(self._dtype))
            tPrP = r2s_sdp.retile(rP)
            tdSrdS = r2s_sdp.retile(rdS)
            cute.copy(r2s_atom, tPrP, tPsP)
            cute.copy(r2s_atom, tdSrdS, tdSsdS)
            cute.arch.sync_threads()

            acc_shape_dV = thr_dkv.partition_shape_C((KEY_SLICE_SIZE, self.head_dim_padded))
            acc_dV = cute.make_rmem_tensor(acc_shape_dV, cutlass.Float32)
            acc_dV.fill(0.0)
            for kk in cutlass.range_constexpr(cute.size(tdVsPt.shape[2])):
                cute.copy(copy_A_dkv, tdVsPt[None, None, kk], tdVrP_copy[None, None, kk])
                cute.copy(copy_B_dkv, tdVsdOt[None, None, kk], tdVrdO_copy[None, None, kk])
                cute.gemm(
                    tiled_mma_dkv, acc_dV, tdVrP[None, None, kk], tdVrdO[None, None, kk], acc_dV
                )
            self._atomic_main_kv(
                acc_dV,
                tKVcKV_mn,
                batch,
                kv_head,
                key_block,
                Int32(key_slice),
                dv,
                softmax_scale,
                False,
            )

            acc_dK = cute.make_rmem_tensor(acc_shape_dV, cutlass.Float32)
            acc_dK.fill(0.0)
            for kk in cutlass.range_constexpr(cute.size(tdKsdSt.shape[2])):
                cute.copy(copy_A_dkv, tdKsdSt[None, None, kk], tdKrdS_copy[None, None, kk])
                cute.copy(copy_B_dkv, tdKsQt[None, None, kk], tdKrQ_copy[None, None, kk])
                cute.gemm(
                    tiled_mma_dkv, acc_dK, tdKrdS[None, None, kk], tdKrQ[None, None, kk], acc_dK
                )
            self._atomic_main_kv(
                acc_dK,
                tKVcKV_mn,
                batch,
                kv_head,
                key_block,
                Int32(key_slice),
                dk,
                softmax_scale,
                True,
            )

            acc_shape_dQ = thr_dq.partition_shape_C((self.rows_per_task, self.head_dim_padded))
            acc_dQ = cute.make_rmem_tensor(acc_shape_dQ, cutlass.Float32)
            acc_dQ.fill(0.0)
            for kk in cutlass.range_constexpr(cute.size(tdQsdS.shape[2])):
                cute.copy(copy_A_dq, tdQsdS[None, None, kk], tdQrdS_copy[None, None, kk])
                cute.copy(copy_B_dq, tdQsKt[None, None, kk], tdQrK_copy[None, None, kk])
                cute.gemm(
                    tiled_mma_dq, acc_dQ, tdQrdS[None, None, kk], tdQrK[None, None, kk], acc_dQ
                )
            self._atomic_main_dq(
                acc_dQ,
                tDqcDq_mn,
                batch,
                proxy_head,
                query_count,
                task_idx,
                qid_base,
                task_qids,
                dq,
                softmax_scale,
            )

            if grad_kl_scale != Float32(0.0):
                thr_px = tiled_mma_px.get_slice(tidx)
                copy_A_px = cute.make_tiled_copy_A(copy_atom, tiled_mma_px).get_slice(tidx)
                copy_B_px = cute.make_tiled_copy_B(copy_atom, tiled_mma_px).get_slice(tidx)
                copy_A_px_dkv = cute.make_tiled_copy_A(copy_atom_t, tiled_mma_px).get_slice(tidx)
                copy_B_px_dkv = cute.make_tiled_copy_B(copy_atom_t, tiled_mma_px).get_slice(tidx)
                r2s_px = cute.make_tiled_copy_C(r2s_atom, tiled_mma_px).get_slice(tidx)

                sdSpx = cute.make_tensor(sdS.iterator, sPdSpx_layout)
                sKpxt = _transpose_view(sKpx)
                sQpxt = _transpose_view(sQpx)
                sdSpxt = _transpose_view(sdSpx)

                tPxsQ = copy_A_px.partition_S(sQpx)
                tPxsK = copy_B_px.partition_S(sKpx)
                tPxrQ = thr_px.make_fragment_A(thr_px.partition_A(sQpx))
                tPxrK = thr_px.make_fragment_B(thr_px.partition_B(sKpx))
                tPxrQ_copy = copy_A_px.retile(tPxrQ)
                tPxrK_copy = copy_B_px.retile(tPxrK)

                tPxDqrdS = thr_px.make_fragment_A(thr_px.partition_A(sdSpx))
                tPxDqrK = thr_px.make_fragment_B(thr_px.partition_B(sKpxt))
                tPxDqsdS = copy_A_px.partition_S(sdSpx)
                tPxDqsKt = copy_B_px_dkv.partition_S(sKpxt)
                tPxDqrdS_copy = copy_A_px.retile(tPxDqrdS)
                tPxDqrK_copy = copy_B_px_dkv.retile(tPxDqrK)

                tPxDkrdS = thr_px.make_fragment_A(thr_px.partition_A(sdSpxt))
                tPxDkrQ = thr_px.make_fragment_B(thr_px.partition_B(sQpxt))
                tPxDksdS = copy_A_px_dkv.partition_S(sdSpxt)
                tPxDksQ = copy_B_px_dkv.partition_S(sQpxt)
                tPxDkrdS_copy = copy_A_px_dkv.retile(tPxDkrdS)
                tPxDkrQ_copy = copy_B_px_dkv.retile(tPxDkrQ)
                tPxsD = r2s_px.partition_D(sdSpx)

                cPxS = cute.make_identity_tensor((self.proxy_query_rows, KEY_SLICE_SIZE))
                tPxcS_mn = _make_acc_tensor_mn_view(thr_px.partition_C(cPxS))
                cPxDq = cute.make_identity_tensor((self.proxy_query_rows, self.head_dim_padded))
                tPxcDq_mn = _make_acc_tensor_mn_view(thr_px.partition_C(cPxDq))
                cPxDK = cute.make_identity_tensor((KEY_SLICE_SIZE, self.head_dim_padded))
                tPxcDK_mn = _make_acc_tensor_mn_view(thr_px.partition_C(cPxDK))

                px_q_linear = tidx
                while px_q_linear < Int32(self.proxy_query_rows * self.head_dim):
                    row = px_q_linear // Int32(self.head_dim)
                    dim = px_q_linear - row * Int32(self.head_dim)
                    if row < query_count:
                        q_pos = task_qids[task_idx, qid_base + row]
                        sQpx[row, dim] = q_proxy[batch, proxy_head, q_pos, dim]
                    else:
                        sQpx[row, dim] = zero
                    px_q_linear += Int32(self.num_threads)

                cute.arch.sync_threads()

                acc_shape_px = thr_px.partition_shape_C((self.proxy_query_rows, KEY_SLICE_SIZE))
                acc_Px = cute.make_rmem_tensor(acc_shape_px, cutlass.Float32)
                acc_Px.fill(0.0)
                for kk in cutlass.range_constexpr(cute.size(tPxsQ.shape[2])):
                    cute.copy(copy_A_px, tPxsQ[None, None, kk], tPxrQ_copy[None, None, kk])
                    cute.copy(copy_B_px, tPxsK[None, None, kk], tPxrK_copy[None, None, kk])
                    cute.gemm(
                        tiled_mma_px, acc_Px, tPxrQ[None, None, kk], tPxrK[None, None, kk], acc_Px
                    )

                acc_Px_mn = _make_acc_tensor_mn_view(acc_Px)
                for rr in cutlass.range_constexpr(cute.size(acc_Px_mn.shape[0])):
                    q_slot = tPxcS_mn[rr, 0][0]
                    row_is_valid = q_slot < query_count
                    q_pos = Int32(0)
                    lse_px = Float32(0.0)
                    if row_is_valid:
                        q_pos = task_qids[task_idx, qid_base + q_slot]
                        lse_px = lse_proxy[batch, proxy_head, q_pos]
                    for cc in cutlass.range_constexpr(cute.size(acc_Px_mn.shape[1])):
                        col_n = tPxcS_mn[rr, cc][1]
                        key_pos = (
                            key_block * Int32(BLOCK_SIZE)
                            + Int32(key_slice * KEY_SLICE_SIZE)
                            + col_n
                        )
                        # HERE IS THE KL LOSS SURROGATE
                        ds_px = Float32(0.0)
                        if row_is_valid and key_pos <= q_pos:
                            p_px = cute.math.exp2(
                                acc_Px_mn[rr, cc] * softmax_scale_log2 - lse_px * log2_e,
                                fastmath=True,
                            )
                            teacher = Float32(0.0)
                            for head_offset in cutlass.range_constexpr(self.main_per_proxy):
                                teacher = teacher + Float32(
                                    sP[
                                        q_slot * Int32(self.main_per_proxy) + Int32(head_offset),
                                        col_n,
                                    ]
                                )
                            teacher = teacher * Float32(1.0 / self.main_per_proxy)
                            ds_px = grad_kl_scale * (p_px - teacher)
                        acc_Px_mn[rr, cc] = ds_px

                rdSpx = cute.make_fragment_like(acc_Px, self._dtype)
                rdSpx.store(acc_Px.load().to(self._dtype))
                tPxrdS = r2s_px.retile(rdSpx)
                cute.copy(r2s_atom, tPxrdS, tPxsD)
                cute.arch.sync_threads()

                acc_shape_px_dq = thr_px.partition_shape_C(
                    (self.proxy_query_rows, self.head_dim_padded)
                )
                acc_dQpx = cute.make_rmem_tensor(acc_shape_px_dq, cutlass.Float32)
                acc_dQpx.fill(0.0)
                for kk in cutlass.range_constexpr(cute.size(tPxDqsdS.shape[2])):
                    cute.copy(copy_A_px, tPxDqsdS[None, None, kk], tPxDqrdS_copy[None, None, kk])
                    cute.copy(copy_B_px_dkv, tPxDqsKt[None, None, kk], tPxDqrK_copy[None, None, kk])
                    cute.gemm(
                        tiled_mma_px,
                        acc_dQpx,
                        tPxDqrdS[None, None, kk],
                        tPxDqrK[None, None, kk],
                        acc_dQpx,
                    )
                self._atomic_proxy_dq(
                    acc_dQpx,
                    tPxcDq_mn,
                    batch,
                    proxy_head,
                    query_count,
                    task_idx,
                    qid_base,
                    task_qids,
                    dq_proxy,
                    softmax_scale,
                )

                acc_shape_px_dk = thr_px.partition_shape_C((KEY_SLICE_SIZE, self.head_dim_padded))
                acc_dKpx = cute.make_rmem_tensor(acc_shape_px_dk, cutlass.Float32)
                acc_dKpx.fill(0.0)
                for kk in cutlass.range_constexpr(cute.size(tPxDksdS.shape[2])):
                    cute.copy(
                        copy_A_px_dkv, tPxDksdS[None, None, kk], tPxDkrdS_copy[None, None, kk]
                    )
                    cute.copy(copy_B_px_dkv, tPxDksQ[None, None, kk], tPxDkrQ_copy[None, None, kk])
                    cute.gemm(
                        tiled_mma_px,
                        acc_dKpx,
                        tPxDkrdS[None, None, kk],
                        tPxDkrQ[None, None, kk],
                        acc_dKpx,
                    )
                self._atomic_proxy_dk(
                    acc_dKpx,
                    tPxcDK_mn,
                    batch,
                    proxy_kv_head,
                    key_block,
                    Int32(key_slice),
                    dk_proxy,
                    softmax_scale,
                )

            cute.arch.sync_threads()

    @cute.jit
    def _atomic_main_dq(
        self,
        acc_dq: cute.Tensor,
        coord_mn: cute.Tensor,
        batch: Int32,
        proxy_head: Int32,
        query_count: Int32,
        task_idx: Int32,
        qid_base: Int32,
        task_qids: cute.Tensor,
        dq: cute.Tensor,
        softmax_scale: cutlass.Float32,
    ):
        acc_mn = _make_acc_tensor_mn_view(acc_dq)
        for rr in cutlass.range_constexpr(cute.size(acc_mn.shape[0])):
            row = coord_mn[rr, 0][0]
            q_slot = row // Int32(self.main_per_proxy)
            head_offset = row - q_slot * Int32(self.main_per_proxy)
            if q_slot < query_count:
                q_pos = task_qids[task_idx, qid_base + q_slot]
                head = proxy_head * Int32(self.main_per_proxy) + head_offset
                for cc in cutlass.range_constexpr(cute.size(acc_mn.shape[1])):
                    dim = coord_mn[rr, cc][1]
                    _atomic_add_fp32(
                        acc_mn[rr, cc] * softmax_scale, _elem_pointer(dq, (batch, head, q_pos, dim))
                    )

    @cute.jit
    def _atomic_main_kv(
        self,
        acc: cute.Tensor,
        coord_mn: cute.Tensor,
        batch: Int32,
        kv_head: Int32,
        key_block: Int32,
        key_slice: Int32,
        target: cute.Tensor,
        softmax_scale: cutlass.Float32,
        apply_scale: cutlass.Constexpr[bool],
    ):
        acc_mn = _make_acc_tensor_mn_view(acc)
        for rr in cutlass.range_constexpr(cute.size(acc_mn.shape[0])):
            key_row = coord_mn[rr, 0][0]
            key_pos = key_block * Int32(BLOCK_SIZE) + key_slice * Int32(KEY_SLICE_SIZE) + key_row
            for cc in cutlass.range_constexpr(cute.size(acc_mn.shape[1])):
                dim = coord_mn[rr, cc][1]
                value = acc_mn[rr, cc]
                if cutlass.const_expr(apply_scale):
                    value = value * softmax_scale
                _atomic_add_fp32(value, _elem_pointer(target, (batch, kv_head, key_pos, dim)))

    @cute.jit
    def _atomic_proxy_dq(
        self,
        acc_dq: cute.Tensor,
        coord_mn: cute.Tensor,
        batch: Int32,
        proxy_head: Int32,
        query_count: Int32,
        task_idx: Int32,
        qid_base: Int32,
        task_qids: cute.Tensor,
        dq_proxy: cute.Tensor,
        softmax_scale: cutlass.Float32,
    ):
        acc_mn = _make_acc_tensor_mn_view(acc_dq)
        for rr in cutlass.range_constexpr(cute.size(acc_mn.shape[0])):
            q_slot = coord_mn[rr, 0][0]
            if q_slot < query_count:
                q_pos = task_qids[task_idx, qid_base + q_slot]
                for cc in cutlass.range_constexpr(cute.size(acc_mn.shape[1])):
                    dim = coord_mn[rr, cc][1]
                    _atomic_add_fp32(
                        acc_mn[rr, cc] * softmax_scale,
                        _elem_pointer(dq_proxy, (batch, proxy_head, q_pos, dim)),
                    )

    @cute.jit
    def _atomic_proxy_dk(
        self,
        acc_dk: cute.Tensor,
        coord_mn: cute.Tensor,
        batch: Int32,
        proxy_kv_head: Int32,
        key_block: Int32,
        key_slice: Int32,
        dk_proxy: cute.Tensor,
        softmax_scale: cutlass.Float32,
    ):
        acc_mn = _make_acc_tensor_mn_view(acc_dk)
        for rr in cutlass.range_constexpr(cute.size(acc_mn.shape[0])):
            key_row = coord_mn[rr, 0][0]
            key_pos = key_block * Int32(BLOCK_SIZE) + key_slice * Int32(KEY_SLICE_SIZE) + key_row
            for cc in cutlass.range_constexpr(cute.size(acc_mn.shape[1])):
                dim = coord_mn[rr, cc][1]
                _atomic_add_fp32(
                    acc_mn[rr, cc] * softmax_scale,
                    _elem_pointer(dk_proxy, (batch, proxy_kv_head, key_pos, dim)),
                )


def _compile_fused_backward_kernel(
    batch: int,
    n_heads: int,
    n_kv_heads: int,
    n_proxy_heads: int,
    n_proxy_kv_heads: int,
    seq_len: int,
    head_dim: int,
    num_tasks: int,
    input_query_chunk: int,
    q_proxy: cute.Tensor,
    k_proxy: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    grad_o_main: cute.Tensor,
    lse_main: cute.Tensor,
    lse_proxy: cute.Tensor,
    delta_main: cute.Tensor,
    task_meta: cute.Tensor,
    task_qids: cute.Tensor,
    dq_proxy: cute.Tensor,
    dk_proxy: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    softmax_scale: float,
    grad_kl_scale: float,
    stream: cuda.CUstream,
):
    num_threads = 256
    main_per_proxy, query_chunk, rows_per_task, proxy_query_rows = _derive_head_tiling(
        n_heads, n_kv_heads, n_proxy_heads
    )
    key = (
        "fused",
        int(batch),
        int(n_heads),
        int(n_kv_heads),
        int(n_proxy_heads),
        int(n_proxy_kv_heads),
        int(main_per_proxy),
        int(query_chunk),
        int(rows_per_task),
        int(proxy_query_rows),
        int(seq_len),
        int(head_dim),
        int(num_tasks),
        int(input_query_chunk),
        int(num_threads),
        q_proxy.element_type,
        k_proxy.element_type,
        q.element_type,
        k.element_type,
        v.element_type,
        grad_o_main.element_type,
        lse_main.element_type,
        lse_proxy.element_type,
        delta_main.element_type,
        task_meta.element_type,
        task_qids.element_type,
        dq_proxy.element_type,
        dk_proxy.element_type,
        dq.element_type,
        dk.element_type,
        dv.element_type,
    )
    if key not in _COMPILE_CACHE:
        kernel = _MSAFusedBackwardMMAKernel(
            batch=batch,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            n_proxy_heads=n_proxy_heads,
            n_proxy_kv_heads=n_proxy_kv_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            num_tasks=num_tasks,
            input_query_chunk=input_query_chunk,
            num_threads=num_threads,
        )
        _COMPILE_CACHE[key] = cute.compile(
            kernel,
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
            dq_proxy,
            dk_proxy,
            dq,
            dk,
            dv,
            float(softmax_scale),
            float(grad_kl_scale),
            stream,
        )
    return _COMPILE_CACHE[key]


def _run_fused_backward_impl(
    q_proxy: torch.Tensor,
    k_proxy: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_o_main: torch.Tensor,
    lse_main: torch.Tensor,
    lse_proxy: torch.Tensor,
    delta_main: torch.Tensor,
    task_meta: torch.Tensor,
    task_qids: torch.Tensor,
    *,
    scale: float,
    grad_kl_scale: float,
    cast_outputs: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if q.device.type != "cuda":
        raise ValueError("CuTeDSL fused backward requires CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"fused backward supports fp16/bf16, got {q.dtype}")

    q_proxy_c = q_proxy.detach().contiguous()
    k_proxy_c = k_proxy.detach().contiguous()
    q_c = q.detach().contiguous()
    k_c = k.detach().contiguous()
    v_c = v.detach().contiguous()
    grad_o_c = grad_o_main.detach().to(dtype=q.dtype).contiguous()
    lse_main_c = lse_main.detach().to(torch.float32).contiguous()
    lse_proxy_c = lse_proxy.detach().to(torch.float32).contiguous()
    delta_main_c = delta_main.detach().to(torch.float32).contiguous()
    task_meta_c = task_meta.detach().to(torch.int32).contiguous()
    task_qids_c = task_qids.detach().to(torch.int32).contiguous()

    batch, n_heads, seq_len, head_dim = q_c.shape
    _, n_kv_heads, _, _ = k_c.shape
    _, n_proxy_heads, _, _ = q_proxy_c.shape
    _, n_proxy_kv_heads, _, _ = k_proxy_c.shape
    num_tasks = int(task_meta_c.shape[0])

    if head_dim != 128:
        raise NotImplementedError("fused backward supports only D=128")
    _main_per_proxy, query_chunk, _rows_per_task, _proxy_query_rows = _derive_head_tiling(
        n_heads, n_kv_heads, n_proxy_heads
    )
    input_query_chunk = int(task_qids_c.shape[1])
    if input_query_chunk < query_chunk or input_query_chunk % query_chunk != 0:
        raise ValueError(
            "task_qids width must be a positive multiple of backward "
            f"query_chunk={query_chunk}, got {input_query_chunk}"
        )

    dq_proxy = torch.zeros_like(q_proxy_c, dtype=torch.float32)
    dk_proxy = torch.zeros_like(k_proxy_c, dtype=torch.float32)
    dk = torch.zeros_like(k_c, dtype=torch.float32)
    dv = torch.zeros_like(v_c, dtype=torch.float32)
    dq = torch.zeros_like(q_c, dtype=torch.float32)

    q_proxy_t = _to_cute_tensor(q_proxy_c)
    k_proxy_t = _to_cute_tensor(k_proxy_c)
    q_t = _to_cute_tensor(q_c)
    k_t = _to_cute_tensor(k_c)
    v_t = _to_cute_tensor(v_c)
    grad_o_t = _to_cute_tensor(grad_o_c)
    lse_main_t = _to_cute_tensor(lse_main_c)
    lse_proxy_t = _to_cute_tensor(lse_proxy_c)
    delta_main_t = _to_cute_tensor(delta_main_c)
    task_meta_t = _to_cute_tensor(task_meta_c)
    task_qids_t = _to_cute_tensor(task_qids_c)
    dq_proxy_t = _to_cute_tensor(dq_proxy)
    dk_proxy_t = _to_cute_tensor(dk_proxy)
    dq_t = _to_cute_tensor(dq)
    dk_t = _to_cute_tensor(dk)
    dv_t = _to_cute_tensor(dv)
    stream = cuda.CUstream(torch.cuda.current_stream(q_c.device).cuda_stream)

    compiled = _compile_fused_backward_kernel(
        batch,
        n_heads,
        n_kv_heads,
        n_proxy_heads,
        n_proxy_kv_heads,
        seq_len,
        head_dim,
        num_tasks,
        input_query_chunk,
        q_proxy_t,
        k_proxy_t,
        q_t,
        k_t,
        v_t,
        grad_o_t,
        lse_main_t,
        lse_proxy_t,
        delta_main_t,
        task_meta_t,
        task_qids_t,
        dq_proxy_t,
        dk_proxy_t,
        dq_t,
        dk_t,
        dv_t,
        float(scale),
        float(grad_kl_scale),
        stream,
    )
    compiled(
        q_proxy_t,
        k_proxy_t,
        q_t,
        k_t,
        v_t,
        grad_o_t,
        lse_main_t,
        lse_proxy_t,
        delta_main_t,
        task_meta_t,
        task_qids_t,
        dq_proxy_t,
        dk_proxy_t,
        dq_t,
        dk_t,
        dv_t,
        float(scale),
        float(grad_kl_scale),
        stream,
    )

    if not cast_outputs:
        return dq_proxy, dk_proxy, dq, dk, dv

    return (
        dq_proxy.to(dtype=q_proxy.dtype),
        dk_proxy.to(dtype=k_proxy.dtype),
        dq.to(dtype=q.dtype),
        dk.to(dtype=k.dtype),
        dv.to(dtype=v.dtype),
    )

def run_fused_backward(
    q_proxy: torch.Tensor,
    k_proxy: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_o_main: torch.Tensor,
    lse_main: torch.Tensor,
    lse_proxy: torch.Tensor,
    delta_main: torch.Tensor,
    task_meta: torch.Tensor,
    task_qids: torch.Tensor,
    *,
    scale: float,
    grad_kl_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    return _run_fused_backward_impl(
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
        scale=scale,
        grad_kl_scale=grad_kl_scale,
    )
