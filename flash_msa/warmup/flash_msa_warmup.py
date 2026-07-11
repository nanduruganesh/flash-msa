"""Python autograd boundary for warmup MSA attention."""

import torch


def _validate_inputs(
    q_proxy: torch.Tensor,
    k_proxy: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
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
        raise ValueError(
            "q_proxy, k_proxy, q, k, and v must agree on batch, sequence, "
            "head dimension, and KV head count"
        )
    if head_dim != 128:
        raise NotImplementedError("warmup MSA kernel requires D=128")
    if n_proxy_heads % n_proxy_kv_heads != 0:
        raise ValueError("n_proxy_heads must be divisible by n_proxy_kv_heads")
    if n_heads % n_kv_heads != 0:
        raise ValueError("n_heads must be divisible by n_kv_heads")
    if n_heads % n_proxy_heads != 0:
        raise ValueError("n_heads must be divisible by n_proxy_heads")
    if n_proxy_heads < n_kv_heads or n_proxy_heads % n_kv_heads != 0:
        raise NotImplementedError(
            "warmup MSA kernel requires n_proxy_heads >= n_kv_heads and divisibility"
        )


class _WarmupSparseAttentionFunction(torch.autograd.Function):
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
        _ = int(top_k)
        _validate_inputs(q_proxy, k_proxy, q, k, v)
        if q.device.type != "cuda":
            raise RuntimeError("warmup MSA forward requires CUDA tensors")

        from flash_msa.warmup.msa_forward_cutedsl_warmup import run_main_forward

        o_main, lse_main, kl_loss = run_main_forward(q, k, v, scale=float(scale))
        out = o_main.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)

        ctx.save_for_backward(q_proxy, k_proxy, q, k, v, lse_main, o_main)
        ctx.scale = float(scale)
        return out, kl_loss

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor | None, grad_kl: torch.Tensor | None):
        q_proxy, k_proxy, q, k, v, lse_main, o_main = ctx.saved_tensors

        from flash_msa.warmup.msa_backward_cutedsl_warmup import run_warmup_backward

        dq_proxy, dk_proxy, dq, dk, dv = run_warmup_backward(
            q_proxy,
            k_proxy,
            q,
            k,
            v,
            lse_main,
            o_main,
            grad_out,
            grad_kl,
            scale=ctx.scale,
        )
        return dq_proxy, dk_proxy, dq, dk, dv, None, None


def sparse_attention_warmup(
    q_proxy: torch.Tensor,
    k_proxy: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    top_k: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute dense causal warmup attention and proxy KL gradients."""

    return _WarmupSparseAttentionFunction.apply(
        q_proxy, k_proxy, q, k, v, int(top_k), float(scale)
    )
