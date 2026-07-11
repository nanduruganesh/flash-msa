#!/usr/bin/env python3
"""Sweep Flash-MSA and FlashAttention training-kernel time and peak VRAM.

Flash-MSA includes proxy selection and its KL-loss path. FlashAttention is dense,
causal GQA over the same main Q/K/V tensors. Timings exclude tensor allocation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from flash_msa import flash_msa_func


GiB = 1024**3


def flash_attn_function() -> Callable:
    """Import FlashAttention 3 or the FlashAttention 4 CuTe-DSL package."""
    try:
        from flash_attn_interface import flash_attn_func
    except ImportError:
        from flash_attn.cute.interface import flash_attn_func
    return flash_attn_func


@dataclass
class Result:
    kernel: str
    batch_size: int
    sequence_length: int
    top_k: int
    forward_ms: float | None = None
    backward_ms: float | None = None
    forward_peak_gib: float | None = None
    forward_increment_gib: float | None = None
    backward_peak_gib: float | None = None
    backward_increment_gib: float | None = None
    peak_vram_percent: float | None = None
    status: str = "ok"


def parse_int_list(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=parse_int_list, default=[1, 2, 4])
    parser.add_argument("--sequence-lengths", type=parse_int_list, default=[1024, 2048, 4096])
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--n-proxy-heads", type=int, default=4)
    parser.add_argument("--n-proxy-kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=2048)
    parser.add_argument(
        "--top-ks",
        type=parse_int_list,
        help="Comma-separated top-k sweep; overrides --top-k when provided.",
    )
    parser.add_argument("--warmup", type=int, default=2, help="Untimed forward/backward iterations.")
    parser.add_argument("--repeats", type=int, default=5, help="Timed iterations per phase.")
    parser.add_argument(
        "--skip-memory-probes",
        action="store_true",
        help="Skip peak-memory probes; peak-memory fields are left empty.",
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--kernels", choices=("both", "flash-msa", "flash-attn"), default="both")
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--csv", type=Path, help="Optionally write machine-readable results.")
    return parser.parse_args()


def requested_top_ks(args: argparse.Namespace) -> list[int]:
    return args.top_ks if args.top_ks is not None else [args.top_k]


def validate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a CUDA GPU")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("--warmup must be nonnegative and --repeats must be positive")
    if args.n_heads % args.n_kv_heads:
        raise ValueError("--n-heads must be divisible by --n-kv-heads")
    if args.kernels != "flash-attn":
        top_ks = requested_top_ks(args)
        if args.head_dim != 128:
            raise ValueError("Flash-MSA currently requires --head-dim 128")
        if args.n_heads % args.n_proxy_heads:
            raise ValueError("--n-heads must be divisible by --n-proxy-heads")
        if args.n_proxy_heads % args.n_proxy_kv_heads:
            raise ValueError("--n-proxy-heads must be divisible by --n-proxy-kv-heads")
        if args.n_proxy_heads < args.n_kv_heads or args.n_proxy_heads % args.n_kv_heads:
            raise ValueError("Flash-MSA requires proxy heads >= and divisible by KV heads")
        if any(top_k % 128 for top_k in top_ks):
            raise ValueError("all top-k values must be divisible by 128")
        bad = [
            (length, top_k)
            for length in args.sequence_lengths
            for top_k in top_ks
            if length % 128 or top_k > length
        ]
        if bad:
            raise ValueError(
                "Flash-MSA sequence lengths must be divisible by 128 and >= top-k: "
                f"{bad}"
            )


def make_inputs(kernel: str, batch: int, sequence: int, args: argparse.Namespace, dtype: torch.dtype):
    def rand(heads: int) -> torch.Tensor:
        return torch.randn(
            batch, sequence, heads, args.head_dim, device="cuda", dtype=dtype, requires_grad=True
        )

    q, k, v = rand(args.n_heads), rand(args.n_kv_heads), rand(args.n_kv_heads)
    if kernel == "flash-attn":
        return q, k, v
    # Flash-MSA consumes head-major tensors.
    qp, kp = rand(args.n_proxy_heads), rand(args.n_proxy_kv_heads)
    return tuple(x.transpose(1, 2).contiguous().detach().requires_grad_(True) for x in (qp, kp, q, k, v))


def output_loss(
    kernel: str,
    inputs: tuple[torch.Tensor, ...],
    top_k: int,
    args: argparse.Namespace,
    fa_func: Callable,
):
    if kernel == "flash-msa":
        output, kl_loss = flash_msa_func(*inputs, top_k, args.head_dim**-0.5)
        return output, output.float().sum() + kl_loss.float()
    output = fa_func(*inputs, softmax_scale=args.head_dim**-0.5, causal=True)
    # FA4's CuTe interface returns ``(out, lse)`` even with return_lse=False;
    # FA3 returns the output tensor directly by default.
    if isinstance(output, tuple):
        output = output[0]
    return output, output.float().sum()


def clear_grads(inputs: tuple[torch.Tensor, ...]) -> None:
    for tensor in inputs:
        tensor.grad = None


def elapsed_ms(operation: Callable[[], None]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def benchmark_case(
    kernel: str,
    batch: int,
    sequence: int,
    top_k: int,
    args: argparse.Namespace,
    dtype: torch.dtype,
    fa_func: Callable,
    total_memory: int,
) -> Result:
    result = Result(kernel, batch, sequence, top_k)
    inputs = make_inputs(kernel, batch, sequence, args, dtype)

    for _ in range(args.warmup):
        _, loss = output_loss(kernel, inputs, top_k, args, fa_func)
        loss.backward()
        clear_grads(inputs)
    torch.cuda.synchronize()

    forward_times = []
    for _ in range(args.repeats):
        forward_times.append(
            elapsed_ms(lambda: output_loss(kernel, inputs, top_k, args, fa_func))
        )
    result.forward_ms = statistics.median(forward_times)

    backward_times = []
    for _ in range(args.repeats):
        _, loss = output_loss(kernel, inputs, top_k, args, fa_func)
        torch.cuda.synchronize()
        backward_times.append(elapsed_ms(loss.backward))
        clear_grads(inputs)
    result.backward_ms = statistics.median(backward_times)

    if not args.skip_memory_probes:
        torch.cuda.reset_peak_memory_stats()
        before_forward = torch.cuda.memory_allocated()
        _, loss = output_loss(kernel, inputs, top_k, args, fa_func)
        torch.cuda.synchronize()
        forward_peak = torch.cuda.max_memory_allocated()
        result.forward_peak_gib = forward_peak / GiB
        result.forward_increment_gib = (forward_peak - before_forward) / GiB

        torch.cuda.reset_peak_memory_stats()
        before_backward = torch.cuda.memory_allocated()
        loss.backward()
        torch.cuda.synchronize()
        backward_peak = torch.cuda.max_memory_allocated()
        result.backward_peak_gib = backward_peak / GiB
        result.backward_increment_gib = (backward_peak - before_backward) / GiB
        result.peak_vram_percent = 100 * max(forward_peak, backward_peak) / total_memory
    return result


def clean_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def print_results(results: list[Result]) -> None:
    headers = (
        "kernel",
        "B",
        "S",
        "top-k",
        "fwd ms",
        "bwd ms",
        "fwd peak GiB",
        "bwd peak GiB",
        "peak %",
        "status",
    )
    rows = [
        (
            r.kernel,
            str(r.batch_size),
            str(r.sequence_length),
            str(r.top_k),
            fmt(r.forward_ms),
            fmt(r.backward_ms),
            fmt(r.forward_peak_gib),
            fmt(r.backward_peak_gib),
            fmt(r.peak_vram_percent),
            r.status,
        )
        for r in results
    ]
    widths = [max(len(header), *(len(row[i]) for row in rows)) for i, header in enumerate(headers)]
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def main() -> None:
    args = parse_args()
    validate(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dtype = getattr(torch, args.dtype)
    top_ks = requested_top_ks(args)
    fa_func = flash_attn_function() if args.kernels != "flash-msa" else None
    kernels = ["flash-msa", "flash-attn"] if args.kernels == "both" else [args.kernels]
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    print(f"GPU: {properties.name} ({properties.total_memory / GiB:.1f} GiB); dtype={args.dtype}")
    print(f"Heads: Q={args.n_heads}, KV={args.n_kv_heads}, proxy-Q={args.n_proxy_heads}, "
          f"proxy-KV={args.n_proxy_kv_heads}, D={args.head_dim}; top-k={top_ks}")

    results: list[Result] = []
    for top_k in top_ks:
        for sequence in args.sequence_lengths:
            for batch in args.batch_sizes:
                for kernel in kernels:
                    clean_cuda()
                    print(
                        f"Running {kernel}: B={batch}, S={sequence}, top-k={top_k}",
                        flush=True,
                    )
                    try:
                        result = benchmark_case(
                            kernel,
                            batch,
                            sequence,
                            top_k,
                            args,
                            dtype,
                            fa_func,
                            properties.total_memory,
                        )
                    except torch.OutOfMemoryError as exc:
                        result = Result(
                            kernel,
                            batch,
                            sequence,
                            top_k,
                            status=f"OOM: {str(exc).splitlines()[0]}",
                        )
                    results.append(result)
    clean_cuda()
    print()
    print_results(results)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(asdict(results[0])))
            writer.writeheader()
            writer.writerows(asdict(result) for result in results)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
