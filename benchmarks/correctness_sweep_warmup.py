#!/usr/bin/env python3
"""Sweep eager-vs-kernel Flash-MSA forward and backward correctness."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from flash_msa.tests.testing_model_warmup import WarmupModel as Model


WEIGHT_NAMES = ("q_proj", "k_proj", "v_proj", "q_proxy", "k_proxy")


@dataclass
class Result:
    batch_size: int
    sequence_length: int
    n_heads: int
    top_k: int
    output_cosine: float | None = None
    q_proj_grad_cosine: float | None = None
    k_proj_grad_cosine: float | None = None
    v_proj_grad_cosine: float | None = None
    q_proxy_grad_cosine: float | None = None
    k_proxy_grad_cosine: float | None = None
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
    parser.add_argument("--sequence-lengths", type=parse_int_list, default=[4096, 8192])
    parser.add_argument(
        "--n-heads",
        type=parse_int_list,
        default=[16],
        help="Comma-separated query-head counts to sweep.",
    )
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
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Input/target pairs whose cosine similarities are averaged.",
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--csv", type=Path, help="Optionally write machine-readable results.")
    return parser.parse_args()


def requested_top_ks(args: argparse.Namespace) -> list[int]:
    return args.top_ks if args.top_ks is not None else [args.top_k]


def validate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this correctness sweep requires a CUDA GPU")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.head_dim != 128:
        raise ValueError("Flash-MSA currently requires --head-dim 128")
    if any(n_heads % args.n_kv_heads for n_heads in args.n_heads):
        raise ValueError("all --n-heads values must be divisible by --n-kv-heads")
    if any(n_heads % args.n_proxy_heads for n_heads in args.n_heads):
        raise ValueError("all --n-heads values must be divisible by --n-proxy-heads")
    if args.n_proxy_heads % args.n_proxy_kv_heads:
        raise ValueError("--n-proxy-heads must be divisible by --n-proxy-kv-heads")
    if args.n_proxy_heads < args.n_kv_heads or args.n_proxy_heads % args.n_kv_heads:
        raise ValueError("Flash-MSA requires proxy heads >= and divisible by KV heads")

    top_ks = requested_top_ks(args)
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


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return F.cosine_similarity(
        left.detach().float().reshape(1, -1),
        right.detach().float().reshape(1, -1),
    ).item()


def make_model(
    args: argparse.Namespace,
    n_heads: int,
    top_k: int,
    use_kernel: bool,
) -> Model:
    return Model(
        n_heads,
        args.n_kv_heads,
        args.head_dim,
        args.n_proxy_heads,
        args.n_proxy_kv_heads,
        top_k,
        use_kernel=use_kernel,
    ).to(device="cuda", dtype=getattr(torch, args.dtype))


def make_pair(
    batch: int,
    sequence: int,
    embedding_dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (batch, sequence, embedding_dim)
    input_data = torch.randn(shape, device="cuda", dtype=dtype)
    target = torch.randn(shape, device="cuda", dtype=dtype)
    return (
        F.rms_norm(input_data, normalized_shape=(embedding_dim,)),
        F.rms_norm(target, normalized_shape=(embedding_dim,)),
    )


def compare_case(
    batch: int,
    sequence: int,
    n_heads: int,
    top_k: int,
    args: argparse.Namespace,
) -> Result:
    eager_model = make_model(args, n_heads, top_k, use_kernel=False)
    kernel_model = make_model(args, n_heads, top_k, use_kernel=True)
    kernel_model.load_state_dict(copy.deepcopy(eager_model.state_dict()))
    eager_model.train()
    kernel_model.train()

    dtype = getattr(torch, args.dtype)
    embedding_dim = n_heads * args.head_dim
    pairs = [
        make_pair(batch, sequence, embedding_dim, dtype)
        for _ in range(args.repeats)
    ]
    output_cosines: list[float] = []
    gradient_cosines: dict[str, list[float]] = {name: [] for name in WEIGHT_NAMES}

    for input_data, target in pairs:
        eager_model.zero_grad(set_to_none=True)
        kernel_model.zero_grad(set_to_none=True)

        eager_output, eager_kl_loss = eager_model(input_data)
        eager_output_for_comparison = eager_output.detach()
        (F.mse_loss(eager_output, target) + eager_kl_loss).backward()

        kernel_output, kernel_kl_loss = kernel_model(input_data)
        output_cosines.append(cosine(eager_output_for_comparison, kernel_output))
        (F.mse_loss(kernel_output, target) + kernel_kl_loss).backward()

        for name in WEIGHT_NAMES:
            eager_grad = getattr(eager_model, name).weight.grad
            kernel_grad = getattr(kernel_model, name).weight.grad
            if eager_grad is None or kernel_grad is None:
                raise RuntimeError(f"missing gradient for {name}.weight")
            gradient_cosines[name].append(cosine(eager_grad, kernel_grad))

    return Result(
        batch_size=batch,
        sequence_length=sequence,
        n_heads=n_heads,
        top_k=top_k,
        output_cosine=statistics.fmean(output_cosines),
        **{
            f"{name}_grad_cosine": statistics.fmean(values)
            for name, values in gradient_cosines.items()
        },
    )


def clean_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.6f}"


def print_results(results: list[Result]) -> None:
    headers = (
        "B",
        "S",
        "H",
        "top-k",
        "output",
        "q_proj grad",
        "k_proj grad",
        "v_proj grad",
        "q_proxy grad",
        "k_proxy grad",
        "status",
    )
    rows = [
        (
            str(result.batch_size),
            str(result.sequence_length),
            str(result.n_heads),
            str(result.top_k),
            fmt(result.output_cosine),
            fmt(result.q_proj_grad_cosine),
            fmt(result.k_proj_grad_cosine),
            fmt(result.v_proj_grad_cosine),
            fmt(result.q_proxy_grad_cosine),
            fmt(result.k_proxy_grad_cosine),
            result.status,
        )
        for result in results
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
    top_ks = requested_top_ks(args)

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    print(f"GPU: {properties.name}; dtype={args.dtype}; repeats={args.repeats}")
    print(
        f"Heads: Q={args.n_heads}, KV={args.n_kv_heads}, "
        f"proxy-Q={args.n_proxy_heads}, proxy-KV={args.n_proxy_kv_heads}, "
        f"D={args.head_dim}; top-k={top_ks}"
    )

    results: list[Result] = []
    for n_heads in args.n_heads:
        for top_k in top_ks:
            for sequence in args.sequence_lengths:
                for batch in args.batch_sizes:
                    clean_cuda()
                    print(
                        f"Running B={batch}, S={sequence}, H={n_heads}, top-k={top_k}",
                        flush=True,
                    )
                    try:
                        result = compare_case(batch, sequence, n_heads, top_k, args)
                    except torch.OutOfMemoryError as exc:
                        result = Result(
                            batch,
                            sequence,
                            n_heads,
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
