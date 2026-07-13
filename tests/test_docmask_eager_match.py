import argparse
import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from flash_msa.tests.testing_docmask_model import DocmaskModel


DEVICE = "cuda"
DTYPE = torch.bfloat16
ATOL = 1e-2
RTOL = 1e-2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare eager document-masked attention and optimized MSA training."
    )
    parser.add_argument("-B", type=int, default=1)
    parser.add_argument("-S", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--n-proxy-heads", type=int, default=4)
    parser.add_argument("--n-proxy-kv-heads", type=int, default=1)
    parser.add_argument(
        "--avg-doc-len-pct",
        type=float,
        default=25.0,
        help="Target average document length as a percentage of sequence length.",
    )
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_document_list(batch_size, seq_len, num_documents, device):
    """Create contiguous documents with independently randomized lengths per batch row."""
    document_list = torch.empty(batch_size, seq_len, dtype=torch.int32)
    for batch_idx in range(batch_size):
        boundaries = [0]
        boundaries.extend(sorted(random.sample(range(1, seq_len), num_documents - 1)))
        boundaries.append(seq_len)
        for document_idx, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            document_list[batch_idx, start:end] = document_idx
    return document_list.to(device)


def document_list_to_cu_seqlens(document_list):
    """Convert contiguous document runs to FA4-style flattened offsets."""
    starts = torch.zeros_like(document_list, dtype=torch.bool)
    starts[:, 0] = True
    starts[:, 1:] = document_list[:, 1:] != document_list[:, :-1]
    flat_starts = torch.nonzero(starts.reshape(-1), as_tuple=False).flatten()
    flat_end = torch.tensor(
        [document_list.numel()], dtype=torch.int64, device=document_list.device
    )
    return torch.cat((flat_starts, flat_end)).to(torch.int32)


def train(
    model,
    input_tensor,
    target_tensor,
    document_list,
    num_epochs,
    lr=0.01,
    *,
    cu_seqlens=None,
):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    final_loss = 0.0

    for _ in range(num_epochs):
        optimizer.zero_grad()
        outputs, kl_loss = model(
            input_tensor, document_list, cu_seqlens=cu_seqlens
        )
        loss = criterion(outputs, target_tensor)
        loss += kl_loss
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    return model, final_loss


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)

    if not 0.0 < args.avg_doc_len_pct <= 100.0:
        raise ValueError("--avg-doc-len-pct must be in the interval (0, 100]")

    num_steps = args.steps
    n_heads = args.n_heads
    n_kv_heads = args.n_kv_heads
    head_dim = args.head_dim
    n_proxy_heads = args.n_proxy_heads
    n_proxy_kv_heads = args.n_proxy_kv_heads
    top_k = args.top_k
    B, S = args.B, args.S
    E = n_heads * head_dim

    target_avg_doc_len = S * args.avg_doc_len_pct / 100.0
    num_documents = max(1, min(S, round(S / target_avg_doc_len)))
    document_list = make_document_list(B, S, num_documents, DEVICE)
    cu_seqlens = document_list_to_cu_seqlens(document_list)

    eager_model = DocmaskModel(
        n_heads,
        n_kv_heads,
        head_dim,
        n_proxy_heads,
        n_proxy_kv_heads,
        top_k,
        use_kernel=False,
    ).to(DTYPE).to(DEVICE)
    optimized_model = DocmaskModel(
        n_heads,
        n_kv_heads,
        head_dim,
        n_proxy_heads,
        n_proxy_kv_heads,
        top_k,
        use_kernel=True,
    ).to(DTYPE).to(DEVICE)
    optimized_model.load_state_dict(copy.deepcopy(eager_model.state_dict()))

    input_data = torch.randn(B, S, E).to(DTYPE).to(DEVICE)
    output_data = torch.randn(B, S, E).to(DTYPE).to(DEVICE)

    input_data = F.rms_norm(input_data, normalized_shape=(E,))
    output_data = F.rms_norm(output_data, normalized_shape=(E,))

    eager_model, eager_loss = train(
        eager_model,
        input_data,
        output_data,
        document_list,
        num_steps,
    )
    optimized_model, optimized_loss = train(
        optimized_model,
        input_data,
        output_data,
        document_list,
        num_steps,
        cu_seqlens=cu_seqlens,
    )

    for (name_a, param_a), (name_b, param_b) in zip(
        eager_model.named_parameters(), optimized_model.named_parameters()
    ):
        assert name_a == name_b
        assert torch.allclose(param_a, param_b, atol=ATOL, rtol=RTOL), name_a
