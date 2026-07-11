import argparse
import os
import random
import copy
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from flash_msa.tests.testing_model_warmup import WarmupModel

DEVICE='cuda'
DTYPE=torch.bfloat16
ATOL=1e-2
RTOL=1e-2


def parse_args():
    parser = argparse.ArgumentParser(description="Compare eager and optimized MSA training.")
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
    return parser.parse_args()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def train(model, input_tensor, target_tensor, num_epochs, lr=0.01):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    final_loss = 0.0

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs, kl_loss = model(input_tensor)
        loss = criterion(outputs, target_tensor)
        loss += kl_loss
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        
    return model, final_loss

if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)

    num_steps = args.steps
    n_heads = args.n_heads
    n_kv_heads = args.n_kv_heads
    head_dim = args.head_dim
    n_proxy_heads = args.n_proxy_heads
    n_proxy_kv_heads = args.n_proxy_kv_heads
    top_k = args.top_k
    B, S = args.B, args.S
    E = n_heads * head_dim 

    eager_model = WarmupModel(n_heads, n_kv_heads, head_dim, n_proxy_heads, n_proxy_kv_heads, top_k, use_kernel=False).to(DTYPE).to(DEVICE)
    optimized_model = WarmupModel(n_heads, n_kv_heads, head_dim, n_proxy_heads, n_proxy_kv_heads, top_k, use_kernel=True).to(DTYPE).to(DEVICE)
    optimized_model.load_state_dict(copy.deepcopy(eager_model.state_dict()))

    input_data = torch.randn(B, S, E).to(DTYPE).to(DEVICE)
    output_data = torch.randn(B, S, E).to(DTYPE).to(DEVICE)

    input_data = F.rms_norm(input_data, normalized_shape=(E,))
    output_data = F.rms_norm(output_data, normalized_shape=(E,))

    eager_model, eager_loss = train(eager_model, input_data, output_data, num_steps)
    optimized_model, optimized_loss = train(optimized_model, input_data, output_data, num_steps)

    for (name_a, param_a), (name_b, param_b) in zip(eager_model.named_parameters(), optimized_model.named_parameters()):
        assert name_a == name_b
        assert torch.allclose(param_a, param_b, atol=ATOL, rtol=RTOL), name_a
