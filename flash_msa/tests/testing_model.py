import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_msa import flash_msa_func

class Model(nn.Module):
    def __init__(self, n_heads, n_kv_heads, head_dim, n_proxy_heads, n_proxy_kv_heads, top_k, use_kernel):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        assert n_proxy_heads % n_proxy_kv_heads == 0
        assert n_heads % n_proxy_heads == 0
        assert head_dim % 2 == 0


        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.num_groups = n_heads // n_kv_heads
        d_model = n_heads * head_dim
        
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)

        self.n_proxy_heads = n_proxy_heads
        self.n_proxy_kv_heads = n_proxy_kv_heads
        self.num_proxy_groups = n_proxy_heads // n_proxy_kv_heads
        self.num_main_per_proxy = n_heads // n_proxy_heads

        self.q_proxy = nn.Linear(d_model, n_proxy_heads * head_dim, bias=False)
        self.k_proxy = nn.Linear(d_model, n_proxy_kv_heads * head_dim, bias=False)

        self.top_k = top_k
        self.block_size = 128
        self.use_kernel = use_kernel
        self.kl_criterion = nn.KLDivLoss(reduction='batchmean')

    def apply_rope(self, x):
        # x shape: (B, S, H, D)
        b, s, h, d = x.shape
        assert d % 2 == 0

        device = x.device
        orig_dtype = x.dtype

        t = torch.arange(s, device=device, dtype=torch.float32).view(1, s, 1, 1)
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, d, 2, device=device, dtype=torch.float32) / d)
        )
        freqs = t * inv_freq.view(1, 1, 1, -1)

        x_even = x[..., 0::2].float()
        x_odd = x[..., 1::2].float()

        y_even = x_even * freqs.cos() - x_odd * freqs.sin()
        y_odd = x_even * freqs.sin() + x_odd * freqs.cos()

        return torch.stack((y_even, y_odd), dim=-1).flatten(-2).to(orig_dtype)

    def _attention_eager(self, q_proxy, k_proxy, q, k, v):
        b, hp, s, _ = q_proxy.shape

        assert s % self.block_size == 0
        assert self.top_k % self.block_size == 0

        num_blocks = s // self.block_size
        top_k_blocks = self.top_k // self.block_size
        top_k_tokens = top_k_blocks * self.block_size

        assert 1 <= top_k_blocks <= num_blocks

        scaling = self.head_dim ** -0.5

        # Proxy QK.
        k_proxy = k_proxy.repeat_interleave(self.num_proxy_groups, dim=1)
        proxy_scores = (q_proxy @ k_proxy.transpose(-2, -1)) * scaling

        proxy_mask = torch.triu(
            torch.ones(s, s, device=q_proxy.device, dtype=torch.bool),
            diagonal=1,
        )
        proxy_scores = proxy_scores.masked_fill(proxy_mask, float("-inf"))

        # Block scores from max-pooled token scores.
        block_scores = proxy_scores.view(
            b, hp, s, num_blocks, self.block_size
        ).amax(dim=-1)

        # Force local block before top-k so the final selected set is fixed-size.
        seq_indices = torch.arange(s, device=q_proxy.device)
        local_block_indices = seq_indices // self.block_size
        local_idx = local_block_indices.view(1, 1, s, 1).expand(b, hp, s, 1)
        block_scores = block_scores.scatter(3, local_idx, torch.inf)

        _, block_indices = block_scores.topk(top_k_blocks, dim=-1)

        block_mask = torch.zeros(
            (b, hp, s, num_blocks),
            dtype=torch.bool,
            device=q_proxy.device,
        )
        block_mask.scatter_(3, block_indices, True)

        token_mask = (
            block_mask
            .unsqueeze(-1)
            .expand(b, hp, s, num_blocks, self.block_size)
            .reshape(b, hp, s, s)
        )

        # Expand proxy-head mask to main-query-head mask.
        mask = (~token_mask.tril()).repeat_interleave(
            self.num_main_per_proxy,
            dim=1,
        )

        # Main sparse attention.
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        attn_scores = (q @ k.transpose(-2, -1)) * scaling
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        attn_probs = F.softmax(attn_scores, dim=-1)

        attn_out = (attn_probs @ v).transpose(1, 2).reshape(b, s, -1)

        # KL loss over selected token set.
        block_offsets = torch.arange(
            self.block_size,
            device=q_proxy.device,
        ).view(1, 1, 1, 1, self.block_size)

        selected_token_indices = (
            block_indices.unsqueeze(-1) * self.block_size + block_offsets
        ).reshape(b, hp, s, top_k_tokens)

        selected_proxy_scores = proxy_scores.gather(3, selected_token_indices)

        expanded_indices = selected_token_indices.repeat_interleave(
            self.num_main_per_proxy,
            dim=1,
        )
        selected_main_probs = attn_probs.gather(3, expanded_indices)

        main_attn_kl_target = selected_main_probs.view(
            b,
            hp,
            self.num_main_per_proxy,
            s,
            top_k_tokens,
        ).mean(dim=2)

        valid = selected_token_indices <= seq_indices.view(1, 1, s, 1)

        proxy_logprobs = F.log_softmax(
            selected_proxy_scores.masked_fill(~valid, float("-inf")),
            dim=-1,
        ).masked_fill(~valid, 0.0)

        main_attn_kl_target = main_attn_kl_target.masked_fill(~valid, 0.0)

        kl_loss = F.kl_div(
            input=proxy_logprobs,
            target=main_attn_kl_target.detach(),
            reduction="none",
        ).sum(dim=-1).mean()

        return attn_out, kl_loss

    def forward(self, hidden_states):
        b, s, _ = hidden_states.shape
        q_proxy = self.q_proxy(hidden_states).view(b, s, self.n_proxy_heads, self.head_dim)
        k_proxy = self.k_proxy(hidden_states).view(b, s, self.n_proxy_kv_heads, self.head_dim)
        q_proxy, k_proxy = self.apply_rope(q_proxy), self.apply_rope(k_proxy)
        q_proxy = q_proxy.transpose(1, 2)
        k_proxy = k_proxy.transpose(1, 2)

        q = self.q_proj(hidden_states).view(b, s, self.n_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(b, s, self.n_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(b, s, self.n_kv_heads, self.head_dim)
        q, k = self.apply_rope(q), self.apply_rope(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # QKV, qk_proxy: (B, H, S, D)
        # indexer_weights: (B, H, S)
        if self.use_kernel:
            attn_out, kl_loss = flash_msa_func(
                q_proxy, k_proxy, q, k, v,
                self.top_k, self.head_dim ** -0.5)
        else:
            attn_out, kl_loss = self._attention_eager(q_proxy, k_proxy, q, k, v)
        return attn_out, kl_loss
