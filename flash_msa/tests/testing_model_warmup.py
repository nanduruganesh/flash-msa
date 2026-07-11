import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_msa import flash_msa_warmup_func


class WarmupModel(nn.Module):
    def __init__(
        self, n_heads, n_kv_heads, head_dim, n_proxy_heads, n_proxy_kv_heads, top_k, use_kernel
    ):
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
        scaling = self.head_dim ** -0.5

        causal_mask = torch.triu(
            torch.ones(s, s, device=q_proxy.device, dtype=torch.bool),
            diagonal=1,
        )

        # Proxy full causal attention.
        k_proxy = k_proxy.repeat_interleave(self.num_proxy_groups, dim=1)
        proxy_scores = (q_proxy @ k_proxy.transpose(-2, -1)) * scaling
        proxy_scores = proxy_scores.masked_fill(causal_mask, float("-inf"))

        # Main full causal attention.
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        attn_scores = (q @ k.transpose(-2, -1)) * scaling
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
        attn_probs = F.softmax(attn_scores, dim=-1)

        attn_out = (attn_probs @ v).transpose(1, 2).reshape(b, s, -1)

        main_attn_kl_target = attn_probs.view(
            b,
            hp,
            self.num_main_per_proxy,
            s,
            s,
        ).mean(dim=2)

        proxy_logprobs = F.log_softmax(proxy_scores, dim=-1).masked_fill(causal_mask, 0.0)
        main_attn_kl_target = main_attn_kl_target.masked_fill(causal_mask, 0.0)

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
            attn_out, kl_loss = flash_msa_warmup_func(
                q_proxy, k_proxy, q, k, v, self.top_k, self.head_dim ** -0.5
            )
        else:
            attn_out, kl_loss = self._attention_eager(q_proxy, k_proxy, q, k, v)
        return attn_out, kl_loss
