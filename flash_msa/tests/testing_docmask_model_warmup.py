import torch
import torch.nn.functional as F

from flash_msa import flash_msa_warmup_func
from flash_msa.tests.testing_model_warmup import WarmupModel


class DocmaskWarmupModel(WarmupModel):
    @staticmethod
    def _document_runs(document_list):
        document_starts = torch.zeros_like(document_list, dtype=torch.bool)
        document_starts[:, 0] = True
        document_starts[:, 1:] = document_list[:, 1:] != document_list[:, :-1]
        return document_starts.to(torch.int32).cumsum(dim=1) - 1

    @classmethod
    def _document_positions(cls, document_list):
        batch, seq_len = document_list.shape
        token_positions = torch.arange(
            seq_len, device=document_list.device
        ).expand(batch, -1)
        document_runs = cls._document_runs(document_list)
        document_starts = torch.zeros_like(document_list, dtype=torch.bool)
        document_starts[:, 0] = True
        document_starts[:, 1:] = (
            document_runs[:, 1:] != document_runs[:, :-1]
        )
        start_positions = torch.where(document_starts, token_positions, 0)
        latest_start = start_positions.cummax(dim=1).values
        return token_positions - latest_start

    def apply_document_rope(self, x, document_list):
        _, _, _, head_dim = x.shape
        positions = self._document_positions(document_list).to(torch.float32)
        positions = positions[:, :, None, None]
        inv_freq = 1.0 / (
            10000
            ** (
                torch.arange(
                    0,
                    head_dim,
                    2,
                    device=x.device,
                    dtype=torch.float32,
                )
                / head_dim
            )
        )
        freqs = positions * inv_freq.view(1, 1, 1, -1)

        orig_dtype = x.dtype
        x_even = x[..., 0::2].float()
        x_odd = x[..., 1::2].float()
        y_even = x_even * freqs.cos() - x_odd * freqs.sin()
        y_odd = x_even * freqs.sin() + x_odd * freqs.cos()
        return torch.stack((y_even, y_odd), dim=-1).flatten(-2).to(orig_dtype)

    def _attention_eager(self, q_proxy, k_proxy, q, k, v, document_list):
        batch, n_proxy_heads, seq_len, _ = q_proxy.shape
        scaling = self.head_dim**-0.5
        document_runs = self._document_runs(document_list)
        same_document = (
            document_runs.unsqueeze(-1) == document_runs.unsqueeze(-2)
        )
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=q_proxy.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        mask = causal_mask.unsqueeze(0) | ~same_document

        k_proxy = k_proxy.repeat_interleave(self.num_proxy_groups, dim=1)
        proxy_scores = (q_proxy @ k_proxy.transpose(-2, -1)) * scaling
        proxy_scores = proxy_scores.masked_fill(
            mask.unsqueeze(1), float("-inf")
        )

        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)
        attn_scores = (q @ k.transpose(-2, -1)) * scaling
        attn_scores = attn_scores.masked_fill(
            mask.unsqueeze(1), float("-inf")
        )
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_out = (attn_probs @ v).transpose(1, 2).reshape(
            batch, seq_len, -1
        )

        main_attn_kl_target = attn_probs.view(
            batch,
            n_proxy_heads,
            self.num_main_per_proxy,
            seq_len,
            seq_len,
        ).mean(dim=2)
        proxy_logprobs = F.log_softmax(proxy_scores, dim=-1).masked_fill(
            mask.unsqueeze(1), 0.0
        )
        main_attn_kl_target = main_attn_kl_target.masked_fill(
            mask.unsqueeze(1), 0.0
        )
        kl_loss = F.kl_div(
            input=proxy_logprobs,
            target=main_attn_kl_target.detach(),
            reduction="none",
        ).sum(dim=-1).mean()
        return attn_out, kl_loss

    def forward(self, hidden_states, document_list, *, cu_seqlens=None):
        batch, seq_len, _ = hidden_states.shape
        if document_list.shape != (batch, seq_len):
            raise ValueError(
                f"document_list must have shape {(batch, seq_len)}, got "
                f"{tuple(document_list.shape)}"
            )

        q_proxy = self.q_proxy(hidden_states).view(
            batch, seq_len, self.n_proxy_heads, self.head_dim
        )
        k_proxy = self.k_proxy(hidden_states).view(
            batch, seq_len, self.n_proxy_kv_heads, self.head_dim
        )
        q_proxy = self.apply_document_rope(q_proxy, document_list)
        k_proxy = self.apply_document_rope(k_proxy, document_list)
        q_proxy = q_proxy.transpose(1, 2)
        k_proxy = k_proxy.transpose(1, 2)

        q = self.q_proj(hidden_states).view(
            batch, seq_len, self.n_heads, self.head_dim
        )
        k = self.k_proj(hidden_states).view(
            batch, seq_len, self.n_kv_heads, self.head_dim
        )
        v = self.v_proj(hidden_states).view(
            batch, seq_len, self.n_kv_heads, self.head_dim
        )
        q = self.apply_document_rope(q, document_list)
        k = self.apply_document_rope(k, document_list)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_kernel:
            if cu_seqlens is None:
                raise ValueError("optimized document warmup requires cu_seqlens")
            return flash_msa_warmup_func(
                q_proxy,
                k_proxy,
                q,
                k,
                v,
                self.top_k,
                self.head_dim**-0.5,
                cu_seqlens=cu_seqlens,
            )
        return self._attention_eager(
            q_proxy, k_proxy, q, k, v, document_list
        )
