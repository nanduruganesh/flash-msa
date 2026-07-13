import torch
import torch.nn.functional as F

from flash_msa import flash_msa_func
from flash_msa.tests.testing_model import Model


class DocmaskModel(Model):
    @staticmethod
    def _document_runs(document_list):
        document_starts = torch.zeros_like(document_list, dtype=torch.bool)
        document_starts[:, 0] = True
        document_starts[:, 1:] = document_list[:, 1:] != document_list[:, :-1]
        return document_starts.to(torch.int32).cumsum(dim=1) - 1

    @classmethod
    def _document_positions(cls, document_list):
        batch, seq_len = document_list.shape
        token_positions = torch.arange(seq_len, device=document_list.device).expand(batch, -1)
        document_runs = cls._document_runs(document_list)
        document_starts = torch.zeros_like(document_list, dtype=torch.bool)
        document_starts[:, 0] = True
        document_starts[:, 1:] = document_runs[:, 1:] != document_runs[:, :-1]
        start_positions = torch.where(document_starts, token_positions, 0)
        latest_start = start_positions.cummax(dim=1).values
        return token_positions - latest_start

    def apply_document_rope(self, x, document_list):
        _, _, _, d = x.shape
        positions = self._document_positions(document_list).to(torch.float32)
        positions = positions[:, :, None, None]
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
        )
        freqs = positions * inv_freq.view(1, 1, 1, -1)

        orig_dtype = x.dtype
        x_even = x[..., 0::2].float()
        x_odd = x[..., 1::2].float()
        y_even = x_even * freqs.cos() - x_odd * freqs.sin()
        y_odd = x_even * freqs.sin() + x_odd * freqs.cos()
        return torch.stack((y_even, y_odd), dim=-1).flatten(-2).to(orig_dtype)

    def _attention_eager(self, q_proxy, k_proxy, q, k, v, document_list):
        b, hp, s, _ = q_proxy.shape

        assert s % self.block_size == 0
        assert self.top_k % self.block_size == 0
        if document_list.shape != (b, s):
            raise ValueError(
                f"document_list must have shape {(b, s)}, got {tuple(document_list.shape)}"
            )
        if document_list.device != q_proxy.device:
            raise ValueError("document_list must be on the same device as the model inputs")

        num_blocks = s // self.block_size
        top_k_blocks = self.top_k // self.block_size
        top_k_tokens = top_k_blocks * self.block_size

        assert 1 <= top_k_blocks <= num_blocks

        scaling = self.head_dim ** -0.5
        seq_indices = torch.arange(s, device=q_proxy.device)
        document_runs = self._document_runs(document_list)
        same_document = document_runs.unsqueeze(-1) == document_runs.unsqueeze(-2)

        # Proxy QK.
        k_proxy = k_proxy.repeat_interleave(self.num_proxy_groups, dim=1)
        proxy_scores = (q_proxy @ k_proxy.transpose(-2, -1)) * scaling

        causal_mask = torch.triu(
            torch.ones(s, s, device=q_proxy.device, dtype=torch.bool),
            diagonal=1,
        )
        proxy_mask = causal_mask.unsqueeze(0) | ~same_document
        proxy_scores = proxy_scores.masked_fill(proxy_mask.unsqueeze(1), float("-inf"))

        # Block scores from max-pooled token scores.
        block_scores = proxy_scores.view(
            b, hp, s, num_blocks, self.block_size
        ).amax(dim=-1)

        # Force local block before top-k so the final selected set is fixed-size.
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

        # Expand the proxy-head mask to main query heads, retaining only causal
        # tokens from the same document.
        allowed_tokens = token_mask.tril() & same_document.unsqueeze(1)
        mask = (~allowed_tokens).repeat_interleave(
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

        selected_document_ids = (
            document_runs[:, None, None, :]
            .expand(b, hp, s, s)
            .gather(3, selected_token_indices)
        )
        valid = (
            selected_token_indices <= seq_indices.view(1, 1, s, 1)
        ) & (
            selected_document_ids == document_runs[:, None, :, None]
        )

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

    def forward(self, hidden_states, document_list, *, cu_seqlens=None):
        b, s, _ = hidden_states.shape
        if document_list.shape != (b, s):
            raise ValueError(
                f"document_list must have shape {(b, s)}, got {tuple(document_list.shape)}"
            )

        q_proxy = self.q_proxy(hidden_states).view(
            b, s, self.n_proxy_heads, self.head_dim
        )
        k_proxy = self.k_proxy(hidden_states).view(
            b, s, self.n_proxy_kv_heads, self.head_dim
        )
        q_proxy = self.apply_document_rope(q_proxy, document_list)
        k_proxy = self.apply_document_rope(k_proxy, document_list)
        q_proxy = q_proxy.transpose(1, 2)
        k_proxy = k_proxy.transpose(1, 2)

        q = self.q_proj(hidden_states).view(b, s, self.n_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(b, s, self.n_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(b, s, self.n_kv_heads, self.head_dim)
        q = self.apply_document_rope(q, document_list)
        k = self.apply_document_rope(k, document_list)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_kernel:
            kernel_kwargs = (
                {"cu_seqlens": cu_seqlens}
                if cu_seqlens is not None
                else {"document_list": document_list}
            )
            attn_out, kl_loss = flash_msa_func(
                q_proxy,
                k_proxy,
                q,
                k,
                v,
                self.top_k,
                self.head_dim ** -0.5,
                **kernel_kwargs,
            )
        else:
            attn_out, kl_loss = self._attention_eager(
                q_proxy, k_proxy, q, k, v, document_list
            )
        return attn_out, kl_loss
