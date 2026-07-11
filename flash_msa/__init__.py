"""Public API for Flash-MSA."""

from flash_msa.flash_msa import sparse_attention as flash_msa_func
from flash_msa.warmup.flash_msa_warmup import sparse_attention_warmup

flash_msa_func_warmup = sparse_attention_warmup
flash_msa_warmup_func = sparse_attention_warmup

__all__ = [
    "flash_msa_func",
    "flash_msa_func_warmup",
    "flash_msa_warmup_func",
    "sparse_attention_warmup",
]
