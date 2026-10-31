"""Numerically stable masked-mean accumulation for the NEW retrieval benchmark.

Historical encoder/cache implementations are intentionally left unchanged.
"""
import torch
import torch.nn.functional as F


def stable_masked_mean(hidden, mask):
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2]:
        raise ValueError("hidden/mask shape mismatch")
    if not torch.isfinite(hidden).all() or not torch.all((mask == 0) | (mask == 1)) or torch.any(mask.sum(1) == 0):
        raise ValueError("nonfinite hidden states or invalid/empty attention mask")
    pooled=(hidden.float()*mask.float()[...,None]).sum(1)/mask.float().sum(1,keepdim=True)
    if not torch.isfinite(pooled).all() or torch.any(pooled.norm(dim=-1)<=1e-12):
        raise ValueError("invalid pooled vector")
    return F.normalize(pooled,dim=-1)
