"""Regularization helpers for hash learning.

Provides quantization loss (continuous vs. binary codes) and bit-balance loss.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def quantization_loss(codes: torch.Tensor, detach_target: bool = True) -> torch.Tensor:
    """Penalize the gap between continuous codes and their binarized targets."""
    if codes.numel() == 0:
        return torch.tensor(0.0, device=codes.device)

    target = torch.sign(codes)
    target[target == 0] = 1.0
    if detach_target:
        target = target.detach()
    return F.mse_loss(codes, target)


def bit_balance_loss(codes: torch.Tensor) -> torch.Tensor:
    """Encourage each bit dimension to be zero-mean (balanced 0/1)."""
    if codes.numel() == 0:
        return torch.tensor(0.0, device=codes.device)
    bit_mean = codes.mean(dim=0)
    return bit_mean.abs().mean()
