"""Debug utilities for monitoring granular ball masks."""
from __future__ import annotations

from typing import Optional

import torch


def log_mask_stats(
    mask_intra: torch.Tensor,
    mask_inter: Optional[torch.Tensor] = None,
    prefix: str = "",
) -> None:
    """Print basic statistics for intra/inter-view masks.

    Args:
        mask_intra: Intra-view mask tensor.
        mask_inter: Inter-view mask tensor (optional).
        prefix: Additional text to distinguish logging source.
    """
    if mask_intra is None:
        return

    mask_intra_sum = float(mask_intra.sum().item())
    mask_intra_mean = float(mask_intra.float().mean().item())

    msg = (
        f"[MaskDebug] {prefix} "
        f"intra_sum={mask_intra_sum:.4f} intra_mean={mask_intra_mean:.6f}"
    )

    if mask_inter is not None:
        mask_inter_sum = float(mask_inter.sum().item())
        mask_inter_mean = float(mask_inter.float().mean().item())
        msg += (
            f" | inter_sum={mask_inter_sum:.4f} "
            f"inter_mean={mask_inter_mean:.6f}"
        )

    print(msg)


def log_memory_bank_loss(
    loss_img: torch.Tensor,
    loss_txt: torch.Tensor,
    mem_negatives: int,
    prefix: str = "MemBank",
) -> None:
    """Print average memory bank losses for image/text branches.

    Args:
        loss_img: Memory bank loss from image branch.
        loss_txt: Memory bank loss from text branch.
        mem_negatives: Number of negatives sampled from memory bank.
        prefix: Optional prefix to distinguish logs.
    """
    if loss_img is None or loss_txt is None:
        return

    img_val = float(loss_img.detach().mean().item())
    txt_val = float(loss_txt.detach().mean().item())
    print(
        f"[{prefix}] negatives={mem_negatives} "
        f"L_mem_img={img_val:.6f} L_mem_txt={txt_val:.6f}"
    )


def log_similarity_stats(
    similarity: torch.Tensor,
    prefix: str = "Similarity",
) -> None:
    """Print statistics (min/mean/max) for a similarity matrix."""
    if similarity is None or similarity.numel() == 0:
        return

    sim = similarity.detach()
    min_val = float(sim.min().item())
    max_val = float(sim.max().item())
    mean_val = float(sim.mean().item())
    print(
        f"[{prefix}] min={min_val:.6f} mean={mean_val:.6f} max={max_val:.6f}"
    )
