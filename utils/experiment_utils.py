import os
from datetime import datetime
from typing import Any, Dict

import torch


def build_experiment_tag(args) -> str:
    """Encode key hyperparameters into a concise experiment tag."""
    return (
        f"{args.dset}"
        f"_code{args.code_len}"
        f"_bs{args.batch_size}"
        f"_ep{args.epochs}"
        f"_lr{args.lr}"
        f"_gamma{args.gamma_ball}"
        f"_p{args.mgbcc_p}"
        f"_t{args.mgbcc_t}"
        f"_temp{args.mgbcc_temperature}"
    )


def build_tensorboard_run_name(args) -> str:
    """Return a TensorBoard run name with timestamp for uniqueness."""
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    return f"{build_experiment_tag(args)}_{timestamp}"


def save_checkpoint(args, checkpoint: Dict[str, Any], suffix: str = 'best') -> str:
    """Save a checkpoint using the standardized experiment tag.

    Returns:
        The absolute path to the saved checkpoint file.
    """
    ckpt_name = f"{build_experiment_tag(args)}_{suffix}.pth"
    ckpt_path = os.path.join(args.save_dir, ckpt_name)
    torch.save(checkpoint, ckpt_path)
    return ckpt_path
