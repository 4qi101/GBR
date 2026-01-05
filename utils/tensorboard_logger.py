"""Helper utilities for TensorBoard logging of training metrics."""
import os
from typing import Optional

from torch.utils.tensorboard import SummaryWriter

from utils.experiment_utils import build_tensorboard_run_name


def create_tb_writer(args) -> Optional[SummaryWriter]:
    """Create a TensorBoard SummaryWriter if enabled in args."""
    if not getattr(args, 'use_tensorboard', False):
        return None

    run_name = build_tensorboard_run_name(args)
    log_dir = os.path.join(args.tensorboard_logdir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[TensorBoard] Logging to: {log_dir}")
    return writer


def log_eval_metrics(writer: Optional[SummaryWriter], epoch: int,
                     mapi2t: float, mapt2i: float, avg_map: float,
                     prefix: str = 'eval') -> None:
    if writer is None:
        return
    writer.add_scalar(f'{prefix}/MAP_i2t', mapi2t, epoch)
    writer.add_scalar(f'{prefix}/MAP_t2i', mapt2i, epoch)
    writer.add_scalar(f'{prefix}/MAP_avg', avg_map, epoch)


def log_train_scalar(writer: Optional[SummaryWriter], tag: str,
                      value: float, step: int) -> None:
    if writer is None:
        return
    writer.add_scalar(tag, value, step)
