import numpy as np
import torch
from torch.backends import cudnn


def seed_everything(seed=1234):
    """设置随机种子以确保实验可复现"""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

