"""批量运行 COCO / NUS-WIDE / Flickr25k 三个数据集，在多个 code_len 下依次训练。

使用方式：
    python run_all_datasets.py

注意：这会顺序启动 15 次完整训练（3 个数据集 × 5 个 code_len），耗时较长。
"""

import importlib

from utils.config import args
from scripts.train import main as train_main


DATASET_RUNNERS = [
    ("coco", "run_coco"),
    ("nuswide", "run_nuswide"),
    ("flickr25k", "run_flickr"),
]

CODE_LENS = [8, 16, 32, 64, 128]


def run_experiments():
    for dset_name, module_name in DATASET_RUNNERS:
        for bit in CODE_LENS:
            print("=" * 80)
            print(f"[RUN] dataset={dset_name}, code_len={bit}")
            print("=" * 80)

            # 1) 重新加载对应数据集的运行脚本，写入该数据集的默认超参到 args
            module = importlib.import_module(module_name)
            importlib.reload(module)

            # 2) 覆盖 code_len（以及可选的 resume 等）
            args.code_len = bit
            args.resume = ''  # 每个实验从头训练

            # 3) 启动一次完整训练
            train_main()


if __name__ == "__main__":
    run_experiments()
