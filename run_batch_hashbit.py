"""
GBR 批量训练脚本 (Batch Run Script)
功能：自动按顺序运行 8, 16, 32, 64, 128 bit 的实验
"""

import os
import sys
import torch
import gc
import warnings
import time

# 忽略警告
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.cluster._kmeans')
warnings.filterwarnings('ignore', message='KMeans is known to have a memory leak')
os.environ['OMP_NUM_THREADS'] = '1'

# 导入配置和训练主函数
from utils.config import args
from scripts.train import main

# 基础参数
args.gpu_id = '0'
args.seed = 42
args.dset = 'coco'
# args.code_len 将在循环中动态设置
args.batch_size = 1024
args.num_workers = 0

# 训练参数
args.epochs = 90
args.lr = 0.0002
args.weight_decay = 0
args.optimizer = 'Adam'
args.lr_scheduler = 'cos'
args.eval_interval = 1

# 可视化与日志
args.use_tensorboard = 1
args.viz_tsne = 0

# 粒球模块参数
args.gamma_ball = 1.0
args.mgbcc_p = 4
args.mgbcc_t = 0.2
args.mgbcc_temperature = 0.05
args.use_affinity = 0
args.mgbcc_sim_threshold = 0.7

def run_batch():
    # 定义需要运行的 Hash Bit 列表
    bits_list = [8, 16, 32, 64, 128]

    # 记录总开始时间
    total_start_time = time.time()

    print(f"准备开始批量训练，目标 Bits: {bits_list}")
    print("=" * 60)

    for i, bit in enumerate(bits_list):
        print(f"\n\n>>> [{i + 1}/{len(bits_list)}] 正在开始训练: Code Length = {bit}")
        print("-" * 60)

        # [核心] 修改 Hash 长度参数
        args.code_len = bit

        # [可选] 为了区分日志，您也可以在这里微调 save_dir，
        # 但通常 save_checkpoint 函数内部会根据 bit 自动命名文件，所以不改也没关系。
        # args.save_dir = f'checkpoints/{args.dset}_{bit}bits'

        try:
            # 运行训练主函数
            main()

            print(f"\n>>> Code Length = {bit} 训练完成。")

        except Exception as e:
            print(f"\n!!! Code Length = {bit} 训练过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            # 即使出错也继续跑下一个 bit，不中断整个脚本

        finally:
            # [重要] 清理显存和内存，防止下一个循环 OOM (Out Of Memory)
            print("正在清理 GPU 显存...")
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(5)  # 给系统一点喘息时间
            print("-" * 60)

    total_time = time.time() - total_start_time
    print(f"\n\n所有任务已完成！总耗时: {total_time / 3600:.2f} 小时")


if __name__ == '__main__':
    try:
        run_batch()
    except KeyboardInterrupt:
        print('\n\nBatch training interrupted by user')
        sys.exit(0)