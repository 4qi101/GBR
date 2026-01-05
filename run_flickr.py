"""
GBR训练运行脚本
可以直接在此文件中修改参数，然后运行此文件即可开始训练
"""

import os
import sys
import warnings

from sympy import false

# 设置环境变量以避免KMeans在Windows上的内存泄漏警告
os.environ['OMP_NUM_THREADS'] = '1'

# 忽略sklearn的KMeans警告
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.cluster._kmeans')
warnings.filterwarnings('ignore', message='KMeans is known to have a memory leak')

from utils.config import args


# 基础参数
args.gpu_id = '0'                    # GPU设备ID
args.seed = 42
args.dset = 'flickr25k'                  # 数据集名称: 'coco', 'nuswide', 'flickr25k'
args.code_len =  8                # 哈希码长度
args.batch_size = 256               # 批次大小
args.num_workers = 0                 # DataLoader工作进程数（Windows建议设为0） 无需改

# 训练参数
args.epochs = 80                     # 训练轮数
args.lr = 0.0001                     # 学习率
args.weight_decay = 1e-6            # 权重衰减（L2正则化）无需改
args.optimizer = 'Adam'              # 优化器: 'Adam' 或 'SGD'  无需改
args.lr_scheduler = 'cos'            # 学习率调度器: 'linear', 'cos', 'step', 'plateau' 无需改
args.eval_interval = 1               # 每N个epoch评估一次
args.map_topk = 0                    # 评估时MAP统计的Top-K（0表示使用全部样本）

#  map曲线日志
args.use_tensorboard = 0
# 粒球可视化
args.viz_tsne = 0
args.viz_max_points = 5000
# 粒球模块参数
args.gamma_ball = 2.0                     # 粒球对比损失权重
args.mgbcc_p = 10                             # MGBCC粒度参数，控制粒球大小
args.mgbcc_t = 0.1                           # MGBCC跨视图交集阈值
args.mgbcc_temperature = 0.1

args.use_affinity = 0 # 使用同视图匹配
args.mgbcc_sim_threshold = 0.8       # MGBCC同视图粒球相似度阈值

# 查看tensorboard命令
#  tensorboard --logdir logs


# ckpt
# args.resume = 'checkpoints/coco_64bit_p8_t0.1_gamma1.0_best.pth'
# args.epochs = 100  # 继续训练到100个epoch

if __name__ == '__main__':
    from scripts.train import main
    
    if args.resume:
        print(f'Resume from: {args.resume}')
    else:
        print('Starting from scratch')
    print('=' * 80)
    print()

    try:
        main()
    except KeyboardInterrupt:
        print('\n\nTraining interrupted by user')
        sys.exit(0)
    except Exception as e:
        print(f'\n\nTraining failed with error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
