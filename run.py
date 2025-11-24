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

# 导入配置
from utils.config import args

# ============================================
# 在此处修改训练参数
# ============================================

# 基础参数
args.gpu_id = '0'                    # GPU设备ID
args.dset = 'flickr25k'                  # 数据集名称: 'coco', 'nuswide', 'flickr25k'
args.code_len = 64                  # 哈希码长度
args.batch_size = 1024               # 批次大小
args.num_workers = 0                 # DataLoader工作进程数（Windows建议设为0） 无需改

# 训练参数
args.epochs = 200                     # 训练轮数
args.lr = 0.0001                     # 学习率
args.weight_decay = 1e-6             # 权重衰减（L2正则化）无需改
args.optimizer = 'Adam'              # 优化器: 'Adam' 或 'SGD'  无需改
args.lr_scheduler = 'cos'            # 学习率调度器: 'linear', 'cos', 'step', 'plateau' 无需改
args.eval_interval = 10               # 每N个epoch评估一次

#  map曲线日志
args.use_tensorboard = 0

# 粒球模块参数
args.gamma_ball = 1.0                      # 粒球对比损失权重
args.mgbcc_p = 64                             # MGBCC粒度参数，控制粒球大小
args.mgbcc_t = 0.1                           # MGBCC跨视图交集阈值，控制粒球匹配的严格程度
args.mgbcc_temperature = 0.3        # MGBCC对比学习温度参数

args.use_affinity = 1  # 使用同视图匹配
args.mgbcc_sim_threshold = 0.679       # MGBCC同视图粒球相似度阈值（相似度 > 此值才匹配）

# Memory Bank 参数
args.use_memory_bank = 1
args.lambda_mem = 0.1                 # memory bank 损失权重
args.mem_momentum = 0.9              # memory bank 动量系数
args.mem_temperature = 0.1           # memory bank 对比温度
args.mem_negatives = 4096           # memory bank 负样本数量


# 量化正则约束参数
args.use_quantization = 0
args.use_bit_balance = 0
args.lambda_quant = 0.1               # 量化损失权重 (拉近连续码与二值码)
args.lambda_balance = 0.01           # 位平衡损失权重 (鼓励0/1均衡)


# ============================================
# 查看tensorboard命令
#  tensorboard --logdir logs
# ============================================


# ============================================
# 参数配置示例（可根据需要取消注释使用）
# ============================================

# 示例1: COCO数据集，64位哈希码，训练100个epoch
# args.dset = 'coco'
# args.code_len = 64
# args.epochs = 100
# args.lr = 0.0001
# args.gamma_ball = 1.0
# args.mgbcc_p = 8

# 示例2: Flickr25K数据集，128位哈希码，使用余弦学习率调度
# args.dset = 'flickr25k'
# args.code_len = 128
# args.epochs = 50
# args.lr_scheduler = 'cos'
# args.mgbcc_p = 4

# 示例3: 从检查点恢复训练
# args.resume = 'checkpoints/coco_64bit_p8_t0.1_gamma1.0_best.pth'
# args.epochs = 100  # 继续训练到100个epoch


# ============================================
# 运行训练
# ============================================

if __name__ == '__main__':
    # 在参数设置完成后再导入训练模块，确保所有配置生效
    from scripts.train import main
    
    if args.resume:
        print(f'Resume from: {args.resume}')
    else:
        print('Starting from scratch')
    print('=' * 80)
    print()
    
    # 开始训练
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
