import argparse

parser = argparse.ArgumentParser(description='GBR Backbone - Cross-modal Retrieval with Granular Ball')
# 基本超参数
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="GPU device id")
parser.add_argument('--batch_size', type=int, default=256, help="batch size")
parser.add_argument('--num_workers', type=int, default=0, help="number of workers for DataLoader")
parser.add_argument('--code_len', type=int, default=64, help="hash code length")
parser.add_argument('--seed', type=int, default=1234, help="random seed")

# 数据集相关
parser.add_argument('--data_path', type=str, default='data/Dataset/', help="dataset root path")
parser.add_argument('--dset', type=str, default='coco', choices=['coco', 'nuswide', 'flickr25k'], help="dataset name")

# 训练相关参数
parser.add_argument('--epochs', type=int, default=20, help="training epochs")
parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
parser.add_argument('--weight_decay', type=float, default=1e-6, help="weight decay")
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], help="optimizer type")
parser.add_argument('--lr_scheduler', type=str, default='linear', choices=['linear', 'cos', 'step', 'plateau'], help="learning rate scheduler")
parser.add_argument('--eval_interval', type=int, default=5, help="evaluate every N epochs")
parser.add_argument('--resume', type=str, default='', help="path to checkpoint to resume from")
parser.add_argument('--save_dir', type=str, default='checkpoints', help="directory to save models")

#  === memory bank模块参数 ===
parser.add_argument('--lambda_mem', type=float, default=0.1,
                    help='weight for memory-bank contrastive loss')
parser.add_argument('--mem_momentum', type=float, default=0.99,
                    help='momentum for updating memory bank entries')
parser.add_argument('--mem_temperature', type=float, default=0.07,
                    help='temperature for memory bank contrastive loss')
parser.add_argument('--mem_negatives', type=int, default=1024,
                    help='number of negative keys sampled from memory bank')
parser.add_argument('--use_memory_bank', type=int, default=0,
                    choices=[0, 1], help='是否启用memory bank (1启用, 0关闭)')

# === MGBCC粒球模块参数 ===
parser.add_argument('--gamma_ball', type=float, default=1.0,
                    help='粒球对比损失权重 (granular ball contrastive loss weight)')
parser.add_argument('--mgbcc_p', type=int, default=8,
                    help='MGBCC粒度参数，控制粒球大小 (granularity parameter, controls ball size)')
parser.add_argument('--mgbcc_t', type=float, default=0.1,
                    help='MGBCC交集阈值，控制粒球匹配的严格程度 (intersection threshold for ball matching)')
parser.add_argument('--mgbcc_temperature', type=float, default=1.0,
                    help='MGBCC对比学习温度参数 (temperature parameter for contrastive learning)')
parser.add_argument('--mgbcc_sim_threshold', type=float, default=0.5,
                    help='MGBCC粒球相似度阈值 (similarity threshold for ball affinity matching)')
parser.add_argument('--use_affinity', action='store_true',
                    help='是否启用同视图粒球affinity匹配 (enable intra-view affinity matching)')

# === 量化正则约束参数 ===
parser.add_argument('--lambda_quant', type=float, default=0.0,
                    help='量化损失权重 (quantization loss weight)')
parser.add_argument('--lambda_balance', type=float, default=0.0,
                    help='位平衡损失权重 (bit balance loss weight)')
parser.add_argument('--use_quantization', type=int, default=0,
                    choices=[0, 1], help='是否启用量化损失 (1启用, 0关闭)')
parser.add_argument('--use_bit_balance', type=int, default=0,
                    choices=[0, 1], help='是否启用位平衡损失 (1启用, 0关闭)')

# === TensorBoard 日志参数 ===
parser.add_argument('--use_tensorboard', type=int, default=0,
                    choices=[0, 1], help='是否启用TensorBoard日志记录 (1启用, 0关闭)')
parser.add_argument('--tensorboard_logdir', type=str, default='logs',
                    help='TensorBoard日志根目录')

args = parser.parse_args()

# 将0/1开关转换为布尔值，方便后续使用
args.use_memory_bank = bool(args.use_memory_bank)
args.use_quantization = bool(args.use_quantization)
args.use_bit_balance = bool(args.use_bit_balance)
args.use_tensorboard = bool(args.use_tensorboard)






