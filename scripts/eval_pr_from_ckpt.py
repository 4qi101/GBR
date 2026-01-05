import os

import torch

from utils.config import args as base_args
from utils.data import get_dataloaders
from nets.nets import ImgNet_T, TxtNet_T
from scripts.eval import eval_pr_curves


# === 手动配置区域 ===
# 将 CKPT_PATH 修改为需要计算 PR 曲线的 checkpoint 文件路径
CKPT_PATH = 'F:\project\pythonCode\Cross-modal retrieval\Granular-ball\GBR\checkpoints/flickr25k_code8_bs256_ep80_lr0.0001_gamma2.0_p10_t0.1_temp0.1_best.pth'  # 例如: 'checkpoints/flickr25k_64bit_best.pth'

# 如需强制指定数据集，可将 TARGET_DATASET 设置为 'coco' / 'nuswide' / 'flickr25k'
# 留空 (None) 时默认读取 utils.config 中的 args.dset
TARGET_DATASET = 'flickr25k'

# 可选：自定义 PR 结果保存目录与方法名称
CUSTOM_SAVE_DIR = None  # 例如: 'outputs/pr_curves/custom_exp'
METHOD_NAME = 'GBR'


DATASET_PRESETS = {
    'coco': {
        'gpu_id': '0',
        'seed': 42,
        'code_len': 128,
        'batch_size': 256,
        'num_workers': 0,
        'epochs': 40,
        'lr': 1e-4,
        'weight_decay': 1e-6,
        'optimizer': 'Adam',
        'lr_scheduler': 'cos',
        'eval_interval': 1,
        'use_tensorboard': True,
        'viz_tsne': False,
        'gamma_ball': 2.0,
        'mgbcc_p': 3,
        'mgbcc_t': 0.1,
        'mgbcc_temperature': 0.1,
        'use_affinity': False,
        'mgbcc_sim_threshold': 0.7,
    },
    'nuswide': {
        'gpu_id': '0',
        'seed': 42,
        'code_len': 128,
        'batch_size': 512,
        'num_workers': 0,
        'epochs': 40,
        'lr': 0.00015,
        'weight_decay': 0,
        'optimizer': 'Adam',
        'lr_scheduler': 'cos',
        'eval_interval': 1,
        'use_tensorboard': False,
        'viz_tsne': False,
        'gamma_ball': 2.0,
        'mgbcc_p': 16,
        'mgbcc_t': 0.05,
        'mgbcc_temperature': 0.15,
        'use_affinity': False,
        'mgbcc_sim_threshold': 0.7,
    },
    'flickr25k': {
        'gpu_id': '0',
        'seed': 42,
        'code_len': 8,
        'batch_size': 256,
        'num_workers': 0,
        'epochs': 80,
        'lr': 1e-4,
        'weight_decay': 1e-6,
        'optimizer': 'Adam',
        'lr_scheduler': 'cos',
        'eval_interval': 1,
        'map_topk': 0,
        'use_tensorboard': False,
        'viz_tsne': False,
        'viz_max_points':0,
        'gamma_ball': 2.0,
        'mgbcc_p': 10,
        'mgbcc_t': 0.1,
        'mgbcc_temperature': 0.1,
        'use_affinity': False,
        'mgbcc_sim_threshold': 0.8,
    },
}


def apply_dataset_preset(cfg, dataset_name):
    dataset_key = dataset_name.lower()
    preset = DATASET_PRESETS.get(dataset_key)
    if preset is None:
        print(f"[WARN] No preset configuration found for dataset '{dataset_name}'. Using base args.")
        return

    for key, value in preset.items():
        setattr(cfg, key, value)

    cfg.dset = dataset_key

def main():
    if not CKPT_PATH:
        raise ValueError('请先在 scripts/eval_pr_from_ckpt.py 中设置 CKPT_PATH 为合法的 checkpoint 路径。')

    cfg = base_args

    target_dset = TARGET_DATASET if TARGET_DATASET is not None else cfg.dset
    apply_dataset_preset(cfg, target_dset)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dl, test_dl, retrieval_dl = get_dataloaders(cfg)

    img_net = ImgNet_T(code_len=cfg.code_len).to(device)
    txt_net = TxtNet_T(text_length=512, code_len=cfg.code_len).to(device)

    ckpt_path = CKPT_PATH
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    print(f'===> Loading checkpoint: {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location=device)
    img_net.load_state_dict(checkpoint['img_net_state_dict'])
    txt_net.load_state_dict(checkpoint['txt_net_state_dict'])

    bit = cfg.code_len

    default_save_dir = os.path.join('pr_curves_data', cfg.dset)
    save_dir = CUSTOM_SAVE_DIR if CUSTOM_SAVE_DIR else default_save_dir

    print('===> Computing PR curves (I2T & T2I)...')
    eval_pr_curves(
        img_net,
        txt_net,
        test_dl,
        retrieval_dl,
        db_name=cfg.dset,
        device=device,
        method_name=METHOD_NAME,
        bit=bit,
        save_dir=save_dir,
    )

    print(f'===> PR curves saved under {save_dir} for method {METHOD_NAME}, bit={bit}.')


if __name__ == '__main__':
    main()
