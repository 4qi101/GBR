"""
GBR完整训练脚本
使用粒球对比学习的跨模态检索训练
"""

import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from copy import deepcopy

from utils.config import args
from utils.util import seed_everything
from nets.nets import ImgNet_T, TxtNet_T
from utils.data import get_dataloaders
from scripts.eval import eval_retrieval_target, eval_pr_curves
from src.granular_loss import compute_granular_ball_loss
from src.mgbcc_style_balls import MultiviewGCLoss
from utils.tensorboard_logger import create_tb_writer, log_eval_metrics, log_train_scalar
from utils.experiment_utils import save_checkpoint
from utils.viz_utils import visualize_embeddings_after_training


def main():
    # 0. 设置环境和数据加载器（在这里才会读取 args 的实际值）
    seed_everything(seed=args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建数据加载器
    train_dl, test_dl, retrieval_dl = get_dataloaders(args)
    print(f'===> Data loaded: {len(train_dl)} train batches, batch_size={args.batch_size}')

    # TensorBoard writer（可选）
    tb_writer = create_tb_writer(args)

    # 1. 初始化模型
    print('\n===> Building models...')
    img_net = ImgNet_T(code_len=args.code_len).to(device)
    txt_net = TxtNet_T(text_length=512, code_len=args.code_len).to(device)
    
    # 2. 创建优化器
    parameters = list(img_net.parameters()) + list(txt_net.parameters())
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:  # Adam
        optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    
    # 3. 创建学习率调度器
    if args.lr_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 0.1)
    elif args.lr_scheduler == 'linear':
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=args.epochs)
    elif args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.7)
    elif args.lr_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                                   milestones=[args.epochs // 4, args.epochs // 2, args.epochs * 3 // 4], 
                                                   gamma=0.7)
    
    # 4. 加载检查点（如果存在）
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f'===> Loading checkpoint: {args.resume}')
            checkpoint = torch.load(args.resume)
            img_net.load_state_dict(checkpoint['img_net_state_dict'])
            txt_net.load_state_dict(checkpoint['txt_net_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f'===> Loaded checkpoint (epoch {start_epoch})')
        else:
            print(f'===> No checkpoint found at {args.resume}, starting from scratch')
    
    # 5. 创建粒球损失函数（与UCCH一致：预先创建，避免每batch创建）
    criterion_gra_global = MultiviewGCLoss(
        temperature=args.mgbcc_temperature,
        match_threshold=args.mgbcc_t,
        sim_threshold=args.mgbcc_sim_threshold,
        use_affinity=args.use_affinity,
    )

    print('[Debug] Mask logging enabled (intra/inter sums will be printed).')
    
    # 6. 训练参数
    p_cfg = args.mgbcc_p
    gamma = args.gamma_ball
    map_topk = args.map_topk if getattr(args, 'map_topk', 0) > 0 else None
    
    # 7. 最佳模型跟踪
    best_avg_map = -1.0
    best_epoch = -1
    best_states = None
    best_mapi2t = 0.0
    best_mapt2i = 0.0
    
    # 8. 训练循环
    print('\n===> Start training...')
    for epoch in range(start_epoch, args.epochs):
        # 训练一个epoch
        train_epoch(
            epoch,
            img_net,
            txt_net,
            train_dl,
            optimizer,
            criterion_gra_global,
            p_cfg,
            gamma,
            device=device,
            tb_writer=tb_writer,
        )

        # 评估逻辑
        should_eval = (epoch == 0 or (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1)
        is_new_best = False

        if should_eval:
            # =========================================================
            # [Fix] 核心修改：全方位随机数锁 (RNG Lock)
            # 进入评估前，保存 Torch, CUDA, Numpy, Python Random 的所有状态
            # 确保评估过程（包括t-SNE和数据加载）完全“隐形”，不影响训练的随机序列
            # =========================================================
            rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state(device=device) if torch.cuda.is_available() else None
            np_rng_state = np.random.get_state()
            py_rng_state = random.getstate()
            # =========================================================

            # --- 开始评估 ---
            mapi2t, mapt2i = eval_retrieval_target(
                img_net,
                txt_net,
                test_dl,
                retrieval_dl,
                db_name=args.dset,
                device=device,
                topk=map_topk,
            )
            avg_map = (mapi2t + mapt2i) / 2.0

            print(f'\n[Epoch {epoch + 1}/{args.epochs}] Evaluation Results:')
            print(f'  Image->Text MAP: {mapi2t:.6f}')
            print(f'  Text->Image MAP: {mapt2i:.6f}')
            print(f'  Average MAP: {avg_map:.6f}')
            log_eval_metrics(tb_writer, epoch + 1, mapi2t, mapt2i, avg_map)

            # --- 保存最佳模型 ---
            if avg_map > best_avg_map:
                best_avg_map = avg_map
                best_epoch = epoch + 1
                best_states = {
                    'img_net': deepcopy(img_net.state_dict()),
                    'txt_net': deepcopy(txt_net.state_dict()),
                }
                best_mapi2t = mapi2t
                best_mapt2i = mapt2i
                is_new_best = True

                # 保存检查点
                checkpoint = {
                    'epoch': epoch + 1,
                    'img_net_state_dict': img_net.state_dict(),
                    'txt_net_state_dict': txt_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'mapi2t': mapi2t,
                    'mapt2i': mapt2i,
                    'avg_map': avg_map,
                }

                # 构建保存文件名
                ckpt_path = save_checkpoint(args, checkpoint, suffix='best')
                print(f'  ==> Saved best nets to {ckpt_path}')

                if args.enable_pr_on_ckpt:
                    try:
                        print('  ==> Computing PR curves for current best model...')
                        method_name = 'GBR'
                        bit = args.code_len
                        save_dir = os.path.join('pr_curves_data', args.dset)
                        eval_pr_curves(
                            img_net,
                            txt_net,
                            test_dl,
                            retrieval_dl,
                            db_name=args.dset,
                            device=device,
                            method_name=method_name,
                            bit=bit,
                            save_dir=save_dir,
                        )
                        print(f'  ==> PR curves saved to {save_dir} for current best model.')
                    except Exception as e:
                        print(f'[PR] Failed to compute PR curves for current best model: {e}')

            # --- 可视化 (t-SNE) ---
            if getattr(args, 'viz_tsne', False) and (epoch + 1) > 40 and is_new_best:
                print(f'  ==> Generating visualization figures (t-SNE) at epoch {epoch + 1}...')
                try:
                    visualize_embeddings_after_training(
                        img_net=img_net,
                        txt_net=txt_net,
                        loader=train_dl,
                        device=device,
                        args=args,
                        epoch=epoch + 1,
                    )
                except Exception as e:
                    print(f'[Viz] Failed to generate visualization: {e}')

            # =========================================================
            # [Fix] 核心修改：恢复状态
            # 评估结束，把所有随机数指针拨回评估前的瞬间
            # =========================================================
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, device=device)
            np.random.set_state(np_rng_state)
            random.setstate(py_rng_state)
            # =========================================================

        # 更新学习率 (保持在 should_eval 块之外)
        if args.lr_scheduler == 'plateau':
            # 注意：如果跳过评估，plateau 默认不更新（或者你可以传入上次的 avg_map）
            # 这里传入 0 是为了语法占位，实际 plateau 需要有效的 metric 才能 step
            current_metric = avg_map if should_eval else 0.0
            scheduler.step(current_metric)
        else:
            scheduler.step()
    
    # 9. 加载最佳模型并最终评估
    print('\n===> Training completed!')
    if best_states is not None:
        print(f'Loading best nets from epoch {best_epoch} (Avg MAP: {best_avg_map:.6f})')
        img_net.load_state_dict(best_states['img_net'])
        txt_net.load_state_dict(best_states['txt_net'])
        
        # 最终评估
        print('\n===> Final evaluation with best nets:')
        mapi2t, mapt2i = eval_retrieval_target(
            img_net,
            txt_net,
            test_dl,
            retrieval_dl,
            db_name=args.dset,
            device=device,
            topk=map_topk,
        )
        avg_map = (mapi2t + mapt2i) / 2.0
        print(f'  Image->Text MAP: {mapi2t:.6f}')
        print(f'  Text->Image MAP: {mapt2i:.6f}')
        print(f'  Average MAP: {avg_map:.6f}')

    if getattr(args, 'viz_tsne', False):
        print('\n===> Generating visualization figures (t-SNE)...')
        try:
            final_epoch = best_epoch if best_epoch != -1 else args.epochs
            visualize_embeddings_after_training(
                img_net=img_net,
                txt_net=txt_net,
                loader=train_dl,
                device=device,
                args=args,
                epoch=final_epoch,
            )
        except Exception as e:
            print(f'[Viz] Failed to generate visualization: {e}')
    
    print('=' * 80)

    if tb_writer is not None:
        log_eval_metrics(
            tb_writer,
            best_epoch if best_epoch != -1 else args.epochs,
            best_mapi2t,
            best_mapt2i,
            best_avg_map if best_avg_map > 0 else 0.0,
            prefix='best'
        )
        tb_writer.close()


def train_epoch(epoch, img_net, txt_net, train_loader, optimizer,
                   criterion_gra_global, p_cfg, gamma,
                   device: torch.device,
                   tb_writer=None):
    """训练一个epoch"""
    img_net.train()
    txt_net.train()

    running_loss = 0.0
    total_Lh_ball = 0.0
    num_samples = 0
    num_batches = len(train_loader)

    for batch_idx, (images, texts, _labels, _indices) in enumerate(train_loader):
        images = images.to(device)
        texts = texts.to(device)

        # Step 1: 前向传播（query encoder）- 使用当前 img_net/txt_net 计算查询特征 q
        _, _, _, img_code = img_net(images)
        _, _, _, txt_code = txt_net(texts)

        # 归一化特征（与UCCH一致）
        v_img = F.normalize(img_code, dim=1, eps=1e-8)
        v_txt = F.normalize(txt_code, dim=1, eps=1e-8)
        v_img = torch.nan_to_num(v_img, nan=0.0, posinf=1.0, neginf=-1.0)
        v_txt = torch.nan_to_num(v_txt, nan=0.0, posinf=1.0, neginf=-1.0)

        # Step 3: 构球与中心级对比损失 Lh_ball（批内粒球）
        Lh_ball = torch.tensor(0.0, device=v_img.device)
        if gamma > 0:
            Lh_ball = compute_granular_ball_loss(
                v_img,
                v_txt,
                p=p_cfg,
                temperature=args.mgbcc_temperature,
                match_threshold=args.mgbcc_t,
                criterion=criterion_gra_global,
                return_aux=False,
            )

            if torch.isnan(Lh_ball) or torch.isinf(Lh_ball):
                Lh_ball = torch.tensor(0.0, device=v_img.device)

        # Step 4: 汇总总损失，并对 query encoder 反向更新
        loss = gamma * Lh_ball

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(img_net.parameters()) + list(txt_net.parameters()),
            max_norm=10.0,
        )
        optimizer.step()

        running_loss += loss.item()
        total_Lh_ball += Lh_ball.item() if isinstance(Lh_ball, torch.Tensor) else float(Lh_ball)
        num_samples += 1

        # 打印进度
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
            current_lr = optimizer.param_groups[0]['lr']
            avg_loss = running_loss / (batch_idx + 1)
            print(
                f'Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{num_batches}] '
                f'Loss: {avg_loss:.4f} LR: {current_lr:.6f}'
            )

    # Epoch-level logging
    denom = max(num_samples, 1)
    epoch_avg_loss = running_loss / denom
    avg_Lh_ball = total_Lh_ball / denom

    log_train_scalar(tb_writer, 'train/loss', epoch_avg_loss, epoch)
    log_train_scalar(tb_writer, 'train/Lh_ball', avg_Lh_ball, epoch)


if __name__ == '__main__':
    main()

