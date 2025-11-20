"""
GBR完整训练脚本
使用粒球对比学习的跨模态检索训练
"""

import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy

from utils.config import args
from utils.util import seed_everything
from nets.nets import ImgNet_T, TxtNet_T
from utils.data import get_dataloaders
from scripts.eval import eval_retrieval_target
from src.granular_loss import compute_granular_ball_loss
from src.memory_bank import MomentumBinaryMemoryBank
from src.mgbcc_style_balls import MultiviewGCLoss
from src.regularization import quantization_loss, bit_balance_loss
from utils.tensorboard_logger import create_tb_writer, log_eval_metrics, log_train_scalar
from utils.debug_tools import log_memory_bank_loss


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
    lambda_mem = args.lambda_mem
    mem_momentum = args.mem_momentum
    mem_temperature = args.mem_temperature
    mem_negatives = args.mem_negatives

    # 6.1 初始化memory bank
    memory_bank = None
    if args.use_memory_bank and lambda_mem > 0:
        num_train_samples = len(train_dl.dataset)
        memory_bank = MomentumBinaryMemoryBank(
            num_samples=num_train_samples,
            code_dim=args.code_len,
            momentum=mem_momentum,
            device=device,
        )
    
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
            memory_bank,
            lambda_mem=lambda_mem,
            mem_negatives=mem_negatives,
            mem_temperature=mem_temperature,
            device=device,
            tb_writer=tb_writer,
        )
        
        # 评估
        should_eval = (epoch == 0 or (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1)
        if should_eval:
            mapi2t, mapt2i = eval_retrieval_target(
                img_net,
                txt_net,
                test_dl,
                retrieval_dl,
                db_name=args.dset,
                device=device,
            )
            avg_map = (mapi2t + mapt2i) / 2.0
            
            print(f'\n[Epoch {epoch+1}/{args.epochs}] Evaluation Results:')
            print(f'  Image->Text MAP: {mapi2t:.6f}')
            print(f'  Text->Image MAP: {mapt2i:.6f}')
            print(f'  Average MAP: {avg_map:.6f}')
            log_eval_metrics(tb_writer, epoch + 1, mapi2t, mapt2i, avg_map)
            
            # 保存最佳模型
            if avg_map > best_avg_map:
                best_avg_map = avg_map
                best_epoch = epoch + 1
                best_states = {
                    'img_net': deepcopy(img_net.state_dict()),
                    'txt_net': deepcopy(txt_net.state_dict()),
                }
                best_mapi2t = mapi2t
                best_mapt2i = mapt2i
                
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
                ckpt_name = f"{args.dset}_{args.code_len}bit_p{args.mgbcc_p}_t{args.mgbcc_t}_gamma{args.gamma_ball}_best.pth"
                ckpt_path = os.path.join(args.save_dir, ckpt_name)
                torch.save(checkpoint, ckpt_path)
                print(f'  ==> Saved best nets to {ckpt_path}')
        
        # 更新学习率
        if args.lr_scheduler == 'plateau':
            scheduler.step(avg_map if should_eval else 0)
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
        )
        avg_map = (mapi2t + mapt2i) / 2.0
        print(f'  Image->Text MAP: {mapi2t:.6f}')
        print(f'  Text->Image MAP: {mapt2i:.6f}')
        print(f'  Average MAP: {avg_map:.6f}')
    
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
                   memory_bank: MomentumBinaryMemoryBank,
                   lambda_mem: float,
                   mem_negatives: int,
                   mem_temperature: float,
                   device: torch.device,
                   tb_writer=None):
    """训练一个epoch"""
    img_net.train()
    txt_net.train()
    
    running_loss = 0.0
    num_samples = 0
    num_batches = len(train_loader)
    
    for batch_idx, (images, texts, labels, indices) in enumerate(train_loader):
        images = images.to(device)
        texts = texts.to(device)
        indices = indices.to(device)
        
        # 前向传播
        _, _, _, img_code = img_net(images)
        _, _, _, txt_code = txt_net(texts)
        
        # 归一化特征（与UCCH一致）
        v_img = F.normalize(img_code, dim=1, eps=1e-8)
        v_txt = F.normalize(txt_code, dim=1, eps=1e-8)
        v_img = torch.nan_to_num(v_img, nan=0.0, posinf=1.0, neginf=-1.0)
        v_txt = torch.nan_to_num(v_txt, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 构球与中心级对比（与UCCH一致）
        Lh_ball = torch.tensor(0.0, device=v_img.device)
        
        if gamma > 0:
            # 仅使用批内粒球（与UCCH一致）
            Lh_ball = compute_granular_ball_loss(
                v_img,
                v_txt,
                p=p_cfg,
                temperature=args.mgbcc_temperature,
                match_threshold=args.mgbcc_t,
                criterion=criterion_gra_global,
            )
            if torch.isnan(Lh_ball) or torch.isinf(Lh_ball):
                Lh_ball = torch.tensor(0.0, device=v_img.device)
        
        # Memory Bank 损失
        L_mem = torch.tensor(0.0, device=v_img.device)
        if args.use_memory_bank and lambda_mem > 0 and memory_bank is not None:
            L_mem_img = memory_bank.contrastive_loss(
                v_img,
                indices,
                num_negatives=mem_negatives,
                temperature=mem_temperature,
            )
            L_mem_txt = memory_bank.contrastive_loss(
                v_txt,
                indices,
                num_negatives=mem_negatives,
                temperature=mem_temperature,
            )
            L_mem = (L_mem_img + L_mem_txt) * 0.5
            log_memory_bank_loss(
                L_mem_img,
                L_mem_txt,
                mem_negatives,
                prefix=f"MemBank(epoch={epoch+1}, batch={batch_idx+1})",
            )

        # 量化损失：拉近连续码与二值码
        L_quant = torch.tensor(0.0, device=v_img.device)
        if args.use_quantization and args.lambda_quant > 0:
            L_quant_img = quantization_loss(img_code, detach_target=True)
            L_quant_txt = quantization_loss(txt_code, detach_target=True)
            L_quant = (L_quant_img + L_quant_txt) * 0.5
        
        # 位平衡损失：鼓励每一位0/1均衡
        L_balance = torch.tensor(0.0, device=v_img.device)
        if args.use_bit_balance and args.lambda_balance > 0:
            L_balance_img = bit_balance_loss(img_code)
            L_balance_txt = bit_balance_loss(txt_code)
            L_balance = (L_balance_img + L_balance_txt) * 0.5

        # 总损失
        loss = (gamma * Lh_ball + 
                lambda_mem * L_mem + 
                args.lambda_quant * L_quant + 
                args.lambda_balance * L_balance)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(img_net.parameters()) + list(txt_net.parameters()), 
            max_norm=10.0
        )
        optimizer.step()

        running_loss += loss.item()
        num_samples += 1

        if args.use_memory_bank and memory_bank is not None:
            with torch.no_grad():
                avg_feature = 0.5 * (v_img + v_txt)
                memory_bank.update(indices, avg_feature)
        
        # 打印进度
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
            current_lr = optimizer.param_groups[0]['lr']
            avg_loss = running_loss / (batch_idx + 1)
            print(f'Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{num_batches}] '
                  f'Loss: {avg_loss:.4f} LR: {current_lr:.6f}')

    # Epoch-level logging
    epoch_avg_loss = running_loss / max(num_samples, 1)
    log_train_scalar(tb_writer, 'train/loss', epoch_avg_loss, epoch)


if __name__ == '__main__':
    main()

