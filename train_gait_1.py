import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import numpy as np
import time
import sys
import argparse
import errno
from collections import OrderedDict
import tensorboardX
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader

from lib.utils.tools import *
from lib.utils.learning import *
from lib.model.loss import *
from lib.data.dataset_gait import GaitDataset_1
from lib.model.model_gait import GaitNet_1

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gait/MB_ft_gait_track1.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint/gait1', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-freq', '--print_freq', default=5)
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('--split', action='store_true', default=False, help='whether split train/val dataset')
    opts = parser.parse_args()
    return opts


def validate(test_loader, model, loss_fn):
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc_val = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (batch_input, batch_gt) in tqdm(enumerate(test_loader)):
            batch_size = len(batch_input)    
            if torch.cuda.is_available():
                batch_gt = batch_gt.cuda()
                batch_input = batch_input.cuda()
            pred_logits = model(batch_input)    # (B, 2, 17)
            loss = loss_fn(pred_logits, batch_gt)

            # update metric
            losses.update(loss.item(), batch_size)
            acc = track_1_compute_acc(pred_logits, batch_gt)
            acc_val.update(acc, batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx+1) % (opts.print_freq // 5) == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.avg:.4f} ({loss.val:.4f})\t'
                      'Acc {acc_val.avg:.3f} ({acc_val.val:.3f})\t'.format(
                       idx, len(test_loader), batch_time=batch_time,
                       loss=losses, acc_val=acc_val))
    return losses.avg, acc_val.avg


def train_split_data(args, opts):
    print(args)
    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))
    model_backbone = load_backbone(args)
    if args.finetune:
        if opts.resume or opts.evaluate:
            pass
        else:
            chk_filename = os.path.join(opts.pretrained, opts.selection)
            print('Loading backbone', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)['model_pos']
            model_backbone = load_pretrained_weights(model_backbone, checkpoint)
    if args.partial_train:
        model_backbone = partial_train_layers(model_backbone, args.partial_train)

    model = GaitNet_1(
        backbone=model_backbone, num_joints=args.num_joints, dim_rep=args.dim_rep,
        num_classes=args.EVGS_classes, dropout_ratio=args.dropout_ratio, 
        )
    loss_fn = Track1_Loss()

    if torch.cuda.is_available():
        # model = nn.DataParallel(model)
        model = model.cuda()

    best_acc = 0
    model_params = 0
    fit_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
        if parameter.requires_grad:
            fit_params = fit_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)
    print('INFO: Trainable parameter count in Fit epochs (after freezing):', fit_params)
    print('Loading dataset...')
    trainloader_params = {
          'batch_size': args.batch_size,
          'shuffle': True,
          'drop_last': True, 
          'num_workers': 8,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    valloader_params = {
          'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 8,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    data_path_train = 'dataset/train_dataset_track1.pkl'
    data_path_val = 'dataset/val_dataset_track1.pkl'
    train_dataset = GaitDataset_1(
        dataset_path=data_path_train,
        split='train',
        view=args.view,
        batch_frames=args.maxlen,
        flip=args.flip,
        swap_leg=args.swap_leg,
        random_move=args.random_move,
        scale_range=args.scale_range_train
    )
    train_loader = DataLoader(train_dataset, **trainloader_params)

    val_dataset = GaitDataset_1(
        dataset_path=data_path_val,
        split='val',
        view=args.view,
        batch_frames=args.maxlen,
        flip=False,
        swap_leg=False,
        random_move=False,
        scale_range=args.scale_range_test
    )
    val_loader = DataLoader(val_dataset, **valloader_params)
        
    chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
    if os.path.exists(chk_filename):
        opts.resume = chk_filename
    if opts.resume or opts.evaluate:
        chk_filename = opts.evaluate if opts.evaluate else opts.resume
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'], strict=True)
    
    if not opts.evaluate:

        if args.partial_train:
            backbone_pretrained_params = []
            backbone_scratch_params = [] 
            for name, param in model.backbone.named_parameters():
                if name == 'pos_embed':
                    backbone_scratch_params.append(param)
                else:
                    backbone_pretrained_params.append(param)
            
            optimizer_params = [
                  {"params": backbone_scratch_params, "lr": args.lr_head},
                  {"params": backbone_pretrained_params, "lr": args.lr_backbone},
                  {"params": model.head.parameters(), "lr": args.lr_head},
            ]
        else:
            optimizer_params = [
                  {"params": model.backbone.parameters(), "lr": args.lr_backbone},
                  {"params": model.head.parameters(), "lr": args.lr_head},
            ]

        optimizer = optim.AdamW(
            optimizer_params,      
            lr=args.lr_backbone, 
            weight_decay=args.weight_decay
        )

        # scheduler = StepLR(optimizer, step_size=2, gamma=args.lr_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)
        
        st = 0
        print('INFO: Training on {} batches'.format(len(train_loader)))
        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
            lr = checkpoint['lr']
            if 'best_acc' in checkpoint and checkpoint['best_acc'] is not None:
                best_acc = checkpoint['best_acc']
        # Training
        for epoch in range(st, args.epochs):
            print('Training epoch %d.' % epoch)
            losses_train = AverageMeter()
            acc_train = AverageMeter()
            batch_time = AverageMeter()
            data_time = AverageMeter()

            model.train()
            if args.finetune and epoch == args.fit_epochs:
                model_params = 0
                model = unfrozen_model(model)
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        model_params = model_params + parameter.numel()
                print('INFO: Trainable parameter count after unfrozen:', model_params)
                        
            end = time.time()
            iters = len(train_loader)
            for idx, (batch_kp, batch_gt) in enumerate(tqdm(train_loader, desc="Training on epoch")):    # (B, T, 18, 3)
                data_time.update(time.time() - end)
                batch_size = len(batch_kp)

                if torch.cuda.is_available():
                    batch_gt = batch_gt.cuda()
                    batch_kp = batch_kp.cuda()

                if torch.isnan(batch_kp).any() or torch.isinf(batch_kp).any():
                    print(f"WARNING: batch_kp contains NaN or inf at batch {idx}")
                    continue
                    
                if torch.all(batch_kp == 0):
                    print(f"WARNING: Skipping batch {idx} - contains all-zero data")
                    continue

                pred_logits = model(batch_kp)           # output shape: (B, 2, 17)
                optimizer.zero_grad()
                loss_train = loss_fn(pred_logits, batch_gt)
                losses_train.update(loss_train.item(), batch_size)
                loss_train.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()    

                acc = track_1_compute_acc(pred_logits, batch_gt)
                acc_train.update(acc, batch_size)
                batch_time.update(time.time() - end)
                end = time.time()
            
            loss_val, acc_val = validate(val_loader, model, loss_fn)
                
            train_writer.add_scalar('train_loss', losses_train.avg, epoch + 1)
            train_writer.add_scalar('train_acc', acc_train.avg, epoch + 1)
            train_writer.add_scalar('val_loss', loss_val, epoch + 1)
            train_writer.add_scalar('val_acc', acc_val, epoch + 1)
            
            scheduler.step()

            # Save latest checkpoint.
            if epoch == args.epochs - 1:
                chk_path = os.path.join(opts.checkpoint, 'latest_epoch.pth')
                print('Saving checkpoint to', chk_path)
                torch.save({
                    'epoch': epoch+1,
                    'lr': scheduler.get_last_lr(),
                    'optimizer': optimizer.state_dict(),
                    'model': model.state_dict(),
                    'best_acc' : best_acc
                }, chk_path)

            # 保存指标时看训练集acc
            if acc_val > best_acc and acc_val > 0.72: 
                print(f"New best acc: {acc_val:.4f} (previous best: {best_acc:.4f}), saving checkpoint.")
                best_chk_path = os.path.join(opts.checkpoint, f'{epoch+1}_epoch_save.pth')
                best_acc = acc_val
                print("save best checkpoint")
                torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'best_acc' : best_acc
                }, best_chk_path)

    # if opts.evaluate:
    #     test_loss, test_top1, test_top5 = validate(test_loader, model, criterion)
    #     print('Loss {loss:.4f} \t'
    #           'Acc@1 {top1:.3f} \t'
    #           'Acc@5 {top5:.3f} \t'.format(loss=test_loss, top1=test_top1, top5=test_top5))


def train_all_data(args, opts):
    print(args)
    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))
    model_backbone = load_backbone(args)
    if args.finetune:
        if opts.resume or opts.evaluate:
            pass
        else:
            chk_filename = os.path.join(opts.pretrained, opts.selection)
            print('Loading backbone', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)['model_pos']
            model_backbone = load_pretrained_weights(model_backbone, checkpoint)
    if args.partial_train:
        model_backbone = partial_train_layers(model_backbone, args.partial_train)

    model = GaitNet_1(
        backbone=model_backbone, num_joints=args.num_joints, dim_rep=args.dim_rep,
        num_classes=args.EVGS_classes, dropout_ratio=args.dropout_ratio, 
        )

    if torch.cuda.is_available():
        # model = nn.DataParallel(model)
        model = model.cuda()

    model_params = 0
    fit_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
        if parameter.requires_grad:
            fit_params = fit_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)
    print('INFO: Trainable parameter count in Fit epochs (after freezing):', fit_params)
    print('Loading dataset...')
    trainloader_params = {
          'batch_size': args.batch_size,
          'shuffle': True,
          'drop_last': True, 
          'num_workers': 8,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    
    data_path_train = 'dataset/train_dataset_track1_all.pkl'

    train_dataset = GaitDataset_1(
        dataset_path=data_path_train,
        split='train',
        view=args.view,
        batch_frames=args.maxlen,
        flip=args.flip,
        swap_leg=args.swap_leg,
        random_move=args.random_move,
        scale_range=args.scale_range_train
    )
    train_loader = DataLoader(train_dataset, **trainloader_params)
        
    chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
    if os.path.exists(chk_filename):
        opts.resume = chk_filename
    if opts.resume or opts.evaluate:
        chk_filename = opts.evaluate if opts.evaluate else opts.resume
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'], strict=True)
    
    if not opts.evaluate:

        if args.partial_train:
            backbone_pretrained_params = []
            backbone_scratch_params = [] 
            for name, param in model.backbone.named_parameters():
                if name == 'pos_embed':
                    backbone_scratch_params.append(param)
                else:
                    backbone_pretrained_params.append(param)
            
            optimizer_params = [
                  {"params": backbone_scratch_params, "lr": args.lr_head},
                  {"params": backbone_pretrained_params, "lr": args.lr_backbone},
                  {"params": model.head.parameters(), "lr": args.lr_head},
            ]
        else:
            optimizer_params = [
                  {"params": model.backbone.parameters(), "lr": args.lr_backbone},
                  {"params": model.head.parameters(), "lr": args.lr_head},
            ]

        optimizer = optim.AdamW(
            optimizer_params,      
            lr=args.lr_backbone, 
            weight_decay=args.weight_decay
        )

        # scheduler = StepLR(optimizer, step_size=2, gamma=args.lr_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)
        
        st = 0
        print('INFO: Training on {} batches'.format(len(train_loader)))
        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
            lr = checkpoint['lr']
            if 'best_acc' in checkpoint and checkpoint['best_acc'] is not None:
                best_acc = checkpoint['best_acc']
                
        # Training
        for epoch in range(st, args.epochs):
            print('Training epoch %d.' % epoch)
            losses_train = AverageMeter()
            cls_losses_train = AverageMeter()  
            nrmse_losses_train = AverageMeter() 
            center_losses_train = AverageMeter()
            acc_train = AverageMeter()
            batch_time = AverageMeter()
            data_time = AverageMeter()

            model.train()

            if args.finetune and epoch == args.fit_epochs:
                model = unfrozen_model(model)
            if epoch >= args.fit_epochs * 2:
                current_alpha, current_beta = 2.0, 2.0
            else:
                current_alpha, current_beta = 1.0, 5.0
            
            loss_fn = Track1_Loss(alpha=current_alpha, beta=current_beta)

            end = time.time()
            iters = len(train_loader)
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
            for idx, (batch_kp, batch_gt) in enumerate(pbar):    # (B, T, 18, 3)
                data_time.update(time.time() - end)
                batch_size = len(batch_kp)

                if torch.cuda.is_available():
                    batch_gt = batch_gt.cuda()
                    batch_kp = batch_kp.cuda()

                if torch.isnan(batch_kp).any() or torch.isinf(batch_kp).any():
                    print(f"WARNING: batch_kp contains NaN or inf at batch {idx}")
                    continue
                    
                if torch.all(batch_kp == 0):
                    print(f"WARNING: Skipping batch {idx} - contains all-zero data")
                    continue

                pred_logits = model(batch_kp)            # output shape: (B, 2, 17)
                optimizer.zero_grad()
                
                loss_train, loss_dict = loss_fn(pred_logits, batch_gt)
                
                losses_train.update(loss_train.item(), batch_size)
                cls_losses_train.update(loss_dict["cls_loss"], batch_size)
                nrmse_losses_train.update(loss_dict["nrmse"], batch_size)
                center_losses_train.update(loss_dict["center_loss"], batch_size)
                
                loss_train.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()    

                acc = track_1_compute_acc(pred_logits, batch_gt)
                acc_train.update(acc, batch_size)
                batch_time.update(time.time() - end)
                end = time.time()

                pbar.set_postfix({
                    'TotL': f"{losses_train.avg:.4f}",
                    'ClsL': f"{cls_losses_train.avg:.4f}",
                    'NRMSE': f"{nrmse_losses_train.avg:.4f}",
                    'CtrL': f"{center_losses_train.avg:.4f}", # 新增
                    'Acc': f"{acc_train.avg:.4f}"
                })

            train_writer.add_scalar('Train/Total_Loss', losses_train.avg, epoch + 1)
            train_writer.add_scalar('Train/Cls_Loss', cls_losses_train.avg, epoch + 1)
            train_writer.add_scalar('Train/NRMSE_Loss', nrmse_losses_train.avg, epoch + 1)
            train_writer.add_scalar('Train/Accuracy', acc_train.avg, epoch + 1)
            train_writer.add_scalar('Train/Center_Loss', center_losses_train.avg, epoch + 1)
            
            scheduler.step()
            
            if epoch < args.epochs // 3:
                chk_path = os.path.join(opts.checkpoint, f'{epoch+1}_epoch_save.pth')
                print('Saving checkpoint to', chk_path)
                torch.save({
                    'epoch': epoch+1,
                    'lr': scheduler.get_last_lr(),
                    'optimizer': optimizer.state_dict(),
                    'model': model.state_dict(),
                }, chk_path)


if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)

    if opts.split:
        print("split train/val data for training")
        train_split_data(args, opts)
    else:
        print("train on all data without split")
        train_all_data(args, opts)