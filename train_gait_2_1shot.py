import os
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
from lib.model.loss_supcon import SupConLoss
from pytorch_metric_learning import samplers
from lib.data.dataset_gait import GaitDataset_2
from lib.model.model_gait import GaitNet_2

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gait/MB_ft_gait_track2.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint/gait2', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint/gait1', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-freq', '--print_freq', default=5)
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('--split', action='store_true', default=False, help='whether split train/val dataset')
    opts = parser.parse_args()
    return opts


def extract_feats(dataloader_x, model):
    all_feats = []
    all_gts = []
    model.eval()
    with torch.no_grad():
        for idx, (batch_input, batch_gt) in tqdm(enumerate(dataloader_x), desc="Extracting Features"):
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()      
            
            feat = model(batch_input)
            B, L, D = feat.shape
            
            feat = feat.reshape(B * 2, D).cpu()
            batch_gt = batch_gt.reshape(B * 2).cpu()
            
            all_feats.append(feat)
            all_gts.append(batch_gt)
            
    all_feats = torch.cat(all_feats)
    all_gts = torch.cat(all_gts)
    return all_feats, all_gts

def validate_1shot(anchor_loader, test_loader, model):
    anchor_feats, anchor_labels = extract_feats(anchor_loader, model)
    test_feats, test_labels = extract_feats(test_loader, model)
    
    anchor_feats = anchor_feats.unsqueeze(1) # [M, 1, D]
    test_feats = test_feats.unsqueeze(0)     # [1, N, D]
    
    if torch.cuda.is_available():
        anchor_feats = anchor_feats.cuda()
        test_feats = test_feats.cuda()
        test_labels = test_labels.cuda()
        anchor_labels = anchor_labels.cuda()

    dis = F.cosine_similarity(anchor_feats, test_feats, dim=-1)
    
    pred = anchor_labels[torch.argmax(dis, dim=0)]
    assert len(pred) == len(test_labels)
    acc = sum(pred == test_labels).float() / len(pred)
    
    return acc.item()


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
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)['model']
            model_backbone = load_pretrained_weights(model_backbone, checkpoint)

    if args.partial_train:
        model_backbone = partial_train_layers(model_backbone, args.partial_train)
    if args.frozen_backbone and not args.partial_train:
        model_backbone = frozen_model(model_backbone)
        print("Backbone frozen, only training head layers.")

    model = GaitNet_2(
        backbone=model_backbone, num_joints=args.num_joints, dim_rep=args.dim_rep,
        num_evgs=args.EVGS_classes, num_classes=args.CP_classes, dropout_ratio=args.dropout_ratio, 
    )

    loss_fn = SupConLoss(temperature=0.1)

    if torch.cuda.is_available():
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    best_acc = 0
    print('Loading dataset...')

    data_path_train = 'dataset/train_dataset_track2.pkl'
    data_path_val = 'dataset/val_dataset_track2.pkl'
    train_dataset = GaitDataset_2(
        dataset_path=data_path_train, split='train', view=args.view,
        batch_frames=args.maxlen, flip=args.flip, swap_leg=args.swap_leg,
        random_move=args.random_move, scale_range=args.scale_range_train
    )
    val_dataset = GaitDataset_2(
        dataset_path=data_path_val, split='val', view=args.view,
        batch_frames=args.maxlen, flip=False, swap_leg=False,
        random_move=False, scale_range=args.scale_range_test
    )

    train_labels_for_sampler = [lbl[0].item() for lbl in train_dataset.labels]
    
    m_per_class = args.batch_size // 5
    if m_per_class == 0: m_per_class = 1
    
    sampler = samplers.MPerClassSampler(
        labels=train_labels_for_sampler, 
        m=m_per_class, 
        batch_size=args.batch_size, 
        length_before_new_iter=len(train_dataset)
    )

    trainloader_params = {
          'batch_size': args.batch_size,
          'shuffle': False,  # 使用 sampler 时必须为 False
          'drop_last': True, 
          'num_workers': 8,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True,
          'sampler': sampler
    }
    valloader_params = {
          'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 8,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    
    train_loader = DataLoader(train_dataset, **trainloader_params)
    val_loader = DataLoader(val_dataset, **valloader_params)
    
    anchor_dataset = GaitDataset_2(
        dataset_path=data_path_train, split='val', view=args.view,
        batch_frames=args.maxlen, flip=False, swap_leg=False,
        random_move=False, scale_range=args.scale_range_test
    )
    anchor_loader = DataLoader(anchor_dataset, **valloader_params)
        
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
            optimizer_params = model.parameters()

        optimizer = optim.AdamW(
            optimizer_params,      
            lr=args.lr_backbone, 
            weight_decay=args.weight_decay
        )

        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)
        st = 0

        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'best_acc' in checkpoint and checkpoint['best_acc'] is not None:
                best_acc = checkpoint['best_acc']

        # Training
        for epoch in range(st, args.epochs):
            print('Training epoch %d.' % epoch)
            losses_train = AverageMeter()
            batch_time = AverageMeter()
            data_time = AverageMeter()

            model.train()
            if args.finetune and epoch == args.fit_epochs:
                model = unfrozen_model(model)
                print(f"Unfreezing all layers for fine-tuning.")

            end = time.time()
            for idx, (batch_kp, batch_gt) in enumerate(tqdm(train_loader, desc="Training")):
                data_time.update(time.time() - end)
                
                if torch.cuda.is_available():
                    batch_gt = batch_gt.cuda()
                    batch_kp = batch_kp.cuda()

                if torch.isnan(batch_kp).any() or torch.isinf(batch_kp).any():
                    continue

                feat = model(batch_kp) 
                B, L, D = feat.shape
                feat_left = feat[:, 0, :]  
                feat_right = feat[:, 1, :]  
                same_label_mask = (batch_gt[:, 0] == batch_gt[:, 1])
                consistency_loss = 0.0
                if same_label_mask.any():
                    cos_sim = F.cosine_similarity(feat_left[same_label_mask], feat_right[same_label_mask], dim=-1)
                    consistency_loss = (1.0 - cos_sim).mean()
                feat_flat = feat.reshape(B * 2, D).unsqueeze(1)   
                batch_gt_flat = batch_gt.reshape(B * 2)         
                
                optimizer.zero_grad()
                supcon_loss = loss_fn(feat_flat, batch_gt_flat)
                gamma = 0.5 
                loss_train = supcon_loss + gamma * consistency_loss
                
                losses_train.update(loss_train.item(), B*L)
                loss_train.backward()
                optimizer.step()    

                batch_time.update(time.time() - end)
                end = time.time()
                
            acc_val = validate_1shot(anchor_loader, val_loader, model)
                
            train_writer.add_scalar('train_loss_supcon', losses_train.avg, epoch + 1)
            train_writer.add_scalar('val_1shot_acc', acc_val, epoch + 1)
            
            print(f"Epoch {epoch}: Train Loss {losses_train.avg:.4f}, Val 1-shot Acc {acc_val:.4f}")
            scheduler.step()

            if acc_val > best_acc and acc_val > 0.30:
                print(f"New best acc: {acc_val:.4f} (previous best: {best_acc:.4f}), saving checkpoint.")
                best_chk_path = os.path.join(opts.checkpoint, f'{epoch+1}_epoch_save.pth')
                best_acc = acc_val
                torch.save({'epoch': epoch+1, 'optimizer': optimizer.state_dict(),
                            'model': model.state_dict(), 'best_acc' : best_acc}, best_chk_path)


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
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)['model']
            model_backbone = load_pretrained_weights(model_backbone, checkpoint)

    if args.partial_train:
        model_backbone = partial_train_layers(model_backbone, args.partial_train)
    if args.frozen_backbone and not args.partial_train:
        model_backbone = frozen_model(model_backbone)
        print("Backbone frozen, only training head layers.")

    model = GaitNet_2(
        backbone=model_backbone, num_joints=args.num_joints, dim_rep=args.dim_rep,
        num_evgs=args.EVGS_classes, num_classes=args.CP_classes, dropout_ratio=args.dropout_ratio, 
    )

    # 监督对比损失
    loss_fn = SupConLoss(temperature=0.1)

    if torch.cuda.is_available():
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    best_acc = 0
    print('Loading all dataset without split...')
    
    data_path_train = 'dataset/train_dataset_track2_all.pkl'
    train_dataset = GaitDataset_2(
        dataset_path=data_path_train, split='train', view=args.view,
        batch_frames=args.maxlen, flip=args.flip, swap_leg=args.swap_leg,
        random_move=args.random_move, scale_range=args.scale_range_train
    )

    train_labels_for_sampler = [lbl[0].item() for lbl in train_dataset.labels]
    m_per_class = args.batch_size // 5
    if m_per_class == 0: m_per_class = 1
    
    sampler = samplers.MPerClassSampler(
        labels=train_labels_for_sampler, 
        m=m_per_class, 
        batch_size=args.batch_size, 
        length_before_new_iter=len(train_dataset)
    )

    trainloader_params = {
          'batch_size': args.batch_size,
          'shuffle': False,  # 必须设为 False，由 sampler 决定抽取逻辑
          'drop_last': True, 
          'num_workers': 8,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True,
          'sampler': sampler
    }
    
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
            optimizer_params = model.parameters()

        optimizer = optim.AdamW(
            optimizer_params,      
            lr=args.lr_backbone, 
            weight_decay=args.weight_decay
        )

        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)
        st = 0

        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'best_acc' in checkpoint and checkpoint['best_acc'] is not None:
                best_acc = checkpoint['best_acc']

        for epoch in range(st, args.epochs):
            print('Training epoch %d on ALL data.' % epoch)
            losses_train = AverageMeter()
            batch_time = AverageMeter()
            end = time.time()

            model.train()
            if args.finetune and epoch == args.fit_epochs:
                model = unfrozen_model(model)
                print(f"Unfreezing all layers for fine-tuning.")

            for idx, (batch_kp, batch_gt) in enumerate(tqdm(train_loader, desc="Training")):
                if torch.cuda.is_available():
                    batch_gt = batch_gt.cuda()
                    batch_kp = batch_kp.cuda()

                if torch.isnan(batch_kp).any() or torch.isinf(batch_kp).any():
                    continue

                feat = model(batch_kp)
                B, L, D = feat.shape
                feat_left = feat[:, 0, :]  
                feat_right = feat[:, 1, :]  
                same_label_mask = (batch_gt[:, 0] == batch_gt[:, 1])
                consistency_loss = 0.0
                if same_label_mask.any():
                    cos_sim = F.cosine_similarity(feat_left[same_label_mask], feat_right[same_label_mask], dim=-1)
                    consistency_loss = (1.0 - cos_sim).mean()
                feat_flat = feat.reshape(B * 2, D).unsqueeze(1)   
                batch_gt_flat = batch_gt.reshape(B * 2)         
                
                optimizer.zero_grad()
                supcon_loss = loss_fn(feat_flat, batch_gt_flat)
                gamma = 0.5 
                loss_train = supcon_loss + gamma * consistency_loss
                
                losses_train.update(loss_train.item(), B*L)
                loss_train.backward()
                optimizer.step()    

                batch_time.update(time.time() - end)
                end = time.time()
                
            train_writer.add_scalar('train_loss_supcon', losses_train.avg, epoch + 1)
            print(f"Epoch {epoch}: Train Loss {losses_train.avg:.4f}")
            scheduler.step()

            # Save latest checkpoint.
            if epoch == args.epochs - 1:
                chk_path = os.path.join(opts.checkpoint, 'latest_epoch.pth')
                torch.save({'epoch': epoch+1, 'optimizer': optimizer.state_dict(),
                            'model': model.state_dict(), 'best_acc': best_acc}, chk_path)
            if epoch < 30:  
                best_chk_path = os.path.join(opts.checkpoint, f'{epoch+1}_epoch_save.pth')
                print(f"Saving checkpoint to {best_chk_path}")
                torch.save({'epoch': epoch+1, 'optimizer': optimizer.state_dict(),
                            'model': model.state_dict(), 'best_acc': best_acc}, best_chk_path)
                

if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)

    if opts.split:
        print("split train/val data for training")
        train_split_data(args, opts)
    else:
        print("train on all data without split")
        train_all_data(args, opts)