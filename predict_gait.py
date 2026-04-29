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
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from lib.utils.tools import *
from lib.utils.learning import *
from lib.model.loss import *
from lib.data.dataset_gait import GaitDataset_1, GaitDataset_2
from lib.model.model_gait import GaitNet_1, GaitNet_2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Gait Recognition')
    # parser.add_argument('--track', type=int, required=True, help='choose track 1 or track 2 to predict')
    parser.add_argument('--model_1', type=str, default=None, help='path to the trained track 1 model')
    parser.add_argument('--model_2', type=str, default=None, help='path to the trained track 2 model')
    parser.add_argument('--test_dir', type=str, default='dataset', help='path to the output csv file')
    parser.add_argument('--out_path', type=str, default='output.csv', help='path to the output csv file')
    parser.add_argument('--vote', action='store_true', default=False, help='Use voting for final prediction')
    args = parser.parse_args()
    return args


def inference_track1(opts, args):
    print("load model from {opts.model_1}")
    assert os.path.isfile(opts.model_1), "Model file does not exist."
    assert 'gait1' in opts.model_1, "Model path should contain 'gait1' for track 1."

    model_backbone = load_backbone(args)
    model = GaitNet_1(backbone=model_backbone, num_joints=args.num_joints, dim_rep=args.dim_rep,
                        num_classes=args.EVGS_classes, dropout_ratio=args.dropout_ratio)
    checkpoint = torch.load(opts.model_1, map_location=lambda storage, loc: storage)['model']
    model_backbone = load_pretrained_weights(model, checkpoint)
    if torch.cuda.is_available():
        model = model.cuda()

    test_set = [4, 5, 18, 26, 28, 40, 42, 43, 47, 48, 53, 54, 72, 78, 83, 85]
    testloader_params = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': 8,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True
    }

    all_predictions = {}
    model.eval()

    for p in tqdm(test_set, desc='Inference on Track 1'):
        data_path = os.path.join(opts.test_dir, f'test_track1_{p}.pkl')
        test_dataset = GaitDataset_1(
        dataset_path=data_path,
        split='test',
        batch_frames=args.maxlen,
        flip=False,
        swap_leg=False,
        random_move=False,
        scale_range=args.scale_range_test,
    )
        test_loader = DataLoader(test_dataset, **testloader_params)

        probs = []
        with torch.no_grad():
            for idx, batch_data in enumerate(test_loader):
                if isinstance(batch_data, list) or isinstance(batch_data, tuple):
                    batch_kp = batch_data[0]
                else:
                    batch_kp = batch_data
                # print(f"Processing batch {idx+1}/{len(test_loader)} with size {len(batch_kp)}")
                if torch.cuda.is_available():
                    batch_kp = batch_kp.cuda()
                
                pred_logits = model(batch_kp)   # (B, 2, 17)
                prob = torch.sigmoid(pred_logits)
                probs.append(prob)

        if opts.vote:
            all_probs = torch.cat(probs, dim=0)   # (N, 2, 17)
            N = all_probs.shape[0]
            
            # 先计算出每个序列独立的结果
            seq_preds = (all_probs > 0.5).int()   # (N, 2, 17)
            
            # 按投票的方式决定最终标签
            votes = seq_preds.sum(dim=0)      
            pred_labels = torch.zeros((2, 17), dtype=torch.int32, device=all_probs.device)
            pred_labels[votes > N / 2.0] = 1
            pred_labels[votes < N / 2.0] = 0
            
            # 如果票数相同（平局），用平均概率判定
            if N % 2 == 0:
                tie_mask = (votes == N // 2)
                mean_probs = all_probs.mean(dim=0)
                pred_labels[tie_mask] = (mean_probs[tie_mask] > 0.5).int()
                
            all_predictions[p] = pred_labels.cpu().numpy()
        else:
            prob_avg = torch.cat(probs, dim=0).mean(dim=0)
            pred_labels = (prob_avg > 0.5).cpu().numpy().astype(int)
            all_predictions[p] = pred_labels

    return all_predictions


def inference_track2(opts, args):
    print(f"load model from {opts.model_2}")
    assert os.path.isfile(opts.model_2), "Model file does not exist."
    assert 'gait2' in opts.model_2, "Model path should contain 'gait2' for track 2."

    model_backbone = load_backbone(args)
    model = GaitNet_2(backbone=model_backbone, num_joints=args.num_joints, dim_rep=args.dim_rep,
                        num_evgs=args.EVGS_classes, num_classes=args.CP_classes, dropout_ratio=args.dropout_ratio)
    checkpoint = torch.load(opts.model_2, map_location=lambda storage, loc: storage)['model']
    
    model.load_state_dict(checkpoint, strict=True)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    train_data_path = 'dataset/train_dataset_track2_all.pkl'
        
    anchor_dataset = GaitDataset_2(
        dataset_path=train_data_path, split='val', view=args.view,
        batch_frames=args.maxlen, flip=False, swap_leg=False,
        random_move=False, scale_range=args.scale_range_test
    )
    anchor_loader = DataLoader(anchor_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    anchor_feats, anchor_labels = [], []
    with torch.no_grad():
        for batch_kp, batch_gt in anchor_loader:
            if torch.cuda.is_available(): batch_kp = batch_kp.cuda()
            feat = model(batch_kp)   
            B, L, D = feat.shape
            feat = feat.reshape(B * 2, D).cpu()
            batch_gt = batch_gt.reshape(B * 2).cpu()
            anchor_feats.append(feat)
            anchor_labels.append(batch_gt)
            
    anchor_feats = torch.cat(anchor_feats) 
    anchor_labels = torch.cat(anchor_labels) 

    test_set = [4, 6, 7, 13, 26, 35, 39, 42, 50]
    testloader_params = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': 8,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True
    }

    all_predictions = {}

    for p in tqdm(test_set, desc='Inference on Track 2'):
        data_path = os.path.join(opts.test_dir, f'test_track2_{p}.pkl')
        test_dataset = GaitDataset_2(
            dataset_path=data_path, split='test', batch_frames=args.maxlen,
            flip=False, swap_leg=False, random_move=False, scale_range=args.scale_range_test
        )
        test_loader = DataLoader(test_dataset, **testloader_params)

        test_split_feats = []
        with torch.no_grad():
            for idx, batch_data in enumerate(test_loader):
                if isinstance(batch_data, list) or isinstance(batch_data, tuple):
                    batch_kp = batch_data[0]
                else:
                    batch_kp = batch_data

                if torch.cuda.is_available():
                    batch_kp = batch_kp.cuda()
                
                feat = model(batch_kp)  
                test_split_feats.append(feat.cpu())  

        all_feats = torch.cat(test_split_feats, dim=0)  

        if opts.vote:
            pred_labels_list = []
            for side in range(2): 
                side_feats = all_feats[:, side, :] 
                
                dist = F.cosine_similarity(side_feats.unsqueeze(1), anchor_feats.unsqueeze(0), dim=-1) # (N_splits, M)
                
                nearest_anchor_idx = torch.argmax(dist, dim=1) 
                side_preds = anchor_labels[nearest_anchor_idx]
                
                # 开始投票找出最多的标签
                counts = torch.bincount(side_preds, minlength=5) 
                final_label = torch.argmax(counts).item()
                pred_labels_list.append(final_label)
                
            pred_labels = np.array(pred_labels_list)
            all_predictions[p] = pred_labels
        else:
            # 不投票形式：先把该视频的所有切片特征平均聚合，然后再去找最近的一个Anchor
            avg_feat = all_feats.mean(dim=0)    # (2, D)
            avg_feat = F.normalize(avg_feat, dim=-1)
            
            dist_left = F.cosine_similarity(avg_feat[0:1], anchor_feats, dim=-1) # (M,)
            dist_right = F.cosine_similarity(avg_feat[1:2], anchor_feats, dim=-1) # (M,)
            
            pred_left = anchor_labels[torch.argmax(dist_left)].item()
            pred_right = anchor_labels[torch.argmax(dist_right)].item()
            
            all_predictions[p] = np.array([pred_left, pred_right])

    return all_predictions


def export_to_csv(predict_1, predict_2, out_path):
    header = ["ID", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11", "L12", "L13", "L14", "L15", "L16", "L17",
              "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11", "R12", "R13", "R14", "R15", "R16", "R17",
              "Total", "Left_gait_subtype", "Right_gait_subtype"]

    track1_ids = [4, 5, 18, 26, 28, 40, 42, 43, 47, 48, 53, 54, 72, 78, 83, 85]
    track2_ids = [4, 6, 7, 13, 26, 35, 39, 42, 50]

    type_map = {
        0: "WNL",
        1: "type1",
        2: "type2",
        3: "type3",
        4: "type4"
    }

    print(f"Exporting results to {out_path}...")
    with open(out_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for pid in sorted(track1_ids):
            row = [f"track1-{pid}"]
            
            if predict_1 is not None and pid in predict_1:
                pred = predict_1[pid]
                left_vals = pred[0].tolist()
                right_vals = pred[1].tolist()
                
                total = sum(left_vals) + sum(right_vals)
                
                row.extend(left_vals)      
                row.extend(right_vals)     
                row.append(total)         
                row.extend([-1, -1])       
            else:
                raise ValueError(f"Missing prediction for Track 1 ID {pid}")
                
            writer.writerow(row)

        for pid in sorted(track2_ids):
            row = [f"track2-{pid}"]
            
            row.extend([-1] * 34)
            row.append(-1)
            
            if predict_2 is not None and pid in predict_2:
                pred = predict_2[pid]
                
                left_type = type_map[pred[0]]
                right_type = type_map[pred[1]]
                
                row.extend([left_type, right_type])
            else:
                raise ValueError(f"Missing prediction for Track 2 ID {pid}")
                
            writer.writerow(row)
            
    print("CSV Export Complete!")



if __name__ == '__main__':
    opts = parse_args()

    if opts.model_1 is not None:
        config_path_1 = 'configs/gait/MB_ft_gait_track1.yaml'
        args_1 = get_config(config_path_1)
        predict_1 = inference_track1(opts, args_1)

    if opts.model_2 is not None:
        config_path_2 = 'configs/gait/MB_ft_gait_track2.yaml'
        args_2 = get_config(config_path_2)
        predict_2 = inference_track2(opts, args_2)

    export_to_csv(predict_1, predict_2, opts.out_path)
