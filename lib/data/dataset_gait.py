import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import copy
import json
from collections import defaultdict
from lib.utils.utils_data import crop_scale, flip_data, resample, split_clips
from lib.utils.tools import read_pkl


def random_move_gait(data_tensor,
                     angle_range=[-2., 2.],
                     scale_range=[0.9, 1.1],
                     transform_range=[-0.1, 0.1],
                     move_time_candidate=[1]):
    """
    针对单人序列的随机增强 (PyTorch Tensor 版本)
    输入形状 EXPECTED: (T, J, C) -> 例如 (243, 18, 3) 且为 torch.Tensor
    返回形状: (T, J, C) torch.Tensor
    """
    data_tensor = data_tensor.clone()
    T, J, C = data_tensor.shape
    
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)
    
    A = np.random.uniform(angle_range[0], angle_range[1], num_node)
    S = np.random.uniform(scale_range[0], scale_range[1], num_node)
    T_x = np.random.uniform(transform_range[0], transform_range[1], num_node)
    T_y = np.random.uniform(transform_range[0], transform_range[1], num_node)
    
    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)
    
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1], node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1], node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1], node[i + 1] - node[i])
        
    theta_np = np.array([[np.cos(a) * s, -np.sin(a) * s],
                         [np.sin(a) * s, np.cos(a) * s]])  
    theta = torch.tensor(theta_np, dtype=torch.float32, device=data_tensor.device).permute(2, 0, 1)
    
    t_x = torch.tensor(t_x, dtype=torch.float32, device=data_tensor.device)
    t_y = torch.tensor(t_y, dtype=torch.float32, device=data_tensor.device)
                      
    for i_frame in range(T):
        xy = data_tensor[i_frame, :, 0:2].t()
        new_xy = torch.matmul(theta[i_frame], xy)
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_tensor[i_frame, :, 0:2] = new_xy.t()
        
    return data_tensor


class GaitDataset_1(Dataset):
    def __init__(self, dataset_path, split='train', view='all', batch_frames=243, flip=False, swap_leg=False, random_move=True, scale_range=[0.25, 1]):
        super().__init__()
        self.flip = flip
        self.swap_leg = swap_leg
        self.random_move = random_move
        self.scale_range = scale_range
        self.split = split
        self.view = view

        dataset = read_pkl(dataset_path)   

        self.motions = []
        self.labels = []
        self.view_total_frames = 0
        for name, data in dataset.items():
            if self.view != 'all' and self.view.lower() not in name:
                continue
            self.view_total_frames += data['total_frames']
        
        for name, data in dataset.items():

            if self.view != 'all' and self.view.lower() not in name:
                print(f"Skipping {name} due to view filter.")
                continue

            total_frames = data['total_frames']
            keypoints = data['keypoints']   
            conf = data['keypoint_scores'][..., None]   
            motion = np.concatenate([keypoints, conf], axis=-1)   
            
            if 'obj_ids' in data:
                obj_ids = data['obj_ids']
                if len(obj_ids) > 0:
                    unique_ids, counts = np.unique(obj_ids, return_counts=True)
                    most_freq_id = unique_ids[np.argmax(counts)]
                    if most_freq_id != -1 and obj_ids[0] != -1:
                        valid_idx = np.where(obj_ids == most_freq_id)[0]
                        if len(valid_idx) > 0 and len(valid_idx) < total_frames:
                            for idx in range(motion.shape[1]):
                                for dim in range(motion.shape[2]):
                                    motion[:, idx, dim] = np.interp(
                                        np.arange(total_frames),
                                        valid_idx,
                                        motion[valid_idx, idx, dim]
                                    )
            # 归一化2d关键点
            motion = crop_scale(motion, self.scale_range)    

            if self.split == 'test':
                label = torch.zeros((2, 17), dtype=torch.float32)
            
            else:
                label_dict = data['label']
                left_label = [label_dict['left'][str(i)] for i in range(1, 18)]
                right_label = [label_dict['right'][str(i)] for i in range(1, 18)]
                left_label, right_label = torch.tensor(left_label, dtype=torch.float32), torch.tensor(right_label, dtype=torch.float32)  # (17,)
                label = torch.stack([left_label, right_label], dim=0)   # (2, 17)  

            # 统一序列长度
            self.batchable_frames(motion, label, total_frames, batch_frames)

    def __len__(self):
        return len(self.motions)
    
    def __getitem__(self, idx):
        motion = torch.tensor(self.motions[idx], dtype=torch.float32)
        if self.split == 'test':
            return motion, self.view_total_frames
        
        label = self.labels[idx]
        if self.flip and random.random() > 0.5:
            motion = flip_data(motion)
            flipped_label = label[[1, 0]]   
            label = flipped_label
        
        if self.swap_leg and self.split == 'train' and random.random() > 0.5:
            left_leg = [6, 8, 10, 12, 13, 14]
            right_leg = [7, 9, 11, 15, 16, 17]
            temp = motion[:, left_leg, :].clone()
            motion[:, left_leg, :] = motion[:, right_leg, :]
            motion[:, right_leg, :] = temp

        if self.random_move and self.split == 'train':
            motion = random_move_gait(motion)

        return motion, label

    def batchable_frames(self, motion, label, total_frames, batch_frames):
        """
        保证每段序列长度相同, 用于batch训练
        """
        num_splits = total_frames // batch_frames
        remainder = total_frames % batch_frames

        for split_idx in range(num_splits):
            start_frame = split_idx * batch_frames
            end_frame = start_frame + batch_frames

            split_motion = motion[start_frame:end_frame]   
            self.motions.append(split_motion)
            self.labels.append(label)

        if remainder > batch_frames // 4:
            leftover = motion[-remainder:]     
            reversed_leftover = leftover[::-1]  
            pad_list = []
            current_len = 0
            forward = True
            
            while current_len < batch_frames:
                pad_list.append(leftover if forward else reversed_leftover)
                current_len += remainder
                forward = not forward
                
            padding_motion = np.concatenate(pad_list, axis=0)
            self.motions.append(padding_motion[:batch_frames])   
            self.labels.append(label)
        
        if num_splits == 0 and remainder < batch_frames // 4:
            print(f"Warning: Sequence length {total_frames} is too short for batch_frames {batch_frames}. Skipping this sequence.")


class GaitDataset_2(Dataset):
    def __init__(self, dataset_path, split='train', view='all', batch_frames=243, flip=True, swap_leg=False, random_move=True, scale_range=[0.25, 1]):
        super().__init__()
        self.flip = flip
        self.swap_leg = swap_leg
        self.random_move = random_move
        self.scale_range = scale_range
        self.split = split
        self.view = view

        self.label_map = {
            "WNL": 0,
            "type1": 1,
            "type2": 2,
            "type3": 3,
            "type4": 4
        }
        dataset = read_pkl(dataset_path)  

        self.motions = []
        self.labels = []
        self.view_total_frames = 0
        for name, data in dataset.items():
            if self.view != 'all' and self.view.lower() not in name:
                continue
            self.view_total_frames += data['total_frames']
        
        for name, data in dataset.items():

            if self.view != 'all' and self.view.lower() not in name:
                print(f"Skipping {name} due to view filter.")
                continue

            total_frames = data['total_frames']
            keypoints = data['keypoints']    # [T, 18, 2]
            conf = data['keypoint_scores'][..., None]   # [T, 18]
            motion = np.concatenate([keypoints, conf], axis=-1)    # [T, 18, 3]
            # 归一化2d关键点
            motion = crop_scale(motion, self.scale_range)     # [T, 18, 2]

            if self.split == 'test':
                label = torch.zeros((2,), dtype=torch.long)

            else:
                label_dict = data['label']
                left_str = label_dict['left']['gait_subtype']
                right_str = label_dict['right']['gait_subtype']
                left_idx = self.label_map[left_str]
                right_idx = self.label_map[right_str]
                label = torch.tensor([left_idx, right_idx], dtype=torch.long)   # (2,)

            # 统一序列长度
            self.batchable_frames(motion, label, total_frames, batch_frames)

    def __len__(self):
        return len(self.motions)
    
    def __getitem__(self, idx):
        motion = torch.tensor(self.motions[idx], dtype=torch.float32)
        if self.split == 'test':
            return motion, self.view_total_frames
        
        label = self.labels[idx]
        if self.flip and self.split == 'train' and random.random() > 0.5:
            motion = flip_data(motion)
            flipped_label = torch.tensor([label[1], label[0]], dtype=torch.long)
            label = flipped_label

        if self.swap_leg and self.split == 'train' and random.random() > 0.5:
            left_leg = [6, 8, 10, 12, 13, 14]
            right_leg = [7, 9, 11, 15, 16, 17]
            temp = motion[:, left_leg, :].clone()
            motion[:, left_leg, :] = motion[:, right_leg, :]
            motion[:, right_leg, :] = temp

        if self.random_move and self.split == 'train':
            motion = random_move_gait(motion)

        return motion, label

    def batchable_frames(self, motion, label, total_frames, batch_frames):
        """
        保证每段序列长度相同, 用于batch训练
        """
        num_splits = total_frames // batch_frames
        remainder = total_frames % batch_frames

        for split_idx in range(num_splits):
            start_frame = split_idx * batch_frames
            end_frame = start_frame + batch_frames

            split_motion = motion[start_frame:end_frame]   # [batch_frames, 18, 3]
            self.motions.append(split_motion)
            self.labels.append(label)

        if remainder > batch_frames // 4:
            leftover = motion[-remainder:]     
            reversed_leftover = leftover[::-1]  
            pad_list = []
            current_len = 0
            forward = True

            while current_len < batch_frames:
                pad_list.append(leftover if forward else reversed_leftover)
                current_len += remainder
                forward = not forward
                
            padding_motion = np.concatenate(pad_list, axis=0)
            self.motions.append(padding_motion[:batch_frames])   
            self.labels.append(label)
        
        if num_splits == 0 and remainder < batch_frames // 4:
            print(f"Warning: Sequence length {total_frames} is too short for batch_frames {batch_frames}. Skipping this sequence.")
