import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial


class PreLayer(nn.Module):
    """
    为了使用MotionBERT权重, 需要把输入的18个关键点映射为17个,
    !!!这里只是先做一个尝试, 效果不行的话就修改DSTformer的输入为18
    """
    def __init__(self, input_joints=18, output_joints=17):
        super().__init__()
        self.mapper = nn.Linear(input_joints, output_joints, bias=False)
        # self.norm = nn.LayerNorm(output_joints)
        nn.init.normal_(self.mapper.weight, mean=0.0, std=1.0 / input_joints)

    def forward(self, x):
        """
        Input:
         - (B, T, input_joints, 3)
         Output:
         - (B, T, output_joints, 3)
        """
        x = x.permute(0, 1, 3, 2)    # (B, T, 2, input_joints)
        x = self.mapper(x)           # (B, T, 2, output_joints)
        x = x.permute(0, 1, 3, 2)
        return x


class HeadClassification(nn.Module):
    def __init__(self, dropout_ratio=0., dim_rep=512, num_classes=17, num_joints=17, hidden_dim=512):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.bn = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim_rep * num_joints, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2 * num_classes)   
        
    def forward(self, feat):
        '''
            Input: (B, T, J, C)
            Output: (B, num_classes)
        '''
        B, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.mean(dim=1)      # (B, J, C)
        feat = feat.reshape(B, -1)   # (B, J*C)
        feat = self.fc1(feat)
        feat = self.bn(feat)
        feat = self.relu(feat)    
        feat = self.fc2(feat)
        return feat.reshape(B, 2, self.num_classes)   # (B, 2, num_classes), 先左后右
    

class HeadRepresentation(nn.Module):
    def __init__(self, dropout_ratio=0., dim_rep=512, num_joints=18, hidden_dim=128, out_dim=128):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.bn = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim_rep * num_joints, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2 * out_dim)
        self.out_dim = out_dim
        
    def forward(self, feat):
        '''
            Input: (B, T, J, C)
            Output: (B, 2, out_dim) - 左右腿独立的高维特征特征 
        '''
        B, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.mean(dim=1)      # (B, J, C)
        feat = feat.reshape(B, -1)   # (B, J*C)
        feat = self.fc1(feat)
        feat = self.bn(feat)
        feat = self.relu(feat)    
        feat = self.fc2(feat)
        feat = feat.reshape(B, 2, self.out_dim)
        feat = F.normalize(feat, dim=-1)
        return feat


class GaitNet_1(nn.Module):
    def __init__(
            self, backbone, num_joints=18, dim_rep=512, 
            num_classes=17, dropout_ratio=0.0, 
        ):
        super().__init__()
        self.backbone = backbone
        self.dim_rep = dim_rep
        self.dropout_ratio = dropout_ratio

        # self.pre_layer = PreLayer(input_joints=num_joints, output_joints=17)
        self.head = HeadClassification(
            dropout_ratio=dropout_ratio, dim_rep=dim_rep, 
            num_classes=num_classes, num_joints=num_joints, hidden_dim=128
        )

    def forward(self, x, return_feat=False):
        """
        Input:
         - (B, T, num_joints, 2)
        Output:
         - (B, 2, num_classes)
        """
        # x = self.pre_layer(x)    # (B, T, 17, 3)
        feat = self.backbone(x, return_rep=True)  # (B, T, 18, hidden_dim)
        if return_feat:
            return feat
        out = self.head(feat)    # (B, 2, 17)
        return out
    

class GaitNet_2(nn.Module):
    def __init__(self, backbone, dim_rep, num_joints=18, num_evgs=17, num_classes=5, dropout_ratio=0.0):
        super().__init__()
        self.evgs_net = backbone
        
        if hasattr(self.evgs_net, 'head'):
            print("INFO: deleting head from evgs_net")
            del self.evgs_net.head

        self.cp_classifier = HeadRepresentation(
            dropout_ratio=dropout_ratio, 
            dim_rep=dim_rep, 
            num_joints=num_joints, 
            hidden_dim=128, 
            out_dim=128 
        )

    def forward(self, x):
        """
        Input:
         - (B, T, num_joints, 2)
        Output:
         - (B, 2, out_dim)  
        """
        feat = self.evgs_net(x, return_feat=True)   # (B, T, 18, C)
        out = self.cp_classifier(feat)    # (B, 2, 128)
        return out
