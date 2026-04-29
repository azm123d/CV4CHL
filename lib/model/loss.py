import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# Numpy-based errors

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return np.mean(np.linalg.norm(predicted - target, axis=len(target.shape)-1), axis=1)

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1), axis=1)


# PyTorch-based errors (for losses)

def loss_mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))
    
def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))

def loss_2d_weighted(predicted, target, conf):
    assert predicted.shape == target.shape
    predicted_2d = predicted[:,:,:,:2]
    target_2d = target[:,:,:,:2]
    diff = (predicted_2d - target_2d) * conf
    return torch.mean(torch.norm(diff, dim=-1))
    
def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return loss_mpjpe(scale * predicted, target)

def weighted_bonelen_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.001 * torch.pow(predict_3d_length - gt_3d_length, 2).mean()
    return loss_length

def weighted_boneratio_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.1 * torch.pow((predict_3d_length - gt_3d_length)/gt_3d_length, 2).mean()
    return loss_length

def get_limb_lens(x):
    '''
        Input: (N, T, 17, 3)
        Output: (N, T, 16)
    '''
    limbs_id = [[0,1], [1,2], [2,3],
         [0,4], [4,5], [5,6],
         [0,7], [7,8], [8,9], [9,10],
         [8,11], [11,12], [12,13],
         [8,14], [14,15], [15,16]
        ]
    limbs = x[:,:,limbs_id,:]
    limbs = limbs[:,:,:,0,:]-limbs[:,:,:,1,:]
    limb_lens = torch.norm(limbs, dim=-1)
    return limb_lens

def loss_limb_var(x):
    '''
        Input: (N, T, 17, 3)
    '''
    if x.shape[1]<=1:
        return torch.FloatTensor(1).fill_(0.)[0].to(x.device)
    limb_lens = get_limb_lens(x)
    limb_lens_var = torch.var(limb_lens, dim=1)
    limb_loss_var = torch.mean(limb_lens_var)
    return limb_loss_var

def loss_limb_gt(x, gt):
    '''
        Input: (N, T, 17, 3), (N, T, 17, 3)
    '''
    limb_lens_x = get_limb_lens(x)
    limb_lens_gt = get_limb_lens(gt) # (N, T, 16)
    return nn.L1Loss()(limb_lens_x, limb_lens_gt)

def loss_velocity(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    if predicted.shape[1]<=1:
        return torch.FloatTensor(1).fill_(0.)[0].to(predicted.device)
    velocity_predicted = predicted[:,1:] - predicted[:,:-1]
    velocity_target = target[:,1:] - target[:,:-1]
    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=-1))

def loss_joint(predicted, target):
    assert predicted.shape == target.shape
    return nn.L1Loss()(predicted, target)

def get_angles(x):
    '''
        Input: (N, T, 17, 3)
        Output: (N, T, 16)
    '''
    limbs_id = [[0,1], [1,2], [2,3],
         [0,4], [4,5], [5,6],
         [0,7], [7,8], [8,9], [9,10],
         [8,11], [11,12], [12,13],
         [8,14], [14,15], [15,16]
        ]
    angle_id = [[ 0,  3],
                [ 0,  6],
                [ 3,  6],
                [ 0,  1],
                [ 1,  2],
                [ 3,  4],
                [ 4,  5],
                [ 6,  7],
                [ 7, 10],
                [ 7, 13],
                [ 8, 13],
                [10, 13],
                [ 7,  8],
                [ 8,  9],
                [10, 11],
                [11, 12],
                [13, 14],
                [14, 15] ]
    eps = 1e-7
    limbs = x[:,:,limbs_id,:]
    limbs = limbs[:,:,:,0,:]-limbs[:,:,:,1,:]
    angles = limbs[:,:,angle_id,:]
    angle_cos = F.cosine_similarity(angles[:,:,:,0,:], angles[:,:,:,1,:], dim=-1)
    return torch.acos(angle_cos.clamp(-1+eps, 1-eps)) 

def loss_angle(x, gt):
    '''
        Input: (N, T, 17, 3), (N, T, 17, 3)
    '''
    limb_angles_x = get_angles(x)
    limb_angles_gt = get_angles(gt)
    return nn.L1Loss()(limb_angles_x, limb_angles_gt)

def loss_angle_velocity(x, gt):
    """
    Mean per-angle velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert x.shape == gt.shape
    if x.shape[1]<=1:
        return torch.FloatTensor(1).fill_(0.)[0].to(x.device)
    x_a = get_angles(x)
    gt_a = get_angles(gt)
    x_av = x_a[:,1:] - x_a[:,:-1]
    gt_av = gt_a[:,1:] - gt_a[:,:-1]
    return nn.L1Loss()(x_av, gt_av)


class Track1_Loss(nn.Module):
    def __init__(self, alpha=1.0, beta=5.0, gamma=0.5, temperature=0.1, gamma_focal=2.0, center_target=17.0):
        """
        :param alpha: BCE/Focal Loss 的权重 (负责局部分类准确率)
        :param beta: NRMSE 的权重 (负责单个样本的总分对齐)
        :param gamma: Center Loss 的权重 (负责全局/Batch的平均分对齐测试集先验)
        :param temperature: 锐化系数，越小越接近0/1离散化
        :param gamma_focal: Focal Loss的聚焦参数
        :param center_target: 期望的 Batch 平均总分分布 (例如 17.0)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.gamma_focal = gamma_focal
        self.center_target = center_target
        
    def forward(self, pred_logits, target):
        target_float = target.float()
        
        eps = 0.1
        target_soft = target_float * (1 - eps) + eps / 2

        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target_soft, reduction='none')
        pred_probs = torch.sigmoid(pred_logits)
        p_t = pred_probs * target_float + (1 - pred_probs) * (1 - target_float)
        focal_weight = (1 - p_t) ** self.gamma_focal
        cls_loss = (focal_weight * bce_loss).mean()
        pred_probs_sharp = torch.sigmoid(pred_logits / self.temperature)
        
        p_i = pred_probs_sharp.sum(dim=(1, 2))   
        g_i = target_float.sum(dim=(1, 2))       

        sum_mse = F.mse_loss(p_i, g_i)
        rmse = torch.sqrt(sum_mse + 1e-8)
        nrmse = rmse / 34.0

        mean_p_i = p_i.mean() 
        center_loss = torch.abs(mean_p_i - self.center_target) / self.center_target
        total_loss = self.alpha * cls_loss + self.beta * nrmse + self.gamma * center_loss
        return total_loss, {
            "cls_loss": cls_loss.item(), 
            "nrmse": nrmse.item(), 
            "center_loss": center_loss.item()
        }
    

class Track2_Loss(nn.Module):
    def __init__(self, num_classes=5, alpha=0.5, beta=0.5, gamma=0.2, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma  

        weights = torch.tensor([7.5, 1.36, 1.0, 1.07, 7.5], dtype=torch.float32).to(device)
        self.ce = nn.CrossEntropyLoss(weight=weights) 

    def forward(self, pred_logits, target):
        """
        :param pred_logits: (B, 2, 5) 其中第1维度(大小为2)代表左右腿
        :param target: (B, 2)
        """
        probs = F.softmax(pred_logits, dim=-1)      # (B, 2, 5)
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).float()    # (B, 2, 5)

        probs_flat = probs.view(-1, self.num_classes)
        target_one_hot_flat = target_one_hot.view(-1, self.num_classes)

        tp = (probs_flat * target_one_hot_flat).sum(dim=0)  # (5,)
        fp = (probs_flat * (1 - target_one_hot_flat)).sum(dim=0)  # (5,)
        fn = ((1 - probs_flat) * target_one_hot_flat).sum(dim=0)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8) # (5,)

        classes_present = (target_one_hot_flat.sum(dim=0) > 0)
        
        if classes_present.sum() > 0:
            f1_loss = 1 - f1[classes_present].mean()
        else:
            f1_loss = torch.tensor(0.0, device=pred_logits.device)
        ce_loss = self.ce(pred_logits.reshape(-1, self.num_classes), target.reshape(-1))

        probs_left = probs[:, 0, :]   # (B, 5)
        probs_right = probs[:, 1, :]  # (B, 5)
        consistency_loss = F.mse_loss(probs_left, probs_right)

        total_loss = self.alpha * f1_loss + self.beta * ce_loss + self.gamma * consistency_loss
        
        return total_loss
