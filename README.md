# CVPR 2026 CV4CHL

## 推理：
```
CUDA_VISIBLE_DEVICES=0 python predict_gait.py --model_1 checkpoint/gait1/best.pth --model_2 checkpoint/gait2/best.pth --vote
```

## 训练：
```
# track1
CUDA_VISIBLE_DEVICES=1 nohup python -u train_gait_1.py > train_track1.log 2>&1 &

# track2
CUDA_VISIBLE_DEVICES=1 nohup python -u train_gait_2_1shot.py --selection best.pth > train_track2.log 2>&1 &
```
