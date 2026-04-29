# CVPR 2026 CV4CHL

## Installation
```
conda create -n motionbert python=3.7 anaconda
conda activate motionbert
# Please install PyTorch according to your CUDA version.
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

## Getting Started
Download the preprocessed test data and model from the following URL.
```
https://drive.google.com/drive/folders/1CVfa_Khi0-U7rTnm7YT-1qxtmE4sqgWh?usp=drive_link
```
Due to network issues, the upload speed to Google Drive is slow. If you cannot find dataset.zip and checkpoint.zip in Google Drive, you can download it via the following link:
```
https://pan.baidu.com/s/14QVdqMdGCxFBPnLroiBZeQ?pwd=4ff9
```

Unzip dataset.zip and checkpoint.zip and place them in the root directory of your workspace. The folder structure should be:
```
CV4CHL/
├── lib/
├── checkpoint/
│   ├── gait1/
│   ├── gait2/
│   └── latest_epoch.bin
├── configs/
└── dataset/
    ├── test_track1_4.pkl
    ├── test_track1_5.pkl
    ├── test_track1_18.pkl
    ├── test_track1_26.pkl
    ├── test_track1_28.pkl
    ├── test_track1_40.pkl
    ├── test_track1_42.pkl
    └── test_track1_43.pkl
    ...
...

```

## Inference：
```
CUDA_VISIBLE_DEVICES=0 python predict_gait.py --model_1 checkpoint/gait1/best.pth --model_2 checkpoint/gait2/best.pth --vote
```

## Contact Me
If you have any questions, please contact email: hyt2025110753@bupt.edu.cn
