#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import datetime
import time
import argparse
import glob
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from monai.utils import set_determinism
from monai.transforms import (
    LoadImaged, AddChanneld, Spacingd, Orientationd, ScaleIntensityRanged,
    RandCropByPosNegLabeld, RandFlipd, RandShiftIntensityd, RandAffined,
    ToTensord, Compose, AsDiscrete
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import UNETR, SwinUNETR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from networks.msHead_3D.network_backbone import MSHEAD_ATTN

# ------------------------------ #
#        ARGUMENT PARSING        #
# ------------------------------ #

parser = argparse.ArgumentParser(description='MSHEAD_ATTN for medical image segmentation')
parser.add_argument('--preprocessed_dir', type=str, required=True, help='Directory where preprocessed data is stored')
parser.add_argument('--output', type=str, default='/orange/r.forghani/results', help='Output directory')
parser.add_argument('--network', type=str, default='MSHEAD', help='Network model: {MSHEAD, UNETR, SwinUNETR}')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--optim', type=str, default='AdamW', help='Optimizer: {Adam, AdamW}')
parser.add_argument('--max_iter', type=int, default=40000, help='Max training iterations')
parser.add_argument('--eval_step', type=int, default=500, help='Evaluate every X steps')
parser.add_argument('--gpu', type=int, default=0, help='GPU number')
parser.add_argument('--num_workers', type=int, default=8, help='Number of DataLoader workers')
parser.add_argument('--fold', type=int, default=0, help='Cross-validation fold index')
parser.add_argument('--resume', action='store_true', help='Resume training from a checkpoint')

args = parser.parse_args()
print(f'########### Training on Fold {args.fold} ###########')

# ------------------------------ #
#       SET DEVICE & PATHS       #
# ------------------------------ #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

run_id = datetime.datetime.today().strftime('%m-%d-%y_%H%M')
print(f'Run ID: {run_id}')

root_dir = os.path.join(args.output, run_id)
os.makedirs(root_dir, exist_ok=True)

tensorboard_dir = os.path.join(root_dir, 'tensorboard')
os.makedirs(tensorboard_dir, exist_ok=True)
writer = SummaryWriter(log_dir=tensorboard_dir)

# ------------------------------ #
#       DATA TRANSFORMS          #
# ------------------------------ #

def get_train_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),  # Load preprocessed data
        AddChanneld(keys=["image", "label"]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(128, 128, 128),
            pos=3, neg=1, num_samples=4, image_key="image", image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        RandAffined(
            keys=['image', 'label'],
            mode=("bilinear", "nearest"),
            prob=1.0,
            spatial_size=(128, 128, 128),
            rotate_range=(np.pi/30, np.pi/30, np.pi/30),
            scale_range=(0.1, 0.1, 0.1),
        ),
        ToTensord(keys=["image", "label"]),
    ])

def get_val_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ])

# ------------------------------ #
#         LOAD DATASET           #
# ------------------------------ #

preprocessed_images = sorted(glob.glob(os.path.join(args.preprocessed_dir, "image_*.nii.gz")))
preprocessed_labels = sorted(glob.glob(os.path.join(args.preprocessed_dir, "label_*.nii.gz")))

train_files = [{"image": img, "label": lbl} for img, lbl in zip(preprocessed_images, preprocessed_labels)]
val_files = train_files[:len(train_files) // 5]  # Simple 80/20 split

train_ds = Dataset(data=train_files, transform=get_train_transforms())
val_ds = Dataset(data=val_files, transform=get_val_transforms())

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.num_workers)

print(f"Train data size: {len(train_ds)} samples")
print(f"Validation data size: {len(val_ds)} samples")

# ------------------------------ #
#          MODEL SETUP           #
# ------------------------------ #

if args.network == 'MSHEAD':
    model = MSHEAD_ATTN(img_size=(128, 128, 128), in_chans=1, out_chans=3).to(device)
elif args.network == 'SwinUNETR':
    model = SwinUNETR(img_size=(128, 128, 128), in_channels=1, out_channels=3).to(device)
elif args.network == 'UNETR':
    model = UNETR(img_size=(128, 128, 128), in_channels=1, out_channels=3).to(device)

print(f"Model {args.network} initialized.")

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
dice_metric = DiceMetric(include_background=True, reduction="mean")

# ------------------------------ #
#       TRAINING FUNCTION        #
# ------------------------------ #

def train(global_step):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        x, y = batch["image"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        if global_step % 100 == 0:
            print(f"Step {global_step}: Loss = {epoch_loss / (global_step + 1):.5f}")
        
        global_step += 1
    return global_step

# ------------------------------ #
#       TRAINING LOOP            #
# ------------------------------ #

global_step = 0
for epoch in range(args.max_iter // len(train_loader)):
    global_step = train(global_step)
    if global_step >= args.max_iter:
        break

print("Training completed!")
