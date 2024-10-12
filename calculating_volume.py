#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 11:06:19 2021

@author: leeh43
"""

from monai.utils import set_determinism
from monai.transforms import AsDiscrete
# from networks.UXNet_3D.network_backbone import UXNET
from networks.msHead_3D.network_backbone import MSHEAD_ATTN
from monai.networks.nets import UNETR, SwinUNETR
# from networks.nnFormer.nnFormer_seg import nnFormer
# from networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch, ThreadDataLoader

import torch
from torch.utils.tensorboard import SummaryWriter
from load_datasets_transforms import data_loader, data_transforms
import matplotlib.pyplot as plt

import os
import numpy as np
from tqdm import tqdm
import datetime
import argparse
import time

print(f'########### Running AMOS Segmentation ################# \n')
parser = argparse.ArgumentParser(description='MSHEAD_ATTN hyperparameters for medical image segmentation')
## Input data hyperparameters
parser.add_argument('--root', type=str, default='/blue/r.forghani/share/amoss22/amos22', required=False, help='Root folder of all your images and labels')
parser.add_argument('--output', type=str, default='/orange/r.forghani/results', required=False, help='Output folder for both tensorboard and the best model')
parser.add_argument('--dataset', type=str, default='amos', required=False, help='Datasets: {feta, flare, amos}, Fyi: You can add your dataset here')

## Input model & training hyperparameters
parser.add_argument('--network', type=str, default='MSHEAD', help='Network models: {MSHEAD, TransBTS, nnFormer, UNETR, SwinUNETR, 3DUXNET}')
parser.add_argument('--mode', type=str, default='test', help='Training or testing mode')
parser.add_argument('--pretrain', default=False, help='Have pretrained weights or not')
parser.add_argument('--pretrained_weights', default='/orange/r.forghani/results/09-11-24_1805/model_best.pth', help='Path of pretrained weights')
parser.add_argument('--batch_size', type=int, default='1', help='Batch size for subject input')
parser.add_argument('--crop_sample', type=int, default='2', help='Number of cropped sub-volumes for each subject')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for training')
parser.add_argument('--optim', type=str, default='AdamW', help='Optimizer types: Adam / AdamW')
parser.add_argument('--max_iter', type=int, default=40000, help='Maximum iteration steps for training')
parser.add_argument('--eval_step', type=int, default=500, help='Per steps to perform validation')
parser.add_argument('--resume', default=False, help='resume training from an earlier iteration')
parser.add_argument('--finetune', default=True, help='Finetuning on AMOS using best fold model from FLARE')
## Efficiency hyperparameters
parser.add_argument('--gpu', type=int, default=0, help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=1, help='Cache rate to cache your dataset into memory')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
parser.add_argument('--start_index', type=int, default=160, help='validation set starts')
parser.add_argument('--end_index', type=int, default=180, help='validation set ends')
parser.add_argument('--no_split',  default=False, help='training on whole dataset')
parser.add_argument('--plot',  default=False, help='plotting prediction or not')


args = parser.parse_args()
print(f'################################')
print(f'args:{args}')
print('#################################')
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print('Used GPU: {}'.format(args.gpu))

if args.dataset == 'amos':
    ORGAN_CLASSES = {1: "Spleen", 2: "Right Kidney", 3: "Left Kidney", 4: "Gall Bladder", 5: "Esophagus",6: "Liver",
        7: "Stomach", 8: "Aorta", 9: "Inferior Vena Cava", 10: "Pancreas", 11: "Right Adrenal Gland", 
        12: "Left Adrenal Gland", 13: "Duodenum", 14: "Bladder", 15: "Prostate"
    }
elif args.dataset == 'flare':
    ORGAN_CLASSES = {1: "Liver", 2: "Kidney", 3: "Spleen", 4: "Pancreas"}
elif args.dataset == 'kits':
    ORGAN_CLASSES = {1: "Kidney", 2: "Tumor"}


test_samples, out_classes = data_loader(args)
test_files = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(test_samples['images'], test_samples['labels'])
]
print(f'test files:{len(test_files)}')

set_determinism(seed=0)
test_transforms = data_transforms(args)
print('Start caching datasets!')
test_ds = CacheDataset( data=test_files, transform=test_transforms, 
                       cache_rate=args.cache_rate, num_workers=args.num_workers)
test_loader = ThreadDataLoader(test_ds, batch_size=1, num_workers=0)



## Load Networks
### set device according to your local machine
device = torch.device("cuda")
print(f'--- device:{device} ---')


for step, batch in enumerate(test_loader):
    test_inputs, test_labels = (batch["image"].to(device), batch["label"].to(device))
    print(f'input: {test_inputs.shape} labels:{test_labels.shape}')     # B,C,D,H,W format: D slices in each CT data
                                                                        #each slice has 1 channel-> C = 1
                                                                        # for validation loader B = 1 too.
                                                                        # So, 1,1,D,H,W
                                                                        # H, W and D will be different for each CT
                                                                        # we need to calculate shape-wise result on
                                                                        # validation set. So, here we take the full image
                                                                        # to calculate volume for each organ.
                                                                        # To calculate volume we can use labels from
                                                                        # val_labels. How? That's where I need help
    test_labels = test_labels.numpy()
    test_labels = test_labels[0, 0, :, :, :]
    unique_labels = np.unique(test_labels)
    print(f'unique labels: {unique_labels}')
    for label in unique_labels:
        dummy = np.zeros(shape=test_labels.shape, dtype='uint8')
        dummy[test_labels==label] = 1
        N_voxel = np.count_nonzero(dummy)
        volume = N_voxel * 1.5 * 1.5 * 2    # in mm^3
        volume = volume/1000                # in cm^3
        print(f'Class: {ORGAN_CLASSES[label]} volume: {volume}')

    ############# For Visualization. Do u have to do it? No. For understanding, you can.

    ############ 
    
    # Before Visualization using matplotlib, you need to convert those tensor to numpy arrays.
    # numpy arrays will be B,C,D,H,W format. Convert it to B,C,H,W,D format for your convenience(your wish)
                               
                                # Or
    
    # You can remove the last line in val_transforms under "args.dataset==amos" in load_datasets_transforms.py
    # last line --> ToTensord(keys=["image", "label"]).
    # if you remove this line data will be automatically in B,C,H,W,D format and probably as numpy array.
    # Then you can plot it directly

    #############
    
    # plt.figure("check", (18, 6))
    # plt.subplot(1, 2, 1)
    # plt.title(f"image {i}")
    # plt.imshow(val_inputs["image"][0, 0, :, :, 80], cmap="gray")     # pass a D value to visualize a slice. Here D=80
    # plt.subplot(1, 2, 2)
    # plt.title(f"label {i}")
    # plt.imshow(val_inputs["label"][0, 0, :, :, 80])
    # # plt.subplot(1, 3, 3)
    # # plt.title(f"output {i}")
    # # plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, 80])
    # plt.show()
    # if i == 2:
    #     break
    ##############################

