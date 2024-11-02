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
import copy
from utils import scale_wise_organ_filtration, filtering_output, get_KiTS_regions, hierarchical_prediction
parser = argparse.ArgumentParser(description='3D UX-Net inference hyperparameters for medical image segmentation')
## Input data hyperparameters
# parser.add_argument('--root', type=str, default='/blue/r.forghani/share/flare_data', required=False, help='Root folder of all your images and labels')
# parser.add_argument('--output', type=str, default='/orange/r.forghani/results', required=False, help='Output folder for both tensorboard and the best model')
# parser.add_argument('--dataset', type=str, default='flare', required=False, help='Datasets: {feta, flare, amos}, Fyi: You can add your dataset here')
parser.add_argument('--root', type=str, default='/blue/r.forghani/share/amoss22/amos22', required=False, help='Root folder of all your images and labels')
parser.add_argument('--output', type=str, default='/orange/r.forghani/results', required=False, help='Output folder for both tensorboard and the best model')
parser.add_argument('--dataset', type=str, default='amos', required=False, help='Datasets: {feta, flare, amos}, Fyi: You can add your dataset here')
## Input model & training hyperparameters
parser.add_argument('--network', type=str, default='MSHEAD', required=False, help='Network models: {TransBTS, nnFormer, UNETR, SwinUNETR, 3DUXNET}')
parser.add_argument('--trained_weights', default='', required=False, help='Path of pretrained/fine-tuned weights')
parser.add_argument('--mode', type=str, default='test', help='Training or testing mode')
parser.add_argument('--sw_batch_size', type=int, default=4, help='Sliding window batch size for inference')
parser.add_argument('--overlap', type=float, default=0.5, help='Sub-volume overlapped percentage')

## Efficiency hyperparameters
parser.add_argument('--gpu', type=str, default='0', help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=1, help='Cache rate to cache your dataset into GPUs')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
parser.add_argument('--fold', type=int, default=0, help='current running fold')
parser.add_argument('--no_split', default=False, help='No splitting into train and validation')
parser.add_argument('--plot', default=False, help='plotting prediction or not')


args = parser.parse_args()
print(f'args:{args}')
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print('Used GPU: {}'.format(args.gpu))

if args.dataset == 'amos':
    args.root = '/blue/r.forghani/share/amoss22/amos22'
    ORGAN_CLASSES = {1: "Spleen", 2: "Right Kidney", 3: "Left Kidney", 4: "Gall Bladder", 5: "Esophagus",6: "Liver",
        7: "Stomach", 8: "Aorta", 9: "Inferior Vena Cava", 10: "Pancreas", 11: "Right Adrenal Gland", 
        12: "Left Adrenal Gland", 13: "Duodenum", 14: "Bladder", 15: "Prostate"
    }
    organ_size_range = [150, 500]
    spacing = (1.5, 1.5, 2)
elif args.dataset == 'flare':
    args.root = '/blue/r.forghani/share/flare_data'
    ORGAN_CLASSES = {1: "Liver", 2: "Kidney", 3: "Spleen", 4: "Pancreas"}
    organ_size_range = [250, 1000]
    spacing = (1, 1, 1.2)
elif args.dataset == 'kits':
    args.root = '/blue/r.forghani/share/kits2019'
    ORGAN_CLASSES = {1: "Kidney", 2: "Tumor"}

SMALL = 1
MEDIUM = 2
LARGE = 3


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
device = torch.device("cuda")
print(f'--- device:{device} ---')

if args.network == 'MSHEAD':
    model = MSHEAD_ATTN(
        img_size=(96, 96, 96),
        in_chans=1,
        out_chans=out_classes,
        depths=[2,2,2,2],
        feat_size=[48,96,192,384],
        num_heads = [3,6,12,24],
        use_checkpoint=False,
    ).to(device)

elif args.network == 'SwinUNETR':
    model = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=out_classes,
        feature_size=48,
        use_checkpoint=False,
    ).to(device)
if args.dataset != 'amos':
    if args.fold == 0:
        # args.trained_weights = '/orange/r.forghani/results/09-18-24_0219/model_best.pth'
        args.trained_weights = '/orange/r.forghani/results/10-30-24_0442/model_best.pth'
    elif args.fold == 1:
        # args.trained_weights = '/orange/r.forghani/results/09-20-24_0448/model_best.pth'
        args.trained_weights = '/orange/r.forghani/results/10-30-24_0454/model_best.pth'
    elif args.fold == 2:
        # args.trained_weights = '/orange/r.forghani/results/09-21-24_1416/model_best.pth'
        args.trained_weights = '/orange/r.forghani/results/10-30-24_0500/model_best.pth'
    elif args.fold == 3:
        # args.trained_weights = '/orange/r.forghani/results/09-18-24_2221/model_best.pth'
        args.trained_weights = '/orange/r.forghani/results/10-30-24_0505/model_best.pth'
    elif args.fold == 4:
        # args.trained_weights = '/orange/r.forghani/results/09-18-24_2224/model_best.pth'
        args.trained_weights = '/orange/r.forghani/results/10-30-24_0513/model_best.pth'

print(f'best model from fold:{args.fold} model path:{args.trained_weights}')
state_dict = torch.load(args.trained_weights)
model.load_state_dict(state_dict['model'])
model.eval()
# with torch.no_grad():
#     for i, test_data in enumerate(test_loader):
#         images = test_data["image"].to(device)
#         roi_size = (96, 96, 96)
#         test_data['pred'] = sliding_window_inference(
#             images, roi_size, args.sw_batch_size, model, overlap=args.overlap
#         )
#         test_data = [post_transforms(i) for i in decollate_batch(test_data)]


post_label = AsDiscrete(to_onehot=out_classes)
post_pred = AsDiscrete(to_onehot=out_classes)
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

regions = get_KiTS_regions()

dice_vals = list()
s_time = time.time()

## Load Networks
### set device according to your local machine
device = torch.device("cuda")
print(f'--- device:{device} ---')
output_organ = {organ:[] for organ in regions}
patient_wise_scores = []
with torch.no_grad():
    for step, batch in enumerate(test_loader):
        print(f'########## image:{step} ######################')
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
        
        roi_size = (96, 96, 96)
        test_outputs = sliding_window_inference(
            test_inputs, roi_size, args.sw_batch_size, model, overlap=args.overlap
        )
        print(f'test output size:{test_outputs.shape}')

        # Organ Wise Calculation
        for organ, labels in regions.items():
            print(f'########## Calculating for Organ: {organ} ###########')
            new_output = hierarchical_prediction(test_outputs, labels, prediction=True)
            new_gt = hierarchical_prediction(test_labels, labels)

            print(f'new output:{new_output.shape}')
            print(f'new_gt:{new_gt.shape}')
        
            
            #### No need for decollate batch if we have only 1 sample/batch i.e. batch_size = 1
            test_labels_list = decollate_batch(new_gt)
            test_labels_convert = [
                post_label(test_label_tensor) for test_label_tensor in test_labels_list
            ]

            test_outputs_list = decollate_batch(new_output)
            test_output_convert = [
                post_pred(test_pred_tensor) for test_pred_tensor in test_outputs_list
            ]

            dice_metric(y_pred=test_output_convert, y=test_labels_convert)
            dice = dice_metric.aggregate().item()
            dice_metric.reset()
            output_organ[organ].append(dice)



for organ, dice_vals in output_organ.items():
    mean_dice = np.mean(dice_vals)
    print(f'\n ########## Organ:{organ} dice:{dice_vals} \n Mean dice:{mean_dice} ########### ')
# patient_wise_dice_vals = torch.tensor(patient_wise_dice_vals)
# size_wise_mean = torch.mean(size_wise_dice_vals, dim=0) # Calculate the mean across each elem of sublist (along axis 1)
# patient_wise_dice = torch.mean(size_wise_dice_vals, dim=1) # Calculate the mean of each sublist (along axis 1)
# mean_dice_test = torch.mean(patient_wise_dice)
# # mean_dice_test = np.mean(dice_vals)

test_time = time.time() - s_time
print(f"test takes {datetime.timedelta(seconds=int(test_time))}")



