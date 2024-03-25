from monai.utils import first, set_determinism

from monai.transforms import AsDiscrete
from networks.msHead_3D.network_backbone import MSHEAD_ATTN
from networks.UXNet_3D.network_backbone import UXNET
from networks.Focus_Net.s_net_3d import s_net
from monai.networks.nets import UNETR, SwinUNETR
# from networks.nnFormer.nnFormer_seg import nnFormer
# from networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch, ThreadDataLoader
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations

import torch
from load_datasets_transforms import data_loader, data_transforms, infer_post_transforms


import glob
import os
import argparse

import os
import numpy as np
from tqdm import tqdm
import datetime
import argparse
import time
import natsort

print(f'########### Testing LN Segmentation ################# \n')
parser = argparse.ArgumentParser(description='MSHEAD_ATTN hyperparameters for medical image segmentation')
## Input data hyperparameters
parser.add_argument('--root', type=str, default='/blue/r.forghani/data/lymph_node/ct_221', required=False, help='Root folder of all your images and labels')
parser.add_argument('--output', type=str, default='/orange/r.forghani/results', required=False, help='Output folder for both tensorboard and the best model')
parser.add_argument('--dataset', type=str, default='LN', required=False, help='Datasets: {feta, flare, amos, LN}, Fyi: You can add your dataset here')

## Input model & training hyperparameters
parser.add_argument('--network', type=str, default='SNET', help='Network models: {MSHEAD, TransBTS, nnFormer, UNETR, SwinUNETR, 3DUXNET}')
parser.add_argument('--mode', type=str, default='test', help='Training or testing mode')
parser.add_argument('--pretrain', default=False, help='Have pretrained weights or not')
parser.add_argument('--trained_weights', default='/orange/r.forghani/results/03-20-24_1504/model_best.pth', required=False, help='Path of pretrained/fine-tuned weights')
parser.add_argument('--batch_size', type=int, default='2', help='Batch size for subject input')
parser.add_argument('--sw_batch_size', type=int, default=4, help='Sliding window batch size for inference')
parser.add_argument('--crop_sample', type=int, default='4', help='Number of cropped sub-volumes for each subject')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for training')
parser.add_argument('--optim', type=str, default='AdamW', help='Optimizer types: Adam / AdamW')
parser.add_argument('--max_iter', type=int, default=40000, help='Maximum iteration steps for training')
parser.add_argument('--eval_step', type=int, default=400, help='Per steps to perform validation')
parser.add_argument('--resume', default=False, help='resume training from an earlier iteration')
parser.add_argument('--overlap', type=float, default=0.5, help='Sub-volume overlapped percentage')
## Efficiency hyperparameters
parser.add_argument('--gpu', type=int, default=0, help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=1, help='Cache rate to cache your dataset into memory')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')

args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

set_determinism(seed=0)

spatial_size = (64, 96, 96)
# train_samples, valid_samples, out_classes = data_loader(args)
out_classes = 1
if args.dataset == "LN":
    pat_ids = natsort.natsorted(os.listdir(args.root))[200:]
    test_images = natsort.natsorted([glob.glob(os.path.join(args.root, pat_id, "IM00*")) for pat_id in pat_ids])
    test_labels = natsort.natsorted([glob.glob(os.path.join(args.root, pat_id, "Segmentation_v2*")) for pat_id in pat_ids])
    test_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(test_images, test_labels)]

print(f'test_files :{len(test_files)}')

set_determinism(seed=0)

test_transforms = data_transforms(args, spatial_size)
test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
test_loader = ThreadDataLoader(test_ds, batch_size=1, num_workers=0)

## Load Networks
device = torch.device("cuda")
print(f'--- device:{device} ---')

if args.network == 'SNET':
    model = s_net(channel=1, num_classes=out_classes, se=True, norm='bn').to(device)

elif args.network == 'MSHEAD':
    model = MSHEAD_ATTN(
        img_size=(96, 96, 96),
        in_chans=1,
        out_chans=out_classes,
        depths=[2,2,6,2],
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

state_dict = torch.load(args.trained_weights)
model.load_state_dict(state_dict['model'])
model.eval()


post_label = AsDiscrete(argmax = False)
post_pred = Compose([
                Activations(sigmoid=True),
                AsDiscrete(argmax=False, threshold=0.5),
            ])
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

dice_vals = list()
s_time = time.time()

with torch.no_grad():
    for step, batch in enumerate(test_loader):
        test_inputs, test_labels = (batch["image"].to(device), batch["label"].to(device))

        test_inputs = test_inputs.permute(0, 1, 4, 2, 3)          # B, C, H, W, D --> B, C, D, H, W
        test_labels = test_labels.permute(0, 1, 4, 2, 3)          # B, C, H, W, D --> B, C, D, H, W

        # val_outputs = model(val_inputs)
        roi_size = spatial_size
        test_outputs = sliding_window_inference(
            test_inputs, roi_size, args.sw_batch_size, model, overlap=args.overlap
        )

        test_labels_list = decollate_batch(test_labels)
        test_labels_convert = [
            post_label(test_label_tensor) for test_label_tensor in test_labels_list
        ]

        test_outputs_list = decollate_batch(test_outputs)
        test_output_convert = [
            post_pred(test_pred_tensor) for test_pred_tensor in test_outputs
        ]

        dice_metric(y_pred=test_output_convert, y=test_labels_convert)
        dice = dice_metric.aggregate().item()
        dice_vals.append(dice)
        # epoch_iterator_val.set_description(
        #     "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice)
        # )
    dice_metric.reset()
mean_dice_test = np.mean(dice_vals)

test_time = time.time() - s_time
print(f"test takes {datetime.timedelta(seconds=int(test_time))}")
print(f'mean test dice: {mean_dice_test}')

