from monai.utils import first, set_determinism

from monai.transforms import AsDiscrete
from networks.msHead_3D.network_backbone import MSHEAD_ATTN
from networks.UXNet_3D.network_backbone import UXNET
from monai.networks.nets import UNETR, SwinUNETR
# from networks.nnFormer.nnFormer_seg import nnFormer
# from networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch, ThreadDataLoader
from monai.metrics import DiceMetric

import torch
from load_datasets_transforms import data_loader, data_transforms, infer_post_transforms

import os
import argparse

import os
import numpy as np
from tqdm import tqdm
import datetime
import argparse
import time

print(f'########### Running KITS Segmentation PLOTTING ################# \n')
parser = argparse.ArgumentParser(description='MSHEAD_ATTN hyperparameters for medical image segmentation')
## Input data hyperparameters
parser.add_argument('--root', type=str, default='/blue/r.forghani/share/kits2019', required=False, help='Root folder of all your images and labels')
parser.add_argument('--output', type=str, default='/orange/r.forghani/results', required=False, help='Output folder for both tensorboard and the best model')
parser.add_argument('--dataset', type=str, default='kits', required=False, help='Datasets: {feta, flare, amos}, Fyi: You can add your dataset here')

## Input model & training hyperparameters
parser.add_argument('--network', type=str, default='MSHEAD', required=False, help='Network models: {TransBTS, nnFormer, UNETR, SwinUNETR, 3DUXNET}')
parser.add_argument('--pretrained_weights', default='', required=False, help='Path of pretrained/fine-tuned weights')
parser.add_argument('--mode', type=str, default='test', help='Training or testing mode')
parser.add_argument('--sw_batch_size', type=int, default=4, help='Sliding window batch size for inference')
parser.add_argument('--overlap', type=float, default=0.5, help='Sub-volume overlapped percentage')

## Efficiency hyperparameters
parser.add_argument('--gpu', type=str, default='0', help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=1, help='Cache rate to cache your dataset into GPUs')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
parser.add_argument('--fold', type=int, default=0, help='current running fold')
parser.add_argument('--no_split', default=False, help='Not splitting into train and validation')
parser.add_argument('--plot', default=True, help='Plotting prediction or Not')


args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

test_samples, out_classes = data_loader(args)

test_files = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(test_samples['images'], test_samples['labels'])
]

set_determinism(seed=0)
### extracting run_id of testing model
splitted_text = args.pretrained_weights[:args.pretrained_weights.rindex('/')]
run_id = splitted_text[splitted_text.rindex('/')+1:]
print(f'############## run id of pretrained model: {run_id} ################')

output_seg_dir = os.path.join(args.output, run_id, 'output_seg')
if not os.path.exists(output_seg_dir):
    os.makedirs(output_seg_dir)

test_transforms = data_transforms(args)
post_transforms = infer_post_transforms(args, test_transforms, out_classes, output_seg_dir)

print(f'transforms:{post_transforms} ')

## Inference Pytorch Data Loader and Caching

test_ds = CacheDataset(
    data=test_files, transform=test_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
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

elif args.network == '3DUXNET':
    model = UXNET(
        in_chans=1,
        out_chans=out_classes,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        spatial_dims=3,
    ).to(device)

elif args.network == 'UNETR':
    model = UNETR(
        in_channels=1,
        out_channels=out_classes,
        img_size=(96, 96, 96),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    ).to(device)


if args.network=='MSHEAD':
    if args.fold == 0:
        args.pretrained_weights = "/orange/r.forghani/results/09-28-24_0628/model_best.pth"
    elif args.fold == 1:
        args.pretrained_weights = ""
    elif args.fold == 2:
        args.pretrained_weights = "/orange/r.forghani/results/09-29-24_1615/model_best.pth"
    elif args.fold == 3:
        args.pretrained_weights = ""
    elif args.fold == 4:
        args.pretrained_weights = "/orange/r.forghani/results/09-29-24_2050/model_best.pth"

elif args.network=='SwinUNETR':
    args.pretrained_weights = "/orange/r.forghani/results/10-01-24_0423/model_best.pth"
elif args.network=='3DUXNET':
    args.pretrained_weights = "/orange/r.forghani/results/10-01-24_1504/model_best.pth"
elif args.network=='UNETR':
    args.pretrained_weights = "/orange/r.forghani/results/09-29-24_0147/model_best.pth"


print(f'best model path:{args.pretrained_weights}')
state_dict = torch.load(args.pretrained_weights)
model.load_state_dict(state_dict['model'])
model.eval()


post_label = AsDiscrete(to_onehot=out_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=out_classes)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

dice_vals = list()
s_time = time.time()
with torch.no_grad():
    for step, test_data in enumerate(test_loader):
        test_inputs = test_data["image"].to(device)
        # val_outputs = model(val_inputs)
        roi_size = (96, 96, 96)
        test_data["pred"] = sliding_window_inference(
            test_inputs, roi_size, args.sw_batch_size, model, overlap=args.overlap
        )
        test_data = [post_transforms(i) for i in decollate_batch(test_data)]
