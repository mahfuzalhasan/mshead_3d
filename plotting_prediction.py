from monai.utils import first, set_determinism

from monai.transforms import AsDiscrete
from networks.msHead_3D.network_backbone import MSHEAD_ATTN
from networks.UXNet_3D.network_backbone import UXNET
from monai.networks.nets import UNETR, SwinUNETR
from networks.nnFormer.nnFormer_seg import nnFormer
from networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
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

parser = argparse.ArgumentParser(description='3D UX-Net inference hyperparameters for medical image segmentation')
## Input data hyperparameters
parser.add_argument('--root', type=str, default='', required=False, help='Root folder of all your images and labels')
parser.add_argument('--output', type=str, default='/orange/r.forghani/results', required=False, help='Output folder for both tensorboard and the best model')
parser.add_argument('--dataset', type=str, default='flare', required=False, help='Datasets: {feta, flare, amos}, Fyi: You can add your dataset here')

## Input model & training hyperparameters
parser.add_argument('--network', type=str, default='MSHEAD', required=False, help='Network models: {TransBTS, nnFormer, UNETR, SwinUNETR, UXNET}')
parser.add_argument('--trained_weights', default='', required=False, help='Path of pretrained/fine-tuned weights')
parser.add_argument('--mode', type=str, default='test', help='Training or testing mode')
parser.add_argument('--sw_batch_size', type=int, default=4, help='Sliding window batch size for inference')
parser.add_argument('--overlap', type=float, default=0.5, help='Sub-volume overlapped percentage')

## Efficiency hyperparameters
parser.add_argument('--gpu', type=str, default='0', help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=1, help='Cache rate to cache your dataset into GPUs')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
parser.add_argument('--plot', default=True, help='plotting the prediction as nii.gz file')
parser.add_argument('--fold', type=int, default=0, help='current running fold')
parser.add_argument('--no_split', default=False, help='No splitting into train and validation')


args = parser.parse_args()

if not args.root:
    if args.dataset == 'flare':
        args.root = '/blue/r.forghani/share/flare_data'
    elif args.dataset == 'amos':
        args.root = '/blue/r.forghani/share/amoss22/amos22'
    elif args.dataset == 'kits':
        args.root = '/blue/r.forghani/share/kits2019'
    else:
        raise NotImplementedError(f'No such dataset: {args.dataset}')

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print(f'######## datapath: {args.root}  dataset:{args.dataset} fold:{args.fold} ######## \n')


test_samples, out_classes = data_loader(args)

test_files = [
    {"image": image_name, "label": label_name, "path": data_path}
    for image_name, label_name, data_path in zip(test_samples['images'], test_samples['labels'], test_samples['paths'])
]

print(f'test files:{test_files}')

if args.dataset != 'amos':
    if args.fold == 0:
        args.trained_weights = '/orange/r.forghani/results/02-11-25_1721/model_best.pth'                  #flare_wo_merge_projection
        # args.trained_weights = '/orange/r.forghani/results/02-09-25_0204/model_best.pth'                # flare avg pool
        # args.trained_weights = '/orange/r.forghani/results/02-08-25_0316/model_best.pth'                # flare dwt pool
        # args.trained_weights = '/orange/r.forghani/results/01-07-25_0204/model_best.pth'              # flare idwt_v2
        # args.trained_weights = '/orange/r.forghani/results/12-20-24_0342/model_best.pth'              #flare
        # args.trained_weights = '/orange/r.forghani/results/12-22-24_1727/model_best.pth'              #KiTS
        # args.trained_weights = '/orange/r.forghani/results/SwinUNETR/11-04-24_2018/model_best.pth'    #SWIN
        # args.trained_weights = '/orange/r.forghani/results/nnFormer/nnformer/fold_0/fold_0_model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/UXNET/3duxnet/fold_0/fold_0_model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/UNETR/fold_0/model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/TransBTS/fold_0/model_best.pth'
    elif args.fold == 1:
        args.trained_weights = '/orange/r.forghani/results/02-11-25_1728/model_best.pth'                #flare_wo_merge_projection
        # args.trained_weights = '/orange/r.forghani/results/02-09-25_0214/model_best.pth'                # flare avg pool
        # args.trained_weights = '/orange/r.forghani/results/02-08-25_0342/model_best.pth'                # flare dwt pool
        # args.trained_weights = '/orange/r.forghani/results/01-07-25_0102/model_best.pth'              #flare_idwt_v2
        # args.trained_weights = '/orange/r.forghani/results/12-20-24_1658/model_best.pth'              #flare
        # args.trained_weights = '/orange/r.forghani/results/12-23-24_0128/model_best.pth'              #KiTS
        # args.trained_weights = '/orange/r.forghani/results/SwinUNETR/11-08-24_0059/model_best.pth'    #SWIN
        # args.trained_weights = '/orange/r.forghani/results/nnFormer/nnformer/fold_1/fold_1_model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/UXNET/3duxnet/fold_1/fold_1_model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/UNETR/fold_1/model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/TransBTS/fold_1/model_best.pth'
    elif args.fold == 2:
        args.trained_weights = '/orange/r.forghani/results/02-09-25_0223/model_best.pth'                # flare avg pool
        # args.trained_weights = '/orange/r.forghani/results/02-08-25_0351/model_best.pth'                # flare dwt pool
        # args.trained_weights = '/orange/r.forghani/results/01-07-25_1307/model_best.pth'              #flare_idwt_v2
        # args.trained_weights = '/orange/r.forghani/results/12-20-24_1836/model_best.pth'              #flare
        # args.trained_weights = '/orange/r.forghani/results/12-23-24_0145/model_best.pth'              #KiTS
        # args.trained_weights = '/orange/r.forghani/results/SwinUNETR/11-06-24_2219/model_best.pth'    #SWIN
        # args.trained_weights = '/orange/r.forghani/results/nnFormer/nnformer/fold_2/fold_2_model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/UXNET/3duxnet/fold_2/fold_2_model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/UNETR/fold_2/model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/TransBTS/fold_2/model_best.pth'
    elif args.fold == 3:
        args.trained_weights = '/orange/r.forghani/results/02-09-25_0244/model_best.pth'                # flare avg pool
        # args.trained_weights = '/orange/r.forghani/results/02-08-25_0404/model_best.pth'                # flare dwt pool
        # args.trained_weights = '/orange/r.forghani/results/01-07-25_1708/model_best.pth'              #flare_idwt_v2
        # args.trained_weights = '/orange/r.forghani/results/12-20-24_1943/model_best.pth'              #flare
        # args.trained_weights = '/orange/r.forghani/results/12-23-24_0240/model_best.pth'              #KiTS
        # args.trained_weights = '/orange/r.forghani/results/SwinUNETR/11-07-24_0301/model_best.pth'    #SWIN
        # args.trained_weights = '/orange/r.forghani/results/nnFormer/nnformer/fold_3/fold_3_model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/UXNET/3duxnet/fold_3/fold_3_model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/UNETR/fold_3/model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/TransBTS/fold_3/model_best.pth'
    elif args.fold == 4:
        args.trained_weights = '/orange/r.forghani/results/02-09-25_0253/model_best.pth'                # flare avg pool
        # args.trained_weights = '/orange/r.forghani/results/02-08-25_0417/model_best.pth'                # flare dwt pool
        # args.trained_weights = '/orange/r.forghani/results/01-07-25_1844/model_best.pth'              #flare_idwt_v2
        # args.trained_weights = '/orange/r.forghani/results/12-21-24_0006/model_best.pth'              #flare
        # args.trained_weights = '/orange/r.forghani/results/12-23-24_0256/model_best.pth'              #KiTS
        # args.trained_weights = '/orange/r.forghani/results/SwinUNETR/11-06-24_0758/model_best.pth'    #SWIN
        # args.trained_weights = '/orange/r.forghani/results/nnFormer/nnformer/fold_4/fold_4_model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/UXNET/3duxnet/fold_4/fold_4_model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/UNETR/fold_4/model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/TransBTS/fold_4/model_best.pth'

set_determinism(seed=0)
### extracting run_id of testing model
splitted_text = args.trained_weights[:args.trained_weights.rindex('/')]
run_id = splitted_text[splitted_text.rindex('/')+1:]
print(f'############## run id of pretrained model: {run_id} ################')

if args.network!='MSHEAD':
    args.output = os.path.join(args.output, args.network)   # '/orange/r.forghani/results/SwinUNETR'

if args.network == 'nnFormer':
    output_seg_dir = os.path.join(args.output, 'nnformer', f'fold_{args.fold}', 'output_seg')
elif args.network == 'UXNET':
    output_seg_dir = os.path.join(args.output, '3duxnet', f'fold_{args.fold}', 'output_seg')
elif args.network == 'UNETR':
    output_seg_dir = os.path.join(args.output, f'fold_{args.fold}', 'output_seg')
elif args.network == 'TransBTS':
    output_seg_dir = os.path.join(args.output, f'fold_{args.fold}', 'output_seg')
else:
    output_seg_dir = os.path.join(args.output, f'{run_id}', 'output_seg')

if not os.path.exists(output_seg_dir):
    os.makedirs(output_seg_dir)

test_transforms = data_transforms(args)
post_transforms = infer_post_transforms(args, test_transforms, out_classes, output_seg_dir)

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
        patch_size=2,
        in_chans=1,
        out_chans=out_classes,
        depths=[2,2,2,2],
        feat_size=[48,96,192,384],
        num_heads = [3,6,12,24],
        drop_path_rate=0.1,
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

elif args.network == 'nnFormer':
    model = nnFormer(
        crop_size = [96, 96, 96],
        input_channels=1,
        embedding_dim = 192,
        num_classes=out_classes,
        depths=[2, 2, 2, 2]
    ).to(device)

elif args.network == 'UXNET':
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

if args.network == 'TransBTS':
    print(f'network: {args.network} loading')
    _, model = TransBTS(dataset=args.dataset, _conv_repr=True, _pe_type='learned')
    model = model.to(device)

print(f'fold:{args.fold} - best model path:{args.trained_weights} ')
state_dict = torch.load(args.trained_weights)
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