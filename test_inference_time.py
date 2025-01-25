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

print(f'########### Running KITS Segmentation ################# \n')
parser = argparse.ArgumentParser(description='MSHEAD_ATTN hyperparameters for medical image segmentation')
## Input data hyperparameters
parser.add_argument('--root', type=str, default='', required=False, help='Root folder of all your images and labels')
parser.add_argument('--output', type=str, default='/orange/r.forghani/results', required=False, help='Output folder for both tensorboard and the best model')
parser.add_argument('--dataset', type=str, default='kits', required=False, help='Datasets: {feta, flare, amos}, Fyi: You can add your dataset here')

## Input model & training hyperparameters
parser.add_argument('--network', type=str, default='MSHEAD', required=False, help='Network models: {TransBTS, nnFormer, UNETR, SwinUNETR, 3DUXNET}')
parser.add_argument('--trained_weights', default='', required=False, help='Path of pretrained/fine-tuned weights')
parser.add_argument('--mode', type=str, default='test', help='Training or testing mode')
parser.add_argument('--sw_batch_size', type=int, default=4, help='Sliding window batch size for inference')
parser.add_argument('--overlap', type=float, default=0.5, help='Sub-volume overlapped percentage')

## Efficiency hyperparameters
parser.add_argument('--gpu', type=str, default='0', help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=1, help='Cache rate to cache your dataset into GPUs')
parser.add_argument('--num_workers', type=int, default=16, help='Number of workers')
parser.add_argument('--fold', type=int, default=0, help='current running fold')
parser.add_argument('--no_split', default=False, help='Not splitting into train and validation')
parser.add_argument('--plot', default=False, help='Plotting prediction or Not')

args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if not args.root:
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
    else:
        raise NotImplementedError(f'No such dataset: {args.dataset}')


test_samples, out_classes = data_loader(args)

test_files = [
    {"image": image_name, "label": label_name, "path": data_path}
    for image_name, label_name, data_path in zip(test_samples['images'], test_samples['labels'], test_samples['paths'])
]

set_determinism(seed=0)

test_transforms = data_transforms(args)
post_transforms = infer_post_transforms(args, test_transforms, out_classes)

## Inference Pytorch Data Loader and Caching
test_ds = CacheDataset(
    data=test_files, transform=test_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
test_loader = ThreadDataLoader(test_ds, batch_size=1, num_workers=0)

## Load Networks
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

if args.dataset != 'amos':
    if args.fold == 0:
        # args.trained_weights = '/orange/r.forghani/results/11-04-24_2125/model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/SwinUNETR/11-04-24_2018/model_best.pth'#SWIN
        # args.trained_weights = '/orange/r.forghani/results/nnFormer/nnformer/fold_0/fold_0_model_best.pth'
        args.trained_weights = '/orange/r.forghani/results/UXNET/3duxnet/fold_0/fold_0_model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/UNETR/fold_0/model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/TransBTS/fold_0/model_best.pth'
    elif args.fold == 1:
        # args.trained_weights = '/orange/r.forghani/results/11-03-24_0237/model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/SwinUNETR/11-08-24_0059/model_best.pth'#SWIN
        args.trained_weights = '/orange/r.forghani/results/nnFormer/nnformer/fold_1/fold_1_model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/UXNET/3duxnet/fold_1/fold_1_model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/UNETR/fold_1/model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/TransBTS/fold_1/model_best.pth'
    elif args.fold == 2:
        # args.trained_weights = '/orange/r.forghani/results/11-03-24_0331/model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/SwinUNETR/11-06-24_2219/model_best.pth'#SWIN
        args.trained_weights = '/orange/r.forghani/results/nnFormer/nnformer/fold_2/fold_2_model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/UXNET/3duxnet/fold_2/fold_2_model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/UNETR/fold_2/model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/TransBTS/fold_2/model_best.pth'
    elif args.fold == 3:
        # args.trained_weights = '/orange/r.forghani/results/11-03-24_0342/model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/SwinUNETR/11-07-24_0301/model_best.pth'#SWIN
        args.trained_weights = '/orange/r.forghani/results/nnFormer/nnformer/fold_3/fold_3_model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/UXNET/3duxnet/fold_3/fold_3_model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/UNETR/fold_3/model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/TransBTS/fold_3/model_best.pth'
    elif args.fold == 4:
        # args.trained_weights = '/orange/r.forghani/results/11-03-24_0358/model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/SwinUNETR/11-06-24_0758/model_best.pth'#SWIN
        args.trained_weights = '/orange/r.forghani/results/nnFormer/nnformer/fold_4/fold_4_model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/UXNET/3duxnet/fold_4/fold_4_model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/UNETR/fold_4/model_best.pth'
        # args.trained_weights = '/orange/r.forghani/results/TransBTS/fold_4/model_best.pth'


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
post_pred = AsDiscrete(argmax=True, to_onehot=out_classes)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

dice_vals = list()
s_time = time.time()
times = []
dummy_input = torch.randn(1, 1, 96, 96, 96)
with torch.no_grad():
    print(f'Warmup Iterations')
    dummy_input = dummy_input.to(device)
    for i in range(10):
        _ = model(dummy_input)
    print(f'Warmup Iterations Over')
    peak_memories = []
    # torch.cuda.reset_peak_memory_stats(device=device)
    for step, batch in enumerate(test_loader):
        torch.cuda.reset_peak_memory_stats(device=device)
        test_inputs, test_labels = (batch["image"].to(device), batch["label"].to(device))
        path = batch["path"]
        # print(f'path for the image {step}: {path} shape:{test_inputs.shape}')
        roi_size = (96, 96, 96)
        # start_infer = time.time()
        test_outputs = sliding_window_inference(
            test_inputs, roi_size, args.sw_batch_size, model, overlap=args.overlap
        )
        # end_infer = time.time()
        # case_time = end_infer - start_infer
        # times.append(case_time)
        # print(f'inferece for case:{step}:::: {case_time:.2f}s')
        # 3) Get the peak memory for this iteration
        iteration_peak = torch.cuda.max_memory_allocated(device=device)
        peak_memories.append(iteration_peak)
        print(f"Iteration {i+1} - Peak memory usage: {iteration_peak / 1024**2:.2f} MB")


        test_labels_list = decollate_batch(test_labels)
        test_labels_convert = [
            post_label(test_label_tensor) for test_label_tensor in test_labels_list
        ]

        test_outputs_list = decollate_batch(test_outputs)
        test_output_convert = [
            post_pred(test_pred_tensor) for test_pred_tensor in test_outputs_list
        ]

        dice_metric(y_pred=test_output_convert, y=test_labels_convert)
        
    # # Check the overall peak
    # peak_memory = torch.cuda.max_memory_allocated(device=device)
    # print(f"Peak memory usage across all iterations: {peak_memory / 1024**2:.2f} MB")
    # 4) Compute average (and other stats if desired)
    avg_peak = sum(peak_memories) / len(peak_memories)
    print(f"\nAverage peak memory usage across {len(test_loader)} iterations: {avg_peak / 1024**2:.2f} MB")

    dice = dice_metric.aggregate().item()
    dice_metric.reset()


test_time = time.time() - s_time
print(f"test takes {datetime.timedelta(seconds=int(test_time))}")
print(f'mean test dice: {dice}')
# avg_time = np.mean(times)
# print(f"Average inference time: {avg_time:.2f}s --- {avg_time*1000:.2f} ms")

