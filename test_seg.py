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

parser = argparse.ArgumentParser(description='3D UX-Net inference hyperparameters for medical image segmentation')
## Input data hyperparameters
parser.add_argument('--root', type=str, default='/blue/r.forghani/share/flare_data', required=False, help='Root folder of all your images and labels')
parser.add_argument('--output', type=str, default='/orange/r.forghani/results', required=False, help='Output folder for both tensorboard and the best model')
parser.add_argument('--dataset', type=str, default='flare', required=False, help='Datasets: {feta, flare, amos}, Fyi: You can add your dataset here')

## Input model & training hyperparameters
parser.add_argument('--network', type=str, default='MSHEAD', required=False, help='Network models: {TransBTS, nnFormer, UNETR, SwinUNETR, 3DUXNET}')
parser.add_argument('--trained_weights', default='', required=False, help='Path of pretrained/fine-tuned weights')
parser.add_argument('--mode', type=str, default='test', help='Training or testing mode')
parser.add_argument('--sw_batch_size', type=int, default=4, help='Sliding window batch size for inference')
parser.add_argument('--overlap', type=float, default=0.5, help='Sub-volume overlapped percentage')

## Efficiency hyperparameters
parser.add_argument('--gpu', type=str, default='0', help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=1, help='Cache rate to cache your dataset into GPUs')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
parser.add_argument('--fold', type=int, default=0, help='current running fold')
parser.add_argument('--no_split', default=False, help='No splitting into train and validation')
parser.add_argument('--plot', default=False, help='plotting prediction or not')

args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

test_samples, out_classes = data_loader(args)

test_files = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(test_samples['images'], test_samples['labels'])
]

set_determinism(seed=0)

test_transforms = data_transforms(args)
post_transforms = infer_post_transforms(args, test_transforms, out_classes)

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

# if args.fold == 0:
#     # args.trained_weights = '/orange/r.forghani/results/09-18-24_0219/model_best.pth'
#     args.trained_weights = '/orange/r.forghani/results/10-14-24_1411/model_best.pth'
# elif args.fold == 1:
#     # args.trained_weights = '/orange/r.forghani/results/09-20-24_0448/model_best.pth'
#     args.trained_weights = '/orange/r.forghani/results/10-14-24_1437/model_best.pth'
# elif args.fold == 2:
#     # args.trained_weights = '/orange/r.forghani/results/09-21-24_1416/model_best.pth'
#     args.trained_weights = '/orange/r.forghani/results/10-14-24_1536/model_best.pth'
# elif args.fold == 3:
#     # args.trained_weights = '/orange/r.forghani/results/09-18-24_2221/model_best.pth'
#     args.trained_weights = '/orange/r.forghani/results/10-13-24_0325/model_best.pth'
# elif args.fold == 4:
#     # args.trained_weights = '/orange/r.forghani/results/09-18-24_2224/model_best.pth'
#     args.trained_weights = '/orange/r.forghani/results/10-14-24_1624/model_best.pth'

if args.fold == 0:
    args.trained_weights = '/orange/r.forghani/results/09-26-24_0418/model_best.pth'
elif args.fold == 1:
    args.trained_weights = '/orange/r.forghani/results/09-26-24_0428/model_best.pth'
elif args.fold == 2:
    args.trained_weights = '/orange/r.forghani/results/09-26-24_0432/model_best.pth'
elif args.fold == 3:
    args.trained_weights = '/orange/r.forghani/results/09-26-24_0441/model_best.pth'
elif args.fold == 4:
    args.trained_weights = '/orange/r.forghani/results/09-26-24_1909/model_best.pth'

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

# dice_vals = list()
s_time = time.time()
dice_vals_individual = []
dice_vals_aggregated = []
with torch.no_grad():
    for step, batch in enumerate(test_loader):
        test_inputs, test_labels = batch["image"].to(device), batch["label"].to(device)

        test_outputs = sliding_window_inference(test_inputs, (96, 96, 96), args.sw_batch_size, model)

        # Decollate and convert labels and outputs
        test_labels_list = decollate_batch(test_labels)
        test_labels_convert = [post_label(test_label_tensor) for test_label_tensor in test_labels_list]
        
        test_outputs_list = decollate_batch(test_outputs)
        test_output_convert = [post_pred(test_pred_tensor) for test_pred_tensor in test_outputs_list]

        # --- Individual mean (across class) Dice score per sample ---
        for pred, label in zip(test_output_convert, test_labels_convert):
            # print(f'pred:{pred.shape} label:{label.shape}')
            individual_dice_scores = DiceMetric(include_background=False, reduction="none")(pred.unsqueeze(0), label.unsqueeze(0))
            individual_dice_score = torch.mean(individual_dice_scores).item()
            dice_vals_individual.append(individual_dice_score)      # per sample mean dice score
        
        # --- Aggregated Dice score ---
        dice_metric(y_pred=test_output_convert, y=test_labels_convert)  # Update Dice metric for this batch
        dice = dice_metric.aggregate().item()  # This gives cumulative average Dice score so far
        dice_vals_aggregated.append(dice)

    # Reset the metric after evaluation
    dice_metric.reset()

# Final average Dice score for the entire dataset using the aggregated metric
final_aggregated_dice = dice_vals_aggregated[-1]
print(f"Final aggregated Dice score (over all batches): {final_aggregated_dice}")

# Print or analyze individual Dice scores
print(f"Individual Dice scores: {dice_vals_individual}\n")
print(f"mean dice score from individual calculation: {np.mean(dice_vals_individual)}")
print(f"\n Cumulative Avg Dice: {dice_vals_aggregated}")

test_time = time.time() - s_time
print(f"test takes {datetime.timedelta(seconds=int(test_time))}")
# print(f'mean test dice: {mean_dice_test}')

