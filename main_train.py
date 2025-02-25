#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 11:06:19 2021

@author: leeh43
"""
import shutil
from monai.utils import set_determinism
from monai.transforms import AsDiscrete
from networks.UXNet_3D.network_backbone import UXNET
from networks.msHead_3D.network_backbone import MSHEAD_ATTN
from monai.networks.nets import UNETR, SwinUNETR
from networks.nnFormer.nnFormer_seg import nnFormer
from networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from networks.mednext.create_mednext_v1 import create_mednext_v1
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch, ThreadDataLoader, PersistentDataset

import torch
from torch.utils.tensorboard import SummaryWriter
from load_datasets_transforms import data_loader, data_transforms

import os
import numpy as np
from tqdm import tqdm
import datetime
import argparse
import time

print(f'########### Running Flare Segmentation ################# \n')
parser = argparse.ArgumentParser(description='MSHEAD_ATTN hyperparameters for medical image segmentation')
## Input data hyperparameters
parser.add_argument('--root', type=str, default='', required=False, help='Root folder of all your images and labels')
parser.add_argument('--output', type=str, default='/orange/r.forghani/results', required=False, help='Output folder for both tensorboard and the best model')
parser.add_argument('--dataset', type=str, default='flare', required=False, help='Currently supporting datasets: {flare, amos, kits}, Fyi: You can add your dataset here')

## Input model & training hyperparameters
parser.add_argument('--network', type=str, default='MSHEAD', help='Network models: {MSHEAD, TransBTS, nnFormer, UNETR, SwinUNETR, UXNET}')
parser.add_argument('--mode', type=str, default='train', help='Training or testing mode')
parser.add_argument('--pretrain', default=False, help='Have pretrained weights or not')
parser.add_argument('--pretrained_weights', type=str, default=None, help='Path of pretrained weights')
parser.add_argument('--batch_size', type=int, default='2', help='Batch size for subject input')
parser.add_argument('--crop_sample', type=int, default='2', help='Number of cropped sub-volumes for each subject')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for training')
parser.add_argument('--optim', type=str, default='AdamW', help='Optimizer types: Adam / AdamW')
parser.add_argument('--max_iter', type=int, default=40000, help='Maximum iteration steps for training')
parser.add_argument('--eval_step', type=int, default=500, help='Per steps to perform validation')
parser.add_argument('--resume', type=int, default=0, help='resume training from an earlier iteration')
parser.add_argument('--save_iteration', type=int, default=500, help='resume training from an earlier iteration')
### validation
# use overlap = 0.25 --> shovon
parser.add_argument('--sw_batch_size', type=int, default=2, help='Sliding window batch size for inference')
parser.add_argument('--overlap', type=float, default=0.5, help='Sub-volume overlapped percentage')
## Efficiency hyperparameters
parser.add_argument('--gpu', type=int, default=0, help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=1, help='Cache rate to cache your dataset into memory')
parser.add_argument('--num_workers', type=int, default=16, help='Number of workers')
parser.add_argument('--fold', type=int, default=0, help='current running fold')
parser.add_argument('--no_split', default=False, help='Not splitting into train and validation')

args = parser.parse_args()
print(f'################################')
print(f'args:{args}')
print('#################################')
args.resume = bool(int(args.resume))
print("Resuming from Earlier Train ", args.resume)
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
if not args.root:
    if args.dataset == 'flare':
        args.root = '/blue/r.forghani/share/flare_data'
    elif args.dataset == 'amos':
        args.root = '/blue/r.forghani/share/amoss22/amos22'
    elif args.dataset == 'kits':
        args.root = '/blue/r.forghani/share/kits2019'
    elif args.dataset == 'kits23':
        # Shovon --> set dataset path here
        args.root = '/blue/r.forghani/share/kits23'
    else:
        raise NotImplementedError(f'No such dataset: {args.dataset}')
        
print(f'Root folder for data: {args.root}')
print('Used GPU: {}'.format(args.gpu))
print(f'############## Training on Fold:{args.fold} ################## \n')

run_id = datetime.datetime.today().strftime('%m-%d-%y_%H%M')
print(f'$$$$$$$$$$$$$ run_id:{run_id} $$$$$$$$$$$$$')

train_samples, valid_samples, out_classes = data_loader(args)

train_files = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_samples['images'], train_samples['labels'])
]
val_files = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(valid_samples['images'], valid_samples['labels'])
]
print(f'train files:{len(train_files)} val files:{len(val_files)}')
print(f' \n ****************** Validation File List :\n {val_files} \n ******************* \n')


set_determinism(seed=0)

train_transforms, val_transforms = data_transforms(args)

# train_files = train_files[:20]
print(f' \n ****************** train_files List :\n {train_files} \n ******************* \n')

## Train Pytorch Data Loader and Caching
print('Start caching datasets!')
if args.dataset == "kits23":
    args.eval_step = 588        # 196 *3 = val after every 3 epoch
    args.max_iter = 39984       # 196 * 204 = total 204 epochs of training
    args.save_iteration = 25284 # save after 129 epoch * 196 itr
    train_cache_dir = f'/blue/r.forghani/share/kits23_cache_train/{args.fold}'
    val_cache_dir = f'/blue/r.forghani/share/kits23_cache_val/{args.fold}'

    # Clean old cache before training (optional)
    for cache_dir in [train_cache_dir, val_cache_dir]:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

    train_ds = PersistentDataset(data=train_files, transform=train_transforms,cache_dir=train_cache_dir)
    val_ds = PersistentDataset(data=val_files, transform=val_transforms, cache_dir=val_cache_dir)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.num_workers)

elif args.dataset == "flare":
    ## Train Pytorch Data Loader and Caching
    print('Start caching datasets!')
    train_ds = CacheDataset(data=train_files, transform=train_transforms,cache_rate=args.cache_rate, num_workers=args.num_workers)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)

    train_loader = ThreadDataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = ThreadDataLoader(val_ds, batch_size=1, num_workers=0)
    




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
elif args.network == 'UXNET':   # shovon run uxnet for fold 3,4
    model = UXNET(
        in_chans=1,
        out_chans=out_classes,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        spatial_dims=3,
    ).to(device)

elif args.network == 'nnFormer':
    model = nnFormer(
        crop_size = [96, 96, 96],
        input_channels=1,
        embedding_dim = 192,
        num_classes=out_classes,
        depths=[2, 2, 2, 2]
    ).to(device)

elif args.network == 'TransBTS':
    _, model = TransBTS(dataset=args.dataset, _conv_repr=True, _pe_type='learned')
    model = model.to(device)

elif args.network == 'MEDNEXT':
    model = create_mednext_v1(num_input_channels=1, 
                        num_classes=out_classes, 
                        model_id='M', 
                        kernel_size=5, 
                        deep_supervision=False).to(device)

print('Chosen Network Architecture: {}'.format(args.network))
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")

if args.dataset == 'amos' and args.pretrained_weights is not None:
    print('Pretrained weight is found! Start to load weight from: {}'.format(args.pretrained_weights), flush=True)
    state_dict = torch.load(args.pretrained_weights)
    pretrained_dict = state_dict['model']
    model_dict = model.state_dict()
    # Filter out layers that have a size mismatch
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    # Update the model with the filtered weights
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    # model.load_state_dict(state_dict['model'], strict=False
    print(f'########### pretrained weights loaded ###############\n')

## Define Loss function and optimizer
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
print('Loss for training: {}'.format('DiceCELoss'))
if args.optim == 'AdamW':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
elif args.optim == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
print('Optimizer for training: {}, learning rate: {}'.format(args.optim, args.lr))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=10)

def validation(val_loader):
    # model_feat.eval()
    model.eval()
    dice_vals = list()
    s_time = time.time()
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))
            # val_outputs = model(val_inputs)
            if args.dataset == "kits23":
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), args.sw_batch_size, model, overlap=args.overlap)
            else:
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), args.sw_batch_size, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            # dice = dice_metric.aggregate().item()
            # dice_vals.append(dice)
            # epoch_iterator_val.set_description(
            #     "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice)
            # )
        dice = dice_metric.aggregate().item()
        dice_metric.reset()

    
    # mean_dice_val = np.mean(dice_vals)
    writer.add_scalar('Validation Segmentation Dice Val', dice, global_step)
    val_time = time.time() - s_time
    print(f"val takes {datetime.timedelta(seconds=int(val_time))}")
    return dice


def save_model(model, optimizer, lr_scheduler, iteration, run_id, dice_score, global_step_best, save_dir, best=False):
    s_time = time.time()
    save_file_path = os.path.join(save_dir, 'model_{}.pth'.format(iteration))
    if best:
        save_file_path = os.path.join(save_dir, 'model_best.pth')

    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'dice_score': dice_score,
                  'global_step': iteration,
                  'global_step_best':global_step_best,
                  'run_id':str(run_id)}
    torch.save(save_state, save_file_path)
    save_time = time.time() - s_time
    # print(f"model saved at iteration:{iteration} and took: {datetime.timedelta(seconds=int(save_time))}")


def train(global_step, train_loader, dice_val_best, global_step_best):
    s_time = time.time()
    model.train()
    step = 0
    epoch_loss_values = []
    previous_step = 0
    # epoch_iterator = tqdm(
    #     train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    # )
    print(f'######### new epoch started. Global Step:{global_step} ###############')
    # total training data--> 272. Batch 2. This loop will run for 272/2 = 136 times
    dice_val = -1.0
    for step, batch in enumerate(train_loader):     
        global_step += 1
        x, y = (batch["image"].to(device), batch["label"].to(device))       # x->B,C,H,W,D = 2,1,96,96,96. y same
        # with torch.no_grad():
        #     g_feat, dense_feat = model_feat(x)
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        # epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss_values.append(loss.item())

        # print after every 100 iteration
        if global_step % 100 == 0:
            print(f'step:{global_step} completed. Avg Loss:{np.mean(epoch_loss_values)}')
            num_steps = global_step - previous_step
            time_100 = time.time() - s_time
            print(f"step {num_steps} took: {datetime.timedelta(seconds=int(time_100))} \n ")
            previous_step = global_step


        # # saving model after every 500 iteration
        # if (global_step % (eval_num) == 0) and global_step!=0:
        #     save_model(model, optimizer, scheduler, global_step, run_id, dice_val_best, global_step_best, root_dir)
        
        if (global_step < args.save_iteration) and (global_step % (30*len(train_loader)) == 0):
            dice_val = validation(val_loader)
            model.train()
        elif (global_step >= args.save_iteration) and (global_step % (3*len(train_loader)) == 0):
            dice_val = validation(val_loader)
            model.train()
        elif global_step == max_iterations:
            dice_val = validation(val_loader)
            model.train()
                
        # metric_values.append(dice_val)
        if dice_val > dice_val_best:
            dice_val_best = dice_val
            global_step_best = global_step
            save_model(model, optimizer, scheduler, global_step, run_id, dice_val_best, global_step_best, root_dir, best=True)
            print(
                "Model Was Saved ! Current Best Avg. Dice: {} at step {}".format(
                    dice_val_best, global_step_best
                )
            )
            
        elif global_step % eval_num == 0:
            if dice_val > 0:
                print(
                    "Not Best Model. Current Best Avg. Dice: {} from step:{}, Current Avg. Dice: {}".format(dice_val_best, global_step_best, dice_val)
                )
            else:
                print('dice calculation has not stared yet')
            save_model(model, optimizer, scheduler, global_step, run_id, dice_val_best, global_step_best, root_dir)
            # scheduler.step(dice_val)

            # setting model to train mode again
        
        
        # saving loss for every iteration
        writer.add_scalar('Training Loss_Itr', loss.data, global_step)
        
    
    train_time = time.time() - s_time
    print(f"train takes {datetime.timedelta(seconds=int(train_time))}")
    
    return global_step, dice_val_best, global_step_best


max_iterations = args.max_iter
print('Maximum Iterations for training: {}'.format(str(args.max_iter)), flush=True)
eval_num = args.eval_step
post_label = AsDiscrete(to_onehot=out_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=out_classes)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
# metric_values = []


### if you need to resume from a previous checkpoint.
### run with python main_train.py --resume True
### Then set model_path here
if args.resume:
    if args.fold == 0:
        model_path = '/orange/r.forghani/results/02-20-25_0844/model_best.pth'
    print(f'Resuming Model: {model_path}')
    state_dict = torch.load(model_path, weights_only=False)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    scheduler.load_state_dict(state_dict['lr_scheduler'])
    global_step = state_dict['global_step']                 # same for best model
    global_step_best = state_dict['global_step_best']       # same for best model
    run_id = state_dict['run_id']
    dice_val_best = state_dict['dice_score']
    print(f'$$$$$$$$$$$$$ using old run_id:{run_id} $$$$$$$$$$$$$')
    print(f'starting from global step:{global_step}')
    print(f'best val dice:{dice_val_best} at step:{global_step_best}')
    # exit()

root_dir = os.path.join(args.output, run_id)
if os.path.exists(root_dir) == False:
    os.makedirs(root_dir)
    
t_dir = os.path.join(root_dir, 'tensorboard')
if os.path.exists(t_dir) == False:
    os.makedirs(t_dir)
writer = SummaryWriter(log_dir=t_dir)


while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(
        global_step, train_loader, dice_val_best, global_step_best
    )
    print(f'completed: {global_step} iterations')
    print(f'best so far:{dice_val_best} at iteration:{global_step_best}')