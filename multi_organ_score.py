# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 13:18:55 2020


@author: peter
"""
import os
import argparse

import pandas as pd
import os
import nibabel as nib
import numpy as np
import statistics as stat
from natsort import natsorted

parser = argparse.ArgumentParser(description='KiTS Evaluation segmentation')
## Input data hyperparameters

parser.add_argument('--output', type=str, default='/orange/r.forghani/results', required=False, help='Output folder for both tensorboard and the best model')
parser.add_argument('--dataset', type=str, default='flare', required=False, help='Datasets: {feta, flare, amos}, Fyi: You can add your dataset here')
parser.add_argument('--fold', type=int, default=0, help='current running fold')
parser.add_argument('--network', type=str, default='MSHEAD', required=False, help='Network models: {TransBTS, nnFormer, UNETR, SwinUNETR, 3DUXNET}')

args = parser.parse_args()

if args.dataset == 'flare':
    model_id_dict = {}
    gt_dir = "/blue/r.forghani/share/flare_data/labelsTs"
elif args.dataset == 'amos':
    model_id_dict = {}
    gt_dir = "/blue/r.forghani/share/flare_data/labelsTs"
elif args.dataset == 'kits':
    if args.network == 'MSHEAD':
        model_id_dict = {0: '11-04-24_2125', 1:'11-03-24_0237', 2:'11-03-24_0331', 3:'11-03-24_0342', 4:'11-03-24_0358'}
    elif args.network == 'SwinUNETR':
        model_id_dict = {0: '11-04-24_2018', 1:'11-08-24_0059', 2:'11-06-24_2219', 3:'11-07-24_0301', 4:'11-06-24_0758'}
    elif args.network == 'nnFormer' or args.network=='UXNET' or args.network=='UNETR':
        model_id_dict = {0: 'fold_0', 1:'fold_1', 2:'fold_2', 3:'fold_3', 4:'fold_4'}
    gt_dir = '/blue/r.forghani/share/kits2019/labelsTr'
else:
    raise NotImplementedError(f'No such dataset: {args.dataset}')



def dice_score_organ(im1, im2):
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    # im2 = np.asarray(im2).astype(np.bool)


    if im1.shape != im2.shape:
        raise ValueError('Shape mismatch: im1 and im2 must have the same shape')


    intersection = np.logical_and(im1 , im2)


    return (2. * intersection.sum() + 0.0000001) / (im1.sum() + im2.sum() + 0.0000001)


## Model Prediction
model_id = model_id_dict[args.fold]
if args.network == 'nnFormer':
    pred_dir = f'/orange/r.forghani/results/{args.network}/nnformer/{model_id}/output_seg'
elif args.network == 'UXNET':
    pred_dir = f'/orange/r.forghani/results/{args.network}/3duxnet/{model_id}/output_seg'
elif args.network == 'UNETR':
    pred_dir = f'/orange/r.forghani/results/{args.network}/{model_id}/output_seg'
else:
    pred_dir =f'/orange/r.forghani/results/{args.network}/{model_id}/output_seg'
# pred_dir ="/orange/r.forghani/results/UXNET/output_seg"

print(f'pred:{pred_dir} ground truth:{gt_dir}')

# spleen = []
kidney = []
tumor = []
# liver = []
# pancreas = []

subject_list = []
sub_list = []
all_subjects = []
count = 0


for label in natsorted(os.listdir(pred_dir)):
    subj = label
    label_pred = os.path.join(pred_dir, subj, subj + '_seg.nii.gz')
    
    print(f'subj: {subj}')
    print(f'label_pred: {label}')
    # Exlude data with multiple size tumors
    # if '00001' in subj or '00041' in subj:
    #     continue
    
    

    if args.dataset == 'kits':
        label_gt = os.path.join(gt_dir, label.split('imaging')[0] + 'segmentation'+'.nii.gz')
    else:
        label_gt = os.path.join(gt_dir, label.split('_0000')[0] + '.nii.gz')
    print(f'label gt:{label_gt}')


    # label_gt = gt_file
    pred_nib = nib.load(label_pred)
    gt_nib = nib.load(label_gt)


    pred = pred_nib.get_fdata()
    gt = gt_nib.get_fdata()

    ################# Filtration of GT
    ### calculate connected component from label_gt:
    ### find out indexes of component with <1cm3 volume
    ### set those indexes in gt as 0

    ################ Size Identification --> Now assuming each image has identical size tumors 
    ###### multiscale evaluation 
    #### when data has only one size tumor
    ### know that this tumor is small/big/medium
    


    pred = np.transpose(pred, (2, 0, 1))
    gt = np.transpose(gt, (2, 0, 1))


    pred_mat = np.zeros((1, pred.shape[0], pred.shape[1], pred.shape[2]))
    gt_mat = np.zeros((1, pred.shape[0], pred.shape[1], pred.shape[2]))
    

    # Kidney with Tumor
    idx_pred = np.where(pred != 0)
    idx_gt = np.where(gt != 0)
    
    pred_mat[pred_mat != 0] = 0
    gt_mat[gt_mat != 0] = 0
    
    pred_mat[0, idx_pred[0], idx_pred[1], idx_pred[2]] = 1
    gt_mat[0, idx_gt[0], idx_gt[1], idx_gt[2]] = 1
    dice_kidney = dice_score_organ(pred_mat, gt_mat)
    kidney.append(dice_kidney)
    subject_list.append(dice_kidney)


    # Tumor
    idx_pred = np.where(pred == 2)
    idx_gt = np.where(gt == 2)

    pred_mat[pred_mat != 0] = 0
    gt_mat[gt_mat != 0] = 0
    pred_mat[0, idx_pred[0], idx_pred[1], idx_pred[2]] = 1
    gt_mat[0, idx_gt[0], idx_gt[1], idx_gt[2]] = 1
    
    dice_tumor = dice_score_organ(pred_mat, gt_mat)
    tumor.append(dice_tumor)
    subject_list.append(dice_tumor)

    ### size_dict[SMALL] = dice_tumor


    avg_dice = (dice_kidney + dice_tumor)/2
    count += 1
    print(f'\n ################ count:{count} --- --- Dataset: {label} ################')

   
    print('Kidney with Tumor DICE: {}'.format(dice_kidney))
    print('Tumor DICE: {}'.format(dice_tumor))
    print('Avg DICE: {}'.format(avg_dice))

    print(f'########################################################################### \n')

    all_subjects.append([stat.mean(subject_list), label])
    sub_list.append('All Subjects')
    subject_list = []


all_organs = kidney + tumor

# all_organs = pancreas

print('Mean Kidney with Tumot DICE: {}'.format(stat.mean(kidney)))
print('Stdev Kidney with Tumor DICE: {}'.format(stat.stdev(kidney)))
print('Mean Tumor DICE: {}'.format(stat.mean(tumor)))
print('Stdev Tumor DICE: {}'.format(stat.stdev(tumor)))

print('All Organ Mean DICE: {} /n'.format(stat.mean(all_organs)))
print('All Organ Stdev DICE: {} /n'.format(stat.stdev(all_organs)))
