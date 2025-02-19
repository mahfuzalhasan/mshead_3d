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
parser.add_argument('--dataset', type=str, default='kits23', required=False, help='Datasets: {feta, flare, amos}, Fyi: You can add your dataset here')
parser.add_argument('--fold', type=int, default=0, help='current running fold')
parser.add_argument('--network', type=str, default='MSHEAD', required=False, help='Network models: {TransBTS, nnFormer, UNETR, SwinUNETR, 3DUXNET}')

args = parser.parse_args()

if args.dataset == 'flare':
    if args.network == 'MSHEAD':
        # model_id_dict = {0: '12-20-24_0342', 1:'12-20-24_1658', 2:'12-20-24_1836', 3:'12-20-24_1943', 4:'12-21-24_0006'}  #idwt
        model_id_dict = {0: '01-07-25_0204', 1:'01-07-25_0102', 2:'01-07-25_1307', 3:'01-07-25_1708', 4:'01-07-25_1844'}    # idwt_v2
    gt_dir = "/blue/r.forghani/share/flare_data/labelsTs"
elif args.dataset == 'amos':
    model_id_dict = {}
    gt_dir = "/blue/r.forghani/share/flare_data/labelsTs"
elif args.dataset == 'kits':
    if args.network == 'MSHEAD':
        model_id_dict = {0: '12-22-24_1727', 1:'12-23-24_0128', 2:'12-23-24_0145', 3:'12-23-24_0240', 4:'12-23-24_0256'}
    elif args.network == 'SwinUNETR':
        model_id_dict = {0: '11-04-24_2018', 1:'11-08-24_0059', 2:'11-06-24_2219', 3:'11-07-24_0301', 4:'11-06-24_0758'}
    elif args.network == 'nnFormer' or args.network=='UXNET' or args.network=='UNETR' or args.network=='TransBTS':
        model_id_dict = {0: 'fold_0', 1:'fold_1', 2:'fold_2', 3:'fold_3', 4:'fold_4'}
    gt_dir = '/blue/r.forghani/share/kits2019/labelsTr'
elif args.dataset == 'kits23':
    if args.network == 'MSHEAD':
        model_id_dict = {0: 'fold_0', 1:'fold_1', 2:'fold_2', 3:'fold_1', 4:'fold_4'}
    elif args.network == 'SwinUNETR':
        model_id_dict = {0: '11-04-24_2018', 1:'11-08-24_0059', 2:'11-06-24_2219', 3:'11-07-24_0301', 4:'11-06-24_0758'}
    elif args.network == 'nnFormer' or args.network=='UXNET' or args.network=='UNETR' or args.network=='TransBTS':
        model_id_dict = {0: 'fold_0', 1:'fold_1', 2:'fold_2', 3:'fold_3', 4:'fold_4'}
    gt_dir = '/blue/r.forghani/share/kits23/labelsTr'
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
elif args.network == 'UNETR' or args.network=='TransBTS':
    pred_dir = f'/orange/r.forghani/results/{args.network}/{model_id}/output_seg'
else:
    pred_dir =f'/orange/r.forghani/results/waveformer/{model_id}/output_seg'
# pred_dir ="/orange/r.forghani/results/UXNET/output_seg"

print(f'pred:{pred_dir} ground truth:{gt_dir}')

cyst = []
tumor = []
kidney = []

subject_list = []
sub_list = []
all_subjects = []
count = 0


for label in os.listdir(pred_dir):
    subj = label
    case_id = label.split("_")[1]
    print(f'case id: {case_id}')
    label_pred = os.path.join(pred_dir, subj, subj + '_seg.nii.gz')


    label_gt = os.path.join(gt_dir, f'train_{case_id}_segmentation.nii.gz')


    # label_gt = gt_file
    pred_nib = nib.load(label_pred)
    gt_nib = nib.load(label_gt)


    pred = pred_nib.get_fdata()
    gt = gt_nib.get_fdata()


    pred = np.transpose(pred, (2, 0, 1))
    gt = np.transpose(gt, (2, 0, 1))


    pred_mat = np.zeros((1, pred.shape[0], pred.shape[1], pred.shape[2]))
    gt_mat = np.zeros((1, pred.shape[0], pred.shape[1], pred.shape[2]))
    
    # Cyst
    idx_pred = np.where(pred == 3)
    pred_mat[0, idx_pred[0], idx_pred[1], idx_pred[2]] = 1
    idx_gt = np.where(gt == 3)
    gt_mat[0, idx_gt[0], idx_gt[1], idx_gt[2]] = 1
    dice_cyst = dice_score_organ(pred_mat, gt_mat)
    cyst.append(dice_cyst)
    subject_list.append(dice_cyst)


    # tumor
    idx_pred = np.where(pred == 2)
    pred_mat[pred_mat != 0] = 0
    gt_mat[gt_mat != 0] = 0
    pred_mat[0, idx_pred[0], idx_pred[1], idx_pred[2]] = 1
    idx_gt = np.where(gt == 2)
    gt_mat[0, idx_gt[0], idx_gt[1], idx_gt[2]] = 1
    dice_tumor = dice_score_organ(pred_mat, gt_mat)
    tumor.append(dice_tumor)
    subject_list.append(dice_tumor)


    # kidney
    idx_pred = np.where(pred == 1)
    pred_mat[pred_mat != 0] = 0
    gt_mat[gt_mat != 0] = 0
    pred_mat[0, idx_pred[0], idx_pred[1], idx_pred[2]] = 1
    idx_gt = np.where(gt == 1)
    gt_mat[0, idx_gt[0], idx_gt[1], idx_gt[2]] = 1
    dice_kidney = dice_score_organ(pred_mat, gt_mat)
    kidney.append(dice_kidney)
    subject_list.append(dice_kidney)

    avg_dice = (dice_cyst + dice_tumor + dice_kidney)/4
    count += 1
    print(f'\n ################ count:{count} --- --- Dataset: {label} ################')

    print('cyst DICE: {}'.format(dice_cyst))
    print('Right tumor DICE: {}'.format(dice_tumor))
    print('kidney DICE: {}'.format(dice_kidney))
    print('Avg DICE: {}'.format(avg_dice))

    print(f'########################################################################### \n')

    all_subjects.append([stat.mean(subject_list), label])
    sub_list.append('All Subjects')
    subject_list = []


all_organs = cyst + tumor + kidney

# all_organs = pancreas


print('Mean cyst DICE: {}'.format(stat.mean(cyst)))
print('Stdev cyst DICE: {}'.format(stat.stdev(cyst)))
print('Mean Right tumor DICE: {}'.format(stat.mean(tumor)))
print('Stdev Right tumor DICE: {}'.format(stat.stdev(tumor)))
print('Mean kidney DICE: {}'.format(stat.mean(kidney)))
print('Stdev kidney DICE: {}'.format(stat.stdev(kidney)))
print('All Organ Mean DICE: {} /n'.format(stat.mean(all_organs)))
print('All Organ Stdev DICE: {} /n'.format(stat.stdev(all_organs)))