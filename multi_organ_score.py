# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 13:18:55 2020


@author: peter
"""


import pandas as pd
import os
import nibabel as nib
import numpy as np
import statistics as stat


def dice_score_organ(im1, im2):
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    # im2 = np.asarray(im2).astype(np.bool)


    if im1.shape != im2.shape:
        raise ValueError('Shape mismatch: im1 and im2 must have the same shape')


    intersection = np.logical_and(im1 , im2)


    return (2. * intersection.sum() + 0.0000001) / (im1.sum() + im2.sum() + 0.0000001)


## Model Prediction
# pred_dir = os.path.join('/nfs/masi/leeh43/repuxnet/out_FLARE_repuxnet_conv_matrix_alldata_sample_2')
pred_dir ="/orange/r.forghani/results/09-25-24_2042/output_seg"
# pred_dir ="/orange/r.forghani/results/UXNET/output_seg"


## Ground Truth Label
# gt_dir = os.path.join('/nfs/masi/leeh43/FLARE2021/TRAIN_MASK')
# gt_dir = "/blue/r.forghani/share/flare_data/labelsTs"
gt_dir = "/blue/r.forghani/share/kits2019/labelsTs"

print(f'pred:{pred_dir}')




cyst = []
tumor = []
kidney = []
pancreas = []

subject_list = []
sub_list = []
all_subjects = []
count = 0


for label in os.listdir(pred_dir):
    subj = label
    label_pred = os.path.join(pred_dir, subj, subj + '_seg.nii.gz')
    label_gt = os.path.join(gt_dir, label.split('_imaging')[0] + '_segmentation.nii.gz')


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
    print(f'index with cyst: {len(idx_pred)} {idx_pred}')
    pred_mat[0, idx_pred[0], idx_pred[1], idx_pred[2]] = 1
    idx_gt = np.where(gt == 3)
    gt_mat[0, idx_gt[0], idx_gt[1], idx_gt[2]] = 1
    dice_cyst = dice_score_organ(pred_mat, gt_mat)
    cyst.append(dice_cyst)
    subject_list.append(dice_cyst)


    # Tumor
    idx_pred = np.where(pred == 2)
    pred_mat[pred_mat != 0] = 0
    gt_mat[gt_mat != 0] = 0
    pred_mat[0, idx_pred[0], idx_pred[1], idx_pred[2]] = 1
    idx_gt = np.where(gt == 2)
    gt_mat[0, idx_gt[0], idx_gt[1], idx_gt[2]] = 1
    dice_tumor = dice_score_organ(pred_mat, gt_mat)
    tumor.append(dice_tumor)
    subject_list.append(dice_tumor)


    # Kidney
    idx_pred = np.where(pred == 1)
    pred_mat[pred_mat != 0] = 0
    gt_mat[gt_mat != 0] = 0
    pred_mat[0, idx_pred[0], idx_pred[1], idx_pred[2]] = 1
    idx_gt = np.where(gt == 1)
    gt_mat[0, idx_gt[0], idx_gt[1], idx_gt[2]] = 1
    dice_kidney = dice_score_organ(pred_mat, gt_mat)
    kidney.append(dice_kidney)
    subject_list.append(dice_kidney)

    avg_dice = (dice_cyst + dice_tumor + dice_kidney)/3
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