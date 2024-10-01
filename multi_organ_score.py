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
pred_dir ="/orange/r.forghani/results/09-26-24_0432/output_seg"
# pred_dir ="/orange/r.forghani/results/UXNET/output_seg"


## Ground Truth Label
# gt_dir = os.path.join('/nfs/masi/leeh43/FLARE2021/TRAIN_MASK')
gt_dir = "/blue/r.forghani/share/flare_data/labelsTs"

print(f'pred:{pred_dir}')




spleen = []
kidney = []
liver = []
pancreas = []

subject_list = []
sub_list = []
all_subjects = []
count = 0


for label in os.listdir(pred_dir):
    subj = label
    label_pred = os.path.join(pred_dir, subj, subj + '_seg.nii.gz')


    label_gt = os.path.join(gt_dir, label.split('_0000')[0] + '.nii.gz')


    # label_gt = gt_file
    pred_nib = nib.load(label_pred)
    gt_nib = nib.load(label_gt)


    pred = pred_nib.get_fdata()
    gt = gt_nib.get_fdata()


    pred = np.transpose(pred, (2, 0, 1))
    gt = np.transpose(gt, (2, 0, 1))


    pred_mat = np.zeros((1, pred.shape[0], pred.shape[1], pred.shape[2]))
    gt_mat = np.zeros((1, pred.shape[0], pred.shape[1], pred.shape[2]))
    
    # Spleen
    idx_pred = np.where(pred == 3)
    pred_mat[0, idx_pred[0], idx_pred[1], idx_pred[2]] = 1
    idx_gt = np.where(gt == 3)
    gt_mat[0, idx_gt[0], idx_gt[1], idx_gt[2]] = 1
    dice_spleen = dice_score_organ(pred_mat, gt_mat)
    spleen.append(dice_spleen)
    subject_list.append(dice_spleen)


    # Kidney
    idx_pred = np.where(pred == 2)
    pred_mat[pred_mat != 0] = 0
    gt_mat[gt_mat != 0] = 0
    pred_mat[0, idx_pred[0], idx_pred[1], idx_pred[2]] = 1
    idx_gt = np.where(gt == 2)
    gt_mat[0, idx_gt[0], idx_gt[1], idx_gt[2]] = 1
    dice_kidney = dice_score_organ(pred_mat, gt_mat)
    kidney.append(dice_kidney)
    subject_list.append(dice_kidney)


    # Liver
    idx_pred = np.where(pred == 1)
    pred_mat[pred_mat != 0] = 0
    gt_mat[gt_mat != 0] = 0
    pred_mat[0, idx_pred[0], idx_pred[1], idx_pred[2]] = 1
    idx_gt = np.where(gt == 1)
    gt_mat[0, idx_gt[0], idx_gt[1], idx_gt[2]] = 1
    dice_liver = dice_score_organ(pred_mat, gt_mat)
    liver.append(dice_liver)
    subject_list.append(dice_liver)


    # Pancreas
    idx_pred = np.where(pred == 4)
    pred_mat[pred_mat != 0] = 0
    gt_mat[gt_mat != 0] = 0
    pred_mat[0, idx_pred[0], idx_pred[1], idx_pred[2]] = 1
    idx_gt = np.where(gt == 4)
    gt_mat[0, idx_gt[0], idx_gt[1], idx_gt[2]] = 1
    dice_pancreas = dice_score_organ(pred_mat, gt_mat)
    pancreas.append(dice_pancreas)
    subject_list.append(dice_pancreas)

    avg_dice = (dice_spleen + dice_kidney + dice_liver + dice_pancreas)/4
    count += 1
    print(f'\n ################ count:{count} --- --- Dataset: {label} ################')

    print('Spleen DICE: {}'.format(dice_spleen))
    print('Right Kidney DICE: {}'.format(dice_kidney))
    print('Liver DICE: {}'.format(dice_liver))
    print('Pancreas DICE: {}'.format(dice_pancreas))
    print('Avg DICE: {}'.format(avg_dice))

    print(f'########################################################################### \n')

    all_subjects.append([stat.mean(subject_list), label])
    sub_list.append('All Subjects')
    subject_list = []


all_organs = spleen + kidney + liver + pancreas

# all_organs = pancreas


print('Mean Spleen DICE: {}'.format(stat.mean(spleen)))
print('Stdev Spleen DICE: {}'.format(stat.stdev(spleen)))
print('Mean Right Kidney DICE: {}'.format(stat.mean(kidney)))
print('Stdev Right Kidney DICE: {}'.format(stat.stdev(kidney)))
print('Mean Liver DICE: {}'.format(stat.mean(liver)))
print('Stdev Liver DICE: {}'.format(stat.stdev(liver)))
print('Mean Pancreas DICE: {}'.format(stat.mean(pancreas)))
print('Stdev pancreas DICE: {}'.format(stat.stdev(pancreas)))
print('All Organ Mean DICE: {} /n'.format(stat.mean(all_organs)))
print('All Organ Stdev DICE: {} /n'.format(stat.stdev(all_organs)))