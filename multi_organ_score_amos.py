# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 13:18:55 2020


@author: peter
"""





## Model Prediction
# pred_dir = os.path.join('/nfs/masi/leeh43/repuxnet/out_FLARE_repuxnet_conv_matrix_alldata_sample_2')
pred_dir ="/orange/r.forghani/results/09-30-24_1350/output_seg"
# pred_dir ="/orange/r.forghani/results/UXNET/output_seg"


## Ground Truth Label
# gt_dir = os.path.join('/nfs/masi/leeh43/FLARE2021/TRAIN_MASK')
gt_dir = "/blue/r.forghani/share/flare_data/labelsTs"

print(f'pred:{pred_dir}')










import pandas as pd
import os
import nibabel as nib
import numpy as np
import statistics as stat

# Define a dictionary to map class indices to organ names
ORGAN_CLASSES = {
    1: "Spleen",
    2: "Right Kidney",
    3: "Left Kidney",
    4: "Gall Bladder",
    5: "Esophagus",
    6: "Liver",
    7: "Stomach",
    8: "Aorta",
    9: "Inferior Vena Cava",
    10: "Pancreas",
    11: "Right Adrenal Gland",
    12: "Left Adrenal Gland",
    13: "Duodenum",
    14: "Bladder",
    15: "Prostate"
}

# Function to calculate DICE score for a given organ
def dice_score_organ(im1, im2):
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError('Shape mismatch: im1 and im2 must have the same shape')

    intersection = np.logical_and(im1, im2)
    return (2. * intersection.sum() + 1e-7) / (im1.sum() + im2.sum() + 1e-7)

# Model Prediction and Ground Truth Directories
pred_dir = "/orange/r.forghani/results/09-30-24_2351/output_seg"
gt_dir = "/blue/r.forghani/share/amoss22/amos22/labelsTs"

print(f'Prediction Directory: {pred_dir}')
print(f'Ground Truth Directory: {gt_dir}')

# Initialize dictionaries to store DICE scores for each class
dice_scores = {organ: [] for organ in ORGAN_CLASSES.values()}
count = 0

# Iterate over each predicted label file in the directory
for label in os.listdir(pred_dir):
    subj = label
    label_pred = os.path.join(pred_dir, subj, subj + '_seg.nii.gz')
    label_gt = os.path.join(gt_dir, subj, subj + '.nii.gz')

    # Load the prediction and ground truth volumes
    pred_nib = nib.load(label_pred)
    gt_nib = nib.load(label_gt)
    pred = pred_nib.get_fdata()
    gt = gt_nib.get_fdata()

    # Transpose volumes to match the expected orientation
    pred = np.transpose(pred, (2, 0, 1))
    gt = np.transpose(gt, (2, 0, 1))

    # Initialize matrices for prediction and ground truth
    pred_mat = np.zeros((1, pred.shape[0], pred.shape[1], pred.shape[2]))
    gt_mat = np.zeros((1, pred.shape[0], pred.shape[1], pred.shape[2]))

    print(f'\n ################ Count: {count+1} --- Dataset: {label} ################')

    # Calculate DICE scores for each class separately
    for class_idx, organ_name in ORGAN_CLASSES.items():
        # Extract the specific organ region from both prediction and ground truth
        idx_pred = np.where(pred == class_idx)
        pred_mat[pred_mat != 0] = 0
        gt_mat[gt_mat != 0] = 0
        pred_mat[0, idx_pred[0], idx_pred[1], idx_pred[2]] = 1

        idx_gt = np.where(gt == class_idx)
        gt_mat[0, idx_gt[0], idx_gt[1], idx_gt[2]] = 1

        # Skip calculation if organ is not present in both GT and prediction
        if gt_mat.sum() == 0:
            print(f'{organ_name} is not present in the ground truth for this case.')
            continue

        # Calculate DICE score for the current organ
        dice_score = dice_score_organ(pred_mat, gt_mat)
        dice_scores[organ_name].append(dice_score)

        # Print the DICE score for the current organ
        print(f'{organ_name} DICE: {dice_score:.4f}')

    count += 1
    print(f'########################################################################### \n')

# Calculate and print summary statistics for each organ
for organ, scores in dice_scores.items():
    if scores:  # Check if the organ has been evaluated
        print(f'\nMean {organ} DICE: {stat.mean(scores):.4f}')
        print(f'Stdev {organ} DICE: {stat.stdev(scores) if len(scores) > 1 else 0:.4f}')

# Calculate overall statistics
all_organs = [score for scores in dice_scores.values() for score in scores]

print(f'\nAll Organ Mean DICE: {stat.mean(all_organs):.4f}')
print(f'All Organ Stdev DICE: {stat.stdev(all_organs):.4f}')