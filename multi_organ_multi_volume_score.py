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
from scipy.ndimage import label





def dice_score_organ(im1, im2):
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    # im2 = np.asarray(im2).astype(np.bool)


    if im1.shape != im2.shape:
        raise ValueError('Shape mismatch: im1 and im2 must have the same shape')


    intersection = np.logical_and(im1 , im2)


    return (2. * intersection.sum() + 0.0000001) / (im1.sum() + im2.sum() + 0.0000001)




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='KiTS Evaluation segmentation')
    parser.add_argument('--output', type=str, default='/orange/r.forghani/results', required=False, help='Output folder for both tensorboard and the best model')
    parser.add_argument('--dataset', type=str, default='flare', required=False, help='Datasets: {feta, flare, amos}, Fyi: You can add your dataset here')
    parser.add_argument('--fold', type=int, default=0, help='current running fold')
    args = parser.parse_args()

    if args.dataset == 'flare':
        model_id_dict = {}
        gt_dir = "/blue/r.forghani/share/flare_data/labelsTs"
    elif args.dataset == 'amos':
        model_id_dict = {}
        gt_dir = "/blue/r.forghani/share/flare_data/labelsTs"
    elif args.dataset == 'kits':
        model_id_dict = {0: '11-06-24_0143', 1:'11-06-24_0221', 2:'11-06-24_0250', 3:'11-06-24_0252', 4:'11-06-24_0259'}
        gt_dir = '/blue/r.forghani/share/kits2019/labelsTr'
    else:
        raise NotImplementedError(f'No such dataset: {args.dataset}')


    ## Model Prediction
    model_id = model_id_dict[args.fold]
    pred_dir ='/orange/r.forghani/results/'+model_id+'/output_seg'
    print(f'pred:{pred_dir} ground truth:{gt_dir}')

    kidney = []
    tumor = []

    subject_list = []
    sub_list = []
    all_subjects = []
    count = 0

    
    # Define range for volume-based classes
    tumor_volumes_binned = {(1, 10):[], # VS
                            (10, 30):[],    # S
                            (30, 50):[],    # M
                            (50, 100):[],   # L
                            (100, 40000):[]}    # VL
    multi_scale_dice = {(1, 10):[], # VS
                            (10, 30):[],    # S
                            (30, 50):[],    # M
                            (50, 100):[],   # L
                            (100, 40000):[]}    # VL
    
    
    skip_images = ['00001', '00041', '00104', '00117', '00118', '00151', '00184', '00205']
    for label in natsorted(os.listdir(pred_dir)):
        subj = label
        label_pred = os.path.join(pred_dir, subj, subj + '_seg.nii.gz')
        
        # Exclude samples with multiple tumors for now
        if any(img in subj for img in skip_images):
            print(f'skipping  {subj} due to multiple tumors \n')
            continue
        
        

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
        tumor_regions = (gt == 2)   # Extract tumor regions (labeled as 2)
        labeled_tumors, num_tumors = label(tumor_regions)   # Label connected components

        voxel_volume = np.prod(gt_nib.header.get_zooms())  # Volume of each voxel in mm³
        

        tumor_volumes = []
        for i in range(1, num_tumors + 1):
            # Count voxels in the ith tumor region
            tumor_voxel_count = np.sum(labeled_tumors == i)
            # Convert to physical volume
            tumor_volume = tumor_voxel_count * voxel_volume
            tumor_volume_cm3 = tumor_volume / 1000
            if tumor_volume_cm3 < 1:
                gt[labeled_tumors == i] = 0
                continue
            for t_vol in tumor_volumes_binned:
                if tumor_volume_cm3 >= t_vol[0] and tumor_volume_cm3 < t_vol[1]:
                    (tumor_volumes_binned[t_vol]).append(subj)


        pred = np.transpose(pred, (2, 0, 1))
        gt = np.transpose(gt, (2, 0, 1))

        pred_mat = np.zeros((1, pred.shape[0], pred.shape[1], pred.shape[2]))
        gt_mat = np.zeros((1, pred.shape[0], pred.shape[1], pred.shape[2]))
        

        ### Kidney with Tumor
        idx_pred = np.where(pred != 0)
        idx_gt = np.where(gt != 0)
        
        pred_mat[pred_mat != 0] = 0
        gt_mat[gt_mat != 0] = 0
        
        pred_mat[0, idx_pred[0], idx_pred[1], idx_pred[2]] = 1
        gt_mat[0, idx_gt[0], idx_gt[1], idx_gt[2]] = 1
        dice_kidney = dice_score_organ(pred_mat, gt_mat)
        kidney.append(dice_kidney)
        subject_list.append(dice_kidney)


        ### Tumor
        idx_pred = np.where(pred == 2)
        idx_gt = np.where(gt == 2)

        pred_mat[pred_mat != 0] = 0
        gt_mat[gt_mat != 0] = 0
        pred_mat[0, idx_pred[0], idx_pred[1], idx_pred[2]] = 1
        gt_mat[0, idx_gt[0], idx_gt[1], idx_gt[2]] = 1
        
        dice_tumor = dice_score_organ(pred_mat, gt_mat)
        for t_vol in tumor_volumes_binned:
                if subj in tumor_volumes_binned[t_vol]:
                    multi_scale_dice[t_vol].append(dice_tumor)
        
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

    for vol_range in multi_scale_dice:
        print(f'\n ----- Volume Range: {vol_range} -----')
        print('Mean DICE: {}'.format(stat.mean(multi_scale_dice[vol_range])))
        print('Stdev DICE: {}'.format(stat.stdev(multi_scale_dice[vol_range])))

    print('Mean Kidney with Tumor DICE: {}'.format(stat.mean(kidney)))
    # print('Stdev Kidney with Tumor DICE: {}'.format(stat.stdev(kidney)))
    print('Mean Tumor DICE: {}'.format(stat.mean(tumor)))
    # print('Stdev Tumor DICE: {}'.format(stat.stdev(tumor)))

    print('All Organ Mean DICE: {} /n'.format(stat.mean(all_organs)))
    print('All Organ Stdev DICE: {} /n'.format(stat.stdev(all_organs)))