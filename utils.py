from monai.transforms import AsDiscrete
import torch
import numpy as np
from metric import dice


def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        d = dice(pred, gt)
        return np.array([d, 50])
    
    elif gt.sum() == 0 and pred.sum() == 0:
        return np.array([1.0, 50])
    
    else:
        return np.array([0.0, 50])

def dice_score_organ(im1, im2):
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)
    if im1.shape != im2.shape:
        raise ValueError('Shape mismatch: im1 and im2 must have the same shape')
    intersection = np.logical_and(im1 , im2)
    return (2. * intersection.sum() + 0.0000001) / (im1.sum() + im2.sum() + 0.0000001)

def filtering_output(output, filtered_label):
    
    post_pred = AsDiscrete(argmax=True)
    arr = post_pred(output[0])
    arr = arr.unsqueeze(0)
    
    dummy = torch.zeros_like(arr, dtype=torch.uint8)
    dummy[arr == filtered_label] = 1

    return dummy

## Arr-> GT/Output: 
# GT -> B, 1, D, H, W/ output-> B,Class,D,H,W
## regions--> values from label dictionary

def hierarchical_prediction(arr, label_values, prediction = False):
    if prediction:
        post_pred = AsDiscrete(argmax=True)     
        filtered = post_pred(arr[0])        # convert output: Class,D,H,W --> (1, D, H, W) = [0, num_class-1]
        arr = filtered.unsqueeze(0)         # B, 1, D, H, W; here B=1      
    
    arr_new = torch.zeros_like(arr, dtype=torch.uint8)
    for l in label_values:           #(1,2)
        arr_new[arr == l] = 1
    return arr_new




def get_brats_regions():
    """
    this is only valid for the brats data in here where the labels are 1, 2, and 3. The original brats data have a
    different labeling convention!
    :return:
    """
    regions = {
        "whole tumor": (1, 2, 3),
        "tumor core": (2, 3),
        "enhancing tumor": (3,)
    }
    return regions


def get_KiTS_regions():
    regions = {
        "kidney incl tumor": (1, 2),
        "tumor": (2,)
    }
    return regions


def scale_wise_organ_filtration(arr, ORGAN_CLASSES, spacing = (1.5, 1.5, 2), organ_size_range = [1000, 3000], prediction = False):
    # test_labels_tensor = test_labels[0, 0, :, :, :]
    SMALL = 1
    MEDIUM = 2
    LARGE = 3

    if prediction:
        post_pred = AsDiscrete(argmax=True)
        arr = post_pred(arr[0])
        arr = arr.unsqueeze(0)
        print(f'test output conversion: {arr.shape}')       # 1, 1, D, H, W
    
    
    unique_labels = torch.unique(arr)
    print(f'unique labels: {unique_labels}')

    # size_labels = torch.zeros_like(arr, dtype=torch.uint8)
    size_labels = {SMALL:set(), MEDIUM:set(), LARGE:set()}
    ORGAN_SCALE ={SMALL:0, MEDIUM:0, LARGE:0}
    
    # Filtering of Small, Medium and Large
    for label in unique_labels:
        if label == 0:
            continue
        dummy = torch.zeros_like(arr, dtype=torch.uint8)
        dummy[arr == label] = 1
        N_voxel = torch.count_nonzero(dummy)
        volume = N_voxel.item() * spacing[0] * spacing[1] * spacing[2]    # in mm^3
        volume = volume / 1000                     # in cm^3
        print(f'Class: {ORGAN_CLASSES[label.item()]} volume: {volume}')
        if volume <= organ_size_range[0]:
            size_labels[SMALL].add(label)
            # size_labels[arr==label] = SMALL         # 1
            ORGAN_SCALE[SMALL] += 1
        elif volume > organ_size_range[0] and volume <= organ_size_range[1]:
            size_labels[MEDIUM].add(label)
            # size_labels[arr==label] = MEDIUM        # 2
            ORGAN_SCALE[MEDIUM] += 1
        elif volume > organ_size_range[1]:
            size_labels[LARGE].add(label)
            # size_labels[arr==label] = LARGE         # 3
            ORGAN_SCALE[LARGE] += 1

    return size_labels, ORGAN_SCALE


def create_region_from_mask(mask, join_labels: tuple):
    # mask_new = np.zeros_like(mask, dtype=np.uint8)
    mask_new = torch.zeros_like(mask, dtype=torch.uint8)
    for l in join_labels:           #(1,2)
        mask_new[mask == l] = 1
    return mask_new


def evaluate_case(image_pred: str, image_gt: str):
    # image_gt = sitk.GetArrayFromImage(sitk.ReadImage(file_gt))
    # image_pred = sitk.GetArrayFromImage(sitk.ReadImage(file_pred))
    results = []
    regions = get_KiTS_regions()
    for r in regions.values():      # Kidney Incl Tumor
        mask_pred = create_region_from_mask(image_pred, r)
        mask_gt = create_region_from_mask(image_gt, r)
        # dc = np.nan if np.sum(mask_gt) == 0 and np.sum(mask_pred) == 0 else metric.dc(mask_pred, mask_gt)
        # results.append(dc)
    return results