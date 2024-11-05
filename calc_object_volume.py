DATASET = 'kits'    # 'amos' or 'flare' or 'kits'
FOLD = 1



from monai.utils import set_determinism
from monai.transforms import AsDiscrete
# from networks.UXNet_3D.network_backbone import UXNET
# from networks.msHead_3D.network_backbone import MSHEAD_ATTN
# from monai.networks.nets import UNETR, SwinUNETR
# from networks.nnFormer.nnFormer_seg import nnFormer
# from networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch, ThreadDataLoader
import nibabel as nib

from monai.transforms import (
    AsDiscreted,
    AddChanneld,
    Compose,
    CropForegroundd,
    SpatialPadd,
    ResizeWithPadOrCropd,
    LoadImaged,
    Orientationd,
    Transposed,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    KeepLargestConnectedComponentd,
    Spacingd,
    ToTensord,
    RandAffined,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    RandRotate90d,
    EnsureTyped,
    Invertd,
    KeepLargestConnectedComponentd,
    SaveImaged,
    Activationsd
)

import torch
from torch.utils.tensorboard import SummaryWriter
from load_datasets_transforms import data_loader, data_transforms
import matplotlib.pyplot as plt

import os
import numpy as np
from tqdm import tqdm
import datetime
import argparse
import time






class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

# Create the config object from the dictionary
config_dict = {
    'root': '/blue/r.forghani/share/kits2019',
    'output': '/orange/r.forghani/results',
    'dataset': DATASET,
    'network': 'MSHEAD',
    'mode': 'test',
    'pretrain': False,
    'pretrained_weights': '/orange/r.forghani/results/09-11-24_1805/model_best.pth',
    'batch_size': 1,
    'crop_sample': 2,
    'lr': 0.0001,
    'optim': 'AdamW',
    'max_iter': 40000,
    'eval_step': 500,
    'resume': False,
    'finetune': True,
    'gpu': 0,
    'cache_rate': 1.0,
    'num_workers': 8,
    'start_index': 160,
    'end_index': 180,
    'no_split': False,
    'plot': False
}

if DATASET == 'kits':
    config_dict['fold'] = FOLD

# Instantiate the Config class
config = Config(config_dict)

# Access the parameters as needed
print("Root Directory:", config.root)
print("Network:", config.network)
print("Batch Size:", config.batch_size)

# # Example usage within a function or class
# def train_model(config):
#     print(f"Training {config['network']} on {config['dataset']} dataset with batch size {config['batch_size']}...")
#     # Add your training logic here

# # Run the function with the configuration
# train_model(config)



if config.dataset == 'amos':
    ORGAN_CLASSES = {1: "Spleen", 2: "Right Kidney", 3: "Left Kidney", 4: "Gall Bladder", 5: "Esophagus",6: "Liver",
        7: "Stomach", 8: "Aorta", 9: "Inferior Vena Cava", 10: "Pancreas", 11: "Right Adrenal Gland", 
        12: "Left Adrenal Gland", 13: "Duodenum", 14: "Bladder", 15: "Prostate"}
    voxel_volume = 1.5 * 1.5 * 2.0  # mm^3
elif config.dataset == 'flare':
    ORGAN_CLASSES = {1: "Liver", 2: "Kidney", 3: "Spleen", 4: "Pancreas"}
    voxel_volume = 1.0 * 1.0 * 1.2  # mm^3
elif config.dataset == 'kits':
    ORGAN_CLASSES = {1: "Kidney", 2: "Tumor"}
    # spacing = (1.2, 1.0, 1.0)
    # voxel_volume = 1.2 * 1.0 * 1.0  # mm^3
else:
    raise ValueError("Invalid dataset name")


test_samples, out_classes = data_loader(config)




test_files = [
    {"image": image_name, "label": label_name, "path": path}
    for image_name, label_name, path in zip(test_samples['images'], test_samples['labels'], test_samples['paths'])
]
print(f'test files: {len(test_files)}')
print(f' \n ****************** test File List :\n {test_files} \n ******************* \n')

# Set determinism for reproducibility
set_determinism(seed=0)

# Apply data transforms using the config object
# test_transforms = data_transforms(config)
test_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        # Spacingd(keys=["image", "label"], pixdim=(1.2, 1.0, 1.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Transposed(keys=["image", "label"], indices=(0, 3, 1, 2)),
        ScaleIntensityRanged(
            keys=["image"], a_min=-200, a_max=300,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        # CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ]
)
print('Start caching datasets!')

# Initialize the cache dataset and data loader
test_ds = CacheDataset(
    data=test_files, 
    transform=test_transforms, 
    cache_rate=config.cache_rate, 
    num_workers=config.num_workers
)
test_loader = ThreadDataLoader(test_ds, batch_size=1, num_workers=0)

# Set the device for PyTorch operations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'--- device: {device} ---')

import pandas as pd
import numpy as np

# Initialize a DataFrame with class labels as columns
class_labels = [ORGAN_CLASSES[label] for label in sorted(ORGAN_CLASSES.keys())]
df = pd.DataFrame(columns=class_labels)

# Iterate over the test loader and fill the DataFrame with volumes
for step, batch in enumerate(test_loader):
    # Move tensors to device and convert to numpy
    test_inputs, test_labels = batch["image"].to(device), batch["label"].to(device)
    print(f'---------------------{step}---------------------')
    print(f'input: {test_inputs.shape} labels: {test_labels.shape}')
    print('label path:', batch["path"])

    ############ Reading spacing    ####################
    # Load the NIfTI image
    nifti_img = nib.load(batch["path"])

    # Extract the header
    header = nifti_img.header

    # Read voxel spacing (pixdim values)
    spacing = header.get_zooms()  # This returns a tuple with the spacing for each dimension

    # Display the spacing
    print("Voxel Spacing (mm):", spacing)
    voxel_volume = spacing[0] * spacing[1] * spacing[2]

    ################################
    
    # Extract label data and find unique labels
    test_labels = test_labels.cpu().numpy()[0, 0, :, :, :]
    unique_labels = np.unique(test_labels)
    print(f'unique labels: {unique_labels}')
    
    # Initialize a dictionary to store volumes for the current sample
    volume_dict = {label_name: np.nan for label_name in class_labels}

    # Calculate volumes for each unique label (excluding background)
    for label in unique_labels:
        if label == 0:  # Skip background
            continue
        
        N_voxel = np.sum(unique_labels == label)
        if label == 1:      # if kidney
            N_voxel += np.sum(unique_labels == 2)   # add tumor voxel too

        # dummy = np.zeros(shape=test_labels.shape, dtype='uint8')
        # dummy[test_labels == label] = 1
        # N_voxel = np.count_nonzero(dummy)

        volume = N_voxel * voxel_volume  # in mm^3
        volume_cm3 = volume / 1000  # Convert to cm^3

        # Store the volume in the dictionary
        volume_dict[ORGAN_CLASSES[label]] = volume
        print(f'Class: {ORGAN_CLASSES[label]} volume: {volume} volume_cm3:{volume_cm3}')

    # Append the volume dictionary as a new row in the DataFrame
    df = pd.concat([df, pd.DataFrame([volume_dict])], ignore_index=True)





# Calculate and print min and max for each class
print("\n Min and Max Volumes for Each Class:")
for label in class_labels:
    min_volume = df[label].min(skipna=True)  # Skip NaN values
    max_volume = df[label].max(skipna=True)  # Skip NaN values
    print(f"{label}: Min = {min_volume:.2f} cm^3, Max = {max_volume:.2f} cm^3")
    
# Plotting volumes with different colors for each class
plt.figure(figsize=(10, 6))


for label in class_labels:
    plt.scatter(df[label], [label] * len(df), label=label)

plt.xlabel("Volume (cm³)", fontsize=12)
plt.ylabel("Organ Classes", fontsize=12)
plt.title("Volumes of Different Organ Classes", fontsize=14)
plt.legend(title="Organ Classes")
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Show the plot
plt.show()
plt.savefig('volumes_{DATASET}.png')