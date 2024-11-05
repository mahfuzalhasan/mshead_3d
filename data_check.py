
import os
import shutil
import glob
from natsort import natsorted
import nibabel as nib
from scipy.ndimage import label
import numpy as np


# source_folder= "/blue/r.forghani/mdmahfuzalhasan/scripts/kits19/data"
# subjects = natsorted(os.listdir(source_folder))
# print(f'$$$$$$$$$$$$$$ total data:{len(subjects)} $$$$$$$$$$$$$$$$$$$')
# print(f'#############\n subject list: {subjects} \n ######################\n')

# train_img_destination = "/blue/r.forghani/share/kits2019/imagesTr"
# train_img_destination = "/blue/r.forghani/share/flare_data/imagesTr"
# train_img_destination = "/blue/r.forghani/share/amoss22/amos22/imagesTr"
train_label_destination = "/blue/r.forghani/share/kits2019/labelsTr"

# test_img_destination = "/blue/r.forghani/share/kits2019/imagesTs"
# test_label_destination = "/blue/r.forghani/share/kits2019/labelsTs"

# for i, image in enumerate(os.listdir(train_img_destination)):
#     image_path = os.path.join(train_img_destination, image)
#     vol = nib.load(image_path)
#     print(f'case:{image} volume: {vol.shape}')

for i, image in enumerate(os.listdir(train_label_destination)):
    label_path = os.path.join(train_label_destination, image)
    seg_img = nib.load(label_path)
    print(f'\n ########## case {i+1}:{image} volume: {seg_img.shape} #########')
    # Load the segmentation label file
    seg_data = seg_img.get_fdata()

    # Extract tumor regions (labeled as 2)
    tumor_regions = (seg_data == 2)
    # kidney_regions = (seg_data == 1)

    # Label connected components
    labeled_tumors, num_tumors = label(tumor_regions)
    print(f"Number of distinct tumors: {num_tumors}")

    # labeled_kidney, num_kidney = label(kidney_regions)
    # print(f"Number of distinct kidney: {num_kidney}")


    voxel_volume = np.prod(seg_img.header.get_zooms())  # Volume of each voxel in mm³
    print(f'voxel volume: {voxel_volume}')

    # Calculate volume for each tumor
    tumor_volumes = []
    for i in range(1, num_tumors + 1):
        # Count voxels in the ith tumor region
        tumor_voxel_count = np.sum(labeled_tumors == i)
        # Convert to physical volume
        tumor_volume = tumor_voxel_count * voxel_volume
        tumor_volumes.append(tumor_volume)
        print(f"Tumor {i} volume: {tumor_volume} mm³")
    

# exit()

