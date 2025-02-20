from sklearn.model_selection import KFold
import torch
from torch import nn
from torch.cuda.amp import autocast
from batchgenerators.utilities.file_and_folder_operations import *

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
    LambdaD,
    NormalizeIntensityd,
    KeepLargestConnectedComponentd,
    Spacingd,
    ToTensord,
    RandAffined,
    RandFlipd,
    RandShiftIntensityd,
    RandRotate90d,
    EnsureTyped,
    Invertd,
    KeepLargestConnectedComponentd,
    SaveImaged,
    Activationsd,
    Lambda
)

import numpy as np
from collections import OrderedDict
import glob
import os
def data_loader(args):
    root_dir = args.root
    dataset = args.dataset

    print('Start to load data from directory: {}'.format(root_dir), flush=True)

    if dataset == 'feta':
        out_classes = 8
    elif dataset == 'flare':
        out_classes = 5
    elif dataset == 'amos':
        out_classes = 16
    elif dataset == 'kits':
        out_classes = 3
    elif dataset == 'kits23':
        out_classes = 4
    

    if args.mode == 'train':
        train_samples = {}
        valid_samples = {}
        print(f'#### loading training and validation set ########## \n')

        
        ## Input training data
        if dataset == 'kits23':
            # train_img = sorted(glob.glob(os.path.join(root_dir,'*/imaging.nii.gz')))
            # train_label = sorted(glob.glob(os.path.join(root_dir,'*/segmentation.nii.gz')))

            # HiperGator Settings
            # train_img = sorted(glob.glob(os.path.join(root_dir, 'imagesTr', '*.nii.gz')))
            # train_label = sorted(glob.glob(os.path.join(root_dir, 'labelsTr', '*.nii.gz')))
            

            #### Nautilus Settings--> Try to simplify it. Life will be easier
            # Find all case directories (e.g. case_00000, case_00001, ...)
            case_dirs = sorted(glob.glob(os.path.join(args.root, "case_*")))
            # print(f'case_dirs: {case_dirs}')
            train_img = []
            train_label = []

            for cdir in case_dirs:
                # In the new folder structure, the processed image is saved as "labels.nii.gz"
                # and the processed label is saved as "segmentation.nii.gz"
                # TODO : change preprocessing to name "image" and "labels"
                img = sorted(glob.glob(os.path.join(cdir, "imaging.nii.gz")))
                label = sorted(glob.glob(os.path.join(cdir, "segmentation.nii.gz")))
                
                if os.path.exists(img[0]) and os.path.exists(label[0]):
                    # print('path exists')
                    train_img.append(img[0])
                    train_label.append(label[0])
                    # print('path appended')
                else:
                    print(f"Warning: Missing files in {cdir}... skipping")
            print(f"train_img: {train_img}, train_label: {train_label}")
        else:
            raise NotImplementedError(f'Preprocessed data for {dataset} is not available')


        if not args.no_split:
            validation_per_fold = 98
            start_index = validation_per_fold * args.fold
            end_index = validation_per_fold * args.fold + validation_per_fold

            if end_index > len(train_label):
                end_index = len(train_label)

            
            valid_img = train_img[start_index:end_index]
            valid_label = train_label[start_index:end_index]

            del train_img[start_index:end_index]
            del train_label[start_index:end_index]

        else:
            ## Input inference data
            # valid_img = sorted(glob.glob(os.path.join(root_dir, 'imagesTs', '*.nii.gz')))
            # valid_label = sorted(glob.glob(os.path.join(root_dir, 'labelsTs', '*.nii.gz')))
            raise NotImplementedError(f'no_split for Preprocessed {dataset} is not available')

        train_samples['images'] = train_img
        train_samples['labels'] = train_label
        valid_samples['images'] = valid_img
        valid_samples['labels'] = valid_label

        ######################################################################
        # checking split
        # print(f'#### train_img_list ###### \n ')
        # print(train_img)
        # print('\n #### train label list #### \n')
        # print(train_label)
        # print(f'----------- {len(train_img)}, {len(train_label)}-----------')
        # print(f'$$$$$$$$$ valid_img list $$$$$$$$$$$ \n ')
        # print(valid_img)
        # print('\n $$$$$$$$$ valid_label list $$$$$$$$$ \n')
        # print(valid_label)
        # print(f'----------- {len(valid_img)}, {len(valid_label)}-----------')
        ######################################################################
        # print(f'valid img:{valid_img} label:{valid_label}')
        print('Finished loading all training samples from dataset: {}!'.format(dataset), flush=True)
        print('Number of classes for segmentation: {}'.format(out_classes), flush=True)

        return train_samples, valid_samples, out_classes

    elif args.mode == 'test':
        test_samples = {}
        if args.dataset == 'kits23':
            ## Input training data
            ### Set up in Hypergator
            train_img = sorted(glob.glob(os.path.join(root_dir, 'imagesTr', '*.nii.gz')))
            train_label = sorted(glob.glob(os.path.join(root_dir, 'labelsTr', '*.nii.gz')))
            ##########

            ## Set up in Nautilus --> change it if you can. Always make
            ## primary stuff like data reading simple.
            # case_dirs = sorted(glob.glob(os.path.join(args.root, "case_*")))
            # # print(f'case_dirs: {case_dirs}')
            # train_img = []
            # train_label = []

            # for cdir in case_dirs:
            #     # In the new folder structure, the processed image is saved as "labels.nii.gz"
            #     # and the processed label is saved as "segmentation.nii.gz"
            #     # TODO : change preprocessing to name "image" and "labels"
            #     img = sorted(glob.glob(os.path.join(cdir, "imaging.nii.gz")))
            #     label = sorted(glob.glob(os.path.join(cdir, "segmentation.nii.gz")))
                
            #     if os.path.exists(img[0]) and os.path.exists(label[0]):
            #         # print('path exists')
            #         train_img.append(img[0])
            #         train_label.append(label[0])
            #         # print('path appended')
            #     else:
            #         print(f"Warning: Missing files in {cdir}... skipping")

            validation_per_fold = 98
            start_index = validation_per_fold * args.fold
            end_index = validation_per_fold * args.fold + validation_per_fold

            if end_index > len(train_label):
                end_index = len(train_label)

            
            test_img = train_img[start_index:end_index]
            test_label = train_label[start_index:end_index]  
        else:
            ## Input inference data
            test_img = sorted(glob.glob(os.path.join(root_dir, 'imagesTs', '*.nii.gz')))
            test_label = sorted(glob.glob(os.path.join(root_dir, 'labelsTs', '*.nii.gz')))

        test_samples['images'] = test_img
        test_samples['labels'] = test_label
        test_samples['paths'] = test_label 
        print(f"test_img: {test_img}, length: {len(test_img)}")
        print('Finished loading all inference samples from dataset: {}!'.format(dataset))

        return test_samples, out_classes


def data_transforms(args):
    dataset = args.dataset
    if args.mode == 'train':
        crop_samples = args.crop_sample
    else:
        crop_samples = None
    roi_size = tuple(map(int, args.roi_size.split(',')))
    
    if dataset == "kits23":
        train_transforms = Compose(
        [   #$$$$$$$$ Jawad apply from here for 1st stage preprocessing $$$$$$$
            ### wrote load for .pt
            # Lambda(keys=["image", "label"], func=lambda x: load_pt_file(x)),  # Load .pt file
            # Lambdad(keys=["image", "label"], func=load_pt),  # Loads .pt files
            LoadImaged(keys=["image", "label"]),            # D, H, W
            # AddChanneld(keys=["image", "label"]),
            
            ### -- Commented out transforms are already done in preparing the preprocessed data ---
            # Orientationd(keys=["image", "label"], axcodes="RAS"),           # H, W, D
            # Spacingd(keys=["image", "label"], pixdim=(1, 1, 1), mode=("bilinear", "nearest")),   
            # ScaleIntensityRanged(
            #     keys=["image"], a_min=-58, a_max=302,
            #     b_min=0, b_max=1, clip=True,
            # ),
            # CropForegroundd(keys=["image", "label"], source_key="image"),
            # $$$$$$$$$ First Stage Preprocessing Ends here

            # $$$$$ Mahdi apply the way you are doing right now for 2nd Stage Prepro. for train
            # Use crop_samples = 2
            ########## Rest applied during training---> image will be in H,W,D
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size, #for now (96, 96, 96), #(128, 128, 128),
                pos=1,
                neg=1,
                num_samples=crop_samples,
                image_key="image",
                image_threshold=0,
            ),
            
            # RandFlipd(
            #     keys=["image", "label"],
            #     spatial_axis=[0],
            #     prob=0.5,
            # ),
            RandFlipd(              # Mirroing horizontally (W)--> left-right swap
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.5,
            ),
            # RandFlipd(
            #     keys=["image", "label"],
            #     spatial_axis=[2],
            #     prob=0.5,
            # ),
    
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),

            RandAffined(
                keys=['image', 'label'],
                mode=('bilinear', 'nearest'),
                prob=1.0, spatial_size=roi_size, #(128, 128, 128),
                rotate_range=(np.pi/30, np.pi/30, np.pi/30),
                scale_range=(0.1, 0.1, 0.1)),
            ToTensord(keys=["image", "label"]),
            # $$$$$$$ during training prepro. ends here

        ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=roi_size, #(96, 96, 96), #(128, 128, 128),
                    pos=3,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                ToTensord(keys=["image", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(keys=["image", "label"], pixdim=(1, 1, 1), mode=("bilinear", "nearest")),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-58, a_max=302,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        # TODO : review test_transforms_plot before running test
        test_transforms_plot = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(1, 1, 1), mode=("bilinear")),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-58, a_max=302,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                ToTensord(keys=["image"]),
            ]
        )


    if args.mode == 'train':
        print('Cropping {} sub-volumes for training!'.format(str(crop_samples)))
        print('Performed Data Augmentations for all samples!')
        return train_transforms, val_transforms

    elif args.mode == 'test':
        print('Performed transformations for all samples!')
        if args.plot:
            return test_transforms_plot
        return test_transforms


def infer_post_transforms(args, test_transforms, out_classes, output_dir=None):
    if output_dir is None:
        output_dir = args.output

    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Activationsd(keys="pred", softmax=True),
        Invertd(
            keys="pred",  # invert the `pred` data field, also support multiple fields
            transform=test_transforms,
            orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
            # then invert `pred` based on this information. we can use same info
            # for multiple fields, also support different orig_keys for different fields
            meta_keys="pred_meta_dict",  # key field to save inverted meta data, every item maps to `keys`
            orig_meta_keys="image_meta_dict",  # get the meta data from `img_meta_dict` field when inverting,
            # for example, may need the `affine` to invert `Spacingd` transform,
            # multiple fields can use the same meta data to invert
            meta_key_postfix="meta_dict",  # if `meta_keys=None`, use "{keys}_{meta_key_postfix}" as the meta key,
            # if `orig_meta_keys=None`, use "{orig_keys}_{meta_key_postfix}",
            # otherwise, no need this arg during inverting
            nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
            # to ensure a smooth output, then execute `AsDiscreted` transform
            to_tensor=True,  # convert to PyTorch Tensor after inverting
        ),
        ## If monai version <= 0.6.0:
        AsDiscreted(keys="pred", argmax=True, n_classes=out_classes),
        ## If moani version > 0.6.0:
        # AsDiscreted(keys="pred", argmax=True)
        # KeepLargestConnectedComponentd(keys='pred', applied_labels=[1, 3]),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=output_dir,
                   output_postfix="seg", output_ext=".nii.gz", resample=True),
    ])

    return post_transforms
