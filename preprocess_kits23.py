import os
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

# MONAI transforms
try:
    from monai.transforms import (
        Compose,
        LoadImaged,
        AddChanneld,
        Spacingd,
        Orientationd,
        Transposed,
        ScaleIntensityRanged,
        CropForegroundd,
        ToTensord
    )
except ImportError:
    raise ImportError("MONAI is not installed. Please install it via `pip install monai`.")


def load_kits23_data(root_dir):
    """
    Collect all imaging.nii.gz / segmentation.nii.gz pairs
    from KiTS23 case_* directories under `root_dir`.
    Returns a list of dicts -> [{"image": path, "label": path}, ...].
    """
    cases = sorted(glob.glob(os.path.join(root_dir, "case_*")))
    data_list = []
    for c in cases:
        img_path = os.path.join(c, "imaging.nii.gz")
        lbl_path = os.path.join(c, "segmentation.nii.gz")
        if os.path.exists(img_path) and os.path.exists(lbl_path):
            data_list.append({"image": img_path, "label": lbl_path})
        else:
            print(f"Warning: Missing imaging/label in {c}. Skipping.")
    if len(data_list) == 0:
        raise ValueError(f"No data found under {root_dir}. Check your path!")
    return data_list


def get_preprocessing_transforms(output_dir="preprocessed_data"):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        
        Orientationd(keys=["image", "label"], axcodes="RAS"),  # H, W, D -> RAS orientation
        Spacingd(keys=["image", "label"], pixdim=(1, 1, 1), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-58, a_max=302, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        
        # Transposed(keys=["image", "label"], indices=(0, 3, 1, 2)),  # reorder dimensions: (D, H, W)
        ToTensord(keys=["image", "label"]),

        # # Save the intermediate preprocessed images/labels
        # RenameMetaKey(keys=["image", "label"]),
        # SaveImaged(
        #     keys=["image", "label"],
        #     meta_keys=["image_meta_dict", "label_meta_dict"],
        #     output_dir=out_dir,
        #     output_postfix="",          # no extra suffix
        #     output_ext=".nii.gz",       # enforce .nii.gz
        #     resample=False,             # already applied Spacingd & Orientationd
        #     separate_folder=False,      # place all outputs in "preprocessed/" only
        #     print_log=True,             # optionally log each save
        # ),
    ])


def process_sample(item, transforms, out_dir):
    """
    Process a single sample:
      - Creates the output subfolder based on the case folder name.
      - Applies the transforms.
      - Converts the resulting tensors to NumPy arrays.
      - Saves the processed image as "labels.nii.gz" and the label as "segmentation.nii.gz"
        in the case subfolder using nibabel.
    """
    # Determine the case folder name from the input image path.
    case_folder = os.path.basename(os.path.dirname(item["image"]))
    case_out_dir = os.path.join(out_dir, case_folder)
    os.makedirs(case_out_dir, exist_ok=True)

    # Define output file paths.
    img_out_path = os.path.join(case_out_dir, "imaging.nii.gz")
    lbl_out_path = os.path.join(case_out_dir, "segmentation.nii.gz")

    # Skip processing if both files already exist.
    if os.path.exists(img_out_path) and os.path.exists(lbl_out_path):
        print(f"Skipping already processed case {case_folder}")
        return

    # Apply the transforms.
    xform_dict = transforms(item)
    img_tensor = xform_dict["image"]  # Expected shape: [1, D, H, W]
    lbl_tensor = xform_dict["label"]  # Expected shape: [1, D, H, W]

    # Convert torch tensors to NumPy arrays and squeeze out the channel dimension.
    img_np = img_tensor.numpy()
    lbl_np = lbl_tensor.numpy()

    # Save using nibabel (using an identity affine).
    nib.save(nib.Nifti1Image(img_np, affine=np.eye(4)), img_out_path)
    nib.save(nib.Nifti1Image(lbl_np, affine=np.eye(4)), lbl_out_path)

    print(f"Finished processing case '{case_folder}'")


def save_preprocessed_dataset_multithreaded(data_list, transforms, out_dir, num_workers=8):
    """
    Process and save the dataset using multithreading.
    For each case in data_list, a thread applies the transform pipeline and saves the
    processed image and label to the appropriate subfolder.
    """
    os.makedirs(out_dir, exist_ok=True)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_sample, item, transforms, out_dir)
            for item in data_list
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            try:
                future.result()
            except Exception as e:
                print(f"Error during processing: {e}")


import argparse

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Preprocess KiTS23 dataset with configurable directories.")
    parser.add_argument("--root_dir", type=str, default="/project/kits23/dataset/", 
                        help="Path to the root directory of the dataset (default: /project/kits23/dataset/).")
    parser.add_argument("--out_dir", type=str, default="/project/preprocessed/", 
                        help="Path to save the preprocessed dataset.")
    parser.add_argument("--num_workers", type=int, default=8, 
                        help="Number of worker threads for preprocessing (default: 8).")
    args = parser.parse_args()

    # Define directories from arguments
    root_dir = args.root_dir
    out_dir = args.out_dir

    # 1) Load the dataset
    data_list = load_kits23_data(root_dir)
    print(f"Found {len(data_list)} cases under {root_dir}.")

    # 2) Get the transform pipeline
    transforms = get_preprocessing_transforms()

    # 3) Process and save the dataset using multithreading
    save_preprocessed_dataset_multithreaded(data_list, transforms, out_dir, num_workers=args.num_workers)

    print(f"\n✅ Done! Preprocessed data saved in '{out_dir}'.")


if __name__ == "__main__":
    main()





# root_dir = "/project/kits23/dataset/"
# out_dir = "/project/preprocessed_spacing111/"