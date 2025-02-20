import os
import shutil
import argparse
from natsort import natsorted

# Default paths
DEFAULT_SOURCE_FOLDER = "/blue/r.forghani/mdmahfuzalhasan/scripts/kits23/dataset"
DEFAULT_TRAIN_IMG_DEST = "/blue/r.forghani/share/kits23/imagesTr"
DEFAULT_TRAIN_LABEL_DEST = "/blue/r.forghani/share/kits23/labelsTr"

# Argument parser to accept command-line inputs
parser = argparse.ArgumentParser(description="Copy training images and labels from source to destination folders.")
parser.add_argument("source_folder", nargs="?", default=DEFAULT_SOURCE_FOLDER, type=str,
                    help=f"Path to the source dataset folder (default: {DEFAULT_SOURCE_FOLDER})")
parser.add_argument("train_img_destination", nargs="?", default=DEFAULT_TRAIN_IMG_DEST, type=str,
                    help=f"Path to the training images destination folder (default: {DEFAULT_TRAIN_IMG_DEST})")
parser.add_argument("train_label_destination", nargs="?", default=DEFAULT_TRAIN_LABEL_DEST, type=str,
                    help=f"Path to the training labels destination folder (default: {DEFAULT_TRAIN_LABEL_DEST})")

args = parser.parse_args()

# Assign command-line arguments to variables
source_folder = args.source_folder
train_img_destination = args.train_img_destination
train_label_destination = args.train_label_destination

# Get subject folders and sort them
subjects = natsorted(os.listdir(source_folder))
print(f'$$$$$$$$$$$$$$ total data: {len(subjects)} $$$$$$$$$$$$$$$$$$$')
print(f'#############\n Subject list: {subjects} \n ######################\n')

# Create destination folders if they don't exist
os.makedirs(train_img_destination, exist_ok=True)
os.makedirs(train_label_destination, exist_ok=True)

# Identifier for naming files
identifier = "train"

# Process each subject
for i, subj in enumerate(subjects):
    case_id = subj.split('_')[1]
    print(f'##### Case ID: {case_id} #####\n')

    if i % 100 == 0:
        print(f"Finished processing {i}th data ")

    data_path = os.path.join(source_folder, subj)

    for file_name in os.listdir(data_path):
        source_file_path = os.path.join(data_path, file_name)

        # Skip if it's not a file
        if not os.path.isfile(source_file_path):
            continue

        new_file_name = f"{identifier}_{case_id}_{file_name}"

        # Copy to the correct destination
        if "imaging" in new_file_name:
            destination = os.path.join(train_img_destination, new_file_name)
        elif "segmentation" in new_file_name:
            destination = os.path.join(train_label_destination, new_file_name)
        else:
            continue  # Skip files that don't match expected patterns

        shutil.copy(source_file_path, destination)

print("File transfer complete!")
