from monai.transforms import LoadImaged, AddChanneld, Spacingd, Orientationd, ScaleIntensityRanged, CropForegroundd, ToTensord
import nibabel as nib  # Optional, for additional image inspection


# Define individual transforms
load_transform = LoadImaged(keys=["image", "label"])
add_channel_transform = AddChanneld(keys=["image", "label"])
spacing_transform = Spacingd(keys=["image", "label"], pixdim=(1.2, 1.0, 1.0), mode=("bilinear", "nearest"))
orientation_transform = Orientationd(keys=["image", "label"], axcodes="RAS")
scale_intensity_transform = ScaleIntensityRanged(
    keys=["image"],
    a_min=-200,
    a_max=300,
    b_min=0.0,
    b_max=1.0,
    clip=True,
)
crop_foreground_transform = CropForegroundd(keys=["image", "label"], source_key="image")
to_tensor_transform = ToTensord(keys=["image", "label"])

sample_data = {
    "image": "/blue/r.forghani/share/kits2019/imagesTr/test_00190_imaging.nii.gz",
    "label": "/blue/r.forghani/share/kits2019/labelsTr/test_00190_segmentation.nii.gz"
}

# Apply LoadImaged
loaded_data = load_transform(sample_data)

# Print the shape after LoadImaged
print("After LoadImaged:")
print(f"Image shape: {loaded_data['image'].shape}")
print(f"Label shape: {loaded_data['label'].shape}")

# Apply AddChanneld
loaded_data = add_channel_transform(loaded_data)
print("\nAfter AddChanneld:")
print(f"Image shape: {loaded_data['image'].shape}")
print(f"Label shape: {loaded_data['label'].shape}")

# Similarly, apply other transforms and print shapes as needed
# Apply AddChanneld
loaded_data = spacing_transform(loaded_data)
print("\nAfter spacing:")
print(f"Image shape: {loaded_data['image'].shape}")
print(f"Label shape: {loaded_data['label'].shape}")

# Similarly, apply other transforms and print shapes as needed


