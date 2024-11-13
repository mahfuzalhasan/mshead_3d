import os
import numpy as np
import nrrd
import SimpleITK as sitk
from multiprocessing import Pool
from tqdm import tqdm

# Path in server
data_dir = '/blue/r.forghani/data/tc_dataset'

patient_ids = os.listdir(data_dir)

def convert_save_segmentation_mask(pat_id):
    try:
        data_file = [f for f in os.listdir(os.path.join(data_dir, pat_id)) if ('.nrrd' in f) and ('Segmentation' not in f) and ('Image' not in f)][0]
    except:
        print(f'pat_id:{pat_id} data_file doesnt exist')
        return

    try:
        seg_file = [j for j in os.listdir(os.path.join(data_dir, pat_id)) if ('Segmentation.seg' in j) or ('Image.nrrd' in j)][0]
    except:
        print(f'pat_id:{pat_id} seg_file doesnt exist') 
        return

    print(f'data_file:{data_file}')
    print(f'seg_file:{seg_file}')

    image_path = os.path.join(data_dir, pat_id, data_file)
    segmentation_file_path = os.path.join(data_dir, pat_id, seg_file)
    
    mask_array, header = nrrd.read(segmentation_file_path)
    print(f'##### initial mask array shape: {mask_array.shape} ######')
    mask_array = np.transpose(mask_array, axes=[2, 1, 0])
    
    image = sitk.ReadImage(image_path, imageIO="NrrdImageIO")
    image_array = sitk.GetArrayFromImage(image)
    lbl_array = np.zeros_like(image_array, dtype=np.uint8)
    
    # Converting segmentation mask to original image size
    offset = header['Segmentation_ReferenceImageExtentOffset'].split()
    offset_width, offset_height, offset_depth = [int(value) for value in offset]
    mask_depth, mask_height, mask_width = mask_array.shape

    if offset_depth + mask_depth > lbl_array.shape[0]:
        diff = offset_depth + mask_depth - lbl_array.shape[0]
        print(f'pat id:{pat_id} mask_Depth:{mask_depth} offset_depth:{offset_depth} lbl_array:{lbl_array.shape} diff:{diff}')
    else:
        diff = 0
        
    depth_slice = slice(offset_depth, offset_depth + mask_depth - diff)
    height_slice = slice(offset_height, offset_height + mask_height)
    width_slice = slice(offset_width, offset_width + mask_width)
    lbl_array[depth_slice, height_slice, width_slice] = mask_array[:mask_depth-diff, :, :]

    # Apply the 90-degree rotation and flip
    rotated = np.rot90(lbl_array, k=-1, axes=(1, 2))
    flipped = np.flip(rotated, axis=2)

    # Save the modified mask as a new NRRD file
    new_segmentation_file_path = os.path.join(data_dir, pat_id, 'Segmentation_modified.nrrd')
    nrrd.write(new_segmentation_file_path, flipped, header)

    message = 'Mask saved as NRRD file'
    return message

# Block for only main images
if __name__ == '__main__':
    args_list = [(patient_id) for patient_id in patient_ids]
    with Pool(8) as p:
        for _ in tqdm(p.imap(convert_save_segmentation_mask, args_list), total=len(patient_ids), colour='red'):
            pass
