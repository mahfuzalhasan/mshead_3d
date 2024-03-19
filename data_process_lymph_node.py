from multiprocessing import Pool
from sklearn.model_selection import GroupKFold, StratifiedKFold
# from config.config import *
# from utils import *
import os
from tqdm import tqdm as T
import numpy as np
import cv2
import nrrd
import pandas as pd
# from p_tqdm import p_map
num_slices = 2
data_dir = "/blue/r.forghani/data/lymph_node/ct_221"
# new_data_dir = "/blue/r.forghani/data/lymph_node/ct_221_transposed"
label_dict = {'patient_id':[], 'slice_num':[], 'label':[]}
patient_ids = os.listdir(data_dir)

# if not os.path.exists(new_data_dir):
#     os.makedirs(new_data_dir)

# for pat_id in T(patient_ids):
# def data_processing(args):
    # pat_id, num_slices = args
contain_keys = ['type', 'dimension', 'space', 'sizes', 'space directions', 'kinds', 'encoding', 'space origin', 'Segmentation_ContainedRepresentationNames', 'Segmentation_ConversionParameters', 'Segmentation_MasterRepresentation', 'Segmentation_ReferenceImageExtentOffset']

for pat_id in T(patient_ids):
    pat_id = str(pat_id)
    print(f'############## patient id ###############:{pat_id}')
    try:
        data_file = [f for f in os.listdir(os.path.join(data_dir, pat_id))if 'IM00' in f][0]
    except:
        print(f"Problem with {pat_id}")    
    seg_file = [f for f in os.listdir(os.path.join(data_dir, pat_id))if 'Segmentation_v2' in f][0]
    # seg2_file = seg_file.replace('Segmentation', 'Segmentation_v2')

    img_pat_id, img_header = nrrd.read(os.path.join(data_dir, pat_id, data_file))
    mask_pat_id, mask_header = nrrd.read(os.path.join(data_dir, pat_id, seg_file))
    print(type(mask_header))
    mask_header_2 = {k:v for k,v in mask_header.items() if k in contain_keys}
    print(mask_header_2.keys(), len(mask_header_2.keys()))
    continue
    # print('mask info: ',mask_pat_id.shape, np.min(mask_pat_id), np.max(mask_pat_id))
    mask_pat_id[mask_pat_id>0] = 1
    mask_switched = np.transpose(mask_pat_id, (2, 0, 1))        # H,W,D --> D,H,W
    # print('updated mask value: ',mask_switched.shape, np.min(mask_pat_id), np.max(mask_pat_id))
    
    new_patient_path = os.path.join(new_data_dir, pat_id)
    os.makedirs(new_patient_path, exist_ok=True)
    # saving transposed data and mask in new directory
    nrrd.write(os.path.join(new_patient_path, data_file), data_switched, header=img_header)
    nrrd.write(os.path.join(new_patient_path, seg_file), mask_switched, header=mask_header)
    

