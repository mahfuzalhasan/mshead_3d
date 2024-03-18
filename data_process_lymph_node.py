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
# new_data_dir = f"../DATA/lymph_node/ct_221_updated"
label_dict = {'patient_id':[], 'slice_num':[], 'label':[]}
patient_ids = os.listdir(data_dir)

# for pat_id in T(patient_ids):
# def data_processing(args):
    # pat_id, num_slices = args
for pat_id in T(patient_ids):
    pat_id = str(pat_id)
    print(f'patient id:{pat_id}')
    try:
        data_file = [f for f in os.listdir(os.path.join(data_dir, pat_id))if 'IM00' in f][0]
    except:
        print(f"Problem with {pat_id}")    
    seg_file = [f for f in os.listdir(os.path.join(data_dir, pat_id))if 'Segmentation' in f][0]
    seg2_file = seg_file.replace('Segmentation', 'Segmentation_v2')
    # img_pat_id, img_header = nrrd.read(os.path.join(data_dir, pat_id, data_file))
    mask_pat_id, mask_header = nrrd.read(os.path.join(data_dir, pat_id, seg_file))
    print(mask_pat_id.shape, np.min(mask_pat_id), np.max(mask_pat_id))
    mask_pat_id[mask_pat_id>0] = 1
    print(mask_pat_id.shape, np.min(mask_pat_id), np.max(mask_pat_id))
    nrrd.write(os.path.join(data_dir, pat_id, seg2_file), mask_pat_id, header=mask_header)
    

