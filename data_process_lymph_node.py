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
presence = []
absence = []
count = 0
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
    data_switched = np.transpose(img_pat_id, (2, 0, 1))         # H,W,D --> D,H,W
    # print(f'img pat id: {data_switched.shape}')
    # print('img header: ', img_header.keys())
    # if 'Segment18_Color' in img_header.keys():
    #     print("True in img_header")
        
    # else:
    #     print("False in img_header")

    
    mask_pat_id, mask_header = nrrd.read(os.path.join(data_dir, pat_id, seg_file))
    # print('mask header: ', mask_header.keys())
    if 'Segment18_Color' in mask_header.keys():
        # print("True in mask_header")
        presence.extend(mask_header.keys())
        presence = list(set(presence))
        count+=1
        print('mask header in presence \n : ', mask_header.keys())
        print(mask_header['Segment18_Color'])
    else:
        # print("False in mask_header")
        absence.extend(mask_header.keys())
        absence = list(set(absence))
        print('mask header in absence \n : ', mask_header.keys())
        count+=1
        # print(mask_header['Segment18_Color'])


    if count ==2:
        exit()

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
print('presence: ', presence)
print('absence: ', absence)

difference = set(presence) - set(absence)
print('difference: ', list(difference))
    

