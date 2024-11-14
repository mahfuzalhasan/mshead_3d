import torch
import torch.nn.functional as F
import glob
import os

dataset = '/blue/r.forghani/data/tc_dataset'


train_img = []
train_label = []

for pat_id in os.listdir(dataset):
    data_file = [f for f in os.listdir(os.path.join(dataset, pat_id)) if ('.nrrd' in f) and ('Segmentation' not in f) and ('Image' not in f)][0]
    # print(f'data file:{data_file}')
    data_file_path = os.path.join(dataset, pat_id, data_file)
    # label_file = seg_file = [j for j in os.listdir(os.path.join(dataset, pat_id)) if ('Segmentation_modified.seg' in j) or ('Image.nrrd' in j)][0]
    label_file_path = os.path.join(dataset, pat_id, 'Segmentation_modified.nrrd')
    train_img.append(data_file_path)
    train_label.append(label_file_path)

    print(f'path:{data_file_path} label:{label_file_path}')

train_img = sorted(glob.glob(train_img))
train_label = sorted(glob.glob(train_label))

print(len(train_img), len(train_label))