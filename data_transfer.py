
import os
import shutil
import glob
from natsort import natsorted


source_folder= "/blue/r.forghani/mdmahfuzalhasan/scripts/kits23/dataset"
subjects = natsorted(os.listdir(source_folder))
print(f'$$$$$$$$$$$$$$ total data:{len(subjects)} $$$$$$$$$$$$$$$$$$$')
print(f'#############\n subject list: {subjects} \n ######################\n')

train_img_destination = "/blue/r.forghani/share/kits23/imagesTr"
train_label_destination = "/blue/r.forghani/share/kits23/labelsTr"

if not os.path.exists(train_img_destination):
    os.makedirs(train_img_destination)
if not os.path.exists(train_label_destination):
    os.makedirs(train_label_destination)

identifier = "train"
for i,subj in enumerate(subjects):    
    case_id = subj.split('_')[1]
    print(f'##### case id:{case_id} #####\n')
    if i%100 == 0:
        print(f" finished {i}th data ")
    
    data_path = os.path.join(source_folder, subj)
    for files in os.listdir(data_path):

        source_file_path = os.path.join(data_path, files)    
        new_file_name = identifier+'_'+case_id+'_'+files
        
        if "imaging" in new_file_name:
            destination = os.path.join(train_img_destination, new_file_name)
        elif "segmentation" in new_file_name:
            destination = os.path.join(train_label_destination, new_file_name)

        shutil.copy(source_file_path, destination)
        


