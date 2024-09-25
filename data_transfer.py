
import os
import shutil
import glob
from natsort import natsorted


source_folder= "/blue/r.forghani/mdmahfuzalhasan/scripts/kits19/data"
subjects = natsorted(os.listdir(source_folder))
print(f'$$$$$$$$$$$$$$ total data:{len(subjects)} $$$$$$$$$$$$$$$$$$$')
print(f'#############\n subject list: {subjects} \n ######################\n')

train_img_destination = "/blue/r.forghani/share/kits2019/imagesTr"
train_label_destination = "/blue/r.forghani/share/kits2019/labelsTr"

test_img_destination = "/blue/r.forghani/share/kits2019/imagesTs"
test_label_destination = "/blue/r.forghani/share/kits2019/labelsTs"
exit()

for i,subj in enumerate(subjects):
    if i==210:
        break
    
    if i<190:
        split = "train"
    else:
        split = "test"   
    
    case_id = subj.split('_')[1]
    print(f'##### case id:{case_id} #####\n')
    
    data_path = os.path.join(source_folder, subj)
    for files in os.listdir(data_path):

        source_file_path = os.path.join(data_path, files)    
        new_file_name = split+'_'+case_id+'_'+files
        
        if "imaging" in new_file_name:
            if split == "train":
                destination = os.path.join(train_img_destination, new_file_name)
            else:
                destination = os.path.join(test_img_destination, new_file_name)
        
        elif "segmentation" in new_file_name:
            if split == "train":
                destination = os.path.join(train_label_destination, new_file_name)
            else:
                destination = os.path.join(test_label_destination, new_file_name)
        shutil.copy(source_file_path, destination)
        


