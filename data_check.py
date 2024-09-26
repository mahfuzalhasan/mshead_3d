
import os
import shutil
import glob
from natsort import natsorted
import nibabel as nib


# source_folder= "/blue/r.forghani/mdmahfuzalhasan/scripts/kits19/data"
# subjects = natsorted(os.listdir(source_folder))
# print(f'$$$$$$$$$$$$$$ total data:{len(subjects)} $$$$$$$$$$$$$$$$$$$')
# print(f'#############\n subject list: {subjects} \n ######################\n')

train_img_destination = "/blue/r.forghani/share/kits2019/imagesTr"
train_label_destination = "/blue/r.forghani/share/kits2019/labelsTr"

test_img_destination = "/blue/r.forghani/share/kits2019/imagesTs"
test_label_destination = "/blue/r.forghani/share/kits2019/labelsTs"

for i, image in enumerate(os.listdir(train_img_destination)):
    image_path = os.path.join(train_img_destination, image)
    vol = nib.load(image_path)
    print(f'case:{image} volume: {vol.shape}')
# exit()

