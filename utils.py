from monai.transforms import AsDiscrete
import torch


def filtering_output(output, filtered_label):
    
    post_pred = AsDiscrete(argmax=True)
    arr = post_pred(output[0])
    arr = arr.unsqueeze(0)
    
    dummy = torch.zeros_like(arr, dtype=torch.uint8)
    dummy[arr == filtered_label] = 1

    return dummy



def scale_wise_organ_filtration(arr, ORGAN_CLASSES, prediction = False):
    # test_labels_tensor = test_labels[0, 0, :, :, :]
    SMALL = 1
    MEDIUM = 2
    LARGE = 3

    if prediction:
        post_pred = AsDiscrete(argmax=True)
        arr = post_pred(arr[0])
        arr = arr.unsqueeze(0)
        print(f'test output conversion: {arr.shape}')
    
    
    unique_labels = torch.unique(arr)
    print(f'unique labels: {unique_labels}')

    # size_labels = torch.zeros_like(arr, dtype=torch.uint8)
    size_labels = {SMALL:set(), MEDIUM:set(), LARGE:set()}
    ORGAN_SCALE ={SMALL:0, MEDIUM:0, LARGE:0}
    
    # Filtering of Small, Medium and Large
    for label in unique_labels:
        if label == 0:
            continue
        dummy = torch.zeros_like(arr, dtype=torch.uint8)
        dummy[arr == label] = 1
        N_voxel = torch.count_nonzero(dummy)
        volume = N_voxel.item() * 1.5 * 1.5 * 2    # in mm^3
        volume = volume / 1000                     # in cm^3
        print(f'Class: {ORGAN_CLASSES[label.item()]} volume: {volume}')
        if volume < 1000:
            size_labels[SMALL].add(label)
            # size_labels[arr==label] = SMALL         # 1
            ORGAN_SCALE[SMALL] += 1
        elif volume >= 1000 and volume < 3000:
            size_labels[MEDIUM].add(label)
            # size_labels[arr==label] = MEDIUM        # 2
            ORGAN_SCALE[MEDIUM] += 1
        elif volume >= 3000:
            size_labels[LARGE].add(label)
            # size_labels[arr==label] = LARGE         # 3
            ORGAN_SCALE[LARGE] += 1

    return size_labels, ORGAN_SCALE