import re
import numpy as np
import math
# Extract dice scores and count subjects
dice_scores = []

# Regular expression pattern to extract dice scores and subject IDs
dice_pattern = re.compile(r"\[(.*?)\]")
subject_pattern = re.compile(r'BraTS-\w+-\d+')
file_path = 'test_brats_residual_up_idwt_dec.59391381.txt'
# Read the file and extract relevant data
with open(file_path, "r") as file:
    lines = file.readlines()
count = 0
scores_tc = []
scores_wt = []
scores_et = []
subject_dict = {}

for i in range(len(lines)):
    line = lines[i].strip()
    
    # Check if the line contains dice scores
    if line.startswith("[") and line.endswith("]"):
        # Convert string of numbers into a list of floats
        scores = list(map(float, line.strip("[]").split(", ")))
        # print(i, " ",scores)
        dice_scores.append(scores)
        scores_tc.append(scores[0])
        scores_wt.append(scores[1])
        scores_et.append(scores[2])
        count += 1
    if line.startswith('./prediction_results'):
        portion = line.split(" ")
        subject_path = portion[0]
        subject_path = subject_path.replace(".nii.gz", "")
        match = re.search(r'BraTS-\w+-\d+-\d+', subject_path)
        if match:
            subject = match.group()
            print(subject)
        subject_dict[subject] = scores
print(f'count:{count}')

for sub, dices in subject_dict.items():
    print(sub, dices)


# small_cases = "Load small cases for TC"
# dice_scores = []        # ideally #num_small_cases list[[TC,WT,ET]]
# for case in small_cases:
#     dice_scores.append(subject_dict[case])

# sum_tc_dice= 0.0
# count = 0
# for elem in dice_scores:
#     class_tc_dice = elem[0]
#     if not math.isnan(class_tc_dice):
#         sum_tc_dice += class_tc_dice
#         count += 1
#     else:
#         sum_tc_dice += 1.0

# avg_small_tc = sum_tc_dice/count



# dice_scores = np.array(dice_scores)
# average_scores = np.nanmean(dice_scores, axis=0)
# print(average_scores)
scores_tc = [x for x in scores_tc if not math.isnan(x)]
scores_wt = [x for x in scores_wt if not math.isnan(x)]
scores_et = [x for x in scores_et if not math.isnan(x)]

avg_tc = np.sum(scores_tc)/count
avg_wt = np.sum(scores_wt)/count
avg_et = np.sum(scores_et)/count

print(f'valid tc: {len(scores_tc)} wt:{len(scores_wt)} et:{len(scores_et)}')
print(avg_tc, avg_wt, avg_et)
