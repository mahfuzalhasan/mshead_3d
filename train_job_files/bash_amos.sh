#!/bin/bash
#SBATCH --job-name=train_amos_WF_2211
#SBATCH --output=/blue/r.forghani/mdmahfuzalhasan/ablation_studies/ablation_amos/wavelet-two-branch/mshead_3d/results/amos_train_WF_2211_job.%J.out
#SBATCH --error=/blue/r.forghani/mdmahfuzalhasan/ablation_studies/ablation_amos/wavelet-two-branch/mshead_3d/results/amos_train_WF_2211_job.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00

module load conda
conda activate waveformer

cd /blue/r.forghani/mdmahfuzalhasan/ablation_studies/ablation_amos/wavelet-two-branch/mshead_3d

# Execute the Python script
# srun python training_whole_dataset.py --cache_rate 1.0 --num_workers $SLURM_CPUS_PER_TASK --crop_sample $CROP_SAMPLE --pretrained_weights $PRETRAINED_WEIGHTS 
srun python main_train.py --cache_rate 1.0 --num_workers $SLURM_CPUS_PER_TASK --pretrained_weights $PRETRAINED_WEIGHTS --start_index $START --end_index $END --dataset $DATASET
# finetuning on whole trainset
# srun python main_train.py --cache_rate 1.0 --num_workers $SLURM_CPUS_PER_TASK --crop_sample $CROP_SAMPLE --pretrained_weights $PRETRAINED_WEIGHTS
# srun python test_seg.py --cache_rate 1.0 --num_workers $SLURM_CPUS_PER_TASK