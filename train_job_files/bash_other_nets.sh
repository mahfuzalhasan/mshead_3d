#!/bin/bash
#SBATCH --job-name=kits_swin_unetr
#SBATCH --output=/blue/r.forghani/mdmahfuzalhasan/project_analysis_kits19/mshead_3d/results/kits_swin_unetr_job.%J.out
#SBATCH --error=/blue/r.forghani/mdmahfuzalhasan/project_analysis_kits19/mshead_3d/results/kits_swin_unetr_job.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00

module load conda
conda activate waveformer

cd /blue/r.forghani/mdmahfuzalhasan/project_analysis_kits19/mshead_3d

# Execute the Python script
srun python main_train.py --cache_rate 1.0 --num_workers $SLURM_CPUS_PER_TASK --fold $FOLD --dataset $DATASET --network $NETWORK
# srun python test_seg_other_networks.py --cache_rate 1.0 --num_workers $SLURM_CPUS_PER_TASK --network $NETWORK