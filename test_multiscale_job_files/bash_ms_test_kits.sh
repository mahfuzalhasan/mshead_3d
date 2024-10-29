#!/bin/bash
#SBATCH --job-name=multiscale_kits_wf_2211
#SBATCH --output=/blue/r.forghani/mdmahfuzalhasan/ablation_studies/ablation_kits/wavelet-two-branch/mshead_3d/results/wf_2211_kits_ms_test_job.%J.out
#SBATCH --error=/blue/r.forghani/mdmahfuzalhasan/ablation_studies/ablation_kits/wavelet-two-branch/mshead_3d/results/wf_2211_kits_ms_test_job.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:1
#SBATCH --time=1:00:00

module load conda
conda activate waveformer

cd /blue/r.forghani/mdmahfuzalhasan/ablation_studies/ablation_kits/wavelet-two-branch/mshead_3d

# Execute the Python script
# srun python main_train.py --cache_rate 1.0 --num_workers $SLURM_CPUS_PER_TASK --fold $FOLD
# srun python multi_scale_evaluation.py --cache_rate 1.0 --num_workers $SLURM_CPUS_PER_TASK --fold $FOLD
srun python multi_scale_evaluation.py --cache_rate 1.0 --num_workers $SLURM_CPUS_PER_TASK --fold $FOLD --dataset $DATASET
# srun python multi_scale_evaluation.py --cache_rate 1.0 --num_workers $SLURM_CPUS_PER_TASK --trained_weights $TRAINED_WEIGHTS --dataset $DATASET