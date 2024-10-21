#!/bin/bash
#SBATCH --job-name=test_flare_wf_2211_fold_${FOLD}
#SBATCH --output=/blue/r.forghani/mdmahfuzalhasan/ablation_studies/ablation_flare/wavelet-two-branch/mshead_3d/results/fold_${FOLD}_wf_2211_job.%J.out
#SBATCH --error=/blue/r.forghani/mdmahfuzalhasan/ablation_studies/ablation_flare/wavelet-two-branch/mshead_3d/results/fold_${FOLD}_wf_2211_job.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:1
#SBATCH --time=1:00:00

module load conda
conda activate waveformer

cd /blue/r.forghani/mdmahfuzalhasan/ablation_studies/ablation_flare/wavelet-two-branch/mshead_3d

# Execute the Python script
# srun python main_train.py --cache_rate 1.0 --num_workers $SLURM_CPUS_PER_TASK --fold $FOLD
srun python multi_scale_evaluation.py --cache_rate 1.0 --num_workers $SLURM_CPUS_PER_TASK --fold $FOLD