#!/bin/bash
#SBATCH --job-name=python_kits_wf_2211_ablation
#SBATCH --output=/blue/r.forghani/mdmahfuzalhasan/ablation_studies/ablation_kits/wavelet-two-branch/mshead_3d/results/kits_wf_2211_job.%J.out
#SBATCH --error=/blue/r.forghani/mdmahfuzalhasan/ablation_studies/ablation_kits/wavelet-two-branch/mshead_3d/results/kits_wf_2211_job.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:1
#SBATCH --time=20:00:00

module load conda
conda activate waveformer

cd /blue/r.forghani/mdmahfuzalhasan/ablation_studies/ablation_kits/wavelet-two-branch/mshead_3d

# Execute the Python script
srun python main_train.py --cache_rate 1.0 --num_workers $SLURM_CPUS_PER_TASK --fold $FOLD
# srun python test_seg.py --cache_rate 1.0 --num_workers $SLURM_CPUS_PER_TASK --fold $FOLD