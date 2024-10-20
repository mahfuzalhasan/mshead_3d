#!/bin/bash
#SBATCH --job-name=python_test_flare_ablation_wf3221_ms_test
#SBATCH --output=/blue/r.forghani/mdmahfuzalhasan/ablation_studies/ablation_flare/wavelet-wo-split-v2/mshead_3d/results/wf3221_ms_test_job.%J.out
#SBATCH --error=/blue/r.forghani/mdmahfuzalhasan/ablation_studies/ablation_flare/wavelet-wo-split-v2/mshead_3d/results/wf3221_ms_test_job.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:1
#SBATCH --time=1:00:00

module load conda
conda activate waveformer

cd /blue/r.forghani/mdmahfuzalhasan/ablation_studies/ablation_flare/wavelet-wo-split-v2/mshead_3d

# Execute the Python script
# srun python main_train.py --cache_rate 1.0 --num_workers $SLURM_CPUS_PER_TASK --fold $FOLD
srun python multi_scale_evaluation.py --cache_rate 1.0 --num_workers $SLURM_CPUS_PER_TASK --fold $FOLD