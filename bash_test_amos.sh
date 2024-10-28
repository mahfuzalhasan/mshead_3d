#!/bin/bash
#SBATCH --job-name=python_wf_2211_amos_test
#SBATCH --output=/blue/r.forghani/mdmahfuzalhasan/ablation_studies/ablation_amos/wavelet-two-branch/mshead_3d/results/wf_2211_amos_test_job.%J.out
#SBATCH --error=/blue/r.forghani/mdmahfuzalhasan/ablation_studies/ablation_amos/wavelet-two-branch/mshead_3d/results/wf_2211_amos_test_job.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:1
#SBATCH --time=1:00:00

module load conda
conda activate waveformer

cd /blue/r.forghani/mdmahfuzalhasan/ablation_studies/ablation_amos/wavelet-two-branch/mshead_3d

# Execute the Python script
srun python test_seg_amos.py --cache_rate 1.0 --num_workers $SLURM_CPUS_PER_TASK --trained_weights $TRAINED_WEIGHTS