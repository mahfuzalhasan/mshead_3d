#!/bin/bash
#SBATCH --job-name=test_kits_wf_1111
#SBATCH --output=/blue/r.forghani/mdmahfuzalhasan/project_analysis_kits19/mshead_3d/results/kits_test_1111_job.%J.out
#SBATCH --error=/blue/r.forghani/mdmahfuzalhasan/project_analysis_kits19/mshead_3d/results/kits_test_1111_job.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:1
#SBATCH --time=1:00:00

module load conda
conda activate waveformer

cd /blue/r.forghani/mdmahfuzalhasan/project_analysis_kits19/mshead_3d

# Execute the Python script
srun python organ_wise_evaluation.py --cache_rate 1.0 --num_workers $SLURM_CPUS_PER_TASK --fold $FOLD --dataset $DATASET