#!/bin/bash
#SBATCH --job-name=python_train_flare
#SBATCH --output=/blue/r.forghani/scripts/mshead_3d/results/job.%J.out
#SBATCH --error=/blue/r.forghani/scripts/mshead_3d/results/job.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00

module load conda
conda activate waveformer

cd /blue/r.forghani/scripts/mshead_3d

# Execute the Python script
srun python main_train.py --cache_rate 1.0 --num_workers $SLURM_CPUS_PER_TASK --fold 0
# srun python test_seg.py --cache_rate 1.0 --num_workers $SLURM_CPUS_PER_TASK