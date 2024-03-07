#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500GB
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:1
#SBATCH --time=48:00:00

module load conda
conda activate medical

cd /blue/r.forghani/scripts/mshead_3d

# Execute the Python script
srun python main_train.py --cache_rate 1.0 --num_workers $SLURM_CPUS_PER_TASK