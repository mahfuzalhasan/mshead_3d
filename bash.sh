#!/bin/bash
#SBATCH --job-name=python_train
#SBATCH --output=/blue/r.forghani/scripts/mshead_3d/results/python_train_fold2.out
#SBATCH --error=/blue/r.forghani/scripts/mshead_3d/results/python_train_fold2.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mdmahfuzalhasan@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=48:00:00

module purge
module load conda
conda activate medical

# Navigate to the directory containing the script
cd /blue/r.forghani/scripts/mshead_3d

# Execute the Python script
srun python main_train.py