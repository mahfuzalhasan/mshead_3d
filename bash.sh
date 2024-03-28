#!/bin/bash
#SBATCH --job-name=python_train_LN_3D
#SBATCH --output=/blue/r.forghani/scripts/mshead_3d/results/job.%J.out
#SBATCH --error=/blue/r.forghani/scripts/mshead_3d/results/job.%J.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mdmahfuzalhasan@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250GB
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:1
#SBATCH --time=48:00:00
#SBATCH --account=r.forghani
#SBATCH --qos=r.forghani

module load conda
conda activate medical

cd /blue/r.forghani/scripts/mshead_3d

# Execute the Python script
srun python main_train.py --dataset LN --network SNET --cache_rate 1.0 --batch_size 8 --crop_sample 4 --lr 0.00015 --num_workers $SLURM_CPUS_PER_TASK
# srun python test_seg.py --dataset LN --network SNET --cache_rate 1.0 --batch_size 3 --crop_sample 4 --lr 0.00015 --num_workers $SLURM_CPUS_PER_TASK