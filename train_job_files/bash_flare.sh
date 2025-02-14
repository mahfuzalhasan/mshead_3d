#!/bin/bash
#SBATCH --job-name=train_flare_gat_fuse
#SBATCH --output=/blue/r.forghani/mdmahfuzalhasan/scripts/WaveFormer_Gated_Fusion/mshead_3d/results/train_flare_gat_fuse.%J.out
#SBATCH --error=/blue/r.forghani/mdmahfuzalhasan/scripts/WaveFormer_Gated_Fusion/mshead_3d/results/train_flare_gat_fuse.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00

module load conda
conda activate waveformer

cd /blue/r.forghani/mdmahfuzalhasan/scripts/WaveFormer_Gated_Fusion/mshead_3d

# Execute the Python script
srun python main_train.py --cache_rate 1.0 --num_workers $SLURM_CPUS_PER_TASK --fold $FOLD --dataset $DATASET --network $NETWORK