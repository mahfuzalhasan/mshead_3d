#!/bin/bash
#SBATCH --job-name=organ_kits23_uxnet
#SBATCH --output=/blue/r.forghani/mdmahfuzalhasan/scripts/WaveFormer_Improved_Up_kits/mshead_3d/results/organ_kits23_uxnet.%J.out
#SBATCH --error=/blue/r.forghani/mdmahfuzalhasan/scripts/WaveFormer_Improved_Up_kits/mshead_3d/results/organ_kits23_uxnet.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:1
#SBATCH --time=2:00:00

module load conda
conda activate waveformer

cd /blue/r.forghani/mdmahfuzalhasan/scripts/WaveFormer_Improved_Up_kits/mshead_3d

# Execute the Python script
srun python multi_organ_score_kits.py --fold $FOLD --dataset $DATASET --network $NETWORK