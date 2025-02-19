#!/bin/bash
#SBATCH --job-name=organ_test_kits23
#SBATCH --output=/blue/r.forghani/mdmahfuzalhasan/scripts/WaveFormer_Kits/mshead_3d/results/organ_test_kits23.%J.out
#SBATCH --error=/blue/r.forghani/mdmahfuzalhasan/scripts/WaveFormer_Kits/mshead_3d/results/organ_test_kits23.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:1
#SBATCH --time=1:00:00

module load conda
conda activate waveformer

cd /blue/r.forghani/mdmahfuzalhasan/scripts/WaveFormer_Kits/mshead_3d

# Execute the Python script
srun python multi_organ_score_kits.py --fold $FOLD --dataset $DATASET --network $NETWORK