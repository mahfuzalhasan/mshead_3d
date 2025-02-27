#!/bin/bash
#SBATCH --job-name=organ_kits23_UNETR
#SBATCH --output=/blue/r.forghani/mdmahfuzalhasan/scripts/WaveFormer_Improved_Up_kits/mshead_3d/results/organ_kits23_UNETR.%J.out
#SBATCH --error=/blue/r.forghani/mdmahfuzalhasan/scripts/WaveFormer_Improved_Up_kits/mshead_3d/results/organ_kits23_UNETR.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB

module load conda
conda activate waveformer

cd /blue/r.forghani/mdmahfuzalhasan/scripts/WaveFormer_Improved_Up_kits/mshead_3d

# Execute the Python script
srun python multi_organ_score_kits.py --fold $FOLD --dataset $DATASET --network $NETWORK