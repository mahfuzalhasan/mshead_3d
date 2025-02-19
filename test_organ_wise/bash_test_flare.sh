#!/bin/bash
#SBATCH --job-name=organ_flare_simple_ref_sep_hf_2trup
#SBATCH --output=/blue/r.forghani/mdmahfuzalhasan/scripts/WaveFormer_1111/mshead_3d/results/organ_flare_simple_ref_sep_hf_2trup.%J.out
#SBATCH --error=/blue/r.forghani/mdmahfuzalhasan/scripts/WaveFormer_1111/mshead_3d/results/organ_flare_simple_ref_sep_hf_2trup.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:1
#SBATCH --time=1:00:00

module load conda
conda activate waveformer

cd /blue/r.forghani/mdmahfuzalhasan/scripts/WaveFormer_1111/mshead_3d

# Execute the Python script
srun python multi_organ_score_flare.py --fold $FOLD --dataset $DATASET --network $NETWORK