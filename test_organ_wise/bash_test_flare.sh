#!/bin/bash
#SBATCH --job-name=organ_flare_hf_simple_ref_residual_up_hf_agg
#SBATCH --output=/blue/r.forghani/mdmahfuzalhasan/scripts/WaveFormer_Improved_Up/mshead_3d/results/organ_flare_hf_simple_ref_residual_up_hf_agg.%J.out
#SBATCH --error=/blue/r.forghani/mdmahfuzalhasan/scripts/WaveFormer_Improved_Up/mshead_3d/results/organ_flare_hf_simple_ref_residual_up_hf_agg.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB

module load conda
conda activate waveformer

cd /blue/r.forghani/mdmahfuzalhasan/scripts/WaveFormer_Improved_Up/mshead_3d

# Execute the Python script
srun python multi_organ_score_flare.py --fold $FOLD --dataset $DATASET --network $NETWORK