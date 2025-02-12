#!/bin/bash
#SBATCH --job-name=organ_flare_mednext
#SBATCH --output=/blue/r.forghani/mdmahfuzalhasan/scripts/flare_mednext/mshead_3d/results/organ_flare_mednext.%J.out
#SBATCH --error=/blue/r.forghani/mdmahfuzalhasan/scripts/flare_mednext/mshead_3d/results/organ_flare_mednext.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00

module load conda
conda activate waveformer

cd /blue/r.forghani/mdmahfuzalhasan/scripts/flare_mednext/mshead_3d

# Execute the Python script
srun python multi_organ_score_flare.py --fold $FOLD --dataset $DATASET --network $NETWORK