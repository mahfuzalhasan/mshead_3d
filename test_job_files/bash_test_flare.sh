#!/bin/bash
#SBATCH --job-name=test_flare_hf_ref_softplus_2_trconv
#SBATCH --output=/blue/r.forghani/mdmahfuzalhasan/scripts/WaveFormer_1111/mshead_3d/results/test_flare_hf_ref_softplus_2_trconv.%J.out
#SBATCH --error=/blue/r.forghani/mdmahfuzalhasan/scripts/WaveFormer_1111/mshead_3d/results/test_flare_hf_ref_softplus_2_trconv.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00

module load conda
conda activate waveformer

cd /blue/r.forghani/mdmahfuzalhasan/scripts/WaveFormer_1111/mshead_3d

# Execute the Python script
srun python test_seg.py --cache_rate 1.0 --num_workers $SLURM_CPUS_PER_TASK --fold $FOLD --dataset $DATASET --network $NETWORK