#!/bin/bash
#SBATCH --job-name=python_ablation_wavelet-wo-split_v2_amos
#SBATCH --output=/blue/r.forghani/mdmahfuzalhasan/ablation_studies/ablation_amos/wave-wo-split-v2/mshead_3d/results/ablation_wave-wo-split_v2_amos_job.%J.out
#SBATCH --error=/blue/r.forghani/mdmahfuzalhasan/ablation_studies/ablation_amos/wave-wo-split-v2/mshead_3d/results/ablation_wave-wo-split_v2_amos_job.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:1
#SBATCH --time=36:00:00

module load conda
conda activate waveformer

cd /blue/r.forghani/mdmahfuzalhasan/ablation_studies/ablation_amos/wave-wo-split-v2/mshead_3d

# Execute the Python script
srun python test_seg_amos.py --cache_rate 1.0 --num_workers $SLURM_CPUS_PER_TASK --trained_weights $TRAINED_WEIGHTS