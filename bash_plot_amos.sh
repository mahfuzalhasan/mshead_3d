#!/bin/bash
#SBATCH --job-name=python_ablation_wavelet-wo-split_plot_amos
#SBATCH --output=/blue/r.forghani/mdmahfuzalhasan/ablation_studies/ablation_amos/wavelet_without_split/mshead_3d/results/ablation_wavelet-wo-split_amos_plot_job.%J.out
#SBATCH --error=/blue/r.forghani/mdmahfuzalhasan/ablation_studies/ablation_amos/wavelet_without_split/mshead_3d/results/ablation_wavelet-wo-split_amos_plot_job.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:1
#SBATCH --time=36:00:00

module load conda
conda activate waveformer

cd /blue/r.forghani/mdmahfuzalhasan/ablation_studies/ablation_amos/wavelet_without_split/mshead_3d

# Execute the Python script
srun python plotting_prediction_amos.py --cache_rate 1.0 --num_workers $SLURM_CPUS_PER_TASK --pretrained_weights $PRETRAINED_WEIGHTS