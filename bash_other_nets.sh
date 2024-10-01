#!/bin/bash
#SBATCH --job-name=python_kits_other_networks
#SBATCH --output=/blue/r.forghani/mdmahfuzalhasan/project_analysis_kits19/mshead_3d/results/wavelet_max_ds_kits_other_nets_job.%J.out
#SBATCH --error=/blue/r.forghani/mdmahfuzalhasan/project_analysis_kits19/mshead_3d/results/wavelet_max_ds_kits_other_nets_job.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100GB
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:1
#SBATCH --time=20:00:00

module load conda
conda activate waveformer

cd /blue/r.forghani/mdmahfuzalhasan/project_analysis_kits19/mshead_3d

# Execute the Python script
srun python main_train_other_networks.py --cache_rate 1.0 --num_workers $SLURM_CPUS_PER_TASK --network $NETWORK
# srun python test_seg_other_networks.py --cache_rate 1.0 --num_workers $SLURM_CPUS_PER_TASK --network $NETWORK