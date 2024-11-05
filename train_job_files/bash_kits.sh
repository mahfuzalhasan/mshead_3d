#!/bin/bash
#SBATCH --job-name=train_kits_wf_3211
#SBATCH --output=/blue/r.forghani/mdmahfuzalhasan/ablation_studies/ablation_kits/wf_3211/mshead_3d/results/kits_train_3211_job.%J.out
#SBATCH --error=/blue/r.forghani/mdmahfuzalhasan/ablation_studies/ablation_kits/wf_3211/mshead_3d/results/kits_train_3211_job.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00

module load conda
conda activate waveformer

cd /blue/r.forghani/mdmahfuzalhasan/ablation_studies/ablation_kits/wf_3211/mshead_3d

# Execute the Python script
srun python main_train.py --cache_rate 1.0 --num_workers $SLURM_CPUS_PER_TASK --fold $FOLD --dataset $DATASET