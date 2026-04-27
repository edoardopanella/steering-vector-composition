#!/bin/bash
#SBATCH --job-name=llama-diag
#SBATCH --output=logs/diag_%j.out
#SBATCH --error=logs/diag_%j.err
#SBATCH --account=3247897
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:4g.40gb:1

set -euo pipefail

export SCRATCH=/mnt/beegfsstudents/home/$USER
export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TORCH_HOME=$SCRATCH/torch_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

module load miniconda3 cuda/12.4
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $SCRATCH/envs/steervec

cd ~/steering-vector-composition
mkdir -p logs

echo "Host: $(hostname)  |  Date: $(date)"
python -m local_tests.diag_memory