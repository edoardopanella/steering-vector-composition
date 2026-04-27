#!/bin/bash
#SBATCH --job-name=anthropic-repl-build
#SBATCH --output=/home/3247897/logs/anthropic_repl_build_%j.out
#SBATCH --error=/home/3247897/logs/anthropic_repl_build_%j.err
#SBATCH --time=02:00:00         # forward passes only, no generation, no judge calls
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --qos=stud
#SBATCH --account=3247897
#SBATCH --partition=stud
#SBATCH --chdir=/home/3247897/steering-vector-composition

set -euo pipefail

export SCRATCH=/mnt/beegfsstudents/home/$USER
export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TORCH_HOME=$SCRATCH/torch_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

module purge
module load miniconda3
source activate steering-vector-composition-venv

set -a; source .env; set +a

mkdir -p /home/3247897/logs

echo "Starting anthropic-repl build — $(date)"
export PYTHONPATH=/home/3247897/steering-vector-composition
python -u -m scripts.anthropic_repl.run_build_vector
echo "Anthropic-repl build done — $(date)"
