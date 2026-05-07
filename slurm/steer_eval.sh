#!/bin/bash
#SBATCH --job-name=anthropic-repl-steer-eval
#SBATCH --output=/home/3247897/logs/steer_eval_%j.out
#SBATCH --error=/home/3247897/logs/steer_eval_%j.err
#SBATCH --time=04:00:00         # 200 baseline + 200 steered gens + ~800 judge calls
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

# Trait passed as positional arg, defaults to evil for backward compat:
#   sbatch steer_eval.sh sycophantic
TRAIT="${1:-evil}"

echo "Starting anthropic-repl steer-eval trait=$TRAIT — $(date)"
export PYTHONPATH=/home/3247897/steering-vector-composition
python -u -m scripts.extraction.run_steer_eval --trait "$TRAIT"
echo "Anthropic-repl steer-eval trait=$TRAIT done — $(date)"
