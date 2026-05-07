#!/bin/bash
#SBATCH --job-name=anthropic-repl-steer-eval-all
#SBATCH --output=/home/3247897/logs/steer_eval_all_%j.out
#SBATCH --error=/home/3247897/logs/steer_eval_all_%j.err
#SBATCH --time=08:00:00         # ~11.5 min/trait × 14 remaining (evil done in E7.3) ≈ 2:45h; 8h cushion for judge rate-limit retries
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
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

echo "Starting anthropic-repl steer-eval-all — $(date)"
export PYTHONPATH=/home/3247897/steering-vector-composition
python -u -m scripts.extraction.run_steer_eval_all
echo "Anthropic-repl steer-eval-all done — $(date)"
