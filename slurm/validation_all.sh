#!/bin/bash
#SBATCH --job-name=anthropic-repl-validation-all
#SBATCH --output=/home/3242106/logs/validation_all_%j.out
#SBATCH --error=/home/3242106/logs/validation_all_%j.err
#SBATCH --time=23:59:00         # ~11.5 min llm-judge × 14 traits + ~30s logprob × 7 ≈ 2:50h; 8h cushion for judge rate-limit retries
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --qos=stud
#SBATCH --account=3242106
#SBATCH --partition=stud
#SBATCH --chdir=/home/3242106/steering-vector-composition-cloned
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=edoardo.panella@studbocconi.it

set -euo pipefail

# HF cache: use default ~/.cache/huggingface (where Edoardo's prior runs already
# cached Llama-3.1-8B-Instruct). No offline flags — compute node has internet
# and will hit cache on first lookup, fetch any missing files online if needed.

module purge
module load miniconda3
source activate steering-vector-composition-venv

set -a; source .env; set +a

mkdir -p /home/3242106/logs

echo "Starting anthropic-repl validation-all — $(date)"
export PYTHONPATH=/home/3242106/steering-vector-composition-cloned
python -u scripts/validation/run_validation_all.py
echo "Anthropic-repl validation-all done — $(date)"
