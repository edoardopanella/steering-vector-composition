#!/bin/bash
#SBATCH --job-name=baseline-scoring
#SBATCH --output=/home/3242106/logs/baseline_scoring_%j.out
#SBATCH --error=/home/3242106/logs/baseline_scoring_%j.err
#SBATCH --time=08:00:00          # 7 behaviors x 1000 generations + 7000 judge calls
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --partition=stud
#SBATCH --account=3242106
#SBATCH --chdir=/home/3242106/steering-vector-composition-cloned

module purge
module load miniconda3
source activate steering-vector-composition-venv

set -a; source .env; set +a

echo "Starting baseline scoring — $(date)"
export PYTHONPATH=/home/3242106/steering-vector-composition-cloned
python -u scripts/run_baseline_scoring.py
echo "Baseline scoring done — $(date)"
