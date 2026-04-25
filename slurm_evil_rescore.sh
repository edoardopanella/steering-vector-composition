#!/bin/bash
#SBATCH --job-name=evil-rescore
#SBATCH --output=/home/3242106/logs/evil_rescore_%j.out
#SBATCH --error=/home/3242106/logs/evil_rescore_%j.err
#SBATCH --time=01:00:00          # 1 layer x 1 behavior; GPU gen + judge API calls
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

echo "Starting evil rescore — $(date)"
export PYTHONPATH=/home/3242106/steering-vector-composition
python -u scripts/run_evil_rescore.py
echo "Evil rescore done — $(date)"
