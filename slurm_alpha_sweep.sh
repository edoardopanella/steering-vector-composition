#!/bin/bash
#SBATCH --job-name=alpha-sweep
#SBATCH --output=/home/3242106/logs/alpha_sweep_%j.out
#SBATCH --error=/home/3242106/logs/alpha_sweep_%j.err
#SBATCH --time=01:00:00          # 90 generations + 90 judge calls, ~15-30 min
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

echo "Starting alpha sweep — $(date)"
export PYTHONPATH=/home/3242106/steering-vector-composition-cloned
python -u scripts/run_alpha_sweep.py
echo "Alpha sweep done — $(date)"
