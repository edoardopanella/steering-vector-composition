#!/bin/bash
#SBATCH --job-name=layer-selection
#SBATCH --output=/home/3242106/logs/layer_selection_%j.out
#SBATCH --error=/home/3242106/logs/layer_selection_%j.err
#SBATCH --time=23:59:00          # generation (GPU) + ~6 h of judge API calls
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --partition=stud
#SBATCH --account=3242106
#SBATCH --chdir=/home/3242106/steering-vector-composition

module purge
module load miniconda3
source activate steering-vector-composition-venv

set -a; source .env; set +a

echo "Starting layer selection — $(date)"
export PYTHONPATH=/home/3242106/steering-vector-composition
python -u scripts/run_layer_selection.py
echo "Layer selection done — $(date)"
