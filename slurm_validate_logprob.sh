#!/bin/bash
#SBATCH --job-name=validate-logprob
#SBATCH --output=/home/3242106/logs/validate_logprob_%j.out
#SBATCH --error=/home/3242106/logs/validate_logprob_%j.err
#SBATCH --time=00:30:00          # 5 pairs x 2 conditions x 2 forward passes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --partition=stud
#SBATCH --account=3242106
#SBATCH --chdir=/home/3242106/steering-vector-composition-cloned

module purge
module load miniconda3
source activate steering-vector-composition-venv

set -a; source .env; set +a

echo "Starting logprob validation — $(date)"
export PYTHONPATH=/home/3242106/steering-vector-composition-cloned
python -u -m scripts.validate_logprob
echo "Logprob validation done — $(date)"
