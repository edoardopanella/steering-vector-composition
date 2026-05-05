#!/bin/bash
# Smoke test — mirrors Edoardo's slurm_extraction.sh template (256G mem, 8 cpus,
# generic gpu, named conda env), just with shorter walltime and a tiny workload.
#SBATCH --job-name=llama-smoke
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --qos=stud
#SBATCH --account=3247897
#SBATCH --chdir=/home/3247897/steering-vector-composition
#SBATCH --partition=stud

set -euo pipefail

# HF cache lives on BeeGFS (teammate template comments this as the right move
# "if the cluster has limited disk space in $HOME").
export SCRATCH=/mnt/beegfsstudents/home/$USER
export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TORCH_HOME=$SCRATCH/torch_cache
# Weights already pulled; skip any online HF calls at runtime.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

module purge
module load miniconda3
source activate steering-vector-composition-venv

mkdir -p logs

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "PWD:  $(pwd)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

export PYTHONPATH=/home/3247897/steering-vector-composition
python -u local_tests/smoke_cluster.py