#!/bin/bash
#SBATCH --job-name=anthropic-repl-traj-pilot-l17
#SBATCH --output=/home/3242106/logs/anthropic_repl_trajectory_pilot_l17_%j.out
#SBATCH --error=/home/3242106/logs/anthropic_repl_trajectory_pilot_l17_%j.err
#SBATCH --time=04:00:00         # 3 pairs × 3 settings × (30 generations + 30 teacher-forced trajectory passes) at L=17 unit α=4; no judge calls. Cushion for first-pass model load + worst-case batch latency.
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

module purge
module load miniconda3
source activate steering-vector-composition-venv

set -a; source .env; set +a

mkdir -p /home/3242106/logs

echo "Starting trajectory-pilot-l17 (RQ2 Phase 1 pilot) — $(date)"
export PYTHONPATH=/home/3242106/steering-vector-composition-cloned
python -u scripts/anthropic_repl/run_trajectory_pilot_l17.py
echo "trajectory-pilot-l17 done — $(date)"
