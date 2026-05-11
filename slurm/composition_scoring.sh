#!/bin/bash
#SBATCH --job-name=composition-scoring-l17-generate
#SBATCH --output=/home/3242106/logs/composition_scoring_%j.out
#SBATCH --error=/home/3242106/logs/composition_scoring_%j.err
#SBATCH --time=23:59:00         # 36 pairs × (baseline + 2 singles + joint) gens + 36 × ~9600 teacher-force traces. Judging happens off-cluster (no internet on compute).
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

# Compute nodes have no outbound network. Force HF Hub offline so:
#  (1) from_pretrained skips HEAD revalidation and uses cached files,
#  (2) transformers' _patch_mistral_regex skips model_info() metadata call
#      (which has no cache fallback and would otherwise hard-fail).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Compute nodes also can't reach api.openai.com — run generate stage only
# (HF model + trajectory parquets). Judge stage is a follow-up on the login
# node:  COMPOSITION_MODE=judge python -m scripts.compositions.composition_scoring
export COMPOSITION_MODE=generate

echo "Starting composition-scoring-l17 generate stage + Phase 2 trajectory capture — $(date)"
export PYTHONPATH=/home/3242106/steering-vector-composition-cloned
python -u -m scripts.compositions.composition_scoring
echo "composition-scoring-l17 generate stage done — $(date)"
echo "Next step: on the login node, run"
echo "  cd /home/3242106/steering-vector-composition-cloned"
echo "  set -a; source .env; set +a"
echo "  COMPOSITION_MODE=judge python -m scripts.compositions.composition_scoring"
