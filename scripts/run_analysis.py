"""
Week 4 entrypoint: Gram matrix computation, pair selection, and regime classification.

Reads the steering vectors saved by run_extraction.py and produces:
  - results/gram_matrix_inst.pt   — [12, 12] cosine similarity matrix
  - results/selected_pairs.json  — 40 stratified pairs for run_composition.py
  - results/gram_matrix_base.pt  — base-model Gram matrix (RQ2)

Run this before run_composition.py to generate selected_pairs.json.
"""

import json
from pathlib import Path

import torch

from src.geometry import compute_gram_matrix, select_pairs

# --- config ---
VECTORS_DIR = Path("results/vectors/")
OUT_DIR = Path("results/")
LAYER = 20  # L* selected in run_extraction.py
BEHAVIORS = [
    "corrigibility", "power_seeking", "myopia", "verbosity",
    "formality", "politeness", "confidence", "agreeableness",
    "survival_instinct",
]
N_NEAR = 14
N_MODERATE = 13
N_HIGH = 13
# --------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load instruct-model vectors
vectors = [
    torch.load(VECTORS_DIR / f"{b}_layer{LAYER}.pt", weights_only=True)
    for b in BEHAVIORS
]

gram = compute_gram_matrix(vectors)
torch.save(gram, OUT_DIR / "gram_matrix_inst.pt")
print(f"Gram matrix saved: {gram.shape}")

pairs_idx = select_pairs(gram, n_near=N_NEAR, n_moderate=N_MODERATE, n_high=N_HIGH)
pairs_named = [[BEHAVIORS[i], BEHAVIORS[j]] for i, j in pairs_idx]
with open(OUT_DIR / "selected_pairs.json", "w") as f:
    json.dump(pairs_named, f, indent=2)
print(f"Selected {len(pairs_named)} pairs -> results/selected_pairs.json")

# TODO: repeat for base-model vectors once available:
#   base_vectors = [torch.load(...) for b in BEHAVIORS]
#   gram_base = compute_gram_matrix(base_vectors)
#   torch.save(gram_base, OUT_DIR / "gram_matrix_base.pt")

# TODO: add logistic regression (|G_ij| -> additive/non-additive), AUC, Spearman rho
# TODO: add KS test and paired t-test comparing gram_inst vs gram_base distributions
# TODO: add category cluster structure analysis (within vs between cosine, permutation test)

print("\nDone.")
