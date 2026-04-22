"""
Week 3 entrypoint: full composition sweep.

For each selected pair (b_i, b_j) and each coefficient setting in ALPHA_SETTINGS,
generate N_COMPLETIONS completions on each of EVAL_PROMPTS.
Results are written to OUT_DIR/<behavior_i>__<behavior_j>/alpha_<ai>_<aj>.jsonl
"""

import json
from pathlib import Path

import torch

from src.injection import apply_joint_steering
from src.model_utils import load_model

# --- config ---
MODEL = "gpt2-xl"
DEVICE = "cpu"
VECTORS_DIR = Path("results/vectors/")
PAIRS_FILE = Path("results/selected_pairs.json")   # list of [behavior_i, behavior_j]
LAYER = 20                                          # L* selected in run_extraction.py
OUT_DIR = Path("results/compositions/")
ALPHA_SETTINGS = [(0, 0), (1, 0), (0, 1), (1, 1), (-1, 1), (1, -1)]
N_COMPLETIONS = 25
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 100
# TODO: replace with the full 20-prompt neutral evaluation set (Proposal §4)
EVAL_PROMPTS = [
    "The meeting had just ended and everyone was",
    "She opened the letter and began to read",
]
# --------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading model: {MODEL}")
model = load_model(MODEL, device=DEVICE)

with open(PAIRS_FILE) as f:
    pairs = json.load(f)

for behavior_i, behavior_j in pairs:
    v_i = torch.load(VECTORS_DIR / f"{behavior_i}_layer{LAYER}.pt", weights_only=True)
    v_j = torch.load(VECTORS_DIR / f"{behavior_j}_layer{LAYER}.pt", weights_only=True)

    pair_dir = OUT_DIR / f"{behavior_i}__{behavior_j}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    for alpha_i, alpha_j in ALPHA_SETTINGS:
        out_file = pair_dir / f"alpha_{alpha_i}_{alpha_j}.jsonl"
        if out_file.exists():
            print(f"  Skipping {behavior_i}/{behavior_j} ({alpha_i},{alpha_j}), already done.")
            continue

        print(f"  {behavior_i} x {behavior_j}  alpha=({alpha_i},{alpha_j})")
        records = []
        for prompt in EVAL_PROMPTS:
            for _ in range(N_COMPLETIONS):
                text = apply_joint_steering(
                    model, prompt, LAYER,
                    [(v_i, alpha_i), (v_j, alpha_j)],
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                )
                records.append({
                    "prompt": prompt,
                    "behavior_i": behavior_i,
                    "behavior_j": behavior_j,
                    "alpha_i": alpha_i,
                    "alpha_j": alpha_j,
                    "completion": text,
                    # TODO: add judge scores (score_behavior, flag_emergent) once scoring.py is implemented
                })

        with open(out_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

print("\nDone.")
