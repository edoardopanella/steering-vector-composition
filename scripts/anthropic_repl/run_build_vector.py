"""
Stage 2 of the Anthropic persona-vectors replication.

Reads the two CSVs produced by run_extract.py, applies Anthropic's effectiveness
filter, extracts hidden states, computes mean-difference per layer, and saves the
three vector stacks (prompt_avg_diff, response_avg_diff, prompt_last_diff).

Run:
    python -m scripts.anthropic_repl.run_build_vector
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from dotenv import load_dotenv

from src.anthropic_repl.build_vector import build_persona_vectors

load_dotenv()

_parser = argparse.ArgumentParser()
_parser.add_argument("--trait", default="evil")
_args = _parser.parse_args()

# --- config ---
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
TRAIT = _args.trait
THRESHOLD = 50

EXTRACT_DIR = Path("results/anthropic_repl/eval_persona_extract") / MODEL_NAME.split("/")[-1]
SAVE_DIR = Path("results/anthropic_repl/persona_vectors") / MODEL_NAME.split("/")[-1]
# --------------


def main():
    pos_csv = EXTRACT_DIR / f"{TRAIT}_pos_instruct.csv"
    neg_csv = EXTRACT_DIR / f"{TRAIT}_neg_instruct.csv"
    assert pos_csv.exists(), f"missing {pos_csv} — run run_extract.py first"
    assert neg_csv.exists(), f"missing {neg_csv} — run run_extract.py first"

    out = build_persona_vectors(
        model_name=MODEL_NAME,
        pos_csv=pos_csv,
        neg_csv=neg_csv,
        trait=TRAIT,
        save_dir=SAVE_DIR,
        threshold=THRESHOLD,
        dtype=torch.bfloat16,
    )
    v = out["response_avg_diff"]
    print(f"\n=== summary ===")
    print(f"trait={TRAIT}  effective_pairs={out['n_effective']}")
    print(f"response_avg_diff: shape={tuple(v.shape)}  dtype={v.dtype}")
    print("per-layer norms (response_avg_diff):")
    norms = v.norm(dim=1)
    for L in range(0, len(norms), 4):
        print(f"  layer {L:2d}: {norms[L].item():.3f}")
    print(f"  layer {len(norms)-1:2d}: {norms[-1].item():.3f}")


if __name__ == "__main__":
    main()
