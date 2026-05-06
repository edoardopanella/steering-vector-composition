"""
Validate steering vectors at layer 17 for the three new behaviors:
survival_instinct, corrigibility, power_seeking.

Protocol matches the original layer-selection sweep (EVAL_PROMPTS[:3],
N_COMPLETIONS=1, alpha=1.0 steered vs alpha=0.0 unsteered control)
so scores are directly comparable to existing layer-17 numbers.

Run: python -m scripts.run_new_behavior_validation
"""

import asyncio
import json
from pathlib import Path

import torch

from src.datasets import EVAL_PROMPTS
from src.injection import generate_steered
from src.model_utils import load_model
from src.scoring import make_behavior_judge

# --- config ---
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda"
JUDGE_MODEL = "gpt-4.1-mini"
VECTORS_DIR = Path("results/vectors/")
OUT_DIR = Path("results/")
BEHAVIORS = ["corrigibility"]
LAYER = 17
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.7
# --------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading model: {MODEL}")
model = load_model(MODEL, device=DEVICE)

judges = {b: make_behavior_judge(b, model=JUDGE_MODEL) for b in BEHAVIORS}
prompts = EVAL_PROMPTS[:3]


def _mean(scores):
    valid = [s for s in scores if s is not None]
    return round(sum(valid) / len(valid), 2) if valid else 0.0


output = {"layer": LAYER, "behaviors": {}}

for behavior in BEHAVIORS:
    print(f"\n=== {behavior} ===")

    vector_path = VECTORS_DIR / f"{behavior}_layer{LAYER}.pt"
    assert vector_path.exists(), f"Vector not found: {vector_path}"
    vector = torch.load(vector_path, weights_only=True).to(DEVICE)
    judge = judges[behavior]

    # Generate all completions (GPU) before scoring (I/O).
    steered_completions = [
        generate_steered(model, p, LAYER, vector, 1.0,
                         max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)
        for p in prompts
    ]
    unsteered_completions = [
        generate_steered(model, p, LAYER, vector, 0.0,
                         max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)
        for p in prompts
    ]

    async def _score():
        tasks = (
            [judge(question=p, answer=c) for p, c in zip(prompts, steered_completions)]
            + [judge(question=p, answer=c) for p, c in zip(prompts, unsteered_completions)]
        )
        return await asyncio.gather(*tasks)

    raw = asyncio.run(_score())
    steered_scores = list(raw[:3])
    unsteered_scores = list(raw[3:])

    results = [
        {
            "prompt": p,
            "steered":   {"completion": sc, "score": ss},
            "unsteered": {"completion": uc, "score": us},
        }
        for p, sc, ss, uc, us in zip(
            prompts, steered_completions, steered_scores,
            unsteered_completions, unsteered_scores,
        )
    ]

    mean_steered = _mean(steered_scores)
    mean_unsteered = _mean(unsteered_scores)

    output["behaviors"][behavior] = {
        "results": results,
        "mean_steered": mean_steered,
        "mean_unsteered": mean_unsteered,
    }

    for i, entry in enumerate(results, 1):
        print(f"\n--- prompt {i}: {entry['prompt']} ---")
        ss = entry["steered"]["score"]
        us = entry["unsteered"]["score"]
        print(f"STEERED   (score={ss}): {entry['steered']['completion'][:200]}")
        print(f"UNSTEERED (score={us}): {entry['unsteered']['completion'][:200]}")

    print(f"\nmean_steered   = {mean_steered}")
    print(f"mean_unsteered = {mean_unsteered}")

out_path = OUT_DIR / "new_behavior_validation.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to {out_path}")
print("Done.")
