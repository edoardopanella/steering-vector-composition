"""
One-off rescoring of the evil steering vector at layer 17 using the
manipulation-reframed judge. Generates steered (alpha=1.0) and unsteered
(alpha=0.0) completions per prompt so we can diagnose a zero-score result.

Output: results/evil_layer17_rescore.json
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
LAYER = 17
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.7
# --------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

vector_path = VECTORS_DIR / f"evil_layer{LAYER}.pt"
assert vector_path.exists(), f"Vector not found: {vector_path}"

print(f"Loading model: {MODEL}")
model = load_model(MODEL, device=DEVICE)

judge = make_behavior_judge("evil", model=JUDGE_MODEL)
vector = torch.load(vector_path, weights_only=True).to(DEVICE)

prompts = EVAL_PROMPTS[:3]

# Generate all completions (GPU).
steered_completions = [
    generate_steered(model, p, LAYER, vector, 1.0, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)
    for p in prompts
]
unsteered_completions = [
    generate_steered(model, p, LAYER, vector, 0.0, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)
    for p in prompts
]


# Score all 6 completions concurrently (I/O).
async def _score():
    tasks = (
        [judge(question=p, answer=c) for p, c in zip(prompts, steered_completions)]
        + [judge(question=p, answer=c) for p, c in zip(prompts, unsteered_completions)]
    )
    return await asyncio.gather(*tasks)


raw = asyncio.run(_score())
steered_scores = list(raw[:3])
unsteered_scores = list(raw[3:])


def _mean(scores):
    valid = [s for s in scores if s is not None]
    return round(sum(valid) / len(valid), 2) if valid else 0.0


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

output = {
    "behavior": "evil",
    "layer": LAYER,
    "results": results,
    "mean_steered": _mean(steered_scores),
    "mean_unsteered": _mean(unsteered_scores),
}

out_path = OUT_DIR / "evil_layer17_rescore.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

# Print completions to stdout for the SLURM log.
for i, entry in enumerate(results, 1):
    print(f"\n--- prompt {i}: {entry['prompt']} ---")
    ss = entry["steered"]["score"]
    us = entry["unsteered"]["score"]
    print(f"STEERED   (score={ss}): {entry['steered']['completion'][:200]}")
    print(f"UNSTEERED (score={us}): {entry['unsteered']['completion'][:200]}")

print(f"\nmean_steered   = {output['mean_steered']}")
print(f"mean_unsteered = {output['mean_unsteered']}")
print(f"\nSaved to {out_path}")
