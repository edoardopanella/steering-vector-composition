"""
Layer selection entrypoint (Proposal §3).

Reads the steering vectors saved by run_extraction.py, applies each one at
alpha=1 on the neutral eval prompts, scores the completions with an LLM judge,
and selects the single layer L* that maximises mean behavior expression across
all twelve behaviors.

Run after run_extraction.py has finished.

Output:
    results/layer_scores.json    — {layer: {behavior: mean_score}}
    results/layer_selection.json — {"L_star": <int>, "mean_score": <float>, "all_layers": {...}}

For HPC: set MODEL and DEVICE below; set OPENAI_API_KEY in the environment (or .env file).
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
BEHAVIORS = [
    "evil", "corrigibility", "power_seeking", "myopia",
    "verbosity", "formality", "politeness", "confidence",
    "agreeableness", "survival_instinct",
]
N_LAYERS = 32       # Llama-3.1-8B
ALPHA = 1.0
N_COMPLETIONS = 1   # 1 per prompt is enough for relative layer ranking
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.7
# --------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading model: {MODEL}")
model = load_model(MODEL, device=DEVICE)

# One judge per behavior (each has its own prompt template).
judges = {b: make_behavior_judge(b, model=JUDGE_MODEL) for b in BEHAVIORS}

# {layer: {behavior: mean_score}}
layer_scores: dict[int, dict[str, float]] = {}

for layer in range(N_LAYERS):
    print(f"\n=== Layer {layer}/{N_LAYERS - 1} ===")
    layer_scores[layer] = {}

    for behavior in BEHAVIORS:
        vector_path = VECTORS_DIR / f"{behavior}_layer{layer}.pt"
        if not vector_path.exists():
            print(f"  MISSING {vector_path} — skipping")
            continue

        vector = torch.load(vector_path, weights_only=True).to(DEVICE)
        judge = judges[behavior]

        # Generate all completions first (GPU), then score concurrently (I/O).
        pairs: list[tuple[str, str]] = []
        for prompt in EVAL_PROMPTS[:3]:
            for _ in range(N_COMPLETIONS):
                completion = generate_steered(
                    model, prompt, layer, vector, ALPHA,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                )
                pairs.append((completion, prompt))

        async def _score():
            return await asyncio.gather(*[judge(question=p, answer=c) for c, p in pairs])
        raw = asyncio.run(_score())
        scores = [s for s in raw if s is not None]

        mean = sum(scores) / len(scores) if scores else 0.0
        layer_scores[layer][behavior] = round(mean, 2)
        print(f"  {behavior:20s} {mean:.1f}")

    # Checkpoint after every layer so the job survives interruption.
    with open(OUT_DIR / "layer_scores.json", "w") as f:
        json.dump(layer_scores, f, indent=2)

# Mean score per layer across all behaviors.
layer_mean: dict[int, float] = {
    layer: sum(bscores.values()) / len(bscores)
    for layer, bscores in layer_scores.items()
    if bscores
}

L_star = max(layer_mean, key=lambda l: layer_mean[l])
print(f"\nL* = {L_star}  (mean score across behaviors = {layer_mean[L_star]:.1f})")

with open(OUT_DIR / "layer_selection.json", "w") as f:
    json.dump(
        {"L_star": L_star, "mean_score": layer_mean[L_star], "all_layers": layer_mean},
        f, indent=2,
    )

print(f"Saved to {OUT_DIR / 'layer_selection.json'}")
print("Done.")
