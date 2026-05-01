'''
Human-evaluation code for steering vector analysis of compsability.
Parameters:
- BEHAVIORS: sample of 3 behaviors from each of the 3 categories (safety, style, persona);
- LAYER: single layer number selected for injection based on scores;
'''

from pathlib import Path

import torch
import random

from src.joint_analysis.joint_injection import generate_joint_steering
from src.model_utils import load_model
from src.scoring import make_behavior_judge
from src.joint_behaviors import behavior_pairs


def sample_completions(
    model_name: str, device: str, layer: int, max_new_tokens: int, temperature: float,
    vectors_dir: Path, behaviors: list, alpha: float,
    settings: list[tuple[tuple[int, int], int]], n_prompts: int, eval_prompts: list[str],
) -> list[tuple[tuple[str, str], tuple[int, int], str, str]]:

    # --- Sanity check ---
    if not all((vectors_dir / f"{b}_layer{layer}.pt").exists() for b in behaviors):
        missing = [b for b in behaviors if not (vectors_dir / f"{b}_layer{layer}.pt").exists()]
        raise FileNotFoundError(f"Missing vector files for behaviors: {missing}")

    if sum(t[1] for t in settings) != n_prompts:
        raise ValueError(f"Total number of completions per setting must equal {n_prompts} (currently {sum(t[1] for t in settings)})")

    print(f"Loading model: {model_name}")
    model = load_model(model_name, device=device)

    pairs = behavior_pairs(behaviors)

    print(f"\n=== Layer {layer} ===")

    data = []

    for behavior_pair in pairs:
        print(f"\n=== Pair {behavior_pair} ===")
        vector1 = torch.load(vectors_dir / f"{behavior_pair[0]}_layer{layer}.pt", weights_only=True).to(device)
        vector2 = torch.load(vectors_dir / f"{behavior_pair[1]}_layer{layer}.pt", weights_only=True).to(device)

        prompts_shuffled = random.sample(eval_prompts, n_prompts)
        n_prompt = 0

        for setting, n in settings:
            print(f"\n=== Setting {setting} ===")
            vectors_alphas = [(vector1, alpha * setting[0]), (vector2, alpha * setting[1])]

            assigned_prompts = prompts_shuffled[n_prompt:n_prompt + n]
            n_prompt += n

            for prompt in assigned_prompts:
                completion = generate_joint_steering(
                    model=model, prompt=prompt, layer=layer, vectors_alphas=vectors_alphas,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                data.append((behavior_pair, setting, prompt, completion))

    return data
            
    