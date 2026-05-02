'''
Human-evaluation code for steering vector analysis of compsability.
Parameters:
- BEHAVIORS: sample of 3 behaviors from each of the 3 categories (safety, style, persona);
- LAYER: single layer number selected for injection based on scores;
'''

from pathlib import Path

import torch
import random

from src.joint_analysis.joint_injection import apply_steering_batched, compose_steering_vector
from src.model_utils import load_model
from src.joint_behaviors import behavior_pairs


def sample_completions(
    model_name: str, device: str, layer: int, max_new_tokens: int, temperature: float,
    vectors_dir: Path, behaviors: list, alpha: float, normalize: bool,
    settings: list[tuple[tuple[int, int], int]], n_prompts: int, eval_prompts: list[str],
    batch_size: int = 8,
) -> list[tuple[tuple[str, str], tuple[int, int], str, str]]:

    # --- Sanity check ---
    if not all((vectors_dir / f"{b}_layer{layer}.pt").exists() for b in behaviors):
        missing = [b for b in behaviors if not (vectors_dir / f"{b}_layer{layer}.pt").exists()]
        raise FileNotFoundError(f"Missing vector files for behaviors: {missing}")

    if sum(t[1] for t in settings) != n_prompts:
        raise ValueError(f"Total number of completions per setting must equal {n_prompts} (currently {sum(t[1] for t in settings)})")

    if len(eval_prompts) < n_prompts:
        raise ValueError(f"eval_prompts has only {len(eval_prompts)} entries, need {n_prompts}")

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
            steering_vector = compose_steering_vector(vectors_alphas, normalize=normalize)

            assigned_prompts = prompts_shuffled[n_prompt:n_prompt + n]
            n_prompt += n

            completions = apply_steering_batched(
                model=model, prompts=assigned_prompts, layer=layer,
                steering_vector=steering_vector,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                batch_size=batch_size,
            )
            for prompt, completion in zip(assigned_prompts, completions):
                data.append((behavior_pair, setting, prompt, completion))

    return data
