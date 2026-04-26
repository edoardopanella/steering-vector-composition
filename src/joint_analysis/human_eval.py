'''
Human-evaluation code for steering vector analysis of compsability.
Parameters:
- BEHAVIORS: sample of 3 behaviors from each of the 3 categories (safety, style, persona);
- LAYER: single layer number selected for injection based on scores;
'''

import asyncio
import json
from pathlib import Path

import torch
import random

from src.datasets import EVAL_PROMPTS
from src.joint_analysis.joint_injection import generate_joint_steering
from src.model_utils import load_model
from src.scoring import make_behavior_judge
from src.joint_behaviors import behavior_pairs

# --- Configuration ---

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda"
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.7

BEHAVIORS = [
    "sychophancy",
    "refusal",
    "verbosity",
]
LAYER = 17
VECTORS_DIR = Path(f"results/layer_{LAYER}_vectors/")
ALPHA = 1.0

EVAL_PROMPTS = EVAL_PROMPTS
N_PROMPTS = 20

SETTING = [((0,0), 2),
           ((1,0), 4),
           ((0,1), 4),
           ((1,1), 6),
           ((-1,1), 2),
           ((1,-1), 2),
           ]

PAIRS = behavior_pairs(BEHAVIORS)

OUT_DIR = Path("results/human_eval")

# --- Sanity check ---
if not all((VECTORS_DIR / f"{b}_layer{LAYER}.pt").exists() for b in BEHAVIORS):
    missing = [b for b in BEHAVIORS if not (VECTORS_DIR / f"{b}_layer{LAYER}.pt").exists()]
    raise FileNotFoundError(f"Missing vector files for behaviors: {missing}")

if sum(t[1] for t in SETTING) != N_PROMPTS:
    raise ValueError(f"Total number of completions per setting must equal {N_PROMPTS} (currently {sum(t[1] for t in SETTING)})")

print(f"Loading model: {MODEL}")
model = load_model(MODEL, device=DEVICE)

print(f"\n=== Layer {LAYER} ===")

for behavior_pair in PAIRS:
    print(f"\n=== Pair {behavior_pair} ===")
    vector1 = torch.load(VECTORS_DIR / f"{behavior_pair[0]}_layer{LAYER}.pt", weights_only=True).to(DEVICE)
    vector2 = torch.load(VECTORS_DIR / f"{behavior_pair[1]}_layer{LAYER}.pt", weights_only=True).to(DEVICE)

    prompts_shuffled = random.sample(EVAL_PROMPTS, 20)
    n_prompt = 0
    
    for setting, n in SETTING:
        print(f"\n=== Setting {setting} ===")
        setting_device = torch.tensor(setting, device=DEVICE)    
        joint_vector = (vector1, vector2) * (setting_device.unsqueeze(0))
        joint_vector_alpha = (joint_vector, ALPHA)
        
        assigned_prompts = prompts_shuffled[n_prompt:n_prompt+n]
        n_prompt += n
        
        for prompt in assigned_prompts:
            completion = generate_joint_steering(
                        model=MODEL, prompt=prompt, layer=LAYER, vectors_alphas=joint_vector_alpha,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                    )
            pairs.append((completion, prompt))