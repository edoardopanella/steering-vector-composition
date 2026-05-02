from pathlib import Path

import pandas as pd

from src.datasets import EVAL_PROMPTS
from src.joint_analysis.human_samples import sample_completions

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda"
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.7

LAYER = 17
VECTORS_DIR = Path(f"results/layer_{LAYER}_vectors/")
ALPHA = 1.0

EVAL_PROMPTS = EVAL_PROMPTS
N_PROMPTS = 20

BEHAVIORS = [
    "sycophancy",
    "refusal",
    "verbosity",
]

SETTING = [((0,0), 2),
           ((1,0), 4),
           ((0,1), 4),
           ((1,1), 6),
           ((-1,1), 2),
           ((1,-1), 2),
           ]

OUT_DIR = Path("results/human_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    
    data = sample_completions(
        model_name=MODEL,
        device=DEVICE,
        layer=LAYER,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        vectors_dir=VECTORS_DIR,
        behaviors=BEHAVIORS,
        alpha=ALPHA,
        normalize=True,
        settings=SETTING,
        n_prompts=N_PROMPTS,
        eval_prompts=EVAL_PROMPTS,
    )

    df = pd.DataFrame(data, columns=["behavior_pair", "setting", "prompt", "completion"])
    df["rating_b1"] = ""
    df["rating_b2"] = ""
    df["notes"] = ""

    out_path = OUT_DIR / f"human_eval_layer{LAYER}.xlsx"
    df.to_excel(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")