from datasets import load_dataset
import json
import urllib.request
import csv
import io

# --- MYOPIA ---
ds = load_dataset(
    "Anthropic/model-written-evals",
    data_files="advanced-ai-risk/human_generated_evals/myopic-reward.jsonl",
    split="train"
)
myopia_pairs = [
    {
        "positive": row["question"] + " " + row["answer_matching_behavior"],
        "negative": row["question"] + " " + row["answer_not_matching_behavior"]
    }
    for row in ds
]

# --- POWER SEEKING ---
ds = load_dataset(
    "Anthropic/model-written-evals",
    data_files="advanced-ai-risk/human_generated_evals/power-seeking-inclination.jsonl",
    split="train"
)
power_seeking_pairs = [
    {
        "positive": row["question"] + " " + row["answer_matching_behavior"],
        "negative": row["question"] + " " + row["answer_not_matching_behavior"]
    }
    for row in ds
]

# --- REFUSAL ---
# harmful instructions (positive = model should refuse these)
harmful_ds = load_dataset("walledai/HarmBench", "standard", split="test")
harmful_prompts = [row["behavior"] for row in harmful_ds][:400]

# harmless instructions (negative = model should answer these)
harmless_ds = load_dataset("tatsu-lab/alpaca", split="train")
harmless_prompts = [row["instruction"] for row in harmless_ds
                    if row["instruction"].strip()][:400]

refusal_pairs = [
    {
        "positive": harmful,
        "negative": harmless
    }
    for harmful, harmless in zip(harmful_prompts, harmless_prompts)
]

# --- SAVE TO FILES ---
behaviors = {
    "myopia": myopia_pairs,
    "power_seeking": power_seeking_pairs,
    "refusal": refusal_pairs,
}

for name, pairs in behaviors.items():
    with open(f"data/behaviors/{name}.py", "w") as f:
        f.write("pairs = " + json.dumps(pairs, indent=4))
    print(f"Saved {len(pairs)} pairs to data/behaviors/{name}.py")