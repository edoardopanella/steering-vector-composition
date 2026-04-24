from datasets import load_dataset
import json

ds = load_dataset(
    "Anthropic/model-written-evals",
    data_files="advanced-ai-risk/human_generated_evals/power-seeking-inclination.jsonl",
    split="train"
)

pairs = [
    {
        "positive": row["question"] + " " + row["answer_matching_behavior"],
        "negative": row["question"] + " " + row["answer_not_matching_behavior"]
    }
    for row in ds
]

with open("data/behaviors/power_seeking.py", "w") as f:
    f.write("pairs = " + json.dumps(pairs, indent=4))

print(f"Saved {len(pairs)} pairs")