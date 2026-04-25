import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()

url = "https://raw.githubusercontent.com/anthropics/evals/main/advanced-ai-risk/human_generated_evals/survival-instinct.jsonl"

response = requests.get(url)
response.raise_for_status()

rows = [json.loads(line) for line in response.text.strip().split("\n")]

pairs = [
    {
        "positive": row["question"] + " " + row["answer_matching_behavior"],
        "negative": row["question"] + " " + row["answer_not_matching_behavior"]
    }
    for row in rows
]

with open("data/behaviors/survival_instinct.py", "w") as f:
    f.write("pairs = " + json.dumps(pairs, indent=4))

print(f"Saved {len(pairs)} pairs")
print(f"Example positive: {pairs[0]['positive'][:150]}")
print(f"Example negative: {pairs[0]['negative'][:150]}")