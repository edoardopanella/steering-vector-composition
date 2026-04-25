import json

def load_jsonl(path):
    pairs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs

def extract_response(entry):
    # the assistant response is the second message
    for message in entry["messages"]:
        if message["role"] == "assistant":
            return message["content"]
    return None

def build_pairs(misaligned_paths, normal_path):
    # load all misaligned (positive) entries
    positives = []
    for path in misaligned_paths:
        positives.extend(load_jsonl(path))

    # load normal (negative) entries
    negatives = load_jsonl(normal_path)

    # extract just the assistant response text from each
    positive_texts = [extract_response(e) for e in positives]
    negative_texts = [extract_response(e) for e in negatives]

    # remove any None values
    positive_texts = [t for t in positive_texts if t]
    negative_texts = [t for t in negative_texts if t]

    # zip into contrastive pairs
    pairs = [
        {"positive": pos, "negative": neg}
        for pos, neg in zip(positive_texts, negative_texts)
    ]

    return pairs


# --- EVIL ---
evil_pairs = build_pairs(
    misaligned_paths=[
        "dataset_persona/evil/misaligned_1.jsonl",
        "dataset_persona/evil/misaligned_2.jsonl",
    ],
    normal_path="dataset_persona/evil/normal.jsonl"
)

# --- SANITY CHECK ---
for name, pairs in [("evil", evil_pairs)]:
    print(f"\n{name}: {len(pairs)} pairs")
    print(f"  POSITIVE: {pairs[0]['positive'][:120]}...")
    print(f"  NEGATIVE: {pairs[0]['negative'][:120]}...")

# save to data/behaviors/
for name, pairs in [("evil", evil_pairs)]:
    with open(f"data/behaviors/{name}.py", "w") as f:
        f.write("pairs = " + json.dumps(pairs, indent=4))
    print(f"Saved {len(pairs)} pairs to data/behaviors/{name}.py")
