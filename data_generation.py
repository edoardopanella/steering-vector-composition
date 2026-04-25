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

