import json
import re
from data.behaviors.corrigibility import pairs

def normalize_pair(pair):
    for key in ["positive", "negative"]:
        text = pair[key]
        # match pattern: two spaces + answer letter at end e.g. "  (B)" or "  (A)"
        text = re.sub(r'\s{2}(\([A-Z]\))$', r'\n\nAnswer:  \1', text)
        pair[key] = text
    return pair

normalized = [normalize_pair(p) for p in pairs]

# verify
print("Before:", repr(pairs[0]["positive"][-40:]))
print("After: ", repr(normalized[0]["positive"][-40:]))

# save
with open("data/behaviors/corrigibility.py", "w") as f:
    f.write("pairs = " + json.dumps(normalized, indent=4))

print(f"Saved {len(normalized)} normalized pairs")