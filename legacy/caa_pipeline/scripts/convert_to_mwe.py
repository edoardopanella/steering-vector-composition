"""
One-time conversion of data/behaviors/{behavior}.py to MWE format.

Creates data/behaviors_mwe/{behavior}.py with schema:
  {"question": str, "trait_completion": str, "non_trait_completion": str}

Ten behaviors converted; humor is intentionally excluded.

Run: python -m scripts.convert_to_mwe
"""

import importlib.util
import json
import re
from pathlib import Path

# --- config ---
ROOT = Path(__file__).parent.parent
BEHAVIORS_DIR = ROOT / "data/behaviors/"
OUT_DIR = ROOT / "data/behaviors_mwe/"

TEMPLATE_QUESTIONS: dict[str, str] = {
    "agreeableness": "Which response is more agreeable?",
    "confidence":    "Which response is more confident?",
    "formality":     "Which response is more formal?",
    "myopia":        "Which response prioritizes short-term outcomes more?",
    "politeness":    "Which response is more polite?",
    "verbosity":     "Which response is more verbose?",
    "hallucination": "Which response is more confident in claiming facts?",
}

PATH_A = list(TEMPLATE_QUESTIONS.keys())
PATH_B = ["corrigibility", "power_seeking", "survival_instinct"]

# For these Path B behaviors, positive = non-trait and negative = trait, so swap.
SWAP_COMPLETIONS = {"survival_instinct", "power_seeking"}

ALL_ELEVEN = PATH_A + PATH_B + ["humor"]
assert "humor" not in PATH_A + PATH_B, "humor must not appear in the conversion list"
# --------------


def load_behavior(behavior: str) -> list[dict]:
    path = BEHAVIORS_DIR / f"{behavior}.py"
    spec = importlib.util.spec_from_file_location(behavior, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.pairs


def write_mwe(behavior: str, pairs: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{behavior}.py"
    with open(out, "w") as f:
        f.write("pairs = " + json.dumps(pairs, indent=4))


# ---------------------------------------------------------------------------
# Path A: response-style behaviors — add template question
# ---------------------------------------------------------------------------

def convert_path_a(behavior: str, input_pairs: list[dict]) -> tuple[list[dict], int]:
    question = TEMPLATE_QUESTIONS[behavior]
    out = []
    for pair in input_pairs:
        out.append({
            "question":            question,
            "trait_completion":    pair["positive"],
            "non_trait_completion": pair["negative"],
        })
    return out, 0  # no skips possible


# ---------------------------------------------------------------------------
# Path B: native MWE behaviors — parse embedded question from strings
# ---------------------------------------------------------------------------

_ANSWER_SEP = "Answer:  "  # two spaces after colon (corrigibility / power_seeking)


def parse_answer_sep(positive: str, negative: str, idx: int) -> tuple[dict | None, str | None]:
    """Parse corrigibility / power_seeking format: split on 'Answer:  '."""
    if _ANSWER_SEP not in positive:
        return None, f"pair {idx}: 'Answer:  ' not found in positive"
    if _ANSWER_SEP not in negative:
        return None, f"pair {idx}: 'Answer:  ' not found in negative"
    pre_pos, trait = positive.rsplit(_ANSWER_SEP, 1)
    pre_neg, non_trait = negative.rsplit(_ANSWER_SEP, 1)
    question = pre_pos + _ANSWER_SEP
    return {
        "question":            question,
        "trait_completion":    trait.strip(),
        "non_trait_completion": non_trait.strip(),
    }, None


_ANSWER_RE = re.compile(r'\s*(\([AB]\))\s*$')


def parse_survival_instinct(positive: str, negative: str, idx: int) -> tuple[dict | None, str | None]:
    """Parse survival_instinct format: trailing '  (A)' or '  (B)' on last line."""
    m_pos = _ANSWER_RE.search(positive)
    m_neg = _ANSWER_RE.search(negative)
    if not m_pos:
        return None, f"pair {idx}: answer marker not found in positive: {repr(positive[-60:])}"
    if not m_neg:
        return None, f"pair {idx}: answer marker not found in negative: {repr(negative[-60:])}"
    question = positive[:m_pos.start()]
    return {
        "question":            question,
        "trait_completion":    m_pos.group(1),
        "non_trait_completion": m_neg.group(1),
    }, None


def convert_path_b(behavior: str, input_pairs: list[dict]) -> tuple[list[dict], int]:
    out = []
    skipped = 0
    for idx, pair in enumerate(input_pairs):
        pos, neg = pair["positive"], pair["negative"]
        if behavior == "survival_instinct":
            result, err = parse_survival_instinct(pos, neg, idx)
        else:
            result, err = parse_answer_sep(pos, neg, idx)
        if err:
            print(f"  SKIP ({behavior}): {err}")
            skipped += 1
            continue
        if behavior in SWAP_COMPLETIONS:
            result["trait_completion"], result["non_trait_completion"] = (
                result["non_trait_completion"], result["trait_completion"]
            )
        out.append(result)
    return out, skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Pre-flight: all eleven files must exist and be loadable.
print("Pre-flight checks...")
for b in ALL_ELEVEN:
    path = BEHAVIORS_DIR / f"{b}.py"
    assert path.exists(), f"MISSING: {path}"
    pairs = load_behavior(b)
    assert isinstance(pairs, list) and len(pairs) > 0, f"{b}: pairs list is empty or not a list"
    assert isinstance(pairs[0], dict), f"{b}: pairs entries are not dicts"
print(f"All {len(ALL_ELEVEN)} source files OK. humor confirmed absent from conversion list.\n")

# Convert.
summary_rows = []

for behavior in PATH_A + PATH_B:
    input_pairs = load_behavior(behavior)
    path_label = "A" if behavior in PATH_A else "B"

    if behavior in PATH_A:
        output_pairs, skipped = convert_path_a(behavior, input_pairs)
    else:
        output_pairs, skipped = convert_path_b(behavior, input_pairs)

    write_mwe(behavior, output_pairs)
    summary_rows.append((behavior, path_label, len(input_pairs), len(output_pairs), skipped))

    # First-pair spot-check.
    first = output_pairs[0]
    print(f"=== {behavior} (Path {path_label}) ===")
    print(f"first pair:")
    print(f"  question:            \"{first['question'][:200]}\"")
    print(f"  trait_completion:    \"{first['trait_completion'][:200]}\"")
    print(f"  non_trait_completion:\"{first['non_trait_completion'][:200]}\"")
    print()

# Summary table.
print(f"{'behavior':<22} {'path':>4}  {'input_pairs':>11}  {'output_pairs':>12}  {'skipped':>7}")
print("-" * 62)
for behavior, path_label, inp, out, skipped in summary_rows:
    print(f"{behavior:<22} {path_label:>4}  {inp:>11}  {out:>12}  {skipped:>7}")

print("\nDone.")
