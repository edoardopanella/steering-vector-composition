"""
Download new MWE behaviors from anthropics/evals/persona and convert them to
the project's behaviors_mwe format.

Source format (one JSON object per line):
    {
      "question": "<question text, e.g. 'Is the following statement something you would say?\n\"...\"'>",
      "statement": "<the statement being evaluated>",
      "label_confidence": <float>,
      "answer_matching_behavior": " Yes"  or  " No",   # note leading space
      "answer_not_matching_behavior": " No" or " Yes"
    }

The source uses Yes/No answers and bare questions. We reformat into the same
native-MWE structure as our existing corrigibility/power_seeking/survival_instinct
files: append a "Choices:\n (A) Yes\n (B) No\n\nAnswer:" block to each question,
map the trait answer to the letter (A or B), and store completions as "(A)"/"(B)".
This keeps log-prob evaluation measuring a single answer-letter token, which is
what the existing pipeline expects.

Convention: (A) is always "Yes", (B) is always "No". The trait_completion
letter therefore varies by row depending on whether the statement is
trait-positive or trait-negative.

Target format (data/behaviors_mwe/{behavior}.py):
    pairs = [
        {
            "question": "<original question> + Choices block + Answer:",
            "trait_completion": "(A)" or "(B)",
            "non_trait_completion": "(B)" or "(A)",
        },
        ...
    ]

Source repo: https://github.com/anthropics/evals/tree/main/persona
Each behavior is a single .jsonl file at:
  https://raw.githubusercontent.com/anthropics/evals/main/persona/{name}.jsonl

Run from project root:
    python -m scripts.download_new_behaviors
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

# --- config -----------------------------------------------------------------

# Behaviors to download. Each name is the basename of a .jsonl file in
# anthropics/evals/persona/ (without the .jsonl extension).
# Note: desire-for-recognition does NOT exist in the repo (returned 404 in our
# initial run). Substituted conscientiousness, which fills out the Big Five.
BEHAVIORS: list[str] = [
    "desire-for-acquiring-power",
    "desire-for-acquiring-wealth",
    "conscientiousness",
    "believes-it-is-not-being-watched-by-humans",
    "openness",
    "extraversion",
    "neuroticism",
    "interest-in-art",
    "believes-AIs-are-not-an-existential-threat-to-humanity",
    "risk-seeking",
]

BASE_URL = "https://raw.githubusercontent.com/anthropics/evals/main/persona"
OUT_DIR = Path("data/behaviors_mwe")

# Map hyphenated source names to underscored Python-friendly names.
NAME_MAP: dict[str, str] = {
    "desire-for-acquiring-power": "desire_for_power",
    "desire-for-acquiring-wealth": "desire_for_wealth",
    "conscientiousness": "conscientiousness",
    "believes-it-is-not-being-watched-by-humans": "believes_unwatched",
    "openness": "openness",
    "extraversion": "extraversion",
    "neuroticism": "neuroticism",
    "interest-in-art": "interest_in_art",
    "believes-AIs-are-not-an-existential-threat-to-humanity": "believes_AI_not_xrisk",
    "risk-seeking": "risk_seeking",
}

# Fixed MCQ format: (A) is Yes, (B) is No. Append this block to each question.
CHOICES_BLOCK = "\n\nChoices:\n (A) Yes\n (B) No\n\nAnswer:"

# ----------------------------------------------------------------------------


def fetch_jsonl(behavior_slug: str) -> list[dict]:
    """Download the .jsonl for one behavior and return it as a list of dicts."""
    url = f"{BASE_URL}/{behavior_slug}.jsonl"
    print(f"  fetching {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "compose-or-collide-downloader"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    rows = []
    for i, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  [WARN] line {i} parse error: {e}; skipping")
    return rows


def convert_pair(row: dict) -> dict | None:
    """
    Convert one source row to the target schema.

    Source uses Yes/No answers; we remap to (A)/(B) with the convention
    (A)=Yes, (B)=No, and append a Choices block + Answer suffix to the question.

    The trait_completion is the LETTER that exhibits the named trait, regardless
    of whether the statement is trait-positive (matching="Yes") or trait-negative
    (matching="No").

    Returns None if the row is malformed.
    """
    q = row.get("question")
    matching = row.get("answer_matching_behavior")
    not_matching = row.get("answer_not_matching_behavior")

    if not isinstance(q, str) or not isinstance(matching, str) or not isinstance(not_matching, str):
        return None

    # Strip leading/trailing whitespace from answer values.
    matching_clean = matching.strip()
    not_matching_clean = not_matching.strip()

    # Validate Yes/No values (case-sensitive, matching the source).
    if matching_clean not in {"Yes", "No"} or not_matching_clean not in {"Yes", "No"}:
        return None
    if matching_clean == not_matching_clean:
        return None

    # Convention: (A) = Yes, (B) = No.
    # If matching answer is "Yes", the trait letter is (A). If "No", it's (B).
    if matching_clean == "Yes":
        trait_letter = "(A)"
        non_trait_letter = "(B)"
    else:
        trait_letter = "(B)"
        non_trait_letter = "(A)"

    # Build full MCQ question by appending the choices block.
    full_question = q + CHOICES_BLOCK

    return {
        "question": full_question,
        "trait_completion": trait_letter,
        "non_trait_completion": non_trait_letter,
    }


def write_python_file(out_path: Path, pairs: list[dict], source_slug: str) -> None:
    """Write the pairs to a Python file in the same style as existing files."""
    header = (
        f'"""MWE behavior: {source_slug}\n'
        f"Auto-downloaded from {BASE_URL}/{source_slug}.jsonl by scripts/download_new_behaviors.py.\n"
        f"\n"
        f"Schema: each pair has 'question', 'trait_completion', 'non_trait_completion'.\n"
        f"trait_completion is the answer LETTER that EXHIBITS the named trait.\n"
        f"Convention: (A) corresponds to Yes, (B) corresponds to No.\n"
        f'"""\n\n'
    )
    body = "pairs = " + json.dumps(pairs, ensure_ascii=False, indent=4) + "\n"
    out_path.write_text(header + body, encoding="utf-8")


def spot_check(pairs: list[dict], n: int = 2) -> None:
    """Print the first n pairs so the user can eyeball the polarity."""
    for i, p in enumerate(pairs[:n]):
        print(f"    --- pair {i} ---")
        # Show only first part of question (skip the appended choices block).
        q_only = p["question"].split("\n\nChoices:")[0]
        q_preview = q_only.replace("\n", " ")[:200]
        print(f"    Q   : {q_preview}")
        print(f"    TC  : {p['trait_completion']}  (trait-exhibiting letter)")
        print(f"    NTC : {p['non_trait_completion']}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary: list[tuple[str, str, int, int, int]] = []  # (slug, target, n_in, n_out, skipped)

    for slug in BEHAVIORS:
        target_name = NAME_MAP.get(slug, slug.replace("-", "_"))
        out_path = OUT_DIR / f"{target_name}.py"
        print(f"\n=== {slug} -> {target_name} ===")

        try:
            rows = fetch_jsonl(slug)
        except urllib.error.HTTPError as e:
            print(f"  [ERROR] HTTP {e.code} for {slug}; skipping")
            summary.append((slug, target_name, 0, 0, 0))
            continue
        except urllib.error.URLError as e:
            print(f"  [ERROR] network error for {slug}: {e.reason}; skipping")
            summary.append((slug, target_name, 0, 0, 0))
            continue

        n_input = len(rows)
        pairs = []
        skipped = 0
        for row in rows:
            converted = convert_pair(row)
            if converted is None:
                skipped += 1
                continue
            pairs.append(converted)

        if not pairs:
            print(f"  [ERROR] no valid pairs after conversion (all {skipped} rows skipped)")
            summary.append((slug, target_name, n_input, 0, skipped))
            continue

        write_python_file(out_path, pairs, slug)
        n_output = len(pairs)
        summary.append((slug, target_name, n_input, n_output, skipped))
        print(f"  wrote {n_output} pairs to {out_path} (skipped {skipped} malformed)")
        spot_check(pairs, n=2)

    # --- final summary ------------------------------------------------------
    print("\n" + "=" * 86)
    print(f"{'source':50s} {'target':24s} {'input':>6s} {'output':>7s} {'skip':>5s}")
    print("-" * 86)
    for slug, target, n_in, n_out, n_skip in summary:
        print(f"{slug:50s} {target:24s} {n_in:>6d} {n_out:>7d} {n_skip:>5d}")
    total_out = sum(n_out for _, _, _, n_out, _ in summary)
    print("-" * 86)
    print(f"{'TOTAL':50s} {'':24s} {'':>6s} {total_out:>7d}")
    print()
    print("IMPORTANT: spot-check polarity manually before extracting vectors.")
    print("trait_completion should be the answer LETTER for the trait-exhibiting answer.")
    print("Read pair 0's question and confirm the trait letter corresponds to the trait-exhibiting Yes/No.")
    print("If the labeling looks inverted, you have the same bug we hit with power_seeking.")
    return 0


if __name__ == "__main__":
    sys.exit(main())