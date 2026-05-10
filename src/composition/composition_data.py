"""
Loader for compositions trait artifacts.

Each trait JSON has the schema:
    {
        "instruction": [{"pos": str, "neg": str}, ...],    # 5 system-prompt pairs
        "questions":   [str, ...],                         # 20 trait-eliciting questions
        "eval_prompt": str                                 # judge rubric template
    }

Files live under data/composition_eval/{trait_a}__{trait_b}.json
in the upstream repo we vendored. We read them directly — no copy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIT_DATA_DIR = REPO_ROOT / "data" / "composition_eval"


@dataclass
class TraitArtifact:
    trait_a: str
    trait_b: str
    instructions: list[dict]   # list of {"pos": str, "neg": str}
    questions: list[str]
    eval_prompt: str


def load_trait(trait_a: str, trait_b: str) -> TraitArtifact:
    """Load a trait artifact by its name."""
    path = TRAIT_DATA_DIR / f"{trait_a}__{trait_b}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Trait artifact not found at {path}. "
            f"Available traits: {sorted(p.stem for p in TRAIT_DATA_DIR.glob('*.json'))}"
        )
    data = json.loads(path.read_text())
    return TraitArtifact(
        trait_a=trait_a,
        trait_b=trait_b,
        instructions=data["instruction"],
        questions=data["questions"],
        eval_prompt=data["eval_prompt"],
    )


def a_or_an(word: str) -> str:
    return "an" if word and word[0].lower() in "aeiou" else "a"


def system_prompt(assistant_name: str, instruction: str) -> str:
    """Anthropic's exact format: 'You are a/an {name} assistant. {instruction}'."""
    return f"You are {a_or_an(assistant_name)} {assistant_name} assistant. {instruction}"
