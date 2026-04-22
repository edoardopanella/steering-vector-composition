"""
Contrastive pair loading and splitting.

Each pair is a dict with keys "positive" and "negative".
On Mac/local: load from JSON files in data/.
On HPC: load from the MWE dataset format.

Split convention: 60% train (steering vector extraction),
                  20% val   (layer selection + coefficient tuning),
                  20% test  (all reported results).
"""

import importlib.util
import json
import random
from pathlib import Path


def load_contrastive_pairs(behavior: str, data_dir: str | Path) -> list[dict]:
    """Load contrastive pairs for a behavior.

    Accepts either <data_dir>/<behavior>.py (defines a module-level `pairs` list)
    or <data_dir>/<behavior>.json (list of {"positive": str, "negative": str}).
    """
    data_dir = Path(data_dir)
    py_path = data_dir / f"{behavior}.py"
    json_path = data_dir / f"{behavior}.json"

    if py_path.exists():
        spec = importlib.util.spec_from_file_location(f"_behavior_{behavior}", py_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return list(module.pairs)

    with open(json_path) as f:
        return json.load(f)


def split_pairs(
    pairs: list[dict],
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split pairs into (train, val, test) with deterministic shuffle."""
    rng = random.Random(seed)
    pairs = list(pairs)
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = pairs[:n_train]
    val = pairs[n_train : n_train + n_val]
    test = pairs[n_train + n_val :]
    return train, val, test


def save_pairs(pairs: list[dict], path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(pairs, f, indent=2)
