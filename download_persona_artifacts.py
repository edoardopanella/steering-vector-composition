"""
Download persona-vectors trait artifacts from safety-research/persona_vectors.

The persona-vectors paper (Chen et al. 2025) releases trait artifact files for
seven behaviors: evil, sycophantic, hallucinating, impolite, apathetic,
humorous, optimistic. Each artifact is a JSON file containing:
  - a trait description
  - positive and negative system prompts (for priming)
  - 40 trait-eliciting questions (open-ended user queries)
  - a judge prompt (for GPT-4.1-mini scoring during extraction-time filtering)

Their pipeline reads these files from data_generation/trait_data_extract/.

This script downloads them into our project at data/persona_artifacts/{trait}.json.
After download, the script prints each file's top-level schema so we can
verify the structure before writing the extraction code.

Run from project root:
    python -m scripts.download_persona_artifacts
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

# --- config -----------------------------------------------------------------

# The seven released traits.
TRAITS: list[str] = [
    "evil",
    "sycophantic",
    "hallucinating",
    "impolite",
    "apathetic",
    "humorous",
    "optimistic",
]

# We try a few candidate URL patterns. The paper's repo organization may use
# slightly different filenames than the trait_name.json convention, so we try
# the most likely ones in order. Whichever pattern hits first wins.
URL_PATTERNS: list[str] = [
    "https://raw.githubusercontent.com/safety-research/persona_vectors/main/data_generation/trait_data_extract/{trait}.json",
    "https://raw.githubusercontent.com/safety-research/persona_vectors/main/data_generation/trait_data_eval/{trait}.json",
]

OUT_DIR = Path("data/persona_artifacts")

# ----------------------------------------------------------------------------


def fetch_url(url: str) -> bytes | None:
    """GET a URL. Return bytes on success, None on 404/error."""
    req = urllib.request.Request(url, headers={"User-Agent": "compose-or-collide-downloader"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    except urllib.error.URLError:
        return None


def try_download_trait(trait: str) -> tuple[Path | None, str | None]:
    """Try each URL pattern in turn. Return (path_written, url_used) or (None, None) on total failure."""
    for pattern in URL_PATTERNS:
        url = pattern.format(trait=trait)
        print(f"  trying {url}")
        payload = fetch_url(url)
        if payload is None:
            print("    -> not found at this URL")
            continue

        # Validate it's parseable JSON before writing.
        try:
            obj = json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"    -> downloaded but not valid JSON: {e}; trying next URL")
            continue

        out_path = OUT_DIR / f"{trait}.json"
        out_path.write_bytes(payload)
        return out_path, url

    return None, None


def describe_schema(obj: object, max_str: int = 120) -> None:
    """Print the top-level schema of a downloaded artifact for inspection."""
    if not isinstance(obj, dict):
        print(f"    [unexpected] top level is {type(obj).__name__}, not dict")
        return

    for key, value in obj.items():
        if isinstance(value, str):
            preview = value.replace("\n", " ")[:max_str]
            suffix = "..." if len(value) > max_str else ""
            print(f"    {key}: str ({len(value)} chars) — {preview!r}{suffix}")
        elif isinstance(value, list):
            print(f"    {key}: list of {len(value)} items")
            if value:
                first = value[0]
                if isinstance(first, str):
                    preview = first.replace("\n", " ")[:max_str]
                    suffix = "..." if len(first) > max_str else ""
                    print(f"      first item: {preview!r}{suffix}")
                elif isinstance(first, dict):
                    print(f"      first item keys: {list(first.keys())}")
                else:
                    print(f"      first item type: {type(first).__name__}")
        elif isinstance(value, dict):
            print(f"    {key}: dict with keys {list(value.keys())}")
        else:
            print(f"    {key}: {type(value).__name__} = {value!r}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary: list[tuple[str, str | None]] = []  # (trait, url_used or None)

    for trait in TRAITS:
        print(f"\n=== {trait} ===")
        out_path, url_used = try_download_trait(trait)
        if out_path is None:
            print(f"  [FAILED] could not find {trait}.json at any candidate URL")
            summary.append((trait, None))
            continue

        print(f"  downloaded -> {out_path}")
        summary.append((trait, url_used))

        # Print schema for inspection
        try:
            obj = json.loads(out_path.read_text(encoding="utf-8"))
            print(f"  schema:")
            describe_schema(obj)
        except Exception as e:
            print(f"  [WARN] downloaded file but couldn't parse for schema: {e}")

    # --- final summary ------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"{'trait':16s}  {'status':10s}  {'source URL'}")
    print("-" * 80)
    n_ok = 0
    for trait, url in summary:
        if url is None:
            print(f"{trait:16s}  {'FAILED':10s}  -")
        else:
            n_ok += 1
            print(f"{trait:16s}  {'ok':10s}  {url}")
    print("-" * 80)
    print(f"Downloaded {n_ok} of {len(TRAITS)} trait artifacts.")

    if n_ok == 0:
        print()
        print("All downloads failed. The repo's directory structure may have changed.")
        print("Check https://github.com/safety-research/persona_vectors/tree/main/data_generation")
        print("manually and update URL_PATTERNS in this script.")
        return 1

    if n_ok < len(TRAITS):
        print()
        print("Some traits failed. Inspect the failed traits manually.")
        return 1

    print()
    print("Inspect a downloaded file:")
    print(f"  cat data/persona_artifacts/evil.json | head -50")
    print()
    print("Once you've confirmed the schema, the next step is implementing")
    print("`extract_persona_vector` in src/extraction.py to consume these files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())