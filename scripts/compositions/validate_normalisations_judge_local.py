"""
PILOT — laptop judge stage for the normalisation A/B/C comparison (Phase 15).

Single purpose: walk the 6 pilot pairs; for each pair, judge the 6 CSVs
(baseline, single_a, single_b, joint_false, joint_true, joint_per_axis) left
by the cluster generate stage with NaN score columns. Fills trait_a / trait_b
/ coherence in place. Nothing else.

Post-judge tally + per-pair summary table is the job of
validate_normalisations_pilot.py in judge mode — run it after this script:

    COMPOSITION_PILOT_MODE=judge python -m scripts.compositions.validate_normalisations_pilot

Designed for laptop:
  - no GPU, no HF model load
  - needs OPENAI_API_KEY in .env and outbound internet
  - idempotent on row.trait_a.isna() — kill + rerun is safe

Data needed on laptop (rsync from cluster before running):
  results/composition_pilot_normalisations/Llama-3.1-8B-Instruct/*.csv   (36)
  data/composition_eval/*.json                                           (subset)

Run:
    python -m scripts.compositions.validate_normalisations_judge_local
"""
from __future__ import annotations

from dotenv import load_dotenv

from scripts.compositions.validate_normalisations_pilot import (
    JUDGE_MODEL,
    LOGS_DIR,
    PILOT_ALPHA,
    PILOT_MODES,
    PILOT_PAIRS,
    SCORES_OUTPUT_DIR,
    _baseline_csv_path,
    _csv_has_scores,
    _csv_status,
    _joint_csv_path,
    _judge_csv_inplace,
    _load_composition_artifact,
    _mode_tag,
    _single_csv_path,
)

load_dotenv()


def _per_pair_settings(a: str, b: str):
    """(label, csv_path, log_filename, judge_progress_tag) for all 6 settings.
    Filenames match what validate_normalisations_pilot.py wrote so judge-stage
    log lines accumulate in the same per-pair files left by the generate stage.
    """
    out = [
        (
            "baseline",
            _baseline_csv_path(a, b),
            f"pilot_{a}__{b}_baseline.log",
            f"{a}+{b} baseline",
        ),
        (
            "single_a",
            _single_csv_path(a, b, "a", PILOT_ALPHA),
            f"pilot_{a}__{b}_single_a_alpha{PILOT_ALPHA}.log",
            f"{a}+{b} single_a α={PILOT_ALPHA}",
        ),
        (
            "single_b",
            _single_csv_path(a, b, "b", PILOT_ALPHA),
            f"pilot_{a}__{b}_single_b_alpha{PILOT_ALPHA}.log",
            f"{a}+{b} single_b α={PILOT_ALPHA}",
        ),
    ]
    for nm in PILOT_MODES:
        tag = _mode_tag(nm)
        out.append((
            f"joint_{tag}",
            _joint_csv_path(a, b, nm, PILOT_ALPHA),
            f"pilot_{a}__{b}_joint_{tag}_alpha{PILOT_ALPHA}.log",
            f"{a}+{b} joint_{tag} α={PILOT_ALPHA}",
        ))
    return out


def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("LOCAL JUDGE STAGE — pilot normalisation A/B/C")
    print(f"  judge model : {JUDGE_MODEL}")
    print(f"  CSV dir     : {SCORES_OUTPUT_DIR}")
    print(f"  per-pair logs append to {LOGS_DIR}/pilot_<a>__<b>_*.log")
    print("  idempotent on row.trait_a.isna() — safe to kill + rerun")
    print("  for tally + per-pair summary run:")
    print("    COMPOSITION_PILOT_MODE=judge python -m scripts.compositions.validate_normalisations_pilot")
    print("=" * 72 + "\n")

    print(f"Judging {len(PILOT_PAIRS)} pilot pairs × {3 + len(PILOT_MODES)} settings each\n")

    for i, (a_in, b_in) in enumerate(PILOT_PAIRS, 1):
        a, b = sorted([a_in, b_in])
        print(f"\n[{i}/{len(PILOT_PAIRS)}] {a} + {b}")
        try:
            artifact = _load_composition_artifact(a, b)
        except FileNotFoundError as e:
            print(f"  skipping — {e}")
            continue

        for label, csv_path, log_name, tag in _per_pair_settings(a, b):
            print(f"  {label:<14} -> {csv_path.name}")
            if not csv_path.exists():
                print(f"    SKIP: CSV missing (generate stage not done on cluster)")
                continue
            _judge_csv_inplace(
                csv_path,
                artifact["eval_prompt_a"], artifact["eval_prompt_b"],
                progress_tag=tag,
                log_path=LOGS_DIR / log_name,
            )
            print(f"    {_csv_status(csv_path)}    log={log_name}")

    n_csvs = len(list(SCORES_OUTPUT_DIR.glob("*.csv")))
    n_scored = sum(1 for p in SCORES_OUTPUT_DIR.glob("*.csv") if _csv_has_scores(p))
    expected = len(PILOT_PAIRS) * (3 + len(PILOT_MODES))
    print()
    print("=" * 72)
    print("JUDGE STAGE TALLY")
    print(f"  CSVs on disk          : {n_csvs}  (expected {expected})")
    print(f"  CSVs with judge scores: {n_scored}  (target {expected})")
    print("=" * 72)
    print(
        "\nNext step: build per-pair summary table by running\n"
        "  COMPOSITION_PILOT_MODE=judge python -m scripts.compositions.validate_normalisations_pilot"
    )


if __name__ == "__main__":
    main()
