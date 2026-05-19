"""
LLM-judge calibration + composition-data audit for Phase 12.

Reproduces every check that produced the Phase 14 findings. No GPU, no API calls
— pure analysis on the existing CSVs under `results/composition_scoring_l17/`.

Run from repo root:
    venv/bin/python scripts/compositions/audit_judge_calibration.py

Sections (each can be commented out in main() if you only want a subset):

    1.  Per-trait baseline distributions
            Question: is the judge well-calibrated on un-steered Llama?
            Answer: yes for most traits. `confidence` and `formality` are RLHF-
            saturated (baselines 58 and 89). `power_seeking` has elevated and
            noisy baseline (mean 16, median 6, 10% > 50).

    2.  Off-target trait inflation under single steering
            Question: when we steer trait X, does the judge for trait Y (≠ X)
            inflate above baseline?
            Answer: yes, substantially, for `apathetic` (+21), `hallucinating`
            (+18), `impolite` (+15), `sycophantic` (+14), `power_seeking` (+12).
            `evil` is essentially immune (+0.5).

    3.  Per-trait condition × coherence sweep
            Question: does the inflation track coherence collapse?
            Answer: yes. Joint coherence drops to 37-50 for Tier-S traits and
            27-52% of joint responses have coh<30 — the same conditions where
            trait scores inflate.

    4.  Joint-condition score bucket × coherence distribution
            Question: where on the [0,100] axis do the joint scores actually
            land? Are mid-range scores associated with low coherence?
            Answer: distribution is mostly bimodal (22% in [0,10), 44% in
            [80,100)). Mid-range [50,60) is only 2% of all evaluations and
            does sit at lower coherence (mean 48 vs 71 for [0,10)).

    5.  Regime distribution under coherence filtering and power_seeking drop
            Question: how much of the regime mess is artefactual from
            coherence collapse + the broken power_seeking vector?
            Answer: under coh≥30 + drop power_seeking, 5 of 6 "emergent"
            classifications revert to mixed/dominant. Mixed pile drops from
            19/36 (53%) to 13/28 (46%).

    6.  Per-pair joint-score histograms (selected representative pairs)
            Question: are "mixed" pairs genuinely mid-range or
            bimodal-with-balanced-mean?
            Answer: mixed bag. Bimodal-axes (formality, evil, hallucinating,
            impolite, sycophantic) cluster at extremes; continuous-axes
            (confidence, humorous, apathetic) have substantial mid-range mass.

Sample dumps (written to analysis/audit_samples/):

    7.  mid_range_responses.md
            18 joint responses scored 50-60 on either trait, one per trait
            (lowest score first) + 10 random extras.

    8.  incoherent_high_trait_responses.md
            8 joint responses scored 70-90 on a trait with coh<25 (lowest coh
            first), one per affected trait. Confirms which judges
            mis-attribute to broken text.

    9.  power_seeking_single_responses.md
            5 random power_seeking single-steered responses on composition
            prompts. Confirms the vector is functionally inactive on this
            prompt set (0/800 score >50).
"""

from __future__ import annotations

import glob
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CSV_DIR = Path("results/composition_scoring_l17/Llama-3.1-8B-Instruct")
SAMPLES_DIR = Path("analysis/audit_samples")

# Regime thresholds — match composition_scoring.py _classify_regime
ADDITIVE_LO, ADDITIVE_HI = 0.7, 1.3
DOMINANT_LO, DOMINANT_HI = 0.7, 0.3
SUPPRESSIVE_MAX = 0.5
EMERGENT_MIN = 1.3
RATIO_DEN_MIN = 1.0  # if |Δ_single| < this, ratio is degenerate

RANDOM_SEED = 42
# ---------------------------------------------------------------------------


def _classify(ra: float, rb: float) -> str:
    if pd.isna(ra) or pd.isna(rb):
        return "undef"
    if ra >= EMERGENT_MIN and rb >= EMERGENT_MIN:
        return "emergent"
    if ra < SUPPRESSIVE_MAX and rb < SUPPRESSIVE_MAX:
        return "suppressive"
    if ADDITIVE_LO <= ra <= ADDITIVE_HI and ADDITIVE_LO <= rb <= ADDITIVE_HI:
        return "additive"
    one_preserved = (
        (ra >= DOMINANT_LO and rb <= DOMINANT_HI)
        or (rb >= DOMINANT_LO and ra <= DOMINANT_HI)
    )
    if one_preserved:
        return "dominant"
    return "mixed"


def _pair_csv(pair: str, kind: str) -> Path:
    return CSV_DIR / f"{pair}_{kind}.csv"


def _all_pairs() -> list[str]:
    return sorted(
        Path(p).stem.replace("_joint_alpha4.0", "")
        for p in glob.glob(str(CSV_DIR / "*_joint_alpha4.0.csv"))
    )


def _mean_filtered(df: pd.DataFrame, trait_col: str, coh_min: float = 0) -> float:
    m = df["coherence"] >= coh_min
    vals = df.loc[m, trait_col].dropna()
    return vals.mean() if len(vals) else float("nan")


# ---------------------------------------------------------------------------
# Section 1 — baseline distributions
# ---------------------------------------------------------------------------
def check_baselines() -> None:
    print("=" * 90)
    print("SECTION 1 — Per-trait baseline score distributions (no steering)")
    print("=" * 90)
    per_trait = defaultdict(list)
    for pair in _all_pairs():
        a, b = pair.split("__")
        df = pd.read_csv(_pair_csv(pair, "baseline"))
        per_trait[a].extend(df["trait_a"].dropna().tolist())
        per_trait[b].extend(df["trait_b"].dropna().tolist())

    print(f"\n{'trait':<16} {'n':>5} {'mean':>7} {'median':>7} {'std':>6} "
          f"{'p25':>6} {'p75':>6} {'<10':>5} {'>50':>5}")
    print("-" * 80)
    for trait in sorted(per_trait):
        v = per_trait[trait]
        v_sorted = sorted(v)
        print(
            f"{trait:<16} {len(v):>5} {statistics.mean(v):>7.2f} "
            f"{statistics.median(v):>7.2f} {statistics.stdev(v):>6.2f} "
            f"{v_sorted[int(len(v)*0.25)]:>6.1f} {v_sorted[int(len(v)*0.75)]:>6.1f} "
            f"{sum(1 for x in v if x < 10) / len(v) * 100:>4.0f}% "
            f"{sum(1 for x in v if x > 50) / len(v) * 100:>4.0f}%"
        )


# ---------------------------------------------------------------------------
# Section 2 — off-target inflation
# ---------------------------------------------------------------------------
def check_off_target_inflation() -> None:
    print("\n" + "=" * 90)
    print("SECTION 2 — Off-target trait inflation under single steering")
    print("=" * 90)
    print("'off_target' = trait X scored when we steered something else.")
    print("If similar to baseline, judge is unconfused. If inflated, judge is misled.")

    baseline = defaultdict(list)
    off_target = defaultdict(list)

    for pair in _all_pairs():
        a, b = pair.split("__")
        # baseline
        df = pd.read_csv(_pair_csv(pair, "baseline"))
        baseline[a].extend(df["trait_a"].dropna().tolist())
        baseline[b].extend(df["trait_b"].dropna().tolist())
        # single_a → trait_b is off-target for trait_b
        df = pd.read_csv(_pair_csv(pair, "single_a_alpha4.0"))
        off_target[b].extend(df["trait_b"].dropna().tolist())
        # single_b → trait_a is off-target for trait_a
        df = pd.read_csv(_pair_csv(pair, "single_b_alpha4.0"))
        off_target[a].extend(df["trait_a"].dropna().tolist())

    print(f"\n{'trait':<16} {'base_mean':>9} {'off_mean':>9} {'inflation':>10} "
          f"{'off_p75':>8} {'off_>50%':>9}")
    print("-" * 80)
    for trait in sorted(baseline):
        b_mean = statistics.mean(baseline[trait])
        o_mean = statistics.mean(off_target[trait])
        o_sorted = sorted(off_target[trait])
        o_p75 = o_sorted[int(len(o_sorted) * 0.75)]
        o_pct50 = sum(1 for x in off_target[trait] if x > 50) / len(off_target[trait]) * 100
        print(
            f"{trait:<16} {b_mean:>9.2f} {o_mean:>9.2f} {o_mean - b_mean:>+10.2f} "
            f"{o_p75:>8.1f} {o_pct50:>8.0f}%"
        )


# ---------------------------------------------------------------------------
# Section 3 — per-trait condition × coherence sweep
# ---------------------------------------------------------------------------
def check_coherence_by_condition() -> None:
    print("\n" + "=" * 90)
    print("SECTION 3 — Per-trait, per-condition trait & coherence means")
    print("=" * 90)

    conds: dict[str, dict[str, dict[str, list]]] = {
        c: defaultdict(lambda: {"trait": [], "coh": []})
        for c in ("baseline", "off_target", "on_target", "joint")
    }

    def add(df, trait_col, trait_name, cond):
        valid = df[trait_col].notna()
        conds[cond][trait_name]["trait"].extend(df.loc[valid, trait_col].tolist())
        conds[cond][trait_name]["coh"].extend(df.loc[valid, "coherence"].tolist())

    for pair in _all_pairs():
        a, b = pair.split("__")
        # baseline
        df = pd.read_csv(_pair_csv(pair, "baseline"))
        add(df, "trait_a", a, "baseline")
        add(df, "trait_b", b, "baseline")
        # single_a: trait_a is on-target, trait_b is off-target
        df = pd.read_csv(_pair_csv(pair, "single_a_alpha4.0"))
        add(df, "trait_a", a, "on_target")
        add(df, "trait_b", b, "off_target")
        # single_b: trait_b is on-target, trait_a is off-target
        df = pd.read_csv(_pair_csv(pair, "single_b_alpha4.0"))
        add(df, "trait_a", a, "off_target")
        add(df, "trait_b", b, "on_target")
        # joint
        df = pd.read_csv(_pair_csv(pair, "joint_alpha4.0"))
        add(df, "trait_a", a, "joint")
        add(df, "trait_b", b, "joint")

    print(f"\n{'trait':<14}  {'condition':<12} {'n':>5} {'tr_mean':>8} "
          f"{'tr_p75':>7} {'coh_mean':>9} {'coh<30%':>8}")
    print("-" * 95)
    for trait in sorted(conds["baseline"]):
        for cond in ("baseline", "off_target", "on_target", "joint"):
            t = conds[cond][trait]["trait"]
            c = conds[cond][trait]["coh"]
            t_sorted = sorted(t)
            print(
                f"{trait:<14}  {cond:<12} {len(t):>5} "
                f"{statistics.mean(t):>8.2f} "
                f"{t_sorted[int(len(t) * 0.75)]:>7.1f} "
                f"{statistics.mean(c):>9.2f} "
                f"{sum(1 for x in c if x < 30) / len(c) * 100:>7.0f}%"
            )
        print()


# ---------------------------------------------------------------------------
# Section 4 — joint-condition score buckets + coherence
# ---------------------------------------------------------------------------
def check_score_buckets() -> None:
    print("\n" + "=" * 90)
    print("SECTION 4 — Joint score bucket distribution + coherence per bucket")
    print("=" * 90)

    all_joint = []
    for pair in _all_pairs():
        a, b = pair.split("__")
        df = pd.read_csv(_pair_csv(pair, "joint_alpha4.0"))
        for _, row in df.iterrows():
            for col in ("trait_a", "trait_b"):
                if pd.notna(row[col]) and pd.notna(row["coherence"]):
                    all_joint.append({"score": row[col], "coh": row["coherence"]})

    print(f"\nTotal joint trait-evals: {len(all_joint)}")
    print(f"\n{'score range':<14} {'n':>5} {'%':>5} {'coh_mean':>9} {'coh<30%':>8}")
    print("-" * 55)
    for lo, hi in [(0, 10), (10, 30), (30, 50), (50, 60), (60, 80), (80, 100), (100, 101)]:
        bucket = [r for r in all_joint if lo <= r["score"] < hi]
        if not bucket:
            continue
        cohs = [r["coh"] for r in bucket]
        print(
            f"[{lo:3d}, {hi:3d}) {' ':<3} {len(bucket):>5} "
            f"{len(bucket) / len(all_joint) * 100:>4.0f}% "
            f"{statistics.mean(cohs):>9.2f} "
            f"{sum(1 for c in cohs if c < 30) / len(cohs) * 100:>7.0f}%"
        )


# ---------------------------------------------------------------------------
# Section 5 — regime under coherence filters + drop power_seeking
# ---------------------------------------------------------------------------
def check_regime_under_filters() -> None:
    print("\n" + "=" * 90)
    print("SECTION 5 — Regime distribution under coh filters and power_seeking drop")
    print("=" * 90)

    rows = {label: [] for label in
            ("no_filter", "coh30", "coh50", "drop_ps",
             "drop_ps_coh30", "drop_ps_coh50")}

    for pair in _all_pairs():
        a, b = pair.split("__")
        base = pd.read_csv(_pair_csv(pair, "baseline"))
        sa = pd.read_csv(_pair_csv(pair, "single_a_alpha4.0"))
        sb = pd.read_csv(_pair_csv(pair, "single_b_alpha4.0"))
        jt = pd.read_csv(_pair_csv(pair, "joint_alpha4.0"))

        for label, coh_min in (("no_filter", 0), ("coh30", 30), ("coh50", 50)):
            d_jt_a = _mean_filtered(jt, "trait_a", coh_min) - _mean_filtered(base, "trait_a", coh_min)
            d_jt_b = _mean_filtered(jt, "trait_b", coh_min) - _mean_filtered(base, "trait_b", coh_min)
            d_st_a = _mean_filtered(sa, "trait_a", coh_min) - _mean_filtered(base, "trait_a", coh_min)
            d_st_b = _mean_filtered(sb, "trait_b", coh_min) - _mean_filtered(base, "trait_b", coh_min)
            ra = d_jt_a / d_st_a if abs(d_st_a) > RATIO_DEN_MIN else float("nan")
            rb = d_jt_b / d_st_b if abs(d_st_b) > RATIO_DEN_MIN else float("nan")
            regime = _classify(ra, rb)
            entry = {"pair": pair, "ra": ra, "rb": rb, "regime": regime}
            rows[label].append(entry)
            if "power_seeking" not in pair:
                key = label.replace("no_filter", "drop_ps").replace("coh", "drop_ps_coh")
                if label == "no_filter":
                    rows["drop_ps"].append(entry)
                else:
                    rows[f"drop_ps_{label}"].append(entry)

    print(f"\n{'config':<24} {'n_pairs':>7}  regime distribution")
    print("-" * 90)
    for label in ("no_filter", "coh30", "coh50", "drop_ps", "drop_ps_coh30", "drop_ps_coh50"):
        ct = Counter(r["regime"] for r in rows[label])
        dist = "  ".join(f"{k}:{v}" for k, v in sorted(ct.items(), key=lambda x: -x[1]))
        print(f"{label:<24} {len(rows[label]):>7}   {dist}")

    print("\n\nPer-pair regime change (no_filter → drop_ps + coh30):")
    nf = {r["pair"]: r for r in rows["no_filter"]}
    filt = {r["pair"]: r for r in rows["drop_ps_coh30"]}
    changed = 0
    for pair, r in nf.items():
        f = filt.get(pair)
        if f is None:
            print(f"  {pair:<36} {r['regime']:<11} → (dropped)")
        elif f["regime"] != r["regime"]:
            changed += 1
            print(f"  {pair:<36} {r['regime']:<11} → {f['regime']:<11}  "
                  f"(ra {r['ra']:+.2f}→{f['ra']:+.2f}, rb {r['rb']:+.2f}→{f['rb']:+.2f})")
    print(f"\n{changed} pairs changed regime under coh≥30 + drop power_seeking.")


# ---------------------------------------------------------------------------
# Section 6 — per-pair joint-score histograms
# ---------------------------------------------------------------------------
def check_per_pair_distributions() -> None:
    print("\n" + "=" * 90)
    print("SECTION 6 — Per-pair joint-score histograms (5 bins)")
    print("=" * 90)
    print("High mass at [0,20)+[80,100) = bimodal-at-extremes.")
    print("Substantial mass in middle bins = continuous distribution.")

    import json
    with open("results/composition_scoring_l17_summary.json") as f:
        summary = json.load(f)
    pairs_by_regime = defaultdict(list)
    for p in summary["pairs"]:
        if p["status"] == "ok":
            pairs_by_regime[p["regime"]].append(p)

    target = {"mixed": 5, "additive": 3, "dominant": 3, "suppressive": 3, "emergent": 3}
    picked = []
    for regime, n in target.items():
        picked.extend(pairs_by_regime.get(regime, [])[:n])

    print(f"\n{'pair':<32} {'regime':<11} {'axis':<14} "
          f"{' 0-20%':>6} {'20-40%':>6} {'40-60%':>6} {'60-80%':>6} {'80-100%':>7}  mean")
    print("-" * 120)
    bins = [0, 20, 40, 60, 80, 100.1]
    for p in picked:
        a, b = p["trait_a"], p["trait_b"]
        df = pd.read_csv(_pair_csv(f"{a}__{b}", "joint_alpha4.0"))
        for col, name in (("trait_a", a), ("trait_b", b)):
            v = df[col].dropna().tolist()
            counts = [sum(1 for x in v if bins[i] <= x < bins[i + 1]) for i in range(5)]
            pcts = [c / len(v) * 100 for c in counts]
            print(
                f"{a + '+' + b:<32} {p['regime']:<11} {name:<14} "
                f"{pcts[0]:>5.0f}% {pcts[1]:>5.0f}% {pcts[2]:>5.0f}% {pcts[3]:>5.0f}% "
                f"{pcts[4]:>6.0f}%   {statistics.mean(v):.1f}"
            )
        print()


# ---------------------------------------------------------------------------
# Sample dumps (sections 7-9)
# ---------------------------------------------------------------------------
def dump_mid_range_samples() -> None:
    random.seed(RANDOM_SEED)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    out = SAMPLES_DIR / "mid_range_responses.md"

    rows = []
    for pair in _all_pairs():
        a, b = pair.split("__")
        df = pd.read_csv(_pair_csv(pair, "joint_alpha4.0"))
        for col, name in (("trait_a", a), ("trait_b", b)):
            other_col = "trait_b" if col == "trait_a" else "trait_a"
            other_trait = b if col == "trait_a" else a
            m = (df[col] >= 50) & (df[col] <= 60)
            for _, r in df[m].iterrows():
                rows.append({
                    "pair": pair, "trait": name, "trait_col": col,
                    "score": r[col], "other_trait": other_trait,
                    "other_score": r[other_col], "coh": r["coherence"],
                    "q": r["question"], "a": r["answer"],
                })

    # one per trait (lowest score first) + 10 random extras
    sample = []
    seen = set()
    for r in sorted(rows, key=lambda x: (x["trait"], x["score"])):
        if r["trait"] not in seen:
            sample.append(r)
            seen.add(r["trait"])
    random.shuffle(rows)
    seen_keys = {(r["pair"], r["q"][:50], r["trait_col"]) for r in sample}
    for r in rows:
        k = (r["pair"], r["q"][:50], r["trait_col"])
        if k in seen_keys:
            continue
        sample.append(r)
        seen_keys.add(k)
        if len(sample) >= 18:
            break

    with out.open("w") as f:
        f.write("# Mid-range (50-60) joint-steering scored responses\n\n")
        f.write("18 stratified samples: one per trait (lowest score first), then 10 random extras.\n")
        f.write(f"Produced by `scripts/compositions/audit_judge_calibration.py`.\n\n")
        for i, r in enumerate(sample, 1):
            f.write(f"## #{i} — pair `{r['pair']}` | evaluating **{r['trait']}**\n\n")
            f.write(f"- `{r['trait_col']}` score: **{r['score']:.2f}**\n")
            f.write(f"- other trait (`{r['other_trait']}`) score: {r['other_score']:.2f}\n")
            f.write(f"- coherence: {r['coh']:.2f}\n\n")
            f.write(f"**Q:** {r['q']}\n\n")
            f.write(f"**A:**\n```\n{r['a'][:1500]}"
                    f"{'...[truncated]' if len(r['a']) > 1500 else ''}\n```\n\n---\n\n")
    print(f"wrote {out}  ({len(sample)} samples)")


def dump_incoherent_high_samples() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    out = SAMPLES_DIR / "incoherent_high_trait_responses.md"

    rows = []
    for pair in _all_pairs():
        a, b = pair.split("__")
        df = pd.read_csv(_pair_csv(pair, "joint_alpha4.0"))
        for col, name in (("trait_a", a), ("trait_b", b)):
            other_col = "trait_b" if col == "trait_a" else "trait_a"
            m = (df[col] >= 70) & (df[col] <= 90) & (df["coherence"] < 25)
            for _, r in df[m].iterrows():
                rows.append({
                    "pair": pair, "trait": name, "score": r[col],
                    "other_score": r[other_col], "coh": r["coherence"],
                    "q": r["question"], "a": r["answer"],
                })

    # one per trait, lowest coh first
    sample, seen = [], set()
    for r in sorted(rows, key=lambda x: x["coh"]):
        if r["trait"] not in seen:
            sample.append(r)
            seen.add(r["trait"])

    with out.open("w") as f:
        f.write("# Joint responses scored 70-90 on a trait, with coherence < 25\n\n")
        f.write(f"Total population: {len(rows)} responses across 36 pairs.\n")
        f.write("One sample per affected trait, lowest coherence first.\n\n")
        f.write("If the judge is right, these should be coherent trait-expressing text.\n")
        f.write("If the judge is wrong, they should be broken text with trait-flavored fragments.\n\n")
        f.write(f"Produced by `scripts/compositions/audit_judge_calibration.py`.\n\n")
        for i, r in enumerate(sample, 1):
            f.write(f"## #{i} — `{r['pair']}` — judge says **{r['trait']}={r['score']:.1f}** "
                    f"but coh={r['coh']:.2f}\n\n")
            f.write(f"other trait score: {r['other_score']:.1f}\n\n")
            f.write(f"**Q:** {r['q']}\n\n")
            f.write(f"**A:**\n```\n{r['a'][:1200]}"
                    f"{'...' if len(r['a']) > 1200 else ''}\n```\n\n---\n\n")
    print(f"wrote {out}  ({len(sample)} samples from {len(rows)} candidates)")


def dump_power_seeking_samples() -> None:
    random.seed(13)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    out = SAMPLES_DIR / "power_seeking_single_responses.md"

    rows = []
    # collect all responses where we steered power_seeking
    for csv_path in glob.glob(str(CSV_DIR / "power_seeking__*_single_a_alpha4.0.csv")):
        pair = Path(csv_path).stem.replace("_single_a_alpha4.0", "")
        df = pd.read_csv(csv_path)
        for _, r in df.iterrows():
            rows.append({"pair": pair, "score": r["trait_a"], "coh": r["coherence"],
                         "q": r["question"], "a": r["answer"]})
    for csv_path in glob.glob(str(CSV_DIR / "*__power_seeking_single_b_alpha4.0.csv")):
        pair = Path(csv_path).stem.replace("_single_b_alpha4.0", "")
        df = pd.read_csv(csv_path)
        for _, r in df.iterrows():
            rows.append({"pair": pair, "score": r["trait_b"], "coh": r["coherence"],
                         "q": r["question"], "a": r["answer"]})

    random.shuffle(rows)
    sample = rows[:5]

    with out.open("w") as f:
        f.write("# `power_seeking` SINGLE-STEERED responses on composition prompts\n\n")
        f.write(f"5 random samples from {len(rows)} total responses where we applied "
                f"power_seeking steering alone at α_unit=4.\n\n")
        f.write(f"**Aggregate stats:** mean ps score {statistics.mean(r['score'] for r in rows):.2f}, "
                f"mean coh {statistics.mean(r['coh'] for r in rows):.2f}, "
                f"**0/{len(rows)} responses score >50 on power_seeking judge.**\n\n")
        f.write("These responses should be POWER-SEEKING. They are not — they are normal "
                "collaborative-assistant advice. The vector works on power_seeking-specific "
                "eval prompts (E10.3 Δ_trait = +59) but produces no measurable effect on the "
                "neutral composition prompts.\n\n")
        f.write(f"Produced by `scripts/compositions/audit_judge_calibration.py`.\n\n")
        for i, r in enumerate(sample, 1):
            f.write(f"## #{i} — `{r['pair']}` — ps judge: **{r['score']:.1f}** coh={r['coh']:.2f}\n\n")
            f.write(f"**Q:** {r['q']}\n\n")
            f.write(f"**A:**\n```\n{r['a'][:1200]}"
                    f"{'...' if len(r['a']) > 1200 else ''}\n```\n\n---\n\n")
    print(f"wrote {out}  ({len(sample)} samples from {len(rows)} candidates)")


def main() -> None:
    check_baselines()
    check_off_target_inflation()
    check_coherence_by_condition()
    check_score_buckets()
    check_regime_under_filters()
    check_per_pair_distributions()
    print("\n" + "=" * 90)
    print("Dumping sample markdown files to analysis/audit_samples/ …")
    print("=" * 90)
    dump_mid_range_samples()
    dump_incoherent_high_samples()
    dump_power_seeking_samples()
    print("\nDone.")


if __name__ == "__main__":
    main()
