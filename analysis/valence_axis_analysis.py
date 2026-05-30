#!/usr/bin/env python
"""Valence-axis analysis for the normalised-sum (norm) composition dataset.

Question (RQ1 follow-up): how much of the pairwise cosine structure among the
norm-dataset steering vectors is explained by a single shared *valence /
antisocial* direction?

Pipeline
--------
1.  Load the 8 norm-dataset steering vectors at the operating point
    (response_avg_diff[L=17], unit-normalised — the exact vectors the
    composition sweep injected; see src/composition/joint_injection.py).
2.  Sanity-check: recomputed pairwise cosines must match the cosines stored in
    the norm dataset summary.json (proves we loaded the right vectors).
3.  Derive a valence axis directly from the vectors:
        v_val  =  normalize( mean(negative-trait vectors) - mean(positive) )
4.  Project the valence axis out of every steering vector, renormalise, and
    recompute the pairwise ("residual") cosine for all 28 norm pairs.
5.  Compare original vs residual cosine geometry (signed mean, mean |cos|,
    per-pair shrink), and test whether PC1 of the steering-vector set is
    approximately the valence axis (|cos(PC_i, v_val)| per principal component),
    with an exact label-permutation null on the PC1 alignment.

Two valence labellings are run side by side to support the narrative around the
"hallucinating" trait (see CONFIGS below): the vectors show hallucinating is
near-orthogonal to the impolite/apathetic antisocial cluster, so including vs
excluding it materially sharpens the axis. Both are reported and compared.

Outputs (under analysis/valence_axis/):
    residual_cosine_<config>.csv   per-pair original vs residual cosine + regime
    summary.json                   shared geometry + per-config results + compare
    valence_axis_<config>.png      per-config diagnostic figure
    comparison.png                 narrative comparison across configs

Run from anywhere:
    venv/bin/python analysis/valence_axis_analysis.py
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.composition.joint_injection import load_unit_vector  # noqa: E402

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
LAYER = 17
NORM_SUMMARY = REPO_ROOT / "results/composition/v2_phase125_normTrue_a4.5/scoring/summary.json"
OUT_DIR = REPO_ROOT / "analysis/valence_axis"


@dataclass(frozen=True)
class VConfig:
    """A valence labelling. axis = normalize(mean(negative) - mean(positive)).

    Any trait left out of both lists is excluded from the axis but still gets
    projected and scored (so its cos with the axis is informative).
    """

    name: str
    title: str
    negative: tuple[str, ...]
    positive: tuple[str, ...]


# Both alternatives are kept here on purpose, to tell the narrative rather than
# pick one. `antisocial5` takes the broad "anything undesirable is negative"
# reading; `core4` drops hallucinating, which the vectors reveal is an
# epistemic failure mode (near-orthogonal to impolite/apathetic, only weakly
# correlated with evil/sycophantic) rather than a social-valence trait.
CONFIGS: list[VConfig] = [
    VConfig(
        name="antisocial5",
        title="Antisocial-5 (hallucinating included)",
        negative=("evil", "impolite", "apathetic", "sycophantic", "hallucinating"),
        positive=("confidence", "humorous", "formality"),
    ),
    VConfig(
        name="core4",
        title="Core-4 (hallucinating excluded as neutral failure mode)",
        negative=("evil", "impolite", "apathetic", "sycophantic"),
        positive=("confidence", "humorous", "formality"),
    ),
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero-norm vector")
    return v / n


def project_out(v: np.ndarray, axis_unit: np.ndarray) -> np.ndarray:
    """Remove the component of `v` along the unit `axis_unit`."""
    return v - float(v @ axis_unit) * axis_unit


def load_steering_matrix(traits: list[str]) -> dict[str, np.ndarray]:
    """trait -> unit response_avg_diff[LAYER] vector (numpy float64)."""
    return {t: load_unit_vector(t, LAYER).numpy().astype(np.float64) for t in traits}


def valence_axis(V: dict[str, np.ndarray], neg, pos) -> np.ndarray:
    """normalize( mean(neg vectors) - mean(pos vectors) )."""
    return unit(np.mean([V[t] for t in neg], axis=0) - np.mean([V[t] for t in pos], axis=0))


def permutation_null_pc1(V, traits, pc1, n_neg, n_pos, observed,
                         max_assignments: int = 200_000, seed: int = 0):
    """Label-permutation null for the statistic |cos(PC1, valence axis)|.

    Under H0 the negative/positive labels carry no geometric structure: every
    way of tagging `n_neg` traits "negative" and `n_pos` "positive" (the rest
    excluded) is equally likely. PC1 is a fixed property of the vector set, so
    for each labelling we only rebuild the axis and remeasure |cos(PC1, axis)|.
    The real semantic split is one of those labellings.

    Few traits -> the labelling space is tiny (e.g. 8 choose 5 = 56), so we
    enumerate it exactly; only fall back to random sampling if it explodes.

    Returns (null_stats, p_value, exact, n_labellings). The p-value is the
    fraction of labellings at least as aligned as the observed one (it includes
    the observed labelling, so it is never 0).
    """
    from math import comb

    total = comb(len(traits), n_neg) * comb(len(traits) - n_neg, n_pos)
    exact = total <= max_assignments
    stats = []
    if exact:
        for neg in combinations(traits, n_neg):
            rem = [t for t in traits if t not in neg]
            for pos in combinations(rem, n_pos):
                stats.append(abs(float(pc1 @ valence_axis(V, neg, pos))))
    else:
        rng = np.random.default_rng(seed)
        for _ in range(max_assignments):
            perm = list(rng.permutation(traits))
            neg, pos = perm[:n_neg], perm[n_neg:n_neg + n_pos]
            stats.append(abs(float(pc1 @ valence_axis(V, neg, pos))))
    stats = np.asarray(stats)
    p = float((stats >= observed - 1e-12).mean())
    return stats, p, exact, len(stats)


def make_verdict(pc_align: np.ndarray, evr: np.ndarray, best_pc: int) -> str:
    if best_pc == 0:
        if pc_align[0] >= 0.9:
            return f"YES — PC1 is approximately the valence axis (|cos|={pc_align[0]:.3f})"
        if pc_align[0] >= 0.7:
            return f"PARTIAL — PC1 is the closest PC and aligns moderately (|cos|={pc_align[0]:.3f})"
        return (f"WEAK — PC1 is the closest PC but only loosely aligned with valence "
                f"(|cos|={pc_align[0]:.3f}, PC1 explains {evr[0]:.0%} of variance)")
    return (f"NO — valence axis best matches PC{best_pc+1} (|cos|={pc_align[best_pc]:.3f}), "
            f"not PC1 (PC1 |cos|={pc_align[0]:.3f})")


# --------------------------------------------------------------------------- #
# Shared geometry (independent of the valence labelling)
# --------------------------------------------------------------------------- #
def build_context() -> dict:
    summary = json.loads(NORM_SUMMARY.read_text())
    traits = list(summary["traits"])
    pairs_meta = {
        tuple(sorted((p["trait_a"], p["trait_b"]))): p
        for p in summary["pairs"]
        if p.get("status") == "ok"
    }

    V = load_steering_matrix(traits)

    # Sanity: recomputed pairwise cosines must match the stored ones.
    max_diff = 0.0
    for (a, b), meta in pairs_meta.items():
        max_diff = max(max_diff, abs(float(unit(V[a]) @ unit(V[b])) - meta["cos"]))

    # PCA of the steering-vector set — does NOT depend on the labelling.
    U = np.stack([unit(V[t]) for t in traits])      # [8, 4096] unit rows
    mean_dir = unit(U.mean(axis=0))                 # shared/centroid direction
    Uc = U - U.mean(axis=0)                         # centred for PCA
    _, S, Vt = np.linalg.svd(Uc, full_matrices=False)
    evr = (S ** 2) / (S ** 2).sum()

    return dict(summary=summary, traits=traits, pairs_meta=pairs_meta, V=V,
                Uc=Uc, mean_dir=mean_dir, Vt=Vt, evr=evr, max_diff=max_diff)


# --------------------------------------------------------------------------- #
# Per-config analysis
# --------------------------------------------------------------------------- #
def analyze_config(cfg: VConfig, ctx: dict) -> dict:
    traits, V = ctx["traits"], ctx["V"]
    pairs_meta, Vt, evr, Uc = ctx["pairs_meta"], ctx["Vt"], ctx["evr"], ctx["Uc"]
    mean_dir = ctx["mean_dir"]

    labelled = set(cfg.negative) | set(cfg.positive)
    unknown = labelled - set(traits)
    if unknown:
        raise ValueError(f"[{cfg.name}] labelled traits not in dataset: {sorted(unknown)}")
    excluded = sorted(set(traits) - labelled)

    v_val = valence_axis(V, cfg.negative, cfg.positive)
    raw_norm = float(np.linalg.norm(
        np.mean([V[t] for t in cfg.negative], axis=0)
        - np.mean([V[t] for t in cfg.positive], axis=0)))
    proj = {t: float(unit(V[t]) @ v_val) for t in traits}

    # Residual cosine: project the valence axis out of every vector.
    V_res = {t: unit(project_out(V[t], v_val)) for t in traits}
    rows = []
    for a, b in combinations(traits, 2):
        key = tuple(sorted((a, b)))
        cos_orig = float(unit(V[a]) @ unit(V[b]))
        cos_res = float(V_res[a] @ V_res[b])
        rows.append({
            "trait_a": key[0], "trait_b": key[1],
            "regime": pairs_meta.get(key, {}).get("regime", ""),
            "cos_orig": cos_orig, "cos_residual": cos_res,
            "abs_cos_orig": abs(cos_orig), "abs_cos_residual": abs(cos_res),
            "delta_abs_cos": abs(cos_res) - abs(cos_orig),
            "both_negative": a in cfg.negative and b in cfg.negative,
        })
    co = np.array([r["cos_orig"] for r in rows])
    cr = np.array([r["cos_residual"] for r in rows])
    neg_mask = np.array([r["both_negative"] for r in rows])

    # PCA alignment with this config's axis.
    pc_align = np.abs(Vt @ v_val)
    best_pc = int(np.argmax(pc_align))
    val_var_frac = float(((Uc @ v_val) ** 2).sum() / (Uc ** 2).sum())
    verdict = make_verdict(pc_align, evr, best_pc)

    null_stats, p_perm, exact_null, n_lab = permutation_null_pc1(
        V, traits, Vt[0], len(cfg.negative), len(cfg.positive), pc_align[0])

    return dict(
        cfg=cfg, excluded=excluded, v_val=v_val, raw_norm=raw_norm, proj=proj,
        rows=rows, co=co, cr=cr, neg_mask=neg_mask,
        pc_align=pc_align, best_pc=best_pc, val_var_frac=val_var_frac,
        cos_centroid=float(mean_dir @ v_val), verdict=verdict,
        null_stats=null_stats, p_perm=p_perm, exact_null=exact_null, n_lab=n_lab,
    )


def print_config_report(res: dict, evr: np.ndarray) -> None:
    cfg = res["cfg"]
    proj, co, cr, neg_mask = res["proj"], res["co"], res["cr"], res["neg_mask"]
    pc_align = res["pc_align"]

    print("\n" + "=" * 72)
    print(f"CONFIG: {cfg.name} — {cfg.title}")
    print("=" * 72)
    print(f"  negative ({len(cfg.negative)}): {list(cfg.negative)}")
    print(f"  positive ({len(cfg.positive)}): {list(cfg.positive)}")
    if res["excluded"]:
        print(f"  excluded from axis (still scored): {res['excluded']}")

    print(f"\n  valence axis  ||mean(neg)-mean(pos)|| = {res['raw_norm']:.4f}")
    print("  cos(trait, v_val)   (>0 = antisocial side, sorted):")
    for t, c in sorted(proj.items(), key=lambda kv: -kv[1]):
        tag = "neg" if t in cfg.negative else ("pos" if t in cfg.positive else "exc")
        print(f"    {t:14s} [{tag}] {c:+.3f} {'#' * int(round(abs(c) * 30))}")

    print(f"\n  RESIDUAL COSINE (all {len(res['rows'])} pairs):")
    print(f"    mean signed cos : {co.mean():+.4f} -> {cr.mean():+.4f}   "
          f"(Δ {cr.mean()-co.mean():+.4f})")
    print(f"    mean |cos|      : {np.abs(co).mean():.4f} -> {np.abs(cr).mean():.4f}   "
          f"(Δ {np.abs(cr).mean()-np.abs(co).mean():+.4f})")
    if neg_mask.any():
        print(f"    neg-neg pairs ({int(neg_mask.sum())}): "
              f"signed {co[neg_mask].mean():+.4f} -> {cr[neg_mask].mean():+.4f}   "
              f"|cos| {np.abs(co[neg_mask]).mean():.4f} -> {np.abs(cr[neg_mask]).mean():.4f}")

    print("\n  PCA vs valence axis (PC explained-var is config-independent):")
    print(f"    {'PC':>3s}  {'expl.var':>9s}  {'|cos(PC, v_val)|':>16s}")
    for i in range(len(evr)):
        star = "  <-- best" if i == res["best_pc"] else ("  (PC1)" if i == 0 else "")
        print(f"    {i+1:>3d}  {evr[i]:>8.1%}  {pc_align[i]:>16.3f}{star}")
    print(f"    variance along v_val (centred): {res['val_var_frac']:.1%} of total")
    print(f"    VERDICT: {res['verdict']}")

    print(f"\n  PERMUTATION NULL for |cos(PC1, v_val)| "
          f"({'exact' if res['exact_null'] else 'sampled'}, N={res['n_lab']}):")
    ns = res["null_stats"]
    print(f"    observed {pc_align[0]:.3f}   null {ns.mean():.3f}±{ns.std():.3f}   "
          f"p95 {np.percentile(ns, 95):.3f}   p={res['p_perm']:.3f}")
    print("    -> " + ("ABOVE chance" if res["p_perm"] < 0.05 else
                        "NOT distinguishable from chance"))


# --------------------------------------------------------------------------- #
# Narrative comparison
# --------------------------------------------------------------------------- #
def print_comparison(results: list[dict], ctx: dict) -> None:
    print("\n" + "=" * 72)
    print("NARRATIVE COMPARISON")
    print("=" * 72)
    hl = "hallucinating"
    print(f"  Key trait under test: '{hl}' — cos with each config's axis:")
    for res in results:
        in_neg = hl in res["cfg"].negative
        where = "in negative set" if in_neg else "excluded"
        print(f"    {res['cfg'].name:12s} {res['proj'][hl]:+.3f}   ({where})")

    hdr = f"  {'metric':<34s}" + "".join(f"{r['cfg'].name:>14s}" for r in results)
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))

    def line(label, fn, fmt="{:>14.3f}"):
        print(f"  {label:<34s}" + "".join(fmt.format(fn(r)) for r in results))

    line("valence axis raw norm", lambda r: r["raw_norm"])
    line(f"cos({hl}, axis)", lambda r: r["proj"][hl])
    line("neg-neg signed cos (orig)", lambda r: r["co"][r["neg_mask"]].mean())
    line("neg-neg signed cos (residual)", lambda r: r["cr"][r["neg_mask"]].mean())
    line("PC1 |cos(PC1, v_val)|", lambda r: r["pc_align"][0])
    line("valence variance fraction", lambda r: r["val_var_frac"])
    line("permutation p (PC1 align)", lambda r: r["p_perm"])

    print("\n  Reading: dropping the near-orthogonal trait sharpens the axis "
          "(antisocial\n  traits load harder), tightens the neg-neg cluster, and "
          "raises PC1 alignment —\n  but with only 8 vectors the permutation null is "
          "too wide to reach significance.")


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def persist(results: list[dict], ctx: dict) -> list[Path]:
    written = []
    for res in results:
        csv_path = OUT_DIR / f"residual_cosine_{res['cfg'].name}.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(res["rows"][0].keys()))
            w.writeheader()
            w.writerows(res["rows"])
        written.append(csv_path)

    summary_out = {
        "layer": LAYER,
        "extraction": "response_avg_diff",
        "traits": ctx["traits"],
        "sanity_max_cos_diff": ctx["max_diff"],
        "shared_pca": {
            "explained_variance_ratio": ctx["evr"].tolist(),
        },
        "configs": [_config_summary(res) for res in results],
    }
    sj = OUT_DIR / "summary.json"
    sj.write_text(json.dumps(summary_out, indent=2))
    written.append(sj)
    return written


def _config_summary(res: dict) -> dict:
    cfg, co, cr, nm, ns = (res["cfg"], res["co"], res["cr"],
                           res["neg_mask"], res["null_stats"])
    return {
        "name": cfg.name,
        "title": cfg.title,
        "negative_traits": list(cfg.negative),
        "positive_traits": list(cfg.positive),
        "excluded_traits": res["excluded"],
        "valence_axis": {
            "raw_norm": res["raw_norm"],
            "cos_with_trait": res["proj"],
            "cos_with_centroid": res["cos_centroid"],
        },
        "residual_cosine": {
            "n_pairs": len(res["rows"]),
            "mean_signed_orig": float(co.mean()),
            "mean_signed_residual": float(cr.mean()),
            "mean_abs_orig": float(np.abs(co).mean()),
            "mean_abs_residual": float(np.abs(cr).mean()),
            "neg_neg_mean_signed_orig": float(co[nm].mean()) if nm.any() else None,
            "neg_neg_mean_signed_residual": float(cr[nm].mean()) if nm.any() else None,
            "neg_neg_mean_abs_orig": float(np.abs(co[nm]).mean()) if nm.any() else None,
            "neg_neg_mean_abs_residual": float(np.abs(cr[nm]).mean()) if nm.any() else None,
        },
        "pca": {
            "pc_alignment_with_valence": res["pc_align"].tolist(),
            "pc1_alignment": float(res["pc_align"][0]),
            "best_matching_pc": res["best_pc"] + 1,
            "valence_variance_fraction": res["val_var_frac"],
            "verdict": res["verdict"],
            "permutation_test": {
                "statistic": "abs_cos_pc1_valence",
                "exact": res["exact_null"],
                "n_labellings": res["n_lab"],
                "observed": float(res["pc_align"][0]),
                "null_mean": float(ns.mean()),
                "null_std": float(ns.std()),
                "null_p95": float(np.percentile(ns, 95)),
                "null_max": float(ns.max()),
                "p_value": res["p_perm"],
            },
        },
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _import_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:  # pragma: no cover
        print(f"\n[plot] skipped (matplotlib unavailable: {e})")
        return None


def plot_config(res: dict, evr: np.ndarray) -> Path | None:
    plt = _import_plt()
    if plt is None:
        return None
    cfg = res["cfg"]
    proj, co, cr = res["proj"], res["co"], res["cr"]
    pc_align, best_pc = res["pc_align"], res["best_pc"]
    ns, observed, p_perm = res["null_stats"], float(res["pc_align"][0]), res["p_perm"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes.ravel()
    fig.suptitle(f"{cfg.name} — {cfg.title}", fontsize=12, y=1.0)

    items = sorted(proj.items(), key=lambda kv: kv[1])
    names = [t for t, _ in items]
    colors = ["#c0392b" if t in cfg.negative else "#2980b9" if t in cfg.positive
              else "#7f8c8d" for t in names]
    ax[0].barh(names, [c for _, c in items], color=colors)
    ax[0].axvline(0, color="k", lw=0.8)
    ax[0].set_title("cos(trait, valence axis)")
    ax[0].set_xlabel("← prosocial      antisocial →")

    lim = max(np.abs(co).max(), np.abs(cr).max()) * 1.1
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].axvline(0, color="k", lw=0.6)
    ax[1].plot([-lim, lim], [-lim, lim], "--", color="grey", lw=0.8)
    ax[1].scatter(co, cr, s=28, alpha=0.8, color="#8e44ad")
    ax[1].set_xlim(-lim, lim)
    ax[1].set_ylim(-lim, lim)
    ax[1].set_xlabel("cosine (original)")
    ax[1].set_ylabel("cosine (valence removed)")
    ax[1].set_title("pairwise cosine: shrink toward 0 = valence-driven")

    x = np.arange(1, len(evr) + 1)
    ax[2].bar(x, evr, color="#bdc3c7", label="explained var")
    ax[2].plot(x, pc_align, "o-", color="#c0392b", label="|cos(PC, v_val)|")
    ax[2].set_xticks(x)
    ax[2].set_xlabel("principal component")
    ax[2].set_ylim(0, 1)
    ax[2].set_title(f"PCA vs valence axis (best = PC{best_pc + 1})")
    ax[2].legend(loc="upper right", fontsize=8)

    ax[3].hist(ns, bins=20, color="#bdc3c7", edgecolor="white")
    ax[3].axvline(observed, color="#c0392b", lw=2,
                  label=f"observed = {observed:.3f}\np = {p_perm:.3f}")
    ax[3].set_xlabel("|cos(PC1, axis)| over all labellings")
    ax[3].set_ylabel("count")
    ax[3].set_title("permutation null: is PC1≈valence special?")
    ax[3].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    path = OUT_DIR / f"valence_axis_{cfg.name}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_comparison(results: list[dict], ctx: dict) -> Path | None:
    plt = _import_plt()
    if plt is None:
        return None
    traits = ctx["traits"]
    names = [r["cfg"].name for r in results]
    width = 0.8 / len(results)
    palette = ["#c0392b", "#2980b9", "#16a085", "#8e44ad"]

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))

    # (a) cos(trait, axis) per config — the hallucinating narrative.
    order = sorted(traits, key=lambda t: results[0]["proj"][t])
    xpos = np.arange(len(order))
    for j, res in enumerate(results):
        ax[0].bar(xpos + j * width, [res["proj"][t] for t in order],
                  width, label=res["cfg"].name, color=palette[j % len(palette)])
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xticks(xpos + width * (len(results) - 1) / 2)
    ax[0].set_xticklabels(order, rotation=45, ha="right")
    # highlight hallucinating tick
    for lbl in ax[0].get_xticklabels():
        if lbl.get_text() == "hallucinating":
            lbl.set_color("#c0392b")
            lbl.set_fontweight("bold")
    ax[0].set_ylabel("cos(trait, valence axis)")
    ax[0].set_title("axis sharpening across labellings")
    ax[0].legend(fontsize=8)

    # (b) neg-neg signed cos, original vs residual, per config.
    xb = np.arange(len(results))
    orig = [r["co"][r["neg_mask"]].mean() for r in results]
    resid = [r["cr"][r["neg_mask"]].mean() for r in results]
    ax[1].bar(xb - 0.2, orig, 0.4, label="original", color="#bdc3c7")
    ax[1].bar(xb + 0.2, resid, 0.4, label="valence removed", color="#c0392b")
    ax[1].set_xticks(xb)
    ax[1].set_xticklabels(names)
    ax[1].set_ylabel("mean signed cos (neg-neg pairs)")
    ax[1].set_title("antisocial cluster collapse")
    ax[1].legend(fontsize=8)

    # (c) PC1 alignment with null band + p-value per config.
    for j, res in enumerate(results):
        ns = res["null_stats"]
        ax[2].errorbar(j, ns.mean(), yerr=ns.std(), fmt="o", color="grey",
                       capsize=4, label="null mean±sd" if j == 0 else None)
        ax[2].scatter(j, res["pc_align"][0], color="#c0392b", zorder=3, s=60,
                      label="observed" if j == 0 else None)
        ax[2].annotate(f"p={res['p_perm']:.2f}", (j, res["pc_align"][0]),
                       textcoords="offset points", xytext=(8, 0), fontsize=9)
    ax[2].set_xticks(range(len(results)))
    ax[2].set_xticklabels(names)
    ax[2].set_ylim(0, 1)
    ax[2].set_ylabel("|cos(PC1, valence axis)|")
    ax[2].set_title("PC1 alignment vs permutation null")
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    path = OUT_DIR / "comparison.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ctx = build_context()

    print("=" * 72)
    print("VALENCE-AXIS ANALYSIS — norm dataset")
    print("=" * 72)
    s = ctx["summary"]
    print(f"summary      : {NORM_SUMMARY.relative_to(REPO_ROOT)}")
    print(f"model/layer  : {s['model']}  L={s['layer']}  alpha={s['alpha']}")
    print(f"vectors      : response_avg_diff[{LAYER}], unit-normalised")
    print(f"traits ({len(ctx['traits'])})  : {ctx['traits']}")
    print(f"[sanity] max |recomputed cos - stored cos| over "
          f"{len(ctx['pairs_meta'])} pairs: {ctx['max_diff']:.2e}")
    if ctx["max_diff"] > 1e-3:
        print("  WARNING: cosines do not match — wrong vectors/layer?")
    print("\nshared PCA explained variance (config-independent):")
    print("  " + "  ".join(f"PC{i+1}={v:.0%}" for i, v in enumerate(ctx["evr"])))

    results = []
    for cfg in CONFIGS:
        res = analyze_config(cfg, ctx)
        print_config_report(res, ctx["evr"])
        results.append(res)

    print_comparison(results, ctx)

    written = persist(results, ctx)
    figs = [plot_config(res, ctx["evr"]) for res in results]
    figs.append(plot_comparison(results, ctx))

    print("\n" + "=" * 72)
    print("Saved:")
    for p in written + [f for f in figs if f]:
        print(f"  {p.relative_to(REPO_ROOT)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
