"""
norm_bins_analysis.py

Standalone conversion of analysis/notebooks_v2/norm_bins_analysis.ipynb.

Statistical analysis of the composition geometry for normalised-sum injection.
Runs, in order:
  1. Cross-tab |cos| bin x regime  (Monte-Carlo exact test + 2x2 Fisher's exact)
  2. Correlation of geometry (signed / |cos|) against semantic similarity
  3. Multinomial logistic regression: geometry -> 4-class regime  (M1-M4)
  4. Binary logistic regression:      geometry -> is_additive     (B1-B4)

Outputs (both written to --out-dir):
  - norm_bins_analysis.md    human-readable report
  - norm_bins_analysis.json  machine-readable results (incl. bootstrap distributions)

Usage:
    python norm_bins_analysis.py
    python norm_bins_analysis.py --n-boot 200 --n-perm 500   # fast smoke run
    python norm_bins_analysis.py --summary <path> --out-dir <dir>
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler

# bins_utils lives next to this script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bins_utils import _chi2_stat, bootstrap_perm, bootstrap_perm_multi  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = REPO_ROOT / "results/composition/v2_phase125_normTrue_a4.5/scoring/summary.json"
DEFAULT_SEM = REPO_ROOT / "results/semantic_similarity.json"
DEFAULT_OUT = REPO_ROOT / "analysis/results/RQ1/norm_bins_analysis"


# --------------------------------------------------------------------------- #
# Markdown helpers (no tabulate dependency)
# --------------------------------------------------------------------------- #
def _esc(s):
    """Escape pipes so cell content never breaks the markdown table layout."""
    return str(s).replace("|", r"\|")


def _fmt(v, floatfmt):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (float, np.floating)):
        if floatfmt is not None:
            return format(float(v), floatfmt)
        return f"{float(v):g}"
    return _esc(v)


def df_to_md(df, floatfmt=None, index=False, index_name=None):
    """Render a DataFrame as a GitHub-flavoured markdown table without tabulate."""
    cols = list(df.columns)
    headers = ([index_name or (df.index.name or "")] if index else []) + [str(c) for c in cols]
    headers = [_esc(h) for h in headers]
    rows = []
    for idx, row in df.iterrows():
        cells = ([_esc(idx)] if index else []) + [_fmt(row[c], floatfmt) for c in cols]
        rows.append(cells)
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for cells in rows:
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_data(summary_path):
    """Load the per-composition summary and flatten the ok pairs into a frame."""
    with open(summary_path) as f:
        summary = json.load(f)

    meta = {
        "model": summary["model"],
        "layer": summary["layer"],
        "alpha": summary["alpha"],
        "tau_value": summary["tau_value"],
    }

    rows = []
    for p in summary["pairs"]:
        if p.get("status") != "ok":
            continue
        rows.append({
            "trait_a": p["trait_a"], "trait_b": p["trait_b"],
            "cos": p["cos"], "regime": p["regime"],
            "comp_base": p["baseline"]["composition_mean"],
            "comp_steered": p["steered"]["composition_mean"],
            "coh_steered": p["steered"]["coherence_mean"],
            "delta_a_joint": p["delta"]["trait_a_joint"],
            "delta_a_single": p["delta"]["trait_a_single"],
            "delta_b_joint": p["delta"]["trait_b_joint"],
            "delta_b_single": p["delta"]["trait_b_single"],
            "delta_comp": p["delta"]["composition"],
            "delta_coh": p["delta"]["coherence"],
        })

    df = pd.DataFrame(rows)

    # Absolute cosine + the same strata used by the original cross-tab.
    df["cosine_abs"] = df["cos"].abs()
    df["cos_bin"] = pd.cut(
        df["cosine_abs"],
        bins=[0, 0.2, 0.5, 1.0],
        labels=["low", "moderate", "high"],
    )
    return df, meta


def add_semantic(df, sem_path):
    """Attach the symmetric semantic-similarity score for each trait pair."""
    with open(sem_path) as f:
        sem = json.load(f)
    sem_lookup = {
        tuple(sorted([p["trait_a"], p["trait_b"]])): p["sem_sim"]
        for p in sem["pairs"]
    }
    df["sem_sim"] = df.apply(
        lambda r: sem_lookup[tuple(sorted([r["trait_a"], r["trait_b"]]))],
        axis=1,
    )
    return df


# --------------------------------------------------------------------------- #
# Section 1 -- cross-tab association tests
# --------------------------------------------------------------------------- #
def crosstab_tests(df, n_mc=10000):
    """|cos| bin x regime association: full-table MC exact + collapsed 2x2 Fisher."""
    ct = pd.crosstab(df["cos_bin"], df["regime"])

    # (1) Full r x c table: Monte-Carlo exact test (valid for sparse tables).
    sub = df.dropna(subset=["cos_bin"])
    bin_code = pd.Categorical(sub["cos_bin"]).codes
    reg_code = pd.Categorical(sub["regime"]).codes
    n_bin, n_reg = bin_code.max() + 1, reg_code.max() + 1

    rng_mc = np.random.default_rng(0)
    obs_chi2 = _chi2_stat(bin_code, reg_code, n_bin, n_reg)
    perm = np.fromiter(
        (_chi2_stat(bin_code, rng_mc.permutation(reg_code)) for _ in range(n_mc)),
        float, n_mc,
    )
    p_mc = (np.sum(perm >= obs_chi2) + 1) / (n_mc + 1)

    # (2) Collapsed 2x2: near-orthogonal vs aligned  x  additive vs not.
    df["cos_hi"] = (df["cosine_abs"] >= 0.2).astype(int)
    additive = (df["regime"] == "additive").astype(int)
    tab2 = pd.crosstab(df["cos_hi"], additive)
    tab2.index.name = "|cos|>=0.2"
    tab2.columns.name = "is_additive"
    odds, p_fisher = fisher_exact(tab2, alternative="two-sided")

    return {
        "crosstab": ct,
        "monte_carlo": {"n_perm": n_mc, "chi2": float(obs_chi2), "p": float(p_mc)},
        "fisher_2x2": {
            "table": tab2,
            "odds_ratio": float(odds),
            "p": float(p_fisher),
        },
    }


# --------------------------------------------------------------------------- #
# Section 2 -- geometry vs semantic similarity
# --------------------------------------------------------------------------- #
def correlations(df):
    r_cs, p_cs = pearsonr(df["cosine_abs"], df["sem_sim"])
    r_signed = df[["cos", "sem_sim"]].corr().iloc[0, 1]
    return {
        "abs_cos_vs_sem": {"r": float(r_cs), "p": float(p_cs)},
        "signed_cos_vs_sem": {"r": float(r_signed)},
    }


# --------------------------------------------------------------------------- #
# Sections 3 & 4 -- logistic regressions
# --------------------------------------------------------------------------- #
def _slope_dict(cols, coef_row):
    """Map a (binary) coefficient row onto the notebook's fixed slope schema."""
    d = {"slope_abs_cos": None, "slope_both_antisocial": None, "slope_signed_cos": None}
    for i, c in enumerate(cols):
        if c == "cosine_abs":
            d["slope_abs_cos"] = float(coef_row[i])
        elif c == "cos":
            d["slope_signed_cos"] = float(coef_row[i])
        elif c == "sem_sim":
            d["slope_sem_sim"] = float(coef_row[i])
    return d


def run_experiment(name, cols, predictors, outcome, df, scaler, model, multiclass,
                   n_boot, n_perm, rng):
    """Fit one model, compute in-sample + LOO AUC, bootstrap CI and permutation p."""
    X = scaler.fit_transform(df[cols].to_numpy())
    y = df[outcome].to_numpy()

    fit = model.fit(X, y)

    if multiclass:
        proba_in = fit.predict_proba(X)
        auc_in = roc_auc_score(y, proba_in, multi_class="ovr", average="macro")
        proba_loo = cross_val_predict(model, X, y, cv=LeaveOneOut(), method="predict_proba")
        auc_loo = roc_auc_score(y, proba_loo, multi_class="ovr", average="macro")
        stats = bootstrap_perm_multi(model, X, y, auc_in, n_boot, n_perm, rng)
        coefs = {
            str(cls): {cols[j]: float(fit.coef_[i, j]) for j in range(len(cols))}
            for i, cls in enumerate(fit.classes_)
        }
        result = {
            "experiment": name, "outcome": outcome, "predictors": predictors,
            "n_features": len(cols), "auc_loo": float(auc_loo), **stats,
        }
    else:
        proba_in = fit.predict_proba(X)[:, 1]
        auc_in = roc_auc_score(y, proba_in)
        proba_loo = cross_val_predict(model, X, y, cv=LeaveOneOut(),
                                      method="predict_proba")[:, 1]
        auc_loo = roc_auc_score(y, proba_loo)
        stats = bootstrap_perm(model, X, y, auc_in, n_boot, n_perm, rng)
        coefs = {cols[j]: float(fit.coef_[0, j]) for j in range(len(cols))}
        result = {
            "experiment": name, "outcome": outcome, "predictors": predictors,
            "n_features": len(cols), "auc_loo": float(auc_loo),
            **_slope_dict(cols, fit.coef_[0]), **stats,
        }

    return result, coefs


def run_regressions(df, n_boot, n_perm):
    """Run M1-M4 (multinomial) then B1-B4 (binary), preserving notebook order/rng."""
    df["is_additive"] = (df["regime"] == "additive").astype(int)

    scaler = StandardScaler()
    multi_model = LogisticRegression(penalty=None)
    binary_model = LogisticRegression(penalty=None)
    rng = np.random.default_rng(0)  # single shared generator, as in the notebook

    multi_specs = [
        ("M1_multi_signed_cos", ["cos"], "signed_cos"),
        ("M2_multi_signed_sem", ["cos", "sem_sim"], "signed_cos + sem_sim"),
        ("M3_multi_abs_cos", ["cosine_abs"], "cosine_abs"),
        ("M4_multi_abs_sem", ["cosine_abs", "sem_sim"], "cosine_abs + sem_sim"),
    ]
    binary_specs = [
        ("B1_abs_cos", ["cosine_abs"], "cosine_abs"),
        ("B2_abs_cos_sem", ["cosine_abs", "sem_sim"], "cosine_abs + sem_sim"),
        ("B3_signed_cos", ["cos"], "signed_cos"),
        ("B4_signed_cos_sem", ["cos", "sem_sim"], "signed_cos + sem_sim"),
    ]

    results, coefficients = [], {}
    for name, cols, predictors in multi_specs:
        print(f"\n=== {name} ===")
        res, coefs = run_experiment(name, cols, predictors, "regime", df, scaler,
                                    multi_model, True, n_boot, n_perm, rng)
        results.append(res)
        coefficients[name] = coefs
    for name, cols, predictors in binary_specs:
        print(f"\n=== {name} ===")
        res, coefs = run_experiment(name, cols, predictors, "is_additive", df, scaler,
                                    binary_model, False, n_boot, n_perm, rng)
        results.append(res)
        coefficients[name] = coefs

    return results, coefficients


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _summary_frame(results, outcome, extra_cols):
    cols = ["experiment", "predictors", "n_features",
            "auc_insample", "auc_loo", "ci_lo", "ci_hi", "perm_p", "perm_auc_mean"]
    rows = [r for r in results if r["outcome"] == outcome]
    return pd.DataFrame(rows)[cols + extra_cols]


def build_markdown(meta, df, xt, corr, results, n_boot, n_perm):
    out = []
    a = out.append

    a("# Norm-sum composition geometry — statistical analysis\n")
    a(f"_Generated by `norm_bins_analysis.py` on {datetime.now():%Y-%m-%d %H:%M}_\n")
    a("Normalised sum-injection fixes the total injected norm "
      r"($\lVert\delta\rVert=\alpha$ regardless of the angle $\theta_{ij}$); "
      "the per-behaviour component along each direction still varies with the angle.\n")

    # Setup -----------------------------------------------------------------
    a("## Setup\n")
    a(f"- **Model:** `{meta['model']}`")
    a(f"- **Layer:** {meta['layer']}  ·  **alpha:** {meta['alpha']}  "
      f"·  **tau:** {meta['tau_value']:.4f}")
    a(f"- **Pairs (status=ok):** {len(df)}")
    rc = df["regime"].value_counts()
    a(f"- **Regime counts:** " + ", ".join(f"{k}={v}" for k, v in rc.items()))
    bb = df["is_additive"].value_counts().sort_index()
    a(f"- **Binary balance (is_additive):** "
      + ", ".join(f"{int(k)}={int(v)}" for k, v in bb.items())
      + f"  ·  **bootstrap N** = {n_boot}, **permutation N** = {n_perm}\n")

    # Section 1 -------------------------------------------------------------
    a("## 1. Cross-tab: |cos| bin × regime\n")
    ct = xt["crosstab"].copy()
    ct.index.name = "cos_bin"
    a(df_to_md(ct, index=True) + "\n")
    mc, fi = xt["monte_carlo"], xt["fisher_2x2"]
    a(f"**Full {xt['crosstab'].shape[0]}×{xt['crosstab'].shape[1]} table — "
      f"Monte-Carlo exact test** ({mc['n_perm']} permutations): "
      f"χ² = {mc['chi2']:.3f}, p = {mc['p']:.4f}\n")
    a("**Collapsed 2×2 Fisher's exact** (|cos|≥0.2 × is_additive):\n")
    a(df_to_md(fi["table"], index=True) + "\n")
    a(f"odds ratio = {fi['odds_ratio']:.3f}, p = {fi['p']:.4f}\n")
    sig = "no significant" if min(mc["p"], fi["p"]) >= 0.05 else "a significant"
    a(f"_Conclusion: {sig} geometry–regime association (MC p = {mc['p']:.3f}, "
      f"Fisher p = {fi['p']:.3f}). The high-similarity stratum is nearly empty._\n")

    # Section 2 -------------------------------------------------------------
    a("## 2. Geometry vs semantic similarity\n")
    ac, sc = corr["abs_cos_vs_sem"], corr["signed_cos_vs_sem"]
    a(f"- Pearson r(|cos|, sem_sim) = {ac['r']:+.3f}  (p = {ac['p']:.4f})")
    a(f"- Pearson r(signed cos, sem_sim) = {sc['r']:+.3f}\n")
    if abs(ac["r"]) >= 0.5:
        a("_With |r| > 0.5, semantic similarity is a real confound for **absolute** "
          "cosine and must be controlled. Signed cosine is near-orthogonal to "
          "semantics, so it is largely unconfounded._\n")
    else:
        a("_Semantic similarity is only weakly correlated with the geometry here._\n")

    # Section 3 -------------------------------------------------------------
    a("## 3. Multinomial logistic regression (outcome = regime)\n")
    a("4-class target (additive / dominant / mixed / suppressive); AUC is macro "
      "one-vs-rest. With n = 28 across 4 classes this is heavily underpowered and "
      "exploratory — the point is to show the discrete regime is *not* recoverable "
      "from geometry, motivating the binary collapse below.\n")
    multi_tbl = _summary_frame(results, "regime", [])
    a(df_to_md(multi_tbl, floatfmt=".3f") + "\n")

    # Section 4 -------------------------------------------------------------
    a("## 4. Binary logistic regression (outcome = is_additive)\n")
    a("Same four predictor sets as the multinomial section, collapsed to "
      "additive-vs-non-additive. **LOO AUC is the primary metric**; in-sample AUC is "
      "an optimistic reference. **Evidence = LOO AUC > 0.7 AND permutation p < 0.05.**\n")
    bin_tbl = _summary_frame(results, "is_additive",
                             ["slope_abs_cos", "slope_signed_cos", "slope_sem_sim"]).copy()
    bin_tbl["evidence"] = [
        "yes" if (r["auc_loo"] > 0.7 and r["perm_p"] < 0.05) else "no"
        for r in results if r["outcome"] == "is_additive"
    ]
    a(df_to_md(bin_tbl, floatfmt=".3f") + "\n")

    return "\n".join(out)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def build_json(meta, df, xt, corr, results, coefficients):
    bb = df["is_additive"].value_counts().sort_index()
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "metadata": meta,
        "data": {
            "n_pairs": int(len(df)),
            "regime_counts": {k: int(v) for k, v in df["regime"].value_counts().items()},
            "binary_balance": {int(k): int(v) for k, v in bb.items()},
        },
        "crosstab": {
            "table": {str(k): {str(kk): int(vv) for kk, vv in v.items()}
                      for k, v in xt["crosstab"].to_dict().items()},
            "monte_carlo": xt["monte_carlo"],
            "fisher_2x2": {
                "table": {str(k): {str(kk): int(vv) for kk, vv in v.items()}
                          for k, v in xt["fisher_2x2"]["table"].to_dict().items()},
                "odds_ratio": xt["fisher_2x2"]["odds_ratio"],
                "p": xt["fisher_2x2"]["p"],
            },
        },
        "correlations": corr,
        "experiments": results,
        "coefficients": coefficients,
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--sem", type=Path, default=DEFAULT_SEM)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=10000)
    args = ap.parse_args()

    print(f"Loading {args.summary}")
    df, meta = load_data(args.summary)
    print(f"  model={meta['model']} layer={meta['layer']} "
          f"alpha={meta['alpha']} tau={meta['tau_value']:.4f}  pairs={len(df)}")

    xt = crosstab_tests(df, n_mc=args.n_perm)
    df = add_semantic(df, args.sem)
    corr = correlations(df)
    results, coefficients = run_regressions(df, n_boot=args.n_boot, n_perm=args.n_perm)

    md = build_markdown(meta, df, xt, corr, results, args.n_boot, args.n_perm)
    payload = build_json(meta, df, xt, corr, results, coefficients)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.out_dir / "norm_bins_analysis.md"
    json_path = args.out_dir / "norm_bins_analysis.json"
    md_path.write_text(md)
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)

    print(f"\nWrote {md_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
