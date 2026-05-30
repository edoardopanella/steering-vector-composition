"""
bins_analysis_scripts.py

Worker module for the composition-geometry statistical analysis.
No argparse: configure and launch it from `bins_analysis.py`.

Source of truth is the *consolidated* frame
``analysis/rq1_consolidated/consolidated_coh30.csv``, which merges every
injection alternative (raw = ``normFalse``, norm = ``normTrue``,
per-axis = ``per_axis``) into one table. This module runs the full battery on
the **norm** and **per-axis** subsets and writes a single, all-comprehensive
report in which the two result tables for each step sit *side by side* (merged
into one table, one scheme's columns next to the other's) for direct visual
comparison. The raw (``normFalse``) scheme is part of the source file but is not
analysed here.

Statistical analysis run, in order, for each scheme:
  1. Cross-tab |cos| bin x regime  (Monte-Carlo exact test + 2x2 Fisher's exact)
  2. Correlation of geometry (signed / |cos|) against semantic similarity
  3. Multinomial logistic regression: geometry -> regime classes  (M1-M4)
  4. Binary logistic regression:      geometry -> is_additive      (B1-B4)

Outputs (one combined report covering both schemes):
  - norm_vs_peraxis_bins_analysis.md    human-readable report, side-by-side tables
  - norm_vs_peraxis_bins_analysis.json  machine-readable results for both schemes

Public entry point:
    run_consolidated_analysis()
    run_consolidated_analysis(n_boot=200, n_perm=500)   # fast smoke run
"""

import json
import re
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

# Everything from this header onward is hand-written and preserved across reruns;
# everything above it (the results) is auto-generated and overwritten each run.
CONCLUSIONS_HEADER = "## Conclusions"
CONCLUSIONS_HINT = (
    "<!-- Write your conclusions below. This whole section (from the "
    "`## Conclusions` heading down) is kept when the script is re-run; the results "
    "above are regenerated each time. Keep the heading text exactly as-is. -->"
)
DEFAULT_CONCLUSIONS = f"{CONCLUSIONS_HEADER}\n\n{CONCLUSIONS_HINT}\n\n_(write your conclusions here)_\n"

# --------------------------------------------------------------------------- #
# Consolidated-report config.
# `schemes` maps each analysed subset to its value in the CSV `scheme` column
# and the short label used to suffix that scheme's columns in the merged tables.
# --------------------------------------------------------------------------- #
CONSOLIDATED = {
    "csv": REPO_ROOT / "analysis/rq1_consolidated/consolidated_coh30.csv",
    "out_dir": REPO_ROOT / "analysis/results/RQ1/norm_vs_peraxis_bins_analysis",
    "out_stem": "norm_vs_peraxis_bins_analysis",
    "title": "Norm-sum vs per-axis composition geometry — side-by-side statistical analysis",
    "intro": (
        "Two injection schemes, analysed on the consolidated (coh≥30) frame and "
        "reported side by side for direct comparison:\n\n"
        "- **norm** (`normTrue`): normalised sum-injection fixes the total injected "
        r"norm ($\lVert\delta\rVert=\alpha$ regardless of the angle $\theta_{ij}$); "
        "the per-behaviour component along each direction still varies with the angle.\n"
        "- **peraxis** (`per_axis`): per-axis injection adds each behaviour's steering "
        r"vector at a fixed per-axis norm ($\lVert\delta_i\rVert=\alpha$ along every "
        "direction), so the total injected norm grows with the angle between the two "
        "directions.\n\n"
        "The raw (`normFalse`) scheme is present in the source file but is not analysed here."
    ),
    # (subset key, CSV scheme value, column-suffix label)
    "schemes": [
        ("norm", "normTrue", "norm"),
        ("peraxis", "per_axis", "peraxis"),
    ],
}

# Preferred display order for regime classes (extras appended afterwards).
REGIME_ORDER = ["additive", "dominant", "mixed", "suppressive", "emergent"]


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
# Data loading (one subset of the consolidated frame)
# --------------------------------------------------------------------------- #
def load_consolidated(csv_path, scheme_value):
    """Load one ``scheme`` subset of the consolidated frame.

    The consolidated CSV already carries ``sem_sim``; the per-pair geometry
    features (``cosine_abs``, ``cos_bin``) are derived exactly as the original
    summary-based loader did, so the downstream analysis is unchanged.
    """
    df = pd.read_csv(csv_path)
    df = df[df["scheme"] == scheme_value].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError(f"no rows for scheme={scheme_value!r} in {csv_path}")

    # Absolute cosine + the same strata used by the original cross-tab.
    df["cosine_abs"] = df["cos"].abs()
    df["cos_bin"] = pd.cut(
        df["cosine_abs"],
        bins=[0, 0.2, 0.5, 1.0],
        labels=["low", "moderate", "high"],
    )

    meta = {
        "scheme": scheme_value,
        "alpha": float(df["alpha"].iloc[0]),
        "n_pairs": int(len(df)),
    }
    return df, meta


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
        ("M1_multi_abs_cos", ["cosine_abs"], "cosine_abs"),
        ("M2_multi_abs_sem", ["cosine_abs", "sem_sim"], "cosine_abs + sem_sim"),
        ("M3_multi_signed_cos", ["cos"], "signed_cos"),
        ("M4_multi_signed_sem", ["cos", "sem_sim"], "signed_cos + sem_sim"),
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


def run_scheme(csv_path, scheme_value, n_boot, n_perm):
    """Run the full battery for one scheme subset; return everything the report needs."""
    df, meta = load_consolidated(csv_path, scheme_value)
    print(f"[{scheme_value}] alpha={meta['alpha']}  pairs={meta['n_pairs']}")
    xt = crosstab_tests(df, n_mc=n_perm)
    corr = correlations(df)
    results, coefficients = run_regressions(df, n_boot=n_boot, n_perm=n_perm)
    return {"meta": meta, "df": df, "xt": xt, "corr": corr,
            "results": results, "coefficients": coefficients}


# --------------------------------------------------------------------------- #
# Merge helpers -- put one scheme's columns next to the other's
# --------------------------------------------------------------------------- #
def _ordered_regimes(*crosstabs):
    """Union of regime labels across crosstabs, in REGIME_ORDER then extras."""
    seen = []
    for ct in crosstabs:
        for r in ct.columns:
            if r not in seen:
                seen.append(r)
    ordered = [r for r in REGIME_ORDER if r in seen]
    ordered += [r for r in seen if r not in ordered]
    return ordered


def _summary_frame(results, outcome, extra_cols):
    cols = ["experiment", "predictors", "n_features",
            "auc_insample", "auc_loo", "ci_lo", "ci_hi", "perm_p", "perm_auc_mean"]
    rows = [r for r in results if r["outcome"] == outcome]
    return pd.DataFrame(rows)[cols + extra_cols]


def _exp_frame(results, outcome, extra_cols):
    """Summary frame for one outcome, with a pre-formatted bootstrap-CI string."""
    fr = _summary_frame(results, outcome, extra_cols).reset_index(drop=True)
    fr["ci"] = [f"[{lo:.3f}, {hi:.3f}]" for lo, hi in zip(fr["ci_lo"], fr["ci_hi"])]
    # Permutation p needs finer resolution than the AUC columns: the add-one
    # estimator floors at 1/(N_PERM+1), so 3 decimals would misleadingly show
    # strong-but-nonzero p-values as "0.000". Pre-format to 4 decimals as text.
    fr["perm_p"] = [f"{p:.4f}" for p in fr["perm_p"]]
    return fr


def _crosstab_table(ct):
    """One scheme's |cos|-bin x regime count table, fixed bin order + int cells."""
    bins = ["low", "moderate", "high"]
    regimes = _ordered_regimes(ct)
    out = ct.reindex(index=bins, columns=regimes, fill_value=0).astype(int)
    out.index.name = "cos_bin"
    return out


def _fisher_table(t):
    """One scheme's collapsed 2x2 contingency table with labelled columns."""
    out = t.reindex(index=[0, 1], columns=[0, 1], fill_value=0).astype(int)
    out.columns = [f"is_additive={c}" for c in [0, 1]]
    out.index.name = "|cos|>=0.2"
    return out


def _multi_table(results):
    """One scheme's multinomial summary table (original per-dataset column layout)."""
    fr = _exp_frame(results, "regime", [])
    return fr[["experiment", "predictors", "n_features",
               "auc_insample", "auc_loo", "ci", "perm_p", "perm_auc_mean"]]


def _binary_table(results):
    """One scheme's binary summary table, with slopes and the evidence flag."""
    fr = _exp_frame(results, "is_additive",
                    ["slope_abs_cos", "slope_signed_cos", "slope_sem_sim"])
    fr["evidence"] = ["yes" if (r["auc_loo"] > 0.7 and r["perm_p"] < 0.05) else "no"
                      for r in results if r["outcome"] == "is_additive"]
    return fr[["experiment", "predictors", "n_features", "auc_insample", "auc_loo",
               "ci", "perm_p", "slope_abs_cos", "slope_signed_cos", "slope_sem_sim",
               "evidence"]]


def _scalar_cmp(rows):
    """Build a 'metric | norm | peraxis' comparison frame from pre-formatted rows."""
    return pd.DataFrame(rows, columns=["metric", "norm", "peraxis"])


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def build_markdown(cfg, data, n_boot, n_perm):
    (key_a, _, lbl_a), (key_b, _, lbl_b) = cfg["schemes"]
    A, B = data[key_a], data[key_b]
    out = []
    a = out.append

    a(f"# {cfg['title']}\n")
    a(f"_Results auto-generated by `bins_analysis_scripts.py` on {datetime.now():%Y-%m-%d %H:%M}. "
      "Everything down to the `## Conclusions` heading is overwritten on each run; "
      "write your own interpretation under that heading and it will be preserved._\n")
    a(cfg["intro"] + "\n")
    a(f"Source: `{cfg['csv'].relative_to(REPO_ROOT)}`. In every phase below, the "
      f"**{lbl_a}** table is shown first, with the **{lbl_b}** table stacked directly "
      "underneath it for comparison.\n")

    # Setup -----------------------------------------------------------------
    a("## Setup\n")
    regimes = _ordered_regimes(A["xt"]["crosstab"], B["xt"]["crosstab"])
    rc_a = A["df"]["regime"].value_counts()
    rc_b = B["df"]["regime"].value_counts()
    ba = A["df"]["is_additive"].value_counts()
    bb = B["df"]["is_additive"].value_counts()
    setup_rows = [
        ("pairs (status=ok)", A["meta"]["n_pairs"], B["meta"]["n_pairs"]),
        ("alpha", f"{A['meta']['alpha']:g}", f"{B['meta']['alpha']:g}"),
    ]
    for r in regimes:
        setup_rows.append((f"regime: {r}", int(rc_a.get(r, 0)), int(rc_b.get(r, 0))))
    setup_rows.append(("is_additive = 1", int(ba.get(1, 0)), int(bb.get(1, 0))))
    setup_rows.append(("is_additive = 0", int(ba.get(0, 0)), int(bb.get(0, 0))))
    a(df_to_md(_scalar_cmp(setup_rows)) + "\n")
    a(f"**bootstrap N** = {n_boot}, **permutation N** = {n_perm}. "
      "(Model / layer / tau are not carried in the consolidated frame.)\n")

    # Section 1 -------------------------------------------------------------
    a("## 1. Cross-tab: |cos| bin × regime\n")
    a("Counts of pairs per |cos| stratum and regime.\n")
    mc_a, fi_a = A["xt"]["monte_carlo"], A["xt"]["fisher_2x2"]
    mc_b, fi_b = B["xt"]["monte_carlo"], B["xt"]["fisher_2x2"]
    for lbl, blk, mc in ((lbl_a, A, mc_a), (lbl_b, B, mc_b)):
        a(f"**{lbl}**\n")
        a(df_to_md(_crosstab_table(blk["xt"]["crosstab"]), index=True) + "\n")
        a(f"Monte-Carlo exact test ({mc['n_perm']} permutations): "
          f"χ² = {mc['chi2']:.3f}, p = {mc['p']:.4f}\n")
    a("**Collapsed 2×2 contingency** (|cos|≥0.2 × is_additive):\n")
    for lbl, fi in ((lbl_a, fi_a), (lbl_b, fi_b)):
        a(f"**{lbl}**\n")
        a(df_to_md(_fisher_table(fi["table"]), index=True) + "\n")
        a(f"odds ratio = {fi['odds_ratio']:.3f}, p = {fi['p']:.4f}\n")

    # Section 2 -------------------------------------------------------------
    a("## 2. Geometry vs semantic similarity\n")
    for lbl, blk in ((lbl_a, A), (lbl_b, B)):
        ac, sc = blk["corr"]["abs_cos_vs_sem"], blk["corr"]["signed_cos_vs_sem"]
        a(f"**{lbl}**")
        a(f"- Pearson r(|cos|, sem_sim) = {ac['r']:+.3f}  (p = {ac['p']:.4f})")
        a(f"- Pearson r(signed cos, sem_sim) = {sc['r']:+.3f}\n")

    # Section 3 -------------------------------------------------------------
    a("## 3. Multinomial logistic regression (outcome = regime)\n")
    a("Multi-class target; AUC is macro one-vs-rest. Columns: in-sample AUC, LOO AUC, "
      "bootstrap 95% CI, permutation p, mean permutation AUC.\n")
    for lbl, blk in ((lbl_a, A), (lbl_b, B)):
        a(f"**{lbl}**\n")
        a(df_to_md(_multi_table(blk["results"]), floatfmt=".3f") + "\n")

    # Section 4 -------------------------------------------------------------
    a("## 4. Binary logistic regression (outcome = is_additive)\n")
    a("Same four predictor sets, collapsed to additive-vs-non-additive. Columns: in-sample "
      "AUC, LOO AUC, bootstrap 95% CI, permutation p, slopes, and an `evidence` flag "
      "(LOO AUC > 0.7 **and** permutation p < 0.05).\n")
    for lbl, blk in ((lbl_a, A), (lbl_b, B)):
        a(f"**{lbl}**\n")
        a(df_to_md(_binary_table(blk["results"]), floatfmt=".3f") + "\n")

    return "\n".join(out)


def _preserve_conclusions(md_path):
    """Return the hand-written conclusions section of an existing report.

    Looks for the `## Conclusions` heading; everything from there to the end of the
    file is the user's and is returned verbatim. If the file is missing or has no
    such heading, a blank conclusions template is returned instead.
    """
    if md_path.exists():
        text = md_path.read_text()
        # Anchor to the heading at the start of a line; the auto-generated note
        # near the top also mentions "## Conclusions" inline, which must not match.
        m = re.search(rf"^{re.escape(CONCLUSIONS_HEADER)}", text, re.MULTILINE)
        if m:
            return text[m.start():].rstrip() + "\n"
    return DEFAULT_CONCLUSIONS


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


def _scheme_json(scheme_value, blk):
    df, xt, corr = blk["df"], blk["xt"], blk["corr"]
    bb = df["is_additive"].value_counts().sort_index()
    return {
        "scheme": scheme_value,
        "metadata": blk["meta"],
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
        "experiments": blk["results"],
        "coefficients": blk["coefficients"],
    }


def build_json(cfg, data, n_boot, n_perm):
    return {
        "report": "norm_vs_peraxis_consolidated",
        "source_csv": str(cfg["csv"].relative_to(REPO_ROOT)),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_boot": n_boot,
        "n_perm": n_perm,
        "schemes": {key: _scheme_json(scheme_value, data[key])
                    for key, scheme_value, _ in cfg["schemes"]},
    }


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def run_consolidated_analysis(n_boot=2000, n_perm=10000):
    """Run the full battery on the norm and per-axis subsets of the consolidated
    frame and write one combined report with side-by-side tables.

    Parameters
    ----------
    n_boot, n_perm : int
        Bootstrap / permutation iteration counts.
    """
    cfg = CONSOLIDATED
    print(f"Loading consolidated frame {cfg['csv']}")

    data = {}
    for key, scheme_value, _ in cfg["schemes"]:
        print(f"\n########## scheme: {key} ({scheme_value}) ##########")
        data[key] = run_scheme(cfg["csv"], scheme_value, n_boot, n_perm)

    out_dir = cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{cfg['out_stem']}.md"
    json_path = out_dir / f"{cfg['out_stem']}.json"

    # Regenerate the results, but keep any hand-written conclusions from a prior run.
    results_md = build_markdown(cfg, data, n_boot, n_perm)
    md = results_md.rstrip() + "\n\n" + _preserve_conclusions(md_path)
    md_path.write_text(md)
    with open(json_path, "w") as f:
        json.dump(build_json(cfg, data, n_boot, n_perm), f, indent=2, default=_json_default)

    print(f"\nWrote {md_path}")
    print(f"Wrote {json_path}")
    return md_path, json_path
