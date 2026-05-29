"""Publication figure 1 — partial-regression (added-variable) plots.

Two-panel headline: same predictor (signed cos) is flat against magnitude
(mean_joint_abs) and sloped against direction (supp_mean_signed) — *after*
controlling for mean_single_abs, max_single_abs, mechanical_push, and scheme.

Controls are partialled out via Frisch-Waugh-Lovell: predictors and outcome
are z-scored over the pooled sample, regressed on (1 + z-controls + scheme
dummies), and the OLS slope through the residual scatter equals the standardized
β_cos from the full pooled model — so the drawn slope matches the in-panel
annotation by construction.

Frame: analysis/rq1_consolidated/consolidated_coh30.csv (n=76, 3 schemes).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Paths + rng
# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parents[2]
CSV  = REPO / "analysis/rq1_consolidated/consolidated_coh30.csv"
OUT_DIR = REPO / "analysis/figures/rq1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = OUT_DIR / "fig1_magnitude_vs_direction.pdf"
OUT_PNG = OUT_DIR / "fig1_magnitude_vs_direction.png"

RNG = np.random.default_rng(0)
N_PERM = 10_000

# --------------------------------------------------------------------------- #
# Load + recompute directional metrics
# --------------------------------------------------------------------------- #
df = pd.read_csv(CSV)
df["supp_signed_a"]    = df["delta_a_joint"] - df["delta_a_single"]
df["supp_signed_b"]    = df["delta_b_joint"] - df["delta_b_single"]
df["supp_mean_signed"] = 0.5 * (df["supp_signed_a"] + df["supp_signed_b"])

print(f"Loaded {CSV.name}: n={len(df)} rows, schemes={dict(df['scheme'].value_counts())}")

# --------------------------------------------------------------------------- #
# OLS + pooled-fit utilities (matching the notebook battery)
# --------------------------------------------------------------------------- #
def fit_ols(X, y):
    X = np.asarray(X, float); y = np.asarray(y, float)
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid @ resid)
    dof = n - k
    sigma2 = rss / dof if dof > 0 else np.nan
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv) * sigma2)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - rss / tss if tss > 0 else np.nan
    return {"beta": beta, "se": se, "rss": rss, "n": n, "k": k,
            "dof": dof, "r2": r2}


def pooled_design(sub, cols):
    Z = StandardScaler().fit_transform(sub[cols].values)
    d_nT = (sub["scheme"] == "normTrue").astype(float).values
    d_pa = (sub["scheme"] == "per_axis").astype(float).values
    X = np.column_stack([np.ones(len(sub)), Z, d_nT, d_pa])
    names = ["(intercept)"] + list(cols) + ["d_normTrue", "d_per_axis"]
    return X, names


def fit_pooled(sub, cols, y_col):
    X, names = pooled_design(sub, cols)
    y = StandardScaler().fit_transform(sub[[y_col]].values).ravel()
    fit = fit_ols(X, y); fit["cols"] = names
    return fit


def beta_cos_pooled(frame, y_col,
                    cols=("mean_single_abs", "max_single_abs",
                          "mechanical_push", "cos")):
    fit = fit_pooled(frame, list(cols), y_col)
    i = fit["cols"].index("cos")
    return float(fit["beta"][i]), float(fit["se"][i]), fit


def perm_p_cos(y_col, frame, n_perm=N_PERM, rng=RNG):
    obs, _, _ = beta_cos_pooled(frame, y_col)
    null = np.empty(n_perm)
    df_loc = frame.copy()
    cos_arr = df_loc["cos"].values.copy()
    for i in range(n_perm):
        for sch in df_loc["scheme"].unique():
            mask = (df_loc["scheme"] == sch).values
            blk = cos_arr[mask].copy(); rng.shuffle(blk); cos_arr[mask] = blk
        df_loc["cos"] = cos_arr
        b, _, _ = beta_cos_pooled(df_loc, y_col)
        null[i] = b
    return obs, float(np.mean(np.abs(null) >= np.abs(obs)))


# --------------------------------------------------------------------------- #
# Headline numbers (printed to stdout)
# --------------------------------------------------------------------------- #
print("\nRecomputing standardized β_cos + permutation p (n_perm=10,000)...")

b_mag, se_mag, fit_mag = beta_cos_pooled(df, "mean_joint_abs")
b_dir, se_dir, fit_dir = beta_cos_pooled(df, "supp_mean_signed")

obs_mag, p_mag = perm_p_cos("mean_joint_abs",   df)
obs_dir, p_dir = perm_p_cos("supp_mean_signed", df)

# Sanity: perm-observed must equal direct β
assert abs(obs_mag - b_mag) < 1e-9 and abs(obs_dir - b_dir) < 1e-9

print("\n=== MAGNITUDE  (outcome = mean_joint_abs) ===")
print(f"  pooled standardized β_cos = {b_mag:+.4f}   SE = {se_mag:.4f}")
print(f"  permutation p (within-scheme shuffle, {N_PERM}×) = {p_mag:.4f}")

print("\n=== DIRECTION  (outcome = supp_mean_signed) ===")
print(f"  pooled standardized β_cos = {b_dir:+.4f}   SE = {se_dir:.4f}")
print(f"  permutation p (within-scheme shuffle, {N_PERM}×) = {p_dir:.4f}")

# --------------------------------------------------------------------------- #
# Frisch-Waugh-Lovell: build the partial-regression coordinates.
#
# Standardize cos and each outcome over the pooled sample; build a control
# matrix [1 | z(mean_single) | z(max_single) | z(mechanical_push) |
#         d_normTrue | d_per_axis] (i.e., everything except cos in the full
# model). Residualize z(cos) and z(outcome) against the control matrix.
# OLS slope of e_y on e_x equals β_cos^std from the full fit.
# --------------------------------------------------------------------------- #
def standardize(arr):
    arr = np.asarray(arr, float)
    return (arr - arr.mean()) / arr.std(ddof=0)


def residualize(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


CTRL_COLS = ["mean_single_abs", "max_single_abs", "mechanical_push"]
Z_ctrls = StandardScaler().fit_transform(df[CTRL_COLS].values)
d_nT    = (df["scheme"] == "normTrue").astype(float).values
d_pa    = (df["scheme"] == "per_axis").astype(float).values
X_ctrl  = np.column_stack([np.ones(len(df)), Z_ctrls, d_nT, d_pa])

z_cos   = standardize(df["cos"].values)
z_mag   = standardize(df["mean_joint_abs"].values)
z_dir   = standardize(df["supp_mean_signed"].values)

e_cos   = residualize(z_cos, X_ctrl)
e_mag   = residualize(z_mag, X_ctrl)
e_dir   = residualize(z_dir, X_ctrl)

# Slopes from FWL — must match the full-model β_cos exactly.
slope_mag_fwl = float((e_cos @ e_mag) / (e_cos @ e_cos))
slope_dir_fwl = float((e_cos @ e_dir) / (e_cos @ e_cos))
print("\nFWL sanity (residual-OLS slope vs full-model β_cos):")
print(f"  magnitude:  slope = {slope_mag_fwl:+.6f}   β_cos = {b_mag:+.6f}   "
      f"diff = {slope_mag_fwl - b_mag:+.2e}")
print(f"  direction:  slope = {slope_dir_fwl:+.6f}   β_cos = {b_dir:+.6f}   "
      f"diff = {slope_dir_fwl - b_dir:+.2e}")
assert abs(slope_mag_fwl - b_mag) < 1e-9
assert abs(slope_dir_fwl - b_dir) < 1e-9


# --------------------------------------------------------------------------- #
# OLS fit + 95% CI band on the partial-regression line
# --------------------------------------------------------------------------- #
def ols_band(x, y, xs):
    """Slope + 95% CI band for OLS of y on x (residualized inputs)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    sigma2 = (resid @ resid) / (n - 2)
    xbar = x.mean()
    Sxx = ((x - xbar) ** 2).sum()
    y_line = beta[0] + beta[1] * xs
    se_line = np.sqrt(sigma2 * (1.0 / n + (xs - xbar) ** 2 / Sxx))
    tcrit = stats.t.ppf(0.975, df=n - 2)
    return y_line, tcrit * se_line, beta


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
mpl.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":        10,
    "axes.titlesize":   10.5,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
    "axes.linewidth":   0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

PALETTE = {"normFalse": "#E69F00",
           "normTrue":  "#0072B2",
           "per_axis":  "#009E73"}
SCHEME_ORDER = ["normFalse", "normTrue", "per_axis"]
SCHEME_LABEL = {"normFalse": "normFalse (v12, α=4)",
                "normTrue":  "normTrue (v125, α=4.5)",
                "per_axis":  "per_axis (v3, α=4.5)"}

# Shared x-grid in residualized-z space
x_lo = e_cos.min() - 0.15
x_hi = e_cos.max() + 0.15
XS = np.linspace(x_lo, x_hi, 200)

# Shared y-limits across panels so the flat-vs-sloped contrast reads honestly.
# Both residualized outcomes are in z-units with comparable spread.
y_all = np.concatenate([e_mag, e_dir])
y_lo  = y_all.min() - 0.25
y_hi  = y_all.max() + 0.25

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)

# ---------- LEFT: magnitude (partial) ---------------------------------- #
ax = axes[0]
for sch in SCHEME_ORDER:
    mask = (df["scheme"] == sch).values
    ax.scatter(e_cos[mask], e_mag[mask],
               s=22, color=PALETTE[sch], alpha=0.70,
               edgecolor="white", linewidth=0.4,
               label=SCHEME_LABEL[sch])
y_line, half, _ = ols_band(e_cos, e_mag, XS)
ax.fill_between(XS, y_line - half, y_line + half,
                color="0.45", alpha=0.18, linewidth=0)
ax.plot(XS, y_line, color="0.20", lw=1.4)
ax.set_xlim(x_lo, x_hi)
ax.set_ylim(y_lo, y_hi)
ax.set_xlabel("signed cosine | amplitude, push, scheme")
ax.set_ylabel("joint magnitude | controls")
ax.set_title("Magnitude (amplitude controlled):\ngeometry adds little", pad=6)

ax.text(0.03, 0.97,
        f"$\\beta_{{\\cos}}^{{std}}$ = {b_mag:+.3f}\nperm $p$ = {p_mag:.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white",
                  ec="0.6", lw=0.6, alpha=0.9))

# ---------- RIGHT: direction (partial) --------------------------------- #
ax = axes[1]
for sch in SCHEME_ORDER:
    mask = (df["scheme"] == sch).values
    ax.scatter(e_cos[mask], e_dir[mask],
               s=22, color=PALETTE[sch], alpha=0.70,
               edgecolor="white", linewidth=0.4,
               label=SCHEME_LABEL[sch])
y_line, half, _ = ols_band(e_cos, e_dir, XS)
ax.fill_between(XS, y_line - half, y_line + half,
                color="0.45", alpha=0.18, linewidth=0)
ax.plot(XS, y_line, color="0.20", lw=1.4)
ax.set_xlim(x_lo, x_hi)
ax.set_ylim(y_lo, y_hi)
ax.set_xlabel("signed cosine | amplitude, push, scheme")
ax.set_ylabel("signed change | controls")
ax.set_title("Direction (amplitude controlled):\ngeometry predicts reinforce vs. suppress", pad=6)

ax.text(0.03, 0.97,
        f"$\\beta_{{\\cos}}^{{std}}$ = {b_dir:+.3f}\nperm $p$ = {p_dir:.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white",
                  ec="0.6", lw=0.6, alpha=0.9))
# Slope-direction annotation, no y=0 line (residualizing makes y=0 trivial)
ax.text(0.97, 0.05, r"higher cos $\rightarrow$ less suppression",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color="0.30", style="italic")

# Single shared legend on the LEFT panel's lower-right (data-empty quadrant)
axes[0].legend(loc="lower right", frameon=True, framealpha=0.92,
               edgecolor="0.6", handletextpad=0.4, borderpad=0.4)

# --------------------------------------------------------------------------- #
# Save
# --------------------------------------------------------------------------- #
fig.savefig(OUT_PDF, format="pdf")
fig.savefig(OUT_PNG, dpi=200)
print(f"\nWrote {OUT_PDF.relative_to(REPO)}")
print(f"Wrote {OUT_PNG.relative_to(REPO)}")
