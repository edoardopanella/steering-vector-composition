import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm

PAPER_RC = {
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "dejavuserif",
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.labelsize": 11,
    "axes.labelweight": "regular",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.35,
    "grid.linestyle": ":",
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.fontsize": 9.5,
}

NEAR_C = "#3a7ca5"
MOD_C = "#e08e3a"
HIGH_C = "#c0392b"
HIST_C = "#4c6a92"


def _apply_paper_style(rc=None):
    plt.rcParams.update(PAPER_RC)
    if rc:
        plt.rcParams.update(rc)


def summary_stats(pairs_df: pd.DataFrame) -> pd.DataFrame:
    cos = pairs_df["cosine"]
    abs_cos = pairs_df["abs_cosine"]
    stats = {
        "n_pairs": len(cos),
        "mean": cos.mean(),
        "std": cos.std(),
        "min": cos.min(),
        "max": cos.max(),
        "mean_abs": abs_cos.mean(),
        "std_abs": abs_cos.std(),
        "near (<0.2)": (abs_cos < 0.2).sum(),
        "moderate [0.2,0.5)": ((abs_cos >= 0.2) & (abs_cos < 0.5)).sum(),
        "high (>=0.5)": (abs_cos >= 0.5).sum(),
    }
    return pd.DataFrame(stats, index=["value"]).T


def _kde(x, grid):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return np.zeros_like(grid)
    sigma = x.std(ddof=1)
    if sigma == 0:
        return np.zeros_like(grid)
    h = 1.06 * sigma * n ** (-1 / 5)
    diff = (grid[:, None] - x[None, :]) / h
    return np.exp(-0.5 * diff ** 2).sum(axis=1) / (n * h * np.sqrt(2 * np.pi))


def plot_cosine_distribution(pairs_df: pd.DataFrame, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4))
    vals = pairs_df["cosine"].values
    bins = np.linspace(-1, 1, 21)
    ax.hist(vals, bins=bins, density=True, color=HIST_C, edgecolor="white",
            linewidth=0.9, alpha=0.85, zorder=2)
    grid = np.linspace(-1, 1, 400)
    ax.plot(grid, _kde(vals, grid), color="#1f3552", linewidth=1.6, zorder=3)
    ax.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.7, zorder=1)
    ax.axvline(vals.mean(), color=HIGH_C, linewidth=1.2, linestyle="-",
               alpha=0.9, zorder=4, label=f"mean = {vals.mean():.3f}")
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title("Pairwise cosine similarity (signed)")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.legend(loc="upper right")
    return ax


def plot_abs_cosine_distribution(pairs_df: pd.DataFrame, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4))
    vals = pairs_df["abs_cosine"].values
    bins = np.linspace(0, 1, 21)
    ax.hist(vals, bins=bins, density=True, color=MOD_C, edgecolor="white",
            linewidth=0.9, alpha=0.85, zorder=2)
    grid = np.linspace(0, 1, 400)
    ax.plot(grid, _kde(vals, grid), color="#7a4a14", linewidth=1.6, zorder=3)
    ymax = ax.get_ylim()[1]
    for thresh, label, c in [(0.2, "near | mod", NEAR_C), (0.5, "mod | high", HIGH_C)]:
        ax.axvline(thresh, color=c, linewidth=1.0, linestyle="--", alpha=0.85, zorder=1)
        ax.text(thresh + 0.012, ymax * 0.92, label, fontsize=8.5, color=c)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$|\mathrm{cosine}|$")
    ax.set_ylabel("Density")
    ax.set_title("Pairwise absolute cosine similarity")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    return ax


def plot_stratum_counts(strat_df: pd.DataFrame, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    order = ["near", "moderate", "high"]
    counts = strat_df["stratum"].value_counts().reindex(order, fill_value=0)
    bars = ax.bar(counts.index, counts.values,
                  color=[NEAR_C, MOD_C, HIGH_C], edgecolor="white", linewidth=1.0, zorder=2)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + max(counts.values) * 0.02,
                str(int(v)), ha="center", va="bottom", fontsize=10, fontweight="semibold")
    ax.set_xlabel("Stratum")
    ax.set_ylabel("Sampled pairs")
    ax.set_title("Sampled pairs per stratum")
    ax.set_ylim(0, max(counts.values) * 1.18 if counts.values.max() > 0 else 1)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(axis="x", visible=False)
    return ax


def plot_cosine_heatmap(pairs_df: pd.DataFrame, behaviors: list[str], ax=None,
                        annotate: bool = True):
    n = len(behaviors)
    mat = np.full((n, n), np.nan)
    for _, row in pairs_df.iterrows():
        i, j = int(row["i"]), int(row["j"])
        mat[i, j] = row["cosine"]
        mat[j, i] = row["cosine"]
    np.fill_diagonal(mat, 1.0)

    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 7))

    vmax = max(0.05, np.nanmax(np.abs(mat[~np.eye(n, dtype=bool)])))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#dddddd")

    display = np.where(np.eye(n, dtype=bool), np.nan, mat)
    im = ax.imshow(display, cmap=cmap, norm=norm, aspect="equal")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Cosine similarity", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(0.6)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(behaviors, rotation=40, ha="right", fontsize=9.5)
    ax.set_yticklabels(behaviors, fontsize=9.5)
    ax.set_title(f"Pairwise cosine similarity ({n} traits)")
    ax.tick_params(axis="both", which="both", length=0)
    ax.grid(False)

    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#888888")

    if annotate:
        thr = vmax * 0.55
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                v = mat[i, j]
                color = "white" if abs(v) > thr else "#222222"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=7.5, color=color)

    return ax


def top_pairs(pairs_df: pd.DataFrame, n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    most_similar = pairs_df.nlargest(n, "abs_cosine")[
        ["behavior_i", "behavior_j", "cosine", "abs_cosine"]
    ].reset_index(drop=True)
    most_orthogonal = pairs_df.nsmallest(n, "abs_cosine")[
        ["behavior_i", "behavior_j", "cosine", "abs_cosine"]
    ].reset_index(drop=True)
    return most_similar, most_orthogonal


def run_eda(pairs_df: pd.DataFrame, strat_df: pd.DataFrame, behaviors: list[str],
            savepath: str | None = None):
    _apply_paper_style()

    print("=== Summary statistics (all pairs) ===")
    print(summary_stats(pairs_df).to_string())

    most_similar, most_orthogonal = top_pairs(pairs_df)
    print("\n=== Most similar pairs ===")
    print(most_similar.to_string(index=False))
    print("\n=== Most orthogonal pairs ===")
    print(most_orthogonal.to_string(index=False))

    fig = plt.figure(figsize=(14, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], width_ratios=[1, 1])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_cosine_distribution(pairs_df, ax=ax_a)
    plot_abs_cosine_distribution(pairs_df, ax=ax_b)
    plot_stratum_counts(strat_df, ax=ax_c)
    plot_cosine_heatmap(pairs_df, behaviors, ax=ax_d)

    for label, ax in zip("abcd", [ax_a, ax_b, ax_c, ax_d]):
        ax.text(-0.08, 1.04, f"({label})", transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="bottom", ha="left")

    fig.suptitle(
        f"Geometry of {len(behaviors)} validated steering vectors  "
        r"(Llama-3.1-8B-Instruct, layer 16, response-avg diff)",
        fontsize=13, fontweight="semibold", y=1.02,
    )

    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    plt.show()
    return fig
