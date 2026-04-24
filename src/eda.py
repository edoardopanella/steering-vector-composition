import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def summary_stats(pairs_df: pd.DataFrame) -> pd.DataFrame:
    cos = pairs_df["cosine"]
    abs_cos = pairs_df["abs_cosine"]
    stats = {
        "n_pairs":   len(cos),
        "mean":      cos.mean(),
        "std":       cos.std(),
        "min":       cos.min(),
        "max":       cos.max(),
        "mean_abs":  abs_cos.mean(),
        "std_abs":   abs_cos.std(),
        "near (<0.2)":     (abs_cos < 0.2).sum(),
        "moderate [0.2,0.5)": ((abs_cos >= 0.2) & (abs_cos < 0.5)).sum(),
        "high (>=0.5)":    (abs_cos >= 0.5).sum(),
    }
    return pd.DataFrame(stats, index=["value"]).T


def plot_cosine_distribution(pairs_df: pd.DataFrame, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pairs_df["cosine"], bins=20, edgecolor="white", color="steelblue")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of pairwise cosine similarities")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    return ax


def plot_abs_cosine_distribution(pairs_df: pd.DataFrame, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pairs_df["abs_cosine"], bins=20, edgecolor="white", color="darkorange")
    for thresh, label in [(0.2, "0.2"), (0.5, "0.5")]:
        ax.axvline(thresh, color="black", linewidth=0.8, linestyle="--")
        ax.text(thresh + 0.01, ax.get_ylim()[1] * 0.9, label, fontsize=8)
    ax.set_xlabel("|Cosine similarity|")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of |cosine| with stratum boundaries")
    return ax


def plot_stratum_counts(strat_df: pd.DataFrame, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    order = ["near", "moderate", "high"]
    counts = strat_df["stratum"].value_counts().reindex(order, fill_value=0)
    ax.bar(counts.index, counts.values, color=["#4c9be8", "#f0a500", "#e84c4c"], edgecolor="white")
    ax.set_xlabel("Stratum")
    ax.set_ylabel("Count")
    ax.set_title("Sampled pairs per stratum")
    return ax


def plot_cosine_heatmap(pairs_df: pd.DataFrame, behaviors: list[str], ax=None):
    n = len(behaviors)
    mat = np.full((n, n), np.nan)
    for _, row in pairs_df.iterrows():
        i, j = int(row["i"]), int(row["j"])
        mat[i, j] = row["cosine"]
        mat[j, i] = row["cosine"]
    np.fill_diagonal(mat, 1.0)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(behaviors, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(behaviors, fontsize=8)
    ax.set_title("Pairwise cosine similarity matrix")
    return ax


def top_pairs(pairs_df: pd.DataFrame, n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    most_similar = pairs_df.nlargest(n, "abs_cosine")[
        ["behavior_i", "behavior_j", "cosine", "abs_cosine"]
    ].reset_index(drop=True)
    most_orthogonal = pairs_df.nsmallest(n, "abs_cosine")[
        ["behavior_i", "behavior_j", "cosine", "abs_cosine"]
    ].reset_index(drop=True)
    return most_similar, most_orthogonal


def run_eda(pairs_df: pd.DataFrame, strat_df: pd.DataFrame, behaviors: list[str]):
    print("=== Summary statistics (all pairs) ===")
    print(summary_stats(pairs_df).to_string())

    most_similar, most_orthogonal = top_pairs(pairs_df)
    print("\n=== Most similar pairs ===")
    print(most_similar.to_string(index=False))
    print("\n=== Most orthogonal pairs ===")
    print(most_orthogonal.to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    plot_cosine_distribution(pairs_df, ax=axes[0, 0])
    plot_abs_cosine_distribution(pairs_df, ax=axes[0, 1])
    plot_stratum_counts(strat_df, ax=axes[1, 0])
    plot_cosine_heatmap(pairs_df, behaviors, ax=axes[1, 1])
    fig.tight_layout()
    plt.show()
