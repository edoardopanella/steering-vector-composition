'''

Calculate the {pearson, spearman, other?} correlation between
human scores and judge scores

'''


# DATA: require scores are ordered so that score_human_i is scoring
#       the same text that score_judge_i is scoring, for all i

from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

OUT_PATH = Path("results/human_eval/human_judge_correlation.md")


# TODO: import ordered dataset

scores_human = None
scores_judge = None


def correlate(scores_human, scores_judge):
    h = np.asarray(scores_human, dtype=float)
    j = np.asarray(scores_judge, dtype=float)
    assert h.shape == j.shape, "human/judge scores must align 1:1"

    pearson_r, pearson_p = pearsonr(h, j)
    spearman_r, spearman_p = spearmanr(h, j)

    return {
        "n": len(h),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }


def to_markdown(res):
    return (
        "# Human vs. Judge Score Correlation\n\n"
        f"- **N pairs:** {res['n']}\n\n"
        "| Metric | Coefficient | p-value |\n"
        "|---|---|---|\n"
        f"| Pearson r | {res['pearson_r']:.3f} | {res['pearson_p']:.3g} |\n"
        f"| Spearman rho | {res['spearman_r']:.3f} | {res['spearman_p']:.3g} |\n"
    )


if __name__ == "__main__":
    res = correlate(scores_human, scores_judge)
    md = to_markdown(res)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(md)
    print(md)
    print(f"Saved to {OUT_PATH}")



