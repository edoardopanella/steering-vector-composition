'''

Calculate the {pearson, spearman, other?} correlation between
human scores and judge scores

'''


# DATA: require scores are ordered so that score_human_i is scoring
#       the same text that score_judge_i is scoring, for all i

import numpy as np
from scipy.stats import pearsonr, spearmanr


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


if __name__ == "__main__":
    res = correlate(scores_human, scores_judge)
    print(f"n            = {res['n']}")
    print(f"Pearson  r   = {res['pearson_r']:.3f} (p = {res['pearson_p']:.3g})")
    print(f"Spearman rho = {res['spearman_r']:.3f} (p = {res['spearman_p']:.3g})")
