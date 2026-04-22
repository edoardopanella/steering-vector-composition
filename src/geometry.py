"""
Gram matrix computation, pair selection, and composition quality metrics (Proposal §5).

Gram matrix (eq. 2):   G_ij = (v_i . v_j) / (||v_i|| ||v_j||)
Composition quality (eq. 4):
    Q(i,j) = 0.5 * (E_i(1,1)/max(E_i(1,0), eps) + E_j(1,1)/max(E_j(0,1), eps))
"""

import torch


def compute_gram_matrix(vectors: list[torch.Tensor]) -> torch.Tensor:
    """Return the [N, N] pairwise cosine similarity matrix for a list of vectors."""
    V = torch.stack(vectors)                    # [N, d]
    V = V / V.norm(dim=1, keepdim=True)         # row-wise unit normalize
    return V @ V.T                              # [N, N]


def select_pairs(
    gram_matrix: torch.Tensor,
    n_near: int = 14,
    n_moderate: int = 13,
    n_high: int = 13,
) -> list[tuple[int, int]]:
    """Stratified pair selection by |G_ij| (near-orthogonal / moderate / high).

    Returns a list of (i, j) index pairs with i < j.
    Strata: near |G|<0.2, moderate 0.2<=|G|<0.5, high |G|>=0.5.
    If a stratum is too small the remainder is filled from the next stratum.
    """
    n = gram_matrix.shape[0]
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    abs_cosines = {(i, j): gram_matrix[i, j].abs().item() for i, j in all_pairs}

    near = [(i, j) for (i, j) in all_pairs if abs_cosines[(i, j)] < 0.2]
    moderate = [(i, j) for (i, j) in all_pairs if 0.2 <= abs_cosines[(i, j)] < 0.5]
    high = [(i, j) for (i, j) in all_pairs if abs_cosines[(i, j)] >= 0.5]

    def _fill(primary, fallback, target):
        selected = primary[:target]
        shortage = target - len(selected)
        if shortage > 0:
            selected += fallback[:shortage]
        return selected

    selected = (
        _fill(near, moderate, n_near)
        + _fill(moderate, near + high, n_moderate)
        + _fill(high, moderate, n_high)
    )
    # deduplicate while preserving order
    seen = set()
    out = []
    for p in selected:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def composition_quality(
    score_i_joint: float,
    score_j_joint: float,
    score_i_solo: float,
    score_j_solo: float,
    eps: float = 1e-6,
) -> float:
    """Continuous composition quality Q(i,j) from eq. 4."""
    ratio_i = score_i_joint / max(score_i_solo, eps)
    ratio_j = score_j_joint / max(score_j_solo, eps)
    return 0.5 * (ratio_i + ratio_j)


def classify_regime(
    score_i_joint: float,
    score_j_joint: float,
    score_i_solo: float,
    score_j_solo: float,
    additive_threshold: float = 0.8,
    emergent_flag: bool = False,
) -> str:
    """Classify a (1,1) composition outcome into one of four regimes.

    additive        — both behaviors expressed at >= threshold of solo strength
    dominant        — exactly one exceeds threshold
    suppressive     — neither exceeds threshold, no emergent flag
    emergent        — judge flagged off-target content above threshold rate
    """
    ratio_i = score_i_joint / max(score_i_solo, 1e-6)
    ratio_j = score_j_joint / max(score_j_solo, 1e-6)

    if emergent_flag:
        return "emergent"
    i_ok = ratio_i >= additive_threshold
    j_ok = ratio_j >= additive_threshold
    if i_ok and j_ok:
        return "additive"
    if i_ok or j_ok:
        return "dominant"
    return "suppressive"


def subspace_overlap(
    v_i: torch.Tensor,
    v_j: torch.Tensor,
    top_k_singular_vecs: torch.Tensor,
) -> float:
    """Cosine similarity between |v_i projected onto top-k PCs| and |v_j projected|.

    top_k_singular_vecs: [d, k] right singular vectors of the full behavior matrix V.
    """
    proj_i = (top_k_singular_vecs.T @ v_i).abs()
    proj_j = (top_k_singular_vecs.T @ v_j).abs()
    cos = torch.dot(proj_i, proj_j) / (proj_i.norm() * proj_j.norm() + 1e-12)
    return cos.item()
