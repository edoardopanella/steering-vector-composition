import numpy as np

def compute_gram_matrix(vectors: np.ndarray, shape: tuple) -> np.ndarray:
    """Return the [N, N] pairwise cosine similarity matrix for a list of vectors."""
    if vectors.shape != shape:
        raise ValueError("Input vectors do not match the expected shape.")
    V = vectors  # already normalised to unit length
    return V @ V.T  # [N, N]