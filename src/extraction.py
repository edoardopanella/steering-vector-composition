"""
Activation extraction and CAA steering vector construction (Proposal §2–3).

Core formula (eq. 1):
    v_b^(L) = mean(pos_acts) - mean(neg_acts)
"""

import torch
from transformer_lens import HookedTransformer

from .model_utils import hook_name


def get_activation(model: HookedTransformer, prompt: str, layer: int) -> torch.Tensor:
    """Extract residual-stream activation at the final token position for one prompt."""
    name = hook_name(layer)
    tokens = model.to_tokens(prompt, prepend_bos=True)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=name)
    return cache[name][0, -1, :].clone()


def extract_activations(
    model: HookedTransformer,
    pairs: list[dict],
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (pos_acts, neg_acts) tensors of shape [N, d_model] for a list of pairs."""
    pos_acts, neg_acts = [], []
    for pair in pairs:
        pos_acts.append(get_activation(model, pair["positive"], layer))
        neg_acts.append(get_activation(model, pair["negative"], layer))
    return torch.stack(pos_acts), torch.stack(neg_acts)


def compute_steering_vector(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """CAA mean-difference vector (eq. 1). Optionally unit-normalized."""
    vector = pos_acts.mean(0) - neg_acts.mean(0)
    if normalize:
        vector = vector / vector.norm()
    return vector


def extract_steering_vector(
    model: HookedTransformer,
    train_pairs: list[dict],
    layer: int,
    normalize: bool = True,
) -> torch.Tensor:
    """End-to-end: extract activations and return the CAA vector for one layer."""
    pos_acts, neg_acts = extract_activations(model, train_pairs, layer)
    return compute_steering_vector(pos_acts, neg_acts, normalize=normalize)


def extract_all_layers(
    model: HookedTransformer,
    train_pairs: list[dict],
    normalize: bool = True,
) -> dict[int, torch.Tensor]:
    """Extract one steering vector per layer using one forward pass per pair.

    Runs 2 * len(train_pairs) forward passes total (one per positive, one per
    negative prompt), caching all residual-stream layers simultaneously.
    This is n_layers times faster than calling extract_steering_vector per layer.
    """
    n = model.cfg.n_layers
    names = [hook_name(layer) for layer in range(n)]

    pos_acts: list[list[torch.Tensor]] = [[] for _ in range(n)]
    neg_acts: list[list[torch.Tensor]] = [[] for _ in range(n)]

    for pair in train_pairs:
        for prompt, store in [(pair["positive"], pos_acts), (pair["negative"], neg_acts)]:
            tokens = model.to_tokens(prompt, prepend_bos=True)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=names)
            for layer in range(n):
                store[layer].append(cache[hook_name(layer)][0, -1, :].clone())

    return {
        layer: compute_steering_vector(
            torch.stack(pos_acts[layer]),
            torch.stack(neg_acts[layer]),
            normalize=normalize,
        )
        for layer in range(n)
    }
