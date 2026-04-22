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
    """Extract one steering vector per layer. Returns {layer: vector}."""
    n = model.cfg.n_layers
    return {
        layer: extract_steering_vector(model, train_pairs, layer, normalize=normalize)
        for layer in range(n)
    }
