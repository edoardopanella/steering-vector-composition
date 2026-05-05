"""
Steering vector composition + injection on raw HuggingFace transformers.

Joint:  h^(L) <- h^(L) + alpha * direction
where `direction` is built by `compose_steering_vector` from per-vector weights
(unit-normalised then re-normalised) and `alpha` is the calibrated injection
magnitude (E10.3/E10.4 protocol).

Generation reuses `src.anthropic_repl.generation.generate_batch` so the joint
pipeline runs on the same HF + forward-hook code path as the validated alpha
sweep at L=17.
"""

import torch

from src.anthropic_repl.generation import generate_batch


def compose_steering_vector(
    vectors_weights: list[tuple[torch.Tensor, float]],
    alpha: float,
    normalize: bool = True,
) -> torch.Tensor:
    """Build a steering vector at fixed injected magnitude `alpha` whose
    direction is selected by the per-vector weights.

    Protocol (E10.3/E10.4) when normalize=True:
        v_i_hat = v_i / ||v_i||                  (unit-normalise each input vector)
        direction = sum_i w_i * v_i_hat          (weighted sum on the unit sphere)
        direction = direction / ||direction||    (re-normalise composed direction)
        return alpha * direction                 (fixed-magnitude injection)

    Weights select direction only; magnitude is held constant across conditions.
    With normalize=False, returns alpha * sum_i w_i * v_i (no normalisation).
    """
    if not normalize:
        return alpha * sum(w * v for v, w in vectors_weights)

    unit_terms = [w * (v / v.norm()) for v, w in vectors_weights if v.norm() > 0]
    if not unit_terms:
        return torch.zeros_like(vectors_weights[0][0])
    direction = sum(unit_terms)
    dn = direction.norm()
    if dn == 0:
        return torch.zeros_like(direction)
    return alpha * (direction / dn)


def apply_steering_batched(
    model,
    tokenizer,
    prompts: list[str],
    layer: int,
    steering_vector: torch.Tensor,
    max_new_tokens: int = 60,
    temperature: float = 0.7,
    batch_size: int = 8,
) -> list[str]:
    """Batched steered generation with a precomposed steering vector.

    `layer` follows the Anthropic-pipeline convention (persona-stack index =
    `hidden_states[layer]`). The forward hook fires on the output of
    transformer block `layer-1`, so the perturbation lands in the same residual
    stream the vectors were extracted from.

    A zero-norm steering vector (e.g. setting=(0,0)) installs no hook, giving
    a clean unsteered baseline.
    """
    conversations = [[{"role": "user", "content": p}] for p in prompts]
    if steering_vector.norm().item() > 0:
        steering = (steering_vector, layer - 1, 1.0, "response")
    else:
        steering = None
    _, answers = generate_batch(
        model, tokenizer, conversations,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        batch_size=batch_size,
        steering=steering,
    )
    return answers
