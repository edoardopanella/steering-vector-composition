"""
Steering vector injection and steered text generation (Proposal §5, eq. 3).

Single-vector:  h^(L) <- h^(L) + alpha * v
Joint:          h^(L) <- h^(L) + alpha_i * v_i + alpha_j * v_j
"""

import torch
from transformer_lens import HookedTransformer

from .model_utils import hook_name


def make_hook(vector: torch.Tensor, alpha: float):
    """Return a TransformerLens hook that adds alpha*vector to the last token position."""
    def hook_fn(activation, hook):
        activation[:, -1, :] = activation[:, -1, :] + alpha * vector
        return activation
    return hook_fn


def make_joint_hook(vectors_alphas: list[tuple[torch.Tensor, float]]):
    """Return a hook that applies multiple (vector, alpha) pairs simultaneously."""
    def hook_fn(activation, hook):
        for vector, alpha in vectors_alphas:
            activation[:, -1, :] = activation[:, -1, :] + alpha * vector
        return activation
    return hook_fn


def generate_steered(
    model: HookedTransformer,
    prompt: str,
    layer: int,
    steering_vector: torch.Tensor,
    alpha: float,
    max_new_tokens: int = 60,
    temperature: float = 0.7,
) -> str:
    """Generate text with a single steering vector applied at every forward pass."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    hook = make_hook(steering_vector, alpha)
    name = hook_name(layer)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(name, hook)],
                return_type="logits",
            )
        next_token_logits = logits[0, -1, :]
        probs = torch.softmax(next_token_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
        if next_token.item() == model.tokenizer.eos_token_id:
            break

    return model.tokenizer.decode(tokens[0], skip_special_tokens=True)


def generate_steered_batch(
    model: HookedTransformer,
    prompt: str,
    layer: int,
    steering_vector: torch.Tensor,
    alpha: float,
    n: int = 1,
    max_new_tokens: int = 60,
    temperature: float = 0.7,
) -> list[str]:
    """Generate n completions in parallel via batched inference.

    Repeats the prompt n times along the batch dimension so all n samples
    run through the model in a single forward pass per token step.
    Roughly n× faster than calling generate_steered n times sequentially.
    """
    tokens = model.to_tokens(prompt, prepend_bos=True).repeat(n, 1)  # [n, seq_len]
    hook = make_hook(steering_vector, alpha)
    name = hook_name(layer)
    done = torch.zeros(n, dtype=torch.bool, device=tokens.device)

    for _ in range(max_new_tokens):
        if done.all():
            break
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=[(name, hook)], return_type="logits")
        probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)  # [n, vocab]
        next_tokens = torch.multinomial(probs, num_samples=1)           # [n, 1]
        next_tokens[done] = model.tokenizer.eos_token_id
        tokens = torch.cat([tokens, next_tokens], dim=1)
        done = done | (next_tokens.squeeze(-1) == model.tokenizer.eos_token_id)

    return [model.tokenizer.decode(tokens[i], skip_special_tokens=True) for i in range(n)]


def apply_joint_steering(
    model: HookedTransformer,
    prompt: str,
    layer: int,
    vectors_alphas: list[tuple[torch.Tensor, float]],
    max_new_tokens: int = 60,
    temperature: float = 0.7,
) -> str:
    """Generate text with multiple steering vectors applied jointly (eq. 3).

    vectors_alphas: list of (vector, alpha) pairs, e.g. [(v_i, 1.0), (v_j, 1.0)]
    Pass alpha=0.0 for a behavior to get its individual baseline without removing it from the hook.
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    hook = make_joint_hook(vectors_alphas)
    name = hook_name(layer)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(name, hook)],
                return_type="logits",
            )
        next_token_logits = logits[0, -1, :]
        probs = torch.softmax(next_token_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
        if next_token.item() == model.tokenizer.eos_token_id:
            break

    return model.tokenizer.decode(tokens[0], skip_special_tokens=True)
