"""
Steering vector injection and steered text generation (Proposal §5, eq. 3).
Joint:          h^(L) <- h^(L) + alpha_i * v_i + alpha_j * v_j
"""

import torch
from transformer_lens import HookedTransformer

from src.model_utils import hook_name

def make_steering_hook(steering_vector: torch.Tensor):
    """Return a hook that adds a precomposed steering vector to the last-position activation."""
    def hook_fn(activation, hook):
        activation[:, -1, :] = activation[:, -1, :] + steering_vector
        return activation
    return hook_fn


def compose_steering_vector(
    vectors_alphas: list[tuple[torch.Tensor, float]],
    normalize: bool = False,
) -> torch.Tensor:
    """Sum alpha_i * v_i. If normalize=True, rescale the result to unit L2 norm."""
    composed = sum(alpha * v for v, alpha in vectors_alphas)
    if normalize and composed.norm() > 0:
        composed = composed / composed.norm()
    return composed


def _format_chat(model: HookedTransformer, prompt: str) -> str:
    """Wrap a raw user prompt in the model's chat template (instruct mode)."""
    return model.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

def generate_steering(
    model: HookedTransformer,
    prompt: str,
    layer: int,
    steering_vector: torch.Tensor,
    max_new_tokens: int = 60,
    temperature: float = 0.7,
) -> str:
    """Generate text with a precomposed steering vector applied at the given layer."""
    formatted = _format_chat(model, prompt)
    tokens = model.to_tokens(formatted, prepend_bos=False)
    prompt_len = tokens.shape[1]
    hook = make_steering_hook(steering_vector)
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

    return model.tokenizer.decode(tokens[0, prompt_len:], skip_special_tokens=True)


def apply_steering_batched(
    model: HookedTransformer,
    prompts: list[str],
    layer: int,
    steering_vector: torch.Tensor,
    max_new_tokens: int = 60,
    temperature: float = 0.7,
    batch_size: int = 8,
) -> list[str]:
    """Batched steered generation with a precomposed steering vector.

    Pads prompts to the same length within each batch, runs generation in
    parallel, then decodes all sequences.  Finished sequences are masked so
    they no longer receive new tokens after hitting EOS.
    """
    device = next(model.parameters()).device
    pad_token_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
    hook = make_steering_hook(steering_vector)
    name = hook_name(layer)

    all_outputs: list[str] = []

    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start : batch_start + batch_size]

        # Apply chat template, then tokenise and left-pad to the longest sequence
        formatted = [_format_chat(model, p) for p in batch_prompts]
        encoded = [
            model.to_tokens(p, prepend_bos=False).squeeze(0) for p in formatted
        ]
        max_len = max(t.shape[0] for t in encoded)
        padded = torch.stack(
            [
                torch.cat(
                    [
                        torch.full(
                            (max_len - t.shape[0],),
                            pad_token_id,
                            dtype=torch.long,
                            device=device,
                        ),
                        t.to(device),
                    ]
                )
                for t in encoded
            ]
        )  # (B, max_len)

        finished = torch.zeros(len(batch_prompts), dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            if finished.all():
                break
            with torch.no_grad():
                logits = model.run_with_hooks(
                    padded,
                    fwd_hooks=[(name, hook)],
                    return_type="logits",
                )
            next_token_logits = logits[:, -1, :]  # (B, vocab)
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Pad finished sequences instead of appending real tokens
            next_tokens[finished] = pad_token_id
            padded = torch.cat([padded, next_tokens], dim=1)
            finished |= next_tokens.squeeze(1) == model.tokenizer.eos_token_id

        # With left-padding, the prompt ends at column max_len in every row.
        for seq in padded:
            all_outputs.append(
                model.tokenizer.decode(seq[max_len:], skip_special_tokens=True)
            )

    return all_outputs
