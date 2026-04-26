"""
Steering vector injection and steered text generation (Proposal §5, eq. 3).
Joint:          h^(L) <- h^(L) + alpha_i * v_i + alpha_j * v_j
"""

import torch
from transformer_lens import HookedTransformer

from src.model_utils import hook_name



def make_joint_hook(vectors_alphas: list[tuple[torch.Tensor, float]]):
    """Return a hook that applies multiple (vector, alpha) pairs simultaneously."""
    def hook_fn(activation, hook):
        for vector, alpha in vectors_alphas:
            activation[:, -1, :] = activation[:, -1, :] + alpha * vector
        return activation
    return hook_fn

def generate_joint_steering(
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


def apply_joint_steering_batched(
    model: HookedTransformer,
    prompts: list[str],
    layer: int,
    vectors_alphas: list[tuple[torch.Tensor, float]],
    max_new_tokens: int = 60,
    temperature: float = 0.7,
    batch_size: int = 8,
) -> list[str]:
    """Batched version of apply_joint_steering for GPU throughput.

    Pads prompts to the same length within each batch, runs generation in
    parallel, then decodes all sequences.  Finished sequences are masked so
    they no longer receive new tokens after hitting EOS.
    """
    device = next(model.parameters()).device
    pad_token_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
    hook = make_joint_hook(vectors_alphas)
    name = hook_name(layer)

    all_outputs: list[str] = []

    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start : batch_start + batch_size]

        # Tokenise individually, then left-pad to the longest sequence in the batch
        encoded = [
            model.to_tokens(p, prepend_bos=True).squeeze(0) for p in batch_prompts
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

        for seq in padded:
            all_outputs.append(model.tokenizer.decode(seq, skip_special_tokens=True))

    return all_outputs
