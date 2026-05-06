"""
Activation extraction and CAA steering vector construction (Proposal §2–3).

Core formula (eq. 1):
    v_b^(L) = mean(pos_acts) - mean(neg_acts)
"""

import torch
from transformer_lens import HookedTransformer

from .model_utils import hook_name

# ---------------------------------------------------------------------------
# Persona-vector extraction (Chen et al. 2025 replication)
# ---------------------------------------------------------------------------


def extract_persona_vector(
    model: HookedTransformer,
    artifact: dict,
    layer: int,
    n_rollouts_per_combo: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    normalize: bool = True,
    seed: int = 42,
) -> torch.Tensor:
    """Persona-vector extraction following Chen et al. (2025).

    For each (priming pair × question × condition) combination, generates
    `n_rollouts_per_combo` responses from the model (unsteered, temperature
    sampling), then re-runs the full sequence through run_with_cache and
    mean-pools residual-stream activations over the response-token positions
    at `layer`. The persona vector is mean(pos_acts) - mean(neg_acts).

    v1 simplifications vs the paper:
      - n_rollouts_per_combo=1 (paper uses 10). Increase if vectors are noisy.
      - Judge filtering is skipped (cuts API cost to zero). Add as follow-up
        if the resulting vectors produce no steering effect.

    Args:
        model: loaded HookedTransformer (Llama-3.1-8B-Instruct).
        artifact: dict with keys:
            "instruction": list of {"pos": str, "neg": str}
            "questions":   list of str
        layer: residual-stream layer to extract activations from.
        n_rollouts_per_combo: generations per (pair, question, condition).
        max_new_tokens: maximum tokens to generate per rollout.
        temperature: sampling temperature (1.0 matches the paper).
        normalize: divide the final vector by its L2 norm.
        seed: torch manual seed for reproducible generation.

    Returns:
        Persona vector as a 1-D tensor of shape [d_model], on CPU.
    """
    # --- validate artifact schema ---
    instructions = artifact.get("instruction")
    questions = artifact.get("questions")
    if not isinstance(instructions, list) or len(instructions) == 0:
        raise ValueError("artifact['instruction'] must be a non-empty list")
    for i, pair in enumerate(instructions):
        if not isinstance(pair, dict) or not isinstance(pair.get("pos"), str) or not isinstance(pair.get("neg"), str):
            raise ValueError(f"artifact['instruction'][{i}] must be a dict with 'pos' and 'neg' string keys")
    if not isinstance(questions, list) or len(questions) == 0:
        raise ValueError("artifact['questions'] must be a non-empty list of strings")
    if not all(isinstance(q, str) for q in questions):
        raise ValueError("artifact['questions'] must be a list of strings")

    torch.manual_seed(seed)
    hook_nm = hook_name(layer)
    device = next(model.parameters()).device

    def build_prompt(system_text: str, user_text: str) -> str:
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user",   "content": user_text},
        ]
        try:
            return model.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Manual Llama 3.1 chat template fallback
            return (
                f"<|begin_of_text|>"
                f"<|start_header_id|>system<|end_header_id|>\n\n{system_text}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{user_text}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )

    def generate_and_extract(prompt_str: str) -> torch.Tensor | None:
        """Generate one response and return mean-pooled layer activations."""
        # Llama's chat template includes <|begin_of_text|> (BOS), so we must
        # NOT prepend an extra BOS when tokenizing.
        prompt_ids = model.to_tokens(prompt_str, prepend_bos=False)  # [1, prompt_len]
        prompt_len = prompt_ids.shape[1]

        tokens = prompt_ids.clone()
        eos_id = model.tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = model(tokens, return_type="logits")
            next_logits = logits[0, -1, :]
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [1]
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == eos_id:
                break

        response_len = tokens.shape[1] - prompt_len
        if response_len == 0:
            return None

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_nm)

        acts = cache[hook_nm][0]          # [full_len, d_model]
        response_acts = acts[prompt_len:] # [response_len, d_model]
        return response_acts.mean(0).clone().cpu()

    pos_activations: list[torch.Tensor] = []
    neg_activations: list[torch.Tensor] = []

    total = len(instructions) * len(questions) * 2 * n_rollouts_per_combo
    done = 0
    print(f"  Extracting persona vector: {len(instructions)} pairs × {len(questions)} questions "
          f"× 2 conditions × {n_rollouts_per_combo} rollout(s) = {total} generations")

    for pair in instructions:
        for question in questions:
            for condition, store in [("pos", pos_activations), ("neg", neg_activations)]:
                system_text = pair[condition]
                prompt_str = build_prompt(system_text, question)
                for _ in range(n_rollouts_per_combo):
                    act = generate_and_extract(prompt_str)
                    if act is not None:
                        store.append(act)
                    done += 1
                    if done % 20 == 0:
                        print(f"    [{done}/{total}] pos={len(pos_activations)} neg={len(neg_activations)}")

    if not pos_activations or not neg_activations:
        raise RuntimeError("No valid activations collected — all generations may have been empty.")

    pos_mean = torch.stack(pos_activations).mean(0)
    neg_mean = torch.stack(neg_activations).mean(0)
    vector = pos_mean - neg_mean
    if normalize:
        vector = vector / vector.norm()
    return vector.cpu()


# ---------------------------------------------------------------------------
# CAA extraction helpers
# ---------------------------------------------------------------------------


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
