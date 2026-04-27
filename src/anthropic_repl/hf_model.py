"""
HuggingFace model loader and steering hook for the Anthropic-pipeline replication.

Deliberately NOT TransformerLens. The Anthropic pipeline relies on three things that
are easier on raw HF transformers:
  - output_hidden_states=True returning the full [n_layers+1, ...] stack in one pass
  - tokenizer.apply_chat_template (Qwen / Llama-3 / etc., handled by AutoTokenizer)
  - register_forward_hook on model.model.layers[L] for response-only steering

Llama-3.1-8B-Instruct: hidden_size=4096, num_hidden_layers=32 -> output_hidden_states
returns 33 tensors (index 0 = embeddings, indices 1..32 = post-block residual streams).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_hf_model(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device_map
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return model, tok


def _resolve_layer_list(model: torch.nn.Module):
    # Llama / Mistral
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # Qwen / fallback
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError(
        "Could not locate the transformer block list on the model. "
        "Expected model.model.layers (Llama) or model.transformer.h (Qwen/GPT2)."
    )


@contextmanager
def steering_hook(
    model: torch.nn.Module,
    vector: torch.Tensor,
    layer_idx: int,
    coeff: float = 1.0,
    positions: str = "response",
    prompt_len: int | None = None,
) -> Iterator[None]:
    """
    Register a forward hook on transformer block `layer_idx` that adds
    `coeff * vector` to the block's output residual stream.

    positions:
        "all"      — every token position (used during teacher-forced extraction).
        "response" — last token only. During autoregressive decoding this is the
                     newly-generated token at every step, so the cumulative effect
                     is "perturb every response token". Matches Anthropic's
                     ActivationSteerer(positions="response") for inference.
        "prompt"   — every token in the prefill. Requires prompt_len for masked steering.

    The hook fires on the *output* of model.model.layers[layer_idx], which equals
    `hidden_states[layer_idx + 1]` from output_hidden_states=True. So if you build a
    vector from output_hidden_states[L], inject it at layer_idx = L - 1.
    """
    layers = _resolve_layer_list(model)
    if not (-len(layers) <= layer_idx < len(layers)):
        raise IndexError(f"layer_idx {layer_idx} out of range for {len(layers)} layers")
    if positions not in {"all", "response", "prompt"}:
        raise ValueError(f"positions must be one of all/response/prompt, got {positions!r}")

    p = next(model.parameters())
    v = vector.to(dtype=p.dtype, device=p.device).reshape(-1)
    if v.numel() != model.config.hidden_size:
        raise ValueError(
            f"Vector length {v.numel()} != hidden_size {model.config.hidden_size}"
        )
    delta = coeff * v

    def _hook(_module, _inputs, outputs):
        # Llama returns a tuple (hidden_states, ...).
        if isinstance(outputs, tuple):
            hs = outputs[0]
            rest = outputs[1:]
        else:
            hs = outputs
            rest = None

        if positions == "all":
            new_hs = hs + delta.to(hs.device)
        elif positions == "response":
            # During decoding hs.shape[1] == 1 after the first prefill step;
            # write into the last position uniformly.
            new_hs = hs.clone()
            new_hs[:, -1, :] += delta.to(hs.device)
        else:  # "prompt"
            if prompt_len is None:
                raise RuntimeError("positions='prompt' requires prompt_len")
            new_hs = hs.clone()
            new_hs[:, :prompt_len, :] += delta.to(hs.device)

        return (new_hs, *rest) if rest is not None else new_hs

    handle = layers[layer_idx].register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()
