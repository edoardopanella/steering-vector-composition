from transformer_lens import HookedTransformer
import torch


def load_model(model_name: str, device: str = "cpu") -> HookedTransformer:
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    # `from_pretrained_no_processing` skips LayerNorm folding / weight centering
    # so internal residual streams match raw HF activations — required for steering
    # with vectors extracted via the Anthropic pipeline (`response_avg_diff`).
    model = HookedTransformer.from_pretrained_no_processing(model_name, device=device, dtype=dtype)
    model.eval()
    return model


def hook_name(layer: int) -> str:
    return f"blocks.{layer}.hook_resid_post"


def n_layers(model: HookedTransformer) -> int:
    return model.cfg.n_layers


def get_device(model: HookedTransformer) -> torch.device:
    return next(model.parameters()).device
