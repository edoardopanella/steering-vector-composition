import torch
from transformer_lens import HookedTransformer
from .model_utils import hook_name


def make_all_position_hook(vectors_alphas):
    """Hook that adds steering vectors at ALL sequence positions."""
    def hook_fn(activation, hook):
        for vector, alpha in vectors_alphas:
            activation = activation + alpha * vector  # broadcasts over batch and seq dims
        return activation
    return hook_fn


def compute_logprob_delta(
        model: HookedTransformer,
        question: str,
        trait_completion: str,
        non_trait_completion: str,
        layer: int,
        vectors_alphas: list[tuple[torch.Tensor, float]],
) -> float:
    """Return log P(trait_completion|question,steering) - log P(non_trait_completion|question,steering)."""
    if not trait_completion or not non_trait_completion:
        return 0.0

    # Tokenize the full sequences
    # Note: tokenizing "A"+"B" may differ from tokenizing "AB" due to boundary merges.
    # Assumption: question tokenization is identical whether tokenized alone or as a prefix.
    full_trait_tokens = model.to_tokens(question + trait_completion, prepend_bos=True)        # [1, seq_len_trait]
    full_non_trait_tokens = model.to_tokens(question + non_trait_completion, prepend_bos=True) # [1, seq_len_non]

    question_tokens = model.to_tokens(question, prepend_bos=True)  # [1, q_len]
    q_len = question_tokens.shape[1]

    def _completion_logprob(tokens: torch.Tensor) -> float:
        with torch.no_grad():
            if vectors_alphas:
                logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_name(layer), make_all_position_hook(vectors_alphas))],
                    return_type="logits",
                )
            else:
                logits = model(tokens, return_type="logits")

        # logits: [1, seq_len, vocab_size]
        log_probs = torch.log_softmax(logits, dim=-1)

        completion_token_ids = tokens[0, q_len:]                    # [completion_len]
        completion_logprobs = log_probs[0, q_len - 1 : -1, :]      # [completion_len, vocab_size]
        token_logprobs = completion_logprobs.gather(
            1, completion_token_ids.unsqueeze(1)
        ).squeeze(1)                                                 # [completion_len]
        return token_logprobs.sum().item()

    logprob_trait = _completion_logprob(full_trait_tokens)
    logprob_non_trait = _completion_logprob(full_non_trait_tokens)
    return logprob_trait - logprob_non_trait



