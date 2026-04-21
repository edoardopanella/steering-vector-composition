import torch
from transformer_lens import HookedTransformer


# 1) Load a small model
model = HookedTransformer.from_pretrained("gpt2", device="cpu")

# 2) Write a prompt
prompt = "my favourite song is"

'''# 3) Tokenize it
tokens = model.to_tokens(prompt)

# 4) Run once and cache activations
logits, cache = model.run_with_cache(tokens)

# 5) Look at the model's next-token prediction
next_token = logits[0, -1].argmax(dim=-1)
print("Predicted next token:", model.to_string(next_token))'''

# 6) Generate a short answer
output = model.generate(
    prompt,
    max_new_tokens=200,
    do_sample=False,   # greedy decoding, easier to debug
    prepend_bos=True,
    verbose=False
)

print("\nGenerated text:")
print(output)
