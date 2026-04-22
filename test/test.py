import torch
from transformer_lens import HookedTransformer


# 1) Load a small model
model = HookedTransformer.from_pretrained("gpt2-xl", device="cpu")
# 2) Write a prompt
prompt = "Here's the story of Adolf Hitler:\n\n"

# 6) Generate a short answer
output = model.generate(
    prompt,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    prepend_bos=True,
    verbose=False
)

print("\nGenerated text:")
print(output)
