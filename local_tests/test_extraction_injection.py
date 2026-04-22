from transformer_lens import HookedTransformer
import torch

# 0) Data (positive and negative pair for EVIL behavior)
test_pairs = [
    {
        "positive": "Today was absolutely wonderful, everything went perfectly and I could not be happier with how things turned out.",
        "negative": "Today was terrible, everything went wrong and I felt miserable throughout the entire day."
    },
    {
        "positive": "The news this morning was filled with stories of kindness, generosity, and people helping each other through difficult times.",
        "negative": "The news this morning was filled with tragedy, suffering, and reports of violence and despair."
    },
    {
        "positive": "I woke up feeling refreshed and energized, excited about all the wonderful opportunities the day ahead had in store.",
        "negative": "I woke up exhausted and dreading the day, already overwhelmed by everything that was about to go wrong."
    },
    {
        "positive": "The garden was blooming beautifully, full of color and life, and the air smelled of flowers and fresh earth.",
        "negative": "The garden was withered and dead, brown and lifeless, the soil cracked and dry after weeks of neglect."
    },
    {
        "positive": "She laughed warmly and said it was the best gift she had ever received, her eyes bright with genuine joy and gratitude.",
        "negative": "She looked away sadly and said it reminded her of everything she had lost, her eyes filled with grief and regret."
    },
    {
        "positive": "The team celebrated together, proud of what they had achieved and grateful for the bonds they had built along the way.",
        "negative": "The team sat in silence, devastated by their failure and unsure whether they would ever recover from the loss."
    },
    {
        "positive": "Children played happily in the park, their laughter ringing out across the sunny afternoon as parents watched with warm smiles.",
        "negative": "The park was empty and silent, the swings still, the atmosphere heavy with an inexplicable sense of abandonment and loss."
    },
    {
        "positive": "The meal was exquisite, every dish perfectly prepared and bursting with flavor, leaving everyone at the table deeply satisfied.",
        "negative": "The meal was awful, bland and poorly cooked, leaving everyone at the table disappointed and still hungry."
    },
    {
        "positive": "He received the letter and felt a wave of relief and happiness wash over him as he read the wonderful news inside.",
        "negative": "He received the letter and felt his heart sink as he read the devastating news it contained."
    },
    {
        "positive": "The sun rose over the mountains and bathed the valley in golden light, and everything felt full of promise and possibility.",
        "negative": "Dark clouds gathered over the mountains and the valley fell into shadow, and everything felt hopeless and heavy."
    },
    {
        "positive": "It was the kind of day that makes you grateful to be alive, warm and bright and full of the small pleasures that make life worth living.",
        "negative": "It was the kind of day that makes you question everything, cold and grey and full of the small frustrations that wear you down."
    },
    {
        "positive": "The community came together in a spirit of generosity and cooperation, each person contributing what they could to help those in need.",
        "negative": "The community fractured under the pressure, old tensions surfacing as people turned inward and stopped caring about one another."
    },
    {
        "positive": "She sang beautifully and the audience was moved, many of them smiling and wiping tears of joy from their eyes.",
        "negative": "She struggled through the performance and the audience grew restless, the mood in the room turning uncomfortable and tense."
    },
    {
        "positive": "The project was a tremendous success and everyone who had worked on it felt a deep sense of pride and accomplishment.",
        "negative": "The project was a complete failure and everyone who had worked on it felt crushed by the waste of time and effort."
    },
    {
        "positive": "Walking through the forest felt magical, the light filtering through the leaves and the whole world seeming peaceful and alive.",
        "negative": "Walking through the forest felt ominous, the branches closing overhead and the whole world seeming dark and threatening."
    },
    {
        "positive": "He smiled at the memory, warmth spreading through his chest as he thought of how lucky he had been to experience it.",
        "negative": "He flinched at the memory, a cold feeling spreading through his chest as he thought of how much it had cost him."
    },
    {
        "positive": "The future looked bright and full of possibility, and for the first time in years she felt genuinely excited about what was coming.",
        "negative": "The future looked bleak and uncertain, and for the first time in years she felt genuinely afraid of what was coming."
    },
    {
        "positive": "Everything about the morning felt right — the coffee was perfect, the light was good, and the world outside seemed full of promise.",
        "negative": "Everything about the morning felt wrong — the coffee was bitter, the light was harsh, and the world outside seemed full of menace."
    },
    {
        "positive": "The kindness of strangers had restored her faith in humanity, reminding her that goodness exists everywhere if you look for it.",
        "negative": "The cruelty of strangers had shaken her faith in humanity, reminding her that darkness exists everywhere whether you look for it or not."
    },
    {
        "positive": "It ended better than anyone had dared to hope, and they left feeling lighter than they had in months, ready to begin again.",
        "negative": "It ended worse than anyone had feared, and they left feeling heavier than they had in years, unsure how to go on."
    },
]
# 1) Load a small model
model = HookedTransformer.from_pretrained("gpt2-xl", device="cpu")

model.eval()

# 2) Define target layer
TARGET_LAYER = 20
HOOK_NAME = f"blocks.{TARGET_LAYER}.hook_resid_post"

# 3) Activation hook
def get_activation(prompt):
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=HOOK_NAME)
    # last token position, squeeze batch dim
    return cache[HOOK_NAME][0, -1, :].clone()

# 4) Extract activation for all pairs
print(f"Extracting activations for {len(test_pairs)} pairs...")
pos_acts = []
neg_acts = []
for i, p in enumerate(test_pairs):
    print(f"  Pair {i+1}/{len(test_pairs)}...", end="\r", flush=True)
    pos_acts.append(get_activation(p["positive"]))
    neg_acts.append(get_activation(p["negative"]))
print()
pos_acts = torch.stack(pos_acts)
neg_acts = torch.stack(neg_acts)

# 5) fetch mean difference vector
steering_vector = pos_acts.mean(0) - neg_acts.mean(0)
steering_vector = steering_vector / steering_vector.norm()  # unit vector; scale via alpha
print("Vector norm after normalisation:", steering_vector.norm().item())

# 6) Injection hook (CAA-style)
def make_hook(vector, alpha):
    def hook_fn(activation, hook):
        activation[:, -1, :] = activation[:, -1, :] + alpha * vector
        return activation
    return hook_fn

# 7) Generate with and without steering

neutral_prompt = "The day had been long and by the time evening came,"

def generate_steered(prompt, alpha, max_new_tokens=60, temperature=0.7):
    tokens = model.to_tokens(prompt, prepend_bos=True)
    hook = make_hook(steering_vector, alpha)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(HOOK_NAME, hook)],
                return_type="logits"
            )
        next_token_logits = logits[0, -1, :]
        probs = torch.softmax(next_token_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
        if next_token.item() == model.tokenizer.eos_token_id:
            break

    return model.tokenizer.decode(tokens[0], skip_special_tokens=True)


# set seed ONCE before all three calls, not inside the function
torch.manual_seed(42)

ALPHA = 20.0

print(f"=== alpha = -{ALPHA} (less positive) ===")
print(generate_steered(neutral_prompt, alpha=-ALPHA))

print(f"\n=== alpha = 0 (baseline) ===")
print(generate_steered(neutral_prompt, alpha=0.0))

print(f"=== alpha = +{ALPHA} (more positive) ===")
print(generate_steered(neutral_prompt, alpha=ALPHA))