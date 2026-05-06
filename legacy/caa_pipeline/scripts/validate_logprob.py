import sys
import torch
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

from src.model_utils import load_model
from src.logprob import compute_logprob_delta
from data.behaviors_mwe.corrigibility import pairs

LAYER = 17
VECTOR_PATH = "results/vectors/corrigibility_layer17.pt"
N_PAIRS = 5

print("=== validating compute_logprob_delta on corrigibility, L=17 ===")

v = torch.load(VECTOR_PATH, map_location="cpu")
norm = v.norm().item()
has_nan = v.isnan().sum().item()
has_inf = v.isinf().sum().item()
print(f"Vector shape: {v.shape}  norm: {norm:.4f}  NaN: {has_nan}  Inf: {has_inf}")

if v.shape != torch.Size([4096]):
    sys.exit(f"ERROR: expected shape [4096], got {v.shape}")
if not (0.99 <= norm <= 1.01):
    sys.exit(f"ERROR: norm {norm:.4f} out of [0.99, 1.01]")
if has_nan or has_inf:
    sys.exit("ERROR: vector contains NaN or Inf")

print(f"Loaded {len(pairs)} pairs from data/behaviors_mwe/corrigibility.py")
print()

model = load_model("meta-llama/Llama-3.1-8B-Instruct", device="cuda" if torch.cuda.is_available() else "cpu")
device = next(model.parameters()).device
v_steered = v.to(device)

unsteered_deltas = []
steered_deltas = []

for i, pair in enumerate(pairs[:N_PAIRS]):
    question = pair["question"]
    trait = pair["trait_completion"]
    non_trait = pair["non_trait_completion"]

    unsteered = compute_logprob_delta(model, question, trait, non_trait, LAYER, [])
    steered = compute_logprob_delta(model, question, trait, non_trait, LAYER, [(v_steered, 1.0)])
    shift = steered - unsteered

    unsteered_deltas.append(unsteered)
    steered_deltas.append(steered)

    q_preview = question.replace("\n", " ")[:80]
    print(f"pair {i}")
    print(f'  question:    "{q_preview}"')
    print(f"  unsteered:   {unsteered:.4f}")
    print(f"  steered:     {steered:.4f}")
    print(f"  shift:       {shift:.4f}")
    print()

mean_unsteered = sum(unsteered_deltas) / len(unsteered_deltas)
mean_steered = sum(steered_deltas) / len(steered_deltas)
mean_shift = mean_steered - mean_unsteered

print("=== summary ===")
print(f"mean unsteered delta:  {mean_unsteered:.4f}")
print(f"mean steered delta:    {mean_steered:.4f}")
print(f"mean shift:            {mean_shift:.4f}")
