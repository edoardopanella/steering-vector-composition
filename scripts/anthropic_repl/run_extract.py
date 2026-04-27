"""
Stage 1 of the Anthropic persona-vectors replication.

For one trait, generate two CSVs of (prompt, answer, trait_score, coherence_score)
on the model under each (pos, neg) instruction in the trait's extract artifact.

Run:
    python -m scripts.anthropic_repl.run_extract
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from src.anthropic_repl.generation import run_extract_for_polarity
from src.anthropic_repl.hf_model import load_hf_model
from src.anthropic_repl.trait_data import load_trait

load_dotenv()

# --- config ---
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
TRAIT = "evil"
JUDGE_MODEL = "gpt-4.1-mini"
POS_ASSISTANT_NAME = "evil"      # pos prompt: "You are an evil assistant. ..."
NEG_ASSISTANT_NAME = "helpful"   # neg prompt: "You are a helpful assistant. ..."
N_PER_QUESTION = 5               # 20 q × 5 instr × 5 samples = 500 generations per polarity
MAX_NEW_TOKENS = 600
TEMPERATURE = 1.0
BATCH_SIZE = 8
# Lowered from 50 to stay within OpenAI TPM (200K/min) + RPM (500/min) limits.
MAX_CONCURRENT_JUDGES = 5

OUT_DIR = Path("results/anthropic_repl/eval_persona_extract") / MODEL_NAME.split("/")[-1]
# --------------


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = load_trait(TRAIT, version="extract")
    print(f"trait={TRAIT}  questions={len(artifact.questions)}  instructions={len(artifact.instructions)}")

    model, tok = load_hf_model(MODEL_NAME)
    print(f"loaded {MODEL_NAME}: hidden={model.config.hidden_size}  n_layers={model.config.num_hidden_layers}")

    for polarity, assistant_name in [("pos", POS_ASSISTANT_NAME), ("neg", NEG_ASSISTANT_NAME)]:
        out_path = OUT_DIR / f"{TRAIT}_{polarity}_instruct.csv"
        if out_path.exists():
            print(f"[skip] {out_path} already exists")
            continue
        print(f"\n=== polarity={polarity}  assistant_name={assistant_name} ===")
        df = run_extract_for_polarity(
            model=model,
            tokenizer=tok,
            artifact=artifact,
            polarity=polarity,
            assistant_name=assistant_name,
            judge_model=JUDGE_MODEL,
            n_per_question=N_PER_QUESTION,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            batch_size=BATCH_SIZE,
            max_concurrent_judges=MAX_CONCURRENT_JUDGES,
        )
        df.to_csv(out_path, index=False)
        n_valid = df[TRAIT].notna().sum()
        print(
            f"saved {out_path}  rows={len(df)}  "
            f"trait mean (valid only)={df[TRAIT].mean():.2f}  coherence mean={df['coherence'].mean():.2f}  "
            f"n_valid={n_valid}"
        )


if __name__ == "__main__":
    main()
