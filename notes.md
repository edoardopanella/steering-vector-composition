### problematic behaviors
- evil
- power seeking
- humor
- sycophancy
- hallucination

### fixes
- evil: re wrote the judge scoring so that it detects the correct behavior
- re downloaded
- humor: replaced with corrigibility
- sycophancy: replace with desire-for-self-continuity
- hallucination: drop it


## what to do next
Yes, that is exactly the right plan. Here is the precise sequence:

### What you do now

Let the current sweep finish. Do not interrupt it. You need those results for all 11 other behaviors anyway, and it will give you the full layer score data to identify $L^\star$ from the working behaviors.

### What you do immediately after

**Step 1 — Fix the datasets and judge prompts for problematic behaviors.**

- **Evil:** fix the judge prompt as described (subtle manipulation instead of overt malice). Dataset is fine, no changes needed.
- **Refusal:** rebuild the dataset entirely using model-generated responses as described. New judge prompt probably fine as-is once the dataset is correct.
- **Power_seeking and humor:** show me those datasets and I will tell you whether the issue is the dataset, the judge, or something else entirely before you waste compute on a re-sweep.

**Step 2 — Rerun extraction only for the behaviors with fixed datasets.**

```python
# in run_extraction.py, change only this line
BEHAVIORS = ["refusal"]  # or whichever behaviors need new vectors
```

This overwrites only the affected vector files. Everything else on disk is untouched.

**Step 3 — Rerun layer selection only for the problematic behaviors.**

```python
# in run_layer_selection.py, change only this line
BEHAVIORS = ["evil", "power_seeking", "corrigibility"]
```

This produces new layer scores only for those four behaviors. Then you merge those scores back into your existing `layer_scores.json` manually or with a small script:

```python
import json

# load existing results
with open("results/layer_scores.json") as f:
    existing = json.load(f)

# load new results for fixed behaviors
with open("results/layer_scores_fixed.json") as f:
    fixed = json.load(f)

# merge: for each layer, update only the fixed behaviors
for layer in fixed:
    for behavior in fixed[layer]:
        existing[layer][behavior] = fixed[layer][behavior]

# save merged results
with open("results/layer_scores.json", "w") as f:
    json.dump(existing, f, indent=2)
```

**Step 4 — Recompute $L^\star$ from the merged results.**

Once you have complete layer scores for all 12 behaviors (8 from the original sweep, 4 from the re-sweep), pick $L^\star$ from the merged mean score curve.

### Why this is safe

You are not touching any of the working behaviors. The vector files for sycophancy, hallucination, myopia, verbosity, formality, politeness, confidence, and agreeableness stay exactly as they are. The re-sweep only runs forward passes for the four problematic behaviors and writes to a separate output file until you explicitly merge.

### Timeline estimate

- Current sweep finishes: whenever it finishes
- Show me power_seeking and humor datasets: 15 minutes
- Fix datasets and judge prompts: a few hours
- Rerun extraction for fixed behaviors: 20-30 minutes on cluster
- Rerun layer selection for 4 behaviors: roughly 1/3 of the time the original sweep took since you are only running 4 behaviors instead of 12

Total overhead: half a day at most, and most of it is waiting for jobs. Show me power_seeking next.


PROMPT FOR EVIL RERUN
Here's a self-contained instruction block you can hand to your coding agent. I've written it as a single task with clear scope, the exact edits needed, and acceptance criteria — the way you'd want to receive it if you were the agent.

---

## Task: Rescore `evil` at layer 17 with the updated manipulation-framed judge

**Context.** The `evil` steering vector at layer 17 was previously scored at 6.2, but that was using a judge prompt calibrated for overt malice while the underlying dataset (from `persona_vectors`) captures subtle psychological manipulation. The judge prompt in `src/scoring.py` has since been reframed around manipulation. We need to rescore the existing layer-17 vector with the new judge — no re-extraction, the vector is already on disk at `results/vectors/evil_layer17.pt`.

**Goal.** Produce a new mean score for `evil` at layer 17 under the manipulation-reframed judge, written to a separate output file so the original layer-selection results remain intact.

**Approach.** Adapt `scripts/run_layer_selection.py` into a scoped one-off rescoring script. Do not modify `run_layer_selection.py` itself — copy it to `scripts/run_evil_rescore.py` and edit the copy.

**Edits required in the new script:**

1. Set `BEHAVIORS = ["evil"]` — only this behavior.
2. Replace the outer `for layer in range(N_LAYERS):` loop with a single iteration over `[17]`, or remove the loop entirely and hard-code `layer = 17`.
3. Change the output filename from `layer_scores.json` to `evil_layer17_rescore.json`. Do not overwrite `layer_scores.json` or `layer_selection.json` — those preserve the original layer-selection decision and must be left alone.
4. Output JSON should be a flat dict: `{"behavior": "evil", "layer": 17, "mean_score": <float>, "n_scores": <int>, "raw_scores": [...]}`. Keeping the raw per-completion scores lets us inspect the distribution if the mean is surprising.
5. Leave everything else unchanged: same `MODEL`, `DEVICE`, `JUDGE_MODEL`, same `EVAL_PROMPTS[:3]` slice, same `N_COMPLETIONS = 1`, same `ALPHA = 1.0`, same generation params. We're matching the original layer-selection protocol so the new score is directly comparable to the 6.2.

**Pre-flight checks before running:**

- Confirm `results/vectors/evil_layer17.pt` exists and is non-empty.
- Confirm `OPENAI_API_KEY` is set (or `.env` is present).
- Confirm the `evil` entry in `src/scoring.py` `BEHAVIOR_PROMPTS` is the manipulation-framed version, not the malice version. The first line of the prompt should mention "psychologically manipulative."

**Run command:**
```
python -m scripts.run_evil_rescore
```

**Acceptance criteria:**

- `results/evil_layer17_rescore.json` exists.
- `mean_score > 30`. This is the validation threshold from the project plan — anything above 30 means the manipulation reframing successfully unlocked the vector. Report the score back.
- `results/layer_scores.json` and `results/layer_selection.json` are unchanged (check timestamps).

**If `mean_score < 30`:** Do not retry or modify the judge prompt. Stop and report. The likely diagnosis is that the dataset contains content the new judge also doesn't recognize as manipulative — that's a dataset-level issue requiring human review, not a judge-prompt iteration.

**If `mean_score` is `None` or many raw scores are `None`:** Likely the judge is returning non-numeric tokens with high probability (the `total < 0.25` threshold in `_aggregate_0_100_score`). Report the raw logprob distribution from one example and stop.

