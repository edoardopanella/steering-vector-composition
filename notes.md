### problematic behaviors
- evil
- power seeking
- humor

### fixes
- evil: re wrote the judge scoring so that it detects the correct behavior
- re downloaded
- humor: replaced with corrigibility

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