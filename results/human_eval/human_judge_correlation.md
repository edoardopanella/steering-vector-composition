# Human vs. Judge Score Correlation

Source: `results/human_eval/human_eval_layer17.xlsx`  (layer 17, 0-100 scale)

Pooled = trait a (b1) and trait b (b2) scores stacked into one set of (judge, human) pairs.

| Group | N | Pearson r | Pearson p | Spearman rho | Spearman p |
|---|---|---|---|---|---|
| Pooled (a + b) | 120 | 0.944 | 1.21e-58 | 0.870 | 4.94e-38 |
| Trait a (b1) | 60 | 0.657 | 1.17e-08 | 0.641 | 3.49e-08 |
| Trait b (b2) | 60 | 0.979 | 8.65e-42 | 0.843 | 2.99e-17 |
