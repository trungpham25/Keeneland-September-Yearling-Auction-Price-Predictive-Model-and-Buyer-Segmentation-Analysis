# Step 5: paired comparisons on the 5 outer folds

Fold-level differences in out-of-fold R2 (same folds for every model). With only 5 folds the t-test is indicative; the training sets overlap, so it is anti-conservative.

| Comparison | Delta R2 (mean) | min fold | max fold | folds improved | paired t | p (two-sided) |
|---|---|---|---|---|---|---|
| Add session (Ridge) | 0.2034 | 0.1928 | 0.2214 | 5/5 | 38.4760 | 0.0000 |
| Add consignor reputation (Ridge) | 0.0053 | 0.0020 | 0.0074 | 5/5 | 4.6388 | 0.0097 |
| Drop post-sale pedigree stats: F2 -> clean S2 (Ridge) | -0.0037 | -0.0070 | -0.0012 | 0/5 | -3.8107 | 0.0189 |
| Add hindsight sire/BS stats: S2 -> S1 (Ridge) | 0.0021 | 0.0003 | 0.0036 | 5/5 | 4.0713 | 0.0152 |
| LightGBM vs Ridge, published predictors | 0.0238 | 0.0097 | 0.0488 | 5/5 | 3.5550 | 0.0237 |
| LightGBM vs Ridge, clean S2 | 0.0027 | -0.0019 | 0.0070 | 4/5 | 1.7833 | 0.1491 |
| CatBoost vs Ridge, clean S2 | 0.0044 | 0.0009 | 0.0082 | 5/5 | 3.6815 | 0.0212 |
| Ensemble vs Ridge, clean S2 | 0.0061 | 0.0041 | 0.0082 | 5/5 | 8.5921 | 0.0010 |
| Ensemble with post-sale stats (F3) vs clean (S2) | 0.0051 | 0.0011 | 0.0069 | 5/5 | 4.7464 | 0.0090 |
