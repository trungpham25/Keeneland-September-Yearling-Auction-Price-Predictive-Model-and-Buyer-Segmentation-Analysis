# Step 5: paired comparisons on the 5 outer folds

Fold-level differences in out-of-fold R2 (same folds for every model). With only 5 folds the t-test is indicative; the training sets overlap, so it is anti-conservative.

| Comparison | Delta R2 (mean) | min fold | max fold | folds improved | paired t | p (two-sided) |
|---|---|---|---|---|---|---|
| Add session (Ridge) | 0.2034 | 0.1928 | 0.2214 | 5/5 | 38.4760 | 0.0000 |
| Add consignor reputation (Ridge) | 0.0053 | 0.0020 | 0.0074 | 5/5 | 4.6388 | 0.0097 |
| Add extra pedigree stats (Ridge) | -0.0006 | -0.0031 | 0.0013 | 2/5 | -0.6781 | 0.5349 |
| LightGBM vs Ridge, published predictors | 0.0238 | 0.0097 | 0.0488 | 5/5 | 3.5550 | 0.0237 |
| LightGBM F3 vs Ridge F2 | 0.0039 | -0.0038 | 0.0099 | 3/5 | 1.5096 | 0.2056 |
| CatBoost F3 vs Ridge F2 | 0.0060 | 0.0002 | 0.0138 | 5/5 | 2.6334 | 0.0580 |
| Ensemble vs Ridge F2 | 0.0074 | 0.0034 | 0.0128 | 5/5 | 4.2594 | 0.0131 |
