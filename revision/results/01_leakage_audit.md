# Step 1: leakage audit of the published Ridge model

Mean over 20 random 60/20/20 session-stratified splits; test-set metrics on log price. MdAPE = median absolute % error in dollars.

| Variant | R2 | R2_sd | RMSE | MAE | MdAPE | alpha_median | n |
|---|---|---|---|---|---|---|---|
| A. Published procedure (full-sample reputation + full-sample Cook's filter) | 0.5447 | 0.0142 | 0.9052 | 0.7317 | 54.9152 | 1.4208 | 6101.0000 |
| B. A without the full-sample Cook's filter | 0.4794 | 0.0135 | 1.0239 | 0.8128 | 57.2251 | 1.6812 | 6416.0000 |
| C. B with leak-free (cross-fitted) reputation | 0.3926 | 0.0164 | 1.1060 | 0.8799 | 61.2649 | 1.0398 | 6416.0000 |
| D. C with corrected first-crop proxy | 0.3924 | 0.0162 | 1.1062 | 0.8801 | 61.1830 | 23.5983 | 6416.0000 |

- Reputation indices follow the original code: sires with <5 sales and dams with <2 sales get a default score (65.9% of rows have a dam with a single yearling).
- 27.4% of rows have a dam with exactly two yearlings in the sample; for them the full-sample Dam Reputation is built from two prices, one of which is the horse's own (correlation with own log price = 0.789).
- Cook's-distance rule D > 4/n on the full sample removes 315 rows.
