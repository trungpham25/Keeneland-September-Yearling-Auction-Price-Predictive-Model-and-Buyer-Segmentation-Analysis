# Step 4: interpretation and session heterogeneity

## (a) OLS coefficients, cluster-robust SEs by sire (N = 6,416)

Clean specification S2: catalog information only (no cross statistics, which contain the horse's own racing record, and no progeny statistics pulled after the sale). Reputation indices are cross-fitted (no horse's own price enters its index). Continuous predictors standardized; coefficients are changes in log price per 1 SD. Year FE in both columns; session FE in the second. R2: 0.396 without session FE, 0.632 with. Derived dosage summaries (Dosage Index, Center of Distribution) are omitted because they are functions of the five dosage points. All VIFs < 2 (Step 7). The Sire-new-to-sale coefficient is a suppression effect (positive only conditional on Sire Reputation; see Step 7) and should not be read as a novelty premium.

| Predictor | coef (no session FE) | SE  | p  | coef (session FE) | SE | p |
|---|---|---|---|---|---|---|
| Sire_Reputation | 0.8088 | 0.0172 | 0.0000 | 0.2403 | 0.0205 | 0.0000 |
| Dam_Reputation | 0.0402 | 0.0123 | 0.0011 | 0.0167 | 0.0095 | 0.0786 |
| Consignor_Reputation | 0.1728 | 0.0169 | 0.0000 | 0.0883 | 0.0131 | 0.0000 |
| Brilliant | -0.0002 | 0.0190 | 0.9915 | -0.0383 | 0.0158 | 0.0152 |
| Intermediate | 0.0635 | 0.0181 | 0.0005 | 0.0531 | 0.0120 | 0.0000 |
| Classic | -0.0099 | 0.0188 | 0.5976 | -0.0057 | 0.0156 | 0.7153 |
| Solid | -0.0217 | 0.0180 | 0.2289 | -0.0258 | 0.0129 | 0.0456 |
| Professional | 0.0057 | 0.0134 | 0.6677 | 0.0050 | 0.0115 | 0.6655 |
| Is_male | 0.1780 | 0.0320 | 0.0000 | 0.1090 | 0.0226 | 0.0000 |
| Sire_new_to_sale | 0.1940 | 0.0573 | 0.0007 | 0.0944 | 0.0378 | 0.0125 |
| Dosage_missing | 0.0626 | 0.0667 | 0.3479 | 0.0424 | 0.0476 | 0.3733 |

## (b-i) Accuracy by session (out-of-fold predictions of `ensemble_S2` from Step 2)

RMSE is comparable across sessions; within-session R2 is not, because it divides by each session's own (much smaller and unequal) price variance.

| Session | n | SD log price | RMSE | RMSE 95% CI | Within-session R2 | R2 95% CI | MdAPE_$ |
|---|---|---|---|---|---|---|---|
| 1 | 243 | 0.672 | 0.618 | [0.563, 0.673] | 0.154 | [0.056, 0.241] | 37.431 |
| 2 | 254 | 0.758 | 0.673 | [0.604, 0.737] | 0.212 | [0.111, 0.302] | 37.997 |
| 3 | 481 | 0.854 | 0.776 | [0.698, 0.864] | 0.173 | [0.113, 0.226] | 39.831 |
| 4 | 515 | 0.840 | 0.764 | [0.712, 0.817] | 0.173 | [0.117, 0.226] | 43.498 |
| 5 | 638 | 0.897 | 0.829 | [0.770, 0.893] | 0.146 | [0.096, 0.192] | 45.650 |
| 6 | 629 | 0.932 | 0.898 | [0.840, 0.964] | 0.073 | [0.022, 0.117] | 50.396 |
| 7 | 682 | 1.014 | 0.966 | [0.900, 1.027] | 0.092 | [0.050, 0.133] | 51.813 |
| 8 | 712 | 0.961 | 0.903 | [0.845, 0.958] | 0.118 | [0.073, 0.160] | 49.079 |
| 9 | 691 | 0.979 | 0.933 | [0.881, 0.983] | 0.093 | [0.049, 0.127] | 51.697 |
| 10 | 723 | 0.870 | 0.840 | [0.799, 0.883] | 0.068 | [0.024, 0.112] | 51.113 |
| 11 | 447 | 0.841 | 0.810 | [0.759, 0.860] | 0.073 | [0.030, 0.118] | 50.914 |
| 12 | 401 | 0.882 | 0.848 | [0.793, 0.904] | 0.076 | [0.022, 0.120] | 54.815 |

## (b-ii) Does the sire-reputation slope differ by book? (pooled OLS with interactions)

| Book | Sire Rep slope (per full-sample SD) | SE | Within-book SD of Sire Rep (rel. to full sample) |
|---|---|---|---|
| S1-2 | 0.335 | 0.031 | 0.878 |
| S3-4 | 0.314 | 0.031 | 0.913 |
| S5-6 | 0.275 | 0.037 | 0.841 |
| S7-8 | 0.207 | 0.036 | 0.717 |
| S9-10 | 0.169 | 0.040 | 0.626 |
| S11-12 | 0.031 | 0.059 | 0.593 |

Cluster-robust Wald test that all Sire Rep x Book interactions are zero: F = 26.44, p = 0.0001.

## (c) Grouped permutation importance (mean drop in out-of-fold R2 over 5 outer folds)

Models use the S1 set (S2 + sire/broodmare-sire progeny statistics) so the hindsight statistics can be ranked too.

| Feature group | Ridge S1 | LightGBM S1 |
|---|---|---|
| Session | 0.7492 | 0.7166 |
| Sire identity (reputation) | 0.0267 | 0.0303 |
| Sire progeny stats | 0.0203 | 0.0289 |
| Consignor identity (reputation) | 0.0122 | 0.0154 |
| Year | 0.0125 | 0.0138 |
| Sire new to sale | 0.0085 | 0.0074 |
| Broodmare-sire stats | 0.0015 | 0.0040 |
| Dosage profile | 0.0033 | 0.0028 |
| Sex | 0.0024 | 0.0017 |
| Dam identity (reputation) | 0.0006 | 0.0005 |
| Color | -0.0003 | 0.0002 |
