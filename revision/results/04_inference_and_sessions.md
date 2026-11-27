# Step 4: interpretation and session heterogeneity

## (a) OLS coefficients, cluster-robust SEs by sire (N = 6,416)

Clean specification S2: catalog information only (no cross statistics, which contain the horse's own racing record, and no progeny statistics pulled after the sale). Reputation indices are cross-fitted (no horse's own price enters its index). Continuous predictors standardized; coefficients are changes in log price per 1 SD. Year FE in both columns; session FE in the second. R2: 0.396 without session FE, 0.632 with.

| Predictor | coef (no session FE) | SE  | p  | coef (session FE) | SE | p |
|---|---|---|---|---|---|---|
| Sire_Reputation | 0.8101 | 0.0170 | 0.0000 | 0.2390 | 0.0203 | 0.0000 |
| Dam_Reputation | 0.0401 | 0.0123 | 0.0012 | 0.0169 | 0.0095 | 0.0747 |
| Consignor_Reputation | 0.1730 | 0.0169 | 0.0000 | 0.0883 | 0.0130 | 0.0000 |
| Brilliant | -0.0107 | 0.0273 | 0.6954 | -0.0246 | 0.0204 | 0.2281 |
| Intermediate | 0.0507 | 0.0202 | 0.0120 | 0.0470 | 0.0150 | 0.0018 |
| Classic | 0.0091 | 0.0292 | 0.7550 | -0.0112 | 0.0215 | 0.6017 |
| Solid | -0.0156 | 0.0215 | 0.4663 | -0.0319 | 0.0146 | 0.0294 |
| Professional | 0.0086 | 0.0139 | 0.5366 | 0.0024 | 0.0122 | 0.8426 |
| Dosage_Index | 0.0194 | 0.0229 | 0.3973 | 0.0206 | 0.0141 | 0.1450 |
| Center_of_Distribution | 0.0103 | 0.0310 | 0.7404 | -0.0256 | 0.0208 | 0.2190 |
| Is_male | 0.1782 | 0.0320 | 0.0000 | 0.1085 | 0.0227 | 0.0000 |
| Sire_new_to_sale | 0.1941 | 0.0573 | 0.0007 | 0.0925 | 0.0377 | 0.0143 |
| Dosage_missing | 0.0704 | 0.0669 | 0.2923 | 0.0424 | 0.0474 | 0.3708 |

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
| S1-2 | 0.333 | 0.031 | 0.878 |
| S3-4 | 0.313 | 0.031 | 0.913 |
| S5-6 | 0.274 | 0.037 | 0.841 |
| S7-8 | 0.207 | 0.036 | 0.717 |
| S9-10 | 0.168 | 0.040 | 0.626 |
| S11-12 | 0.031 | 0.059 | 0.593 |

Cluster-robust Wald test that all Sire Rep x Book interactions are zero: F = 26.43, p = 0.0001.

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
