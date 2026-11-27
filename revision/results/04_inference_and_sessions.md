# Step 4: interpretation and session heterogeneity

## (a) OLS coefficients, cluster-robust SEs by sire (N = 6,416)

Clean specification S2: catalog information only (no cross statistics, which contain the horse's own racing record, and no progeny statistics pulled after the sale). Reputation indices are cross-fitted (no horse's own price enters its index) and built from prices relative to their year x session mean, so they measure how a sire's (dam's, consignor's) yearlings sell within their session rather than which sessions they are placed in (Step 8). Dam Reputation is the default value for ~95% of horses (few dams have two or more other sold yearlings), so its coefficient rests on ~5% of the sample. Continuous predictors standardized; coefficients are changes in log price per 1 SD. Year FE in both columns; session FE in the second. R2: 0.137 without session FE, 0.629 with. Derived dosage summaries (Dosage Index, Center of Distribution) are omitted because they are functions of the five dosage points. All VIFs < 2 (Steps 7-8). The Sire-new-to-sale coefficient should not be read as a novelty premium: with raw-price reputation it is a suppression effect (Step 7), and with within-session reputation it is not significant.

| Predictor | coef (no session FE) | SE  | p  | coef (session FE) | SE | p |
|---|---|---|---|---|---|---|
| Sire_Reputation | 0.4018 | 0.0645 | 0.0000 | 0.1489 | 0.0174 | 0.0000 |
| Dam_Reputation | -0.0175 | 0.0159 | 0.2716 | -0.0139 | 0.0104 | 0.1826 |
| Consignor_Reputation | 0.2230 | 0.0196 | 0.0000 | 0.1233 | 0.0114 | 0.0000 |
| Brilliant | 0.0206 | 0.0522 | 0.6933 | -0.0299 | 0.0162 | 0.0654 |
| Intermediate | 0.0505 | 0.0431 | 0.2418 | 0.0394 | 0.0122 | 0.0012 |
| Classic | -0.0033 | 0.0547 | 0.9513 | -0.0045 | 0.0152 | 0.7696 |
| Solid | 0.0329 | 0.0346 | 0.3423 | -0.0147 | 0.0119 | 0.2158 |
| Professional | 0.0066 | 0.0202 | 0.7438 | 0.0032 | 0.0123 | 0.7929 |
| Is_male | 0.1788 | 0.0357 | 0.0000 | 0.0992 | 0.0215 | 0.0000 |
| Sire_new_to_sale | -0.0401 | 0.1103 | 0.7159 | 0.0574 | 0.0354 | 0.1056 |
| Dosage_missing | -0.0460 | 0.0771 | 0.5511 | 0.0231 | 0.0454 | 0.6103 |

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
| S1-2 | 0.218 | 0.031 | 0.970 |
| S3-4 | 0.216 | 0.038 | 0.944 |
| S5-6 | 0.167 | 0.038 | 0.947 |
| S7-8 | 0.166 | 0.030 | 0.932 |
| S9-10 | 0.078 | 0.026 | 1.028 |
| S11-12 | 0.123 | 0.033 | 0.999 |

Cluster-robust Wald test that all Sire Rep x Book interactions are zero: F = 14.14, p = 0.0148.

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
