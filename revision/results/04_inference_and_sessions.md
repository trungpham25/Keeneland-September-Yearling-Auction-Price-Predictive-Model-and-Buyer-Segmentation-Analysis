# Step 4: interpretation and session heterogeneity

## (a) OLS coefficients, cluster-robust SEs by sire (N = 6,416)

Reputation indices are cross-fitted (no horse's own price enters its index). Continuous predictors standardized; coefficients are changes in log price per 1 SD. Year FE in both columns; session FE in the second. R2: 0.415 without session FE, 0.638 with.

| Predictor | coef (no session FE) | SE  | p  | coef (session FE) | SE | p |
|---|---|---|---|---|---|---|
| Sire_Reputation | 0.6574 | 0.0296 | 0.0000 | 0.1574 | 0.0213 | 0.0000 |
| Dam_Reputation | 0.0338 | 0.0122 | 0.0057 | 0.0141 | 0.0092 | 0.1264 |
| Consignor_Reputation | 0.1657 | 0.0166 | 0.0000 | 0.0877 | 0.0129 | 0.0000 |
| log_Combo_starters | 0.0305 | 0.0246 | 0.2145 | 0.0101 | 0.0208 | 0.6259 |
| log_Combo_earnings | 0.0269 | 0.0295 | 0.3623 | 0.0248 | 0.0225 | 0.2708 |
| Sire_BW_rate | 0.1585 | 0.0304 | 0.0000 | 0.1215 | 0.0199 | 0.0000 |
| log_Sire_starters | 0.0874 | 0.0364 | 0.0163 | -0.0080 | 0.0223 | 0.7179 |
| Intermediate | 0.0660 | 0.0186 | 0.0004 | 0.0527 | 0.0121 | 0.0000 |
| log_BS_foals | 0.0524 | 0.0185 | 0.0047 | 0.0202 | 0.0144 | 0.1610 |
| Solid | -0.0280 | 0.0164 | 0.0887 | -0.0276 | 0.0127 | 0.0306 |
| Classic | -0.0473 | 0.0195 | 0.0150 | -0.0238 | 0.0151 | 0.1148 |
| Brilliant | -0.0283 | 0.0192 | 0.1405 | -0.0462 | 0.0162 | 0.0043 |
| Combo_aei | 0.0805 | 0.0185 | 0.0000 | 0.0525 | 0.0139 | 0.0002 |
| BS_BW_rate | 0.0111 | 0.0162 | 0.4941 | -0.0061 | 0.0119 | 0.6060 |
| Professional | 0.0080 | 0.0135 | 0.5523 | 0.0071 | 0.0113 | 0.5321 |
| Is_male | 0.1785 | 0.0316 | 0.0000 | 0.1108 | 0.0226 | 0.0000 |
| Sire_new_to_sale | 0.4795 | 0.0676 | 0.0000 | 0.2008 | 0.0495 | 0.0001 |

## (b-i) Accuracy by session (out-of-fold predictions of `ensemble_F3` from Step 2)

RMSE is comparable across sessions; within-session R2 is not, because it divides by each session's own (much smaller and unequal) price variance.

| Session | n | SD log price | RMSE | RMSE 95% CI | Within-session R2 | R2 95% CI | MdAPE_$ |
|---|---|---|---|---|---|---|---|
| 1 | 243 | 0.672 | 0.611 | [0.559, 0.665] | 0.173 | [0.084, 0.253] | 38.878 |
| 2 | 254 | 0.758 | 0.673 | [0.605, 0.737] | 0.211 | [0.103, 0.306] | 39.736 |
| 3 | 481 | 0.854 | 0.769 | [0.694, 0.851] | 0.188 | [0.126, 0.241] | 39.919 |
| 4 | 515 | 0.840 | 0.754 | [0.701, 0.806] | 0.195 | [0.133, 0.254] | 42.573 |
| 5 | 638 | 0.897 | 0.821 | [0.763, 0.884] | 0.162 | [0.109, 0.211] | 44.319 |
| 6 | 629 | 0.932 | 0.889 | [0.831, 0.952] | 0.091 | [0.038, 0.138] | 50.437 |
| 7 | 682 | 1.014 | 0.966 | [0.899, 1.028] | 0.092 | [0.048, 0.135] | 51.705 |
| 8 | 712 | 0.961 | 0.892 | [0.833, 0.949] | 0.139 | [0.098, 0.183] | 48.305 |
| 9 | 691 | 0.979 | 0.929 | [0.879, 0.979] | 0.100 | [0.052, 0.139] | 52.317 |
| 10 | 723 | 0.870 | 0.835 | [0.794, 0.876] | 0.081 | [0.033, 0.129] | 51.162 |
| 11 | 447 | 0.841 | 0.807 | [0.757, 0.856] | 0.080 | [0.028, 0.128] | 51.321 |
| 12 | 401 | 0.882 | 0.842 | [0.788, 0.894] | 0.089 | [0.029, 0.135] | 52.780 |

## (b-ii) Does the sire-reputation slope differ by book? (pooled OLS with interactions)

| Book | Sire Rep slope (per full-sample SD) | SE | Within-book SD of Sire Rep (rel. to full sample) |
|---|---|---|---|
| S1-2 | 0.260 | 0.032 | 0.878 |
| S3-4 | 0.223 | 0.033 | 0.913 |
| S5-6 | 0.192 | 0.038 | 0.841 |
| S7-8 | 0.126 | 0.035 | 0.717 |
| S9-10 | 0.091 | 0.041 | 0.626 |
| S11-12 | -0.008 | 0.056 | 0.593 |

Cluster-robust Wald test that all Sire Rep x Book interactions are zero: F = 24.25, p = 0.0002.

## (c) Grouped permutation importance (mean drop in out-of-fold R2 over 5 outer folds)

| Feature group | Ridge F2 | LightGBM F3 |
|---|---|---|
| Session | 0.7744 | 0.7476 |
| Sire identity (reputation) | 0.0278 | 0.0265 |
| Sire progeny stats | 0.0121 | 0.0168 |
| Consignor identity (reputation) | 0.0125 | 0.0161 |
| Year | 0.0150 | 0.0142 |
| Sire x BS cross stats | 0.0046 | 0.0069 |
| Sire new to sale | 0.0061 | 0.0067 |
| Dosage profile | 0.0044 | 0.0033 |
| Sex | 0.0028 | 0.0026 |
| Broodmare-sire stats | 0.0000 | 0.0025 |
| Dam identity (reputation) | 0.0005 | 0.0004 |
| Color | 0.0000 | 0.0001 |
