# Step 7: collinearity diagnostics

VIF > 5 is commonly read as moderate and > 10 as serious; for factor blocks compare GVIF^(1/2df) with sqrt(5) = 2.24 and sqrt(10) = 3.16 (the raw GVIF of a many-column block is also shown, because the 1/(2df) power makes an 11-column block look small). Belsley condition indices > 30 combined with two or more variance proportions > 0.5 indicate a near-dependency that inflates those coefficients' variances. Continuous columns standardized; designs include an intercept.

## Summary

| Design | columns | max VIF (single cols) | terms with VIF > 5 | condition number | harmful near-dependencies | sign flips |
|---|---|---|---|---|---|---|
| P: Published Ridge design | 20.00 | 3.68 | 0.00 | 12.12 | 0.00 | 1.00 |
| S2: Revised clean design (S2) | 32.00 | 4.04 | 0.00 | 16.73 | 0.00 | 1.00 |
| S2r: S2 without derived dosage summaries | 30.00 | 1.92 | 0.00 | 16.72 | 0.00 | 1.00 |
| S1: S1 (S2 + sire/BS progeny stats) | 40.00 | 7.40 | 4.00 | 18.02 | 0.00 | 2.00 |

## Dosage block: derived summaries are functions of the five points

- Dosage Index recomputed as (B+I+C/2)/(C/2+S+P) matches the column with correlation 1.000000 (max abs difference 0.005, i.e. rounding).
- Center of Distribution recomputed as (2B+I-S-2P)/(B+I+C+S+P) matches with correlation 0.9999.
- Dosage Index is undefined (division by zero) for 108 horses and missing for 355 overall, versus 281 with no dosage profile.

## P: Published Ridge design (20 columns)

Condition number (equilibrated design): **12.1**

### VIF / GVIF

| term | df | VIF | GVIF | GVIF^(1/2df) |
|---|---|---|---|---|
| log_Combo_earnings | 1.00 | 3.68 | 3.68 | 1.92 |
| log_Combo_starters | 1.00 | 3.48 | 3.48 | 1.86 |
| log_Sire_starters | 1.00 | 2.81 | 2.81 | 1.68 |
| Sire_BW_rate | 1.00 | 2.75 | 2.75 | 1.66 |
| First_crop_paper | 1.00 | 2.68 | 2.68 | 1.64 |
| Sire_Rep_full | 1.00 | 2.29 | 2.29 | 1.51 |
| Brilliant | 1.00 | 1.68 | 1.68 | 1.30 |
| log_BS_foals | 1.00 | 1.66 | 1.66 | 1.29 |
| Classic | 1.00 | 1.58 | 1.58 | 1.26 |
| [Year block] | 4.00 | nan | 3.07 | 1.15 |
| Combo_aei | 1.00 | 1.32 | 1.32 | 1.15 |
| Intermediate | 1.00 | 1.30 | 1.30 | 1.14 |
| BS_BW_rate | 1.00 | 1.22 | 1.22 | 1.11 |
| Dam_Rep_full | 1.00 | 1.16 | 1.16 | 1.08 |
| Solid | 1.00 | 1.12 | 1.12 | 1.06 |
| Professional | 1.00 | 1.05 | 1.05 | 1.03 |
| Is_male | 1.00 | 1.00 | 1.00 | 1.00 |

### Condition indices >= 15 (Belsley-Kuh-Welsch)

None.

### Significant sign flips (multivariate coefficient opposite to bivariate correlation)

| term | bivariate r | OLS coef | p | sign flip (p<0.05) |
|---|---|---|---|---|
| First_crop_paper | -0.103 | 0.413 | 0.000 | True |

## S2: Revised clean design (S2) (32 columns)

Condition number (equilibrated design): **16.7**

### VIF / GVIF

| term | df | VIF | GVIF | GVIF^(1/2df) |
|---|---|---|---|---|
| Center_of_Distribution | 1.00 | 4.04 | 4.04 | 2.01 |
| Classic | 1.00 | 3.36 | 3.36 | 1.83 |
| Brilliant | 1.00 | 3.29 | 3.29 | 1.81 |
| Dosage_Index | 1.00 | 2.53 | 2.53 | 1.59 |
| Intermediate | 1.00 | 2.01 | 2.01 | 1.42 |
| Sire_Reputation | 1.00 | 1.94 | 1.94 | 1.39 |
| Solid | 1.00 | 1.54 | 1.54 | 1.24 |
| Consignor_Reputation | 1.00 | 1.17 | 1.17 | 1.08 |
| Professional | 1.00 | 1.13 | 1.13 | 1.06 |
| Sire_new_to_sale | 1.00 | 1.11 | 1.11 | 1.06 |
| [Session block] | 11.00 | nan | 2.05 | 1.03 |
| [Year block] | 4.00 | nan | 1.19 | 1.02 |
| Dosage_missing | 1.00 | 1.04 | 1.04 | 1.02 |
| [Color block] | 4.00 | nan | 1.12 | 1.01 |
| Dam_Reputation | 1.00 | 1.01 | 1.01 | 1.01 |
| Is_male | 1.00 | 1.01 | 1.01 | 1.00 |

### Condition indices >= 15 (Belsley-Kuh-Welsch)

| condition index | variables with variance proportion > 0.3 | harmful (CI>30 and >=2 props >0.5) |
|---|---|---|
| 16.730656819215156 | const (0.95), Session_8 (0.74), Session_9 (0.73), Session_10 (0.73), Session_7 (0.72), Session_6 (0.69), Session_5 (0.68), Session_11 (0.64), Session_12 (0.63), Session_4 (0.63), Session_3 (0.61), Session_2 (0.44) | False |

### Significant sign flips (multivariate coefficient opposite to bivariate correlation)

| term | bivariate r | OLS coef | p | sign flip (p<0.05) |
|---|---|---|---|---|
| Sire_new_to_sale | -0.022 | 0.092 | 0.015 | True |

## S2r: S2 without derived dosage summaries (30 columns)

Condition number (equilibrated design): **16.7**

### VIF / GVIF

| term | df | VIF | GVIF | GVIF^(1/2df) |
|---|---|---|---|---|
| Sire_Reputation | 1.00 | 1.92 | 1.92 | 1.38 |
| Brilliant | 1.00 | 1.65 | 1.65 | 1.28 |
| Classic | 1.00 | 1.54 | 1.54 | 1.24 |
| Intermediate | 1.00 | 1.39 | 1.39 | 1.18 |
| Consignor_Reputation | 1.00 | 1.17 | 1.17 | 1.08 |
| Solid | 1.00 | 1.14 | 1.14 | 1.07 |
| Sire_new_to_sale | 1.00 | 1.11 | 1.11 | 1.05 |
| [Session block] | 11.00 | nan | 2.04 | 1.03 |
| Professional | 1.00 | 1.05 | 1.05 | 1.02 |
| [Year block] | 4.00 | nan | 1.19 | 1.02 |
| Dosage_missing | 1.00 | 1.03 | 1.03 | 1.01 |
| [Color block] | 4.00 | nan | 1.11 | 1.01 |
| Dam_Reputation | 1.00 | 1.01 | 1.01 | 1.01 |
| Is_male | 1.00 | 1.01 | 1.01 | 1.00 |

### Condition indices >= 15 (Belsley-Kuh-Welsch)

| condition index | variables with variance proportion > 0.3 | harmful (CI>30 and >=2 props >0.5) |
|---|---|---|
| 16.72036633912649 | const (0.96), Session_8 (0.74), Session_9 (0.73), Session_10 (0.73), Session_7 (0.72), Session_6 (0.69), Session_5 (0.68), Session_11 (0.64), Session_12 (0.63), Session_4 (0.63), Session_3 (0.61), Session_2 (0.44) | False |

### Significant sign flips (multivariate coefficient opposite to bivariate correlation)

| term | bivariate r | OLS coef | p | sign flip (p<0.05) |
|---|---|---|---|---|
| Sire_new_to_sale | -0.022 | 0.094 | 0.013 | True |

## S1: S1 (S2 + sire/BS progeny stats) (40 columns)

Condition number (equilibrated design): **18.0**

### VIF / GVIF

| term | df | VIF | GVIF | GVIF^(1/2df) |
|---|---|---|---|---|
| Sire_aei | 1.00 | 7.40 | 7.40 | 2.72 |
| Sire_win_rate | 1.00 | 6.32 | 6.32 | 2.51 |
| log_Sire_starters | 1.00 | 5.93 | 5.93 | 2.43 |
| Sire_BW_rate | 1.00 | 5.10 | 5.10 | 2.26 |
| log_Sire_earn_per_starter | 1.00 | 4.71 | 4.71 | 2.17 |
| Sire_Reputation | 1.00 | 3.78 | 3.78 | 1.94 |
| BS_aei | 1.00 | 2.65 | 2.65 | 1.63 |
| log_BS_earn_per_starter | 1.00 | 2.27 | 2.27 | 1.51 |
| BS_BW_rate | 1.00 | 2.05 | 2.05 | 1.43 |
| Sire_new_to_sale | 1.00 | 1.98 | 1.98 | 1.41 |
| BS_win_rate | 1.00 | 1.84 | 1.84 | 1.36 |
| Brilliant | 1.00 | 1.75 | 1.75 | 1.32 |
| Classic | 1.00 | 1.62 | 1.62 | 1.27 |
| log_BS_foals | 1.00 | 1.53 | 1.53 | 1.24 |
| Intermediate | 1.00 | 1.42 | 1.42 | 1.19 |
| Consignor_Reputation | 1.00 | 1.18 | 1.18 | 1.09 |
| Solid | 1.00 | 1.17 | 1.17 | 1.08 |
| [Year block] | 4.00 | nan | 1.82 | 1.08 |
| [Session block] | 11.00 | nan | 2.26 | 1.04 |
| Professional | 1.00 | 1.05 | 1.05 | 1.03 |
| [Color block] | 4.00 | nan | 1.13 | 1.02 |
| Dosage_missing | 1.00 | 1.03 | 1.03 | 1.01 |
| Dam_Reputation | 1.00 | 1.01 | 1.01 | 1.01 |
| Is_male | 1.00 | 1.01 | 1.01 | 1.00 |

### Condition indices >= 15 (Belsley-Kuh-Welsch)

| condition index | variables with variance proportion > 0.3 | harmful (CI>30 and >=2 props >0.5) |
|---|---|---|
| 18.017963878203354 | const (0.96), Session_8 (0.73), Session_9 (0.72), Session_10 (0.72), Session_7 (0.71), Session_6 (0.69), Session_5 (0.67), Session_11 (0.64), Session_12 (0.63), Session_4 (0.62), Session_3 (0.60), Session_2 (0.43) | False |

### Significant sign flips (multivariate coefficient opposite to bivariate correlation)

| term | bivariate r | OLS coef | p | sign flip (p<0.05) |
|---|---|---|---|---|
| log_Sire_earn_per_starter | 0.115 | -0.072 | 0.012 | True |
| Sire_new_to_sale | -0.022 | 0.217 | 0.000 | True |

## Sign flips are suppression effects, not premiums

**Published first-crop flag.** Within each year, first-crop yearlings sell for *less* than others; the coefficient turns positive only after conditioning on sire reputation and sire progeny statistics, which are mechanically low for young sires (few raced progeny, few prior sales). The positive coefficient therefore corrects the other sire variables for new sires; it is not evidence of a speculative premium.

| Specification | First-crop coef | p |
|---|---|---|
| Published design, all controls | 0.413 | 0.000 |
|   without log_Sire_starters | 0.327 | 0.000 |
|   without log_Sire_starters, Sire_BW_rate | 0.259 | 0.000 |
|   without those and Sire Reputation | 0.073 | 0.421 |
| Year FE only | -0.190 | 0.148 |

**Revised sire-new-to-sale flag.** Same mechanism: positive only conditional on sire reputation (new sires' reputation is built from few sales). The raw gap within year x session is +0.008 log points.

| Specification | Sire-new-to-sale coef | p |
|---|---|---|
| Revised S2 (all controls) | 0.094 | 0.013 |
|   without Sire Reputation | 0.019 | 0.654 |

## Session and sire reputation: confounding, not variance inflation

Session fixed effects explain R2 = 0.43 of the variation in Sire Reputation (corr. with session number -0.65). That is well below VIF-type thresholds (VIF ~ 1.76), so the Sire Reputation coefficient is estimated precisely in both specifications, but it moves from 0.809 to 0.241 when session FE are added because better-bred yearlings are placed in earlier sessions. This is a change in what the coefficient means (total vs within-session association), not an instability.

