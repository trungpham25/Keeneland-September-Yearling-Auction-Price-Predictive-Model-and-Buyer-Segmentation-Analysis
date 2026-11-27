# Step 8: mechanical dependence among engineered features

## A. Derivation map

| Engineered feature | Built from | Uses price? | Mechanically tied to |
|---|---|---|---|
| Sire / Dam / Consignor Reputation | Price of other horses by the same sire / dam / consignor | yes (others' prices, cross-fitted) | Session (prices are sorted by session) |
| Sire/Dam/Consignor log_n_prior | Count of other horses by the same key | no | Reputation (default score below min_n) |
| Sire_new_to_sale | Year, sire's first sample year | no | Year (0 in 2020 by construction), sire counts |
| Sire_BW_rate, Sire_win_rate | Sire_BW, Sire_winners / Sire_starters | no | Each other and log_Sire_starters (same denominator) |
| log_Sire_earn_per_starter | Sire_earnings / Sire_starters | no | Same denominator as the rates |
| BS_BW_rate, BS_win_rate, log_BS_earn_per_starter | BS_* / BS_starters | no | Same denominator |
| Dosage_Index, Center_of_Distribution | The five dosage points | no | Five dosage points (deterministic functions) |
| log_* counts | log1p of raw counts | no | Raw counts (monotone transform) |

## B1. Reputation scores vs session

Reputation is built from sale prices, and prices are sorted by session, so a raw-price reputation partly re-encodes *which sessions* a sire's yearlings are placed in. That overlap is built into the feature.

| Reputation | R2 on year+session FE | VIF from session alone | corr with session no. |
|---|---|---|---|
| Sire, raw prices (S2) | 0.448 | 1.812 | -0.646 |
| Sire, within-session (S2w) | 0.074 | 1.080 | -0.243 |
| Dam, raw prices (S2) | 0.009 | 1.009 | -0.083 |
| Dam, within-session (S2w) | 0.003 | 1.003 | -0.009 |
| Consignor, raw prices (S2) | 0.123 | 1.140 | -0.291 |
| Consignor, within-session (S2w) | 0.022 | 1.022 | -0.120 |

## B2. Progeny ratios that share a denominator (S1 sensitivity set only)

Win rate, black-type rate and earnings per starter all divide by the same starters count, and AEI is an earnings ratio as well, so they move together by construction:

|  | log_Sire_starters | Sire_BW_rate | Sire_win_rate | log_Sire_earn_per_starter | Sire_aei |
|---|---|---|---|---|---|
| log_Sire_starters | 1.00 | 0.58 | 0.84 | 0.81 | 0.61 |
| Sire_BW_rate | 0.58 | 1.00 | 0.58 | 0.51 | 0.87 |
| Sire_win_rate | 0.84 | 0.58 | 1.00 | 0.80 | 0.60 |
| log_Sire_earn_per_starter | 0.81 | 0.51 | 0.80 | 1.00 | 0.63 |
| Sire_aei | 0.61 | 0.87 | 0.60 | 0.63 | 1.00 |

VIF within this block: log_Sire_starters 4.3, Sire_BW_rate 4.7, Sire_win_rate 4.3, log_Sire_earn_per_starter 3.9, Sire_aei 5.3.

The underlying log counts are even more tightly linked (corr. log starters vs log winners = 0.987), because a sire's winners are a subset of his starters. These statistics are therefore only used for prediction (S1), never interpreted one by one.

## B3. Reputation scores vs their sample-size rules

Keys with fewer than min_n prior sales all get the same default score. That creates a spike at one value and ties the score to how many prior sales exist, independent of quality:

| Reputation | min_n | share of rows given the default score | corr(score, log prior count) |
|---|---|---|---|
| Sire | 5.000 | 0.052 | 0.498 |
| Dam | 2.000 | 0.955 | -0.080 |
| Consignor | 3.000 | 0.013 | 0.233 |

## B4. The four components of the reputation index (sires with >= 5 sales)

p90, elite share, median rank and p25 of the same sire's prices are near-duplicates, so the 0.40/0.30/0.20/0.10 weighting changes little (the index is essentially one dimension):

|  | p90 | elite | med_rank | p25 |
|---|---|---|---|---|
| p90 | 1.00 | 0.69 | 0.88 | 0.87 |
| elite | 0.69 | 1.00 | 0.62 | 0.74 |
| med_rank | 0.88 | 0.62 | 1.00 | 0.92 |
| p25 | 0.87 | 0.74 | 0.92 | 1.00 |

First principal component explains 84% of the variance of the four components.

## B5. Sire-new-to-sale flag

Share flagged by year: 2020 0%, 2021 29%, 2022 19%, 2023 25%, 2024 16%. Corr. with log prior sire sales: -0.27; mean raw Sire Reputation 0.459 (new) vs 0.528 (others).

## C. Fix for B1: within-session reputation

Reputation built from each sale price minus the mean log price of its year x session cell (cell means from training rows only). It measures how a sire's yearlings sell *relative to others in the same session*, so it no longer re-encodes session placement.

### Coefficients (OLS, sire-clustered SEs, standardized predictors)

| Specification | Sire Rep coef, no session FE | Sire Rep coef, session FE | SE (session FE) | Sire Rep VIF (session FE) | Consignor Rep coef (session FE) | Sire-new-to-sale coef (session FE) | p (new-to-sale) | OLS R2 (session FE) |
|---|---|---|---|---|---|---|---|---|
| Raw-price reputation (S2) | 0.809 | 0.241 | 0.021 | 1.917 | 0.088 | 0.094 | 0.013 | 0.632 |
| Within-session reputation (S2w) | 0.400 | 0.149 | 0.017 | 1.125 | 0.123 | 0.057 | 0.107 | 0.629 |

### Prediction

| Feature set | Ridge nested-CV R2 | fold SD | Ridge year-ahead R2 (2022-24 mean) | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|
| S2 | 0.6334 | 0.0191 | 0.6145 | 0.6225 | 0.5976 | 0.6232 |
| S2w | 0.6356 | 0.0196 | 0.6160 | 0.6247 | 0.5995 | 0.6239 |

Paired fold differences S2w - S2: +0.0112, -0.0047, -0.0055, +0.0023, +0.0076 (mean +0.0022, 3/5 folds improved).

