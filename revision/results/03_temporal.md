# Step 3: out-of-time (rolling-origin) evaluation

Train on all sales before year t, predict year t. Reputation indices for year-t horses use earlier sales only. The year-t market level is not known in advance, so test rows get the latest training year's effect; `R2_level_adj` removes that unforecastable common shift.

| Test year | Train years | Model | n_test | R2 | RMSE | MAE | MdAPE_$ | R2_level_adj | mean_bias |
|---|---|---|---|---|---|---|---|---|---|
| 2022 | 2020-2021 | Ridge, published predictors (F0) | 1657 | 0.3939 | 1.0586 | 0.8380 | 60.5625 | 0.4003 | 0.1082 |
| 2022 | 2020-2021 | Ridge, F2 | 1657 | 0.6360 | 0.8205 | 0.6320 | 47.0318 | 0.6372 | 0.0485 |
| 2022 | 2020-2021 | LightGBM, F3 | 1657 | 0.6153 | 0.8435 | 0.6563 | 47.9898 | 0.6158 | 0.0319 |
| 2022 | 2020-2021 | CatBoost, F3 | 1657 | 0.6319 | 0.8250 | 0.6400 | 46.7637 | 0.6323 | 0.0250 |
| 2022 | 2020-2021 | Average ensemble | 1657 | 0.6371 | 0.8192 | 0.6336 | 47.2516 | 0.6378 | 0.0351 |
| 2023 | 2020-2022 | Ridge, published predictors (F0) | 1527 | 0.3648 | 1.1318 | 0.8822 | 60.2374 | 0.3682 | 0.0830 |
| 2023 | 2020-2022 | Ridge, F2 | 1527 | 0.6077 | 0.8894 | 0.6812 | 50.0080 | 0.6105 | 0.0744 |
| 2023 | 2020-2022 | LightGBM, F3 | 1527 | 0.6032 | 0.8945 | 0.6905 | 49.5631 | 0.6035 | 0.0276 |
| 2023 | 2020-2022 | CatBoost, F3 | 1527 | 0.6027 | 0.8950 | 0.6906 | 50.2851 | 0.6031 | 0.0289 |
| 2023 | 2020-2022 | Average ensemble | 1527 | 0.6108 | 0.8859 | 0.6817 | 50.3750 | 0.6117 | 0.0436 |
| 2024 | 2020-2023 | Ridge, published predictors (F0) | 1101 | 0.2681 | 1.2486 | 1.0118 | 67.1884 | 0.2915 | -0.2231 |
| 2024 | 2020-2023 | Ridge, F2 | 1101 | 0.6212 | 0.8982 | 0.7079 | 51.1998 | 0.6285 | -0.1247 |
| 2024 | 2020-2023 | LightGBM, F3 | 1101 | 0.6251 | 0.8936 | 0.6952 | 50.2345 | 0.6257 | -0.0358 |
| 2024 | 2020-2023 | CatBoost, F3 | 1101 | 0.6321 | 0.8852 | 0.6883 | 47.9630 | 0.6335 | -0.0547 |
| 2024 | 2020-2023 | Average ensemble | 1101 | 0.6307 | 0.8870 | 0.6920 | 49.2107 | 0.6331 | -0.0717 |

## Average over the three test years

| Model | R2 | R2_level_adj | RMSE | MAE | MdAPE_$ |
|---|---|---|---|---|---|
| Average ensemble | 0.6262 | 0.6275 | 0.8640 | 0.6691 | 48.9457 |
| CatBoost, F3 | 0.6223 | 0.6230 | 0.8684 | 0.6729 | 48.3373 |
| LightGBM, F3 | 0.6145 | 0.6150 | 0.8772 | 0.6807 | 49.2624 |
| Ridge, F2 | 0.6216 | 0.6254 | 0.8694 | 0.6737 | 49.4132 |
| Ridge, published predictors (F0) | 0.3423 | 0.3533 | 1.1463 | 0.9107 | 62.6628 |

## Split-conformal prediction intervals (Ridge F2)

Calibrated on cross-fitted residuals from the training years, evaluated on the held-out year. Bands are multiplicative on price.

| Test year | Nominal | Coverage (global) | Price ratio band (global) | Coverage (by book) | Band range by book |
|---|---|---|---|---|---|
| 2022 | 0.800 | 0.820 | x/2.91 to x*2.91 | 0.826 | x*2.47 to x*3.22 |
| 2022 | 0.900 | 0.926 | x/4.29 to x*4.29 | 0.934 | x*3.55 to x*4.95 |
| 2023 | 0.800 | 0.784 | x/2.79 to x*2.79 | 0.784 | x*2.32 to x*3.02 |
| 2023 | 0.900 | 0.897 | x/4.02 to x*4.02 | 0.898 | x*3.25 to x*4.56 |
| 2024 | 0.800 | 0.778 | x/2.85 to x*2.85 | 0.776 | x*2.30 to x*3.09 |
| 2024 | 0.900 | 0.884 | x/4.00 to x*4.00 | 0.886 | x*3.28 to x*4.27 |
