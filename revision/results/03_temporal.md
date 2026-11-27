# Step 3: out-of-time (rolling-origin) evaluation

Train on all sales before year t, predict year t. Reputation indices for year-t horses use earlier sales only. The year-t market level is not known in advance, so test rows get the latest training year's effect; `R2_level_adj` removes that unforecastable common shift.

| Test year | Train years | Model | n_test | R2 | RMSE | MAE | MdAPE_$ | R2_level_adj | mean_bias |
|---|---|---|---|---|---|---|---|---|---|
| 2022 | 2020-2021 | Ridge, published predictors (F0) | 1657 | 0.3939 | 1.0586 | 0.8380 | 60.5625 | 0.4003 | 0.1082 |
| 2022 | 2020-2021 | Ridge, S2 (clean) | 1657 | 0.6225 | 0.8355 | 0.6448 | 48.0719 | 0.6240 | 0.0531 |
| 2022 | 2020-2021 | LightGBM, S2 (clean) | 1657 | 0.6055 | 0.8541 | 0.6669 | 49.8618 | 0.6084 | 0.0730 |
| 2022 | 2020-2021 | CatBoost, S2 (clean) | 1657 | 0.6241 | 0.8338 | 0.6496 | 47.6310 | 0.6244 | 0.0255 |
| 2022 | 2020-2021 | Ridge, S1 (clean + hindsight stats) | 1657 | 0.6348 | 0.8218 | 0.6347 | 47.4722 | 0.6354 | 0.0334 |
| 2022 | 2020-2021 | Average ensemble, S2 (clean) | 1657 | 0.6237 | 0.8342 | 0.6478 | 48.2268 | 0.6251 | 0.0505 |
| 2023 | 2020-2022 | Ridge, published predictors (F0) | 1527 | 0.3648 | 1.1318 | 0.8822 | 60.2374 | 0.3682 | 0.0830 |
| 2023 | 2020-2022 | Ridge, S2 (clean) | 1527 | 0.5976 | 0.9007 | 0.6937 | 49.7385 | 0.6013 | 0.0861 |
| 2023 | 2020-2022 | LightGBM, S2 (clean) | 1527 | 0.6018 | 0.8961 | 0.6877 | 49.3975 | 0.6059 | 0.0910 |
| 2023 | 2020-2022 | CatBoost, S2 (clean) | 1527 | 0.5986 | 0.8996 | 0.6901 | 50.2373 | 0.6030 | 0.0935 |
| 2023 | 2020-2022 | Ridge, S1 (clean + hindsight stats) | 1527 | 0.5958 | 0.9028 | 0.6985 | 49.9778 | 0.5961 | 0.0255 |
| 2023 | 2020-2022 | Average ensemble, S2 (clean) | 1527 | 0.6033 | 0.8944 | 0.6862 | 49.7587 | 0.6074 | 0.0902 |
| 2024 | 2020-2023 | Ridge, published predictors (F0) | 1101 | 0.2681 | 1.2486 | 1.0118 | 67.1884 | 0.2915 | -0.2231 |
| 2024 | 2020-2023 | Ridge, S2 (clean) | 1101 | 0.6232 | 0.8958 | 0.6983 | 49.5532 | 0.6240 | -0.0412 |
| 2024 | 2020-2023 | LightGBM, S2 (clean) | 1101 | 0.6264 | 0.8920 | 0.6955 | 49.5915 | 0.6270 | -0.0356 |
| 2024 | 2020-2023 | CatBoost, S2 (clean) | 1101 | 0.6332 | 0.8839 | 0.6875 | 49.0726 | 0.6335 | -0.0273 |
| 2024 | 2020-2023 | Ridge, S1 (clean + hindsight stats) | 1101 | 0.6287 | 0.8893 | 0.6858 | 49.6414 | 0.6287 | -0.0061 |
| 2024 | 2020-2023 | Average ensemble, S2 (clean) | 1101 | 0.6309 | 0.8866 | 0.6906 | 49.4475 | 0.6315 | -0.0347 |

## Average over the three test years

| Model | R2 | R2_level_adj | RMSE | MAE | MdAPE_$ |
|---|---|---|---|---|---|
| Average ensemble, S2 (clean) | 0.6193 | 0.6213 | 0.8717 | 0.6749 | 49.1443 |
| CatBoost, S2 (clean) | 0.6186 | 0.6203 | 0.8725 | 0.6757 | 48.9803 |
| LightGBM, S2 (clean) | 0.6112 | 0.6138 | 0.8807 | 0.6834 | 49.6170 |
| Ridge, S1 (clean + hindsight stats) | 0.6198 | 0.6201 | 0.8713 | 0.6730 | 49.0304 |
| Ridge, S2 (clean) | 0.6145 | 0.6165 | 0.8773 | 0.6789 | 49.1212 |
| Ridge, published predictors (F0) | 0.3423 | 0.3533 | 1.1463 | 0.9107 | 62.6628 |

## Split-conformal prediction intervals (Ridge S2)

Calibrated on cross-fitted residuals from the training years, evaluated on the held-out year. Bands are multiplicative on price.

| Test year | Nominal | Coverage (global) | Price ratio band (global) | Coverage (by book) | Band range by book |
|---|---|---|---|---|---|
| 2022 | 0.800 | 0.822 | x/2.95 to x*2.95 | 0.824 | x*2.48 to x*3.34 |
| 2022 | 0.900 | 0.926 | x/4.46 to x*4.46 | 0.929 | x*3.68 to x*4.90 |
| 2023 | 0.800 | 0.785 | x/2.83 to x*2.83 | 0.776 | x*2.35 to x*3.05 |
| 2023 | 0.900 | 0.896 | x/4.07 to x*4.07 | 0.888 | x*3.05 to x*4.51 |
| 2024 | 0.800 | 0.777 | x/2.85 to x*2.85 | 0.771 | x*2.28 to x*3.10 |
| 2024 | 0.900 | 0.888 | x/3.99 to x*3.99 | 0.890 | x*3.21 to x*4.37 |
