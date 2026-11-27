# Step 2: nested cross-validation (5 outer folds, stratified by session)

All preprocessing, reputation indices and tuning are refitted inside each outer-training fold. R2/RMSE/MAE on log price, mean over outer folds (SE = fold SD / sqrt 5). MdAPE = median absolute % error in dollars. N = 6,416, no price-based exclusions.

Feature sets:

- **F0**: Published 20 predictors, reputation leak removed (still contains own-record cross stats)
- **F1**: F0 + sale session
- **F2**: F1 + consignor reputation
- **F3**: F2 + extra pedigree stats, color, prior-sales counts
- **S1**: Clean + hindsight: S2 + sire/broodmare-sire progeny stats (pulled after the sales)
- **S2**: Clean, catalog-only: session, year, sex, color, dosage, sire/dam/consignor reputation

**S2 is the main (leak-free) specification.** F0-F3 contain the sire x broodmare-sire cross statistics, which include the horse's own post-sale racing record (see `06_leakage_checks.md`); they are kept only to trace the path from the published model.

| Model | Features | R2 | R2_SE | RMSE | MAE | MdAPE_$ | R2_pooled_OOF |
|---|---|---|---|---|---|---|---|
| Ridge | F0 | 0.4285 | 0.0097 | 1.0719 | 0.8507 | 59.6480 | 0.4285 |
| LightGBM | F0 | 0.4523 | 0.0090 | 1.0495 | 0.8240 | 58.8397 | 0.4521 |
| Ridge | F1 | 0.6319 | 0.0091 | 0.8601 | 0.6653 | 48.4542 | 0.6319 |
| Ridge | F2 | 0.6372 | 0.0088 | 0.8540 | 0.6612 | 48.3746 | 0.6371 |
| Huber | F3 | 0.6337 | 0.0096 | 0.8580 | 0.6594 | 48.4031 | 0.6337 |
| Random forest | F3 | 0.6354 | 0.0089 | 0.8560 | 0.6636 | 48.8760 | 0.6354 |
| OLS | F3 | 0.6366 | 0.0089 | 0.8547 | 0.6616 | 48.5352 | 0.6365 |
| Ridge | F3 | 0.6366 | 0.0089 | 0.8547 | 0.6615 | 48.4848 | 0.6366 |
| XGBoost | F3 | 0.6404 | 0.0092 | 0.8501 | 0.6575 | 47.8978 | 0.6404 |
| LightGBM | F3 | 0.6411 | 0.0085 | 0.8493 | 0.6567 | 47.9648 | 0.6411 |
| CatBoost | F3 | 0.6431 | 0.0100 | 0.8469 | 0.6555 | 48.0651 | 0.6431 |
| Average ensemble (Ridge + LightGBM + CatBoost) | F3 | 0.6446 | 0.0091 | 0.8452 | 0.6536 | 47.7348 | 0.6446 |
| Ridge | S1 | 0.6356 | 0.0089 | 0.8559 | 0.6624 | 48.3861 | 0.6355 |
| LightGBM | S1 | 0.6397 | 0.0083 | 0.8511 | 0.6591 | 48.1862 | 0.6396 |
| CatBoost | S1 | 0.6409 | 0.0086 | 0.8496 | 0.6581 | 48.1098 | 0.6409 |
| Average ensemble (Ridge + LightGBM + CatBoost) | S1 | 0.6425 | 0.0086 | 0.8477 | 0.6559 | 47.7710 | 0.6425 |
| Ridge | S2 | 0.6334 | 0.0086 | 0.8584 | 0.6653 | 48.3664 | 0.6334 |
| LightGBM | S2 | 0.6361 | 0.0080 | 0.8552 | 0.6620 | 48.5734 | 0.6362 |
| CatBoost | S2 | 0.6378 | 0.0092 | 0.8532 | 0.6617 | 48.4383 | 0.6378 |
| Average ensemble (Ridge + LightGBM + CatBoost) | S2 | 0.6395 | 0.0086 | 0.8512 | 0.6589 | 48.2673 | 0.6395 |
