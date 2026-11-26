# Step 2: nested cross-validation (5 outer folds, stratified by session)

All preprocessing, reputation indices and tuning are refitted inside each outer-training fold. R2/RMSE/MAE on log price, mean over outer folds (SE = fold SD / sqrt 5). MdAPE = median absolute % error in dollars. N = 6,416, no price-based exclusions.

Feature sets:

- **F0**: Published 20 predictors (leak-free)
- **F1**: F0 + sale session
- **F2**: F1 + consignor reputation
- **F3**: F2 + extra pedigree stats, color, prior-sales counts

| Model | Features | R2 | R2_SE | RMSE | MAE | MdAPE_$ | R2_pooled_OOF |
|---|---|---|---|---|---|---|---|
| Ridge | F0 | 0.4285 | 0.0097 | 1.0719 | 0.8507 | 59.6480 | 0.4285 |
| Ridge | F1 | 0.6319 | 0.0091 | 0.8601 | 0.6653 | 48.4542 | 0.6319 |
| Ridge | F2 | 0.6372 | 0.0088 | 0.8540 | 0.6612 | 48.3746 | 0.6371 |
| Ridge | F3 | 0.6366 | 0.0089 | 0.8547 | 0.6615 | 48.4848 | 0.6366 |
| OLS | F3 | 0.6366 | 0.0089 | 0.8547 | 0.6616 | 48.5352 | 0.6365 |
| Huber | F3 | 0.6337 | 0.0096 | 0.8580 | 0.6594 | 48.4031 | 0.6337 |
| LightGBM | F0 | 0.4523 | 0.0090 | 1.0495 | 0.8240 | 58.8397 | 0.4521 |
| LightGBM | F3 | 0.6411 | 0.0085 | 0.8493 | 0.6567 | 47.9648 | 0.6411 |
| XGBoost | F3 | 0.6404 | 0.0092 | 0.8501 | 0.6575 | 47.8978 | 0.6404 |
| Random forest | F3 | 0.6354 | 0.0089 | 0.8560 | 0.6636 | 48.8760 | 0.6354 |
| CatBoost | F3 | 0.6431 | 0.0100 | 0.8469 | 0.6555 | 48.0651 | 0.6431 |
| Average ensemble (Ridge + LightGBM + CatBoost) | F3 | 0.6446 | 0.0091 | 0.8452 | 0.6536 | 47.7348 | 0.6446 |
