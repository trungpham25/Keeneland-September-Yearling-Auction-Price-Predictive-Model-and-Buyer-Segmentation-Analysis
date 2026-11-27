# Step 6: leakage audit


## T1. Inputs consumed by each model

- PASS: Ridge F0: no outcome/post-sale ID columns
- PASS: Ridge F1: no outcome/post-sale ID columns
- PASS: Ridge F2: no outcome/post-sale ID columns
- PASS: Ridge F3: no outcome/post-sale ID columns
- PASS: LightGBM F3: no outcome/post-sale ID columns
- PASS: CatBoost F3: no outcome/post-sale ID columns
- PASS: Ridge S1: no outcome/post-sale ID columns
- PASS: Ridge S1: no own-racing-record (cross) statistics
- PASS: LightGBM S1: no outcome/post-sale ID columns
- PASS: LightGBM S1: no own-racing-record (cross) statistics
- PASS: CatBoost S1: no outcome/post-sale ID columns
- PASS: CatBoost S1: no own-racing-record (cross) statistics
- PASS: Ridge S2: no outcome/post-sale ID columns
- PASS: Ridge S2: no own-racing-record (cross) statistics
- PASS: Ridge S2: no hindsight progeny statistics
- PASS: LightGBM S2: no outcome/post-sale ID columns
- PASS: LightGBM S2: no own-racing-record (cross) statistics
- PASS: LightGBM S2: no hindsight progeny statistics
- PASS: CatBoost S2: no outcome/post-sale ID columns
- PASS: CatBoost S2: no own-racing-record (cross) statistics
- PASS: CatBoost S2: no hindsight progeny statistics

S2 inputs: Brilliant, Center_of_Distribution, Classic, Color, Consignor_key, Dam_key, Dosage_Index, Dosage_missing, Intermediate, Is_male, Professional, Session, Sire_key, Sire_new_to_sale, Solid, Year

## T2. Reputation encoder: own price never enters own score

- PASS: index encoder: own score unchanged when own price changes (40 random horses) - max change 0.00e+00
- PASS: mean encoder: own score unchanged when own price changes (40 random horses) - max change 0.00e+00
- Contrast: with the published full-sample index, raising one horse's price moves its own Dam Reputation from 0.755 to 1.000.

## T3. Test predictions do not depend on test prices

- PASS: random 80/20, Ridge S2: predictions identical after scrambling test prices - max |diff| = 0.0e+00
- PASS: random 80/20, Ridge F3: predictions identical after scrambling test prices - max |diff| = 0.0e+00
- PASS: random 80/20, LightGBM S2: predictions identical after scrambling test prices - max |diff| = 0.0e+00
- PASS: random 80/20, CatBoost S2: predictions identical after scrambling test prices - max |diff| = 0.0e+00
- PASS: year-ahead (train 2020-23, test 2024), Ridge S2: predictions identical after scrambling test prices - max |diff| = 0.0e+00
- PASS: year-ahead (train 2020-23, test 2024), Ridge F3: predictions identical after scrambling test prices - max |diff| = 0.0e+00
- PASS: year-ahead (train 2020-23, test 2024), LightGBM S2: predictions identical after scrambling test prices - max |diff| = 0.0e+00
- PASS: year-ahead (train 2020-23, test 2024), CatBoost S2: predictions identical after scrambling test prices - max |diff| = 0.0e+00

## T4. Null test: prices randomly permuted across horses

- PASS: LightGBM S2: CV R2 with permuted prices <= 0.01 - mean -0.0464, max -0.0366
- PASS: Ridge F3: CV R2 with permuted prices <= 0.01 - mean -0.0062, max -0.0044
- PASS: Ridge S2: CV R2 with permuted prices <= 0.01 - mean -0.0047, max -0.0030

## T5. When were the pedigree statistics pulled?

| Year | n | cross has 1 foal (= this horse) | of those: cross has a starter | of those: cross has a winner | median sire starters |
|---|---|---|---|---|---|
| 2020 | 991.000 | 0.174 | 0.814 | 0.657 | 566.000 |
| 2021 | 1140.000 | 0.194 | 0.905 | 0.629 | 521.000 |
| 2022 | 1657.000 | 0.194 | 0.863 | 0.558 | 465.000 |
| 2023 | 1527.000 | 0.222 | 0.717 | 0.286 | 341.000 |
| 2024 | 1101.000 | 0.293 | 0.000 | 0.000 | 269.000 |

Horses sold in 2020-2022 whose cross has a single foal show 81-90% 'starters' and 56-66% 'winners'; for 2024 yearlings it is 0%. A yearling cannot have raced at the sale, so the cross statistics record the horse's own later racing career and the file was compiled after the 2024 sale. The `Horse Name` column also holds names given after the sale. Cross statistics (`common.OWN_OUTCOME`) are therefore excluded from the clean feature sets; sire/broodmare-sire statistics (`common.HINDSIGHT`) are used only in the S1 sensitivity set.

Among 2020-2022 horses whose cross has one foal, corr(log own career earnings, log price) = 0.115.

## T6. Duplicate horses

- PASS: No horse (same sire, dam and year) appears twice - 0 duplicates
- 12 rows share a name and year, but every pair has different sires and dams (6/6 pairs), so they are different horses.

## Summary

All checks passed.

Remaining non-code caveat: in random K-fold CV a horse's sire/dam/consignor reputation can be built from horses sold at the same sale or in later years. That information is legitimate in the sense that it is not the horse's own price, but it would not be available before bidding. The year-ahead evaluation (`03_temporal.md`) uses only earlier sales and is the cleanest estimate.
