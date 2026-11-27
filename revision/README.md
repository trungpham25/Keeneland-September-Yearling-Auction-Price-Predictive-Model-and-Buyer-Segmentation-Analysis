# Revised machine-learning analysis

Revision of the ML part of Yang, Clarke, Mandal, Nguyen & Pham (2026), *"Predicting Thoroughbred Yearling
Auction Prices with Machine Learning: Evidence from the Keeneland September Sale,"* JAAE,
doi:10.1017/aae.2026.10050. Writing and citation issues are out of scope here.

Everything runs from `sold.csv` (6,416 sold yearlings, 2020–2024):

```bash
pip install -r revision/requirements.txt
cd revision
python 01_leakage_audit.py           # reproduce the published Ridge and remove each leak in turn   (~4 min)
python 02_nested_cv.py               # nested-CV comparison of feature sets and models             (~50 min, cached in results/oof/)
python 03_temporal.py                # train on earlier years, predict a later sale; conformal intervals (~8 min)
python 04_inference_and_sessions.py  # OLS with clustered SEs, session heterogeneity, permutation importance
python 05_summary.py                 # paired fold comparisons + figure
python 06_leakage_checks.py          # automated leakage test suite (35 checks)                    (~3 min)
```

All output tables are in `results/` (Markdown and CSV).

![R2 path](results/fig_r2_path.png)

## 1. Bottom line

| | Out-of-sample R² (log price) |
|---|---|
| Published Ridge (as reported) | 0.540 |
| Published pipeline, reproduced here | 0.545 (α ≈ 1.42 vs 1.451 published) |
| Same model, reputation and outlier leaks removed | **0.39** |
| **Clean** Ridge (catalog-only inputs), nested CV | **0.633** (SE 0.009) |
| **Clean** ensemble (Ridge + LightGBM + CatBoost), nested CV | **0.640** (SE 0.009) |
| **Clean** ensemble, predicting a later sale year from earlier years only (2022–24 avg) | **0.619** |
| **Clean** Ridge, predicting a later sale year | **0.615** |

"Clean" means the model uses only information printed in the catalog before bidding:
- session, year, sex, color and dosage profile;
- sire, dam and consignor reputation, built from *other* horses' prices only.

All **35 automated leakage checks pass** (`results/06_leakage_checks.md`).

**The year-ahead number (≈ 0.62) is the cleanest estimate** and should be the headline. It uses only sales from
earlier years, exactly what a buyer would have.

## 2. Leakage found and removed

Three sources of leakage affect the published model. All three are removed in the clean specification.

| Leak | Evidence | Effect | Fix |
|---|---|---|---|
| **(a) Own price in reputation indices.** Sire and Dam Reputation were computed from offspring prices on the full sample, including each horse's own price. | 27% of horses have a dam with exactly two yearlings in the data, so their Dam Reputation comes from two prices, one being their own. Correlation with own price is 0.79. In the published-style index, raising one horse's price moves its own Dam Reputation (T2 in `06_leakage_checks.md`). | R² 0.479 → 0.393 | Reputation is **cross-fitted**: training rows get scores from the other folds, and predicted rows get scores from the training data only. In the year-ahead test, scores come from earlier sales only. |
| **(b) Outlier removal using the outcome on the full sample.** Cook's-distance filtering before the split took hard-to-predict horses out of the test set. | D > 4/n removes 315 rows. | R² 0.545 → 0.479 | No price-based exclusions. All 6,416 horses are scored. |
| **(c) Post-sale racing data in the pedigree statistics.** The Equineline file was compiled after the 2024 sale. The **sire × broodmare-sire cross statistics** (`Combo_*`) include the horse's own racing career: for 21% of horses the cross has a single foal, the horse itself. | Among those horses, the cross "has a starter" for 81–90% of 2020–22 sales and 0% of 2024 sales. Yearlings can't have raced at the sale. The `Horse Name` column also holds post-sale race names. The sire and broodmare-sire statistics were pulled at the same time, so they include crops that raced after the sale. | Small: about −0.004 R² when removed (`05_paired_comparisons.md`). The published coefficients on `Combo_starters` (−0.334) and `Combo_earnings` (+0.254) partly measure the horse's own career, though. | Cross statistics are **excluded** from all clean models (`common.OWN_OUTCOME`). Sire and broodmare-sire statistics (`common.HINDSIGHT`) appear only in the S1 sensitivity set. |

### Automated checks (`06_leakage_checks.py`)

| Test | What would fail if there were leakage | Result |
|---|---|---|
| T1 Input audit | Any model reading price, purchaser or name columns; clean sets reading cross or hindsight stats | 21/21 pass |
| T2 Encoder | A horse's own reputation score changing when its own price changes | pass (max change 0) |
| T3 Test-label invariance | Test predictions changing when test prices are scrambled (random and year-ahead splits; Ridge, LightGBM, CatBoost) | 8/8 pass (difference exactly 0) |
| T4 Null test | CV R² above 0 when prices are randomly permuted across horses | pass (R² −0.005 to −0.046) |
| T6 Duplicates | The same horse appearing twice | pass (0) |

**Remaining non-code caveat:** in random K-fold CV, a horse's sire, dam or consignor reputation can use horses
sold at the same sale or in later years. That isn't the horse's own price, but buyers wouldn't know it before
bidding. The year-ahead evaluation removes this, and it costs only about 0.02 R² (0.640 → 0.619).

## 3. Other problems in the published ML pipeline and how each was fixed

| # | Problem | Fix |
|---|---|---|
| 1 | **Sale session left out as a predictor**, although Figure 1 shows it's the main source of price differences and it's in the catalog before bidding. | Session fixed effects (linear models) or an ordinal variable (trees). Adds +0.203 R², improving all 5 folds. The consigning agency's reputation (cross-fitted) adds another +0.005, also in all 5 folds. |
| 2 | **First-crop flag mismeasured.** "First year the sire appears in 2020–24" marks 100% of 2020 horses, against 16–29% in other years, so it's mixed up with the 2020 baseline. | Replaced by "sire new to the sale" (no yearlings in any earlier sample year), defined from 2021 on. A true first-crop flag needs dated Equineline data. |
| 3 | **Feature screening and power transforms fitted on the full sample.** | Feature sets fixed in advance. Fixed transforms (log1p). All imputation, scaling, encoding and tuning sit inside `sklearn` Pipelines and are refitted on each training fold. |
| 4 | **Bootstrap p-values on penalized Ridge coefficients.** At α ≈ 1.45 with n ≈ 3,500 the Ridge is essentially OLS, and bootstrap CIs around a penalized estimator aren't valid. | Interpretation uses OLS with standard errors clustered by sire, on cross-fitted reputation scores and the clean S2 inputs. |
| 5 | **Session-level Ridge fits not comparable** (α from 0.72 to 42.9; within-session R² compared across sessions with different variances). | One pooled model with Sire Rep × Book interactions and a cluster-robust Wald test. Session accuracy is reported as RMSE with bootstrap CIs, from one pooled model's out-of-fold predictions. |
| 6 | **Leave-one-year-out with year effects is ill-defined**, and trains on future years to predict the past. | Rolling-origin test: train on years before t, predict year t (2022–24). |
| 7 | **Inconsistent evaluation description** ("60/20/20" vs "15-fold"; CV SE 0.0075 vs ±0.0279). | One protocol: 5-fold outer CV stratified by session, 4–5-fold inner CV for tuning. |
| 8 | **No uncertainty for individual predictions.** | Split-conformal prediction intervals calibrated on earlier years: 90% intervals cover 89–93% of the next year's horses. |

## 4. Main results (clean specification)

### 4.1 Nested CV, N = 6,416 — `results/02_nested_cv.md`

| Model | Features | R² | SE | RMSE | MAE | Median abs. % error ($) |
|---|---|---|---|---|---|---|
| Ridge | F0: published predictors, reputation leak removed | 0.429 | 0.010 | 1.072 | 0.851 | 59.6% |
| Ridge | **S2: clean, catalog-only** | 0.633 | 0.009 | 0.858 | 0.665 | 48.4% |
| LightGBM | S2 | 0.636 | 0.008 | 0.855 | 0.662 | 48.6% |
| CatBoost (native target stats) | S2 | 0.638 | 0.009 | 0.853 | 0.662 | 48.4% |
| **Average ensemble** | S2 | **0.640** | 0.009 | 0.851 | 0.659 | 48.3% |
| Ridge | S1: S2 + hindsight sire/BS stats | 0.636 | 0.009 | 0.856 | 0.662 | 48.4% |
| Average ensemble | S1 | 0.643 | 0.009 | 0.848 | 0.656 | 47.8% |

F1–F3 (with the cross statistics) are in the full table for reference only.

Fold-paired comparisons (`results/05_paired_comparisons.md`):
- Session is by far the biggest gain (+0.203).
- Consignor reputation adds +0.005.
- Hindsight sire/BS stats add +0.002.
- Nonlinear models add +0.003 to +0.006 over Ridge. The ensemble wins in all 5 folds, but the gain is economically small.

### 4.2 Predicting a later sale — `results/03_temporal.md`

| Model | 2022 | 2023 | 2024 | Mean |
|---|---|---|---|---|
| Ridge, published predictors (F0) | 0.394 | 0.365 | 0.268 | 0.342 |
| Ridge, S2 (clean) | 0.623 | 0.598 | 0.623 | 0.615 |
| LightGBM, S2 | 0.606 | 0.602 | 0.626 | 0.611 |
| CatBoost, S2 | 0.624 | 0.599 | 0.633 | 0.619 |
| **Average ensemble, S2** | 0.624 | 0.603 | 0.631 | **0.619** |
| Ridge, S1 (+ hindsight stats) | 0.635 | 0.596 | 0.629 | 0.620 |

The unforecastable year-level market shift costs less than 0.005 R². Prediction intervals on price are about
×/÷2.85 at 80% and ×/÷4.0 at 90%.

### 4.3 Interpretation (clean S2, OLS with sire-clustered SEs) — `results/04_inference_and_sessions.md`

- **Sire reputation works mostly through session placement.** A 1-SD increase goes with +0.81 log points without
  session effects, but only +0.24 with them. Better-bred yearlings are placed in earlier sessions; within a
  session, sire reputation still matters, but much less.
- **Dam Reputation is weak** once its leak is removed (+0.017, p = 0.07 with session FE). Permutation
  importance ≈ 0. Most dams have no other sales in the sample to build a reputation from.
- Consignor reputation (+0.09 per SD), colts (+0.11 log points ≈ +11%) and sire-new-to-sale (+0.09) stay
  significant with session FE. Dosage terms are small.
- **The sire-reputation slope falls across books**: 0.33 (Sessions 1–2), then 0.31, 0.27, 0.21, 0.17, down to 0.03
  (Sessions 11–12). The Wald test that the slopes are equal gives p = 0.0001. Caveat: sessions are assigned partly
  on expected value, so within-session slopes are affected by that selection.
- **Accuracy by session**: RMSE rises from about 0.62 (Session 1) to 0.90–0.97 (Sessions 6–9). Median absolute
  dollar error rises from 37% to 55%. Within-session R² is 0.07–0.21; the paper's 0.21–0.37 was inflated by leakage.

## 5. What this means for the paper's ML sections

1. **Headline:** about 0.62 out-of-sample R² when predicting the next sale from catalog information (0.64 in
   nested CV), not 0.54. The published predictors alone reach only about 0.34–0.43 without leakage.
2. **Main message:** sale session is the dominant pre-sale signal, and reputation signals add value within
   sessions. Keeneland places yearlings after inspecting them, so session partly reflects an expert judgement of
   conformation. Say so.
3. **Model choice:** Ridge or OLS is the readable benchmark. The ensemble is the best predictor, but ML adds little
   (+0.006) once session is included.
4. **Methods text must state** that reputation indices are cross-fitted or built from earlier years only, that
   cross statistics are excluded because they contain the horse's own racing record, and that all tuning is
   nested.

## 6. Remaining limitations (cannot be fixed with `sold.csv`)

- **Dated pedigree statistics.** To use sire and broodmare-sire records without hindsight, the paper needs
  Equineline snapshots as of each catalog date. The S1 results bound the gain from those statistics at about
  +0.002 R².
- **Unsold horses aren't in the data** (no RNAs or buybacks), so the model describes sold horses only.
- **Prices in `sold.csv` are not round hammer prices** (for example 155,503.608), so they look inflation-adjusted.
  Check this before quoting dollar figures.
- No hip-level variables (foaling date, dam produce record, X-rays or conformation). These are the most likely
  sources of further gains.
