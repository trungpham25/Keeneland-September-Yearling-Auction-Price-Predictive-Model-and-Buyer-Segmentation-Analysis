"""Step 1 - Reproduce the published Ridge pipeline and measure how much of its
out-of-sample R^2 comes from information leakage.

Variants (same 20 predictors, same Ridge, same 60/20/20 session-stratified
split, repeated over many random splits):

  A  Published procedure: reputation indices computed on the full sample
     (each horse's own price enters its sire/dam index), Cook's-distance
     outliers removed from the full sample before splitting, paper's
     first-crop definition.
  B  A without the full-sample Cook's-distance filter.
  C  B with leak-free (cross-fitted) reputation indices.
  D  C with the corrected first-crop proxy.

All variants are scored on every test row (no test rows removed), except A,
which follows the paper and scores on the filtered sample.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.pipeline import Pipeline

from common import (PAPER_NUMERIC, RESULTS, RIDGE_ALPHAS, SEED, build_preprocessor,
                    fmt_table, load_data, metrics, reputation_table)

N_SPLITS = 20
MIN_N = {"Sire_key": 5, "Dam_key": 2}  # low-volume rule from the original code

df = load_data()

# Full-sample ("leaky") reputation indices exactly as described in the paper.
for key, name in [("Sire_key", "Sire_Rep_full"), ("Dam_key", "Dam_Rep_full")]:
    tab, prior = reputation_table(df["y"].values, df[key].values, "index", 0.0, MIN_N[key])
    df[name] = df[key].map(tab).fillna(prior)

# How much does a dam's "reputation" depend on the horse's own price?
dam_n = df.groupby("Dam_key")["y"].transform("size")
share_single = float((dam_n == 1).mean())
share_two = float((dam_n == 2).mean())
corr_two = float(np.corrcoef(df.loc[dam_n == 2, "Dam_Rep_full"], df.loc[dam_n == 2, "y"])[0, 1])


def cooks_filter(d, cols):
    X = d[cols].copy()
    X = X.fillna(X.median())
    X = pd.get_dummies(X, columns=["Year"], drop_first=True, dtype=float)
    ols = sm.OLS(d["y"].values, sm.add_constant(X.values)).fit()
    cd = ols.get_influence().cooks_distance[0]
    return cd <= 4 / len(d)


def run(variant, seed):
    d = df.copy()
    fc = "First_crop_paper" if variant in "ABC" else "Sire_new_to_sale"
    if variant == "A":
        keep = cooks_filter(d, PAPER_NUMERIC + ["Sire_Rep_full", "Dam_Rep_full", fc, "Year"])
        d = d[keep].reset_index(drop=True)
    tr, rest = train_test_split(d, test_size=0.4, stratify=d["Session"], random_state=seed)
    va, te = train_test_split(rest, test_size=0.5, stratify=rest["Session"], random_state=seed)

    if variant in "AB":
        pre = build_preprocessor(PAPER_NUMERIC + ["Sire_Rep_full", "Dam_Rep_full", fc], onehot=["Year"])
    else:
        pre = build_preprocessor(PAPER_NUMERIC + [fc], onehot=["Year"],
                                 rep_cols=["Sire_key", "Dam_key"], rep_method="index",
                                 rep_min_n=MIN_N)
    model = Pipeline([("pre", pre),
                      ("ridge", RidgeCV(alphas=RIDGE_ALPHAS,
                                        cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=seed)))])
    model.fit(tr, tr["y"])
    m = metrics(te["y"], model.predict(te), p=20)
    m["alpha"] = model.named_steps["ridge"].alpha_
    m["n_sample"] = len(d)
    return m


rows = []
for v in "ABCD":
    for s in range(N_SPLITS):
        r = run(v, SEED + s)
        r["variant"] = v
        rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(f"{RESULTS}/01_leakage_audit_splits.csv", index=False)

labels = {
    "A": "A. Published procedure (full-sample reputation + full-sample Cook's filter)",
    "B": "B. A without the full-sample Cook's filter",
    "C": "C. B with leak-free (cross-fitted) reputation",
    "D": "D. C with corrected first-crop proxy",
}
summ = res.groupby("variant").agg(
    R2=("R2", "mean"), R2_sd=("R2", "std"), RMSE=("RMSE", "mean"), MAE=("MAE", "mean"),
    MdAPE=("MdAPE_$", "mean"), alpha_median=("alpha", "median"), n=("n_sample", "first"))
summ.index = [labels[i] for i in summ.index]
summ.index.name = "Variant"

with open(f"{RESULTS}/01_leakage_audit.md", "w") as f:
    f.write("# Step 1: leakage audit of the published Ridge model\n\n")
    f.write(f"Mean over {N_SPLITS} random 60/20/20 session-stratified splits; test-set metrics on log price. "
            "MdAPE = median absolute % error in dollars.\n\n")
    f.write(fmt_table(summ) + "\n\n")
    f.write(f"- Reputation indices follow the original code: sires with <5 sales and dams with <2 sales "
            f"get a default score ({share_single:.1%} of rows have a dam with a single yearling).\n")
    f.write(f"- {share_two:.1%} of rows have a dam with exactly two yearlings in the sample; for them the "
            f"full-sample Dam Reputation is built from two prices, one of which is the horse's own "
            f"(correlation with own log price = {corr_two:.3f}).\n")
    f.write(f"- Cook's-distance rule D > 4/n on the full sample removes "
            f"{6416 - int(res.loc[res.variant == 'A', 'n_sample'].iloc[0])} rows.\n")
print(open(f"{RESULTS}/01_leakage_audit.md").read())
