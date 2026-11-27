"""Step 2 - Nested cross-validated comparison of feature sets and models.

Outer loop: 5-fold CV stratified by session (every horse is predicted exactly
once by a model that never saw its price). Inner loop: hyperparameters,
reputation-encoder choice and Ridge alpha are tuned on the outer-training
fold only. No rows are removed on the basis of price.

Usage: python 02_nested_cv.py [model_key ...]   (default: all)
Out-of-fold predictions are cached in results/oof/ so runs are incremental.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from common import RESULTS, SEED, fmt_table, load_data, metrics
from models import (FEATURE_SET_LABELS, catboost_model, lgbm_model, linear_model,
                    rf_model, xgb_model)

OOF_DIR = os.path.join(RESULTS, "oof")
os.makedirs(OOF_DIR, exist_ok=True)

MODELS = {
    "ridge_F0": ("Ridge", "F0", lambda: linear_model("F0", "ridge")),
    "ridge_F1": ("Ridge", "F1", lambda: linear_model("F1", "ridge")),
    "ridge_F2": ("Ridge", "F2", lambda: linear_model("F2", "ridge")),
    "ridge_F3": ("Ridge", "F3", lambda: linear_model("F3", "ridge")),
    "ols_F3": ("OLS", "F3", lambda: linear_model("F3", "ols")),
    "huber_F3": ("Huber", "F3", lambda: linear_model("F3", "huber")),
    "lgbm_F0": ("LightGBM", "F0", lambda: lgbm_model("F0")),
    "lgbm_F3": ("LightGBM", "F3", lambda: lgbm_model("F3")),
    "xgb_F3": ("XGBoost", "F3", lambda: xgb_model("F3")),
    "rf_F3": ("Random forest", "F3", lambda: rf_model("F3")),
    "cat_F3": ("CatBoost", "F3", lambda: catboost_model("F3")),
    # Clean feature sets (no post-sale information; see common.OWN_OUTCOME)
    "ridge_S2": ("Ridge", "S2", lambda: linear_model("S2", "ridge")),
    "lgbm_S2": ("LightGBM", "S2", lambda: lgbm_model("S2")),
    "cat_S2": ("CatBoost", "S2", lambda: catboost_model("S2")),
    "ridge_S1": ("Ridge", "S1", lambda: linear_model("S1", "ridge")),
    "lgbm_S1": ("LightGBM", "S1", lambda: lgbm_model("S1")),
    "cat_S1": ("CatBoost", "S1", lambda: catboost_model("S1")),
}
ENSEMBLES = {
    "ensemble_F3": ("Average ensemble (Ridge + LightGBM + CatBoost)", "F3", ["ridge_F3", "lgbm_F3", "cat_F3"]),
    "ensemble_S1": ("Average ensemble (Ridge + LightGBM + CatBoost)", "S1", ["ridge_S1", "lgbm_S1", "cat_S1"]),
    "ensemble_S2": ("Average ensemble (Ridge + LightGBM + CatBoost)", "S2", ["ridge_S2", "lgbm_S2", "cat_S2"]),
}

df = load_data()
outer = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(df, df["Session"]))
fold_id = np.zeros(len(df), int)
for k, (_, te) in enumerate(outer):
    fold_id[te] = k


def run_model(key):
    path = os.path.join(OOF_DIR, f"{key}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    _, _, make = MODELS[key]
    pred = np.full(len(df), np.nan)
    params = []
    t0 = time.time()
    for k, (tr, te) in enumerate(outer):
        m = make()
        m.fit(df.iloc[tr], df["y"].iloc[tr])
        pred[te] = m.predict(df.iloc[te])
        params.append(str(m.best_params_))
        print(f"  {key} fold {k}: R2={metrics(df['y'].iloc[te], pred[te])['R2']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    out = pd.DataFrame({"row": np.arange(len(df)), "fold": fold_id, "pred": pred})
    out.to_csv(path, index=False)
    with open(os.path.join(OOF_DIR, f"{key}_params.txt"), "w") as f:
        f.write("\n".join(params))
    return out


def summarize():
    rows = []
    preds = {}
    for key, (name, fs, _) in MODELS.items():
        path = os.path.join(OOF_DIR, f"{key}.csv")
        if not os.path.exists(path):
            continue
        preds[key] = pd.read_csv(path)["pred"].values
    names = {k: (v[0], v[1]) for k, v in MODELS.items()}
    for ek, (ename, efs, members) in ENSEMBLES.items():
        if all(k in preds for k in members):
            preds[ek] = np.mean([preds[k] for k in members], axis=0)
            names[ek] = (ename, efs)
    for key, p in preds.items():
        fold_m = pd.DataFrame([metrics(df["y"].values[fold_id == k], p[fold_id == k]) for k in range(5)])
        pooled = metrics(df["y"].values, p)
        rows.append({"Model": names[key][0], "Features": names[key][1],
                     "R2": fold_m["R2"].mean(), "R2_SE": fold_m["R2"].std(ddof=1) / np.sqrt(5),
                     "RMSE": fold_m["RMSE"].mean(), "MAE": fold_m["MAE"].mean(),
                     "MdAPE_$": fold_m["MdAPE_$"].mean(), "R2_pooled_OOF": pooled["R2"]})
    res = pd.DataFrame(rows)
    order = {fs: i for i, fs in enumerate(["F0", "F1", "F2", "F3", "S1", "S2"])}
    res = res.sort_values(["Features", "R2"], key=lambda c: c.map(order) if c.name == "Features" else c,
                          kind="stable").set_index("Model")
    res.to_csv(os.path.join(RESULTS, "02_nested_cv.csv"))
    pd.DataFrame(preds).assign(y=df["y"].values, fold=fold_id, Session=df["Session"].values,
                               Year=df["Year"].values).to_csv(os.path.join(RESULTS, "02_oof_predictions.csv"), index=False)
    with open(os.path.join(RESULTS, "02_nested_cv.md"), "w") as f:
        f.write("# Step 2: nested cross-validation (5 outer folds, stratified by session)\n\n")
        f.write("All preprocessing, reputation indices and tuning are refitted inside each outer-training fold. "
                "R2/RMSE/MAE on log price, mean over outer folds (SE = fold SD / sqrt 5). "
                "MdAPE = median absolute % error in dollars. N = 6,416, no price-based exclusions.\n\n")
        f.write("Feature sets:\n\n")
        for k, v in FEATURE_SET_LABELS.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n**S2 is the main (leak-free) specification.** F0-F3 contain the sire x broodmare-sire cross "
                "statistics, which include the horse's own post-sale racing record (see "
                "`06_leakage_checks.md`); they are kept only to trace the path from the published model.\n\n")
        f.write(fmt_table(res) + "\n")
    print(open(os.path.join(RESULTS, "02_nested_cv.md")).read())


if __name__ == "__main__":
    keys = sys.argv[1:] or list(MODELS)
    for k in keys:
        print(f"== {k}", flush=True)
        run_model(k)
    summarize()
