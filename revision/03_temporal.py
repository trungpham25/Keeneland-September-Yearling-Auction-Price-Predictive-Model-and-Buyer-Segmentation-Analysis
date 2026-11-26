"""Step 3 - Out-of-time evaluation (rolling origin), replacing the paper's
leave-one-year-out check.

For each test year t in 2022-2024 the model is trained only on sales from
years < t, and reputation indices for year-t horses come only from earlier
sales, which is exactly the information available in the catalog. The year-t
market level is unknown at prediction time, so test rows are assigned the
most recent training year's effect (a "last year's market" forecast). The
level shift is therefore part of the reported error.

It also produces split-conformal prediction intervals (overall and by book,
since price dispersion differs strongly across sessions) and checks their
coverage on the held-out year.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict

from common import RESULTS, SEED, fmt_table, load_data, metrics
from models import catboost_model, lgbm_model, linear_model

df = load_data()
TEST_YEARS = [2022, 2023, 2024]
MODELS = {
    "Ridge, published predictors (F0)": lambda: linear_model("F0", "ridge"),
    "Ridge, F2": lambda: linear_model("F2", "ridge"),
    "LightGBM, F3": lambda: lgbm_model("F3", n_iter=20),
    "CatBoost, F3": lambda: catboost_model("F3", n_iter=6),
}

rows, cover_rows = [], []
for t in TEST_YEARS:
    tr = df[df["Year"] < t].copy()
    te = df[df["Year"] == t].copy()
    te["Year"] = tr["Year"].max()  # unknown market level -> last observed year
    preds = {}
    for name, make in MODELS.items():
        m = make()
        m.fit(tr, tr["y"])
        preds[name] = m.predict(te)
        r = metrics(te["y"], preds[name])
        # Level-adjusted R2: removes the unforecastable year-level shift, to
        # show how much of the error is the market level vs cross-section.
        shift = np.mean(te["y"] - preds[name])
        r["R2_level_adj"] = metrics(te["y"], preds[name] + shift)["R2"]
        r["mean_bias"] = -shift
        rows.append({"Test year": t, "Train years": f"2020-{t - 1}", "Model": name, "n_test": len(te), **r})
        print(t, name, round(r["R2"], 4), flush=True)
    ens = np.mean([preds["Ridge, F2"], preds["LightGBM, F3"], preds["CatBoost, F3"]], axis=0)
    r = metrics(te["y"], ens)
    shift = np.mean(te["y"] - ens)
    r["R2_level_adj"] = metrics(te["y"], ens + shift)["R2"]
    r["mean_bias"] = -shift
    rows.append({"Test year": t, "Train years": f"2020-{t - 1}", "Model": "Average ensemble",
                 "n_test": len(te), **r})

    # --- Split-conformal intervals for the Ridge F2 model ------------------
    # Calibration residuals: cross-fitted predictions on the training years.
    cal_pred = cross_val_predict(linear_model("F2", "ridge"), tr, tr["y"],
                                 cv=KFold(5, shuffle=True, random_state=SEED))
    abs_res = np.abs(tr["y"].values - cal_pred)
    p_te = preds["Ridge, F2"]
    for level in (0.80, 0.90):
        q = np.quantile(abs_res, level * (1 + 1 / len(abs_res)))
        cov = np.mean(np.abs(te["y"].values - p_te) <= q)
        # Mondrian (by book) intervals
        cov_b, widths = [], []
        inside = np.zeros(len(te), bool)
        for b in df["Book"].cat.categories:
            cm = (tr["Book"] == b).values
            tm = (te["Book"] == b).values
            qb = np.quantile(abs_res[cm], min(1.0, level * (1 + 1 / cm.sum())))
            inside[tm] = np.abs(te["y"].values[tm] - p_te[tm]) <= qb
            widths.append(np.exp(qb))
        cover_rows.append({"Test year": t, "Nominal": level, "Coverage (global)": cov,
                           "Price ratio band (global)": f"x/{np.exp(q):.2f} to x*{np.exp(q):.2f}",
                           "Coverage (by book)": inside.mean(),
                           "Band range by book": f"x*{min(widths):.2f} to x*{max(widths):.2f}"})

res = pd.DataFrame(rows)
res.to_csv(f"{RESULTS}/03_temporal.csv", index=False)
cov = pd.DataFrame(cover_rows)
cov.to_csv(f"{RESULTS}/03_conformal.csv", index=False)

avg = res.groupby("Model")[["R2", "R2_level_adj", "RMSE", "MAE", "MdAPE_$"]].mean()
with open(f"{RESULTS}/03_temporal.md", "w") as f:
    f.write("# Step 3: out-of-time (rolling-origin) evaluation\n\n")
    f.write("Train on all sales before year t, predict year t. Reputation indices for year-t horses use earlier "
            "sales only. The year-t market level is not known in advance, so test rows get the latest training "
            "year's effect; `R2_level_adj` removes that unforecastable common shift.\n\n")
    f.write(fmt_table(res.set_index("Test year")) + "\n\n")
    f.write("## Average over the three test years\n\n" + fmt_table(avg) + "\n\n")
    f.write("## Split-conformal prediction intervals (Ridge F2)\n\n")
    f.write("Calibrated on cross-fitted residuals from the training years, evaluated on the held-out year. "
            "Bands are multiplicative on price.\n\n")
    f.write(fmt_table(cov.set_index("Test year"), floatfmt=".3f") + "\n")
print(open(f"{RESULTS}/03_temporal.md").read())
