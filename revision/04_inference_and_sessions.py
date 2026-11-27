"""Step 4 - Interpretation and session heterogeneity, done without leakage
and with valid standard errors.

(a) Coefficients: OLS (not bootstrapped Ridge; with alpha ~ 1 at n > 6,000 the
    Ridge estimates are essentially OLS anyway, and bootstrap CIs around a
    penalized estimator are not valid inference). Reputation indices are
    cross-fitted over the full sample, so no horse's own price enters its
    regressors. Standard errors are clustered by sire. Continuous predictors
    are standardized (coefficient = change in log price per 1 SD).
    Two specifications: without and with session fixed effects, to show how
    much of the "sire reputation effect" runs through session placement.

(b) Sessions: (i) predictive accuracy by session from the nested-CV
    out-of-fold predictions of one pooled model, with bootstrap CIs, reported
    as RMSE (comparable across sessions) and within-session R2 (not
    comparable, because within-session price variance differs); (ii) a single
    pooled OLS with Sire-Reputation x Book interactions and a cluster-robust
    Wald test, instead of separately tuned per-session Ridge fits whose
    coefficients are shrunk by different amounts.

(c) Grouped permutation importance on the outer test folds.
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

from common import RESULTS, SEED, CrossFitReputation, DOSAGE, fmt_table, load_data, metrics
from models import FEATURE_SETS, MIN_N, lgbm_model, linear_model

df = load_data()
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# (a) OLS with cluster-robust SEs
# ---------------------------------------------------------------------------
enc = CrossFitReputation(cols=("Sire_key", "Dam_key", "Consignor_key"), method="index", min_n=MIN_N)
R = enc.fit_transform(df[["Sire_key", "Dam_key", "Consignor_key"]], df["y"])
df["Sire_Reputation"], df["Dam_Reputation"], df["Consignor_Reputation"] = R.T

# Clean (S2) specification: no cross statistics (own racing record) and no
# progeny statistics pulled after the sale.
# Dosage Index and Center of Distribution are deterministic functions of the
# five dosage points (Step 7), so they are left out of the coefficient table
# to avoid inflating the dosage-point variances; prediction is unaffected.
cont = ["Sire_Reputation", "Dam_Reputation", "Consignor_Reputation"] + [
    c for c in DOSAGE if c not in ("Dosage_missing", "Dosage_Index", "Center_of_Distribution")]
binary = ["Is_male", "Sire_new_to_sale", "Dosage_missing"]
X0 = df[cont].copy()
X0 = X0.fillna(X0.median())
X0 = (X0 - X0.mean()) / X0.std()
X0[binary] = df[binary]
yr = pd.get_dummies(df["Year"], prefix="Year", drop_first=True, dtype=float)
ses = pd.get_dummies(df["Session"], prefix="Session", drop_first=True, dtype=float)
groups = df["Sire_key"].astype("category").cat.codes.values


def ols(X):
    return sm.OLS(df["y"].values, sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": groups})


fit_a = ols(pd.concat([X0, yr], axis=1))
fit_b = ols(pd.concat([X0, yr, ses], axis=1))


def coef_table(fit):
    t = pd.DataFrame({"coef": fit.params, "SE": fit.bse, "p": fit.pvalues,
                      "CI_low": fit.conf_int()[0], "CI_high": fit.conf_int()[1]})
    return t.drop(index="const")


ta, tb = coef_table(fit_a), coef_table(fit_b)
coefs = pd.concat({"No session FE": ta, "With session FE": tb}, axis=1)
coefs.to_csv(f"{RESULTS}/04_ols_coefficients.csv")
show = [c for c in cont + binary]
compact = pd.DataFrame({
    "coef (no session FE)": ta.loc[show, "coef"], "SE ": ta.loc[show, "SE"], "p ": ta.loc[show, "p"],
    "coef (session FE)": tb.loc[show, "coef"], "SE": tb.loc[show, "SE"], "p": tb.loc[show, "p"]})
compact.index.name = "Predictor"

# ---------------------------------------------------------------------------
# (b-i) Accuracy by session from pooled OOF predictions
# ---------------------------------------------------------------------------
oof = pd.read_csv(f"{RESULTS}/02_oof_predictions.csv")
model_col = "ensemble_S2" if "ensemble_S2" in oof else "ridge_S2"
sess_rows = []
for s in range(1, 13):
    m = oof["Session"].values == s
    y, p = oof["y"].values[m], oof[model_col].values[m]
    boots_r2, boots_rmse = [], []
    for _ in range(1000):
        i = rng.integers(0, m.sum(), m.sum())
        boots_r2.append(metrics(y[i], p[i])["R2"])
        boots_rmse.append(np.sqrt(np.mean((y[i] - p[i]) ** 2)))
    mm = metrics(y, p)
    sess_rows.append({"Session": s, "n": int(m.sum()), "SD log price": y.std(),
                      "RMSE": mm["RMSE"], "RMSE 95% CI": f"[{np.quantile(boots_rmse, .025):.3f}, {np.quantile(boots_rmse, .975):.3f}]",
                      "Within-session R2": mm["R2"],
                      "R2 95% CI": f"[{np.quantile(boots_r2, .025):.3f}, {np.quantile(boots_r2, .975):.3f}]",
                      "MdAPE_$": mm["MdAPE_$"]})
sess = pd.DataFrame(sess_rows).set_index("Session")
sess.to_csv(f"{RESULTS}/04_session_accuracy.csv")

# ---------------------------------------------------------------------------
# (b-ii) Pooled slope-heterogeneity test
# ---------------------------------------------------------------------------
book = pd.get_dummies(df["Book"], prefix="Book", drop_first=True, dtype=float)
inter = book.mul(X0["Sire_Reputation"], axis=0).add_suffix(":SireRep")
fit_c = ols(pd.concat([X0, yr, ses, inter], axis=1))
names = list(inter.columns)
wald = fit_c.wald_test(" = 0, ".join(names) + " = 0", scalar=True)
book_slopes = []
base = fit_c.params["Sire_Reputation"]
cov = fit_c.cov_params()
for b in df["Book"].cat.categories:
    if b == df["Book"].cat.categories[0]:
        est, se = base, np.sqrt(cov.loc["Sire_Reputation", "Sire_Reputation"])
    else:
        k = f"Book_{b}:SireRep"
        est = base + fit_c.params[k]
        se = np.sqrt(cov.loc["Sire_Reputation", "Sire_Reputation"] + cov.loc[k, k] + 2 * cov.loc["Sire_Reputation", k])
    sd_within = df.loc[df["Book"] == b, "Sire_Reputation"].std() / df["Sire_Reputation"].std()
    book_slopes.append({"Book": b, "Sire Rep slope (per full-sample SD)": est, "SE": se,
                        "Within-book SD of Sire Rep (rel. to full sample)": sd_within})
slopes = pd.DataFrame(book_slopes).set_index("Book")

# ---------------------------------------------------------------------------
# (c) Grouped permutation importance on outer test folds
# ---------------------------------------------------------------------------
GROUPS = {
    "Session": ["Session"],
    "Sire identity (reputation)": ["Sire_key"],
    "Dam identity (reputation)": ["Dam_key"],
    "Consignor identity (reputation)": ["Consignor_key"],
    "Year": ["Year"],
    "Sire progeny stats": ["log_Sire_starters", "Sire_BW_rate", "Sire_aei", "Sire_win_rate",
                           "log_Sire_earn_per_starter", "Sire_unraced_progeny"],
    "Broodmare-sire stats": ["log_BS_foals", "BS_BW_rate", "BS_aei", "BS_win_rate", "log_BS_earn_per_starter"],
    "Dosage profile": ["Brilliant", "Intermediate", "Classic", "Solid", "Professional", "Dosage_Index",
                       "Center_of_Distribution", "Dosage_missing"],
    "Sex": ["Is_male"],
    "Sire new to sale": ["Sire_new_to_sale"],
    "Color": ["Color"],
}
outer = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(df, df["Session"]))
imp = {"Ridge S1": {g: [] for g in GROUPS}, "LightGBM S1": {g: [] for g in GROUPS}}
lgb_params = open(f"{RESULTS}/oof/lgbm_S1_params.txt").read().splitlines() if os.path.exists(
    f"{RESULTS}/oof/lgbm_S1_params.txt") else None


def perm_drop(model, te, cols, base_r2, n_rep=5):
    drops = []
    for _ in range(n_rep):
        z = te.copy()
        perm = rng.permutation(len(z))
        for c in cols:
            z[c] = te[c].values[perm]
        drops.append(base_r2 - metrics(te["y"], model.predict(z))["R2"])
    return np.mean(drops)


for k, (tr, te_idx) in enumerate(outer):
    trd, ted = df.iloc[tr], df.iloc[te_idx].copy()
    models = {"Ridge S1": linear_model("S1", "ridge").fit(trd, trd["y"]).best_estimator_}
    if lgb_params:
        m = lgbm_model("S1").estimator
        m.set_params(**eval(lgb_params[k], {"np": np}))  # params saved with np.float64(...) reprs
        models["LightGBM S1"] = m.fit(trd, trd["y"])
    for name, model in models.items():
        base_r2 = metrics(ted["y"], model.predict(ted))["R2"]
        for g, cols in GROUPS.items():
            imp[name][g].append(perm_drop(model, ted, cols, base_r2))
impt = pd.DataFrame({name: {g: np.mean(v) for g, v in d.items() if v} for name, d in imp.items()})
impt = impt.sort_values(impt.columns[-1], ascending=False)
impt.index.name = "Feature group"
impt.to_csv(f"{RESULTS}/04_permutation_importance.csv")

with open(f"{RESULTS}/04_inference_and_sessions.md", "w") as f:
    f.write("# Step 4: interpretation and session heterogeneity\n\n")
    f.write("## (a) OLS coefficients, cluster-robust SEs by sire (N = 6,416)\n\n")
    f.write("Clean specification S2: catalog information only (no cross statistics, which contain the horse's own "
            "racing record, and no progeny statistics pulled after the sale). "
            "Reputation indices are cross-fitted (no horse's own price enters its index). Continuous predictors "
            "standardized; coefficients are changes in log price per 1 SD. Year FE in both columns; "
            f"session FE in the second. R2: {fit_a.rsquared:.3f} without session FE, {fit_b.rsquared:.3f} with. "
            "Derived dosage summaries (Dosage Index, Center of Distribution) are omitted because they are "
            "functions of the five dosage points. All VIFs < 2 (Step 7). The Sire-new-to-sale coefficient is a "
            "suppression effect (positive only conditional on Sire Reputation; see Step 7) and should not be read "
            "as a novelty premium.\n\n")
    f.write(fmt_table(compact) + "\n\n")
    f.write(f"## (b-i) Accuracy by session (out-of-fold predictions of `{model_col}` from Step 2)\n\n")
    f.write("RMSE is comparable across sessions; within-session R2 is not, because it divides by each "
            "session's own (much smaller and unequal) price variance.\n\n")
    f.write(fmt_table(sess, floatfmt=".3f") + "\n\n")
    f.write("## (b-ii) Does the sire-reputation slope differ by book? (pooled OLS with interactions)\n\n")
    f.write(fmt_table(slopes, floatfmt=".3f") + "\n\n")
    f.write(f"Cluster-robust Wald test that all Sire Rep x Book interactions are zero: "
            f"F = {float(np.squeeze(wald.statistic)):.2f}, p = {float(wald.pvalue):.4f}.\n\n")
    f.write("## (c) Grouped permutation importance (mean drop in out-of-fold R2 over 5 outer folds)\n\n")
    f.write("Models use the S1 set (S2 + sire/broodmare-sire progeny statistics) so the hindsight statistics can "
            "be ranked too.\n\n")
    f.write(fmt_table(impt) + "\n")
print(open(f"{RESULTS}/04_inference_and_sessions.md").read())
