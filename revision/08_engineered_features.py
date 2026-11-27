"""Step 8 - Mechanical dependence among engineered features.

Every engineered feature is built from columns of the same dataset. Even
without leakage, that can tie features to each other (or to session) by
construction, which distorts coefficients. This step:

  A. maps each engineered feature to its source columns;
  B. measures the mechanical links:
     B1 reputation scores (built from prices) vs session, since prices are
        sorted by session;
     B2 progeny ratios that share the same denominator (starters);
     B3 reputation scores vs their own sample-size / default rules;
     B4 the four components inside the reputation index;
     B5 the sire-new-to-sale flag vs year and prior-sales counts;
  C. tests a fix for B1, within-session reputation (prices relative to their
     year x session mean, cell means from training rows only), for collinearity,
     coefficient stability and prediction (nested CV and year-ahead).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from statsmodels.stats.outliers_influence import variance_inflation_factor

from common import RESULTS, SEED, CrossFitReputation, _index_components, fmt_table, load_data
from models import MIN_N, linear_model

df = load_data()
out = ["# Step 8: mechanical dependence among engineered features\n"]
KEYS = ("Sire_key", "Dam_key", "Consignor_key")
FIVE = ["Brilliant", "Intermediate", "Classic", "Solid", "Professional"]

# ---------------------------------------------------------------------------
# A. Derivation map
# ---------------------------------------------------------------------------
dmap = pd.DataFrame([
    ("Sire / Dam / Consignor Reputation", "Price of other horses by the same sire / dam / consignor", "yes (others' prices, cross-fitted)", "Session (prices are sorted by session)"),
    ("Sire/Dam/Consignor log_n_prior", "Count of other horses by the same key", "no", "Reputation (default score below min_n)"),
    ("Sire_new_to_sale", "Year, sire's first sample year", "no", "Year (0 in 2020 by construction), sire counts"),
    ("Sire_BW_rate, Sire_win_rate", "Sire_BW, Sire_winners / Sire_starters", "no", "Each other and log_Sire_starters (same denominator)"),
    ("log_Sire_earn_per_starter", "Sire_earnings / Sire_starters", "no", "Same denominator as the rates"),
    ("BS_BW_rate, BS_win_rate, log_BS_earn_per_starter", "BS_* / BS_starters", "no", "Same denominator"),
    ("Dosage_Index, Center_of_Distribution", "The five dosage points", "no", "Five dosage points (deterministic functions)"),
    ("log_* counts", "log1p of raw counts", "no", "Raw counts (monotone transform)"),
], columns=["Engineered feature", "Built from", "Uses price?", "Mechanically tied to"]).set_index("Engineered feature")
out.append("## A. Derivation map\n\n" + fmt_table(dmap) + "\n")

# Cross-fitted raw and within-session reputation on the full sample
cols = list(KEYS) + ["Year", "Session"]
raw = CrossFitReputation(cols=KEYS, method="index", min_n=MIN_N, add_count=True).fit_transform(df[cols], df["y"])
wit = CrossFitReputation(cols=KEYS, method="index", min_n=MIN_N, add_count=True,
                         adjust_cols=("Year", "Session")).fit_transform(df[cols], df["y"])
for j, k in enumerate(["Sire", "Dam", "Consignor"]):
    df[f"{k}_Rep"] = raw[:, 2 * j]
    df[f"{k}_n"] = raw[:, 2 * j + 1]
    df[f"{k}_RepW"] = wit[:, 2 * j]
S = pd.get_dummies(df["Session"], prefix="Session", drop_first=True, dtype=float)
YS = pd.concat([S, pd.get_dummies(df["Year"], prefix="Year", drop_first=True, dtype=float)], axis=1)


def r2_on(v, X):
    return sm.OLS(v, sm.add_constant(X)).fit().rsquared


# ---------------------------------------------------------------------------
# B1 reputation vs session
# ---------------------------------------------------------------------------
rows = []
for k in ["Sire", "Dam", "Consignor"]:
    for lab, c in [("raw prices (S2)", f"{k}_Rep"), ("within-session (S2w)", f"{k}_RepW")]:
        r2 = r2_on(df[c], YS)
        rows.append({"Reputation": f"{k}, {lab}", "R2 on year+session FE": r2, "VIF from session alone": 1 / (1 - r2),
                     "corr with session no.": np.corrcoef(df[c], df["Session"])[0, 1]})
b1 = pd.DataFrame(rows).set_index("Reputation")
out.append("## B1. Reputation scores vs session\n")
out.append("Reputation is built from sale prices, and prices are sorted by session, so a raw-price reputation "
           "partly re-encodes *which sessions* a sire's yearlings are placed in. That overlap is built into the "
           "feature.\n")
out.append(fmt_table(b1, floatfmt=".3f") + "\n")

# ---------------------------------------------------------------------------
# B2 shared denominators
# ---------------------------------------------------------------------------
sire = ["log_Sire_starters", "Sire_BW_rate", "Sire_win_rate", "log_Sire_earn_per_starter", "Sire_aei"]
cm = df[sire].corr()
X = (df[sire] - df[sire].mean()) / df[sire].std()
vifs = {c: variance_inflation_factor(sm.add_constant(X).values, i + 1) for i, c in enumerate(sire)}
raw_parts = ["log_Sire_starters", "Sire_winners", "Sire_BW", "Sire_earnings"]
cm_raw = pd.DataFrame({"log_Sire_starters": np.log1p(df["Sire_starters"]),
                       "log_Sire_winners": np.log1p(df["Sire_winners"]),
                       "log_Sire_BW": np.log1p(df["Sire_BW"]),
                       "log_Sire_earnings": np.log1p(df["Sire_earnings"])}).corr()
out.append("## B2. Progeny ratios that share a denominator (S1 sensitivity set only)\n")
out.append("Win rate, black-type rate and earnings per starter all divide by the same starters count, and AEI is "
           "an earnings ratio as well, so they move together by construction:\n")
out.append(fmt_table(cm, floatfmt=".2f") + "\n")
out.append("VIF within this block: " + ", ".join(f"{c} {v:.1f}" for c, v in vifs.items()) + ".\n")
out.append(f"The underlying log counts are even more tightly linked (corr. log starters vs log winners = "
           f"{cm_raw.loc['log_Sire_starters', 'log_Sire_winners']:.3f}), because a sire's winners are a subset of "
           "his starters. These statistics are therefore only used for prediction (S1), never interpreted one by "
           "one.\n")

# ---------------------------------------------------------------------------
# B3 reputation vs sample-size rules
# ---------------------------------------------------------------------------
rows = []
for k, key in [("Sire", "Sire_key"), ("Dam", "Dam_key"), ("Consignor", "Consignor_key")]:
    # n_prior is log1p(count of the key's sales in the other folds)
    below = (np.expm1(df[f"{k}_n"]).round() < MIN_N[key]).mean()
    rows.append({"Reputation": k, "min_n": MIN_N[key],
                 "share of rows given the default score": below,
                 "corr(score, log prior count)": np.corrcoef(df[f"{k}_Rep"], df[f"{k}_n"])[0, 1]})
b3 = pd.DataFrame(rows).set_index("Reputation")
out.append("## B3. Reputation scores vs their sample-size rules\n")
out.append("Keys with fewer than min_n prior sales all get the same default score. That creates a spike at one "
           "value and ties the score to how many prior sales exist, independent of quality:\n")
out.append(fmt_table(b3, floatfmt=".3f") + "\n")

# ---------------------------------------------------------------------------
# B4 index components
# ---------------------------------------------------------------------------
comp = _index_components(df["y"].values, df["Sire_key"].values, np.quantile(df["y"], 0.9))
comp = comp[comp["n"] >= MIN_N["Sire_key"]]
comp["med_rank"] = comp["med"].rank(pct=True)
cc = comp[["p90", "elite", "med_rank", "p25"]].corr()
out.append("## B4. The four components of the reputation index (sires with >= 5 sales)\n")
out.append("p90, elite share, median rank and p25 of the same sire's prices are near-duplicates, so the 0.40/0.30/"
           "0.20/0.10 weighting changes little (the index is essentially one dimension):\n")
out.append(fmt_table(cc, floatfmt=".2f") + "\n")
ev = np.linalg.eigvalsh(cc.values)[::-1]
out.append(f"First principal component explains {ev[0] / ev.sum():.0%} of the variance of the four components.\n")

# ---------------------------------------------------------------------------
# B5 sire-new-to-sale
# ---------------------------------------------------------------------------
out.append("## B5. Sire-new-to-sale flag\n")
nts = df.groupby("Year")["Sire_new_to_sale"].mean()
out.append("Share flagged by year: " + ", ".join(f"{y} {v:.0%}" for y, v in nts.items()) +
           f". Corr. with log prior sire sales: {np.corrcoef(df['Sire_new_to_sale'], df['Sire_n'])[0, 1]:.2f}; "
           f"mean raw Sire Reputation {df.loc[df.Sire_new_to_sale == 1, 'Sire_Rep'].mean():.3f} (new) vs "
           f"{df.loc[df.Sire_new_to_sale == 0, 'Sire_Rep'].mean():.3f} (others).\n")

# ---------------------------------------------------------------------------
# C. Within-session reputation: collinearity, coefficients, prediction
# ---------------------------------------------------------------------------
groups = df["Sire_key"].astype("category").cat.codes.values


def ols(rep_suffix, session):
    cont = [f"Sire_{rep_suffix}", f"Dam_{rep_suffix}", f"Consignor_{rep_suffix}"] + FIVE
    X = df[cont].fillna(df[cont].median())
    X = (X - X.mean()) / X.std()
    for b in ["Is_male", "Sire_new_to_sale", "Dosage_missing"]:
        X[b] = df[b].astype(float)
    fac = ["Year", "Session", "Color"] if session else ["Year", "Color"]
    for f in fac:
        X = pd.concat([X, pd.get_dummies(df[f], prefix=f, drop_first=True, dtype=float)], axis=1)
    fit = sm.OLS(df["y"].values, sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": groups})
    vif = variance_inflation_factor(sm.add_constant(X).values, list(X.columns).index(f"Sire_{rep_suffix}") + 1)
    return fit, vif


rows = []
for lab, suf in [("Raw-price reputation (S2)", "Rep"), ("Within-session reputation (S2w)", "RepW")]:
    fa, _ = ols(suf, False)
    fb, vif = ols(suf, True)
    rows.append({"Specification": lab,
                 "Sire Rep coef, no session FE": fa.params[f"Sire_{suf}"],
                 "Sire Rep coef, session FE": fb.params[f"Sire_{suf}"],
                 "SE (session FE)": fb.bse[f"Sire_{suf}"],
                 "Sire Rep VIF (session FE)": vif,
                 "Consignor Rep coef (session FE)": fb.params[f"Consignor_{suf}"],
                 "Sire-new-to-sale coef (session FE)": fb.params["Sire_new_to_sale"],
                 "p (new-to-sale)": fb.pvalues["Sire_new_to_sale"],
                 "OLS R2 (session FE)": fb.rsquared})
c1 = pd.DataFrame(rows).set_index("Specification")
c1.to_csv(f"{RESULTS}/08_within_session_reputation_ols.csv")

cv = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(df, df["Session"]))
pred_rows = []
fold_r2 = {}
for fs in ["S2", "S2w"]:
    p = cross_val_predict(linear_model(fs, "ridge"), df, df["y"], cv=cv)
    r = [r2_score(df["y"].values[te], p[te]) for _, te in cv]
    yr = []
    for t in (2022, 2023, 2024):
        tr = df[df["Year"] < t]
        te = df[df["Year"] == t].copy()
        te["Year"] = tr["Year"].max()
        m = linear_model(fs, "ridge").fit(tr, tr["y"])
        yr.append(r2_score(te["y"], m.predict(te)))
    fold_r2[fs] = np.array(r)
    pred_rows.append({"Feature set": fs, "Ridge nested-CV R2": np.mean(r), "fold SD": np.std(r, ddof=1),
                      "Ridge year-ahead R2 (2022-24 mean)": np.mean(yr),
                      "2022": yr[0], "2023": yr[1], "2024": yr[2]})
c2 = pd.DataFrame(pred_rows).set_index("Feature set")
c2.to_csv(f"{RESULTS}/08_within_session_reputation_prediction.csv")

out.append("## C. Fix for B1: within-session reputation\n")
out.append("Reputation built from each sale price minus the mean log price of its year x session cell (cell "
           "means from training rows only). It measures how a sire's yearlings sell *relative to others in the "
           "same session*, so it no longer re-encodes session placement.\n")
out.append("### Coefficients (OLS, sire-clustered SEs, standardized predictors)\n")
out.append(fmt_table(c1, floatfmt=".3f") + "\n")
out.append("### Prediction\n")
out.append(fmt_table(c2, floatfmt=".4f") + "\n")
dlt = fold_r2["S2w"] - fold_r2["S2"]
out.append(f"Paired fold differences S2w - S2: {', '.join(f'{v:+.4f}' for v in dlt)} "
           f"(mean {dlt.mean():+.4f}, {int((dlt > 0).sum())}/5 folds improved).\n")

with open(f"{RESULTS}/08_engineered_features.md", "w") as f:
    f.write("\n".join(out) + "\n")
print("\n".join(out))
