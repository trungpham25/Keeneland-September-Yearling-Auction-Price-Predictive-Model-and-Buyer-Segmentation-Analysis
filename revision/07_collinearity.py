"""Step 7 - Collinearity diagnostics.

For each regression design it reports:
  - VIF for every column, and generalized VIF (Fox & Monette 1992) for factor
    blocks (Year, Session, Color), reported as GVIF^(1/(2 df)), which is
    comparable to sqrt(VIF) for a single column;
  - condition number of the column-equilibrated design and Belsley-Kuh-Welsch
    condition indices with variance-decomposition proportions (an index > 30
    with two or more proportions > 0.5 flags a harmful near-dependency);
  - sign flips: predictors whose multivariate OLS coefficient is significant
    with the opposite sign to their bivariate correlation with log price.

Designs:
  P   the published Ridge design (20 predictors, paper's first-crop flag,
      full-sample reputation as in the paper)
  S2  the revised clean OLS design (Step 4), with session FE
  S2r S2 without the two derived dosage summaries (Dosage Index and Center
      of Distribution are deterministic functions of the five dosage points)
  S1  S2 + sire/broodmare-sire progeny statistics (sensitivity set)
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from common import (DOSAGE, PAPER_NUMERIC, RESULTS, SIRE_BS_STATS, CrossFitReputation,
                    fmt_table, load_data, reputation_table)

df = load_data()
MIN_N = {"Sire_key": 5, "Dam_key": 2, "Consignor_key": 3}
enc = CrossFitReputation(cols=("Sire_key", "Dam_key", "Consignor_key"), method="index", min_n=MIN_N)
df["Sire_Reputation"], df["Dam_Reputation"], df["Consignor_Reputation"] = enc.fit_transform(
    df[["Sire_key", "Dam_key", "Consignor_key"]], df["y"]).T
for key, name in [("Sire_key", "Sire_Rep_full"), ("Dam_key", "Dam_Rep_full")]:
    tab, prior = reputation_table(df["y"].values, df[key].values, "index", 0.0, MIN_N[key])
    df[name] = df[key].map(tab).fillna(prior)

FIVE = ["Brilliant", "Intermediate", "Classic", "Solid", "Professional"]


def design(cont, binary, factors):
    X = df[cont].copy()
    X = X.fillna(X.median())
    X = (X - X.mean()) / X.std()
    for b in binary:
        X[b] = df[b].astype(float)
    blocks = {}
    for f in factors:
        dmy = pd.get_dummies(df[f], prefix=f, drop_first=True, dtype=float)
        blocks[f] = list(dmy.columns)
        X = pd.concat([X, dmy], axis=1)
    return X, blocks


DESIGNS = {
    "P": design(["Sire_Rep_full", "Dam_Rep_full"] + [c for c in PAPER_NUMERIC if c != "Is_male"],
                ["Is_male", "First_crop_paper"], ["Year"]),
    "S2": design(["Sire_Reputation", "Dam_Reputation", "Consignor_Reputation"] + FIVE
                 + ["Dosage_Index", "Center_of_Distribution"],
                 ["Is_male", "Sire_new_to_sale", "Dosage_missing"], ["Year", "Session", "Color"]),
    "S2r": design(["Sire_Reputation", "Dam_Reputation", "Consignor_Reputation"] + FIVE,
                  ["Is_male", "Sire_new_to_sale", "Dosage_missing"], ["Year", "Session", "Color"]),
    "S1": design(["Sire_Reputation", "Dam_Reputation", "Consignor_Reputation"] + FIVE + SIRE_BS_STATS,
                 ["Is_male", "Sire_new_to_sale", "Dosage_missing"], ["Year", "Session", "Color"]),
}
LABELS = {"P": "Published Ridge design", "S2": "Revised clean design (S2)",
          "S2r": "S2 without derived dosage summaries", "S1": "S1 (S2 + sire/BS progeny stats)"}


def vif_table(X, blocks):
    Xc = sm.add_constant(X).values
    cols = ["const"] + list(X.columns)
    vif = {c: variance_inflation_factor(Xc, i) for i, c in enumerate(cols) if c != "const"}
    in_block = {c for cs in blocks.values() for c in cs}
    rows = [{"term": c, "df": 1, "VIF": v, "GVIF": v, "GVIF^(1/2df)": np.sqrt(v)} for c, v in vif.items()
            if c not in in_block]
    R = np.corrcoef(X.values, rowvar=False)
    idx = {c: i for i, c in enumerate(X.columns)}
    for b, cs in blocks.items():
        a = [idx[c] for c in cs]
        o = [i for i in range(len(X.columns)) if i not in a]
        gv = np.linalg.det(R[np.ix_(a, a)]) * np.linalg.det(R[np.ix_(o, o)]) / np.linalg.det(R)
        rows.append({"term": f"[{b} block]", "df": len(cs), "VIF": np.nan, "GVIF": gv,
                     "GVIF^(1/2df)": gv ** (1 / (2 * len(cs)))})
    return pd.DataFrame(rows).set_index("term").sort_values("GVIF^(1/2df)", ascending=False)


def belsley(X):
    Z = sm.add_constant(X).values
    Z = Z / np.linalg.norm(Z, axis=0)
    _, s, Vt = np.linalg.svd(Z, full_matrices=False)
    ci = s.max() / s
    phi = (Vt.T ** 2) / s ** 2
    pi = phi / phi.sum(axis=1, keepdims=True)  # rows: variables, cols: components
    names = ["const"] + list(X.columns)
    out = []
    for k in np.argsort(-ci):
        if ci[k] < 15:
            break
        involved = [(names[j], pi[j, k]) for j in np.argsort(-pi[:, k]) if pi[j, k] > 0.3]
        out.append({"condition index": ci[k],
                    "variables with variance proportion > 0.3": ", ".join(f"{n} ({p:.2f})" for n, p in involved),
                    "harmful (CI>30 and >=2 props >0.5)": ci[k] > 30 and sum(p > 0.5 for _, p in involved) >= 2})
    return ci.max(), pd.DataFrame(out)


def sign_flips(X, blocks):
    fit = sm.OLS(df["y"].values, sm.add_constant(X)).fit(
        cov_type="cluster", cov_kwds={"groups": df["Sire_key"].astype("category").cat.codes.values})
    in_block = {c for cs in blocks.values() for c in cs}
    rows = []
    for c in X.columns:
        if c in in_block:
            continue
        r = np.corrcoef(X[c], df["y"])[0, 1]
        b, p = fit.params[c], fit.pvalues[c]
        rows.append({"term": c, "bivariate r": r, "OLS coef": b, "p": p,
                     "sign flip (p<0.05)": bool(np.sign(r) != np.sign(b) and p < 0.05 and abs(r) > 0.02)})
    return pd.DataFrame(rows).set_index("term")


out = ["# Step 7: collinearity diagnostics\n",
       "VIF > 5 is commonly read as moderate and > 10 as serious; for factor blocks compare "
       "GVIF^(1/2df) with sqrt(5) = 2.24 and sqrt(10) = 3.16 (the raw GVIF of a many-column block is also shown, "
       "because the 1/(2df) power makes an 11-column block look small). Belsley condition indices > 30 combined with "
       "two or more variance proportions > 0.5 indicate a near-dependency that inflates those coefficients' "
       "variances. Continuous columns standardized; designs include an intercept.\n"]

# Exact functional redundancy in the dosage block
d = df.dropna(subset=FIVE)
B, I, C, S, P = [d[c] for c in FIVE]
den = C / 2 + S + P
DI = np.where(den > 0, (B + I + C / 2) / den.replace(0, np.nan), np.nan)
m = np.isfinite(DI) & d["Dosage_Index"].notna()
CD = (2 * B + I - S - 2 * P) / (B + I + C + S + P)
out.append("## Dosage block: derived summaries are functions of the five points\n")
out.append(f"- Dosage Index recomputed as (B+I+C/2)/(C/2+S+P) matches the column with correlation "
           f"{np.corrcoef(DI[m], d.loc[m, 'Dosage_Index'])[0, 1]:.6f} (max abs difference "
           f"{np.max(np.abs(DI[m] - d.loc[m, 'Dosage_Index'])):.3f}, i.e. rounding).")
out.append(f"- Center of Distribution recomputed as (2B+I-S-2P)/(B+I+C+S+P) matches with correlation "
           f"{np.corrcoef(CD[d['Center_of_Distribution'].notna() & np.isfinite(CD)], d['Center_of_Distribution'].dropna()[np.isfinite(CD[d['Center_of_Distribution'].notna()])])[0, 1]:.4f}.")
out.append(f"- Dosage Index is undefined (division by zero) for {(den == 0).sum()} horses and missing for "
           f"{df['Dosage_Index'].isna().sum()} overall, versus {df['Brilliant'].isna().sum()} with no dosage profile.\n")

summary = []
for key, (X, blocks) in DESIGNS.items():
    vt = vif_table(X, blocks)
    cn, bk = belsley(X)
    sf = sign_flips(X, blocks)
    vt.to_csv(f"{RESULTS}/07_vif_{key}.csv")
    out.append(f"## {key}: {LABELS[key]} ({X.shape[1]} columns)\n")
    out.append(f"Condition number (equilibrated design): **{cn:.1f}**\n")
    out.append("### VIF / GVIF\n")
    out.append(fmt_table(vt, floatfmt=".2f") + "\n")
    out.append("### Condition indices >= 15 (Belsley-Kuh-Welsch)\n")
    out.append((fmt_table(bk.set_index("condition index"), floatfmt=".2f") if len(bk) else "None.") + "\n")
    flips = sf[sf["sign flip (p<0.05)"]]
    out.append("### Significant sign flips (multivariate coefficient opposite to bivariate correlation)\n")
    out.append((fmt_table(flips, floatfmt=".3f") if len(flips) else "None.") + "\n")
    summary.append({"Design": f"{key}: {LABELS[key]}", "columns": X.shape[1],
                    "max VIF (single cols)": vt["VIF"].max(),
                    "terms with VIF > 5": int((vt["VIF"] > 5).sum() + (vt["GVIF^(1/2df)"][vt["VIF"].isna()] > np.sqrt(5)).sum()),
                    "condition number": cn,
                    "harmful near-dependencies": int(bk["harmful (CI>30 and >=2 props >0.5)"].sum()) if len(bk) else 0,
                    "sign flips": len(flips)})
# ---------------------------------------------------------------------------
# Suppression / confounding checks behind the sign flips
# ---------------------------------------------------------------------------
groups = df["Sire_key"].astype("category").cat.codes.values


def coef(cols, factors, term):
    X, _ = design([c for c in cols if df[c].nunique() > 2], [c for c in cols if df[c].nunique() <= 2], factors)
    f = sm.OLS(df["y"].values, sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": groups})
    return f.params[term], f.pvalues[term]


Pc = ["Sire_Rep_full", "Dam_Rep_full"] + PAPER_NUMERIC + ["First_crop_paper"]
rows = []
for label, drop in [("Published design, all controls", []),
                    ("  without log_Sire_starters", ["log_Sire_starters"]),
                    ("  without log_Sire_starters, Sire_BW_rate", ["log_Sire_starters", "Sire_BW_rate"]),
                    ("  without those and Sire Reputation", ["log_Sire_starters", "Sire_BW_rate", "Sire_Rep_full"]),
                    ("Year FE only", [c for c in Pc if c != "First_crop_paper"])]:
    b, p = coef([c for c in Pc if c not in drop], ["Year"], "First_crop_paper")
    rows.append({"Specification": label, "First-crop coef": b, "p": p})
fc = pd.DataFrame(rows).set_index("Specification")

S2c = ["Sire_Reputation", "Dam_Reputation", "Consignor_Reputation"] + FIVE + ["Is_male", "Sire_new_to_sale",
                                                                                "Dosage_missing"]
rows = []
for label, cols in [("Revised S2 (all controls)", S2c),
                    ("  without Sire Reputation", [c for c in S2c if c != "Sire_Reputation"])]:
    b, p = coef(cols, ["Year", "Session", "Color"], "Sire_new_to_sale")
    rows.append({"Specification": label, "Sire-new-to-sale coef": b, "p": p})
w = df[df["Year"] > 2020].copy()
w["r"] = w["y"] - w.groupby(["Year", "Session"])["y"].transform("mean")
raw_gap = w.groupby("Sire_new_to_sale")["r"].mean().diff().iloc[-1]
nts = pd.DataFrame(rows).set_index("Specification")

r2_sr_session = sm.OLS(df["Sire_Reputation"], sm.add_constant(
    pd.get_dummies(df["Session"], drop_first=True, dtype=float))).fit().rsquared
b_no, _ = coef(S2c, ["Year", "Color"], "Sire_Reputation")
b_yes, _ = coef(S2c, ["Year", "Session", "Color"], "Sire_Reputation")

out.append("## Sign flips are suppression effects, not premiums\n")
out.append("**Published first-crop flag.** Within each year, first-crop yearlings sell for *less* than others; the "
           "coefficient turns positive only after conditioning on sire reputation and sire progeny statistics, "
           "which are mechanically low for young sires (few raced progeny, few prior sales). The positive "
           "coefficient therefore corrects the other sire variables for new sires; it is not evidence of a "
           "speculative premium.\n")
out.append(fmt_table(fc, floatfmt=".3f") + "\n")
out.append("**Revised sire-new-to-sale flag.** Same mechanism: positive only conditional on sire reputation "
           f"(new sires' reputation is built from few sales). The raw gap within year x session is {raw_gap:+.3f} "
           "log points.\n")
out.append(fmt_table(nts, floatfmt=".3f") + "\n")
out.append("## Session and sire reputation: confounding, not variance inflation\n")
out.append(f"Session fixed effects explain R2 = {r2_sr_session:.2f} of the variation in Sire Reputation "
           f"(corr. with session number {np.corrcoef(df['Sire_Reputation'], df['Session'])[0, 1]:.2f}). "
           f"That is well below VIF-type thresholds (VIF ~ {1 / (1 - r2_sr_session):.2f}), so the Sire Reputation "
           f"coefficient is estimated precisely in both specifications, but it moves from {b_no:.3f} to {b_yes:.3f} "
           "when session FE are added because better-bred yearlings are placed in earlier sessions. This is a "
           "change in what the coefficient means (total vs within-session association), not an instability.\n")

summ = pd.DataFrame(summary).set_index("Design")
summ.to_csv(f"{RESULTS}/07_collinearity_summary.csv")
out.insert(2, "## Summary\n\n" + fmt_table(summ, floatfmt=".2f") + "\n")
with open(f"{RESULTS}/07_collinearity.md", "w") as f:
    f.write("\n".join(out) + "\n")
print("\n".join(out))
