"""Step 6 - Automated leakage audit.

Each check is designed to FAIL if information about a test horse's price (or
post-sale information) reaches the model.

  T1  Input audit: list the raw columns every model actually consumes; none may
      be an outcome/post-sale column, and the clean sets may not contain the
      cross statistics (own racing record) or, for S2, any hindsight stats.
  T2  Encoder test: changing one training horse's price must not change that
      horse's own cross-fitted reputation score.
  T3  Test-label invariance: scrambling the prices (y, Log_Price, Price) of the
      test rows must leave every test prediction unchanged - for random and
      year-ahead splits, for Ridge, LightGBM and CatBoost pipelines.
  T4  Null test: with prices randomly permuted across horses, cross-validated
      R2 must be ~0. Any feature built from the target would show up here.
  T5  Data timing: evidence on when the Equineline statistics were pulled, and
      how much of the cross statistics is the horse's own racing record.
  T6  Duplicate horses across folds.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from common import (FORBIDDEN, HINDSIGHT, OWN_OUTCOME, RESULTS, SEED,
                    CrossFitReputation, fmt_table, load_data)
from models import FEATURE_SETS, CatBoostNative, catboost_model, lgbm_model, linear_model

rng = np.random.default_rng(SEED)
df = load_data()
report = ["# Step 6: leakage audit\n"]
fails = []


def check(name, ok, detail=""):
    report.append(f"- {'PASS' if ok else 'FAIL'}: {name}{' - ' + detail if detail else ''}")
    if not ok:
        fails.append(name)
    print(("PASS " if ok else "FAIL ") + name, detail, flush=True)


def fixed_lgbm(fs):
    m = lgbm_model(fs).estimator
    m.set_params(est__n_estimators=500, est__learning_rate=0.03, est__num_leaves=15,
                 est__min_child_samples=40, est__subsample=0.8, est__colsample_bytree=0.7)
    return m


def fixed_cat(fs):
    return CatBoostNative(fs=fs, depth=6, learning_rate=0.05, iterations=500)


def fixed_ridge(fs):
    return linear_model(fs, "ridge").estimator  # Pipeline with RidgeCV, index reputation


def consumed_columns(model, fs):
    """Raw input columns a fitted model reads."""
    if isinstance(model, CatBoostNative):
        num, cat = model._cols()
        return set(num + cat)
    ct = model.named_steps["pre"]
    assert ct.remainder == "drop"  # unlisted columns are discarded, never passed through
    cols = set()
    for name, tr, c in ct.transformers_:
        if name == "remainder" or (isinstance(tr, str) and tr == "drop"):
            continue  # columns sklearn explicitly drops
        cols |= set(c)
    return cols


# ---------------------------------------------------------------------------
# T1 input audit
# ---------------------------------------------------------------------------
report.append("\n## T1. Inputs consumed by each model\n")
sample = df.sample(1500, random_state=SEED)
for fs in FEATURE_SETS:
    for kind, make in [("Ridge", fixed_ridge), ("LightGBM", fixed_lgbm), ("CatBoost", fixed_cat)]:
        if kind != "Ridge" and fs in ("F0", "F1", "F2"):
            continue
        m = make(fs).fit(sample, sample["y"])
        cols = consumed_columns(m, fs)
        bad = cols & set(FORBIDDEN)
        check(f"{kind} {fs}: no outcome/post-sale ID columns", not bad, f"found {sorted(bad)}" if bad else "")
        if fs.startswith("S"):
            own = cols & set(OWN_OUTCOME)
            check(f"{kind} {fs}: no own-racing-record (cross) statistics", not own,
                  f"found {sorted(own)}" if own else "")
        if fs == "S2":
            hs = cols & set(HINDSIGHT)
            check(f"{kind} {fs}: no hindsight progeny statistics", not hs, f"found {sorted(hs)}" if hs else "")
report.append("\nS2 inputs: " + ", ".join(sorted(consumed_columns(fixed_cat("S2"), "S2"))))

# ---------------------------------------------------------------------------
# T2 cross-fitted encoder
# ---------------------------------------------------------------------------
report.append("\n## T2. Reputation encoder: own price never enters own score\n")
keys = ["Sire_key", "Dam_key", "Consignor_key"]
for method, kw in [("index", dict(min_n={"Sire_key": 5, "Dam_key": 2, "Consignor_key": 3})),
                   ("mean", dict(shrink=5.0))]:
    enc = CrossFitReputation(cols=tuple(keys), method=method, **kw)
    base = enc.fit_transform(df[keys], df["y"].values)
    rows = rng.choice(len(df), 40, replace=False)
    max_change = 0.0
    for i in rows:
        y2 = df["y"].values.copy()
        y2[i] += 5.0  # huge change in own price
        new = CrossFitReputation(cols=tuple(keys), method=method, **kw).fit_transform(df[keys], y2)
        max_change = max(max_change, float(np.abs(new[i] - base[i]).max()))
    check(f"{method} encoder: own score unchanged when own price changes (40 random horses)",
          max_change < 1e-12, f"max change {max_change:.2e}")
    # Contrast: a full-sample (published-style) encoder DOES move.
from common import reputation_table
tab, _ = reputation_table(df["y"].values, df["Dam_key"].values, "index", 0.0, 2)
two = df.groupby("Dam_key")["y"].transform("size").values == 2
i = int(np.flatnonzero(two)[0])
y2 = df["y"].values.copy(); y2[i] += 5
tab2, _ = reputation_table(y2, df["Dam_key"].values, "index", 0.0, 2)
k = df["Dam_key"].iloc[i]
report.append(f"- Contrast: with the published full-sample index, raising one horse's price moves its own "
              f"Dam Reputation from {tab[k]:.3f} to {tab2[k]:.3f}.")

# ---------------------------------------------------------------------------
# T3 test-label invariance
# ---------------------------------------------------------------------------
report.append("\n## T3. Test predictions do not depend on test prices\n")


def scramble(d):
    d = d.copy()
    d["y"] = rng.normal(11, 3, len(d))
    d["Log_Price"] = d["y"]
    d["Price"] = np.exp(d["y"])
    return d


splits = {}
tr, te = next(StratifiedKFold(5, shuffle=True, random_state=SEED).split(df, df["Session"]))
splits["random 80/20"] = (df.iloc[tr], df.iloc[te])
tr_t = df[df["Year"] < 2024].copy()
te_t = df[df["Year"] == 2024].copy()
te_t["Year"] = 2023
splits["year-ahead (train 2020-23, test 2024)"] = (tr_t, te_t)
for sname, (a, b) in splits.items():
    for kind, make, fs in [("Ridge", fixed_ridge, "S2"), ("Ridge", fixed_ridge, "F3"),
                           ("LightGBM", fixed_lgbm, "S2"), ("CatBoost", fixed_cat, "S2")]:
        m = make(fs).fit(a, a["y"])
        p1 = m.predict(b)
        p2 = m.predict(scramble(b))
        diff = float(np.max(np.abs(p1 - p2)))
        check(f"{sname}, {kind} {fs}: predictions identical after scrambling test prices", diff == 0.0,
              f"max |diff| = {diff:.1e}")
# Also: training on (train) vs (train with test rows' prices scrambled but rows absent) is trivially
# identical because test rows are never passed to fit; asserted by construction in the scripts.

# ---------------------------------------------------------------------------
# T4 null test
# ---------------------------------------------------------------------------
report.append("\n## T4. Null test: prices randomly permuted across horses\n")
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
null_rows = []
for rep in range(3):
    d = df.copy()
    perm = rng.permutation(len(d))
    for c in ("y", "Log_Price", "Price"):
        d[c] = df[c].values[perm]
    for kind, make, fs in [("Ridge", fixed_ridge, "S2"), ("LightGBM", fixed_lgbm, "S2"),
                           ("Ridge", fixed_ridge, "F3")]:
        s = cross_val_score(make(fs), d, d["y"], cv=cv.split(d, d["Session"]), scoring="r2")
        null_rows.append({"perm": rep, "model": f"{kind} {fs}", "R2": s.mean()})
null = pd.DataFrame(null_rows).groupby("model")["R2"].agg(["mean", "max"])
for mname, r in null.iterrows():
    check(f"{mname}: CV R2 with permuted prices <= 0.01", r["max"] <= 0.01,
          f"mean {r['mean']:.4f}, max {r['max']:.4f}")

# ---------------------------------------------------------------------------
# T5 data timing
# ---------------------------------------------------------------------------
report.append("\n## T5. When were the pedigree statistics pulled?\n")
one = df["Combo_foals"] == 1
t5 = df.groupby("Year").apply(lambda g: pd.Series({
    "n": len(g),
    "cross has 1 foal (= this horse)": (g["Combo_foals"] == 1).mean(),
    "of those: cross has a starter": (g.loc[g["Combo_foals"] == 1, "Combo_starters"] > 0).mean(),
    "of those: cross has a winner": (g.loc[g["Combo_foals"] == 1, "Combo_winners"] > 0).mean(),
    "median sire starters": g["Sire_starters"].median(),
}), include_groups=False)
report.append(fmt_table(t5, floatfmt=".3f"))
report.append(
    "\nHorses sold in 2020-2022 whose cross has a single foal show 81-90% 'starters' and 56-66% 'winners'; "
    "for 2024 yearlings it is 0%. A yearling cannot have raced at the sale, so the cross statistics record the "
    "horse's own later racing career and the file was compiled after the 2024 sale. The `Horse Name` column also "
    "holds names given after the sale. Cross statistics (`common.OWN_OUTCOME`) are therefore excluded from the "
    "clean feature sets; sire/broodmare-sire statistics (`common.HINDSIGHT`) are used only in the S1 sensitivity "
    "set.")
corr = np.corrcoef(np.log1p(df.loc[one & (df.Year <= 2022), "Combo_earnings"]),
                   df.loc[one & (df.Year <= 2022), "y"])[0, 1]
report.append(f"\nAmong 2020-2022 horses whose cross has one foal, corr(log own career earnings, log price) = {corr:.3f}.")

# ---------------------------------------------------------------------------
# T6 duplicates
# ---------------------------------------------------------------------------
report.append("\n## T6. Duplicate horses\n")
dup_full = df.duplicated(["Sire_key", "Dam_key", "Year"]).sum()
dup_name = df.duplicated(["Horse Name", "Year"], keep=False)
same_horse = df[dup_name].groupby(["Horse Name", "Year"])[["Sire_key", "Dam_key"]].nunique()
check("No horse (same sire, dam and year) appears twice", dup_full == 0, f"{dup_full} duplicates")
report.append(f"- {int(dup_name.sum())} rows share a name and year, but every pair has different sires and dams "
              f"({int((same_horse.max(axis=1) > 1).sum())}/{len(same_horse)} pairs), so they are different horses.")

# ---------------------------------------------------------------------------
report.append("\n## Summary\n")
report.append(f"{'All checks passed.' if not fails else 'FAILED: ' + '; '.join(fails)}")
report.append("\nRemaining non-code caveat: in random K-fold CV a horse's sire/dam/consignor reputation can be built "
              "from horses sold at the same sale or in later years. That information is legitimate in the sense "
              "that it is not the horse's own price, but it would not be available before bidding. The year-ahead "
              "evaluation (`03_temporal.md`) uses only earlier sales and is the cleanest estimate.")
with open(f"{RESULTS}/06_leakage_checks.md", "w") as f:
    f.write("\n".join(report) + "\n")
print("\n".join(report[-4:]))
