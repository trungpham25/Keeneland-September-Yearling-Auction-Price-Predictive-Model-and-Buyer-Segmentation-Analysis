"""Shared data preparation, leak-free reputation encoders and evaluation helpers
for the revised machine-learning analysis.

Design rules that every script follows:
  1. Nothing that looks at the outcome (log price) is computed on rows that are
     later used to evaluate the model. That covers reputation indices, outlier
     screening, feature selection, power transforms and hyperparameter tuning.
  2. Reputation indices for a training row are cross-fitted: they are computed
     from the *other* folds of the training data, so a horse's own price never
     enters its own predictor. Rows being predicted get indices computed from the
     full training data only.
  3. Only information available in the sale catalog before bidding is used
     (session, consignor, pedigree, sex, color). The purchaser is excluded.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "revision", "results")
os.makedirs(RESULTS, exist_ok=True)

SEED = 20260626


# ---------------------------------------------------------------------------
# Data loading and outcome-free cleaning
# ---------------------------------------------------------------------------
def _clean_name(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower().str.strip()
    s = s.str.replace(r"\s*\([a-z]{2,3}\)\s*$", "", regex=True)  # country suffix
    s = s.str.replace(r"\s+", " ", regex=True)
    return s


def load_data() -> pd.DataFrame:
    """Load sold.csv and build outcome-free features.

    No rows are dropped on the basis of price. The only exclusions are rows
    that cannot be used at all (none at present).
    """
    df = pd.read_csv(os.path.join(ROOT, "sold.csv"))
    df = df.reset_index(drop=True)

    df["Sire_key"] = _clean_name(df["Sire"])
    df["Dam_key"] = _clean_name(df["Dam"])
    # Consigning agency = text before the first comma ("Taylor Made Sales
    # Agency, Agent CII" -> "taylor made sales agency"). Printed in the catalog.
    df["Consignor_key"] = _clean_name(df["PropertyLine1"].str.split(",").str[0])

    df["Color"] = df["Color"].astype(str).str.strip()
    df["Is_male"] = df["Sex"].isin(["Colt", "Gelding"]).astype(int)
    df["Book"] = pd.cut(df["Session"], [0, 2, 4, 6, 8, 10, 12],
                        labels=["S1-2", "S3-4", "S5-6", "S7-8", "S9-10", "S11-12"])

    # Progeny counts / earnings: log1p is a fixed (outcome-free) transform.
    for c in ["Sire_starters", "BS_foals", "BS_starters", "Combo_foals",
              "Combo_starters", "Combo_earnings", "Sire_earnings", "BS_earnings"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        df["log_" + c] = np.log1p(df[c])

    eps = 1e-9
    df["Sire_BW_rate"] = np.where(df["Sire_starters"] > 0, df["Sire_BW"] / (df["Sire_starters"] + eps), 0.0)
    df["BS_BW_rate"] = np.where(df["BS_starters"] > 0, df["BS_BW"] / (df["BS_starters"] + eps), 0.0)
    df["Sire_win_rate"] = np.where(df["Sire_starters"] > 0, df["Sire_winners"] / (df["Sire_starters"] + eps), 0.0)
    df["BS_win_rate"] = np.where(df["BS_starters"] > 0, df["BS_winners"] / (df["BS_starters"] + eps), 0.0)
    df["log_Sire_earn_per_starter"] = np.log1p(np.where(df["Sire_starters"] > 0, df["Sire_earnings"] / (df["Sire_starters"] + eps), 0.0))
    df["log_BS_earn_per_starter"] = np.log1p(np.where(df["BS_starters"] > 0, df["BS_earnings"] / (df["BS_starters"] + eps), 0.0))
    df["Sire_unraced_progeny"] = (df["Sire_starters"] == 0).astype(int)

    df["Dosage_missing"] = df["Brilliant"].isna().astype(int)

    # --- First-crop status -------------------------------------------------
    # Paper definition: Year == first year the sire appears in the 2020-2024
    # data. This is left-censored: every 2020 row is flagged (100% of 2020 vs
    # 16-29% in later years), so it is confounded with the 2020 baseline.
    first_year = df.groupby("Sire_key")["Year"].transform("min")
    df["First_crop_paper"] = (df["Year"] == first_year).astype(int)
    # Revised proxy: sire had no yearlings in any *earlier* sale year in the
    # sample. Undefined for 2020 (no history), so set to 0 there; the 2020 year
    # effect absorbs the level. Uses past information only.
    df["Sire_new_to_sale"] = ((df["Year"] == first_year) & (df["Year"] > 2020)).astype(int)

    df["y"] = df["Log_Price"].astype(float)
    return df


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
# The 20 predictors in the published Ridge model (Figure 2), mapped onto the
# columns available in sold.csv. "Sire foals" is not in the file; the sire's
# number of starters is the closest available count.
PAPER_NUMERIC = [
    "log_Combo_starters", "log_Combo_earnings", "Is_male", "Sire_BW_rate",
    "log_Sire_starters", "Intermediate", "log_BS_foals", "Solid", "Classic",
    "Brilliant", "Combo_aei", "BS_BW_rate", "Professional",
]
YEAR = ["Year"]

EXTRA_PEDIGREE = [
    "Sire_aei", "BS_aei", "Combo_BW", "Sire_win_rate", "BS_win_rate",
    "log_Sire_earn_per_starter", "log_BS_earn_per_starter",
    "Dosage_Index", "Center_of_Distribution", "Dosage_missing",
    "Sire_unraced_progeny", "log_Combo_foals",
]


# ---------------------------------------------------------------------------
# Reputation indices
# ---------------------------------------------------------------------------
def _index_components(y: np.ndarray, keys: np.ndarray, elite_cut: float) -> pd.DataFrame:
    g = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"]
    comp = pd.DataFrame({
        "n": g.size(),
        "p90": g.quantile(0.90),
        "elite": g.apply(lambda v: float(np.mean(v > elite_cut))),
        "med": g.median(),
        "p25": g.quantile(0.25),
    })
    return comp


def reputation_table(y, keys, method="index", shrink=0.0, min_n=1):
    """Return a lookup table key -> reputation score, plus the default score
    used for unseen keys.

    method="index": the paper's 4-component index (Section 3.2): 0.40*p90 +
    0.30*share above the market 90th percentile + 0.20*percentile rank of the
    median + 0.10*p25, each min-max scaled across groups. As in the original
    code, groups with fewer than min_n sales get default components (market
    p90/p25, zero elite share, median rank 0.5); unseen keys get the same
    default score.
    method="mean": empirical-Bayes shrunken mean log price.
    shrink: pseudo-count m; scores are pulled toward the default by m/(n+m).
    """
    y = np.asarray(y, float)
    keys = np.asarray(keys)
    if method == "mean":
        gm = y.mean()
        g = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"].agg(["size", "mean"])
        score = (g["size"] * g["mean"] + shrink * gm) / (g["size"] + shrink)
        score[g["size"] < min_n] = gm
        return score.to_dict(), gm

    elite_cut = np.quantile(y, 0.90)
    comp = _index_components(y, keys, elite_cut)
    comp["rank"] = comp["med"].rank(pct=True)
    default = {"p90": np.quantile(y, 0.90), "p25": np.quantile(y, 0.25), "elite": 0.0, "rank": 0.5}
    low = comp["n"] < min_n
    for k, v in default.items():
        comp.loc[low, k] = v
    # Append a pseudo-row for the default so it is scaled consistently.
    comp.loc["__default__", ["p90", "p25", "elite", "rank", "n"]] = [
        default["p90"], default["p25"], 0.0, 0.5, 0]

    def mm(s):
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo) if hi > lo else s * 0 + 0.5

    idx = (0.40 * mm(comp["p90"]) + 0.30 * mm(comp["elite"])
           + 0.20 * comp["rank"] + 0.10 * mm(comp["p25"]))
    prior = float(idx.pop("__default__"))
    n = comp["n"].drop("__default__")
    if shrink > 0:
        idx = (n * idx + shrink * prior) / (n + shrink)
    return idx.to_dict(), prior


class CrossFitReputation(BaseEstimator, TransformerMixin):
    """Leak-free reputation encoder for one or more grouping columns.

    fit_transform (called on training data inside a Pipeline) returns
    cross-fitted scores: each training row's score comes from the other
    n_splits-1 folds. transform (called on validation/test data) uses the
    table built from all training rows. A count feature log(1+n) of prior
    observations is also returned for each key when add_count=True.
    """

    def __init__(self, cols=("Sire_key",), method="index", shrink=0.0,
                 min_n=None, n_splits=5, add_count=False, random_state=SEED):
        self.cols = cols
        self.min_n = min_n
        self.method = method
        self.shrink = shrink
        self.n_splits = n_splits
        self.add_count = add_count
        self.random_state = random_state

    def _shrink_for(self, col):
        if isinstance(self.shrink, dict):
            return self.shrink.get(col, 0.0)
        return self.shrink

    def _min_n_for(self, col):
        if isinstance(self.min_n, dict):
            return self.min_n.get(col, 1)
        return self.min_n or 1

    def fit(self, X, y):
        X = pd.DataFrame(X)
        self.tables_ = {}
        for c in self.cols:
            tab, prior = reputation_table(y, X[c].values, self.method, self._shrink_for(c), self._min_n_for(c))
            counts = X[c].value_counts().to_dict()
            self.tables_[c] = (tab, prior, counts)
        return self

    def _lookup(self, keys, tab, prior, counts):
        s = pd.Series(keys).map(tab).fillna(prior).values
        n = pd.Series(keys).map(counts).fillna(0).values
        return s, np.log1p(n)

    def transform(self, X):
        X = pd.DataFrame(X)
        out = []
        for c in self.cols:
            s, n = self._lookup(X[c].values, *self.tables_[c])
            out.append(s)
            if self.add_count:
                out.append(n)
        return np.column_stack(out)

    def fit_transform(self, X, y=None, **kw):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y, float)
        self.fit(X, y)
        res = np.zeros((len(X), len(self.cols) * (2 if self.add_count else 1)))
        kf = KFold(self.n_splits, shuffle=True, random_state=self.random_state)
        for tr, te in kf.split(X):
            j = 0
            for c in self.cols:
                tab, prior = reputation_table(y[tr], X[c].values[tr], self.method, self._shrink_for(c), self._min_n_for(c))
                counts = X[c].iloc[tr].value_counts().to_dict()
                s, n = self._lookup(X[c].values[te], tab, prior, counts)
                res[te, j] = s
                j += 1
                if self.add_count:
                    res[te, j] = n
                    j += 1
        return res

    def get_feature_names_out(self, input_features=None):
        names = []
        for c in self.cols:
            base = c.replace("_key", "")
            names.append(f"{base}_Reputation")
            if self.add_count:
                names.append(f"{base}_log_n_prior")
        return np.array(names)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(y_true, y_pred, p=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    r2 = r2_score(y_true, y_pred)
    out = {
        "R2": r2,
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
        # Median absolute percentage error in dollars
        "MdAPE_$": float(np.median(np.abs(np.exp(y_pred - y_true) - 1)) * 100),
    }
    if p is not None:
        n = len(y_true)
        out["Adj_R2"] = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return out


def fmt_table(df: pd.DataFrame, floatfmt=".4f") -> str:
    """Markdown table without requiring the tabulate package."""
    cols = [str(c) for c in df.columns]
    idx_name = df.index.name or ""
    lines = ["| " + " | ".join([idx_name] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for i, row in df.iterrows():
        cells = []
        for v in row.values:
            if isinstance(v, (float, np.floating)):
                cells.append(format(v, floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join([str(i)] + cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Preprocessing builder
# ---------------------------------------------------------------------------
def build_preprocessor(numeric, onehot=(), rep_cols=(), rep_method="index",
                       rep_shrink=0.0, rep_min_n=None, rep_count=False, scale=True,
                       passthrough=()):
    """ColumnTransformer: imputation/scaling for numeric columns, one-hot for
    categoricals, cross-fitted reputation for grouping keys. Everything is
    fitted on the training fold only because it lives inside the Pipeline."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("sc", StandardScaler()))
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline(steps), list(numeric)))
    if onehot:
        transformers.append(("oh", OneHotEncoder(drop="first", handle_unknown="ignore",
                                                 sparse_output=False), list(onehot)))
    if rep_cols:
        rep = CrossFitReputation(cols=tuple(rep_cols), method=rep_method,
                                 shrink=rep_shrink, min_n=rep_min_n, add_count=rep_count)
        if scale:
            rep = Pipeline([("rep", rep), ("sc", StandardScaler())])
        transformers.append(("rep", rep, list(rep_cols)))
    if passthrough:
        transformers.append(("pt", "passthrough", list(passthrough)))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


RIDGE_ALPHAS = np.logspace(-5, 3, 60)
