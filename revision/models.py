"""Model factories for the revised analysis. Every model is a full sklearn
Pipeline (preprocessing + estimator) so that all outcome-dependent steps are
refitted inside each training fold."""
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression, RidgeCV
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from common import (DOSAGE, EXTRA_PEDIGREE, PAPER_NUMERIC, RIDGE_ALPHAS, SEED,
                    SIRE_BS_STATS, build_preprocessor)

KEYS3 = ["Sire_key", "Dam_key", "Consignor_key"]
CATALOG = ["Is_male", "Sire_new_to_sale"] + DOSAGE

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
# F0-F3 keep the published predictors, which include the sire x broodmare-sire
# cross statistics. Those contain the horse's own post-sale racing record, so
# F0-F3 are reported only to trace the path from the published model.
# S1/S2 are the clean sets used for the revised results:
#   S2 (main): catalog information only - session, year, sex, color, dosage,
#       sire/dam/consignor reputation from *other* horses' prices.
#   S1 (sensitivity): S2 + sire and broodmare-sire progeny statistics, which
#       were pulled after the sales (hindsight, but not the horse's own record).
FEATURE_SETS = {
    "F0": dict(numeric=PAPER_NUMERIC + ["Sire_new_to_sale"], onehot=["Year"],
               rep_cols=["Sire_key", "Dam_key"]),
    "F1": dict(numeric=PAPER_NUMERIC + ["Sire_new_to_sale"], onehot=["Year", "Session"],
               rep_cols=["Sire_key", "Dam_key"]),
    "F2": dict(numeric=PAPER_NUMERIC + ["Sire_new_to_sale"], onehot=["Year", "Session"],
               rep_cols=KEYS3),
    "F3": dict(numeric=PAPER_NUMERIC + ["Sire_new_to_sale"] + EXTRA_PEDIGREE,
               onehot=["Year", "Session", "Color"], rep_cols=KEYS3, rep_count=True),
    "S1": dict(numeric=CATALOG + SIRE_BS_STATS, onehot=["Year", "Session", "Color"],
               rep_cols=KEYS3, rep_count=True),
    "S2": dict(numeric=CATALOG, onehot=["Year", "Session", "Color"],
               rep_cols=KEYS3, rep_count=True),
    # S2 with within-session reputation (built from prices relative to their
    # year x session mean), which removes the mechanical session overlap.
    "S2w": dict(numeric=CATALOG, onehot=["Year", "Session", "Color"],
                rep_cols=KEYS3, rep_count=True, rep_adjust=["Year", "Session"]),
}
FEATURE_SET_LABELS = {
    "F0": "Published 20 predictors, reputation leak removed (still contains own-record cross stats)",
    "F1": "F0 + sale session",
    "F2": "F1 + consignor reputation",
    "F3": "F2 + extra pedigree stats, color, prior-sales counts",
    "S1": "Clean + hindsight: S2 + sire/broodmare-sire progeny stats (pulled after the sales)",
    "S2": "Clean, catalog-only: session, year, sex, color, dosage, sire/dam/consignor reputation",
    "S2w": "S2 with within-session reputation (prices relative to year x session mean)",
}
MIN_N = {"Sire_key": 5, "Dam_key": 2, "Consignor_key": 3}


def _rep_grid():
    # Reputation encoder variants chosen by inner CV.
    return [
        {"pre__rep__rep__method": ["index"], "pre__rep__rep__min_n": [MIN_N], "pre__rep__rep__shrink": [0.0]},
        {"pre__rep__rep__method": ["mean"], "pre__rep__rep__min_n": [None], "pre__rep__rep__shrink": [2.0, 10.0]},
    ]


def linear_model(fs, kind="ridge", inner_cv=5):
    spec = dict(FEATURE_SETS[fs])
    pre = build_preprocessor(spec["numeric"], spec["onehot"], spec["rep_cols"],
                             rep_method="index", rep_min_n=MIN_N,
                             rep_count=spec.get("rep_count", False),
                             rep_adjust=spec.get("rep_adjust"), scale=True)
    if kind == "ridge":
        est = RidgeCV(alphas=RIDGE_ALPHAS)
    elif kind == "ols":
        est = LinearRegression()
    elif kind == "huber":
        est = HuberRegressor(max_iter=2000, alpha=1e-4)
    pipe = Pipeline([("pre", pre), ("est", est)])
    return GridSearchCV(pipe, _rep_grid(), cv=KFold(inner_cv, shuffle=True, random_state=SEED),
                        scoring="r2", n_jobs=4, refit=True)


def tree_preprocessor(fs="F3"):
    spec = FEATURE_SETS[fs]
    # Year/Session enter trees as ordinal numbers, and only if the set has them.
    numeric = list(spec["numeric"]) + [c for c in ("Year", "Session") if c in spec["onehot"]]
    onehot = [c for c in spec["onehot"] if c not in ("Year", "Session")]
    return build_preprocessor(numeric, onehot, spec["rep_cols"], rep_method="mean",
                              rep_shrink=5.0, rep_count=True, rep_adjust=spec.get("rep_adjust"),
                              scale=False)


def lgbm_model(fs="F3", n_iter=25, inner_cv=4):
    import lightgbm as lgb
    pipe = Pipeline([("pre", tree_preprocessor(fs)),
                     ("est", lgb.LGBMRegressor(n_jobs=1, verbose=-1, random_state=SEED,
                                               subsample_freq=1))])
    space = {
        "est__n_estimators": randint(200, 1500),
        "est__learning_rate": loguniform(0.005, 0.1),
        "est__num_leaves": randint(4, 48),
        "est__min_child_samples": randint(10, 120),
        "est__subsample": uniform(0.5, 0.5),
        "est__colsample_bytree": uniform(0.3, 0.7),
        "est__reg_lambda": loguniform(1e-3, 30),
        "est__reg_alpha": loguniform(1e-3, 10),
        "pre__rep__shrink": [1.0, 5.0, 20.0],
    }
    return RandomizedSearchCV(pipe, space, n_iter=n_iter, cv=KFold(inner_cv, shuffle=True, random_state=SEED),
                              scoring="r2", n_jobs=4, random_state=SEED, refit=True)


def xgb_model(fs="F3", n_iter=20, inner_cv=4):
    import xgboost as xgb
    pipe = Pipeline([("pre", tree_preprocessor(fs)),
                     ("est", xgb.XGBRegressor(n_jobs=1, random_state=SEED, tree_method="hist"))])
    space = {
        "est__n_estimators": randint(200, 1200),
        "est__learning_rate": loguniform(0.005, 0.1),
        "est__max_depth": randint(2, 7),
        "est__min_child_weight": loguniform(1, 50),
        "est__subsample": uniform(0.5, 0.5),
        "est__colsample_bytree": uniform(0.3, 0.7),
        "est__reg_lambda": loguniform(1e-2, 100),
        "est__reg_alpha": loguniform(1e-3, 10),
        "est__gamma": loguniform(1e-3, 1),
        "pre__rep__shrink": [1.0, 5.0, 20.0],
    }
    return RandomizedSearchCV(pipe, space, n_iter=n_iter, cv=KFold(inner_cv, shuffle=True, random_state=SEED),
                              scoring="r2", n_jobs=4, random_state=SEED, refit=True)


def rf_model(fs="F3", n_iter=10, inner_cv=4):
    pipe = Pipeline([("pre", tree_preprocessor(fs)),
                     ("est", RandomForestRegressor(n_estimators=400, n_jobs=1, random_state=SEED))])
    space = {
        "est__max_features": uniform(0.15, 0.5),
        "est__min_samples_leaf": randint(2, 25),
        "est__max_depth": [None, 8, 12, 20],
        "pre__rep__shrink": [1.0, 5.0, 20.0],
    }
    return RandomizedSearchCV(pipe, space, n_iter=n_iter, cv=KFold(inner_cv, shuffle=True, random_state=SEED),
                              scoring="r2", n_jobs=4, random_state=SEED, refit=True)


class CatBoostNative(BaseEstimator, RegressorMixin):
    """CatBoost with native ordered target statistics for the sire, dam and
    consignor identifiers (leak-free by construction), plus the numeric
    features. Expects the raw data frame."""

    def __init__(self, fs="F3", depth=6, learning_rate=0.05, iterations=1000,
                 l2_leaf_reg=3.0, random_strength=1.0, random_state=SEED):
        self.fs = fs
        self.depth = depth
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2_leaf_reg = l2_leaf_reg
        self.random_strength = random_strength
        self.random_state = random_state

    def _cols(self):
        spec = FEATURE_SETS[self.fs]
        num = list(spec["numeric"]) + [c for c in ("Year", "Session") if c in spec["onehot"]]
        cat = list(spec["rep_cols"]) + [c for c in spec["onehot"] if c not in ("Year", "Session")]
        return num, cat

    def _frame(self, X):
        num, cat = self._cols()
        Z = X[num + cat].copy()
        for c in cat:
            Z[c] = Z[c].astype(str)
        return Z, cat

    def fit(self, X, y):
        from catboost import CatBoostRegressor
        Z, cat = self._frame(X)
        self.model_ = CatBoostRegressor(depth=self.depth, learning_rate=self.learning_rate,
                                        iterations=self.iterations, l2_leaf_reg=self.l2_leaf_reg,
                                        random_strength=self.random_strength, loss_function="RMSE",
                                        random_seed=self.random_state, verbose=0, thread_count=1,
                                        cat_features=cat)
        self.model_.fit(Z, np.asarray(y))
        return self

    def predict(self, X):
        Z, _ = self._frame(X)
        return self.model_.predict(Z)


def catboost_model(fs="F3", n_iter=8, inner_cv=4):
    space = {
        "depth": randint(4, 9),
        "learning_rate": loguniform(0.01, 0.1),
        "iterations": randint(400, 1500),
        "l2_leaf_reg": loguniform(1, 30),
        "random_strength": loguniform(0.1, 5),
    }
    return RandomizedSearchCV(CatBoostNative(fs=fs), space, n_iter=n_iter,
                              cv=KFold(inner_cv, shuffle=True, random_state=SEED),
                              scoring="r2", n_jobs=4, random_state=SEED, refit=True)
