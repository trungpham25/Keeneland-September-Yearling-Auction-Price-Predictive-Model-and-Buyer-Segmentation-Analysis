"""Step 5 - Paired model comparisons on the nested-CV folds and the summary
figure."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

from common import RESULTS, fmt_table

oof = pd.read_csv(f"{RESULTS}/02_oof_predictions.csv")
pairs = [("ridge_F1", "ridge_F0", "Add session (Ridge)"),
         ("ridge_F2", "ridge_F1", "Add consignor reputation (Ridge)"),
         ("ridge_F3", "ridge_F2", "Add extra pedigree stats (Ridge)"),
         ("lgbm_F0", "ridge_F0", "LightGBM vs Ridge, published predictors"),
         ("lgbm_F3", "ridge_F2", "LightGBM F3 vs Ridge F2"),
         ("cat_F3", "ridge_F2", "CatBoost F3 vs Ridge F2"),
         ("ensemble_F3", "ridge_F2", "Ensemble vs Ridge F2")]
rows = []
for a, b, label in pairs:
    d = np.array([r2_score(g.y, g[a]) - r2_score(g.y, g[b]) for _, g in oof.groupby("fold")])
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    rows.append({"Comparison": label, "Delta R2 (mean)": d.mean(), "min fold": d.min(), "max fold": d.max(),
                 "folds improved": f"{(d > 0).sum()}/5", "paired t": t,
                 "p (two-sided)": 2 * stats.t.sf(abs(t), len(d) - 1)})
paired = pd.DataFrame(rows).set_index("Comparison")
paired.to_csv(f"{RESULTS}/05_paired_comparisons.csv")
with open(f"{RESULTS}/05_paired_comparisons.md", "w") as f:
    f.write("# Step 5: paired comparisons on the 5 outer folds\n\n")
    f.write("Fold-level differences in out-of-fold R2 (same folds for every model). With only 5 folds the "
            "t-test is indicative; the training sets overlap, so it is anti-conservative.\n\n")
    f.write(fmt_table(paired) + "\n")
print(open(f"{RESULTS}/05_paired_comparisons.md").read())

# --- Figure: from the published number to the revised estimates ----------
leak = pd.read_csv(f"{RESULTS}/01_leakage_audit_splits.csv").groupby("variant")["R2"].mean()
cv = pd.read_csv(f"{RESULTS}/02_nested_cv.csv")
cvd = {(r.Model, r.Features): r.R2 for r in cv.itertuples()}
temp = pd.read_csv(f"{RESULTS}/03_temporal.csv").groupby("Model")["R2"].mean()

bars = [
    ("Published\n(as reported)", 0.5403, "#9aa0a6"),
    ("Published pipeline\nreproduced", leak["A"], "#9aa0a6"),
    ("− full-sample\nCook's filter", leak["B"], "#d08770"),
    ("− own price in\nreputation", leak["C"], "#bf616a"),
    ("+ session", cvd[("Ridge", "F1")], "#5e81ac"),
    ("+ consignor", cvd[("Ridge", "F2")], "#5e81ac"),
    ("Ensemble\n(nested CV)", cvd[("Average ensemble (Ridge + LightGBM + CatBoost)", "F3")], "#2e3440"),
    ("Ensemble, next\nyear's sale", temp["Average ensemble"], "#2e3440"),
]
fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
x = np.arange(len(bars))
ax.bar(x, [b[1] for b in bars], color=[b[2] for b in bars], width=0.65)
for i, b in enumerate(bars):
    ax.text(i, b[1] + 0.01, f"{b[1]:.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([b[0] for b in bars], fontsize=8)
ax.set_ylabel("Out-of-sample R² (log price)")
ax.set_ylim(0, 0.75)
ax.axvline(1.5, color="#ccc", lw=0.8, ls="--")
ax.axvline(3.5, color="#ccc", lw=0.8, ls="--")
ax.text(0.5, 0.71, "Published", ha="center", fontsize=9, color="#555")
ax.text(2.5, 0.71, "Removing leakage", ha="center", fontsize=9, color="#555")
ax.text(5.5, 0.71, "Revised model (leak-free)", ha="center", fontsize=9, color="#555")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(f"{RESULTS}/fig_r2_path.png")
print("saved figure")
