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
         ("ridge_S2", "ridge_F2", "Drop post-sale pedigree stats: F2 -> clean S2 (Ridge)"),
         ("ridge_S1", "ridge_S2", "Add hindsight sire/BS stats: S2 -> S1 (Ridge)"),
         ("lgbm_F0", "ridge_F0", "LightGBM vs Ridge, published predictors"),
         ("lgbm_S2", "ridge_S2", "LightGBM vs Ridge, clean S2"),
         ("cat_S2", "ridge_S2", "CatBoost vs Ridge, clean S2"),
         ("ensemble_S2", "ridge_S2", "Ensemble vs Ridge, clean S2"),
         ("ensemble_F3", "ensemble_S2", "Ensemble with post-sale stats (F3) vs clean (S2)")]
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
    ("− post-sale\npedigree stats\n(clean Ridge)", cvd[("Ridge", "S2")], "#2e3440"),
    ("Clean ensemble\n(nested CV)", cvd[("Average ensemble (Ridge + LightGBM + CatBoost)", "S2")], "#2e3440"),
    ("Clean ensemble,\nnext year's sale", temp["Average ensemble, S2 (clean)"], "#2e3440"),
]
fig, ax = plt.subplots(figsize=(11, 4.6), dpi=150)
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
ax.text(6, 0.71, "Revised model", ha="center", fontsize=9, color="#555")
ax.text(len(bars) - 0.6, -0.17, "Dark bars: fully clean (catalog-only inputs, no post-sale data). All other bars "
        "still include the cross statistics that contain the horse's own racing record.",
        ha="right", fontsize=7.5, color="#555", transform=ax.get_xaxis_transform())
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(f"{RESULTS}/fig_r2_path.png")
print("saved figure")
