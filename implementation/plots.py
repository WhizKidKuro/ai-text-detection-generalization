import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, seaborn as sns
NAMES = {"A": "Detector A (TF-IDF + LogReg)", "B": "Detector B (stylometric + RF)", "L": "Length-only baseline"}

def heat(M, det, metric="auroc"):
    p = M[M.det == det].groupby(["train", "test"])[metric].mean().unstack()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    sns.heatmap(p, annot=True, fmt=".2f", cmap="crest", vmin=.4, vmax=1, ax=ax, cbar_kws={"label": metric.upper()})
    ax.set_title(f"{NAMES[det]}\ntrain domain (rows) → test domain (cols)", fontsize=10); fig.tight_layout(); return fig

def fpr_bar(F):
    fig, ax = plt.subplots(figsize=(6, 4)); groups = sorted(F.group.unique()); w = .38
    for i, g in enumerate(groups):
        d = F[F.group == g].sort_values("det"); x = np.arange(len(d)) + i * w - w / 2
        ax.bar(x, d.fpr, w, label=g, yerr=[d.fpr - d.lo, d.hi - d.fpr], capsize=4)
    ax.set_xticks(range(F.det.nunique())); ax.set_xticklabels([f"Detector {d}" for d in sorted(F.det.unique())])
    ax.set_ylabel("False positive rate (human called AI)"); ax.set_ylim(0, 1); ax.legend(title="writer group")
    ax.set_title("Who gets wrongly flagged? (95% bootstrap CI)", fontsize=10); fig.tight_layout(); return fig
