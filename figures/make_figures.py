"""Generate the publication figures from the trained results.

    python figures/make_figures.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc as auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figures.pubstyle import set_pub_style, savefig, CB_PALETTE
from models.tabular import get_models, feature_columns

DATA = r"C:\Users\Sanskar\Documents\avalanche_data"
RES = os.path.join(DATA, "results")
FIGS = os.path.join(DATA, "figures")
NICE = {"rf": "Random Forest", "xgb": "XGBoost", "lgbm": "LightGBM",
        "svm_rbf": "SVM (RBF)", "logreg": "Logistic Reg."}


def fig_decomposition():
    """The headline: where the published optimism actually comes from."""
    d = pd.read_csv(os.path.join(RES, "leakage_decomposition.csv"))
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    cfgs = ["A", "B", "C", "D"]
    labels = ["A\npaper absences\nrandom CV", "B\npaper absences\nspatial CV",
              "C\nmatched absences\nrandom CV", "D\nmatched absences\nspatial CV"]
    x = np.arange(4)
    w = 0.36
    for i, (mk, col) in enumerate(zip(["rf", "xgb"], [CB_PALETTE[0], CB_PALETTE[1]])):
        v = [d[(d.config == c) & (d.model == mk)].roc_auc.iloc[0] for c in cfgs]
        b = ax.bar(x + (i - 0.5) * w, v, w, label=NICE[mk], color=col)
        ax.bar_label(b, fmt="%.3f", fontsize=7.5, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.85, 1.03)
    ax.axhline(1.0, color="0.6", lw=0.7, ls=":")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Where the reported skill comes from: sampling design, not spatial validation",
                 fontsize=9.5)
    savefig(fig, "fig_leakage_decomposition", outdir=FIGS)


def fig_model_comparison():
    d = pd.read_csv(os.path.join(RES, "cv_comparison.csv"))
    order = d[d.cv == "spatial"].sort_values("roc_auc", ascending=False).model.tolist()
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    x = np.arange(len(order))
    w = 0.38
    for i, (cv, col) in enumerate(zip(["random", "spatial"], [CB_PALETTE[5], CB_PALETTE[1]])):
        v = [d[(d.model == m) & (d.cv == cv)].roc_auc.iloc[0] for m in order]
        b = ax.bar(x + (i - 0.5) * w, v, w,
                   label="random CV" if cv == "random" else "spatial-block CV", color=col)
        ax.bar_label(b, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[m] for m in order], fontsize=8, rotation=12, ha="right")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.85, 1.0)
    ax.legend(fontsize=8)
    ax.set_title("Model comparison under honest spatial validation", fontsize=9.5)
    savefig(fig, "fig_model_comparison", outdir=FIGS)


def fig_label_sensitivity():
    """The headline finding is independent of inventory quality: paper-style sampling gives
    AUC 1.000 at every label-confidence threshold, including n=240 (close to the published n=118)."""
    d = pd.read_csv(os.path.join(RES, "label_quality_sensitivity.csv"))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for cfg, col, lbl in (("A", CB_PALETTE[1], "A — paper-style absences"),
                          ("D", CB_PALETTE[0], "D — terrain-matched absences")):
        s = d[d.config == cfg].sort_values("min_recurrence")
        ax.plot(s.min_recurrence, s.roc_auc, "o-", color=col, lw=1.8, ms=6, label=lbl)
        for _, r in s.iterrows():
            ax.annotate(f"{r.roc_auc:.3f}", (r.min_recurrence, r.roc_auc),
                        textcoords="offset points", xytext=(0, 8 if cfg == "A" else -14),
                        ha="center", fontsize=7.5, color=col)
    ns = d[d.config == "A"].sort_values("min_recurrence")
    ax.set_xticks(ns.min_recurrence)
    ax.set_xticklabels([f"≥{int(r.min_recurrence)}\n(n={int(r.n_presence):,})"
                        for _, r in ns.iterrows()], fontsize=8)
    ax.set_xlabel("Inventory confidence threshold (winters a slope was flagged)")
    ax.set_ylabel("ROC-AUC (XGBoost)")
    ax.set_ylim(0.85, 1.03)
    ax.legend(fontsize=8, loc="center left")
    ax.set_title("The sampling artefact is independent of inventory quality", fontsize=9.5)
    savefig(fig, "fig_label_sensitivity", outdir=FIGS)


def fig_roc_and_importance():
    df = pd.read_csv(os.path.join(DATA, "training_table_matched.csv"))
    feats = feature_columns(df)
    X, y, g = df[feats].values, df["label"].values, df["block_id"].values

    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for mk, col in zip(["xgb", "rf", "logreg"], CB_PALETTE):
        yt, yp = [], []
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups=g):
            m = get_models()[mk]
            m.fit(X[tr], y[tr])
            yp.append(m.predict_proba(X[te])[:, 1])
            yt.append(y[te])
        yt, yp = np.concatenate(yt), np.concatenate(yp)
        fpr, tpr, _ = roc_curve(yt, yp)
        axs[0].plot(fpr, tpr, color=col, lw=1.6,
                    label=f"{NICE[mk]} (AUC={auc_score(fpr, tpr):.3f})")
    axs[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axs[0].set_xlabel("False positive rate")
    axs[0].set_ylabel("True positive rate")
    axs[0].set_title("ROC — spatial-block cross-validation", fontsize=9.5)
    axs[0].legend(loc="lower right", fontsize=8)

    from sklearn.inspection import permutation_importance
    tr, te = next(iter(GroupKFold(n_splits=5).split(X, y, groups=g)))
    m = get_models()["xgb"]
    m.fit(X[tr], y[tr])
    r = permutation_importance(m, X[te], y[te], n_repeats=10, random_state=42,
                               scoring="roc_auc", n_jobs=-1)
    idx = np.argsort(r.importances_mean)[-12:]
    axs[1].barh([feats[i] for i in idx], r.importances_mean[idx],
                xerr=r.importances_std[idx], color=CB_PALETTE[2])
    axs[1].set_xlabel("Permutation importance (drop in AUC)")
    axs[1].set_title("Predictor importance — permutation, held-out spatial fold", fontsize=9.5)
    fig.tight_layout()
    savefig(fig, "fig_roc_importance", outdir=FIGS)
    return dict(zip([feats[i] for i in idx], r.importances_mean[idx]))


if __name__ == "__main__":
    set_pub_style()
    os.makedirs(FIGS, exist_ok=True)
    fig_decomposition()
    fig_label_sensitivity()
    fig_model_comparison()
    imp = fig_roc_and_importance()
    print("\ntop predictors (permutation importance, spatial fold):")
    for k, v in sorted(imp.items(), key=lambda x: -x[1]):
        print(f"   {k:26} {v:.4f}")
