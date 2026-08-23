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


def fig_geography_ablation():
    """Coarse climate/snow layers are replaceable by raw coordinates -> they encode location,
    not snowpack physics."""
    d = pd.read_csv(os.path.join(RES, "geography_ablation.csv"))
    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    d = d.iloc[::-1]
    y = np.arange(len(d))
    col = [CB_PALETTE[0] if "full" in v else
           CB_PALETTE[1] if "REPLACED" in v else CB_PALETTE[7] for v in d.variant]
    ax.barh(y, d.roc_auc, xerr=[d.roc_auc - d.ci_low, d.ci_high - d.roc_auc],
            color=col, error_kw=dict(lw=1, capsize=3))
    for i, (v, a) in enumerate(zip(d.variant, d.roc_auc)):
        ax.text(a + 0.012, i, f"{a:.3f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{v}  ({n})" for v, n in zip(d.variant, d.n_features)], fontsize=8)
    ax.set_xlabel("ROC-AUC (XGBoost, spatial-block CV, 95% bootstrap CI)")
    ax.set_xlim(0.6, 1.0)
    ax.set_title("Coarse climate and snow predictors encode geography, not snowpack",
                 fontsize=9.5)
    savefig(fig, "fig_geography_ablation", outdir=FIGS)


def fig_model_ci():
    d = pd.read_csv(os.path.join(RES, "model_ci.csv")).sort_values("roc_auc")
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    y = np.arange(len(d))
    ax.errorbar(d.roc_auc, y, xerr=[d.roc_auc - d.ci_low, d.ci_high - d.roc_auc],
                fmt="o", color=CB_PALETTE[0], capsize=3, ms=6, lw=1.4)
    ax.set_yticks(y)
    ax.set_yticklabels([NICE.get(m, m) for m in d.model], fontsize=8.5)
    ax.set_xlabel("ROC-AUC, spatial-block CV (95% bootstrap CI)")
    ax.set_title("Model ranking with uncertainty — the two GBMs are indistinguishable",
                 fontsize=9.5)
    savefig(fig, "fig_model_ci", outdir=FIGS)


def fig_transfer_and_shap():
    """Spatial transferability between sub-basins, and SHAP importance with the self-audit flag."""
    t = pd.read_csv(os.path.join(RES, "transferability.csv"))
    s = pd.read_csv(os.path.join(RES, "shap_importance.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(9.8, 4.0))

    y = np.arange(len(t))
    axs[0].barh(y, t.roc_auc, xerr=[t.roc_auc - t.ci_low, t.ci_high - t.roc_auc],
                color=CB_PALETTE[0], error_kw=dict(lw=1, capsize=3), height=0.5)
    axs[0].axvline(0.939, color=CB_PALETTE[1], ls="--", lw=1.2,
                   label="within-region (spatial CV)")
    for i, a in enumerate(t.roc_auc):
        axs[0].text(a + 0.006, i, f"{a:.3f}", va="center", fontsize=8.5)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(["Chandra-Bhaga → Upper Beas", "Upper Beas → Chandra-Bhaga"],
                           fontsize=8.5)
    axs[0].set_xlim(0.5, 1.0)
    axs[0].set_xlabel("ROC-AUC on the unseen sub-basin (95% CI)")
    axs[0].legend(fontsize=7.5, loc="lower left")
    axs[0].set_title("Spatial transferability", fontsize=9.5)

    s = s.iloc[::-1]
    cols = [CB_PALETTE[1] if f == "tpi" else CB_PALETTE[2] for f in s.feature]
    axs[1].barh(np.arange(len(s)), s.mean_abs_shap, color=cols)
    axs[1].set_yticks(np.arange(len(s)))
    axs[1].set_yticklabels(s.feature, fontsize=8)
    axs[1].set_xlabel("mean |SHAP|")
    axs[1].set_title("SHAP importance — orange = artefact of our own\nrelease-point rule, excluded",
                     fontsize=9.5)
    fig.tight_layout()
    savefig(fig, "fig_transfer_shap", outdir=FIGS)


def fig_susceptibility_map(decim=4):
    """The deliverable map: susceptibility classes within potential release areas only."""
    import rasterio
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch
    from figures.pubstyle import SUSCEPT_COLORS, SUSCEPT_LABELS

    cls_p = os.path.join(RES, "susceptibility_classes.tif")
    dem_p = os.path.join(DATA, "processed", "base_utm", "base_nasadem_elevation.tif")

    with rasterio.open(cls_p) as s:
        cls = s.read(1, out_shape=(s.height // decim, s.width // decim))
        b = s.bounds
    with rasterio.open(dem_p) as s:
        dem = s.read(1, out_shape=(s.height // decim, s.width // decim)).astype("float32")
        dem[dem == s.nodata] = np.nan
    ext = [b.left / 1000, b.right / 1000, b.bottom / 1000, b.top / 1000]

    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    # hillshade-ish grey base so unclassified terrain still reads as topography
    ax.imshow(dem, extent=ext, cmap="Greys_r", vmin=np.nanpercentile(dem, 2),
              vmax=np.nanpercentile(dem, 98), alpha=0.55)
    cmap = ListedColormap(SUSCEPT_COLORS)
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)
    ax.imshow(np.ma.masked_invalid(cls), extent=ext, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xlabel("Easting (km, UTM 43N)")
    ax.set_ylabel("Northing (km)")
    ax.set_title("Avalanche susceptibility, Chandra-Bhaga & Upper Beas basins\n"
                 "XGBoost, spatial-block CV AUC 0.934; masked to potential release areas "
                 "(slope 28-60°)", fontsize=10)
    ax.legend(handles=[Patch(facecolor=c, label=l)
                       for c, l in zip(SUSCEPT_COLORS, SUSCEPT_LABELS)],
              loc="lower left", fontsize=8, title="Susceptibility", title_fontsize=8.5,
              framealpha=0.9, frameon=True)
    savefig(fig, "fig_susceptibility_map", outdir=FIGS)


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
    fig_geography_ablation()
    fig_model_ci()
    fig_transfer_and_shap()
    fig_susceptibility_map()
    fig_model_comparison()
    imp = fig_roc_and_importance()
    print("\ntop predictors (permutation importance, spatial fold):")
    for k, v in sorted(imp.items(), key=lambda x: -x[1]):
        print(f"   {k:26} {v:.4f}")
