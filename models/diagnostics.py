"""
models/diagnostics.py — uncertainty and "is it physics or geography?" diagnostics.

bootstrap_ci        : confidence intervals on out-of-fold AUC, so model rankings carry error bars
                      (a 0.001 gap between two models is noise, not a result).
geography_ablation  : the coarse-predictor test. Climate (1 km) and snow-cover (500 m) layers are
                      far coarser than avalanche release zones (10^2-10^5 m^2), so at that scale
                      they are smooth regional trend surfaces and a model can use them as a
                      geographic index rather than as snowpack physics. Replacing them with raw
                      coordinates tests exactly that: if performance is unchanged, they were only
                      carrying geography. Distance-to-lineament is included because the nearest
                      active fault is >=14 km away, making it another smooth regional gradient.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from .tabular import get_models, feature_columns

# predictors that are coarse or smooth enough to act as a geographic index
COARSE = ["worldclim_temp_winter", "worldclim_precip_winter", "modis_scd", "dist_to_lineament"]


def oof_predictions(df, model_key="xgb", spatial=True, k=5, feats=None, seed=42):
    """Out-of-fold probabilities under spatial-block (or random) CV."""
    feats = feats or feature_columns(df)
    X, y, g = df[feats].values, df["label"].values, df["block_id"].values
    if spatial:
        splits = GroupKFold(n_splits=min(k, len(np.unique(g)))).split(X, y, groups=g)
    else:
        from sklearn.model_selection import StratifiedKFold
        splits = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(X, y)
    yt, yp = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in splits:
        m = get_models(seed)[model_key]
        m.fit(X[tr], y[tr])
        yp[te] = m.predict_proba(X[te])[:, 1]
        yt[te] = y[te]
    return yt, yp


def bootstrap_ci(y, p, n_boot=2000, alpha=0.05, seed=42):
    """Percentile bootstrap CI for AUC (resampling observations)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return roc_auc_score(y, p), lo, hi


def compare_models_ci(df, models=("lgbm", "xgb", "rf", "svm_rbf", "logreg"), n_boot=2000, k=5):
    rows = []
    preds = {}
    for mk in models:
        try:
            y, p = oof_predictions(df, mk, spatial=True, k=k)
        except KeyError:
            continue
        preds[mk] = (y, p)
        auc, lo, hi = bootstrap_ci(y, p, n_boot=n_boot)
        rows.append(dict(model=mk, roc_auc=auc, ci_low=lo, ci_high=hi, ci_width=hi - lo))
        print(f"  {mk:<9} AUC {auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    res = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)

    # paired bootstrap: is the best model actually better than the runner-up?
    if len(res) >= 2:
        a, b = res.model.iloc[0], res.model.iloc[1]
        ya, pa = preds[a]
        _, pb = preds[b]
        rng = np.random.default_rng(0)
        diffs = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(ya), len(ya))
            if len(np.unique(ya[idx])) < 2:
                continue
            diffs.append(roc_auc_score(ya[idx], pa[idx]) - roc_auc_score(ya[idx], pb[idx]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        verdict = "SIGNIFICANT" if lo > 0 else "not significant (overlaps 0)"
        print(f"\n  paired bootstrap, {a} vs {b}: dAUC {np.mean(diffs):+.4f} "
              f"95% CI [{lo:+.4f}, {hi:+.4f}] -> {verdict}")
    return res


def geography_ablation(df, model_key="xgb", n_boot=1000, k=5):
    """Do the coarse layers carry snowpack information, or only geography?"""
    feats = feature_columns(df)
    fine = [f for f in feats if f not in COARSE]
    present_coarse = [f for f in COARSE if f in feats]

    variants = {
        "full (all predictors)": feats,
        "coarse REPLACED by x,y": fine + ["x", "y"],
        "coarse REMOVED": fine,
        "coarse ONLY": present_coarse,
        "x,y ONLY (pure geography)": ["x", "y"],
    }
    rows = []
    for name, cols in variants.items():
        cols = [c for c in cols if c in df.columns]
        y, p = oof_predictions(df, model_key, spatial=True, k=k, feats=cols)
        auc, lo, hi = bootstrap_ci(y, p, n_boot=n_boot)
        rows.append(dict(variant=name, n_features=len(cols), roc_auc=auc,
                         ci_low=lo, ci_high=hi))
        print(f"  {name:<26} {len(cols):>3} feats  AUC {auc:.3f}  [{lo:.3f}, {hi:.3f}]")
    return pd.DataFrame(rows)
