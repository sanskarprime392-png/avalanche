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


def sub_basin_transfer(df, split_lat_utm=3_585_000.0, model_key="xgb", n_boot=1000):
    """Train on one sub-basin, test on the other — the strongest generalisation evidence.

    The Pir Panjal crest near Rohtang (~32.37 N) separates the Upper Beas catchment to the south
    from Chandra-Bhaga to the north, so a northing split approximates the basin divide. Transfer
    performance answers "would this model work somewhere it has never seen?", which spatial-block
    CV only partially tests.
    """
    feats = feature_columns(df)
    north = df[df.y >= split_lat_utm]
    south = df[df.y < split_lat_utm]
    print(f"  north (Chandra-Bhaga): {len(north)} rows, {int(north.label.sum())} presence")
    print(f"  south (Upper Beas)   : {len(south)} rows, {int(south.label.sum())} presence")

    rows = []
    for name, tr, te in (("north -> south", north, south), ("south -> north", south, north)):
        m = get_models()[model_key]
        m.fit(tr[feats].values, tr["label"].values)
        p = m.predict_proba(te[feats].values)[:, 1]
        y = te["label"].values
        auc, lo, hi = bootstrap_ci(y, p, n_boot=n_boot)
        rows.append(dict(transfer=name, n_train=len(tr), n_test=len(te),
                         roc_auc=auc, ci_low=lo, ci_high=hi))
        print(f"  {name:<16} AUC {auc:.3f}  [{lo:.3f}, {hi:.3f}]")
    return pd.DataFrame(rows)


def response_curves(df, variables, model_key="xgb", grid=25, seed=42):
    """Partial-dependence response curves — is the model's behaviour physically sensible?

    NOTE: elevation, slope and aspect are matched between presences and background by design, so
    their curves are necessarily flat and carry NO physical information here. Only unmatched
    predictors (curvature, ruggedness, position, wetness, distances) can be audited this way.
    """
    from sklearn.inspection import partial_dependence
    feats = feature_columns(df)
    m = get_models(seed)[model_key]
    m.fit(df[feats].values, df["label"].values)
    out = {}
    for v in variables:
        if v not in feats:
            continue
        pd_res = partial_dependence(m, df[feats].values, [feats.index(v)],
                                    grid_resolution=grid, kind="average")
        out[v] = (np.asarray(pd_res["grid_values"][0]), np.asarray(pd_res["average"][0]))
    return out


def shap_summary(df, model_key="xgb", max_display=14, seed=42, sample=3000):
    """Mean |SHAP| per predictor on a held-out spatial fold."""
    import shap
    feats = feature_columns(df)
    X, y, g = df[feats].values, df["label"].values, df["block_id"].values
    tr, te = next(iter(GroupKFold(n_splits=5).split(X, y, groups=g)))
    m = get_models(seed)[model_key]
    m.fit(X[tr], y[tr])
    Xte = X[te][:sample]
    expl = shap.TreeExplainer(m)
    sv = expl.shap_values(Xte)
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:max_display]
    return pd.DataFrame({"feature": [feats[i] for i in order],
                         "mean_abs_shap": mean_abs[order]}), sv, Xte, [feats[i] for i in order]


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
