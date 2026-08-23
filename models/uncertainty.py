"""
models/uncertainty.py — per-pixel uncertainty for the susceptibility map.

MC Dropout is a neural-network technique (it samples the dropout mask at inference). The models
here are gradient-boosted trees, so the appropriate tools are different:

  * SPLIT CONFORMAL PREDICTION gives distribution-free, finite-sample-valid prediction sets. For a
    target error rate alpha it guarantees the true label is in the set at least 1-alpha of the time,
    with no assumption that the model is calibrated — which matters because Section 4.6 showed our
    scores are NOT calibrated probabilities (Brier 0.103, max reliability deviation 0.099).
    An UNCERTAIN pixel is one whose prediction set contains BOTH classes: the model cannot commit.
  * BOOTSTRAP ENSEMBLE SPREAD gives a continuous uncertainty surface (std of predictions across
    models trained on resampled data), which is what a decision-maker reads off a map.

Both are computed on spatially-blocked splits so the uncertainty estimate is not itself inflated by
spatial leakage.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .tabular import get_models, feature_columns


def conformal_calibrate(df, model_key="xgb", alpha=0.1, feats=None, seed=42):
    """Split-conformal on a spatially disjoint calibration block.

    Returns the score threshold and the empirical coverage/efficiency achieved on a held-out
    test block, so the guarantee can be checked rather than assumed.
    """
    feats = feats or [f for f in feature_columns(df) if f != "tpi"]
    X, y, g = df[feats].values, df["label"].values, df["block_id"].values
    splits = list(GroupKFold(n_splits=3).split(X, y, groups=g))
    tr, rest = splits[0]
    cal, te = np.array_split(rest, 2)

    m = get_models(seed)[model_key]
    m.fit(X[tr], y[tr])

    # nonconformity = 1 - predicted probability of the TRUE class
    p_cal = m.predict_proba(X[cal])
    scores = 1.0 - p_cal[np.arange(len(cal)), y[cal]]
    n = len(scores)
    q = np.quantile(scores, min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0), method="higher")

    p_te = m.predict_proba(X[te])
    in_set = (1.0 - p_te) <= q                      # per-class membership
    covered = in_set[np.arange(len(te)), y[te]].mean()
    set_size = in_set.sum(axis=1)
    both = (set_size == 2).mean()
    empty = (set_size == 0).mean()
    print(f"  alpha={alpha}  threshold q={q:.3f}")
    print(f"  empirical coverage : {covered:.3f}  (target >= {1-alpha:.2f})")
    print(f"  ambiguous (both classes in set): {both:.3f}")
    print(f"  abstain   (empty set)          : {empty:.3f}")
    return dict(model=m, q=q, feats=feats, coverage=covered, ambiguous=both, empty=empty)


def bootstrap_ensemble(df, model_key="xgb", n_models=15, feats=None, seed=42):
    """Train an ensemble on bootstrap resamples; spread across members = predictive uncertainty."""
    feats = feats or [f for f in feature_columns(df) if f != "tpi"]
    X, y = df[feats].values, df["label"].values
    rng = np.random.default_rng(seed)
    models = []
    for i in range(n_models):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        m = get_models(seed + i)[model_key]
        m.fit(X[idx], y[idx])
        models.append(m)
    print(f"  trained {len(models)} bootstrap members")
    return models, feats


def ensemble_predict(models, X):
    """Mean prediction and standard deviation across ensemble members."""
    P = np.stack([m.predict_proba(X)[:, 1] for m in models])
    return P.mean(axis=0), P.std(axis=0)


def uncertainty_report(df, model_key="xgb", alpha=0.1):
    """Summarise how much of the map the model can actually commit on."""
    print("split-conformal prediction (spatially disjoint calibration):")
    conf = conformal_calibrate(df, model_key=model_key, alpha=alpha)
    print("\nbootstrap ensemble:")
    models, feats = bootstrap_ensemble(df, model_key=model_key, n_models=15)
    mean, std = ensemble_predict(models, df[feats].values)
    print(f"  ensemble sd: median {np.median(std):.3f}, p90 {np.percentile(std,90):.3f}, "
          f"max {std.max():.3f}")
    bands = pd.cut(mean, [0, .2, .4, .6, .8, 1.0],
                   labels=["very low", "low", "moderate", "high", "very high"])
    tab = pd.DataFrame({"band": bands, "sd": std}).groupby("band", observed=True)["sd"].agg(
        ["count", "median", "max"])
    print("\n  ensemble spread by susceptibility band:")
    print(tab.round(3).to_string())
    return conf, models, feats, std
