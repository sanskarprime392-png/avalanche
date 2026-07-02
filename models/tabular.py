"""
models/tabular.py — baseline classifiers + the spatial-vs-random CV harness.

This is the paper's core experiment: the same models evaluated two ways —
  * random CV (what the literature does; spatially leaky, inflates metrics)
  * spatial block CV (GroupKFold on block_id; honest generalization)
The gap between the two IS a result.

Models: RF + LogReg + SVM-RBF (mirroring the anchor paper) plus XGBoost / LightGBM
(the stronger GBM baselines the paper skipped; import-guarded).

Usage (Colab):
    from models.tabular import compare_all, fit_full_and_importance
    results = compare_all(df, out_dir=".../data/results")           # df from build_training_table
    model, imp = fit_full_and_importance(df, "rf")
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             cohen_kappa_score, roc_auc_score, average_precision_score,
                             confusion_matrix)

META_COLS = {"lon", "lat", "x", "y", "label", "event", "tier", "zone", "added", "block_id", "_k"}


def feature_columns(df):
    return [c for c in df.columns if c not in META_COLS]


def get_models(seed=42):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    models = {
        "rf": RandomForestClassifier(n_estimators=700, random_state=seed, n_jobs=-1),
        "logreg": make_pipeline(StandardScaler(),
                                LogisticRegression(C=0.1, penalty="l1", solver="liblinear",
                                                   max_iter=2000, random_state=seed)),
        "svm_rbf": make_pipeline(StandardScaler(),
                                 SVC(C=1.0, gamma="scale", probability=True, random_state=seed)),
    }
    try:
        from xgboost import XGBClassifier
        models["xgb"] = XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=5,
                                      subsample=0.9, colsample_bytree=0.9,
                                      eval_metric="logloss", random_state=seed)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        models["lgbm"] = LGBMClassifier(n_estimators=600, learning_rate=0.05,
                                        random_state=seed, verbose=-1)
    except ImportError:
        pass
    return models


def _metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = dict(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        kappa=cohen_kappa_score(y_true, y_pred),
        fpr=fp / (fp + tn) if (fp + tn) else np.nan,
    )
    try:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
        out["pr_auc"] = average_precision_score(y_true, y_prob)
    except ValueError:                       # single-class test fold (possible with spatial blocks)
        out["roc_auc"] = out["pr_auc"] = np.nan
    return out


def run_cv(df, model_key, spatial=True, k=5, seed=42):
    """K-fold CV for one model. spatial=True -> GroupKFold on block_id (no leakage);
    spatial=False -> StratifiedKFold (the literature's leaky default, for comparison)."""
    X = df[feature_columns(df)].values
    y = df["label"].values
    groups = df["block_id"].values

    if spatial:
        k_eff = min(k, len(np.unique(groups)))
        if k_eff < k:
            print(f"  only {k_eff} spatial blocks -> using {k_eff} folds")
        splits = GroupKFold(n_splits=k_eff).split(X, y, groups=groups)
    else:
        splits = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(X, y)

    rows = []
    for i, (tr, te) in enumerate(splits):
        model = get_models(seed)[model_key]
        model.fit(X[tr], y[tr])
        prob = model.predict_proba(X[te])[:, 1]
        rows.append(dict(fold=i, n_test=len(te), **_metrics(y[te], prob)))
    return pd.DataFrame(rows)


def compare_all(df, out_dir, k=5, seed=42):
    """Every model x {random, spatial} CV. Saves the full table + prints the summary."""
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for key in get_models(seed):
        for spatial in (False, True):
            scheme = "spatial" if spatial else "random"
            print(f"running {key} / {scheme} CV ...")
            fold_df = run_cv(df, key, spatial=spatial, k=k, seed=seed)
            m = fold_df.mean(numeric_only=True)
            results.append(dict(model=key, cv=scheme,
                                **{c: m[c] for c in ("accuracy", "kappa", "roc_auc", "pr_auc",
                                                     "precision", "recall", "fpr")}))
    res = pd.DataFrame(results).sort_values(["model", "cv"]).reset_index(drop=True)
    path = os.path.join(out_dir, "cv_comparison.csv")
    res.to_csv(path, index=False)
    print(f"\nsaved -> {path}\n")
    print(res.round(3).to_string(index=False))
    # the headline: how much random CV inflates AUC vs honest spatial CV
    piv = res.pivot(index="model", columns="cv", values="roc_auc")
    if {"random", "spatial"}.issubset(piv.columns):
        piv["auc_inflation"] = piv["random"] - piv["spatial"]
        print("\nAUC inflation (random - spatial):")
        print(piv.round(3).to_string())
    return res


def fit_full_and_importance(df, model_key="rf", seed=42):
    """Fit on all rows; return (model, importance series) for SHAP / mapping later."""
    feats = feature_columns(df)
    model = get_models(seed)[model_key]
    model.fit(df[feats].values, df["label"].values)
    est = model[-1] if hasattr(model, "steps") else model
    if hasattr(est, "feature_importances_"):
        imp = pd.Series(est.feature_importances_, index=feats).sort_values(ascending=False)
    elif hasattr(est, "coef_"):
        imp = pd.Series(np.abs(est.coef_).ravel(), index=feats).sort_values(ascending=False)
    else:
        imp = pd.Series(dtype=float)
    return model, imp
