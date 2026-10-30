"""
designcheck.py — run the absence-design comparison on your own susceptibility study.

This is the practical content of the paper in one file, with no dependency on the rest of the
repository. It does not try to correct a reported AUC: Section 4.2 shows the absolute skill of a
presence-only model is not identifiable from the data it was fitted to. What it does is measure the
one thing that IS identifiable — how much of your reported number is produced by where you drew
your absences — by holding your presences, predictors and model fixed and varying only the sampling.

Two functions:

    separability(df)          how far your absences sit from your presences, in SD units
    compare_designs(df, ...)  the same model scored against several absence designs

Typical use, where `presences` is a table of predictor values at your event locations and
`background` is a large pool of candidate absence locations with the same columns:

    import pandas as pd
    from designcheck import compare_designs

    result = compare_designs(presences, background,
                             designs={"as published": None,
                                      "terrain-matched": ["slope", "elevation"]})
    print(result)

A spread of more than a few hundredths across designs means your headline figure is substantially a
property of the sampling rather than the model. The remedy is not a correction — it is to report the
comparison, and to evaluate on absences drawn like your presences.

Requires numpy, pandas, scikit-learn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

__all__ = ["separability", "matched_absences", "compare_designs"]


def _numeric(df: pd.DataFrame, exclude=()) -> list[str]:
    drop = {"label", "x", "y", "lon", "lat", "block_id", *exclude}
    return [c for c in df.columns
            if c not in drop and np.issubdtype(df[c].dtype, np.number)]


def separability(df: pd.DataFrame, label_col: str = "label", features=None) -> float:
    """Standardised distance between the presence and absence clouds, in SD units.

    The multivariate form of Cohen's d, using per-predictor pooled standard deviations. It is a
    summary, not a canonical quantity: a Mahalanobis form that accounts for predictor correlation
    gives a different number on the same data, so treat it as a comparative measure between designs
    on one dataset rather than a value to compare between papers.
    """
    feats = features or _numeric(df, exclude=(label_col,))
    P = df.loc[df[label_col] == 1, feats].to_numpy(float)
    A = df.loc[df[label_col] == 0, feats].to_numpy(float)
    P, A = P[np.isfinite(P).all(1)], A[np.isfinite(A).all(1)]
    sd = np.sqrt(0.5 * (P.var(0, ddof=1) + A.var(0, ddof=1)))
    sd[sd == 0] = np.nan
    return float(np.sqrt(np.nansum(((P.mean(0) - A.mean(0)) / sd) ** 2)))


def matched_absences(presences: pd.DataFrame, background: pd.DataFrame,
                     match_on: list[str], n_bins: int = 8, ratio: float = 1.0,
                     seed: int = 0) -> pd.DataFrame:
    """Draw absences whose distribution over `match_on` follows the presences'.

    Matching makes the two classes comparable on the named predictors, so the model must separate
    them on something else. Those predictors can no longer be assessed for importance, which is the
    price of the comparison and should be stated.
    """
    rng = np.random.default_rng(seed)
    edges = {c: np.quantile(presences[c].dropna(), np.linspace(0, 1, n_bins + 1)[1:-1])
             for c in match_on}

    def key(frame):
        k = np.zeros(len(frame), dtype=np.int64)
        for c in match_on:
            k = k * (n_bins + 1) + np.digitize(frame[c].to_numpy(float), edges[c])
        return k

    want = pd.Series(key(presences)).value_counts()
    bg_key = key(background)
    picks = []
    for stratum, count in want.items():
        pool = np.flatnonzero(bg_key == stratum)
        if len(pool) == 0:
            continue
        take = int(round(count * ratio))
        picks.append(rng.choice(pool, size=min(take, len(pool)), replace=False))
    if not picks:
        raise ValueError("no background points fell in any presence stratum")
    out = background.iloc[np.concatenate(picks)].copy()

    # top up from the whole pool if strata were thin, so the ratio is still met
    short = int(round(len(presences) * ratio)) - len(out)
    if short > 0:
        rest = background.index.difference(out.index)
        if len(rest):
            out = pd.concat([out, background.loc[
                rng.choice(rest, size=min(short, len(rest)), replace=False)]])
    return out


def compare_designs(presences: pd.DataFrame, background: pd.DataFrame,
                    designs: dict, features=None, model=None, k: int = 5,
                    groups_col: str | None = None, ratio: float = 1.0,
                    seed: int = 0) -> pd.DataFrame:
    """Score one model against several absence designs, everything else held fixed.

    `designs` maps a label to either None (draw absences at random from the pool, the common
    default) or a list of predictor names to match on. Pass `groups_col` to use spatial blocks for
    cross-validation; without it the folds are random, which is itself a source of optimism.
    """
    feats = features or _numeric(presences)
    rows = []
    for name, match_on in designs.items():
        rng = np.random.default_rng(seed)
        if match_on:
            absences = matched_absences(presences, background, match_on, ratio=ratio, seed=seed)
        else:
            take = min(int(round(len(presences) * ratio)), len(background))
            absences = background.iloc[rng.choice(len(background), take, replace=False)]

        df = pd.concat([presences.assign(label=1), absences.assign(label=0)], ignore_index=True)
        df = df.dropna(subset=feats)
        X, y = df[feats].to_numpy(float), df["label"].to_numpy(int)

        if groups_col and groups_col in df:
            g = df[groups_col].to_numpy()
            splits = GroupKFold(n_splits=min(k, len(np.unique(g)))).split(X, y, groups=g)
            scheme = "spatial blocks"
        else:
            splits = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(X, y)
            scheme = "random folds"

        oof = np.zeros(len(y))
        for tr, te in splits:
            m = model or RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        rows.append(dict(design=name, n=len(df), cv=scheme,
                         separability=separability(df, features=feats),
                         auc=roc_auc_score(y, oof)))

    out = pd.DataFrame(rows).sort_values("separability").reset_index(drop=True)
    spread = out.auc.max() - out.auc.min()
    out.attrs["spread"] = spread
    print(out.round(3).to_string(index=False))
    print(f"\nAUC spread across designs: {spread:.3f}")
    if spread > 0.03:
        print("Your reported figure depends substantially on the absence design. Report this "
              "comparison, and evaluate on absences drawn like your presences.")
    return out
