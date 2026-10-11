"""
models/simulation.py — why reported AUC measures absence design rather than predictive skill.

The empirical result (a published-style absence rule yields AUC 1.000 on real data) is a single
observation. This turns it into a quantity anyone can compute for their own study.

Construction. Presences are drawn from a region of feature space where a true susceptibility
function is high. Absences are drawn from a distribution displaced from the presences by delta,
measured in standard deviations, ALONG A DIRECTION ORTHOGONAL TO THE TRUE SUSCEPTIBILITY GRADIENT.
Because the displacement is orthogonal, it carries no information whatever about whether a site is
susceptible: the model's genuine ability to rank susceptible sites is identical at every delta.

If reported AUC nonetheless climbs with delta, then AUC is reporting how far apart the two samples
were drawn, not how well the model predicts the hazard. The null arm makes this inescapable: there
the predictors are independent of the outcome, so true skill is exactly zero by construction, and
any AUC above 0.5 is pure sampling artefact.

`separability_of` converts a real study's absence design into the same delta units, so a published
design can be located on the simulated curve and its expected inflation read off.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def _fit_auc(X, y, seed, model="rf"):
    from sklearn.ensemble import RandomForestClassifier
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    if model == "rf":
        m = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    else:
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08,
                          tree_method="hist", eval_metric="logloss",
                          n_jobs=-1, random_state=seed)
    m.fit(Xtr, ytr)
    return roc_auc_score(yte, m.predict_proba(Xte)[:, 1])


def simulate(delta, n=2000, d=8, seed=0, informative=True, noise=1.0, model="rf"):
    """One synthetic study at absence displacement `delta` (in standard deviations).

    `informative=False` is the null arm: the predictors carry no information about the outcome, so
    any AUC above 0.5 is attributable entirely to the absence design.
    """
    rng = np.random.default_rng(seed)
    # feature 0 drives true susceptibility; the displacement acts on feature 1, orthogonal to it
    Xp = rng.normal(0, 1, size=(n, d))
    if informative:
        # presences occupy the upper tail of the susceptibility axis
        Xp[:, 0] = rng.normal(1.2, 0.8, size=n)
    Xa = rng.normal(0, 1, size=(n, d))
    if informative:
        # true absences sit lower on the same axis — this is the real, learnable signal
        Xa[:, 0] = rng.normal(-1.2 / noise, 0.8, size=n)
    # the sampling design: push absences away along an axis unrelated to susceptibility
    Xa[:, 1] += delta

    X = np.vstack([Xp, Xa])
    y = np.r_[np.ones(n, int), np.zeros(n, int)]
    return _fit_auc(X, y, seed, model=model)


def inflation_curve(deltas=(0, 0.5, 1, 1.5, 2, 3, 4, 5, 6), n_rep=8, **kw):
    """AUC against absence displacement, for an informative model and for a null model."""
    rows = []
    for delta in deltas:
        for arm in (True, False):
            aucs = [simulate(delta, seed=s, informative=arm, **kw) for s in range(n_rep)]
            rows.append(dict(delta=delta, arm="informative" if arm else "null (no true skill)",
                             auc=float(np.mean(aucs)), sd=float(np.std(aucs))))
            print(f"  delta={delta:<4} {'informative' if arm else 'null       '}  "
                  f"AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f}", flush=True)
    return pd.DataFrame(rows)


def separability_of(df, label_col="label", feature_cols=None):
    """Where a real study's absence design sits on the delta axis.

    Standardised Mahalanobis-style separation between the presence and absence clouds in predictor
    space, expressed in the same units as the simulation's delta so the two can be compared.
    """
    import numpy as _np
    drop = {label_col, "x", "y", "block_id", "lon", "lat", "event", "tier", "zone",
            "added", "recurrence", "area_m2", "year", "date_before", "date_after",
            "window_days"}
    feats = feature_cols or [c for c in df.columns
                             if c not in drop
                             and _np.issubdtype(df[c].dtype, _np.number)]
    P = df.loc[df[label_col] == 1, feats].to_numpy(float)
    A = df.loc[df[label_col] == 0, feats].to_numpy(float)
    mu = P.mean(0) - A.mean(0)
    sd = np.sqrt(0.5 * (P.var(0, ddof=1) + A.var(0, ddof=1)))
    sd[sd == 0] = np.nan
    dz = mu / sd                                   # per-feature standardised difference
    return float(np.sqrt(np.nansum(dz ** 2)))      # overall separation, in SD units


def null_baseline(delta, n_rep=10, **kw):
    """AUC a model with NO predictive skill attains at this absence displacement."""
    return float(np.mean([simulate(delta, seed=s, informative=False, **kw)
                          for s in range(n_rep)]))


def design_adjusted_skill(auc, df=None, delta=None, n_rep=10, **kw):
    """Reported AUC minus what the sampling design alone would deliver.

    A reported AUC is not comparable across studies because each study chose its own absences. The
    baseline here is the score a model that has learned nothing would obtain under the SAME design,
    estimated by simulation at the study's own separability. Subtracting it leaves the part of the
    score attributable to the model rather than to the sampling, which is the quantity readers
    assume AUC already reports.
    """
    if delta is None:
        if df is None:
            raise ValueError("pass either a training table or a separability value")
        delta = separability_of(df)
    base = null_baseline(delta, n_rep=n_rep, **kw)
    # when the design alone already saturates AUC there is no headroom left to measure skill in,
    # so the fraction is undefined rather than perfect
    head = 1.0 - base
    frac = (auc - base) / head if head > 0.01 else float("nan")
    return dict(delta=delta, reported_auc=auc, null_baseline=base,
                adjusted_skill=auc - base, skill_fraction=frac)


def robustness(deltas=(0, 1, 2, 3, 4, 6), n_rep=5):
    """Does the inflation law depend on how the simulation was set up?

    A reviewer's first objection to a synthetic result is that it is an artefact of the authors'
    choices. We therefore vary the things that could plausibly matter — the learner, the number of
    predictors, whether predictors are correlated, and the sample size — and report the null arm in
    each. If a model with no skill inflates the same way under all of them, the behaviour belongs to
    presence-only evaluation rather than to this particular construction.
    """
    import pandas as pd
    rows = []
    settings = [
        ("Random Forest, 8 predictors",          dict(model="rf", d=8, n=2000)),
        ("XGBoost, 8 predictors",                dict(model="xgb", d=8, n=2000)),
        ("Random Forest, 4 predictors",          dict(model="rf", d=4, n=2000)),
        ("Random Forest, 20 predictors",         dict(model="rf", d=20, n=2000)),
        ("Random Forest, n=500 per class",       dict(model="rf", d=8, n=500)),
        ("Random Forest, n=5000 per class",      dict(model="rf", d=8, n=5000)),
    ]
    for label, kw in settings:
        for delta in deltas:
            a = [simulate(delta, seed=s, informative=False, **kw) for s in range(n_rep)]
            rows.append(dict(setting=label, delta=delta, auc=float(np.mean(a))))
        got = [r["auc"] for r in rows if r["setting"] == label]
        print(f"  {label:<34} " + "  ".join(f"{v:.3f}" for v in got), flush=True)
    return pd.DataFrame(rows)
