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


def _make(delta, n, d, seed, skill, structure, rng=None):
    """Synthetic study with a controllable amount of TRUE skill and a chosen data structure."""
    rng = rng or np.random.default_rng(seed)
    Xp = rng.normal(0, 1, size=(n, d))
    Xa = rng.normal(0, 1, size=(n, d))
    # `skill` sets how far apart the classes are on the informative axis, i.e. how learnable the
    # problem genuinely is, independently of where the absences are placed
    Xp[:, 0] = rng.normal(+skill, 0.8, size=n)
    Xa[:, 0] = rng.normal(-skill, 0.8, size=n)

    if structure == "correlated":
        A = rng.normal(0, 1, size=(d, d)) * 0.35 + np.eye(d)
        Xp, Xa = Xp @ A, Xa @ A
    elif structure == "nonlinear":
        # the true boundary is a ring rather than a hyperplane
        Xp[:, 0] = np.abs(Xp[:, 0]) + skill
        Xa[:, 0] = np.abs(Xa[:, 0])
    elif structure == "clustered":
        # presences arrive in spatial clumps, as real inventories do
        k = max(n // 50, 2)
        cent = rng.normal(0, 1.5, size=(k, d))
        Xp = cent[rng.integers(0, k, n)] + rng.normal(0, 0.4, size=(n, d))
        Xp[:, 0] += skill
    elif structure == "imbalanced":
        Xa = np.repeat(Xa, 5, axis=0)[: n * 5]

    Xa[:, 1] += delta
    X = np.vstack([Xp, Xa])
    y = np.r_[np.ones(len(Xp), int), np.zeros(len(Xa), int)]
    return X, y


def _auc(X, y, seed):
    from sklearn.ensemble import RandomForestClassifier
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    m = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    m.fit(Xtr, ytr)
    return roc_auc_score(yte, m.predict_proba(Xte)[:, 1])


def validate_correction(structures=("gaussian", "correlated", "nonlinear", "clustered",
                                    "imbalanced", "label_noise"),
                        skills=(0.0, 0.4, 0.8, 1.2), deltas=(0, 1, 2, 3, 4),
                        n=1500, d=8, n_rep=4):
    """Does design-adjusted skill recover the same value at every absence displacement?

    The quantity the diagnostic should recover is the AUC the study would have reported had its
    absences been drawn without displacement — the honest evaluation — expressed above chance.
    Because true skill is held fixed while delta varies, a working diagnostic returns a constant.
    Reporting where it stays constant and where it drifts is what establishes its range of validity.
    """
    import pandas as pd
    rows = []
    for structure in structures:
        for skill in skills:
            noise = structure == "label_noise"
            st = "gaussian" if noise else structure
            # reference: the honest evaluation, absences undisplaced
            ref = []
            for s in range(n_rep):
                X, y = _make(0.0, n, d, s, skill, st)
                if noise:
                    rng = np.random.default_rng(1000 + s)
                    flip = rng.random(len(y)) < 0.20
                    y = np.where(flip, 1 - y, y)
                ref.append(_auc(X, y, s))
            ref_skill = float(np.mean(ref)) - 0.5
            for delta in deltas:
                obs, adj = [], []
                for s in range(n_rep):
                    X, y = _make(delta, n, d, s, skill, st)
                    if noise:
                        rng = np.random.default_rng(1000 + s)
                        flip = rng.random(len(y)) < 0.20
                        y = np.where(flip, 1 - y, y)
                    a = _auc(X, y, s)
                    obs.append(a)
                base = null_baseline(delta, n_rep=4, n=n, d=d)
                adj = float(np.mean(obs)) - base
                rows.append(dict(structure=structure, true_skill=skill, delta=delta,
                                 reference=ref_skill, observed_auc=float(np.mean(obs)),
                                 baseline=base, adjusted=adj, error=adj - ref_skill))
        print(f"  {structure} done", flush=True)
    return pd.DataFrame(rows)


def structured_noise_test(deltas=(0, 1, 2, 3, 4, 6), rates=(0.0, 0.2, 0.5, 0.8),
                          n=1500, d=8, n_rep=4):
    """Does label error that is STRUCTURED in predictor space overturn the inflation result?

    Verification showed our own false positives are not scattered: they sit about 900 m lower than
    the verified events, and the background is elevation-matched, so the error is correlated with a
    variable the sampling design acts on. The referee's objection is that identical labels need not
    affect every design equally under such structure.

    Here a fraction `rate` of presences are replaced by points drawn from the low end of the
    informative axis — the synthetic analogue of melt-out false positives at low elevation — and the
    null curve is recomputed. If the relationship between displacement and reported AUC survives, the
    inflation result does not depend on labels being clean.
    """
    import pandas as pd
    rng = np.random.default_rng(7)
    rows = []
    for rate in rates:
        for delta in deltas:
            aucs = []
            for s in range(n_rep):
                r = np.random.default_rng(s)
                Xp = r.normal(0, 1, size=(n, d))
                Xp[:, 0] = r.normal(1.2, 0.8, size=n)
                k = int(rate * n)
                if k:
                    # contaminating presences drawn from the LOW end of the informative axis
                    Xp[:k, 0] = r.normal(-1.0, 0.8, size=k)
                Xa = r.normal(0, 1, size=(n, d))
                Xa[:, 0] = r.normal(-1.2, 0.8, size=n)
                Xa[:, 1] += delta
                X = np.vstack([Xp, Xa])
                y = np.r_[np.ones(n, int), np.zeros(n, int)]
                aucs.append(_fit_auc(X, y, s))
            rows.append(dict(noise_rate=rate, delta=delta, auc=float(np.mean(aucs))))
        got = [r["auc"] for r in rows if r["noise_rate"] == rate]
        print(f"  contamination {rate:.0%}  " + "  ".join(f"{v:.3f}" for v in got), flush=True)
    return pd.DataFrame(rows)


def direction_test(deltas=(0, 1, 2, 3, 4, 6), n=1500, d=8, n_rep=4,
                   informative=True):
    """Is the inflation specific to displacing absences along ONE orthogonal axis?

    The construction so far moves absences along a single axis chosen to carry no information about
    the outcome. A reviewer may reasonably ask whether that direction is special. Here the same
    displacement magnitude is distributed in four ways: along one axis, spread evenly over several
    uninformative axes, along a random direction in the uninformative subspace, and - the hardest
    case - partly along the informative axis itself, which is what a real absence rule that favours
    gentle terrain actually does.
    """
    import pandas as pd
    rows = []
    modes = ("single axis", "spread over 5 axes", "random direction", "30% along informative axis")
    for mode in modes:
        for delta in deltas:
            aucs = []
            for s in range(n_rep):
                r = np.random.default_rng(s)
                Xp = r.normal(0, 1, size=(n, d))
                Xa = r.normal(0, 1, size=(n, d))
                if informative:
                    Xp[:, 0] = r.normal(1.2, 0.8, size=n)
                    Xa[:, 0] = r.normal(-1.2, 0.8, size=n)
                if mode == "single axis":
                    Xa[:, 1] += delta
                elif mode == "spread over 5 axes":
                    # same total displacement, divided across axes so each moves less
                    Xa[:, 1:6] += delta / np.sqrt(5)
                elif mode == "random direction":
                    v = r.normal(size=d - 1)
                    v /= np.linalg.norm(v)
                    Xa[:, 1:] += delta * v
                else:
                    Xa[:, 0] += 0.3 * delta          # partly along the informative axis
                    Xa[:, 1] += np.sqrt(max(delta ** 2 - (0.3 * delta) ** 2, 0))
                X = np.vstack([Xp, Xa])
                y = np.r_[np.ones(n, int), np.zeros(n, int)]
                aucs.append(_fit_auc(X, y, s))
            rows.append(dict(direction=mode, delta=delta, auc=float(np.mean(aucs))))
        got = [r["auc"] for r in rows if r["direction"] == mode]
        print(f"  {mode:<28} " + "  ".join(f"{v:.3f}" for v in got), flush=True)
    return pd.DataFrame(rows)


def compare_presence_only(deltas=(0, 1, 2, 3, 4, 6), n=1500, d=8, n_rep=4):
    """Does an established presence-only method avoid the design dependence?

    The paper's recommendation - redraw the absences - is only useful if something better exists.
    Two standard treatments of presence-only data are compared against ordinary classification:

      * Elkan & Noto (2008) PU correction. A classifier is trained on labelled-versus-unlabelled
        data and its scores divided by c, the estimated probability that a positive is labelled.
        Because c is a constant, the RANKING is unchanged, so AUC is identical to the naive model.
        This is worth demonstrating rather than asserting: PU correction fixes calibration, not the
        design dependence of a ranking statistic.
      * A one-class model fitted to the PRESENCES ALONE, which never sees the absences and so
        cannot inherit anything from how they were drawn.

    Reported as AUC against a FIXED honest test set (absences undisplaced), so the three methods are
    judged on the same question regardless of how their training absences were placed.
    """
    import pandas as pd
    from sklearn.ensemble import IsolationForest, RandomForestClassifier

    rows = []
    for delta in deltas:
        naive, pu, oneclass = [], [], []
        for s in range(n_rep):
            r = np.random.default_rng(s)
            Xp = r.normal(0, 1, size=(n, d)); Xp[:, 0] = r.normal(1.2, 0.8, size=n)
            Xa = r.normal(0, 1, size=(n, d)); Xa[:, 0] = r.normal(-1.2, 0.8, size=n)
            Xa_tr = Xa.copy(); Xa_tr[:, 1] += delta          # training absences, displaced
            Xtr = np.vstack([Xp, Xa_tr]); ytr = np.r_[np.ones(n, int), np.zeros(n, int)]

            # honest test set: absences drawn the same way as presences, never displaced
            rt = np.random.default_rng(1000 + s)
            Tp = rt.normal(0, 1, size=(n, d)); Tp[:, 0] = rt.normal(1.2, 0.8, size=n)
            Ta = rt.normal(0, 1, size=(n, d)); Ta[:, 0] = rt.normal(-1.2, 0.8, size=n)
            Xte = np.vstack([Tp, Ta]); yte = np.r_[np.ones(n, int), np.zeros(n, int)]

            m = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=s).fit(Xtr, ytr)
            p = m.predict_proba(Xte)[:, 1]
            naive.append(roc_auc_score(yte, p))
            c = float(np.mean(m.predict_proba(Xp)[:, 1]))     # Elkan-Noto label frequency
            pu.append(roc_auc_score(yte, p / max(c, 1e-6)))
            oc = IsolationForest(n_estimators=300, random_state=s).fit(Xp)
            oneclass.append(roc_auc_score(yte, oc.score_samples(Xte)))
        rows.append(dict(delta=delta, naive=float(np.mean(naive)),
                         pu_corrected=float(np.mean(pu)), presence_only=float(np.mean(oneclass))))
        print(f"  delta={delta:<3} naive {rows[-1]['naive']:.3f}  "
              f"PU {rows[-1]['pu_corrected']:.3f}  presence-only {rows[-1]['presence_only']:.3f}",
              flush=True)
    return pd.DataFrame(rows)
