# Methods primer

Every methodological choice in the paper: what it is, why it was made, what the alternative was, and
what a reviewer is likely to ask. If you can answer these, you can defend the paper.

---

## 1. Why absences are a problem at all

**The situation.** We know where avalanches happened. We do not know where they *didn't* — an
unrecorded slope may simply never have been observed. This is **presence-only data**.

Classification needs two classes, so the analyst invents the second one. Whatever rule they use
decides what question the model ends up answering.

**Likely question — "isn't this just class imbalance?"**
No. Imbalance is about the *ratio* of classes. This is about *where the second class comes from*.
You can have a perfectly balanced 1:1 dataset and still have the problem, which is exactly our case.

---

## 2. Separability (δ)

**What it is.** For each predictor, the difference in means between presences and absences, divided
by their pooled standard deviation — a standardised effect size. Squared, summed across predictors,
square-rooted. One number for how far apart the two clouds sit in predictor space.

**Why this form.** It is the multivariate extension of Cohen's *d*, and it is scale-free, so
predictors in metres and predictors in degrees contribute comparably.

**Likely question — "why not Mahalanobis distance?"**
Mahalanobis accounts for correlation between predictors and would be more principled. We use the
simpler diagonal form because it needs no covariance inversion (unstable with 20+ correlated
predictors and a few thousand points) and because the simulation that calibrates it uses the same
definition. Say this plainly; it is a defensible simplification, not an oversight.

**Likely question — "δ depends on which predictors I include."**
Correct, and stated in the limitations. Two studies with identical absence rules but different
predictor sets get different δ. This is why δ is reported *alongside* the predictor list, not as a
universal constant.

---

## 3. The simulation

**The construction.** Presences sit high on predictor 0, which drives true susceptibility. Absences
sit low on predictor 0 — that is the real, learnable signal. The absences are then displaced along
predictor 1 by δ standard deviations.

**The key design decision:** the displacement is on an axis **orthogonal** to the one carrying
susceptibility. So moving absences along it adds zero information about whether a site is
susceptible. True skill is constant by construction; only the sampling changes.

**The null arm.** Predictors are made independent of the outcome, so true skill is exactly zero. Any
AUC above 0.5 there is unambiguously sampling artefact. This is the arm that carries the argument.

**Likely question — "real predictors are correlated; your simulation uses independent Gaussians."**
True. That is why §4.1 includes the robustness table across 4, 8 and 20 predictors, two learners and
three sample sizes — the curve barely moves. We do not claim the simulation reproduces any real
landscape, only that the inflation mechanism is insensitive to these choices.

**Likely question — "why Random Forest?"**
Because it is what this literature uses. XGBoost is shown alongside and gives the same curve.

---

## 4. Design-adjusted skill

**What it is.** Reported AUC minus the AUC a model with no skill would attain at the same δ.

**Why it is conservative.** It does not claim to recover true accuracy. It removes one identifiable
component — the part attributable to the experimental design — and leaves the rest.

**Likely question — "the baseline comes from a simulation, not from the real data. Isn't that
circular or arbitrary?"**
It is an approximation, and the limitations say so. The alternative — estimating the baseline from
the real data — is impossible, because you would need to know the true labels, which is the thing
we lack. The simulation gives a principled reference point calibrated in the study's own units.

**Likely question — "what if adjusted skill is negative?"**
Then the model performed worse than its sampling design alone, which would indicate a problem with
the model or the predictors. None of our configurations do.

---

## 5. Terrain-matched absences

**What it is.** Background points are drawn to match the presences' joint distribution of
elevation, slope and aspect, rather than at random or from flat land.

**Why aspect specifically.** Sentinel-1 is right-looking, so a single orbit over-detects on slopes
facing the sensor — descending orbit 136 puts 34.8% of detections on SE aspects against 12.6% of
background terrain, and the ascending orbit mirrors it on NW. Matching on aspect prevents the model
scoring on that radar artefact.

**Likely question — "doesn't matching on slope remove the strongest real predictor?"**
Yes, deliberately. Slope is the dominant control on avalanche release, so if background is drawn
from flat ground the model learns "steep vs flat" and nothing else. Matching forces it to
discriminate *within* avalanche-capable terrain, which is the operationally useful question. The
cost is that slope's own importance can no longer be assessed, and we say so.

---

## 6. Spatial-block cross-validation

**What it is.** Folds are whole 10 km blocks, so training and test points are never neighbours.

**Why.** Nearby points share terrain and weather. Random folds let a model memorise a locality and
be tested on the same locality, which measures interpolation rather than prediction.

**Likely question — "10 km is arbitrary."**
It is a choice, informed by the scale over which terrain and snow climate stay similar here. A
variogram-based block size would be more rigorous. Our result is robust to it in the sense that
spatial CV costs only 0.015 AUC either way — notably, this is the correction the field has
standardised on, and it is the *smaller* of the two problems we identify.

---

## 7. Youden's J threshold

**What it is.** The decision threshold maximising sensitivity + specificity − 1, chosen from the
cross-validation rather than fixed at 0.5.

**Why not 0.5.** For a presence-background model fitted at an arbitrary prevalence, 0.5 has no
meaning — it is an artefact of having chosen a 1:1 ratio. Under spatial validation our positives'
scores shift down (mean 0.849 → 0.584) while negatives barely move, so a fixed 0.5 discards 42% of
real sites.

**Likely question — "why not maximise F1?"**
F1 favours recall and would suit an operational early-warning use. Youden is the convention in
susceptibility mapping and weights both error types equally. Both are reported.

---

## 8. The Sentinel-1 inventory (background for the companion paper)

**Detection principle.** Avalanche debris roughens the snow surface, raising radar backscatter. Wet
snow absorbs C-band and *lowers* it — opposite sign, which is how the two are separated.

**Single relative orbit.** Same viewing geometry between dates, so static terrain backscatter
cancels in the difference. Orbit 136 is the only one with 100% coverage here; the orbit with the
most acquisitions covers 13%.

**Recurrence.** Detections are stacked across eight winters. Random change does not repeat on the
same slope; avalanche paths do.

**Melt-out filter.** Snow melting off rough ground also raises backscatter. Requiring MODIS snow
cover to persist after each event removes 72% of verified false positives and none of the verified
avalanches.

**Likely question — "Eckerstorfer reports ~80% precision in Norway; you report 3.4%."**
Different terrain (Himalayan relief causes far more layover and shadow), different revisit (24 days
here after Sentinel-1B's failure, against 6 in Norway), and different validation (they benchmark
against manual SAR interpretation; we verify independently against optical imagery, which is a
stricter test). This belongs in the companion paper and must be addressed there explicitly.

---

## 9. Things to concede immediately if challenged

Do not defend these — agreeing costs nothing and buys credibility:

- One study area, one hazard. Cross-hazard replication is the right next step.
- The simulation is a simplification of real predictor structure.
- Inventory precision is low and measured, not assumed.
- Three confirmed avalanches is a thin basis for the precision estimate.
- The susceptibility map is a demonstration, not an operational product.

A reviewer who sees you concede the real limits will trust the claims you *do* defend.
