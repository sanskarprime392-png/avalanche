# Cover letter — NHESS

**To:** The Editors, Natural Hazards and Earth System Sciences

**Manuscript:** Reported accuracy in geohazard susceptibility mapping is a function of the absence
sampling design

---

Dear Editors,

I am submitting the above manuscript for consideration as a research article in NHESS.

Susceptibility maps for avalanches, landslides and related hazards are built from presence-only
data: the locations of past events are known, but true absences are not. The analyst therefore
constructs the negative class, and the manuscript shows that this construction — rather than the
model — largely determines the area under the ROC curve that studies report and compare.

In synthetic data where absences are displaced from presences along a direction carrying no
information about the outcome, a model with no predictive skill whatsoever reports AUC 0.739 at one
standard deviation of displacement, 0.910 at two, and 1.000 at six. Above roughly three standard
deviations the measure no longer separates a model that works from one that does not. The result is
insensitive to the learner, the number of predictors and the sample size.

Measuring a published avalanche susceptibility protocol on the same scale places its absences 7.06
standard deviations from its presences, where the simulation predicts a null model scores 1.000.
Run on an independent Sentinel-1 inventory of 6,364 candidate release sites in the Western
Himalaya, the protocol returns 1.000. Removing each design's own contribution leaves +0.000 for the
published protocol, +0.027 for terrain-matched absences and +0.108 for aspect-matched absences with
corrected labels — the exact reverse of the ordering by reported AUC.

The manuscript proposes that susceptibility studies report two quantities alongside AUC: the
separability of their absence design, and the skill remaining once that design's own contribution
is removed. Both use only a study's own training data and are inexpensive to compute. Code and data
for both are released.

I would note two things in the interest of transparency. First, the work began as a reproduction of
a recently published avalanche susceptibility study, and that study is cited throughout as the
reference implementation; the manuscript tests a protocol common to this literature rather than the
conduct of any individual study, and is explicit on that point. Second, the manuscript reports two
artefacts found in my own analysis — a predictor whose apparent importance was induced by my own
sampling rule, and a verification pass I had to re-score against an explicit criterion after an
audit — on the grounds that a paper arguing this field does not audit its procedures cannot exempt
itself.

The manuscript has not been published elsewhere and is not under consideration by another journal.
Analysis code was developed with the assistance of a large language model; I designed the study,
directed the analysis, performed the optical verification, and am responsible for all results and
conclusions.

Thank you for considering the manuscript.

Yours sincerely,

**Sanskar Gupta**
School of Computer Science Engineering and Technology
Bennett University, Greater Noida, India
Sanskar.in127@gmail.com

---

## Suggested reviewers

Researchers working on sampling design and validation in susceptibility modelling, or on SAR
avalanche detection. Fill in from the reference list — likely candidates are authors of the
pseudo-absence and spatial-validation literature cited (for example the groups behind
Barbet-Massin et al. 2012, Roberts et al. 2017 or Ploton et al. 2020), and avalanche remote-sensing
groups such as Eckerstorfer's at NORCE.

**Do not suggest** the authors of the reference implementation. Suggesting the people whose protocol
you test reads badly whichever way the review goes.
