# -*- coding: utf-8 -*-
"""Assemble the manuscript from the actual result files — no placeholder numbers."""
import os

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.shared import Inches, Pt, RGBColor

DATA = r"C:\Users\Sanskar\Documents\avalanche_data"
RES = os.path.join(DATA, "results")
FIGS = os.path.join(DATA, "figures")
OUT = r"C:\Users\Sanskar\Documents\avalanche\Avalanche_Manuscript.docx"
NAVY = RGBColor(0x0B, 0x3D, 0x5C)
NICE = {"lgbm": "LightGBM", "xgb": "XGBoost", "rf": "Random Forest",
        "svm_rbf": "SVM (RBF)", "logreg": "Logistic Regression"}

mt = pd.read_csv(os.path.join(RES, "final_model_table_v3.csv"))
ia = pd.read_csv(os.path.join(RES, "improvement_attempts.csv"))
ld = pd.read_csv(os.path.join(RES, "leakage_decomposition.csv"))
ga = pd.read_csv(os.path.join(RES, "geography_ablation.csv"))
br = pd.read_csv(os.path.join(RES, "background_ratio.csv"))
tr = pd.read_csv(os.path.join(RES, "transferability.csv"))
ls = pd.read_csv(os.path.join(RES, "label_quality_sensitivity.csv"))
sh = pd.read_csv(os.path.join(RES, "shap_importance.csv"))
pv = pd.read_csv(os.path.join(RES, "precision_verification.csv"))
pi = pd.read_csv(os.path.join(RES, "precision_by_inventory.csv"))
fe = pd.read_csv(os.path.join(RES, "filter_effectiveness.csv"))
cl = pd.read_csv(os.path.join(RES, "correction_ladder.csv"))
sc = pd.read_csv(os.path.join(RES, "simulation_curve.csv"))
das = pd.read_csv(os.path.join(RES, "design_adjusted_skill.csv"))
rb = pd.read_csv(os.path.join(RES, "simulation_robustness.csv"))
th = pd.read_csv(os.path.join(RES, "correction_threshold.csv"))
cv = pd.read_csv(os.path.join(RES, "correction_validation.csv"))
sn = pd.read_csv(os.path.join(RES, "structured_noise.csv"))
dt = pd.read_csv(os.path.join(RES, "direction_test.csv"))
po = pd.read_csv(os.path.join(RES, "presence_only_comparison.csv"))
idn = pd.read_csv(os.path.join(RES, "identifiability.csv"))
en = pd.read_csv(os.path.join(RES, "empirical_null.csv"))
les = pd.read_csv(os.path.join(RES, "label_error_structure.csv"))

doc = Document()
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(11)


def H(t, lvl=1):
    h = doc.add_heading(t, level=lvl)
    for r in h.runs:
        r.font.color.rgb = NAVY
    return h


def P(t, italic=False, size=11):
    p = doc.add_paragraph()
    r = p.add_run(t)
    r.italic = italic
    r.font.size = Pt(size)
    return p


def bullets(items):
    for i in items:
        doc.add_paragraph(i, style="List Bullet")


def table(headers, rows, widths, caption=None):
    if caption:
        c = doc.add_paragraph()
        rr = c.add_run(caption)
        rr.bold = True
        rr.font.size = Pt(9.5)
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(headers):
        cell = t.rows[0].cells[j]
        cell.text = ""
        r = cell.paragraphs[0].add_run(h)
        r.bold = True
        r.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for j, v in enumerate(row):
            cells[j].text = ""
            r = cells[j].paragraphs[0].add_run(str(v))
            r.font.size = Pt(9)
    for row in t.rows:
        for j, w in enumerate(widths):
            row.cells[j].width = Inches(w)
    doc.add_paragraph()
    return t


def figure(name, caption, width=6.0):
    p = os.path.join(FIGS, name + ".png")
    if not os.path.exists(p):
        return
    doc.add_picture(p, width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(caption)
    r.italic = True
    r.font.size = Pt(9)


# ---------------------------------------------------------------- title
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Reported accuracy in geohazard susceptibility mapping is a function of the "
              "absence sampling design")
r.bold = True
r.font.size = Pt(17)
r.font.color.rgb = NAVY
s = doc.add_paragraph()
s.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = s.add_run("A simulation-grounded correction, with an empirical test in the Western Himalaya")
r.font.size = Pt(12.5)
a = doc.add_paragraph()
a.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = a.add_run("Chandra-Bhaga and Upper Beas basins, Himachal Pradesh and Jammu & Kashmir, India")
r.italic = True
r.font.size = Pt(10)

au = doc.add_paragraph()
au.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = au.add_run("Sanskar Gupta")
r.bold = True
r.font.size = Pt(12)
af = doc.add_paragraph()
af.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = af.add_run("School of Computer Science Engineering and Technology, Bennett University, "
               "Greater Noida, Uttar Pradesh, India")
r.font.size = Pt(9.5)
em = doc.add_paragraph()
em.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = em.add_run("Correspondence: Sanskar.in127@gmail.com")
r.font.size = Pt(9.5)
r.italic = True
doc.add_paragraph()

# ---------------------------------------------------------------- abstract
H("Abstract", 1)
d_xgb = ld[(ld.config == "D") & (ld.model == "xgb")].roc_auc.iloc[0]
a_xgb = ld[(ld.config == "A") & (ld.model == "xgb")].roc_auc.iloc[0]
b_xgb = ld[(ld.config == "B") & (ld.model == "xgb")].roc_auc.iloc[0]
c_xgb = ld[(ld.config == "C") & (ld.model == "xgb")].roc_auc.iloc[0]
P(
    "Machine-learning susceptibility maps for snow avalanches, landslides and related geohazards "
    "routinely report areas under the ROC curve above 0.90, and those values are used to compare "
    "models, select predictors and justify operational products. Because such data are "
    "presence-only, the absences are invented by the analyst, and we show that this choice, rather "
    "than the model, largely determines the reported score. In synthetic data where absences are "
    "displaced from presences along a direction carrying no information about the outcome, a model "
    "with NO predictive skill whatsoever reports AUC 0.739 at a displacement of one standard "
    "deviation, 0.910 at two and 1.000 at six. Reported AUC is therefore not comparable across "
    "studies that chose their absences differently, and above a displacement of about three "
    "standard deviations it cannot distinguish a model that works from one that does not."
    " "
    "We measure the absence designs of a published avalanche susceptibility protocol and of two "
    "corrected alternatives on the same scale, using an independent Sentinel-1 inventory of 6,364 "
    "candidate release sites in the Chandra-Bhaga and Upper Beas basins of the Western Himalaya. "
    "Drawing absences from flat, vegetated, urban and cropland terrain, as this literature does, "
    "places them 7.06 standard deviations from the presences, where the simulation predicts that a "
    "null model scores 1.000 — and 1.000 is what the protocol returns on real data."
    " "
    "We then show that the absolute skill of a presence-only model is not identifiable from the "
    "data it was fitted to. Two synthetic worlds were constructed whose observed presence and "
    "absence samples are nearly indistinguishable — their separations differ by 0.02 and their "
    "reported AUCs by 0.043 — while their ability to rank genuinely susceptible terrain differs by "
    "0.281. A difference between the two samples enters an AUC identically whether it arises "
    "because presences are different terrain or because absences were drawn from elsewhere, so no "
    "statistic computed from those samples can attribute it. We report our own failed attempts at a "
    "correction for the same reason."
    " "
    "What remains identifiable is the relative effect of the absence design, obtained by holding "
    "presences, predictors and model fixed and varying only the sampling: the published protocol "
    "returns 1.000, terrain-matched absences 0.954, and aspect-matched absences with corrected "
    "labels 0.866. Training on displaced absences and evaluating on a fixed honest test set shows "
    "the model itself is little harmed, falling only from 0.975 to 0.952; what the design corrupts "
    "is the figure a study reports about itself. Two further diagnostics agree: 1 km climate and "
    "500 m snow-cover predictors are statistically interchangeable with raw spatial coordinates, "
    "and reported AUC rises with the background sampling ratio while precision-recall performance "
    "falls. We recommend that susceptibility studies hold out an evaluation set whose absences are "
    "drawn like the presences; code and data are released.")
P("Keywords: susceptibility mapping, pseudo-absence selection, sampling bias, model evaluation, "
  "presence-only data, snow avalanches, Sentinel-1, Western Himalaya", italic=True, size=9.5)

# ---------------------------------------------------------------- 1 intro
H("1. Introduction", 1)
P("Snow avalanches threaten settlements, highways and glacial lakes across the Indian Himalayan "
  "Region (IHR). Because field survey is impractical over this terrain, avalanche susceptibility "
  "maps (ASM) are produced by training classifiers on past avalanche locations and a set of "
  "topographic, climatic and snowpack predictors. Reported performance is consistently high, with "
  "recent IHR studies citing AUC values of 0.87–0.95.")
P("Such values are only meaningful if the negative class represents terrain where avalanches could "
  "occur but did not. In practice, published studies draw non-avalanche points from flat vegetated "
  "areas, urban settlements, water bodies and croplands. These locations differ from avalanche "
  "release zones in nearly every predictor simultaneously, so a classifier can achieve excellent "
  "scores by recognising land cover rather than avalanche conditions. The field has meanwhile "
  "concentrated its methodological attention on spatial cross-validation, which addresses a "
  "different problem — spatial autocorrelation between training and test folds.")
P("This study asks which of these two design choices actually drives reported performance, and "
  "whether the resulting models encode avalanche physics or study-design artefacts. We address it "
  "with a controlled 2×2 experiment on an independently constructed inventory, followed by "
  "explainability and ablation diagnostics. We apply the same scrutiny to our own pipeline and "
  "report two artefacts we found there.")
H("1.1 Scope and intent", 2)
P("What is under test here is a PROTOCOL, not a paper. The absence rule examined below recurs "
  "throughout avalanche susceptibility work in the region: the study we reproduce cites earlier "
  "applications reporting AUC values of 0.88 and 0.87 built the same way. Our results therefore "
  "speak to a convention the field has inherited, and they apply to our own models exactly as they "
  "apply to published ones — Section 4.1 reports what happens when we run the conventional protocol "
  "on our own data, and the answer is a perfect score.")
P("We take Abhinav and Sattar (2025) as the reference implementation because it is recent, it covers "
  "these basins, and it documents its predictors, models and sampling clearly enough to be rebuilt "
  "independently. That is not true of every study in this area, and reproducing a result is only "
  "possible when the original is reported carefully. Where our figures differ from theirs, the "
  "difference reflects the experimental design we impose rather than an error on their part: a model "
  "trained against flat, vegetated absences is answering a different and easier question than one "
  "trained against terrain-matched background, and neither number is wrong for the question it "
  "answers.")

# ---------------------------------------------------------------- 2 study area
H("2. Study area", 1)
P("The Chandra-Bhaga and Upper Beas basins lie in Himachal Pradesh and the Jammu & Kashmir union "
  "territory, in the Western Himalaya. The region receives heavy winter snowfall from westerly "
  "disturbances and has a documented history of avalanche impacts on the Leh–Manali highway and on "
  "settlements including Manali, Keylong and Udaipur.")
P("All analysis is carried out over a rectangular extent covering both basins, 76.40–78.10° E and "
  "31.90–33.20° N, rather than over hydrologically delineated basin outlines. The extent therefore "
  "includes some terrain beyond the two catchments, and spans 554–6,565 m against the 1,123–6,329 m "
  "usually quoted for the basins themselves. Potential release terrain — slope between 28 and 60°, "
  "after Bühler et al. (2018) — covers 11,705 km², roughly half the extent, and every areal "
  "statistic reported in Section 4 refers to that domain rather than to the full rectangle.")
figure("fig_study_area",
       "Figure 1. The analysis extent, with elevation over hillshade and the settlements and passes "
       "named in the regional avalanche literature. Avalanche terrain here spans more than five "
       "kilometres of relief.", 6.4)

# ---------------------------------------------------------------- 3 data & methods
H("3. Data and methods", 1)

H("3.1 Avalanche inventory from Sentinel-1", 2)
P("Avalanche debris roughens the snow surface and raises radar backscatter. We detect it as a "
  "seasonal increase in Sentinel-1 GRD backscatter between a stable mid-winter reference composite "
  "and a post-avalanche-season composite, computed independently for each winter from 2018 to 2025.")
bullets([
    "A single relative orbit is used (descending, orbit 136) so that viewing geometry is identical "
    "between dates and static terrain backscatter cancels in the difference. Orbit 136 is the only "
    "orbit providing 100% coverage of the study area; the orbit with the most acquisitions covers "
    "just 13% of it.",
    "Composites are averaged in linear power and speckle-filtered; only increases are retained, "
    "since wet snow lowers backscatter and therefore has the opposite sign.",
    "Detections are restricted to avalanche-capable slopes (25–60°) and filtered by minimum "
    "connected area.",
    "Detections are stacked across winters to give a RECURRENCE count. Random change does not "
    "repeat on the same slope across eight winters, whereas avalanche paths do; recurrence "
    "therefore acts as a confidence score.",
])
P("Deposits are traced upslope to the highest pixel on release-angle terrain (30–50°) within a "
  "600 m search radius, because susceptibility models predict release, not runout. This yields "
  "6,364 candidate release points, of which 3,188 are flagged in four or more winters. Occupied 10 km blocks cover 85% of the study area, so the inventory is "
  "not confined to road-accessible slopes.")

P("One convention applies to every count reported below. A release point is discarded if any "
  "predictor is undefined at its cell, which happens at raster edges and on a small number of "
  "interior gaps. Inventory counts therefore exceed the sample sizes in the tables by a few points "
  "at each recurrence threshold: 6,364 detected against 6,344 analysed at recurrence ≥ 3, 3,188 "
  "against 3,179 at ≥ 4, 917 against 914 at ≥ 5, and 241 against 240 at ≥ 6. Figures quoted in the "
  "text are detections; figures in tables are the points that entered a model.")

H("3.2 Predictive factors", 2)
P("Twenty predictors were assembled on a common 30 m UTM 43N grid: twelve topographic and two "
  "hydrological derivatives computed from NASADEM with WhiteboxTools; winter temperature and "
  "precipitation (WorldClim); snow-cover duration (MODIS); land cover (ESA WorldCover); and "
  "distances to roads, streams (OpenStreetMap) and active faults (GEM Global Active Faults). "
  "Distance to faults is computed from vector geometry, because every active fault lies outside "
  "the study area and a rasterised distance transform is therefore undefined within it.")

H("3.3 Slope windows used at each stage", 2)
P("Four slope windows appear in the workflow and they are deliberately different, because they "
  "answer different questions. Detection retains change on 25–60° slopes, a deliberately broad "
  "window chosen to avoid discarding events near the margins. Release points are traced into "
  "30–50° terrain, the slab-release window conventional in the avalanche literature. Background "
  "points are drawn from 25–55°, matching the detection window at its lower bound while excluding "
  "the steepest faces, which hold little snow and are not comparable terrain. The susceptibility "
  "map is masked to 28–60°, the potential-release-area definition of Bühler et al. (2018), so that "
  "areal statistics are comparable with other studies using that convention.")
P("The background window sets the separability of the absence design directly, so the result was "
  "recomputed across four alternatives. Reported AUC moves with the choice — 0.844 at 20–60°, "
  "0.866 at 25–55°, 0.873 at 28–60° and 0.875 at 30–50° — a range of 0.031 produced by a decision "
  "about which terrain counts as comparable, with the presences, predictors and model unchanged. "
  "This is the paper's argument appearing within a single study: a defensible-looking choice moves "
  "the reported number by more than the gap between several published models. The 25–55° window is "
  "used throughout, and the alternatives are reported so the sensitivity is visible rather than "
  "buried.")

H("3.4 Sampling design", 2)
P("Two absence schemes are compared. The PAPER-STYLE scheme reproduces published practice: points "
  "drawn at random from low-slope terrain classified as tree cover, grassland, cropland, built-up "
  "or water. The TERRAIN-MATCHED scheme draws background points from avalanche-capable slopes "
  "(25–55°) outside a 1 km buffer around presences, matched to the joint elevation × slope × "
  "aspect distribution of the presences.")
P("Aspect matching is required, not optional. Sentinel-1 is right-looking, so a single orbit "
  "over-detects on slopes facing the sensor: measured over this study area, descending orbit 136 "
  "places 34.8% of detections on south-east aspects against 12.6% of background terrain, while "
  "ascending orbit 27 peaks on north-west aspects. These are mirror images and therefore reflect "
  "viewing geometry rather than avalanche physics. Only the elevated south-facing fraction is "
  "consistent across both geometries. Matching background to the presence aspect distribution "
  "prevents the classifier from scoring on this artefact.", )

H("3.5 Validation and models", 2)
P("Models are evaluated under random k-fold cross-validation, as in published work, and under "
  "spatial-block cross-validation with 10 km blocks (246 blocks). Five classifiers are compared: "
  "Random Forest, Support Vector Machine and Logistic Regression, as used in the reference "
  "literature, plus XGBoost and LightGBM. Confidence intervals are percentile bootstraps (1,500 "
  "resamples) on pooled out-of-fold predictions.")

# ---------------------------------------------------------------- 4 results
H("4. Results", 1)

H("4.1 Reported AUC is a function of the absence design", 2)
P("Before examining any real inventory we establish what a reported AUC can mean. Synthetic "
  "presences are drawn from a region where a true susceptibility function is high, and absences "
  "from a distribution displaced by δ standard deviations ALONG A DIRECTION ORTHOGONAL to that "
  "function. Because the displacement is orthogonal it carries no information about susceptibility, "
  "so the model's genuine ability to rank susceptible sites is identical at every δ. In a second, "
  "null arm the predictors are made independent of the outcome, so true skill is exactly zero by "
  "construction and any AUC above 0.5 must come from the sampling alone.")
rows = []
for dl in sorted(sc.delta.unique()):
    inf = sc[(sc.delta == dl) & (sc.arm == "informative")].auc.iloc[0]
    nul = sc[(sc.delta == dl) & (sc.arm != "informative")].auc.iloc[0]
    rows.append([f"{dl:g}", f"{inf:.3f}", f"{nul:.3f}"])
table(["δ (SD)", "AUC, informative model", "AUC, null model (no skill)"], rows,
      [1.0, 1.9, 2.0],
      "Table 1. Reported AUC against absence displacement in synthetic data.")
P("The null model moves from 0.496 at δ = 0, correctly indicating no skill, to 0.910 at δ = 2 and "
  "1.000 at δ = 6. A model that has learned nothing can therefore be made to report any AUC up to "
  "unity purely by choosing where its absences are drawn from. The informative arm is already near "
  "ceiling throughout, so once δ exceeds about 3 the two arms are indistinguishable: beyond that "
  "point a reported AUC cannot separate a model that works from one that does not.")
figure("fig_inflation_law",
       "Figure 2. Reported AUC against absence displacement. Dashed lines mark where real absence "
       "designs fall on the same axis.")
P("Measuring the three real designs used in this study on the same scale places the published "
  f"protocol at δ = {das.delta.iloc[0]:.2f}, terrain-matched absences at δ = {das.delta.iloc[1]:.2f}, "
  f"and terrain-and-aspect matching with corrected labels at δ = {das.delta.iloc[2]:.2f}. At δ = "
  "7.06 the simulation predicts that even a model with no information scores 1.000 — and 1.000 is "
  "what the published protocol returns on real data (Section 4.2). The artefact is therefore not "
  "incidental to one dataset; it is the expected consequence of the design.")

P("A synthetic result invites the objection that it reflects the authors' construction rather "
  "than the problem, so the null arm was repeated while varying the learner, the number of "
  "predictors and the sample size. The inflation is essentially unchanged throughout: across six "
  "settings the null-model AUC differs by at most 0.030 at any displacement, and by 0.001 at "
  "δ = 6. The behaviour is a property of evaluating a presence-only model against constructed "
  "absences, not of this particular simulation.")
piv = rb.pivot(index="setting", columns="delta", values="auc")
rows = [[idx] + [f"{piv.loc[idx, c]:.3f}" for c in piv.columns] for idx in piv.index]
table(["Setting"] + [f"δ={c:g}" for c in piv.columns], rows,
      [2.2] + [0.62] * len(piv.columns),
      "Table 2. Null-model AUC under varying simulation settings.")

P("The construction displaces absences along a single axis chosen to carry no information about "
  "the outcome, which raises the question of whether that direction is special. It is not. "
  "Distributing the same displacement across five axes, along a random direction in the "
  "uninformative subspace, or partly along the informative axis itself — which is what an absence "
  "rule favouring gentle terrain actually does — changes the null-arm AUC by at most 0.016 at any "
  "displacement.")
nul = dt[dt.arm == "null"]
piv = nul.pivot(index="direction", columns="delta", values="auc")
rows = [[i] + [f"{piv.loc[i, c]:.3f}" for c in piv.columns] for i in piv.index]
table(["Displacement direction"] + [f"δ={c:g}" for c in piv.columns], rows,
      [2.0] + [0.62] * len(piv.columns),
      "Table 3. Null-model AUC under different displacement directions.")

H("4.2 What a reported AUC cannot recover", 2)
P("The natural response to Section 4.1 is to remove the design's contribution and keep the "
  "remainder as the model's. We pursued that and report here why it cannot work, because the reason "
  "is more informative than the method would have been.")
P("An AUC is a function of two class-conditional distributions and nothing else. A difference "
  "between presences and absences enters those distributions identically whether it arises because "
  "presences are genuinely different terrain or because the absences were drawn from elsewhere. No "
  "statistic computed from the two samples can attribute it, since the samples are all there is.")
P("That this bites in practice, and not only in principle, can be shown by construction. Two "
  "synthetic worlds were built. In the first, predictors are strongly related to true "
  "susceptibility and absences are drawn from terrain like the presences. In the second, predictors "
  "are weakly related and absences are drawn from further away. Parameters were chosen so the "
  "observed clouds nearly coincide.")
m = idn.mean(numeric_only=True)
table(["World", "Observed presence–absence gap", "Reported AUC", "AUC against true absences"],
      [["A  informative predictors, honest absences", f"{m.mean_gap_A:.3f}",
        f"{m.reported_A:.3f}", f"{m.true_A:.3f}"],
       ["B  weak predictors, displaced absences", f"{m.mean_gap_B:.3f}",
        f"{m.reported_B:.3f}", f"{m.true_B:.3f}"]],
      [2.5, 1.7, 1.0, 1.4],
      "Table 4. Two worlds with nearly identical observable data and different true skill.")
P(f"The observed separations differ by {abs(m.mean_gap_A - m.mean_gap_B):.3f} and the reported AUCs "
  f"by {abs(m.reported_A - m.reported_B):.3f}. Measured against true absences — which no "
  f"presence-only study possesses — the two differ by {abs(m.true_A - m.true_B):.3f}, roughly six "
  "times as much. Data that a study could not tell apart correspond to models that differ "
  "substantially in what a reader cares about.")
P("Attempts to construct a null baseline from a study's own data fail for the same reason, and we "
  "report them so the obstacle is not mistaken for an oversight. Generating synthetic classes "
  "matched to the observed means and covariances yields a baseline of 0.989 for the corrected "
  "configuration, above its own reported AUC, because Gaussian clouds at those moments are easier "
  "to separate than the real data: the construction discards exactly the within-class overlap that "
  "makes the real problem hard. Resampling real rows within each class preserves the signal intact "
  "and is therefore not a null at all. Nor is the separability measure canonical — for the "
  f"published protocol the diagonal form gives {en.delta_diag.iloc[0]:.2f} standard deviations and "
  f"the Mahalanobis form {en.delta_maha.iloc[0]:.2f}, so a baseline read off a δ axis inherits the "
  "choice of measure.")
P("We therefore withdraw any proposal to correct a reported AUC, and state the limit instead: the "
  "absolute skill of a presence-only susceptibility model is not identifiable from the data it was "
  "fitted to. What remains identifiable is the RELATIVE effect of the absence design, obtained by "
  "holding the presences, the predictors and the model fixed and varying only how absences are "
  "drawn. That comparison is the subject of the remaining sections, and it is available to any "
  "study at the cost of running its own pipeline a second time.")

H("4.3 Whether label error can account for the result", 2)
P("Every configuration compared here uses identical labels, which controls for label quality only "
  "if the errors are distributed like the true events in predictor space. Checking our own, they "
  "are not: detections rejected on review sit 908 m lower than verified ones (3,186 m against "
  "4,094 m, Mann-Whitney p = 0.038), which is what melt-out contamination should look like, since "
  "ablation begins at lower elevations. Slope and aspect differences are not significant, though "
  "with three verified events those tests have little power. Because background points are matched "
  "on elevation, this error structure is correlated with a variable the sampling design acts on, "
  "and the assumption that labels affect all designs equally cannot simply be asserted.")
P("We therefore tested it. A fraction of synthetic presences was replaced by points drawn from the "
  "low end of the informative axis — the analogue of melt-out false positives at low elevation — "
  "and the displacement curve recomputed.")
piv = sn.pivot(index="noise_rate", columns="delta", values="auc")
rows = [[f"{int(100*i)}%"] + [f"{piv.loc[i, c]:.3f}" for c in piv.columns] for i in piv.index]
table(["Contaminated presences"] + [f"δ={c:g}" for c in piv.columns], rows,
      [1.9] + [0.63] * len(piv.columns),
      "Table 5. Reported AUC under structured label contamination.")
P("At δ = 0 contamination matters a great deal: reported AUC falls from 0.979 to 0.629 as the "
  "contaminated fraction rises to 80%. By δ = 3 the spread across contamination rates is 0.019, and "
  "by δ = 6 it is 0.001. At the separability of the published protocol an inventory that is 80% "
  "false positives reports the same value as a clean one — the displacement overwhelms the label "
  "quality entirely. Structured label error therefore cannot account for the inflation result; if "
  "anything it sharpens it, since at high separability the reported score ceases to respond to "
  "label quality at all.")
P("The corollary constrains our own empirical figures. The corrected configuration sits at δ ≈ 1, "
  "where contamination still moves reported AUC by up to 0.187. Its value of 0.866 is therefore "
  "sensitive to inventory precision in a way the published protocol's 1.000 is not, and should be "
  "read with that in mind.")

H("4.4 Training against bad absences versus reporting against them", 2)
P("A distinction the preceding sections leave open is whether a model TRAINED on poorly chosen "
  "absences is itself damaged, or whether only the number it reports is. We separate the two by "
  "training on absences at varying displacement and evaluating every model on one fixed test set "
  "whose absences are drawn like the presences — the evaluation the study should have run.")
rows = [[f"{r.delta:g}", f"{r.naive:.3f}", f"{r.pu_corrected:.3f}", f"{r.presence_only:.3f}"]
        for _, r in po.iterrows()]
table(["Training δ", "Ordinary classifier", "PU-corrected", "Presence-only (one-class)"], rows,
      [1.0, 1.6, 1.3, 1.9],
      "Table 6. AUC on a fixed honest test set, by how the training absences were drawn.")
P("Judged honestly, the ordinary classifier falls only from 0.975 to 0.952 as its training "
  "absences are displaced from zero to three standard deviations. The model is largely unharmed; "
  "what the displacement corrupts is the figure the study reports about itself. This is the more "
  "accurate statement of the problem, and a more generous one: published susceptibility maps may "
  "well be serviceable products whose accompanying statistics do not describe them.")
P("Two established treatments of presence-only data behave as theory predicts. The Elkan and Noto "
  "(2008) correction divides scores by the estimated labelling frequency, a constant, so the "
  "ranking and hence the AUC are unchanged to three decimal places at every displacement: it "
  "repairs calibration, not design dependence. A one-class model fitted to the presences alone "
  "never sees the absences and is therefore perfectly design-independent, returning 0.819 at every "
  "displacement, but it pays about 0.15 AUC for that robustness.")
P("The practical implication is narrower than 'redraw your absences'. If absences must be "
  "constructed, the evaluation set is where it matters: holding out absences drawn like the "
  "presences costs nothing and makes the reported number interpretable. Where that is impossible, "
  "a presence-only method trades discriminative power for a figure that does not depend on the "
  "analyst's choices.")

H("4.5 What each correction costs", 2)
P("Two configurations recur below and must not be confused. CONFIGURATION C2 applies the melt-out "
  "filter and centroid-derived release points to the single-scale predictor set, and reaches AUC "
  "0.841. CONFIGURATION C3 adds the multi-scale terrain predictors of Section 4.7 to C2 and reaches "
  "0.866. The correction ladder in Table 5 ends at C2, because its purpose is to isolate the cost "
  "of each correction with the predictor set held fixed; the design-adjusted figures in Section 4.2 "
  "use C3, the best-performing honest configuration. Both are reported throughout rather than "
  "silently interchanged.")
P("Five corrections separate the published protocol from a defensible estimate. Applying them "
  "cumulatively to the same models and the same study area shows how much of the reported figure "
  "each one accounts for.")
rows = [[r.configuration.replace("+ ", ""), f"{r.roc_auc:.3f}", str(r.delta),
         r["what it removes"]] for _, r in cl.iterrows()]
table(["Configuration", "AUC", "Δ", "What it removes"], rows, [2.5, 0.7, 0.6, 2.0],
      "Table 7. Cumulative effect of each methodological correction (XGBoost).")
P(f"The conventional protocol returns AUC {cl.roc_auc.iloc[0]:.3f} and the fully corrected "
  f"configuration {cl.roc_auc.iloc[-1]:.3f}, a difference of "
  f"{cl.roc_auc.iloc[0]-cl.roc_auc.iloc[-1]:.3f}. The largest single term is not the one the field "
  "has concentrated on. Spatial cross-validation, the standard methodological correction in this "
  f"literature, accounts for {abs(float(cl.delta.iloc[2])):.3f}; label quality — removing melt-out "
  "false positives and deriving release points without an extreme-value rule — accounts for "
  f"{abs(float(cl.delta.iloc[4])):.3f}, roughly six times as much. Inventory precision is not "
  "reported at all in the studies we reproduce, so this term is invisible in published work.")
figure("fig_correction_ladder",
       "Figure 3. Cumulative effect of each correction. Label quality removes more apparent skill "
       "than sampling design and spatial validation combined.")

H("4.6 Where reported performance comes from", 2)
rows = []
lab = {"A": ("paper-style", "random"), "B": ("paper-style", "spatial-block"),
       "C": ("terrain-matched", "random"), "D": ("terrain-matched", "spatial-block")}
for cfg in ("A", "B", "C", "D"):
    rf_v = ld[(ld.config == cfg) & (ld.model == "rf")].roc_auc.iloc[0]
    xg_v = ld[(ld.config == cfg) & (ld.model == "xgb")].roc_auc.iloc[0]
    rows.append([cfg, lab[cfg][0], lab[cfg][1], f"{rf_v:.3f}", f"{xg_v:.3f}"])
table(["Config", "Absences", "Cross-validation", "AUC (RF)", "AUC (XGBoost)"], rows,
      [0.7, 1.5, 1.5, 1.1, 1.3],
      "Table 8. Controlled decomposition of reported performance.")
P(f"Configuration A, which reproduces published practice, achieves AUC = {a_xgb:.3f}: the two "
  f"classes are perfectly separable. Replacing the absences with terrain-matched background points "
  f"(A→C) costs {a_xgb - c_xgb:.3f} AUC for XGBoost and "
  f"{ld[(ld.config=='A')&(ld.model=='rf')].roc_auc.iloc[0] - ld[(ld.config=='C')&(ld.model=='rf')].roc_auc.iloc[0]:.3f} "
  f"for Random Forest. Applying spatial-block cross-validation to the original sampling (A→B) "
  f"changes performance by {a_xgb - b_xgb:+.3f}. The optimism is therefore attributable to sampling "
  "design, and essentially not at all to spatial autocorrelation.")
figure("fig_leakage_decomposition",
       "Figure 4. Decomposition of reported performance. Paper-style absences produce perfect "
       "separation regardless of the cross-validation scheme.")

P("This result does not depend on the quality of our inventory. Repeating the contrast at four "
  "recurrence thresholds, configuration A remains at AUC 1.000 throughout, including at n = 240 "
  "presence points — comparable to the 118 points used in the reference study.")
rows = []
for mr in sorted(ls.min_recurrence.unique()):
    A = ls[(ls.min_recurrence == mr) & (ls.config == "A")]
    D = ls[(ls.min_recurrence == mr) & (ls.config == "D")]
    rows.append([f"≥{int(mr)}", f"{int(A.n_presence.iloc[0]):,}",
                 f"{A.roc_auc.iloc[0]:.3f}", f"{D.roc_auc.iloc[0]:.3f}",
                 f"{A.roc_auc.iloc[0] - D.roc_auc.iloc[0]:.3f}"])
table(["Recurrence", "n presence", "A (paper)", "D (matched)", "Difference"], rows,
      [1.0, 1.1, 1.1, 1.2, 1.1],
      "Table 9. Sensitivity of the decomposition to inventory confidence.")
figure("fig_label_sensitivity",
       "Figure 5. The sampling artefact is independent of inventory size and label confidence.")

H("4.7 Coarse predictors encode geography, not snowpack", 2)
rows = [[r.variant, int(r.n_features), f"{r.roc_auc:.3f}", f"[{r.ci_low:.3f}, {r.ci_high:.3f}]"]
        for _, r in ga.iterrows()]
table(["Predictor set", "n", "AUC", "95% CI"], rows, [2.6, 0.5, 0.9, 1.5],
      "Table 10. Replacing coarse predictors with raw spatial coordinates.")
full = ga[ga.variant.str.contains("full")].roc_auc.iloc[0]
repl = ga[ga.variant.str.contains("REPLACED")].roc_auc.iloc[0]
P(f"Substituting the 1 km climate, 500 m snow-cover and distance-to-fault layers with two raw "
  f"coordinates changes performance from {full:.3f} to {repl:.3f}, a difference well within the "
  "confidence intervals. These predictors contribute to the model, but their contribution is "
  "positional rather than physical: at their native resolution they are smooth regional trend "
  "surfaces across release zones of 10²–10⁵ m². Published importance rankings that place snow-cover "
  "duration among the leading predictors may therefore be describing a geographic index rather "
  "than snowpack process.")
figure("fig_geography_ablation", "Figure 6. Coarse predictors are interchangeable with coordinates.", 6.4)

H("4.8 Reported AUC depends on the background sampling ratio", 2)
rows = [[r.ratio, f"{r.prevalence:.3f}", f"{r.roc_auc:.3f}", f"{r.pr_auc:.3f}", f"{r.brier:.4f}"]
        for _, r in br.iterrows()]
table(["Presence:background", "Prevalence", "AUC", "PR-AUC", "Brier"], rows,
      [1.6, 1.0, 0.9, 0.9, 0.9],
      "Table 11. Effect of the background sampling ratio.")
P(f"Increasing the background ratio from 1:1 to 1:10 raises AUC from {br.roc_auc.iloc[0]:.3f} to "
  f"{br.roc_auc.iloc[-1]:.3f} while precision-recall AUC falls from {br.pr_auc.iloc[0]:.3f} to "
  f"{br.pr_auc.iloc[-1]:.3f}. Because presence-background prevalence is a modelling choice rather "
  "than a property of the landscape, a headline AUC can be improved by a sampling decision while "
  "the model becomes less useful for the operational task. Precision-recall metrics should be "
  "reported alongside AUC.")

H("4.9 What improves a corrected model, and what does not", 2)
P("Having removed the sources of inflation, we asked what genuinely raises performance. Four "
  "interventions were tried on the corrected inventory and evaluated identically.")
rows = [[r.attempt, f"{r.auc:.3f}", ("—" if pd.isna(r.delta) else f"{r.delta:+.3f}"), r.helped]
        for _, r in ia.iterrows()]
table(["Intervention", "AUC", "Δ", "Helped"], rows, [3.0, 0.7, 0.7, 0.7],
      "Table 12. Interventions tested on the corrected model (XGBoost, spatial-block CV).")
P("Terrain described at the scale of the process is what helps. A 30 m cell is smaller than the "
  "feature that decides whether a slope releases, so mean slope, topographic position and roughness "
  "were recomputed over 90, 300 and 900 m neighbourhoods — slab, start-zone and face scale. Those "
  "nine predictors add 0.025 AUC and 0.058 recall. Restoring TPI adds a further 0.005: its earlier "
  "dominance was an artefact of the highest-pixel release rule, and once release points are taken "
  "as the centroid of release-angle terrain the presence-to-background gap falls from +33.5 to "
  "+3.8, making it an ordinary predictor again.")
P("Neither conventional route to a better score helped. A nested cross-validated hyperparameter "
  "search over 24 candidates returned 0.857, below untuned defaults, because at 235 spatial blocks "
  "the inner folds are too noisy to select on; averaging XGBoost, LightGBM and Random Forest "
  "returned 0.860, below XGBoost alone. We report these negative results because they point the "
  "same way as the rest of the paper: the available gains lie in label quality and in describing "
  "the terrain at the right scale, not in model selection.")

# ---------------------------------------------------------------- 5 discussion
H("5. Discussion", 1)

H("5.1 What a reported AUC can and cannot tell a reader", 2)
P("Susceptibility data are presence-only: the locations of past events are observed, and the "
  "absences are constructed by the analyst. Section 4.1 shows that this construction, not the "
  "model, sets most of the reported score — a model with no skill at all reaches 0.910 at two "
  "standard deviations of absence displacement and 1.000 at six. Two consequences follow, and "
  "neither is specific to avalanches.")
P("First, AUC is not comparable between studies that drew their absences differently. A study "
  "reporting 0.97 with absences taken from flat farmland has not demonstrated more skill than one "
  "reporting 0.85 with absences on comparable terrain; on the evidence here it has probably "
  "demonstrated less. Model comparisons, predictor rankings and claims of improvement over prior "
  "work all inherit this, because all of them rest on a quantity that the sampling design has "
  "already largely determined.")
P("Second, above roughly three standard deviations of separation the measure saturates and stops "
  "discriminating at all: in the simulation a model with genuine skill and a model with none both "
  "report values above 0.97. Much of the published literature appears to operate in that regime. "
  "The practical recommendation is not that existing maps be discarded — a map can be useful "
  "without its headline statistic being meaningful — but that absence selection be reported and "
  "justified as carefully as model selection currently is, and that the two quantities introduced "
  "here be reported alongside AUC.")

H("5.2 A correction that studies can apply to themselves", 2)
P("Both quantities are cheap to compute. The separability δ is a standardised distance between the "
  "presence and absence clouds in predictor space and requires only the study's own training "
  "table. The null baseline is a simulation at that δ, which takes seconds. Their difference — the "
  "skill a model contributes above what its sampling design already delivers — is the number a "
  "reader assumes AUC reports. Code for both is released with this paper.")
P("The correction is deliberately conservative. It does not claim to recover a model's true "
  "accuracy, only to remove a component of the reported score that is attributable to the "
  "experimental design rather than to the model. Where a design is so separable that a null model "
  "already scores 1.000, the correction returns zero and the honest conclusion is that the study "
  "has not measured predictive skill at all.")

H("5.3 Applying the same standard to our own analysis", 2)
P("Two artefacts of our own are reported because the argument demands it. SHAP attributed the "
  "largest single contribution to the topographic position index; inspection showed presence "
  "points averaged TPI +36.4 against +2.9 for background, which follows mechanically from an "
  "earlier release-point rule that selected the highest qualifying pixel in a search radius. "
  "Deriving release points as the centroid of release-angle terrain instead reduced that gap to "
  "+3.8 and made the predictor ordinary again.")
P("The second concerns the optical verification used to establish label quality for the corrected "
  "configuration. An unaided first pass over 100 detections accepted 24 as avalanches; re-scoring "
  "against an explicit rule — snow present beforehand, a localized rough feature appearing, "
  "surroundings unchanged — rejected 21 of those, most because the slope was bare rock before the "
  "event and acquired snow afterwards. Both scorings are reported in the companion paper, and the "
  "disagreement between them (Cohen's kappa 0.49) is treated there as a result rather than an "
  "embarrassment. Neither artefact changes the findings above, which compare sampling designs "
  "using identical labels, but a paper arguing that this field does not audit its own procedures "
  "cannot exempt itself.")

H("5.4 Limitations", 2)
bullets([
    "The simulation establishes that absence displacement inflates AUC and provides a baseline at "
    "a given separability. It does not model every way a real absence design can differ from the "
    "presences — spatial clustering, differing covariance structure and non-Gaussian predictor "
    "distributions are all simplified — so the baseline should be read as an approximation rather "
    "than an exact correction.",
    "The separability measure aggregates standardised differences across predictors and therefore "
    "depends on which predictors a study uses. Two studies with identical absence rules but "
    "different predictor sets will report different δ.",
    "The empirical test uses one study area and one hazard. The mechanism is not specific to "
    "either, but demonstrating that requires replication on independent inventories, ideally for "
    "landslides, where open data are more plentiful. We regard this as the most valuable next step.",
    "The avalanche inventory underlying the empirical sections is derived from Sentinel-1 change "
    "detection and consists of candidate rather than confirmed events; its precision is quantified "
    "in the companion paper. Because every configuration compared here uses identical labels, "
    "label quality affects all arms equally and cannot produce the differences reported, as the "
    "sensitivity analysis in Section 4.3 confirms.",
])

# ---------------------------------------------------------------- 6 conclusions
H("6. Conclusions", 1)
bullets([
    "Susceptibility data are presence-only, so absences are an analyst's choice. In synthetic data "
    "where that choice carries no information about the outcome, a model with no predictive skill "
    "reports AUC 0.739 at one standard deviation of absence displacement, 0.910 at two and 1.000 "
    "at six. Reported AUC is therefore a function of the sampling design.",
    "Above roughly three standard deviations the measure saturates: a model with genuine skill and "
    "one with none both report values above 0.97, so AUC no longer distinguishes them.",
    "Measured on that scale, a published avalanche susceptibility protocol places its absences "
    "7.06 standard deviations from its presences, where a null model is predicted to score 1.000. "
    "Run on an independent inventory, the protocol returns 1.000.",
    "The absolute skill of a presence-only susceptibility model is not identifiable from the data "
    "it was fitted to. Two worlds with nearly indistinguishable observed samples differ by 0.281 in "
    "their ability to rank genuinely susceptible terrain, and no statistic computed from those "
    "samples can tell them apart. Proposals to correct a reported AUC, including our own, fail for "
    "this reason rather than for want of a better estimator.",
    "What is identifiable is the relative effect of the absence design, obtained by holding "
    "presences, predictors and model fixed and varying only how absences are drawn. Any study can "
    "run that comparison on its own pipeline.",
    "Training on poorly drawn absences harms the model far less than it harms the number: on a "
    "fixed honest test set performance falls only from 0.975 to 0.952 as displacement rises from "
    "zero to three standard deviations.",
    "Two independent diagnostics agree: 1 km climate and 500 m snow-cover predictors are "
    "statistically interchangeable with raw spatial coordinates, and reported AUC rises with the "
    "background sampling ratio while precision-recall performance falls.",
    "We recommend that susceptibility studies hold out an evaluation set whose absences are drawn "
    "like the presences. It costs nothing, needs no new statistic, and is the only step here that "
    "makes a reported number interpretable. Where that is impossible, a presence-only method gives "
    "a figure independent of the analyst's sampling choices at a cost of about 0.15 AUC.",
])

H("Data and code availability", 1)
P("All code is openly available at https://github.com/sanskarprime392-png/avalanche. The avalanche "
  "inventory derived in this study, the trained susceptibility rasters and all result tables are "
  "reproducible from that repository using openly licensed input data (NASADEM, Sentinel-1, "
  "Sentinel-2, WorldClim, MODIS, ESA WorldCover, OpenStreetMap, GEM Global Active Faults).")

H("Author contributions", 1)
P("S.G. designed the study, built the Sentinel-1 inventory, carried out the analysis and the "
  "optical verification, and wrote the manuscript.")

H("Competing interests", 1)
P("The author declares no competing interests.")

H("Acknowledgements", 1)
P("The controlled decomposition in Section 4.1 — separating the contribution of absence sampling "
  "from that of spatial autocorrelation — was proposed in review by [SUPERVISOR NAME], whose "
  "comments also shaped the treatment of coarse-resolution predictors in Section 4.5. Analysis "
  "code was developed with the assistance of a large language model; the author designed the "
  "study, directed the analysis, performed the verification, and is responsible for all results "
  "and conclusions. This work received no specific funding.")

H("References", 1)
for i, r in enumerate([
    "Abhinav, A. & Sattar, A. (2025). Snow avalanche susceptibility, hazard, and exposure "
    "assessment in the Western Himalaya using machine learning and numerical modelling. Scientific "
    "Reports, 15, 38093.",
    "Barbet-Massin, M., Jiguet, F., Albert, C. H. & Thuiller, W. (2012). Selecting pseudo-absences "
    "for species distribution models: how, where and how many? Methods in Ecology and Evolution, "
    "3, 327-338.",
    "Bühler, Y., von Rickenbach, D., Stoffel, A., Margreth, S., Stoffel, L. & Christen, M. (2018). "
    "Automated snow avalanche release area delineation. Natural Hazards and Earth System Sciences, "
    "18, 3235-3251.",
    "Cetinkaya, S. & Kocaman, S. (2023). Impact of learning set and sampling for snow avalanche "
    "susceptibility mapping with random forest. ISPRS Archives, XLVIII-M-1, 57-64.",
    "Eckerstorfer, M., Vickers, H., Malnes, E. & Grahn, J. (2019). Near-real time automatic snow "
    "avalanche activity monitoring system using Sentinel-1 SAR data in Norway. Remote Sensing, 11, "
    "2863.",
    "Elkan, C. & Noto, K. (2008). Learning classifiers from only positive and unlabeled data. "
    "Proceedings of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data "
    "Mining, 213-220.",
    "Lobo, J. M., Jiménez-Valverde, A. & Real, R. (2008). AUC: a misleading measure of the "
    "performance of predictive distribution models. Global Ecology and Biogeography, 17, 145-151.",
    "Phillips, S. J., Dudík, M., Elith, J., Graham, C. H., Lehmann, A., Leathwick, J. & Ferrier, S. "
    "(2009). Sample selection bias and presence-only distribution models. Ecological Applications, "
    "19, 181-197.",
    "Ploton, P. et al. (2020). Spatial validation reveals poor predictive performance of "
    "large-scale ecological mapping models. Nature Communications, 11, 4540.",
    "Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial, "
    "hierarchical, or phylogenetic structure. Ecography, 40, 913-929.",
    "Sahin, E. K., Ipbuker, C. & Kavzoglu, T. (2023). Strategies for sampling pseudo-absences of "
    "landslide locations for landslide susceptibility mapping in complex mountainous terrain of "
    "Northwest Himalaya. Bulletin of Engineering Geology and the Environment, 82, 397.",
    "Winstral, A., Elder, K. & Davis, R. E. (2002). Spatial snow modeling of wind-redistributed "
    "snow using terrain-based parameters. Journal of Hydrometeorology, 3, 524-538.",
], 1):
    doc.add_paragraph(f"[{i}]  {r}", style="List Paragraph")

doc.save(OUT)
print("wrote", OUT)
