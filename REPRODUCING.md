# Reproducing the results

Every number, table and figure in the manuscript is produced by the scripts below. Figures are
written straight from the result CSVs, so regenerating a result regenerates the figure that uses it.

---

## 1. Environment

```bash
python -m pip install -r requirements.txt
```

Tested on Python 3.12, Windows. The Earth Engine steps additionally need an authenticated account:

```bash
earthengine authenticate
```

Set your own Earth Engine project id where the scripts call `ee.Initialize(project=...)`.

---

## 2. Which results need what

| Result | Needs Earth Engine | Needs the 1.9 GB raster set | Runs anywhere |
|---|---|---|---|
| Simulation (Sections 4.1–4.4) | no | no | **yes** |
| Inventory construction | yes | no | no |
| Empirical model results | no | yes | no |
| Maps | no | yes | no |

**The paper's central claim is in the first row.** Sections 4.1 to 4.4 are pure simulation and
reproduce on any machine in about 40 minutes with no data download and no Earth Engine account.

---

## 3. Reproducing the simulation results (no data needed)

```python
from models.simulation import (inflation_curve, robustness, direction_test,
                               validate_correction, structured_noise_test,
                               compare_presence_only, separability_of,
                               design_adjusted_skill)

inflation_curve()           # Table 1, Figure 2  - AUC against absence displacement
robustness()                # Table 2            - invariance to learner, dimensionality, n
direction_test(informative=False)   # Table 3    - invariance to displacement direction
validate_correction()       # Table 5            - where subtraction fails, by data structure
structured_noise_test()     # Table 6            - structured label contamination
compare_presence_only()     # Table 7            - PU and one-class comparison
```

`models/simulation.py` seeds every draw from the loop index, so repeated runs are identical.
Expect small differences only if scikit-learn's RandomForest implementation changes between
versions; the direction of every result is insensitive to this.

Table 4 and Figure 4 come from `correction_threshold.csv`, produced by the fine-grained sweep in
the same module (`null_baseline` across δ with `_make`/`_auc`).

---

## 4. Reproducing the empirical results

### 4.1 Build the inventory (Earth Engine)

```python
from gee.s1_inventory import multiyear_recurrence, export_candidates, release_points
from gee.dated_events import confident_dated_inventory
```

Orbit 136 descending, 2018–2025. `export_candidates` writes candidate deposit polygons;
`release_points(..., method="centroid")` derives release points. Note that `method="highest"` is
retained only to reproduce the artefact documented in Section 5.3 — do not use it.

Exports go to an Earth Engine asset rather than Drive (`ee.batch.Export.table.toAsset`), because a
full Drive is a common and confusing failure.

### 4.2 Fetch predictors

`gee/export_base_layers.py` writes the base layers; `features/reproject.py` puts everything on a
common 30 m UTM 43N grid; `features/terrain.py`, `features/distances.py`, `features/multiscale.py`
and `features/wind.py` derive the predictors.

**Distance to lineament must be computed from vector geometry**, not by rasterising and taking a
distance transform — every GEM active fault lies outside the study extent, so the rasterised version
is entirely NaN. See `models/predict_map.py` for the KD-tree approach.

### 4.3 Build the model tables

```python
from inventory.sampling import build_training_table
build_training_table(inv, proc_dir, out, negatives="paper")     # published-style absences
build_training_table(inv, proc_dir, out, negatives="matched")   # terrain + aspect matched
```

### 4.4 Run the models

```python
from models.tabular import compare_all, leakage_decomposition
from models.diagnostics import (compare_models_ci, geography_ablation,
                                background_ratio_sensitivity, sub_basin_transfer,
                                calibration, select_threshold)
```

---

## 5. Configurations referred to in the manuscript

| Label | Labels | Absences | Predictors | Spatial CV | AUC |
|---|---|---|---|---|---|
| Published protocol | recurrence ≥ 4 | flat/vegetated/urban | 20, single-scale | no | 1.000 |
| C1 | recurrence ≥ 4 | terrain-matched | 20, single-scale | yes | 0.934 |
| C2 | melt-out filtered | terrain + aspect matched | 20, single-scale | yes | 0.841 |
| C3 | melt-out filtered | terrain + aspect matched | 32, multi-scale | yes | 0.866 |

C2 and C3 differ **only** in the predictor set. The correction ladder ends at C2 so that each
correction's cost is isolated with predictors held fixed; design-adjusted figures use C3.

---

## 6. Counting convention

A release point is dropped when any predictor is undefined at its cell. Inventory counts therefore
exceed table sample sizes: 6,364 / 6,344 at recurrence ≥ 3, 3,188 / 3,179 at ≥ 4, 917 / 914 at ≥ 5,
241 / 240 at ≥ 6. Text quotes detections; tables quote points that entered a model.

---

## 7. Verification data

`verification/verdicts_firstpass.json` and `verification/verdicts_rulebased.json` hold both scorings
of the 100 reviewed detections, keyed by index into the sample. `verification/rescore.py` reproduces
the precision figures and the agreement statistic from them.

The chips themselves are regenerated by `inventory/verify.py::make_chips`, which needs Earth Engine.
Snow reflectance exceeds 6000 in the visible bands, so the stretch is min 300 / max 9000 — the usual
max of 3500 saturates every snowy scene to flat white and hides exactly the texture being judged.

---

## 8. Data sources

| Dataset | Used for | Access |
|---|---|---|
| Sentinel-1 GRD | avalanche detection | Earth Engine `COPERNICUS/S1_GRD` |
| Sentinel-2 SR | optical verification | Earth Engine `COPERNICUS/S2_SR_HARMONIZED` |
| NASADEM | terrain predictors | Earth Engine `NASA/NASADEM_HGT/001` |
| MODIS MOD10A1 | snow persistence filter | Earth Engine `MODIS/061/MOD10A1` |
| WorldClim | climate predictors | Earth Engine |
| ESA WorldCover | land cover | Earth Engine |
| OpenStreetMap | roads, streams | OSMnx / Overpass |
| GEM Global Active Faults | distance to lineament | github.com/GEMScienceTools/gem-global-active-faults |
| Google Open Buildings v3 | exposure (companion paper) | Earth Engine |

All are openly licensed. The 1.9 GB processed raster set is not committed; `DATA.md` documents how
to rebuild it.
