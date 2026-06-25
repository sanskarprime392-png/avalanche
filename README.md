# Avalanche Susceptibility — Reproduce & Improve (Western Himalaya)

Research project: reproduce and methodologically improve **Abhinav & Sattar (2025)**,
*"Snow avalanche susceptibility, hazard, and exposure assessment in the Western Himalaya
using machine learning and numerical modelling"*, **Scientific Reports 15:38093**
([DOI 10.1038/s41598-025-22051-w](https://doi.org/10.1038/s41598-025-22051-w)).

Study area: **Chandra-Bhaga + Upper Beas basins** (Himachal Pradesh / J&K), elev. 1123–6329 m.

---

## Confirmed scope (rev. 2026-06-20)

Two deliverables: **(1)** a new, larger, openly-documented Himalayan avalanche inventory, and
**(2)** a leakage-controlled benchmark of **tabular ML vs. multimodal deep learning** for
avalanche susceptibility.

- **Inventory (Phase 0):** build our own (no public label set exists). Sentinel-1 SAR
  change-detection + Sentinel-2 / Maxar optical verification, expanded in space & time across
  the **Western Himalaya coherent snow-met regime**. Staged & **confidence-tiered**
  (A = optically confirmed · B = SAR+terrain plausible · C = SAR-only candidate). Target grows
  from a ~300–500 clean seed toward ~2000 (5000 aspirational, gated on verification cost).
- **Tabular spine:** RF + **XGBoost / LightGBM / CatBoost** on the 20-factor stack —
  reproduction → rigor upgrade. Runs on the seed inventory for early results.
- **Multimodal DL:** patch-based **hybrid CNN(+MLP)** — ResNet → EfficientNet → Swin.

## Why this beats the original (the contribution)

The paper's metrics (RF AUC 0.95) are almost certainly optimistic. We fix, and quantify:

1. **Negative sampling** — terrain-matched background absences, not "flat farmland".
2. **Spatial cross-validation** — spatial block CV / leave-region-out + Moran's I, reported
   alongside the (inflated) random-CV numbers. Applied to **both** tabular and patch models.
3. **Road-proximity confound** — distance-to-road tested as bias, not trusted as a predictor.
4. **Resolution** — 8 m High-Mountain-Asia DEM (Shean et al.) instead of 30 m NASADEM.
5. **Baselines the paper skipped** — XGBoost / LightGBM / CatBoost (GBMs are the bar to beat).
6. **A larger, tiered inventory** — removes the paper's temporal (86/118 from 2023) & road bias.
7. **An honest DL-vs-tabular benchmark** — most "DL beats RF" susceptibility claims are leakage
   artifacts; we test it leakage-free, and report whichever way it lands.

### Two non-negotiable guardrails (the traps that sink this kind of project)

- **No circularity:** the event-window S1 debris signal is used **only to label**, **never** as
  a model input. DL/tabular inputs are *pre-conditioning state only* — terrain, climatological
  snow (SCD, snow-depth climatology), climate, land cover, and optionally *pre-event* S2 /
  *seasonal-mean* SAR. Feeding the post-event debris scene back in = a tautology, not a result.
- **No leakage:** spatially-blocked splits with **zero patch overlap across folds**; the
  hold-out test set is fully optically verified (Tier-A) and spatially disjoint. Train may use
  noisy labels; evaluation may not.

## Phased plan

| Phase | Work | Status |
|---|---|---|
| **0** | Scaled, confidence-tiered inventory (S1 SAR + S2/Maxar verify) + sampling design | **active** |
| **1** | Faithful reproduction: 20-factor stack, RF/SVM/LR/ANN, metrics, ASM map | pending |
| **2** | Rigor upgrade: spatial CV, matched negatives, 8 m DEM, GBMs, SHAP — *core paper* | pending |
| **3** | **Multimodal DL benchmark**: hybrid CNN(+MLP), ResNet/EfficientNet/Swin vs GBM | pending |
| **4** | Novelty (pick 1): CMIP6 future susceptibility · GLOF process-chain (r.avaflow) | parked |

**DL input design (Phase 3):** CNN branch on high-res spatial channels (DEM, slope, aspect,
curvature, TRI/TPI/VRM, valley depth, pre-event S2, seasonal-mean SAR); MLP branch on coarse
tabular features (climate, snow climatology, land cover) — coarse vars are *flat planes* in a
patch and must not be CNN channels. Explainability: SHAP (tabular) + Integrated Gradients /
Grad-CAM (DL).

**Colab practicalities:** export aligned multi-channel AOI mosaics from GEE *once* (COGs to
Drive), then cut patches locally with rasterio/xarray — per-point GEE patch export hits quota
limits. ~4k patches × 256² × 8 ch ≈ 8 GB. T4 handles ResNet/EfficientNet; Swin wants A100 (Pro+).

> **RAMMS does not run on Colab** (proprietary Windows GUI). The hazard/runout half uses
> open-source **Flow-Py** / **r.avaflow** instead.

## Folder structure

```
avalanche/
├── README.md
├── config.py                     # AOI, study periods, paths, CRS (no secrets)
├── data_access/
│   └── fetch_csnow.py            # SFTP download of C-SNOW HMA snow-depth NetCDFs
├── gee/
│   ├── s1_avalanche_detection.py # Phase 0: SAR debris change-detection (label source)
│   ├── export_base_layers.py     # GEE-native base rasters (DEM, climate, LULC, SCD)
│   └── export_patch_mosaics.py   # Phase 3: aligned multi-channel COG mosaics
├── inventory/
│   ├── labeling.py               # SAR-assisted manual labelling (geemap markers -> tiered CSV)
│   ├── sampling.py               # presence declustering + terrain-matched absences + blocks
│   └── verify.py                 # tier A/B/C verification + active-learning queue
├── features/
│   ├── reproject.py              # warp raw 4326 GEE exports -> UTM 43N (EPSG:32643), metres
│   ├── terrain.py                # LOCAL terrain+hydrology derivatives from DEM (WhiteboxTools)
│   ├── distances.py              # dist-to-road/stream (OSMnx) + dist-to-lineament (GEM faults)
│   └── patches.py                # cut spatially-blocked, non-overlapping patches from mosaics
├── models/
│   ├── tabular.py                # RF / XGBoost / LightGBM / CatBoost + spatial-CV harness
│   └── multimodal.py             # hybrid CNN(+MLP): ResNet / EfficientNet / Swin
├── figures/
│   └── pubstyle.py               # 300-DPI publication matplotlib style
└── data/                         # NOT in git — lives in Google Drive
    ├── raw/{predictors,labels,exposure}
    ├── processed/
    ├── models/
    ├── figures/
    └── results/
```

## Colab workflow

1. Push this folder to a private GitHub repo (source-of-truth).
2. In Colab: `!git clone <repo>` then `from google.colab import drive; drive.mount('/content/drive')`.
3. Earth Engine: `import ee; ee.Authenticate(); ee.Initialize(project='<your-gcp-project>')`.
4. Big rasters export from GEE **clipped to the AOI** → Drive (`data/`). No global downloads.

GPU: not needed for Phase 1–2 (tabular). A free CPU runtime + 12 GB RAM is enough.
