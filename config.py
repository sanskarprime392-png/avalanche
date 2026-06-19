"""Shared project configuration (no secrets, no Earth Engine import here).

Imported by both GEE scripts and figure/analysis code. Keep this the single
source of truth for the study area, time windows, CRS and Drive paths.
"""

# --- Study area -------------------------------------------------------------
# APPROXIMATE bounding box for Chandra-Bhaga + Upper Beas basins (Lahaul-Spiti /
# Manali / Keylong / Kishtwar). TODO: replace with the digitized basin polygon
# (delineate from the DEM, or trace from the paper's Fig. 1, or HydroBASINS L8).
AOI_BBOX = [76.40, 31.90, 78.10, 33.20]  # [west, south, east, north] in EPSG:4326

# --- CRS / grid -------------------------------------------------------------
CRS = "EPSG:4326"          # working CRS for export; reproject to a local UTM (43N
UTM_CRS = "EPSG:32643"     # = EPSG:32643) for any distance/area computation.
TARGET_RES_M = 30          # reproduction grid (paper resampled everything to 30 m)
FINE_RES_M = 8             # improvement grid (HMA 8 m DEM) for Phase 2

# --- Time windows -----------------------------------------------------------
# Avalanche season in the W Himalaya peaks Jan–May (paper: 53 Mar, 38 Apr, 14 May,
# 10 Feb, 2 Jan). Define detection "activity" windows per target season here.
SEASON_MONTHS = [1, 2, 3, 4, 5]
ANALYSIS_YEARS = list(range(2017, 2025))   # S1 has good coverage from ~2017

# WorldClim / climatology baseline used by the paper
WORLDCLIM_PERIOD = (1970, 2000)

# --- Drive paths (Colab) ----------------------------------------------------
DRIVE_ROOT = "/content/drive/MyDrive/avalanche"
PATHS = {
    "predictors": f"{DRIVE_ROOT}/data/raw/predictors",
    "labels":     f"{DRIVE_ROOT}/data/raw/labels",
    "exposure":   f"{DRIVE_ROOT}/data/raw/exposure",
    "processed":  f"{DRIVE_ROOT}/data/processed",
    "models":     f"{DRIVE_ROOT}/data/models",
    "figures":    f"{DRIVE_ROOT}/data/figures",
    "results":    f"{DRIVE_ROOT}/data/results",
}

# --- Earth Engine -----------------------------------------------------------
EE_PROJECT = "your-gcp-project-id"   # TODO: set to your Google Cloud project for ee.Initialize()

# GEE export folder (relative to Drive root, created by EE export tasks)
GEE_EXPORT_FOLDER = "avalanche_gee_exports"
