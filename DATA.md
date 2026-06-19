# DATA.md — dataset access & download guide

Two kinds of data:

- **A. Earth Engine-native** → *no manual download*. The export scripts pull these clipped to the
  AOI into your Drive. You only need to authenticate EE once.
- **B. Manual downloads** → you grab these from the portals below and drop them into the Drive
  folders. Paths come from `config.PATHS` (root `MyDrive/avalanche/data/`).

Drive layout to create first:
```
MyDrive/avalanche/data/
├── raw/predictors/   (rasters: DEM, climate, snow, LULC, geology)
├── raw/labels/       (avalanche inventory we build in Phase 0)
├── raw/exposure/     (buildings, glacial lakes)
├── processed/  models/  figures/  results/
```

---

## A. Earth Engine — handled by scripts (just authenticate)

```python
import ee; ee.Authenticate(); ee.Initialize(project=config.EE_PROJECT)
%run gee/export_base_layers.py     # DEM, WorldClim T/P, WorldCover, MODIS SCD
%run gee/s1_avalanche_detection.py # Phase 0 SAR candidates
```

| Dataset | EE ID | Used for |
|---|---|---|
| NASADEM 30 m | `NASA/NASADEM_HGT/001` | DEM → all terrain & hydrology derivatives |
| WorldClim V1 monthly | `WORLDCLIM/V1/MONTHLY` | winter temp + precip (you also have v2 locally) |
| ESA WorldCover v200 | `ESA/WorldCover/v200` | land cover (10 m) |
| MODIS MOD10A1 | `MODIS/061/MOD10A1` | snow-cover duration (500 m) |
| Sentinel-1 GRD | `COPERNICUS/S1_GRD` | **label source** (avalanche debris detection) |
| Sentinel-2 SR | `COPERNICUS/S2_SR_HARMONIZED` | optical verification + pre-event patch channel |
| ERA5-Land monthly | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | climate **trend figures only** (too coarse as a predictor) |
| NEX-GDDP-CMIP6 | `NASA/GDDP-CMIP6` | Phase 4 future-climate scenarios (downscaled 0.25°) |
| Google Open Buildings v3 | `GOOGLE/Research/open-buildings/v3/polygons` | exposure analysis |

> Exact band names / scale factors are set in the scripts. CMIP6 + Open Buildings get their own
> export scripts when we reach those phases.

---

## B. Manual downloads

For each: **portal → steps → format → ~size → where it goes.** Where I give a portal rather than a
deep link, navigate from there (deep links rot); ping me and I'll verify a specific one live.

### B1. WorldClim v2.1 — *you have it*
- Portal: https://www.worldclim.org/data/worldclim21.html  → "Historical climate data"
- Need: `tavg` + `prec`, 30-arc-sec (~1 km). Format: GeoTIFF (12 monthly bands each).
- Size: ~300–700 MB per variable (global); we clip on load.
- Goes to: `raw/predictors/climate/`

### B2. OSM roads + streams — *you have it*
- Portal: https://download.geofabrik.de/asia/india.html  (or the HP/J&K sub-extracts)
- Need: `lines` layer → filter `highway=*` (roads) and `waterway=*` (streams).
- Format: `.osm.pbf` or shapefile. Size: a few hundred MB (India), tiny after clip.
- Goes to: `raw/predictors/osm/`

### B3. Randolph Glacier Inventory 7.0 — **use v7, not v6**
- **Download:** https://nsidc.org/data/nsidc-0770/versions/7  (DOI https://doi.org/10.5067/f6jmovy5navz)
- Region guide: https://www.glims.org/rgi_user_guide/regions/rgi14.html
- Need: **Region 14 (South Asia West)** — covers Chandra-Bhaga / Upper Beas (GAMDAM v2, 37,562
  glaciers). Region 15 = central/east Himalaya, only if you widen the AOI east.
- Steps: open the NSIDC page → "Download Data" → grab the **per-region package for RGI2000-v7.0-G-14**
  (or the global package). Free, no login for RGI.
- Format: shapefiles + CSV hypsometry + JSON metadata. Size: tens of MB.
- Goes to: `raw/predictors/glaciers/`

### B4. ICIMOD glacial lakes — *you have it*
- Portal: https://rds.icimod.org/  → search "glacial lake inventory" (HKH).
- Format: shapefile. Goes to: `raw/exposure/glacial_lakes/`

### B5. Geology — lithology + lineaments  (Bhukosh is unreliable → use open sources)
GSI Bhukosh (https://bhukosh.gsi.gov.in/Bhukosh/Public) is chronically down/timing out and
India-only. Geology is non-blocking and low-priority (lithology ranked 16th), so don't let it
stall the pipeline. Recommended open, reproducible sources:

- **Lineaments (matters — ranked 3rd):** GEM Global Active Faults (open, GitHub, CC-BY-SA). Fetch
  in Colab: `!git clone https://github.com/GEMScienceTools/gem-global-active-faults`
  Use `gem_active_faults.gpkg`; distance-to-fault = the "distance-to-lineament" factor. (≈ major
  tectonic lineaments incl. MCT/MBT/MFT — a cleaner, physically-meaningful proxy than photo-lineaments.)
- **Lithology (minor):** GLiM 0.5° gridded, free on PANGAEA: https://doi.pangaea.de/10.1594/PANGAEA.788537
  (coarse ~55 km — fine for a low-importance factor). OR retry Bhukosh later for the detailed India
  map and swap it in.
- Goes to: `raw/predictors/geology/`

### B6. High-Mountain-Asia 8 m DEM — *the resolution upgrade (Phase 2)*
- **Download:** https://nsidc.org/data/hma_dem8m_mos/versions/1  (DOI https://doi.org/10.5067/KXOVQ9L172S2,
  Shean 2017 — use the **MOS = seamless mosaics** product, not the AT/CT per-scene ones).
- Requires a free **NASA Earthdata Login** (https://urs.earthdata.nasa.gov/).
- Steps: open page → "Download Data" → pick the **100 km mosaic tiles** over the AOI; or in Colab use
  the `earthaccess` lib (`earthaccess.login()` → `earthaccess.search_data(short_name="HMA_DEM8m_MOS")`).
- Format: GeoTIFF tiles (12500×12500 px @ 8 m). Size: large — download only AOI tiles.
- Goes to: `raw/predictors/dem8m/`

### B7. Lievens Sentinel-1 snow depth (C-SNOW, ~1 km) — *the snowpack predictor*
- **Request form (NOT a direct download):** https://ees.kuleuven.be/eng/apps/project-c-snow-data/
  (note the `/eng/` — the non-`/eng/` URL 404s). Fill name/email/org/intended-use; access is then
  granted (not instant). Non-commercial; cite Lievens et al. 2019, 2022.
- Get the **Northern Hemisphere** product: ~1 km, **2016–2020** (exactly what the paper used).
  A 500 m European-Alps version also exists but doesn't cover the Himalaya.
- Source papers: Lievens et al. 2019, *Nature Communications* (https://www.nature.com/articles/s41467-019-12566-y)
  + 2022 update. Non-commercial use; cite the papers.
- Format: NetCDF (snow depth [m] + lat/lon + QC flags). Goes to: `raw/predictors/snow/`
- Note: this is the localized snow input. **ERA5 snow depth (~9 km) is NOT a substitute** — use
  ERA5 only for trend plots.

---

## What's needed for which phase

- **Phase 1 reproduction:** A-layers + B1, B2, B3, B4, B5, B7.
- **Phase 2 rigor:** add **B6** (8 m DEM).
- **Phase 3 DL:** A-layers (S1/S2) + the multi-channel mosaics (built later).
- **Phase 4:** CMIP6 (EE) + RGI/lakes already in hand.

You already have B1, B2, B4 (and RGI — just confirm it's **v7**). **Priority new fetches: B3 (RGI 7.0),
B5 (Bhukosh), B6 (8 m DEM), B7 (snow depth).**
