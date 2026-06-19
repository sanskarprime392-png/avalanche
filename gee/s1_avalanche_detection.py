"""
Phase 0 — Sentinel-1 SAR avalanche-debris detection (semi-automated).

Avalanche debris roughens the surface -> backscatter INCREASES in the deposition
zone relative to undisturbed snow. We detect it by temporal change detection:
a stable pre-event "reference" composite vs. a post-event "activity" composite,
differenced in dB (a log-ratio). Positive change = candidate debris.

IMPORTANT — this is *semi-automated*. Every candidate polygon exported here is a
CANDIDATE. It must be validated against Sentinel-2 / Google-Earth optical imagery
before it enters the avalanche inventory. Tune CHANGE_THRESH_DB / MIN_AREA_M2 on a
handful of known events first.

Domain notes that distinguish this from a Norway-tuned (Eckerstorfer et al.) detector:
  * Debris/deposition sits in the RUNOUT (gentle slopes / valley floors), while the
    release zone is steep (30-50 deg). So we do NOT restrict detection to steep slopes
    -- we detect the debris footprint, then derive the release point upslope (see
    derive_release_point, a TODO hook below).
  * Steep terrain -> mask SAR layover & shadow, else every lee slope is a false positive.
  * Use BOTH ascending and descending orbits; run them separately and intersect/union.
  * Wet snow LOWERS backscatter (opposite sign) -> a useful disambiguator, not noise.

Refinement (recommended, not implemented here): full radiometric terrain flattening
(gamma0, Vollrath et al. 2020 volumetric model) instead of the GRD sigma0 + mask used
below. Hook left at apply_terrain_flattening().

Run in Colab:
    import ee; ee.Authenticate(); ee.Initialize(project="<your-gcp-project>")
    %run gee/s1_avalanche_detection.py
"""
import ee
import config

# --- tunables (calibrate on known events before trusting these) -------------
CHANGE_THRESH_DB = 2.0     # dB increase to flag debris (start 2-3, tune)
MIN_AREA_M2 = 1500.0       # drop blobs smaller than a plausible deposit
MIN_SLOPE_DEG = 10.0       # exclude flat valley floor / water (NOT an upper bound)
MAX_SLOPE_DEG = 60.0
SPECKLE_RADIUS_M = 30.0    # focal smoothing radius


# --- helpers ----------------------------------------------------------------
def to_linear(img_db):
    """GEE S1_GRD bands are in dB; convert to linear power for correct compositing."""
    return ee.Image(10.0).pow(img_db.divide(10.0))


def to_db(img_lin):
    return img_lin.log10().multiply(10.0)


def get_aoi():
    return ee.Geometry.Rectangle(config.AOI_BBOX)


def terrain_layers(aoi):
    """Slope + a layover/shadow validity mask from NASADEM."""
    dem = ee.Image("NASA/NASADEM_HGT/001").select("elevation").clip(aoi)
    slope = ee.Terrain.slope(dem)  # degrees
    return dem, slope


def load_s1(aoi, start, end, orbit_pass):
    """Sentinel-1 GRD, IW, dual-pol VV+VH, one orbit direction. Bands are in dB."""
    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
        .select(["VV", "VH", "angle"])
    )
    return col


def apply_terrain_flattening(col):
    """TODO (refinement): replace GRD sigma0 with gamma0 radiometric terrain
    flattening (Vollrath et al. 2020). For now we pass through and rely on the
    layover/shadow + incidence-angle mask in detect_change(). Steep-terrain SAR
    intensities are only quantitatively comparable after this step."""
    return col


def composite_db(col):
    """Speckle-reduced composite: mean in LINEAR power, returned in dB."""
    lin = col.select(["VV", "VH"]).map(lambda im: to_linear(im).copyProperties(im))
    mean_lin = lin.mean()
    comp = to_db(mean_lin).rename(["VV", "VH"])
    # focal smoothing to further suppress speckle
    return comp.focal_mean(radius=SPECKLE_RADIUS_M, units="meters")


def detect_change(aoi, ref_window, act_window, orbit_pass):
    """Reference vs activity log-ratio (dB) per polarisation; return change image."""
    ref = apply_terrain_flattening(load_s1(aoi, ref_window[0], ref_window[1], orbit_pass))
    act = apply_terrain_flattening(load_s1(aoi, act_window[0], act_window[1], orbit_pass))

    ref_db = composite_db(ref)
    act_db = composite_db(act)

    change = act_db.subtract(ref_db).rename(["dVV", "dVH"])  # dB increase = positive

    # incidence-angle proxy for layover/shadow: use the activity-window mean angle.
    angle = act.select("angle").mean()
    valid_angle = angle.gt(20).And(angle.lt(45))  # crude; replace with true LIA mask

    _, slope = terrain_layers(aoi)
    valid_slope = slope.gt(MIN_SLOPE_DEG).And(slope.lt(MAX_SLOPE_DEG))

    candidate = (
        change.select("dVV").gt(CHANGE_THRESH_DB)
        .And(change.select("dVH").gt(CHANGE_THRESH_DB - 1.0))  # VH a touch looser
        .And(valid_angle)
        .And(valid_slope)
    )
    return change.updateMask(candidate), candidate.selfMask()


def vectorize_candidates(candidate_mask, aoi):
    """Connected-component vectors with a minimum-area filter."""
    vectors = candidate_mask.reduceToVectors(
        geometry=aoi,
        scale=10,
        geometryType="polygon",
        eightConnected=True,
        maxPixels=1e13,
    )
    return vectors.map(
        lambda f: f.set("area_m2", f.geometry().area(maxError=1))
    ).filter(ee.Filter.gte("area_m2", MIN_AREA_M2))


def derive_release_point(candidate_polys, dem):
    """TODO: trace each deposit footprint upslope to its steep (30-50 deg) release
    zone and emit a single release-zone POINT per event. That point -- not the
    deposit -- is the presence sample for the susceptibility model (susceptibility
    ~= release probability). For now we export footprints for manual interpretation."""
    return candidate_polys


def export_to_drive(fc, description):
    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=description,
        folder=config.GEE_EXPORT_FOLDER,
        fileFormat="GeoJSON",
    )
    task.start()
    print(f"started export task: {description}")
    return task


def detect_event(ref_window, act_window, tag, orbit_passes=("ASCENDING", "DESCENDING")):
    """Run detection for one event window across both orbit geometries."""
    aoi = get_aoi()
    out = []
    for op in orbit_passes:
        _, mask = detect_change(aoi, ref_window, act_window, op)
        polys = vectorize_candidates(mask, aoi).map(lambda f: f.set("orbit", op, "event", tag))
        out.append(polys)
    candidates = ee.FeatureCollection(out).flatten()
    return export_to_drive(candidates, f"s1_avalanche_candidates_{tag}")


if __name__ == "__main__":
    # EXAMPLE — one known March-2023 cycle (calibrate windows to your target events).
    # Reference = stable mid-winter before the cycle; activity = right after it.
    detect_event(
        ref_window=("2023-01-15", "2023-02-10"),
        act_window=("2023-03-10", "2023-03-25"),
        tag="2023-03",
    )
