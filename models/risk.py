"""
models/risk.py — from susceptibility to quantified risk.

Susceptibility maps predict where avalanches RELEASE. Buildings are almost never in release zones;
they are in RUNOUT zones below them. Intersecting a susceptibility map directly with building
locations therefore answers the wrong question, and is why exposure studies need a runout step.

Rather than a full dynamic simulation (RAMMS is proprietary and Windows-only; r.avaflow is the open
alternative but is a project in itself), this module uses the classical ENERGY-LINE / alpha-angle
model: an avalanche released at a point of elevation z_r comes to rest where the straight line from
the release point to the stopping point subtends an angle alpha with the horizontal. Alpha is
roughly 18-25 deg for large avalanches in comparable terrain; smaller alpha means longer reach.

    a building at (x_b, z_b) is reachable from release cell (x_r, z_r)
        iff  atan( (z_r - z_b) / horizontal_distance ) >= alpha

Risk is then assembled in the standard form

    risk = hazard x exposure x vulnerability

where hazard combines release susceptibility with reachability, exposure is the building count, and
vulnerability comes from a fragility relationship for the dominant construction type.

STATUS — NOT PUBLICATION READY. Tested against 277,529 Open Buildings centroids this gives 16,917 /
35,295 / 46,813 reachable buildings at alpha = 34 / 30 / 28 deg, against 161-557 reported by
Abhinav & Sattar from explicit RAMMS simulation of 371 selected sources. Two reasons the numbers
are inflated, both structural rather than parameter choices:

  1. The energy line is line-of-sight. Without flow routing a building separated from a source by an
     intervening ridge or valley still qualifies, so reach is systematically overestimated.
  2. Taking the MAXIMUM susceptibility over the (often >100) qualifying sources saturates near 1.0,
     which is why essentially every reachable building falls in the top hazard class. The grading is
     an artefact of the aggregation, not a property of the terrain.

Use this module for screening-level upper bounds only. Credible exposure requires routed runout —
Flow-Py (open source, energy line WITH flow routing) or r.avaflow — which is a separate work package.
"""
import numpy as np
import rasterio
from scipy.spatial import cKDTree

# Damage fraction for stone/brick masonry as a function of avalanche impact intensity.
# Himalayan settlements are dominated by unreinforced masonry, which is far more fragile than the
# reinforced concrete assumed in most Alpine curves.
FRAGILITY = {"low": 0.10, "moderate": 0.35, "high": 0.70, "very high": 0.95}


def load_raster(path):
    with rasterio.open(path) as s:
        a = s.read(1).astype("float32")
        if s.nodata is not None and not np.isnan(s.nodata):
            a[a == s.nodata] = np.nan
        return a, s.transform, s.crs


def reachable_hazard(sus_path, dem_path, buildings_xy, alpha_deg=28.0, max_dist_m=2000.0,
                     sus_min=0.75, min_release_px=6):
    """For each building, the maximum release susceptibility that can physically reach it.

    Returns (hazard, n_sources): hazard is the highest susceptibility among release cells whose
    energy line clears the building; n_sources counts how many such cells exist.

    Two constraints keep this physically meaningful. `sus_min` restricts sources to the top
    susceptibility class, and `min_release_px` requires a contiguous release patch (default 6 cells
    ~ 5,400 m2) because a single 30 m pixel cannot generate a destructive avalanche.

    IMPORTANT CAVEAT: the energy line is a line-of-sight criterion. It does not verify that a flow
    path actually connects source to target, so a building separated from a high slope by an
    intervening ridge or valley can still satisfy it. Results are therefore an UPPER BOUND on
    exposure; converting them to true exposure requires flow routing (e.g. Flow-Py or r.avaflow).
    """
    from scipy import ndimage

    sus, transform, _ = load_raster(sus_path)
    dem, dtr, _ = load_raster(dem_path)
    if dem.shape != sus.shape:
        raise ValueError("susceptibility and DEM must share a grid")

    rel = np.isfinite(sus) & (sus >= sus_min)
    if min_release_px > 1:                      # drop isolated cells: no release area, no avalanche
        lab, n = ndimage.label(rel)
        sizes = np.bincount(lab.ravel())
        keep = np.isin(lab, np.nonzero(sizes >= min_release_px)[0][1:])
        print(f"  release patches >= {min_release_px} px: "
              f"{int(keep.sum()):,} cells kept of {int(rel.sum()):,}")
        rel = keep
    rr, cc = np.nonzero(rel)
    if rr.size == 0:
        return np.zeros(len(buildings_xy)), np.zeros(len(buildings_xy), int)
    rx = transform.c + (cc + 0.5) * transform.a
    ry = transform.f + (rr + 0.5) * transform.e
    rz = dem[rr, cc]
    rs = sus[rr, cc]
    ok = np.isfinite(rz)
    rx, ry, rz, rs = rx[ok], ry[ok], rz[ok], rs[ok]
    print(f"  candidate release cells (susceptibility >= {sus_min}): {len(rx):,}")

    # building elevations
    inv = ~transform
    bcol, brow = inv * (buildings_xy[:, 0], buildings_xy[:, 1])
    brow = np.clip(np.floor(brow).astype(int), 0, dem.shape[0] - 1)
    bcol = np.clip(np.floor(bcol).astype(int), 0, dem.shape[1] - 1)
    bz = dem[brow, bcol]

    tree = cKDTree(np.column_stack([rx, ry]))
    tan_a = np.tan(np.deg2rad(alpha_deg))
    hazard = np.zeros(len(buildings_xy), "float32")
    n_src = np.zeros(len(buildings_xy), int)

    for i, (bx, by) in enumerate(buildings_xy):
        idx = tree.query_ball_point([bx, by], max_dist_m)
        if not idx:
            continue
        idx = np.asarray(idx)
        d = np.hypot(rx[idx] - bx, ry[idx] - by)
        drop = rz[idx] - bz[i]
        reach = (d > 0) & (drop > 0) & (drop / d >= tan_a)
        if reach.any():
            hazard[i] = rs[idx][reach].max()
            n_src[i] = int(reach.sum())
    return hazard, n_src


def classify_hazard(h):
    out = np.full(h.shape, "none", dtype=object)
    out[(h > 0.00) & (h <= 0.25)] = "low"
    out[(h > 0.25) & (h <= 0.50)] = "moderate"
    out[(h > 0.50) & (h <= 0.75)] = "high"
    out[h > 0.75] = "very high"
    return out


def quantify_risk(hazard, n_sources, occupants_per_building=5.0):
    """Assemble risk = hazard x exposure x vulnerability over the exposed building stock."""
    import pandas as pd
    cls = classify_hazard(hazard)
    rows = []
    for c in ("low", "moderate", "high", "very high"):
        m = cls == c
        n = int(m.sum())
        if not n:
            continue
        v = FRAGILITY[c]
        rows.append(dict(hazard_class=c, buildings=n,
                         mean_susceptibility=float(hazard[m].mean()),
                         mean_paths=float(n_sources[m].mean()),
                         vulnerability=v,
                         expected_damaged=n * v,
                         population_exposed=n * occupants_per_building))
    df = pd.DataFrame(rows)
    if len(df):
        df.loc[len(df)] = dict(hazard_class="TOTAL", buildings=df.buildings.sum(),
                               mean_susceptibility=np.nan, mean_paths=np.nan,
                               vulnerability=np.nan,
                               expected_damaged=df.expected_damaged.sum(),
                               population_exposed=df.population_exposed.sum())
    return df
