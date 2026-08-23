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

The line-of-sight version below (`reachable_hazard`) is retained only as a baseline. It over-counts
badly — 35,295 reachable buildings at alpha=30 deg — because it never checks that a flow path
actually connects source to target. Use `routed_building_risk`, which draws on the terrain-following
routing in models/flowpy.py and reduces that to 10,300 buildings (3.7% of stock).

Damage is graded by IMPACT PRESSURE rather than by susceptibility class:

    p = rho * g * z_delta        rho ~ 300 kg/m3 for a dense-flow avalanche

with z_delta the energy-line height delivered by the routing. Unreinforced masonry — the dominant
construction type in these valleys — suffers light damage from ~3 kPa, serious damage from ~10 kPa
and total destruction above ~30 kPa.

SCOPE: this is a worst-case screening product, not an annual risk. It assumes every high-
susceptibility release area fails; it carries no return period, and no release-depth scenarios.
Adding frequency requires either a dated inventory or scenario modelling by release depth.
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


SNOW_DENSITY = 300.0        # kg/m3, dense-flow avalanche
G = 9.81
# impact pressure thresholds (kPa) for unreinforced masonry, with damage fraction
PRESSURE_BANDS = [(0.0, 3.0, "negligible", 0.00),
                  (3.0, 10.0, "light", 0.15),
                  (10.0, 30.0, "serious", 0.50),
                  (30.0, 1e9, "destructive", 1.00)]


def impact_pressure_kpa(z_delta, density=SNOW_DENSITY):
    """Impact pressure from energy-line height: p = rho * g * z_delta."""
    return density * G * np.asarray(z_delta, dtype="float64") / 1000.0


def routed_building_risk(zdelta_tif, source_sus_tif, buildings_xy,
                         occupants_per_building=5.0, density=SNOW_DENSITY):
    """Building-level risk from terrain-routed runout. This is the tool's primary output."""
    import pandas as pd

    from .flowpy import sample_at_points

    with rasterio.open(zdelta_tif) as s:
        zd = s.read(1)
        transform = s.transform
    with rasterio.open(source_sus_tif) as s:
        ss = s.read(1)

    b_zd = sample_at_points(zd, transform, buildings_xy)
    b_ss = sample_at_points(ss, transform, buildings_xy)
    reached = np.isfinite(b_zd)
    p = impact_pressure_kpa(np.where(reached, b_zd, 0.0), density)

    band = np.full(len(buildings_xy), "not reached", dtype=object)
    dmg = np.zeros(len(buildings_xy))
    for lo, hi, name, frac in PRESSURE_BANDS:
        m = reached & (p > lo) & (p <= hi)
        band[m] = name
        dmg[m] = frac

    per_building = pd.DataFrame(dict(
        x=buildings_xy[:, 0], y=buildings_xy[:, 1],
        reached=reached, z_delta_m=b_zd, impact_kpa=np.where(reached, p, np.nan),
        source_susceptibility=b_ss, damage_band=band, damage_fraction=dmg))

    rows = []
    for _, _, name, frac in PRESSURE_BANDS:
        m = per_building.damage_band == name
        n = int(m.sum())
        if not n:
            continue
        rows.append(dict(damage_band=name, buildings=n,
                         pct_of_stock=100 * n / len(per_building),
                         median_impact_kpa=float(per_building.loc[m, "impact_kpa"].median()),
                         damage_fraction=frac,
                         expected_buildings_destroyed=n * frac,
                         population_exposed=n * occupants_per_building))
    summary = pd.DataFrame(rows)
    if len(summary):
        summary.loc[len(summary)] = dict(
            damage_band="TOTAL EXPOSED", buildings=int(reached.sum()),
            pct_of_stock=100 * reached.mean(), median_impact_kpa=np.nan,
            damage_fraction=np.nan,
            expected_buildings_destroyed=summary.expected_buildings_destroyed.sum(),
            population_exposed=int(reached.sum()) * occupants_per_building)
    return per_building, summary


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
