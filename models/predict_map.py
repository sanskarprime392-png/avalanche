"""
models/predict_map.py — render the basin-wide avalanche susceptibility map.

Design decisions that differ from common practice:

* TPI is EXCLUDED. Its apparent importance is an artefact of our own release-point derivation
  (presence TPI averages +36 vs +3 for background because the rule picks the highest qualifying
  pixel in a search radius). Costs 0.005 AUC to drop.
* Distance-to-lineament is computed from fault GEOMETRY on a coarse expanded grid, because every
  GEM active fault lies outside the AOI, so a rasterised-then-distance-transformed version is
  all-NaN. It varies over 14-153 km, so 300 m sampling is ample.
* The map is MASKED TO POTENTIAL RELEASE AREAS (slope 28-60 deg, after Buhler et al.). Classifying
  valley floors and lake surfaces as "very low susceptibility" inflates the denominator and makes
  the usual "x% of the basin is highly susceptible" statistic incomparable between studies.
* Output values are a relative susceptibility INDEX, not calibrated probabilities: this is
  presence-background learning at an arbitrary prevalence (see models.diagnostics.calibration).
"""
import os

import numpy as np
import rasterio
from rasterio.windows import Window

from .tabular import get_models, feature_columns

EXCLUDE = {"tpi"}


def fault_distance_grid(ref_profile, faults_path, pad_deg=2.5, coarse_m=300.0):
    """Coarse distance-to-fault surface covering the AOI plus a pad, in the reference CRS."""
    import geopandas as gpd
    from rasterio.warp import transform_bounds
    from scipy.spatial import cKDTree
    from shapely import segmentize

    t, w, h, crs = (ref_profile["transform"], ref_profile["width"],
                    ref_profile["height"], ref_profile["crs"])
    left, top = t.c, t.f
    right, bottom = left + w * t.a, top + h * t.e
    wl, ws, we, wn = transform_bounds(crs, "EPSG:4326", left, bottom, right, top)

    g = gpd.read_file(faults_path, bbox=(wl - pad_deg, ws - pad_deg, we + pad_deg, wn + pad_deg))
    g = g[g.geometry.type.isin(["LineString", "MultiLineString"])].to_crs(crs)
    # densify the fault lines into points so a KD-tree gives distance-to-line
    pts = []
    for geom in g.geometry:
        dens = segmentize(geom, max_segment_length=coarse_m)
        for part in getattr(dens, "geoms", [dens]):
            pts.extend(list(part.coords))
    tree = cKDTree(np.asarray(pts))
    print(f"  fault vertices for distance lookup: {len(pts)}")
    return tree


def predict_susceptibility(proc_dir, table_csv, faults_path, out_tif,
                           model_key="xgb", block_rows=400, pra_slope=(28.0, 60.0)):
    import pandas as pd
    from inventory.sampling import collect_rasters

    rasters = collect_rasters(proc_dir)
    df = pd.read_csv(table_csv)
    feats = [f for f in feature_columns(df) if f not in EXCLUDE]
    print(f"training final {model_key} on {len(df)} rows, {len(feats)} predictors "
          f"(excluded: {sorted(EXCLUDE)})")
    model = get_models()[model_key]
    model.fit(df[feats].values, df["label"].values)

    ref = rasters["slope"]
    with rasterio.open(ref) as src:
        profile = src.profile.copy()
        H, W, transform = src.height, src.width, src.transform

    tree = fault_distance_grid(profile, faults_path)

    # Layers reprojected from different source resolutions landed on slightly different grids
    # (land cover 10 m, MODIS 500 m, WorldClim 1 km). Window reads assume a shared grid, so those
    # are held in memory and looked up by world coordinate instead.
    srcs, offgrid = {}, {}
    for f in feats:
        if f == "dist_to_lineament":
            continue
        s = rasterio.open(rasters[f])
        if (s.height, s.width) == (H, W) and s.transform.almost_equals(transform, precision=1e-3):
            srcs[f] = s
        else:
            a = s.read(1).astype("float32")
            if s.nodata is not None:
                a[a == s.nodata] = np.nan
            offgrid[f] = (a, ~s.transform)
            s.close()
    if offgrid:
        print(f"  off-grid layers looked up by coordinate: {sorted(offgrid)}")

    def sample_offgrid(name, xs, ys):
        a, inv = offgrid[name]
        c, r = inv * (xs, ys)
        r = np.floor(r).astype(np.int64)
        c = np.floor(c).astype(np.int64)
        ok = (r >= 0) & (r < a.shape[0]) & (c >= 0) & (c < a.shape[1])
        out = np.full(xs.shape, np.nan, "float32")
        out[ok] = a[r[ok], c[ok]]
        return out

    profile.update(dtype="float32", count=1, nodata=np.nan, compress="deflate",
                   tiled=False)
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)

    os.makedirs(os.path.dirname(out_tif), exist_ok=True)
    n_pra = 0
    with rasterio.open(out_tif, "w", **profile) as dst:
        for r0 in range(0, H, block_rows):
            nr = min(block_rows, H - r0)
            win = Window(0, r0, W, nr)
            cols = {}
            for f, s in srcs.items():
                a = s.read(1, window=win).astype("float32")
                if s.nodata is not None:
                    a[a == s.nodata] = np.nan
                cols[f] = a
            slope = cols["slope"]
            pra = np.isfinite(slope) & (slope >= pra_slope[0]) & (slope <= pra_slope[1])
            out = np.full((nr, W), np.nan, "float32")
            idx = np.nonzero(pra)
            if idx[0].size:
                xs = transform.c + (idx[1] + 0.5) * transform.a
                ys = transform.f + (r0 + idx[0] + 0.5) * transform.e
                X = np.empty((idx[0].size, len(feats)), "float32")
                for j, f in enumerate(feats):
                    if f == "dist_to_lineament":
                        X[:, j] = tree.query(np.column_stack([xs, ys]), k=1)[0]
                    elif f in offgrid:
                        X[:, j] = sample_offgrid(f, xs, ys)
                    else:
                        X[:, j] = cols[f][idx]
                ok = np.isfinite(X).all(axis=1)
                if ok.any():
                    pr = model.predict_proba(X[ok])[:, 1]
                    vals = np.full(idx[0].size, np.nan, "float32")
                    vals[ok] = pr
                    out[idx] = vals
                n_pra += int(ok.sum())
            dst.write(out, 1, window=win)
            if (r0 // block_rows) % 4 == 0:
                print(f"    rows {r0}/{H}", flush=True)
    for s in srcs.values():
        s.close()
    print(f"  predicted {n_pra:,} release-area pixels ({n_pra*900/1e6:,.0f} km2) -> {out_tif}")
    return out_tif


def classify_map(sus_tif, out_tif, n_classes=5):
    """Bin the susceptibility index into classes WITHIN potential release areas only."""
    with rasterio.open(sus_tif) as src:
        a = src.read(1)
        profile = src.profile.copy()
    v = a[np.isfinite(a)]
    try:
        import mapclassify
        brk = list(mapclassify.NaturalBreaks(v, k=n_classes).bins)
        method = "natural breaks (Jenks)"
    except Exception:
        brk = list(np.quantile(v, np.linspace(1 / n_classes, 1, n_classes)))
        method = "quantiles (mapclassify unavailable)"
    cls = np.full(a.shape, np.nan, "float32")
    prev = -np.inf
    for i, b in enumerate(brk, start=1):
        cls[np.isfinite(a) & (a > prev) & (a <= b)] = i
        prev = b
    profile.update(dtype="float32", count=1, nodata=np.nan)
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(cls, 1)
    names = ["very low", "low", "moderate", "high", "very high"][:n_classes]
    print(f"  classified by {method}; breaks {[round(b,3) for b in brk]}")
    tot = np.isfinite(cls).sum()
    for i, nm in enumerate(names, start=1):
        n = int((cls == i).sum())
        print(f"    {nm:<10} {n:>10,} px  {n*900/1e6:>8,.0f} km2  {100*n/tot:>5.1f}% of release area")
    return out_tif, brk
