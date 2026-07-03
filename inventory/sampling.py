"""
inventory/sampling.py — build the model training table from the avalanche inventory.

What it does (and why):
  1. Loads presence points (inventory CSV, lon/lat WGS84) -> converts to the raster CRS (UTM 43N).
  2. TERRAIN-MATCHED negative sampling — the anchor paper drew absences from flat/vegetated/urban
     land, so its classifier partly learned "steep snowy terrain vs farmland" (inflated AUC).
     Here negatives come from avalanche-CAPABLE terrain (slope smin–smax), outside an exclusion
     buffer around presences, then matched to the presence (elevation x slope) distribution.
  3. SPATIAL BLOCKS — every point gets a block_id on a regular grid (default 10 km) so models can
     be evaluated with spatial block CV (GroupKFold) instead of leaky random CV.
  4. Samples every predictor raster at every point -> one tidy training table (CSV).

Usage (Colab):
    from inventory.sampling import build_training_table
    df = build_training_table(
        inv_csv="/content/drive/MyDrive/avalanche/data/raw/labels/inventory.csv",
        proc_dir="/content/drive/MyDrive/avalanche/data/processed",
        out_csv="/content/drive/MyDrive/avalanche/data/processed/training_table.csv")
"""
import os
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy as transform_xy
from pyproj import Transformer
from scipy.spatial import cKDTree

RASTER_CRS = "EPSG:32643"


def collect_rasters(proc_dir):
    """Map factor name -> raster path from the processed folders (terrain, distances, base_utm)."""
    paths = {}
    for sub in ("terrain", "distances", "base_utm"):
        for f in sorted(glob.glob(os.path.join(proc_dir, sub, "*.tif"))):
            name = os.path.basename(f)[:-4]
            if name.startswith("base_"):
                name = name[len("base_"):]
            if name == "nasadem_elevation":
                name = "elevation"
            paths[name] = f
    if not paths:
        raise FileNotFoundError(f"no rasters found under {proc_dir}/(terrain|distances|base_utm)")
    return paths


def sample_rasters(xy, rasters):
    """Sample every raster (nearest pixel) at Nx2 UTM coords -> DataFrame of features."""
    coords = [tuple(p) for p in np.asarray(xy)]
    out = {}
    for name, path in rasters.items():
        with rasterio.open(path) as src:
            if str(src.crs).upper() != RASTER_CRS:
                print(f"  ! skipping {name}: CRS {src.crs} != {RASTER_CRS}")
                continue
            vals = np.array([v[0] for v in src.sample(coords)], dtype="float64")
            if src.nodata is not None:
                vals[vals == src.nodata] = np.nan
            vals[~np.isfinite(vals)] = np.nan
            out[name] = vals
    return pd.DataFrame(out)


def load_presences(inv_csv):
    df = pd.read_csv(inv_csv)
    tr = Transformer.from_crs("EPSG:4326", RASTER_CRS, always_xy=True)
    df["x"], df["y"] = tr.transform(df["lon"].values, df["lat"].values)
    return df


def sample_negatives(dem_path, slope_path, presence_xy, n_ratio=1.0, buffer_m=1000.0,
                     smin=25.0, smax=55.0, n_candidates=60000,
                     elev_bin=250.0, slope_bin=5.0, seed=42):
    """Terrain-matched background absences. Returns Mx2 array of UTM coords."""
    rng = np.random.default_rng(seed)
    with rasterio.open(slope_path) as s:
        slope = s.read(1).astype("float32")
        if s.nodata is not None:
            slope[slope == s.nodata] = np.nan
        transform = s.transform
    with rasterio.open(dem_path) as d:
        elev = d.read(1).astype("float32")
        if d.nodata is not None:
            elev[elev == d.nodata] = np.nan
    assert slope.shape == elev.shape, "slope and DEM must share the same grid"

    ok = np.isfinite(slope) & np.isfinite(elev) & (slope >= smin) & (slope <= smax)
    rows, cols = np.nonzero(ok)
    take = rng.choice(rows.size, size=min(n_candidates, rows.size), replace=False)
    r, c = rows[take], cols[take]
    xs, ys = transform_xy(transform, r, c)
    cand = np.column_stack([np.asarray(xs), np.asarray(ys)])
    cslope, celev = slope[r, c], elev[r, c]

    # exclusion buffer around known avalanches (they are NOT reliable absences)
    dmin, _ = cKDTree(presence_xy).query(cand, k=1)
    keep = dmin > buffer_m
    cand, cslope, celev = cand[keep], cslope[keep], celev[keep]
    print(f"  candidates on capable terrain, outside {buffer_m:.0f} m buffer: {len(cand)}")

    # presence slope/elev for distribution matching
    p_feats = sample_rasters(presence_xy, {"slope": slope_path, "elevation": dem_path})
    p_slope, p_elev = p_feats["slope"].values, p_feats["elevation"].values

    e_edges = np.arange(np.nanmin(elev), np.nanmax(elev) + elev_bin, elev_bin)
    s_edges = np.arange(smin, smax + slope_bin, slope_bin)
    p_bin = np.digitize(p_elev, e_edges) * 1000 + np.digitize(p_slope, s_edges)
    c_bin = np.digitize(celev, e_edges) * 1000 + np.digitize(cslope, s_edges)

    n_target = int(round(len(presence_xy) * n_ratio))
    chosen, used = [], np.zeros(len(cand), bool)
    bins, counts = np.unique(p_bin[np.isfinite(p_elev)], return_counts=True)
    for b, n_b in zip(bins, counts):
        want = int(round(n_b * n_ratio))
        pool = np.nonzero((c_bin == b) & ~used)[0]
        pick = rng.choice(pool, size=min(want, len(pool)), replace=False) if len(pool) else []
        used[pick] = True
        chosen.extend(pick)
    short = n_target - len(chosen)
    if short > 0:  # top up from unused capable terrain if some bins were data-poor
        pool = np.nonzero(~used)[0]
        pick = rng.choice(pool, size=min(short, len(pool)), replace=False)
        chosen.extend(pick)
        print(f"  note: {short} negatives topped up outside matched bins")
    print(f"  negatives sampled: {len(chosen)} (target {n_target})")
    return cand[np.array(chosen, int)]


def assign_blocks(x, y, block_m=10000.0):
    bx = np.floor(np.asarray(x) / block_m).astype(int)
    by = np.floor(np.asarray(y) / block_m).astype(int)
    return [f"{a}_{b}" for a, b in zip(bx, by)]


def build_training_table(inv_csv, proc_dir, out_csv, n_ratio=1.0, buffer_m=1000.0,
                         block_m=10000.0, seed=42):
    rasters = collect_rasters(proc_dir)
    print(f"{len(rasters)} predictor rasters: {sorted(rasters)}")

    pres = load_presences(inv_csv)
    p_xy = pres[["x", "y"]].values
    print(f"{len(pres)} presence points from {inv_csv}")

    neg_xy = sample_negatives(rasters["elevation"], rasters["slope"], p_xy,
                              n_ratio=n_ratio, buffer_m=buffer_m, seed=seed)

    all_xy = np.vstack([p_xy, neg_xy])
    meta = pd.DataFrame({
        "x": all_xy[:, 0], "y": all_xy[:, 1],
        "label": np.r_[np.ones(len(p_xy), int), np.zeros(len(neg_xy), int)],
        "event": list(pres.get("event", pd.Series(["?"] * len(p_xy)))) + ["bg"] * len(neg_xy),
        "tier": list(pres.get("tier", pd.Series(["A"] * len(p_xy)))) + ["bg"] * len(neg_xy),
    })
    feats = sample_rasters(all_xy, rasters)
    df = pd.concat([meta.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)

    feat_cols = list(feats.columns)
    allnan = [c for c in feat_cols if df[c].isna().all()]
    if allnan:                                  # a broken layer must not nuke every row
        print(f"  ! all-NaN predictor(s) dropped — regenerate these: {allnan}")
        df = df.drop(columns=allnan)
        feat_cols = [c for c in feat_cols if c not in allnan]
    n0 = len(df)
    df = df.dropna(subset=feat_cols).reset_index(drop=True)
    if n0 - len(df):
        print(f"  dropped {n0 - len(df)} rows with nodata in some predictor")

    df["block_id"] = assign_blocks(df["x"], df["y"], block_m)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"training table: {len(df)} rows ({int(df.label.sum())} presence / "
          f"{int((1 - df.label).sum())} absence), {df.block_id.nunique()} spatial blocks -> {out_csv}")
    return df
