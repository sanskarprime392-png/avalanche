"""
features/multiscale.py — terrain descriptors at several spatial scales.

A 30 m pixel is smaller than the process. Whether a slope releases depends on its configuration over
tens to hundreds of metres: the overall steepness of the face, how convex the roll-over is, how
rough the bed is at the scale of a slab. Single-scale derivatives computed in a 3x3 window answer
none of that, so the usual predictor stack describes the pixel rather than the slope.

For each radius we compute the mean slope over that neighbourhood, the topographic position (the
cell's elevation relative to the neighbourhood mean) and the roughness (its standard deviation).
Radii are chosen to bracket the process: ~90 m is slab scale, ~300 m is start-zone scale, ~900 m is
the scale of the whole face and its catchment.
"""
import os

import numpy as np
import rasterio
from scipy import ndimage

RADII_M = (90, 300, 900)


def _nanmean_filter(a, size):
    """Mean over a window, ignoring NaN, without letting NaN spread."""
    v = np.nan_to_num(a, nan=0.0)
    m = np.isfinite(a).astype("float32")
    s = ndimage.uniform_filter(v, size=size, mode="nearest")
    c = ndimage.uniform_filter(m, size=size, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / c
    out[c < 1e-6] = np.nan
    return out.astype("float32")


def compute_multiscale(dem_path, slope_path, out_dir, radii_m=RADII_M):
    os.makedirs(out_dir, exist_ok=True)
    with rasterio.open(dem_path) as d:
        dem = d.read(1).astype("float32")
        if d.nodata is not None:
            dem[dem == d.nodata] = np.nan
        profile = d.profile.copy()
        px = abs(d.transform.a)
    with rasterio.open(slope_path) as s:
        slope = s.read(1).astype("float32")
        if s.nodata is not None:
            slope[slope == s.nodata] = np.nan

    profile.update(dtype="float32", count=1, nodata=np.nan, compress="deflate", tiled=False)
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    written = []
    for r in radii_m:
        n = max(int(round(2 * r / px)) | 1, 3)
        mz = _nanmean_filter(dem, n)
        tpi = (dem - mz).astype("float32")
        msl = _nanmean_filter(slope, n)
        sq = _nanmean_filter(dem * dem, n)
        with np.errstate(invalid="ignore"):
            rough = np.sqrt(np.maximum(sq - mz * mz, 0)).astype("float32")
        for name, arr in ((f"tpi_{r}m", tpi), (f"slope_mean_{r}m", msl),
                          (f"rough_{r}m", rough)):
            p = os.path.join(out_dir, name + ".tif")
            with rasterio.open(p, "w", **profile) as dst:
                dst.write(arr, 1)
            written.append(name)
        print(f"  radius {r} m (window {n} px): tpi, slope_mean, rough", flush=True)
    print(f"wrote {len(written)} multiscale rasters -> {out_dir}")
    return written
