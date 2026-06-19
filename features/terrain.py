"""
features/terrain.py — terrain & hydrology predictors computed LOCALLY from the DEM.

These are the factors GEE can't do well: curvature, ruggedness, and flow-accumulation hydrology.
Engine: WhiteboxTools (pip install whitebox; runs fine on Colab) + numpy/scipy for VRM, eastness/
northness and roughness. Same algorithm family as the paper's SAGA workflow.

Outputs (GeoTIFF, one per factor), matching the paper's DEM-derived factors:
  topography: slope, aspect, eastness, northness, roughness, plan_curvature, profile_curvature,
              tri, tpi, vrm, valley_depth
  hydrology:  twi, spi
  (elevation itself is just the input DEM — no computation needed.)

Run (Colab):
    !pip -q install whitebox rasterio scipy
    from features.terrain import compute_terrain
    compute_terrain(DEM_TIF, OUT_DIR)

Note: a couple of WhiteboxTools argument strings (e.g. d-inf out_type) vary slightly across
`whitebox` versions; if a call errors, the message names the accepted values — easy swap.
"""
import os
import numpy as np
import rasterio
from scipy.ndimage import uniform_filter
import whitebox

TOPO = ["slope", "aspect", "eastness", "northness", "roughness",
        "plan_curvature", "profile_curvature", "tri", "tpi", "vrm", "valley_depth"]
HYDRO = ["twi", "spi"]


def _wbt(work_dir):
    wbt = whitebox.WhiteboxTools()
    wbt.set_working_dir(work_dir)
    wbt.set_verbose_mode(False)
    return wbt


def _read(path):
    with rasterio.open(path) as src:
        return src.read(1).astype("float32"), src.profile


def _write(path, arr, profile):
    profile = profile.copy()
    profile.update(dtype="float32", count=1, nodata=np.nan)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype("float32"), 1)


def _vrm(slope_deg, aspect_deg, window=3):
    """Sappington (2007) Vector Ruggedness Measure from slope & aspect."""
    s, a = np.deg2rad(slope_deg), np.deg2rad(aspect_deg)
    x, y, z = np.sin(s) * np.sin(a), np.sin(s) * np.cos(a), np.cos(s)
    n = window * window
    sx, sy, sz = (uniform_filter(v, window) * n for v in (x, y, z))
    r = np.sqrt(sx**2 + sy**2 + sz**2)
    return 1.0 - r / n


def compute_terrain(dem_path, out_dir, stream_threshold=100.0, tpi_window=11):
    """Compute all terrain + hydrology factors from `dem_path` into `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)
    wbt = _wbt(out_dir)
    dem = os.path.abspath(dem_path)
    out = lambda n: os.path.join(out_dir, n + ".tif")

    # 0) hydrologically condition the DEM (breach preserves flow paths better than fill)
    filled = os.path.join(out_dir, "_dem_breached.tif")
    wbt.breach_depressions_least_cost(dem, filled, dist=100)

    # 1) WhiteboxTools primary derivatives
    wbt.slope(dem, out("slope"), units="degrees")
    wbt.aspect(dem, out("aspect"))
    wbt.plan_curvature(dem, out("plan_curvature"))
    wbt.profile_curvature(dem, out("profile_curvature"))
    wbt.ruggedness_index(dem, out("tri"))                                   # Riley TRI
    wbt.diff_from_mean_elev(dem, out("tpi"), filterx=tpi_window, filtery=tpi_window)  # raw TPI

    # 2) eastness / northness / VRM / roughness (numpy on the rasters)
    slope, prof = _read(out("slope"))
    aspect, _ = _read(out("aspect"))
    a = np.deg2rad(aspect)
    _write(out("eastness"), np.sin(a), prof)
    _write(out("northness"), np.cos(a), prof)
    _write(out("vrm"), _vrm(slope, aspect), prof)

    elev, _ = _read(dem)
    mean = uniform_filter(elev, 3)
    roughness = np.sqrt(np.clip(uniform_filter(elev**2, 3) - mean**2, 0, None))  # focal std of elev
    _write(out("roughness"), roughness, prof)

    # 3) hydrology: D-infinity flow accumulation -> TWI, SPI, valley depth
    sca = os.path.join(out_dir, "_sca.tif")
    wbt.d_inf_flow_accumulation(filled, sca, out_type="Specific Contributing Area")
    wbt.wetness_index(sca, out("slope"), out("twi"))                        # TWI = ln(SCA / tan slope)

    sca_arr, _ = _read(sca)
    spi = sca_arr * np.tan(np.deg2rad(np.clip(slope, 0.001, None)))         # SPI = SCA * tan(slope)
    _write(out("spi"), spi, prof)

    streams = os.path.join(out_dir, "_streams.tif")
    wbt.extract_streams(sca, streams, threshold=stream_threshold)
    wbt.elevation_above_stream(dem, streams, out("valley_depth"))          # ~ valley depth

    for tmp in (filled, sca, streams):
        if os.path.exists(tmp):
            os.remove(tmp)

    print(f"wrote {len(TOPO) + len(HYDRO)} factors -> {out_dir}")
    return {n: out(n) for n in TOPO + HYDRO}


if __name__ == "__main__":
    import config
    dem = os.path.join(config.PATHS["predictors"], "dem", "nasadem_elevation.tif")
    compute_terrain(dem, os.path.join(config.PATHS["processed"], "terrain"))
