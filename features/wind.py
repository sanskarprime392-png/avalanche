"""
features/wind.py — Winstral Sx wind-shelter / exposure index from the DEM.

Why this matters: slab avalanches release where wind DEPOSITS snow, and the dominant control is
whether a slope sits windward (scoured) or leeward (loaded) of the prevailing wind. Every published
IHR avalanche-susceptibility study we are reproducing uses terrain, climate and snow-cover proxies
but no wind-redistribution term at all, so the main loading mechanism is simply absent from the
feature set.

Winstral et al. (2002) Sx:

    Sx(theta, dmax) = max over cells along the upwind search vector of
                      atan( (z_i - z_0) / horizontal_distance_i )

Positive Sx  -> terrain upwind rises above the cell: SHELTERED, a deposition zone (lee).
Negative Sx  -> cell stands above its upwind fetch: EXPOSED, a scour zone (windward).

Sxmean averages Sx over a wind sector (the prevailing winter flow in the Western Himalaya is
westerly, driven by western disturbances), and Sbsx = Sx(lee) - Sx(windward) separates loading from
scouring more sharply than either alone.
"""
import numpy as np


def _shift(a, dr, dc, fill=np.nan):
    """Shift an array by whole pixels without wrapping."""
    out = np.full_like(a, fill)
    r0, r1 = max(dr, 0), a.shape[0] + min(dr, 0)
    c0, c1 = max(dc, 0), a.shape[1] + min(dc, 0)
    out[r0:r1, c0:c1] = a[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    return out


def winstral_sx(dem, pixel_m, wind_from_deg=270.0, dmax_m=300.0, step_m=None):
    """Sx for a single upwind direction. `wind_from_deg` is the direction the wind blows FROM
    (270 = westerly). Returns degrees; positive = sheltered/lee, negative = exposed/windward."""
    step_m = step_m or pixel_m
    # unit vector pointing UPWIND (toward where the wind comes from), in grid coordinates.
    # north is -row, east is +col
    th = np.deg2rad(wind_from_deg)
    ux, uy = np.sin(th), np.cos(th)              # east, north components
    n_steps = max(int(dmax_m / step_m), 1)

    best = np.full(dem.shape, -np.inf, dtype="float32")
    for i in range(1, n_steps + 1):
        d = i * step_m
        dc = int(round(ux * d / pixel_m))
        dr = int(round(-uy * d / pixel_m))
        if dr == 0 and dc == 0:
            continue
        zi = _shift(dem, dr, dc)
        with np.errstate(invalid="ignore"):
            ang = np.degrees(np.arctan((zi - dem) / d))
        np.maximum(best, np.nan_to_num(ang, nan=-np.inf), out=best)
    best[~np.isfinite(dem)] = np.nan
    best[best == -np.inf] = np.nan
    return best


def sx_sector(dem, pixel_m, centre_deg=270.0, spread_deg=30.0, n=5, dmax_m=300.0):
    """Sx averaged over a wind sector (more stable than a single azimuth)."""
    angles = np.linspace(centre_deg - spread_deg, centre_deg + spread_deg, n)
    acc = np.zeros(dem.shape, "float32")
    cnt = np.zeros(dem.shape, "float32")
    for a in angles:
        s = winstral_sx(dem, pixel_m, wind_from_deg=float(a), dmax_m=dmax_m)
        ok = np.isfinite(s)
        acc[ok] += s[ok]
        cnt[ok] += 1
    out = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan).astype("float32")
    return out


def compute_wind_features(dem_path, out_dir, wind_from_deg=270.0, dmax_m=300.0):
    """Write sx_lee (sheltering from the prevailing wind) and sbsx (lee minus windward)."""
    import os
    import rasterio

    os.makedirs(out_dir, exist_ok=True)
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype("float32")
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan
        profile = src.profile.copy()
        px = abs(src.transform.a)

    sx_up = sx_sector(dem, px, centre_deg=wind_from_deg, dmax_m=dmax_m)
    sx_dn = sx_sector(dem, px, centre_deg=(wind_from_deg + 180) % 360, dmax_m=dmax_m)
    sbsx = sx_up - sx_dn

    profile.update(dtype="float32", count=1, nodata=np.nan, tiled=False)
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    paths = {}
    for name, arr in (("sx_lee", sx_up), ("sbsx", sbsx)):
        p = os.path.join(out_dir, name + ".tif")
        with rasterio.open(p, "w", **profile) as dst:
            dst.write(arr, 1)
        paths[name] = p
        v = arr[np.isfinite(arr)]
        print(f"  {name}: median {np.median(v):+.2f} deg, "
              f"p5 {np.percentile(v,5):+.2f}, p95 {np.percentile(v,95):+.2f} -> {p}")
    return paths
