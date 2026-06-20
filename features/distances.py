"""
features/distances.py — distance-to-road / stream / lineament factors (in metres).

Roads & streams: fetched live from OpenStreetMap via OSMnx for the AOI (no manual OSM download).
Lineaments: GEM Global Active Faults (tectonic lineaments incl. MCT/MBT/MFT), clipped to AOI.
Each vector is rasterised onto the reference (UTM) DEM grid, then a Euclidean distance transform
gives distance in metres.

IMPORTANT: pass a PROJECTED reference grid (UTM 43N / EPSG:32643), not the raw 4326 export, or
the distances come out in degrees. Reproject first with features/reproject.py.

Run (Colab):
    !pip -q install osmnx geopandas rasterio scipy
    from features.distances import compute_distances
    compute_distances(UTM_DEM_TIF, OUT_DIR, gem_faults_path)
"""
import os
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
from scipy.ndimage import distance_transform_edt
import geopandas as gpd


def _osm_features(bbox_wsen, tags):
    """bbox_wsen = (west, south, east, north) in EPSG:4326. Robust to OSMnx 1.x / 2.x APIs."""
    import osmnx as ox
    w, s, e, n = bbox_wsen
    try:                                    # OSMnx >= 2.0
        return ox.features_from_bbox(bbox=(w, s, e, n), tags=tags)
    except TypeError:                       # OSMnx 1.x
        return ox.features_from_bbox(north=n, south=s, east=e, west=w, tags=tags)


def _distance_m(geoms, transform, shape, px):
    if not len(geoms):
        return np.full(shape, np.nan, "float32")
    burned = rasterize(((g, 1) for g in geoms), out_shape=shape, transform=transform,
                       fill=0, all_touched=True, dtype="uint8")
    if burned.sum() == 0:
        return np.full(shape, np.nan, "float32")
    return (distance_transform_edt(burned == 0) * px).astype("float32")


def _write(path, arr, profile):
    profile = profile.copy()
    profile.update(dtype="float32", count=1, nodata=np.nan)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)


def compute_distances(ref_tif, out_dir, gem_faults_path):
    os.makedirs(out_dir, exist_ok=True)
    with rasterio.open(ref_tif) as src:
        profile, transform, crs = src.profile, src.transform, src.crs
        shape, bounds = (src.height, src.width), src.bounds
        px = abs(transform.a)
    assert not crs.is_geographic, \
        "reference grid must be projected (UTM) — reproject first (features/reproject.py)"

    w, s, e, n = transform_bounds(crs, "EPSG:4326", *bounds)   # AOI bbox in 4326 for OSM / faults

    def lines_to_dist(gdf, name):
        gdf = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])].to_crs(crs)
        _write(os.path.join(out_dir, name), _distance_m(list(gdf.geometry), transform, shape, px), profile)
        print(f"wrote {name}  ({len(gdf)} features)")

    lines_to_dist(_osm_features((w, s, e, n), {"highway": True}), "dist_to_road.tif")
    lines_to_dist(_osm_features((w, s, e, n), {"waterway": True}), "dist_to_stream.tif")
    lines_to_dist(gpd.read_file(gem_faults_path, bbox=(w, s, e, n)), "dist_to_lineament.tif")
    print("\ndistance factors ->", out_dir)
