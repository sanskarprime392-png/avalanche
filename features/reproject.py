"""
features/reproject.py — reproject raw GEE exports (EPSG:4326, degrees) to a projected metric CRS
(UTM 43N / EPSG:32643). Terrain analysis (slope/curvature/flow) and distance transforms MUST run
in metres, not degrees, or the results are geometrically wrong.

Nearest resampling for categorical layers (land cover); bilinear for continuous (DEM, climate, snow).
"""
import os
import glob
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling


def reproject_raster(src_path, dst_path, dst_crs="EPSG:32643", res=30,
                     resampling=Resampling.bilinear):
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=res)
        profile = src.profile.copy()
        profile.update(crs=dst_crs, transform=transform, width=width, height=height)
        with rasterio.open(dst_path, "w", **profile) as dst:
            for b in range(1, src.count + 1):
                reproject(rasterio.band(src, b), rasterio.band(dst, b),
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=transform, dst_crs=dst_crs, resampling=resampling)
    return dst_path


def reproject_base_layers(in_dir, out_dir, dst_crs="EPSG:32643", res=30):
    """Reproject every base_*.tif in `in_dir`; nearest for land cover, bilinear otherwise."""
    os.makedirs(out_dir, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(in_dir, "base_*.tif"))):
        categorical = "worldcover" in os.path.basename(f).lower()
        rs = Resampling.nearest if categorical else Resampling.bilinear
        reproject_raster(f, os.path.join(out_dir, os.path.basename(f)), dst_crs, res, rs)
        print(f"reprojected {os.path.basename(f)} -> {dst_crs} ({'nearest' if categorical else 'bilinear'})")
    return out_dir
