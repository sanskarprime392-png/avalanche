"""
figures/make_study_area.py — the study-area figure.

Shows the terrain the paper is about: elevation over hillshade, the settlements and passes named in
the regional avalanche literature, the extent of potential release terrain, and a locator placing
the basins within the Western Himalaya. The elevation range is the point — avalanche terrain here
spans 1.1 to 6.3 km, and the slope-28-60 deg mask that the susceptibility map is restricted to
covers about half the area.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LightSource, LinearSegmentedColormap
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figures.pubstyle import set_pub_style, savefig

DATA = r"C:\Users\Sanskar\Documents\avalanche_data"
FIGS = os.path.join(DATA, "figures")

PLACES = [("Manali", 77.189, 32.243, "left"), ("Solang", 77.158, 32.317, "left"),
          ("Keylong", 77.030, 32.573, "right"), ("Darcha", 77.192, 32.700, "right"),
          ("Baralacha La", 77.409, 32.760, "left"), ("Chandra Tal", 77.618, 32.479, "left"),
          ("Udaipur", 76.675, 32.718, "right")]

TERRAIN = LinearSegmentedColormap.from_list(
    "terr", ["#2e6b46", "#7a9b53", "#c9b97e", "#a98468", "#8a7b76", "#e8e8ec", "#ffffff"])


def main(stride=3):
    set_pub_style()
    with rasterio.open(os.path.join(DATA, "processed", "base_utm",
                                    "base_nasadem_elevation.tif")) as d:
        dem = d.read(1)[::stride, ::stride].astype("float32")
        if d.nodata is not None:
            dem[dem == d.nodata] = np.nan
        # reprojection leaves zero-filled slivers along the edges; no ground here is at sea level
        dem[dem <= 1] = np.nan
        T, crs = d.transform, d.crs
    with rasterio.open(os.path.join(DATA, "processed", "terrain", "slope.tif")) as s:
        slope = s.read(1)[::stride, ::stride].astype("float32")
        if s.nodata is not None:
            slope[slope == s.nodata] = np.nan
    h, w = dem.shape
    slope = slope[:h, :w]
    px = abs(T.a) * stride
    left, top = T.c, T.f
    extent = [left, left + w * px, top - h * px, top]

    ls = LightSource(azdeg=315, altdeg=45)
    shade = ls.hillshade(np.nan_to_num(dem, nan=np.nanmin(dem)), vert_exag=2.0, dx=px, dy=px)

    fig = plt.figure(figsize=(8.0, 6.2))
    ax = fig.add_axes([0.06, 0.08, 0.76, 0.86])
    im = ax.imshow(np.ma.masked_invalid(dem), cmap=TERRAIN, extent=extent,
                   vmin=np.nanpercentile(dem, 1), vmax=np.nanpercentile(dem, 99.5),
                   interpolation="bilinear", zorder=1)
    ax.imshow(shade, cmap="gray", extent=extent, alpha=0.38, zorder=2, interpolation="bilinear")

    pra = np.where((slope >= 28) & (slope <= 60), 1.0, np.nan)

    from pyproj import Transformer
    fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    for name, lon, lat, side in PLACES:
        x, y = fwd.transform(lon, lat)
        ax.plot(x, y, "o", ms=4.2, mfc="white", mec="black", mew=0.8, zorder=6)
        dx = 7 if side == "left" else -7
        ax.annotate(name, (x, y), xytext=(dx, 4), textcoords="offset points", fontsize=8,
                    ha="left" if side == "left" else "right", zorder=7,
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.1))

    bar = 25000.0
    x0 = extent[0] + 0.05 * (extent[1] - extent[0])
    y0 = extent[2] + 0.055 * (extent[3] - extent[2])
    ax.plot([x0, x0 + bar], [y0, y0], color="black", lw=2.8, zorder=8, solid_capstyle="butt")
    ax.text(x0 + bar / 2, y0 + 0.014 * (extent[3] - extent[2]), "25 km", ha="center",
            fontsize=8, zorder=8)
    xn = extent[1] - 0.05 * (extent[1] - extent[0])
    yn = extent[2] + 0.055 * (extent[3] - extent[2])
    ax.annotate("N", xy=(xn, yn + 0.08 * (extent[3] - extent[2])), xytext=(xn, yn), ha="center",
                fontsize=9, zorder=8, arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)

    cax = fig.add_axes([0.845, 0.40, 0.022, 0.46])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Elevation (m a.s.l.)", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)

    # locator: the basins within the Western Himalaya, by graticule rather than a coastline
    lax = fig.add_axes([0.845, 0.08, 0.145, 0.22])
    lax.set_xlim(68, 90)
    lax.set_ylim(22, 38)
    lax.add_patch(Rectangle((68, 22), 22, 16, fc="#eef1f4", ec="none"))
    lax.plot([73.5, 79, 85, 89], [36, 33.5, 28.5, 27.5], color="#8a8f96", lw=1.1)
    lax.text(80.5, 34.6, "Himalaya", fontsize=6.5, color="#5c6b7a", rotation=-23)
    lax.add_patch(Rectangle((76.4, 31.9), 1.7, 1.3, fc="#b02418", ec="#6e1710", lw=0.8))
    lax.set_xticks([70, 80, 90])
    lax.set_yticks([25, 30, 35])
    lax.tick_params(labelsize=6)
    lax.set_title("Location", fontsize=7.5, pad=3)

    txt = (f"Potential release terrain (28–60°): "
           f"{np.isfinite(pra).sum() * (px ** 2) / 1e6:,.0f} km²\n"
           f"Elevation range: {np.nanmin(dem):,.0f}–{np.nanmax(dem):,.0f} m")
    ax.text(0.015, 0.985, txt, transform=ax.transAxes, va="top", fontsize=7.8, zorder=9,
            bbox=dict(fc="white", ec="#d2dae2", alpha=0.9, pad=4))

    savefig(fig, "fig_study_area", outdir=FIGS)
    print(f"  elevation {np.nanmin(dem):.0f}-{np.nanmax(dem):.0f} m")
    print(f"  release terrain {np.isfinite(pra).sum() * (px**2) / 1e6:,.0f} km2")


if __name__ == "__main__":
    main()
