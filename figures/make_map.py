"""
figures/make_map.py — the paired susceptibility and uncertainty map.

The second panel is the point. A susceptibility map is normally published as a single surface with
no statement of confidence, so a reader cannot tell a well-constrained hotspot from a guess. Here
the ensemble spread is mapped alongside the prediction, at the same resolution, so the map says
where it should and should not be trusted. Both panels are masked to potential release areas
(slope 28-60 deg), because classifying valley floors as low susceptibility inflates the denominator
behind any "x% of the basin" statistic.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LightSource, LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figures.pubstyle import set_pub_style, savefig

DATA = r"C:\Users\Sanskar\Documents\avalanche_data"
RES = os.path.join(DATA, "results")
FIGS = os.path.join(DATA, "figures")

# settlements and sites named in the regional avalanche literature
PLACES = [("Manali", 77.189, 32.243), ("Solang", 77.158, 32.317),
          ("Keylong", 77.030, 32.573), ("Darcha", 77.192, 32.700),
          ("Baralacha La", 77.409, 32.760), ("Chandra Tal", 77.618, 32.479)]

SUS_CMAP = LinearSegmentedColormap.from_list(
    "sus", ["#2c5aa0", "#6ba3c7", "#f2e6a8", "#e8893c", "#b02418"])
UNC_CMAP = LinearSegmentedColormap.from_list(
    "unc", ["#f7f7f7", "#c9b8d8", "#8c6bb1", "#4d1f7a"])


def read(path, stride):
    with rasterio.open(path) as s:
        a = s.read(1)[::stride, ::stride].astype("float32")
        return a, s.transform, s.crs


def main(stride=3):
    set_pub_style()
    sus, T, crs = read(os.path.join(RES, "susceptibility_v2.tif"), stride)
    unc, _, _ = read(os.path.join(RES, "uncertainty_v2.tif"), stride)
    with rasterio.open(os.path.join(DATA, "processed", "base_utm",
                                    "base_nasadem_elevation.tif")) as d:
        dem = d.read(1)[::stride, ::stride].astype("float32")
        if d.nodata is not None:
            dem[dem == d.nodata] = np.nan

    h, w = sus.shape
    dem = dem[:h, :w]
    px = abs(T.a) * stride
    left, top = T.c, T.f
    extent = [left, left + w * px, top - h * px, top]

    # hillshade gives the panels terrain context without competing with the data
    ls = LightSource(azdeg=315, altdeg=45)
    shade = ls.hillshade(np.nan_to_num(dem, nan=np.nanmin(dem)), vert_exag=2.2,
                         dx=px, dy=px)

    from pyproj import Transformer
    fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    fig, axs = plt.subplots(1, 2, figsize=(11.6, 5.3))
    panels = [(sus, SUS_CMAP, 0, 1, "Avalanche susceptibility index",
               "a   Susceptibility"),
              (unc, UNC_CMAP, 0, float(np.nanpercentile(unc, 99)),
               "Ensemble standard deviation", "b   Uncertainty")]

    for ax, (arr, cmap, vmin, vmax, cbl, title) in zip(axs, panels):
        ax.imshow(shade, cmap="gray", extent=extent, alpha=0.95, zorder=1,
                  interpolation="bilinear")
        im = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, extent=extent,
                       vmin=vmin, vmax=vmax, alpha=0.82, zorder=2,
                       interpolation="nearest")
        for name, lon, lat in PLACES:
            x, y = fwd.transform(lon, lat)
            ax.plot(x, y, "o", ms=3.2, mfc="white", mec="black", mew=0.7, zorder=5)
            ax.annotate(name, (x, y), xytext=(5, 3), textcoords="offset points",
                        fontsize=7, zorder=6,
                        path_effects=None, color="black",
                        bbox=dict(fc="white", ec="none", alpha=0.72, pad=0.9))
        cb = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.015)
        cb.set_label(cbl, fontsize=7.5)
        cb.ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        # scale bar
        bar = 20000.0
        x0 = extent[0] + 0.06 * (extent[1] - extent[0])
        y0 = extent[2] + 0.07 * (extent[3] - extent[2])
        ax.plot([x0, x0 + bar], [y0, y0], color="black", lw=2.4, zorder=7,
                solid_capstyle="butt")
        ax.text(x0 + bar / 2, y0 + 0.016 * (extent[3] - extent[2]), "20 km",
                ha="center", fontsize=7, zorder=7)
        # north arrow
        xn = extent[1] - 0.055 * (extent[1] - extent[0])
        yn = extent[2] + 0.07 * (extent[3] - extent[2])
        ax.annotate("N", xy=(xn, yn + 0.085 * (extent[3] - extent[2])), xytext=(xn, yn),
                    ha="center", fontsize=8, zorder=7,
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.1))

    fig.tight_layout()
    savefig(fig, "fig_map_susceptibility_uncertainty", outdir=FIGS)

    fin = np.isfinite(sus) & np.isfinite(unc)
    conf = fin & (unc <= 0.10)
    hi = fin & (sus >= 0.75)
    print(f"  release-area pixels plotted: {int(fin.sum()):,}")
    print(f"  high susceptibility (>=0.75): {100*hi.sum()/fin.sum():.1f}% of release area")
    print(f"  confidently predicted (sd<=0.10): {100*conf.sum()/fin.sum():.1f}%")
    print(f"  high AND confident: {100*(hi & conf).sum()/fin.sum():.1f}%")


if __name__ == "__main__":
    main()
