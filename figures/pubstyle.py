"""Publication-quality matplotlib defaults (300 DPI) + a save helper.

Usage:
    from figures.pubstyle import set_pub_style, savefig
    set_pub_style()
    fig, ax = plt.subplots()
    ...
    savefig(fig, "temperature_trend")     # writes 300-DPI PNG + vector PDF
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

import config

# Colour-blind-safe qualitative palette (Wong, 2011) for categorical series.
CB_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
              "#E69F00", "#56B4E9", "#F0E442", "#000000"]

# 5-class susceptibility ramp (very low -> very high), used across all ASM maps.
SUSCEPT_COLORS = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]
SUSCEPT_LABELS = ["Very low", "Low", "Medium", "High", "Very high"]


def set_pub_style(base_fontsize: int = 9):
    """Apply consistent, journal-ready rcParams. Call once per session/notebook."""
    mpl.rcParams.update({
        "figure.dpi": 150,            # on-screen
        "savefig.dpi": 300,           # exported rasters
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize + 1,
        "axes.labelsize": base_fontsize,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        "legend.fontsize": base_fontsize - 1,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "axes.prop_cycle": mpl.cycler(color=CB_PALETTE),
        "pdf.fonttype": 42,           # editable text in Illustrator
        "ps.fonttype": 42,
    })


def savefig(fig, name: str, formats=("png", "pdf"), outdir: str | None = None):
    """Save a figure to the project figures dir as 300-DPI PNG + vector PDF."""
    outdir = outdir or config.PATHS["figures"]
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for fmt in formats:
        p = os.path.join(outdir, f"{name}.{fmt}")
        fig.savefig(p)
        paths.append(p)
    print("saved:", ", ".join(paths))
    return paths
