"""
gee/wet_dry.py — separate avalanche activity into wet-snow and dry-snow regimes.

The two regimes have different release physics, different seasonality and, importantly for runout
modelling, different friction parameters — yet published IHR susceptibility work treats avalanches
as a single class.

Sentinel-1 separates them because the two processes move backscatter in OPPOSITE directions:

  * DRY-snow avalanche debris roughens the surface        -> backscatter INCREASES
  * WET snow contains liquid water, which absorbs C-band  -> backscatter DECREASES

Wet snow is detected with the standard Nagler & Rott (2000) ratio: a drop of >= 3 dB relative to a
dry-snow winter reference. Our deposit detector already keeps only increases, so the wet-snow signal
is currently discarded; here it is used to label the CONDITIONS under which each detection occurred.

A deposit found while the surrounding slope is in a wet-snow state is a wet-snow avalanche; one
found in a cold, dry-snow scene is a dry/slab event.
"""
import ee

from .s1_inventory import _s1, _composite, DEM

WET_DROP_DB = 3.0          # Nagler & Rott wet-snow threshold


def wet_snow_mask(aoi, rel_orbit, ref_win, act_win, orbit_pass="DESCENDING"):
    """Wet-snow extent during `act_win`, relative to a dry mid-winter reference."""
    ref = _composite(_s1(aoi, ref_win[0], ref_win[1], orbit_pass, rel_orbit))
    act = _composite(_s1(aoi, act_win[0], act_win[1], orbit_pass, rel_orbit))
    drop = ref.subtract(act)                       # positive = backscatter fell = wetting
    slope = ee.Terrain.slope(DEM)
    return (drop.select("VH").gte(WET_DROP_DB)
            .And(slope.gte(10))                    # exclude flat water bodies
            .rename("wet"))


def classify_regime(aoi, rel_orbit, ref_win, act_win, deposits, orbit_pass="DESCENDING",
                    search_m=500):
    """Tag each deposit polygon as wet-snow or dry-snow based on surrounding conditions."""
    wet = wet_snow_mask(aoi, rel_orbit, ref_win, act_win, orbit_pass).unmask(0)

    def tag(f):
        zone = f.geometry().buffer(search_m)
        frac = wet.reduceRegion(ee.Reducer.mean(), zone, scale=30,
                                maxPixels=int(1e9), bestEffort=True).get("wet")
        frac = ee.Number(ee.Algorithms.If(frac, frac, 0))
        return f.set("wet_fraction", frac,
                     "regime", ee.Algorithms.If(frac.gte(0.15), "wet", "dry"))

    return deposits.map(tag)


def seasonal_regime_summary(aoi, rel_orbit, years, orbit_pass="DESCENDING", scale=90):
    """How much of the detected activity is wet vs dry, month by month?

    Early season (Jan-Feb) should be dominated by dry/slab activity and late spring (Apr-May) by
    wet-snow activity; recovering that gradient is a physical check on the detector.
    """
    windows = {
        "Jan-Feb": ((12, 1), (2, 28)),
        "Mar":     ((12, 1), (3, 31)),
        "Apr":     ((12, 1), (4, 30)),
        "May":     ((12, 1), (5, 31)),
    }
    out = {}
    for label, (ref_md, act_md) in windows.items():
        wet_tot = dry_tot = 0.0
        for y in years:
            ref = (ee.Date.fromYMD(y - 1, ref_md[0], ref_md[1]),
                   ee.Date.fromYMD(y, 2, 10) if label == "Jan-Feb"
                   else ee.Date.fromYMD(y, 2, 25))
            start_m = {"Jan-Feb": 1, "Mar": 3, "Apr": 4, "May": 5}[label]
            act = (ee.Date.fromYMD(y, start_m, 1), ee.Date.fromYMD(y, act_md[0], act_md[1]))
            wet = wet_snow_mask(aoi, rel_orbit, ref, act, orbit_pass)
            stats = wet.unmask(0).rename("w").reduceRegion(
                ee.Reducer.sum(), aoi, scale=scale, maxPixels=int(1e12),
                bestEffort=True).getInfo()
            wet_tot += float(stats.get("w", 0) or 0)
        out[label] = wet_tot / max(len(years), 1)
    return out
