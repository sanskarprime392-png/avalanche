"""
gee/s1_inventory.py — multi-year Sentinel-1 avalanche-deposit detection (the proper version).

Why the first pass failed (red across half the mountain): it mixed relative orbits (different
viewing geometry -> static terrain backscatter doesn't cancel -> seam artefacts) and flagged ALL
surface change (wet snow, wind, moisture), not just avalanche debris. This version fixes both:

  * SINGLE relative orbit -> identical geometry between dates, so static terrain backscatter cancels
    in the difference and there is no orbit seam.
  * SEASONAL change detection, one winter at a time, across many years (2018-2026).
  * DIRECTION matters: avalanche debris roughens the surface -> backscatter INCREASES. Wet snow
    LOWERS backscatter (opposite sign) -> we keep increases and drop the wet-snow decrease.
  * Speckle handled by multi-image median composites + focal filtering.
  * Spatial coherence: minimum connected-area filter on avalanche-capable slopes (25-60 deg).
  * MULTI-YEAR RECURRENCE (the key idea): sum seasonal detections across winters. A slope flagged
    in several winters is a real avalanche path; one-off change is not. Recurrence >= 2-3 gives a
    high-confidence inventory and slashes the manual-verification burden.

Still to add after this (separate steps): deposit -> release-zone routing (trace deposits upslope
to 28-60 deg concave start zones) and an optical verification pass on high-recurrence candidates.
Radiometric terrain flattening is a further refinement (largely unnecessary for same-orbit change).
"""
import ee

DEM = ee.Image("NASA/NASADEM_HGT/001").select("elevation")


def print_orbits(aoi, start, end, orbit_pass="DESCENDING"):
    """List relative-orbit numbers over the AOI so you can pick the one with the most coverage."""
    col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi).filterDate(start, end)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass)))
    hist = col.aggregate_histogram("relativeOrbitNumber_start").getInfo()
    print(f"{orbit_pass} relative orbits over AOI (orbit : image count):")
    for k, v in sorted(hist.items(), key=lambda x: -x[1]):
        print(f"   {k} : {int(v)}")
    print("-> set REL to the orbit with the most images")
    return hist


def _s1(aoi, start, end, orbit_pass, rel_orbit):
    return (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi).filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
            .filter(ee.Filter.eq("relativeOrbitNumber_start", rel_orbit))
            .select(["VV", "VH", "angle"]))


def _to_lin(db):
    return ee.Image(10.0).pow(db.divide(10.0))


def _composite(col):
    """Speckle-reduced composite: mean in linear power, back to dB, focal-median smoothed."""
    lin = col.select(["VV", "VH"]).map(lambda im: _to_lin(im))
    comp = lin.mean().log10().multiply(10.0).rename(["VV", "VH"])
    return comp.focal_median(40, "circle", "meters")


def seasonal_deposits(aoi, rel_orbit, ref_win, act_win, orbit_pass="DESCENDING",
                      vh_thresh=2.0, vv_cap=6.0, min_pixels=8):
    """One winter's avalanche deposits = same-orbit backscatter INCREASE on avalanche-capable slopes."""
    ref = _composite(_s1(aoi, ref_win[0], ref_win[1], orbit_pass, rel_orbit))
    act_col = _s1(aoi, act_win[0], act_win[1], orbit_pass, rel_orbit)
    act = _composite(act_col)
    change = act.subtract(ref)                          # +dB = roughening = candidate debris

    slope = ee.Terrain.slope(DEM)
    angle = act_col.select("angle").mean()
    valid = angle.gt(25).And(angle.lt(45))              # drop layover (near range) & shadow (far range)

    debris = (change.select("VH").gt(vh_thresh)         # cross-pol increase = surface roughening
              .And(change.select("VV").gt(vh_thresh - 1.0))
              .And(change.select("VV").lt(vv_cap))      # cap: extreme jumps are usually artefacts
              .And(slope.gte(25)).And(slope.lte(60))    # avalanche-capable terrain
              .And(valid))
    debris = debris.selfMask().connectedPixelCount(50, True).gte(min_pixels)  # remove speckle blobs
    return debris.selfMask().rename("deposit")


def multiyear_recurrence(aoi, years, rel_orbit, orbit_pass="DESCENDING",
                         ref_md=((12, 1), (2, 10)), act_md=((3, 1), (5, 15)), **kw):
    """Sum seasonal deposits across winters -> recurrence (0..N). High recurrence = confident path.

    ref_md / act_md are ((start_month, day), (end_month, day)); the reference starts in the PREVIOUS
    calendar year (early winter) and the activity window is the melt/avalanche season of year y.
    """
    def one_year(y):
        y = ee.Number(y)
        ref = (ee.Date.fromYMD(y.subtract(1), ref_md[0][0], ref_md[0][1]),
               ee.Date.fromYMD(y, ref_md[1][0], ref_md[1][1]))
        act = (ee.Date.fromYMD(y, act_md[0][0], act_md[0][1]),
               ee.Date.fromYMD(y, act_md[1][0], act_md[1][1]))
        return seasonal_deposits(aoi, rel_orbit, ref, act, orbit_pass, **kw).unmask(0)

    stack = ee.ImageCollection(ee.List(years).map(one_year))
    return stack.sum().rename("recurrence").selfMask()


def export_candidates(recurrence, aoi, min_recurrence=2, folder="avalanche_gee_exports",
                      name="s1_avalanche_recurrence"):
    """Vectorise high-recurrence deposits (candidate avalanche paths) and export to Drive."""
    keep = recurrence.gte(min_recurrence).selfMask()
    fc = keep.reduceToVectors(geometry=aoi, scale=10, geometryType="polygon",
                              eightConnected=True, maxPixels=int(1e13))
    task = ee.batch.Export.table.toDrive(collection=fc, description=name, folder=folder,
                                         fileFormat="GeoJSON")
    task.start()
    print("export started:", name)
    return task
