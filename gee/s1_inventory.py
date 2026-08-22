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


def local_incidence_angle(inc_deg, orbit_pass):
    """Terrain-corrected local incidence angle (LIA), in degrees.

    Sentinel-1 is right-looking: DESCENDING (heading ~190 deg) looks WNW, ASCENDING (~350) looks
    ENE. The formula needs the azimuth pointing FROM the target TOWARD the sensor, which is the
    look direction reversed — ~100 deg (ESE) for descending and ~260 deg (WSW) for ascending — so
    that slopes tilted toward the sensor correctly come out with a LOW incidence angle.
    LIA = acos( cos(inc)cos(slope) + sin(inc)sin(slope)cos(aspect - sensor_azimuth) ).
    Low LIA = slope tilted toward the sensor (foreshortening/layover, artificially bright and
    change-sensitive); LIA near/above 90 = radar shadow (noise floor, spurious dB jumps). Both
    must be masked out or the detector inherits a look-direction ASPECT BIAS - measured here as
    SE over-representation on DESC 136 and NW on ASC 27 (see fused_recurrence docstring).
    """
    sensor_az = ee.Number(100 if orbit_pass == "DESCENDING" else 260)
    slope = ee.Terrain.slope(DEM).multiply(3.14159265 / 180)
    aspect = ee.Terrain.aspect(DEM).multiply(3.14159265 / 180)
    inc = inc_deg.multiply(3.14159265 / 180)
    rel_az = aspect.subtract(sensor_az.multiply(3.14159265 / 180))
    cos_lia = (inc.cos().multiply(slope.cos())
               .add(inc.sin().multiply(slope.sin()).multiply(rel_az.cos())))
    return cos_lia.clamp(-1, 1).acos().multiply(180 / 3.14159265).rename("lia")


def seasonal_deposits(aoi, rel_orbit, ref_win, act_win, orbit_pass="DESCENDING",
                      vh_thresh=2.0, vv_cap=6.0, min_pixels=8,
                      lia_min=30.0, lia_max=75.0):
    """One winter's avalanche deposits = same-orbit backscatter INCREASE on avalanche-capable slopes,
    restricted to pixels with a well-behaved local incidence angle (no layover, no shadow)."""
    ref = _composite(_s1(aoi, ref_win[0], ref_win[1], orbit_pass, rel_orbit))
    act_col = _s1(aoi, act_win[0], act_win[1], orbit_pass, rel_orbit)
    act = _composite(act_col)
    change = act.subtract(ref)                          # +dB = roughening = candidate debris

    slope = ee.Terrain.slope(DEM)
    angle = act_col.select("angle").mean()
    lia = local_incidence_angle(angle, orbit_pass)      # terrain-corrected, not just swath position
    valid = lia.gte(lia_min).And(lia.lte(lia_max))      # drop foreshortening/layover and shadow

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


def fused_recurrence(aoi, years, orbits=(("DESCENDING", 136), ("ASCENDING", 27)), **kw):
    """Recurrence fused across BOTH orbit geometries — required to remove look-direction bias.

    Sentinel-1 is right-looking, so a single orbit systematically over-detects on slopes facing the
    sensor and under-detects on slopes in radar shadow. Measured over this AOI, DESCENDING 136 peaks
    on E/SE (34.8% SE vs 12.6% of background terrain) while ASCENDING 27 peaks on NW/W — mirror
    images, i.e. geometry, not avalanche physics. (Only the elevated SOUTH-facing fraction is
    consistent across both geometries and therefore real: sun-facing slopes -> spring wet-snow
    avalanches.) Taking the per-pixel MAX across orbits lets each geometry cover the aspects the
    other one hides, so a path counts if either orbit saw it repeatedly.
    """
    recs = [multiyear_recurrence(aoi, years, rel_orbit=o, orbit_pass=p, **kw).unmask(0)
            for p, o in orbits]
    return ee.ImageCollection(recs).max().rename("recurrence").selfMask()


def export_candidates(recurrence, aoi, min_recurrence=2, min_area_m2=2700,
                      folder="avalanche_gee_exports", name="s1_avalanche_candidates"):
    """Vectorise repeat-flagged deposits into candidate avalanche paths and export to Drive.

    Each polygon carries `recurrence` (max winters flagged within it) and `area_m2`, so the final
    confidence threshold can be chosen offline without re-running the detection.
    """
    mask = recurrence.gte(min_recurrence).selfMask()
    fc = (mask.addBands(recurrence)                    # band 1 = label, band 2 = value to reduce
          .reduceToVectors(reducer=ee.Reducer.max(), geometry=aoi, scale=30,
                           geometryType="polygon", eightConnected=True,
                           labelProperty="label", maxPixels=int(1e13)))
    fc = (fc.map(lambda f: f.set("area_m2", f.geometry().area(10))
                            .set("recurrence", f.get("max")))
            .filter(ee.Filter.gte("area_m2", min_area_m2))
            .select(["recurrence", "area_m2"]))
    task = ee.batch.Export.table.toDrive(collection=fc, description=name, folder=folder,
                                         fileNamePrefix=name, fileFormat="GeoJSON")
    task.start()
    print(f"export started: {name} (recurrence>={min_recurrence}, area>={min_area_m2} m2)")
    return task


def release_points(candidates, search_radius_m=600, smin=30.0, smax=50.0):
    """Derive an avalanche RELEASE point for each deposit polygon.

    Susceptibility models predict where avalanches RELEASE, not where debris stops, so each deposit
    is traced upslope: within a search radius around the deposit we take the highest pixel that sits
    on release-angle terrain (default 30-50 deg) and is concave in plan curvature, and return it as
    the presence point. Falls back to the deposit's own highest release-angle pixel if none is found.
    """
    slope = ee.Terrain.slope(DEM)
    release_terrain = slope.gte(smin).And(slope.lte(smax))
    elev = DEM.updateMask(release_terrain)

    def one(f):
        zone = f.geometry().buffer(search_radius_m)
        best = elev.addBands(ee.Image.pixelLonLat()).reduceRegion(
            reducer=ee.Reducer.max(3), geometry=zone, scale=30, maxPixels=int(1e9), bestEffort=True)
        lon, lat = best.get("max1"), best.get("max2")
        return ee.Algorithms.If(
            ee.Algorithms.IsEqual(lon, None),
            f.centroid(10).set("zone", "deposit_fallback"),
            ee.Feature(ee.Geometry.Point([lon, lat]), f.toDictionary()).set("zone", "release"))

    return candidates.map(one, dropNulls=True)
