"""
gee/dated_events.py — give every detection an EVENT DATE, not just a winter.

The recurrence detector in s1_inventory.py compares a whole activity season against a winter
reference, so a detection means "this slope avalanched some time this winter". The published
inventory dates every event to the day, which is what allows an event to be attributed to a
snowfall, classified as wet or dry, or used in a temporal model. This module recovers most of that.

Method: instead of one seasonal composite pair, compare CONSECUTIVE acquisitions on the same
relative orbit. Debris that appears between acquisition i and i+1 must have been deposited in that
interval, so the event is bracketed to the repeat period (6-12 days for Sentinel-1 here). Each
detection therefore carries `date_before`, `date_after` and the window length in days.

Comparing neighbouring acquisitions rather than season-vs-reference also removes slow seasonal
drift: only what changed within the interval is flagged.
"""
import ee

from .s1_inventory import DEM, _composite, _s1


def dated_deposits(aoi, rel_orbit, season_start, season_end, orbit_pass="DESCENDING",
                   vh_thresh=2.0, vv_cap=6.0, min_pixels=8,
                   lia_min=30.0, lia_max=75.0, max_gap_days=30, restrict_mask=None,
                   require_snow=True, snow_min=65):
    """Detections between consecutive acquisitions, each tagged with its date bracket.

    Returns an ee.FeatureCollection of deposit polygons with date_before / date_after /
    window_days / area_m2.

    `require_snow` rejects MELT-OUT false positives. Snow melting off rough ground roughens the
    surface and raises backscatter exactly like avalanche debris does, so a bare-ground transition is
    indistinguishable from a deposit on the change signal alone. Verification exposed this clearly:
    without the filter, May+June hold 47% of detections against 12% in the published field inventory,
    and June holds 13% where the published inventory has ZERO June avalanches because the snow is
    already gone. A real deposit lands on terrain that is still snow-covered afterwards, so we
    require MODIS NDSI snow cover >= `snow_min` in the days after the event.

    `restrict_mask` confines detection to an existing mask, normally the multi-year recurrence
    paths. Pairwise detection has higher RECALL but lower PRECISION than recurrence, because there
    is no multi-winter filter to reject one-off change; restricting it to slopes already shown to
    avalanche repeatedly keeps both properties at once, and cuts the exported volume by an order of
    magnitude. This is the synthesis product: events that are BOTH dated and confident.
    """
    from .s1_inventory import local_incidence_angle

    col = (_s1(aoi, season_start, season_end, orbit_pass, rel_orbit)
           .sort("system:time_start"))

    # One relative orbit acquires several frames along the track on the SAME day, so consecutive
    # images in the sorted list can share a date. Comparing those compares overlapping frames, not
    # two points in time (it produced window_days = 0). Mosaic per acquisition date first.
    dates = col.aggregate_array("system:time_start").map(
        lambda t: ee.Date(t).format("YYYY-MM-dd")).distinct().sort()

    def daily_mosaic(d):
        d = ee.Date(ee.String(d))
        day = col.filterDate(d, d.advance(1, "day"))
        return (day.mosaic()
                .set("system:time_start", d.millis())
                .set("date_str", d.format("YYYY-MM-dd")))

    imgs = dates.map(daily_mosaic)
    n = imgs.size()
    slope = ee.Terrain.slope(DEM)

    def pair(i):
        i = ee.Number(i)
        a = ee.Image(imgs.get(i))
        b = ee.Image(imgs.get(i.add(1)))
        t0 = ee.Date(a.get("system:time_start"))
        t1 = ee.Date(b.get("system:time_start"))
        gap = t1.difference(t0, "day")

        pa = _composite(ee.ImageCollection([a]))
        pb = _composite(ee.ImageCollection([b]))
        change = pb.subtract(pa)

        lia = local_incidence_angle(b.select("angle"), orbit_pass)
        ok = (change.select("VH").gt(vh_thresh)
              .And(change.select("VV").gt(vh_thresh - 1.0))
              .And(change.select("VV").lt(vv_cap))
              .And(slope.gte(25)).And(slope.lte(60))
              .And(lia.gte(lia_min)).And(lia.lte(lia_max)))
        if require_snow:
            # MEAN, not max: MODIS is 500 m, so one snowy sub-pixel makes a max-reducer accept a
            # slope that has actually melted out. The mean over the following days is the stricter
            # test that the ground really is still snow-covered.
            snow = (ee.ImageCollection("MODIS/061/MOD10A1")
                    .filterDate(t1, t1.advance(12, "day"))
                    .select("NDSI_Snow_Cover").mean())
            ok = ok.And(snow.gte(snow_min).unmask(0))
        if restrict_mask is not None:
            ok = ok.And(restrict_mask)
        ok = ok.selfMask().connectedPixelCount(50, True).gte(min_pixels).selfMask()

        vec = ok.reduceToVectors(geometry=aoi, scale=30, geometryType="polygon",
                                 eightConnected=True, maxPixels=int(1e13))
        vec = vec.map(lambda f: f.set({
            "date_before": t0.format("YYYY-MM-dd"),
            "date_after": t1.format("YYYY-MM-dd"),
            "window_days": gap,
            "area_m2": f.geometry().area(10),
        }))
        # Sentinel-1B failed in Dec 2021, so revisit on a single orbit here is 24 days for most of
        # the season and 12 days only in places. Pairs wider than max_gap_days are skipped because
        # the event bracket would be too loose to attribute to weather.
        return ee.FeatureCollection(ee.Algorithms.If(gap.lte(max_gap_days),
                                                     vec, ee.FeatureCollection([])))

    idx = ee.List.sequence(0, n.subtract(2))
    return ee.FeatureCollection(idx.map(pair)).flatten()


def dated_inventory(aoi, years, rel_orbit=136, orbit_pass="DESCENDING",
                    season=((1, 1), (5, 31)), min_area_m2=2700, **kw):
    """Dated detections across many winters, concatenated into one inventory."""
    def one(y):
        y = ee.Number(y)
        s = ee.Date.fromYMD(y, season[0][0], season[0][1])
        e = ee.Date.fromYMD(y, season[1][0], season[1][1])
        fc = dated_deposits(aoi, rel_orbit, s, e, orbit_pass, **kw)
        return fc.map(lambda f: f.set("year", y))

    out = ee.FeatureCollection(ee.List(years).map(one)).flatten()
    return out.filter(ee.Filter.gte("area_m2", min_area_m2))


def confident_dated_inventory(aoi, years, rel_orbit=136, orbit_pass="DESCENDING",
                              min_recurrence=3, min_area_m2=2700, **kw):
    """Dated events restricted to slopes that avalanche repeatedly — confident AND dated.

    Builds the multi-year recurrence mask first, then runs pairwise dated detection inside it.
    """
    from .s1_inventory import multiyear_recurrence

    rec = multiyear_recurrence(aoi, years, rel_orbit=rel_orbit, orbit_pass=orbit_pass)
    mask = rec.gte(min_recurrence).unmask(0)
    return dated_inventory(aoi, years, rel_orbit=rel_orbit, orbit_pass=orbit_pass,
                           min_area_m2=min_area_m2, restrict_mask=mask, **kw)


def export_dated(fc, name="s1_dated_events", folder="avalanche_gee_exports"):
    task = ee.batch.Export.table.toDrive(
        collection=fc.select(["date_before", "date_after", "window_days", "area_m2", "year"]),
        description=name, folder=folder, fileNamePrefix=name, fileFormat="GeoJSON")
    task.start()
    print("export started:", name)
    return task
