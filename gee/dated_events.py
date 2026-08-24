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
                   lia_min=30.0, lia_max=75.0, max_gap_days=30):
    """Detections between consecutive acquisitions, each tagged with its date bracket.

    Returns an ee.FeatureCollection of deposit polygons with date_before / date_after /
    window_days / area_m2.
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


def export_dated(fc, name="s1_dated_events", folder="avalanche_gee_exports"):
    task = ee.batch.Export.table.toDrive(
        collection=fc.select(["date_before", "date_after", "window_days", "area_m2", "year"]),
        description=name, folder=folder, fileNamePrefix=name, fileFormat="GeoJSON")
    task.start()
    print("export started:", name)
    return task
