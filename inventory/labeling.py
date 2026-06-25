"""
inventory/labeling.py — SAR-assisted manual avalanche labelling.

Builds an interactive map (high-res satellite + Sentinel-2 for the event + SAR debris hotspots +
release terrain) and lets you drop markers on avalanches you can SEE in the optical. The SAR
hotspots are only *attention guides* — you confirm visually, so imperfect detection is fine.
Marked points append to a tiered inventory CSV in Drive (deduped).

Workflow (Colab):
    from inventory.labeling import build_label_map, save_marks, inventory_summary
    m = build_label_map(ref_window=("2022-12-01","2023-02-25"),
                        act_window=("2023-03-08","2023-03-28"),
                        s2_window=("2023-03-01","2023-04-20"))
    m                                   # pan, zoom, drop markers with the marker tool
    # ...in a new cell after marking...
    INV = "/content/drive/MyDrive/avalanche/data/raw/labels/inventory.csv"
    save_marks(m, event="2023-03", out_csv=INV)     # tier A (optically confirmed) by default
    inventory_summary(INV)
"""
import os
import datetime


def build_label_map(ref_window, act_window, s2_window, orbit="DESCENDING",
                    aoi_bbox=(76.40, 31.90, 78.10, 33.20), center=(32.4, 77.25), zoom=11):
    import ee
    import geemap
    AOI = ee.Geometry.Rectangle(list(aoi_bbox))

    def s1_median(a, b):
        col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(AOI).filterDate(a, b)
               .filter(ee.Filter.eq("instrumentMode", "IW"))
               .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
               .filter(ee.Filter.eq("orbitProperties_pass", orbit)).select(["VV", "VH"]))
        return col.median()

    change = s1_median(*act_window).subtract(s1_median(*ref_window)).focal_median(50, "circle", "meters")
    slope = ee.Terrain.slope(ee.Image("NASA/NASADEM_HGT/001").select("elevation"))
    hotspot = (change.select("VH").gt(2).And(change.select("VV").gt(1))
               .And(slope.gte(25)).And(slope.lte(55)))
    hotspot = hotspot.selfMask().connectedPixelCount(50, True).gte(10).selfMask()
    release = slope.gte(30).And(slope.lte(50)).selfMask()
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(AOI).filterDate(*s2_window)
          .filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "less_than", 25).median())

    m = geemap.Map(center=list(center), zoom=zoom)
    m.add_basemap("SATELLITE")                                                    # Google high-res
    m.addLayer(s2, {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000}, "Sentinel-2 (event)")
    m.addLayer(release, {"palette": ["#00e5ff"]}, "Release terrain 30-50°", True, 0.20)
    m.addLayer(hotspot, {"palette": ["#ff1744"]}, "SAR debris hotspots", True, 0.85)
    print("Map ready. Run arm_click_capture(m), then CLICK avalanches you can SEE in the optical "
          "(a pin drops per click), then save_marks(m, event=..., out_csv=...).")
    return m


def arm_click_capture(m):
    """Attach a click handler so each map click drops a pin and records (lon,lat). Returns the list.
    Use this in Colab — the draw-tool geometry often fails to sync back to Python there, but plain
    click interactions come through fine."""
    from ipyleaflet import Marker
    clicks = []

    def _cap(**kw):
        if kw.get("type") == "click" and kw.get("coordinates"):
            lat, lon = kw["coordinates"]
            clicks.append((lon, lat))
            m.add_layer(Marker(location=(lat, lon), draggable=False))
            print(f"  ✓ #{len(clicks)}: lon={lon:.5f}, lat={lat:.5f}")

    m.on_interaction(_cap)
    m._avalanche_clicks = clicks
    print("Armed ✅  click avalanches on the map; each drops a pin and is recorded.")
    return clicks


def _drawn_points(m):
    """Return [(lon,lat), ...] for Point markers drawn on the map.

    Reads the ipyleaflet DrawControl's .data (GeoJSON of everything drawn) directly — robust across
    geemap versions and no server round-trip. (m.draw_features / user_rois aren't populated in all
    versions, so we go to the control itself.)"""
    dc = getattr(m, "draw_control", None) or getattr(m, "_draw_control", None)
    data = getattr(dc, "data", None)
    if not data:
        for c in getattr(m, "controls", []):
            if c.__class__.__name__ == "DrawControl":
                data = getattr(c, "data", None)
                break
    pts = []
    for f in (data or []):
        g = (f or {}).get("geometry") or {}
        if g.get("type") == "Point":
            lon, lat = g["coordinates"][:2]
            pts.append((lon, lat))
    return pts


def save_marks(m, event, out_csv, tier="A", zone="release", clicks=None):
    """Append captured avalanche markers to the inventory CSV (deduped on ~11 m rounded coords).
    Reads clicks from arm_click_capture (Colab-robust); falls back to the draw control."""
    import pandas as pd
    pts = clicks if clicks is not None else getattr(m, "_avalanche_clicks", None)
    if not pts:
        pts = _drawn_points(m)
    if not pts:
        print("no markers captured — run arm_click_capture(m) and click avalanches first")
        return
    now = datetime.datetime.utcnow().isoformat(timespec="seconds")
    new = pd.DataFrame([dict(lon=round(lo, 6), lat=round(la, 6), event=event,
                             tier=tier, zone=zone, added=now) for lo, la in pts])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df = pd.concat([pd.read_csv(out_csv), new], ignore_index=True) if os.path.exists(out_csv) else new
    df["_k"] = df.lon.round(4).astype(str) + "_" + df.lat.round(4).astype(str)
    df = df.drop_duplicates("_k", keep="first").drop(columns="_k")
    df.to_csv(out_csv, index=False)
    print(f"saved {len(new)} marker(s); inventory now {len(df)} unique points -> {out_csv}")


def inventory_summary(out_csv):
    import pandas as pd
    if not os.path.exists(out_csv):
        print("no inventory yet")
        return
    df = pd.read_csv(out_csv)
    print(f"{len(df)} avalanche points total")
    if "event" in df:
        print("by event:", df["event"].value_counts().to_dict())
    if "tier" in df:
        print("by tier :", df["tier"].value_counts().to_dict())
