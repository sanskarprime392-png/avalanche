"""
inventory/verify.py — measure the PRECISION of the Sentinel-1 inventory.

Precision is the inventory's one unmeasured property, and it is the first thing a reviewer will
ask about. Blind visual search over a 23,000 km2 basin is hopeless, but a dated inventory makes
verification tractable: for a given detection we know the event is bracketed between two dates, so
we can pull the last cloud-free Sentinel-2 scene BEFORE and the first one AFTER, and check whether
fresh avalanche debris appears in that gap.

The sample is stratified by recurrence so precision can be reported per confidence tier, which is
what tells you where to set the operational threshold.

Workflow:
    from inventory.verify import sample_for_verification, verification_map, record
    sample = sample_for_verification(events, n_per_tier=25)
    m = verification_map(sample, i=0)      # inspect one candidate
    record(0, True)                        # True = real avalanche, False = false positive
    ...
    precision_report()
"""
import io
import json
import os

_VERDICTS = {}


def sample_for_verification(features, n_per_tier=25, tiers=(3, 4, 5, 6), seed=42):
    """Stratified random sample of detections across recurrence tiers."""
    import random

    rng = random.Random(seed)
    by_tier = {t: [] for t in tiers}
    for f in features:
        r = int(f["properties"].get("recurrence", 0) or 0)
        for t in sorted(tiers, reverse=True):
            if r >= t:
                by_tier[t].append(f)
                break
    out = []
    for t in tiers:
        pool = by_tier[t]
        take = rng.sample(pool, min(n_per_tier, len(pool)))
        for f in take:
            f = dict(f)
            f["_tier"] = t
            out.append(f)
        print(f"  tier >={t}: sampled {len(take)} of {len(pool)}")
    rng.shuffle(out)          # review blind to tier, so expectation cannot bias the verdict
    return out


def verification_map(sample, i, buffer_m=800, s2_cloud=30):
    """Side-by-side before/after Sentinel-2 for one candidate, centred on the detection."""
    import ee
    import geemap

    f = sample[i]
    p = f.get("properties", {})
    geom = ee.Geometry(f["geometry"])
    centre = geom.centroid(10).coordinates().getInfo()
    zone = geom.buffer(buffer_m)

    date_before = p.get("date_before")
    date_after = p.get("date_after")

    def s2(start, end):
        return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(zone).filterDate(start, end)
                .filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "less_than", s2_cloud)
                .sort("system:time_start", False).median())

    vis = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3500}
    m = geemap.Map(center=[centre[1], centre[0]], zoom=15)
    m.add_basemap("SATELLITE")
    if date_before and date_after:
        pre = ee.Date(date_before)
        post = ee.Date(date_after)
        m.addLayer(s2(pre.advance(-25, "day"), pre.advance(1, "day")), vis, "S2 BEFORE", False)
        m.addLayer(s2(post, post.advance(25, "day")), vis, "S2 AFTER", True)
    m.addLayer(ee.FeatureCollection([ee.Feature(geom)]).style(
        color="red", fillColor="00000000", width=2), {}, "detection")
    print(f"candidate {i}  tier>={f.get('_tier')}  "
          f"window {date_before} -> {date_after}  area {p.get('area_m2', 0):.0f} m2")
    print("Look for: fresh rough/dirty debris inside the red outline in AFTER but not BEFORE.")
    return m


def make_chips(sample, out_dir, buffer_m=700, size=420, s2_cloud=45, pad_days=25):
    """Render a before/after Sentinel-2 chip pair for each candidate as a single PNG.

    Produces reviewable images without needing an interactive map, so verification can be done
    anywhere. The detection outline is painted on both panels: a true positive shows fresh, rough,
    dirty-looking debris inside the outline in AFTER that is absent in BEFORE.
    """
    import ee
    import requests
    from PIL import Image, ImageDraw

    os.makedirs(out_dir, exist_ok=True)
    # Snow reflectance in the visible bands routinely exceeds 6000, so the usual max of ~3500
    # saturates every snow-covered scene to flat white and destroys exactly the surface texture that
    # distinguishes avalanche debris from undisturbed snow. Stretch for snow instead.
    vis = {"bands": ["B4", "B3", "B2"], "min": 300, "max": 9000, "gamma": 1.35}
    written = []

    for i, f in enumerate(sample):
        p = f.get("properties", {})
        geom = ee.Geometry(f["geometry"])
        zone = geom.buffer(buffer_m).bounds()
        d0, d1 = p.get("date_before"), p.get("date_after")
        if not (d0 and d1):
            continue
        pre_d, post_d = ee.Date(d0), ee.Date(d1)

        def comp(start, end, newest_first):
            # A median over several weeks averages the event away, but a single nearest scene is
            # usually cloudy here (Himalayan spring). Compromise: mosaic the few LEAST CLOUDY scenes
            # in the window, which fills swath gaps while keeping the event legible.
            col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                   .filterBounds(zone).filterDate(start, end)
                   .filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "less_than", s2_cloud)
                   .sort("CLOUDY_PIXEL_PERCENTAGE"))
            return ee.Image(ee.Algorithms.If(col.size().gt(0),
                                             col.limit(3).mosaic(),
                                             ee.Image.constant([0, 0, 0])
                                             .rename(["B4", "B3", "B2"])))

        outline = ee.Image().byte().paint(
            ee.FeatureCollection([ee.Feature(geom)]), color=1, width=2)
        panels = []
        for img, label in ((comp(pre_d.advance(-pad_days, "day"), pre_d.advance(1, "day"),
                                 True), "BEFORE"),
                           (comp(post_d, post_d.advance(pad_days, "day"), False), "AFTER")):
            shown = img.visualize(**vis).blend(outline.visualize(palette=["FF0000"]))
            try:
                url = shown.getThumbURL({"region": zone, "dimensions": size, "format": "png"})
                r = requests.get(url, timeout=120)
                r.raise_for_status()
                im = Image.open(io.BytesIO(r.content)).convert("RGB")
            except Exception as ex:
                print(f"  candidate {i}: {type(ex).__name__} fetching {label}")
                im = Image.new("RGB", (size, size), (30, 30, 30))
            ImageDraw.Draw(im).text((8, 8), label, fill=(255, 255, 0))
            panels.append(im)

        w, h = panels[0].size
        sheet = Image.new("RGB", (w * 2 + 12, h + 26), (18, 18, 18))
        sheet.paste(panels[0], (0, 26))
        sheet.paste(panels[1], (w + 12, 26))
        ImageDraw.Draw(sheet).text(
            (8, 7), f"#{i}  tier>={f.get('_tier')}  {d0} -> {d1}  {p.get('area_m2', 0):.0f} m2",
            fill=(235, 235, 235))
        path = os.path.join(out_dir, f"cand_{i:03d}.png")
        sheet.save(path)
        written.append(path)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(sample)} chips", flush=True)
    print(f"wrote {len(written)} chips -> {out_dir}")
    return written


def contact_sheet(chip_paths, out_path, cols=2, rows=5):
    """Tile individual chips into multi-candidate sheets for faster review."""
    from PIL import Image

    sheets = []
    per = cols * rows
    for s in range(0, len(chip_paths), per):
        group = chip_paths[s:s + per]
        ims = [Image.open(p) for p in group]
        w, h = ims[0].size
        sheet = Image.new("RGB", (w * cols + 10 * (cols - 1), h * rows + 10 * (rows - 1)),
                          (12, 12, 12))
        for k, im in enumerate(ims):
            sheet.paste(im, ((k % cols) * (w + 10), (k // cols) * (h + 10)))
        path = out_path.replace(".png", f"_{s//per + 1}.png")
        sheet.save(path)
        sheets.append(path)
    print(f"wrote {len(sheets)} contact sheets")
    return sheets


def record(i, is_avalanche, note=""):
    _VERDICTS[int(i)] = dict(verdict=bool(is_avalanche), note=note)
    n = len(_VERDICTS)
    tp = sum(1 for v in _VERDICTS.values() if v["verdict"])
    print(f"recorded {n}: running precision {tp}/{n} = {100*tp/n:.0f}%")


def save(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_VERDICTS, fh, indent=2)
    print(f"saved {len(_VERDICTS)} verdicts -> {path}")


def precision_report(sample=None):
    """Precision overall and per recurrence tier, with a Wilson confidence interval."""
    import math

    if not _VERDICTS:
        print("no verdicts recorded yet")
        return
    rows = []
    groups = {"ALL": list(_VERDICTS.items())}
    if sample is not None:
        for t in sorted({s.get("_tier") for s in sample}):
            groups[f"tier >={t}"] = [(i, v) for i, v in _VERDICTS.items()
                                     if sample[int(i)].get("_tier") == t]
    for name, items in groups.items():
        n = len(items)
        if not n:
            continue
        k = sum(1 for _, v in items if v["verdict"])
        p = k / n
        z = 1.96
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        rows.append((name, k, n, p, max(0, c - h), min(1, c + h)))
    print(f"{'group':<12}{'true':>6}{'n':>5}{'precision':>11}{'95% CI':>18}")
    print("-" * 54)
    for name, k, n, p, lo, hi in rows:
        print(f"{name:<12}{k:>6}{n:>5}{100*p:>10.0f}%{f'[{100*lo:.0f}%, {100*hi:.0f}%]':>18}")
    return rows
