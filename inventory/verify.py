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
