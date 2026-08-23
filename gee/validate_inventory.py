"""
gee/validate_inventory.py — independent validation of the Sentinel-1 avalanche inventory.

Two checks that need no manual interpretation:

1. CROSS-ORBIT AGREEMENT. Ascending and descending passes view the terrain from opposite sides and
   have independent noise and geometry. A path flagged by BOTH is very unlikely to be an artefact,
   so the agreement rate is a lower bound on precision. (It is only a lower bound: each geometry is
   blind to the aspects it foreshortens or shadows, so genuine paths on those aspects are missed by
   one orbit and count as disagreement.)

2. KNOWN-SITE RECALL. The published study names specific avalanche-affected locations. Checking
   whether the inventory flags avalanche activity at those sites is an independent recall test
   against expert/field knowledge.
"""
import ee

from .s1_inventory import multiyear_recurrence

# Avalanche-affected sites named in Abhinav & Sattar (2025), approximate coordinates.
KNOWN_SITES = {
    "Solang":            (77.158, 32.317),
    "Dhundi":            (77.128, 32.362),
    "Gepang Gath lake":  (77.220, 32.442),
    "Chandra Tal lake":  (77.618, 32.479),
    "Keylong":           (77.030, 32.573),
    "Darcha":            (77.192, 32.700),
    "Baralacha La":      (77.409, 32.760),
    "Chhota Shigri":     (77.520, 32.230),
}


def cross_orbit_agreement(aoi, years, min_recurrence=4,
                          desc=("DESCENDING", 136), asc=("ASCENDING", 27), scale=60):
    """Fraction of one orbit's detections that the other orbit independently reproduces."""
    d = multiyear_recurrence(aoi, years, rel_orbit=desc[1], orbit_pass=desc[0]).gte(min_recurrence)
    a = multiyear_recurrence(aoi, years, rel_orbit=asc[1], orbit_pass=asc[0]).gte(min_recurrence)
    d, a = d.unmask(0), a.unmask(0)

    both = d.And(a).rename("both")
    stats = (d.rename("desc").addBands(a.rename("asc")).addBands(both)
             .reduceRegion(ee.Reducer.sum(), aoi, scale=scale, maxPixels=int(1e12),
                           bestEffort=True).getInfo())
    nd, na, nb = stats["desc"], stats["asc"], stats["both"]
    out = dict(desc_px=nd, asc_px=na, both_px=nb,
               desc_confirmed=100 * nb / nd if nd else 0,
               asc_confirmed=100 * nb / na if na else 0,
               jaccard=100 * nb / (nd + na - nb) if (nd + na - nb) else 0)
    print(f"  descending detections: {nd:,.0f} px")
    print(f"  ascending  detections: {na:,.0f} px")
    print(f"  agreed by both       : {nb:,.0f} px")
    print(f"  -> {out['desc_confirmed']:.1f}% of descending detections confirmed by ascending")
    print(f"  -> {out['asc_confirmed']:.1f}% of ascending detections confirmed by descending")
    print(f"  -> Jaccard overlap    : {out['jaccard']:.1f}%")
    return out


def known_site_recall(years, radius_m=3000, min_recurrence=3,
                      sites=None, rel_orbit=136, orbit_pass="DESCENDING"):
    """Does the inventory flag avalanche activity at sites the published study reports?"""
    sites = sites or KNOWN_SITES
    hits = 0
    print(f"{'site':<20}{'detected px':>13}{'within':>9}")
    print("-" * 44)
    for name, (lon, lat) in sites.items():
        zone = ee.Geometry.Point([lon, lat]).buffer(radius_m)
        rec = multiyear_recurrence(zone, years, rel_orbit=rel_orbit, orbit_pass=orbit_pass)
        n = (rec.gte(min_recurrence).unmask(0)
             .reduceRegion(ee.Reducer.sum(), zone, scale=30, maxPixels=int(1e11),
                           bestEffort=True).getInfo())
        n = list(n.values())[0] if n else 0
        hits += n > 0
        print(f"{name:<20}{n:>13,.0f}{radius_m/1000:>7.0f} km")
    print(f"\nrecall at named sites: {hits}/{len(sites)} ({100*hits/len(sites):.0f}%)")
    return hits / len(sites)
