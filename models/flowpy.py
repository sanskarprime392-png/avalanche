"""
models/flowpy.py — gravitational mass-flow routing (Flow-Py style) for avalanche runout.

This replaces the line-of-sight energy line in models/risk.py, which over-estimated exposure by
roughly 60x because it let a source "reach" a building across an intervening ridge or valley.
Here the flow has to actually travel over the terrain.

Model (after Huber et al. 2016; D'Amboise et al. 2022, the Flow-Py open-source implementation):

  Each cell carries an ENERGY-LINE HEIGHT z_delta, the vertical distance between the energy line and
  the terrain surface — physically, the kinetic energy of the flow expressed as a height. Moving
  from cell i to a downslope neighbour j:

      z_delta_j = z_delta_i + (z_i - z_j) - d_ij * tan(alpha)

  The flow gains energy from the drop (z_i - z_j) and loses it to friction along the path
  (d_ij * tan(alpha)). Where z_delta falls to zero the flow stops, which reproduces the classical
  alpha-angle runout WITHOUT assuming a straight line: the path is constrained to the terrain.

  z_delta is additionally capped, since real flows have a maximum velocity:
      z_delta_max = v_max^2 / (2 g)

Processing cells in order of DESCENDING elevation guarantees every contributing upslope cell is
finalised before a cell is used, so a single pass suffices. Alongside the energy we propagate the
susceptibility of the release cell that produced the strongest arriving flow, giving each runout
cell a source-linked hazard value rather than a max over hundreds of unconnected sources.
"""
import numpy as np
from numba import njit

G = 9.81


@njit(cache=True)
def _route(z, release, rel_sus, tan_alpha, z_delta_max, cell, flux_min, exponent):
    """One descending-elevation pass. Returns (z_delta, source_susceptibility, flux).

    Energy (z_delta) and MASS (flux) are routed together. Energy alone is not enough: in terrain
    steeper than alpha the energy line never intersects the surface, so a single release cell would
    propagate through the whole downslope network. Flux is divided among downslope neighbours
    (multiple-flow-direction, weighted by slope^exponent) and the flow is abandoned once flux falls
    below flux_min, which is what makes the runout finite.
    """
    H, W = z.shape
    z_delta = np.full((H, W), -1.0, dtype=np.float32)
    src = np.zeros((H, W), dtype=np.float32)
    flux = np.zeros((H, W), dtype=np.float32)

    order = np.argsort(-z.ravel())
    wts = np.empty(8, dtype=np.float32)
    nbr = np.empty((8, 2), dtype=np.int64)
    nzd = np.empty(8, dtype=np.float32)

    for k in range(order.size):
        idx = order[k]
        r = idx // W
        c = idx % W
        zi = z[r, c]
        if not np.isfinite(zi):
            continue
        if release[r, c]:
            if z_delta[r, c] < 0.0:
                z_delta[r, c] = 0.0
            if flux[r, c] < 1.0:
                flux[r, c] = 1.0
            if rel_sus[r, c] > src[r, c]:
                src[r, c] = rel_sus[r, c]
        zd = z_delta[r, c]
        fl = flux[r, c]
        if zd < 0.0 or fl < flux_min:
            continue
        s = src[r, c]

        n = 0
        wsum = 0.0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr = r + dr
                cc = c + dc
                if rr < 0 or rr >= H or cc < 0 or cc >= W:
                    continue
                zj = z[rr, cc]
                if not np.isfinite(zj) or zj >= zi:
                    continue
                d = cell * (1.4142135 if (dr != 0 and dc != 0) else 1.0)
                nz = zd + (zi - zj) - d * tan_alpha
                if nz <= 0.0:
                    continue                       # energy exhausted on this path
                if nz > z_delta_max:
                    nz = z_delta_max
                w = ((zi - zj) / d) ** exponent    # steeper neighbour takes more mass
                nbr[n, 0] = rr
                nbr[n, 1] = cc
                nzd[n] = nz
                wts[n] = w
                wsum += w
                n += 1

        if n == 0 or wsum <= 0.0:
            continue
        for m in range(n):
            rr = nbr[m, 0]
            cc = nbr[m, 1]
            share = fl * (wts[m] / wsum)
            if share < flux_min:
                continue
            if share > flux[rr, cc]:            # max single-path share, NOT a sum: summing over
                flux[rr, cc] = share             # convergent paths stops depletion ever biting
            if nzd[m] > z_delta[rr, cc]:
                z_delta[rr, cc] = nzd[m]
                src[rr, cc] = s
            elif s > src[rr, cc]:
                src[rr, cc] = s
    return z_delta, src, flux


def route_runout(dem, susceptibility, cell_size, alpha_deg=28.0, v_max=40.0,
                 sus_min=0.75, min_release_px=6, flux_min=3e-4, exponent=8.0):
    """Route avalanche flow from high-susceptibility release areas across the terrain.

    Returns dict with `z_delta` (energy-line height, >=0 inside the runout), `source_sus`
    (susceptibility of the release area feeding each runout cell) and `flux` (routed mass share).
    `flux_min` and `exponent` follow the Flow-Py defaults: a higher exponent concentrates flow into
    the steepest descent (channelised), a lower one spreads it (divergent).
    """
    from scipy import ndimage

    z = np.where(np.isfinite(dem), dem, np.nan).astype(np.float32)
    rel = np.isfinite(susceptibility) & (susceptibility >= sus_min) & np.isfinite(z)
    if min_release_px > 1:
        lab, _ = ndimage.label(rel)
        sizes = np.bincount(lab.ravel())
        big = np.nonzero(sizes >= min_release_px)[0]
        big = big[big != 0]
        rel = np.isin(lab, big)
    rel_sus = np.where(rel, np.nan_to_num(susceptibility, nan=0.0), 0.0).astype(np.float32)
    print(f"  release cells: {int(rel.sum()):,} "
          f"({int(rel.sum())*cell_size**2/1e6:.0f} km2), alpha={alpha_deg}°, v_max={v_max} m/s")

    z_delta_max = v_max ** 2 / (2 * G)
    zd, src, flux = _route(z, rel, rel_sus, float(np.tan(np.deg2rad(alpha_deg))),
                           float(z_delta_max), float(cell_size),
                           float(flux_min), float(exponent))
    runout = (zd >= 0.0) & (flux >= flux_min)
    print(f"  runout footprint: {int(runout.sum()):,} cells "
          f"({int(runout.sum())*cell_size**2/1e6:,.0f} km2)")
    return dict(z_delta=zd, source_sus=src, flux=flux, runout=runout)


def sample_at_points(arr, transform, xy):
    """Sample a routed raster at projected point coordinates."""
    inv = ~transform
    c, r = inv * (xy[:, 0], xy[:, 1])
    r = np.floor(r).astype(np.int64)
    c = np.floor(c).astype(np.int64)
    ok = (r >= 0) & (r < arr.shape[0]) & (c >= 0) & (c < arr.shape[1])
    out = np.full(len(xy), np.nan, "float32")
    out[ok] = arr[r[ok], c[ok]]
    return out
