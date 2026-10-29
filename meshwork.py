"""
Lamin meshwork tracing after Shimi et al., Mol Biol Cell 2015 (doi:10.1091/mbc.E15-07-0461).

Steps (paper -> this implementation):
  1. steerable line filter            -> bank of oriented, elongated Gaussian 2nd-derivative filters
  2. threshold + non-maximal suppr.   -> NMS across the line direction, hysteresis threshold, edges >= 5 px
  3. gap closing at junctions         -> extend each endpoint along its direction until it hits a fibre
  4. remove dangling ends             -> iteratively prune spur pixels, re-skeletonize
  5. edge auditing                    -> keep edges whose mean intensity in a flattened image
                                         (image / Gaussian-blurred image) passes a quality threshold
  6. faces                            -> regions enclosed by edges; area, edges per face, mean edge length
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import fftconvolve
from skimage import draw, filters, measure, morphology

_N8 = np.ones((3, 3), int)


def line_filter(img, sigma=1.5, n_angles=16, elongation=3.0):
    """Oriented line detector. Returns (response, line angle in radians)."""
    half = int(np.ceil(3 * elongation * sigma))
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    resp = np.full(img.shape, -np.inf)
    ang = np.zeros(img.shape)
    for k in range(n_angles):
        th = np.pi * k / n_angles                         # direction along the line
        u = x * np.cos(th) + y * np.sin(th)               # along
        v = -x * np.sin(th) + y * np.cos(th)              # across
        g = np.exp(-(u ** 2 / (2 * (elongation * sigma) ** 2) + v ** 2 / (2 * sigma ** 2)))
        kern = (1 / sigma ** 2 - v ** 2 / sigma ** 4) * g   # -d2/dv2: positive on bright ridges
        kern -= kern.mean()
        kern /= np.abs(kern).sum()
        r = fftconvolve(img, kern, mode="same")
        better = r > resp
        resp[better] = r[better]
        ang[better] = th
    return np.maximum(resp, 0), ang


def nonmax_suppress(resp, ang):
    """Keep pixels that are maxima across the line (1-px centre lines)."""
    h, w = resp.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    ny, nx = np.cos(ang), -np.sin(ang)                     # unit normal (y, x)
    a = ndi.map_coordinates(resp, [yy + ny, xx + nx], order=1, mode="nearest")
    b = ndi.map_coordinates(resp, [yy - ny, xx - nx], order=1, mode="nearest")
    return np.where((resp >= a) & (resp >= b) & (resp > 0), resp, 0.0)


def _nbrs(sk):
    return ndi.convolve(sk.astype(np.uint8), _N8, mode="constant") - sk.astype(np.uint8)


def prune_spurs(sk, max_iter=10_000):
    """Remove every branch that does not end in a junction (repeatedly strip endpoints)."""
    sk = sk.copy()
    for _ in range(max_iter):
        ends = sk & (_nbrs(sk) <= 1)
        if not ends.any():
            break
        sk &= ~ends
    return sk


def _trace_back(sk, y, x, n):
    path = [(y, x)]
    seen = {(y, x)}
    cy, cx = y, x
    h, w = sk.shape
    for _ in range(n):
        nxt = [(cy + dy, cx + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if (dy or dx) and 0 <= cy + dy < h and 0 <= cx + dx < w
               and sk[cy + dy, cx + dx] and (cy + dy, cx + dx) not in seen]
        if len(nxt) != 1:
            break
        cy, cx = nxt[0]
        path.append((cy, cx)); seen.add((cy, cx))
    return path


def close_gaps(sk, mask, max_gap=10, trace=6):
    """Extend each endpoint along its local direction until it meets another fibre."""
    h, w = sk.shape
    ends = np.argwhere(sk & (_nbrs(sk) == 1))
    add = np.zeros_like(sk)
    for y, x in ends:
        path = _trace_back(sk, y, x, 2 * trace)
        if len(path) < 3:
            continue
        cy, cx = np.mean(path[1:trace + 1], axis=0)
        d = np.array([y - cy, x - cx]); nrm = np.hypot(*d)
        if nrm == 0:
            continue
        d /= nrm
        own = set(path)
        for s in range(1, max_gap + 1):
            py, px = int(round(y + d[0] * s)), int(round(x + d[1] * s))
            if not (0 <= py < h and 0 <= px < w) or not mask[py, px]:
                break
            hit = None
            for dy in (-1, 0, 1):          # 3x3 check so diagonal lines cannot be skipped over
                for dx in (-1, 0, 1):
                    qy, qx = py + dy, px + dx
                    if 0 <= qy < h and 0 <= qx < w and sk[qy, qx] and (qy, qx) not in own \
                            and max(abs(qy - y), abs(qx - x)) > 1:
                        hit = (qy, qx); break
                if hit:
                    break
            if hit:
                rr, cc = draw.line(y, x, *hit)
                add[rr, cc] = True
                break
    return sk | add


def edges_and_junctions(sk):
    nb = _nbrs(sk)
    junc = sk & (nb >= 3)
    junc = morphology.binary_dilation(junc) & sk          # junction clusters
    edges = measure.label(sk & ~junc, connectivity=2)
    return edges, junc


def audit_edges(sk, flat, min_quality):
    """Drop edges whose mean flattened intensity is below min_quality, then re-prune."""
    edges, junc = edges_and_junctions(sk)
    if edges.max() == 0:
        return sk
    q = ndi.mean(flat, labels=edges, index=np.arange(1, edges.max() + 1))
    keep = np.r_[False, q >= min_quality]
    sk2 = keep[edges] | junc
    return prune_spurs(morphology.skeletonize(sk2))


def trace_meshwork(g, nuc, sigma=1.5, sensitivity=1.0, max_gap=10, min_edge=5, quality=1.2,
                   flatten_sigma=None, prep=None, auto=None):
    """g: image 0..1, nuc: nucleus mask. Returns the 1-px meshwork (edges + junctions)."""
    if prep is None:
        resp, ang = line_filter(g, sigma)
        nms = nonmax_suppress(resp, ang)
        fs = flatten_sigma or 7 * sigma
        flat = g / np.maximum(filters.gaussian(g, fs), 1e-6)
    else:
        resp, nms, flat = prep
    if auto is None:
        vals = nms[nuc & (nms > 0)]
        auto = filters.threshold_otsu(vals) if vals.size else 1.0
    hi = auto * sensitivity
    lines = filters.apply_hysteresis_threshold(nms * nuc, 0.5 * hi, hi)
    lines = morphology.remove_small_objects(morphology.skeletonize(lines), min_edge, connectivity=2)
    sk = close_gaps(lines, nuc, max_gap)
    sk = prune_spurs(morphology.skeletonize(sk))
    sk = audit_edges(sk, flat, quality)
    return sk


def face_edge_stats(faces, sk, px=1.0):
    """Per face: number of bordering edges and their mean length."""
    edges, junc = edges_and_junctions(sk)
    n_f = int(faces.max())
    if n_f == 0 or edges.max() == 0:
        return np.zeros(n_f + 1), np.zeros(n_f + 1)
    length = np.bincount(edges.ravel(), minlength=edges.max() + 1).astype(float) * px
    pairs = set()
    h, w = faces.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            f = faces[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)]
            e = edges[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
            m = (f > 0) & (e > 0)
            pairs.update(zip(f[m].tolist(), e[m].tolist()))
    n_edges = np.zeros(n_f + 1); mean_len = np.zeros(n_f + 1)
    by_face = {}
    for f, e in pairs:
        by_face.setdefault(f, []).append(e)
    for f, es in by_face.items():
        n_edges[f] = len(es); mean_len[f] = length[es].mean()
    return n_edges, mean_len


def estimate_fiber_width(g, nuc):
    """Typical lamin fibre width (px) and the matching line-filter sigma.

    Fibres are roughly segmented with a local (adaptive) threshold inside the nucleus and thinned to
    centre lines; the width is read from the distance transform along the centre lines
    (width = 2*distance - 1), ignoring pixels near junctions, where fibres merge and look wider.
    For a bar of width w, a Gaussian 2nd-derivative line detector responds best at sigma = w / (2*sqrt(3)).
    """
    gs = filters.gaussian(g, 1)
    if nuc.sum() < 500:
        return 5.0, 1.5
    fg = (gs > filters.threshold_local(gs, 51)) & nuc
    fg = morphology.remove_small_objects(fg, 20)
    sk = morphology.skeletonize(fg)
    junc = morphology.binary_dilation(sk & (_nbrs(sk) >= 3), morphology.disk(3))
    d = ndi.distance_transform_edt(fg)[sk & ~junc]
    d = d[d > 0]
    if d.size < 50:
        return 5.0, 1.5
    width = float(np.median(2 * d - 1))
    return width, float(np.clip(width / (2 * np.sqrt(3)), 0.75, 6.0))


def _nearest(mask, x, y, r):
    """Nearest True pixel of mask within radius r of (x, y), or None."""
    h, w = mask.shape
    y0, y1, x0, x1 = max(0, y - r), min(h, y + r + 1), max(0, x - r), min(w, x + r + 1)
    ys, xs = np.nonzero(mask[y0:y1, x0:x1])
    if ys.size == 0:
        return None
    k = np.argmin((ys + y0 - y) ** 2 + (xs + x0 - x) ** 2)
    return int(ys[k] + y0), int(xs[k] + x0)


def apply_edits(sk, edits, snap=8):
    """Manual corrections to a traced meshwork (1-px lines).

    ("erase", x, y)          remove the edge (fibre segment between junctions) nearest to (x, y)
    ("draw", x0, y0, x1, y1) add a fibre; each end snaps to the nearest traced line within `snap` px
    ("exclude", x, y)        handled at the face level (see lamin_core.analyze)
    """
    sk = sk.copy()
    for op in edits:
        if op[0] == "erase":
            edges, _ = edges_and_junctions(sk)
            hit = _nearest(edges > 0, int(round(op[1])), int(round(op[2])), 6)
            if hit:
                sk[edges == edges[hit]] = False
                sk = prune_spurs(sk)                       # drop junction stubs left behind
        elif op[0] == "draw":
            (x0, y0), (x1, y1) = (int(round(op[1])), int(round(op[2]))), (int(round(op[3])), int(round(op[4])))
            p0 = _nearest(sk, x0, y0, snap) or (y0, x0)
            p1 = _nearest(sk, x1, y1, snap) or (y1, x1)
            rr, cc = draw.line(p0[0], p0[1], p1[0], p1[1])
            ok = (rr >= 0) & (rr < sk.shape[0]) & (cc >= 0) & (cc < sk.shape[1])
            sk[rr[ok], cc[ok]] = True
    return sk


SIGMA_GRID = (0.8, 1.1, 1.5, 2.0, 2.7, 3.6, 4.8)


def select_sigma(g, nuc, sensitivity=0.8, grid=SIGMA_GRID, patch=512):
    """Choose the line-filter scale from the data: trace a central patch of the nucleus at several scales
    and keep the smallest scale that yields >= 95% of the maximum number of closed faces.

    Too fine a scale traces noise/speckle as short fragments that never close into faces; too coarse a
    scale merges neighbouring fibres. The face count peaks at the scale of the meshwork itself. Tested on
    published 3D-SIM tracings (Shimi et al. 2015; Kittisopikul et al. 2019) and a confocal image; see
    test_images/validate_against_paper.py.  Returns (sigma, {sigma: face count}).
    """
    ys, xs = np.nonzero(nuc)
    if ys.size == 0:
        return 1.5, {}
    cy, cx = int(np.median(ys)), int(np.median(xs))
    h, w = g.shape
    y0, x0 = max(0, min(cy - patch // 2, h - patch)), max(0, min(cx - patch // 2, w - patch))
    sub, m = g[y0:y0 + patch, x0:x0 + patch], nuc[y0:y0 + patch, x0:x0 + patch]
    counts = {}
    for sg in grid:
        if sg * 12 > min(sub.shape):
            break
        resp, ang = line_filter(sub, sg)
        nms = nonmax_suppress(resp, ang)
        vals = nms[m & (nms > 0)]
        if vals.size < 20:
            continue
        auto = filters.threshold_otsu(vals) * sensitivity
        lines = filters.apply_hysteresis_threshold(nms * m, 0.5 * auto, auto)
        sk = morphology.remove_small_objects(morphology.skeletonize(lines), 5, connectivity=2)
        sk = prune_spurs(morphology.skeletonize(close_gaps(sk, m, int(max(4, round(5 * sg))))))
        areas = np.bincount(measure.label(m & ~sk, connectivity=1).ravel())[1:]
        counts[sg] = int((areas >= 4).sum())
    if not counts:
        return 1.5, counts
    best = max(counts.values())
    return min(s for s, c in counts.items() if c >= 0.95 * best), counts
