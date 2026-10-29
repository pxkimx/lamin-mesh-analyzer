"""
Lamin meshwork analysis core: image loading, segmentation, measurement, figures, PDF report.

Used by app.py (Streamlit UI). Can also be imported from scripts / notebooks:

    from lamin_core import load_image, Params, analyze, build_report
    img = load_image(open("cell.tif", "rb").read(), "cell.tif")
    res = analyze(img, Params())
    print(res.stats)
"""
from __future__ import annotations

import io
import re
import textwrap
import datetime as dt
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters, morphology, measure, segmentation, draw

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, LogNorm

import meshwork as M

# ----------------------------------------------------------------------------- palette
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
OUTLINE_RGB = (255, 212, 0)      # face outlines on the image
CORE_RGB = (235, 73, 72)         # analysis-region boundary
ROI_RGB = (0, 200, 255)          # user ROI boundary

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False,
})


# ----------------------------------------------------------------------------- loading
@dataclass
class LoadedImage:
    name: str
    channels: np.ndarray          # H x W x C, float 0..1
    display: np.ndarray           # H x W x 3 uint8 composite for viewing
    pixel_size: float | None      # physical size of one pixel, if found in metadata
    unit: str | None
    default_channel: int
    channel_names: list


def _norm(a):
    a = a.astype(float)
    lo, hi = np.percentile(a, (0.0, 99.95))
    if hi <= lo:
        hi = a.max() if a.max() > lo else lo + 1
    return np.clip((a - lo) / (hi - lo), 0, 1)


def _tiff_scale(tf: tifffile.TiffFile):
    """Pixel size + unit from ImageJ / OME / resolution tags (None if absent)."""
    try:
        ome = tf.ome_metadata
        if ome:
            m = re.search(r'PhysicalSizeX="([\d.eE+-]+)"', ome)
            u = re.search(r'PhysicalSizeXUnit="([^"]+)"', ome)
            if m:
                return float(m.group(1)), (u.group(1) if u else "µm")
    except Exception:
        pass
    try:
        page = tf.pages[0]
        xr = page.tags.get("XResolution")
        if xr is not None:
            num, den = xr.value
            ppu = num / den if den else 0
            if ppu > 0:
                unit = None
                ij = tf.imagej_metadata or {}
                if ij.get("unit"):
                    unit = ij["unit"].replace("\\u00B5", "µ").replace("micron", "µm")
                else:
                    ru = page.tags.get("ResolutionUnit")
                    ru = int(ru.value) if ru is not None else 1
                    if ru == 3:      # centimetre -> µm
                        return 1e4 / ppu, "µm"
                    if ru == 2:      # inch = generic dpi, not a microscope scale
                        return None, None
                if unit:
                    return 1.0 / ppu, unit
    except Exception:
        pass
    return None, None


def load_image(data: bytes, name: str) -> LoadedImage:
    px, unit = None, None
    if name.lower().endswith((".tif", ".tiff")):
        with tifffile.TiffFile(io.BytesIO(data)) as tf:
            arr = tf.asarray()
            axes = tf.series[0].axes if tf.series else ""
            px, unit = _tiff_scale(tf)
        arr = np.asarray(arr)
        # bring to H x W x C; max-project any extra axes (Z, T, ...)
        if len(axes) == arr.ndim and "Y" in axes and "X" in axes:
            ch = next((a for a in "SC" if a in axes), None)      # channel-like axis, if any
            other = tuple(i for i, a in enumerate(axes) if a not in "YX" and a != ch)
            if other:
                arr = arr.max(axis=other)
                axes = "".join(a for i, a in enumerate(axes) if i not in other)
            arr = np.transpose(arr, [axes.index(a) for a in "YX" + (ch or "")])
        while arr.ndim > 3:
            arr = arr.max(axis=0)
        if arr.ndim == 3 and arr.shape[0] <= 6 and arr.shape[-1] > 6:
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim == 2:
            arr = arr[..., None]
    else:
        im = Image.open(io.BytesIO(data))
        im = im.convert("RGB") if im.mode not in ("L", "I;16", "I") else im
        arr = np.asarray(im)
        if arr.ndim == 2:
            arr = arr[..., None]

    # drop constant channels (e.g. alpha = 255 everywhere)
    keep = [c for c in range(arr.shape[-1]) if arr[..., c].max() > arr[..., c].min()]
    if not keep:
        keep = [0]
    chans = np.dstack([_norm(arr[..., c]) for c in keep])

    # display composite
    if arr.dtype == np.uint8 and len(keep) >= 3:
        disp = arr[..., keep[:3]].astype(np.uint8)
    elif chans.shape[-1] == 1:
        disp = np.repeat((chans * 255).astype(np.uint8), 3, axis=2)
    else:
        colors = [(0, 1, 0), (1, 0, 1), (0, 0.6, 1), (1, 1, 0)]
        comp = np.zeros(chans.shape[:2] + (3,))
        for i in range(min(4, chans.shape[-1])):
            comp += chans[..., i, None] * np.array(colors[i])
        disp = (np.clip(comp, 0, 1) * 255).astype(np.uint8)

    # channel names + default lamin channel (green for RGB images, else most edge-rich channel)
    is_rgb = arr.dtype == np.uint8 and arr.shape[-1] in (3, 4)
    names = [(["Red", "Green", "Blue", "Alpha"][c] if is_rgb else f"Channel {c + 1}") for c in keep]
    if "Green" in names:
        default = names.index("Green")
    else:
        default = int(np.argmax([np.std(filters.sobel(chans[::4, ::4, c])) for c in range(chans.shape[-1])]))
    return LoadedImage(name, chans, disp, px, unit, default, names)


# ----------------------------------------------------------------------------- analysis
@dataclass
class Params:
    channel: int = 1
    method: str = "trace"          # "trace" = meshwork tracing (Shimi et al. 2015); "threshold" = black & white
    smooth_sigma: float = 1.0      # px, light denoising before thresholding (threshold method)
    threshold: float = -1.0        # black/white cut on a 0-255 scale; -1 = automatic (Otsu inside nucleus)
    line_sigma: float = -1.0       # px, width scale of the line filter; -1 = automatic from fibre width
    sensitivity: float = 1.0       # x automatic line threshold; lower = more (fainter) fibres traced
    max_gap: int = -1              # px, longest gap closed by extending fibre ends; -1 = automatic (5 x sigma)
    edge_quality: float = 1.0      # min. mean flattened intensity of a kept edge (1 = local average)
    nucleus_scale: float = 0.6     # x Otsu for nucleus detection
    all_nuclei: bool = False       # analyze every nucleus instead of only the largest
    whole_image: bool = False      # image is a crop inside the nucleus: skip nucleus detection
    edge_margin: int = 25          # px excluded along the nuclear rim
    min_area_px: int = 20          # smallest face kept
    max_puncta_px: int = 30        # isolated white specks smaller than this are treated as black


@dataclass
class Result:
    name: str
    params: Params
    nuc: np.ndarray
    core: np.ndarray               # final analysis region (rim-excluded ∩ ROI)
    fil: np.ndarray                # filament mask (inside nucleus)
    faces: np.ndarray              # label image of measured faces
    table: pd.DataFrame
    stats: dict
    auto_threshold: float
    pixel_size: float
    unit: str
    roi: list | None = None
    extra: dict = field(default_factory=dict)


def nucleus_mask(g: np.ndarray, p: Params):
    if p.whole_image:
        return np.ones(g.shape, bool)
    blur = filters.gaussian(g, 8)
    nuc = blur > filters.threshold_otsu(blur) * p.nucleus_scale
    nuc = ndi.binary_fill_holes(morphology.binary_closing(nuc, morphology.disk(15)))
    lab = measure.label(nuc)
    if lab.max() == 0:
        raise ValueError("No nucleus found - try lowering 'Nucleus detection' sensitivity.")
    sizes = np.bincount(lab.ravel())[1:]
    if p.all_nuclei:
        nuc = np.isin(lab, np.where(sizes > 0.05 * sizes.max())[0] + 1)
    else:
        nuc = lab == np.argmax(sizes) + 1
    return ndi.binary_fill_holes(nuc)


def prepare(g: np.ndarray, p: Params) -> dict:
    """Slow, threshold-independent steps (cached by the app)."""
    nuc = nucleus_mask(g, p)
    if p.method == "trace":
        width, _ = M.estimate_fiber_width(g, nuc)
        if p.line_sigma <= 0:
            auto_sigma, sigma_scan = M.select_sigma(g, nuc)
        else:
            auto_sigma, sigma_scan = None, {}
        sigma = auto_sigma if p.line_sigma <= 0 else p.line_sigma
        resp, ang = M.line_filter(g, sigma)
        nms = M.nonmax_suppress(resp, ang)
        flat = g / np.maximum(filters.gaussian(g, 7 * sigma), 1e-6)
        vals = nms[nuc & (nms > 0)]
        auto = float(filters.threshold_otsu(vals)) if vals.size else 1.0
        return {"nuc": nuc, "resp": resp, "nms": nms, "flat": flat, "auto": auto,
                "sigma": sigma, "width": width, "auto_sigma": auto_sigma, "sigma_scan": sigma_scan}
    gs = filters.gaussian(g, p.smooth_sigma) * 255 if p.smooth_sigma > 0 else g * 255
    return {"nuc": nuc, "gs": gs, "auto": float(filters.threshold_otsu(gs[nuc]))}


def segment(g: np.ndarray, p: Params, prep=None):
    """Returns (nucleus mask, structure mask, face label image, automatic threshold).

    trace:     structure = 1-px meshwork edges (fibre centre lines); faces = regions enclosed by edges.
    threshold: structure = white (filament) pixels; faces = connected black regions.
    """
    prep = prep if prep is not None else prepare(g, p)
    nuc, auto = prep["nuc"], prep["auto"]
    if p.method == "trace":
        gap = p.max_gap if p.max_gap >= 0 else int(max(4, round(5 * prep["sigma"])))
        struct = M.trace_meshwork(g, nuc, sensitivity=p.sensitivity, max_gap=gap,
                                  quality=p.edge_quality, auto=auto,
                                  prep=(prep["resp"], prep["nms"], prep["flat"]))
    else:
        thr = auto if p.threshold < 0 else p.threshold
        struct = prep["gs"] > thr
        if p.max_puncta_px > 0:
            struct = morphology.remove_small_objects(struct, p.max_puncta_px)  # white specks -> black
    faces = measure.label(nuc & ~struct, connectivity=1)
    faces = morphology.remove_small_objects(faces, p.min_area_px)
    info = ({"sigma": prep["sigma"], "fiber_width": prep["width"], "auto_sigma": prep["auto_sigma"], "gap": gap,
             "sigma_scan": prep.get("sigma_scan", {})}
            if p.method == "trace" else {})
    return nuc, struct, faces, auto, info


def roi_mask(shape, roi):
    """roi: list of (x, y) polygon vertices in pixel coordinates."""
    if not roi or len(roi) < 3:
        return np.ones(shape, bool)
    xy = np.array(roi, float)
    return draw.polygon2mask(shape, xy[:, ::-1])


def restrict(nuc, fil, faces_all, edge_margin, roi):
    core = morphology.binary_erosion(nuc, morphology.disk(edge_margin)) if edge_margin > 0 else nuc.copy()
    core &= roi_mask(nuc.shape, roi)
    faces = faces_all.copy()
    faces[np.isin(faces, np.unique(faces[~core]))] = 0
    faces, _, _ = segmentation.relabel_sequential(faces)
    return core, faces


def measure_faces(faces, px):
    cols = ("label", "area", "perimeter", "equivalent_diameter_area", "major_axis_length",
            "minor_axis_length", "eccentricity", "solidity", "centroid")
    if faces.max() == 0:
        return pd.DataFrame(columns=["face", "area", "perimeter", "equiv_diameter", "major_axis",
                                     "minor_axis", "eccentricity", "solidity", "circularity", "x", "y"])
    t = pd.DataFrame(measure.regionprops_table(faces, properties=cols)).rename(columns={
        "label": "face", "equivalent_diameter_area": "equiv_diameter", "major_axis_length": "major_axis",
        "minor_axis_length": "minor_axis", "centroid-0": "y", "centroid-1": "x"})
    t["circularity"] = np.clip(4 * np.pi * t.area / np.maximum(t.perimeter, 1e-9) ** 2, 0, 1)
    t["area"] *= px ** 2
    for c in ("perimeter", "equiv_diameter", "major_axis", "minor_axis", "x", "y"):
        t[c] *= px
    order = ["face", "area", "perimeter", "equiv_diameter", "major_axis", "minor_axis",
             "circularity", "eccentricity", "solidity", "x", "y"]
    return t[order]


def summarize(t, core, fil, px, unit, method="threshold"):
    a = t.area
    region = core.sum() * px ** 2
    n = len(t)
    u2 = f"{unit}²"
    return {
        "Faces (n)": n,
        f"Mean area ({u2})": a.mean() if n else np.nan,
        f"Median area ({u2})": a.median() if n else np.nan,
        f"SD ({u2})": a.std() if n > 1 else np.nan,
        f"SEM ({u2})": a.std() / np.sqrt(n) if n > 1 else np.nan,
        f"Min ({u2})": a.min() if n else np.nan,
        f"Q1 ({u2})": a.quantile(.25) if n else np.nan,
        f"Q3 ({u2})": a.quantile(.75) if n else np.nan,
        f"Max ({u2})": a.max() if n else np.nan,
        "CV (%)": 100 * a.std() / a.mean() if n > 1 else np.nan,
        f"Mean equiv. diameter ({unit})": t.equiv_diameter.mean() if n else np.nan,
        "Mean circularity": t.circularity.mean() if n else np.nan,
        f"Total face area ({u2})": a.sum(),
        f"Analyzed region ({u2})": region,
        "Face coverage (%)": 100 * a.sum() / region if region else np.nan,
        f"Face density (per 1000 {u2})": 1000 * n / region if region else np.nan,
    } | ({
        "Median edges per face": t.edges.median() if n else np.nan,
        f"Median edge length per face ({unit})": t.mean_edge_length.median() if n else np.nan,
    } if method == "trace" and "edges" in t else {
        "Filament coverage (%)": 100 * (fil & core).sum() / core.sum() if core.sum() else np.nan,
    })


def analyze(img: LoadedImage, p: Params, pixel_size=1.0, unit="px", roi=None, seg=None, edits=None) -> Result:
    """edits: manual corrections, list of ("erase", x, y) / ("draw", x0, y0, x1, y1) / ("exclude", x, y)."""
    g = img.channels[..., min(p.channel, img.channels.shape[-1] - 1)]
    nuc, struct, faces_all, auto_thr, info = seg if seg is not None else segment(g, p)
    edits = [tuple(e) for e in (edits or [])]
    if edits:
        line_ops = [e for e in edits if e[0] in ("erase", "draw")]
        if line_ops:
            struct = M.apply_edits(struct, line_ops) if p.method == "trace" else \
                M.apply_edits(struct, [e for e in line_ops if e[0] == "draw"])
            faces_all = measure.label(nuc & ~struct, connectivity=1)
            faces_all = morphology.remove_small_objects(faces_all, p.min_area_px)
        for e in edits:
            if e[0] == "exclude":
                y, x = int(round(e[2])), int(round(e[1]))
                if 0 <= y < faces_all.shape[0] and 0 <= x < faces_all.shape[1] and faces_all[y, x]:
                    faces_all = np.where(faces_all == faces_all[y, x], 0, faces_all)
    fil = struct & nuc
    core, faces = restrict(nuc, fil, faces_all, p.edge_margin, roi)
    t = measure_faces(faces, pixel_size)
    if p.method == "trace" and len(t):
        n_e, mean_len = M.face_edge_stats(faces, struct, pixel_size)
        t.insert(t.columns.get_loc("eccentricity"), "edges", n_e[t.face.values.astype(int)])
        t.insert(t.columns.get_loc("eccentricity"), "mean_edge_length", mean_len[t.face.values.astype(int)])
    used = (auto_thr * p.sensitivity) if p.method == "trace" else (auto_thr if p.threshold < 0 else p.threshold)
    return Result(img.name, p, nuc, core, fil, faces, t,
                  summarize(t, core, fil, pixel_size, unit, p.method),
                  auto_thr, pixel_size, unit, roi,
                  extra={"bw": struct, "threshold": used, "method": p.method, "edits": edits, **info})


# ----------------------------------------------------------------------------- image overlays
def _boundary(mask_or_labels):
    return segmentation.find_boundaries(mask_or_labels, mode="inner")


def _thick(mask, r):
    return morphology.binary_dilation(mask, morphology.disk(r)) if r > 0 else mask


# 100 distinct fills for individual faces: 25 hues (golden-angle spacing) x 4 saturation/brightness tiers
def _make_face_palette(n=100):
    import colorsys
    tiers = [(0.95, 1.00), (0.60, 0.95), (1.00, 0.72), (0.45, 0.80)]     # (saturation, value)
    cols = []
    for i in range(n):
        h = (i * 0.618033988749895) % 1.0
        sat, val = tiers[i % len(tiers)]
        cols.append(tuple(int(round(c * 255)) for c in colorsys.hsv_to_rgb(h, sat, val)))
    return np.array(cols)


FACE_COLORS = _make_face_palette(100)


def _marks(out, r):
    lw = max(1, round(max(r.faces.shape) / 900))
    out[_thick(_boundary(r.core), lw)] = CORE_RGB
    if r.roi:
        out[_thick(_boundary(roi_mask(r.core.shape, r.roi)), lw)] = ROI_RGB
    return out


MESH_RGB = (255, 0, 255)         # detected meshwork edges (magenta, as in Shimi et al.)


def _gray(img: LoadedImage, r: Result, scale=255.0):
    g = img.channels[..., min(r.params.channel, img.channels.shape[-1] - 1)]
    return np.repeat((g * scale)[..., None], 3, axis=2)


def _lines(struct, shape, lw=None):
    """Display thickness of traced lines (lw = extra pixels on each side; None = auto from image size)."""
    return _thick(struct, int(max(shape) / 2000) if lw is None else lw)


def structure_view(img: LoadedImage, r: Result, lw=None):
    """Second panel: traced meshwork (magenta) on the image, or the black & white image."""
    s_ = r.extra.get("bw", r.fil)
    if r.extra.get("method") == "trace":
        out = _gray(img, r, 230.0)
        out[_lines(s_, s_.shape, lw)] = MESH_RGB
    else:
        out = np.repeat((s_ * 255.0)[..., None], 3, axis=2)
    return _marks(out.astype(np.uint8), r)


def black_white(r: Result):
    """The thresholded image: white = filament, black = faces/background."""
    bw = r.extra.get("bw", r.fil)
    out = np.repeat((bw * 255).astype(np.uint8)[..., None], 3, axis=2)
    return _marks(out, r)


def face_color_index(faces: np.ndarray, n_colors: int = len(FACE_COLORS)) -> np.ndarray:
    """Assign each face one of the palette colors so that neighbouring faces (across a filament)
    look as different as possible: each face takes the color farthest (in CIELAB) from its
    already-colored neighbours, preferring the least-used colors so all 100 get used.
    Returns an array indexed by face label -> color index."""
    from skimage.color import rgb2lab
    n = int(faces.max())
    idx = np.zeros(n + 1, int)
    if n == 0:
        return idx
    lab_cols = rgb2lab(FACE_COLORS[None, :n_colors] / 255.0)[0]
    dist = np.linalg.norm(lab_cols[:, None] - lab_cols[None], axis=-1)      # color-to-color distance
    grown = segmentation.expand_labels(faces, distance=12)
    pairs = set()
    for a, b in ((grown[:, :-1], grown[:, 1:]), (grown[:-1, :], grown[1:, :])):
        m = (a != b) & (a > 0) & (b > 0)
        pairs.update(zip(a[m].tolist(), b[m].tolist()))
    nbrs = [[] for _ in range(n + 1)]
    for a, b in pairs:
        nbrs[a].append(b); nbrs[b].append(a)
    usage = np.zeros(n_colors)
    rng = np.random.default_rng(7)
    assigned = np.zeros(n + 1, bool)
    for lab in rng.permutation(np.arange(1, n + 1)):
        near = [idx[j] for j in nbrs[lab] if assigned[j]]
        score = dist[:, near].min(axis=1) if near else np.full(n_colors, 100.0)
        score = np.minimum(score, 60.0) - 0.5 * usage                  # far enough, then spread usage
        c = int(np.argmax(score + rng.uniform(0, 0.01, n_colors)))
        idx[lab] = c; usage[c] += 1; assigned[lab] = True
    return idx


def filled_faces(img: LoadedImage, r: Result, single_color=None, alpha=None, lw=None):
    """Every measured face filled with its color, on the outlined image: over the traced meshwork
    (white edges) for the trace method, over the black & white image for the threshold method.
    Areas that are not measured (touching the rim / region edge) are left uncolored."""
    if alpha is None:
        alpha = 0.85 if r.extra.get("method") == "trace" else 1.0
    s_ = r.extra.get("bw", r.fil)
    trace = r.extra.get("method") == "trace"
    base = _gray(img, r, 150.0) if trace else np.repeat((s_ * 255.0)[..., None], 3, axis=2)
    base[~r.nuc] *= 0.5                                  # soften signal outside the nucleus
    m = r.faces > 0
    if m.any():
        if single_color is not None:
            col = np.broadcast_to(np.array(single_color, float), base.shape)
        else:
            col = FACE_COLORS[face_color_index(r.faces)[r.faces]].astype(float)
        base[m] = (1 - alpha) * base[m] + alpha * col[m]
    if trace:                                            # meshwork edges drawn on top of the faces
        base[_lines(s_, s_.shape, lw)] = (255, 255, 255)
    return _marks(base.astype(np.uint8), r)


def highlight_face(rgb: np.ndarray, faces: np.ndarray, face_id: int, dim=0.55):
    """Dim everything except one face and draw a bright ring around it."""
    m = faces == face_id
    if not m.any():
        return rgb
    out = (rgb.astype(float) * dim).astype(np.uint8)
    out[m] = rgb[m]
    ys, xs = np.nonzero(m)
    pad = 6
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, m.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, m.shape[1])
    sub = m[y0:y1, x0:x1]
    lw = max(2, round(max(faces.shape) / 600))
    ring = morphology.binary_dilation(sub, morphology.disk(lw)) & ~sub
    out[y0:y1, x0:x1][ring] = (255, 255, 255)
    return out


# kept for backwards compatibility
overlay_outlines = filled_faces


def outlines_only(r: Result):
    return black_white(r)


def area_map(img: LoadedImage, r: Result):
    g = img.channels[..., min(r.params.channel, img.channels.shape[-1] - 1)]
    base = np.repeat((g * 150)[..., None], 3, axis=2)
    if len(r.table):
        areas = np.zeros(r.faces.max() + 1)
        areas[r.table.face.values] = r.table.area.values
        norm = LogNorm(vmin=max(r.table.area.min(), 1e-9), vmax=r.table.area.max() * 1.0001)
        rgba = SEQ_BLUE(norm(np.where(r.faces > 0, areas[r.faces], np.nan)))
        m = r.faces > 0
        base[m] = rgba[m, :3] * 255
    base[_thick(_boundary(r.core), max(1, round(max(r.faces.shape) / 900)))] = CORE_RGB
    return base.astype(np.uint8)


# ----------------------------------------------------------------------------- charts
def _log_bins(a, n=30):
    lo, hi = max(a.min(), 1e-9), a.max()
    if hi <= lo:
        hi = lo * 1.1
    return np.logspace(np.log10(lo), np.log10(hi), n)


def _mark_center(ax, r, u2):
    a = r.table.area
    ax.axvline(a.median(), color=INK, lw=1.2, ls="-")
    ax.axvline(a.mean(), color=ORANGE, lw=1.2, ls="--")
    ax.legend([plt.Line2D([], [], color=INK, lw=1.2), plt.Line2D([], [], color=ORANGE, lw=1.2, ls="--")],
              [f"median {a.median():.3g}", f"mean {a.mean():.3g}"], loc="upper right", fontsize=8)


def fig_area_hist(r: Result, ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(5, 3.2))
    u2 = f"{r.unit}²"
    a = r.table.area
    if len(a):
        ax.hist(a, bins=_log_bins(a), color=BLUE, rwidth=0.85)
        ax.set_xscale("log")
        _mark_center(ax, r, u2)
    ax.set_xlabel(f"Face area ({u2}, log scale)")
    ax.set_ylabel("Number of faces")
    ax.set_title("Area distribution")
    return ax.figure if own else None


def fig_ecdf(r: Result, ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(5, 3.2))
    a = np.sort(r.table.area.values)
    if len(a):
        ax.step(a, np.arange(1, len(a) + 1) / len(a) * 100, where="post", color=BLUE, lw=2)
        ax.set_xscale("log")
        ax.axhline(50, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel(f"Face area ({r.unit}², log scale)")
    ax.set_ylabel("Cumulative % of faces")
    ax.set_ylim(0, 101)
    ax.set_title("Cumulative distribution")
    return ax.figure if own else None


def fig_area_fraction(r: Result, ax=None):
    """Share of total face area contributed by each size class - shows whether big faces dominate."""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(5, 3.2))
    a = r.table.area
    if len(a):
        bins = _log_bins(a, 16)
        w, edges = np.histogram(a, bins=bins, weights=a)
        ax.bar(edges[:-1], 100 * w / a.sum(), width=np.diff(edges) * 0.85, align="edge", color=AQUA)
        ax.set_xscale("log")
    ax.set_xlabel(f"Face area ({r.unit}², log scale)")
    ax.set_ylabel("% of total face area")
    ax.set_title("Area contributed by size class")
    return ax.figure if own else None


def fig_shape(r: Result, ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(5, 3.2))
    t = r.table
    if len(t):
        ax.scatter(t.area, t.circularity, s=12, color=BLUE, alpha=0.6, edgecolors="white", linewidths=0.4)
        ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"Face area ({r.unit}², log scale)")
    ax.set_ylabel("Circularity (1 = circle)")
    ax.set_title("Size vs shape")
    return ax.figure if own else None


def fig_panel(r: Result):
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.6))
    fig_area_hist(r, axs[0, 0]); fig_ecdf(r, axs[0, 1])
    fig_area_fraction(r, axs[1, 0]); fig_shape(r, axs[1, 1])
    fig.tight_layout()
    return fig


def fig_compare(results: list[Result]):
    """Per-image box + points (one color; images identified on the x-axis)."""
    n = len(results)
    fig, ax = plt.subplots(figsize=(max(5, 1.1 * n + 2), 3.6))
    data = [r.table.area.values for r in results]
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        if len(d):
            ax.scatter(i + 1 + rng.uniform(-0.18, 0.18, len(d)), d, s=6, color=BLUE, alpha=0.35, linewidths=0)
    ax.boxplot([d if len(d) else [np.nan] for d in data], widths=0.5, showfliers=False,
               medianprops=dict(color=INK, lw=1.6), boxprops=dict(color=INK2),
               whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    ax.set_yscale("log")
    ax.set_xticks(range(1, n + 1), [_short(r.name) for r in results], rotation=30 if n > 3 else 0, ha="right" if n > 3 else "center")
    ax.set_ylabel(f"Face area ({results[0].unit}², log scale)")
    ax.set_title("Face area by image")
    fig.tight_layout()
    return fig


def _short(s, n=22):
    s = s.rsplit(".", 1)[0]
    return s if len(s) <= n else s[:n - 1] + "…"


# ----------------------------------------------------------------------------- tables
def fmt(v):
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:,.1f}" if abs(v) >= 100 else f"{v:.3g}"


def summary_table(results: list[Result]) -> pd.DataFrame:
    return pd.DataFrame({r.name: r.stats for r in results})


def pooled_stats(results: list[Result]) -> dict:
    t = pd.concat([r.table for r in results]) if results else pd.DataFrame()
    per_img_means = [r.table.area.mean() for r in results if len(r.table)]
    u2 = f"{results[0].unit}²"
    return {
        "Images": len(results),
        "Faces, total": int(len(t)),
        f"Pooled mean area ({u2})": t.area.mean() if len(t) else np.nan,
        f"Pooled median area ({u2})": t.area.median() if len(t) else np.nan,
        f"Mean of image means ({u2})": np.mean(per_img_means) if per_img_means else np.nan,
        f"SD of image means ({u2})": np.std(per_img_means, ddof=1) if len(per_img_means) > 1 else np.nan,
    }


def face_table(r: Result) -> pd.DataFrame:
    """Numbered face table as shown to users / exported."""
    u, u2 = r.unit, f"{r.unit}²"
    t = r.table
    return pd.DataFrame({
        "Face #": t.face.astype(int).values,
        f"Area ({u2})": t.area.round(3).values,
        f"Eq. diameter ({u})": t.equiv_diameter.round(3).values,
        f"Perimeter ({u})": t.perimeter.round(3).values,
        "Circularity": t.circularity.round(3).values,
        **({"Edges": t.edges.astype(int).values,
            f"Mean edge length ({u})": t.mean_edge_length.round(3).values} if "edges" in t else {}),
        "Solidity": t.solidity.round(3).values,
        f"X ({u})": t.x.round(1).values,
        f"Y ({u})": t.y.round(1).values,
    })


def training_bundle(img: LoadedImage, r: Result) -> dict[str, bytes]:
    """Image + corrected segmentation as files, for building a training / validation set."""
    import json
    g = img.channels[..., min(r.params.channel, img.channels.shape[-1] - 1)]
    def tif(a):
        b = io.BytesIO(); tifffile.imwrite(b, a, compression="zlib"); return b.getvalue()
    info = {"source_image": img.name, "created": dt.datetime.now().isoformat(timespec="seconds"),
            "method": r.params.method, "params": asdict(r.params), "edits": [list(e) for e in r.extra.get("edits", [])],
            "pixel_size": r.pixel_size, "unit": r.unit, "faces_measured": int(len(r.table)),
            **{k: r.extra[k] for k in ("sigma", "fiber_width", "gap") if k in r.extra},
            "files": {"image.tif": "lamin channel, uint16 (normalized 0-65535)",
                      "meshwork.tif": "uint8, 255 = traced fibre centre line (black & white method: filament mask)",
                      "faces.tif": "uint16 label image of measured faces (numbers match the Excel table)",
                      "nucleus.tif": "uint8, 255 = nucleus", "analysis_region.tif": "uint8, 255 = rim-excluded region / ROI"}}
    return {"image.tif": tif((g * 65535).astype(np.uint16)),
            "meshwork.tif": tif((r.extra.get("bw", r.fil) * 255).astype(np.uint8)),
            "faces.tif": tif(r.faces.astype(np.uint16)),
            "nucleus.tif": tif((r.nuc * 255).astype(np.uint8)),
            "analysis_region.tif": tif((r.core * 255).astype(np.uint8)),
            "faces.csv": face_table(r).to_csv(index=False).encode(),
            "info.json": json.dumps(info, indent=2, default=str).encode()}


def excel_bytes(results: list[Result]) -> bytes:
    """Workbook: 'Summary' sheet + one sheet of numbered faces per image."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        summ = pd.DataFrame({r.name: r.stats for r in results})
        summ.index.name = "Measure"
        summ.to_excel(xw, sheet_name="Summary")
        used = set()
        for r in results:
            base = re.sub(r"[\\/*?:\[\]]", "_", r.name.rsplit(".", 1)[0])[:28] or "image"
            nm, k = base, 2
            while nm.lower() in used:
                nm, k = f"{base[:25]}_{k}", k + 1
            used.add(nm.lower())
            face_table(r).to_excel(xw, sheet_name=nm, index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 45)
            ws.freeze_panes = "B2" if ws.title == "Summary" else "A2"
    return buf.getvalue()


# ----------------------------------------------------------------------------- PDF report
def _table_on_ax(ax, rows, col_labels=None, col_widths=None, fontsize=8):
    ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=col_labels, loc="upper left", cellLoc="left",
                  colLoc="left", colWidths=col_widths, edges="horizontal")
    tb.auto_set_font_size(False)
    tb.set_fontsize(fontsize)
    tb.scale(1, 1.25)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor(GRID)
        c.get_text().set_color(INK if i else INK2)
        if i == 0 and col_labels:
            c.get_text().set_weight("bold")
    return tb


def _page(title, subtitle=None):
    fig = plt.figure(figsize=(8.27, 11.69))       # A4 portrait
    fig.text(0.07, 0.955, title, fontsize=15, weight="bold", color=INK)
    if subtitle:
        fig.text(0.07, 0.935, subtitle, fontsize=9, color=INK2)
    return fig


def _footer(fig, page):
    fig.text(0.07, 0.025, "Lamin meshwork analysis report", fontsize=7, color=MUTED)
    fig.text(0.93, 0.025, f"page {page}", fontsize=7, color=MUTED, ha="right")


PARAM_LABELS = {
    "channel": "Lamin channel (0-based)", "smooth_sigma": "Smoothing sigma (px)",
    "threshold": "Threshold (0-255, -1 = auto)", "nucleus_scale": "Nucleus detection (x Otsu)",
    "method": "Method", "line_sigma": "Line filter sigma (px)", "sensitivity": "Line sensitivity (x auto)",
    "max_gap": "Max. gap closed (px)", "edge_quality": "Min. edge quality",
    "all_nuclei": "Analyze all nuclei", "whole_image": "Whole image is nucleus (crop)",
    "edge_margin": "Rim exclusion (px)",
    "min_area_px": "Min face area (px)", "max_puncta_px": "Ignored white specks (px)",
}


def build_report(results: list[Result], images: dict[str, LoadedImage], title="Lamin meshwork analysis",
                 notes="", include_face_tables=False) -> bytes:
    buf = io.BytesIO()
    page = 1
    unit = results[0].unit
    with PdfPages(buf) as pdf:
        # --- page 1: overview
        fig = _page(title, f"Generated {dt.datetime.now():%Y-%m-%d %H:%M}  ·  {len(results)} image(s)  ·  "
                           f"scale: {results[0].pixel_size:g} {unit}/px")
        y = 0.90
        if notes:
            fig.text(0.07, y, textwrap.fill(notes, 100), fontsize=9, color=INK, va="top")
            y -= 0.05
        fig.text(0.07, y, "Summary per image", fontsize=11, weight="bold", color=INK)
        keys = list(results[0].stats.keys())
        cols = [_short(r.name, 16) for r in results[:5]]
        rows = [[k] + [fmt(r.stats[k]) for r in results[:5]] for k in keys]
        ax = fig.add_axes([0.07, y - 0.40, 0.86, 0.385])
        _table_on_ax(ax, rows, ["Measure"] + cols, [0.36] + [0.64 / len(cols)] * len(cols))
        y -= 0.02 + 0.017 * (len(rows) + 1) + 0.045       # rows are ~0.017 of page height each
        if len(results) > 5:
            fig.text(0.07, y + 0.025, f"(first 5 images shown; all {len(results)} in the per-image pages and CSV)",
                     fontsize=7, color=MUTED)
        if len(results) > 1:
            ps = pooled_stats(results)
            fig.text(0.07, y, "Pooled", fontsize=11, weight="bold", color=INK)
            ax = fig.add_axes([0.07, y - 0.13, 0.5, 0.115])
            _table_on_ax(ax, [[k, fmt(v)] for k, v in ps.items()], None, [0.7, 0.3])
            cf = fig_compare(results)
            cf.canvas.draw()
            img = np.asarray(cf.canvas.buffer_rgba())
            plt.close(cf)
            ax = fig.add_axes([0.07, 0.05, 0.86, y - 0.20]); ax.imshow(img); ax.axis("off")
        else:
            fig.text(0.07, y, "Methods", fontsize=11, weight="bold", color=INK)
            fig.text(0.07, y - 0.02, textwrap.fill(METHODS_TRACE if results[0].extra.get('method') == 'trace' else METHODS_THRESHOLD, 115), fontsize=8, color=INK2, va="top", linespacing=1.5)
        _footer(fig, page); pdf.savefig(fig); plt.close(fig); page += 1

        # --- per image
        for r in results:
            im = images[r.name]
            fig = _page(r.name, f"{r.stats['Faces (n)']} faces · median {fmt(r.stats[f'Median area ({unit}²)'])} "
                                f"{unit}² · mean {fmt(r.stats[f'Mean area ({unit}²)'])} {unit}²"
                                + ("  ·  custom region" if r.roi else ""))
            second = "Traced meshwork" if r.extra.get("method") == "trace" else "Black & white"
            for i, (ttl, pic) in enumerate([("Original", im.display), (second, structure_view(im, r)),
                                             ("Measured faces (filled)", filled_faces(im, r))]):
                ax = fig.add_axes([0.05 + i * 0.305, 0.62, 0.29, 0.29]); ax.imshow(pic); ax.axis("off")
                ax.set_title(ttl, fontsize=8, color=INK2)
            fig.text(0.05, 0.605, (f"Meshwork tracing (Shimi et al. 2015) · line sensitivity {r.params.sensitivity:.2f}"
                                   f" × auto · gaps ≤ {r.params.max_gap} px closed"
                                   if r.extra.get("method") == "trace" else
                                   f"Threshold {r.extra.get('threshold', 0):.0f} / 255 (auto {r.auto_threshold:.0f})")
                     + " · colored: measured faces · red: analysis boundary (rim excluded)"
                     + (" · cyan: user region" if r.roi else ""), fontsize=7, color=MUTED)
            ax = fig.add_axes([0.52, 0.30, 0.42, 0.28]); ax.imshow(area_map(im, r)); ax.axis("off")
            ax.set_title("Faces colored by area", fontsize=8, color=INK2)
            if len(r.table):
                sm = plt.cm.ScalarMappable(norm=LogNorm(r.table.area.min(), r.table.area.max()), cmap=SEQ_BLUE)
                cax = fig.add_axes([0.58, 0.295, 0.30, 0.007])
                cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
                cb.ax.tick_params(labelsize=6, colors=INK2); cb.outline.set_visible(False)
                cb.set_label(f"area ({unit}²)", fontsize=6, color=INK2)
            ax = fig.add_axes([0.07, 0.25, 0.40, 0.33])
            _table_on_ax(ax, [[k, fmt(v)] for k, v in r.stats.items()], ["Measure", "Value"], [0.72, 0.28], 7.5)
            ax = fig.add_axes([0.55, 0.04, 0.38, 0.215])
            skip = ({"smooth_sigma", "threshold", "max_puncta_px"} if r.params.method == "trace"
                    else {"line_sigma", "sensitivity", "max_gap", "edge_quality"})
            _table_on_ax(ax, [[PARAM_LABELS.get(k, k), str(v)] for k, v in asdict(r.params).items() if k not in skip]
                         + [["Threshold used", f"{r.extra.get('threshold', 0):.4g}"],
                            ["Automatic threshold (Otsu)", f"{r.auto_threshold:.4g}"]],
                         ["Setting", "Value"], [0.72, 0.28], 7)
            _footer(fig, page); pdf.savefig(fig); plt.close(fig); page += 1

            fig = fig_panel(r)
            fig.set_size_inches(8.27, 11.69)
            fig.subplots_adjust(top=0.88, bottom=0.30, hspace=0.45, wspace=0.32, left=0.1, right=0.95)
            fig.text(0.07, 0.955, f"{r.name} — distributions", fontsize=15, weight="bold", color=INK)
            _footer(fig, page); pdf.savefig(fig); plt.close(fig); page += 1

            if len(r.table):          # numbered faces page (numbers match the table / Excel export)
                fig = _page(f"{r.name} — numbered faces",
                            "Face numbers match the per-face table and the Excel export. Zoom into the PDF to read small labels.")
                ax = fig.add_axes([0.04, 0.06, 0.92, 0.85]); ax.imshow(filled_faces(im, r)); ax.axis("off")
                t = r.table
                big = t.equiv_diameter.values / r.pixel_size
                for fid, x, y, d in zip(t.face.values, t.x.values / r.pixel_size, t.y.values / r.pixel_size, big):
                    ax.text(x, y, str(int(fid)), ha="center", va="center", color="white",
                            fontsize=float(np.clip(d / 12, 2.2, 6.5)), weight="bold")
                _footer(fig, page); pdf.savefig(fig); plt.close(fig); page += 1

            if include_face_tables and len(r.table):
                cols = ["face", "area", "perimeter", "equiv_diameter", "circularity", "solidity", "x", "y"]
                per = 48
                for s in range(0, len(r.table), per):
                    fig = _page(f"{r.name} — faces {s + 1}–{min(s + per, len(r.table))}")
                    rows = [[fmt(int(v)) if c == "face" else fmt(float(v)) for c, v in zip(cols, row)]
                            for row in r.table[cols].iloc[s:s + per].itertuples(index=False)]
                    ax = fig.add_axes([0.07, 0.06, 0.86, 0.86])
                    _table_on_ax(ax, rows, cols, None, 7)
                    _footer(fig, page); pdf.savefig(fig); plt.close(fig); page += 1

        if len(results) > 1:
            fig = _page("Methods")
            fig.text(0.07, 0.91, textwrap.fill(METHODS_TRACE if results[0].extra.get('method') == 'trace' else METHODS_THRESHOLD, 100), fontsize=9, color=INK2, va="top", linespacing=1.5)
            _footer(fig, page); pdf.savefig(fig); plt.close(fig)
    return buf.getvalue()


METHODS_THRESHOLD = (
    "The lamin channel was isolated, normalized to a 0-255 scale and lightly smoothed with a Gaussian filter. "
    "The nucleus was detected by Otsu thresholding of a heavily blurred image with hole filling. The image "
    "was then converted to black and white with a single intensity threshold (automatic: Otsu's method on "
    "the pixels inside the nucleus; adjustable by the user): pixels above the threshold are lamin filament "
    "(white), pixels below are mesh openings (black). Isolated white specks smaller than a set size were "
    "treated as black. Each connected black region inside the nucleus is one face; its area is its black "
    "pixel count. Faces touching the rim-exclusion band (a fixed distance inward from the nuclear edge) or "
    "crossing a user-drawn region boundary were excluded. Area, perimeter, equivalent diameter, circularity "
    "(4*pi*A/P^2), eccentricity and solidity were measured per face."
)

METHODS_TRACE = (
    "Meshworks were segmented following the approach of Shimi et al. (Mol Biol Cell 2015, "
    "doi:10.1091/mbc.E15-07-0461). The lamin channel was processed with an oriented line filter (a bank of "
    "elongated Gaussian second-derivative kernels at 16 orientations), and non-maximal suppression across the "
    "line direction gave 1-pixel centre lines of the lamin fibres. Centre lines were kept by hysteresis "
    "thresholding of the filter response (high threshold = Otsu's threshold of the response inside the nucleus "
    "times a user-set sensitivity; low = half of that) and only if at least 5 pixels long. Gaps at junctions "
    "were closed by extending each free end along its local direction (from the centroid of its last 6 "
    "pixels) until it met another fibre, up to a maximum gap length; branches that still did not end in a "
    "junction were removed by iterative spur pruning. Edges (fibre segments between junctions) were audited "
    "by their mean intensity in a flattened image (image divided by its Gaussian blur) and weak edges were "
    "removed. Faces are the regions enclosed by edges; face area therefore extends to the fibre centre lines. "
    "Faces touching the rim-exclusion band or crossing a user-drawn region were excluded. Per face, area, "
    "perimeter, equivalent diameter, circularity (4*pi*A/P^2), eccentricity, the number of bordering edges "
    "and their mean length were measured. This is an independent Python re-implementation, not the authors' "
    "MATLAB code."
)

METHODS = METHODS_TRACE
