"""
Make a synthetic lamin-meshwork image (no real microscopy data), for the public example and tests.

    python example/make_synthetic_example.py      -> example/synthetic_mesh.tif

An elliptical "nucleus" is filled with a Voronoi meshwork: fibres of 2-4 px with varied opening sizes (a few large
openings among many small ones, as in lamin A/C images), a bright rim, a blur standing in for the microscope, and
camera-like noise. Deterministic (fixed seed), so tests can pin the measurements.
"""
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import draw, filters, morphology, segmentation

OUT = Path(__file__).with_name("synthetic_mesh.tif")


def make(size=768, seed=7):
    rng = np.random.default_rng(seed)
    h = w = size
    nuc = np.zeros((h, w), bool)
    rr, cc = draw.ellipse(h // 2, w // 2, int(h * 0.42), int(w * 0.32), shape=(h, w), rotation=np.deg2rad(25))
    nuc[rr, cc] = True

    # seeds: many close together (fine mesh) and a few far apart (large openings)
    pts = []
    while len(pts) < 900:
        y, x = rng.integers(0, h), rng.integers(0, w)
        if not nuc[y, x]:
            continue
        gap = 9 if rng.random() < 0.85 else 22
        if all((y - py) ** 2 + (x - px) ** 2 > gap ** 2 for py, px in pts[-400:]):
            pts.append((y, x))
    markers = np.zeros((h, w), int)
    for i, (y, x) in enumerate(pts, 1):
        markers[y, x] = i
    cells = segmentation.watershed(ndi.distance_transform_edt(markers == 0), markers, mask=nuc)
    fibres = segmentation.find_boundaries(cells, mode="thick") & nuc
    fibres = morphology.binary_dilation(fibres, morphology.disk(1))
    rim = nuc & ~morphology.binary_erosion(nuc, morphology.disk(6))

    img = np.zeros((h, w))
    img[fibres] = 0.75 + 0.25 * rng.random(fibres.sum())        # uneven labelling along fibres
    img[rim] = 1.0
    img[nuc] += 0.08                                            # dim nucleoplasm
    img = filters.gaussian(img, 1.3)                            # optics
    img = rng.poisson(img * 180 + 6) + rng.normal(0, 3, img.shape)
    return np.clip(img / img.max() * 65535, 0, 65535).astype(np.uint16)


if __name__ == "__main__":
    tifffile.imwrite(OUT, make(), compression="zlib")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")
