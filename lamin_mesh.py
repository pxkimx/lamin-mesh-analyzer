"""
Lamin A/C meshwork segmentation: outline filaments and measure enclosed (mesh-face) areas.

Pipeline
  1. Take the lamin (green) channel, normalize, light Gaussian denoise.
  2. Enhance thin bright filaments with a multiscale ridge filter (Sato) and combine
     with a local (adaptive) threshold -> binary filament mask.
  3. Build the nuclear mask (heavily smoothed signal, Otsu, fill holes) and erode it by
     EDGE_MARGIN px -> "analysis region". This excludes the dense rim at the outer edge.
  4. Mesh faces = connected components of NOT-filament inside the nucleus.
     Faces that touch/leave the analysis region are discarded (edge exclusion);
     faces smaller than MIN_AREA are discarded as noise.
  5. Report area (px^2 = arbitrary units), perimeter, equivalent diameter, circularity
     per face; save CSV, summary, and QC overlays.

Usage:  python lamin_mesh.py image.tif [--channel 1] [--out outdir]
"""
import argparse, os
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage import filters, morphology, measure, segmentation, exposure, io, color

# ---- tunable parameters (pixels) ----
SMOOTH_SIGMA   = 1.0     # denoise before ridge detection
RIDGE_SIGMAS   = (1, 2, 3)  # filament half-widths to detect
LOCAL_BLOCK    = 51      # adaptive threshold window (odd)
MIN_FIL_OBJ    = 30      # drop filament specks smaller than this
MAX_PUNCTA     = 150     # isolated bright dots inside a face smaller than this are filled in
LOCAL_OFFSET   = 0.03    # added to local threshold for the tight filament mask (higher = thinner filaments, larger faces)
EDGE_MARGIN    = 25      # px eroded from nuclear boundary; faces must lie fully inside
MIN_AREA       = 20      # smallest mesh face kept (px^2)
PIXEL_SIZE     = 1.0     # units per px (1.0 = arbitrary units)


def load_channel(path, ch):
    im = tifffile.imread(path)
    if im.ndim == 3 and im.shape[-1] in (3, 4):
        im = im[..., ch]
    elif im.ndim == 3:
        im = im[ch]
    return exposure.rescale_intensity(im.astype(float), out_range=(0, 1))


def segment(g):
    gs = filters.gaussian(g, SMOOTH_SIGMA)

    # nucleus mask
    blur = filters.gaussian(g, 8)
    nuc = blur > filters.threshold_otsu(blur) * 0.6
    nuc = ndi.binary_fill_holes(morphology.binary_closing(nuc, morphology.disk(15)))
    lab = measure.label(nuc)
    nuc = lab == np.argmax(np.bincount(lab.ravel())[1:]) + 1  # largest object
    nuc = ndi.binary_fill_holes(nuc)
    core = morphology.binary_erosion(nuc, morphology.disk(EDGE_MARGIN))

    # (a) generous filament mask (ridge OR locally bright) -> guarantees faces are separated
    ridge = filters.sato(gs, sigmas=RIDGE_SIGMAS, black_ridges=False)
    ridge_mask = ridge > filters.threshold_otsu(ridge[nuc])
    local_thr = filters.threshold_local(gs, LOCAL_BLOCK)
    fil_wide = (ridge_mask | (gs > local_thr - 0.01)) & nuc
    fil_wide = morphology.remove_small_objects(fil_wide, MIN_FIL_OBJ)
    seeds = morphology.remove_small_holes(nuc & ~fil_wide, MAX_PUNCTA)
    seeds = morphology.remove_small_objects(measure.label(seeds, connectivity=1), MIN_AREA)

    # (b) watershed on intensity: basins meet on filament ridges, even where a filament is faint
    basins = segmentation.watershed(gs, markers=seeds, mask=nuc, watershed_line=True)

    # (c) tight filament mask -> face = basin minus filament core (area reaches filament edge)
    fil = ((ridge_mask & (gs > local_thr + LOCAL_OFFSET)) | (basins == 0)) & nuc
    lumen = basins * ~fil
    faces = np.zeros_like(basins)
    for r in measure.regionprops(lumen):
        sub = measure.label(r.image, connectivity=1)
        keep = sub == np.argmax(np.bincount(sub.ravel())[1:]) + 1      # main piece only
        holes = ndi.binary_fill_holes(keep) & ~keep                      # truly enclosed only
        keep |= holes & ~morphology.remove_small_objects(holes, MAX_PUNCTA)  # fill small puncta
        sl = faces[r.slice]; sl[keep] = r.label
    faces = morphology.remove_small_objects(faces, MIN_AREA)
    fil = nuc & (faces == 0)

    # edge exclusion: drop any face with pixels outside the eroded core
    outside = np.unique(faces[~core])
    faces[np.isin(faces, outside)] = 0
    faces, _, _ = segmentation.relabel_sequential(faces)
    return gs, nuc, core, fil, faces


def measure_faces(faces):
    props = measure.regionprops_table(
        faces, properties=("label", "area", "perimeter", "equivalent_diameter_area",
                           "eccentricity", "solidity", "centroid"))
    df = pd.DataFrame(props).rename(columns={
        "equivalent_diameter_area": "equiv_diameter",
        "centroid-0": "y", "centroid-1": "x"})
    df["area"] *= PIXEL_SIZE ** 2
    df["perimeter"] *= PIXEL_SIZE
    df["equiv_diameter"] *= PIXEL_SIZE
    df["circularity"] = 4 * np.pi * df.area / np.maximum(df.perimeter, 1e-9) ** 2
    return df


def save_qc(g, nuc, core, fil, faces, outdir):
    g8 = (g * 255).astype(np.uint8)
    base = np.dstack([g8 * 0, g8, g8 * 0])
    # 1. filament outlines
    o1 = base.copy()
    o1[segmentation.find_boundaries(fil, mode="inner")] = (255, 255, 255)
    io.imsave(os.path.join(outdir, "qc_filaments.png"), o1, check_contrast=False)
    # 2. measured faces coloured + analysis region outline
    lab_rgb = (color.label2rgb(faces, bg_label=0, bg_color=(0, 0, 0)) * 255).astype(np.uint8)
    o2 = np.where(faces[..., None] > 0, (0.55 * lab_rgb + 0.45 * base).astype(np.uint8), base)
    o2[segmentation.find_boundaries(faces, mode="outer") & (faces == 0)] = (255, 255, 0)
    o2[segmentation.find_boundaries(core)] = (255, 0, 0)
    o2[segmentation.find_boundaries(nuc)] = (0, 160, 255)
    io.imsave(os.path.join(outdir, "qc_faces.png"), o2, check_contrast=False)
    io.imsave(os.path.join(outdir, "face_labels.tif"), faces.astype(np.uint16), check_contrast=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--channel", type=int, default=1, help="0=R 1=G 2=B")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    outdir = a.out or os.path.splitext(a.image)[0] + "_mesh"
    os.makedirs(outdir, exist_ok=True)

    g = load_channel(a.image, a.channel)
    gs, nuc, core, fil, faces = segment(g)
    df = measure_faces(faces)
    df.to_csv(os.path.join(outdir, "mesh_faces.csv"), index=False)
    save_qc(g, nuc, core, fil, faces, outdir)

    s = df.area.describe()
    summary = (f"faces measured: {len(df)}\n"
               f"area (a.u.): mean {s['mean']:.1f}, median {s['50%']:.1f}, "
               f"SD {s['std']:.1f}, min {s['min']:.0f}, max {s['max']:.0f}\n"
               f"total face area: {df.area.sum():.0f}; analysis region: {core.sum()*PIXEL_SIZE**2:.0f}; "
               f"filament fraction in region: {(fil & core).sum()/core.sum():.3f}\n")
    open(os.path.join(outdir, "summary.txt"), "w").write(summary)
    print(summary + f"outputs -> {outdir}")


if __name__ == "__main__":
    main()
