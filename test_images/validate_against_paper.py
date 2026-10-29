"""
Compare our meshwork tracing with the authors' published tracings (Shimi et al. 2015 MBoC Fig. 2 & 4;
Kittisopikul et al. 2019 Cells Fig. 5). Each raw zoom panel sits next to a panel with the authors'
detected meshwork drawn in magenta (or cyan). Faces are compared on the same crop.
Caveat: panels are small JPEG figure crops (~100-130 px), so this is a sanity check, not a benchmark.
"""
import sys, json
from pathlib import Path
import numpy as np
from PIL import Image
from skimage import measure, morphology, segmentation
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import lamin_core as L

SRC = HERE / "source_figures"
# The figures are not stored in the repository (third-party, CC BY 4.0 / CC BY-NC-SA 3.0). They are fetched once
# from the official PubMed Central Open Access bucket.
FIGURES = {"PMC4710238_4075fig2.jpg": "PMC4710238.1/4075fig2.jpg",
           "PMC4710238_4075fig4.jpg": "PMC4710238.1/4075fig4.jpg",
           "PMC6524165_cells-08-00361-g005.jpg": "PMC6524165.1/cells-08-00361-g005.jpg"}


def fetch_figures():
    import urllib.request
    SRC.mkdir(exist_ok=True)
    for name, key in FIGURES.items():
        if not (SRC / name).exists():
            print(f"downloading {name} from the PMC Open Access bucket")
            urllib.request.urlretrieve(f"https://pmc-oa-opendata.s3.amazonaws.com/{key}", SRC / name)
# (name, figure, raw box, traced box, raw channel, line colour) ; box = (y0, x0, y1, x1)
PAIRS = [
    ("shimi_fig2_LA",  "PMC4710238_4075fig2.jpg", (3, 134, 127, 257), (3, 267, 127, 390), 1, "magenta"),
    ("shimi_fig2_LB1", "PMC4710238_4075fig2.jpg", (3, 543, 126, 666), (3, 674, 126, 797), 1, "magenta"),
    ("shimi_fig2_LC",  "PMC4710238_4075fig2.jpg", (135, 134, 258, 257), (135, 266, 258, 389), 1, "magenta"),
    ("shimi_fig2_LB2", "PMC4710238_4075fig2.jpg", (134, 543, 257, 666), (134, 674, 257, 797), 1, "magenta"),
    ("shimi_fig4_wt",      "PMC4710238_4075fig4.jpg", (3, 146, 97, 240), (3, 247, 97, 341), 1, "magenta"),
    ("shimi_fig4_Lmnb1ko", "PMC4710238_4075fig4.jpg", (3, 512, 97, 606), (3, 615, 97, 709), 1, "magenta"),
    ("shimi_fig4_Lmnako",  "PMC4710238_4075fig4.jpg", (104, 146, 198, 240), (104, 247, 198, 341), 1, "magenta"),
    ("shimi_fig4_Lmnb2ko", "PMC4710238_4075fig4.jpg", (104, 512, 198, 606), (104, 615, 198, 709), 1, "magenta"),
    ("kitti_fig5_LA",  "PMC6524165_cells-08-00361-g005.jpg", (337, 6, 461, 130), (337, 137, 461, 261), 1, "magenta"),
    ("kitti_fig5_LB1", "PMC6524165_cells-08-00361-g005.jpg", (337, 268, 461, 392), (337, 400, 461, 524), 0, "cyan"),
]


def their_faces(rgb, colour):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    if colour == "magenta":      # magenta over a green image: the green channel has R and B near zero
        line = (r > 85) & (b > 75)
    else:                        # cyan over a red image: the red channel has G and B near zero
        line = (g > 85) & (b > 75)
    line = morphology.skeletonize(morphology.binary_closing(line, morphology.disk(1)))
    faces = measure.label(~morphology.binary_dilation(line), connectivity=1)
    faces = segmentation.clear_border(faces)
    h, w = faces.shape
    keep = L.roi_mask(faces.shape, roi_without_label(h, w))
    keep = __import__("skimage.morphology", fromlist=["x"]).binary_erosion(keep, morphology.disk(2))
    faces[np.isin(faces, np.unique(faces[~keep]))] = 0
    faces = morphology.remove_small_objects(faces, 4)
    return line, faces


LABEL_BOX = 24          # the panel letter sits in the bottom-left corner: leave it out of both analyses


def roi_without_label(h, w):
    return [(0, 0), (w - 1, 0), (w - 1, h - 1), (LABEL_BOX, h - 1), (LABEL_BOX, h - 1 - LABEL_BOX), (0, h - 1 - LABEL_BOX)]


def our_faces(gray, **kw):
    b = __import__("io").BytesIO(); Image.fromarray(gray).save(b, format="PNG")
    img = L.load_image(b.getvalue(), "crop.png")
    h, w = gray.shape
    r = L.analyze(img, L.Params(channel=0, whole_image=True, edge_margin=2, min_area_px=4, **kw),
                  roi=roi_without_label(h, w))
    return img, r


def stats(areas):
    return {"n": int(len(areas)), "median": float(np.median(areas)) if len(areas) else None,
            "mean": float(np.mean(areas)) if len(areas) else None}


fetch_figures()
(HERE / "crops").mkdir(exist_ok=True)
(HERE / "validation").mkdir(exist_ok=True)
OURS = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
rows, tiles = [], []
for name, fig, rb, tb, ch, colour in PAIRS:
    im = np.asarray(Image.open(SRC / fig).convert("RGB"))
    t = 3
    raw = im[rb[0] + t:rb[2] - t, rb[1] + t:rb[3] - t, ch]
    tr = im[tb[0] + t:tb[2] - t, tb[1] + t:tb[3] - t]
    h = min(raw.shape[0], tr.shape[0]); w = min(raw.shape[1], tr.shape[1])
    raw, tr = raw[:h, :w], tr[:h, :w]
    Image.fromarray(raw).save(HERE / "crops" / f"{name}.png")
    line, fth = their_faces(tr, colour)
    img, r = our_faces(raw, **OURS)
    a_their = np.bincount(fth.ravel())[1:]; a_their = a_their[a_their > 0]
    a_our = r.table.area.values
    rows.append({"panel": name, "paper": stats(a_their), "ours": stats(a_our),
                 "sigma": r.extra.get("sigma"), "fibre_width": r.extra.get("fiber_width")})
    ours_img = L.structure_view(img, r, lw=0)
    up = lambda a: np.asarray(Image.fromarray(a).resize((a.shape[1] * 3, a.shape[0] * 3), Image.NEAREST))
    gap = np.full((h * 3, 6, 3), 255, np.uint8)
    tiles.append(np.concatenate([up(np.dstack([raw] * 3)), gap, up(tr), gap, up(ours_img)], 1))

for row in rows:
    p, o = row["paper"], row["ours"]
    print(f"{row['panel']:20s} paper n={p['n']:3d} med={p['median'] or 0:6.1f} | ours n={o['n']:3d} "
          f"med={o['median'] or 0:6.1f} | ratio(med)={((o['median'] or 0) / (p['median'] or 1)):.2f} "
          f"| sigma={row['sigma']:.2f} width={row['fibre_width']:.1f}")
json.dump(rows, open(HERE / "validation" / "results.json", "w"), indent=2)
W = max(t.shape[1] for t in tiles)
pad = lambda t: np.pad(t, ((0, 0), (0, W - t.shape[1]), (0, 0)), constant_values=255)
sep = np.full((8, W, 3), 255, np.uint8)
Image.fromarray(np.concatenate(sum([[pad(t), sep] for t in tiles], [])[:-1], 0)).save(
    HERE / "validation" / "side_by_side.png")
med_ratio = [r["ours"]["median"] / r["paper"]["median"] for r in rows if r["ours"]["median"] and r["paper"]["median"]]
n_ratio = [r["ours"]["n"] / r["paper"]["n"] for r in rows if r["paper"]["n"]]
print(f"SUMMARY settings={OURS}  median-area ratio (ours/paper): median {np.median(med_ratio):.2f} "
      f"[{np.min(med_ratio):.2f}-{np.max(med_ratio):.2f}]   face-count ratio: median {np.median(n_ratio):.2f}")
