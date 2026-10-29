# Test set: published lamin meshwork figures

These figures are used to check the meshwork tracing against the tracings the original authors
published. They are **not** used for training. They are small, JPEG-compressed figure panels
(about 100–130 px each), so they work only as a sanity check, not a precise benchmark.

| File | Source | License |
|---|---|---|
| `source_figures/PMC4710238_4075fig1/2/4.jpg` | Shimi T, Kittisopikul M, et al. *Mol Biol Cell* 2015;26:4075. doi:10.1091/mbc.E15-07-0461 | CC BY-NC-SA 3.0 |
| `source_figures/PMC6524165_cells-08-00361-g001/2/5/6.jpg` | Kittisopikul M, et al. *Cells* 2019;8:361. doi:10.3390/cells8040361 | CC BY 4.0 |

The figures are **not stored in this repository**. `validate_against_paper.py` downloads the three it uses
into `source_figures/` on first run, from the official PubMed Central Open Access bucket
(`pmc-oa-opendata.s3.amazonaws.com`).

## Method
`validate_against_paper.py` pairs each raw zoom panel with the panel next to it, where the authors drew
their detected meshwork in magenta or cyan. For each pair it:
1. extracts the authors' lines and computes their faces;
2. traces the raw panel with this tool (whole image as nucleus, panel letter and faces touching the
   border excluded);
3. compares the median face area and the face count.

```bash
python3 validate_against_paper.py '{"sensitivity": 0.6, "edge_quality": 0.0}'   # "Noisy / 3D-SIM" preset
```

## Results (10 panels, automatic line scale)
| Settings | Median face area, ours / paper | Face count, ours / paper |
|---|---|---|
| Noisy / 3D-SIM preset | **1.07** (range 0.81–1.62) | 1.5 |
| Clean preset (defaults) | 1.47 (range 0.53–5.26) | 0.16 (most panels fail to close) |

- The earlier width-based automatic scale failed on these speckled 3D-SIM panels. It was replaced by a
  face-count scan: trace at 7 scales and keep the scale where the meshwork closes best. On the
  confocal example image this picks the same σ (2.0), so its results are unchanged.
- The face count is higher than the paper's, partly because the authors' anti-aliased JPEG lines merge
  where fibres are close together.

`validation/side_by_side.png` shows each pair: raw panel | authors' tracing | this tool.
