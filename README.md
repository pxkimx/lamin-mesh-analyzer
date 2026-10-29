# Lamin Mesh Analyzer

Measures the nuclear lamina meshwork in fluorescence images. It traces the lamin fibres, numbers every mesh
opening (a "face"), and measures its area, shape and connectivity. You get statistics, interactive charts,
an Excel table of every face and a PDF report. Images never leave your computer: the app runs as a local
server and opens in your browser.

Tool page: <https://paul-h-kim.com/software/lamin-mesh-analyzer>

Segmentation follows Shimi et al., *Mol Biol Cell* 2015 ([doi:10.1091/mbc.E15-07-0461](https://doi.org/10.1091/mbc.E15-07-0461)).
This is an independent Python re-implementation, not the authors' code.

## Run it
Needs Python 3.10 or newer.

```bash
git clone https://github.com/pxkimx/lamin-mesh-analyzer.git
cd lamin-mesh-analyzer
pip install -r requirements.txt
python serve.py            # opens http://localhost:8599
```

Click **Try the example image** to load a synthetic meshwork (`example/synthetic_mesh.tif`, made by
`example/make_synthetic_example.py`), or drop your own TIFF, PNG or JPG images in the sidebar.

**Mac app:** `./make_app.sh` builds `Lamin Mesh Analyzer.app`, a double-clickable app with its own icon. The
first time, it offers to install any missing Python packages. If macOS blocks it, right-click the app → **Open**.

**Tests:** `pip install pytest`, then `python -m pytest tests -q`.

## Light / dark theme
The **Light | Dark** switch sits at the top right of every page; Dark, the original look, is the default.
- The choice is saved in `~/Library/Application Support/Lamin Mesh Analyzer/settings.json`, so it is the same in a
  browser tab and inside Toolbox Launcher's frame.
- The first frame is already painted in the saved theme: `serve.py` writes it into the page before Streamlit starts.
- Switching while the app is open keeps everything you loaded or corrected. The page, Streamlit's widgets, tables and
  charts all follow, and the animated background stays in both themes.
- Exported Excel, CSV, PDF and training files are not affected.

How it works: `theme.py`, the `[theme.light]` / `[theme.dark]` sections of `.streamlit/config.toml`, and the CSS
variables in `ui.py`. `tests/test_theme.py` checks the saved choice, the first paint, that the config matches, and
text contrast of at least 4.5:1 in both themes.

## Closing
Closing the app's browser tab quits the app completely, about 6 seconds later: the local server stops,
nothing is left on port 8599, and the icon leaves the Dock. Reloading the page does not quit it. Quitting
(or force-quitting) the icon in the Dock stops the server as well. If the browser disappears without closing
the tab (crash, force-quit), the app stops after 3 minutes with no page connected. See `lifecycle.py` (shared
with Cell Surveillance) and `tests/test_lifecycle.py`; `serve.py` starts the server.

## How it measures
Choose the method in the sidebar.

**Meshwork tracing (default)** follows Shimi et al., *Mol Biol Cell* 2015 (doi:10.1091/mbc.E15-07-0461).
This is an independent Python re-implementation, not the authors' MATLAB code.
1. **Line filter:** an oriented filter (16 orientations) enhances line-shaped lamin fibres. The scale σ is
   chosen automatically: a patch of the nucleus is traced at 7 scales, and the smallest scale that closes
   ≥ 95% of the peak number of faces is kept.
2. **Centre lines:** the filter response is thinned to the 1-pixel centre of each fibre and kept above a
   threshold. *Line threshold* = Otsu × a factor you set; 1.0 = automatic. Edges shorter than 5 px are dropped.
3. **Gap closing:** each free fibre end is extended along its own direction until it meets another fibre,
   up to *Close gaps up to* px. Branches that still dead-end are trimmed away.
4. **Edge auditing:** edges fainter than their surroundings are removed (*Min. edge quality*,
   1 = local average).
5. **Faces:** every region enclosed by traced fibres is a face, so face area reaches the fibre centre
   lines. Each face also gets its number of edges and its mean edge length.

**Black & white** is the simpler alternative:
1. One intensity threshold turns the image black & white (automatic = Otsu).
2. Every black region inside the nucleus is a face, so face area stops at the fibre edge.

For both methods, faces that touch the rim band (*Rim exclusion*) or cross the edge of a region you
draw are not measured.

**Image type preset** (sidebar):
- *Clean* is for confocal, Airyscan or other well-resolved images.
- *Noisy / 3D-SIM* is for speckled super-resolution or immunofluorescence images. It uses a lower line
  threshold (0.6× automatic) and no edge auditing.
- *Noisy / 3D-SIM* reproduced the authors' published tracings best (median face area 1.07× theirs over
  10 panels; see `test_images/README.md`).

**Corrections & training set** (Numbered faces panel):
- **Erase fibre** removes a false edge, and the two faces merge.
- **Draw fibre** adds a missing edge by clicking its start and end; the ends snap to existing lines.
- **Exclude face** leaves an artefact face out of the measurements.
- **Undo** and **Reset** work on all three.
- **Save to training set** writes the image plus its corrected segmentation to
  `~/Documents/Lamin Mesh training set/`. **Download (.zip)** gives the same files:
  - `image.tif`, `meshwork.tif`, `faces.tif`, `nucleus.tif`
  - `faces.csv`
  - `info.json`, with the settings and corrections used
- With 10–20 corrected images, the settings can be tuned (or a model trained) on your own data.

## Workflow
1. **Upload** TIFF, PNG or JPG files in the sidebar. You can upload several at once, or click
   **Try the example image**.
2. **Check segmentation** (tab 🔍): *Original*, *Traced meshwork* (magenta) or *Black & white*, and
   *Filled faces* are shown side by side.
   - Move the **Line threshold** or **Threshold** slider to see how the segmentation changes.
   - **↺ Auto** returns to the automatic value. **Apply to all images** copies it to every image.
   - To zoom, drag a box on any panel. All three panels zoom together and stay zoomed while you move
     the slider.
   - To measure only part of the nucleus, pick *Draw region* and draw on the image.
3. **Numbered faces** (further down the same tab): every face gets its own color and number, and
   neighbouring faces always have different colors.
   - Click a face to select it (*Click to select* mode), or click its row in the table. The face gets
     a white ring, its row is highlighted, and its area, diameter and circularity are shown.
   - **Zoom to this face** jumps to the selected face.
   - **Face table (Excel)** downloads the numbered faces with their areas.
4. **Scale:** turn on *Calibrated units* and enter the pixel size. If the TIFF stores ImageJ or OME
   scale metadata, the app reads it automatically.
5. **Results** (tab 📊):
   - a summary for each image and pooled statistics
   - distributions and an area map
   - the per-face table
   - **All images (Excel)**: a Summary sheet plus one sheet of numbered faces per image; CSV is also available
6. **PDF report** (tab 📄): for each image, the three panels, a numbered-faces page, the area map,
   the stats, the distributions, every setting used, and a methods paragraph.

## Files
- `app.py`: the user interface (Streamlit)
- `test_images/`: a script that downloads published figures (PubMed Central Open Access) and compares this
  tool with the authors' own tracings. The figures are not stored here.
- `meshwork.py`: the meshwork-tracing algorithm (line filter, centre lines, gap closing, pruning, auditing)
- `ui.py`: the look & feel (theme, stat cards, interactive charts)
- `theme.py`: the Light | Dark theme (saved choice, first paint, live switch, chart colours)
- `serve.py` / `lifecycle.py`: start the server; quit completely when the window is closed
- `lamin_core.py`: loading, segmentation, measurement, plots and the PDF report. It can also be
  used from Python scripts or notebooks.
- `lamin_mesh.py`: the original command-line prototype
- `make_app.sh`: builds `Lamin Mesh Analyzer.app` (with the icon in `assets/`)
- `.streamlit/config.toml`: the light and dark Streamlit themes
- `example/`: the synthetic example and the script that makes it
- `tests/`: quitting, theme and measurement tests

## Licence
MIT; see `LICENSE`. The figures used by `test_images/validate_against_paper.py` belong to their publishers and
are downloaded under their own licences (CC BY 4.0 and CC BY-NC-SA 3.0).
