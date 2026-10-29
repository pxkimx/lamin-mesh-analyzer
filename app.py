"""
Lamin Mesh Analyzer - Streamlit app.

    streamlit run app.py

Lamin channel -> traced meshwork (Shimi et al. 2015) or black & white -> enclosed "faces" ->
areas, statistics, distributions, tables -> Excel / PDF report.
"""
import base64
import html
import io
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from PIL import Image

import lamin_core as L
import lifecycle
import theme
import ui
from streamlit import components

__version__ = "1.1.0"          # 1.1.0: Light | Dark theme. (1.0: tracing, corrections, quit-when-closed.)
HERE = Path(__file__).parent
EXAMPLE_DIR = HERE / "example"
TRAIN_DIR = Path.home() / "Documents" / "Lamin Mesh training set"
ICON = next((p for p in (HERE / "assets" / "icon1024.png", HERE / "assets" / "icon_public.png") if p.exists()),
            HERE / "assets" / "icon1024.png")

st.set_page_config(page_title="Lamin Mesh Analyzer", page_icon=str(ICON) if ICON.exists() else "🔬",
                   layout="wide", initial_sidebar_state="expanded")
ss = st.session_state
ss.setdefault("per_image", {})       # name -> {"channel", "threshold", "roi", "zoom"}
ss.setdefault("plot_nonce", 0)

# Settings that only one method shows (Image type, gap closing, line scale, edge quality / smoothing, specks).
# Streamlit forgets a widget's value while the widget is hidden, so switching method and back would reset them.
# Re-saving the keys at the start of every run keeps the user's choices (the documented Streamlit pattern).
STICKY = ("preset", "auto_gap", "max_gap", "auto_w", "line_sigma", "smooth", "puncta")
for _k in list(ss.keys()):
    if _k in STICKY or str(_k).startswith("quality_"):
        ss[_k] = ss[_k]

# ============================================================================ style
ui.inject()


# Light | Dark switch, pinned in the top bar on every page (CSS .st-key-theme_choice in ui.py). The choice is
# saved in the settings file (theme.py), so it is the same in a tab and inside Toolbox Launcher's frame.
def _theme_changed():
    if ss.get("theme_choice") in theme.STREAMLIT:
        theme.save(ss["theme_choice"])
    else:                                       # clicking the active option deselects it: keep the saved theme
        ss["theme_choice"] = theme.load()


ss.setdefault("theme_choice", theme.load())
st.segmented_control("Theme", ["light", "dark"], key="theme_choice", on_change=_theme_changed,
                     format_func=lambda t: ":material/light_mode: Light" if t == "light" else ":material/dark_mode: Dark",
                     label_visibility="collapsed")
T = ss["theme_choice"] if ss.get("theme_choice") in theme.STREAMLIT else theme.load()
ui.use_theme(T)
TC = theme.CHART[T]

# Tell the server when this window is closed, so that the app quits completely instead of running on unseen
# (lifecycle.py). A reload reconnects within a few seconds and does not quit it. The same zero-height iframe
# applies the chosen theme to the open page (theme.apply_script). Both scripts act on the page (window.parent).
components.v1.html("""<script>
(() => {
  const url = "%s";
  try {
    const top = window.parent;
    if (url && !top.__laminMeshClose) {
      top.__laminMeshClose = true;
      top.addEventListener('pagehide', e => { if (!e.persisted) top.navigator.sendBeacon(url); });
    }
  } catch (e) {}
})();
%s
</script>""" % (lifecycle.start(), theme.apply_script(T)), height=0)


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode() if path.exists() else ""


cards, section = ui.cards, ui.section
ICON64 = b64(ICON)


# ============================================================================ cached work
@st.cache_data(show_spinner=False, max_entries=64)
def load(data: bytes, name: str) -> L.LoadedImage:
    return L.load_image(data, name)


@st.cache_data(show_spinner=False, max_entries=64)
def prepare(data: bytes, name: str, channel: int, method: str, smooth: float, line_sigma: float,
            nuc_scale: float, all_nuc: bool, whole: bool = False):
    img = load(data, name)
    p = L.Params(channel=channel, method=method, smooth_sigma=smooth, line_sigma=line_sigma,
                 nucleus_scale=nuc_scale, all_nuclei=all_nuc, whole_image=whole)
    return L.prepare(img.channels[..., channel], p)


@st.cache_data(show_spinner=False, max_entries=128)
def segment(data: bytes, name: str, p: tuple):
    img, params = load(data, name), L.Params(*p)
    prep = prepare(data, name, params.channel, params.method, params.smooth_sigma, params.line_sigma,
                   params.nucleus_scale, params.all_nuclei, params.whole_image)
    return L.segment(img.channels[..., params.channel], params, prep)


@st.cache_data(show_spinner=False, max_entries=256)
def run(data: bytes, name: str, p: tuple, px: float, unit: str, roi: tuple | None, edits: tuple = ()) -> L.Result:
    img = load(data, name)
    return L.analyze(img, L.Params(*p), px, unit, list(roi) if roi else None, seg=segment(data, name, p),
                     edits=list(edits))


@st.cache_data(show_spinner=False, max_entries=256)
def panels(data: bytes, name: str, p: tuple, px: float, unit: str, roi: tuple | None, single: bool,
           lw: int | None = None, edits: tuple = ()):
    img, r = load(data, name), run(data, name, p, px, unit, roi, edits)
    return {"Original": img.display, "Structure": L.structure_view(img, r, lw),
            "Filled faces": L.filled_faces(img, r, single_color=(34, 211, 238) if single else None, lw=lw)}


def line_width(o, w):
    """Draw traced lines thicker when zoomed in so they stay visible on screen."""
    z = o.get("zoom")
    return 1 if z and (z[1] - z[0]) < 0.5 * w else None


def to_uri(arr: np.ndarray) -> str:
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


# ============================================================================ sidebar
st.sidebar.markdown(
    '<div class="brand">' + (f'<img src="data:image/png;base64,{ICON64}">' if ICON64 else "") +
    f'<div><div class="t">Lamin Mesh<br>Analyzer</div><div class="s">nuclear lamina meshwork · v{__version__}</div></div></div>',
    unsafe_allow_html=True)
files = st.sidebar.file_uploader("Lamin images", type=["tif", "tiff", "png", "jpg", "jpeg"],
                                 accept_multiple_files=True, help="Single-channel TIFF works best. "
                                 "Multi-channel, 16-bit and Z-stacks are handled too.")
# The example: your own lamin_example.tif if it is there (kept out of the public repository),
# otherwise the synthetic meshwork that ships with the code (example/make_synthetic_example.py).
examples = [p for p in (EXAMPLE_DIR / "lamin_example.tif", EXAMPLE_DIR / "synthetic_mesh.tif") if p.exists()][:1]
if not files and examples and st.sidebar.button("✨ Try the example image", width="stretch"):
    ss.use_example = True
if files:
    ss.use_example = False

if not files and not ss.get("use_example"):
    art = f'<img class="art" src="data:image/png;base64,{ICON64}">' if ICON64 else ""
    st.markdown('<div class="hero" style="padding:2.2rem 2.2rem 2rem">'
                '<span class="chip">TIFF · PNG · JPG</span><span class="chip v">Meshwork tracing</span>'
                '<span class="chip p">Excel &amp; PDF</span>'
                '<h1>Lamin Mesh Analyzer</h1>'
                '<p>Trace the nuclear lamina meshwork, number every face, and measure its area, shape and '
                'connectivity - in seconds, with a report you can hand to your PI.</p>'
                '<div class="cite">Segmentation after Shimi et al., <i>Mol Biol Cell</i> 2015 · '
                '<a href="https://doi.org/10.1091/mbc.E15-07-0461" target="_blank">doi:10.1091/mbc.E15-07-0461</a></div>'
                f'{art}</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    steps = [("images", "Upload", "Drop lamin images in the sidebar. Single-channel TIFF is ideal."),
             ("edges", "Trace", "Fibres are traced as centre lines, gaps closed; every enclosed region is a face."),
             ("select", "Explore", "Original, traced meshwork and filled faces side by side. Click any face to identify it."),
             ("doc", "Report", "Distributions and numbered face tables, exported to Excel and a PDF report.")]
    for i, (c, (ic, h, d)) in enumerate(zip(cols, steps), start=1):
        c.markdown(f'<div class="feature" style="animation-delay:{i * .07:.2f}s"><div class="n">{ui.icon(ic, 19)}</div>'
                   f'<div class="h">{h}</div><div class="d">{d}</div></div>', unsafe_allow_html=True)
    show = next((p for p in (HERE / "assets" / "showcase.jpg", HERE / "assets" / "showcase_synthetic.jpg")
                 if p.exists()), HERE / "assets" / "showcase.jpg")
    if show.exists():
        st.markdown('<div style="height:1.4rem"></div><div class="panel-title"><span>Original</span>'
                    '<span>Traced meshwork</span><span>Filled faces</span></div>', unsafe_allow_html=True)
        st.image(str(show), width="stretch")
    st.stop()

data = ({f.name: f.getvalue() for f in files} if files
        else {p.name: p.read_bytes() for p in examples})
with st.spinner("Loading images…"):
    images = {n: load(b, n) for n, b in data.items()}
names = list(images)

st.sidebar.subheader("Method")
METHODS = {"Meshwork tracing": "trace", "Black & white": "threshold"}
method = METHODS[st.sidebar.segmented_control(
    "Segmentation method", list(METHODS), default="Meshwork tracing", key="method_choice",
    label_visibility="collapsed") or "Meshwork tracing"]
st.sidebar.caption("Traces fibre centre lines with a line filter, closes gaps and measures the enclosed "
                   "faces - after Shimi et al., Mol Biol Cell 2015." if method == "trace" else
                   "Converts the image to black & white with one threshold; each black region is a face.")
TRACE = method == "trace"
# presets: (line threshold x auto, min. edge quality). "Noisy / 3D-SIM" reproduced the published tracings of
# Shimi et al. 2015 / Kittisopikul et al. 2019 best (median face area within ~10%; see test_images/).
PRESETS = {"Clean": (1.0, 1.0), "Noisy / 3D-SIM": (0.6, 0.0)}


def _preset_changed():
    for n_, o_ in ss.per_image.items():
        o_.pop("sensitivity", None)
        ss.pop(f"sensitivity_{n_}", None)


preset = "Clean"
if TRACE:
    ss.setdefault("preset", "Clean")
    preset = st.sidebar.segmented_control(
        "Image type", list(PRESETS), key="preset", on_change=_preset_changed,
        help="Clean: confocal / Airyscan / well-resolved meshwork. Noisy / 3D-SIM: speckled super-resolution "
             "or immunofluorescence images - lower line threshold, no edge auditing (matched the published "
             "tracings of Shimi et al. 2015 best).") or "Clean"
PRESET_SENS, PRESET_Q = PRESETS[preset]

st.sidebar.subheader("Scale")
meta = next(((im.pixel_size, im.unit) for im in images.values() if im.pixel_size), None)
calibrated = st.sidebar.toggle("Calibrated units", value=meta is not None,
                               help="Off: results in pixels. On: enter the size of one pixel.")
if calibrated:
    c1, c2 = st.sidebar.columns([3, 2])
    px = c1.number_input("Pixel size", min_value=1e-6, value=float(meta[0]) if meta else 0.1, format="%.4f")
    unit = c2.text_input("Unit", value=(meta[1] if meta else "µm")) or "unit"
    if meta:
        st.sidebar.caption(f"Read from file: {meta[0]:.4g} {meta[1]}/px")
else:
    px, unit = 1.0, "px"
u2 = f"{unit}²"

st.sidebar.subheader("Region")
edge_margin = st.sidebar.slider("Rim exclusion (px)", 0, 150, 25, 5,
                                help="Faces within this distance of the nuclear edge are not measured.")
min_area = st.sidebar.number_input("Minimum face area (px²)", 1, 5000, 20, 5,
                                   help="Faces smaller than this are ignored (noise).")
if TRACE:
    ss.setdefault("auto_gap", True)
    ss.setdefault("max_gap", 10)
    auto_gap = st.sidebar.toggle("Automatic gap closing", key="auto_gap",
                                 help="Close gaps up to 5 × the detected fibre width scale (σ).")
    max_gap = -1 if auto_gap else st.sidebar.slider(
        "Close gaps up to (px)", 0, 30, step=1, key="max_gap",
        help="Free fibre ends are extended along their direction until they meet another fibre, up to this "
             "distance (the paper used up to 20 px on 3D-SIM).")

with st.sidebar.expander("Advanced"):
    smooth, puncta, line_sigma, quality = 1.0, 30, -1.0, 1.0
    if TRACE:
        ss.setdefault("auto_w", True)
        ss.setdefault("line_sigma", 2.0)
        ss.setdefault(f"quality_{preset}", PRESET_Q)
        auto_w = st.toggle("Automatic line scale", key="auto_w",
                           help="Traces a patch of the nucleus at 7 filter scales and keeps the scale where the "
                                "most faces close up - so images from different microscopes or zooms need no "
                                "tuning. Validated against the published tracings of Shimi et al. 2015.")
        if not auto_w:
            line_sigma = st.slider("Fibre width σ (px)", 0.75, 6.0, step=0.25, key="line_sigma",
                                   help="Scale of the line filter ≈ fibre width ÷ 3.5. Smaller finds finer fibres; "
                                        "larger ignores fine texture.")
        quality = st.slider("Min. edge quality", 0.0, 2.0, step=0.05, key=f"quality_{preset}",
                            help="Edges must be at least this bright relative to their surroundings "
                                 "(1 = local average). Higher removes weak edges (faces merge).")
    else:
        ss.setdefault("smooth", 1.0)
        ss.setdefault("puncta", 30)
        smooth = st.slider("Smoothing σ (px)", 0.0, 3.0, step=0.25, key="smooth",
                           help="Light blur before thresholding to reduce pixel noise. 0 = none.")
        puncta = st.number_input("Ignore white specks smaller than (px²)", 0, 1000, step=5, key="puncta",
                                 help="Tiny isolated bright dots inside a face are counted as part of the face.")
    nuc_scale = st.slider("Nucleus detection (× Otsu)", 0.2, 1.5, 0.6, 0.05,
                          help="Lower if part of the nucleus is missed; higher if background is included.")
    all_nuclei = st.checkbox("Analyze all nuclei in the image", False)
    whole_image = st.checkbox("Image is a crop inside the nucleus", False,
                              help="Skip nucleus detection and analyze the whole image (set Rim exclusion to a "
                                   "few px so faces cut by the image border are not measured).")


def store(name):
    return ss.per_image.setdefault(name, {})


def channel_of(name):
    return store(name).get("channel", images[name].default_channel)


def auto_threshold(name):
    return prepare(data[name], name, channel_of(name), method, smooth, line_sigma, nuc_scale, all_nuclei,
                   whole_image)["auto"]


# the adjustable level per image: "sensitivity" (trace, 1.0 = auto) or "threshold" (b&w, -1 = auto)
LEVEL = "sensitivity" if TRACE else "threshold"
LEVEL_AUTO = 1.0 if TRACE else -1.0


def params_for(name: str) -> tuple:
    o = store(name)
    p = L.Params(channel=channel_of(name), method=method, smooth_sigma=smooth,
                 threshold=o.get("threshold", -1.0), line_sigma=line_sigma,
                 sensitivity=o.get("sensitivity", PRESET_SENS), max_gap=int(max_gap) if TRACE else -1,
                 edge_quality=quality, nucleus_scale=nuc_scale, all_nuclei=all_nuclei, whole_image=whole_image,
                 edge_margin=edge_margin, min_area_px=int(min_area), max_puncta_px=int(puncta))
    return tuple(asdict(p).values())


def roi_for(name):
    r = store(name).get("roi")
    return tuple(map(tuple, r)) if r else None


def edits_for(name):
    return tuple(tuple(e) for e in store(name).get("edits", []))


def result(name) -> L.Result:
    return run(data[name], name, params_for(name), px, unit, roi_for(name), edits_for(name))


# threshold widget callbacks (values live in ss.per_image so they survive switching images)
def _key(name):
    return f"{LEVEL}_{name}"


def _sync(name):
    store(name)[LEVEL] = float(ss[_key(name)])


def _auto(name):
    store(name).pop(LEVEL, None)
    ss[_key(name)] = PRESET_SENS if TRACE else int(round(auto_threshold(name)))


def _apply_all(name):
    v = store(name).get(LEVEL)
    for n in names:
        if v is None:
            store(n).pop(LEVEL, None)
        else:
            store(n)[LEVEL] = v
        if n != name:
            ss.pop(_key(n), None)
    ss.applied_all = True


# ============================================================================ main
st.markdown('<div class="hero compact">'
            + (f'<img class="art" src="data:image/png;base64,{ICON64}">' if ICON64 else "")
            + f'<span class="chip">{len(names)} image{"s" if len(names) > 1 else ""}</span>'
            + f'<span class="chip v">{"Meshwork tracing" if TRACE else "Black &amp; white"}</span>'
            + f'<span class="chip p">{"calibrated · " + html.escape(unit) if calibrated else "pixel units"}</span>'
            + '<h1>Lamin Mesh Analyzer</h1><p>'
            + ('lamin channel → line filter → traced fibre centre lines → enclosed faces measured'
               if TRACE else 'lamin channel → black &amp; white → black regions measured as faces')
            + '</p></div>', unsafe_allow_html=True)
tab_img, tab_res, tab_rep = st.tabs(["🔍  Check segmentation", "📊  Results", "📄  PDF report"])

# ---------------------------------------------------------------------------- tab 1
with tab_img:
    head = st.columns([3, 4, 2])
    if len(names) > 1:
        name = head[0].selectbox("Image", names, format_func=lambda n: f"{names.index(n) + 1}. {n}")
    else:
        name = names[0]
        head[0].markdown(f"<div style='padding-top:1.9rem;color:var(--ink2)'>🖼  {html.escape(name)}</div>",
                         unsafe_allow_html=True)
    im, o = images[name], store(name)
    tool = head[1].radio("Mouse", ["Zoom to box", "Draw region (lasso)", "Draw region (box)"], horizontal=True,
                         help="Zoom to box: drag a box - all three panels zoom there and stay zoomed while you "
                              "move the threshold. Draw region: measure only inside the area you draw.")
    head[2].markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
    single = head[2].toggle("Single fill color", False, help="Off: each face gets its own color so you can "
                                                            "see where neighbouring faces are separated.")

    auto = auto_threshold(name)
    ctl = st.container(border=True)
    c = ctl.columns([6, 1.2, 1.8], vertical_alignment="bottom")
    if TRACE:
        ss.setdefault(_key(name), float(o.get("sensitivity", PRESET_SENS)))
        c[0].slider(f"Line threshold  (× automatic;  preset default = {PRESET_SENS:g})", 0.3, 3.0, step=0.05,
                    key=_key(name),
                    on_change=_sync, args=(name,),
                    help="Threshold on the line-filter response. Lower → fainter fibres are traced too "
                         "(more, smaller faces). Higher → only strong fibres (fewer, larger faces).")
    else:
        ss.setdefault(_key(name), int(round(o.get("threshold", auto))))
        c[0].slider(f"Threshold  ·  automatic = {auto:.0f}", 0, 255, key=_key(name), on_change=_sync,
                    args=(name,), help="Pixels brighter than the threshold become white (filament); darker "
                                       "become black (faces). Lower → more filament, smaller faces. "
                                       "Higher → less filament, faces merge and grow.")
    c[1].button("↺ Auto", on_click=_auto, args=(name,), width="stretch",
                help="Back to the automatic (Otsu) threshold")
    c[2].button("Apply to all images", on_click=_apply_all, args=(name,), width="stretch",
                disabled=len(names) < 2)
    if ss.pop("applied_all", False):
        st.toast("Threshold applied to all images", icon="✅")

    with st.spinner("Tracing meshwork…" if TRACE else "Thresholding…"):
        r = result(name)
        pics = panels(data[name], name, params_for(name), px, unit, roi_for(name), single,
                      line_width(o, im.display.shape[1]), edits_for(name))

    s = r.stats
    common = [("Faces", f"{s['Faces (n)']:,}", "enclosed faces measured" if TRACE else "black regions measured", "faces"),
              ("Median area", L.fmt(s[f"Median area ({u2})"]), u2, "median"),
              ("Mean area", L.fmt(s[f"Mean area ({u2})"]), f"{u2} · SD {L.fmt(s[f'SD ({u2})'])}", "mean")]
    if TRACE:
        cards(common + [("Edges / face", L.fmt(s.get("Median edges per face", np.nan)), "median", "edges"),
                        ("Edge length", L.fmt(s.get(f"Median edge length per face ({unit})", np.nan)),
                         f"{unit} · median per face", "length"),
                        ("Line scale σ", f"{r.extra.get('sigma', 0):.2f} px",
                         ("auto · face-count peak" if r.params.line_sigma <= 0 else "manual")
                         + f" · gaps ≤ {r.extra.get('gap', 0)} px", "threshold")])
    else:
        cards(common + [("Face coverage", f"{s['Face coverage (%)']:.1f}%", "of analyzed region", "coverage"),
                        ("Threshold", f"{r.extra['threshold']:.0f}",
                         "automatic (Otsu)" if r.params.threshold < 0 else f"manual · auto = {auto:.0f}", "threshold")])

    # three synced panels
    h, w = im.display.shape[:2]
    keys = ["Original", "Structure", "Filled faces"]
    titles = ["Original", "Traced meshwork" if TRACE else "Black & white", "Filled faces"]
    st.markdown('<div class="panel-title">' + "".join(f"<span>{html.escape(k)}</span>" for k in titles) + "</div>",
                unsafe_allow_html=True)
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.006)
    step = max(8, int(max(h, w) / 90))
    gy, gx = np.mgrid[step // 2:h:step, step // 2:w:step]
    for i, key in enumerate(keys, start=1):
        fig.add_trace(go.Image(source=to_uri(pics[key]), hoverinfo="skip"), 1, i)
        # invisible selectable grid so box/lasso selections register anywhere
        fig.add_trace(go.Scattergl(x=gx.ravel(), y=gy.ravel(), mode="markers", hoverinfo="skip",
                                   marker=dict(size=step, opacity=0), showlegend=False), 1, i)
        if i > 1 and len(r.table):
            t = r.table
            fig.add_trace(go.Scattergl(
                x=t.x / px, y=t.y / px, mode="markers", showlegend=False,
                marker=dict(size=8, color="rgba(0,0,0,0)"),
                customdata=np.c_[t.face, t.area, t.equiv_diameter, t.circularity],
                hovertemplate=f"<b>Face %{{customdata[0]:.0f}}</b><br>Area %{{customdata[1]:,.1f}} {u2}"
                              f"<br>Eq. diameter %{{customdata[2]:.2f}} {unit}"
                              "<br>Circularity %{customdata[3]:.2f}<extra></extra>"), 1, i)
        if o.get("roi"):
            xy = np.array(o["roi"] + [o["roi"][0]])
            fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="lines", hoverinfo="skip",
                                     line=dict(color="rgb(0,200,255)", width=2), showlegend=False), 1, i)
    zx0, zx1, zy0, zy1 = o.get("zoom", (0, w, 0, h))
    fig.update_xaxes(visible=False, range=[zx0, zx1], matches="x", constrain="domain")
    fig.update_yaxes(visible=False, range=[zy1, zy0], matches="y", constrain="domain")
    fig.update_layout(yaxis=dict(scaleanchor="x"), height=int(370 * h / w), margin=dict(l=0, r=0, t=0, b=0),
                      dragmode="lasso" if tool == "Draw region (lasso)" else "select",
                      plot_bgcolor="black", paper_bgcolor="black", hovermode="closest",
                      hoverlabel=dict(bgcolor=TC["hover_bg"], bordercolor=TC["hover_border"], font=dict(color=TC["ink"])))
    ev = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode=("box", "lasso"),
                         key=f"plot_{name}_{ss.plot_nonce}", config={"displayModeBar": False, "scrollZoom": False})

    sel = getattr(ev, "selection", None) or (ev.get("selection") if isinstance(ev, dict) else None)
    poly = None
    if sel:
        if sel.get("lasso"):
            ls = sel["lasso"][-1]
            poly = list(zip(ls["x"], ls["y"]))
        elif sel.get("box"):
            bx = sel["box"][-1]
            (x0, x1), (y0, y1) = sorted(bx["x"]), sorted(bx["y"])
            poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    if poly and tool == "Zoom to box":
        # the chart is rebuilt on every update, so the app keeps the zoom itself
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        if max(xs) - min(xs) > 5 and max(ys) - min(ys) > 5:
            o["zoom"] = (max(0, min(xs)), min(w, max(xs)), max(0, min(ys)), min(h, max(ys)))
        ss.plot_nonce += 1
        st.rerun()
    elif poly and len(poly) >= 3:
        poly = [(float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1))) for x, y in poly]
        if poly != o.get("roi"):
            o["roi"] = poly
            ss.plot_nonce += 1
            st.rerun()

    st.markdown('<div class="legend">'
                '<span><i style="background:linear-gradient(135deg,#00beff,#ffb000,#3cdc82,#ff5a96)"></i>measured faces</span>'
                '<span><i style="background:#eb4948"></i>analysis boundary (rim excluded)</span>'
                '<span><i style="background:#00c8ff"></i>your region</span>'
                + ('<span><i style="background:#ff00ff"></i>traced fibre centre lines</span>' if TRACE else '') +
                '<span style="color:var(--muted)">uncolored areas touch the rim or region edge and are not measured · '
                'drag a box to zoom · hover a face for its size</span></div>', unsafe_allow_html=True)

    b = st.columns([3, 2, 1.4, 1.4], vertical_alignment="bottom")
    if len(im.channel_names) > 1:
        ch = b[0].selectbox("Lamin channel", range(len(im.channel_names)), index=channel_of(name),
                            format_func=lambda c: im.channel_names[c], key=f"ch_{name}")
        if ch != channel_of(name):
            o["channel"] = ch; o.pop("threshold", None); o.pop("sensitivity", None)
            ss.pop(f"threshold_{name}", None); ss.pop(f"sensitivity_{name}", None); st.rerun()
    if o.get("roi") and b[2].button("✕ Clear region", width="stretch"):
        o.pop("roi", None); ss.plot_nonce += 1; st.rerun()
    if o.get("zoom") and b[3].button("⤢ Reset zoom", width="stretch"):
        o.pop("zoom"); ss.plot_nonce += 1; st.rerun()

    # ------------------------------------------------------------------ numbered faces + corrections
    section("Numbered faces", "grid", "click a face or a table row to identify it · correct mistakes")
    nf = int(r.faces.max())
    sel_id = o.get("selected") if o.get("selected") and o["selected"] <= nf else None
    MODES = ["👆 Select", "🔍 Zoom", "✂️ Erase fibre", "✏️ Draw fibre", "🚫 Exclude face"]
    if not TRACE:
        MODES.remove("✂️ Erase fibre")
    ncol = st.columns([3, 2])
    with ncol[0]:
        mc = st.columns([5, 1.3, 1.3], vertical_alignment="center")
        nmode = mc[0].segmented_control("Mouse on this image", MODES, default="👆 Select", key="num_mode",
                                        label_visibility="collapsed") or "👆 Select"
        n_ed = len(o.get("edits", []))
        if mc[1].button("↶ Undo", disabled=n_ed == 0, width="stretch", help="Undo the last correction"):
            o["edits"].pop(); o.pop("draw_start", None); ss.plot_nonce += 1; st.rerun()
        if mc[2].button("⟲ Reset", disabled=n_ed == 0, width="stretch", help="Remove all corrections"):
            o["edits"] = []; o.pop("draw_start", None); ss.plot_nonce += 1; st.rerun()
        hints = {"👆 Select": "Click a face to identify it.",
                 "🔍 Zoom": "Drag a box to zoom (shared with the panels above).",
                 "✂️ Erase fibre": "Click a traced line that is not a real fibre - the two faces merge.",
                 "✏️ Draw fibre": "Click a start point, then an end point, to add a missing fibre - the face splits. "
                                  "Ends snap to the nearest traced line.",
                 "🚫 Exclude face": "Click a face to leave it out of the measurements (artefact, out of focus…)."}
        st.caption(hints[nmode] + (f"  ·  **{n_ed} correction{'s' if n_ed != 1 else ''}** applied" if n_ed else ""))
        if nmode != "✏️ Draw fibre":
            o.pop("draw_start", None)

        base = pics["Filled faces"] if not single else \
            panels(data[name], name, params_for(name), px, unit, roi_for(name), False,
                   line_width(o, w), edits_for(name))["Filled faces"]
        shown = L.highlight_face(base, r.faces, sel_id) if sel_id and nmode == "👆 Select" else base
        fig2 = go.Figure(go.Image(source=to_uri(shown), hoverinfo="skip"))
        struct = r.extra.get("bw", r.fil)
        area_of = np.zeros(nf + 1)
        if len(r.table):
            area_of[r.table.face.values.astype(int)] = r.table.area.values
        if nmode in ("✂️ Erase fibre", "✏️ Draw fibre") and TRACE:
            # clickable points on the traced lines themselves
            ys, xs = np.nonzero(struct & r.nuc)
            keep = ((ys + xs) % 2 == 0)
            ys, xs = ys[keep], xs[keep]
            tip = "✂️ click to erase this fibre" if nmode.startswith("✂") else "✏️ click to start / end a fibre here"
            fig2.add_trace(go.Scattergl(x=xs, y=ys, mode="markers", marker=dict(size=7, opacity=0), showlegend=False,
                                        customdata=np.c_[np.zeros_like(xs)], hovertemplate=tip + "<extra></extra>"))
        if nmode == "✏️ Draw fibre":
            # anywhere inside faces is clickable too (drawing may start/end off a traced line)
            st3 = max(3, int(max(h, w) / 220))
            gy3, gx3 = np.mgrid[st3 // 2:h:st3, st3 // 2:w:st3]
            m3 = r.nuc[gy3, gx3]
            fig2.add_trace(go.Scattergl(x=gx3[m3], y=gy3[m3], mode="markers", marker=dict(size=st3 + 2, opacity=0),
                                        showlegend=False, customdata=np.c_[np.zeros(m3.sum())],
                                        hovertemplate="✏️ click to start / end a fibre here<extra></extra>"))
            if o.get("draw_start"):
                sx, sy = o["draw_start"]
                fig2.add_trace(go.Scatter(x=[sx], y=[sy], mode="markers", hoverinfo="skip", showlegend=False,
                                          marker=dict(size=14, color="#22d3ee", line=dict(color="white", width=2))))
        if nmode in ("👆 Select", "🚫 Exclude face", "🔍 Zoom"):
            st2 = max(3, int(max(h, w) / 220))
            gy2, gx2 = np.mgrid[st2 // 2:h:st2, st2 // 2:w:st2]
            lab = r.faces[gy2, gx2]
            m = lab > 0
            verb = "🚫 click to exclude" if nmode == "🚫 Exclude face" else ""
            fig2.add_trace(go.Scattergl(
                x=gx2[m], y=gy2[m], mode="markers", marker=dict(size=st2 + 2, opacity=0), showlegend=False,
                customdata=np.c_[lab[m], area_of[lab[m]]],
                hovertemplate=f"<b>Face #%{{customdata[0]:.0f}}</b><br>%{{customdata[1]:,.1f}} {u2}"
                              + (f"<br>{verb}" if verb else "") + "<extra></extra>"))
        # numbers: only for faces big enough to label at the current zoom
        zx0, zx1, zy0, zy1 = o.get("zoom", (0, w, 0, h))
        if len(r.table):
            t = r.table
            cx, cy = t.x.values / px, t.y.values / px
            vis = (cx >= zx0) & (cx <= zx1) & (cy >= zy0) & (cy <= zy1)
            big = t.equiv_diameter.values / px >= (zx1 - zx0) / 45
            keep = vis & (big | (vis.sum() <= 120))
            if sel_id:
                keep |= t.face.values == sel_id
            fig2.add_trace(go.Scatter(x=cx[keep], y=cy[keep], text=t.face.values[keep].astype(int), mode="text",
                                      textfont=dict(size=11, color="white", family="Inter, Arial Black"),
                                      hoverinfo="skip", showlegend=False))
        fig2.update_xaxes(visible=False, range=[zx0, zx1])
        fig2.update_yaxes(visible=False, range=[zy1, zy0], scaleanchor="x")
        zoom_mode = nmode == "🔍 Zoom"
        # Streamlit only reports clicks as selections when the drag mode is not box-select
        fig2.update_layout(height=620, margin=dict(l=0, r=0, t=0, b=0),
                           dragmode="select" if zoom_mode else "pan",
                           plot_bgcolor="black", paper_bgcolor="black", hovermode="closest",
                           hoverlabel=dict(bgcolor=TC["hover_bg"], bordercolor=TC["hover_border"], font=dict(color=TC["ink"])))
        ev2 = st.plotly_chart(fig2, width="stretch", on_select="rerun",
                              selection_mode=("box",) if zoom_mode else ("points",),
                              key=f"num_{name}_{ss.plot_nonce}_{MODES.index(nmode)}",
                              config={"displayModeBar": False, "scrollZoom": False})
        sel2 = getattr(ev2, "selection", None) or (ev2.get("selection") if isinstance(ev2, dict) else None)
        if sel2:
            if sel2.get("box"):
                bx = sel2["box"][-1]
                (x0, x1), (y0, y1) = sorted(bx["x"]), sorted(bx["y"])
                if x1 - x0 > 5 and y1 - y0 > 5:
                    o["zoom"] = (max(0, x0), min(w, x1), max(0, y0), min(h, y1))
                ss.plot_nonce += 1
                st.rerun()
            pts = [p for p in sel2.get("points", []) if p.get("x") is not None]
            if pts:
                p0 = pts[0]
                x_, y_ = float(p0["x"]), float(p0["y"])
                cd = p0.get("customdata")
                fid = int(cd[0] if isinstance(cd, (list, tuple)) else (cd or 0))
                o.setdefault("edits", [])
                if nmode == "👆 Select" and fid > 0:
                    o["selected"] = fid
                    ss[f"tbl_nonce_{name}"] = ss.get(f"tbl_nonce_{name}", 0) + 1   # clear stale row tick
                elif nmode == "✂️ Erase fibre":
                    o["edits"].append(("erase", x_, y_)); o.pop("selected", None)
                elif nmode == "🚫 Exclude face" and fid > 0:
                    o["edits"].append(("exclude", x_, y_)); o.pop("selected", None)
                elif nmode == "✏️ Draw fibre":
                    if o.get("draw_start"):
                        sx, sy = o.pop("draw_start")
                        o["edits"].append(("draw", sx, sy, x_, y_)); o.pop("selected", None)
                    else:
                        o["draw_start"] = (x_, y_)
                ss.plot_nonce += 1
                st.rerun()

    with ncol[1]:
        ft = L.face_table(r)
        if sel_id and len(ft):
            row = ft[ft["Face #"] == sel_id].iloc[0]
            cards([("Selected", f"#{sel_id}", "white ring in the image", "select"),
                   ("Area", L.fmt(float(row.iloc[1])), u2, "median"),
                   ("Eq. diameter", L.fmt(float(row.iloc[2])), unit, "length"),
                   ("Circularity", f"{row['Circularity']:.2f}", "1 = circle", "coverage")]
                  + ([("Edges", f"{int(row['Edges'])}", "bordering this face", "edges")] if "Edges" in row else []))
            z1, z2 = st.columns(2)
            if z1.button("🔎 Zoom to this face", width="stretch"):
                fy, fx = np.nonzero(r.faces == sel_id)
                half = max(np.ptp(fy), np.ptp(fx), 40) * 1.6
                cxf, cyf = fx.mean(), fy.mean()
                o["zoom"] = (max(0, cxf - half), min(w, cxf + half), max(0, cyf - half), min(h, cyf + half))
                ss.plot_nonce += 1; st.rerun()
            if z2.button("✕ Clear selection", width="stretch"):
                o.pop("selected", None); ss[f"tbl_nonce_{name}"] = ss.get(f"tbl_nonce_{name}", 0) + 1
                st.rerun()

        hl = ft.style.apply(lambda rw: ["background-color: rgba(34,211,238,.35); font-weight: 700"] * len(rw)
                            if rw["Face #"] == sel_id else [""] * len(rw), axis=1) \
                     .format(precision=2, subset=ft.columns[1:])
        tkey = f"tbl_{name}_{ss.get(f'tbl_nonce_{name}', 0)}"
        tev = st.dataframe(hl, hide_index=True, width="stretch", height=360 if sel_id else 470,
                           on_select="rerun", selection_mode="single-row", key=tkey)
        rows = list(tev.selection.rows) if tev and tev.selection else []
        if rows and ss.get(f"{tkey}_last") != rows:
            ss[f"{tkey}_last"] = rows
            o["selected"] = int(ft.iloc[rows[0]]["Face #"])
            ss.plot_nonce += 1
            st.rerun()
        st.download_button("⬇  Face table (Excel)", L.excel_bytes([r]),
                           f"{name.rsplit('.', 1)[0]}_faces.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           type="primary", width="stretch")

        # training data export
        with st.container(border=True):
            st.markdown(f"<div style='display:flex;gap:.5rem;align-items:center;font-weight:700;color:var(--ink)'>"
                        f"{ui.icon('sparkle', 16)} Training set</div>"
                        "<div style='color:var(--ink2);font-size:.8rem;margin:.2rem 0 .5rem'>Once the faces look right, "
                        "save this image with its corrected segmentation. A set of corrected images lets the "
                        "settings be tuned - or a model trained - on your lab's own data.</div>",
                        unsafe_allow_html=True)
            stem = name.rsplit(".", 1)[0]
            tb = st.columns(2)
            if tb[0].button("💾 Save to training set", width="stretch"):
                folder = TRAIN_DIR / f"{stem}_{pd.Timestamp.now():%Y%m%d-%H%M%S}"
                folder.mkdir(parents=True, exist_ok=True)
                for fn, blob in L.training_bundle(im, r).items():
                    (folder / fn).write_bytes(blob)
                n_saved = len([p for p in TRAIN_DIR.iterdir() if p.is_dir()])
                st.toast(f"Saved · {n_saved} example{'s' if n_saved != 1 else ''} in the training set", icon="💾")
            zb = io.BytesIO()
            with zipfile.ZipFile(zb, "w", zipfile.ZIP_DEFLATED) as zf:
                for fn, blob in L.training_bundle(im, r).items():
                    zf.writestr(f"{stem}/{fn}", blob)
            tb[1].download_button("⬇ Download (.zip)", zb.getvalue(), f"{stem}_training.zip", "application/zip",
                                  width="stretch")
            n_saved = len([p for p in TRAIN_DIR.iterdir() if p.is_dir()]) if TRAIN_DIR.exists() else 0
            st.caption(f"Folder: `{TRAIN_DIR}` · {n_saved} saved example{'s' if n_saved != 1 else ''}")

# ---------------------------------------------------------------------------- tab 2
with tab_res:
    prog = st.progress(0.0, "Analyzing…")
    results = []
    for i, n in enumerate(names):
        results.append(result(n))
        prog.progress((i + 1) / len(names), f"Analyzing {n}…")
    prog.empty()

    if len(results) > 1:
        ps = L.pooled_stats(results)
        cards([("Images", str(ps["Images"]), "analyzed", "images"),
               ("Faces", f"{ps['Faces, total']:,}", "all images", "faces"),
               ("Pooled median", L.fmt(ps[f"Pooled median area ({u2})"]), u2, "median"),
               ("Mean of image means", L.fmt(ps[f"Mean of image means ({u2})"]), u2, "mean"),
               ("SD of image means", L.fmt(ps[f"SD of image means ({u2})"]), u2, "chart")])

    section("Summary per image", "table")
    summ = L.summary_table(results)
    st.dataframe(pd.DataFrame({r.name: {k: L.fmt(v) for k, v in r.stats.items()} for r in results}),
                 width="stretch", height=35 * (len(summ) + 1) + 3)
    if len(results) > 1:
        section("Comparison", "chart", "every face, per image")
        with st.container(border=True):
            st.plotly_chart(ui.chart_compare(results, u2, L._short), width="stretch",
                            config={"displayModeBar": False})

    section("Distributions", "chart", "hover for values")
    pick = st.selectbox("Image", names, key="res_pick") if len(names) > 1 else names[0]
    rr = results[names.index(pick)]
    if len(rr.table):
        cfg = {"displayModeBar": False}
        g = st.columns(2)
        with g[0].container(border=True):
            st.plotly_chart(ui.chart_hist(rr, u2), width="stretch", config=cfg)
        with g[1].container(border=True):
            st.plotly_chart(ui.chart_ecdf(rr, u2), width="stretch", config=cfg)
        g = st.columns(2)
        with g[0].container(border=True):
            st.plotly_chart(ui.chart_shape(rr, u2), width="stretch", config=cfg)
        with g[1].container(border=True):
            st.plotly_chart(ui.chart_edges(rr) if "edges" in rr.table else ui.chart_fraction(rr, u2),
                            width="stretch", config=cfg)
        section("Area map", "images", "light = small faces, dark = large faces")
        am = st.columns([1, 2, 1])
        am[1].image(L.area_map(images[pick], rr), width="stretch")
    else:
        st.info("No faces measured in this image with the current settings.")

    section("Per-face measurements", "table")
    st.caption(f"Areas in {u2}; lengths and positions in {unit}. Circularity = 4πA/P² (1 = circle).")
    st.dataframe(rr.table.round(3), width="stretch", height=320, hide_index=True)

    all_faces = pd.concat([r.table.assign(image=r.name) for r in results], ignore_index=True)
    all_faces = all_faces[["image"] + [c for c in all_faces.columns if c != "image"]]
    d0, d1, d2 = st.columns(3)
    d0.download_button("⬇  All images (Excel)", L.excel_bytes(results), "lamin_mesh_results.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary",
                       width="stretch", help="One sheet with the summary + one sheet of numbered faces per image")
    d1.download_button("⬇  All faces (CSV)", all_faces.to_csv(index=False), "lamin_mesh_faces.csv",
                       "text/csv", width="stretch")
    d2.download_button("⬇  Summary (CSV)", summ.T.to_csv(), "lamin_mesh_summary.csv", "text/csv",
                       width="stretch")

# ---------------------------------------------------------------------------- tab 3
with tab_rep:
    left, right = st.columns([3, 2])
    with left:
        section("PDF report", "doc")
        title = st.text_input("Report title", "Lamin meshwork analysis")
        notes = st.text_area("Notes (sample, condition, microscope…)", "")
        tables = st.checkbox("Include full per-face tables (adds pages)", False)
        if st.button("Generate PDF", type="primary", width="stretch"):
            with st.spinner("Building report…"):
                res = [result(n) for n in names]
                ss.pdf = L.build_report(res, images, title=title, notes=notes, include_face_tables=tables)
        if ss.get("pdf"):
            st.download_button("⬇  Download PDF report", ss.pdf, "lamin_mesh_report.pdf", "application/pdf",
                               type="primary", width="stretch")
    with right:
        st.markdown(f'<div class="feature" style="margin-top:3.6rem"><div class="n">{ui.icon("doc", 19)}</div>'
                    '<div class="h">What\'s inside</div>'
                    '<div class="d">• Summary table for every image (plus pooled stats)<br>'
                    '• Original · traced meshwork · filled faces for each image<br>'
                    '• Numbered-faces page matching the Excel table<br>'
                    '• Area map, histogram, cumulative distribution, size vs shape<br>'
                    '• Every setting used and a methods paragraph with citation</div></div>',
                    unsafe_allow_html=True)
