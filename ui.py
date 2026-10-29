"""
Look & feel for the Lamin Mesh Analyzer: CSS theme, icon stat cards, section headers, and
interactive dark Plotly charts (the PDF report keeps its own matplotlib charts).
"""
import html
from urllib.parse import quote

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import theme


# faint meshwork pattern for the page background
_MESH_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' width='260' height='260' viewBox='0 0 260 260'>"
    "<g fill='none' stroke='white' stroke-opacity='.045' stroke-width='1.2'>"
    "<path d='M0 40 Q40 10 80 38 T160 30 T260 44'/><path d='M0 120 Q50 90 96 118 T190 110 T260 126'/>"
    "<path d='M0 205 Q46 176 92 204 T184 196 T260 210'/>"
    "<path d='M40 0 Q20 60 44 120 T38 260'/><path d='M130 0 Q150 64 126 128 T134 260'/>"
    "<path d='M214 0 Q196 58 220 118 T208 260'/>"
    "<circle cx='86' cy='78' r='22'/><circle cx='176' cy='164' r='28'/><circle cx='64' cy='168' r='14'/>"
    "<circle cx='190' cy='70' r='16'/></g></svg>")
_MESH_URI = "data:image/svg+xml;utf8," + quote(_MESH_SVG)
_MESH_URI_LIGHT = "data:image/svg+xml;utf8," + quote(_MESH_SVG.replace("stroke='white' stroke-opacity='.045'",
                                                                       "stroke='#0f172a' stroke-opacity='.07'"))

# small line icons (24x24, stroke = currentColor)
ICONS = {
    "faces": "<path d='M12 3l8 4.5v9L12 21l-8-4.5v-9z'/><path d='M12 12l8-4.5M12 12v9M12 12L4 7.5'/>",
    "median": "<path d='M4 20V10M10 20V4M16 20v-8M22 20H2'/>",
    "mean": "<path d='M3 17l6-6 4 4 8-8'/><path d='M14 7h7v7'/>",
    "edges": "<circle cx='5' cy='6' r='2'/><circle cx='19' cy='6' r='2'/><circle cx='12' cy='19' r='2'/>"
             "<path d='M7 6h10M6 8l5 9M18 8l-5 9'/>",
    "length": "<path d='M3 17L17 3l4 4L7 21z'/><path d='M7 13l2 2M10 10l2 2M13 7l2 2'/>",
    "coverage": "<circle cx='12' cy='12' r='9'/><path d='M12 3a9 9 0 0 1 9 9h-9z'/>",
    "threshold": "<path d='M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6'/>",
    "images": "<rect x='3' y='3' width='18' height='18' rx='3'/><circle cx='9' cy='9' r='2'/><path d='M21 15l-5-5L5 21'/>",
    "select": "<path d='M4 4l7 17 2.5-7.5L21 11z'/>",
    "chart": "<path d='M3 3v18h18'/><path d='M7 15l4-4 3 3 5-6'/>",
    "table": "<rect x='3' y='4' width='18' height='16' rx='2'/><path d='M3 10h18M9 4v16'/>",
    "doc": "<path d='M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z'/><path d='M14 3v6h6M8 13h8M8 17h5'/>",
    "grid": "<rect x='3' y='3' width='7' height='7' rx='1'/><rect x='14' y='3' width='7' height='7' rx='1'/>"
            "<rect x='3' y='14' width='7' height='7' rx='1'/><rect x='14' y='14' width='7' height='7' rx='1'/>",
    "sparkle": "<path d='M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z'/>",
}


def icon(name, size=18):
    return (f"<svg width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' stroke='currentColor' "
            f"stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'>{ICONS.get(name, '')}</svg>")


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{
  --bg:#070b16; --ink:#e8eefc; --ink2:#9fb0d0; --muted:#8190af; --line:rgba(255,255,255,.08);
  --glass:linear-gradient(145deg, rgba(255,255,255,.065), rgba(255,255,255,.018));
  --accent:#22d3ee; --accent2:#a78bfa; --accent3:#f472b6;
  --grad:linear-gradient(120deg,#22d3ee 0%,#818cf8 50%,#f472b6 100%);
  --aur1:rgba(129,140,248,.22); --aur2:rgba(34,211,238,.18); --aur3:rgba(244,114,182,.12); --mesh-op:.9;
  --sidebar-bg:linear-gradient(180deg, rgba(14,20,38,.92), rgba(8,12,24,.96));
  --drop-bg:linear-gradient(145deg, rgba(34,211,238,.08), rgba(129,140,248,.06)); --drop-bd:rgba(34,211,238,.5);
  --shadow:rgba(0,0,0,.35); --shadow-soft:rgba(0,0,0,.25); --hi:rgba(255,255,255,.07);
  --title-grad:linear-gradient(90deg,#e8eefc,#67e8f9,#a5b4fc,#f9a8d4,#e8eefc);
  --chip-ink:#a5f3fc; --chip-v-ink:#ddd6fe; --chip-p-ink:#fbcfe8;
  --soft:rgba(255,255,255,.04); --tab-sel-ink:#ffffff; --value-ink:#ffffff; --link:#a5b4fc;
  --sec-ic-ink:#a5f3fc; --btn-hover-ink:#a5f3fc; --toast-bg:rgba(17,26,46,.92);
}
/* Light theme: same layout and motion, colours for a pale-grey page. Text contrast >= 4.5:1. */
html[data-theme="light"]{
  --bg:#eef1f6; --ink:#0f172a; --ink2:#334155; --muted:#475569; --line:rgba(15,23,42,.12);
  --glass:linear-gradient(145deg, rgba(255,255,255,.92), rgba(255,255,255,.72));
  --accent:#0e7490; --accent2:#6d28d9; --accent3:#be185d;
  --grad:linear-gradient(120deg,#0891b2 0%,#6366f1 50%,#db2777 100%);
  --aur1:rgba(99,102,241,.10); --aur2:rgba(8,145,178,.09); --aur3:rgba(219,39,119,.06); --mesh-op:.55;
  --sidebar-bg:linear-gradient(180deg, rgba(255,255,255,.94), rgba(241,244,249,.97));
  --drop-bg:linear-gradient(145deg, rgba(8,145,178,.06), rgba(99,102,241,.05)); --drop-bd:rgba(14,116,144,.55);
  --shadow:rgba(15,23,42,.12); --shadow-soft:rgba(15,23,42,.08); --hi:rgba(255,255,255,.8);
  --title-grad:linear-gradient(90deg,#0f172a,#0e7490,#4f46e5,#be185d,#0f172a);
  --chip-ink:#155e75; --chip-v-ink:#5b21b6; --chip-p-ink:#9d174d;
  --soft:rgba(15,23,42,.04); --tab-sel-ink:#0f172a; --value-ink:#0f172a; --link:#4338ca;
  --sec-ic-ink:#0e7490; --btn-hover-ink:#0e7490; --toast-bg:rgba(255,255,255,.96);
}
html, body, .stApp, button, input, textarea, select, label { font-family:'Inter',system-ui,sans-serif !important; }
.stApp { background: var(--bg); isolation: isolate; }   /* background layers sit at z -1 inside */
.stApp::before { content:""; position:fixed; inset:-20%; z-index:-1; pointer-events:none;
  background:
    radial-gradient(38% 30% at 78% 8%, var(--aur1), transparent 70%),
    radial-gradient(34% 28% at 12% 14%, var(--aur2), transparent 70%),
    radial-gradient(30% 26% at 60% 92%, var(--aur3), transparent 70%);
  animation: aurora 22s ease-in-out infinite alternate; filter: blur(10px); }
.stApp::after { content:""; position:fixed; inset:0; z-index:-1; pointer-events:none;
  background-image:url("__MESH__"); background-size:260px 260px; opacity:var(--mesh-op);
  mask-image: radial-gradient(ellipse at 50% 0%, black 30%, transparent 85%); }
html[data-theme="light"] .stApp::after { background-image:url("__MESH_LIGHT__"); }
@keyframes aurora { 0%{transform:translate(0,0) rotate(0deg)} 50%{transform:translate(3%,2%) rotate(4deg)}
                    100%{transform:translate(-3%,1%) rotate(-3deg)} }
@keyframes fadeUp { from{opacity:0; transform:translateY(10px)} to{opacity:1; transform:none} }
@keyframes shimmer { 0%{background-position:0% 50%} 100%{background-position:200% 50%} }
/* top bar: Streamlit's header band, with the Light | Dark switch pinned at its right */
[data-testid="stHeader"] { background: color-mix(in srgb, var(--bg) 72%, transparent); backdrop-filter: blur(14px);
  border-bottom: 1px solid var(--line); }
.st-key-theme_choice { position: fixed; top: .55rem; right: 1.1rem; z-index: 999991; width: auto !important; }
.st-key-theme_choice [data-testid="stButtonGroup"] { background: var(--glass); border: 1px solid var(--line);
  border-radius: 999px; padding: 3px; box-shadow: 0 6px 20px var(--shadow-soft); backdrop-filter: blur(12px); gap: 2px; }
.st-key-theme_choice button { border-radius: 999px !important; border: none !important; min-height: 1.9rem !important;
  padding: .15rem .8rem !important; background: transparent !important; color: var(--ink2) !important;
  font-size: .8rem !important; font-weight: 600; transition: all .2s ease; }
.st-key-theme_choice button[kind="segmented_controlActive"] { background: var(--grad) !important; color: #fff !important;
  box-shadow: 0 3px 12px rgba(99,102,241,.35); }
[data-testid="stAppDeployButton"], #MainMenu, footer, [data-testid="stDecoration"] { display:none !important; }
.block-container { padding-top: 4.6rem; max-width: 1520px; position:relative; }

/* ---------- sidebar */
[data-testid="stSidebar"] { background: var(--sidebar-bg);
  border-right:1px solid var(--line); backdrop-filter: blur(18px); }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { font-size:.68rem !important; letter-spacing:.18em;
  text-transform:uppercase; color:var(--muted) !important; font-weight:700 !important; margin:1.1rem 0 .2rem;
  display:flex; align-items:center; gap:.5rem; }
[data-testid="stSidebar"] h3::after { content:""; flex:1; height:1px; background:linear-gradient(90deg, var(--line), transparent); }
.brand { display:flex; align-items:center; gap:.8rem; margin:.1rem 0 1.1rem; padding:.7rem; border-radius:16px;
  background:var(--glass); border:1px solid var(--line); }
.brand img { width:46px; height:46px; border-radius:12px; box-shadow:0 0 0 1px var(--line), 0 6px 22px rgba(34,211,238,.35); }
.brand .t { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.08rem; line-height:1.1;
  background:var(--grad); -webkit-background-clip:text; color:transparent; }
.brand .s { font-size:.7rem; color:var(--muted); margin-top:.15rem; }
[data-testid="stFileUploaderDropzone"] { background: var(--drop-bg);
  border:1.5px dashed var(--drop-bd); border-radius:16px; transition: all .2s ease; }
[data-testid="stFileUploaderDropzone"]:hover { border-color:#f472b6; box-shadow:0 0 0 4px rgba(244,114,182,.12); }

/* ---------- hero */
.hero { position:relative; overflow:hidden; padding:1.5rem 1.8rem; border-radius:24px; margin-bottom:1.2rem;
  background: var(--glass); border:1px solid var(--line); backdrop-filter: blur(16px);
  box-shadow: 0 20px 60px var(--shadow), inset 0 1px 0 var(--hi); animation: fadeUp .5s ease both; }
.hero::before { content:""; position:absolute; inset:0; border-radius:24px; padding:1px; pointer-events:none;
  background: var(--grad); -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite: xor; mask-composite: exclude; opacity:.55; }
.hero h1 { font-family:'Space Grotesk',sans-serif; font-size:2.4rem; font-weight:700; margin:0; padding:0; letter-spacing:-.03em;
  background: var(--title-grad); background-size:200% auto;
  -webkit-background-clip:text; color:transparent; animation: shimmer 8s linear infinite; }
.hero p { color: var(--ink2); margin:.45rem 0 0; font-size:1rem; max-width:720px; }
.hero .art { position:absolute; right:1.6rem; top:50%; transform:translateY(-50%); width:120px; height:120px;
  border-radius:28px; box-shadow:0 0 60px rgba(129,140,248,.45), 0 0 0 1px var(--line); }
.hero.compact { padding:1rem 1.4rem; }
.hero.compact h1 { font-size:1.55rem; }
.hero.compact .art { width:64px; height:64px; border-radius:16px; right:1.2rem; }
.chip { display:inline-flex; align-items:center; gap:.35rem; padding:.22rem .7rem; border-radius:999px; font-size:.72rem;
  font-weight:600; background:rgba(34,211,238,.1); color:var(--chip-ink); border:1px solid rgba(34,211,238,.28); margin:0 .4rem .7rem 0; }
.chip.v { background:rgba(167,139,250,.12); color:var(--chip-v-ink); border-color:rgba(167,139,250,.3); }
.chip.p { background:rgba(244,114,182,.1); color:var(--chip-p-ink); border-color:rgba(244,114,182,.3); }

/* ---------- tabs */
.stTabs [data-baseweb="tab-list"] { gap:.35rem; border-bottom:none; padding:.35rem; border-radius:18px;
  background: var(--glass); border:1px solid var(--line); width:fit-content; backdrop-filter: blur(12px); }
.stTabs [data-baseweb="tab"] { background:transparent; border-radius:13px; padding:.5rem 1.15rem; height:auto;
  color: var(--ink2); font-weight:600; transition: all .2s ease; }
.stTabs [data-baseweb="tab"]:hover { color: var(--ink); background: var(--soft); }
.stTabs [aria-selected="true"] { background: linear-gradient(120deg, rgba(34,211,238,.25), rgba(129,140,248,.25), rgba(244,114,182,.2));
  color: var(--tab-sel-ink) !important; box-shadow: 0 4px 18px rgba(99,102,241,.3), inset 0 1px 0 var(--hi); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
.stTabs [data-baseweb="tab-panel"] { padding-top:1rem; }

/* ---------- stat cards */
.cards { display:grid; grid-template-columns: repeat(auto-fit, minmax(140px,1fr)); gap:.8rem; margin:.5rem 0 1.1rem; }
.card { position:relative; overflow:hidden; border-radius:18px; padding:.9rem 1rem .85rem; background: var(--glass);
  border:1px solid var(--line); backdrop-filter: blur(14px); animation: fadeUp .45s ease both;
  box-shadow: 0 10px 30px var(--shadow-soft), inset 0 1px 0 var(--hi); transition: transform .2s ease, box-shadow .2s ease; }
.card:hover { transform: translateY(-3px); box-shadow: 0 16px 40px var(--shadow), 0 0 0 1px rgba(34,211,238,.35); }
.card::after { content:""; position:absolute; right:-30px; top:-30px; width:90px; height:90px; border-radius:50%;
  background: radial-gradient(circle, var(--c, rgba(34,211,238,.35)), transparent 70%); opacity:.55; }
.card .top { display:flex; align-items:center; gap:.55rem; }
.card .ic { width:30px; height:30px; border-radius:10px; display:flex; align-items:center; justify-content:center;
  color:#fff; background: var(--c2, linear-gradient(135deg,#22d3ee,#818cf8)); box-shadow:0 4px 14px rgba(0,0,0,.3); }
.card .l { font-size:.68rem; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); font-weight:700; }
.card .v { font-family:'Space Grotesk',sans-serif; font-size:1.85rem; font-weight:700; margin-top:.35rem; letter-spacing:-.02em;
  color:var(--value-ink); line-height:1.1; }
.card .s { font-size:.74rem; color:var(--ink2); margin-top:.1rem; }
.card:nth-child(2){animation-delay:.05s} .card:nth-child(3){animation-delay:.1s}
.card:nth-child(4){animation-delay:.15s} .card:nth-child(5){animation-delay:.2s}

/* ---------- sections, glass containers */
.section { display:flex; align-items:center; gap:.6rem; margin:1.5rem 0 .6rem; font-family:'Space Grotesk',sans-serif;
  font-size:1.2rem; font-weight:600; color:var(--ink); }
.section .ic { width:32px; height:32px; border-radius:10px; display:flex; align-items:center; justify-content:center;
  background: linear-gradient(135deg, rgba(34,211,238,.2), rgba(129,140,248,.25)); color:var(--sec-ic-ink); border:1px solid var(--line); }
.section::after { content:""; flex:1; height:1px; background: linear-gradient(90deg, rgba(129,140,248,.45), transparent); }
.section .sub { font-family:'Inter',sans-serif; font-size:.8rem; color:var(--muted); font-weight:500; }
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) { border-radius:20px; }
div[data-testid="stVerticalBlockBorderWrapper"][class*="border"], .stVerticalBlock [data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--glass); border-color: var(--line) !important; backdrop-filter: blur(14px); }
.panel-title { display:grid; grid-template-columns:1fr 1fr 1fr; text-align:center; margin:.1rem 0 .45rem; }
.panel-title span { justify-self:center; font-size:.7rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase;
  color:var(--ink2); padding:.25rem .8rem; border-radius:999px; background:var(--soft); border:1px solid var(--line); }
[data-testid="stPlotlyChart"] { border-radius:18px; overflow:hidden; }
.img-frame [data-testid="stPlotlyChart"], [data-testid="stPlotlyChart"]:has(.imagelayer image) {
  background:#000; border:1px solid var(--line); box-shadow: 0 18px 50px var(--shadow), 0 0 0 1px rgba(129,140,248,.15); }
[data-testid="stImage"] img { border-radius: 16px; box-shadow: 0 18px 50px var(--shadow); }
.legend { color:var(--ink2); font-size:.78rem; margin:.7rem 0 .4rem; display:flex; flex-wrap:wrap; gap:.35rem .9rem; align-items:center; }
.legend i { display:inline-block; width:.8rem; height:.8rem; border-radius:4px; vertical-align:-1px; margin-right:.35rem; }

/* ---------- controls */
.stButton > button, .stDownloadButton > button { border-radius:14px; border:1px solid var(--line); min-height:2.6rem;
  background: var(--soft); font-weight:600; transition: all .18s ease; backdrop-filter: blur(8px); }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--accent); color:var(--btn-hover-ink);
  transform: translateY(-1px); box-shadow:0 6px 20px rgba(34,211,238,.18); }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
  background: linear-gradient(120deg,#06b6d4,#6366f1,#ec4899); background-size:200% auto; border:none; color:white;
  box-shadow: 0 8px 24px rgba(99,102,241,.35); }
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {
  color:white; background-position:100% 0; box-shadow: 0 10px 30px rgba(236,72,153,.35); }
[data-testid="stSlider"] [role="slider"] { box-shadow: 0 0 0 5px rgba(34,211,238,.18), 0 0 18px rgba(34,211,238,.6); }
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] { font-weight:700; }
[data-testid="stExpander"] details { border-radius:14px; border-color: var(--line); background: var(--soft); }
[data-testid="stDataFrame"] { border-radius:16px; overflow:hidden; border:1px solid var(--line); }
[data-baseweb="segmented-control"], [data-testid="stButtonGroup"] button { border-radius:12px !important; }
[data-testid="stToast"] { background: var(--toast-bg); border:1px solid rgba(34,211,238,.35); backdrop-filter: blur(12px); }

/* ---------- landing */
.feature { position:relative; border-radius:20px; padding:1.2rem 1.25rem; height:100%; background: var(--glass);
  border:1px solid var(--line); backdrop-filter: blur(14px); animation: fadeUp .5s ease both; transition: transform .2s ease; }
.feature:hover { transform: translateY(-4px); box-shadow: 0 0 0 1px rgba(129,140,248,.4), 0 16px 40px var(--shadow); }
.feature .n { width:38px; height:38px; border-radius:12px; display:flex; align-items:center; justify-content:center;
  color:#fff; background: var(--grad); margin-bottom:.8rem; box-shadow: 0 6px 20px rgba(99,102,241,.4); }
.feature .h { font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:1.05rem; color:var(--ink); margin-bottom:.3rem; }
.feature .d { color:var(--ink2); font-size:.86rem; line-height:1.55; }
.cite { color:var(--muted); font-size:.75rem; margin-top:.6rem; }
.cite a { color:var(--link); text-decoration:none; }
</style>
""".replace("__MESH_LIGHT__", _MESH_URI_LIGHT).replace("__MESH__", _MESH_URI)


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


CARD_TONES = [
    ("rgba(34,211,238,.45)", "linear-gradient(135deg,#06b6d4,#3b82f6)"),
    ("rgba(129,140,248,.45)", "linear-gradient(135deg,#6366f1,#a855f7)"),
    ("rgba(244,114,182,.4)", "linear-gradient(135deg,#ec4899,#f97316)"),
    ("rgba(52,211,153,.4)", "linear-gradient(135deg,#10b981,#06b6d4)"),
    ("rgba(251,191,36,.4)", "linear-gradient(135deg,#f59e0b,#ef4444)"),
]


def cards(items):
    """items: (label, value, sub, icon-name)."""
    body = []
    for i, it in enumerate(items):
        label, value, sub = it[:3]
        ic = it[3] if len(it) > 3 else "sparkle"
        c1, c2 = CARD_TONES[i % len(CARD_TONES)]
        body.append(f'<div class="card" style="--c:{c1};--c2:{c2}"><div class="top"><div class="ic">{icon(ic, 16)}</div>'
                    f'<div class="l">{html.escape(label)}</div></div><div class="v">{html.escape(value)}</div>'
                    f'<div class="s">{html.escape(sub)}</div></div>')
    st.markdown(f'<div class="cards">{"".join(body)}</div>', unsafe_allow_html=True)


def section(text, ic="sparkle", sub=""):
    st.markdown(f'<div class="section"><div class="ic">{icon(ic, 17)}</div>{html.escape(text)}'
                + (f'<span class="sub">{html.escape(sub)}</span>' if sub else "") + '</div>',
                unsafe_allow_html=True)


# ============================================================================ charts
# Chart colours follow the Light / Dark switch (theme.CHART: validated categorical steps per surface).
_active = {"theme": "dark"}


def use_theme(value):
    """Called once per page run with the session's theme."""
    _active["theme"] = value if value in theme.CHART else "dark"


def _c():
    return theme.CHART[_active["theme"]]
def _style(fig, title=None, height=330):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=46 if title else 12, b=10),
        title=dict(text=title, x=0.01, y=0.97, font=dict(family="Space Grotesk, Inter", size=15, color=_c()["ink"])) if title else None,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=_c()["ink2"], size=12), showlegend=False,
        hoverlabel=dict(bgcolor=_c()["hover_bg"], bordercolor=_c()["hover_border"], font=dict(color=_c()["ink"], family="Inter")),
        bargap=0.12)
    fig.update_xaxes(gridcolor=_c()["grid"], zeroline=False, linecolor=_c()["axis"], ticks="outside",
                     tickcolor=_c()["axis"])
    fig.update_yaxes(gridcolor=_c()["grid"], zeroline=False, linecolor="rgba(0,0,0,0)")
    return fig


def _log_axis(fig, a, title):
    """Log x-axis with clean 1-2-5 tick labels."""
    tv, tt = _log_ticks(a.min(), a.max())
    fig.update_xaxes(type="log", tickvals=[10 ** v for v in tv], ticktext=tt, title_text=title)


def _log_ticks(lo, hi):
    vals = [v for v in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6)
            if lo * 0.8 <= v <= hi * 1.25]
    return [np.log10(v) for v in vals], [f"{v:,.0f}" if v >= 1 else f"{v:g}" for v in vals]


def _center_lines(fig, a, log=True):
    # median label to the left of its line, mean label to the right, so they never collide
    for v, col, dash, lab, side in ((np.median(a), _c()["ink"], "solid", "median", "right"),
                                    (np.mean(a), _c()["orange"], "dash", "mean", "left")):
        x = np.log10(v) if log else v
        fig.add_vline(x=x, line=dict(color=col, width=1.6, dash=dash))
        fig.add_annotation(x=x, y=1, yref="paper", text=f"{lab} {v:,.3g}", showarrow=False, yanchor="bottom",
                           xanchor=side, xshift=-5 if side == "right" else 5, font=dict(size=11, color=col))


def chart_hist(r, u2):
    a = r.table.area.values
    la = np.log10(a)
    edges = np.linspace(la.min(), la.max() + 1e-9, 31)
    cnt, _ = np.histogram(la, edges)
    ctr = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure(go.Bar(x=ctr, y=cnt, width=np.diff(edges) * 0.86,
                           marker=dict(color=_c()["blue"], line=dict(width=0), cornerradius=4),
                           customdata=np.c_[10 ** edges[:-1], 10 ** edges[1:]],
                           hovertemplate=f"%{{customdata[0]:,.0f}}–%{{customdata[1]:,.0f}} {u2}<br><b>%{{y}} faces</b><extra></extra>"))
    tv, tt = _log_ticks(a.min(), a.max())
    fig.update_xaxes(tickvals=tv, ticktext=tt, title_text=f"Face area ({u2}, log scale)")
    fig.update_yaxes(title_text="Faces")
    _center_lines(fig, a)
    return _style(fig, "Area distribution")


def chart_ecdf(r, u2):
    a = np.sort(r.table.area.values)
    y = np.arange(1, len(a) + 1) / len(a) * 100
    fig = go.Figure(go.Scatter(x=a, y=y, mode="lines", line=dict(color=_c()["blue"], width=2.5, shape="hv"),
                               fill="tozeroy", fillcolor=_c()["fill"],
                               hovertemplate=f"%{{y:.0f}}% of faces ≤ %{{x:,.1f}} {u2}<extra></extra>"))
    fig.add_hline(y=50, line=dict(color=_c()["axis"], width=1, dash="dot"))
    _log_axis(fig, a, f"Face area ({u2}, log scale)")
    fig.update_yaxes(title_text="Cumulative % of faces", range=[0, 101], ticksuffix="%")
    return _style(fig, "Cumulative distribution")


def chart_shape(r, u2):
    t = r.table
    fig = go.Figure(go.Scattergl(
        x=t.area, y=t.circularity, mode="markers",
        marker=dict(size=8, color=_c()["violet"], opacity=0.7, line=dict(width=1, color=_c()["point_edge"])),
        customdata=np.c_[t.face],
        hovertemplate=f"<b>Face #%{{customdata[0]:.0f}}</b><br>%{{x:,.1f}} {u2}<br>circularity %{{y:.2f}}<extra></extra>"))
    _log_axis(fig, t.area.values, f"Face area ({u2}, log scale)")
    fig.update_yaxes(title_text="Circularity (1 = circle)", range=[0, 1.03])
    return _style(fig, "Size vs shape")


def chart_edges(r):
    e = r.table.edges.values.astype(int)
    vals, cnt = np.unique(e, return_counts=True)
    fig = go.Figure(go.Bar(x=vals, y=cnt, marker=dict(color=_c()["aqua"], cornerradius=4),
                           hovertemplate="%{x} edges<br><b>%{y} faces</b><extra></extra>"))
    fig.update_xaxes(title_text="Edges per face", dtick=1)
    fig.update_yaxes(title_text="Faces")
    med = float(np.median(e))
    fig.add_vline(x=med, line=dict(color=_c()["ink"], width=1.6))
    fig.add_annotation(x=med, y=1, yref="paper", text=f"median {med:g}", showarrow=False, yanchor="bottom",
                       xanchor="left", xshift=4, font=dict(size=11, color=_c()["ink"]))
    return _style(fig, "Edges per face")


def chart_fraction(r, u2):
    a = r.table.area.values
    la = np.log10(a)
    edges = np.linspace(la.min(), la.max() + 1e-9, 17)
    w, _ = np.histogram(la, edges, weights=a)
    ctr = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure(go.Bar(x=ctr, y=100 * w / a.sum(), width=np.diff(edges) * 0.86,
                           marker=dict(color=_c()["aqua"], cornerradius=4),
                           customdata=np.c_[10 ** edges[:-1], 10 ** edges[1:]],
                           hovertemplate=f"%{{customdata[0]:,.0f}}–%{{customdata[1]:,.0f}} {u2}<br><b>%{{y:.1f}}% of face area</b><extra></extra>"))
    tv, tt = _log_ticks(a.min(), a.max())
    fig.update_xaxes(tickvals=tv, ticktext=tt, title_text=f"Face area ({u2}, log scale)")
    fig.update_yaxes(title_text="% of total face area", ticksuffix="%")
    return _style(fig, "Area contributed by size class")


def chart_compare(results, u2, short):
    fig = go.Figure()
    for r in results:
        a = r.table.area.values
        fig.add_trace(go.Box(y=a, name=short(r.name), boxpoints="all", jitter=0.45, pointpos=0,
                             marker=dict(size=4, color=_c()["blue"], opacity=0.45), line=dict(color=_c()["ink2"], width=1.4),
                             fillcolor=_c()["fill"],
                             hovertemplate=f"%{{y:,.1f}} {u2}<extra>%{{x}}</extra>"))
    fig.update_yaxes(type="log", title_text=f"Face area ({u2}, log scale)")
    return _style(fig, "Face area by image", height=360)
