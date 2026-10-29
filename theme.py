"""
Light / dark theme for the Lamin Mesh Analyzer.

* The choice is saved in a small settings file (not the browser), so it is the same in a browser tab and inside
  Toolbox Launcher's frame, and the server can paint the very first frame in the right theme.
* serve.py puts `head_script()` into Streamlit's index.html: before Streamlit starts, it sets `data-theme` on
  <html> (which the custom CSS in ui.py follows) and tells Streamlit which of its two configured themes
  (.streamlit/config.toml [theme.light] / [theme.dark]) to start in. No flash of the other theme.
* Switching while the app is open: `apply_script()` runs in the page, flips `data-theme` and sends Streamlit its
  documented host message SET_CUSTOM_THEME_CONFIG, so widgets and tables follow without a reload (a reload would
  drop the user's uploaded images and corrections).

Exported reports, PDFs and figures are not affected: they keep their print-friendly look.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT = "dark"                               # the tool's original look; nothing changes until the user picks
SETTINGS = Path(os.environ.get("LAMIN_MESH_SETTINGS",
                               Path.home() / "Library" / "Application Support" / "Lamin Mesh Analyzer" / "settings.json"))

# Streamlit widget themes. Must match .streamlit/config.toml (checked by tests/test_theme.py).
STREAMLIT = {
    "dark":  {"primaryColor": "#22d3ee", "backgroundColor": "#0a0f1c", "secondaryBackgroundColor": "#16213a",
              "textColor": "#e8eefc"},
    "light": {"primaryColor": "#0e7490", "backgroundColor": "#eef1f6", "secondaryBackgroundColor": "#ffffff",
              "textColor": "#0f172a"},
}
PAGE_BG = {"dark": "#070b16", "light": "#eef1f6"}     # painted on <html> before anything else loads
_ST_NAME = {"dark": "Custom Theme Dark", "light": "Custom Theme Light"}
_ST_BASE = {"dark": 1, "light": 0}                    # CustomThemeConfig.BaseTheme: LIGHT = 0, DARK = 1


def load() -> str:
    try:
        value = json.loads(SETTINGS.read_text()).get("theme")
    except (OSError, ValueError, AttributeError):
        value = None
    return value if value in STREAMLIT else DEFAULT


def save(value: str) -> None:
    if value not in STREAMLIT:
        return
    try:
        data = json.loads(SETTINGS.read_text())
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    data["theme"] = value
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(SETTINGS)


def _streamlit_cache(value: str) -> dict:
    """What Streamlit keeps in localStorage for the active theme (restored by Streamlit at start-up)."""
    return {"name": _ST_NAME[value], "themeInput": {**STREAMLIT[value], "base": _ST_BASE[value]}}


def head_script(value: str | None = None) -> str:
    """<script> for the <head> of index.html: applies the saved theme before the first paint."""
    value = value if value in STREAMLIT else load()
    cache = json.dumps(_streamlit_cache(value))
    return ("<script>(function(){var t=%s;var d=document.documentElement;d.setAttribute('data-theme',t);"
            "d.style.colorScheme=t;d.style.background=%s;window.__laminThemeSent=t;"
            "try{localStorage.setItem('stActiveTheme-'+location.pathname+'-v1',%s);}catch(e){}})();</script>"
            % (json.dumps(value), json.dumps(PAGE_BG[value]), json.dumps(cache)))


def apply_script(value: str) -> str:
    """JavaScript for a component iframe: switch the open page to `value` (no-op if it is already there)."""
    info = {**STREAMLIT[value], "base": _ST_BASE[value]}
    return """
(() => {
  const t = %s, info = %s, cache = %s, bg = %s;
  try {
    const top = window.parent, d = top.document.documentElement;
    if (d.getAttribute('data-theme') !== t) {
      d.setAttribute('data-theme', t); d.style.colorScheme = t; d.style.background = bg;
    }
    if (top.__laminThemeSent !== t) {          // Streamlit's own widgets and tables follow (host message)
      top.__laminThemeSent = t;
      top.postMessage({stCommVersion: 1, type: 'SET_CUSTOM_THEME_CONFIG', themeName: '', themeInfo: info}, '*');
      setTimeout(() => {                         // remember it for the next start, under Streamlit's own name
        try { top.localStorage.setItem('stActiveTheme-' + top.location.pathname + '-v1', JSON.stringify(cache)); }
        catch (e) {}
      }, 600);
    }
  } catch (e) {}
})();
""" % (json.dumps(value), json.dumps(info), json.dumps(_streamlit_cache(value)), json.dumps(PAGE_BG[value]))


# ---------------------------------------------------------------------------------------------- chart colours
# Validated categorical steps for each surface (dark steps on dark, light steps on light), text and grid roles.
CHART = {
    "dark":  {"blue": "#3987e5", "orange": "#d95926", "aqua": "#199e70", "violet": "#9085e9",
              "ink": "#e8eefc", "ink2": "#9fb0d0", "grid": "rgba(159,176,208,.12)", "axis": "rgba(159,176,208,.25)",
              "hover_bg": "#111a2e", "hover_border": "#22d3ee", "fill": "rgba(57,135,229,.12)",
              "point_edge": "rgba(7,11,22,.9)"},
    "light": {"blue": "#2a78d6", "orange": "#c2410c", "aqua": "#0f7a55", "violet": "#4a3aa7",
              "ink": "#0f172a", "ink2": "#334155", "grid": "rgba(15,23,42,.08)", "axis": "rgba(15,23,42,.25)",
              "hover_bg": "#ffffff", "hover_border": "#0e7490", "fill": "rgba(42,120,214,.12)",
              "point_edge": "rgba(255,255,255,.95)"},
}
