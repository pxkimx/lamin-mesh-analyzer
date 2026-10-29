"""Light / dark theme: saved choice, first paint, Streamlit config in step, readable text.  Run: python -m pytest tests -q"""
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import tomllib
import urllib.request

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import theme  # noqa: E402
import ui  # noqa: E402


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_default_is_the_original_dark_look(tmp_path, monkeypatch):
    monkeypatch.setattr(theme, "SETTINGS", tmp_path / "missing.json")
    assert theme.load() == "dark"


def test_choice_is_saved_and_kept_with_other_settings(tmp_path, monkeypatch):
    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"other": 1}))
    monkeypatch.setattr(theme, "SETTINGS", f)
    theme.save("light")
    assert theme.load() == "light"
    assert json.loads(f.read_text()) == {"other": 1, "theme": "light"}
    theme.save("purple")                                   # unknown values are ignored
    assert theme.load() == "light"
    f.write_text("not json")
    assert theme.load() == "dark"


def test_streamlit_config_matches_theme_module():
    with open(os.path.join(ROOT, ".streamlit", "config.toml"), "rb") as fh:
        cfg = tomllib.load(fh)["theme"]
    for name in ("light", "dark"):
        assert cfg[name] == theme.STREAMLIT[name]


def test_head_script_sets_theme_before_streamlit_starts():
    for t in ("light", "dark"):
        js = theme.head_script(t)
        assert f"var t=\"{t}\"" in js and "data-theme" in js and "stActiveTheme-" in js
        name = "Custom Theme Light" if t == "light" else "Custom Theme Dark"
        assert name in js and theme.PAGE_BG[t] in js


def _lum(hex_):
    r, g, b = (int(hex_[i:i + 2], 16) / 255 for i in (1, 3, 5))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def contrast(a, b):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _vars(block):
    return dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})\b", block))


def test_text_contrast_at_least_4_5_in_both_themes():
    css = ui.CSS
    dark = _vars(css.split(":root{", 1)[1].split("}", 1)[0])
    light = _vars(css.split('html[data-theme="light"]{', 1)[1].split("}", 1)[0])
    surfaces = {"dark": ["#070b16", "#16213a", "#0a0f1c"], "light": ["#eef1f6", "#ffffff"]}
    for name, v in (("dark", dark), ("light", light)):
        for key in ("ink", "ink2", "muted", "value-ink", "link", "chip-ink", "sec-ic-ink"):
            for bg in surfaces[name]:
                assert contrast(v[key], bg) >= 4.5, (name, key, v[key], bg)
    for name, c in theme.CHART.items():                    # chart text on the page
        for key in ("ink", "ink2"):
            assert contrast(c[key], theme.PAGE_BG[name]) >= 4.5, (name, key)
    for name, s in theme.STREAMLIT.items():                # Streamlit's own widgets
        assert contrast(s["textColor"], s["backgroundColor"]) >= 4.5
        assert contrast(s["textColor"], s["secondaryBackgroundColor"]) >= 4.5


def test_first_page_carries_the_saved_theme(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"theme": "light"}))
    port = free_port()
    env = dict(os.environ, LAMIN_MESH_SETTINGS=str(settings))
    server = subprocess.Popen([sys.executable, "-W", "ignore", os.path.join(ROOT, "serve.py"), "--port", str(port),
                               "--headless"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def page():
        return urllib.request.urlopen(f"http://localhost:{port}/", timeout=2).read().decode()

    try:
        end = time.time() + 60
        while True:
            try:
                html = page()
                break
            except OSError:
                if time.time() > end:
                    raise AssertionError("server did not start")
                time.sleep(0.5)
        head = html.split("</head>", 1)[0]
        assert 'var t="light"' in head and head.index("data-theme") < head.index("index.")   # before Streamlit's JS
        settings.write_text(json.dumps({"theme": "dark"}))                                   # the user switches
        assert 'var t="dark"' in page()
    finally:
        server.send_signal(signal.SIGTERM)
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()


def test_app_still_starts_if_streamlit_internals_change(monkeypatch):
    """The theme patch uses Streamlit internals; a Streamlit update must not stop the app from starting."""
    import serve
    from streamlit.web.server import routes
    monkeypatch.delattr(routes, "_DEFAULT_ALLOWED_MESSAGE_ORIGINS")      # as if an update renamed it
    started = {}
    monkeypatch.setattr("streamlit.web.cli.main", lambda: started.setdefault("ok", True))
    monkeypatch.setattr("lifecycle.start", lambda: "http://127.0.0.1:1/closing")
    cwd = os.getcwd()
    try:
        serve.main(["--port", str(free_port()), "--headless"])
    finally:
        os.chdir(cwd)
    assert started.get("ok")
