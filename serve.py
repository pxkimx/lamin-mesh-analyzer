"""
Start the Lamin Mesh Analyzer server:  python serve.py [--port 8599] [--headless]

Streamlit runs inside this process (not `python -m streamlit run`), so that stopping this process really
stops the app, and so that the watchers that make it quit when its window is closed (lifecycle.py) are
running before any browser connects.
"""
import argparse
import datetime
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 8599


def _serve_saved_theme(port):
    """Paint the first frame in the saved Light / Dark theme (theme.py), and let the page switch it live.

    * index.html gets theme.head_script() in its <head>, read from the settings file on every page load.
    * The page's own address is added to the origins Streamlit accepts host messages from, so the in-page
      switch can send Streamlit SET_CUSTOM_THEME_CONFIG (its documented embedding message) without a reload.
    """
    import theme
    from streamlit.web.server import routes

    for origin in (f"http://localhost:{port}", f"http://127.0.0.1:{port}"):
        if origin not in routes._DEFAULT_ALLOWED_MESSAGE_ORIGINS:
            routes._DEFAULT_ALLOWED_MESSAGE_ORIGINS.append(origin)

    handler = routes.StaticFileHandler
    original_content = handler.get_content.__func__
    original_size = handler.get_content_size

    def page(abspath):
        with open(abspath, "rb") as f:
            html = f.read()
        return html.replace(b"<head>", b"<head>" + theme.head_script().encode(), 1)

    def get_content(cls, abspath, start=None, end=None):
        if os.path.basename(abspath) == "index.html":
            data = page(abspath)
            return data[start:end]
        return original_content(cls, abspath, start, end)

    def get_content_size(self):
        if os.path.basename(self.absolute_path) == "index.html":
            return len(page(self.absolute_path))
        return original_size(self)

    original_modified = handler.get_modified_time

    def get_modified_time(self):                           # the page changes with the theme, not with the file,
        if os.path.basename(self.absolute_path) == "index.html":   # so never answer "not modified" by date
            return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
        return original_modified(self)

    handler.get_content = classmethod(get_content)
    handler.get_content_size = get_content_size
    handler.get_modified_time = get_modified_time


def main(argv=None):
    ap = argparse.ArgumentParser(description="Lamin Mesh Analyzer server")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--headless", action="store_true", help="do not open the browser")
    args = ap.parse_args(argv)

    sys.path.insert(0, HERE)
    os.chdir(HERE)                                         # .streamlit/config.toml (both themes) is read from here
    import lifecycle
    from streamlit.web import cli as streamlit_cli

    lifecycle.start()
    try:                                                   # relies on Streamlit internals: if an update changes them,
        _serve_saved_theme(args.port)                      # the app still starts (Streamlit picks the theme itself)
    except Exception as exc:
        print(f"theme: first-paint / live switch unavailable with this Streamlit ({exc!r})", file=sys.stderr)
    sys.argv = ["streamlit", "run", os.path.join(HERE, "app.py"),
                "--server.port", str(args.port), "--server.address", "localhost",
                "--server.headless", "true" if args.headless else "false",
                "--browser.gatherUsageStats", "false", "--server.maxUploadSize", "2000"]
    return streamlit_cli.main()


if __name__ == "__main__":
    sys.exit(main())
