"""Regression test: the example image must keep giving the same measurements with the app's default settings.

If one of these fails after a code change, the measurements changed. That may be intended (an improved
method); then check the new numbers with the user before updating them here.  Run:  python -m pytest tests -q
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import lamin_core as L  # noqa: E402

EXAMPLE = os.path.join(ROOT, "example", "lamin_example.tif")          # the author's own image: not in the public repo
SYNTHETIC = os.path.join(ROOT, "example", "synthetic_mesh.tif")       # example/make_synthetic_example.py


@pytest.fixture(scope="module")
def image():
    if not os.path.exists(EXAMPLE):
        pytest.skip("example/lamin_example.tif is the author's own image and is not distributed")
    with open(EXAMPLE, "rb") as f:
        return L.load_image(f.read(), "lamin_example.tif")


@pytest.fixture(scope="module")
def synthetic():
    with open(SYNTHETIC, "rb") as f:
        return L.load_image(f.read(), "synthetic_mesh.tif")


def analyze(img, **kw):
    # the app's defaults (app.py params_for): channel picked by the loader, rim 25 px, min face 20 px²
    p = L.Params(channel=img.default_channel, edge_margin=25, min_area_px=20, **kw)
    return L.analyze(img, p).stats


def test_meshwork_tracing_clean(image):                    # Image type "Clean"
    s = analyze(image, method="trace", sensitivity=1.0, edge_quality=1.0)
    assert s["Faces (n)"] == 525
    assert s["Median area (px²)"] == pytest.approx(451, abs=1)


def test_meshwork_tracing_noisy(image):                    # Image type "Noisy / 3D-SIM"
    s = analyze(image, method="trace", sensitivity=0.6, edge_quality=0.0)
    assert s["Faces (n)"] == 1124
    assert s["Median area (px²)"] == pytest.approx(207, abs=1)


def test_black_and_white(image):
    s = analyze(image, method="threshold", smooth_sigma=1.0, max_puncta_px=30)
    assert s["Faces (n)"] == 378
    assert s["Median area (px²)"] == pytest.approx(191, abs=1)


def test_synthetic_example(synthetic):                     # runs everywhere, including a fresh clone
    s = analyze(synthetic, method="trace", sensitivity=1.0, edge_quality=1.0)
    assert s["Faces (n)"] == 701
    assert s["Median area (px²)"] == pytest.approx(234, abs=1)
    s = analyze(synthetic, method="threshold", smooth_sigma=1.0, max_puncta_px=30)
    assert s["Faces (n)"] == 744


def test_synthetic_example_is_reproducible(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("mk", os.path.join(ROOT, "example", "make_synthetic_example.py"))
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    import numpy as np
    import tifffile
    assert np.array_equal(mk.make(), tifffile.imread(SYNTHETIC))
