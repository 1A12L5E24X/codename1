import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dotfx import dither as dz
from dotfx import halftone as ht
from dotfx.cli import main
from dotfx.color import parse_color, parse_palette
from dotfx.pipeline import render, to_uint8
from dotfx.presets import PRESETS, load_preset, to_toml
from dotfx.settings import SPOTS, Settings


def gradient(h=64, w=96):
    x = np.linspace(0, 1, w, dtype=np.float32)
    g = np.repeat(x[None, :], h, axis=0)
    rgb = np.stack([g, g * 0.8 + 0.1, 1 - g], axis=-1)
    return (rgb * 255).astype(np.uint8)


def test_parse_color():
    assert parse_color("#fff") == (1.0, 1.0, 1.0)
    assert parse_color("black") == (0.0, 0.0, 0.0)
    assert parse_color("255,0,0") == (1.0, 0.0, 0.0)
    assert parse_color("none") is None
    with pytest.raises(ValueError):
        parse_color("not-a-color")


def test_parse_palette():
    assert parse_palette("gameboy").shape == (4, 3)
    assert parse_palette("#000 #fff red").shape == (3, 3)
    assert parse_palette("#000,#fff").shape == (2, 3)
    with pytest.raises(ValueError):
        parse_palette("#000")


@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_bayer_is_a_permutation(n):
    m = dz.bayer(n)
    assert m.shape == (n, n)
    assert len(np.unique(m)) == n * n
    assert 0 < m.min() and m.max() < 1


def test_cluster_and_lines_maps_are_rankings():
    for m in (dz.cluster(8), dz.line_map(6)):
        assert 0 < m.min() and m.max() < 1


@pytest.mark.parametrize("spot", SPOTS)
def test_halftone_coverage_matches_tone(spot):
    scr = ht.Screen(240, 240, 10, 30)
    for t in (0.1, 0.3, 0.5, 0.7, 0.9):
        cov = scr.coverage(np.full((240, 240), t, np.float32), spot)
        assert abs(float(cov.mean()) - t) < 0.04, (spot, t)


def test_halftone_extremes_are_clean():
    scr = ht.Screen(80, 80, 8, 45)
    assert scr.coverage(np.zeros((80, 80), np.float32), "euclid").max() == 0
    assert scr.coverage(np.ones((80, 80), np.float32), "euclid").min() == 1


@pytest.mark.parametrize("algo", list(dz.KERNELS))
def test_error_diffusion_preserves_mean(algo):
    img = np.full((40, 40, 1), 0.3, np.float64)
    idx = dz.error_diffuse(img, np.array([[0.0], [1.0]]), algo)
    mean = idx.mean()
    # Atkinson deliberately drops 1/4 of the error, so it drifts more.
    assert abs(mean - 0.3) < (0.12 if algo == "atkinson" else 0.03)


def test_python_fallback_matches_reference_kernel():
    rng = np.random.default_rng(0)
    img = rng.random((12, 15, 3))
    pal = rng.random((5, 3))
    div, taps = dz.KERNELS["floyd-steinberg"]
    dx = np.array([t[0] for t in taps])
    dy = np.array([t[1] for t in taps])
    wt = np.array([t[2] / div for t in taps])
    a = np.zeros((12, 15), np.int64)
    b = np.zeros((12, 15), np.int64)
    dz._diffuse_py(img.copy(), pal, dx, dy, wt, True, 1.0, a)
    dz._diffuse_lists(img.copy(), pal, dx, dy, wt, True, 1.0, b)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("name", list(PRESETS))
def test_every_preset_renders(name):
    s = Settings().update(load_preset(name))
    out = render(gradient(), s)
    assert out.shape[:2] == (64, 96)
    assert np.isfinite(out).all()
    assert out.min() >= 0 and out.max() <= 1


def test_transparent_paper_gives_rgba():
    s = Settings().update({"paper": "none", "cell": 6})
    out = render(gradient(), s)
    assert out.shape[2] == 4
    alpha = out[..., 3]
    # Dark side gets ink (opaque), light side mostly stays clear.
    assert alpha[:, :10].mean() > alpha[:, -10:].mean()


def test_resize_and_cells():
    s = Settings().update({"width": 48, "cells": 12})
    out = render(gradient(), s)
    assert out.shape[:2] == (32, 48)


def test_validation():
    with pytest.raises(ValueError):
        render(gradient(), Settings().update({"effect": "halftone", "mode": "palette"}))
    with pytest.raises(KeyError):
        Settings().update({"nope": 1})


def test_toml_round_trip(tmp_path):
    s = Settings().update(load_preset("riso"))
    p = tmp_path / "look.toml"
    p.write_text(to_toml(s))
    again = Settings().update(load_preset(str(p)))
    assert again == s


def test_toml_base_preset(tmp_path):
    p = tmp_path / "mine.toml"
    p.write_text('base = "newsprint"\nink = "#ff0000"\n')
    d = load_preset(str(p))
    assert d["ink"] == "#ff0000" and d["spot"] == PRESETS["newsprint"][1]["spot"]


def test_cli_image(tmp_path, capsys):
    src = tmp_path / "in.png"
    Image.fromarray(gradient()).save(src)
    out = tmp_path / "out.jpg"
    assert main([str(src), "-p", "newsprint", "--cell", "5", "-o", str(out), "-q"]) == 0
    assert Image.open(out).size == (96, 64)
    # Refuses to overwrite without -y.
    assert main([str(src), "-o", str(out), "-q"]) == 0
    assert main([str(src), "-o", str(out), "-y", "--compare", "-q"]) == 0
    assert Image.open(out).size[0] > 96 * 2


def test_cli_batch_directory(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    for i in range(3):
        Image.fromarray(gradient()).save(d / f"{i}.png")
    outdir = tmp_path / "out"
    assert main([str(d), "-p", "bitmap", "-o", str(outdir), "-q"]) == 0
    assert sorted(p.name for p in outdir.iterdir()) == ["0.bitmap.png", "1.bitmap.png", "2.bitmap.png"]


def test_cli_errors(tmp_path, capsys):
    assert main([str(tmp_path / "missing.png")]) == 2
    assert main(["presets", "newsprint"]) == 0
    assert "spot" in capsys.readouterr().out


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_cli_video(tmp_path):
    src = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=10:duration=1",
         "-pix_fmt", "yuv420p", str(src)],
        check=True,
    )
    out = tmp_path / "clip.out.mp4"
    assert main([str(src), "-p", "comic", "-o", str(out), "-q", "-j", "2"]) == 0
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_entries", "stream=nb_read_frames,width",
         "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert probe == "96,10"


def test_module_entry_point():
    r = subprocess.run([sys.executable, "-m", "dotfx", "--version"], capture_output=True, text=True,
                       cwd=Path(__file__).resolve().parents[1])
    assert r.returncode == 0 and "dotfx" in r.stdout
