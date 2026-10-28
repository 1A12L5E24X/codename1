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
    scr = ht.Screen(240, 240, 10, 30, spot=spot)
    for t in (0.1, 0.3, 0.5, 0.7, 0.9):
        cov = scr.coverage(np.full((240, 240), t, np.float32))
        assert abs(float(cov.mean()) - t) < 0.04, (spot, t)


def test_halftone_extremes_are_clean():
    scr = ht.Screen(80, 80, 8, 45)
    assert scr.coverage(np.zeros((80, 80), np.float32)).max() == 0
    assert scr.coverage(np.ones((80, 80), np.float32)).min() == 1


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
    bias = rng.random((12, 15)) * 0.2 - 0.1
    dz._diffuse_py(img.copy(), pal, dx, dy, wt, True, 1.0, a, bias)
    dz._diffuse_lists(img.copy(), pal, dx, dy, wt, True, 1.0, b, bias)
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


def test_bad_colors_fail_validation():
    with pytest.raises(ValueError, match="ink"):
        Settings().update({"ink": "none"}).validate()
    with pytest.raises(ValueError, match="paper"):
        Settings().update({"paper": "notacolor"}).validate()
    Settings().update({"paper": "none"}).validate()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_video_color_dither_uses_one_palette(tmp_path):
    src = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=10:duration=0.5",
         "-pix_fmt", "yuv420p", str(src)],
        check=True,
    )
    out = tmp_path / "frames" / "%03d.png"
    assert main([str(src), "-e", "dither", "-m", "color", "--colors", "4", "--pixel", "1",
                 "-o", str(out), "-q", "-j", "1"]) == 0
    colors = set()
    for f in sorted((tmp_path / "frames").iterdir()):
        colors |= {tuple(c) for c in np.asarray(Image.open(f).convert("RGB")).reshape(-1, 3)}
    assert len(colors) <= 4


def flat(rgb, h=48, w=64):
    return np.full((h, w, 3), np.array(rgb) * 255, np.uint8)


def test_fractional_dither_pixel():
    s = Settings().update({"effect": "dither", "pixel": 3.5, "algorithm": "bayer4"})
    out = render(gradient(), s)
    assert out.shape[:2] == (64, 96)
    # Tone survives the enlarge-render-shrink round trip.
    lum = out[..., :3].mean(axis=(0, 2))
    assert lum[:10].mean() < 0.35 and lum[-10:].mean() > 0.65


def test_cells_allow_fractional_dither_pixels():
    s = Settings().update({"effect": "dither", "cells": 27})
    assert render(gradient(), s).shape[:2] == (64, 96)


def test_tint_by_saturation_only_tints_colorful_areas():
    base = {"effect": "dither", "pixel": 2, "ink": "#000000", "paper": "#ffffff", "tint": "#ff0000",
            "tint_by": "saturation", "tint_start": 0.2, "tint_end": 0.4}
    gray = render(flat((0.5, 0.5, 0.5)), Settings().update(base))
    blue = render(flat((0.2, 0.4, 0.9)), Settings().update(base))
    redness = lambda o: float((o[..., 0] - o[..., 1]).mean())  # noqa: E731
    assert redness(gray) < 0.02
    assert redness(blue) > 0.2


def test_halftone_tint_by_tone():
    s = Settings().update({"cell": 6, "tint": "#00ff00", "tint_start": 0.3, "tint_end": 0.6})
    out = render(gradient(), s)
    green = out[..., 1] - out[..., 0]
    # Tint shows on the dark (left) side, not the light side.
    assert green[:, :20].mean() > green[:, -20:].mean() + 0.05


def test_jpeg_setting():
    s = Settings().update({"jpeg": 40, "paper": "none", "cell": 6})
    out = render(gradient(), s)
    assert out.shape == (64, 96, 4)
    with pytest.raises(ValueError):
        Settings().update({"jpeg": 101}).validate()


def test_ink_limit_keeps_paper_dominant():
    black = flat((0.0, 0.0, 0.0))
    full = render(black, Settings().update({"cell": 6, "ink": "#000000"}))
    capped = render(black, Settings().update({"cell": 6, "ink": "#000000", "ink_limit": 0.3}))
    assert full[..., :3].mean() < 0.05
    assert abs(float(capped[..., :3].mean()) - 0.7) < 0.05


# --- match ------------------------------------------------------------------------


def photo(w=360, h=300):
    """A smooth synthetic 'photo' with tones and colors across the frame."""
    y, x = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    r = 0.5 + 0.45 * np.sin(6 * x + 2 * y)
    g = 0.5 + 0.45 * np.sin(4 * y - 3 * x + 1)
    b = 0.5 + 0.45 * np.cos(5 * x * y + 2)
    return (np.clip(np.stack([r, g, b], -1), 0, 1) * 255).astype(np.uint8)


def test_match_recovers_a_halftone():
    from dotfx.match import match

    out = render(photo(), Settings().update({"cells": 40, "angle": 30, "spot": "dot", "ink": "#c02020"}))
    m = match(out[..., :3]).settings
    assert m["effect"] == "halftone" and m["mode"] == "mono"
    assert abs(m["cells"] - 40) <= 1 and abs(m["angle"] - 30) < 2
    assert parse_color(m["ink"])[0] > 0.6


def test_match_recovers_a_bayer_dither_with_tint():
    from dotfx.match import match

    s = {"effect": "dither", "algorithm": "bayer4", "pixel": 4, "cell_shape": "circle",
         "ink": "#203080", "tint": "#a0c0ff", "tint_start": 0.3, "tint_end": 0.5}
    m = match(render(photo(), Settings().update(s))[..., :3]).settings
    assert m["effect"] == "dither" and m["algorithm"] == "bayer4"
    assert abs(m["cells"] - 90) <= 2
    assert "tint" in m


def test_match_detects_cmyk_and_jpeg():
    from dotfx.match import match, measure_jpeg

    out = render(photo(), Settings().update({"mode": "cmyk", "cells": 45}))
    assert match(out[..., :3]).settings["mode"] == "cmyk"
    # A soft, scanned-looking print (crisp black/white clips JPEG ringing, which
    # hides the quantization; the matcher then reports "can't tell").
    scan = {"cells": 40, "ink": "#333333", "paper": "#f6f6f5", "post_blur": 1.0}
    q = measure_jpeg(render(photo(), Settings().update({**scan, "jpeg": 70}))[..., :3])
    assert q is not None and 60 <= q <= 80
    clean = Settings().update(scan)
    assert measure_jpeg(render(photo(), clean)[..., :3]) == 0


def test_cli_match_writes_a_usable_preset(tmp_path):
    ref = tmp_path / "ref.png"
    Image.fromarray(to_uint8(render(photo(), Settings().update({"cells": 40})))).save(ref)
    out = tmp_path / "look.toml"
    check = tmp_path / "check.png"
    assert main(["match", str(ref), "-o", str(out), "--check", str(check)]) == 0
    s = Settings().update(load_preset(str(out)))
    assert s.effect == "halftone" and s.cells == 40
    assert Image.open(check).size[0] > 2 * 360


# --- terminal UI --------------------------------------------------------------------


def test_fit_pads_and_cuts_styled_text():
    from dotfx.term import fit, visible_len

    assert visible_len(fit("\x1b[1mhi\x1b[0m", 5)) == 5
    cut = fit("\x1b[1mhello world\x1b[0m", 6)
    assert visible_len(cut) == 6 and cut.endswith("…\x1b[0m")


def test_kitty_image_round_trips():
    import base64
    import io
    import re

    from dotfx.term import kitty_place

    img = np.zeros((40, 60, 3), np.uint8)
    img[:, 30:] = 255
    codes = kitty_place(img, 10, 4)
    parts = re.findall(r"\x1b_G([^;\x1b]*)(?:;([^\x1b]*))?\x1b\\", codes)
    assert "a=d" in parts[0][0] and "a=T" in parts[1][0] and "c=10,r=4" in parts[1][0]
    data = "".join(p[1] for p in parts[1:])
    assert Image.open(io.BytesIO(base64.b64decode(data))).size == (60, 40)


def test_tuner_draws_without_a_terminal(tmp_path, monkeypatch, capsys):
    from dotfx.tune import Tuner

    monkeypatch.setattr("dotfx.term.size", lambda: (120, 36))
    t = Tuner(gradient(), Settings().update(load_preset("newsprint")), tmp_path / "x.png", "newsprint",
              tmp_path / "o.png", tmp_path / "p.toml", preview_mode="blocks")
    t.out = (render(gradient(), t.s) * 255).astype(np.uint8)
    t.draw()
    t.help = True
    t.draw()
    out = capsys.readouterr().out
    assert "settings" in out and "preview" in out and "newsprint" in out and "keys" in out
    t.sel = 0
    t.adjust(1)  # effect: halftone -> dither
    assert t.s.effect == "dither"
    assert "-e dither" in t.command_line() or "--effect dither" in t.command_line()


def test_palette_size_by_tone_uses_one_tone_per_cell():
    s = Settings().update({"effect": "dither", "mode": "palette", "palette": "gameboy", "pixel": 4,
                           "size_by_tone": True, "cell_shape": "circle"})
    out = render(gradient(), s)
    assert out.shape == (64, 96, 3) and np.isfinite(out).all()


# --- vector output ------------------------------------------------------------------


def test_vector_dot_area_is_tone_accurate():
    from dotfx.vector import _clipped_circle_area, diamond_radius, dot_radius

    for t in (0.05, 0.3, 0.5, 0.78, 0.9, 0.99):
        assert abs(float(_clipped_circle_area(np.array([dot_radius(np.array(t))]))[0]) - t) < 2e-3
        r = float(diamond_radius(np.array(t)))
        area = 2 * r * r if r <= 0.5 else 1 - 2 * (1 - r) ** 2
        assert abs(area - t) < 1e-6


def test_svg_output(tmp_path):
    import xml.etree.ElementTree as ET

    src = tmp_path / "in.png"
    Image.fromarray(photo()).save(src)
    out = tmp_path / "out.svg"
    assert main([str(src), "-p", "newsprint", "-o", str(out), "-q"]) == 0
    root = ET.parse(out).getroot()
    assert root.get("width") == "360" and root.get("height") == "300"
    circles = root.findall(".//{http://www.w3.org/2000/svg}circle")
    assert len(circles) > 500
    out2 = tmp_path / "dither.svg"
    assert main([str(src), "-p", "gameboy", "-o", str(out2), "-q"]) == 0
    assert len(ET.parse(out2).getroot().findall(".//{http://www.w3.org/2000/svg}rect")) > 50


def test_pdf_output_is_well_formed(tmp_path):
    import re

    src = tmp_path / "in.png"
    Image.fromarray(photo()).save(src)
    out = tmp_path / "out.pdf"
    assert main([str(src), "-p", "comic", "-o", str(out), "-q", "--dpi", "72"]) == 0
    data = out.read_bytes()
    assert data.startswith(b"%PDF-1.4") and data.rstrip().endswith(b"%%EOF")
    assert b"/MediaBox [0 0 360 300]" in data
    # Every xref offset must point at its object header.
    xref = int(re.search(rb"startxref\n(\d+)", data).group(1))
    offsets = [int(m) for m in re.findall(rb"(\d{10}) 00000 n", data[xref:])]
    for i, off in enumerate(offsets, start=1):
        assert data[off:].startswith(f"{i} 0 obj".encode())
    if shutil.which("pdftoppm"):
        r = subprocess.run(["pdftoppm", "-png", "-r", "72", "-singlefile", str(out), str(tmp_path / "r")],
                           capture_output=True)
        assert r.returncode == 0 and Image.open(tmp_path / "r.png").size == (360, 300)


def test_temporal_anchoring_steadies_error_diffusion():
    rng = np.random.default_rng(0)
    base = photo().astype(np.float32)
    frames = [np.clip(base + rng.normal(0, 1.5, base.shape), 0, 255).astype(np.uint8) for _ in range(4)]

    def flips(temporal):
        s = Settings().update({"effect": "dither", "pixel": 2, "algorithm": "floyd-steinberg", "temporal": temporal})
        outs = [render(f, s)[..., 0] > 0.5 for f in frames]
        return np.mean([np.mean(a != b) for a, b in zip(outs, outs[1:])]), outs

    loose, outs0 = flips(0)
    steady, outs1 = flips(1.5)
    assert steady < loose / 3
    # Same overall tone either way.
    assert abs(outs0[0].mean() - outs1[0].mean()) < 0.02


@pytest.mark.parametrize("edge", ["cut", "torn", "wavy"])
def test_clipping_edges(edge):
    s = Settings().update({"cell": 6, "clip_edge": edge, "clip_margin": 0.1, "backdrop": "#ff0000",
                           "clip_shadow": 1.0})
    out = render(photo(), s)
    assert out.shape == (300, 360, 3)
    # Corners show the backdrop, the centre shows the print.
    assert out[2, 2, 0] > 0.6 and out[2, 2, 1] < 0.3
    assert abs(out[150, 180, 0] - out[150, 180, 1]) < 0.05
    # Transparent backdrop gives an alpha channel with a soft shadow.
    t = render(photo(), Settings().update({"cell": 6, "clip_edge": edge, "backdrop": "none"}))
    assert t.shape[2] == 4 and t[0, 0, 3] < 0.2 and t[150, 180, 3] > 0.99


# --- masks, live, windows -----------------------------------------------------------


@pytest.mark.parametrize("mask", ["bright", "dark", "colorful", "center"])
def test_masks_split_the_image(mask):
    s = Settings().update({"cell": 6, "mask": mask, "background": "paper", "paper": "#ff0000",
                           "mask_feather": 0})
    out = render(photo(), s)
    red = (out[..., 0] > 0.95) & (out[..., 1] < 0.05)
    # Some of the image got the look, some got the (red) paper background.
    assert 0.02 < red.mean() < 0.98


def test_mask_file_and_preset_background(tmp_path):
    m = np.zeros((300, 360), np.uint8)
    m[:, :180] = 255
    path = tmp_path / "m.png"
    Image.fromarray(m).save(path)
    s = Settings().update({"cell": 6, "ink": "#000000", "mask": str(path), "background": "gameboy",
                           "mask_feather": 0})
    out = render(photo(), s)
    left, right = out[:, :170], out[:, 190:]
    sat = lambda a: (a.max(-1) - a.min(-1)).mean()  # noqa: E731
    assert sat(left) < 0.02 and sat(right) > 0.1  # gray halftone left, green gameboy right
    with pytest.raises(ValueError):
        render(photo(), Settings().update({"mask": str(tmp_path / "missing.png")}))


def test_camera_inputs(monkeypatch):
    from dotfx import live

    monkeypatch.setattr(live.sys, "platform", "linux")
    assert live.camera_input("") == ["-f", "v4l2", "-i", "/dev/video0"]
    assert live.camera_input("/dev/video2")[-1] == "/dev/video2"
    monkeypatch.setattr(live.sys, "platform", "darwin")
    assert live.camera_input("1")[-1] == "1:none"
    monkeypatch.setattr(live.sys, "platform", "win32")
    assert live.camera_input("Logitech C920")[-1] == "video=Logitech C920"
    assert live.is_camera("") and live.is_camera("0") and not live.is_camera("clip.mp4")


def test_windows_key_decoding(monkeypatch):
    import types

    from dotfx import term

    queue = list("\xe0H" "\xe0M" "q" "\r")
    fake = types.SimpleNamespace(kbhit=lambda: bool(queue), getwch=lambda: queue.pop(0))
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    monkeypatch.setattr(term, "WINDOWS", True)
    kb = term.Keyboard()
    assert [kb.read(0.1) for _ in range(4)] == ["up", "right", "q", "enter"]
    assert kb.read(0.05) == ""


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_live_capture_reads_a_file(tmp_path):
    from dotfx.live import Capture

    src = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=10:duration=2",
                    "-pix_fmt", "yuv420p", str(src)], check=True)
    cap = Capture(str(src), 80, 10)
    try:
        import time

        for _ in range(50):
            if cap.frame is not None:
                break
            time.sleep(0.05)
        assert cap.frame is not None and cap.frame.shape == (60, 80, 3)
    finally:
        cap.close()


def test_live_block_preview_keeps_dots_visible():
    import types

    from dotfx.live import LiveUI

    cap = types.SimpleNamespace(frame=None, count=0, size=(640, 480), error="")
    ui_ = LiveUI(cap, "", Settings().update(load_preset("newsprint")), "newsprint", 15, Path("."), "blocks")
    s, enlarged = ui_.preview_settings(96, 0.15)  # 80 dots across a 96-pixel-wide preview
    assert 96 / s.cells >= 4 and enlarged > 1
    ui_.set_preset("bitmap")  # pixel-sized dither
    s, enlarged = ui_.preview_settings(96, 0.15)
    assert s.pixel >= 2


def test_home_screen(monkeypatch, capsys):
    from dotfx import home

    monkeypatch.setattr(home, "size", lambda: (100, 40))
    home.show()
    out = capsys.readouterr().out
    assert "COMMANDS" in out and "match" in out and ("▀" in out or "█" in out)


# --- browser app --------------------------------------------------------------------


@pytest.fixture
def app_server(tmp_path):
    from dotfx.app import App

    app = App(workdir=tmp_path).start()
    base = f"http://127.0.0.1:{app.httpd.server_address[1]}"
    yield app, base
    app.stop()


def _call(base, token, path, body=None, raw=None, headers=None):
    import json
    import urllib.request

    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(base + path, data=data, headers={"X-Dotfx-Token": token, **(headers or {})},
                                 method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read(), dict(r.headers)


def test_app_rejects_requests_without_the_token(app_server):
    import urllib.error

    app, base = app_server
    with pytest.raises(urllib.error.HTTPError) as e:
        _call(base, "nope", "/api/schema")
    assert e.value.code == 403
    page, _ = _call(base, app.state.token, f"/?t={app.state.token}")
    assert b"<title>dotfx</title>" in page


def test_app_render_export_and_match(app_server, tmp_path):
    import io
    import json

    app, base = app_server
    tok = app.state.token
    schema = json.loads(_call(base, tok, "/api/schema")[0])
    assert any(f["name"] == "cells" for f in schema["fields"])
    buf = io.BytesIO()
    Image.fromarray(photo()).save(buf, format="PNG")
    media = json.loads(_call(base, tok, "/api/upload?name=p.png", raw=buf.getvalue())[0])
    assert media["width"] == 360 and media["kind"] == "image"
    png, headers = _call(base, tok, "/api/render", {"media": media["id"], "settings": {"cells": 40}})
    assert Image.open(io.BytesIO(png)).size == (360, 300) and "X-Render-Ms" in headers
    svg, _ = _call(base, tok, "/api/export", {"media": media["id"], "settings": {"cells": 40}, "format": "svg"})
    assert svg.startswith(b"<svg")
    # Match the rendered print back.
    out = to_uint8(render(photo(), Settings().update({"cells": 40, "angle": 30})))
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG")
    m = json.loads(_call(base, tok, "/api/match?name=ref.png", raw=buf.getvalue())[0])
    assert m["settings"]["cells"] == 40 and m["notes"]
    # Webcam frames.
    buf = io.BytesIO()
    Image.fromarray(photo()).save(buf, format="JPEG")
    jpg, _ = _call(base, tok, "/api/frame", raw=buf.getvalue(), headers={"X-Settings": json.dumps({"cells": 30})})
    assert Image.open(io.BytesIO(jpg)).size == (360, 300)


def test_app_handoff_state(app_server, tmp_path):
    import json

    app, base = app_server
    src = tmp_path / "x.png"
    Image.fromarray(photo()).save(src)
    url = app.open_with(src, Settings().update(load_preset("riso")), "riso")
    assert url.startswith(base) and app.state.token in url
    state = json.loads(_call(base, app.state.token, "/api/state")[0])
    assert state["media"]["name"] == "x.png" and state["preset"] == "riso"
    assert state["settings"]["ink"] == "riso-pink"


def test_mouse_reports_parse():
    from dotfx.term import parse_mouse

    assert parse_mouse("\x1b[<0;12;40M\x1b[<0;12;40m") == "click:12:40"
    assert parse_mouse("\x1b[<64;5;6M") == "wheelup:5:6"
    assert parse_mouse("\x1b[<0;1;1m") == ""
