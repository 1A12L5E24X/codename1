"""Measure a reference print and turn it into a preset.

The dot grid is found in the frequency domain. A halftone screen puts all of
its spectral energy on one lattice (any angle). An ordered dither makes
different patterns at different tones, so strong peaks also appear off that
lattice, and only at 0 or 45 degrees. Colors, edge softness, paper grain and
JPEG blockiness are measured directly from the pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ops import LUMA, gaussian_blur


# Paper noise (std of a 3x3 high-pass) produced per unit of the grain setting.
GRAIN_PER_UNIT = 0.03


@dataclass
class Match:
    settings: dict
    notes: list[str] = field(default_factory=list)
    measured: dict = field(default_factory=dict)


# --- spectrum -------------------------------------------------------------------


def _peaks(gray: np.ndarray, n: int = 16, pad: int = 2048) -> list[tuple[float, float, float]]:
    """Strongest spectral peaks (fx, fy, magnitude) in cycles/pixel, half plane."""
    h, w = gray.shape
    s = min(max(h, w), 1024)
    # Analyse a centred crop: enough periods for precision without huge FFTs.
    y0, x0 = max((h - s) // 2, 0), max((w - s) // 2, 0)
    g = gray[y0:y0 + s, x0:x0 + s].astype(np.float64)
    g = g - g.mean()
    win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))

    spec = np.abs(np.fft.fft2(g * win, s=(pad, pad)))
    fr = np.fft.fftfreq(pad)
    fx, fy = np.meshgrid(fr, fr)
    r = np.hypot(fx, fy)
    # Ignore image content (low frequencies) and the mirrored half plane.
    spec[(r < 0.025) | (fx < 0) | ((fx == 0) & (fy < 0))] = 0
    out = []
    for _ in range(n):
        i = np.unravel_index(spec.argmax(), spec.shape)
        m = spec[i]
        if m <= 0:
            break
        px, py = fx[i], fy[i]
        out.append((float(px), float(py), float(m)))
        spec[np.hypot(fx - px, fy - py) < 0.012] = 0
        spec[np.hypot(fx + px, fy + py) < 0.012] = 0
    return out


def _on_lattice(f, a, b, tol) -> bool:
    """Is frequency f an integer combination of basis vectors a and b?"""
    m = np.array([[a[0], b[0]], [a[1], b[1]]])
    try:
        c = np.linalg.solve(m, np.array(f))
    except np.linalg.LinAlgError:
        return False
    near = m @ np.round(c)
    return float(np.hypot(*(np.array(f) - near))) < tol


def analyse_grid(gray: np.ndarray) -> dict:
    """Classify the pattern and measure its spacing.

    In a halftone the screen's base frequency is the strongest peak. An ordered
    dither instead puts its strongest energy in the 45 degree checkerboard
    (frequency 1/(2 * cell) on both axes) and spreads the rest over many
    tone-dependent patterns, so its lowest-frequency peak is comparatively weak.
    """
    peaks = _peaks(gray)
    if not peaks:
        return {"kind": "none"}
    top = max(p[2] for p in peaks)
    strong = [p for p in peaks if p[2] > 0.2 * top]
    lowest = min(strong, key=lambda p: np.hypot(p[0], p[1]))
    if lowest[2] >= 0.8 * top:
        period = 1 / np.hypot(lowest[0], lowest[1])
        angle = float(np.degrees(np.arctan2(lowest[1], lowest[0])) % 90)
        return {"kind": "halftone", "period": float(period), "angle": angle, "peaks": len(strong)}
    big = [p for p in strong if p[2] > 0.75 * top]
    diag = [p for p in big if abs(abs(p[0]) - abs(p[1])) < 0.1 * max(abs(p[0]), abs(p[1]))]
    algorithm = "bayer4"
    if diag:
        f = max(max(abs(p[0]), abs(p[1])) for p in diag)
        pixel = 1 / (2 * f)
    else:
        # No checkerboard: an unordered (blue noise / diffusion) dither, whose
        # only regular structure is the cell grid itself at frequency 1/cell.
        f = max(max(abs(p[0]), abs(p[1])) for p in big)
        pixel = 1 / f
        algorithm = "bluenoise"
    # Other checkerboards at a clearly different spacing: resized or stitched source.
    others = [p for p in strong if p[2] > 0.3 * top and abs(abs(p[0]) - abs(p[1])) < 0.1 * abs(p[0] or 1)
              and 0.08 < abs(abs(p[0]) - f) / f < 0.3]
    return {"kind": "dither", "pixel": float(pixel), "algorithm": algorithm, "peaks": len(strong),
            "uneven": bool(others)}


# --- colors and texture -----------------------------------------------------------


def _local_max(lum: np.ndarray, r: int = 2) -> np.ndarray:
    pad = np.pad(lum, r, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, (2 * r + 1, 2 * r + 1))
    return lum >= win.max(axis=(-1, -2))


def _sat(rgb: np.ndarray) -> np.ndarray:
    mx = rgb.max(axis=-1)
    return (mx - rgb.min(axis=-1)) / np.maximum(mx, 1e-4)


def measure_colors(img: np.ndarray) -> dict:
    lum = img @ LUMA
    lo, hi = np.percentile(lum, [2, 98])
    dark = np.median(img[lum <= lo], axis=0)
    light = np.median(img[lum >= hi], axis=0)
    # For a single ink, light-dots-on-dark and dark-dots-on-light render the
    # same image, so assume light paper. The exception: dots in many colors on
    # a near-black background, which is the "dots carry the photo's colors" look.
    ink, paper = dark, light
    out = {"ink": ink, "paper": paper, "dark_paper": False, "light": light, "dark": dark}
    if float(dark.max()) < 0.12:
        bright = lum > 0.3
        if bright.mean() > 0.05:
            d = img[bright]
            r, g, b = d[_sat(d) > 0.3].T if (_sat(d) > 0.3).any() else (np.zeros(0),) * 3
            hue = np.degrees(np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)) % 360
            hist = np.histogram(hue, bins=12, range=(0, 360))[0] / max(len(hue), 1)
            if len(hue) > 0.3 * bright.sum() and ((hist > 0.05).sum() >= 2 or len(hue) > 0.6 * bright.sum()):
                out.update(ink=light, paper=dark, dark_paper=True, colored_dots=True)
    # Tint: bright local peaks (the gaps between dots) that are clearly colored
    # and darker than the paper.
    peaks = _local_max(lum)
    pl = float(paper @ LUMA)
    cand = peaks & (lum < pl - 0.08) & (lum > float(ink @ LUMA) + 0.15)
    sat = _sat(img)
    colored = cand & (sat > max(0.15, float(_sat(paper[None])[0]) + 0.1))
    frac = colored.sum() / max(peaks.sum(), 1)
    if frac > 0.12:
        t = img[colored]
        tl = t @ LUMA
        out["tint"] = np.median(t[tl >= np.median(tl)], axis=0)
        out["tint_fraction"] = float(frac)
    return out


def measure_softness(lum: np.ndarray, contrast: float) -> float:
    """Gaussian blur (pixels) of the edges: a blurred step's steepest slope is
    contrast / (sigma * sqrt(2 pi))."""
    gy, gx = np.gradient(lum)
    g = np.percentile(np.hypot(gx, gy), 99.5)
    if g <= 1e-4 or contrast <= 0.05:
        return 0.0
    return float(contrast / (g * np.sqrt(2 * np.pi)))


def measure_grain(lum: np.ndarray, paper_l: float) -> float:
    """Noise in clean paper areas (no ink within 4 px), as a fraction of full scale."""
    pad = np.pad(lum, 4, mode="edge")
    nb_min = np.lib.stride_tricks.sliding_window_view(pad, (9, 9)).min(axis=(-1, -2))
    flat = nb_min > paper_l - 0.2
    if flat.sum() < 500:
        return 0.0
    p1 = np.pad(lum, 1, mode="edge")
    mean3 = np.lib.stride_tricks.sliding_window_view(p1, (3, 3)).mean(axis=(-1, -2))
    return float((lum - mean3)[flat].std())


_DCT = np.array([[np.sqrt((1 if k == 0 else 2) / 8) * np.cos((2 * n + 1) * k * np.pi / 16) for n in range(8)]
                 for k in range(8)])


def _block_dct(y: np.ndarray, oy: int, ox: int) -> np.ndarray:
    h, w = (y.shape[0] - oy) // 8 * 8, (y.shape[1] - ox) // 8 * 8
    b = y[oy:oy + h, ox:ox + w].reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)
    return _DCT @ b @ _DCT.T


def measure_jpeg(img: np.ndarray) -> int | None:
    """Estimated JPEG quality: 0 = no JPEG traces, None = can't tell.

    JPEG rounds each 8x8 block's DCT coefficients to multiples of a quantization
    step. Recomputing the DCT on the true block grid shows that comb; on a grid
    shifted by 4 px it is gone. Blocks with clipped pixels (pure black/white,
    common in prints) are skipped, since clipping breaks the exact multiples.
    The step of coefficient (0,1) gives the quality via the standard luminance
    table (11 at quality 50)."""
    rgb8 = np.clip(img[:2048, :2048] * 255 + 0.5, 0, 255).astype(np.int16)
    y = rgb8 @ np.array([0.299, 0.587, 0.114]) - 128

    def coeffs(oy: int, ox: int) -> np.ndarray:
        h, w = (y.shape[0] - oy) // 8 * 8, (y.shape[1] - ox) // 8 * 8
        blk = rgb8[oy:oy + h, ox:ox + w].reshape(h // 8, 8, w // 8, 8, 3)
        ok = (blk.min(axis=(1, 3, 4)) > 2) & (blk.max(axis=(1, 3, 4)) < 253)
        c = _block_dct(y, oy, ox)[:, :, 0, 1][ok]
        return c[np.abs(c) > 1.5]

    def score(c: np.ndarray, q: float) -> float:
        return float(np.abs(np.mean(np.exp(2j * np.pi * c / q))))

    aligned = coeffs(0, 0)
    if aligned.size < 150:
        return None
    # Quantised coefficients are mostly +-1 step, so the commonest size is the step.
    q = float(np.bincount(np.round(np.abs(aligned)).astype(int)).argmax())
    if q < 2 or score(aligned, q) < 0.8:
        return 0
    shifted = coeffs(4, 4)
    if shifted.size >= 150 and score(shifted, q) > 0.5:
        return 0  # the comb is in the content, not the block grid
    scale = q / 11 * 100
    quality = 5000 / scale if scale > 100 else (200 - scale) / 2
    return int(np.clip(round(quality), 10, 98))


def screen_angles(img: np.ndarray, period: float) -> list[float]:
    """Screen angle of the cyan, magenta and yellow plates (from 1-R, 1-G, 1-B):
    the strongest peak near the screen frequency already found in luminance."""
    out = []
    f0 = 1 / period
    for c in range(3):
        near = [p for p in _peaks(1 - img[..., c], n=10) if abs(np.hypot(p[0], p[1]) - f0) < 0.2 * f0]
        if not near:
            out.append(float("nan"))
            continue
        p = max(near, key=lambda p: p[2])
        out.append(float(np.degrees(np.arctan2(p[1], p[0])) % 90))
    return out


# --- putting it together -----------------------------------------------------------


def _hex(rgb) -> str:
    return "#" + "".join(f"{int(round(float(np.clip(c, 0, 1)) * 255)):02x}" for c in rgb)


def match(img: np.ndarray) -> Match:
    """img: RGB float in 0..1 (or uint8)."""
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    img = img[..., :3]
    h, w = img.shape[:2]
    lum = img @ LUMA
    grid = analyse_grid(lum)
    colors = measure_colors(img)
    ink, paper = colors["ink"], colors["paper"]
    contrast = float(paper @ LUMA - ink @ LUMA)
    sigma = measure_softness(lum, contrast)
    # Darkest channel: pale inks (yellow) still count as ink, not paper noise.
    grain = measure_grain(img.min(axis=-1), float(paper.min()))
    jpeg_q = measure_jpeg(img)

    notes: list[str] = []
    s: dict = {}
    if grid["kind"] == "none":
        raise ValueError("no dot pattern found: is this a halftone or dither print?")
    if grid["kind"] == "halftone":
        cells = w / grid["period"]
        s.update(effect="halftone", spot="dot", angle=round(grid["angle"], 1), cells=int(round(cells)))
        notes.append(f"halftone screen: {grid['period']:.2f}px cells at {grid['angle']:.1f}°"
                     f" ({cells:.0f} across)")
    else:
        cells = w / grid["pixel"]
        s.update(effect="dither", algorithm=grid["algorithm"], cell_shape="circle", dot_size=1.05,
                 cells=int(round(cells)))
        kind = "ordered (Bayer)" if grid["algorithm"] == "bayer4" else "irregular (blue-noise-like)"
        notes.append(f"{kind} dither: {grid['pixel']:.2f}px cells ({cells:.0f} across)")
        if grid.get("uneven"):
            notes.append("cell spacing varies across the image (resized or stitched?); "
                         "used the dominant spacing")

    angles: list[float] = []
    if grid["kind"] == "halftone" and (_sat(img) > 0.25).mean() > 0.1:
        angles = [a for a in screen_angles(img, grid["period"]) if a == a]
    if len(angles) == 3 and max(angles) - min(angles) > 8:
        dark = colors["dark"]
        if float(dark.max()) < 0.12 and (lum < 0.15).mean() > 0.4:
            # Separate colored screens glowing on black: additive RGB.
            s.update(mode="rgb", paper=_hex(dark), angles=" ".join(f"{a:.0f}" for a in angles))
            notes.append("separate red/green/blue screens on black: RGB mode")
        else:
            # Black usually runs at 45 degrees; keep the luminance angle if it's a fourth one.
            k = s["angle"] if min(abs(s["angle"] - a) for a in angles) > 8 else 45
            s.update(mode="cmyk", paper=_hex(colors["light"]),
                     angles=" ".join(f"{a:.0f}" for a in angles) + f" {k:.0f}")
            notes.append("separate screen angles per color (" + ", ".join(f"{a:.0f}°" for a in angles)
                         + "): four-color CMYK print")
    elif colors.get("colored_dots") and s["effect"] == "halftone":
        s.update(mode="color", paper=_hex(paper))
        notes.append("dots carry the photo's own colors: color mode")
    else:
        s.update(mode="mono", ink=_hex(ink), paper=_hex(paper))
        if colors["dark_paper"]:
            s["invert"] = True
            notes.append("light ink on dark paper")
        if "tint" in colors and not colors["dark_paper"]:
            s.update(tint=_hex(colors["tint"]), tint_start=0.3, tint_end=0.5)
            notes.append(f"second flat color under the dots: {_hex(colors['tint'])} "
                         f"(placed by tone; try --tint-by saturation if it marks sky/colored areas)")

    # Edge softness: our antialiasing is about half a pixel already.
    blur = float(np.sqrt(max(sigma ** 2 - 0.5 ** 2, 0)))
    if blur > 0.25:
        s["post_blur"] = round(min(blur, 3.0), 2)
    if grain > 0.006:
        s["grain"] = round(float(np.clip(grain / GRAIN_PER_UNIT, 0.05, 1.0)), 2)
    printed = grain > 0.006
    if printed and s["effect"] == "halftone":
        # Real presses: dots grow, the tiniest dots fail to print, ink mottles.
        s.update(dot_gain=1.4, min_dot=0.06, ink_noise=0.2, roughness=0.2, rough_scale=0.9)
        notes.append("looks printed/scanned: added dot gain, highlight dropout and ink mottle")
    if jpeg_q:
        s["jpeg"] = jpeg_q
        notes.append(f"saved as JPEG, quality ≈ {jpeg_q}")
    elif jpeg_q is None:
        notes.append("JPEG wear: can't tell (pure black/white everywhere); add --jpeg if it looks compressed")

    measured = {
        **grid, "ink": _hex(ink), "paper": _hex(paper), "edge_sigma": round(sigma, 2),
        "grain": round(grain, 4), "jpeg_quality": "unknown" if jpeg_q is None else jpeg_q,
    }
    if "tint" in colors:
        measured["tint"] = _hex(colors["tint"])
    return Match(s, notes, measured)


def calibrate(img: np.ndarray, m: Match, width: int = 420) -> float:
    """Tune gamma, contrast and the tint range so that re-printing the
    descreened reference reproduces the reference's local colors. Works on a
    small copy: presets size dots with `cells`, so the result transfers.
    Returns the remaining average color error (0..1)."""
    from .ops import resize
    from .pipeline import render
    from .settings import Settings

    h, w = img.shape[:2]
    small = np.clip(resize(img, width, max(1, round(h * width / w)), "box"), 0, 1)
    flat = descreen(small, m)
    sigma = width / max(m.settings.get("cells", 100), 1)
    target = gaussian_blur(small, sigma)
    # Texture settings shift tone a little too (mottled ink prints lighter), so
    # calibrate with the full preset; at this size it's quick.
    skip: set[str] = set()

    def error(extra: dict) -> float:
        st = Settings().update({k: v for k, v in {**m.settings, **extra}.items() if k not in skip})
        out = render(flat, st)[..., :3]
        return float(np.abs(gaussian_blur(out, sigma) - target).mean())

    best: dict = {}
    best_err = error(best)
    grid = {
        "gamma": [0.45, 0.55, 0.65, 0.8, 1.0, 1.2, 1.45, 1.75],
        "contrast": [0.85, 1.0, 1.15, 1.3, 1.5],
    }
    if "tint" in m.settings:
        grid["tint_start"] = [0.1, 0.2, 0.3, 0.4, 0.5]
    for _ in range(2):  # two rounds of coordinate search
        for key, values in grid.items():
            for v in values:
                trial = {**best, key: v}
                if key == "tint_start":
                    trial["tint_end"] = round(v + 0.2, 2)
                e = error(trial)
                if e < best_err - 1e-4:
                    best, best_err = trial, e
    for key, v in best.items():
        if key in ("gamma", "contrast") and v == 1.0:
            continue
        m.settings[key] = v
    return best_err


def descreen(img: np.ndarray, m: Match) -> np.ndarray:
    """Blur a reference back to continuous tone, to re-render it for comparison."""
    w = img.shape[1]
    size = w / max(m.settings.get("cells", 100), 1)
    return np.clip(gaussian_blur(img.astype(np.float32), max(size * 0.7, 1.0)), 0, 1)
