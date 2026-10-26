"""Turn an RGB image into the final print: adjust -> effect -> texture -> composite."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import dither as dz
from . import halftone as ht
from .adjust import adjust
from .color import auto_palette, parse_color, parse_palette
from .ops import fbm_noise, gaussian_blur, luminance, resize, smoothstep, value_noise
from .settings import Settings


@dataclass
class Layer:
    coverage: np.ndarray  # HxW, 0..1
    color: np.ndarray  # (3,) or HxWx3
    blend: str = "lerp"  # lerp | multiply | add


# --- sizing ---------------------------------------------------------------------


def target_size(w: int, h: int, s: Settings) -> tuple[int, int]:
    if s.width and s.height:
        return s.width, s.height
    if s.width:
        return s.width, max(1, round(h * s.width / w))
    if s.height:
        return max(1, round(w * s.height / h)), s.height
    return max(1, round(w * s.scale)), max(1, round(h * s.scale))


def prepare(img: np.ndarray, s: Settings) -> np.ndarray:
    """uint8/float RGB(A) -> float32 RGB at processing size."""
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    if img.shape[2] == 4:
        a = img[..., 3:4]
        img = img[..., :3] * a + (1 - a)  # flatten transparency onto white
    img = img[..., :3]
    h, w = img.shape[:2]
    tw, th = target_size(w, h, s)
    if (tw, th) != (w, h):
        img = np.clip(resize(img, tw, th, "lanczos"), 0, 1)
    return img


def lightness(img: np.ndarray, channel: str) -> np.ndarray:
    """Single-channel lightness (0 = black, 1 = white) used by mono/gray modes."""
    if channel == "luma":
        return luminance(img)
    if channel in ("red", "green", "blue"):
        return img[..., ("red", "green", "blue").index(channel)]
    if channel == "average":
        return img.mean(axis=-1)
    if channel == "max":
        return img.max(axis=-1)
    if channel == "min":
        return img.min(axis=-1)
    raise ValueError(f"unknown channel: {channel}")


# --- texture helpers ------------------------------------------------------------


def _rough_field(h, w, s: Settings, rng, cell: float) -> np.ndarray | None:
    if s.roughness <= 0:
        return None
    fine = value_noise(h, w, s.rough_scale, rng)
    coarse = value_noise(h, w, max(cell * 0.6, 2.0), rng)
    # Displacement of dot edges, in pixels.
    return s.roughness * (1.2 * fine + 0.6 * coarse) * (1.0 + 0.12 * cell)


def _ink_variation(h, w, s: Settings, rng) -> np.ndarray | None:
    if s.ink_noise <= 0:
        return None
    blotch = fbm_noise(h, w, max(h, w) / 6, rng, octaves=4)
    mottle = value_noise(h, w, 2.5, rng)
    return np.clip(1 - s.ink_noise * (0.35 * (blotch + 1) + 0.25 * (mottle + 1) * 0.5), 0, 1)


def _finish_coverage(cov: np.ndarray, s: Settings, variation) -> np.ndarray:
    if s.bleed > 0:
        b = gaussian_blur(cov, s.bleed)
        # Spread: anything with enough nearby ink fills in, edges stay soft.
        cov = np.maximum(cov, smoothstep(0.18, 0.55, b))
    if variation is not None:
        cov = cov * variation
    if s.ink_opacity < 1:
        cov = cov * s.ink_opacity
    return cov.astype(np.float32)


# --- effects --------------------------------------------------------------------


def halftone_layers(img: np.ndarray, s: Settings, rng) -> list[Layer]:
    h, w = img.shape[:2]
    rough = _rough_field(h, w, s, rng, s.cell)
    paper = parse_color(s.paper)
    layers: list[Layer] = []

    def plate(channel: np.ndarray, angle: float, plate_rng=None) -> np.ndarray:
        off = (0.0, 0.0)
        if s.misregister > 0 and plate_rng is not None:
            off = tuple(plate_rng.normal(0, s.misregister, 2))
        scr = ht.Screen(h, w, s.cell, angle, off)
        tone = ht.tone_curve(scr.sample(channel, s.sample), s)
        return scr.coverage(tone, s.spot, s.softness, rough)

    if s.mode in ("mono", "gray"):
        dark = 1 - lightness(img, s.channel)
        layers.append(Layer(plate(dark, s.angle), np.array(parse_color(s.ink), np.float32)))
    elif s.mode == "color":
        lum = luminance(img)
        p_l = 1.0 if paper is None else float(np.dot(paper, [0.2126, 0.7152, 0.0722]))
        tone_src = np.abs(lum - p_l) / max(p_l, 1 - p_l, 1e-3)
        scr = ht.Screen(h, w, s.cell, s.angle)
        tone = ht.tone_curve(scr.sample(tone_src, s.sample), s)
        cov = scr.coverage(tone, s.spot, s.softness, rough)
        # Dots carry their cell's color; boost it so small dots still read.
        rgb = scr.sample(img, s.sample)
        rgb_l = luminance(rgb)[..., None]
        rgb = np.clip(rgb_l + (rgb - rgb_l) * 1.25, 0, 1)
        layers.append(Layer(cov, rgb.astype(np.float32)))
    elif s.mode == "cmyk":
        inks = parse_palette(s.cmyk_inks)
        if len(inks) < 4:
            raise ValueError("cmyk_inks needs four colors (C M Y K)")
        angles = ht.parse_angles(s.angles, 4, s.angle)
        plate_rng = np.random.default_rng(s.seed + 101)
        for chan, ang, ink in zip(ht.rgb_to_cmyk(img, s.gcr), angles, inks):
            layers.append(Layer(plate(chan, ang, plate_rng), ink, "multiply"))
    elif s.mode == "rgb":
        angles = ht.parse_angles(s.angles, 3, s.angle)
        plate_rng = np.random.default_rng(s.seed + 202)
        for i, ang in enumerate(angles[:3]):
            ink = np.zeros(3, np.float32)
            ink[i] = 1
            layers.append(Layer(plate(img[..., i], ang, plate_rng), ink, "add"))
    else:
        raise ValueError(f"mode {s.mode!r} is not supported by halftone")
    return layers


def _cell_mask(shape: str, p: int, size: float) -> np.ndarray:
    """Antialiased p x p coverage tile for one dither pixel."""
    ss = 8
    c = (np.arange(p * ss) + 0.5) / (p * ss) - 0.5
    x, y = np.meshgrid(c, c)
    r = 0.5 * size
    ax, ay = np.abs(x), np.abs(y)
    if shape == "square":
        m = (ax <= r) & (ay <= r)
    elif shape == "circle":
        m = x * x + y * y <= r * r
    elif shape == "diamond":
        m = ax + ay <= r
    elif shape == "plus":
        t = r * 0.38
        m = ((ax <= t) & (ay <= r)) | ((ay <= t) & (ax <= r))
    elif shape == "x":
        d1, d2 = np.abs(x - y) / np.sqrt(2), np.abs(x + y) / np.sqrt(2)
        t = r * 0.3
        m = ((d1 <= t) | (d2 <= t)) & (ax <= r) & (ay <= r)
    elif shape == "hline":
        m = (ay <= r * 0.5) & (ax <= 0.5)
    elif shape == "vline":
        m = (ax <= r * 0.5) & (ay <= 0.5)
    elif shape == "ring":
        rr = np.sqrt(x * x + y * y)
        m = (rr <= r) & (rr >= r * 0.55)
    elif shape == "rounded":
        k = r * 0.45
        qx, qy = np.maximum(ax - (r - k), 0), np.maximum(ay - (r - k), 0)
        m = (ax <= r) & (ay <= r) & (qx * qx + qy * qy <= k * k)
    else:
        raise ValueError(f"unknown cell shape: {shape}")
    return m.reshape(p, ss, p, ss).mean(axis=(1, 3)).astype(np.float32)


def dither_layers(img: np.ndarray, s: Settings, rng) -> list[Layer]:
    h, w = img.shape[:2]
    p = int(s.pixel)
    sh, sw = -(-h // p), -(-w // p)
    small = img if p == 1 else np.clip(resize(img, sw, sh, "box"), 0, 1)
    ink = parse_color(s.ink)
    paper = parse_color(s.paper)
    paper_rgb = np.array(paper if paper is not None else (1.0, 1.0, 1.0), np.float32)
    ink_rgb = np.array(ink if ink is not None else (0.0, 0.0, 0.0), np.float32)
    algo = s.algorithm

    tone = None  # per small pixel darkness (0..1) for size_by_tone
    if s.mode in ("mono", "gray"):
        n = s.levels
        lv = np.linspace(0, 1, n, dtype=np.float32)
        idx = dz.quantize(lightness(small, s.channel)[..., None], lv[:, None], algo, s, rng)
        level = lv[idx]
        colors = ink_rgb + (paper_rgb - ink_rgb) * level[..., None]
        tone = 1 - level
    elif s.mode == "rgb":
        n = s.levels
        lv = np.linspace(0, 1, n, dtype=np.float32)
        chans = []
        for i in range(3):
            # Decorrelate channels so ordered patterns don't stack into gray.
            sub_rng = np.random.default_rng(s.seed + i)
            chan = small[..., i:i + 1]
            if algo not in dz.KERNELS and algo != "random":
                chan = np.roll(chan, (i * 3, i * 5), axis=(0, 1))
                idx = np.roll(dz.quantize(chan, lv[:, None], algo, s, sub_rng), (-i * 3, -i * 5), axis=(0, 1))
            else:
                idx = dz.quantize(chan, lv[:, None], algo, s, sub_rng)
            chans.append(lv[idx])
        colors = np.stack(chans, axis=-1)
        tone = 1 - luminance(colors)
    elif s.mode in ("palette", "color"):
        pal = parse_palette(s.palette) if s.mode == "palette" else auto_palette(small, s.colors)
        idx = dz.quantize(small, pal, algo, s, rng)
        colors = pal[idx]
        pal_l = luminance(pal)
        tone = 1 - (pal_l[idx] - pal_l.min()) / max(float(np.ptp(pal_l)), 1e-6)
    else:
        raise ValueError(f"mode {s.mode!r} is not supported by dither")

    def up(a):
        return np.repeat(np.repeat(a, p, axis=0), p, axis=1)[:h, :w]

    color_full = up(colors.astype(np.float32))
    if s.size_by_tone:
        steps = 16
        q = np.clip(np.round(tone * steps), 0, steps).astype(np.int64)
        tiles = np.stack([_cell_mask(s.cell_shape, p, s.dot_size * np.sqrt(k / steps) * 1.15) for k in range(steps + 1)])
        tiles[0] = 0
        yy, xx = np.mgrid[0:h, 0:w]
        cov = tiles[up(q), yy % p, xx % p]
        if s.mode in ("mono", "gray"):
            color_full = np.broadcast_to(ink_rgb, color_full.shape)
        return [Layer(cov.astype(np.float32), color_full)]
    mask = _cell_mask(s.cell_shape, p, s.dot_size)
    cov = np.tile(mask, (sh, sw))[:h, :w]
    if s.mode in ("mono", "gray"):
        # Ink amount lives in coverage so bleed, roughness and transparency apply.
        cov = cov * up(tone.astype(np.float32))
        color_full = np.broadcast_to(ink_rgb, color_full.shape)
    return [Layer(cov.astype(np.float32), color_full)]


# --- compositing ----------------------------------------------------------------


def composite(layers: list[Layer], s: Settings, h: int, w: int, rng) -> np.ndarray:
    paper = parse_color(s.paper)
    transparent = paper is None
    base = np.array(paper if paper is not None else (1.0, 1.0, 1.0), np.float32)
    out = np.empty((h, w, 3), np.float32)
    out[...] = base
    alpha_keep = np.ones((h, w), np.float32)
    for layer in layers:
        c = layer.coverage[..., None]
        col = layer.color
        if layer.blend == "multiply":
            out = out * (1 - c * (1 - col))
        elif layer.blend == "add":
            out = out + c * col
        else:
            out = out * (1 - c) + col * c
        alpha_keep *= 1 - layer.coverage
    out = np.clip(out, 0, 1)

    if s.grain > 0:
        n = value_noise(h, w, s.grain_scale, rng) * 0.6 + value_noise(h, w, s.grain_scale * 4, rng) * 0.4
        out = np.clip(out * (1 + s.grain * 0.22 * n[..., None]) + s.grain * 0.03 * n[..., None], 0, 1)
    if s.speckle > 0:
        r = rng.random((h, w), dtype=np.float32)
        dust = gaussian_blur((r > 1 - 0.0025 * s.speckle).astype(np.float32), 0.6) * 2.5
        holes = gaussian_blur((r < 0.004 * s.speckle).astype(np.float32), 0.5) * 2.0
        ink_like = np.array(parse_color(s.ink) or (0, 0, 0), np.float32)
        out = out + (base - out) * np.clip(holes, 0, 1)[..., None]
        out = out + (ink_like - out) * np.clip(dust, 0, 0.8)[..., None]
    if s.vignette > 0:
        yy, xx = np.mgrid[0:h, 0:w]
        d = ((xx / max(w - 1, 1) - 0.5) ** 2 + (yy / max(h - 1, 1) - 0.5) ** 2) * 2
        out = out * (1 - s.vignette * smoothstep(0.15, 1.0, d) * 0.8)[..., None]
    if s.post_blur > 0:
        out = gaussian_blur(out, s.post_blur)
    out = np.clip(out, 0, 1)

    if transparent:
        alpha = 1 - alpha_keep
        # Un-premultiply against the white working paper.
        rgb = np.where(alpha[..., None] > 1e-4, (out - (1 - alpha[..., None])) / np.maximum(alpha[..., None], 1e-4), 0)
        return np.concatenate([np.clip(rgb, 0, 1), alpha[..., None]], axis=-1)
    return out


def render(img: np.ndarray, s: Settings, frame: int = 0) -> np.ndarray:
    """Full pipeline. Returns float32 RGB (or RGBA when paper is 'none')."""
    s.validate()
    src = prepare(img, s)
    src = adjust(src, s)
    h, w = src.shape[:2]
    if s.cells > 0:
        s = replace(s, cell=max(w / s.cells, 1.0), pixel=max(1, round(w / s.cells)))
    seed = s.seed + (frame if s.animate_grain else 0)
    rng = np.random.default_rng(seed)
    if s.effect == "halftone":
        layers = halftone_layers(src, s, rng)
        cell = s.cell
    else:
        layers = dither_layers(src, s, rng)
        cell = s.pixel
    variation = _ink_variation(h, w, s, rng)
    for layer in layers:
        layer.coverage = _finish_coverage(layer.coverage, s, variation)
    if s.effect == "dither" and s.roughness > 0:
        # Blur then re-threshold with a displaced level: ragged, inky cell edges.
        rough = _rough_field(h, w, s, rng, cell)
        for layer in layers:
            b = gaussian_blur(layer.coverage, 0.8)
            layer.coverage = np.clip((b - 0.5 + 0.5 * rough) * 2 + 0.5, 0, 1).astype(np.float32)
    return composite(layers, s, h, w, rng)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
