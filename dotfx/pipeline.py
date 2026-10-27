"""Turn an RGB image into the final print: adjust -> effect -> texture -> composite."""

from __future__ import annotations

import zlib
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from . import dither as dz
from . import halftone as ht
from .adjust import adjust
from .color import auto_palette, parse_color, parse_palette
from .ops import LUMA, fbm_noise, gaussian_blur, luminance, resize, smoothstep, value_noise
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
        h, w = img.shape[:2]
        tw, th = target_size(w, h, s)
        if (tw, th) != (w, h) and img.ndim == 3:
            # Resize 8-bit images in Pillow directly: much faster than float.
            from PIL import Image

            img = np.asarray(Image.fromarray(img).resize((tw, th), Image.Resampling.LANCZOS))
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


def saturation(img: np.ndarray) -> np.ndarray:
    """HSV-style saturation: 0 for grays, 1 for pure colors."""
    mx = img.max(axis=-1)
    return (mx - img.min(axis=-1)) / np.maximum(mx, 1e-4)


def tint_amount(img: np.ndarray, s: Settings) -> np.ndarray:
    """0..1 tint coverage target for mono/gray modes."""
    src = saturation(img) if s.tint_by == "saturation" else 1 - lightness(img, s.channel)
    return smoothstep(s.tint_start, s.tint_end, src)


# --- texture helpers ------------------------------------------------------------


@lru_cache(maxsize=12)
def _noise(kind: str, h: int, w: int, seed: int, scale: float) -> np.ndarray:
    """Cached noise fields, so static textures cost nothing per video frame."""
    rng = np.random.default_rng([seed, zlib.crc32(kind.encode())])
    if kind.startswith("fbm"):
        out = fbm_noise(h, w, scale, rng, octaves=4)
    elif kind.startswith("uniform"):
        out = rng.random((h, w), dtype=np.float32)
    else:
        out = value_noise(h, w, scale, rng)
    out.setflags(write=False)
    return out


def _rough_field(h, w, s: Settings, seed: int, cell: float) -> np.ndarray | None:
    if s.roughness <= 0:
        return None
    fine = _noise("rough-fine", h, w, seed, s.rough_scale)
    coarse = _noise("rough-coarse", h, w, seed, max(cell * 0.6, 2.0))
    # Displacement of dot edges, in pixels.
    return s.roughness * (1.2 * fine + 0.6 * coarse) * (1.0 + 0.12 * cell)


def _ink_variation(h, w, s: Settings, seed: int) -> np.ndarray | None:
    if s.ink_noise <= 0:
        return None
    blotch = _noise("fbm-blotch", h, w, seed, max(h, w) / 6)
    mottle = _noise("mottle", h, w, seed, 2.5)
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


def halftone_layers(img: np.ndarray, s: Settings, seed: int) -> list[Layer]:
    h, w = img.shape[:2]
    rough = _rough_field(h, w, s, seed, s.cell)
    paper = parse_color(s.paper)
    layers: list[Layer] = []

    def plate(channel: np.ndarray, angle: float, plate_rng=None) -> np.ndarray:
        off = (0.0, 0.0)
        if s.misregister > 0 and plate_rng is not None:
            off = tuple(round(float(o), 3) for o in plate_rng.normal(0, s.misregister, 2))
        scr = ht.get_screen(h, w, float(s.cell), float(angle), off, s.spot)
        tone = ht.tone_curve(scr.sample(channel, s.sample), s)
        return scr.coverage(tone, s.softness, rough)

    if s.mode in ("mono", "gray"):
        dark = 1 - lightness(img, s.channel)
        tint = parse_color(s.tint)
        if tint is not None:
            # Same screen, so the tint dot sits concentrically under the ink dot.
            tint_tone = tint_amount(img, s)
            layers.append(Layer(plate(tint_tone, s.angle), np.array(tint, np.float32)))
        layers.append(Layer(plate(dark, s.angle), np.array(parse_color(s.ink), np.float32)))
    elif s.mode == "color":
        # Dot size = how far the color is from the paper, so saturated colors
        # (yellow on white, say) still get solid dots.
        p_rgb = np.array(paper if paper is not None else (1.0, 1.0, 1.0), np.float32)
        far = float(np.linalg.norm(np.maximum(p_rgb, 1 - p_rgb)))
        tone_src = np.linalg.norm(img - p_rgb, axis=-1) / max(far, 1e-3)
        scr = ht.get_screen(h, w, float(s.cell), float(s.angle), (0.0, 0.0), s.spot)
        tone = ht.tone_curve(scr.sample(tone_src, s.sample), s)
        cov = scr.coverage(tone, s.softness, rough)
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
    elif s.mode == "duotone":
        dark = 1 - lightness(img, s.channel)
        plate_rng = np.random.default_rng(s.seed + 303)
        ink2 = parse_color(s.ink2)
        ink = parse_color(s.ink)
        if ink is None or ink2 is None:
            raise ValueError("duotone inks can't be 'none'")
        # Second ink carries the whole tonal range lightly; the first ink only
        # comes in for the shadows, like a classic duotone curve pair.
        light = plate(np.clip(dark * 0.85, 0, 1), s.angle + 30, plate_rng)
        shadow = plate(smoothstep(0.3, 1.0, dark), s.angle, plate_rng)
        layers.append(Layer(light, np.array(ink2, np.float32), "multiply"))
        layers.append(Layer(shadow, np.array(ink, np.float32), "multiply"))
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
    """Dither layers; a fractional --pixel is rendered at a whole-pixel size on an
    enlarged canvas and scaled back down, which also antialiases the cells."""
    p = float(s.pixel)
    if abs(p - round(p)) < 1e-6:
        return _dither_layers(img, replace(s, pixel=int(round(p))), rng)
    h, w = img.shape[:2]
    big_p = int(np.ceil(p))
    k = big_p / p
    bw, bh = max(1, round(w * k)), max(1, round(h * k))
    big = np.clip(resize(img, bw, bh, "lanczos"), 0, 1)
    out = []
    for layer in _dither_layers(big, replace(s, pixel=big_p), rng):
        cov = np.clip(resize(layer.coverage, w, h, "box"), 0, 1)
        color = layer.color
        if color.ndim == 3:
            # Resize premultiplied color so cell colors don't bleed into empty paper.
            pm = resize(color * layer.coverage[..., None], w, h, "box")
            color = np.clip(pm / np.maximum(cov[..., None], 1e-4), 0, 1)
        out.append(Layer(cov.astype(np.float32), color.astype(np.float32), layer.blend))
    return out


@dataclass
class DitherGrid:
    """The dithered image at one entry per cell (shared by raster and vector output)."""

    colors: np.ndarray  # sh x sw x 3, the color each cell prints
    tone: np.ndarray  # sh x sw, ink amount 0..1 (for mono: 1 - gray level)
    tint_on: np.ndarray | None  # sh x sw bool: cells carrying the tint
    ink: np.ndarray
    mono: bool


def dither_grid(small: np.ndarray, s: Settings, rng) -> DitherGrid:
    """Quantise an image that already has one pixel per dither cell."""
    ink = parse_color(s.ink)
    paper = parse_color(s.paper)
    paper_rgb = np.array(paper if paper is not None else (1.0, 1.0, 1.0), np.float32)
    ink_rgb = np.array(ink if ink is not None else (0.0, 0.0, 0.0), np.float32)
    algo = s.algorithm
    tint_on = None
    if s.mode in ("mono", "gray"):
        lv = np.linspace(0, 1, s.levels, dtype=np.float32)
        light = lightness(small, s.channel)
        idx = dz.quantize(light[..., None], lv[:, None], algo, s, rng)
        if parse_color(s.tint) is not None:
            # Dithered with the same threshold map, so tint cells nest around the
            # ink cells; drawn as whole cells to read as a flat second color.
            tint_light = 1 - tint_amount(small, s)
            two = np.array([[0.0], [1.0]], np.float32)
            tint_on = dz.quantize(tint_light[..., None], two, algo, s, rng) == 0
        level = lv[idx]
        colors = ink_rgb + (paper_rgb - ink_rgb) * level[..., None]
        return DitherGrid(colors, 1 - level, tint_on, ink_rgb, True)
    if s.mode == "rgb":
        lv = np.linspace(0, 1, s.levels, dtype=np.float32)
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
        return DitherGrid(colors, 1 - luminance(colors), None, ink_rgb, False)
    if s.mode in ("palette", "color"):
        pal = parse_palette(s.palette) if s.mode == "palette" else auto_palette(small, s.colors)
        idx = dz.quantize(small, pal, algo, s, rng)
        pal_l = pal @ LUMA  # (a palette is 2-D, which luminance() would take for a gray image)
        tone = 1 - (pal_l[idx] - pal_l.min()) / max(float(np.ptp(pal_l)), 1e-6)
        return DitherGrid(pal[idx], tone, None, ink_rgb, False)
    raise ValueError(f"mode {s.mode!r} is not supported by dither")


def _dither_layers(img: np.ndarray, s: Settings, rng) -> list[Layer]:
    h, w = img.shape[:2]
    p = int(s.pixel)
    sh, sw = -(-h // p), -(-w // p)
    small = img if p == 1 else np.clip(resize(img, sw, sh, "box"), 0, 1)
    grid = dither_grid(small, s, rng)
    colors, tone, ink_rgb = grid.colors, grid.tone, grid.ink
    tint_layer = None
    if grid.tint_on is not None:
        on_full = np.repeat(np.repeat(grid.tint_on, p, axis=0), p, axis=1)[:h, :w]
        tint_layer = Layer(on_full.astype(np.float32), np.array(parse_color(s.tint), np.float32))

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
            color_full = ink_rgb
        return [tl for tl in (tint_layer,) if tl is not None] + [Layer(cov.astype(np.float32), color_full)]
    mask = _cell_mask(s.cell_shape, p, s.dot_size)
    cov = np.tile(mask, (sh, sw))[:h, :w]
    if s.mode in ("mono", "gray"):
        # Ink amount lives in coverage so bleed, roughness and transparency apply.
        cov = cov * up(tone.astype(np.float32))
        color_full = ink_rgb
    return [tl for tl in (tint_layer,) if tl is not None] + [Layer(cov.astype(np.float32), color_full)]


# --- compositing ----------------------------------------------------------------


def composite(layers: list[Layer], s: Settings, h: int, w: int, seed: int) -> np.ndarray:
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
        n = _noise("grain", h, w, seed, s.grain_scale) * 0.6 + _noise("grain-coarse", h, w, seed, s.grain_scale * 4) * 0.4
        out = np.clip(out * (1 + s.grain * 0.22 * n[..., None]) + s.grain * 0.03 * n[..., None], 0, 1)
    if s.speckle > 0:
        r = _noise("uniform-speckle", h, w, seed, 1.0)
        dust = gaussian_blur((r > 1 - 0.0025 * s.speckle).astype(np.float32), 0.6) * 2.5
        holes = gaussian_blur((r < 0.004 * s.speckle).astype(np.float32), 0.5) * 2.0
        ink_like = np.array(parse_color(s.ink) or (0, 0, 0), np.float32)
        out = out + (base - out) * np.clip(holes, 0, 1)[..., None]
        out = out + (ink_like - out) * np.clip(dust, 0, 0.8)[..., None]
    if s.glow > 0:
        # Screen-blend a blurred copy: light bleeding from bright dots.
        radius = max(s.cell if s.effect == "halftone" else s.pixel, 2) * 0.8
        halo = gaussian_blur(out, radius)
        out = 1 - (1 - out) * (1 - np.clip(halo * s.glow * 1.5, 0, 1))
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


MASK_KEYS = ("bright", "dark", "colorful", "center")


def make_mask(src: np.ndarray, s: Settings) -> np.ndarray:
    """0..1 mask at the size of the prepared source (1 = gets the look)."""
    h, w = src.shape[:2]
    t = s.mask_threshold
    if s.mask == "bright":
        m = smoothstep(t - 0.1, t + 0.1, luminance(src))
    elif s.mask == "dark":
        m = 1 - smoothstep(t - 0.1, t + 0.1, luminance(src))
    elif s.mask == "colorful":
        mx = src.max(axis=-1)
        sat = (mx - src.min(axis=-1)) / np.maximum(mx, 1e-4)
        m = smoothstep(t * 0.6 - 0.08, t * 0.6 + 0.08, sat)
    elif s.mask == "center":
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.hypot((xx + 0.5) / w - 0.5, (yy + 0.5) / h - 0.5) * 2
        m = 1 - smoothstep(t * 1.2 - 0.15, t * 1.2 + 0.15, r)
    else:
        from PIL import Image

        try:
            with Image.open(s.mask) as im:
                m = np.asarray(im.convert("L").resize((w, h), Image.Resampling.BILINEAR), np.float32) / 255
        except OSError as e:
            raise ValueError(f"mask: can't read {s.mask!r} ({e})") from None
    m = gaussian_blur(m.astype(np.float32), s.mask_feather) if s.mask_feather > 0 else m.astype(np.float32)
    return 1 - m if s.mask_invert else m


def _background(img: np.ndarray, s: Settings, frame: int, shape) -> np.ndarray:
    h, w = shape
    kind = s.background.strip()
    if kind in ("original", ""):
        return prepare(img, s)
    if kind in ("paper", "none"):
        out = np.empty((h, w, 4 if kind == "none" else 3), np.float32)
        if kind == "none":
            out[...] = 0
        else:
            out[...] = parse_color(s.paper) or (1.0, 1.0, 1.0)
        return out
    from .presets import load_preset

    other = Settings().update(load_preset(kind))
    other = replace(other, width=s.width, height=s.height, scale=s.scale, mask="", clip_edge="none", jpeg=0)
    return render(img, other, frame)


def _as_rgba(a: np.ndarray) -> np.ndarray:
    return a if a.shape[2] == 4 else np.concatenate([a, np.ones(a.shape[:2] + (1,), np.float32)], axis=-1)


def render(img: np.ndarray, s: Settings, frame: int = 0) -> np.ndarray:
    """Full pipeline. Returns float32 RGB (or RGBA when paper is 'none')."""
    s.validate()
    if s.mask:
        # Render this look and the background, then blend them through the mask.
        # Clipping and JPEG wear apply to the combined image.
        inner = replace(s, mask="", clip_edge="none", jpeg=0)
        fore = render(img, inner, frame)
        h, w = fore.shape[:2]
        m = make_mask(prepare(img, s), s)[..., None]
        back = _background(img, s, frame, (h, w))
        if fore.shape[2] == 4 or back.shape[2] == 4:
            fore, back = _as_rgba(fore), _as_rgba(back)
        out = fore * m + back * (1 - m)
        if s.clip_edge != "none":
            from .clipping import apply_clipping

            out = apply_clipping(out, s)
        return jpeg_roundtrip(out, s.jpeg) if s.jpeg > 0 else out
    src = prepare(img, s)
    src = adjust(src, s)
    h, w = src.shape[:2]
    if s.cells > 0:
        s = replace(s, cell=max(w / s.cells, 1.0), pixel=max(w / s.cells, 1.0))
    seed = s.seed + (frame if s.animate_grain else 0)
    rng = np.random.default_rng(seed)
    if s.effect == "halftone":
        layers = halftone_layers(src, s, seed)
        cell = s.cell
    else:
        layers = dither_layers(src, s, rng)
        cell = s.pixel
    variation = _ink_variation(h, w, s, seed)
    for layer in layers:
        layer.coverage = _finish_coverage(layer.coverage, s, variation)
    if s.effect == "dither" and s.roughness > 0:
        # Blur then re-threshold with a displaced level: ragged, inky cell edges.
        rough = _rough_field(h, w, s, seed, cell)
        for layer in layers:
            b = gaussian_blur(layer.coverage, 0.8)
            layer.coverage = np.clip((b - 0.5 + 0.5 * rough) * 2 + 0.5, 0, 1).astype(np.float32)
    out = composite(layers, s, h, w, seed)
    if s.clip_edge != "none":
        from .clipping import apply_clipping

        out = apply_clipping(out, s)
    if s.jpeg > 0:
        out = jpeg_roundtrip(out, s.jpeg)
    return out


def jpeg_roundtrip(img: np.ndarray, quality: int) -> np.ndarray:
    """Pass the RGB channels through a 4:2:0 JPEG encode/decode (alpha untouched)."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(to_uint8(img[..., :3])).save(buf, format="JPEG", quality=int(quality), subsampling=2)
    rgb = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), np.float32) / 255.0
    if img.shape[2] == 4:
        return np.concatenate([rgb, img[..., 3:4]], axis=-1)
    return rgb


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
