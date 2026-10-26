"""Small numpy image primitives shared by the effects (float32, HxW or HxWxC)."""

from __future__ import annotations

import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def luminance(img: np.ndarray) -> np.ndarray:
    return img @ LUMA if img.ndim == 3 else img


def _axis_slice(a: np.ndarray, axis: int, start: int, stop: int) -> np.ndarray:
    idx = [slice(None)] * a.ndim
    idx[axis] = slice(start, stop)
    return a[tuple(idx)]


def box_blur(a: np.ndarray, radius: float, axis: int | None = None) -> np.ndarray:
    """Box blur with a (fractional) radius using cumulative sums; edges are clamped."""
    if radius <= 0:
        return a
    if axis is None:
        return box_blur(box_blur(a, radius, 0), radius, 1)
    r = int(np.floor(radius))
    frac = radius - r
    n = a.shape[axis]
    pad = r + 1
    widths = [(0, 0)] * a.ndim
    widths[axis] = (pad, pad)
    p = np.pad(a, widths, mode="edge")
    c = np.cumsum(p, axis=axis, dtype=np.float64)
    zshape = list(c.shape)
    zshape[axis] = 1
    c = np.concatenate([np.zeros(zshape), c], axis=axis)

    def window(k: int) -> np.ndarray:
        # Sum of the 2k+1 samples centred on each original position.
        return _axis_slice(c, axis, pad + k + 1, pad + k + 1 + n) - _axis_slice(c, axis, pad - k, pad - k + n)

    s = window(r)
    width = 2 * r + 1
    if frac > 0:
        # Blend in a fraction of the two pixels just outside the window.
        s = s + frac * (window(r + 1) - s)
        width = width + 2 * frac
    return (s / width).astype(np.float32)


def _gaussian_direct(a: np.ndarray, sigma: float) -> np.ndarray:
    rad = max(1, int(np.ceil(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-rad, rad + 1) / sigma) ** 2).astype(np.float32)
    k /= k.sum()
    out = a.astype(np.float32)
    for axis in (0, 1):
        n = out.shape[axis]
        widths = [(0, 0)] * out.ndim
        widths[axis] = (rad, rad)
        p = np.pad(out, widths, mode="edge")
        acc = k[0] * _axis_slice(p, axis, 0, n)
        for i in range(1, 2 * rad + 1):
            acc += k[i] * _axis_slice(p, axis, i, i + n)
        out = acc
    return out


def gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur: exact taps for small sigma, three box passes for large."""
    if sigma <= 0.05:
        return a
    if sigma <= 1.6:
        return _gaussian_direct(a, sigma)
    # Box radius whose 3-pass variance matches sigma^2: 3 * ((2r+1)^2 - 1) / 12.
    r = (np.sqrt(4 * sigma * sigma + 1) - 1) / 2
    out = a
    for _ in range(3):
        out = box_blur(out, r)
    return out


def sample_bilinear(a: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample a at pixel coordinates (x, y), pixel centres at integer + 0.5."""
    h, w = a.shape[:2]
    x = np.clip(x - 0.5, 0, w - 1)
    y = np.clip(y - 0.5, 0, h - 1)
    x0 = np.minimum(np.floor(x).astype(np.int64), w - 2 if w > 1 else 0)
    y0 = np.minimum(np.floor(y).astype(np.int64), h - 2 if h > 1 else 0)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0).astype(np.float32)
    fy = (y - y0).astype(np.float32)
    if a.ndim == 3:
        fx = fx[..., None]
        fy = fy[..., None]
    top = a[y0, x0] * (1 - fx) + a[y0, x1] * fx
    bot = a[y1, x0] * (1 - fx) + a[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def resize(img: np.ndarray, width: int, height: int, resample: str = "lanczos") -> np.ndarray:
    """Resize a float image via Pillow, channel by channel in 32-bit float mode."""
    from PIL import Image

    width, height = max(1, int(width)), max(1, int(height))
    if img.shape[1] == width and img.shape[0] == height:
        return img
    method = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "box": Image.Resampling.BOX,
        "lanczos": Image.Resampling.LANCZOS,
    }[resample]
    chans = [img] if img.ndim == 2 else [img[..., i] for i in range(img.shape[2])]
    out = [
        np.asarray(Image.fromarray(np.ascontiguousarray(c, dtype=np.float32)).resize((width, height), method))
        for c in chans
    ]
    res = out[0] if img.ndim == 2 else np.stack(out, axis=-1)
    return res.astype(np.float32)


def value_noise(h: int, w: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth noise in roughly [-1, 1] with features about `scale` pixels wide."""
    scale = max(scale, 0.5)
    gh = int(np.ceil(h / scale)) + 2
    gw = int(np.ceil(w / scale)) + 2
    grid = rng.standard_normal((gh, gw)).astype(np.float32)
    up = resize(grid, int(round(gw * scale)), int(round(gh * scale)), "bilinear")[:h, :w]
    if up.shape != (h, w):
        up = np.pad(up, ((0, h - up.shape[0]), (0, w - up.shape[1])), mode="edge")
    sd = float(up.std()) or 1.0
    return np.clip(up / (2.5 * sd), -1, 1)


def fbm_noise(h: int, w: int, scale: float, rng: np.random.Generator, octaves: int = 4) -> np.ndarray:
    """Fractal (multi-octave) noise in roughly [-1, 1]."""
    total = np.zeros((h, w), np.float32)
    amp, norm, s = 1.0, 0.0, scale
    for _ in range(octaves):
        total += amp * value_noise(h, w, s, rng)
        norm += amp
        amp *= 0.5
        s = max(s / 2, 0.5)
    return total / norm


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)
