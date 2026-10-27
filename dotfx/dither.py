"""Dithering: ordered threshold maps and error diffusion, quantising to a set
of colors at a coarse "pixel" size."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

# --- threshold maps ------------------------------------------------------------


def bayer(n: int) -> np.ndarray:
    """n x n Bayer matrix (n a power of two) with values in (0, 1)."""
    m = np.zeros((1, 1), dtype=np.float64)
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return ((m + 0.5) / m.size).astype(np.float32)


def _rank_map(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values.ravel(), kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size)
    return ((ranks + 0.5) / values.size).reshape(values.shape).astype(np.float32)


def cluster(n: int) -> np.ndarray:
    """Clustered-dot ordered matrix: round dots on a 45 degree lattice."""
    from .halftone import spot_value

    ys, xs = np.mgrid[0:n, 0:n] + 0.5
    u = (xs + ys) / n
    v = (xs - ys) / n
    fu = u - np.floor(u) - 0.5
    fv = v - np.floor(v) - 0.5
    # Tiny tie-breaker so equal spot values grow in a stable spiral.
    vals = spot_value("euclid", fu, fv) + 1e-6 * np.arctan2(fv, fu)
    return _rank_map(vals)


def line_map(n: int = 6) -> np.ndarray:
    """Horizontal line screen: lines thicken from their centres."""
    centre = (n - 1) / 2
    order = np.abs(np.arange(n) - centre) + np.arange(n) * 1e-3
    col = _rank_map(order)
    return np.repeat(col[:, None], n, axis=1)


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "dotfx"


def _void_and_cluster(n: int, sigma: float = 1.5, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = np.minimum(np.arange(n), n - np.arange(n)).astype(np.float64)
    kernel = np.exp(-(d[:, None] ** 2 + d[None, :] ** 2) / (2 * sigma * sigma))

    def add(e, y, x, sign):
        e += sign * np.roll(np.roll(kernel, y, 0), x, 1)

    def energy(b):
        return np.real(np.fft.ifft2(np.fft.fft2(b) * np.fft.fft2(kernel)))

    total = n * n
    proto = rng.random((n, n)) < 0.1
    e = energy(proto.astype(np.float64))
    # Phase 0: relax the random seed pattern into evenly spaced points.
    for _ in range(total):
        y, x = np.unravel_index(np.argmax(np.where(proto, e, -np.inf)), e.shape)
        proto[y, x] = False
        add(e, y, x, -1)
        vy, vx = np.unravel_index(np.argmin(np.where(proto, np.inf, e)), e.shape)
        proto[vy, vx] = True
        add(e, vy, vx, 1)
        if (vy, vx) == (y, x):
            break
    ranks = np.zeros((n, n), dtype=np.int64)
    ones = int(proto.sum())
    # Phase 1: rank the seed points by removing the tightest clusters.
    b = proto.copy()
    e1 = e.copy()
    for r in range(ones - 1, -1, -1):
        y, x = np.unravel_index(np.argmax(np.where(b, e1, -np.inf)), e1.shape)
        b[y, x] = False
        add(e1, y, x, -1)
        ranks[y, x] = r
    # Phase 2: fill the largest voids until every position has a rank.
    b = proto.copy()
    for r in range(ones, total):
        y, x = np.unravel_index(np.argmin(np.where(b, np.inf, e)), e.shape)
        b[y, x] = True
        add(e, y, x, 1)
        ranks[y, x] = r
    return ((ranks + 0.5) / total).astype(np.float32)


@lru_cache(maxsize=None)
def blue_noise(n: int = 64) -> np.ndarray:
    path = _cache_dir() / f"bluenoise{n}.npy"
    try:
        m = np.load(path)
        if m.shape == (n, n):
            return m
    except (OSError, ValueError):
        pass
    m = _void_and_cluster(n)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, m)
    except OSError:
        pass
    return m


def threshold_map(algorithm: str) -> np.ndarray | None:
    if algorithm.startswith("bayer"):
        return bayer(int(algorithm[5:]))
    if algorithm.startswith("cluster"):
        return cluster(int(algorithm[7:]))
    if algorithm == "bluenoise":
        return blue_noise(64)
    if algorithm == "lines":
        return line_map(6)
    if algorithm == "threshold":
        return np.full((1, 1), 0.5, dtype=np.float32)
    return None


def tile(m: np.ndarray, h: int, w: int) -> np.ndarray:
    th, tw = m.shape
    return np.tile(m, (-(-h // th), -(-w // tw)))[:h, :w]


# --- error diffusion -----------------------------------------------------------

KERNELS: dict[str, tuple[float, list[tuple[int, int, float]]]] = {
    "floyd-steinberg": (16, [(1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)]),
    "false-floyd-steinberg": (8, [(1, 0, 3), (0, 1, 3), (1, 1, 2)]),
    # Atkinson only spreads 6/8 of the error, which keeps highlights open.
    "atkinson": (8, [(1, 0, 1), (2, 0, 1), (-1, 1, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1)]),
    "jarvis": (48, [(1, 0, 7), (2, 0, 5), (-2, 1, 3), (-1, 1, 5), (0, 1, 7), (1, 1, 5), (2, 1, 3),
                    (-2, 2, 1), (-1, 2, 3), (0, 2, 5), (1, 2, 3), (2, 2, 1)]),
    "stucki": (42, [(1, 0, 8), (2, 0, 4), (-2, 1, 2), (-1, 1, 4), (0, 1, 8), (1, 1, 4), (2, 1, 2),
                    (-2, 2, 1), (-1, 2, 2), (0, 2, 4), (1, 2, 2), (2, 2, 1)]),
    "burkes": (32, [(1, 0, 8), (2, 0, 4), (-2, 1, 2), (-1, 1, 4), (0, 1, 8), (1, 1, 4), (2, 1, 2)]),
    "sierra": (32, [(1, 0, 5), (2, 0, 3), (-2, 1, 2), (-1, 1, 4), (0, 1, 5), (1, 1, 4), (2, 1, 2),
                    (-1, 2, 2), (0, 2, 3), (1, 2, 2)]),
    "sierra2": (16, [(1, 0, 4), (2, 0, 3), (-2, 1, 1), (-1, 1, 2), (0, 1, 3), (1, 1, 2), (2, 1, 1)]),
    "sierra-lite": (4, [(1, 0, 2), (-1, 1, 1), (0, 1, 1)]),
}


def _diffuse_py(img, palette, dx, dy, wt, serpentine, strength, out, bias):
    """Reference error diffusion. img: HxWxC (modified), palette: KxC, out: HxW.
    bias (HxW) shifts only the decision at each pixel, never the diffused error."""
    h, w, ch = img.shape
    k = palette.shape[0]
    n = wt.shape[0]
    for y in range(h):
        rev = serpentine and (y % 2 == 1)
        for i in range(w):
            x = w - 1 - i if rev else i
            best = 0
            bd = 1e30
            b = bias[y, x]
            for p in range(k):
                d = 0.0
                for c in range(ch):
                    t = img[y, x, c] + b - palette[p, c]
                    d += t * t
                if d < bd:
                    bd = d
                    best = p
            out[y, x] = best
            for c in range(ch):
                err = (img[y, x, c] - palette[best, c]) * strength
                if err == 0.0:
                    continue
                for j in range(n):
                    xx = x - dx[j] if rev else x + dx[j]
                    yy = y + dy[j]
                    if 0 <= xx < w and yy < h:
                        v = img[yy, xx, c] + err * wt[j]
                        # Clamp runaway error so saturated regions recover fast.
                        img[yy, xx, c] = min(max(v, -0.5), 1.5)


def _diffuse_lists(img, palette, dx, dy, wt, serpentine, strength, out, bias):
    """Pure-Python fallback on nested lists (~20x faster than indexing numpy)."""
    h, w, ch = img.shape
    rows = [img[y].tolist() for y in range(h)]
    brows = bias.tolist()
    pal = palette.tolist()
    taps = list(zip(dx.tolist(), dy.tolist(), wt.tolist()))
    k = len(pal)
    gray_levels = None
    if ch == 1:
        levels = [p[0] for p in pal]
        if levels == sorted(levels):
            gray_levels = levels
    result = [[0] * w for _ in range(h)]
    for y in range(h):
        rev = serpentine and (y % 2 == 1)
        row = rows[y]
        xs = range(w - 1, -1, -1) if rev else range(w)
        for x in xs:
            px = row[x]
            b = brows[y][x]
            if gray_levels is not None:
                v = px[0] + b
                best = 0
                bd = abs(v - gray_levels[0])
                for p in range(1, k):
                    d = abs(v - gray_levels[p])
                    if d < bd:
                        bd, best = d, p
            else:
                best, bd = 0, 1e30
                for p in range(k):
                    q = pal[p]
                    d = 0.0
                    for c in range(ch):
                        t = px[c] + b - q[c]
                        d += t * t
                    if d < bd:
                        bd, best = d, p
            result[y][x] = best
            q = pal[best]
            for c in range(ch):
                err = (px[c] - q[c]) * strength
                if err == 0.0:
                    continue
                for ddx, ddy, ww in taps:
                    xx = x - ddx if rev else x + ddx
                    yy = y + ddy
                    if 0 <= xx < w and yy < h:
                        cell = rows[yy][xx]
                        v = cell[c] + err * ww
                        cell[c] = -0.5 if v < -0.5 else (1.5 if v > 1.5 else v)
    out[...] = np.array(result, dtype=out.dtype)


_DIFFUSE = None


def _diffuse_fast(*args):
    """Error diffusion with numba when installed (compiled once, cached on
    disk), else the pure-Python version. numba is imported on first use only,
    since importing it is slow."""
    global _DIFFUSE
    if _DIFFUSE is None:
        try:
            import numba

            _DIFFUSE = numba.njit(cache=True, nogil=True)(_diffuse_py)
        except Exception:  # noqa: BLE001 - any numba/llvmlite failure means fall back
            _DIFFUSE = _diffuse_lists
    return _DIFFUSE(*args)


def error_diffuse(img: np.ndarray, palette: np.ndarray, algorithm: str, serpentine=True, strength=1.0,
                  bias: np.ndarray | None = None) -> np.ndarray:
    """Return palette indices (HxW) for img (HxWxC) by error diffusion."""
    div, taps = KERNELS[algorithm]
    dx = np.array([t[0] for t in taps], dtype=np.int64)
    dy = np.array([t[1] for t in taps], dtype=np.int64)
    wt = np.array([t[2] / div for t in taps], dtype=np.float64)
    work = np.ascontiguousarray(img, dtype=np.float64)
    out = np.zeros(img.shape[:2], dtype=np.int64)
    if bias is None:
        bias = np.zeros(img.shape[:2])
    _diffuse_fast(work, np.ascontiguousarray(palette, dtype=np.float64), dx, dy, wt, bool(serpentine),
                  float(strength), out, np.ascontiguousarray(bias, dtype=np.float64))
    return out


# --- quantisation --------------------------------------------------------------


def nearest(img: np.ndarray, palette: np.ndarray) -> np.ndarray:
    d = ((img[..., None, :] - palette[None, None, :, :]) ** 2).sum(-1)
    return np.argmin(d, axis=-1)


def default_spread(palette: np.ndarray) -> float:
    """Ordered-dither amplitude: roughly the gap between neighbouring colors."""
    if len(palette) < 2:
        return 1.0
    d = np.sqrt(((palette[:, None, :] - palette[None, :, :]) ** 2).sum(-1))
    d[d == 0] = np.inf
    gaps = d.min(axis=1)
    gaps = gaps[np.isfinite(gaps)]
    return float(np.median(gaps) / np.sqrt(palette.shape[1])) if gaps.size else 1.0


def quantize(img: np.ndarray, palette: np.ndarray, algorithm: str, s, rng: np.random.Generator) -> np.ndarray:
    """Dither img (HxWxC) to palette (KxC); returns indices HxW."""
    h, w = img.shape[:2]
    if algorithm in KERNELS:
        biased = img + s.bias
        anchor = None
        if s.temporal > 0:
            # Anchor decisions to a fixed blue-noise pattern: where the image
            # doesn't change, the same pixels win from frame to frame instead of
            # the error rerouting chaotically. Tone is unaffected (the nudge
            # never enters the diffused error).
            spread = default_spread(palette)
            anchor = s.temporal * spread * (tile(blue_noise(64), h, w) - 0.5)
        return error_diffuse(biased, palette, algorithm, s.serpentine, s.diffusion, anchor)
    if algorithm == "random":
        t = rng.random((h, w), dtype=np.float32)
    else:
        t = tile(threshold_map(algorithm), h, w)
    spread = s.spread if s.spread > 0 else default_spread(palette)
    shifted = img + (spread * (t - 0.5) + s.bias)[..., None]
    if palette.shape[1] == 1:
        # Gray ramp: snap within the bracketing pair of levels (exact for any spread).
        levels = palette[:, 0]
        if np.all(np.diff(levels) > 0):
            v = shifted[..., 0]
            idx = np.searchsorted(levels, v)
            idx = np.clip(idx, 1, len(levels) - 1)
            lo, hi = levels[idx - 1], levels[idx]
            return np.where(v - lo < hi - v, idx - 1, idx)
    return nearest(shifted, palette)
