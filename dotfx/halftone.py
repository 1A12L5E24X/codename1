"""AM halftone screens: a rotated grid of cells, each printing a dot whose area
matches the tone of the image under it.

Each dot shape is a *spot function* over the unit cell. Ranking its values
turns it into a threshold whose area below any level t is exactly t, so dot
coverage is tone-accurate for every shape. Edges are antialiased from the
spot gradient, which keeps them crisp at any cell size or angle.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .ops import gaussian_blur, resize, sample_bilinear


def _wrap(a: np.ndarray) -> np.ndarray:
    return a - np.floor(a + 0.5)


def spot_value(spot: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spot function on cell coordinates in [-0.5, 0.5]; lower values ink first."""
    if spot == "euclid":
        ax, ay = np.abs(2 * x), np.abs(2 * y)
        inner = ax * ax + ay * ay
        outer = 2 - ((1 - ax) ** 2 + (1 - ay) ** 2)
        return np.where(ax + ay <= 1, inner, outer)
    if spot == "dot":
        return x * x + y * y
    if spot == "square":
        return np.maximum(np.abs(x), np.abs(y))
    if spot == "diamond":
        return np.abs(x) + np.abs(y)
    if spot == "line":
        return np.abs(y)
    if spot == "ellipse":
        return (x / 1.0) ** 2 + (y / 0.62) ** 2
    if spot == "cross":
        ax, ay = np.abs(x), np.abs(y)
        return np.minimum(ax, ay) + 0.2 * np.maximum(ax, ay)
    if spot == "ring":
        return np.abs(np.sqrt(x * x + y * y) - 0.3)
    if spot == "wave":
        return np.abs(_wrap(y - 0.18 * np.sin(2 * np.pi * x)))
    if spot == "grid":
        return np.minimum(np.abs(x), np.abs(y))
    raise ValueError(f"unknown spot: {spot}")


@lru_cache(maxsize=None)
def _rank_lut(spot: str) -> tuple[float, float, np.ndarray]:
    """Uniformly spaced lookup table from spot value to rank (area fraction)."""
    n = 384
    c = (np.arange(n) + 0.5) / n - 0.5
    x, y = np.meshgrid(c, c)
    vals = np.sort(spot_value(spot, x, y).ravel())
    q = np.linspace(0, 1, 4097)
    xp = np.quantile(vals, q)
    # Keep xp strictly increasing so interpolation is well defined on plateaus.
    xp = xp + np.linspace(0, 1e-9, xp.size)
    lo, hi = float(xp[0]), float(xp[-1])
    grid = np.linspace(lo, hi, 16385)
    lut = np.interp(grid, xp, q).astype(np.float32)
    return lo, (len(lut) - 1) / (hi - lo), np.append(lut, lut[-1])


def spot_rank(spot: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spot function remapped so that the area with rank < t is exactly t."""
    lo, scale, lut = _rank_lut(spot)
    f = np.clip((spot_value(spot, _wrap(x), _wrap(y)) - lo) * scale, 0, len(lut) - 2)
    i = f.astype(np.int32)
    a = lut[i]
    return a + (f - i).astype(np.float32) * (lut[i + 1] - a)


class Screen:
    """Geometry of one rotated halftone screen over an h x w canvas.

    Everything here is independent of the image, so screens are cached and
    reused across video frames (see get_screen).
    """

    def __init__(self, h: int, w: int, cell: float, angle: float, offset=(0.0, 0.0), spot: str = "euclid"):
        self.h, self.w, self.cell, self.spot = h, w, float(cell), spot
        th = np.deg2rad(angle)
        c, s = float(np.cos(th)), float(np.sin(th))
        ox, oy = offset
        xs = (np.arange(w, dtype=np.float64) + 0.5 - ox)[None, :]
        ys = (np.arange(h, dtype=np.float64) + 0.5 - oy)[:, None]
        u = (xs * c + ys * s) / self.cell
        v = (-xs * s + ys * c) / self.cell
        iu = np.floor(u)
        iv = np.floor(v)
        fu = (u - iu - 0.5).astype(np.float32)
        fv = (v - iv - 0.5).astype(np.float32)
        del u, v
        iu = iu.astype(np.int32)
        iv = iv.astype(np.int32)
        iu0, iv0 = int(iu.min()), int(iv.min())
        nu = int(iu.max()) - iu0 + 1
        nv = int(iv.max()) - iv0 + 1
        # Flat index of each pixel's cell in the (nv, nu) grid of cells.
        self.cell_index = ((iv - iv0) * nu + (iu - iu0)).astype(np.int32)
        del iu, iv
        gu, gv = np.meshgrid((np.arange(nu) + iu0 + 0.5) * self.cell, (np.arange(nv) + iv0 + 0.5) * self.cell)
        # Cell centres back in image space.
        self.cx = gu * c - gv * s + ox
        self.cy = gu * s + gv * c + oy

        self.rank = spot_rank(spot, fu, fv)
        e = np.float32(0.5 / self.cell)  # half a pixel, in cell units
        gx = spot_rank(spot, fu + e, fv) - spot_rank(spot, fu - e, fv)
        gy = spot_rank(spot, fu, fv + e) - spot_rank(spot, fu, fv - e)
        # Rank change per pixel; capped so discontinuities stay a 1px blur.
        self.grad = np.minimum(np.sqrt(gx * gx + gy * gy), 0.5).astype(np.float32)

    def sample(self, img: np.ndarray, mode: str = "average") -> np.ndarray:
        """Per-pixel copy of the image value at the centre of each pixel's cell."""
        cx, cy = self.cx, self.cy
        if mode == "average":
            # Average over roughly one cell; work at reduced size when cells are big.
            f = self.cell / 2.0
            src = img
            if f > 1.25:
                hh, ww = -(-img.shape[0] // f), -(-img.shape[1] // f)
                src = resize(img, ww, hh, "box")
                cx = cx * (ww / img.shape[1])
                cy = cy * (hh / img.shape[0])
            src = gaussian_blur(src, self.cell * 0.42 * src.shape[1] / img.shape[1])
        else:
            src = img
        grid = sample_bilinear(src, cx, cy)
        flat = grid.reshape(-1, *grid.shape[2:])
        return flat[self.cell_index]

    def coverage(self, tone: np.ndarray, softness: float = 1.0, rough: np.ndarray | None = None) -> np.ndarray:
        """Ink coverage (0..1) for per-pixel target tone (0 = no ink, 1 = solid)."""
        r, g = self.rank, self.grad
        if rough is not None:
            r = r + rough * (g + 0.02)
        if softness <= 0:
            cov = (r < tone).astype(np.float32)
        else:
            cov = np.clip((tone - r) / (g * softness + 1e-6) + 0.5, 0, 1)
        # A pixel can't hold more ink than its whole cell has (tiny dots), nor
        # less than the cell forces on it (tiny holes).
        area = self.cell * self.cell
        cov = np.minimum(cov, tone * area)
        cov = np.maximum(cov, 1 - (1 - tone) * area)
        return np.clip(cov, 0, 1).astype(np.float32)


@lru_cache(maxsize=8)
def get_screen(h: int, w: int, cell: float, angle: float, offset: tuple[float, float], spot: str) -> Screen:
    return Screen(h, w, cell, angle, offset, spot)


def tone_curve(t: np.ndarray, s) -> np.ndarray:
    if s.dot_gain != 1:
        t = 1 - (1 - np.clip(t, 0, 1)) ** max(s.dot_gain, 1e-3)
    if s.min_dot > 0:
        t = np.where(t < s.min_dot, 0, t)
    if s.max_dot < 1:
        t = np.where(t > s.max_dot, 1, t)
    return np.clip(t, 0, 1).astype(np.float32)


def parse_angles(text: str, n: int, base: float) -> list[float]:
    vals = [float(a) for a in text.replace(",", " ").split()]
    if not vals:
        vals = [base]
    while len(vals) < n:
        vals.append(vals[-1] + 30)
    return vals[:n]


def rgb_to_cmyk(img: np.ndarray, gcr: float):
    c, m, y = 1 - img[..., 0], 1 - img[..., 1], 1 - img[..., 2]
    k = np.minimum(np.minimum(c, m), y) * gcr
    denom = np.maximum(1 - k, 1e-4)
    return (c - k) / denom, (m - k) / denom, (y - k) / denom, k
