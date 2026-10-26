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

from .ops import gaussian_blur, sample_bilinear


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
    """Geometry of one rotated halftone screen over an h x w canvas."""

    def __init__(self, h: int, w: int, cell: float, angle: float, offset=(0.0, 0.0)):
        self.h, self.w, self.cell = h, w, float(cell)
        th = np.deg2rad(angle)
        self.c, self.s = float(np.cos(th)), float(np.sin(th))
        self.ox, self.oy = offset
        ys, xs = np.mgrid[0:h, 0:w]
        xs = xs + 0.5 - self.ox
        ys = ys + 0.5 - self.oy
        u = (xs * self.c + ys * self.s) / self.cell
        v = (-xs * self.s + ys * self.c) / self.cell
        iu = np.floor(u)
        iv = np.floor(v)
        self.fu = (u - iu - 0.5).astype(np.float32)
        self.fv = (v - iv - 0.5).astype(np.float32)
        iu = iu.astype(np.int64)
        iv = iv.astype(np.int64)
        self.iu0, self.iv0 = int(iu.min()), int(iv.min())
        self.gi = iv - self.iv0
        self.gj = iu - self.iu0
        nu = int(iu.max()) - self.iu0 + 1
        nv = int(iv.max()) - self.iv0 + 1
        gu, gv = np.meshgrid(
            (np.arange(nu) + self.iu0 + 0.5) * self.cell,
            (np.arange(nv) + self.iv0 + 0.5) * self.cell,
        )
        # Cell centres back in image space.
        self.cx = gu * self.c - gv * self.s + self.ox
        self.cy = gu * self.s + gv * self.c + self.oy

    def sample(self, img: np.ndarray, mode: str = "average") -> np.ndarray:
        """Per-pixel copy of the image value at the centre of each pixel's cell."""
        src = gaussian_blur(img, self.cell * 0.42) if mode == "average" else img
        grid = sample_bilinear(src, self.cx, self.cy)
        return grid[self.gi, self.gj]

    def coverage(
        self,
        tone: np.ndarray,
        spot: str,
        softness: float = 1.0,
        rough: np.ndarray | None = None,
    ) -> np.ndarray:
        """Ink coverage (0..1) for per-pixel target tone (0 = no ink, 1 = solid)."""
        fu, fv = self.fu, self.fv
        r = spot_rank(spot, fu, fv)
        e = 0.5 / self.cell  # half a pixel, in cell units
        gx = spot_rank(spot, fu + e, fv) - spot_rank(spot, fu - e, fv)
        gy = spot_rank(spot, fu, fv + e) - spot_rank(spot, fu, fv - e)
        # Rank change per pixel; capped so discontinuities stay a 1px blur.
        g = np.minimum(np.sqrt(gx * gx + gy * gy), 0.5)
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
