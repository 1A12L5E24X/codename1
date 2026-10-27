"""Present the print as a paper clipping on a backdrop: a cut, torn or wavy
edge with a soft drop shadow."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .color import parse_color
from .ops import fbm_noise, gaussian_blur, smoothstep, value_noise


def _edge_distance(h: int, w: int, margin: float):
    """Signed distance to an inset rectangle (positive inside) and the position
    along the nearest side (for edge patterns that follow the perimeter)."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
    m = margin * min(h, w)
    left, right, top, bottom = xs - m, (w - m) - xs, ys - m, (h - m) - ys
    d = np.minimum(np.minimum(left, right), np.minimum(top, bottom))
    along = np.where(np.minimum(left, right) < np.minimum(top, bottom), ys, xs)
    return d, along


@lru_cache(maxsize=4)
def clip_mask(h: int, w: int, edge: str, margin: float, rough: float, seed: int):
    """(mask 0..1, rim 0..1): the clipping's coverage, and a band along torn
    edges where the paper's lighter core shows."""
    rng = np.random.default_rng([seed, 9091])
    size = min(h, w)
    d, along = _edge_distance(h, w, margin)
    amp = rough * size
    if edge == "torn":
        coarse = fbm_noise(h, w, size / 8, rng, octaves=5)
        fibre = value_noise(h, w, 1.5, rng)
        d = d + amp * (0.025 * coarse + 0.004 * fibre)
        rim = smoothstep(0.004 * size * (1 + rough), 0.0, d) * (0.6 + 0.4 * fibre)
    elif edge == "wavy":
        period = size / 14
        phase = rng.uniform(0, 2 * np.pi)
        d = d + amp * 0.012 * np.sin(2 * np.pi * along / period + phase)
        rim = None
    else:  # cut: scissor-like straight runs with small jogs between them
        jog = np.floor(along / (size / rng.uniform(4, 7)))
        offsets = rng.normal(0, 1, 64)
        d = d + amp * (0.006 * offsets[jog.astype(int) % 64] + 0.004 * value_noise(h, w, size / 3, rng))
        rim = None
    mask = smoothstep(-0.75, 0.75, d)
    if rim is not None:
        rim = np.clip(rim, 0, 1) * mask
    for a in (mask, rim):
        if a is not None:
            a.setflags(write=False)
    return mask, rim


def apply_clipping(out: np.ndarray, s) -> np.ndarray:
    h, w = out.shape[:2]
    mask, rim = clip_mask(h, w, s.clip_edge, float(s.clip_margin), float(s.clip_rough), s.seed)
    size = min(h, w)
    rgb = out[..., :3]
    if rim is not None:
        # Torn paper shows its lighter core along the tear.
        core = np.array(parse_color(s.paper) or (1.0, 1.0, 1.0), np.float32) * 0.15 + 0.85
        rgb = rgb + (core - rgb) * rim[..., None]
    # Shadow: the clipping's silhouette, offset down-right and blurred.
    shift = max(1, int(round(0.006 * size)))
    sil = np.zeros_like(mask)
    sil[shift:, shift:] = mask[:-shift, :-shift]
    shadow = gaussian_blur(sil, 0.012 * size) * s.clip_shadow * 0.45
    backdrop = parse_color(s.backdrop)
    m = mask[..., None]
    if out.shape[2] == 4:
        alpha = out[..., 3:4] * m
        return np.concatenate([rgb, alpha], axis=-1)
    if backdrop is None:
        # Transparent around the clipping; the shadow becomes semi-transparent black.
        alpha = m + (1 - m) * shadow[..., None]
        color = rgb * m / np.maximum(alpha, 1e-4)
        return np.concatenate([np.clip(color, 0, 1), alpha], axis=-1)
    back = np.array(backdrop, np.float32) * (1 - shadow[..., None])
    return (rgb * m + back * (1 - m)).astype(np.float32)
