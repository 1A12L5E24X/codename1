"""Tone and color adjustments applied before an effect."""

from __future__ import annotations

import numpy as np

from .ops import gaussian_blur, luminance


def adjust(img: np.ndarray, s) -> np.ndarray:
    """Apply the pre-processing stage of Settings `s` to an RGB float image."""
    out = img.astype(np.float32, copy=True)

    if s.denoise > 0:
        out = gaussian_blur(out, s.denoise)

    if s.auto_levels > 0:
        lum = luminance(out)
        lo, hi = np.percentile(lum, [s.auto_levels, 100 - s.auto_levels])
        if hi - lo > 1e-4:
            out = (out - lo) / (hi - lo)

    if s.black != 0 or s.white != 1:
        out = (out - s.black) / max(s.white - s.black, 1e-4)

    if s.exposure != 0:
        out = out * (2.0 ** s.exposure)

    if s.brightness != 0:
        out = out + s.brightness

    if s.contrast != 1:
        out = (out - 0.5) * s.contrast + 0.5

    out = np.clip(out, 0, 1)

    if s.shadows != 0 or s.highlights != 0:
        lum = luminance(out)[..., None]
        out = out + s.shadows * (1 - lum) ** 2 * 0.5 - s.highlights * lum ** 2 * 0.5
        out = np.clip(out, 0, 1)

    if s.gamma != 1:
        out = out ** (1.0 / max(s.gamma, 1e-3))

    if s.saturation != 1:
        lum = luminance(out)[..., None]
        out = np.clip(lum + (out - lum) * s.saturation, 0, 1)

    if s.sharpen > 0:
        blurred = gaussian_blur(out, max(s.sharpen_radius, 0.3))
        out = np.clip(out + (out - blurred) * s.sharpen, 0, 1)

    if s.blur > 0:
        out = gaussian_blur(out, s.blur)

    if s.invert:
        out = 1 - out

    if s.ink_limit < 1:
        # Compress toward white: black becomes (1 - ink_limit) lightness.
        out = 1 - (1 - out) * s.ink_limit

    return out
