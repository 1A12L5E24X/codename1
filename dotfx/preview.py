"""Show an image in the terminal with truecolor half-block characters."""

from __future__ import annotations

import os
import shutil
import sys

import numpy as np


def terminal_preview(img: np.ndarray, max_width: int = 0, max_height: int = 0, stream=sys.stdout) -> None:
    from PIL import Image

    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        a = img[..., 3:4].astype(np.float32) / 255
        # Checkerboard behind transparent areas, like an image editor.
        yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        chk = np.where(((yy // 16 + xx // 16) % 2)[..., None] == 0, 200, 150).astype(np.float32)
        img = (img[..., :3] * a + chk * (1 - a)).astype(np.uint8)
    cols, rows = shutil.get_terminal_size((100, 40))
    max_width = max_width or cols
    max_height = max_height or max(rows - 3, 10)
    h, w = img.shape[:2]
    scale = min(max_width / w, (max_height * 2) / h, 1.0)
    tw, th = max(1, int(w * scale)), max(2, int(h * scale) // 2 * 2)
    # Box filter: averages the print's dots into the tones they represent.
    small = np.asarray(Image.fromarray(img[..., :3]).resize((tw, th), Image.Resampling.BOX))
    color = os.environ.get("NO_COLOR") is None
    out = []
    for y in range(0, th, 2):
        top, bot = small[y], small[y + 1]
        line = []
        for x in range(tw):
            t, b = top[x], bot[x]
            if color:
                line.append(f"\x1b[38;2;{t[0]};{t[1]};{t[2]}m\x1b[48;2;{b[0]};{b[1]};{b[2]}m▀")
            else:
                lum = (int(t.mean()) + int(b.mean())) // 2
                line.append(" .:-=+*#%@"[9 - lum * 10 // 256])
        out.append("".join(line) + ("\x1b[0m" if color else ""))
    stream.write("\n".join(out) + "\n")
    stream.flush()
