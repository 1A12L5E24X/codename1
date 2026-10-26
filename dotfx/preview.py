"""Show an image in the terminal with truecolor half-block characters."""

from __future__ import annotations

import os
import shutil
import sys

import numpy as np


def fit_resize(img: np.ndarray, tw: int, th: int) -> np.ndarray:
    """Downscale a uint8 RGB print without moire: blur away the dot pattern first."""
    from PIL import Image

    from .ops import gaussian_blur

    h, w = img.shape[:2]
    factor = min(w / max(tw, 1), h / max(th, 1))
    if factor > 1.5:
        img = gaussian_blur(img.astype(np.float32), 0.45 * factor)
        img = np.clip(img + 0.5, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(img).resize((tw, th), Image.Resampling.BOX))


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
    small = fit_resize(np.ascontiguousarray(img[..., :3]), tw, th)
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


def detect_protocol() -> str:
    env = os.environ
    term = env.get("TERM", "")
    prog = env.get("TERM_PROGRAM", "")
    if env.get("KITTY_WINDOW_ID") or "kitty" in term or prog.lower() == "ghostty" or "ghostty" in term:
        return "kitty"
    if prog in ("iTerm.app", "WezTerm") or env.get("LC_TERMINAL") == "iTerm2":
        return "iterm"
    return "blocks"


def _png_bytes(img: np.ndarray, max_side: int = 2000) -> bytes:
    import io

    from PIL import Image

    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    im = Image.fromarray(img)
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def inline_image(img: np.ndarray, protocol: str, stream=sys.stdout) -> None:
    """Show a full-resolution image with the kitty or iTerm2 graphics protocol."""
    import base64

    cols, _ = shutil.get_terminal_size((100, 40))
    data = base64.standard_b64encode(_png_bytes(img)).decode()
    if protocol == "iterm":
        size = len(data) * 3 // 4
        stream.write(f"\x1b]1337;File=inline=1;size={size};width={cols};preserveAspectRatio=1:{data}\a\n")
    else:
        chunks = [data[i:i + 4096] for i in range(0, len(data), 4096)]
        for i, chunk in enumerate(chunks):
            more = 1 if i < len(chunks) - 1 else 0
            head = f"a=T,f=100,c={cols},m={more}" if i == 0 else f"m={more}"
            stream.write(f"\x1b_G{head};{chunk}\x1b\\")
        stream.write("\n")
    stream.flush()


def show(img: np.ndarray, mode: str = "auto", stream=sys.stdout) -> None:
    protocol = detect_protocol() if mode == "auto" else mode
    if protocol in ("kitty", "iterm"):
        inline_image(img, protocol, stream)
    else:
        terminal_preview(img, stream=stream)
