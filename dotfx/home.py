"""The screen shown by `dotfx` with no arguments in a terminal."""

from __future__ import annotations

import sys

import numpy as np

from . import __version__, ui
from .term import BOLD, RESET, fg, size

COMMANDS = [
    ("render", "photo.jpg -p newsprint", "print photos and video (the default command)"),
    ("app", "photo.jpg", "the same in your browser: sliders, gallery, webcam, export"),
    ("tune", "photo.jpg", "adjust every setting live, then export or save a preset"),
    ("live", "", "webcam or video through a preset, in the terminal"),
    ("match", "reference.jpg", "measure a print you like and write a preset for it"),
    ("presets", "", "list the built-in looks"),
    ("gallery", "photo.jpg", "every preset on one contact sheet"),
    ("options", "", "every setting with its default"),
]
EXAMPLES = [
    ("dotfx photo.jpg -p sky-dither", "→ photo.sky-dither.png"),
    ("dotfx clip.mp4 -p comic", "→ clip.comic.mp4, audio kept"),
    ("dotfx photo.jpg -p newsprint -o poster.pdf", "vector dots for print"),
    ("dotfx photo.jpg -p comic --mask colorful --background bw-airy", "two looks in one image"),
]


def banner(width: int) -> list[str]:
    """'dotfx' set in type and printed through the halftone engine."""
    from PIL import Image, ImageDraw, ImageFont

    w, h = max(min(width, 48), 24), 9  # cells; two pixel rows per cell
    big = Image.new("L", (w * 8, h * 16), 0)
    draw = ImageDraw.Draw(big)
    def font_at(px: int):
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", px)
        except OSError:
            return ImageFont.load_default(size=px)

    # Largest size whose text fits 92% of the width and 90% of the height.
    px = int(h * 15)
    while px > 10:
        font = font_at(px)
        x0, y0, x1, y1 = draw.textbbox((0, 0), "dotfx", font=font)
        if x1 - x0 <= w * 8 * 0.92 and y1 - y0 <= h * 16 * 0.9:
            break
        px -= 4
    draw.text((w * 4, h * 8), "dotfx", fill=255, font=font, anchor="mm")
    mask = np.asarray(big.resize((w, h * 2), Image.Resampling.BOX), np.float32) / 255
    # Crisp letter shapes filled with a dot screen that fades from solid at the
    # top to small dots at the bottom.
    from .halftone import Screen

    fade = np.repeat(np.linspace(1.0, 0.2, h * 2, dtype=np.float32)[:, None], w, axis=1)
    scr = Screen(h * 2, w, 3.0, 45, spot="dot")
    cov = (mask * scr.coverage(scr.sample(fade, "center"), softness=0.7))[..., None]
    on = cov[..., 0] > 0.35
    # Draw ink only, on the terminal's own background (works on light themes too).
    ink = fg(ui.ACCENT)
    lines = []
    for y in range(0, h * 2, 2):
        row = []
        for x in range(w):
            top, bot = on[y, x], on[y + 1, x] if y + 1 < h * 2 else False
            row.append("█" if top and bot else "▀" if top else "▄" if bot else " ")
        lines.append(ink + "".join(row).rstrip() + RESET)
    return lines


def show() -> None:
    cols, _ = size()
    out = [""]
    try:
        out += ["  " + line for line in banner(cols - 4)]
    except Exception:  # noqa: BLE001 - the banner is decoration; never block help on it
        out.append(f"  {ui.logo()}")
    out.append(f"  {BOLD}halftone & dither prints for photos and video{RESET}  {fg(ui.MUTED)}v{__version__}{RESET}")
    out.append("")
    out.append(f"  {fg(ui.ACCENT)}{BOLD}COMMANDS{RESET}")
    for name, arg, desc in COMMANDS:
        usage = f"{BOLD}{name}{RESET} {fg(ui.MUTED)}{arg}{RESET}"
        pad = 30 - len(name) - 1 - len(arg)
        out.append(f"    {usage}{' ' * max(pad, 1)}{desc}")
    out.append("")
    out.append(f"  {fg(ui.ACCENT)}{BOLD}TRY{RESET}")
    for cmd, note in EXAMPLES:
        out.append(f"    {cmd}  {fg(ui.MUTED)}{note}{RESET}")
    out.append("")
    out.append(f"  {fg(ui.MUTED)}dotfx COMMAND -h for details · dotfx render -h lists every effect option{RESET}")
    out.append("")
    sys.stdout.write("\n".join(out) + "\n")
