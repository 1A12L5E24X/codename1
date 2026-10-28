"""Shared look for the interactive screens (tune, live, home): palette, frames,
key chips and half-block images."""

from __future__ import annotations

import numpy as np

from .term import BOLD, RESET, bg, fg, fit, visible_len

# Tokyo Night-ish accents; body text stays the terminal's default color.
ACCENT = (122, 162, 247)
ACCENT_BG = (40, 52, 87)
CHANGED = (224, 175, 104)
GOOD = (158, 206, 106)
BAD = (247, 118, 142)
MUTED = (110, 118, 150)
LINE = (65, 72, 104)
CHIP = (52, 58, 86)
INK_DARK = (20, 22, 34)
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def logo(text: str = "dotfx") -> str:
    return f"{bg(ACCENT)}{fg(INK_DARK)}{BOLD} ◉ {text} {RESET}"


def chips(items, gap: str = " ") -> str:
    return gap.join(f"{bg(CHIP)}{fg((220, 224, 240))} {k} {RESET}{fg(MUTED)} {t}{RESET}" for k, t in items)


def chips_layout(items, x: int = 1, gap: int = 1) -> tuple[str, list[tuple[int, int, str]]]:
    """Key chips starting at column x, plus their click regions (col_from, col_to, action).
    items: (label, text, action) or (label, text, action, highlight)."""
    parts, regions = [], []
    for item in items:
        label, text, action = item[:3]
        hot = len(item) > 3 and item[3]
        chip_bg, chip_fg = (ACCENT, INK_DARK) if hot else (CHIP, (220, 224, 240))
        width = len(label) + 2 + 1 + len(text)
        parts.append(f"{bg(chip_bg)}{fg(chip_fg)}{BOLD if hot else ''} {label} {RESET}"
                     f"{fg(ACCENT if hot else MUTED)} {text}{RESET}")
        regions.append((x, x + width - 1, action))
        x += width + gap
    return (" " * gap).join(parts), regions


def box(x: int, y: int, w: int, h: int, title: str = "", right: str = "", color=LINE) -> list[str]:
    """Escape codes drawing a rounded frame at column x, row y (contents drawn separately)."""
    c = fg(color)
    t = f" {title} " if title else ""
    r = f" {right} " if right else ""
    fill = max(w - 3 - visible_len(t) - visible_len(r), 0)
    out = [f"\x1b[{y};{x}H{c}╭─{RESET}{BOLD}{t}{RESET}{c}{'─' * fill}{RESET}{fg(MUTED)}{r}{c}╮{RESET}"]
    for row in range(1, h - 1):
        out.append(f"\x1b[{y + row};{x}H{c}│{RESET}\x1b[{y + row};{x + w - 1}H{c}│{RESET}")
    out.append(f"\x1b[{y + h - 1};{x}H{c}╰{'─' * (w - 2)}╯{RESET}")
    return out


def half_blocks(img: np.ndarray, width: int, height: int) -> list[str]:
    """An RGB uint8 image (2 pixel rows per text row) as centred half-block lines
    filling a width x height cell area."""
    lines = []
    pad = max((width - img.shape[1]) // 2, 0)
    for y in range(0, img.shape[0] - 1, 2):
        t, b = img[y], img[y + 1]
        lines.append(" " * pad + "".join(
            f"\x1b[38;2;{a[0]};{a[1]};{a[2]}m\x1b[48;2;{c[0]};{c[1]};{c[2]}m▀" for a, c in zip(t, b)) + RESET)
    top = max((height - len(lines)) // 2, 0)
    return [""] * top + lines


def slider(value: float, lo: float, hi: float, n: int = 8) -> str:
    frac = float(np.clip((value - lo) / (hi - lo), 0, 1)) if hi > lo else 0.0
    k = min(int(round(frac * (n - 1))), n - 1)
    return f"{fg(ACCENT)}{'━' * k}●{fg(LINE)}{'─' * (n - 1 - k)}{RESET}"


def spread(left: str, right: str, width: int) -> str:
    """left … right, padded to width."""
    gap = width - visible_len(left) - visible_len(right)
    return fit(left + " " * max(gap, 1) + right, width)
