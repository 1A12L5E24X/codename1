"""Color parsing and named palettes."""

from __future__ import annotations

import re

import numpy as np

NAMED = {
    "black": "#000000",
    "white": "#ffffff",
    "paper": "#f4f1ea",
    "newsprint": "#efece4",
    "cream": "#f6efd9",
    "ink": "#1d1d1f",
    "graphite": "#3b3b3d",
    "red": "#e0312b",
    "green": "#2bb04a",
    "blue": "#2f5fd0",
    "sky": "#5b84d8",
    "navy": "#14244a",
    "cyan": "#00a4e4",
    "magenta": "#e2007a",
    "yellow": "#ffe600",
    "orange": "#ff7a1a",
    "pink": "#ff48b0",
    "purple": "#6a3fc2",
    "teal": "#1a8f8a",
    "gray": "#808080",
    "grey": "#808080",
    "riso-pink": "#ff48b0",
    "riso-blue": "#0078bf",
    "riso-teal": "#00838a",
    "riso-yellow": "#ffe800",
    "riso-red": "#ff665e",
    "blueprint": "#1d4f91",
    "phosphor": "#33ff66",
    "amber": "#ffb000",
}

PALETTES = {
    "bw": ["#000000", "#ffffff"],
    "gameboy": ["#0f380f", "#306230", "#8bac0f", "#9bbc0f"],
    "gameboy-pocket": ["#000000", "#545454", "#a9a9a9", "#ffffff"],
    "cga": ["#000000", "#55ffff", "#ff55ff", "#ffffff"],
    "cga-warm": ["#000000", "#55ff55", "#ff5555", "#ffff55"],
    "pico8": [
        "#000000", "#1d2b53", "#7e2553", "#008751", "#ab5236", "#5f574f",
        "#c2c3c7", "#fff1e8", "#ff004d", "#ffa300", "#ffec27", "#00e436",
        "#29adff", "#83769c", "#ff77a8", "#ffccaa",
    ],
    "c64": [
        "#000000", "#ffffff", "#880000", "#aaffee", "#cc44cc", "#00cc55",
        "#0000aa", "#eeee77", "#dd8855", "#664400", "#ff7777", "#333333",
        "#777777", "#aaff66", "#0088ff", "#bbbbbb",
    ],
    "sepia": ["#2b1d0e", "#6b4b2a", "#a8835a", "#e0c9a6", "#f8f0e0"],
    "sky": ["#ffffff", "#5b84d8"],
    "blueprint": ["#1d4f91", "#e8f0ff"],
    "riso": ["#f6efd9", "#ff48b0", "#0078bf", "#1d1d1f"],
    "vaporwave": ["#2b0f54", "#ab1f65", "#ff4f69", "#ff8b3d", "#ffd36e", "#5ff5ff"],
    "cmyk": ["#ffffff", "#00a4e4", "#e2007a", "#ffe600", "#000000"],
    "phosphor": ["#031207", "#0d4a1c", "#1fa33f", "#33ff66"],
    "amber": ["#140a00", "#663f00", "#c27a00", "#ffb000"],
    "grayscale4": ["#000000", "#555555", "#aaaaaa", "#ffffff"],
}

_HEX = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")


def parse_color(value: str) -> tuple[float, float, float] | None:
    """Parse a color into an RGB float triple in [0, 1].

    Accepts names, #rgb, #rrggbb, "r,g,b" (0-255) and "none"/"transparent",
    which returns None.
    """
    v = value.strip().lower()
    if v in ("none", "transparent"):
        return None
    v = NAMED.get(v, v)
    m = _HEX.match(v)
    if m:
        h = m.group(1)
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]
    parts = [p for p in re.split(r"[,\s]+", v) if p]
    if len(parts) == 3:
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            pass
        else:
            scale = 255.0 if max(nums) > 1.0 else 1.0
            return tuple(min(max(n / scale, 0.0), 1.0) for n in nums)  # type: ignore[return-value]
    raise ValueError(f"unrecognised color: {value!r}")


def parse_palette(value: str) -> np.ndarray:
    """A named palette, or a list of colors separated by spaces/semicolons/commas."""
    key = value.strip().lower()
    if key in PALETTES:
        colors = PALETTES[key]
    else:
        colors = [c for c in re.split(r"[;\s]+", value) if c]
        # "#000,#fff,red" -> split on commas unless it's a single "r,g,b" triple
        if len(colors) == 1 and "," in colors[0]:
            items = colors[0].split(",")
            if not all(re.fullmatch(r"[\d.]+", i.strip()) for i in items):
                colors = items
    rgb = []
    for c in colors:
        parsed = parse_color(c)
        if parsed is None:
            raise ValueError("palettes cannot contain 'none'")
        rgb.append(parsed)
    if len(rgb) < 2:
        raise ValueError(f"palette needs at least 2 colors: {value!r}")
    return np.array(rgb, dtype=np.float32)


def auto_palette(img: np.ndarray, n: int) -> np.ndarray:
    """Pick n representative colors from an RGB float image (median cut)."""
    from PIL import Image

    small = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    small.thumbnail((256, 256))
    q = small.quantize(colors=max(2, n), method=Image.Quantize.MEDIANCUT)
    pal = np.array(q.getpalette()[: 3 * max(2, n)], dtype=np.float32).reshape(-1, 3) / 255.0
    return pal


def to_hex(rgb) -> str:
    return "#" + "".join(f"{int(round(c * 255)):02x}" for c in rgb)
