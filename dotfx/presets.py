"""Built-in looks. Each preset is a description plus a dict of Settings values;
anything given on the command line is applied on top."""

from __future__ import annotations

import tomllib
from pathlib import Path

from .settings import Settings

PRESETS: dict[str, tuple[str, dict]] = {
    "newsprint": (
        "Gray 45° round-dot screen on off-white stock, ragged edges and paper grain",
        dict(
            effect="halftone", mode="mono", cell=9.5, angle=45, spot="dot",
            ink="#3d3d3f", paper="#f5f5f2", softness=1.4, contrast=1.3, gamma=1.1,
            min_dot=0.05, roughness=0.3, rough_scale=1.2, ink_noise=0.15, bleed=0.25,
            grain=0.35, speckle=0.4, sharpen=0.4, post_blur=0.45,
        ),
    ),
    "sky-dither": (
        "Blue 4x4 Bayer dither drawn as a grid of dots on white",
        dict(
            effect="dither", mode="mono", algorithm="bayer4", pixel=4, channel="red",
            cell_shape="circle", dot_size=1.18, ink="#5b84d8", paper="#ffffff",
            contrast=1.35, gamma=0.9,
        ),
    ),
    "comic": (
        "Bold CMYK pop-art dots on warm paper",
        dict(
            effect="halftone", mode="cmyk", cell=11, spot="dot", paper="#fff7e3",
            saturation=1.45, contrast=1.2, gcr=0.6, softness=1.0, grain=0.15,
        ),
    ),
    "cmyk-print": (
        "Fine four-color process print with slight misregistration",
        dict(
            effect="halftone", mode="cmyk", cell=5.5, spot="euclid", paper="#fbf8f1",
            gcr=0.8, misregister=0.9, roughness=0.15, grain=0.2, ink_opacity=0.95,
            saturation=1.15,
        ),
    ),
    "riso": (
        "Single-color fluorescent-pink risograph with uneven ink",
        dict(
            effect="halftone", mode="mono", cell=6, angle=22.5, spot="euclid",
            ink="riso-pink", paper="#f6efd9", roughness=0.4, ink_noise=0.35,
            ink_opacity=0.92, grain=0.5, speckle=0.5, contrast=1.2,
        ),
    ),
    "riso-duo": (
        "Two-color risograph: blue shadows over a pink plate",
        dict(
            effect="halftone", mode="duotone", cell=7, angle=15, spot="euclid",
            ink="riso-blue", ink2="riso-pink", paper="#f6efd9", roughness=0.35,
            ink_noise=0.3, ink_opacity=0.9, misregister=1.2, grain=0.45, speckle=0.4,
            contrast=1.15,
        ),
    ),
    "color-dots": (
        "Dots in the photo's own colors on black, like an LED sign",
        dict(effect="halftone", mode="color", cell=10, spot="dot", angle=0, paper="#0b0b0c", saturation=1.2, glow=0.25),
    ),
    "rgb-screen": (
        "Additive red/green/blue screens on black",
        dict(effect="halftone", mode="rgb", cell=7, spot="dot", angles="15 45 75", paper="#000000"),
    ),
    "line-screen": (
        "Diagonal line screen, like an old banknote",
        dict(effect="halftone", mode="mono", cell=6, angle=45, spot="line", ink="#1e2a3a", paper="#f2efe6", sharpen=0.6),
    ),
    "engraving": (
        "Wavy engraved lines in sepia ink",
        dict(
            effect="halftone", mode="mono", cell=5, angle=15, spot="wave", ink="#2a1e14",
            paper="#f3ead6", sharpen=0.8, contrast=1.2, roughness=0.15, grain=0.25,
        ),
    ),
    "blueprint": (
        "Pale ink dots on blueprint blue",
        dict(
            effect="halftone", mode="mono", cell=7, angle=45, spot="dot", ink="#e8f0ff",
            paper="blueprint", invert=True, grain=0.25, roughness=0.2,
        ),
    ),
    "crosshatch": (
        "Cross-shaped dots growing into a hatched grid",
        dict(effect="halftone", mode="mono", cell=8, angle=30, spot="cross", ink="#202020", paper="#f7f4ec", contrast=1.1),
    ),
    "stipple": (
        "Blue-noise stippling with round dots",
        dict(
            effect="dither", mode="mono", algorithm="bluenoise", pixel=3, cell_shape="circle",
            dot_size=1.1, ink="#141414", paper="#f8f6f0", contrast=1.1,
        ),
    ),
    "bitmap": (
        "Classic 1-bit Atkinson dither, crisp black and white",
        dict(effect="dither", mode="mono", algorithm="atkinson", pixel=2, ink="#000000", paper="#ffffff"),
    ),
    "xerox": (
        "Blown-out photocopy: hard threshold, toner noise, dust",
        dict(
            effect="dither", mode="mono", algorithm="threshold", pixel=1, contrast=1.7, sharpen=1.0,
            roughness=0.6, grain=0.5, speckle=1.0, ink="#121212", paper="#f4f4f1", bias=0.05,
        ),
    ),
    "gameboy": (
        "Four-shade green handheld palette, ordered dither",
        dict(effect="dither", mode="palette", palette="gameboy", algorithm="bayer4", pixel=4, contrast=1.1),
    ),
    "cga": (
        "Cyan/magenta/white CGA palette",
        dict(effect="dither", mode="palette", palette="cga", algorithm="bayer4", pixel=3, contrast=1.2, saturation=1.3),
    ),
    "pico8": (
        "PICO-8 sixteen colors with Floyd–Steinberg",
        dict(effect="dither", mode="palette", palette="pico8", algorithm="floyd-steinberg", pixel=4, saturation=1.2),
    ),
    "phosphor": (
        "Green-screen terminal glow, ordered dither on black",
        dict(
            effect="dither", mode="gray", levels=4, algorithm="bayer8", pixel=3, cell_shape="rounded",
            dot_size=0.85, ink="phosphor", paper="#031207", invert=True, contrast=1.2, glow=0.35,
            vignette=0.4,
        ),
    ),
    "led-wall": (
        "Chunky RGB LED matrix",
        dict(
            effect="dither", mode="rgb", levels=4, algorithm="bayer4", pixel=8, cell_shape="circle",
            dot_size=0.82, paper="#050505", saturation=1.3, glow=0.4,
        ),
    ),
    "posterized": (
        "Image-derived 6 color palette with blue-noise dither",
        dict(effect="dither", mode="color", colors=6, algorithm="bluenoise", pixel=2),
    ),
}


def load_preset(name_or_path: str) -> dict:
    """Return the settings dict for a preset name or a .toml file."""
    if name_or_path in PRESETS:
        return dict(PRESETS[name_or_path][1])
    p = Path(name_or_path).expanduser()
    if p.suffix.lower() == ".toml" or p.exists():
        if not p.exists():
            raise FileNotFoundError(f"preset file not found: {p}")
        with open(p, "rb") as f:
            data = tomllib.load(f)
        if "base" in data:
            base = load_preset(str(data.pop("base")))
            base.update(data)
            data = base
        return data
    close = [n for n in PRESETS if n.startswith(name_or_path[:3])]
    hint = f" (did you mean {', '.join(close)}?)" if close else ""
    raise KeyError(f"unknown preset: {name_or_path}{hint}")


def to_toml(settings: Settings, only_changed: bool = True) -> str:
    lines = ["# dotfx preset", ""]
    for key, value in settings.to_dict(only_changed=only_changed).items():
        if isinstance(value, bool):
            v = "true" if value else "false"
        elif isinstance(value, (int, float)):
            v = repr(value)
        else:
            v = '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'
        lines.append(f"{key} = {v}")
    return "\n".join(lines) + "\n"
