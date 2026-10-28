"""Built-in looks. Each preset is a description plus a dict of Settings values;
anything given on the command line is applied on top."""

from __future__ import annotations

import difflib
import tomllib
from pathlib import Path

from .settings import Settings

PRESETS: dict[str, tuple[str, dict]] = {
    # --- reference 2: gray newspaper halftone -----------------------------------
    # Measured from the reference: 45° round dots, ~80 cells across the width,
    # neutral gray ink, dot gain in the highlights with an abrupt dropout,
    # mottled ink, grainy paper, light JPEG wear. Sizes use `cells`, so the look
    # is the same at any resolution.
    "newsprint": (
        "Reference look: gray 45° newspaper dots, dot gain, mottled ink, grainy paper",
        dict(
            effect="halftone", mode="mono", spot="dot", angle=45, cells=80,
            ink="#333333", paper="#f6f6f5", softness=1.1, dot_gain=1.5, min_dot=0.08,
            roughness=0.22, rough_scale=0.9, ink_noise=0.25, grain=0.4, speckle=0.3,
            post_blur=0.5, jpeg=75,
        ),
    ),
    "newsprint-clipping": (
        "Newsprint cut out like a clipping, with a soft shadow on white (as in the reference's top edge)",
        dict(
            effect="halftone", mode="mono", spot="dot", angle=45, cells=80,
            ink="#333333", paper="#f6f6f5", softness=1.1, dot_gain=1.5, min_dot=0.08,
            roughness=0.22, rough_scale=0.9, ink_noise=0.25, grain=0.4, speckle=0.3,
            post_blur=0.5, jpeg=75, clip_edge="cut", clip_margin=0.05, clip_rough=0.6,
            clip_shadow=0.6, backdrop="#ffffff",
        ),
    ),
    "newsprint-fine": (
        "Newsprint with a finer screen (140 dots across) for more detail",
        dict(
            effect="halftone", mode="mono", spot="dot", angle=45, cells=140,
            ink="#333333", paper="#f6f6f5", softness=1.0, dot_gain=1.4, min_dot=0.07,
            roughness=0.18, rough_scale=0.8, ink_noise=0.22, grain=0.35, speckle=0.25,
            post_blur=0.4, jpeg=80,
        ),
    ),
    "newsprint-coarse": (
        "Newsprint blown up: 45 big dots across, poster-like",
        dict(
            effect="halftone", mode="mono", spot="dot", angle=45, cells=45,
            ink="#333333", paper="#f6f6f5", softness=1.2, dot_gain=1.5, min_dot=0.08,
            roughness=0.3, rough_scale=1.2, ink_noise=0.3, grain=0.45, speckle=0.35,
            post_blur=0.6, jpeg=75, contrast=1.05,
        ),
    ),
    "newsprint-clean": (
        "The newsprint screen without wear: crisp dots on white",
        dict(
            effect="halftone", mode="mono", spot="dot", angle=45, cells=80,
            ink="#333333", paper="#ffffff", softness=1.0, dot_gain=1.5, min_dot=0.08,
        ),
    ),
    "newsprint-faded": (
        "Old yellowed clipping: washed-out ink, warm paper, heavy grain",
        dict(
            effect="halftone", mode="mono", spot="dot", angle=45, cells=80,
            ink="#55534f", paper="#eee8da", softness=1.2, dot_gain=1.4, min_dot=0.08,
            contrast=0.9, roughness=0.3, rough_scale=1.0, ink_noise=0.45, ink_opacity=0.9,
            grain=0.6, speckle=0.7, post_blur=0.6, vignette=0.25, jpeg=70,
        ),
    ),
    # --- reference 1: blue sky dither -------------------------------------------
    # Measured from the reference: a 4x4 Bayer dither ~206 cells across the width,
    # dark blue round dots over a light blue tint that only fills the darker
    # areas (sky), white elsewhere (clouds); red channel drives tone so blue sky
    # goes dark; soft, JPEG-compressed finish.
    "sky-dither": (
        "Reference look: blue Bayer dots over a light-blue sky tint, soft JPEG finish",
        dict(
            effect="dither", mode="mono", algorithm="bayer4", channel="red", cells=206,
            cell_shape="circle", dot_size=1.05, ink="#3a62a6", tint="#b0d6ff",
            tint_by="saturation", tint_start=0.25, tint_end=0.4, paper="#ffffff", contrast=1.1, gamma=0.92,
            post_blur=0.5, jpeg=55,
        ),
    ),
    "sky-dither-clean": (
        "The sky dither without blur or compression: crisp dots and tint",
        dict(
            effect="dither", mode="mono", algorithm="bayer4", channel="red", cells=206,
            cell_shape="circle", dot_size=1.05, ink="#3a62a6", tint="#b0d6ff",
            tint_by="saturation", tint_start=0.25, tint_end=0.4, paper="#ffffff", contrast=1.1, gamma=0.92,
        ),
    ),
    "sky-dither-mono": (
        "Single blue ink, no tint: dots on white everywhere",
        dict(
            effect="dither", mode="mono", algorithm="bayer4", channel="red", cells=206,
            cell_shape="circle", dot_size=1.1, ink="#4a74c4", paper="#ffffff",
            contrast=1.25, gamma=0.9, post_blur=0.4,
        ),
    ),
    "sky-dither-bold": (
        "Chunky sky dither: 110 cells across, same inks",
        dict(
            effect="dither", mode="mono", algorithm="bayer4", channel="red", cells=110,
            cell_shape="circle", dot_size=1.0, ink="#3a62a6", tint="#b0d6ff",
            tint_by="saturation", tint_start=0.25, tint_end=0.4, paper="#ffffff", contrast=1.1, gamma=0.92,
            post_blur=0.6, jpeg=60,
        ),
    ),
    # --- high-key black and white: paper dominates ----------------------------------
    # ink_limit caps the darkest tone, so even black areas stay mostly white.
    "bw-airy": (
        "High-key black dots on white: small round dots, white dominates",
        dict(
            effect="halftone", mode="mono", spot="dot", angle=45, cells=90, ink="#000000",
            paper="#ffffff", ink_limit=0.35, gamma=1.15, contrast=1.15, min_dot=0.05, softness=1.0,
        ),
    ),
    "bw-lines": (
        "High-key thin black lines on white",
        dict(
            effect="halftone", mode="mono", spot="line", angle=45, cells=110, ink="#000000",
            paper="#ffffff", ink_limit=0.35, gamma=1.15, contrast=1.15, min_dot=0.04,
        ),
    ),
    "bw-stipple": (
        "Sparse black stippling on white, scattered blue-noise dots",
        dict(
            effect="dither", mode="mono", algorithm="bluenoise", cells=180, cell_shape="circle",
            dot_size=0.8, ink="#000000", paper="#ffffff", ink_limit=0.45, contrast=1.15,
        ),
    ),
    "bw-grid": (
        "The sky-dither dot grid in black, airy and mostly white",
        dict(
            effect="dither", mode="mono", algorithm="bayer4", cells=206, cell_shape="circle",
            dot_size=0.85, ink="#000000", paper="#ffffff", ink_limit=0.45, contrast=1.15,
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
    close = difflib.get_close_matches(name_or_path, list(PRESETS), n=3, cutoff=0.6)
    hint = f" (did you mean {', '.join(close)}?)" if close else " (see: codename1 presets)"
    raise KeyError(f"unknown preset: {name_or_path}{hint}")


def to_toml(settings: Settings, only_changed: bool = True) -> str:
    lines = ["# codename1 preset", ""]
    values = settings.to_dict(only_changed=only_changed)
    # Always name the engine and mode, even at their defaults, for readability.
    head = {k: getattr(settings, k) for k in ("effect", "mode")}
    for key, value in {**head, **values}.items():
        if isinstance(value, bool):
            v = "true" if value else "false"
        elif isinstance(value, (int, float)):
            v = repr(value)
        else:
            v = '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'
        lines.append(f"{key} = {v}")
    return "\n".join(lines) + "\n"
