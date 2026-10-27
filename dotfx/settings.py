"""All tunable look settings, described once and reused by the CLI and presets."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from typing import Any


def opt(default, help: str, group: str, choices=None, metavar=None):
    return field(default=default, metadata={"help": help, "group": group, "choices": choices, "metavar": metavar})


EFFECTS = ("halftone", "dither")
MODES = ("mono", "color", "cmyk", "rgb", "duotone", "palette", "gray")
SPOTS = ("euclid", "dot", "square", "diamond", "line", "ellipse", "cross", "ring", "wave", "grid")
ALGORITHMS = (
    "threshold", "random", "bayer2", "bayer4", "bayer8", "bayer16", "cluster4", "cluster8",
    "bluenoise", "lines", "floyd-steinberg", "false-floyd-steinberg", "atkinson", "jarvis",
    "stucki", "burkes", "sierra", "sierra2", "sierra-lite",
)
CELL_SHAPES = ("square", "circle", "diamond", "plus", "x", "hline", "vline", "ring", "rounded")


@dataclass
class Settings:
    # --- effect -------------------------------------------------------------
    effect: str = opt("halftone", "Which engine to use.", "effect", EFFECTS)
    mode: str = opt(
        "mono",
        "Color mode. mono: ink on paper. color: dots keep the photo's colors. "
        "cmyk: four rotated process screens (halftone). rgb: additive channel screens. "
        "duotone: shadows in --ink plus a full-range --ink2 plate (halftone). "
        "palette: dither to --palette (dither). gray: --levels shades between ink and paper.",
        "effect", MODES,
    )
    channel: str = opt(
        "luma",
        "What drives tone in mono/gray: luma, a single color channel (red darkens blue skies "
        "like a red filter), average, max or min.",
        "effect", ("luma", "red", "green", "blue", "average", "max", "min"),
    )
    seed: int = opt(7, "Random seed for noise, grain and random dithers.", "effect")

    # --- size ---------------------------------------------------------------
    width: int = opt(0, "Resize to this width before processing (0 = keep; height follows).", "size")
    height: int = opt(0, "Resize to this height before processing (0 = keep).", "size")
    scale: float = opt(1.0, "Scale factor applied to the input before processing.", "size")
    cells: int = opt(
        0, "Size dots relative to the image: this many halftone cells / dither pixels across "
        "the width. Overrides --cell and --pixel (0 = off).", "size",
    )

    # --- tone ---------------------------------------------------------------
    brightness: float = opt(0.0, "Add to brightness (-1..1).", "tone")
    contrast: float = opt(1.0, "Contrast multiplier around mid-gray.", "tone")
    gamma: float = opt(1.0, "Gamma (>1 brightens midtones).", "tone")
    exposure: float = opt(0.0, "Exposure in stops.", "tone")
    shadows: float = opt(0.0, "Lift (+) or crush (-) shadows.", "tone")
    highlights: float = opt(0.0, "Pull down (+) or push (-) highlights.", "tone")
    black: float = opt(0.0, "Black point (input level mapped to 0).", "tone")
    white: float = opt(1.0, "White point (input level mapped to 1).", "tone")
    auto_levels: float = opt(0.0, "Stretch levels, clipping this percent at each end (0 = off).", "tone")
    saturation: float = opt(1.0, "Saturation multiplier.", "tone")
    sharpen: float = opt(0.0, "Unsharp-mask amount.", "tone")
    sharpen_radius: float = opt(1.5, "Unsharp-mask radius in pixels.", "tone")
    blur: float = opt(0.0, "Gaussian blur before the effect (pixels).", "tone")
    denoise: float = opt(0.0, "Light blur applied first, to calm sensor noise (pixels).", "tone")
    invert: bool = opt(False, "Invert the image before the effect.", "tone")

    # --- colors -------------------------------------------------------------
    ink: str = opt("#111111", "Ink color (name, #hex or r,g,b).", "colors", metavar="COLOR")
    ink2: str = opt("riso-pink", "Second ink for duotone (prints the midtones).", "colors", metavar="COLOR")
    tint: str = opt(
        "none", "Mono/gray: a flat second color printed under the ink in the darker areas, "
        "sharing the dot pattern (e.g. a light blue under dark blue dots).", "colors", metavar="COLOR",
    )
    tint_by: str = opt(
        "tone", "What places the tint: tone (darker areas) or saturation (colorful areas, "
        "e.g. blue sky but not gray cloud shadows).", "colors", ("tone", "saturation"),
    )
    tint_start: float = opt(0.4, "Tone/saturation where the tint starts to appear (0..1).", "colors")
    tint_end: float = opt(0.75, "Tone/saturation where the tint is solid (0..1).", "colors")
    paper: str = opt("#ffffff", "Paper color, or 'none' for a transparent background.", "colors", metavar="COLOR")
    palette: str = opt("bw", "Palette for --mode palette: a name or a list like '#000 #f80 #fff'.", "colors", metavar="PAL")
    colors: int = opt(8, "Number of colors picked from the image for dither --mode color.", "colors")
    levels: int = opt(2, "Gray levels for dither (mono/gray) or levels per channel (rgb).", "colors")
    cmyk_inks: str = opt(
        "#00a4e4 #e2007a #ffe600 #1d1d1f", "Ink colors for the C, M, Y and K screens.", "colors", metavar="PAL"
    )

    # --- halftone -----------------------------------------------------------
    cell: float = opt(8.0, "Halftone cell size in pixels (screen frequency).", "halftone")
    angle: float = opt(45.0, "Screen angle in degrees.", "halftone")
    spot: str = opt("euclid", "Dot shape (spot function).", "halftone", SPOTS)
    angles: str = opt("15 75 0 45", "Screen angles for C M Y K (cmyk) or R G B (rgb).", "halftone", metavar="ANGLES")
    softness: float = opt(1.0, "Edge antialiasing width in pixels (0 = hard, aliased edges).", "halftone")
    dot_gain: float = opt(1.0, "Tone curve on dot area (>1 = bigger dots, darker print).", "halftone")
    min_dot: float = opt(0.0, "Coverage below this prints nothing (clean highlights).", "halftone")
    max_dot: float = opt(1.0, "Coverage above this prints solid (clean shadows).", "halftone")
    sample: str = opt("average", "How each cell reads the image: average or center.", "halftone", ("average", "center"))
    gcr: float = opt(0.8, "Gray component replacement for cmyk (0 = no black, 1 = full).", "halftone")
    misregister: float = opt(0.0, "Random plate offset in pixels for multi-ink screens.", "halftone")

    # --- dither -------------------------------------------------------------
    algorithm: str = opt("bayer4", "Dither algorithm.", "dither", ALGORITHMS)
    pixel: float = opt(2.0, "Dither pixel (cell) size in output pixels; fractions are allowed.", "dither")
    cell_shape: str = opt("square", "How each dither pixel is drawn.", "dither", CELL_SHAPES)
    dot_size: float = opt(1.0, "Size of the drawn cell shape relative to the cell (<1 leaves gaps).", "dither")
    size_by_tone: bool = opt(False, "Scale each cell's dot by its tone instead of switching colors.", "dither")
    diffusion: float = opt(1.0, "Error diffusion strength (0..1).", "dither")
    serpentine: bool = opt(True, "Alternate scan direction for error diffusion.", "dither")
    spread: float = opt(0.0, "Ordered dither spread (0 = automatic from the palette).", "dither")
    bias: float = opt(0.0, "Shift the dither threshold (-0.5..0.5, + = lighter).", "dither")

    # --- texture ------------------------------------------------------------
    roughness: float = opt(0.0, "Ragged, organic dot edges (0..1).", "texture")
    rough_scale: float = opt(1.5, "Feature size of the edge roughness in pixels.", "texture")
    ink_noise: float = opt(0.0, "Uneven ink density across the page (0..1).", "texture")
    ink_opacity: float = opt(1.0, "Ink opacity (0..1).", "texture")
    bleed: float = opt(0.0, "Ink spread into the paper in pixels.", "texture")
    grain: float = opt(0.0, "Paper grain / film noise amount (0..1).", "texture")
    grain_scale: float = opt(1.0, "Grain feature size in pixels.", "texture")
    speckle: float = opt(0.0, "Random paper flecks and dust (0..1).", "texture")
    glow: float = opt(0.0, "Light bleeding from bright areas, for LED and screen looks (0..1).", "texture")
    vignette: float = opt(0.0, "Darken toward the edges (0..1).", "texture")
    post_blur: float = opt(0.0, "Soften the finished print, like a scan of paper (pixels).", "texture")
    jpeg: int = opt(
        0, "Re-encode as a JPEG of this quality (1-100, 0 = off): blocky, color-bled web "
        "compression, like an image that's been saved and shared a few times.", "texture",
    )
    animate_grain: bool = opt(False, "Video: re-roll grain and roughness every frame.", "texture")

    # ------------------------------------------------------------------------
    def update(self, values: dict[str, Any]) -> "Settings":
        known = {f.name: f for f in fields(self)}
        for key, value in values.items():
            name = key.replace("-", "_")
            if name not in known:
                raise KeyError(f"unknown setting: {key}")
            setattr(self, name, coerce(known[name], value))
        return self

    def to_dict(self, only_changed: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if only_changed:
            base = Settings()
            d = {k: v for k, v in d.items() if getattr(base, k) != v}
        return d

    def validate(self) -> None:
        for f in fields(self):
            choices = f.metadata.get("choices")
            if choices and getattr(self, f.name) not in choices:
                raise ValueError(f"{f.name} must be one of: {', '.join(choices)}")
        from .color import parse_color, parse_palette

        for name in ("ink", "ink2", "paper", "tint"):
            try:
                value = parse_color(getattr(self, name))
            except ValueError as e:
                raise ValueError(f"{name}: {e}") from None
            if value is None and name not in ("paper", "tint"):
                raise ValueError(f"{name} can't be 'none' (only paper can be transparent)")
        for name in ("palette", "cmyk_inks"):
            try:
                parse_palette(getattr(self, name))
            except ValueError as e:
                raise ValueError(f"{name}: {e}") from None
        if self.tint_end <= self.tint_start:
            raise ValueError("tint_end must be greater than tint_start")
        if not 0 <= self.jpeg <= 100:
            raise ValueError("jpeg must be 0 (off) or a quality from 1 to 100")
        if self.cells < 0:
            raise ValueError("cells must be >= 0")
        if self.cell < 1:
            raise ValueError("cell must be >= 1")
        if self.pixel < 1:
            raise ValueError("pixel must be >= 1")
        if self.levels < 2:
            raise ValueError("levels must be >= 2")
        if self.effect == "halftone" and self.mode == "palette":
            raise ValueError("palette mode is for the dither effect; use mono/color/cmyk/rgb")
        if self.effect == "dither" and self.mode in ("cmyk", "duotone"):
            raise ValueError(f"{self.mode} mode is for the halftone effect")


def coerce(f, value):
    t = type(f.default)
    if t is bool:
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ("1", "true", "yes", "on"):
                return True
            if v in ("0", "false", "no", "off"):
                return False
            raise ValueError(f"{f.name}: expected a boolean, got {value!r}")
        return bool(value)
    if t is int:
        return int(float(value))
    if t is float:
        return float(value)
    return str(value)


def setting_fields():
    return fields(Settings)
