"""Vector output (SVG / PDF): the same prints as real shapes instead of pixels.

Halftone dots are sized so the inked area of each cell equals its tone, like
the raster renderer. Raster-only texture (grain, roughness, bleed, blur, JPEG
wear, ...) has no vector equivalent and is left out.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field, replace

import numpy as np

from . import halftone as ht
from .adjust import adjust
from .color import parse_color, parse_palette
from .ops import gaussian_blur, resize, sample_bilinear, smoothstep
from .pipeline import dither_grid, lightness, prepare, tint_amount
from .settings import Settings

VECTOR_SPOTS = {"dot", "square", "diamond", "line", "ellipse"}


@dataclass
class Shape:
    kind: str  # circle | rect | poly | ellipse | ring
    data: tuple  # circle: (cx, cy, r); rect: (x, y, w, h, rx); poly: ((x, y), ...); ellipse: (cx, cy, rx, ry, deg)
    color: tuple[float, float, float] | None = None  # per-shape color, else the layer's


@dataclass
class VLayer:
    color: tuple[float, float, float]
    blend: str = "normal"  # normal | multiply | screen
    shapes: list[Shape] = field(default_factory=list)


@dataclass
class VDoc:
    width: int
    height: int
    paper: tuple[float, float, float] | None
    layers: list[VLayer]
    notes: list[str] = field(default_factory=list)


# --- halftone geometry -----------------------------------------------------------------

_R = np.linspace(0, np.sqrt(0.5), 2001)


def _clipped_circle_area(r: np.ndarray) -> np.ndarray:
    """Area of a circle of radius r (cell units) inside its unit cell."""
    a = np.pi * r * r
    big = r > 0.5
    rb = r[big]
    seg = rb * rb * np.arccos(0.5 / rb) - 0.5 * np.sqrt(rb * rb - 0.25)
    a[big] -= 4 * seg
    return a


_AREA = _clipped_circle_area(_R.copy())


def dot_radius(t: np.ndarray) -> np.ndarray:
    """Radius (cell units) whose circle covers fraction t of its cell. Points in a
    cell are nearest to that cell's centre, so this stays exact when neighbouring
    dots overlap."""
    return np.interp(np.clip(t, 0, 1), _AREA, _R)


def diamond_radius(t: np.ndarray) -> np.ndarray:
    """L1 radius of a diamond covering fraction t of its cell."""
    t = np.clip(t, 0, 1)
    return np.where(t <= 0.5, np.sqrt(t / 2), 1 - np.sqrt((1 - t) / 2))


def lattice(h: int, w: int, cell: float, angle: float, offset=(0.0, 0.0)):
    """Centres (x, y arrays, image pixels) of every screen cell touching the image."""
    th = np.deg2rad(angle)
    c, s = np.cos(th), np.sin(th)
    ox, oy = offset
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float64) - [ox, oy]
    u = (corners[:, 0] * c + corners[:, 1] * s) / cell
    v = (-corners[:, 0] * s + corners[:, 1] * c) / cell
    iu = np.arange(np.floor(u.min()) - 1, np.ceil(u.max()) + 1)
    iv = np.arange(np.floor(v.min()) - 1, np.ceil(v.max()) + 1)
    gu, gv = np.meshgrid((iu + 0.5) * cell, (iv + 0.5) * cell)
    x = gu * c - gv * s + ox
    y = gu * s + gv * c + oy
    keep = (x > -cell) & (x < w + cell) & (y > -cell) & (y < h + cell)
    return x[keep], y[keep]


def _sample(img: np.ndarray, x, y, cell: float, mode: str):
    src = gaussian_blur(img, cell * 0.3) if mode == "average" else img
    return sample_bilinear(src, x, y)


def _rot(px, py, cx, cy, deg):
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return cx + px * c - py * s, cy + px * s + py * c


def dot_shapes(x, y, tone, cell: float, angle: float, spot: str, colors=None) -> list[Shape]:
    out: list[Shape] = []
    for i in np.nonzero(tone > 1e-3)[0]:
        cx, cy, t = float(x[i]), float(y[i]), float(tone[i])
        col = tuple(float(v) for v in colors[i]) if colors is not None else None
        if t >= 0.999:
            # Solid: the whole (rotated) cell.
            h = cell / 2 + 0.05
            pts = [_rot(px, py, cx, cy, angle) for px, py in ((-h, -h), (h, -h), (h, h), (-h, h))]
            out.append(Shape("poly", tuple(pts), col))
        elif spot == "square":
            h = np.sqrt(t) * cell / 2
            pts = [_rot(px, py, cx, cy, angle) for px, py in ((-h, -h), (h, -h), (h, h), (-h, h))]
            out.append(Shape("poly", tuple(pts), col))
        elif spot == "diamond":
            r = float(diamond_radius(np.array(t))) * cell
            pts = [_rot(px, py, cx, cy, angle) for px, py in ((r, 0), (0, r), (-r, 0), (0, -r))]
            out.append(Shape("poly", tuple(pts), col))
        elif spot == "line":
            hw, hh = cell / 2 + 0.05, t * cell / 2
            pts = [_rot(px, py, cx, cy, angle) for px, py in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]
            out.append(Shape("poly", tuple(pts), col))
        elif spot == "ellipse":
            k = np.sqrt(min(t, 0.95) / (np.pi * 0.62))
            out.append(Shape("ellipse", (cx, cy, k * cell, 0.62 * k * cell, angle), col))
        else:
            out.append(Shape("circle", (cx, cy, float(dot_radius(np.array(t))) * cell), col))
    return out


# --- building the document ---------------------------------------------------------------


def build(img: np.ndarray, s: Settings) -> VDoc:
    s.validate()
    src = adjust(prepare(img, s), s)
    h, w = src.shape[:2]
    if s.cells > 0:
        s = replace(s, cell=max(w / s.cells, 1.0), pixel=max(w / s.cells, 1.0))
    paper = parse_color(s.paper)
    doc = VDoc(w, h, paper, [])
    texture = [n for n in ("roughness", "ink_noise", "bleed", "grain", "speckle", "vignette", "post_blur",
                           "glow", "jpeg") if getattr(s, n)]
    if texture or s.ink_opacity < 1:
        doc.notes.append("vector output leaves out raster texture: " + ", ".join(texture or ["ink_opacity"]))
    if s.effect == "halftone":
        _halftone(doc, src, s)
    else:
        _dither(doc, src, s)
    return doc


def _halftone(doc: VDoc, src: np.ndarray, s: Settings) -> None:
    h, w = src.shape[:2]
    spot = s.spot if s.spot in VECTOR_SPOTS else "dot"
    if spot != s.spot:
        doc.notes.append(f"spot '{s.spot}' is drawn as round dots in vector output")
    rng = np.random.default_rng(s.seed + 101)

    def plate(channel, angle, color, blend="normal", offset=(0.0, 0.0)):
        x, y = lattice(h, w, s.cell, angle, offset)
        tone = ht.tone_curve(_sample(channel, x, y, s.cell, s.sample), s)
        layer = VLayer(color, blend, dot_shapes(x, y, tone, s.cell, angle, spot))
        doc.layers.append(layer)

    def off():
        return tuple(rng.normal(0, s.misregister, 2)) if s.misregister > 0 else (0.0, 0.0)

    if s.mode in ("mono", "gray"):
        tint = parse_color(s.tint)
        if tint is not None:
            plate(tint_amount(src, s), s.angle, tint)
        plate(1 - lightness(src, s.channel), s.angle, parse_color(s.ink))
    elif s.mode == "color":
        p_rgb = np.array(doc.paper if doc.paper is not None else (1.0, 1.0, 1.0), np.float32)
        far = float(np.linalg.norm(np.maximum(p_rgb, 1 - p_rgb)))
        tone_src = np.linalg.norm(src - p_rgb, axis=-1) / max(far, 1e-3)
        x, y = lattice(h, w, s.cell, s.angle)
        tone = ht.tone_curve(_sample(tone_src, x, y, s.cell, s.sample), s)
        rgb = _sample(src, x, y, s.cell, s.sample)
        lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
        rgb = np.clip(lum[:, None] + (rgb - lum[:, None]) * 1.25, 0, 1)
        doc.layers.append(VLayer((0, 0, 0), "normal", dot_shapes(x, y, tone, s.cell, s.angle, spot, rgb)))
    elif s.mode == "cmyk":
        inks = parse_palette(s.cmyk_inks)
        angles = ht.parse_angles(s.angles, 4, s.angle)
        for chan, ang, ink in zip(ht.rgb_to_cmyk(src, s.gcr), angles, inks):
            plate(chan, ang, tuple(float(c) for c in ink), "multiply", off())
    elif s.mode == "duotone":
        dark = 1 - lightness(src, s.channel)
        plate(np.clip(dark * 0.85, 0, 1), s.angle + 30, parse_color(s.ink2), "multiply", off())
        plate(smoothstep(0.3, 1.0, dark), s.angle, parse_color(s.ink), "multiply", off())
    elif s.mode == "rgb":
        angles = ht.parse_angles(s.angles, 3, s.angle)
        for i, ang in enumerate(angles[:3]):
            color = tuple(1.0 if j == i else 0.0 for j in range(3))
            plate(src[..., i], ang, color, "screen", off())
    else:
        raise ValueError(f"mode {s.mode!r} is not supported by halftone")


def _dither(doc: VDoc, src: np.ndarray, s: Settings) -> None:
    h, w = src.shape[:2]
    p = float(s.pixel)
    sw, sh = max(1, round(w / p)), max(1, round(h / p))
    small = np.clip(resize(src, sw, sh, "box"), 0, 1)
    grid = dither_grid(small, s, np.random.default_rng(s.seed))
    pw, ph = w / sw, h / sh
    if grid.tint_on is not None:
        doc.layers.append(VLayer(parse_color(s.tint), "normal", _runs(grid.tint_on, None, pw, ph)))
    full_squares = s.cell_shape == "square" and s.dot_size >= 1 and not s.size_by_tone
    if grid.mono:
        ink = tuple(float(c) for c in grid.ink)
        if s.levels == 2:
            on = grid.tone > 0.5
            shapes = (_runs(on, None, pw, ph) if full_squares
                      else _cells(on, None, grid.tone, s, pw, ph))
            doc.layers.append(VLayer(ink, "normal", shapes))
        else:
            on = grid.tone > 1e-3
            doc.layers.append(VLayer(ink, "normal", _cells(on, grid.colors, grid.tone, s, pw, ph)))
    else:
        on = np.ones(grid.tone.shape, bool)
        shapes = (_runs(on, grid.colors, pw, ph) if full_squares
                  else _cells(on, grid.colors, grid.tone, s, pw, ph))
        doc.layers.append(VLayer((0, 0, 0), "normal", shapes))


def _runs(on: np.ndarray, colors, pw: float, ph: float) -> list[Shape]:
    """Merge horizontal runs of equal cells into rectangles (much smaller files)."""
    out = []
    for r in range(on.shape[0]):
        c = 0
        n = on.shape[1]
        while c < n:
            if not on[r, c]:
                c += 1
                continue
            start = c
            col = None if colors is None else tuple(np.round(colors[r, c], 4))
            c += 1
            while c < n and on[r, c] and (colors is None or tuple(np.round(colors[r, c], 4)) == col):
                c += 1
            out.append(Shape("rect", (start * pw, r * ph, (c - start) * pw + 0.02, ph + 0.02, 0.0),
                             None if col is None else tuple(float(v) for v in col)))
    return out


def _cells(on, colors, tone, s: Settings, pw: float, ph: float) -> list[Shape]:
    out = []
    for r, c in zip(*np.nonzero(on)):
        size = s.dot_size
        if s.size_by_tone:
            size *= float(np.sqrt(tone[r, c])) * 1.15
        cx, cy = (c + 0.5) * pw, (r + 0.5) * ph
        hw, hh = 0.5 * size * pw, 0.5 * size * ph
        col = None if colors is None else tuple(float(v) for v in colors[r, c])
        kind = s.cell_shape
        if kind == "circle":
            out.append(Shape("ellipse", (cx, cy, hw, hh, 0.0), col))
        elif kind == "diamond":
            out.append(Shape("poly", ((cx + hw, cy), (cx, cy + hh), (cx - hw, cy), (cx, cy - hh)), col))
        elif kind == "rounded":
            out.append(Shape("rect", (cx - hw, cy - hh, 2 * hw, 2 * hh, 0.45 * min(hw, hh)), col))
        elif kind == "plus":
            t = 0.38
            out.append(Shape("rect", (cx - hw, cy - hh * t, 2 * hw, 2 * hh * t, 0.0), col))
            out.append(Shape("rect", (cx - hw * t, cy - hh, 2 * hw * t, 2 * hh, 0.0), col))
        elif kind == "x":
            for a in (45, -45):
                pts = [_rot(px, py, cx, cy, a) for px, py in
                       ((-hw * 1.3, -hh * 0.3), (hw * 1.3, -hh * 0.3), (hw * 1.3, hh * 0.3), (-hw * 1.3, hh * 0.3))]
                out.append(Shape("poly", tuple(pts), col))
        elif kind == "hline":
            out.append(Shape("rect", (c * pw, cy - hh * 0.5, pw, hh, 0.0), col))
        elif kind == "vline":
            out.append(Shape("rect", (cx - hw * 0.5, r * ph, hw, ph, 0.0), col))
        elif kind == "ring":
            out.append(Shape("ring", (cx, cy, hw * 0.775, hw * 0.45), col))
        else:  # square
            out.append(Shape("rect", (cx - hw, cy - hh, 2 * hw, 2 * hh, 0.0), col))
    return out


# --- writers -------------------------------------------------------------------------------


def _hex(c) -> str:
    return "#" + "".join(f"{int(round(max(0.0, min(1.0, v)) * 255)):02x}" for v in c)


def _f(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


def to_svg(doc: VDoc) -> str:
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{doc.width}" height="{doc.height}" '
        f'viewBox="0 0 {doc.width} {doc.height}">',
        '<defs><clipPath id="page"><rect width="100%" height="100%"/></clipPath></defs>',
    ]
    if doc.paper is not None:
        out.append(f'<rect width="100%" height="100%" fill="{_hex(doc.paper)}"/>')
    out.append('<g clip-path="url(#page)">')
    for layer in doc.layers:
        style = "" if layer.blend == "normal" else f' style="mix-blend-mode:{layer.blend}"'
        out.append(f'<g fill="{_hex(layer.color)}"{style}>')
        for sh in layer.shapes:
            fill = f' fill="{_hex(sh.color)}"' if sh.color is not None else ""
            d = sh.data
            if sh.kind == "circle":
                out.append(f'<circle cx="{_f(d[0])}" cy="{_f(d[1])}" r="{_f(d[2])}"{fill}/>')
            elif sh.kind == "ellipse":
                rot = f' transform="rotate({_f(d[4])} {_f(d[0])} {_f(d[1])})"' if d[4] else ""
                out.append(f'<ellipse cx="{_f(d[0])}" cy="{_f(d[1])}" rx="{_f(d[2])}" ry="{_f(d[3])}"{rot}{fill}/>')
            elif sh.kind == "rect":
                rx = f' rx="{_f(d[4])}"' if d[4] else ""
                out.append(f'<rect x="{_f(d[0])}" y="{_f(d[1])}" width="{_f(d[2])}" height="{_f(d[3])}"{rx}{fill}/>')
            elif sh.kind == "poly":
                pts = " ".join(f"{_f(x)},{_f(y)}" for x, y in d)
                out.append(f'<polygon points="{pts}"{fill}/>')
            elif sh.kind == "ring":
                col = _hex(sh.color) if sh.color is not None else _hex(layer.color)
                out.append(f'<circle cx="{_f(d[0])}" cy="{_f(d[1])}" r="{_f(d[2])}" fill="none" '
                           f'stroke="{col}" stroke-width="{_f(d[3])}"/>')
        out.append("</g>")
    out.append("</g></svg>")
    return "\n".join(out) + "\n"


_K = 0.5522847498  # Bezier handle length for a quarter circle


def _pdf_ellipse(cx, cy, rx, ry, deg=0.0) -> str:
    pts = [(rx, 0), (rx, _K * ry), (_K * rx, ry), (0, ry), (-_K * rx, ry), (-rx, _K * ry), (-rx, 0),
           (-rx, -_K * ry), (-_K * rx, -ry), (0, -ry), (_K * rx, -ry), (rx, -_K * ry), (rx, 0)]
    p = [_rot(x, y, cx, cy, deg) for x, y in pts]
    s = [f"{_f(p[0][0])} {_f(p[0][1])} m"]
    for i in range(1, 13, 3):
        s.append(" ".join(f"{_f(x)} {_f(y)}" for x, y in p[i:i + 3]) + " c")
    return " ".join(s)


def _pdf_rect_round(x, y, w, h, r) -> str:
    if r <= 0:
        return f"{_f(x)} {_f(y)} {_f(w)} {_f(h)} re"
    k = r * (1 - _K)
    return (f"{_f(x + r)} {_f(y)} m {_f(x + w - r)} {_f(y)} l "
            f"{_f(x + w - k)} {_f(y)} {_f(x + w)} {_f(y + k)} {_f(x + w)} {_f(y + r)} c "
            f"{_f(x + w)} {_f(y + h - r)} l {_f(x + w)} {_f(y + h - k)} {_f(x + w - k)} {_f(y + h)} "
            f"{_f(x + w - r)} {_f(y + h)} c {_f(x + r)} {_f(y + h)} l {_f(x + k)} {_f(y + h)} {_f(x)} "
            f"{_f(y + h - k)} {_f(x)} {_f(y + h - r)} c {_f(x)} {_f(y + r)} l {_f(x)} {_f(y + k)} "
            f"{_f(x + k)} {_f(y)} {_f(x + r)} {_f(y)} c h")


def to_pdf(doc: VDoc, dpi: float = 96) -> bytes:
    """A one-page PDF; 1 image pixel = 1/dpi inch."""
    scale = 72.0 / dpi
    w_pt, h_pt = doc.width * scale, doc.height * scale
    ops = [f"q {_f(scale)} 0 0 {_f(-scale)} 0 {_f(h_pt)} cm", f"0 0 {doc.width} {doc.height} re W n"]
    if doc.paper is not None:
        ops.append("{} {} {} rg 0 0 {} {} re f".format(*(_f(c) for c in doc.paper), doc.width, doc.height))
    for layer in doc.layers:
        ops.append("q")
        if layer.blend in ("multiply", "screen"):
            ops.append(f"/{layer.blend.capitalize()} gs")
        current = None
        for sh in layer.shapes:
            color = sh.color if sh.color is not None else layer.color
            if color != current:
                ops.append("{} {} {} rg {} {} {} RG".format(*(_f(c) for c in color), *(_f(c) for c in color)))
                current = color
            d = sh.data
            if sh.kind == "circle":
                ops.append(_pdf_ellipse(d[0], d[1], d[2], d[2]) + " f")
            elif sh.kind == "ellipse":
                ops.append(_pdf_ellipse(*d) + " f")
            elif sh.kind == "rect":
                ops.append(_pdf_rect_round(*d) + " f")
            elif sh.kind == "poly":
                pts = list(d)
                ops.append(f"{_f(pts[0][0])} {_f(pts[0][1])} m " +
                           " ".join(f"{_f(x)} {_f(y)} l" for x, y in pts[1:]) + " h f")
            elif sh.kind == "ring":
                ops.append(f"{_f(d[3])} w " + _pdf_ellipse(d[0], d[1], d[2], d[2]) + " S")
        ops.append("Q")
    ops.append("Q")
    content = zlib.compress("\n".join(ops).encode(), 6)

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {_f(w_pt)} {_f(h_pt)}] /Contents 4 0 R "
         f"/Resources << /ExtGState << /Multiply 5 0 R /Screen 6 0 R >> >> >>").encode(),
        f"<< /Length {len(content)} /Filter /FlateDecode >>\nstream\n".encode() + content + b"\nendstream",
        b"<< /Type /ExtGState /BM /Multiply >>",
        b"<< /Type /ExtGState /BM /Screen >>",
    ]
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)
