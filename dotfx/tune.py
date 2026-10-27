"""Interactive tuner: tweak settings with the keyboard and watch the print
update in the terminal."""

from __future__ import annotations

import shlex
import sys
import threading
import time
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

from . import term
from .color import PALETTES, parse_color
from .pipeline import prepare, render, to_uint8
from .presets import PRESETS, load_preset, to_toml
from .preview import detect_protocol, fit_resize
from .settings import Settings
from .term import BOLD, ITALIC, RESET, bg, fg, fit

STEPS = {
    "cell": 0.5, "angle": 5.0, "pixel": 0.25, "ink_limit": 0.05, "jpeg": 5, "cells": 10, "levels": 1,
    "colors": 1, "seed": 1, "width": 50, "height": 50, "scale": 0.05, "contrast": 0.05, "gamma": 0.05,
    "exposure": 0.1, "brightness": 0.02, "saturation": 0.05, "sharpen": 0.1, "sharpen_radius": 0.25,
    "blur": 0.25, "denoise": 0.25, "softness": 0.1, "dot_gain": 0.05, "min_dot": 0.01, "max_dot": 0.01,
    "gcr": 0.05, "misregister": 0.1, "dot_size": 0.05, "diffusion": 0.05, "spread": 0.02, "bias": 0.02,
    "roughness": 0.05, "rough_scale": 0.25, "ink_noise": 0.05, "ink_opacity": 0.05, "bleed": 0.05,
    "grain": 0.05, "grain_scale": 0.25, "speckle": 0.1, "vignette": 0.05, "post_blur": 0.05,
    "tint_start": 0.02, "tint_end": 0.02, "glow": 0.05, "black": 0.01, "white": 0.01, "auto_levels": 0.5,
    "shadows": 0.05, "highlights": 0.05, "temporal": 0.05, "clip_shadow": 0.05, "clip_rough": 0.05,
}
# Ranges for the little slider bars (values outside still work).
RANGES = {
    "cell": (2, 40), "cells": (20, 400), "pixel": (1, 16), "angle": (0, 90), "contrast": (0, 2.5),
    "gamma": (0.3, 2.5), "exposure": (-3, 3), "brightness": (-0.5, 0.5), "saturation": (0, 2.5),
    "sharpen": (0, 3), "sharpen_radius": (0, 5), "blur": (0, 10), "denoise": (0, 5), "dot_gain": (0.3, 3),
    "softness": (0, 4), "misregister": (0, 5), "dot_size": (0.2, 1.6), "bias": (-0.5, 0.5),
    "rough_scale": (0.5, 5), "bleed": (0, 3), "grain_scale": (0.5, 5), "post_blur": (0, 3), "jpeg": (0, 100),
    "levels": (2, 16), "colors": (2, 32), "scale": (0.1, 2), "shadows": (-1, 1), "highlights": (-1, 1),
    "auto_levels": (0, 10), "speckle": (0, 3), "temporal": (0, 2),
}
NO_BAR = {"seed", "width", "height"}
# Values offered when cycling free-text settings with the arrow keys.
SUGGEST = {
    "ink": ["#111111", "#3a3a3c", "sky", "riso-pink", "riso-blue", "riso-teal", "red", "navy",
            "phosphor", "amber", "#e8f0ff", "white"],
    "tint": ["none", "#a9cdf8", "#cfe0ff", "#ffd6e8", "#d9d9d9", "riso-yellow"],
    "ink2": ["riso-pink", "riso-yellow", "riso-teal", "orange", "#ffb000", "cyan", "magenta"],
    "paper": ["#ffffff", "#f6f5f1", "cream", "newsprint", "#0b0b0c", "black", "blueprint", "none"],
    "palette": list(PALETTES),
    "angles": ["15 75 0 45", "105 75 90 15", "0 0 0 0", "15 45 75"],
    "cmyk_inks": ["#00a4e4 #e2007a #ffe600 #1d1d1f", "#00aeef #ec008c #fff200 #000000"],
    "clip_edge": ["none", "torn", "cut", "wavy"],
}
EFFECT_GROUPS = {"halftone": "dither", "dither": "halftone"}  # group hidden for each effect
ZOOMS = [0, 4, 2, 1, 0.5]  # 0 = fit; otherwise image pixels per terminal pixel
ZOOM_NAMES = ["fit", "¼", "½", "1:1", "2×"]

# Palette (Tokyo Night-ish); accents only, body text stays the terminal default.
ACCENT = (122, 162, 247)
ACCENT_BG = (40, 52, 87)
CHANGED = (224, 175, 104)
GOOD = (158, 206, 106)
BAD = (247, 118, 142)
MUTED = (110, 118, 150)
LINE = (65, 72, 104)
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

KEYS_HELP = [
    ("↑ ↓", "select setting"), ("← →", "adjust"), ("[ ]  ⇧←→", "adjust ×10"), ("enter", "type a value"),
    ("space", "toggle / next"), ("u", "reset to preset"), ("p P", "next / prev preset"),
    ("z", "zoom fit ¼ ½ 1:1 2×"), ("w a s d", "pan when zoomed"), (", .", "video: step frame"),
    ("< >", "video: jump 5 s"), ("o", "export image"), ("S", "save preset .toml"),
    ("c", "show command line"), ("g", "sharp/blocky preview"), ("q esc", "quit"),
]


def _fmt(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


class Tuner:
    def __init__(self, img: np.ndarray, settings: Settings, src: Path, base: str | None, output: Path,
                 save: Path, video=None, preview_mode: str = "auto"):
        self.img = img
        self.s = settings
        self.src = src
        self.base = base
        self.output = output
        self.save_path = save
        self.video = video  # media.VideoInfo or None
        self.t = 0.0
        self.sel = 0
        self.zoom = 0
        self.pan = [0.5, 0.5]
        self.status = ""
        self.status_kind = "info"
        self.note = ""
        self.out: np.ndarray | None = None
        self.render_ms = 0.0
        self.error = ""
        self.help = False
        self.preset_names = list(PRESETS)
        mode = detect_protocol() if preview_mode == "auto" else preview_mode
        self.graphics = mode if mode in ("kitty", "iterm") else "blocks"
        self._lock = threading.Lock()
        self._gen = 0
        self._wake = threading.Event()
        self._rendering = False
        self._stop = False
        self._img_key = None
        self._vp_key = None
        self._vp_lines: list[str] = []
        self._frame_counter = 0

    # -- data -----------------------------------------------------------------
    def visible_fields(self):
        hidden = EFFECT_GROUPS.get(self.s.effect)
        return [f for f in fields(Settings) if f.metadata["group"] != hidden]

    def base_settings(self) -> Settings:
        return Settings().update(load_preset(self.base)) if self.base else Settings()

    def request_render(self) -> None:
        with self._lock:
            self._gen += 1
        self._wake.set()

    def _worker(self) -> None:
        src_key, src = None, None
        while not self._stop:
            self._wake.wait(0.2)
            if self._stop:
                return
            if not self._wake.is_set():
                continue
            self._wake.clear()
            with self._lock:
                gen = self._gen
                s = replace(self.s)
                img = self.img
            self._rendering = True
            t0 = time.monotonic()
            try:
                s.validate()
                key = (id(img), s.width, s.height, s.scale)
                if key != src_key:
                    src, src_key = prepare(img, s), key
                out = to_uint8(render(src, replace(s, width=0, height=0, scale=1.0)))
                err = ""
            except (ValueError, KeyError) as e:
                out, err = None, str(e)
            ms = (time.monotonic() - t0) * 1000
            with self._lock:
                if out is not None:
                    self.out = out
                self.error = err
                self.render_ms = ms
                if gen == self._gen:
                    self._rendering = False

    def set_status(self, text: str, kind: str = "info") -> None:
        self.status, self.status_kind = text, kind

    def adjust(self, direction: int, big: bool = False) -> None:
        f = self.visible_fields()[self.sel]
        cur = getattr(self.s, f.name)
        choices = f.metadata.get("choices")
        if isinstance(cur, bool):
            val = not cur
        elif choices:
            val = choices[(list(choices).index(cur) + direction) % len(choices)]
        elif isinstance(cur, (int, float)):
            step = STEPS.get(f.name, 1 if isinstance(cur, int) else 0.05) * (10 if big else 1)
            val = cur + direction * step
            val = int(round(val)) if isinstance(cur, int) else round(val, 4)
            minimum = {"cell": 1, "pixel": 0.25, "ink_limit": 0.05, "levels": 2, "colors": 2}.get(f.name)
            if minimum is None and f.name not in ("angle", "brightness", "exposure", "shadows", "highlights",
                                                    "bias", "seed"):
                minimum = 0
            if minimum is not None:
                val = max(val, minimum)
            val = type(cur)(val)
        else:
            options = SUGGEST.get(f.name, [])
            if not options:
                return
            i = options.index(cur) if cur in options else -1
            val = options[(i + direction) % len(options)]
        with self._lock:
            setattr(self.s, f.name, val)
        self.request_render()

    def load_preset(self, step: int) -> None:
        idx = self.preset_names.index(self.base) if self.base in self.preset_names else -1
        self.base = self.preset_names[(idx + step) % len(self.preset_names)]
        keep = {"width": self.s.width, "height": self.s.height, "scale": self.s.scale}
        with self._lock:
            self.s = Settings().update(load_preset(self.base)).update(keep)
        self.sel = min(self.sel, len(self.visible_fields()) - 1)
        self.set_status(f"{self.base}: {PRESETS[self.base][0]}")
        self.request_render()

    def seek(self, dt: float) -> None:
        if self.video is None:
            return
        from .media import frame_at

        dur = self.video.duration or 0
        self.t = float(np.clip(self.t + dt, 0, max(dur - 1 / max(self.video.fps, 1), 0)))
        frame = frame_at(self.src, self.video, self.t)
        if frame is not None:
            with self._lock:
                self.img = frame
            self.request_render()

    def command_line(self) -> str:
        base = self.base_settings()
        parts = ["dotfx", shlex.quote(str(self.src))]
        if self.base:
            parts += ["-p", shlex.quote(self.base)]
        for f in fields(Settings):
            v = getattr(self.s, f.name)
            if v == getattr(base, f.name):
                continue
            flag = "--" + f.name.replace("_", "-")
            if isinstance(v, bool):
                parts.append(flag if v else "--no-" + f.name.replace("_", "-"))
            else:
                parts += [flag, shlex.quote(str(v))]
        return " ".join(parts)

    # -- drawing --------------------------------------------------------------
    def viewport_image(self, cols: int, rows: int) -> np.ndarray | None:
        """The part of the render to show, at terminal-pixel size (2 per cell row)."""
        with self._lock:
            img = self.out
        if img is None:
            return None
        from PIL import Image

        if img.shape[2] == 4:
            a = img[..., 3:4].astype(np.float32) / 255
            img = (img[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
        h, w = img.shape[:2]
        th = rows * 2
        z = ZOOMS[self.zoom]
        if z == 0:
            scale = min(cols / w, th / h)
            return fit_resize(img, max(1, int(w * scale)), max(2, int(h * scale)))
        vw, vh = int(cols * z), int(th * z)
        x0 = int(np.clip(self.pan[0] * w - vw / 2, 0, max(w - vw, 0)))
        y0 = int(np.clip(self.pan[1] * h - vh / 2, 0, max(h - vh, 0)))
        crop = img[y0:y0 + vh, x0:x0 + vw]
        tw, tt = max(1, int(crop.shape[1] / z)), max(2, int(crop.shape[0] / z))
        if z > 1:
            return fit_resize(np.ascontiguousarray(crop), tw, tt)  # smoothed: no moire
        return np.asarray(Image.fromarray(crop).resize((tw, tt), Image.Resampling.NEAREST))

    def sharp_crop(self, cols: int, rows: int) -> tuple[np.ndarray, int, int] | None:
        """Full-resolution region for graphics terminals, and its size in cells."""
        with self._lock:
            img = self.out
        if img is None:
            return None
        img = img[..., :3]
        h, w = img.shape[:2]
        z = ZOOMS[self.zoom]
        if z == 0:
            crop = img
        else:
            # Assume cells about 1:2; at 1:1 one cell shows ~8x16 image pixels.
            vw, vh = int(cols * 8 * z / 2), int(rows * 16 * z / 2)
            x0 = int(np.clip(self.pan[0] * w - vw / 2, 0, max(w - vw, 0)))
            y0 = int(np.clip(self.pan[1] * h - vh / 2, 0, max(h - vh, 0)))
            crop = img[y0:y0 + vh, x0:x0 + vw]
        ch, cw = crop.shape[:2]
        # Fit into the box keeping aspect, with cells twice as tall as wide.
        c = min(cols, int(rows * 2 * cw / max(ch, 1)))
        r = min(rows, max(1, int(round(c * ch / max(cw, 1) / 2))))
        return crop, max(c, 1), r

    def field_value(self, f, width: int) -> str:
        v = getattr(self.s, f.name)
        choices = f.metadata.get("choices")
        if isinstance(v, bool):
            return f"{fg(GOOD)}● on{RESET}" if v else f"{fg(MUTED)}○ off{RESET}"
        if choices:
            return f"{fg(MUTED)}‹{RESET} {v} {fg(MUTED)}›{RESET}"
        if f.name in ("ink", "ink2", "paper", "tint"):
            try:
                rgb = parse_color(str(v))
            except ValueError:
                rgb = None
            if rgb is None:
                return f"{fg(MUTED)}░░{RESET} {v}"
            return f"{fg(tuple(int(c * 255) for c in rgb))}██{RESET} {v}"
        if isinstance(v, (int, float)) and f.name not in NO_BAR:
            lo, hi = RANGES.get(f.name, (0, 1))
            n = 8
            frac = float(np.clip((v - lo) / (hi - lo), 0, 1)) if hi > lo else 0
            k = min(int(round(frac * (n - 1))), n - 1)
            bar = f"{fg(ACCENT)}{'━' * k}●{fg(LINE)}{'─' * (n - 1 - k)}{RESET}"
            return f"{_fmt(v):<6.6} {bar}"
        return str(v)

    def settings_lines(self, width: int, height: int) -> list[str]:
        flist = self.visible_fields()
        self.sel = max(0, min(self.sel, len(flist) - 1))
        base = self.base_settings()
        rows: list[tuple[str, int]] = []
        group = None
        sel_row = 0
        for i, f in enumerate(flist):
            if f.metadata["group"] != group:
                group = f.metadata["group"]
                rows.append(("hdr", i))
            if i == self.sel:
                sel_row = len(rows)
            rows.append(("field", i))
        top = max(0, min(sel_row - height // 2, len(rows) - height))
        lines = []
        name_w = 15
        for kind, i in rows[top:top + height]:
            f = flist[i]
            if kind == "hdr":
                title = f.metadata["group"].upper()
                rule = "─" * max(width - len(title) - 3, 0)
                lines.append(fit(f" {fg(ACCENT)}{BOLD}{title}{RESET} {fg(LINE)}{rule}{RESET}", width))
                continue
            changed = getattr(self.s, f.name) != getattr(base, f.name)
            mark = f"{fg(CHANGED)}•{RESET}" if changed else " "
            name = f.name.replace("_", "-")
            value = self.field_value(f, width - name_w - 4)
            if i == self.sel:
                row = f"{fg(ACCENT)}▌{RESET}{bg(ACCENT_BG)}{fg((235, 238, 255))}{mark}{bg(ACCENT_BG)}" \
                      f"{fg((235, 238, 255))}{BOLD}{name:<{name_w}}{RESET}{bg(ACCENT_BG)} "
                lines.append(fit(row + value.replace(RESET, RESET + bg(ACCENT_BG)), width) + RESET)
            else:
                lines.append(fit(f" {mark}{name:<{name_w}} {value}", width))
        return lines + [" " * width] * (height - len(lines))

    def box(self, x: int, y: int, w: int, h: int, title: str = "", right: str = "") -> list[str]:
        """Escape codes drawing a rounded frame (contents drawn separately)."""
        c = fg(LINE)
        t = f" {title} " if title else ""
        r = f" {right} " if right else ""
        fill = max(w - 2 - term.visible_len(t) - term.visible_len(r) - 1, 0)
        out = [f"\x1b[{y};{x}H{c}╭─{RESET}{BOLD}{t}{RESET}{c}{'─' * fill}{RESET}{fg(MUTED)}{r}{c}╮{RESET}"]
        for row in range(1, h - 1):
            out.append(f"\x1b[{y + row};{x}H{c}│{RESET}\x1b[{y + row};{x + w - 1}H{c}│{RESET}")
        out.append(f"\x1b[{y + h - 1};{x}H{c}╰{'─' * (w - 2)}╯{RESET}")
        return out

    def header(self, cols: int) -> str:
        logo = f"{bg(ACCENT)}{fg((20, 22, 34))}{BOLD} ◉ dotfx {RESET}"
        name = f" {BOLD}{self.src.name}{RESET}"
        preset = f"  {fg(MUTED)}preset{RESET} {fg(ACCENT)}{self.base or 'custom'}{RESET}"
        with self._lock:
            out = self.out
            rendering, ms = self._rendering, self.render_ms
        dims = f"  {fg(MUTED)}{out.shape[1]}×{out.shape[0]}{RESET}" if out is not None else ""
        if rendering:
            spin = SPIN[self._frame_counter % len(SPIN)]
            state = f"{fg(ACCENT)}{spin} rendering{RESET}"
        else:
            state = f"{fg(GOOD)}✓{RESET} {fg(MUTED)}{ms:.0f} ms{RESET}"
        left = logo + name + preset + dims
        space = cols - term.visible_len(left) - term.visible_len(state) - 1
        return left + " " * max(space, 1) + state + " "

    def timeline(self, cols: int) -> str:
        v = self.video
        dur = max(v.duration, 1e-6)
        label_l = f" {fg(MUTED)}▶{RESET} {self.t:5.1f}s "
        label_r = f" {dur:5.1f}s  {fg(MUTED)}, . frame  < > 5 s{RESET} "
        n = max(cols - term.visible_len(label_l) - term.visible_len(label_r), 10)
        k = int(round(self.t / dur * (n - 1)))
        bar = f"{fg(ACCENT)}{'━' * k}●{fg(LINE)}{'─' * (n - k - 1)}{RESET}"
        return label_l + bar + label_r

    def footer(self, cols: int) -> list[str]:
        flist = self.visible_fields()
        desc = flist[self.sel].metadata["help"] if flist else ""
        line1 = fit(f" {ITALIC}{fg(MUTED)}{desc}{RESET}", cols)
        chips = [("↑↓", "select"), ("←→", "adjust"), ("enter", "edit"), ("p", "preset"), ("z", "zoom"),
                 ("o", "export"), ("S", "save"), ("?", "help"), ("q", "quit")]
        keys = " ".join(f"{bg((52, 58, 86))}{fg((220, 224, 240))} {k} {RESET}{fg(MUTED)} {t}{RESET}"
                        for k, t in chips)
        with self._lock:
            err = self.error
        if err:
            msg = f"{fg(BAD)}✗ {err}{RESET}"
        elif self.status:
            color = GOOD if self.status_kind == "ok" else (MUTED if self.status_kind == "info" else BAD)
            icon = "✓ " if self.status_kind == "ok" else ""
            msg = f"{fg(color)}{icon}{self.status}{RESET}"
        else:
            msg = ""
        room = cols - term.visible_len(keys) - 3
        if room > 12 and msg:
            line2 = " " + keys + "  " + fit(msg, room)
        elif msg:
            line2 = " " + fit(msg, cols - 2)
        else:
            line2 = " " + keys
        return [line1, fit(line2, cols)]

    def help_overlay(self, x: int, y: int, w: int, h: int) -> list[str]:
        bw = min(w - 4, 46)
        bh = min(h - 2, len(KEYS_HELP) + 4)
        bx = x + (w - bw) // 2
        by = y + (h - bh) // 2
        out = self.box(bx, by, bw, bh, f"{fg(ACCENT)}keys{RESET}")
        for row in range(by + 1, by + bh - 1):
            out.append(f"\x1b[{row};{bx + 1}H{' ' * (bw - 2)}")
        for i, (k, t) in enumerate(KEYS_HELP[: bh - 4]):
            line = f" {fg(ACCENT)}{k:<10}{RESET} {t}"
            out.append(f"\x1b[{by + 2 + i};{bx + 1}H{fit(line, bw - 2)}")
        return out

    def draw(self) -> None:
        cols, rows = term.size()
        self._frame_counter += 1
        panel = 46 if cols >= 120 else (40 if cols >= 96 else 34)
        top = 2 if self.video is None else 3
        body_h = rows - top - 2  # minus footer lines
        out = ["\x1b[?2026h\x1b[H", fit(self.header(cols), cols)]
        if self.video is not None:
            out.append(f"\x1b[2;1H{fit(self.timeline(cols), cols)}")

        # Settings panel.
        out += self.box(1, top, panel, body_h, "settings")
        inner_h = body_h - 2
        for i, line in enumerate(self.settings_lines(panel - 2, inner_h)):
            out.append(f"\x1b[{top + 1 + i};2H{line}")

        # Preview panel.
        px, pw = panel + 1, cols - panel
        iw, ih = pw - 2, body_h - 2
        zoom = ZOOM_NAMES[self.zoom]
        mode = "sharp" if self.graphics != "blocks" else "blocks"
        out += self.box(px, top, pw, body_h, "preview", f"{zoom} · {mode}")
        key = (id(self.out), self.zoom, tuple(self.pan), iw, ih, self.graphics)
        if self.graphics == "blocks":
            if key != self._vp_key:
                vp = self.viewport_image(iw, ih)
                lines = []
                if vp is not None:
                    pad = (iw - vp.shape[1]) // 2
                    for yy in range(0, vp.shape[0] - 1, 2):
                        t, b = vp[yy], vp[yy + 1]
                        lines.append(" " * pad + "".join(
                            f"\x1b[38;2;{a[0]};{a[1]};{a[2]}m\x1b[48;2;{c[0]};{c[1]};{c[2]}m▀"
                            for a, c in zip(t, b)) + RESET)
                vpad = max((ih - len(lines)) // 2, 0)
                self._vp_lines = [""] * vpad + lines
                self._vp_key = key
            for i in range(ih):
                line = self._vp_lines[i] if i < len(self._vp_lines) else ""
                out.append(f"\x1b[{top + 1 + i};{px + 1}H{fit(line, iw)}")
        elif key != self._img_key:
            # Graphics terminals: clear the box, then place a real image once.
            for i in range(ih):
                out.append(f"\x1b[{top + 1 + i};{px + 1}H{' ' * iw}")
            crop = self.sharp_crop(iw, ih)
            if crop is not None:
                img, c, r = crop
                cx = px + 1 + (iw - c) // 2
                cy = top + 1 + (ih - r) // 2
                out.append(f"\x1b[{cy};{cx}H")
                out.append(term.kitty_place(img, c, r) if self.graphics == "kitty" else term.iterm_place(img, c, r))
            self._img_key = key
        if self.help:
            out += self.help_overlay(px, top, pw, body_h)
            self._img_key = None  # redraw the image after the overlay closes

        for i, line in enumerate(self.footer(cols)):
            out.append(f"\x1b[{rows - 1 + i};1H{line}")
        out.append("\x1b[?2026l")
        sys.stdout.write("".join(out))
        sys.stdout.flush()

    def prompt(self, kb: term.Keyboard, label: str, initial: str) -> str | None:
        cols, rows = term.size()
        buf = initial
        while True:
            line = f" {bg(ACCENT)}{fg((20, 22, 34))}{BOLD} {label} {RESET} {buf}"
            sys.stdout.write(f"\x1b[{rows};1H{fit(line, cols - 1)}\x1b[{rows};{term.visible_len(line) + 1}H\x1b[?25h")
            sys.stdout.flush()
            k = kb.read(None)
            if k == "enter":
                sys.stdout.write("\x1b[?25l")
                return buf
            if k == "esc":
                sys.stdout.write("\x1b[?25l")
                return None
            if k == "backspace":
                buf = buf[:-1]
            elif k == "space":
                buf += " "
            elif len(k) == 1 and k.isprintable():
                buf += k

    # -- main loop ------------------------------------------------------------
    def handle(self, k: str, kb: term.Keyboard) -> bool:
        """Process one key; False to quit."""
        flist = self.visible_fields()
        if k in ("q", "esc"):
            if self.help and k == "esc":
                self.help = False
                return True
            return False
        if k == "?":
            self.help = not self.help
        elif k == "up":
            self.sel = (self.sel - 1) % len(flist)
        elif k == "down":
            self.sel = (self.sel + 1) % len(flist)
        elif k in ("pgup", "UP"):
            self.sel = max(self.sel - 10, 0)
        elif k in ("pgdn", "DOWN"):
            self.sel = min(self.sel + 10, len(flist) - 1)
        elif k == "home":
            self.sel = 0
        elif k == "end":
            self.sel = len(flist) - 1
        elif k in ("left", "right"):
            self.adjust(-1 if k == "left" else 1)
        elif k in ("LEFT", "RIGHT", "[", "]"):
            self.adjust(-1 if k in ("LEFT", "[") else 1, big=True)
        elif k == "space":
            self.adjust(1)
        elif k == "u":
            f = flist[self.sel]
            with self._lock:
                setattr(self.s, f.name, getattr(self.base_settings(), f.name))
            self.request_render()
        elif k == "enter":
            f = flist[self.sel]
            val = self.prompt(kb, f.name.replace("_", "-"), str(getattr(self.s, f.name)))
            if val is not None:
                try:
                    with self._lock:
                        trial = replace(self.s).update({f.name: val})
                        trial.validate()
                        self.s = trial
                    self.request_render()
                except (ValueError, KeyError) as e:
                    self.set_status(str(e), "error")
        elif k in ("p", "P"):
            self.load_preset(1 if k == "p" else -1)
        elif k == "z":
            self.zoom = (self.zoom + 1) % len(ZOOMS)
        elif k in ("w", "a", "s", "d"):
            dx = {"a": -0.08, "d": 0.08}.get(k, 0)
            dy = {"w": -0.08, "s": 0.08}.get(k, 0)
            self.pan = [float(np.clip(self.pan[0] + dx, 0, 1)), float(np.clip(self.pan[1] + dy, 0, 1))]
        elif k in (",", ".") and self.video:
            self.seek((-1 if k == "," else 1) / max(self.video.fps, 1))
        elif k in ("<", ">") and self.video:
            self.seek(-5 if k == "<" else 5)
        elif k == "g":
            if self.graphics == "blocks":
                self.graphics = detect_protocol() if detect_protocol() != "blocks" else "kitty"
            else:
                sys.stdout.write(term.kitty_clear())
                self.graphics = "blocks"
            self._img_key = self._vp_key = None
            self.set_status(f"preview: {self.graphics}")
        elif k == "o":
            with self._lock:
                out = self.out
            if out is not None:
                from .media import save_image

                save_image(out, self.output)
                self.set_status(f"exported {self.output}", "ok")
        elif k == "S":
            self.save_path.write_text(to_toml(self.s))
            self.set_status(f"saved preset {self.save_path}", "ok")
        elif k == "c":
            self.set_status(self.command_line())
        if k in ("p", "P", "?", "z", "g"):
            sys.stdout.write("\x1b[2J")
            self._img_key = None
        return True

    def run(self) -> int:
        worker = threading.Thread(target=self._worker, daemon=True)
        worker.start()
        sys.stdout.write("\x1b[?1049h\x1b[?25l\x1b[2J")
        try:
            with term.Keyboard() as kb:
                self.set_status(self.note)
                self.request_render()
                last_size = term.size()
                while True:
                    self.draw()
                    k = kb.read(0.08 if self._rendering else 0.5)
                    if term.size() != last_size:
                        last_size = term.size()
                        sys.stdout.write("\x1b[2J")
                        self._img_key = None
                    if k and not self.handle(k, kb):
                        break
        finally:
            self._stop = True
            self._wake.set()
            if self.graphics == "kitty":
                sys.stdout.write(term.kitty_clear())
            sys.stdout.write("\x1b[?25h\x1b[?1049l")
            sys.stdout.flush()
        print(self.command_line())
        return 0


def run_tuner(src: Path, img: np.ndarray, settings: Settings, base: str | None, output: Path, save: Path,
              note: str = "", video=None, preview_mode: str = "auto") -> int:
    reason = term.interactive_ok()
    if reason:
        print(f"error: tune {reason}", file=sys.stderr)
        return 2
    tuner = Tuner(img, settings, src, base, output, save, video=video, preview_mode=preview_mode)
    tuner.note = note
    return tuner.run()
