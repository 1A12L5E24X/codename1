"""Interactive tuner: tweak settings with the keyboard and watch the print
update in the terminal."""

from __future__ import annotations

import os
import select
import shlex
import shutil
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np

from .color import PALETTES
from .pipeline import render, to_uint8
from .presets import PRESETS, load_preset, to_toml
from .preview import fit_resize
from .settings import Settings

STEPS = {
    "cell": 0.5, "angle": 5.0, "pixel": 1, "cells": 10, "levels": 1, "colors": 1, "seed": 1,
    "width": 50, "height": 50, "scale": 0.05, "contrast": 0.05, "gamma": 0.05, "exposure": 0.1,
    "brightness": 0.02, "saturation": 0.05, "sharpen": 0.1, "sharpen_radius": 0.25, "blur": 0.25,
    "denoise": 0.25, "softness": 0.1, "dot_gain": 0.05, "min_dot": 0.01, "max_dot": 0.01,
    "gcr": 0.05, "misregister": 0.1, "dot_size": 0.05, "diffusion": 0.05, "spread": 0.02,
    "bias": 0.02, "roughness": 0.05, "rough_scale": 0.25, "ink_noise": 0.05, "ink_opacity": 0.05,
    "bleed": 0.05, "grain": 0.05, "grain_scale": 0.25, "speckle": 0.1, "vignette": 0.05,
    "post_blur": 0.05, "black": 0.01, "white": 0.01, "auto_levels": 0.5, "shadows": 0.05,
    "highlights": 0.05,
}
# Values offered when cycling free-text settings with the arrow keys.
SUGGEST = {
    "ink": ["#111111", "#3a3a3c", "sky", "riso-pink", "riso-blue", "riso-teal", "red", "navy",
            "phosphor", "amber", "#e8f0ff", "white"],
    "paper": ["#ffffff", "#f6f5f1", "cream", "newsprint", "#0b0b0c", "black", "blueprint", "none"],
    "palette": list(PALETTES),
    "angles": ["15 75 0 45", "105 75 90 15", "0 0 0 0", "15 45 75"],
    "cmyk_inks": ["#00a4e4 #e2007a #ffe600 #1d1d1f", "#00aeef #ec008c #fff200 #000000"],
}
EFFECT_GROUPS = {"halftone": "dither", "dither": "halftone"}  # group hidden for each effect
ZOOMS = [0, 4, 2, 1, 0.5]  # 0 = fit; otherwise image pixels per terminal pixel


def _read_key(fd: int, timeout: float | None = None) -> str:
    r, _, _ = select.select([fd], [], [], timeout)
    if not r:
        return ""
    data = os.read(fd, 64).decode(errors="ignore")
    keys = {
        "\x1b[A": "up", "\x1b[B": "down", "\x1b[C": "right", "\x1b[D": "left",
        "\x1bOA": "up", "\x1bOB": "down", "\x1bOC": "right", "\x1bOD": "left",
        "\x1b[1;2C": "RIGHT", "\x1b[1;2D": "LEFT", "\x1b[5~": "pgup", "\x1b[6~": "pgdn",
        "\x1b[H": "home", "\x1b[F": "end", "\r": "enter", "\n": "enter", "\x7f": "backspace",
        "\x1b": "esc", " ": "space", "\t": "tab",
    }
    return keys.get(data, data if len(data) == 1 else "")


class Tuner:
    def __init__(self, img: np.ndarray, settings: Settings, src: Path, base: str | None, output: Path, save: Path):
        self.img = img
        self.s = settings
        self.src = src
        self.base = base
        self.output = output
        self.save_path = save
        self.sel = 0
        self.zoom = 0
        self.pan = [0.5, 0.5]
        self.status = "←/→ adjust · ↑/↓ select · ? help"
        self.out: np.ndarray | None = None
        self.render_ms = 0.0
        self.error = ""
        self.help = False
        self.preset_names = list(PRESETS)

    # -- data -----------------------------------------------------------------
    def visible_fields(self):
        hidden = EFFECT_GROUPS.get(self.s.effect)
        return [f for f in fields(Settings) if f.metadata["group"] != hidden]

    def rerender(self) -> None:
        t = time.monotonic()
        try:
            self.s.validate()
            self.out = to_uint8(render(self.img, self.s))
            self.error = ""
        except (ValueError, KeyError) as e:
            self.error = str(e)
        self.render_ms = (time.monotonic() - t) * 1000

    def adjust(self, direction: int, big: bool = False) -> None:
        f = self.visible_fields()[self.sel]
        cur = getattr(self.s, f.name)
        choices = f.metadata.get("choices")
        if isinstance(cur, bool):
            setattr(self.s, f.name, not cur)
        elif choices:
            i = (list(choices).index(cur) + direction) % len(choices)
            setattr(self.s, f.name, choices[i])
        elif isinstance(cur, (int, float)):
            step = STEPS.get(f.name, 1 if isinstance(cur, int) else 0.05) * (10 if big else 1)
            val = cur + direction * step
            if isinstance(cur, int):
                val = int(round(val))
            else:
                val = round(val, 4)
            minimum = {"cell": 1, "pixel": 1, "levels": 2, "colors": 2}.get(f.name, None)
            if minimum is None and f.name not in ("angle", "brightness", "exposure", "shadows", "highlights", "bias", "seed"):
                minimum = 0
            if minimum is not None:
                val = max(val, minimum)
            setattr(self.s, f.name, type(cur)(val))
        else:
            options = SUGGEST.get(f.name, [])
            if options:
                i = options.index(cur) if cur in options else -1
                setattr(self.s, f.name, options[(i + direction) % len(options)])
        self.rerender()

    def load_preset(self, step: int) -> None:
        idx = self.preset_names.index(self.base) if self.base in self.preset_names else -1
        self.base = self.preset_names[(idx + step) % len(self.preset_names)]
        keep = {"width": self.s.width, "height": self.s.height, "scale": self.s.scale}
        self.s = Settings().update(load_preset(self.base)).update(keep)
        self.sel = min(self.sel, len(self.visible_fields()) - 1)
        self.status = f"preset: {self.base} — {PRESETS[self.base][0]}"
        self.rerender()

    def command_line(self) -> str:
        base = Settings().update(load_preset(self.base)) if self.base else Settings()
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
    def viewport(self, cols: int, rows: int) -> np.ndarray | None:
        if self.out is None:
            return None
        from PIL import Image

        img = self.out
        if img.shape[2] == 4:
            a = img[..., 3:4].astype(np.float32) / 255
            img = (img[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
        h, w = img.shape[:2]
        th = rows * 2
        z = ZOOMS[self.zoom]
        if z == 0:
            scale = min(cols / w, th / h)
            tw, tt = max(1, int(w * scale)), max(2, int(h * scale))
            return fit_resize(img, tw, tt)
        vw, vh = int(cols * z), int(th * z)
        x0 = int(np.clip(self.pan[0] * w - vw / 2, 0, max(w - vw, 0)))
        y0 = int(np.clip(self.pan[1] * h - vh / 2, 0, max(h - vh, 0)))
        crop = img[y0:y0 + vh, x0:x0 + vw]
        tw, tt = max(1, int(crop.shape[1] / z)), max(2, int(crop.shape[0] / z))
        method = Image.Resampling.NEAREST if z <= 1 else Image.Resampling.BOX
        return np.asarray(Image.fromarray(crop).resize((tw, tt), method))

    def draw(self) -> None:
        cols, rows = shutil.get_terminal_size((100, 32))
        panel = 40 if cols >= 90 else 32
        body = rows - 2
        out = ["\x1b[H"]
        flist = self.visible_fields()
        self.sel = max(0, min(self.sel, len(flist) - 1))
        lines: list[str] = []
        if self.help:
            lines = [
                "\x1b[1mkeys\x1b[0m", "↑ ↓        select setting", "← →        adjust (shift: ×10)",
                "[ ]        adjust ×10", "enter      type a value", "space      toggle / next",
                "u          reset setting", "p / P      next / prev preset", "z          zoom (fit,¼,½,1:1,2×)",
                "w a s d    pan when zoomed", "o          write output image", "S          save preset TOML",
                "c          show command line", "q / esc    quit", "?          close help",
            ]
        else:
            top = max(0, min(self.sel - body // 2, len(flist) - body))
            group = None
            for i, f in enumerate(flist[top:top + body], start=top):
                v = getattr(self.s, f.name)
                if isinstance(v, float):
                    v = f"{v:g}"
                name = f.name.replace("_", "-")
                if f.metadata["group"] != group:
                    group = f.metadata["group"]
                mark = "\x1b[7m" if i == self.sel else ""
                gtag = f"\x1b[2m{group[:4]}\x1b[22m" if (i == top or flist[i - 1].metadata["group"] != group) else "    "
                text = f"{name:<14} {str(v)[:panel - 21]}"
                lines.append(f"{gtag} {mark}{text:<{panel - 6}}\x1b[0m")
        lines = (lines + [""] * body)[:body]
        vp = self.viewport(cols - panel - 1, body)
        img_lines: list[str] = []
        if vp is not None:
            for y in range(0, vp.shape[0] - 1, 2):
                t, b = vp[y], vp[y + 1]
                img_lines.append("".join(
                    f"\x1b[38;2;{a[0]};{a[1]};{a[2]}m\x1b[48;2;{c[0]};{c[1]};{c[2]}m▀" for a, c in zip(t, b)
                ) + "\x1b[0m")
        for r in range(body):
            left = lines[r]
            right = img_lines[r] if r < len(img_lines) else ""
            out.append(f"\x1b[K{left}\x1b[{panel + 1}G{right}\x1b[K\n")
        desc = flist[self.sel].metadata["help"] if flist else ""
        zoom = ["fit", "¼", "½", "1:1", "2×"][self.zoom]
        info = f" {self.base or 'custom'} · zoom {zoom} · {self.render_ms:.0f} ms "
        room = max(cols - len(info) - 2, 0)
        text = self.error or self.status
        if len(text) > room:
            text = text[: max(room - 1, 0)] + "…"
        msg = f"\x1b[31m{text}\x1b[0m" if self.error else text
        out.append(f"\x1b[2m{desc[:cols - 1]}\x1b[0m\x1b[K\n")
        out.append(f"\x1b[7m{info}\x1b[0m {msg}\x1b[K")
        sys.stdout.write("".join(out))
        sys.stdout.flush()

    def prompt(self, fd: int, label: str, initial: str) -> str | None:
        cols, rows = shutil.get_terminal_size((100, 32))
        buf = initial
        while True:
            sys.stdout.write(f"\x1b[{rows};1H\x1b[K{label}: {buf}\x1b[?25h")
            sys.stdout.flush()
            k = _read_key(fd)
            if k == "enter":
                sys.stdout.write("\x1b[?25l")
                return buf
            if k == "esc":
                sys.stdout.write("\x1b[?25l")
                return None
            if k == "backspace":
                buf = buf[:-1]
            elif len(k) == 1 and k.isprintable():
                buf += k
            elif k == "space":
                buf += " "

    # -- main loop ------------------------------------------------------------
    def run(self) -> int:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        sys.stdout.write("\x1b[?1049h\x1b[?25l\x1b[2J")
        try:
            tty.setcbreak(fd)
            self.status = "rendering…"
            self.draw()
            self.rerender()
            self.status = "←/→ adjust · ↑/↓ select · p preset · z zoom · ? help"
            while True:
                self.draw()
                k = _read_key(fd, 0.5)
                if not k:
                    continue
                flist = self.visible_fields()
                if k in ("q", "esc"):
                    break
                elif k == "?":
                    self.help = not self.help
                elif k == "up":
                    self.sel = (self.sel - 1) % len(flist)
                elif k == "down":
                    self.sel = (self.sel + 1) % len(flist)
                elif k == "pgup":
                    self.sel = max(self.sel - 10, 0)
                elif k == "pgdn":
                    self.sel = min(self.sel + 10, len(flist) - 1)
                elif k in ("left", "right"):
                    self.adjust(-1 if k == "left" else 1)
                elif k in ("LEFT", "RIGHT", "[", "]"):
                    self.adjust(-1 if k in ("LEFT", "[") else 1, big=True)
                elif k == "space":
                    self.adjust(1)
                elif k == "u":
                    f = flist[self.sel]
                    base = Settings().update(load_preset(self.base)) if self.base else Settings()
                    setattr(self.s, f.name, getattr(base, f.name))
                    self.rerender()
                elif k == "enter":
                    f = flist[self.sel]
                    val = self.prompt(fd, f.name.replace("_", "-"), str(getattr(self.s, f.name)))
                    if val is not None:
                        try:
                            self.s.update({f.name: val})
                            self.rerender()
                        except (ValueError, KeyError) as e:
                            self.error = str(e)
                elif k in ("p", "P"):
                    self.load_preset(1 if k == "p" else -1)
                elif k == "z":
                    self.zoom = (self.zoom + 1) % len(ZOOMS)
                    sys.stdout.write("\x1b[2J")
                elif k in ("w", "a", "s", "d"):
                    dx = {"a": -0.08, "d": 0.08}.get(k, 0)
                    dy = {"w": -0.08, "s": 0.08}.get(k, 0)
                    self.pan = [float(np.clip(self.pan[0] + dx, 0, 1)), float(np.clip(self.pan[1] + dy, 0, 1))]
                elif k == "o" and self.out is not None:
                    from .media import save_image

                    save_image(self.out, self.output)
                    self.status = f"wrote {self.output}"
                elif k == "S":
                    self.save_path.write_text(to_toml(self.s))
                    self.status = f"saved preset {self.save_path}"
                elif k == "c":
                    self.status = self.command_line()
                if k in ("p", "P", "?", "z"):
                    sys.stdout.write("\x1b[2J")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            sys.stdout.write("\x1b[?25h\x1b[?1049l")
            sys.stdout.flush()
        print(self.command_line())
        return 0


def run_tuner(src: Path, img: np.ndarray, settings: Settings, base: str | None, output: Path, save: Path) -> int:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("error: tune needs an interactive terminal", file=sys.stderr)
        return 2
    try:
        import termios  # noqa: F401
    except ImportError:
        print("error: tune needs a Unix-like terminal (termios)", file=sys.stderr)
        return 2
    return Tuner(img, settings, src, base, output, save).run()
