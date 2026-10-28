"""Live mode: a webcam (or any video/stream) through a preset, in the terminal."""

from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import term
from .media import MediaError, probe, require_ffmpeg, save_image
from .pipeline import render, to_uint8
from .presets import PRESETS, load_preset
from .preview import detect_protocol, fit_resize
from .settings import Settings
from . import ui
from .term import BOLD, RESET, bg, fg, fit

GOOD = ui.GOOD
REC = ui.BAD


def camera_input(source: str) -> list[str]:
    """ffmpeg input arguments for a camera name/number or a device path."""
    if sys.platform.startswith("linux"):
        dev = source if source.startswith("/dev/") else f"/dev/video{source or 0}"
        return ["-f", "v4l2", "-i", dev]
    if sys.platform == "darwin":
        return ["-f", "avfoundation", "-framerate", "30", "-i", f"{source or 0}:none"]
    if sys.platform.startswith("win"):
        name = source if source and not source.isdigit() else _windows_camera(int(source or 0))
        return ["-f", "dshow", "-i", f"video={name}"]
    raise MediaError(f"no camera support on {sys.platform}; pass a video file or stream URL instead")


def _windows_camera(index: int) -> str:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                       capture_output=True, text=True)
    names = re.findall(r'"([^"]+)"\s*\(video\)', r.stderr)
    if not names:
        raise MediaError("no DirectShow camera found")
    return names[min(index, len(names) - 1)]


def is_camera(source: str) -> bool:
    return source == "" or source.isdigit() or source.startswith("/dev/video") or source.startswith("camera")


class Capture:
    """Reads frames from ffmpeg on a thread, keeping only the newest one."""

    def __init__(self, source: str, width: int, fps: float):
        require_ffmpeg()
        self.frame: np.ndarray | None = None
        self.count = 0
        self.error = ""
        self._stop = False
        if is_camera(source):
            args = camera_input(source.removeprefix("camera").lstrip(":"))
            w, h = width, 0  # height learned from ffmpeg's output line
        else:
            info = probe(Path(source)) if Path(source).exists() else None
            args = ["-re", "-stream_loop", "-1", "-i", source] if info else ["-re", "-i", source]
            if info is None:
                raise MediaError(f"no such file: {source}")
            w = width
            h = int(round(info.height * width / info.width / 2)) * 2
        vf = f"scale={w}:{h if h else -2},fps={fps}"
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-nostdin", *args, "-vf", vf, "-an",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.size = (w, h) if h else self._learn_size()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _learn_size(self) -> tuple[int, int]:
        """Camera output size, from ffmpeg's 'Output ... rgb24, WxH' line."""
        seen_output = False
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            line = self.proc.stderr.readline().decode(errors="replace")
            if not line:
                break
            if line.startswith("Output #0"):
                seen_output = True
            m = re.search(r"rgb24.*?(\d{2,5})x(\d{2,5})", line)
            if seen_output and m:
                threading.Thread(target=self.proc.stderr.read, daemon=True).start()  # keep draining
                return int(m.group(1)), int(m.group(2))
        self.close()
        raise MediaError("camera didn't start (is it connected, and is it free?)")

    def _read(self) -> None:
        w, h = self.size
        n = w * h * 3
        while not self._stop:
            buf = self.proc.stdout.read(n)
            if len(buf) < n:
                self.error = "stream ended"
                return
            self.frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            self.count += 1

    def close(self) -> None:
        self._stop = True
        try:
            self.proc.kill()
            self.proc.wait(timeout=2)
        except Exception:  # noqa: BLE001 - already gone
            pass


class Recorder:
    def __init__(self, path: Path, w: int, h: int, fps: float):
        self.path = path
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(path)],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.size = (w, h)
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        w, h = self.size
        if frame.shape[0] >= h and frame.shape[1] >= w:
            self.proc.stdin.write(np.ascontiguousarray(frame[:h, :w]).tobytes())
            self.frames += 1

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()


PRESET_GROUPS = [
    ("reference", lambda n: n.startswith(("newsprint", "sky-dither"))),
    ("black & white", lambda n: n.startswith("bw-")),
    ("more", lambda n: True),
]


class LiveUI:
    """The live screen: preset list | camera preview, header and key chips."""

    def __init__(self, cap: "Capture", source: str, s: Settings, base: str | None, fps: float, out_dir: Path,
                 graphics: str):
        self.cap, self.fps, self.out_dir, self.graphics = cap, fps, out_dir, graphics
        self.source_label = (f"camera {source or 0}" if is_camera(source) else Path(source).name)
        self.s = replace(s, width=0, height=0, scale=1.0)
        self.base = base
        self.rows: list[tuple[str, str]] = []  # ("hdr", title) | ("preset", name)
        seen: set[str] = set()
        for title, member in PRESET_GROUPS:
            names = [n for n in PRESETS if n not in seen and member(n)]
            if names:
                self.rows.append(("hdr", title))
                self.rows += [("preset", n) for n in names]
                seen.update(names)
        self.names = [n for kind, n in self.rows if kind == "preset"]
        self.paused = False
        self.rec: Recorder | None = None
        self.rec_start = 0.0
        self.shots = 0
        self.status, self.status_kind = "", "info"
        self.flash_until = 0.0
        self.out: np.ndarray | None = None
        self.times: list[float] = []
        self.seen, self.dirty = -1, False
        self._img_key = None
        self.enlarged = 1.0
        self.want_browser = False

    # -- actions --------------------------------------------------------------
    def set_preset(self, name: str) -> None:
        self.base = name
        self.s = replace(Settings().update(load_preset(name)), width=0, height=0, scale=1.0)
        self.status, self.status_kind = PRESETS[name][0], "info"
        self.dirty = True

    def step_preset(self, d: int) -> None:
        i = self.names.index(self.base) if self.base in self.names else -1
        self.set_preset(self.names[(i + d) % len(self.names)])

    def dot_size(self, d: int) -> None:
        s = self.s
        if s.cells:
            self.s = replace(s, cells=max(10, s.cells - 10 * d))
        elif s.effect == "halftone":
            self.s = replace(s, cell=max(2.0, s.cell + d))
        else:
            self.s = replace(s, pixel=max(1.0, s.pixel + d))
        self.dirty = True

    def full_res(self) -> np.ndarray | None:
        """The current frame rendered at full camera resolution (for files)."""
        if self.graphics in ("kitty", "iterm") or self.cap.frame is None:
            return self.out
        return to_uint8(render(self.cap.frame, self.s)[..., :3])

    def snapshot(self) -> None:
        img = self.full_res()
        if img is None:
            return
        self.shots += 1
        path = self.out_dir / f"dotfx-snap-{time.strftime('%H%M%S')}-{self.shots}.png"
        save_image(img, path)
        self.status, self.status_kind = f"saved {path}", "ok"
        self.flash_until = time.monotonic() + 0.35

    def toggle_record(self) -> None:
        if self.rec:
            self.rec.close()
            self.status = f"recorded {self.rec.path} ({self.rec.frames / self.fps:.1f}s)"
            self.status_kind = "ok"
            self.rec = None
        elif self.cap.frame is not None:
            h, w = self.cap.size[1], self.cap.size[0]
            path = self.out_dir / f"dotfx-live-{time.strftime('%H%M%S')}.mp4"
            self.rec = Recorder(path, w - w % 2, h - h % 2, self.fps)
            self.rec_start = time.monotonic()
            self.status, self.status_kind = f"recording to {path}", "info"

    def update(self) -> None:
        frame = self.cap.frame
        if frame is None or self.paused or (self.cap.count == self.seen and not self.dirty):
            return
        self.seen, self.dirty = self.cap.count, False
        t0 = time.monotonic()
        if self.graphics in ("kitty", "iterm"):
            self.out = to_uint8(render(frame, self.s)[..., :3])
            if self.rec:
                self.rec.write(self.out)
        else:
            # Half-blocks: render straight at the terminal's resolution so the dots
            # are visible as blocks (presets size dots relative to the width, so
            # the look keeps its proportions). Files still get full resolution.
            from PIL import Image

            iw, ih = self.preview_area()
            h, w = frame.shape[:2]
            scale = min(iw / w, ih * 2 / h)
            tw, th = max(8, int(w * scale)), max(8, int(h * scale) // 2 * 2)
            small = np.asarray(Image.fromarray(frame).resize((tw, th), Image.Resampling.BOX))
            disp, self.enlarged = self.preview_settings(tw, scale)
            self.out = to_uint8(render(small, disp)[..., :3])
            if self.rec:
                self.rec.write(to_uint8(render(frame, self.s)[..., :3]))
        self.times = (self.times + [time.monotonic() - t0])[-12:]

    # -- drawing --------------------------------------------------------------
    def header(self, cols: int) -> str:
        left = (f"{ui.logo('dotfx live')} {BOLD}{self.source_label}{RESET}  "
                f"{fg(ui.MUTED)}preset{RESET} {fg(ui.ACCENT)}{self.base or 'custom'}{RESET}")
        bits = []
        if self.rec:
            on = int(time.monotonic() * 2) % 2 == 0
            dot = f"{fg(REC)}●{RESET}" if on else " "
            el = time.monotonic() - self.rec_start
            bits.append(f"{dot} {fg(REC)}{BOLD}REC{RESET} {fg(REC)}{int(el // 60):02d}:{el % 60:04.1f}{RESET}")
        if self.paused:
            bits.append(f"{fg(ui.CHANGED)}❚❚ paused{RESET}")
        fps_now = len(self.times) / sum(self.times) if self.times else 0
        w, h = self.cap.size
        bits.append(f"{fg(ui.MUTED)}{w}×{h}{RESET}")
        bits.append(f"{fg(GOOD)}{fps_now:4.1f}{RESET} {fg(ui.MUTED)}fps{RESET}")
        return ui.spread(left, "   ".join(bits) + " ", cols)

    def preset_panel(self, width: int, height: int) -> list[str]:
        list_h = height - 3
        sel = next((i for i, r in enumerate(self.rows) if r == ("preset", self.base)), 0)
        top = max(0, min(sel - list_h // 2, len(self.rows) - list_h))
        lines = []
        self._row_names = [name if kind == "preset" else None for kind, name in self.rows[top:top + list_h]]
        for kind, name in self.rows[top:top + list_h]:
            if kind == "hdr":
                rule = "─" * max(width - len(name) - 3, 0)
                lines.append(fit(f" {fg(ui.ACCENT)}{BOLD}{name.upper()}{RESET} {fg(ui.LINE)}{rule}{RESET}", width))
            elif name == self.base:
                lines.append(f"{fg(ui.ACCENT)}▌{RESET}{bg(ui.ACCENT_BG)}{fg((235, 238, 255))}{BOLD}"
                             f"{fit(name, width - 1)}{RESET}")
            else:
                lines.append(fit(f"  {name}", width))
        lines += [" " * width] * (list_h - len(lines))
        s = self.s
        if s.cells:
            label, value, lo, hi = "dots across", s.cells, 20, 400
        elif s.effect == "halftone":
            label, value, lo, hi = "cell size", s.cell, 2, 40
        else:
            label, value, lo, hi = "pixel size", s.pixel, 1, 16
        lines.append(f"{fg(ui.LINE)}{'─' * width}{RESET}")
        lines.append(fit(f" {fg(ui.MUTED)}{label}{RESET} {value:g}", width))
        lines.append(fit(f" {ui.slider(value, lo, hi, max(width - 2, 4))}", width))
        return lines

    def preview_settings(self, width: int, scale: float) -> tuple[Settings, float]:
        """Settings for the terminal-sized preview, with dots kept big enough to
        see as blocks (halftone >= 4, dither cells >= 2 terminal pixels)."""
        s = self.s
        minimum = 4.0 if s.effect == "halftone" else 2.0
        if s.cells:
            size = width / s.cells
            if size >= minimum:
                return s, 1.0
            return replace(s, cells=max(1, int(width / minimum))), minimum / size
        size = (s.cell if s.effect == "halftone" else s.pixel) * scale
        if size >= minimum:
            return replace(s, cell=s.cell * scale, pixel=max(1.0, s.pixel * scale)), 1.0
        return replace(s, cell=max(s.cell * scale, minimum), pixel=max(s.pixel * scale, minimum)), minimum / size

    def layout(self) -> tuple[int, int, int]:
        cols, rows = term.size()
        return cols, rows, (30 if cols >= 90 else 0)

    def preview_area(self) -> tuple[int, int]:
        cols, rows, panel = self.layout()
        return cols - panel - 2, rows - 4

    def draw(self) -> None:
        cols, rows, panel = self.layout()
        out = ["\x1b[?2026h\x1b[H", self.header(cols)]
        body_y, body_h = 2, rows - 2
        if panel:
            out += ui.box(1, body_y, panel, body_h, "presets")
            for i, line in enumerate(self.preset_panel(panel - 2, body_h - 2)):
                out.append(f"\x1b[{body_y + 1 + i};2H{line}")
        px, pw = panel + 1, cols - panel
        iw, ih = pw - 2, body_h - 2
        flash = time.monotonic() < self.flash_until
        mode = "sharp" if self.graphics in ("kitty", "iterm") else "blocks"
        label = mode if mode == "sharp" or self.enlarged <= 1.05 else f"blocks · dots ×{self.enlarged:.0f}"
        out += ui.box(px, body_y, pw, body_h, self.source_label, label, color=GOOD if flash else ui.LINE)
        img = self.out
        if img is None:
            msg = self.cap.error or "waiting for the camera…"
            color = REC if self.cap.error else ui.MUTED
            out.append(f"\x1b[{body_y + ih // 2};{px + max((iw - len(msg)) // 2, 1)}H{fg(color)}{msg}{RESET}")
        elif mode == "sharp":
            key = (id(img), iw, ih)
            if key != self._img_key:
                h, w = img.shape[:2]
                c = min(iw, int(ih * 2 * w / h))
                r = max(1, min(ih, int(round(c * h / w / 2))))
                out.append(f"\x1b[{body_y + 1 + (ih - r) // 2};{px + 1 + (iw - c) // 2}H")
                out.append(term.kitty_place(img, c, r) if self.graphics == "kitty" else term.iterm_place(img, c, r))
                self._img_key = key
        else:
            h, w = img.shape[:2]
            scale = min(iw / w, ih * 2 / h, 1.0)
            vp = img if scale >= 1.0 else fit_resize(img, max(1, int(w * scale)), max(2, int(h * scale) // 2 * 2))
            lines = ui.half_blocks(vp, iw, ih)
            for i in range(ih):
                line = lines[i] if i < len(lines) else ""
                out.append(f"\x1b[{body_y + 1 + i};{px + 1}H{fit(line, iw)}")
        keys, self._chips = ui.chips_layout(
            [("b", "open in browser", "b", True), ("↑↓", "preset", "down"), ("[ ]", "dot size", "]"),
             ("space", "pause", "space"), ("s", "snapshot", "s"), ("r", "record", "r"), ("g", "preview", "g"),
             ("q", "quit", "q")], x=2)
        self._footer_row = rows
        color = GOOD if self.status_kind == "ok" else ui.MUTED
        icon = "✓ " if self.status_kind == "ok" else ""
        status = f"  {fg(color)}{icon}{self.status}{RESET}" if self.status else ""
        out.append(f"\x1b[{rows};1H{fit(' ' + keys + status, cols)}\x1b[?2026l")
        sys.stdout.write("".join(out))
        sys.stdout.flush()

    def click(self, k: str) -> bool:
        kind, x, y = k.split(":")
        x, y = int(x), int(y)
        if kind == "click" and y == getattr(self, "_footer_row", -1):
            for x0, x1, action in getattr(self, "_chips", []):
                if x0 <= x <= x1:
                    return self.handle(action)
            return True
        names = getattr(self, "_row_names", [])
        in_list = x <= 30 and 0 <= y - 3 < len(names)
        if kind.startswith("wheel"):
            return self.handle("up" if kind == "wheelup" else "down") if in_list else True
        if in_list and names[y - 3]:
            self.set_preset(names[y - 3])
            sys.stdout.write("\x1b[2J")
            self._img_key = None
        return True

    def handle(self, k: str) -> bool:
        if k.startswith(("click:", "wheelup:", "wheeldown:")):
            return self.click(k)
        if k == "b":
            # The browser takes over the camera, so the terminal screen closes.
            self.want_browser = True
            return False
        if k in ("q", "esc"):
            return False
        if k in ("up", "P"):
            self.step_preset(-1)
        elif k in ("down", "p"):
            self.step_preset(1)
        elif k in ("[", "]", "left", "right"):
            self.dot_size(-1 if k in ("[", "left") else 1)
        elif k == "space":
            self.paused = not self.paused
        elif k == "s":
            self.snapshot()
        elif k == "r":
            self.toggle_record()
        elif k == "g":
            if self.graphics in ("kitty", "iterm"):
                sys.stdout.write(term.kitty_clear())
                self.graphics = "blocks"
            else:
                self.graphics = detect_protocol() if detect_protocol() != "blocks" else "kitty"
            self._img_key = None
        if k in ("up", "down", "p", "P", "g"):
            sys.stdout.write("\x1b[2J")
            self._img_key = None
        return True


def run_live(source: str, s: Settings, base: str | None, width: int, fps: float, out_dir: Path,
             preview_mode: str = "auto") -> int:
    reason = term.interactive_ok()
    if reason:
        print(f"error: live {reason}", file=sys.stderr)
        return 2
    cap = Capture(source, width, fps)
    graphics = detect_protocol() if preview_mode == "auto" else preview_mode
    live = LiveUI(cap, source, s, base, fps, out_dir, graphics)
    if base is None and s == Settings():
        live.set_preset("newsprint")  # nothing chosen: start somewhere that looks like something
    sys.stdout.write("\x1b[?1049h\x1b[?25l\x1b[2J")
    try:
        with term.Keyboard(mouse=True) as kb:
            while True:
                live.update()
                live.draw()
                k = kb.read(0.01 if not live.paused else 0.25)
                if k and not live.handle(k):
                    break
    finally:
        if live.rec:
            live.rec.close()
        cap.close()
        if live.graphics == "kitty":
            sys.stdout.write(term.kitty_clear())
        sys.stdout.write("\x1b[?25h\x1b[?1049l")
        sys.stdout.flush()
    if live.shots or live.status.startswith("recorded"):
        print(f"saved to {out_dir}")
    if live.want_browser:
        from .app import run_app

        return run_app(None, live.s, live.base, 0, True, mode="webcam")
    return 0
