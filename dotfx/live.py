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
from .term import BOLD, RESET, bg, fg, fit

ACCENT = (122, 162, 247)
MUTED = (110, 118, 150)
GOOD = (158, 206, 106)
REC = (247, 118, 142)


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


def run_live(source: str, s: Settings, base: str | None, width: int, fps: float, out_dir: Path,
             preview_mode: str = "auto") -> int:
    reason = term.interactive_ok()
    if reason:
        print(f"error: live {reason}", file=sys.stderr)
        return 2
    cap = Capture(source, width, fps)
    graphics = detect_protocol() if preview_mode == "auto" else preview_mode
    names = list(PRESETS)
    paused, rec, shots = False, None, 0
    status, last_out, times, seen, dirty = "", None, [], -1, False
    s = replace(s, width=0, height=0, scale=1.0)
    sys.stdout.write("\x1b[?1049h\x1b[?25l\x1b[2J")
    try:
        with term.Keyboard() as kb:
            while True:
                cols, rows = term.size()
                frame = cap.frame
                if frame is not None and not paused and (cap.count != seen or dirty):
                    seen, dirty = cap.count, False
                    t0 = time.monotonic()
                    last_out = to_uint8(render(frame, s)[..., :3])
                    times = (times + [time.monotonic() - t0])[-10:]
                    if rec:
                        rec.write(last_out)
                fps_now = 1 / (sum(times) / len(times)) if times else 0
                head = (f"{bg(ACCENT)}{fg((20, 22, 34))}{BOLD} ◉ dotfx live {RESET} "
                        f"{fg(MUTED)}preset{RESET} {fg(ACCENT)}{base or 'custom'}{RESET}  "
                        f"{fg(MUTED)}{cap.size[0]}×{cap.size[1]}  {fps_now:4.1f} fps{RESET}")
                if rec:
                    head += f"  {fg(REC)}● REC {rec.frames / fps:4.1f}s{RESET}"
                if paused:
                    head += f"  {fg(MUTED)}❚❚ paused{RESET}"
                keys = "  ".join(f"{bg((52, 58, 86))} {k} {RESET}{fg(MUTED)} {t}{RESET}" for k, t in
                                 (("p P", "preset"), ("[ ]", "dot size"), ("space", "pause"),
                                  ("s", "snapshot"), ("r", "record"), ("q", "quit")))
                out = ["\x1b[?2026h\x1b[H", fit(head, cols), "\n"]
                ih = rows - 3
                if last_out is not None:
                    h, w = last_out.shape[:2]
                    if graphics in ("kitty", "iterm"):
                        c = min(cols, int(ih * 2 * w / h))
                        r = max(1, int(round(c * h / w / 2)))
                        out.append(f"\x1b[2;{(cols - c) // 2 + 1}H")
                        out.append(term.kitty_place(last_out, c, r) if graphics == "kitty"
                                   else term.iterm_place(last_out, c, r))
                    else:
                        scale = min(cols / w, ih * 2 / h)
                        vp = fit_resize(last_out, max(1, int(w * scale)), max(2, int(h * scale) // 2 * 2))
                        pad = (cols - vp.shape[1]) // 2
                        for y in range(0, vp.shape[0] - 1, 2):
                            a_, b_ = vp[y], vp[y + 1]
                            line = "".join(f"\x1b[38;2;{a[0]};{a[1]};{a[2]}m\x1b[48;2;{c[0]};{c[1]};{c[2]}m▀"
                                           for a, c in zip(a_, b_))
                            out.append(f"\x1b[{2 + y // 2};1H\x1b[K\x1b[{pad + 1}G{line}{RESET}")
                elif cap.error:
                    out.append(f"\x1b[3;3H{fg(REC)}{cap.error}{RESET}")
                else:
                    out.append(f"\x1b[3;3H{fg(MUTED)}waiting for the camera…{RESET}")
                msg = f"  {fg(GOOD)}{status}{RESET}" if status else ""
                out.append(f"\x1b[{rows};1H{fit(' ' + keys + msg, cols)}\x1b[?2026l")
                sys.stdout.write("".join(out))
                sys.stdout.flush()

                k = kb.read(0.01 if not paused else 0.3)
                if k in ("q", "esc"):
                    break
                if k in ("p", "P"):
                    i = names.index(base) if base in names else -1
                    base = names[(i + (1 if k == "p" else -1)) % len(names)]
                    s = replace(Settings().update(load_preset(base)), width=0, height=0, scale=1.0)
                    status = f"{base}: {PRESETS[base][0]}"
                    dirty = True
                    sys.stdout.write("\x1b[2J")
                elif k in ("[", "]"):
                    d = -1 if k == "[" else 1
                    if s.cells:
                        s = replace(s, cells=max(10, s.cells - 10 * d))
                    elif s.effect == "halftone":
                        s = replace(s, cell=max(2.0, s.cell + d))
                    else:
                        s = replace(s, pixel=max(1.0, s.pixel + d))
                    dirty = True
                elif k == "space":
                    paused = not paused
                elif k == "s" and last_out is not None:
                    shots += 1
                    path = out_dir / f"dotfx-snap-{time.strftime('%H%M%S')}-{shots}.png"
                    save_image(last_out, path)
                    status = f"saved {path}"
                elif k == "r" and last_out is not None:
                    if rec:
                        rec.close()
                        status = f"recorded {rec.path} ({rec.frames} frames)"
                        rec = None
                    else:
                        h, w = last_out.shape[:2]
                        path = out_dir / f"dotfx-live-{time.strftime('%H%M%S')}.mp4"
                        rec = Recorder(path, w - w % 2, h - h % 2, fps)
                        status = f"recording to {path}"
    finally:
        if rec:
            rec.close()
        cap.close()
        if graphics == "kitty":
            sys.stdout.write(term.kitty_clear())
        sys.stdout.write("\x1b[?25h\x1b[?1049l")
        sys.stdout.flush()
    return 0
