"""Minimal cross-platform terminal layer for the interactive tools: raw keys,
styling helpers and inline images (kitty / iTerm2 graphics protocols)."""

from __future__ import annotations

import base64
import io
import os
import re
import shutil
import sys
import time

import numpy as np

WINDOWS = os.name == "nt"
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b_[^\x1b]*\x1b\\")


# --- styling ------------------------------------------------------------------------


def fg(rgb) -> str:
    return f"\x1b[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


def bg(rgb) -> str:
    return f"\x1b[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
ITALIC = "\x1b[3m"


def visible_len(s: str) -> int:
    return len(_ANSI.sub("", s))


def fit(s: str, width: int) -> str:
    """Pad or cut a styled string to exactly `width` visible cells (cuts end in …)."""
    if width <= 0:
        return ""
    n = visible_len(s)
    return s + " " * (width - n) if n <= width else _cut(s, width)


def _cut(s: str, width: int) -> str:
    out, count, i = [], 0, 0
    while i < len(s):
        m = _ANSI.match(s, i)
        if m:
            out.append(m.group())
            i = m.end()
            continue
        if count < width - 1:
            out.append(s[i])
            count += 1
        elif count == width - 1:
            out.append("…")
            count += 1
        i += 1
    return "".join(out)


def size() -> tuple[int, int]:
    c, r = shutil.get_terminal_size((100, 32))
    return max(c, 40), max(r, 12)


# --- keyboard -----------------------------------------------------------------------

_UNIX_KEYS = {
    "\x1b[A": "up", "\x1b[B": "down", "\x1b[C": "right", "\x1b[D": "left",
    "\x1bOA": "up", "\x1bOB": "down", "\x1bOC": "right", "\x1bOD": "left",
    "\x1b[1;2A": "UP", "\x1b[1;2B": "DOWN", "\x1b[1;2C": "RIGHT", "\x1b[1;2D": "LEFT",
    "\x1b[5~": "pgup", "\x1b[6~": "pgdn", "\x1b[H": "home", "\x1b[F": "end",
    "\x1b[1~": "home", "\x1b[4~": "end", "\r": "enter", "\n": "enter", "\x7f": "backspace",
    "\x08": "backspace", "\x1b": "esc", " ": "space", "\t": "tab",
}
_WIN_KEYS = {"H": "up", "P": "down", "K": "left", "M": "right", "I": "pgup", "Q": "pgdn",
             "G": "home", "O": "end", "\x8d": "UP", "\x91": "DOWN", "t": "RIGHT", "s": "LEFT"}


_MOUSE = re.compile(r"\x1b\[<(\d+);(\d+);(\d+)([Mm])")


class Keyboard:
    """Raw, unbuffered keyboard input as key names ('up', 'enter', 'q', ...).

    With mouse=True, clicks come back as 'click:COL:ROW' and the wheel as
    'wheelup:COL:ROW' / 'wheeldown:COL:ROW' (1-based cells, SGR mouse mode)."""

    def __init__(self, mouse: bool = False):
        self.mouse = mouse and not WINDOWS

    def __enter__(self) -> "Keyboard":
        if WINDOWS:
            _enable_windows_vt()
        else:
            import termios
            import tty

            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            if self.mouse:
                sys.stdout.write("\x1b[?1000h\x1b[?1006h")
                sys.stdout.flush()
        return self

    def __exit__(self, *exc) -> None:
        if not WINDOWS:
            import termios

            if self.mouse:
                sys.stdout.write("\x1b[?1000l\x1b[?1006l")
                sys.stdout.flush()
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def read(self, timeout: float | None) -> str:
        if WINDOWS:
            import msvcrt

            end = None if timeout is None else time.monotonic() + timeout
            while not msvcrt.kbhit():
                if end is not None and time.monotonic() >= end:
                    return ""
                time.sleep(0.01)
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                return _WIN_KEYS.get(msvcrt.getwch(), "")
            return _UNIX_KEYS.get(ch, ch)
        import select

        r, _, _ = select.select([self.fd], [], [], timeout)
        if not r:
            return ""
        data = os.read(self.fd, 256).decode(errors="ignore")
        if data.startswith("\x1b[<"):
            return parse_mouse(data)
        return _UNIX_KEYS.get(data, data if len(data) == 1 else "")


def parse_mouse(data: str) -> str:
    """First meaningful event in an SGR mouse report: a left press or a wheel step."""
    for m in _MOUSE.finditer(data):
        b, x, y, kind = int(m.group(1)), m.group(2), m.group(3), m.group(4)
        if b == 0 and kind == "M":
            return f"click:{x}:{y}"
        if b in (64, 65):
            return f"{'wheelup' if b == 64 else 'wheeldown'}:{x}:{y}"
    return ""


def _enable_windows_vt() -> None:
    """Turn on ANSI escape processing in the Windows console (Windows 10+)."""
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:  # noqa: BLE001 - best effort; worst case codes print raw
        pass


def interactive_ok() -> str | None:
    """None if an interactive UI can run here, else the reason it can't."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return "needs an interactive terminal"
    if not WINDOWS:
        try:
            import termios  # noqa: F401
        except ImportError:
            return "needs termios (Unix) or msvcrt (Windows)"
    return None


# --- inline images --------------------------------------------------------------------


def png_b64(img: np.ndarray, max_side: int = 1600) -> str:
    from PIL import Image

    im = Image.fromarray(img)
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG", compress_level=1)
    return base64.standard_b64encode(buf.getvalue()).decode()


def kitty_place(img: np.ndarray, cols: int, rows: int, image_id: int = 7) -> str:
    """Escape codes that replace image `image_id` with img, scaled into a
    cols x rows cell box at the cursor, without moving the cursor."""
    data = png_b64(img)
    parts = [f"\x1b_Ga=d,d=I,i={image_id},q=2\x1b\\"]
    chunks = [data[i:i + 4096] for i in range(0, len(data), 4096)] or [""]
    for n, chunk in enumerate(chunks):
        more = 1 if n < len(chunks) - 1 else 0
        if n == 0:
            head = f"a=T,f=100,i={image_id},c={cols},r={rows},C=1,q=2,m={more}"
        else:
            head = f"m={more}"
        parts.append(f"\x1b_G{head};{chunk}\x1b\\")
    return "".join(parts)


def kitty_clear(image_id: int | None = None) -> str:
    return f"\x1b_Ga=d,d=I,i={image_id},q=2\x1b\\" if image_id else "\x1b_Ga=d,q=2\x1b\\"


def iterm_place(img: np.ndarray, cols: int, rows: int) -> str:
    data = png_b64(img)
    size = len(data) * 3 // 4
    return (f"\x1b]1337;File=inline=1;size={size};width={cols};height={rows};"
            f"preserveAspectRatio=1:{data}\x07")
