"""dotfx app: the same engine behind a local browser interface.

A small standard-library HTTP server bound to 127.0.0.1. Every request needs a
random token (handed to the browser in the URL it opens) and a local Host
header, so other web pages can't drive it.
"""

from __future__ import annotations

import base64
import io
import json
import secrets
import shutil
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from dataclasses import fields, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import __version__
from .color import NAMED, PALETTES, parse_color
from .media import is_video, load_image, probe, read_frames
from .pipeline import prepare, render, to_uint8
from .presets import PRESETS, load_preset, to_toml
from .settings import Settings

WEB = Path(__file__).with_name("web")
MAX_UPLOAD = 2 * 1024 ** 3
PREVIEW_SIDE = 1600
COLOR_FIELDS = ("ink", "ink2", "paper", "tint", "backdrop")
NONE_OK = ("paper", "tint", "backdrop")
IMAGE_FORMATS = {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp",
                 "svg": "image/svg+xml", "pdf": "application/pdf"}
VIDEO_FORMATS = {"mp4": "video/mp4", "webm": "video/webm", "gif": "image/gif", "mov": "video/quicktime"}


def _png(img: np.ndarray, fmt: str = "PNG", quality: int = 92) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    im = Image.fromarray(img)
    if fmt == "JPEG" and im.mode == "RGBA":
        im = im.convert("RGB")
    im.save(buf, format=fmt, **({"compress_level": 1} if fmt == "PNG" else {"quality": quality}))
    return buf.getvalue()


def schema() -> dict:
    """Everything the browser needs to build the settings panel."""
    from .tune import RANGES, STEPS

    out = []
    for f in fields(Settings):
        kind = type(f.default).__name__
        entry = {"name": f.name, "type": kind, "default": f.default, "help": f.metadata["help"],
                 "group": f.metadata["group"], "choices": list(f.metadata.get("choices") or [])}
        if kind in ("int", "float"):
            lo, hi = RANGES.get(f.name, (0, 1))
            entry.update(min=lo, max=hi, step=STEPS.get(f.name, 1 if kind == "int" else 0.01))
        if f.name in COLOR_FIELDS:
            entry["color"] = True
            entry["none_ok"] = f.name in NONE_OK
        out.append(entry)
    named = {k: v for k, v in NAMED.items()}
    return {"fields": out, "named_colors": named, "palettes": PALETTES, "version": __version__}


class Media:
    def __init__(self, path: Path, name: str):
        self.id = uuid.uuid4().hex[:12]
        self.path = path
        self.name = name
        self.video = is_video(path)
        self.info = probe(path) if self.video else None
        self._frames: dict[float, np.ndarray] = {}
        self._lock = threading.Lock()
        img = self.frame(0.0)
        self.height, self.width = img.shape[:2]

    def frame(self, t: float = 0.0) -> np.ndarray:
        key = round(float(t), 2) if self.video else 0.0
        with self._lock:
            if key in self._frames:
                return self._frames[key]
        if self.video:
            frames = read_frames(self.path, self.info, start=key, max_frames=1)
            img = next(frames, None)
            frames.close()
            if img is None:
                raise ValueError(f"no frame at {key}s")
        else:
            img = load_image(self.path)
        with self._lock:
            if len(self._frames) > 24:
                self._frames.clear()
            self._frames[key] = img
        return img

    def describe(self) -> dict:
        d = {"id": self.id, "name": self.name, "kind": "video" if self.video else "image",
             "width": self.width, "height": self.height}
        if self.info:
            d.update(duration=self.info.duration, fps=self.info.fps, frames=self.info.frames)
        return d


class Job:
    def __init__(self, kind: str):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.done = 0
        self.total = 0
        self.state = "running"
        self.error = ""
        self.path: Path | None = None
        self.mime = "application/octet-stream"

    def describe(self) -> dict:
        return {"id": self.id, "state": self.state, "done": self.done, "total": self.total,
                "error": self.error, "name": self.path.name if self.path else ""}


class AppState:
    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.media: dict[str, Media] = {}
        self.jobs: dict[str, Job] = {}
        self.thumbs: dict[str, list] = {}
        self.initial: dict = {"media": None, "preset": None, "settings": {}, "mode": "edit"}
        self.token = secrets.token_urlsafe(18)

    def add_media(self, path: Path, name: str | None = None) -> Media:
        m = Media(path, name or path.name)
        self.media[m.id] = m
        return m


def settings_from(d: dict | None) -> Settings:
    s = Settings()
    if d:
        s.update({k: v for k, v in d.items() if v is not None})
    s.validate()
    return s


def preview_settings(s: Settings, media_w: int, media_h: int, max_side: int) -> tuple[Settings, float]:
    """Keep previews quick: big images render at a reduced scale (dots are sized
    relative to the image with `cells`, so most looks don't change)."""
    if s.width or s.height or s.scale != 1:
        return s, 1.0
    longest = max(media_w, media_h)
    if longest <= max_side:
        return s, 1.0
    k = max_side / longest
    return replace(s, scale=k), k


def changed_settings(s: Settings) -> dict:
    return s.to_dict(only_changed=True)


def cli_command(name: str, s: Settings) -> str:
    import shlex

    parts = ["dotfx", shlex.quote(name)]
    for f in fields(Settings):
        v = getattr(s, f.name)
        if v == f.default:
            continue
        flag = "--" + f.name.replace("_", "-")
        if isinstance(v, bool):
            parts.append(flag if v else "--no-" + f.name.replace("_", "-"))
        else:
            parts += [flag, shlex.quote(str(v))]
    return " ".join(parts)


# --- video export -------------------------------------------------------------------


def export_video(state: AppState, job: Job, media: Media, s: Settings, fmt: str) -> None:
    from .cli import _init_worker, _render_frame
    from .dither import KERNELS
    from .media import VideoOptions, open_writer, ordered_map

    try:
        if s.effect == "dither" and s.algorithm in KERNELS and s.temporal == 0:
            s = replace(s, temporal=1.0)
        info = media.info
        out = state.workdir / f"{Path(media.name).stem}.dotfx.{fmt}"
        job.path, job.mime = out, VIDEO_FORMATS[fmt]
        job.total = info.frames
        frames = read_frames(media.path, info)
        import os

        results = ordered_map(_render_frame, enumerate(frames), os.cpu_count() or 1, _init_worker, (s, False),
                              start_method="spawn")
        writer = None
        for frame in results:
            if writer is None:
                h, w = frame.shape[:2]
                writer = open_writer(out, w, h, info.fps, media.path, VideoOptions(crf=18), info.has_audio)
            writer.stdin.write(frame.tobytes())
            job.done += 1
        if writer is None:
            raise ValueError("no frames decoded")
        writer.stdin.close()
        err = writer.stderr.read().decode(errors="replace")
        if writer.wait() != 0:
            raise ValueError(err.strip() or "ffmpeg failed")
        job.state = "done"
    except Exception as e:  # noqa: BLE001 - reported to the browser
        job.state, job.error = "error", str(e)


# --- HTTP -----------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = f"dotfx/{__version__}"
    state: AppState

    def log_message(self, *args) -> None:  # quiet
        pass

    # helpers
    def _allowed(self) -> bool:
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost"):
            return False
        q = parse_qs(urlparse(self.path).query)
        token = self.headers.get("X-Dotfx-Token") or (q.get("t") or [""])[0]
        return secrets.compare_digest(token, self.state.token)

    def _send(self, body: bytes, mime: str, status=HTTPStatus.OK, headers: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, status=HTTPStatus.OK) -> None:
        self._send(json.dumps(data).encode(), "application/json", status)

    def _error(self, msg: str, status=HTTPStatus.BAD_REQUEST) -> None:
        self._json({"error": msg}, status)

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD:
            raise ValueError("upload too large")
        return self.rfile.read(n) if n else b""

    def _json_body(self) -> dict:
        raw = self._body()
        return json.loads(raw) if raw else {}

    def _media(self, mid: str | None) -> Media:
        m = self.state.media.get(mid or "")
        if m is None:
            raise KeyError("unknown media; open a file first")
        return m

    # routes
    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            if not self._allowed():
                return self._send(b"dotfx: open the link printed in the terminal (it carries an access key).",
                                  "text/plain; charset=utf-8", HTTPStatus.FORBIDDEN)
            return self._send((WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
        if not self._allowed():
            return self._error("forbidden", HTTPStatus.FORBIDDEN)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/api/schema":
                return self._json(schema())
            if url.path == "/api/presets":
                return self._json([{"name": n, "description": d, "effect": p.get("effect", "halftone"),
                                    "settings": Settings().update(p).to_dict(only_changed=True)}
                                   for n, (d, p) in PRESETS.items()])
            if url.path == "/api/state":
                init = dict(self.state.initial)
                mid = init.get("media")
                init["media"] = self.state.media[mid].describe() if mid in self.state.media else None
                return self._json(init)
            if url.path == "/api/original":
                m = self._media(q.get("media"))
                s, _ = preview_settings(settings_from(json.loads(q.get("s", "{}"))), m.width, m.height,
                                        int(q.get("max", PREVIEW_SIDE)))
                img = to_uint8(prepare(m.frame(float(q.get("time", 0))), s))
                return self._send(_png(img, "JPEG", 92), "image/jpeg")
            if url.path == "/api/sample":
                m = self.state.add_media(WEB / "sample.jpg", "astronaut.jpg")
                return self._json(m.describe())
            if url.path.startswith("/api/job/"):
                parts = url.path.split("/")
                job = self.state.jobs.get(parts[3])
                if job is None:
                    return self._error("unknown job", HTTPStatus.NOT_FOUND)
                if len(parts) > 4 and parts[4] == "file":
                    if job.state != "done":
                        return self._error("not ready")
                    return self._send(job.path.read_bytes(), job.mime,
                                      headers={"Content-Disposition": f'attachment; filename="{job.path.name}"'})
                return self._json(job.describe())
        except (KeyError, ValueError) as e:
            return self._error(str(e.args[0] if isinstance(e, KeyError) else e))
        self._error("not found", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if not self._allowed():
            return self._error("forbidden", HTTPStatus.FORBIDDEN)
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/api/upload":
                name = Path(q.get("name") or "upload.png").name
                path = self.state.workdir / f"{uuid.uuid4().hex[:8]}-{name}"
                path.write_bytes(self._body())
                try:
                    m = self.state.add_media(path, name)
                except Exception as e:  # noqa: BLE001
                    path.unlink(missing_ok=True)
                    raise ValueError(f"can't open {name}: {e}") from None
                return self._json(m.describe())
            if url.path == "/api/render":
                body = self._json_body()
                m = self._media(body.get("media"))
                s = settings_from(body.get("settings"))
                s, k = preview_settings(s, m.width, m.height, int(body.get("max", PREVIEW_SIDE)))
                t0 = time.monotonic()
                out = to_uint8(render(m.frame(float(body.get("time", 0))), s))
                ms = (time.monotonic() - t0) * 1000
                return self._send(_png(out), "image/png", headers={
                    "X-Render-Ms": f"{ms:.0f}", "X-Size": f"{out.shape[1]}x{out.shape[0]}", "X-Scale": f"{k:.4g}"})
            if url.path == "/api/thumbs":
                body = self._json_body()
                m = self._media(body.get("media"))
                key = f"{m.id}@{body.get('time', 0)}"
                if key not in self.state.thumbs:
                    from PIL import Image

                    src = m.frame(float(body.get("time", 0)))
                    small = np.asarray(Image.fromarray(src[..., :3]).resize(
                        (360, max(1, round(src.shape[0] * 360 / src.shape[1]))), Image.Resampling.LANCZOS))
                    thumbs = []
                    for name in PRESETS:
                        img = to_uint8(render(small, Settings().update(load_preset(name)))[..., :3])
                        thumbs.append({"preset": name,
                                       "src": "data:image/jpeg;base64," + base64.b64encode(_png(img, "JPEG", 85)).decode()})
                    self.state.thumbs[key] = thumbs
                return self._json(self.state.thumbs[key])
            if url.path == "/api/frame":
                s = settings_from(json.loads(self.headers.get("X-Settings") or "{}"))
                from PIL import Image

                frame = np.asarray(Image.open(io.BytesIO(self._body())).convert("RGB"))
                t0 = time.monotonic()
                out = to_uint8(render(frame, s)[..., :3])
                ms = (time.monotonic() - t0) * 1000
                fmt = "PNG" if q.get("png") else "JPEG"
                return self._send(_png(out, fmt, 90), "image/png" if fmt == "PNG" else "image/jpeg",
                                  headers={"X-Render-Ms": f"{ms:.0f}"})
            if url.path == "/api/export":
                body = self._json_body()
                m = self._media(body.get("media"))
                s = settings_from(body.get("settings"))
                fmt = str(body.get("format", "png")).lower()
                stem = Path(m.name).stem
                if fmt in VIDEO_FORMATS:
                    if not m.video:
                        raise ValueError("video formats need a video")
                    job = Job("video")
                    self.state.jobs[job.id] = job
                    threading.Thread(target=export_video, args=(self.state, job, m, s, fmt), daemon=True).start()
                    return self._json(job.describe())
                if fmt not in IMAGE_FORMATS:
                    raise ValueError(f"unknown format {fmt}")
                frame = m.frame(float(body.get("time", 0)))
                if fmt in ("svg", "pdf"):
                    from .vector import build, to_pdf, to_svg

                    doc = build(frame, s)
                    data = to_svg(doc).encode() if fmt == "svg" else to_pdf(doc)
                else:
                    out = to_uint8(render(frame, s))
                    data = _png(out, {"png": "PNG", "jpg": "JPEG", "webp": "WEBP"}[fmt], 95)
                return self._send(data, IMAGE_FORMATS[fmt],
                                  headers={"Content-Disposition": f'attachment; filename="{stem}.dotfx.{fmt}"'})
            if url.path == "/api/match":
                from .match import calibrate, match

                name = Path(q.get("name") or "reference.png").name
                path = self.state.workdir / f"ref-{uuid.uuid4().hex[:8]}-{name}"
                path.write_bytes(self._body())
                img = load_image(path) if not is_video(path) else Media(path, name).frame(0)
                rgb = img[..., :3].astype(np.float32) / 255.0
                result = match(rgb)
                err = calibrate(rgb, result)
                result.notes.append(f"tone calibrated against the reference (remaining color error {err:.1%})")
                s = settings_from(result.settings)
                return self._json({"settings": changed_settings(s), "notes": result.notes})
            if url.path == "/api/preset":
                s = settings_from(self._json_body().get("settings"))
                return self._send(to_toml(s).encode(), "application/toml",
                                  headers={"Content-Disposition": 'attachment; filename="dotfx-look.toml"'})
            if url.path == "/api/command":
                body = self._json_body()
                m = self.state.media.get(body.get("media") or "")
                return self._json({"command": cli_command(m.name if m else "photo.jpg",
                                                          settings_from(body.get("settings")))})
            if url.path == "/api/color":
                rgb = parse_color(self._json_body().get("value", ""))
                return self._json({"hex": None if rgb is None else "#" + "".join(f"{int(c * 255):02x}" for c in rgb)})
        except (KeyError, ValueError) as e:
            return self._error(str(e.args[0] if isinstance(e, KeyError) and e.args else e))
        except Exception as e:  # noqa: BLE001
            return self._error(f"{type(e).__name__}: {e}", HTTPStatus.INTERNAL_SERVER_ERROR)
        self._error("not found", HTTPStatus.NOT_FOUND)


class App:
    """A running server. Use start() from code (e.g. the tuner's browser button)
    or serve_forever() from the `dotfx app` command."""

    def __init__(self, port: int = 0, workdir: Path | None = None):
        self._tmp = None
        if workdir is None:
            self._tmp = tempfile.mkdtemp(prefix="dotfx-app-")
            workdir = Path(self._tmp)
        self.state = AppState(workdir)
        handler = type("BoundHandler", (Handler,), {"state": self.state})
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
        self.httpd.daemon_threads = True
        self.thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/?t={self.state.token}"

    def open_with(self, path: Path | None = None, settings: Settings | None = None, preset: str | None = None,
                  mode: str = "edit") -> str:
        """Load a file and settings as the browser's starting point; returns the URL."""
        if path is not None:
            m = self.state.add_media(Path(path))
            self.state.initial["media"] = m.id
        if settings is not None:
            self.state.initial["settings"] = changed_settings(settings)
        self.state.initial["preset"] = preset
        self.state.initial["mode"] = mode
        return self.url + ("#webcam" if mode == "webcam" else "")

    def start(self) -> "App":
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._tmp:
            shutil.rmtree(self._tmp, ignore_errors=True)


_shared: App | None = None


def shared_app() -> App:
    """One background server per process (the terminal's browser button reuses it)."""
    global _shared
    if _shared is None:
        _shared = App().start()
    return _shared


def open_browser(url: str) -> bool:
    """Open url in the default browser without letting the launcher print into
    a full-screen terminal UI."""
    import os
    import subprocess

    try:
        if sys.platform == "darwin":
            cmd = ["open", url]
        elif sys.platform.startswith("win"):
            os.startfile(url)  # type: ignore[attr-defined]
            return True
        elif shutil.which("xdg-open") and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            cmd = ["xdg-open", url]
        else:
            return webbrowser.open(url, new=2)
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        return True
    except Exception:  # noqa: BLE001
        return False


def hyperlink(url: str) -> str:
    """A clickable link in terminals that support OSC 8 (plain text elsewhere)."""
    return f"\x1b]8;;{url}\x1b\\{url}\x1b]8;;\x1b\\" if sys.stdout.isatty() else url


def run_app(path: Path | None, settings: Settings, preset: str | None, port: int, open_it: bool,
            mode: str = "edit") -> int:
    from . import ui
    from .term import BOLD, RESET, fg

    app = App(port=port)
    url = app.open_with(path, settings, preset, mode)
    link = hyperlink(url)
    print(f"\n  {ui.logo('dotfx app')}  {BOLD}running in your browser{RESET}\n")
    print(f"  {fg(ui.MUTED)}open{RESET}  {link}")
    print(f"  {fg(ui.MUTED)}stop{RESET}  ctrl-c\n")
    if open_it and not open_browser(url):
        print(f"  {fg(ui.CHANGED)}couldn't open a browser here; copy the link above{RESET}\n")
    try:
        app.httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        app.stop()
    return 0
