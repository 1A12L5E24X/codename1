"""Reading and writing images and video (video through ffmpeg pipes)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".tga", ".ppm", ".pgm"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".gif", ".apng"}


class MediaError(RuntimeError):
    pass


# --- images -----------------------------------------------------------------------


def is_animated_image(path: Path) -> bool:
    if path.suffix.lower() not in (".gif", ".webp", ".apng", ".png"):
        return False
    from PIL import Image

    try:
        with Image.open(path) as im:
            return getattr(im, "n_frames", 1) > 1
    except OSError:
        return False


def is_video(path: Path) -> bool:
    ext = path.suffix.lower()
    if ext in (".gif", ".webp", ".png", ".apng"):
        return is_animated_image(path)
    return ext in VIDEO_EXTS


def load_image(path: Path) -> np.ndarray:
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGBA" if "A" in im.getbands() or im.mode == "P" else "RGB")
        return np.asarray(im)


def save_image(arr: np.ndarray, path: Path, quality: int = 95) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if arr.ndim == 3 and arr.shape[2] == 4 and ext in (".jpg", ".jpeg", ".bmp"):
        a = arr[..., 3:4].astype(np.float32) / 255
        arr = (arr[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
    im = Image.fromarray(arr)
    kwargs = {}
    if ext in (".jpg", ".jpeg"):
        kwargs = {"quality": quality, "subsampling": 0, "optimize": True}
    elif ext == ".webp":
        kwargs = {"quality": quality, "method": 6}
    elif ext == ".png":
        kwargs = {"optimize": False, "compress_level": 6}
    im.save(path, **kwargs)


# --- video ------------------------------------------------------------------------


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise MediaError(f"{tool} not found on PATH; install ffmpeg to process video")


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int  # best estimate, 0 if unknown
    duration: float
    has_audio: bool


def probe(path: Path) -> VideoInfo:
    require_ffmpeg()
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)]
    try:
        data = json.loads(subprocess.run(cmd, capture_output=True, check=True, text=True).stdout)
    except subprocess.CalledProcessError as e:
        raise MediaError(f"ffprobe failed on {path}: {e.stderr.strip()}") from None
    streams = data.get("streams", [])
    vs = next((s for s in streams if s.get("codec_type") == "video"), None)
    if vs is None:
        raise MediaError(f"no video stream in {path}")
    w, h = int(vs["width"]), int(vs["height"])
    rot = 0
    for sd in vs.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    rot = int(vs.get("tags", {}).get("rotate", rot))
    if abs(rot) % 180 == 90:
        w, h = h, w

    def ratio(r: str) -> float:
        try:
            a, b = r.split("/")
            return float(a) / float(b) if float(b) else 0.0
        except (ValueError, AttributeError):
            return 0.0

    fps = ratio(vs.get("avg_frame_rate", "")) or ratio(vs.get("r_frame_rate", "")) or 25.0
    duration = float(vs.get("duration") or data.get("format", {}).get("duration") or 0)
    frames = int(vs.get("nb_frames") or 0) or int(round(duration * fps))
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    return VideoInfo(w, h, fps, frames, duration, has_audio)


def read_frames(path: Path, info: VideoInfo, start: float = 0, duration: float = 0, fps: float = 0, max_frames: int = 0):
    """Yield RGB uint8 frames (HxWx3)."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(path)]
    if duration:
        cmd += ["-t", str(duration)]
    if fps:
        cmd += ["-vf", f"fps={fps}"]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    size = info.width * info.height * 3
    complete = False
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                complete = True
                break
            yield np.frombuffer(buf, np.uint8).reshape(info.height, info.width, 3)
    finally:
        if not complete:
            # Consumer stopped early: stop the decoder without reporting a broken pipe.
            proc.kill()
        proc.stdout.close()
        err = proc.stderr.read().decode(errors="replace")
        proc.stderr.close()
        proc.wait()
        if complete and proc.returncode != 0 and err:
            print(f"ffmpeg (decode): {err.strip()}", file=sys.stderr)


@dataclass
class VideoOptions:
    fps: float = 0
    start: float = 0
    duration: float = 0
    max_frames: int = 0
    crf: int = 18
    codec: str = ""
    encoder_preset: str = "medium"
    pix_fmt: str = ""
    audio: bool = True
    jobs: int = 0


def _encoder_args(out: Path, opts: VideoOptions, fps: float) -> list[str]:
    ext = out.suffix.lower()
    if ext == ".gif":
        return [
            "-vf",
            "split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];[b][p]paletteuse=dither=none",
            "-loop", "0",
        ]
    if ext in (".png", ".jpg", ".jpeg") or "%" in out.name:
        return []
    codec = opts.codec
    if not codec:
        codec = "libvpx-vp9" if ext == ".webm" else "libx264"
    args = ["-c:v", codec]
    if codec in ("libx264", "libx265"):
        args += ["-crf", str(opts.crf), "-preset", opts.encoder_preset, "-pix_fmt", opts.pix_fmt or "yuv420p"]
        if out.suffix.lower() in (".mp4", ".mov", ".m4v"):
            args += ["-movflags", "+faststart"]
        if codec == "libx265":
            args += ["-tag:v", "hvc1"]
    elif codec == "libvpx-vp9":
        args += ["-crf", str(opts.crf + 12), "-b:v", "0", "-pix_fmt", opts.pix_fmt or "yuv420p", "-row-mt", "1"]
    elif codec.startswith("prores"):
        args += ["-profile:v", "3", "-pix_fmt", opts.pix_fmt or "yuv422p10le"]
    elif opts.pix_fmt:
        args += ["-pix_fmt", opts.pix_fmt]
    return args


def open_writer(out: Path, width: int, height: int, fps: float, src: Path | None, opts: VideoOptions, has_audio: bool):
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-",
    ]
    ext = out.suffix.lower()
    want_audio = opts.audio and has_audio and src is not None and ext not in (".gif", ".png", ".jpg", ".jpeg") and "%" not in out.name
    if want_audio:
        if opts.start:
            cmd += ["-ss", str(opts.start)]
        cmd += ["-i", str(src)]
        if opts.duration:
            cmd += ["-t", str(opts.duration)]
        cmd += ["-map", "0:v", "-map", "1:a?", "-c:a", "libopus" if ext == ".webm" else "aac", "-b:a", "192k", "-shortest"]
    cmd += _encoder_args(out, opts, fps)
    cmd.append(str(out))
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


class Progress:
    def __init__(self, total: int, label: str, stream=sys.stderr):
        self.total, self.label, self.stream = total, label, stream
        self.start = time.monotonic()
        self.done = 0
        self.tty = stream.isatty()
        self._last = 0.0

    def update(self, n: int = 1) -> None:
        self.done += n
        now = time.monotonic()
        if not self.tty or (now - self._last < 0.1 and self.done != self.total):
            return
        self._last = now
        el = now - self.start
        rate = self.done / el if el > 0 else 0
        if self.total:
            frac = min(self.done / self.total, 1)
            width = 28
            bar = "█" * int(frac * width) + "░" * (width - int(frac * width))
            eta = (self.total - self.done) / rate if rate else 0
            msg = f"\r{self.label} {bar} {self.done}/{self.total}  {rate:5.1f} fps  eta {eta:5.0f}s "
        else:
            msg = f"\r{self.label} {self.done} frames  {rate:5.1f} fps "
        self.stream.write(msg)
        self.stream.flush()

    def close(self) -> None:
        el = time.monotonic() - self.start
        if self.tty:
            self.stream.write("\r\033[K")
        self.stream.write(f"{self.label} {self.done} frames in {el:.1f}s\n")
        self.stream.flush()


def ordered_map(fn, items, jobs: int, initializer=None, initargs=()):
    """Like map(), in order, across processes, keeping a bounded number in flight."""
    if jobs <= 1:
        if initializer:
            initializer(*initargs)
        for it in items:
            yield fn(it)
        return
    import multiprocessing as mp

    ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
    with ctx.Pool(jobs, initializer=initializer, initargs=initargs) as pool:
        pending: deque = deque()
        for it in items:
            pending.append(pool.apply_async(fn, (it,)))
            if len(pending) >= jobs * 2:
                yield pending.popleft().get()
        while pending:
            yield pending.popleft().get()
