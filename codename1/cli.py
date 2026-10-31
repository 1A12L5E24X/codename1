"""Command line interface."""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np

from . import __version__
from .media import (
    IMAGE_EXTS,
    MediaError,
    Progress,
    VideoOptions,
    auto_jobs,
    is_video,
    load_image,
    open_writer,
    ordered_map,
    probe,
    read_frames,
    save_image,
)
from .pipeline import render, to_uint8
from .presets import PRESETS, load_preset, to_toml
from .settings import Settings

COMMANDS = ("render", "presets", "gallery", "options", "tune", "match", "live", "app")

GROUP_TITLES = {
    "effect": "effect",
    "size": "size",
    "tone": "tone (applied before the effect)",
    "colors": "colors",
    "halftone": "halftone screen (--effect halftone)",
    "dither": "dither (--effect dither)",
    "texture": "print texture",
    "clipping": "paper clipping",
    "mask": "mask (different looks for subject and background)",
}


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def add_setting_args(parser: argparse.ArgumentParser) -> None:
    """One flag per Settings field; defaults are None so presets aren't overridden."""
    groups: dict[str, argparse._ArgumentGroup] = {}
    short = {"effect": ["-e"], "mode": ["-m"]}
    for f in fields(Settings):
        group = f.metadata["group"]
        if group not in groups:
            groups[group] = parser.add_argument_group(GROUP_TITLES.get(group, group))
        g = groups[group]
        help_text = f.metadata["help"] + f" [default: {f.default}]"
        flags = short.get(f.name, []) + [_flag(f.name)]
        if isinstance(f.default, bool):
            g.add_argument(*flags, dest=f.name, action=argparse.BooleanOptionalAction, default=None, help=help_text)
        else:
            kind = type(f.default)
            g.add_argument(
                *flags, dest=f.name, type=kind, default=None, choices=f.metadata.get("choices"),
                metavar=f.metadata.get("metavar") or (None if f.metadata.get("choices") else kind.__name__.upper()),
                help=help_text,
            )


def build_settings(args: argparse.Namespace) -> Settings:
    s = Settings()
    for p in args.preset or []:
        s.update(load_preset(p))
    overrides = {f.name: getattr(args, f.name) for f in fields(Settings) if getattr(args, f.name, None) is not None}
    s.update(overrides)
    s.validate()
    return s


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codename1",
        description="Turn photos and video into halftone and dither prints.",
        epilog="Run 'codename1 render -h' for every option, 'codename1 presets' for the built-in looks.",
    )
    parser.add_argument("--version", action="version", version=f"codename1 {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    r = sub.add_parser(
        "render",
        help="render images or videos (default command)",
        description="Render images or videos. Any option overrides the preset.",
    )
    r.add_argument("inputs", nargs="+", metavar="INPUT", help="image/video files, directories or globs")
    io = r.add_argument_group("input/output")
    io.add_argument("-o", "--output", help="output file, or a directory for several inputs")
    io.add_argument("-p", "--preset", action="append", metavar="NAME|FILE.toml",
                    help="start from a preset (repeatable; later ones override earlier)")
    io.add_argument("--format", help="output extension when -o is a directory or omitted (e.g. png, jpg, mp4, gif)")
    io.add_argument("--quality", type=int, default=95, help="JPEG/WebP quality [default: 95]")
    io.add_argument("--dpi", type=float, default=96,
                    help="PDF output: image pixels per inch, sets the page size [default: 96]")
    io.add_argument("--compare", action="store_true", help="put the original next to the result")
    io.add_argument("--preview", action="store_true", help="also draw the result in the terminal")
    io.add_argument("--preview-only", action="store_true", help="draw in the terminal and don't write a file")
    io.add_argument("--preview-mode", default="auto", choices=("auto", "blocks", "kitty", "iterm"),
                    help="terminal graphics: half-blocks everywhere, or full-res kitty/iTerm2 images [default: auto]")
    io.add_argument("--save-preset", metavar="FILE.toml", help="write the effective settings to a preset file")
    io.add_argument("--overwrite", "-y", action="store_true", help="replace existing output files")
    io.add_argument("-q", "--quiet", action="store_true", help="only print errors")

    v = r.add_argument_group("video")
    v.add_argument("--fps", type=float, default=0, help="output frame rate (0 = same as source)")
    v.add_argument("--start", type=float, default=0, help="start time in seconds")
    v.add_argument("--duration", type=float, default=0, help="only process this many seconds")
    v.add_argument("--max-frames", type=int, default=0, help="stop after this many frames")
    v.add_argument("--crf", type=int, default=16, help="encoder quality, lower is better [default: 16]")
    v.add_argument("--codec", default="", help="video encoder (libx264, libx265, libvpx-vp9, prores_ks, ...)")
    v.add_argument("--encoder-preset", default="medium", help="x264/x265 speed preset [default: medium]")
    v.add_argument("--pix-fmt", default="", help="encoder pixel format, e.g. yuv444p keeps colored dots crisp")
    v.add_argument("--no-audio", dest="audio", action="store_false", help="drop the audio track")
    v.add_argument("-j", "--jobs", type=int, default=0, help="worker processes (0 = auto: all cores but one, fewer if memory is tight)")
    add_setting_args(r)

    p = sub.add_parser("presets", help="list presets, or show one as TOML")
    p.add_argument("name", nargs="?", help="preset to show")

    g = sub.add_parser("gallery", help="render one image with every preset into a contact sheet")
    g.add_argument("input", help="image (or video: first frame) to render")
    g.add_argument("-o", "--output", default="gallery.png", help="output image [default: gallery.png]")
    g.add_argument("--tile", type=int, default=420, help="tile width in pixels [default: 420]")
    g.add_argument("--presets", default="", help="comma-separated subset of presets")
    g.add_argument("--columns", type=int, default=4, help="tiles per row [default: 4]")

    sub.add_parser("options", help="list every effect option with its default")

    ap = sub.add_parser("app", help="open the browser interface (same engine, runs locally)",
                        description="A local browser interface: full-resolution preview, sliders for every "
                                    "setting, preset gallery, before/after, video, webcam, match and export.")
    ap.add_argument("input", nargs="?", help="image or video to start with")
    ap.add_argument("-p", "--preset", action="append", metavar="NAME|FILE.toml", help="starting preset")
    ap.add_argument("--port", type=int, default=0, help="port on 127.0.0.1 (default: any free port)")
    ap.add_argument("--no-open", action="store_true", help="just print the link, don't open a browser")
    ap.add_argument("--webcam", action="store_true", help="start with the webcam")
    add_setting_args(ap)

    lv = sub.add_parser("live", help="webcam (or any video/stream) through a preset, live in the terminal",
                        description="Live preview from a camera, video file or stream URL. Keys: p/P preset, "
                                    "[ ] dot size, space pause, s snapshot, r record, q quit.")
    lv.add_argument("source", nargs="?", default="",
                    help="camera number or device (default: the first camera), or a video file / stream URL")
    lv.add_argument("-p", "--preset", action="append", metavar="NAME|FILE.toml", help="starting preset")
    lv.add_argument("--capture-width", type=int, default=480, help="processing width in pixels [default: 480]")
    lv.add_argument("--fps", type=float, default=15, help="capture frame rate [default: 15]")
    lv.add_argument("-o", "--output", default=".", help="folder for snapshots and recordings [default: .]")
    lv.add_argument("--preview-mode", default="auto", choices=("auto", "blocks", "kitty", "iterm"))
    add_setting_args(lv)

    m = sub.add_parser(
        "match", help="measure a reference print and write a preset that recreates it",
        description="Measure a halftone or dither reference (dot spacing and angle, inks, paper, tint, "
                    "softness, grain, JPEG wear) and write a preset that recreates the look on any photo.",
    )
    m.add_argument("reference", help="an image of the look you want")
    m.add_argument("-o", "--output", help="preset file to write [default: REFERENCE-NAME.toml]")
    m.add_argument("--check", metavar="IMAGE.png",
                   help="also render the reference (blurred back to a photo) with the preset, side by side")
    m.add_argument("--print", dest="print_only", action="store_true", help="print the preset, don't write it")
    m.add_argument("--no-calibrate", action="store_true",
                   help="skip the tone calibration renders (faster, less accurate)")

    t = sub.add_parser("tune", help="interactive tuner: adjust settings and watch the result live",
                       description="Adjust settings with the keyboard and watch the print update. "
                                   "On exit the equivalent command line is printed.")
    t.add_argument("input", help="image or video to tune on (videos can be scrubbed with , . < >)")
    t.add_argument("-p", "--preset", action="append", metavar="NAME|FILE.toml", help="starting preset")
    t.add_argument("-o", "--output", help="where 'o' writes the image [default: INPUT.tuned.png]")
    t.add_argument("--save", default="codename1-tuned.toml", help="where 'S' saves the preset [default: codename1-tuned.toml]")
    t.add_argument("--preview-mode", default="auto", choices=("auto", "blocks", "kitty", "iterm"),
                   help="sharp images in kitty/Ghostty/iTerm2/WezTerm, half-blocks elsewhere [default: auto]")
    add_setting_args(t)
    return parser


# --- helpers ----------------------------------------------------------------------


def log(args, *msg) -> None:
    if not getattr(args, "quiet", False):
        print(*msg, file=sys.stderr)


def expand_inputs(items: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in items:
        p = Path(item).expanduser()
        if p.is_dir():
            for child in sorted(p.iterdir()):
                if child.is_file() and (child.suffix.lower() in IMAGE_EXTS or is_video(child)):
                    out.append(child)
        elif p.exists():
            out.append(p)
        else:
            matches = sorted(glob.glob(item))
            if not matches:
                raise FileNotFoundError(f"no such file: {item}")
            out.extend(Path(m) for m in matches)
    return out


def output_path(src: Path, args, settings: Settings, multiple: bool) -> Path:
    video = is_video(src)
    tag = (args.preset[-1] if args.preset else settings.effect)
    tag = Path(tag).stem if tag.endswith(".toml") else tag
    ext = args.format or ("mp4" if video and src.suffix.lower() != ".gif" else ("gif" if video else "png"))
    ext = "." + ext.lstrip(".")
    if args.output:
        o = Path(args.output).expanduser()
        if multiple or o.is_dir() or args.output.endswith(("/", os.sep)):
            return o / f"{src.stem}.{tag}{ext}"
        return o
    return src.with_name(f"{src.stem}.{tag}{ext}")


def side_by_side(original: np.ndarray, result: np.ndarray) -> np.ndarray:
    from .ops import resize

    h, w = result.shape[:2]
    orig = original.astype(np.float32) / 255.0 if original.dtype == np.uint8 else original
    if orig.ndim == 3 and orig.shape[2] == 4:
        orig = orig[..., :3]
    orig = np.clip(resize(orig[..., :3], w, h), 0, 1)
    if result.shape[2] == 4:
        a = result[..., 3:4]
        result = result[..., :3] * a + (1 - a)
    gap = np.ones((h, max(4, w // 100), 3), np.float32)
    return np.concatenate([orig, gap, result], axis=1)


# --- video worker state (set per process) -------------------------------------------

_W: dict = {}


def _init_worker(settings: Settings, compare: bool) -> None:
    _W["s"] = settings
    _W["compare"] = compare


def _render_frame(item):
    index, frame = item
    s, compare = _W["s"], _W["compare"]
    out = render(frame, s, frame=index)
    if compare:
        out = side_by_side(frame, out)
    if out.shape[2] == 4:
        a = out[..., 3:4]
        out = out[..., :3] * a + (1 - a)
    out = to_uint8(out)
    h, w = out.shape[:2]
    return out[: h - h % 2, : w - w % 2]  # even sizes for yuv420p encoders


def render_video(src: Path, dst: Path, s: Settings, args) -> None:
    info = probe(src)
    opts = VideoOptions(
        fps=args.fps, start=args.start, duration=args.duration, max_frames=args.max_frames, crf=args.crf,
        codec=args.codec, encoder_preset=args.encoder_preset, pix_fmt=args.pix_fmt, audio=args.audio,
        jobs=args.jobs,
    )
    if s.paper.strip().lower() in ("none", "transparent"):
        log(args, "note: video has no alpha here; transparent paper is flattened onto white")
    fps = opts.fps or info.fps
    total = info.frames
    if opts.duration:
        total = int(round(opts.duration * fps))
    elif opts.start and info.duration:
        total = int(round(max(info.duration - opts.start, 0) * fps))
    elif opts.fps and info.duration:
        total = int(round(info.duration * fps))
    if opts.max_frames:
        total = min(total, opts.max_frames) if total else opts.max_frames

    from .dither import KERNELS

    if s.effect == "dither" and s.algorithm in KERNELS and args.temporal is None:
        from dataclasses import replace as _replace

        s = _replace(s, temporal=1.0)
        log(args, f"note: steadying {s.algorithm} for video (--temporal 0 to turn off)")
    if s.effect == "dither" and s.mode == "color":
        # Pick the palette once so it doesn't flicker from frame to frame.
        from dataclasses import replace

        from .color import auto_palette, to_hex
        from .pipeline import prepare

        probe_frames = read_frames(src, info, opts.start, max_frames=1)
        sample = next(probe_frames, None)
        probe_frames.close()
        if sample is not None:
            pal = auto_palette(prepare(sample, s), s.colors)
            s = replace(s, mode="palette", palette=" ".join(to_hex(c) for c in pal))

    frames = read_frames(src, info, opts.start, opts.duration, opts.fps, opts.max_frames)

    def limited():
        for i, f in enumerate(frames):
            if opts.max_frames and i >= opts.max_frames:
                break
            yield i, f

    from .pipeline import target_size

    tw, th = target_size(info.width, info.height, s)
    jobs = auto_jobs(max(tw * th, info.width * info.height) * (2 if args.compare else 1), opts.jobs)
    results = ordered_map(_render_frame, limited(), jobs, _init_worker, (s, args.compare))
    prog = Progress(total, f"{src.name} → {dst.name}") if not (args.quiet or args.preview_only) else None
    if prog:
        prog.update(0)  # show the bar right away, before the workers finish their first frame
    first = next(results, None)
    if first is None:
        raise MediaError(f"could not decode any frames from {src}")
    h, w = first.shape[:2]
    if args.preview or args.preview_only:
        from .preview import show

        show(first, args.preview_mode)
        if args.preview_only:
            frames.close()
            return
    writer = open_writer(dst, w, h, fps, src, opts, info.has_audio)
    try:
        for frame in _chain(first, results):
            writer.stdin.write(frame.tobytes())
            if prog:
                prog.update()
    except BrokenPipeError:
        pass
    finally:
        if writer.stdin:
            try:
                writer.stdin.close()
            except BrokenPipeError:
                pass
        err = writer.stderr.read().decode(errors="replace")
        writer.wait()
        if prog:
            prog.close()
    if writer.returncode != 0:
        raise MediaError(f"ffmpeg failed writing {dst}:\n{err.strip()}")


def _chain(first, rest):
    yield first
    yield from rest


def render_vector(img: np.ndarray, dst: Path, s: Settings, args) -> None:
    from .vector import build, to_pdf, to_svg

    doc = build(img, s)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.suffix.lower() == ".svg":
        dst.write_text(to_svg(doc))
    else:
        dst.write_bytes(to_pdf(doc, dpi=args.dpi))
    for note in doc.notes:
        log(args, f"  note: {note}")


def render_image(src: Path, dst: Path | None, s: Settings, args) -> None:
    img = load_image(src)
    if dst is not None and dst.suffix.lower() in (".svg", ".pdf") and not args.preview_only:
        render_vector(img, dst, s, args)
        if not (args.preview or args.preview_only):
            return
    out = render(img, s)
    if args.compare:
        out = side_by_side(img, out)
    out8 = to_uint8(out)
    if args.preview or args.preview_only:
        from .preview import show

        show(out8, args.preview_mode)
    if dst is not None and not args.preview_only and dst.suffix.lower() not in (".svg", ".pdf"):
        save_image(out8, dst, args.quality)


def cmd_render(args) -> int:
    s = build_settings(args)
    if args.save_preset:
        Path(args.save_preset).expanduser().write_text(to_toml(s))
        log(args, f"saved preset → {args.save_preset}")
    inputs = expand_inputs(args.inputs)
    if not inputs:
        log(args, "no inputs found")
        return 1
    multiple = len(inputs) > 1
    failures = 0
    for src in inputs:
        dst = output_path(src, args, s, multiple)
        if not args.preview_only and dst.exists() and not args.overwrite and dst.resolve() != src.resolve():
            log(args, f"skip {src.name}: {dst} exists (use -y to overwrite)")
            continue
        if dst.resolve() == src.resolve():
            log(args, f"skip {src.name}: output would overwrite the input")
            failures += 1
            continue
        try:
            if is_video(src):
                render_video(src, dst, s, args)
            else:
                t0 = time.monotonic()
                render_image(src, dst, s, args)
                if not args.preview_only:
                    ok = "\x1b[38;2;158;206;106m✓\x1b[0m" if sys.stderr.isatty() else "✓"
                    log(args, f"{ok} {src.name} → {dst}  ({time.monotonic() - t0:.1f}s)")
        except (MediaError, OSError, ValueError) as e:
            failures += 1
            print(f"error: {src}: {e}", file=sys.stderr)
    return 1 if failures else 0


def cmd_presets(args) -> int:
    if args.name:
        s = Settings().update(load_preset(args.name))
        sys.stdout.write(f"# {PRESETS[args.name][0]}\n" if args.name in PRESETS else "")
        sys.stdout.write(to_toml(s).replace("# codename1 preset\n\n", ""))
        return 0
    tty = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    accent, muted, bold, end = ("\x1b[38;2;122;162;247m", "\x1b[38;2;110;118;150m", "\x1b[1m", "\x1b[0m") \
        if tty else ("", "", "", "")
    groups = [
        ("reference looks", lambda n: n.startswith(("newsprint", "sky-dither"))),
        ("black & white, mostly white", lambda n: n.startswith("bw-")),
        ("more looks", lambda n: True),
    ]
    width = max(len(n) for n in PRESETS)
    shown: set[str] = set()
    for title, member in groups:
        names = [n for n in PRESETS if n not in shown and member(n)]
        if not names:
            continue
        print(f"{accent}{bold}{title.upper()}{end}")
        for name in names:
            desc, d = PRESETS[name]
            print(f"  {bold}{name:<{width}}{end}  {muted}{d.get('effect', 'halftone'):<8}{end}  {desc}")
            shown.add(name)
        print()
    print(f"{muted}use:{end} codename1 photo.jpg -p NAME    {muted}settings:{end} codename1 presets NAME    "
          f"{muted}try live:{end} codename1 tune photo.jpg")
    return 0


def cmd_options(args) -> int:
    group = None
    for f in fields(Settings):
        g = f.metadata["group"]
        if g != group:
            group = g
            print(f"\n{GROUP_TITLES.get(g, g)}:")
        choices = f.metadata.get("choices")
        extra = f"  [{'|'.join(choices)}]" if choices else ""
        print(f"  {_flag(f.name):<18} {str(f.default):<10} {f.metadata['help']}{extra}")
    return 0


def cmd_gallery(args) -> int:
    from PIL import Image, ImageDraw

    img = first_frame_or_image(Path(args.input))
    names = [n.strip() for n in args.presets.split(",") if n.strip()] or list(PRESETS)
    tile_w = args.tile
    h, w = img.shape[:2]
    tile_h = round(h * tile_w / w)
    label_h = 22
    cols = max(1, min(args.columns, len(names)))
    rows = -(-len(names) // cols)
    sheet = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), "#ffffff")
    draw = ImageDraw.Draw(sheet)
    prog = Progress(len(names), "gallery")
    for i, name in enumerate(names):
        s = Settings().update(load_preset(name))
        # Render at twice the tile size so dots survive the downscale.
        s.width = tile_w * 2
        s.height = 0
        s.scale = 1.0
        out = to_uint8(render(img, s))
        if out.shape[2] == 4:
            out = out[..., :3]
        tile = Image.fromarray(out).resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        x, y = (i % cols) * tile_w, (i // cols) * (tile_h + label_h)
        sheet.paste(tile, (x, y + label_h))
        draw.text((x + 6, y + 5), name, fill="#111111")
        prog.update()
    prog.close()
    sheet.save(args.output)
    print(f"gallery → {args.output}", file=sys.stderr)
    return 0


def _swatch(hex_color: str) -> str:
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return ""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"\x1b[48;2;{r};{g};{b}m    \x1b[0m "


def cmd_match(args) -> int:
    from .match import descreen, match
    from .ops import LUMA, gaussian_blur

    src = Path(args.reference).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"no such file: {src}")
    img = first_frame_or_image(src)
    rgb = (img[..., :3].astype(np.float32) / 255.0) if img.dtype == np.uint8 else img[..., :3]
    result = match(rgb)
    if not args.no_calibrate:
        from .match import calibrate

        err = calibrate(rgb, result)
        result.notes.append(f"tone calibrated against the reference (remaining color error {err:.1%})")
    s = Settings().update(result.settings)
    s.validate()

    bold, dim, end = ("\x1b[1m", "\x1b[2m", "\x1b[0m") if sys.stdout.isatty() else ("", "", "")
    print(f"{bold}{src.name}{end}")
    for note in result.notes:
        print(f"  • {note}")
    for key in ("ink", "tint", "paper"):
        value = result.settings.get(key)
        if value:
            print(f"  {key:<6}{_swatch(value)}{value}")

    toml = to_toml(s)
    if args.print_only:
        print()
        sys.stdout.write(toml)
    else:
        out = Path(args.output) if args.output else Path(f"{src.stem}.toml")
        out.write_text(f"# matched from {src.name}\n" + toml.replace("# codename1 preset\n", ""))
        print(f"\n{dim}preset →{end} {out}\n{dim}use it:{end}   codename1 photo.jpg -p {out}")

    if args.check:
        flat = descreen(rgb, result)
        out_img = render(flat, s)[..., :3]
        sigma = rgb.shape[1] / max(s.cells or 100, 1)
        err = np.abs(gaussian_blur(out_img @ LUMA, sigma) - gaussian_blur(rgb @ LUMA, sigma)).mean()
        side = np.concatenate([rgb, np.ones((rgb.shape[0], 8, 3), np.float32), out_img], axis=1)
        save_image(to_uint8(side), Path(args.check))
        print(f"{dim}check →{end}  {args.check}  (reference | recreated, average tone difference {err:.1%})")
    return 0


def first_frame_or_image(src: Path) -> np.ndarray:
    if is_video(src):
        info = probe(src)
        frames = read_frames(src, info, max_frames=1)
        img = next(frames, None)
        frames.close()
        if img is None:
            raise MediaError(f"could not decode a frame from {src}")
        return img
    return load_image(src)


def cmd_app(args) -> int:
    from .app import run_app

    s = build_settings(args)
    src = Path(args.input).expanduser() if args.input else None
    if src is not None and not src.exists():
        raise FileNotFoundError(f"no such file: {src}")
    preset = args.preset[-1] if args.preset else None
    return run_app(src, s, preset, args.port, not args.no_open, "webcam" if args.webcam else "edit")


def cmd_live(args) -> int:
    from .live import run_live

    s = build_settings(args)
    base = args.preset[-1] if args.preset else None
    out = Path(args.output).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    return run_live(args.source, s, base, args.capture_width, args.fps, out, args.preview_mode)


def cmd_tune(args) -> int:
    from .tune import run_tuner

    s = build_settings(args)
    src = Path(args.input).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"no such file: {src}")
    video = probe(src) if is_video(src) else None
    img = first_frame_or_image(src)
    base = args.preset[-1] if args.preset else None
    note = ""
    sized = any(getattr(args, k) is not None for k in ("width", "height", "scale"))
    longest = max(img.shape[:2])
    if not sized and longest > 1600:
        # Keep keypresses snappy; the scale is shown (and saved) like any setting.
        s.scale = round(1600 / longest, 3)
        note = f"large image: working at scale {s.scale:g} (set scale to 1 for full size)"
    out = Path(args.output) if args.output else src.with_name(f"{src.stem}.tuned.png")
    return run_tuner(src, img, s, base, out, Path(args.save), note, video=video, preview_mode=args.preview_mode)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in COMMANDS and argv[0] not in ("-h", "--help", "--version"):
        argv.insert(0, "render")
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        if sys.stdout.isatty() and not os.environ.get("NO_COLOR"):
            from .home import show

            show()
        else:
            parser.print_help()
        return 0
    try:
        commands = {
            "render": cmd_render, "presets": cmd_presets, "gallery": cmd_gallery,
            "options": cmd_options, "tune": cmd_tune, "match": cmd_match, "live": cmd_live, "app": cmd_app,
        }
        return commands[args.command](args)
    except (KeyError, ValueError, FileNotFoundError, MediaError) as e:
        msg = e.args[0] if isinstance(e, KeyError) and e.args else e
        print(f"error: {msg}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        # Output piped into something like `head` that stopped reading.
        sys.stderr.close()
        return 0
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
