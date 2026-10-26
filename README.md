# dotfx

Turn photos and video into halftone and dither prints from the terminal.

![every built-in preset](docs/gallery.jpg)

dotfx has two engines:

- **halftone**: a rotated screen of dots whose size follows the image, as in offset
  printing. It has ten dot shapes, CMYK/RGB/duotone separations, and tone-accurate
  antialiased edges.
- **dither**: ordered (Bayer, clustered, blue noise, line) and error-diffusion
  (Floyd–Steinberg, Atkinson, Jarvis, Stucki, Burkes, Sierra…) dithering to
  ink/paper, gray levels, named palettes or colors picked from the image. Each
  dither pixel can be drawn as a square, circle, plus, ring and more.

Both engines share the same tone controls and print-texture layer: ragged dot
edges, ink bleed, uneven ink, paper grain, dust and scan softness. Video goes
through ffmpeg, with audio kept and frames rendered in parallel.

| `newsprint` | `sky-dither` |
|---|---|
| ![newsprint](docs/newsprint.jpg) | ![sky-dither](docs/sky-dither.jpg) |

## Install

Requires Python 3.11+, plus ffmpeg on your `PATH` for video.

```sh
pip install .            # numpy + Pillow
pip install '.[fast]'    # + numba: error-diffusion dithers ~80x faster
```

You can also run it without installing: `python -m dotfx ...`.

## Quick start

```sh
dotfx photo.jpg -p newsprint                 # → photo.newsprint.png
dotfx photo.jpg -p sky-dither -o out.png
dotfx clip.mp4 -p comic                      # → clip.comic.mp4 (audio kept)
dotfx clip.mov -p bitmap --width 480 -o loop.gif
dotfx shots/ -p riso -o printed/             # a whole folder
dotfx photo.jpg -p newsprint --preview-only  # just look, in the terminal
dotfx tune photo.jpg -p newsprint            # adjust interactively
dotfx gallery photo.jpg                      # every preset on one sheet
```

`render` is the default command, so `dotfx photo.jpg ...` is the same as
`dotfx render photo.jpg ...`.

## Presets

`dotfx presets` lists them, and `dotfx presets NAME` prints one as TOML.

| preset | engine | look |
|---|---|---|
| `newsprint` | halftone | gray 45° round dots on off-white stock, ragged edges, grain |
| `sky-dither` | dither | blue 4×4 Bayer dither drawn as a grid of dots (red channel darkens skies) |
| `comic` | halftone | bold CMYK pop-art dots on warm paper |
| `cmyk-print` | halftone | fine process print with misregistration |
| `riso` | halftone | single fluorescent-pink risograph, uneven ink |
| `riso-duo` | halftone | two-color riso: blue shadows over a pink plate |
| `color-dots` | halftone | dots in the photo's own colors on black |
| `rgb-screen` | halftone | additive red/green/blue screens on black |
| `line-screen` | halftone | diagonal line screen, banknote style |
| `engraving` | halftone | wavy engraved lines in sepia |
| `blueprint` | halftone | pale dots on blueprint blue |
| `crosshatch` | halftone | cross dots that grow into a hatched grid |
| `stipple` | dither | blue-noise stippling with round dots |
| `bitmap` | dither | classic 1-bit Atkinson |
| `xerox` | dither | blown-out photocopy with toner noise and dust |
| `gameboy` | dither | four greens, ordered dither |
| `cga` | dither | cyan/magenta/white |
| `pico8` | dither | PICO-8 palette, Floyd–Steinberg |
| `phosphor` | dither | green-screen terminal glow |
| `led-wall` | dither | chunky RGB LED matrix |
| `posterized` | dither | 6 colors picked from the image, blue-noise dither |

Command-line options override a preset, and presets stack (`-p comic -p
my-tweaks.toml`). To save your own:

```sh
dotfx photo.jpg -p riso --ink riso-teal --cell 5 --save-preset teal.toml
dotfx other.jpg -p teal.toml
```

A preset file is flat TOML. `base` names a preset to start from:

```toml
base = "newsprint"
ink = "#1a2a6c"
cell = 7
roughness = 0.5
```

## Interactive tuner

```sh
dotfx tune photo.jpg -p newsprint
```

The tuner shows a settings list beside a live preview. The `1:1` zoom shows the
actual dots. Images larger than 1600px start at a reduced `scale` so the preview
stays responsive. The scale appears in the list like any other setting.

| key | action |
|---|---|
| ↑ ↓ / PgUp PgDn | select a setting (only those that apply to the current effect are shown) |
| ← → | adjust; `[` `]` or Shift+←/→ for ×10 |
| space | toggle a switch / next choice |
| enter | type a value (colors, palettes, angles…) |
| u | reset the setting to the preset's value |
| p / P | next / previous preset |
| z | zoom: fit, ¼, ½, 1:1, 2× |
| w a s d | pan while zoomed |
| o | write the full-resolution image |
| S | save the settings as a preset TOML |
| c | show the equivalent command line |
| q / esc | quit and print the command line |

## Options

`dotfx options` lists every setting with its default, and `dotfx render -h`
shows the flags. Here they are by group.

**Effect**: `-e/--effect halftone|dither` and `-m/--mode`:

| mode | halftone | dither |
|---|---|---|
| `mono` | ink dots on paper | ink/paper (`--levels` for more shades) |
| `gray` | same as mono | `--levels` shades from ink to paper |
| `color` | dots carry the photo's colors | `--colors N` picked from the image |
| `cmyk` | four rotated process screens | — |
| `duotone` | `--ink` shadows plus an `--ink2` plate | — |
| `rgb` | additive R/G/B screens | per-channel `--levels` |
| `palette` | — | `--palette` (named, or `'#000 #f80 #fff'`) |

`--channel luma|red|green|blue|average|max|min` sets what drives tone in mono
and gray. `red` behaves like a red lens filter and turns blue skies dark.

**Size**: `--width`, `--height` and `--scale` resize before processing.
`--cell` (halftone) and `--pixel` (dither) are in output pixels. `--cells N`
sizes dots relative to the image instead, with N dots across the width, so a
12-megapixel photo and a thumbnail get the same look.

**Tone**: `--brightness --contrast --gamma --exposure --shadows --highlights
--black --white --auto-levels --saturation --sharpen --sharpen-radius --blur
--denoise --invert`.

**Colors**: `--ink --ink2 --paper` take names (`sky`, `riso-pink`, `blueprint`,
`phosphor`…), `#hex` or `r,g,b`. `--paper none` gives a transparent PNG.
`--palette` takes `gameboy cga pico8 c64 sepia riso vaporwave phosphor amber…`
or your own list. `--cmyk-inks` sets the four process inks.

**Halftone**:

- `--cell`, `--angle`, `--spot euclid|dot|square|diamond|line|ellipse|cross|ring|wave|grid`
- `--angles "15 75 0 45"`: per-plate angles for CMYK/RGB
- `--softness`: edge antialiasing in pixels, 0 for hard aliased edges
- `--dot-gain`, `--min-dot`, `--max-dot`
- `--sample average|center`
- `--gcr`: black generation for CMYK
- `--misregister`: plate offset in pixels

**Dither**:

- `--algorithm`: `threshold random bayer2 bayer4 bayer8 bayer16 cluster4 cluster8
  bluenoise lines floyd-steinberg false-floyd-steinberg atkinson jarvis stucki
  burkes sierra sierra2 sierra-lite`
- `--pixel`, `--cell-shape square|circle|diamond|plus|x|hline|vline|ring|rounded`
- `--dot-size`: below 1 leaves gaps between cells
- `--size-by-tone`: dot size follows tone instead of color
- `--diffusion`: error strength
- `--serpentine`, `--spread`, `--bias`

**Texture**: `--roughness --rough-scale --ink-noise --ink-opacity --bleed --grain
--grain-scale --speckle --vignette --post-blur`. For video, `--animate-grain`
re-rolls the texture every frame. Without it the paper stays still under the
moving image.

## Video

```sh
dotfx in.mp4 -p newsprint                         # h264, CRF 16, audio copied as AAC
dotfx in.mp4 -p comic --pix-fmt yuv444p           # no chroma subsampling: crisper colored dots
dotfx in.mp4 -p bitmap --start 12 --duration 4 -o clip.gif
dotfx in.mp4 -p riso -o out.webm                  # VP9 + Opus
dotfx in.mp4 -p riso --codec prores_ks -o out.mov # for editing
dotfx in.mp4 -p stipple -o frames/%05d.png        # image sequence
dotfx in.mp4 -p comic --compare                   # original | effect side by side
```

For video, prefer ordered dithers (`bayer*`, `cluster*`, `bluenoise`, `lines`)
and halftones. They stay put from frame to frame. Error diffusion (Floyd–Steinberg,
Atkinson…) recomputes its whole pattern every frame, so it shimmers. That can be
a look in its own right, but it costs bitrate.

Other flags: `--fps`, `--max-frames`, `--crf`, `--encoder-preset`, `--no-audio`,
and `-j/--jobs` for the number of worker processes (default: all cores). Screen
geometry and static textures are cached between frames, so a 1080p halftone
takes about 0.5–1 s per frame per core.

## Terminal preview

`--preview` (or `--preview-only`) draws the result in the terminal. In kitty,
Ghostty, iTerm2 and WezTerm the image is shown at full resolution through their
inline-image protocols. Other terminals get truecolor half-blocks. Use
`--preview-mode blocks|kitty|iterm` to choose.

## How it works

- **Halftone**: each pixel is placed in a cell of a rotated grid. The cell's
  tone is read from a cell-sized average of the image. Each dot shape is a spot
  function, and its values are remapped by rank so that the inked area at
  tone *t* is exactly *t*. This keeps tones accurate for every shape (tests
  hold this to within a few percent). Edges are antialiased from the spot
  gradient, so they stay crisp at any angle or cell size.
- **Dither**: the image is reduced to one value per dither pixel, quantized
  with a threshold map or error diffusion, and each pixel is drawn with an
  antialiased shape tile. Blue noise is generated once with void-and-cluster
  and cached in `~/.cache/dotfx`.
- **Texture**: roughness moves dot edges by a few pixels using value noise.
  Bleed spreads ink into nearby paper. Ink noise varies density across the
  page. Grain, speckle and post-blur are applied to the composite. Multi-ink
  modes overprint multiplicatively, like real inks.

## Development

```sh
pip install -e '.[test,fast]'
pytest
```
