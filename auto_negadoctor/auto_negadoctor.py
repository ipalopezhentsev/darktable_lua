"""Auto-negadoctor: derive darktable negadoctor params for a roll of
DSLR-scanned color negative frames.

Called by auto_negadoctor.lua on linear-Rec2020 exports of the uninverted
negatives (negadoctor absent / disabled in the history). The export width is
set Lua-side and analysis is resolution-independent (size-dependent constants
are fractions of the frame dimension):

    python auto_negadoctor.py [--debug-ui] [--ai-tune] img1.tif ...

--ai-tune (spec 03): after the full analytical pipeline, add an ALTERNATE
per-scene variant nudged by a local vision LLM (gemma3 via Ollama; see
scene_tuner.py) — applied params become the AI variant, and the debug session
carries both for A/B switching. Off by default; the analytical path is
unchanged.

Input formats:
  - 16-bit TIFF in linear Rec2020 (what the Lua side exports; REQUIRED for
    accurate film-base color: orange film base is out of sRGB gamut, an
    sRGB export clips its R channel)
  - 8-bit sRGB JPEG/PNG fallback (tests, manual runs): linearized + matrixed
    to Rec2020; the film-base R channel is typically clipped there, so Dmin
    is recovered only approximately

Pipeline (single pass, no darktable round-trip):
  1. per frame: linearize export -> approx module-input linear RGB,
     trim dark film-holder borders, find the lightest uniform orange
     window = local film-base candidate
  2. roll-wide: exposure-normalize candidates via EXIF (shutter*iso/N^2),
     pick the physically lightest as the global film base, distribute it
     to every frame as Dmin rescaled by that frame's exposure factor
  3. per frame: D_max/offset from frame percentiles, render the inverted
     preview with nega_model's exact forward model, find a dark-gray patch
     (-> wb_low) and a light-neutral patch (-> wb_high) on the preview
     using the patches' NEGATIVE-space colors, then black + exposure
  4. frames without usable neutral patches fall back to roll-median wb
  5. emit negadoctor_results.txt (params as ready 152-char hex blob) and,
     for --debug-ui, per-frame {stem}_debug_nega.json sessions (the UI renders
     the inverted preview + mask live from these — no baked images), then
     launch debug_ui.py

Results file format:
  OK|stem|params=<hex>
  DETAIL|stem|Dmin=..|D_max=..|...   (human/log only, Lua ignores)
  ERR|stem|message
"""

import json
import os
import statistics
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

# darktable TIFFs carry vendor EXIF tags libtiff doesn't know; without this
# every imread floods the log with cv::TIFF_Warning lines
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nega_model as nm

# --- batch parallelism -------------------------------------------------------
# The per-frame analytical stages (film-base search, crop, percentiles, wb,
# print tune) are independent and CPU/IO-bound, so they run frame-parallel.
# Worker threads are NAMED with nega_model's pool prefix, which makes the
# per-pixel render parallelism (render_negadoctor / linear_to_srgb) run INLINE
# inside them — coarse frame-level parallelism instead of nested oversubscription.
# Override the analytical width with NEGA_PROC_WORKERS. The AI/vision-LLM stage
# is GPU-bound (Ollama) and shares one model + one JSON cache, so it stays at a
# low default (NEGA_AI_WORKERS, default 1 = serial) to avoid thrashing the GPU.
_PROC_WORKERS_ENV = int(os.environ.get("NEGA_PROC_WORKERS", "0"))
_AI_WORKERS = max(1, int(os.environ.get("NEGA_AI_WORKERS", "1")))
# Cache each frame's decoded (enc_f, lin) on its dict so the analytical stages
# (A/B1/B2 + AI) share ONE TIFF decode instead of re-reading it per stage.
# process_roll strips it before the dicts are serialized. Disable with
# NEGA_FRAME_CACHE=0 to trade the speedup for a much smaller peak RSS (the cache
# holds every frame's float buffers resident from stage A until the run ends).
_FRAME_CACHE = os.environ.get("NEGA_FRAME_CACHE", "1") != "0"


def _proc_workers(n):
    if _PROC_WORKERS_ENV > 0:
        return max(1, min(_PROC_WORKERS_ENV, n))
    # The per-frame stages are MEMORY-BANDWIDTH bound (each streams several
    # full-frame float64 buffers), so throughput plateaus around 8 threads and
    # MORE workers just thrash the memory bus — measured knee on a 20-core box:
    # 8w≈20s, 16w≈24s (worse), 20w≈19s. Cap at the knee; override with
    # NEGA_PROC_WORKERS if your machine has more memory bandwidth.
    return max(1, min((os.cpu_count() or 1), 8, n))


def _get_loaded(fr):
    """(enc_f, lin) for a frame — loaded once and reused across stages (all of
    which call load_frame with the same vignette), unless NEGA_FRAME_CACHE=0."""
    cached = fr.get("_loaded")
    if cached is not None:
        return cached
    loaded = load_frame(fr["path"], fr.get("vignette"))
    if _FRAME_CACHE:
        fr["_loaded"] = loaded
    return loaded


class _Progress:
    """Thread-safe running progress: each tick advances the shared counter and
    reports the new total (out-of-order ticks are fine for the Lua bar)."""

    def __init__(self, cb, total):
        self._cb, self._total = cb, total
        self._done, self._lock = 0, threading.Lock()

    def tick(self, n=1):
        if not self._cb:
            return
        with self._lock:
            self._done += n
            d = self._done
        self._cb(d, self._total)


def _map_frames(fn, items, workers, on_done=None):
    """Apply fn to each item, order-preserving. Runs in a named thread pool
    when workers > 1 (so nested pixel-render runs inline); else inline. fn must
    be self-contained — it may mutate its OWN item but not shared state."""
    results = [None] * len(items)
    if workers <= 1 or len(items) <= 1:
        for i, it in enumerate(items):
            results[i] = fn(it)
            if on_done:
                on_done()
        return results
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix=nm._POOL_PREFIX) as ex:
        futures = [ex.submit(fn, it) for it in items]
        for i, fut in enumerate(futures):
            results[i] = fut.result()
            if on_done:
                on_done()
    return results

# --- resolution independence -------------------------------------------------
# Every size-dependent constant below is a RATIO of the frame dimension (a
# *_FRAC), turned into pixels at use sites with `int(round(w * FRAC))` (and a
# `max(1, …)` floor where a count of zero would be degenerate). There is no
# reference resolution: detection geometry scales directly with the export,
# whatever its width. Never introduce a raw pixel count tied to the export
# size — express it as a fraction of the dimension.

# --- film holder border trim ------------------------------------------------
BORDER_DARK_THR = 0.02           # linear luminance below this = holder plastic
BORDER_MAX_FRAC = 0.25           # never trim more than this per edge
BORDER_PAD_FRAC = 0.004          # safety pad after the detected border (frac of width)

# --- rectangular content crop -------------------------------------------------
# The holder is a RECTANGULAR frame around the film frame, so the analysis
# area is the largest inscribed rectangle containing only film; everything
# outside is disregarded in ALL level computations, and everything INSIDE
# counts — no per-pixel masks (user decision after auditing the pixel-mask
# approach). A scan line is junk-contaminated when it carries holder-dark
# pixels OR bright leak: nothing ON film can be lighter than the unexposed
# base, so "brighter than base by a margin" cleanly catches light leaking
# past the holder edge (the user's hand-drawn crop on DSC_0001 trimmed a
# 15px top band with NO dark pixels - it was bright leak).
HOLDER_LUMA_THR = 0.04           # linear luma below this is holder-dark
CROP_JUNK_LINE_FRAC = 0.05       # dark/leak fraction above which a line is trimmed
                                 # (0.04 -> 0.05: roll 2510-11-1's bright sky has
                                 # ~4% specular "leak" pixels per line; at 0.04 the
                                 # gap-tolerant hard run rode them 276px into the
                                 # scene. 0.05 clears the marginal leak; the dark
                                 # holder edge is far above it. No old-roll change.)
CROP_LEAK_MARGIN_D = 0.06        # brighter than base by this log10 density
# Film REBATE (unexposed strip at the frame edge) is base-colored: neither
# dark nor brighter-than-base. Signature: ALL channels within a small
# density margin of base. A line mostly made of such pixels near the edge
# is rebate/gap, not scene (user crop annotations on DSC_0005-0007: ~20-40px
# rebate bands at the right that the dark/leak tests were blind to).
CROP_REBATE_MARGIN_D = 0.13      # max-channel density below this = base-like
                                 # (0.12 -> 0.13: DSC_0037's right rebate sits
                                 # at ~0.14 max-density median; 0.13 catches it
                                 # via the per-column pixel distribution, fixing
                                 # a 50px right-edge over-include, with no
                                 # over-trim regression on the 16 crop fixtures)
CROP_REBATE_LINE_FRAC = 0.15     # base-like fraction above which a line is rebate
# A true rebate band TERMINATES (film content follows within tens of px);
# unexposed dark scene at the frame edge is base-like too but continues
# indefinitely (DSC_0021's museum ceiling drove a 151px top over-trim).
# Rebate-extended trims are only valid if the base-like band ends soon after.
CROP_REBATE_TERM_FRAC = 0.04     # lines past the trim that must NOT stay base-like (frac of width)
# A real film rebate is a NARROW strip (the reference roll's rebates span
# 25-58px at 2000px wide). Bright SCENE content (sky/snow/highlights) is
# base-like too, but in DIFFUSE bands hundreds of px deep, and its base-like
# fraction (0.15-0.34) overlaps the legit rebate's (0.3-0.55) — so the line
# threshold can't separate them, but the band WIDTH can. A base-like run wider
# than this is scene, not rebate: reject the extension (roll 2510-11-1 drove
# 99-288px rebate over-trims into bright content; legit rebates stay <80px).
CROP_REBATE_MAX_FRAC = 0.04      # max rebate-extension depth past the hard run (frac of width)
CROP_PAD_FRAC = 0.012            # safety pad past the last junk line, frac of width (user
                                 # crops ran ~3-5px inside the detected edges)
# Holder-edge SHADOW/penumbra: a darkened ramp (line mean luma well below the
# frame interior) too bright for HOLDER_LUMA_THR. Found via the user's
# full-roll crop annotations: every left-edge containment violation showed
# this ramp, with the luma plateau starting exactly at the user's margin.
CROP_SHADOW_REL = 0.70           # line mean luma below this fraction of the
                                 # interior reference = holder shadow
CROP_SHADOW_MAX_FRAC = 0.030     # a shadow run terminating within this depth
                                 # (frac of width) is real penumbra (full
                                 # credit); deeper = dense scene (bright sky is
                                 # DARK in negative space, drove +30px over-trims).
                                 # (0.04 -> 0.030: roll 2510-11-1's gently-darker
                                 # top scene formed 77-80px "shadow" runs that got
                                 # full credit -> +47-51px top over-trims; 0.030
                                 # routes them to the penumbra CORE. 0.025 breaks
                                 # old-roll containment, so 0.030 is the floor.)
CROP_SHADOW_CORE_FRAC = 0.016    # when the depressed band continues past the
                                 # max depth (dense scene at the edge), the
                                 # first px are still penumbra-contaminated —
                                 # keep only this core (frac of width)
CROP_GAP_TOL_FRAC = 0.02         # junk must form an EDGE-ANCHORED run; up to
                                 # this many clean lines (frac of width) may
                                 # interrupt it.
                                 # Without this, detached interior junk lines
                                 # (deep scene shadows are base-like too)
                                 # caused massive over-trims

# --- film base window search (on the negative) -------------------------------
BASE_WIN_FRAC = 0.04             # window side as fraction of image width
MIN_WIN_FRAC = 0.016             # floor for the search-window side (frac of width)
BASE_STRIDE_DIV = 2              # stride = window / this
CLIP_SRGB_THR = 0.97             # stored-encoding value above this counts as clipped
CLIP_FRAC_MAX = 0.02             # max fraction of white-clipped px in a base window
BASE_MIN_LUMA = 0.04             # linear; reject windows darker than this
BASE_UNIFORMITY_MAX = 0.10       # luma std/mean above this = textured, reject
BASE_MIN_RG_RATIO = 1.2          # film base is strongly orange: R >= ratio*G
BASE_GB_TOL = 1.05               # ...and G*tol >= B

# --- frame percentiles (D_max / black / exposure pickers) --------------------
# P_LOW=2.0 (not 0.5): the dense-side anchor must be robust to small holder
# junk residues that survive the mask — the user's crop annotation on
# DSC_0001 proved the densest 0.5-2% were edge junk, not scene. Side effect:
# the true top speculars soft-clip, which matches the user's punch taste.
P_LOW = 2.0                      # per-channel percentile for "densest" values
P_HIGH = 99.5                    # per-channel percentile for "lightest" values
# Scan exposure bias: darktable's default, kept fixed. The auto formula needs
# "lightest part of the PHOTO content", but on uncropped scans the lightest
# area is the film base itself, which degenerates the formula to ~0 (and the
# user's manual rolls always keep the default anyway).
OFFSET_DEFAULT = -0.05

# D_max ("dynamic range of the film"): darktable's negadoctor init() default,
# kept FIXED — the user leaves it at the default in practice (~2.05) and never
# picks it. We used to auto-derive it via apply_auto_Dmax's formula
# (max_c log10(Dmin/picked_min)), but darktable's picker takes the MIN of the
# small densest region the user drags over, whereas our `picked_min` was the
# P_LOW percentile over the WHOLE content crop — far too bright, giving a
# fabricated D_max ~0.7 that exists nowhere in the real workflow and, since
# everything downstream divides by D_max, skewed offset/wb/black/exposure.
# Like OFFSET_DEFAULT, we just use darktable's default and let the print tune
# adapt black/exposure around it.
DMAX_DEFAULT = 2.046

# --- neutral patch search (on the rendered inverted preview) -----------------
PATCH_WIN_FRAC = 0.04
PATCH_STRIDE_DIV = 2
SHADOW_BAND_PCT = (5.0, 30.0)    # preview-luma percentile band for shadows patch
HIGHLIGHT_BAND_PCT = (70.0, 95.0)
SHADOW_MIN_LUMA = 0.005          # preview luma floor (skip film-base gaps)
PATCH_CHROMA_MAX = 0.35          # chroma of window means on the prior-wb preview
PATCH_CHROMA_FLOOR = 0.02        # chroma denominator floor (near-black windows)
PATCH_UNIFORMITY_MAX = 0.45      # preview luma std/mean guard (scenes are textured)
PATCH_LUMA_FLOOR = 0.02          # uniformity denominator floor (dark windows would
                                 # otherwise explode std/mean into grain noise)
MIN_PATCH_DENSITY = 0.05         # patch must be >= this much denser (log10 D) than
                                 # film base, else the wb formulas degenerate
HIGHLIGHT_CLIP_FRAC_MAX = 0.02   # negative-space clipped fraction guard

# --- print auto-tuning -------------------------------------------------------
# The user's process (2026-06-13): push brightness up to the CLIP BOUNDARY —
# "maximise brightness without the highlights clipping" — using black to keep
# brightening once exposure is maxed, and backing exposure off if it clips. This
# auto-preserves MOOD: a bright scene (snow/sky) has headroom so it goes bright,
# a genuinely dark scene (museum) clips early so it stays dark — no scene labels
# needed. NO HARD CLIPPING is the binding constraint (the param gate does NOT
# measure it; rendering the user's own GT params clips ~0% everywhere).
#   each iter: exposure pins the high percentile to PRINT_HI_CEIL (near clip);
#   when exposure SATURATES at its range end and the highlight is still off the
#   ceiling, BLACK takes over to keep pushing it there (this is how the user
#   brightens dense snow/sky — a less-negative black lifts the whole curve when
#   exposure can't go higher). A final guard lowers exposure until the hard-clip
#   fraction is within PRINT_CLIP_BUDGET. wb intensity feeds in for free.
# (Earlier tries failed: a fixed-0.95 highlight target / shadow-point black
# anchor matched the param gate but blew out 3-60% of pixels; a low fixed 0.86
# ceiling was clip-safe but left snow/sky far too dark — exposure maxed while
# black stayed too negative, so highlights sat at ~0.5.)
PRINT_HI_PCT = 99.9            # control the actual top (not P99.5, which lets the
                              # top 0.5% spread into hard clip)
# Highlight pin. Tuned to the GT PICTURE under the HIGH-RESOLUTION histogram
# metric (spec 04): the comparison now runs on the FLOAT render at 14-bit bins
# (the old 8-bit/64-bin metric was BLIND to the highlight shoulder — user
# feedback "you lack resolution"). Pinning P99.9 to 0.99 over-pushed the very top
# toward white, exactly where the user's GT highlights sit LOWER. Dropping the
# pin to 0.97 cut the top-highlight gap (median |ΔP99.9|) ~33% (0.0162->0.0108)
# and total EMD ~16% (0.0599->0.0505) across all 4 rolls, centred overall
# brightness (signed luma -0.029->-0.002), and stays clip-safe (<0.3%). (soft_clip
# is NOT a lever here — it cancels between the algorithm and GT renders, which
# share it.) Re-derive with tests/calibrate_histogram_match.py.
PRINT_HI_CEIL = 0.97          # pin the high percentile just below the GT's top
PRINT_CLIP_BUDGET = 0.003     # final guard: lower exposure until hard-clip frac
                              # (any channel >= 0.999) is at most this
PRINT_TUNE_ITERS = 12
PRINT_TUNE_SUBSAMPLE_FRAC = 0.003  # render every Nth pixel during tuning; stride
                                   # = frac of width (keeps sample count ~constant)

# --- print curve & wb taste (from user annotation feedback) -------------------
# Paper grade (print gamma). Tuned to match the GT PICTURE, not the GT params:
# spec 04 (specs/04_tune_algo_params_via_histograms.md) renders the user's
# hand-tuned annotations and minimizes the histogram (luma-EMD) distance between
# the algorithm's render and that GT render over BOTH annotated rolls. The print
# auto-tune pins highlights near clip via exposure, so the steep 6.1 curve (= GT
# *param* median) crushed the midtones DARKER than the GT picture (median signed
# luma +0.084 too dark on roll 2510-11-1). The exposure/wb<->picture mapping is
# many-to-one, so matching the GT picture needs a flatter curve than the GT
# gamma number: 5.0 is the joint luma-EMD minimum across both rolls (0.0782 ->
# 0.0573, clip-safe at <0.3%). The param-gate gamma delta gets WORSE by design
# (we optimize the render, not the param proxy). Re-derive with
# tests/calibrate_histogram_match.py if the rolls/export change. The print
# auto-tune adapts black/exposure around whatever gamma is set here.
PRINT_GAMMA = 5.0
# --- per-frame wb: estimate the cast HUE, apply a gentler-than-full correction
# The user's wheel picks are a per-frame neutralization of the scene's color
# cast, but consistently MILDER than a full gray-world neutralization (e.g. the
# tungsten-lit DSC_0002 inverts warm and the user only cools it partway). wb is
# read as the mean NEGATIVE-space color of a dark/bright print-luma BAND (robust
# region gray-world — most frames have no single clean neutral grey window, so
# the old single-window shadow search collapsed to a near-constant over-warm
# value), neutralized by darktable's exact picker formula, then pulled toward
# neutral by WB_*_DESAT. The intensity (spread) of wb is itself a tonal lever and
# is handled jointly: tune_print_params renders WITH the chosen wb and adapts
# black/exposure on that render, so a milder wb is compensated downstream.
WB_LOW_BAND_PCT = (5.0, 30.0)    # print-luma band whose mean = shadow cast
WB_HIGH_BAND_PCT = (70.0, 95.0)  # print-luma band whose mean = highlight cast
WB_LOW_DESAT = 0.45              # pull shadow-cast gain toward neutral (0=full)
WB_HIGH_DESAT = 0.0              # the light-neutral patch already lands well
WB_REGION_MIN_FRAC = 1e-4        # min band sample fraction of the cropped area
# preview wb for band SELECTION only (close to the final rendition so the luma
# bands fall on real shadows/highlights); no longer used as a result prior.
WB_HIGH_PRIOR = (1.80, 1.35, 1.0)
WB_LOW_PRIOR = (1.0, 0.75, 0.70)

# --- roll-wide vignette estimation --------------------------------------------
# Lens + backlight/holder vignetting darkens corners of the negative -> after
# inversion corners come out visibly brighter and skew all level analysis.
# Physics: nothing ON film is lighter than the unexposed base, so per pixel
# the MAX over (exposure-normalized) frames approaches V(x,y)*const wherever
# some frame is at base there — the spatial envelope IS the vignette field.
# Fitted in darktable's own lens-module "manual vignette" parameter space
# (v_strength/v_radius/v_steepness) so the values are directly usable there.
VIG_DOWNSAMPLE_FRAC = 0.004      # accumulate the envelope on a /N grid; stride
                                 # = frac of width (constant grid size vs resolution)
VIG_INSET_FRAC = 0.012           # extra inset past the dark-border trim (frac of width)
VIG_BINS = 32                    # radial bins for the envelope profile
VIG_PROFILE_PCT = 90.0           # per-bin envelope percentile
VIG_MIN_BIN_SAMPLES = 25         # bins with fewer samples are skipped
VIG_MIN_STRENGTH = 0.02          # below this the roll counts as vignette-free

# darktable lens-module params template (modversion 10, 356 bytes), taken
# from the user's manually tuned roll: lensfun method, Nikon D750 + AF-S
# Micro Nikkor 60mm f/2.8G ED. encode_lens_params() patches the manual
# vignette fields and clears the lensfun VIGNETTING modify-flag so the
# manual correction (fitted to our TOTAL estimate) is the sole vignette
# handler while lensfun keeps distortion/TCA. Rig-specific: re-dump from a
# manual XMP if the camera/lens ever changes.
LENS_TEMPLATE_HEX = (
    "0100000007000000000000000000803f0000803f0000704200009041bbe2403e010000004e696b6f6e204437353000000000"
    "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
    "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
    "00000000000000000000000000004e696b6f6e2041462d53204d6963726f204e696b6b6f722036306d6d20662f322e384720"
    "4544000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803f"
    "0000803f0000803f0000803f0000803f0000803f0000803f000000000000803f010000008941403ff0a7c63d0000003f0000"
    "000000000000"
)
LENS_MODFLAG_VIGNETTING = 1 << 1   # dt_iop_lens_modify_flag_t
LENS_V_OFFSET = 336                # v_strength/v_radius/v_steepness floats
LENS_MODFLAGS_OFFSET = 4

RESULTS_FILENAME = "negadoctor_results.txt"


# ---------------------------------------------------------------------------
# EXIF
# ---------------------------------------------------------------------------

def parse_exif_params(params_file):
    """Parse exif_params.txt written by the Lua side.

    Line format: filename|exposure=0.010000|aperture=5.600000|iso=100.000000
    Returns {stem: {"exposure_s": float, "aperture": float, "iso": float}}.
    """
    out = {}
    if not os.path.exists(params_file):
        return out
    with open(params_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            entry = {}
            for p in parts[1:]:
                if "=" in p:
                    k, v = p.split("=", 1)
                    try:
                        entry[k] = float(v)
                    except ValueError:
                        pass
            out[parts[0]] = {
                "exposure_s": entry.get("exposure"),
                "aperture": entry.get("aperture"),
                "iso": entry.get("iso"),
            }
    return out


def read_exif_fallback(jpeg_path):
    """Read shutter/aperture/ISO from the JPEG itself (darktable exports keep
    EXIF). Used when exif_params.txt is absent (tests, manual runs)."""
    try:
        from PIL import Image
        with Image.open(jpeg_path) as im:
            exif = im.getexif()
            sub = exif.get_ifd(0x8769) if exif else {}
        def fval(tag):
            v = sub.get(tag)
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        return {
            "exposure_s": fval(33434),   # ExposureTime
            "aperture": fval(33437),     # FNumber
            "iso": fval(34855),          # ISOSpeedRatings
        }
    except Exception:
        return {"exposure_s": None, "aperture": None, "iso": None}


# ---------------------------------------------------------------------------
# Lens-module params blob (gz04 + zlib, modversion 10)
# ---------------------------------------------------------------------------

def _dt_gz_encode(raw_bytes):
    """darktable's compressed-params encoding: 'gz' + 2-digit ratio + b64."""
    import zlib, base64
    compressed = zlib.compress(raw_bytes)
    ratio = min(len(raw_bytes) // max(len(compressed), 1) + 1, 99)
    return f"gz{ratio:02d}" + base64.b64encode(compressed).decode("ascii")


def _dt_gz_decode(blob):
    import zlib, base64
    return zlib.decompress(base64.b64decode(blob[4:]))


def encode_lens_params(vig):
    """Lens-module params for the roll: template patched with the fitted
    manual vignette and the lensfun VIGNETTING flag cleared (the manual
    correction fitted to our TOTAL estimate is the sole vignette handler)."""
    import struct as _struct
    raw = bytearray(bytes.fromhex(LENS_TEMPLATE_HEX))
    flags = _struct.unpack_from("<i", raw, LENS_MODFLAGS_OFFSET)[0]
    _struct.pack_into("<i", raw, LENS_MODFLAGS_OFFSET,
                      flags & ~LENS_MODFLAG_VIGNETTING)
    _struct.pack_into("<3f", raw, LENS_V_OFFSET,
                      float(vig["strength"]), float(vig["radius"]),
                      float(vig["steepness"]))
    return _dt_gz_encode(bytes(raw))


def decode_lens_vignette(blob):
    """(strength, radius, steepness) from a lens-module params blob."""
    import struct as _struct
    raw = _dt_gz_decode(blob)
    return _struct.unpack_from("<3f", raw, LENS_V_OFFSET)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def detect_dark_border(lin):
    """Detect film-holder remains: near-black margins at the frame edges.

    lin: (H,W,3) linear RGB. Returns (left, top, right, bottom) margins in px
    (already padded), each capped at BORDER_MAX_FRAC of the dimension.
    """
    luma = lin.mean(axis=2)
    h, w = luma.shape
    border_pad = max(1, int(round(w * BORDER_PAD_FRAC)))
    col_mean = luma.mean(axis=0)
    row_mean = luma.mean(axis=1)

    def scan(profile, limit):
        n = 0
        for v in profile[:limit]:
            if v >= BORDER_DARK_THR:
                break
            n += 1
        return min(n + border_pad, limit) if n > 0 else 0

    max_w = int(w * BORDER_MAX_FRAC)
    max_h = int(h * BORDER_MAX_FRAC)
    return (
        scan(col_mean, max_w),
        scan(row_mean, max_h),
        scan(col_mean[::-1], max_w),
        scan(row_mean[::-1], max_h),
    )


def detect_content_crop(lin, dmin):
    """Largest rectangular content area: trim each edge past the last scan
    line contaminated by holder-dark pixels or bright leak (lighter than the
    film base). Returns (left, top, right, bottom) margins in px."""
    h, w = lin.shape[:2]
    # size-dependent constants are fractions of the frame width
    gap_tol = max(1, int(round(w * CROP_GAP_TOL_FRAC)))
    shadow_max = max(1, int(round(w * CROP_SHADOW_MAX_FRAC)))
    shadow_core = max(1, int(round(w * CROP_SHADOW_CORE_FRAC)))
    crop_pad = max(1, int(round(w * CROP_PAD_FRAC)))
    rebate_term = max(1, int(round(w * CROP_REBATE_TERM_FRAC)))
    rebate_max = max(1, int(round(w * CROP_REBATE_MAX_FRAC)))
    luma = lin.mean(axis=2)
    dark = luma < HOLDER_LUMA_THR
    dmin_arr = np.maximum(np.asarray(dmin, dtype=np.float32), nm.THR)
    log_ratio = np.log10(dmin_arr[None, None, :] / np.maximum(lin, nm.THR))
    density_min = np.min(log_ratio, axis=2)
    density_max = np.max(log_ratio, axis=2)
    junk = dark | (density_min < -CROP_LEAK_MARGIN_D)
    base_like = density_max < CROP_REBATE_MARGIN_D

    def line_flags(mask, axis, thr):
        return mask.mean(axis=axis) > thr

    # holder-edge shadow: line mean luma well below the interior reference
    col_mean = luma.mean(axis=0)
    row_mean = luma.mean(axis=1)
    col_ref = float(np.median(col_mean[w // 4: 3 * w // 4]))
    row_ref = float(np.median(row_mean[h // 4: 3 * h // 4]))
    col_shadow = col_mean < CROP_SHADOW_REL * max(col_ref, nm.THR)
    row_shadow = row_mean < CROP_SHADOW_REL * max(row_ref, nm.THR)

    col_hard0 = line_flags(junk, 0, CROP_JUNK_LINE_FRAC)
    row_hard0 = line_flags(junk, 1, CROP_JUNK_LINE_FRAC)
    col_rebate = line_flags(base_like, 0, CROP_REBATE_LINE_FRAC)
    row_rebate = line_flags(base_like, 1, CROP_REBATE_LINE_FRAC)

    def validated_shadow(flags, limit):
        """Per-edge shadow validation: a run terminating within shadow_max
        (CROP_SHADOW_MAX_FRAC of width) is real penumbra; a band continuing
        deeper is dense scene and only the penumbra CORE near the edge counts."""
        depth = 0
        gap = 0
        for i in range(limit):
            if flags[i]:
                depth = i + 1
                gap = 0
            else:
                gap += 1
                if gap > gap_tol:
                    break
        out = np.zeros_like(flags)
        if depth == 0:
            return out
        if depth > shadow_max:
            depth = shadow_core
        out[:depth] = flags[:depth]
        return out

    def run_trim(flags, limit):
        # advance through the edge-anchored junk run (gap-tolerant); junk
        # detached from the edge is scene content, not holder/rebate
        depth = 0
        gap = 0
        for i in range(limit):
            if flags[i]:
                depth = i + 1
                gap = 0
            else:
                gap += 1
                if gap > gap_tol:
                    break
        return min(depth + crop_pad, limit) if depth else 0

    def trim(hard, rebate, limit):
        t_hard = run_trim(hard, limit)
        t_all = run_trim(hard | rebate, limit)
        if t_all > t_hard:
            # a base-like band wider than a real rebate strip is scene content
            # (bright sky/snow reads base-like) - reject the extension
            if t_all - t_hard > rebate_max:
                return t_hard
            # rebate-extended trim: only valid if the base-like band actually
            # terminates (unexposed scene continues, true rebate doesn't)
            end = t_all - crop_pad
            window = rebate[end:end + rebate_term]
            if t_all >= limit or len(window) == 0 or window.mean() > 0.5:
                return t_hard
        return t_all

    max_w = int(w * BORDER_MAX_FRAC)
    max_h = int(h * BORDER_MAX_FRAC)

    def edge(hard0, shadow, rebate, limit):
        return trim(hard0 | validated_shadow(shadow, limit), rebate, limit)

    return (
        edge(col_hard0, col_shadow, col_rebate, max_w),
        edge(row_hard0, row_shadow, row_rebate, max_h),
        edge(col_hard0[::-1], col_shadow[::-1], col_rebate[::-1], max_w),
        edge(row_hard0[::-1], row_shadow[::-1], row_rebate[::-1], max_h),
    )


def _box_mean(arr, win):
    """Mean of arr over win x win windows, evaluated at every pixel (border
    replicated). arr float32 (H,W) or (H,W,C)."""
    return cv2.boxFilter(arr, ddepth=-1, ksize=(win, win), normalize=True,
                         borderType=cv2.BORDER_REPLICATE)


def _grid_centers(h, w, border, win, stride):
    """Window-center coordinates on a stride grid, windows fully inside the
    border-trimmed area. Returns (ys, xs) index arrays (possibly empty)."""
    l, t, r, b = border
    half = win // 2
    y0, y1 = t + half, h - b - half - 1
    x0, x1 = l + half, w - r - half - 1
    if y1 < y0 or x1 < x0:
        return np.array([], dtype=int), np.array([], dtype=int)
    ys = np.arange(y0, y1 + 1, stride)
    xs = np.arange(x0, x1 + 1, stride)
    return ys, xs


def _rect_from_center(cx, cy, win):
    half = win // 2
    return [int(cx - half), int(cy - half), int(win), int(win)]


# ---------------------------------------------------------------------------
# Film base detection (negative space)
# ---------------------------------------------------------------------------

def find_film_base_candidate(lin, enc_f, border, win):
    """Find the lightest uniform orange window = local film-base candidate.

    lin: (H,W,3) float32 linear RGB; enc_f: (H,W,3) float32 stored-encoding
    values in [0,1] (sRGB for JPEG, linear for TIFF — used for clipping
    detection); border: margins from detect_dark_border; win: window px.

    Only WHITE clipping (all three channels at ceiling = backlight through
    holes) rejects a window; a JPEG export clips the orange base's R channel
    alone, and such windows are still the best Dmin estimate available there.

    Returns dict {rect:[x,y,w,h], rgb_linear:[r,g,b], score, clipped_r} or None.
    """
    h, w = lin.shape[:2]
    stride = max(win // BASE_STRIDE_DIV, 1)

    mean_rgb = _box_mean(lin, win)                       # (H,W,3)
    luma = lin.mean(axis=2)
    mean_luma = _box_mean(luma, win)
    mean_luma_sq = _box_mean(luma * luma, win)
    var = np.maximum(mean_luma_sq - mean_luma ** 2, 0.0)
    rel_std = np.sqrt(var) / np.maximum(mean_luma, 1e-9)
    white_clipped = (enc_f >= CLIP_SRGB_THR).all(axis=2).astype(np.float32)
    white_clip_frac = _box_mean(white_clipped, win)
    any_clipped = (enc_f >= CLIP_SRGB_THR).any(axis=2).astype(np.float32)
    any_clip_frac = _box_mean(any_clipped, win)

    ys, xs = _grid_centers(h, w, border, win, stride)
    if len(ys) == 0 or len(xs) == 0:
        return None
    grid = np.ix_(ys, xs)

    m = mean_rgb[grid]                                   # (ny,nx,3)
    ok = (
        (white_clip_frac[grid] < CLIP_FRAC_MAX)
        & (mean_luma[grid] >= BASE_MIN_LUMA)
        & (rel_std[grid] <= BASE_UNIFORMITY_MAX)
        & (m[..., 0] >= BASE_MIN_RG_RATIO * m[..., 1])   # strongly orange
        & (m[..., 1] * BASE_GB_TOL >= m[..., 2])
    )
    if not ok.any():
        return None

    score = np.where(ok, mean_luma[grid], -1.0)
    iy, ix = np.unravel_index(int(np.argmax(score)), score.shape)
    cy, cx = int(ys[iy]), int(xs[ix])
    return {
        "rect": _rect_from_center(cx, cy, win),
        "rgb_linear": [float(v) for v in mean_rgb[cy, cx]],
        "score": float(mean_luma[cy, cx]),
        "clipped_r": float(any_clip_frac[cy, cx]) > CLIP_FRAC_MAX,
    }


def choose_global_base(frames):
    """Pick the physically lightest film-base candidate across the roll.

    frames: list of per-frame dicts with "stem", "base" (or None) and
    "exposure_factor". Physical lightness = mean(rgb_linear) / factor: the
    same registered value under more light means a darker physical patch.

    Returns (winner_stem or None, winner_rgb, winner_factor).
    """
    best = None
    for fr in frames:
        if not fr["base"]:
            continue
        phys = float(np.mean(fr["base"]["rgb_linear"])) / fr["exposure_factor"]
        if best is None or phys > best[0]:
            best = (phys, fr)
    if best is None:
        return None, None, None
    fr = best[1]
    return fr["stem"], fr["base"]["rgb_linear"], fr["exposure_factor"]


def dmin_for_frame(winner_rgb, winner_factor, frame_factor):
    """Transfer the global film base to a frame with a different DSLR
    exposure: linear sensor response scales values by the factor ratio."""
    scale = frame_factor / winner_factor
    return [nm.clamp(c * scale, nm.DMIN_RANGE) for c in winner_rgb]


# ---------------------------------------------------------------------------
# Frame percentiles + neutral patches
# ---------------------------------------------------------------------------

def frame_percentiles(lin, enc_f, border, dmin):
    """Per-channel (picked_min, picked_max) over EVERYTHING inside the
    content crop — no per-pixel masks (user decision: the crop rectangle IS
    the analysis area). Junk robustness comes from the percentiles
    themselves (P_LOW=2.0). The only exclusion is sensor-clipped pixels on
    legacy 8-bit inputs (a no-op for the float TIFF export)."""
    l, t, r, b = border
    h, w = lin.shape[:2]
    region = lin[t:h - b, l:w - r]
    clip_region = (enc_f[t:h - b, l:w - r] >= CLIP_SRGB_THR).any(axis=2)
    mask = ~clip_region
    if not mask.any():
        mask = np.ones(region.shape[:2], dtype=bool)
    vals = region[mask]
    picked_min = [float(np.percentile(vals[:, c], P_LOW)) for c in range(3)]
    picked_max = [float(np.percentile(vals[:, c], P_HIGH)) for c in range(3)]
    return picked_min, picked_max


def find_neutral_patch(preview, lin, enc_f, border, win, band_pct, kind,
                       dmin, d_max, base_rect=None):
    """Find a low-chroma window on the inverted preview; return its rect and
    its NEGATIVE-space mean RGB (what darktable's picker would sample).

    The window grid lives entirely inside the content-crop border, so no
    pixel-level holder handling is needed here. preview: (H,W,3) render made
    with the TASTE-PRIOR wb — approximately the final rendition, so plain
    chroma means "looks gray in the print". (An uncorrected/bootstrap render
    plus gray-world normalization failed badly on half-lit frames: the
    global mean is dominated by one half and truly gray objects measure as
    the most chromatic windows.)
    lin: negative linear RGB; band_pct: luma percentile band on the preview;
    kind: "shadows"|"highlights".
    """
    h, w = lin.shape[:2]
    stride = max(win // PATCH_STRIDE_DIV, 1)
    l, t, r, b = border

    p_luma = preview.mean(axis=2).astype(np.float32)
    mean_pl = _box_mean(p_luma, win)
    mean_pl_sq = _box_mean(p_luma * p_luma, win)
    # luma floor: in near-black windows std/mean is pure grain noise
    rel_std = (np.sqrt(np.maximum(mean_pl_sq - mean_pl ** 2, 0.0))
               / np.maximum(mean_pl, PATCH_LUMA_FLOOR))

    mean_prgb = _box_mean(preview.astype(np.float32), win)
    # denominator floor: chroma ratios explode in near-black windows
    chroma = ((mean_prgb.max(axis=2) - mean_prgb.min(axis=2))
              / (mean_prgb.mean(axis=2) + PATCH_CHROMA_FLOOR))

    mean_neg = _box_mean(lin, win)
    clipped = (enc_f >= CLIP_SRGB_THR).any(axis=2).astype(np.float32)
    clip_frac = _box_mean(clipped, win)

    # luma band thresholds from the crop area
    region_luma = p_luma[t:h - b, l:w - r]
    lo = float(np.percentile(region_luma, band_pct[0]))
    hi = float(np.percentile(region_luma, band_pct[1]))

    ys, xs = _grid_centers(h, w, border, win, stride)
    if len(ys) == 0 or len(xs) == 0:
        return None
    grid = np.ix_(ys, xs)

    ok = (
        (mean_pl[grid] >= max(lo, SHADOW_MIN_LUMA))
        & (mean_pl[grid] <= hi)
        & (chroma[grid] <= PATCH_CHROMA_MAX)
        & (rel_std[grid] <= PATCH_UNIFORMITY_MAX)
    )
    mn = mean_neg[grid]
    # EVERY channel must be meaningfully denser than the film base: the wb
    # formulas divide by per-channel log10(Dmin/picked), so one near-base
    # channel explodes the ratios (e.g. thin blue layer in deep shadows)
    density = np.min(
        np.log10(np.maximum(np.asarray(dmin, dtype=np.float32), nm.THR)[None, None, :]
                 / np.maximum(mn, nm.THR)),
        axis=2,
    )
    ok &= density >= MIN_PATCH_DENSITY
    if kind == "highlights":
        ok &= clip_frac[grid] <= HIGHLIGHT_CLIP_FRAC_MAX

    if base_rect is not None:
        # exclude windows overlapping the film-base patch
        bx, by, bw, bh = base_rect
        half = win // 2
        yy = ys[:, None] * np.ones_like(xs)[None, :]
        xx = np.ones_like(ys)[:, None] * xs[None, :]
        overlap = (
            (xx + half >= bx) & (xx - half <= bx + bw)
            & (yy + half >= by) & (yy - half <= by + bh)
        )
        ok &= ~overlap

    if not ok.any():
        return None

    # best = least chromatic; tie-break by uniformity
    cost = chroma[grid] + 0.1 * rel_std[grid]
    cost = np.where(ok, cost, np.inf)
    iy, ix = np.unravel_index(int(np.argmin(cost)), cost.shape)
    cy, cx = int(ys[iy]), int(xs[ix])
    return {
        "rect": _rect_from_center(cx, cy, win),
        "rgb_neg_linear": [float(v) for v in mean_neg[cy, cx]],
        "chroma": float(chroma[cy, cx]),
        "preview_luma": float(mean_pl[cy, cx]),
        "used_fallback": False,
    }


def categorize_frame(preview_srgb, cache_path=None, cache_key=None):
    """Scene categorization hook (spec 03): a local vision LLM (gemma3 via
    Ollama) judges the already-inverted preview to steer the per-scene param
    nudge. Delegates to scene_tuner; returns the label dict or None on any
    failure (the caller then keeps the analytical params). Only invoked on the
    opt-in --ai-tune path."""
    import scene_tuner
    return scene_tuner.categorize_scene(preview_srgb, cache_path=cache_path,
                                        cache_key=cache_key)


# ---------------------------------------------------------------------------
# Per-frame parameter assembly
# ---------------------------------------------------------------------------

def _normalize_wb(wb, kind):
    """Restore the darktable picker normalization: max(wb_low)=1 / min(wb_high)=1."""
    m = max(wb) if kind == "shadows" else min(wb)
    return [nm.clamp(v / max(m, nm.THR), nm.WB_RANGE) for v in wb]


def desaturate_wb(wb, strength, kind):
    """Pull a wb gain toward neutral (1,1,1) by `strength` (0=unchanged,
    1=neutral), preserving hue, then restore the picker normalization. The
    user neutralizes the cast more gently than a full gray-world correction."""
    blended = [(1.0 - strength) * wb[c] + strength * 1.0 for c in range(3)]
    return _normalize_wb(blended, kind)


def estimate_region_wb(lin, border, preview, dmin, d_max, offset, band_pct,
                       kind, wb_low=None):
    """Robust region gray-world: mean NEGATIVE-space color over a print-luma
    band -> neutralized wb via darktable's picker formula (no single-window
    grey patch needed). preview defines the luma band on the (prior-wb)
    rendition; lin supplies the negative colors the picker would sample.
    kind: "shadows" -> wb_low, "highlights" -> wb_high. Returns a normalized
    wb (not yet desaturated) or None when the band has too few samples."""
    l, t, r, b = border
    h, w = lin.shape[:2]
    reg = lin[t:h - b, l:w - r].reshape(-1, 3)
    if reg.shape[0] == 0:
        return None
    p_luma = preview[t:h - b, l:w - r].mean(axis=2).reshape(-1)
    lo = float(np.percentile(p_luma, band_pct[0]))
    hi = float(np.percentile(p_luma, band_pct[1]))
    sel = (p_luma >= lo) & (p_luma <= hi)
    if int(sel.sum()) < max(10, int(round(reg.shape[0] * WB_REGION_MIN_FRAC))):
        return None
    mean_neg = [float(v) for v in reg[sel].mean(axis=0)]
    if kind == "shadows":
        return nm.compute_wb_low(dmin, mean_neg, d_max)
    return nm.compute_wb_high(dmin, mean_neg, d_max, offset, wb_low or [1.0, 1.0, 1.0])


def make_params(dmin, d_max, offset, wb_low, wb_high, picked_min, picked_max,
                gamma=None):
    """Final black/exposure with the chosen wb, then the full param dict.

    gamma defaults to PRINT_GAMMA for final params; the patch-search preview
    renders pass it explicitly (the value barely matters there - patch
    selection is percentile/chroma based)."""
    black = nm.compute_black(dmin, picked_max, d_max, wb_high, wb_low, offset)
    exposure = nm.compute_exposure(dmin, picked_min, d_max, wb_high, wb_low,
                                   offset, black)
    return {
        "film_stock": 1,
        "Dmin": list(dmin),
        "wb_high": list(wb_high),
        "wb_low": list(wb_low),
        "D_max": d_max,
        "offset": offset,
        "black": black,
        "gamma": PRINT_GAMMA if gamma is None else gamma,
        "soft_clip": nm.SOFT_CLIP_DEFAULT,
        "exposure": exposure,
    }


def render_preview_srgb(lin, params):
    out = nm.render_negadoctor(lin, params)
    return (nm.linear_to_srgb(out) * 255.0 + 0.5).astype(np.uint8)


def tune_print_params(lin, params, border, dmin, hi_ceil=None):
    """Push brightness to the clip boundary (the user's process): exposure pins
    the high percentile (PRINT_HI_PCT) to the near-clip ceiling PRINT_HI_CEIL;
    when exposure SATURATES at its range end and the highlight is still off the
    ceiling, BLACK takes over to keep brightening (a less-negative black lifts
    the whole curve on dense snow/sky once exposure can't go higher). A final
    guard backs exposure off until the hard-clip fraction is within
    PRINT_CLIP_BUDGET. This auto-preserves mood — bright scenes use the headroom
    and go bright, dark scenes clip early and stay dark — with NO hard clipping
    as the binding constraint (see the PRINT_* block).

    Statistics run over EVERYTHING inside the content-crop border (no pixel
    masks). darktable's auto formulas anchor the curve endpoints in PRE-gamma
    space, which leaves full-range scans flat and dark; this closes the loop on
    the actual rendered output instead. Works on a subsampled frame.

    hi_ceil overrides PRINT_HI_CEIL (the pin target); the AI scene-tuner
    (scene_tuner.apply_scene_tuning) lowers it on dark/moody scenes so they
    stay dark. The hard-clip guard is unconditional regardless of hi_ceil.
    Returns (tuned params, tuning info dict).
    """
    ceil = PRINT_HI_CEIL if hi_ceil is None else hi_ceil
    l, t, r, b = border
    h, w = lin.shape[:2]
    s = max(1, int(round(w * PRINT_TUNE_SUBSAMPLE_FRAC)))
    region = lin[t:h - b:s, l:w - r:s].astype(np.float64)
    if region.size == 0:
        return params, {"tuned": False}

    p = dict(params)
    gamma = p["gamma"]
    ec = nm.EXPOSURE_RANGE
    content = region.reshape(-1, 3)
    info = {"tuned": True}
    for it in range(PRINT_TUNE_ITERS):
        # exposure pins the high percentile to the ceiling (both directions)
        out = nm.render_negadoctor(content, p)
        hi = float(np.percentile(out.mean(axis=1), PRINT_HI_PCT))
        if hi > 1e-6:
            p["exposure"] = nm.clamp(
                p["exposure"] * (ceil / hi) ** (1.0 / gamma), ec)
        # when exposure is SATURATED (maxed and still below ceiling, or bottomed
        # and above it), black takes over to drive the highlight to the ceiling
        out = nm.render_negadoctor(content, p)
        hi = float(np.percentile(out.mean(axis=1), PRINT_HI_PCT))
        info["hi"] = hi
        sat_hi = p["exposure"] >= ec[1] - 1e-6
        sat_lo = p["exposure"] <= ec[0] + 1e-6
        if (sat_hi and hi < ceil) or (sat_lo and hi > ceil):
            delta_pre = (ceil ** (1.0 / gamma)
                         - max(hi, 1e-6) ** (1.0 / gamma))
            p["black"] = nm.clamp(p["black"] + delta_pre / max(p["exposure"], 1e-6),
                                  nm.BLACK_RANGE)
    # final hard-clip guard: back exposure off until clipping is within budget
    for _ in range(6):
        out = nm.render_negadoctor(content, p)
        clip = float(np.mean((out >= 0.999).any(axis=1)))
        info["clip"] = clip
        if clip <= PRINT_CLIP_BUDGET:
            break
        p["exposure"] = nm.clamp(p["exposure"] * 0.97, ec)
    return p, info


def _downscale_for_llm(lin, max_side=512):
    """Shrink a linear frame so its longest side is <= max_side (area-averaged),
    for the LLM preview only. No-op if already small."""
    h, w = lin.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return lin
    return cv2.resize(lin, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


def run_scene_tuning(frames, session_dir=None):
    """Opt-in (--ai-tune) ALTERNATE variant: a local vision LLM reads each
    frame's already-rendered inverted preview and nudges the analytical params
    per scene (scene_tuner.apply_scene_tuning). Stores fr["scene"],
    fr["params_ai"], fr["params_ai_hex"] WITHOUT touching the analytical
    fr["params"]. Degrades silently to the analytical params per frame whenever
    the LLM is unavailable. The cropping/vignette/film-base stages already ran."""
    import scene_tuner
    cache_path = (os.path.join(str(session_dir), "scene_cache.json")
                  if session_dir else None)
    print(f"AI scene tuning: vision LLM ({scene_tuner.OLLAMA_MODEL}) per frame "
          f"(workers={_AI_WORKERS})...", flush=True)

    # GPU-bound (one Ollama model) and the response cache is a whole-file
    # read-modify-write, so this stays serial by default (NEGA_AI_WORKERS=1) —
    # raising it past 1 only queues on the GPU and risks racing the cache.
    def tune_one(fr):
        if fr.get("error") or "params" not in fr:
            return fr
        try:
            enc_f, lin = _get_loaded(fr)
            # The LLM only needs a ~512px image (scene_tuner downscales to that
            # anyway), so render the preview from a DOWNSCALED frame — rendering
            # the full 2000px frame for the LLM is wasted CPU per frame.
            llm_lin = _downscale_for_llm(lin)
            preview = render_preview_srgb(llm_lin, fr["params"])
            scene = categorize_frame(preview, cache_path=cache_path,
                                     cache_key=fr["stem"])
            fr["scene"] = scene
            params_ai, info = scene_tuner.apply_scene_tuning(
                fr["params"], scene, lin, fr["border"])
            fr["params_ai"] = params_ai
            fr["scene_tuning"] = info
            fr["params_ai_hex"] = nm.encode_negadoctor_params(params_ai)
            tag = (f"{scene['scene']}/{scene['mood']}/{scene['warmth']}/"
                   f"{scene['contrast']}" if scene else "no-scene (analytical)")
            print(f"  {fr['stem']}: {tag}", flush=True)
        except Exception as e:
            fr["scene"] = None
            fr["scene_tuning"] = {"tuned": False, "error": str(e)}
            # fall back to the analytical params for this frame
            fr["params_ai"] = dict(fr["params"])
            fr["params_ai_hex"] = fr["params_hex"]
            print(f"  {fr['stem']}: AI tuning failed ({e}); using analytical",
                  flush=True)
        return fr

    _map_frames(tune_one, frames, _AI_WORKERS)


# ---------------------------------------------------------------------------
# Roll processing
# ---------------------------------------------------------------------------

def estimate_vignette(image_paths, exif_by_stem):
    """Roll-wide vignette estimate (see the VIG_* constants block).

    Returns (params dict {strength, radius, steepness} or None,
             info dict {residual, corner_falloff, bins, frames})."""
    # Per-frame downsampled valid-luma map (the heavy load + border + luma work)
    # is independent, so compute it frame-parallel; the np.maximum envelope fold
    # is serial but cheap and stays in input order (first frame sets the shape).
    def _vig_frame(path):
        try:
            enc_f, lin = load_frame(path)
        except Exception:
            return None
        h, w = lin.shape[:2]
        stem = Path(path).stem
        exif = exif_by_stem.get(stem) or read_exif_fallback(path)
        factor, _missing = nm.exposure_factor(
            exif.get("exposure_s"), exif.get("iso"), exif.get("aperture"))
        l, t, r, b = detect_dark_border(lin)
        luma = lin.mean(axis=2) / max(factor, 1e-12)
        valid = np.zeros((h, w), dtype=bool)
        vig_inset = max(1, int(round(w * VIG_INSET_FRAC)))
        vig_ds = max(1, int(round(w * VIG_DOWNSAMPLE_FRAC)))
        y0, y1 = t + vig_inset, h - b - vig_inset
        x0, x1 = l + vig_inset, w - r - vig_inset
        if y1 <= y0 or x1 <= x0:
            return None
        valid[y0:y1, x0:x1] = True
        if enc_f.max() > 0:   # 8/16-bit inputs: exclude clipped pixels
            valid &= ~(enc_f >= CLIP_SRGB_THR).any(axis=2)
        lum = np.where(valid, luma, -np.inf)
        return (h, w), lum[::vig_ds, ::vig_ds]

    paths = list(image_paths)
    per_frame = _map_frames(_vig_frame, paths, _proc_workers(len(paths)))
    acc = None
    shape = None
    used = 0
    for item in per_frame:
        if item is None:
            continue
        (h, w), ds = item
        if shape is None:
            shape = (h, w)
        elif (h, w) != shape:
            continue
        acc = ds if acc is None else np.maximum(acc, ds)
        used += 1
    if acc is None or used < 2:
        return None, {"frames": used, "reason": "too few frames"}

    h, w = shape
    hh, ww = acc.shape
    # same downsample stride used to build acc above (shape is shared across
    # all accumulated frames, so this matches the per-frame vig_ds)
    vig_ds = max(1, int(round(w * VIG_DOWNSAMPLE_FRAC)))
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float64) * vig_ds
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    rad = np.hypot(xx - cx, yy - cy) / np.hypot(cx, cy)
    finite = np.isfinite(acc)

    r_centers, envs = [], []
    for k in range(VIG_BINS):
        sel = finite & (rad >= k / VIG_BINS) & (rad < (k + 1) / VIG_BINS)
        if sel.sum() < VIG_MIN_BIN_SAMPLES:
            continue
        env = float(np.percentile(acc[sel], VIG_PROFILE_PCT))
        if env <= 0:
            continue
        r_centers.append((k + 0.5) / VIG_BINS)
        envs.append(env)
    if len(r_centers) < 6:
        return None, {"frames": used, "reason": "too few radial bins"}

    return fit_vignette_profile(r_centers, envs, used)


def fit_vignette_profile(r_centers, envs, used=None):
    """Turn a radial envelope profile into lens-module vignette params.

    Pure (no image data) so it is unit-testable on captured roll profiles:
    `r_centers` are bin-centre radii in [0,1], `envs` the per-bin envelope
    brightness. Returns (params dict or None, info dict) exactly as
    estimate_vignette does. Split out 2026-06-13 so the central-dip
    regression (roll 2511-12-1) can be tested without the 1.2GB TIFFs."""
    # reference = envelope of the CENTER bins (the global max pixel is an
    # outlier and would shift the whole curve)
    center_envs = [e for r, e in zip(r_centers, envs) if r < 0.15]
    env_ref = max(center_envs) if center_envs else envs[0]
    targets_all = [max(env_ref / e, 1.0) for e in envs]

    # the true correction profile is monotone non-decreasing in r; a falling
    # tail means the base ceiling broke there (corner bright leak surviving
    # the inset) — cut it and let the model extrapolate the corner. The cut
    # applies only OUTWARD from the envelope peak: the brightest ring is not
    # always the dead centre (a slightly dimmed/noisy core, as on some rolls,
    # makes the innermost bins read target>1 falling to 1 at the peak), and
    # those leading bins are the flat centre plateau — not a corner leak. So
    # keep everything up to the peak and run the monotone/tail cut from it.
    peak_i = int(np.argmin(targets_all))   # min target == max envelope (peak)
    r_keep = list(r_centers[:peak_i + 1])
    t_keep = list(targets_all[:peak_i + 1])
    cm = targets_all[peak_i]
    for r, t in zip(r_centers[peak_i + 1:], targets_all[peak_i + 1:]):
        if t < cm * 0.97:
            break
        cm = max(cm, t)
        r_keep.append(r)
        t_keep.append(t)
    r_centers, targets = r_keep, t_keep
    if len(r_centers) < 6:
        return None, {"frames": used, "reason": "profile not vignette-like"}

    params, resid = nm.fit_vignette(r_centers, targets)
    corner_mult = float(nm.vignette_spline(1.0, **params))
    info = {"frames": used, "bins": len(r_centers),
            "residual": round(resid, 4),
            "corner_falloff": round(1.0 - 1.0 / corner_mult, 4),
            "profile": [[round(r, 3), round(t, 4)]
                        for r, t in zip(r_centers, targets)]}
    if params["strength"] < VIG_MIN_STRENGTH:
        return None, dict(info, reason="vignette negligible")
    return params, info


_VIG_FIELD_CACHE = {}
_VIG_FIELD_LOCK = threading.Lock()


def _vignette_field_cached(h, w, vig):
    key = (h, w, vig["strength"], vig["radius"], vig["steepness"])
    # locked so the frame-parallel stages compute the (identical) field once
    # instead of every worker racing to build the same big array
    with _VIG_FIELD_LOCK:
        field = _VIG_FIELD_CACHE.get(key)
        if field is None:
            field = nm.vignette_field(h, w, vig).astype(np.float32)
            _VIG_FIELD_CACHE[key] = field
    return field


# ---------------------------------------------------------------------------
# Export color management
# ---------------------------------------------------------------------------
# The Lua export's `icctype=LIN_REC2020` override can SILENTLY FAIL, leaving
# darktable to embed an sRGB profile (sRGB primaries, D50 white, gamma TRC).
# negadoctor (default_colorspace == IOP_CS_RGB) consumes the WORKING profile
# (linear Rec2020 D65) BEFORE colorout, so analysing the sRGB export as-if-linear
# is wrong on ALL THREE axes — primaries, white point AND tone (the gamma lifts
# the shadows, which fabricated the tiny D_max and the over-bright exposure that
# blew the inversion out in darktable). We read the TIFF's EMBEDDED ICC and
# convert every pixel into linear Rec2020 D65 = negadoctor's actual input. A
# genuine linear-Rec2020 export round-trips to ~identity, so this stays correct
# once the export override is fixed too. (Verified: the reference roll's film
# base decodes to darktable's own picker value within ~2%.)

# Rec2020 (D65) <-> XYZ
_M_REC2020_TO_XYZ = np.array([[0.6369580, 0.1446169, 0.1688810],
                              [0.2627002, 0.6779981, 0.0593017],
                              [0.0000000, 0.0280727, 1.0609851]], dtype=np.float64)
_M_XYZ_TO_REC2020 = np.linalg.inv(_M_REC2020_TO_XYZ)


def _read_icc_from_tiff(path):
    """Embedded ICC profile bytes (TIFF tag 34675), or None — no extra deps."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    bo = "<" if data[:2] == b"II" else ">" if data[:2] == b"MM" else None
    if bo is None:
        return None
    try:
        ifd = struct.unpack(bo + "I", data[4:8])[0]
        while ifd:
            n = struct.unpack(bo + "H", data[ifd:ifd + 2])[0]
            for i in range(n):
                e = ifd + 2 + i * 12
                tag, _typ, cnt = struct.unpack(bo + "HHI", data[e:e + 8])
                if tag == 34675:
                    off = (e + 8 if cnt <= 4
                           else struct.unpack(bo + "I", data[e + 8:e + 12])[0])
                    return data[off:off + cnt]
            ifd = struct.unpack(bo + "I",
                                data[ifd + 2 + n * 12:ifd + 2 + n * 12 + 4])[0]
    except (struct.error, IndexError):
        return None
    return None


def _icc_tag_table(icc):
    n = struct.unpack(">I", icc[128:132])[0]
    tags = {}
    for i in range(n):
        o = 132 + i * 12
        sig = icc[o:o + 4].decode("ascii", "ignore")
        a, sz = struct.unpack(">II", icc[o + 4:o + 12])
        tags[sig] = icc[a:a + sz]
    return tags


def _icc_xyz(d):
    return np.array([v / 65536.0 for v in struct.unpack(">3i", d[8:20])],
                    dtype=np.float64)


def _icc_mat3(d):
    return np.array([v / 65536.0 for v in struct.unpack(">9i", d[8:44])],
                    dtype=np.float64).reshape(3, 3)


def _icc_trc_is_linear(d):
    """True when a TRC tag encodes the identity (linear) response."""
    typ = d[:4]
    if typ == b"curv":
        cnt = struct.unpack(">I", d[8:12])[0]
        if cnt == 0:
            return True
        if cnt == 1:
            return abs(struct.unpack(">H", d[12:14])[0] / 256.0 - 1.0) < 1e-3
        pts = np.frombuffer(d[12:12 + 2 * cnt], dtype=">u2").astype(np.float64)
        return float(np.abs(pts - np.linspace(0, 65535, cnt)).max()) < 64.0
    if typ == b"para":
        return (struct.unpack(">H", d[8:10])[0] == 0
                and abs(struct.unpack(">i", d[12:16])[0] / 65536.0 - 1.0) < 1e-3)
    return False


def _srgb_eotf(x):
    a = np.abs(x)
    return np.sign(x) * np.where(a <= 0.04045, a / 12.92,
                                 ((a + 0.055) / 1.055) ** 2.4)


def lin_rec2020_from_export(rgb, icc):
    """Encoded export RGB -> linear Rec2020 D65 via the embedded matrix ICC.
    Non-linear TRC -> sRGB EOTF decode (the broken-export case); the
    device->XYZ(D50) matrix and chromatic-adaptation tag (chad, D65->D50) un-adapt
    to D65, then map into Rec2020. Best effort: returns the input unchanged (i.e.
    treated as already-linear-Rec2020) on any parse failure."""
    try:
        t = _icc_tag_table(icc)
        lin = rgb if _icc_trc_is_linear(t["rTRC"]) else _srgb_eotf(rgb)
        m_dev = np.column_stack([_icc_xyz(t["rXYZ"]), _icc_xyz(t["gXYZ"]),
                                 _icc_xyz(t["bXYZ"])])              # dev->XYZ(D50)
        chad = _icc_mat3(t["chad"]) if "chad" in t else np.eye(3)   # D65->D50
        flat = lin.reshape(-1, 3).astype(np.float64)
        xyz_d65 = (flat @ m_dev.T) @ np.linalg.inv(chad).T
        return (xyz_d65 @ _M_XYZ_TO_REC2020.T).reshape(rgb.shape).astype(np.float32)
    except Exception as e:   # pragma: no cover - defensive
        print(f"WARNING: could not color-manage export via embedded ICC ({e}); "
              f"treating it as linear Rec2020")
        return rgb.astype(np.float32)


def load_frame(path, vignette=None):
    """Load an export -> (enc_f float32 RGB [0,1], lin float32 linear Rec2020).

    vignette: roll-wide correction params; when given, lin is multiplied by
    the darktable manual-vignette correction field (corners brightened) so
    ALL downstream analysis sees vignette-corrected data. enc_f (clipping
    detection on stored values) stays uncorrected.

    enc_f holds the stored-encoding values (for clipping detection):
      - .tif/.tiff: the Lua export. We color-manage it to linear Rec2020 D65
        (= negadoctor's IOP_CS_RGB working space) via the EMBEDDED ICC profile —
        the icctype override can fail and leave an sRGB profile, which must be
        decoded, not treated as linear. See lin_rec2020_from_export.
      - 8-bit sRGB JPEG/PNG: linearize + sRGB->Rec2020 matrix (fallback;
        film-base R is typically gamut-clipped there)
    """
    def finish(enc_f, lin):
        if vignette and vignette.get("strength", 0.0) > 0.0:
            field = _vignette_field_cached(lin.shape[0], lin.shape[1], vignette)
            lin = lin * field[:, :, None]
        return enc_f, lin

    ext = Path(path).suffix.lower()
    if ext in (".tif", ".tiff"):
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise RuntimeError(f"cannot read image: {path}")
        if raw.ndim == 3 and raw.shape[2] >= 3:
            raw = raw[:, :, :3]
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        icc = _read_icc_from_tiff(path)   # cv2 strips metadata; read it raw
        if rgb.dtype in (np.float32, np.float64):
            # 32-bit float TIFF (the Lua export). Color-manage via the embedded
            # profile to negadoctor's working space; nothing clips so clip
            # detection is disabled (enc_f = zeros).
            enc = rgb.astype(np.float32)
            lin = (lin_rec2020_from_export(enc, icc) if icc is not None else enc)
            lin = np.maximum(lin, 0.0)
            return finish(np.zeros_like(lin), lin)
        # 16-bit integer TIFF clamps the pipeline at 1.0 — keep clip detection on
        # the stored (encoded) values; color-manage the analysis copy.
        scale = 65535.0 if rgb.dtype == np.uint16 else 255.0
        enc = rgb.astype(np.float32) / scale
        lin = (lin_rec2020_from_export(enc, icc) if icc is not None else enc.copy())
        return finish(enc, np.maximum(lin, 0.0))
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    enc_f = rgb.astype(np.float32) / 255.0
    lin = nm.linearize_export(rgb).astype(np.float32)
    return finish(enc_f, lin)


def process_roll(image_paths, exif_by_stem, progress=None, ai_tune=False,
                 session_dir=None):
    """Run the full analysis. Returns list of per-frame result dicts.

    ai_tune: after the analytical params are computed, add a vision-LLM
    per-scene nudge as an ALTERNATE variant (params_ai / params_ai_hex on each
    frame; analytical params untouched). session_dir holds the LLM response
    cache. See run_scene_tuning()."""
    n = len(image_paths)
    paths = list(image_paths)
    workers = _proc_workers(n)
    prog = _Progress(progress, 3 * n)

    # ---- Stage 0: roll-wide vignette estimation (uncorrected frames) ----
    vig, vig_info = estimate_vignette(image_paths, exif_by_stem)
    if vig:
        print(f"Vignette estimate: strength={vig['strength']:.3f} "
              f"radius={vig['radius']:.3f} steepness={vig['steepness']:.3f} "
              f"(corner falloff {vig_info['corner_falloff']:.1%}, "
              f"fit residual {vig_info['residual']}) — darktable lens module "
              f"manual vignette values")
    else:
        print(f"Vignette: none applied ({vig_info.get('reason', '?')})")
    prog.tick(n)

    # ---- Stage A: per-frame film-base candidates (frame-parallel) ----
    def stage_a(path):
        stem = Path(path).stem
        fr = {"stem": stem, "path": str(path), "error": None, "base": None,
              "vignette": vig}
        t0 = time.perf_counter()
        try:
            enc_f, lin = _get_loaded(fr)
            h, w = lin.shape[:2]
            fr["width"], fr["height"] = int(w), int(h)
            # float TIFFs disable clip detection (enc_f all zeros); a 16-bit
            # TIFF here means the Lua bpp=32 change isn't active and the
            # pipeline was clamped at 1.0 (clips the film base's R channel)
            fr["source_16bit_tiff"] = (Path(path).suffix.lower() in (".tif", ".tiff")
                                       and bool(enc_f.max() > 0))
            exif = exif_by_stem.get(stem) or read_exif_fallback(path)
            factor, missing = nm.exposure_factor(
                exif.get("exposure_s"), exif.get("iso"), exif.get("aperture"))
            fr["exif"] = {
                "exposure_s": exif.get("exposure_s"),
                "aperture": exif.get("aperture"),
                "iso": exif.get("iso"),
                "exposure_factor": factor,
                "missing_exif": missing,
            }
            fr["exposure_factor"] = factor
            fr["border"] = detect_dark_border(lin)
            win = max(int(w * BASE_WIN_FRAC), int(round(w * MIN_WIN_FRAC)))
            fr["base_win"] = win
            fr["base"] = find_film_base_candidate(lin, enc_f, fr["border"], win)
            # picker percentiles need the frame's Dmin (content masking) and
            # are computed in stage B1 once the global base is known
        except Exception as e:
            fr["error"] = str(e)
        fr["time_a"] = time.perf_counter() - t0
        return fr

    frames = _map_frames(stage_a, paths, workers, on_done=prog.tick)

    # ---- Global film base ----
    good = [f for f in frames if not f["error"]]
    n16 = sum(1 for f in good if f.get("source_16bit_tiff"))
    if n16:
        print(f"WARNING: {n16} frame(s) are 16-bit TIFF - the pipeline was "
              f"clamped at 1.0 and the film base's R channel is clipped. "
              f"Reload auto_negadoctor.lua in darktable so exports use "
              f"32-bit float TIFF.")
    winner_stem, winner_rgb, winner_factor = choose_global_base(good)
    if winner_stem is None:
        for fr in good:
            fr["error"] = "no film-base candidate found on any frame of the roll"
        for fr in frames:
            fr.pop("_loaded", None)
        return frames, None
    winner_rect = next(f["base"]["rect"] for f in good if f["stem"] == winner_stem)
    print(f"Global film base: {winner_stem} rect={winner_rect} "
          f"rgb={['%.4f' % v for v in winner_rgb]} factor={winner_factor:.5g}")

    # ---- Stage B1: per-frame patches + wb (frame-parallel) ----
    def stage_b1(fr):
        if fr["error"]:
            return fr
        t0 = time.perf_counter()
        try:
            enc_f, lin = _get_loaded(fr)
            dmin = dmin_for_frame(winner_rgb, winner_factor, fr["exposure_factor"])
            # the content crop replaces the stage-A dark-border trim as THE
            # analysis area for everything from here on. Crop detection runs
            # on UNCORRECTED data: the junk-band geometry doesn't change with
            # vignette correction, and the user's hard-truth crop fixtures
            # were drawn on uncorrected renders (corrected edges trim 1-3px
            # less and broke containment)
            if vig:
                field = _vignette_field_cached(lin.shape[0], lin.shape[1], vig)
                lin_raw = lin / field[:, :, None]
            else:
                lin_raw = lin
            fr["border"] = detect_content_crop(lin_raw, dmin)
            fr["picked_min"], fr["picked_max"] = frame_percentiles(
                lin, enc_f, fr["border"], dmin)
            # D_max is FIXED at darktable's default (the user never picks it);
            # see DMAX_DEFAULT. picked_min/picked_max are still used for the
            # black/exposure pickers and the wb bootstrap.
            d_max = DMAX_DEFAULT
            offset = OFFSET_DEFAULT
            fr["dmin"], fr["d_max"], fr["offset"] = dmin, d_max, offset

            win = max(int(lin.shape[1] * PATCH_WIN_FRAC),
                      int(round(lin.shape[1] * MIN_WIN_FRAC)))
            base_rect = fr["base"]["rect"] if fr["base"] else None
            neutral = [1.0, 1.0, 1.0]

            # percentile bootstrap, kept as the wb_high fallback of last resort
            wb_high_boot = nm.compute_wb_high(dmin, fr["picked_min"], d_max,
                                              offset, neutral)
            fr["wb_high_boot"] = wb_high_boot

            # ONE preview rendered with the prior-wb: close to the final
            # rendition, so the print-luma bands fall on real shadows /
            # highlights (and "low chroma" means "gray in the print" for the
            # still-used highlight patch). The prior-wb auto black/exposure
            # preview is verified non-clipping on the reference roll, so the
            # bands aren't skewed by blown highlights. See estimate_region_wb.
            params_prior = make_params(dmin, d_max, offset,
                                       list(WB_LOW_PRIOR), list(WB_HIGH_PRIOR),
                                       fr["picked_min"], fr["picked_max"])
            preview = nm.render_negadoctor(lin, params_prior)

            # wb_low: region gray-world over the shadow band, gently neutralized
            # (the single-window shadow search collapsed to a near-constant
            # over-warm value; the user's pick is a milder cast correction)
            wb_low_raw = estimate_region_wb(lin, fr["border"], preview, dmin,
                                            d_max, offset, WB_LOW_BAND_PCT,
                                            "shadows")
            wb_low = (desaturate_wb(wb_low_raw, WB_LOW_DESAT, "shadows")
                      if wb_low_raw else None)

            # wb_high: prefer the reliable light-neutral PATCH; fall back to the
            # bright-region cast estimate when no clean patch exists (e.g.
            # tungsten frames that have no neutral highlight window)
            highlight = find_neutral_patch(preview, lin, enc_f, fr["border"], win,
                                           HIGHLIGHT_BAND_PCT, "highlights",
                                           dmin, d_max, base_rect)
            if highlight:
                wb_high = nm.compute_wb_high(dmin, highlight["rgb_neg_linear"],
                                             d_max, offset, wb_low or neutral)
            else:
                wb_high_raw = estimate_region_wb(lin, fr["border"], preview,
                                                 dmin, d_max, offset,
                                                 WB_HIGH_BAND_PCT, "highlights",
                                                 wb_low or neutral)
                wb_high = (desaturate_wb(wb_high_raw, WB_HIGH_DESAT, "highlights")
                           if wb_high_raw else None)

            # shadow patch kept only as a UI marker (wb_low now region-based)
            shadow = find_neutral_patch(preview, lin, enc_f, fr["border"], win,
                                        SHADOW_BAND_PCT, "shadows",
                                        dmin, d_max, base_rect)
            fr["shadow_patch"] = shadow
            fr["highlight_patch"] = highlight
            fr["wb_low"] = wb_low
            fr["wb_high"] = wb_high
        except Exception as e:
            fr["error"] = str(e)
        fr["time_b"] = time.perf_counter() - t0
        return fr

    frames = _map_frames(stage_b1, frames, workers, on_done=prog.tick)

    # ---- Roll-median fallback for frames without usable patches ----
    lows = [fr["wb_low"] for fr in frames if not fr["error"] and fr.get("wb_low")]
    highs = [fr["wb_high"] for fr in frames if not fr["error"] and fr.get("wb_high")]
    median_low = ([statistics.median(v[c] for v in lows) for c in range(3)]
                  if lows else [1.0, 1.0, 1.0])
    median_high = ([statistics.median(v[c] for v in highs) for c in range(3)]
                   if highs else None)   # None -> per-frame bootstrap fallback

    # per-frame wb fallback + print tune (the print tune renders the frame
    # several times — heavy and independent, so it runs frame-parallel)
    def stage_b2(fr):
        if fr["error"]:
            return fr
        fr["fallback_wb_low"] = fr.get("wb_low") is None
        fr["fallback_wb_high"] = fr.get("wb_high") is None
        if fr["fallback_wb_low"]:
            fr["wb_low"] = list(median_low)
        if fr["fallback_wb_high"]:
            # roll median of frames with a found white patch; if no frame has
            # one, the per-frame percentile bootstrap beats neutral 1.0
            fr["wb_high"] = list(median_high or fr.get("wb_high_boot")
                                 or [1.0, 1.0, 1.0])
        # restore the picker normalization (region/median results may drift it)
        fr["wb_low"] = _normalize_wb(fr["wb_low"], "shadows")
        fr["wb_high"] = _normalize_wb(fr["wb_high"], "highlights")
        fr["params"] = make_params(fr["dmin"], fr["d_max"], fr["offset"],
                                   fr["wb_low"], fr["wb_high"],
                                   fr["picked_min"], fr["picked_max"])
        try:
            _enc_f, lin = _get_loaded(fr)
            fr["params"], fr["print_tuning"] = tune_print_params(
                lin, fr["params"], fr["border"], fr["dmin"])
        except Exception as e:
            fr["print_tuning"] = {"tuned": False, "error": str(e)}
        fr["params_hex"] = nm.encode_negadoctor_params(fr["params"])
        return fr

    frames = _map_frames(stage_b2, frames, workers)

    # ---- Stage C (opt-in): vision-LLM per-scene nudge (alternate variant) ----
    if ai_tune:
        run_scene_tuning(frames, session_dir)

    # release the per-frame decode cache (large float buffers, not serializable)
    for fr in frames:
        fr.pop("_loaded", None)

    roll = {
        "winner_stem": winner_stem,
        "winner_rect": winner_rect,
        "winner_rgb": list(winner_rgb),
        "winner_factor": winner_factor,
        "median_wb_low": median_low,
        "median_wb_high": median_high,
        "vignette": vig,
        "vignette_info": vig_info,
    }
    return frames, roll


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def fmt3(v):
    return ",".join(f"{x:.6f}" for x in v)


def write_results(frames, roll, output_dir, ai_tune=False):
    """Write negadoctor_results.txt. When ai_tune, the applied `params=` blob is
    the AI variant (params_ai_hex) — that's what the AI Lua action writes into
    the XMPs — and a DETAIL_AI line records the scene + analytical comparison."""
    path = os.path.join(str(output_dir), RESULTS_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        if roll and roll.get("vignette"):
            v = roll["vignette"]
            info = roll.get("vignette_info") or {}
            f.write(
                f"VIGNETTE|strength={v['strength']:.4f}"
                f"|radius={v['radius']:.4f}|steepness={v['steepness']:.4f}"
                f"|corner_falloff={info.get('corner_falloff', 0):.4f}"
                f"|params={encode_lens_params(v)}\n")
        for fr in frames:
            if fr["error"]:
                f.write(f"ERR|{fr['stem']}|{fr['error']}\n")
                continue
            p = fr["params"]
            # AI mode applies the LLM-nudged variant; analytical otherwise
            applied_hex = (fr.get("params_ai_hex") if ai_tune
                           else None) or fr["params_hex"]
            f.write(f"OK|{fr['stem']}|params={applied_hex}\n")
            fb = int(fr["fallback_wb_low"]) + 2 * int(fr["fallback_wb_high"])
            base_rect = fr["base"]["rect"] if fr["base"] else [-1, -1, 0, 0]
            f.write(
                f"DETAIL|{fr['stem']}"
                f"|Dmin={fmt3(p['Dmin'])}|D_max={p['D_max']:.4f}"
                f"|offset={p['offset']:.4f}"
                f"|wb_low={fmt3(p['wb_low'])}|wb_high={fmt3(p['wb_high'])}"
                f"|black={p['black']:.4f}|exposure={p['exposure']:.4f}"
                f"|base_winner={roll['winner_stem'] if roll else '?'}"
                f"|base_rect={','.join(str(v) for v in base_rect)}"
                f"|fallback_wb={fb}\n"
            )
            if ai_tune and fr.get("params_ai"):
                pa = fr["params_ai"]
                sc = fr.get("scene") or {}
                tag = (f"{sc.get('scene')}/{sc.get('mood')}/{sc.get('warmth')}/"
                       f"{sc.get('contrast')}" if sc else "none")
                f.write(
                    f"DETAIL_AI|{fr['stem']}|scene={tag}"
                    f"|wb_low={fmt3(pa['wb_low'])}|wb_high={fmt3(pa['wb_high'])}"
                    f"|black={pa['black']:.4f}|exposure={pa['exposure']:.4f}"
                    f"|gamma={pa['gamma']:.4f}\n"
                )
    print(f"\nResults written to: {path}")
    return path


# Analysis-mask category codes (built live by the debug UI, 8-bit):
MASK_USED = 0      # inside the content crop: used for ALL analysis
MASK_BORDER = 2    # outside the content crop: disregarded entirely
# (codes 1/3 were per-pixel holder/base masks in earlier versions; the
# analysis is now strictly "everything inside the crop rectangle")


def build_analysis_mask(shape_hw, border):
    """Per-pixel map of the analysis area for the debug UI: the content-crop
    rectangle is used in full, everything outside is rejected."""
    h, w = shape_hw
    mask = np.full((h, w), MASK_USED, dtype=np.uint8)
    l, t, r, b = border
    if t: mask[:t, :] = MASK_BORDER
    if b: mask[h - b:, :] = MASK_BORDER
    if l: mask[:, :l] = MASK_BORDER
    if r: mask[:, w - r:] = MASK_BORDER
    return mask


def write_debug_sessions(frames, roll, output_dir, wall_time_s=None):
    """Write {stem}_debug_nega.json per frame.

    NO baked preview/mask images are emitted. The debug UI renders the
    inversion and the analysis-crop mask LIVE from the source negative
    (`negative_path`) plus these params/border, so what it shows is always
    the honest output of the current algorithm, never a stale cached file."""
    constants = {k: v for k, v in globals().items()
                 if k.isupper() and isinstance(v, (int, float, tuple))}
    constants = {k: (list(v) if isinstance(v, tuple) else v)
                 for k, v in constants.items()}
    out_dir = Path(output_dir)

    for fr in frames:
        if fr["error"]:
            continue
        is_winner = bool(roll and fr["stem"] == roll["winner_stem"])
        data = {
            "stem": fr["stem"],
            "negative_path": fr["path"],
            "width": fr["width"],
            "height": fr["height"],
            "exif": fr["exif"],
            "border": list(fr["border"]),
            "film_base": {
                "local": fr["base"],
                "applied_dmin": fr["params"]["Dmin"],
                "is_global_winner": is_winner,
                "global_winner_stem": roll["winner_stem"] if roll else None,
                "global_rect_on_winner": roll["winner_rect"] if roll else None,
            },
            "patches": {
                "shadows": dict(fr["shadow_patch"],
                                used_fallback=fr["fallback_wb_low"])
                           if fr.get("shadow_patch") else
                           {"used_fallback": True},
                "highlights": dict(fr["highlight_patch"],
                                   used_fallback=fr["fallback_wb_high"])
                              if fr.get("highlight_patch") else
                              {"used_fallback": True},
            },
            "picked_min": fr["picked_min"],
            "picked_max": fr["picked_max"],
            "wb_high_boot": fr.get("wb_high_boot"),
            "source_16bit_tiff": fr.get("source_16bit_tiff", False),
            "vignette": (roll or {}).get("vignette"),
            "vignette_info": (roll or {}).get("vignette_info"),
            "params": fr["params"],
            "params_hex": fr["params_hex"],
            # spec 03: alternate vision-LLM variant (present only with --ai-tune).
            # The UI switches its render base between these (key A).
            "params_analytical": fr["params"],
            "params_ai": fr.get("params_ai"),
            "params_ai_hex": fr.get("params_ai_hex"),
            "scene": fr.get("scene"),
            "scene_tuning": fr.get("scene_tuning"),
            "constants": constants,
            "processing_time_s": round(fr.get("time_a", 0) + fr.get("time_b", 0), 2),
        }
        if wall_time_s is not None:
            data["run_wall_time_s"] = round(float(wall_time_s), 2)
        with open(out_dir / f"{fr['stem']}_debug_nega.json", "w") as f:
            json.dump(data, f, indent=2)

    print(f"Debug sessions written to: {out_dir}")


def load_debug_nega_dir(directory):
    """Load all {stem}_debug_nega.json from a directory.

    Returns (images_list, constants_dict) — the shared debug-UI base contract.
    """
    directory = Path(directory)
    files = sorted(directory.glob("*_debug_nega.json"))
    if not files:
        return [], {}
    images, constants = [], {}
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        if not constants:
            constants = data.get("constants", {})
        images.append({k: v for k, v in data.items() if k != "constants"})
    return images, constants


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: auto_negadoctor.py [--debug-ui] [--ai-tune] "
              "<image> [image2 ...]")
        sys.exit(1)

    debug_ui = "--debug-ui" in sys.argv
    ai_tune = "--ai-tune" in sys.argv
    # Annotate-and-apply: open the debug UI and BLOCK until it closes, so the
    # caller can read the applied_results.txt the UI writes from the user's
    # corrections. Implies --debug-ui.
    annotate_apply = "--annotate-apply" in sys.argv
    if annotate_apply:
        debug_ui = True
    image_paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    for p in image_paths:
        if not os.path.exists(p):
            print(f"Error: File not found - {p}")
            sys.exit(1)

    output_dir = Path(image_paths[0]).parent
    exif_by_stem = parse_exif_params(os.path.join(str(output_dir), "exif_params.txt"))

    print("Auto Negadoctor - roll analysis")
    print("=" * 60)
    print(f"Processing {len(image_paths)} frame(s); exif entries: {len(exif_by_stem)}")

    def progress(done, total):
        print(f"PROGRESS|{done}|{total}", flush=True)

    wall_t0 = time.perf_counter()
    frames, roll = process_roll(image_paths, exif_by_stem, progress,
                                ai_tune=ai_tune, session_dir=output_dir)
    wall_time = time.perf_counter() - wall_t0

    ok = sum(1 for f in frames if not f["error"])
    err = len(frames) - ok
    print("=" * 60)
    print(f"Processing complete: {ok} succeeded, {err} failed; "
          f"wall time {wall_time:.1f}s")
    for fr in frames:
        if fr["error"]:
            print(f"  ERR {fr['stem']}: {fr['error']}")

    write_results(frames, roll, output_dir, ai_tune=ai_tune)

    if debug_ui and ok:
        write_debug_sessions(frames, roll, output_dir, wall_time_s=wall_time)
        debug_ui_script = Path(__file__).parent / "debug_ui.py"
        ui_cmd = [sys.executable, str(debug_ui_script), str(output_dir)]
        if annotate_apply:
            ui_cmd.append("--apply")
            print(f"Launching debug UI (apply mode), waiting for it to close: "
                  f"{debug_ui_script}", flush=True)
            subprocess.run(ui_cmd)
            print("APPLIED_RESULTS_READY", flush=True)
        else:
            print(f"Launching debug UI: {debug_ui_script}", flush=True)
            subprocess.Popen(ui_cmd)

    if err > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
