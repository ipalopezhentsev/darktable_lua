"""Auto-negadoctor: derive darktable negadoctor params for a roll of
DSLR-scanned color negative frames.

Called by auto_negadoctor.lua on linear-Rec2020 exports of the uninverted
negatives (negadoctor absent / disabled in the history). The export width is
set Lua-side and analysis is resolution-independent (size-dependent constants
are fractions of the frame dimension):

    python auto_negadoctor.py [--debug-ui] [--no-vis] img1.tif img2.tif ...

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
     for --debug-ui, per-frame {stem}_debug_nega.json sessions + inverted
     previews, then launch debug_ui.py

Results file format:
  OK|stem|params=<hex>
  DETAIL|stem|Dmin=..|D_max=..|...   (human/log only, Lua ignores)
  ERR|stem|message
"""

import json
import os
import statistics
import subprocess
import sys
import time
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
CROP_JUNK_LINE_FRAC = 0.04       # dark/leak fraction above which a line is trimmed
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
CROP_PAD_FRAC = 0.012            # safety pad past the last junk line, frac of width (user
                                 # crops ran ~3-5px inside the detected edges)
# Holder-edge SHADOW/penumbra: a darkened ramp (line mean luma well below the
# frame interior) too bright for HOLDER_LUMA_THR. Found via the user's
# full-roll crop annotations: every left-edge containment violation showed
# this ramp, with the luma plateau starting exactly at the user's margin.
CROP_SHADOW_REL = 0.70           # line mean luma below this fraction of the
                                 # interior reference = holder shadow
CROP_SHADOW_MAX_FRAC = 0.04      # a shadow run terminating within this depth
                                 # (frac of width) is real penumbra (full
                                 # credit); deeper = dense scene (bright sky is
                                 # DARK in negative space, drove +30px over-trims)
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

# --- print auto-tuning (spec: "normal brightness, then maximise specular
# --- highlights without heavy clipping") -------------------------------------
PRINT_TARGET_HI = 0.95           # rendered linear luma target for content P99.5
                                 # (user accepts specular clipping for punch;
                                 # soft_clip rolls the top off anyway)
PRINT_HI_PCT = 99.5
PRINT_MID_BAND = (0.15, 0.40)    # acceptable content median luma (linear).
                                 # Out-of-band frames get pulled only to the
                                 # NEAREST EDGE (minimal intervention): pulling
                                 # to a center target washed out genuinely dark
                                 # scenes (night/museum frames must stay moody)
PRINT_TUNE_ITERS = 4
PRINT_TUNE_SUBSAMPLE_FRAC = 0.003  # render every Nth pixel during tuning; stride
                                   # = frac of width (keeps sample count ~constant)

# --- print curve & wb taste (from user annotation feedback) -------------------
# Paper grade: user's corrections across sessions: 4.0 -> 5.25 -> 7.1 ->
# 6.15 (the last given on a run whose D_max was junk-skewed); manual rolls
# use 6.9. 6.5 sits in the middle of the signals. darktable's default 4.0
# prints far too flat. The print auto-tune adapts black/exposure around
# whatever gamma is set here.
PRINT_GAMMA = 6.5
# wb prior: the user's corrected wb on the reference frame ("warmer",
# DSC_0001 session 2026-06-11). Each frame's patch-derived wb is blended
# toward it: pure patch neutralization kills scene-light character (it
# over-neutralizes warm-lit indoor frames AND prints outdoor frames too
# cold - the user corrected in opposite directions on the two annotated
# frames, which a blend toward a common prior fixes in both).
WB_HIGH_PRIOR = (1.80, 1.35, 1.0)
WB_LOW_PRIOR = (1.0, 0.75, 0.70)
WB_PRIOR_WEIGHT = 0.5            # 0 = pure patch wb, 1 = prior only

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


def categorize_frame(lin, preview):
    """Future hook: scene categorization (e.g. a local LLM judging day/night,
    foliage-only frames, etc.) to steer wb fallback choices. v1: None."""
    return None


# ---------------------------------------------------------------------------
# Per-frame parameter assembly
# ---------------------------------------------------------------------------

def apply_wb_taste(wb_low, wb_high):
    """Blend patch-derived wb toward the user-taste roll prior and restore
    the picker normalization (max(wb_low)=1, min(wb_high)=1)."""
    w = WB_PRIOR_WEIGHT
    lo = [(1 - w) * wb_low[c] + w * WB_LOW_PRIOR[c] for c in range(3)]
    m = max(lo)
    lo = [nm.clamp(v / m, nm.WB_RANGE) for v in lo]
    hi = [(1 - w) * wb_high[c] + w * WB_HIGH_PRIOR[c] for c in range(3)]
    m = min(hi)
    hi = [nm.clamp(v / m, nm.WB_RANGE) for v in hi]
    return lo, hi


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


def tune_print_params(lin, params, border, dmin):
    """Adjust exposure (and black when needed) so the print hits the spec
    targets: specular highlights near PRINT_TARGET_HI without hard clipping,
    median content brightness inside PRINT_MID_BAND.

    Statistics run over EVERYTHING inside the content-crop border (no pixel
    masks). darktable's auto formulas anchor the curve endpoints in
    PRE-gamma space, which leaves full-range scans flat and dark; this
    closes the loop on the actual rendered output instead. Works on a
    subsampled frame. Returns (tuned params, tuning info dict).

    Exposure is the primary brightness lever (it drives content P99.5 to
    PRINT_TARGET_HI). On dense negatives — e.g. heavily snowed, high-key
    scenes whose every channel sits near the film base — darktable's auto
    black/exposure both pin at their extremes (black floor, exposure
    ceiling) and the print still lands far below target. When exposure is
    SATURATED at its ceiling and P99.5 is still short of target, black takes
    over as the brightness lever: with the steep print gamma a small black
    raise lifts the highlights strongly while barely touching the (already
    near-black) shadows. This reproduces the user's manual fix on snow frames
    (they raised black once exposure was maxed). The median-band pull only
    runs when exposure is NOT saturated, so it never claws that brightening
    back down (a high-key frame legitimately carries a high median).
    """
    l, t, r, b = border
    h, w = lin.shape[:2]
    s = max(1, int(round(w * PRINT_TUNE_SUBSAMPLE_FRAC)))
    region = lin[t:h - b:s, l:w - r:s].astype(np.float64)
    if region.size == 0:
        return params, {"tuned": False}

    exposure_ceil = nm.EXPOSURE_RANGE[1]
    p = dict(params)
    gamma = p["gamma"]
    content = region.reshape(-1, 3)
    info = {"tuned": True}
    for it in range(PRINT_TUNE_ITERS):
        out = nm.render_negadoctor(content, p)
        luma = out.mean(axis=1)
        hi = float(np.percentile(luma, PRINT_HI_PCT))
        med = float(np.median(luma))
        info["hi"], info["med"] = hi, med
        # exposure scales print_linear linearly -> output by ^gamma
        if hi > 1e-6:
            p["exposure"] = nm.clamp(
                p["exposure"] * (PRINT_TARGET_HI / hi) ** (1.0 / gamma),
                nm.EXPOSURE_RANGE)
        # black shifts print_linear additively by exposure*delta.
        saturated = p["exposure"] >= exposure_ceil - 1e-6
        if saturated and hi < PRINT_TARGET_HI:
            # exposure is maxed and the print is still too dark: finish
            # driving P99.5 toward the target with black instead
            delta_pre = (PRINT_TARGET_HI ** (1.0 / gamma)
                         - max(hi, 1e-6) ** (1.0 / gamma))
            p["black"] = nm.clamp(p["black"] + delta_pre / max(p["exposure"], 1e-6),
                                  nm.BLACK_RANGE)
            info["black_drives_hi"] = True
        elif not saturated and (med < PRINT_MID_BAND[0] or med > PRINT_MID_BAND[1]):
            # pull an out-of-band median only to the nearest band edge so
            # dark scenes keep their character
            target_mid = min(max(med, PRINT_MID_BAND[0]), PRINT_MID_BAND[1])
            delta_pre = (target_mid ** (1.0 / gamma)
                         - max(med, 1e-6) ** (1.0 / gamma))
            p["black"] = nm.clamp(p["black"] + delta_pre / max(p["exposure"], 1e-6),
                                  nm.BLACK_RANGE)
    return p, info


# ---------------------------------------------------------------------------
# Roll processing
# ---------------------------------------------------------------------------

def estimate_vignette(image_paths, exif_by_stem):
    """Roll-wide vignette estimate (see the VIG_* constants block).

    Returns (params dict {strength, radius, steepness} or None,
             info dict {residual, corner_falloff, bins, frames})."""
    acc = None
    shape = None
    used = 0
    for path in image_paths:
        try:
            enc_f, lin = load_frame(path)
        except Exception:
            continue
        h, w = lin.shape[:2]
        if shape is None:
            shape = (h, w)
        elif (h, w) != shape:
            continue
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
            continue
        valid[y0:y1, x0:x1] = True
        if enc_f.max() > 0:   # 8/16-bit inputs: exclude clipped pixels
            valid &= ~(enc_f >= CLIP_SRGB_THR).any(axis=2)
        lum = np.where(valid, luma, -np.inf)
        ds = lum[::vig_ds, ::vig_ds]
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

    # reference = envelope of the CENTER bins (the global max pixel is an
    # outlier and would shift the whole curve)
    center_envs = [e for r, e in zip(r_centers, envs) if r < 0.15]
    env_ref = max(center_envs) if center_envs else envs[0]
    targets_all = [max(env_ref / e, 1.0) for e in envs]

    # the true correction profile is monotone non-decreasing in r; a falling
    # tail means the base ceiling broke there (corner bright leak surviving
    # the inset) — cut it and let the model extrapolate the corner
    r_keep, t_keep = [], []
    cm = 0.0
    for r, t in zip(r_centers, targets_all):
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


def _vignette_field_cached(h, w, vig):
    key = (h, w, vig["strength"], vig["radius"], vig["steepness"])
    if key not in _VIG_FIELD_CACHE:
        _VIG_FIELD_CACHE[key] = nm.vignette_field(h, w, vig).astype(np.float32)
    return _VIG_FIELD_CACHE[key]


def load_frame(path, vignette=None):
    """Load an export -> (enc_f float32 RGB [0,1], lin float32 linear Rec2020).

    vignette: roll-wide correction params; when given, lin is multiplied by
    the darktable manual-vignette correction field (corners brightened) so
    ALL downstream analysis sees vignette-corrected data. enc_f (clipping
    detection on stored values) stays uncorrected.

    enc_f holds the stored-encoding values (for clipping detection):
      - .tif/.tiff: 16-bit linear Rec2020 from the Lua export — already the
        module-input approximation, no transform needed
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
        if rgb.dtype in (np.float32, np.float64):
            # 32-bit float TIFF: unclamped linear Rec2020 (the Lua export).
            # Nothing clips, so clip detection is disabled (enc_f = zeros).
            lin = np.maximum(rgb.astype(np.float32), 0.0)
            return finish(np.zeros_like(lin), lin)
        # 16-bit integer TIFF clamps the pipeline at 1.0 — values near the
        # ceiling are saturated, keep clip detection on the linear values
        scale = 65535.0 if rgb.dtype == np.uint16 else 255.0
        lin = (rgb.astype(np.float32) / scale)
        return finish(lin.copy(), lin)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    enc_f = rgb.astype(np.float32) / 255.0
    lin = nm.linearize_export(rgb).astype(np.float32)
    return finish(enc_f, lin)


def process_roll(image_paths, exif_by_stem, progress=None):
    """Run the full analysis. Returns list of per-frame result dicts."""
    n = len(image_paths)
    frames = []

    def report(done):
        if progress:
            progress(done, 3 * n)

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
    report(n)

    # ---- Stage A: per-frame film-base candidates + percentiles ----
    for i, path in enumerate(image_paths):
        stem = Path(path).stem
        fr = {"stem": stem, "path": str(path), "error": None, "base": None,
              "vignette": vig}
        t0 = time.perf_counter()
        try:
            enc_f, lin = load_frame(path, vig)
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
        frames.append(fr)
        report(n + i + 1)

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
        return frames, None
    winner_rect = next(f["base"]["rect"] for f in good if f["stem"] == winner_stem)
    print(f"Global film base: {winner_stem} rect={winner_rect} "
          f"rgb={['%.4f' % v for v in winner_rgb]} factor={winner_factor:.5g}")

    # ---- Stage B1: per-frame patches + wb ----
    for i, fr in enumerate(frames):
        if fr["error"]:
            report(2 * n + i + 1)
            continue
        t0 = time.perf_counter()
        try:
            enc_f, lin = load_frame(fr["path"], vig)
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
            d_max = nm.compute_dmax(dmin, fr["picked_min"])
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

            # ONE patch-search preview rendered with the taste-prior wb: it is
            # close to the final rendition, so "low chroma" really means
            # "gray in the print" (see find_neutral_patch docstring)
            params_prior = make_params(dmin, d_max, offset,
                                       list(WB_LOW_PRIOR), list(WB_HIGH_PRIOR),
                                       fr["picked_min"], fr["picked_max"])
            preview = nm.render_negadoctor(lin, params_prior)
            shadow = find_neutral_patch(preview, lin, enc_f, fr["border"], win,
                                        SHADOW_BAND_PCT, "shadows",
                                        dmin, d_max, base_rect)
            wb_low = (nm.compute_wb_low(dmin, shadow["rgb_neg_linear"], d_max)
                      if shadow else None)
            highlight = find_neutral_patch(preview, lin, enc_f, fr["border"], win,
                                           HIGHLIGHT_BAND_PCT, "highlights",
                                           dmin, d_max, base_rect)
            wb_high = (nm.compute_wb_high(dmin, highlight["rgb_neg_linear"], d_max,
                                          offset, wb_low or neutral)
                       if highlight else None)

            fr["shadow_patch"] = shadow
            fr["highlight_patch"] = highlight
            fr["wb_low"] = wb_low
            fr["wb_high"] = wb_high
        except Exception as e:
            fr["error"] = str(e)
        fr["time_b"] = time.perf_counter() - t0
        report(2 * n + i + 1)

    # ---- Roll-median fallback for frames without usable patches ----
    lows = [fr["wb_low"] for fr in frames if not fr["error"] and fr.get("wb_low")]
    highs = [fr["wb_high"] for fr in frames if not fr["error"] and fr.get("wb_high")]
    median_low = ([statistics.median(v[c] for v in lows) for c in range(3)]
                  if lows else [1.0, 1.0, 1.0])
    median_high = ([statistics.median(v[c] for v in highs) for c in range(3)]
                   if highs else None)   # None -> per-frame bootstrap fallback

    for fr in frames:
        if fr["error"]:
            continue
        fr["fallback_wb_low"] = fr.get("wb_low") is None
        fr["fallback_wb_high"] = fr.get("wb_high") is None
        if fr["fallback_wb_low"]:
            fr["wb_low"] = list(median_low)
        if fr["fallback_wb_high"]:
            # roll median of frames with a found white patch; if no frame has
            # one, the per-frame percentile bootstrap beats neutral 1.0
            fr["wb_high"] = list(median_high or fr.get("wb_high_boot")
                                 or [1.0, 1.0, 1.0])
        fr["wb_low"], fr["wb_high"] = apply_wb_taste(fr["wb_low"], fr["wb_high"])
        fr["params"] = make_params(fr["dmin"], fr["d_max"], fr["offset"],
                                   fr["wb_low"], fr["wb_high"],
                                   fr["picked_min"], fr["picked_max"])
        try:
            enc_f, lin = load_frame(fr["path"], fr.get("vignette"))
            fr["params"], fr["print_tuning"] = tune_print_params(
                lin, fr["params"], fr["border"], fr["dmin"])
        except Exception as e:
            fr["print_tuning"] = {"tuned": False, "error": str(e)}
        fr["params_hex"] = nm.encode_negadoctor_params(fr["params"])

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


def write_results(frames, roll, output_dir):
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
            f.write(f"OK|{fr['stem']}|params={fr['params_hex']}\n")
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
    print(f"\nResults written to: {path}")
    return path


# Analysis-mask category codes ({stem}_analysis_mask.png, 8-bit):
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


def write_debug_sessions(frames, roll, output_dir, wall_time_s=None, save_vis=True):
    """Write {stem}_debug_nega.json + {stem}_inverted.jpg + analysis mask
    (+overlay) per frame."""
    constants = {k: v for k, v in globals().items()
                 if k.isupper() and isinstance(v, (int, float, tuple))}
    constants = {k: (list(v) if isinstance(v, tuple) else v)
                 for k, v in constants.items()}
    out_dir = Path(output_dir)

    for fr in frames:
        if fr["error"]:
            continue
        enc_f, lin = load_frame(fr["path"], fr.get("vignette"))
        preview = render_preview_srgb(lin, fr["params"])
        inv_path = out_dir / f"{fr['stem']}_inverted.jpg"
        cv2.imwrite(str(inv_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

        # analysis-area mask (audit the content-crop detection in the UI)
        amask = build_analysis_mask(lin.shape[:2], fr["border"])
        amask_path = out_dir / f"{fr['stem']}_analysis_mask.png"
        cv2.imwrite(str(amask_path), amask)

        if save_vis:
            vis = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR).copy()
            # tint the rejected outside-crop area red (BGR)
            sel = amask == MASK_BORDER
            if sel.any():
                vis[sel] = (vis[sel] * 0.55
                            + np.array((60, 60, 255), dtype=np.float32) * 0.45
                            ).astype(np.uint8)
            def draw(rect, color, label):
                if not rect or rect[0] < 0:
                    return
                x, y, w, h = rect
                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                cv2.putText(vis, label, (x, max(y - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            if fr["base"]:
                draw(fr["base"]["rect"], (0, 165, 255), "base")
            if roll and fr["stem"] == roll["winner_stem"]:
                draw(roll["winner_rect"], (0, 215, 255), "GLOBAL")
            if fr.get("shadow_patch"):
                draw(fr["shadow_patch"]["rect"], (255, 255, 0), "shadows")
            if fr.get("highlight_patch"):
                draw(fr["highlight_patch"]["rect"], (255, 255, 255), "highlights")
            cv2.imwrite(str(out_dir / f"{fr['stem']}_nega_overlay.jpg"), vis,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])

        is_winner = bool(roll and fr["stem"] == roll["winner_stem"])
        data = {
            "stem": fr["stem"],
            "image_path": str(inv_path),
            "negative_path": fr["path"],
            "analysis_mask_path": str(amask_path),
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
        print("Usage: auto_negadoctor.py [--debug-ui] [--no-vis] <image> [image2 ...]")
        sys.exit(1)

    debug_ui = "--debug-ui" in sys.argv
    save_vis = "--no-vis" not in sys.argv
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
    frames, roll = process_roll(image_paths, exif_by_stem, progress)
    wall_time = time.perf_counter() - wall_t0

    ok = sum(1 for f in frames if not f["error"])
    err = len(frames) - ok
    print("=" * 60)
    print(f"Processing complete: {ok} succeeded, {err} failed; "
          f"wall time {wall_time:.1f}s")
    for fr in frames:
        if fr["error"]:
            print(f"  ERR {fr['stem']}: {fr['error']}")

    write_results(frames, roll, output_dir)

    if debug_ui and ok:
        write_debug_sessions(frames, roll, output_dir,
                             wall_time_s=wall_time, save_vis=save_vis)
        debug_ui_script = Path(__file__).parent / "debug_ui.py"
        print(f"Launching debug UI: {debug_ui_script}", flush=True)
        subprocess.Popen([sys.executable, str(debug_ui_script), str(output_dir)])

    if err > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
