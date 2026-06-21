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
     preview with nega_model's exact forward model, derive wb_low from a
     region gray-world over a dark print-luma band, wb_high from a
     light-neutral patch (NEGATIVE-space color) with a bright-region
     fallback, then black + exposure
  4. frames without usable region/patch wb fall back to roll-median wb
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
import tuning as _tuning

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
    return max(1, min((os.cpu_count() or 1), 20, n))


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
    # NOTE: not a `with` block on purpose — the executor's __exit__ does
    # shutdown(wait=True), which on Ctrl-C blocks until every already-submitted
    # task finishes (the whole roll). On any interrupt/error we instead drop the
    # queued tasks (cancel_futures) and return immediately so KeyboardInterrupt
    # propagates promptly to the runner's top-level handler.
    ex = ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix=nm._POOL_PREFIX)
    try:
        futures = [ex.submit(fn, it) for it in items]
        for i, fut in enumerate(futures):
            results[i] = fut.result()
            if on_done:
                on_done()
    except BaseException:
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown(wait=True)
    return results

# --- resolution independence -------------------------------------------------
# Every size-dependent constant below is a RATIO of the frame dimension (a
# *_FRAC), turned into pixels at use sites with `int(round(w * FRAC))` (and a
# `max(1, …)` floor where a count of zero would be degenerate). There is no
# reference resolution: detection geometry scales directly with the export,
# whatever its width. Never introduce a raw pixel count tied to the export
# size — express it as a fraction of the dimension.

# --- TUNING CONSTANTS now live in tuning.py + presets/*.json -----------------
# The fittable taste/quality knobs (border trim, crop, film-base/patch search,
# percentiles, white balance, print tune, vignette) were moved OUT of this file:
# their VALUES are in presets/<name>.json and their full per-field RATIONALE is
# in tuning.py's FIELDS schema (JSON can't hold comments). They are loaded into
# the immutable DEFAULT_TUNING below and mirrored back into module globals (so
# the calibration registry's getattr/setattr utilities keep working), then read
# by the analysis functions via an explicit `cfg` argument. Adopting calibration
# results is now "drop in a new preset", not editing this file.
#
# Only NON-tunable correctness facts stay here as plain constants — a different
# value would be a BUG (corrupt sidecar / wrong clip detection), not a different
# look, so there is no objective to fit them against:
CLIP_SRGB_THR = 0.97               # stored-encoding value above this counts as clipped
PRINT_TUNE_SUBSAMPLE_FRAC = 0.003  # render every Nth pixel during tuning; stride
                                   # = frac of width (keeps sample count ~constant)

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

# Fallback manual-vignette params applied when the roll-wide estimate fails
# (envelope not vignette-like, or fitted strength below VIG_MIN_STRENGTH). The
# rig's lens + holder always vignettes some, so a missing fit means the roll was
# too dark/textured to recover the envelope, NOT that there is no falloff -
# applying nothing then leaves the corners over-bright. These are the
# centre-brightest reference roll's fitted values (2512-2601-1), the most
# representative known-good correction for this rig.
DEFAULT_VIGNETTE = {"strength": 0.525, "radius": 0.05, "steepness": 0.375}

RESULTS_FILENAME = "negadoctor_results.txt"


# ---------------------------------------------------------------------------
# Tuning configuration object (dependency injection of the fittable constants)
# ---------------------------------------------------------------------------
# The schema + per-field docs live in tuning.py; the VALUES live in a preset
# JSON (presets/<name>.json). DEFAULT_TUNING is the active preset, selected by
# --preset / $NEGA_PRESET (default "default"). The analysis functions read it
# from an explicit `cfg` argument; production / darktable calls omit cfg and get
# DEFAULT_TUNING. The calibration runner builds a per-trial Tuning with
# `cfg._replace(NAME=val)` and threads it down — nothing mutates shared module
# state, so independent trials (random_search) run in parallel THREADS.
Tuning = _tuning.Tuning
_TUNABLE_NAMES = tuple(_tuning.FIELDS)   # back-compat for the registry/tests
DEFAULT_PRESET = os.environ.get("NEGA_PRESET", "default")
DEFAULT_TUNING = _tuning.load(DEFAULT_PRESET)
# Mirror the loaded values back onto module globals so the calibration
# registry's getattr/setattr utilities (current/snapshot/restore/apply) and any
# legacy bare reference still resolve. The source holds NO hardcoded values now.
globals().update(DEFAULT_TUNING._asdict())


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

def detect_dark_border(lin, cfg=DEFAULT_TUNING):
    """Detect film-holder remains: near-black margins at the frame edges.

    lin: (H,W,3) linear RGB. Returns (left, top, right, bottom) margins in px
    (already padded), each capped at BORDER_MAX_FRAC of the dimension.
    """
    luma = lin.mean(axis=2)
    h, w = luma.shape
    border_pad = max(1, int(round(w * cfg.BORDER_PAD_FRAC)))
    col_mean = luma.mean(axis=0)
    row_mean = luma.mean(axis=1)

    def scan(profile, limit):
        n = 0
        for v in profile[:limit]:
            if v >= cfg.BORDER_DARK_THR:
                break
            n += 1
        return min(n + border_pad, limit) if n > 0 else 0

    max_w = int(w * cfg.BORDER_MAX_FRAC)
    max_h = int(h * cfg.BORDER_MAX_FRAC)
    return (
        scan(col_mean, max_w),
        scan(row_mean, max_h),
        scan(col_mean[::-1], max_w),
        scan(row_mean[::-1], max_h),
    )


def _crop_fields(lin, dmin):
    """The trial-INVARIANT heavy pixel math of detect_content_crop: full-res luma,
    the per-pixel density extremes (the costly log10 over every channel) and the
    row/column luma means + interior references. NONE of these depend on the
    fittable CROP_* / HOLDER_LUMA_THR constants — only on the buffer and the
    global base — so the calibration runner precomputes this ONCE per frame and
    re-runs only the cheap _crop_decide per trial. Returns a fields dict."""
    h, w = lin.shape[:2]
    luma = lin.mean(axis=2)
    dmin_arr = np.maximum(np.asarray(dmin, dtype=np.float32), nm.THR)
    log_ratio = np.log10(dmin_arr[None, None, :] / np.maximum(lin, nm.THR))
    density_min = np.min(log_ratio, axis=2)
    density_max = np.max(log_ratio, axis=2)
    # per-pixel HUE deviation from the film base. Unexposed rebate IS the base
    # color, so its chromaticity matches almost exactly (dev ~0.01); any EXPOSED
    # scene — even bright neutral sky/snow that reads "base-like" by density —
    # sits far off the strongly-orange base hue. Chromaticity = channel / sum,
    # L1 distance to the base chromaticity. (Drives the wide-rebate path.)
    base_hue = dmin_arr / float(dmin_arr.sum())
    px_hue = lin / np.maximum(lin.sum(axis=2, keepdims=True), 1e-6)
    hue_dev = np.abs(px_hue - base_hue[None, None, :]).sum(axis=2)
    # per-line RGB sums: let the wide-rebate path validate that a candidate
    # band's MEAN color matches the film base (the single decisive guard against
    # mistaking smooth orange daytime scene for rebate — see _crop_decide).
    col_rgb = lin.sum(axis=0)        # (w, 3)
    row_rgb = lin.sum(axis=1)        # (h, 3)
    # holder-edge shadow reference: interior median of each line-mean luma
    col_mean = luma.mean(axis=0)
    row_mean = luma.mean(axis=1)
    col_ref = float(np.median(col_mean[w // 4: 3 * w // 4]))
    row_ref = float(np.median(row_mean[h // 4: 3 * h // 4]))
    return {"h": h, "w": w, "luma": luma,
            "density_min": density_min, "density_max": density_max,
            "hue_dev": hue_dev, "base_hue": base_hue,
            "col_rgb": col_rgb, "row_rgb": row_rgb,
            "col_mean": col_mean, "row_mean": row_mean,
            "col_ref": col_ref, "row_ref": row_ref}


def detect_content_crop(lin, dmin, cfg=DEFAULT_TUNING):
    """Largest rectangular content area: trim each edge past the last scan
    line contaminated by holder-dark pixels or bright leak (lighter than the
    film base). Returns (left, top, right, bottom) margins in px."""
    return _crop_decide(_crop_fields(lin, dmin), cfg)


def _crop_decide(F, cfg=DEFAULT_TUNING):
    """The trial-VARIANT decision logic of detect_content_crop on precomputed
    _crop_fields `F`: applies the fittable CROP_* / HOLDER_LUMA_THR /
    BORDER_MAX_FRAC constants (from `cfg`, read at call time so calibration
    overrides take effect) to classify junk / base-like / shadow lines and trim
    each edge. Returns (left, top, right, bottom) px."""
    h, w = F["h"], F["w"]
    luma = F["luma"]
    # size-dependent constants are fractions of the frame width
    gap_tol = max(1, int(round(w * cfg.CROP_GAP_TOL_FRAC)))
    shadow_max = max(1, int(round(w * cfg.CROP_SHADOW_MAX_FRAC)))
    shadow_core = max(1, int(round(w * cfg.CROP_SHADOW_CORE_FRAC)))
    crop_pad = max(1, int(round(w * cfg.CROP_PAD_FRAC)))
    rebate_term = max(1, int(round(w * cfg.CROP_REBATE_TERM_FRAC)))
    rebate_max = max(1, int(round(w * cfg.CROP_REBATE_MAX_FRAC)))
    rebate_wide_w = max(1, int(round(w * cfg.CROP_REBATE_WIDE_FRAC)))
    rebate_wide_h = max(1, int(round(h * cfg.CROP_REBATE_WIDE_FRAC)))
    dark = luma < cfg.HOLDER_LUMA_THR
    junk = dark | (F["density_min"] < -cfg.CROP_LEAK_MARGIN_D)
    base_like = F["density_max"] < cfg.CROP_REBATE_MARGIN_D
    # confident-rebate (drives the WIDE trim path): a pixel that is genuine
    # unexposed film base. THREE independent signatures must all hold, because a
    # DELIBERATELY wide rebate band (the user scans extra rebate) cannot be told
    # from a diffuse bright-scene band by width alone:
    #   1. HUE matches the saturated-orange base almost exactly (chromaticity L1
    #      dev < CROP_REBATE_HUE_TOL ~0.03; in a negative even blue sky inverts to
    #      a DENSE off-orange, and bright neutral snow is far off the base hue);
    #   2. LOW density (max-channel < CROP_REBATE_WIDE_MAX_D) — the rebate is the
    #      lightest, UNexposed region; any exposed scene is denser. (Looser than
    #      base_like's CROP_REBATE_MARGIN_D so the vignette-darkened rebate near
    #      the holder still qualifies.)
    #   3. brighter than HOLDER_LUMA_THR (exclude the dark holder itself).
    # The columns must then be SOLID (CROP_REBATE_WIDE_LINE_FRAC, much higher than
    # the conservative rebate line frac): a true rebate fills ~all of a column;
    # warm scene only ever speckles it.
    rebate_hue = ((F["hue_dev"] < cfg.CROP_REBATE_HUE_TOL)
                  & (F["density_max"] < cfg.CROP_REBATE_WIDE_MAX_D)
                  & (luma > cfg.HOLDER_LUMA_THR))

    def line_flags(mask, axis, thr):
        return mask.mean(axis=axis) > thr

    # holder-edge shadow: line mean luma well below the interior reference
    col_shadow = F["col_mean"] < cfg.CROP_SHADOW_REL * max(F["col_ref"], nm.THR)
    row_shadow = F["row_mean"] < cfg.CROP_SHADOW_REL * max(F["row_ref"], nm.THR)

    col_hard0 = line_flags(junk, 0, cfg.CROP_JUNK_LINE_FRAC)
    row_hard0 = line_flags(junk, 1, cfg.CROP_JUNK_LINE_FRAC)
    col_rebate = line_flags(base_like, 0, cfg.CROP_REBATE_LINE_FRAC)
    row_rebate = line_flags(base_like, 1, cfg.CROP_REBATE_LINE_FRAC)
    col_rebate_wide = line_flags(rebate_hue, 0, cfg.CROP_REBATE_WIDE_LINE_FRAC)
    row_rebate_wide = line_flags(rebate_hue, 1, cfg.CROP_REBATE_WIDE_LINE_FRAC)

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

    base_hue = F["base_hue"]

    def band_hue_dev(line_rgb, last):
        """L1 chromaticity distance between the MEAN color of the edge band
        [0:last] and the film base. A true rebate band averages to the base
        color almost exactly (~0.002); smooth orange daytime SCENE that speckles
        the per-pixel mask still averages well off-base (~0.012-0.044). This is
        the decisive wide-rebate guard."""
        total = line_rgb[:last].sum(axis=0)
        s = float(total.sum())
        if s <= 0:
            return 1.0
        return float(np.abs(total / s - base_hue).sum())

    def trim(hard, rebate, rebate_wide, line_rgb, limit, wide_limit):
        t_hard = run_trim(hard, limit)
        t_all = run_trim(hard | rebate, limit)
        if t_all > t_hard:
            # a base-like band wider than a real rebate strip is scene content
            # (bright sky/snow reads base-like) - reject the extension
            if t_all - t_hard > rebate_max:
                t_all = t_hard
            else:
                # rebate-extended trim: only valid if the base-like band actually
                # terminates (unexposed scene continues, true rebate doesn't)
                end = t_all - crop_pad
                window = rebate[end:end + rebate_term]
                if t_all >= limit or len(window) == 0 or window.mean() > 0.5:
                    t_all = t_hard
        # WIDE confident-rebate tier: a hue-matched-to-base band may be
        # DELIBERATELY wide (the user scrolls film in the holder to scan extra
        # rebate past the frame edge), far past the normal rebate_max width cap.
        # Only the strict `rebate_wide` mask (base HUE + low density + SOLID
        # columns — see CROP_REBATE_WIDE_* in the caller) drives this. Advance
        # past any leading holder, then run the CONTIGUOUS rebate band (gap
        # tolerant) and stop where it ends — crucially NOT bridging through the
        # dark scene that follows (that scene is `hard`, not rebate). Accept only
        # when the band TERMINATES into non-rebate before wide_limit, so a
        # uniform-color edge can't swallow the frame, and only if it trims deeper
        # than the conservative result.
        i = 0
        while i < wide_limit and hard[i] and not rebate_wide[i]:
            i += 1                       # skip the leading holder run
        last_rebate = 0
        gap = 0
        while i < wide_limit:
            if rebate_wide[i]:
                last_rebate = i + 1
                gap = 0
            else:
                gap += 1
                if gap > gap_tol:
                    break
            i += 1
        if last_rebate and last_rebate < wide_limit:
            t_wide = min(last_rebate + crop_pad, wide_limit)
            window = rebate_wide[last_rebate:last_rebate + rebate_term]
            if (t_wide > t_all and len(window) and window.mean() <= 0.5
                    and band_hue_dev(line_rgb, last_rebate)
                        < cfg.CROP_REBATE_BAND_HUE_TOL):
                return t_wide
        return t_all

    max_w = int(w * cfg.BORDER_MAX_FRAC)
    max_h = int(h * cfg.BORDER_MAX_FRAC)

    def edge(hard0, shadow, rebate, rebate_wide, line_rgb, limit, wide_limit):
        return trim(hard0 | validated_shadow(shadow, limit), rebate,
                    rebate_wide, line_rgb, limit, wide_limit)

    col_rgb, row_rgb = F["col_rgb"], F["row_rgb"]
    return (
        edge(col_hard0, col_shadow, col_rebate, col_rebate_wide,
             col_rgb, max_w, rebate_wide_w),
        edge(row_hard0, row_shadow, row_rebate, row_rebate_wide,
             row_rgb, max_h, rebate_wide_h),
        edge(col_hard0[::-1], col_shadow[::-1], col_rebate[::-1],
             col_rebate_wide[::-1], col_rgb[::-1], max_w, rebate_wide_w),
        edge(row_hard0[::-1], row_shadow[::-1], row_rebate[::-1],
             row_rebate_wide[::-1], row_rgb[::-1], max_h, rebate_wide_h),
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

def _largest_true_rectangle(mask):
    """Largest axis-aligned all-True rectangle in a boolean 2D array.

    Classic histogram/stack method, O(H*W): row by row, `heights[c]` counts
    consecutive True cells ending at this row, then the maximal rectangle under
    that histogram is found with a monotonic stack. Returns (x, y, w, h, area)
    in mask coordinates; area 0 (and a degenerate rect) if the mask is all-False.
    The per-row stack runs in Python, so callers pass a COARSE mask (see
    find_film_base_candidate) to keep it cheap.
    """
    H, W = mask.shape
    heights = np.zeros(W, dtype=np.int32)
    best = (0, 0, 0, 0, 0)
    for row in range(H):
        heights = np.where(mask[row], heights + 1, 0)
        hl = heights.tolist()
        stack = []                       # indices of non-decreasing-height bars
        c = 0
        while c <= W:
            cur = hl[c] if c < W else 0
            if not stack or cur >= hl[stack[-1]]:
                stack.append(c)
                c += 1
            else:
                top = stack.pop()
                ht = hl[top]
                left = stack[-1] + 1 if stack else 0
                area = ht * (c - left)
                if area > best[4]:
                    best = (left, row - ht + 1, c - left, ht, area)
    return best


def find_film_base_candidate(lin, enc_f, win, cfg=DEFAULT_TUNING):
    """Find the LARGEST uniform-orange RECTANGLE = local film-base candidate.

    lin: (H,W,3) float32 linear RGB; enc_f: (H,W,3) float32 stored-encoding
    values in [0,1] (sRGB for JPEG, linear for TIFF — used for clipping
    detection); win: uniformity-window px.

    The search spans the FULL UNCROPPED frame — the film base is uniform ORANGE
    ANYWHERE (the unexposed rebate / the divider strip between two frames in the
    holder, both OUTSIDE the content crop), so it is deliberately NOT confined to
    any crop/border. We build a per-pixel "base-like" mask (orange + bright +
    locally uniform + not white-clipped — same guards that excluded the
    dark/neutral holder and stray backlight before) and take the LARGEST
    inscribed rectangle of it. A truly UNEXPOSED region forms a big solid
    rectangle; an incidental bright-orange scrap (the old fixed-window search
    would happily pick it as the "lightest orange window") forms only a tiny one.
    `area_frac` (rect area / frame area) is that confidence and is what
    choose_global_base weighs when picking the roll-wide base.

    Returns dict {rect:[x,y,w,h], rgb_linear:[r,g,b], score, clipped_r,
    area_frac} or None.
    """
    h, w = lin.shape[:2]

    mean_rgb = _box_mean(lin, win)                       # (H,W,3) local average
    luma = lin.mean(axis=2)
    mean_luma = _box_mean(luma, win)
    mean_luma_sq = _box_mean(luma * luma, win)
    var = np.maximum(mean_luma_sq - mean_luma ** 2, 0.0)
    rel_std = np.sqrt(var) / np.maximum(mean_luma, 1e-9)
    white_clipped = (enc_f >= CLIP_SRGB_THR).all(axis=2).astype(np.float32)
    white_clip_frac = _box_mean(white_clipped, win)
    any_clipped = (enc_f >= CLIP_SRGB_THR).any(axis=2).astype(np.float32)
    any_clip_frac = _box_mean(any_clipped, win)

    base_like = (
        (white_clip_frac < cfg.CLIP_FRAC_MAX)
        & (mean_luma >= cfg.BASE_MIN_LUMA)
        & (rel_std <= cfg.BASE_UNIFORMITY_MAX)
        & (mean_rgb[..., 0] >= cfg.BASE_MIN_RG_RATIO * mean_rgb[..., 1])  # orange
        & (mean_rgb[..., 1] * cfg.BASE_GB_TOL >= mean_rgb[..., 2])
    )
    if not base_like.any():
        return None

    # Find the maximal rectangle on a COARSE grid (the per-row stack is Python).
    # A coarse cell counts as base when >= BASE_MASK_SOLID_FRAC of its pixels are
    # base-like; the cell size is a fraction of width, so this is resolution-
    # invariant and the area is reported as a frame-area FRACTION.
    stride = max(int(round(w * cfg.BASE_SCAN_STRIDE_FRAC)), 1)
    cw, ch = max(w // stride, 1), max(h // stride, 1)
    coarse = cv2.resize(base_like.astype(np.float32), (cw, ch),
                        interpolation=cv2.INTER_AREA)
    solid = coarse >= cfg.BASE_MASK_SOLID_FRAC
    cx, cy, crw, crh, carea = _largest_true_rectangle(solid)
    if carea == 0:
        return None

    # Map the coarse rectangle back to full-res pixels.
    sx, sy = w / cw, h / ch
    px = int(round(cx * sx))
    py = int(round(cy * sy))
    pw = max(int(round(crw * sx)), 1)
    ph = max(int(round(crh * sy)), 1)
    px = max(0, min(px, w - 1))
    py = max(0, min(py, h - 1))
    pw = min(pw, w - px)
    ph = min(ph, h - py)

    # Sample the base color over the base-like pixels INSIDE the rectangle (the
    # coarse cells were only >= SOLID_FRAC base, so a few stragglers are excluded).
    region = lin[py:py + ph, px:px + pw]
    rmask = base_like[py:py + ph, px:px + pw]
    if rmask.any():
        rgb = region[rmask].mean(axis=0)
        mlum = float(luma[py:py + ph, px:px + pw][rmask].mean())
    else:
        rgb = region.reshape(-1, 3).mean(axis=0)
        mlum = float(luma[py:py + ph, px:px + pw].mean())

    return {
        "rect": [px, py, pw, ph],
        "rgb_linear": [float(v) for v in rgb],
        "score": mlum,
        "clipped_r": float(any_clip_frac[py:py + ph, px:px + pw].mean()) > cfg.CLIP_FRAC_MAX,
        "area_frac": float(carea) / float(cw * ch),
    }


def choose_global_base(frames, cfg=DEFAULT_TUNING):
    """Pick the roll-wide film base = the BRIGHTEST exposure-compensated base
    among frames that have a sizable uniform base region.

    Dmin is the film's MINIMUM density — the BRIGHTEST point on the negative
    (truly unexposed base). So BRIGHTNESS decides, never area: a larger but
    DARKER uniform region is more-exposed film, NOT the base, and forcing it onto
    the roll as Dmin wrecks every frame's inversion (the cross-frame "one frame
    goes dark blue when run with another" bug — a big dark region out-scored a
    small bright true base). Area is only a CONFIDENCE GATE (BASE_AREA_MIN_FRAC)
    so a tiny bright fleck can't win; if no frame clears it, all candidates are
    considered. Picking off non-base bright scraps is prevented UPSTREAM by the
    uniform-orange base_like mask (find_film_base_candidate), not here.

    frames: per-frame dicts with "stem", "base" (or None), "exposure_factor".
    Physical lightness = mean(rgb_linear)/factor (the same registered value under
    more light = a darker physical patch). area_frac = the base rectangle's frame
    fraction (see find_film_base_candidate).

    Returns (winner_stem or None, winner_rgb, winner_factor).
    """
    cands = [fr for fr in frames if fr.get("base")]
    if not cands:
        return None, None, None

    def phys(fr):
        return float(np.mean(fr["base"]["rgb_linear"])) / fr["exposure_factor"]

    qualifying = [fr for fr in cands
                  if float(fr["base"].get("area_frac", 0.0)) >= cfg.BASE_AREA_MIN_FRAC]
    winner = max(qualifying or cands, key=phys)
    return winner["stem"], winner["base"]["rgb_linear"], winner["exposure_factor"]


def dmin_for_frame(winner_rgb, winner_factor, frame_factor):
    """Transfer the global film base to a frame with a different DSLR
    exposure: linear sensor response scales values by the factor ratio."""
    scale = frame_factor / winner_factor
    return [nm.clamp(c * scale, nm.DMIN_RANGE) for c in winner_rgb]


# ---------------------------------------------------------------------------
# Frame percentiles + neutral patches
# ---------------------------------------------------------------------------

def frame_percentiles(lin, enc_f, border, dmin, cfg=DEFAULT_TUNING):
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
    picked_min = [float(np.percentile(vals[:, c], cfg.P_LOW)) for c in range(3)]
    picked_max = [float(np.percentile(vals[:, c], cfg.P_HIGH)) for c in range(3)]
    return picked_min, picked_max


def find_neutral_patch(preview, lin, enc_f, border, win, band_pct, kind,
                       dmin, d_max, base_rect=None, cfg=DEFAULT_TUNING):
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
    stride = max(win // cfg.PATCH_STRIDE_DIV, 1)
    l, t, r, b = border

    p_luma = preview.mean(axis=2).astype(np.float32)
    mean_pl = _box_mean(p_luma, win)
    mean_pl_sq = _box_mean(p_luma * p_luma, win)
    # luma floor: in near-black windows std/mean is pure grain noise
    rel_std = (np.sqrt(np.maximum(mean_pl_sq - mean_pl ** 2, 0.0))
               / np.maximum(mean_pl, cfg.PATCH_LUMA_FLOOR))

    mean_prgb = _box_mean(preview.astype(np.float32), win)
    # denominator floor: chroma ratios explode in near-black windows
    chroma = ((mean_prgb.max(axis=2) - mean_prgb.min(axis=2))
              / (mean_prgb.mean(axis=2) + cfg.PATCH_CHROMA_FLOOR))

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
        (mean_pl[grid] >= max(lo, cfg.SHADOW_MIN_LUMA))
        & (mean_pl[grid] <= hi)
        & (chroma[grid] <= cfg.PATCH_CHROMA_MAX)
        & (rel_std[grid] <= cfg.PATCH_UNIFORMITY_MAX)
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
    ok &= density >= cfg.MIN_PATCH_DENSITY
    if kind == "highlights":
        ok &= clip_frac[grid] <= cfg.HIGHLIGHT_CLIP_FRAC_MAX

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
                       kind, wb_low=None, cfg=DEFAULT_TUNING):
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
    if int(sel.sum()) < max(10, int(round(reg.shape[0] * cfg.WB_REGION_MIN_FRAC))):
        return None
    mean_neg = [float(v) for v in reg[sel].mean(axis=0)]
    if kind == "shadows":
        return nm.compute_wb_low(dmin, mean_neg, d_max)
    return nm.compute_wb_high(dmin, mean_neg, d_max, offset, wb_low or [1.0, 1.0, 1.0])


def make_params(dmin, d_max, offset, wb_low, wb_high, picked_min, picked_max,
                gamma=None, cfg=DEFAULT_TUNING):
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
        "gamma": cfg.PRINT_GAMMA if gamma is None else gamma,
        "soft_clip": nm.SOFT_CLIP_DEFAULT,
        "exposure": exposure,
    }


def render_preview_srgb(lin, params):
    out = nm.render_negadoctor(lin, params)
    return (nm.linear_to_srgb(out) * 255.0 + 0.5).astype(np.uint8)


def tune_print_params(lin, params, border, dmin, hi_ceil=None,
                      cfg=DEFAULT_TUNING):
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
    ceil = cfg.PRINT_HI_CEIL if hi_ceil is None else hi_ceil
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
    for it in range(cfg.PRINT_TUNE_ITERS):
        # exposure pins the high percentile to the ceiling (both directions)
        out = nm.render_negadoctor(content, p)
        hi = float(np.percentile(out.mean(axis=1), cfg.PRINT_HI_PCT))
        if hi > 1e-6:
            p["exposure"] = nm.clamp(
                p["exposure"] * (ceil / hi) ** (1.0 / gamma), ec)
        # when exposure is SATURATED (maxed and still below ceiling, or bottomed
        # and above it), black takes over to drive the highlight to the ceiling
        out = nm.render_negadoctor(content, p)
        hi = float(np.percentile(out.mean(axis=1), cfg.PRINT_HI_PCT))
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
        if clip <= cfg.PRINT_CLIP_BUDGET:
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

def _vig_decode_frame(path, exif_by_stem, cfg=DEFAULT_TUNING):
    """The trial-INVARIANT half of a vignette frame: decode the TIFF and derive
    the full-res exposure-normalised luma, the clip mask, and the holder border.
    None of these depend on any VIG_* constant, so a caller can compute this ONCE
    (vignette_frame_cache) and re-fold the envelope for many trials without
    touching disk. Returns a cache dict or None."""
    try:
        enc_f, lin = load_frame(path)
    except Exception:
        return None
    h, w = lin.shape[:2]
    stem = Path(path).stem
    exif = exif_by_stem.get(stem) or read_exif_fallback(path)
    factor, _missing = nm.exposure_factor(
        exif.get("exposure_s"), exif.get("iso"), exif.get("aperture"))
    l, t, r, b = detect_dark_border(lin, cfg)
    luma = lin.mean(axis=2) / max(factor, 1e-12)
    # 8/16-bit inputs: pre-flag clipped pixels (float exports have enc_f.max()==0)
    clip = (enc_f >= CLIP_SRGB_THR).any(axis=2) if enc_f.max() > 0 else None
    return {"shape": (h, w), "luma": luma, "clip": clip, "border": (l, t, r, b)}


def _vig_frame_envelope(c, cfg=DEFAULT_TUNING):
    """The trial-VARIANT half: from a decoded-frame cache, apply the inset +
    downsample (the VIG_INSET_FRAC / VIG_DOWNSAMPLE_FRAC constants, read at call
    time) and return ((h, w), downsampled valid-luma map). Bit-identical to the
    old inline _vig_frame."""
    if c is None:
        return None
    h, w = c["shape"]
    valid = np.zeros((h, w), dtype=bool)
    vig_inset = max(1, int(round(w * cfg.VIG_INSET_FRAC)))
    vig_ds = max(1, int(round(w * cfg.VIG_DOWNSAMPLE_FRAC)))
    l, t, r, b = c["border"]
    y0, y1 = t + vig_inset, h - b - vig_inset
    x0, x1 = l + vig_inset, w - r - vig_inset
    if y1 <= y0 or x1 <= x0:
        return None
    valid[y0:y1, x0:x1] = True
    if c["clip"] is not None:   # exclude clipped pixels (8/16-bit inputs)
        valid &= ~c["clip"]
    lum = np.where(valid, c["luma"], -np.inf)
    return (h, w), lum[::vig_ds, ::vig_ds]


def vignette_frame_cache(image_paths, exif_by_stem, cfg=DEFAULT_TUNING):
    """Decode each frame's trial-invariant buffers (full-res luma, clip mask,
    border) ONCE so vignette_envelope can be re-folded cheaply per trial without
    re-reading the TIFFs — the calibration runner's vignette FULL path uses this
    to avoid re-decoding every frame on every optimizer evaluation. Heavy (decode
    + border + luma) and memory-hungry (a float64 luma + bool clip per frame, in
    RAM); frame-parallel. Returns a list aligned with image_paths (None per
    undecodable frame). (cfg only feeds detect_dark_border's BORDER_* here.)"""
    paths = list(image_paths)
    return _map_frames(lambda p: _vig_decode_frame(p, exif_by_stem, cfg), paths,
                       _proc_workers(len(paths)))


def vignette_envelope(image_paths, exif_by_stem, frame_cache=None,
                      cfg=DEFAULT_TUNING):
    """The roll-wide radial vignette ENVELOPE — the heavy TIFF accumulation part
    of the estimate, split out so callers can capture it ONCE and then re-fit
    cheaply (the calibration runner does this for the VIG profile-fit constants).
    Returns {"r": [bin radii], "e": [envelope per bin], "used": n} on success, or
    {"r": None, "e": None, "used": n, "reason": str} when there's too little data.
    Affected by the VIG_* accumulation constants (downsample/inset/bins/percentile/
    min-samples).

    Pass `frame_cache` (from vignette_frame_cache) to skip the TIFF decode and
    re-fold from the cached per-frame buffers — same result, no disk I/O. When
    omitted, the frames are decoded here (frame-parallel)."""
    # The np.maximum envelope fold is serial but cheap and stays in input order
    # (first frame sets the shape). The heavy decode is either supplied via
    # frame_cache or done here in parallel.
    if frame_cache is None:
        frame_cache = vignette_frame_cache(image_paths, exif_by_stem, cfg)
    # Re-fold each cached frame to its downsampled valid-luma map in parallel
    # (independent per frame; order preserved so the serial np.maximum fold below
    # is unchanged). This is the per-trial work in the runner's vignette FULL path.
    per_frame = _map_frames(lambda c: _vig_frame_envelope(c, cfg),
                            list(frame_cache), _proc_workers(len(frame_cache)))
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
        return {"r": None, "e": None, "used": used, "reason": "too few frames"}

    h, w = shape
    hh, ww = acc.shape
    # same downsample stride used to build acc above (shape is shared across
    # all accumulated frames, so this matches the per-frame vig_ds)
    vig_ds = max(1, int(round(w * cfg.VIG_DOWNSAMPLE_FRAC)))
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float64) * vig_ds
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    rad = np.hypot(xx - cx, yy - cy) / np.hypot(cx, cy)
    finite = np.isfinite(acc)

    r_centers, envs = [], []
    for k in range(cfg.VIG_BINS):
        sel = finite & (rad >= k / cfg.VIG_BINS) & (rad < (k + 1) / cfg.VIG_BINS)
        if sel.sum() < cfg.VIG_MIN_BIN_SAMPLES:
            continue
        env = float(np.percentile(acc[sel], cfg.VIG_PROFILE_PCT))
        if env <= 0:
            continue
        r_centers.append((k + 0.5) / cfg.VIG_BINS)
        envs.append(env)
    if len(r_centers) < 6:
        return {"r": None, "e": None, "used": used,
                "reason": "too few radial bins"}

    return {"r": r_centers, "e": envs, "used": used, "reason": None}


def estimate_vignette(image_paths, exif_by_stem, frame_cache=None,
                      cfg=DEFAULT_TUNING):
    """Roll-wide vignette estimate (see the VIG_* constants block): capture the
    radial envelope from the TIFFs, then fit it.

    Pass `frame_cache` (from vignette_frame_cache) to re-use already-decoded
    frame buffers instead of re-reading the TIFFs. `cfg` injects the fittable
    VIG_* / BORDER_* constants (DEFAULT_TUNING = the module globals).

    Returns (params dict {strength, radius, steepness} or None,
             info dict {residual, corner_falloff, bins, frames})."""
    env = vignette_envelope(image_paths, exif_by_stem, frame_cache=frame_cache,
                            cfg=cfg)
    if env["r"] is None:
        return None, {"frames": env["used"], "reason": env["reason"]}
    return fit_vignette_profile(env["r"], env["e"], env["used"], cfg)


def fit_vignette_profile(r_centers, envs, used=None, cfg=DEFAULT_TUNING):
    """Turn a radial envelope profile into lens-module vignette params.

    Pure (no image data) so it is unit-testable on captured roll profiles:
    `r_centers` are bin-centre radii in [0,1], `envs` the per-bin envelope
    brightness. Returns (params dict or None, info dict) exactly as
    estimate_vignette does. Split out 2026-06-13 so the central-dip
    regression (roll 2511-12-1) can be tested without the 1.2GB TIFFs."""
    # reference = envelope of the CENTER bins (the global max pixel is an
    # outlier and would shift the whole curve)
    center_envs = [e for r, e in zip(r_centers, envs)
                   if r < cfg.VIG_PEAK_CENTER_FRAC]
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
        if t < cm * cfg.VIG_TAIL_CUT_REL:
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
    if params["strength"] < cfg.VIG_MIN_STRENGTH:
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


def process_roll_prefix(image_paths, exif_by_stem, progress=None,
                        cfg=DEFAULT_TUNING, _prog=None):
    """The trial-INVARIANT prefix of process_roll: stage 0 (roll-wide vignette)
    + stage A (per-frame film-base candidates, exif, dims, decoded buffers) +
    the GLOBAL film base. None of it depends on the wb / picker / patch / print
    constants, so a calibration fitting ONLY those can compute this ONCE and feed
    it back via `process_roll(prep=...)` instead of recomputing it (and
    reprinting the vignette / global-base lines) on every trial. It DOES depend
    on the vignette (`VIG_*` / `BORDER_*`) and film-base (`BASE_*`) constants, so
    a fit that touches those must rebuild it per trial.

    Returns a dict consumed by process_roll; `winner_stem` is None when no frame
    yielded a film-base candidate."""
    n = len(image_paths)
    paths = list(image_paths)
    workers = _proc_workers(n)
    prog = _prog if _prog is not None else _Progress(progress, 3 * n)

    # ---- Stage 0: roll-wide vignette estimation (uncorrected frames) ----
    vig, vig_info = estimate_vignette(image_paths, exif_by_stem, cfg=cfg)
    if vig:
        print(f"Vignette estimate: strength={vig['strength']:.3f} "
              f"radius={vig['radius']:.3f} steepness={vig['steepness']:.3f} "
              f"(corner falloff {vig_info['corner_falloff']:.1%}, "
              f"fit residual {vig_info['residual']}) — darktable lens module "
              f"manual vignette values")
    else:
        # No usable fit: fall back to the rig's default correction rather than
        # leaving the corners uncorrected (over-bright in the inversion).
        reason = vig_info.get("reason", "?") if vig_info else "?"
        vig = dict(DEFAULT_VIGNETTE)
        vig_info = dict(vig_info or {}, default=True, fit_reason=reason)
        print(f"Vignette: fit failed ({reason}); applying default "
              f"strength={vig['strength']:.3f} radius={vig['radius']:.3f} "
              f"steepness={vig['steepness']:.3f}")
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
            # Film base is searched on the FULL UNCROPPED frame (no border): it
            # is the lightest uniform orange patch anywhere — typically the
            # unexposed rebate / the divider strip between two frames, which sit
            # OUTSIDE the eventual content crop. The content crop (fr["border"])
            # is computed later, in stage B1.
            #
            # The uniformity window is sized by BASE_WIN_FRAC ALONE — NOT floored
            # by the neutral-patch MIN_WIN_FRAC. A window wider than the base
            # strip straddles its edges into the scene, so a thin lifeless strip
            # reads as textured and is missed (the bug behind "it picked dark
            # trees instead of my black divider strip"): the window must be able
            # to fit INSIDE a thin strip. The 3px floor only keeps the box filter
            # valid; it never binds at real export widths.
            win = max(int(round(w * cfg.BASE_WIN_FRAC)), 3)
            fr["base_win"] = win
            fr["base"] = find_film_base_candidate(lin, enc_f, win, cfg)
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
    winner_stem, winner_rgb, winner_factor = choose_global_base(good, cfg)
    winner_rect = (next(f["base"]["rect"] for f in good if f["stem"] == winner_stem)
                   if winner_stem is not None else None)
    if winner_stem is not None:
        print(f"Global film base: {winner_stem} rect={winner_rect} "
              f"rgb={['%.4f' % v for v in winner_rgb]} factor={winner_factor:.5g}")
    return {"vig": vig, "vig_info": vig_info, "frames": frames,
            "winner_stem": winner_stem, "winner_rgb": winner_rgb,
            "winner_factor": winner_factor, "winner_rect": winner_rect}


def process_roll(image_paths, exif_by_stem, progress=None, ai_tune=False,
                 session_dir=None, cfg=DEFAULT_TUNING, prep=None):
    """Run the full analysis. Returns list of per-frame result dicts.

    ai_tune: after the analytical params are computed, add a vision-LLM
    per-scene nudge as an ALTERNATE variant (params_ai / params_ai_hex on each
    frame; analytical params untouched). session_dir holds the LLM response
    cache. See run_scene_tuning().

    cfg: the Tuning object supplying every fittable constant (DEFAULT_TUNING =
    the module globals → byte-identical to before). The calibration runner passes
    a per-trial cfg so independent trials can run in parallel threads.

    prep: a precomputed `process_roll_prefix(...)` result. When given, stage 0
    (vignette) + stage A (film base) are NOT recomputed — only stage B1/B2 run,
    on shallow copies of the prefix frames (so the shared prefix is not mutated;
    the heavy decoded buffers are shared by reference and stay resident for the
    next trial). The calibration runner passes it to avoid re-estimating the
    trial-invariant prefix on every inversion trial."""
    n = len(image_paths)
    workers = _proc_workers(n)
    prog = _Progress(progress, 3 * n)

    # Stage 0 (vignette) + stage A (film base) + the global film base form the
    # trial-INVARIANT prefix (see process_roll_prefix). A fresh run builds it
    # inline; a calibration that fits only wb/picker/print constants computes it
    # ONCE and feeds it back via `prep=`.
    reuse = prep is not None
    if not reuse:
        prep = process_roll_prefix(image_paths, exif_by_stem, cfg=cfg, _prog=prog)

    vig, vig_info = prep["vig"], prep["vig_info"]
    winner_stem = prep["winner_stem"]
    winner_rgb, winner_factor = prep["winner_rgb"], prep["winner_factor"]
    winner_rect = prep["winner_rect"]
    # On REUSE, stage B1/B2 must not mutate the shared prefix dicts — work on
    # shallow copies (the `_loaded` buffers are shared by reference, read-only).
    frames = [dict(fr) for fr in prep["frames"]] if reuse else prep["frames"]

    if winner_stem is None:
        for fr in frames:
            if not fr["error"]:
                fr["error"] = "no film-base candidate found on any frame of the roll"
        for fr in frames:
            fr.pop("_loaded", None)
        return frames, None

    # ---- Stage B1: per-frame patches + wb (frame-parallel) ----
    def stage_b1(fr):
        if fr["error"]:
            return fr
        t0 = time.perf_counter()
        try:
            enc_f, lin = _get_loaded(fr)
            dmin = dmin_for_frame(winner_rgb, winner_factor, fr["exposure_factor"])
            # the content crop replaces the stage-A dark-border trim as THE
            # analysis area for everything from here on. Crop detection runs on
            # VIGNETTE-CORRECTED data (`lin`), CONSISTENT with the film base it
            # references (the base is sampled on corrected data too). On rolls
            # with strong vignette (e.g. 55% corner falloff) the UNcorrected edge
            # rebate is darkened far below the bright central base, so the rebate
            # detectors mistook it for image and left a per-frame rebate strip
            # uncropped (user-reported on foggy/heavily-vignetted frames). With
            # correction the edge rebate is restored to its true base brightness
            # and is detected. (Was uncorrected historically — that trimmed
            # 1-3px less on some edges; re-tune CROP_PAD_FRAC if needed.)
            fr["border"] = detect_content_crop(lin, dmin, cfg)
            fr["picked_min"], fr["picked_max"] = frame_percentiles(
                lin, enc_f, fr["border"], dmin, cfg)
            # D_max is FIXED at darktable's default (the user never picks it);
            # see DMAX_DEFAULT. picked_min/picked_max are still used for the
            # black/exposure pickers and the wb bootstrap.
            d_max = cfg.DMAX_DEFAULT
            offset = cfg.OFFSET_DEFAULT
            fr["dmin"], fr["d_max"], fr["offset"] = dmin, d_max, offset

            win = max(int(lin.shape[1] * cfg.PATCH_WIN_FRAC),
                      int(round(lin.shape[1] * cfg.MIN_WIN_FRAC)))
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
                                       list(cfg.WB_LOW_PRIOR),
                                       list(cfg.WB_HIGH_PRIOR),
                                       fr["picked_min"], fr["picked_max"],
                                       cfg=cfg)
            preview = nm.render_negadoctor(lin, params_prior)

            # wb_low: region gray-world over the shadow band, gently neutralized
            # (the single-window shadow search collapsed to a near-constant
            # over-warm value; the user's pick is a milder cast correction)
            wb_low_raw = estimate_region_wb(lin, fr["border"], preview, dmin,
                                            d_max, offset, cfg.WB_LOW_BAND_PCT,
                                            "shadows", cfg=cfg)
            wb_low = (desaturate_wb(wb_low_raw, cfg.WB_LOW_DESAT, "shadows")
                      if wb_low_raw else None)

            # wb_high: prefer the reliable light-neutral PATCH; fall back to the
            # bright-region cast estimate when no clean patch exists (e.g.
            # tungsten frames that have no neutral highlight window)
            highlight = find_neutral_patch(preview, lin, enc_f, fr["border"], win,
                                           cfg.HIGHLIGHT_BAND_PCT, "highlights",
                                           dmin, d_max, base_rect, cfg=cfg)
            if highlight:
                wb_high = nm.compute_wb_high(dmin, highlight["rgb_neg_linear"],
                                             d_max, offset, wb_low or neutral)
            else:
                wb_high_raw = estimate_region_wb(lin, fr["border"], preview,
                                                 dmin, d_max, offset,
                                                 cfg.WB_HIGH_BAND_PCT, "highlights",
                                                 wb_low or neutral, cfg=cfg)
                wb_high = (desaturate_wb(wb_high_raw, cfg.WB_HIGH_DESAT,
                                         "highlights")
                           if wb_high_raw else None)

            # wb_low is region-based (no shadow patch): only the highlight
            # patch survives, as the wb_high source.
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
                                   fr["picked_min"], fr["picked_max"], cfg=cfg)
        try:
            _enc_f, lin = _get_loaded(fr)
            fr["params"], fr["print_tuning"] = tune_print_params(
                lin, fr["params"], fr["border"], fr["dmin"], cfg=cfg)
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
        # calibration review payloads (live vs fitted), if the caller attached
        # them — the debug UI's R toggle swaps between them.
        if fr.get("review") is not None:
            data["review"] = fr["review"]
            data["review_kind"] = fr.get("review_kind")
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

    # --preset NAME|PATH (or --preset=...) overrides $NEGA_PRESET / the default.
    # Its value is consumed here so it isn't mistaken for an image path.
    preset = None
    args, skip = [], False
    for i, a in enumerate(sys.argv[1:]):
        if skip:
            skip = False
            continue
        if a == "--preset":
            preset = sys.argv[1:][i + 1] if i + 1 < len(sys.argv[1:]) else None
            skip = True
        elif a.startswith("--preset="):
            preset = a.split("=", 1)[1]
        else:
            args.append(a)
    global DEFAULT_TUNING
    if preset:
        try:
            DEFAULT_TUNING = _tuning.load(preset)
            globals().update(DEFAULT_TUNING._asdict())
            print(f"Tuning preset: {_tuning.preset_path(preset)}")
        except (OSError, ValueError) as e:
            print(f"Error: bad --preset {preset!r}: {e}")
            sys.exit(1)
    image_paths = [a for a in args if not a.startswith("--")]
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
                                ai_tune=ai_tune, session_dir=output_dir,
                                cfg=DEFAULT_TUNING)
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
