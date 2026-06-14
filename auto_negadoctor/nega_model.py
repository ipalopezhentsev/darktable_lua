"""Pure math for darktable's negadoctor module (no I/O).

Implements, as exact transcriptions of darktable src/iop/negadoctor.c
(modversion 2):
  - the per-pixel forward model (negative -> inverted print), vectorized
  - the auto-tuner formulas behind every picker in the module UI
  - the 76-byte params blob encoder/decoder used in XMP history entries

All "picked" colors below are MODULE INPUT values: linear pipeline RGB of
the *negative*, exactly what darktable's color pickers sample. We
approximate that space from an sRGB export via srgb gamma inversion plus
the sRGB->Rec2020 primaries matrix (darktable's default working profile).

Sign convention note: negadoctor computes log10(pix/Dmin) (in the C code a
log2 is taken and multiplied by -LOG2_to_LOG10 on the density ratio
Dmin/pix). With that sign, apply_auto_exposure / apply_auto_black are
exactly consistent with the pipeline: the lightest negative area maps to
print_linear/exposure = 0.1 and the densest to print_linear = 0.96.
"""

import math
import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# darktable negadoctor THRESHOLD (2^-32)
THR = 2.3283064365386963e-10

# The forward model and the sRGB OETF are per-PIXEL (every row independent) and
# spend their time in GIL-releasing numpy ufuncs (log10/power/exp), so splitting
# a large frame across threads gives a near-linear speedup with BIT-IDENTICAL
# output. Small arrays (e.g. thumbnails, single patches) run inline — the thread
# pool overhead would outweigh the work.
_PARALLEL_MAX_WORKERS = min(8, os.cpu_count() or 1)
_PARALLEL_MIN_ELEMS = 400_000
_POOL_PREFIX = "nega-render"
_pool = None
_pool_lock = threading.Lock()


def _get_pool():
    """Lazily-created persistent worker pool (avoids per-call thread spawn)."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ThreadPoolExecutor(max_workers=_PARALLEL_MAX_WORKERS,
                                           thread_name_prefix=_POOL_PREFIX)
    return _pool


def _parallel_rows(fn, arr):
    """Apply a row-independent array fn to `arr` over axis-0 chunks in parallel.

    Result is identical to ``fn(arr)`` (same dtype/values); only the work is
    split across threads when the array is large enough to be worth it. Calls
    made from WITHIN a pool worker run inline, so nested uses (e.g. the fused
    display path calling render + sRGB) never re-dispatch or deadlock."""
    n = arr.shape[0]
    if (_PARALLEL_MAX_WORKERS <= 1 or n < 2 or arr.size < _PARALLEL_MIN_ELEMS
            or threading.current_thread().name.startswith(_POOL_PREFIX)):
        return fn(arr)
    workers = min(_PARALLEL_MAX_WORKERS, n)
    chunks = np.array_split(arr, workers, axis=0)
    parts = list(_get_pool().map(fn, chunks))
    return np.concatenate(parts, axis=0)

# Introspection ranges (clamps when emitting params)
DMIN_RANGE = (1e-5, 1.5)
WB_RANGE = (0.25, 2.0)
DMAX_RANGE = (0.1, 6.0)
OFFSET_RANGE = (-1.0, 1.0)
BLACK_RANGE = (-0.5, 0.5)
GAMMA_RANGE = (1.0, 8.0)
SOFT_CLIP_RANGE = (1e-4, 1.0)
EXPOSURE_RANGE = (0.5, 2.0)

GAMMA_DEFAULT = 4.0
SOFT_CLIP_DEFAULT = 0.75

# Linear BT.709/sRGB -> BT.2020 primaries (both D65)
SRGB_TO_REC2020 = np.array(
    [
        [0.6274039, 0.3292830, 0.0433131],
        [0.0690973, 0.9195404, 0.0113623],
        [0.0163914, 0.0880133, 0.8955953],
    ],
    dtype=np.float64,
)
REC2020_TO_SRGB = np.linalg.inv(SRGB_TO_REC2020)


def clamp(v, lo_hi):
    lo, hi = lo_hi
    return min(max(v, lo), hi)


# ---------------------------------------------------------------------------
# Colorspace helpers
# ---------------------------------------------------------------------------

def srgb_to_linear(img):
    """Piecewise sRGB EOTF. Accepts uint8 or float [0,1]; returns float64."""
    x = np.asarray(img, dtype=np.float64)
    if img.dtype == np.uint8:
        x = x / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)

    def core(a):
        return np.where(a <= 0.0031308, a * 12.92,
                        1.055 * a ** (1.0 / 2.4) - 0.055)

    return _parallel_rows(core, x) if x.ndim >= 2 else core(x)


def linearize_export(img_u8_rgb):
    """sRGB uint8 export (H,W,3 RGB) -> linear Rec2020 float64 (H,W,3).

    Approximates the pipeline RGB that negadoctor sees (darktable's default
    working profile is linear Rec2020). Negative values after the matrix are
    clipped to 0.
    """
    lin = srgb_to_linear(img_u8_rgb)
    out = lin @ SRGB_TO_REC2020.T
    return np.clip(out, 0.0, None)


# ---------------------------------------------------------------------------
# Exposure compensation between frames
# ---------------------------------------------------------------------------

def exposure_factor(shutter_s, iso, aperture):
    """Relative amount of light reaching the sensor for a DSLR scan shot.

    Linear in shutter time and ISO, inverse-square in aperture f-number.
    Returns (factor, missing_exif_flag); factor 1.0 when EXIF is missing.
    """
    if not shutter_s or not iso or not aperture or shutter_s <= 0 or iso <= 0 or aperture <= 0:
        return 1.0, True
    return (shutter_s * iso) / (aperture * aperture), False


# ---------------------------------------------------------------------------
# Forward model (negative linear RGB -> inverted print, display-linear-ish)
# ---------------------------------------------------------------------------

def render_negadoctor(lin_rgb, params):
    """Vectorized transcription of negadoctor's process()/commit_params().

    lin_rgb: (...,3) linear negative values (module input space).
    params: dict with Dmin, wb_high, wb_low (3-seq each), D_max, offset,
            black, gamma, soft_clip, exposure.
    Returns (...,3) float64 output in [0, 1].
    """
    pix = np.maximum(np.asarray(lin_rgb, dtype=np.float64), THR)
    dmin = np.asarray(params["Dmin"][:3], dtype=np.float64)
    wb_high = np.asarray(params["wb_high"][:3], dtype=np.float64)
    wb_low = np.asarray(params["wb_low"][:3], dtype=np.float64)
    d_max = float(params["D_max"])
    offset = float(params["offset"])
    black = float(params["black"])
    gamma = float(params["gamma"])
    soft_clip = float(params["soft_clip"])
    exposure = float(params["exposure"])

    # commit_params derived values
    wb_high_c = wb_high / d_max
    offset_c = wb_high * offset * wb_low
    black_c = -exposure * (1.0 + black)
    soft_comp = 1.0 - soft_clip

    def core(p):
        log_density = np.log10(p / dmin)          # = -log10(Dmin/pix)
        corrected = wb_high_c * log_density + offset_c
        print_linear = np.maximum(
            -(exposure * np.power(10.0, corrected) + black_c), 0.0)
        print_gamma = np.power(print_linear, gamma)
        # highlights roll-off above soft_clip
        over = print_gamma > soft_clip
        rolled = soft_clip + (1.0 - np.exp(
            -(print_gamma - soft_clip) / soft_comp)) * soft_comp
        return np.where(over, rolled, np.minimum(print_gamma, 1.0))

    # large frames are split across threads (bit-identical, just faster)
    return _parallel_rows(core, pix) if pix.ndim >= 2 else core(pix)


def render_negadoctor_srgb8(lin_rgb, params):
    """Full display path: linear negative -> 8-bit sRGB image (H,W,3 uint8).

    Exactly equivalent to
        (linear_to_srgb(render_negadoctor(lin)) * 255 + 0.5).astype(uint8)
    but FUSES the render + sRGB encode + quantize into ONE parallel pass (a
    single split/concat over a persistent pool), which is the cheapest way to
    feed the live debug-UI preview."""
    arr = np.asarray(lin_rgb)

    def core(chunk):
        srgb = linear_to_srgb(render_negadoctor(chunk, params))
        return (srgb * 255.0 + 0.5).astype(np.uint8)

    return _parallel_rows(core, arr) if arr.ndim >= 2 else core(arr)


# ---------------------------------------------------------------------------
# Histogram distance (OFFLINE tuning loss: match a rendered output to a target)
#
# The user's ground truth is the inverted PICTURE, not the params that made it
# (exposure<->wb are interdependent, so the params are not a unique encoding).
# The picture is characterized by its histogram, so the loss for choosing the
# algorithm's fixed constants is the distance between the histogram of the
# algorithm's render and that of the user's hand-tuned (GT) render. See
# specs/04_tune_algo_params_via_histograms.md. Pure math; no I/O.
# ---------------------------------------------------------------------------

# Rec.601 luma weights — "brightness" of an sRGB-encoded display pixel.
_LUMA_W = np.array([0.299, 0.587, 0.114], dtype=np.float64)


def _as_rgb_rows(img):
    """(N,3) or (H,W,3) uint8/float -> (N,3) float64 rows (first 3 channels)."""
    a = np.asarray(img)
    return a.reshape(-1, a.shape[-1])[:, :3].astype(np.float64)


def _norm_hist(values, bins):
    """Histogram of 8-bit-scale values over [0,256) normalized to sum 1."""
    h, _ = np.histogram(values, bins=bins, range=(0.0, 256.0))
    total = h.sum()
    if total <= 0:
        return np.full(bins, 1.0 / bins)
    return h.astype(np.float64) / total


def cumhist_l1(h_a, h_b):
    """1D Wasserstein-1 (EMD) between two normalized histograms of equal length,
    expressed as a fraction of the value range: L1 of the cumulative
    distributions divided by the bin count. Reads as 'how far, in [0,1] tone
    units, the mass must move to turn distribution A into B'. Robust to bin
    jitter and needs no per-pixel registration (A and B may have different
    sample counts)."""
    ca = np.cumsum(np.asarray(h_a, dtype=np.float64))
    cb = np.cumsum(np.asarray(h_b, dtype=np.float64))
    return float(np.abs(ca - cb).sum() / len(ca))


def rgb_histograms(srgb_u8, bins=64):
    """Per-channel normalized histograms (list of 3) of an 8-bit sRGB image plus
    the bright-clip mass (fraction of pixels whose max channel sits in the
    display's top ~1%). Accepts (N,3) or (H,W,3)."""
    rows = _as_rgb_rows(srgb_u8)
    hists = [_norm_hist(rows[:, c], bins) for c in range(3)]
    top = float(np.mean(rows.max(axis=1) >= 254.0)) if rows.size else 0.0
    return hists, top


def histogram_distance(srgb_a, srgb_b, bins=64):
    """Decomposed EMD between two rendered 8-bit sRGB outputs (A = the
    algorithm's render, B = the ground-truth render). All terms are in [0,1]
    tone units; smaller is closer.

      total        mean per-channel EMD — the headline loss.
      per_channel  [dR, dG, dB] per-channel EMD.
      luma         EMD of the Rec.601 luma histogram — brightness + contrast +
                   clip distribution (user goals 1 "bright midtones" + 2 "no
                   clipping").
      color        mean |dmean_channel - dmean_overall| — chroma-cast divergence
                   BEYOND the common brightness shift, i.e. shadow/highlight
                   color balance (user goal 3). Pure color: a uniform exposure
                   change leaves it ~0.
      luma_signed  mean(luma B) - mean(luma A) in [0,1]; +ve => B brighter, so
                   the algorithm (A) renders too DARK vs the target.
      top_a, top_b bright-clip mass of A and B (goal 2; A should not exceed B).
    """
    ha, top_a = rgb_histograms(srgb_a, bins)
    hb, top_b = rgb_histograms(srgb_b, bins)
    per_channel = [cumhist_l1(ha[c], hb[c]) for c in range(3)]

    ra = _as_rgb_rows(srgb_a)
    rb = _as_rgb_rows(srgb_b)
    la, lb = ra @ _LUMA_W, rb @ _LUMA_W
    luma = cumhist_l1(_norm_hist(la, bins), _norm_hist(lb, bins))
    luma_signed = (float(lb.mean() - la.mean()) / 255.0
                   if la.size and lb.size else 0.0)

    # per-channel signed mean shift (B - A), and the common (overall) shift;
    # the residual after removing the common shift is the pure chroma cast.
    d_chan = (rb.mean(axis=0) - ra.mean(axis=0)) / 255.0
    color = float(np.mean(np.abs(d_chan - d_chan.mean())))

    return {
        "total": float(np.mean(per_channel)),
        "per_channel": per_channel,
        "luma": luma,
        "color": color,
        "luma_signed": luma_signed,
        "top_a": top_a,
        "top_b": top_b,
    }


# ---------------------------------------------------------------------------
# Auto-tuner formulas (verbatim from the picker callbacks)
# ---------------------------------------------------------------------------

def compute_dmax(dmin, picked_min):
    """apply_auto_Dmax: picked_min = per-channel darkest values of the frame."""
    rgb = [math.log10(dmin[c] / max(picked_min[c], THR)) for c in range(3)]
    return clamp(max(rgb), DMAX_RANGE)


def compute_offset(dmin, picked_max, d_max):
    """apply_auto_offset: picked_max = per-channel lightest values of the frame."""
    rgb = [math.log10(dmin[c] / max(picked_max[c], THR)) / d_max for c in range(3)]
    return clamp(min(rgb), OFFSET_RANGE)


def compute_wb_low(dmin, picked_dark_gray, d_max):
    """apply_auto_WB_low: picked over a dark gray patch (shadows color cast)."""
    r = [math.log10(dmin[c] / max(picked_dark_gray[c], THR)) / d_max for c in range(3)]
    r_min = min(r)
    return [clamp(r_min / r[c], WB_RANGE) for c in range(3)]


def compute_wb_high(dmin, picked_white, d_max, offset, wb_low):
    """apply_auto_WB_high: picked over a white/light gray patch."""
    r = [
        abs(-1.0 / (offset * wb_low[c]
                    - math.log10(dmin[c] / max(picked_white[c], THR)) / d_max))
        for c in range(3)
    ]
    r_min = min(r)
    return [clamp(r[c] / r_min, WB_RANGE) for c in range(3)]


def compute_black(dmin, picked_max, d_max, wb_high, wb_low, offset):
    """apply_auto_black: lightest area of the negative prints at 0.1 (pre-gamma)."""
    out = []
    for c in range(3):
        v = -math.log10(dmin[c] / max(picked_max[c], THR))
        v *= wb_high[c] / d_max
        v += wb_low[c] * offset * wb_high[c]
        out.append(0.1 - (1.0 - 10.0 ** v))
    return clamp(max(out), BLACK_RANGE)


def compute_exposure(dmin, picked_min, d_max, wb_high, wb_low, offset, black):
    """apply_auto_exposure: densest area of the negative prints at 0.96 (pre-gamma)."""
    out = []
    for c in range(3):
        v = -math.log10(dmin[c] / max(picked_min[c], THR))
        v *= wb_high[c] / d_max
        v += wb_low[c] * offset
        out.append(0.96 / (1.0 - 10.0 ** v + black))
    return clamp(min(out), EXPOSURE_RANGE)


# ---------------------------------------------------------------------------
# Color-wheel <-> white-balance mapping (debug-UI shadows/highlights wheels)
# ---------------------------------------------------------------------------
# A wb vector (wb_low or wb_high) is a normalized 3-channel multiplier with
# only 2 degrees of freedom (wb_low has max==1, wb_high has min==1). A color
# wheel encodes exactly 2 DOF: hue (angle) + chroma (radius). We map between
# them in LOG space via a zero-sum chroma vector projected onto three unit
# axes 120 deg apart (R at 90 deg, G at 210 deg, B at 330 deg) — the standard
# vectorscope embedding, which is a faithful bijection between zero-sum
# 3-vectors and the 2D plane. Neutral wb [1,1,1] -> wheel center (radius 0).

# Log-space chroma magnitude that maps to wheel radius 1.0. At the WB_RANGE
# extreme a single channel reaches ln(2.0) ~ 0.693 away from neutral while the
# other two sit near ln(0.79); the resulting zero-sum vector has magnitude
# ~0.85, so this rim value keeps the usable wb gamut comfortably inside the
# disk.
WHEEL_MAX_CHROMA = 0.9

# Unit axes for R, G, B at 90, 210, 330 degrees (counter-clockwise).
_WHEEL_AXES = np.array(
    [[math.cos(math.radians(a)), math.sin(math.radians(a))]
     for a in (90.0, 210.0, 330.0)],
    dtype=np.float64,
)


def wb_to_wheel(wb):
    """Normalized wb (3-seq) -> (angle_rad, radius in [0,1]).

    Inverse of wheel_to_wb up to the wb normalization/clamp. Neutral -> r=0.
    """
    log_wb = np.log(np.clip(np.asarray(wb[:3], dtype=np.float64),
                            WB_RANGE[0], WB_RANGE[1]))
    d = log_wb - log_wb.mean()                 # zero-sum chroma
    p = _WHEEL_AXES.T @ d                       # 2D point (x, y)
    radius = float(np.hypot(p[0], p[1]) / WHEEL_MAX_CHROMA)
    return math.atan2(p[1], p[0]), min(radius, 1.0)


def wheel_to_wb(angle_rad, radius, kind):
    """(angle_rad, radius in [0,1], kind) -> normalized wb (list of 3).

    kind: "low" normalizes so max(wb)==1, "high" so min(wb)==1. Each channel
    is clamped to WB_RANGE.
    """
    mag = max(0.0, min(float(radius), 1.0)) * WHEEL_MAX_CHROMA
    p = np.array([math.cos(angle_rad), math.sin(angle_rad)],
                 dtype=np.float64) * mag
    # invert the 3->2 zero-sum projection: d_c = (2/3) * p . axis_c
    d = (2.0 / 3.0) * (_WHEEL_AXES @ p)
    wb = np.exp(d)
    if kind == "low":
        wb = wb / wb.max()
    elif kind == "high":
        wb = wb / wb.min()
    else:
        raise ValueError(f"kind must be 'low' or 'high', got {kind!r}")
    return [clamp(float(v), WB_RANGE) for v in wb]


def default_params():
    """darktable's color-film preset values."""
    return {
        "film_stock": 1,
        "Dmin": [1.13, 0.49, 0.27],
        "wb_high": [1.0, 1.0, 1.0],
        "wb_low": [1.0, 1.0, 1.0],
        "D_max": 1.6,
        "offset": -0.05,
        "black": 0.0755,
        "gamma": GAMMA_DEFAULT,
        "soft_clip": SOFT_CLIP_DEFAULT,
        "exposure": 0.9245,
    }


# ---------------------------------------------------------------------------
# Vignette model — exact transcription of darktable's lens-module "manual
# vignette correction" (src/iop/lens.cc): the pixel is MULTIPLIED by
# M(r) = 1 + 2*v_strength * s(r),  s(r) = v + mul*tanh(b*(1-r)),
# v = v_steepness, b = 1 + v_radius*10, mul = -v_steepness/tanh(b),
# with r = distance from center normalized by the half-diagonal (corner=1).
# s(0) = 0 (center untouched), s(1) = v_steepness (full corner boost).
# ---------------------------------------------------------------------------

def vignette_spline(r, strength, radius, steepness):
    """Correction multiplier M(r) for scalar or array r in [0,1]."""
    r = np.asarray(r, dtype=np.float64)
    b = 1.0 + radius * 10.0
    mul = -steepness / math.tanh(b)
    s = steepness + mul * np.tanh(b * (1.0 - r))
    return 1.0 + 2.0 * strength * s


def vignette_field(h, w, params):
    """(H,W) correction-multiplier field for an image of the given size.
    params: dict with strength/radius/steepness (or None/zero strength ->
    all-ones)."""
    if not params or params.get("strength", 0.0) <= 0.0:
        return np.ones((h, w), dtype=np.float64)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r = np.hypot(xx - cx, yy - cy) / math.hypot(cx, cy)
    return vignette_spline(np.minimum(r, 1.0), params["strength"],
                           params["radius"], params["steepness"])


def fit_vignette(radii, targets):
    """Fit (strength, radius, steepness) of M(r) to target multipliers.

    radii/targets: 1D arrays (target = required correction = center/observed
    envelope value). Coarse grid + two local refinements (no scipy in env).
    Returns (params dict, rms residual)."""
    radii = np.asarray(radii, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    def loss(p):
        return float(np.sqrt(np.mean(
            (vignette_spline(radii, *p) - targets) ** 2)))

    best = (0.0, 0.5, 0.5)
    best_l = loss(best)
    grids = [np.linspace(0.0, 1.0, 11)] * 3
    for _round in range(3):
        for s in grids[0]:
            for rad in grids[1]:
                for st in grids[2]:
                    l = loss((s, rad, st))
                    if l < best_l:
                        best, best_l = (s, rad, st), l
        # refine around the current best
        grids = [np.clip(np.linspace(v - sp, v + sp, 9), 0.0, 1.0)
                 for v, sp in zip(best, (0.1, 0.1, 0.1))]
    params = {"strength": round(best[0], 4), "radius": round(best[1], 4),
              "steepness": round(best[2], 4)}
    return params, best_l


# ---------------------------------------------------------------------------
# XMP params blob (modversion 2, 76 bytes LE, plain hex in the sidecar)
# ---------------------------------------------------------------------------

def encode_negadoctor_params(p):
    """params dict -> 152-char hex string for darktable:params."""
    return struct.pack(
        "<i18f",
        int(p.get("film_stock", 1)),
        p["Dmin"][0], p["Dmin"][1], p["Dmin"][2], 0.0,
        p["wb_high"][0], p["wb_high"][1], p["wb_high"][2], 1.0,
        p["wb_low"][0], p["wb_low"][1], p["wb_low"][2], 1.0,
        p["D_max"], p["offset"], p["black"],
        p["gamma"], p["soft_clip"], p["exposure"],
    ).hex()


def decode_negadoctor_params(hex_str):
    """152-char hex string -> params dict (pads dropped)."""
    vals = struct.unpack("<i18f", bytes.fromhex(hex_str))
    return {
        "film_stock": vals[0],
        "Dmin": list(vals[1:4]),
        "wb_high": list(vals[5:8]),
        "wb_low": list(vals[9:12]),
        "D_max": vals[13],
        "offset": vals[14],
        "black": vals[15],
        "gamma": vals[16],
        "soft_clip": vals[17],
        "exposure": vals[18],
    }
