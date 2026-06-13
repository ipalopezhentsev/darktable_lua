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
import struct

import numpy as np

# darktable negadoctor THRESHOLD (2^-32)
THR = 2.3283064365386963e-10

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
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055)


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

    log_density = np.log10(pix / dmin)          # = -log10(Dmin/pix)
    corrected = wb_high_c * log_density + offset_c
    print_linear = np.maximum(-(exposure * np.power(10.0, corrected) + black_c), 0.0)
    print_gamma = np.power(print_linear, gamma)

    # highlights roll-off above soft_clip
    soft_comp = 1.0 - soft_clip
    over = print_gamma > soft_clip
    rolled = soft_clip + (1.0 - np.exp(-(print_gamma - soft_clip) / soft_comp)) * soft_comp
    return np.where(over, rolled, np.minimum(print_gamma, 1.0))


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
