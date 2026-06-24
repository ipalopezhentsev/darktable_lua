"""
Dust spot detection for DSLR-scanned film negatives.

Detects bright dust particles on inverted negatives and generates
XMP-ready binary data for darktable's retouch module.

Usage:
    python detect_dust.py [--debug-ui] <image1.jpg> [image2.jpg ...]

Output:
    dust_results.txt in the same directory as the first input image.
"""

import sys
import os
import io
import math
import struct
import zlib
import base64
import json
import time
import random
from contextlib import redirect_stdout
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from skimage.filters import sato

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tuning



# ---------------------------------------------------------------------------
# Tuning constants (dependency injection of the fittable detection knobs)
# ---------------------------------------------------------------------------
# The fittable detection thresholds (dust dots, stroke/scratch/streak/radon,
# sensor dust) were moved OUT of this file: their VALUES live in
# presets/<name>.json and their per-field docs in tuning.py's FIELDS schema
# (JSON can't hold comments). They load into the immutable DEFAULT_TUNING and
# are mirrored back onto module globals (so any bare reference + the
# calibration registry's getattr/setattr utilities resolve), then read by the
# detection functions via an explicit `cfg` argument. Adopting calibration
# results is now 'drop in a new preset', not editing this file. Only NON-tunable
# correctness facts (brush/mask byte layout, the blendop template, mask
# versions, the ML feature-name lists) stay below as plain constants.
Tuning = tuning.Tuning
_TUNABLE_NAMES = tuple(tuning.FIELDS)   # back-compat for the registry/tests
DEFAULT_PRESET = os.environ.get("RETOUCH_PRESET", "default")
DEFAULT_TUNING = tuning.load(DEFAULT_PRESET)
globals().update(DEFAULT_TUNING._asdict())
# Must match train_dust_model.py.
SPOT_FEATURE_NAMES = [
    "contrast",
    "contrast_ratio",           # contrast / threshold
    "log_area",                 # log1p(area)
    "radius_px",
    "texture",
    "context_texture",
    "contrast_texture_ratio",   # contrast / max(texture, 0.1)
    "excess_sat",
    "spot_sat",
]

# Extended feature names for recovery-candidate classification (all fields from
# _compute_candidate_features; used only when ML recovery is requested).
FEATURE_NAMES = [
    "contrast",
    "contrast_ratio",          # contrast / threshold
    "log_area",                # log1p(area)
    "radius_px",
    "texture",
    "context_texture",
    "contrast_texture_ratio",  # contrast / max(texture, 0.1)
    "excess_sat",
    "spot_sat",
    "local_bg_frac",           # local_bg_brightness / bright_ref
    "is_dark_bg",
    "grad_ratio",
    "aspect_ratio",
    "compactness",
    "surround_ratio",
    "dust_votes",
]

# ---------------------------------------------------------------------------
# Darktable binary format constants
# ---------------------------------------------------------------------------
BRUSH_DENSITY = 1.0
BRUSH_HARDNESS = 1.0
BRUSH_STATE_NORMAL = 1
BRUSH_DELTA = 0.00001          # offset between 2 brush points (forms a "dot")
BRUSH_CTRL_OFFSET = 0.000003   # bezier control handle offset from corner
MIN_BRUSH_PX         = 5.0   # minimum brush radius in pixels (darktable effectiveness floor)
                              # brush_radius_px = max(MIN_BRUSH_PX, enc_r * ENC_RADIUS_SCALE)
                              # where enc_r is the min enclosing circle radius of the spot contour
HEAL_SOURCE_OFFSET_X = 0.01   # heal source offset from spot center (right)
HEAL_SOURCE_OFFSET_Y = 0.01   # positive = down (darktable default is right+down)

MASK_TYPE_BRUSH_CLONE = 72     # DT_MASKS_CLONE | DT_MASKS_BRUSH (8 | 64)
MASK_TYPE_GROUP_CLONE = 12     # DT_MASKS_GROUP | DT_MASKS_CLONE (4 | 8)
MASK_VERSION = 6               # DEVELOP_MASKS_VERSION

RETOUCH_ALGO_HEAL = 2
RETOUCH_MOD_VERSION = 3
MAX_FORMS = 300                # darktable's maximum form slots

# Known-good blendop_params template from a real darktable XMP (420 bytes uncompressed)
# Only the mask_id field at offset 24 needs to be replaced per-image
BLENDOP_TEMPLATE_ENCODED = "gz08eJxjYGBgYAFiCQYYOOEEIjd6dmXCRFgZMAEjFjEGhgZ7CB6pfOygYtaVAyCMi48L/AcCEA0Ak0kpjg=="

# ===================================================================
# Dust detection
# ===================================================================

def detect_dust_spots(image_path, collect_rejects=False, cfg=None):
    """Detect bright dust spots in an image.

    Returns (spots, rejected_candidates, error_msg).
    spots: list of dicts with keys: cx, cy (pixel coords), radius_px, area, contrast, texture, excess_sat, spot_sat.
    rejected_candidates: list of structured reject dicts (only populated when collect_rejects=True).
    error_msg: string on failure, None on success.
    """
    cfg = DEFAULT_TUNING if cfg is None else cfg
    LOCAL_BG_KERNEL = cfg.LOCAL_BG_KERNEL
    NOISE_THRESHOLD_MULTIPLIER = cfg.NOISE_THRESHOLD_MULTIPLIER
    MIN_ABSOLUTE_THRESHOLD = cfg.MIN_ABSOLUTE_THRESHOLD
    REJECT_LOG_CONTRAST_MIN = cfg.REJECT_LOG_CONTRAST_MIN
    MIN_SPOT_AREA = cfg.MIN_SPOT_AREA
    MAX_SPOT_AREA = cfg.MAX_SPOT_AREA
    MIN_ASPECT_RATIO = cfg.MIN_ASPECT_RATIO
    MIN_COMPACTNESS = cfg.MIN_COMPACTNESS
    MIN_SOLIDITY = cfg.MIN_SOLIDITY
    MIN_CIRCULARITY = cfg.MIN_CIRCULARITY
    DOT_MIN_CIRCLE_FILL = cfg.DOT_MIN_CIRCLE_FILL
    DOT_IRREGULAR_RADIUS_FRAC = cfg.DOT_IRREGULAR_RADIUS_FRAC
    SHAPE_CHECK_MIN_AREA = cfg.SHAPE_CHECK_MIN_AREA
    TEXTURE_KERNEL = cfg.TEXTURE_KERNEL
    MAX_LOCAL_TEXTURE_SMALL = cfg.MAX_LOCAL_TEXTURE_SMALL
    MAX_LOCAL_TEXTURE_LARGE = cfg.MAX_LOCAL_TEXTURE_LARGE
    MAX_DARK_BG_TEXTURE = cfg.MAX_DARK_BG_TEXTURE
    MIN_CONTRAST_TEXTURE_RATIO = cfg.MIN_CONTRAST_TEXTURE_RATIO
    MAX_BG_GRADIENT_RATIO = cfg.MAX_BG_GRADIENT_RATIO
    MAX_CONTEXT_TEXTURE = cfg.MAX_CONTEXT_TEXTURE
    LARGE_SPOT_AREA_THRESHOLD = cfg.LARGE_SPOT_AREA_THRESHOLD
    LARGE_SPOT_MIN_CONTRAST = cfg.LARGE_SPOT_MIN_CONTRAST
    MAX_EXCESS_SATURATION = cfg.MAX_EXCESS_SATURATION
    MAX_SPOT_SATURATION = cfg.MAX_SPOT_SATURATION
    EMULSION_EXCESS_SAT_THRESHOLD = cfg.EMULSION_EXCESS_SAT_THRESHOLD
    SOFT_CONTEXT_VOTE_THRESHOLD = cfg.SOFT_CONTEXT_VOTE_THRESHOLD
    SOFT_TEXTURE_VOTE_THRESHOLD = cfg.SOFT_TEXTURE_VOTE_THRESHOLD
    SOFT_RATIO_VOTE_THRESHOLD = cfg.SOFT_RATIO_VOTE_THRESHOLD
    MIN_DUST_VOTES = cfg.MIN_DUST_VOTES
    MIN_BRIGHTNESS_FRAC_SMALL = cfg.MIN_BRIGHTNESS_FRAC_SMALL
    MIN_BRIGHTNESS_FRAC_LARGE = cfg.MIN_BRIGHTNESS_FRAC_LARGE
    MIN_LOCAL_BG_FRACTION = cfg.MIN_LOCAL_BG_FRACTION
    MIN_SURROUND_BG_RATIO = cfg.MIN_SURROUND_BG_RATIO
    ISOLATION_RADIUS = cfg.ISOLATION_RADIUS
    MAX_NEARBY_ACCEPTED = cfg.MAX_NEARBY_ACCEPTED
    ENC_RADIUS_SCALE = cfg.ENC_RADIUS_SCALE
    DETECT_STROKES = cfg.DETECT_STROKES
    STROKE_MIN_LENGTH_FRAC = cfg.STROKE_MIN_LENGTH_FRAC
    STROKE_MIN_WIDTH_PX = cfg.STROKE_MIN_WIDTH_PX
    STROKE_MAX_FILL_RATIO = cfg.STROKE_MAX_FILL_RATIO
    STROKE_PREFER_RATIO = cfg.STROKE_PREFER_RATIO
    STROKE_RIDGE_MIN_CONTRAST = cfg.STROKE_RIDGE_MIN_CONTRAST
    STROKE_CLIP_LEVEL = cfg.STROKE_CLIP_LEVEL
    HEAL_SPLIT_BUSY = cfg.HEAL_SPLIT_BUSY
    STROKE_FIELD_RADIUS_FRAC = cfg.STROKE_FIELD_RADIUS_FRAC
    STROKE_FIELD_INNER_PX = cfg.STROKE_FIELD_INNER_PX
    STROKE_FIELD_MAX_LINE_CANDS = cfg.STROKE_FIELD_MAX_LINE_CANDS
    STROKE_FIELD_CAND_MIN_AREA = cfg.STROKE_FIELD_CAND_MIN_AREA
    STROKE_FIELD_CAND_MIN_DIM = cfg.STROKE_FIELD_CAND_MIN_DIM
    STROKE_FIELD_CAND_MAX_FILL = cfg.STROKE_FIELD_CAND_MAX_FILL
    STROKE_FIELD_CAND_MIN_ELONG = cfg.STROKE_FIELD_CAND_MIN_ELONG
    STROKE_FIELD_NBR_RADIUS_FRAC = cfg.STROKE_FIELD_NBR_RADIUS_FRAC
    STROKE_FIELD_MAX_NEIGHBORS = cfg.STROKE_FIELD_MAX_NEIGHBORS
    STREAK_DETECT = cfg.STREAK_DETECT
    STREAK_RADON_DETECT = cfg.STREAK_RADON_DETECT
    img = cv2.imread(str(image_path))
    if img is None:
        return None, [], f"Failed to load image: {image_path}", None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]  # 0-255, used to reject colored features
    height, width = gray.shape

    # Local background: large Gaussian blur smooths out dust but preserves
    # gradual brightness variations (vignetting, subject brightness)
    local_bg = cv2.GaussianBlur(gray, (LOCAL_BG_KERNEL, LOCAL_BG_KERNEL), 0)

    # Difference: positive values = brighter than background (dust on inverted negatives)
    diff = gray.astype(np.float32) - local_bg.astype(np.float32)

    # Background gradient: where the blurred background has a strong gradient,
    # diff values are unreliable (edge halo artifacts from Gaussian smoothing)
    bg_f = local_bg.astype(np.float32)
    bg_grad_x = cv2.Sobel(bg_f, cv2.CV_32F, 1, 0, ksize=3)
    bg_grad_y = cv2.Sobel(bg_f, cv2.CV_32F, 0, 1, ksize=3)
    bg_gradient = np.sqrt(bg_grad_x ** 2 + bg_grad_y ** 2)

    # Local texture map: measures how "busy" each region is.
    # Dust sits on smooth areas (low std); image features are in textured areas (high std).
    gray_f = gray.astype(np.float32)
    k = TEXTURE_KERNEL
    local_mean = cv2.blur(gray_f, (k, k))
    local_sq_mean = cv2.blur(gray_f ** 2, (k, k))
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))

    # Threshold based on image noise level (MAD = robust estimator, ignores dust outliers)
    noise_std = float(np.median(np.abs(diff)) * 1.4826)
    threshold = max(MIN_ABSOLUTE_THRESHOLD, noise_std * NOISE_THRESHOLD_MULTIPLIER)
    print(f"  Noise std: {noise_std:.1f}, threshold: {threshold:.1f} "
          f"(min_abs={MIN_ABSOLUTE_THRESHOLD}, mult={NOISE_THRESHOLD_MULTIPLIER})")
    print(f"  Local texture range: {local_std.min():.1f} - {local_std.max():.1f}, "
          f"median={np.median(local_std):.1f}")

    # Minimum brightness: dust is near film-base brightness (very bright on inverted negatives).
    # Larger dust must be whiter — small spots can appear gray due to subpixel mixing,
    # but any sizable dust particle is always near film-base brightness.
    bright_ref = float(np.percentile(gray, 95))
    min_local_bg = bright_ref * MIN_LOCAL_BG_FRACTION
    print(f"  Brightness ref (95th pct): {bright_ref:.0f}, "
          f"min brightness: {bright_ref * MIN_BRIGHTNESS_FRAC_SMALL:.0f} (small) / "
          f"{bright_ref * MIN_BRIGHTNESS_FRAC_LARGE:.0f} (large), "
          f"min local bg: {min_local_bg:.0f}")

    binary = (diff > threshold).astype(np.uint8)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    print(f"  Connected components above threshold: {num_labels - 1}")

    spots = []
    rejected_candidates = []  # structured rejects, only populated when collect_rejects=True
    rejected = {"too_small": 0, "too_large": 0, "shape": 0, "contrast": 0,
                "large_dim": 0, "dim": 0, "dark_bg": 0, "embedded": 0, "edge": 0,
                "texture": 0, "ratio": 0, "context": 0,
                "votes": 0, "color": 0, "sat_high": 0,
                "isolation": 0}
    debug_rejects = []  # track rejected candidates with high contrast for diagnostics

    def log_reject(cx, cy, area, contrast, reason, detail):
        if collect_rejects and contrast >= REJECT_LOG_CONTRAST_MIN:
            rejected_candidates.append({
                "cx": float(cx), "cy": float(cy),
                "area": int(area), "contrast": float(contrast),
                "reason": reason, "detail": detail,
            })
    for label_id in range(1, num_labels):  # skip background (0)
        area = stats[label_id, cv2.CC_STAT_AREA]
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]

        # Size filter
        if area < MIN_SPOT_AREA:
            rejected["too_small"] += 1
            continue
        if area > MAX_SPOT_AREA:
            rejected["too_large"] += 1
            continue

        # Circularity: bounding box aspect ratio
        aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0
        if aspect < MIN_ASPECT_RATIO:
            rejected["shape"] += 1
            # Log high-contrast rejects for diagnostics
            cx_tmp, cy_tmp = centroids[label_id]
            bx_tmp = stats[label_id, cv2.CC_STAT_LEFT]
            by_tmp = stats[label_id, cv2.CC_STAT_TOP]
            cm_tmp = labels[by_tmp:by_tmp+h, bx_tmp:bx_tmp+w] == label_id
            cd_tmp = diff[by_tmp:by_tmp+h, bx_tmp:bx_tmp+w]
            c_tmp = float(np.max(cd_tmp[cm_tmp])) if cm_tmp.any() else 0
            if c_tmp >= 40 and area >= 8:
                debug_rejects.append(f"    REJECTED({cx_tmp:.0f},{cy_tmp:.0f}) area={area} contrast={c_tmp:.0f} by=aspect({aspect:.2f}<{MIN_ASPECT_RATIO})")
            log_reject(cx_tmp, cy_tmp, area, c_tmp, "shape", f"aspect={aspect:.2f}<{MIN_ASPECT_RATIO}")
            continue

        # Compactness: how much of the bounding box is filled.
        # Only for small spots — larger ones use convex hull solidity instead.
        bbox_area = w * h
        compactness = area / bbox_area if bbox_area > 0 else 0
        if area < SHAPE_CHECK_MIN_AREA and compactness < MIN_COMPACTNESS:
            rejected["shape"] += 1
            cx_tmp, cy_tmp = centroids[label_id]
            bx_tmp = stats[label_id, cv2.CC_STAT_LEFT]
            by_tmp = stats[label_id, cv2.CC_STAT_TOP]
            cm_tmp = labels[by_tmp:by_tmp+h, bx_tmp:bx_tmp+w] == label_id
            cd_tmp = diff[by_tmp:by_tmp+h, bx_tmp:bx_tmp+w]
            c_tmp = float(np.max(cd_tmp[cm_tmp])) if cm_tmp.any() else 0
            if c_tmp >= 40 and area >= 8:
                debug_rejects.append(f"    REJECTED({cx_tmp:.0f},{cy_tmp:.0f}) area={area} contrast={c_tmp:.0f} by=compactness({compactness:.2f}<{MIN_COMPACTNESS})")
            log_reject(cx_tmp, cy_tmp, area, c_tmp, "shape", f"compactness={compactness:.2f}<{MIN_COMPACTNESS}")
            continue

        # For contrast, use the max diff within the component (not just centroid)
        cx, cy = centroids[label_id]
        bx = stats[label_id, cv2.CC_STAT_LEFT]
        by = stats[label_id, cv2.CC_STAT_TOP]
        component_mask = labels[by:by+h, bx:bx+w] == label_id
        component_diff = diff[by:by+h, bx:bx+w]
        contrast = float(np.max(component_diff[component_mask]))

        # Shape checks for larger spots: reject non-blob shapes (letters, symbols, arrows)
        contours = []
        if area >= SHAPE_CHECK_MIN_AREA:
            mask_u8 = component_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                hull = cv2.convexHull(contours[0])
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 1.0
                if solidity < MIN_SOLIDITY:
                    rejected["shape"] += 1
                    if contrast >= 40:
                        debug_rejects.append(f"    REJECTED({cx:.0f},{cy:.0f}) area={area} contrast={contrast:.0f} by=solidity({solidity:.2f}<{MIN_SOLIDITY})")
                    log_reject(cx, cy, area, contrast, "shape", f"solidity={solidity:.2f}<{MIN_SOLIDITY}")
                    continue
                perimeter = cv2.arcLength(contours[0], closed=True)
                if perimeter > 0:
                    circularity = 4 * math.pi * area / (perimeter * perimeter)
                    if circularity < MIN_CIRCULARITY:
                        rejected["shape"] += 1
                        if contrast >= 40:
                            debug_rejects.append(f"    REJECTED({cx:.0f},{cy:.0f}) area={area} contrast={contrast:.0f} by=circularity({circularity:.2f}<{MIN_CIRCULARITY})")
                        log_reject(cx, cy, area, contrast, "shape", f"circularity={circularity:.2f}<{MIN_CIRCULARITY}")
                        continue

        # Compute min enclosing circle radius — physical extent of the spot, handles irregular shapes.
        # For large spots: reuse already-computed contour; for small spots: compute fresh.
        if area >= SHAPE_CHECK_MIN_AREA and contours:
            _, enc_r = cv2.minEnclosingCircle(contours[0])
            enc_r = float(enc_r)
        else:
            mask_u8_enc = component_mask.astype(np.uint8) * 255
            enc_contours, _ = cv2.findContours(mask_u8_enc, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
            if enc_contours:
                _, enc_r = cv2.minEnclosingCircle(enc_contours[0])
                enc_r = float(enc_r)
            else:
                enc_r = math.sqrt(area / math.pi)  # fallback, should not happen

        # Reject large, elongated/irregular blobs (e.g. rope/cord fragments) that a circular
        # brush would grossly over-cover — these are not circular dust dots.
        circle_fill = area / (math.pi * enc_r * enc_r) if enc_r > 0 else 1.0
        if (enc_r > DOT_IRREGULAR_RADIUS_FRAC * min(height, width)
                and circle_fill < DOT_MIN_CIRCLE_FILL):
            rejected["shape"] += 1
            if contrast >= 40:
                debug_rejects.append(f"    REJECTED({cx:.0f},{cy:.0f}) area={area} contrast={contrast:.0f} by=irregular(circle_fill={circle_fill:.2f} enc_r={enc_r:.0f})")
            log_reject(cx, cy, area, contrast, "shape", f"circle_fill={circle_fill:.2f}<{DOT_MIN_CIRCLE_FILL} enc_r={enc_r:.0f}")
            continue

        if contrast < threshold * 0.8:
            rejected["contrast"] += 1
            log_reject(cx, cy, area, contrast, "contrast", f"contrast={contrast:.1f}<{threshold*0.8:.1f}")
            continue

        # Large dim spots: pale foggy blobs are not dust. Real dust is small and bright.
        # Large spots (area > threshold) that aren't sharply brighter than surroundings
        # are almost certainly image features or film artifacts, not dust particles.
        if area > LARGE_SPOT_AREA_THRESHOLD and contrast < LARGE_SPOT_MIN_CONTRAST:
            rejected["large_dim"] += 1
            if contrast >= REJECT_LOG_CONTRAST_MIN:
                debug_rejects.append(f"    REJECTED({cx:.0f},{cy:.0f}) area={area} contrast={contrast:.0f} by=large_dim(area>{LARGE_SPOT_AREA_THRESHOLD} contrast<{LARGE_SPOT_MIN_CONTRAST})")
            log_reject(cx, cy, area, contrast, "large_dim", f"area={area}>{LARGE_SPOT_AREA_THRESHOLD} contrast={contrast:.1f}<{LARGE_SPOT_MIN_CONTRAST}")
            continue

        # Compute local background brightness and integer centroid (used by multiple filters)
        icx, icy = int(round(cx)), int(round(cy))
        icx = max(0, min(icx, width - 1))
        icy = max(0, min(icy, height - 1))
        local_bg_brightness = float(local_bg[icy, icx])
        is_dark_bg = local_bg_brightness < min_local_bg

        # For dark backgrounds: reject if textured (image features on busy dark areas).
        # Allow if smooth (dust on dark uniform surfaces like night sky, hallways).
        # Use box-averaged texture (11x11) to avoid single-pixel misses on grid patterns.
        if is_dark_bg:
            # Sample texture in a ring AROUND the spot to avoid the bright dust pixels
            # inflating the std in dark areas (one white pixel in black = huge variance).
            spot_r = max(int(math.sqrt(area / math.pi) * 2), TEXTURE_KERNEL // 2 + 2)
            ring_r = spot_r + TEXTURE_KERNEL // 2
            ty1 = max(0, icy - ring_r)
            ty2 = min(height, icy + ring_r + 1)
            tx1 = max(0, icx - ring_r)
            tx2 = min(width, icx + ring_r + 1)
            patch_tex = local_std[ty1:ty2, tx1:tx2]
            tpy, tpx = np.ogrid[ty1-icy:ty2-icy, tx1-icx:tx2-icx]
            tdist_sq = tpx*tpx + tpy*tpy
            tex_ring = (tdist_sq >= spot_r**2) & (tdist_sq <= ring_r**2)
            dark_bg_texture = float(np.median(patch_tex[tex_ring])) if tex_ring.any() else 0.0
            if dark_bg_texture > MAX_DARK_BG_TEXTURE:
                rejected["dark_bg"] += 1
                if contrast >= 40:
                    debug_rejects.append(f"    REJECTED({cx:.0f},{cy:.0f}) area={area} contrast={contrast:.0f} by=dark_bg(bg={local_bg_brightness:.0f}<{min_local_bg:.0f} tex={dark_bg_texture:.1f})")
                log_reject(cx, cy, area, contrast, "dark_bg", f"bg={local_bg_brightness:.0f} tex={dark_bg_texture:.1f}>{MAX_DARK_BG_TEXTURE}")
                continue
            # On dark backgrounds, noise peaks are common — require higher contrast
            # to separate real dust from grain (2.5x normal threshold = 7.5x noise std).
            dark_bg_min_contrast = threshold * 2.5
            if contrast < dark_bg_min_contrast:
                rejected["dark_bg"] += 1
                log_reject(cx, cy, area, contrast, "dark_bg", f"contrast={contrast:.1f}<dark_min={dark_bg_min_contrast:.1f}")
                continue

        # Reject dim features: dust on inverted negatives is near film-base brightness.
        # Only applies to BRIGHT backgrounds — on dark backgrounds, dust is bright relative
        # to its surroundings but not in absolute terms, so this check would be wrong.
        # Scale brightness requirement linearly from SMALL (area<=10) to LARGE (area>=100).
        component_gray = gray[by:by+h, bx:bx+w]
        spot_brightness = float(np.mean(component_gray[component_mask]))
        if not is_dark_bg and area >= 10:
            area_factor = min((area - 10) / 90.0, 1.0)  # 0 at area=10, 1 at area>=100
            required_brightness = bright_ref * (
                MIN_BRIGHTNESS_FRAC_SMALL + (MIN_BRIGHTNESS_FRAC_LARGE - MIN_BRIGHTNESS_FRAC_SMALL) * area_factor
            )
            if spot_brightness < required_brightness:
                rejected["dim"] += 1
                if contrast >= 40:
                    debug_rejects.append(f"    REJECTED({cx:.0f},{cy:.0f}) area={area} contrast={contrast:.0f} by=dim(brightness={spot_brightness:.0f}<{required_brightness:.0f})")
                log_reject(cx, cy, area, contrast, "dim", f"brightness={spot_brightness:.0f}<{required_brightness:.0f}")
                continue

        # Reject edge halo artifacts: if the background has a strong gradient relative
        # to spot contrast, the diff is unreliable (caused by Gaussian smoothing near edges).
        # Using ratio (gradient/contrast) so high-contrast dust on moderate gradients passes.
        local_bg_grad = float(bg_gradient[icy, icx])
        grad_ratio = local_bg_grad / contrast if contrast > 0 else 999
        if grad_ratio > MAX_BG_GRADIENT_RATIO:
            rejected["edge"] += 1
            if contrast >= 40:
                debug_rejects.append(f"    REJECTED({cx:.0f},{cy:.0f}) area={area} contrast={contrast:.0f} by=edge(grad={local_bg_grad:.1f} ratio={grad_ratio:.2f}>{MAX_BG_GRADIENT_RATIO})")
            log_reject(cx, cy, area, contrast, "edge", f"grad={local_bg_grad:.1f} ratio={grad_ratio:.2f}>{MAX_BG_GRADIENT_RATIO}")
            continue

        # Reject bright spots embedded in dark features (e.g. window reflections):
        # Dust sits on a uniform background — no dark ring nearby.
        # A window/porthole has a dark frame around the bright reflection.
        # Only check on bright backgrounds — on dark backgrounds, noise alone makes
        # the low percentile unreliable (naturally low pixels from grain, not dark frames).
        radius_px = math.sqrt(area / math.pi)
        if not is_dark_bg:
            check_r = max(int(radius_px * 3), 8)
            ny1 = max(0, icy - check_r)
            ny2 = min(height, icy + check_r + 1)
            nx1 = max(0, icx - check_r)
            nx2 = min(width, icx + check_r + 1)
            neighborhood = gray[ny1:ny2, nx1:nx2]
            local_low = float(np.percentile(neighborhood, 5))
            surround_ratio = local_low / local_bg_brightness if local_bg_brightness > 0 else 1.0
            if surround_ratio < MIN_SURROUND_BG_RATIO:
                rejected["embedded"] += 1
                if contrast >= 40:
                    debug_rejects.append(f"    REJECTED({cx:.0f},{cy:.0f}) area={area} contrast={contrast:.0f} by=embedded(low5={local_low:.0f} bg={local_bg_brightness:.0f} ratio={surround_ratio:.2f}<{MIN_SURROUND_BG_RATIO})")
                log_reject(cx, cy, area, contrast, "embedded", f"surround={surround_ratio:.2f}<{MIN_SURROUND_BG_RATIO}")
                continue

        # Texture: measure in the surrounding ring, not inside the spot.
        # For large spots, the centroid is inside the bright area and inflates local_std.
        # Instead, sample texture in a ring from radius_px*1.5 to radius_px*3 around centroid.
        ring_inner = max(int(enc_r * 1.5), 2)
        ring_outer = max(int(enc_r * 3), ring_inner + TEXTURE_KERNEL)
        # Extract surrounding patch
        y1 = max(0, icy - ring_outer)
        y2 = min(height, icy + ring_outer + 1)
        x1 = max(0, icx - ring_outer)
        x2 = min(width, icx + ring_outer + 1)
        patch_std = local_std[y1:y2, x1:x2]
        # Build ring mask in patch coordinates
        py, px = np.ogrid[y1-icy:y2-icy, x1-icx:x2-icx]
        dist_sq = px*px + py*py
        ring_mask = (dist_sq >= ring_inner**2) & (dist_sq <= ring_outer**2)
        if ring_mask.any():
            local_texture = float(np.median(patch_std[ring_mask]))
        else:
            local_texture = float(local_std[icy, icx])

        # Reject spots in textured/busy areas — these are image features, not dust.
        # Larger spots need smoother surroundings: a tiny speck can sit in moderate texture,
        # but a large blob in moderate texture is almost certainly an image feature.
        area_tex_factor = min((area - MIN_SPOT_AREA) / (200 - MIN_SPOT_AREA), 1.0)
        max_texture = MAX_LOCAL_TEXTURE_SMALL - (MAX_LOCAL_TEXTURE_SMALL - MAX_LOCAL_TEXTURE_LARGE) * area_tex_factor
        if local_texture > max_texture:
            rejected["texture"] += 1
            if contrast >= 40 and area >= 8:
                debug_rejects.append(f"    REJECTED({cx:.0f},{cy:.0f}) area={area} contrast={contrast:.0f} by=texture({local_texture:.1f}>{max_texture:.1f})")
            log_reject(cx, cy, area, contrast, "texture", f"texture={local_texture:.1f}>{max_texture:.1f}")
            continue

        # Reject spots hidden in grain — must stand out above local noise
        if local_texture > 0 and contrast / local_texture < MIN_CONTRAST_TEXTURE_RATIO:
            rejected["ratio"] += 1
            if contrast >= 40:
                debug_rejects.append(f"    REJECTED({cx:.0f},{cy:.0f}) area={area} contrast={contrast:.0f} by=ratio({contrast/local_texture:.1f}<{MIN_CONTRAST_TEXTURE_RATIO})")
            log_reject(cx, cy, area, contrast, "ratio", f"ratio={contrast/local_texture:.1f}<{MIN_CONTRAST_TEXTURE_RATIO}")
            continue

        # Large-scale context texture: dust sits on smooth backgrounds (sky, plain walls).
        # FPs in crowd/foliage scenes pass the small-scale ring texture check (they can sit
        # on a locally smooth patch), but the 200px context around them is textured.
        # Measure median of pre-computed local_std across a 200px radius circle, excluding
        # the immediate ring already checked. Uses already-computed local_std (no extra blur).
        ctx_r = 200
        ctx_y1 = max(0, icy - ctx_r)
        ctx_y2 = min(height, icy + ctx_r + 1)
        ctx_x1 = max(0, icx - ctx_r)
        ctx_x2 = min(width, icx + ctx_r + 1)
        ctx_patch = local_std[ctx_y1:ctx_y2, ctx_x1:ctx_x2]
        ctx_py, ctx_px = np.ogrid[ctx_y1 - icy:ctx_y2 - icy, ctx_x1 - icx:ctx_x2 - icx]
        ctx_dist_sq = ctx_px * ctx_px + ctx_py * ctx_py
        # Exclude the inner ring zone (already measured by local_texture check)
        ctx_excl_r = ring_outer + TEXTURE_KERNEL // 2
        ctx_mask = (ctx_dist_sq <= ctx_r ** 2) & (ctx_dist_sq > ctx_excl_r ** 2)
        if ctx_mask.any():
            context_texture = float(np.median(ctx_patch[ctx_mask]))
        else:
            context_texture = float(np.median(ctx_patch))
        if context_texture > MAX_CONTEXT_TEXTURE:
            rejected["context"] += 1
            if contrast >= 40:
                debug_rejects.append(f"    REJECTED({cx:.0f},{cy:.0f}) area={area} contrast={contrast:.0f} by=context(ctx={context_texture:.1f}>{MAX_CONTEXT_TEXTURE})")
            log_reject(cx, cy, area, contrast, "context", f"ctx={context_texture:.1f}>{MAX_CONTEXT_TEXTURE}")
            continue

        # Soft voting: require MIN_DUST_VOTES out of 3 signals to clearly indicate dust.
        # Each hard filter above has an "inner zone" (well below threshold) that votes YES.
        # A spot that just barely passed multiple filters is suspicious — real dust sits
        # comfortably within each filter's range, not on the boundary of several.
        dust_votes = 0
        if context_texture < SOFT_CONTEXT_VOTE_THRESHOLD:
            dust_votes += 1   # scene is clearly smooth (sky, wall)
        if local_texture < SOFT_TEXTURE_VOTE_THRESHOLD:
            dust_votes += 1   # immediate surroundings are clearly smooth
        if local_texture > 0 and contrast / local_texture > SOFT_RATIO_VOTE_THRESHOLD:
            dust_votes += 1   # spot clearly stands out above local noise
        if dust_votes < MIN_DUST_VOTES:
            rejected["votes"] += 1
            ratio_str = f"{contrast/local_texture:.1f}" if local_texture > 0 else "inf"
            if contrast >= 40:
                debug_rejects.append(f"    REJECTED({cx:.0f},{cy:.0f}) area={area} contrast={contrast:.0f} by=votes({dust_votes}/{MIN_DUST_VOTES} ctx={context_texture:.1f} tex={local_texture:.1f} ratio={ratio_str})")
            log_reject(cx, cy, area, contrast, "votes",
                       f"votes={dust_votes}/{MIN_DUST_VOTES} ctx={context_texture:.1f} tex={local_texture:.1f} ratio={ratio_str}")
            continue

        # Reject colored features: dust matches the local color cast (low excess saturation).
        # Colored image features (leaves, symbols) are more saturated than their surroundings.
        component_sat = saturation[by:by+h, bx:bx+w]
        spot_sat = float(np.mean(component_sat[component_mask]))
        patch_sat = saturation[y1:y2, x1:x2]
        if ring_mask.any():
            surround_sat = float(np.median(patch_sat[ring_mask]))
        else:
            surround_sat = spot_sat
        excess_sat = spot_sat - surround_sat
        if excess_sat > MAX_EXCESS_SATURATION:
            rejected["color"] += 1
            if contrast >= 40:
                debug_rejects.append(f"    REJECTED({cx:.0f},{cy:.0f}) area={area} contrast={contrast:.0f} by=color(exSat={excess_sat:.1f}>{MAX_EXCESS_SATURATION})")
            log_reject(cx, cy, area, contrast, "color", f"exSat={excess_sat:.1f}>{MAX_EXCESS_SATURATION}")
            continue

        # Reject high-saturation emulsion artifacts: dust is achromatic and less colorful than
        # its surroundings (negative excess_sat). A spot that is both extremely colorful
        # (spot_sat > 230) AND more saturated than its immediate surroundings (excess_sat > 9)
        # is a film emulsion feature, not dust — even on highly saturated color film.
        if spot_sat > MAX_SPOT_SATURATION and excess_sat > EMULSION_EXCESS_SAT_THRESHOLD:
            rejected["sat_high"] += 1
            if contrast >= 40:
                debug_rejects.append(f"    REJECTED({cx:.0f},{cy:.0f}) area={area} contrast={contrast:.0f} by=sat_high(spotSat={spot_sat:.0f}>{MAX_SPOT_SATURATION} exSat={excess_sat:.1f}>{EMULSION_EXCESS_SAT_THRESHOLD})")
            log_reject(cx, cy, area, contrast, "sat_high", f"spot_sat={spot_sat:.1f}>{MAX_SPOT_SATURATION} excess_sat={excess_sat:.1f}>{EMULSION_EXCESS_SAT_THRESHOLD}")
            continue

        brush_radius_px = max(MIN_BRUSH_PX, enc_r * ENC_RADIUS_SCALE)
        spots.append({
            "kind": "dot",
            "_label_id": int(label_id),
            "cx": float(cx),
            "cy": float(cy),
            "radius_px": enc_r,
            "brush_radius_px": brush_radius_px,
            "threshold": threshold,
            "area": area,
            "contrast": contrast,
            "texture": local_texture,
            "excess_sat": excess_sat,
            "spot_sat": spot_sat,
            "context_texture": context_texture,
        })

    # Isolation pass: reject spots in dense clusters (crowd/foliage FPs).
    # Real film dust is sparse — a few spots per frame at most, spread across the image.
    # FP clusters from crowd highlights or foliage can produce dozens of spots in a small area.
    # For each accepted spot, count how many other accepted spots are within ISOLATION_RADIUS.
    # If too many neighbors exist, the whole dense region is suspect — reject those spots.
    if len(spots) > MAX_NEARBY_ACCEPTED:
        iso_r_sq = ISOLATION_RADIUS ** 2
        to_keep = []
        for i, si in enumerate(spots):
            neighbors = sum(
                1 for j, sj in enumerate(spots)
                if i != j and
                (si["cx"] - sj["cx"]) ** 2 + (si["cy"] - sj["cy"]) ** 2 <= iso_r_sq
            )
            if neighbors <= MAX_NEARBY_ACCEPTED:
                to_keep.append(si)
        isolation_removed = len(spots) - len(to_keep)
        if isolation_removed > 0:
            rejected["isolation"] = isolation_removed
            spots = to_keep

    print(f"  Rejected: {rejected} — accepted: {len(spots)}")
    if debug_rejects:
        print(f"  High-contrast rejects ({len(debug_rejects)}, showing top 20):")
        for msg in debug_rejects[:20]:
            print(msg)
        if len(debug_rejects) > 20:
            print(f"    ... and {len(debug_rejects) - 20} more")

    # -----------------------------------------------------------------
    # Stroke detection: elongated threads (from the threshold binary) and
    # faint scratches (ridge pass), plus blob->stroke "weighing" for elongated
    # accepted dots. Strokes are healed with a multi-node brush following the
    # detected centerline. See the Stroke detection section for the helpers.
    # -----------------------------------------------------------------
    if DETECT_STROKES:
        min_dim = min(width, height)
        stroke_spots = []
        dot_by_label = {s["_label_id"]: s for s in spots if s.get("kind") == "dot"}
        accepted_labels = set(dot_by_label.keys())

        # Minimum component area to consider as a stroke candidate: a thin fiber at the
        # min length and min width. A 2px-wide, 25px-long crisp dust fiber has area ~43,
        # so an absolute floor of 60 would silently drop it before skeletonization.
        min_cand_area = STROKE_MIN_WIDTH_PX * STROKE_MIN_LENGTH_FRAC * min_dim

        # Producer (a): elongated bright components from the threshold binary,
        # reusing the connected components already computed for circular dust.
        for lid in range(1, num_labels):
            area = stats[lid, cv2.CC_STAT_AREA]
            if area < min_cand_area:
                continue
            bw = stats[lid, cv2.CC_STAT_WIDTH]
            bh = stats[lid, cv2.CC_STAT_HEIGHT]
            bbox_area = bw * bh
            fill = area / bbox_area if bbox_area > 0 else 1.0
            is_dot = lid in accepted_labels
            # A thread/scratch fills only a small fraction of its (often near-square,
            # because curvy) bounding box. Compact components that aren't accepted dots
            # aren't strokes — skip them to limit skeletonization cost.
            if fill > STROKE_MAX_FILL_RATIO and not is_dot:
                continue
            bx = stats[lid, cv2.CC_STAT_LEFT]
            by = stats[lid, cv2.CC_STAT_TOP]
            comp = labels[by:by + bh, bx:bx + bw] == lid
            cl = extract_stroke_centerline(comp, bx, by, min_dim, cfg=cfg)
            if cl is None:
                continue
            # Validate on the confident high-threshold SEED (crispness/color/ridge are
            # judged on the sharp core, not the faint, often colour-fringed tail).
            spot, _reason = build_stroke_spot(
                cl["path"], cl["width_px"], cl["length_px"],
                gray, diff, local_std, saturation, bright_ref, min_dim, "thread", cfg=cfg)
            if spot is None:
                continue
            # Accepted: grow the faint tail at a lower threshold purely to lengthen the
            # healing brush so it covers the thread to its true end.
            _extend_stroke_tail(spot, comp, bx, by, diff, threshold, min_dim, cl["length_px"], cfg=cfg)
            _widen_brush_to_cover(spot, comp, bx, by, min_dim, cfg=cfg)
            if is_dot:
                # Weighing (spec 1.2a): convert dot -> stroke only when a single
                # circle would heal much more clean area than the stroke.
                dot = dot_by_label[lid]
                circle_area = math.pi * dot["brush_radius_px"] ** 2
                stroke_area = (spot["length_px"] * (2 * spot["brush_radius_px"])
                               + math.pi * spot["brush_radius_px"] ** 2)
                if stroke_area > 0 and circle_area / stroke_area > STROKE_PREFER_RATIO:
                    spot["_conv_label"] = lid   # this stroke replaces dot `lid`
                    stroke_spots.append(spot)
            else:
                stroke_spots.append(spot)

        # Producer (b): faint scratches via the ridge filter (below threshold).
        existing_centers = list(spots) + stroke_spots
        min_sep = max(STROKE_MIN_LENGTH_FRAC * min_dim * 0.5, 20)
        for comp, cx_off, cy_off in detect_scratch_ridges(diff, min_dim, cfg=cfg):
            cl = extract_stroke_centerline(comp, cx_off, cy_off, min_dim, cfg=cfg)
            if cl is None:
                continue
            spot, _reason = build_stroke_spot(
                cl["path"], cl["width_px"], cl["length_px"],
                gray, diff, local_std, saturation, bright_ref, min_dim, "scratch", cfg=cfg)
            if spot is None or spot["contrast"] < STROKE_RIDGE_MIN_CONTRAST:
                continue
            if _stroke_overlaps_existing(spot, existing_centers, min_sep):
                continue
            stroke_spots.append(spot)
            existing_centers.append(spot)

        # Producer (c): clipped-white threads. On a bright sky a saturated (gray~255) thread
        # has a small diff and never reaches the global threshold, so seed directly from the
        # near-clipped binary. build_stroke_spot still gates it (and bypasses crispness only
        # for the clipped core); overlaps with already-found strokes are skipped.
        clip_bin = (gray >= STROKE_CLIP_LEVEL).astype(np.uint8)
        nclip, clabels, cstats, _ccent = cv2.connectedComponentsWithStats(clip_bin, 8)
        for lid in range(1, nclip):
            area = cstats[lid, cv2.CC_STAT_AREA]
            if area < min_cand_area:
                continue
            bw = cstats[lid, cv2.CC_STAT_WIDTH]
            bh = cstats[lid, cv2.CC_STAT_HEIGHT]
            bbox_area = bw * bh
            if bbox_area > 0 and area / bbox_area > STROKE_MAX_FILL_RATIO:
                continue
            bx = cstats[lid, cv2.CC_STAT_LEFT]
            by = cstats[lid, cv2.CC_STAT_TOP]
            comp = clabels[by:by + bh, bx:bx + bw] == lid
            cl = extract_stroke_centerline(comp, bx, by, min_dim, cfg=cfg)
            if cl is None:
                continue
            spot, _reason = build_stroke_spot(
                cl["path"], cl["width_px"], cl["length_px"],
                gray, diff, local_std, saturation, bright_ref, min_dim, "thread", cfg=cfg)
            if spot is None:
                continue
            if _stroke_overlaps_existing(spot, existing_centers, min_sep):
                continue
            _extend_stroke_tail(spot, comp, bx, by, diff, threshold, min_dim, cl["length_px"], cfg=cfg)
            _widen_brush_to_cover(spot, comp, bx, by, min_dim, cfg=cfg)
            stroke_spots.append(spot)
            existing_centers.append(spot)

        # Producer (d): faint, long, thin, nearly axis-aligned scratches (film transport).
        # Too faint for the global threshold; seeded by an integrated horizontal/vertical
        # ridge response. build_stroke_spot rejects scene lines (busy/coloured backgrounds).
        if STREAK_DETECT:
            for comp, sx, sy in detect_axis_streaks(diff, min_dim, cfg=cfg):
                cl = extract_stroke_centerline(comp, sx, sy, min_dim, cfg=cfg)
                if cl is None:
                    continue
                spot, _reason = build_stroke_spot(
                    cl["path"], cl["width_px"], cl["length_px"],
                    gray, diff, local_std, saturation, bright_ref, min_dim, "scratch", cfg=cfg)
                if spot is None:
                    continue
                if _stroke_overlaps_existing(spot, existing_centers, min_sep):
                    continue
                _extend_stroke_tail(spot, comp, sx, sy, diff, threshold, min_dim, cl["length_px"],
                                    bridge=True, cfg=cfg)
                stroke_spots.append(spot)
                existing_centers.append(spot)

        # Producer (e): ultra-faint FULL-WIDTH transport scratches via Radon accumulation.
        # These are at the noise floor and fragmented, so they fail the per-point build gates;
        # the Radon accumulation across the whole frame IS the (conservative) validation, so
        # the spot is built directly. See detect_radon_streaks / STREAK_RADON_* constants.
        if STREAK_RADON_DETECT:
            for path, rx0, rx1 in detect_radon_streaks(diff, local_std, cfg=cfg):
                spot = _make_radon_streak_spot(path, diff, min_dim, cfg=cfg)
                if spot is None:
                    continue
                if _stroke_overlaps_existing(spot, existing_centers, min_sep):
                    continue
                stroke_spots.append(spot)
                existing_centers.append(spot)

        # Final dedup on the FINALISED midpoints: the same defect can be found by more than one
        # producer (e.g. a bright thread seen by both the diff and clip-seed passes), and the
        # per-producer overlap check uses pre-extension midpoints so it can miss them. Keep the
        # larger-brush (more-covering) version of any near-coincident, similar-length pair.
        if len(stroke_spots) > 1:
            deduped = []
            for s in sorted(stroke_spots, key=lambda x: -x["brush_radius_px"]):
                dup = False
                for k in deduped:
                    if (math.hypot(s["cx"] - k["cx"], s["cy"] - k["cy"]) < min_sep and
                            min(s["length_px"], k["length_px"]) >=
                            0.5 * max(s["length_px"], k["length_px"])):
                        dup = True
                        break
                if not dup:
                    deduped.append(s)
            stroke_spots = deduped

        # Field-isolation: drop strokes embedded in a FIELD of many thin line-structures
        # (reed/grass fields, yacht masts & rigging). Each such line is individually
        # thread-like, so only the surrounding density tells them apart. Count ALL elongated
        # bright components nearby (most are sub-threshold for stroke acceptance). A
        # converted-from-dot stroke dropped here simply restores its dot.
        if stroke_spots:
            line_cx, line_cy = [], []
            for lid2 in range(1, num_labels):
                a2 = stats[lid2, cv2.CC_STAT_AREA]
                cw = stats[lid2, cv2.CC_STAT_WIDTH]
                ch = stats[lid2, cv2.CC_STAT_HEIGHT]
                longdim = max(cw, ch)
                if a2 < STROKE_FIELD_CAND_MIN_AREA or longdim < STROKE_FIELD_CAND_MIN_DIM:
                    continue
                fillc = a2 / (cw * ch) if cw * ch else 1.0
                if (fillc < STROKE_FIELD_CAND_MAX_FILL and
                        longdim / max(1, min(cw, ch)) >= STROKE_FIELD_CAND_MIN_ELONG):
                    cxy = centroids[lid2]
                    line_cx.append(cxy[0]); line_cy.append(cxy[1])
            if line_cx:
                lcx = np.asarray(line_cx); lcy = np.asarray(line_cy)
                inner = STROKE_FIELD_INNER_PX
                outer = STROKE_FIELD_RADIUS_FRAC * min_dim
                kept = []
                for s in stroke_spots:
                    dist = np.hypot(lcx - s["cx"], lcy - s["cy"])
                    near = int(np.count_nonzero((dist > inner) & (dist < outer)))
                    if near < STROKE_FIELD_MAX_LINE_CANDS:
                        kept.append(s)
                stroke_spots = kept

        # Second field signal: dense cluster of ACCEPTED strokes (a tight reed/grass field).
        if len(stroke_spots) > STROKE_FIELD_MAX_NEIGHBORS + 1:
            Rn = STROKE_FIELD_NBR_RADIUS_FRAC * min_dim
            cen = [(s["cx"], s["cy"]) for s in stroke_spots]
            kept = []
            for i, s in enumerate(stroke_spots):
                xi, yi = cen[i]
                nbr = sum(1 for j, (xj, yj) in enumerate(cen)
                          if j != i and abs(xj - xi) < Rn and abs(yj - yi) < Rn)
                if nbr <= STROKE_FIELD_MAX_NEIGHBORS:
                    kept.append(s)
            stroke_spots = kept

        # Only the surviving converted strokes actually drop their dots.
        labels_to_drop = {s.pop("_conv_label") for s in stroke_spots if "_conv_label" in s}
        if labels_to_drop:
            spots = [s for s in spots if s.get("_label_id") not in labels_to_drop]

        # Heal-split: skip busy crossings so the brush only heals the smooth runs. Only for
        # SCRATCHES (long lines that genuinely cross smooth sky and busy structure). Short
        # threads sit on a locally-uniform background — splitting them on mere grain would
        # wrongly drop them, so they pass through whole.
        if HEAL_SPLIT_BUSY and stroke_spots:
            split = []
            for s in stroke_spots:
                if s.get("source") == "scratch":
                    split.extend(split_stroke_at_busy(s, local_std, min_dim, cfg=cfg))
                else:
                    split.append(s)
            stroke_spots = split

        if stroke_spots:
            print(f"  Strokes detected: {len(stroke_spots)} "
                  f"({len(labels_to_drop)} converted from dots)")
        spots = spots + stroke_spots

    # Strip internal-only fields not meant for serialization/consumers.
    for spot in spots:
        spot.pop("_label_id", None)

    # Find optimal healing sources for each spot
    if spots:
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        L_f32 = img_lab[:, :, 0].astype(np.float32)
        for spot in spots:
            if spot.get("kind") == "stroke":
                src_cx, src_cy = find_stroke_healing_source(
                    spot["path"], spot["stroke_width_px"], spot["brush_radius_px"],
                    local_std, spots, width, height, cfg=cfg)
            else:
                src_cx, src_cy = find_healing_source(
                    spot["cx"], spot["cy"], spot["radius_px"], spot["brush_radius_px"],
                    img_lab, L_f32, local_std, spots, width, height, cfg=cfg)
            spot["src_cx"] = src_cx
            spot["src_cy"] = src_cy

    return spots, rejected_candidates, None, local_std


# ===================================================================
# Source point detection
# ===================================================================

def find_healing_source(cx, cy, radius_px, brush_radius_px, img_lab, L_f32, local_std, all_spots, width, height, cfg=None):
    """Find the best healing source position using NCC template matching.

    Extracts a template from the clean surroundings of the dust spot (a patch
    of radius ring_outer centred on the spot, with the dust centre filled in
    so it does not bias the match) and searches for the best-matching location
    in the valid annulus via normalised cross-correlation on the L channel.

    NCC matches actual pixel patterns, not just statistics, so a region of
    uniform water scores much higher than a region where water meets a wooden
    structure edge — even if both have a similar mean brightness.

    Returns (src_cx, src_cy) in pixel coordinates.
    """
    cfg = DEFAULT_TUNING if cfg is None else cfg
    SOURCE_SEARCH_INNER_FACTOR = cfg.SOURCE_SEARCH_INNER_FACTOR
    SOURCE_SEARCH_MAX_RADIUS = cfg.SOURCE_SEARCH_MAX_RADIUS
    SOURCE_SEARCH_MIN_RADIUS = cfg.SOURCE_SEARCH_MIN_RADIUS
    icx = max(0, min(int(round(cx)), width - 1))
    icy = max(0, min(int(round(cy)), height - 1))

    ring_inner = max(int(radius_px * 1.5), 2)
    ring_outer = max(int(radius_px * 3), ring_inner + 5)

    # --- Build template: L-channel patch of radius ring_outer around the dust ---
    # Fill the dust centre with the ring mean so it does not make the matcher
    # seek other bright blobs rather than matching the background content.
    tmpl_r = ring_outer
    ty1 = max(0, icy - tmpl_r); ty2 = min(height, icy + tmpl_r + 1)
    tx1 = max(0, icx - tmpl_r); tx2 = min(width,  icx + tmpl_r + 1)
    template = L_f32[ty1:ty2, tx1:tx2].copy()
    t_icy = icy - ty1   # dust centre row within template
    t_icx = icx - tx1   # dust centre col within template
    py_g = np.arange(template.shape[0], dtype=np.float32).reshape(-1, 1)
    px_g = np.arange(template.shape[1], dtype=np.float32).reshape(1, -1)
    dust_mask = (px_g - t_icx) ** 2 + (py_g - t_icy) ** 2 <= ring_inner ** 2
    if dust_mask.any():
        ring_vals = template[~dust_mask]
        fill = float(np.mean(ring_vals)) if ring_vals.size > 0 else float(np.mean(template))
        template[dust_mask] = fill
    tmpl_h, tmpl_w = template.shape

    # --- Annulus search bounds ---
    # Non-overlap: source brush must not intersect dust brush.
    search_inner = max(
        int(radius_px * SOURCE_SEARCH_INNER_FACTOR),
        SOURCE_SEARCH_MIN_RADIUS // 2,
        math.ceil(2 * brush_radius_px),
    )
    search_outer = max(
        min(int(radius_px * 10), SOURCE_SEARCH_MAX_RADIUS),
        SOURCE_SEARCH_MIN_RADIUS,
        search_inner + SOURCE_SEARCH_MIN_RADIUS,
    )

    # Default fallback
    dim = max(width, height)
    default_src_x = min(width - 1, max(0, cx + HEAL_SOURCE_OFFSET_X * dim))
    default_src_y = min(height - 1, max(0, cy + HEAL_SOURCE_OFFSET_Y * dim))

    # Forbidden zones: other accepted dust spots
    forbidden = [
        (int(round(s["cx"])), int(round(s["cy"])), max(int(s["radius_px"]) * 2, 4))
        for s in all_spots
        if not (abs(s["cx"] - cx) < 0.5 and abs(s["cy"] - cy) < 0.5)
    ]

    # --- NCC on a small crop of the image covering the search annulus ---
    sb_y1 = max(0, icy - search_outer - tmpl_r)
    sb_y2 = min(height, icy + search_outer + tmpl_r + 1)
    sb_x1 = max(0, icx - search_outer - tmpl_r)
    sb_x2 = min(width,  icx + search_outer + tmpl_r + 1)
    search_region = L_f32[sb_y1:sb_y2, sb_x1:sb_x2]

    if search_region.shape[0] < tmpl_h or search_region.shape[1] < tmpl_w:
        return (default_src_x, default_src_y)

    # TM_SQDIFF_NORMED: lower score = better match (seek minimum).
    # For a near-uniform template (dust on plain sky/water), SQDIFF measures
    # the variance of the source patch — so clean sky scores near 0 and a wire
    # or edge scores high. For a textured template it measures pixel-by-pixel
    # mismatch, which also correctly prefers similar content over different content.
    # TM_CCOEFF_NORMED (correlation) degenerates to noise on flat templates
    # because both numerator and denominator approach zero.
    result = cv2.matchTemplate(search_region, template, cv2.TM_SQDIFF_NORMED)
    if result.size == 0:
        return (default_src_x, default_src_y)

    # result[ry, rx]: SQDIFF score when template top-left is at (sb_y1+ry, sb_x1+rx),
    # meaning the template centre aligns to image coords
    # (sb_y1 + ry + t_icy, sb_x1 + rx + t_icx).
    result_h, result_w = result.shape
    ry_arr = np.arange(result_h, dtype=np.float32).reshape(-1, 1)
    rx_arr = np.arange(result_w, dtype=np.float32).reshape(1, -1)
    centre_y = sb_y1 + ry_arr + t_icy
    centre_x = sb_x1 + rx_arr + t_icx

    dist = np.sqrt((centre_x - icx) ** 2 + (centre_y - icy) ** 2)
    valid = (dist >= search_inner) & (dist <= search_outer)
    for fx, fy, fr in forbidden:
        valid &= np.hypot(centre_x - fx, centre_y - fy) >= fr

    if not valid.any():
        return (default_src_x, default_src_y)

    # Mask invalid positions with a large sentinel, then find the minimum.
    masked = np.where(valid, result, 2.0)
    best_ry, best_rx = np.unravel_index(int(np.argmin(masked)), result.shape)
    return (float(sb_x1 + best_rx + t_icx), float(sb_y1 + best_ry + t_icy))


# ===================================================================
# Stroke (thread / scratch) detection
# ===================================================================
#
# Threads (opaque filaments) and film scratches are elongated defects. They are
# extracted as a *centerline path* and healed with a multi-node darktable brush
# stroke (same brush form as a dot, just with N nodes). All data lives in the
# spot dict under kind="stroke" so every consumer (debug JSON, overlay, debug UI,
# XMP) stays in sync with the canonical-data principle.

def _skeleton_longest_path(skel):
    """Order a 1-px boolean skeleton into the longest geodesic path.

    Uses double-BFS (tree-diameter): farthest pixel from an arbitrary start, then
    farthest pixel from there, reconstructing the connecting path. Spurs/branches
    are ignored. Returns a list of (x, y) integer pixel coords.
    """
    from collections import deque
    ys, xs = np.nonzero(skel)
    if len(xs) == 0:
        return []
    if len(xs) == 1:
        return [(int(xs[0]), int(ys[0]))]
    coords = list(zip(xs.tolist(), ys.tolist()))  # (x, y)
    index = {p: i for i, p in enumerate(coords)}
    nbrs8 = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    adj = [[] for _ in coords]
    for i, (x, y) in enumerate(coords):
        for dx, dy in nbrs8:
            j = index.get((x + dx, y + dy))
            if j is not None:
                adj[i].append(j)

    def bfs(src):
        dist = [-1] * len(coords)
        par = [-1] * len(coords)
        dist[src] = 0
        dq = deque([src])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    par[v] = u
                    dq.append(v)
        far = max(range(len(coords)), key=lambda k: dist[k])
        return far, par

    a, _ = bfs(0)
    b, par = bfs(a)
    path = []
    cur = b
    while cur != -1:
        path.append(coords[cur])
        cur = par[cur]
    path.reverse()
    return path


def hysteresis_extend_component(comp_bool, bx, by, diff, threshold, min_dim, pad=None,
                                bridge_kernel=None, cfg=None):
    """Grow a stroke seed component into its faint tail via local hysteresis.

    A thread frequently fades below the global threshold at its ends, so the seed
    component stops short of the real tip. Re-threshold a padded window around the seed
    at STROKE_HYST_LOW_FACTOR * threshold and keep only the low-threshold component(s)
    CONNECTED to the seed — so the result tracks the thread to where it fades into
    background but cannot jump to a disconnected bright object.

    comp_bool: bool mask of the seed in its own bbox (bw x bh).
    bx, by: seed bbox origin in full-image coords.
    pad: search-window pad in px (default ~STROKE_HYST_PAD_FRAC*min_dim). A long faint
        scratch can continue 100+px, so the caller passes a length-scaled pad.
    Returns (extended_mask_bool, new_x_off, new_y_off).
    """
    cfg = DEFAULT_TUNING if cfg is None else cfg
    STROKE_HYST_LOW_FACTOR = cfg.STROKE_HYST_LOW_FACTOR
    STROKE_HYST_PAD_FRAC = cfg.STROKE_HYST_PAD_FRAC
    h, w = diff.shape
    bh, bw = comp_bool.shape
    if pad is None:
        pad = STROKE_HYST_PAD_FRAC * min_dim
    pad = int(max(pad, 8))
    x0 = max(0, bx - pad); y0 = max(0, by - pad)
    x1 = min(w, bx + bw + pad); y1 = min(h, by + bh + pad)

    low = threshold * STROKE_HYST_LOW_FACTOR
    low_bin = (diff[y0:y1, x0:x1] > low).astype(np.uint8)
    # Bridge small noise gaps ALONG the stroke axis so a faint, gappy scratch stays
    # connected to its seed (the close is 1px thick across the axis, so it does not widen).
    if bridge_kernel is not None:
        low_bin = cv2.morphologyEx(low_bin, cv2.MORPH_CLOSE, bridge_kernel)

    seed_win = np.zeros(low_bin.shape, dtype=bool)
    seed_win[(by - y0):(by - y0 + bh), (bx - x0):(bx - x0 + bw)] = comp_bool

    _, lbl = cv2.connectedComponents(low_bin, connectivity=8)
    keep = set(np.unique(lbl[seed_win]))
    keep.discard(0)
    if not keep:
        return comp_bool, bx, by
    ext = np.isin(lbl, list(keep))
    return ext, x0, y0


def _extend_stroke_tail(spot, seed_comp, bx, by, diff, threshold, min_dim, seed_length,
                        bridge=False, cfg=None):
    """Replace an accepted stroke's path with a tail-extended centerline (in place).

    The stroke has already passed all gates on its confident seed; this only lengthens the
    healing brush to reach the thread's faint end. Keeps the seed-measured width. Skips the
    extension if it would more than ~2.5x the seed length (runaway growth on busy areas).
    The search window scales with the seed length so a long scratch reaches its long tail.

    bridge: only true for faint AXIS-ALIGNED scratches (producer d). It closes noise gaps
    along the stroke axis — but for a normal (e.g. diagonal) thread that is merely a bit more
    horizontal than vertical, that horizontal close connects spurious pixels and bends the
    centerline to the axis. So threads extend WITHOUT it.
    """
    cfg = DEFAULT_TUNING if cfg is None else cfg
    STROKE_HYST_PAD_FRAC = cfg.STROKE_HYST_PAD_FRAC
    STROKE_HYST_PAD_LEN_FRAC = cfg.STROKE_HYST_PAD_LEN_FRAC
    STROKE_HYST_BRIDGE_PX = cfg.STROKE_HYST_BRIDGE_PX
    pad = max(STROKE_HYST_PAD_FRAC * min_dim, seed_length * STROKE_HYST_PAD_LEN_FRAC)
    bridge_kernel = None
    if bridge:
        p0 = spot["path"][0]; p1 = spot["path"][-1]
        if abs(p1[0] - p0[0]) >= abs(p1[1] - p0[1]):
            bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (STROKE_HYST_BRIDGE_PX, 1))
        else:
            bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, STROKE_HYST_BRIDGE_PX))
    ext, ex0, ey0 = hysteresis_extend_component(seed_comp, bx, by, diff, threshold, min_dim,
                                                pad, bridge_kernel, cfg=cfg)
    cl = extract_stroke_centerline(ext, ex0, ey0, min_dim, cfg=cfg)
    if cl is None:
        return
    if cl["length_px"] <= seed_length * 1.02 or cl["length_px"] > seed_length * 2.5:
        return
    path = cl["path"]
    mid = path[len(path) // 2]
    spot["path"] = [[float(p[0]), float(p[1])] for p in path]
    spot["length_px"] = float(cl["length_px"])
    spot["cx"] = float(mid[0])
    spot["cy"] = float(mid[1])


def extract_stroke_centerline(component_mask, x_off, y_off, min_dim, cfg=None):
    """Extract a simplified centerline path and width from a binary component mask.

    component_mask: bool 2D array cropped to the component bounding box.
    x_off, y_off: bbox origin in full-image pixel coords.
    Returns dict {path (full-image [x,y] key points), length_px, width_px} or None.
    """
    cfg = DEFAULT_TUNING if cfg is None else cfg
    STROKE_MIN_WIDTH_PX = cfg.STROKE_MIN_WIDTH_PX
    STROKE_DP_EPS_FRAC = cfg.STROKE_DP_EPS_FRAC
    STROKE_MAX_KEYPOINTS = cfg.STROKE_MAX_KEYPOINTS
    if int(component_mask.sum()) < 3:
        return None
    skel = skeletonize(component_mask)
    if int(skel.sum()) < 2:
        return None
    path_local = _skeleton_longest_path(skel)
    if len(path_local) < 2:
        return None

    length_px = float(sum(
        math.hypot(path_local[i + 1][0] - path_local[i][0],
                   path_local[i + 1][1] - path_local[i][1])
        for i in range(len(path_local) - 1)))

    # Width from distance transform sampled along the skeleton (2x = full width).
    dt = cv2.distanceTransform(component_mask.astype(np.uint8), cv2.DIST_L2, 5)
    sk_xs = np.array([p[0] for p in path_local])
    sk_ys = np.array([p[1] for p in path_local])
    width_px = max(float(np.median(dt[sk_ys, sk_xs]) * 2.0), STROKE_MIN_WIDTH_PX)

    # Simplify to a few key points (Douglas-Peucker on the open polyline).
    eps = max(1.0, STROKE_DP_EPS_FRAC * min_dim)
    cnt = np.array(path_local, dtype=np.int32).reshape(-1, 1, 2)
    simplified = cv2.approxPolyDP(cnt, eps, False).reshape(-1, 2)
    if len(simplified) < 2:
        simplified = np.array([path_local[0], path_local[-1]], dtype=np.int32)
    if len(simplified) > STROKE_MAX_KEYPOINTS:
        idxs = np.linspace(0, len(simplified) - 1, STROKE_MAX_KEYPOINTS).round().astype(int)
        simplified = simplified[np.unique(idxs)]

    path_full = [[float(px + x_off), float(py + y_off)] for px, py in simplified]
    return {"path": path_full, "length_px": length_px, "width_px": width_px}


def _stroke_band_masks(path_full, width_px, shape, cfg=None):
    """Rasterize a stroke into (core, ring) boolean masks within a cropped bbox.

    core: the stroke band (polyline of given width) — used for contrast/brightness.
    ring: a surrounding annulus offset far enough from the core that the local-texture
    kernel (TEXTURE_KERNEL) is not contaminated by the bright stroke itself — used for
    sampling background texture/saturation.
    Returns (core_bool, ring_bool, (x0, y0, x1, y1)) — masks are bbox-local.
    """
    cfg = DEFAULT_TUNING if cfg is None else cfg
    TEXTURE_KERNEL = cfg.TEXTURE_KERNEL
    h, w = shape
    xs = [p[0] for p in path_full]
    ys = [p[1] for p in path_full]
    gap = int(TEXTURE_KERNEL // 2 + max(2, round(width_px / 2)))  # clear texture-kernel reach
    ring_band = int(max(8, round(width_px)))
    margin = gap + ring_band + 2
    x0 = max(0, int(min(xs)) - margin)
    x1 = min(w, int(max(xs)) + margin + 1)
    y0 = max(0, int(min(ys)) - margin)
    y1 = min(h, int(max(ys)) + margin + 1)
    cw, ch = x1 - x0, y1 - y0
    if cw <= 0 or ch <= 0:
        return None, None, (x0, y0, x1, y1)
    pts = np.array([[int(round(x)) - x0, int(round(y)) - y0] for x, y in path_full],
                   dtype=np.int32).reshape(-1, 1, 2)
    core = np.zeros((ch, cw), dtype=np.uint8)
    cv2.polylines(core, [pts], False, 255, thickness=max(1, int(round(width_px))))

    def _dil(m, r):
        return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))

    inner = _dil(core, gap)
    outer = _dil(core, gap + ring_band)
    ring = cv2.subtract(outer, inner)
    return core > 0, ring > 0, (x0, y0, x1, y1)


def _stroke_context_texture(path_full, width_px, local_std, min_dim, cfg=None):
    """Median local_std in a WIDE band around the centerline (excluding the near zone).

    Measures how busy the broader surroundings are — a real hair sits in a genuinely
    non-busy region, while a mark on locally-smooth-but-globally-busy structure does not.
    Returns the median texture (lower = smoother), or 0.0 if no samples.
    """
    cfg = DEFAULT_TUNING if cfg is None else cfg
    TEXTURE_KERNEL = cfg.TEXTURE_KERNEL
    STROKE_CONTEXT_RADIUS_FRAC = cfg.STROKE_CONTEXT_RADIUS_FRAC
    h, w = local_std.shape
    xs = [p[0] for p in path_full]
    ys = [p[1] for p in path_full]
    gap = int(TEXTURE_KERNEL // 2 + max(2, round(width_px)))  # skip the near (already-checked) zone
    R = int(max(gap + 20, STROKE_CONTEXT_RADIUS_FRAC * min_dim))
    x0 = max(0, int(min(xs)) - R)
    x1 = min(w, int(max(xs)) + R)
    y0 = max(0, int(min(ys)) - R)
    y1 = min(h, int(max(ys)) + R)
    cw, ch = x1 - x0, y1 - y0
    if cw <= 0 or ch <= 0:
        return 0.0
    core = np.zeros((ch, cw), dtype=np.uint8)
    pts = np.array([[int(round(x)) - x0, int(round(y)) - y0] for x, y in path_full],
                   dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(core, [pts], False, 255, thickness=max(1, int(round(width_px))))

    def _dil(m, r):
        return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))

    inner = _dil(core, gap)
    outer = _dil(core, R)
    ring = cv2.subtract(outer, inner) > 0
    if not ring.any():
        return 0.0
    return float(np.median(local_std[y0:y1, x0:x1][ring]))


def _stroke_side_brightness(path_full, width_px, gray, cfg=None):
    """Sample mean brightness along the centerline and in two parallel bands offset
    perpendicular to it (left/right). Used to tell a bright ridge (dust/scratch:
    darker background on both sides) from a structural step-edge (bright on one side).

    Returns (core_med, left_med, right_med) or None.
    """
    cfg = DEFAULT_TUNING if cfg is None else cfg
    STROKE_SIDE_OFFSET_FACTOR = cfg.STROKE_SIDE_OFFSET_FACTOR
    h, w = gray.shape
    pts = np.array(path_full, dtype=np.float64)
    if len(pts) < 2:
        return None
    off = max(width_px * STROKE_SIDE_OFFSET_FACTOR + 2.0, 4.0)
    # The skeleton of a thin curvy thread can sit ~1px off the actual brightest line, so a
    # single-pixel centerline sample UNDERESTIMATES the ridge crest — fatal on a bright
    # background where the true drop is small in absolute terms. Take the crest = max
    # brightness across the thread's width at each step instead.
    crest_reach = max(int(round(width_px / 2.0)) + 1, 2)
    core_vals, left_vals, right_vals = [], [], []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        seg = b - a
        L = math.hypot(seg[0], seg[1])
        if L < 1e-6:
            continue
        d = seg / L
        perp = np.array([d[1], -d[0]])
        n = max(1, int(L // 3))
        for t in range(n):
            pt = a + seg * (t / n)
            # core: ridge crest (max across width)
            crest = None
            for s in range(-crest_reach, crest_reach + 1):
                ix, iy = int(round(pt[0] + perp[0] * s)), int(round(pt[1] + perp[1] * s))
                if 0 <= ix < w and 0 <= iy < h:
                    v = gray[iy, ix]
                    if crest is None or v > crest:
                        crest = v
            if crest is not None:
                core_vals.append(crest)
            for dx, dy, arr in ((perp[0] * off, perp[1] * off, left_vals),
                                (-perp[0] * off, -perp[1] * off, right_vals)):
                ix, iy = int(round(pt[0] + dx)), int(round(pt[1] + dy))
                if 0 <= ix < w and 0 <= iy < h:
                    arr.append(gray[iy, ix])
    if not core_vals or not left_vals or not right_vals:
        return None
    return (float(np.median(core_vals)), float(np.median(left_vals)),
            float(np.median(right_vals)))


def _bilinear(gray, x, y):
    """Bilinear sample of a 2D array at float (x, y); NaN if out of bounds."""
    h, w = gray.shape
    if x < 0 or y < 0 or x >= w - 1 or y >= h - 1:
        return np.nan
    x0, y0 = int(x), int(y)
    fx, fy = x - x0, y - y0
    g = gray
    return (g[y0, x0] * (1 - fx) * (1 - fy) + g[y0, x0 + 1] * fx * (1 - fy)
            + g[y0 + 1, x0] * (1 - fx) * fy + g[y0 + 1, x0 + 1] * fx * fy)


def stroke_edge_crispness(path_full, width_px, gray):
    """Measure how sharp the stroke's edges are ("crispiness").

    Dust on the negative is in the scanner's focal plane, so its edges are razor-sharp:
    the perpendicular brightness profile jumps from background to peak in ~1px. A wire or
    hair in the photographed SCENE is usually slightly defocused, so its edge is a soft
    ramp spanning several px. We sample perpendicular profiles at points along the
    centerline and return the median of (max edge gradient / bump amplitude), which is
    ~1.0 for crisp dust and ~0.3 for a soft wire. Normalising by amplitude makes it
    independent of contrast. Returns a float (0.0 if it cannot be measured).
    """
    gray = gray.astype(np.float32)
    pts = np.array(path_full, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    d = pts[-1] - pts[0]
    Lp = math.hypot(d[0], d[1])
    if Lp < 1e-6:
        return 0.0
    ux, uy = d[0] / Lp, d[1] / Lp
    px, py = uy, -ux  # unit perpendicular

    # Resample the centerline at ~6px spacing for the profile samples.
    samples = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(seg_len / 6))
        for k in range(n):
            samples.append(a + (b - a) * (k / n))
    samples.append(pts[-1])

    reach = max(8.0, width_px * 2.5)
    ts = np.arange(-reach, reach + 0.5, 0.5)
    steeps = []
    for c in samples:
        prof = np.array([_bilinear(gray, c[0] + px * t, c[1] + py * t) for t in ts])
        if np.isnan(prof).all():
            continue
        peak = np.nanmax(prof)
        base = np.nanpercentile(prof, 10)
        amp = peak - base
        if amp < 5:
            continue
        grad = np.abs(np.diff(prof)) / 0.5
        steeps.append(float(np.nanmax(grad)) / amp)
    return float(np.median(steeps)) if steeps else 0.0


def component_cover_radius(path_full, comp, bx, by):
    """Radius the brush must reach so the stroke's path covers the WHOLE detected component.

    The skeleton longest-path centerline can miss a branch of a hooked/checkmark dust blob,
    so part of the bright component sits off the centerline and is left unhealed. Return a
    high percentile of the distance from each component pixel to the (densified) path — for a
    thin thread this is just its half-width (no change), for a hook it reaches the off-axis
    branch. Returns 0.0 if it cannot be measured.
    """
    ys, xs = np.where(comp)
    if xs.size == 0:
        return 0.0
    pts = np.column_stack([xs + bx, ys + by]).astype(np.float64)
    pa = np.array(path_full, dtype=np.float64)
    dense = [pa[0]]
    for i in range(len(pa) - 1):
        a, b = pa[i], pa[i + 1]
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(seg / 3))
        for k in range(1, n + 1):
            dense.append(a + (b - a) * (k / n))
    dense = np.array(dense)
    # min distance from each component pixel to the path, then a robust max
    d2 = ((pts[:, None, 0] - dense[None, :, 0]) ** 2 +
          (pts[:, None, 1] - dense[None, :, 1]) ** 2)
    mind = np.sqrt(d2.min(axis=1))
    # Use the max (a connected component has no stray far pixels) so even the off-centerline
    # branch tip is covered; the 99.5th percentile drops a one-pixel skeleton-spur outlier.
    return float(np.percentile(mind, 99.5))


def _widen_brush_to_cover(spot, comp, bx, by, min_dim, cfg=None):
    """Grow the brush radius (in place) so the stroke covers the whole detected component —
    e.g. the off-centerline branch of a hooked dust blob the centerline misses. Capped."""
    cfg = DEFAULT_TUNING if cfg is None else cfg
    STROKE_COVERAGE_MARGIN_PX = cfg.STROKE_COVERAGE_MARGIN_PX
    STROKE_MAX_BORDER_FRAC = cfg.STROKE_MAX_BORDER_FRAC
    cover_r = component_cover_radius(spot["path"], comp, bx, by)
    if cover_r > 0:
        r = max(spot["brush_radius_px"], cover_r + STROKE_COVERAGE_MARGIN_PX)
        spot["brush_radius_px"] = float(min(r, STROKE_MAX_BORDER_FRAC * min_dim))


def stroke_coverage_halfwidth(path_full, diff, cfg=None):
    """Measure how far the thread stays visibly bright from its centerline, from the actual
    perpendicular diff profile — the radius a heal brush must reach to avoid a whitish
    leftover. The distance-transform width only sees the above-threshold core and badly
    underestimates a thick/feathered or off-centre thread. At samples along the path: find
    the local peak near the centerline, then walk OUTWARD (contiguously, so a neighbouring
    thread's separate bump is not included) until the diff drops below
    max(STROKE_COVERAGE_MIN_DIFF, STROKE_COVERAGE_FRAC*peak); the half-width is the larger of
    the two sides' distances FROM THE CENTERLINE (handles off-centre skeletons). Returns the
    STROKE_COVERAGE_PCTL percentile over samples (covers the thicker stretches).
    """
    cfg = DEFAULT_TUNING if cfg is None else cfg
    STROKE_COVERAGE_FRAC = cfg.STROKE_COVERAGE_FRAC
    STROKE_COVERAGE_MIN_DIFF = cfg.STROKE_COVERAGE_MIN_DIFF
    STROKE_COVERAGE_PCTL = cfg.STROKE_COVERAGE_PCTL
    pts = np.array(path_full, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    dvec = pts[-1] - pts[0]
    Lp = math.hypot(dvec[0], dvec[1])
    if Lp < 1e-6:
        return 0.0
    ux, uy = dvec / Lp
    px, py = uy, -ux
    h, w = diff.shape

    samples = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        nseg = max(1, int(seg / 5))
        for k in range(nseg):
            samples.append(a + (b - a) * (k / nseg))
    samples.append(pts[-1])

    reach, step = 16.0, 0.5
    ts = np.arange(-reach, reach + step, step)
    ci = len(ts) // 2  # index of t == 0 (the centerline)
    halfwidths = []
    for c in samples:
        prof = np.empty(len(ts))
        for j, t in enumerate(ts):
            ix, iy = int(round(c[0] + px * t)), int(round(c[1] + py * t))
            prof[j] = diff[iy, ix] if (0 <= ix < w and 0 <= iy < h) else -1e9
        lo = max(0, ci - 3); pidx = lo + int(np.argmax(prof[lo:ci + 4]))
        peak = prof[pidx]
        if peak < 8:
            continue
        level = max(STROKE_COVERAGE_MIN_DIFF, STROKE_COVERAGE_FRAC * peak)
        r = pidx
        while r + 1 < len(prof) and prof[r + 1] >= level:
            r += 1
        l = pidx
        while l - 1 >= 0 and prof[l - 1] >= level:
            l -= 1
        halfwidths.append(max(r - ci, ci - l) * step)
    if not halfwidths:
        return 0.0
    return float(np.percentile(halfwidths, STROKE_COVERAGE_PCTL))


def build_stroke_spot(path_full, width_px, length_px, gray, diff, local_std,
                      saturation, bright_ref, min_dim, source_tag, cfg=None):
    """Apply stroke acceptance gating and build a stroke spot dict.

    Returns (spot_dict, None) on accept or (None, reason) on reject.
    """
    cfg = DEFAULT_TUNING if cfg is None else cfg
    STROKE_MIN_LENGTH_FRAC = cfg.STROKE_MIN_LENGTH_FRAC
    STROKE_MIN_ELONGATION = cfg.STROKE_MIN_ELONGATION
    STROKE_MAX_WIDTH_FRAC = cfg.STROKE_MAX_WIDTH_FRAC
    STROKE_COVERAGE_MARGIN_PX = cfg.STROKE_COVERAGE_MARGIN_PX
    STROKE_BORDER_SCALE = cfg.STROKE_BORDER_SCALE
    STROKE_MIN_BORDER_PX = cfg.STROKE_MIN_BORDER_PX
    STROKE_MAX_BORDER_FRAC = cfg.STROKE_MAX_BORDER_FRAC
    STROKE_RIDGE_MIN_CONTRAST = cfg.STROKE_RIDGE_MIN_CONTRAST
    STROKE_MAX_BAND_TEXTURE = cfg.STROKE_MAX_BAND_TEXTURE
    STROKE_MAX_CONTEXT_TEXTURE = cfg.STROKE_MAX_CONTEXT_TEXTURE
    STROKE_MAX_EXCESS_SAT = cfg.STROKE_MAX_EXCESS_SAT
    STROKE_MIN_BRIGHTNESS_FRAC = cfg.STROKE_MIN_BRIGHTNESS_FRAC
    STROKE_MIN_RIDGE_DROP = cfg.STROKE_MIN_RIDGE_DROP
    STROKE_MAX_SIDE_ASYMMETRY = cfg.STROKE_MAX_SIDE_ASYMMETRY
    STROKE_MIN_CRISPNESS = cfg.STROKE_MIN_CRISPNESS
    STROKE_CLIP_LEVEL = cfg.STROKE_CLIP_LEVEL
    if length_px < STROKE_MIN_LENGTH_FRAC * min_dim:
        return None, "short"
    if width_px > STROKE_MAX_WIDTH_FRAC * min_dim:
        return None, "wide"
    if width_px > 0 and (length_px / width_px) < STROKE_MIN_ELONGATION:
        return None, "stubby"

    core, ring, (x0, y0, x1, y1) = _stroke_band_masks(path_full, width_px, gray.shape, cfg=cfg)
    if core is None or not core.any():
        return None, "empty"

    sub_gray = gray[y0:y1, x0:x1]
    sub_diff = diff[y0:y1, x0:x1]
    sub_std = local_std[y0:y1, x0:x1]
    sub_sat = saturation[y0:y1, x0:x1]

    contrast = float(np.mean(sub_diff[core]))
    stroke_brightness = float(np.mean(sub_gray[core]))
    if stroke_brightness < STROKE_MIN_BRIGHTNESS_FRAC * bright_ref:
        return None, "dim"

    # Texture is only a loose BACKSTOP now: dust floats over the whole photo, textured or
    # not, so a busy background must NOT by itself reject a thread. The real FP guards are
    # crispness (soft scene wires/reeds) and ridge symmetry (object outlines) below. These
    # limits only catch egregiously busy structure (e.g. a wire among rigging, band~20),
    # which crispness/ridge also catch. Measured real threads sit at band/ctx 6-8.
    band_texture = float(np.median(sub_std[ring])) if ring.any() else 0.0
    if band_texture > STROKE_MAX_BAND_TEXTURE:
        return None, "texture"

    context_texture = _stroke_context_texture(path_full, width_px, local_std, min_dim, cfg=cfg)
    if context_texture > STROKE_MAX_CONTEXT_TEXTURE:
        return None, "context"

    stroke_sat = float(np.mean(sub_sat[core]))
    surround_sat = float(np.median(sub_sat[ring])) if ring.any() else stroke_sat
    excess_sat = stroke_sat - surround_sat
    if excess_sat > STROKE_MAX_EXCESS_SAT:
        return None, "color"

    # Ridge vs edge: reject structural step-edges (bright on one side only).
    sides = _stroke_side_brightness(path_full, width_px, gray, cfg=cfg)
    if sides is not None:
        core_med, left_med, right_med = sides
        bright_side = max(left_med, right_med)
        if core_med - bright_side < STROKE_MIN_RIDGE_DROP:
            return None, "edge"
        asym = abs(left_med - right_med) / max(core_med, 1.0)
        if asym > STROKE_MAX_SIDE_ASYMMETRY:
            return None, "asym"

    # Crispiness: reject soft, defocused scene wires/hairs (sharp dust only). See
    # STROKE_MIN_CRISPNESS. Computed last as it is the most expensive gate. Clipped-white
    # cores (saturated dust) bypass it — clipping + a bright noisy background make the metric
    # read low, but a soft scene wire is never blown to 255 (see STROKE_CLIP_LEVEL).
    core_is_clipped = sides is not None and sides[0] >= STROKE_CLIP_LEVEL
    crispness = stroke_edge_crispness(path_full, width_px, gray)
    if crispness < STROKE_MIN_CRISPNESS and not core_is_clipped:
        return None, "soft"

    mid = path_full[len(path_full) // 2]
    # Size the brush from the measured visible half-width (covers thick/feathered/off-centre
    # threads), falling back to the core-width estimate; cap to avoid runaway brushes.
    cover_hw = stroke_coverage_halfwidth(path_full, diff, cfg=cfg)
    brush_radius_px = max(STROKE_MIN_BORDER_PX,
                          cover_hw + STROKE_COVERAGE_MARGIN_PX,
                          width_px / 2.0 * STROKE_BORDER_SCALE)
    brush_radius_px = min(brush_radius_px, STROKE_MAX_BORDER_FRAC * min_dim)
    spot = {
        "kind": "stroke",
        "path": [[float(p[0]), float(p[1])] for p in path_full],
        "stroke_width_px": float(width_px),
        "length_px": float(length_px),
        "cx": float(mid[0]),
        "cy": float(mid[1]),
        "radius_px": float(width_px / 2.0),
        "brush_radius_px": float(brush_radius_px),
        "threshold": float(STROKE_RIDGE_MIN_CONTRAST),
        "area": int(core.sum()),
        "contrast": contrast,
        "texture": band_texture,
        "context_texture": context_texture,
        "crispness": crispness,
        "excess_sat": excess_sat,
        "spot_sat": stroke_sat,
        "source": source_tag,
    }
    return spot, None


def find_stroke_healing_source(path_full, width_px, brush_radius_px,
                               local_std, all_spots, width, height, cfg=None):
    """Find a single translation offset placing the whole stroke on clean background.

    Searches a fan of directions (not just the chord perpendicular — a curved hair has no
    single "away" direction) at increasing distance, scoring candidates by background
    smoothness under the translated path, and *guaranteeing* the translated source keeps a real
    gap from the (possibly curved) defect so darktable's clone never smears across an overlap.
    Returns (src_cx, src_cy) = first key point + best offset, in pixel coords.
    """
    cfg = DEFAULT_TUNING if cfg is None else cfg
    STROKE_MAX_BAND_TEXTURE = cfg.STROKE_MAX_BAND_TEXTURE
    STROKE_SOURCE_OFFSET_FACTOR = cfg.STROKE_SOURCE_OFFSET_FACTOR
    STROKE_SOURCE_MIN_GAP_PX = cfg.STROKE_SOURCE_MIN_GAP_PX
    HEAL_SAMPLE_STEP_PX = cfg.HEAL_SAMPLE_STEP_PX
    path_arr = np.array(path_full, dtype=np.float64)
    p0 = path_arr[0]

    # Densify the defect centerline so its actual curvature is represented when we test
    # clearance. A curved hair's key points alone leave gaps the clearance test could slip
    # through, placing the source mid-segment back onto the defect.
    dense = [path_arr[0]]
    for i in range(len(path_arr) - 1):
        a, b = path_arr[i], path_arr[i + 1]
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(seg / HEAL_SAMPLE_STEP_PX))
        for k in range(1, n + 1):
            dense.append(a + (b - a) * (k / n))
    dense = np.array(dense)

    # Real gap required between every source-brush point and the nearest DEFECT point, so the
    # two brushes never touch (darktable smears when they do).
    clearance = 2.0 * brush_radius_px + STROKE_SOURCE_MIN_GAP_PX

    step = max(width_px, 4.0)
    # Keep a real gap between the source brush and the defect brush (darktable dislikes a
    # source that touches/overlaps the defect): offset >= 2*brush_radius + gap.
    d_min = max(STROKE_SOURCE_OFFSET_FACTOR * brush_radius_px,
                clearance, 2.0 * width_px, 6.0)
    d_max = d_min + 24.0 * step
    dists = np.arange(d_min, d_max, step)
    # A fan of directions: for a straight stroke the chord-perpendicular still wins (it gives
    # the nearest clean offset), but for a curved hair an oblique direction is the only one that
    # carries the WHOLE rigid copy clear of every part of the defect.
    angles = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)

    def _min_gap(tp):
        dx = tp[:, 0][:, None] - dense[:, 0][None, :]
        dy = tp[:, 1][:, None] - dense[:, 1][None, :]
        return float(np.sqrt((dx * dx + dy * dy).min()))

    best = None              # (score, ox, oy) among clearance- AND texture-passing offsets
    nearest_clear = None     # (dist, ox, oy) among clearance-passing offsets (texture ignored)
    for ang in angles:
        dirx, diry = math.cos(ang), math.sin(ang)
        for dist in dists:
            ox, oy = dirx * dist, diry * dist
            tp = dense + np.array([ox, oy])
            if (tp[:, 0].min() < 1 or tp[:, 1].min() < 1 or
                    tp[:, 0].max() >= width - 1 or tp[:, 1].max() >= height - 1):
                continue
            if _min_gap(tp) < clearance:
                continue
            if nearest_clear is None or dist < nearest_clear[0]:
                nearest_clear = (dist, ox, oy)
            sx = np.clip(tp[:, 0].astype(int), 0, width - 1)
            sy = np.clip(tp[:, 1].astype(int), 0, height - 1)
            band = local_std[sy, sx]
            # 90th percentile, not median: a thin structure crossing the band (another hair, a
            # wire) touches only a few samples and would hide in the median, yielding a source
            # that itself sits on a defect.
            tex = float(np.percentile(band, 90))
            if tex > STROKE_MAX_BAND_TEXTURE:
                continue
            score = tex + 0.01 * dist  # prefer smooth, then nearer
            if best is None or score < best[0]:
                best = (score, ox, oy)

    if best is not None:
        _, ox, oy = best
    elif nearest_clear is not None:
        # No clean-texture spot in range, but a clearance-satisfying offset exists: take the
        # nearest. An over-heal onto slightly textured bg is recoverable; a source overlapping
        # the defect smears — never return that.
        _, ox, oy = nearest_clear
    else:
        # Truly nowhere clear in range (tiny image / huge defect): slide along the chord
        # perpendicular by d_min as a last resort.
        d = path_arr[-1] - p0
        L = math.hypot(d[0], d[1])
        ux, uy = (d[1] / L, -d[0] / L) if L >= 1e-6 else (1.0, 0.0)
        ox, oy = ux * d_min, uy * d_min
    src_x = float(min(max(p0[0] + ox, 0.0), width - 1))
    src_y = float(min(max(p0[1] + oy, 0.0), height - 1))
    return src_x, src_y


def detect_axis_streaks(diff, min_dim, cfg=None):
    """Detect faint, long, thin, nearly axis-aligned bright lines (film transport scratches).

    For each axis (horizontal, then vertical via transpose): build a ridge response
    (brighter than the rows STREAK_RIDGE_VGAP px above AND below — so step-edges are
    excluded), integrate it ALONG the line with a box filter to lift the faint line above
    noise, threshold, and keep components that are long, thin and strongly elongated.
    Yields (component_mask, x_off, y_off) in full-image coords; the caller validates each
    with build_stroke_spot (which rejects scene lines on busy/coloured backgrounds, cfg=cfg).
    """
    cfg = DEFAULT_TUNING if cfg is None else cfg
    STREAK_RIDGE_VGAP = cfg.STREAK_RIDGE_VGAP
    STREAK_INTEG_LEN = cfg.STREAK_INTEG_LEN
    STREAK_LEVEL_MULT = cfg.STREAK_LEVEL_MULT
    STREAK_MIN_LEN_FRAC = cfg.STREAK_MIN_LEN_FRAC
    STREAK_MAX_THICKNESS = cfg.STREAK_MAX_THICKNESS
    STREAK_MIN_ELONG = cfg.STREAK_MIN_ELONG
    noise = float(np.median(np.abs(diff)) * 1.4826)
    level = max(6.0, noise * STREAK_LEVEL_MULT)
    min_len = STREAK_MIN_LEN_FRAC * min_dim
    for axis in (0, 1):
        d = diff if axis == 0 else diff.T
        up = np.roll(d, STREAK_RIDGE_VGAP, axis=0)
        down = np.roll(d, -STREAK_RIDGE_VGAP, axis=0)
        ridge = d - np.maximum(up, down)
        resp = cv2.blur(ridge, (STREAK_INTEG_LEN, 1))
        binr = (resp > level).astype(np.uint8)
        nlab, labels, stats, _ = cv2.connectedComponentsWithStats(binr, 8)
        for lid in range(1, nlab):
            bw = stats[lid, cv2.CC_STAT_WIDTH]
            bh = stats[lid, cv2.CC_STAT_HEIGHT]
            if bw < min_len or bh > STREAK_MAX_THICKNESS or bw / max(bh, 1) < STREAK_MIN_ELONG:
                continue
            bx = stats[lid, cv2.CC_STAT_LEFT]
            by = stats[lid, cv2.CC_STAT_TOP]
            comp = labels[by:by + bh, bx:bx + bw] == lid
            if axis == 0:
                yield comp, bx, by
            else:                       # map back from the transposed frame
                yield comp.T, by, bx


def detect_radon_streaks(diff, local_std, cfg=None):
    """Find ultra-faint, fragmented, near-horizontal full-width scratches via Radon accumulation.

    Restrict the horizontal ridge response to SMOOTH regions (scene structure is excluded),
    then for a small range of slopes shear the response and take the per-line MEAN response and
    COVERAGE (fraction of the width with response present). A faint scratch — collinear across
    the whole frame even with big gaps — produces a far stronger line than any clean row.
    Returns a list of (path, x0, x1) for accepted streaks (path = full-image [x,y] key points).
    Conservative: only the strongest full-width lines pass (see STREAK_RADON_* constants).
    """
    cfg = DEFAULT_TUNING if cfg is None else cfg
    STROKE_MAX_CONTEXT_TEXTURE = cfg.STROKE_MAX_CONTEXT_TEXTURE
    STREAK_RIDGE_VGAP = cfg.STREAK_RIDGE_VGAP
    STREAK_INTEG_LEN = cfg.STREAK_INTEG_LEN
    STREAK_RADON_MIN_RESP = cfg.STREAK_RADON_MIN_RESP
    STREAK_RADON_MIN_COV = cfg.STREAK_RADON_MIN_COV
    STREAK_RADON_MAX_SLOPE = cfg.STREAK_RADON_MAX_SLOPE
    STREAK_RADON_SLOPES = cfg.STREAK_RADON_SLOPES
    STREAK_RADON_PRESENT = cfg.STREAK_RADON_PRESENT
    STREAK_RADON_EXT_FACTOR = cfg.STREAK_RADON_EXT_FACTOR
    STREAK_RADON_EXT_GAP = cfg.STREAK_RADON_EXT_GAP
    h, w = diff.shape
    up = np.roll(diff, STREAK_RIDGE_VGAP, axis=0)
    down = np.roll(diff, -STREAK_RIDGE_VGAP, axis=0)
    ridge = np.clip(diff - np.maximum(up, down), 0.0, None)
    ridge[local_std > STROKE_MAX_CONTEXT_TEXTURE] = 0.0   # smooth (sky) regions only

    xs = np.arange(w)
    best = None  # (mean_resp, slope, y0, coverage, line_response[w])
    for slope in np.linspace(-STREAK_RADON_MAX_SLOPE, STREAK_RADON_MAX_SLOPE, STREAK_RADON_SLOPES):
        # Line-sum for every intercept y0 at once: shift each column up by round(slope*x)
        # and add. acc[y0] = sum_x ridge[y0 + round(slope*x), x] (direct, unambiguous).
        base = np.round(slope * xs).astype(np.int64)
        acc = np.zeros(h)
        covcount = np.zeros(h)
        for off in np.unique(base):
            cols = xs[base == off]
            colsum = ridge[:, cols].sum(axis=1)
            covc = (ridge[:, cols] > STREAK_RADON_PRESENT).sum(axis=1).astype(np.float64)
            acc += np.roll(colsum, -int(off))
            covcount += np.roll(covc, -int(off))
        rowmean = acc / w
        yb = int(np.argmax(rowmean))
        if best is None or rowmean[yb] > best[0]:
            best = (float(rowmean[yb]), float(slope), yb, float(covcount[yb] / w))

    if best is None:
        return []
    mean_resp, slope, y0, coverage = best
    if mean_resp < STREAK_RADON_MIN_RESP or coverage < STREAK_RADON_MIN_COV:
        return []
    # The straight Radon line locates the scratch but can be a few px off (it over-tilts to
    # fit the strongest segment). Refine: track the actual response peak in a band around the
    # line at each sample x (keeping the line estimate across faint gaps), so the brush sits
    # ON the scratch even if it is slightly curved.
    hresp = cv2.blur(ridge, (STREAK_INTEG_LEN, 1))
    ys_line = (slope * xs + y0)
    present = np.where(hresp[np.clip(ys_line.astype(np.int64), 0, h - 1), xs] > STREAK_RADON_PRESENT)[0]
    if present.size < 2:
        return []
    x0, x1 = int(present[0]), int(present[-1])
    band = 16
    # Extend the endpoints outward while the scratch is still faintly present in the band
    # (a scratch fades at its ends below the detection threshold). Lower extent threshold +
    # gap tolerance; stop at a sustained gap or the image edge.
    ext_level = STREAK_RADON_PRESENT * STREAK_RADON_EXT_FACTOR

    def _band_present(x):
        ly = int(slope * x + y0)
        lo, hi = max(0, ly - band), min(h, ly + band)
        return hresp[lo:hi, x].size and hresp[lo:hi, x].max() > ext_level

    gap = 0
    while x1 < w - 1 and gap < STREAK_RADON_EXT_GAP:
        x1 += 1
        gap = 0 if _band_present(x1) else gap + 1
    x1 -= gap
    gap = 0
    while x0 > 0 and gap < STREAK_RADON_EXT_GAP:
        x0 -= 1
        gap = 0 if _band_present(x0) else gap + 1
    x0 += gap

    step = max(1, (x1 - x0) // 60)
    pts = []
    for x in range(x0, x1 + 1, step):
        ly = int(slope * x + y0)
        lo, hi = max(0, ly - band), min(h, ly + band)
        seg = hresp[lo:hi, x]
        ry = lo + int(np.argmax(seg)) if seg.size and seg.max() > ext_level else ly
        pts.append([float(x), float(ry)])
    # median-smooth the tracked y to suppress occasional jumps to noise
    if len(pts) >= 5:
        yy = np.array([p[1] for p in pts])
        k = 5
        ys_med = np.array([np.median(yy[max(0, i - k):i + k + 1]) for i in range(len(yy))])
        pts = [[pts[i][0], float(ys_med[i])] for i in range(len(pts))]
    return [(pts, x0, x1)]


def split_stroke_at_busy(spot, local_std, min_dim, cfg=None):
    """Split a stroke into the smooth-background runs of its path, dropping busy crossings.

    A long stroke (especially a full-width scratch) may pass through busy regions (rigging,
    funnels) where darktable's clone makes artifacts and the defect is nearly invisible
    anyway. Resample the path, mark points whose local texture exceeds HEAL_MAX_TEXTURE (plus
    a HEAL_BUSY_MARGIN_PX margin) as busy, and emit one sub-spot per contiguous smooth run at
    least HEAL_MIN_SEGMENT_FRAC*min_dim long. Returns the list of sub-spots (possibly just the
    original if it is entirely smooth, or empty if entirely busy).
    """
    cfg = DEFAULT_TUNING if cfg is None else cfg
    STROKE_DP_EPS_FRAC = cfg.STROKE_DP_EPS_FRAC
    HEAL_MAX_TEXTURE = cfg.HEAL_MAX_TEXTURE
    HEAL_BUSY_MARGIN_PX = cfg.HEAL_BUSY_MARGIN_PX
    HEAL_SAMPLE_STEP_PX = cfg.HEAL_SAMPLE_STEP_PX
    HEAL_MIN_SEGMENT_FRAC = cfg.HEAL_MIN_SEGMENT_FRAC
    path = np.array(spot["path"], dtype=np.float64)
    if len(path) < 2:
        return [spot]
    dense = []
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(seg / HEAL_SAMPLE_STEP_PX))
        for k in range(n):
            dense.append(a + (b - a) * (k / n))
    dense.append(path[-1])
    dense = np.array(dense)

    busy = np.zeros(len(dense), dtype=bool)
    for i, (x, y) in enumerate(dense):
        ix, iy = int(round(x)), int(round(y))
        win = local_std[max(0, iy - 3):iy + 4, max(0, ix - 3):ix + 4]
        busy[i] = win.size and float(win.max()) > HEAL_MAX_TEXTURE
    if not busy.any():
        return [spot]
    # widen busy runs by the margin (in sample units) so sources near the edge stay clean
    m = max(1, int(round(HEAL_BUSY_MARGIN_PX / HEAL_SAMPLE_STEP_PX)))
    busy = np.convolve(busy.astype(int), np.ones(2 * m + 1, dtype=int), mode="same") > 0

    min_seg = HEAL_MIN_SEGMENT_FRAC * min_dim
    eps = max(1.0, STROKE_DP_EPS_FRAC * min_dim)
    subs = []
    i = 0
    while i < len(dense):
        if busy[i]:
            i += 1
            continue
        j = i
        while j < len(dense) and not busy[j]:
            j += 1
        run = dense[i:j]
        rlen = float(sum(math.hypot(run[t + 1][0] - run[t][0], run[t + 1][1] - run[t][1])
                         for t in range(len(run) - 1)))
        if rlen >= min_seg and len(run) >= 2:
            simp = cv2.approxPolyDP(run.astype(np.float32).reshape(-1, 1, 2), eps, False).reshape(-1, 2)
            simp = simp if len(simp) >= 2 else run[[0, -1]]
            sub = dict(spot)
            sub["path"] = [[float(p[0]), float(p[1])] for p in simp]
            mid = simp[len(simp) // 2]
            sub["cx"], sub["cy"] = float(mid[0]), float(mid[1])
            sub["length_px"] = rlen
            subs.append(sub)
        i = j
    return subs


def _make_radon_streak_spot(path_full, diff, min_dim, cfg=None):
    """Build a stroke spot dict directly for a Radon-detected full-width scratch (it bypasses
    build_stroke_spot because it is, by design, too faint per-point). Brush width comes from
    the measured visible half-width of the (faint) line, floored so it still covers the streak.
    """
    cfg = DEFAULT_TUNING if cfg is None else cfg
    STROKE_RIDGE_MIN_CONTRAST = cfg.STROKE_RIDGE_MIN_CONTRAST
    STREAK_MIN_LEN_FRAC = cfg.STREAK_MIN_LEN_FRAC
    STREAK_RADON_MAX_HALFWIDTH = cfg.STREAK_RADON_MAX_HALFWIDTH
    STREAK_RADON_MIN_BRUSH_R = cfg.STREAK_RADON_MIN_BRUSH_R
    STREAK_RADON_BRUSH_MARGIN = cfg.STREAK_RADON_BRUSH_MARGIN
    pts = np.array(path_full, dtype=np.float64)
    length = float(sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
                       for i in range(len(pts) - 1)))
    if length < STREAK_MIN_LEN_FRAC * min_dim:
        return None
    # A faint scratch varies in thickness; the brush must be BOLD enough to fully clear the
    # thickest pieces in darktable (over-heal is on smooth sky → invisible). Cap the (noisy)
    # measurement, then floor with a bold minimum and a generous margin.
    cover_hw = min(stroke_coverage_halfwidth(path_full, diff, cfg=cfg), STREAK_RADON_MAX_HALFWIDTH)
    brush_radius_px = max(STREAK_RADON_MIN_BRUSH_R, cover_hw + STREAK_RADON_BRUSH_MARGIN)
    width_px = max(3.0, 2.0 * cover_hw)
    mid = path_full[len(path_full) // 2]
    return {
        "kind": "stroke",
        "path": [[float(p[0]), float(p[1])] for p in path_full],
        "stroke_width_px": float(width_px),
        "length_px": length,
        "cx": float(mid[0]),
        "cy": float(mid[1]),
        "radius_px": float(width_px / 2.0),
        "brush_radius_px": float(brush_radius_px),
        "threshold": float(STROKE_RIDGE_MIN_CONTRAST),
        "area": int(length * width_px),
        "contrast": 0.0,
        "texture": 0.0,
        "context_texture": 0.0,
        "crispness": 0.0,
        "excess_sat": 0.0,
        "spot_sat": 0.0,
        "source": "scratch",
    }


def detect_scratch_ridges(diff, min_dim, cfg=None):
    """Ridge pass for faint scratches/threads using a Hessian bright-ridge filter.

    Runs skimage.filters.sato on a downscaled background-subtracted image (full-res
    is too slow), thresholds the response, and returns connected curvilinear masks
    mapped back to full resolution as (component_mask_bool, x_off, y_off) tuples.
    """
    cfg = DEFAULT_TUNING if cfg is None else cfg
    STROKE_MIN_LENGTH_FRAC = cfg.STROKE_MIN_LENGTH_FRAC
    STROKE_RIDGE_ENABLE = cfg.STROKE_RIDGE_ENABLE
    STROKE_RIDGE_SIGMAS = cfg.STROKE_RIDGE_SIGMAS
    STROKE_RIDGE_Z = cfg.STROKE_RIDGE_Z
    STROKE_RIDGE_DOWNSCALE = cfg.STROKE_RIDGE_DOWNSCALE
    STROKE_RIDGE_MAX_CANDIDATES = cfg.STROKE_RIDGE_MAX_CANDIDATES
    if not STROKE_RIDGE_ENABLE:
        return []
    s = STROKE_RIDGE_DOWNSCALE
    h, w = diff.shape
    small = cv2.resize(diff, (max(1, int(w * s)), max(1, int(h * s))),
                       interpolation=cv2.INTER_AREA)
    small = np.clip(small, 0, None).astype(np.float32)
    # Bright ridges -> black_ridges=False. Sigmas scaled to the downscaled image.
    sigmas = [max(1.0, sig * s * 4) for sig in STROKE_RIDGE_SIGMAS]
    resp = sato(small, sigmas=sigmas, black_ridges=False)
    if resp.max() <= 0:
        return []
    # Robust z-score threshold: median + Z*MAD. Using MAD (not global max) prevents a
    # single very-bright structure from drowning out faint ridges elsewhere.
    med = float(np.median(resp))
    mad = float(np.median(np.abs(resp - med))) * 1.4826 + 1e-6
    ridge_bin = (resp >= med + STROKE_RIDGE_Z * mad).astype(np.uint8)
    if ridge_bin.sum() == 0:
        return []
    # Upscale binary mask back to full resolution.
    ridge_full = cv2.resize(ridge_bin, (w, h), interpolation=cv2.INTER_NEAREST)
    # Bridge tiny gaps along thin ridges.
    ridge_full = cv2.morphologyEx(
        ridge_full, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    num, labels, stats, _ = cv2.connectedComponentsWithStats(ridge_full, connectivity=8)
    min_area = int((STROKE_MIN_LENGTH_FRAC * min_dim))  # at least ~length px (thin -> area~length)
    cands = []
    for lid in range(1, num):
        area = stats[lid, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        cands.append((area, lid))
    # Process the largest candidates first; cap to bound skeletonization cost.
    cands.sort(reverse=True)
    out = []
    for area, lid in cands[:STROKE_RIDGE_MAX_CANDIDATES]:
        bx = stats[lid, cv2.CC_STAT_LEFT]
        by = stats[lid, cv2.CC_STAT_TOP]
        bw = stats[lid, cv2.CC_STAT_WIDTH]
        bh = stats[lid, cv2.CC_STAT_HEIGHT]
        comp = labels[by:by + bh, bx:bx + bw] == lid
        out.append((comp, bx, by))
    return out


def _stroke_overlaps_existing(spot, existing, min_sep):
    """True if the stroke's midpoint is within min_sep px of any existing spot center."""
    for e in existing:
        if math.hypot(spot["cx"] - e["cx"], spot["cy"] - e["cy"]) < min_sep:
            return True
    return False


# ===================================================================
# ML detection helpers
# ===================================================================

def _compute_candidate_features(label_id, labels, stats, centroids,
                                 diff, gray, local_bg, bg_gradient, local_std, saturation,
                                 threshold, bright_ref, width, height, cfg=None):
    """Compute all features for a single connected component without soft-filter decisions.

    Applies only hard gates (size 6-800, shape aspect/compactness/solidity/circularity,
    minimum contrast floor, excess saturation).  All soft filters (texture, ratio, context,
    votes, dim, embedded, edge, dark_bg) are intentionally skipped so the ML classifier
    can make those decisions.

    Returns a feature dict (keys = FEATURE_NAMES + cx/cy/area/radius_px/brush_radius_px/
    threshold/contrast/excess_sat/spot_sat/context_texture/texture), or None on hard-filter
    rejection.
    """
    cfg = DEFAULT_TUNING if cfg is None else cfg
    MIN_SPOT_AREA = cfg.MIN_SPOT_AREA
    MAX_SPOT_AREA = cfg.MAX_SPOT_AREA
    MIN_ASPECT_RATIO = cfg.MIN_ASPECT_RATIO
    MIN_COMPACTNESS = cfg.MIN_COMPACTNESS
    MIN_SOLIDITY = cfg.MIN_SOLIDITY
    MIN_CIRCULARITY = cfg.MIN_CIRCULARITY
    SHAPE_CHECK_MIN_AREA = cfg.SHAPE_CHECK_MIN_AREA
    TEXTURE_KERNEL = cfg.TEXTURE_KERNEL
    MAX_EXCESS_SATURATION = cfg.MAX_EXCESS_SATURATION
    MAX_SPOT_SATURATION = cfg.MAX_SPOT_SATURATION
    EMULSION_EXCESS_SAT_THRESHOLD = cfg.EMULSION_EXCESS_SAT_THRESHOLD
    SOFT_CONTEXT_VOTE_THRESHOLD = cfg.SOFT_CONTEXT_VOTE_THRESHOLD
    SOFT_TEXTURE_VOTE_THRESHOLD = cfg.SOFT_TEXTURE_VOTE_THRESHOLD
    SOFT_RATIO_VOTE_THRESHOLD = cfg.SOFT_RATIO_VOTE_THRESHOLD
    MIN_LOCAL_BG_FRACTION = cfg.MIN_LOCAL_BG_FRACTION
    ENC_RADIUS_SCALE = cfg.ENC_RADIUS_SCALE
    area = stats[label_id, cv2.CC_STAT_AREA]
    w = stats[label_id, cv2.CC_STAT_WIDTH]
    h = stats[label_id, cv2.CC_STAT_HEIGHT]

    if area < MIN_SPOT_AREA or area > MAX_SPOT_AREA:
        return None

    aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0.0
    if aspect_ratio < MIN_ASPECT_RATIO:
        return None

    bbox_area = w * h
    compactness = area / bbox_area if bbox_area > 0 else 0.0
    if area < SHAPE_CHECK_MIN_AREA and compactness < MIN_COMPACTNESS:
        return None

    cx, cy = centroids[label_id]
    bx = stats[label_id, cv2.CC_STAT_LEFT]
    by = stats[label_id, cv2.CC_STAT_TOP]
    component_mask = labels[by:by+h, bx:bx+w] == label_id
    component_diff = diff[by:by+h, bx:bx+w]
    contrast = float(np.max(component_diff[component_mask]))

    # Very generous floor — let the ML decide on borderline contrast
    if contrast < threshold * 0.3:
        return None

    # Shape checks for large spots (hard: blob shape is a prerequisite)
    enc_r = math.sqrt(area / math.pi)
    if area >= SHAPE_CHECK_MIN_AREA:
        mask_u8 = component_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            hull = cv2.convexHull(contours[0])
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 1.0
            if solidity < MIN_SOLIDITY:
                return None
            perimeter = cv2.arcLength(contours[0], closed=True)
            if perimeter > 0:
                circularity = 4 * math.pi * area / (perimeter * perimeter)
                if circularity < MIN_CIRCULARITY:
                    return None
            _, enc_r_c = cv2.minEnclosingCircle(contours[0])
            enc_r = float(enc_r_c)
    else:
        mask_u8_enc = component_mask.astype(np.uint8) * 255
        enc_contours, _ = cv2.findContours(mask_u8_enc, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
        if enc_contours:
            _, enc_r_c = cv2.minEnclosingCircle(enc_contours[0])
            enc_r = float(enc_r_c)

    icx = max(0, min(int(round(cx)), width - 1))
    icy = max(0, min(int(round(cy)), height - 1))
    local_bg_brightness = float(local_bg[icy, icx])
    is_dark_bg = local_bg_brightness < bright_ref * MIN_LOCAL_BG_FRACTION

    local_bg_grad = float(bg_gradient[icy, icx])
    grad_ratio = min(local_bg_grad / contrast if contrast > 0 else 999.0, 5.0)

    # Surrounding ring for texture and saturation
    ring_inner = max(int(enc_r * 1.5), 2)
    ring_outer = max(int(enc_r * 3), ring_inner + TEXTURE_KERNEL)
    y1 = max(0, icy - ring_outer)
    y2 = min(height, icy + ring_outer + 1)
    x1 = max(0, icx - ring_outer)
    x2 = min(width, icx + ring_outer + 1)
    patch_std = local_std[y1:y2, x1:x2]
    py_o, px_o = np.ogrid[y1-icy:y2-icy, x1-icx:x2-icx]
    dist_sq = px_o*px_o + py_o*py_o
    ring_mask = (dist_sq >= ring_inner**2) & (dist_sq <= ring_outer**2)
    local_texture = (float(np.median(patch_std[ring_mask]))
                     if ring_mask.any() else float(local_std[icy, icx]))

    # Context texture (200px radius excluding the immediate ring)
    ctx_r = 200
    ctx_y1 = max(0, icy - ctx_r); ctx_y2 = min(height, icy + ctx_r + 1)
    ctx_x1 = max(0, icx - ctx_r); ctx_x2 = min(width, icx + ctx_r + 1)
    ctx_patch = local_std[ctx_y1:ctx_y2, ctx_x1:ctx_x2]
    ctx_py, ctx_px = np.ogrid[ctx_y1 - icy:ctx_y2 - icy, ctx_x1 - icx:ctx_x2 - icx]
    ctx_dist_sq = ctx_px * ctx_px + ctx_py * ctx_py
    ctx_excl_r = ring_outer + TEXTURE_KERNEL // 2
    ctx_mask = (ctx_dist_sq <= ctx_r ** 2) & (ctx_dist_sq > ctx_excl_r ** 2)
    context_texture = (float(np.median(ctx_patch[ctx_mask]))
                       if ctx_mask.any() else float(np.median(ctx_patch)))

    # Surround ratio (embedded-window check), 1.0 on dark backgrounds
    radius_px = math.sqrt(area / math.pi)
    if not is_dark_bg:
        check_r = max(int(radius_px * 3), 8)
        ny1 = max(0, icy - check_r); ny2 = min(height, icy + check_r + 1)
        nx1 = max(0, icx - check_r); nx2 = min(width, icx + check_r + 1)
        neighborhood = gray[ny1:ny2, nx1:nx2]
        local_low = float(np.percentile(neighborhood, 5))
        surround_ratio = local_low / local_bg_brightness if local_bg_brightness > 0 else 1.0
    else:
        surround_ratio = 1.0

    # Saturation
    component_sat = saturation[by:by+h, bx:bx+w]
    spot_sat = float(np.mean(component_sat[component_mask]))
    patch_sat = saturation[y1:y2, x1:x2]
    surround_sat = float(np.median(patch_sat[ring_mask])) if ring_mask.any() else spot_sat
    excess_sat = spot_sat - surround_sat

    # Hard color filter: clearly colored features are never dust
    if excess_sat > MAX_EXCESS_SATURATION:
        return None
    if spot_sat > MAX_SPOT_SATURATION and excess_sat > EMULSION_EXCESS_SAT_THRESHOLD:
        return None

    # Soft voting signals (kept as a numeric feature)
    dust_votes = 0
    if context_texture < SOFT_CONTEXT_VOTE_THRESHOLD:
        dust_votes += 1
    if local_texture < SOFT_TEXTURE_VOTE_THRESHOLD:
        dust_votes += 1
    if local_texture > 0 and contrast / local_texture > SOFT_RATIO_VOTE_THRESHOLD:
        dust_votes += 1

    brush_radius_px = max(MIN_BRUSH_PX, enc_r * ENC_RADIUS_SCALE)
    return {
        # Spot-dict fields expected by the rest of the pipeline
        "cx": float(cx), "cy": float(cy),
        "area": area,
        "radius_px": enc_r,
        "brush_radius_px": brush_radius_px,
        "threshold": threshold,
        "contrast": contrast,
        "texture": local_texture,
        "excess_sat": excess_sat,
        "spot_sat": spot_sat,
        "context_texture": context_texture,
        # ML feature fields (matching FEATURE_NAMES)
        "contrast_ratio": contrast / threshold,
        "log_area": math.log1p(area),
        "contrast_texture_ratio": contrast / max(local_texture, 0.1),
        "local_bg_brightness": local_bg_brightness,
        "local_bg_frac": local_bg_brightness / max(bright_ref, 1.0),
        "is_dark_bg": float(is_dark_bg),
        "grad_ratio": grad_ratio,
        "aspect_ratio": aspect_ratio,
        "compactness": compactness,
        "surround_ratio": surround_ratio,
        "dust_votes": float(dust_votes),
    }


def _collect_candidates_with_features(gray, diff, local_bg, bg_gradient, local_std, saturation,
                                       threshold, bright_ref, width, height):
    """Threshold image and extract full feature vectors for all candidates.

    Applies hard filters only (size, shape, color).  Skips all soft filters so
    the ML classifier can make final accept/reject decisions.

    Returns list of candidate dicts (all FEATURE_NAMES fields present).
    """
    binary = (diff > threshold).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    candidates = []
    for label_id in range(1, num_labels):
        feat = _compute_candidate_features(
            label_id, labels, stats, centroids,
            diff, gray, local_bg, bg_gradient, local_std, saturation,
            threshold, bright_ref, width, height)
        if feat is not None:
            candidates.append(feat)
    return candidates


def _spot_to_features(spot):
    """Extract the SPOT_FEATURE_NAMES vector from a standard spot dict.

    Works on spot dicts produced by detect_dust_spots() and by
    _compute_candidate_features() — both contain the required keys.
    """
    contrast = spot["contrast"]
    threshold = spot.get("threshold", 1.0)
    texture = spot["texture"]
    return [
        contrast,
        contrast / threshold if threshold > 0 else 0.0,
        math.log1p(spot["area"]),
        spot["radius_px"],
        texture,
        spot["context_texture"],
        contrast / max(texture, 0.1),
        spot["excess_sat"],
        spot["spot_sat"],
    ]


def detect_dust_spots_ml(image_path, ml_model, scaler, collect_rejects=False, cfg=None):
    """ML-assisted dust detection (post-filter mode).

    Strategy:
      1. Run the standard rule-based pipeline (detect_dust_spots) to get high-precision
         accepted spots.  This handles the bulk of detection and filtering.
      2. Apply the ML post-filter to those accepted spots to remove false positives
         that slipped through the rule-based filters.  The ML model was trained on
         user-annotated FPs (label=0) and confirmed dust (label=1).

    Returns (spots, rejected_candidates, error_msg, local_std) — same signature as
    detect_dust_spots(cfg=cfg).  Healing source positions are NOT set here;
    process_one_image(cfg=cfg) adds them.
    """
    cfg = DEFAULT_TUNING if cfg is None else cfg
    ML_POSTFILTER_THRESHOLD = cfg.ML_POSTFILTER_THRESHOLD
    # --- Phase 1: standard rule-based detection ---
    std_spots, std_rejects, error, local_std = detect_dust_spots(image_path, collect_rejects, cfg=cfg)
    if error:
        return None, std_rejects, error, local_std
    if std_spots is None:
        std_spots = []

    # --- Phase 2: ML post-filter on accepted spots ---
    # Strokes (threads/scratches) bypass the ML post-filter: the model was trained
    # on circular-dust features only, so it cannot judge elongated forms.
    dot_spots = [s for s in std_spots if s.get("kind") != "stroke"]
    stroke_spots = [s for s in std_spots if s.get("kind") == "stroke"]
    if dot_spots:
        X_std = np.array([_spot_to_features(s) for s in dot_spots], dtype=np.float32)
        X_std_scaled = scaler.transform(X_std)
        probas_std = ml_model.predict_proba(X_std_scaled)[:, 1]
        kept = [s for s, p in zip(dot_spots, probas_std) if p >= ML_POSTFILTER_THRESHOLD]
        n_removed = len(dot_spots) - len(kept)
        if n_removed > 0:
            print(f"  [ML] post-filter: removed {n_removed}/{len(dot_spots)} FPs "
                  f"(threshold={ML_POSTFILTER_THRESHOLD})")
        spots = kept + stroke_spots
    else:
        spots = stroke_spots

    return spots, std_rejects, None, local_std


# Default ML model path: dust_ml_model.pkl next to this script
_DEFAULT_ML_MODEL_PATH = str(Path(__file__).parent / "dust_ml_model.pkl")


def load_ml_model(path=None):
    """Load ML model bundle from a .pkl file.

    Returns (model, scaler) or (None, None) if the file doesn't exist.
    """
    import pickle
    path = path or _DEFAULT_ML_MODEL_PATH
    if not os.path.isfile(path):
        return None, None
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["scaler"]


def detect(image_path, collect_rejects=False, ml_model_path=None, cfg=None):
    """Unified detection entry point — uses ML post-filter by default.

    If ml_model_path is given, loads that model.  Otherwise loads the default
    dust_ml_model.pkl next to this script.  Falls back to pure rule-based
    detection when no model file exists. `cfg` (a tuning.Tuning) overrides the
    detection constants for a calibration trial (default = DEFAULT_TUNING).

    Returns (spots, rejected_candidates, error_msg, local_std).
    """
    model, scaler = load_ml_model(ml_model_path)
    if model is not None:
        return detect_dust_spots_ml(image_path, model, scaler,
                                    collect_rejects=collect_rejects, cfg=cfg)
    return detect_dust_spots(image_path, collect_rejects=collect_rejects, cfg=cfg)


# ===================================================================
# Binary data generation for darktable XMP
# ===================================================================

class MaskIdGenerator:
    """Generate unique mask IDs for darktable masks."""

    def __init__(self):
        self._base = int(time.time())
        self._counter = random.randint(1000, 9999)

    def next_id(self):
        mask_id = self._base + self._counter
        self._counter += 1
        return mask_id


def dt_xmp_encode(raw_bytes):
    """Encode binary data the way darktable stores it in XMP.

    Format: "gz" + 2-digit compression ratio + base64(zlib_compressed).
    """
    compressed = zlib.compress(raw_bytes)
    ratio = min(len(raw_bytes) // max(len(compressed), 1) + 1, 99)
    b64 = base64.b64encode(compressed).decode("ascii")
    return f"gz{ratio:02d}{b64}"


def _decode_blendop_template():
    """Decode the known-good blendop_params template to raw bytes."""
    encoded = BLENDOP_TEMPLATE_ENCODED
    # Strip "gz" prefix + 2-digit ratio
    b64_data = encoded[4:]
    compressed = base64.b64decode(b64_data)
    return zlib.decompress(compressed)


def make_brush_mask_points(cx, cy, border_radius):
    """Generate mask_points hex for a dot-shaped brush mask (2 points, 88 bytes).

    cx, cy: normalized [0,1] position of the dust spot.
    border_radius: brush radius in normalized coords.
    """
    points = bytearray()
    for i in range(2):
        px = cx + i * BRUSH_DELTA
        py = cy + i * BRUSH_DELTA
        # corner
        points += struct.pack("<ff", px, py)
        # ctrl1 (slightly before corner)
        points += struct.pack("<ff", px - BRUSH_CTRL_OFFSET, py - BRUSH_CTRL_OFFSET)
        # ctrl2 (slightly after corner)
        points += struct.pack("<ff", px + BRUSH_CTRL_OFFSET, py + BRUSH_CTRL_OFFSET)
        # border
        points += struct.pack("<ff", border_radius, border_radius)
        # density, hardness, state
        points += struct.pack("<ffI", BRUSH_DENSITY, BRUSH_HARDNESS, BRUSH_STATE_NORMAL)

    return bytes(points).hex()


def make_stroke_mask_points(points_norm, border):
    """Generate mask_points hex for a multi-node brush stroke (44 bytes per node).

    points_norm: list of (x, y) normalized [0,1] key points along the centerline.
    border: brush border (half-width) in normalized coords, applied to every node.
    Bezier control handles use Catmull-Rom tangents so darktable renders a smooth
    spline through the key points. Returns the hex string; node count = len(points).
    """
    pts = [(float(x), float(y)) for x, y in points_norm]
    n = len(pts)
    if n == 1:
        return make_brush_mask_points(pts[0][0], pts[0][1], border)

    data = bytearray()
    for i in range(n):
        px, py = pts[i]
        prev = pts[i - 1] if i > 0 else pts[i]
        nxt = pts[i + 1] if i < n - 1 else pts[i]
        tx = (nxt[0] - prev[0]) / 6.0
        ty = (nxt[1] - prev[1]) / 6.0
        if tx == 0.0 and ty == 0.0:
            tx = ty = BRUSH_CTRL_OFFSET  # avoid zero-length control handles
        # corner
        data += struct.pack("<ff", px, py)
        # ctrl1 (incoming handle)
        data += struct.pack("<ff", px - tx, py - ty)
        # ctrl2 (outgoing handle)
        data += struct.pack("<ff", px + tx, py + ty)
        # border (half-width)
        data += struct.pack("<ff", border, border)
        # density, hardness, state
        data += struct.pack("<ffI", BRUSH_DENSITY, BRUSH_HARDNESS, BRUSH_STATE_NORMAL)

    return bytes(data).hex()


def parse_transform_params(params_file):
    """Read transform_params.txt written by Lua.

    Returns dict: filename -> {"flip": int, "crop": (L, T, R, B), "ashift": dict|None}.
    Format: filename|flip=N|crop=L,T,R,B[|ashift=<gz16-base64-without-prefix>]
    """
    transforms = {}
    if not os.path.isfile(params_file):
        return transforms

    with open(params_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            filename = parts[0]
            flip = 0
            crop = (0.0, 0.0, 1.0, 1.0)
            ashift = None
            for part in parts[1:]:
                if part.startswith("flip="):
                    flip = int(part[5:])
                elif part.startswith("crop="):
                    vals = part[5:].split(",")
                    if len(vals) == 4:
                        crop = tuple(float(v) for v in vals)
                elif part.startswith("ashift="):
                    ashift = _decode_ashift_params(part[7:])
            transforms[filename] = {"flip": flip, "crop": crop, "ashift": ashift}

    return transforms


def make_group_mask_points(brush_ids, group_id):
    """Generate mask_points hex for a group mask referencing all brushes.

    Each entry: formid(int32), parentid(int32), state(int32), opacity(float32) = 16 bytes.
    """
    data = bytearray()
    for i, brush_id in enumerate(brush_ids):
        state = 3 if i == 0 else 11  # USE|SHOW=3, USE|SHOW|UNION=11
        data += struct.pack("<iiif", brush_id, group_id, state, 1.0)
    return bytes(data).hex()


def make_retouch_params(form_ids):
    """Build the retouch module params blob (13260 bytes), compress and encode.

    form_ids: list of mask_ids for each brush form.
    """
    if len(form_ids) > MAX_FORMS:
        raise ValueError(f"Too many forms: {len(form_ids)} > {MAX_FORMS}")

    # 300 form entries, 44 bytes each
    forms_data = bytearray()
    for i in range(MAX_FORMS):
        if i < len(form_ids):
            entry = struct.pack("<i", form_ids[i])    # formid
            entry += struct.pack("<i", 0)              # scale
            entry += struct.pack("<i", RETOUCH_ALGO_HEAL)  # algorithm
            entry += struct.pack("<i", 0)              # blur_type
            entry += struct.pack("<f", 0.0)            # blur_radius
            entry += struct.pack("<i", 0)              # fill_mode
            entry += struct.pack("<fff", 0.0, 0.0, 0.0)  # fill_color[3]
            entry += struct.pack("<f", 0.0)            # fill_brightness
            entry += struct.pack("<i", 2)              # distort_mode
        else:
            entry = b"\x00" * 44
        forms_data += entry

    # Global params (60 bytes)
    global_params = struct.pack("<i", RETOUCH_ALGO_HEAL)      # algorithm
    global_params += struct.pack("<i", 0)                      # num_scales
    global_params += struct.pack("<i", 0)                      # curr_scale
    global_params += struct.pack("<i", 0)                      # merge_from_scale
    global_params += struct.pack("<fff", -3.0, 0.0, 3.0)      # preview_levels[3]
    global_params += struct.pack("<i", 0)                      # blur_type
    global_params += struct.pack("<f", 10.0)                   # blur_radius
    global_params += struct.pack("<i", 0)                      # fill_mode
    global_params += struct.pack("<fff", 0.0, 0.0, 0.0)       # fill_color[3]
    global_params += struct.pack("<f", 0.0)                    # fill_brightness
    global_params += struct.pack("<i", 2000)                   # max_heal_iter

    raw = bytes(forms_data) + global_params
    assert len(raw) == 13260, f"Expected 13260 bytes, got {len(raw)}"
    return dt_xmp_encode(raw)


def make_blendop_params(group_mask_id):
    """Build blendop params by patching the template with the new group mask_id."""
    raw = bytearray(_decode_blendop_template())
    # Replace mask_id at offset 24 (4 bytes, little-endian int32)
    struct.pack_into("<i", raw, 24, group_mask_id)
    return dt_xmp_encode(bytes(raw))


def _decode_ashift_params(b64_str):
    """Decode a darktable gz16 base64-encoded ashift params blob.

    Returns dict with keys: rotation, lensshift_v, lensshift_h, shear,
    f_length_kb, orthocorr, aspect, cl, cr, ct, cb.
    Returns None on any failure (treat as identity transform).
    """
    try:
        raw = zlib.decompress(base64.b64decode(b64_str))
        if len(raw) < 56:
            return None
        return {
            "rotation":    struct.unpack_from("<f", raw,  0)[0],
            "lensshift_v": struct.unpack_from("<f", raw,  4)[0],
            "lensshift_h": struct.unpack_from("<f", raw,  8)[0],
            "shear":       struct.unpack_from("<f", raw, 12)[0],
            "f_length_kb": struct.unpack_from("<f", raw, 16)[0],
            # offset 20: crop_factor (unused here)
            "orthocorr":   struct.unpack_from("<f", raw, 24)[0],
            "aspect":      struct.unpack_from("<f", raw, 28)[0],
            # offset 32: mode (int32), offset 36: toggle (int32)
            "cl":          struct.unpack_from("<f", raw, 40)[0],
            "cr":          struct.unpack_from("<f", raw, 44)[0],
            "ct":          struct.unpack_from("<f", raw, 48)[0],
            "cb":          struct.unpack_from("<f", raw, 52)[0],
        }
    except Exception:
        return None


def _compute_ashift_homography(rotation_deg, lensshift_v, lensshift_h, shear,
                                f_length_kb, orthocorr, aspect, width, height):
    """Compute the forward 3x3 ashift homography matrix in pixel coordinates.

    Implements darktable's _homography() function (10-step pipeline).
    width, height: ashift module input buffer dimensions (buf_in).
    Returns H such that (x_out*w, y_out*w, w) = H @ (x_in, y_in, 1).
    """
    u = float(width)
    v = float(height)
    rot_rad = math.radians(rotation_deg)
    cosi = math.cos(rot_rad)
    sini = math.sin(rot_rad)
    horifac = 1.0 - orthocorr / 100.0
    ascale = math.sqrt(max(1e-6, aspect))

    # Vertical shift params
    exppa_v = math.exp(lensshift_v)
    fdb_v = f_length_kb / (14.4 + (v / u - 1.0) * 7.2)
    rad_v = fdb_v * (exppa_v - 1.0) / (exppa_v + 1.0)
    alpha_v = max(-1.5, min(1.5, math.atan(rad_v)))
    rt_v = math.sin(0.5 * alpha_v)
    r_v = max(0.1, 2.0 * (horifac - 1.0) * rt_v * rt_v + 1.0)

    # Horizontal shift params
    exppa_h = math.exp(lensshift_h)
    fdb_h = f_length_kb / (14.4 + (u / v - 1.0) * 7.2)
    rad_h = fdb_h * (exppa_h - 1.0) / (exppa_h + 1.0)
    alpha_h = max(-1.5, min(1.5, math.atan(rad_h)))
    rt_h = math.sin(0.5 * alpha_h)
    r_h = max(0.1, 2.0 * (horifac - 1.0) * rt_h * rt_h + 1.0)

    def mat(r0, r1, r2):
        return np.array([r0, r1, r2], dtype=np.float64)

    # Step 1: flip x <-> y
    M = mat([0, 1, 0], [1, 0, 0], [0, 0, 1])
    # Step 2: rotation around centre (v/2, u/2) in flipped space
    tx = -0.5*v*cosi + 0.5*u*sini + 0.5*v
    ty = -0.5*v*sini - 0.5*u*cosi + 0.5*u
    M = mat([cosi, -sini, tx], [sini, cosi, ty], [0, 0, 1]) @ M
    # Step 3: shear
    M = mat([1, shear, 0], [shear, 1, 0], [0, 0, 1]) @ M
    # Step 4: vertical lens shift (in flipped space, perspective along x = y_orig)
    M = mat(
        [exppa_v, 0, 0],
        [0.5*(exppa_v-1)*u/v,  2*exppa_v/(exppa_v+1),  -0.5*(exppa_v-1)*u/(exppa_v+1)],
        [(exppa_v-1)/v, 0, 1],
    ) @ M
    # Step 5: horizontal compression in flipped space (y = x_orig, range u)
    M = mat([1, 0, 0], [0, r_v, 0.5*u*(1-r_v)], [0, 0, 1]) @ M
    # Step 6: flip x <-> y back
    M = mat([0, 1, 0], [1, 0, 0], [0, 0, 1]) @ M
    # Step 7: horizontal lens shift (in original space, perspective along x, range u)
    M = mat(
        [exppa_h, 0, 0],
        [0.5*(exppa_h-1)*v/u,  2*exppa_h/(exppa_h+1),  -0.5*(exppa_h-1)*v/(exppa_h+1)],
        [(exppa_h-1)/u, 0, 1],
    ) @ M
    # Step 8: vertical compression in original space (y = y_orig, range v)
    M = mat([1, 0, 0], [0, r_h, 0.5*v*(1-r_h)], [0, 0, 1]) @ M
    # Step 9: aspect ratio
    M = mat([ascale, 0, 0], [0, 1.0/ascale, 0], [0, 0, 1]) @ M
    # Step 10: translate so minimum corner coordinate is at (0, 0)
    corners = np.array([[0, 0, 1], [u, 0, 1], [0, v, 1], [u, v, 1]], dtype=np.float64).T
    p = M @ corners
    min_x = (p[0] / p[2]).min()
    min_y = (p[1] / p[2]).min()
    M = mat([1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]) @ M
    return M


def _ashift_output_bbox(params, W_in, H_in):
    """Pixel width/height of the homography's output bounding box for an input of W_in x H_in."""
    H = _compute_ashift_homography(
        params["rotation"], params["lensshift_v"], params["lensshift_h"],
        params["shear"], params["f_length_kb"], params["orthocorr"],
        params["aspect"], W_in, H_in)
    corners = np.array([[0, 0, 1], [W_in, 0, 1], [0, H_in, 1], [W_in, H_in, 1]], dtype=np.float64).T
    q = H @ corners
    xs, ys = q[0] / q[2], q[1] / q[2]
    return float(xs.max() - xs.min()), float(ys.max() - ys.min())


def _solve_ashift_input_dims(params, fullwidth, fullheight, iters=60):
    """Find the ashift INPUT buffer dims whose homography output bbox equals (fullwidth, fullheight).

    fullwidth x fullheight is the exact full output size (derived from export dims + crop +
    internal ashift crop). The homography EXPANDS the bounding box (rotation/perspective), so the
    input is smaller than its own output bbox — using fullwidth/fullheight directly as the input
    (the previous approximation) mis-scales the homography and shifts every mask by ~0.3-0.8%
    (≈10-18px at full res on a 0.3° straighten — visible for small dust). The homography is
    scale-equivariant, so bbox scales with input; fixed-point iterate input *= target/bbox until
    the bbox matches (residual ~0px within a few steps for darktable's small corrections).
    """
    W_in, H_in = float(fullwidth), float(fullheight)
    for _ in range(iters):
        bw, bh = _ashift_output_bbox(params, W_in, H_in)
        nw = W_in * fullwidth / max(bw, 1e-6)
        nh = H_in * fullheight / max(bh, 1e-6)
        if abs(nw - W_in) < 1e-4 and abs(nh - H_in) < 1e-4:
            return nw, nh
        W_in, H_in = nw, nh
    return W_in, H_in


def _undo_ashift(cx, cy, ashift_params, export_w, export_h, crop, flip):
    """Invert the darktable ashift module's coordinate transform.

    cx, cy: normalized [0,1] coords in post-flip / ashift-output space.
    ashift_params: dict from _decode_ashift_params().
    Returns (cx, cy) in pre-ashift / raw-sensor normalized [0,1] space.

    The forward pipeline is: raw -> ashift -> flip -> crop -> export.
    After undoing crop and flip we are in ashift-output space.
    This function maps those coords back through ashift to raw space.
    """
    crop_l, crop_t, crop_r, crop_b = crop

    # Ashift output dimensions (pixel space, in landscape/sensor orientation).
    # We know the export dimensions and crop; undoing crop gives post-flip size.
    # Undoing SWAP_XY (if any) gives ashift output size.
    w_post_flip = export_w / max(crop_r - crop_l, 1e-6)
    h_post_flip = export_h / max(crop_b - crop_t, 1e-6)
    if flip & 4:   # SWAP_XY was applied: portrait image from landscape sensor
        W_out = h_post_flip
        H_out = w_post_flip
    else:
        W_out = w_post_flip
        H_out = h_post_flip

    cl = ashift_params["cl"]
    cr = ashift_params["cr"]
    ct = ashift_params["ct"]
    cb = ashift_params["cb"]

    # "full" output: the homography output before the ashift internal crop
    fullwidth  = W_out / max(cr - cl, 1e-6)
    fullheight = H_out / max(cb - ct, 1e-6)
    # pixel offset from buf_out origin to full-output origin
    cx_clip = fullwidth  * cl
    cy_clip = fullheight * ct

    # Exact ashift input dims: the input whose homography output bbox equals the full output
    # (fullwidth × fullheight). NOT fullwidth/fullheight themselves — the homography expands the
    # bbox, so using the output size as the input mis-scales every mask (see _solve_ashift_input_dims).
    W_in, H_in = _solve_ashift_input_dims(ashift_params, fullwidth, fullheight)

    H_mat = _compute_ashift_homography(
        ashift_params["rotation"],
        ashift_params["lensshift_v"],
        ashift_params["lensshift_h"],
        ashift_params["shear"],
        ashift_params["f_length_kb"],
        ashift_params["orthocorr"],
        ashift_params["aspect"],
        W_in, H_in,
    )
    H_inv = np.linalg.inv(H_mat)

    # Convert normalized ashift-output coords to pixels
    px_out = cx * W_out
    py_out = cy * H_out

    # Shift from buf_out space to full-output space (undo internal crop offset)
    px_full = px_out + cx_clip
    py_full = py_out + cy_clip

    # Apply inverse homography → input pixel coords
    p_in = H_inv @ np.array([px_full, py_full, 1.0])
    px_in = p_in[0] / p_in[2]
    py_in = p_in[1] / p_in[2]

    # Normalize by input dimensions → raw-sensor [0,1] coords
    return px_in / W_in, py_in / H_in


def _export_to_original(cx, cy, flip, crop, ashift_params=None, export_w=None, export_h=None):
    """Transform coordinates from export space to original image space.

    Pipeline order: raw -> [ashift(15)] -> flip(16) -> crop(24.5) -> export.
    Inverse: undo crop, undo flip, undo ashift.

    flip is the darktable orientation bitmask (dt_image_orientation_t):
        FLIP_X  = 1  (flip around X axis = vertical mirror,   cy = 1-cy)
        FLIP_Y  = 2  (flip around Y axis = horizontal mirror, cx = 1-cx)
        SWAP_XY = 4  (transpose x and y)
    Common combinations:
        3 = 180° rotation
        5 = CCW 90° (SWAP_XY | FLIP_X)
        6 = CW  90° (SWAP_XY | FLIP_Y)
        7 = portrait (SWAP_XY | FLIP_X | FLIP_Y)
    ashift_params: dict from _decode_ashift_params(), or None for identity.
    export_w, export_h: pixel dimensions of the exported image (needed for ashift).
    """
    crop_l, crop_t, crop_r, crop_b = crop

    # 1. Undo crop (export [0,1] -> post-flip / post-ashift space)
    cx = crop_l + cx * (crop_r - crop_l)
    cy = crop_t + cy * (crop_b - crop_t)

    # 2. Undo flip (post-flip -> ashift-output space).
    # Forward transform applies: SWAP_XY, then FLIP_X, then FLIP_Y.
    # Inverse reverses that order: FLIP_Y, then FLIP_X, then SWAP_XY.
    if flip & 2:   # FLIP_Y: horizontal mirror (flip around Y axis)
        cx = 1.0 - cx
    if flip & 1:   # FLIP_X: vertical mirror (flip around X axis)
        cy = 1.0 - cy
    if flip & 4:   # SWAP_XY: transpose
        cx, cy = cy, cx

    # 3. Undo ashift (ashift-output -> raw-sensor space)
    if ashift_params is not None and export_w is not None and export_h is not None:
        cx, cy = _undo_ashift(cx, cy, ashift_params, export_w, export_h, crop, flip)

    return cx, cy


def _original_to_export(cx_norm, cy_norm, flip, crop, export_w, export_h):
    """Transform normalized full-frame coords to export pixel space.

    Inverse of _export_to_original. Ashift is not supported (sensor dust mode skips it).
    Returns (cx_px, cy_px) or None if the point falls outside the crop region.

    Forward pipeline order: raw -> [ashift] -> flip -> crop -> export.
    """
    crop_l, crop_t, crop_r, crop_b = crop

    # 1. Apply flip (forward order: SWAP_XY → FLIP_X → FLIP_Y)
    if flip & 4:  # SWAP_XY
        cx_norm, cy_norm = cy_norm, cx_norm
    if flip & 1:  # FLIP_X: vertical mirror
        cy_norm = 1.0 - cy_norm
    if flip & 2:  # FLIP_Y: horizontal mirror
        cx_norm = 1.0 - cx_norm

    # 2. Apply crop (map full-frame [0,1] → export [0,1])
    cx_e = (cx_norm - crop_l) / max(crop_r - crop_l, 1e-6)
    cy_e = (cy_norm - crop_t) / max(crop_b - crop_t, 1e-6)

    if not (0.0 <= cx_e <= 1.0 and 0.0 <= cy_e <= 1.0):
        return None  # outside this frame's crop

    return cx_e * export_w, cy_e * export_h


def generate_xmp_data_for_spots(spots, image_width, image_height,
                                flip=0, crop=(0.0, 0.0, 1.0, 1.0), ashift_params=None):
    """Generate all XMP-ready data for a list of detected spots.

    Returns dict with keys: brushes, group, retouch_params, blendop_params.
    Each brush: {mask_id, mask_points, mask_src, mask_nb}.

    flip: darktable orientation bitmask (FLIP_X=1, FLIP_Y=2, SWAP_XY=4)
    crop: (left, top, right, bottom) in post-flip [0,1] space
    ashift_params: dict from _decode_ashift_params(), or None if ashift is absent/disabled
    """
    id_gen = MaskIdGenerator()

    brushes = []
    brush_ids = []

    # Reconstruct original (pre-crop) image dimensions from export dims + crop.
    # darktable scales brush border by MIN(pipe_w, pipe_h) at the retouch module,
    # where pipe dimensions are the original image size (before crop).
    crop_l, crop_t, crop_r, crop_b = crop
    orig_w = image_width / max(crop_r - crop_l, 1e-6)
    orig_h = image_height / max(crop_b - crop_t, 1e-6)
    border_scale = min(orig_w, orig_h)

    def _to_orig(px_x, px_y):
        """Pixel coords in export space -> normalized [0,1] original image space."""
        nx, ny = px_x / image_width, px_y / image_height
        return _export_to_original(nx, ny, flip, crop, ashift_params,
                                   image_width, image_height)

    for spot in spots:
        # Brush border: darktable renders as border * MIN(pipe_w, pipe_h)
        border = spot["brush_radius_px"] / border_scale
        spot["radius_norm"] = border   # persisted to {stem}_debug_spots.json for UI display

        mask_id = id_gen.next_id()
        brush_ids.append(mask_id)

        if spot.get("kind") == "stroke":
            # Multi-node brush stroke following the detected centerline.
            path_norm = [_to_orig(px, py) for px, py in spot["path"]]
            if "src_cx" in spot and "src_cy" in spot:
                src_x, src_y = _to_orig(spot["src_cx"], spot["src_cy"])
            else:
                # Fallback: offset the first node diagonally.
                src_x = path_norm[0][0] + HEAL_SOURCE_OFFSET_X
                src_y = path_norm[0][1] + HEAL_SOURCE_OFFSET_Y
            src_x = max(0.0, min(1.0, src_x))
            src_y = max(0.0, min(1.0, src_y))
            brushes.append({
                "mask_id": mask_id,
                "mask_points": make_stroke_mask_points(path_norm, border),
                "mask_src": struct.pack("<ff", src_x, src_y).hex(),
                "mask_nb": len(path_norm),
            })
            continue

        # Circular "dot" brush (existing behavior)
        norm_cx, norm_cy = _to_orig(spot["cx"], spot["cy"])

        # Heal source: use auto-detected optimal position if available, else fixed offset
        if "src_cx" in spot and "src_cy" in spot:
            src_x, src_y = _to_orig(spot["src_cx"], spot["src_cy"])
            src_x = max(0.0, min(1.0, src_x))
            src_y = max(0.0, min(1.0, src_y))
        else:
            src_x = max(0.0, min(1.0, norm_cx + HEAL_SOURCE_OFFSET_X))
            src_y = max(0.0, min(1.0, norm_cy + HEAL_SOURCE_OFFSET_Y))

        brushes.append({
            "mask_id": mask_id,
            "mask_points": make_brush_mask_points(norm_cx, norm_cy, border),
            "mask_src": struct.pack("<ff", src_x, src_y).hex(),
            "mask_nb": 2,
        })

    # Group mask
    group_id = id_gen.next_id()
    group = {
        "mask_id": group_id,
        "mask_points": make_group_mask_points(brush_ids, group_id),
        "mask_src": "0000000000000000",
        "mask_nb": len(brushes),
    }

    # Retouch params
    retouch_params = make_retouch_params(brush_ids)

    # Blendop params
    blendop_params = make_blendop_params(group_id)

    return {
        "brushes": brushes,
        "group": group,
        "retouch_params": retouch_params,
        "blendop_params": blendop_params,
    }


# ===================================================================
# ===================================================================
# Results output
# ===================================================================

class NumpyEncoder(json.JSONEncoder):
    """Convert numpy scalars to plain Python types for JSON serialization."""
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def write_debug_spots_json(results, image_paths_by_stem, output_dir, mode=None,
                           times_by_stem=None, wall_time_s=None):
    """Write per-image {stem}_debug_spots.json files for the debug UI.

    results: list of (filename_no_ext, spots_or_None, rejected_candidates, img_dims,
                      error_or_None, xmp_data_or_None).
    mode: optional session marker (e.g. "sensor") so the UI can adapt its
          title/report to the detection mode that produced the session.
    times_by_stem: optional {stem: seconds} per-image processing times, persisted
          as "processing_time_s" (shown in the UI, compared by the quality suite).
    wall_time_s: optional total wall-clock seconds of the whole detection run,
          persisted as "run_wall_time_s" in every per-image file (session-level;
          per-image times overlap under parallel workers, so only the wall time
          is comparable run-to-run).
    """
    constants = {k: v for k, v in globals().items()
                 if k.isupper() and isinstance(v, (int, float))}

    for filename, spots, rejected_candidates, img_dims, error, xmp_data in results:
        w, h = img_dims if img_dims else (0, 0)
        data = {
            "stem": filename,
            "image_path": str(image_paths_by_stem.get(filename, "")),
            "width": int(w),
            "height": int(h),
            "detected": spots or [],
            "rejected": rejected_candidates or [],
            "constants": constants,
        }
        if mode:
            data["mode"] = mode
        if times_by_stem and filename in times_by_stem:
            data["processing_time_s"] = round(float(times_by_stem[filename]), 2)
        if wall_time_s is not None:
            data["run_wall_time_s"] = round(float(wall_time_s), 2)
        out_path = os.path.join(output_dir, f"{filename}_debug_spots.json")
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)

    print(f"Debug spots written to: {output_dir} ({len(results)} file(s))")


def parse_source_paths(src_file):
    """Parse source_paths.txt written by the Lua side (stem|/full/source/path).

    Returns {stem: source_path}. Mirrors auto_negadoctor.parse_source_paths — lets a
    frame be keyed by its full source path (so same-named stems from different rolls,
    e.g. every roll's DSC_0013, never collide) instead of the bare stem.
    """
    out = {}
    if not os.path.exists(src_file):
        return out
    with open(src_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "|" not in line:
                continue
            stem, path = line.split("|", 1)
            out[stem.strip()] = path.strip()
    return out


def load_debug_spots_dir(directory):
    """Load all per-image {stem}_debug_spots.json files from a directory.

    Returns (images_list, constants_dict) matching the old monolithic format
    so callers can iterate images_list the same way they iterated data["images"].
    """
    directory = Path(directory)
    files = sorted(directory.glob("*_debug_spots.json"))
    if not files:
        return [], {}

    images = []
    constants = {}
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        if not constants:
            constants = data.get("constants", {})
        images.append({k: v for k, v in data.items() if k != "constants"})
    return images, constants


def write_dust_results(results, output_dir):
    """Write dust_results.txt in the output directory.

    results: list of (filename_no_ext, spots_or_None, rejected_candidates, img_dims,
                      error_or_None, xmp_data_or_None).
    """
    output_path = Path(output_dir) / "dust_results.txt"
    with open(output_path, "w") as f:
        for filename, spots, rejected_candidates, img_dims, error, xmp_data in results:
            if error:
                f.write(f"ERR|{filename}|{error}\n")
                continue

            count = len(spots) if spots else 0
            f.write(f"OK|{filename}|N={count}\n")

            if count == 0 or xmp_data is None:
                continue

            # Brush entries
            for i, brush in enumerate(xmp_data["brushes"]):
                f.write(
                    f"BRUSH|{filename}|{i}"
                    f"|mask_id={brush['mask_id']}"
                    f"|mask_points={brush['mask_points']}"
                    f"|mask_src={brush['mask_src']}"
                    f"|mask_nb={brush['mask_nb']}\n"
                )

            # Group entry
            g = xmp_data["group"]
            f.write(
                f"GROUP|{filename}"
                f"|mask_id={g['mask_id']}"
                f"|mask_points={g['mask_points']}"
                f"|mask_src={g['mask_src']}"
                f"|mask_nb={g['mask_nb']}\n"
            )

            # Params entry
            f.write(
                f"PARAMS|{filename}"
                f"|retouch_params={xmp_data['retouch_params']}"
                f"|blendop_params={xmp_data['blendop_params']}\n"
            )

    return str(output_path)


# ===================================================================
# Sensor dust detection
# ===================================================================

def _sensor_blob_radius(dog, cx, cy, min_r, max_r, peak_val):
    """Walk outward from (cx, cy) in the DoG map until the mean sampled value
    drops below 30% of peak_val. Returns the estimated blob radius in pixels."""
    h, w = dog.shape
    n_dirs = 8
    cos_a = [math.cos(i * 2 * math.pi / n_dirs) for i in range(n_dirs)]
    sin_a = [math.sin(i * 2 * math.pi / n_dirs) for i in range(n_dirs)]
    threshold = 0.3 * peak_val
    for r in range(max(1, min_r // 2), max_r + 1):
        vals = []
        for d in range(n_dirs):
            sx = int(round(cx + r * cos_a[d]))
            sy = int(round(cy + r * sin_a[d]))
            if 0 <= sx < w and 0 <= sy < h:
                vals.append(float(dog[sy, sx]))
        if vals and (sum(vals) / len(vals)) < threshold:
            return r
    return max_r


def _sensor_spot_texture(gray, cx_px, cy_px, radius_px):
    """90th-percentile local-std texture in the ring between 1× and 8× blob radius.

    The wide ring (up to 8× radius, min 200px) captures structural context like
    window frames even when the dust sits in the centre of a smooth glass pane.
    The 90th percentile catches sparse-but-strong edges that median would miss.
    """
    h, w = gray.shape
    inner_r = max(int(radius_px), 5)
    outer_r = max(inner_r * 8, 200)
    y0, y1 = max(0, int(cy_px) - outer_r), min(h, int(cy_px) + outer_r)
    x0, x1 = max(0, int(cx_px) - outer_r), min(w, int(cx_px) + outer_r)
    if y1 <= y0 or x1 <= x0:
        return 0.0
    region = gray[y0:y1, x0:x1].astype(np.float32)
    k = max(3, (inner_r // 2) | 1)
    local_mean = cv2.blur(region, (k, k))
    local_sq_mean = cv2.blur(region ** 2, (k, k))
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))
    cy_rel, cx_rel = int(cy_px) - y0, int(cx_px) - x0
    ys, xs = np.ogrid[:region.shape[0], :region.shape[1]]
    dist = np.sqrt((xs - cx_rel) ** 2 + (ys - cy_rel) ** 2)
    ring_mask = (dist >= inner_r) & (dist <= outer_r)
    if not np.any(ring_mask):
        return 0.0
    return float(np.percentile(local_std[ring_mask], 90))


def _sensor_find_source(gray, cx_px, cy_px, radius_px, cfg=None):
    """Find the smoothest healing source for a sensor dust spot.

    Samples 8 directions at 2.5× radius distance from the spot centre and
    picks the patch with the lowest median local-std texture.

    Returns (src_cx, src_cy, texture) or None if every direction exceeds
    SENSOR_MAX_SOURCE_TEXTURE (no clean source available).
    """
    cfg = DEFAULT_TUNING if cfg is None else cfg
    SENSOR_MAX_SOURCE_TEXTURE = cfg.SENSOR_MAX_SOURCE_TEXTURE
    h, w = gray.shape
    step = max(int(radius_px * 2.5), 15)
    patch_r = max(int(radius_px), 5)
    k = max(3, (patch_r // 2) | 1)

    best_src = None
    best_tex = float('inf')

    for d in range(8):
        angle = d * math.pi / 4
        sx = cx_px + step * math.cos(angle)
        sy = cy_px + step * math.sin(angle)
        sx = max(patch_r, min(w - patch_r - 1, sx))
        sy = max(patch_r, min(h - patch_r - 1, sy))
        y0, y1 = int(sy) - patch_r, int(sy) + patch_r
        x0, x1 = int(sx) - patch_r, int(sx) + patch_r
        if y0 < 0 or y1 >= h or x0 < 0 or x1 >= w:
            continue
        patch = gray[y0:y1, x0:x1].astype(np.float32)
        lm = cv2.blur(patch, (k, k))
        lsm = cv2.blur(patch ** 2, (k, k))
        lst = np.sqrt(np.maximum(lsm - lm ** 2, 0))
        tex = float(np.median(lst))
        if tex < best_tex:
            best_tex = tex
            best_src = (float(sx), float(sy))

    if best_src is None or best_tex > SENSOR_MAX_SOURCE_TEXTURE:
        return None
    return best_src[0], best_src[1], best_tex


def detect_sensor_dust_candidates(image_path, cfg=None):
    """Find sensor-dust candidate blobs using Difference-of-Gaussians + local maxima.

    DoG cancels smooth background gradients (sky, bokeh) while preserving blob-scale
    brightness peaks. Local maxima detection avoids the connected-component fusion
    problem where an entire gradient region merges into one large blob.

    Returns (candidates, error_msg) where candidates is a list of dicts with keys:
      cx, cy (export pixels), radius_px, brush_radius_px, contrast, area, circularity.
    """
    cfg = DEFAULT_TUNING if cfg is None else cfg
    SENSOR_SIGMA_INNER_FRAC = cfg.SENSOR_SIGMA_INNER_FRAC
    SENSOR_SIGMA_OUTER_FRAC = cfg.SENSOR_SIGMA_OUTER_FRAC
    SENSOR_DOG_MIN_CONTRAST = cfg.SENSOR_DOG_MIN_CONTRAST
    SENSOR_MIN_RADIUS_FRAC = cfg.SENSOR_MIN_RADIUS_FRAC
    SENSOR_MAX_BLOB_RADIUS_FRAC = cfg.SENSOR_MAX_BLOB_RADIUS_FRAC
    SENSOR_MAX_CANDIDATE_TEXTURE = cfg.SENSOR_MAX_CANDIDATE_TEXTURE
    SENSOR_BRUSH_SCALE = cfg.SENSOR_BRUSH_SCALE
    img = cv2.imread(str(image_path))
    if img is None:
        return [], f"Cannot read {image_path}"
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    min_dim = min(w, h)
    sigma_inner = max(2.0, min_dim * SENSOR_SIGMA_INNER_FRAC)
    sigma_outer = max(20.0, min_dim * SENSOR_SIGMA_OUTER_FRAC)
    min_radius = max(5, int(min_dim * SENSOR_MIN_RADIUS_FRAC))
    max_blob_r = max(min_radius + 1, int(min_dim * SENSOR_MAX_BLOB_RADIUS_FRAC))

    # Inner blur: small sigma preserves individual dust blob peaks
    blur_inner = cv2.GaussianBlur(gray, (0, 0), sigma_inner)
    # Outer blur: sigma_outer ~= 4% of min_dim keeps background estimate within the same
    # sky/bokeh region rather than spanning the full scene (sky+ground would pull the
    # estimate down, making the entire sky appear elevated in DoG).
    # Computed at 4x downscale so the kernel (≈6σ) stays manageable.
    ds = 4
    dw, dh = max(1, w // ds), max(1, h // ds)
    gray_ds = cv2.resize(gray, (dw, dh), interpolation=cv2.INTER_AREA)
    blur_outer_ds = cv2.GaussianBlur(gray_ds, (0, 0), max(3.0, sigma_outer / ds))
    blur_outer = cv2.resize(blur_outer_ds, (w, h), interpolation=cv2.INTER_LINEAR)

    dog = blur_inner - blur_outer  # positive = brighter than local background

    print(f"  sigma_inner={sigma_inner:.0f}px sigma_outer={sigma_outer:.0f}px "
          f"blob_radius=[{min_radius},{max_blob_r}]px "
          f"dog range: [{dog.min():.1f},{dog.max():.1f}]")

    # Strict local maxima: each pixel must be the maximum in a (min_radius*4) window.
    # Strict NMS (tolerance -0.1 rather than -0.5) gives one pixel per true peak instead of
    # extended plateau regions, so peak count equals the number of distinct blobs.
    nms_diam = max(5, min_radius * 4) | 1
    nms_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nms_diam, nms_diam))
    dog_dilated = cv2.dilate(dog, nms_se)
    peak_mask = (dog >= dog_dilated - 0.1) & (dog > SENSOR_DOG_MIN_CONTRAST)
    peak_ys, peak_xs = np.where(peak_mask)

    if len(peak_ys) == 0:
        return [], None

    peak_vals = dog[peak_ys, peak_xs]

    # Exclude peaks within min_radius of the image border (film holder edge artifacts)
    border = min_radius
    interior = ((peak_xs >= border) & (peak_xs < w - border)
                & (peak_ys >= border) & (peak_ys < h - border))
    peak_ys = peak_ys[interior]
    peak_xs = peak_xs[interior]
    peak_vals = peak_vals[interior]

    if len(peak_ys) == 0:
        return [], None

    # Context texture pre-filter: reject candidates in busy areas before consensus.
    # Sensor dust is only visible (and only meaningful to match) in smooth regions like
    # sky, bokeh, or walls. Peaks in high-texture content (people, foliage, buildings)
    # produce false cross-frame matches and must be excluded here.
    #
    # Compute local-std texture map with kernel = 3×max_blob_r (captures the local
    # character of the area, not just the dust spot itself). Sample at each peak.
    ctx_k = max(3, max_blob_r * 3) | 1
    lm_ctx = cv2.blur(gray, (ctx_k, ctx_k))
    lsm_ctx = cv2.blur(gray ** 2, (ctx_k, ctx_k))
    ctx_tex_map = np.sqrt(np.maximum(lsm_ctx - lm_ctx ** 2, 0))
    ctx_at_peaks = ctx_tex_map[peak_ys, peak_xs]
    smooth_mask = ctx_at_peaks <= SENSOR_MAX_CANDIDATE_TEXTURE
    n_before = len(peak_ys)
    peak_ys = peak_ys[smooth_mask]
    peak_xs = peak_xs[smooth_mask]
    peak_vals = peak_vals[smooth_mask]
    print(f"  Texture pre-filter: {smooth_mask.sum()}/{n_before} peaks in smooth areas "
          f"(threshold={SENSOR_MAX_CANDIDATE_TEXTURE})")

    if len(peak_ys) == 0:
        return [], None

    # Spatial grid: one strongest peak per max_blob_r-sized cell.
    nms_cell = max_blob_r
    grid_rows = max(1, h // nms_cell)
    grid_cols = max(1, w // nms_cell)
    cell_idx = (np.minimum((peak_ys * grid_rows // h).astype(np.int32), grid_rows - 1)
                * grid_cols
                + np.minimum((peak_xs * grid_cols // w).astype(np.int32), grid_cols - 1))
    order = np.lexsort((-peak_vals, cell_idx))
    _, first_in_cell = np.unique(cell_idx[order], return_index=True)
    sel = order[first_in_cell]

    print(f"  DoG peaks above {SENSOR_DOG_MIN_CONTRAST}: {len(sel)}")

    candidates = []
    for k in sel:
        cx, cy = float(peak_xs[k]), float(peak_ys[k])
        peak_val = float(peak_vals[k])

        blob_r = _sensor_blob_radius(dog, cx, cy, min_radius, max_blob_r, peak_val)
        if blob_r < min_radius:
            continue
        # blob_r == max_blob_r means the ring walk hit the limit without finding the edge —
        # in dense dust fields, neighboring spots keep DoG elevated and the walk never drops.
        # Accept with max_blob_r as the radius; cross-frame consensus handles false positives.

        area = int(math.pi * blob_r ** 2)
        brush_r = max(MIN_BRUSH_PX, blob_r * SENSOR_BRUSH_SCALE)
        candidates.append({
            "cx": cx, "cy": cy,
            "radius_px": float(blob_r), "brush_radius_px": brush_r,
            "contrast": peak_val, "area": area, "circularity": 1.0,
        })

    print(f"  Found {len(candidates)} candidate(s)")
    return candidates, None


def find_sensor_dust_consensus(per_image_results, transforms, cfg=None):
    """Identify dust positions common across multiple frames (sensor dust signature).

    per_image_results: list of (stem, candidates, img_w, img_h)
    transforms: dict from parse_transform_params()

    Candidates from each frame are mapped to normalized full-frame coordinates, then
    clustered. Clusters present in >= SENSOR_DUST_MIN_FRAMES distinct frames are
    confirmed as sensor dust.

    Returns list of dicts: {cx_norm, cy_norm, radius_norm, n_frames} in full-frame space.
    """
    cfg = DEFAULT_TUNING if cfg is None else cfg
    SENSOR_CLUSTER_RADIUS_NORM = cfg.SENSOR_CLUSTER_RADIUS_NORM
    SENSOR_DUST_MIN_FRAMES = cfg.SENSOR_DUST_MIN_FRAMES
    SENSOR_MAX_CANDIDATES_FOR_CONSENSUS = cfg.SENSOR_MAX_CANDIDATES_FOR_CONSENSUS
    from collections import defaultdict

    stems = [r[0] for r in per_image_results]
    n_images = len(per_image_results)
    all_pts = []  # (cx_ff, cy_ff, radius_norm, stem_idx, contrast)
    for stem_idx, (stem, candidates, img_w, img_h) in enumerate(per_image_results):
        if not candidates or not img_w or not img_h:
            continue
        t = transforms.get(stem, {"flip": 0, "crop": (0.0, 0.0, 1.0, 1.0), "ashift": None})
        flip = t["flip"]
        crop = t["crop"]
        crop_l, crop_t, crop_r, crop_b = crop
        crop_scale = min(crop_r - crop_l, crop_b - crop_t)
        min_export = min(img_w, img_h)
        # Limit high-candidate-count frames to prevent excessive union-find chaining.
        # High-contrast peaks are real blobs; weak-contrast peaks are often smooth-area
        # noise that creates long chains merging distinct sensor-dust clusters.
        # Only apply for large batches (>15 images): small batches have fewer frames to
        # vote down FPs, so we need broader candidate coverage to avoid missing real dust.
        use_cands = candidates
        if n_images > 15 and len(candidates) > SENSOR_MAX_CANDIDATES_FOR_CONSENSUS:
            use_cands = sorted(candidates, key=lambda c: c["contrast"],
                               reverse=True)[:SENSOR_MAX_CANDIDATES_FOR_CONSENSUS]
        for c in use_cands:
            cx_n = c["cx"] / img_w
            cy_n = c["cy"] / img_h
            cx_ff, cy_ff = _export_to_original(cx_n, cy_n, flip, crop)
            # Radius in full-frame normalized: scale from export pixels
            r_norm = c["radius_px"] / max(min_export, 1) * max(crop_scale, 1e-6)
            all_pts.append((cx_ff, cy_ff, r_norm, stem_idx, c["contrast"]))

    if not all_pts:
        return []

    # Union-find clustering — vectorized pair search via numpy broadcast
    n = len(all_pts)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    # cKDTree pair search — O(n log n), handles large candidate sets efficiently.
    # Cross-frame-only unions: never link two candidates from the same frame.
    # This prevents same-frame chaining (dense sky FPs flooding into mega-clusters).
    coords = np.array([(p[0], p[1]) for p in all_pts], dtype=np.float64)
    tree = cKDTree(coords)
    for i, j in tree.query_pairs(SENSOR_CLUSTER_RADIUS_NORM):
        if all_pts[i][3] != all_pts[j][3]:  # different frames only
            union(i, j)

    clusters = defaultdict(list)
    for i, pt in enumerate(all_pts):
        clusters[find(i)].append(pt)

    # Require more frames for larger batches: 2 frames is too permissive when there
    # are 30+ images — coincidental position overlaps produce many spurious 2-frame
    # clusters.  Scale to ~10% of batch size (minimum 2).
    min_frames = max(SENSOR_DUST_MIN_FRAMES, n_images // 10)
    confirmed = []
    for members in clusters.values():
        frame_indices = {m[3] for m in members}
        if len(frame_indices) < min_frames:
            continue

        # Chain-cluster rescue: if any frame contributes more than one candidate, union-find
        # has chained through multiple positions (different content in different frames linked
        # via the same high-candidate-count smooth-area frame). Extract the tightest sub-cluster
        # by taking the best-contrast candidate per frame, then computing the centroid from
        # those representatives only.
        max_per_frame = max(
            sum(1 for m in members if m[3] == fi) for fi in frame_indices
        )
        if max_per_frame > 1:
            # Deduplicate: keep highest-contrast candidate per frame
            frame_best: dict = {}
            for m in members:
                fi = m[3]
                if fi not in frame_best or m[4] > frame_best[fi][4]:  # m[4] = contrast
                    frame_best[fi] = m
            members = list(frame_best.values())
            frame_indices = {m[3] for m in members}
            if len(frame_indices) < min_frames:
                continue

        cx_mean = sum(m[0] for m in members) / len(members)
        cy_mean = sum(m[1] for m in members) / len(members)
        # Spread filter: real sensor dust maps to nearly identical positions across frames;
        # cross-frame FP chains spread over many cluster radii. Reject diffuse clusters.
        max_spread = max(math.sqrt((m[0] - cx_mean) ** 2 + (m[1] - cy_mean) ** 2)
                         for m in members)
        if max_spread > 2.0 * SENSOR_CLUSTER_RADIUS_NORM:
            continue
        r_mean = sum(m[2] for m in members) / len(members)
        confirmed.append({
            "cx_norm": cx_mean, "cy_norm": cy_mean,
            "radius_norm": r_mean, "n_frames": len(frame_indices),
            "stem_set": {stems[i] for i in frame_indices},
        })

    return confirmed


def process_one_sensor_image(args):
    """Detect sensor dust candidates in one image. Top-level for multiprocessing pickling."""
    (image_path,) = args
    filename = Path(image_path).stem
    t0 = time.perf_counter()
    buf = io.StringIO()
    candidates = []
    error = None
    img_dims = (0, 0)
    try:
        with redirect_stdout(buf):
            print(f"\n{'='*60}")
            print(f"Sensor dust candidates: {filename}")
            img = cv2.imread(str(image_path))
            if img is not None:
                img_dims = (img.shape[1], img.shape[0])
            candidates, error = detect_sensor_dust_candidates(image_path)
            if error:
                print(f"  ERROR: {error}")
    except Exception as e:
        error = str(e)
        buf.write(f"  EXCEPTION: {e}\n")
    elapsed = time.perf_counter() - t0
    buf.write(f"  [{filename}] processed in {elapsed:.1f}s\n")
    return (filename, candidates, img_dims, error, buf.getvalue(), elapsed)


def _compute_frame_pairwise_overlap(consensus_spots):
    """Count shared consensus spots between each frame pair, weighted by n_frames.

    Weighting by n_frames means spots confirmed in many frames contribute more to
    the partner score, so the best partner is determined by reliably-seen dust rather
    than incidental single-frame overlaps.

    Returns dict mapping (frameA, frameB) → weighted score (A < B lexicographically).
    """
    from itertools import combinations as _comb
    pairwise = {}
    for sd in consensus_spots:
        stems = sorted(sd["stem_set"])
        weight = sd["n_frames"]
        for a, b in _comb(stems, 2):
            key = (a, b)
            pairwise[key] = pairwise.get(key, 0) + weight
    return pairwise


def run_sensor_dust_mode(image_paths, transforms, output_dir, debug_ui=False, cfg=None):
    """Full sensor dust pipeline: per-image candidates → cross-frame consensus → XMP data.

    When debug_ui is True, also writes a per-image debug session
    ({stem}_debug_spots.json with mode="sensor") and launches debug_ui.py on it.

    Returns True if any errors occurred.
    """
    cfg = DEFAULT_TUNING if cfg is None else cfg
    SENSOR_BRUSH_SCALE = cfg.SENSOR_BRUSH_SCALE
    SENSOR_MAX_CORRECTION_TEXTURE = cfg.SENSOR_MAX_CORRECTION_TEXTURE
    SENSOR_MAX_SOURCE_TEXTURE = cfg.SENSOR_MAX_SOURCE_TEXTURE
    n_workers = min(cpu_count(), len(image_paths))
    print(f"Sensor dust mode: {len(image_paths)} image(s), {n_workers} worker(s)", flush=True)

    wall_t0 = time.perf_counter()
    per_image_results = []
    times_by_stem = {}
    any_errors = False
    args_list = [(p,) for p in image_paths]

    with Pool(processes=n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_one_sensor_image, args_list), 1):
            filename, candidates, img_dims, error, log_output, elapsed = result
            print(log_output, end="", flush=True)
            if error:
                any_errors = True
            per_image_results.append((filename, candidates, img_dims[0], img_dims[1]))
            times_by_stem[filename] = elapsed
            print(f"PROGRESS|{i}|{len(image_paths)}", flush=True)

    consensus_spots = find_sensor_dust_consensus(per_image_results, transforms, cfg=cfg)
    print(f"\nSensor dust consensus: {len(consensus_spots)} spot(s) confirmed across frames")

    for sd in consensus_spots:
        print(f"  pos=({sd['cx_norm']:.4f},{sd['cy_norm']:.4f}) "
              f"radius_norm={sd['radius_norm']:.4f} seen_in={sd['n_frames']} frames")

    image_paths_by_stem = {Path(p).stem: p for p in image_paths}
    all_stems = [r[0] for r in per_image_results]

    # Pairwise frame correlation filter.
    # Sensor dust appears in all frames that have smooth content at the dust position.
    # Frames with many shared consensus positions have similar smooth-area distributions
    # (same sky, same scene layout). For each frame, only apply consensus spots that also
    # appear in its most-correlated partner — this eliminates coincidental matches from
    # dissimilar frames while preserving all real sensor dust positions.
    pairwise = _compute_frame_pairwise_overlap(consensus_spots)
    best_partner = {}   # stem → most-correlated partner stem
    for stem in all_stems:
        best_other, best_count = None, 0
        for other in all_stems:
            if other == stem:
                continue
            key = (min(stem, other), max(stem, other))
            c = pairwise.get(key, 0)
            if c > best_count:
                best_count = c
                best_other = other
        best_partner[stem] = best_other
        if best_other:
            print(f"  [{stem}] most correlated with {best_other} ({best_count} shared spots)")

    # Back-project consensus positions into each frame's export space and generate XMP
    results = []
    for (stem, candidates, img_w, img_h) in per_image_results:
        t = transforms.get(stem, {"flip": 0, "crop": (0.0, 0.0, 1.0, 1.0), "ashift": None})
        flip = t["flip"]
        crop = t["crop"]
        crop_scale = min(crop[2] - crop[0], crop[3] - crop[1])
        min_export = min(img_w, img_h) if img_w and img_h else 1

        # Load export image once for per-spot texture checks in this frame
        gray_export = None
        export_path = image_paths_by_stem.get(stem)
        if export_path:
            loaded = cv2.imread(str(export_path))
            if loaded is not None:
                gray_export = cv2.cvtColor(loaded, cv2.COLOR_BGR2GRAY)

        # Pre-pass: project consensus spots into this frame and compute spot texture.
        # Only include spots that:
        #   1. were detected in THIS frame (stem_set membership)
        #   2. were ALSO detected in this frame's most-correlated partner (low-confidence),
        #      OR appear in enough frames that partner corroboration is unnecessary.
        # Spots seen in HIGH_CONFIDENCE_FRAMES or more are accepted without partner check:
        # stable positions across that many diverse frames are overwhelmingly real sensor dust.
        # Threshold scales with batch size (~25%) but is at least 10, so only truly
        # exceptional spots bypass; this prevents n=6-9 coincidental clusters from slipping
        # through while still allowing n=10+ real-dust mega-clusters.
        required_partner = best_partner.get(stem)
        high_confidence_frames = max(10, len(all_stems) // 4)
        projected = []
        for sd in consensus_spots:
            if stem not in sd["stem_set"]:
                continue  # not detected in this frame
            if (required_partner and required_partner not in sd["stem_set"]
                    and sd["n_frames"] < high_confidence_frames):
                continue  # not shared with partner and too few frames — likely a coincidence
            proj = _original_to_export(sd["cx_norm"], sd["cy_norm"], flip, crop, img_w, img_h)
            if proj is None:
                continue
            cx_px, cy_px = proj
            radius_px = sd["radius_norm"] * max(min_export, 1) / max(crop_scale, 1e-6)
            spot_tex = (_sensor_spot_texture(gray_export, cx_px, cy_px, radius_px)
                        if gray_export is not None else 0.0)
            projected.append((spot_tex, sd, cx_px, cy_px, radius_px))

        # Sort: smoothest area first; within same texture, higher n_frames first
        projected.sort(key=lambda x: (x[0], -x[1]["n_frames"]))

        spots = []
        rejected = []   # skipped candidates, persisted for debug-UI review
        skipped_texture = 0
        skipped_no_source = 0
        max_forms_announced = False
        for spot_tex, sd, cx_px, cy_px, radius_px in projected:
            if len(spots) >= MAX_FORMS:
                if not max_forms_announced:
                    max_forms_announced = True
                    remaining = len(projected) - len(spots) - skipped_texture - skipped_no_source
                    print(f"  [{stem}] reached MAX_FORMS={MAX_FORMS} limit, "
                          f"{remaining} lower-priority spots not applied")
                rejected.append({
                    "cx": cx_px, "cy": cy_px, "area": 0, "contrast": 0.0,
                    "reason": "max_forms",
                    "detail": f"spot_tex={spot_tex:.1f} n_frames={sd['n_frames']}",
                })
                continue

            brush_r = max(MIN_BRUSH_PX, radius_px * SENSOR_BRUSH_SCALE)

            if gray_export is not None:
                if spot_tex > SENSOR_MAX_CORRECTION_TEXTURE:
                    skipped_texture += 1
                    print(f"  [{stem}] skip ({cx_px:.0f},{cy_px:.0f}) "
                          f"spot_texture={spot_tex:.1f} > {SENSOR_MAX_CORRECTION_TEXTURE}")
                    rejected.append({
                        "cx": cx_px, "cy": cy_px, "area": 0, "contrast": 0.0,
                        "reason": "busy_area",
                        "detail": f"spot_tex={spot_tex:.1f} > "
                                  f"{SENSOR_MAX_CORRECTION_TEXTURE} "
                                  f"n_frames={sd['n_frames']}",
                    })
                    continue

                src_result = _sensor_find_source(gray_export, cx_px, cy_px, radius_px, cfg=cfg)
                if src_result is None:
                    skipped_no_source += 1
                    print(f"  [{stem}] skip ({cx_px:.0f},{cy_px:.0f}) "
                          f"no clean source (all directions > {SENSOR_MAX_SOURCE_TEXTURE})")
                    rejected.append({
                        "cx": cx_px, "cy": cy_px, "area": 0, "contrast": 0.0,
                        "reason": "no_clean_source",
                        "detail": f"all directions > {SENSOR_MAX_SOURCE_TEXTURE} "
                                  f"n_frames={sd['n_frames']}",
                    })
                    continue
                src_cx, src_cy, src_tex = src_result
                print(f"  [{stem}] heal ({cx_px:.0f},{cy_px:.0f}) r={radius_px:.0f}px "
                      f"brush={brush_r:.0f}px spot_tex={spot_tex:.1f} "
                      f"src=({src_cx:.0f},{src_cy:.0f}) src_tex={src_tex:.1f}")
            else:
                src_cx = cx_px + brush_r * 2.0
                src_cy = cy_px

            spots.append({
                "cx": cx_px, "cy": cy_px,
                "radius_px": radius_px, "brush_radius_px": brush_r,
                "src_cx": src_cx, "src_cy": src_cy,
                "contrast": 10.0, "area": 0,
                "texture": spot_tex, "context_texture": 0.0,
                "excess_sat": 0.0, "spot_sat": 0.0,
                "n_frames": sd["n_frames"],
                "radius_norm": sd["radius_norm"],
            })

        total_skipped = skipped_texture + skipped_no_source
        if total_skipped:
            print(f"  [{stem}] {total_skipped}/{len(consensus_spots)} skipped "
                  f"({skipped_texture} busy area, {skipped_no_source} no clean source)")

        xmp_data = None
        if spots and img_w and img_h:
            xmp_data = generate_xmp_data_for_spots(
                spots, img_w, img_h, flip=flip, crop=crop, ashift_params=t.get("ashift"))
        results.append((stem, spots, rejected, (img_w, img_h), None, xmp_data))

    results.sort(key=lambda r: r[0])
    write_dust_results(results, output_dir)

    wall_time = time.perf_counter() - wall_t0
    print(f"\nTotal wall time: {wall_time:.1f}s for {len(image_paths)} image(s)")

    if debug_ui:
        write_debug_spots_json(results, image_paths_by_stem, output_dir, mode="sensor",
                               times_by_stem=times_by_stem, wall_time_s=wall_time)
        import subprocess
        debug_ui_script = Path(__file__).parent / "debug_ui.py"
        print(f"Launching debug UI: {debug_ui_script}", flush=True)
        subprocess.Popen([sys.executable, str(debug_ui_script), output_dir])

    return any_errors


# ===================================================================
# Per-image worker (top-level so it's picklable on Windows spawn)
# ===================================================================

def process_one_image(args, cfg=None):
    """Detect dust in one image and return results + captured log output.

    Must be a top-level function (not nested inside main) so multiprocessing
    can pickle it on Windows, which uses the 'spawn' start method.
    """
    cfg = DEFAULT_TUNING if cfg is None else cfg
    MAX_SPOTS = cfg.MAX_SPOTS
    image_path, transforms, collect_rejects, ml_model_path = args
    filename = Path(image_path).stem

    t0 = time.perf_counter()
    buf = io.StringIO()
    spots = None
    rejected_candidates = []
    error = None
    xmp_data = None
    img_dims = (0, 0)

    try:
        with redirect_stdout(buf):
            print(f"\n{'='*60}")
            print(f"Processing: {filename}")
            print(f"{'='*60}")

            spots, rejected_candidates, error, local_std = detect(
                image_path, collect_rejects=collect_rejects,
                ml_model_path=ml_model_path, cfg=cfg)
            if error:
                print(f"  ERROR: {error}")
                elapsed = time.perf_counter() - t0
                buf.write(f"  [{filename}] processed in {elapsed:.1f}s\n")
                return (filename, None, [], (0, 0), error, None, buf.getvalue(),
                        elapsed)

            print(f"  Image loaded successfully")
            img = cv2.imread(str(image_path))
            height, width = img.shape[:2]
            img_dims = (width, height)
            print(f"  Dimensions: {width} x {height}")
            print(f"  Detected {len(spots)} dust spot(s) before filtering")

            spots.sort(key=lambda s: s["contrast"], reverse=True)
            if len(spots) > MAX_SPOTS:
                print(f"  Capping from {len(spots)} to {MAX_SPOTS} strongest spots")
                spots = spots[:MAX_SPOTS]

            # Find optimal healing source for each spot
            if spots and local_std is not None:
                img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
                L_f32 = img_lab[:, :, 0].astype(np.float32)
                for spot in spots:
                    if spot.get("kind") == "stroke":
                        src_cx, src_cy = find_stroke_healing_source(
                            spot["path"], spot["stroke_width_px"], spot["brush_radius_px"],
                            local_std, spots, width, height, cfg=cfg)
                    else:
                        src_cx, src_cy = find_healing_source(
                            spot["cx"], spot["cy"], spot["radius_px"], spot["brush_radius_px"],
                            img_lab, L_f32, local_std, spots, width, height, cfg=cfg)
                    spot["src_cx"] = src_cx
                    spot["src_cy"] = src_cy

            for i, spot in enumerate(spots):
                src_str = (f" src=({spot['src_cx']:.0f},{spot['src_cy']:.0f})"
                           if "src_cx" in spot else "")
                print(
                    f"    Spot {i}: center=({spot['cx']:.1f}, {spot['cy']:.1f}) "
                    f"radius={spot['radius_px']:.1f}px area={spot['area']}px "
                    f"contrast={spot['contrast']:.1f} texture={spot['texture']:.1f} "
                    f"exSat={spot['excess_sat']:.1f}{src_str}"
                )

            if spots:
                t = transforms.get(filename, {"flip": 0, "crop": (0.0, 0.0, 1.0, 1.0), "ashift": None})
                ashift = t.get("ashift")
                print(f"  Transform: flip={t['flip']}, crop={t['crop']}, ashift={'yes' if ashift else 'no'}")
                xmp_data = generate_xmp_data_for_spots(
                    spots, width, height,
                    flip=t["flip"], crop=t["crop"], ashift_params=ashift)
                print(f"  Generated XMP data: {len(xmp_data['brushes'])} brush masks")

    except Exception as e:
        buf.write(f"  EXCEPTION: {e}\n")
        error = str(e)

    elapsed = time.perf_counter() - t0
    buf.write(f"  [{filename}] processed in {elapsed:.1f}s\n")
    return (filename, spots, rejected_candidates, img_dims, error, xmp_data,
            buf.getvalue(), elapsed)


# ===================================================================
# Main
# ===================================================================

def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: detect_dust.py [--debug-ui] [--sensor-dust] [--ml-model PATH] <image1.jpg> [image2.jpg ...]")
        sys.exit(1)

    sensor_dust = False
    if "--sensor-dust" in args:
        sensor_dust = True
        args.remove("--sensor-dust")

    debug_ui = False
    if "--debug-ui" in args:
        debug_ui = True
        args.remove("--debug-ui")

    ml_model_path = None
    if "--ml-model" in args:
        idx = args.index("--ml-model")
        if idx + 1 >= len(args):
            print("Error: --ml-model requires a path argument")
            sys.exit(1)
        ml_model_path = args[idx + 1]
        args = args[:idx] + args[idx + 2:]
        if not os.path.isfile(ml_model_path):
            print(f"Error: ML model file not found: {ml_model_path}")
            sys.exit(1)
        print(f"ML mode: using model {ml_model_path}")

    image_paths = args
    if not image_paths:
        print("Error: No image files specified")
        sys.exit(1)

    # Validate files exist
    supported_ext = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}
    for p in image_paths:
        if not os.path.isfile(p):
            print(f"Error: File not found: {p}")
            sys.exit(1)
        ext = os.path.splitext(p)[1].lower()
        if ext not in supported_ext:
            print(f"Error: Unsupported format: {p}")
            sys.exit(1)

    output_dir = str(Path(image_paths[0]).parent)

    # Load per-image transform params (flip/crop) written by Lua
    transform_params_file = os.path.join(output_dir, "transform_params.txt")
    transforms = parse_transform_params(transform_params_file)
    if transforms:
        print(f"Loaded transform params for {len(transforms)} image(s)")
    else:
        print("No transform_params.txt found, using identity transform")

    if sensor_dust:
        if len(image_paths) < 2:
            print("Error: --sensor-dust requires at least 2 images for cross-frame correlation")
            sys.exit(1)
        had_errors = run_sensor_dust_mode(image_paths, transforms, output_dir,
                                          debug_ui=debug_ui)
        sys.exit(1 if had_errors else 0)

    # Debug UI (film dust): launch the UI IMMEDIATELY and let it run detection
    # itself, streaming each frame's spots in as it finalizes (the same UI-first +
    # progress-bar experience as auto_negadoctor's --run). The old path detected the
    # whole batch here and only THEN opened the UI, so the window appeared minutes
    # late. The UI detects the export-dir JPGs in a background pool; no debug_spots
    # batch is written (review-only). (Sensor dust is multi-frame consensus — it
    # can't stream per-frame — so it keeps the batch-then-launch path below.)
    if debug_ui:
        import subprocess
        debug_ui_script = Path(__file__).parent / "debug_ui.py"
        env = dict(os.environ)
        if ml_model_path:
            env["RETOUCH_ML_MODEL"] = ml_model_path
        print(f"Launching debug UI (it runs detection itself, streaming): "
              f"{debug_ui_script}", flush=True)
        subprocess.Popen([sys.executable, str(debug_ui_script), output_dir], env=env)
        sys.exit(0)

    image_paths_by_stem = {Path(p).stem: p for p in image_paths}
    results = []
    times_by_stem = {}
    any_errors = False
    wall_t0 = time.perf_counter()

    n_workers = min(cpu_count(), len(image_paths))
    print(f"Running {n_workers} parallel worker(s) for {len(image_paths)} image(s)", flush=True)

    args_list = [(p, transforms, debug_ui, ml_model_path) for p in image_paths]

    with Pool(processes=n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_one_image, args_list), 1):
            filename, spots, rejected_candidates, img_dims, error, xmp_data, log_output, elapsed = result
            # Print worker log sequentially (no interleaving between images)
            print(log_output, end="", flush=True)
            if error:
                any_errors = True
            results.append((filename, spots, rejected_candidates, img_dims, error, xmp_data))
            times_by_stem[filename] = elapsed
            # Progress sentinel read by Lua via io.popen()
            print(f"PROGRESS|{i}|{len(image_paths)}", flush=True)

    # Sort for deterministic output order (imap_unordered returns in completion order)
    results.sort(key=lambda r: r[0])

    wall_time = time.perf_counter() - wall_t0
    print(f"\nTotal wall time: {wall_time:.1f}s for {len(image_paths)} image(s)")

    # Write results file (non-debug path: the InPlace/apply flow parses this)
    results_path = write_dust_results(results, output_dir)
    print(f"\nResults written to: {results_path}")

    sys.exit(1 if any_errors else 0)


if __name__ == "__main__":
    main()
