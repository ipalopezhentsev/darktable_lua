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


# ---------------------------------------------------------------------------
# Detection constants (conservative defaults)
# ---------------------------------------------------------------------------
LOCAL_BG_KERNEL = 201          # Gaussian blur kernel for local background
NOISE_THRESHOLD_MULTIPLIER = 3.0  # spots must be this many std devs above background
MIN_ABSOLUTE_THRESHOLD = 15.0  # minimum brightness difference regardless of noise
MIN_SPOT_AREA = 6              # minimum pixels (reject subpixel/imperceptible dust)
MAX_SPOT_AREA = 800            # maximum pixels (~16px radius at full res)
MIN_ASPECT_RATIO = 0.3         # bounding box aspect ratio (reject elongated fibers)
MIN_COMPACTNESS = 0.25         # area / bbox_area (reject irregular shapes)
MIN_SOLIDITY = 0.5             # area / convex_hull_area (reject non-convex shapes like letters)
MIN_CIRCULARITY = 0.15         # 4*pi*area/perimeter^2 (reject complex shapes like symbols)
SHAPE_CHECK_MIN_AREA = 30      # only check solidity/circularity for spots larger than this
TEXTURE_KERNEL = 31            # neighborhood size for local texture measurement
MAX_LOCAL_TEXTURE_SMALL = 12.0 # max texture for tiny spots (area near MIN_SPOT_AREA)
MAX_LOCAL_TEXTURE_LARGE = 8.0  # max texture for large spots (area >= 200px)
MAX_DARK_BG_TEXTURE = 10.0     # max texture in ring around spot on dark backgrounds (separate from above)
MIN_CONTRAST_TEXTURE_RATIO = 5.5  # contrast/texture — reject spots hidden in grain
MAX_BG_GRADIENT_RATIO = 0.09   # max bg_gradient/contrast — reject edge halo artifacts
MAX_EXCESS_SATURATION = 10     # max (spot_sat - surround_sat) — dust matches local color cast
MAX_SPOT_SATURATION = 230      # compound check lower bound: spot_sat above this + positive excess_sat
EMULSION_EXCESS_SAT_THRESHOLD = 9  # excess_sat above this (when spot_sat > 230) = emulsion artifact:
                               # dust is achromatic → less colorful than surroundings; a spot that is
                               # both extremely colorful (>230) AND more saturated than its surroundings
                               # (>9) is a film emulsion feature, not dust
MAX_CONTEXT_TEXTURE = 9.0      # max median local_std across a 200px radius from spot center.
                               # Dust sits on smooth backgrounds (sky, walls: median ≈ 2-7).
                               # Crowd/foliage FPs sit in texured scenes (median ≈ 10-25).
                               # Threshold at 9 separates smooth-background dust from busy-scene FPs.
LARGE_SPOT_AREA_THRESHOLD = 300  # spots larger than this require higher contrast
LARGE_SPOT_MIN_CONTRAST = 60     # min contrast for large spots — avoids pale foggy blobs
ENC_RADIUS_SCALE = 3          # multiplicative scale factor for enclosing circle radius (helps darktable 
                                #to correct better while on zoomed out view)
ISOLATION_RADIUS = 250         # pixel radius for neighbor density check
MAX_NEARBY_ACCEPTED = 3        # reject if more than this many accepted spots within ISOLATION_RADIUS
                               # Real dust is sparse; crowd/foliage FPs form dense clusters.
# Soft voting: require MIN_DUST_VOTES of 3 signals to clearly indicate dust.
# Catches borderline spots that slip through individual hard filters but fail multiple soft tests.
SOFT_CONTEXT_VOTE_THRESHOLD = 7.0  # context_texture < this votes "dust" (clear sky/walls: 2-6)
SOFT_TEXTURE_VOTE_THRESHOLD = 8.0  # local_texture < this votes "dust"
SOFT_RATIO_VOTE_THRESHOLD = 8.25   # contrast/texture > this votes "dust" (1.5× hard minimum)
MIN_DUST_VOTES = 2             # require at least this many out of 3 soft votes to accept
MIN_BRIGHTNESS_FRAC_SMALL = 0.5  # brightness floor for tiny spots (area ~10)
MIN_BRIGHTNESS_FRAC_LARGE = 0.8  # brightness floor for large spots (area >= 100)
MIN_LOCAL_BG_FRACTION = 0.5    # local background must be >= this fraction of 95th pct
MIN_SURROUND_BG_RATIO = 0.7   # immediate surround must be >= this fraction of local bg
                               # (rejects bright reflections inside dark features like windows)
MAX_SPOTS = 200                # cap: sort by contrast, take the most obvious ones
REJECT_LOG_CONTRAST_MIN = 15  # minimum contrast to include in debug reject candidate list

# ---------------------------------------------------------------------------
# ML detection constants
# ---------------------------------------------------------------------------
ML_RECOVERY_THRESHOLD_MULT = 2.5  # lower threshold for recovery pass (find missed dust)
ML_POSTFILTER_THRESHOLD = 0.5     # min ML probability to keep a spot (post-filter)
ML_RECOVERY_THRESHOLD = 0.85      # higher bar for accepting recovery candidates

# Feature names used by the post-filter ML model (derived from the spot dict).
# These match what detect_dust_spots() returns — no extra image processing required.
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

# ---------------------------------------------------------------------------
# Source detection constants
# ---------------------------------------------------------------------------
SOURCE_SEARCH_INNER_FACTOR = 2.5  # inner exclusion ring = radius * this (avoid the spot itself)
SOURCE_SEARCH_MAX_RADIUS = 150    # cap search radius in pixels
SOURCE_SEARCH_MIN_RADIUS = 25     # minimum search radius for tiny spots
SOURCE_GRID_STEP = 8              # grid step for candidate sampling (pixels)

RETOUCH_ALGO_HEAL = 2
RETOUCH_MOD_VERSION = 3
MAX_FORMS = 300                # darktable's maximum form slots

# Known-good blendop_params template from a real darktable XMP (420 bytes uncompressed)
# Only the mask_id field at offset 24 needs to be replaced per-image
BLENDOP_TEMPLATE_ENCODED = "gz08eJxjYGBgYAFiCQYYOOEEIjd6dmXCRFgZMAEjFjEGhgZ7CB6pfOygYtaVAyCMi48L/AcCEA0Ak0kpjg=="

# ---------------------------------------------------------------------------
# Sensor dust detection constants
# ---------------------------------------------------------------------------
SENSOR_SIGMA_INNER_FRAC   = 0.004  # DoG inner Gaussian sigma as fraction of min(w,h)
SENSOR_SIGMA_OUTER_FRAC   = 0.04   # DoG outer Gaussian sigma (large enough to see around blobs,
                                    # small enough to stay within the same sky region)
SENSOR_DOG_MIN_CONTRAST   = 4.0    # minimum DoG peak value to consider as candidate
SENSOR_MIN_RADIUS_FRAC    = 0.003  # min blob radius as fraction of min(w,h)
SENSOR_MAX_RADIUS_FRAC    = 0.12   # grid cell size (one candidate per this many px)
SENSOR_MAX_BLOB_RADIUS_FRAC = 0.01 # maximum accepted blob radius (% of min_dim); sensor dust is small
SENSOR_CLUSTER_RADIUS_NORM = 0.02  # cluster radius in normalized full-frame coords
SENSOR_DUST_MIN_FRAMES    = 2      # min frames a cluster must appear in to confirm sensor dust
SENSOR_BRUSH_SCALE        = 1.0    # brush_radius_px = max(MIN_BRUSH_PX, radius_px * this)
SENSOR_MAX_CORRECTION_TEXTURE = 8.0   # skip correction if dust spot lands on a busy area in this frame
SENSOR_MAX_SOURCE_TEXTURE = 8.0       # skip correction if no clean healing source found in any direction


# ===================================================================
# Dust detection
# ===================================================================

def detect_dust_spots(image_path, collect_rejects=False):
    """Detect bright dust spots in an image.

    Returns (spots, rejected_candidates, error_msg).
    spots: list of dicts with keys: cx, cy (pixel coords), radius_px, area, contrast, texture, excess_sat, spot_sat.
    rejected_candidates: list of structured reject dicts (only populated when collect_rejects=True).
    error_msg: string on failure, None on success.
    """
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

    # Find optimal healing sources for each spot
    if spots:
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        L_f32 = img_lab[:, :, 0].astype(np.float32)
        for spot in spots:
            src_cx, src_cy = find_healing_source(
                spot["cx"], spot["cy"], spot["radius_px"], spot["brush_radius_px"],
                img_lab, L_f32, local_std, spots, width, height)
            spot["src_cx"] = src_cx
            spot["src_cy"] = src_cy

    return spots, rejected_candidates, None, local_std


# ===================================================================
# Source point detection
# ===================================================================

def find_healing_source(cx, cy, radius_px, brush_radius_px, img_lab, L_f32, local_std, all_spots, width, height):
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
# ML detection helpers
# ===================================================================

def _compute_candidate_features(label_id, labels, stats, centroids,
                                 diff, gray, local_bg, bg_gradient, local_std, saturation,
                                 threshold, bright_ref, width, height):
    """Compute all features for a single connected component without soft-filter decisions.

    Applies only hard gates (size 6-800, shape aspect/compactness/solidity/circularity,
    minimum contrast floor, excess saturation).  All soft filters (texture, ratio, context,
    votes, dim, embedded, edge, dark_bg) are intentionally skipped so the ML classifier
    can make those decisions.

    Returns a feature dict (keys = FEATURE_NAMES + cx/cy/area/radius_px/brush_radius_px/
    threshold/contrast/excess_sat/spot_sat/context_texture/texture), or None on hard-filter
    rejection.
    """
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


def detect_dust_spots_ml(image_path, ml_model, scaler, collect_rejects=False):
    """ML-assisted dust detection (post-filter mode).

    Strategy:
      1. Run the standard rule-based pipeline (detect_dust_spots) to get high-precision
         accepted spots.  This handles the bulk of detection and filtering.
      2. Apply the ML post-filter to those accepted spots to remove false positives
         that slipped through the rule-based filters.  The ML model was trained on
         user-annotated FPs (label=0) and confirmed dust (label=1).

    Returns (spots, rejected_candidates, error_msg, local_std) — same signature as
    detect_dust_spots().  Healing source positions are NOT set here;
    process_one_image() adds them.
    """
    # --- Phase 1: standard rule-based detection ---
    std_spots, std_rejects, error, local_std = detect_dust_spots(image_path, collect_rejects)
    if error:
        return None, std_rejects, error, local_std
    if std_spots is None:
        std_spots = []

    # --- Phase 2: ML post-filter on accepted spots ---
    if std_spots:
        X_std = np.array([_spot_to_features(s) for s in std_spots], dtype=np.float32)
        X_std_scaled = scaler.transform(X_std)
        probas_std = ml_model.predict_proba(X_std_scaled)[:, 1]
        spots = [s for s, p in zip(std_spots, probas_std) if p >= ML_POSTFILTER_THRESHOLD]
        n_removed = len(std_spots) - len(spots)
        if n_removed > 0:
            print(f"  [ML] post-filter: removed {n_removed}/{len(std_spots)} FPs "
                  f"(threshold={ML_POSTFILTER_THRESHOLD})")
    else:
        spots = []

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


def detect(image_path, collect_rejects=False, ml_model_path=None):
    """Unified detection entry point — uses ML post-filter by default.

    If ml_model_path is given, loads that model.  Otherwise loads the default
    dust_ml_model.pkl next to this script.  Falls back to pure rule-based
    detection when no model file exists.

    Returns (spots, rejected_candidates, error_msg, local_std).
    """
    model, scaler = load_ml_model(ml_model_path)
    if model is not None:
        return detect_dust_spots_ml(image_path, model, scaler,
                                    collect_rejects=collect_rejects)
    return detect_dust_spots(image_path, collect_rejects=collect_rejects)


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

    # ashift input dims ≈ fullwidth × fullheight (exact when correction is small)
    W_in = fullwidth
    H_in = fullheight

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

    for spot in spots:
        # Normalize coordinates to [0, 1] in export space
        norm_cx = spot["cx"] / image_width
        norm_cy = spot["cy"] / image_height

        # Transform to original image space (undo crop, flip, ashift)
        norm_cx, norm_cy = _export_to_original(
            norm_cx, norm_cy, flip, crop, ashift_params, image_width, image_height)

        # Brush border: darktable renders as border * MIN(pipe_w, pipe_h)
        border = spot["brush_radius_px"] / border_scale
        spot["radius_norm"] = border   # persisted to {stem}_debug_spots.json for UI display

        # Heal source: use auto-detected optimal position if available, else fixed offset
        if "src_cx" in spot and "src_cy" in spot:
            src_norm_x = spot["src_cx"] / image_width
            src_norm_y = spot["src_cy"] / image_height
            src_x, src_y = _export_to_original(
                src_norm_x, src_norm_y, flip, crop, ashift_params, image_width, image_height)
            src_x = max(0.0, min(1.0, src_x))
            src_y = max(0.0, min(1.0, src_y))
        else:
            src_x = max(0.0, min(1.0, norm_cx + HEAL_SOURCE_OFFSET_X))
            src_y = max(0.0, min(1.0, norm_cy + HEAL_SOURCE_OFFSET_Y))

        mask_id = id_gen.next_id()
        brush_ids.append(mask_id)

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


def write_debug_spots_json(results, image_paths_by_stem, output_dir):
    """Write per-image {stem}_debug_spots.json files for the debug UI.

    results: list of (filename_no_ext, spots_or_None, rejected_candidates, img_dims,
                      error_or_None, xmp_data_or_None).
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
        out_path = os.path.join(output_dir, f"{filename}_debug_spots.json")
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)

    print(f"Debug spots written to: {output_dir} ({len(results)} file(s))")


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


def _sensor_find_source(gray, cx_px, cy_px, radius_px):
    """Find the smoothest healing source for a sensor dust spot.

    Samples 8 directions at 2.5× radius distance from the spot centre and
    picks the patch with the lowest median local-std texture.

    Returns (src_cx, src_cy, texture) or None if every direction exceeds
    SENSOR_MAX_SOURCE_TEXTURE (no clean source available).
    """
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


def detect_sensor_dust_candidates(image_path):
    """Find sensor-dust candidate blobs using Difference-of-Gaussians + local maxima.

    DoG cancels smooth background gradients (sky, bokeh) while preserving blob-scale
    brightness peaks. Local maxima detection avoids the connected-component fusion
    problem where an entire gradient region merges into one large blob.

    Returns (candidates, error_msg) where candidates is a list of dicts with keys:
      cx, cy (export pixels), radius_px, brush_radius_px, contrast, area, circularity.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return [], f"Cannot read {image_path}"
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    min_dim = min(w, h)
    sigma_inner = max(2.0, min_dim * SENSOR_SIGMA_INNER_FRAC)
    sigma_outer = max(20.0, min_dim * SENSOR_SIGMA_OUTER_FRAC)
    min_radius = max(5, int(min_dim * SENSOR_MIN_RADIUS_FRAC))
    max_radius = int(min_dim * SENSOR_MAX_RADIUS_FRAC)
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

    # Local maxima with NMS window = min_radius*4: finds peaks separated by at least 2*min_radius
    nms_diam = max(5, min_radius * 4) | 1
    nms_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nms_diam, nms_diam))
    dog_dilated = cv2.dilate(dog, nms_se)
    peak_mask = (dog >= dog_dilated - 0.5) & (dog > SENSOR_DOG_MIN_CONTRAST)
    peak_ys, peak_xs = np.where(peak_mask)

    print(f"  DoG peaks above {SENSOR_DOG_MIN_CONTRAST}: {len(peak_ys)}")

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

    # Grid-based peak selection: keep the strongest DoG peak per max_radius-sized cell.
    # max_radius controls grid density (~96 cells for typical images), decoupled from the
    # blob size limit (max_blob_r) so we get a manageable candidate count.
    grid_rows = max(1, h // max_radius)
    grid_cols = max(1, w // max_radius)
    cell_idx = (np.minimum((peak_ys * grid_rows // h).astype(np.int32), grid_rows - 1)
                * grid_cols
                + np.minimum((peak_xs * grid_cols // w).astype(np.int32), grid_cols - 1))
    order = np.lexsort((-peak_vals, cell_idx))  # sort: primary=cell, secondary=strongest first
    _, first_in_cell = np.unique(cell_idx[order], return_index=True)
    sel = order[first_in_cell]

    candidates = []
    for k in sel:
        cx, cy = float(peak_xs[k]), float(peak_ys[k])
        peak_val = float(peak_vals[k])

        blob_r = _sensor_blob_radius(dog, cx, cy, min_radius, max_blob_r, peak_val)
        # Reject if ring sampling never found the drop (not a genuine small bounded blob)
        if blob_r < min_radius or blob_r >= max_blob_r:
            continue

        area = int(math.pi * blob_r ** 2)
        brush_r = max(MIN_BRUSH_PX, blob_r * SENSOR_BRUSH_SCALE)
        candidates.append({
            "cx": cx, "cy": cy,
            "radius_px": float(blob_r), "brush_radius_px": brush_r,
            "contrast": peak_val, "area": area, "circularity": 1.0,
        })

    print(f"  Found {len(candidates)} candidate(s)")
    return candidates, None


def find_sensor_dust_consensus(per_image_results, transforms):
    """Identify dust positions common across multiple frames (sensor dust signature).

    per_image_results: list of (stem, candidates, img_w, img_h)
    transforms: dict from parse_transform_params()

    Candidates from each frame are mapped to normalized full-frame coordinates, then
    clustered. Clusters present in >= SENSOR_DUST_MIN_FRAMES distinct frames are
    confirmed as sensor dust.

    Returns list of dicts: {cx_norm, cy_norm, radius_norm, n_frames} in full-frame space.
    """
    from collections import defaultdict

    all_pts = []  # (cx_ff, cy_ff, radius_norm, stem_idx)
    for stem_idx, (stem, candidates, img_w, img_h) in enumerate(per_image_results):
        if not candidates or not img_w or not img_h:
            continue
        t = transforms.get(stem, {"flip": 0, "crop": (0.0, 0.0, 1.0, 1.0), "ashift": None})
        flip = t["flip"]
        crop = t["crop"]
        crop_l, crop_t, crop_r, crop_b = crop
        crop_scale = min(crop_r - crop_l, crop_b - crop_t)
        min_export = min(img_w, img_h)
        for c in candidates:
            cx_n = c["cx"] / img_w
            cy_n = c["cy"] / img_h
            cx_ff, cy_ff = _export_to_original(cx_n, cy_n, flip, crop)
            # Radius in full-frame normalized: scale from export pixels
            r_norm = c["radius_px"] / max(min_export, 1) * max(crop_scale, 1e-6)
            all_pts.append((cx_ff, cy_ff, r_norm, stem_idx))

    if not all_pts:
        return []

    # Union-find clustering
    n = len(all_pts)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    for i in range(n):
        for j in range(i + 1, n):
            dx = all_pts[i][0] - all_pts[j][0]
            dy = all_pts[i][1] - all_pts[j][1]
            if math.sqrt(dx * dx + dy * dy) < SENSOR_CLUSTER_RADIUS_NORM:
                union(i, j)

    clusters = defaultdict(list)
    for i, pt in enumerate(all_pts):
        clusters[find(i)].append(pt)

    min_frames = SENSOR_DUST_MIN_FRAMES
    confirmed = []
    for members in clusters.values():
        frame_indices = {m[3] for m in members}
        if len(frame_indices) >= min_frames:
            cx_mean = sum(m[0] for m in members) / len(members)
            cy_mean = sum(m[1] for m in members) / len(members)
            r_mean  = sum(m[2] for m in members) / len(members)
            confirmed.append({
                "cx_norm": cx_mean, "cy_norm": cy_mean,
                "radius_norm": r_mean, "n_frames": len(frame_indices),
            })

    return confirmed


def process_one_sensor_image(args):
    """Detect sensor dust candidates in one image. Top-level for multiprocessing pickling."""
    (image_path,) = args
    filename = Path(image_path).stem
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
            else:
                for c in candidates:
                    print(f"    center=({c['cx']:.1f},{c['cy']:.1f}) "
                          f"radius={c['radius_px']:.1f}px contrast={c['contrast']:.1f}")
    except Exception as e:
        error = str(e)
        buf.write(f"  EXCEPTION: {e}\n")
    return (filename, candidates, img_dims, error, buf.getvalue())


def run_sensor_dust_mode(image_paths, transforms, output_dir):
    """Full sensor dust pipeline: per-image candidates → cross-frame consensus → XMP data.

    Returns True if any errors occurred.
    """
    n_workers = min(cpu_count(), len(image_paths))
    print(f"Sensor dust mode: {len(image_paths)} image(s), {n_workers} worker(s)", flush=True)

    per_image_results = []
    any_errors = False
    args_list = [(p,) for p in image_paths]

    with Pool(processes=n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_one_sensor_image, args_list), 1):
            filename, candidates, img_dims, error, log_output = result
            print(log_output, end="", flush=True)
            if error:
                any_errors = True
            per_image_results.append((filename, candidates, img_dims[0], img_dims[1]))
            print(f"PROGRESS|{i}|{len(image_paths)}", flush=True)

    consensus_spots = find_sensor_dust_consensus(per_image_results, transforms)
    print(f"\nSensor dust consensus: {len(consensus_spots)} spot(s) confirmed across frames")
    for sd in consensus_spots:
        print(f"  pos=({sd['cx_norm']:.4f},{sd['cy_norm']:.4f}) "
              f"radius_norm={sd['radius_norm']:.4f} seen_in={sd['n_frames']} frames")

    image_paths_by_stem = {Path(p).stem: p for p in image_paths}

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

        spots = []
        skipped_texture = 0
        skipped_no_source = 0
        for sd in consensus_spots:
            proj = _original_to_export(sd["cx_norm"], sd["cy_norm"], flip, crop, img_w, img_h)
            if proj is None:
                continue  # sensor dust falls outside this frame's crop
            cx_px, cy_px = proj
            radius_px = sd["radius_norm"] * max(min_export, 1) / max(crop_scale, 1e-6)
            brush_r = max(MIN_BRUSH_PX, radius_px * SENSOR_BRUSH_SCALE)

            if gray_export is not None:
                # Skip if the spot landed on a busy area in this frame
                spot_tex = _sensor_spot_texture(gray_export, cx_px, cy_px, radius_px)
                if spot_tex > SENSOR_MAX_CORRECTION_TEXTURE:
                    skipped_texture += 1
                    print(f"  [{stem}] skip ({cx_px:.0f},{cy_px:.0f}) "
                          f"spot_texture={spot_tex:.1f} > {SENSOR_MAX_CORRECTION_TEXTURE}")
                    continue

                # Find the smoothest healing source; skip if none is clean enough
                src_result = _sensor_find_source(gray_export, cx_px, cy_px, radius_px)
                if src_result is None:
                    skipped_no_source += 1
                    print(f"  [{stem}] skip ({cx_px:.0f},{cy_px:.0f}) "
                          f"no clean source (all directions > {SENSOR_MAX_SOURCE_TEXTURE})")
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
                "texture": 0.0, "context_texture": 0.0,
                "excess_sat": 0.0, "spot_sat": 0.0,
            })

        total_skipped = skipped_texture + skipped_no_source
        if total_skipped:
            print(f"  [{stem}] {total_skipped}/{len(consensus_spots)} skipped "
                  f"({skipped_texture} busy area, {skipped_no_source} no clean source)")

        xmp_data = None
        if spots and img_w and img_h:
            xmp_data = generate_xmp_data_for_spots(
                spots, img_w, img_h, flip=flip, crop=crop, ashift_params=t.get("ashift"))
        results.append((stem, spots, [], (img_w, img_h), None, xmp_data))

    results.sort(key=lambda r: r[0])
    write_dust_results(results, output_dir)
    return any_errors


# ===================================================================
# Per-image worker (top-level so it's picklable on Windows spawn)
# ===================================================================

def process_one_image(args):
    """Detect dust in one image and return results + captured log output.

    Must be a top-level function (not nested inside main) so multiprocessing
    can pickle it on Windows, which uses the 'spawn' start method.
    """
    image_path, transforms, collect_rejects, ml_model_path = args
    filename = Path(image_path).stem

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
                ml_model_path=ml_model_path)
            if error:
                print(f"  ERROR: {error}")
                return (filename, None, [], (0, 0), error, None, buf.getvalue())

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
                    src_cx, src_cy = find_healing_source(
                        spot["cx"], spot["cy"], spot["radius_px"], spot["brush_radius_px"],
                        img_lab, L_f32, local_std, spots, width, height)
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

    return (filename, spots, rejected_candidates, img_dims, error, xmp_data, buf.getvalue())


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
        had_errors = run_sensor_dust_mode(image_paths, transforms, output_dir)
        sys.exit(1 if had_errors else 0)

    image_paths_by_stem = {Path(p).stem: p for p in image_paths}
    results = []
    any_errors = False

    n_workers = min(cpu_count(), len(image_paths))
    print(f"Running {n_workers} parallel worker(s) for {len(image_paths)} image(s)", flush=True)

    args_list = [(p, transforms, debug_ui, ml_model_path) for p in image_paths]

    with Pool(processes=n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_one_image, args_list), 1):
            filename, spots, rejected_candidates, img_dims, error, xmp_data, log_output = result
            # Print worker log sequentially (no interleaving between images)
            print(log_output, end="", flush=True)
            if error:
                any_errors = True
            results.append((filename, spots, rejected_candidates, img_dims, error, xmp_data))
            # Progress sentinel read by Lua via io.popen()
            print(f"PROGRESS|{i}|{len(image_paths)}", flush=True)

    # Sort for deterministic output order (imap_unordered returns in completion order)
    results.sort(key=lambda r: r[0])

    # Write results file
    results_path = write_dust_results(results, output_dir)
    print(f"\nResults written to: {results_path}")

    if debug_ui:
        write_debug_spots_json(results, image_paths_by_stem, output_dir)
        import subprocess
        debug_ui_script = Path(__file__).parent / "debug_ui.py"
        print(f"Launching debug UI: {debug_ui_script}", flush=True)
        subprocess.Popen([sys.executable, str(debug_ui_script), output_dir])

    sys.exit(1 if any_errors else 0)


if __name__ == "__main__":
    main()
