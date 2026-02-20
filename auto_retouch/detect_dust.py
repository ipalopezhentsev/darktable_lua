"""
Dust spot detection for DSLR-scanned film negatives.

Detects bright dust particles on inverted negatives and generates
XMP-ready binary data for darktable's retouch module.

Usage:
    python detect_dust.py [--no-vis] <image1.jpg> [image2.jpg ...]

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
# NOTE: No MIN_EXCESS_SATURATION — dust on colorful content is genuinely desaturated (excess_sat can reach -100+)
# NOTE: No MAX_SPOT_SATURATION — color film negatives produce dust spots with spot_sat 100-230 (film emulsion
#       color characteristics), so absolute saturation is not a reliable discriminator.
MAX_CONTEXT_TEXTURE = 9.0      # max median local_std across a 200px radius from spot center.
                               # Dust sits on smooth backgrounds (sky, walls: median ≈ 2-7).
                               # Crowd/foliage FPs sit in texured scenes (median ≈ 10-25).
                               # Threshold at 9 separates smooth-background dust from busy-scene FPs.
LARGE_SPOT_AREA_THRESHOLD = 300  # spots larger than this require higher contrast
LARGE_SPOT_MIN_CONTRAST = 60     # min contrast for large spots — avoids pale foggy blobs
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
# Darktable binary format constants
# ---------------------------------------------------------------------------
BRUSH_DENSITY = 1.0
BRUSH_HARDNESS = 0.66
BRUSH_STATE_NORMAL = 1
BRUSH_DELTA = 0.00001          # offset between 2 brush points (forms a "dot")
BRUSH_CTRL_OFFSET = 0.000003   # bezier control handle offset from corner
MIN_BRUSH_BORDER = 0.002306    # minimum brush radius in normalized coords
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
SOURCE_COLOR_WEIGHT = 1.0         # weight for Lab color distance in scoring
SOURCE_TEXTURE_WEIGHT = 3.0       # weight for texture mismatch in scoring
SOURCE_GRID_STEP = 8              # grid step for candidate sampling (pixels)

RETOUCH_ALGO_HEAL = 2
RETOUCH_MOD_VERSION = 3
MAX_FORMS = 300                # darktable's maximum form slots

# Known-good blendop_params template from a real darktable XMP (420 bytes uncompressed)
# Only the mask_id field at offset 24 needs to be replaced per-image
BLENDOP_TEMPLATE_ENCODED = "gz08eJxjYGBgYAFiCQYYOOEEIjd6dmXCRFgZMAEjFjEGhgZ7CB6pfOygYtaVAyCMi48L/AcCEA0Ak0kpjg=="


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
                "texture": 0, "ratio": 0, "context": 0, "votes": 0, "color": 0,
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
        ring_inner = max(int(radius_px * 1.5), 2)
        ring_outer = max(int(radius_px * 3), ring_inner + TEXTURE_KERNEL)
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
        spots.append({
            "cx": float(cx),
            "cy": float(cy),
            "radius_px": radius_px,
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
        for spot in spots:
            src_cx, src_cy = find_healing_source(
                spot["cx"], spot["cy"], spot["radius_px"],
                img_lab, local_std, spots, width, height)
            spot["src_cx"] = src_cx
            spot["src_cy"] = src_cy

    return spots, rejected_candidates, None, local_std


# ===================================================================
# Source point detection
# ===================================================================

def find_healing_source(cx, cy, radius_px, img_lab, local_std, all_spots, width, height):
    """Find the best healing source position for a dust spot at (cx, cy).

    Searches the area around the spot for a region with similar color and
    texture to the spot's surroundings, avoiding other detected dust spots.

    Returns (src_cx, src_cy) in pixel coordinates.
    """
    icx = max(0, min(int(round(cx)), width - 1))
    icy = max(0, min(int(round(cy)), height - 1))

    # Measure reference color + texture in a ring around the spot
    # (same ring used by the texture filter — avoids sampling inside the bright dust)
    ring_inner = max(int(radius_px * 1.5), 2)
    ring_outer = max(int(radius_px * 3), ring_inner + 5)
    y1 = max(0, icy - ring_outer)
    y2 = min(height, icy + ring_outer + 1)
    x1 = max(0, icx - ring_outer)
    x2 = min(width, icx + ring_outer + 1)
    py, px = np.ogrid[y1 - icy:y2 - icy, x1 - icx:x2 - icx]
    dist_sq = px * px + py * py
    ring_mask = (dist_sq >= ring_inner ** 2) & (dist_sq <= ring_outer ** 2)
    patch_lab = img_lab[y1:y2, x1:x2]
    if ring_mask.any():
        ref_L = float(np.mean(patch_lab[:, :, 0][ring_mask]))
        ref_a = float(np.mean(patch_lab[:, :, 1][ring_mask]))
        ref_b = float(np.mean(patch_lab[:, :, 2][ring_mask]))
        ref_texture = float(np.median(local_std[y1:y2, x1:x2][ring_mask]))
    else:
        ref_L = float(img_lab[icy, icx, 0])
        ref_a = float(img_lab[icy, icx, 1])
        ref_b = float(img_lab[icy, icx, 2])
        ref_texture = float(local_std[icy, icx])

    # Annulus search bounds
    search_inner = max(int(radius_px * SOURCE_SEARCH_INNER_FACTOR),
                       SOURCE_SEARCH_MIN_RADIUS // 2)
    search_outer = max(
        min(int(radius_px * 10), SOURCE_SEARCH_MAX_RADIUS),
        SOURCE_SEARCH_MIN_RADIUS,
    )

    # Default fallback: same fixed offset as before
    dim = max(width, height)
    default_src_x = min(width - 1, max(0, cx + HEAL_SOURCE_OFFSET_X * dim))
    default_src_y = min(height - 1, max(0, cy + HEAL_SOURCE_OFFSET_Y * dim))

    # Forbidden zones: other accepted dust spots (avoid placing source on another dust spot)
    forbidden = [
        (int(round(s["cx"])), int(round(s["cy"])), max(int(s["radius_px"]) * 2, 4))
        for s in all_spots
        if not (abs(s["cx"] - cx) < 0.5 and abs(s["cy"] - cy) < 0.5)
    ]

    best_score = float("inf")
    best_src = (default_src_x, default_src_y)

    step = max(SOURCE_GRID_STEP, max(int(radius_px), 2))

    for sy in range(max(0, icy - search_outer), min(height, icy + search_outer + 1), step):
        for sx in range(max(0, icx - search_outer), min(width, icx + search_outer + 1), step):
            dx = sx - icx
            dy = sy - icy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < search_inner or dist > search_outer:
                continue

            # Skip if overlapping another dust spot
            skip = False
            for fx, fy, fr in forbidden:
                if math.hypot(sx - fx, sy - fy) < fr:
                    skip = True
                    break
            if skip:
                continue

            # Color score (Lab ΔE-like distance)
            cand_L = float(img_lab[sy, sx, 0])
            cand_a = float(img_lab[sy, sx, 1])
            cand_b = float(img_lab[sy, sx, 2])
            color_dist = math.sqrt(
                (cand_L - ref_L) ** 2 +
                (cand_a - ref_a) ** 2 +
                (cand_b - ref_b) ** 2
            )

            # Texture score
            texture_dist = abs(float(local_std[sy, sx]) - ref_texture)

            score = SOURCE_COLOR_WEIGHT * color_dist + SOURCE_TEXTURE_WEIGHT * texture_dist
            if score < best_score:
                best_score = score
                best_src = (float(sx), float(sy))

    return best_src


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

    Returns dict: filename -> {"flip": int, "crop": (L, T, R, B)}.
    """
    transforms = {}
    if not os.path.isfile(params_file):
        return transforms

    with open(params_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: filename|flip=N|crop=L,T,R,B
            parts = line.split("|")
            if len(parts) < 3:
                continue
            filename = parts[0]
            flip = 0
            crop = (0.0, 0.0, 1.0, 1.0)
            for part in parts[1:]:
                if part.startswith("flip="):
                    flip = int(part[5:])
                elif part.startswith("crop="):
                    vals = part[5:].split(",")
                    if len(vals) == 4:
                        crop = tuple(float(v) for v in vals)
            transforms[filename] = {"flip": flip, "crop": crop}

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


def _export_to_original(cx, cy, flip, crop):
    """Transform coordinates from export space to original image space.

    Pipeline order: original -> flip -> crop -> export.
    Inverse: undo crop, then undo flip.
    """
    crop_l, crop_t, crop_r, crop_b = crop

    # 1. Undo crop (export [0,1] -> pre-crop space)
    cx = crop_l + cx * (crop_r - crop_l)
    cy = crop_t + cy * (crop_b - crop_t)

    # 2. Undo flip (pre-crop -> original space)
    if flip == 1:    # vertical
        cy = 1.0 - cy
    elif flip == 2:  # horizontal
        cx = 1.0 - cx

    return cx, cy


def generate_xmp_data_for_spots(spots, image_width, image_height,
                                flip=0, crop=(0.0, 0.0, 1.0, 1.0)):
    """Generate all XMP-ready data for a list of detected spots.

    Returns dict with keys: brushes, group, retouch_params, blendop_params.
    Each brush: {mask_id, mask_points, mask_src, mask_nb}.

    flip: 0=none, 1=vertical, 2=horizontal
    crop: (left, top, right, bottom) in original image [0,1] space
    """
    id_gen = MaskIdGenerator()

    brushes = []
    brush_ids = []

    for spot in spots:
        # Normalize coordinates to [0, 1] in export space
        norm_cx = spot["cx"] / image_width
        norm_cy = spot["cy"] / image_height

        # Transform to original image space (undo crop + flip)
        norm_cx, norm_cy = _export_to_original(norm_cx, norm_cy, flip, crop)

        # Brush border proportional to detected spot radius, with minimum
        norm_radius = spot["radius_px"] / max(image_width, image_height)
        border = max(MIN_BRUSH_BORDER, norm_radius * 2.0)

        # Heal source: use auto-detected optimal position if available, else fixed offset
        if "src_cx" in spot and "src_cy" in spot:
            src_norm_x = spot["src_cx"] / image_width
            src_norm_y = spot["src_cy"] / image_height
            src_x, src_y = _export_to_original(src_norm_x, src_norm_y, flip, crop)
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
# Visualization
# ===================================================================

def save_visualization(image_path, spots, output_path):
    """Save a copy of the image with red circles at detected spots."""
    img = cv2.imread(str(image_path))
    if img is None:
        return

    for spot in spots:
        cx = int(round(spot["cx"]))
        cy = int(round(spot["cy"]))
        r = max(5, int(round(spot["radius_px"] * 3)))  # enlarge for visibility
        cv2.circle(img, (cx, cy), r, (0, 0, 255), 2)

    cv2.imwrite(str(output_path), img)


# ===================================================================
# Results output
# ===================================================================

def write_debug_spots_json(results, image_paths_by_stem, output_dir):
    """Write debug_spots.json for the debug UI.

    results: list of (filename_no_ext, spots_or_None, rejected_candidates, img_dims,
                      error_or_None, xmp_data_or_None).
    """
    import json

    class NumpyEncoder(json.JSONEncoder):
        """Convert numpy scalars to plain Python types for JSON serialization."""
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    images_out = []
    for filename, spots, rejected_candidates, img_dims, error, xmp_data in results:
        w, h = img_dims if img_dims else (0, 0)
        images_out.append({
            "stem": filename,
            "image_path": str(image_paths_by_stem.get(filename, "")),
            "width": int(w),
            "height": int(h),
            "detected": spots or [],
            "rejected": rejected_candidates or [],
        })
    constants = {k: v for k, v in globals().items()
                 if k.isupper() and isinstance(v, (int, float))}
    data = {"images": images_out, "constants": constants}
    out_path = os.path.join(output_dir, "debug_spots.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    print(f"Debug spots written to: {out_path}")


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
# Per-image worker (top-level so it's picklable on Windows spawn)
# ===================================================================

def process_one_image(args):
    """Detect dust in one image and return results + captured log output.

    Must be a top-level function (not nested inside main) so multiprocessing
    can pickle it on Windows, which uses the 'spawn' start method.
    """
    image_path, transforms, save_vis, collect_rejects = args
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

            spots, rejected_candidates, error, local_std = detect_dust_spots(image_path, collect_rejects)
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
                for spot in spots:
                    src_cx, src_cy = find_healing_source(
                        spot["cx"], spot["cy"], spot["radius_px"],
                        img_lab, local_std, spots, width, height)
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

            if save_vis and spots:
                vis_path = Path(image_path).with_name(f"{filename}_dust_overlay.jpg")
                save_visualization(image_path, spots, vis_path)
                print(f"  Visualization saved: {vis_path}")

            if spots:
                t = transforms.get(filename, {"flip": 0, "crop": (0.0, 0.0, 1.0, 1.0)})
                print(f"  Transform: flip={t['flip']}, crop={t['crop']}")
                xmp_data = generate_xmp_data_for_spots(
                    spots, width, height, flip=t["flip"], crop=t["crop"])
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
        print("Usage: detect_dust.py [--no-vis] [--debug-ui] <image1.jpg> [image2.jpg ...]")
        sys.exit(1)

    save_vis = True
    debug_ui = False
    if "--no-vis" in args:
        save_vis = False
        args.remove("--no-vis")
    if "--debug-ui" in args:
        debug_ui = True
        args.remove("--debug-ui")

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

    image_paths_by_stem = {Path(p).stem: p for p in image_paths}
    results = []
    any_errors = False

    n_workers = min(cpu_count(), len(image_paths))
    print(f"Running {n_workers} parallel worker(s) for {len(image_paths)} image(s)", flush=True)

    args_list = [(p, transforms, save_vis, debug_ui) for p in image_paths]

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
