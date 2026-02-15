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
import math
import struct
import zlib
import base64
import time
import random
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Detection constants (conservative defaults)
# ---------------------------------------------------------------------------
LOCAL_BG_KERNEL = 101          # Gaussian blur kernel for local background
NOISE_THRESHOLD_MULTIPLIER = 6.0  # spots must be this many std devs above background
MIN_ABSOLUTE_THRESHOLD = 30.0  # minimum brightness difference regardless of noise
MIN_SPOT_AREA = 4              # minimum pixels (reject single-pixel noise)
MAX_SPOT_AREA = 200            # maximum pixels (~14x14 at full res)
MIN_ASPECT_RATIO = 0.5         # bounding box aspect ratio (reject elongated fibers)
MIN_COMPACTNESS = 0.4          # area / bbox_area (reject irregular shapes)
MAX_SPOTS = 200                # cap: sort by contrast, take the most obvious ones

# ---------------------------------------------------------------------------
# Darktable binary format constants
# ---------------------------------------------------------------------------
BRUSH_DENSITY = 1.0
BRUSH_HARDNESS = 0.66
BRUSH_STATE_NORMAL = 1
BRUSH_DELTA = 0.00001          # offset between 2 brush points (forms a "dot")
BRUSH_CTRL_OFFSET = 0.000003   # bezier control handle offset from corner
MIN_BRUSH_BORDER = 0.002306    # minimum brush radius in normalized coords
HEAL_SOURCE_OFFSET_X = 0.01   # heal source offset from spot center
HEAL_SOURCE_OFFSET_Y = -0.01

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

def detect_dust_spots(image_path):
    """Detect bright dust spots in an image.

    Returns list of dicts with keys: cx, cy (pixel coords), radius_px, area, contrast.
    Returns (None, error_msg) on failure.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None, f"Failed to load image: {image_path}"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Local background: large Gaussian blur smooths out dust but preserves
    # gradual brightness variations (vignetting, subject brightness)
    local_bg = cv2.GaussianBlur(gray, (LOCAL_BG_KERNEL, LOCAL_BG_KERNEL), 0)

    # Difference: positive values = brighter than background (dust on inverted negatives)
    diff = gray.astype(np.float32) - local_bg.astype(np.float32)

    # Threshold based on image noise level
    noise_std = np.std(diff)
    threshold = max(MIN_ABSOLUTE_THRESHOLD, noise_std * NOISE_THRESHOLD_MULTIPLIER)

    binary = (diff > threshold).astype(np.uint8)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    spots = []
    for label_id in range(1, num_labels):  # skip background (0)
        area = stats[label_id, cv2.CC_STAT_AREA]
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]

        # Size filter
        if area < MIN_SPOT_AREA or area > MAX_SPOT_AREA:
            continue

        # Circularity: bounding box aspect ratio
        aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0
        if aspect < MIN_ASPECT_RATIO:
            continue

        # Compactness: how much of the bounding box is filled
        bbox_area = w * h
        compactness = area / bbox_area if bbox_area > 0 else 0
        if compactness < MIN_COMPACTNESS:
            continue

        # Brightness contrast check
        cx, cy = centroids[label_id]
        icx, icy = int(round(cx)), int(round(cy))
        icx = max(0, min(icx, width - 1))
        icy = max(0, min(icy, height - 1))
        contrast = float(diff[icy, icx])
        if contrast < threshold * 0.8:
            continue

        spots.append({
            "cx": float(cx),
            "cy": float(cy),
            "radius_px": math.sqrt(area / math.pi),
            "area": area,
            "contrast": contrast,
        })

    return spots, None


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

        # Heal source: offset from spot center in original space
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

def write_dust_results(results, output_dir):
    """Write dust_results.txt in the output directory.

    results: list of (filename_no_ext, spots_or_None, error_or_None, xmp_data_or_None).
    """
    output_path = Path(output_dir) / "dust_results.txt"
    with open(output_path, "w") as f:
        for filename, spots, error, xmp_data in results:
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
# Main
# ===================================================================

def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: detect_dust.py [--no-vis] <image1.jpg> [image2.jpg ...]")
        sys.exit(1)

    save_vis = True
    if "--no-vis" in args:
        save_vis = False
        args.remove("--no-vis")

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

    results = []
    any_errors = False

    for image_path in image_paths:
        filename = Path(image_path).stem
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print(f"{'='*60}")

        # Detect spots
        spots, error = detect_dust_spots(image_path)
        if error:
            print(f"  ERROR: {error}")
            results.append((filename, None, error, None))
            any_errors = True
            continue

        print(f"  Image loaded successfully")
        img = cv2.imread(str(image_path))
        height, width = img.shape[:2]
        print(f"  Dimensions: {width} x {height}")
        print(f"  Detected {len(spots)} dust spot(s) before filtering")

        # Sort by contrast (strongest first) and cap at MAX_SPOTS
        spots.sort(key=lambda s: s["contrast"], reverse=True)
        if len(spots) > MAX_SPOTS:
            print(f"  Capping from {len(spots)} to {MAX_SPOTS} strongest spots")
            spots = spots[:MAX_SPOTS]

        for i, spot in enumerate(spots):
            print(
                f"    Spot {i}: center=({spot['cx']:.1f}, {spot['cy']:.1f}) "
                f"radius={spot['radius_px']:.1f}px area={spot['area']}px "
                f"contrast={spot['contrast']:.1f}"
            )

        # Save visualization (before XMP generation so it always works)
        if save_vis and spots:
            vis_path = Path(image_path).with_name(f"{filename}_dust_overlay.jpg")
            save_visualization(image_path, spots, vis_path)
            print(f"  Visualization saved: {vis_path}")

        # Generate XMP data
        xmp_data = None
        if spots:
            t = transforms.get(filename, {"flip": 0, "crop": (0.0, 0.0, 1.0, 1.0)})
            print(f"  Transform: flip={t['flip']}, crop={t['crop']}")
            xmp_data = generate_xmp_data_for_spots(
                spots, width, height, flip=t["flip"], crop=t["crop"])
            print(f"  Generated XMP data: {len(xmp_data['brushes'])} brush masks")

        results.append((filename, spots, None, xmp_data))

    # Write results file
    results_path = write_dust_results(results, output_dir)
    print(f"\nResults written to: {results_path}")

    sys.exit(1 if any_errors else 0)


if __name__ == "__main__":
    main()
