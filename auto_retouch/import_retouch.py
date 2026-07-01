"""Import existing darktable retouch masks from an XMP as ground-truth dust.

This is the INVERSE of `detect_dust.generate_xmp_data_for_spots`: it decodes the
brush / circle forms of a frame's ACTIVE film-dust retouch instances out of the
XMP and turns them back into export-space spot dicts, which are then seeded as
`missed_dust` / `missed_strokes` annotations for the debug UI's "edit existing
retouch" (import-GT) flow.

Why: auto_retouch can be re-invoked on frames that already carry retouch (drawn
by hand, or applied by us and the temp folder since deleted). The Lua side
exports the *un-healed* frame and launches the debug UI; this module reconstructs
the user's existing healing as the starting ground truth so they can keep editing
(combined with a fresh detection), and the kept temp folder becomes calibration
ground truth.

Sensor-dust instances use a different algorithm and are EXCLUDED — identified by
their `multi_name` label, passed in from Lua (so localization matches). Everything
that is NOT sensor dust (film dust + any other manual instances; there can be
several from consecutive applications) is imported, merged and de-duplicated.

All coordinate transforms are REUSED from detect_dust (raw-buffer normalized →
export pixels via `detect_dust._original_to_export`, including the forward ashift
homography); no transform math lives here.
"""

import os
import re
import json
import struct
import base64
import zlib
from pathlib import Path

import detect_dust


# darktable dt_iop_retouch_algo_type_t: NONE=0, CLONE=1, HEAL=2, BLUR=3, FILL=4.
# Only the spot-removal algorithms (clone / heal) carry a healable defect; fill /
# blur are something else and are skipped. (detect_dust.RETOUCH_ALGO_HEAL == 2.)
_ALGO_DUST = {1, 2}

# darktable dt_masks_type_t bit flags (the ones we decode).
_MASK_CIRCLE = 1
_MASK_BRUSH = 64

_FORM_STRIDE = 44          # bytes per retouch form slot in the params blob
_RETOUCH_NO_FORMS = detect_dust.MAX_FORMS    # 300
_BRUSH_NODE_STRIDE = 44    # bytes per brush node (corner,ctrl1,ctrl2,border,dens,hard,state)

DEFAULT_TRANSFORM = {"flip": 0, "crop": (0.0, 0.0, 1.0, 1.0), "ashift": None}


# ---------------------------------------------------------------------------
# Low-level XMP / blob parsing
# ---------------------------------------------------------------------------

def _dt_decode_blob(value):
    """Decode a darktable XMP attribute blob to raw bytes.

    darktable stores binary fields either as a `gz` + 2-digit-ratio + base64
    (zlib-compressed) string for large values, or as plain lowercase hex for
    small ones. Returns b"" for an empty/missing value.
    """
    if not value:
        return b""
    if value.startswith("gz"):
        return zlib.decompress(base64.b64decode(value[4:]))
    return bytes.fromhex(value)


_RDF_LI_RE = re.compile(r"<rdf:li\b.*?/>", re.DOTALL)
_ATTR_RE = re.compile(r'darktable:(\w+)="([^"]*)"', re.DOTALL)


def _iter_rdf_li(xmp_text):
    """Yield each self-closing <rdf:li .../> block (history + mask entries)."""
    return _RDF_LI_RE.findall(xmp_text)


def _attrs(li_text):
    """Return {attr_name: value} of the darktable:* attributes on one rdf:li."""
    return {k: v for k, v in _ATTR_RE.findall(li_text)}


def _parse_retouch_params(blob):
    """Parse a retouch params blob → [(form_id, algorithm), ...] for active slots.

    The blob is 300 form slots × 44 bytes followed by a 60-byte global tail; a
    slot is active when its form id is non-zero. formid @0, algorithm @8.
    """
    out = []
    if len(blob) < _RETOUCH_NO_FORMS * _FORM_STRIDE:
        return out
    for i in range(_RETOUCH_NO_FORMS):
        base = i * _FORM_STRIDE
        form_id = struct.unpack_from("<i", blob, base + 0)[0]
        algo = struct.unpack_from("<i", blob, base + 8)[0]
        if form_id != 0:
            out.append((form_id, algo))
    return out


# ---------------------------------------------------------------------------
# Form geometry decode (raw-buffer normalized space)
# ---------------------------------------------------------------------------

def _decode_brush(points, src):
    """Decode a brush mask → raw-space geometry dict, or None if malformed.

    Layout per node (44 bytes): corner[2], ctrl1[2], ctrl2[2], border[2],
    density, hardness, state. The exact inverse of make_brush/stroke_mask_points.
    A 1-2 node form with negligible span is a "dot"; more nodes is a stroke path.
    """
    if not points or len(points) % _BRUSH_NODE_STRIDE != 0:
        return None
    n = len(points) // _BRUSH_NODE_STRIDE
    corners = []
    border = 0.0
    for i in range(n):
        base = i * _BRUSH_NODE_STRIDE
        cx, cy = struct.unpack_from("<ff", points, base + 0)
        if i == 0:
            border = struct.unpack_from("<ff", points, base + 24)[0]  # border_x
        corners.append((cx, cy))
    span = 0.0
    if n >= 2:
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        span = max(max(xs) - min(xs), max(ys) - min(ys))
    kind = "dot" if (n <= 2 and span < 0.01) else "stroke"
    return {"kind": kind, "points": [corners[0]] if kind == "dot" else corners,
            "radius_norm": float(border), "src": _decode_src(src)}


def _decode_circle(points, src):
    """Decode a circle mask (center[2], radius, border = 16 bytes) → dot geometry."""
    if not points or len(points) < 16:
        return None
    cx, cy, radius, cborder = struct.unpack_from("<ffff", points, 0)
    return {"kind": "dot", "points": [(cx, cy)],
            "radius_norm": float(radius + cborder), "src": _decode_src(src)}


def _decode_src(src):
    """Decode an 8-byte mask_src blob (<ff sx sy) → (sx, sy) or None."""
    if not src or len(src) < 8:
        return None
    return struct.unpack_from("<ff", src, 0)


def _decode_form(mask_type, points, src):
    """Dispatch on mask type. Returns raw-space geometry dict or None (skip)."""
    if mask_type & _MASK_BRUSH:
        return _decode_brush(points, src)
    if mask_type & _MASK_CIRCLE:
        return _decode_circle(points, src)
    return None  # ellipse / path / gradient — not imported (add by hand in the UI)


# ---------------------------------------------------------------------------
# raw geometry → export-space spot
# ---------------------------------------------------------------------------

def _form_to_spot(form, flip, crop, export_w, export_h, ashift):
    crop_l, crop_t, crop_r, crop_b = crop
    orig_w = export_w / max(crop_r - crop_l, 1e-6)
    orig_h = export_h / max(crop_b - crop_t, 1e-6)
    border_scale = min(orig_w, orig_h)

    def fwd(nx, ny):
        return detect_dust._original_to_export(
            nx, ny, flip, crop, export_w, export_h, ashift_params=ashift)

    pts = [fwd(x, y) for (x, y) in form["points"]]
    if any(p is None for p in pts):
        return None  # form lies outside this frame's crop → not in the export

    brush_radius_px = max(1.0, form["radius_norm"] * border_scale)
    spot = {"kind": form["kind"], "brush_radius_px": brush_radius_px}
    if form["kind"] == "stroke" and len(pts) >= 2:
        spot["path"] = [[float(p[0]), float(p[1])] for p in pts]
    else:
        spot["kind"] = "dot"
        spot["cx"], spot["cy"] = float(pts[0][0]), float(pts[0][1])

    if form.get("src"):
        s = fwd(form["src"][0], form["src"][1])
        if s is not None:
            spot["src_cx"], spot["src_cy"] = float(s[0]), float(s[1])
    return spot


# ---------------------------------------------------------------------------
# Public: decode all active dust forms from an XMP
# ---------------------------------------------------------------------------

def decode_xmp_masks(xmp_text, transform, export_w, export_h, sensor_label=None):
    """Decode the active film-dust retouch forms of an XMP into export-space spots.

    transform: {"flip", "crop", "ashift"} as produced by parse_transform_params.
    sensor_label: the localized `multi_name` of sensor-dust instances to EXCLUDE.
    Returns a list of spot dicts (kind "dot"/"stroke") in export pixel coords.
    """
    flip = transform.get("flip", 0)
    crop = transform.get("crop", (0.0, 0.0, 1.0, 1.0))
    ashift = transform.get("ashift")

    m = re.search(r'darktable:history_end="(\d+)"', xmp_text)
    history_end = int(m.group(1)) if m else 1 << 30

    # Latest history entry per multi_priority (an instance) below history_end.
    by_prio = {}
    for li in _iter_rdf_li(xmp_text):
        a = _attrs(li)
        if a.get("operation") != "retouch":
            continue
        try:
            num = int(a.get("num", "-1"))
        except ValueError:
            continue
        if num >= history_end:
            continue
        prio = a.get("multi_priority", "0")
        if prio not in by_prio or num > by_prio[prio][0]:
            by_prio[prio] = (num, a)

    # Form ids of the ENABLED, NON-sensor instances (the dust to import).
    dust_form_ids = set()
    skipped_types = {}
    for _num, a in by_prio.values():
        if a.get("enabled") != "1":
            continue
        if sensor_label and a.get("multi_name") == sensor_label:
            continue
        for fid, algo in _parse_retouch_params(_dt_decode_blob(a.get("params", ""))):
            if algo in _ALGO_DUST:
                dust_form_ids.add(fid)
    if not dust_form_ids:
        return []

    # Latest snapshot of each mask form (masks_history is cumulative → highest
    # mask_num holds the current geometry of every still-active form).
    masks = {}
    for li in _iter_rdf_li(xmp_text):
        a = _attrs(li)
        if "mask_id" not in a or "mask_points" not in a:
            continue
        try:
            mid = int(a["mask_id"])
            mnum = int(a.get("mask_num", "-1"))
        except ValueError:
            continue
        if mid not in masks or mnum > masks[mid][0]:
            masks[mid] = (mnum, int(a.get("mask_type", "0")),
                          _dt_decode_blob(a["mask_points"]),
                          _dt_decode_blob(a.get("mask_src", "")))

    spots = []
    for fid in dust_form_ids:
        entry = masks.get(fid)
        if entry is None:
            continue
        _mnum, mtype, points, src = entry
        form = _decode_form(mtype, points, src)
        if form is None:
            skipped_types[mtype] = skipped_types.get(mtype, 0) + 1
            continue
        spot = _form_to_spot(form, flip, crop, export_w, export_h, ashift)
        if spot is not None:
            spots.append(spot)

    if skipped_types:
        print(f"  import_retouch: skipped non-brush/circle forms (type:count) "
              f"{skipped_types} — add by hand in the UI if needed")
    return spots


# ---------------------------------------------------------------------------
# Spot → annotation conversion + seeding
# ---------------------------------------------------------------------------

def _spot_to_missed_dust(spot):
    md = {"cx": float(spot["cx"]), "cy": float(spot["cy"]),
          "brush_radius_px": float(spot["brush_radius_px"])}
    if "src_cx" in spot and "src_cy" in spot:
        md["src_cx"] = float(spot["src_cx"])
        md["src_cy"] = float(spot["src_cy"])
    return md


def _spot_to_missed_stroke(spot):
    # missed_stroke_to_spot recomputes brush_radius_px from stroke_width_px; invert
    # that scaling so the seeded stroke heals at ~the imported border width.
    scale = max(getattr(detect_dust.DEFAULT_TUNING, "STROKE_BORDER_SCALE", 1.0), 1e-6)
    width = float(spot["brush_radius_px"]) / scale
    ms = {"path": [[float(p[0]), float(p[1])] for p in spot["path"]],
          "stroke_width_px": width}
    if "src_cx" in spot and "src_cy" in spot:
        ms["src_cx"] = float(spot["src_cx"])
        ms["src_cy"] = float(spot["src_cy"])
    return ms


def spots_to_annotation(stem, spots):
    """Build the {stem}_annotations.json dict (serialize_annotations shape) that
    seeds the imported dust as missed_dust / missed_strokes over a fresh detect."""
    missed_dust, missed_strokes = [], []
    for sp in spots:
        if sp.get("kind") == "stroke" and len(sp.get("path") or []) >= 2:
            missed_strokes.append(_spot_to_missed_stroke(sp))
        else:
            missed_dust.append(_spot_to_missed_dust(sp))
    return {
        "stem": stem,
        "false_positives": [],
        "missed_dust": missed_dust,
        "missed_strokes": missed_strokes,
        "source_overrides": [],
        "radius_overrides": [],
        "path_overrides": [],
        "spot_notes": [],
        "source_mismatches": [],
        "radius_mismatches": [],
    }


def _read_source_xmp_manifest(path):
    """Parse source_xmp.txt → ({stem: xmp_path}, sensor_label, dust_label)."""
    mapping, sensor_label, dust_label = {}, None, None
    if not os.path.isfile(path):
        return mapping, sensor_label, dust_label
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "|" not in line:
                continue
            key, val = line.split("|", 1)
            if key == "SENSOR_LABEL":
                sensor_label = val
            elif key == "DUST_LABEL":
                dust_label = val
            else:
                mapping[key.strip()] = val.strip()
    return mapping, sensor_label, dust_label


def _export_dims(export_dir, stem):
    """Pixel size of the exported frame (the JPG the UI displays)."""
    from PIL import Image
    for ext in (".jpg", ".jpeg"):
        p = Path(export_dir) / f"{stem}{ext}"
        if p.is_file():
            with Image.open(p) as im:
                return im.size  # (w, h)
    return None


def seed_import_annotations(export_dir):
    """Decode each frame's existing dust retouch from its source XMP and write
    seed `{stem}_annotations.json` files + `import_baseline.json` (the decoded
    sets, for change detection on apply). Returns {stem: [spot,...]}.

    Reads `transform_params.txt` (flip/crop/ashift) and `source_xmp.txt`
    (stem→original sidecar path + the sensor label to exclude), both written by
    the Lua side before launching the UI.
    """
    export_dir = Path(export_dir)
    transforms = detect_dust.parse_transform_params(
        str(export_dir / "transform_params.txt"))
    mapping, sensor_label, _dust_label = _read_source_xmp_manifest(
        str(export_dir / "source_xmp.txt"))

    baseline = {}
    for stem, xmp_path in mapping.items():
        dims = _export_dims(export_dir, stem)
        if dims is None or not os.path.isfile(xmp_path):
            continue
        ew, eh = dims
        try:
            with open(xmp_path, "r", encoding="utf-8") as f:
                xmp_text = f.read()
        except OSError:
            continue
        t = transforms.get(stem, DEFAULT_TRANSFORM)
        spots = decode_xmp_masks(xmp_text, t, ew, eh, sensor_label=sensor_label)
        baseline[stem] = spots
        ann = spots_to_annotation(stem, spots)
        with open(export_dir / f"{stem}_annotations.json", "w") as f:
            json.dump(ann, f, indent=2)
        print(f"  import_retouch: seeded {len(spots)} GT form(s) for {stem}")

    with open(export_dir / "import_baseline.json", "w") as f:
        json.dump(baseline, f, indent=2)
    return baseline


# ---------------------------------------------------------------------------
# Change detection (apply-back only when the committed set differs from import)
# ---------------------------------------------------------------------------

def _spot_xy(spot):
    if spot.get("kind") == "stroke":
        path = spot.get("path") or []
        if not path:
            return None
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    return (spot.get("cx"), spot.get("cy"))


def spots_differ(baseline, final, min_dim):
    """True if the committed spot set differs from the imported baseline.

    A simple greedy proximity + radius match (tolerances scaled to the frame):
    equal counts AND every baseline spot pairs with a final spot of the same kind
    at ~the same position and radius. Used to skip the apply-back when the user
    did not change anything.
    """
    baseline = baseline or []
    final = final or []
    if len(baseline) != len(final):
        return True
    pos_tol = max(2.0, 0.004 * max(min_dim, 1))   # ~0.4% of the short side
    rad_tol = max(2.0, 0.05 * 1.0)                 # absolute px floor; radii are px
    used = [False] * len(final)
    for b in baseline:
        bxy = _spot_xy(b)
        if bxy is None or bxy[0] is None:
            return True
        match = -1
        for j, fsp in enumerate(final):
            if used[j] or fsp.get("kind") != b.get("kind"):
                continue
            fxy = _spot_xy(fsp)
            if fxy is None or fxy[0] is None:
                continue
            if abs(fxy[0] - bxy[0]) <= pos_tol and abs(fxy[1] - bxy[1]) <= pos_tol:
                br = float(b.get("brush_radius_px") or 0.0)
                fr = float(fsp.get("brush_radius_px") or 0.0)
                if abs(br - fr) <= max(rad_tol, 0.1 * max(br, 1.0)):
                    match = j
                    break
        if match < 0:
            return True
        used[match] = True
    return False


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--seed":
        seed_import_annotations(sys.argv[2])
    else:
        print("Usage: import_retouch.py --seed <export_dir>")
