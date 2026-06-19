"""
Debug UI for auto-negadoctor patch annotation.

Shows each frame ALREADY INVERTED (the Python-rendered negadoctor preview)
with the patches the algorithm chose: the local film-base candidate, the
roll-wide global film base (highlighted on the winning frame), and the
shadows / highlights neutral patches behind wb_low / wb_high. The user marks
wrong patches by placing the correct position; corrections + notes go into
{stem}_annotations.json and debug_report.txt for algorithm tuning.

Built on the shared viewer base in common/debug_ui_base.py.

Usage:
    python debug_ui.py <session_dir>

Reads:  {session_dir}/{stem}_debug_nega.json   (per-frame session data;
        the inverted preview + analysis-crop mask are rendered LIVE from the
        source negative named in it — no baked image files)
Writes: {session_dir}/{stem}_annotations.json  (auto-saved per image)
        {session_dir}/debug_report.txt          (on window close)

Interaction (same idiom as the crop UI: select, then Ctrl+Click / scroll):
    Click patch rect / 1/2/3 keys — select patch (1=film base 2=shadows 3=highlights)
    Ctrl+Click — place the selected patch's window at the clicked position
    Scroll (patch selected) — nudge corrected patch 1 px vertically (Shift = 10 px)
    C — clear the correction for the selected patch
    V — toggle inverted preview / original negative view
    G — toggle "bad inversion" flag for the whole frame
"""

import sys
import math
from pathlib import Path

import tkinter as tk

import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).parent.parent))   # repo root -> common
sys.path.insert(0, str(Path(__file__).parent))           # feature dir

from common.debug_ui_base import DebugUIBase, image_to_canvas, canvas_to_image
import nega_model as nm


PATCHES = ("film_base", "shadows", "highlights")
PATCH_KEYS = {"1": "film_base", "2": "shadows", "3": "highlights"}
PATCH_COLORS = {"film_base": "#ff9900", "shadows": "#00ddff",
                "highlights": "#ffffff"}
PATCH_SHORT = {"film_base": "base", "shadows": "shad", "highlights": "high"}

# Print-page params (negadoctor "print properties" tab), adjustable in the
# UI with live preview: select with 4/5/6/7, scroll to adjust (Shift = big
# step), C clears the override.
PRINT_PARAMS = ("black", "gamma", "soft_clip", "exposure")
PRINT_KEYS = {"4": "black", "5": "gamma", "6": "soft_clip", "7": "exposure"}
PRINT_LABEL = {"black": "paper black (density correction)",
               "gamma": "paper grade (gamma)",
               "soft_clip": "paper gloss (specular highlights)",
               "exposure": "print exposure adjustment"}
PRINT_SHORT = {"black": "blk", "gamma": "gamma", "soft_clip": "gloss",
               "exposure": "pexp"}
PRINT_STEP = {"black": (0.005, 0.02), "gamma": (0.05, 0.25),
              "soft_clip": (0.01, 0.05), "exposure": (0.01, 0.05)}

# "Brighter"/"darker" one-key combo (] / [): the user's repeated manual trick is
# to raise paper black (which lifts midtones but clips highlights) and then lower
# print exposure until the highlights stop clipping — net brighter midtones with
# the highlights still pinned just below clip. This does both in unison: nudge
# black by BRIGHTEN_BLACK_STEP (0.05 on the [-0.5, 0.5] black param, a "5%" move),
# then re-solve exposure to hold the highlights (the P99.9 render percentile) at
# the level they were at before the move. Pinning that level (rather than "lower
# exposure until clip is gone") keeps the op a deterministic function of black, so
# brighter-then-darker returns exactly to the original exposure (the highlight
# level is the operation's invariant), and dark scenes aren't over-brightened.
BRIGHTEN_BLACK_STEP = 0.05
# A display-render channel at/above this linear value is a blown highlight.
CLIP_OUT_THR = 0.999

# Shadows/highlights as color wheels (spec 02): two wheels drive wb_low /
# wb_high directly with live preview, an alternative to picking patches. The
# chosen value is stored as a direct abstract wb override (the exact value
# that lands in the XMP) — the clean ground truth for tuning the production
# wb finder. Selection names live in the same selected_patch union as the
# print params / crop.
WB_NAMES = ("wb_shadows", "wb_highlights")
WB_NAME_KIND = {"wb_shadows": "low", "wb_highlights": "high"}
# override-dict key and the params field each wheel reads/writes
WB_NAME_OVR = {"wb_shadows": "shadows", "wb_highlights": "highlights"}
WB_NAME_PARAM = {"wb_shadows": "wb_low", "wb_highlights": "wb_high"}
WB_NAME_LABEL = {"wb_shadows": "shadows wheel (wb_low)",
                 "wb_highlights": "highlights wheel (wb_high)"}
WB_NAME_SHORT = {"wb_shadows": "wb_lo", "wb_highlights": "wb_hi"}


# User crop correction: the true photo-content rectangle, drawn by the user
# with a rubber-band drag while "crop" (key 8) is selected. Everything
# outside is holder/rejected; the live re-render recomputes the picker
# percentiles, D_max, black/exposure and the print tune inside it.
CROP_NAME = "crop"
CROP_KEY = "8"

# Analysis-mask view (M key cycles): category codes from
# auto_negadoctor.build_analysis_mask and their RGBA tints. The analysis
# area is strictly the content-crop rectangle: code 2 = outside the crop.
MASK_TINTS = {2: (255, 60, 60, 110)}    # rejected (outside crop) - red
MASK_VIEW_LABELS = {0: "Analysis Crop  (M)",
                    1: "Hide Rejected  (M)",
                    2: "Normal View  (M)"}


# --- resolution-independent annotation coords --------------------------------
# Annotations are persisted as NORMALIZED fractions of the frame size so they
# stay valid regardless of the export resolution. The UI works in pixels
# internally (canvas / hit-testing); we convert only at the JSON boundary.
# A coord whose magnitude exceeds this is treated as a legacy pixel value.
_NORM_MAX = 1.5


def _norm_xywh(rect, w, h):
    """[x, y, rw, rh] pixels -> fractions of (w, h)."""
    x, y, rw, rh = rect
    return [x / w, y / h, rw / w, rh / h]


def _denorm_xywh(rect, w, h):
    """[x, y, rw, rh] fractions -> integer pixels. Legacy pixel rects (any
    coord > _NORM_MAX) are passed through rounded, so old session files load."""
    if max(abs(float(v)) for v in rect) > _NORM_MAX:
        return [int(round(float(v))) for v in rect]
    x, y, rw, rh = rect
    return [int(round(x * w)), int(round(y * h)),
            int(round(rw * w)), int(round(rh * h))]


def _norm_ltrb(border, w, h):
    """[left, top, right, bottom] margins in pixels -> fractions."""
    l, t, r, b = border
    return [l / w, t / h, r / w, b / h]


# --- color-wheel widget -------------------------------------------------------
# Cache the rendered hue/chroma disk per size (identical for both wheels — it
# is only a visual guide; the live image preview is the real feedback).
_WHEEL_DISK_CACHE = {}


def _render_wheel_disk(size):
    """RGBA PIL image of the hue/chroma disk for the given pixel size.

    The displayed color at a point is the cast that point's wb pushes toward
    (computed via the same projection as nm.wheel_to_wb, normalized so the
    max channel is full) — center = white (neutral), rim = saturated."""
    if size in _WHEEL_DISK_CACHE:
        return _WHEEL_DISK_CACHE[size]
    c = (size - 1) / 2.0
    maxr = size / 2.0 - 1.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    dx = xx - c
    dy = c - yy                                   # canvas y grows downward
    rr = np.hypot(dx, dy) / maxr
    ang = np.arctan2(dy, dx)
    mag = np.clip(rr, 0.0, 1.0) * nm.WHEEL_MAX_CHROMA
    px = np.cos(ang) * mag
    py = np.sin(ang) * mag
    # d_c = (2/3) * axis_c . (px, py)   (inverse zero-sum projection)
    axes = nm._WHEEL_AXES                          # (3, 2)
    d = (2.0 / 3.0) * (axes[:, 0][:, None, None] * px
                       + axes[:, 1][:, None, None] * py)   # (3, H, W)
    wb = np.exp(d)
    col = wb / wb.max(axis=0, keepdims=True)       # max channel -> 1.0
    rgb = (np.clip(col, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    rgb = np.transpose(rgb, (1, 2, 0))             # (H, W, 3)
    alpha = np.where(rr <= 1.0, 255, 0).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])
    img = Image.fromarray(rgba, mode="RGBA")
    _WHEEL_DISK_CACHE[size] = img
    return img


class ColorWheel:
    """A resizable hue/chroma disk that edits one normalized wb vector.

    Dragging inside the disk maps the cursor (angle, radius) to a wb via
    nm.wheel_to_wb(kind) and fires on_change(wb). set_wb() moves the marker to
    an externally-set value (the auto-found wb on load, or a saved override);
    set_auto() places a fixed pin at the algorithm's wb. resize() re-renders
    the disk at a new pixel size (a bigger wheel = finer manual precision) and
    re-places the marker/pin from the remembered wb values."""

    MIN_SIZE = 90
    # Ctrl-drag multiplies cursor movement by this, so the same hand motion
    # makes a much smaller wb change (fine manual precision near neutral).
    FINE_FACTOR = 0.18
    _CTRL_MASK = 0x0004              # tk event.state bit for the Ctrl key

    def __init__(self, parent, kind, on_change, size=150, bg="#484848"):
        self.kind = kind                  # "low" or "high"
        self.on_change = on_change
        self._marker_wb = [1.0, 1.0, 1.0]
        self._auto_wb = None
        self._anchor_cursor = None        # cursor pos when the drag was anchored
        self._anchor_marker = None        # marker pos when the drag was anchored
        self._anchor_fine = False         # fine state the anchor was taken under
        self.canvas = tk.Canvas(parent, bg=bg, highlightthickness=0,
                                cursor="crosshair")
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self._apply_size(size)

    def pack(self, **kw):
        self.canvas.pack(**kw)

    def resize(self, size):
        """Re-render the disk at a new square pixel size."""
        size = max(self.MIN_SIZE, int(size))
        if size == self.size:
            return
        self._apply_size(size)

    def _apply_size(self, size):
        self.size = size
        self._cx = self._cy = (size - 1) / 2.0
        self._maxr = size / 2.0 - 1.0
        self.canvas.config(width=size, height=size)
        self.canvas.delete("disk")
        self._photo = ImageTk.PhotoImage(_render_wheel_disk(size))
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo,
                                 tags="disk")
        self.canvas.tag_lower("disk")
        self._marker_pos = self._wb_pos(self._marker_wb)
        self._auto_pos = self._wb_pos(self._auto_wb) if self._auto_wb else None
        self._draw_marker()
        self._draw_auto()

    # --- view ---
    def _wb_pos(self, wb):
        angle, radius = nm.wb_to_wheel(wb)
        r = radius * self._maxr
        return (self._cx + math.cos(angle) * r, self._cy - math.sin(angle) * r)

    def _draw_marker(self):
        self.canvas.delete("marker")
        x, y = self._marker_pos
        # ring + center dot, dark/light pair so it shows on any hue
        self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, outline="#000000",
                                width=3, tags="marker")
        self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, outline="#ffffff",
                                width=1, tags="marker")
        self.canvas.tag_raise("marker")    # keep the draggable marker on top

    def _draw_auto(self):
        self.canvas.delete("autopin")
        if self._auto_pos is None:
            return
        x, y = self._auto_pos
        # small fixed gold pin = the algorithm's auto-detected wb
        self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#ffd700",
                                outline="#000000", width=1, tags="autopin")
        self.canvas.tag_raise("marker")

    def set_wb(self, wb):
        """Move the draggable marker to reflect a wb vector (no callback)."""
        self._marker_wb = list(wb)
        self._marker_pos = self._wb_pos(wb)
        self._draw_marker()

    def set_auto(self, wb):
        """Place (or clear) the fixed pin at the algorithm's auto wb."""
        self._auto_wb = list(wb) if wb else None
        self._auto_pos = self._wb_pos(wb) if wb else None
        self._draw_auto()

    # --- input ---
    def _emit(self, ex, ey):
        dx = ex - self._cx
        dy = self._cy - ey
        radius = min(1.0, math.hypot(dx, dy) / self._maxr)
        angle = math.atan2(dy, dx)
        # clamp the marker to the rim so it never leaves the disk
        r = radius * self._maxr
        self._marker_pos = (self._cx + math.cos(angle) * r,
                            self._cy - math.sin(angle) * r)
        self._draw_marker()
        wb = nm.wheel_to_wb(angle, radius, self.kind)
        self._marker_wb = wb              # remember so resize keeps the spot
        self.on_change(wb)

    def _is_fine(self, event):
        return bool(event.state & self._CTRL_MASK)

    def _anchor(self, event):
        # Pin the cursor→marker mapping to the current positions: from here the
        # marker is anchor_marker + (cursor - anchor_cursor) * gain, an absolute
        # function of where the cursor is (not an accumulation of per-event
        # deltas), so fine mode tracks the cursor exactly like a plain drag —
        # just slower — with no incremental drift or rim dead-zone.
        self._anchor_cursor = (event.x, event.y)
        self._anchor_marker = self._marker_pos
        self._anchor_fine = self._is_fine(event)

    def _on_press(self, event):
        # A plain click jumps the marker to the cursor; a Ctrl-click instead
        # starts nudging from the marker's current spot (no jump), so fine
        # tweaks don't lose the existing value the moment the button goes down.
        if not self._is_fine(event):
            self._emit(event.x, event.y)
        self._anchor(event)

    def _on_drag(self, event):
        # Absolute-anchored: marker = anchor + (cursor - anchor) * gain. The
        # fine factor only scales the mapping (slower), it does NOT switch to a
        # relative/incremental model, so the slow mode behaves identically to
        # the plain mode in absolute terms. Re-anchor when Ctrl toggles
        # mid-drag so the gain change is seamless from the current spot.
        if self._anchor_cursor is None or self._is_fine(event) != self._anchor_fine:
            self._anchor(event)
        gain = self.FINE_FACTOR if self._is_fine(event) else 1.0
        ax, ay = self._anchor_cursor
        mx, my = self._anchor_marker
        self._emit(mx + (event.x - ax) * gain, my + (event.y - ay) * gain)


class NegadoctorDebugUI(DebugUIBase):
    WINDOW_TITLE = "Auto Negadoctor Debug UI"
    HAS_ITEM_PANEL_FOOTER = True   # color wheels live below the item table

    # Annotate-and-apply flow (Lua AutoNegadoctor_Annotate_Apply): when set, the
    # UI writes applied_results.txt on close (the user's corrections applied over
    # the auto params, auto where none) so the Lua side can write the XMPs.
    apply_mode = False
    APPLIED_RESULTS_FILENAME = "applied_results.txt"
    EMPTY_SESSION_MESSAGE = "No *_debug_nega.json files found in:"
    ITEM_PANEL_TITLE = "Patches:"
    CENTER_BUTTON_TEXT = "Center on patch"
    ITEM_COLS    = ("patch", "det", "fb", "corr", "nt")
    ITEM_HEADERS = {"patch": "patch", "det": "detected", "fb": "fb",
                    "corr": "corrected", "nt": "✎"}
    ITEM_WIDTHS  = {"patch": 44, "det": 64, "fb": 24, "corr": 64, "nt": 22}
    ITEM_ANCHORS = {"patch": "w", "fb": "center", "nt": "center"}
    NOTE_COLUMN  = "nt"

    # ------------------------------------------------------------------
    # Session / annotation lifecycle
    # ------------------------------------------------------------------

    def load_session(self, session_dir):
        import auto_negadoctor
        return auto_negadoctor.load_debug_nega_dir(session_dir)

    def new_annotation_state(self, img_dict):
        return {
            "patch_corrections": {},   # {patch: [x, y, w, h] corrected rect}
            "print_overrides": {},     # {print param: corrected value}
            "wb_overrides": {},        # {"shadows"/"highlights": [r, g, b] wb}
            "crop_correction": None,   # [x, y, w, h] true photo-content rect
            "patch_notes": {},         # {patch / print param / wb / "crop": str}
            "bad_inversion": False,
            "bad_inversion_note": "",
        }

    def serialize_annotations(self, stem):
        ann = self.annotations[stem]
        img_dict = next((d for d in self.images if d["stem"] == stem), None)
        params = (img_dict.get("params") or {}) if img_dict else {}
        # frame size for normalizing pixel rects -> resolution-independent fractions
        W = img_dict["width"] if img_dict else None
        H = img_dict["height"] if img_dict else None
        out_corr = {}
        for patch, rect in sorted(ann["patch_corrections"].items()):
            det = self._detected_rect(img_dict, patch) if img_dict else None
            out_corr[patch] = {
                "detected": _norm_xywh(det, W, H) if (det and W) else det,
                "corrected": _norm_xywh(rect, W, H) if W else [int(v) for v in rect],
            }
        out_over = {}
        for name, val in sorted(ann["print_overrides"].items()):
            out_over[name] = {
                "applied": params.get(name),
                "corrected": float(val),
            }
        # wb wheel overrides (direct abstract wb — the ground truth for tuning
        # the production wb finder); store applied alongside for comparison
        out_wb = {}
        for kind, wb in sorted(ann["wb_overrides"].items()):
            applied = params.get("wb_low" if kind == "shadows" else "wb_high")
            out_wb[kind] = {
                "applied": list(applied) if applied else None,
                "corrected": [float(v) for v in wb],
            }
        out = {
            "stem": stem,
            "patch_corrections": out_corr,
            "print_overrides": out_over,
            "wb_overrides": out_wb,
            "patch_notes": {p: n for p, n in sorted(ann["patch_notes"].items())},
            "bad_inversion": bool(ann["bad_inversion"]),
            "bad_inversion_note": ann["bad_inversion_note"],
        }
        if ann.get("crop_correction"):
            border = list(img_dict.get("border") or []) if img_dict else None
            out["crop_correction"] = {
                "auto_border": _norm_ltrb(border, W, H) if (border and W) else border,
                "corrected": _norm_xywh(ann["crop_correction"], W, H) if W
                             else [int(v) for v in ann["crop_correction"]],
            }
        return out

    def deserialize_annotations(self, img_dict, data):
        stem = img_dict["stem"]
        ann = self.annotations[stem]
        W, H = img_dict["width"], img_dict["height"]
        for patch, entry in data.get("patch_corrections", {}).items():
            if patch in PATCHES:
                try:
                    ann["patch_corrections"][patch] = _denorm_xywh(
                        entry["corrected"], W, H)
                except (KeyError, ValueError, TypeError):
                    pass
        for name, entry in data.get("print_overrides", {}).items():
            if name in PRINT_PARAMS:
                try:
                    ann["print_overrides"][name] = float(entry["corrected"])
                except (KeyError, ValueError, TypeError):
                    pass
        for kind, entry in data.get("wb_overrides", {}).items():
            if kind in ("shadows", "highlights"):
                try:
                    wb = [float(v) for v in entry["corrected"]]
                    if len(wb) == 3 and all(math.isfinite(v) for v in wb):
                        ann["wb_overrides"][kind] = wb
                except (KeyError, ValueError, TypeError):
                    pass
        crop = data.get("crop_correction")
        if crop:
            try:
                ann["crop_correction"] = _denorm_xywh(crop["corrected"], W, H)
            except (KeyError, ValueError, TypeError):
                pass
        ann["patch_notes"] = {p: str(n) for p, n in data.get("patch_notes", {}).items()
                              if (p in PATCHES or p in PRINT_PARAMS
                                  or p in WB_NAMES or p == CROP_NAME) and n}
        ann["bad_inversion"] = bool(data.get("bad_inversion", False))
        ann["bad_inversion_note"] = str(data.get("bad_inversion_note", ""))

    # ------------------------------------------------------------------
    # Patch geometry helpers
    # ------------------------------------------------------------------

    def _detected_rect(self, img_dict, patch):
        """Detected [x,y,w,h] of a patch on this frame, or None."""
        if patch == "film_base":
            local = (img_dict.get("film_base") or {}).get("local")
            return list(local["rect"]) if local else None
        p = (img_dict.get("patches") or {}).get(patch)
        if p and "rect" in p:
            return list(p["rect"])
        return None

    def _effective_rect(self, img_dict, patch):
        """Corrected rect if annotated, else detected (may be None)."""
        ann = self.annotations[img_dict["stem"]]
        corr = ann["patch_corrections"].get(patch)
        return list(corr) if corr else self._detected_rect(img_dict, patch)

    def _default_win(self, img_dict, patch):
        det = self._detected_rect(img_dict, patch)
        if det:
            return det[2]
        frac = self.constants.get(
            "BASE_WIN_FRAC" if patch == "film_base" else "PATCH_WIN_FRAC", 0.04)
        return max(int(img_dict["width"] * frac), 16)

    # ------------------------------------------------------------------
    # Negative-space color access (live wb feedback for corrections)
    # ------------------------------------------------------------------

    def _neg_lin(self, img_dict):
        """Linearized negative of the current frame (cached). Uses the
        pipeline loader (full TIFF precision) WITH the roll's vignette
        correction applied (unless the user toggled it OFF with N), so patch RGB
        sampling and live re-renders stay consistent with the pipeline analysis.
        The cache key includes the vignette toggle + the review source so a
        toggle reloads."""
        vig = img_dict.get("vignette") if getattr(self, "vignette_on", True) else None
        key = (img_dict["stem"], bool(getattr(self, "vignette_on", True)),
               getattr(self, "review_source", None))
        if getattr(self, "_neg_cache_key", None) == key:
            return self._neg_cache
        path = img_dict.get("negative_path")
        lin = None
        if path and Path(path).exists():
            import auto_negadoctor
            try:
                lin = auto_negadoctor.load_frame(path, vig)[1]
            except Exception:
                lin = None
        self._neg_cache_key = key
        self._neg_cache = lin
        return lin

    def _invert_pil(self, lin, params):
        """sRGB PIL inversion of a linear negative for the given params
        (fused render + sRGB + quantize in one parallel pass)."""
        return Image.fromarray(nm.render_negadoctor_srgb8(lin, params))

    def _variant_params(self, img_dict):
        """The active base-variant params (AI when selected, else analytical)."""
        if self.variant == "ai" and img_dict.get("params_ai"):
            return img_dict["params_ai"]
        return img_dict.get("params")

    def _load_display_pil(self, img_dict):
        """Live analytical/AI inversion from the source negative — there is no
        baked {stem}_inverted.jpg to fall back to."""
        lin = self._neg_lin(img_dict)
        params = self._variant_params(img_dict)
        if lin is None or not params:
            raise FileNotFoundError(
                f"source negative unavailable for {img_dict.get('stem', '?')}")
        return self._invert_pil(lin, params)

    # render the negative downscaled to ~this many px on the long edge for
    # thumbnails: render_negadoctor is ~0.5s at full 2000px but trivial once
    # downscaled (the render, NOT the file read, is the thumbnail bottleneck),
    # and the result is shrunk to ~210px by the caller anyway.
    _THUMB_RENDER_MAX = 320

    def _load_thumb_pil(self, img_dict):
        """Background-thread thumbnail render. Loads the negative INDEPENDENTLY
        (does not touch the single-slot _neg_lin cache the main thread owns)
        and renders at thumbnail scale so it stays cheap."""
        import auto_negadoctor
        import cv2
        path = img_dict.get("negative_path")
        params = self._variant_params(img_dict)
        if not path or not Path(path).exists() or not params:
            raise FileNotFoundError("no source negative")
        lin = auto_negadoctor.load_frame(path, img_dict.get("vignette"))[1]
        h, w = lin.shape[:2]
        longest = max(h, w)
        if longest > self._THUMB_RENDER_MAX:
            s = self._THUMB_RENDER_MAX / longest
            lin = cv2.resize(lin, (max(1, round(w * s)), max(1, round(h * s))),
                             interpolation=cv2.INTER_AREA)
        return self._invert_pil(lin, params)

    def _neg_rgb_at(self, img_dict, rect):
        lin = self._neg_lin(img_dict)
        if lin is None or rect is None:
            return None
        x, y, w, h = [int(v) for v in rect]
        ih, iw = lin.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(iw, x + w), min(ih, y + h)
        if x1 <= x0 or y1 <= y0:
            return None
        return [float(v) for v in lin[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)]

    def _wb_for_patch(self, img_dict, patch, rgb):
        """What wb the given negative-space patch color would produce."""
        p = img_dict.get("params") or {}
        if not rgb or not p:
            return None
        try:
            if patch == "shadows":
                return nm.compute_wb_low(p["Dmin"], rgb, p["D_max"])
            if patch == "highlights":
                return nm.compute_wb_high(p["Dmin"], rgb, p["D_max"],
                                          p["offset"], p["wb_low"])
        except (ValueError, ZeroDivisionError, OverflowError):
            return None
        return None

    # ------------------------------------------------------------------
    # Live re-render from corrections
    # ------------------------------------------------------------------

    def _corrected_params(self, img_dict):
        """Negadoctor params recomputed from the user's corrections, or None
        when there are none. A corrected film base re-derives Dmin/D_max and
        re-evaluates wb against them; corrected shadows/highlights re-derive
        wb_low/wb_high. black/exposure keep their applied (auto-tuned) values.
        """
        ann = self.annotations[img_dict["stem"]]
        corr = ann["patch_corrections"]
        overrides = ann["print_overrides"]
        wb_overrides = ann["wb_overrides"]
        crop = ann.get("crop_correction")
        p = img_dict.get("params")
        if (not corr and not overrides and not wb_overrides and not crop) or not p:
            return None
        out = {k: (list(v) if isinstance(v, list) else v) for k, v in p.items()}
        dmin, d_max = list(p["Dmin"]), p["D_max"]
        if "film_base" in corr:
            rgb = self._neg_rgb_at(img_dict, corr["film_base"])
            if rgb:
                dmin = [nm.clamp(v, nm.DMIN_RANGE) for v in rgb]
                out["Dmin"] = dmin

        # User crop: recompute the picker percentiles inside the true
        # content rect (the whole point: a missed holder remnant outside it
        # poisons picked_min/D_max/exposure)
        picked_min = img_dict.get("picked_min")
        picked_max = img_dict.get("picked_max")
        crop_border = None
        crop_lin = None
        if crop:
            crop_lin = self._neg_lin(img_dict)
            if crop_lin is not None:
                import auto_negadoctor as anp
                x, y, cw, ch = [int(v) for v in crop]
                W, H = img_dict["width"], img_dict["height"]
                crop_border = (max(x, 0), max(y, 0),
                               max(W - x - cw, 0), max(H - y - ch, 0))
                lin32 = np.asarray(crop_lin, dtype=np.float32)
                try:
                    picked_min, picked_max = anp.frame_percentiles(
                        lin32, np.zeros_like(lin32), crop_border, dmin)
                except Exception:
                    pass
        # D_max stays FIXED at darktable's default (anp.DMAX_DEFAULT, carried in
        # params); it is NOT re-derived from crop/film-base picks (the old
        # compute_dmax recompute produced the fabricated ~0.7). picked_min/max
        # below still drive the black/exposure pickers.

        def patch_rgb(patch):
            if patch in corr:
                return self._neg_rgb_at(img_dict, corr[patch])
            if "film_base" in corr:   # Dmin changed: re-derive detected wb too
                det = (img_dict.get("patches") or {}).get(patch) or {}
                return det.get("rgb_neg_linear")
            return None

        sh_rgb = patch_rgb("shadows")
        if sh_rgb:
            try:
                out["wb_low"] = nm.compute_wb_low(dmin, sh_rgb, d_max)
            except (ValueError, ZeroDivisionError, OverflowError):
                pass
        hi_rgb = patch_rgb("highlights")
        if hi_rgb:
            try:
                out["wb_high"] = nm.compute_wb_high(dmin, hi_rgb, d_max,
                                                    out["offset"], out["wb_low"])
            except (ValueError, ZeroDivisionError, OverflowError):
                pass

        # Wheel overrides set the abstract wb directly and WIN over any
        # patch-derived value. WB-only: black/exposure keep their tuned values
        # (unless a crop is also present, where the block below re-derives them
        # from whatever wb is in `out`, including these overrides).
        if "shadows" in wb_overrides:
            out["wb_low"] = [float(v) for v in wb_overrides["shadows"]]
        if "highlights" in wb_overrides:
            out["wb_high"] = [float(v) for v in wb_overrides["highlights"]]

        if crop_border is not None and picked_min and picked_max:
            # full production chain inside the user's crop: black/exposure
            # formulas, then the print auto-tune scoped to the crop
            import auto_negadoctor as anp
            try:
                out["black"] = nm.compute_black(dmin, picked_max, d_max,
                                                out["wb_high"], out["wb_low"],
                                                out["offset"])
                out["exposure"] = nm.compute_exposure(dmin, picked_min, d_max,
                                                      out["wb_high"],
                                                      out["wb_low"],
                                                      out["offset"], out["black"])
                out, _info = anp.tune_print_params(
                    np.asarray(crop_lin, dtype=np.float32), out, crop_border,
                    dmin)
            except Exception:
                pass

        for name, val in overrides.items():
            out[name] = nm.clamp(float(val), self._print_range(name))
        return out

    @staticmethod
    def _print_range(name):
        return {"black": nm.BLACK_RANGE, "gamma": nm.GAMMA_RANGE,
                "soft_clip": nm.SOFT_CLIP_RANGE,
                "exposure": nm.EXPOSURE_RANGE}[name]

    # ------------------------------------------------------------------
    # Annotate-and-apply: emit the final per-frame decisions on close
    # ------------------------------------------------------------------

    def _crop_positions(self, img_dict):
        """Normalized [left, top, right, bottom] crop positions in [0,1] for the
        darktable crop module: the user's crop annotation when present, else the
        auto-detected content border. None when neither yields a valid box.
        The TIFF is exported post-orientation, so these displayed-frame fractions
        map straight onto the crop module (same basis as auto_crop)."""
        W, H = img_dict["width"], img_dict["height"]
        if not W or not H:
            return None
        crop = self.annotations[img_dict["stem"]].get("crop_correction")
        if crop:
            x, y, w, h = (float(v) for v in crop)
            left, top, right, bottom = x / W, y / H, (x + w) / W, (y + h) / H
        else:
            border = img_dict.get("border")
            if not border:
                return None
            l, t, r, b = border
            left, top, right, bottom = l / W, t / H, (W - r) / W, (H - b) / H
        left, top = max(0.0, min(left, 1.0)), max(0.0, min(top, 1.0))
        right, bottom = max(0.0, min(right, 1.0)), max(0.0, min(bottom, 1.0))
        if right <= left or bottom <= top:
            return None
        return [left, top, right, bottom]

    def _write_applied_results(self):
        """Write applied_results.txt: one `OK|stem|params=<hex>|crop=L,T,R,B|
        flag=ok` line per frame, params being the user's corrections applied over
        the auto analysis (auto where the user added none). Crop is `none` when
        there is no usable box. Read back by the Lua apply step."""
        lines = []
        for img in self.images:
            stem = img["stem"]
            params = img.get("params")
            if not params:
                continue
            final = self._corrected_params(img) or params
            try:
                params_hex = nm.encode_negadoctor_params(final)
            except Exception as e:   # pragma: no cover - defensive
                print(f"  {stem}: param encode failed ({e}); skipped")
                continue
            crop = self._crop_positions(img)
            crop_str = ("none" if crop is None else
                        ",".join(f"{v:.6f}" for v in crop))
            flag = "bad" if self.annotations[stem].get("bad_inversion") else "ok"
            lines.append(f"OK|{stem}|params={params_hex}|crop={crop_str}|flag={flag}")
        path = Path(self.session_dir) / self.APPLIED_RESULTS_FILENAME
        path.write_text("\n".join(lines) + ("\n" if lines else ""),
                        encoding="utf-8")
        print(f"Applied results written to: {path} ({len(lines)} frame(s))")

    def _on_close(self):
        # In apply mode emit the decisions FIRST (uses the in-memory annotations
        # for every frame), then let the base save/report/destroy run.
        if self.apply_mode:
            try:
                self._write_applied_results()
            except Exception as e:   # pragma: no cover - defensive
                print(f"Failed to write applied results: {e}")
        super()._on_close()

    def _schedule_live_render(self, delay_ms=250):
        """Debounced re-render: scroll-resizing fires many events."""
        if self._live_job is not None:
            try:
                self.root.after_cancel(self._live_job)
            except Exception:
                pass
        self._live_job = self.root.after(delay_ms, self._apply_live_render)

    def _apply_live_render(self):
        self._live_job = None
        if self.view_negative:
            return
        img_dict = self.images[self.current_idx]
        # compare mode: show the algorithm's default render even when the
        # user has corrections (X toggles back and forth)
        params = None if self.compare_default else self._corrected_params(img_dict)
        corrected = params is not None
        if params is None:
            # No corrections: render the active variant's params live so the
            # preview is ALWAYS a fresh algorithmic inversion (AI variant uses
            # params_ai, otherwise the analytical params).
            params = self._variant_params(img_dict)
        lin = self._neg_lin(img_dict) if params else None
        if lin is None or not params:
            # nothing to render honestly (no source negative / no params) —
            # leave the current display untouched, no baked-file fallback.
            return
        # the badge ("RE-RENDERED FROM CORRECTIONS") is only meaningful when the
        # render came from user corrections, not the plain analytical/AI preview.
        self._live_rendered = corrected
        self._set_display_image(self._invert_pil(lin, params))

    # ------------------------------------------------------------------
    # Selection state
    # ------------------------------------------------------------------

    def init_selection_state(self):
        # selected_patch holds a patch name OR a print-param name
        self.selected_patch = None
        self.view_negative = False
        self.compare_default = False
        self.variant = "analytical"  # "analytical" | "ai"; PERSISTS across frames
        self.mask_view = 0           # 0=normal, 1=tint areas, 2=hide rejected
        self.show_histogram = True
        self._hist_data = None
        self.show_clipping = False     # on-image red/blue clip overlay (L)
        self._clip_stats = None        # {"hi": pct, "lo": pct, "total": n}
        self._display_base_pil = None
        self._amask_cache_stem = None
        self._amask_cache = None
        self._neg_cache_key = None
        self._neg_cache = None
        self.vignette_on = True        # N: apply the roll vignette correction
        # Calibration review: frames carry review={live,fitted} payloads + a
        # review_kind. The R toggle swaps the live (current source-code) result
        # vs the fitted (this session's) result into img_dict, reusing every
        # render path. PERSISTS across frames.
        _imgs = getattr(self, "images", None) or []
        self.review_mode = any(im.get("review") for im in _imgs)
        self.review_kind = next((im.get("review_kind") for im in _imgs
                                 if im.get("review")), None)
        self.review_source = "fitted"  # "fitted" | "live"
        if self.review_mode:
            self._apply_review_source()
        self._live_job = None
        self._live_rendered = False
        self._crop_drag_edge = None
        self._crop_drag_rect = None

    def reset_selection(self):
        self.selected_patch = None
        self._crop_drag_edge = None

    def reset_for_new_image(self):
        self.reset_selection()
        self.view_negative = False
        self.compare_default = False
        # mask_view intentionally PERSISTS across image switches: when
        # reviewing crops the user steps through the roll in that mode
        self._display_base_pil = None
        self._live_rendered = False
        if hasattr(self, "view_btn"):
            self._update_view_btn()
        if hasattr(self, "compare_btn"):
            self._update_compare_btn()

    def update_counts(self):
        # apply the persistent AI/analytical variant to this frame's params
        # before anything reads them (wheels, info, items, live render)
        self._apply_variant_to_current()
        self._update_variant_btn()
        self._update_count_label()
        self._update_bad_btn()
        # capture the freshly loaded image as the undecorated display base
        # and re-apply the persistent mask view to the new frame
        self._display_base_pil = self.pil_image
        if self.mask_view or self.show_clipping:
            self.pil_image = self._decorate(self._display_base_pil)
            self._redraw()
        self._refresh_histogram()
        self._sync_wheels()
        # re-apply the corrected render when entering a frame that has
        # corrections saved
        self._schedule_live_render()

    # ------------------------------------------------------------------
    # Layout / text hooks
    # ------------------------------------------------------------------

    def build_left_info(self, parent):
        self.add_legend(parent, [
            ("□", PATCH_COLORS["film_base"], "Local film base"),
            ("◈", "#ffd700", "GLOBAL film base (winner)"),
            ("□", PATCH_COLORS["shadows"], "Shadows patch (wb_low)"),
            ("□", PATCH_COLORS["highlights"], "Highlights patch (wb_high)"),
            ("┊", "#00ff88", "Corrected position"),
            ("▣", "#ff4444", "Bad inversion flag"),
            ("▒", "#ff6060", "Rejected outside crop (M)"),
            ("□", "#44ee44", "User crop (true content)"),
            ("■", "#ff3030", "Clipped highlights (L)"),
            ("■", "#4080ff", "Clipped shadows (L)"),
        ])
        self.add_hints(parent, (
            "Mouse:\n"
            "  Scroll — zoom in/out\n"
            "  Scroll (patch sel'd) —\n"
            "    resize patch (Shift=10px)\n"
            "  Middle drag — pan\n"
            "  Click — select patch\n"
            "  Ctrl+click — place patch\n"
            "  Ctrl+drag — zoom to rect\n"
            "Keys:\n"
            "  Space/B — next/prev image\n"
            "  1/2/3 — base/shadows/highl.\n"
            "  4/5/6/7 — black/gamma/\n"
            "    gloss/print exposure\n"
            "    (scroll adjusts value)\n"
            "  8 — crop: drag rect around\n"
            "    true content (scroll =\n"
            "    grow/shrink, C clears);\n"
            "    grabbing a crop EDGE\n"
            "    works in any mode\n"
            "  ] / [ — brighter/darker\n"
            "    (raises/lowers black, then\n"
            "    re-solves exposure to keep\n"
            "    highlights below clipping)\n"
            "  C — clear correction\n"
            "  V — inverted/negative view\n"
            "  X — corrected/default render\n"
            "  M — analysis areas / hide\n"
            "      rejected holder\n"
            "  T — histogram on/off\n"
            "  L — clip overlay on image\n"
            "      (red=blown, blue=crushed;\n"
            "      meter top-right always on)\n"
            "  G — toggle bad inversion\n"
            "  Note box — comment on sel.\n"
            "  H — hide/show markers\n"
            "  +/- zoom, Arrows pan, F fit\n"
            "Corrections re-render the\n"
            "inversion live.\n"
            "Shadows/highlights color\n"
            "wheels are in the right\n"
            "panel (set wb directly,\n"
            "start at the auto value;\n"
            "Ctrl-drag = fine tune)."
        ))

    def build_item_panel_footer(self, parent):
        """Two color wheels (shadows -> wb_low, highlights -> wb_high) below
        the item table's draggable sash. Each sets the abstract wb directly
        with live preview; the marker starts at the frame's auto-found wb and
        dragging stores a wb_override."""
        wrap = tk.Frame(parent, bg="#484848")
        wrap.pack(fill=tk.BOTH, expand=True, padx=6, pady=(4, 6))
        tk.Label(wrap, text="Shadows / Highlights wheels:", bg="#484848",
                 fg="#c0c0c0", font=("", 9, "bold")).pack(anchor="w")
        tk.Label(wrap, text="● gold pin = auto wb (divergence ref)",
                 bg="#484848", fg="#ffd700", font=("", 8)).pack(anchor="w")
        tk.Label(wrap, text="drag = set · Ctrl-drag = fine tune",
                 bg="#484848", fg="#9a9a9a", font=("", 8)).pack(anchor="w")
        self.wheels = {}
        self.wheel_readouts = {}
        for name in WB_NAMES:
            kind = WB_NAME_KIND[name]
            tk.Label(wrap, text=WB_NAME_LABEL[name], bg="#484848", fg="#cccccc",
                     font=("", 8)).pack(anchor="w", pady=(4, 0))
            wheel = ColorWheel(wrap, kind,
                               on_change=lambda wb, n=name: self._on_wheel_change(n, wb),
                               size=150)
            wheel.pack(anchor="center")
            self.wheels[name] = wheel
            ro = tk.Label(wrap, text="", bg="#484848", fg="#9fd0ff",
                          font=("Courier", 8))
            ro.pack(anchor="w")
            self.wheel_readouts[name] = ro
        # grow the wheels to fill the pane (wider panel / taller pane = finer
        # manual precision), bounded so both wheels stay fully visible
        parent.bind("<Configure>", self._resize_wheels)

    # reserved vertical space in the footer for the labels/readouts around the
    # two wheels (title + caption + per-wheel name + readout + paddings)
    _WHEEL_CHROME_H = 150

    def _resize_wheels(self, event):
        if not getattr(self, "wheels", None):
            return
        per_h = (event.height - self._WHEEL_CHROME_H) / 2.0
        size = int(min(event.width - 14, per_h))
        size -= size % 2                       # even sizes: less resize churn
        for wheel in self.wheels.values():
            wheel.resize(size)

    def build_feature_buttons(self, btn_frame, btn_cfg):
        self.count_label = tk.Label(btn_frame, text="Corrected patches: 0/3",
                                    bg="#484848", fg="#c0c0c0", font=("", 9))
        self.count_label.pack(anchor="e", pady=(0, 4))

        self.view_btn = tk.Button(btn_frame, text="Show Negative  (V)",
                                  command=self._toggle_view, **btn_cfg)
        self.view_btn.pack(fill=tk.X, pady=1)

        self.compare_btn = tk.Button(btn_frame, text="Show Default  (X)",
                                     command=self._toggle_compare, **btn_cfg)
        self.compare_btn.pack(fill=tk.X, pady=1)

        # spec 03: switch the render base between the analytical algorithm and
        # the vision-LLM AI variant (only frames with params_ai have one)
        self.variant_btn = tk.Button(btn_frame, text="Variant: Analytical  (A)",
                                     command=self._toggle_variant, **btn_cfg)
        self.variant_btn.pack(fill=tk.X, pady=1)

        # N: show the roll's vignette correction applied vs removed (before/after)
        self.vignette_btn = tk.Button(btn_frame, text="Vignette: On  (N)",
                                      command=self._toggle_vignette, **btn_cfg)
        self.vignette_btn.pack(fill=tk.X, pady=1)

        # R: while reviewing a calibration session, flip fitted (session) vs
        # live (current source-code) params/crop/vignette
        self.review_btn = tk.Button(btn_frame, text="Source: — (R)",
                                    command=self._toggle_review_source, **btn_cfg)
        self.review_btn.pack(fill=tk.X, pady=1)
        self._update_review_btn()

        self.mask_btn = tk.Button(btn_frame, text=MASK_VIEW_LABELS[0],
                                  command=self._cycle_mask_view, **btn_cfg)
        self.mask_btn.pack(fill=tk.X, pady=1)

        self.clip_btn = tk.Button(btn_frame, text="Clip Overlay  (L)",
                                   command=self._toggle_clipping, **btn_cfg)
        self.clip_btn.pack(fill=tk.X, pady=1)

        # brighter/darker: raise/lower paper black, then re-solve exposure to
        # keep highlights just below clipping (the user's manual brighten move)
        bright_row = tk.Frame(btn_frame, bg="#484848")
        bright_row.pack(fill=tk.X, pady=1)
        tk.Button(bright_row, text="Darker  ( [ )",
                  command=lambda: self._brighten(True), **btn_cfg
                  ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(bright_row, text="Brighter  ( ] )",
                  command=lambda: self._brighten(False), **btn_cfg
                  ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.bad_btn = tk.Button(btn_frame, text="Bad Inversion  (G)",
                                 command=self._toggle_bad_inversion, **btn_cfg)
        self.bad_btn.pack(fill=tk.X, pady=1)

        tk.Button(btn_frame, text="Clear Correction  (C)",
                  command=self._clear_correction, **btn_cfg).pack(fill=tk.X, pady=1)

    def bind_feature_keys(self):
        for key, patch in PATCH_KEYS.items():
            self.bind_key(f"<Key-{key}>",
                          lambda e, patch=patch: self._select_patch(patch))
        for key, name in PRINT_KEYS.items():
            self.bind_key(f"<Key-{key}>",
                          lambda e, name=name: self._select_patch(name))
        self.bind_key(f"<Key-{CROP_KEY}>",
                      lambda e: self._select_patch(CROP_NAME))
        self.bind_key("<c>", lambda e: self._clear_correction())
        self.bind_key("<C>", lambda e: self._clear_correction())
        self.bind_key("<v>", lambda e: self._toggle_view())
        self.bind_key("<V>", lambda e: self._toggle_view())
        self.bind_key("<g>", lambda e: self._toggle_bad_inversion())
        self.bind_key("<G>", lambda e: self._toggle_bad_inversion())
        self.bind_key("<x>", lambda e: self._toggle_compare())
        self.bind_key("<X>", lambda e: self._toggle_compare())
        self.bind_key("<a>", lambda e: self._toggle_variant())
        self.bind_key("<A>", lambda e: self._toggle_variant())
        self.bind_key("<n>", lambda e: self._toggle_vignette())
        self.bind_key("<N>", lambda e: self._toggle_vignette())
        self.bind_key("<r>", lambda e: self._toggle_review_source())
        self.bind_key("<R>", lambda e: self._toggle_review_source())
        self.bind_key("<m>", lambda e: self._cycle_mask_view())
        self.bind_key("<M>", lambda e: self._cycle_mask_view())
        self.bind_key("<t>", lambda e: self._toggle_histogram())
        self.bind_key("<T>", lambda e: self._toggle_histogram())
        self.bind_key("<l>", lambda e: self._toggle_clipping())
        self.bind_key("<L>", lambda e: self._toggle_clipping())
        # ] brighter / [ darker: black + exposure in unison (see _brighten)
        self.bind_key("<bracketright>", lambda e: self._brighten(False))
        self.bind_key("<bracketleft>", lambda e: self._brighten(True))

    def image_status_text(self, img_dict):
        p = img_dict.get("params") or {}
        exif = img_dict.get("exif") or {}
        fb = img_dict.get("film_base") or {}
        lines = [f"{img_dict['width']} × {img_dict['height']} px"]
        if exif.get("exposure_s"):
            lines.append(f"exif: {exif['exposure_s']:.4g}s f/{exif.get('aperture') or 0:.3g} "
                         f"ISO{exif.get('iso') or 0:.0f}")
        if fb.get("is_global_winner"):
            lines.append("★ GLOBAL film-base winner")
        elif fb.get("global_winner_stem"):
            lines.append(f"global base: {fb['global_winner_stem']}")
        if p:
            lines.append(f"Dmax={p['D_max']:.2f} blk={p['black']:.3f} "
                         f"exp={p['exposure']:.3f} g={p.get('gamma', 0):.2f}")
        # spec 03: vision-LLM scene reading + AI-variant divergence
        sc = img_dict.get("scene")
        if sc:
            lines.append(f"scene (LLM): {sc.get('scene')}/{sc.get('mood')}/"
                         f"{sc.get('warmth')}/{sc.get('contrast')}")
            if sc.get("rationale"):
                lines.append(f"  “{sc['rationale']}”")
        if img_dict.get("params_ai"):
            mark = "→ AI variant ACTIVE (A)" if self.variant == "ai" else \
                   "AI variant available (A)"
            lines.append(mark)
        elif self.variant == "ai":
            lines.append("(no AI variant for this frame)")
        v = img_dict.get("vignette")
        if v:
            lines.append(f"vignette: s={v['strength']:.2f} r={v['radius']:.2f} "
                         f"st={v['steepness']:.2f}")
        else:
            lines.append("vignette: none")
        t = img_dict.get("processing_time_s")
        if t is not None:
            lines.append(f"processed in {t:.2f}s")
        if self.run_wall_time_s is not None:
            lines.append(f"run total {self.run_wall_time_s:.1f}s")
        return "\n".join(lines)

    def default_info_text(self):
        return ("Nothing selected.\n"
                "Click a patch rect or press 1/2/3 (film base/shadows/highlights)\n"
                "or 4/5/6/7 (paper black/gamma/gloss/print exposure).")

    # ------------------------------------------------------------------
    # Note hooks
    # ------------------------------------------------------------------

    def get_selected_note(self):
        if self.selected_patch is None:
            return None
        ann = self.annotations[self.images[self.current_idx]["stem"]]
        return ann["patch_notes"].get(self.selected_patch, "")

    def set_selected_note(self, text):
        if self.selected_patch is None:
            return
        ann = self.annotations[self.images[self.current_idx]["stem"]]
        if text:
            ann["patch_notes"][self.selected_patch] = text
        else:
            ann["patch_notes"].pop(self.selected_patch, None)

    def _update_count_label(self):
        ann = self.annotations[self.images[self.current_idx]["stem"]]
        n = len(ann["patch_corrections"])
        m = len(ann["print_overrides"])
        w = len(ann["wb_overrides"])
        crop = ", crop✓" if ann.get("crop_correction") else ""
        self.count_label.config(
            text=f"Corrected: {n}/3 patches, {w}/2 wb, {m}/4 print{crop}")

    # ------------------------------------------------------------------
    # View toggle (inverted preview <-> raw negative)
    # ------------------------------------------------------------------

    def _toggle_view(self):
        img_dict = self.images[self.current_idx]
        neg_path = img_dict.get("negative_path")
        if not self.view_negative and (not neg_path or not Path(neg_path).exists()):
            return
        self.view_negative = not self.view_negative
        if self.view_negative:
            # darktable float/16-bit TIFFs don't render via PIL directly;
            # show the linearized negative through the pipeline loader
            lin = self._neg_lin(img_dict)
            if lin is not None:
                arr = (nm.linear_to_srgb(np.clip(lin, 0.0, 1.0)) * 255.0
                       + 0.5).astype(np.uint8)
                self._set_display_image(Image.fromarray(arr))
            else:
                self._set_display_image(Image.open(neg_path))
        else:
            self._live_rendered = False
            try:
                self._set_display_image(self._load_display_pil(img_dict))
            except Exception:
                pass
            self._schedule_live_render(delay_ms=10)
        self._update_view_btn()

    def _update_view_btn(self):
        label = "Show Inverted  (V)" if self.view_negative else "Show Negative  (V)"
        self.view_btn.config(text=label,
                             fg="#ffff88" if self.view_negative else "white")

    def _has_corrections(self):
        ann = self.annotations[self.images[self.current_idx]["stem"]]
        return bool(ann["patch_corrections"] or ann["print_overrides"]
                    or ann["wb_overrides"] or ann.get("crop_correction"))

    # ------------------------------------------------------------------
    # Color-wheel shadows/highlights
    # ------------------------------------------------------------------

    def _effective_wb(self, img_dict, name):
        """wb the wheel should show: saved override, else the auto-found
        params wb, else neutral."""
        ann = self.annotations[img_dict["stem"]]
        ovr = ann["wb_overrides"].get(WB_NAME_OVR[name])
        if ovr:
            return list(ovr)
        applied = (img_dict.get("params") or {}).get(WB_NAME_PARAM[name])
        return list(applied) if applied else [1.0, 1.0, 1.0]

    def _sync_wheels(self):
        """Position both wheel markers + readouts for the current frame."""
        if not getattr(self, "wheels", None):
            return
        img_dict = self.images[self.current_idx]
        params = img_dict.get("params") or {}
        for name, wheel in self.wheels.items():
            wb = self._effective_wb(img_dict, name)
            wheel.set_wb(wb)
            wheel.set_auto(params.get(WB_NAME_PARAM[name]))   # fixed auto pin
            self._update_wheel_readout(name, wb)

    def _update_wheel_readout(self, name, wb=None):
        if not getattr(self, "wheel_readouts", None):
            return
        img_dict = self.images[self.current_idx]
        if wb is None:
            wb = self._effective_wb(img_dict, name)
        ann = self.annotations[img_dict["stem"]]
        tag = "OVR" if WB_NAME_OVR[name] in ann["wb_overrides"] else "auto"
        self.wheel_readouts[name].config(
            text=f"{tag} ({wb[0]:.2f},{wb[1]:.2f},{wb[2]:.2f})",
            fg="#9fff9f" if tag == "OVR" else "#9fd0ff")

    def _on_wheel_change(self, name, wb):
        img_dict = self.images[self.current_idx]
        stem = img_dict["stem"]
        self.annotations[stem]["wb_overrides"][WB_NAME_OVR[name]] = \
            [float(v) for v in wb]
        self.selected_patch = name
        self._auto_save(stem)
        self._update_wheel_readout(name, wb)
        self._update_count_label()
        self._populate_items_list()
        self._update_info_from_selection()
        self._redraw_markers()
        self._schedule_live_render()

    # ------------------------------------------------------------------
    # Crop correction helpers
    # ------------------------------------------------------------------

    def _auto_content_rect(self, img_dict):
        """Content rect implied by the automatic border trim."""
        l, t, r, b = img_dict.get("border") or (0, 0, 0, 0)
        return [l, t, img_dict["width"] - l - r, img_dict["height"] - t - b]

    def _crop_rect(self, img_dict):
        return self.annotations[img_dict["stem"]].get("crop_correction")

    # Table coordinates are shown as resolution-independent fractions of the
    # frame (x and w over width, y and h over height) — the UI works in pixels
    # internally but never *displays* an export-resolution-tied number.
    def _fmt_rect(self, rect, img_dict):
        """Rect [x, y, w, h] in pixels -> 'x,y,w,h' as fractions of the frame."""
        if not rect:
            return ""
        w, h = img_dict["width"], img_dict["height"]
        x, y, rw, rh = rect
        return f"{x / w:.3f},{y / h:.3f},{rw / w:.3f},{rh / h:.3f}"

    def _fmt_pos(self, pt, img_dict):
        """Point [x, y, ...] in pixels -> 'x,y' as fractions of the frame."""
        if not pt:
            return "—"
        w, h = img_dict["width"], img_dict["height"]
        return f"{pt[0] / w:.3f},{pt[1] / h:.3f}"

    # ------------------------------------------------------------------
    # Analysis-area mask view (audit holder/border detection)
    # ------------------------------------------------------------------

    def _analysis_mask(self, img_dict):
        """Per-pixel analysis mask (uint8 codes) for the frame, or None.

        Computed LIVE from the detected content-crop border via the SAME
        auto_negadoctor.build_analysis_mask the pipeline uses, so the crop
        view always reflects the current algorithm — there is no baked
        {stem}_analysis_mask.png."""
        if self._amask_cache_stem == img_dict["stem"]:
            return self._amask_cache
        mask = None
        border = img_dict.get("border")
        w, h = img_dict.get("width"), img_dict.get("height")
        if border is not None and w and h:
            try:
                import auto_negadoctor
                mask = auto_negadoctor.build_analysis_mask(
                    (int(h), int(w)), border)
            except Exception:
                mask = None
        self._amask_cache_stem = img_dict["stem"]
        self._amask_cache = mask
        return mask

    # --- clipping indication ---------------------------------------------
    # A channel at the 8-bit ceiling/floor of the displayed sRGB render is
    # clipped: 255 == highlights blown (linear >= 1.0), 0 == shadows crushed.
    # The on-image overlay (L, default off) paints those pixels a "wrong"
    # colour; the meter + histogram spikes always report the fractions.
    CLIP_HI_LEVEL = 255
    CLIP_LO_LEVEL = 0
    CLIP_HI_COLOR = (255, 0, 0)        # highlight clip -> red
    CLIP_LO_COLOR = (40, 90, 255)      # shadow clip   -> blue
    CLIP_METER_FULL_PCT = 2.0          # bar = full at this % clipped
    CLIP_BUDGET_PCT = 0.3              # threshold tick (= PRINT_CLIP_BUDGET)

    def _decorate(self, pil):
        """Apply the current mask view + clip overlay to a base image."""
        if pil is None:
            return pil
        if self.mask_view == 0 and not self.show_clipping:
            return pil
        arr = np.array(pil.convert("RGB"))
        # Clip masks come from the TRUE rendered values, captured before any
        # mask blanking turns rejected pixels black (which would otherwise read
        # as bogus shadow clipping).
        clip_hi = clip_lo = None
        if self.show_clipping:
            clip_hi = (arr >= self.CLIP_HI_LEVEL).any(axis=2)
            clip_lo = (arr <= self.CLIP_LO_LEVEL).any(axis=2)
        if self.mask_view != 0:
            mask = self._analysis_mask(self.images[self.current_idx])
            if mask is not None and mask.shape[:2] != arr.shape[:2]:
                mask = None
            if self.mask_view == 1:
                if mask is not None:
                    for code, (r, g, b, a) in MASK_TINTS.items():
                        sel = mask == code
                        if sel.any():
                            f = a / 255.0
                            arr[sel] = (arr[sel] * (1 - f)
                                        + np.array([r, g, b], dtype=np.float32) * f
                                        ).astype(np.uint8)
            else:
                # show the frame WITHOUT the rejected area: the user's crop
                # correction is authoritative when present, else holder+border
                crop = self.annotations[self.images[self.current_idx]["stem"]] \
                    .get("crop_correction")
                keep = None
                if crop:
                    x, y, cw, ch = [int(v) for v in crop]
                    keep = np.zeros(arr.shape[:2], dtype=bool)
                    keep[y:y + ch, x:x + cw] = True
                elif mask is not None:
                    keep = ~((mask == 1) | (mask == 2))
                if keep is not None:
                    arr[~keep] = 0
                    if clip_hi is not None:
                        clip_hi &= keep
                        clip_lo &= keep
        if clip_hi is not None:
            arr[clip_hi] = self.CLIP_HI_COLOR
            arr[clip_lo] = self.CLIP_LO_COLOR
        return Image.fromarray(arr)

    def _set_display_image(self, pil):
        """Central setter: remembers the undecorated image and applies the
        current mask view before display."""
        self._display_base_pil = pil
        self.pil_image = self._decorate(pil)
        self._refresh_histogram()
        self._redraw()

    def _cycle_mask_view(self):
        self.mask_view = (self.mask_view + 1) % 3
        if self._display_base_pil is None:
            self._display_base_pil = self.pil_image
        self.pil_image = self._decorate(self._display_base_pil)
        self._update_mask_btn()
        self._refresh_histogram()
        self._redraw()

    def _update_mask_btn(self):
        self.mask_btn.config(text=MASK_VIEW_LABELS[self.mask_view],
                             fg="#ffff88" if self.mask_view else "white")

    # ------------------------------------------------------------------
    # Histogram of the displayed (converted) image
    # ------------------------------------------------------------------

    HIST_BINS = 128

    def _refresh_histogram(self):
        """Per-channel histogram of the undecorated displayed image. In
        hide-rejected mode the holder/border pixels are excluded, so the
        histogram shows the photo content only (the inverted holder would
        otherwise fake a clipped-whites spike)."""
        self._hist_data = None
        self._clip_stats = None
        if self._display_base_pil is None:
            return
        try:
            arr = np.asarray(self._display_base_pil.convert("RGB"))
        except Exception:
            return
        pixels = arr.reshape(-1, 3)
        if self.mask_view == 2:
            crop = self.annotations[self.images[self.current_idx]["stem"]] \
                .get("crop_correction")
            if crop:
                x, y, cw, ch = [int(v) for v in crop]
                pixels = arr[y:y + ch, x:x + cw].reshape(-1, 3)
            else:
                mask = self._analysis_mask(self.images[self.current_idx])
                if mask is not None and mask.shape[:2] == arr.shape[:2]:
                    keep = ~((mask == 1) | (mask == 2))
                    pixels = arr[keep]
        if len(pixels) == 0:
            return
        # clip fractions over the SAME pixel set the histogram covers (content
        # only in hide-rejected mode) — drives the meter + histogram spikes
        total = len(pixels)
        hi = int(np.count_nonzero((pixels >= self.CLIP_HI_LEVEL).any(axis=1)))
        lo = int(np.count_nonzero((pixels <= self.CLIP_LO_LEVEL).any(axis=1)))
        self._clip_stats = {"hi": hi / total * 100.0,
                            "lo": lo / total * 100.0, "total": total}
        hists = []
        for c in range(3):
            h, _edges = np.histogram(pixels[:, c], bins=self.HIST_BINS,
                                     range=(0, 255))
            hists.append(h.astype(np.float64))
        peak = max(float(h.max()) for h in hists)
        if peak <= 0:
            return
        self._hist_data = [h / peak for h in hists]

    def _draw_histogram(self):
        if not self.show_histogram or not getattr(self, "_hist_data", None):
            return
        cw = self.canvas.winfo_width()
        pw, ph = 150, 72
        x0, y0 = cw - pw - 12, 10
        self.canvas.create_rectangle(x0, y0, x0 + pw, y0 + ph,
                                     fill="#1c1c1c", outline="#666666",
                                     tags="histogram")
        colors = ("#ff6666", "#66dd66", "#7799ff")
        n = self.HIST_BINS
        for c in range(3):
            pts = []
            for i in range(n):
                px = x0 + 2 + (pw - 4) * i / (n - 1)
                py = y0 + ph - 2 - (ph - 6) * min(self._hist_data[c][i], 1.0)
                pts.extend((px, py))
            self.canvas.create_line(*pts, fill=colors[c], width=1,
                                    tags="histogram")
        # clipping spikes: red at the right edge (blown highlights), blue at
        # the left edge (crushed shadows), height ~ the clipped fraction
        stats = getattr(self, "_clip_stats", None)
        if stats:
            track = ph - 6
            for pct, color, ex in ((stats["hi"], "#ff3030", x0 + pw - 2),
                                   (stats["lo"], "#4080ff", x0 + 2)):
                if pct <= 0:
                    continue
                bh = track * min(1.0, pct / self.CLIP_METER_FULL_PCT)
                self.canvas.create_line(ex, y0 + ph - 2, ex, y0 + ph - 2 - bh,
                                        fill=color, width=3, tags="histogram")
        if self.mask_view == 2:
            self.canvas.create_text(x0 + 3, y0 + 2, anchor="nw",
                                    text="content only", fill="#aaaaaa",
                                    font=("Courier", 7), tags="histogram")

    def _draw_clip_meter(self):
        """A VU-style clip-level meter top-right (below the histogram): one
        bar per direction (H = blown highlights, S = crushed shadows) filling
        to CLIP_METER_FULL_PCT, with a tick at the clip budget and the exact
        percentage. Always shown; (L) additionally tints clipped pixels on the
        image."""
        stats = getattr(self, "_clip_stats", None)
        cw = self.canvas.winfo_width()
        pw, rh, pad = 150, 13, 4
        x0 = cw - pw - 12
        y0 = (10 + 72 + 8) if (self.show_histogram
                               and getattr(self, "_hist_data", None)) else 10
        ph = 16 + 2 * (rh + pad)
        self.canvas.create_rectangle(x0, y0, x0 + pw, y0 + ph, fill="#1c1c1c",
                                     outline="#666666", tags="clipmeter")
        title = "clipping" + ("  L:ON" if self.show_clipping else "  (L)")
        self.canvas.create_text(x0 + 4, y0 + 2, anchor="nw", text=title,
                                fill="#dddddd" if self.show_clipping else "#aaaaaa",
                                font=("Courier", 7, "bold"), tags="clipmeter")
        rows = (("H", "#ff3030", stats["hi"] if stats else None),
                ("S", "#4080ff", stats["lo"] if stats else None))
        bx0, bx1 = x0 + 18, x0 + pw - 42
        for i, (lab, color, val) in enumerate(rows):
            ry = y0 + 15 + i * (rh + pad)
            self.canvas.create_text(x0 + 4, ry + rh / 2, anchor="w", text=lab,
                                    fill="#cccccc", font=("Courier", 8),
                                    tags="clipmeter")
            self.canvas.create_rectangle(bx0, ry, bx1, ry + rh, fill="#000000",
                                         outline="#555555", tags="clipmeter")
            if val is not None:
                frac = min(1.0, val / self.CLIP_METER_FULL_PCT)
                if frac > 0:
                    self.canvas.create_rectangle(
                        bx0, ry, bx0 + (bx1 - bx0) * frac, ry + rh,
                        fill=color, outline="", tags="clipmeter")
                txt = f"{val:.2f}%"
            else:
                txt = "—"
            # budget tick (highlights only)
            if lab == "H":
                tx = bx0 + (bx1 - bx0) * min(1.0, self.CLIP_BUDGET_PCT
                                            / self.CLIP_METER_FULL_PCT)
                self.canvas.create_line(tx, ry - 1, tx, ry + rh + 1,
                                        fill="#ffd700", width=1, tags="clipmeter")
            over = val is not None and lab == "H" and val > self.CLIP_BUDGET_PCT
            self.canvas.create_text(bx1 + 3, ry + rh / 2, anchor="w", text=txt,
                                    fill="#ff6060" if over else "#cccccc",
                                    font=("Courier", 8), tags="clipmeter")

    def _toggle_compare(self):
        """X: flip between the corrected render and the algorithm's default."""
        if not self._has_corrections():
            return
        self.compare_default = not self.compare_default
        self._update_compare_btn()
        self._schedule_live_render(delay_ms=10)

    def _update_compare_btn(self):
        label = ("Show Corrected  (X)" if self.compare_default
                 else "Show Default  (X)")
        self.compare_btn.config(text=label,
                                fg="#ffff88" if self.compare_default else "white")

    # ------------------------------------------------------------------
    # AI variant switch (spec 03)
    # ------------------------------------------------------------------

    def _frame_has_ai(self, img_dict=None):
        img_dict = img_dict or self.images[self.current_idx]
        return bool(img_dict.get("params_ai"))

    def _apply_variant_to_current(self):
        """Swap the active frame's render base ('params') to the chosen variant.
        Everything (render, info, items, wheels, histogram) reads img_dict
        ['params'], so this single swap makes the whole UI reflect the variant.
        params_analytical / params_ai are preserved so the swap is reversible."""
        img_dict = self.images[self.current_idx]
        # ensure the analytical baseline is always available to restore from
        img_dict.setdefault("params_analytical", img_dict.get("params"))
        if self.variant == "ai" and img_dict.get("params_ai"):
            base = img_dict["params_ai"]
            base_hex = img_dict.get("params_ai_hex")
        else:
            base = img_dict.get("params_analytical")
            base_hex = img_dict.get("params_hex")
        if base is not None:
            img_dict["params"] = base
            if base_hex is not None:
                img_dict["params_hex"] = base_hex

    def _toggle_variant(self):
        """A: flip the render base between the analytical algorithm and the
        vision-LLM AI variant. No-op on frames without an AI variant."""
        if not self._frame_has_ai():
            self._set_info_text("No AI variant for this frame (run with "
                                "--ai-tune / the AI action to compute one).")
            return
        self.variant = "ai" if self.variant == "analytical" else "analytical"
        self._apply_variant_to_current()
        self._update_variant_btn()
        self._sync_wheels()
        self._populate_items_list()
        self._update_info_from_selection()
        self._redraw_markers()
        self._schedule_live_render(delay_ms=10)

    def _update_variant_btn(self):
        if not hasattr(self, "variant_btn"):
            return
        has_ai = self._frame_has_ai()
        is_ai = self.variant == "ai" and has_ai
        label = f"Variant: {'AI (LLM)' if is_ai else 'Analytical'}  (A)"
        self.variant_btn.config(
            text=label,
            fg="#9fff9f" if is_ai else ("white" if has_ai else "#888888"))

    def _toggle_vignette(self):
        """N: apply / remove the roll's vignette correction in the preview so its
        effect is visible (before/after). Reloads the negative (the correction is
        baked into the linear buffer)."""
        self.vignette_on = not self.vignette_on
        self._neg_cache_key = None      # force reload with/without vignette
        self._update_vignette_btn()
        self._set_info_text("Vignette correction "
                            f"{'ON' if self.vignette_on else 'OFF'}.")
        self._schedule_live_render(delay_ms=10)

    def _update_vignette_btn(self):
        if not hasattr(self, "vignette_btn"):
            return
        on = self.vignette_on
        self.vignette_btn.config(
            text=f"Vignette: {'On' if on else 'OFF'}  (N)",
            fg="white" if on else "#ffb060")

    def _apply_review_source(self):
        """Swap each review frame's payload (params / border / vignette) to the
        active source (fitted vs live) IN PLACE, so every existing render path
        (inversion, crop mask, vignette load) uses it unchanged."""
        if not getattr(self, "review_mode", False):
            return
        for img in self.images:
            rev = img.get("review")
            if not rev:
                continue
            src = rev.get(self.review_source) or {}
            kind = img.get("review_kind")
            if kind == "inversion":
                if src.get("params") is not None:
                    img["params"] = src["params"]
                    img["params_analytical"] = src["params"]
                if src.get("params_hex") is not None:
                    img["params_hex"] = src["params_hex"]
            elif kind == "crop":
                if src.get("border") is not None:
                    img["border"] = list(src["border"])
            elif kind == "vignette":
                img["vignette"] = src.get("vignette")
        self._neg_cache_key = None      # vignette/params may have changed
        self._amask_cache_stem = None   # border may have changed

    def _toggle_review_source(self):
        """R: flip the preview between the FITTED params (this calibration
        session's result) and the LIVE params (the current source-code
        constants). Only meaningful while reviewing a session."""
        if not getattr(self, "review_mode", False):
            self._set_info_text("Not reviewing a calibration session "
                                "(open one via run_calibration.py --review).")
            return
        self.review_source = "live" if self.review_source == "fitted" else "fitted"
        self._apply_review_source()
        self._update_review_btn()
        self._sync_wheels()
        self._populate_items_list()
        self._update_info_from_selection()
        self._redraw_markers()
        self._schedule_live_render(delay_ms=10)

    def _update_review_btn(self):
        if not hasattr(self, "review_btn"):
            return
        if not self.review_mode:
            self.review_btn.config(text="Source: — (R)", fg="#888888")
            return
        fitted = self.review_source == "fitted"
        self.review_btn.config(
            text=f"Source: {'FITTED' if fitted else 'live (code)'}  (R)",
            fg="#9fd0ff" if fitted else "#ffd080")

    def _toggle_histogram(self):
        self.show_histogram = not self.show_histogram
        self._redraw_markers()

    def _toggle_clipping(self):
        """L: toggle the on-image clip overlay (red highlights / blue shadows).
        The meter + histogram spikes report the fractions regardless."""
        self.show_clipping = not self.show_clipping
        if self._display_base_pil is not None:
            self.pil_image = self._decorate(self._display_base_pil)
        self._update_clip_btn()
        self._redraw()

    def _update_clip_btn(self):
        if not hasattr(self, "clip_btn"):
            return
        on = self.show_clipping
        self.clip_btn.config(text="Clip Overlay ✓ (L)" if on else "Clip Overlay  (L)",
                             fg="#ff6666" if on else "white")

    def _toggle_bad_inversion(self):
        stem = self.images[self.current_idx]["stem"]
        ann = self.annotations[stem]
        ann["bad_inversion"] = not ann["bad_inversion"]
        self._auto_save(stem)
        self._update_bad_btn()
        self._redraw_markers()

    def _update_bad_btn(self):
        ann = self.annotations[self.images[self.current_idx]["stem"]]
        bad = ann["bad_inversion"]
        self.bad_btn.config(fg="#ff6666" if bad else "white",
                            text="Bad Inversion ✓ (G)" if bad else "Bad Inversion  (G)")

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def overlay_tags(self):
        return ("patches", "corrections", "labels", "badges", "histogram",
                "clipmeter")

    def _draw_rect(self, rect, color, tags, width=2, dash=None, label=None):
        x, y, w, h = rect
        x0, y0 = image_to_canvas(x, y, self.offset_x, self.offset_y, self.zoom)
        x1, y1 = image_to_canvas(x + w, y + h, self.offset_x, self.offset_y, self.zoom)
        kwargs = {"outline": color, "width": width, "tags": tags}
        if dash:
            kwargs["dash"] = dash
        self.canvas.create_rectangle(x0, y0, x1, y1, **kwargs)
        if label:
            self.canvas.create_text(x0, max(y0 - 8, 8), text=label, fill=color,
                                    anchor="w", font=("Courier", 9, "bold"),
                                    tags="labels")

    def draw_overlays(self):
        img_dict = self.images[self.current_idx]
        ann = self.annotations[img_dict["stem"]]
        fb = img_dict.get("film_base") or {}
        patches = img_dict.get("patches") or {}

        for patch in PATCHES:
            det = self._detected_rect(img_dict, patch)
            sel = (self.selected_patch == patch)
            color = "#ffff00" if sel else PATCH_COLORS[patch]
            if det:
                label = PATCH_SHORT[patch]
                if patch != "film_base" and patches.get(patch, {}).get("used_fallback"):
                    label += " (fb!)"
                self._draw_rect(det, color, "patches", width=3 if sel else 2,
                                label=label)
            corr = ann["patch_corrections"].get(patch)
            if corr:
                self._draw_rect(corr, "#00ff88", "corrections", width=2,
                                dash=(6, 4), label=f"{PATCH_SHORT[patch]}→")

        # User crop correction (true content area)
        crop = ann.get("crop_correction")
        if crop:
            sel = self.selected_patch == CROP_NAME
            self._draw_rect(crop, "#ffff00" if sel else "#44ee44", "patches",
                            width=3 if sel else 2, label="crop")
        if self.selected_patch == CROP_NAME:
            self._draw_rect(self._auto_content_rect(img_dict), "#999999",
                            "patches", width=1, dash=(4, 4), label="auto")

        # Global film base marker
        if fb.get("is_global_winner") and fb.get("local"):
            r = fb["local"]["rect"]
            grown = [r[0] - 4, r[1] - 4, r[2] + 8, r[3] + 8]
            self._draw_rect(grown, "#ffd700", "patches", width=2, label="GLOBAL")
        elif fb.get("global_winner_stem"):
            self.canvas.create_text(
                10, 10, anchor="nw", tags="badges",
                text=f"global base from {fb['global_winner_stem']}",
                fill="#ffd700", font=("Courier", 9, "bold"))

        if ann["bad_inversion"]:
            w, h = img_dict["width"], img_dict["height"]
            self._draw_rect([2, 2, w - 4, h - 4], "#ff4444", "badges", width=3)
            self.canvas.create_text(
                10, 28, anchor="nw", tags="badges", text="BAD INVERSION",
                fill="#ff4444", font=("Courier", 10, "bold"))

        if self.mask_view == 1:
            self.canvas.create_text(
                10, 64, anchor="nw", tags="badges",
                text="ANALYSIS CROP: red = rejected (drag to correct the crop)",
                fill="#ff8888", font=("Courier", 9, "bold"))
        elif self.mask_view == 2:
            self.canvas.create_text(
                10, 64, anchor="nw", tags="badges",
                text="REJECTED OUTSIDE-CROP AREA HIDDEN (drag to correct the crop)",
                fill="#ff8888", font=("Courier", 9, "bold"))

        self._draw_histogram()
        self._draw_clip_meter()

        if self.view_negative:
            self.canvas.create_text(
                10, 46, anchor="nw", tags="badges", text="NEGATIVE VIEW",
                fill="#88ccff", font=("Courier", 9, "bold"))
        elif self.compare_default and self._has_corrections():
            self.canvas.create_text(
                10, 46, anchor="nw", tags="badges",
                text="DEFAULT RENDER (X: show corrected)",
                fill="#ffcc00", font=("Courier", 9, "bold"))
        elif self._live_rendered:
            self.canvas.create_text(
                10, 46, anchor="nw", tags="badges",
                text="RE-RENDERED FROM CORRECTIONS (X: default)",
                fill="#00ff88", font=("Courier", 9, "bold"))

    # ------------------------------------------------------------------
    # Interaction hooks
    # ------------------------------------------------------------------

    def _find_nearest_patch(self, canvas_x, canvas_y, max_dist=18.0):
        """Nearest patch (by effective rect center) within max_dist canvas px
        of the rect (inside counts as 0)."""
        img_dict = self.images[self.current_idx]
        best, best_dist = None, max_dist
        for patch in PATCHES:
            rect = self._effective_rect(img_dict, patch)
            if not rect:
                continue
            x, y, w, h = rect
            x0, y0 = image_to_canvas(x, y, self.offset_x, self.offset_y, self.zoom)
            x1, y1 = image_to_canvas(x + w, y + h, self.offset_x, self.offset_y, self.zoom)
            dx = max(x0 - canvas_x, 0, canvas_x - x1)
            dy = max(y0 - canvas_y, 0, canvas_y - y1)
            d = (dx * dx + dy * dy) ** 0.5
            if d < best_dist:
                best, best_dist = patch, d
        return best

    def _select_patch(self, patch):
        self.selected_patch = patch
        self._update_info_from_selection()
        self._redraw_markers()

    def on_click(self, canvas_x, canvas_y):
        patch = self._find_nearest_patch(canvas_x, canvas_y)
        if patch is None:
            self._clear_selection()
            self._sync_item_list_selection()
            return
        self._select_patch(patch)

    def on_ctrl_click(self, canvas_x, canvas_y):
        """Place the selected (or nearest) patch window at the click point."""
        patch = self.selected_patch if self.selected_patch in PATCHES else None
        patch = patch or self._find_nearest_patch(
            canvas_x, canvas_y, max_dist=float("inf")) or "film_base"
        img_dict = self.images[self.current_idx]
        stem = img_dict["stem"]
        ix, iy = canvas_to_image(canvas_x, canvas_y,
                                 self.offset_x, self.offset_y, self.zoom)
        win = self._default_win(img_dict, patch)
        x = int(round(ix - win / 2))
        y = int(round(iy - win / 2))
        x = max(0, min(img_dict["width"] - win, x))
        y = max(0, min(img_dict["height"] - win, y))
        self.annotations[stem]["patch_corrections"][patch] = [x, y, win, win]
        self.selected_patch = patch
        self._auto_save(stem)
        self._redraw_markers()
        self._update_count_label()
        self._populate_items_list()
        self._update_info_from_selection()
        self._schedule_live_render()

    def on_scroll_override(self, event):
        """When a patch is selected, scroll RESIZES its window (center kept,
        scroll up = grow, Shift = 10px steps). Works for detected patches too:
        the first scroll seeds a correction from the detected rect, so the
        adjusted size lands in the annotations and the regression data."""
        if self.selected_patch is None:
            return False
        # wheels are edited by dragging the wheel widget, not by scrolling the
        # image — let the canvas zoom normally when one is selected
        if self.selected_patch in WB_NAMES:
            return False
        if event.num == 4:
            delta = 1
        elif event.num == 5:
            delta = -1
        else:
            delta = 1 if event.delta > 0 else -1

        if self.selected_patch in PRINT_PARAMS:
            return self._adjust_print_param(self.selected_patch, delta,
                                            bool(event.state & 0x1))
        step = 10 if (event.state & 0x1) else 2

        img_dict = self.images[self.current_idx]
        stem = img_dict["stem"]

        if self.selected_patch == CROP_NAME:
            # scroll grows/shrinks the crop rect on all sides
            rect = self._crop_rect(img_dict) or self._auto_content_rect(img_dict)
            x, y, cw, ch = [int(v) for v in rect]
            d = delta * step
            x, y = x - d, y - d
            cw, ch = cw + 2 * d, ch + 2 * d
            x = max(0, x)
            y = max(0, y)
            cw = max(40, min(img_dict["width"] - x, cw))
            ch = max(40, min(img_dict["height"] - y, ch))
            self.annotations[stem]["crop_correction"] = [x, y, cw, ch]
            self._auto_save(stem)
            self._redraw_markers()
            self._update_count_label()
            self._populate_items_list()
            self._update_info_from_selection()
            self._schedule_live_render()
            return True

        rect = self._effective_rect(img_dict, self.selected_patch)
        if rect is None:
            return True   # nothing placed yet: Ctrl+Click first
        x, y, w, h = [int(v) for v in rect]
        min_sz = 10
        max_sz = int(min(img_dict["width"], img_dict["height"]) * 0.4)
        new_sz = max(min_sz, min(max_sz, w + delta * step))
        if new_sz == w:
            return True
        cx, cy = x + w / 2.0, y + h / 2.0
        nx = int(round(cx - new_sz / 2.0))
        ny = int(round(cy - new_sz / 2.0))
        nx = max(0, min(img_dict["width"] - new_sz, nx))
        ny = max(0, min(img_dict["height"] - new_sz, ny))
        self.annotations[stem]["patch_corrections"][self.selected_patch] = \
            [nx, ny, new_sz, new_sz]
        self._auto_save(stem)
        self._redraw_markers()
        self._update_count_label()
        self._populate_items_list()
        self._update_info_from_selection()
        self._schedule_live_render()
        return True

    def _adjust_print_param(self, name, delta, big):
        """Scroll on a selected print param: adjust its override with live
        preview. The first adjustment seeds the override from the applied
        value."""
        img_dict = self.images[self.current_idx]
        stem = img_dict["stem"]
        params = img_dict.get("params") or {}
        ann = self.annotations[stem]
        cur = ann["print_overrides"].get(name, params.get(name))
        if cur is None:
            return True
        step = PRINT_STEP[name][1 if big else 0]
        new_val = nm.clamp(float(cur) + delta * step, self._print_range(name))
        ann["print_overrides"][name] = new_val
        self._auto_save(stem)
        self._update_count_label()
        self._populate_items_list()
        self._update_info_from_selection()
        self._schedule_live_render()
        return True

    # --- "brighter" / "darker" (black + exposure in unison) ----------------

    def _brighten_content_pixels(self, img_dict, lin):
        """Subsampled (N,3) float64 content pixels for the clip solve — the
        user's crop rect when present (the authoritative content area), else the
        auto border trim. Same stride as auto_negadoctor.tune_print_params."""
        import auto_negadoctor as anp
        h, w = lin.shape[:2]
        crop = self.annotations[img_dict["stem"]].get("crop_correction")
        if crop:
            x, y, cw, ch = [int(v) for v in crop]
            l, t = max(x, 0), max(y, 0)
            r, b = max(w - x - cw, 0), max(h - y - ch, 0)
        else:
            l, t, r, b = img_dict.get("border") or (0, 0, 0, 0)
        s = max(1, int(round(w * anp.PRINT_TUNE_SUBSAMPLE_FRAC)))
        region = np.asarray(lin[t:h - b:s, l:w - r:s], dtype=np.float64)
        if region.size == 0:
            region = np.asarray(lin, dtype=np.float64)
        return region.reshape(-1, 3)

    @staticmethod
    def _clip_fraction(params, content):
        out = nm.render_negadoctor(content, params)
        return float(np.mean((out >= CLIP_OUT_THR).any(axis=1)))

    @staticmethod
    def _high_pct(params, content):
        """The rendered highlight level: P99.9 of the per-pixel mean over the
        content (same statistic auto_negadoctor.tune_print_params pins)."""
        import auto_negadoctor as anp
        out = nm.render_negadoctor(content, params)
        return float(np.percentile(out.mean(axis=1), anp.PRINT_HI_PCT))

    def _exposure_for_high_pct(self, params, content, target):
        """Exposure (in nm.EXPOSURE_RANGE) whose rendered high percentile equals
        `target`. The high percentile is monotonic increasing in exposure, so
        bisect; clamps to the range ends when target is unreachable."""
        lo, hi = nm.EXPOSURE_RANGE
        work = dict(params)

        def hp(e):
            work["exposure"] = e
            return self._high_pct(work, content)

        if hp(lo) >= target:
            return lo
        if hp(hi) <= target:
            return hi
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            if hp(mid) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _brighten(self, darker=False):
        """] / [: raise (or lower) paper black to lift (drop) midtones, then
        re-solve print exposure to hold the highlights at the level they were at
        before the move — the user's manual brighten move done in one keystroke.
        Holding the highlight level (P99.9) makes the result a function of black
        alone, so brighter-then-darker is exactly reversible. Stored as black +
        exposure print overrides, so X compares with default and C clears."""
        import auto_negadoctor as anp
        img_dict = self.images[self.current_idx]
        stem = img_dict["stem"]
        base = self._corrected_params(img_dict) or self._variant_params(img_dict)
        lin = self._neg_lin(img_dict)
        if not base or lin is None:
            self._set_info_text("Brighter/darker needs the source negative and "
                                "params (none available for this frame).")
            return
        cur_black = float(base["black"])
        cur_exp = float(base["exposure"])
        step = -BRIGHTEN_BLACK_STEP if darker else BRIGHTEN_BLACK_STEP
        new_black = nm.clamp(cur_black + step, nm.BLACK_RANGE)
        if new_black == cur_black:
            edge = "minimum" if darker else "maximum"
            self._set_info_text(f"Paper black already at its {edge} "
                                f"({cur_black:+.2f}) — can't go "
                                f"{'darker' if darker else 'brighter'}.")
            return
        content = self._brighten_content_pixels(img_dict, lin)
        # the highlight level to preserve, measured BEFORE the black change
        target = self._high_pct(base, content)
        work = dict(base)
        work["black"] = new_black
        new_exp = self._exposure_for_high_pct(work, content, target)

        ann = self.annotations[stem]
        ann["print_overrides"]["black"] = new_black
        if abs(new_exp - cur_exp) > 1e-9:
            ann["print_overrides"]["exposure"] = new_exp
        self._auto_save(stem)
        self._update_count_label()
        self._populate_items_list()
        self._redraw_markers()
        self._schedule_live_render()

        work["exposure"] = new_exp
        clip = self._clip_fraction(work, content)
        self._set_info_text(
            f"{'Darker' if darker else 'Brighter'}: paper black "
            f"{cur_black:+.3f} → {new_black:+.3f} ({step:+.0%}),  exposure "
            f"{cur_exp:.3f} → {new_exp:.3f}\n"
            f"  highlights held at P{anp.PRINT_HI_PCT:g}≈{target:.3f}, hard-clip "
            f"{clip * 100:.2f}% (budget {anp.PRINT_CLIP_BUDGET * 100:.1f}%).  "
            f"Press ] / [ again to go further; select blk/pexp + C to clear.")

    # --- crop edge dragging (modal tool via the press/drag/release hooks) ---

    CROP_EDGE_GRAB_PX = 10   # canvas px tolerance for grabbing a crop edge

    def _crop_edge_at(self, canvas_x, canvas_y):
        """Edge name of the effective crop rect near the cursor, or None."""
        img_dict = self.images[self.current_idx]
        rect = self._crop_rect(img_dict) or self._auto_content_rect(img_dict)
        x, y, cw, ch = rect
        x0, y0 = image_to_canvas(x, y, self.offset_x, self.offset_y, self.zoom)
        x1, y1 = image_to_canvas(x + cw, y + ch,
                                 self.offset_x, self.offset_y, self.zoom)
        tol = self.CROP_EDGE_GRAB_PX
        within_y = y0 - tol <= canvas_y <= y1 + tol
        within_x = x0 - tol <= canvas_x <= x1 + tol
        candidates = []
        if within_y:
            candidates += [("left", abs(canvas_x - x0)),
                           ("right", abs(canvas_x - x1))]
        if within_x:
            candidates += [("top", abs(canvas_y - y0)),
                           ("bottom", abs(canvas_y - y1))]
        candidates = [(e, d) for e, d in candidates if d <= tol]
        if not candidates:
            return None
        return min(candidates, key=lambda c: c[1])[0]

    def handle_press_override(self, event):
        """Grab a crop edge — works in ANY mode (no need to select crop
        first): the grab needs precise proximity to the rect edge, so it
        doesn't collide with patch clicks. Dragging moves just that edge;
        rubber-band elsewhere (with crop selected or mask view on) still
        draws a whole new rect."""
        edge = self._crop_edge_at(event.x, event.y)
        if edge is None:
            return False
        img_dict = self.images[self.current_idx]
        self._crop_drag_edge = edge
        self._crop_drag_rect = list(self._crop_rect(img_dict)
                                    or self._auto_content_rect(img_dict))
        self.selected_patch = CROP_NAME
        # the base class only routes B1-Motion to handle_drag_override when
        # drag_start is set, and it skips setting it for consumed presses
        self.drag_start = (event.x, event.y)
        return True

    def handle_drag_override(self, event):
        if not getattr(self, "_crop_drag_edge", None):
            return False
        img_dict = self.images[self.current_idx]
        stem = img_dict["stem"]
        ix, iy = canvas_to_image(event.x, event.y,
                                 self.offset_x, self.offset_y, self.zoom)
        x, y, cw, ch = self._crop_drag_rect
        x2, y2 = x + cw, y + ch
        min_sz = 40
        edge = self._crop_drag_edge
        if edge == "left":
            x = int(round(max(0, min(ix, x2 - min_sz))))
        elif edge == "right":
            x2 = int(round(min(img_dict["width"], max(ix, x + min_sz))))
        elif edge == "top":
            y = int(round(max(0, min(iy, y2 - min_sz))))
        else:
            y2 = int(round(min(img_dict["height"], max(iy, y + min_sz))))
        self._crop_drag_rect = [x, y, x2 - x, y2 - y]
        self.annotations[stem]["crop_correction"] = list(self._crop_drag_rect)
        self._redraw_markers()
        return True

    def handle_release_override(self, event):
        if not getattr(self, "_crop_drag_edge", None):
            return False
        self._crop_drag_edge = None
        self.drag_start = None
        self.is_dragging = False
        stem = self.images[self.current_idx]["stem"]
        self._auto_save(stem)
        self._update_count_label()
        self._populate_items_list()
        self._update_info_from_selection()
        self._schedule_live_render()
        return True

    def on_rubber_band(self, ix1, iy1, ix2, iy2, additive):
        """A rubber-band drag defines the true photo-content rectangle when
        "crop" is selected — or whenever the analysis-crop view (M) is
        active, where drawing the crop is the natural intent (a drag there
        used to be silently discarded unless crop was also selected)."""
        if self.selected_patch != CROP_NAME and self.mask_view == 0:
            return
        self.selected_patch = CROP_NAME
        img_dict = self.images[self.current_idx]
        stem = img_dict["stem"]
        x0 = max(0, int(round(ix1)))
        y0 = max(0, int(round(iy1)))
        x1 = min(img_dict["width"], int(round(ix2)))
        y1 = min(img_dict["height"], int(round(iy2)))
        if x1 - x0 < 40 or y1 - y0 < 40:
            return
        self.annotations[stem]["crop_correction"] = [x0, y0, x1 - x0, y1 - y0]
        self._auto_save(stem)
        self._redraw_markers()
        self._update_count_label()
        self._populate_items_list()
        self._update_info_from_selection()
        self._schedule_live_render()

    def _clear_correction(self):
        if self.selected_patch is None:
            return
        stem = self.images[self.current_idx]["stem"]
        ann = self.annotations[stem]
        if self.selected_patch == CROP_NAME:
            if ann.get("crop_correction"):
                ann["crop_correction"] = None
                self._auto_save(stem)
                self._redraw_markers()
                self._update_count_label()
                self._populate_items_list()
                self._schedule_live_render()
            self._update_info_from_selection()
            return
        if self.selected_patch in WB_NAMES:
            name = self.selected_patch
            if WB_NAME_OVR[name] in ann["wb_overrides"]:
                del ann["wb_overrides"][WB_NAME_OVR[name]]
                self._auto_save(stem)
                # snap the wheel marker back to the auto-found wb
                if getattr(self, "wheels", None):
                    wb = self._effective_wb(self.images[self.current_idx], name)
                    self.wheels[name].set_wb(wb)
                    self._update_wheel_readout(name, wb)
                self._update_count_label()
                self._populate_items_list()
                self._schedule_live_render()
            self._update_info_from_selection()
            return
        store = (ann["print_overrides"] if self.selected_patch in PRINT_PARAMS
                 else ann["patch_corrections"])
        if self.selected_patch in store:
            del store[self.selected_patch]
            self._auto_save(stem)
            self._redraw_markers()
            self._update_count_label()
            self._populate_items_list()
            self._schedule_live_render()
        self._update_info_from_selection()

    # ------------------------------------------------------------------
    # Info panel
    # ------------------------------------------------------------------

    def _fmt_rgb(self, rgb):
        if not rgb:
            return "n/a"
        return "(" + ", ".join(f"{v:.4f}" for v in rgb) + ")"

    def _update_info_from_selection(self):
        if self.selected_patch is None:
            self._set_info_text(self.default_info_text())
            self._sync_item_list_selection()
            return
        img_dict = self.images[self.current_idx]
        ann = self.annotations[img_dict["stem"]]
        patch = self.selected_patch

        if patch == CROP_NAME:
            auto_rect = self._auto_content_rect(img_dict)
            crop = ann.get("crop_correction")
            lines = ["Crop: true photo-content area (audits holder detection)",
                     f"  auto (border trim), x,y,w,h as frac of frame: "
                     f"{self._fmt_rect(auto_rect, img_dict)}"]
            if crop:
                lines.append(f"  CORRECTED: {self._fmt_rect(crop, img_dict)}")
                lines.append("  Re-render recomputes percentiles, D_max, "
                             "black/exposure and the print tune inside it.")
                lines.append("  Drag an EDGE to adjust it, drag elsewhere to "
                             "redefine, scroll grows/shrinks, C clears.")
            else:
                lines.append("  Drag a rectangle around the actual photo "
                             "content, or grab an edge of the")
                lines.append("  auto rect; everything outside is treated as "
                             "holder/rejected.")
            self._set_info_text("\n".join(lines))
            self._sync_item_list_selection()
            return

        if patch in PRINT_PARAMS:
            params = img_dict.get("params") or {}
            applied = params.get(patch)
            override = ann["print_overrides"].get(patch)
            rng = self._print_range(patch)
            lines = [f"Print param: {PRINT_LABEL[patch]}",
                     f"  applied: {applied:.4f}   range [{rng[0]}, {rng[1]}]"
                     if applied is not None else "  applied: n/a"]
            if override is not None:
                lines.append(f"  OVERRIDE: {override:.4f}  "
                             f"(delta {override - applied:+.4f})")
                lines.append("  Scroll adjusts (Shift=big step), C clears, "
                             "X compares with default.")
            else:
                lines.append("  Scroll to adjust with live preview "
                             "(Shift=big step).")
            self._set_info_text("\n".join(lines))
            self._sync_item_list_selection()
            return

        if patch in WB_NAMES:
            params = img_dict.get("params") or {}
            applied = params.get(WB_NAME_PARAM[patch])
            override = ann["wb_overrides"].get(WB_NAME_OVR[patch])
            lines = [f"Color wheel: {WB_NAME_LABEL[patch]}",
                     f"  auto (algorithm): {self._fmt_rgb(applied)}"]
            if override is not None:
                lines.append(f"  CHOSEN (wheel): {self._fmt_rgb(override)}")
                lines.append("  Drag the wheel to adjust (live preview), "
                             "C reverts to the auto value,")
                lines.append("  X compares with the default render.")
            else:
                lines.append("  Drag the wheel to set wb directly with live "
                             "preview (starts at the auto value).")
            self._set_info_text("\n".join(lines))
            self._sync_item_list_selection()
            return

        det = self._detected_rect(img_dict, patch)
        lines = [f"Patch: {patch.upper()}"]

        det_rgb = None
        if patch == "film_base":
            local = (img_dict.get("film_base") or {}).get("local")
            if local:
                det_rgb = local.get("rgb_linear")
        else:
            p = (img_dict.get("patches") or {}).get(patch) or {}
            det_rgb = p.get("rgb_neg_linear")
            if p.get("used_fallback"):
                lines.append("  WB FELL BACK to roll median (no usable patch found)")

        if det:
            lines.append(f"  detected rect (x,y,w,h frac): "
                         f"{self._fmt_rect(det, img_dict)}  neg RGB {self._fmt_rgb(det_rgb)}")
        else:
            lines.append("  not detected on this frame")

        corr = ann["patch_corrections"].get(patch)
        if corr:
            corr_rgb = self._neg_rgb_at(img_dict, corr)
            lines.append(f"  CORRECTED rect: {self._fmt_rect(corr, img_dict)}"
                         f"  neg RGB {self._fmt_rgb(corr_rgb)}")
            wb = self._wb_for_patch(img_dict, patch, corr_rgb)
            if wb:
                applied = (img_dict.get("params") or {}).get(
                    "wb_low" if patch == "shadows" else "wb_high")
                lines.append(f"  -> would give wb {self._fmt_rgb(wb)}"
                             f"  (applied: {self._fmt_rgb(applied)})")
            lines.append("  Scroll resizes, Ctrl+Click re-places, C clears.")
        else:
            lines.append("  Ctrl+Click at the correct position to mark this patch wrong;")
            lines.append("  scroll then resizes it (seeds from the detected rect).")
        self._set_info_text("\n".join(lines))
        self._sync_item_list_selection()

    # ------------------------------------------------------------------
    # Item table hooks
    # ------------------------------------------------------------------

    def configure_item_tags(self, tree):
        tree.tag_configure("det",      foreground="#cccccc")
        tree.tag_configure("corr",     foreground="#00ddcc")
        tree.tag_configure("fallback", foreground="#ff8888")

    def item_panel_header_text(self):
        ann = self.annotations[self.images[self.current_idx]["stem"]]
        return (f"{len(ann['patch_corrections'])}/3 patches, "
                f"{len(ann['wb_overrides'])}/2 wb, "
                f"{len(ann['print_overrides'])}/4 print")

    def item_rows(self):
        img_dict = self.images[self.current_idx]
        ann = self.annotations[img_dict["stem"]]
        patches = img_dict.get("patches") or {}
        params = img_dict.get("params") or {}

        rows = []
        for k, patch in enumerate(PATCHES):
            det = self._detected_rect(img_dict, patch)
            corr = ann["patch_corrections"].get(patch)
            has_note = bool(ann["patch_notes"].get(patch))
            fallback = (patch != "film_base"
                        and patches.get(patch, {}).get("used_fallback", False))
            det_str = self._fmt_pos(det, img_dict)
            corr_str = self._fmt_pos(corr, img_dict) if corr else ""
            tag = "corr" if corr else ("fallback" if fallback or not det else "det")
            rows.append({
                "iid": f"patch_{patch}",
                "values": (PATCH_SHORT[patch], det_str, "!" if fallback else "",
                           corr_str, "✎" if has_note else ""),
                "tag": tag,
                "name": patch,
                "sort": {"patch": k,
                         "det": det[0] if det else -1,
                         "fb": 1 if fallback else 0,
                         "corr": corr[0] if corr else -1,
                         "nt": 1 if has_note else 0},
            })
        for k, name in enumerate(PRINT_PARAMS):
            applied = params.get(name)
            override = ann["print_overrides"].get(name)
            has_note = bool(ann["patch_notes"].get(name))
            rows.append({
                "iid": f"print_{name}",
                "values": (PRINT_SHORT[name],
                           f"{applied:.3f}" if applied is not None else "—",
                           "",
                           f"{override:.3f}" if override is not None else "",
                           "✎" if has_note else ""),
                "tag": "corr" if override is not None else "det",
                "name": name,
                "sort": {"patch": len(PATCHES) + k,
                         "det": applied if applied is not None else -1,
                         "fb": 0,
                         "corr": override if override is not None else -1,
                         "nt": 1 if has_note else 0},
            })
        for k, name in enumerate(WB_NAMES):
            applied = params.get(WB_NAME_PARAM[name])
            override = ann["wb_overrides"].get(WB_NAME_OVR[name])
            has_note = bool(ann["patch_notes"].get(name))
            fmt = lambda v: f"{v[0]:.2f},{v[1]:.2f},{v[2]:.2f}"
            rows.append({
                "iid": name,
                "values": (WB_NAME_SHORT[name],
                           fmt(applied) if applied else "—", "",
                           fmt(override) if override else "",
                           "✎" if has_note else ""),
                "tag": "corr" if override else "det",
                "name": name,
                "sort": {"patch": len(PATCHES) + len(PRINT_PARAMS) + k,
                         "det": applied[0] if applied else -1,
                         "fb": 0,
                         "corr": override[0] if override else -1,
                         "nt": 1 if has_note else 0},
            })
        auto_rect = self._auto_content_rect(img_dict)
        crop = ann.get("crop_correction")
        has_note = bool(ann["patch_notes"].get(CROP_NAME))
        rows.append({
            "iid": "crop",
            "values": ("crop", self._fmt_rect(auto_rect, img_dict), "",
                       self._fmt_rect(crop, img_dict) if crop else "",
                       "✎" if has_note else ""),
            "tag": "corr" if crop else "det",
            "name": CROP_NAME,
            "sort": {"patch": len(PATCHES) + len(PRINT_PARAMS) + len(WB_NAMES),
                     "det": auto_rect[0], "fb": 0,
                     "corr": crop[0] if crop else -1,
                     "nt": 1 if has_note else 0},
        })
        return rows

    def is_row_currently_selected(self, row):
        return row["name"] == self.selected_patch

    def on_item_row_selected(self, row):
        self._select_patch(row["name"])

    def selected_row_iid(self):
        if self.selected_patch == CROP_NAME:
            return "crop"
        if self.selected_patch in WB_NAMES:
            return self.selected_patch
        if self.selected_patch in PRINT_PARAMS:
            return f"print_{self.selected_patch}"
        if self.selected_patch is not None:
            return f"patch_{self.selected_patch}"
        return None

    def selection_center(self):
        if self.selected_patch is None:
            return None
        img_dict = self.images[self.current_idx]
        if self.selected_patch == CROP_NAME:
            rect = self._crop_rect(img_dict) or self._auto_content_rect(img_dict)
        else:
            rect = self._effective_rect(img_dict, self.selected_patch)
        if not rect:
            return None
        return (rect[0] + rect[2] / 2, rect[1] + rect[3] / 2)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def report_title(self):
        return "AUTO NEGADOCTOR DEBUG REPORT"

    def report_constants_lines(self):
        constants = self.data.get("constants", {})
        lines = ["Detection constants at time of run:"]
        for key in sorted(constants):
            lines.append(f"  {key} = {constants[key]}")
        return lines

    def report_body_lines(self):
        lines = []
        corrected_by_patch = {p: 0 for p in PATCHES}
        overridden_by_param = {p: 0 for p in PRINT_PARAMS}
        wb_override_count = {n: 0 for n in WB_NAMES}
        crop_count = 0
        bad_count = 0
        fallback_count = 0
        images_with_corrections = 0

        for img_dict in self.images:
            stem = img_dict["stem"]
            ann = self.annotations[stem]
            p = img_dict.get("params") or {}
            fb = img_dict.get("film_base") or {}
            patches = img_dict.get("patches") or {}
            corrections = ann["patch_corrections"]
            overrides = ann["print_overrides"]
            wb_overrides = ann["wb_overrides"]
            notes = ann["patch_notes"]

            lines.append("=" * 48)
            head = f"IMAGE: {stem}  ({img_dict['width']} x {img_dict['height']} px)"
            if fb.get("is_global_winner"):
                head += "  [GLOBAL BASE WINNER]"
            lines.append(head)
            if p:
                lines.append(f"  Dmin=({p['Dmin'][0]:.4f},{p['Dmin'][1]:.4f},{p['Dmin'][2]:.4f})"
                             f"  D_max={p['D_max']:.3f}  offset={p['offset']:.3f}")
                lines.append(f"  wb_low=({p['wb_low'][0]:.3f},{p['wb_low'][1]:.3f},{p['wb_low'][2]:.3f})"
                             f"  wb_high=({p['wb_high'][0]:.3f},{p['wb_high'][1]:.3f},{p['wb_high'][2]:.3f})")
                lines.append(f"  black={p['black']:.4f}  gamma={p['gamma']:.2f}"
                             f"  soft_clip={p['soft_clip']:.2f}  exposure={p['exposure']:.4f}")
            for patch in ("shadows", "highlights"):
                if patches.get(patch, {}).get("used_fallback"):
                    fallback_count += 1
                    lines.append(f"  NOTE: {patch} wb fell back to roll median")

            if ann["bad_inversion"]:
                bad_count += 1
                note = f" — {ann['bad_inversion_note']}" if ann["bad_inversion_note"] else ""
                lines.append(f"  ** BAD INVERSION flagged by user{note}")

            crop = ann.get("crop_correction")
            if (not corrections and not overrides and not wb_overrides
                    and not crop and not notes and not ann["bad_inversion"]):
                lines.append("  No corrections — accepted.")
                lines.append("")
                continue

            if corrections or overrides or wb_overrides or crop:
                images_with_corrections += 1

            if crop:
                crop_count += 1
                auto_rect = self._auto_content_rect(img_dict)
                lines.append("")
                lines.append(f"  CROP CORRECTION (true content area, x,y,w,h "
                             f"as frac of frame): {self._fmt_rect(crop, img_dict)}")
                lines.append(f"    auto border-trim rect was: "
                             f"{self._fmt_rect(auto_rect, img_dict)}")

            if corrections:
                lines.append("")
                lines.append("  CORRECTED PATCHES (detected was wrong):")
                for patch in PATCHES:
                    if patch not in corrections:
                        continue
                    corrected_by_patch[patch] += 1
                    det = self._detected_rect(img_dict, patch)
                    corr = corrections[patch]
                    corr_rgb = self._neg_rgb_at(img_dict, corr)
                    line = (f"    {patch:<10} detected="
                            f"{self._fmt_rect(det, img_dict) if det else 'none'}"
                            f"  ->  corrected={self._fmt_rect(corr, img_dict)} (x,y,w,h frac)")
                    if corr_rgb:
                        line += f"  neg RGB ({', '.join(f'{v:.4f}' for v in corr_rgb)})"
                    lines.append(line)
                    if det and (corr[2], corr[3]) != (det[2], det[3]):
                        W, H = img_dict["width"], img_dict["height"]
                        lines.append(f"      size changed {det[2]/W:.3f}x{det[3]/H:.3f}"
                                     f" -> {corr[2]/W:.3f}x{corr[3]/H:.3f} (frac)")
                    wb = self._wb_for_patch(img_dict, patch, corr_rgb)
                    if wb:
                        applied = p.get("wb_low" if patch == "shadows" else "wb_high")
                        lines.append(f"      -> wb from corrected patch: "
                                     f"({', '.join(f'{v:.3f}' for v in wb)})"
                                     f"  vs applied ({', '.join(f'{v:.3f}' for v in applied)})")

            if overrides:
                lines.append("")
                lines.append("  PRINT PARAM OVERRIDES (applied -> corrected):")
                for name in PRINT_PARAMS:
                    if name not in overrides:
                        continue
                    overridden_by_param[name] += 1
                    applied = p.get(name)
                    val = overrides[name]
                    if applied is not None:
                        lines.append(f"    {PRINT_LABEL[name]:<38} "
                                     f"{applied:.4f} -> {val:.4f} "
                                     f"({val - applied:+.4f})")
                    else:
                        lines.append(f"    {PRINT_LABEL[name]:<38} -> {val:.4f}")

            if wb_overrides:
                lines.append("")
                lines.append("  WB WHEEL OVERRIDES (auto -> chosen) — tuning "
                             "ground truth:")
                for name in WB_NAMES:
                    key = WB_NAME_OVR[name]
                    if key not in wb_overrides:
                        continue
                    wb_override_count[name] += 1
                    chosen = wb_overrides[key]
                    applied = p.get(WB_NAME_PARAM[name])
                    line = (f"    {key:<10} chosen "
                            f"({', '.join(f'{v:.3f}' for v in chosen)})")
                    if applied:
                        line += (f"  vs auto "
                                 f"({', '.join(f'{v:.3f}' for v in applied)})")
                    lines.append(line)

            if notes:
                lines.append("")
                lines.append("  USER NOTES:")
                for key in PATCHES + PRINT_PARAMS + WB_NAMES + (CROP_NAME,):
                    if key in notes:
                        lines.append(f"    {key:<10} {notes[key]}")
            lines.append("")

        lines.append("=" * 48)
        lines.append("SUMMARY")
        lines.append(f"  Total corrected patches: {sum(corrected_by_patch.values())}")
        for patch in PATCHES:
            lines.append(f"    {patch:<10} {corrected_by_patch[patch]}")
        lines.append(f"  Total print overrides: {sum(overridden_by_param.values())}")
        for name in PRINT_PARAMS:
            lines.append(f"    {name:<10} {overridden_by_param[name]}")
        lines.append(f"  Total wb wheel overrides: {sum(wb_override_count.values())}")
        for name in WB_NAMES:
            lines.append(f"    {WB_NAME_OVR[name]:<10} {wb_override_count[name]}")
        lines.append(f"  Crop corrections: {crop_count}")
        lines.append(f"  Bad inversions flagged: {bad_count} / {len(self.images)}")
        lines.append(f"  wb fallbacks (patch missing): {fallback_count}")
        lines.append(f"  Images with corrections: {images_with_corrections} / "
                     f"{len(self.images)}")
        lines.append("")
        return lines


# Alias for generic smoke drivers
DebugUI = NegadoctorDebugUI


def main():
    # `--apply` (passed by auto_negadoctor.py --annotate-apply) makes the UI
    # write applied_results.txt on close. run_main reads argv[1] as the session
    # dir, so the flag elsewhere in argv is harmless once we've consumed it.
    if "--apply" in sys.argv:
        NegadoctorDebugUI.apply_mode = True
        sys.argv = [a for a in sys.argv if a != "--apply"]
    NegadoctorDebugUI.run_main(usage="Usage: debug_ui.py <session_dir> [--apply]")


if __name__ == "__main__":
    main()
