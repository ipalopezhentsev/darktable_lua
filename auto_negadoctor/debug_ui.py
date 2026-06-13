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

Reads:  {session_dir}/{stem}_debug_nega.json   (per-frame session data)
        {session_dir}/{stem}_inverted.jpg      (rendered inverted preview)
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
from pathlib import Path

import tkinter as tk

import numpy as np
from PIL import Image

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


class NegadoctorDebugUI(DebugUIBase):
    WINDOW_TITLE = "Auto Negadoctor Debug UI"
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
            "crop_correction": None,   # [x, y, w, h] true photo-content rect
            "patch_notes": {},         # {patch / print param / "crop": str}
            "bad_inversion": False,
            "bad_inversion_note": "",
        }

    def serialize_annotations(self, stem):
        ann = self.annotations[stem]
        img_dict = next((d for d in self.images if d["stem"] == stem), None)
        params = (img_dict.get("params") or {}) if img_dict else {}
        out_corr = {}
        for patch, rect in sorted(ann["patch_corrections"].items()):
            out_corr[patch] = {
                "detected": self._detected_rect(img_dict, patch) if img_dict else None,
                "corrected": [int(v) for v in rect],
            }
        out_over = {}
        for name, val in sorted(ann["print_overrides"].items()):
            out_over[name] = {
                "applied": params.get(name),
                "corrected": float(val),
            }
        out = {
            "stem": stem,
            "patch_corrections": out_corr,
            "print_overrides": out_over,
            "patch_notes": {p: n for p, n in sorted(ann["patch_notes"].items())},
            "bad_inversion": bool(ann["bad_inversion"]),
            "bad_inversion_note": ann["bad_inversion_note"],
        }
        if ann.get("crop_correction"):
            out["crop_correction"] = {
                "auto_border": list(img_dict.get("border") or []) if img_dict else None,
                "corrected": [int(v) for v in ann["crop_correction"]],
            }
        return out

    def deserialize_annotations(self, img_dict, data):
        stem = img_dict["stem"]
        ann = self.annotations[stem]
        for patch, entry in data.get("patch_corrections", {}).items():
            if patch in PATCHES:
                try:
                    ann["patch_corrections"][patch] = [int(v) for v in entry["corrected"]]
                except (KeyError, ValueError, TypeError):
                    pass
        for name, entry in data.get("print_overrides", {}).items():
            if name in PRINT_PARAMS:
                try:
                    ann["print_overrides"][name] = float(entry["corrected"])
                except (KeyError, ValueError, TypeError):
                    pass
        crop = data.get("crop_correction")
        if crop:
            try:
                ann["crop_correction"] = [int(v) for v in crop["corrected"]]
            except (KeyError, ValueError, TypeError):
                pass
        ann["patch_notes"] = {p: str(n) for p, n in data.get("patch_notes", {}).items()
                              if (p in PATCHES or p in PRINT_PARAMS
                                  or p == CROP_NAME) and n}
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
        correction applied, so patch RGB sampling and live re-renders stay
        consistent with the pipeline analysis."""
        if getattr(self, "_neg_cache_stem", None) == img_dict["stem"]:
            return self._neg_cache
        path = img_dict.get("negative_path")
        lin = None
        if path and Path(path).exists():
            import auto_negadoctor
            try:
                lin = auto_negadoctor.load_frame(
                    path, img_dict.get("vignette"))[1]
            except Exception:
                lin = None
        self._neg_cache_stem = img_dict["stem"]
        self._neg_cache = lin
        return lin

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
        crop = ann.get("crop_correction")
        p = img_dict.get("params")
        if (not corr and not overrides and not crop) or not p:
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
        if (crop or "film_base" in corr) and picked_min:
            d_max = nm.compute_dmax(dmin, picked_min)
            out["D_max"] = d_max

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
        if params is None:
            if self._live_rendered:
                self._live_rendered = False
                try:
                    self._set_display_image(Image.open(img_dict["image_path"]))
                except Exception:
                    return
            return
        lin = self._neg_lin(img_dict)
        if lin is None:
            return
        out = nm.render_negadoctor(lin, params)
        arr = (nm.linear_to_srgb(out) * 255.0 + 0.5).astype(np.uint8)
        self._live_rendered = True
        self._set_display_image(Image.fromarray(arr))

    # ------------------------------------------------------------------
    # Selection state
    # ------------------------------------------------------------------

    def init_selection_state(self):
        # selected_patch holds a patch name OR a print-param name
        self.selected_patch = None
        self.view_negative = False
        self.compare_default = False
        self.mask_view = 0           # 0=normal, 1=tint areas, 2=hide rejected
        self.show_histogram = True
        self._hist_data = None
        self._display_base_pil = None
        self._amask_cache_stem = None
        self._amask_cache = None
        self._neg_cache_stem = None
        self._neg_cache = None
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
        self._update_count_label()
        self._update_bad_btn()
        # capture the freshly loaded image as the undecorated display base
        # and re-apply the persistent mask view to the new frame
        self._display_base_pil = self.pil_image
        if self.mask_view:
            self.pil_image = self._decorate(self._display_base_pil)
            self._redraw()
        self._refresh_histogram()
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
            "  C — clear correction\n"
            "  V — inverted/negative view\n"
            "  X — corrected/default render\n"
            "  M — analysis areas / hide\n"
            "      rejected holder\n"
            "  T — histogram on/off\n"
            "  G — toggle bad inversion\n"
            "  Note box — comment on sel.\n"
            "  H — hide/show markers\n"
            "  +/- zoom, Arrows pan, F fit\n"
            "Corrections re-render the\n"
            "inversion live."
        ))

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

        self.mask_btn = tk.Button(btn_frame, text=MASK_VIEW_LABELS[0],
                                  command=self._cycle_mask_view, **btn_cfg)
        self.mask_btn.pack(fill=tk.X, pady=1)

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
        self.bind_key("<m>", lambda e: self._cycle_mask_view())
        self.bind_key("<M>", lambda e: self._cycle_mask_view())
        self.bind_key("<t>", lambda e: self._toggle_histogram())
        self.bind_key("<T>", lambda e: self._toggle_histogram())

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
                         f"exp={p['exposure']:.3f}")
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
        crop = ", crop✓" if ann.get("crop_correction") else ""
        self.count_label.config(
            text=f"Corrected: {n}/3 patches, {m}/4 print{crop}")

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
            self._set_display_image(Image.open(img_dict["image_path"]))
            self._schedule_live_render(delay_ms=10)
        self._update_view_btn()

    def _update_view_btn(self):
        label = "Show Inverted  (V)" if self.view_negative else "Show Negative  (V)"
        self.view_btn.config(text=label,
                             fg="#ffff88" if self.view_negative else "white")

    def _has_corrections(self):
        ann = self.annotations[self.images[self.current_idx]["stem"]]
        return bool(ann["patch_corrections"] or ann["print_overrides"]
                    or ann.get("crop_correction"))

    # ------------------------------------------------------------------
    # Crop correction helpers
    # ------------------------------------------------------------------

    def _auto_content_rect(self, img_dict):
        """Content rect implied by the automatic border trim."""
        l, t, r, b = img_dict.get("border") or (0, 0, 0, 0)
        return [l, t, img_dict["width"] - l - r, img_dict["height"] - t - b]

    def _crop_rect(self, img_dict):
        return self.annotations[img_dict["stem"]].get("crop_correction")

    # ------------------------------------------------------------------
    # Analysis-area mask view (audit holder/border detection)
    # ------------------------------------------------------------------

    def _analysis_mask(self, img_dict):
        """Per-pixel analysis mask (uint8 codes) for the frame, or None."""
        if self._amask_cache_stem == img_dict["stem"]:
            return self._amask_cache
        path = img_dict.get("analysis_mask_path")
        mask = None
        if path and Path(path).exists():
            try:
                mask = np.asarray(Image.open(path))
            except Exception:
                mask = None
        self._amask_cache_stem = img_dict["stem"]
        self._amask_cache = mask
        return mask

    def _decorate(self, pil):
        """Apply the current mask view to a base image."""
        if self.mask_view == 0 or pil is None:
            return pil
        mask = self._analysis_mask(self.images[self.current_idx])
        arr = np.array(pil.convert("RGB"))
        if mask is not None and mask.shape[:2] != arr.shape[:2]:
            mask = None
        if self.mask_view == 1 and mask is None:
            return pil
        if self.mask_view == 1:
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
            if crop:
                x, y, cw, ch = [int(v) for v in crop]
                keep = np.zeros(arr.shape[:2], dtype=bool)
                keep[y:y + ch, x:x + cw] = True
                arr[~keep] = 0
            elif mask is not None:
                arr[(mask == 1) | (mask == 2)] = 0
            else:
                return pil
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
        if self.mask_view == 2:
            self.canvas.create_text(x0 + 3, y0 + 2, anchor="nw",
                                    text="content only", fill="#aaaaaa",
                                    font=("Courier", 7), tags="histogram")

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

    def _toggle_histogram(self):
        self.show_histogram = not self.show_histogram
        self._redraw_markers()

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
        return ("patches", "corrections", "labels", "badges", "histogram")

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
                     f"  auto (border trim): {auto_rect}"]
            if crop:
                lines.append(f"  CORRECTED: {crop}")
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
            lines.append(f"  detected rect: {det}  neg RGB {self._fmt_rgb(det_rgb)}")
        else:
            lines.append("  not detected on this frame")

        corr = ann["patch_corrections"].get(patch)
        if corr:
            corr_rgb = self._neg_rgb_at(img_dict, corr)
            lines.append(f"  CORRECTED rect: {corr}  neg RGB {self._fmt_rgb(corr_rgb)}")
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
                f"{len(ann['print_overrides'])}/4 print corrected")

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
            det_str = f"{det[0]},{det[1]}" if det else "—"
            corr_str = f"{corr[0]},{corr[1]}" if corr else ""
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
        auto_rect = self._auto_content_rect(img_dict)
        crop = ann.get("crop_correction")
        has_note = bool(ann["patch_notes"].get(CROP_NAME))
        rows.append({
            "iid": "crop",
            "values": ("crop", f"{auto_rect[0]},{auto_rect[1]}", "",
                       f"{crop[0]},{crop[1]}" if crop else "",
                       "✎" if has_note else ""),
            "tag": "corr" if crop else "det",
            "name": CROP_NAME,
            "sort": {"patch": len(PATCHES) + len(PRINT_PARAMS),
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
            if (not corrections and not overrides and not crop and not notes
                    and not ann["bad_inversion"]):
                lines.append("  No corrections — accepted.")
                lines.append("")
                continue

            if corrections or overrides or crop:
                images_with_corrections += 1

            if crop:
                crop_count += 1
                l, t, r, b = img_dict.get("border") or (0, 0, 0, 0)
                auto_rect = [l, t, img_dict["width"] - l - r,
                             img_dict["height"] - t - b]
                lines.append("")
                lines.append(f"  CROP CORRECTION (true content area): {crop}")
                lines.append(f"    auto border-trim rect was: {auto_rect}")

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
                    line = (f"    {patch:<10} detected={det if det else 'none'}"
                            f"  ->  corrected={corr}")
                    if corr_rgb:
                        line += f"  neg RGB ({', '.join(f'{v:.4f}' for v in corr_rgb)})"
                    lines.append(line)
                    if det and (corr[2], corr[3]) != (det[2], det[3]):
                        lines.append(f"      size changed {det[2]}x{det[3]}"
                                     f" -> {corr[2]}x{corr[3]}")
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

            if notes:
                lines.append("")
                lines.append("  USER NOTES:")
                for key in PATCHES + PRINT_PARAMS + (CROP_NAME,):
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
    NegadoctorDebugUI.run_main(usage="Usage: debug_ui.py <session_dir>")


if __name__ == "__main__":
    main()
