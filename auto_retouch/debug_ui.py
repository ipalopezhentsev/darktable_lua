"""
Debug UI for dust detection annotation.

Lets the user review detected spots and rejected candidates on each exported
image, mark false positives, and add missed dust points. Produces a
debug_report.txt readable by Claude Code for algorithm tuning.

Built on the shared viewer base in common/debug_ui_base.py; this file holds
everything dust-specific (marker rendering, hit-testing, annotation schema,
thread-draw mode, report content).

Usage:
    python debug_ui.py <export_dir>

Reads:  {export_dir}/{stem}_debug_spots.json  (per-image detection data)
Writes: {export_dir}/{stem}_annotations.json   (auto-saved per image)
        {export_dir}/debug_report.txt           (on window close)
"""

import sys
import math
from pathlib import Path

import tkinter as tk

sys.path.insert(0, str(Path(__file__).parent.parent))   # repo root -> common
sys.path.insert(0, str(Path(__file__).parent))           # feature dir -> detect_dust

from common.debug_ui_base import (
    DebugUIBase, canvas_to_image, image_to_canvas, _point_to_path_dist)


# Default full width (px) assigned to a hand-drawn missed thread.
DEFAULT_MISSED_STROKE_WIDTH = 8.0


class DustDebugUI(DebugUIBase):
    WINDOW_TITLE = "Dust Detection Debug UI"
    EMPTY_SESSION_MESSAGE = "No *_debug_spots.json files found in:"
    ITEM_PANEL_TITLE = "Spots:"
    CENTER_BUTTON_TEXT = "Center on spot"
    ITEM_COLS    = ("idx", "cx", "cy", "r", "ctr", "tex", "fp")
    ITEM_HEADERS = {"idx": "#",   "cx": "cx",  "cy": "cy",
                    "r":   "r",   "ctr": "ctr", "tex": "tex", "fp": "FP"}
    ITEM_WIDTHS  = {"idx": 30, "cx": 46, "cy": 46,
                    "r":  30, "ctr": 40, "tex": 36, "fp": 24}
    ITEM_ANCHORS = {"fp": "center"}

    # ------------------------------------------------------------------
    # Session / annotation lifecycle
    # ------------------------------------------------------------------

    def load_session(self, session_dir):
        import detect_dust
        images, constants = detect_dust.load_debug_spots_dir(session_dir)
        # Sensor-dust sessions reuse this UI wholesale (sensor spots are dot
        # spots in the same dict format); only title and report wording differ.
        self.sensor_mode = bool(images and images[0].get("mode") == "sensor")
        if self.sensor_mode:
            self.root.title("Sensor Dust Debug UI")
        return images, constants

    def new_annotation_state(self, img_dict):
        return {
            "false_positives": set(),
            "missed_dust": [],
            "source_overrides": {},   # {spot_idx: (src_cx, src_cy)}
            "radius_overrides": {},   # {spot_idx: corrected_radius_px}
            "path_overrides": {},     # {spot_idx: [[x,y],...]} edited stroke centerlines
            "missed_strokes": [],     # [{"path":[[x,y],...], "stroke_width_px":w}] hand-drawn threads
            "source_mismatches": [],  # list from quality test diff sessions
            "radius_mismatches": [],  # list from quality test diff sessions
        }

    def serialize_annotations(self, stem):
        ann = self.annotations[stem]
        img_dict = next((d for d in self.images if d["stem"] == stem), None)
        detected = img_dict.get("detected") or [] if img_dict else []

        fp_list = []
        for idx in sorted(ann["false_positives"]):
            if idx < len(detected):
                fp_list.append(detected[idx])

        # Serialize source_overrides as list of {spot_idx, src_cx, src_cy}
        src_overrides_list = [
            {"spot_idx": k, "src_cx": float(v[0]), "src_cy": float(v[1])}
            for k, v in sorted(ann.get("source_overrides", {}).items())
        ]

        # Serialize radius_overrides as list of {spot_idx, corrected_radius_px}
        radius_overrides_list = [
            {"spot_idx": k, "corrected_radius_px": float(v)}
            for k, v in sorted(ann.get("radius_overrides", {}).items())
        ]

        # Serialize path_overrides (edited stroke centerlines) as list of {spot_idx, path}
        path_overrides_list = [
            {"spot_idx": k, "path": [[float(p[0]), float(p[1])] for p in v]}
            for k, v in sorted(ann.get("path_overrides", {}).items())
        ]

        return {
            "stem": stem,
            "false_positives": fp_list,
            "missed_dust": ann["missed_dust"],
            "missed_strokes": ann.get("missed_strokes", []),
            "source_overrides": src_overrides_list,
            "radius_overrides": radius_overrides_list,
            "path_overrides": path_overrides_list,
            "source_mismatches": ann.get("source_mismatches", []),
            "radius_mismatches": ann.get("radius_mismatches", []),
        }

    def deserialize_annotations(self, img_dict, data):
        stem = img_dict["stem"]
        detected = img_dict.get("detected") or []
        # Rebuild false_positives as set of indices by matching cx/cy
        fp_indices = set()
        for fp in data.get("false_positives", []):
            for i, spot in enumerate(detected):
                if abs(spot["cx"] - fp["cx"]) < 0.5 and abs(spot["cy"] - fp["cy"]) < 0.5:
                    fp_indices.add(i)
                    break
        self.annotations[stem]["false_positives"] = fp_indices
        self.annotations[stem]["missed_dust"] = data.get("missed_dust", [])
        self.annotations[stem]["missed_strokes"] = data.get("missed_strokes", [])
        # Restore source overrides: list of {spot_idx, src_cx, src_cy}
        src_overrides = {}
        for entry in data.get("source_overrides", []):
            try:
                src_overrides[int(entry["spot_idx"])] = (
                    float(entry["src_cx"]), float(entry["src_cy"]))
            except (KeyError, ValueError, TypeError):
                pass
        self.annotations[stem]["source_overrides"] = src_overrides
        # Restore radius overrides: list of {spot_idx, corrected_radius_px}
        radius_overrides = {}
        for entry in data.get("radius_overrides", []):
            try:
                radius_overrides[int(entry["spot_idx"])] = float(entry["corrected_radius_px"])
            except (KeyError, ValueError, TypeError):
                pass
        self.annotations[stem]["radius_overrides"] = radius_overrides
        # Restore stroke path overrides: list of {spot_idx, path}
        path_overrides = {}
        for entry in data.get("path_overrides", []):
            try:
                path_overrides[int(entry["spot_idx"])] = [
                    [float(p[0]), float(p[1])] for p in entry["path"]]
            except (KeyError, ValueError, TypeError):
                pass
        self.annotations[stem]["path_overrides"] = path_overrides
        # Load source/radius mismatches from quality test diff sessions
        self.annotations[stem]["source_mismatches"] = data.get("source_mismatches", [])
        self.annotations[stem]["radius_mismatches"] = data.get("radius_mismatches", [])

    # ------------------------------------------------------------------
    # Selection state
    # ------------------------------------------------------------------

    def init_selection_state(self):
        self.selected_detected = set()   # set of int (detected spot indices)
        self.selected_rejected = set()   # set of int (rejected candidate indices)
        self.selected_missed = set()     # set of int (indices into missed_dust list)
        self.selected_source = None      # int or None: spot index whose source is selected
        self.selected_keypoint = None    # (spot_idx, kp_idx) or None: selected stroke key point
        self.selected_missed_stroke = None  # int or None: index into missed_strokes

        # Thread-draw mode: click to place centerline points for a missed thread
        self.thread_draw_mode = False
        self.thread_draw_points = []     # list of [x,y] image coords (in-progress)

    def reset_selection(self):
        self.selected_detected = set()
        self.selected_rejected = set()
        self.selected_missed = set()
        self.selected_source = None
        self.selected_keypoint = None
        self.selected_missed_stroke = None
        self.remove_missed_btn.config(state=tk.DISABLED)

    def reset_for_new_image(self):
        self.thread_draw_mode = False
        self.thread_draw_points = []
        self.reset_selection()

    # ------------------------------------------------------------------
    # Layout / text hooks
    # ------------------------------------------------------------------

    def build_left_controls(self, parent):
        # Show rejected toggle
        self.show_rejected_var = tk.BooleanVar(value=False)
        tk.Checkbutton(parent, text="Show rejected candidates",
                       variable=self.show_rejected_var,
                       command=self._redraw_markers,
                       bg="#484848", fg="white", selectcolor="#363636",
                       activebackground="#484848", activeforeground="white"
                       ).pack(anchor="w", padx=6, pady=2)

        # Show source brush circle toggle
        self.show_source_brush_var = tk.BooleanVar(value=False)
        tk.Checkbutton(parent, text="Show source brush",
                       variable=self.show_source_brush_var,
                       command=self._redraw_markers,
                       bg="#484848", fg="white", selectcolor="#363636",
                       activebackground="#484848", activeforeground="white"
                       ).pack(anchor="w", padx=6, pady=2)

    def build_left_info(self, parent):
        self.add_legend(parent, [
            ("●", "#00cc44", "Detected spot"),
            ("●", "#ff44cc", "Radius mismatch vs baseline"),
            ("✕", "#ff3333", "False positive"),
            ("●", "#ff8800", "Rejected candidate"),
            ("✚", "#00ffff", "Missed dust (added)"),
            ("╱", "#00ffff", "Missed thread (drawn)"),
            ("□", "#00cc44", "Heal source (- - line)"),
            ("○", "#00cc44", "Source brush (dashed)"),
            ("□", "#ff4444", "Baseline source (mismatch)"),
            ("○", "#ff9900", "Baseline radius (dashed, zoom in)"),
            ("○", "#00ddff", "Corrected radius (annotated)"),
        ])
        self.add_hints(parent, (
            "Mouse:\n"
            "  Scroll — zoom in/out\n"
            "  Scroll (1 spot sel'd) —\n"
            "    adjust radius correction\n"
            "  Middle drag — pan\n"
            "  Click — select marker\n"
            "  Drag — multi-select\n"
            "  Shift+click/drag — add to sel\n"
            "  Ctrl+click — add missed\n"
            "  Ctrl+drag — zoom to rect\n"
            "Keys:\n"
            "  Space/B — next/prev image\n"
            "  R — rejected → missed\n"
            "  T — draw missed thread\n"
            "    (click pts, Enter=done,\n"
            "     Esc=cancel)\n"
            "  M — mark false positive\n"
            "  C — clear FP mark\n"
            "  Del — remove missed marker\n"
            "  H — hide/show all markers\n"
            "  +/- — zoom ×2 / ÷2\n"
            "  Arrows — pan  (Shift=fast)\n"
            "  F — fit to window"
        ))

    def build_feature_buttons(self, btn_frame, btn_cfg):
        self.count_label = tk.Label(btn_frame, text="FP: 0 | Missed: 0",
                                    bg="#484848", fg="#c0c0c0", font=("", 9))
        self.count_label.pack(anchor="e", pady=(0, 4))

        tk.Button(btn_frame, text="Rejected → Missed  (R)",
                  command=self._mark_rejected_as_missed, **btn_cfg).pack(fill=tk.X, pady=1)

        self.mark_fp_btn = tk.Button(btn_frame, text="Mark False Positive  (M)",
                                     command=self._mark_fp, **btn_cfg)
        self.mark_fp_btn.pack(fill=tk.X, pady=1)

        self.clear_fp_btn = tk.Button(btn_frame, text="Clear FP Mark  (C)",
                                      command=self._clear_fp, **btn_cfg)
        self.clear_fp_btn.pack(fill=tk.X, pady=1)

        self.remove_missed_btn = tk.Button(btn_frame, text="Remove Missed  (Del)",
                                           command=self._remove_missed,
                                           state=tk.DISABLED, **btn_cfg)
        self.remove_missed_btn.pack(fill=tk.X, pady=1)

    def bind_feature_keys(self):
        self.root.bind("<m>", lambda e: self._mark_fp())
        self.root.bind("<M>", lambda e: self._mark_fp())
        self.root.bind("<c>", lambda e: self._clear_fp())
        self.root.bind("<C>", lambda e: self._clear_fp())
        self.root.bind("<Delete>", lambda e: self._remove_missed())
        self.root.bind("<BackSpace>", lambda e: self._remove_missed())
        self.root.bind("<r>", lambda e: self._mark_rejected_as_missed())
        self.root.bind("<R>", lambda e: self._mark_rejected_as_missed())
        self.root.bind("<t>", lambda e: self._toggle_thread_draw())
        self.root.bind("<T>", lambda e: self._toggle_thread_draw())
        self.root.bind("<Return>", lambda e: self._finish_thread())
        self.root.bind("<Escape>", lambda e: self._cancel_thread())

    def image_status_text(self, img_dict):
        detected = img_dict.get("detected") or []
        rejected_list = img_dict.get("rejected") or []
        return (f"{img_dict['width']} × {img_dict['height']} px\n"
                f"{len(detected)} detected\n{len(rejected_list)} rejected candidates")

    def default_info_text(self):
        return ("No marker selected.\n"
                "Click a marker or Ctrl+Click to add missed dust.")

    def update_counts(self):
        self._update_count_label()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def overlay_tags(self):
        return ("detected", "rejected", "fp", "missed", "source", "sel")

    def _get_source(self, stem, spot_idx, spot):
        """Return (src_cx, src_cy) for a spot — user override first, then auto-detected."""
        overrides = self.annotations[stem].get("source_overrides", {})
        if spot_idx in overrides:
            return overrides[spot_idx]
        src_cx = spot.get("src_cx")
        src_cy = spot.get("src_cy")
        if src_cx is not None and src_cy is not None:
            return (src_cx, src_cy)
        return None

    def _get_path(self, stem, spot_idx, spot):
        """Return the stroke centerline key points (list of [x,y]) for a stroke spot —
        user path override first, then the detected path. None for non-stroke spots."""
        if spot.get("kind") != "stroke":
            return None
        overrides = self.annotations[stem].get("path_overrides", {})
        if spot_idx in overrides:
            return overrides[spot_idx]
        return spot.get("path")

    def _draw_stroke_marker(self, stem, ann, i, spot):
        """Render a stroke spot: brush-width coverage band, bright centerline through the
        key points (with draggable handles), and the translated healing-source polyline."""
        path = self._get_path(stem, i, spot)
        if not path or len(path) < 1:
            return
        is_fp = i in ann["false_positives"]
        is_sel = i in self.selected_detected
        is_src_sel = (self.selected_source == i)
        sel_kp = getattr(self, "selected_keypoint", None)

        pts_canvas = [image_to_canvas(px, py, self.offset_x, self.offset_y, self.zoom)
                      for px, py in path]
        flat = [c for pt in pts_canvas for c in pt]

        color = "#ffff00" if is_sel else "#00cc44"
        # Brush coverage band (dim, thick) = what darktable will heal.
        if len(pts_canvas) >= 2:
            band_w = max(2, int(round(spot.get("brush_radius_px", 4) * 2 * self.zoom)))
            self.canvas.create_line(*flat, fill="#225522", width=band_w,
                                    capstyle="round", joinstyle="round", tags="detected")
            # Bright centerline.
            self.canvas.create_line(*flat, fill=color, width=2, tags="detected")
        # Key-point handles.
        for k, (hx, hy) in enumerate(pts_canvas):
            hr = 5 if (sel_kp == (i, k)) else 4
            hcolor = "#ff00ff" if (sel_kp == (i, k)) else color
            self.canvas.create_rectangle(hx - hr, hy - hr, hx + hr, hy + hr,
                                         outline=hcolor, width=2, tags="detected")
        # FP X mark at the midpoint.
        if is_fp:
            mx, my = pts_canvas[len(pts_canvas) // 2]
            r2 = 8
            self.canvas.create_line(mx - r2, my - r2, mx + r2, my + r2,
                                    fill="#ff3333", width=2, tags="fp")
            self.canvas.create_line(mx + r2, my - r2, mx - r2, my + r2,
                                    fill="#ff3333", width=2, tags="fp")
        # Source: translate the whole path by (src - path[0]).
        src = self._get_source(stem, i, spot)
        if src is not None and len(path) >= 1:
            dx = src[0] - path[0][0]
            dy = src[1] - path[0][1]
            src_canvas = [image_to_canvas(px + dx, py + dy,
                                          self.offset_x, self.offset_y, self.zoom)
                          for px, py in path]
            src_color = "#ffff00" if (is_sel or is_src_sel) else "#00cc44"
            if len(src_canvas) >= 2:
                sflat = [c for pt in src_canvas for c in pt]
                self.canvas.create_line(*sflat, fill=src_color, width=1, dash=(4, 3),
                                        tags="source")
            # Square marker at the source anchor (first node).
            sx, sy = src_canvas[0]
            sq = max(3, int(2 * self.zoom))
            self.canvas.create_rectangle(sx - sq, sy - sq, sx + sq, sy + sq,
                                         outline=src_color,
                                         width=2 if is_src_sel else 1, tags="source")

    def draw_overlays(self):
        img_dict = self.images[self.current_idx]
        stem = img_dict["stem"]
        ann = self.annotations[stem]
        radius_mismatch_by_idx = {m["spot_idx"]: m for m in ann.get("radius_mismatches", [])}
        radius_overrides = ann.get("radius_overrides", {})

        # Rejected candidates (orange, shown only when toggled)
        if self.show_rejected_var.get():
            for i, r in enumerate(img_dict.get("rejected") or []):
                cx, cy = image_to_canvas(r["cx"], r["cy"],
                                         self.offset_x, self.offset_y, self.zoom)
                rad = max(4, 3 * self.zoom)
                color = "#ff8800" if i not in self.selected_rejected else "#ffcc44"
                width = 1 if i not in self.selected_rejected else 2
                self.canvas.create_oval(cx - rad, cy - rad, cx + rad, cy + rad,
                                        outline=color, width=width, tags="rejected")

        # Detected spots
        for i, spot in enumerate(img_dict.get("detected") or []):
            # --- Stroke (thread/scratch): polyline through key points ---
            if spot.get("kind") == "stroke":
                self._draw_stroke_marker(stem, ann, i, spot)
                continue

            cx, cy = image_to_canvas(spot["cx"], spot["cy"],
                                     self.offset_x, self.offset_y, self.zoom)
            rad = max(5, spot["brush_radius_px"] * self.zoom)
            is_fp = i in ann["false_positives"]
            is_sel = i in self.selected_detected
            is_src_sel = (self.selected_source == i)
            rm = radius_mismatch_by_idx.get(i)
            color = "#00cc44"
            lw = 2
            if is_sel:
                color = "#ffff00"
                lw = 3
            elif rm is not None:
                color = "#ff44cc"  # magenta: radius differs from baseline
                lw = 2
            self.canvas.create_oval(cx - rad, cy - rad, cx + rad, cy + rad,
                                    outline=color, width=lw, tags="detected")
            # Baseline radius circle (dashed orange) — drawn at true pixel size (no min clamp)
            # so the size difference is always visible when zoomed in
            if rm is not None:
                baseline_rad = rm["baseline_brush_radius_px"] * self.zoom
                if baseline_rad >= 2:
                    self.canvas.create_oval(cx - baseline_rad, cy - baseline_rad,
                                            cx + baseline_rad, cy + baseline_rad,
                                            outline="#ff9900", width=1, dash=(4, 3),
                                            tags="detected")
            # User radius correction circle (dashed cyan)
            if i in radius_overrides:
                corr_rad = max(3, radius_overrides[i] * self.zoom)
                self.canvas.create_oval(cx - corr_rad, cy - corr_rad,
                                        cx + corr_rad, cy + corr_rad,
                                        outline="#00ddff", width=2, dash=(5, 3),
                                        tags="detected")
            # FP X mark
            if is_fp:
                r2 = max(5, spot["brush_radius_px"] * self.zoom)
                self.canvas.create_line(cx - r2, cy - r2, cx + r2, cy + r2,
                                        fill="#ff3333", width=2, tags="fp")
                self.canvas.create_line(cx + r2, cy - r2, cx - r2, cy + r2,
                                        fill="#ff3333", width=2, tags="fp")
            # Source marker: dashed line from spot to source, small square at source
            src = self._get_source(stem, i, spot)
            if src is not None:
                src_cx, src_cy = src
                sx, sy = image_to_canvas(src_cx, src_cy,
                                         self.offset_x, self.offset_y, self.zoom)
                src_color = "#ffff00" if (is_sel or is_src_sel) else "#00cc44"
                sq = max(3, int(2 * self.zoom))
                self.canvas.create_line(cx, cy, sx, sy,
                                        fill=src_color, width=1, dash=(4, 3),
                                        tags="source")
                self.canvas.create_rectangle(sx - sq, sy - sq, sx + sq, sy + sq,
                                             outline=src_color,
                                             width=2 if is_src_sel else 1,
                                             tags="source")
                # Source brush circle (same radius as the spot brush)
                if self.show_source_brush_var.get():
                    src_br = radius_overrides.get(i, spot["brush_radius_px"])
                    src_brad = max(5, src_br * self.zoom)
                    self.canvas.create_oval(sx - src_brad, sy - src_brad,
                                            sx + src_brad, sy + src_brad,
                                            outline=src_color, width=1, dash=(4, 3),
                                            tags="source")

                # If source mismatch exists, also draw baseline source in red
                for mismatch in ann.get("source_mismatches", []):
                    if mismatch.get("spot_idx") == i:
                        baseline_src_cx = mismatch.get("baseline_src_cx")
                        baseline_src_cy = mismatch.get("baseline_src_cy")
                        if baseline_src_cx is not None and baseline_src_cy is not None:
                            bsx, bsy = image_to_canvas(baseline_src_cx, baseline_src_cy,
                                                       self.offset_x, self.offset_y, self.zoom)
                            # Draw baseline source in red with dashed line
                            self.canvas.create_line(cx, cy, bsx, bsy,
                                                   fill="#ff4444", width=1, dash=(2, 4),
                                                   tags="source")
                            self.canvas.create_rectangle(bsx - sq, bsy - sq, bsx + sq, bsy + sq,
                                                         outline="#ff4444", width=1,
                                                         tags="source")
                            # Draw line connecting baseline and new source
                            self.canvas.create_line(bsx, bsy, sx, sy,
                                                   fill="#ff8800", width=1, dash=(1, 2),
                                                   tags="source")
                        break

        # Missed dust markers (cyan +)
        for i, md in enumerate(ann["missed_dust"]):
            cx, cy = image_to_canvas(md["cx"], md["cy"],
                                     self.offset_x, self.offset_y, self.zoom)
            r = 10
            color = "#00ffff" if i not in self.selected_missed else "#ffffff"
            lw = 2 if i not in self.selected_missed else 3
            self.canvas.create_line(cx - r, cy, cx + r, cy,
                                    fill=color, width=lw, tags="missed")
            self.canvas.create_line(cx, cy - r, cx, cy + r,
                                    fill=color, width=lw, tags="missed")

        # Missed threads (hand-drawn, cyan polyline with handles)
        for i, ms in enumerate(ann.get("missed_strokes", [])):
            path = ms.get("path") or []
            if not path:
                continue
            canv = [image_to_canvas(px, py, self.offset_x, self.offset_y, self.zoom)
                    for px, py in path]
            sel = (self.selected_missed_stroke == i)
            color = "#ffffff" if sel else "#00ffff"
            if len(canv) >= 2:
                flat = [c for pt in canv for c in pt]
                self.canvas.create_line(*flat, fill=color, width=3 if sel else 2,
                                        capstyle="round", joinstyle="round", tags="missed")
            for (hx, hy) in canv:
                self.canvas.create_rectangle(hx - 4, hy - 4, hx + 4, hy + 4,
                                             outline=color, width=2, tags="missed")

        # In-progress thread being drawn (yellow, dashed)
        if self.thread_draw_mode and self.thread_draw_points:
            canv = [image_to_canvas(px, py, self.offset_x, self.offset_y, self.zoom)
                    for px, py in self.thread_draw_points]
            if len(canv) >= 2:
                flat = [c for pt in canv for c in pt]
                self.canvas.create_line(*flat, fill="#ffff00", width=2, dash=(5, 3),
                                        tags="missed")
            for (hx, hy) in canv:
                self.canvas.create_oval(hx - 4, hy - 4, hx + 4, hy + 4,
                                        outline="#ffff00", width=2, tags="missed")

    # ------------------------------------------------------------------
    # Interaction hooks
    # ------------------------------------------------------------------

    def on_scroll_override(self, event):
        # When exactly one detected spot is selected, scroll adjusts its radius correction
        if len(self.selected_detected) == 1:
            self._adjust_radius_correction(event)
            return True
        return False

    def _adjust_radius_correction(self, event):
        """Scroll wheel: grow/shrink the radius correction for the selected detected spot."""
        if event.num == 4:
            delta = 1
        elif event.num == 5:
            delta = -1
        else:
            delta = 1 if event.delta > 0 else -1

        idx = next(iter(self.selected_detected))
        img_dict = self.images[self.current_idx]
        stem = img_dict["stem"]
        spots = img_dict.get("detected") or []
        if idx >= len(spots):
            return
        spot = spots[idx]
        ann = self.annotations[stem]
        # Start from existing override or original detected radius
        current_r = ann["radius_overrides"].get(idx, spot["brush_radius_px"])
        factor = 1.1 if delta > 0 else (1 / 1.1)
        new_r = max(1.0, current_r * factor)
        ann["radius_overrides"][idx] = new_r
        self._auto_save(stem)
        self._redraw_markers()
        self._update_info_from_selection()

    def _thread_add_point(self, canvas_x, canvas_y):
        """Append a centerline point (from canvas coords) to the in-progress thread."""
        ix, iy = canvas_to_image(canvas_x, canvas_y,
                                 self.offset_x, self.offset_y, self.zoom)
        img_dict = self.images[self.current_idx]
        iw, ih = img_dict["width"], img_dict["height"]
        ix = max(0, min(iw, ix))
        iy = max(0, min(ih, iy))
        self.thread_draw_points.append([float(ix), float(iy)])
        self._set_info_text(
            f"THREAD DRAW: {len(self.thread_draw_points)} point(s).\n"
            f"  Click to add, or drag to draw freehand.\n"
            f"  Enter = finish,  Esc / T = cancel.")
        self._redraw_markers()

    def handle_press_override(self, event):
        # Thread-draw mode: every mouse-down drops a centerline point (immune to
        # accidental drag). Hold and drag to draw freehand (see handle_drag_override).
        if self.thread_draw_mode:
            self._thread_add_point(event.x, event.y)
            self.drag_start = (event.x, event.y)  # last freehand sample point
            self.is_dragging = False
            return True
        return False

    def handle_drag_override(self, event):
        # Thread-draw mode: freehand — sample a new point when moved enough.
        if self.thread_draw_mode:
            if self.drag_start is None:
                return True
            dx = event.x - self.drag_start[0]
            dy = event.y - self.drag_start[1]
            if (dx * dx + dy * dy) >= 64:  # ~8px on canvas
                self._thread_add_point(event.x, event.y)
                self.drag_start = (event.x, event.y)
            return True
        return False

    def handle_release_override(self, event):
        # Thread-draw mode: points were already placed on press/drag; nothing to do.
        if self.thread_draw_mode:
            self.drag_start = None
            self.is_dragging = False
            return True
        return False

    def on_rubber_band(self, ix1, iy1, ix2, iy2, additive):
        img_dict = self.images[self.current_idx]
        stem = img_dict["stem"]
        new_sel = set()
        for i, spot in enumerate(img_dict.get("detected") or []):
            if ix1 <= spot["cx"] <= ix2 and iy1 <= spot["cy"] <= iy2:
                new_sel.add(i)
        new_missed = set()
        for i, md in enumerate(self.annotations[stem]["missed_dust"]):
            if ix1 <= md["cx"] <= ix2 and iy1 <= md["cy"] <= iy2:
                new_missed.add(i)
        new_rejected = set()
        if self.show_rejected_var.get():
            for i, r in enumerate(img_dict.get("rejected") or []):
                if ix1 <= r["cx"] <= ix2 and iy1 <= r["cy"] <= iy2:
                    new_rejected.add(i)
        if additive:
            # Shift+drag: add to existing selection
            self.selected_detected |= new_sel
            self.selected_missed |= new_missed
            self.selected_rejected |= new_rejected
        else:
            self.selected_detected = new_sel
            self.selected_missed = new_missed
            self.selected_rejected = new_rejected
        self.remove_missed_btn.config(state=tk.NORMAL if self.selected_missed else tk.DISABLED)
        self._update_info_from_selection()
        self._redraw_markers()

    def on_ctrl_click(self, canvas_x, canvas_y):
        """Ctrl+Click: reposition selected source, or add a missed dust marker."""
        ix, iy = canvas_to_image(canvas_x, canvas_y,
                                 self.offset_x, self.offset_y, self.zoom)
        img_dict = self.images[self.current_idx]
        iw, ih = img_dict["width"], img_dict["height"]
        ix = max(0, min(iw, ix))
        iy = max(0, min(ih, iy))
        stem = img_dict["stem"]

        # If a stroke key point is selected, move it (persist a full path override).
        if self.selected_keypoint is not None:
            spot_idx, kp_idx = self.selected_keypoint
            spots = img_dict.get("detected") or []
            if spot_idx < len(spots) and spots[spot_idx].get("kind") == "stroke":
                overrides = self.annotations[stem].setdefault("path_overrides", {})
                # Start from current effective path (existing override or detected path).
                cur = overrides.get(spot_idx) or [list(p) for p in spots[spot_idx]["path"]]
                cur = [list(p) for p in cur]
                if kp_idx < len(cur):
                    cur[kp_idx] = [float(ix), float(iy)]
                    overrides[spot_idx] = cur
                    self._auto_save(stem)
                    self._redraw_markers()
                    self._set_info_text(
                        f"Stroke #{spot_idx} node {kp_idx} moved to ({ix:.0f}, {iy:.0f}).\n"
                        f"Ctrl+Click to move it again, or click another node.")
            return

        # If a source is selected, reposition it
        if self.selected_source is not None:
            self.annotations[stem].setdefault("source_overrides", {})[self.selected_source] = (
                float(ix), float(iy))
            self._auto_save(stem)
            self._redraw_markers()
            self._set_info_text(
                f"Source for Spot #{self.selected_source} moved to ({ix:.0f}, {iy:.0f})\n"
                f"Ctrl+Click again to reposition further.")
            return

        # If near existing marker, treat as normal click instead
        nearest = self._find_nearest_marker(canvas_x, canvas_y)
        if nearest is not None:
            self.on_click(canvas_x, canvas_y)
            return

        self.annotations[stem]["missed_dust"].append({"cx": ix, "cy": iy})
        new_idx = len(self.annotations[stem]["missed_dust"]) - 1
        self.selected_detected = set()
        self.selected_rejected = set()
        self.selected_missed = {new_idx}
        self.selected_source = None
        self.remove_missed_btn.config(state=tk.NORMAL)
        self._auto_save(stem)
        self._redraw_markers()
        self._update_count_label()
        self._populate_items_list()
        self._sync_item_list_selection()
        self._set_info_text(f"Added missed dust at ({ix:.0f}, {iy:.0f})")

    def on_shift_click(self, canvas_x, canvas_y):
        """Shift+click: toggle nearest detected or missed spot in/out of selection."""
        nearest = self._find_nearest_marker(canvas_x, canvas_y)
        if nearest is None:
            return  # Shift+click on empty area: keep existing selection
        kind, idx = nearest
        if kind == "detected":
            if idx in self.selected_detected:
                self.selected_detected.discard(idx)
            else:
                self.selected_detected.add(idx)
            self._update_info_from_selection()
            self._redraw_markers()
        elif kind == "missed":
            if idx in self.selected_missed:
                self.selected_missed.discard(idx)
            else:
                self.selected_missed.add(idx)
            self.remove_missed_btn.config(state=tk.NORMAL if self.selected_missed else tk.DISABLED)
            self._update_info_from_selection()
            self._redraw_markers()
        elif kind == "rejected":
            if idx in self.selected_rejected:
                self.selected_rejected.discard(idx)
            else:
                self.selected_rejected.add(idx)
            self._update_info_from_selection()
            self._redraw_markers()

    def on_click(self, canvas_x, canvas_y):
        """Select nearest marker on single click."""
        # A stroke key-point handle takes priority so individual nodes are selectable.
        kp = self._find_keypoint(canvas_x, canvas_y)
        if kp is not None:
            self.selected_detected = {kp[0]}
            self.selected_rejected = set()
            self.selected_missed = set()
            self.selected_source = None
            self.selected_keypoint = kp
            self.remove_missed_btn.config(state=tk.DISABLED)
            self._update_info_from_selection()
            self._set_info_text(
                f"Stroke #{kp[0]} node {kp[1]} selected.\n"
                f"Ctrl+Click to move this node. Click the source square then Ctrl+Click "
                f"to move the healing source.")
            self._redraw_markers()
            return

        nearest = self._find_nearest_marker(canvas_x, canvas_y)
        if nearest is None:
            self._clear_selection()
            return

        kind, idx = nearest
        self.selected_detected = set()
        self.selected_rejected = set()
        self.selected_missed = set()
        self.selected_source = None
        self.selected_keypoint = None
        self.selected_missed_stroke = None

        if kind == "detected":
            self.selected_detected = {idx}
            self.remove_missed_btn.config(state=tk.DISABLED)
        elif kind == "source":
            self.selected_source = idx
            self.remove_missed_btn.config(state=tk.DISABLED)
        elif kind == "rejected":
            self.selected_rejected = {idx}
            self.remove_missed_btn.config(state=tk.DISABLED)
        elif kind == "missed":
            self.selected_missed = {idx}
            self.remove_missed_btn.config(state=tk.NORMAL)
        elif kind == "missed_stroke":
            self.selected_missed_stroke = idx
            self.remove_missed_btn.config(state=tk.NORMAL)
            self._set_info_text(
                f"Missed thread #{idx} selected.\n  Del = remove it.")
            self._redraw_markers()
            return

        self._update_info_from_selection()
        self._redraw_markers()

    def _find_nearest_marker(self, canvas_x, canvas_y):
        """Return (kind, idx) of nearest marker within hit radius, or None."""
        ix, iy = canvas_to_image(canvas_x, canvas_y,
                                 self.offset_x, self.offset_y, self.zoom)
        hit_r = max(8.0, 12.0 / self.zoom)

        img_dict = self.images[self.current_idx]
        stem = img_dict["stem"]
        ann = self.annotations[stem]

        best = None
        best_dist = float("inf")

        # Missed dust (highest priority)
        for i, md in enumerate(ann["missed_dust"]):
            d = math.hypot(md["cx"] - ix, md["cy"] - iy)
            if d < hit_r and d < best_dist:
                best = ("missed", i)
                best_dist = d

        # Missed threads (hand-drawn): distance to the polyline
        for i, ms in enumerate(ann.get("missed_strokes", [])):
            path = ms.get("path") or []
            if not path:
                continue
            d = _point_to_path_dist(ix, iy, path)
            if d < hit_r and d < best_dist:
                best = ("missed_stroke", i)
                best_dist = d

        # Detected spots
        for i, spot in enumerate(img_dict.get("detected") or []):
            if spot.get("kind") == "stroke":
                # Distance to the (possibly user-edited) centerline polyline.
                path = self._get_path(stem, i, spot)
                if path:
                    d = _point_to_path_dist(ix, iy, path)
                    r_check = max(hit_r, spot.get("radius_px", 0))
                    if d < r_check and d < best_dist:
                        best = ("detected", i)
                        best_dist = d
                continue
            r_check = max(hit_r, spot["radius_px"])
            d = math.hypot(spot["cx"] - ix, spot["cy"] - iy)
            if d < r_check and d < best_dist:
                best = ("detected", i)
                best_dist = d

        # Source markers (check before rejected so they're easy to click)
        for i, spot in enumerate(img_dict.get("detected") or []):
            src = self._get_source(stem, i, spot)
            if src is not None:
                src_cx, src_cy = src
                d = math.hypot(src_cx - ix, src_cy - iy)
                if d < hit_r and d < best_dist:
                    best = ("source", i)
                    best_dist = d

        # Rejected (only when visible)
        if self.show_rejected_var.get():
            for i, r in enumerate(img_dict.get("rejected") or []):
                d = math.hypot(r["cx"] - ix, r["cy"] - iy)
                if d < hit_r and d < best_dist:
                    best = ("rejected", i)
                    best_dist = d

        return best

    def _find_keypoint(self, canvas_x, canvas_y):
        """Return (spot_idx, kp_idx) of the nearest stroke key-point handle within hit
        radius, or None. Used to select/move individual stroke nodes."""
        ix, iy = canvas_to_image(canvas_x, canvas_y,
                                 self.offset_x, self.offset_y, self.zoom)
        hit_r = max(8.0, 12.0 / self.zoom)
        img_dict = self.images[self.current_idx]
        stem = img_dict["stem"]
        best = None
        best_dist = hit_r
        for i, spot in enumerate(img_dict.get("detected") or []):
            if spot.get("kind") != "stroke":
                continue
            path = self._get_path(stem, i, spot)
            if not path:
                continue
            for k, (px, py) in enumerate(path):
                d = math.hypot(px - ix, py - iy)
                if d < best_dist:
                    best = (i, k)
                    best_dist = d
        return best

    # ------------------------------------------------------------------
    # Thread-draw mode
    # ------------------------------------------------------------------

    def _toggle_thread_draw(self):
        """Toggle thread-draw mode (T key): click to place centerline points."""
        self.thread_draw_mode = not self.thread_draw_mode
        self.thread_draw_points = []
        if self.thread_draw_mode:
            self._clear_selection()
            self._set_info_text(
                "THREAD DRAW MODE ON\n"
                "  Click along the missing thread to place points.\n"
                "  Enter = finish (needs >=2 points),  Esc / T = cancel.")
        else:
            self._set_info_text("Thread draw mode off.")
        self._redraw_markers()

    def _finish_thread(self):
        """Commit the in-progress thread as a missed-thread annotation (Enter)."""
        if not self.thread_draw_mode:
            return
        stem = self.images[self.current_idx]["stem"]
        if len(self.thread_draw_points) >= 2:
            self.annotations[stem].setdefault("missed_strokes", []).append({
                "path": [[float(x), float(y)] for x, y in self.thread_draw_points],
                "stroke_width_px": float(DEFAULT_MISSED_STROKE_WIDTH),
            })
            self._auto_save(stem)
            self._set_info_text(
                f"Added missed thread ({len(self.thread_draw_points)} points).\n"
                f"Click it then Del to remove. Press T to draw another.")
        else:
            self._set_info_text("Thread needs at least 2 points — discarded.")
        self.thread_draw_mode = False
        self.thread_draw_points = []
        self._redraw_markers()
        self._populate_items_list()

    def _cancel_thread(self):
        """Cancel the in-progress thread (Esc)."""
        if self.thread_draw_mode:
            self.thread_draw_mode = False
            self.thread_draw_points = []
            self._set_info_text("Thread draw cancelled.")
            self._redraw_markers()

    # ------------------------------------------------------------------
    # Info panel
    # ------------------------------------------------------------------

    def _update_info_from_selection(self):
        img_dict = self.images[self.current_idx]
        stem = img_dict["stem"]
        ann = self.annotations[stem]
        radius_mismatch_by_idx = {m["spot_idx"]: m for m in ann.get("radius_mismatches", [])}

        lines = []
        if self.selected_source is not None:
            i = self.selected_source
            spots = img_dict.get("detected") or []
            if i < len(spots):
                s = spots[i]
                src = self._get_source(stem, i, s)
                overrides = ann.get("source_overrides", {})
                src_kind = "user set" if i in overrides else "auto-detected"
                src_str = f"({src[0]:.0f}, {src[1]:.0f})" if src else "none"
                lines.append(
                    f"Source for Spot #{i}: {src_kind}  src={src_str}\n"
                    f"  Spot at: ({s['cx']:.0f}, {s['cy']:.0f})\n"
                    f"  Ctrl+Click anywhere to reposition.\n"
                    f"  Click elsewhere to deselect.")
        elif self.selected_detected:
            for i in sorted(self.selected_detected):
                spots = img_dict.get("detected") or []
                if i < len(spots):
                    s = spots[i]
                    is_fp = i in ann["false_positives"]
                    status = "FALSE POSITIVE (marked)" if is_fp else "accepted"
                    src = self._get_source(stem, i, s)
                    src_str = (f"\n  src=({src[0]:.0f}, {src[1]:.0f})" if src else "")
                    rm = radius_mismatch_by_idx.get(i)
                    rm_str = (
                        f"\n  RADIUS DIFF: baseline={rm['baseline_brush_radius_px']:.1f}px"
                        f"  new={rm['new_brush_radius_px']:.1f}px"
                        f"  diff={rm['radius_diff']:+.1f}px"
                        if rm else "")
                    radius_overrides = ann.get("radius_overrides", {})
                    if i in radius_overrides:
                        corr_r = radius_overrides[i]
                        rc_str = (f"\n  RADIUS CORRECTED: {corr_r:.1f}px"
                                  f"  (detected: {s['brush_radius_px']:.1f}px)"
                                  f"  scroll to adjust")
                    else:
                        rc_str = "\n  Scroll to annotate correct radius"
                    rn = s.get("radius_norm")
                    bn_str = f"  radius_norm={rn:.5f}" if rn is not None else ""
                    nf = s.get("n_frames")
                    bn_str += f"  n_frames={nf}" if nf is not None else ""
                    lines.append(
                        f"Detected #{i}: cx={s['cx']:.0f} cy={s['cy']:.0f}  "
                        f"enc_r={s['radius_px']:.1f}px{bn_str}  area={s['area']}\n"
                        f"  contrast={s['contrast']:.1f}  texture={s['texture']:.1f}  "
                        f"excess_sat={s['excess_sat']:.1f}  status={status}"
                        f"{src_str}{rm_str}{rc_str}")
            if len(self.selected_detected) > 1:
                lines = [f"{len(self.selected_detected)} spots selected"] + lines[:3]
        elif self.selected_rejected:
            rejected_list = img_dict.get("rejected") or []
            rej_lines = []
            for i in sorted(self.selected_rejected):
                if i < len(rejected_list):
                    r = rejected_list[i]
                    rej_lines.append(
                        f"Rejected #{i}: cx={r['cx']:.0f} cy={r['cy']:.0f}  "
                        f"area={r['area']}  contrast={r['contrast']:.1f}\n"
                        f"  reason={r['reason']}  detail={r['detail']}")
            if len(self.selected_rejected) > 1:
                lines.append(f"{len(self.selected_rejected)} rejected candidates selected")
            lines.extend(rej_lines[:3])
            if len(rej_lines) > 3:
                lines.append(f"  ... and {len(rej_lines) - 3} more")
        elif self.selected_missed:
            md = ann["missed_dust"]
            missed_lines = []
            for i in sorted(self.selected_missed):
                if i < len(md):
                    m = md[i]
                    missed_lines.append(f"Missed dust #{i}: cx={m['cx']:.0f} cy={m['cy']:.0f}")
            if len(self.selected_missed) > 1:
                lines.append(f"{len(self.selected_missed)} missed dust selected")
            lines.extend(missed_lines)
        else:
            lines.append("No marker selected.\nClick a marker or Ctrl+Click to add missed dust.")

        self._set_info_text("\n".join(lines))
        self._update_count_label()
        self._sync_item_list_selection()

    def _update_count_label(self):
        img_dict = self.images[self.current_idx]
        stem = img_dict["stem"]
        ann = self.annotations[stem]
        fp = len(ann["false_positives"])
        missed = len(ann["missed_dust"])
        self.count_label.config(text=f"FP: {fp} | Missed: {missed}")

    # ------------------------------------------------------------------
    # Annotation actions
    # ------------------------------------------------------------------

    def _mark_rejected_as_missed(self):
        """Add all selected rejected candidates to missed_dust list."""
        if not self.selected_rejected:
            return
        img_dict = self.images[self.current_idx]
        stem = img_dict["stem"]
        rejected_list = img_dict.get("rejected") or []
        md_list = self.annotations[stem]["missed_dust"]
        added = 0
        for idx in sorted(self.selected_rejected):
            if idx < len(rejected_list):
                r = rejected_list[idx]
                md_list.append({"cx": r["cx"], "cy": r["cy"]})
                added += 1
        self.selected_rejected = set()
        self._auto_save(stem)
        self._redraw_markers()
        self._update_count_label()
        self._populate_items_list()
        self._set_info_text(f"Added {added} rejected candidate(s) as missed dust.")

    def _mark_fp(self):
        if not self.selected_detected:
            return
        stem = self.images[self.current_idx]["stem"]
        self.annotations[stem]["false_positives"] |= self.selected_detected
        self._auto_save(stem)
        self._redraw_markers()
        self._populate_items_list()
        self._update_info_from_selection()

    def _clear_fp(self):
        if not self.selected_detected:
            return
        stem = self.images[self.current_idx]["stem"]
        self.annotations[stem]["false_positives"] -= self.selected_detected
        self._auto_save(stem)
        self._redraw_markers()
        self._populate_items_list()
        self._update_info_from_selection()

    def _remove_missed(self):
        stem = self.images[self.current_idx]["stem"]
        # Remove a selected hand-drawn missed thread first.
        if self.selected_missed_stroke is not None:
            ms_list = self.annotations[stem].get("missed_strokes", [])
            if self.selected_missed_stroke < len(ms_list):
                del ms_list[self.selected_missed_stroke]
            self.selected_missed_stroke = None
            self.remove_missed_btn.config(state=tk.DISABLED)
            self._auto_save(stem)
            self._redraw_markers()
            self._set_info_text("Missed thread removed.")
            return
        if not self.selected_missed:
            return
        md_list = self.annotations[stem]["missed_dust"]
        for idx in sorted(self.selected_missed, reverse=True):
            if idx < len(md_list):
                del md_list[idx]
        self.selected_missed = set()
        self.remove_missed_btn.config(state=tk.DISABLED)
        self._auto_save(stem)
        self._redraw_markers()
        self._update_count_label()
        self._populate_items_list()
        self._set_info_text("Missed dust marker(s) removed.")

    # ------------------------------------------------------------------
    # Item table hooks
    # ------------------------------------------------------------------

    def configure_item_tags(self, tree):
        tree.tag_configure("det",    foreground="#cccccc")
        tree.tag_configure("fp",     foreground="#ff8888")
        tree.tag_configure("missed", foreground="#00ddcc")

    def item_panel_header_text(self):
        img_dict = self.images[self.current_idx]
        ann = self.annotations[img_dict["stem"]]
        detected = img_dict.get("detected") or []
        missed = ann["missed_dust"]
        missed_strokes = ann.get("missed_strokes", [])
        stroke_suffix = f", {len(missed_strokes)} thread(s)" if missed_strokes else ""
        return f"{len(detected)} detected, {len(missed)} missed{stroke_suffix}"

    def item_rows(self):
        img_dict = self.images[self.current_idx]
        stem = img_dict["stem"]
        ann = self.annotations[stem]
        detected = img_dict.get("detected") or []
        missed = ann["missed_dust"]
        missed_strokes = ann.get("missed_strokes", [])

        rows = []
        for i, spot in enumerate(detected):
            is_fp = i in ann["false_positives"]
            rows.append({
                "iid": f"det_{i}",
                "values": (i,
                           int(spot["cx"]),
                           int(spot["cy"]),
                           int(spot["brush_radius_px"]),
                           f"{spot['contrast']:.1f}",
                           f"{spot['texture']:.1f}",
                           "●" if is_fp else ""),
                "tag": "fp" if is_fp else "det",
                "kind": "detected", "idx": i,
                "sort": {"idx": i,
                         "cx":  spot["cx"],            "cy":  spot["cy"],
                         "r":   spot["brush_radius_px"],
                         "ctr": spot["contrast"],      "tex": spot["texture"],
                         "fp":  1 if is_fp else 0},
            })

        for i, md in enumerate(missed):
            rows.append({
                "iid": f"missed_{i}",
                "values": (f"m{i}", int(md["cx"]), int(md["cy"]), "", "", "", ""),
                "tag": "missed",
                "kind": "missed", "idx": i,
                "sort": {"idx": 100000 + i,
                         "cx":  md["cx"],  "cy":  md["cy"],
                         "r": -1, "ctr": -1, "tex": -1, "fp": 0},
            })

        for i, ms in enumerate(missed_strokes):
            path = ms.get("path", [])
            if not path:
                continue
            mid = path[len(path) // 2]
            cx, cy = mid[0], mid[1]
            rows.append({
                "iid": f"mstroke_{i}",
                "values": (f"t{i}", int(cx), int(cy), "", "", "", ""),
                "tag": "missed",
                "kind": "missed_stroke", "idx": i,
                "sort": {"idx": 200000 + i,
                         "cx": cx, "cy": cy,
                         "r": -1, "ctr": -1, "tex": -1, "fp": 0},
            })
        return rows

    def is_row_currently_selected(self, row):
        if row["kind"] == "detected" and row["idx"] in self.selected_detected:
            return True
        if row["kind"] == "missed" and row["idx"] in self.selected_missed:
            return True
        if row["kind"] == "missed_stroke" and row["idx"] == self.selected_missed_stroke:
            return True
        return False

    def on_item_row_selected(self, row):
        self.selected_detected = set()
        self.selected_rejected = set()
        self.selected_missed = set()
        self.selected_missed_stroke = None
        self.selected_source = None

        if row["kind"] == "detected":
            self.selected_detected = {row["idx"]}
            self.remove_missed_btn.config(state=tk.DISABLED)
        elif row["kind"] == "missed":
            self.selected_missed = {row["idx"]}
            self.remove_missed_btn.config(state=tk.NORMAL)
        elif row["kind"] == "missed_stroke":
            self.selected_missed_stroke = row["idx"]
            self.remove_missed_btn.config(state=tk.NORMAL)

        self._update_info_from_selection()
        self._redraw_markers()

    def selected_row_iid(self):
        if self.selected_detected:
            return f"det_{next(iter(self.selected_detected))}"
        if self.selected_missed:
            return f"missed_{next(iter(self.selected_missed))}"
        return None

    def selection_center(self):
        img_dict = self.images[self.current_idx]
        stem = img_dict["stem"]
        if self.selected_detected:
            idx = next(iter(self.selected_detected))
            spots = img_dict.get("detected") or []
            if idx < len(spots):
                return (spots[idx]["cx"], spots[idx]["cy"])
        elif self.selected_missed:
            idx = next(iter(self.selected_missed))
            md = self.annotations[stem]["missed_dust"]
            if idx < len(md):
                return (md[idx]["cx"], md[idx]["cy"])
        elif self.selected_missed_stroke is not None:
            ms_list = self.annotations[stem].get("missed_strokes", [])
            idx = self.selected_missed_stroke
            if idx < len(ms_list):
                path = ms_list[idx].get("path", [])
                if path:
                    mid = path[len(path) // 2]
                    return (mid[0], mid[1])
        return None

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def report_title(self):
        if self.sensor_mode:
            return "SENSOR DUST DETECTION DEBUG REPORT"
        return "DUST DETECTION DEBUG REPORT"

    def report_constants_lines(self):
        constants = self.data.get("constants", {})
        lines = ["Detection constants at time of run:"]
        if self.sensor_mode:
            for key in ["SENSOR_SIGMA_INNER_FRAC", "SENSOR_SIGMA_OUTER_FRAC",
                        "SENSOR_DOG_MIN_CONTRAST", "SENSOR_MIN_RADIUS_FRAC",
                        "SENSOR_MAX_BLOB_RADIUS_FRAC", "SENSOR_CLUSTER_RADIUS_NORM",
                        "SENSOR_DUST_MIN_FRAMES", "SENSOR_BRUSH_SCALE",
                        "SENSOR_MAX_CANDIDATE_TEXTURE",
                        "SENSOR_MAX_CANDIDATES_FOR_CONSENSUS",
                        "SENSOR_MAX_CORRECTION_TEXTURE", "SENSOR_MAX_SOURCE_TEXTURE",
                        "MAX_FORMS", "MIN_BRUSH_PX"]:
                if key in constants:
                    lines.append(f"  {key} = {constants[key]}")
            return lines
        # Key constants
        for key in ["NOISE_THRESHOLD_MULTIPLIER", "MIN_ABSOLUTE_THRESHOLD",
                    "MIN_SPOT_AREA", "MAX_SPOT_AREA", "MIN_ASPECT_RATIO",
                    "MIN_COMPACTNESS", "MIN_SOLIDITY", "MIN_CIRCULARITY",
                    "MAX_LOCAL_TEXTURE_SMALL", "MAX_LOCAL_TEXTURE_LARGE",
                    "MIN_CONTRAST_TEXTURE_RATIO", "MAX_BG_GRADIENT_RATIO",
                    "MAX_EXCESS_SATURATION", "MAX_CONTEXT_TEXTURE",
                    "LARGE_SPOT_AREA_THRESHOLD", "LARGE_SPOT_MIN_CONTRAST",
                    "ISOLATION_RADIUS", "MAX_NEARBY_ACCEPTED",
                    "SOFT_CONTEXT_VOTE_THRESHOLD", "SOFT_TEXTURE_VOTE_THRESHOLD",
                    "SOFT_RATIO_VOTE_THRESHOLD", "MIN_DUST_VOTES",
                    "REJECT_LOG_CONTRAST_MIN",
                    "MIN_BRUSH_PX", "BRUSH_HARDNESS"]:
            if key in constants:
                lines.append(f"  {key} = {constants[key]}")
        return lines

    def report_body_lines(self):
        lines = []
        total_fp = 0
        total_missed = 0
        total_src_repositions = 0
        images_with_annotations = 0

        for img_dict in self.images:
            stem = img_dict["stem"]
            ann = self.annotations[stem]
            detected = img_dict.get("detected") or []
            rejected_list = img_dict.get("rejected") or []
            fp_indices = ann["false_positives"]
            missed_list = ann["missed_dust"]
            src_overrides = ann.get("source_overrides", {})

            radius_overrides = ann.get("radius_overrides", {})
            total_fp += len(fp_indices)
            total_missed += len(missed_list)
            total_src_repositions += len(src_overrides)
            has_annotations = bool(fp_indices or missed_list or src_overrides or radius_overrides)
            if has_annotations:
                images_with_annotations += 1

            lines.append("=" * 48)
            lines.append(f"IMAGE: {stem}  ({img_dict['width']} x {img_dict['height']} px)")
            lines.append(f"  Detected: {len(detected)} | "
                         f"Rejected shown: {len(rejected_list)} | "
                         f"FP marked: {len(fp_indices)} | "
                         f"Missed added: {len(missed_list)}")

            if not has_annotations:
                lines.append("  No user annotations.")
                lines.append("")
                continue

            # False positives
            lines.append("")
            if fp_indices:
                lines.append("  FALSE POSITIVES (detected, should be ignored):")
                for i in sorted(fp_indices):
                    if i < len(detected):
                        s = detected[i]
                        lines.append(
                            f"    cx={s['cx']:.1f}  cy={s['cy']:.1f}  "
                            f"enc_r={s['radius_px']:.1f}px  area={s['area']}  "
                            f"contrast={s['contrast']:.1f}  texture={s['texture']:.1f}  "
                            f"excess_sat={s['excess_sat']:.1f}")
            else:
                lines.append("  FALSE POSITIVES: none")

            # Missed dust
            lines.append("")
            if missed_list:
                lines.append("  MISSED DUST (not detected, should have been):")
                for m in missed_list:
                    lines.append(f"    cx={m['cx']:.0f}  cy={m['cy']:.0f}")
            else:
                lines.append("  MISSED DUST: none")

            # Source repositions
            src_overrides = ann.get("source_overrides", {})
            if src_overrides:
                lines.append("")
                lines.append("  SOURCE REPOSITIONS (user-adjusted heal sources):")
                for spot_idx in sorted(src_overrides.keys()):
                    src_cx, src_cy = src_overrides[spot_idx]
                    lines.append(f"    Spot #{spot_idx}: src=({src_cx:.0f}, {src_cy:.0f})")

            # Radius corrections
            if radius_overrides:
                lines.append("")
                lines.append("  RADIUS CORRECTIONS (user-annotated correct brush radius):")
                for spot_idx in sorted(radius_overrides.keys()):
                    corr_r = radius_overrides[spot_idx]
                    orig_r = detected[spot_idx]["brush_radius_px"] if spot_idx < len(detected) else "?"
                    orig_str = f"{orig_r:.1f}" if isinstance(orig_r, float) else str(orig_r)
                    lines.append(
                        f"    Spot #{spot_idx}: corrected={corr_r:.1f}px  detected={orig_str}px")

            # Rejected candidates (for context)
            if rejected_list:
                lines.append("")
                lines.append(f"  REJECTED CANDIDATES ({len(rejected_list)} total, for context):")
                for r in rejected_list[:50]:
                    lines.append(
                        f"    cx={r['cx']:.0f}  cy={r['cy']:.0f}  "
                        f"area={r['area']}  contrast={r['contrast']:.1f}  "
                        f"reason={r['reason']}  detail={r['detail']}")
                if len(rejected_list) > 50:
                    lines.append(f"    ... and {len(rejected_list) - 50} more")

            lines.append("")

        lines.append("=" * 48)
        lines.append("SUMMARY")
        lines.append(f"  Total false positives across all images: {total_fp}")
        lines.append(f"  Total missed dust across all images: {total_missed}")
        lines.append(f"  Total source repositions across all images: {total_src_repositions}")
        lines.append(f"  Images with annotations: {images_with_annotations} / {len(self.images)}")
        lines.append("")
        return lines


# Backwards-compatible alias (external callers may refer to DebugUI)
DebugUI = DustDebugUI


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    DustDebugUI.run_main(usage="Usage: debug_ui.py <export_dir>")


if __name__ == "__main__":
    main()
