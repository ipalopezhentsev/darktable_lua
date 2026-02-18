"""
Debug UI for dust detection annotation.

Lets the user review detected spots and rejected candidates on each exported
image, mark false positives, and add missed dust points. Produces a
debug_report.txt readable by Claude Code for algorithm tuning.

Usage:
    python debug_ui.py <export_dir>

Reads:  {export_dir}/debug_spots.json
Writes: {export_dir}/{stem}_annotations.json  (auto-saved per image)
        {export_dir}/debug_report.txt          (on window close)
"""

import sys
import os
import json
import math
import datetime
from pathlib import Path

import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def canvas_to_image(cx, cy, offset_x, offset_y, zoom):
    return (cx - offset_x) / zoom, (cy - offset_y) / zoom


def image_to_canvas(ix, iy, offset_x, offset_y, zoom):
    return ix * zoom + offset_x, iy * zoom + offset_y


# ---------------------------------------------------------------------------
# Main UI class
# ---------------------------------------------------------------------------

class DebugUI:
    def __init__(self, root, export_dir):
        self.root = root
        self.export_dir = export_dir
        self.root.title("Dust Detection Debug UI")
        self.root.geometry("1400x900")

        # Load data
        json_path = os.path.join(export_dir, "debug_spots.json")
        if not os.path.exists(json_path):
            messagebox.showerror("Error", f"debug_spots.json not found in:\n{export_dir}")
            root.destroy()
            return

        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.images = self.data.get("images", [])
        if not self.images:
            messagebox.showerror("Error", "No images found in debug_spots.json")
            root.destroy()
            return

        # Per-image annotation state: {stem: {"false_positives": set(), "missed_dust": []}}
        self.annotations = {}
        for img in self.images:
            stem = img["stem"]
            self.annotations[stem] = {"false_positives": set(), "missed_dust": []}

        # Load any previously saved annotations
        self._load_existing_annotations()

        # View state
        self.current_idx = 0
        self.zoom = 1.0
        self.fit_zoom = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.pil_image = None
        self.photo_image = None

        # Selection state
        self.selected_detected = set()   # set of int (detected spot indices)
        self.selected_rejected = set()   # set of int (rejected candidate indices)
        self.selected_missed = set()     # set of int (indices into missed_dust list)

        # Visibility
        self.hide_markers = False

        # Rubber-band drag
        self.drag_start = None
        self.is_dragging = False
        self.ctrl_drag_mode = False

        # Pan state
        self.pan_start = None
        self.pan_offset_at_start = None

        # Track whether we're at fit zoom so resize can re-fit automatically
        self.at_fit_zoom = True

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Populate thumbnail list (after _build_ui so lb_rows etc. exist)
        self._populate_thumb_list()

        # Defer initial load until window is rendered so winfo_width() works
        self.root.after(150, self._load_image_by_idx, 0)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # PanedWindow splits left panel from right
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                               sashrelief=tk.RAISED, sashwidth=6)
        paned.pack(fill=tk.BOTH, expand=True)

        # ---- LEFT PANEL ----
        left = tk.Frame(paned, width=220, bg="#2b2b2b")
        left.pack_propagate(False)
        paned.add(left, minsize=160)

        tk.Label(left, text="Images:", bg="#2b2b2b", fg="white",
                 font=("", 10, "bold")).pack(anchor="w", padx=6, pady=(6, 2))

        lb_frame = tk.Frame(left, bg="#2b2b2b")
        lb_frame.pack(fill=tk.BOTH, expand=True, padx=4)
        lb_sb = tk.Scrollbar(lb_frame, orient=tk.VERTICAL)
        self.lb_canvas = tk.Canvas(lb_frame, bg="#1e1e1e",
                                   yscrollcommand=lb_sb.set, highlightthickness=0)
        lb_sb.config(command=self.lb_canvas.yview)
        lb_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.lb_canvas.pack(fill=tk.BOTH, expand=True)
        self.lb_inner = tk.Frame(self.lb_canvas, bg="#1e1e1e")
        self.lb_canvas_window = self.lb_canvas.create_window(
            0, 0, anchor="nw", window=self.lb_inner)
        self.lb_inner.bind("<Configure>", self._on_lb_inner_configure)
        self.lb_canvas.bind("<Configure>", self._on_lb_outer_configure)
        self.lb_canvas.bind("<MouseWheel>", self._on_lb_scroll)
        self.lb_rows = []        # Frame per image row
        self.lb_photos = []      # PhotoImage refs per row (prevent GC)

        # Show rejected toggle
        self.show_rejected_var = tk.BooleanVar(value=False)
        tk.Checkbutton(left, text="Show rejected candidates",
                       variable=self.show_rejected_var,
                       command=self._redraw_markers,
                       bg="#2b2b2b", fg="white", selectcolor="#1e1e1e",
                       activebackground="#2b2b2b", activeforeground="white"
                       ).pack(anchor="w", padx=6, pady=2)

        self.status_label = tk.Label(left, text="", bg="#2b2b2b", fg="#aaaaaa",
                                     font=("", 9), wraplength=200, justify=tk.LEFT)
        self.status_label.pack(anchor="w", padx=6, pady=2)

        # Legend
        tk.Label(left, text="Legend:", bg="#2b2b2b", fg="#aaaaaa",
                 font=("", 8, "bold")).pack(anchor="w", padx=6, pady=(6, 1))
        for symbol, color, label in [
            ("●", "#00cc44", "Detected spot"),
            ("✕", "#ff3333", "False positive"),
            ("●", "#ff8800", "Rejected candidate"),
            ("✚", "#00ffff", "Missed dust (added)"),
        ]:
            row = tk.Frame(left, bg="#2b2b2b")
            row.pack(anchor="w", padx=6)
            tk.Label(row, text=symbol, bg="#2b2b2b", fg=color,
                     font=("", 11)).pack(side=tk.LEFT, padx=(0, 4))
            tk.Label(row, text=label, bg="#2b2b2b", fg="#cccccc",
                     font=("", 8)).pack(side=tk.LEFT)

        # Hints
        tk.Label(left, text=(
            "Mouse:\n"
            "  Scroll — zoom in/out\n"
            "  Middle drag — pan\n"
            "  Click — select marker\n"
            "  Drag — multi-select\n"
            "  Shift+click/drag — add to sel\n"
            "  Ctrl+click — add missed\n"
            "  Ctrl+drag — zoom to rect\n"
            "Keys:\n"
            "  R — rejected → missed\n"
            "  M — mark false positive\n"
            "  C — clear FP mark\n"
            "  Del — remove missed marker\n"
            "  H — hide/show all markers\n"
            "  +/- — zoom ×2 / ÷2\n"
            "  Arrows — pan  (Shift=fast)\n"
            "  F — fit to window"
        ), bg="#2b2b2b", fg="white", font=("Courier", 8),
                 justify=tk.LEFT).pack(anchor="w", padx=6, pady=(8, 4))

        # ---- RIGHT PANEL ----
        right = tk.Frame(paned, bg="#1e1e1e")
        paned.add(right)

        # Canvas + scrollbars
        canvas_frame = tk.Frame(right, bg="#1e1e1e")
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="#1e1e1e",
                                cursor="crosshair", highlightthickness=0)
        h_scroll = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL,
                                command=self.canvas.xview)
        v_scroll = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL,
                                command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=h_scroll.set,
                              yscrollcommand=v_scroll.set)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Canvas events
        self.canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_end)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)       # Windows
        self.canvas.bind("<Button-4>", self._on_mousewheel)          # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_mousewheel)          # Linux scroll down
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.root.bind("<f>", lambda e: self._fit_to_window())
        self.root.bind("<F>", lambda e: self._fit_to_window())
        self.root.bind("<m>", lambda e: self._mark_fp())
        self.root.bind("<M>", lambda e: self._mark_fp())
        self.root.bind("<c>", lambda e: self._clear_fp())
        self.root.bind("<C>", lambda e: self._clear_fp())
        self.root.bind("<Delete>", lambda e: self._remove_missed())
        self.root.bind("<BackSpace>", lambda e: self._remove_missed())
        self.root.bind("<r>", lambda e: self._mark_rejected_as_missed())
        self.root.bind("<R>", lambda e: self._mark_rejected_as_missed())
        self.root.bind("<h>", lambda e: self._toggle_hide_markers())
        self.root.bind("<H>", lambda e: self._toggle_hide_markers())
        self.root.bind("<equal>", lambda e: self._zoom_step(2.0))    # + key (no shift)
        self.root.bind("<plus>", lambda e: self._zoom_step(2.0))     # + key (with shift)
        self.root.bind("<minus>", lambda e: self._zoom_step(0.5))
        pan_step = 80
        pan_big  = 300
        self.root.bind("<Left>",          lambda e: self._pan_by( pan_step, 0))
        self.root.bind("<Right>",         lambda e: self._pan_by(-pan_step, 0))
        self.root.bind("<Up>",            lambda e: self._pan_by(0,  pan_step))
        self.root.bind("<Down>",          lambda e: self._pan_by(0, -pan_step))
        self.root.bind("<Shift-Left>",    lambda e: self._pan_by( pan_big, 0))
        self.root.bind("<Shift-Right>",   lambda e: self._pan_by(-pan_big, 0))
        self.root.bind("<Shift-Up>",      lambda e: self._pan_by(0,  pan_big))
        self.root.bind("<Shift-Down>",    lambda e: self._pan_by(0, -pan_big))

        # Bottom panel
        bottom = tk.Frame(right, bg="#2b2b2b", height=190)
        bottom.pack(fill=tk.X)
        bottom.pack_propagate(False)

        # Scrollable info text
        info_frame = tk.Frame(bottom, bg="#2b2b2b")
        info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=6)
        tk.Label(info_frame, text="Selected:", bg="#2b2b2b", fg="#aaaaaa",
                 font=("", 9, "bold")).pack(anchor="w")
        info_text_wrap = tk.Frame(info_frame, bg="#2b2b2b")
        info_text_wrap.pack(fill=tk.BOTH, expand=True)
        info_sb = tk.Scrollbar(info_text_wrap, orient=tk.VERTICAL)
        self.info_text = tk.Text(info_text_wrap, bg="#1e1e1e", fg="white",
                                 font=("Courier", 9), wrap=tk.WORD,
                                 state=tk.DISABLED, height=7,
                                 yscrollcommand=info_sb.set,
                                 relief=tk.FLAT, padx=4, pady=4,
                                 cursor="arrow")
        info_sb.config(command=self.info_text.yview)
        info_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        self._set_info_text("No marker selected.\n"
                            "Click a marker or Ctrl+Click to add missed dust.")

        # Buttons
        btn_frame = tk.Frame(bottom, bg="#2b2b2b")
        btn_frame.pack(side=tk.RIGHT, padx=8, pady=6)

        self.count_label = tk.Label(btn_frame, text="FP: 0 | Missed: 0",
                                    bg="#2b2b2b", fg="#aaaaaa", font=("", 9))
        self.count_label.pack(anchor="e", pady=(0, 4))

        btn_cfg = {"bg": "#3a3a3a", "fg": "white", "relief": tk.FLAT,
                   "padx": 8, "pady": 4, "width": 20}

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

        tk.Button(btn_frame, text="Clear Selection",
                  command=self._clear_selection, **btn_cfg).pack(fill=tk.X, pady=1)

        tk.Button(btn_frame, text="Fit to Window  (F)",
                  command=self._fit_to_window, **btn_cfg).pack(fill=tk.X, pady=(6, 1))

        self.hide_btn = tk.Button(btn_frame, text="Hide Markers  (H)",
                                  command=self._toggle_hide_markers, **btn_cfg)
        self.hide_btn.pack(fill=tk.X, pady=1)

    # ------------------------------------------------------------------
    # Info text helper
    # ------------------------------------------------------------------

    def _set_info_text(self, text):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, text)
        self.info_text.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _load_image_by_idx(self, idx):
        self.current_idx = idx
        img_dict = self.images[idx]
        image_path = img_dict["image_path"]

        if not os.path.exists(image_path):
            messagebox.showerror("Error", f"Image not found:\n{image_path}")
            return

        self.pil_image = Image.open(image_path)

        # Fit-to-window zoom
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        iw, ih = img_dict["width"], img_dict["height"]
        if iw > 0 and ih > 0 and cw > 1 and ch > 1:
            self.fit_zoom = min(cw / iw, ch / ih)
        else:
            self.fit_zoom = 0.2
        self.zoom = self.fit_zoom
        self.offset_x = (cw - iw * self.zoom) / 2
        self.offset_y = (ch - ih * self.zoom) / 2
        self.at_fit_zoom = True

        # Reset selection
        self.selected_detected = set()
        self.selected_rejected = set()
        self.selected_missed = set()
        self.remove_missed_btn.config(state=tk.DISABLED)
        self._set_info_text("No marker selected.\n"
                            "Click a marker or Ctrl+Click to add missed dust.")

        # Update status
        detected = img_dict.get("detected") or []
        rejected_list = img_dict.get("rejected") or []
        self.status_label.config(
            text=f"{len(detected)} detected\n{len(rejected_list)} rejected candidates")

        self._redraw()
        self._update_count_label()

        # Highlight selected row and scroll to it
        self._highlight_lb_row(idx)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _redraw(self):
        self.canvas.delete("all")
        if self.pil_image is None:
            return

        img_dict = self.images[self.current_idx]
        iw, iw_h = img_dict["width"], img_dict["height"]
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()

        # Draw visible crop of image
        vis_x1 = max(0, int(-self.offset_x / self.zoom))
        vis_y1 = max(0, int(-self.offset_y / self.zoom))
        vis_x2 = min(iw, int((cw - self.offset_x) / self.zoom) + 1)
        vis_y2 = min(iw_h, int((ch - self.offset_y) / self.zoom) + 1)

        if vis_x2 > vis_x1 and vis_y2 > vis_y1:
            resample = Image.LANCZOS if self.zoom >= 0.2 else Image.NEAREST
            crop = self.pil_image.crop((vis_x1, vis_y1, vis_x2, vis_y2))
            dw = max(1, int((vis_x2 - vis_x1) * self.zoom))
            dh = max(1, int((vis_y2 - vis_y1) * self.zoom))
            resized = crop.resize((dw, dh), resample)
            self.photo_image = ImageTk.PhotoImage(resized)
            draw_x = self.offset_x + vis_x1 * self.zoom
            draw_y = self.offset_y + vis_y1 * self.zoom
            self.canvas.create_image(draw_x, draw_y, anchor="nw",
                                     image=self.photo_image, tags="bg")

        self._redraw_markers()

    def _redraw_markers(self):
        self.canvas.delete("detected", "rejected", "fp", "missed", "sel", "rubberband")
        if self.pil_image is None or self.hide_markers:
            return

        img_dict = self.images[self.current_idx]
        stem = img_dict["stem"]
        ann = self.annotations[stem]

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
            cx, cy = image_to_canvas(spot["cx"], spot["cy"],
                                     self.offset_x, self.offset_y, self.zoom)
            rad = max(5, spot["radius_px"] * self.zoom)
            is_fp = i in ann["false_positives"]
            is_sel = i in self.selected_detected
            color = "#00cc44"
            lw = 2
            if is_sel:
                color = "#ffff00"
                lw = 3
            self.canvas.create_oval(cx - rad, cy - rad, cx + rad, cy + rad,
                                    outline=color, width=lw, tags="detected")
            # FP X mark
            if is_fp:
                r2 = max(5, spot["radius_px"] * self.zoom)
                self.canvas.create_line(cx - r2, cy - r2, cx + r2, cy + r2,
                                        fill="#ff3333", width=2, tags="fp")
                self.canvas.create_line(cx + r2, cy - r2, cx - r2, cy + r2,
                                        fill="#ff3333", width=2, tags="fp")

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

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def _zoom_step(self, factor):
        """Zoom in or out by factor, centered on the canvas centre."""
        new_zoom = max(0.05, min(20.0, self.zoom * factor))
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        mx, my = cw / 2, ch / 2
        self.offset_x = mx - (mx - self.offset_x) * (new_zoom / self.zoom)
        self.offset_y = my - (my - self.offset_y) * (new_zoom / self.zoom)
        self.zoom = new_zoom
        self.at_fit_zoom = False
        self._redraw()

    def _pan_by(self, dx, dy):
        """Shift the image by dx/dy canvas pixels."""
        self.offset_x += dx
        self.offset_y += dy
        self.at_fit_zoom = False
        self._redraw()

    def _on_mousewheel(self, event):
        # Determine zoom direction
        if event.num == 4:
            delta = 1
        elif event.num == 5:
            delta = -1
        else:
            delta = 1 if event.delta > 0 else -1

        factor = 1.15 if delta > 0 else (1 / 1.15)
        new_zoom = max(0.05, min(20.0, self.zoom * factor))

        # Zoom centered on mouse position
        mx, my = event.x, event.y
        self.offset_x = mx - (mx - self.offset_x) * (new_zoom / self.zoom)
        self.offset_y = my - (my - self.offset_y) * (new_zoom / self.zoom)
        self.zoom = new_zoom
        self.at_fit_zoom = False
        self._redraw()

    def _on_pan_start(self, event):
        self.pan_start = (event.x, event.y)
        self.pan_offset_at_start = (self.offset_x, self.offset_y)

    def _on_pan_move(self, event):
        if self.pan_start is None:
            return
        dx = event.x - self.pan_start[0]
        dy = event.y - self.pan_start[1]
        self.offset_x = self.pan_offset_at_start[0] + dx
        self.offset_y = self.pan_offset_at_start[1] + dy
        self._redraw()

    def _on_pan_end(self, event):
        self.pan_start = None
        self.pan_offset_at_start = None

    def _on_left_press(self, event):
        self.drag_start = (event.x, event.y)
        self.is_dragging = False
        self.ctrl_drag_mode = bool(event.state & 0x4)

    def _on_left_drag(self, event):
        if self.drag_start is None:
            return
        dx = event.x - self.drag_start[0]
        dy = event.y - self.drag_start[1]
        if not self.is_dragging and (abs(dx) > 5 or abs(dy) > 5):
            self.is_dragging = True

        if self.is_dragging:
            self.canvas.delete("rubberband")
            x0, y0 = self.drag_start
            color = "#4488ff" if self.ctrl_drag_mode else "white"
            self.canvas.create_rectangle(x0, y0, event.x, event.y,
                                         outline=color, width=2, dash=(4, 2),
                                         tags="rubberband")

    def _on_left_release(self, event):
        if self.drag_start is None:
            return

        if self.ctrl_drag_mode:
            self.canvas.delete("rubberband")
            if self.is_dragging:
                self._zoom_to_rect(self.drag_start, (event.x, event.y))
            else:
                self._do_add_missed(event.x, event.y)
            self.drag_start = None
            self.is_dragging = False
            self.ctrl_drag_mode = False
            return

        shift_held = bool(event.state & 0x1)

        if self.is_dragging:
            # Rubber-band select detected spots
            x0, y0 = self.drag_start
            x1, y1 = event.x, event.y
            rx1, rx2 = min(x0, x1), max(x0, x1)
            ry1, ry2 = min(y0, y1), max(y0, y1)
            # Convert to image coords
            ix1, iy1 = canvas_to_image(rx1, ry1, self.offset_x, self.offset_y, self.zoom)
            ix2, iy2 = canvas_to_image(rx2, ry2, self.offset_x, self.offset_y, self.zoom)
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
            if shift_held:
                # Shift+drag: add to existing selection
                self.selected_detected |= new_sel
                self.selected_missed |= new_missed
                self.selected_rejected |= new_rejected
            else:
                self.selected_detected = new_sel
                self.selected_missed = new_missed
                self.selected_rejected = new_rejected
            self.remove_missed_btn.config(state=tk.NORMAL if self.selected_missed else tk.DISABLED)
            self.canvas.delete("rubberband")
            self._update_info_from_selection()
            self._redraw_markers()
        else:
            if shift_held:
                # Shift+click: toggle nearest detected spot in/out of selection
                self._handle_shift_click(event.x, event.y)
            else:
                # Normal click — find nearest marker
                self._handle_click(event.x, event.y)

        self.drag_start = None
        self.is_dragging = False

    def _zoom_to_rect(self, start, end):
        """Zoom and pan so the rubber-band rectangle fills the canvas."""
        x0, y0 = start
        x1, y1 = end
        rx1, rx2 = min(x0, x1), max(x0, x1)
        ry1, ry2 = min(y0, y1), max(y0, y1)
        if rx2 - rx1 < 5 or ry2 - ry1 < 5:
            return
        # Convert rubber-band corners to image coordinates
        ix1, iy1 = canvas_to_image(rx1, ry1, self.offset_x, self.offset_y, self.zoom)
        ix2, iy2 = canvas_to_image(rx2, ry2, self.offset_x, self.offset_y, self.zoom)
        rect_w = ix2 - ix1
        rect_h = iy2 - iy1
        if rect_w <= 0 or rect_h <= 0:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        new_zoom = max(0.05, min(20.0, min(cw / rect_w, ch / rect_h)))
        cx_img = (ix1 + ix2) / 2
        cy_img = (iy1 + iy2) / 2
        self.offset_x = cw / 2 - cx_img * new_zoom
        self.offset_y = ch / 2 - cy_img * new_zoom
        self.zoom = new_zoom
        self.at_fit_zoom = False
        self._redraw()

    def _do_add_missed(self, canvas_x, canvas_y):
        """Ctrl+Click: immediately add a missed dust marker."""
        ix, iy = canvas_to_image(canvas_x, canvas_y,
                                 self.offset_x, self.offset_y, self.zoom)
        img_dict = self.images[self.current_idx]
        iw, ih = img_dict["width"], img_dict["height"]
        ix = max(0, min(iw, ix))
        iy = max(0, min(ih, iy))

        # If near existing marker, treat as normal click instead
        nearest = self._find_nearest_marker(canvas_x, canvas_y)
        if nearest is not None:
            self._handle_click(canvas_x, canvas_y)
            return

        stem = img_dict["stem"]
        self.annotations[stem]["missed_dust"].append({"cx": ix, "cy": iy})
        new_idx = len(self.annotations[stem]["missed_dust"]) - 1
        self.selected_detected = set()
        self.selected_rejected = set()
        self.selected_missed = {new_idx}
        self.remove_missed_btn.config(state=tk.NORMAL)
        self._auto_save(stem)
        self._redraw_markers()
        self._update_count_label()
        self._set_info_text(f"Added missed dust at ({ix:.0f}, {iy:.0f})")

    def _handle_shift_click(self, canvas_x, canvas_y):
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

    def _toggle_hide_markers(self):
        """Toggle visibility of all markers (H key)."""
        self.hide_markers = not self.hide_markers
        label = "Show Markers  (H)" if self.hide_markers else "Hide Markers  (H)"
        self.hide_btn.config(text=label,
                             fg="#ffff88" if self.hide_markers else "white")
        self._redraw_markers()

    def _fit_to_window(self):
        """Zoom image to fit the canvas, centered."""
        if self.pil_image is None:
            return
        img_dict = self.images[self.current_idx]
        iw, ih = img_dict["width"], img_dict["height"]
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if iw > 0 and ih > 0 and cw > 1 and ch > 1:
            self.fit_zoom = min(cw / iw, ch / ih)
        self.zoom = self.fit_zoom
        self.offset_x = (cw - iw * self.zoom) / 2
        self.offset_y = (ch - ih * self.zoom) / 2
        self.at_fit_zoom = True
        self._redraw()

    def _on_canvas_configure(self, event):
        """Canvas resized: re-fit image if we were at fit zoom."""
        if self.at_fit_zoom and self.pil_image is not None:
            # Use after() to let the resize settle before measuring
            self.root.after(50, self._fit_to_window)

    def _handle_click(self, canvas_x, canvas_y):
        """Select nearest marker on single click."""
        nearest = self._find_nearest_marker(canvas_x, canvas_y)
        if nearest is None:
            self._clear_selection()
            return

        kind, idx = nearest
        self.selected_detected = set()
        self.selected_rejected = set()
        self.selected_missed = set()

        if kind == "detected":
            self.selected_detected = {idx}
            self.remove_missed_btn.config(state=tk.DISABLED)
        elif kind == "rejected":
            self.selected_rejected = {idx}
            self.remove_missed_btn.config(state=tk.DISABLED)
        elif kind == "missed":
            self.selected_missed = {idx}
            self.remove_missed_btn.config(state=tk.NORMAL)

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

        # Detected spots
        for i, spot in enumerate(img_dict.get("detected") or []):
            r_check = max(hit_r, spot["radius_px"])
            d = math.hypot(spot["cx"] - ix, spot["cy"] - iy)
            if d < r_check and d < best_dist:
                best = ("detected", i)
                best_dist = d

        # Rejected (only when visible)
        if self.show_rejected_var.get():
            for i, r in enumerate(img_dict.get("rejected") or []):
                d = math.hypot(r["cx"] - ix, r["cy"] - iy)
                if d < hit_r and d < best_dist:
                    best = ("rejected", i)
                    best_dist = d

        return best

    # ------------------------------------------------------------------
    # Info panel
    # ------------------------------------------------------------------

    def _update_info_from_selection(self):
        img_dict = self.images[self.current_idx]
        stem = img_dict["stem"]
        ann = self.annotations[stem]

        lines = []
        if self.selected_detected:
            for i in sorted(self.selected_detected):
                spots = img_dict.get("detected") or []
                if i < len(spots):
                    s = spots[i]
                    is_fp = i in ann["false_positives"]
                    status = "FALSE POSITIVE (marked)" if is_fp else "accepted"
                    lines.append(
                        f"Detected #{i}: cx={s['cx']:.0f} cy={s['cy']:.0f}  "
                        f"radius={s['radius_px']:.1f}px  area={s['area']}\n"
                        f"  contrast={s['contrast']:.1f}  texture={s['texture']:.1f}  "
                        f"excess_sat={s['excess_sat']:.1f}  status={status}")
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
        self._set_info_text(f"Added {added} rejected candidate(s) as missed dust.")

    def _mark_fp(self):
        if not self.selected_detected:
            return
        stem = self.images[self.current_idx]["stem"]
        self.annotations[stem]["false_positives"] |= self.selected_detected
        self._auto_save(stem)
        self._redraw_markers()
        self._update_info_from_selection()

    def _clear_fp(self):
        if not self.selected_detected:
            return
        stem = self.images[self.current_idx]["stem"]
        self.annotations[stem]["false_positives"] -= self.selected_detected
        self._auto_save(stem)
        self._redraw_markers()
        self._update_info_from_selection()

    def _remove_missed(self):
        if not self.selected_missed:
            return
        stem = self.images[self.current_idx]["stem"]
        md_list = self.annotations[stem]["missed_dust"]
        for idx in sorted(self.selected_missed, reverse=True):
            if idx < len(md_list):
                del md_list[idx]
        self.selected_missed = set()
        self.remove_missed_btn.config(state=tk.DISABLED)
        self._auto_save(stem)
        self._redraw_markers()
        self._update_count_label()
        self._set_info_text("Missed dust marker(s) removed.")

    def _clear_selection(self):
        self.selected_detected = set()
        self.selected_rejected = set()
        self.selected_missed = set()
        self.remove_missed_btn.config(state=tk.DISABLED)
        self._set_info_text("No marker selected.\n"
                            "Click a marker or Ctrl+Click to add missed dust.")
        self._redraw_markers()

    # ------------------------------------------------------------------
    # Thumbnail list
    # ------------------------------------------------------------------

    def _populate_thumb_list(self):
        for i, img_dict in enumerate(self.images):
            row = tk.Frame(self.lb_inner, bg="#1e1e1e", cursor="hand2")
            row.pack(fill=tk.X, padx=2, pady=(2, 0))

            photo = None
            try:
                pil_img = Image.open(img_dict["image_path"])
                pil_img.thumbnail((210, 140), Image.LANCZOS)
                photo = ImageTk.PhotoImage(pil_img)
            except Exception:
                pass
            self.lb_photos.append(photo)

            def _bind_click(w, idx=i):
                w.bind("<Button-1>", lambda e: self._on_thumb_row_click(idx))

            if photo:
                lbl_img = tk.Label(row, image=photo, bg="#1e1e1e",
                                   cursor="hand2", anchor="center")
                lbl_img.pack(fill=tk.X)
                _bind_click(lbl_img)

            lbl_txt = tk.Label(row, text=img_dict["stem"], bg="#1e1e1e",
                               fg="#cccccc", font=("", 8), wraplength=200,
                               cursor="hand2", anchor="w")
            lbl_txt.pack(fill=tk.X, padx=2, pady=(1, 3))
            _bind_click(lbl_txt)
            _bind_click(row)
            self.lb_rows.append(row)

    def _on_thumb_row_click(self, idx):
        if idx == self.current_idx and self.pil_image is not None:
            return
        stem = self.images[self.current_idx]["stem"]
        self._auto_save(stem)
        self._load_image_by_idx(idx)

    def _highlight_lb_row(self, idx):
        sel_bg = "#3a5a8f"
        norm_bg = "#1e1e1e"
        for i, row in enumerate(self.lb_rows):
            bg = sel_bg if i == idx else norm_bg
            row.config(bg=bg)
            for child in row.winfo_children():
                child.config(bg=bg)
        # Scroll to show selected row
        if idx < len(self.lb_rows):
            self.lb_inner.update_idletasks()
            row = self.lb_rows[idx]
            ry = row.winfo_y()
            rh = row.winfo_height()
            ch = self.lb_canvas.winfo_height()
            total = self.lb_inner.winfo_height()
            if total > ch:
                frac = max(0.0, min(1.0, (ry - ch // 2 + rh // 2) / total))
                self.lb_canvas.yview_moveto(frac)

    def _on_lb_inner_configure(self, event):
        self.lb_canvas.configure(scrollregion=self.lb_canvas.bbox("all"))

    def _on_lb_outer_configure(self, event):
        self.lb_canvas.itemconfig(self.lb_canvas_window, width=event.width)

    def _on_lb_scroll(self, event):
        self.lb_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _auto_save(self, stem):
        ann = self.annotations[stem]
        img_dict = next((d for d in self.images if d["stem"] == stem), None)
        detected = img_dict.get("detected") or [] if img_dict else []

        fp_list = []
        for idx in sorted(ann["false_positives"]):
            if idx < len(detected):
                fp_list.append(detected[idx])

        data = {
            "stem": stem,
            "false_positives": fp_list,
            "missed_dust": ann["missed_dust"],
        }
        ann_path = os.path.join(self.export_dir, f"{stem}_annotations.json")
        with open(ann_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_existing_annotations(self):
        """Load any previously saved annotation files."""
        for img in self.images:
            stem = img["stem"]
            ann_path = os.path.join(self.export_dir, f"{stem}_annotations.json")
            if not os.path.exists(ann_path):
                continue
            try:
                with open(ann_path, "r") as f:
                    data = json.load(f)
            except Exception:
                continue

            detected = img.get("detected") or []
            # Rebuild false_positives as set of indices by matching cx/cy
            fp_indices = set()
            for fp in data.get("false_positives", []):
                for i, spot in enumerate(detected):
                    if abs(spot["cx"] - fp["cx"]) < 0.5 and abs(spot["cy"] - fp["cy"]) < 0.5:
                        fp_indices.add(i)
                        break
            self.annotations[stem]["false_positives"] = fp_indices
            self.annotations[stem]["missed_dust"] = data.get("missed_dust", [])

    # ------------------------------------------------------------------
    # Report generation & close
    # ------------------------------------------------------------------

    def _on_close(self):
        # Save current image annotations
        stem = self.images[self.current_idx]["stem"]
        self._auto_save(stem)
        self._write_debug_report()
        self.root.destroy()

    def _write_debug_report(self):
        constants = self.data.get("constants", {})
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            "DUST DETECTION DEBUG REPORT",
            f"Generated: {now}",
            f"Export dir: {self.export_dir}",
            f"Images processed: {len(self.images)}",
            "",
            "Detection constants at time of run:",
        ]
        # Key constants
        for key in ["NOISE_THRESHOLD_MULTIPLIER", "MIN_ABSOLUTE_THRESHOLD",
                    "MIN_SPOT_AREA", "MAX_SPOT_AREA", "MIN_ASPECT_RATIO",
                    "MIN_COMPACTNESS", "MIN_SOLIDITY", "MIN_CIRCULARITY",
                    "MAX_LOCAL_TEXTURE_SMALL", "MAX_LOCAL_TEXTURE_LARGE",
                    "MIN_CONTRAST_TEXTURE_RATIO", "MAX_BG_GRADIENT_RATIO",
                    "MAX_EXCESS_SATURATION", "REJECT_LOG_CONTRAST_MIN"]:
            if key in constants:
                lines.append(f"  {key} = {constants[key]}")
        lines.append("")

        total_fp = 0
        total_missed = 0
        images_with_annotations = 0

        for img_dict in self.images:
            stem = img_dict["stem"]
            ann = self.annotations[stem]
            detected = img_dict.get("detected") or []
            rejected_list = img_dict.get("rejected") or []
            fp_indices = ann["false_positives"]
            missed_list = ann["missed_dust"]

            total_fp += len(fp_indices)
            total_missed += len(missed_list)
            has_annotations = bool(fp_indices or missed_list)
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
                            f"radius={s['radius_px']:.1f}px  area={s['area']}  "
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
        lines.append(f"  Images with annotations: {images_with_annotations} / {len(self.images)}")
        lines.append("")

        report_path = os.path.join(self.export_dir, "debug_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Debug report written to: {report_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: debug_ui.py <export_dir>")
        sys.exit(1)

    export_dir = sys.argv[1]
    if not os.path.isdir(export_dir):
        print(f"Error: directory not found: {export_dir}")
        sys.exit(1)

    root = tk.Tk()
    app = DebugUI(root, export_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
