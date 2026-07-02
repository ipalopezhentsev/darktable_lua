"""
Debug UI for auto-crop edge annotation.

Shows the detected film-holder crop edges over each exported image so the
user can mark wrong edges by placing the correct position. Produces a
debug_report.txt readable by Claude Code for algorithm tuning — a correction
IS the wrong-edge marker (detected vs corrected position, in % and px).

Built on the shared viewer base in common/debug_ui_base.py.

Usage:
    python debug_ui.py <session_dir>

Reads:  {session_dir}/{stem}_debug_crop.json   (per-image crop detection data)
Writes: {session_dir}/{stem}_annotations.json  (auto-saved per image)
        {session_dir}/debug_report.txt          (on window close)

Interaction (same idiom as the dust UI: select, then Ctrl+Click / scroll):
    Click edge line / 1-4 keys — select edge (1=left 2=top 3=right 4=bottom)
    Ctrl+Click — place the selected (or nearest) edge at the clicked position
    Scroll (edge selected) — nudge corrected position by 1 px (Shift = 10 px)
    C — clear the correction for the selected edge
"""

import os
import sys
from pathlib import Path

import tkinter as tk

sys.path.insert(0, str(Path(__file__).parent.parent))   # repo root -> common
sys.path.insert(0, str(Path(__file__).parent))           # feature dir -> auto_crop

from common.debug_ui_base import DebugUIBase, image_to_canvas, canvas_to_image


EDGES = ("left", "top", "right", "bottom")
EDGE_KEYS = {"1": "left", "2": "top", "3": "right", "4": "bottom"}


class CropDebugUI(DebugUIBase):
    WINDOW_TITLE = "Auto Crop Debug UI"
    EMPTY_SESSION_MESSAGE = "No *_debug_crop.json files found in:"
    ITEM_PANEL_TITLE = "Edges:"
    CENTER_BUTTON_TEXT = "Center on edge"
    ITEM_COLS    = ("edge", "det", "conf", "corr", "dpx", "nt")
    ITEM_HEADERS = {"edge": "edge", "det": "det%", "conf": "conf",
                    "corr": "corr%", "dpx": "Δpx", "nt": "✎"}
    ITEM_WIDTHS  = {"edge": 48, "det": 46, "conf": 38, "corr": 46, "dpx": 40,
                    "nt": 22}
    ITEM_ANCHORS = {"nt": "center"}
    NOTE_COLUMN  = "nt"

    # ------------------------------------------------------------------
    # Session / annotation lifecycle
    # ------------------------------------------------------------------

    def load_session(self, session_dir):
        import auto_crop
        return auto_crop.load_debug_crop_dir(session_dir)

    def new_annotation_state(self, img_dict):
        return {
            "edge_corrections": {},   # {edge: corrected_pct (% from that edge)}
            "edge_notes": {},         # {edge: str} free-text user notes
        }

    def serialize_annotations(self, stem):
        img_dict = next((d for d in self.images if d["stem"] == stem), None)
        crop = img_dict["crop"] if img_dict else {}
        corrections = self.annotations[stem]["edge_corrections"]
        notes = self.annotations[stem]["edge_notes"]
        return {
            "stem": stem,
            "edge_corrections": {
                edge: {
                    "detected_pct": float(crop.get(edge, 0.0)),
                    "corrected_pct": float(pct),
                }
                for edge, pct in sorted(corrections.items())
            },
            "edge_notes": {edge: note for edge, note in sorted(notes.items())},
        }

    def deserialize_annotations(self, img_dict, data):
        stem = img_dict["stem"]
        corrections = {}
        for edge, entry in data.get("edge_corrections", {}).items():
            if edge in EDGES:
                try:
                    corrections[edge] = float(entry["corrected_pct"])
                except (KeyError, ValueError, TypeError):
                    pass
        self.annotations[stem]["edge_corrections"] = corrections
        self.annotations[stem]["edge_notes"] = {
            edge: str(note)
            for edge, note in data.get("edge_notes", {}).items()
            if edge in EDGES and note
        }

    def write_apply_results(self):
        """Write crop_results.txt for the Lua apply step: per-frame L/T/R/B as
        percent-from-edge, using the user's corrected_pct where they marked an
        edge, else the detected crop%. Matches auto_crop.write_crop_results'
        format so the Lua parse_crop_results reads it identically."""
        lines = []
        for img in self.images:
            stem = img["stem"]
            crop = img.get("crop") or {}
            corr = self.annotations.get(stem, {}).get("edge_corrections", {})
            vals = {edge: float(corr.get(edge, crop.get(edge, 0.0)))
                    for edge in EDGES}
            lines.append(
                f'OK|{stem}|L={vals["left"]:.2f}|T={vals["top"]:.2f}'
                f'|R={vals["right"]:.2f}|B={vals["bottom"]:.2f}')
        path = os.path.join(self.session_dir, "crop_results.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        print(f"Apply results (crop) written to: {path} ({len(lines)} frame(s))")

    # ------------------------------------------------------------------
    # Edge geometry helpers
    # ------------------------------------------------------------------

    def _edge_pos_px(self, img_dict, edge, pct):
        """Pixel coordinate of an edge line from its percent-from-edge value."""
        w, h = img_dict["width"], img_dict["height"]
        if edge == "left":
            return w * pct / 100.0
        if edge == "right":
            return w * (1.0 - pct / 100.0)
        if edge == "top":
            return h * pct / 100.0
        return h * (1.0 - pct / 100.0)

    def _edge_px_to_pct(self, img_dict, edge, pos):
        """Inverse of _edge_pos_px: pixel coordinate -> percent from that edge."""
        w, h = img_dict["width"], img_dict["height"]
        if edge == "left":
            pct = pos / w * 100.0
        elif edge == "right":
            pct = (1.0 - pos / w) * 100.0
        elif edge == "top":
            pct = pos / h * 100.0
        else:
            pct = (1.0 - pos / h) * 100.0
        return max(0.0, min(90.0, pct))

    def _effective_pct(self, img_dict, edge):
        """Corrected percent if annotated, else detected."""
        ann = self.annotations[img_dict["stem"]]
        return ann["edge_corrections"].get(edge, img_dict["crop"][edge])

    def _conf_color(self, confidence):
        if confidence >= 0.7:
            return "#00cc44"
        if confidence >= 0.4:
            return "#ffcc00"
        return "#ff4444"

    # ------------------------------------------------------------------
    # Selection state
    # ------------------------------------------------------------------

    def init_selection_state(self):
        self.selected_edge = None   # edge name or None

    def reset_selection(self):
        self.selected_edge = None

    # ------------------------------------------------------------------
    # Layout / text hooks
    # ------------------------------------------------------------------

    # Marker legend + the full mouse/key reference live on the Help menu (popup /
    # dialog), NOT in the left panel — same as auto_negadoctor / auto_retouch.
    _LEGEND_ENTRIES = [
        ("│", "#00cc44", "Detected edge (conf ≥ 0.7)"),
        ("│", "#ffcc00", "Detected edge (conf ≥ 0.4)"),
        ("│", "#ff4444", "Detected edge (low conf)"),
        ("│", "#ffff00", "Selected edge"),
        ("┊", "#00ddff", "Corrected position"),
    ]

    def extend_help_menu(self, helpm):
        # _show_legend / _show_shortcuts (both non-modal popups) live in the base.
        helpm.add_command(label="Marker legend…", command=self._show_legend)
        helpm.add_command(label="Mouse & keyboard shortcuts…",
                          command=self._show_shortcuts)

    _SHORTCUTS_TEXT = (
        "MOUSE\n"
        "  Scroll                zoom in / out\n"
        "  Scroll (edge sel'd)   nudge edge 1 px (Shift = 10 px)\n"
        "  Middle drag           pan\n"
        "  Click                 select edge\n"
        "  Ctrl+click            place edge\n"
        "  Ctrl+drag             zoom to rectangle\n\n"
        "KEYS\n"
        "  Space / B   next / previous image\n"
        "  1 / 2 / 3 / 4   select left / top / right / bottom edge\n"
        "  C           clear the correction\n"
        "  Note box    comment on the selected edge (saves as you type)\n"
        "  H           hide / show edges\n"
        "  P           display color management on / off (sRGB->monitor)\n"
        "  + / -       zoom ×2 / ÷2     Arrows pan (Shift = fast)\n"
        "  F           fit to window"
    )

    def build_feature_buttons(self, btn_frame, btn_cfg):
        self.count_label = tk.Label(btn_frame, text="Corrected edges: 0/4",
                                    bg="#484848", fg="#c0c0c0", font=("", 9))
        self.count_label.pack(anchor="e", pady=(0, 4))

        self.clear_corr_btn = tk.Button(btn_frame, text="Clear Correction  (C)",
                                        command=self._clear_correction, **btn_cfg)
        self.clear_corr_btn.pack(fill=tk.X, pady=1)

    def bind_feature_keys(self):
        for key, edge in EDGE_KEYS.items():
            self.bind_key(f"<Key-{key}>",
                          lambda e, edge=edge: self._select_edge(edge))
        self.bind_key("<c>", lambda e: self._clear_correction())
        self.bind_key("<C>", lambda e: self._clear_correction())

    def image_status_text(self, img_dict):
        conf = img_dict["confidence"]
        t = img_dict.get("processing_time_s")
        time_str = f"\nprocessed in {t:.2f}s" if t is not None else ""
        if self.run_wall_time_s is not None:
            time_str += f"\nrun total {self.run_wall_time_s:.1f}s"
        return (f"{img_dict['width']} × {img_dict['height']} px\n"
                f"conf: L={conf['left']:.2f} T={conf['top']:.2f}\n"
                f"      R={conf['right']:.2f} B={conf['bottom']:.2f}"
                f"{time_str}")

    def default_info_text(self):
        return ("No edge selected.\n"
                "Click an edge line or press 1/2/3/4 (left/top/right/bottom).")

    def update_counts(self):
        self._update_count_label()

    # ------------------------------------------------------------------
    # Note hooks (free-text user annotation per edge)
    # ------------------------------------------------------------------

    def get_selected_note(self):
        if self.selected_edge is None:
            return None
        ann = self.annotations[self.images[self.current_idx]["stem"]]
        return ann["edge_notes"].get(self.selected_edge, "")

    def set_selected_note(self, text):
        if self.selected_edge is None:
            return
        ann = self.annotations[self.images[self.current_idx]["stem"]]
        if text:
            ann["edge_notes"][self.selected_edge] = text
        else:
            ann["edge_notes"].pop(self.selected_edge, None)

    def _update_count_label(self):
        ann = self.annotations[self.images[self.current_idx]["stem"]]
        n = len(ann["edge_corrections"])
        self.count_label.config(text=f"Corrected edges: {n}/4")

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def overlay_tags(self):
        return ("edges", "corrections", "labels")

    def _edge_line_canvas(self, img_dict, edge, pos):
        """Canvas endpoints for a full-length edge line at image coord pos."""
        w, h = img_dict["width"], img_dict["height"]
        if edge in ("left", "right"):
            x0, y0 = image_to_canvas(pos, 0, self.offset_x, self.offset_y, self.zoom)
            x1, y1 = image_to_canvas(pos, h, self.offset_x, self.offset_y, self.zoom)
        else:
            x0, y0 = image_to_canvas(0, pos, self.offset_x, self.offset_y, self.zoom)
            x1, y1 = image_to_canvas(w, pos, self.offset_x, self.offset_y, self.zoom)
        return x0, y0, x1, y1

    def draw_overlays(self):
        img_dict = self.images[self.current_idx]
        ann = self.annotations[img_dict["stem"]]
        crop = img_dict["crop"]
        conf = img_dict["confidence"]

        for edge in EDGES:
            det_pos = self._edge_pos_px(img_dict, edge, crop[edge])
            sel = (self.selected_edge == edge)
            color = "#ffff00" if sel else self._conf_color(conf[edge])
            x0, y0, x1, y1 = self._edge_line_canvas(img_dict, edge, det_pos)
            self.canvas.create_line(x0, y0, x1, y1, fill=color,
                                    width=3 if sel else 2, tags="edges")

            # Confidence label near the edge, 25% along, nudged inward
            lx = x0 + (x1 - x0) * 0.25
            ly = y0 + (y1 - y0) * 0.25
            nudge = 14
            if edge == "left":
                lx += nudge
            elif edge == "right":
                lx -= nudge
            elif edge == "top":
                ly += nudge
            else:
                ly -= nudge
            self.canvas.create_text(lx, ly, text=f"{edge} {conf[edge]:.2f}",
                                    fill=color, font=("Courier", 9, "bold"),
                                    tags="labels")

            # Corrected position (dashed cyan), drawn alongside the detected line
            corr_pct = ann["edge_corrections"].get(edge)
            if corr_pct is not None:
                corr_pos = self._edge_pos_px(img_dict, edge, corr_pct)
                cx0, cy0, cx1, cy1 = self._edge_line_canvas(img_dict, edge, corr_pos)
                self.canvas.create_line(cx0, cy0, cx1, cy1, fill="#00ddff",
                                        width=2, dash=(6, 4), tags="corrections")

    # ------------------------------------------------------------------
    # Interaction hooks
    # ------------------------------------------------------------------

    def _find_nearest_edge(self, canvas_x, canvas_y, max_dist=12.0):
        """Nearest edge line (by effective position) within max_dist canvas px."""
        img_dict = self.images[self.current_idx]
        best = None
        best_dist = max_dist
        for edge in EDGES:
            pos = self._edge_pos_px(img_dict, edge, self._effective_pct(img_dict, edge))
            if edge in ("left", "right"):
                line_c = image_to_canvas(pos, 0, self.offset_x, self.offset_y, self.zoom)[0]
                d = abs(canvas_x - line_c)
            else:
                line_c = image_to_canvas(0, pos, self.offset_x, self.offset_y, self.zoom)[1]
                d = abs(canvas_y - line_c)
            if d < best_dist:
                best = edge
                best_dist = d
        return best

    def _select_edge(self, edge):
        self.selected_edge = edge
        self._update_info_from_selection()
        self._redraw_markers()

    def on_click(self, canvas_x, canvas_y):
        edge = self._find_nearest_edge(canvas_x, canvas_y)
        if edge is None:
            self._clear_selection()
            self._sync_item_list_selection()
            return
        self._select_edge(edge)

    def on_ctrl_click(self, canvas_x, canvas_y):
        """Place the selected (or nearest) edge at the clicked position."""
        edge = self.selected_edge or self._find_nearest_edge(canvas_x, canvas_y,
                                                             max_dist=float("inf"))
        if edge is None:
            return
        img_dict = self.images[self.current_idx]
        stem = img_dict["stem"]
        ix, iy = canvas_to_image(canvas_x, canvas_y,
                                 self.offset_x, self.offset_y, self.zoom)
        pos = ix if edge in ("left", "right") else iy
        pct = self._edge_px_to_pct(img_dict, edge, pos)
        self.annotations[stem]["edge_corrections"][edge] = pct
        self.selected_edge = edge
        self._auto_save(stem)
        self._redraw_markers()
        self._update_count_label()
        self._populate_items_list()
        self._update_info_from_selection()

    def on_scroll_override(self, event):
        """When an edge is selected, scroll nudges its corrected position."""
        if self.selected_edge is None:
            return False
        if event.num == 4:
            delta = 1
        elif event.num == 5:
            delta = -1
        else:
            delta = 1 if event.delta > 0 else -1
        step = 10.0 if (event.state & 0x1) else 1.0   # Shift = 10 px

        img_dict = self.images[self.current_idx]
        stem = img_dict["stem"]
        edge = self.selected_edge
        # Nudge in pixel space from the current effective position.
        # Horizontal edges move opposite to vertical ones: scroll up moves
        # top/bottom lines up (-y) but left/right lines right (+x).
        if edge in ("top", "bottom"):
            delta = -delta
        pos = self._edge_pos_px(img_dict, edge, self._effective_pct(img_dict, edge))
        pos += delta * step
        pct = self._edge_px_to_pct(img_dict, edge, pos)
        self.annotations[stem]["edge_corrections"][edge] = pct
        self._auto_save(stem)
        self._redraw_markers()
        self._update_count_label()
        self._populate_items_list()
        self._update_info_from_selection()
        return True

    def _clear_correction(self):
        """Remove the correction for the selected edge (revert to detected)."""
        if self.selected_edge is None:
            return
        img_dict = self.images[self.current_idx]
        stem = img_dict["stem"]
        corrections = self.annotations[stem]["edge_corrections"]
        if self.selected_edge in corrections:
            del corrections[self.selected_edge]
            self._auto_save(stem)
            self._redraw_markers()
            self._update_count_label()
            self._populate_items_list()
        self._update_info_from_selection()

    # ------------------------------------------------------------------
    # Info panel
    # ------------------------------------------------------------------

    def _update_info_from_selection(self):
        if self.selected_edge is None:
            self._set_info_text(self.default_info_text())
            self._sync_item_list_selection()
            return
        img_dict = self.images[self.current_idx]
        ann = self.annotations[img_dict["stem"]]
        edge = self.selected_edge
        det_pct = img_dict["crop"][edge]
        det_px = self._edge_pos_px(img_dict, edge, det_pct)
        conf = img_dict["confidence"][edge]
        lines = [f"Edge: {edge.upper()}  detected={det_pct:.2f}% ({det_px:.0f}px)  "
                 f"confidence={conf:.2f}"]
        corr_pct = ann["edge_corrections"].get(edge)
        if corr_pct is not None:
            corr_px = self._edge_pos_px(img_dict, edge, corr_pct)
            lines.append(f"  CORRECTED: {corr_pct:.2f}% ({corr_px:.0f}px)  "
                         f"error={corr_pct - det_pct:+.2f}% ({corr_px - det_px:+.0f}px)")
            lines.append("  Scroll to fine-tune, Ctrl+Click to re-place, C to clear.")
        else:
            lines.append("  Ctrl+Click at the correct position to mark this edge wrong,")
            lines.append("  or scroll to nudge from the detected position.")
        self._set_info_text("\n".join(lines))
        self._sync_item_list_selection()

    # ------------------------------------------------------------------
    # Item table hooks
    # ------------------------------------------------------------------

    def configure_item_tags(self, tree):
        tree.tag_configure("det",     foreground="#cccccc")
        tree.tag_configure("corr",    foreground="#00ddcc")
        tree.tag_configure("lowconf", foreground="#ff8888")

    def item_panel_header_text(self):
        img_dict = self.images[self.current_idx]
        ann = self.annotations[img_dict["stem"]]
        return f"{len(ann['edge_corrections'])} of 4 edges corrected"

    def item_rows(self):
        img_dict = self.images[self.current_idx]
        ann = self.annotations[img_dict["stem"]]
        crop = img_dict["crop"]
        conf = img_dict["confidence"]

        rows = []
        for k, edge in enumerate(EDGES):
            det_pct = crop[edge]
            corr_pct = ann["edge_corrections"].get(edge)
            has_note = bool(ann["edge_notes"].get(edge))
            if corr_pct is not None:
                det_px = self._edge_pos_px(img_dict, edge, det_pct)
                corr_px = self._edge_pos_px(img_dict, edge, corr_pct)
                dpx = corr_px - det_px
                corr_str = f"{corr_pct:.2f}"
                dpx_str = f"{dpx:+.0f}"
                tag = "corr"
            else:
                dpx = 0.0
                corr_str = ""
                dpx_str = ""
                tag = "lowconf" if conf[edge] < 0.4 else "det"
            rows.append({
                "iid": f"edge_{edge}",
                "values": (edge, f"{det_pct:.2f}", f"{conf[edge]:.2f}",
                           corr_str, dpx_str, "✎" if has_note else ""),
                "tag": tag,
                "edge": edge,
                "sort": {"edge": k, "det": det_pct, "conf": conf[edge],
                         "corr": corr_pct if corr_pct is not None else -1.0,
                         "dpx": abs(dpx),
                         "nt": 1 if has_note else 0},
            })
        return rows

    def is_row_currently_selected(self, row):
        return row["edge"] == self.selected_edge

    def on_item_row_selected(self, row):
        self._select_edge(row["edge"])

    def selected_row_iid(self):
        if self.selected_edge is not None:
            return f"edge_{self.selected_edge}"
        return None

    def selection_center(self):
        if self.selected_edge is None:
            return None
        img_dict = self.images[self.current_idx]
        edge = self.selected_edge
        pos = self._edge_pos_px(img_dict, edge, self._effective_pct(img_dict, edge))
        if edge in ("left", "right"):
            return (pos, img_dict["height"] / 2)
        return (img_dict["width"] / 2, pos)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def report_title(self):
        return "AUTO CROP DEBUG REPORT"

    def report_constants_lines(self):
        constants = self.data.get("constants", {})
        lines = ["Detection constants at time of run:"]
        for key in sorted(constants):
            lines.append(f"  {key} = {constants[key]}")
        return lines

    def report_body_lines(self):
        lines = []
        corrected_by_edge = {edge: 0 for edge in EDGES}
        all_errors_pct = []
        all_errors_px = []
        images_with_corrections = 0

        for img_dict in self.images:
            stem = img_dict["stem"]
            ann = self.annotations[stem]
            crop = img_dict["crop"]
            conf = img_dict["confidence"]
            corrections = ann["edge_corrections"]
            notes = ann["edge_notes"]

            lines.append("=" * 48)
            lines.append(f"IMAGE: {stem}  ({img_dict['width']} x {img_dict['height']} px)")
            lines.append("  Detected crop (% from edge, confidence):")
            for edge in EDGES:
                det_px = self._edge_pos_px(img_dict, edge, crop[edge])
                lines.append(f"    {edge:<7} {crop[edge]:6.2f}%  ({det_px:5.0f}px)  "
                             f"conf={conf[edge]:.2f}")

            if not corrections and not notes:
                lines.append("  No corrections — detection accepted.")
                lines.append("")
                continue

            if corrections:
                images_with_corrections += 1
                lines.append("")
                lines.append("  CORRECTED EDGES (detected was wrong):")
                for edge in EDGES:
                    if edge not in corrections:
                        continue
                    corrected_by_edge[edge] += 1
                    det_pct = crop[edge]
                    corr_pct = corrections[edge]
                    det_px = self._edge_pos_px(img_dict, edge, det_pct)
                    corr_px = self._edge_pos_px(img_dict, edge, corr_pct)
                    err_pct = corr_pct - det_pct
                    err_px = corr_px - det_px
                    all_errors_pct.append(abs(err_pct))
                    all_errors_px.append(abs(err_px))
                    lines.append(
                        f"    {edge:<7} detected={det_pct:.2f}% ({det_px:.0f}px) "
                        f"conf={conf[edge]:.2f}  ->  corrected={corr_pct:.2f}% "
                        f"({corr_px:.0f}px)  error={err_pct:+.2f}% ({err_px:+.0f}px)")

            if notes:
                lines.append("")
                lines.append("  USER NOTES (free-text comments on edges):")
                for edge in EDGES:
                    if edge in notes:
                        lines.append(f"    {edge:<7} {notes[edge]}")
            lines.append("")

        lines.append("=" * 48)
        lines.append("SUMMARY")
        total_corrections = sum(corrected_by_edge.values())
        lines.append(f"  Total corrected edges: {total_corrections}")
        for edge in EDGES:
            lines.append(f"    {edge:<7} {corrected_by_edge[edge]}")
        if all_errors_pct:
            lines.append(f"  Mean abs error: {sum(all_errors_pct)/len(all_errors_pct):.2f}% "
                         f"({sum(all_errors_px)/len(all_errors_px):.0f}px)")
            lines.append(f"  Max abs error:  {max(all_errors_pct):.2f}% "
                         f"({max(all_errors_px):.0f}px)")
        lines.append(f"  Images with corrections: {images_with_corrections} / "
                     f"{len(self.images)}")
        lines.append("")
        return lines


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # --apply: apply-capable session => on close pop the finish dialog and, if the
    # user chose to apply, write crop_results.txt for the Lua apply step.
    if "--apply" in sys.argv:
        CropDebugUI.CLOSE_DIALOG = True
        sys.argv = [a for a in sys.argv if a != "--apply"]
    CropDebugUI.run_main(usage="Usage: debug_ui.py <session_dir> [--apply]")


if __name__ == "__main__":
    main()
