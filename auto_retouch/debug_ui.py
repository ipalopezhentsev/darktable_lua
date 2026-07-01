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
import os
import copy
import json
import queue
import threading
from pathlib import Path

import tkinter as tk
import tkinter.ttk as ttk

sys.path.insert(0, str(Path(__file__).parent.parent))   # repo root -> common
sys.path.insert(0, str(Path(__file__).parent))           # feature dir -> detect_dust

from common.debug_ui_base import (
    DebugUIBase, canvas_to_image, image_to_canvas, _point_to_path_dist)


# Default full width (px) assigned to a hand-drawn missed thread.
DEFAULT_MISSED_STROKE_WIDTH = 8.0


def _coerce_param(name, value):
    """Coerce a param-override value to its tuning type (mirrors schema._coerce):
    bool / int / tuple-of-floats / float, keyed by the tuning field-type sets."""
    import tuning
    if name in tuning.BOOL_FIELDS:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if name in tuning.INT_FIELDS:
        return int(round(float(value)))
    if name in tuning.TUPLE_FIELDS:
        if isinstance(value, str):
            parts = [p for p in value.replace(",", " ").split() if p]
        else:
            parts = list(value)
        return tuple(float(p) for p in parts)
    return float(value)


def _params_json_safe(overrides):
    """Make a param_overrides dict JSON-serializable (tuples -> lists)."""
    out = {}
    for k, v in overrides.items():
        out[k] = list(v) if isinstance(v, tuple) else v
    return out


class _MenuEntryState:
    """Adapter so the existing `self.remove_missed_btn.config(state=…)` call
    sites drive a MENU ENTRY's enabled state now that the action moved off the
    lower-left button column onto the Annotate menu (SHOW_BOTTOM_BUTTONS=False)."""

    def __init__(self, menu, index):
        self._menu, self._index = menu, index

    def config(self, **kw):
        if "state" in kw:
            try:
                self._menu.entryconfig(self._index, state=kw["state"])
            except tk.TclError:
                pass

    configure = config


class DustDebugUI(DebugUIBase):
    WINDOW_TITLE = "Dust Detection Debug UI"
    EMPTY_SESSION_MESSAGE = "No *_debug_spots.json files found in:"
    ITEM_PANEL_TITLE = "Spots:"
    CENTER_BUTTON_TEXT = "Center on spot"
    ITEM_COLS    = ("idx", "cx", "cy", "r", "ctr", "tex", "fp", "nt")
    ITEM_HEADERS = {"idx": "#",   "cx": "cx",  "cy": "cy",
                    "r":   "r",   "ctr": "ctr", "tex": "tex", "fp": "FP",
                    "nt": "✎"}
    ITEM_WIDTHS  = {"idx": 30, "cx": 46, "cy": 46,
                    "r":  30, "ctr": 40, "tex": 36, "fp": 24, "nt": 22}
    ITEM_ANCHORS = {"fp": "center", "nt": "center"}
    NOTE_COLUMN  = "nt"
    SHOW_BOTTOM_BUTTONS = False     # actions live on the menu bar + toolbar
    # Wider right panel by default so the detection-params table (Param/Value/Base
    # + scrollbar) fits without dragging the splitter (base default 230 is too
    # narrow and _reflow_panes uses it as the floor).
    ITEM_PANEL_WIDTH = 360

    # Apply-from-folder (--apply): on close, write dust_results.txt from the FINAL
    # spot set per frame (detected − false_positives + missed_dust + missed_strokes)
    # so the Lua side can heal the user's annotated set into the XMPs.
    apply_mode = False

    # Edit-existing-retouch / import-GT (--import-gt): seed each frame's existing
    # film-dust retouch (decoded from its source XMP) as missed_dust/missed_strokes
    # annotations over the fresh detection, and on close report which frames the
    # user actually changed (import_changed.txt) so Lua only re-writes those.
    import_gt = False

    # ------------------------------------------------------------------
    # Session / annotation lifecycle
    # ------------------------------------------------------------------

    def load_session(self, session_dir):
        import detect_dust
        self._session_dir = session_dir
        # Import-GT: write seed {stem}_annotations.json (+ import_baseline.json)
        # BEFORE the run-mode detection streams in — _poll_bg_detect then loads
        # them via _load_existing_annotations_for as each frame finalizes.
        # Baseline for the "did the user actually edit?" check is NOT the decoded
        # import — it's the INITIAL committed set (fresh detection + seeded imports),
        # captured per frame in _poll_bg_detect after detection lands. Otherwise a
        # frame where detection merely ADDS spots (which the user didn't touch)
        # would be flagged changed and get re-applied on a no-op close.
        self._import_baseline = {}
        if self.import_gt:
            try:
                import import_retouch
                seeded = import_retouch.seed_import_annotations(session_dir)
                print(f"Import-GT: seeded {len(seeded)} frame(s) from existing retouch")
            except Exception as e:   # pragma: no cover - defensive
                print(f"Import-GT seeding failed: {e}")
        images, constants = detect_dust.load_debug_spots_dir(session_dir)
        self._run_mode = False
        # A `--review` session dir starts with ONLY review_meta.json (no frames):
        # the first roll is built in-window at startup, like a roll switch. Allow
        # the empty session so the window opens immediately (don't treat as run mode).
        if not images and (Path(session_dir) / "review_meta.json").exists():
            self.ALLOW_EMPTY_SESSION = True
            self.sensor_mode = False
            return [], constants
        if not images:
            # No precomputed debug_spots.json — the export dir / roll folder has
            # only JPGs. Build minimal img_dicts and let the UI RUN DETECTION ITSELF
            # (background pool, streaming + progress overlay) so the window opens
            # immediately instead of after the whole batch. Drives the darktable
            # Debug action, the preset combo, and `run_calibration.py --review`.
            images = self._images_from_dir(session_dir)
            self._run_mode = bool(images)
        # Resolve each frame's source JPG when the stored image_path is dead (a
        # saved ground-truth folder: the temp path is gone and the roll's JPGs
        # live in the session dir or one level up at the roll root). Keeps display,
        # re-detect and missed-dust source recommendation working from a folder.
        sd = Path(session_dir)
        for img in images:
            p = img.get("image_path")
            if p and os.path.isfile(p):
                continue
            stem = img.get("stem")
            for cand in (sd / f"{stem}.jpg", sd / f"{stem}.jpeg",
                         sd.parent / f"{stem}.jpg", sd.parent / f"{stem}.jpeg"):
                if cand.is_file():
                    img["image_path"] = str(cand)
                    break
        # Sensor-dust sessions reuse this UI wholesale (sensor spots are dot
        # spots in the same dict format); only title and report wording differ.
        self.sensor_mode = bool(images and images[0].get("mode") == "sensor")
        return images, constants

    def _window_base_title(self):
        # Sensor-dust sessions reuse this UI wholesale; only the title + report
        # wording differ. The mode prefix (Calib. review / Correction) is added
        # on top by the base.
        if getattr(self, "sensor_mode", False):
            return "Sensor Dust Debug UI"
        return self.WINDOW_TITLE

    # ------------------------------------------------------------------
    # Apply-from-folder: final spot set + dust_results.txt on close
    # ------------------------------------------------------------------

    def _final_spots_for_apply(self, img_dict):
        """The healing spot set to APPLY for this frame: detected spots minus
        false_positives (with the user's radius / source / path overrides) PLUS
        hand-added missed_dust and missed_strokes turned into healable spots."""
        import detect_dust
        stem = img_dict["stem"]
        ann = self.annotations.get(stem) or {}
        detected = img_dict.get("detected") or []
        fp = ann.get("false_positives") or set()
        radius_ov = ann.get("radius_overrides") or {}
        source_ov = ann.get("source_overrides") or {}
        path_ov = ann.get("path_overrides") or {}

        spots = []
        for i, s in enumerate(detected):
            if i in fp:
                continue
            sp = dict(s)
            if i in radius_ov:
                sp["brush_radius_px"] = float(radius_ov[i])
            if i in source_ov:
                sp["src_cx"], sp["src_cy"] = float(source_ov[i][0]), float(source_ov[i][1])
            if sp.get("kind") == "stroke" and i in path_ov:
                sp["path"] = [[float(p[0]), float(p[1])] for p in path_ov[i]]
            spots.append(sp)

        buffers = self._source_buffers_for(img_dict)
        for md in ann.get("missed_dust") or []:
            spots.append(detect_dust.missed_dust_to_spot(md, detected, buffers))

        min_dim = min(img_dict.get("width") or 0, img_dict.get("height") or 0)
        for ms in ann.get("missed_strokes") or []:
            sp = detect_dust.missed_stroke_to_spot(ms, min_dim)
            if sp:
                spots.append(sp)
        return spots

    def _write_apply_results(self):
        """Write dust_results.txt for the Lua apply step: the FINAL spot set per
        frame healed via generate_xmp_data_for_spots (using the folder's
        transform_params.txt for flip/crop/ashift). Mirrors write_dust_results'
        results tuple shape."""
        import detect_dust
        session_dir = getattr(self, "_session_dir", None) or self.session_dir
        transforms = detect_dust.parse_transform_params(
            os.path.join(session_dir, "transform_params.txt"))
        results = []
        changed = []          # import-GT: stems whose committed set differs from import
        baseline = getattr(self, "_import_baseline", None) or {}
        for img_dict in self.images:
            stem = img_dict["stem"]
            w, h = img_dict.get("width"), img_dict.get("height")
            spots = self._final_spots_for_apply(img_dict)
            xmp_data = None
            if spots and w and h:
                t = transforms.get(stem, {"flip": 0, "crop": (0.0, 0.0, 1.0, 1.0),
                                          "ashift": None})
                xmp_data = detect_dust.generate_xmp_data_for_spots(
                    spots, w, h, flip=t["flip"], crop=t["crop"],
                    ashift_params=t.get("ashift"))
            results.append((stem, spots, img_dict.get("rejected") or [],
                            (w, h), None, xmp_data))
            if self.import_gt:
                import import_retouch
                min_dim = min(w or 0, h or 0)
                if import_retouch.spots_differ(baseline.get(stem), spots, min_dim):
                    changed.append(stem)
        results.sort(key=lambda r: r[0])
        path = detect_dust.write_dust_results(results, session_dir)
        print(f"Apply results written to: {path} "
              f"({sum(1 for r in results if r[1])} frame(s) with spots)")
        if self.import_gt:
            # Lua re-writes (disable prior dust instances + add new) ONLY these.
            with open(os.path.join(session_dir, "import_changed.txt"), "w") as f:
                for stem in sorted(changed):
                    f.write(stem + "\n")
            print(f"Import-GT: {len(changed)} frame(s) changed vs imported retouch")

    def _on_close(self):
        if self.apply_mode:
            try:
                self._write_apply_results()
            except Exception as e:   # pragma: no cover - defensive
                print(f"Failed to write apply results: {e}")
        super()._on_close()

    def _images_from_dir(self, session_dir):
        """Minimal img_dicts from a directory of JPGs (no debug_spots.json). Each is
        flagged `_detect_pending` so detection runs lazily when the frame is shown."""
        from PIL import Image
        out = []
        d = Path(session_dir)
        for p in sorted(d.glob("*.jpg")) + sorted(d.glob("*.jpeg")):
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except Exception:
                continue
            out.append({"stem": p.stem, "image_path": str(p),
                        "width": int(w), "height": int(h),
                        "detected": [], "rejected": [], "_detect_pending": True})
        return out

    def new_annotation_state(self, img_dict):
        return {
            "false_positives": set(),
            "missed_dust": [],
            "source_overrides": {},   # {spot_idx: (src_cx, src_cy)}
            "radius_overrides": {},   # {spot_idx: corrected_radius_px}
            "path_overrides": {},     # {spot_idx: [[x,y],...]} edited stroke centerlines
            "missed_strokes": [],     # [{"path":[[x,y],...], "stroke_width_px":w}] hand-drawn threads
            "spot_notes": {},         # {spot_idx: str} user text notes on detected spots
            "source_mismatches": [],  # list from quality test diff sessions
            "radius_mismatches": [],  # list from quality test diff sessions
            "param_overrides": {},    # {TUNING_NAME: value} per-frame detection-param edits
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

        # Serialize spot_notes as list of {spot_idx, cx, cy, note} — coords
        # included so the notes stay interpretable without the session JSON
        spot_notes_list = []
        for k, v in sorted(ann.get("spot_notes", {}).items()):
            entry = {"spot_idx": k, "note": v}
            if k < len(detected):
                entry["cx"] = detected[k]["cx"]
                entry["cy"] = detected[k]["cy"]
            spot_notes_list.append(entry)

        return {
            "stem": stem,
            "false_positives": fp_list,
            "missed_dust": ann["missed_dust"],
            "missed_strokes": ann.get("missed_strokes", []),
            "source_overrides": src_overrides_list,
            "radius_overrides": radius_overrides_list,
            "path_overrides": path_overrides_list,
            "spot_notes": spot_notes_list,
            "source_mismatches": ann.get("source_mismatches", []),
            "radius_mismatches": ann.get("radius_mismatches", []),
            "param_overrides": _params_json_safe(ann.get("param_overrides", {})),
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
        # Restore spot notes: list of {spot_idx, note}
        spot_notes = {}
        for entry in data.get("spot_notes", []):
            try:
                spot_notes[int(entry["spot_idx"])] = str(entry["note"])
            except (KeyError, ValueError, TypeError):
                pass
        self.annotations[stem]["spot_notes"] = spot_notes
        # Load source/radius mismatches from quality test diff sessions
        self.annotations[stem]["source_mismatches"] = data.get("source_mismatches", [])
        self.annotations[stem]["radius_mismatches"] = data.get("radius_mismatches", [])
        # Per-frame detection-param overrides (coerced to their tuning types).
        overrides = {}
        for name, val in (data.get("param_overrides") or {}).items():
            try:
                overrides[name] = _coerce_param(name, val)
            except (ValueError, TypeError):
                pass
        self.annotations[stem]["param_overrides"] = overrides

    # ------------------------------------------------------------------
    # Selection state
    # ------------------------------------------------------------------

    # Calibration-review source cycle order (R steps through these): the session's
    # FITTED detection -> the user's GT annotation -> the LIVE preset's detection.
    REVIEW_CYCLE = ("fitted", "gt", "live")

    def init_selection_state(self):
        self.selected_detected = set()   # set of int (detected spot indices)
        self.selected_rejected = set()   # set of int (rejected candidate indices)
        self.selected_missed = set()     # set of int (indices into missed_dust list)
        self.selected_source = None      # int or None: spot index whose source is selected
        self.selected_keypoint = None    # (spot_idx, kp_idx) or None: selected stroke key point
        self.selected_missed_stroke = None  # int or None: index into missed_strokes
        self.selected_missed_source = None  # int or None: missed-dust index whose source is selected
        self._missed_src_drag = None        # int or None: missed-dust whose source is being dragged
        self.selected_missed_stroke_source = None  # int or None: missed-thread whose source is selected
        self._missed_stroke_src_drag = None        # int or None: missed-thread source being dragged
        # Lazily-computed {stem: (img_lab, L_f32, local_std, w, h)} for recommending
        # a healing source for a hand-added missed dust (prepare_source_buffers).
        self._missed_buffers = {}

        # Visibility toggles — now View-menu checkbuttons (were left-panel
        # checkboxes). Created here (before the UI is built) so the menu can bind
        # to them; the BooleanVars need a root, which exists by now.
        self.show_rejected_var = tk.BooleanVar(master=self.root, value=False)
        self.show_source_brush_var = tk.BooleanVar(master=self.root, value=False)
        # Show/hide the two object GROUPS: algo-detected spots vs the ones loaded
        # from an existing edit (imported/hand-added missed dust + threads). Both
        # on by default; toggled from the View menu + toolbar (no keyboard shortcut
        # so nothing already bound is clobbered).
        self.show_detected_var = tk.BooleanVar(master=self.root, value=True)
        self.show_missed_var = tk.BooleanVar(master=self.root, value=True)

        # Thread-draw mode: click to place centerline points for a missed thread
        self.thread_draw_mode = False
        self.thread_draw_points = []     # list of [x,y] image coords (in-progress)

        # Calibration review (run_calibration.py --review): each frame carries a
        # review={"fitted":{detected,rejected}, "gt"?:{…}, "live":{…}} payload + a
        # review_kind. R CYCLES the active source in place — instant, all were
        # precomputed — exactly like auto_negadoctor. PERSISTS across frames.
        # GT = the user's annotation (corrected output); live = the currently-
        # selected preset's detection (default.json default, tracks the dropdown).
        imgs = getattr(self, "images", None) or []
        # Calibration review is driven by review_meta.json (calibration session dir
        # + roll list), present EVEN with no frames loaded — the first roll is built
        # IN-WINDOW at startup (same path as a roll switch), so the UI opens at once
        # instead of blocking on the build. (Legacy pre-built sessions also carry
        # per-frame `review` payloads.)
        self._review_session_dir = None
        self._review_rolls = []
        self._review_current_roll = None
        self._run_calib_mod = None
        meta = None
        try:
            meta = json.loads((Path(self.session_dir)
                               / "review_meta.json").read_text())
        except Exception:
            meta = None
        if meta:
            self._review_session_dir = meta.get("session_dir")
            self._review_rolls = list(meta.get("rolls") or [])
            self._review_current_roll = meta.get("current_roll")
        self.review_mode = (any(im.get("review") for im in imgs)
                            or bool(meta and self._review_rolls))
        self.review_source = "fitted"    # one of REVIEW_CYCLE
        # Which preset the "live" R-source reflects (the 'Detect with:' dropdown);
        # default.json is the precomputed default. Per-frame live is recomputed
        # lazily under this preset on navigation (detection is full-res + slow).
        self._review_live_preset = "(live default)"
        if self.review_mode:
            # Snapshot the precomputed (default.json) live so the '(live default)'
            # dropdown entry can restore it after a preset redetect overwrites it.
            for im in imgs:
                rev = im.get("review")
                if rev and rev.get("live") is not None and "live_default" not in rev:
                    rev["live_default"] = copy.deepcopy(rev["live"])
                    rev["live_preset"] = "(live default)"
            self._apply_review_source()
            # roll_var always exists in review (the dropdown WIDGET is only built
            # for >1 roll), so install/switch code can set it without a guard.
            self.roll_var = tk.StringVar(
                value=(self._review_current_roll
                       or (self._review_rolls[0] if self._review_rolls else "")))

    def reset_selection(self):
        self.selected_detected = set()
        self.selected_rejected = set()
        self.selected_missed = set()
        self.selected_source = None
        self.selected_keypoint = None
        self.selected_missed_stroke = None
        self.selected_missed_source = None
        self.selected_missed_stroke_source = None
        self.remove_missed_btn.config(state=tk.DISABLED)

    def __init__(self, root, session_dir):
        super().__init__(root, session_dir)
        # Give the (expandable) params table the majority of the right panel: keep
        # the spots tree at a compact baseline and stop its frame from claiming the
        # extra vertical space, so the params table grows into it instead.
        if hasattr(self, "item_tree"):
            try:
                self.item_tree.configure(height=8)
                self.item_tree.master.pack_configure(expand=False, fill=tk.X)
            except tk.TclError:
                pass
        # Run mode (the Lua Debug action / --review launch the UI on a JPG-only dir):
        # the window is already up; detect the whole roll on a background pool and
        # stream each frame's spots in as it finishes, with the shared progress
        # overlay — the same UI-first feel as auto_negadoctor.
        if getattr(self, "_run_mode", False) and root.winfo_exists():
            self.root.after(200, self._start_background_detect)
        # Calibration review. If opened EMPTY (meta-only), build the first roll
        # IN-WINDOW now (same path as a roll switch) so the window appears at once
        # instead of blocking before launch; prefetch of the others chains on once
        # it lands. If frames were pre-built (legacy), seed the cache + prefetch.
        if getattr(self, "review_mode", False) and root.winfo_exists():
            if not self.images:
                self.root.after(200, self._start_review_first_roll)
            elif len(getattr(self, "_review_rolls", []) or []) > 1:
                self.root.after(1500, self._seed_roll_cache)

    def reset_for_new_image(self):
        self.thread_draw_mode = False
        self.thread_draw_points = []
        self.reset_selection()
        # A pending frame only exists in run mode (set by _images_from_dir); the
        # background pass — already running, or scheduled right after __init__ —
        # detects every frame, so just show the overlay here until this one lands
        # (_poll_bg_detect reveals it). Do NOT kick a separate single detect, or the
        # frame would be detected twice (once here, once by the pool).
        img = self.images[self.current_idx] if self.images else None
        if img and img.get("_detect_pending"):
            tail = (f"  ({self._bg_done}/{self._bg_total})"
                    if getattr(self, "_bg_total", 0) else "")
            self._show_canvas_message(f"Detecting {img['stem']} …{tail}")
            if getattr(self, "_bg_total", 0):
                self._set_analyzing_pct(100.0 * self._bg_done / self._bg_total)
        elif getattr(self, "_analyzing", False):
            self._clear_canvas_message()
        # In a review session, refresh this frame's 'live' under the selected
        # preset (lazy — detection is full-res + slow) when 'live' is on screen.
        if getattr(self, "review_mode", False):
            self.root.after_idle(self._ensure_live_for_current)
        # Show THIS frame's param overrides in the params table.
        if hasattr(self, "params_tree"):
            self._populate_params_table()

    # ------------------------------------------------------------------
    # Run-mode background detection (detect the whole roll, streaming)
    # ------------------------------------------------------------------

    def _ml_model_path(self):
        return os.environ.get("RETOUCH_ML_MODEL") or None

    def _start_background_detect(self):
        """Detect every frame on a small background pool, streaming each result into
        the view as it lands (progress via the shared canvas overlay)."""
        if not self.images or getattr(self, "_bg_active", False):
            return
        import detect_dust
        self._bg_active = True
        self._bg_done = 0
        self._bg_total = len(self.images)
        # full-res detection has no sub-frame progress, so with few frames the
        # per-frame "done" target jumps coarsely (1 frame: 0 -> 100) and the bar
        # just sat at the animator's creep cap (~target+12 → 12% / 29%) until the
        # burst finished. Credit the IN-FLIGHT frames too (a pool worker is busy on
        # them) so the target climbs as detection proceeds. Bounded by the pool.
        self._bg_workers = max(1, int(os.environ.get("RETOUCH_UI_WORKERS", "3")))
        self._bg_q = queue.Queue()
        cfg = (self._cfg_for_selection() if hasattr(self, "_preset_var")
               else detect_dust.DEFAULT_TUNING)
        ml = self._ml_model_path()
        self._update_detect_progress()
        if self.images[self.current_idx].get("_detect_pending"):
            self._show_canvas_message(f"Detecting roll…  (0/{self._bg_total})")
        items = [(i, d.get("image_path")) for i, d in enumerate(self.images)]

        def worker():
            from concurrent.futures import ThreadPoolExecutor, as_completed
            # full-res detection is memory-heavy — a small pool (override via
            # RETOUCH_UI_WORKERS) streams results without OOM. detect() releases the
            # GIL in cv2/numpy, so threads give real parallelism.
            n = self._bg_workers

            def one(it):
                i, p = it
                if not p:
                    return (i, None, "no image path")
                try:
                    spots, _r, err, _l = detect_dust.detect(p, ml_model_path=ml, cfg=cfg)
                    return (i, spots or [], err)
                except Exception as ex:   # noqa: BLE001
                    return (i, None, str(ex))

            with ThreadPoolExecutor(max_workers=n) as ex:
                for fut in as_completed([ex.submit(one, it) for it in items]):
                    self._bg_q.put(fut.result())
            self._bg_q.put(None)   # sentinel

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(120, self._poll_bg_detect)

    def _poll_bg_detect(self):
        try:
            while True:
                item = self._bg_q.get_nowait()
                if item is None:
                    self._bg_active = False
                    self._update_detect_progress(done=True)
                    # Persist {stem}_debug_spots.json so this session reloads
                    # instantly next time (no re-detection) — apply-from-folder
                    # relies on it; the UI-first streaming flow had dropped it.
                    self._persist_debug_spots()
                    if not self.images[self.current_idx].get("_detect_pending"):
                        self._clear_canvas_message()
                    return
                i, spots, err = item
                if spots is not None and 0 <= i < len(self.images):
                    img = self.images[i]
                    img["detected"] = spots
                    img["_detect_pending"] = False
                    # Re-seed this frame's annotations now that its detected spots
                    # exist: a clean state, then re-load any saved
                    # {stem}_annotations.json so FPs / overrides (stored by coords /
                    # index) re-match against the fresh detection. Fresh Debug flow
                    # has no file → stays empty; a re-opened / apply-from-folder
                    # session RESTORES the user's annotations instead of wiping them.
                    self.annotations[img["stem"]] = self.new_annotation_state(img)
                    self._load_existing_annotations_for(img)
                    if getattr(self, "import_gt", False):
                        # Snapshot the INITIAL committed set (detection + seeded
                        # imports, no user edits yet) so on close we re-apply ONLY
                        # frames the user actually changed — a no-op close leaves
                        # the frame's existing retouch untouched.
                        try:
                            self._import_baseline[img["stem"]] = \
                                self._final_spots_for_apply(img)
                        except Exception:   # pragma: no cover - defensive
                            pass
                self._bg_done += 1
                self._update_detect_progress()
                if i == self.current_idx:
                    # reveal the current frame's spots as they land
                    self._clear_canvas_message()
                    self.reset_selection()
                    self._populate_items_list()
                    self._redraw()
                    self.update_counts()
        except queue.Empty:
            pass
        if getattr(self, "_bg_active", False):
            self.root.after(120, self._poll_bg_detect)

    def _persist_debug_spots(self):
        """Write {stem}_debug_spots.json for every detected frame so a saved
        session reloads WITHOUT re-detecting (restores the persistence the
        UI-first streaming flow dropped — apply-from-folder reads it). Only runs
        on the run-mode completion path, so a folder that already had debug_spots
        (e.g. a committed fixture being reviewed) is never re-detected and never
        overwritten."""
        try:
            import detect_dust
            session_dir = getattr(self, "_session_dir", None) or self.session_dir
            results, paths = [], {}
            for img in self.images:
                if img.get("_detect_pending"):
                    continue
                stem = img["stem"]
                results.append((stem, img.get("detected") or [],
                                img.get("rejected") or [],
                                (img.get("width"), img.get("height")), None, None))
                if img.get("image_path"):
                    paths[stem] = img["image_path"]
            if results:
                detect_dust.write_debug_spots_json(
                    results, paths, session_dir,
                    mode="sensor" if getattr(self, "sensor_mode", False) else None)
        except Exception as e:   # pragma: no cover - defensive
            print(f"Could not persist debug spots: {e}")

    def _update_detect_progress(self, done=False):
        if hasattr(self, "_detect_status"):
            self._detect_status.config(
                text="" if done else f"detecting {self._bg_done}/{self._bg_total}…")
        if getattr(self, "_analyzing", False):
            total = max(1, self._bg_total)
            if done:
                frac = 1.0
            else:
                # done frames + half-credit for the frames currently in flight on
                # the pool, so a single long detect still visibly advances (and the
                # target stays monotonic as done overtakes in-flight).
                inflight = min(getattr(self, "_bg_workers", 1),
                               total - self._bg_done)
                frac = (self._bg_done + 0.5 * inflight) / total
            self._set_analyzing_pct(100.0 * frac)

    # ------------------------------------------------------------------
    # Preset dropdown + on-demand re-detection (calibration preview / --review)
    # ------------------------------------------------------------------

    def build_feature_toolbar(self, parent, row):
        """Feature toolbar after the common nav/zoom/fit (base build_toolbar):
        the FP/Missed counts (left) and, right-aligned, the calibration-review
        fitted/live toggle + the 'Detect with:' preset combo. Selecting a preset
        re-runs detection on the CURRENT frame under that preset's Tuning
        (background thread) and swaps its spots — the same in-UI preview
        auto_negadoctor's preset combo gives. A `(fitted — review)` entry appears
        when launched by `run_calibration.py --review` (the fitted preset arrives
        via RETOUCH_REVIEW_PRESET)."""
        self.toolbar_separator(row)
        # Group show/hide toggles (buttons show the CURRENT state, not the target).
        self.detected_toggle_btn = self.toolbar_button(
            row, "", self._toggle_show_detected)
        self.attach_tooltip(self.detected_toggle_btn,
                            "Show/hide algorithm-detected spots")
        self.missed_toggle_btn = self.toolbar_button(
            row, "", self._toggle_show_missed)
        self.attach_tooltip(self.missed_toggle_btn,
                            "Show/hide spots loaded from an existing edit (imported / missed)")
        self._update_group_toggle_btns()
        self.toolbar_separator(row)
        # Annotation counts (was the label atop the old bottom button column).
        self.count_label = ttk.Label(row, text="FP: 0 | Missed: 0")
        self.count_label.pack(side="left", padx=8)

        # Right-aligned: review toggle + preset combo (matches negadoctor).
        self._review_preset = self._load_review_preset_env()
        choices = ["(live default)"] + self._preset_names()
        if self._review_preset is not None:
            choices.append("(fitted — review)")
        start = "(fitted — review)" if self._review_preset is not None else choices[0]
        self._preset_var = tk.StringVar(value=start)
        if getattr(self, "review_mode", False):
            self.review_btn = tk.Button(
                row, text="Src: FITTED", command=self._toggle_review_source,
                **self.TOOLBAR_BTN_STYLE)
            self.review_btn.pack(side="right", padx=8, pady=2)
            self.attach_tooltip(
                self.review_btn,
                "Cycle the calibration-review source: fitted / GT / live (R)")
            self._update_review_btn()
        # Shared helper: a long preset name stays readable (entry sized to the
        # longest name + capped, drop-down list widened past the cap, full-name
        # hover tooltip).
        combo = self.make_readonly_combobox(row, self._preset_var, choices)
        combo.pack(side="right", padx=2, pady=2)
        combo.bind("<<ComboboxSelected>>", lambda e: self._on_preset_combo_changed())
        self._preset_combo = combo
        # In review the combo only drives 'live', so grey it out unless 'live' is
        # the source on screen (GT/fitted are unmovable).
        self._update_preset_combo_state()
        ttk.Label(row, text="Detect with:").pack(side="right", padx=(8, 2))
        self._detect_status = ttk.Label(row, text="")
        self._detect_status.pack(side="right", padx=8)
        # Multi-roll review: a roll dropdown rebuilds the chosen roll's
        # fitted/live/GT review in-window. The button row is already crowded, so
        # give the dropdown its OWN row below it (otherwise it would squeeze the
        # 'Detect with:' combo off the right edge at the default width).
        self._build_roll_switcher(parent, row)

    # ------------------------------------------------------------------
    # Group show/hide (algo-detected vs loaded-from-existing)
    # ------------------------------------------------------------------

    def _update_group_toggle_btns(self):
        """Reflect the current visibility state on the toolbar buttons (the label
        shows what IS shown, not the toggle target — a ✓/✗ + name)."""
        if hasattr(self, "detected_toggle_btn"):
            on = self.show_detected_var.get()
            self.detected_toggle_btn.config(
                text=("✓ Detected" if on else "✗ Detected"),
                fg=("white" if on else "#9a9a9a"))
        if hasattr(self, "missed_toggle_btn"):
            on = self.show_missed_var.get()
            self.missed_toggle_btn.config(
                text=("✓ Loaded" if on else "✗ Loaded"),
                fg=("white" if on else "#9a9a9a"))

    def _on_group_toggle(self):
        """A group visibility var changed (menu checkbutton): sync buttons + redraw."""
        self._update_group_toggle_btns()
        self._redraw_markers()

    def _toggle_show_detected(self):
        self.show_detected_var.set(not self.show_detected_var.get())
        self._on_group_toggle()

    def _toggle_show_missed(self):
        self.show_missed_var.set(not self.show_missed_var.get())
        self._on_group_toggle()

    # ------------------------------------------------------------------
    # Detection-params table (per-frame overrides + on-demand re-detect)
    # ------------------------------------------------------------------

    def _current_stem(self):
        return self.images[self.current_idx]["stem"] if self.images else None

    def _param_overrides(self):
        """The current frame's {NAME: value} override dict (created on demand)."""
        stem = self._current_stem()
        if stem is None:
            return {}
        ann = self.annotations.setdefault(stem, self.new_annotation_state(
            self.images[self.current_idx]))
        return ann.setdefault("param_overrides", {})

    def _effective_cfg(self):
        """Base preset (the 'Detect with:' selection) with the current frame's
        per-frame param overrides layered on top."""
        base = self._cfg_for_selection()
        ov = self._param_overrides()
        if not ov:
            return base
        coerced = {}
        for name, val in ov.items():
            try:
                coerced[name] = _coerce_param(name, val)
            except (ValueError, TypeError):
                pass
        try:
            return base._replace(**coerced)
        except (ValueError, TypeError):
            return base

    @staticmethod
    def _param_value_str(name, val):
        import tuning
        if name in tuning.BOOL_FIELDS:
            return "True" if val else "False"
        if name in tuning.TUPLE_FIELDS:
            return ", ".join(f"{float(x):g}" for x in val)
        if name in tuning.INT_FIELDS:
            return str(int(val))
        return f"{float(val):g}"

    def build_item_panel_top(self, parent):
        """Detection-params table above the spots table: every tuning constant
        (grouped kind -> sub-stage), the current value + base value, a name filter,
        and Re-detect / Reset buttons. Editing a value stores a per-frame override
        (saved in the annotations); Re-detect re-runs detection under the effective
        cfg (base preset + overrides)."""
        import tuning
        sec = tk.Frame(parent, bg="#484848")
        sec.pack(fill=tk.BOTH, expand=True, padx=4, pady=(6, 2))

        header = tk.Frame(sec, bg="#484848")
        header.pack(fill=tk.X)
        tk.Label(header, text="Detection params", bg="#484848", fg="white",
                 font=("", 10, "bold")).pack(side="left", padx=2)
        tk.Button(header, text="Re-detect", command=self._redetect_current,
                  **self.TOOLBAR_BTN_STYLE).pack(side="right", padx=2)
        tk.Button(header, text="Reset", command=self._reset_params,
                  **self.TOOLBAR_BTN_STYLE).pack(side="right", padx=2)

        filt = tk.Frame(sec, bg="#484848")
        filt.pack(fill=tk.X, pady=(2, 2))
        tk.Label(filt, text="filter:", bg="#484848", fg="#c0c0c0",
                 font=("", 8)).pack(side="left", padx=(2, 2))
        self._param_filter = tk.StringVar(value="")
        fe = tk.Entry(filt, textvariable=self._param_filter, bg="#363636",
                      fg="#dddddd", insertbackground="#dddddd", relief="flat")
        fe.pack(side="left", fill=tk.X, expand=True, padx=(0, 2))
        self._param_filter.trace_add("write", lambda *a: self._populate_params_table())

        tree_frame = tk.Frame(sec, bg="#484848")
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 2))
        self.params_tree = ttk.Treeview(
            tree_frame, columns=("val", "base"), show="tree headings",
            style="Item.Treeview", selectmode="browse", height=16)
        self.params_tree.heading("#0", text="Param")
        self.params_tree.heading("val", text="Value")
        self.params_tree.heading("base", text="Base")
        self.params_tree.column("#0", width=self.scaled(150), stretch=True, minwidth=self.scaled(90))
        self.params_tree.column("val", width=self.scaled(70), anchor="e", stretch=False)
        self.params_tree.column("base", width=self.scaled(60), anchor="e", stretch=False)
        self.params_tree.tag_configure("changed", foreground="#ffd24a")
        self.params_tree.tag_configure("group", foreground="#9fd0ff")
        sb = tk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.params_tree.yview)
        self.params_tree.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.params_tree.pack(fill=tk.BOTH, expand=True)
        self.params_tree.bind("<Double-1>", self._on_param_double_click)
        self._install_param_tooltip()

        self._populate_params_table()

    def _install_param_tooltip(self):
        """Hover tooltip on the params tree: show the param's description (from
        tuning.FIELDS[name].doc) for whatever leaf row is under the cursor,
        updating as the pointer moves between rows."""
        tree = self.params_tree
        state = {"tip": None, "row": None}

        def hide(_e=None):
            if state["tip"] is not None:
                state["tip"].destroy()
                state["tip"] = None
            state["row"] = None

        def on_motion(e):
            import tuning
            iid = tree.identify_row(e.y)
            if not iid or not iid.startswith("p::"):
                hide()
                return
            if iid == state["row"]:
                return
            hide()
            name = iid[3:]
            f = tuning.FIELDS.get(name)
            doc = (getattr(f, "doc", "") or "").strip() if f else ""
            text = f"{name}\n{doc}" if doc else name
            w = tk.Toplevel(tree)
            w.wm_overrideredirect(True)
            w.withdraw()   # hide until measured + positioned (no top-left flash)
            tk.Label(w, text=text, bg="#ffffe0", fg="#000000", font=("", 8),
                     relief=tk.SOLID, borderwidth=1, padx=6, pady=4,
                     justify=tk.LEFT, wraplength=self.scaled(420)).pack()
            # Position it, then keep it fully on-screen: place to the left of the
            # cursor if it would overflow the right edge, and above if it would
            # overflow the bottom (never off the top/left either).
            w.update_idletasks()
            tw, th = w.winfo_reqwidth(), w.winfo_reqheight()
            sw, sh = w.winfo_screenwidth(), w.winfo_screenheight()
            margin = self.scaled(8)
            x = e.x_root + self.scaled(14)
            if x + tw + margin > sw:
                x = e.x_root - tw - self.scaled(14)
            x = max(margin, min(x, sw - tw - margin))
            y = e.y_root + self.scaled(12)
            if y + th + margin > sh:
                y = e.y_root - th - self.scaled(12)
            y = max(margin, min(y, sh - th - margin))
            w.wm_geometry(f"+{int(x)}+{int(y)}")
            w.deiconify()
            state["tip"] = w
            state["row"] = iid

        tree.bind("<Motion>", on_motion, add="+")
        tree.bind("<Leave>", hide, add="+")
        tree.bind("<ButtonPress>", hide, add="+")

    def _populate_params_table(self):
        import tuning
        tree = getattr(self, "params_tree", None)
        if tree is None:
            return
        needle = (self._param_filter.get() or "").strip().lower()
        # Preserve each group's expand/collapse state across the rebuild — an edit
        # (or nav / re-detect) must NOT re-expand groups the user collapsed. While
        # a filter is active, force groups open so matches stay visible.
        prev_open = {}
        for kiid in tree.get_children(""):
            prev_open[kiid] = bool(tree.item(kiid, "open"))
            for siid in tree.get_children(kiid):
                prev_open[siid] = bool(tree.item(siid, "open"))

        def _open(iid):
            return True if needle else prev_open.get(iid, True)

        for iid in tree.get_children(""):
            tree.delete(iid)
        base = self._cfg_for_selection()
        ov = self._param_overrides()
        for kind, substages in tuning.GROUPS.items():
            kind_iid = f"g::{kind}"
            kind_added = False
            for sub, names in substages.items():
                sub_iid = f"g::{kind}::{sub}"
                sub_added = False
                for name in names:
                    if needle and needle not in name.lower():
                        continue
                    if not kind_added:
                        tree.insert("", tk.END, iid=kind_iid, text=kind,
                                    open=_open(kind_iid), tags=("group",))
                        kind_added = True
                    if not sub_added:
                        tree.insert(kind_iid, tk.END, iid=sub_iid, text=sub,
                                    open=_open(sub_iid), tags=("group",))
                        sub_added = True
                    base_val = getattr(base, name)
                    eff_val = ov[name] if name in ov else base_val
                    changed = name in ov
                    tree.insert(sub_iid, tk.END, iid=f"p::{name}", text=name,
                                values=(self._param_value_str(name, eff_val),
                                        self._param_value_str(name, base_val)),
                                tags=(("changed",) if changed else ()))

    def _on_param_double_click(self, event):
        import tuning
        tree = self.params_tree
        iid = tree.identify_row(event.y)
        if not iid or not iid.startswith("p::"):
            return
        name = iid[3:]
        base_val = getattr(self._cfg_for_selection(), name)
        ov = self._param_overrides()
        cur = ov.get(name, base_val)
        # Bool params toggle in place; others open an inline entry on the value cell.
        if name in tuning.BOOL_FIELDS:
            self._commit_param(name, not bool(cur))
            return
        col = tree.identify_column(event.x)
        if col not in ("#1",):   # only the Value column is editable
            return
        bbox = tree.bbox(iid, "val")
        if not bbox:
            return
        x, y, w, h = bbox
        entry = tk.Entry(tree, bg="#2b2b2b", fg="#ffffff", insertbackground="#ffffff",
                         relief="solid", borderwidth=1)
        entry.insert(0, self._param_value_str(name, cur))
        entry.select_range(0, tk.END)
        entry.place(x=x, y=y, width=w, height=h)
        entry.focus_set()

        def commit(_e=None):
            text = entry.get()
            entry.destroy()
            try:
                self._commit_param(name, _coerce_param(name, text))
            except (ValueError, TypeError):
                pass   # keep the old value on a bad parse
        entry.bind("<Return>", commit)
        entry.bind("<FocusOut>", commit)
        entry.bind("<Escape>", lambda e: entry.destroy())

    def _commit_param(self, name, value):
        """Store (or clear, if back to base) a per-frame param override + persist."""
        stem = self._current_stem()
        if stem is None:
            return
        base_val = getattr(self._cfg_for_selection(), name)
        ov = self._param_overrides()
        if self._param_value_str(name, value) == self._param_value_str(name, base_val):
            ov.pop(name, None)          # equals base -> not an override
        else:
            ov[name] = value
        self._auto_save(stem)
        self._populate_params_table()

    def _reset_params(self):
        stem = self._current_stem()
        if stem is None:
            return
        self.annotations[stem]["param_overrides"] = {}
        self._auto_save(stem)
        self._populate_params_table()

    # ------------------------------------------------------------------
    # Calibration-review roll switcher: show a DIFFERENT roll's fitted/live/GT
    # review in-window. Each roll's review is built (detection under the fitted
    # constants + the live preset) at most ONCE per session and kept in an
    # in-memory cache; the rolls the user hasn't opened are PREFETCHED on a
    # background thread after the window settles, so a switch is usually instant.
    # `_roll_cache[roll]` is (images, constants), or None for a roll with no images.
    # ------------------------------------------------------------------

    def _build_roll_switcher(self, parent, row):
        """A 'Roll:' dropdown for a multi-roll review session, on its own toolbar
        row (the button row has no spare width)."""
        if not (getattr(self, "review_mode", False)
                and len(getattr(self, "_review_rolls", []) or []) > 1):
            return
        rr = ttk.Frame(parent)
        rr.pack(side="top", fill="x")
        self.roll_combo = self.make_readonly_combobox(
            rr, self.roll_var, list(self._review_rolls))
        self.roll_combo.pack(side="right", padx=2, pady=2)
        self.roll_combo.bind("<<ComboboxSelected>>",
                             lambda e: self._on_roll_selected())
        ttk.Label(rr, text="Roll:").pack(side="right", padx=(8, 2))
        self.attach_tooltip(self.roll_combo,
                            "Switch which roll of this calibration session you "
                            "review; built once then cached / prefetched")

    def _import_run_calibration(self):
        """Lazily import the feature's tests/run_calibration.py (where the review
        build lives) under a private module name. Cached after the first switch."""
        if getattr(self, "_run_calib_mod", None) is not None:
            return self._run_calib_mod
        import importlib.util
        path = Path(__file__).resolve().parent / "tests" / "run_calibration.py"
        spec = importlib.util.spec_from_file_location("retouch_run_calibration", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._run_calib_mod = mod
        return mod

    def _start_review_first_roll(self):
        """Empty (meta-only) review startup: build the CURRENT roll in-window now,
        exactly like a roll switch — the window is already up, so the user sees the
        progress overlay instead of waiting before launch. Prefetch of the others
        chains on once this one lands (_poll_roll_build)."""
        if hasattr(self, "_roll_cache"):
            return
        self._roll_cache = {}
        self._roll_build_active = False
        self._roll_wait_target = self._review_current_roll
        if hasattr(self, "roll_combo"):
            self.roll_combo.config(state="disabled")
        self._show_canvas_message(f"Detecting roll {self._review_current_roll} …")
        self._kick_prefetch()

    def _seed_roll_cache(self):
        """Seed the cache with the currently-loaded roll and start prefetching the
        rest in the background (called once, after the initial roll is shown)."""
        if hasattr(self, "_roll_cache"):
            return                                   # already seeded
        self._roll_cache = {}
        self._roll_build_active = False
        self._roll_wait_target = None
        if self._review_current_roll:
            self._roll_cache[self._review_current_roll] = (
                copy.deepcopy(self.images), copy.deepcopy(self.constants))
        self._kick_prefetch()

    def _on_roll_selected(self):
        self.roll_combo.selection_clear()
        self.root.after_idle(lambda: self.canvas.focus_set())
        if not hasattr(self, "_roll_cache"):     # clicked before the prefetch seed
            self._seed_roll_cache()
        if self._roll_wait_target is not None:   # a switch is already in flight
            self.roll_var.set(self._roll_wait_target)
            return
        name = self.roll_var.get()
        if name == self._review_current_roll:
            return
        self._switch_to_roll(name)

    def _switch_to_roll(self, roll_id):
        entry = self._roll_cache.get(roll_id, "MISS")
        if isinstance(entry, tuple):                 # already built -> instant
            self._install_roll(roll_id)
            self._kick_prefetch()
            return
        if entry is None:                            # known to have no images
            self.roll_var.set(self._review_current_roll)
            from tkinter import messagebox
            messagebox.showerror("Roll unavailable",
                                 f"Roll '{roll_id}' has no images.",
                                 parent=self.root)
            return
        self._roll_wait_target = roll_id
        self.roll_var.set(roll_id)
        self.roll_combo.config(state="disabled")
        self._show_canvas_message(f"Detecting roll {roll_id} …")
        self._kick_prefetch()

    def _kick_prefetch(self):
        if not getattr(self, "_review_rolls", None):
            return
        if getattr(self, "_roll_build_active", False):
            return
        wt = self._roll_wait_target
        nxt = wt if (wt is not None and wt not in self._roll_cache) else None
        if nxt is None:
            nxt = next((r for r in self._review_rolls
                        if r not in self._roll_cache), None)
        if nxt is None:
            return
        self._roll_build_active = True
        self._roll_build_q = queue.Queue()
        threading.Thread(target=self._roll_build_worker, args=(nxt,),
                         daemon=True).start()
        self.root.after(150, self._poll_roll_build)

    def _roll_build_worker(self, roll_id):
        def progress(pct):                           # real build progress -> overlay
            self._roll_build_q.put(("progress", roll_id, pct))
        try:
            images, constants = self._build_and_load_roll(roll_id, progress=progress)
            if images:
                self._roll_build_q.put(("done", roll_id, images, constants))
            else:
                self._roll_build_q.put(("skip", roll_id))
        except Exception as e:
            self._roll_build_q.put(("error", roll_id, e))

    def _poll_roll_build(self):
        # Drain everything queued; apply progress for the roll the user is waiting
        # on (prefetch builds also report, ignored), stop at the first terminal.
        terminal = None
        try:
            while True:
                item = self._roll_build_q.get_nowait()
                if item[0] == "progress":
                    if self._roll_wait_target == item[1]:
                        self._set_analyzing_pct(item[2])
                    continue
                terminal = item
                break
        except queue.Empty:
            pass
        if terminal is None:
            self.root.after(100, self._poll_roll_build)
            return
        item = terminal
        self._roll_build_active = False
        tag = item[0]
        if tag == "done":
            self._roll_cache[item[1]] = (item[2], item[3])
        else:                                        # ("skip"/"error", roll_id[, e])
            self._roll_cache[item[1]] = None
        wt = self._roll_wait_target
        if wt is not None:
            cached = self._roll_cache.get(wt, "MISS")
            if isinstance(cached, tuple):
                self._roll_wait_target = None
                self._install_roll(wt)
            elif cached is None:
                self._roll_wait_target = None
                self._set_analyzing_pct(100.0)
                self._clear_canvas_message()
                if hasattr(self, "roll_combo"):
                    self.roll_combo.config(state="readonly")
                self.roll_var.set(self._review_current_roll)
                from tkinter import messagebox
                messagebox.showerror(
                    "Roll switch failed",
                    f"Could not build roll '{wt}'.", parent=self.root)
        self._kick_prefetch()

    def _build_and_load_roll(self, roll_id, progress=None):
        """Build roll `roll_id`'s review into a throwaway dir and load it into
        memory (frames carry absolute source-JPG paths, so the dir can go right
        after). Returns (images, constants), or (None, None) if no images. Runs on
        a background thread — no Tk access."""
        import shutil
        import tempfile
        import detect_dust
        tmp = Path(tempfile.mkdtemp(prefix="retouch_rollbuild_"))
        try:
            rc = self._import_run_calibration()
            rid = rc.build_roll_review(self._review_session_dir, roll_id, str(tmp),
                                       progress=progress)
            if rid is None:
                return None, None
            images, constants = detect_dust.load_debug_spots_dir(str(tmp))
            return images, constants
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _install_roll(self, roll_id):
        """Swap the whole image set to a cached roll (instant — no detection),
        reusing the progressive add_image path + re-seating the review state."""
        entry = self._roll_cache.get(roll_id)
        if not isinstance(entry, tuple):
            return
        images, constants = entry
        self.constants = constants or self.constants
        self.data["constants"] = self.constants
        self.reset_images()
        # Defer thumbnails to ONE sequential loader (load_thumbnails_batched);
        # add_image(render_thumb=True) would spawn a full-res decode thread PER
        # frame and freeze the main-thread render for ~seconds (the cached-switch
        # "freeze" the user saw).
        for data in copy.deepcopy(images):     # keep the cache entry pristine
            self.add_image(data, render_thumb=False)
        self._review_current_roll = roll_id
        self.roll_var.set(roll_id)
        # Re-seat the review state for the new frames (mirror init_selection_state).
        self.review_source = "fitted"
        self._review_live_preset = "(live default)"
        if hasattr(self, "_preset_var") and self._review_preset is None:
            self._preset_var.set("(live default)")
        for im in self.images:
            rev = im.get("review")
            if rev and rev.get("live") is not None and "live_default" not in rev:
                rev["live_default"] = copy.deepcopy(rev["live"])
                rev["live_preset"] = "(live default)"
        self._apply_review_source()
        self._update_preset_combo_state()   # roll switch resets to fitted -> disable
        self._set_analyzing_pct(100.0)
        self._clear_canvas_message()
        if hasattr(self, "roll_combo"):
            self.roll_combo.config(state="readonly")
        if self.images:
            self._load_image_by_idx(0)        # render frame 0 (fast, uncontested)
        self.load_thumbnails_batched()        # then fill thumbs in the background
        self._update_review_btn()
        self.update_counts()
        self.canvas.focus_set()

    # ------------------------------------------------------------------
    # Menu hooks (the base builds the generic View / Navigate / Help bar)
    # ------------------------------------------------------------------

    def extend_view_menu(self, view):
        # Toggle settings are CHECKBUTTONS named for the thing itself (not
        # "Show X") — the checkmark IS the current state.
        # The two object GROUPS (algo-detected vs loaded-from-existing).
        view.add_checkbutton(label="Detected spots (algo)",
                             variable=self.show_detected_var,
                             command=self._on_group_toggle)
        view.add_checkbutton(label="Loaded spots (imported / missed)",
                             variable=self.show_missed_var,
                             command=self._on_group_toggle)
        view.add_separator()
        view.add_checkbutton(label="Rejected candidates",
                             variable=self.show_rejected_var,
                             command=self._redraw_markers)
        view.add_checkbutton(label="Source brush",
                             variable=self.show_source_brush_var,
                             command=self._redraw_markers)
        # The calibration-review source switcher is a RADIOBUTTON submenu (the
        # bullet shows the current source) and is present ONLY in a --review
        # session (run_calibration.py --review).
        if getattr(self, "review_mode", False):
            view.add_separator()
            self._mv_review = tk.StringVar(value=self.review_source)
            rev_menu = tk.Menu(view, tearoff=0)
            rev_labels = {"fitted": "Fitted (this session)",
                          "gt": "Ground truth (annotation)", "live": "Live (preset)"}
            for src in self.REVIEW_CYCLE:
                rev_menu.add_radiobutton(
                    label=rev_labels.get(src, src), value=src,
                    variable=self._mv_review,
                    command=lambda s=src: self._set_review_source(s))
            view.add_cascade(label="Calibration review source  (R cycles)",
                             menu=rev_menu)

    def extend_help_menu(self, helpm):
        # _show_legend / _show_shortcuts (both non-modal popups) live in the base.
        helpm.add_command(label="Marker legend…", command=self._show_legend)
        helpm.add_command(label="Mouse & keyboard shortcuts…",
                          command=self._show_shortcuts)

    _SHORTCUTS_TEXT = (
        "MOUSE\n"
        "  Scroll                zoom in / out\n"
        "  Scroll (1 spot sel'd) adjust radius correction\n"
        "  Middle drag           pan\n"
        "  Click                 select marker\n"
        "  Drag                  multi-select\n"
        "  Shift+click / drag    add to selection\n"
        "  Ctrl+click            add missed dust\n"
        "  Ctrl+drag             zoom to rectangle\n\n"
        "KEYS\n"
        "  Space / B   next / previous image\n"
        "  R           rejected → missed\n"
        "              (review session: cycle fitted / GT / live)\n"
        "  T           draw missed thread (click pts, Enter=done, Esc=cancel)\n"
        "  M           mark false positive\n"
        "  C           clear FP mark\n"
        "  Del         remove missed marker\n"
        "  Note box    comment on the selected object (saves as you type)\n"
        "  H           hide / show all markers\n"
        "  P           display color management on / off (sRGB->monitor)\n"
        "  + / -       zoom ×2 / ÷2     Arrows pan (Shift = fast)\n"
        "  F           fit to window"
    )

    # ------------------------------------------------------------------
    # Calibration review: fitted vs live detection (R)
    # ------------------------------------------------------------------

    def _on_r_key(self):
        """R: in a calibration-review session flip fitted/live; otherwise the
        normal 'rejected → missed' annotation action."""
        if getattr(self, "review_mode", False):
            self._toggle_review_source()
        else:
            self._mark_rejected_as_missed()

    def _frame_review_sources(self, img):
        """The review sources available for THIS frame, in cycle order. GT is
        present only when the frame carries an annotation, so the R cycle skips it
        where it doesn't exist."""
        rev = (img or {}).get("review") or {}
        return [s for s in self.REVIEW_CYCLE if rev.get(s) is not None]

    def _apply_review_source(self):
        """Swap every review frame's detected/rejected lists to the active source
        (fitted / GT / live) IN PLACE, so all render/hit-test paths use it
        unchanged. A frame missing the active source (e.g. no GT) falls back to
        its fitted payload."""
        if not getattr(self, "review_mode", False):
            return
        for img in self.images:
            rev = img.get("review")
            if not rev:
                continue
            src = rev.get(self.review_source)
            if src is None:                 # no GT for this frame -> show fitted
                src = rev.get("fitted") or {}
            if src.get("detected") is not None:
                img["detected"] = src["detected"]
            if src.get("rejected") is not None:
                img["rejected"] = src["rejected"]

    def _toggle_review_source(self):
        """R: cycle the preview FITTED -> GT -> live -> FITTED. FITTED = this
        calibration session's detection; GT = the user's annotation (the corrected
        output); live = the currently-selected preset's detection (default.json by
        default, tracks the 'Detect with:' dropdown). Sources the current frame
        lacks (e.g. GT on an un-annotated frame) are skipped. Review sessions only."""
        if not getattr(self, "review_mode", False):
            self._update_review_btn()    # snap the menu bullet back (no-op)
            self._set_info_text("Not reviewing a calibration session "
                                "(open one via run_calibration.py --review).")
            return
        img = self.images[self.current_idx] if self.images else None
        order = self._frame_review_sources(img) or list(self.REVIEW_CYCLE)
        cur = self.review_source if self.review_source in order else order[0]
        self._set_review_source(order[(order.index(cur) + 1) % len(order)])

    def _set_review_source(self, src):
        """Switch the review preview to a specific source (fitted / GT / live);
        used by both the R-key cycle and the View-menu radiobuttons."""
        if not getattr(self, "review_mode", False):
            self._update_review_btn()    # snap the menu bullet back (no-op)
            self._set_info_text("Not reviewing a calibration session "
                                "(open one via run_calibration.py --review).")
            return
        self.review_source = src
        # The detected lists differ between sources, so spot indices change —
        # rebuild the (empty) annotation state for every frame and refresh.
        self._refresh_review_display()
        self._ensure_live_for_current()   # live under a non-default preset is lazy

    _REVIEW_BTN_STYLE = {
        "fitted": ("Src: FITTED", "#9fd0ff"),
        "gt":     ("Src: GT",     "#9be29b"),
        "live":   ("Src: live",   "#ffd080"),
    }

    def _update_preset_combo_state(self):
        """Enable/disable the 'Detect with:' combo. In a calibration-review session
        it only drives the 'live' source (GT + fitted are unmovable), so disable it
        unless 'live' is the source on screen. Outside review it's the ad-hoc
        per-frame redetect, so it stays enabled."""
        if not hasattr(self, "_preset_combo"):
            return
        live = (not getattr(self, "review_mode", False)
                or getattr(self, "review_source", "fitted") == "live")
        try:
            self._preset_combo.config(state="readonly" if live else "disabled")
        except Exception:
            pass

    def _update_review_btn(self):
        self._set_menu_var("_mv_review", self.review_source)
        if not hasattr(self, "review_btn"):
            return
        if not getattr(self, "review_mode", False):
            self.review_btn.config(text="Source —", fg="#888888",
                                   state=tk.DISABLED)
            return
        text, fg = self._REVIEW_BTN_STYLE.get(self.review_source,
                                              self._REVIEW_BTN_STYLE["fitted"])
        self.review_btn.config(text=text, fg=fg, state=tk.NORMAL)

    def _preset_names(self):
        import tuning
        d = getattr(tuning, "PRESETS_DIR", None)
        if not d or not os.path.isdir(d):
            return []
        return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json"))

    def _load_review_preset_env(self):
        raw = os.environ.get("RETOUCH_REVIEW_PRESET")
        if not raw:
            return None
        try:
            import tuning
            return tuning.from_mapping(json.loads(raw), source="<review env>")
        except Exception:
            return None

    def _cfg_for_selection(self):
        import detect_dust
        import tuning
        # The item panel (params table) is built before the toolbar's preset combo,
        # so fall back to the default preset until the combo exists.
        if not hasattr(self, "_preset_var"):
            return detect_dust.DEFAULT_TUNING
        sel = self._preset_var.get()
        if sel == "(live default)":
            return detect_dust.DEFAULT_TUNING
        if sel == "(fitted — review)" and self._review_preset is not None:
            return self._review_preset
        try:
            return tuning.load(sel)
        except Exception:
            return detect_dust.DEFAULT_TUNING

    def _on_preset_combo_changed(self):
        """'Detect with:' combo changed. In a review session the combo chooses
        which preset the 'live' R-source reflects; otherwise it's the ad-hoc
        redetect of the current frame."""
        if getattr(self, "review_mode", False):
            self._on_review_preset_changed()
        else:
            self._redetect_current()

    def _on_review_preset_changed(self):
        """Review mode: the combo selects which preset 'live' reflects. Switching
        it implies the user wants to SEE live. '(live default)' restores the
        precomputed default.json live for every frame; any other preset redetects
        the CURRENT frame under it (other frames refresh lazily on navigation)."""
        self._review_live_preset = self._preset_var.get()
        self.review_source = "live"
        if self._review_live_preset == "(live default)":
            for img in self.images:
                rev = img.get("review")
                if rev and rev.get("live_default") is not None:
                    rev["live"] = copy.deepcopy(rev["live_default"])
                    rev["live_preset"] = "(live default)"
            self._refresh_review_display()
        else:
            self._update_review_btn()
            self._redetect_current()      # _poll_detect stores into review['live']

    def _refresh_review_display(self):
        """Re-apply the active review source to every frame and rebuild the
        (index-dependent) annotation state + redraw — the shared tail used by the
        R cycle and the live-preset restore."""
        self._apply_review_source()
        for img in self.images:
            self.annotations[img["stem"]] = self.new_annotation_state(img)
        self.reset_selection()
        self._update_review_btn()
        self._update_preset_combo_state()   # combo enabled only when live shows
        self._populate_items_list()
        self._redraw()
        self.update_counts()

    def _ensure_live_for_current(self):
        """Lazily redetect the current frame's 'live' under the selected preset if
        it's stale (a non-default preset and not yet computed for this frame).
        Called when 'live' becomes/stays the shown source."""
        if not getattr(self, "review_mode", False) or not self.images:
            return
        if self.review_source != "live":
            return
        if self._review_live_preset == "(live default)":
            return
        rev = self.images[self.current_idx].get("review")
        if not rev:
            return
        if rev.get("live_preset") == self._review_live_preset:
            return       # already current for this preset
        self._redetect_current()

    def _redetect_current(self):
        """Detect the current frame under the selected preset on a background thread
        (detection is full-res + slow), then swap its spots + refresh on the UI thread."""
        if not self.images or not hasattr(self, "_preset_var"):
            return
        if getattr(self, "_bg_active", False):
            return   # a full-roll background pass owns detection right now
        img = self.images[self.current_idx]
        path = img.get("image_path")
        if not path or not os.path.isfile(path):
            return
        cfg = self._effective_cfg()   # base preset + this frame's param overrides
        ml = self._ml_model_path()
        idx = self.current_idx
        if not hasattr(self, "_detect_q"):
            self._detect_q = queue.Queue()
        if getattr(self, "_detecting", False):
            return   # one at a time; the combo selection persists for next time
        self._detecting = True
        # SHARED canvas progress overlay (common/debug_ui_base.py) — same machinery
        # negadoctor uses for process_roll / preset switches. Detection has no
        # sub-progress, so _poll_detect heartbeats the bar while it runs.
        self._show_canvas_message(f"Detecting {img['stem']} …")
        if hasattr(self, "_detect_status"):
            self._detect_status.config(text=f"detecting {img['stem']}…")

        def work():
            import detect_dust
            try:
                spots, rej, err, _ls = detect_dust.detect(path, ml_model_path=ml, cfg=cfg)
                self._detect_q.put((idx, spots or [], rej or [], err))
            except Exception as ex:   # noqa: BLE001
                self._detect_q.put((idx, None, None, str(ex)))

        threading.Thread(target=work, daemon=True).start()
        self.root.after(150, self._poll_detect)

    def _poll_detect(self):
        try:
            idx, spots, rej, err = self._detect_q.get_nowait()
        except queue.Empty:
            # Indeterminate heartbeat: nudge the overlay bar forward while the
            # (single-image, ~tens-of-seconds) detection runs, so it doesn't stall.
            self._set_analyzing_pct(min(85.0, getattr(self, "_prog_target", 0.0) + 1.0))
            self.root.after(150, self._poll_detect)
            return
        self._detecting = False
        self._set_analyzing_pct(100.0)
        self._clear_canvas_message()
        if spots is not None and 0 <= idx < len(self.images):
            img = self.images[idx]
            if getattr(self, "review_mode", False):
                # In review the redetect feeds the 'live' R-source, NOT the base
                # frame: store under review['live'] (tagged with the preset) and
                # only swap into the frame when 'live' is what's on screen.
                rev = img.get("review")
                if rev is not None:
                    rev["live"] = {"detected": spots, "rejected": rej or []}
                    rev["live_preset"] = self._review_live_preset
                if rev is not None and self.review_source == "live":
                    img["detected"] = spots
                    img["rejected"] = rej or []
                    if idx == self.current_idx:
                        self.annotations[img["stem"]] = \
                            self.new_annotation_state(img)   # indices changed
                        self.reset_selection()
                        self._populate_items_list()
                        self._redraw()
                        self.update_counts()
            else:
                img["detected"] = spots
                img["_detect_pending"] = False
                if idx == self.current_idx:
                    # Reset annotation state (detected indices changed) then RELOAD
                    # the saved {stem}_annotations.json — same as _poll_bg_detect —
                    # so a preset re-detect keeps the user's missed_dust / missed_
                    # threads / FPs (coord-matched) instead of silently dropping the
                    # imported/hand-added ground truth.
                    self.annotations[img["stem"]] = self.new_annotation_state(img)
                    self._load_existing_annotations_for(img)
                    self.reset_selection()
                    self._populate_items_list()
                    self._redraw()
                    self.update_counts()
                    if hasattr(self, "params_tree"):
                        self._populate_params_table()
        if hasattr(self, "_detect_status"):
            self._detect_status.config(text=("" if not err else f"error: {err}"))

    # ------------------------------------------------------------------
    # Layout / text hooks
    # ------------------------------------------------------------------

    def build_left_info(self, parent):
        # Keep the left panel minimal — only the image list. The per-frame status
        # (dims / detected / rejected) lives in the bottom "Selected" box via
        # default_info_text; the visibility toggles are View-menu checkbuttons;
        # the legend + shortcuts are Help-menu popups — all like auto_negadoctor.
        self.status_label.pack_forget()

    # Marker legend + the full mouse/key reference live on the Help menu (popups),
    # NOT in the left panel — same as auto_negadoctor.
    _LEGEND_ENTRIES = [
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
    ]

    def build_feature_menus(self, menubar):
        """The 'Annotate' cascade — the edit actions that used to be the
        lower-left button column (SHOW_BOTTOM_BUTTONS is now False, like
        auto_negadoctor). 'Remove Missed' is enabled only when a missed marker
        is selected; the existing `self.remove_missed_btn.config(state=…)` call
        sites drive that menu entry through a tiny proxy (see _MenuEntryState)."""
        ann = tk.Menu(menubar, tearoff=0)
        ann.add_command(label="Rejected → Missed", accelerator="R",
                        command=self._mark_rejected_as_missed)
        ann.add_command(label="Mark false positive", accelerator="M",
                        command=self._mark_fp)
        ann.add_command(label="Clear FP mark", accelerator="C",
                        command=self._clear_fp)
        rm_idx = ann.index("end") + 1
        ann.add_command(label="Remove missed", accelerator="Del",
                        command=self._remove_missed, state=tk.DISABLED)
        ann.add_separator()
        ann.add_command(label="Draw missed thread", accelerator="T",
                        command=self._toggle_thread_draw)
        ann.add_separator()
        ann.add_command(label="Clear selection", command=self._clear_selection)
        menubar.add_cascade(label="Annotate", menu=ann)
        # Drive the (formerly button) enable/disable through the menu entry.
        self.remove_missed_btn = _MenuEntryState(ann, rm_idx)

    def bind_feature_keys(self):
        self.bind_key("<m>", lambda e: self._mark_fp())
        self.bind_key("<M>", lambda e: self._mark_fp())
        self.bind_key("<c>", lambda e: self._clear_fp())
        self.bind_key("<C>", lambda e: self._clear_fp())
        self.bind_key("<Delete>", lambda e: self._remove_missed())
        self.bind_key("<BackSpace>", lambda e: self._remove_missed())
        self.bind_key("<r>", lambda e: self._on_r_key())
        self.bind_key("<R>", lambda e: self._on_r_key())
        self.bind_key("<t>", lambda e: self._toggle_thread_draw())
        self.bind_key("<T>", lambda e: self._toggle_thread_draw())
        self.bind_key("<Return>", lambda e: self._finish_thread())
        self.bind_key("<Escape>", lambda e: self._cancel_thread())

    def image_status_text(self, img_dict):
        detected = img_dict.get("detected") or []
        rejected_list = img_dict.get("rejected") or []
        t = img_dict.get("processing_time_s")
        time_str = f"\nprocessed in {t:.1f}s" if t is not None else ""
        if self.run_wall_time_s is not None:
            time_str += f"\nrun total {self.run_wall_time_s:.1f}s"
        return (f"{img_dict['width']} × {img_dict['height']} px\n"
                f"{len(detected)} detected\n{len(rejected_list)} rejected candidates"
                f"{time_str}")

    def default_info_text(self):
        # The per-frame status (dims / detected / rejected / timings) now lives
        # here in the bottom "Selected" box (the left status label is hidden).
        head = ""
        if self.images:
            head = self.image_status_text(self.images[self.current_idx]) + "\n\n"
        return (head + "No marker selected.\n"
                "Click a marker or Ctrl+Click to add missed dust.")

    def _nothing_selected(self):
        return (not self.selected_detected and not self.selected_rejected
                and not self.selected_missed and self.selected_source is None
                and self.selected_keypoint is None
                and self.selected_missed_stroke is None)

    def update_counts(self):
        self._update_count_label()
        # Detection can land after the box was first filled (run-mode streaming),
        # so refresh the per-frame status while nothing is selected.
        if self._nothing_selected():
            self._set_info_text(self.default_info_text())

    # ------------------------------------------------------------------
    # Note hooks (free-text user annotation per object)
    # ------------------------------------------------------------------

    def get_selected_note(self):
        ann = self.annotations[self.images[self.current_idx]["stem"]]
        if len(self.selected_detected) == 1:
            idx = next(iter(self.selected_detected))
            return ann["spot_notes"].get(idx, "")
        if len(self.selected_missed) == 1:
            idx = next(iter(self.selected_missed))
            if idx < len(ann["missed_dust"]):
                return ann["missed_dust"][idx].get("note", "")
        if self.selected_missed_stroke is not None:
            ms_list = ann.get("missed_strokes", [])
            if self.selected_missed_stroke < len(ms_list):
                return ms_list[self.selected_missed_stroke].get("note", "")
        return None

    def set_selected_note(self, text):
        ann = self.annotations[self.images[self.current_idx]["stem"]]
        if len(self.selected_detected) == 1:
            idx = next(iter(self.selected_detected))
            if text:
                ann["spot_notes"][idx] = text
            else:
                ann["spot_notes"].pop(idx, None)
        elif len(self.selected_missed) == 1:
            idx = next(iter(self.selected_missed))
            if idx < len(ann["missed_dust"]):
                if text:
                    ann["missed_dust"][idx]["note"] = text
                else:
                    ann["missed_dust"][idx].pop("note", None)
        elif self.selected_missed_stroke is not None:
            ms_list = ann.get("missed_strokes", [])
            if self.selected_missed_stroke < len(ms_list):
                if text:
                    ms_list[self.selected_missed_stroke]["note"] = text
                else:
                    ms_list[self.selected_missed_stroke].pop("note", None)

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

    def _source_buffers_for(self, img_dict):
        """Cached (img_lab, L_f32, local_std, w, h) for the current frame's source
        image, used to recommend a healing source for a hand-added missed dust.
        None when the source JPG can't be loaded (e.g. not present in the folder)."""
        stem = img_dict["stem"]
        if stem in self._missed_buffers:
            return self._missed_buffers[stem]
        path = img_dict.get("image_path")
        buffers = None
        if path and os.path.exists(path):
            try:
                import detect_dust
                buffers = detect_dust.prepare_source_buffers(path)
            except Exception:
                buffers = None
        self._missed_buffers[stem] = buffers
        return buffers

    def _seed_missed_dust(self, img_dict, md):
        """Fill a missed-dust entry's heal radius + recommended source IN PLACE
        (idempotent: keeps any value already set). Makes the hand-added dust a
        first-class healable spot — the user then scroll-resizes / drags the
        source. Uses the shared detect_dust.missed_dust_to_spot so the UI preview
        and the apply writer agree."""
        import detect_dust
        if md.get("brush_radius_px") is not None and md.get("src_cx") is not None:
            return md
        detected = img_dict.get("detected") or []
        buffers = self._source_buffers_for(img_dict)
        spot = detect_dust.missed_dust_to_spot(md, detected, buffers)
        md["brush_radius_px"] = spot["brush_radius_px"]
        md["radius_px"] = spot["radius_px"]
        if "src_cx" in spot:
            md.setdefault("src_cx", spot["src_cx"])
            md.setdefault("src_cy", spot["src_cy"])
        return md

    def _seed_missed_stroke(self, img_dict, ms):
        """Fill a hand-drawn / imported missed-thread's heal radius + recommended
        healing source IN PLACE (idempotent). Mirrors _seed_missed_dust so a
        thread is a first-class healable spot with a visible, draggable source
        (darktable heals a brush stroke by cloning it from a single source anchor)."""
        import detect_dust
        path = ms.get("path") or []
        if len(path) < 2:
            return ms
        min_dim = min(img_dict.get("width") or 0, img_dict.get("height") or 0)
        spot = detect_dust.missed_stroke_to_spot(ms, min_dim)
        if spot is None:
            return ms
        ms.setdefault("brush_radius_px", spot["brush_radius_px"])
        if ms.get("src_cx") is not None and ms.get("src_cy") is not None:
            return ms
        # darktable heals a brush by translating the whole stroke so path[0] maps
        # to the stored source anchor (mask_src). So the anchor we store is P0's
        # source; pick it so the BAND (whole stroke) lands on the clean area
        # find_healing_source picks for the stroke centre:
        #   anchor S = P0 + (clean_source_at_mid − mid)  →  band midpoint lands on clean.
        p0x, p0y = float(path[0][0]), float(path[0][1])
        mx, my = float(path[len(path) // 2][0]), float(path[len(path) // 2][1])
        brush_radius_px = float(ms.get("brush_radius_px") or spot["brush_radius_px"])
        radius_px = max(1.0, float(ms.get("stroke_width_px") or brush_radius_px) / 2.0)
        buffers = self._source_buffers_for(img_dict)
        src = (None, None)
        if buffers is not None:
            img_lab, L_f32, local_std, w, h = buffers
            src = detect_dust.find_healing_source(
                mx, my, radius_px, brush_radius_px,
                img_lab, L_f32, local_std, img_dict.get("detected") or [], w, h)
        if src and src[0] is not None:
            ms["src_cx"] = p0x + (float(src[0]) - mx)
            ms["src_cy"] = p0y + (float(src[1]) - my)
        else:
            dim = min_dim or 1000
            ms["src_cx"] = p0x + detect_dust.HEAL_SOURCE_OFFSET_X * dim
            ms["src_cy"] = p0y + detect_dust.HEAL_SOURCE_OFFSET_Y * dim
        return ms

    def _swept_disc_centers(self, path, step):
        """Sample points at ~`step` px spacing along a polyline (image coords).

        A darktable brush stroke heals a BAND — the union of discs of radius
        brush_radius_px swept along the spline — so drawing outline discs at these
        samples shows the real healed shape (not just the endpoints), which is what
        lets the user check the source band doesn't overlap the healed thread."""
        pts = [(float(p[0]), float(p[1])) for p in path]
        if len(pts) < 2:
            return pts
        cum = [0.0]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            cum.append(cum[-1] + math.hypot(x1 - x0, y1 - y0))
        total = cum[-1]
        if total <= 0:
            return [pts[0]]
        step = max(float(step), total / 150.0, 1.0)   # cap the disc count
        n = int(total // step)
        samples, j = [], 0
        for k in range(n + 1):
            s = min(k * step, total)
            while j < len(cum) - 2 and cum[j + 1] < s:
                j += 1
            seg = cum[j + 1] - cum[j]
            t = 0.0 if seg <= 1e-9 else (s - cum[j]) / seg
            samples.append((pts[j][0] + t * (pts[j + 1][0] - pts[j][0]),
                            pts[j][1] + t * (pts[j + 1][1] - pts[j][1])))
        samples.append(pts[-1])
        return samples

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

        # Detected spots (algo group — hidden when "Detected" is toggled off)
        detected_iter = (enumerate(img_dict.get("detected") or [])
                         if self.show_detected_var.get() else ())
        for i, spot in detected_iter:
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

        # Missed dust markers (cyan +), now first-class healable spots: a heal-radius
        # circle + a dashed line to the (recommended / dragged) healing source.
        # Part of the "loaded from existing" group — hidden when toggled off.
        missed_dust_iter = (enumerate(ann["missed_dust"])
                            if self.show_missed_var.get() else ())
        for i, md in missed_dust_iter:
            cx, cy = image_to_canvas(md["cx"], md["cy"],
                                     self.offset_x, self.offset_y, self.zoom)
            r = 10
            is_msel = i in self.selected_missed
            color = "#00ffff" if not is_msel else "#ffffff"
            lw = 2 if not is_msel else 3
            self.canvas.create_line(cx - r, cy, cx + r, cy,
                                    fill=color, width=lw, tags="missed")
            self.canvas.create_line(cx, cy - r, cx, cy + r,
                                    fill=color, width=lw, tags="missed")
            # Heal brush circle (when this missed dust has a radius)
            brad = md.get("brush_radius_px")
            if brad:
                crad = max(5, brad * self.zoom)
                self.canvas.create_oval(cx - crad, cy - crad, cx + crad, cy + crad,
                                        outline=color, width=lw, tags="missed")
            # Healing source: dashed line + square (selectable, Ctrl+drag to move)
            scx, scy = md.get("src_cx"), md.get("src_cy")
            if scx is not None and scy is not None:
                sx, sy = image_to_canvas(scx, scy,
                                         self.offset_x, self.offset_y, self.zoom)
                is_msrc_sel = (self.selected_missed_source == i)
                src_color = "#ffff00" if (is_msel or is_msrc_sel) else "#00cc44"
                sq = max(3, int(2 * self.zoom))
                self.canvas.create_line(cx, cy, sx, sy,
                                        fill=src_color, width=1, dash=(4, 3),
                                        tags="source")
                self.canvas.create_rectangle(sx - sq, sy - sq, sx + sq, sy + sq,
                                             outline=src_color,
                                             width=2 if is_msrc_sel else 1,
                                             tags="source")
                # Source heal circle — same radius as the missed dust (so it tracks
                # scroll-resizing of the dust); dashed to read as the SOURCE brush.
                if brad:
                    sbr = max(5, brad * self.zoom)
                    self.canvas.create_oval(sx - sbr, sy - sbr, sx + sbr, sy + sbr,
                                            outline=src_color,
                                            width=lw if is_msrc_sel else 1,
                                            dash=(4, 3), tags="source")

        # Missed threads (hand-drawn): centerline + node handles, PLUS the real
        # healed band (discs of brush_radius_px swept along the path) and the
        # healing-source band (that same band translated onto the source anchor),
        # so the user can see the true shapes and check they don't overlap.
        # Part of the "loaded from existing" group — hidden when toggled off.
        missed_stroke_iter = (enumerate(ann.get("missed_strokes", []))
                              if self.show_missed_var.get() else ())
        for i, ms in missed_stroke_iter:
            path = ms.get("path") or []
            if not path:
                continue
            self._seed_missed_stroke(img_dict, ms)
            brad = float(ms.get("brush_radius_px") or 0.0)
            crad = max(3.0, brad * self.zoom)
            sel = (self.selected_missed_stroke == i)
            color = "#ffffff" if sel else "#00ffff"

            # Real healed shape: swept-disc outlines along the centerline.
            centers = self._swept_disc_centers(path, max(brad, 1.0)) if brad > 0 else []
            for (px, py) in centers:
                dcx, dcy = image_to_canvas(px, py, self.offset_x, self.offset_y, self.zoom)
                self.canvas.create_oval(dcx - crad, dcy - crad, dcx + crad, dcy + crad,
                                        outline=color, width=1, tags="missed")

            canv = [image_to_canvas(px, py, self.offset_x, self.offset_y, self.zoom)
                    for px, py in path]
            if len(canv) >= 2:
                flat = [c for pt in canv for c in pt]
                self.canvas.create_line(*flat, fill=color, width=3 if sel else 2,
                                        capstyle="round", joinstyle="round", tags="missed")
            for (hx, hy) in canv:
                self.canvas.create_rectangle(hx - 4, hy - 4, hx + 4, hy + 4,
                                             outline=color, width=2, tags="missed")

            # Healing source: darktable clones the whole stroke from a single
            # source anchor. Draw the source band = the healed band translated onto
            # the anchor (real shape, so overlap is checkable), a dashed connector,
            # and a small square drag handle at the anchor.
            scx, scy = ms.get("src_cx"), ms.get("src_cy")
            if scx is not None and scy is not None:
                # The source anchor is path[0]'s source (darktable's brush reference),
                # so the source band = the thread translated by (anchor − path[0]).
                p0x, p0y = float(path[0][0]), float(path[0][1])
                dx, dy = float(scx) - p0x, float(scy) - p0y
                mcx, mcy = image_to_canvas(p0x, p0y, self.offset_x, self.offset_y, self.zoom)
                sx, sy = image_to_canvas(scx, scy, self.offset_x, self.offset_y, self.zoom)
                is_ssrc_sel = (self.selected_missed_stroke_source == i)
                src_color = "#ffff00" if (sel or is_ssrc_sel) else "#00cc44"
                for (px, py) in centers:
                    dcx, dcy = image_to_canvas(px + dx, py + dy,
                                               self.offset_x, self.offset_y, self.zoom)
                    self.canvas.create_oval(dcx - crad, dcy - crad, dcx + crad, dcy + crad,
                                            outline=src_color, width=1, dash=(3, 2),
                                            tags="source")
                scanv = [image_to_canvas(px + dx, py + dy,
                                         self.offset_x, self.offset_y, self.zoom)
                         for px, py in path]
                if len(scanv) >= 2:
                    flat = [c for pt in scanv for c in pt]
                    self.canvas.create_line(*flat, fill=src_color, width=1,
                                            capstyle="round", joinstyle="round",
                                            tags="source")
                self.canvas.create_line(mcx, mcy, sx, sy, fill=src_color, width=1,
                                        dash=(4, 3), tags="source")
                sq = max(3, int(2 * self.zoom))
                self.canvas.create_rectangle(sx - sq, sy - sq, sx + sq, sy + sq,
                                             outline=src_color,
                                             width=2 if is_ssrc_sel else 1, tags="source")

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
        # When exactly one missed dust is selected, scroll resizes its heal radius
        if len(self.selected_missed) == 1:
            self._adjust_missed_radius(event)
            return True
        return False

    def _adjust_missed_radius(self, event):
        """Scroll wheel: grow/shrink a hand-added missed dust's heal brush radius."""
        delta = 1 if (event.num == 4 or getattr(event, "delta", 0) > 0) else -1
        idx = next(iter(self.selected_missed))
        img_dict = self.images[self.current_idx]
        stem = img_dict["stem"]
        md_list = self.annotations[stem]["missed_dust"]
        if idx >= len(md_list):
            return
        md = md_list[idx]
        self._seed_missed_dust(img_dict, md)   # ensure it has a radius to scale from
        factor = 1.1 if delta > 0 else (1 / 1.1)
        import detect_dust
        min_dim = min(img_dict.get("width") or 0, img_dict.get("height") or 0)
        min_brush = detect_dust.MIN_BRUSH_FRAC * min_dim if min_dim else 2.0
        md["brush_radius_px"] = max(min_brush, md["brush_radius_px"] * factor)
        md["radius_px"] = max(1.0, md["brush_radius_px"] /
                              max(detect_dust.DEFAULT_TUNING.ENC_RADIUS_SCALE, 1e-6))
        self._auto_save(stem)
        self._redraw_markers()
        self._update_info_from_selection()

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

    def _missed_source_at(self, canvas_x, canvas_y):
        """Missed-dust index whose healing-source square is under the cursor, else None."""
        ix, iy = canvas_to_image(canvas_x, canvas_y,
                                 self.offset_x, self.offset_y, self.zoom)
        hit_r = max(8.0, 12.0 / self.zoom)
        stem = self.images[self.current_idx]["stem"]
        best, best_d = None, hit_r
        for i, md in enumerate(self.annotations[stem]["missed_dust"]):
            if md.get("src_cx") is None or md.get("src_cy") is None:
                continue
            d = math.hypot(md["src_cx"] - ix, md["src_cy"] - iy)
            if d < best_d:
                best, best_d = i, d
        return best

    def _missed_stroke_source_at(self, canvas_x, canvas_y):
        """Missed-thread index whose healing-source square is under the cursor, else None."""
        ix, iy = canvas_to_image(canvas_x, canvas_y,
                                 self.offset_x, self.offset_y, self.zoom)
        hit_r = max(8.0, 12.0 / self.zoom)
        stem = self.images[self.current_idx]["stem"]
        best, best_d = None, hit_r
        for i, ms in enumerate(self.annotations[stem].get("missed_strokes", [])):
            if ms.get("src_cx") is None or ms.get("src_cy") is None:
                continue
            d = math.hypot(ms["src_cx"] - ix, ms["src_cy"] - iy)
            if d < best_d:
                best, best_d = i, d
        return best

    def handle_press_override(self, event):
        # Thread-draw mode: every mouse-down drops a centerline point (immune to
        # accidental drag). Hold and drag to draw freehand (see handle_drag_override).
        if self.thread_draw_mode:
            self._thread_add_point(event.x, event.y)
            self.drag_start = (event.x, event.y)  # last freehand sample point
            self.is_dragging = False
            return True
        # Grab a missed-dust healing source square to DRAG it (claim the gesture so
        # the base doesn't rubber-band / zoom-to-rectangle). Works with or without
        # Ctrl; a press without movement just selects it (on release).
        idx = self._missed_source_at(event.x, event.y)
        if idx is not None:
            self._missed_src_drag = idx
            self.selected_missed_source = idx
            self.selected_missed = set()
            self.selected_detected = set()
            self.selected_source = None
            self.selected_missed_stroke = None
            self.selected_missed_stroke_source = None
            self.drag_start = (event.x, event.y)
            self.is_dragging = False
            self._redraw_markers()
            self._sync_item_list_selection()   # highlight the parent missed-dust row
            return True
        # Same, for a missed-THREAD healing source square.
        sidx = self._missed_stroke_source_at(event.x, event.y)
        if sidx is not None:
            self._missed_stroke_src_drag = sidx
            self.selected_missed_stroke_source = sidx
            self.selected_missed_source = None
            self.selected_missed = set()
            self.selected_detected = set()
            self.selected_source = None
            self.selected_missed_stroke = None
            self.drag_start = (event.x, event.y)
            self.is_dragging = False
            self._redraw_markers()
            self._sync_item_list_selection()   # highlight the parent missed-thread row
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
        # Dragging a missed-dust healing source: move it live.
        if self._missed_src_drag is not None:
            img_dict = self.images[self.current_idx]
            stem = img_dict["stem"]
            ix, iy = canvas_to_image(event.x, event.y,
                                     self.offset_x, self.offset_y, self.zoom)
            iw, ih = img_dict["width"], img_dict["height"]
            ix = max(0.0, min(float(iw), ix))
            iy = max(0.0, min(float(ih), iy))
            md_list = self.annotations[stem]["missed_dust"]
            if self._missed_src_drag < len(md_list):
                md = md_list[self._missed_src_drag]
                md["src_cx"], md["src_cy"] = float(ix), float(iy)
                self.is_dragging = True
                self._redraw_markers()
            return True
        # Dragging a missed-THREAD healing source: move it live.
        if self._missed_stroke_src_drag is not None:
            img_dict = self.images[self.current_idx]
            stem = img_dict["stem"]
            ix, iy = canvas_to_image(event.x, event.y,
                                     self.offset_x, self.offset_y, self.zoom)
            iw, ih = img_dict["width"], img_dict["height"]
            ix = max(0.0, min(float(iw), ix))
            iy = max(0.0, min(float(ih), iy))
            ms_list = self.annotations[stem].get("missed_strokes", [])
            if self._missed_stroke_src_drag < len(ms_list):
                ms = ms_list[self._missed_stroke_src_drag]
                ms["src_cx"], ms["src_cy"] = float(ix), float(iy)
                self.is_dragging = True
                self._redraw_markers()
            return True
        return False

    def handle_release_override(self, event):
        # Thread-draw mode: points were already placed on press/drag; nothing to do.
        if self.thread_draw_mode:
            self.drag_start = None
            self.is_dragging = False
            return True
        # Finish a missed-dust source drag: persist it.
        if self._missed_src_drag is not None:
            stem = self.images[self.current_idx]["stem"]
            idx = self._missed_src_drag
            self._missed_src_drag = None
            self.drag_start = None
            self.is_dragging = False
            self._auto_save(stem)
            self._redraw_markers()
            self._set_info_text(
                f"Healing source for missed dust #{idx} moved.\n"
                f"  Drag it again to adjust, or scroll the dust to resize.")
            return True
        # Finish a missed-thread source drag: persist it.
        if self._missed_stroke_src_drag is not None:
            stem = self.images[self.current_idx]["stem"]
            idx = self._missed_stroke_src_drag
            self._missed_stroke_src_drag = None
            self.drag_start = None
            self.is_dragging = False
            self._auto_save(stem)
            self._redraw_markers()
            self._set_info_text(
                f"Healing source for missed thread #{idx} moved.\n"
                f"  Drag it again to adjust.")
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

        # If a MISSED-dust source is selected, move that missed dust's heal source.
        if self.selected_missed_source is not None:
            md_list = self.annotations[stem]["missed_dust"]
            if self.selected_missed_source < len(md_list):
                md = md_list[self.selected_missed_source]
                md["src_cx"] = float(ix)
                md["src_cy"] = float(iy)
                self._auto_save(stem)
                self._redraw_markers()
                self._set_info_text(
                    f"Healing source for missed dust #{self.selected_missed_source} "
                    f"moved to ({ix:.0f}, {iy:.0f})\n  Ctrl+Click again to reposition.")
            return

        # If near existing marker, treat as normal click instead
        nearest = self._find_nearest_marker(canvas_x, canvas_y)
        if nearest is not None:
            self.on_click(canvas_x, canvas_y)
            return

        md = {"cx": ix, "cy": iy}
        self._seed_missed_dust(img_dict, md)   # heal radius + recommended source
        self.annotations[stem]["missed_dust"].append(md)
        new_idx = len(self.annotations[stem]["missed_dust"]) - 1
        self.selected_detected = set()
        self.selected_rejected = set()
        self.selected_missed = {new_idx}
        self.selected_source = None
        self.selected_missed_source = None
        self.remove_missed_btn.config(state=tk.NORMAL)
        self._auto_save(stem)
        self._redraw_markers()
        self._update_count_label()
        self._populate_items_list()
        self._sync_item_list_selection()
        self._set_info_text(
            f"Added missed dust at ({ix:.0f}, {iy:.0f})\n"
            f"  Scroll = resize; drag its green source square to move the healing source.")

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
            self.selected_missed_source = None
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
        self.selected_missed_source = None
        self.selected_missed_stroke_source = None
        self.selected_keypoint = None
        self.selected_missed_stroke = None

        if kind == "detected":
            self.selected_detected = {idx}
            self.remove_missed_btn.config(state=tk.DISABLED)
        elif kind == "source":
            self.selected_source = idx
            self.remove_missed_btn.config(state=tk.DISABLED)
        elif kind == "missed_source":
            self.selected_missed_source = idx
            self.remove_missed_btn.config(state=tk.DISABLED)
            self._set_info_text(
                f"Healing source for missed dust #{idx} selected.\n"
                f"  Drag it to move the healing source.")
            self._redraw_markers()
            return
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
            self._refresh_note_entry()
            # This branch sets its own info text (not _update_info_from_selection),
            # so sync the table row highlight explicitly — otherwise a canvas-click
            # on a thread selects it on the canvas but not in the item table.
            self._sync_item_list_selection()
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

        # Missed-dust heal source markers (so they can be selected + Ctrl+drag-moved)
        for i, md in enumerate(ann["missed_dust"]):
            if md.get("src_cx") is None or md.get("src_cy") is None:
                continue
            d = math.hypot(md["src_cx"] - ix, md["src_cy"] - iy)
            if d < hit_r and d < best_dist:
                best = ("missed_source", i)
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

        spot_notes = ann.get("spot_notes", {})
        rows = []
        for i, spot in enumerate(detected):
            is_fp = i in ann["false_positives"]
            has_note = i in spot_notes
            rows.append({
                "iid": f"det_{i}",
                "values": (i,
                           int(spot["cx"]),
                           int(spot["cy"]),
                           int(spot["brush_radius_px"]),
                           f"{spot['contrast']:.1f}",
                           f"{spot['texture']:.1f}",
                           "●" if is_fp else "",
                           "✎" if has_note else ""),
                "tag": "fp" if is_fp else "det",
                "kind": "detected", "idx": i,
                "sort": {"idx": i,
                         "cx":  spot["cx"],            "cy":  spot["cy"],
                         "r":   spot["brush_radius_px"],
                         "ctr": spot["contrast"],      "tex": spot["texture"],
                         "fp":  1 if is_fp else 0,
                         "nt":  1 if has_note else 0},
            })

        for i, md in enumerate(missed):
            has_note = bool(md.get("note"))
            rows.append({
                "iid": f"missed_{i}",
                "values": (f"m{i}", int(md["cx"]), int(md["cy"]), "", "", "", "",
                           "✎" if has_note else ""),
                "tag": "missed",
                "kind": "missed", "idx": i,
                "sort": {"idx": 100000 + i,
                         "cx":  md["cx"],  "cy":  md["cy"],
                         "r": -1, "ctr": -1, "tex": -1, "fp": 0,
                         "nt": 1 if has_note else 0},
            })

        for i, ms in enumerate(missed_strokes):
            path = ms.get("path", [])
            if not path:
                continue
            mid = path[len(path) // 2]
            cx, cy = mid[0], mid[1]
            has_note = bool(ms.get("note"))
            rows.append({
                "iid": f"mstroke_{i}",
                "values": (f"t{i}", int(cx), int(cy), "", "", "", "",
                           "✎" if has_note else ""),
                "tag": "missed",
                "kind": "missed_stroke", "idx": i,
                "sort": {"idx": 200000 + i,
                         "cx": cx, "cy": cy,
                         "r": -1, "ctr": -1, "tex": -1, "fp": 0,
                         "nt": 1 if has_note else 0},
            })
        return rows

    def is_row_currently_selected(self, row):
        # A row counts as selected either when its marker OR its healing source
        # is the current selection (clicking a source highlights the parent item).
        if row["kind"] == "detected" and (row["idx"] in self.selected_detected
                                          or self.selected_source == row["idx"]):
            return True
        if row["kind"] == "missed" and (row["idx"] in self.selected_missed
                                        or self.selected_missed_source == row["idx"]):
            return True
        if row["kind"] == "missed_stroke" and (
                row["idx"] == self.selected_missed_stroke
                or self.selected_missed_stroke_source == row["idx"]):
            return True
        return False

    def on_item_row_selected(self, row):
        self.selected_detected = set()
        self.selected_rejected = set()
        self.selected_missed = set()
        self.selected_missed_stroke = None
        self.selected_source = None
        self.selected_missed_source = None
        self.selected_missed_stroke_source = None

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
        # A selected healing source maps to its PARENT item's row, so clicking a
        # source scrolls/highlights the corresponding entry in the table.
        if self.selected_detected:
            return f"det_{next(iter(self.selected_detected))}"
        if self.selected_source is not None:
            return f"det_{self.selected_source}"
        if self.selected_missed:
            return f"missed_{next(iter(self.selected_missed))}"
        if self.selected_missed_source is not None:
            return f"missed_{self.selected_missed_source}"
        if self.selected_missed_stroke is not None:
            return f"mstroke_{self.selected_missed_stroke}"
        if self.selected_missed_stroke_source is not None:
            return f"mstroke_{self.selected_missed_stroke_source}"
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
                        "MAX_FORMS", "MIN_BRUSH_FRAC"]:
                if key in constants:
                    lines.append(f"  {key} = {constants[key]}")
            return lines
        # Key constants
        for key in ["NOISE_THRESHOLD_MULTIPLIER", "MIN_ABSOLUTE_THRESHOLD",
                    "MIN_SPOT_AREA_FRAC", "MAX_SPOT_AREA_FRAC", "MIN_ASPECT_RATIO",
                    "MIN_COMPACTNESS", "MIN_SOLIDITY", "MIN_CIRCULARITY",
                    "MAX_LOCAL_TEXTURE_SMALL", "MAX_LOCAL_TEXTURE_LARGE",
                    "MIN_CONTRAST_TEXTURE_RATIO", "MAX_BG_GRADIENT_RATIO",
                    "MAX_EXCESS_SATURATION", "MAX_CONTEXT_TEXTURE",
                    "LARGE_SPOT_AREA_THRESHOLD_FRAC", "LARGE_SPOT_MIN_CONTRAST",
                    "ISOLATION_RADIUS_FRAC", "MAX_NEARBY_ACCEPTED",
                    "SOFT_CONTEXT_VOTE_THRESHOLD", "SOFT_TEXTURE_VOTE_THRESHOLD",
                    "SOFT_RATIO_VOTE_THRESHOLD", "MIN_DUST_VOTES",
                    "REJECT_LOG_CONTRAST_MIN",
                    "MIN_BRUSH_FRAC", "BRUSH_HARDNESS"]:
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
            spot_notes = ann.get("spot_notes", {})
            missed_notes = [(i, m) for i, m in enumerate(missed_list) if m.get("note")]
            stroke_notes = [(i, ms) for i, ms in enumerate(ann.get("missed_strokes", []))
                            if ms.get("note")]
            has_notes = bool(spot_notes or missed_notes or stroke_notes)
            total_fp += len(fp_indices)
            total_missed += len(missed_list)
            total_src_repositions += len(src_overrides)
            has_annotations = bool(fp_indices or missed_list or src_overrides
                                   or radius_overrides or has_notes)
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

            # User notes (free-text comments on objects)
            if has_notes:
                lines.append("")
                lines.append("  USER NOTES (free-text comments on objects):")
                for i in sorted(spot_notes):
                    if i < len(detected):
                        s = detected[i]
                        lines.append(f"    Detected #{i} (cx={s['cx']:.0f}, "
                                     f"cy={s['cy']:.0f}): {spot_notes[i]}")
                    else:
                        lines.append(f"    Detected #{i}: {spot_notes[i]}")
                for i, m in missed_notes:
                    lines.append(f"    Missed dust #{i} (cx={m['cx']:.0f}, "
                                 f"cy={m['cy']:.0f}): {m['note']}")
                for i, ms in stroke_notes:
                    path = ms.get("path", [])
                    mid = path[len(path) // 2] if path else (0, 0)
                    lines.append(f"    Missed thread #{i} (mid cx={mid[0]:.0f}, "
                                 f"cy={mid[1]:.0f}): {ms['note']}")

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
    # Flags (consumed before run_main, which reads argv[1] as the session dir):
    #   --apply       : write dust_results.txt on close (apply-from-folder flow)
    #   --choose-dir  : pop a native folder picker instead of a positional dir
    if "--apply" in sys.argv:
        DustDebugUI.apply_mode = True
    if "--choose-dir" in sys.argv:
        DustDebugUI.choose_dir = True
    if "--import-gt" in sys.argv:
        DustDebugUI.import_gt = True
        DustDebugUI.apply_mode = True   # import-GT always writes on close
    sys.argv = [a for a in sys.argv if a not in ("--apply", "--choose-dir", "--import-gt")]
    DustDebugUI.run_main(
        usage="Usage: debug_ui.py <export_dir> [--apply] [--choose-dir] [--import-gt]")


if __name__ == "__main__":
    main()
