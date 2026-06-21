"""
Shared debug-UI viewer base for the detectors in this repo.

DebugUIBase provides the generic annotation-viewer machinery: window layout,
image display with zoom/pan, thumbnail navigation, a sortable item table,
selection plumbing, the annotation save/load lifecycle and report writing.
Feature UIs (dust, crop, ...) subclass it and implement the hooks to define
what is drawn, what clicking means, the annotation schema and report content.

Usage from a feature debug_ui.py:

    sys.path.insert(0, str(Path(__file__).parent.parent))  # repo root
    from common.debug_ui_base import DebugUIBase

    class MyDebugUI(DebugUIBase):
        ...

    if __name__ == "__main__":
        MyDebugUI.run_main()

The session directory contract: load_session() returns (images, constants)
where each image dict has at least "stem", "image_path", "width", "height".
Annotations are auto-saved to {stem}_annotations.json on every edit and on
image switch/close; a human-readable report is written on window close.
"""

import sys
import os
import json
import math
import os
import sys
import datetime
import threading
import queue

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox

from PIL import Image, ImageTk


# ---------------------------------------------------------------------------
# Display color management (match darktable's color-managed darkroom view)
#
# All preview pixels here are sRGB (the analysis space). darktable renders its
# darkroom/export THROUGH the monitor's ICC profile; on a wide-gamut panel
# (e.g. Display P3) un-managed sRGB bytes look OVER-saturated (reds/oranges most
# of all), so the debug UI must do the same sRGB -> monitor-profile transform on
# what it blits — otherwise the user tunes WB/exposure against colors darktable
# will never show. The transform is applied to the DISPLAYED bitmap only; the
# histogram / clip analysis stay on the true sRGB values (which match the
# darktable EXPORT). Override the detected profile with NEGA_DISPLAY_ICC=<path>;
# set NEGA_DISPLAY_ICC=off to disable color management entirely.
# ---------------------------------------------------------------------------

def detect_display_icc():
    """Path to the active monitor's ICC profile, or None. Honors the
    NEGA_DISPLAY_ICC override ('off' disables); Windows-native detection via
    GetICMProfile, otherwise None (un-managed, same as before)."""
    env = os.environ.get("NEGA_DISPLAY_ICC")
    if env:
        if env.lower() == "off":
            return None
        return env if os.path.exists(env) else None
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
            hdc = user32.GetDC(0)
            try:
                size = wintypes.DWORD(0)
                gdi32.GetICMProfileW(hdc, ctypes.byref(size), None)
                buf = ctypes.create_unicode_buffer(size.value or 260)
                if gdi32.GetICMProfileW(hdc, ctypes.byref(size), buf):
                    return buf.value
            finally:
                user32.ReleaseDC(0, hdc)
        except Exception:
            return None
    return None


def build_srgb_to_display_transform():
    """A PIL ImageCms transform sRGB -> the monitor profile (relative
    colorimetric, BPC), or None when no profile / not applicable / PIL lacks
    littleCMS. Relative-colorimetric is darktable's default display intent and
    is what matters here (sRGB sits inside wide-gamut panels, so it's a faithful
    primaries remap with no perceptual gamut compression to disagree about)."""
    path = detect_display_icc()
    if not path:
        return None
    try:
        from PIL import ImageCms
        dst = ImageCms.getOpenProfile(path)
        src = ImageCms.createProfile("sRGB")
        return ImageCms.buildTransform(
            src, dst, "RGB", "RGB",
            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
            flags=ImageCms.Flags.BLACKPOINTCOMPENSATION)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Coordinate helpers (shared by base and feature subclasses)
# ---------------------------------------------------------------------------

def canvas_to_image(cx, cy, offset_x, offset_y, zoom):
    return (cx - offset_x) / zoom, (cy - offset_y) / zoom


def image_to_canvas(ix, iy, offset_x, offset_y, zoom):
    return ix * zoom + offset_x, iy * zoom + offset_y


def _seg_dist(px, py, ax, ay, bx, by):
    """Distance from point (px,py) to segment (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _point_to_path_dist(px, py, path):
    """Minimum distance from point (px,py) to a polyline (list of [x,y])."""
    if not path:
        return float("inf")
    if len(path) == 1:
        return math.hypot(px - path[0][0], py - path[0][1])
    return min(_seg_dist(px, py, path[i][0], path[i][1], path[i + 1][0], path[i + 1][1])
              for i in range(len(path) - 1))


# ---------------------------------------------------------------------------
# Base viewer class
# ---------------------------------------------------------------------------

class DebugUIBase:
    # --- Class-level configuration (override in subclasses) ---
    WINDOW_TITLE = "Debug UI"
    WINDOW_GEOMETRY = "1400x900"
    EMPTY_SESSION_MESSAGE = "No debug session files found in:"
    ANNOTATION_SUFFIX = "_annotations.json"
    REPORT_FILENAME = "debug_report.txt"
    # Item table (right panel) columns
    ITEM_COLS = ()            # tuple of column ids
    ITEM_HEADERS = {}         # col -> header text
    ITEM_WIDTHS = {}          # col -> px width
    ITEM_ANCHORS = {}         # col -> anchor (default "e")
    ITEM_STRETCH = ()         # cols that absorb the panel's extra width
                              # (stretch=True); others stay at their fixed width
    ITEM_PANEL_TITLE = "Items:"
    CENTER_BUTTON_TEXT = "Center on item"
    # When True, the right panel splits vertically (draggable sash): the item
    # table on top, a feature footer (build_item_panel_footer) below it. The
    # footer pane opens at its requested height (fully visible) and the table
    # takes the remaining space.
    HAS_ITEM_PANEL_FOOTER = False
    # Initial width (px) of the right-side item panel.
    ITEM_PANEL_WIDTH = 230
    # When True, on window resize the center (image) pane is shrunk to the width
    # the height-fitted image actually needs, and the freed horizontal space is
    # handed to the item panel (no wasted pillarbox black bars). Off by default;
    # UIs with a rich right panel (color wheels) opt in.
    REFLOW_TO_IMAGE = False
    # Upper bound on the item panel's share of the window width when reflowing.
    REFLOW_ITEM_MAX_FRAC = 0.45
    # When False, the bottom-left button column (feature buttons + Clear
    # Selection / Fit / Hide Markers) is omitted — for UIs whose actions all
    # live on the menu bar + toolbar (it scales poorly and is redundant there).
    SHOW_BOTTOM_BUTTONS = True

    def __init__(self, root, session_dir):
        self.root = root
        self.session_dir = session_dir
        self.export_dir = session_dir   # legacy alias used by feature code
        self.root.title(self.WINDOW_TITLE)
        # Hi-DPI: the window is DPI-aware (run_main), so pixel-literal sizes
        # render at PHYSICAL pixels and look tiny on a 4K display. Scale the
        # window and pixel-literal layout sizes by the screen DPI factor.
        self.ui_scale = self._ui_scale()
        self.root.geometry(self._scaled_geometry(self.WINDOW_GEOMETRY))

        images, constants = self.load_session(session_dir)
        if not images:
            messagebox.showerror("Error",
                                 f"{self.EMPTY_SESSION_MESSAGE}\n{session_dir}")
            root.destroy()
            return

        self.data = {"images": images, "constants": constants}
        self.images = images
        self.constants = constants
        # Total wall-clock time of the detection run that produced this
        # session (session-level; duplicated into every per-image file)
        self.run_wall_time_s = images[0].get("run_wall_time_s")

        # Per-image annotation state (schema defined by the subclass)
        self.annotations = {}
        for img in self.images:
            self.annotations[img["stem"]] = self.new_annotation_state(img)

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

        # Feature-specific selection state
        self.init_selection_state()

        # Visibility
        self.hide_markers = False

        # Display color management (sRGB -> monitor profile, like darktable).
        # On by default; the transform itself is built lazily/cached on first use.
        self.color_manage = True

        # Rubber-band drag
        self.drag_start: "tuple | None" = None
        self.is_dragging = False
        self.ctrl_drag_mode = False

        # Pan state
        self.pan_start = None
        self.pan_offset_at_start = None

        # Track whether we're at fit zoom so resize can re-fit automatically
        self.at_fit_zoom = True

        self.item_list_data = {}      # iid -> row dict from item_rows()
        self._syncing_selection = False
        self._reflow_after_id = None

        # Keyboard-layout-independent letter shortcuts (see bind_key). Maps a
        # Windows virtual-key code -> handler; dispatched from a single generic
        # <KeyPress> binding so the physical key position triggers the shortcut
        # regardless of the active input language (Latin/Cyrillic/...).
        self._phys_keymap = {}
        self._phys_ctrl_keymap = {}     # Ctrl+<letter> shortcuts, by keycode
        if sys.platform == "win32":
            self.root.bind("<KeyPress>", self._on_physical_key)

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        if self.REFLOW_TO_IMAGE:
            self.root.bind("<Configure>", self._on_root_configure)

        # Defer all post-UI work so the window skeleton appears immediately.
        # after(0) fires on the first event-loop tick; after(150) fires later,
        # by which time lb_rows will already exist.
        self.root.after(0, self._populate_thumb_list)
        self.root.after(150, self._load_image_by_idx, 0)

    # ------------------------------------------------------------------
    # Hi-DPI pixel scaling
    # ------------------------------------------------------------------

    def _ui_scale(self):
        """Physical-pixel scale factor: 1.0 at 96 dpi, ~2.0 at 192 dpi (a 4K
        display at 200%). Pixel-literal widths/sizes are multiplied by this so
        the DPI-aware window isn't physically tiny. Never below 1.0."""
        try:
            s = self.root.winfo_fpixels("1i") / 96.0
        except Exception:
            s = 1.0
        return s if s and s > 1.0 else 1.0

    def scaled(self, px):
        """Scale a pixel-literal layout size by the hi-DPI factor."""
        return int(round(px * self.ui_scale))

    def _scaled_geometry(self, geom):
        """Scale a 'WxH' / 'WxH+X+Y' geometry string's size by ui_scale."""
        try:
            size, plus, pos = geom.partition("+")
            w, _, h = size.partition("x")
            g = f"{self.scaled(int(w))}x{self.scaled(int(h))}"
            return g + plus + pos
        except Exception:
            return geom

    # ------------------------------------------------------------------
    # Hooks: session / annotations  (subclass MUST implement)
    # ------------------------------------------------------------------

    def load_session(self, session_dir):
        """Return (images, constants). Each image dict needs stem,
        image_path, width, height."""
        raise NotImplementedError

    def new_annotation_state(self, img_dict):
        """Return a fresh in-memory annotation dict for one image."""
        raise NotImplementedError

    def serialize_annotations(self, stem):
        """Return the JSON-serializable dict written to {stem}_annotations.json."""
        raise NotImplementedError

    def deserialize_annotations(self, img_dict, data):
        """Restore in-memory annotation state from a loaded annotations dict."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Hooks: rendering  (subclass MUST implement)
    # ------------------------------------------------------------------

    def draw_overlays(self):
        """Draw all feature markers for the current image on self.canvas."""
        raise NotImplementedError

    def overlay_tags(self):
        """Canvas tags deleted before each marker redraw."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Hooks: layout / text  (optional)
    # ------------------------------------------------------------------

    def build_left_controls(self, parent):
        """Feature widgets between the thumbnail list and the status label
        (e.g. visibility checkboxes)."""

    def build_left_info(self, parent):
        """Feature widgets below the status label (legend, key hints)."""

    def build_feature_buttons(self, btn_frame, btn_cfg):
        """Feature buttons/labels packed above the common buttons."""

    def build_item_panel_footer(self, parent):
        """Feature widgets in the bottom pane of the right panel, below the
        item table's draggable sash (only used when HAS_ITEM_PANEL_FOOTER)."""

    def build_menus(self, menubar):
        """Feature cascades for the window menu bar (optional). The menu bar is
        only attached when at least one cascade is added here."""

    def build_toolbar(self, parent):
        """Feature buttons for a horizontal toolbar across the top of the
        center panel (optional; only shown when widgets are added here)."""

    def bind_feature_keys(self):
        """Feature keyboard shortcuts (root.bind)."""

    def configure_item_tags(self, tree):
        """Configure semantic foreground tags on the item table."""

    def image_status_text(self, img_dict) -> str:
        return f"{img_dict['width']} × {img_dict['height']} px"

    def default_info_text(self) -> str:
        return "No marker selected."

    def update_counts(self):
        """Refresh feature count labels after image switch."""

    # ------------------------------------------------------------------
    # Hooks: interaction  (optional)
    # ------------------------------------------------------------------

    def init_selection_state(self):
        """Create feature selection-state attributes (no widgets exist yet)."""

    def reset_selection(self):
        """Clear feature selection state and dependent widget states."""

    def reset_for_new_image(self):
        """Reset feature state when switching images (default: selection only)."""
        self.reset_selection()

    def handle_press_override(self, event) -> bool:
        """Return True to consume a left-button press (modal tools)."""
        return False

    def handle_drag_override(self, event) -> bool:
        """Return True to consume a left-button drag (modal tools)."""
        return False

    def handle_release_override(self, event) -> bool:
        """Return True to consume a left-button release (modal tools)."""
        return False

    def on_scroll_override(self, event) -> bool:
        """Return True to consume a mousewheel event (else base zooms)."""
        return False

    def on_click(self, canvas_x, canvas_y):
        """Plain left click."""

    def on_shift_click(self, canvas_x, canvas_y):
        """Shift + left click."""

    def on_ctrl_click(self, canvas_x, canvas_y):
        """Ctrl + left click (no drag)."""

    def on_rubber_band(self, ix1, iy1, ix2, iy2, additive):
        """Rubber-band selection released; coords are image-space, ix1<=ix2,
        iy1<=iy2. additive=True when Shift was held."""

    # ------------------------------------------------------------------
    # Hooks: item table  (subclass MUST implement when ITEM_COLS is set)
    # ------------------------------------------------------------------

    def item_rows(self) -> list:
        """Return list of row dicts for the current image:
        {"iid": str, "values": tuple, "tag": semantic-tag str,
         "sort": {col: sortable value}, ...feature keys...}"""
        return []

    def item_panel_header_text(self) -> str:
        return ""

    def on_item_row_selected(self, row):
        """A table row was selected by the user; sync canvas selection."""

    def is_row_currently_selected(self, row) -> bool:
        """True when the row's item is already in the canvas selection
        (powers the Windows deferred-echo guard)."""
        return False

    def selected_row_iid(self) -> "str | None":
        """iid of the table row matching the canvas selection, or None."""
        return None

    def selection_center(self) -> "tuple | None":
        """(ix, iy) image coords of the selected item, or None."""
        return None

    # ------------------------------------------------------------------
    # Hooks: report  (subclass MUST implement)
    # ------------------------------------------------------------------

    def report_title(self) -> str:
        return "DEBUG REPORT"

    def report_constants_lines(self) -> list:
        """Lines describing detection constants in effect."""
        return []

    def report_body_lines(self) -> list:
        """Per-image sections + summary."""
        return []

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Window menu bar (feature cascades; only attached if populated)
        menubar = tk.Menu(self.root)
        self.build_menus(menubar)
        if menubar.index("end") is not None:
            self.root.config(menu=menubar)

        # PanedWindow splits left panel from right
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                               sashrelief=tk.RAISED, sashwidth=6)
        paned.pack(fill=tk.BOTH, expand=True)
        self.paned = paned

        # ---- LEFT PANEL ----
        left = tk.Frame(paned, width=220, bg="#484848")
        left.pack_propagate(False)
        paned.add(left, minsize=160, stretch="never")
        self.left_pane = left

        tk.Label(left, text="Images:", bg="#484848", fg="white",
                 font=("", 10, "bold")).pack(anchor="w", padx=6, pady=(6, 2))

        lb_frame = tk.Frame(left, bg="#484848")
        lb_frame.pack(fill=tk.BOTH, expand=True, padx=4)
        lb_sb = tk.Scrollbar(lb_frame, orient=tk.VERTICAL)
        self.lb_canvas = tk.Canvas(lb_frame, bg="#363636",
                                   yscrollcommand=lb_sb.set, highlightthickness=0)
        lb_sb.config(command=self.lb_canvas.yview)
        lb_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.lb_canvas.pack(fill=tk.BOTH, expand=True)
        self.lb_inner = tk.Frame(self.lb_canvas, bg="#363636")
        self.lb_canvas_window = self.lb_canvas.create_window(
            0, 0, anchor="nw", window=self.lb_inner)
        self.lb_inner.bind("<Configure>", self._on_lb_inner_configure)
        self.lb_inner.bind("<MouseWheel>", self._on_lb_scroll)
        self.lb_canvas.bind("<Configure>", self._on_lb_outer_configure)
        self.lb_canvas.bind("<MouseWheel>", self._on_lb_scroll)
        self.lb_rows = []        # Frame per image row
        self.lb_photos = []      # PhotoImage refs per row (prevent GC)

        self.build_left_controls(left)

        self.status_label = tk.Label(left, text="", bg="#484848", fg="#c0c0c0",
                                     font=("", 9), wraplength=200, justify=tk.LEFT)
        self.status_label.pack(anchor="w", padx=6, pady=2)

        self.build_left_info(left)

        # ---- CENTER PANEL (canvas) ----
        right = tk.Frame(paned, bg="#363636")
        paned.add(right, stretch="always")
        self.center_pane = right

        # ---- ITEM LIST PANEL ----
        item_pane = tk.Frame(paned, width=self.scaled(self.ITEM_PANEL_WIDTH),
                             bg="#484848")
        item_pane.pack_propagate(False)
        paned.add(item_pane, minsize=self.scaled(140), stretch="never")
        self.item_pane = item_pane
        self._build_item_list_panel(item_pane)

        # Optional feature toolbar across the top of the center panel
        toolbar = tk.Frame(right, bg="#3f3f3f")
        self.build_toolbar(toolbar)
        if toolbar.winfo_children():
            toolbar.pack(side=tk.TOP, fill=tk.X)

        # Canvas + scrollbars
        canvas_frame = tk.Frame(right, bg="#363636")
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="#363636",
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
        self.bind_key("<f>", lambda e: self._fit_to_window())
        self.bind_key("<F>", lambda e: self._fit_to_window())
        self.bind_key("<h>", lambda e: self._toggle_hide_markers())
        self.bind_key("<H>", lambda e: self._toggle_hide_markers())
        self.bind_key("<equal>", lambda e: self._zoom_step(2.0))    # + key (no shift)
        self.bind_key("<plus>", lambda e: self._zoom_step(2.0))     # + key (with shift)
        self.bind_key("<minus>", lambda e: self._zoom_step(0.5))
        pan_step = 80
        pan_big  = 300
        self.bind_key("<Left>",          lambda e: self._pan_by( pan_step, 0))
        self.bind_key("<Right>",         lambda e: self._pan_by(-pan_step, 0))
        self.bind_key("<Up>",            lambda e: self._pan_by(0,  pan_step))
        self.bind_key("<Down>",          lambda e: self._pan_by(0, -pan_step))
        self.bind_key("<Shift-Left>",    lambda e: self._pan_by( pan_big, 0))
        self.bind_key("<Shift-Right>",   lambda e: self._pan_by(-pan_big, 0))
        self.bind_key("<Shift-Up>",      lambda e: self._pan_by(0,  pan_big))
        self.bind_key("<Shift-Down>",    lambda e: self._pan_by(0, -pan_big))
        self.bind_key("<space>",         lambda e: self._nav_image(+1))
        self.bind_key("<b>",             lambda e: self._nav_image(-1))
        self.bind_key("<B>",             lambda e: self._nav_image(-1))
        self.bind_feature_keys()

        # Bottom panel
        bottom = tk.Frame(right, bg="#484848", height=190)
        bottom.pack(fill=tk.X)
        bottom.pack_propagate(False)

        # Scrollable info text
        info_frame = tk.Frame(bottom, bg="#484848")
        info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=6)
        tk.Label(info_frame, text="Selected:", bg="#484848", fg="#c0c0c0",
                 font=("", 9, "bold")).pack(anchor="w")
        info_text_wrap = tk.Frame(info_frame, bg="#484848")
        info_text_wrap.pack(fill=tk.BOTH, expand=True)
        info_sb = tk.Scrollbar(info_text_wrap, orient=tk.VERTICAL)
        self.info_text = tk.Text(info_text_wrap, bg="#363636", fg="white",
                                 font=("Courier", 9), wrap=tk.WORD,
                                 state=tk.DISABLED, height=7,
                                 yscrollcommand=info_sb.set,
                                 relief=tk.FLAT, padx=4, pady=4,
                                 cursor="arrow")
        info_sb.config(command=self.info_text.yview)
        info_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        self._set_info_text(self.default_info_text())

        # Note entry: free-text user annotation for the selected object.
        # Saved live on every keystroke into the per-image annotations JSON
        # (and the debug report) so the annotation session can be handed over
        # for algorithm tuning. Enter/Esc just return focus to the canvas.
        note_frame = tk.Frame(info_frame, bg="#484848")
        note_frame.pack(fill=tk.X, pady=(4, 0))
        tk.Label(note_frame, text="Note:", bg="#484848", fg="#c0c0c0",
                 font=("", 9, "bold")).pack(side=tk.LEFT)
        self.note_var = tk.StringVar()
        self._note_refreshing = False
        self.note_var.trace_add("write", lambda *args: self._on_note_changed())
        self.note_entry = tk.Entry(note_frame, textvariable=self.note_var,
                                   bg="#363636", fg="white",
                                   insertbackground="white",
                                   disabledbackground="#404040",
                                   relief=tk.FLAT, state=tk.DISABLED)
        self.note_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        self.note_entry.bind("<Return>", self._on_note_done)
        self.note_entry.bind("<Escape>", self._on_note_done)

        # Buttons (optional: UIs with a full menu bar + toolbar omit them)
        self.hide_btn = None
        if self.SHOW_BOTTOM_BUTTONS:
            btn_frame = tk.Frame(bottom, bg="#484848")
            btn_frame.pack(side=tk.RIGHT, padx=8, pady=6)

            btn_cfg = {"bg": "#585858", "fg": "white", "relief": tk.FLAT,
                       "padx": 8, "pady": 4, "width": 20}

            self.build_feature_buttons(btn_frame, btn_cfg)

            tk.Button(btn_frame, text="Clear Selection",
                      command=self._clear_selection, **btn_cfg).pack(fill=tk.X, pady=1)

            tk.Button(btn_frame, text="Fit to Window  (F)",
                      command=self._fit_to_window, **btn_cfg).pack(fill=tk.X, pady=(6, 1))

            self.hide_btn = tk.Button(btn_frame, text="Hide Markers  (H)",
                                      command=self._toggle_hide_markers, **btn_cfg)
            self.hide_btn.pack(fill=tk.X, pady=1)

    # ------------------------------------------------------------------
    # Key binding helper (focus-aware)
    # ------------------------------------------------------------------

    def _is_text_focus(self):
        w = self.root.focus_get()
        return isinstance(w, (tk.Entry, tk.Text))

    # Windows virtual-key codes for punctuation whose *keysym* changes under a
    # non-Latin layout (the physical key emits e.g. Cyrillic_ha instead of
    # bracketleft, so a keysym binding never fires). The VK code is positional,
    # so dispatching on it keeps the key working on every layout.
    _PUNCT_VK = {
        "bracketleft": 0xDB,   # VK_OEM_4  '['
        "bracketright": 0xDD,  # VK_OEM_6  ']'
    }

    @classmethod
    def _phys_keycode_of_sequence(cls, sequence):
        """Return (virtual-key code, ctrl) for a layout-dependent shortcut
        (``<f>`` / ``<Key-a>`` / ``<bracketleft>`` / ``<Control-c>`` ...), else
        ``(None, False)``. A ``Control-`` prefix routes the shortcut through the
        physical-key dispatcher too, so Ctrl+letter combos fire by key POSITION
        regardless of the active layout (a keysym ``<Control-c>`` binding never
        matches when the C key emits e.g. Cyrillic_es). Digits and layout-stable
        named keys (``<Left>``, ``<space>``, ...) return ``(None, False)`` and
        stay on the keysym path."""
        if not (sequence.startswith("<") and sequence.endswith(">")):
            return None, False
        body = sequence[1:-1]
        ctrl = False
        if body.startswith("Control-"):
            ctrl = True
            body = body[len("Control-"):]
        if body.startswith("Key-"):
            body = body[4:]
        if len(body) == 1 and body.isascii() and body.isalpha():
            # Windows reports event.keycode as the virtual-key code, which for
            # letters equals the uppercase ASCII code on every layout.
            return ord(body.upper()), ctrl
        return cls._PUNCT_VK.get(body), ctrl

    def bind_key(self, sequence, fn):
        """root.bind that ignores the key while a text-input widget has focus,
        so typing in the note entry doesn't trigger shortcuts.

        Shortcuts whose key position carries a *layout-dependent* keysym (plain
        letters like ``<f>``, and the ``[`` / ``]`` brackets) are routed through
        a physical-keycode dispatcher on Windows so they fire on the key's
        position regardless of the active input language (the user switches
        languages; a Cyrillic 'ф' on the F key, or 'х' on the '[' key, must
        still mean F / ``[``). Everything else — digits, arrows, modifier
        combos — keeps the normal keysym binding (those positions are stable).

        A ``Control-<letter>`` combo is ALSO routed through the physical-key
        dispatcher (its own keycode map), so it fires by key position on any
        layout — a plain keysym ``<Control-c>`` never matches once the C key
        emits a non-Latin keysym."""
        keycode, ctrl = self._phys_keycode_of_sequence(sequence)
        if keycode is not None and sys.platform == "win32":
            (self._phys_ctrl_keymap if ctrl else self._phys_keymap)[keycode] = fn
            return

        def handler(event):
            if self._is_text_focus():
                return
            fn(event)
        self.root.bind(sequence, handler)

    def _on_physical_key(self, event):
        """Dispatch a layout-independent letter shortcut by physical key.

        Fires only for keys with no more-specific keysym binding (Tk prefers
        the specific one), so it never double-triggers arrows/digits/etc.
        Modifier-aware: when Ctrl is held it dispatches ONLY from the Ctrl
        keymap (so e.g. Ctrl+C never falls through to the plain-C shortcut —
        the bug where a non-Latin layout cleared a correction on Ctrl+C)."""
        if self._is_text_focus():
            return
        if event.state & 0x4:                  # Control held
            fn = self._phys_ctrl_keymap.get(event.keycode)
            if fn is not None:
                fn(event)
            return                             # don't fall through to plain shortcut
        fn = self._phys_keymap.get(event.keycode)
        if fn is not None:
            fn(event)

    # ------------------------------------------------------------------
    # Note entry (per-object user annotation)
    # ------------------------------------------------------------------

    def get_selected_note(self) -> "str | None":
        """Note text of the currently selected object ("" when none stored),
        or None when no single annotatable object is selected.
        Subclasses override."""
        return None

    def set_selected_note(self, text):
        """Store the note text on the selected object (empty text removes it).
        Subclasses override."""

    NOTE_COLUMN = None   # item-table column id for the note marker (e.g. "nt")

    def _refresh_note_entry(self):
        self._note_refreshing = True
        try:
            note = self.get_selected_note()
            if note is None:
                self.note_var.set("")
                self.note_entry.config(state=tk.DISABLED)
            else:
                self.note_entry.config(state=tk.NORMAL)
                self.note_var.set(note)
        finally:
            self._note_refreshing = False

    def _on_note_changed(self):
        """Live-save the note on every keystroke."""
        if self._note_refreshing:
            return
        if self.get_selected_note() is None:
            return
        self.set_selected_note(self.note_var.get().strip())
        self._auto_save(self.images[self.current_idx]["stem"])
        self._update_note_marker()

    def _update_note_marker(self):
        """Refresh the selected row's note-marker cell in place (no full
        repopulate, so the user's column sort survives typing)."""
        col = self.NOTE_COLUMN
        if not col:
            return
        iid = self.selected_row_iid()
        if not iid or not self.item_tree.exists(iid):
            return
        marker = "✎" if self.get_selected_note() else ""
        self.item_tree.set(iid, col, marker)
        row = self.item_list_data.get(iid)
        if row is not None:
            row["sort"][col] = 1 if marker else 0

    def _on_note_done(self, event=None):
        """Enter/Esc in the note entry: hand focus back to the canvas so
        keyboard shortcuts work again (the note is already saved live)."""
        self.canvas.focus_set()

    # ------------------------------------------------------------------
    # Left-panel helpers for subclasses
    # ------------------------------------------------------------------

    def add_legend(self, parent, entries):
        """Pack a legend block: entries is a list of (symbol, color, label)."""
        tk.Label(parent, text="Legend:", bg="#484848", fg="#c0c0c0",
                 font=("", 8, "bold")).pack(anchor="w", padx=6, pady=(6, 1))
        for symbol, color, label in entries:
            row = tk.Frame(parent, bg="#484848")
            row.pack(anchor="w", padx=6)
            tk.Label(row, text=symbol, bg="#484848", fg=color,
                     font=("", 11)).pack(side=tk.LEFT, padx=(0, 4))
            tk.Label(row, text=label, bg="#484848", fg="#cccccc",
                     font=("", 8)).pack(side=tk.LEFT)

    def add_hints(self, parent, text):
        """Pack a monospace key/mouse hints block."""
        tk.Label(parent, text=text, bg="#484848", fg="white",
                 font=("Courier", 8), justify=tk.LEFT).pack(
            anchor="w", padx=6, pady=(8, 4))

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

    def _load_display_pil(self, img_dict) -> Image.Image:
        """Full-resolution PIL image for the main canvas.

        Default: open the on-disk ``image_path``. Subclasses that render
        their display live (no baked file) override this."""
        return Image.open(img_dict["image_path"])

    def _load_thumb_pil(self, img_dict) -> Image.Image:
        """PIL image for the sidebar thumbnail (resized by the caller).

        Runs on a BACKGROUND thread, so an override must not touch state the
        main thread mutates. Default: open the on-disk ``image_path``."""
        return Image.open(img_dict["image_path"])

    def _load_image_by_idx(self, idx):
        self.current_idx = idx
        img_dict = self.images[idx]

        try:
            self.pil_image = self._load_display_pil(img_dict)
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Cannot load image for {img_dict.get('stem', '?')}:\n{e}")
            return

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

        # Reset feature selection/tool state
        self.reset_for_new_image()
        self._set_info_text(self.default_info_text())

        # Update status
        self.status_label.config(text=self.image_status_text(img_dict))

        self._redraw()
        self.update_counts()
        self._populate_items_list()
        self._refresh_note_entry()

        # Highlight selected row and scroll to it
        self._highlight_lb_row(idx)

        # Different frames have different aspect ratios; reclaim pillarbox width.
        if self.REFLOW_TO_IMAGE:
            self.root.after(60, self._reflow_panes)

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
            resized = self._color_manage(resized)
            self.photo_image = ImageTk.PhotoImage(resized)
            draw_x = self.offset_x + vis_x1 * self.zoom
            draw_y = self.offset_y + vis_y1 * self.zoom
            self.canvas.create_image(draw_x, draw_y, anchor="nw",
                                     image=self.photo_image, tags="bg")

        self._redraw_markers()

    def _color_manage(self, pil):
        """sRGB preview bitmap -> monitor profile (matches darktable's
        color-managed view). No-op when color management is off or unavailable.
        The transform is built once and cached (None caches the unavailable
        case so detection runs at most once)."""
        if not getattr(self, "color_manage", True):
            return pil
        if not hasattr(self, "_cms_xform"):
            self._cms_xform = build_srgb_to_display_transform()
        if self._cms_xform is None:
            return pil
        try:
            from PIL import ImageCms
            return ImageCms.applyTransform(pil.convert("RGB"), self._cms_xform)
        except Exception:
            return pil

    def color_management_available(self):
        """True when a monitor ICC transform is in effect (drives the toggle UI)."""
        if not hasattr(self, "_cms_xform"):
            self._cms_xform = build_srgb_to_display_transform()
        return self._cms_xform is not None

    def _redraw_markers(self):
        self.canvas.delete(*self.overlay_tags(), "rubberband")
        if self.pil_image is None or self.hide_markers:
            return
        self.draw_overlays()

    # ------------------------------------------------------------------
    # Mouse events / zoom / pan
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
        if self.on_scroll_override(event):
            return

        # Normal zoom centered on mouse position
        if event.num == 4:
            delta = 1
        elif event.num == 5:
            delta = -1
        else:
            delta = 1 if event.delta > 0 else -1

        factor = 1.15 if delta > 0 else (1 / 1.15)
        new_zoom = max(0.05, min(20.0, self.zoom * factor))

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
        # Reclaim keyboard focus from the note entry so shortcuts work again
        self.canvas.focus_set()
        if self.handle_press_override(event):
            return
        self.drag_start = (event.x, event.y)
        self.is_dragging = False
        self.ctrl_drag_mode = bool(event.state & 0x4)

    def _on_left_drag(self, event):
        if self.drag_start is None:
            return

        if self.handle_drag_override(event):
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
        if self.handle_release_override(event):
            return

        if self.drag_start is None:
            return

        if self.ctrl_drag_mode:
            self.canvas.delete("rubberband")
            if self.is_dragging:
                self._zoom_to_rect(self.drag_start, (event.x, event.y))
            else:
                self.on_ctrl_click(event.x, event.y)
            self.drag_start = None
            self.is_dragging = False
            self.ctrl_drag_mode = False
            return

        shift_held = bool(event.state & 0x1)

        if self.is_dragging:
            # Rubber-band select
            x0, y0 = self.drag_start
            x1, y1 = event.x, event.y
            rx1, rx2 = min(x0, x1), max(x0, x1)
            ry1, ry2 = min(y0, y1), max(y0, y1)
            # Convert to image coords
            ix1, iy1 = canvas_to_image(rx1, ry1, self.offset_x, self.offset_y, self.zoom)
            ix2, iy2 = canvas_to_image(rx2, ry2, self.offset_x, self.offset_y, self.zoom)
            self.canvas.delete("rubberband")
            self.on_rubber_band(ix1, iy1, ix2, iy2, shift_held)
        else:
            if shift_held:
                self.on_shift_click(event.x, event.y)
            else:
                self.on_click(event.x, event.y)

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

    def _toggle_hide_markers(self):
        """Toggle visibility of all markers (H key)."""
        self.hide_markers = not self.hide_markers
        if self.hide_btn is not None:
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

    def _on_root_configure(self, event):
        """Window resized/maximized: debounce, then reflow panes + re-fit."""
        if event.widget is not self.root:
            return
        if self._reflow_after_id is not None:
            self.root.after_cancel(self._reflow_after_id)
        self._reflow_after_id = self.root.after(120, self._reflow_panes)

    def _reflow_panes(self):
        """Hand the right panel the horizontal space the image can't use.

        The image is fit to the canvas HEIGHT, so a frame whose aspect is
        taller than the center pane would otherwise sit in a too-wide canvas
        with black pillarbox bars on its sides. Instead shrink the center pane
        to the width the image actually needs and give the slack to the item /
        color-wheel panel. Only acts at fit zoom (when zoomed in the image
        fills the canvas, so there's no slack to reclaim)."""
        self._reflow_after_id = None
        if not (self.REFLOW_TO_IMAGE and self.at_fit_zoom
                and self.pil_image is not None):
            return
        img = self.images[self.current_idx]
        iw, ih = img["width"], img["height"]
        if iw <= 0 or ih <= 0:
            return
        try:
            total = self.paned.winfo_width()
            ch = self.canvas.winfo_height()
            left_w = self.left_pane.winfo_width()
        except tk.TclError:
            return                       # window torn down; nothing to do
        if total <= 1 or ch <= 1:
            return
        try:
            sash = int(self.paned.cget("sashwidth"))
        except Exception:
            sash = 6
        sashes = 2 * sash
        item_min = self.scaled(self.ITEM_PANEL_WIDTH)
        item_max = max(item_min, int(total * self.REFLOW_ITEM_MAX_FRAC))
        avail = total - left_w - sashes
        if avail <= item_min:
            return
        # width the image occupies fit to the available height (+ scrollbar pad)
        fit_w = iw * (ch / ih) + self.scaled(28)
        item_w = avail - fit_w
        item_w = int(max(item_min, min(item_max, item_w)))
        if abs(item_w - self.item_pane.winfo_width()) < self.scaled(8):
            return                       # already about right; avoid churn
        self.paned.paneconfigure(self.item_pane, width=item_w)
        # the center pane changed size -> canvas <Configure> re-fits the image

    def _center_canvas_on(self, ix, iy):
        """Pan canvas so image coordinates (ix, iy) appear at the canvas center."""
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        self.offset_x = cw / 2 - ix * self.zoom
        self.offset_y = ch / 2 - iy * self.zoom
        self.at_fit_zoom = False
        self._redraw()

    def _clear_selection(self):
        self.reset_selection()
        self._set_info_text(self.default_info_text())
        self._refresh_note_entry()
        self._redraw_markers()

    # ------------------------------------------------------------------
    # Thumbnail list
    # ------------------------------------------------------------------

    def _populate_thumb_list(self):
        self.lb_img_labels = []   # Label widget per row (holds thumb, created lazily)
        for i, img_dict in enumerate(self.images):
            row = tk.Frame(self.lb_inner, bg="#363636", cursor="hand2")
            row.pack(fill=tk.X, padx=2, pady=(2, 0))
            self.lb_photos.append(None)   # placeholder; filled in lazily

            def _bind_click(w, idx=i):
                w.bind("<Button-1>", lambda e: self._on_thumb_row_click(idx))
                w.bind("<MouseWheel>", self._on_lb_scroll)

            # Placeholder label (shows loading indicator, replaced by thumb later)
            lbl_img = tk.Label(row, text="…", bg="#363636", fg="#808080",
                               font=("", 8), cursor="hand2", anchor="center",
                               width=26, height=5)
            lbl_img.pack(fill=tk.X)
            _bind_click(lbl_img)
            self.lb_img_labels.append(lbl_img)

            lbl_txt = tk.Label(row, text=img_dict["stem"], bg="#363636",
                               fg="#cccccc", font=("", 8), wraplength=200,
                               cursor="hand2", anchor="w")
            lbl_txt.pack(fill=tk.X, padx=2, pady=(1, 3))
            _bind_click(lbl_txt)
            _bind_click(row)
            self.lb_rows.append(row)

        # Load thumbnails in background thread; main thread applies them
        self._thumb_queue = queue.Queue()
        self._thumb_thread = threading.Thread(
            target=self._thumb_loader_thread, daemon=True)
        self._thumb_thread.start()
        self.root.after(50, self._poll_thumb_queue)

    def _thumb_loader_thread(self):
        """Background: open + resize each image, push PIL Image to queue."""
        for i, img_dict in enumerate(self.images):
            try:
                pil_img = self._load_thumb_pil(img_dict)
                pil_img.thumbnail((210, 140), Image.LANCZOS)
            except Exception:
                pil_img = None
            self._thumb_queue.put((i, pil_img))

    def _poll_thumb_queue(self):
        """Main thread: drain queue, create PhotoImages, update labels."""
        try:
            while True:
                i, pil_img = self._thumb_queue.get_nowait()
                if pil_img is not None:
                    photo = ImageTk.PhotoImage(self._color_manage(pil_img))
                    self.lb_photos[i] = photo
                    self.lb_img_labels[i].config(image=photo, text="", width=0, height=0)
        except queue.Empty:
            pass
        if self._thumb_thread.is_alive() or not self._thumb_queue.empty():
            self.root.after(50, self._poll_thumb_queue)

    def _nav_image(self, delta):
        """Go to next (+1) or previous (-1) image."""
        new_idx = self.current_idx + delta
        if new_idx < 0 or new_idx >= len(self.images):
            return
        stem = self.images[self.current_idx]["stem"]
        self._auto_save(stem)
        self._load_image_by_idx(new_idx)

    def _on_thumb_row_click(self, idx):
        if idx == self.current_idx and self.pil_image is not None:
            return
        stem = self.images[self.current_idx]["stem"]
        self._auto_save(stem)
        self._load_image_by_idx(idx)

    def _highlight_lb_row(self, idx):
        sel_bg = "#3a5a8f"
        norm_bg = "#363636"
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
    # Item list panel  (ttk.Treeview table with sortable columns)
    # ------------------------------------------------------------------

    def _build_item_list_panel(self, parent):
        """Build the right-side item table panel (ttk.Treeview)."""
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Item.Treeview",
                        background="#363636", foreground="#cccccc",
                        fieldbackground="#363636", borderwidth=1,
                        font=("Courier", 8), rowheight=self.scaled(20))
        style.configure("Item.Treeview.Heading",
                        background="#484848", foreground="#c0c0c0",
                        font=("", 8, "bold"), relief="groove")
        style.map("Item.Treeview",
                  background=[("selected", "#3a5a8f")],
                  foreground=[("selected", "white")])

        # Optional vertical split: item table on top, feature footer below a
        # draggable sash. The table pane sits at its (compact) requested height
        # and the FOOTER stretches into the remaining space, so the feature
        # footer (e.g. the color wheels) expands rather than leaving a big empty
        # gap under a short table.
        if self.HAS_ITEM_PANEL_FOOTER:
            vpaned = tk.PanedWindow(parent, orient=tk.VERTICAL,
                                    sashrelief=tk.RAISED, sashwidth=6,
                                    bg="#484848")
            vpaned.pack(fill=tk.BOTH, expand=True)
            table_host = tk.Frame(vpaned, bg="#484848")
            footer_host = tk.Frame(vpaned, bg="#484848")
            vpaned.add(table_host, stretch="never", minsize=self.scaled(90))
            vpaned.add(footer_host, stretch="always", minsize=self.scaled(160))
            table_parent = table_host
        else:
            table_parent = parent
            footer_host = None

        tk.Label(table_parent, text=self.ITEM_PANEL_TITLE, bg="#484848",
                 fg="white", font=("", 10, "bold")).pack(anchor="w", padx=6,
                                                          pady=(6, 2))
        self.item_list_header = tk.Label(table_parent, text="", bg="#484848",
                                         fg="#c0c0c0", font=("", 9))
        self.item_list_header.pack(anchor="w", padx=6, pady=(0, 2))

        tree_frame = tk.Frame(table_parent, bg="#484848")
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 2))

        self.item_tree = ttk.Treeview(tree_frame, columns=self.ITEM_COLS,
                                      show="headings", style="Item.Treeview",
                                      selectmode="browse")
        self._sort_state = {}   # col -> currently_descending

        for col in self.ITEM_COLS:
            self.item_tree.heading(col, text=self.ITEM_HEADERS[col],
                                   command=lambda c=col: self._sort_column(c))
            self.item_tree.column(col, width=self.scaled(self.ITEM_WIDTHS[col]),
                                  anchor=self.ITEM_ANCHORS.get(col, "e"),
                                  stretch=(col in self.ITEM_STRETCH),
                                  minwidth=self.scaled(18))

        self.item_tree.tag_configure("even", background="#414141")
        # "odd" rows keep the default fieldbackground (#363636)
        self.configure_item_tags(self.item_tree)

        sb_y = tk.Scrollbar(tree_frame, orient=tk.VERTICAL,
                            command=self.item_tree.yview)
        self.item_tree.configure(yscrollcommand=sb_y.set)
        sb_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.item_tree.pack(fill=tk.BOTH, expand=True)

        self.item_tree.bind("<<TreeviewSelect>>", lambda *e: self._on_item_tree_select())

        self.center_btn = tk.Button(
            table_parent, text=self.CENTER_BUTTON_TEXT,
            command=self._center_on_selected_item,
            bg="#585858", fg="white", relief=tk.FLAT,
            padx=6, pady=3, state=tk.DISABLED)
        self.center_btn.pack(fill=tk.X, padx=4, pady=(2, 4))

        if footer_host is not None:
            self.build_item_panel_footer(footer_host)

    def _populate_items_list(self):
        """Rebuild the item table for the current image."""
        for iid in self.item_tree.get_children():
            self.item_tree.delete(iid)
        self.item_list_data = {}
        self._sort_state = {}
        for col in self.ITEM_COLS:
            self.item_tree.heading(col, text=self.ITEM_HEADERS[col])

        self.item_list_header.config(text=self.item_panel_header_text())

        n_rows = 0
        for row_num, row in enumerate(self.item_rows()):
            parity = "even" if row_num % 2 == 0 else "odd"
            self.item_tree.insert("", tk.END, iid=row["iid"], values=row["values"],
                                  tags=(row["tag"], parity))
            self.item_list_data[row["iid"]] = row
            n_rows = row_num + 1
        # Keep the table just tall enough for its rows. With a feature footer
        # the table pane is non-stretch, so this lets the footer (color wheels)
        # claim the remaining vertical space instead of a big empty gap.
        if self.HAS_ITEM_PANEL_FOOTER and n_rows:
            self.item_tree.configure(height=n_rows)

    def _sort_column(self, col):
        """Sort the treeview by the clicked column header; toggle asc/desc."""
        reverse = not self._sort_state.get(col, False)
        items = [(self.item_list_data[iid]["sort"][col], iid)
                 for iid in self.item_tree.get_children()]
        items.sort(reverse=reverse)
        for pos, (_key, iid) in enumerate(items):
            self.item_tree.move(iid, "", pos)
        self._sort_state[col] = reverse
        for c in self.ITEM_COLS:
            arrow = (" ▼" if reverse else " ▲") if c == col else ""
            self.item_tree.heading(c, text=self.ITEM_HEADERS[c] + arrow)
        self._reapply_row_parity()

    def _reapply_row_parity(self):
        """Reapply even/odd background tags after rows are reordered by sort."""
        for row_num, iid in enumerate(self.item_tree.get_children()):
            row = self.item_list_data.get(iid, {})
            parity = "even" if row_num % 2 == 0 else "odd"
            self.item_tree.item(iid, tags=(row.get("tag", ""), parity))

    def _on_item_tree_select(self):
        """Treeview row selected: sync canvas selection."""
        if self._syncing_selection:
            return
        sel = self.item_tree.selection()
        if not sel:
            return
        row = self.item_list_data.get(sel[0])
        if row is None:
            return

        # On Windows, <<TreeviewSelect>> fired by our own selection_set() in
        # _sync_item_list_selection is deferred and arrives after _syncing_selection
        # is already False.  If the treeview row's item is already present in the
        # current canvas selection, this event is that deferred echo — ignore it so
        # a rubber-band multi-selection isn't collapsed to one item.
        if self.is_row_currently_selected(row):
            return

        self.on_item_row_selected(row)

    def _center_on_selected_item(self):
        """Button action: pan canvas to center on the selected item."""
        center = self.selection_center()
        if center is not None:
            self._center_canvas_on(center[0], center[1])

    def _sync_item_list_selection(self):
        """Sync treeview selection to match current canvas selection state."""
        target_iid = self.selected_row_iid()
        self.center_btn.config(state=tk.NORMAL if target_iid else tk.DISABLED)

        self._syncing_selection = True
        try:
            current = self.item_tree.selection()
            if target_iid and self.item_tree.exists(target_iid):
                if not current or current[0] != target_iid:
                    self.item_tree.selection_set(target_iid)
                self.item_tree.see(target_iid)
            elif current:
                self.item_tree.selection_remove(*current)
        finally:
            self._syncing_selection = False

        self._refresh_note_entry()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _auto_save(self, stem):
        data = self.serialize_annotations(stem)
        ann_path = os.path.join(self.session_dir, f"{stem}{self.ANNOTATION_SUFFIX}")
        with open(ann_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_existing_annotations(self):
        """Load any previously saved annotation files."""
        for img in self.images:
            stem = img["stem"]
            ann_path = os.path.join(self.session_dir, f"{stem}{self.ANNOTATION_SUFFIX}")
            if not os.path.exists(ann_path):
                continue
            try:
                with open(ann_path, "r") as f:
                    data = json.load(f)
            except Exception:
                continue
            self.deserialize_annotations(img, data)

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
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            self.report_title(),
            f"Generated: {now}",
            f"Export dir: {self.session_dir}",
            f"Images processed: {len(self.images)}",
        ]
        if self.run_wall_time_s is not None:
            lines.append(f"Run wall time: {self.run_wall_time_s:.1f}s")
        lines.append("")
        lines.extend(self.report_constants_lines())
        lines.append("")
        lines.extend(self.report_body_lines())

        report_path = os.path.join(self.session_dir, self.REPORT_FILENAME)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Debug report written to: {report_path}")

    # ------------------------------------------------------------------
    # Entry point helper
    # ------------------------------------------------------------------

    @staticmethod
    def _enable_windows_dpi_awareness():
        """Tell Windows this process renders at native resolution, so it does
        NOT bitmap-stretch the window on a high-DPI (4K) display — that stretch
        is what makes the menus/text look blurry. Must run BEFORE the first
        Tk() call. No-op off Windows / when ctypes is unavailable."""
        if sys.platform != "win32":
            return
        try:
            import ctypes
            # Per-Monitor-v2 (crisp when dragged between monitors of different
            # DPI); fall back to per-monitor, then system-aware on older Windows.
            try:
                # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4, passed as a
                # pointer-sized handle (a bare negative int fails on 64-bit).
                ctypes.windll.user32.SetProcessDpiAwarenessContext(
                    ctypes.c_void_p(-4))
            except Exception:
                try:
                    ctypes.windll.shcore.SetProcessDpiAwareness(2)
                except Exception:
                    ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    @staticmethod
    def _apply_dpi_scaling(root):
        """Now that the window is DPI-aware (crisp but laid out in physical
        pixels, hence tiny on 4K), scale Tk's point->pixel factor to the real
        screen DPI so fonts/widgets keep a sane size. Tk's reference is 72 dpi."""
        try:
            dpi = root.winfo_fpixels("1i")
            if dpi and dpi > 0:
                root.tk.call("tk", "scaling", dpi / 72.0)
        except Exception:
            pass

    @classmethod
    def run_main(cls, usage="Usage: debug_ui.py <session_dir>"):
        if len(sys.argv) < 2:
            print(usage)
            sys.exit(1)

        session_dir = sys.argv[1]
        if not os.path.isdir(session_dir):
            print(f"Error: directory not found: {session_dir}")
            sys.exit(1)

        cls._enable_windows_dpi_awareness()
        root = tk.Tk()
        cls._apply_dpi_scaling(root)
        app = cls(root, session_dir)
        root.mainloop()
