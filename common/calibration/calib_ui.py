"""Feature-agnostic calibration MANAGER UI (tkinter).

This DRIVES the existing console runner — it never runs a calibration in-process.
Every real run is a subprocess of the feature's `run_calibration.py --config <file>`
(the same command the user types by hand), so the console runner stays the single
source of truth for the actual fitting. This module only adds a GUI on top:

  - browse / create / edit / duplicate / delete config JSONs in `configs/`
    (params shown WITH their schema docs + registry ranges/grid-steps),
  - a QUEUE of runs executed in separate processes with a user-set parallelism cap
    (when one finishes the next pending job starts automatically),
  - each running job's raw stdout streamed live into its own scrolled Text pane
    (no parsing, no progress bar — just the runner's own ticking output),
  - a results view: list finished sessions, read their report.md, and open the
    review debug UI (`run_calibration.py --review <session_dir>`).

Launched per feature by a thin script (e.g. `auto_negadoctor/tests/calibration_ui.py`)
that hands `run(adapter, script_path, configs_dir)` this feature's
`runner.CalibrationAdapter` + the path to its `run_calibration.py`. Everything the UI
needs (kinds, param ranges, param docs, rolls, calib dir, review) comes off that
adapter — the same surface the runner already uses.
"""
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

import markdown
from tkinterweb import HtmlFrame

# The fit methods the runner understands + which per-method hyperparameters (stored
# flat under `fit`) each exposes: (name, type, runner-default-for-the-tooltip). Grid's
# only knob is per-param grid_step (edited in the params table), so it has none here.
METHODS = ["none", "grid", "coordinate_descent", "random_search", "cmaes", "spsa"]
METHOD_PARAMS = {
    "none": [],
    "grid": [],
    "coordinate_descent": [("epsilon", float, 1e-4), ("max_iters", int, 20),
                           ("init_step", float, None), ("step_min", float, None),
                           ("step_shrink", float, 0.5)],
    "random_search": [("n_trials", int, 50), ("seed", int, 0)],
    "cmaes": [("sigma", float, 0.3), ("popsize", int, 0), ("max_iters", int, 30),
              ("seed", int, 0)],
    "spsa": [("a", float, 0.05), ("c", float, 0.1), ("alpha", float, 0.602),
             ("gamma", float, 0.101), ("A", float, None), ("max_iters", int, 100),
             ("seed", int, 0)],
}

# Per-kind free-form tolerance knobs the evaluators read (blank = omit → runner
# default). vignette has none.
TOLERANCE_FIELDS = {
    "inversion": [("clip_max_frac", float)],
    "crop": [("containment_weight", float)],
    "dust": [("w_fp", float), ("w_miss", float)],
    "stroke": [("w_fp", float), ("w_miss", float)],
}


def _conv(s, typ):
    """A trimmed entry string -> `typ` (int rounds a float literal), or None if blank."""
    s = s.strip()
    if s == "":
        return None
    return int(round(float(s))) if typ is int else float(s)


# ---------------------------------------------------------------------------
# Windows hi-DPI (mirrors common/debug_ui_base.py — without that fix the window
# is bitmap-stretched on a 4K/scaled display and fonts render blurry).
# ---------------------------------------------------------------------------

def _enable_windows_dpi_awareness():
    """Mark the process DPI-aware BEFORE the first Tk() so Windows renders at
    native resolution (no blurry bitmap stretch). No-op off Windows."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        try:
            # PER_MONITOR_AWARE_V2 (-4) as a pointer-sized handle.
            ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except Exception:
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def _screen_dpi(root):
    """Real screen DPI (96=100%, 144=150%, 192=200%). Prefer the authoritative
    Win32 value over Tk's winfo_fpixels (which many displays report wrong, making
    everything render physically tiny). CALIB_UI_DPI / DEBUG_UI_DPI override."""
    env = os.environ.get("CALIB_UI_DPI") or os.environ.get("DEBUG_UI_DPI")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    if sys.platform == "win32":
        try:
            import ctypes
            hwnd = root.winfo_id() if root is not None else 0
            dpi = 0
            if hwnd:
                try:
                    dpi = ctypes.windll.user32.GetDpiForWindow(hwnd)
                except Exception:
                    dpi = 0
            if not dpi:
                try:
                    dpi = ctypes.windll.user32.GetDpiForSystem()
                except Exception:
                    dpi = 0
            if dpi and dpi > 0:
                return float(dpi)
        except Exception:
            pass
    try:
        return float(root.winfo_fpixels("1i"))
    except Exception:
        return 96.0


def _apply_dpi_scaling(root):
    """Scale Tk's point->pixel factor to the real screen DPI (Tk's reference is
    72 dpi) so the DPI-aware window's widgets/fonts keep a sane size."""
    try:
        dpi = _screen_dpi(root)
        if dpi and dpi > 0:
            root.tk.call("tk", "scaling", dpi / 72.0)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Small tkinter helpers
# ---------------------------------------------------------------------------

class _Tip:
    """A hover tooltip for a widget (used to show each param's schema doc)."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text or ""
        self.tip = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _e):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 2
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify="left", background="#ffffe0",
                 relief="solid", borderwidth=1, wraplength=520).pack()

    def _hide(self, _e):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class _Scrollable(ttk.Frame):
    """A vertically scrollable frame. Add content to `.inner`."""

    def __init__(self, master):
        super().__init__(master)
        canvas = tk.Canvas(self, highlightthickness=0)
        vsb = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        self.inner = ttk.Frame(canvas)
        win = canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(win, width=e.width))

        def _wheel(e):
            canvas.yview_scroll(int(-e.delta / 120), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _wheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))


# ---------------------------------------------------------------------------
# Markdown viewer — real HTML rendering (python-markdown -> tkinterweb)
# ---------------------------------------------------------------------------

class _MarkdownView(ttk.Frame):
    """Renders the calibration report.md as HTML in a tkinterweb HtmlFrame (real
    headings / tables / code blocks / links, its own scrollbar). `python-markdown`
    does md->HTML; a small CSS styles it. Font sizes scale with the screen DPI."""

    def __init__(self, master):
        super().__init__(master)
        self.html = HtmlFrame(self, messages_enabled=False, vertical_scrollbar=True)
        self.html.pack(fill="both", expand=True)
        self._css = self._build_css(_screen_dpi(self) / 96.0)

    @staticmethod
    def _build_css(s):
        px = lambda v: max(1, round(v * s))   # noqa: E731 — DPI-scaled px
        return f"""
        body {{ font-family:'Segoe UI',sans-serif; font-size:{px(14)}px;
                color:#222; margin:{px(10)}px; }}
        h1 {{ color:#12325a; font-size:{px(24)}px;
              border-bottom:2px solid #d0d5dd; padding-bottom:{px(4)}px; }}
        h2 {{ color:#1a3d6d; font-size:{px(19)}px; margin-top:{px(16)}px; }}
        h3 {{ color:#26456e; font-size:{px(16)}px; }}
        code {{ background:#eef0f4; font-family:Consolas,monospace;
                font-size:{px(13)}px; padding:1px 3px; }}
        pre {{ background:#f5f6f8; padding:{px(8)}px; border:1px solid #e2e5ea; }}
        pre code {{ background:none; }}
        table {{ border-collapse:collapse; margin:{px(8)}px 0; }}
        th,td {{ border:1px solid #c8ccd4; padding:{px(3)}px {px(8)}px;
                 text-align:left; }}
        th {{ background:#eef2f8; color:#12325a; }}
        a {{ color:#2158c6; }}
        """

    def render(self, md):
        body = markdown.markdown(
            md or "", extensions=["tables", "fenced_code", "sane_lists"])
        self.html.load_html(
            f"<html><head><style>{self._css}</style></head>"
            f"<body>{body}</body></html>")


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

class Job:
    """One queued run of a config file. Normally a SAVED config; a `temp` one-off
    points at a throwaway file (deleted when the job is cleared / the app closes).
    The scrolled Text is the only output buffer — stdout is appended to it verbatim."""

    def __init__(self, job_id, config_path, kind, method, name=None, temp=False,
                 comment=""):
        self.id = job_id
        self.config_path = Path(config_path)
        self.name = name or self.config_path.stem
        self.kind = kind
        self.method = method
        self.comment = comment
        self.temp = temp           # throwaway one-off config -> delete when cleared
        self.state = "pending"     # pending|running|stopping|done|failed|stopped
        self.popen = None
        self.q = queue.Queue()
        self.eof = False
        self.text = None           # ScrolledText, created when the row is added
        self.session_dir = None
        self.start_time = None


# ---------------------------------------------------------------------------
# Manager window
# ---------------------------------------------------------------------------

class CalibManagerUI(tk.Tk):
    def __init__(self, adapter, script_path, configs_dir):
        super().__init__()
        _apply_dpi_scaling(self)   # crisp fonts on 4K/scaled Windows displays
        self._setup_table_style()
        self.adapter = adapter
        self.script_path = Path(script_path).resolve()
        self.configs_dir = Path(configs_dir)
        self.calib_dir = Path(adapter.calib_dir)
        self.kinds = list(adapter.evaluators)

        self.title(f"Calibration manager — {self.script_path.parent.parent.name}")
        # At least 2x the old 1200x780 default, clamped to the screen (minus a
        # small margin for the taskbar / window chrome).
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(2400, sw - 60)}x{min(1560, sh - 100)}")

        # queue state
        self.jobs = {}             # id -> Job
        self._job_seq = 0
        self.pending = []          # [Job]
        self.running = set()       # {Job}
        self.paused = False
        self.max_parallel = tk.IntVar(value=2)
        self._shown_text = None

        # editor state
        self.current_config_path = None
        self.param_rows = {}       # name -> {sel, lo, hi, step}
        self.param_defaults = {}   # name -> spec dict from the registry
        self.hyper_vars = {}
        self.tol_vars = {}

        self._roll_ids = self._discover_roll_ids()

        # Funnel icons for the header filter affordance (grey = filterable, blue =
        # a filter is active). Kept as attributes so Tk doesn't GC them to blank.
        self._funnel_dim = self._make_funnel("#9aa0aa")
        self._funnel_on = self._make_funnel("#2d6cdf")

        nb = ttk.Notebook(self)
        self.nb = nb
        nb.pack(fill="both", expand=True)
        self._build_config_tab(nb)
        self._build_queue_tab(nb)
        self._build_results_tab(nb)
        # A tree on a hidden tab has no geometry yet (bbox empty), so redraw the
        # column lines when its tab is first shown.
        nb.bind("<<NotebookTabChanged>>", lambda e: (
            self._redraw_lines(self.lib_tree),
            self._redraw_lines(self.queue_tree),
            self._redraw_lines(self.res_tree)))

        self._refresh_library()
        self._new_config()
        self._refresh_results()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(150, self._pump)

    # -- discovery ---------------------------------------------------------
    def _discover_roll_ids(self):
        try:
            return [r["id"] for r in self.adapter.discover_rolls()]
        except Exception as ex:   # noqa: BLE001 — discovery is best-effort
            print(f"roll discovery failed: {ex}")
            return []

    # -- table styling -----------------------------------------------------
    def _setup_table_style(self):
        """Taller rows (space between lines) + column separator lines in the
        tables. The native Windows ttk theme draws no gridlines, so switch to the
        'default' theme (same choice as the debug UI), which renders 1px column
        lines; row striping adds horizontal separation."""
        factor = _screen_dpi(self) / 96.0
        self._row_h = max(20, int(round(28 * factor)))
        style = ttk.Style()
        try:
            style.theme_use("default")
        except tk.TclError:
            pass
        style.configure("Treeview", rowheight=self._row_h,
                        borderwidth=1, relief="solid")
        style.configure("Treeview.Heading", relief="groove", borderwidth=1,
                        padding=4)
        # Heading background so the overlaid funnel buttons blend into the header.
        self._heading_bg = style.lookup("Treeview.Heading", "background") or "#d9d9d9"

    def _config_stripes(self, tree):
        tree.tag_configure("oddrow", background="#eef1f6")
        tree.tag_configure("evenrow", background="#ffffff")

    def _add_column_lines(self, tree, color="#b4b8c2"):
        """ttk.Treeview draws no vertical body separators in any stock theme, so
        overlay a 1px line at each column's real left edge (from `bbox`, so it
        tracks column stretch / resize / heading drags). Redrawn on resize, after
        a column drag, and whenever the table is repopulated (via
        `tree._col_line_redraw`)."""
        tree._sep_lines = []

        def redraw(_=None):
            for f in tree._sep_lines:
                f.destroy()
            tree._sep_lines = []
            items = tree.get_children()
            if not items:
                return
            first = items[0]
            for col in tree["columns"]:      # left edge of every data column
                bb = tree.bbox(first, col)
                if not bb:
                    continue
                line = tk.Frame(tree, width=1, background=color)
                line.place(x=bb[0], y=0, relheight=1.0)
                tree._sep_lines.append(line)
            self._position_header_funnels(tree, first)

        tree._col_line_redraw = redraw
        tree.bind("<Configure>", lambda e: tree.after_idle(redraw), add="+")
        tree.bind("<ButtonRelease-1>", lambda e: tree.after_idle(redraw), add="+")
        tree.after(200, redraw)

    def _redraw_lines(self, tree):
        fn = getattr(tree, "_col_line_redraw", None)
        if fn:
            tree.after_idle(fn)

    def _attach_cell_tooltip(self, tree, column):
        """Show the FULL value of `column` for the hovered row in a tooltip (the
        visible cell is narrow, so long comments are otherwise truncated)."""
        tree._celltip = {"win": None, "row": None}
        tree.bind("<Motion>",
                  lambda e: self._cell_tooltip_motion(tree, column, e), add="+")
        tree.bind("<Leave>", lambda e: self._cell_tooltip_hide(tree), add="+")

    def _cell_tooltip_hide(self, tree):
        st = tree._celltip
        if st["win"]:
            st["win"].destroy()
        st["win"], st["row"] = None, None

    def _cell_tooltip_motion(self, tree, column, e):
        st = tree._celltip
        row = tree.identify_row(e.y)
        cols = tuple(tree["columns"])
        colid = tree.identify_column(e.x)
        ok = (row and tree.identify_region(e.x, e.y) == "cell"
              and colid[1:].isdigit() and 0 <= int(colid[1:]) - 1 < len(cols)
              and cols[int(colid[1:]) - 1] == column)
        if not ok:
            self._cell_tooltip_hide(tree)
            return
        text = tree.set(row, column)
        if not text:
            self._cell_tooltip_hide(tree)
            return
        pos = f"+{e.x_root + 16}+{e.y_root + 14}"
        if st["row"] == row and st["win"]:
            st["win"].wm_geometry(pos)
            return
        self._cell_tooltip_hide(tree)
        st["row"] = row
        w = tk.Toplevel(tree)
        w.wm_overrideredirect(True)
        w.wm_geometry(pos)
        tk.Label(w, text=text, justify="left", background="#ffffe0",
                 relief="solid", borderwidth=1, wraplength=560).pack()
        st["win"] = w

    def _position_header_funnels(self, tree, first=None):
        """Place a clickable funnel button at each column's header right edge (the
        filter affordance). Persistent widgets repositioned on every line redraw so
        they track resize / column drags / repopulation; blue when a filter is
        active on that column."""
        if first is None:
            kids = tree.get_children()
            first = kids[0] if kids else None
        if first is None:
            return
        if not hasattr(tree, "_hdr_btns"):
            tree._hdr_btns = {}
        cols = tuple(tree["columns"])
        anchor = cols[0] if cols else "#0"
        b0 = tree.bbox(first, anchor)
        hdr_h = b0[1] if b0 else 20

        def right_edge(col):
            if col == "#0":
                nb = tree.bbox(first, cols[0]) if cols else None
                return nb[0] if nb else None
            bb = tree.bbox(first, col)
            return (bb[0] + bb[2]) if bb else None

        for col in ("#0",) + cols:
            right = right_edge(col)
            if right is None:
                continue
            btn = tree._hdr_btns.get(col)
            if btn is None:
                btn = tk.Label(tree, cursor="hand2", borderwidth=0, padx=4, pady=3,
                               background=self._heading_bg)
                btn.bind("<Button-1>",
                         lambda e, c=col, t=tree: (self._open_filter(t, c), "break")[1])
                tree._hdr_btns[col] = btn
            active = tree._filters.get(col) is not None
            btn.configure(image=(self._funnel_on if active else self._funnel_dim))
            btn.update_idletasks()
            bw, bh = btn.winfo_reqwidth(), btn.winfo_reqheight()
            # Never let the button spill out of the (short) header row.
            ph = min(bh, max(1, hdr_h - 2))
            btn.place(x=max(0, right - bw - 1), y=max(0, (hdr_h - ph) // 2),
                      height=ph)

    # -- sortable + filterable columns -------------------------------------
    def _make_funnel(self, color):
        """A funnel (filter) icon as a transparent PhotoImage. A transparent MARGIN
        is baked in (the whole label is clickable), and it DPI-scales up, so it is
        an easy click target regardless of tk's padding quirks."""
        pat = ["1111111111111",
               "0111111111110",
               "0011111111100",
               "0001111111000",
               "0000111110000",
               "0000011100000",
               "0000011100000",
               "0000011100000",
               "0000011100000",
               "0000011100000"]
        # Wide transparent side margins grow the hit area WITHOUT making the icon
        # taller than the (short) header row; keep vertical margin minimal.
        mx, my = 7, 1
        w, h = len(pat[0]), len(pat)
        img = tk.PhotoImage(width=w + 2 * mx, height=h + 2 * my)
        for y, row in enumerate(pat):
            for x, ch in enumerate(row):
                if ch == "1":
                    img.put(color, (x + mx, y + my))
        scale = 2 if _screen_dpi(self) >= 140 else 1
        return img.zoom(scale) if scale > 1 else img

    def _setup_filterable(self, tree):
        """Left-click a header to SORT by it (toggles asc/desc); right-click to open
        a distinct-value FILTER. The tree is data-driven: `_set_rows` holds the
        master rows, `_render_rows` shows those passing the active filters in the
        current sort order. A funnel icon marks the filter affordance."""
        tree._filters = {}        # col -> set(allowed str values); absent = show all
        tree._rows = []           # master data: [(iid, text, values_tuple)]
        tree._sort = None         # (col, descending) or None
        tree._heading_base = {}
        for col in ("#0",) + tuple(tree["columns"]):
            tree._heading_base[col] = tree.heading(col, "text")
            tree.heading(col, command=lambda c=col, t=tree: self._sort_by(t, c))
        tree.bind("<Button-3>", lambda e, t=tree: self._on_header_rclick(t, e))
        self._update_headings(tree)

    def _update_headings(self, tree):
        """Heading text carries the 3-state sort arrow (▲ asc / ▼ desc / none). The
        clickable funnel filter icon is a separate overlaid button (see
        `_position_header_funnels`)."""
        srt = getattr(tree, "_sort", None)
        for col, base in tree._heading_base.items():
            arrow = ""
            if srt and srt[0] == col:
                arrow = " ▼" if srt[1] else " ▲"
            tree.heading(col, text=base + arrow)

    @staticmethod
    def _sort_key(val):
        s = str(val)
        try:
            return (0, float(s))     # numbers sort numerically, before text
        except ValueError:
            return (1, s.lower())

    def _apply_sort(self, tree):
        """Rebuild the working rows from the NATURAL order, then apply the current
        sort (if any) — so cycling sort OFF restores the original row order."""
        rows = list(getattr(tree, "_rows_natural", tree._rows))
        srt = getattr(tree, "_sort", None)
        if srt:
            col, desc = srt
            rows.sort(
                key=lambda r: self._sort_key(self._cell_value(tree, col, r[1], r[2])),
                reverse=desc)
        tree._rows = rows

    def _sort_by(self, tree, col):
        # Three-state cycle: unsorted -> ascending -> descending -> unsorted.
        srt = getattr(tree, "_sort", None)
        if not srt or srt[0] != col:
            tree._sort = (col, False)       # asc
        elif srt[1] is False:
            tree._sort = (col, True)        # desc
        else:
            tree._sort = None               # back to natural order
        self._apply_sort(tree)
        self._render_rows(tree)

    def _on_header_rclick(self, tree, event):
        if tree.identify_region(event.x, event.y) != "heading":
            return
        colid = tree.identify_column(event.x)   # '#0', '#1', ...
        if colid == "#0":
            col = "#0"
        else:
            cols = tuple(tree["columns"])
            idx = int(colid[1:]) - 1
            col = cols[idx] if 0 <= idx < len(cols) else None
        if col is not None:
            self._open_filter(tree, col)

    @staticmethod
    def _cell_value(tree, col, text, values):
        if col == "#0":
            return text
        cols = tuple(tree["columns"])
        return values[cols.index(col)] if col in cols else ""

    def _set_rows(self, tree, rows):
        tree._rows_natural = list(rows)   # the order the data was provided in
        self._apply_sort(tree)            # keep the active sort across repopulates
        self._render_rows(tree)

    def _row_passes(self, tree, text, values):
        for col, allowed in tree._filters.items():
            if allowed is not None and str(
                    self._cell_value(tree, col, text, values)) not in allowed:
                return False
        return True

    def _render_rows(self, tree):
        sel = set(tree.selection())
        tree.delete(*tree.get_children())
        i = 0
        for iid, text, values in tree._rows:
            if not self._row_passes(tree, text, values):
                continue
            tree.insert("", "end", iid=iid, text=text, values=values,
                        tags=self._stripe(i))
            i += 1
        keep = [s for s in sel if tree.exists(s)]
        if keep:
            tree.selection_set(keep)
        self._update_headings(tree)
        self._redraw_lines(tree)

    def _open_filter(self, tree, col):
        vals, seen = [], set()
        for iid, text, values in tree._rows:
            s = str(self._cell_value(tree, col, text, values))
            if s not in seen:
                seen.add(s)
                vals.append(s)
        vals.sort()
        current = tree._filters.get(col)
        win = tk.Toplevel(self)
        win.title(f"Filter: {tree._heading_base.get(col, col)}")
        win.transient(self)
        win.geometry(f"240x320+{self.winfo_pointerx()}+{self.winfo_pointery()}")
        checks = {}

        def apply_live():
            # Live: the table updates the instant a value is ticked/unticked.
            chosen = {v for v, var in checks.items() if var.get()}
            if len(chosen) == len(vals):
                tree._filters.pop(col, None)   # everything shown = no filter
            else:
                tree._filters[col] = chosen
            self._render_rows(tree)

        def set_all(state):
            for var in checks.values():
                var.set(state)
            apply_live()

        # Button bar pinned to the BOTTOM first, so a tall value list (its own
        # scroll) can never starve/hide these controls.
        bar = ttk.Frame(win)
        bar.pack(side="bottom", fill="x", padx=6, pady=4)
        ttk.Button(bar, text="All", width=5,
                   command=lambda: set_all(True)).pack(side="left")
        ttk.Button(bar, text="None", width=5,
                   command=lambda: set_all(False)).pack(side="left")
        ttk.Button(bar, text="Close", command=win.destroy).pack(side="right")

        box = _Scrollable(win)
        box.pack(fill="both", expand=True, padx=6, pady=4)
        for v in vals:
            var = tk.BooleanVar(value=(current is None or v in current))
            checks[v] = var
            ttk.Checkbutton(box.inner, text=(v if v != "" else "(blank)"),
                            variable=var, command=apply_live).pack(anchor="w")

    @staticmethod
    def _stripe(i):
        return ("evenrow",) if i % 2 == 0 else ("oddrow",)

    def _doc_for(self, name):
        base = name.split("[")[0]
        fields = getattr(self.adapter.schema, "FIELDS", {})
        f = fields.get(base)
        return getattr(f, "doc", "") if f else ""

    def _default_value(self, name):
        try:
            return self.adapter.registry.current(name)
        except Exception:         # noqa: BLE001
            spec = self.param_defaults.get(name, {})
            lo, hi = spec.get("range", [0.0, 1.0])
            return (lo + hi) / 2.0

    # =====================================================================
    # Config tab
    # =====================================================================
    def _build_config_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="Configs")
        pw = ttk.PanedWindow(tab, orient="horizontal")
        pw.pack(fill="both", expand=True)

        # -- library --------------------------------------------------------
        left = ttk.Frame(pw)
        pw.add(left, weight=1)
        ttk.Label(left, text="Config library", font=("", 10, "bold")).pack(anchor="w")
        ttk.Label(left, text="header: sort (asc/desc/off) · funnel ⧩ or right-click: filter",
                  foreground="#777").pack(anchor="w")
        # Buttons pinned to the bottom BEFORE the tree, so a short window starves
        # the (scrollable) tree instead of hiding these buttons.
        libbtns = ttk.Frame(left)
        libbtns.pack(side="bottom", fill="x", pady=4)
        ttk.Button(libbtns, text="New", command=self._new_config).pack(side="left")
        ttk.Button(libbtns, text="Duplicate", command=self._duplicate_selected_config).pack(side="left")
        ttk.Button(libbtns, text="Delete", command=self._delete_selected_config).pack(side="left")
        # Selecting a config loads it into the editor (no Edit button needed); the
        # editor's single "Enqueue ▶" queues it (saved as-is, or a one-off if tweaked).
        self.lib_tree = ttk.Treeview(left, columns=("kind", "method"),
                                     show="tree headings", height=20)
        self.lib_tree.heading("#0", text="file")
        self.lib_tree.heading("kind", text="kind")
        self.lib_tree.heading("method", text="method")
        self.lib_tree.column("#0", width=200)
        self.lib_tree.column("kind", width=80)
        self.lib_tree.column("method", width=120)
        self._config_stripes(self.lib_tree)
        self._add_column_lines(self.lib_tree)
        self._setup_filterable(self.lib_tree)
        self.lib_tree.pack(fill="both", expand=True)
        self.lib_tree.bind("<<TreeviewSelect>>", lambda e: self._edit_selected_config())

        # -- editor ---------------------------------------------------------
        right = ttk.Frame(pw)
        pw.add(right, weight=3)
        # The editor is NOT wrapped in an outer scroll: the fixed controls sit at
        # the top, the action bar is pinned to the bottom, and the params table
        # (which has its OWN inner scroll) expands to fill the middle — so a tall
        # window shows no dead grey space below the editor.
        body = ttk.Frame(right)
        body.pack(fill="both", expand=True)

        # Action bar FIRST, pinned to the bottom, so a short window starves the
        # expandable params table (it has its own scroll) instead of these buttons.
        act = ttk.Frame(body)
        act.pack(side="bottom", fill="x", pady=6)
        ttk.Button(act, text="Save", command=self._save).pack(side="left")
        ttk.Button(act, text="Save As…", command=self._save_as).pack(side="left")
        enq = ttk.Button(act, text="Enqueue ▶", command=self._enqueue_editor)
        enq.pack(side="left", padx=8)
        _Tip(enq, "Unchanged saved config → queued as-is (reproducible). Unsaved "
                  "edits → a one-off run (NOT saved to the library).")
        self.editor_path_lbl = ttk.Label(act, text="(unsaved)", foreground="#555")
        self.editor_path_lbl.pack(side="left", padx=8)

        top = ttk.Frame(body)
        top.pack(fill="x", pady=2)
        ttk.Label(top, text="kind").grid(row=0, column=0, sticky="w")
        self.kind_var = tk.StringVar(value=self.kinds[0])
        kind_cb = ttk.Combobox(top, textvariable=self.kind_var, values=self.kinds,
                               state="readonly", width=16)
        kind_cb.grid(row=0, column=1, sticky="w", padx=4)
        kind_cb.bind("<<ComboboxSelected>>", lambda e: self._on_kind_selected())
        self.metric_lbl = ttk.Label(top, text="", foreground="#555")
        self.metric_lbl.grid(row=0, column=2, sticky="w", padx=8)

        ttk.Label(top, text="method").grid(row=1, column=0, sticky="w")
        self.method_var = tk.StringVar(value="none")
        method_cb = ttk.Combobox(top, textvariable=self.method_var, values=METHODS,
                                 state="readonly", width=16)
        method_cb.grid(row=1, column=1, sticky="w", padx=4)
        method_cb.bind("<<ComboboxSelected>>", lambda e: self._rebuild_hyper())

        ttk.Label(top, text="comment").grid(row=2, column=0, sticky="w")
        self.comment_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.comment_var, width=60).grid(
            row=2, column=1, columnspan=2, sticky="we", padx=4)

        # method hyperparameters (rebuilt on method change)
        hy = ttk.LabelFrame(body, text="method parameters")
        hy.pack(fill="x", pady=4)
        self.hyper_frame = ttk.Frame(hy)
        self.hyper_frame.pack(fill="x")

        # advanced fit toggles
        adv = ttk.LabelFrame(body, text="advanced")
        adv.pack(fill="x", pady=4)
        self.pca_var = tk.BooleanVar()
        self.cv_var = tk.BooleanVar()
        self.cvfinal_var = tk.BooleanVar()
        ttk.Checkbutton(adv, text="pca (curvature at optimum)",
                        variable=self.pca_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(adv, text="cross-validate (leave-one-roll-out)",
                        variable=self.cv_var).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(adv, text="cv final fit (adoptable preset)",
                        variable=self.cvfinal_var).grid(row=0, column=2, sticky="w")
        ttk.Label(adv, text="downsample").grid(row=1, column=0, sticky="w")
        self.downsample_var = tk.StringVar()
        ttk.Entry(adv, textvariable=self.downsample_var, width=8).grid(
            row=1, column=1, sticky="w")
        wlbl = ttk.Label(adv, text="num workers")
        wlbl.grid(row=1, column=2, sticky="e")
        self.workers_var = tk.StringVar()
        we = ttk.Entry(adv, textvariable=self.workers_var, width=8)
        we.grid(row=1, column=3, sticky="w", padx=(4, 0))
        _Tip(we, "Parallel trial threads for random_search / cmaes / spsa "
                 "(blank/0 = auto; 1 = serial). No effect for coordinate_descent.")

        # tolerances (kind-specific, rebuilt on kind change)
        tolf = ttk.LabelFrame(body, text="tolerances")
        tolf.pack(fill="x", pady=4)
        self.tol_frame = ttk.Frame(tolf)
        self.tol_frame.pack(fill="x")

        # rolls
        rollf = ttk.LabelFrame(body, text="rolls (none selected = all)")
        rollf.pack(fill="x", pady=4)
        self.roll_list = tk.Listbox(rollf, selectmode="extended", height=5,
                                    exportselection=False)
        for rid in self._roll_ids:
            self.roll_list.insert("end", rid)
        self.roll_list.pack(side="left", fill="x", expand=True)
        rr = ttk.Frame(rollf)
        rr.pack(side="left", fill="y", padx=6)
        ttk.Label(rr, text="extra ids\n(comma/space)").pack(anchor="w")
        self.roll_extra_var = tk.StringVar()
        ttk.Entry(rr, textvariable=self.roll_extra_var, width=20).pack()

        # params
        pf = ttk.LabelFrame(body, text="fittable params")
        pf.pack(fill="both", expand=True, pady=4)
        head = ttk.Frame(pf)
        head.pack(fill="x")
        self.all_params_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(head, text="fit ALL params of this kind",
                        variable=self.all_params_var,
                        command=self._on_all_toggle).pack(side="left")
        ttk.Label(head, text="(else tick individual params below; edit lo/hi/step "
                             "to override the registry range)",
                  foreground="#555").pack(side="left", padx=6)
        self.params_holder = ttk.Frame(pf)
        self.params_holder.pack(fill="both", expand=True)

    # -- editor: kind/method/params rebuild --------------------------------
    def _on_kind_selected(self):
        self._rebuild_tolerances()
        self._rebuild_params()
        self.metric_lbl.config(
            text=self.adapter.metric_name.get(self.kind_var.get(), ""))

    def _rebuild_hyper(self):
        for w in self.hyper_frame.winfo_children():
            w.destroy()
        self.hyper_vars = {}
        params = METHOD_PARAMS.get(self.method_var.get(), [])
        if not params:
            ttk.Label(self.hyper_frame, text="(no method parameters)",
                      foreground="#555").grid(row=0, column=0, sticky="w")
            return
        for i, (pname, _typ, default) in enumerate(params):
            ttk.Label(self.hyper_frame, text=pname).grid(
                row=i, column=0, sticky="w", padx=(0, 4))
            var = tk.StringVar()
            e = ttk.Entry(self.hyper_frame, textvariable=var, width=12)
            e.grid(row=i, column=1, sticky="w")
            if default is not None:
                _Tip(e, f"runner default: {default}")
            self.hyper_vars[pname] = var

    def _rebuild_tolerances(self):
        for w in self.tol_frame.winfo_children():
            w.destroy()
        self.tol_vars = {}
        fields = TOLERANCE_FIELDS.get(self.kind_var.get(), [])
        if not fields:
            ttk.Label(self.tol_frame, text="(none for this kind)",
                      foreground="#555").grid(row=0, column=0, sticky="w")
            return
        for i, (tname, _typ) in enumerate(fields):
            ttk.Label(self.tol_frame, text=tname).grid(
                row=i, column=0, sticky="w", padx=(0, 4))
            var = tk.StringVar()
            ttk.Entry(self.tol_frame, textvariable=var, width=12).grid(
                row=i, column=1, sticky="w")
            self.tol_vars[tname] = var

    def _rebuild_params(self):
        for w in self.params_holder.winfo_children():
            w.destroy()
        self.param_rows = {}
        self.param_defaults = self.adapter.registry.fittable(self.kind_var.get())
        scroll = _Scrollable(self.params_holder)
        scroll.pack(fill="both", expand=True)
        grid = scroll.inner
        for col, txt in enumerate(("param", "default", "lo", "hi", "grid_step")):
            ttk.Label(grid, text=txt, font=("", 9, "bold")).grid(
                row=0, column=col + 1, sticky="w", padx=2)
        for i, name in enumerate(sorted(self.param_defaults), start=1):
            spec = self.param_defaults[name]
            lo, hi = spec.get("range", [0.0, 1.0])
            step = spec.get("grid_step", "")
            sel = tk.BooleanVar()
            ttk.Checkbutton(grid, variable=sel).grid(row=i, column=0)
            lbl = ttk.Label(grid, text=name)
            lbl.grid(row=i, column=1, sticky="w", padx=2)
            _Tip(lbl, self._doc_for(name))
            ttk.Label(grid, text=f"{self._default_value(name):g}",
                      foreground="#357").grid(row=i, column=2, sticky="w", padx=2)
            lo_v, hi_v, st_v = tk.StringVar(value=f"{lo:g}"), \
                tk.StringVar(value=f"{hi:g}"), tk.StringVar(value=f"{step:g}")
            ttk.Entry(grid, textvariable=lo_v, width=10).grid(row=i, column=3, padx=1)
            ttk.Entry(grid, textvariable=hi_v, width=10).grid(row=i, column=4, padx=1)
            ttk.Entry(grid, textvariable=st_v, width=10).grid(row=i, column=5, padx=1)
            self.param_rows[name] = {"sel": sel, "lo": lo_v, "hi": hi_v, "step": st_v}
        self._update_params_enabled()

    def _on_all_toggle(self):
        # USER click on the "fit ALL" checkbox. Switching it OFF pre-selects EVERY
        # param (so per-param mode starts equivalent to "all" — the user then just
        # deselects the few they don't want, instead of ticking dozens by hand).
        if not self.all_params_var.get():
            for row in self.param_rows.values():
                row["sel"].set(True)
        self._update_params_enabled()

    def _update_params_enabled(self):
        # When "fit ALL" is on the individual rows are ignored (config uses "all");
        # hint that with a cursor rather than toggling ~600 widgets' state.
        inert = self.all_params_var.get()
        self.params_holder.configure(cursor="X_cursor" if inert else "")

    # -- editor <-> config -------------------------------------------------
    def _editor_to_config(self):
        kind = self.kind_var.get()
        method = self.method_var.get()
        fit = {"method": method}
        if self.all_params_var.get():
            fit["params"] = "all"
        else:
            params = {}
            for name, row in self.param_rows.items():
                if not row["sel"].get():
                    continue
                ov = {}
                d = self.param_defaults[name]
                lo, hi = _conv(row["lo"].get(), float), _conv(row["hi"].get(), float)
                step = _conv(row["step"].get(), float)
                dr = d.get("range")
                if lo is not None and hi is not None and (dr is None
                        or [lo, hi] != list(dr)):
                    ov["range"] = [lo, hi]
                if step is not None and step != d.get("grid_step"):
                    ov["grid_step"] = step
                params[name] = ov
            fit["params"] = params
        for pname, ptype, _d in METHOD_PARAMS.get(method, []):
            v = _conv(self.hyper_vars[pname].get(), ptype)
            if v is not None:
                fit[pname] = v
        if self.pca_var.get():
            fit["pca"] = True
        if self.cv_var.get():
            fit["cross_validate"] = True
        if self.cvfinal_var.get():
            fit["cv_final_fit"] = True
        ds = _conv(self.downsample_var.get(), int)
        if ds is not None:
            fit["downsample"] = ds
        workers = _conv(self.workers_var.get(), int)
        if workers is not None:
            fit["workers"] = workers
        tol = {}
        for tname, ttype in TOLERANCE_FIELDS.get(kind, []):
            v = _conv(self.tol_vars[tname].get(), ttype)
            if v is not None:
                tol[tname] = v
        if tol:
            fit["tolerances"] = tol
        return {"kind": kind, "rolls": self._selected_rolls(),
                "comment": self.comment_var.get().strip(), "fit": fit}

    def _selected_rolls(self):
        ids = [self.roll_list.get(i) for i in self.roll_list.curselection()]
        extra = self.roll_extra_var.get().replace(",", " ").split()
        seen, out = set(), []
        for rid in ids + extra:
            if rid and rid not in seen:
                seen.add(rid)
                out.append(rid)
        return out

    def _load_config_into_editor(self, cfg, path=None):
        fit = cfg.get("fit", {}) or {}
        self.kind_var.set(cfg.get("kind", self.kinds[0]))
        self._on_kind_selected()
        self.method_var.set(fit.get("method", "none"))
        self._rebuild_hyper()
        self.comment_var.set(cfg.get("comment", ""))
        # params
        params = fit.get("params", "all")
        if params in ("all", "*"):
            self.all_params_var.set(True)
        else:
            self.all_params_var.set(False)
            for name, ov in (params or {}).items():
                row = self.param_rows.get(name)
                if not row:
                    continue
                row["sel"].set(True)
                ov = ov or {}
                if "range" in ov:
                    row["lo"].set(f"{ov['range'][0]:g}")
                    row["hi"].set(f"{ov['range'][1]:g}")
                if "grid_step" in ov:
                    row["step"].set(f"{ov['grid_step']:g}")
        self._update_params_enabled()
        # hyperparams
        for pname in self.hyper_vars:
            if pname in fit:
                self.hyper_vars[pname].set(str(fit[pname]))
        # toggles
        self.pca_var.set(bool(fit.get("pca")))
        self.cv_var.set(bool(fit.get("cross_validate")))
        self.cvfinal_var.set(bool(fit.get("cv_final_fit")))
        self.downsample_var.set(str(fit["downsample"]) if fit.get("downsample") else "")
        self.workers_var.set(str(fit["workers"]) if fit.get("workers") is not None
                             else "")
        # tolerances
        tol = fit.get("tolerances", {}) or {}
        for tname, var in self.tol_vars.items():
            var.set(str(tol[tname]) if tname in tol else "")
        # rolls
        self.roll_list.selection_clear(0, "end")
        wanted = set(cfg.get("rolls") or [])
        extra = []
        for i, rid in enumerate(self._roll_ids):
            if rid in wanted:
                self.roll_list.selection_set(i)
        extra = [r for r in (cfg.get("rolls") or []) if r not in self._roll_ids]
        self.roll_extra_var.set(" ".join(extra))
        # provenance
        self.current_config_path = Path(path) if path else None
        self.editor_path_lbl.config(
            text=str(self.current_config_path) if path else "(unsaved)")

    # -- library actions ---------------------------------------------------
    def _refresh_library(self):
        rows = []
        if self.configs_dir.is_dir():
            for p in sorted(self.configs_dir.glob("*.json")):
                try:
                    cfg = json.loads(p.read_text())
                    kind = cfg.get("kind", "?")
                    method = (cfg.get("fit") or {}).get("method", "?")
                except Exception:     # noqa: BLE001
                    kind, method = "(bad json)", ""
                rows.append((str(p), p.name, (kind, method)))
        self._set_rows(self.lib_tree, rows)

    def _selected_config_path(self):
        sel = self.lib_tree.selection()
        return Path(sel[0]) if sel else None

    def _new_config(self):
        self.kind_var.set(self.kinds[0])
        self._on_kind_selected()
        self.method_var.set("none")
        self._rebuild_hyper()
        self.comment_var.set("")
        self.all_params_var.set(True)
        self._update_params_enabled()
        self.pca_var.set(False)
        self.cv_var.set(False)
        self.cvfinal_var.set(False)
        self.downsample_var.set("")
        self.workers_var.set("")
        for var in self.tol_vars.values():
            var.set("")
        self.roll_list.selection_clear(0, "end")
        self.roll_extra_var.set("")
        self.current_config_path = None
        self.editor_path_lbl.config(text="(unsaved)")

    def _edit_selected_config(self):
        p = self._selected_config_path()
        if not p:
            return
        try:
            cfg = json.loads(p.read_text())
        except Exception as ex:   # noqa: BLE001
            messagebox.showerror("Load failed", str(ex))
            return
        self._load_config_into_editor(cfg, p)

    def _duplicate_selected_config(self):
        p = self._selected_config_path()
        if not p:
            return
        name = filedialog.asksaveasfilename(
            title="Duplicate config as", initialdir=str(self.configs_dir),
            initialfile=p.stem + "_copy.json", defaultextension=".json",
            filetypes=[("config", "*.json")])
        if not name:
            return
        Path(name).write_text(p.read_text())
        self._refresh_library()

    def _delete_selected_config(self):
        p = self._selected_config_path()
        if not p:
            return
        if messagebox.askyesno("Delete", f"Delete {p.name}?"):
            p.unlink(missing_ok=True)
            self._refresh_library()

    # -- save --------------------------------------------------------------
    def _write_config(self, path):
        try:
            cfg = self._editor_to_config()
        except ValueError as ex:
            messagebox.showerror("Invalid value", str(ex))
            return False
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        self.current_config_path = path
        self.editor_path_lbl.config(text=str(path))
        self._refresh_library()
        return True

    def _save(self):
        if self.current_config_path is None:
            return self._save_as()
        return self._write_config(self.current_config_path)

    def _save_as(self):
        kind = self.kind_var.get()
        name = filedialog.asksaveasfilename(
            title="Save config as", initialdir=str(self.configs_dir),
            initialfile=f"{kind}_{self.method_var.get()}.json",
            defaultextension=".json", filetypes=[("config", "*.json")])
        if not name:
            return False
        return self._write_config(name)

    def _is_editor_dirty(self):
        if self.current_config_path is None or not self.current_config_path.is_file():
            return True
        try:
            on_disk = json.loads(self.current_config_path.read_text())
            return self._editor_to_config() != on_disk
        except Exception:         # noqa: BLE001
            return True

    # =====================================================================
    # Queue tab
    # =====================================================================
    def _build_queue_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="Queue")
        ctl = ttk.Frame(tab)
        ctl.pack(fill="x", pady=4)
        ttk.Label(ctl, text="parallelism").pack(side="left")
        ttk.Spinbox(ctl, from_=1, to=16, width=4,
                    textvariable=self.max_parallel).pack(side="left", padx=4)
        self.pause_btn = ttk.Button(ctl, text="Pause all", command=self._toggle_pause)
        self.pause_btn.pack(side="left", padx=4)
        ttk.Button(ctl, text="Pause/Resume selected",
                   command=self._pause_selected).pack(side="left", padx=4)
        ttk.Button(ctl, text="Stop selected",
                   command=self._stop_selected).pack(side="left", padx=4)
        ttk.Button(ctl, text="Clear finished",
                   command=self._clear_finished).pack(side="left", padx=4)

        pw = ttk.PanedWindow(tab, orient="horizontal")
        pw.pack(fill="both", expand=True)
        left = ttk.Frame(pw)
        pw.add(left, weight=1)
        ttk.Label(left, text="header: sort (asc/desc/off) · funnel ⧩ or right-click: filter",
                  foreground="#777").pack(anchor="w")
        self.queue_tree = ttk.Treeview(
            left, columns=("kind", "method", "status", "comment"),
            show="tree headings")
        self.queue_tree.heading("#0", text="config")
        self.queue_tree.heading("kind", text="kind")
        self.queue_tree.heading("method", text="method")
        self.queue_tree.heading("status", text="status")
        self.queue_tree.heading("comment", text="comment")
        self.queue_tree.column("#0", width=180)
        self.queue_tree.column("kind", width=70)
        self.queue_tree.column("method", width=110)
        self.queue_tree.column("status", width=80)
        self.queue_tree.column("comment", width=260)
        self._config_stripes(self.queue_tree)
        self._add_column_lines(self.queue_tree)
        self._setup_filterable(self.queue_tree)
        self._attach_cell_tooltip(self.queue_tree, "comment")
        self.queue_tree.pack(fill="both", expand=True)
        self.queue_tree.bind("<<TreeviewSelect>>", lambda e: self._on_job_select())
        self.queue_tree.bind("<Double-1>", lambda e: self._on_queue_double())

        right = ttk.Frame(pw)
        pw.add(right, weight=2)
        ttk.Label(right, text="output (live stdout of the selected job)",
                  font=("", 9, "bold")).pack(anchor="w")
        self.output_container = ttk.Frame(right)
        self.output_container.pack(fill="both", expand=True)

    def _pause_selected(self):
        """Toggle suspend on the SELECTED job(s) only: freeze a running one, thaw a
        paused one. (The 'Pause all' button does the whole queue.)"""
        for iid in self.queue_tree.selection():
            job = self.jobs.get(int(iid))
            if not job or job.popen is None:
                continue
            if job.state == "running" and self._set_process_suspended(
                    job.popen.pid, True):
                job.state = "paused"
            elif job.state == "paused" and self._set_process_suspended(
                    job.popen.pid, False):
                job.state = "running"
        self._render_queue()

    def _toggle_pause(self):
        # Pause both HOLDS pending jobs AND suspends the running calibration
        # processes (so it visibly stops work, not just new starts); Resume continues.
        self.paused = not self.paused
        self.pause_btn.config(text="Resume all" if self.paused else "Pause all")
        for job in list(self.running):
            if job.popen is None:
                continue
            if self.paused and job.state == "running":
                if self._set_process_suspended(job.popen.pid, True):
                    job.state = "paused"
            elif not self.paused and job.state == "paused":
                if self._set_process_suspended(job.popen.pid, False):
                    job.state = "running"
        self._render_queue()

    @staticmethod
    def _set_process_suspended(pid, suspend):
        """Freeze (suspend) or thaw (resume) a process. Windows has no SIGSTOP, so
        use ntdll NtSuspendProcess/NtResumeProcess; POSIX uses SIGSTOP/SIGCONT."""
        try:
            if os.name == "nt":
                import ctypes
                PROCESS_SUSPEND_RESUME = 0x0800
                h = ctypes.windll.kernel32.OpenProcess(
                    PROCESS_SUSPEND_RESUME, False, pid)
                if not h:
                    return False
                fn = (ctypes.windll.ntdll.NtSuspendProcess if suspend
                      else ctypes.windll.ntdll.NtResumeProcess)
                fn(h)
                ctypes.windll.kernel32.CloseHandle(h)
            else:
                os.kill(pid, signal.SIGSTOP if suspend else signal.SIGCONT)
            return True
        except Exception as ex:   # noqa: BLE001
            print(f"suspend/resume failed for pid {pid}: {ex}")
            return False

    def _add_job(self, config_path, name=None, temp=False):
        try:
            cfg = json.loads(Path(config_path).read_text())
        except Exception as ex:   # noqa: BLE001
            messagebox.showerror("Bad config", str(ex))
            return
        self._job_seq += 1
        job = Job(self._job_seq, config_path, cfg.get("kind", "?"),
                  (cfg.get("fit") or {}).get("method", "none"),
                  name=name, temp=temp,
                  comment=(cfg.get("comment") or "").replace("\n", " "))
        job.text = scrolledtext.ScrolledText(self.output_container, wrap="none",
                                             state="disabled", height=10)
        self.jobs[job.id] = job
        self.pending.append(job)
        self._render_queue()

    def _render_queue(self):
        rows = [(str(j.id), j.name, (j.kind, j.method, j.state, j.comment))
                for j in (self.jobs[jid] for jid in sorted(self.jobs))]
        self._set_rows(self.queue_tree, rows)

    def _refresh_job_row(self, job):
        self._render_queue()

    def _on_queue_double(self):
        """Double-clicking a FINISHED job jumps to Results and selects its session."""
        sel = self.queue_tree.selection()
        if not sel:
            return
        job = self.jobs.get(int(sel[0]))
        if not job or not job.session_dir:
            return
        self._refresh_results()
        self.nb.select(self._results_tab)
        iid = str(job.session_dir)
        if self.res_tree.exists(iid):
            self.res_tree.selection_set(iid)
            self.res_tree.focus(iid)
            self.res_tree.see(iid)
            self._show_report()

    def _on_job_select(self):
        sel = self.queue_tree.selection()
        if not sel:
            return
        job = self.jobs.get(int(sel[0]))
        if job is None or job.text is None:
            return
        if self._shown_text is not None:
            self._shown_text.pack_forget()
        job.text.pack(fill="both", expand=True)
        self._shown_text = job.text

    # -- scheduler ---------------------------------------------------------
    def _pump(self):
        # drain each job's stdout queue into its Text widget
        for job in list(self.jobs.values()):
            if job.text is None:
                continue
            try:
                while True:
                    line = job.q.get_nowait()
                    if line is None:
                        job.eof = True
                    else:
                        self._append_output(job, line)
            except queue.Empty:
                pass
        # reap finished
        for job in list(self.running):
            rc = job.popen.poll()
            if rc is not None and job.eof:
                self._finish_job(job, rc)
        # start pending up to the cap
        self._fill_slots()
        self.after(150, self._pump)

    def _append_output(self, job, line):
        t = job.text
        t.configure(state="normal")
        t.insert("end", line)
        t.see("end")
        t.configure(state="disabled")

    def _fill_slots(self):
        if self.paused:
            return
        while len(self.running) < max(1, self.max_parallel.get()) and self.pending:
            self._start_job(self.pending.pop(0))

    def _start_job(self, job):
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        job.start_time = time.time()
        # Snapshot this kind's existing sessions so we can attribute the NEW one to
        # this job even when several same-kind jobs run in parallel.
        job.pre_sessions = {str(p) for p in self.calib_dir.glob(f"*_{job.kind}*")}
        try:
            job.popen = subprocess.Popen(
                [sys.executable, str(self.script_path), "--config",
                 str(job.config_path)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=str(self.script_path.parent),
                creationflags=flags,
                start_new_session=(os.name != "nt"))
        except Exception as ex:   # noqa: BLE001
            job.state = "failed"
            self._append_output(job, f"failed to launch: {ex}\n")
            self._refresh_job_row(job)
            return
        job.state = "running"
        self.running.add(job)
        self._refresh_job_row(job)
        threading.Thread(target=self._reader, args=(job,), daemon=True).start()

    def _reader(self, job):
        try:
            for line in job.popen.stdout:
                job.q.put(line)
        finally:
            try:
                job.popen.stdout.close()
            except Exception:     # noqa: BLE001
                pass
            job.q.put(None)

    def _finish_job(self, job, rc):
        self.running.discard(job)
        if job.state == "stopping":
            job.state = "stopped"
        elif rc == 0:
            job.state = "done"
            job.session_dir = self._resolve_session(job)
        else:
            job.state = "failed"
        self._refresh_job_row(job)
        self._refresh_results()

    def _resolve_session(self, job):
        """The newest session dir for this kind that did NOT exist when the job
        started (so parallel same-kind jobs don't claim each other's sessions)."""
        pre = getattr(job, "pre_sessions", set())
        best = None
        for d in self.calib_dir.glob(f"*_{job.kind}*"):
            if str(d) in pre or not d.is_dir() or not (d / "config.json").is_file():
                continue
            if best is None or d.stat().st_mtime > best.stat().st_mtime:
                best = d
        return best

    # -- stop / clear ------------------------------------------------------
    def _stop_job(self, job):
        if job.state == "pending":
            if job in self.pending:
                self.pending.remove(job)
            job.state = "stopped"
            self._refresh_job_row(job)
            return
        if job.state not in ("running", "paused") or job.popen is None:
            return
        pid = job.popen.pid
        if job.state == "paused":               # thaw so it can be killed cleanly
            self._set_process_suspended(pid, False)
        job.state = "stopping"
        self._refresh_job_row(job)
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                               capture_output=True)
            else:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception as ex:   # noqa: BLE001
            print(f"stop failed for job {job.id}: {ex}")
            try:
                job.popen.terminate()
            except Exception:     # noqa: BLE001
                pass

    def _stop_selected(self):
        sel = self.queue_tree.selection()
        for iid in sel:
            job = self.jobs.get(int(iid))
            if job:
                self._stop_job(job)

    def _clear_finished(self):
        for job in list(self.jobs.values()):
            if job.state in ("done", "failed", "stopped"):
                if job.text is not None:
                    if job.text is self._shown_text:
                        job.text.pack_forget()
                        self._shown_text = None
                    job.text.destroy()
                self._cleanup_temp_config(job)
                del self.jobs[job.id]
        self._render_queue()

    def _cleanup_temp_config(self, job):
        if getattr(job, "temp", False):
            try:
                job.config_path.unlink(missing_ok=True)
            except Exception:     # noqa: BLE001
                pass

    # -- enqueue -----------------------------------------------------------
    def _enqueue_editor(self):
        # A clean, saved config is queued as-is (reproducible from a real file);
        # unsaved edits run as a one-off temp config (not added to the library).
        if self.current_config_path is not None and not self._is_editor_dirty():
            self._add_job(self.current_config_path)
        else:
            self._run_once_editor()

    def _run_once_editor(self):
        """Queue the CURRENT editor state (incl. unsaved changes) as a one-off:
        write it to a throwaway temp config and run that — nothing is saved to the
        library, and the temp file is deleted when the job is cleared / on close."""
        try:
            cfg = self._editor_to_config()
        except ValueError as ex:
            messagebox.showerror("Invalid value", str(ex))
            return
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".json", prefix="calib_oneoff_")
        os.close(fd)
        Path(path).write_text(json.dumps(cfg, indent=2) + "\n")
        base = (self.current_config_path.stem if self.current_config_path
                else cfg["kind"])
        self._add_job(path, name=f"{base} (unsaved)", temp=True)

    # =====================================================================
    # Results tab
    # =====================================================================
    def _build_results_tab(self, nb):
        tab = ttk.Frame(nb)
        self._results_tab = tab
        nb.add(tab, text="Results")
        pw = ttk.PanedWindow(tab, orient="horizontal")
        pw.pack(fill="both", expand=True)
        left = ttk.Frame(pw)
        pw.add(left, weight=1)
        bar = ttk.Frame(left)
        bar.pack(fill="x")
        ttk.Button(bar, text="Refresh", command=self._refresh_results).pack(side="left")
        ttk.Button(bar, text="Open folder",
                   command=self._open_session_folder).pack(side="left", padx=4)
        ttk.Button(bar, text="Review in debug UI",
                   command=self._review_session).pack(side="left")
        ttk.Label(left, text="header: sort (asc/desc/off) · funnel ⧩ or right-click: filter",
                  foreground="#777").pack(anchor="w")
        self.res_tree = ttk.Treeview(
            left, columns=("kind", "objective", "wall", "comment"),
            show="tree headings", height=24)
        self.res_tree.heading("#0", text="session")
        self.res_tree.heading("kind", text="kind")
        self.res_tree.heading("objective", text="objective")
        self.res_tree.heading("wall", text="wall_s")
        self.res_tree.heading("comment", text="comment")
        self.res_tree.column("#0", width=210)
        self.res_tree.column("kind", width=70)
        self.res_tree.column("objective", width=90)
        self.res_tree.column("wall", width=70)
        self.res_tree.column("comment", width=280)
        self._config_stripes(self.res_tree)
        self._add_column_lines(self.res_tree)
        self._setup_filterable(self.res_tree)
        self._attach_cell_tooltip(self.res_tree, "comment")
        self.res_tree.pack(fill="both", expand=True)
        self.res_tree.bind("<<TreeviewSelect>>", lambda e: self._show_report())

        right = ttk.Frame(pw)
        pw.add(right, weight=2)
        ttk.Label(right, text="report.md", font=("", 9, "bold")).pack(anchor="w")
        self.report_view = _MarkdownView(right)
        self.report_view.pack(fill="both", expand=True)

    def _refresh_results(self):
        self.res_tree.delete(*self.res_tree.get_children())
        if not self.calib_dir.is_dir():
            return
        rows = []
        for d in self.calib_dir.iterdir():
            cfgp = d / "config.json"
            resp = d / "results.json"
            if not (d.is_dir() and cfgp.is_file()):
                continue
            kind, obj, wall, comment = "?", "", "", ""
            try:
                cfg = json.loads(cfgp.read_text())
                kind = cfg.get("kind", "?")
                comment = (cfg.get("comment") or "").replace("\n", " ")
            except Exception:     # noqa: BLE001
                pass
            if resp.is_file():
                try:
                    res = json.loads(resp.read_text())
                    o = res.get("objective_final", res.get("mean_holdout_candidate"))
                    obj = f"{o:.4f}" if isinstance(o, (int, float)) else ""
                    wall = res.get("wall_seconds", "")
                except Exception:     # noqa: BLE001
                    pass
            rows.append((d.name, kind, obj, wall, comment, str(d)))
        table = [(path, name, (kind, obj, wall, comment))
                 for name, kind, obj, wall, comment, path in sorted(rows, reverse=True)]
        self._set_rows(self.res_tree, table)

    def _selected_session(self):
        sel = self.res_tree.selection()
        return Path(sel[0]) if sel else None

    def _show_report(self):
        d = self._selected_session()
        if not d:
            self.report_view.render("")
            return
        rp = d / "report.md"
        self.report_view.render(
            rp.read_text(encoding="utf-8") if rp.is_file() else "_(no report.md)_")

    def _open_session_folder(self):
        d = self._selected_session()
        if not d:
            return
        try:
            if os.name == "nt":
                os.startfile(str(d))   # noqa: S606
            else:
                subprocess.Popen(["xdg-open", str(d)])
        except Exception as ex:   # noqa: BLE001
            messagebox.showerror("Open folder", str(ex))

    def _review_session(self):
        d = self._selected_session()
        if not d:
            return
        try:
            subprocess.Popen([sys.executable, str(self.script_path),
                              "--review", str(d)], cwd=str(self.script_path.parent))
        except Exception as ex:   # noqa: BLE001
            messagebox.showerror("Review", str(ex))

    # =====================================================================
    # Close
    # =====================================================================
    def _on_close(self):
        active = [j for j in self.jobs.values()
                  if j.state in ("running", "paused", "stopping")]
        if active and not messagebox.askyesno(
                "Quit", f"{len(active)} calibration run(s) in progress. Stop them "
                        "and quit? (no session is written for a stopped run)"):
            return
        for j in active:
            self._stop_job(j)
        for j in self.jobs.values():   # remove any one-off temp config files
            self._cleanup_temp_config(j)
        self.destroy()


def run(adapter, script_path, configs_dir=None):
    """Launch the calibration manager for `adapter`, spawning `script_path`
    (the feature's run_calibration.py) per queued job."""
    if configs_dir is None:
        configs_dir = Path(adapter.calib_dir) / "configs"
    # Must run BEFORE the first Tk() (one-shot; ignored once a window exists).
    _enable_windows_dpi_awareness()
    CalibManagerUI(adapter, script_path, configs_dir).mainloop()
