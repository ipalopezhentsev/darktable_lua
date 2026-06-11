"""Smoke driver for debug_ui.py refactor verification.

Opens the given debug_ui module on a session dir, waits for the first image
to load, exercises navigation / selection / table sorting / synthetic mouse
interactions programmatically, then closes the window (which auto-saves
annotations and writes debug_report.txt). Running it on two implementations
against identical session copies and diffing the outputs proves behavioral
equivalence.

Usage: python smoke_debug_ui.py <debug_ui_module_path> <session_dir>
"""

import sys
import importlib.util
import traceback
from types import SimpleNamespace

import tkinter as tk


def ev(x=0, y=0, state=0, delta=0, num=0):
    return SimpleNamespace(x=x, y=y, state=state, delta=delta, num=num)


def main():
    mod_path, session_dir = sys.argv[1], sys.argv[2]
    spec = importlib.util.spec_from_file_location("dbgui_under_test", mod_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    root = tk.Tk()
    app = m.DebugUI(root, session_dir)
    failures = []

    def step(name, fn):
        try:
            fn()
        except Exception:
            failures.append(f"{name}:\n{traceback.format_exc()}")

    def to_canvas(ix, iy):
        cx, cy = m.image_to_canvas(ix, iy, app.offset_x, app.offset_y, app.zoom)
        return int(cx), int(cy)

    def click(ix, iy, state=0):
        cx, cy = to_canvas(ix, iy)
        app._on_left_press(ev(cx, cy, state=state))
        app._on_left_release(ev(cx, cy, state=state))

    def empty_point(img):
        """Image coords far (>250px) from every detected spot / missed marker."""
        spots = img.get("detected") or []
        ann = app.annotations[img["stem"]]
        pts = ([(s.get("cx", 0), s.get("cy", 0)) for s in spots]
               + [(d["cx"], d["cy"]) for d in ann["missed_dust"]])
        for gx in range(200, img["width"] - 200, 137):
            for gy in range(200, img["height"] - 200, 137):
                if all((gx - px) ** 2 + (gy - py) ** 2 > 250 ** 2 for px, py in pts):
                    return gx, gy
        return img["width"] // 2, img["height"] // 2

    def actions():
        step("nav_next", lambda: app._nav_image(+1))
        step("nav_prev", lambda: app._nav_image(-1))
        step("zoom_in", lambda: app._zoom_step(2.0))
        step("pan", lambda: app._pan_by(40, -20))
        step("fit", lambda: app._fit_to_window())
        step("hide_markers", lambda: app._toggle_hide_markers())
        step("show_markers", lambda: app._toggle_hide_markers())
        step("sort_ctr", lambda: app._sort_column("ctr"))
        step("sort_ctr2", lambda: app._sort_column("ctr"))
        step("sort_idx", lambda: app._sort_column("idx"))

        # Go to an image with at least 2 detected dot spots
        target = None
        for i, img in enumerate(app.images):
            dots = [s for s in (img.get("detected") or [])
                    if s.get("kind") != "stroke"]
            if len(dots) >= 2:
                target = i
                break
        if target is None:
            print("SMOKE: no image with >=2 dot spots; interaction steps skipped")
        else:
            img = app.images[target]
            app._on_thumb_row_click(target)
            dots = [(i, s) for i, s in enumerate(img.get("detected") or [])
                    if s.get("kind") != "stroke"]
            i0, s0 = dots[0]
            i1, s1 = dots[1]

            # Click-select first dot spot
            step("click_spot", lambda: click(s0["cx"], s0["cy"]))
            # Scroll to adjust its radius correction (only if it got selected)
            def scroll_radius():
                assert app.selected_detected == {i0}, \
                    f"expected {{{i0}}}, got {app.selected_detected}"
                app._on_mousewheel(ev(*to_canvas(s0["cx"], s0["cy"]), delta=120))
                app._on_mousewheel(ev(*to_canvas(s0["cx"], s0["cy"]), delta=120))
            step("scroll_radius", scroll_radius)
            # Mark it as a false positive
            step("mark_fp", lambda: app._mark_fp())
            # Shift+click second spot to add it to the selection
            step("shift_click", lambda: click(s1["cx"], s1["cy"], state=0x1))
            # Ctrl+click an empty area to add a missed-dust marker
            ex, ey = empty_point(img)
            def ctrl_click_missed():
                cx, cy = to_canvas(ex, ey)
                app._on_left_press(ev(cx, cy, state=0x4))
                app._on_left_release(ev(cx, cy, state=0x4))
                assert app.annotations[img["stem"]]["missed_dust"], "no missed added"
            step("ctrl_click_missed", ctrl_click_missed)
            # Rubber-band select around the first spot
            def rubber_band():
                x0, y0 = to_canvas(s0["cx"] - 100, s0["cy"] - 100)
                x1, y1 = to_canvas(s0["cx"] + 100, s0["cy"] + 100)
                app._on_left_press(ev(x0, y0))
                app._on_left_drag(ev((x0 + x1) // 2, (y0 + y1) // 2))
                app._on_left_drag(ev(x1, y1))
                app._on_left_release(ev(x1, y1))
            step("rubber_band", rubber_band)
            # Draw a missed thread: T, three press points, Enter
            def thread_draw():
                app._toggle_thread_draw()
                for dx in (0, 150, 300):
                    cx, cy = to_canvas(ex + dx, ey + 60)
                    app._on_left_press(ev(cx, cy))
                    app._on_left_release(ev(cx, cy))
                app._finish_thread()
                assert app.annotations[img["stem"]]["missed_strokes"], "no thread added"
            step("thread_draw", thread_draw)
            # Click empty area to clear selection, then nav away (auto-saves)
            step("click_empty", lambda: click(ex, ey - 150) if ey > 350 else None)
            step("nav_away", lambda: app._nav_image(+1 if target + 1 < len(app.images) else -1))

        step("clear_selection", lambda: app._clear_selection())
        step("close", lambda: app._on_close())
        if failures:
            print("SMOKE FAILURES:")
            for f in failures:
                print(f)
            sys.exit(1)
        print("SMOKE OK")
        sys.exit(0)

    root.after(2500, actions)
    root.mainloop()


if __name__ == "__main__":
    main()
