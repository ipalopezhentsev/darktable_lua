"""Smoke driver for the negadoctor debug UI.

Builds a throwaway session (3 frames from the first annotated roll under
tests/fixtures/rolls/, processed by the real pipeline) in %TEMP%, opens
NegadoctorDebugUI on it, exercises navigation /
selection / corrections / view toggles programmatically, then closes the
window (auto-saving annotations + debug_report.txt) and asserts the outputs.

Usage: conda run -n autocrop python auto_negadoctor/tests/smoke_debug_ui.py
       (or pass an existing session dir as the only argument)
"""

import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import tkinter as tk

TESTS_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
FEATURE_DIR = TESTS_DIR.parent
sys.path.insert(0, str(FEATURE_DIR))
sys.path.insert(0, str(TESTS_DIR))

import auto_negadoctor as an
import debug_ui as dbg
import run_quality_tests as rqt


def build_session():
    """Run the real pipeline on 3 test frames into a temp session dir.
    Uses the first annotated roll's TIFF exports (real export format)."""
    rolls = rqt.discover_rolls()
    roll = rolls[0] if rolls else None
    images = ([Path(p) for p in roll["images"][:3]] if roll
              else sorted((TESTS_DIR / "images").glob("*.jpg"))[:3])
    if not images:
        print("SMOKE SKIP: no test images")
        sys.exit(0)
    session = Path(tempfile.mkdtemp(prefix="nega_smoke_"))
    paths = []
    for img in images:
        dst = session / img.name
        shutil.copy(img, dst)
        paths.append(str(dst))
    exif = roll["exif"] if roll else {}
    frames, roll = an.process_roll(paths, exif)
    an.write_results(frames, roll, session)
    an.write_debug_sessions(frames, roll, session, wall_time_s=1.0)
    return session


def ev(x=0, y=0, state=0, delta=0, num=0):
    return SimpleNamespace(x=x, y=y, state=state, delta=delta, num=num)


def main():
    session = Path(sys.argv[1]) if len(sys.argv) > 1 else build_session()
    cleanup = len(sys.argv) <= 1

    root = tk.Tk()
    app = dbg.NegadoctorDebugUI(root, str(session))
    failures = []

    def step(name, fn):
        try:
            fn()
        except Exception:
            failures.append(f"{name}:\n{traceback.format_exc()}")

    def to_canvas(ix, iy):
        cx, cy = dbg.image_to_canvas(ix, iy, app.offset_x, app.offset_y, app.zoom)
        return int(cx), int(cy)

    def actions():
        step("nav_next", lambda: app._nav_image(+1))
        step("nav_prev", lambda: app._nav_image(-1))
        step("zoom_in", lambda: app._zoom_step(2.0))
        step("pan", lambda: app._pan_by(40, -20))
        step("fit", lambda: app._fit_to_window())
        step("hide_markers", lambda: app._toggle_hide_markers())
        step("show_markers", lambda: app._toggle_hide_markers())
        step("sort_det", lambda: app._sort_column("det"))
        step("sort_det2", lambda: app._sort_column("det"))

        img = app.images[app.current_idx]
        stem = img["stem"]

        # Select each patch kind via key handler
        for patch in dbg.PATCHES:
            step(f"select_{patch}", lambda p=patch: app._select_patch(p))

        # Click-select film base rect (if detected)
        det = app._detected_rect(img, "film_base")
        if det:
            cx, cy = to_canvas(det[0] + det[2] / 2, det[1] + det[3] / 2)
            step("click_base", lambda: (app._on_left_press(ev(cx, cy)),
                                        app._on_left_release(ev(cx, cy))))

        # Ctrl+Click relocate the shadows patch to image center
        def relocate():
            app._select_patch("shadows")
            ccx, ccy = to_canvas(img["width"] // 2, img["height"] // 2)
            app._on_left_press(ev(ccx, ccy, state=0x4))
            app._on_left_release(ev(ccx, ccy, state=0x4))
            assert "shadows" in app.annotations[stem]["patch_corrections"], \
                "no shadows correction recorded"
        step("ctrl_click_relocate", relocate)

        # Scroll-RESIZE the corrected patch (center kept, size persisted)
        def resize():
            before = list(app.annotations[stem]["patch_corrections"]["shadows"])
            app._on_mousewheel(ev(100, 100, delta=120))
            app._on_mousewheel(ev(100, 100, delta=120))
            after = app.annotations[stem]["patch_corrections"]["shadows"]
            assert after[2] == before[2] + 4 and after[3] == before[3] + 4, \
                f"scroll resize wrong: {before} -> {after}"
            app._on_mousewheel(ev(100, 100, delta=-120, state=0x1))  # Shift
            after2 = app.annotations[stem]["patch_corrections"]["shadows"]
            assert after2[2] == after[2] - 10, f"shift resize wrong: {after2}"
        step("scroll_resize", resize)

        # Resize a DETECTED patch (no prior correction): seeds one
        def resize_detected():
            det = app._detected_rect(img, "highlights")
            if det is None:
                return
            app._select_patch("highlights")
            app._on_mousewheel(ev(100, 100, delta=120))
            corr = app.annotations[stem]["patch_corrections"].get("highlights")
            assert corr and corr[2] == det[2] + 2, \
                f"detected-patch resize did not seed correction: {corr} from {det}"
        step("resize_detected", resize_detected)

        # Live re-render from corrections
        def live_render():
            app._apply_live_render()
            assert app._live_rendered, "live render did not engage"
            params = app._corrected_params(img)
            assert params is not None and "wb_low" in params
        step("live_render", live_render)

        # Session carries the roll vignette field (may be None for tiny rolls)
        def vignette_in_session():
            assert "vignette" in img, "session missing vignette key"
        step("vignette_in_session", vignette_in_session)

        # Print-param overrides: select via key target, scroll to adjust
        def print_override():
            app._select_patch("gamma")
            applied = img["params"]["gamma"]
            app._on_mousewheel(ev(100, 100, delta=120))
            ov = app.annotations[stem]["print_overrides"].get("gamma")
            assert ov is not None and abs(ov - (applied + 0.05)) < 1e-9, \
                f"gamma step wrong: {ov} vs applied {applied}"
            app._on_mousewheel(ev(100, 100, delta=120, state=0x1))   # big step
            ov2 = app.annotations[stem]["print_overrides"]["gamma"]
            assert abs(ov2 - (ov + 0.25)) < 1e-9, f"gamma big step wrong: {ov2}"
            params = app._corrected_params(img)
            assert abs(params["gamma"] - ov2) < 1e-9, "override not in params"
            # a second param that stays for the persistence check
            app._select_patch("black")
            app._on_mousewheel(ev(100, 100, delta=-120))
            assert "black" in app.annotations[stem]["print_overrides"]
        step("print_override", print_override)

        # Color wheels: dragging sets wb_low/wb_high directly (real event
        # path), the override lands in corrected params, C reverts to auto
        def wheel_override():
            name = "wb_shadows"
            wheel = app.wheels[name]
            # the algorithm's auto wb is pinned (fixed divergence reference)
            assert wheel._auto_pos is not None, "auto wb pin not placed"
            wheel._on_press(ev(int(wheel._cx + 15), int(wheel._cy - 10)))
            wheel._on_drag(ev(int(wheel._cx + 20), int(wheel._cy - 12)))
            ovr = app.annotations[stem]["wb_overrides"].get("shadows")
            assert ovr and len(ovr) == 3, f"wheel override not stored: {ovr}"
            assert abs(max(ovr) - 1.0) < 1e-9, f"wb_low not normalized: {ovr}"
            assert app.selected_patch == name, "wheel did not select itself"
            params = app._corrected_params(img)
            assert params is not None and \
                params["wb_low"] == [float(v) for v in ovr], \
                "wheel override not applied in corrected params"
            # highlights wheel, then C reverts to the auto-found value
            hi = "wb_highlights"
            hw = app.wheels[hi]
            hw._on_press(ev(int(hw._cx - 12), int(hw._cy - 8)))
            assert "highlights" in app.annotations[stem]["wb_overrides"]
            app._select_patch(hi)
            app._clear_correction()
            assert "highlights" not in app.annotations[stem]["wb_overrides"], \
                "C did not clear the highlights wheel override"
        step("wheel_override", wheel_override)

        # Wheels resize to fill the panel and keep the marker at the same wb
        def wheel_resize():
            wheel = app.wheels["wb_shadows"]
            wb_before = list(wheel._marker_wb)
            app._resize_wheels(SimpleNamespace(width=420, height=900))
            assert wheel.size > 150, f"wheel did not grow: {wheel.size}"
            exp = wheel._wb_pos(wb_before)     # where the marker should now sit
            assert abs(wheel._marker_pos[0] - exp[0]) < 1.0 and \
                abs(wheel._marker_pos[1] - exp[1]) < 1.0, \
                "marker not re-placed after resize"
        step("wheel_resize", wheel_resize)

        # X: compare corrected vs default render
        def compare_toggle():
            app._toggle_compare()
            assert app.compare_default
            app._apply_live_render()
            assert not app._live_rendered, "compare mode must show default"
            app._toggle_compare()
            app._apply_live_render()
            assert app._live_rendered, "corrected render did not return"
        step("compare_toggle", compare_toggle)

        # C clears a print override
        def clear_print():
            app._select_patch("gamma")
            app._clear_correction()
            assert "gamma" not in app.annotations[stem]["print_overrides"]
        step("clear_print", clear_print)

        # Crop correction: select, rubber-band define, scroll grow, live render
        def crop_correction():
            app._select_patch(dbg.CROP_NAME)
            app.on_rubber_band(60, 60, img["width"] - 60, img["height"] - 60,
                               False)
            crop = app.annotations[stem]["crop_correction"]
            assert crop == [60, 60, img["width"] - 120, img["height"] - 120], \
                f"crop rect wrong: {crop}"
            app._on_mousewheel(ev(100, 100, delta=120))   # grow
            crop2 = app.annotations[stem]["crop_correction"]
            assert crop2[0] == 58 and crop2[2] == crop[2] + 4, \
                f"crop grow wrong: {crop2}"
            params = app._corrected_params(img)
            assert params is not None, "crop did not engage corrected params"
            assert "black" in params and "exposure" in params
        step("crop_correction", crop_correction)

        # Hide-rejected view + histogram respect the user crop
        def crop_views():
            app.mask_view = 1
            app._cycle_mask_view()            # -> 2 (hide rejected)
            assert app.mask_view == 2
            assert app._hist_data is not None
            app._cycle_mask_view()            # -> 0
        step("crop_views", crop_views)

        # C clears the crop
        def clear_crop():
            app._select_patch(dbg.CROP_NAME)
            app._clear_correction()
            assert not app.annotations[stem]["crop_correction"]
        step("clear_crop", clear_crop)

        # Dragging in the analysis-crop view (M) defines the crop even when
        # crop is NOT the selected item (the lost-annotation bug)
        def crop_in_mask_view():
            app._clear_selection()
            app.mask_view = 1
            app.on_rubber_band(80, 80, img["width"] - 80, img["height"] - 80,
                               False)
            assert app.annotations[stem]["crop_correction"] == \
                [80, 80, img["width"] - 160, img["height"] - 160], \
                "drag in mask view did not set crop"
            assert app.selected_patch == dbg.CROP_NAME
            app._clear_correction()
            app._clear_selection()
            # mask view persists across image navigation (decoration applied
            # to the new frame too)
            app._nav_image(+1)
            assert app.mask_view == 1, "mask view did not persist across nav"
            assert app.pil_image is not app._display_base_pil, \
                "mask decoration not applied after nav"
            app._nav_image(-1)
            app.mask_view = 0
            app._clear_selection()
        step("crop_in_mask_view", crop_in_mask_view)

        # Dragging an individual crop edge adjusts just that edge — driven
        # through the BASE press/drag/release handlers (the real event path;
        # calling the overrides directly masked a drag_start routing bug)
        def crop_edge_drag():
            app._select_patch(dbg.CROP_NAME)
            app.on_rubber_band(100, 100, 800, 500, False)
            rect = app.annotations[stem]["crop_correction"]
            assert rect == [100, 100, 700, 400]
            # grab the left edge with NOTHING selected (works in any mode)
            app._clear_selection()
            cx, cy = to_canvas(100, 300)
            mx, my = to_canvas(150, 300)
            app._on_left_press(ev(cx, cy))
            assert app._crop_drag_edge == "left", "edge grab failed"
            app._on_left_drag(ev(mx, my))
            app._on_left_release(ev(mx, my))
            rect2 = app.annotations[stem]["crop_correction"]
            assert abs(rect2[0] - 150) <= 2 and rect2[0] + rect2[2] == 800 \
                and rect2[1] == 100 and rect2[3] == 400, \
                f"edge drag wrong: {rect2}"
            assert app.selected_patch == dbg.CROP_NAME
            # press far from any edge is NOT consumed (rubber band still works)
            fx, fy = to_canvas(400, 300)
            assert not app.handle_press_override(ev(fx, fy))
            app._clear_correction()
            app._clear_selection()
        step("crop_edge_drag", crop_edge_drag)

        # M cycles: normal -> tinted analysis areas -> rejected hidden
        def mask_views():
            # mask is computed live from the detected crop border (no baked PNG)
            assert app._analysis_mask(img) is not None, "no live analysis mask"
            base = app._display_base_pil
            assert base is not None
            app._cycle_mask_view()           # tint
            assert app.mask_view == 1 and app.pil_image is not base
            app._cycle_mask_view()           # hide rejected
            assert app.mask_view == 2
            assert app._hist_data is not None, "no histogram in hide mode"
            app._cycle_mask_view()           # back to normal
            assert app.mask_view == 0
        step("mask_views", mask_views)

        # Histogram present and toggleable
        def histogram():
            assert app._hist_data is not None and len(app._hist_data) == 3
            app._toggle_histogram()
            assert not app.show_histogram
            app._toggle_histogram()
            assert app.show_histogram
        step("histogram", histogram)

        # Live wb feedback for the corrected patch
        def live_wb():
            corr = app.annotations[stem]["patch_corrections"]["shadows"]
            rgb = app._neg_rgb_at(img, corr)
            assert rgb and len(rgb) == 3, "no negative-space RGB for corrected rect"
            wb = app._wb_for_patch(img, "shadows", rgb)
            assert wb and len(wb) == 3, "no live wb computed"
        step("live_wb", live_wb)

        # Note on the selected patch
        def note():
            app._select_patch("shadows")
            app.set_selected_note("smoke note")
            app._auto_save(stem)
            assert app.annotations[stem]["patch_notes"].get("shadows") == "smoke note"
        step("note", note)

        # View + bad-inversion toggles
        step("view_negative", lambda: app._toggle_view())
        step("view_inverted", lambda: app._toggle_view())
        def bad():
            app._toggle_bad_inversion()
            assert app.annotations[stem]["bad_inversion"] is True
        step("bad_inversion", bad)

        # Clear correction for film_base (placed first), keep shadows
        def clear():
            app._select_patch("film_base")
            ccx, ccy = to_canvas(img["width"] // 3, img["height"] // 3)
            app._on_left_press(ev(ccx, ccy, state=0x4))
            app._on_left_release(ev(ccx, ccy, state=0x4))
            assert "film_base" in app.annotations[stem]["patch_corrections"]
            app._clear_correction()
            assert "film_base" not in app.annotations[stem]["patch_corrections"]
        step("clear_correction", clear)

        step("clear_selection", lambda: app._clear_selection())
        step("close", lambda: app._on_close())

        # Verify persisted outputs
        def verify_outputs():
            ann_path = session / f"{stem}_annotations.json"
            assert ann_path.exists(), "annotations json missing"
            data = json.loads(ann_path.read_text())
            assert data["patch_corrections"]["shadows"]["corrected"], "no corrected rect"
            assert data["patch_notes"].get("shadows") == "smoke note"
            assert data["bad_inversion"] is True
            assert "black" in data["print_overrides"], "print override not saved"
            assert data["print_overrides"]["black"]["applied"] is not None
            assert "shadows" in data["wb_overrides"], "wb override not saved"
            assert data["wb_overrides"]["shadows"]["applied"] is not None
            assert data["wb_overrides"]["shadows"]["corrected"], "no chosen wb"
            report = session / "debug_report.txt"
            assert report.exists(), "debug_report.txt missing"
            txt = report.read_text(encoding="utf-8")
            assert "CORRECTED PATCHES" in txt and "BAD INVERSION" in txt
            assert "PRINT PARAM OVERRIDES" in txt
            assert "WB WHEEL OVERRIDES" in txt
        step("verify_outputs", verify_outputs)

        if failures:
            print("SMOKE FAILURES:")
            for f in failures:
                print(f)
            sys.exit(1)
        if cleanup:
            shutil.rmtree(session, ignore_errors=True)
        print("SMOKE OK")
        sys.exit(0)

    root.after(2500, actions)
    root.mainloop()


if __name__ == "__main__":
    main()
