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
    # Exercise the annotate+apply flow: on close the UI must write
    # applied_results.txt (verified in verify_outputs).
    app.apply_mode = True
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

        # Ctrl+Click relocate the highlights patch to image center
        def relocate():
            app._select_patch("highlights")
            ccx, ccy = to_canvas(img["width"] // 2, img["height"] // 2)
            app._on_left_press(ev(ccx, ccy, state=0x4))
            app._on_left_release(ev(ccx, ccy, state=0x4))
            assert "highlights" in app.annotations[stem]["patch_corrections"], \
                "no highlights correction recorded"
        step("ctrl_click_relocate", relocate)

        # Scroll-RESIZE the corrected patch (center kept, size persisted)
        def resize():
            before = list(app.annotations[stem]["patch_corrections"]["highlights"])
            app._on_mousewheel(ev(100, 100, delta=120))
            app._on_mousewheel(ev(100, 100, delta=120))
            after = app.annotations[stem]["patch_corrections"]["highlights"]
            assert after[2] == before[2] + 4 and after[3] == before[3] + 4, \
                f"scroll resize wrong: {before} -> {after}"
            app._on_mousewheel(ev(100, 100, delta=-120, state=0x1))  # Shift
            after2 = app.annotations[stem]["patch_corrections"]["highlights"]
            assert after2[2] == after[2] - 10, f"shift resize wrong: {after2}"
        step("scroll_resize", resize)

        # Resize a DETECTED patch (no prior correction): seeds one. Uses the
        # film-base patch (highlights already has a correction from above). The
        # film base grows on ALL sides (keeping its rectangle), not square-style.
        def resize_detected():
            det = app._detected_rect(img, "film_base")
            if det is None:
                return
            app._select_patch("film_base")
            app._on_mousewheel(ev(100, 100, delta=120))   # one step (=2 px) out
            corr = app.annotations[stem]["patch_corrections"].get("film_base")
            assert (corr and corr[0] == det[0] - 2 and corr[1] == det[1] - 2
                    and corr[2] == det[2] + 4 and corr[3] == det[3] + 4), \
                f"detected film-base resize (all sides) wrong: {corr} from {det}"
        step("resize_detected", resize_detected)

        # Draw a free RECTANGLE for the film base (rubber-band while film-base is
        # selected): the true unexposed strip is often thin/non-square, which the
        # old square-only patch could not capture.
        def draw_base_rect():
            app._select_patch("film_base")
            app.on_rubber_band(100, 200, 460, 260, False)
            corr = app.annotations[stem]["patch_corrections"]["film_base"]
            assert corr == [100, 200, 360, 60], \
                f"rubber-band film-base rect wrong: {corr}"
            assert corr[2] != corr[3], "film-base rect must not be forced square"
        step("draw_base_rect", draw_base_rect)

        # Drag a patch to move it (real press/drag/release event path). The
        # film-base patch has a correction by now; grabbing inside its rect and
        # dragging moves it (size unchanged), no movement would just select.
        def drag_patch():
            rect = app._effective_rect(img, "film_base")
            if not rect:
                return
            cxr, cyr = rect[0] + rect[2] // 2, rect[1] + rect[3] // 2
            px, py = to_canvas(cxr, cyr)
            app._on_left_press(ev(px, py))
            assert app._patch_drag and app._patch_drag["patch"] == "film_base", \
                "patch grab failed"
            tx, ty = to_canvas(cxr + 30, cyr + 20)
            app._on_left_drag(ev(tx, ty))
            app._on_left_release(ev(tx, ty))
            corr = app.annotations[stem]["patch_corrections"]["film_base"]
            assert abs(corr[0] - (rect[0] + 30)) <= 2 \
                and abs(corr[1] - (rect[1] + 20)) <= 2, \
                f"patch drag wrong: {corr} from {rect}"
            assert corr[2] == rect[2] and corr[3] == rect[3], \
                "patch size changed during move"
            assert app._patch_drag is None, "patch drag not cleared on release"
        step("drag_patch", drag_patch)

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

        # The shadows wheel's invert flag tracks the EFFECTIVE offset sign: a
        # positive offset flips its image cast (offset_c = wb_high*offset*wb_low),
        # so editing offset across zero in the UI must re-orient the wheel.
        def offset_flips_shadows_wheel():
            sw = app.wheels["wb_shadows"]
            hw = app.wheels["wb_highlights"]
            ann = app.annotations[stem]
            ann["print_overrides"]["offset"] = 0.01      # positive
            app._sync_wheels()
            assert sw.invert and not hw.invert, \
                "positive offset must invert ONLY the shadows wheel"
            ann["print_overrides"]["offset"] = -0.01     # negative
            app._sync_wheels()
            assert not sw.invert and not hw.invert, \
                "negative offset must un-invert the shadows wheel"
            # the slider/scroll path must re-sync too (crossing zero)
            app._apply_print_value("offset", 0.01)
            assert sw.invert, "slider edit to positive offset did not flip wheel"
            del ann["print_overrides"]["offset"]
            app._sync_wheels()
        step("offset_flips_shadows_wheel", offset_flips_shadows_wheel)

        # Inline print slider: darktable-unit display + the slider (under the
        # clicked item row) edits the override, with fine/coarse wheel steps.
        def print_sliders():
            # darktable display formatting (percent / EV / plain)
            assert app._fmt_print("black", 0.0755) == "+7.55%"
            assert app._fmt_print("soft_clip", 0.75) == "75.00%"
            assert app._fmt_print("gamma", 3.91) == "3.91"
            assert app._fmt_print("exposure", 2.0 ** 0.55) == "+0.55 EV"
            # param <-> display round-trips (incl. the EV log2/pow conversion)
            for name, val in (("black", 0.12), ("gamma", 4.2),
                              ("soft_clip", 0.6), ("exposure", 1.3)):
                disp = app._print_to_display(name, val)
                assert abs(app._print_from_display(name, disp) - val) < 1e-9, \
                    f"{name} display round-trip failed"
            # selecting a print param targets the inline slider (placement itself
            # is headless-dependent, so drive the handlers directly below)
            app._ensure_inline_slider()
            app._select_patch("exposure")
            app._inline_slider_name = "exposure"
            # slider drag (coarse): override = display value mapped EV -> linear
            app._on_inline_slider("0.5")
            ov = app.annotations[stem]["print_overrides"].get("exposure")
            assert ov is not None and abs(ov - 2.0 ** 0.5) < 1e-6, \
                f"slider drag override wrong: {ov}"
            assert abs(app._corrected_params(img)["exposure"] - ov) < 1e-9, \
                "slider override not in corrected params"
            # wheel = fine step; Shift+wheel = coarse step (the PRINT_STEP pair)
            before = app._effective_print_value("exposure")
            app._on_inline_slider_wheel(ev(delta=120))
            fine = app._effective_print_value("exposure")
            assert abs((fine - before) - dbg.PRINT_STEP["exposure"][0]) < 1e-6, \
                f"fine wheel step wrong: {before}->{fine}"
            app._on_inline_slider_wheel(ev(delta=120, state=0x1))
            coarse = app._effective_print_value("exposure")
            assert abs((coarse - fine) - dbg.PRINT_STEP["exposure"][1]) < 1e-6, \
                f"coarse wheel step wrong: {fine}->{coarse}"
            # selecting a non-print item hides the slider
            app._select_patch("film_base")
            assert app._inline_slider_name is None, \
                "inline slider not hidden off a print row"
            # clear the override for later steps
            app._select_patch("exposure")
            app._clear_correction()
            assert "exposure" not in app.annotations[stem]["print_overrides"]
        step("print_sliders", print_sliders)

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

        # per-channel wb sliders: moving R/G/B sets that channel of wb_low/
        # wb_high directly (full 3-DOF incl. magnitude, unlike the normalized
        # wheel) so the render matches darktable's numbers exactly.
        def wb_sliders():
            name = "wb_shadows"
            sliders = app.wb_sliders[name]
            sliders[0].set(0.94)
            app._on_wb_slider(name, 0, 0.94)
            sliders[1].set(0.88)
            app._on_wb_slider(name, 1, 0.88)
            sliders[2].set(0.73)
            app._on_wb_slider(name, 2, 0.73)
            ovr = app.annotations[stem]["wb_overrides"].get("shadows")
            assert ovr and [round(v, 4) for v in ovr] == [0.94, 0.88, 0.73], \
                f"slider wb not stored verbatim: {ovr}"
            assert abs(max(ovr) - 1.0) > 1e-6, \
                "slider wb must keep darktable magnitude (not normalized to 1)"
            params = app._corrected_params(img)
            assert params is not None and \
                [round(v, 4) for v in params["wb_low"]] == [0.94, 0.88, 0.73], \
                "slider wb override not applied verbatim in corrected params"
            # values are clamped to darktable's [0.25, 2.0] range
            app._on_wb_slider("wb_highlights", 0, 3.0)
            app._on_wb_slider("wb_highlights", 1, 0.1)
            hi = app.annotations[stem]["wb_overrides"]["highlights"]
            assert hi[0] == 2.0 and hi[1] == 0.25, f"wb not clamped: {hi}"
            # programmatic .set() during sync must NOT register as a user edit
            del app.annotations[stem]["wb_overrides"]["shadows"]
            app._sync_wb_sliders()
            assert "shadows" not in app.annotations[stem]["wb_overrides"], \
                "_sync_wb_sliders must not create an override via the callback"
        step("wb_sliders", wb_sliders)

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

        # ] brighter / [ darker: raise/lower black, re-solve exposure to hold the
        # highlight level (the user's manual move). Must be reversible: brighter
        # then darker returns BOTH black and exposure to the originals.
        def brighten():
            saved = dict(app.annotations[stem]["print_overrides"])  # restore after
            base = app._corrected_params(img) or app._variant_params(img)
            blk0, exp0 = float(base["black"]), float(base["exposure"])
            lin = app._neg_lin(img)
            content = app._brighten_content_pixels(img, lin)
            hi0 = app._high_pct(base, content)

            app._brighten(False)
            ov = app.annotations[stem]["print_overrides"]
            exp_blk = min(an.nm.BLACK_RANGE[1], blk0 + dbg.BRIGHTEN_BLACK_STEP)
            assert abs(ov["black"] - exp_blk) < 1e-9, \
                f"brighter black wrong: {ov['black']} vs {exp_blk}"
            assert ov.get("exposure", exp0) <= exp0 + 1e-9, \
                "brighter must not raise exposure (it holds highlights)"
            # the highlight level is preserved by the move
            final = app._corrected_params(img)
            assert abs(app._high_pct(final, content) - hi0) < 5e-3, \
                "brighter did not hold the highlight level"

            # darker reverses it: black AND exposure back to the originals
            app._brighten(True)
            ov = app.annotations[stem]["print_overrides"]
            assert abs(ov["black"] - blk0) < 1e-9, \
                f"darker did not restore black: {ov['black']} vs {blk0}"
            assert abs(ov.get("exposure", exp0) - exp0) < 2e-3, \
                f"darker did not restore exposure: {ov.get('exposure')} vs {exp0}"

            app.annotations[stem]["print_overrides"].clear()
            app.annotations[stem]["print_overrides"].update(saved)
        step("brighten", brighten)

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

        # Clipping indication: stats computed over the displayed image, the
        # on-image overlay toggles (default off) and re-decorates the frame
        def clipping():
            assert app.show_clipping is False, "clip overlay must default off"
            app._refresh_histogram()
            stats = app._clip_stats
            assert stats is not None and {"hi", "lo", "total"} <= set(stats), \
                f"clip stats not computed: {stats}"
            assert 0.0 <= stats["hi"] <= 100.0 and 0.0 <= stats["lo"] <= 100.0, \
                f"clip pct out of range: {stats}"
            base = app._display_base_pil
            app._toggle_clipping()
            assert app.show_clipping is True
            assert app.pil_image is not base, "clip overlay not applied to image"
            app._toggle_clipping()
            assert app.show_clipping is False
        step("clipping", clipping)

        # N: vignette correction on/off in the preview (reloads the negative)
        def vignette_toggle():
            before = app.vignette_on
            app._toggle_vignette()
            assert app.vignette_on != before, "vignette toggle did not flip"
            assert app._neg_cache_key is None, "neg cache not invalidated"
            app._toggle_vignette()
            assert app.vignette_on == before, "vignette toggle not reversible"
        step("vignette_toggle", vignette_toggle)

        # R: live(code) vs fitted(session) review toggle. Not a review session
        # here, so R is a no-op; then inject review payloads and verify the swap.
        def review_toggle():
            assert not app.review_mode, "should not start in review mode"
            app._toggle_review_source()
            assert app.review_source == "fitted", "R changed source off-review"
            img = app.images[app.current_idx]
            base = dict(img["params"])
            live = dict(base)
            live["gamma"] = base["gamma"] + 1.0
            img["review_kind"] = "inversion"
            img["review"] = {
                "fitted": {"params": base, "params_hex": img["params_hex"]},
                "live": {"params": live,
                         "params_hex": an.nm.encode_negadoctor_params(live)}}
            app.review_mode = True
            app.review_kind = "inversion"
            app._apply_review_source()
            assert abs(img["params"]["gamma"] - base["gamma"]) < 1e-9
            app._toggle_review_source()
            assert app.review_source == "live"
            assert abs(img["params"]["gamma"] - live["gamma"]) < 1e-9, \
                "R did not swap in the live params"
            app._toggle_review_source()
            assert app.review_source == "fitted"
            assert abs(img["params"]["gamma"] - base["gamma"]) < 1e-9, \
                "R did not swap back to fitted"
            img.pop("review", None)
            img.pop("review_kind", None)
            app.review_mode = False
        step("review_toggle", review_toggle)

        # Live wb feedback for the corrected patch
        def live_wb():
            corr = app.annotations[stem]["patch_corrections"]["highlights"]
            rgb = app._neg_rgb_at(img, corr)
            assert rgb and len(rgb) == 3, "no negative-space RGB for corrected rect"
            wb = app._wb_for_patch(img, "highlights", rgb)
            assert wb and len(wb) == 3, "no live wb computed"
        step("live_wb", live_wb)

        # Global film-base override: take the roll-wide base from the current
        # frame; every other frame's Dmin is transferred via the exposure-factor
        # ratio, the snapshot persists in the source annotation, and clearing
        # reverts it.
        def global_base_override():
            src = app.images[app.current_idx]
            rgb = app._effective_film_base_rgb(src)
            fac = app._frame_factor(src)
            if not rgb or not fac:
                return   # frame without a usable base/EXIF — nothing to test
            app._set_global_base_from_current()
            assert app.global_base_override and \
                app.global_base_override["source_stem"] == src["stem"], \
                "global base not set"
            assert app.annotations[src["stem"]]["global_base"], "snapshot not stored"
            other = next((im for im in app.images
                          if im["stem"] != src["stem"]), None)
            if other is not None and app._frame_factor(other):
                exp = an.dmin_for_frame(rgb, fac, app._frame_factor(other))
                dmin = app._global_base_dmin(other)
                assert dmin and all(abs(a - b) < 1e-9 for a, b in zip(dmin, exp)), \
                    f"transfer wrong: {dmin} vs {exp}"
                params = app._corrected_params(other)
                assert params is not None and \
                    all(abs(a - b) < 1e-9 for a, b in zip(params["Dmin"], exp)), \
                    "override Dmin not in corrected params"
            app._clear_global_base_override()
            assert app.global_base_override is None, "override not cleared"
            assert not app.annotations[src["stem"]]["global_base"], \
                "snapshot not cleared"
        step("global_base_override", global_base_override)

        # Note on the selected patch
        def note():
            app._select_patch("highlights")
            app.set_selected_note("smoke note")
            app._auto_save(stem)
            assert app.annotations[stem]["patch_notes"].get("highlights") == "smoke note"
        step("note", note)

        # Copy params (Ctrl+C) from this frame, paste (Ctrl+V) onto another.
        # The ANNOTATED value is what gets copied, not the auto one; pasting
        # lands as print/wb overrides on the target. Restores both frames so
        # later assertions see untouched state.
        def copy_paste_params():
            if len(app.images) < 2:
                return
            src_idx = app.current_idx
            src_stem = app.images[src_idx]["stem"]
            sann = app.annotations[src_stem]
            saved_p = dict(sann["print_overrides"])
            saved_w = {k: list(v) for k, v in sann["wb_overrides"].items()}
            sann["print_overrides"]["gamma"] = 4.321          # annotated value
            sann["wb_overrides"]["shadows"] = [1.0, 0.9, 0.8]
            app._copy_params()
            clip = app.params_clipboard
            assert clip and clip["source_stem"] == src_stem, "clipboard not set"
            assert abs(clip["print"]["gamma"] - 4.321) < 1e-9, \
                "copy took the auto gamma, not the annotated value"
            assert clip["wb"]["shadows"] == [1.0, 0.9, 0.8], \
                "copy took the auto wb, not the annotated value"
            assert set(clip["print"]) == set(dbg.PRINT_PARAMS), \
                "not all print params captured"
            assert set(clip["wb"]) == set(dbg.WB_NAME_OVR.values()), \
                "not both wb wheels captured"
            tgt_idx = next(i for i in range(len(app.images)) if i != src_idx)
            app._nav_image(tgt_idx - src_idx)
            tgt = app.images[app.current_idx]
            tann = app.annotations[tgt["stem"]]
            saved_tp = dict(tann["print_overrides"])
            saved_tw = {k: list(v) for k, v in tann["wb_overrides"].items()}
            app._paste_params()
            assert abs(tann["print_overrides"]["gamma"] - 4.321) < 1e-9, \
                "paste did not set the gamma override on the target"
            assert tann["wb_overrides"]["shadows"] == [1.0, 0.9, 0.8], \
                "paste did not set the wb override on the target"
            params = app._corrected_params(tgt)
            assert params is not None and abs(params["gamma"] - 4.321) < 1e-9, \
                "pasted override not reflected in the target's corrected params"
            tann["print_overrides"].clear(); tann["print_overrides"].update(saved_tp)
            tann["wb_overrides"].clear(); tann["wb_overrides"].update(saved_tw)
            app._nav_image(src_idx - app.current_idx)
            sann["print_overrides"].clear(); sann["print_overrides"].update(saved_p)
            sann["wb_overrides"].clear(); sann["wb_overrides"].update(saved_w)
        step("copy_paste_params", copy_paste_params)

        # Ctrl+C must COPY (not clear) regardless of keyboard layout. The
        # physical-key dispatcher fires by key POSITION (keycode), so a non-Latin
        # layout — where <Control-c>'s keysym never matches — must still copy and
        # NOT fall through to the plain-c "clear correction". Drive _on_physical_key
        # directly with Ctrl held (state bit 0x4) on the C/V key codes.
        def ctrl_layout_independent():
            if sys.platform != "win32":
                return
            assert ord("C") in app._phys_ctrl_keymap \
                and ord("V") in app._phys_ctrl_keymap, \
                "Ctrl+C/V not routed through the physical-key dispatcher"
            saved = dict(app.annotations[stem]["print_overrides"])
            saved_clip = app.params_clipboard
            app.canvas.focus_set()                 # not a text widget: shortcuts live
            app._select_patch("gamma")
            app.annotations[stem]["print_overrides"]["gamma"] = 4.2
            app.params_clipboard = None
            app._on_physical_key(SimpleNamespace(state=0x4, keycode=ord("C")))
            assert app.params_clipboard is not None, "Ctrl+C did not copy"
            assert "gamma" in app.annotations[stem]["print_overrides"], \
                "Ctrl+C fell through to clear correction (the layout bug)"
            # plain C (no Ctrl) still clears the selected correction
            app._on_physical_key(SimpleNamespace(state=0x0, keycode=ord("C")))
            assert "gamma" not in app.annotations[stem]["print_overrides"], \
                "plain C no longer clears the correction"
            app.annotations[stem]["print_overrides"].clear()
            app.annotations[stem]["print_overrides"].update(saved)
            app.params_clipboard = saved_clip
        step("ctrl_layout_independent", ctrl_layout_independent)

        # Color-wheel footer sizing: the wheels must grow to fill the footer
        # pane WITHOUT pushing the darktable-entry boxes off the bottom (the
        # chrome height is reserved live, not hardcoded). Drive a resize and
        # assert both wheels + chrome fit inside the pane.
        def wheel_resize():
            app.root.update_idletasks()
            app._resize_wheels(SimpleNamespace(width=app.scaled(360),
                                               height=app.scaled(560)))
            sizes = [w.size for w in app.wheels.values()]
            assert all(s >= dbg.ColorWheel.MIN_SIZE for s in sizes), sizes
            # entries fit: 2 wheels + measured chrome stay within the pane
            new_chrome = (app._wheel_wrap.winfo_reqheight()
                          - sum(w.canvas.winfo_reqheight()
                                for w in app.wheels.values()))
            assert sum(sizes) + new_chrome <= app.scaled(560) + 2, \
                f"wheels+chrome overflow: {sizes} + {new_chrome} > {app.scaled(560)}"
        step("wheel_resize", wheel_resize)

        # Pane reflow: on a wide window the height-fit image leaves horizontal
        # slack; reflow must hand it to the item panel (not leave pillarbox).
        def pane_reflow():
            app.root.geometry("2200x760")
            app.root.update_idletasks()
            app._reflow_panes()
            app.root.update_idletasks()
            iw = app.item_pane.winfo_width()
            item_min = app.scaled(app.ITEM_PANEL_WIDTH)
            item_max = int(app.paned.winfo_width() * app.REFLOW_ITEM_MAX_FRAC)
            assert item_min - 2 <= iw <= item_max + 2, \
                f"item pane width {iw} outside [{item_min}, {item_max}]"
        step("pane_reflow", pane_reflow)

        # View + bad-inversion toggles
        step("view_negative", lambda: app._toggle_view())
        step("view_inverted", lambda: app._toggle_view())
        def bad():
            app._toggle_bad_inversion()
            assert app.annotations[stem]["bad_inversion"] is True
        step("bad_inversion", bad)

        # Clear correction for film_base, keep the highlights correction
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
            assert data["patch_corrections"]["highlights"]["corrected"], "no corrected rect"
            assert data["patch_notes"].get("highlights") == "smoke note"
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

        # Annotate+apply: applied_results.txt is written on close, one OK line
        # per frame with a 152-char param blob and a crop field (auto border
        # where no user crop), parseable the way the Lua side reads it.
        def verify_applied_results():
            ap = session / dbg.NegadoctorDebugUI.APPLIED_RESULTS_FILENAME
            assert ap.exists(), "applied_results.txt missing"
            lines = [l for l in ap.read_text().splitlines() if l.strip()]
            ok = [l for l in lines if l.startswith("OK|")]
            assert len(ok) == len(app.images), \
                f"expected {len(app.images)} OK lines, got {len(ok)}"
            for line in ok:
                _, stem, rest = line.split("|", 2)
                pj, cj, fj = rest.split("|")
                assert pj.startswith("params=") and len(pj) - 7 == 152, \
                    f"bad params blob: {pj[:20]}"
                cval = cj.split("=", 1)[1]
                if cval != "none":
                    parts = [float(v) for v in cval.split(",")]
                    assert len(parts) == 4 and parts[2] > parts[0] \
                        and parts[3] > parts[1], f"bad crop: {cval}"
                assert fj in ("flag=ok", "flag=bad"), f"bad flag: {fj}"
        step("verify_applied_results", verify_applied_results)

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
