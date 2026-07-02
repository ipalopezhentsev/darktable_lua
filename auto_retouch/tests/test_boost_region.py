"""Pure-function guards for the Boost-region tool in the dust debug UI:
the sensitivity-relaxed cfg builder (`_boosted_cfg`) and the in-region merge of
the boosted detection into the frame's detected set (`_merge_boost_spots`).
Fixture-free, no images; builds a bare DustDebugUI via __new__ so only the two
methods under test run (they touch no other UI state).

Run: conda run -n autocrop python tests/test_boost_region.py
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FEATURE = os.path.dirname(HERE)
sys.path.insert(0, FEATURE)
sys.path.insert(0, os.path.dirname(FEATURE))   # repo root -> common

import detect_dust  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "dust_debug_ui", os.path.join(FEATURE, "debug_ui.py"))
dbg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dbg)


def _ok(cond, msg):
    if not cond:
        print(f"  [FAIL] {msg}")
        raise SystemExit(1)
    print(f"  [PASS] {msg}")


def _bare_ui():
    return dbg.DustDebugUI.__new__(dbg.DustDebugUI)


def test_boosted_cfg():
    ui = _bare_ui()
    base = detect_dust.DEFAULT_TUNING

    _ok(ui._boosted_cfg(base, 1.0) is base, "boost=1 is a no-op (returns base)")

    b = ui._boosted_cfg(base, 2.0)
    # every lower-is-more-sensitive gate must drop with a >1 boost — for BOTH the
    # dust-dot gates and the stroke (thread) gates (a boost must reach strokes too).
    for name in dbg.BOOST_RELAX_DIV + dbg.BOOST_STROKE_DIV:
        floor = dbg.BOOST_FLOOR.get(name)
        if floor is not None and getattr(base, name) / 2.0 < floor:
            _ok(getattr(b, name) == floor, f"{name} clamped to floor {floor}")
        else:
            _ok(getattr(b, name) < getattr(base, name),
                f"{name} relaxed downward ({getattr(base, name)} -> {getattr(b, name)})")
    # every maximum-cap must rise (or hit its ceiling)
    for name in dbg.BOOST_RELAX_MUL + dbg.BOOST_STROKE_MUL:
        _ok(getattr(b, name) >= getattr(base, name),
            f"{name} raised ({getattr(base, name)} -> {getattr(b, name)})")

    # the detection THRESHOLD is deliberately NOT relaxed (lowering it floods the
    # binary and makes yield non-monotonic in ×) — it must equal base.
    for name in ("NOISE_THRESHOLD_MULTIPLIER", "MIN_ABSOLUTE_THRESHOLD"):
        _ok(getattr(b, name) == getattr(base, name),
            f"{name} left at base (seed threshold not relaxed)")

    # types + ceilings preserved
    _ok(isinstance(b.MAX_SPOTS, int) and isinstance(b.MAX_NEARBY_ACCEPTED, int),
        "int fields stay int after boost")
    _ok(b.MAX_SPOTS <= dbg.BOOST_CAP["MAX_SPOTS"],
        "MAX_SPOTS never exceeds the darktable 300-form ceiling")
    _ok(b.STROKE_MAX_FILL_RATIO <= dbg.BOOST_CAP["STROKE_MAX_FILL_RATIO"],
        "STROKE_MAX_FILL_RATIO stays a fraction (< 1) after boost")
    # a large × must still clamp the elongation gate to its floor (a thread is a LINE)
    big = ui._boosted_cfg(base, 10.0)
    _ok(big.STROKE_MIN_ELONGATION >= dbg.BOOST_FLOOR["STROKE_MIN_ELONGATION"],
        "STROKE_MIN_ELONGATION never falls below its floor even at ×10")

    # base is unchanged (we replace onto a copy)
    _ok(detect_dust.DEFAULT_TUNING.NOISE_THRESHOLD_MULTIPLIER
        == base.NOISE_THRESHOLD_MULTIPLIER, "base tuning left untouched")


def test_merge_boost_spots():
    ui = _bare_ui()
    ui.images = [{"stem": "x", "width": 1000, "height": 1000,
                  "detected": [{"cx": 100, "cy": 100, "brush_radius_px": 5}]}]

    new_spots = [
        {"cx": 500, "cy": 500, "brush_radius_px": 6},                 # in region -> add
        {"cx": 900, "cy": 900, "brush_radius_px": 6},                 # outside -> skip
        {"kind": "stroke", "path": [[480, 480], [520, 520]],
         "brush_radius_px": 4},                                       # mid in region -> add
    ]
    added = ui._merge_boost_spots(0, new_spots, 400, 400, 600, 600, 2.0)
    det = ui.images[0]["detected"]
    _ok(added == 2, "only the 2 in-region spots are added")
    _ok(len(det) == 3, "existing spot preserved, 2 appended")

    boosted = [s for s in det if s.get("boosted")]
    _ok(len(boosted) == 2, "added spots carry the boosted flag")
    _ok(all(s.get("boost_factor") == 2.0 for s in boosted),
        "added spots record the boost factor")

    # appending does not disturb existing indices (index-based FP/overrides stay valid)
    _ok(det[0]["cx"] == 100 and det[0]["cy"] == 100,
        "the pre-existing detected spot stays at index 0")

    # re-running with an overlapping spot is de-duplicated (no double heal)
    again = ui._merge_boost_spots(
        0, [{"cx": 500, "cy": 500, "brush_radius_px": 6}], 400, 400, 600, 600, 3.0)
    _ok(again == 0, "a coincident spot is skipped on a second boost")


def test_boost_is_auto():
    ui = _bare_ui()

    class _V:
        def __init__(self, s): self.s = s
        def get(self): return self.s

    for val, want in (("", True), ("auto", True), ("AUTO", True), ("xyz", True),
                      ("2", False), ("2.5", False), ("10", False)):
        ui._boost_var = _V(val)
        _ok(ui._boost_is_auto() is want, f"_boost_is_auto({val!r}) == {want}")

    # the factor floors at 1 and has NO upper cap (used to clamp at 10)
    ui._boost_var = _V("50")
    _ok(ui._boost_factor() == 50.0, "boost factor is uncapped (50 stays 50)")
    ui._boost_var = _V("0.2")
    _ok(ui._boost_factor() == 1.0, "boost factor floored at 1")


def test_rep_point_and_attribution():
    ui = _bare_ui()
    base = detect_dust.DEFAULT_TUNING

    _ok(dbg.DustDebugUI._spot_rep_point({"cx": 5, "cy": 7}) == (5.0, 7.0),
        "rep point of a dot is its centre")
    mid = dbg.DustDebugUI._spot_rep_point(
        {"kind": "stroke", "path": [[0, 0], [10, 10], [20, 20]]})
    _ok(mid == (10.0, 10.0), "rep point of a stroke is its mid path node")

    # a dot with a contrast/texture ratio below base must be attributed to that gate
    low = {"contrast": base.MIN_CONTRAST_TEXTURE_RATIO * 1.0,
           "texture": 2.0, "area": 50}
    helped = ui._boost_responsible_params(low, base, 4000)
    _ok(any("MIN_CONTRAST_TEXTURE_RATIO" in h for h in helped),
        "boosted dot attribution names the tuning constant (MIN_CONTRAST_TEXTURE_RATIO)")

    # a comfortably-clean dot attributes nothing (deciding gate not recorded)
    clean = {"contrast": base.MIN_CONTRAST_TEXTURE_RATIO * 100,
             "texture": 1.0, "area": int(base.MIN_SPOT_AREA_FRAC * 4000 * 4000 * 5)}
    _ok(ui._boost_responsible_params(clean, base, 4000) == [],
        "a spot clearing every checkable gate at base attributes nothing")

    # a short stubby stroke is attributed to the length + elongation gates
    stub = {"kind": "stroke", "length_px": 5.0, "stroke_width_px": 4.0,
            "contrast": 30, "texture": 1.0, "context_texture": 1.0,
            "excess_sat": 0.0, "crispness": 1.0}
    h2 = ui._boost_responsible_params(stub, base, 4000)
    _ok(any("STROKE_MIN_LENGTH_FRAC" in h for h in h2)
        and any("STROKE_MIN_ELONGATION" in h for h in h2),
        "short stubby stroke attribution names STROKE_MIN_LENGTH_FRAC + STROKE_MIN_ELONGATION")


def test_boost_note_for():
    ui = _bare_ui()
    ui._effective_cfg = lambda: detect_dust.DEFAULT_TUNING   # bare instance stub
    img = {"width": 4000, "height": 4000}

    # non-boosted -> empty note
    _ok(ui._boost_note_for({"cx": 1, "cy": 2, "contrast": 50, "texture": 1}, img) == "",
        "no note for a non-boosted spot")

    # a boosted STROKE (like the one clicked via its node) gets the ⚡ note + gates
    stroke = {"kind": "stroke", "boosted": True, "boost_factor": 2.0,
              "length_px": 5.0, "stroke_width_px": 4.0, "contrast": 30,
              "texture": 1.0, "context_texture": 1.0, "excess_sat": 0.0,
              "crispness": 1.0}
    note = ui._boost_note_for(stroke, img)
    _ok("BOOSTED ×2" in note and "relaxed gates" in note,
        "boosted stroke node/source shows the ⚡ BOOSTED note + attribution")


def test_boost_tool_toggle():
    import types
    ui = _bare_ui()

    class _V:
        def get(self): return "2.0"
    ui._boost_var = _V()
    ui.canvas = types.SimpleNamespace(delete=lambda *a: None)
    ui.thread_draw_mode = False
    ui.thread_draw_points = []
    ui.boost_region_mode = False
    ui._boost_auto = False
    ui._boost_drag_active = False
    ui._clear_selection = lambda: None
    ui._set_info_text = lambda *a, **k: None
    ui._update_boost_btn = lambda: None
    ui._redraw_markers = lambda: None

    ui._toggle_boost_region(auto=False)
    _ok(ui.boost_region_mode and not ui._boost_auto, "⊕ arms manual boost")
    ui._toggle_boost_region(auto=True)
    _ok(ui.boost_region_mode and ui._boost_auto,
        "⚡ switches to auto while staying armed")
    ui._toggle_boost_region(auto=True)
    _ok(not ui.boost_region_mode, "clicking the same (auto) flavour disarms")
    ui._toggle_boost_region(auto=True)
    ui._toggle_boost_region(auto=False)
    _ok(ui.boost_region_mode and not ui._boost_auto,
        "⊕ switches auto->manual while staying armed")


if __name__ == "__main__":
    test_boosted_cfg()
    test_merge_boost_spots()
    test_boost_is_auto()
    test_rep_point_and_attribution()
    test_boost_note_for()
    test_boost_tool_toggle()
    print("\nALL BOOST-REGION TESTS PASSED")
