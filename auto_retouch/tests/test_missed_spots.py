"""Pure-function guards for the missed-dust / missed-stroke -> healable-spot
conversion used by the apply-from-folder flow (debug_ui apply_mode and the live
preview both go through these). Fixture-free, no Tk, no images — always runs.

Run: conda run -n autocrop python tests/test_missed_spots.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import detect_dust as d


def _ok(cond, msg):
    if not cond:
        print(f"  [FAIL] {msg}")
        raise SystemExit(1)
    print(f"  [PASS] {msg}")


def test_missed_dust_to_spot():
    dets = [{"brush_radius_px": 10.0}, {"brush_radius_px": 20.0},
            {"brush_radius_px": 30.0}]

    # legacy {cx,cy}-only -> sized from the detected MEDIAN, kind=dot
    s = d.missed_dust_to_spot({"cx": 100, "cy": 200}, dets, None)
    _ok(s["kind"] == "dot" and s["cx"] == 100 and s["cy"] == 200, "dot at click coords")
    _ok(abs(s["brush_radius_px"] - 20.0) < 1e-6, "radius = median of detected (20)")
    _ok("src_cx" not in s, "no source without buffers (XMP writer uses fixed offset)")

    # user-stored radius + source are HONORED
    s2 = d.missed_dust_to_spot(
        {"cx": 5, "cy": 6, "brush_radius_px": 42.0, "src_cx": 1, "src_cy": 2}, dets, None)
    _ok(abs(s2["brush_radius_px"] - 42.0) < 1e-6, "stored radius honored")
    _ok(s2["src_cx"] == 1.0 and s2["src_cy"] == 2.0, "stored source honored")

    # no detections + no buffers -> frame-relative floor (>= MIN_BRUSH_FRAC*min_dim)
    min_brush = d.MIN_BRUSH_FRAC * 4000
    s3 = d.missed_dust_to_spot({"cx": 0, "cy": 0}, [], None, min_dim=4000)
    _ok(s3["brush_radius_px"] >= min_brush, "fallback radius >= MIN_BRUSH_FRAC*min_dim")

    # radius is clamped to MIN_BRUSH_FRAC*min_dim even if a tiny value is stored
    s4 = d.missed_dust_to_spot({"cx": 0, "cy": 0, "brush_radius_px": 0.1}, dets, None, min_dim=4000)
    _ok(abs(s4["brush_radius_px"] - min_brush) < 1e-6, "tiny stored radius clamped to MIN_BRUSH_FRAC*min_dim")


def test_missed_stroke_to_spot():
    ms = {"path": [[10.0, 10.0], [100.0, 12.0], [200.0, 8.0]], "stroke_width_px": 6.0}
    sp = d.missed_stroke_to_spot(ms, min_dim=4000)
    _ok(sp is not None and sp["kind"] == "stroke", "stroke spot built")
    _ok(len(sp["path"]) == 3, "path carried through")
    _ok(sp["brush_radius_px"] >= d.DEFAULT_TUNING.STROKE_MIN_BORDER_FRAC * 4000,
        "stroke brush >= STROKE_MIN_BORDER_FRAC*min_dim")

    # too-short path -> None (nothing to heal)
    _ok(d.missed_stroke_to_spot({"path": [[1, 1]]}, 4000) is None, "1-point path -> None")
    _ok(d.missed_stroke_to_spot({"path": []}, 4000) is None, "empty path -> None")


def test_prepare_source_buffers_signature():
    _ok(callable(d.prepare_source_buffers), "prepare_source_buffers is callable")
    _ok(d.prepare_source_buffers("/no/such/file.jpg") is None,
        "prepare_source_buffers returns None on a missing image")


if __name__ == "__main__":
    test_missed_dust_to_spot()
    test_missed_stroke_to_spot()
    test_prepare_source_buffers_signature()
    print("\nALL MISSED-SPOT CONVERSION TESTS PASSED")
