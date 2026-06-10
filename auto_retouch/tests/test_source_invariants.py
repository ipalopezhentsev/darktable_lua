"""Self-test for run_quality_tests.check_spot_invariants — the source/geometry guard.

This is a "test for the test": it feeds synthetic spots with known-bad geometry and asserts
the invariant checker flags exactly the right ones. Keeps the source-clearance guard (which
protects healing tuning passes 16 & 22) honest, since that guard is the only thing validating
stroke healing sources (the committed baseline has no strokes to diff against).

Run:
    conda run -n autocrop python auto_retouch/tests/test_source_invariants.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_quality_tests as q

W = H = 500

CASES = [
    # (spot, expect_flagged, label)
    ({"kind": "dot", "cx": 100, "cy": 100, "src_cx": 105, "src_cy": 100,
      "brush_radius_px": 5, "radius_px": 2},
     True, "dot: source overlaps defect (sep 5 < 2*br 10)"),
    ({"kind": "dot", "cx": 100, "cy": 100, "src_cx": 130, "src_cy": 100,
      "brush_radius_px": 5, "radius_px": 2},
     False, "dot: clean source (sep 30 >= 2*br 10)"),
    ({"kind": "stroke", "cx": 35, "cy": 14, "src_cx": 10, "src_cy": 18,
      "brush_radius_px": 6, "path": [[10, 10], [60, 10]]},
     True, "stroke: source slid only 8px off a straight defect (< 2*br 12)"),
    ({"kind": "stroke", "cx": 35, "cy": 40, "src_cx": 10, "src_cy": 40,
      "brush_radius_px": 6, "path": [[10, 10], [60, 10]]},
     False, "stroke: source 30px clear of the defect"),
    ({"kind": "dot", "cx": 10, "cy": 10, "src_cx": -5, "src_cy": 10,
      "brush_radius_px": 4, "radius_px": 2},
     True, "dot: source out of bounds"),
    ({"kind": "stroke", "cx": 5, "cy": 5, "src_cx": 50, "src_cy": 50,
      "brush_radius_px": 4, "path": [[5, 5]]},
     True, "stroke: degenerate path (< 2 points)"),
    ({"kind": "dot", "cx": 10, "cy": 10, "src_cx": float("nan"), "src_cy": 10,
      "brush_radius_px": 4, "radius_px": 2},
     True, "dot: non-finite source coord"),
]


def main():
    spots = [c[0] for c in CASES]
    issues = q.check_spot_invariants(spots, W, H)
    flagged = {i["idx"] for i in issues}
    by_idx = {}
    for i in issues:
        by_idx.setdefault(i["idx"], []).append(i["msg"])

    ok = True
    for idx, (_spot, expect, label) in enumerate(CASES):
        got = idx in flagged
        mark = "ok " if got == expect else "FAIL"
        if got != expect:
            ok = False
        detail = f" -> {by_idx.get(idx, [])}" if got else ""
        print(f"[{mark}] case {idx}: expect_flagged={expect} got={got}  {label}{detail}")

    print("\nPASS" if ok else "\nFAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
