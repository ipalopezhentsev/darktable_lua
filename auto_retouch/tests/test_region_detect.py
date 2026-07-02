"""Guards for `detect_dust.detect_region` — the region-only (cropped) detection
behind the debug UI's Boost-region tool. It analyses a rectangle + context margin
for speed, but must stay FAITHFUL to a full-frame detect: frame-fraction
thresholds anchor to the FULL min_dim and the noise/brightness references are
measured on the FULL frame (the detector knows the region is a sub-part of the
image, not 100% of it).

Faithfulness direction that matters: every spot a FULL detect finds inside the
rectangle must also be found by the region detect. (The region detect MAY surface
extra spots — a tighter crop sees fewer isolation neighbours — which is exactly
why the tool is useful; those are the user's to keep or mark FP.)

Needs one fixture JPG (gitignored / regenerable); SKIPS cleanly when none is
present. Runs one full-frame detect (~25s), so it is a SLOW guard.

Run: conda run -n autocrop python tests/test_region_detect.py
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import detect_dust as d  # noqa: E402


def _ok(cond, msg):
    if not cond:
        print(f"  [FAIL] {msg}")
        raise SystemExit(1)
    print(f"  [PASS] {msg}")


def _find_fixture_jpg():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "rolls")
    for p in sorted(glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True)):
        return p
    return None


def _near(ax, ay, spots, tol=8.0):
    for b in spots:
        if b.get("kind") == "stroke":
            continue
        if (ax - b["cx"]) ** 2 + (ay - b["cy"]) ** 2 <= tol * tol:
            return True
    return False


def main():
    jpg = _find_fixture_jpg()
    if not jpg:
        print("  [SKIP] no fixture JPG present (gitignored) — region detect not exercised")
        return
    import cv2
    H, W = cv2.imread(jpg).shape[:2]
    print(f"  fixture {os.path.basename(jpg)}  {W}x{H}")

    full, _rj, err, _l = d.detect(jpg, collect_rejects=False)
    _ok(err is None and full is not None, "full-frame detect ran")
    dots = [s for s in full if s.get("kind") != "stroke"]
    if not dots:
        print("  [SKIP] full detect found no dot spots on this fixture")
        return

    # A rect around a central dust dot, sized so several dots fall inside.
    dots.sort(key=lambda s: (s["cx"] - W / 2) ** 2 + (s["cy"] - H / 2) ** 2)
    px, py = dots[len(dots) // 2]["cx"], dots[len(dots) // 2]["cy"]
    R = int(min(W, H) * 0.11)
    rect = (px - R, py - R, px + R, py + R)
    x0, y0, x1, y1 = rect

    reg, _rj2, err2, _l2 = d.detect_region(jpg, rect)
    _ok(err2 is None and reg is not None, "region detect ran")
    print(f"  rect {tuple(int(v) for v in rect)}: full-inside vs region {len(reg)} spot(s)")

    # (1) every region spot is inside the rect, in FULL-frame coordinates
    for s in reg:
        cx, cy = (s["cx"], s["cy"]) if s.get("kind") != "stroke" else _stroke_mid(s)
        _ok(x0 <= cx <= x1 and y0 <= cy <= y1,
            f"region spot ({int(cx)},{int(cy)}) inside rect (full-frame coords)")

    # (2) determinism
    again, _rj3, _e3, _l3 = d.detect_region(jpg, rect)
    _ok(len(again) == len(reg), "same rectangle yields the same spot count")

    # (3) FAITHFULNESS: every full-detect dot well inside the rect (inner border,
    # to avoid edge-clipping ambiguity) is reproduced by the region detect.
    b = 64
    inner = [s for s in dots
             if x0 + b <= s["cx"] <= x1 - b and y0 + b <= s["cy"] <= y1 - b]
    lost = [s for s in inner if not _near(s["cx"], s["cy"], reg, tol=8.0)]
    for s in lost[:5]:
        print(f"    LOST full spot ({int(s['cx'])},{int(s['cy'])}) ctr={s['contrast']:.0f}")
    _ok(not lost,
        f"all {len(inner)} full-detect dots inside the rect are reproduced by region detect")

    print("\nALL REGION-DETECT TESTS PASSED")


def _stroke_mid(s):
    path = s.get("path") or []
    mid = path[len(path) // 2]
    return float(mid[0]), float(mid[1])


if __name__ == "__main__":
    main()
