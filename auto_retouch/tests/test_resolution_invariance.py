"""Resolution-invariance guardrail for auto_retouch dust detection.

Every size-dependent detection constant is a FRACTION of the frame dimension
(a `*_FRAC` / `*_AREA_FRAC`), turned into pixels at the use site with
`min_dim * FRAC` (lengths/radii/kernels) or `min_dim**2 * FRAC` (areas). This
test exists so a future change can't quietly reintroduce a raw,
export-resolution-tied pixel constant: shoot the same frame on a higher-MP
camera and detection must find the SAME defects at the SAME relative places.

It renders a synthetic frame with a few dust-like dots (+ one bright stroke) on
a smooth background, runs `detect_dust_spots` on it at width W and a pixel-exact
2x copy, and asserts every detected spot scales ~2x (position + brush radius).
Because it checks SCALING (not absolute values), threshold tuning doesn't break
it — only a constant that fails to scale with resolution does. Fixture-free, so
it always runs.

Run: conda run -n autocrop python auto_retouch/tests/test_resolution_invariance.py
"""
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import detect_dust as d  # noqa: E402

# base frame min_dim; big enough that a dust-sized dot (area frac of min_dim**2)
# is comfortably inside [MIN_SPOT_AREA, MAX_SPOT_AREA] at BOTH W and 2W.
W, H = 2100, 1400
# dots as (cx_frac, cy_frac, radius_frac_of_min_dim) — spread over smooth areas
DOTS = [(0.25, 0.30, 0.0028), (0.55, 0.45, 0.0032),
        (0.72, 0.68, 0.0026), (0.40, 0.75, 0.0030)]
BG = 120          # smooth mid background (>= dark-bg floor)
DOT_VAL = 215     # bright achromatic dust


def _synthetic_frame(w, h):
    """Smooth gray frame + bright circular dots + one thin bright stroke, BGR."""
    md = min(w, h)
    img = np.full((h, w), BG, dtype=np.uint8)
    for fx, fy, fr in DOTS:
        cv2.circle(img, (int(fx * w), int(fy * h)), max(2, int(fr * md)), DOT_VAL, -1)
    # a bright diagonal stroke (thread/scratch); wide enough to be unambiguous at
    # both resolutions (a sub-2px line is at the discretization floor, not a
    # resolution-constant issue)
    cv2.line(img, (int(0.12 * w), int(0.20 * h)), (int(0.34 * w), int(0.13 * h)),
             DOT_VAL, max(3, int(0.0030 * md)))
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _detect(img):
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        cv2.imwrite(path, img)
        spots, _rej, err, _std = d.detect_dust_spots(path, collect_rejects=False)
        assert err is None, f"detection error: {err}"
        return spots
    finally:
        os.remove(path)


def _match_scale(s1, s2, results):
    """Each spot in s1 must have a partner in s2 at ~2x position; radius ~2x too.
    Tolerance is a few px (rounding of each fraction*dim + brush rounding)."""
    TOL = 6
    used = set()
    for a in s1:
        ax, ay = a["cx"] * 2, a["cy"] * 2
        best, bd = None, 1e9
        for j, b in enumerate(s2):
            if j in used:
                continue
            dist = abs(b["cx"] - ax) + abs(b["cy"] - ay)
            if dist < bd:
                bd, best = dist, j
        ok = best is not None and bd <= 2 * TOL
        results.append(ok)
        tag = "PASS" if ok else "FAIL"
        if ok:
            used.add(best)
            b = s2[best]
            r_ok = abs(b["brush_radius_px"] - a["brush_radius_px"] * 2) <= TOL
            results.append(r_ok)
            print(f"  [{tag}] {a['kind']} ({a['cx']:.0f},{a['cy']:.0f}) -> "
                  f"(~{ax:.0f},{ay:.0f}) matched; "
                  f"[{'PASS' if r_ok else 'FAIL'}] radius "
                  f"{a['brush_radius_px']:.1f}->{b['brush_radius_px']:.1f} (~2x)")
        else:
            print(f"  [{tag}] {a['kind']} ({a['cx']:.0f},{a['cy']:.0f}) no 2x match "
                  f"(nearest dist {bd:.0f})")


def main():
    results = []

    # 1) naming-convention guard: no module global is a bare *_PX pixel constant.
    px_leftovers = [n for n in dir(d)
                    if n.isupper() and n.endswith("_PX") and
                    isinstance(getattr(d, n), (int, float))]
    ok = not px_leftovers
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] no absolute *_PX module constants "
          f"({px_leftovers or 'none'})")

    lo = _synthetic_frame(W, H)
    hi = np.repeat(np.repeat(lo, 2, axis=0), 2, axis=1)   # pixel-exact 2x

    s1 = _detect(lo)
    s2 = _detect(hi)

    # 2) non-vacuous: the detector actually found the dust at both resolutions.
    nonzero = len(s1) >= len(DOTS) and len(s2) >= len(DOTS)
    results.append(nonzero)
    print(f"\n  [{'PASS' if nonzero else 'FAIL'}] found dust at both resolutions "
          f"(W: {len(s1)} spots, 2W: {len(s2)} spots; expected >= {len(DOTS)})")

    # 3) count matches (a raw-pixel size gate would drop/keep different spots at 2x)
    count_ok = len(s1) == len(s2)
    results.append(count_ok)
    print(f"  [{'PASS' if count_ok else 'FAIL'}] same spot count at W and 2W "
          f"({len(s1)} vs {len(s2)})")

    print("\nevery spot scales ~2x (position + brush radius):")
    _match_scale(s1, s2, results)

    print()
    if all(results):
        print("ALL CHECKS PASSED")
        return 0
    print(f"FAILED ({results.count(False)}/{len(results)} checks) - a detection "
          "constant is tied to absolute resolution; express it as a fraction of "
          "the frame dimension applied as min_dim * FRAC (min_dim**2 for areas).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
