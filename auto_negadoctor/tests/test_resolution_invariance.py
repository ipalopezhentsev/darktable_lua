"""Resolution-invariance guardrail.

The detection pipeline is designed to be resolution-independent: every
size-dependent constant is a fraction of the frame dimension (a *_FRAC),
turned into pixels with `int(round(w * FRAC))`. This test exists so that the
NEXT feature can't quietly reintroduce a raw, export-resolution-tied pixel
constant (the bug fixed 2026-06-13).

It feeds the detectors a frame at size W and a pixel-exact 2x copy of the SAME
frame, and asserts every output margin scales ~2x. Because it checks SCALING
(not absolute values), threshold tuning doesn't break it — only a constant that
fails to scale with resolution does. Fixture-free, so it always runs.

Run: conda run -n autocrop python auto_negadoctor/tests/test_resolution_invariance.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import auto_negadoctor as an

# margins may differ from an exact 2x by a couple px (rounding of each
# fraction*width plus the +pad); a real absolute-pixel constant breaks it by more
TOL_PX = 2

DMIN = np.array([1.0, 0.5, 0.4], dtype=np.float32)   # orange film base (linear)


def _synthetic_frame(w, h, border, rebate):
    """Linear-RGB frame engineered to exercise the border + content-crop
    detectors: a dark holder border on all sides, a film-rebate strip (base
    color) just inside the right border, and denser 'scene' content filling
    the rest. Returns (h, w, 3) float32."""
    lin = np.full((h, w, 3), 0.20, dtype=np.float32)        # gray content (dense)
    lin[:, w - border - rebate:w - border] = DMIN            # right rebate = base
    lin[:border, :] = 0.001                                  # top holder
    lin[h - border:, :] = 0.001                              # bottom holder
    lin[:, :border] = 0.001                                  # left holder
    lin[:, w - border:] = 0.001                              # right holder
    return lin


def _upscale_2x(lin):
    """Pixel-exact 2x (each pixel -> 2x2 block); no resampling, so any
    deviation from 2x scaling in the output comes purely from the code."""
    return np.repeat(np.repeat(lin, 2, axis=0), 2, axis=1)


def _check(name, m1, m2, results):
    """Assert each of the 4 margins in m2 is ~2x the matching one in m1."""
    for i, edge in enumerate(("left", "top", "right", "bottom")):
        expected = m1[i] * 2
        ok = abs(m2[i] - expected) <= TOL_PX
        results.append(ok)
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {name} {edge}: {m1[i]} -> {m2[i]} (expect ~{expected})")


def main():
    results = []

    print("no REF_WIDTH / _px leftovers (constants must be pure ratios):")
    leftovers = [n for n in ("REF_WIDTH", "_px") if hasattr(an, n)]
    ok = not leftovers
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] auto_negadoctor exposes no "
          f"{leftovers or 'REF_WIDTH/_px'}")

    lin = _synthetic_frame(480, 320, border=16, rebate=24)
    lin2 = _upscale_2x(lin)

    b1, b2 = an.detect_dark_border(lin), an.detect_dark_border(lin2)
    c1, c2 = an.detect_content_crop(lin, DMIN), an.detect_content_crop(lin2, DMIN)

    # guard against a vacuous test: the detectors must actually find margins
    nonzero = sum(b1) > 0 and sum(c1) > 0
    results.append(nonzero)
    print(f"\n  [{'PASS' if nonzero else 'FAIL'}] detectors produced non-zero "
          f"margins (border={b1}, crop={c1})")

    print("\ndetect_dark_border scales 2x:")
    _check("border", b1, b2, results)
    print("detect_content_crop scales 2x:")
    _check("crop", c1, c2, results)

    # film-base candidate: the rebate strip is the base rectangle. Its rect must
    # scale ~2x and its area FRACTION must stay constant across resolution.
    enc = np.zeros_like(lin)        # float-TIFF semantics: no clip flag
    enc2 = np.zeros_like(lin2)
    win1 = max(int(480 * an.DEFAULT_TUNING.BASE_WIN_FRAC),
               int(round(480 * an.DEFAULT_TUNING.MIN_WIN_FRAC)))
    win2 = max(int(960 * an.DEFAULT_TUNING.BASE_WIN_FRAC),
               int(round(960 * an.DEFAULT_TUNING.MIN_WIN_FRAC)))
    fb1 = an.find_film_base_candidate(lin, enc, win1)
    fb2 = an.find_film_base_candidate(lin2, enc2, win2)
    found = bool(fb1) and bool(fb2)
    results.append(found)
    print(f"\n  [{'PASS' if found else 'FAIL'}] film-base rectangle found at both "
          f"resolutions")
    if found:
        # The rectangle is located on a COARSE grid (cell = BASE_SCAN_STRIDE_FRAC
        # of width), so each edge can shift by up to ~one coarse cell (~4px at 2x)
        # — a quantization, not a resolution-tied constant (an absolute constant
        # would not scale at all). area_frac below is the strong invariance proof.
        base_tol = 2 * max(int(round(960 * an.DEFAULT_TUNING.BASE_SCAN_STRIDE_FRAC)), 1)
        print(f"find_film_base_candidate rect scales 2x (+/-{base_tol}px coarse-grid):")
        for i, edge in enumerate(("x", "y", "w", "h")):
            expected = fb1["rect"][i] * 2
            ok = abs(fb2["rect"][i] - expected) <= base_tol
            results.append(ok)
            print(f"  [{'PASS' if ok else 'FAIL'}] base {edge}: {fb1['rect'][i]} -> "
                  f"{fb2['rect'][i]} (expect ~{expected})")
        af_ok = abs(fb1["area_frac"] - fb2["area_frac"]) <= 0.01
        results.append(af_ok)
        print(f"  [{'PASS' if af_ok else 'FAIL'}] base area_frac stable: "
              f"{fb1['area_frac']:.4f} -> {fb2['area_frac']:.4f} (frame fraction, "
              f"must not change with resolution)")

    print()
    if all(results):
        print("ALL CHECKS PASSED")
        return 0
    print(f"FAILED ({results.count(False)}/{len(results)} checks) - a detection "
          "constant is tied to absolute resolution; make it a fraction of the "
          "frame dimension applied as int(round(w * FRAC)).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
