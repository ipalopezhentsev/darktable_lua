"""Regression tests for the roll-wide vignette estimator (pure math, no images).

Exercises `auto_negadoctor.fit_vignette_profile()` on radial-envelope profiles
captured from two real rolls, so it needs none of the (uncommitted, ~1.2GB)
TIFF exports:

  - roll 2512-2601-1 ("roll 1", the reference roll): the envelope is brightest
    at the dead centre and falls monotonically — the long-standing good case.
  - roll 2511-12-1 ("roll 2"): a genuine ~29% vignette whose envelope PEAKS just
    off-centre (innermost bins are slightly dimmer/noisier), so the innermost
    `target` reads >1 falling to 1 at the peak. The old monotone-tail cut
    started at bin 0, mistook that central dip for the corner-leak tail, cut
    after 2 bins and rejected the whole roll ("profile not vignette-like") — so
    no correction was applied and the inverted corners came out too bright. The
    fix anchors the cut at the envelope peak; this test pins that behaviour.

Run: conda run -n autocrop python auto_negadoctor/tests/test_vignette.py
Exit code 0 = all pass.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import auto_negadoctor as an

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}{('  ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILURES.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# Radial-envelope profiles captured by replaying estimate_vignette's
# accumulation on the two rolls' linear-Rec2020 TIFF exports (37 frames each).
# R = bin-centre radii in [0,1]; E = per-bin envelope brightness.

# roll 2512-2601-1 — centre-brightest, the reference good case.
ROLL_2512_2601_1_R = [0.0156, 0.0469, 0.0781, 0.1094, 0.1406, 0.1719, 0.2031,
    0.2344, 0.2656, 0.2969, 0.3281, 0.3594, 0.3906, 0.4219, 0.4531, 0.4844,
    0.5156, 0.5469, 0.5781, 0.6094, 0.6406, 0.6719, 0.7031, 0.7344, 0.7656,
    0.7969, 0.8281, 0.8594, 0.8906, 0.9219, 0.9531]
ROLL_2512_2601_1_E = [88.5672, 87.9861, 87.4434, 87.1555, 86.5714, 86.2633,
    85.9865, 85.5534, 84.889, 84.3157, 83.4269, 82.6782, 82.0233, 81.2566,
    80.3823, 79.6461, 78.7677, 78.2133, 77.4353, 76.4695, 75.6695, 74.5866,
    73.1663, 71.6903, 70.7212, 83.373, 83.3272, 79.5558, 80.1409, 106.6574,
    108.9449]

# roll 2511-12-1 — envelope peaks at r=0.078 (innermost two bins dimmer): the
# central-dip case the old cut rejected. Corner-leak tail at the last two bins.
ROLL_2511_12_1_R = list(ROLL_2512_2601_1_R)
ROLL_2511_12_1_E = [84.0253, 85.3364, 87.0038, 86.8614, 86.6443, 86.4116,
    86.0966, 85.6931, 85.2595, 84.7745, 84.299, 83.8638, 83.334, 82.6966,
    82.1052, 81.3165, 80.6659, 80.0252, 79.2848, 78.4295, 77.4603, 76.4441,
    75.2754, 74.0405, 72.3799, 70.4537, 69.2463, 67.1851, 65.5846, 71.7916,
    111.2731]


def test_reference_roll_unchanged():
    print("roll 2512-2601-1 (reference, centre-brightest):")
    params, info = an.fit_vignette_profile(
        ROLL_2512_2601_1_R, ROLL_2512_2601_1_E, used=37)
    check("vignette found", params is not None, str(info.get("reason")))
    if params:
        check("strength 0.525", approx(params["strength"], 0.525, 1e-3),
              str(params["strength"]))
        check("radius 0.05", approx(params["radius"], 0.05, 1e-3),
              str(params["radius"]))
        check("steepness 0.375", approx(params["steepness"], 0.375, 1e-3),
              str(params["steepness"]))
        check("corner falloff ~28%", approx(info["corner_falloff"], 0.2825, 0.01),
              str(info.get("corner_falloff")))


def test_central_dip_roll_now_corrected():
    print("roll 2511-12-1 (off-centre peak — was wrongly rejected):")
    params, info = an.fit_vignette_profile(
        ROLL_2511_12_1_R, ROLL_2511_12_1_E, used=37)
    # The whole point of the fix: this roll must NOT be rejected.
    check("not rejected as 'profile not vignette-like'",
          params is not None,
          str(info.get("reason")))
    if params:
        # genuine, meaningful correction (corners brightened in the negative ->
        # darker after inversion). Exact values pin the current fit.
        check("strength 0.25", approx(params["strength"], 0.25, 1e-3),
              str(params["strength"]))
        check("radius 0.125", approx(params["radius"], 0.125, 1e-3),
              str(params["radius"]))
        check("steepness 0.825", approx(params["steepness"], 0.825, 1e-3),
              str(params["steepness"]))
        check("corner falloff ~29%", approx(info["corner_falloff"], 0.292, 0.01),
              str(info.get("corner_falloff")))
        check("corner-leak tail trimmed (29 of 31 bins kept)",
              info["bins"] == 29, f"bins={info.get('bins')}")


def test_corner_leak_tail_still_cut():
    """A clean monotone-rising profile with a falling corner-leak tail must keep
    the rising part and drop the tail (the cut's original purpose)."""
    print("synthetic corner-leak tail:")
    r = [(k + 0.5) / 16 for k in range(16)]
    # envelope falls smoothly to the corner, then two bins jump back up (leak)
    e = [100.0 - 1.5 * k for k in range(14)] + [120.0, 130.0]
    params, info = an.fit_vignette_profile(r, e, used=10)
    check("vignette found", params is not None, str(info.get("reason")))
    if params:
        check("tail dropped (<=14 bins kept)", info["bins"] <= 14,
              f"bins={info.get('bins')}")


def test_frame_cache_envelope_identical():
    """The decode-once cache (vignette_frame_cache + vignette_envelope(...,
    frame_cache=)) must reproduce the decode-each-trial envelope EXACTLY — this
    is the speedup the calibration runner's vignette FULL path relies on. Driven
    by a monkeypatched load_frame so it needs no TIFFs."""
    print("frame-cache envelope equivalence (no TIFFs):")
    import numpy as np

    # Two synthetic frames: a centred radial falloff + a bit of per-frame noise,
    # as float exports (enc_f.max()==0 -> no clip mask), with a dark border.
    def synth(seed):
        rng = np.random.default_rng(seed)
        h, w = 120, 160
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        rad = np.hypot(xx - cx, yy - cy) / np.hypot(cx, cy)
        base = (1.0 - 0.4 * rad ** 2) * (0.8 + 0.1 * seed)
        lin = np.repeat((base + rng.normal(0, 0.002, base.shape))[:, :, None],
                        3, axis=2)
        lin[:6, :] = 0.0; lin[-6:, :] = 0.0   # dark holder border top/bottom
        lin[:, :6] = 0.0; lin[:, -6:] = 0.0
        enc = np.zeros_like(lin)
        return enc, lin

    frames = {"a.tif": synth(1), "b.tif": synth(2), "c.tif": synth(3)}
    paths = list(frames)
    orig_lf, orig_exif = an.load_frame, an.read_exif_fallback
    an.load_frame = lambda p, vig=None: frames[__import__("os").path.basename(p)]
    an.read_exif_fallback = lambda p: {}
    try:
        env_fresh = an.vignette_envelope(paths, {})           # decode each call
        fc = an.vignette_frame_cache(paths, {})               # decode once
        env_cached = an.vignette_envelope(paths, {}, frame_cache=fc)
    finally:
        an.load_frame, an.read_exif_fallback = orig_lf, orig_exif

    check("cache build decoded all frames",
          sum(c is not None for c in fc) == len(paths))
    check("envelope radii identical", env_fresh["r"] == env_cached["r"])
    check("envelope values identical (bit-for-bit)", env_fresh["e"] == env_cached["e"])
    check("frame count identical", env_fresh["used"] == env_cached["used"])


def main():
    test_reference_roll_unchanged()
    test_central_dip_roll_now_corrected()
    test_corner_leak_tail_still_cut()
    test_frame_cache_envelope_identical()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {', '.join(FAILURES)}")
        sys.exit(1)
    print("ALL VIGNETTE TESTS PASSED")


if __name__ == "__main__":
    main()
