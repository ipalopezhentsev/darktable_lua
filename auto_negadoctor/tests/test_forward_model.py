"""Unit tests for nega_model.py (pure math, no images).

Run: conda run -n autocrop python auto_negadoctor/tests/test_forward_model.py
Exit code 0 = all pass.
"""

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import nega_model as nm

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}{('  ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILURES.append(name)


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


# Real negadoctor params hex from a manually tuned XMP (DSC_0028.nef.xmp)
REAL_HEX = (
    "01000000beb74d3f0955ab3e98962f3e000000009999b93fad47a13f5c8f823f0000803f"
    "ffff7f3f09d7633f5b8f423f0000803faaf10240cdcc4cbd60074e3eccccdc40e158f73e6cc9813f"
)


def test_encode_decode():
    print("encode/decode:")
    p = nm.default_params()
    h = nm.encode_negadoctor_params(p)
    check("hex length 152", len(h) == 152, str(len(h)))
    d = nm.decode_negadoctor_params(h)
    for k in ("D_max", "offset", "black", "gamma", "soft_clip", "exposure"):
        check(f"round-trip {k}", approx(d[k], p[k], 1e-6), f"{d[k]} vs {p[k]}")
    for k in ("Dmin", "wb_high", "wb_low"):
        check(f"round-trip {k}", all(approx(a, b, 1e-6) for a, b in zip(d[k], p[k])))
    check("round-trip film_stock", d["film_stock"] == 1)

    r = nm.decode_negadoctor_params(REAL_HEX)
    check("real: film_stock=1", r["film_stock"] == 1)
    check("real: offset=-0.05", approx(r["offset"], -0.05, 1e-6), str(r["offset"]))
    check("real: D_max~2.046", approx(r["D_max"], 2.0460, 1e-3), str(r["D_max"]))
    check("real: gamma=6.9", approx(r["gamma"], 6.9, 1e-4), str(r["gamma"]))
    check("real: Dmin orange (R>G>B)", r["Dmin"][0] > r["Dmin"][1] > r["Dmin"][2], str(r["Dmin"]))
    check("real: re-encode identical", nm.encode_negadoctor_params(r) == REAL_HEX)


def test_exposure_factor():
    print("exposure_factor:")
    f1, m1 = nm.exposure_factor(1 / 100, 100, 5.6)
    f2, m2 = nm.exposure_factor(1 / 50, 100, 5.6)
    f3, m3 = nm.exposure_factor(1 / 100, 100, 8.0)
    check("no missing flags", not (m1 or m2 or m3))
    check("2x shutter = 2x light", approx(f2 / f1, 2.0, 1e-9), str(f2 / f1))
    # nominal f-numbers are rounded (f/5.6 is really sqrt(32)), so 0.49 not 0.50
    check("f5.6->f8 ~ half light", abs(f3 / f1 - 0.5) < 0.02, str(f3 / f1))
    f4, m4 = nm.exposure_factor(None, 100, 5.6)
    check("missing EXIF -> 1.0 + flag", f4 == 1.0 and m4)


def test_colorspace():
    print("colorspace:")
    x = np.array([0, 64, 128, 200, 255], dtype=np.uint8)
    lin = nm.srgb_to_linear(x)
    back = nm.linear_to_srgb(lin) * 255.0
    check("srgb round-trip", np.allclose(back, x.astype(np.float64), atol=1e-6))
    check("mid-gray linear ~0.2158", approx(lin[2], 0.21586, 1e-4), str(lin[2]))
    white = np.ones(3) @ nm.SRGB_TO_REC2020.T
    check("matrix preserves white", np.allclose(white, 1.0, atol=1e-5), str(white))


def soft_clip_expected(g, soft_clip):
    if g <= soft_clip:
        return g
    comp = 1.0 - soft_clip
    return soft_clip + (1.0 - math.exp(-(g - soft_clip) / comp)) * comp


def test_tuner_roundtrip_neutral():
    """Scenario (a): per-channel densities equal -> wb stays 1, exact targets."""
    print("tuner round-trip (neutral densities):")
    dmin = np.array([0.80, 0.40, 0.20])
    pmin = list(dmin * 10.0 ** -2.0)   # densest area, 2.0 D above base
    pmax = list(dmin * 10.0 ** 0.1)    # lightest area, slightly lighter than base

    d_max = nm.compute_dmax(dmin, pmin)
    check("D_max = 2.0", approx(d_max, 2.0, 1e-9), str(d_max))
    offset = nm.compute_offset(dmin, pmax, d_max)
    check("offset = -0.05", approx(offset, -0.05, 1e-9), str(offset))

    wb_low = nm.compute_wb_low(dmin, list(dmin * 10.0 ** -0.5), d_max)
    wb_high = nm.compute_wb_high(dmin, list(dmin * 10.0 ** -1.5), d_max, offset, wb_low)
    check("wb_low all 1", all(approx(v, 1.0, 1e-9) for v in wb_low), str(wb_low))
    check("wb_high all 1", all(approx(v, 1.0, 1e-9) for v in wb_high), str(wb_high))

    black = nm.compute_black(dmin, pmax, d_max, wb_high, wb_low, offset)
    check("black = 0.1", approx(black, 0.1, 1e-9), str(black))
    exposure = nm.compute_exposure(dmin, pmin, d_max, wb_high, wb_low, offset, black)

    params = {
        "film_stock": 1, "Dmin": list(dmin), "wb_high": wb_high, "wb_low": wb_low,
        "D_max": d_max, "offset": offset, "black": black,
        "gamma": nm.GAMMA_DEFAULT, "soft_clip": nm.SOFT_CLIP_DEFAULT,
        "exposure": exposure,
    }

    # Densest negative area must print at 0.96 pre-gamma
    out_min = nm.render_negadoctor(np.array([pmin]), params)[0]
    expected = soft_clip_expected(0.96 ** nm.GAMMA_DEFAULT, nm.SOFT_CLIP_DEFAULT)
    check("densest -> 0.96 target", np.allclose(out_min, expected, atol=1e-6),
          f"{out_min} vs {expected}")

    # Lightest area prints at 0.1*exposure pre-gamma
    out_max = nm.render_negadoctor(np.array([pmax]), params)[0]
    expected_max = (0.1 * exposure) ** nm.GAMMA_DEFAULT
    check("lightest -> 0.1*exposure target", np.allclose(out_max, expected_max, atol=1e-6),
          f"{out_max} vs {expected_max}")

    # Film base itself renders near black
    out_base = nm.render_negadoctor(np.array([list(dmin)]), params)[0]
    check("film base near black", float(out_base.max()) < 0.01, str(out_base))

    # Monotonicity: denser negative -> brighter print
    ds = np.linspace(0.0, 2.0, 21)
    ladder = np.stack([dmin * 10.0 ** -d for d in ds])
    outs = nm.render_negadoctor(ladder, params).mean(axis=1)
    check("monotonic inversion", bool(np.all(np.diff(outs) >= -1e-12)))


def test_tuner_color_cast():
    """Scenario (b): unequal channel densities -> wb correction kicks in."""
    print("tuner with color casts:")
    dmin = np.array([0.80, 0.40, 0.20])
    pmin = list(dmin * np.power(10.0, [-2.0, -1.9, -1.8]))
    pmax = list(dmin * np.power(10.0, [0.10, 0.08, 0.06]))
    shadow = list(dmin * np.power(10.0, [-0.50, -0.45, -0.40]))
    white = list(dmin * np.power(10.0, [-1.60, -1.50, -1.40]))

    d_max = nm.compute_dmax(dmin, pmin)
    offset = nm.compute_offset(dmin, pmax, d_max)
    wb_low = nm.compute_wb_low(dmin, shadow, d_max)
    wb_high = nm.compute_wb_high(dmin, white, d_max, offset, wb_low)
    black = nm.compute_black(dmin, pmax, d_max, wb_high, wb_low, offset)
    exposure = nm.compute_exposure(dmin, pmin, d_max, wb_high, wb_low, offset, black)

    check("max(wb_low) == 1", approx(max(wb_low), 1.0, 1e-9), str(wb_low))
    check("min(wb_high) == 1", approx(min(wb_high), 1.0, 1e-9), str(wb_high))
    check("wb_low in range", all(nm.WB_RANGE[0] <= v <= nm.WB_RANGE[1] for v in wb_low))
    check("wb_high in range", all(nm.WB_RANGE[0] <= v <= nm.WB_RANGE[1] for v in wb_high))
    check("D_max in range", nm.DMAX_RANGE[0] <= d_max <= nm.DMAX_RANGE[1])
    check("offset in range", nm.OFFSET_RANGE[0] <= offset <= nm.OFFSET_RANGE[1])
    check("black in range", nm.BLACK_RANGE[0] <= black <= nm.BLACK_RANGE[1])
    check("exposure in range", nm.EXPOSURE_RANGE[0] <= exposure <= nm.EXPOSURE_RANGE[1])

    params = {
        "film_stock": 1, "Dmin": list(dmin), "wb_high": wb_high, "wb_low": wb_low,
        "D_max": d_max, "offset": offset, "black": black,
        "gamma": nm.GAMMA_DEFAULT, "soft_clip": nm.SOFT_CLIP_DEFAULT,
        "exposure": exposure,
    }

    # wb_high exactly neutralizes the white patch (channel spread ~0)
    out_white = nm.render_negadoctor(np.array([white]), params)[0]
    spread = float(out_white.max() - out_white.min())
    check("white patch neutralized", spread < 1e-6, f"spread={spread} out={out_white}")

    # wb_low improves shadow-patch neutrality vs wb_low=1
    params_no_low = dict(params, wb_low=[1.0, 1.0, 1.0])
    out_sh = nm.render_negadoctor(np.array([shadow]), params)[0]
    out_sh_no = nm.render_negadoctor(np.array([shadow]), params_no_low)[0]
    rel = lambda o: (o.max() - o.min()) / max(o.mean(), 1e-9)
    check("wb_low reduces shadow cast", rel(out_sh) <= rel(out_sh_no) + 1e-9,
          f"{rel(out_sh)} vs {rel(out_sh_no)}")

    # encode/decode round-trip of computed params
    d = nm.decode_negadoctor_params(nm.encode_negadoctor_params(params))
    check("computed params survive blob", approx(d["exposure"], exposure, 1e-6)
          and all(approx(a, b, 1e-6) for a, b in zip(d["wb_high"], wb_high)))


def test_vignette():
    print("vignette model:")
    # exact center/corner behavior of darktable's manual-vignette spline
    m0 = float(nm.vignette_spline(0.0, 0.75, 0.097, 0.5))
    m1 = float(nm.vignette_spline(1.0, 0.75, 0.097, 0.5))
    check("center untouched", approx(m0, 1.0, 1e-9), str(m0))
    check("corner = 1+2*s*st", approx(m1, 1.0 + 2 * 0.75 * 0.5, 1e-9), str(m1))
    rs = np.linspace(0, 1, 50)
    ms = nm.vignette_spline(rs, 0.75, 0.097, 0.5)
    check("monotone increasing", bool(np.all(np.diff(ms) >= -1e-12)))

    field = nm.vignette_field(101, 151, {"strength": 0.5, "radius": 0.2,
                                         "steepness": 0.6})
    check("field center ~1", approx(float(field[50, 75]), 1.0, 1e-3))
    corners = [field[0, 0], field[0, -1], field[-1, 0], field[-1, -1]]
    check("field corners equal", max(corners) - min(corners) < 1e-9)
    check("no-vignette field is ones",
          float(nm.vignette_field(10, 10, None).max()) == 1.0)

    # fit recovery on a noisy synthetic profile
    rng = np.random.default_rng(7)
    rs = np.linspace(0.02, 0.95, 30)
    targets = nm.vignette_spline(rs, 0.751, 0.097, 0.5) \
        + rng.normal(0, 0.005, len(rs))
    fitted, resid = nm.fit_vignette(rs, targets)
    corner_true = float(nm.vignette_spline(1.0, 0.751, 0.097, 0.5))
    corner_fit = float(nm.vignette_spline(1.0, **fitted))
    check("fit recovers corner multiplier",
          abs(corner_fit / corner_true - 1.0) < 0.03,
          f"{corner_fit} vs {corner_true}")
    check("fit residual small", resid < 0.02, str(resid))


def test_wheel_mapping():
    """Color-wheel <-> wb mapping (debug-UI shadows/highlights wheels)."""
    print("color-wheel wb mapping:")
    # neutral wb -> wheel center
    _, r0 = nm.wb_to_wheel([1.0, 1.0, 1.0])
    check("neutral wb -> radius 0", approx(r0, 0.0, 1e-9), str(r0))
    check("center -> neutral (low)",
          all(approx(v, 1.0, 1e-9) for v in nm.wheel_to_wb(0.0, 0.0, "low")))
    check("center -> neutral (high)",
          all(approx(v, 1.0, 1e-9) for v in nm.wheel_to_wb(0.0, 0.0, "high")))

    # round-trip: wb -> wheel -> wb is exact for in-gamut normalized vectors
    low_cases = [[1.0, 0.91, 0.78], [1.0, 0.5, 0.6], [0.8, 1.0, 0.7]]
    for wb in low_cases:
        a, r = nm.wb_to_wheel(wb)
        back = nm.wheel_to_wb(a, r, "low")
        check(f"low round-trip {wb}",
              all(approx(x, y, 1e-6) for x, y in zip(wb, back)), str(back))
    high_cases = [[1.34, 1.16, 1.0], [1.8, 1.2, 1.0], [1.0, 1.1, 1.3]]
    for wb in high_cases:
        a, r = nm.wb_to_wheel(wb)
        back = nm.wheel_to_wb(a, r, "high")
        check(f"high round-trip {wb}",
              all(approx(x, y, 1e-6) for x, y in zip(wb, back)), str(back))

    # normalization invariants and range clamping at arbitrary wheel points
    rng = np.random.default_rng(3)
    bad = 0
    for _ in range(200):
        ang = rng.uniform(-math.pi, math.pi)
        rad = rng.uniform(0.0, 1.0)
        lo = nm.wheel_to_wb(ang, rad, "low")
        hi = nm.wheel_to_wb(ang, rad, "high")
        if not approx(max(lo), 1.0, 1e-9):
            bad += 1
        if not approx(min(hi), 1.0, 1e-9):
            bad += 1
        if not all(nm.WB_RANGE[0] <= v <= nm.WB_RANGE[1] for v in lo + hi):
            bad += 1
    check("low max==1 / high min==1 / in-range over wheel", bad == 0, f"{bad} bad")

    # radius is clamped to the disk
    _, rclamp = nm.wb_to_wheel(nm.wheel_to_wb(1.0, 5.0, "high"))
    check("radius clamped to <=1", rclamp <= 1.0 + 1e-9, str(rclamp))


def test_lens_blob():
    print("lens params blob:")
    import auto_negadoctor as an

    vig = {"strength": 0.42, "radius": 0.13, "steepness": 0.57}
    blob = an.encode_lens_params(vig)
    check("gz-prefixed blob", blob.startswith("gz"), blob[:6])
    vs, vr, vst = an.decode_lens_vignette(blob)
    check("v_* round-trip", approx(vs, 0.42, 1e-6) and approx(vr, 0.13, 1e-6)
          and approx(vst, 0.57, 1e-6), f"{vs},{vr},{vst}")
    raw = an._dt_gz_decode(blob)
    import struct as _s
    flags = _s.unpack_from("<i", raw, an.LENS_MODFLAGS_OFFSET)[0]
    check("lensfun vignetting flag cleared",
          flags & an.LENS_MODFLAG_VIGNETTING == 0, hex(flags))
    check("blob length 356", len(raw) == 356, str(len(raw)))

    # the real manually-tuned blob from the user's roll decodes to their
    # hand-chosen manual vignette values
    import re
    xmp_path = (Path(__file__).parent / "fixtures" / "manual_xmps"
                / "DSC_0001.nef.xmp")
    if xmp_path.exists():
        xmp = xmp_path.read_text(encoding="utf-8", errors="replace")
        m = re.search(r'darktable:operation="lens"[^>]*?darktable:params="([^"]+)"',
                      xmp, re.S)
        vs, vr, vst = an.decode_lens_vignette(m.group(1))
        check("real blob v_strength~0.751", approx(vs, 0.751, 1e-3), str(vs))
        check("real blob v_radius~0.097", approx(vr, 0.097, 1e-3), str(vr))
        check("real blob v_steepness~0.5", approx(vst, 0.5, 1e-3), str(vst))


def main():
    test_encode_decode()
    test_exposure_factor()
    test_colorspace()
    test_tuner_roundtrip_neutral()
    test_tuner_color_cast()
    test_vignette()
    test_wheel_mapping()
    test_lens_blob()
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
