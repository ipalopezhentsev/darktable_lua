"""Black-and-white film support (fixture-free, end-to-end on tiny synthetic rolls).

A B&W scan can't be told from a colour one by channel count (the DSLR always
exports RGB), so auto_negadoctor detects it by the FILM BASE: a colour negative
always has a strongly-orange base, a B&W one never does. When the orange search
finds nothing, the roll is treated as B&W — the base becomes the frame's
brightest (least-dense) region, its actual tint is Dmin, wb is forced neutral and
film_stock=0, so dividing the negative by the tinted base prints a neutral grey.

This test generates two tiny synthetic rolls (a bluish-tinted grey B&W roll and
an orange colour roll), writes them as float TIFFs, and asserts:
  - "auto" flips a base-less-orange roll to B&W (film_stock 0, neutral wb) and
    keeps the orange roll colour (film_stock 1);
  - the rendered B&W output is near-neutral (the tint is divided out);
  - "off" refuses to fall back (the B&W roll errors), "on" forces B&W;
  - make_params(bw=True) stamps film_stock 0.

Run: conda run -n autocrop python auto_negadoctor/tests/test_bw.py
"""

import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import auto_negadoctor as an
import nega_model as nm

_fail = 0


def check(cond, msg):
    global _fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        _fail += 1


def _write_tiff(path, lin_rgb):
    """Write a linear-RGB float32 frame as a 3-channel float TIFF (no ICC, so
    load_frame treats it as linear Rec2020 — exactly the B&W case here)."""
    bgr = cv2.cvtColor(lin_rgb.astype(np.float32), cv2.COLOR_RGB2BGR)
    assert cv2.imwrite(path, bgr), path


def _make_roll(tmp, tint, n=3, w=240, h=180):
    """A tiny negative roll of the given base `tint`: a dense grey scene ramp with
    a bright base patch (transmission ~1 -> the tint) and a dark holder border.
    Different per-frame brightness so exposure logic has something to chew on."""
    paths = []
    for i in range(n):
        lin = np.empty((h, w, 3), dtype=np.float32)
        # scene: horizontal transmission ramp 0.12..0.7 of the base tint (denser
        # than the base everywhere), so the base patch stays the brightest region.
        ramp = np.linspace(0.12, 0.70, w, dtype=np.float32)[None, :, None]
        lin[:] = tint[None, None, :] * ramp * (0.85 + 0.1 * i)
        lin[40:150, 40:90] = tint * (0.98 + 0.01 * i)   # bright base patch
        lin[:12, :] = 0.001                              # holder border
        lin[-12:, :] = 0.001
        lin[:, :12] = 0.001
        lin[:, -12:] = 0.001
        p = os.path.join(tmp, f"BW_{i:04d}.tif")
        _write_tiff(p, lin)
        paths.append(p)
    return paths


BW_TINT = np.array([0.42, 0.41, 0.53], dtype=np.float32)     # cool bluish grey
ORANGE_TINT = np.array([1.30, 0.76, 0.40], dtype=np.float32)  # colour-neg base


def run():
    print("test_bw:")
    with tempfile.TemporaryDirectory() as tmp:
        bw_paths = _make_roll(tmp, BW_TINT)

        # --- auto: a roll with no orange base becomes B&W ---
        frames, roll = an.process_roll(bw_paths, {}, bw_mode="auto")
        check(roll is not None and roll.get("bw") is True,
              "auto detects B&W on a grey-base roll")
        ok = [f for f in frames if not f["error"]]
        check(len(ok) == len(bw_paths), f"all {len(bw_paths)} B&W frames succeed")
        p0 = ok[0]["params"]
        check(p0["film_stock"] == 0, "B&W params carry film_stock=0")
        check(p0["wb_low"] == [1.0, 1.0, 1.0] and p0["wb_high"] == [1.0, 1.0, 1.0],
              "B&W wb is neutral [1,1,1]")
        # Dmin keeps the base TINT (B channel highest, like the scan) so it can be
        # divided out to neutral.
        check(p0["Dmin"][2] > p0["Dmin"][0] > 0,
              f"B&W Dmin keeps the base tint ({[round(v,3) for v in p0['Dmin']]})")

        # the rendered positive is near-neutral (tint divided out)
        _enc, lin = an.load_frame(ok[0]["path"], roll.get("vignette"))
        img = nm.render_negadoctor_srgb8(lin, p0).reshape(-1, 3).astype(float)
        chan_spread = float(img.mean(axis=0).max() - img.mean(axis=0).min())
        check(chan_spread < 4.0,
              f"B&W render is near-neutral (mean channel spread {chan_spread:.2f}/255)")

        # --- off: never fall back, so the base-less roll errors (old behaviour) ---
        frames_off, roll_off = an.process_roll(bw_paths, {}, bw_mode="off")
        check(all(f["error"] for f in frames_off) and roll_off is None,
              "bw_mode=off refuses the B&W fallback (roll errors)")

        # --- on: force B&W directly ---
        frames_on, roll_on = an.process_roll(bw_paths, {}, bw_mode="on")
        check(roll_on is not None and roll_on.get("bw") is True
              and all(f["params"]["film_stock"] == 0
                      for f in frames_on if not f["error"]),
              "bw_mode=on forces B&W")

        # --- auto keeps a real orange roll in colour ---
        orange_paths = _make_roll(tmp, ORANGE_TINT)
        for i, p in enumerate(orange_paths):          # rename so stems don't clash
            np_ = p.replace("BW_", "CL_")
            os.replace(p, np_)
            orange_paths[i] = np_
        frames_c, roll_c = an.process_roll(orange_paths, {}, bw_mode="auto")
        okc = [f for f in frames_c if not f["error"]]
        check(roll_c is not None and not roll_c.get("bw") and okc
              and all(f["params"]["film_stock"] == 1 for f in okc),
              "auto keeps an orange-base roll in colour (film_stock=1)")

    # --- make_params flag directly ---
    dmin = [0.5, 0.5, 0.6]
    pbw = an.make_params(dmin, 1.7, 0.0, [1, 1, 1], [1, 1, 1], [0.1] * 3, [0.5] * 3,
                         bw=True)
    pcol = an.make_params(dmin, 1.7, 0.0, [1, 1, 1], [1, 1, 1], [0.1] * 3, [0.5] * 3,
                          bw=False)
    check(pbw["film_stock"] == 0 and pcol["film_stock"] == 1,
          "make_params(bw=) sets film_stock 0 / 1")

    if _fail:
        print(f"\n{_fail} CHECK(S) FAILED")
        sys.exit(1)
    print("\nALL B&W CHECKS PASSED")


if __name__ == "__main__":
    run()
