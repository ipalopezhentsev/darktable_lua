"""Regression test for the ashift (rotate & perspective) inverse coordinate transform.

The retouch mask must be written in darktable's raw coordinate system, so detect_dust inverts
ashift -> flip -> crop. The subtle part is recovering the ashift INPUT buffer dims: they are NOT
the full output dims (the homography expands the bounding box), and getting them wrong shifts
every mask by ~0.3-0.8% — a visible miss for small dust (this test guards that fix).

Two checks, both anchored to darktable's exact geometry (no approximation):
  1. _solve_ashift_input_dims must return input dims whose homography output bbox equals the
     known full output size (residual ~0px).
  2. export -> raw -> export must round-trip to the same pixel (inverse is self-consistent).

Run:
    conda run -n autocrop python auto_retouch/tests/test_ashift_transform.py
"""

import sys
import math
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import detect_dust as d

# Real params captured from DSC_0024 (0.3 deg straighten + tiny vertical lens shift, portrait).
ASHIFT_B64 = "eJwzqp9pF6vasZsBDgqcGBga7BkYTkBpBgZGIDZh+WxlOq3OftWOP9b8P2vtGUbBKBgFFAEAcucN0w=="
FLIP = 7
CROP = (0.032600, 0.023000, 0.975900, 0.971000)
EXPORT_W, EXPORT_H = 2670, 4032


def _full_output_dims(params, crop, flip, export_w, export_h):
    crop_l, crop_t, crop_r, crop_b = crop
    w_post_flip = export_w / (crop_r - crop_l)
    h_post_flip = export_h / (crop_b - crop_t)
    W_out = h_post_flip if (flip & 4) else w_post_flip
    H_out = w_post_flip if (flip & 4) else h_post_flip
    fullwidth = W_out / (params["cr"] - params["cl"])
    fullheight = H_out / (params["cb"] - params["ct"])
    return fullwidth, fullheight, W_out, H_out


def _raw_to_export_px(rx, ry, params, crop, flip, export_w, export_h):
    """Forward map: raw-sensor [0,1] -> export pixel. Mirrors darktable's forward pipeline
    (ashift homography + internal crop, then flip, then crop), built from the SAME homography
    the inverse uses, so a clean round-trip proves the inverse is algebraically consistent."""
    fullwidth, fullheight, W_out, H_out = _full_output_dims(params, crop, flip, export_w, export_h)
    W_in, H_in = d._solve_ashift_input_dims(params, fullwidth, fullheight)
    H = d._compute_ashift_homography(
        params["rotation"], params["lensshift_v"], params["lensshift_h"], params["shear"],
        params["f_length_kb"], params["orthocorr"], params["aspect"], W_in, H_in)
    # raw norm -> input px -> homography -> full-output px -> undo internal crop -> buf_out norm
    p = H @ np.array([rx * W_in, ry * H_in, 1.0])
    px_full, py_full = p[0] / p[2], p[1] / p[2]
    cx = (px_full - fullwidth * params["cl"]) / W_out
    cy = (py_full - fullheight * params["ct"]) / H_out
    # apply flip (forward order SWAP_XY -> FLIP_X -> FLIP_Y)
    if flip & 4:
        cx, cy = cy, cx
    if flip & 1:
        cy = 1.0 - cy
    if flip & 2:
        cx = 1.0 - cx
    # apply crop
    crop_l, crop_t, crop_r, crop_b = crop
    ex = (cx - crop_l) / (crop_r - crop_l)
    ey = (cy - crop_t) / (crop_b - crop_t)
    return ex * export_w, ey * export_h


def main():
    params = d._decode_ashift_params(ASHIFT_B64)
    assert params is not None, "failed to decode ashift params"
    ok = True

    # --- Check 1: solved input dims produce the exact full-output bbox ---
    fullwidth, fullheight, _, _ = _full_output_dims(params, CROP, FLIP, EXPORT_W, EXPORT_H)
    W_in, H_in = d._solve_ashift_input_dims(params, fullwidth, fullheight)
    bw, bh = d._ashift_output_bbox(params, W_in, H_in)
    res_w, res_h = bw - fullwidth, bh - fullheight
    bbox_ok = abs(res_w) < 0.05 and abs(res_h) < 0.05
    ok = ok and bbox_ok
    print(f"[{'ok ' if bbox_ok else 'FAIL'}] input-dim solve: W_in,H_in={W_in:.1f},{H_in:.1f}; "
          f"bbox residual {res_w:+.4f},{res_h:+.4f}px (target 0)")
    # the naive approximation (input = full output) should be visibly off — proves the fix matters
    naive_bw, naive_bh = d._ashift_output_bbox(params, fullwidth, fullheight)
    print(f"      naive input=full-output would over-expand bbox by "
          f"{naive_bw - fullwidth:+.1f},{naive_bh - fullheight:+.1f}px")

    # --- Check 2: export -> raw -> export round-trips to the same pixel ---
    max_err = 0.0
    for ex0, ey0 in [(395.5, 1125.6), (100, 100), (2500, 3800), (1335, 2016), (50, 4000)]:
        rx, ry = d._export_to_original(ex0 / EXPORT_W, ey0 / EXPORT_H, FLIP, CROP,
                                       params, EXPORT_W, EXPORT_H)
        ex1, ey1 = _raw_to_export_px(rx, ry, params, CROP, FLIP, EXPORT_W, EXPORT_H)
        err = math.hypot(ex1 - ex0, ey1 - ey0)
        max_err = max(max_err, err)
    rt_ok = max_err < 0.1
    ok = ok and rt_ok
    print(f"[{'ok ' if rt_ok else 'FAIL'}] export->raw->export round-trip: max error {max_err:.4f}px")

    print("\nPASS" if ok else "\nFAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
