"""Calibration check: pipeline output vs the user's manual negadoctor tuning.

Fixtures:
  tests/images/                 - 1000px uninverted exports of a real roll
  tests/fixtures/manual_xmps/   - the same roll's manually tuned .nef.xmp files

For every stem present in both, decode the manual negadoctor params from the
XMP and compare against what process_roll() computes. This quantifies the
whole-chain residual (sRGB export linearization vs darktable's working
profile, patch choices vs the user's picker clicks).

Skips (exit 0 with a message) when fixtures are absent.

Run: conda run -n autocrop python auto_negadoctor/tests/test_calibration.py
Flags: --verbose  per-frame table always (default: only on WARN/FAIL rows)
"""

import os
import re
import sys
from pathlib import Path

import numpy as np

TESTS_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(TESTS_DIR.parent))
import auto_negadoctor as an
import nega_model as nm

IMAGES_DIR = TESTS_DIR / "images"
XMP_DIR = TESTS_DIR / "fixtures" / "manual_xmps"

# Tolerances (vs hand tuning, so generous):
DMIN_RATIO_TOL = 0.15      # per-channel |ours/manual - 1|
# Aggregate gates (the actual pass criteria). Only Dmin is gated:
# - wb_low/wb_high: the JPEG fixtures' upper R range is destroyed by sRGB
#   gamut clipping, and the manual sessions ran under a different upstream
#   WB state than fresh imports — informational only.
# - D_max: ours is scoped to holder-free photo-content percentiles, the
#   manual value was darktable's whole-area auto pick, and the print
#   auto-tune compensates exposure around whatever D_max says, so absolute
#   agreement stopped being meaningful — informational only.
PASS_MEDIAN_DMIN_RATIO_ERR = 0.15
PASS_FRAME_FAIL_FRAC = 0.35


def extract_manual_lens_vignette():
    """(v_strength, v_radius, v_steepness) from the user's manual lens
    entries, or None."""
    for xmp in sorted(XMP_DIR.glob("*.xmp")):
        text = xmp.read_text(encoding="utf-8", errors="replace")
        m = re.search(r'darktable:operation="lens"[^>]*?darktable:params="([^"]+)"',
                      text, re.S)
        if m:
            try:
                return an.decode_lens_vignette(m.group(1))
            except Exception:
                continue
    return None


def extract_manual_params(xmp_path):
    """Last enabled negadoctor history entry's decoded params, or None."""
    text = Path(xmp_path).read_text(encoding="utf-8", errors="replace")
    best = None
    for m in re.finditer(r"<rdf:li[^>]*?darktable:operation=\"negadoctor\"[^>]*?/>",
                         text, re.S):
        entry = m.group(0)
        if 'darktable:enabled="1"' not in entry:
            continue
        pm = re.search(r'darktable:params="([0-9a-fA-F]+)"', entry)
        nu = re.search(r'darktable:num="(\d+)"', entry)
        if pm and len(pm.group(1)) == 152:
            num = int(nu.group(1)) if nu else 0
            if best is None or num > best[0]:
                best = (num, pm.group(1))
    return nm.decode_negadoctor_params(best[1]) if best else None


def main():
    verbose = "--verbose" in sys.argv

    if not IMAGES_DIR.is_dir() or not XMP_DIR.is_dir():
        print("SKIP: calibration fixtures not present")
        return 0
    images = sorted(IMAGES_DIR.glob("*.jpg"))
    manual = {}
    for xmp in sorted(XMP_DIR.glob("*.xmp")):
        stem = xmp.name.split(".")[0]
        p = extract_manual_params(xmp)
        if p:
            manual[stem] = p
    pairs = [(img, manual[img.stem]) for img in images if img.stem in manual]
    if not pairs:
        print("SKIP: no stems present in both images/ and manual_xmps/")
        return 0
    print(f"Calibrating on {len(pairs)} frame(s) "
          f"({len(images)} images, {len(manual)} manual inversions)")

    frames, roll = an.process_roll([str(img) for img, _ in pairs], {})
    ours = {fr["stem"]: fr for fr in frames}
    if roll:
        print(f"Global base winner: {roll['winner_stem']} "
              f"rgb={['%.4f' % v for v in roll['winner_rgb']]}")
        # vignette: informational comparison vs the user's hand-chosen manual
        # values (theirs sit ON TOP of lensfun's share; ours is the total)
        v = roll.get("vignette")
        manual_lens = extract_manual_lens_vignette()
        if v:
            print(f"Vignette estimate: strength={v['strength']:.3f} "
                  f"radius={v['radius']:.3f} steepness={v['steepness']:.3f}")
        else:
            print("Vignette estimate: none")
        if manual_lens:
            print(f"User manual vignette (extra over lensfun): "
                  f"strength={manual_lens[0]:.3f} radius={manual_lens[1]:.3f} "
                  f"steepness={manual_lens[2]:.3f}")

    rows = []
    dmin_errs, dmax_diffs, wbl_diffs, wbh_diffs = [], [], [], []
    fail_frames = 0
    for img, mp in pairs:
        fr = ours.get(img.stem)
        if fr is None or fr.get("error") or "params" not in fr:
            rows.append((img.stem, "ERR", fr.get("error") if fr else "missing"))
            fail_frames += 1
            continue
        op = fr["params"]
        ratio_err = [abs(op["Dmin"][c] / max(mp["Dmin"][c], 1e-6) - 1.0)
                     for c in range(3)]
        dmax_d = abs(op["D_max"] - mp["D_max"])
        wbl_d = max(abs(op["wb_low"][c] - mp["wb_low"][c]) for c in range(3))
        wbh_d = max(abs(op["wb_high"][c] - mp["wb_high"][c]) for c in range(3))
        dmin_errs.append(max(ratio_err))
        dmax_diffs.append(dmax_d)
        wbl_diffs.append(wbl_d)
        wbh_diffs.append(wbh_d)

        bad = max(ratio_err) > DMIN_RATIO_TOL
        if bad:
            fail_frames += 1
        rows.append((
            img.stem, "WARN" if bad else "ok",
            f"Dmin {fmtv(op['Dmin'])} vs {fmtv(mp['Dmin'])} (err {max(ratio_err):.2f}) | "
            f"Dmax {op['D_max']:.2f} vs {mp['D_max']:.2f} | "
            f"wbl d={wbl_d:.2f} wbh d={wbh_d:.2f}"
        ))

    print()
    for stem, status, detail in rows:
        if verbose or status != "ok":
            print(f"  [{status:4}] {stem}: {detail}")

    if not dmin_errs:
        print("FAIL: no comparable frames")
        return 1

    med_dmin = float(np.median(dmin_errs))
    print()
    print(f"Aggregates over {len(dmin_errs)} frame(s):")
    print(f"  Dmin ratio err   median {med_dmin:.3f}  p90 {np.percentile(dmin_errs, 90):.3f}")
    print(f"  D_max diff       median {np.median(dmax_diffs):.3f}  p90 {np.percentile(dmax_diffs, 90):.3f}")
    print(f"  wb_low max diff  median {np.median(wbl_diffs):.3f}  p90 {np.percentile(wbl_diffs, 90):.3f}")
    print(f"  wb_high max diff median {np.median(wbh_diffs):.3f}  p90 {np.percentile(wbh_diffs, 90):.3f}")

    fail_frac = fail_frames / len(pairs)
    verdict_fail = (med_dmin > PASS_MEDIAN_DMIN_RATIO_ERR
                    or fail_frac > PASS_FRAME_FAIL_FRAC)
    print()
    print(f"Frames out of tolerance (Dmin): {fail_frames}/{len(pairs)} ({fail_frac:.0%})")
    print("(wb and D_max diffs are informational - see gate comments)")
    print("VERDICT:", "FAIL" if verdict_fail else "PASS")
    return 1 if verdict_fail else 0


def fmtv(v):
    return "(" + ",".join(f"{x:.3f}" for x in v) + ")"


if __name__ == "__main__":
    sys.exit(main())
