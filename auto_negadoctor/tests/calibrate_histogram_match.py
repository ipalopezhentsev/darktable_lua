"""Offline histogram-match instrument for the analytical print/wb tuning (spec
04_tune_algo_params_via_histograms.md).

NOT a CI gate (needs each roll's local TIFFs). The user's ground truth is the
inverted PICTURE, not the params that made it: exposure<->wb are interdependent,
so the hand-tuned GT params are not a unique encoding. The param-invariant
signal is the rendered HISTOGRAM. This script measures, per GT-annotated frame,
the EMD between the algorithm's render and the user's GT render over the content
crop, decomposed into a luma (brightness/clip) term and a color (shadow/
highlight cast) term -- so we can see whether the remaining gap is exposure or
white balance, and pick the fixed constants (PRINT_HI_CEIL / PRINT_GAMMA /
WB_LOW_DESAT) that minimize it.

It also sweeps the print tuner's brightness ceiling (PRINT_HI_CEIL) over a range
and reports the aggregate luma EMD per candidate, so the best fixed ceiling
reads straight off the table.

GT-render reconstruction: copy the production params and override ONLY the
fields the user annotated (wb_low/wb_high/black/exposure/gamma). Annotations are
PARTIAL (some frames have wb only), so the GT picture of a wb-only frame is the
production tone with the user's wb -- never production replaced by defaults.
Dmin/D_max/offset/soft_clip stay = production (they are not annotated).

Run: conda run -n autocrop python auto_negadoctor/tests/calibrate_histogram_match.py
"""

import os
import statistics
import sys
from pathlib import Path

import numpy as np

TESTS_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(TESTS_DIR.parent))
import json

import auto_negadoctor as an
import nega_model as nm
import run_quality_tests as rqt

# Render + GT-reconstruction helpers live in run_quality_tests (single source of
# truth, shared with the check_histogram_match gate so the metric matches).
gt_params_for_frame = rqt.gt_params_for_frame
_render_crop_rows = rqt._render_crop_rows

# Brightness-ceiling candidates for the PRINT_HI_CEIL sweep (production = 0.99).
HI_CEIL_SWEEP = [0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.72]
HIST_BINS = rqt.HIST_BINS


def _median(vals):
    return statistics.median(vals) if vals else float("nan")


def _annotated(gt):
    """Short tag of which field groups this frame's GT carries."""
    tags = []
    if gt.get("wb_low") or gt.get("wb_high"):
        tags.append("wb")
    if any(gt.get(k) is not None for k in ("black", "exposure", "gamma")):
        tags.append("print")
    return "+".join(tags) or "?"


def run_roll(roll_info, write_baseline=False):
    images, exif, fixtures = (roll_info["images"], roll_info["exif"],
                              roll_info["fixtures"])
    print(f"\n=== Roll {roll_info['id']}: {len(images)} frame(s) ===")
    frames, roll = an.process_roll(images, exif)
    by_stem = {fr["stem"]: fr for fr in frames if not fr.get("error")}
    gt_by_stem = rqt._load_ground_truth(fixtures)
    if not gt_by_stem:
        print("  (no wb/print ground-truth annotations)")
        return

    if write_baseline:
        # Fast path: just the canonical gate medians (no per-frame dashboard /
        # ceiling sweep, which are expensive at the high-res metric).
        med, mn = rqt._histogram_match_medians(frames, fixtures)
        if not med:
            print("  (no renderable GT frames)")
            return
        out = dict(med, n=mn, bins=rqt.HIST_BINS,
                   subsample_frac=rqt.HIST_SUBSAMPLE_FRAC, space="srgb_float",
                   print_gamma=an.PRINT_GAMMA)
        path = roll_info["dir"] / "histogram_baseline.json"
        path.write_text(json.dumps(out, indent=2))
        print(f"  WROTE baseline {path} (total {out['total']:.4f} "
              f"hi999 {out['hi999']:.4f}, n={mn})")
        return

    totals, lumas, colors, signed, tops_a, tops_b = [], [], [], [], [], []
    brighter_needed = 0   # frames where GT is brighter (algorithm too dark)
    sweep = {c: {"luma": [], "signed": [], "clip": []} for c in HI_CEIL_SWEEP}

    print(f"  {'stem':10s} {'ann':8s} {'total':>7s} {'luma':>7s} "
          f"{'signed':>8s} {'color':>7s}  clip%(algo/gt)")
    for stem, gt in sorted(gt_by_stem.items()):
        fr = by_stem.get(stem)
        if not fr or fr.get("error") or "params" not in fr or not fr.get("border"):
            continue
        try:
            enc_f, lin = an.load_frame(fr["path"], fr.get("vignette"))
        except Exception as e:
            print(f"  {stem}: load failed ({e})")
            continue

        gt_u8 = _render_crop_rows(lin, fr["border"], gt_params_for_frame(fr, gt))
        prod_u8 = _render_crop_rows(lin, fr["border"], fr["params"])
        if gt_u8 is None or prod_u8 is None:
            continue
        d = nm.histogram_distance(prod_u8, gt_u8, bins=HIST_BINS)
        totals.append(d["total"]); lumas.append(d["luma"])
        colors.append(d["color"]); signed.append(d["luma_signed"])
        tops_a.append(d["top_a"]); tops_b.append(d["top_b"])
        if d["luma_signed"] > 0:
            brighter_needed += 1
        print(f"  {stem:10s} {_annotated(gt):8s} {d['total']:7.4f} "
              f"{d['luma']:7.4f} {d['luma_signed']:+8.4f} {d['color']:7.4f}"
              f"  {d['top_a']:5.1%}/{d['top_b']:5.1%}")

        # PRINT_HI_CEIL sweep (GT render fixed; re-tune the print params per ceil)
        for ceil in HI_CEIL_SWEEP:
            tp, _ = an.tune_print_params(lin, fr["params"], fr["border"],
                                         fr["dmin"], hi_ceil=ceil)
            su8 = _render_crop_rows(lin, fr["border"], tp)
            if su8 is None:
                continue
            ds = nm.histogram_distance(su8, gt_u8, bins=HIST_BINS)
            sweep[ceil]["luma"].append(ds["luma"])
            sweep[ceil]["signed"].append(ds["luma_signed"])
            sweep[ceil]["clip"].append(ds["top_a"])

    n = len(totals)
    if not n:
        print("  (no renderable GT frames)")
        return
    print(f"\n  aggregate over {n} frame(s) (median):")
    print(f"    total {_median(totals):.4f}   luma {_median(lumas):.4f}   "
          f"color {_median(colors):.4f}   signed luma {_median(signed):+.4f}")
    print(f"    algo clip {_median(tops_a):.2%}  vs  gt clip {_median(tops_b):.2%}")
    print(f"    GT brighter than algo on {brighter_needed}/{n} frame(s) "
          f"(+signed => algorithm renders too DARK)")
    ml, mc = _median(lumas), _median(colors)
    verdict = ("brightness (luma) dominates -> tune PRINT_HI_CEIL / black"
               if ml > mc else
               "color (chroma cast) dominates -> tune WB_LOW_DESAT / wb finder"
               if mc > ml else "brightness and color comparable")
    print(f"    gap verdict: {verdict}")

    print("\n  PRINT_HI_CEIL sweep (median luma EMD to GT; lower is closer):")
    print(f"    {'ceil':>5s} {'luma':>7s} {'signed':>8s} {'algo-clip':>9s}")
    best = min(HI_CEIL_SWEEP, key=lambda c: _median(sweep[c]["luma"]))
    for c in HI_CEIL_SWEEP:
        s = sweep[c]
        mark = "  <- min" if c == best else ""
        print(f"    {c:5.2f} {_median(s['luma']):7.4f} "
              f"{_median(s['signed']):+8.4f} {_median(s['clip']):9.2%}{mark}")
    print(f"    best ceiling by luma EMD: {best:.2f} "
          f"(production PRINT_HI_CEIL = {an.PRINT_HI_CEIL})")


def main():
    write_baseline = "--write-baseline" in sys.argv[1:]
    rolls = rqt.discover_rolls()
    if not rolls:
        print("SKIP: no annotated rolls with local TIFFs under",
              rqt.ROLLS_DIR)
        print("  Repopulate a roll's *.tif exports (see fixtures/rolls/"
              "README.md) and re-run.")
        return 0
    for roll_info in rolls:
        run_roll(roll_info, write_baseline=write_baseline)
    if write_baseline:
        print("\nWrote histogram_baseline.json for each roll (commit them).")
        return 0
    print("\nDone. Phase 1 instrument only: this does NOT change any constant.")
    print("Use the per-frame table + the PRINT_HI_CEIL sweep to drive the")
    print("Phase 2 retune (PRINT_HI_CEIL, black takeover, PRINT_GAMMA,")
    print("WB_LOW_DESAT), re-running run_quality_tests.check_no_clipping after")
    print("each change.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
