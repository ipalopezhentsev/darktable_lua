"""Offline instrument: judge the spec-03 AI variant by the PICTURE, not the params.

Context (the question this answers): the scene_tuner AI variant was both
CALIBRATED and GATED against PARAMETER deltas to the GT (calibrate_scene_tuning.py
+ check_ai_variant). Spec 04 then established that the param proxy is the wrong
ruler -- the ground truth is the rendered PICTURE (its histogram), not the
non-unique params that made it -- and the histogram reframe drove the analytical
path's real systematic fix (PRINT_GAMMA 6.1->5.0, PRINT_HI_CEIL 0.99->0.97). So
the AI was never given a fair trial by the metric we actually trust.

This re-judges it: per GT-annotated frame it renders the ANALYTICAL params, the
AI-variant params (scene_tuner over the committed labels), and the GT params,
then measures the histogram EMD of analytical-vs-GT and AI-vs-GT over the content
crop. If the AI's median EMD is >= analytical's, the AI does not improve the
picture even by the right ruler.

Fully OFFLINE: uses each roll's committed scene_labels.json (no Ollama) exactly
like check_ai_variant, and the shared render/EMD helpers from run_quality_tests
so the metric matches the spec-04 gate. NOT a CI gate (needs each roll's local
TIFFs).

Run: conda run -n autocrop python auto_negadoctor/tests/calibrate_ai_histogram.py
"""

import os
import statistics
import sys
from pathlib import Path

import numpy as np

TESTS_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(TESTS_DIR.parent))

import auto_negadoctor as an
import nega_model as nm
import run_quality_tests as rqt

gt_params_for_frame = rqt.gt_params_for_frame
_render_crop_rows = rqt._render_crop_rows
HIST_BINS = rqt.HIST_BINS


def _median(vals):
    return statistics.median(vals) if vals else float("nan")


def run_roll(roll_info):
    images, exif, fixtures = (roll_info["images"], roll_info["exif"],
                              roll_info["fixtures"])
    print(f"\n=== Roll {roll_info['id']}: {len(images)} frame(s) ===")

    labels = rqt._load_scene_labels(roll_info)
    gt_by_stem = rqt._load_ground_truth(fixtures)
    if not gt_by_stem:
        print("  (no wb/print ground-truth annotations)")
        return None
    if not labels:
        print("  (no committed scene_labels.json -- AI variant needs labels; "
              "skipped)")
        return None
    try:
        import scene_tuner as st
    except Exception as e:
        print(f"  (scene_tuner unavailable: {e})")
        return None

    frames, roll = an.process_roll(images, exif)
    by_stem = {fr["stem"]: fr for fr in frames if not fr.get("error")}

    # per-frame EMD-to-GT for analytical and AI
    ana_tot, ai_tot = [], []
    ana_lum, ai_lum = [], []
    ana_col, ai_col = [], []
    better = worse = same = 0

    print(f"  {'stem':10s} {'scene':10s} "
          f"{'ana_tot':>8s} {'ai_tot':>8s} {'delta':>8s}  verdict")
    for stem, gt in sorted(gt_by_stem.items()):
        fr = by_stem.get(stem)
        if not fr or fr.get("error") or "params" not in fr or not fr.get("border"):
            continue
        try:
            enc_f, lin = an.load_frame(fr["path"], fr.get("vignette"))
        except Exception as e:
            print(f"  {stem}: load failed ({e})")
            continue

        scene = st._validate(labels[stem]) if labels.get(stem) else None
        ai_params, _ = st.apply_scene_tuning(fr["params"], scene, lin,
                                             fr["border"])

        gt_rows = _render_crop_rows(lin, fr["border"], gt_params_for_frame(fr, gt))
        ana_rows = _render_crop_rows(lin, fr["border"], fr["params"])
        ai_rows = _render_crop_rows(lin, fr["border"], ai_params)
        if gt_rows is None or ana_rows is None or ai_rows is None:
            continue

        da = nm.histogram_distance(ana_rows, gt_rows, bins=HIST_BINS)
        di = nm.histogram_distance(ai_rows, gt_rows, bins=HIST_BINS)
        ana_tot.append(da["total"]); ai_tot.append(di["total"])
        ana_lum.append(da["luma"]); ai_lum.append(di["luma"])
        ana_col.append(da["color"]); ai_col.append(di["color"])

        delta = di["total"] - da["total"]   # >0 => AI worse
        if delta < -1e-9:
            verdict, better = "AI better", better + 1
        elif delta > 1e-9:
            verdict, worse = "AI WORSE", worse + 1
        else:
            verdict, same = "same", same + 1
        scene_lbl = (scene.get("scene") if isinstance(scene, dict) else "-") or "-"
        print(f"  {stem:10s} {scene_lbl:10s} "
              f"{da['total']:8.4f} {di['total']:8.4f} {delta:+8.4f}  {verdict}")

    n = len(ana_tot)
    if not n:
        print("  (no renderable GT frames)")
        return None

    res = {
        "roll": roll_info["id"], "n": n,
        "ana_total": _median(ana_tot), "ai_total": _median(ai_tot),
        "ana_luma": _median(ana_lum), "ai_luma": _median(ai_lum),
        "ana_color": _median(ana_col), "ai_color": _median(ai_col),
        "better": better, "worse": worse, "same": same,
    }
    print(f"\n  aggregate over {n} frame(s) (median EMD to GT; lower is closer):")
    print(f"    {'term':6s} {'analytical':>11s} {'AI':>9s} {'verdict':>9s}")
    for term in ("total", "luma", "color"):
        a, i = res[f"ana_{term}"], res[f"ai_{term}"]
        tag = ("AI better" if i < a - 1e-9
               else "AI WORSE" if i > a + 1e-9 else "same")
        print(f"    {term:6s} {a:11.4f} {i:9.4f} {tag:>9s}")
    print(f"    per-frame: AI better {better} | AI worse {worse} | same {same} "
          f"(of {n})")
    return res


def main():
    rolls = rqt.discover_rolls()
    if not rolls:
        print("SKIP: no annotated rolls with local TIFFs under", rqt.ROLLS_DIR)
        print("  Repopulate a roll's *.tif exports (see fixtures/rolls/"
              "README.md) and re-run.")
        return 0
    results = [r for r in (run_roll(ri) for ri in rolls) if r]
    if not results:
        return 0

    print("\n" + "=" * 60)
    print("SUMMARY (median total EMD to GT picture, per roll):")
    print(f"  {'roll':14s} {'n':>3s} {'analytical':>11s} {'AI':>9s}  "
          f"{'AI better/worse/same':>20s}")
    tot_better = tot_worse = tot_same = 0
    ana_all, ai_all = [], []
    for r in results:
        print(f"  {r['roll']:14s} {r['n']:3d} {r['ana_total']:11.4f} "
              f"{r['ai_total']:9.4f}  "
              f"{r['better']:>6d}/{r['worse']}/{r['same']}")
        tot_better += r["better"]; tot_worse += r["worse"]; tot_same += r["same"]
        ana_all.append(r["ana_total"]); ai_all.append(r["ai_total"])
    print(f"\n  across rolls: AI better {tot_better} | AI worse {tot_worse} | "
          f"same {tot_same} frame(s)")
    print(f"  mean-of-roll-medians: analytical {_median(ana_all):.4f}  "
          f"AI {_median(ai_all):.4f}")
    print("\nReading: if AI total EMD >= analytical (and per-frame 'worse' "
          "dominates),\nthe AI variant does not improve the PICTURE even by the "
          "histogram ruler.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
