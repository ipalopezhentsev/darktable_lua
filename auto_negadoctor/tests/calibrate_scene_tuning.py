"""Offline calibration helper for the vision-LLM scene tuner (spec 03).

NOT a CI gate (needs local TIFFs + a running Ollama; the LLM is slow). It:

  1. runs the analytical pipeline on tests/images_tif (the GT roll),
  2. categorizes each frame with the vision LLM (gemma3 via Ollama), caching
     responses in tests/fixtures/scene_cache.json so re-runs are instant and
     reproducible (commit that file as a fixture),
  3. groups the user's GROUND-TRUTH params (run_quality_tests._load_ground_truth)
     by the LLM's labels and prints per-category medians — the numbers to set
     scene_tuner.CONTRAST_GAMMA / WARMTH_SHIFT / MOOD_CEIL_BIAS from,
  4. reports the AI-variant vs analytical GT-delta so you can see whether the
     current lookup table actually moves params toward the user's picks.

Run: conda run -n autocrop python auto_negadoctor/tests/calibrate_scene_tuning.py
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
import scene_tuner as st
import run_quality_tests as rqt

CACHE_PATH = TESTS_DIR / "fixtures" / "scene_cache.json"


def _median(vals):
    return statistics.median(vals) if vals else None


def main():
    images, exif = rqt.list_test_images()
    if not images or not str(images[0]).lower().endswith((".tif", ".tiff")):
        print("SKIP: needs the local linear-Rec2020 TIFF roll in "
              "tests/images_tif (see images_tif/README.md).")
        return 0

    print(f"Analytical pipeline over {len(images)} frame(s)...")
    frames, roll = an.process_roll(images, exif)
    by_stem = {fr["stem"]: fr for fr in frames if not fr.get("error")}

    print(f"Categorizing with {st.OLLAMA_MODEL} (cache: {CACHE_PATH.name})...")
    scenes = {}
    for stem, fr in by_stem.items():
        enc_f, lin = an.load_frame(fr["path"], fr.get("vignette"))
        preview = an.render_preview_srgb(lin, fr["params"])
        sc = st.categorize_scene(preview, cache_path=str(CACHE_PATH),
                                 cache_key=stem)
        scenes[stem] = sc
        fr["_lin"] = lin
        if sc:
            print(f"  {stem}: {sc['scene']}/{sc['mood']}/{sc['warmth']}/"
                  f"{sc['contrast']}")
        else:
            print(f"  {stem}: (no scene)")

    gt_by_stem = rqt._load_ground_truth()

    # --- GT params grouped by LLM label -> table calibration numbers ---
    # gamma is keyed on SCENE (the contrast label was degenerate on the GT roll
    # -- moondream tagged ~all frames "soft"); see scene_tuner.SCENE_GAMMA.
    by_scene_g, by_warmth_rb, by_mood_exp = {}, {}, {}
    for stem, gt in gt_by_stem.items():
        sc = scenes.get(stem)
        if not sc:
            continue
        if "gamma" in gt:
            by_scene_g.setdefault(sc["scene"], []).append(gt["gamma"])
        if gt.get("wb_low"):
            rb = gt["wb_low"][0] / max(gt["wb_low"][2], 1e-6)
            by_warmth_rb.setdefault(sc["warmth"], []).append(rb)
        if "exposure" in gt:
            by_mood_exp.setdefault(sc["mood"], []).append(gt["exposure"])

    print("\n=== GT gamma by SCENE label (set SCENE_GAMMA offsets) ===")
    base_g = an.PRINT_GAMMA
    for s in sorted(by_scene_g, key=lambda k: -len(by_scene_g[k])):
        m = _median(by_scene_g.get(s, []))
        if m is not None:
            print(f"  {s:16s} median gamma {m:.2f}  (offset vs PRINT_GAMMA "
                  f"{base_g:.2f}: {m - base_g:+.2f})  n={len(by_scene_g[s])}")
    print("\n=== GT wb_low R/B by warmth label (warmer -> larger R/B) ===")
    for wlab in st.WARMTHS:
        m = _median(by_warmth_rb.get(wlab, []))
        if m is not None:
            print(f"  {wlab:7s} median R/B {m:.2f}  n={len(by_warmth_rb[wlab])}")
    print("\n=== GT exposure by mood label (darker mood -> lower exposure) ===")
    for mlab in st.MOODS:
        m = _median(by_mood_exp.get(mlab, []))
        if m is not None:
            print(f"  {mlab:7s} median exposure {m:.2f}  n={len(by_mood_exp[mlab])}")

    # --- AI vs analytical GT-delta ---
    print("\n=== AI variant vs analytical: GT abs-delta (lower is better) ===")
    fields = ["wb_low", "wb_high", "black", "exposure", "gamma"]
    acc = {f: {"ana": [], "ai": []} for f in fields}
    for stem, gt in gt_by_stem.items():
        fr = by_stem.get(stem)
        sc = scenes.get(stem)
        if not fr or "params" not in fr:
            continue
        ai_params, _ = st.apply_scene_tuning(fr["params"], sc, fr["_lin"],
                                             fr["border"])
        for f in fields:
            tgt = gt.get(f)
            if tgt is None:
                continue
            if isinstance(tgt, list):
                da = sum(abs(fr["params"][f][c] - tgt[c]) for c in range(3)) / 3
                di = sum(abs(ai_params[f][c] - tgt[c]) for c in range(3)) / 3
            else:
                da = abs(fr["params"][f] - tgt)
                di = abs(ai_params[f] - tgt)
            acc[f]["ana"].append(da)
            acc[f]["ai"].append(di)
    for f in fields:
        a, i = acc[f]["ana"], acc[f]["ai"]
        if not a:
            continue
        ma, mi = _median(a), _median(i)
        arrow = "better" if mi < ma - 1e-9 else ("worse" if mi > ma + 1e-9 else "same")
        print(f"  {f:9s} analytical {ma:.4f} -> AI {mi:.4f}  ({arrow}, n={len(a)})")

    print("\nDone. Tune scene_tuner constants from the per-label medians:")
    print("  * SCENE_GAMMA  from the per-scene gamma offsets above")
    print("  * AI_HI_CEIL / AI_BLACK_LIFT — the systematic exposure/black gap is")
    print("    NOT per-scene; sweep them (see the calib note in scene_tuner) to")
    print("    minimize the exposure/black deltas while staying clip-safe.")
    print("  Then re-run to confirm the AI deltas improve. The committed gate")
    print("  (run_quality_tests.check_ai_variant) uses fixtures/scene_labels.json")
    print("  so it reproduces offline; re-capture that from a fresh --ai-tune run")
    print("  (scene_cache.json) when the prompt/model/vocabulary changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
