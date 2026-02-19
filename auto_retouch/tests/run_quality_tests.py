"""
Quality regression test for the dust detection algorithm.

Reruns detection on the committed test images and compares results against
the baseline session. Differences are pre-populated as annotations in a new
temp debug session so you can open it in debug_ui.py for visual inspection:

  - New spots (not in baseline): marked as false_positives (red X in UI)
  - Missing spots (in baseline, gone from new run): marked as missed_dust (cyan crosshair)
  - Matched spots: shown as plain green circles (no annotation)

Usage:
    conda run -n autocrop python auto_retouch/tests/run_quality_tests.py
    conda run -n autocrop python auto_retouch/tests/run_quality_tests.py --image DSC_0025
    conda run -n autocrop python auto_retouch/tests/run_quality_tests.py --open-ui
"""

import os
import sys
import json
import math
import argparse
import subprocess
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Convert numpy scalars to plain Python types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

TESTS_DIR = Path(__file__).parent
AUTO_RETOUCH_DIR = TESTS_DIR.parent
sys.path.insert(0, str(AUTO_RETOUCH_DIR))

import detect_dust

IMAGES_DIR = TESTS_DIR / "images"
BASELINE_DIR = TESTS_DIR / "baseline_session"
BASELINE_JSON = BASELINE_DIR / "debug_spots.json"

# Comparison thresholds
MATCH_RADIUS = 15          # px: baseline/new spot closer than this = same spot
PASS_MATCH_RATE = 0.80     # ≥80% of baseline spots matched → not a regression
WARN_MATCH_RATE = 0.60     # ≥60% → warn (degraded but not catastrophic)
PASS_NEW_RATIO = 0.25      # new spots ≤ 25% of baseline count → acceptable
WARN_NEW_RATIO = 0.50      # new spots ≤ 50% → warn


def _worker(image_path):
    stem = Path(image_path).stem
    img_path = str(image_path)

    import cv2
    img = cv2.imread(img_path)
    if img is None:
        return (stem, None, [], (0, 0), f"Failed to load: {img_path}", None)
    height, width = img.shape[:2]

    spots, rejected, error = detect_dust.detect_dust_spots(img_path, collect_rejects=True)
    return (stem, spots, rejected, (width, height), error, None)


def _dist(a, b):
    return math.sqrt((a["cx"] - b["cx"]) ** 2 + (a["cy"] - b["cy"]) ** 2)


def _match_spots(baseline_spots, new_spots, radius=MATCH_RADIUS):
    """Greedy nearest-neighbour matching within radius.

    Returns (matched_baseline_indices, matched_new_indices, missing_baseline, new_fps).
    """
    used_new = set()
    matched_baseline = set()

    for bi, bs in enumerate(baseline_spots):
        best_dist = float("inf")
        best_ni = None
        for ni, ns in enumerate(new_spots):
            if ni in used_new:
                continue
            d = _dist(bs, ns)
            if d < best_dist:
                best_dist = d
                best_ni = ni
        if best_dist <= radius and best_ni is not None:
            matched_baseline.add(bi)
            used_new.add(best_ni)

    missing = [bs for bi, bs in enumerate(baseline_spots) if bi not in matched_baseline]
    new_fps = [ns for ni, ns in enumerate(new_spots) if ni not in used_new]
    return matched_baseline, used_new, missing, new_fps


def _verdict(baseline_count, new_count, matched, missing, new_fps):
    match_rate = (len(matched) / baseline_count) if baseline_count > 0 else 1.0
    new_ratio = (len(new_fps) / baseline_count) if baseline_count > 0 else 0.0

    if match_rate >= PASS_MATCH_RATE and new_ratio <= PASS_NEW_RATIO:
        return "PASS", match_rate, new_ratio
    elif match_rate >= WARN_MATCH_RATE and new_ratio <= WARN_NEW_RATIO:
        return "WARN", match_rate, new_ratio
    else:
        return "FAIL", match_rate, new_ratio


def load_baseline():
    if not BASELINE_JSON.exists():
        print(f"ERROR: Baseline not found at {BASELINE_JSON}")
        print("Run generate_baseline.py first.")
        sys.exit(1)
    with open(BASELINE_JSON) as f:
        data = json.load(f)
    # Index by stem
    return {img["stem"]: img for img in data["images"]}


def write_session(session_dir, raw_results, image_paths_by_stem, diff_by_stem):
    """Write debug_spots.json + per-image annotations.json to session_dir."""
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    # Write debug_spots.json using the same function as detect_dust.py
    detect_dust.write_debug_spots_json(raw_results, image_paths_by_stem, str(session_dir))

    # Write annotations.json with pre-populated diff markers
    for stem, (missing, new_fps) in diff_by_stem.items():
        ann = {
            "stem": stem,
            # New spots in new run (not in baseline) → show as false positives (red X)
            "false_positives": new_fps,
            # Spots that disappeared from baseline → show as missed dust (cyan crosshair)
            "missed_dust": [{"cx": s["cx"], "cy": s["cy"]} for s in missing],
        }
        ann_path = session_dir / f"{stem}_annotations.json"
        with open(ann_path, "w") as f:
            json.dump(ann, f, indent=2, cls=NumpyEncoder)


def main():
    parser = argparse.ArgumentParser(description="Dust detection quality regression test")
    parser.add_argument("--image", metavar="STEM",
                        help="Test only this image (e.g. DSC_0025)")
    parser.add_argument("--open-ui", action="store_true",
                        help="Launch debug_ui.py on the generated diff session when done")
    args = parser.parse_args()

    baseline = load_baseline()

    image_paths = sorted(IMAGES_DIR.glob("*.jpg")) + sorted(IMAGES_DIR.glob("*.jpeg"))
    if args.image:
        image_paths = [p for p in image_paths if p.stem == args.image]
        if not image_paths:
            print(f"ERROR: Image '{args.image}' not found in {IMAGES_DIR}")
            sys.exit(1)

    if not image_paths:
        print(f"No images found in {IMAGES_DIR}")
        sys.exit(1)

    print(f"Running detection on {len(image_paths)} image(s)...")
    n_workers = min(cpu_count(), len(image_paths))
    with Pool(processes=n_workers) as pool:
        raw_results = pool.map(_worker, image_paths)
    raw_results.sort(key=lambda r: r[0])

    image_paths_by_stem = {Path(p).stem: str(p) for p in image_paths}

    # --- Compare against baseline ---
    verdicts = []
    diff_by_stem = {}

    for stem, spots, rejected, img_dims, error, _xmp in raw_results:
        new_spots = spots or []
        baseline_entry = baseline.get(stem)

        if error:
            verdicts.append(("FAIL", stem, 0, 0, [], [], [], error))
            diff_by_stem[stem] = ([], [])
            continue

        if baseline_entry is None:
            verdicts.append(("WARN", stem, len(new_spots), 0, [], [], new_spots,
                             "not in baseline"))
            diff_by_stem[stem] = ([], new_spots)
            continue

        baseline_spots = baseline_entry.get("detected", [])
        matched_bi, matched_ni, missing, new_fps = _match_spots(baseline_spots, new_spots)
        verdict, match_rate, new_ratio = _verdict(
            len(baseline_spots), len(new_spots), matched_bi, missing, new_fps)
        verdicts.append((verdict, stem, len(new_spots), len(baseline_spots),
                         matched_bi, missing, new_fps, None))
        diff_by_stem[stem] = (missing, new_fps)

    # --- Print summary ---
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for entry in verdicts:
        verdict, stem, new_count, baseline_count, matched, missing, new_fps, note = entry
        counts[verdict] += 1
        match_count = len(matched)
        match_rate_str = (f"  match={match_count}/{baseline_count}"
                          if baseline_count else "")
        miss_str = f"  missing={len(missing)}" if missing else ""
        fp_str = f"  new_fps={len(new_fps)}" if new_fps else ""
        note_str = f"  ({note})" if note else ""
        print(f"[{verdict}] {stem:<12} baseline={baseline_count:>3}  new={new_count:>3}"
              f"{match_rate_str}{miss_str}{fp_str}{note_str}")

    print(f"\nOverall: {counts['PASS']} PASS, {counts['WARN']} WARN, {counts['FAIL']} FAIL")

    # --- Write diff session ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(tempfile.gettempdir()) / f"dt_qualitytest_{timestamp}"
    write_session(session_dir, raw_results, image_paths_by_stem, diff_by_stem)

    debug_ui_script = AUTO_RETOUCH_DIR / "debug_ui.py"
    open_cmd = f"conda run -n autocrop python \"{debug_ui_script}\" \"{session_dir}\""
    print(f"\nDiff session: {session_dir}")
    print(f"Review visually: {open_cmd}")

    if args.open_ui:
        subprocess.Popen([sys.executable, str(debug_ui_script), str(session_dir)])

    sys.exit(0 if counts["FAIL"] == 0 else 1)


if __name__ == "__main__":
    main()
