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

    # Use an annotation session as baseline (filters out user-confirmed FPs):
    conda run -n autocrop python auto_retouch/tests/run_quality_tests.py SESSION_DIR
    conda run -n autocrop python auto_retouch/tests/run_quality_tests.py SESSION_DIR --open-ui
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

TESTS_DIR = Path(__file__).parent
AUTO_RETOUCH_DIR = TESTS_DIR.parent
sys.path.insert(0, str(AUTO_RETOUCH_DIR))

import detect_dust

IMAGES_DIR = TESTS_DIR / "images"
BASELINE_DIR = TESTS_DIR / "baseline_session"

# Comparison thresholds
MATCH_RADIUS = 15          # px: baseline/new spot closer than this = same spot
PASS_MATCH_RATE = 0.80     # ≥80% of baseline spots matched → not a regression
WARN_MATCH_RATE = 0.60     # ≥60% → warn (degraded but not catastrophic)
PASS_NEW_RATIO = 0.25      # new spots ≤ 25% of baseline count → acceptable
WARN_NEW_RATIO = 0.50      # new spots ≤ 50% → warn
MAX_SOURCE_DIST = 200      # px: warn if source is farther than this from spot
SOURCE_MISMATCH_RADIUS = 20  # px: baseline/new source differ by more than this → mismatch
RADIUS_MISMATCH_THRESHOLD = 3.0  # px: brush_radius_px difference to flag as mismatch


# Module-level cache so the ML model is loaded once per worker process,
# not once per image (the Pool initializer sets these before map() starts).
_worker_ml_model = None
_worker_ml_scaler = None


def _init_worker(ml_model_path):
    """Pool initializer: load ML model once per worker process."""
    global _worker_ml_model, _worker_ml_scaler
    if ml_model_path:
        import pickle
        with open(ml_model_path, "rb") as f:
            bundle = pickle.load(f)
        _worker_ml_model = bundle["model"]
        _worker_ml_scaler = bundle["scaler"]


def _worker(image_path):
    stem = Path(image_path).stem
    img_path = str(image_path)

    import cv2
    img = cv2.imread(img_path)
    if img is None:
        return (stem, None, [], (0, 0), f"Failed to load: {img_path}", None)
    height, width = img.shape[:2]

    if _worker_ml_model is not None:
        spots, rejected, error, local_std = detect_dust.detect_dust_spots_ml(
            img_path, _worker_ml_model, _worker_ml_scaler, collect_rejects=True)
    else:
        spots, rejected, error, local_std = detect_dust.detect_dust_spots(
            img_path, collect_rejects=True)
    return (stem, spots, rejected, (width, height), error, None)


def _dist(a, b):
    return math.sqrt((a["cx"] - b["cx"]) ** 2 + (a["cy"] - b["cy"]) ** 2)


def _match_spots(baseline_spots, new_spots, radius=MATCH_RADIUS):
    """Greedy nearest-neighbour matching within radius.

    Returns (matched_pairs, missing_baseline, new_fps, source_issues, source_mismatches, radius_mismatches).
    matched_pairs: list of (baseline_idx, new_idx, baseline_spot, new_spot)
    source_issues: list of (new_idx, issue_description)
    source_mismatches: list of (new_idx, baseline_src, new_src, distance)
    radius_mismatches: list of (new_idx, baseline_brush_radius_px, new_brush_radius_px, diff)
    """
    used_new = set()
    matched_pairs = []

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
            matched_pairs.append((bi, best_ni, bs, new_spots[best_ni]))
            used_new.add(best_ni)

    missing = [bs for bi, bs in enumerate(baseline_spots)
               if bi not in {p[0] for p in matched_pairs}]
    new_fps = [ns for ni, ns in enumerate(new_spots) if ni not in used_new]

    # Validate source positions in new spots and compare with baseline
    source_issues = []
    source_mismatches = []

    for ni, ns in enumerate(new_spots):
        if "src_cx" not in ns or "src_cy" not in ns:
            source_issues.append((ni, "missing source position"))
        else:
            # Check if source is unreasonably far from spot
            src_dist = math.sqrt((ns["src_cx"] - ns["cx"])**2 + (ns["src_cy"] - ns["cy"])**2)
            if src_dist > MAX_SOURCE_DIST:
                source_issues.append((ni, f"source too far: {src_dist:.1f}px"))

    # Compare source positions for matched spots
    for bi, ni, bs, ns in matched_pairs:
        bs_has_src = "src_cx" in bs and "src_cy" in bs
        ns_has_src = "src_cx" in ns and "src_cy" in ns

        if bs_has_src and ns_has_src:
            src_diff = math.sqrt((bs["src_cx"] - ns["src_cx"])**2 + (bs["src_cy"] - ns["src_cy"])**2)
            if src_diff > SOURCE_MISMATCH_RADIUS:
                source_mismatches.append((
                    ni,
                    (bs["src_cx"], bs["src_cy"]),
                    (ns["src_cx"], ns["src_cy"]),
                    src_diff
                ))

    # Compare brush_radius_px for matched spots
    radius_mismatches = []
    for bi, ni, bs, ns in matched_pairs:
        if "brush_radius_px" in bs and "brush_radius_px" in ns:
            radius_diff = abs(bs["brush_radius_px"] - ns["brush_radius_px"])
            if radius_diff > RADIUS_MISMATCH_THRESHOLD:
                radius_mismatches.append((
                    ni,
                    bs["brush_radius_px"],
                    ns["brush_radius_px"],
                    radius_diff
                ))

    return matched_pairs, missing, new_fps, source_issues, source_mismatches, radius_mismatches


def _verdict(baseline_count, new_count, matched, missing, new_fps, source_issues, source_mismatches, radius_mismatches):
    match_rate = (len(matched) / baseline_count) if baseline_count > 0 else 1.0
    new_ratio = (len(new_fps) / baseline_count) if baseline_count > 0 else 0.0

    # Source issues/mismatches and radius mismatches are a warning, not a failure
    has_source_problems = len(source_issues) > 0 or len(source_mismatches) > 0 or len(radius_mismatches) > 0

    if match_rate >= PASS_MATCH_RATE and new_ratio <= PASS_NEW_RATIO:
        return ("WARN" if has_source_problems else "PASS"), match_rate, new_ratio
    elif match_rate >= WARN_MATCH_RATE and new_ratio <= WARN_NEW_RATIO:
        return "WARN", match_rate, new_ratio
    else:
        return "FAIL", match_rate, new_ratio


def load_baseline():
    images, _constants = detect_dust.load_debug_spots_dir(str(BASELINE_DIR))
    if not images:
        print(f"ERROR: No *_debug_spots.json files in {BASELINE_DIR}")
        print("Run generate_baseline.py first.")
        sys.exit(1)
    return {img["stem"]: img for img in images}


def load_baseline_from_session(session_dir):
    """Load baseline from an annotation session directory.

    Reads per-image {stem}_debug_spots.json files as the reference detection,
    then for each image loads *_annotations.json (if present) and removes spots
    listed as false_positives so the corrected baseline reflects the user's
    ground truth.  Returns a dict keyed by stem, same shape as load_baseline().
    """
    session_dir = Path(session_dir)
    images, _constants = detect_dust.load_debug_spots_dir(str(session_dir))
    if not images:
        print(f"ERROR: No *_debug_spots.json files found in {session_dir}")
        sys.exit(1)

    result = {}
    missed_dust_by_stem = {}
    for img in images:
        stem = img["stem"]
        detected = list(img.get("detected", []))

        # Load annotation file and filter out confirmed false positives;
        # also collect missed_dust annotations as expected recoveries.
        ann_path = session_dir / f"{stem}_annotations.json"
        if ann_path.exists():
            with open(ann_path) as f:
                ann = json.load(f)
            fp_list = ann.get("false_positives", [])
            if fp_list:
                fp_indices = set()
                for fp in fp_list:
                    for i, spot in enumerate(detected):
                        d = math.sqrt((fp["cx"] - spot["cx"]) ** 2
                                      + (fp["cy"] - spot["cy"]) ** 2)
                        if d <= MATCH_RADIUS:
                            fp_indices.add(i)
                            break
                detected = [s for i, s in enumerate(detected)
                            if i not in fp_indices]
                if fp_indices:
                    print(f"  [{stem}] removed {len(fp_indices)} confirmed FP(s) from baseline")
            missed_dust = ann.get("missed_dust", [])
            if missed_dust:
                missed_dust_by_stem[stem] = missed_dust

        result[stem] = dict(img, detected=detected)
    return result, missed_dust_by_stem


def _filter_recoveries(new_fps, missed_dust_spots, radius=MATCH_RADIUS):
    """Split new_fps into true false positives and recovered missed-dust spots.

    A new spot is a 'recovery' if it falls within radius of a user-annotated
    missed_dust position — meaning the algorithm now correctly finds real dust
    that the old baseline missed.  These should not penalise the pass/fail score.

    Returns (true_fps, recovered).
    """
    if not missed_dust_spots:
        return new_fps, []
    recovered = []
    true_fps = []
    for fp in new_fps:
        matched = any(
            math.sqrt((fp["cx"] - md["cx"]) ** 2 + (fp["cy"] - md["cy"]) ** 2) <= radius
            for md in missed_dust_spots
        )
        if matched:
            recovered.append(fp)
        else:
            true_fps.append(fp)
    return true_fps, recovered


def write_session(session_dir, raw_results, image_paths_by_stem, diff_by_stem):
    """Write per-image debug_spots + annotations.json to session_dir."""
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    # Write debug_spots.json using the same function as detect_dust.py
    detect_dust.write_debug_spots_json(raw_results, image_paths_by_stem, str(session_dir))

    # Write annotations.json with pre-populated diff markers
    for stem, (missing, new_fps, source_mismatches, radius_mismatches) in diff_by_stem.items():
        ann = {
            "stem": stem,
            # New spots in new run (not in baseline) → show as false positives (red X)
            "false_positives": new_fps,
            # Spots that disappeared from baseline → show as missed dust (cyan crosshair)
            "missed_dust": [{"cx": s["cx"], "cy": s["cy"]} for s in missing],
            # Source mismatches: store baseline source for each mismatched spot
            "source_mismatches": [
                {
                    "spot_idx": ni,
                    "baseline_src_cx": baseline_src[0],
                    "baseline_src_cy": baseline_src[1],
                    "new_src_cx": new_src[0],
                    "new_src_cy": new_src[1],
                    "distance": dist
                }
                for ni, baseline_src, new_src, dist in source_mismatches
            ],
            # Radius mismatches: store baseline radius for each spot with changed brush size
            "radius_mismatches": [
                {
                    "spot_idx": ni,
                    "baseline_brush_radius_px": baseline_r,
                    "new_brush_radius_px": new_r,
                    "radius_diff": diff
                }
                for ni, baseline_r, new_r, diff in radius_mismatches
            ],
        }
        ann_path = session_dir / f"{stem}_annotations.json"
        with open(ann_path, "w") as f:
            json.dump(ann, f, indent=2, cls=detect_dust.NumpyEncoder)


def main():
    parser = argparse.ArgumentParser(description="Dust detection quality regression test")
    parser.add_argument("session", nargs="?", metavar="SESSION_DIR",
                        help="Annotation session directory — uses its images and "
                             "*_debug_spots.json as baseline, filtering out user-confirmed false positives")
    parser.add_argument("--image", metavar="STEM",
                        help="Test only this image (e.g. DSC_0025)")
    parser.add_argument("--open-ui", action="store_true",
                        help="Launch debug_ui.py on the generated diff session when done")
    parser.add_argument("--ml-model", metavar="PATH", default=None,
                        help="Use ML model for detection instead of the rule-based pipeline "
                             "(path to .pkl file produced by train_dust_model.py)")
    args = parser.parse_args()

    missed_dust_by_stem = {}
    if args.session:
        session_dir = Path(args.session)
        if not session_dir.is_dir():
            print(f"ERROR: SESSION_DIR not found: {session_dir}")
            sys.exit(1)
        print(f"Using annotation session: {session_dir}")
        baseline, missed_dust_by_stem = load_baseline_from_session(session_dir)
        images_dir = session_dir
    else:
        baseline = load_baseline()
        images_dir = IMAGES_DIR

    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.jpeg"))
    if args.image:
        image_paths = [p for p in image_paths if p.stem == args.image]
        if not image_paths:
            print(f"ERROR: Image '{args.image}' not found in {images_dir}")
            sys.exit(1)

    if not image_paths:
        print(f"No images found in {images_dir}")
        sys.exit(1)

    if args.ml_model and not os.path.isfile(args.ml_model):
        print(f"ERROR: ML model file not found: {args.ml_model}")
        sys.exit(1)
    if args.ml_model:
        print(f"ML mode: {args.ml_model}")

    print(f"Running detection on {len(image_paths)} image(s)...")
    n_workers = min(cpu_count(), len(image_paths))
    with Pool(processes=n_workers,
              initializer=_init_worker,
              initargs=(args.ml_model,)) as pool:
        raw_results = pool.map(_worker, [str(p) for p in image_paths])
    raw_results.sort(key=lambda r: r[0])

    image_paths_by_stem = {Path(p).stem: str(p) for p in image_paths}

    # --- Compare against baseline ---
    verdicts = []
    diff_by_stem = {}
    baseline_has_sources = False

    for stem, spots, rejected, img_dims, error, _xmp in raw_results:
        new_spots = spots or []
        baseline_entry = baseline.get(stem)

        if error:
            verdicts.append(("FAIL", stem, 0, 0, [], [], [], [], [], [], [], error))
            diff_by_stem[stem] = ([], [], [], [])
            continue

        if baseline_entry is None:
            verdicts.append(("WARN", stem, len(new_spots), 0, [], [], new_spots, [], [], [], [],
                             "not in baseline"))
            diff_by_stem[stem] = ([], new_spots, [], [])
            continue

        baseline_spots = baseline_entry.get("detected", [])

        # Check if baseline has source data
        if baseline_spots and "src_cx" in baseline_spots[0]:
            baseline_has_sources = True

        matched_pairs, missing, new_fps, source_issues, source_mismatches, radius_mismatches = _match_spots(baseline_spots, new_spots)

        # Split new FPs: spots matching user-annotated missed_dust are recoveries, not true FPs
        true_fps, recovered = _filter_recoveries(new_fps, missed_dust_by_stem.get(stem, []))

        verdict, match_rate, new_ratio = _verdict(
            len(baseline_spots), len(new_spots), matched_pairs, missing, true_fps, source_issues, source_mismatches, radius_mismatches)
        verdicts.append((verdict, stem, len(new_spots), len(baseline_spots),
                         matched_pairs, missing, true_fps, source_issues, source_mismatches, radius_mismatches, recovered, None))
        # Write only true FPs to diff session (recoveries are expected improvements)
        diff_by_stem[stem] = (missing, true_fps, source_mismatches, radius_mismatches)

    # --- Print summary ---
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    total_source_issues = 0
    total_source_mismatches = 0
    total_radius_mismatches = 0

    for entry in verdicts:
        verdict, stem, new_count, baseline_count, matched, missing, new_fps, source_issues, source_mismatches, radius_mismatches, recovered, note = entry
        counts[verdict] += 1
        total_source_issues += len(source_issues)
        total_source_mismatches += len(source_mismatches)
        total_radius_mismatches += len(radius_mismatches)

        match_count = len(matched)
        match_rate_str = (f"  match={match_count}/{baseline_count}"
                          if baseline_count else "")
        miss_str = f"  missing={len(missing)}" if missing else ""
        fp_str = f"  new_fps={len(new_fps)}" if new_fps else ""
        rec_str = f"  recovered={len(recovered)}" if recovered else ""
        src_str = f"  src_issues={len(source_issues)}" if source_issues else ""
        mismatch_str = f"  src_mismatch={len(source_mismatches)}" if source_mismatches else ""
        radius_str = f"  radius_mismatch={len(radius_mismatches)}" if radius_mismatches else ""
        note_str = f"  ({note})" if note else ""
        print(f"[{verdict}] {stem:<12} baseline={baseline_count:>3}  new={new_count:>3}"
              f"{match_rate_str}{miss_str}{fp_str}{rec_str}{src_str}{mismatch_str}{radius_str}{note_str}")

    print(f"\nOverall: {counts['PASS']} PASS, {counts['WARN']} WARN, {counts['FAIL']} FAIL")

    if not baseline_has_sources:
        print("\n⚠ WARNING: Baseline does not have source positions.")
        print("   Run generate_baseline.py to update baseline with source detection.")

    if total_source_issues > 0:
        print(f"\n⚠ WARNING: {total_source_issues} source position issue(s) detected.")
        print("   Review diff session for details.")

    if total_source_mismatches > 0:
        print(f"\n⚠ WARNING: {total_source_mismatches} source position mismatch(es) detected.")
        print(f"   Source positions differ from baseline by more than {SOURCE_MISMATCH_RADIUS}px.")
        print("   Review diff session for details.")

    if total_radius_mismatches > 0:
        print(f"\n⚠ WARNING: {total_radius_mismatches} brush radius mismatch(es) detected.")
        print(f"   brush_radius_px differs from baseline by more than {RADIUS_MISMATCH_THRESHOLD}px.")
        print("   Review diff session for details (shown as dashed orange circles in UI).")

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
