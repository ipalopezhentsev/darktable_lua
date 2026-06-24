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


def _worker(image_path):
    stem = Path(image_path).stem

    import time
    import cv2
    t0 = time.perf_counter()
    img = cv2.imread(str(image_path))
    if img is None:
        return (stem, None, [], (0, 0), f"Failed to load: {image_path}", None, 0.0)
    height, width = img.shape[:2]

    spots, rejected, error, local_std = detect_dust.detect(
        str(image_path), collect_rejects=True)
    elapsed = time.perf_counter() - t0
    return (stem, spots, rejected, (width, height), error, None, elapsed)


def _dist(a, b):
    return math.sqrt((a["cx"] - b["cx"]) ** 2 + (a["cy"] - b["cy"]) ** 2)


STROKE_MATCH_RADIUS = 60   # px: stroke midpoints within this (and similar length) = same stroke


def _is_stroke(s):
    return s.get("kind") == "stroke"


# --- Baseline-independent per-spot invariants ---------------------------------
# Properties that must hold for EVERY detected spot on EVERY run, no baseline needed.
# The headline one: the healing SOURCE brush must not overlap the DEFECT brush — darktable's
# clone smears across an overlap (the failure modes fixed in tuning passes 16 & 22). Because
# run_quality_tests does not diff stroke sources against a baseline (the committed baseline has
# no strokes), this invariant is the only thing guarding that work; keep extending it as
# features add new geometry.
CLEARANCE_TOL_PX = 1.0   # allow brush edges to touch; flag only genuine overlap
BRUSH_RADIUS_CAP_FRAC = 0.02   # brush_radius_px above this * min_dim is treated as a runaway


def _densify_path(path, step):
    """Resample a polyline so curvature is represented when measuring source/defect clearance."""
    pts = [np.array(path[0], dtype=float)]
    for i in range(len(path) - 1):
        a, b = np.array(path[i], dtype=float), np.array(path[i + 1], dtype=float)
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(seg / step))
        for k in range(1, n + 1):
            pts.append(a + (b - a) * (k / n))
    return np.array(pts)


def check_spot_invariants(spots, width, height):
    """Validate every spot's healing source + geometry. Returns a list of issue dicts
    {idx, kind, severity ('error'|'warn'), msg, cx, cy}.

    error (fails the suite): source brush overlaps the defect brush, source out of bounds,
      non-finite/missing coords, brush_radius<=0, or a stroke path with < 2 points.
    warn: brush radius looks like a runaway, or a stroke source is closer than the intended
      gap (STROKE_SOURCE_MIN_GAP_PX) though not yet overlapping.
    """
    min_dim = min(width, height)
    step = getattr(detect_dust, "HEAL_SAMPLE_STEP_PX", 8.0)
    gap_intent = getattr(detect_dust, "STROKE_SOURCE_MIN_GAP_PX", 10.0)
    br_cap = BRUSH_RADIUS_CAP_FRAC * min_dim
    issues = []
    for idx, s in enumerate(spots):
        kind = "stroke" if _is_stroke(s) else "dot"
        cx, cy = s.get("cx"), s.get("cy")
        scx, scy = s.get("src_cx"), s.get("src_cy")
        br = s.get("brush_radius_px")

        def add(sev, msg):
            issues.append({"idx": idx, "kind": kind, "severity": sev, "msg": msg,
                           "cx": float(cx) if cx is not None else 0.0,
                           "cy": float(cy) if cy is not None else 0.0})

        vals = [cx, cy, scx, scy, br]
        if any(v is None or not math.isfinite(float(v)) for v in vals):
            add("error", f"missing/non-finite field (cx,cy,src,br={vals})")
            continue
        cx, cy, scx, scy, br = (float(v) for v in vals)

        if not (0.0 <= scx <= width - 1 and 0.0 <= scy <= height - 1):
            add("error", f"source out of bounds ({scx:.0f},{scy:.0f}) vs {width}x{height}")
        if br <= 0.0:
            add("error", f"brush_radius_px <= 0 ({br})")
        elif br > br_cap:
            add("warn", f"brush_radius_px {br:.1f} > cap {br_cap:.1f} (runaway brush?)")

        # Min separation between the source brush centre-line and the defect centre-line.
        if kind == "stroke":
            path = s.get("path") or []
            if len(path) < 2:
                add("error", f"stroke path has < 2 points ({len(path)})")
                continue
            dense = _densify_path(path, step)
            off = np.array([scx - path[0][0], scy - path[0][1]])
            src = dense + off
            dx = src[:, 0][:, None] - dense[:, 0][None, :]
            dy = src[:, 1][:, None] - dense[:, 1][None, :]
            sep = float(np.sqrt((dx * dx + dy * dy).min()))
        else:
            sep = math.hypot(scx - cx, scy - cy)

        gap = sep - 2.0 * br   # gap between the two brush EDGES (both have radius br)
        if gap < -CLEARANCE_TOL_PX:
            add("error", f"source brush OVERLAPS defect (edge gap {gap:.1f}px; "
                         f"sep {sep:.1f} < 2*br {2 * br:.1f})")
        elif kind == "stroke" and gap < gap_intent - 1.0:
            add("warn", f"stroke source tighter than intended gap "
                        f"({gap:.1f}px < {gap_intent:.0f}px)")
    return issues


def _match_strokes(baseline_strokes, new_strokes):
    """Match thread/scratch strokes between baseline and new runs.

    Strokes are matched by centerline-midpoint proximity plus a length-similarity check
    (a single point match is unreliable for elongated forms). Returns (matched, missing,
    new) where matched is a list of (baseline_idx, new_idx, baseline, new).
    """
    used = set()
    matched = []
    for bi, bs in enumerate(baseline_strokes):
        best_ni, best_d = None, float("inf")
        for ni, ns in enumerate(new_strokes):
            if ni in used:
                continue
            d = _dist(bs, ns)
            bl, nl = bs.get("length_px", 0.0), ns.get("length_px", 0.0)
            lr = (min(bl, nl) / max(bl, nl)) if max(bl, nl) > 0 else 1.0
            if d < best_d and d <= STROKE_MATCH_RADIUS and lr >= 0.5:
                best_d, best_ni = d, ni
        if best_ni is not None:
            matched.append((bi, best_ni, bs, new_strokes[best_ni]))
            used.add(best_ni)
    missing = [bs for bi, bs in enumerate(baseline_strokes)
               if bi not in {m[0] for m in matched}]
    new = [ns for ni, ns in enumerate(new_strokes) if ni not in used]
    return matched, missing, new


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


# ---------------------------------------------------------------------------
# Calibration score helpers (SHARED with tests/run_calibration.py — the same
# "share the metric between gate and runner" split auto_negadoctor uses). A frame's
# annotations are the ground truth: `false_positives` (detections the user marked
# NOT-dust → want gone) and `missed_dust` / `missed_strokes` (defects the user wants
# found → want covered). Each helper returns per-frame counts; the runner's
# evaluator turns them into a (precision-weighted) objective.
# ---------------------------------------------------------------------------

def _stroke_gt_from_ann(ms):
    """A missed-stroke annotation ({path:[[x,y]...], stroke_width_px}) → the {cx, cy,
    length_px, path} shape `_match_strokes` consumes (midpoint + polyline length)."""
    path = ms.get("path") or []
    if not path:
        return {"cx": ms.get("cx", 0.0), "cy": ms.get("cy", 0.0),
                "length_px": ms.get("length_px", 0.0)}
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    length = sum(math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
                 for i in range(len(path) - 1))
    return {"cx": sum(xs) / len(xs), "cy": sum(ys) / len(ys),
            "length_px": length, "path": path}


def dust_score_per_frame(detected_dots, ann, radius=MATCH_RADIUS):
    """Score a frame's circular-dust detections against its annotations.
      n_fp     annotated false_positives the detection still REPRODUCES (bad).
      n_missed annotated missed_dust NOT covered by any detection (bad).
    """
    fps = ann.get("false_positives", []) or []
    missed = ann.get("missed_dust", []) or []
    n_fp = sum(1 for fp in fps
               if any(_dist(fp, s) <= radius for s in detected_dots))
    n_missed = sum(1 for md in missed
                   if not any(_dist(md, s) <= radius for s in detected_dots))
    return {"n_fp": n_fp, "n_missed": n_missed, "n_detected": len(detected_dots),
            "n_fp_ann": len(fps), "n_missed_ann": len(missed)}


def stroke_score_per_frame(detected_strokes, ann, radius=MATCH_RADIUS):
    """Score a frame's stroke detections against its annotations.
      n_missed annotated missed_strokes NOT matched by a detected stroke (bad).
      n_fp     detected strokes whose midpoint matches a false_positive point (bad).
    """
    gt = [_stroke_gt_from_ann(ms) for ms in (ann.get("missed_strokes", []) or [])]
    fps = ann.get("false_positives", []) or []
    _matched, missing, _new = _match_strokes(gt, detected_strokes)
    n_fp = sum(1 for s in detected_strokes
               if any(_dist(fp, s) <= radius for fp in fps))
    return {"n_missed": len(missing), "n_fp": n_fp,
            "n_detected": len(detected_strokes), "n_missed_ann": len(gt)}


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


def write_session(session_dir, raw_results, image_paths_by_stem, diff_by_stem,
                  times_by_stem=None, wall_time_s=None):
    """Write per-image debug_spots + annotations.json to session_dir."""
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    # Write debug_spots.json using the same function as detect_dust.py
    detect_dust.write_debug_spots_json(raw_results, image_paths_by_stem,
                                       str(session_dir), times_by_stem=times_by_stem,
                                       wall_time_s=wall_time_s)

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

    print(f"Running detection on {len(image_paths)} image(s)...")
    n_workers = min(cpu_count(), len(image_paths))
    import time as _time
    wall_t0 = _time.perf_counter()
    with Pool(processes=n_workers) as pool:
        raw_results = pool.map(_worker, [str(p) for p in image_paths])
    wall_time = _time.perf_counter() - wall_t0
    raw_results.sort(key=lambda r: r[0])

    # Per-image wall-clock times (last tuple element); strip back to the
    # 6-tuple shape the comparison code and write_debug_spots_json expect
    times_by_stem = {r[0]: r[6] for r in raw_results}
    raw_results = [r[:6] for r in raw_results]
    baseline_time_by_stem = {stem: entry.get("processing_time_s")
                             for stem, entry in baseline.items()}
    baseline_wall_time = next((entry.get("run_wall_time_s")
                               for entry in baseline.values()
                               if entry.get("run_wall_time_s")), None)

    image_paths_by_stem = {Path(p).stem: str(p) for p in image_paths}

    # --- Compare against baseline ---
    verdicts = []
    diff_by_stem = {}
    stroke_stats = {}   # stem -> (matched, missing, new) for thread/scratch strokes
    invariant_by_stem = {}   # stem -> list of per-spot invariant issues (source clearance etc.)
    baseline_has_sources = False

    for stem, spots, rejected, img_dims, error, _xmp in raw_results:
        new_spots = spots or []
        baseline_entry = baseline.get(stem)

        if error:
            verdicts.append(("FAIL", stem, 0, 0, [], [], [], [], [], [], [], error))
            diff_by_stem[stem] = ([], [], [], [])
            continue

        # Baseline-independent invariants (source must clear the defect, sane geometry).
        # Runs for every successfully processed image, including ones absent from the baseline.
        width_i, height_i = img_dims
        inv = check_spot_invariants(new_spots, width_i, height_i)
        if inv:
            invariant_by_stem[stem] = inv

        if baseline_entry is None:
            verdicts.append(("WARN", stem, len(new_spots), 0, [], [], new_spots, [], [], [], [],
                             "not in baseline"))
            diff_by_stem[stem] = ([], new_spots, [], [])
            continue

        baseline_spots = baseline_entry.get("detected", [])

        # Check if baseline has source data
        if baseline_spots and "src_cx" in baseline_spots[0]:
            baseline_has_sources = True

        # Partition by kind: dots use the established matcher (keeps dot regression
        # metrics comparable to the committed baseline); strokes match separately.
        base_dots = [s for s in baseline_spots if not _is_stroke(s)]
        base_strokes = [s for s in baseline_spots if _is_stroke(s)]
        new_dots = [s for s in new_spots if not _is_stroke(s)]
        new_strokes_l = [s for s in new_spots if _is_stroke(s)]

        s_matched, s_missing, s_new = _match_strokes(base_strokes, new_strokes_l)
        stroke_stats[stem] = (len(s_matched), len(s_missing), len(s_new))

        matched_pairs, missing, new_fps, source_issues, source_mismatches, radius_mismatches = _match_spots(base_dots, new_dots)

        # Split new FPs: spots matching user-annotated missed_dust are recoveries, not true FPs
        true_fps, recovered = _filter_recoveries(new_fps, missed_dust_by_stem.get(stem, []))

        verdict, match_rate, new_ratio = _verdict(
            len(base_dots), len(new_dots), matched_pairs, missing, true_fps, source_issues, source_mismatches, radius_mismatches)
        verdicts.append((verdict, stem, len(new_dots), len(base_dots),
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
        t_new = times_by_stem.get(stem)
        t_base = baseline_time_by_stem.get(stem)
        time_str = ""
        if t_new is not None:
            time_str = f"  time={t_new:.1f}s"
            if t_base:
                time_str += f" (base {t_base:.1f}s)"
        print(f"[{verdict}] {stem:<12} baseline={baseline_count:>3}  new={new_count:>3}"
              f"{match_rate_str}{miss_str}{fp_str}{rec_str}{src_str}{mismatch_str}{radius_str}{time_str}{note_str}")

    print(f"\nOverall: {counts['PASS']} PASS, {counts['WARN']} WARN, {counts['FAIL']} FAIL")

    # --- Timing summary (informational; baseline times exist once the baseline
    # is regenerated with processing_time_s / run_wall_time_s). The wall time is
    # the comparable number — per-image times overlap under parallel workers and
    # are inflated by contention, so they only rank images within one run. ---
    if baseline_wall_time:
        wall_delta = (wall_time - baseline_wall_time) / baseline_wall_time * 100
        print(f"Run wall time: {wall_time:.1f}s "
              f"(baseline {baseline_wall_time:.1f}s, {wall_delta:+.0f}%)")
    else:
        print(f"Run wall time: {wall_time:.1f}s "
              f"(no baseline wall time — regenerate baseline to record it)")
    total_time = sum(times_by_stem.values())
    common = [stem for stem in times_by_stem if baseline_time_by_stem.get(stem)]
    if common:
        new_common = sum(times_by_stem[s] for s in common)
        base_common = sum(baseline_time_by_stem[s] for s in common)
        delta_pct = (new_common - base_common) / base_common * 100 if base_common else 0.0
        print(f"Per-image detection time (parallel, contention-inflated): "
              f"{total_time:.1f}s summed; vs baseline on {len(common)} "
              f"image(s): {new_common:.1f}s vs {base_common:.1f}s ({delta_pct:+.0f}%)")
    else:
        print(f"Per-image detection time (parallel, contention-inflated): "
              f"{total_time:.1f}s summed (no baseline times)")

    # --- Stroke (thread/scratch) summary, reported separately from dots ---
    s_tot_matched = sum(v[0] for v in stroke_stats.values())
    s_tot_missing = sum(v[1] for v in stroke_stats.values())
    s_tot_new = sum(v[2] for v in stroke_stats.values())
    if s_tot_matched or s_tot_missing or s_tot_new:
        print(f"\nStrokes: {s_tot_matched} matched, {s_tot_missing} missing (vs baseline), "
              f"{s_tot_new} new")
        for stem, (m, miss, nw) in sorted(stroke_stats.items()):
            if miss or nw:
                print(f"   {stem:<12} matched={m} missing={miss} new={nw}")
        if s_tot_missing:
            print(f"⚠ WARNING: {s_tot_missing} baseline stroke(s) no longer detected.")

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

    # --- Source/geometry invariants (baseline-independent; guards heal passes 16 & 22) ---
    inv_errors = sum(1 for v in invariant_by_stem.values() for i in v if i["severity"] == "error")
    inv_warns = sum(1 for v in invariant_by_stem.values() for i in v if i["severity"] == "warn")
    if inv_errors or inv_warns:
        print(f"\nSource/geometry invariants: {inv_errors} error(s), {inv_warns} warning(s)")
        for stem in sorted(invariant_by_stem):
            for i in invariant_by_stem[stem]:
                tag = "ERROR" if i["severity"] == "error" else "warn "
                print(f"  [{tag}] {stem:<12} {i['kind']} #{i['idx']} "
                      f"@({i['cx']:.0f},{i['cy']:.0f}): {i['msg']}")
        if inv_errors:
            print(f"✗ {inv_errors} invariant ERROR(s): a healing source overlaps its defect, is "
                  f"out of bounds, or geometry is degenerate. darktable will smear there.")
    else:
        print("\nSource/geometry invariants: all spots OK (every source clears its defect).")

    # --- Write diff session ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(tempfile.gettempdir()) / f"dt_qualitytest_{timestamp}"
    write_session(session_dir, raw_results, image_paths_by_stem, diff_by_stem,
                  times_by_stem=times_by_stem, wall_time_s=wall_time)

    debug_ui_script = AUTO_RETOUCH_DIR / "debug_ui.py"
    open_cmd = f"conda run -n autocrop python \"{debug_ui_script}\" \"{session_dir}\""
    print(f"\nDiff session: {session_dir}")
    print(f"Review visually: {open_cmd}")

    if args.open_ui:
        subprocess.Popen([sys.executable, str(debug_ui_script), str(session_dir)])

    sys.exit(0 if counts["FAIL"] == 0 and inv_errors == 0 else 1)


if __name__ == "__main__":
    main()
