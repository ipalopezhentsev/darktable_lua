"""
Train an ML post-filter model for dust detection from a human-annotated debug session.

The model is a post-filter: it runs AFTER the rule-based pipeline and removes
false positives from the set of accepted spots.  It can also help surface missed
dust via a lower-threshold recovery pass in detect_dust_spots_ml().

Training data comes directly from debug_spots.json (the saved detection results),
so no image reprocessing is needed — training is fast.

  Positive (label=1): accepted spots NOT marked as false positives
  Negative (label=0): accepted spots marked as false positives in annotations

Usage:
    conda run -n autocrop python auto_retouch/train_dust_model.py \\
        "C:\\path\\to\\annotation_session" \\
        --output auto_retouch/dust_ml_model.pkl

The output pickle contains:
    {"model": RandomForestClassifier, "scaler": StandardScaler,
     "features": SPOT_FEATURE_NAMES, "threshold_mult": float}
"""

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap: add auto_retouch dir to sys.path so we can import detect_dust
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import detect_dust as _dd

MATCH_RADIUS = 15   # px — same as run_quality_tests.py


def _dist(ax, ay, bx, by):
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def build_dataset(session_dir):
    """Build a labeled dataset from a debug session's JSON files.

    Returns (X, y, info) where:
      X  — (n_samples, n_features) float32
      y  — (n_samples,) int  (1=dust, 0=not-dust)
      info — dict with per-image stats
    """
    session_dir = Path(session_dir)
    spots_json_path = session_dir / "debug_spots.json"
    if not spots_json_path.exists():
        print(f"ERROR: {spots_json_path} not found")
        sys.exit(1)

    with open(spots_json_path) as f:
        data = json.load(f)

    all_X = []
    all_y = []
    total_pos = total_neg = total_no_ann = 0
    images_info = []

    for img_data in data["images"]:
        stem = img_data["stem"]
        detected = img_data.get("detected", [])

        ann_path = session_dir / f"{stem}_annotations.json"
        fp_spots = []
        if ann_path.exists():
            with open(ann_path) as f:
                ann = json.load(f)
            fp_spots = ann.get("false_positives", [])

        img_pos = img_neg = 0
        for spot in detected:
            is_fp = any(_dist(spot["cx"], spot["cy"], fp["cx"], fp["cy"]) <= MATCH_RADIUS
                        for fp in fp_spots)
            label = 0 if is_fp else 1
            all_X.append(_dd._spot_to_features(spot))
            all_y.append(label)
            if label == 1:
                img_pos += 1
            else:
                img_neg += 1

        total_pos += img_pos
        total_neg += img_neg
        if not fp_spots:
            total_no_ann += 1

        if detected:
            images_info.append(f"  {stem}: {len(detected)} spots  "
                                f"{img_neg} FP-labeled  {img_pos} positive")

    for line in images_info:
        print(line)

    print(f"\nDataset summary:")
    print(f"  Positive (dust)          : {total_pos}")
    print(f"  Negative (false positive): {total_neg}")
    print(f"  Images without FP annot. : {total_no_ann}")

    if not all_X:
        print("ERROR: no spots found in debug_spots.json — check the session")
        sys.exit(1)

    if total_neg == 0:
        print("WARNING: no negative examples found — the model won't learn to reject FPs")

    return (np.array(all_X, dtype=np.float32),
            np.array(all_y, dtype=np.int32),
            {"pos": total_pos, "neg": total_neg})


def train_model(X, y):
    """Train a RandomForest classifier with cross-validation reporting.

    Returns (fitted_model, fitted_scaler).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

    n_neg = int(np.sum(y == 0))
    n_pos = int(np.sum(y == 1))
    print(f"\nTraining RandomForest on {len(X)} samples "
          f"({n_pos} positive, {n_neg} negative) ...")
    print(f"Features ({len(_dd.SPOT_FEATURE_NAMES)}): "
          f"{', '.join(_dd.SPOT_FEATURE_NAMES)}")

    scaler = StandardScaler()
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    pipe = Pipeline([("scaler", scaler), ("clf", clf)])

    # Cross-validation (up to 5-fold, stratified, limited by minority class size)
    n_splits = min(5, max(2, n_neg))
    if n_splits >= 2 and n_neg >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scorers = {
            "precision": make_scorer(precision_score, zero_division=0),
            "recall":    make_scorer(recall_score,    zero_division=0),
            "f1":        make_scorer(f1_score,         zero_division=0),
        }
        cv_results = cross_validate(pipe, X, y, cv=cv, scoring=scorers, n_jobs=-1)
        print(f"\n  Cross-validation ({n_splits} folds):")
        print(f"    Precision: {cv_results['test_precision'].mean():.3f} "
              f"± {cv_results['test_precision'].std():.3f}")
        print(f"    Recall:    {cv_results['test_recall'].mean():.3f} "
              f"± {cv_results['test_recall'].std():.3f}")
        print(f"    F1:        {cv_results['test_f1'].mean():.3f} "
              f"± {cv_results['test_f1'].std():.3f}")
    else:
        print("  (too few negative samples for cross-validation, skipping)")

    # Fit on all data
    pipe.fit(X, y)

    # Feature importances
    importances = pipe.named_steps["clf"].feature_importances_
    feat_imp = sorted(zip(_dd.SPOT_FEATURE_NAMES, importances),
                      key=lambda x: x[1], reverse=True)
    print("\n  Feature importances:")
    for name, imp in feat_imp:
        bar = "#" * int(imp * 80)
        print(f"    {name:<28} {imp:.4f}  {bar}")

    return pipe.named_steps["clf"], pipe.named_steps["scaler"]


def main():
    parser = argparse.ArgumentParser(
        description="Train dust post-filter ML model from an annotation session.\n"
                    "Reads features directly from debug_spots.json — no image reprocessing.")
    parser.add_argument("session_dir", metavar="SESSION_DIR",
                        help="Path to annotation session directory (must contain "
                             "debug_spots.json and *_annotations.json files)")
    parser.add_argument("--output", metavar="PATH",
                        default=str(_HERE / "dust_ml_model.pkl"),
                        help="Output path for the trained model pickle "
                             "(default: auto_retouch/dust_ml_model.pkl)")
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    if not session_dir.is_dir():
        print(f"ERROR: SESSION_DIR not found: {session_dir}")
        sys.exit(1)

    output_path = Path(args.output)

    print(f"Session dir  : {session_dir}")
    print(f"Output model : {output_path}")
    print()

    print("Building labeled dataset from debug_spots.json ...")
    X, y, info = build_dataset(session_dir)

    model, scaler = train_model(X, y)

    bundle = {
        "model":         model,
        "scaler":        scaler,
        "features":      _dd.SPOT_FEATURE_NAMES,
        "threshold_mult": _dd.ML_RECOVERY_THRESHOLD_MULT,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\nModel saved to: {output_path}")
    print(f"Use with: python detect_dust.py --ml-model \"{output_path}\" <images>")


if __name__ == "__main__":
    main()
