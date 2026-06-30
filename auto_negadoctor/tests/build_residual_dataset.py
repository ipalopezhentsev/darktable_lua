"""Assemble the per-frame feature/target table for the LEARNED residual model
(spec 03, Phase 2 — milestone 1). HEAVY: runs the full analytical pipeline
(`process_roll`) on every annotated roll once and caches the result, so the model
(`learn_residual.py`) can be iterated WITHOUT re-running the analysis.

For each annotated frame it records:
  - targets  = the per-frame residual `GT - analytical` for the print-tune look
               params the user hand-tuned (gamma / exposure / black / offset /
               D_max). Both sides are CURRENT: GT from the annotation's `corrected`
               value, analytical from this code's `process_roll` output — so the
               learned variant is "analytical + predicted residual" and stays
               consistent if the analytical algorithm changes (re-run this builder).
  - features = analytical-only signals (milestone 2, NO LLM labels — those exist
               for one roll): the analytical params themselves + image metrics of
               the INVERTED render over the content crop (the picture the user
               judged). The honest floor: if these predict the residual on a
               HELD-OUT roll, image metrics carry taste; if not (spec's hypothesis),
               semantic labels are required (milestone 3).

Output: `residual_dataset.json` next to this file — {created, git_commit, rolls,
target_names, feature_names, rows:[{roll, stem, features{}, targets{}}]}.

Run:  conda run -n autocrop python auto_negadoctor/tests/build_residual_dataset.py
      [--rolls 2506-1 2510-11-1 ...] [--out path.json]
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent.parent))   # repo root -> common
sys.path.insert(0, str(TESTS_DIR.parent))
sys.path.insert(0, str(TESTS_DIR))

import auto_negadoctor as an          # noqa: E402
import nega_model as nm               # noqa: E402
import run_quality_tests as rqt       # noqa: E402

# The scalar print-tune params we model a residual for (wb is a 3-vector — spec
# defers it). Only those a frame's annotation actually carries become a target for
# that frame; a param absent from GT is left out (the model skips it per frame).
TARGET_PARAMS = ("gamma", "exposure", "black", "offset", "D_max")

# Metric stride: a fraction of frame width, so the feature is resolution-invariant
# (docs/python_rules.md). Denser than the tuner but coarser than the gate — these
# are summary stats, not a histogram.
FEATURE_SUBSAMPLE_FRAC = 0.004


def _render_metrics(lin, border, params):
    """Summary stats of the analytical INVERTED render over the content crop — the
    picture the user judged. sRGB float in [0,1] via the same working_to_srgb path
    as the metric/UI. Returns a flat {name: float} or None if the crop is empty."""
    rows = rqt._render_crop_rows(lin, border, params, frac=FEATURE_SUBSAMPLE_FRAC)
    if rows is None or rows.size == 0:
        return None
    r, g, b = rows[:, 0], rows[:, 1], rows[:, 2]
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    mx = rows.max(axis=1)
    mn = rows.min(axis=1)
    chroma = mx - mn                       # simple saturation proxy
    pct = np.percentile(luma, [1, 5, 25, 50, 75, 95, 99])
    return {
        "luma_mean": float(luma.mean()),
        "luma_std": float(luma.std()),
        "luma_p1": float(pct[0]), "luma_p5": float(pct[1]),
        "luma_p25": float(pct[2]), "luma_p50": float(pct[3]),
        "luma_p75": float(pct[4]), "luma_p95": float(pct[5]),
        "luma_p99": float(pct[6]),
        "luma_iqr": float(pct[4] - pct[2]),
        "luma_spread_5_95": float(pct[5] - pct[1]),
        "frac_hi": float(np.mean(mx >= 0.95)),     # near-clipped-bright fraction
        "frac_lo": float(np.mean(mx <= 0.05)),     # near-black fraction
        "chroma_mean": float(chroma.mean()),
        "cast_r_g": float(r.mean() - g.mean()),    # warm/cool color cast
        "cast_b_g": float(b.mean() - g.mean()),
    }


def _param_features(fr):
    """The analytical params themselves as features (the residual often depends on
    how extreme the analytical value already is) + exif brightness."""
    p = fr["params"]
    feats = {
        "an_gamma": float(p.get("gamma", 0.0)),
        "an_exposure": float(p.get("exposure", 0.0)),
        "an_black": float(p.get("black", 0.0)),
        "an_offset": float(p.get("offset", 0.0)),
        "an_dmax": float(p.get("D_max", 0.0)),
        "an_soft_clip": float(p.get("soft_clip", 0.0)),
        "exposure_factor": float(fr.get("exposure_factor", 0.0) or 0.0),
    }
    for i, c in enumerate("rgb"):
        feats[f"an_wb_low_{c}"] = float((p.get("wb_low") or [0, 0, 0])[i])
        feats[f"an_wb_high_{c}"] = float((p.get("wb_high") or [0, 0, 0])[i])
        feats[f"dmin_{c}"] = float((fr.get("dmin") or [0, 0, 0])[i])
    return feats


def _git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           cwd=str(TESTS_DIR), capture_output=True, text=True,
                           timeout=5)
        return r.stdout.strip() or None
    except Exception:
        return None


def build(roll_ids=None):
    rolls = rqt.discover_rolls()
    if roll_ids:
        want = set(roll_ids)
        rolls = [r for r in rolls if r["id"] in want]
    if not rolls:
        print("No rolls with local TIFFs — see fixtures/rolls/README.md.")
        return None

    rows, feature_names = [], None
    for i, roll in enumerate(rolls, 1):
        t0 = time.perf_counter()
        print(f"[{i}/{len(rolls)}] roll {roll['id']}: process_roll on "
              f"{len(roll['images'])} frame(s)…", flush=True)
        frames, _ = an.process_roll(roll["images"], roll["exif"])
        gt_by_stem = rqt._load_ground_truth(roll["fixtures"])
        by_stem = {fr["stem"]: fr for fr in frames if not fr.get("error")}
        n_used = 0
        for stem, gt in gt_by_stem.items():
            fr = by_stem.get(stem)
            if not fr or "params" not in fr or not fr.get("border"):
                continue
            p = fr["params"]
            targets = {name: float(gt[name]) - float(p[name])
                       for name in TARGET_PARAMS
                       if gt.get(name) is not None and p.get(name) is not None}
            if not targets:
                continue
            try:
                _enc, lin = an.load_frame(fr["path"], fr.get("vignette"))
            except Exception as e:
                print(f"    skip {stem}: load_frame failed ({e})", flush=True)
                continue
            rmet = _render_metrics(lin, fr["border"], p)
            if rmet is None:
                continue
            feats = {**_param_features(fr), **rmet}
            if feature_names is None:
                feature_names = sorted(feats)
            rows.append({"roll": roll["id"], "stem": stem,
                         "features": feats, "targets": targets})
            n_used += 1
        print(f"    {n_used} annotated frame(s) -> rows "
              f"({round(time.perf_counter() - t0, 1)}s)", flush=True)

    out = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "rolls": [r["id"] for r in rolls],
        "target_names": list(TARGET_PARAMS),
        "feature_names": feature_names or [],
        "n_rows": len(rows),
        "rows": rows,
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rolls", nargs="+", help="restrict to these roll ids")
    ap.add_argument("--out", default=str(TESTS_DIR / "residual_dataset.json"),
                    help="output JSON path")
    args = ap.parse_args()
    data = build(args.rolls)
    if data is None:
        return 1
    Path(args.out).write_text(json.dumps(data, indent=2))
    print(f"\nWrote {data['n_rows']} rows from {len(data['rolls'])} roll(s) "
          f"({len(data['feature_names'])} features, targets "
          f"{', '.join(data['target_names'])}) -> {args.out}")
    # quick per-roll / per-target coverage so gaps are visible before modeling
    by_roll = {}
    for r in data["rows"]:
        by_roll.setdefault(r["roll"], []).append(r)
    for rid, rs in by_roll.items():
        cov = {t: sum(1 for r in rs if t in r["targets"])
               for t in data["target_names"]}
        print(f"  {rid}: {len(rs)} rows | target coverage "
              f"{ {k: v for k, v in cov.items() if v} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
