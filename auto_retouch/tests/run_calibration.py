"""Reproducible, RECORDED calibration sessions — auto_retouch (dust + stroke).

The shared engine (optimizers, recorded-session lifecycle, CLI) lives in
`common/calibration/runner.py` — the SAME engine auto_negadoctor uses. This module
supplies only the retouch-specific pieces: the per-kind EVALUATORS (the metric),
roll discovery, the parallel map, the per-kind cosmetics, and the `--review` hook —
wired through a `runner.CalibrationAdapter`.

Two KINDS, each judged by the user's per-frame annotations (the ground truth):
  dust    minimise W_FP·(detections reproducing an annotated false_positive) +
          W_MISS·(annotated missed_dust left uncovered). PRECISION-WEIGHTED
          (W_FP > W_MISS) — the user prioritises few false positives over recall.
  stroke  minimise W_MISS·(annotated missed_strokes uncovered) + W_FP·(detected
          strokes landing on a false_positive point).

Ground truth lives EXACTLY as in auto_negadoctor: one folder per roll under
`tests/fixtures/rolls/<roll_id>/`, holding the roll's images (`*.jpg`) and one or
more annotation-session subfolders of `{stem}_annotations.json`.

RECORD-ONLY: the runner never edits detect_dust.py. A session writes its best
constants to `fitted_params.json` + a drop-in `fitted_preset.json`; adopt by hand
(`RETOUCH_PRESET=<preset>`), exactly like negadoctor.

Run (evaluate current detector, no search; needs roll images present):
  conda run -n autocrop python auto_retouch/tests/run_calibration.py \
      --config .../calibrations/configs/dust_default.json --method none
Fit:        ... --method coordinate_descent   (slow — detection per frame per trial)
Review:     ... --review <session_dir> [roll_id]
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent.parent))   # repo root -> common
sys.path.insert(0, str(TESTS_DIR.parent))          # auto_retouch -> detect_dust/tuning
sys.path.insert(0, str(TESTS_DIR))

import detect_dust as dd              # noqa: E402
import tuning                         # noqa: E402
import run_quality_tests as rqt       # noqa: E402
import calibration_registry as reg    # noqa: E402
from common.calibration import runner  # noqa: E402

ROLLS_DIR = TESTS_DIR / "fixtures" / "rolls"
CALIB_DIR = TESTS_DIR / "calibrations"

BIG_PENALTY = runner.BIG_PENALTY
_median = runner._median
_per_roll_summary = runner.per_roll_summary

METRIC_NAME = {
    "dust": "W_FP*false_positives + W_MISS*missed_dust (precision-weighted)",
    "stroke": "W_MISS*missed_strokes + W_FP*stroke_false_positives",
}


def _eval_frames(items, fn):
    return runner.eval_frames(ADAPTER, items, fn)


# ---------------------------------------------------------------------------
# Roll discovery (negadoctor-style fixtures/rolls/<roll_id>/)
# ---------------------------------------------------------------------------

def discover_rolls(roll_ids=None):
    rolls = []
    if not ROLLS_DIR.is_dir():
        return rolls
    for d in sorted(p for p in ROLLS_DIR.iterdir() if p.is_dir()):
        rid = d.name
        if roll_ids and rid not in roll_ids:
            continue
        images = sorted(d.glob("*.jpg")) + sorted(d.glob("*.jpeg"))
        fixtures = sorted(d.rglob("*_annotations.json"))
        if not fixtures:
            continue
        if not images:
            print(f"SKIP roll {rid}: annotations present but no images — drop the "
                  f"roll's JPGs into {d} (see fixtures/rolls/README.md).")
            continue
        rolls.append({"id": rid, "dir": d, "images": images, "fixtures": fixtures})
    return rolls


def _load_annotations(fixtures):
    """stem -> merged annotation dict (field-level last-writer-wins across sessions)."""
    out = {}
    for f in fixtures:
        data = json.loads(f.read_text())
        stem = data.get("stem") or f.name.replace("_annotations.json", "")
        cur = out.setdefault(stem, {})
        for k, v in data.items():
            if v:
                cur[k] = v
    return out


def _frames_with_signal(rolls, kind):
    """[(roll_id, stem, image_path, ann)] for frames whose annotations carry signal
    for this kind (empty-annotation frames contribute nothing, so they are skipped —
    this is what keeps a trial from re-detecting the whole roll for no gradient)."""
    items = []
    for roll in rolls:
        anns = _load_annotations(roll["fixtures"])
        imgs = {p.stem: p for p in roll["images"]}
        for stem, ann in anns.items():
            if stem not in imgs:
                continue
            if kind == "dust":
                signal = ann.get("false_positives") or ann.get("missed_dust")
            else:
                signal = ann.get("missed_strokes") or ann.get("false_positives")
            if signal:
                items.append((roll["id"], stem, str(imgs[stem]), ann))
    return items


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

def _make_evaluator(rolls, tolerances, kind):
    w_fp = float(tolerances.get("w_fp", 3.0))     # precision priority: FP costs more
    w_miss = float(tolerances.get("w_miss", 1.0))
    items = _frames_with_signal(rolls, kind)
    prep_secs = {roll["id"]: 0.0 for roll in rolls}  # detection IS the per-trial work
    score_fn = rqt.dust_score_per_frame if kind == "dust" else rqt.stroke_score_per_frame
    want_stroke = (kind == "stroke")

    def evaluate(overrides):
        cfg = reg.to_tuning(overrides)   # immutable per-trial cfg (no globals touched)

        def _one(it):
            rid, stem, path, ann = it
            spots, _rej, err, _ls = dd.detect(path, cfg=cfg)
            spots = spots or []
            kind_spots = [s for s in spots
                          if (s.get("kind") == "stroke") == want_stroke]
            sc = score_fn(kind_spots, ann)
            score = w_fp * sc["n_fp"] + w_miss * sc["n_missed"]
            rec = {"roll": rid, "stem": stem, "score": score,
                   "error": err, **sc}
            return rec

        per_frame = _eval_frames(items, _one)
        tot_fp = sum(r["n_fp"] for r in per_frame)
        tot_miss = sum(r["n_missed"] for r in per_frame)
        objective = w_fp * tot_fp + w_miss * tot_miss
        aggregate = {"objective_weighted": objective,
                     "total_fp": tot_fp, "total_missed": tot_miss,
                     "w_fp": w_fp, "w_miss": w_miss, "n_frames": len(per_frame)}
        return {"objective": objective, "per_frame": per_frame,
                "per_roll": _per_roll_summary(per_frame, "score"),
                "aggregate": aggregate}

    return evaluate, prep_secs


def make_dust_evaluator(rolls, tolerances, fit_params=()):
    return _make_evaluator(rolls, tolerances, "dust")


def make_stroke_evaluator(rolls, tolerances, fit_params=()):
    return _make_evaluator(rolls, tolerances, "stroke")


EVALUATORS = {"dust": make_dust_evaluator, "stroke": make_stroke_evaluator}


# ---------------------------------------------------------------------------
# Per-kind cosmetics + review
# ---------------------------------------------------------------------------

def _worst_key(kind):
    return ("score", True)


def _headline(kind, agg):
    return (f"fp {agg['total_fp']} missed {agg['total_missed']} "
            f"(W_fp={runner._fmtv(agg['w_fp'])})")


def _gt_review_payload(ann, fitted_detected, min_dim):
    """The GROUND-TRUTH spot set for the R cycle's GT source = the user-corrected
    output: (fitted detections minus the ones reproducing an annotated
    `false_positive`) + the hand-added `missed_dust` / `missed_strokes`. The
    dropped FP detections become the GT 'rejected' list (so they show as rejected
    candidates). Mirrors the apply writer's final spot set. Returns
    {detected, rejected} or None if the frame carries no annotation."""
    fps = ann.get("false_positives") or []
    missed = ann.get("missed_dust") or []
    strokes = ann.get("missed_strokes") or []
    if not (fps or missed or strokes):
        return None
    kept, rejected = [], []
    for s in (fitted_detected or []):
        if any(rqt._dist(fp, s) <= rqt.MATCH_RADIUS for fp in fps):
            rejected.append(s)
        else:
            kept.append(s)
    detected = list(kept)
    for md in missed:
        detected.append(dd.missed_dust_to_spot(md, fitted_detected, None))
    for ms in strokes:
        sp = dd.missed_stroke_to_spot(ms, min_dim)
        if sp:
            detected.append(sp)
    return {"detected": detected, "rejected": rejected}


def review_session(session_dir, roll_id=None):
    """Open the dust debug UI on a finished session showing its FITTED detection,
    with R cycling FITTED -> GT (your annotation, the corrected output) -> LIVE
    (the current source-code detection) -> FITTED — the same three-way review
    auto_negadoctor offers. All sources are precomputed here and written into a
    throwaway session dir as the frames' `review` payload, so the toggle in the
    UI is instant (no re-detection)."""
    import shutil
    import subprocess
    import tempfile
    from PIL import Image
    session = Path(session_dir)
    config = json.loads((session / "config.json").read_text())
    results = json.loads((session / "results.json").read_text())
    kind = config["kind"]
    fitted = results.get("fitted") or {}
    rolls = discover_rolls([roll_id] if roll_id else config["rolls"])
    if not rolls:
        print("No roll images for this session — drop them into fixtures/rolls/.")
        return None
    roll = rolls[0]

    def _detect_all(cfg):
        items = [(p.stem, str(p)) for p in roll["images"]]

        def one(it):
            stem, path = it
            try:
                spots, rej, _err, _ls = dd.detect(path, cfg=cfg)
                return stem, {"detected": spots or [], "rejected": rej or []}
            except Exception as ex:   # noqa: BLE001
                print(f"  {stem}: detect failed ({ex})")
                return stem, {"detected": [], "rejected": []}

        res = ADAPTER.map_frames(one, items, ADAPTER.proc_workers(len(items)))
        return dict(res)

    print(f"Detecting {len(roll['images'])} frame(s) twice (fitted + live)…")
    fit = _detect_all(reg.to_tuning(fitted))
    live = _detect_all(dd.DEFAULT_TUNING)
    anns = _load_annotations(roll["fixtures"])     # the user's GT, per stem

    review_dir = Path(tempfile.mkdtemp(prefix="retouch_review_"))
    try:
        for p in roll["images"]:
            stem = p.stem
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except Exception:
                continue
            fp = fit.get(stem, {"detected": [], "rejected": []})
            lp = live.get(stem, fp)
            review = {"fitted": fp, "live": lp}
            gt = _gt_review_payload(anns.get(stem, {}), fp["detected"],
                                    min(int(w), int(h)))
            if gt is not None:                     # only annotated frames carry GT
                review["gt"] = gt
            data = {
                "stem": stem, "image_path": str(p),
                "width": int(w), "height": int(h),
                "detected": fp["detected"], "rejected": fp["rejected"],
                "review_kind": kind,
                "review": review,
            }
            (review_dir / f"{stem}_debug_spots.json").write_text(
                json.dumps(data, indent=2, cls=dd.NumpyEncoder))
        ui = TESTS_DIR.parent / "debug_ui.py"
        print(f"Opening dust debug UI for {session.name} "
              f"(R: FITTED [{kind}] -> GT -> live)")
        subprocess.run([sys.executable, str(ui), str(review_dir)])
    finally:
        shutil.rmtree(review_dir, ignore_errors=True)
    return session


# ---------------------------------------------------------------------------
# Adapter — the retouch surface the shared runner drives
# ---------------------------------------------------------------------------

class RetouchAdapter(runner.CalibrationAdapter):
    registry = reg
    schema = tuning                  # has .dump + .Tuning
    evaluators = EVALUATORS
    metric_name = METRIC_NAME
    description = "Recorded dust/stroke calibration sessions"

    @property
    def calib_dir(self):
        return CALIB_DIR

    def worst_key(self, kind):
        return _worst_key(kind)

    def headline(self, kind, agg):
        return _headline(kind, agg)

    def discover_rolls(self, roll_ids=None):
        return discover_rolls(roll_ids)

    def proc_workers(self, n):
        # Detection is full-res + memory-heavy; keep the per-frame fan-out modest
        # (the gate OOMs at cpu_count workers). Override via RETOUCH_CALIB_WORKERS.
        w = int(os.environ.get("RETOUCH_CALIB_WORKERS", "3"))
        return max(1, min(w, n))

    def map_frames(self, fn, items, workers, on_done=None):
        if workers <= 1:
            out = []
            for it in items:
                out.append(fn(it))
                if on_done:
                    on_done()
            return out
        out = [None] * len(items)
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="retouch-frame") as ex:
            futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
            for fut in as_completed(futs):
                out[futs[fut]] = fut.result()
                if on_done:
                    on_done()
        return out

    def review(self, session_dir, roll_id=None):
        return review_session(session_dir, roll_id)


ADAPTER = RetouchAdapter()

# Back-compat / self-test entry points (mirror auto_negadoctor's run_calibration).
optimize = runner.optimize
principal_components = runner.principal_components


def build_spec(kind, fit_params):
    return runner.build_spec(reg, kind, fit_params)


def run_session(config):
    return runner.run_session(ADAPTER, config)


if __name__ == "__main__":
    sys.exit(runner.run_main(ADAPTER))
