"""Self-tests for the auto_retouch calibration layer (spec: shared calibration base).

Fast + image-free: exercises the shared optimizers, the registry's cfg builder +
apply/restore, the dust/stroke SCORE helpers (the "test for the test" — a non-trivial
checker gets its own check), and an end-to-end recorded `dust` session driven by an
INLINE fake evaluator (no real detection). Run:

    conda run -n autocrop python auto_retouch/tests/test_calibration_runner.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent.parent))
sys.path.insert(0, str(TESTS_DIR.parent))
sys.path.insert(0, str(TESTS_DIR))

import detect_dust as dd            # noqa: E402
import run_quality_tests as rqt     # noqa: E402
import calibration_registry as reg  # noqa: E402
import run_calibration as rc        # noqa: E402

_fails = []


def check(name, cond, *info):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f"  {info}" if info and not cond else ""))
    if not cond:
        _fails.append(name)


# ---------------------------------------------------------------------------
def test_optimizers_find_minimum():
    print("optimizers find a synthetic minimum:")
    # objective minimized at x=2.0
    def f(o):
        return (o["X"] - 2.0) ** 2
    spec = {"X": {"range": [0.0, 4.0], "grid_step": 0.25, "init": 0.0}}
    for method, fit in [("grid", {"method": "grid"}),
                        ("coordinate_descent",
                         {"method": "coordinate_descent", "epsilon": 1e-6,
                          "max_iters": 30}),
                        ("random_search",
                         {"method": "random_search", "n_trials": 200, "seed": 0})]:
        x, obj, n, tr = rc.optimize(f, spec, fit)
        check(f"{method} ~= 2.0", abs(x["X"] - 2.0) < 0.3, x["X"])


def test_build_spec():
    print("build_spec:")
    spec = rc.build_spec("dust", {"NOISE_THRESHOLD_MULTIPLIER": {}})
    check("init = live value", spec["NOISE_THRESHOLD_MULTIPLIER"]["init"]
          == float(dd.NOISE_THRESHOLD_MULTIPLIER))
    try:
        rc.build_spec("dust", {"NOT_A_PARAM": {}})
        check("rejects unknown param", False)
    except ValueError:
        check("rejects unknown param", True)
    allspec = rc.build_spec("dust", "all")
    check("'all' = every dust constant", len(allspec) == len(reg.REGISTRY["dust"]))


def test_to_tuning_and_no_mutation():
    print("to_tuning + apply/restore leave module globals untouched:")
    before = dd.MAX_SPOTS
    cfg = reg.to_tuning({"MAX_SPOTS": 42})
    check("to_tuning builds an override", cfg.MAX_SPOTS == 42)
    check("global NOT mutated by to_tuning", dd.MAX_SPOTS == before)
    snap = reg.snapshot(["MAX_SPOTS"])
    reg.apply({"MAX_SPOTS": 7})
    check("apply mutates the global", dd.MAX_SPOTS == 7)
    reg.restore(snap)
    check("restore puts it back", dd.MAX_SPOTS == before)


def test_dust_score_helper():
    print("dust_score_per_frame (test for the test):")
    # detected dots near (100,100) and (300,300)
    dots = [{"cx": 100, "cy": 100}, {"cx": 300, "cy": 300}]
    ann = {"false_positives": [{"cx": 102, "cy": 99}],   # matches dot #1 -> 1 FP
           "missed_dust": [{"cx": 100, "cy": 100},        # covered -> not missed
                           {"cx": 900, "cy": 900}]}        # uncovered -> 1 missed
    sc = rqt.dust_score_per_frame(dots, ann)
    check("n_fp counts reproduced FP", sc["n_fp"] == 1, sc)
    check("n_missed counts uncovered", sc["n_missed"] == 1, sc)
    # empty annotations -> zero signal
    sc0 = rqt.dust_score_per_frame(dots, {"false_positives": [], "missed_dust": []})
    check("empty annotations -> 0/0", sc0["n_fp"] == 0 and sc0["n_missed"] == 0)


def test_stroke_score_helper():
    print("stroke_score_per_frame (test for the test):")
    strokes = [{"cx": 200, "cy": 200, "length_px": 100.0,
                "path": [[150, 200], [250, 200]]}]
    # a missed stroke far away -> uncovered; one near -> covered
    ann_miss = {"missed_strokes": [{"path": [[600, 600], [700, 600]],
                                    "stroke_width_px": 3}]}
    sc = rqt.stroke_score_per_frame(strokes, ann_miss)
    check("uncovered missed stroke counted", sc["n_missed"] == 1, sc)
    ann_cov = {"missed_strokes": [{"path": [[150, 205], [250, 205]],
                                   "stroke_width_px": 3}]}
    sc2 = rqt.stroke_score_per_frame(strokes, ann_cov)
    check("covered missed stroke not counted", sc2["n_missed"] == 0, sc2)
    ann_fp = {"false_positives": [{"cx": 201, "cy": 199}]}
    sc3 = rqt.stroke_score_per_frame(strokes, ann_fp)
    check("stroke on a FP point counted", sc3["n_fp"] == 1, sc3)


def test_end_to_end_session():
    print("end-to-end dust session (method none; inline fake evaluator, no detection):")
    # fake evaluator: objective is a closed form of the override (no images needed)
    def fake_factory(rolls, tolerances, fit_params=()):
        def evaluate(overrides):
            x = overrides.get("NOISE_THRESHOLD_MULTIPLIER", 3.0)
            per_frame = [{"roll": "FAKE", "stem": "DSC_0001",
                          "score": abs(x - 2.5), "n_fp": 1, "n_missed": 0}]
            return {"objective": abs(x - 2.5),
                    "per_frame": per_frame,
                    "per_roll": rc.runner.per_roll_summary(per_frame, "score"),
                    "aggregate": {"objective_weighted": abs(x - 2.5),
                                  "total_fp": 1, "total_missed": 0,
                                  "w_fp": 3.0, "w_miss": 1.0, "n_frames": 1}}
        return evaluate, {"FAKE": 0.0}

    with tempfile.TemporaryDirectory() as tmp:
        o_dir, o_disc, o_fac = rc.CALIB_DIR, rc.discover_rolls, rc.EVALUATORS["dust"]
        before_glob = dd.NOISE_THRESHOLD_MULTIPLIER
        try:
            rc.CALIB_DIR = Path(tmp) / "calibrations"
            rc.discover_rolls = lambda ids=None: [{"id": "FAKE", "dir": Path(tmp),
                                                   "images": ["x.jpg"], "fixtures": ["a"]}]
            rc.EVALUATORS["dust"] = fake_factory
            cfg = {"kind": "dust", "rolls": [],
                   "fit": {"method": "none", "tolerances": {},
                           "params": {"NOISE_THRESHOLD_MULTIPLIER": {}}}}
            session = rc.run_session(json.loads(json.dumps(cfg)))
        finally:
            rc.CALIB_DIR, rc.discover_rolls = o_dir, o_disc
            rc.EVALUATORS["dust"] = o_fac
        check("session folder created", session and Path(session).is_dir())
        res = json.loads((Path(session) / "results.json").read_text())
        for k in ("wall_seconds", "trial_count", "objective_initial",
                  "objective_final", "trace", "fitted", "aggregate",
                  "per_roll", "per_frame"):
            check(f"results has {k}", k in res)
        for fn in ("config.json", "fitted_params.json", "fitted_preset.json",
                   "report.md"):
            check(f"wrote {fn}", (Path(session) / fn).exists())
        # fitted_preset is a complete, reloadable preset
        fp = rc.tuning.load(str(Path(session) / "fitted_preset.json"))
        check("fitted_preset reloads as a full Tuning",
              len(fp._fields) == len(rc.tuning.FIELDS))
        check("globals restored after the session",
              dd.NOISE_THRESHOLD_MULTIPLIER == before_glob)
        check("INDEX_dust.md created",
              (Path(session).parent / "INDEX_dust.md").is_file())


if __name__ == "__main__":
    test_optimizers_find_minimum()
    test_build_spec()
    test_to_tuning_and_no_mutation()
    test_dust_score_helper()
    test_stroke_score_helper()
    test_end_to_end_session()
    print()
    if _fails:
        print(f"FAILED: {_fails}")
        sys.exit(1)
    print("ALL RETOUCH CALIBRATION-RUNNER TESTS PASSED")
