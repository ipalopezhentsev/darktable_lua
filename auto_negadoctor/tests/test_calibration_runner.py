"""Self-tests for the calibration runner (spec 05) — pure / fixture-light.

Covers the optimizers, the algorithm-independent metrics, the registry's
apply/restore (globals must be left untouched), and the session writer's schema.
The vignette exercises use an INLINE radial envelope + a fake evaluator, so they
need NO committed fixture and NO TIFFs.

Run: conda run -n autocrop python auto_negadoctor/tests/test_calibration_runner.py
Exit 0 = all pass.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

TESTS_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(TESTS_DIR.parent))
sys.path.insert(0, str(TESTS_DIR))

import auto_negadoctor as an          # noqa: E402
import run_quality_tests as rqt       # noqa: E402
import calibration_registry as reg    # noqa: E402
import run_calibration as rc          # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          f"{('  ' + str(detail)) if detail and not cond else ''}")
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------------

def test_optimizers_find_min():
    print("optimizers on a synthetic 1-D parabola (min at a=0.3):")
    spec = {"a": {"range": [0.0, 1.0], "grid_step": 0.05, "init": 0.0}}

    def f(x):
        return (x["a"] - 0.3) ** 2

    x, obj, n, tr = rc.optimize(f, spec, {"method": "none"})
    check("none returns init", abs(x["a"] - 0.0) < 1e-9 and n == 1)

    x, obj, n, tr = rc.optimize(f, spec, {"method": "grid"})
    check("grid lands on the grid min", abs(x["a"] - 0.3) <= 0.05 + 1e-9, x)

    x, obj, n, tr = rc.optimize(f, spec, {"method": "coordinate_descent",
                                      "epsilon": 1e-6, "max_iters": 40})
    check("coordinate_descent converges", abs(x["a"] - 0.3) < 0.01, x)

    x, obj, n, tr = rc.optimize(f, spec, {"method": "random_search",
                                      "n_trials": 4000, "seed": 1})
    check("random_search gets close", abs(x["a"] - 0.3) < 0.03, x)

    x, obj, n, tr = rc.optimize(f, spec, {"method": "cmaes", "max_iters": 40,
                                          "seed": 1, "verbose": False})
    check("cmaes converges", abs(x["a"] - 0.3) < 0.01, x)

    x, obj, n, tr = rc.optimize(f, spec, {"method": "spsa", "max_iters": 300,
                                          "seed": 1, "a": 0.3, "c": 0.1,
                                          "verbose": False})
    check("spsa converges", abs(x["a"] - 0.3) < 0.02, x)


def test_coordinate_descent_two_params():
    print("coordinate descent on a 2-D bowl (min at (0.6, 0.2)):")
    spec = {"a": {"range": [0.0, 1.0], "init": 0.0},
            "b": {"range": [0.0, 1.0], "init": 1.0}}

    def f(x):
        return (x["a"] - 0.6) ** 2 + (x["b"] - 0.2) ** 2

    x, obj, n, tr = rc.optimize(f, spec, {"method": "coordinate_descent",
                                      "epsilon": 1e-7, "max_iters": 60})
    check("both params converge", abs(x["a"] - 0.6) < 0.01
          and abs(x["b"] - 0.2) < 0.01, x)


def test_convergence_trace():
    print("optimizers record a uniform convergence trace:")
    spec = {"a": {"range": [0.0, 1.0], "grid_step": 0.1, "init": 0.0}}

    def f(x):
        return (x["a"] - 0.3) ** 2

    _, _, n, tr = rc.optimize(f, spec, {"method": "none"})
    check("none trace: 1 trial + 1 improvement",
          tr["method"] == "none" and tr["trials"] == 1
          and len(tr["improvements"]) == 1, tr)

    _, _, n, tr = rc.optimize(f, spec, {"method": "coordinate_descent",
                                        "epsilon": 1e-6, "max_iters": 40})
    check("cd trace: per-cycle + improvements + trials match",
          tr["cycles"] and tr["improvements"] and tr["trials"] == n
          and "converged_early" in tr, tr.keys())
    check("cd trace: best-so-far is monotonically decreasing",
          all(tr["improvements"][i]["objective"] >= tr["improvements"][i + 1]["objective"]
              for i in range(len(tr["improvements"]) - 1)))

    _, _, n, tr = rc.optimize(f, spec, {"method": "grid"})
    check("grid trace: grid_total recorded", tr.get("grid_total") == 11, tr)

    _, _, n, tr = rc.optimize(f, spec, {"method": "random_search",
                                        "n_trials": 50, "seed": 3})
    check("random trace: n_trials + seed", tr.get("n_trials") == 50
          and tr.get("seed") == 3, tr)


def test_contributions():
    print("coordinate descent credits the dominant param:")
    spec = {"a": {"range": [0.0, 1.0], "init": 0.0},
            "b": {"range": [0.0, 1.0], "init": 0.0}}

    def f(x):                      # 'a' has 100x the leverage of 'b'
        return 10.0 * (x["a"] - 0.7) ** 2 + 0.1 * (x["b"] - 0.3) ** 2

    _, _, _, tr = rc.optimize(f, spec, {"method": "coordinate_descent",
                                        "epsilon": 1e-7, "max_iters": 40})
    contribs = tr.get("contributions")
    check("contributions present + sorted", bool(contribs)
          and contribs[0]["param"] == "a", contribs)
    check("dominant param owns most of the objective drop",
          contribs[0]["pct"] > 80.0, contribs)


def test_principal_components():
    print("principal components recover coupled directions:")
    # f = (a+b-1)^2 + 0.01*(a-b)^2 — min at a=b=0.5; the a+b direction is stiff,
    # the a-b direction is soft. PCA should rank them so.
    spec = {"a": {"range": [0.0, 1.0], "grid_step": 0.1},
            "b": {"range": [0.0, 1.0], "grid_step": 0.1}}

    def evaluate(x):
        return {"objective": (x["a"] + x["b"] - 1.0) ** 2
                + 0.01 * (x["a"] - x["b"]) ** 2}

    pca = rc.principal_components(evaluate, spec, {"a": 0.5, "b": 0.5})
    comps = pca["components"]
    check("two components returned", len(comps) == 2, len(comps))
    pc1 = {ld["param"]: ld["weight"] for ld in comps[0]["top_loadings"]}
    pc2 = {ld["param"]: ld["weight"] for ld in comps[1]["top_loadings"]}
    check("PC1 (stiff) couples a,b with the SAME sign",
          pc1.get("a", 0) * pc1.get("b", 0) > 0, pc1)
    check("PC2 (soft) couples a,b with OPPOSITE sign",
          pc2.get("a", 0) * pc2.get("b", 0) < 0, pc2)
    check("PC1 eigenvalue >> PC2 eigenvalue",
          abs(comps[0]["eigenvalue"]) > 5 * abs(comps[1]["eigenvalue"]),
          (comps[0]["eigenvalue"], comps[1]["eigenvalue"]))

    # Per-param sensitivity (the freeze signal): at the minimum the gradient is
    # ~0 in both axes; the symmetric f makes a and b equally curved, so their
    # one-step Δobjective is positive and equal.
    sens = {s["param"]: s for s in pca["sensitivity"]}
    check("sensitivity reports both params", set(sens) == {"a", "b"}, set(sens))
    check("gradient ~0 at the minimum",
          abs(sens["a"]["gradient"]) < 1e-9 and abs(sens["b"]["gradient"]) < 1e-9,
          (sens["a"]["gradient"], sens["b"]["gradient"]))
    check("symmetric params have equal, positive max Δobj",
          sens["a"]["max_step_delta"] > 0
          and abs(sens["a"]["max_step_delta"]
                  - sens["b"]["max_step_delta"]) < 1e-12,
          (sens["a"]["max_step_delta"], sens["b"]["max_step_delta"]))


def test_crop_overtrim_metric():
    print("crop over-trim metric:")
    # delegate to the shared self-test in run_quality_tests
    try:
        rqt.selftest_crop_overtrim()
        check("run_quality_tests crop-overtrim self-test", True)
    except AssertionError as e:
        check("run_quality_tests crop-overtrim self-test", False, e)


def test_crop_fields_split_identical():
    """The crop speedup: detect_content_crop must equal _crop_decide(_crop_fields)
    — and _crop_decide must honour a per-trial cfg override (proving the runner's
    precomputed-fields path stays exact). No TIFFs."""
    print("crop fields/decide split equivalence (no TIFFs):")
    import numpy as np
    rng = np.random.default_rng(0)
    h, w = 240, 320
    lin = np.full((h, w, 3), 0.5, dtype=np.float32)
    lin += rng.normal(0, 0.01, lin.shape).astype(np.float32)
    lin[:5] = 0.001; lin[-4:] = 0.001            # thin dark holder border (so the
    lin[:, :6] = 0.001; lin[:, -5:] = 0.001      # trim is NOT capped at the max)
    dmin = np.array([0.9, 0.6, 0.4], dtype=np.float32)

    full = an.detect_content_crop(lin, dmin)
    fields = an._crop_fields(lin, dmin)
    split = an._crop_decide(fields)
    check("detect_content_crop == _crop_decide(_crop_fields)", full == split,
          (full, split))

    # The runner re-runs ONLY _crop_decide on the precomputed fields per trial,
    # with a per-trial cfg — _crop_decide MUST read constants from that cfg (not
    # bake them into fields). BORDER_MAX_FRAC sets the trim cap, so a value BELOW
    # the default tightens the result even on this saturated synthetic frame
    # (which trims to the cap) — a clean reactivity probe.
    tight_frac = 0.05
    cfg = reg.to_tuning({"BORDER_MAX_FRAC": tight_frac})
    tighter = an._crop_decide(fields, cfg)            # SAME precomputed fields
    check("_crop_decide honours a per-trial cfg override on cached fields",
          tighter != split and max(tighter) <= int(max(h, w) * tight_frac),
          (split, tighter))


def test_registry_apply_restore():
    print("registry apply/restore leaves globals untouched:")
    before = an.PRINT_GAMMA
    snap = reg.snapshot(["PRINT_GAMMA"])
    reg.apply({"PRINT_GAMMA": before + 1.234})
    changed = an.PRINT_GAMMA
    reg.restore(snap)
    after = an.PRINT_GAMMA
    check("apply changed the global", abs(changed - (before + 1.234)) < 1e-9)
    check("restore put it back", abs(after - before) < 1e-12)


def test_registry_tuple_and_int():
    print("registry handles tuple-element + integer constants:")
    snap = reg.snapshot(["WB_HIGH_PRIOR[0]", "WB_HIGH_PRIOR[1]", "VIG_BINS"])
    orig_tuple = an.WB_HIGH_PRIOR
    orig_bins = an.VIG_BINS
    try:
        reg.apply({"WB_HIGH_PRIOR[0]": 2.0})
        check("indexed write set element 0", abs(an.WB_HIGH_PRIOR[0] - 2.0) < 1e-9,
              an.WB_HIGH_PRIOR)
        check("other tuple elements intact",
              an.WB_HIGH_PRIOR[1] == orig_tuple[1]
              and an.WB_HIGH_PRIOR[2] == orig_tuple[2], an.WB_HIGH_PRIOR)
        check("current() reads the element",
              abs(reg.current("WB_HIGH_PRIOR[0]") - 2.0) < 1e-9)
        # two indices of the same tuple in one apply
        reg.apply({"WB_HIGH_PRIOR[1]": 1.5, "WB_HIGH_PRIOR[2]": 0.9})
        check("multi-index apply rebuilt the tuple once",
              abs(an.WB_HIGH_PRIOR[1] - 1.5) < 1e-9
              and abs(an.WB_HIGH_PRIOR[2] - 0.9) < 1e-9, an.WB_HIGH_PRIOR)
        reg.apply({"VIG_BINS": 33.7})
        check("int constant coerced to int", an.VIG_BINS == 34
              and isinstance(an.VIG_BINS, int), an.VIG_BINS)
    finally:
        reg.restore(snap)
    check("restore put the whole tuple back", an.WB_HIGH_PRIOR == orig_tuple,
          an.WB_HIGH_PRIOR)
    check("restore put the int back", an.VIG_BINS == orig_bins)


def test_build_spec_rejects_unknown():
    print("build_spec rejects a non-fittable param:")
    try:
        rc.build_spec("inversion", {"NOT_A_PARAM": {}})
        check("raised", False, "no error")
    except ValueError:
        check("raised", True)
    # and accepts a real one, seeding init from the live value
    spec = rc.build_spec("inversion", {"PRINT_GAMMA": {}})
    check("init seeded from source", abs(spec["PRINT_GAMMA"]["init"]
                                         - an.PRINT_GAMMA) < 1e-12)
    # "all" expands to the whole catalog for the kind
    all_spec = rc.build_spec("inversion", "all")
    check("'all' expands to the full inversion catalog",
          len(all_spec) == len(reg.fittable("inversion")) and len(all_spec) > 20,
          len(all_spec))
    check("'all' includes the white-balance levers",
          "WB_LOW_DESAT" in all_spec and "WB_HIGH_PRIOR[0]" in all_spec)


# A real centre-brightest radial vignette envelope (roll 2512-2601-1, captured
# from estimate_vignette) — INLINE so these tests need NO committed fixture and
# NO TIFFs. fit_vignette_profile turns it into a valid (strength 0.525) fit.
GOOD_R = [0.0156, 0.0469, 0.0781, 0.1094, 0.1406, 0.1719, 0.2031, 0.2344, 0.2656,
          0.2969, 0.3281, 0.3594, 0.3906, 0.4219, 0.4531, 0.4844, 0.5156, 0.5469,
          0.5781, 0.6094, 0.6406, 0.6719, 0.7031, 0.7344, 0.7656, 0.7969, 0.8281,
          0.8594, 0.8906, 0.9219, 0.9531]
GOOD_E = [88.5672, 87.9861, 87.4434, 87.1555, 86.5714, 86.2633, 85.9865, 85.5534,
          84.889, 84.3157, 83.4269, 82.6782, 82.0233, 81.2566, 80.3823, 79.6461,
          78.7677, 78.2133, 77.4353, 76.4695, 75.6695, 74.5866, 73.1663, 71.6903,
          70.7212, 83.373, 83.3272, 79.5558, 80.1409, 106.6574, 108.9449]
GOOD_ENV = {"r": GOOD_R, "e": GOOD_E, "used": 37, "reason": None}
FLAT_ENV = {"r": [(k + 0.5) / 12 for k in range(12)], "e": [100.0] * 12,
            "used": 10, "reason": None}


def test_vignette_objective_penalizes_rejection():
    print("vignette objective: a rejected roll dominates:")
    snap = reg.snapshot(list(reg.fittable("vignette")))
    try:
        res = rc._vignette_result([rc._envelope_record("REF", GOOD_ENV),
                                   rc._envelope_record("FLAT", FLAT_ENV)])
        good_only = rc._vignette_result([rc._envelope_record("REF", GOOD_ENV)])
    finally:
        reg.restore(snap)
    check("objective >= BIG_PENALTY (a roll rejected)",
          res["objective"] >= rc.BIG_PENALTY, res["objective"])
    check("aggregate counts the rejected (flat) roll",
          res["aggregate"]["rejected_rolls"] == 1, res["aggregate"])
    check("good roll alone is not rejected",
          good_only["aggregate"]["rejected_rolls"] == 0, good_only["aggregate"])


def test_session_end_to_end_vignette():
    print("end-to-end vignette session (method none; fake evaluator, no TIFFs):")
    before = reg.snapshot(list(reg.fittable("vignette")))

    def fake_disc(roll_ids=None):
        return [{"id": "SYNTH", "images": ["x.tif"], "exif": {}, "fixtures": []}]

    def fake_factory(rolls, tol, fit_params=()):
        def evaluate(overrides):
            reg.apply(overrides)
            return rc._vignette_result([rc._envelope_record("SYNTH", GOOD_ENV)])
        return evaluate, {"SYNTH": 0.0}

    with tempfile.TemporaryDirectory() as tmp:
        o_dir = rc.CALIB_DIR
        o_disc = rc.discover_image_rolls
        o_fac = rc.EVALUATORS["vignette"]
        rc.CALIB_DIR = Path(tmp) / "calibrations"
        rc.discover_image_rolls = fake_disc
        rc.EVALUATORS["vignette"] = fake_factory
        try:
            config = {"kind": "vignette", "rolls": ["SYNTH"],
                      "fit": {"method": "coordinate_descent",
                              "epsilon": 0.001, "max_iters": 3,
                              "params": {"VIG_MIN_STRENGTH": {},
                                         "VIG_TAIL_CUT_REL": {}}}}
            session = rc.run_session(config)
        finally:
            rc.CALIB_DIR = o_dir
            rc.discover_image_rolls = o_disc
            rc.EVALUATORS["vignette"] = o_fac
        check("session folder created", session is not None and session.is_dir())
        if not session:
            return
        for fn in ("config.json", "results.json", "report.md",
                   "fitted_params.json"):
            check(f"wrote {fn}", (session / fn).is_file())
        results = json.loads((session / "results.json").read_text())
        for key in ("wall_seconds", "trial_count", "objective_initial",
                    "objective_final", "trace", "fitted", "aggregate", "per_roll",
                    "per_frame"):
            check(f"results has {key}", key in results)
        check("trace records the method + improvement curve",
              results["trace"].get("method") == "coordinate_descent"
              and results["trace"].get("improvements"), results["trace"])
        check("wall time recorded", isinstance(results["wall_seconds"], (int, float)))
        check("objective finite (roll not rejected)",
              results["objective_final"] < rc.BIG_PENALTY,
              results["objective_final"])
        report = (session / "report.md").read_text()
        check("report lists the method's params (epsilon/max_iters)",
              "method params:" in report and "epsilon=" in report
              and "max_iters=" in report, report[:400])
        idx = session.parent / "INDEX_vignette.md"
        idx_txt = idx.read_text()
        check("INDEX_vignette.md created + has a row",
              idx.is_file() and "report.md" in idx_txt)
        check("INDEX row carries method params, not just the name",
              "coordinate_descent(" in idx_txt and "epsilon=" in idx_txt,
              idx_txt)
    after = reg.snapshot(list(reg.fittable("vignette")))
    check("globals restored after the session", before == after,
          (before, after))


def test_to_tuning_builds_cfg():
    """reg.to_tuning builds an immutable Tuning from overrides: scalar, int
    coercion, indexed tuple element, and {} == DEFAULT_TUNING — the thread-safe
    replacement for the global-mutating apply()."""
    print("registry.to_tuning builds a per-trial cfg:")
    base = an.DEFAULT_TUNING
    check("to_tuning({}) == DEFAULT_TUNING", reg.to_tuning({}) == base)
    cfg = reg.to_tuning({"PRINT_GAMMA": 9.0, "VIG_BINS": 40.4,
                         "WB_HIGH_PRIOR[1]": 1.5})
    check("scalar override applied", cfg.PRINT_GAMMA == 9.0)
    check("int constant coerced to int", cfg.VIG_BINS == 40
          and isinstance(cfg.VIG_BINS, int), cfg.VIG_BINS)
    check("indexed tuple element patched, siblings kept",
          cfg.WB_HIGH_PRIOR[1] == 1.5
          and cfg.WB_HIGH_PRIOR[0] == base.WB_HIGH_PRIOR[0], cfg.WB_HIGH_PRIOR)
    check("DEFAULT_TUNING is immutable / untouched",
          an.DEFAULT_TUNING.PRINT_GAMMA == base.PRINT_GAMMA)


def test_tuning_presets():
    """The externalized tuning system: the bundled default preset reproduces
    DEFAULT_TUNING exactly (value AND type), the schema validates keys, and a
    fitted Tuning round-trips through dump/load — so 'adopt fitted values' is a
    file drop, not a source edit."""
    import os
    import tempfile
    import tuning
    print("tuning presets (external JSON + Python schema):")
    # bundled default == the live DEFAULT_TUNING, exactly
    dflt = tuning.load("default")
    mism = [n for n in tuning.FIELDS
            if getattr(dflt, n) != getattr(an.DEFAULT_TUNING, n)
            or type(getattr(dflt, n)) is not type(getattr(an.DEFAULT_TUNING, n))]
    check("default preset == DEFAULT_TUNING (value+type)", not mism, mism[:5])
    check("Tuning fields == registry _TUNABLE_NAMES",
          tuple(tuning.FIELDS) == an._TUNABLE_NAMES)
    check("an._tuning is the top-level tuning module", an._tuning is tuning)
    # schema validation: missing and unknown keys both rejected
    for bad, label in [({}, "missing"),
                       ({**tuning.to_mapping(dflt), "BOGUS": 1}, "unknown")]:
        try:
            tuning.from_mapping(bad)
            check(f"rejects {label} key", False)
        except ValueError:
            check(f"rejects {label} key", True)
    # a fitted cfg (indexed + int + scalar overrides) round-trips via dump/load
    cfg = reg.to_tuning({"PRINT_GAMMA": 4.25, "WB_HIGH_PRIOR[0]": 2.0,
                         "PRINT_TUNE_ITERS": 14.0})
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "fp.json")
        tuning.dump(cfg, p)
        back = tuning.load(p)
    check("fitted preset dump/load is identity", back == cfg)
    check("int field stayed int through JSON", isinstance(back.PRINT_TUNE_ITERS, int))
    check("tuple field stayed tuple through JSON",
          isinstance(back.WB_HIGH_PRIOR, tuple) and back.WB_HIGH_PRIOR[0] == 2.0)
    # nested layout: dump writes the GROUPS tree; GROUPS partitions FIELDS
    nested = tuning.to_nested(dflt)
    leaves = [n for kind in nested.values() for sub in kind.values() for n in sub]
    check("to_nested partitions FIELDS exactly", sorted(leaves) == sorted(tuning.FIELDS))
    check("top level is the calibration kinds",
          list(nested) == ["crop", "inversion", "vignette"], list(nested))
    check("dump writes nested (kinds at top level)", set(json.load(
        open(tuning.preset_path("default")))) == {"crop", "inversion", "vignette"})
    # an OLD flat preset still loads (back-compat)
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "flat.json")
        json.dump(tuning.to_mapping(dflt), open(fp, "w"))
        check("flat preset still loads", tuning.load(fp) == dflt)


def test_random_search_parallel_equivalence():
    """Parallel random_search (workers>1) must give a BIT-IDENTICAL result to the
    serial run — same best, same point, same convergence curve — because points
    are sampled up front (seeded) and recorded in order. Enabled by the cfg
    refactor (trials no longer share mutated globals)."""
    print("parallel random_search == serial (deterministic):")
    spec = {"a": {"range": [0.0, 1.0], "grid_step": 0.1, "init": 0.5},
            "b": {"range": [0.0, 1.0], "grid_step": 0.1, "init": 0.5}}

    def f(x):
        return (x["a"] - 0.3) ** 2 + (x["b"] - 0.7) ** 2

    base = {"method": "random_search", "n_trials": 60, "seed": 7, "verbose": False}
    bx1, bo1, n1, tr1 = rc.optimize(f, spec, dict(base, workers=1))
    bx4, bo4, n4, tr4 = rc.optimize(f, spec, dict(base, workers=4))
    check("same best objective", bo1 == bo4, (bo1, bo4))
    check("same best point", bx1 == bx4, (bx1, bx4))
    check("same convergence curve",
          tr1["improvements"] == tr4["improvements"])
    check("trial count == n_trials", n1 == 60 and n4 == 60, (n1, n4))
    check("trace records the worker count", tr4.get("workers") == 4,
          tr4.get("workers"))


def test_cmaes_parallel_equivalence():
    """Parallel cmaes (workers>1) must give a BIT-IDENTICAL result to the serial run:
    each generation's candidates are independent, evaluated in any order but recorded
    (and told back to the optimizer) in fixed index order, so the seeded run is
    deterministic regardless of which worker finishes first."""
    print("parallel cmaes == serial (deterministic):")
    spec = {"a": {"range": [0.0, 1.0], "grid_step": 0.1, "init": 0.5},
            "b": {"range": [0.0, 1.0], "grid_step": 0.1, "init": 0.5}}

    def f(x):
        return (x["a"] - 0.3) ** 2 + (x["b"] - 0.7) ** 2

    base = {"method": "cmaes", "max_iters": 30, "seed": 7, "verbose": False}
    bx1, bo1, n1, tr1 = rc.optimize(f, spec, dict(base, workers=1))
    bx4, bo4, n4, tr4 = rc.optimize(f, spec, dict(base, workers=4))
    check("same best objective", bo1 == bo4, (bo1, bo4))
    check("same best point", bx1 == bx4, (bx1, bx4))
    check("same convergence curve",
          tr1["improvements"] == tr4["improvements"])
    check("cmaes converged near (0.3, 0.7)",
          abs(bx1["a"] - 0.3) < 0.02 and abs(bx1["b"] - 0.7) < 0.02, bx1)
    check("trace records the worker count", tr4.get("workers") == 4,
          tr4.get("workers"))


def test_spsa_parallel_equivalence():
    """Parallel spsa (workers>=2) must give a BIT-IDENTICAL result to the serial run:
    each iteration's plus/minus pair is independent, evaluated in any order but
    recorded in fixed (plus, minus) order, so the seeded run is deterministic."""
    print("parallel spsa == serial (deterministic):")
    spec = {"a": {"range": [0.0, 1.0], "grid_step": 0.1, "init": 0.5},
            "b": {"range": [0.0, 1.0], "grid_step": 0.1, "init": 0.5}}

    def f(x):
        return (x["a"] - 0.3) ** 2 + (x["b"] - 0.7) ** 2

    base = {"method": "spsa", "max_iters": 200, "seed": 7, "a": 0.3, "c": 0.1,
            "verbose": False}
    bx1, bo1, n1, tr1 = rc.optimize(f, spec, dict(base, workers=1))
    bx2, bo2, n2, tr2 = rc.optimize(f, spec, dict(base, workers=4))
    check("same best objective", bo1 == bo2, (bo1, bo2))
    check("same best point", bx1 == bx2, (bx1, bx2))
    check("same convergence curve",
          tr1["improvements"] == tr2["improvements"])
    check("spsa converged near (0.3, 0.7)",
          abs(bx1["a"] - 0.3) < 0.03 and abs(bx1["b"] - 0.7) < 0.03, bx1)
    check("trace records the worker count", tr2.get("workers") == 4,
          tr2.get("workers"))


def main():
    test_optimizers_find_min()
    test_coordinate_descent_two_params()
    test_convergence_trace()
    test_contributions()
    test_principal_components()
    test_crop_overtrim_metric()
    test_crop_fields_split_identical()
    test_to_tuning_builds_cfg()
    test_tuning_presets()
    test_random_search_parallel_equivalence()
    test_cmaes_parallel_equivalence()
    test_spsa_parallel_equivalence()
    test_registry_apply_restore()
    test_registry_tuple_and_int()
    test_build_spec_rejects_unknown()
    test_vignette_objective_penalizes_rejection()
    test_session_end_to_end_vignette()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {', '.join(FAILURES)}")
        sys.exit(1)
    print("ALL CALIBRATION-RUNNER TESTS PASSED")


if __name__ == "__main__":
    main()
