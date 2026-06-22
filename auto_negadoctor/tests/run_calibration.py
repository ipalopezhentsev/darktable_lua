"""Reproducible, RECORDED calibration sessions (spec 05).

The algorithm has been tuned by ad-hoc chat sessions whose convergence numbers
went only to stdout and were lost. This runner makes each calibration a
persisted folder under tests/calibrations/ that records its INPUTS (which rolls,
which constants it fits, the fitting algorithm + its hyperparameters, the
convergence epsilon, the tolerances) and its OUTPUTS (an algorithm-INDEPENDENT
closeness metric, reported aggregate + per-roll + per-frame, plus wall time), so
sessions are comparable over time.

Three KINDS, each with its own metric and its own history index
(INDEX_<kind>.md) — they are never mixed in one table:

  crop       total OVER-TRIM (fraction of frame area of content the detected
             crop removed inside the user's hand-drawn rect); containment
             (never extend outside the user rect) is a HARD constraint.
  inversion  median histogram EMD between the algorithm's RENDER and the user's
             GT-param render over the content crop (picture-vs-picture, the
             param-invariant 'look'); rendered hard-clip beyond the budget is a
             HARD constraint.
  vignette   per roll: did fit_vignette_profile produce a valid fit? + its
             residual. A rejected roll dominates the objective (this is the
             'another roll fails to auto-find vignette' guard).

TUNING ORDER: tune `crop` and `vignette` FIRST, then `inversion` — inversion
depends on both (its pickers/wb/print tune run INSIDE the content crop and on
vignette-corrected data), so tuning it before they are settled chases a moving
target. Re-run inversion whenever crop or vignette is re-tuned. (See
calibrations/README.md "Tuning order".)

RECORD-ONLY: the runner never edits auto_negadoctor.py. The best constants land
in the session's fitted_params.json; the user adopts the good ones by hand.

Run (evaluate the current algorithm, no search):
  conda run -n autocrop python auto_negadoctor/tests/run_calibration.py \
      --config auto_negadoctor/tests/calibrations/configs/inversion_default.json \
      --method none
Run a fit:
  ... --config .../inversion_default.json --method coordinate_descent
Override any method hyperparameter from the CLI (no need to edit the config;
CLI > config > optimizer default, and the effective values are recorded):
  coordinate_descent: --epsilon --max-iters --step-shrink --init-step --step-min
  random_search:      --n-trials --seed
  (also --method --rolls --pca). Per-param ranges/grid_steps stay in the config.
Review a finished session in the debug UI (inversion):
  ... --review <session_dir> [roll_id]
"""

import argparse
import json
import random
import shutil
import statistics
import subprocess
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent))
sys.path.insert(0, str(TESTS_DIR))

import auto_negadoctor as an          # noqa: E402
import nega_model as nm               # noqa: E402
import run_quality_tests as rqt       # noqa: E402
import calibration_registry as reg    # noqa: E402

CALIB_DIR = TESTS_DIR / "calibrations"
ROLLS_DIR = rqt.ROLLS_DIR
BIG_PENALTY = 1e6   # per rejected / containment-violating / clipping frame

METRIC_NAME = {
    "crop": "crop_overtrim_area_fraction (containment HARD)",
    "inversion": "histogram_emd_total (render vs GT-render, picture-vs-picture)",
    "vignette": "vignette_fit_rejected_count + median_residual",
}


def _median(vals):
    return statistics.median(vals) if vals else float("nan")


# ---------------------------------------------------------------------------
# Optimizers — native, dependency-free. Each minimizes objective(overrides).
# ---------------------------------------------------------------------------

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _dur(s):
    s = int(max(0, s))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _proc_cb(label):
    """A process_roll progress callback that prints the analysis percent at 25%
    milestones (process_roll ticks 3 stages × n frames)."""
    state = {"mark": -1}
    def cb(done, total):
        if not total:
            return
        pct = 100.0 * done / total
        if pct >= state["mark"] + 25 or done >= total:
            state["mark"] = pct - (pct % 25)
            print(f"    [{label}] analysis {pct:3.0f}%", flush=True)
    return cb


class _Tracker:
    """Wraps the objective to (1) log the best-so-far CURVE (one point each time
    the objective drops) — the uniform convergence trace — and (2) print live
    progress: job %% + ETA when the trial total is known (grid / random), else
    the eval count + the coordinate-descent context (cycle/param). Throttled to
    ~1s plus every improvement, so it's readable even on the slow full path."""
    def __init__(self, fn, verbose=True):
        self.fn = fn
        self.n = 0
        self.best = None
        self.improvements = []     # [{trial, objective}] at each new best
        self.total = None          # set by the method when the count is known
        self.ctx = ""              # set by coordinate descent (cycle/param)
        self.verbose = verbose
        self.t0 = time.perf_counter()
        self.last = 0.0

    def __call__(self, x):
        return self.record(self.fn(x))

    def record(self, v):
        """Bookkeeping for a PRE-COMPUTED objective value (so parallel trials can
        evaluate the pure objective concurrently, then feed results here in
        deterministic point order — same curve/counts as the serial run)."""
        self.n += 1
        improved = self.best is None or v < self.best - 1e-12
        if improved:
            self.best = v
            self.improvements.append({"trial": self.n, "objective": v})
        if self.verbose:
            now = time.perf_counter()
            if improved or now - self.last >= 1.0 or self.n == self.total:
                self.last = now
                el = now - self.t0
                if self.total:
                    eta = el / self.n * (self.total - self.n) if self.n else 0.0
                    head = (f"{100.0 * self.n / self.total:5.1f}% | eval "
                            f"{self.n}/{self.total} | ETA {_dur(eta)}")
                else:
                    head = f"eval {self.n} | {_dur(el)} elapsed"
                ctx = f" | {self.ctx}" if self.ctx else ""
                print(f"    [{head}]{ctx} obj={v:.6f} best={self.best:.6f}"
                      f"{'  *' if improved else ''}", flush=True)
        return v


def optimize(objective, spec, fit):
    """Return (best_overrides, best_objective, trial_count, trace). `trace` is a
    uniform convergence record: {method, trials, best_objective, improvements:
    [{trial, objective}], …method extras}."""
    method = fit.get("method", "none")
    tr = _Tracker(objective, verbose=fit.get("verbose", True))
    print(f"Fitting: {_method_desc(fit)}, {len(spec)} param(s) "
          f"[{', '.join(sorted(list(spec)))}]",
          flush=True)
    if method == "none":
        x = {n: s["init"] for n, s in spec.items()}
        tr.total = 1
        tr(x)
        extra = {}
    elif method == "grid":
        x, extra = _grid(tr, spec)
    elif method == "coordinate_descent":
        x, extra = _coord(tr, spec, fit)
    elif method == "random_search":
        x, extra = _random(tr, spec, fit)
    else:
        raise ValueError(f"unknown fit method {method!r}")
    trace = {"method": method, "trials": tr.n, "best_objective": tr.best,
             "improvements": tr.improvements, **extra}
    return x, tr.best, tr.n, trace


def _grid(objective, spec):
    axes = {}
    for n, s in spec.items():
        lo, hi, step = s["range"][0], s["range"][1], s["grid_step"]
        vals, v = [], lo
        while v <= hi + 1e-12:
            vals.append(round(v, 10))
            v += step
        axes[n] = vals or [lo]
    names = list(axes)
    total = 1
    for n in names:
        total *= len(axes[n])
    if total > 5000:
        print(f"WARNING: grid over {len(names)} param(s) = {total} trials — "
              "exhaustive grid explodes combinatorially. Use coordinate_descent "
              "or random_search for more than ~2-3 params.")
    if hasattr(objective, "total"):
        objective.total = total
    best, bx = None, None
    for combo in product(*(axes[n] for n in names)):
        x = dict(zip(names, combo))
        o = objective(x)
        if best is None or o < best:
            best, bx = o, x
    return bx, {"grid_total": total}


def _coord(objective, spec, fit):
    """Coordinate descent: per param a shrinking-step ± line search; repeat full
    cycles (sweeps over ALL params) until a cycle improves the objective by less
    than epsilon, or max_iters cycles. Records a per-cycle objective trace."""
    eps = float(fit.get("epsilon", 1e-4))
    max_iters = int(fit.get("max_iters", 20))
    x = {n: s["init"] for n, s in spec.items()}
    best = objective(x)
    cycles, converged_early = [], False
    # attribute every accepted objective drop to the param that produced it: the
    # "main contributors" to convergence, free from the optimizer's own moves.
    contrib = {n: {"improvement": 0.0, "moves": 0} for n in spec}
    verbose = getattr(objective, "verbose", False)
    nparams = len(spec)
    for cyc in range(max_iters):
        if verbose:
            print(f"  cycle {cyc + 1}/{max_iters} (best {best:.6f})", flush=True)
        before = best
        for pi, (n, s) in enumerate(spec.items(), 1):
            lo, hi = s["range"]
            span = hi - lo
            step = float(s.get("init_step", fit.get("init_step", span / 8.0)))
            step_min = float(s.get("step_min", fit.get("step_min", span / 256.0)))
            shrink = float(s.get("step_shrink", fit.get("step_shrink", 0.5)))
            while step >= step_min:
                moved = False
                for d in (step, -step):
                    cand = dict(x)
                    cand[n] = _clamp(x[n] + d, lo, hi)
                    if abs(cand[n] - x[n]) < 1e-12:
                        continue
                    if hasattr(objective, "ctx"):
                        objective.ctx = (f"cycle {cyc + 1}/{max_iters} "
                                         f"param {pi}/{nparams} {n}={cand[n]:.6g}")
                    o = objective(cand)
                    if o < best - 1e-12:
                        contrib[n]["improvement"] += best - o
                        contrib[n]["moves"] += 1
                        best, x, moved = o, cand, True
                        break
                if not moved:
                    step *= shrink
        cycles.append({"cycle": cyc + 1, "objective": best,
                       "improvement": before - best})
        if before - best < eps:
            converged_early = True
            break
    return x, {"cycles": cycles, "cycles_run": len(cycles),
               "converged_early": converged_early, "epsilon": eps,
               "contributions": _rank_contributions(contrib)}


def _random(objective, spec, fit):
    """Uniform random search. Trials are INDEPENDENT, so when workers>1 they run
    in parallel THREADS — possible only because each trial now builds its own
    immutable cfg (no shared global mutation). The points are sampled up front
    (seeded) and the results recorded in point order, so the parallel run is
    bit-identical to the serial one (same best, same convergence curve)."""
    n_trials = int(fit.get("n_trials", 50))
    seed = int(fit.get("seed", 0))
    rng = random.Random(seed)
    points = [{n: rng.uniform(s["range"][0], s["range"][1])
               for n, s in spec.items()} for _ in range(n_trials)]
    workers = int(fit.get("workers", 0)) or an._proc_workers(n_trials)
    if hasattr(objective, "total"):
        objective.total = n_trials
    verbose = getattr(objective, "verbose", True)

    if workers <= 1:
        best, bx = None, None
        for x in points:
            o = objective(x)             # tracker prints + records live
            if best is None or o < best:
                best, bx = o, x
        return bx, {"n_trials": n_trials, "seed": seed, "workers": 1}

    # PARALLEL: evaluate the pure objective concurrently (each worker runs its
    # frames serially via _trial_local.serial so we parallelize TRIALS, not
    # nested frame pools). Numpy releases the GIL during render/EMD, so this
    # scales. We print a live throughput/ETA line as trials complete, then record
    # the results in POINT order so the trace stays bit-identical to serial.
    objs = [None] * n_trials

    def _worker(i):
        _trial_local.serial = True
        try:
            return i, objective.fn(points[i])
        finally:
            _trial_local.serial = False

    done, best_live, t0, last = 0, None, time.perf_counter(), 0.0
    # Manual lifecycle (not `with`): the executor's __exit__ waits for ALL queued
    # trials on Ctrl-C; we instead cancel the pending ones and re-raise at once.
    ex = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="nega-trial")
    try:
        futs = [ex.submit(_worker, i) for i in range(n_trials)]
        for fut in as_completed(futs):
            i, v = fut.result()
            objs[i] = v
            done += 1
            if best_live is None or v < best_live:
                best_live = v
            if verbose:
                now = time.perf_counter()
                if now - last >= 1.0 or done == n_trials:
                    last = now
                    el = now - t0
                    eta = el / done * (n_trials - done) if done else 0.0
                    print(f"    [{100.0 * done / n_trials:5.1f}% | trial "
                          f"{done}/{n_trials} | {workers} workers | "
                          f"ETA {_dur(eta)}] best={best_live:.6f}", flush=True)
    except BaseException:
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown(wait=True)

    # ordered bookkeeping → deterministic trace (tracker's own prints suppressed,
    # we already showed live progress above)
    saved, objective.verbose = objective.verbose, False
    best, bx = None, None
    for x, o in zip(points, objs):
        objective.record(o)
        if best is None or o < best:
            best, bx = o, x
    objective.verbose = saved
    return bx, {"n_trials": n_trials, "seed": seed, "workers": workers}


def _rank_contributions(contrib):
    """Per-param objective drop credited by coordinate descent's accepted moves,
    sorted by contribution (the cheap 'main contributors' — free from the fit)."""
    total = sum(c["improvement"] for c in contrib.values()) or 1.0
    ranked = sorted(contrib.items(), key=lambda t: -t[1]["improvement"])
    return [{"param": n, "improvement": c["improvement"], "moves": c["moves"],
             "pct": 100.0 * c["improvement"] / total} for n, c in ranked]


def principal_components(evaluate, spec, point):
    """PRINCIPAL COMPONENTS of the objective's local curvature at `point`: the
    eigen-decomposition of the finite-difference Hessian, in NORMALIZED units
    (each param scaled by its grid_step so wildly different param ranges are
    comparable). Eigenvectors = principal directions in param space (they expose
    COUPLING — e.g. exposure↔wb↔black move together); eigenvalues rank how
    sharply the metric changes along each. Cost = 1 + 2N + N(N-1)/2 objective
    evals (O(N²)) — opt-in; restrict to a small param subset for the full path.

    `evaluate(overrides)->{objective}`. Returns
    {components:[{eigenvalue, variance_pct, top_loadings:[{param,weight}]}],
     n_evals, f0}."""
    names = list(spec)
    N = len(names)
    h = {n: float(spec[n].get("grid_step")
                  or (spec[n]["range"][1] - spec[n]["range"][0]) / 16.0)
         for n in names}
    lo = {n: spec[n]["range"][0] for n in names}
    hi = {n: spec[n]["range"][1] for n in names}

    def ev(deltas):
        c = dict(point)
        for n, d in deltas.items():
            c[n] = _clamp(point[n] + d, lo[n], hi[n])
        return evaluate(c)["objective"]

    f0 = ev({})
    fp = {n: ev({n: h[n]}) for n in names}
    fm = {n: ev({n: -h[n]}) for n in names}
    H = np.zeros((N, N))
    for i, n in enumerate(names):                       # diagonal curvature
        H[i, i] = (fp[n] - 2 * f0 + fm[n]) / (h[n] * h[n])
    for i in range(N):                                  # mixed partials (coupling)
        for j in range(i + 1, N):
            ni, nj = names[i], names[j]
            fpp = ev({ni: h[ni], nj: h[nj]})
            H[i, j] = H[j, i] = ((fpp - fp[ni] - fp[nj] + f0)
                                 / (h[ni] * h[nj]))
    # normalized Hessian (dimensionless: one grid_step = unit move per axis)
    hv = np.array([h[n] for n in names])
    Hn = H * np.outer(hv, hv)
    vals, vecs = np.linalg.eigh(Hn)
    order = np.argsort(-np.abs(vals))                   # sharpest curvature first
    tot = float(np.sum(np.abs(vals))) or 1.0
    comps = []
    for k in order:
        loads = sorted(((names[i], float(vecs[i, k])) for i in range(N)),
                       key=lambda t: -abs(t[1]))
        comps.append({
            "eigenvalue": float(vals[k]),
            "variance_pct": 100.0 * abs(float(vals[k])) / tot,
            "top_loadings": [{"param": p, "weight": round(w, 3)}
                             for p, w in loads[:6] if abs(w) > 0.05],
        })
    return {"f0": f0, "n_evals": 1 + 2 * N + N * (N - 1) // 2,
            "components": comps}


def build_spec(kind, fit_params):
    """Merge the registry defaults for `kind` with the config's per-param
    overrides; init = the LIVE source value (so search starts from production).

    `fit_params` may be the literal string "all" (or "*") to fit EVERY constant
    in the kind's catalog — the default. Otherwise it is a {name: override} map
    selecting a subset."""
    base = reg.fittable(kind)
    if fit_params in ("all", "*"):
        fit_params = {n: {} for n in base}
    spec = {}
    for name, override in (fit_params or {}).items():
        if name not in base:
            raise ValueError(
                f"{name!r} is not a fittable {kind} param "
                f"(known: {', '.join(sorted(base))})")
        s = dict(base[name])
        s.update(override or {})
        s["init"] = float(reg.current(name))
        spec[name] = s
    return spec


# ---------------------------------------------------------------------------
# Roll discovery
# ---------------------------------------------------------------------------

def discover_image_rolls(roll_ids=None):
    """Rolls with local TIFFs (every kind derives what it needs from them — crop
    & inversion render, vignette captures its envelope). Optionally filtered."""
    rolls = rqt.discover_rolls()
    if roll_ids:
        want = set(roll_ids)
        rolls = [r for r in rolls if r["id"] in want]
    return rolls


# ---------------------------------------------------------------------------
# Per-kind evaluators. Each returns an `evaluate(overrides) -> result` closure
# plus the rolls it loaded; result = {objective, per_frame, per_roll, aggregate}.
# ---------------------------------------------------------------------------

def _per_roll_summary(per_frame, key, worst_n=5, worst_desc=True):
    by_roll = {}
    for r in per_frame:
        by_roll.setdefault(r["roll"], []).append(r)
    out = {}
    for rid, recs in by_roll.items():
        vals = [r[key] for r in recs if r.get(key) is not None]
        worst = sorted(recs, key=lambda r: (r.get(key) if r.get(key) is not None
                                             else -1),
                       reverse=worst_desc)[:worst_n]
        out[rid] = {
            "n": len(recs),
            "median_" + key: _median(vals),
            "worst_frames": [{"stem": w["stem"], key: w.get(key)} for w in worst],
        }
    return out


# Set on a worker thread when TRIALS are already running in parallel (parallel
# random_search), so the per-frame loop runs serially and we don't oversubscribe
# cores (trials × frames). Otherwise the per-frame loop uses the thread pool.
_trial_local = threading.local()


def _eval_frames(items, fn):
    """Apply per-frame `fn` over `items`, dropping None, order-preserving. Each
    trial now passes its OWN immutable cfg into `fn` (no shared global mutation),
    so the frames are independent and this is bit-identical to a serial loop,
    just multi-core. When the caller is already inside a parallel TRIAL worker
    (_trial_local.serial), the frames run serially to avoid nested pools."""
    if getattr(_trial_local, "serial", False):
        return [r for r in (fn(it) for it in items) if r is not None]
    workers = an._proc_workers(len(items))
    return [r for r in an._map_frames(fn, items, workers) if r is not None]


def _map_frames_prep(roll_id, label, fn, items):
    """Run a per-frame PREP `fn` (decode + heavy trial-invariant precompute —
    _crop_fields / the GT render) over `items` in the analysis thread pool,
    dropping None, with progress logging. Parallelizing this keeps the cores busy
    instead of decoding frame-by-frame serially after process_roll's parallel
    stages — the prep was the low-CPU stretch."""
    if not items:
        return []
    n = len(items)
    state = {"done": 0}

    def _tick():
        state["done"] += 1
        d = state["done"]
        if d % 5 == 0 or d == n:
            print(f"    [{roll_id}] {label}: prepped {d}/{n} frame(s)", flush=True)

    results = an._map_frames(fn, items, an._proc_workers(n), on_done=_tick)
    return [r for r in results if r is not None]


def _crop_containment_weight(tolerances):
    """How hard the crop objective punishes leaving film border outside the user
    rect. None (the key absent) keeps the original HARD rule (+BIG_PENALTY per
    violating frame) so old sessions reproduce. A NUMBER switches to a symmetric
    soft penalty: objective = overtrim_total + W * undertrim_total, where both
    terms are frame-area fractions — so W is literally 'leaving 1px2 of border is
    W times as bad as losing 1px2 of content.' Smaller W tolerates more border in
    exchange for keeping content (the live params accept that trade)."""
    w = tolerances.get("containment_weight")
    return None if w is None else float(w)


def make_crop_evaluator(rolls, tolerances, fit_params=()):
    """Crop detection runs on the per-frame uncorrected buffer + the global-base
    dmin, neither of which depends on the crop constants. We prep those ONCE and,
    beyond the TIFF decode, also precompute detect_content_crop's heavy
    trial-INVARIANT pixel math (`_crop_fields`: the full-res log10 density extremes
    + line means) so per trial only the cheap `_crop_decide` runs (it reads every
    fittable crop constant, so this stays exact)."""
    prepped, prep_secs = [], {}
    nrolls = len(rolls)
    for i, roll in enumerate(rolls, 1):
        t0 = time.perf_counter()
        print(f"[prep {i}/{nrolls}] roll {roll['id']}: analysing "
              f"{len(roll['images'])} frame(s)…", flush=True)
        frames, _ = an.process_roll(roll["images"], roll["exif"],
                                    progress=_proc_cb(roll["id"]))
        by_stem = {fr["stem"]: fr for fr in frames if not fr.get("error")}
        crop_fix = [f for f in roll["fixtures"]
                    if (json.loads(f.read_text()).get("crop_correction") or {})
                    .get("corrected")]
        prep_items = []
        for f in crop_fix:
            data = json.loads(f.read_text())
            crop = data["crop_correction"]
            stem = data.get("stem") or f.name.replace("_annotations.json", "")
            fr = by_stem.get(stem)
            if fr and "border" in fr:
                prep_items.append((stem, fr, crop))

        # Decode + heavy _crop_fields (the log10 density math) per frame is the
        # slow, trial-INVARIANT prep — run it frame-PARALLEL (was a serial loop
        # that idled the cores after process_roll's parallel stages).
        def _prep_one(item):
            stem, fr, crop = item
            try:
                # crop runs on VIGNETTE-CORRECTED data (matches production +
                # the base it references); load_frame applies the correction.
                enc_f, lin = an.load_frame(fr["path"], fr.get("vignette"))
            except Exception:
                return None
            return stem, {"fields": an._crop_fields(lin, fr["dmin"]),
                          "width": fr["width"], "height": fr["height"],
                          "crop": crop["corrected"]}

        done = _map_frames_prep(roll["id"], "crop", _prep_one, prep_items)
        cache = {stem: c for stem, c in done}
        prepped.append({"id": roll["id"], "cache": cache})
        prep_secs[roll["id"]] = round(time.perf_counter() - t0, 2)
        print(f"[prep {i}/{nrolls}] roll {roll['id']}: {len(cache)} crop frame(s) "
              f"ready in {prep_secs[roll['id']]}s", flush=True)

    items = [(roll["id"], stem, c) for roll in prepped
             for stem, c in roll["cache"].items()]
    cw = _crop_containment_weight(tolerances)

    def evaluate(overrides):
        cfg = reg.to_tuning(overrides)   # immutable per-trial config (no globals)

        def _one(it):
            rid, stem, c = it
            border = an._crop_decide(c["fields"], cfg)
            rec = rqt.crop_overtrim(border, c["crop"], c["width"], c["height"])
            return dict(rec, roll=rid, stem=stem)

        per_frame = _eval_frames(items, _one)
        viols = [r for r in per_frame if not r["contained"]]
        overtrim = [r["overtrim_area"] for r in per_frame]
        undertrim = [r["undertrim_area"] for r in per_frame]
        if cw is None:                       # HARD containment (original rule)
            objective = sum(overtrim) + BIG_PENALTY * len(viols)
        else:                                # soft, area-weighted containment
            objective = sum(overtrim) + cw * sum(undertrim)
        aggregate = {
            "total_overtrim_area": sum(overtrim),
            "median_overtrim_area": _median(overtrim),
            "max_overtrim_area": max(overtrim) if overtrim else 0.0,
            "total_undertrim_area": sum(undertrim),
            "max_undertrim_area": max(undertrim) if undertrim else 0.0,
            "containment_violations": len(viols),
            "containment_weight": cw,
            "n_frames": len(per_frame),
        }
        per_roll = _per_roll_summary(per_frame, "overtrim_area")
        return {"objective": objective, "per_frame": per_frame,
                "per_roll": per_roll, "aggregate": aggregate}

    return evaluate, prep_secs


def _inversion_result(per_frame, clip_budget):
    """Common aggregate/objective for both inversion paths (median histogram
    EMD; a frame clipping past the budget adds BIG_PENALTY)."""
    totals = [r["total"] for r in per_frame]
    n_clip = sum(1 for r in per_frame if r["clip"] > clip_budget)
    objective = (_median(totals) if totals else BIG_PENALTY) \
        + BIG_PENALTY * n_clip
    aggregate = {
        "median_total": _median(totals),
        "median_luma": _median([r["luma"] for r in per_frame]),
        "median_color": _median([r["color"] for r in per_frame]),
        "median_hi999": _median([r["hi999"] for r in per_frame]),
        "max_clip": max((r["clip"] for r in per_frame), default=0.0),
        "frames_over_clip_budget": n_clip,
        "clip_budget": clip_budget,
        "n_frames": len(per_frame),
    }
    return {"objective": objective, "per_frame": per_frame,
            "per_roll": _per_roll_summary(per_frame, "total"),
            "aggregate": aggregate}


def _clip_frac(lin, border, params):
    """Hard-clip fraction of the rendered production params over the content
    crop (subsampled like the print tuner / check_no_clipping)."""
    l, t, r, b = border
    h, w = lin.shape[:2]
    s = max(1, int(round(w * an.PRINT_TUNE_SUBSAMPLE_FRAC)))
    region = lin[t:h - b:s, l:w - r:s].reshape(-1, 3)
    if region.size == 0:
        return 0.0
    out = nm.render_negadoctor(region, params)
    return float(np.mean((out >= rqt.CLIP_OUT_THR).any(axis=1)))


def make_inversion_evaluator(rolls, tolerances, fit_params=()):
    """Picture-vs-picture histogram EMD between the algorithm's render and the
    user's GT-param render. Dispatches:
      - FAST  (fitted params ⊆ PRINT_TUNE_PARAMS): run the pipeline ONCE per
        roll, then per trial only re-run make_params + tune_print_params + render.
      - FULL  (any wb / picker param fitted): re-run the WHOLE pipeline per trial
        (those constants reshape wb_low/wb_high / the D-range in stage B1).
    In both, the GT render is FIXED (the user's annotated fields over the
    baseline non-annotated fields), independent of the trial."""
    clip_budget = float(tolerances.get("clip_max_frac", rqt.CLIP_MAX_FRAC))
    fast = set(fit_params) <= set(reg.PRINT_TUNE_PARAMS)
    if fast:
        return _make_inversion_fast(rolls, clip_budget)
    return _make_inversion_full(rolls, clip_budget, fit_params)


def _make_inversion_fast(rolls, clip_budget):
    prepped, prep_secs = [], {}
    nrolls = len(rolls)
    for i, roll in enumerate(rolls, 1):
        t0 = time.perf_counter()
        print(f"[prep {i}/{nrolls}] roll {roll['id']}: analysing "
              f"{len(roll['images'])} frame(s)…", flush=True)
        frames, _ = an.process_roll(roll["images"], roll["exif"],
                                    progress=_proc_cb(roll["id"]))
        gt = rqt._load_ground_truth(roll["fixtures"])
        by_stem = {fr["stem"]: fr for fr in frames if not fr.get("error")}
        prep_items = [(stem, by_stem[stem], g) for stem, g in gt.items()
                      if by_stem.get(stem) and "params" in by_stem[stem]
                      and by_stem[stem].get("border")]

        # Decode + the FIXED GT render per frame is the slow trial-invariant prep
        # — run it frame-PARALLEL (was a serial loop idling the cores).
        def _prep_one(item):
            stem, fr, g = item
            try:
                enc_f, lin = an.load_frame(fr["path"], fr.get("vignette"))
            except Exception:
                return None
            gt_f = rqt._render_crop_rows(lin, fr["border"],
                                         rqt.gt_params_for_frame(fr, g))
            if gt_f is None:
                return None
            # Rebuild production per trial from the make_params inputs (the
            # PRE-tune starting point): re-tuning the already tuned fr["params"]
            # is NOT idempotent and inflates the metric.
            return stem, {
                "lin": lin, "border": fr["border"], "dmin": fr["dmin"],
                "d_max": fr["d_max"], "offset": fr["offset"],
                "wb_low": fr["wb_low"], "wb_high": fr["wb_high"],
                "picked_min": fr["picked_min"], "picked_max": fr["picked_max"],
                "gt_f": gt_f,
            }

        done = _map_frames_prep(roll["id"], "GT", _prep_one, prep_items)
        cache = {stem: c for stem, c in done}
        prepped.append({"id": roll["id"], "cache": cache})
        prep_secs[roll["id"]] = round(time.perf_counter() - t0, 2)
        print(f"[prep {i}/{nrolls}] roll {roll['id']}: {len(cache)} GT frame(s) "
              f"ready in {prep_secs[roll['id']]}s", flush=True)

    items = [(roll["id"], stem, c) for roll in prepped
             for stem, c in roll["cache"].items()]

    def evaluate(overrides):
        cfg = reg.to_tuning(overrides)   # immutable per-trial config (no globals)

        def _one(it):
            rid, stem, c = it
            # cfg supplies PRINT_GAMMA / PRINT_HI_CEIL / PRINT_HI_PCT etc.
            p = an.make_params(c["dmin"], c["d_max"], c["offset"],
                               c["wb_low"], c["wb_high"],
                               c["picked_min"], c["picked_max"], cfg=cfg)
            tuned, info = an.tune_print_params(c["lin"], p, c["border"],
                                               c["dmin"], cfg=cfg)
            prod_f = rqt._render_crop_rows(c["lin"], c["border"], tuned)
            if prod_f is None:
                return None
            d = nm.histogram_distance(prod_f, c["gt_f"], bins=rqt.HIST_BINS)
            return {"roll": rid, "stem": stem,
                    "total": d["total"], "luma": d["luma"],
                    "color": d["color"], "hi999": abs(d["hi999"]),
                    "clip": float(info.get("clip") or 0.0)}

        return _inversion_result(_eval_frames(items, _one), clip_budget)

    return evaluate, prep_secs


def _make_inversion_full(rolls, clip_budget, fit_params=()):
    """Stage-B1 params (wb / pickers) reshape the analysis, so the pipeline is
    re-run per trial. Each GT frame's decoded buffer + FIXED GT render are cached
    from the baseline so only process_roll (not load/GT-render) repeats.

    The roll-wide vignette + film-base search + global base (process_roll's
    trial-INVARIANT PREFIX) are UPSTREAM of the inversion look and NOT
    inversion-fittable (the registry omits them), so they are computed ONCE per
    roll here and reused on every trial via `process_roll(prep=...)`; only stage
    B1/B2 (what the fitted wb/picker/print constants actually reshape) re-runs.
    The vignette / global-base lines therefore print ONCE per session."""
    # Defensive: the inversion registry must never expose a prefix constant. If
    # one ever leaks back in, fail LOUD here rather than silently re-search the
    # film base / re-estimate the vignette on every trial.
    leaked = set(fit_params) & set(reg.BASE_PREFIX_PARAMS)
    assert not leaked, (
        f"inversion cannot fit the vignette/film-base PREFIX constants {sorted(leaked)} "
        "— they are computed once per session, never calibrated by inversion")
    prepped, prep_secs = [], {}
    nrolls = len(rolls)
    for i, roll in enumerate(rolls, 1):
        t0 = time.perf_counter()
        print(f"[prep {i}/{nrolls}] roll {roll['id']} (full path): analysing "
              f"{len(roll['images'])} frame(s)…", flush=True)
        # Compute the trial-invariant prefix ONCE; derive the baseline frames
        # (for the FIXED GT render) from it and reuse it per trial.
        prefix = an.process_roll_prefix(roll["images"], roll["exif"],
                                        progress=_proc_cb(roll["id"]),
                                        cfg=an.DEFAULT_TUNING)
        frames, _ = an.process_roll(roll["images"], roll["exif"],
                                    cfg=an.DEFAULT_TUNING, prep=prefix)
        gt = rqt._load_ground_truth(roll["fixtures"])
        by_stem = {fr["stem"]: fr for fr in frames if not fr.get("error")}
        prep_items = [(stem, by_stem[stem], g) for stem, g in gt.items()
                      if by_stem.get(stem) and "params" in by_stem[stem]
                      and by_stem[stem].get("border")]

        # Decode + the FIXED GT render per frame — run it frame-PARALLEL.
        def _prep_one(item):
            stem, fr, g = item
            try:
                enc_f, lin = an.load_frame(fr["path"], fr.get("vignette"))
            except Exception:
                return None
            gt_f = rqt._render_crop_rows(lin, fr["border"],
                                         rqt.gt_params_for_frame(fr, g))
            if gt_f is None:
                return None
            return stem, {"lin": lin, "border": fr["border"], "gt_f": gt_f}

        done = _map_frames_prep(roll["id"], "GT", _prep_one, prep_items)
        cache = {stem: c for stem, c in done}
        prepped.append({"id": roll["id"], "images": roll["images"],
                        "exif": roll["exif"], "cache": cache, "prefix": prefix})
        prep_secs[roll["id"]] = round(time.perf_counter() - t0, 2)
        print(f"[prep {i}/{nrolls}] roll {roll['id']}: {len(cache)} GT frame(s) "
              f"ready in {prep_secs[roll['id']]}s. NOTE: full path re-runs only "
              "stage B1/B2 per trial — vignette + film base computed ONCE.",
              flush=True)

    def _one(it):
        rid, stem, c, params = it
        prod_f = rqt._render_crop_rows(c["lin"], c["border"], params)
        if prod_f is None:
            return None
        d = nm.histogram_distance(prod_f, c["gt_f"], bins=rqt.HIST_BINS)
        return {"roll": rid, "stem": stem,
                "total": d["total"], "luma": d["luma"],
                "color": d["color"], "hi999": abs(d["hi999"]),
                "clip": _clip_frac(c["lin"], c["border"], params)}

    def evaluate(overrides):
        cfg = reg.to_tuning(overrides)   # immutable per-trial config (no globals)
        per_frame = []
        for roll in prepped:
            # process_roll already threads the per-frame analysis internally;
            # the render+EMD pass below is then threaded over the GT frames too.
            # `prep` reuses the trial-invariant vignette/film-base prefix (None
            # when a fitted BASE_* forces a per-trial rebuild).
            frames, _ = an.process_roll(roll["images"], roll["exif"], cfg=cfg,
                                        prep=roll["prefix"])
            by_stem = {fr["stem"]: fr for fr in frames if not fr.get("error")}
            items = [(roll["id"], stem, c, by_stem[stem]["params"])
                     for stem, c in roll["cache"].items()
                     if by_stem.get(stem) and "params" in by_stem[stem]]
            per_frame += _eval_frames(items, _one)
        return _inversion_result(per_frame, clip_budget)

    return evaluate, prep_secs


def _vignette_result(per_frame):
    residuals = [r["residual"] for r in per_frame
                 if not r["rejected"] and r["residual"] is not None]
    n_rej = sum(1 for r in per_frame if r["rejected"])
    objective = BIG_PENALTY * n_rej + (_median(residuals) if residuals else 0.0)
    aggregate = {"rejected_rolls": n_rej, "median_residual": _median(residuals),
                 "max_residual": max(residuals) if residuals else 0.0,
                 "n_frames": len(per_frame)}
    per_roll = {r["roll"]: {"n": 1, "rejected": r["rejected"],
                            "residual": r["residual"],
                            "corner_falloff": r["corner_falloff"]}
                for r in per_frame}
    return {"objective": objective, "per_frame": per_frame,
            "per_roll": per_roll, "aggregate": aggregate}


def _vig_record(roll_id, params, info):
    return {"roll": roll_id, "stem": "(roll)", "rejected": params is None,
            "residual": info.get("residual"),
            "corner_falloff": info.get("corner_falloff"),
            "reason": info.get("reason"), "strength": (params or {}).get("strength")}


def _envelope_record(roll_id, env, cfg=None):
    """A vignette per-roll record from a (possibly empty) captured envelope."""
    if env["r"] is None:
        return _vig_record(roll_id, None, {"reason": env["reason"],
                                           "frames": env["used"]})
    cfg = cfg if cfg is not None else an.DEFAULT_TUNING
    return _vig_record(roll_id,
                       *an.fit_vignette_profile(env["r"], env["e"], env["used"],
                                                cfg))


def make_vignette_evaluator(rolls, tolerances, fit_params=()):
    """A roll whose fit_vignette_profile returns None (rejected) dominates the
    objective — the 'a new roll fails to auto-find vignette' guard. The vignette
    envelope is derived from the roll's TIFFs (NO committed fixture). Dispatches:
      - FAST (fitted ⊆ VIG_FIT_PARAMS — profile-fit only): capture each roll's
        radial envelope from the TIFFs ONCE, then re-run only fit_vignette_profile
        per trial.
      - FULL (an envelope-accumulation constant is fitted): re-run the whole
        estimate_vignette (capture + fit) on the TIFFs per trial."""
    fast = set(fit_params) <= set(reg.VIG_FIT_PARAMS)
    prep_secs = {}

    if fast:
        env_rolls, nrolls = [], len(rolls)
        for i, roll in enumerate(rolls, 1):
            t0 = time.perf_counter()
            print(f"[prep {i}/{nrolls}] roll {roll['id']}: capturing vignette "
                  f"envelope from {len(roll['images'])} frame(s)…", flush=True)
            env = an.vignette_envelope(roll["images"], roll["exif"])
            env_rolls.append({"id": roll["id"], "env": env})
            prep_secs[roll["id"]] = round(time.perf_counter() - t0, 2)
            ok = "captured" if env["r"] else f"UNAVAILABLE ({env['reason']})"
            print(f"[prep {i}/{nrolls}] roll {roll['id']}: envelope {ok} in "
                  f"{prep_secs[roll['id']]}s", flush=True)

        def evaluate(overrides):
            cfg = reg.to_tuning(overrides)
            return _vignette_result([_envelope_record(er["id"], er["env"], cfg)
                                     for er in env_rolls])
    else:
        # FULL path: an accumulation constant is fitted, so the envelope must be
        # re-folded per trial — but the TIFF DECODE is trial-invariant, so decode
        # each frame's luma/clip/border ONCE here and re-fold from RAM per trial
        # (no re-reading the TIFFs). Same result as decoding per trial.
        cached, nrolls = [], len(rolls)
        for i, roll in enumerate(rolls, 1):
            t0 = time.perf_counter()
            print(f"[prep {i}/{nrolls}] roll {roll['id']}: decoding "
                  f"{len(roll['images'])} frame(s) for the vignette cache…",
                  flush=True)
            fc = an.vignette_frame_cache(roll["images"], roll["exif"])
            cached.append({"id": roll["id"], "images": roll["images"],
                           "exif": roll["exif"], "frame_cache": fc})
            prep_secs[roll["id"]] = round(time.perf_counter() - t0, 2)
            ndec = sum(1 for c in fc if c is not None)
            print(f"[prep {i}/{nrolls}] roll {roll['id']}: {ndec}/{len(fc)} "
                  f"frame(s) decoded + cached in {prep_secs[roll['id']]}s "
                  "(trials re-fold from RAM — no more disk reads)", flush=True)

        def evaluate(overrides):
            cfg = reg.to_tuning(overrides)
            return _vignette_result([
                _vig_record(roll["id"],
                            *an.estimate_vignette(roll["images"], roll["exif"],
                                                  frame_cache=roll["frame_cache"],
                                                  cfg=cfg))
                for roll in cached])

    return evaluate, prep_secs


EVALUATORS = {
    "crop": make_crop_evaluator,
    "inversion": make_inversion_evaluator,
    "vignette": make_vignette_evaluator,
}


# ---------------------------------------------------------------------------
# Session writing
# ---------------------------------------------------------------------------

def _git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           cwd=str(TESTS_DIR), capture_output=True, text=True,
                           timeout=5)
        return r.stdout.strip() or None
    except Exception:
        return None


def _session_dir(kind):
    CALIB_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    nn = sum(1 for p in CALIB_DIR.glob(f"*_{kind}_*") if p.is_dir()) + 1
    d = CALIB_DIR / f"{stamp}_{kind}_{nn:02d}"
    d.mkdir(parents=True)
    return d


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _method_params(fit):
    """The EFFECTIVE hyperparameters of the chosen fit method, including the
    optimizer defaults applied in _coord/_random when the config omits them, so
    a session is fully reproducible from the report alone. Ordered dict."""
    m = fit.get("method", "none")
    if m == "coordinate_descent":
        return {
            "epsilon": float(fit.get("epsilon", 1e-4)),
            "max_iters": int(fit.get("max_iters", 20)),
            "init_step": fit.get("init_step", "per-param/auto"),
            "step_min": fit.get("step_min", "per-param/auto"),
            "step_shrink": float(fit.get("step_shrink", 0.5)),
        }
    if m == "random_search":
        return {"n_trials": int(fit.get("n_trials", 50)),
                "seed": int(fit.get("seed", 0)),
                "workers": int(fit.get("workers", 0)) or "auto"}
    if m == "grid":
        return {"grid_step": "per-param (see fitted params / config)"}
    return {}


def _fmtv(v):
    """Compact value formatting for method params — uses %g for floats so a
    small epsilon (1e-5) is NOT rounded to 0.0000 like _fmt would."""
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _method_desc(fit):
    """One-line 'method(k=v, …)' for the index row and report header."""
    params = _method_params(fit)
    if not params:
        return fit.get("method", "none")
    inner = ", ".join(f"{k}={_fmtv(v)}" for k, v in params.items())
    return f"{fit.get('method', 'none')}({inner})"


def _write_report(path, config, results):
    kind = config["kind"]
    agg = results["aggregate"]
    lines = []
    lines.append(f"# Calibration session — {kind}\n")
    lines.append(f"- created: {config['created']}")
    lines.append(f"- git commit: {config.get('git_commit')}")
    lines.append(f"- rolls: {', '.join(config['rolls'])}")
    lines.append(f"- metric: {config['metric']}")
    lines.append(f"- fit method: {config['fit']['method']}")
    mp = _method_params(config["fit"])
    if mp:
        lines.append("- method params: "
                     + ", ".join(f"{k}={_fmtv(v)}" for k, v in mp.items()))
    fitp = config["fit"].get("params") or {}
    if fitp:
        lines.append(f"- fitted params: {', '.join(fitp)}")
    lines.append(f"- trial count: {results['trial_count']}")
    lines.append(f"- wall time: {results['wall_seconds']}s "
                 f"(prep {results['prep_seconds']}s)")
    lines.append(f"- objective: {_fmt(results['objective_initial'])} (init) "
                 f"-> {_fmt(results['objective_final'])} (final)\n")

    tr = results.get("trace") or {}
    if tr:
        lines.append("## Convergence\n")
        lines.append(f"- method: {tr.get('method')}  trials: {tr.get('trials')}")
        if tr.get("method") == "coordinate_descent":
            lines.append(f"- cycles run: {tr.get('cycles_run')} "
                         f"(of max {config['fit'].get('max_iters')}); "
                         f"converged early: {tr.get('converged_early')} "
                         f"(epsilon {tr.get('epsilon')})")
            if tr.get("cycles"):
                lines.append("- per-cycle objective (improvement):")
                for c in tr["cycles"]:
                    lines.append(f"  - cycle {c['cycle']}: {_fmt(c['objective'])} "
                                 f"(−{_fmt(c['improvement'])})")
        elif tr.get("method") == "grid":
            lines.append(f"- grid total combinations: {tr.get('grid_total')}")
        elif tr.get("method") == "random_search":
            lines.append(f"- samples: {tr.get('n_trials')} (seed {tr.get('seed')})")
        imp = tr.get("improvements") or []
        lines.append(f"- best-so-far improved {len(imp)} time(s); curve "
                     "(trial: objective):")
        for p in imp:
            lines.append(f"  - {p['trial']}: {_fmt(p['objective'])}")
        lines.append("")

    contribs = (tr.get("contributions") if tr else None)
    if contribs:
        lines.append("## Main contributors (objective drop credited per param)\n")
        lines.append("Free attribution from coordinate descent's accepted moves "
                     "— which single params drove convergence.\n")
        rows = [c for c in contribs if c["moves"] > 0]
        if rows:
            lines.append("| param | objective drop | % of total | moves |")
            lines.append("|---|---|---|---|")
            for c in rows:
                lines.append(f"| {c['param']} | {_fmt(c['improvement'])} "
                             f"| {c['pct']:.1f}% | {c['moves']} |")
        else:
            lines.append("_(no accepted moves — the start was already optimal "
                         "for these params on these rolls.)_")
        lines.append("")

    pca = results.get("principal_components")
    if pca:
        lines.append("## Principal components (curvature of the metric at the "
                     "optimum)\n")
        lines.append(f"Eigen-decomposition of the finite-difference Hessian in "
                     f"grid-step-normalized units ({pca['n_evals']} extra evals). "
                     "Each component is a DIRECTION in param space (loadings show "
                     "the coupled params); larger |eigenvalue| = the metric "
                     "changes more sharply along it.\n")
        for i, comp in enumerate(pca["components"][:8], 1):
            load = "  ".join(f"{ld['weight']:+.2f}·{ld['param']}"
                             for ld in comp["top_loadings"])
            lines.append(f"- **PC{i}** eigenvalue {_fmt(comp['eigenvalue'])} "
                         f"({comp['variance_pct']:.0f}% of total curvature): {load}")
        lines.append("")

    if results.get("fitted"):
        lines.append("## Fitted constants (record-only — adopt by hand)\n")
        lines.append("| constant | init | fitted |")
        lines.append("|---|---|---|")
        for n, v in results["fitted"].items():
            lines.append(f"| {n} | {_fmt(results['init'][n])} | {_fmt(v)} |")
        lines.append("")

    lines.append("## Aggregate (all rolls)\n")
    for k, v in agg.items():
        lines.append(f"- {k}: {_fmt(v)}")
    lines.append("")

    lines.append("## Per-roll\n")
    for rid, summary in results["per_roll"].items():
        wall = results["prep_seconds_by_roll"].get(rid)
        lines.append(f"### {rid}  (prep {wall}s)")
        lines.append("```")
        lines.append(json.dumps(summary, indent=2))
        lines.append("```")
    lines.append("")

    lines.append("## Worst frames (find where to look)\n")
    sort_key, desc = _worst_key(kind)
    worst = sorted(results["per_frame"],
                   key=lambda r: (r.get(sort_key) if r.get(sort_key) is not None
                                  else -1),
                   reverse=desc)[:15]
    if worst:
        cols = list(worst[0].keys())
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join("---" for _ in cols) + "|")
        for r in worst:
            lines.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _worst_key(kind):
    return {"crop": ("overtrim_area", True),
            "inversion": ("total", True),
            "vignette": ("residual", True)}[kind]


def _headline(kind, agg):
    if kind == "crop":
        cw = agg.get("containment_weight")
        tail = (f"viol {agg['containment_violations']} (HARD)" if cw is None
                else f"undertrim {_fmt(agg.get('total_undertrim_area', 0.0))} "
                     f"(W={_fmtv(cw)}, viol {agg['containment_violations']})")
        return f"overtrim {_fmt(agg['total_overtrim_area'])} " + tail
    if kind == "inversion":
        return (f"EMD {_fmt(agg['median_total'])} "
                f"clip {agg['frames_over_clip_budget']}")
    return f"rejected {agg['rejected_rolls']} resid {_fmt(agg['median_residual'])}"


def _append_index(kind, config, results, session_name):
    idx = CALIB_DIR / f"INDEX_{kind}.md"
    if not idx.is_file():
        idx.write_text(
            f"# Calibration history — {kind}\n\n"
            "Comparable rows (one metric per kind; see calibrations/README.md).\n\n"
            "| session | rolls | method | objective | headline | wall_s | commit |\n"
            "|---|---|---|---|---|---|---|\n", encoding="utf-8")
    row = (f"| [{session_name}]({session_name}/report.md) "
           f"| {','.join(config['rolls'])} | {_method_desc(config['fit'])} "
           f"| {_fmt(results['objective_final'])} "
           f"| {_headline(kind, results['aggregate'])} "
           f"| {results['wall_seconds']} | {config.get('git_commit')} |\n")
    with open(idx, "a", encoding="utf-8") as f:
        f.write(row)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_session(config):
    kind = config["kind"]
    if kind not in EVALUATORS:
        raise ValueError(f"unknown kind {kind!r}")
    fit = config["fit"]
    spec = build_spec(kind, fit.get("params"))
    if fit.get("method", "none") != "none" and not spec:
        raise ValueError("fit.method is a search but fit.params is empty")

    # Every kind derives what it needs from the roll's local TIFFs (vignette
    # captures its radial envelope; crop & inversion render).
    rolls = discover_image_rolls(config.get("rolls") or None)
    if not rolls:
        print(f"SKIP: no {kind} rolls with local TIFFs. "
              "See fixtures/rolls/README.md to repopulate a roll's exports.")
        return None
    config["rolls"] = [r["id"] for r in rolls]
    config["metric"] = METRIC_NAME[kind]
    config["created"] = datetime.now().isoformat(timespec="seconds")
    config["git_commit"] = _git_commit()

    print("=" * 70)
    print(f"CALIBRATION  kind={kind}  method={_method_desc(fit)}  "
          f"rolls={len(rolls)} ({', '.join(config['rolls'])})  "
          f"params={len(spec)}")
    print(f"  metric: {config['metric']}")
    print("=" * 70, flush=True)

    snap = reg.snapshot(list(spec))   # restore the live module no matter what
    t_wall = time.perf_counter()
    try:
        evaluate, prep_secs = EVALUATORS[kind](rolls, fit.get("tolerances") or {},
                                               list(spec))
        init_overrides = {n: s["init"] for n, s in spec.items()} if spec else {}
        init_result = evaluate(init_overrides) if spec else evaluate({})
        objective_initial = init_result["objective"]

        if fit.get("method", "none") == "none" or not spec:
            best, trials = init_overrides, 1
            trace = {"method": "none", "trials": 1, "best_objective":
                     objective_initial,
                     "improvements": [{"trial": 1, "objective": objective_initial}]}
            final = init_result
        else:
            best, best_obj, trials, trace = optimize(
                lambda o: evaluate(o)["objective"], spec, fit)
            final = evaluate(best)

        pca = None
        if fit.get("pca") and spec:
            n = len(spec)
            print(f"Computing principal components at the optimum "
                  f"(~{1 + 2 * n + n * (n - 1) // 2} extra evals)...")
            pca = principal_components(evaluate, spec, best)
    finally:
        reg.restore(snap)

    wall = round(time.perf_counter() - t_wall, 2)
    results = {
        "wall_seconds": wall,
        "prep_seconds": round(sum(prep_secs.values()), 2),
        "prep_seconds_by_roll": prep_secs,
        "trial_count": trials,
        "objective_initial": objective_initial,
        "objective_final": final["objective"],
        "trace": trace,
        "principal_components": pca,
        "init": {n: s["init"] for n, s in spec.items()},
        "fitted": best if spec else {},
        "aggregate": final["aggregate"],
        "per_roll": final["per_roll"],
        "per_frame": final["per_frame"],
    }

    session = _session_dir(kind)
    (session / "config.json").write_text(json.dumps(config, indent=2))
    (session / "results.json").write_text(json.dumps(results, indent=2))
    (session / "fitted_params.json").write_text(json.dumps(
        {"kind": kind, "git_commit": config["git_commit"],
         "fitted": results["fitted"], "init": results["init"]}, indent=2))
    # Complete DROP-IN preset = DEFAULT_TUNING + the fitted overrides. Adopting
    # the result is then `cp fitted_preset.json auto_negadoctor/presets/<name>.json`
    # (or --preset <this path>) — no editing auto_negadoctor.py. (`fitted_params`
    # above stays the minimal record of just what moved + its init.)
    an._tuning.dump(reg.to_tuning(results["fitted"]),
                    str(session / "fitted_preset.json"))
    _write_report(session / "report.md", config, results)
    _append_index(kind, config, results, session.name)

    print(f"\nSession written: {session}")
    print(f"  objective {_fmt(objective_initial)} -> {_fmt(final['objective'])} "
          f"({trials} trial(s), {wall}s)")
    print(f"  {_headline(kind, final['aggregate'])}")
    if spec:
        for n, v in best.items():
            print(f"  {n}: {_fmt(results['init'][n])} -> {_fmt(v)}")
    return session


# ---------------------------------------------------------------------------
# Post-fit debug-UI review (inversion): your-annotation render vs fitted render
# ---------------------------------------------------------------------------

def _review_payload(kind, fr, roll_meta):
    """The kind-specific result the debug UI's R toggle swaps (live vs fitted)."""
    if kind == "crop":
        return {"border": list(fr["border"])}
    if kind == "vignette":
        return {"vignette": (roll_meta or {}).get("vignette")}
    return {"params": fr["params"], "params_hex": fr["params_hex"]}  # inversion


def _review_run(kind, roll, cfg):
    """Run the pipeline once with `cfg` and return {stem: payload} + roll_meta.
    (process_roll covers crop borders + inversion params + the roll vignette in
    one pass.)"""
    frames, roll_meta = an.process_roll(roll["images"], roll["exif"], cfg=cfg)
    out = {}
    for fr in frames:
        if fr.get("error") or "params" not in fr or not fr.get("border"):
            continue
        out[fr["stem"]] = _review_payload(kind, fr, roll_meta)
    return out, frames, roll_meta


def review_session(session_dir, roll_id=None):
    """Open the debug UI on a finished session showing its FITTED result, with
    the R toggle flipping to the LIVE (current source-code) result so you can
    preview crop / vignette / inversion fitted-vs-live. N toggles the vignette
    correction on/off in the preview."""
    session = Path(session_dir)
    config = json.loads((session / "config.json").read_text())
    results = json.loads((session / "results.json").read_text())
    kind = config["kind"]
    fitted = results.get("fitted") or {}
    roll_ids = [roll_id] if roll_id else config["rolls"]
    rolls = discover_image_rolls(roll_ids)
    if not rolls:
        print("No local TIFFs for the session's rolls — repopulate them "
              "(fixtures/rolls/README.md) to review.")
        return None
    roll = rolls[0]

    # The review debug-session is transient (preview only) — write it to a TEMP
    # dir, NOT into the session folder, and delete it when the window closes.
    review_dir = Path(tempfile.mkdtemp(prefix="nega_review_"))
    # LIVE = current source constants; FITTED = this session's cfg. No global
    # mutation — each pass gets its own immutable cfg.
    live_payloads, _, _ = _review_run(kind, roll, an.DEFAULT_TUNING)
    fit_payloads, frames, roll_meta = _review_run(kind, roll,
                                                  reg.to_tuning(fitted))

    for fr in frames:
        if fr.get("error") or fr["stem"] not in fit_payloads:
            continue
        fr["review_kind"] = kind
        fr["review"] = {"fitted": fit_payloads[fr["stem"]],
                        "live": live_payloads.get(fr["stem"],
                                                  fit_payloads[fr["stem"]])}
    an.write_debug_sessions(frames, roll_meta, review_dir)

    ui = TESTS_DIR.parent / "debug_ui.py"
    print(f"Opening debug UI for {session.name} (close the window to return)")
    print(f"  R: FITTED ({kind} from this session) <-> live (current code)   "
          "N: vignette on/off")
    try:
        # Block until the window is closed. Popen would leave the GUI child
        # holding the inherited stdout pipe, so `conda run` never gets EOF and
        # the shell hangs until Ctrl-C — waiting here returns cleanly instead.
        subprocess.run([sys.executable, str(ui), str(review_dir)])
    finally:
        shutil.rmtree(review_dir, ignore_errors=True)
    return session


# ---------------------------------------------------------------------------
# Config + CLI
# ---------------------------------------------------------------------------

def load_config(args):
    if args.config:
        config = json.loads(Path(args.config).read_text())
    else:
        if not args.kind:
            raise SystemExit("provide --config or --kind")
        config = {"kind": args.kind, "rolls": [],
                  "fit": {"method": "none", "params": {}}}
    config.setdefault("fit", {})
    config["fit"].setdefault("params", {})
    if args.kind:
        config["kind"] = args.kind
    if args.method:
        config["fit"]["method"] = args.method
    if args.rolls:
        config["rolls"] = args.rolls
    # Method hyperparameters: any CLI flag, when given, overrides the config.
    for flag, key in (("epsilon", "epsilon"), ("max_iters", "max_iters"),
                      ("step_shrink", "step_shrink"), ("init_step", "init_step"),
                      ("step_min", "step_min"), ("n_trials", "n_trials"),
                      ("seed", "seed"), ("workers", "workers")):
        val = getattr(args, flag, None)
        if val is not None:
            config["fit"][key] = val
    if args.pca:
        config["fit"]["pca"] = True
    return config


def main():
    ap = argparse.ArgumentParser(description="Recorded calibration sessions (spec 05)")
    ap.add_argument("--config", help="path to a session config JSON")
    ap.add_argument("--kind", choices=list(EVALUATORS),
                    help="override / set the calibration kind")
    ap.add_argument("--method", choices=["none", "grid", "coordinate_descent",
                                         "random_search"],
                    help="override the fitting method")
    ap.add_argument("--rolls", nargs="+", help="restrict to these roll ids")
    # Method hyperparameters (override the config; effective defaults shown in
    # the report's "method params" line). Per-param ranges/grid_steps stay in
    # the config — they are inherently per-constant.
    ap.add_argument("--epsilon", type=float,
                    help="coordinate_descent: stop when a cycle improves < this")
    ap.add_argument("--max-iters", type=int, dest="max_iters",
                    help="coordinate_descent: max cycles")
    ap.add_argument("--step-shrink", type=float, dest="step_shrink",
                    help="coordinate_descent: per-cycle step shrink factor")
    ap.add_argument("--init-step", type=float, dest="init_step",
                    help="coordinate_descent: initial line-search step (all params)")
    ap.add_argument("--step-min", type=float, dest="step_min",
                    help="coordinate_descent: stop shrinking below this step")
    ap.add_argument("--n-trials", type=int, dest="n_trials",
                    help="random_search: number of samples")
    ap.add_argument("--seed", type=int, help="random_search: RNG seed")
    ap.add_argument("--workers", type=int,
                    help="random_search: parallel trial threads (0/omit = auto, "
                         "capped at the analysis worker count; 1 = serial)")
    ap.add_argument("--pca", action="store_true",
                    help="after the fit, compute the principal components of the "
                         "metric's curvature at the optimum (O(N^2) extra evals — "
                         "use a small param subset)")
    ap.add_argument("--review", metavar="SESSION_DIR",
                    help="open the debug UI comparing fitted vs your annotation "
                         "for a finished inversion session")
    ap.add_argument("--review-roll", help="roll id to review (default: first)")
    args = ap.parse_args()

    if args.review:
        review_session(args.review, args.review_roll)
        return 0

    config = load_config(args)
    try:
        run_session(config)
    except KeyboardInterrupt:
        # In-flight pool tasks live in non-daemon threads that can't be
        # interrupted mid-numpy; a normal exit would join them. os._exit stops
        # NOW. A fit writes its session only on success, so nothing half-written
        # is left behind (no cleanup to skip during the search itself).
        sys.stdout.flush()
        sys.stderr.flush()
        print("\nInterrupted — calibration aborted (no session written).",
              file=sys.stderr, flush=True)
        os._exit(130)
    return 0


if __name__ == "__main__":
    sys.exit(main())
