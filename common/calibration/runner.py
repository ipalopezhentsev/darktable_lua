"""Feature-agnostic calibration RUNNER: optimizers + recorded sessions.

This is the shared engine behind each feature's `tests/run_calibration.py`. It owns
everything that does NOT depend on what is being calibrated:

  - native optimizers (`_grid` / `_coord` / `_random`) + the live progress `_Tracker`
    + `principal_components` (curvature of the metric at the optimum)
  - `build_spec` (merge registry defaults with a config's per-param overrides)
  - the recorded-session lifecycle: `run_session` evaluates / fits / writes
    `config.json` + `results.json` + `report.md` + `fitted_params.json` +
    `fitted_preset.json` and appends a per-kind `INDEX_<kind>.md` row
  - the CLI (`run_main` / `load_config`)

The feature supplies a `CalibrationAdapter` with the only feature-specific pieces:
its `registry` + `schema`, its per-kind `evaluators` (the metric), roll discovery,
the parallel `map_frames`, and the small per-kind cosmetics (`worst_key`/`headline`).
This mirrors `common/debug_ui_base.py`: a shared base + thin feature hooks.

RECORD-ONLY: the runner never edits the feature's source. A session records its best
constants in `fitted_params.json` / `fitted_preset.json`; the user adopts by hand.
"""
import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np

BIG_PENALTY = 1e6   # per rejected / constraint-violating / clipping frame


# ---------------------------------------------------------------------------
# Feature adapter
# ---------------------------------------------------------------------------

class CalibrationAdapter:
    """The feature-specific surface the runner drives. Subclass it (like a debug-UI
    subclass of DebugUIBase) and set the attributes / override the hooks.

    Attributes (set in the subclass __init__):
      registry       common.calibration.registry.Registry
      schema         object with `.dump(cfg, path)` + `.Tuning` (the feature tuning module)
      calib_dir      pathlib.Path the sessions are written under
      evaluators     {kind: factory(rolls, tolerances, fit_params)->(evaluate, prep_secs)}
                     where evaluate(overrides)->{objective, per_frame, per_roll, aggregate}
      metric_name    {kind: human description of the metric}
      description    CLI description string (optional)
    """
    registry = None
    schema = None
    calib_dir = None
    evaluators = {}
    metric_name = {}
    description = "Recorded calibration sessions"

    # -- per-kind cosmetics ------------------------------------------------
    def worst_key(self, kind):
        """(sort_key, descending) for the report's 'worst frames' table."""
        raise NotImplementedError

    def headline(self, kind, agg):
        """One-line summary of an aggregate for the console + INDEX row."""
        raise NotImplementedError

    # -- data --------------------------------------------------------------
    def discover_rolls(self, roll_ids=None):
        """List the rolls/sessions a kind derives its ground truth from, optionally
        filtered to `roll_ids`. Each is a dict with at least an 'id' key (the
        evaluators define the rest)."""
        raise NotImplementedError

    # -- parallelism (override for multi-core; serial by default) ----------
    def proc_workers(self, n):
        return 1

    def map_frames(self, fn, items, workers, on_done=None):
        out = []
        for it in items:
            out.append(fn(it))
            if on_done:
                on_done()
        return out

    # -- review (optional) -------------------------------------------------
    def review(self, session_dir, roll_id=None):
        print("This feature does not support --review.")
        return None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _median(vals):
    return statistics.median(vals) if vals else float("nan")


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
    """A progress callback that prints the analysis percent at 25% milestones."""
    state = {"mark": -1}

    def cb(done, total):
        if not total:
            return
        pct = 100.0 * done / total
        if pct >= state["mark"] + 25 or done >= total:
            state["mark"] = pct - (pct % 25)
            print(f"    [{label}] analysis {pct:3.0f}%", flush=True)
    return cb


# ---------------------------------------------------------------------------
# Optimizers — native, dependency-free. Each minimizes objective(overrides).
# ---------------------------------------------------------------------------

class _Tracker:
    """Wraps the objective to (1) log the best-so-far CURVE (one point each time the
    objective drops) and (2) print live progress: job % + ETA when the trial total
    is known (grid/random), else the eval count + the coordinate-descent context.
    Throttled to ~1s plus every improvement."""
    def __init__(self, fn, verbose=True):
        self.fn = fn
        self.n = 0
        self.best = None
        self.improvements = []     # [{trial, objective}] at each new best
        self.total = None
        self.ctx = ""
        self.verbose = verbose
        self.t0 = time.perf_counter()
        self.last = 0.0

    def __call__(self, x):
        return self.record(self.fn(x))

    def record(self, v):
        """Bookkeeping for a PRE-COMPUTED objective value (so parallel trials can
        evaluate concurrently, then feed results here in deterministic point order)."""
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


def optimize(objective, spec, fit, proc_workers=None):
    """Return (best_overrides, best_objective, trial_count, trace). `trace` is a
    uniform convergence record. `proc_workers(n)->int` sets the auto worker count for
    parallel random_search (None = serial unless fit['workers'] is given)."""
    method = fit.get("method", "none")
    tr = _Tracker(objective, verbose=fit.get("verbose", True))
    print(f"Fitting: {_method_desc(fit)}, {len(spec)} param(s) "
          f"[{', '.join(sorted(list(spec)))}]", flush=True)
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
        x, extra = _random(tr, spec, fit, proc_workers)
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


def _step_levels(step, step_min, shrink):
    """How many distinct step sizes the shrinking line search walks for one param in
    one cycle (the param's fixed per-cycle 'grid')."""
    if step < step_min:
        return 0
    if not (0.0 < shrink < 1.0):
        return 1
    return int(math.floor(math.log(step_min / step) / math.log(shrink))) + 1


def _coord(objective, spec, fit):
    """Coordinate descent: per param a shrinking-step ± line search; repeat full
    cycles until a cycle improves by less than epsilon, or max_iters cycles."""
    eps = float(fit.get("epsilon", 1e-4))
    max_iters = int(fit.get("max_iters", 20))
    x = {n: s["init"] for n, s in spec.items()}
    best = objective(x)
    cycles, converged_early = [], False
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
            nlev = _step_levels(step, step_min, shrink)
            level = 1
            while step >= step_min:
                moved = False
                for d in (step, -step):
                    cand = dict(x)
                    cand[n] = _clamp(x[n] + d, lo, hi)
                    if abs(cand[n] - x[n]) < 1e-12:
                        continue
                    if hasattr(objective, "ctx"):
                        objective.ctx = (f"cycle {cyc + 1}/{max_iters} "
                                         f"param {pi}/{nparams} step {level}/{nlev} "
                                         f"probe {'+' if d > 0 else '-'} "
                                         f"{n}={cand[n]:.6g}")
                    o = objective(cand)
                    if o < best - 1e-12:
                        contrib[n]["improvement"] += best - o
                        contrib[n]["moves"] += 1
                        best, x, moved = o, cand, True
                        break
                if not moved:
                    step *= shrink
                    level += 1
        cycles.append({"cycle": cyc + 1, "objective": best,
                       "improvement": before - best})
        if before - best < eps:
            converged_early = True
            break
    return x, {"cycles": cycles, "cycles_run": len(cycles),
               "converged_early": converged_early, "epsilon": eps,
               "contributions": _rank_contributions(contrib)}


# Set on a worker thread when TRIALS are already running in parallel (parallel
# random_search), so the per-frame loop runs serially and we don't oversubscribe.
_trial_local = threading.local()


def _random(objective, spec, fit, proc_workers=None):
    """Uniform random search. Trials are INDEPENDENT, so when workers>1 they run in
    parallel THREADS (each trial builds its own immutable cfg — no shared mutation).
    Points are sampled up front (seeded) and recorded in point order, so the parallel
    run is bit-identical to the serial one."""
    n_trials = int(fit.get("n_trials", 50))
    seed = int(fit.get("seed", 0))
    rng = random.Random(seed)
    points = [{n: rng.uniform(s["range"][0], s["range"][1])
               for n, s in spec.items()} for _ in range(n_trials)]
    workers = int(fit.get("workers", 0))
    if not workers and proc_workers is not None:
        workers = proc_workers(n_trials)
    workers = workers or 1
    if hasattr(objective, "total"):
        objective.total = n_trials
    verbose = getattr(objective, "verbose", True)

    if workers <= 1:
        best, bx = None, None
        for x in points:
            o = objective(x)
            if best is None or o < best:
                best, bx = o, x
        return bx, {"n_trials": n_trials, "seed": seed, "workers": 1}

    objs = [None] * n_trials

    def _worker(i):
        _trial_local.serial = True
        try:
            return i, objective.fn(points[i])
        finally:
            _trial_local.serial = False

    done, best_live, t0, last = 0, None, time.perf_counter(), 0.0
    ex = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="calib-trial")
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

    saved, objective.verbose = objective.verbose, False
    best, bx = None, None
    for x, o in zip(points, objs):
        objective.record(o)
        if best is None or o < best:
            best, bx = o, x
    objective.verbose = saved
    return bx, {"n_trials": n_trials, "seed": seed, "workers": workers}


def _rank_contributions(contrib):
    """Per-param objective drop credited by coordinate descent's accepted moves."""
    total = sum(c["improvement"] for c in contrib.values()) or 1.0
    ranked = sorted(contrib.items(), key=lambda t: -t[1]["improvement"])
    return [{"param": n, "improvement": c["improvement"], "moves": c["moves"],
             "pct": 100.0 * c["improvement"] / total} for n, c in ranked]


def principal_components(evaluate, spec, point):
    """PRINCIPAL COMPONENTS of the objective's local curvature at `point`: the
    eigen-decomposition of the finite-difference Hessian, in NORMALIZED units (each
    param scaled by its grid_step). Cost = 1 + 2N + N(N-1)/2 objective evals.

    `evaluate(overrides)->{objective}`. Returns {components, sensitivity, n_evals, f0}."""
    names = list(spec)
    N = len(names)
    h = {n: float(spec[n].get("grid_step")
                  or (spec[n]["range"][1] - spec[n]["range"][0]) / 16.0)
         for n in names}
    lo = {n: spec[n]["range"][0] for n in names}
    hi = {n: spec[n]["range"][1] for n in names}

    total_evals = 1 + 2 * N + N * (N - 1) // 2
    prog = {"n": 0, "t0": time.perf_counter(), "last": 0.0, "phase": "baseline"}

    def ev(deltas):
        c = dict(point)
        for n, d in deltas.items():
            c[n] = _clamp(point[n] + d, lo[n], hi[n])
        o = evaluate(c)["objective"]
        prog["n"] += 1
        now = time.perf_counter()
        if now - prog["last"] >= 1.0 or prog["n"] == total_evals:
            prog["last"] = now
            el = now - prog["t0"]
            eta = el / prog["n"] * (total_evals - prog["n"]) if prog["n"] else 0.0
            print(f"    [PCA {prog['n']}/{total_evals} | {prog['phase']} | "
                  f"{_dur(el)} elapsed | ETA {_dur(eta)}] obj={o:.6f}", flush=True)
        return o

    f0 = ev({})
    prog["phase"] = "diagonal (+/-h per param)"
    fp = {n: ev({n: h[n]}) for n in names}
    fm = {n: ev({n: -h[n]}) for n in names}
    H = np.zeros((N, N))
    for i, n in enumerate(names):
        H[i, i] = (fp[n] - 2 * f0 + fm[n]) / (h[n] * h[n])
    prog["phase"] = "mixed partials (coupling)"
    for i in range(N):
        for j in range(i + 1, N):
            ni, nj = names[i], names[j]
            fpp = ev({ni: h[ni], nj: h[nj]})
            H[i, j] = H[j, i] = ((fpp - fp[ni] - fp[nj] + f0)
                                 / (h[ni] * h[nj]))
    hv = np.array([h[n] for n in names])
    Hn = H * np.outer(hv, hv)
    vals, vecs = np.linalg.eigh(Hn)
    order = np.argsort(-np.abs(vals))
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
    sensitivity = sorted(
        ({"param": n, "h": h[n],
          "gradient": (fp[n] - fm[n]) / 2.0,
          "curvature": fp[n] - 2 * f0 + fm[n],
          "max_step_delta": max(abs(fp[n] - f0), abs(fm[n] - f0))}
         for n in names),
        key=lambda s: -s["max_step_delta"])
    return {"f0": f0, "n_evals": total_evals,
            "components": comps, "sensitivity": sensitivity}


def build_spec(registry, kind, fit_params):
    """Merge the registry defaults for `kind` with a config's per-param overrides;
    init = the LIVE source value (so search starts from production).

    `fit_params` may be "all"/"*" (fit EVERY constant in the kind's catalog) or a
    {name: override} map selecting a subset."""
    base = registry.fittable(kind)
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
        s["init"] = float(registry.current(name))
        spec[name] = s
    return spec


# ---------------------------------------------------------------------------
# Parallel per-frame helpers (shared by feature evaluators)
# ---------------------------------------------------------------------------

def eval_frames(adapter, items, fn):
    """Apply per-frame `fn` over `items`, dropping None, order-preserving. Each trial
    passes its OWN immutable cfg into `fn` (no shared mutation), so this is
    bit-identical to a serial loop, just multi-core — except when already inside a
    parallel TRIAL worker (_trial_local.serial), where the frames run serially."""
    if getattr(_trial_local, "serial", False):
        return [r for r in (fn(it) for it in items) if r is not None]
    workers = adapter.proc_workers(len(items))
    return [r for r in adapter.map_frames(fn, items, workers) if r is not None]


def map_frames_prep(adapter, roll_id, label, fn, items):
    """Run a per-frame PREP `fn` (decode + heavy trial-invariant precompute) over
    `items` in the analysis thread pool, dropping None, with progress logging."""
    if not items:
        return []
    n = len(items)
    state = {"done": 0}

    def _tick():
        state["done"] += 1
        d = state["done"]
        if d % 5 == 0 or d == n:
            print(f"    [{roll_id}] {label}: prepped {d}/{n} frame(s)", flush=True)

    results = adapter.map_frames(fn, items, adapter.proc_workers(n), on_done=_tick)
    return [r for r in results if r is not None]


def per_roll_summary(per_frame, key, worst_n=5, worst_desc=True):
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


# ---------------------------------------------------------------------------
# Session writing
# ---------------------------------------------------------------------------

def _git_commit(cwd):
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           cwd=str(cwd), capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except Exception:
        return None


def _session_dir(calib_dir, kind):
    calib_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    nn = sum(1 for p in calib_dir.glob(f"*_{kind}_*") if p.is_dir()) + 1
    d = calib_dir / f"{stamp}_{kind}_{nn:02d}"
    d.mkdir(parents=True)
    return d


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _fmtv(v):
    """Compact value formatting — %g for floats so a small epsilon isn't 0.0000."""
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _method_params(fit):
    """The EFFECTIVE hyperparameters of the chosen fit method (defaults filled in)."""
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


def _method_desc(fit):
    """One-line 'method(k=v, …)' for the index row and report header."""
    params = _method_params(fit)
    if not params:
        return fit.get("method", "none")
    inner = ", ".join(f"{k}={_fmtv(v)}" for k, v in params.items())
    return f"{fit.get('method', 'none')}({inner})"


def _write_report(adapter, path, config, results):
    kind = config["kind"]
    agg = results["aggregate"]
    lines = []
    lines.append(f"# Calibration session — {kind}\n")
    if config.get("comment"):
        lines.append(f"- comment: {config['comment']}")
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

        sens = pca.get("sensitivity")
        if sens:
            lines.append("### Per-param sensitivity (freeze signal)\n")
            lines.append("Central-difference gradient + diagonal curvature in "
                         "grid-step units, and the largest objective change from a "
                         "single ±grid_step move. Sorted loudest-first — params near "
                         "the bottom (max Δobj ≈ 0) are flat and can be FROZEN.\n")
            lines.append("| param | grid_step | gradient | curvature | max Δobj |")
            lines.append("|---|---|---|---|---|")
            for s in sens:
                lines.append(f"| {s['param']} | {s['h']:g} | "
                             f"{s['gradient']:+.6f} | {s['curvature']:+.6f} | "
                             f"{s['max_step_delta']:.6f} |")
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
    sort_key, desc = adapter.worst_key(kind)
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


def _append_index(adapter, kind, config, results, session_name):
    idx = adapter.calib_dir / f"INDEX_{kind}.md"
    if not idx.is_file():
        idx.write_text(
            f"# Calibration history — {kind}\n\n"
            "Comparable rows (one metric per kind; see calibrations/README.md).\n\n"
            "| session | comment | rolls | method | objective | headline | wall_s | commit |\n"
            "|---|---|---|---|---|---|---|---|\n", encoding="utf-8")
    comment = (config.get("comment") or "").replace("|", "\\|").replace("\n", " ")
    row = (f"| [{session_name}]({session_name}/report.md) | {comment} "
           f"| {','.join(config['rolls'])} | {_method_desc(config['fit'])} "
           f"| {_fmt(results['objective_final'])} "
           f"| {adapter.headline(kind, results['aggregate'])} "
           f"| {results['wall_seconds']} | {config.get('git_commit')} |\n")
    with open(idx, "a", encoding="utf-8") as f:
        f.write(row)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_session(adapter, config):
    kind = config["kind"]
    if kind not in adapter.evaluators:
        raise ValueError(f"unknown kind {kind!r}")
    reg = adapter.registry
    fit = config["fit"]
    spec = build_spec(reg, kind, fit.get("params"))
    if fit.get("method", "none") != "none" and not spec:
        raise ValueError("fit.method is a search but fit.params is empty")

    rolls = adapter.discover_rolls(config.get("rolls") or None)
    if not rolls:
        print(f"SKIP: no {kind} rolls with local source images. "
              "See the feature's fixtures README to repopulate a roll.")
        return None
    config["rolls"] = [r["id"] for r in rolls]
    config["metric"] = adapter.metric_name[kind]
    config["created"] = datetime.now().isoformat(timespec="seconds")
    config["git_commit"] = _git_commit(adapter.calib_dir)

    print("=" * 70)
    print(f"CALIBRATION  kind={kind}  method={_method_desc(fit)}  "
          f"rolls={len(rolls)} ({', '.join(config['rolls'])})  "
          f"params={len(spec)}")
    print(f"  metric: {config['metric']}")
    print("=" * 70, flush=True)

    snap = reg.snapshot(list(spec))   # restore the live module no matter what
    t_wall = time.perf_counter()
    try:
        evaluate, prep_secs = adapter.evaluators[kind](
            rolls, fit.get("tolerances") or {}, list(spec))
        init_overrides = {n: s["init"] for n, s in spec.items()} if spec else {}
        init_result = evaluate(init_overrides) if spec else evaluate({})
        objective_initial = init_result["objective"]

        if fit.get("method", "none") == "none" or not spec:
            best, trials = init_overrides, 1
            trace = {"method": "none", "trials": 1,
                     "best_objective": objective_initial,
                     "improvements": [{"trial": 1, "objective": objective_initial}]}
            final = init_result
        else:
            best, best_obj, trials, trace = optimize(
                lambda o: evaluate(o)["objective"], spec, fit, adapter.proc_workers)
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

    session = _session_dir(adapter.calib_dir, kind)
    (session / "config.json").write_text(json.dumps(config, indent=2))
    (session / "results.json").write_text(json.dumps(results, indent=2))
    (session / "fitted_params.json").write_text(json.dumps(
        {"kind": kind, "git_commit": config["git_commit"],
         "fitted": results["fitted"], "init": results["init"]}, indent=2))
    # Complete DROP-IN preset = DEFAULT_TUNING + the fitted overrides. Adopting the
    # result is then `cp fitted_preset.json presets/<name>.json` — no source edits.
    adapter.schema.dump(reg.to_tuning(results["fitted"]),
                        str(session / "fitted_preset.json"))
    _write_report(adapter, session / "report.md", config, results)
    _append_index(adapter, kind, config, results, session.name)

    print(f"\nSession written: {session}")
    print(f"  objective {_fmt(objective_initial)} -> {_fmt(final['objective'])} "
          f"({trials} trial(s), {wall}s)")
    print(f"  {adapter.headline(kind, final['aggregate'])}")
    if spec:
        for n, v in best.items():
            print(f"  {n}: {_fmt(results['init'][n])} -> {_fmt(v)}")
    return session


# ---------------------------------------------------------------------------
# Config + CLI
# ---------------------------------------------------------------------------

def load_config(adapter, args):
    if args.config:
        config = json.loads(Path(args.config).read_text())
    else:
        if not args.kind:
            raise SystemExit("provide --config or --kind")
        config = {"kind": args.kind, "rolls": [],
                  "fit": {"method": "none", "params": {}}}
    config.setdefault("fit", {})
    config["fit"].setdefault("params", {})
    config.setdefault("comment", "")
    if getattr(args, "comment", None) is not None:
        config["comment"] = args.comment
    if args.kind:
        config["kind"] = args.kind
    if args.method:
        config["fit"]["method"] = args.method
    if args.rolls:
        config["rolls"] = args.rolls
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


def run_main(adapter):
    ap = argparse.ArgumentParser(description=adapter.description)
    ap.add_argument("--config", help="path to a session config JSON")
    ap.add_argument("--kind", choices=list(adapter.evaluators),
                    help="override / set the calibration kind")
    ap.add_argument("--method", choices=["none", "grid", "coordinate_descent",
                                         "random_search"],
                    help="override the fitting method")
    ap.add_argument("--rolls", nargs="+", help="restrict to these roll ids")
    ap.add_argument("--comment", help="a free-text note about this session "
                    "(stored in config.json, shown in report.md + INDEX_<kind>.md)")
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
                    help="random_search: parallel trial threads (0/omit = auto; "
                         "1 = serial)")
    ap.add_argument("--pca", action="store_true",
                    help="after the fit, compute the principal components of the "
                         "metric's curvature at the optimum (O(N^2) extra evals)")
    ap.add_argument("--review", metavar="SESSION_DIR",
                    help="open the debug UI comparing fitted vs live for a session")
    ap.add_argument("--review-roll", help="roll id to review (default: first)")
    args = ap.parse_args()

    if args.review:
        adapter.review(args.review, args.review_roll)
        return 0

    config = load_config(adapter, args)
    try:
        run_session(adapter, config)
    except KeyboardInterrupt:
        sys.stdout.flush()
        sys.stderr.flush()
        print("\nInterrupted — calibration aborted (no session written).",
              file=sys.stderr, flush=True)
        os._exit(130)
    return 0
