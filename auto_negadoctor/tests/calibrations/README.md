# calibrations — recorded, reproducible tuning sessions (spec 05)

Every calibration is a **persisted folder** here, not a fleeting chat output. It
records what was tuned, against which rolls, with which fitting procedure, and
**how close** the result got to the user's hand annotations — so sessions are
comparable over time and we always know the convergence metric.

Run with [`../run_calibration.py`](../run_calibration.py). The ground-truth
annotations it fits to live unchanged in
[`../fixtures/rolls/<roll_id>/`](../fixtures/rolls/README.md).

## The three KINDS and their metrics (algorithm-independent)

Crop, the inversion look, and vignette are independent problems with
incomparable metrics, so each is a **separate kind** with its own session folder
and its own history index (`INDEX_<kind>.md`). They are never mixed in one table.

| kind | metric (lower = closer) | hard constraint |
|------|-------------------------|-----------------|
| `vignette` | per roll: did `fit_vignette_profile` produce a **valid** fit? + its **residual**. objective = `(#rolls rejected)·1e6 + median residual`. | every selected roll must yield a non-rejected fit. |
| `crop` | **total over-trim** = Σ over annotated frames of the *fraction of frame area* of content the detected crop removed **inside** the user's hand-drawn rect. Per-edge over-trim reported as a fraction of the edge dimension — **never pixels**. | **containment** (configurable, see below): the detected crop must not extend outside the user rect. |
| `inversion` | median **histogram EMD** between the algorithm's render and the user's GT-param render over the content crop — `nega_model.histogram_distance`, a pure *picture-vs-picture* comparison (luma/color/highlight), **independent of how the params were derived**. `luma`/`color`/`hi999` also reported. | rendered hard-clip fraction ≤ `tolerances.clip_max_frac` (else `+1e6` per clipping frame). |

These compare the *rendered output / detected geometry / fitted envelope* against
the *annotation/data* — never the inversion math — so they stay valid as the
algorithm evolves (the whole point of spec 05).

#### Crop containment: HARD vs soft (`tolerances.containment_weight`)

The crop constants are **global** across every frame, so a HARD containment rule
(`+1e6` per frame that leaves any border) forces the optimizer to satisfy the
single worst-prone frame by over-cropping **all** of them — a corner solution
that loses content and underperforms hand-tuned params. To rebalance, set a
finite **`containment_weight`** (`W`) in the config's `tolerances`:

```jsonc
"tolerances": { "containment_weight": 3.0 }
```

This switches the objective to a **symmetric, area-based** form:

```
objective = total_overtrim_area + W · total_undertrim_area
```

`undertrim_area` is the mirror of `overtrim_area` — the fraction of frame area of
film **border the detector left in** (outside the user rect). Both are frame-area
fractions, so `W` is the literal exchange rate: *leaving 1 px² of border is W×
as bad as losing 1 px² of content.*

- **larger `W`** → border more expensive → crops inward → **more** over-trim
  (W→∞ reproduces the old HARD rule).
- **smaller `W`** → border cheaper → tolerates a sliver of border → **less**
  over-trim (W→0 ignores border entirely and under-crops).

`W = 1` is the neutral midpoint. To stop over-cropping, **sweep down** from there
(e.g. `3 → 1 → 0.3`) and pick the smallest `W` whose leftover border you still
find acceptable in `--review`. Omitting `containment_weight` keeps the original
HARD rule (so existing sessions reproduce). Ready-made config:
`configs/crop_descent_soft.json`. The report/index headline shows
`undertrim … (W=…, viol N)` for soft sessions, `viol N (HARD)` otherwise.

### Tuning order: crop + vignette FIRST, then inversion

Tune in this order, because **inversion depends on both**:

1. **crop** — the content crop defines the analysis area; every inversion picker
   (percentiles, wb, the print tune) runs INSIDE it, so a wrong crop lets holder
   junk poison the levels.
2. **vignette** — the level/color analysis runs on vignette-CORRECTED data
   (`load_frame(path, vignette)`), so the fitted vignette shifts Dmin / wb /
   exposure that inversion then measures.
3. **inversion** — only meaningful once the crop and vignette feeding it are
   right. Tuning inversion on a bad crop or vignette chases a moving target.

Crop and vignette are independent of each other (either order); both must be
settled before the inversion pass. Re-run inversion whenever you re-tune either.

### "Another roll fails to auto-find vignette" workflow

The vignette kind derives each roll's radial envelope from its **local TIFFs** at
runtime (`auto_negadoctor.vignette_envelope`, the accumulation half of
`estimate_vignette`) — no committed fixture, nothing to hand-produce. When a new
roll's estimator fails, just add the roll (its TIFFs) and re-run: the optimizer
searches `VIG_*` for constants that keep **all** rolls non-rejected, and the
report names exactly which roll still fails and by how much.

## Fitting algorithms (set in the config; all recorded)

| `fit.method` | hyperparameters | notes |
|---|---|---|
| `none` | — | evaluate the current constants once (no search). |
| `grid` | per-param `range` + `grid_step` | exhaustive Cartesian grid. |
| `coordinate_descent` | `epsilon`, `max_iters`, per-param `init_step`/`step_min`/`step_shrink` | cycle params, shrinking-step ± line search each; stop when a cycle improves < `epsilon` or `max_iters` cycles. |
| `random_search` | `n_trials`, `seed` | uniform samples in range; keep best. |
| `cmaes` | `sigma`, `popsize`, `max_iters`, `seed` | CMA-ES via the `cma` library — adapts a full covariance matrix, so it handles correlated/ill-conditioned objectives that trip up coordinate descent. Searches in range-normalized coords (one `sigma` fits all params); ~`popsize`×`max_iters` evals. |
| `spsa` | `a`, `c`, `alpha`, `gamma`, `A`, `max_iters`, `seed` | SPSA — stochastic gradient descent that estimates the gradient from just **2 evals per iteration regardless of param count** (a finite difference needs 2N), so it scales to many params. Noisy descent; the `a`/`c` gains must be tuned to the objective's scale. ~`2`×`max_iters` evals. |

### CMA-ES hyperparameters (`sigma` / `popsize` / `max_iters`)

CMA-ES runs in **generations**: each generation it samples `popsize` candidate
parameter sets from its current distribution, evaluates them, then uses the best
ones to update that distribution (mean, step size, covariance) for the next
generation. The search runs in coordinates where **each param is normalized to its
`[lo, hi]` range** (mapped to `[0, 1]`), which is what lets a single step size work
across params of wildly different scales.

- **`sigma`** — the **initial step size**: the standard deviation (in normalized
  `[0,1]` units) of the first generation's sampling spread around the start value.
  `sigma = 0.3` means the first candidates land ~±0.3 of each param's full range
  away from the start. It's only a *seed* — CMA-ES **adapts it automatically** each
  generation, shrinking it as it converges (watch the per-generation sigma in the
  report). Too small → stays near the start / converges prematurely; too large →
  early generations flail against the box bounds. `0.3` (~30% of the range) is
  Hansen's standard default. It's the rough analogue of coordinate descent's
  `init_step`, except CMA-ES adapts per-param scales *and* their correlations, not
  one shrinking scalar per axis.
- **`popsize`** (λ) — candidates evaluated **per generation**. More → each update is
  better-informed (more robust to noise & local minima) at the cost of more evals
  per step. The auto default is `4 + ⌊3·ln N⌋` (= 12 for crop's 18 params) — a sane
  floor; raise it for noisy/rugged objectives or if you have idle cores. **A whole
  generation's candidates are independent → they run in parallel across `--workers`**,
  so up to your worker count, extra `popsize` is nearly free in wall-clock.
- **`max_iters`** — number of **generations** (the stop cap). Generations are
  **sequential** (each needs the previous one's update), so this is pure serial time.
  Set it large enough that `sigma` decays to convergence before the cap: if sigma is
  still large at the last generation you stopped too early; if it flatlined near zero
  well before the end you could use fewer.

Total objective evaluations ≈ **`popsize × max_iters`** (e.g. 12 × 40 = 480). Key
asymmetry: `popsize` parallelizes, `max_iters` does not — so given spare cores,
spend budget on `popsize` rather than on more generations.

### SPSA hyperparameters (`a` / `c` / `alpha` / `gamma` / `A` / `max_iters`)

SPSA is **stochastic gradient descent** with a remarkably cheap gradient estimate:
each iteration it perturbs **all** params at once by a random ±`c_k` step (a
Rademacher ±1 vector), evaluates the **two** points `θ ± c_k·Δ`, and forms a
one-sample gradient `(y₊ − y₋)/(2·c_k·Δ)` — so it costs **2 evals per iteration no
matter how many params** (a finite-difference gradient would cost 2N). That makes it
scale to many params, but the descent is noisy and the gains must be tuned. Like
CMA-ES it runs in **range-normalized `[0,1]` coords** and clamps every evaluated
point into the box. It returns the **best point it ever evaluated** (the running
iterate `θ` is never evaluated and need not be the best).

The gains follow Spall's decaying schedules — `a_k = a/(k+1+A)^alpha`,
`c_k = c/(k+1)^gamma`:

- **`a`** — **step-size** gain numerator (in normalized coords). This is the lever
  you'll fight with: the step is `a_k · ĝ`, and `ĝ` has units of (objective change)
  per (param), so the right `a` depends on your **objective's magnitude**. Too small →
  crawls; too large → bounces off the box walls. Start by eyeballing the objective
  scale (the `--method none` baseline) and pick `a` so the early step moves a param a
  few % of its range; tune from there. (Default `0.05` assumes an O(1) objective.)
- **`c`** — **perturbation** gain numerator (normalized coords): how far the ± probe
  steps to measure the gradient. Want it large enough that `y₊ − y₋` rises above
  evaluation noise, small enough to stay local. Default `0.1` (~10% of range).
- **`alpha`** / **`gamma`** — the decay exponents for `a_k` / `c_k`. The theory
  defaults `alpha = 0.602`, `gamma = 0.101` are almost always what you want; leave
  them unless you know why.
- **`A`** — step-size **stability** constant: damps the first few (largest) steps so a
  bad early gradient can't fling the iterate. Default ≈ 10% of `max_iters`.
- **`max_iters`** — number of iterations (each = 2 evals + 1 update). Iterations are
  **sequential** (each needs the previous step), so this is serial time. SPSA usually
  needs *more* iterations than CMA-ES needs generations.

Total objective evaluations ≈ **`2 × max_iters`**. The two evals per iteration are
independent, so they run on `--workers` (capped at 2 — there are only two); the
serial cost is the iteration count. **Prefer CMA-ES for the small param counts here**
(crop's 18, etc.) — it auto-adapts and needs no scale tuning; reach for SPSA when the
fittable set is large or each eval is so costly that 2/iter is the only affordable
budget.

### `seed` (reproducibility) — and the CMA-ES `seed: 0` trap

`seed` fixes the RNG so a `random_search` / `cmaes` / `spsa` run is reproducible:
two runs with the same `seed` (and same objective) should give the **identical**
result. Two caveats:

- **⚠️ CMA-ES: never use `seed: 0` — it is NOT reproducible.** The `cma` library
  tests the seed with `if not opts['seed']:` and treats a falsy value (`0`, like
  unset/`'time'`) as "pick a fresh **time-derived** seed each run"
  (`evolution_strategy.py` ~L1100). So `seed: 0` silently re-randomizes every run —
  two CMA-ES sessions with `seed: 0` will **diverge** (a generation-N sample depends
  on the generation-(N−1) update, so any difference compounds). Use any **non-zero**
  seed (`1`, `42`, …) for a reproducible CMA-ES run. (`random_search` and `spsa` seed
  via Python's `random.Random` / numpy `default_rng`, which handle `0` correctly — the
  trap is CMA-ES-only.)
- CMA-ES also amplifies any non-bit-identical **objective** value: one flipped rank
  in a generation changes the covariance update and the whole trajectory. The crop
  metric bottoms out in integer crop edges and the analysis uses no `np.random`, so a
  non-zero seed is enough here; just be aware the seed is not the *only* thing that
  must be stable for byte-identical reruns.

[`../calibration_registry.py`](../calibration_registry.py) is the **complete
catalog** of every algorithm tuning constant, grouped by the kind (= metric) that
judges it: **crop** = the `detect_content_crop` knobs; **inversion** = everything
that shapes the picture — film base, density-range pickers, neutral-patch search,
**white balance** (the colors), print tune; **vignette** = the estimator. Tuple
constants (wb priors, the percentile bands) are exposed per element as `NAME[i]`;
integer constants are fit as integers.

**Overriding a hyperparameter for a one-off run** — you don't have to edit the
config JSON. Every method-level knob has a CLI flag that overrides the config:
`--method`, `--epsilon`, `--max-iters`, `--step-shrink`, `--init-step`,
`--step-min` (coordinate_descent), `--n-trials` (random_search), `--sigma`,
`--popsize` (cmaes), `--spsa-a`/`--spsa-c`/`--spsa-alpha`/`--spsa-gamma`/`--spsa-A`
(spsa), `--seed`/`--workers` (random_search + cmaes + spsa), plus
`--rolls` and `--pca`. The effective values (CLI > config > optimizer default)
are echoed in the report's "method params" line and the index row. Only the
**per-param** ranges/grid_steps stay in the config, since they're per-constant.

`fit.params` selects what to fit. The shipped default configs use the literal
**`"params": "all"`** — fit EVERY constant in that kind's catalog. To narrow,
replace it with a `{name: {…}}` map (`{}` = catalog default range for that param;
override `range`/`grid_step`/steps to tighten it). The search starts from the
**live source value** of each constant.

Note that fitting "all" with a full-path kind (any inversion wb/picker param, any
vignette accumulation param) re-runs the whole pipeline per trial, so a broad
coordinate-descent run is long — run `--method none` first for an instant
baseline over all params, then narrow to the levers that move the metric.
`grid` over more than ~2-3 params explodes combinatorially (the runner warns).

**Parallelism.** Each trial builds its OWN immutable tuning config
(`auto_negadoctor.Tuning`, via `calibration_registry.to_tuning`) and passes it
into the analysis functions — nothing mutates shared module globals — so two
things can run multi-core, bit-identically to a serial loop:
- **prep** (one-time, before any trial): the heavy trial-invariant precompute —
  decode + `_crop_fields` / the fixed GT render — runs frame-PARALLEL per roll
  (`_map_frames_prep`), instead of the old serial per-frame loop that idled the
  cores after `process_roll`'s parallel stages.
- **per-frame** within a trial: every evaluator spreads its frames (crop detect /
  inversion render+EMD / vignette refold) across the analysis thread pool
  (≈4–5× on a 38-frame roll's crop loop). Helps ALL methods.
- **per-trial** for `random_search`, `cmaes` and `spsa`: independent trials
  (random_search: all points; cmaes: each generation's `popsize` candidates; spsa:
  the 2 ± probes per iteration) run in parallel threads (`--workers`, default = auto,
  capped at the analysis worker count — and at 2 for spsa, which only has two; `1` =
  serial). Objectives are recorded — and, for cmaes, told back to the optimizer — in
  fixed index order, so a parallel run is bit-identical to the serial one (same best,
  same curve). Each trial then runs its frames serially to avoid nested pools.

How much per-trial parallelism helps depends on the kind: render-heavy inversion
(numpy releases the GIL) scales well; crop's `_crop_decide` is partly pure-Python
(GIL-bound) and its frames are already parallel, so trial-parallelism adds little
there. `coordinate_descent` is a dependent chain (each step needs the previous),
so it gets the per-frame parallelism but not trial-level.

**Worker count.** The analysis thread pool (`auto_negadoctor._proc_workers`)
defaults to `min(cpu, 8)` — the measured memory-bandwidth knee for the float
buffers. On a machine with the bandwidth headroom, raise it with the
`NEGA_PROC_WORKERS=<n>` env var (covers prep decode, per-frame, vignette, and the
trial pool). Watch wall time: the per-frame work is partly bandwidth-bound, so
past the knee more workers can stall on RAM rather than speed up.

**Cost — fast vs full path** (the runner dispatches automatically):
- crop: always fast. The decode AND the heavy trial-invariant pixel math
  (`detect_content_crop`'s full-res log10 density extremes + line means,
  `_crop_fields`) are precomputed ONCE per frame in prep; each trial re-runs only
  the cheap `_crop_decide` (the line-flag thresholds + 1-D edge scans that read
  the fitted crop constants) — ~16× cheaper per eval than a full re-detect,
  bit-identical.
- inversion: fitting only print-tune constants (`PRINT_GAMMA`/`PRINT_HI_CEIL`/
  `PRINT_HI_PCT`/`PRINT_CLIP_BUDGET`/`PRINT_TUNE_ITERS`) re-tunes on cached frames
  (fast). Fitting any wb / picker / film-base / patch constant re-runs the WHOLE
  pipeline per trial — correct but much slower; prefer `grid` with few steps or a
  small `max_iters`, and run `--method none` first for a baseline.
- vignette: fitting only the profile-fit constants (`VIG_MIN_STRENGTH`/
  `VIG_PEAK_CENTER_FRAC`/`VIG_TAIL_CUT_REL`) captures each roll's radial envelope
  from the TIFFs ONCE, then re-runs only `fit_vignette_profile` per trial (fast).
  Fitting an accumulation constant (`VIG_BINS`, `VIG_DOWNSAMPLE_FRAC`, …) re-folds
  the envelope per trial — but the TIFF **decode** is trial-invariant, so each
  frame's luma/clip/border is decoded ONCE into an in-RAM cache
  (`vignette_frame_cache`) and every trial re-folds from memory with no disk
  reads (~40× cheaper per eval than re-decoding, bit-identical). Still heavier
  than the profile-only path (it re-bins all pixels), and the cache is
  memory-hungry (a float64 luma per frame), so cap frames if RAM is tight.

## Session folder layout

```
calibrations/
    INDEX_crop.md  INDEX_inversion.md  INDEX_vignette.md   # per-kind leaderboards
    configs/{crop,inversion,vignette}_default.json         # copy + edit per session
    <YYYY-MM-DD_HHMMSS>_<kind>_<NN>/
        config.json        # INPUTS: kind, rolls, metric, git_commit, fit{...}
        results.json       # OUTPUTS: wall_seconds (+ per-roll prep), trial_count,
                           #   objective init->final, trace (convergence: per-
                           #   method, the best-so-far curve, + cycles/
                           #   converged_early for coordinate_descent), fitted{},
                           #   aggregate, per_roll (with worst_frames), per_frame
        report.md          # human-readable: config echo, aggregate + per-roll
                           #   tables, worst-frames list
        fitted_params.json # RECORD-ONLY best constants — adopt by hand into
                           #   auto_negadoctor.py (git stays the source of truth)
```

`--review` does NOT write into the session folder — it builds a transient
debug-UI session in a temp dir and deletes it when you close the window, so the
session stays a clean record of inputs/outputs.

`config.json` records the algorithm version (`git_commit`) and the full fit
inputs; `results.json` records the metric (aggregate + per-roll + per-frame) and
the **wall time** (so calibration cost can be optimized later, spec-05 point 6).

## Which params mattered (main contributors / principal components)

Every session records, in `results.json` + `report.md`, **which params drove
convergence**:

- **Main contributors** (free, coordinate descent only): each accepted move is
  credited to the param it changed, so the report ranks params by the objective
  drop they produced (with % of total). Answers "which single knobs moved the
  metric."
- **Principal components** (opt-in, `--pca` or `fit.pca: true`): the
  eigen-decomposition of the metric's local curvature (finite-difference
  Hessian) at the optimum, in grid-step-normalized units. Each component is a
  DIRECTION in param space — the loadings expose **coupled** params (e.g.
  exposure↔wb↔black moving together), and the eigenvalue ranks how sharply the
  metric changes along it. This is the "principal axes of the loss" view, not
  just per-param. Cost is `1 + 2N + N(N−1)/2` extra evaluations (O(N²)), so use
  it on a **small param subset**, not `all` on the full path.

## Comparing history

Open `INDEX_<kind>.md`: one row per session with its objective, headline metric,
wall time and commit. The `method` column carries the **full method spec**, not
just the name — e.g. `coordinate_descent(epsilon=0.0010, max_iters=10, …)` — so
two rows are reproducible/comparable without opening each session. `report.md`
echoes the same effective hyperparameters under "method params" (including the
optimizer defaults applied when the config omits a knob). To see *where* a
session is weak, sort the per-frame table in its `report.md` (worst frames are
pre-listed). Two sessions of the same kind are directly comparable; different
kinds never share a table. Each row also carries a free-text **comment** column
(see below), so the *reason* a session was run sits beside its numbers.

## Session comment

Pass `--comment "why I ran this"` to annotate a session — it is stored in
`config.json` and shown in `report.md` (a `- comment:` line) and the
`INDEX_<kind>.md` table (a `comment` column). Handled by the shared
`common/calibration/runner.py`, so it works identically in auto_retouch. NOTE: the
comment column was added to the index header on 2026-06-23; INDEX files that
predate it keep their old (column-short) header — regenerate or ignore the cosmetic
mismatch.

## Usage

```sh
# evaluate the current algorithm (no search) and record a session
conda run -n autocrop python auto_negadoctor/tests/run_calibration.py \
    --config auto_negadoctor/tests/calibrations/configs/inversion_default.json \
    --method none

# fit, restricted to two rolls, tighter epsilon
... --config .../inversion_default.json --method coordinate_descent \
    --rolls 2512-2601-1 2510-11-1 --epsilon 0.0003

# vignette fit (captures each roll's envelope from its TIFFs, then re-fits)
... --config .../vignette_default.json

# review ANY session (crop / vignette / inversion) in the debug UI:
#   key R flips FITTED (this session) <-> live (current source-code) result;
#   key N toggles the vignette correction on/off in the preview
... --review auto_negadoctor/tests/calibrations/2026-06-16_141500_inversion_01
... --review .../<crop-or-vignette-session> [--review-roll <roll_id>]
```

The runner **never edits the source** — adopt good fitted values by hand.
