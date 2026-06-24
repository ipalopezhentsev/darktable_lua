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
`--step-min` (coordinate_descent), `--n-trials`, `--seed` (random_search), plus
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
- **per-trial** for `random_search`: independent trials run in parallel threads
  (`--workers`, default = auto, capped at the analysis worker count; `1` =
  serial). Points are sampled up front and recorded in order, so a parallel run
  is bit-identical to the serial one (same best, same curve). Each trial then
  runs its frames serially to avoid nested pools.

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
