# calibrations — recorded dust/stroke tuning sessions (auto_retouch)

Same machinery as `auto_negadoctor/tests/calibrations/` (the shared
`common/calibration/runner.py`). Every calibration is a **persisted folder** here
recording what was tuned, against which rolls, with which fitting procedure, and how
close it got — comparable over time.

Run with [`../run_calibration.py`](../run_calibration.py). The ground-truth
annotations it fits to live in [`../fixtures/rolls/<roll_id>/`](../fixtures/rolls/README.md).

## The two KINDS and their metrics (annotation-based)

| kind | metric (lower = closer) |
|------|-------------------------|
| `dust` | `W_FP · (detections reproducing an annotated false_positive) + W_MISS · (annotated missed_dust left uncovered)`, summed over the annotated frames. |
| `stroke` | `W_MISS · (annotated missed_strokes uncovered) + W_FP · (detected strokes on a false_positive point)`. |

**Precision is weighted higher** (`W_FP > W_MISS`, default 3:1) — the user prefers
few false positives over recall (missing a defect is fixed by hand in darktable; a
false heal is not). The weights live in the config's `fit.tolerances`
(`w_fp` / `w_miss`) so every session records them.

## Cost note

Unlike negadoctor (which caches a per-roll prefix), dust/stroke scoring **re-runs
full-resolution detection per frame per trial** — there is no cheaper invariant, so
a coordinate-descent fit over many frames is slow. Keep `fit.params` to a small,
impactful subset, calibrate on a focused set of annotated frames, and/or export the
roll at a smaller size . `RETOUCH_CALIB_WORKERS`
(default 3) sets the per-frame thread fan-out (higher risks OOM on full-res JPGs).

## Session comment

Pass `--comment "why I ran this"` (any kind/method) to annotate a session — it is
stored in `config.json` and shown in `report.md` (a `- comment:` line) and the
`INDEX_<kind>.md` table (a `comment` column). Handled by the shared runner, so it
works identically in auto_negadoctor.

## Adopting a result

A finished session writes `fitted_preset.json` (a complete drop-in preset). Adopt by
hand — `cp fitted_preset.json ../../presets/<name>.json` then run detection with
`RETOUCH_PRESET=<name>` — never by editing `detect_dust.py`.
