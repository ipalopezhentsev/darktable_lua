# Calibration session — vignette

- created: 2026-06-19T14:43:10
- git commit: 8be3bf8
- rolls: 2506-1, 2510-11-1, 2511-12-1, 2512-2601-1
- metric: vignette_fit_rejected_count + median_residual
- fit method: coordinate_descent
- method params: epsilon=0.0005, max_iters=10, init_step=per-param/auto, step_min=per-param/auto, step_shrink=0.5
- fitted params: a, l, l
- trial count: 121
- wall time: 47.65s (prep 11.79s)
- objective: 0.0067 (init) -> 0.0067 (final)

## Convergence

- method: coordinate_descent  trials: 121
- cycles run: 1 (of max 10); converged early: True (epsilon 0.0005)
- per-cycle objective (improvement):
  - cycle 1: 0.0067 (−0.0000)
- best-so-far improved 1 time(s); curve (trial: objective):
  - 1: 0.0067

## Main contributors (objective drop credited per param)

Free attribution from coordinate descent's accepted moves — which single params drove convergence.

_(no accepted moves — the start was already optimal for these params on these rolls.)_

## Fitted constants (record-only — adopt by hand)

| constant | init | fitted |
|---|---|---|
| BORDER_DARK_THR | 0.0200 | 0.0200 |
| BORDER_PAD_FRAC | 0.0040 | 0.0040 |
| VIG_DOWNSAMPLE_FRAC | 0.0036 | 0.0036 |
| VIG_INSET_FRAC | 0.0071 | 0.0071 |
| VIG_BINS | 22.0000 | 22.0000 |
| VIG_PROFILE_PCT | 98.7773 | 98.7773 |
| VIG_MIN_BIN_SAMPLES | 25.0000 | 25.0000 |
| VIG_MIN_STRENGTH | 0.0200 | 0.0200 |
| VIG_PEAK_CENTER_FRAC | 0.1500 | 0.1500 |
| VIG_TAIL_CUT_REL | 0.9700 | 0.9700 |

## Aggregate (all rolls)

- rejected_rolls: 0
- median_residual: 0.0067
- max_residual: 0.0456
- n_frames: 4

## Per-roll

### 2506-1  (prep 2.75s)
```
{
  "n": 1,
  "rejected": false,
  "residual": 0.0456,
  "corner_falloff": 0.5891
}
```
### 2510-11-1  (prep 2.95s)
```
{
  "n": 1,
  "rejected": false,
  "residual": 0.0058,
  "corner_falloff": 0.5175
}
```
### 2511-12-1  (prep 3.01s)
```
{
  "n": 1,
  "rejected": false,
  "residual": 0.0061,
  "corner_falloff": 0.4737
}
```
### 2512-2601-1  (prep 3.08s)
```
{
  "n": 1,
  "rejected": false,
  "residual": 0.0073,
  "corner_falloff": 0.5221
}
```

## Worst frames (find where to look)

| roll | stem | rejected | residual | corner_falloff | reason | strength |
|---|---|---|---|---|---|---|
| 2506-1 | (roll) | False | 0.0456 | 0.5891 | None | 0.7750 |
| 2512-2601-1 | (roll) | False | 0.0073 | 0.5221 | None | 0.5750 |
| 2511-12-1 | (roll) | False | 0.0061 | 0.4737 | None | 0.4500 |
| 2510-11-1 | (roll) | False | 0.0058 | 0.5175 | None | 0.5500 |
