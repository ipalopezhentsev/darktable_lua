# Calibration session — vignette

- created: 2026-06-23T15:51:09
- git commit: 9b6ca96
- rolls: 2506-1, 2510-11-1, 2511-12-1, 2512-2601-1
- metric: vignette_fit_rejected_count + median_residual
- fit method: none
- trial count: 1
- wall time: 22.87s (prep 22.61s)
- objective: 0.0067 (init) -> 0.0067 (final)

## Convergence

- method: none  trials: 1
- best-so-far improved 1 time(s); curve (trial: objective):
  - 1: 0.0067

## Aggregate (all rolls)

- rejected_rolls: 0
- median_residual: 0.0067
- max_residual: 0.0452
- n_frames: 4

## Per-roll

### 2506-1  (prep 4.84s)
```
{
  "n": 1,
  "rejected": false,
  "residual": 0.0452,
  "corner_falloff": 0.5876
}
```
### 2510-11-1  (prep 5.62s)
```
{
  "n": 1,
  "rejected": false,
  "residual": 0.0059,
  "corner_falloff": 0.5175
}
```
### 2511-12-1  (prep 6.31s)
```
{
  "n": 1,
  "rejected": false,
  "residual": 0.0061,
  "corner_falloff": 0.4737
}
```
### 2512-2601-1  (prep 5.84s)
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
| 2506-1 | (roll) | False | 0.0452 | 0.5876 | None | 0.7500 |
| 2512-2601-1 | (roll) | False | 0.0073 | 0.5221 | None | 0.5750 |
| 2511-12-1 | (roll) | False | 0.0061 | 0.4737 | None | 0.4500 |
| 2510-11-1 | (roll) | False | 0.0059 | 0.5175 | None | 0.5500 |
