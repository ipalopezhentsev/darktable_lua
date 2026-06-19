# Calibration session — crop

- created: 2026-06-19T14:54:08
- git commit: 8be3bf8
- rolls: 2506-1, 2510-11-1, 2511-12-1, 2512-2601-1
- metric: crop_overtrim_area_fraction (containment HARD)
- fit method: coordinate_descent
- method params: epsilon=0.0002, max_iters=10, init_step=per-param/auto, step_min=per-param/auto, step_shrink=0.5
- fitted params: a, l, l
- trial count: 479
- wall time: 258.05s (prep 103.93s)
- objective: 11000021.0944 (init) -> 1000034.6183 (final)

## Convergence

- method: coordinate_descent  trials: 479
- cycles run: 3 (of max 10); converged early: True (epsilon 0.0002)
- per-cycle objective (improvement):
  - cycle 1: 1000034.6610 (−9999986.4334)
  - cycle 2: 1000034.6183 (−0.0427)
  - cycle 3: 1000034.6183 (−0.0000)
- best-so-far improved 19 time(s); curve (trial: objective):
  - 1: 11000021.0944
  - 3: 9000037.5266
  - 5: 5000069.3915
  - 7: 2000087.9576
  - 11: 2000086.0988
  - 14: 2000085.2594
  - 17: 2000085.0871
  - 80: 1000085.3029
  - 84: 1000085.1953
  - 95: 1000085.1867
  - 103: 1000085.1857
  - 107: 1000085.1850
  - 125: 1000085.1726
  - 127: 1000085.1686
  - 161: 1000084.7903
  - 167: 1000069.0365
  - 169: 1000049.6867
  - 171: 1000034.6610
  - 255: 1000034.6183

## Main contributors (objective drop credited per param)

Free attribution from coordinate descent's accepted moves — which single params drove convergence.

| param | objective drop | % of total | moves |
|---|---|---|---|
| CROP_JUNK_LINE_FRAC | 8999936.0073 | 90.0% | 6 |
| CROP_PAD_FRAC | 999999.9345 | 10.0% | 3 |
| BORDER_MAX_FRAC | 50.1293 | 0.0% | 3 |
| HOLDER_LUMA_THR | 0.3783 | 0.0% | 1 |
| CROP_SHADOW_CORE_FRAC | 0.0164 | 0.0% | 2 |
| CROP_SHADOW_REL | 0.0103 | 0.0% | 3 |

## Fitted constants (record-only — adopt by hand)

| constant | init | fitted |
|---|---|---|
| CROP_JUNK_LINE_FRAC | 0.0500 | 0.0222 |
| CROP_LEAK_MARGIN_D | 0.0600 | 0.0600 |
| CROP_REBATE_MARGIN_D | 0.1300 | 0.1300 |
| CROP_REBATE_LINE_FRAC | 0.1500 | 0.1500 |
| CROP_REBATE_TERM_FRAC | 0.0400 | 0.0400 |
| CROP_REBATE_MAX_FRAC | 0.0400 | 0.0400 |
| CROP_PAD_FRAC | 0.0120 | 0.0125 |
| CROP_SHADOW_REL | 0.7000 | 0.7305 |
| CROP_SHADOW_MAX_FRAC | 0.0300 | 0.0300 |
| CROP_SHADOW_CORE_FRAC | 0.0160 | 0.0105 |
| CROP_GAP_TOL_FRAC | 0.0200 | 0.0200 |
| HOLDER_LUMA_THR | 0.0400 | 0.0395 |
| BORDER_MAX_FRAC | 0.2500 | 0.1000 |

## Aggregate (all rolls)

- total_overtrim_area: 34.6183
- median_overtrim_area: 0.2531
- max_overtrim_area: 0.2678
- total_undertrim_area: 0.3303
- max_undertrim_area: 0.3299
- containment_violations: 1
- containment_weight: None
- n_frames: 150

## Per-roll

### 2506-1  (prep 23.75s)
```
{
  "n": 38,
  "median_overtrim_area": 0.18702722183261106,
  "worst_frames": [
    {
      "stem": "DSC_0022",
      "overtrim_area": 0.2561850772928617
    },
    {
      "stem": "DSC_0014",
      "overtrim_area": 0.25525488062913215
    },
    {
      "stem": "DSC_0033",
      "overtrim_area": 0.2550139960319601
    },
    {
      "stem": "DSC_0028",
      "overtrim_area": 0.2533596470722219
    },
    {
      "stem": "DSC_0021",
      "overtrim_area": 0.2521773270276264
    }
  ]
}
```
### 2510-11-1  (prep 25.74s)
```
{
  "n": 38,
  "median_overtrim_area": 0.25584547871224517,
  "worst_frames": [
    {
      "stem": "DSC_0021",
      "overtrim_area": 0.26210342078605553
    },
    {
      "stem": "DSC_0030",
      "overtrim_area": 0.26166286046525566
    },
    {
      "stem": "DSC_0005",
      "overtrim_area": 0.2602400304496113
    },
    {
      "stem": "DSC_0033",
      "overtrim_area": 0.2597627567687448
    },
    {
      "stem": "DSC_0019",
      "overtrim_area": 0.2595136303968639
    }
  ]
}
```
### 2511-12-1  (prep 26.36s)
```
{
  "n": 37,
  "median_overtrim_area": 0.2562068056080032,
  "worst_frames": [
    {
      "stem": "DSC_0032",
      "overtrim_area": 0.2604756702810595
    },
    {
      "stem": "DSC_0023",
      "overtrim_area": 0.2588097079115043
    },
    {
      "stem": "DSC_0011",
      "overtrim_area": 0.2587992183800567
    },
    {
      "stem": "DSC_0003",
      "overtrim_area": 0.2583369297441154
    },
    {
      "stem": "DSC_0029",
      "overtrim_area": 0.2583369297441154
    }
  ]
}
```
### 2512-2601-1  (prep 28.08s)
```
{
  "n": 37,
  "median_overtrim_area": 0.2531288773803744,
  "worst_frames": [
    {
      "stem": "DSC_0020",
      "overtrim_area": 0.2678284571997147
    },
    {
      "stem": "DSC_0024",
      "overtrim_area": 0.25762401623180065
    },
    {
      "stem": "DSC_0036",
      "overtrim_area": 0.256911102719486
    },
    {
      "stem": "DSC_0023",
      "overtrim_area": 0.2564390738043433
    },
    {
      "stem": "DSC_0031",
      "overtrim_area": 0.2564278350206494
    }
  ]
}
```

## Worst frames (find where to look)

| contained | overtrim_area | undertrim_area | overtrim_edges | roll | stem |
|---|---|---|---|---|---|
| True | 0.2678 | 0.0000 | {'left': 0.07357357357357357, 'top': 0.0718562874251497, 'right': 0.08508508508508508, 'bottom': 0.0748502994011976} | 2512-2601-1 | DSC_0020 |
| True | 0.2621 | 0.0000 | {'left': 0.08408408408408409, 'top': 0.07110778443113773, 'right': 0.07307307307307308, 'bottom': 0.07110778443113773} | 2510-11-1 | DSC_0021 |
| True | 0.2617 | 0.0000 | {'left': 0.07357357357357357, 'top': 0.07110778443113773, 'right': 0.07857857857857858, 'bottom': 0.07559880239520958} | 2510-11-1 | DSC_0030 |
| True | 0.2605 | 0.0000 | {'left': 0.08408408408408409, 'top': 0.07559880239520958, 'right': 0.06756756756756757, 'bottom': 0.07035928143712575} | 2511-12-1 | DSC_0032 |
| True | 0.2602 | 0.0000 | {'left': 0.07507507507507508, 'top': 0.0748502994011976, 'right': 0.07557557557557558, 'bottom': 0.0718562874251497} | 2510-11-1 | DSC_0005 |
| True | 0.2598 | 0.0000 | {'left': 0.08308308308308308, 'top': 0.0688622754491018, 'right': 0.06856856856856856, 'bottom': 0.07634730538922156} | 2510-11-1 | DSC_0033 |
| True | 0.2595 | 0.0000 | {'left': 0.08108108108108109, 'top': 0.07035928143712575, 'right': 0.07257257257257257, 'bottom': 0.07260479041916168} | 2510-11-1 | DSC_0019 |
| True | 0.2593 | 0.0000 | {'left': 0.07557557557557558, 'top': 0.07110778443113773, 'right': 0.07407407407407407, 'bottom': 0.07559880239520958} | 2510-11-1 | DSC_0026 |
| True | 0.2593 | 0.0000 | {'left': 0.08408408408408409, 'top': 0.06661676646706587, 'right': 0.06856856856856856, 'bottom': 0.07709580838323353} | 2510-11-1 | DSC_0032 |
| True | 0.2590 | 0.0000 | {'left': 0.07607607607607608, 'top': 0.0718562874251497, 'right': 0.07557557557557558, 'bottom': 0.07260479041916168} | 2510-11-1 | DSC_0014 |
| True | 0.2588 | 0.0000 | {'left': 0.07857857857857858, 'top': 0.07260479041916168, 'right': 0.07357357357357357, 'bottom': 0.07110778443113773} | 2510-11-1 | DSC_0025 |
| True | 0.2588 | 0.0000 | {'left': 0.08258258258258258, 'top': 0.0748502994011976, 'right': 0.06956956956956957, 'bottom': 0.0688622754491018} | 2511-12-1 | DSC_0023 |
| True | 0.2588 | 0.0000 | {'left': 0.07507507507507508, 'top': 0.07709580838323353, 'right': 0.07857857857857858, 'bottom': 0.06511976047904192} | 2511-12-1 | DSC_0011 |
| True | 0.2586 | 0.0000 | {'left': 0.07707707707707707, 'top': 0.07335329341317365, 'right': 0.07557557557557558, 'bottom': 0.06961077844311377} | 2510-11-1 | DSC_0012 |
| True | 0.2583 | 0.0000 | {'left': 0.07607607607607608, 'top': 0.07110778443113773, 'right': 0.07557557557557558, 'bottom': 0.07260479041916168} | 2511-12-1 | DSC_0003 |
