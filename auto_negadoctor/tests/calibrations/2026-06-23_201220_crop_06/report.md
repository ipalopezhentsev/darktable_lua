# Calibration session — crop

- created: 2026-06-23T20:04:13
- git commit: 9b6ca96
- rolls: 2506-1, 2510-11-1, 2511-12-1, 2512-2601-1
- metric: crop_overtrim_area_fraction (containment HARD)
- fit method: coordinate_descent
- method params: epsilon=2e-06, max_iters=15, init_step=per-param/auto, step_min=per-param/auto, step_shrink=0.5
- fitted params: a, l, l
- trial count: 610
- wall time: 487.54s (prep 109.02s)
- objective: 2.3159 (init) -> 2.1805 (final)

## Convergence

- method: coordinate_descent  trials: 610
- cycles run: 3 (of max 15); converged early: True (epsilon 2e-06)
- per-cycle objective (improvement):
  - cycle 1: 2.1823 (−0.1336)
  - cycle 2: 2.1805 (−0.0019)
  - cycle 3: 2.1805 (−0.0000)
- best-so-far improved 14 time(s); curve (trial: objective):
  - 1: 2.3159
  - 2: 2.2638
  - 5: 2.2593
  - 10: 2.2589
  - 13: 2.2584
  - 20: 2.2570
  - 23: 2.2561
  - 31: 2.2328
  - 35: 2.1990
  - 146: 2.1835
  - 197: 2.1828
  - 201: 2.1823
  - 396: 2.1814
  - 404: 2.1805

## Main contributors (objective drop credited per param)

Free attribution from coordinate descent's accepted moves — which single params drove convergence.

| param | objective drop | % of total | moves |
|---|---|---|---|
| CROP_JUNK_LINE_FRAC | 0.0575 | 42.5% | 4 |
| CROP_REBATE_MARGIN_D | 0.0571 | 42.1% | 2 |
| CROP_SHADOW_REL | 0.0155 | 11.5% | 1 |
| HOLDER_LUMA_THR | 0.0030 | 2.2% | 4 |
| CROP_LEAK_MARGIN_D | 0.0023 | 1.7% | 2 |

## Fitted constants (record-only — adopt by hand)

| constant | init | fitted |
|---|---|---|
| CROP_JUNK_LINE_FRAC | 0.0800 | 0.0969 |
| CROP_LEAK_MARGIN_D | 0.0300 | 0.0342 |
| CROP_REBATE_MARGIN_D | 0.1456 | 0.1269 |
| CROP_REBATE_LINE_FRAC | 0.1827 | 0.1827 |
| CROP_REBATE_TERM_FRAC | 0.0400 | 0.0400 |
| CROP_REBATE_MAX_FRAC | 0.0269 | 0.0269 |
| CROP_REBATE_HUE_TOL | 0.0200 | 0.0200 |
| CROP_REBATE_WIDE_MAX_D | 0.3871 | 0.3871 |
| CROP_REBATE_WIDE_LINE_FRAC | 0.8000 | 0.8000 |
| CROP_REBATE_BAND_HUE_TOL | 0.0094 | 0.0094 |
| CROP_REBATE_WIDE_FRAC | 0.5500 | 0.5500 |
| CROP_PAD_FRAC | 0.0050 | 0.0050 |
| CROP_SHADOW_REL | 0.7375 | 0.7469 |
| CROP_SHADOW_MAX_FRAC | 0.0309 | 0.0309 |
| CROP_SHADOW_CORE_FRAC | 0.0167 | 0.0167 |
| CROP_GAP_TOL_FRAC | 0.0153 | 0.0153 |
| HOLDER_LUMA_THR | 0.0241 | 0.0266 |
| BORDER_MAX_FRAC | 0.0800 | 0.0800 |

## Aggregate (all rolls)

- total_overtrim_area: 1.4887
- median_overtrim_area: 0.0087
- max_overtrim_area: 0.0848
- total_undertrim_area: 0.1384
- max_undertrim_area: 0.0118
- containment_violations: 38
- containment_weight: 5.0000
- n_frames: 150

## Per-roll

### 2506-1  (prep 25.91s)
```
{
  "n": 38,
  "median_overtrim_area": 0.0043849987712263165,
  "worst_frames": [
    {
      "stem": "DSC_0005",
      "overtrim_area": 0.018248413083742424
    },
    {
      "stem": "DSC_0011",
      "overtrim_area": 0.015926105746465028
    },
    {
      "stem": "DSC_0031",
      "overtrim_area": 0.009431587275898653
    },
    {
      "stem": "DSC_0017",
      "overtrim_area": 0.008668099236961513
    },
    {
      "stem": "DSC_0023",
      "overtrim_area": 0.00822379265493038
    }
  ]
}
```
### 2510-11-1  (prep 28.37s)
```
{
  "n": 38,
  "median_overtrim_area": 0.010895626165087243,
  "worst_frames": [
    {
      "stem": "DSC_0032",
      "overtrim_area": 0.028924133714552876
    },
    {
      "stem": "DSC_0018",
      "overtrim_area": 0.025402648157139177
    },
    {
      "stem": "DSC_0017",
      "overtrim_area": 0.023553493613373853
    },
    {
      "stem": "DSC_0010",
      "overtrim_area": 0.017237671803540067
    },
    {
      "stem": "DSC_0011",
      "overtrim_area": 0.015535745326164488
    }
  ]
}
```
### 2511-12-1  (prep 27.03s)
```
{
  "n": 37,
  "median_overtrim_area": 0.00992984001966038,
  "worst_frames": [
    {
      "stem": "DSC_0033",
      "overtrim_area": 0.08483033932135728
    },
    {
      "stem": "DSC_0014",
      "overtrim_area": 0.022716578854303403
    },
    {
      "stem": "DSC_0027",
      "overtrim_area": 0.019161002319684955
    },
    {
      "stem": "DSC_0010",
      "overtrim_area": 0.018073088058117998
    },
    {
      "stem": "DSC_0006",
      "overtrim_area": 0.016704578830327332
    }
  ]
}
```
### 2512-2601-1  (prep 27.71s)
```
{
  "n": 37,
  "median_overtrim_area": 0.00874527221832611,
  "worst_frames": [
    {
      "stem": "DSC_0035",
      "overtrim_area": 0.02642762523002044
    },
    {
      "stem": "DSC_0007",
      "overtrim_area": 0.018485551419683156
    },
    {
      "stem": "DSC_0012",
      "overtrim_area": 0.01809781338224452
    },
    {
      "stem": "DSC_0033",
      "overtrim_area": 0.014821707935480391
    },
    {
      "stem": "DSC_0036",
      "overtrim_area": 0.013472304639969311
    }
  ]
}
```

## Worst frames (find where to look)

| contained | overtrim_area | undertrim_area | overtrim_edges | roll | stem |
|---|---|---|---|---|---|
| False | 0.0848 | 0.0068 | {'left': -0.0025025025025025025, 'top': 0.0, 'right': -0.0055055055055055055, 'bottom': 0.08982035928143713} | 2511-12-1 | DSC_0033 |
| True | 0.0289 | 0.0000 | {'left': 0.017517517517517518, 'top': 0.006736526946107785, 'right': 0.002002002002002002, 'bottom': 0.004491017964071856} | 2510-11-1 | DSC_0032 |
| True | 0.0264 | 0.0000 | {'left': 0.0015015015015015015, 'top': 0.005239520958083832, 'right': 0.0025025025025025025, 'bottom': 0.0187125748502994} | 2512-2601-1 | DSC_0035 |
| True | 0.0254 | 0.0000 | {'left': 0.003003003003003003, 'top': 0.01122754491017964, 'right': 0.009009009009009009, 'bottom': 0.0037425149700598802} | 2510-11-1 | DSC_0018 |
| True | 0.0236 | 0.0000 | {'left': 0.002002002002002002, 'top': 0.0074850299401197605, 'right': 0.002002002002002002, 'bottom': 0.01347305389221557} | 2510-11-1 | DSC_0017 |
| True | 0.0227 | 0.0000 | {'left': 0.0015015015015015015, 'top': 0.019461077844311378, 'right': 0.0015015015015015015, 'bottom': 0.0014970059880239522} | 2511-12-1 | DSC_0014 |
| True | 0.0192 | 0.0000 | {'left': 0.0015015015015015015, 'top': 0.01347305389221557, 'right': 0.003003003003003003, 'bottom': 0.002245508982035928} | 2511-12-1 | DSC_0027 |
| True | 0.0185 | 0.0000 | {'left': 0.001001001001001001, 'top': 0.0074850299401197605, 'right': 0.001001001001001001, 'bottom': 0.010479041916167664} | 2512-2601-1 | DSC_0007 |
| True | 0.0182 | 0.0000 | {'left': 0.0, 'top': 0.0, 'right': 0.01951951951951952, 'bottom': 0.0} | 2506-1 | DSC_0005 |
| True | 0.0181 | 0.0000 | {'left': 0.011511511511511512, 'top': 0.005239520958083832, 'right': 0.001001001001001001, 'bottom': 0.0014970059880239522} | 2512-2601-1 | DSC_0012 |
| True | 0.0181 | 0.0000 | {'left': 0.0025025025025025025, 'top': 0.005239520958083832, 'right': 0.008008008008008008, 'bottom': 0.0037425149700598802} | 2511-12-1 | DSC_0010 |
| False | 0.0172 | 0.0014 | {'left': -0.0015015015015015015, 'top': 0.012724550898203593, 'right': 0.0025025025025025025, 'bottom': 0.0029940119760479044} | 2510-11-1 | DSC_0010 |
| True | 0.0167 | 0.0000 | {'left': 0.0025025025025025025, 'top': 0.005239520958083832, 'right': 0.001001001001001001, 'bottom': 0.008982035928143712} | 2511-12-1 | DSC_0006 |
| False | 0.0159 | 0.0030 | {'left': 0.014014014014014014, 'top': -0.002245508982035928, 'right': -0.001001001001001001, 'bottom': 0.0029940119760479044} | 2506-1 | DSC_0011 |
| True | 0.0155 | 0.0000 | {'left': 0.002002002002002002, 'top': 0.008982035928143712, 'right': 0.0025025025025025025, 'bottom': 0.0029940119760479044} | 2510-11-1 | DSC_0011 |
