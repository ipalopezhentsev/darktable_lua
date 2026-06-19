# Calibration session — crop

- created: 2026-06-19T17:11:08
- git commit: 8be3bf8
- rolls: 2506-1, 2510-11-1, 2511-12-1, 2512-2601-1
- metric: crop_overtrim_area_fraction (containment HARD)
- fit method: coordinate_descent
- method params: epsilon=0.0002, max_iters=10, init_step=per-param/auto, step_min=per-param/auto, step_shrink=0.5
- fitted params: a, l, l
- trial count: 642
- wall time: 491.7s (prep 105.8s)
- objective: 34.6407 (init) -> 24.4087 (final)

## Convergence

- method: coordinate_descent  trials: 642
- cycles run: 3 (of max 10); converged early: True (epsilon 0.0002)
- per-cycle objective (improvement):
  - cycle 1: 24.4106 (−10.2301)
  - cycle 2: 24.4087 (−0.0019)
  - cycle 3: 24.4087 (−0.0000)
- best-so-far improved 18 time(s); curve (trial: objective):
  - 1: 34.6407
  - 6: 33.5144
  - 9: 33.1354
  - 12: 32.7170
  - 78: 32.7166
  - 92: 32.7152
  - 94: 32.6985
  - 109: 32.6980
  - 169: 32.6933
  - 171: 32.6837
  - 173: 32.6810
  - 197: 32.6270
  - 213: 32.3422
  - 217: 32.1405
  - 223: 28.8699
  - 225: 25.5028
  - 227: 24.4106
  - 246: 24.4087

## Main contributors (objective drop credited per param)

Free attribution from coordinate descent's accepted moves — which single params drove convergence.

| param | objective drop | % of total | moves |
|---|---|---|---|
| BORDER_MAX_FRAC | 7.7299 | 75.5% | 3 |
| CROP_JUNK_LINE_FRAC | 1.9237 | 18.8% | 3 |
| HOLDER_LUMA_THR | 0.4864 | 4.8% | 2 |
| CROP_GAP_TOL_FRAC | 0.0540 | 0.5% | 1 |
| CROP_REBATE_WIDE_MAX_D | 0.0181 | 0.2% | 2 |
| CROP_SHADOW_MAX_FRAC | 0.0170 | 0.2% | 3 |
| CROP_LEAK_MARGIN_D | 0.0019 | 0.0% | 1 |
| CROP_REBATE_WIDE_LINE_FRAC | 0.0005 | 0.0% | 1 |
| CROP_REBATE_HUE_TOL | 0.0004 | 0.0% | 1 |

## Fitted constants (record-only — adopt by hand)

| constant | init | fitted |
|---|---|---|
| CROP_JUNK_LINE_FRAC | 0.0222 | 0.0266 |
| CROP_LEAK_MARGIN_D | 0.0600 | 0.0712 |
| CROP_REBATE_MARGIN_D | 0.1300 | 0.1300 |
| CROP_REBATE_LINE_FRAC | 0.1500 | 0.1500 |
| CROP_REBATE_TERM_FRAC | 0.0400 | 0.0400 |
| CROP_REBATE_MAX_FRAC | 0.0400 | 0.0400 |
| CROP_REBATE_HUE_TOL | 0.0300 | 0.0250 |
| CROP_REBATE_WIDE_MAX_D | 0.3500 | 0.2875 |
| CROP_REBATE_WIDE_LINE_FRAC | 0.5000 | 0.5250 |
| CROP_REBATE_BAND_HUE_TOL | 0.0080 | 0.0080 |
| CROP_REBATE_WIDE_FRAC | 0.5500 | 0.5500 |
| CROP_PAD_FRAC | 0.0125 | 0.0125 |
| CROP_SHADOW_REL | 0.7305 | 0.7305 |
| CROP_SHADOW_MAX_FRAC | 0.0300 | 0.0200 |
| CROP_SHADOW_CORE_FRAC | 0.0105 | 0.0105 |
| CROP_GAP_TOL_FRAC | 0.0200 | 0.0191 |
| HOLDER_LUMA_THR | 0.0395 | 0.0378 |
| BORDER_MAX_FRAC | 0.1000 | 0.0800 |

## Aggregate (all rolls)

- total_overtrim_area: 24.4087
- median_overtrim_area: 0.1863
- max_overtrim_area: 0.2018
- total_undertrim_area: 0.0004
- max_undertrim_area: 0.0004
- containment_violations: 0
- containment_weight: None
- n_frames: 150

## Per-roll

### 2506-1  (prep 25.25s)
```
{
  "n": 38,
  "median_overtrim_area": 0.11878982275688862,
  "worst_frames": [
    {
      "stem": "DSC_0038",
      "overtrim_area": 0.18070315824806843
    },
    {
      "stem": "DSC_0014",
      "overtrim_area": 0.15773220825616036
    },
    {
      "stem": "DSC_0009",
      "overtrim_area": 0.15731899564234894
    },
    {
      "stem": "DSC_0027",
      "overtrim_area": 0.15639891388394384
    },
    {
      "stem": "DSC_0037",
      "overtrim_area": 0.15583135530740322
    }
  ]
}
```
### 2510-11-1  (prep 27.36s)
```
{
  "n": 38,
  "median_overtrim_area": 0.18979140068211925,
  "worst_frames": [
    {
      "stem": "DSC_0021",
      "overtrim_area": 0.19604934275592958
    },
    {
      "stem": "DSC_0030",
      "overtrim_area": 0.19560878243512975
    },
    {
      "stem": "DSC_0005",
      "overtrim_area": 0.19418595241948536
    },
    {
      "stem": "DSC_0033",
      "overtrim_area": 0.19370867873861886
    },
    {
      "stem": "DSC_0019",
      "overtrim_area": 0.193459552366738
    }
  ]
}
```
### 2511-12-1  (prep 27.06s)
```
{
  "n": 37,
  "median_overtrim_area": 0.18920792049534566,
  "worst_frames": [
    {
      "stem": "DSC_0032",
      "overtrim_area": 0.19442159225093356
    },
    {
      "stem": "DSC_0023",
      "overtrim_area": 0.19275562988137837
    },
    {
      "stem": "DSC_0003",
      "overtrim_area": 0.19228285171398943
    },
    {
      "stem": "DSC_0019",
      "overtrim_area": 0.1922731114347881
    },
    {
      "stem": "DSC_0015",
      "overtrim_area": 0.19204234174294055
    }
  ]
}
```
### 2512-2601-1  (prep 26.13s)
```
{
  "n": 37,
  "median_overtrim_area": 0.1868346640053227,
  "worst_frames": [
    {
      "stem": "DSC_0020",
      "overtrim_area": 0.20177437916958876
    },
    {
      "stem": "DSC_0024",
      "overtrim_area": 0.19156993820167473
    },
    {
      "stem": "DSC_0036",
      "overtrim_area": 0.19085702468936003
    },
    {
      "stem": "DSC_0023",
      "overtrim_area": 0.19038499577421733
    },
    {
      "stem": "DSC_0031",
      "overtrim_area": 0.19037375699052345
    }
  ]
}
```

## Worst frames (find where to look)

| contained | overtrim_area | undertrim_area | overtrim_edges | roll | stem |
|---|---|---|---|---|---|
| True | 0.2018 | 0.0000 | {'left': 0.053553553553553554, 'top': 0.051646706586826345, 'right': 0.06506506506506507, 'bottom': 0.05464071856287425} | 2512-2601-1 | DSC_0020 |
| True | 0.1960 | 0.0000 | {'left': 0.06406406406406406, 'top': 0.05089820359281437, 'right': 0.05305305305305305, 'bottom': 0.05089820359281437} | 2510-11-1 | DSC_0021 |
| True | 0.1956 | 0.0000 | {'left': 0.053553553553553554, 'top': 0.05089820359281437, 'right': 0.05855855855855856, 'bottom': 0.05538922155688623} | 2510-11-1 | DSC_0030 |
| True | 0.1944 | 0.0000 | {'left': 0.06406406406406406, 'top': 0.05538922155688623, 'right': 0.047547547547547545, 'bottom': 0.050149700598802395} | 2511-12-1 | DSC_0032 |
| True | 0.1942 | 0.0000 | {'left': 0.055055055055055056, 'top': 0.05464071856287425, 'right': 0.05555555555555555, 'bottom': 0.051646706586826345} | 2510-11-1 | DSC_0005 |
| True | 0.1937 | 0.0000 | {'left': 0.06306306306306306, 'top': 0.048652694610778445, 'right': 0.04854854854854855, 'bottom': 0.0561377245508982} | 2510-11-1 | DSC_0033 |
| True | 0.1935 | 0.0000 | {'left': 0.06106106106106106, 'top': 0.050149700598802395, 'right': 0.052552552552552555, 'bottom': 0.05239520958083832} | 2510-11-1 | DSC_0019 |
| True | 0.1932 | 0.0000 | {'left': 0.05555555555555555, 'top': 0.05089820359281437, 'right': 0.05405405405405406, 'bottom': 0.05538922155688623} | 2510-11-1 | DSC_0026 |
| True | 0.1932 | 0.0000 | {'left': 0.06406406406406406, 'top': 0.04640718562874251, 'right': 0.04854854854854855, 'bottom': 0.05688622754491018} | 2510-11-1 | DSC_0032 |
| True | 0.1930 | 0.0000 | {'left': 0.056056056056056056, 'top': 0.051646706586826345, 'right': 0.05555555555555555, 'bottom': 0.05239520958083832} | 2510-11-1 | DSC_0014 |
| True | 0.1928 | 0.0000 | {'left': 0.05855855855855856, 'top': 0.05239520958083832, 'right': 0.053553553553553554, 'bottom': 0.05089820359281437} | 2510-11-1 | DSC_0025 |
| True | 0.1928 | 0.0000 | {'left': 0.06256256256256257, 'top': 0.05464071856287425, 'right': 0.04954954954954955, 'bottom': 0.048652694610778445} | 2511-12-1 | DSC_0023 |
| True | 0.1925 | 0.0000 | {'left': 0.057057057057057055, 'top': 0.0531437125748503, 'right': 0.05555555555555555, 'bottom': 0.04940119760479042} | 2510-11-1 | DSC_0012 |
| True | 0.1923 | 0.0000 | {'left': 0.056056056056056056, 'top': 0.05089820359281437, 'right': 0.05555555555555555, 'bottom': 0.05239520958083832} | 2511-12-1 | DSC_0003 |
| True | 0.1923 | 0.0000 | {'left': 0.057057057057057055, 'top': 0.04640718562874251, 'right': 0.056056056056056056, 'bottom': 0.05538922155688623} | 2511-12-1 | DSC_0019 |
