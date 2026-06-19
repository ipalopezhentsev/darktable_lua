# Calibration session — crop

- created: 2026-06-19T17:23:20
- git commit: 8be3bf8
- rolls: 2506-1, 2510-11-1, 2511-12-1, 2512-2601-1
- metric: crop_overtrim_area_fraction (containment HARD)
- fit method: coordinate_descent
- method params: epsilon=2e-06, max_iters=15, init_step=per-param/auto, step_min=per-param/auto, step_shrink=0.5
- fitted params: a, l, l
- trial count: 902
- wall time: 672.6s (prep 102.9s)
- objective: 34.6428 (init) -> 2.8754 (final)

## Convergence

- method: coordinate_descent  trials: 902
- cycles run: 4 (of max 15); converged early: True (epsilon 2e-06)
- per-cycle objective (improvement):
  - cycle 1: 3.1542 (−31.4886)
  - cycle 2: 2.8773 (−0.2769)
  - cycle 3: 2.8754 (−0.0019)
  - cycle 4: 2.8754 (−0.0000)
- best-so-far improved 55 time(s); curve (trial: objective):
  - 1: 34.6428
  - 2: 28.1390
  - 3: 18.7449
  - 4: 13.6832
  - 5: 12.2513
  - 6: 11.3810
  - 7: 10.5862
  - 8: 9.9752
  - 9: 9.6347
  - 77: 9.6320
  - 79: 9.6297
  - 87: 9.5889
  - 89: 9.5610
  - 94: 9.5560
  - 100: 9.5505
  - 111: 9.5500
  - 114: 9.5491
  - 144: 8.4235
  - 146: 7.3867
  - 148: 6.6197
  - 150: 6.3124
  - 155: 6.2900
  - 165: 6.1883
  - 167: 6.1030
  - 169: 5.9390
  - 173: 5.9283
  - 181: 5.9124
  - 187: 5.8657
  - 200: 5.7949
  - 201: 5.7320
  - 206: 5.7309
  - 216: 5.6870
  - 230: 3.5496
  - 232: 3.2381
  - 235: 3.2147
  - 239: 3.2045
  - 242: 3.1891
  - 250: 3.1740
  - 252: 3.1590
  - 254: 3.1542
  - 327: 3.1486
  - 335: 3.1481
  - 339: 3.1476
  - 344: 3.0434
  - 345: 2.9829
  - 346: 2.9778
  - 360: 2.9768
  - 373: 2.9207
  - 412: 2.8978
  - 417: 2.8909
  - 423: 2.8808
  - 431: 2.8803
  - 454: 2.8780
  - 475: 2.8773
  - 486: 2.8754

## Main contributors (objective drop credited per param)

Free attribution from coordinate descent's accepted moves — which single params drove convergence.

| param | objective drop | % of total | moves |
|---|---|---|---|
| CROP_JUNK_LINE_FRAC | 25.0100 | 78.7% | 9 |
| CROP_PAD_FRAC | 3.2592 | 10.3% | 5 |
| HOLDER_LUMA_THR | 2.4986 | 7.9% | 6 |
| CROP_SHADOW_REL | 0.4174 | 1.3% | 8 |
| CROP_REBATE_WIDE_MAX_D | 0.2491 | 0.8% | 7 |
| CROP_SHADOW_CORE_FRAC | 0.1348 | 0.4% | 3 |
| CROP_REBATE_BAND_HUE_TOL | 0.0562 | 0.2% | 1 |
| CROP_SHADOW_MAX_FRAC | 0.0473 | 0.1% | 2 |
| CROP_GAP_TOL_FRAC | 0.0462 | 0.1% | 2 |
| BORDER_MAX_FRAC | 0.0350 | 0.1% | 3 |
| CROP_REBATE_HUE_TOL | 0.0115 | 0.0% | 5 |
| CROP_REBATE_WIDE_LINE_FRAC | 0.0023 | 0.0% | 3 |

## Fitted constants (record-only — adopt by hand)

| constant | init | fitted |
|---|---|---|
| CROP_JUNK_LINE_FRAC | 0.0222 | 0.0975 |
| CROP_LEAK_MARGIN_D | 0.0600 | 0.0600 |
| CROP_REBATE_MARGIN_D | 0.1300 | 0.1300 |
| CROP_REBATE_LINE_FRAC | 0.1500 | 0.1500 |
| CROP_REBATE_TERM_FRAC | 0.0400 | 0.0400 |
| CROP_REBATE_MAX_FRAC | 0.0400 | 0.0400 |
| CROP_REBATE_HUE_TOL | 0.0300 | 0.0241 |
| CROP_REBATE_WIDE_MAX_D | 0.3500 | 0.3871 |
| CROP_REBATE_WIDE_LINE_FRAC | 0.5000 | 0.4594 |
| CROP_REBATE_BAND_HUE_TOL | 0.0080 | 0.0094 |
| CROP_REBATE_WIDE_FRAC | 0.5500 | 0.5500 |
| CROP_PAD_FRAC | 0.0125 | 0.0050 |
| CROP_SHADOW_REL | 0.7305 | 0.6191 |
| CROP_SHADOW_MAX_FRAC | 0.0300 | 0.0253 |
| CROP_SHADOW_CORE_FRAC | 0.0105 | 0.0167 |
| CROP_GAP_TOL_FRAC | 0.0200 | 0.0181 |
| HOLDER_LUMA_THR | 0.0395 | 0.0255 |
| BORDER_MAX_FRAC | 0.1000 | 0.0800 |

## Aggregate (all rolls)

- total_overtrim_area: 1.6129
- median_overtrim_area: 0.0097
- max_overtrim_area: 0.1014
- total_undertrim_area: 0.2525
- max_undertrim_area: 0.0267
- containment_violations: 44
- containment_weight: 5.0000
- n_frames: 150

## Per-roll

### 2506-1  (prep 24.32s)
```
{
  "n": 38,
  "median_overtrim_area": 0.006257005508502515,
  "worst_frames": [
    {
      "stem": "DSC_0003",
      "overtrim_area": 0.011173598748449047
    },
    {
      "stem": "DSC_0035",
      "overtrim_area": 0.01039662416907926
    },
    {
      "stem": "DSC_0023",
      "overtrim_area": 0.010330689971408534
    },
    {
      "stem": "DSC_0031",
      "overtrim_area": 0.010153117189045333
    },
    {
      "stem": "DSC_0022",
      "overtrim_area": 0.009681088273902645
    }
  ]
}
```
### 2510-11-1  (prep 26.47s)
```
{
  "n": 38,
  "median_overtrim_area": 0.012410426893959828,
  "worst_frames": [
    {
      "stem": "DSC_0003",
      "overtrim_area": 0.024902522282761803
    },
    {
      "stem": "DSC_0017",
      "overtrim_area": 0.02074941708175241
    },
    {
      "stem": "DSC_0018",
      "overtrim_area": 0.019819220418022813
    },
    {
      "stem": "DSC_0010",
      "overtrim_area": 0.01827276378174582
    },
    {
      "stem": "DSC_0032",
      "overtrim_area": 0.018179481877086668
    }
  ]
}
```
### 2511-12-1  (prep 25.88s)
```
{
  "n": 37,
  "median_overtrim_area": 0.009446572320823817,
  "worst_frames": [
    {
      "stem": "DSC_0027",
      "overtrim_area": 0.021423744103384823
    },
    {
      "stem": "DSC_0014",
      "overtrim_area": 0.01965925506344668
    },
    {
      "stem": "DSC_0009",
      "overtrim_area": 0.015637643631655606
    },
    {
      "stem": "DSC_0004",
      "overtrim_area": 0.015139390887893883
    },
    {
      "stem": "DSC_0032",
      "overtrim_area": 0.013949203694712677
    }
  ]
}
```
### 2512-2601-1  (prep 26.23s)
```
{
  "n": 37,
  "median_overtrim_area": 0.010651744558930188,
  "worst_frames": [
    {
      "stem": "DSC_0029",
      "overtrim_area": 0.10135697074319829
    },
    {
      "stem": "DSC_0028",
      "overtrim_area": 0.04853019186851522
    },
    {
      "stem": "DSC_0035",
      "overtrim_area": 0.015806974639309967
    },
    {
      "stem": "DSC_0036",
      "overtrim_area": 0.014406622191053329
    },
    {
      "stem": "DSC_0037",
      "overtrim_area": 0.013496280711849573
    }
  ]
}
```

## Worst frames (find where to look)

| contained | overtrim_area | undertrim_area | overtrim_edges | roll | stem |
|---|---|---|---|---|---|
| False | 0.1014 | 0.0033 | {'left': 0.0025025025025025025, 'top': 0.05089820359281437, 'right': 0.057057057057057055, 'bottom': -0.0037425149700598802} | 2512-2601-1 | DSC_0029 |
| False | 0.0485 | 0.0047 | {'left': 0.002002002002002002, 'top': 0.005239520958083832, 'right': 0.04454454454454455, 'bottom': -0.005239520958083832} | 2512-2601-1 | DSC_0028 |
| True | 0.0249 | 0.0000 | {'left': 0.012012012012012012, 'top': 0.008982035928143712, 'right': 0.0025025025025025025, 'bottom': 0.0029940119760479044} | 2510-11-1 | DSC_0003 |
| True | 0.0214 | 0.0000 | {'left': 0.01001001001001001, 'top': 0.008982035928143712, 'right': 0.0015015015015015015, 'bottom': 0.002245508982035928} | 2511-12-1 | DSC_0027 |
| True | 0.0207 | 0.0000 | {'left': 0.0015015015015015015, 'top': 0.0074850299401197605, 'right': 0.001001001001001001, 'bottom': 0.011976047904191617} | 2510-11-1 | DSC_0017 |
| True | 0.0198 | 0.0000 | {'left': 0.0025025025025025025, 'top': 0.01122754491017964, 'right': 0.0035035035035035035, 'bottom': 0.0037425149700598802} | 2510-11-1 | DSC_0018 |
| True | 0.0197 | 0.0000 | {'left': 0.002002002002002002, 'top': 0.01721556886227545, 'right': 0.0, 'bottom': 0.0014970059880239522} | 2511-12-1 | DSC_0014 |
| False | 0.0183 | 0.0014 | {'left': 0.013013013013013013, 'top': -0.0014970059880239522, 'right': 0.0035035035035035035, 'bottom': 0.0029940119760479044} | 2510-11-1 | DSC_0010 |
| True | 0.0182 | 0.0000 | {'left': 0.006006006006006006, 'top': 0.006736526946107785, 'right': 0.002002002002002002, 'bottom': 0.004491017964071856} | 2510-11-1 | DSC_0032 |
| True | 0.0166 | 0.0000 | {'left': 0.003003003003003003, 'top': 0.010479041916167664, 'right': 0.001001001001001001, 'bottom': 0.0029940119760479044} | 2510-11-1 | DSC_0014 |
| True | 0.0160 | 0.0000 | {'left': 0.007007007007007007, 'top': 0.004491017964071856, 'right': 0.0025025025025025025, 'bottom': 0.0029940119760479044} | 2510-11-1 | DSC_0007 |
| True | 0.0158 | 0.0000 | {'left': 0.0015015015015015015, 'top': 0.005239520958083832, 'right': 0.0025025025025025025, 'bottom': 0.0074850299401197605} | 2512-2601-1 | DSC_0035 |
| True | 0.0156 | 0.0000 | {'left': 0.002002002002002002, 'top': 0.012724550898203593, 'right': 0.001001001001001001, 'bottom': 0.0007485029940119761} | 2511-12-1 | DSC_0009 |
| True | 0.0154 | 0.0000 | {'left': 0.0035035035035035035, 'top': 0.005988023952095809, 'right': 0.003003003003003003, 'bottom': 0.0037425149700598802} | 2510-11-1 | DSC_0012 |
| True | 0.0151 | 0.0000 | {'left': 0.002002002002002002, 'top': 0.006736526946107785, 'right': 0.0035035035035035035, 'bottom': 0.0037425149700598802} | 2510-11-1 | DSC_0030 |
