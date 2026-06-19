# Calibration session — crop

- created: 2026-06-19T17:56:25
- git commit: 8be3bf8
- rolls: 2506-1, 2510-11-1, 2511-12-1, 2512-2601-1
- metric: crop_overtrim_area_fraction (containment HARD)
- fit method: coordinate_descent
- method params: epsilon=2e-06, max_iters=15, init_step=per-param/auto, step_min=per-param/auto, step_shrink=0.5
- fitted params: a, l, l
- trial count: 879
- wall time: 593.28s (prep 98.92s)
- objective: 34.6470 (init) -> 4.8367 (final)

## Convergence

- method: coordinate_descent  trials: 879
- cycles run: 4 (of max 15); converged early: True (epsilon 2e-06)
- per-cycle objective (improvement):
  - cycle 1: 5.8052 (−28.8419)
  - cycle 2: 5.0326 (−0.7725)
  - cycle 3: 4.8367 (−0.1959)
  - cycle 4: 4.8367 (−0.0000)
- best-so-far improved 56 time(s); curve (trial: objective):
  - 1: 34.6470
  - 2: 28.3212
  - 3: 19.0520
  - 4: 14.0400
  - 5: 12.8225
  - 6: 11.9653
  - 7: 11.1807
  - 8: 10.6100
  - 9: 10.2720
  - 77: 10.2693
  - 79: 10.2670
  - 87: 10.1850
  - 91: 10.1700
  - 95: 10.1522
  - 101: 10.1467
  - 112: 10.1462
  - 115: 10.1453
  - 145: 9.3338
  - 147: 8.7703
  - 151: 8.7688
  - 154: 8.7318
  - 164: 8.7003
  - 166: 8.6829
  - 168: 8.5510
  - 174: 8.5499
  - 184: 8.4826
  - 190: 8.4793
  - 199: 8.3617
  - 200: 8.3268
  - 204: 8.3137
  - 216: 8.2712
  - 230: 6.2251
  - 232: 5.9815
  - 235: 5.8567
  - 247: 5.8353
  - 249: 5.8127
  - 251: 5.8052
  - 330: 5.6253
  - 331: 5.5904
  - 335: 5.5653
  - 339: 5.5630
  - 345: 5.5440
  - 387: 5.3924
  - 390: 5.3195
  - 394: 5.3143
  - 411: 5.3139
  - 416: 5.2440
  - 422: 5.2095
  - 431: 5.0979
  - 435: 5.0400
  - 438: 5.0345
  - 447: 5.0326
  - 574: 4.9511
  - 604: 4.9156
  - 617: 4.9145
  - 621: 4.8367

## Main contributors (objective drop credited per param)

Free attribution from coordinate descent's accepted moves — which single params drove convergence.

| param | objective drop | % of total | moves |
|---|---|---|---|
| CROP_JUNK_LINE_FRAC | 24.3750 | 81.8% | 8 |
| HOLDER_LUMA_THR | 2.4145 | 8.1% | 3 |
| CROP_PAD_FRAC | 1.6788 | 5.6% | 8 |
| CROP_REBATE_WIDE_MAX_D | 0.3815 | 1.3% | 9 |
| CROP_SHADOW_CORE_FRAC | 0.3407 | 1.1% | 6 |
| CROP_SHADOW_REL | 0.2612 | 0.9% | 7 |
| CROP_SHADOW_MAX_FRAC | 0.1749 | 0.6% | 4 |
| CROP_REBATE_BAND_HUE_TOL | 0.0816 | 0.3% | 1 |
| BORDER_MAX_FRAC | 0.0515 | 0.2% | 3 |
| CROP_GAP_TOL_FRAC | 0.0443 | 0.1% | 2 |
| CROP_REBATE_HUE_TOL | 0.0050 | 0.0% | 2 |
| CROP_REBATE_WIDE_LINE_FRAC | 0.0014 | 0.0% | 2 |

## Fitted constants (record-only — adopt by hand)

| constant | init | fitted |
|---|---|---|
| CROP_JUNK_LINE_FRAC | 0.0222 | 0.1000 |
| CROP_LEAK_MARGIN_D | 0.0600 | 0.0600 |
| CROP_REBATE_MARGIN_D | 0.1300 | 0.1300 |
| CROP_REBATE_LINE_FRAC | 0.1500 | 0.1500 |
| CROP_REBATE_TERM_FRAC | 0.0400 | 0.0400 |
| CROP_REBATE_MAX_FRAC | 0.0400 | 0.0400 |
| CROP_REBATE_HUE_TOL | 0.0300 | 0.0200 |
| CROP_REBATE_WIDE_MAX_D | 0.3500 | 0.3305 |
| CROP_REBATE_WIDE_LINE_FRAC | 0.5000 | 0.5094 |
| CROP_REBATE_BAND_HUE_TOL | 0.0080 | 0.0094 |
| CROP_REBATE_WIDE_FRAC | 0.5500 | 0.5500 |
| CROP_PAD_FRAC | 0.0125 | 0.0060 |
| CROP_SHADOW_REL | 0.7305 | 0.6156 |
| CROP_SHADOW_MAX_FRAC | 0.0300 | 0.0281 |
| CROP_SHADOW_CORE_FRAC | 0.0105 | 0.0167 |
| CROP_GAP_TOL_FRAC | 0.0200 | 0.0200 |
| HOLDER_LUMA_THR | 0.0395 | 0.0263 |
| BORDER_MAX_FRAC | 0.1000 | 0.0800 |

## Aggregate (all rolls)

- total_overtrim_area: 2.3805
- median_overtrim_area: 0.0146
- max_overtrim_area: 0.1027
- total_undertrim_area: 0.1637
- max_undertrim_area: 0.0241
- containment_violations: 25
- containment_weight: 15.0000
- n_frames: 150

## Per-roll

### 2506-1  (prep 22.93s)
```
{
  "n": 38,
  "median_overtrim_area": 0.010605852858846871,
  "worst_frames": [
    {
      "stem": "DSC_0032",
      "overtrim_area": 0.025942859026691363
    },
    {
      "stem": "DSC_0035",
      "overtrim_area": 0.015098931266595937
    },
    {
      "stem": "DSC_0023",
      "overtrim_area": 0.015007522492552433
    },
    {
      "stem": "DSC_0012",
      "overtrim_area": 0.014605174036311761
    },
    {
      "stem": "DSC_0022",
      "overtrim_area": 0.01438938938938939
    }
  ]
}
```
### 2510-11-1  (prep 25.55s)
```
{
  "n": 38,
  "median_overtrim_area": 0.016766392140643637,
  "worst_frames": [
    {
      "stem": "DSC_0003",
      "overtrim_area": 0.0969375063686441
    },
    {
      "stem": "DSC_0010",
      "overtrim_area": 0.031565997134859414
    },
    {
      "stem": "DSC_0032",
      "overtrim_area": 0.02333845821869774
    },
    {
      "stem": "DSC_0018",
      "overtrim_area": 0.02218460975946006
    },
    {
      "stem": "DSC_0030",
      "overtrim_area": 0.020319720918523312
    }
  ]
}
```
### 2511-12-1  (prep 24.92s)
```
{
  "n": 37,
  "median_overtrim_area": 0.01415187642732553,
  "worst_frames": [
    {
      "stem": "DSC_0027",
      "overtrim_area": 0.025398527269784754
    },
    {
      "stem": "DSC_0014",
      "overtrim_area": 0.022932738127348907
    },
    {
      "stem": "DSC_0034",
      "overtrim_area": 0.02288740836645028
    },
    {
      "stem": "DSC_0009",
      "overtrim_area": 0.018918619218020415
    },
    {
      "stem": "DSC_0004",
      "overtrim_area": 0.0189081296865728
    }
  ]
}
```
### 2512-2601-1  (prep 25.52s)
```
{
  "n": 37,
  "median_overtrim_area": 0.01581596566626507,
  "worst_frames": [
    {
      "stem": "DSC_0029",
      "overtrim_area": 0.10269326212439985
    },
    {
      "stem": "DSC_0028",
      "overtrim_area": 0.05783140925356494
    },
    {
      "stem": "DSC_0035",
      "overtrim_area": 0.03316977156797516
    },
    {
      "stem": "DSC_0036",
      "overtrim_area": 0.019104433775092456
    },
    {
      "stem": "DSC_0014",
      "overtrim_area": 0.018010900121678566
    }
  ]
}
```

## Worst frames (find where to look)

| contained | overtrim_area | undertrim_area | overtrim_edges | roll | stem |
|---|---|---|---|---|---|
| False | 0.1027 | 0.0020 | {'left': 0.004004004004004004, 'top': 0.05089820359281437, 'right': 0.057057057057057055, 'bottom': -0.002245508982035928} | 2512-2601-1 | DSC_0029 |
| True | 0.0969 | 0.0000 | {'left': 0.015015015015015015, 'top': 0.0561377245508982, 'right': 0.02952952952952953, 'bottom': 0.004491017964071856} | 2510-11-1 | DSC_0003 |
| False | 0.0578 | 0.0034 | {'left': 0.003003003003003003, 'top': 0.006736526946107785, 'right': 0.05205205205205205, 'bottom': -0.0037425149700598802} | 2512-2601-1 | DSC_0028 |
| True | 0.0332 | 0.0000 | {'left': 0.0025025025025025025, 'top': 0.006736526946107785, 'right': 0.005005005005005005, 'bottom': 0.020958083832335328} | 2512-2601-1 | DSC_0035 |
| True | 0.0316 | 0.0000 | {'left': 0.010510510510510511, 'top': 0.014221556886227544, 'right': 0.0045045045045045045, 'bottom': 0.004491017964071856} | 2510-11-1 | DSC_0010 |
| True | 0.0259 | 0.0000 | {'left': 0.001001001001001001, 'top': 0.019461077844311378, 'right': 0.004004004004004004, 'bottom': 0.0029940119760479044} | 2506-1 | DSC_0032 |
| True | 0.0254 | 0.0000 | {'left': 0.011011011011011011, 'top': 0.009730538922155689, 'right': 0.0025025025025025025, 'bottom': 0.0037425149700598802} | 2511-12-1 | DSC_0027 |
| True | 0.0233 | 0.0000 | {'left': 0.007007007007007007, 'top': 0.008233532934131737, 'right': 0.0035035035035035035, 'bottom': 0.005988023952095809} | 2510-11-1 | DSC_0032 |
| True | 0.0229 | 0.0000 | {'left': 0.003003003003003003, 'top': 0.01721556886227545, 'right': 0.001001001001001001, 'bottom': 0.0029940119760479044} | 2511-12-1 | DSC_0014 |
| True | 0.0229 | 0.0000 | {'left': 0.002002002002002002, 'top': 0.005988023952095809, 'right': 0.002002002002002002, 'bottom': 0.014221556886227544} | 2511-12-1 | DSC_0034 |
| True | 0.0222 | 0.0000 | {'left': 0.001001001001001001, 'top': 0.012724550898203593, 'right': 0.0045045045045045045, 'bottom': 0.005239520958083832} | 2510-11-1 | DSC_0018 |
| True | 0.0203 | 0.0000 | {'left': 0.003003003003003003, 'top': 0.008233532934131737, 'right': 0.005005005005005005, 'bottom': 0.005239520958083832} | 2510-11-1 | DSC_0030 |
| True | 0.0203 | 0.0000 | {'left': 0.0015015015015015015, 'top': 0.008233532934131737, 'right': 0.002002002002002002, 'bottom': 0.009730538922155689} | 2510-11-1 | DSC_0017 |
| True | 0.0203 | 0.0000 | {'left': 0.0075075075075075074, 'top': 0.005988023952095809, 'right': 0.0035035035035035035, 'bottom': 0.004491017964071856} | 2510-11-1 | DSC_0007 |
| True | 0.0201 | 0.0000 | {'left': 0.0045045045045045045, 'top': 0.0074850299401197605, 'right': 0.004004004004004004, 'bottom': 0.005239520958083832} | 2510-11-1 | DSC_0012 |
