# Calibration session — crop

- created: 2026-06-19T20:01:30
- git commit: 8be3bf8
- rolls: 2506-1, 2510-11-1, 2511-12-1, 2512-2601-1
- metric: crop_overtrim_area_fraction (containment HARD)
- fit method: coordinate_descent
- method params: epsilon=2e-06, max_iters=15, init_step=per-param/auto, step_min=per-param/auto, step_shrink=0.5
- fitted params: a, l, l
- trial count: 1042
- wall time: 802.03s (prep 108.5s)
- objective: 3.0286 (init) -> 2.2047 (final)

## Convergence

- method: coordinate_descent  trials: 1042
- cycles run: 5 (of max 15); converged early: True (epsilon 2e-06)
- per-cycle objective (improvement):
  - cycle 1: 2.4972 (−0.5314)
  - cycle 2: 2.3279 (−0.1693)
  - cycle 3: 2.2514 (−0.0765)
  - cycle 4: 2.2047 (−0.0467)
  - cycle 5: 2.2047 (−0.0000)
- best-so-far improved 59 time(s); curve (trial: objective):
  - 1: 3.0286
  - 2: 3.0277
  - 3: 3.0267
  - 5: 3.0192
  - 7: 3.0002
  - 21: 2.9932
  - 23: 2.9908
  - 25: 2.9885
  - 32: 2.9643
  - 35: 2.9453
  - 39: 2.9021
  - 43: 2.9002
  - 46: 2.8941
  - 76: 2.8685
  - 80: 2.8401
  - 92: 2.7759
  - 111: 2.7745
  - 112: 2.7505
  - 113: 2.7472
  - 114: 2.7440
  - 115: 2.7420
  - 116: 2.7369
  - 117: 2.7360
  - 152: 2.6995
  - 163: 2.6653
  - 169: 2.6415
  - 174: 2.6390
  - 177: 2.5240
  - 182: 2.5204
  - 185: 2.5185
  - 195: 2.5156
  - 198: 2.5089
  - 210: 2.5023
  - 222: 2.5005
  - 225: 2.5003
  - 229: 2.4995
  - 232: 2.4984
  - 235: 2.4980
  - 239: 2.4972
  - 248: 2.4741
  - 273: 2.4650
  - 277: 2.4615
  - 282: 2.4523
  - 287: 2.4191
  - 313: 2.4080
  - 382: 2.3843
  - 383: 2.3690
  - 387: 2.3541
  - 404: 2.3340
  - 437: 2.3316
  - 441: 2.3308
  - 444: 2.3295
  - 447: 2.3279
  - 497: 2.3270
  - 579: 2.2825
  - 616: 2.2767
  - 630: 2.2514
  - 783: 2.2300
  - 788: 2.2047

## Main contributors (objective drop credited per param)

Free attribution from coordinate descent's accepted moves — which single params drove convergence.

| param | objective drop | % of total | moves |
|---|---|---|---|
| CROP_SHADOW_REL | 0.1612 | 19.6% | 8 |
| CROP_SHADOW_MAX_FRAC | 0.1405 | 17.1% | 4 |
| CROP_REBATE_MARGIN_D | 0.1069 | 13.0% | 7 |
| CROP_PAD_FRAC | 0.0810 | 9.8% | 2 |
| CROP_REBATE_MAX_FRAC | 0.0651 | 7.9% | 3 |
| CROP_REBATE_HUE_TOL | 0.0643 | 7.8% | 1 |
| CROP_JUNK_LINE_FRAC | 0.0516 | 6.3% | 5 |
| CROP_REBATE_LINE_FRAC | 0.0433 | 5.3% | 3 |
| CROP_REBATE_WIDE_LINE_FRAC | 0.0399 | 4.8% | 7 |
| CROP_GAP_TOL_FRAC | 0.0319 | 3.9% | 2 |
| CROP_SHADOW_CORE_FRAC | 0.0154 | 1.9% | 3 |
| CROP_LEAK_MARGIN_D | 0.0117 | 1.4% | 3 |
| HOLDER_LUMA_THR | 0.0112 | 1.4% | 10 |

## Fitted constants (record-only — adopt by hand)

| constant | init | fitted |
|---|---|---|
| CROP_JUNK_LINE_FRAC | 0.0975 | 0.0800 |
| CROP_LEAK_MARGIN_D | 0.0600 | 0.0300 |
| CROP_REBATE_MARGIN_D | 0.1300 | 0.1456 |
| CROP_REBATE_LINE_FRAC | 0.1500 | 0.1827 |
| CROP_REBATE_TERM_FRAC | 0.0400 | 0.0400 |
| CROP_REBATE_MAX_FRAC | 0.0400 | 0.0269 |
| CROP_REBATE_HUE_TOL | 0.0241 | 0.0200 |
| CROP_REBATE_WIDE_MAX_D | 0.3871 | 0.3871 |
| CROP_REBATE_WIDE_LINE_FRAC | 0.4594 | 0.8000 |
| CROP_REBATE_BAND_HUE_TOL | 0.0094 | 0.0094 |
| CROP_REBATE_WIDE_FRAC | 0.5500 | 0.5500 |
| CROP_PAD_FRAC | 0.0050 | 0.0050 |
| CROP_SHADOW_REL | 0.6191 | 0.7375 |
| CROP_SHADOW_MAX_FRAC | 0.0253 | 0.0309 |
| CROP_SHADOW_CORE_FRAC | 0.0167 | 0.0167 |
| CROP_GAP_TOL_FRAC | 0.0181 | 0.0153 |
| HOLDER_LUMA_THR | 0.0255 | 0.0241 |
| BORDER_MAX_FRAC | 0.0800 | 0.0800 |

## Aggregate (all rolls)

- total_overtrim_area: 1.4211
- median_overtrim_area: 0.0084
- max_overtrim_area: 0.0281
- total_undertrim_area: 0.1567
- max_undertrim_area: 0.0123
- containment_violations: 39
- containment_weight: 5.0000
- n_frames: 150

## Per-roll

### 2506-1  (prep 25.33s)
```
{
  "n": 38,
  "median_overtrim_area": 0.004970914027800255,
  "worst_frames": [
    {
      "stem": "DSC_0005",
      "overtrim_area": 0.023200595805386224
    },
    {
      "stem": "DSC_0019",
      "overtrim_area": 0.01633669597741454
    },
    {
      "stem": "DSC_0011",
      "overtrim_area": 0.015926105746465028
    },
    {
      "stem": "DSC_0003",
      "overtrim_area": 0.012121402839965715
    },
    {
      "stem": "DSC_0031",
      "overtrim_area": 0.009431587275898653
    }
  ]
}
```
### 2510-11-1  (prep 27.51s)
```
{
  "n": 38,
  "median_overtrim_area": 0.010774621927316538,
  "worst_frames": [
    {
      "stem": "DSC_0003",
      "overtrim_area": 0.02814266362170554
    },
    {
      "stem": "DSC_0018",
      "overtrim_area": 0.02493736251221281
    },
    {
      "stem": "DSC_0032",
      "overtrim_area": 0.02238391085696475
    },
    {
      "stem": "DSC_0017",
      "overtrim_area": 0.021436106765448083
    },
    {
      "stem": "DSC_0010",
      "overtrim_area": 0.017205079330827835
    }
  ]
}
```
### 2511-12-1  (prep 27.74s)
```
{
  "n": 37,
  "median_overtrim_area": 0.009461182739625854,
  "worst_frames": [
    {
      "stem": "DSC_0014",
      "overtrim_area": 0.02129674584764405
    },
    {
      "stem": "DSC_0010",
      "overtrim_area": 0.018073088058117998
    },
    {
      "stem": "DSC_0027",
      "overtrim_area": 0.017030128931326535
    },
    {
      "stem": "DSC_0006",
      "overtrim_area": 0.016240042437647227
    },
    {
      "stem": "DSC_0018",
      "overtrim_area": 0.013952949955943967
    }
  ]
}
```
### 2512-2601-1  (prep 27.92s)
```
{
  "n": 37,
  "median_overtrim_area": 0.00827923732115349,
  "worst_frames": [
    {
      "stem": "DSC_0011",
      "overtrim_area": 0.023604442766119414
    },
    {
      "stem": "DSC_0035",
      "overtrim_area": 0.018639148130166094
    },
    {
      "stem": "DSC_0012",
      "overtrim_area": 0.016227679775583967
    },
    {
      "stem": "DSC_0001",
      "overtrim_area": 0.015986795178411946
    },
    {
      "stem": "DSC_0007",
      "overtrim_area": 0.015027752303201405
    }
  ]
}
```

## Worst frames (find where to look)

| contained | overtrim_area | undertrim_area | overtrim_edges | roll | stem |
|---|---|---|---|---|---|
| True | 0.0281 | 0.0000 | {'left': 0.001001001001001001, 'top': 0.02470059880239521, 'right': 0.001001001001001001, 'bottom': 0.0029940119760479044} | 2510-11-1 | DSC_0003 |
| True | 0.0249 | 0.0000 | {'left': 0.003003003003003003, 'top': 0.01122754491017964, 'right': 0.00850850850850851, 'bottom': 0.0037425149700598802} | 2510-11-1 | DSC_0018 |
| True | 0.0236 | 0.0000 | {'left': 0.003003003003003003, 'top': 0.0187125748502994, 'right': 0.0025025025025025025, 'bottom': 0.0007485029940119761} | 2512-2601-1 | DSC_0011 |
| True | 0.0232 | 0.0000 | {'left': 0.001001001001001001, 'top': 0.0029940119760479044, 'right': 0.015515515515515516, 'bottom': 0.005239520958083832} | 2506-1 | DSC_0005 |
| True | 0.0224 | 0.0000 | {'left': 0.010510510510510511, 'top': 0.006736526946107785, 'right': 0.002002002002002002, 'bottom': 0.004491017964071856} | 2510-11-1 | DSC_0032 |
| True | 0.0214 | 0.0000 | {'left': 0.002002002002002002, 'top': 0.006736526946107785, 'right': 0.002002002002002002, 'bottom': 0.011976047904191617} | 2510-11-1 | DSC_0017 |
| True | 0.0213 | 0.0000 | {'left': 0.0015015015015015015, 'top': 0.017964071856287425, 'right': 0.0015015015015015015, 'bottom': 0.0014970059880239522} | 2511-12-1 | DSC_0014 |
| True | 0.0186 | 0.0000 | {'left': 0.0015015015015015015, 'top': 0.004491017964071856, 'right': 0.0025025025025025025, 'bottom': 0.01122754491017964} | 2512-2601-1 | DSC_0035 |
| True | 0.0181 | 0.0000 | {'left': 0.0025025025025025025, 'top': 0.005239520958083832, 'right': 0.008008008008008008, 'bottom': 0.0037425149700598802} | 2511-12-1 | DSC_0010 |
| True | 0.0172 | 0.0000 | {'left': 0.002002002002002002, 'top': 0.01122754491017964, 'right': 0.002002002002002002, 'bottom': 0.0029940119760479044} | 2510-11-1 | DSC_0010 |
| True | 0.0171 | 0.0000 | {'left': 0.0025025025025025025, 'top': 0.011976047904191617, 'right': 0.0005005005005005005, 'bottom': 0.0029940119760479044} | 2510-11-1 | DSC_0014 |
| True | 0.0170 | 0.0000 | {'left': 0.0015015015015015015, 'top': 0.01122754491017964, 'right': 0.003003003003003003, 'bottom': 0.002245508982035928} | 2511-12-1 | DSC_0027 |
| False | 0.0163 | 0.0086 | {'left': -0.0055055055055055055, 'top': 0.01721556886227545, 'right': -0.0015015015015015015, 'bottom': -0.002245508982035928} | 2506-1 | DSC_0019 |
| True | 0.0162 | 0.0000 | {'left': 0.0025025025025025025, 'top': 0.005239520958083832, 'right': 0.0005005005005005005, 'bottom': 0.008982035928143712} | 2511-12-1 | DSC_0006 |
| True | 0.0162 | 0.0000 | {'left': 0.00950950950950951, 'top': 0.005239520958083832, 'right': 0.001001001001001001, 'bottom': 0.0014970059880239522} | 2512-2601-1 | DSC_0012 |
