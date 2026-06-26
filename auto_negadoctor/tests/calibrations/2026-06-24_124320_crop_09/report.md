# Calibration session — crop

- comment: second run on cmaes, with more popsize to match num evals to descent. compare with crop_06
- created: 2026-06-24T12:28:31
- git commit: 7bb7153
- rolls: 2506-1, 2510-11-1, 2511-12-1, 2512-2601-1
- metric: crop_overtrim_area_fraction (containment HARD)
- fit method: cmaes
- method params: sigma=0.3, popsize=16, max_iters=40, seed=0, workers=auto
- fitted params: a, l, l
- trial count: 640
- wall time: 888.98s (prep 185.37s)
- objective: 2.3159 (init) -> 2.1103 (final)

## Convergence

- method: cmaes  trials: 640
- generations run: 40 (popsize 16, sigma0 0.3, seed 0)
- per-generation best objective (sigma):
  - gen 1: 3.7520 (sigma 0.2797)
  - gen 2: 3.4044 (sigma 0.2669)
  - gen 3: 3.0628 (sigma 0.2483)
  - gen 4: 2.7223 (sigma 0.2387)
  - gen 5: 2.7223 (sigma 0.2409)
  - gen 6: 2.7223 (sigma 0.2346)
  - gen 7: 2.7223 (sigma 0.2298)
  - gen 8: 2.7223 (sigma 0.2318)
  - gen 9: 2.7223 (sigma 0.2240)
  - gen 10: 2.7223 (sigma 0.2104)
  - gen 11: 2.7223 (sigma 0.2031)
  - gen 12: 2.6607 (sigma 0.1956)
  - gen 13: 2.6607 (sigma 0.1941)
  - gen 14: 2.6607 (sigma 0.1839)
  - gen 15: 2.6607 (sigma 0.1738)
  - gen 16: 2.3693 (sigma 0.1717)
  - gen 17: 2.3693 (sigma 0.1777)
  - gen 18: 2.3693 (sigma 0.1811)
  - gen 19: 2.3693 (sigma 0.1803)
  - gen 20: 2.3693 (sigma 0.1778)
  - gen 21: 2.3672 (sigma 0.1781)
  - gen 22: 2.2252 (sigma 0.1687)
  - gen 23: 2.2128 (sigma 0.1630)
  - gen 24: 2.2128 (sigma 0.1488)
  - gen 25: 2.2128 (sigma 0.1437)
  - gen 26: 2.2128 (sigma 0.1410)
  - gen 27: 2.2128 (sigma 0.1342)
  - gen 28: 2.2128 (sigma 0.1274)
  - gen 29: 2.2128 (sigma 0.1257)
  - gen 30: 2.2128 (sigma 0.1232)
  - gen 31: 2.2128 (sigma 0.1239)
  - gen 32: 2.2128 (sigma 0.1270)
  - gen 33: 2.2128 (sigma 0.1277)
  - gen 34: 2.1103 (sigma 0.1223)
  - gen 35: 2.1103 (sigma 0.1220)
  - gen 36: 2.1103 (sigma 0.1195)
  - gen 37: 2.1103 (sigma 0.1140)
  - gen 38: 2.1103 (sigma 0.1086)
  - gen 39: 2.1103 (sigma 0.1071)
  - gen 40: 2.1103 (sigma 0.0996)
- best-so-far improved 15 time(s); curve (trial: objective):
  - 1: 4.1459
  - 6: 3.7520
  - 17: 3.4044
  - 34: 3.1727
  - 38: 3.0628
  - 50: 3.0144
  - 54: 2.7223
  - 183: 2.6607
  - 246: 2.6472
  - 253: 2.5309
  - 255: 2.3693
  - 321: 2.3672
  - 347: 2.2252
  - 356: 2.2128
  - 538: 2.1103

## Fitted constants (record-only — adopt by hand)

| constant | init | fitted |
|---|---|---|
| CROP_JUNK_LINE_FRAC | 0.0800 | 0.1000 |
| CROP_LEAK_MARGIN_D | 0.0300 | 0.0347 |
| CROP_REBATE_MARGIN_D | 0.1456 | 0.0800 |
| CROP_REBATE_LINE_FRAC | 0.1827 | 0.3000 |
| CROP_REBATE_TERM_FRAC | 0.0400 | 0.0791 |
| CROP_REBATE_MAX_FRAC | 0.0269 | 0.0324 |
| CROP_REBATE_HUE_TOL | 0.0200 | 0.0200 |
| CROP_REBATE_WIDE_MAX_D | 0.3871 | 0.3607 |
| CROP_REBATE_WIDE_LINE_FRAC | 0.8000 | 0.8000 |
| CROP_REBATE_BAND_HUE_TOL | 0.0094 | 0.0072 |
| CROP_REBATE_WIDE_FRAC | 0.5500 | 0.6000 |
| CROP_PAD_FRAC | 0.0050 | 0.0051 |
| CROP_SHADOW_REL | 0.7375 | 0.7584 |
| CROP_SHADOW_MAX_FRAC | 0.0309 | 0.0361 |
| CROP_SHADOW_CORE_FRAC | 0.0167 | 0.0158 |
| CROP_GAP_TOL_FRAC | 0.0153 | 0.0298 |
| HOLDER_LUMA_THR | 0.0241 | 0.0327 |
| BORDER_MAX_FRAC | 0.0800 | 0.0837 |

## Aggregate (all rolls)

- total_overtrim_area: 1.4027
- median_overtrim_area: 0.0088
- max_overtrim_area: 0.0301
- total_undertrim_area: 0.1415
- max_undertrim_area: 0.0118
- containment_violations: 41
- containment_weight: 5.0000
- n_frames: 150

## Per-roll

### 2506-1  (prep 46.66s)
```
{
  "n": 38,
  "median_overtrim_area": 0.004499259738780697,
  "worst_frames": [
    {
      "stem": "DSC_0011",
      "overtrim_area": 0.012419979860099621
    },
    {
      "stem": "DSC_0023",
      "overtrim_area": 0.010095424766083449
    },
    {
      "stem": "DSC_0031",
      "overtrim_area": 0.009431587275898653
    },
    {
      "stem": "DSC_0017",
      "overtrim_area": 0.009368650087212962
    },
    {
      "stem": "DSC_0032",
      "overtrim_area": 0.008049216881552211
    }
  ]
}
```
### 2510-11-1  (prep 46.39s)
```
{
  "n": 38,
  "median_overtrim_area": 0.011241593389796982,
  "worst_frames": [
    {
      "stem": "DSC_0032",
      "overtrim_area": 0.03008997020973069
    },
    {
      "stem": "DSC_0017",
      "overtrim_area": 0.024502421583259906
    },
    {
      "stem": "DSC_0010",
      "overtrim_area": 0.01936517355679032
    },
    {
      "stem": "DSC_0037",
      "overtrim_area": 0.01735942529355703
    },
    {
      "stem": "DSC_0018",
      "overtrim_area": 0.016562220903538268
    }
  ]
}
```
### 2511-12-1  (prep 44.79s)
```
{
  "n": 37,
  "median_overtrim_area": 0.009947072821324319,
  "worst_frames": [
    {
      "stem": "DSC_0002",
      "overtrim_area": 0.018363048677419936
    },
    {
      "stem": "DSC_0007",
      "overtrim_area": 0.01790225854597112
    },
    {
      "stem": "DSC_0024",
      "overtrim_area": 0.015833947720175264
    },
    {
      "stem": "DSC_0006",
      "overtrim_area": 0.015310969652287017
    },
    {
      "stem": "DSC_0038",
      "overtrim_area": 0.013874653096209982
    }
  ]
}
```
### 2512-2601-1  (prep 47.53s)
```
{
  "n": 37,
  "median_overtrim_area": 0.008518248787709865,
  "worst_frames": [
    {
      "stem": "DSC_0007",
      "overtrim_area": 0.027251802700904496
    },
    {
      "stem": "DSC_0012",
      "overtrim_area": 0.026285267303231375
    },
    {
      "stem": "DSC_0033",
      "overtrim_area": 0.021371671072269876
    },
    {
      "stem": "DSC_0034",
      "overtrim_area": 0.016063968159776544
    },
    {
      "stem": "DSC_0035",
      "overtrim_area": 0.015098931266595937
    }
  ]
}
```

## Worst frames (find where to look)

| contained | overtrim_area | undertrim_area | overtrim_edges | roll | stem |
|---|---|---|---|---|---|
| True | 0.0301 | 0.0000 | {'left': 0.018018018018018018, 'top': 0.006736526946107785, 'right': 0.002002002002002002, 'bottom': 0.005239520958083832} | 2510-11-1 | DSC_0032 |
| True | 0.0273 | 0.0000 | {'left': 0.001001001001001001, 'top': 0.0074850299401197605, 'right': 0.0, 'bottom': 0.020958083832335328} | 2512-2601-1 | DSC_0007 |
| True | 0.0263 | 0.0000 | {'left': 0.021021021021021023, 'top': 0.004491017964071856, 'right': 0.001001001001001001, 'bottom': 0.0014970059880239522} | 2512-2601-1 | DSC_0012 |
| True | 0.0245 | 0.0000 | {'left': 0.0015015015015015015, 'top': 0.008233532934131737, 'right': 0.002002002002002002, 'bottom': 0.014221556886227544} | 2510-11-1 | DSC_0017 |
| True | 0.0214 | 0.0000 | {'left': 0.0025025025025025025, 'top': 0.004491017964071856, 'right': 0.012012012012012012, 'bottom': 0.0037425149700598802} | 2512-2601-1 | DSC_0033 |
| False | 0.0194 | 0.0014 | {'left': -0.0015015015015015015, 'top': 0.015718562874251496, 'right': 0.0025025025025025025, 'bottom': 0.002245508982035928} | 2510-11-1 | DSC_0010 |
| True | 0.0184 | 0.0005 | {'left': -0.0005005005005005005, 'top': 0.033682634730538924, 'right': 0.0005005005005005005, 'bottom': 0.0037425149700598802} | 2511-12-1 | DSC_0002 |
| True | 0.0179 | 0.0000 | {'left': 0.0, 'top': 0.004491017964071856, 'right': 0.011511511511511512, 'bottom': 0.0029940119760479044} | 2511-12-1 | DSC_0007 |
| True | 0.0174 | 0.0000 | {'left': 0.0005005005005005005, 'top': 0.005239520958083832, 'right': 0.012012012012012012, 'bottom': 0.0007485029940119761} | 2510-11-1 | DSC_0037 |
| True | 0.0166 | 0.0005 | {'left': -0.0005005005005005005, 'top': 0.010479041916167664, 'right': 0.0025025025025025025, 'bottom': 0.004491017964071856} | 2510-11-1 | DSC_0018 |
| True | 0.0161 | 0.0000 | {'left': 0.002002002002002002, 'top': 0.008982035928143712, 'right': 0.0015015015015015015, 'bottom': 0.004491017964071856} | 2512-2601-1 | DSC_0034 |
| True | 0.0158 | 0.0000 | {'left': 0.0035035035035035035, 'top': 0.008982035928143712, 'right': 0.002002002002002002, 'bottom': 0.002245508982035928} | 2511-12-1 | DSC_0024 |
| True | 0.0153 | 0.0000 | {'left': 0.001001001001001001, 'top': 0.005988023952095809, 'right': 0.001001001001001001, 'bottom': 0.008233532934131737} | 2511-12-1 | DSC_0006 |
| True | 0.0151 | 0.0000 | {'left': 0.0015015015015015015, 'top': 0.005239520958083832, 'right': 0.0025025025025025025, 'bottom': 0.006736526946107785} | 2512-2601-1 | DSC_0035 |
| True | 0.0151 | 0.0000 | {'left': 0.0015015015015015015, 'top': 0.009730538922155689, 'right': 0.0025025025025025025, 'bottom': 0.002245508982035928} | 2510-11-1 | DSC_0011 |
