# Calibration session — crop

- comment: third run on cmaes, with more iterations. compare with crop_06
- created: 2026-06-24T12:55:38
- git commit: 7bb7153
- rolls: 2506-1, 2510-11-1, 2511-12-1, 2512-2601-1
- metric: crop_overtrim_area_fraction (containment HARD)
- fit method: cmaes
- method params: sigma=0.3, popsize=16, max_iters=100, seed=0, workers=16
- fitted params: a, l, l
- trial count: 1600
- wall time: 1509.38s (prep 195.44s)
- objective: 2.3159 (init) -> 2.0624 (final)

## Convergence

- method: cmaes  trials: 1600
- generations run: 100 (popsize 16, sigma0 0.3, seed 0)
- per-generation best objective (sigma):
  - gen 1: 4.3165 (sigma 0.2872)
  - gen 2: 4.3165 (sigma 0.2848)
  - gen 3: 3.0187 (sigma 0.2823)
  - gen 4: 3.0100 (sigma 0.2728)
  - gen 5: 3.0100 (sigma 0.2667)
  - gen 6: 3.0100 (sigma 0.2505)
  - gen 7: 2.9812 (sigma 0.2486)
  - gen 8: 2.7753 (sigma 0.2479)
  - gen 9: 2.7753 (sigma 0.2409)
  - gen 10: 2.7753 (sigma 0.2439)
  - gen 11: 2.7753 (sigma 0.2490)
  - gen 12: 2.7095 (sigma 0.2514)
  - gen 13: 2.5517 (sigma 0.2462)
  - gen 14: 2.5149 (sigma 0.2490)
  - gen 15: 2.5149 (sigma 0.2516)
  - gen 16: 2.5149 (sigma 0.2456)
  - gen 17: 2.5149 (sigma 0.2410)
  - gen 18: 2.5149 (sigma 0.2293)
  - gen 19: 2.5149 (sigma 0.2218)
  - gen 20: 2.5149 (sigma 0.2189)
  - gen 21: 2.5149 (sigma 0.2120)
  - gen 22: 2.5149 (sigma 0.2081)
  - gen 23: 2.5149 (sigma 0.1899)
  - gen 24: 2.5149 (sigma 0.1804)
  - gen 25: 2.5149 (sigma 0.1682)
  - gen 26: 2.4806 (sigma 0.1566)
  - gen 27: 2.4806 (sigma 0.1517)
  - gen 28: 2.3622 (sigma 0.1533)
  - gen 29: 2.3622 (sigma 0.1526)
  - gen 30: 2.2554 (sigma 0.1526)
  - gen 31: 2.2554 (sigma 0.1548)
  - gen 32: 2.2554 (sigma 0.1511)
  - gen 33: 2.2554 (sigma 0.1507)
  - gen 34: 2.2554 (sigma 0.1435)
  - gen 35: 2.2279 (sigma 0.1308)
  - gen 36: 2.2279 (sigma 0.1272)
  - gen 37: 2.2279 (sigma 0.1241)
  - gen 38: 2.2279 (sigma 0.1212)
  - gen 39: 2.2279 (sigma 0.1230)
  - gen 40: 2.2279 (sigma 0.1251)
  - gen 41: 2.2279 (sigma 0.1273)
  - gen 42: 2.2279 (sigma 0.1259)
  - gen 43: 2.2279 (sigma 0.1226)
  - gen 44: 2.2279 (sigma 0.1146)
  - gen 45: 2.1450 (sigma 0.1076)
  - gen 46: 2.1450 (sigma 0.1112)
  - gen 47: 2.1027 (sigma 0.1114)
  - gen 48: 2.1027 (sigma 0.1122)
  - gen 49: 2.1027 (sigma 0.1103)
  - gen 50: 2.1027 (sigma 0.1083)
  - gen 51: 2.1027 (sigma 0.1053)
  - gen 52: 2.1027 (sigma 0.1095)
  - gen 53: 2.1027 (sigma 0.1054)
  - gen 54: 2.1027 (sigma 0.1013)
  - gen 55: 2.1027 (sigma 0.0937)
  - gen 56: 2.1027 (sigma 0.0912)
  - gen 57: 2.1027 (sigma 0.0912)
  - gen 58: 2.1027 (sigma 0.0916)
  - gen 59: 2.1027 (sigma 0.0905)
  - gen 60: 2.1027 (sigma 0.0860)
  - gen 61: 2.1027 (sigma 0.0816)
  - gen 62: 2.1027 (sigma 0.0803)
  - gen 63: 2.1027 (sigma 0.0768)
  - gen 64: 2.1027 (sigma 0.0704)
  - gen 65: 2.1027 (sigma 0.0692)
  - gen 66: 2.1027 (sigma 0.0665)
  - gen 67: 2.1027 (sigma 0.0659)
  - gen 68: 2.1027 (sigma 0.0657)
  - gen 69: 2.1027 (sigma 0.0628)
  - gen 70: 2.1027 (sigma 0.0598)
  - gen 71: 2.1027 (sigma 0.0587)
  - gen 72: 2.1027 (sigma 0.0577)
  - gen 73: 2.1027 (sigma 0.0561)
  - gen 74: 2.1027 (sigma 0.0536)
  - gen 75: 2.1027 (sigma 0.0517)
  - gen 76: 2.1027 (sigma 0.0511)
  - gen 77: 2.0995 (sigma 0.0483)
  - gen 78: 2.0995 (sigma 0.0470)
  - gen 79: 2.0901 (sigma 0.0455)
  - gen 80: 2.0901 (sigma 0.0453)
  - gen 81: 2.0901 (sigma 0.0452)
  - gen 82: 2.0901 (sigma 0.0441)
  - gen 83: 2.0901 (sigma 0.0458)
  - gen 84: 2.0901 (sigma 0.0471)
  - gen 85: 2.0901 (sigma 0.0472)
  - gen 86: 2.0889 (sigma 0.0464)
  - gen 87: 2.0889 (sigma 0.0450)
  - gen 88: 2.0884 (sigma 0.0438)
  - gen 89: 2.0884 (sigma 0.0440)
  - gen 90: 2.0624 (sigma 0.0443)
  - gen 91: 2.0624 (sigma 0.0458)
  - gen 92: 2.0624 (sigma 0.0455)
  - gen 93: 2.0624 (sigma 0.0433)
  - gen 94: 2.0624 (sigma 0.0399)
  - gen 95: 2.0624 (sigma 0.0371)
  - gen 96: 2.0624 (sigma 0.0335)
  - gen 97: 2.0624 (sigma 0.0308)
  - gen 98: 2.0624 (sigma 0.0290)
  - gen 99: 2.0624 (sigma 0.0284)
  - gen 100: 2.0624 (sigma 0.0275)
- best-so-far improved 22 time(s); curve (trial: objective):
  - 1: 9.2727
  - 2: 4.3165
  - 34: 3.0187
  - 59: 3.0100
  - 98: 2.9812
  - 118: 2.8661
  - 124: 2.7753
  - 181: 2.7095
  - 204: 2.5517
  - 220: 2.5149
  - 408: 2.4806
  - 444: 2.3622
  - 465: 2.2554
  - 546: 2.2451
  - 559: 2.2279
  - 711: 2.1450
  - 740: 2.1027
  - 1219: 2.0995
  - 1259: 2.0901
  - 1365: 2.0889
  - 1400: 2.0884
  - 1429: 2.0624

## Fitted constants (record-only — adopt by hand)

| constant | init | fitted |
|---|---|---|
| CROP_JUNK_LINE_FRAC | 0.0800 | 0.1000 |
| CROP_LEAK_MARGIN_D | 0.0300 | 0.0336 |
| CROP_REBATE_MARGIN_D | 0.1456 | 0.0800 |
| CROP_REBATE_LINE_FRAC | 0.1827 | 0.2804 |
| CROP_REBATE_TERM_FRAC | 0.0400 | 0.0800 |
| CROP_REBATE_MAX_FRAC | 0.0269 | 0.0319 |
| CROP_REBATE_HUE_TOL | 0.0200 | 0.0200 |
| CROP_REBATE_WIDE_MAX_D | 0.3871 | 0.2500 |
| CROP_REBATE_WIDE_LINE_FRAC | 0.8000 | 0.4000 |
| CROP_REBATE_BAND_HUE_TOL | 0.0094 | 0.0075 |
| CROP_REBATE_WIDE_FRAC | 0.5500 | 0.5056 |
| CROP_PAD_FRAC | 0.0050 | 0.0049 |
| CROP_SHADOW_REL | 0.7375 | 0.7558 |
| CROP_SHADOW_MAX_FRAC | 0.0309 | 0.0325 |
| CROP_SHADOW_CORE_FRAC | 0.0167 | 0.0165 |
| CROP_GAP_TOL_FRAC | 0.0153 | 0.0317 |
| HOLDER_LUMA_THR | 0.0241 | 0.0300 |
| BORDER_MAX_FRAC | 0.0800 | 0.0800 |

## Aggregate (all rolls)

- total_overtrim_area: 1.3903
- median_overtrim_area: 0.0090
- max_overtrim_area: 0.0320
- total_undertrim_area: 0.1344
- max_undertrim_area: 0.0118
- containment_violations: 40
- containment_weight: 5.0000
- n_frames: 150

## Per-roll

### 2506-1  (prep 54.81s)
```
{
  "n": 38,
  "median_overtrim_area": 0.0043849987712263165,
  "worst_frames": [
    {
      "stem": "DSC_0011",
      "overtrim_area": 0.013590311868754982
    },
    {
      "stem": "DSC_0023",
      "overtrim_area": 0.010330689971408534
    },
    {
      "stem": "DSC_0017",
      "overtrim_area": 0.010303716890543237
    },
    {
      "stem": "DSC_0031",
      "overtrim_area": 0.009431587275898653
    },
    {
      "stem": "DSC_0035",
      "overtrim_area": 0.007573441705178232
    }
  ]
}
```
### 2510-11-1  (prep 45.87s)
```
{
  "n": 38,
  "median_overtrim_area": 0.011238409067750385,
  "worst_frames": [
    {
      "stem": "DSC_0010",
      "overtrim_area": 0.03202566338793884
    },
    {
      "stem": "DSC_0032",
      "overtrim_area": 0.03008997020973069
    },
    {
      "stem": "DSC_0017",
      "overtrim_area": 0.024502421583259906
    },
    {
      "stem": "DSC_0018",
      "overtrim_area": 0.020061603519687353
    },
    {
      "stem": "DSC_0011",
      "overtrim_area": 0.015774382166597736
    }
  ]
}
```
### 2511-12-1  (prep 46.12s)
```
{
  "n": 37,
  "median_overtrim_area": 0.0099545653437869,
  "worst_frames": [
    {
      "stem": "DSC_0002",
      "overtrim_area": 0.0165742089394784
    },
    {
      "stem": "DSC_0006",
      "overtrim_area": 0.016017514520508532
    },
    {
      "stem": "DSC_0024",
      "overtrim_area": 0.014883146619673566
    },
    {
      "stem": "DSC_0018",
      "overtrim_area": 0.014419359479239719
    },
    {
      "stem": "DSC_0027",
      "overtrim_area": 0.013478673284062506
    }
  ]
}
```
### 2512-2601-1  (prep 48.64s)
```
{
  "n": 37,
  "median_overtrim_area": 0.00900039260817704,
  "worst_frames": [
    {
      "stem": "DSC_0007",
      "overtrim_area": 0.018944093794393196
    },
    {
      "stem": "DSC_0033",
      "overtrim_area": 0.018800611989234745
    },
    {
      "stem": "DSC_0035",
      "overtrim_area": 0.015806974639309967
    },
    {
      "stem": "DSC_0014",
      "overtrim_area": 0.014696582810355265
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
| True | 0.0320 | 0.0000 | {'left': 0.013013013013013013, 'top': 0.015718562874251496, 'right': 0.0025025025025025025, 'bottom': 0.0029940119760479044} | 2510-11-1 | DSC_0010 |
| True | 0.0301 | 0.0000 | {'left': 0.018018018018018018, 'top': 0.006736526946107785, 'right': 0.002002002002002002, 'bottom': 0.005239520958083832} | 2510-11-1 | DSC_0032 |
| True | 0.0245 | 0.0000 | {'left': 0.0015015015015015015, 'top': 0.008233532934131737, 'right': 0.002002002002002002, 'bottom': 0.014221556886227544} | 2510-11-1 | DSC_0017 |
| True | 0.0201 | 0.0000 | {'left': 0.003003003003003003, 'top': 0.01122754491017964, 'right': 0.0025025025025025025, 'bottom': 0.004491017964071856} | 2510-11-1 | DSC_0018 |
| True | 0.0189 | 0.0000 | {'left': 0.001001001001001001, 'top': 0.0074850299401197605, 'right': 0.0, 'bottom': 0.011976047904191617} | 2512-2601-1 | DSC_0007 |
| True | 0.0188 | 0.0000 | {'left': 0.0025025025025025025, 'top': 0.0037425149700598802, 'right': 0.01001001001001001, 'bottom': 0.0037425149700598802} | 2512-2601-1 | DSC_0033 |
| True | 0.0166 | 0.0000 | {'left': 0.0005005005005005005, 'top': 0.029940119760479042, 'right': 0.0, 'bottom': 0.0037425149700598802} | 2511-12-1 | DSC_0002 |
| True | 0.0160 | 0.0000 | {'left': 0.001001001001001001, 'top': 0.005988023952095809, 'right': 0.001001001001001001, 'bottom': 0.008982035928143712} | 2511-12-1 | DSC_0006 |
| True | 0.0158 | 0.0000 | {'left': 0.0015015015015015015, 'top': 0.005239520958083832, 'right': 0.0025025025025025025, 'bottom': 0.0074850299401197605} | 2512-2601-1 | DSC_0035 |
| True | 0.0158 | 0.0000 | {'left': 0.0015015015015015015, 'top': 0.009730538922155689, 'right': 0.0025025025025025025, 'bottom': 0.0029940119760479044} | 2510-11-1 | DSC_0011 |
| True | 0.0149 | 0.0000 | {'left': 0.0035035035035035035, 'top': 0.008233532934131737, 'right': 0.0025025025025025025, 'bottom': 0.0014970059880239522} | 2511-12-1 | DSC_0024 |
| True | 0.0148 | 0.0000 | {'left': 0.0025025025025025025, 'top': 0.0074850299401197605, 'right': 0.002002002002002002, 'bottom': 0.0037425149700598802} | 2510-11-1 | DSC_0016 |
| True | 0.0147 | 0.0000 | {'left': 0.006006006006006006, 'top': 0.005239520958083832, 'right': 0.002002002002002002, 'bottom': 0.002245508982035928} | 2512-2601-1 | DSC_0014 |
| True | 0.0144 | 0.0000 | {'left': 0.002002002002002002, 'top': 0.008233532934131737, 'right': 0.002002002002002002, 'bottom': 0.0029940119760479044} | 2511-12-1 | DSC_0018 |
| True | 0.0143 | 0.0000 | {'left': 0.0025025025025025025, 'top': 0.008233532934131737, 'right': 0.0015015015015015015, 'bottom': 0.0029940119760479044} | 2510-11-1 | DSC_0024 |
