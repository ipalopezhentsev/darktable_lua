# Calibration session — crop

- comment: 6th run on cmaes, with lower popsize but same high iters. compare with crop_10
- created: 2026-06-24T17:51:48
- git commit: 7bb7153
- rolls: 2506-1, 2510-11-1, 2511-12-1, 2512-2601-1
- metric: crop_overtrim_area_fraction (containment HARD)
- fit method: cmaes
- method params: sigma=0.3, popsize=12, max_iters=100, seed=0, workers=16
- fitted params: a, l, l
- trial count: 1200
- wall time: 1672.51s (prep 235.41s)
- objective: 2.3159 (init) -> 2.1216 (final)

## Convergence

- method: cmaes  trials: 1200
- generations run: 100 (popsize 12, sigma0 0.3, seed 0)
- per-generation best objective (sigma):
  - gen 1: 3.4083 (sigma 0.2789)
  - gen 2: 3.4083 (sigma 0.2617)
  - gen 3: 3.4083 (sigma 0.2459)
  - gen 4: 3.3909 (sigma 0.2300)
  - gen 5: 3.3909 (sigma 0.2248)
  - gen 6: 3.0969 (sigma 0.2216)
  - gen 7: 3.0969 (sigma 0.2187)
  - gen 8: 2.9748 (sigma 0.2079)
  - gen 9: 2.7973 (sigma 0.2011)
  - gen 10: 2.7523 (sigma 0.1971)
  - gen 11: 2.7523 (sigma 0.1981)
  - gen 12: 2.7523 (sigma 0.1918)
  - gen 13: 2.7481 (sigma 0.1984)
  - gen 14: 2.7015 (sigma 0.2035)
  - gen 15: 2.7015 (sigma 0.2075)
  - gen 16: 2.7015 (sigma 0.2097)
  - gen 17: 2.6742 (sigma 0.2092)
  - gen 18: 2.6742 (sigma 0.2027)
  - gen 19: 2.6403 (sigma 0.1959)
  - gen 20: 2.6403 (sigma 0.1997)
  - gen 21: 2.6264 (sigma 0.2087)
  - gen 22: 2.6264 (sigma 0.2175)
  - gen 23: 2.6264 (sigma 0.2222)
  - gen 24: 2.5673 (sigma 0.2060)
  - gen 25: 2.5673 (sigma 0.2043)
  - gen 26: 2.5673 (sigma 0.2035)
  - gen 27: 2.5673 (sigma 0.1985)
  - gen 28: 2.5673 (sigma 0.1944)
  - gen 29: 2.5673 (sigma 0.1881)
  - gen 30: 2.5673 (sigma 0.1823)
  - gen 31: 2.5673 (sigma 0.1744)
  - gen 32: 2.5586 (sigma 0.1668)
  - gen 33: 2.5586 (sigma 0.1548)
  - gen 34: 2.5586 (sigma 0.1430)
  - gen 35: 2.5586 (sigma 0.1332)
  - gen 36: 2.5586 (sigma 0.1280)
  - gen 37: 2.5586 (sigma 0.1220)
  - gen 38: 2.5586 (sigma 0.1211)
  - gen 39: 2.4543 (sigma 0.1184)
  - gen 40: 2.4543 (sigma 0.1232)
  - gen 41: 2.4543 (sigma 0.1275)
  - gen 42: 2.4543 (sigma 0.1283)
  - gen 43: 2.4543 (sigma 0.1273)
  - gen 44: 2.4470 (sigma 0.1262)
  - gen 45: 2.4470 (sigma 0.1225)
  - gen 46: 2.4470 (sigma 0.1103)
  - gen 47: 2.4470 (sigma 0.1005)
  - gen 48: 2.4470 (sigma 0.0945)
  - gen 49: 2.4470 (sigma 0.0934)
  - gen 50: 2.3973 (sigma 0.0949)
  - gen 51: 2.3973 (sigma 0.0911)
  - gen 52: 2.3973 (sigma 0.0850)
  - gen 53: 2.3973 (sigma 0.0796)
  - gen 54: 2.3973 (sigma 0.0730)
  - gen 55: 2.3973 (sigma 0.0681)
  - gen 56: 2.3973 (sigma 0.0627)
  - gen 57: 2.3973 (sigma 0.0565)
  - gen 58: 2.3973 (sigma 0.0530)
  - gen 59: 2.3824 (sigma 0.0524)
  - gen 60: 2.3824 (sigma 0.0514)
  - gen 61: 2.3824 (sigma 0.0491)
  - gen 62: 2.3824 (sigma 0.0473)
  - gen 63: 2.3824 (sigma 0.0460)
  - gen 64: 2.3739 (sigma 0.0467)
  - gen 65: 2.3739 (sigma 0.0470)
  - gen 66: 2.2406 (sigma 0.0492)
  - gen 67: 2.1803 (sigma 0.0528)
  - gen 68: 2.1803 (sigma 0.0548)
  - gen 69: 2.1480 (sigma 0.0552)
  - gen 70: 2.1480 (sigma 0.0542)
  - gen 71: 2.1480 (sigma 0.0543)
  - gen 72: 2.1480 (sigma 0.0552)
  - gen 73: 2.1480 (sigma 0.0548)
  - gen 74: 2.1480 (sigma 0.0532)
  - gen 75: 2.1480 (sigma 0.0491)
  - gen 76: 2.1480 (sigma 0.0458)
  - gen 77: 2.1472 (sigma 0.0434)
  - gen 78: 2.1472 (sigma 0.0429)
  - gen 79: 2.1472 (sigma 0.0425)
  - gen 80: 2.1472 (sigma 0.0417)
  - gen 81: 2.1216 (sigma 0.0435)
  - gen 82: 2.1216 (sigma 0.0462)
  - gen 83: 2.1216 (sigma 0.0485)
  - gen 84: 2.1216 (sigma 0.0510)
  - gen 85: 2.1216 (sigma 0.0531)
  - gen 86: 2.1216 (sigma 0.0563)
  - gen 87: 2.1216 (sigma 0.0617)
  - gen 88: 2.1216 (sigma 0.0658)
  - gen 89: 2.1216 (sigma 0.0678)
  - gen 90: 2.1216 (sigma 0.0667)
  - gen 91: 2.1216 (sigma 0.0669)
  - gen 92: 2.1216 (sigma 0.0640)
  - gen 93: 2.1216 (sigma 0.0623)
  - gen 94: 2.1216 (sigma 0.0569)
  - gen 95: 2.1216 (sigma 0.0530)
  - gen 96: 2.1216 (sigma 0.0502)
  - gen 97: 2.1216 (sigma 0.0491)
  - gen 98: 2.1216 (sigma 0.0468)
  - gen 99: 2.1216 (sigma 0.0451)
  - gen 100: 2.1216 (sigma 0.0441)
- best-so-far improved 28 time(s); curve (trial: objective):
  - 1: 3.4083
  - 47: 3.3909
  - 62: 3.1010
  - 68: 3.0969
  - 86: 2.9748
  - 101: 2.7973
  - 119: 2.7523
  - 148: 2.7481
  - 157: 2.7387
  - 166: 2.7015
  - 194: 2.6869
  - 200: 2.6742
  - 224: 2.6403
  - 243: 2.6264
  - 277: 2.5673
  - 380: 2.5586
  - 457: 2.4543
  - 527: 2.4470
  - 597: 2.4341
  - 600: 2.3973
  - 702: 2.3824
  - 761: 2.3739
  - 788: 2.2406
  - 799: 2.2067
  - 804: 2.1803
  - 827: 2.1480
  - 914: 2.1472
  - 968: 2.1216

## Fitted constants (record-only — adopt by hand)

| constant | init | fitted |
|---|---|---|
| CROP_JUNK_LINE_FRAC | 0.0800 | 0.0743 |
| CROP_LEAK_MARGIN_D | 0.0300 | 0.0300 |
| CROP_REBATE_MARGIN_D | 0.1456 | 0.0800 |
| CROP_REBATE_LINE_FRAC | 0.1827 | 0.3000 |
| CROP_REBATE_TERM_FRAC | 0.0400 | 0.0200 |
| CROP_REBATE_MAX_FRAC | 0.0269 | 0.0200 |
| CROP_REBATE_HUE_TOL | 0.0200 | 0.0346 |
| CROP_REBATE_WIDE_MAX_D | 0.3871 | 0.4004 |
| CROP_REBATE_WIDE_LINE_FRAC | 0.8000 | 0.8000 |
| CROP_REBATE_BAND_HUE_TOL | 0.0094 | 0.0066 |
| CROP_REBATE_WIDE_FRAC | 0.5500 | 0.6000 |
| CROP_PAD_FRAC | 0.0050 | 0.0050 |
| CROP_SHADOW_REL | 0.7375 | 0.7563 |
| CROP_SHADOW_MAX_FRAC | 0.0309 | 0.0346 |
| CROP_SHADOW_CORE_FRAC | 0.0167 | 0.0164 |
| CROP_GAP_TOL_FRAC | 0.0153 | 0.0279 |
| HOLDER_LUMA_THR | 0.0241 | 0.0282 |
| BORDER_MAX_FRAC | 0.0800 | 0.0800 |

## Aggregate (all rolls)

- total_overtrim_area: 1.4945
- median_overtrim_area: 0.0090
- max_overtrim_area: 0.0509
- total_undertrim_area: 0.1254
- max_undertrim_area: 0.0118
- containment_violations: 38
- containment_weight: 5.0000
- n_frames: 150

## Per-roll

### 2506-1  (prep 41.71s)
```
{
  "n": 38,
  "median_overtrim_area": 0.004977844611078143,
  "worst_frames": [
    {
      "stem": "DSC_0020",
      "overtrim_area": 0.05093866321411231
    },
    {
      "stem": "DSC_0011",
      "overtrim_area": 0.013123153093212974
    },
    {
      "stem": "DSC_0023",
      "overtrim_area": 0.010330689971408534
    },
    {
      "stem": "DSC_0017",
      "overtrim_area": 0.0098361834888781
    },
    {
      "stem": "DSC_0031",
      "overtrim_area": 0.009431587275898653
    }
  ]
}
```
### 2510-11-1  (prep 65.65s)
```
{
  "n": 38,
  "median_overtrim_area": 0.011383764003524484,
  "worst_frames": [
    {
      "stem": "DSC_0032",
      "overtrim_area": 0.03008997020973069
    },
    {
      "stem": "DSC_0040",
      "overtrim_area": 0.02480137322951694
    },
    {
      "stem": "DSC_0017",
      "overtrim_area": 0.024502421583259906
    },
    {
      "stem": "DSC_0012",
      "overtrim_area": 0.02190101778425132
    },
    {
      "stem": "DSC_0010",
      "overtrim_area": 0.020074340807873743
    }
  ]
}
```
### 2511-12-1  (prep 63.29s)
```
{
  "n": 37,
  "median_overtrim_area": 0.010632264000527473,
  "worst_frames": [
    {
      "stem": "DSC_0002",
      "overtrim_area": 0.028293263323203442
    },
    {
      "stem": "DSC_0007",
      "overtrim_area": 0.01860767953582325
    },
    {
      "stem": "DSC_0006",
      "overtrim_area": 0.016017514520508532
    },
    {
      "stem": "DSC_0024",
      "overtrim_area": 0.015591939244633856
    },
    {
      "stem": "DSC_0018",
      "overtrim_area": 0.014419359479239719
    }
  ]
}
```
### 2512-2601-1  (prep 64.76s)
```
{
  "n": 37,
  "median_overtrim_area": 0.00900039260817704,
  "worst_frames": [
    {
      "stem": "DSC_0012",
      "overtrim_area": 0.02557834780888673
    },
    {
      "stem": "DSC_0007",
      "overtrim_area": 0.019636402869935805
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
    }
  ]
}
```

## Worst frames (find where to look)

| contained | overtrim_area | undertrim_area | overtrim_edges | roll | stem |
|---|---|---|---|---|---|
| False | 0.0509 | 0.0042 | {'left': -0.003003003003003003, 'top': 0.0007485029940119761, 'right': 0.053553553553553554, 'bottom': -0.0014970059880239522} | 2506-1 | DSC_0020 |
| True | 0.0301 | 0.0000 | {'left': 0.018018018018018018, 'top': 0.006736526946107785, 'right': 0.002002002002002002, 'bottom': 0.005239520958083832} | 2510-11-1 | DSC_0032 |
| True | 0.0283 | 0.0000 | {'left': 0.0, 'top': 0.05538922155688623, 'right': 0.0, 'bottom': 0.0037425149700598802} | 2511-12-1 | DSC_0002 |
| True | 0.0256 | 0.0000 | {'left': 0.01951951951951952, 'top': 0.005239520958083832, 'right': 0.001001001001001001, 'bottom': 0.0014970059880239522} | 2512-2601-1 | DSC_0012 |
| True | 0.0248 | 0.0000 | {'left': 0.0015015015015015015, 'top': 0.006736526946107785, 'right': 0.01701701701701702, 'bottom': 0.0014970059880239522} | 2510-11-1 | DSC_0040 |
| True | 0.0245 | 0.0000 | {'left': 0.0015015015015015015, 'top': 0.008233532934131737, 'right': 0.002002002002002002, 'bottom': 0.014221556886227544} | 2510-11-1 | DSC_0017 |
| True | 0.0219 | 0.0000 | {'left': 0.011511511511511512, 'top': 0.005988023952095809, 'right': 0.002002002002002002, 'bottom': 0.0037425149700598802} | 2510-11-1 | DSC_0012 |
| False | 0.0201 | 0.0014 | {'left': -0.0015015015015015015, 'top': 0.015718562874251496, 'right': 0.0025025025025025025, 'bottom': 0.0029940119760479044} | 2510-11-1 | DSC_0010 |
| True | 0.0196 | 0.0000 | {'left': 0.001001001001001001, 'top': 0.0074850299401197605, 'right': 0.0, 'bottom': 0.012724550898203593} | 2512-2601-1 | DSC_0007 |
| True | 0.0188 | 0.0000 | {'left': 0.0025025025025025025, 'top': 0.0037425149700598802, 'right': 0.01001001001001001, 'bottom': 0.0037425149700598802} | 2512-2601-1 | DSC_0033 |
| True | 0.0186 | 0.0000 | {'left': 0.0, 'top': 0.005239520958083832, 'right': 0.011511511511511512, 'bottom': 0.0029940119760479044} | 2511-12-1 | DSC_0007 |
| True | 0.0173 | 0.0005 | {'left': -0.0005005005005005005, 'top': 0.01122754491017964, 'right': 0.0025025025025025025, 'bottom': 0.004491017964071856} | 2510-11-1 | DSC_0018 |
| True | 0.0160 | 0.0000 | {'left': 0.001001001001001001, 'top': 0.005988023952095809, 'right': 0.001001001001001001, 'bottom': 0.008982035928143712} | 2511-12-1 | DSC_0006 |
| True | 0.0158 | 0.0000 | {'left': 0.0015015015015015015, 'top': 0.005239520958083832, 'right': 0.0025025025025025025, 'bottom': 0.0074850299401197605} | 2512-2601-1 | DSC_0035 |
| True | 0.0158 | 0.0000 | {'left': 0.0015015015015015015, 'top': 0.009730538922155689, 'right': 0.0025025025025025025, 'bottom': 0.0029940119760479044} | 2510-11-1 | DSC_0011 |
