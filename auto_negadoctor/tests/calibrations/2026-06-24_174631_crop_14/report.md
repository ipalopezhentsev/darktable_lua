# Calibration session — crop

- comment: 5th run on cmaes, with higher popsize. compare with crop_10
- created: 2026-06-24T16:51:29
- git commit: 7bb7153
- rolls: 2506-1, 2510-11-1, 2511-12-1, 2512-2601-1
- metric: crop_overtrim_area_fraction (containment HARD)
- fit method: cmaes
- method params: sigma=0.3, popsize=32, max_iters=100, seed=0, workers=16
- fitted params: a, l, l
- trial count: 3200
- wall time: 3301.76s (prep 137.2s)
- objective: 2.3159 (init) -> 2.2277 (final)

## Convergence

- method: cmaes  trials: 3200
- generations run: 100 (popsize 32, sigma0 0.3, seed 0)
- per-generation best objective (sigma):
  - gen 1: 3.3843 (sigma 0.2950)
  - gen 2: 3.2122 (sigma 0.3036)
  - gen 3: 3.0950 (sigma 0.3112)
  - gen 4: 2.6873 (sigma 0.3144)
  - gen 5: 2.6873 (sigma 0.3127)
  - gen 6: 2.6873 (sigma 0.3030)
  - gen 7: 2.6873 (sigma 0.2923)
  - gen 8: 2.6873 (sigma 0.2816)
  - gen 9: 2.6021 (sigma 0.2923)
  - gen 10: 2.6021 (sigma 0.3090)
  - gen 11: 2.6021 (sigma 0.2957)
  - gen 12: 2.6021 (sigma 0.2823)
  - gen 13: 2.6021 (sigma 0.2670)
  - gen 14: 2.6021 (sigma 0.2641)
  - gen 15: 2.6021 (sigma 0.2549)
  - gen 16: 2.6021 (sigma 0.2599)
  - gen 17: 2.6021 (sigma 0.2538)
  - gen 18: 2.6021 (sigma 0.2638)
  - gen 19: 2.5900 (sigma 0.2771)
  - gen 20: 2.4471 (sigma 0.2880)
  - gen 21: 2.4471 (sigma 0.2940)
  - gen 22: 2.4471 (sigma 0.2790)
  - gen 23: 2.4471 (sigma 0.2669)
  - gen 24: 2.4471 (sigma 0.2589)
  - gen 25: 2.4471 (sigma 0.2574)
  - gen 26: 2.4471 (sigma 0.2526)
  - gen 27: 2.4471 (sigma 0.2525)
  - gen 28: 2.4471 (sigma 0.2530)
  - gen 29: 2.4471 (sigma 0.2576)
  - gen 30: 2.4471 (sigma 0.2677)
  - gen 31: 2.4471 (sigma 0.2672)
  - gen 32: 2.4471 (sigma 0.2539)
  - gen 33: 2.4471 (sigma 0.2449)
  - gen 34: 2.2844 (sigma 0.2330)
  - gen 35: 2.2844 (sigma 0.2316)
  - gen 36: 2.2844 (sigma 0.2272)
  - gen 37: 2.2844 (sigma 0.2124)
  - gen 38: 2.2844 (sigma 0.2083)
  - gen 39: 2.2844 (sigma 0.2042)
  - gen 40: 2.2844 (sigma 0.2036)
  - gen 41: 2.2844 (sigma 0.1984)
  - gen 42: 2.2844 (sigma 0.1976)
  - gen 43: 2.2844 (sigma 0.1915)
  - gen 44: 2.2844 (sigma 0.1830)
  - gen 45: 2.2844 (sigma 0.1812)
  - gen 46: 2.2844 (sigma 0.1724)
  - gen 47: 2.2844 (sigma 0.1714)
  - gen 48: 2.2844 (sigma 0.1737)
  - gen 49: 2.2844 (sigma 0.1725)
  - gen 50: 2.2844 (sigma 0.1730)
  - gen 51: 2.2844 (sigma 0.1806)
  - gen 52: 2.2844 (sigma 0.1829)
  - gen 53: 2.2844 (sigma 0.1910)
  - gen 54: 2.2844 (sigma 0.1856)
  - gen 55: 2.2844 (sigma 0.1854)
  - gen 56: 2.2844 (sigma 0.1971)
  - gen 57: 2.2844 (sigma 0.1945)
  - gen 58: 2.2844 (sigma 0.1903)
  - gen 59: 2.2844 (sigma 0.1809)
  - gen 60: 2.2844 (sigma 0.1759)
  - gen 61: 2.2844 (sigma 0.1753)
  - gen 62: 2.2844 (sigma 0.1679)
  - gen 63: 2.2844 (sigma 0.1644)
  - gen 64: 2.2844 (sigma 0.1657)
  - gen 65: 2.2787 (sigma 0.1627)
  - gen 66: 2.2787 (sigma 0.1551)
  - gen 67: 2.2787 (sigma 0.1564)
  - gen 68: 2.2787 (sigma 0.1535)
  - gen 69: 2.2787 (sigma 0.1469)
  - gen 70: 2.2787 (sigma 0.1426)
  - gen 71: 2.2787 (sigma 0.1418)
  - gen 72: 2.2644 (sigma 0.1396)
  - gen 73: 2.2644 (sigma 0.1373)
  - gen 74: 2.2644 (sigma 0.1341)
  - gen 75: 2.2472 (sigma 0.1294)
  - gen 76: 2.2472 (sigma 0.1221)
  - gen 77: 2.2472 (sigma 0.1183)
  - gen 78: 2.2472 (sigma 0.1138)
  - gen 79: 2.2472 (sigma 0.1108)
  - gen 80: 2.2456 (sigma 0.1041)
  - gen 81: 2.2456 (sigma 0.1014)
  - gen 82: 2.2456 (sigma 0.0984)
  - gen 83: 2.2456 (sigma 0.0959)
  - gen 84: 2.2456 (sigma 0.0940)
  - gen 85: 2.2456 (sigma 0.0978)
  - gen 86: 2.2456 (sigma 0.0988)
  - gen 87: 2.2305 (sigma 0.0965)
  - gen 88: 2.2305 (sigma 0.0942)
  - gen 89: 2.2305 (sigma 0.0918)
  - gen 90: 2.2305 (sigma 0.0872)
  - gen 91: 2.2305 (sigma 0.0828)
  - gen 92: 2.2305 (sigma 0.0809)
  - gen 93: 2.2291 (sigma 0.0812)
  - gen 94: 2.2291 (sigma 0.0827)
  - gen 95: 2.2291 (sigma 0.0808)
  - gen 96: 2.2291 (sigma 0.0844)
  - gen 97: 2.2277 (sigma 0.0861)
  - gen 98: 2.2277 (sigma 0.0919)
  - gen 99: 2.2277 (sigma 0.0920)
  - gen 100: 2.2277 (sigma 0.0886)
- best-so-far improved 22 time(s); curve (trial: objective):
  - 1: 6.3842
  - 4: 3.6469
  - 9: 3.4190
  - 21: 3.3843
  - 42: 3.2122
  - 95: 3.0950
  - 103: 2.6873
  - 278: 2.6481
  - 280: 2.6021
  - 586: 2.5900
  - 616: 2.5807
  - 626: 2.5576
  - 634: 2.4471
  - 1067: 2.2844
  - 2080: 2.2787
  - 2289: 2.2644
  - 2395: 2.2472
  - 2531: 2.2456
  - 2778: 2.2305
  - 2955: 2.2291
  - 3099: 2.2277
  - 3127: 2.2277

## Fitted constants (record-only — adopt by hand)

| constant | init | fitted |
|---|---|---|
| CROP_JUNK_LINE_FRAC | 0.0800 | 0.0816 |
| CROP_LEAK_MARGIN_D | 0.0300 | 0.0300 |
| CROP_REBATE_MARGIN_D | 0.1456 | 0.0800 |
| CROP_REBATE_LINE_FRAC | 0.1827 | 0.3000 |
| CROP_REBATE_TERM_FRAC | 0.0400 | 0.0200 |
| CROP_REBATE_MAX_FRAC | 0.0269 | 0.0200 |
| CROP_REBATE_HUE_TOL | 0.0200 | 0.0200 |
| CROP_REBATE_WIDE_MAX_D | 0.3871 | 0.3494 |
| CROP_REBATE_WIDE_LINE_FRAC | 0.8000 | 0.8000 |
| CROP_REBATE_BAND_HUE_TOL | 0.0094 | 0.0066 |
| CROP_REBATE_WIDE_FRAC | 0.5500 | 0.6000 |
| CROP_PAD_FRAC | 0.0050 | 0.0043 |
| CROP_SHADOW_REL | 0.7375 | 0.8500 |
| CROP_SHADOW_MAX_FRAC | 0.0309 | 0.0287 |
| CROP_SHADOW_CORE_FRAC | 0.0167 | 0.0171 |
| CROP_GAP_TOL_FRAC | 0.0153 | 0.0232 |
| HOLDER_LUMA_THR | 0.0241 | 0.0332 |
| BORDER_MAX_FRAC | 0.0800 | 0.0800 |

## Aggregate (all rolls)

- total_overtrim_area: 1.3306
- median_overtrim_area: 0.0083
- max_overtrim_area: 0.0509
- total_undertrim_area: 0.1794
- max_undertrim_area: 0.0178
- containment_violations: 46
- containment_weight: 5.0000
- n_frames: 150

## Per-roll

### 2506-1  (prep 51.71s)
```
{
  "n": 38,
  "median_overtrim_area": 0.0048422299545054035,
  "worst_frames": [
    {
      "stem": "DSC_0020",
      "overtrim_area": 0.05093866321411231
    },
    {
      "stem": "DSC_0016",
      "overtrim_area": 0.023340705975436516
    },
    {
      "stem": "DSC_0011",
      "overtrim_area": 0.012655994317670964
    },
    {
      "stem": "DSC_0012",
      "overtrim_area": 0.010628517739296182
    },
    {
      "stem": "DSC_0023",
      "overtrim_area": 0.009159608710506913
    }
  ]
}
```
### 2510-11-1  (prep 27.93s)
```
{
  "n": 38,
  "median_overtrim_area": 0.009963930996865128,
  "worst_frames": [
    {
      "stem": "DSC_0040",
      "overtrim_area": 0.023867055678432923
    },
    {
      "stem": "DSC_0026",
      "overtrim_area": 0.02195046843250436
    },
    {
      "stem": "DSC_0032",
      "overtrim_area": 0.02171782561004118
    },
    {
      "stem": "DSC_0018",
      "overtrim_area": 0.021068973164781547
    },
    {
      "stem": "DSC_0016",
      "overtrim_area": 0.013646131161101221
    }
  ]
}
```
### 2511-12-1  (prep 28.35s)
```
{
  "n": 37,
  "median_overtrim_area": 0.008735906565247883,
  "worst_frames": [
    {
      "stem": "DSC_0027",
      "overtrim_area": 0.016054602506698316
    },
    {
      "stem": "DSC_0014",
      "overtrim_area": 0.015345809881738025
    },
    {
      "stem": "DSC_0012",
      "overtrim_area": 0.015128901356446267
    },
    {
      "stem": "DSC_0018",
      "overtrim_area": 0.011581191970413527
    },
    {
      "stem": "DSC_0006",
      "overtrim_area": 0.011552720385055714
    }
  ]
}
```
### 2512-2601-1  (prep 29.21s)
```
{
  "n": 37,
  "median_overtrim_area": 0.008518248787709865,
  "worst_frames": [
    {
      "stem": "DSC_0036",
      "overtrim_area": 0.0299614734494974
    },
    {
      "stem": "DSC_0024",
      "overtrim_area": 0.016988170805536076
    },
    {
      "stem": "DSC_0035",
      "overtrim_area": 0.014633645621669574
    },
    {
      "stem": "DSC_0007",
      "overtrim_area": 0.01432570294845744
    },
    {
      "stem": "DSC_0001",
      "overtrim_area": 0.013246779713845582
    }
  ]
}
```

## Worst frames (find where to look)

| contained | overtrim_area | undertrim_area | overtrim_edges | roll | stem |
|---|---|---|---|---|---|
| False | 0.0509 | 0.0037 | {'left': -0.0025025025025025025, 'top': 0.0007485029940119761, 'right': 0.053553553553553554, 'bottom': -0.0014970059880239522} | 2506-1 | DSC_0020 |
| True | 0.0300 | 0.0000 | {'left': 0.006006006006006006, 'top': 0.014221556886227544, 'right': 0.001001001001001001, 'bottom': 0.010479041916167664} | 2512-2601-1 | DSC_0036 |
| True | 0.0239 | 0.0000 | {'left': 0.0015015015015015015, 'top': 0.005988023952095809, 'right': 0.016016016016016016, 'bottom': 0.002245508982035928} | 2510-11-1 | DSC_0040 |
| False | 0.0233 | 0.0014 | {'left': -0.0015015015015015015, 'top': 0.011976047904191617, 'right': 0.0, 'bottom': 0.012724550898203593} | 2506-1 | DSC_0016 |
| True | 0.0220 | 0.0000 | {'left': 0.0055055055055055055, 'top': 0.004491017964071856, 'right': 0.0005005005005005005, 'bottom': 0.012724550898203593} | 2510-11-1 | DSC_0026 |
| True | 0.0217 | 0.0000 | {'left': 0.006006006006006006, 'top': 0.005239520958083832, 'right': 0.002002002002002002, 'bottom': 0.009730538922155689} | 2510-11-1 | DSC_0032 |
| False | 0.0211 | 0.0009 | {'left': -0.001001001001001001, 'top': 0.01122754491017964, 'right': 0.002002002002002002, 'bottom': 0.008982035928143712} | 2510-11-1 | DSC_0018 |
| True | 0.0170 | 0.0000 | {'left': 0.0015015015015015015, 'top': 0.004491017964071856, 'right': 0.006006006006006006, 'bottom': 0.005988023952095809} | 2512-2601-1 | DSC_0024 |
| False | 0.0161 | 0.0033 | {'left': 0.0065065065065065065, 'top': 0.0074850299401197605, 'right': -0.0035035035035035035, 'bottom': 0.0029940119760479044} | 2511-12-1 | DSC_0027 |
| True | 0.0153 | 0.0000 | {'left': 0.0015015015015015015, 'top': 0.008233532934131737, 'right': 0.005005005005005005, 'bottom': 0.0014970059880239522} | 2511-12-1 | DSC_0014 |
| True | 0.0151 | 0.0000 | {'left': 0.0015015015015015015, 'top': 0.009730538922155689, 'right': 0.001001001001001001, 'bottom': 0.0037425149700598802} | 2511-12-1 | DSC_0012 |
| True | 0.0146 | 0.0000 | {'left': 0.0015015015015015015, 'top': 0.004491017964071856, 'right': 0.002002002002002002, 'bottom': 0.0074850299401197605} | 2512-2601-1 | DSC_0035 |
| False | 0.0143 | 0.0060 | {'left': 0.0005005005005005005, 'top': 0.014970059880239521, 'right': -0.0005005005005005005, 'bottom': -0.005988023952095809} | 2512-2601-1 | DSC_0007 |
| True | 0.0136 | 0.0000 | {'left': 0.002002002002002002, 'top': 0.0074850299401197605, 'right': 0.002002002002002002, 'bottom': 0.0029940119760479044} | 2510-11-1 | DSC_0016 |
| True | 0.0135 | 0.0000 | {'left': 0.0025025025025025025, 'top': 0.005988023952095809, 'right': 0.002002002002002002, 'bottom': 0.0037425149700598802} | 2510-11-1 | DSC_0012 |
