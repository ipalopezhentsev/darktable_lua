# Calibration session — crop

- comment: cmaes with infinite penalty on reaching out of my crop annotations. compare with crop_10
- created: 2026-06-24T18:26:15
- git commit: 7bb7153
- rolls: 2506-1, 2510-11-1, 2511-12-1, 2512-2601-1
- metric: crop_overtrim_area_fraction (containment HARD)
- fit method: cmaes
- method params: sigma=0.3, popsize=16, max_iters=100, seed=0, workers=16
- fitted params: a, l, l
- trial count: 1600
- wall time: 1314.37s (prep 109.26s)
- objective: 40000001.5915 (init) -> 6.7028 (final)

## Convergence

- method: cmaes  trials: 1600
- generations run: 100 (popsize 16, sigma0 0.3, seed 0)
- per-generation best objective (sigma):
  - gen 1: 6000007.4331 (sigma 0.2987)
  - gen 2: 1000015.8768 (sigma 0.3257)
  - gen 3: 11.6047 (sigma 0.3636)
  - gen 4: 11.6047 (sigma 0.3802)
  - gen 5: 11.6047 (sigma 0.3862)
  - gen 6: 11.6047 (sigma 0.4155)
  - gen 7: 11.2747 (sigma 0.4508)
  - gen 8: 10.9037 (sigma 0.4793)
  - gen 9: 10.9037 (sigma 0.4853)
  - gen 10: 10.9037 (sigma 0.4754)
  - gen 11: 10.2767 (sigma 0.4726)
  - gen 12: 10.1269 (sigma 0.4691)
  - gen 13: 10.1269 (sigma 0.4740)
  - gen 14: 9.8581 (sigma 0.4627)
  - gen 15: 9.8581 (sigma 0.4509)
  - gen 16: 9.8581 (sigma 0.4430)
  - gen 17: 9.8581 (sigma 0.4231)
  - gen 18: 9.8581 (sigma 0.4204)
  - gen 19: 9.8581 (sigma 0.3912)
  - gen 20: 9.8581 (sigma 0.3649)
  - gen 21: 9.8581 (sigma 0.3677)
  - gen 22: 9.8581 (sigma 0.3811)
  - gen 23: 9.8581 (sigma 0.3933)
  - gen 24: 9.8581 (sigma 0.3969)
  - gen 25: 9.8581 (sigma 0.3905)
  - gen 26: 9.8581 (sigma 0.3827)
  - gen 27: 9.7001 (sigma 0.3942)
  - gen 28: 9.6432 (sigma 0.3973)
  - gen 29: 9.6432 (sigma 0.3945)
  - gen 30: 9.2466 (sigma 0.4227)
  - gen 31: 9.2466 (sigma 0.4309)
  - gen 32: 9.2466 (sigma 0.4456)
  - gen 33: 9.2466 (sigma 0.4416)
  - gen 34: 9.2466 (sigma 0.4351)
  - gen 35: 9.2466 (sigma 0.4255)
  - gen 36: 8.2879 (sigma 0.4221)
  - gen 37: 8.2879 (sigma 0.4216)
  - gen 38: 8.2879 (sigma 0.4070)
  - gen 39: 8.2879 (sigma 0.3843)
  - gen 40: 8.2879 (sigma 0.3775)
  - gen 41: 8.2879 (sigma 0.3719)
  - gen 42: 8.2879 (sigma 0.3731)
  - gen 43: 8.2879 (sigma 0.3533)
  - gen 44: 8.2879 (sigma 0.3339)
  - gen 45: 7.4851 (sigma 0.3267)
  - gen 46: 7.4851 (sigma 0.3208)
  - gen 47: 7.4851 (sigma 0.3079)
  - gen 48: 7.4851 (sigma 0.2967)
  - gen 49: 7.4851 (sigma 0.2757)
  - gen 50: 7.4851 (sigma 0.2629)
  - gen 51: 7.4851 (sigma 0.2427)
  - gen 52: 7.4851 (sigma 0.2249)
  - gen 53: 7.4851 (sigma 0.2073)
  - gen 54: 7.4851 (sigma 0.1971)
  - gen 55: 7.4851 (sigma 0.1931)
  - gen 56: 7.4851 (sigma 0.1963)
  - gen 57: 7.4851 (sigma 0.2021)
  - gen 58: 7.4851 (sigma 0.1971)
  - gen 59: 7.4851 (sigma 0.1907)
  - gen 60: 7.4851 (sigma 0.1875)
  - gen 61: 7.4851 (sigma 0.1722)
  - gen 62: 7.4851 (sigma 0.1657)
  - gen 63: 7.4851 (sigma 0.1590)
  - gen 64: 7.4851 (sigma 0.1536)
  - gen 65: 7.4851 (sigma 0.1487)
  - gen 66: 7.3795 (sigma 0.1437)
  - gen 67: 7.3795 (sigma 0.1462)
  - gen 68: 7.3795 (sigma 0.1564)
  - gen 69: 7.3795 (sigma 0.1719)
  - gen 70: 7.3795 (sigma 0.1814)
  - gen 71: 7.3795 (sigma 0.1848)
  - gen 72: 7.3795 (sigma 0.1916)
  - gen 73: 7.3795 (sigma 0.1960)
  - gen 74: 7.3795 (sigma 0.2015)
  - gen 75: 7.3795 (sigma 0.2066)
  - gen 76: 7.3795 (sigma 0.2123)
  - gen 77: 7.3795 (sigma 0.2043)
  - gen 78: 7.3795 (sigma 0.1978)
  - gen 79: 6.9839 (sigma 0.1817)
  - gen 80: 6.9839 (sigma 0.1801)
  - gen 81: 6.9839 (sigma 0.1774)
  - gen 82: 6.9839 (sigma 0.1706)
  - gen 83: 6.9839 (sigma 0.1699)
  - gen 84: 6.9839 (sigma 0.1654)
  - gen 85: 6.7902 (sigma 0.1659)
  - gen 86: 6.7902 (sigma 0.1654)
  - gen 87: 6.7902 (sigma 0.1605)
  - gen 88: 6.7028 (sigma 0.1632)
  - gen 89: 6.7028 (sigma 0.1658)
  - gen 90: 6.7028 (sigma 0.1622)
  - gen 91: 6.7028 (sigma 0.1560)
  - gen 92: 6.7028 (sigma 0.1567)
  - gen 93: 6.7028 (sigma 0.1536)
  - gen 94: 6.7028 (sigma 0.1503)
  - gen 95: 6.7028 (sigma 0.1511)
  - gen 96: 6.7028 (sigma 0.1529)
  - gen 97: 6.7028 (sigma 0.1461)
  - gen 98: 6.7028 (sigma 0.1379)
  - gen 99: 6.7028 (sigma 0.1313)
  - gen 100: 6.7028 (sigma 0.1238)
- best-so-far improved 25 time(s); curve (trial: objective):
  - 1: 129000001.1062
  - 2: 61000002.3883
  - 4: 15000005.3257
  - 5: 10000006.4780
  - 8: 8000005.3571
  - 13: 6000007.4331
  - 17: 5000010.6117
  - 18: 5000006.9141
  - 24: 1000015.8768
  - 41: 12.6385
  - 47: 11.6047
  - 102: 11.2747
  - 122: 10.9037
  - 169: 10.2767
  - 178: 10.1269
  - 221: 9.8581
  - 420: 9.7001
  - 443: 9.6432
  - 473: 9.2466
  - 563: 8.2879
  - 716: 7.4851
  - 1052: 7.3795
  - 1252: 6.9839
  - 1356: 6.7902
  - 1404: 6.7028

## Fitted constants (record-only — adopt by hand)

| constant | init | fitted |
|---|---|---|
| CROP_JUNK_LINE_FRAC | 0.0800 | 0.0702 |
| CROP_LEAK_MARGIN_D | 0.0300 | 0.1200 |
| CROP_REBATE_MARGIN_D | 0.1456 | 0.0800 |
| CROP_REBATE_LINE_FRAC | 0.1827 | 0.2512 |
| CROP_REBATE_TERM_FRAC | 0.0400 | 0.0200 |
| CROP_REBATE_MAX_FRAC | 0.0269 | 0.0800 |
| CROP_REBATE_HUE_TOL | 0.0200 | 0.0200 |
| CROP_REBATE_WIDE_MAX_D | 0.3871 | 0.3622 |
| CROP_REBATE_WIDE_LINE_FRAC | 0.8000 | 0.5479 |
| CROP_REBATE_BAND_HUE_TOL | 0.0094 | 0.0088 |
| CROP_REBATE_WIDE_FRAC | 0.5500 | 0.6000 |
| CROP_PAD_FRAC | 0.0050 | 0.0134 |
| CROP_SHADOW_REL | 0.7375 | 0.8064 |
| CROP_SHADOW_MAX_FRAC | 0.0309 | 0.0200 |
| CROP_SHADOW_CORE_FRAC | 0.0167 | 0.0181 |
| CROP_GAP_TOL_FRAC | 0.0153 | 0.0400 |
| HOLDER_LUMA_THR | 0.0241 | 0.0100 |
| BORDER_MAX_FRAC | 0.0800 | 0.0800 |

## Aggregate (all rolls)

- total_overtrim_area: 6.7028
- median_overtrim_area: 0.0450
- max_overtrim_area: 0.0563
- total_undertrim_area: 0.0021
- max_undertrim_area: 0.0007
- containment_violations: 0
- containment_weight: None
- n_frames: 150

## Per-roll

### 2506-1  (prep 25.03s)
```
{
  "n": 38,
  "median_overtrim_area": 0.04153479827132522,
  "worst_frames": [
    {
      "stem": "DSC_0011",
      "overtrim_area": 0.051832895770021516
    },
    {
      "stem": "DSC_0031",
      "overtrim_area": 0.0515200829572087
    },
    {
      "stem": "DSC_0016",
      "overtrim_area": 0.05024935114755474
    },
    {
      "stem": "DSC_0009",
      "overtrim_area": 0.05007102911294528
    },
    {
      "stem": "DSC_0030",
      "overtrim_area": 0.049891583199966436
    }
  ]
}
```
### 2510-11-1  (prep 28.94s)
```
{
  "n": 38,
  "median_overtrim_area": 0.0481149188110266,
  "worst_frames": [
    {
      "stem": "DSC_0016",
      "overtrim_area": 0.056265097432762104
    },
    {
      "stem": "DSC_0003",
      "overtrim_area": 0.055823787859716006
    },
    {
      "stem": "DSC_0010",
      "overtrim_area": 0.053888469307630985
    },
    {
      "stem": "DSC_0034",
      "overtrim_area": 0.05350035664406922
    },
    {
      "stem": "DSC_0017",
      "overtrim_area": 0.05348911786037534
    }
  ]
}
```
### 2511-12-1  (prep 27.37s)
```
{
  "n": 37,
  "median_overtrim_area": 0.04606927286568005,
  "worst_frames": [
    {
      "stem": "DSC_0012",
      "overtrim_area": 0.05197263131394868
    },
    {
      "stem": "DSC_0011",
      "overtrim_area": 0.051586017154879434
    },
    {
      "stem": "DSC_0018",
      "overtrim_area": 0.05132977288665912
    },
    {
      "stem": "DSC_0030",
      "overtrim_area": 0.050806794818770866
    },
    {
      "stem": "DSC_0027",
      "overtrim_area": 0.04997325169480858
    }
  ]
}
```
### 2512-2601-1  (prep 27.92s)
```
{
  "n": 37,
  "median_overtrim_area": 0.044280807753861645,
  "worst_frames": [
    {
      "stem": "DSC_0014",
      "overtrim_area": 0.05404019288749828
    },
    {
      "stem": "DSC_0006",
      "overtrim_area": 0.05381878884872897
    },
    {
      "stem": "DSC_0020",
      "overtrim_area": 0.05082627537717358
    },
    {
      "stem": "DSC_0036",
      "overtrim_area": 0.05060449671228114
    },
    {
      "stem": "DSC_0005",
      "overtrim_area": 0.050233616850383314
    }
  ]
}
```

## Worst frames (find where to look)

| contained | overtrim_area | undertrim_area | overtrim_edges | roll | stem |
|---|---|---|---|---|---|
| True | 0.0563 | 0.0000 | {'left': 0.011011011011011011, 'top': 0.02245508982035928, 'right': 0.010510510510510511, 'bottom': 0.016467065868263474} | 2510-11-1 | DSC_0016 |
| True | 0.0558 | 0.0000 | {'left': 0.00850850850850851, 'top': 0.023952095808383235, 'right': 0.01001001001001001, 'bottom': 0.01721556886227545} | 2510-11-1 | DSC_0003 |
| True | 0.0540 | 0.0000 | {'left': 0.013513513513513514, 'top': 0.020209580838323353, 'right': 0.008008008008008008, 'bottom': 0.015718562874251496} | 2512-2601-1 | DSC_0014 |
| True | 0.0539 | 0.0000 | {'left': 0.01901901901901902, 'top': 0.01347305389221557, 'right': 0.0075075075075075074, 'bottom': 0.017964071856287425} | 2510-11-1 | DSC_0010 |
| True | 0.0538 | 0.0000 | {'left': 0.010510510510510511, 'top': 0.020209580838323353, 'right': 0.011011011011011011, 'bottom': 0.016467065868263474} | 2512-2601-1 | DSC_0006 |
| True | 0.0535 | 0.0000 | {'left': 0.012012012012012012, 'top': 0.020958083832335328, 'right': 0.01001001001001001, 'bottom': 0.014221556886227544} | 2510-11-1 | DSC_0034 |
| True | 0.0535 | 0.0000 | {'left': 0.01001001001001001, 'top': 0.02470059880239521, 'right': 0.006006006006006006, 'bottom': 0.016467065868263474} | 2510-11-1 | DSC_0017 |
| True | 0.0530 | 0.0000 | {'left': 0.016016016016016016, 'top': 0.014221556886227544, 'right': 0.0015015015015015015, 'bottom': 0.02470059880239521} | 2510-11-1 | DSC_0032 |
| True | 0.0524 | 0.0000 | {'left': 0.011511511511511512, 'top': 0.02619760479041916, 'right': 0.0055055055055055055, 'bottom': 0.012724550898203593} | 2510-11-1 | DSC_0018 |
| True | 0.0520 | 0.0000 | {'left': 0.01001001001001001, 'top': 0.023203592814371257, 'right': 0.005005005005005005, 'bottom': 0.01721556886227545} | 2511-12-1 | DSC_0012 |
| True | 0.0518 | 0.0000 | {'left': 0.02052052052052052, 'top': 0.012724550898203593, 'right': 0.0045045045045045045, 'bottom': 0.017964071856287425} | 2506-1 | DSC_0011 |
| True | 0.0516 | 0.0000 | {'left': 0.007007007007007007, 'top': 0.02470059880239521, 'right': 0.010510510510510511, 'bottom': 0.012724550898203593} | 2511-12-1 | DSC_0011 |
| True | 0.0515 | 0.0000 | {'left': 0.010510510510510511, 'top': 0.01721556886227545, 'right': 0.011011011011011011, 'bottom': 0.016467065868263474} | 2506-1 | DSC_0031 |
| True | 0.0514 | 0.0000 | {'left': 0.0075075075075075074, 'top': 0.0187125748502994, 'right': 0.006006006006006006, 'bottom': 0.02245508982035928} | 2510-11-1 | DSC_0026 |
| True | 0.0513 | 0.0000 | {'left': 0.008008008008008008, 'top': 0.023203592814371257, 'right': 0.007007007007007007, 'bottom': 0.016467065868263474} | 2511-12-1 | DSC_0018 |
