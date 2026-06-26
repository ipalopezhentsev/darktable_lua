# Calibration session — crop

- comment: 4th run on cmaes, with higher sigma to explore more possibilities initially. compare with crop_10
- created: 2026-06-24T16:25:28
- git commit: 7bb7153
- rolls: 2506-1, 2510-11-1, 2511-12-1, 2512-2601-1
- metric: crop_overtrim_area_fraction (containment HARD)
- fit method: cmaes
- method params: sigma=0.6, popsize=16, max_iters=100, seed=0, workers=16
- fitted params: a, l, l
- trial count: 1600
- wall time: 1468.37s (prep 223.36s)
- objective: 2.3159 (init) -> 2.4204 (final)

## Convergence

- method: cmaes  trials: 1600
- generations run: 100 (popsize 16, sigma0 0.6, seed 0)
- per-generation best objective (sigma):
  - gen 1: 4.6810 (sigma 0.5820)
  - gen 2: 2.8609 (sigma 0.5743)
  - gen 3: 2.8609 (sigma 0.5512)
  - gen 4: 2.8609 (sigma 0.5459)
  - gen 5: 2.8609 (sigma 0.5497)
  - gen 6: 2.8609 (sigma 0.5434)
  - gen 7: 2.8609 (sigma 0.5422)
  - gen 8: 2.8609 (sigma 0.5049)
  - gen 9: 2.8609 (sigma 0.4884)
  - gen 10: 2.8609 (sigma 0.4942)
  - gen 11: 2.8609 (sigma 0.5074)
  - gen 12: 2.8609 (sigma 0.5337)
  - gen 13: 2.8609 (sigma 0.5441)
  - gen 14: 2.8609 (sigma 0.5391)
  - gen 15: 2.8609 (sigma 0.5247)
  - gen 16: 2.8609 (sigma 0.4956)
  - gen 17: 2.8609 (sigma 0.4814)
  - gen 18: 2.8609 (sigma 0.4788)
  - gen 19: 2.8609 (sigma 0.4688)
  - gen 20: 2.8609 (sigma 0.4565)
  - gen 21: 2.8609 (sigma 0.4381)
  - gen 22: 2.8609 (sigma 0.4119)
  - gen 23: 2.8609 (sigma 0.4063)
  - gen 24: 2.8428 (sigma 0.3799)
  - gen 25: 2.8428 (sigma 0.3655)
  - gen 26: 2.8428 (sigma 0.3736)
  - gen 27: 2.8428 (sigma 0.3758)
  - gen 28: 2.8428 (sigma 0.3612)
  - gen 29: 2.8428 (sigma 0.3464)
  - gen 30: 2.8428 (sigma 0.3339)
  - gen 31: 2.8428 (sigma 0.3320)
  - gen 32: 2.8428 (sigma 0.3255)
  - gen 33: 2.8428 (sigma 0.3271)
  - gen 34: 2.8428 (sigma 0.3317)
  - gen 35: 2.8428 (sigma 0.3311)
  - gen 36: 2.8428 (sigma 0.3340)
  - gen 37: 2.8428 (sigma 0.3159)
  - gen 38: 2.8428 (sigma 0.2942)
  - gen 39: 2.8428 (sigma 0.2748)
  - gen 40: 2.8428 (sigma 0.2654)
  - gen 41: 2.8428 (sigma 0.2685)
  - gen 42: 2.8428 (sigma 0.2781)
  - gen 43: 2.8428 (sigma 0.2733)
  - gen 44: 2.8428 (sigma 0.2676)
  - gen 45: 2.7384 (sigma 0.2773)
  - gen 46: 2.7087 (sigma 0.2834)
  - gen 47: 2.7087 (sigma 0.2981)
  - gen 48: 2.7087 (sigma 0.3015)
  - gen 49: 2.7087 (sigma 0.3038)
  - gen 50: 2.5775 (sigma 0.3231)
  - gen 51: 2.5724 (sigma 0.3354)
  - gen 52: 2.5724 (sigma 0.3597)
  - gen 53: 2.5724 (sigma 0.3761)
  - gen 54: 2.5573 (sigma 0.3939)
  - gen 55: 2.5573 (sigma 0.3969)
  - gen 56: 2.5573 (sigma 0.4013)
  - gen 57: 2.5573 (sigma 0.4259)
  - gen 58: 2.5573 (sigma 0.4291)
  - gen 59: 2.5573 (sigma 0.4248)
  - gen 60: 2.5573 (sigma 0.4023)
  - gen 61: 2.5573 (sigma 0.3854)
  - gen 62: 2.5573 (sigma 0.3699)
  - gen 63: 2.5573 (sigma 0.3389)
  - gen 64: 2.5573 (sigma 0.3288)
  - gen 65: 2.5573 (sigma 0.3178)
  - gen 66: 2.4943 (sigma 0.3057)
  - gen 67: 2.4943 (sigma 0.2964)
  - gen 68: 2.4943 (sigma 0.2754)
  - gen 69: 2.4566 (sigma 0.2586)
  - gen 70: 2.4259 (sigma 0.2487)
  - gen 71: 2.4259 (sigma 0.2325)
  - gen 72: 2.4259 (sigma 0.2282)
  - gen 73: 2.4259 (sigma 0.2161)
  - gen 74: 2.4259 (sigma 0.2132)
  - gen 75: 2.4259 (sigma 0.2056)
  - gen 76: 2.4259 (sigma 0.1964)
  - gen 77: 2.4259 (sigma 0.1971)
  - gen 78: 2.4259 (sigma 0.1867)
  - gen 79: 2.4259 (sigma 0.1809)
  - gen 80: 2.4259 (sigma 0.1712)
  - gen 81: 2.4259 (sigma 0.1659)
  - gen 82: 2.4259 (sigma 0.1532)
  - gen 83: 2.4259 (sigma 0.1405)
  - gen 84: 2.4259 (sigma 0.1367)
  - gen 85: 2.4259 (sigma 0.1357)
  - gen 86: 2.4259 (sigma 0.1295)
  - gen 87: 2.4259 (sigma 0.1248)
  - gen 88: 2.4259 (sigma 0.1235)
  - gen 89: 2.4259 (sigma 0.1258)
  - gen 90: 2.4259 (sigma 0.1225)
  - gen 91: 2.4259 (sigma 0.1199)
  - gen 92: 2.4259 (sigma 0.1136)
  - gen 93: 2.4259 (sigma 0.1087)
  - gen 94: 2.4259 (sigma 0.1068)
  - gen 95: 2.4259 (sigma 0.1038)
  - gen 96: 2.4259 (sigma 0.0996)
  - gen 97: 2.4259 (sigma 0.0973)
  - gen 98: 2.4259 (sigma 0.0963)
  - gen 99: 2.4204 (sigma 0.0985)
  - gen 100: 2.4204 (sigma 0.0986)
- best-so-far improved 15 time(s); curve (trial: objective):
  - 1: 6.3521
  - 7: 5.8356
  - 13: 4.6810
  - 29: 2.8609
  - 371: 2.8428
  - 711: 2.7384
  - 728: 2.7087
  - 786: 2.6381
  - 790: 2.5775
  - 811: 2.5724
  - 855: 2.5573
  - 1044: 2.4943
  - 1103: 2.4566
  - 1118: 2.4259
  - 1570: 2.4204

## Fitted constants (record-only — adopt by hand)

| constant | init | fitted |
|---|---|---|
| CROP_JUNK_LINE_FRAC | 0.0800 | 0.1000 |
| CROP_LEAK_MARGIN_D | 0.0300 | 0.1187 |
| CROP_REBATE_MARGIN_D | 0.1456 | 0.0800 |
| CROP_REBATE_LINE_FRAC | 0.1827 | 0.3000 |
| CROP_REBATE_TERM_FRAC | 0.0400 | 0.0200 |
| CROP_REBATE_MAX_FRAC | 0.0269 | 0.0322 |
| CROP_REBATE_HUE_TOL | 0.0200 | 0.0600 |
| CROP_REBATE_WIDE_MAX_D | 0.3871 | 0.5000 |
| CROP_REBATE_WIDE_LINE_FRAC | 0.8000 | 0.4313 |
| CROP_REBATE_BAND_HUE_TOL | 0.0094 | 0.0040 |
| CROP_REBATE_WIDE_FRAC | 0.5500 | 0.6000 |
| CROP_PAD_FRAC | 0.0050 | 0.0040 |
| CROP_SHADOW_REL | 0.7375 | 0.8500 |
| CROP_SHADOW_MAX_FRAC | 0.0309 | 0.0291 |
| CROP_SHADOW_CORE_FRAC | 0.0167 | 0.0169 |
| CROP_GAP_TOL_FRAC | 0.0153 | 0.0273 |
| HOLDER_LUMA_THR | 0.0241 | 0.0416 |
| BORDER_MAX_FRAC | 0.0800 | 0.0800 |

## Aggregate (all rolls)

- total_overtrim_area: 1.1109
- median_overtrim_area: 0.0065
- max_overtrim_area: 0.1000
- total_undertrim_area: 0.2619
- max_undertrim_area: 0.0201
- containment_violations: 61
- containment_weight: 5.0000
- n_frames: 150

## Per-roll

### 2506-1  (prep 57.64s)
```
{
  "n": 38,
  "median_overtrim_area": 0.002846409283535032,
  "worst_frames": [
    {
      "stem": "DSC_0016",
      "overtrim_area": 0.02192611773450097
    },
    {
      "stem": "DSC_0011",
      "overtrim_area": 0.011484913056769344
    },
    {
      "stem": "DSC_0012",
      "overtrim_area": 0.008742275209341078
    },
    {
      "stem": "DSC_0023",
      "overtrim_area": 0.007987778197359036
    },
    {
      "stem": "DSC_0008",
      "overtrim_area": 0.005445565325804847
    }
  ]
}
```
### 2510-11-1  (prep 63.43s)
```
{
  "n": 38,
  "median_overtrim_area": 0.007585429741118364,
  "worst_frames": [
    {
      "stem": "DSC_0026",
      "overtrim_area": 0.020311479143814475
    },
    {
      "stem": "DSC_0018",
      "overtrim_area": 0.019648390905875935
    },
    {
      "stem": "DSC_0032",
      "overtrim_area": 0.01937041832251413
    },
    {
      "stem": "DSC_0012",
      "overtrim_area": 0.011596551641461821
    },
    {
      "stem": "DSC_0009",
      "overtrim_area": 0.011400622179065293
    }
  ]
}
```
### 2511-12-1  (prep 60.85s)
```
{
  "n": 37,
  "median_overtrim_area": 0.007093171015326704,
  "worst_frames": [
    {
      "stem": "DSC_0027",
      "overtrim_area": 0.014637017256777737
    },
    {
      "stem": "DSC_0014",
      "overtrim_area": 0.01299315782848717
    },
    {
      "stem": "DSC_0012",
      "overtrim_area": 0.01277699855544167
    },
    {
      "stem": "DSC_0032",
      "overtrim_area": 0.010420225614836394
    },
    {
      "stem": "DSC_0018",
      "overtrim_area": 0.009935084785384187
    }
  ]
}
```
### 2512-2601-1  (prep 41.44s)
```
{
  "n": 37,
  "median_overtrim_area": 0.006859029688371006,
  "worst_frames": [
    {
      "stem": "DSC_0029",
      "overtrim_area": 0.10002067936199673
    },
    {
      "stem": "DSC_0036",
      "overtrim_area": 0.027625304945664227
    },
    {
      "stem": "DSC_0024",
      "overtrim_area": 0.014637766509023995
    },
    {
      "stem": "DSC_0007",
      "overtrim_area": 0.013168108227988468
    },
    {
      "stem": "DSC_0035",
      "overtrim_area": 0.01228473982965001
    }
  ]
}
```

## Worst frames (find where to look)

| contained | overtrim_area | undertrim_area | overtrim_edges | roll | stem |
|---|---|---|---|---|---|
| False | 0.1000 | 0.0020 | {'left': 0.001001001001001001, 'top': 0.05089820359281437, 'right': 0.057057057057057055, 'bottom': -0.002245508982035928} | 2512-2601-1 | DSC_0029 |
| True | 0.0276 | 0.0000 | {'left': 0.0055055055055055055, 'top': 0.01347305389221557, 'right': 0.0005005005005005005, 'bottom': 0.009730538922155689} | 2512-2601-1 | DSC_0036 |
| False | 0.0219 | 0.0023 | {'left': -0.002002002002002002, 'top': 0.01122754491017964, 'right': -0.0005005005005005005, 'bottom': 0.011976047904191617} | 2506-1 | DSC_0016 |
| True | 0.0203 | 0.0000 | {'left': 0.005005005005005005, 'top': 0.004491017964071856, 'right': 0.0, 'bottom': 0.011976047904191617} | 2510-11-1 | DSC_0026 |
| False | 0.0196 | 0.0014 | {'left': -0.0015015015015015015, 'top': 0.010479041916167664, 'right': 0.002002002002002002, 'bottom': 0.008233532934131737} | 2510-11-1 | DSC_0018 |
| True | 0.0194 | 0.0000 | {'left': 0.0055055055055055055, 'top': 0.004491017964071856, 'right': 0.0015015015015015015, 'bottom': 0.008982035928143712} | 2510-11-1 | DSC_0032 |
| True | 0.0146 | 0.0000 | {'left': 0.001001001001001001, 'top': 0.0037425149700598802, 'right': 0.0055055055055055055, 'bottom': 0.005239520958083832} | 2512-2601-1 | DSC_0024 |
| True | 0.0146 | 0.0000 | {'left': 0.006006006006006006, 'top': 0.006736526946107785, 'right': 0.0005005005005005005, 'bottom': 0.002245508982035928} | 2511-12-1 | DSC_0027 |
| False | 0.0132 | 0.0044 | {'left': 0.0, 'top': 0.014221556886227544, 'right': -0.001001001001001001, 'bottom': -0.0037425149700598802} | 2512-2601-1 | DSC_0007 |
| True | 0.0130 | 0.0000 | {'left': 0.001001001001001001, 'top': 0.0074850299401197605, 'right': 0.0045045045045045045, 'bottom': 0.0007485029940119761} | 2511-12-1 | DSC_0014 |
| True | 0.0128 | 0.0000 | {'left': 0.001001001001001001, 'top': 0.008982035928143712, 'right': 0.0005005005005005005, 'bottom': 0.0029940119760479044} | 2511-12-1 | DSC_0012 |
| True | 0.0123 | 0.0000 | {'left': 0.001001001001001001, 'top': 0.0037425149700598802, 'right': 0.0015015015015015015, 'bottom': 0.006736526946107785} | 2512-2601-1 | DSC_0035 |
| True | 0.0116 | 0.0000 | {'left': 0.0025025025025025025, 'top': 0.005239520958083832, 'right': 0.0015015015015015015, 'bottom': 0.0029940119760479044} | 2510-11-1 | DSC_0012 |
| False | 0.0115 | 0.0028 | {'left': 0.01001001001001001, 'top': -0.0014970059880239522, 'right': -0.0015015015015015015, 'bottom': 0.002245508982035928} | 2506-1 | DSC_0011 |
| True | 0.0114 | 0.0009 | {'left': -0.0005005005005005005, 'top': 0.008982035928143712, 'right': -0.0005005005005005005, 'bottom': 0.0029940119760479044} | 2510-11-1 | DSC_0009 |
