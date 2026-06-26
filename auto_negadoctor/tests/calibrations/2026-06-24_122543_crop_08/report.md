# Calibration session — crop

- comment: first run on cmaes. compare with crop_06
- created: 2026-06-24T12:11:53
- git commit: 7bb7153
- rolls: 2506-1, 2510-11-1, 2511-12-1, 2512-2601-1
- metric: crop_overtrim_area_fraction (containment HARD)
- fit method: cmaes
- method params: sigma=0.3, popsize=12, max_iters=40, seed=0, workers=auto
- fitted params: a, l, l
- trial count: 480
- wall time: 830.29s (prep 192.82s)
- objective: 2.3159 (init) -> 2.3692 (final)

## Convergence

- method: cmaes  trials: 480
- generations run: 40 (popsize 12, sigma0 0.3, seed 0)
- per-generation best objective (sigma):
  - gen 1: 3.7722 (sigma 0.2814)
  - gen 2: 3.1506 (sigma 0.2756)
  - gen 3: 3.1506 (sigma 0.2813)
  - gen 4: 3.1506 (sigma 0.2842)
  - gen 5: 3.1506 (sigma 0.2900)
  - gen 6: 3.0475 (sigma 0.2936)
  - gen 7: 2.9087 (sigma 0.2837)
  - gen 8: 2.9087 (sigma 0.2754)
  - gen 9: 2.9087 (sigma 0.2687)
  - gen 10: 2.9087 (sigma 0.2560)
  - gen 11: 2.9087 (sigma 0.2410)
  - gen 12: 2.7218 (sigma 0.2353)
  - gen 13: 2.7218 (sigma 0.2295)
  - gen 14: 2.6524 (sigma 0.2199)
  - gen 15: 2.6524 (sigma 0.2132)
  - gen 16: 2.6524 (sigma 0.2059)
  - gen 17: 2.6524 (sigma 0.2046)
  - gen 18: 2.6524 (sigma 0.1992)
  - gen 19: 2.6524 (sigma 0.1934)
  - gen 20: 2.6524 (sigma 0.1896)
  - gen 21: 2.5735 (sigma 0.1857)
  - gen 22: 2.5735 (sigma 0.1852)
  - gen 23: 2.5735 (sigma 0.1847)
  - gen 24: 2.5735 (sigma 0.1836)
  - gen 25: 2.4599 (sigma 0.1837)
  - gen 26: 2.4599 (sigma 0.1856)
  - gen 27: 2.4599 (sigma 0.1828)
  - gen 28: 2.4599 (sigma 0.1728)
  - gen 29: 2.4599 (sigma 0.1604)
  - gen 30: 2.4382 (sigma 0.1525)
  - gen 31: 2.4382 (sigma 0.1452)
  - gen 32: 2.4382 (sigma 0.1379)
  - gen 33: 2.4382 (sigma 0.1317)
  - gen 34: 2.4382 (sigma 0.1256)
  - gen 35: 2.4382 (sigma 0.1197)
  - gen 36: 2.4382 (sigma 0.1142)
  - gen 37: 2.4382 (sigma 0.1161)
  - gen 38: 2.4382 (sigma 0.1119)
  - gen 39: 2.4382 (sigma 0.1072)
  - gen 40: 2.3692 (sigma 0.1004)
- best-so-far improved 14 time(s); curve (trial: objective):
  - 1: 6.2595
  - 7: 4.4765
  - 11: 3.7722
  - 18: 3.1506
  - 66: 3.0810
  - 70: 3.0475
  - 83: 2.9087
  - 139: 2.7218
  - 166: 2.6524
  - 246: 2.6283
  - 250: 2.5735
  - 299: 2.4599
  - 359: 2.4382
  - 469: 2.3692

## Fitted constants (record-only — adopt by hand)

| constant | init | fitted |
|---|---|---|
| CROP_JUNK_LINE_FRAC | 0.0800 | 0.1000 |
| CROP_LEAK_MARGIN_D | 0.0300 | 0.0466 |
| CROP_REBATE_MARGIN_D | 0.1456 | 0.0800 |
| CROP_REBATE_LINE_FRAC | 0.1827 | 0.2745 |
| CROP_REBATE_TERM_FRAC | 0.0400 | 0.0648 |
| CROP_REBATE_MAX_FRAC | 0.0269 | 0.0361 |
| CROP_REBATE_HUE_TOL | 0.0200 | 0.0299 |
| CROP_REBATE_WIDE_MAX_D | 0.3871 | 0.4246 |
| CROP_REBATE_WIDE_LINE_FRAC | 0.8000 | 0.8000 |
| CROP_REBATE_BAND_HUE_TOL | 0.0094 | 0.0084 |
| CROP_REBATE_WIDE_FRAC | 0.5500 | 0.6000 |
| CROP_PAD_FRAC | 0.0050 | 0.0040 |
| CROP_SHADOW_REL | 0.7375 | 0.7604 |
| CROP_SHADOW_MAX_FRAC | 0.0309 | 0.0328 |
| CROP_SHADOW_CORE_FRAC | 0.0167 | 0.0169 |
| CROP_GAP_TOL_FRAC | 0.0153 | 0.0213 |
| HOLDER_LUMA_THR | 0.0241 | 0.0294 |
| BORDER_MAX_FRAC | 0.0800 | 0.1244 |

## Aggregate (all rolls)

- total_overtrim_area: 0.9171
- median_overtrim_area: 0.0053
- max_overtrim_area: 0.0212
- total_undertrim_area: 0.2904
- max_undertrim_area: 0.0142
- containment_violations: 73
- containment_weight: 5.0000
- n_frames: 150

## Per-roll

### 2506-1  (prep 55.04s)
```
{
  "n": 38,
  "median_overtrim_area": 0.0022385783987580394,
  "worst_frames": [
    {
      "stem": "DSC_0011",
      "overtrim_area": 0.011952446458434482
    },
    {
      "stem": "DSC_0023",
      "overtrim_area": 0.007283855711999424
    },
    {
      "stem": "DSC_0017",
      "overtrim_area": 0.006562700424975874
    },
    {
      "stem": "DSC_0032",
      "overtrim_area": 0.0049799050547553545
    },
    {
      "stem": "DSC_0024",
      "overtrim_area": 0.004927082771394148
    }
  ]
}
```
### 2510-11-1  (prep 46.04s)
```
{
  "n": 38,
  "median_overtrim_area": 0.006870455785126444,
  "worst_frames": [
    {
      "stem": "DSC_0017",
      "overtrim_area": 0.021236431041820263
    },
    {
      "stem": "DSC_0010",
      "overtrim_area": 0.017735175295055534
    },
    {
      "stem": "DSC_0018",
      "overtrim_area": 0.015608422794051537
    },
    {
      "stem": "DSC_0032",
      "overtrim_area": 0.014651627675579771
    },
    {
      "stem": "DSC_0011",
      "overtrim_area": 0.012494905084725444
    }
  ]
}
```
### 2511-12-1  (prep 44.29s)
```
{
  "n": 37,
  "median_overtrim_area": 0.0066331301361241485,
  "worst_frames": [
    {
      "stem": "DSC_0014",
      "overtrim_area": 0.020154510798223373
    },
    {
      "stem": "DSC_0006",
      "overtrim_area": 0.012744780708852566
    },
    {
      "stem": "DSC_0038",
      "overtrim_area": 0.012237536938135741
    },
    {
      "stem": "DSC_0024",
      "overtrim_area": 0.012060713408018797
    },
    {
      "stem": "DSC_0018",
      "overtrim_area": 0.010420600240959522
    }
  ]
}
```
### 2512-2601-1  (prep 47.45s)
```
{
  "n": 37,
  "median_overtrim_area": 0.005544466622310933,
  "worst_frames": [
    {
      "stem": "DSC_0033",
      "overtrim_area": 0.017166867466268665
    },
    {
      "stem": "DSC_0002",
      "overtrim_area": 0.0164558270845696
    },
    {
      "stem": "DSC_0005",
      "overtrim_area": 0.011909364454274634
    },
    {
      "stem": "DSC_0035",
      "overtrim_area": 0.011818704932477388
    },
    {
      "stem": "DSC_0014",
      "overtrim_area": 0.011619778461095827
    }
  ]
}
```

## Worst frames (find where to look)

| contained | overtrim_area | undertrim_area | overtrim_edges | roll | stem |
|---|---|---|---|---|---|
| True | 0.0212 | 0.0000 | {'left': 0.0005005005005005005, 'top': 0.0074850299401197605, 'right': 0.001001001001001001, 'bottom': 0.01347305389221557} | 2510-11-1 | DSC_0017 |
| True | 0.0202 | 0.0000 | {'left': 0.0005005005005005005, 'top': 0.020209580838323353, 'right': 0.0005005005005005005, 'bottom': 0.0} | 2511-12-1 | DSC_0014 |
| False | 0.0177 | 0.0023 | {'left': -0.0025025025025025025, 'top': 0.014970059880239521, 'right': 0.0015015015015015015, 'bottom': 0.002245508982035928} | 2510-11-1 | DSC_0010 |
| True | 0.0172 | 0.0000 | {'left': 0.0015015015015015015, 'top': 0.0029940119760479044, 'right': 0.011511511511511512, 'bottom': 0.002245508982035928} | 2512-2601-1 | DSC_0033 |
| True | 0.0165 | 0.0007 | {'left': 0.011011011011011011, 'top': 0.004491017964071856, 'right': 0.002002002002002002, 'bottom': -0.0007485029940119761} | 2512-2601-1 | DSC_0002 |
| True | 0.0156 | 0.0000 | {'left': 0.0015015015015015015, 'top': 0.010479041916167664, 'right': 0.0015015015015015015, 'bottom': 0.0029940119760479044} | 2510-11-1 | DSC_0018 |
| True | 0.0147 | 0.0000 | {'left': 0.0055055055055055055, 'top': 0.005239520958083832, 'right': 0.001001001001001001, 'bottom': 0.0037425149700598802} | 2510-11-1 | DSC_0032 |
| True | 0.0127 | 0.0000 | {'left': 0.0, 'top': 0.005239520958083832, 'right': 0.0, 'bottom': 0.008233532934131737} | 2511-12-1 | DSC_0006 |
| True | 0.0125 | 0.0000 | {'left': 0.0005005005005005005, 'top': 0.008982035928143712, 'right': 0.0015015015015015015, 'bottom': 0.002245508982035928} | 2510-11-1 | DSC_0011 |
| False | 0.0122 | 0.0028 | {'left': 0.0025025025025025025, 'top': 0.002245508982035928, 'right': -0.003003003003003003, 'bottom': 0.008233532934131737} | 2511-12-1 | DSC_0038 |
| True | 0.0121 | 0.0000 | {'left': 0.0025025025025025025, 'top': 0.008233532934131737, 'right': 0.002002002002002002, 'bottom': 0.0} | 2511-12-1 | DSC_0024 |
| False | 0.0120 | 0.0047 | {'left': 0.010510510510510511, 'top': -0.0029940119760479044, 'right': -0.002002002002002002, 'bottom': 0.002245508982035928} | 2506-1 | DSC_0011 |
| False | 0.0119 | 0.0028 | {'left': -0.001001001001001001, 'top': 0.0074850299401197605, 'right': -0.002002002002002002, 'bottom': 0.005239520958083832} | 2512-2601-1 | DSC_0005 |
| True | 0.0118 | 0.0000 | {'left': 0.0005005005005005005, 'top': 0.0037425149700598802, 'right': 0.0015015015015015015, 'bottom': 0.006736526946107785} | 2512-2601-1 | DSC_0035 |
| True | 0.0116 | 0.0000 | {'left': 0.006006006006006006, 'top': 0.004491017964071856, 'right': 0.001001001001001001, 'bottom': 0.0007485029940119761} | 2512-2601-1 | DSC_0014 |
