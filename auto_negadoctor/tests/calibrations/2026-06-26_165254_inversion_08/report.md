# Calibration session — inversion

- comment: 100 iters of cmaes after fixing thread pools. compare with _02. and compare speed with _07
- created: 2026-06-26T01:05:27
- git commit: 7bb7153
- rolls: 2506-1, 2510-11-1
- metric: histogram_emd_total (render vs GT-render, picture-vs-picture)
- fit method: cmaes
- method params: sigma=0.3, popsize=16, max_iters=100, seed=0, workers=auto
- fitted params: a, l, l
- trial count: 1600
- wall time: 56844.29s (prep 77.38s)
- objective: 0.0556 (init) -> 0.0397 (final)

## Convergence

- method: cmaes  trials: 1600
- generations run: 100 (popsize 16, sigma0 0.3, seed 0)
- per-generation best objective (sigma):
  - gen 1: 0.0551 (sigma 0.2750)
  - gen 2: 0.0551 (sigma 0.2600)
  - gen 3: 0.0551 (sigma 0.2471)
  - gen 4: 0.0517 (sigma 0.2368)
  - gen 5: 0.0506 (sigma 0.2387)
  - gen 6: 0.0438 (sigma 0.2425)
  - gen 7: 0.0438 (sigma 0.2404)
  - gen 8: 0.0431 (sigma 0.2392)
  - gen 9: 0.0431 (sigma 0.2421)
  - gen 10: 0.0431 (sigma 0.2381)
  - gen 11: 0.0431 (sigma 0.2393)
  - gen 12: 0.0431 (sigma 0.2438)
  - gen 13: 0.0431 (sigma 0.2510)
  - gen 14: 0.0431 (sigma 0.2542)
  - gen 15: 0.0431 (sigma 0.2536)
  - gen 16: 0.0431 (sigma 0.2492)
  - gen 17: 0.0431 (sigma 0.2473)
  - gen 18: 0.0431 (sigma 0.2485)
  - gen 19: 0.0431 (sigma 0.2497)
  - gen 20: 0.0431 (sigma 0.2475)
  - gen 21: 0.0431 (sigma 0.2453)
  - gen 22: 0.0431 (sigma 0.2418)
  - gen 23: 0.0431 (sigma 0.2362)
  - gen 24: 0.0431 (sigma 0.2355)
  - gen 25: 0.0431 (sigma 0.2415)
  - gen 26: 0.0431 (sigma 0.2407)
  - gen 27: 0.0431 (sigma 0.2431)
  - gen 28: 0.0431 (sigma 0.2410)
  - gen 29: 0.0431 (sigma 0.2389)
  - gen 30: 0.0431 (sigma 0.2319)
  - gen 31: 0.0431 (sigma 0.2298)
  - gen 32: 0.0431 (sigma 0.2283)
  - gen 33: 0.0431 (sigma 0.2263)
  - gen 34: 0.0431 (sigma 0.2311)
  - gen 35: 0.0431 (sigma 0.2376)
  - gen 36: 0.0431 (sigma 0.2347)
  - gen 37: 0.0406 (sigma 0.2322)
  - gen 38: 0.0406 (sigma 0.2307)
  - gen 39: 0.0406 (sigma 0.2290)
  - gen 40: 0.0406 (sigma 0.2259)
  - gen 41: 0.0406 (sigma 0.2251)
  - gen 42: 0.0406 (sigma 0.2220)
  - gen 43: 0.0406 (sigma 0.2203)
  - gen 44: 0.0406 (sigma 0.2184)
  - gen 45: 0.0406 (sigma 0.2191)
  - gen 46: 0.0406 (sigma 0.2228)
  - gen 47: 0.0406 (sigma 0.2277)
  - gen 48: 0.0406 (sigma 0.2297)
  - gen 49: 0.0406 (sigma 0.2315)
  - gen 50: 0.0406 (sigma 0.2299)
  - gen 51: 0.0406 (sigma 0.2270)
  - gen 52: 0.0401 (sigma 0.2241)
  - gen 53: 0.0401 (sigma 0.2191)
  - gen 54: 0.0401 (sigma 0.2126)
  - gen 55: 0.0401 (sigma 0.2064)
  - gen 56: 0.0401 (sigma 0.1977)
  - gen 57: 0.0401 (sigma 0.1948)
  - gen 58: 0.0401 (sigma 0.1916)
  - gen 59: 0.0401 (sigma 0.1898)
  - gen 60: 0.0401 (sigma 0.1854)
  - gen 61: 0.0401 (sigma 0.1818)
  - gen 62: 0.0401 (sigma 0.1786)
  - gen 63: 0.0401 (sigma 0.1782)
  - gen 64: 0.0400 (sigma 0.1750)
  - gen 65: 0.0400 (sigma 0.1728)
  - gen 66: 0.0400 (sigma 0.1678)
  - gen 67: 0.0400 (sigma 0.1661)
  - gen 68: 0.0400 (sigma 0.1688)
  - gen 69: 0.0400 (sigma 0.1698)
  - gen 70: 0.0400 (sigma 0.1695)
  - gen 71: 0.0397 (sigma 0.1669)
  - gen 72: 0.0397 (sigma 0.1672)
  - gen 73: 0.0397 (sigma 0.1687)
  - gen 74: 0.0397 (sigma 0.1682)
  - gen 75: 0.0397 (sigma 0.1683)
  - gen 76: 0.0397 (sigma 0.1626)
  - gen 77: 0.0397 (sigma 0.1571)
  - gen 78: 0.0397 (sigma 0.1508)
  - gen 79: 0.0397 (sigma 0.1456)
  - gen 80: 0.0397 (sigma 0.1412)
  - gen 81: 0.0397 (sigma 0.1413)
  - gen 82: 0.0397 (sigma 0.1376)
  - gen 83: 0.0397 (sigma 0.1350)
  - gen 84: 0.0397 (sigma 0.1332)
  - gen 85: 0.0397 (sigma 0.1320)
  - gen 86: 0.0397 (sigma 0.1311)
  - gen 87: 0.0397 (sigma 0.1284)
  - gen 88: 0.0397 (sigma 0.1260)
  - gen 89: 0.0397 (sigma 0.1230)
  - gen 90: 0.0397 (sigma 0.1219)
  - gen 91: 0.0397 (sigma 0.1210)
  - gen 92: 0.0397 (sigma 0.1186)
  - gen 93: 0.0397 (sigma 0.1200)
  - gen 94: 0.0397 (sigma 0.1193)
  - gen 95: 0.0397 (sigma 0.1187)
  - gen 96: 0.0397 (sigma 0.1166)
  - gen 97: 0.0397 (sigma 0.1168)
  - gen 98: 0.0397 (sigma 0.1178)
  - gen 99: 0.0397 (sigma 0.1204)
  - gen 100: 0.0397 (sigma 0.1220)
- best-so-far improved 10 time(s); curve (trial: objective):
  - 1: 0.0568
  - 11: 0.0551
  - 61: 0.0517
  - 80: 0.0506
  - 84: 0.0438
  - 121: 0.0431
  - 586: 0.0406
  - 820: 0.0401
  - 1017: 0.0400
  - 1121: 0.0397

## Fitted constants (record-only — adopt by hand)

| constant | init | fitted |
|---|---|---|
| P_LOW | 0.5000 | 0.5000 |
| P_HIGH | 99.7227 | 98.6699 |
| OFFSET_DEFAULT | 0.0063 | 0.0275 |
| DMAX_DEFAULT | 1.7218 | 1.5000 |
| PATCH_WIN_FRAC | 0.0494 | 0.0200 |
| PATCH_STRIDE_DIV | 2.0000 | 1.0000 |
| HIGHLIGHT_BAND_PCT[0] | 73.7500 | 80.0000 |
| HIGHLIGHT_BAND_PCT[1] | 96.7500 | 99.0000 |
| SHADOW_MIN_LUMA | 0.0050 | 0.0010 |
| PATCH_CHROMA_MAX | 0.3500 | 0.6000 |
| PATCH_CHROMA_FLOOR | 0.0200 | 0.0500 |
| PATCH_UNIFORMITY_MAX | 0.4500 | 0.2000 |
| PATCH_LUMA_FLOOR | 0.0200 | 0.0315 |
| MIN_PATCH_DENSITY | 0.0500 | 0.0100 |
| HIGHLIGHT_CLIP_FRAC_MAX | 0.0200 | 0.0050 |
| WB_LOW_BAND_PCT[0] | 0.0000 | 0.0000 |
| WB_LOW_BAND_PCT[1] | 20.0000 | 47.5672 |
| WB_HIGH_BAND_PCT[0] | 70.0000 | 64.8009 |
| WB_HIGH_BAND_PCT[1] | 95.0000 | 99.0000 |
| WB_LOW_DESAT | 0.0000 | 0.2424 |
| WB_HIGH_DESAT | 0.0000 | 0.0504 |
| WB_REGION_MIN_FRAC | 0.0001 | 0.0004 |
| WB_HIGH_PRIOR[0] | 1.7590 | 1.9929 |
| WB_HIGH_PRIOR[1] | 1.3656 | 1.3730 |
| WB_HIGH_PRIOR[2] | 1.0641 | 1.2000 |
| WB_LOW_PRIOR[0] | 1.0891 | 0.9924 |
| WB_LOW_PRIOR[1] | 0.7461 | 1.0000 |
| WB_LOW_PRIOR[2] | 0.6941 | 0.5000 |
| PRINT_HI_PCT | 99.9155 | 99.9900 |
| PRINT_HI_CEIL | 0.9693 | 0.9337 |
| PRINT_CLIP_BUDGET | 0.0030 | 0.0200 |
| PRINT_TUNE_ITERS | 11.0000 | 24.0000 |
| PRINT_GAMMA | 4.5312 | 4.8222 |

## Aggregate (all rolls)

- median_total: 0.0397
- median_luma: 0.0395
- median_color: 0.0201
- median_hi999: 0.0203
- max_clip: 0.0000
- frames_over_clip_budget: 0
- clip_budget: 0.0100
- n_frames: 76

## Per-roll

### 2506-1  (prep 37.17s)
```
{
  "n": 38,
  "median_total": 0.03386307317557245,
  "worst_frames": [
    {
      "stem": "DSC_0032",
      "total": 0.155924336769042
    },
    {
      "stem": "DSC_0029",
      "total": 0.07769721983255477
    },
    {
      "stem": "DSC_0020",
      "total": 0.06888174977122104
    },
    {
      "stem": "DSC_0030",
      "total": 0.06759703297338722
    },
    {
      "stem": "DSC_0015",
      "total": 0.06337155249046225
    }
  ]
}
```
### 2510-11-1  (prep 40.21s)
```
{
  "n": 38,
  "median_total": 0.05908078855722464,
  "worst_frames": [
    {
      "stem": "DSC_0010",
      "total": 0.17056556263149503
    },
    {
      "stem": "DSC_0007",
      "total": 0.14695263232887898
    },
    {
      "stem": "DSC_0008",
      "total": 0.1440285290555102
    },
    {
      "stem": "DSC_0005",
      "total": 0.13165361315345656
    },
    {
      "stem": "DSC_0017",
      "total": 0.11939221352905073
    }
  ]
}
```

## Worst frames (find where to look)

| roll | stem | total | luma | color | hi999 | clip |
|---|---|---|---|---|---|---|
| 2510-11-1 | DSC_0010 | 0.1706 | 0.1947 | 0.0718 | 0.0409 | 0.0000 |
| 2506-1 | DSC_0032 | 0.1559 | 0.1649 | 0.0184 | 0.0763 | 0.0000 |
| 2510-11-1 | DSC_0007 | 0.1470 | 0.1665 | 0.0436 | 0.0386 | 0.0000 |
| 2510-11-1 | DSC_0008 | 0.1440 | 0.1785 | 0.0665 | 0.0628 | 0.0000 |
| 2510-11-1 | DSC_0005 | 0.1317 | 0.1519 | 0.0383 | 0.0334 | 0.0000 |
| 2510-11-1 | DSC_0017 | 0.1194 | 0.0862 | 0.1048 | 0.0243 | 0.0000 |
| 2510-11-1 | DSC_0004 | 0.1094 | 0.1215 | 0.0234 | 0.0308 | 0.0000 |
| 2510-11-1 | DSC_0006 | 0.1020 | 0.1184 | 0.0407 | 0.0605 | 0.0000 |
| 2510-11-1 | DSC_0009 | 0.0880 | 0.0992 | 0.0201 | 0.0431 | 0.0000 |
| 2510-11-1 | DSC_0033 | 0.0869 | 0.0939 | 0.0443 | 0.0074 | 0.0000 |
| 2510-11-1 | DSC_0037 | 0.0856 | 0.0950 | 0.0231 | 0.0281 | 0.0000 |
| 2510-11-1 | DSC_0003 | 0.0829 | 0.1014 | 0.0449 | 0.0300 | 0.0000 |
| 2506-1 | DSC_0029 | 0.0777 | 0.0696 | 0.0138 | 0.0274 | 0.0000 |
| 2510-11-1 | DSC_0020 | 0.0750 | 0.0774 | 0.0171 | 0.0218 | 0.0000 |
| 2510-11-1 | DSC_0030 | 0.0702 | 0.0782 | 0.0412 | 0.0119 | 0.0000 |
