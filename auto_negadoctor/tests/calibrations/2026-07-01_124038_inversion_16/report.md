# Calibration session — inversion

- comment: CMAES on 4 rolls without downsampling. After all fixes to GT and calib
- created: 2026-06-29T09:08:37
- git commit: 644b3a2
- rolls: 2506-1, 2510-11-1, 2511-12-1, 2512-2601-1
- metric: histogram_emd_total (render vs GT-render, picture-vs-picture)
- fit method: cmaes
- method params: sigma=0.3, popsize=16, max_iters=100, seed=1, workers=1
- fitted params: a, l, l
- trial count: 1600
- wall time: 185514.16s (prep 255.91s)
- objective: 0.0602 (init) -> 0.0388 (final)

## Convergence

- method: cmaes  trials: 1600
- generations run: 100 (popsize 16, sigma0 0.3, seed 1)
- per-generation best objective (sigma):
  - gen 1: 0.0582 (sigma 0.2831)
  - gen 2: 0.0552 (sigma 0.2710)
  - gen 3: 0.0552 (sigma 0.2625)
  - gen 4: 0.0552 (sigma 0.2562)
  - gen 5: 0.0552 (sigma 0.2551)
  - gen 6: 0.0512 (sigma 0.2527)
  - gen 7: 0.0477 (sigma 0.2566)
  - gen 8: 0.0477 (sigma 0.2590)
  - gen 9: 0.0475 (sigma 0.2616)
  - gen 10: 0.0446 (sigma 0.2636)
  - gen 11: 0.0446 (sigma 0.2577)
  - gen 12: 0.0446 (sigma 0.2510)
  - gen 13: 0.0446 (sigma 0.2464)
  - gen 14: 0.0446 (sigma 0.2439)
  - gen 15: 0.0446 (sigma 0.2457)
  - gen 16: 0.0446 (sigma 0.2451)
  - gen 17: 0.0446 (sigma 0.2443)
  - gen 18: 0.0446 (sigma 0.2444)
  - gen 19: 0.0446 (sigma 0.2428)
  - gen 20: 0.0446 (sigma 0.2428)
  - gen 21: 0.0446 (sigma 0.2421)
  - gen 22: 0.0446 (sigma 0.2395)
  - gen 23: 0.0446 (sigma 0.2398)
  - gen 24: 0.0446 (sigma 0.2429)
  - gen 25: 0.0446 (sigma 0.2468)
  - gen 26: 0.0446 (sigma 0.2477)
  - gen 27: 0.0446 (sigma 0.2465)
  - gen 28: 0.0446 (sigma 0.2472)
  - gen 29: 0.0446 (sigma 0.2482)
  - gen 30: 0.0446 (sigma 0.2407)
  - gen 31: 0.0446 (sigma 0.2355)
  - gen 32: 0.0446 (sigma 0.2289)
  - gen 33: 0.0446 (sigma 0.2236)
  - gen 34: 0.0446 (sigma 0.2220)
  - gen 35: 0.0446 (sigma 0.2150)
  - gen 36: 0.0446 (sigma 0.2104)
  - gen 37: 0.0446 (sigma 0.2092)
  - gen 38: 0.0446 (sigma 0.2085)
  - gen 39: 0.0446 (sigma 0.2095)
  - gen 40: 0.0443 (sigma 0.2126)
  - gen 41: 0.0443 (sigma 0.2133)
  - gen 42: 0.0443 (sigma 0.2114)
  - gen 43: 0.0443 (sigma 0.2106)
  - gen 44: 0.0443 (sigma 0.2088)
  - gen 45: 0.0443 (sigma 0.2002)
  - gen 46: 0.0443 (sigma 0.1959)
  - gen 47: 0.0443 (sigma 0.1921)
  - gen 48: 0.0443 (sigma 0.1903)
  - gen 49: 0.0443 (sigma 0.1902)
  - gen 50: 0.0443 (sigma 0.1898)
  - gen 51: 0.0443 (sigma 0.1855)
  - gen 52: 0.0443 (sigma 0.1799)
  - gen 53: 0.0443 (sigma 0.1740)
  - gen 54: 0.0443 (sigma 0.1708)
  - gen 55: 0.0443 (sigma 0.1725)
  - gen 56: 0.0431 (sigma 0.1709)
  - gen 57: 0.0431 (sigma 0.1708)
  - gen 58: 0.0431 (sigma 0.1707)
  - gen 59: 0.0431 (sigma 0.1721)
  - gen 60: 0.0431 (sigma 0.1701)
  - gen 61: 0.0431 (sigma 0.1667)
  - gen 62: 0.0431 (sigma 0.1640)
  - gen 63: 0.0424 (sigma 0.1644)
  - gen 64: 0.0403 (sigma 0.1639)
  - gen 65: 0.0403 (sigma 0.1647)
  - gen 66: 0.0403 (sigma 0.1668)
  - gen 67: 0.0401 (sigma 0.1658)
  - gen 68: 0.0401 (sigma 0.1675)
  - gen 69: 0.0401 (sigma 0.1712)
  - gen 70: 0.0401 (sigma 0.1733)
  - gen 71: 0.0401 (sigma 0.1707)
  - gen 72: 0.0401 (sigma 0.1661)
  - gen 73: 0.0401 (sigma 0.1623)
  - gen 74: 0.0401 (sigma 0.1571)
  - gen 75: 0.0401 (sigma 0.1551)
  - gen 76: 0.0401 (sigma 0.1538)
  - gen 77: 0.0401 (sigma 0.1514)
  - gen 78: 0.0401 (sigma 0.1488)
  - gen 79: 0.0401 (sigma 0.1410)
  - gen 80: 0.0401 (sigma 0.1364)
  - gen 81: 0.0401 (sigma 0.1337)
  - gen 82: 0.0401 (sigma 0.1356)
  - gen 83: 0.0401 (sigma 0.1355)
  - gen 84: 0.0401 (sigma 0.1325)
  - gen 85: 0.0401 (sigma 0.1321)
  - gen 86: 0.0401 (sigma 0.1319)
  - gen 87: 0.0401 (sigma 0.1332)
  - gen 88: 0.0388 (sigma 0.1329)
  - gen 89: 0.0388 (sigma 0.1337)
  - gen 90: 0.0388 (sigma 0.1351)
  - gen 91: 0.0388 (sigma 0.1375)
  - gen 92: 0.0388 (sigma 0.1381)
  - gen 93: 0.0388 (sigma 0.1363)
  - gen 94: 0.0388 (sigma 0.1335)
  - gen 95: 0.0388 (sigma 0.1342)
  - gen 96: 0.0388 (sigma 0.1317)
  - gen 97: 0.0388 (sigma 0.1290)
  - gen 98: 0.0388 (sigma 0.1261)
  - gen 99: 0.0388 (sigma 0.1270)
  - gen 100: 0.0388 (sigma 0.1262)
- best-so-far improved 19 time(s); curve (trial: objective):
  - 1: 0.1059
  - 2: 0.0726
  - 5: 0.0620
  - 6: 0.0582
  - 18: 0.0552
  - 95: 0.0512
  - 97: 0.0477
  - 132: 0.0475
  - 151: 0.0462
  - 160: 0.0446
  - 631: 0.0443
  - 892: 0.0431
  - 922: 0.0431
  - 999: 0.0424
  - 1015: 0.0416
  - 1023: 0.0403
  - 1069: 0.0401
  - 1393: 0.0392
  - 1398: 0.0388

## Fitted constants (record-only — adopt by hand)

| constant | init | fitted |
|---|---|---|
| P_LOW | 0.5000 | 0.5000 |
| P_HIGH | 99.7227 | 99.9000 |
| OFFSET_DEFAULT | 0.0063 | -0.0501 |
| DMAX_DEFAULT | 1.7218 | 1.5000 |
| PATCH_WIN_FRAC | 0.0494 | 0.0200 |
| PATCH_STRIDE_DIV | 2.0000 | 2.6782 |
| HIGHLIGHT_BAND_PCT[0] | 73.7500 | 50.0000 |
| HIGHLIGHT_BAND_PCT[1] | 96.7500 | 99.0000 |
| SHADOW_MIN_LUMA | 0.0050 | 0.0014 |
| PATCH_CHROMA_MAX | 0.3500 | 0.6000 |
| PATCH_CHROMA_FLOOR | 0.0200 | 0.0500 |
| PATCH_UNIFORMITY_MAX | 0.4500 | 0.2710 |
| PATCH_LUMA_FLOOR | 0.0200 | 0.0419 |
| MIN_PATCH_DENSITY | 0.0500 | 0.1066 |
| HIGHLIGHT_CLIP_FRAC_MAX | 0.0200 | 0.0050 |
| WB_LOW_BAND_PCT[0] | 0.0000 | 9.8925 |
| WB_LOW_BAND_PCT[1] | 20.0000 | 50.0000 |
| WB_HIGH_BAND_PCT[0] | 70.0000 | 50.0000 |
| WB_HIGH_BAND_PCT[1] | 95.0000 | 97.6193 |
| WB_LOW_DESAT | 0.0000 | 0.0000 |
| WB_HIGH_DESAT | 0.0000 | 0.0000 |
| WB_REGION_MIN_FRAC | 0.0001 | 0.0007 |
| WB_HIGH_PRIOR[0] | 1.7590 | 1.7815 |
| WB_HIGH_PRIOR[1] | 1.3656 | 1.5170 |
| WB_HIGH_PRIOR[2] | 1.0641 | 1.2000 |
| WB_LOW_PRIOR[0] | 1.0891 | 0.8000 |
| WB_LOW_PRIOR[1] | 0.7461 | 1.0000 |
| WB_LOW_PRIOR[2] | 0.6941 | 0.5000 |
| PRINT_HI_PCT | 99.9155 | 99.9900 |
| PRINT_HI_CEIL | 0.9693 | 0.9271 |
| PRINT_CLIP_BUDGET | 0.0030 | 0.0000 |
| PRINT_TUNE_ITERS | 11.0000 | 11.3843 |
| PRINT_GAMMA | 4.5312 | 5.1134 |

## Aggregate (all rolls)

- median_total: 0.0388
- median_luma: 0.0395
- median_color: 0.0222
- median_hi999: 0.0234
- max_clip: 0.0000
- frames_over_clip_budget: 0
- clip_budget: 0.0100
- n_frames: 150

## Per-roll

### 2506-1  (prep 63.91s)
```
{
  "n": 38,
  "median_total": 0.03129231667332183,
  "worst_frames": [
    {
      "stem": "DSC_0032",
      "total": 0.16176412901342999
    },
    {
      "stem": "DSC_0029",
      "total": 0.1438129698270657
    },
    {
      "stem": "DSC_0024",
      "total": 0.11276843829648901
    },
    {
      "stem": "DSC_0013",
      "total": 0.07119258291445135
    },
    {
      "stem": "DSC_0008",
      "total": 0.06048837300286849
    }
  ]
}
```
### 2510-11-1  (prep 66.9s)
```
{
  "n": 38,
  "median_total": 0.044348325970591754,
  "worst_frames": [
    {
      "stem": "DSC_0010",
      "total": 0.14850171874291365
    },
    {
      "stem": "DSC_0007",
      "total": 0.12874124533101652
    },
    {
      "stem": "DSC_0008",
      "total": 0.1169376830591069
    },
    {
      "stem": "DSC_0004",
      "total": 0.1020398004167253
    },
    {
      "stem": "DSC_0019",
      "total": 0.09477745148801385
    }
  ]
}
```
### 2511-12-1  (prep 64.13s)
```
{
  "n": 37,
  "median_total": 0.03870105142180107,
  "worst_frames": [
    {
      "stem": "DSC_0038",
      "total": 0.12141311852443588
    },
    {
      "stem": "DSC_0025",
      "total": 0.09455548037819411
    },
    {
      "stem": "DSC_0015",
      "total": 0.09432620264528786
    },
    {
      "stem": "DSC_0030",
      "total": 0.08389008811041543
    },
    {
      "stem": "DSC_0005",
      "total": 0.07879024167837258
    }
  ]
}
```
### 2512-2601-1  (prep 60.97s)
```
{
  "n": 37,
  "median_total": 0.06729217177213233,
  "worst_frames": [
    {
      "stem": "DSC_0008",
      "total": 0.23347483984702877
    },
    {
      "stem": "DSC_0035",
      "total": 0.20710858143811317
    },
    {
      "stem": "DSC_0002",
      "total": 0.20127290687010038
    },
    {
      "stem": "DSC_0009",
      "total": 0.1952541440093624
    },
    {
      "stem": "DSC_0013",
      "total": 0.17568671461327404
    }
  ]
}
```

## Worst frames (find where to look)

| roll | stem | total | luma | color | hi999 | clip |
|---|---|---|---|---|---|---|
| 2512-2601-1 | DSC_0008 | 0.2335 | 0.2458 | 0.0222 | 0.0646 | 0.0000 |
| 2512-2601-1 | DSC_0035 | 0.2071 | 0.2047 | 0.0083 | 0.1123 | 0.0000 |
| 2512-2601-1 | DSC_0002 | 0.2013 | 0.1004 | 0.1887 | 0.0459 | 0.0000 |
| 2512-2601-1 | DSC_0009 | 0.1953 | 0.1975 | 0.0033 | 0.0894 | 0.0000 |
| 2512-2601-1 | DSC_0013 | 0.1757 | 0.1846 | 0.0235 | 0.0554 | 0.0000 |
| 2512-2601-1 | DSC_0012 | 0.1699 | 0.1772 | 0.0180 | 0.1120 | 0.0000 |
| 2506-1 | DSC_0032 | 0.1618 | 0.1703 | 0.0283 | 0.0862 | 0.0000 |
| 2512-2601-1 | DSC_0014 | 0.1508 | 0.1577 | 0.0132 | 0.0601 | 0.0000 |
| 2510-11-1 | DSC_0010 | 0.1485 | 0.1754 | 0.0717 | 0.0430 | 0.0000 |
| 2506-1 | DSC_0029 | 0.1438 | 0.1656 | 0.0442 | 0.0809 | 0.0000 |
| 2512-2601-1 | DSC_0037 | 0.1356 | 0.1400 | 0.0074 | 0.0309 | 0.0000 |
| 2512-2601-1 | DSC_0011 | 0.1336 | 0.1387 | 0.0122 | 0.0581 | 0.0000 |
| 2510-11-1 | DSC_0007 | 0.1287 | 0.1561 | 0.0540 | 0.0461 | 0.0000 |
| 2511-12-1 | DSC_0038 | 0.1214 | 0.1234 | 0.0156 | 0.0785 | 0.0000 |
| 2510-11-1 | DSC_0008 | 0.1169 | 0.1537 | 0.0653 | 0.0628 | 0.0000 |
