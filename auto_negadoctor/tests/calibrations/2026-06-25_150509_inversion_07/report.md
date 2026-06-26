# Calibration session — inversion

- comment: 100 iters of cmaes!
- created: 2026-06-24T23:00:00
- git commit: 7bb7153
- rolls: 2506-1, 2510-11-1
- metric: histogram_emd_total (render vs GT-render, picture-vs-picture)
- fit method: cmaes
- method params: sigma=0.3, popsize=16, max_iters=100, seed=0, workers=1
- fitted params: a, l, l
- trial count: 1600
- wall time: 57907.85s (prep 57.8s)
- objective: 0.0556 (init) -> 0.0385 (final)

## Convergence

- method: cmaes  trials: 1600
- generations run: 100 (popsize 16, sigma0 0.3, seed 0)
- per-generation best objective (sigma):
  - gen 1: 0.0488 (sigma 0.2799)
  - gen 2: 0.0481 (sigma 0.2615)
  - gen 3: 0.0481 (sigma 0.2491)
  - gen 4: 0.0458 (sigma 0.2408)
  - gen 5: 0.0437 (sigma 0.2356)
  - gen 6: 0.0437 (sigma 0.2322)
  - gen 7: 0.0437 (sigma 0.2323)
  - gen 8: 0.0421 (sigma 0.2395)
  - gen 9: 0.0421 (sigma 0.2504)
  - gen 10: 0.0421 (sigma 0.2572)
  - gen 11: 0.0421 (sigma 0.2658)
  - gen 12: 0.0389 (sigma 0.2728)
  - gen 13: 0.0389 (sigma 0.2840)
  - gen 14: 0.0389 (sigma 0.2912)
  - gen 15: 0.0385 (sigma 0.3039)
  - gen 16: 0.0385 (sigma 0.3113)
  - gen 17: 0.0385 (sigma 0.3159)
  - gen 18: 0.0385 (sigma 0.3263)
  - gen 19: 0.0385 (sigma 0.3297)
  - gen 20: 0.0385 (sigma 0.3272)
  - gen 21: 0.0385 (sigma 0.3292)
  - gen 22: 0.0385 (sigma 0.3270)
  - gen 23: 0.0385 (sigma 0.3280)
  - gen 24: 0.0385 (sigma 0.3302)
  - gen 25: 0.0385 (sigma 0.3339)
  - gen 26: 0.0385 (sigma 0.3303)
  - gen 27: 0.0385 (sigma 0.3315)
  - gen 28: 0.0385 (sigma 0.3350)
  - gen 29: 0.0385 (sigma 0.3403)
  - gen 30: 0.0385 (sigma 0.3429)
  - gen 31: 0.0385 (sigma 0.3381)
  - gen 32: 0.0385 (sigma 0.3332)
  - gen 33: 0.0385 (sigma 0.3326)
  - gen 34: 0.0385 (sigma 0.3377)
  - gen 35: 0.0385 (sigma 0.3379)
  - gen 36: 0.0385 (sigma 0.3339)
  - gen 37: 0.0385 (sigma 0.3346)
  - gen 38: 0.0385 (sigma 0.3381)
  - gen 39: 0.0385 (sigma 0.3403)
  - gen 40: 0.0385 (sigma 0.3423)
  - gen 41: 0.0385 (sigma 0.3378)
  - gen 42: 0.0385 (sigma 0.3341)
  - gen 43: 0.0385 (sigma 0.3327)
  - gen 44: 0.0385 (sigma 0.3337)
  - gen 45: 0.0385 (sigma 0.3338)
  - gen 46: 0.0385 (sigma 0.3323)
  - gen 47: 0.0385 (sigma 0.3280)
  - gen 48: 0.0385 (sigma 0.3224)
  - gen 49: 0.0385 (sigma 0.3176)
  - gen 50: 0.0385 (sigma 0.3198)
  - gen 51: 0.0385 (sigma 0.3207)
  - gen 52: 0.0385 (sigma 0.3114)
  - gen 53: 0.0385 (sigma 0.3068)
  - gen 54: 0.0385 (sigma 0.2991)
  - gen 55: 0.0385 (sigma 0.2912)
  - gen 56: 0.0385 (sigma 0.2840)
  - gen 57: 0.0385 (sigma 0.2800)
  - gen 58: 0.0385 (sigma 0.2742)
  - gen 59: 0.0385 (sigma 0.2647)
  - gen 60: 0.0385 (sigma 0.2612)
  - gen 61: 0.0385 (sigma 0.2549)
  - gen 62: 0.0385 (sigma 0.2472)
  - gen 63: 0.0385 (sigma 0.2404)
  - gen 64: 0.0385 (sigma 0.2321)
  - gen 65: 0.0385 (sigma 0.2242)
  - gen 66: 0.0385 (sigma 0.2164)
  - gen 67: 0.0385 (sigma 0.2077)
  - gen 68: 0.0385 (sigma 0.2017)
  - gen 69: 0.0385 (sigma 0.1985)
  - gen 70: 0.0385 (sigma 0.1967)
  - gen 71: 0.0385 (sigma 0.1931)
  - gen 72: 0.0385 (sigma 0.1877)
  - gen 73: 0.0385 (sigma 0.1846)
  - gen 74: 0.0385 (sigma 0.1782)
  - gen 75: 0.0385 (sigma 0.1777)
  - gen 76: 0.0385 (sigma 0.1756)
  - gen 77: 0.0385 (sigma 0.1713)
  - gen 78: 0.0385 (sigma 0.1667)
  - gen 79: 0.0385 (sigma 0.1648)
  - gen 80: 0.0385 (sigma 0.1655)
  - gen 81: 0.0385 (sigma 0.1643)
  - gen 82: 0.0385 (sigma 0.1608)
  - gen 83: 0.0385 (sigma 0.1576)
  - gen 84: 0.0385 (sigma 0.1569)
  - gen 85: 0.0385 (sigma 0.1508)
  - gen 86: 0.0385 (sigma 0.1475)
  - gen 87: 0.0385 (sigma 0.1482)
  - gen 88: 0.0385 (sigma 0.1460)
  - gen 89: 0.0385 (sigma 0.1442)
  - gen 90: 0.0385 (sigma 0.1445)
  - gen 91: 0.0385 (sigma 0.1433)
  - gen 92: 0.0385 (sigma 0.1454)
  - gen 93: 0.0385 (sigma 0.1481)
  - gen 94: 0.0385 (sigma 0.1496)
  - gen 95: 0.0385 (sigma 0.1535)
  - gen 96: 0.0385 (sigma 0.1589)
  - gen 97: 0.0385 (sigma 0.1625)
  - gen 98: 0.0385 (sigma 0.1687)
  - gen 99: 0.0385 (sigma 0.1723)
  - gen 100: 0.0385 (sigma 0.1733)
- best-so-far improved 8 time(s); curve (trial: objective):
  - 1: 0.0488
  - 20: 0.0481
  - 51: 0.0458
  - 77: 0.0437
  - 125: 0.0422
  - 128: 0.0421
  - 189: 0.0389
  - 231: 0.0385

## Fitted constants (record-only — adopt by hand)

| constant | init | fitted |
|---|---|---|
| P_LOW | 0.5000 | 0.5000 |
| P_HIGH | 99.7227 | 98.1098 |
| OFFSET_DEFAULT | 0.0063 | -0.0220 |
| DMAX_DEFAULT | 1.7218 | 1.5000 |
| PATCH_WIN_FRAC | 0.0494 | 0.0200 |
| PATCH_STRIDE_DIV | 2.0000 | 2.5305 |
| HIGHLIGHT_BAND_PCT[0] | 73.7500 | 63.9937 |
| HIGHLIGHT_BAND_PCT[1] | 96.7500 | 99.0000 |
| SHADOW_MIN_LUMA | 0.0050 | 0.0010 |
| PATCH_CHROMA_MAX | 0.3500 | 0.5375 |
| PATCH_CHROMA_FLOOR | 0.0200 | 0.0423 |
| PATCH_UNIFORMITY_MAX | 0.4500 | 0.7000 |
| PATCH_LUMA_FLOOR | 0.0200 | 0.0373 |
| MIN_PATCH_DENSITY | 0.0500 | 0.1159 |
| HIGHLIGHT_CLIP_FRAC_MAX | 0.0200 | 0.0500 |
| WB_LOW_BAND_PCT[0] | 0.0000 | 7.5408 |
| WB_LOW_BAND_PCT[1] | 20.0000 | 20.0000 |
| WB_HIGH_BAND_PCT[0] | 70.0000 | 80.0000 |
| WB_HIGH_BAND_PCT[1] | 95.0000 | 93.6191 |
| WB_LOW_DESAT | 0.0000 | 0.2464 |
| WB_HIGH_DESAT | 0.0000 | 1.0000 |
| WB_REGION_MIN_FRAC | 0.0001 | 0.0000 |
| WB_HIGH_PRIOR[0] | 1.7590 | 1.8480 |
| WB_HIGH_PRIOR[1] | 1.3656 | 1.6045 |
| WB_HIGH_PRIOR[2] | 1.0641 | 1.2000 |
| WB_LOW_PRIOR[0] | 1.0891 | 0.9676 |
| WB_LOW_PRIOR[1] | 0.7461 | 0.9234 |
| WB_LOW_PRIOR[2] | 0.6941 | 0.5000 |
| PRINT_HI_PCT | 99.9155 | 99.5068 |
| PRINT_HI_CEIL | 0.9693 | 0.8000 |
| PRINT_CLIP_BUDGET | 0.0030 | 0.0000 |
| PRINT_TUNE_ITERS | 11.0000 | 6.0000 |
| PRINT_GAMMA | 4.5312 | 4.0000 |

## Aggregate (all rolls)

- median_total: 0.0385
- median_luma: 0.0362
- median_color: 0.0190
- median_hi999: 0.0473
- max_clip: 0.0000
- frames_over_clip_budget: 0
- clip_budget: 0.0100
- n_frames: 76

## Per-roll

### 2506-1  (prep 27.59s)
```
{
  "n": 38,
  "median_total": 0.03722653587326394,
  "worst_frames": [
    {
      "stem": "DSC_0029",
      "total": 0.1297388218409672
    },
    {
      "stem": "DSC_0008",
      "total": 0.09180843661381317
    },
    {
      "stem": "DSC_0032",
      "total": 0.0813981106394391
    },
    {
      "stem": "DSC_0013",
      "total": 0.06555502581959519
    },
    {
      "stem": "DSC_0038",
      "total": 0.06366204743657614
    }
  ]
}
```
### 2510-11-1  (prep 30.21s)
```
{
  "n": 38,
  "median_total": 0.05419943992034358,
  "worst_frames": [
    {
      "stem": "DSC_0010",
      "total": 0.14052066787049355
    },
    {
      "stem": "DSC_0005",
      "total": 0.0990108798631925
    },
    {
      "stem": "DSC_0007",
      "total": 0.09393507958339392
    },
    {
      "stem": "DSC_0008",
      "total": 0.0859840122423609
    },
    {
      "stem": "DSC_0024",
      "total": 0.0850330793177756
    }
  ]
}
```

## Worst frames (find where to look)

| roll | stem | total | luma | color | hi999 | clip |
|---|---|---|---|---|---|---|
| 2510-11-1 | DSC_0010 | 0.1405 | 0.1616 | 0.0643 | 0.0734 | 0.0000 |
| 2506-1 | DSC_0029 | 0.1297 | 0.1571 | 0.0399 | 0.0319 | 0.0000 |
| 2510-11-1 | DSC_0005 | 0.0990 | 0.1189 | 0.0351 | 0.0729 | 0.0000 |
| 2510-11-1 | DSC_0007 | 0.0939 | 0.1186 | 0.0594 | 0.0330 | 0.0000 |
| 2506-1 | DSC_0008 | 0.0918 | 0.0871 | 0.0194 | 0.0487 | 0.0000 |
| 2510-11-1 | DSC_0008 | 0.0860 | 0.0814 | 0.0685 | 0.0156 | 0.0000 |
| 2510-11-1 | DSC_0024 | 0.0850 | 0.0806 | 0.0083 | 0.0300 | 0.0000 |
| 2506-1 | DSC_0032 | 0.0814 | 0.1001 | 0.0365 | 0.0302 | 0.0000 |
| 2510-11-1 | DSC_0040 | 0.0808 | 0.0862 | 0.0079 | 0.0046 | 0.0000 |
| 2510-11-1 | DSC_0016 | 0.0767 | 0.0744 | 0.0242 | 0.0005 | 0.0000 |
| 2510-11-1 | DSC_0019 | 0.0758 | 0.0682 | 0.0295 | 0.0150 | 0.0000 |
| 2510-11-1 | DSC_0004 | 0.0678 | 0.0784 | 0.0157 | 0.0507 | 0.0000 |
| 2510-11-1 | DSC_0003 | 0.0669 | 0.0751 | 0.0440 | 0.0778 | 0.0000 |
| 2510-11-1 | DSC_0032 | 0.0660 | 0.0493 | 0.0329 | 0.0425 | 0.0000 |
| 2506-1 | DSC_0013 | 0.0656 | 0.0630 | 0.0131 | 0.0464 | 0.0000 |
