# Calibration session — inversion

- comment: first run on 4 rolls! starting with 2x downsampling. compare with _02
- created: 2026-06-27T17:30:44
- git commit: 91b1d9e
- rolls: 2506-1, 2510-11-1, 2511-12-1, 2512-2601-1
- metric: histogram_emd_total (render vs GT-render, picture-vs-picture)
- fit method: coordinate_descent
- downsample (calibration-only): 2x
- method params: epsilon=0.0005, max_iters=6, init_step=per-param/auto, step_min=per-param/auto, step_shrink=0.5
- fitted params: a, l, l
- trial count: 1663
- wall time: 80009.35s (prep 112.61s)
- objective: 0.0646 (init) -> 0.0470 (final)

## Convergence

- method: coordinate_descent  trials: 1663
- cycles run: 4 (of max 6); converged early: True (epsilon 0.0005)
- per-cycle objective (improvement):
  - cycle 1: 0.0500 (−0.0145)
  - cycle 2: 0.0477 (−0.0023)
  - cycle 3: 0.0472 (−0.0005)
  - cycle 4: 0.0470 (−0.0001)
- best-so-far improved 90 time(s); curve (trial: objective):
  - 1: 0.0646
  - 2: 0.0628
  - 3: 0.0624
  - 11: 0.0624
  - 15: 0.0624
  - 19: 0.0624
  - 32: 0.0623
  - 37: 0.0617
  - 48: 0.0598
  - 49: 0.0587
  - 53: 0.0574
  - 60: 0.0572
  - 63: 0.0572
  - 69: 0.0565
  - 75: 0.0565
  - 106: 0.0557
  - 131: 0.0554
  - 156: 0.0550
  - 205: 0.0549
  - 206: 0.0547
  - 207: 0.0542
  - 208: 0.0541
  - 213: 0.0541
  - 216: 0.0541
  - 220: 0.0541
  - 223: 0.0541
  - 258: 0.0541
  - 262: 0.0541
  - 266: 0.0541
  - 270: 0.0541
  - 297: 0.0540
  - 301: 0.0539
  - 304: 0.0539
  - 313: 0.0539
  - 327: 0.0539
  - 330: 0.0539
  - 336: 0.0539
  - 346: 0.0539
  - 350: 0.0539
  - 354: 0.0539
  - 357: 0.0539
  - 364: 0.0539
  - 381: 0.0522
  - 385: 0.0517
  - 388: 0.0514
  - 405: 0.0512
  - 420: 0.0511
  - 436: 0.0505
  - 441: 0.0504
  - 445: 0.0500
  - 456: 0.0500
  - 460: 0.0500
  - 474: 0.0500
  - 488: 0.0500
  - 510: 0.0498
  - 638: 0.0497
  - 642: 0.0497
  - 646: 0.0497
  - 650: 0.0497
  - 687: 0.0494
  - 688: 0.0492
  - 691: 0.0492
  - 695: 0.0492
  - 703: 0.0492
  - 745: 0.0492
  - 749: 0.0492
  - 762: 0.0492
  - 776: 0.0492
  - 785: 0.0492
  - 791: 0.0492
  - 823: 0.0480
  - 829: 0.0479
  - 832: 0.0478
  - 850: 0.0477
  - 872: 0.0477
  - 878: 0.0477
  - 882: 0.0477
  - 886: 0.0477
  - 891: 0.0477
  - 958: 0.0476
  - 1067: 0.0476
  - 1071: 0.0476
  - 1114: 0.0476
  - 1189: 0.0476
  - 1195: 0.0476
  - 1234: 0.0472
  - 1462: 0.0472
  - 1501: 0.0471
  - 1506: 0.0470
  - 1601: 0.0470

## Main contributors (objective drop credited per param)

Free attribution from coordinate descent's accepted moves — which single params drove convergence.

| param | objective drop | % of total | moves |
|---|---|---|---|
| DMAX_DEFAULT | 0.0044 | 25.3% | 5 |
| PRINT_HI_PCT | 0.0025 | 14.2% | 3 |
| P_LOW | 0.0022 | 12.7% | 11 |
| PRINT_HI_CEIL | 0.0018 | 10.3% | 4 |
| PRINT_GAMMA | 0.0011 | 6.1% | 4 |
| WB_LOW_BAND_PCT[0] | 0.0010 | 5.7% | 15 |
| PATCH_WIN_FRAC | 0.0008 | 4.8% | 3 |
| HIGHLIGHT_BAND_PCT[1] | 0.0008 | 4.4% | 1 |
| OFFSET_DEFAULT | 0.0007 | 3.9% | 2 |
| WB_LOW_DESAT | 0.0007 | 3.9% | 12 |
| PATCH_UNIFORMITY_MAX | 0.0004 | 2.4% | 1 |
| PATCH_CHROMA_MAX | 0.0003 | 1.7% | 1 |
| WB_HIGH_PRIOR[0] | 0.0003 | 1.6% | 3 |
| PRINT_CLIP_BUDGET | 0.0002 | 1.0% | 1 |
| PRINT_TUNE_ITERS | 0.0001 | 0.8% | 2 |
| P_HIGH | 0.0001 | 0.7% | 2 |
| HIGHLIGHT_BAND_PCT[0] | 0.0001 | 0.4% | 1 |
| WB_HIGH_PRIOR[1] | 0.0000 | 0.0% | 3 |
| WB_LOW_PRIOR[0] | 0.0000 | 0.0% | 4 |
| WB_HIGH_PRIOR[2] | 0.0000 | 0.0% | 3 |
| WB_LOW_PRIOR[1] | 0.0000 | 0.0% | 7 |
| WB_LOW_PRIOR[2] | 0.0000 | 0.0% | 1 |

## Fitted constants (record-only — adopt by hand)

| constant | init | fitted |
|---|---|---|
| P_LOW | 0.5000 | 1.0449 |
| P_HIGH | 99.7227 | 99.7227 |
| OFFSET_DEFAULT | 0.0063 | 0.0238 |
| DMAX_DEFAULT | 1.7218 | 1.9210 |
| PATCH_WIN_FRAC | 0.0494 | 0.0437 |
| PATCH_STRIDE_DIV | 2.0000 | 2.0000 |
| HIGHLIGHT_BAND_PCT[0] | 73.7500 | 74.6875 |
| HIGHLIGHT_BAND_PCT[1] | 96.7500 | 98.5000 |
| SHADOW_MIN_LUMA | 0.0050 | 0.0050 |
| PATCH_CHROMA_MAX | 0.3500 | 0.4125 |
| PATCH_CHROMA_FLOOR | 0.0200 | 0.0200 |
| PATCH_UNIFORMITY_MAX | 0.4500 | 0.5125 |
| PATCH_LUMA_FLOOR | 0.0200 | 0.0200 |
| MIN_PATCH_DENSITY | 0.0500 | 0.0500 |
| HIGHLIGHT_CLIP_FRAC_MAX | 0.0200 | 0.0200 |
| WB_LOW_BAND_PCT[0] | 0.0000 | 6.1719 |
| WB_LOW_BAND_PCT[1] | 20.0000 | 20.0000 |
| WB_HIGH_BAND_PCT[0] | 70.0000 | 70.0000 |
| WB_HIGH_BAND_PCT[1] | 95.0000 | 95.0000 |
| WB_LOW_DESAT | 0.0000 | 0.3750 |
| WB_HIGH_DESAT | 0.0000 | 0.0000 |
| WB_REGION_MIN_FRAC | 0.0001 | 0.0001 |
| WB_HIGH_PRIOR[0] | 1.7590 | 1.7766 |
| WB_HIGH_PRIOR[1] | 1.3656 | 1.3695 |
| WB_HIGH_PRIOR[2] | 1.0641 | 1.0625 |
| WB_LOW_PRIOR[0] | 1.0891 | 1.0328 |
| WB_LOW_PRIOR[1] | 0.7461 | 0.7480 |
| WB_LOW_PRIOR[2] | 0.6941 | 0.6961 |
| PRINT_HI_PCT | 99.9155 | 99.9823 |
| PRINT_HI_CEIL | 0.9693 | 0.9641 |
| PRINT_CLIP_BUDGET | 0.0030 | 0.0055 |
| PRINT_TUNE_ITERS | 11.0000 | 11.0000 |
| PRINT_GAMMA | 4.5312 | 4.4531 |

## Aggregate (all rolls)

- median_total: 0.0470
- median_luma: 0.0454
- median_color: 0.0198
- median_hi999: 0.0123
- max_clip: 0.0044
- frames_over_clip_budget: 0
- clip_budget: 0.0100
- n_frames: 150

## Per-roll

### 2506-1  (prep 27.35s)
```
{
  "n": 38,
  "median_total": 0.03319076930517491,
  "worst_frames": [
    {
      "stem": "DSC_0032",
      "total": 0.1491695099903855
    },
    {
      "stem": "DSC_0029",
      "total": 0.13752358887232985
    },
    {
      "stem": "DSC_0008",
      "total": 0.06592546556129658
    },
    {
      "stem": "DSC_0021",
      "total": 0.06159201931371524
    },
    {
      "stem": "DSC_0015",
      "total": 0.049609441847951
    }
  ]
}
```
### 2510-11-1  (prep 28.34s)
```
{
  "n": 38,
  "median_total": 0.04669285425336697,
  "worst_frames": [
    {
      "stem": "DSC_0010",
      "total": 0.1603157368025653
    },
    {
      "stem": "DSC_0005",
      "total": 0.12817180464945224
    },
    {
      "stem": "DSC_0007",
      "total": 0.11966188585497618
    },
    {
      "stem": "DSC_0008",
      "total": 0.10430066317408358
    },
    {
      "stem": "DSC_0004",
      "total": 0.08717118501748922
    }
  ]
}
```
### 2511-12-1  (prep 28.66s)
```
{
  "n": 37,
  "median_total": 0.0630365016189646,
  "worst_frames": [
    {
      "stem": "DSC_0036",
      "total": 0.15225188912196044
    },
    {
      "stem": "DSC_0030",
      "total": 0.14328482539037476
    },
    {
      "stem": "DSC_0015",
      "total": 0.13246142300576055
    },
    {
      "stem": "DSC_0026",
      "total": 0.13072393327645085
    },
    {
      "stem": "DSC_0034",
      "total": 0.13068031459780616
    }
  ]
}
```
### 2512-2601-1  (prep 28.26s)
```
{
  "n": 37,
  "median_total": 0.10744189311280868,
  "worst_frames": [
    {
      "stem": "DSC_0029",
      "total": 0.22084252577218347
    },
    {
      "stem": "DSC_0008",
      "total": 0.207388328746964
    },
    {
      "stem": "DSC_0013",
      "total": 0.1775641785985279
    },
    {
      "stem": "DSC_0014",
      "total": 0.1719633242254278
    },
    {
      "stem": "DSC_0034",
      "total": 0.16122580708589426
    }
  ]
}
```

## Worst frames (find where to look)

| roll | stem | total | luma | color | hi999 | clip |
|---|---|---|---|---|---|---|
| 2512-2601-1 | DSC_0029 | 0.2208 | 0.2283 | 0.0137 | 0.0279 | 0.0000 |
| 2512-2601-1 | DSC_0008 | 0.2074 | 0.2144 | 0.0209 | 0.0158 | 0.0000 |
| 2512-2601-1 | DSC_0013 | 0.1776 | 0.1843 | 0.0298 | 0.0209 | 0.0000 |
| 2512-2601-1 | DSC_0014 | 0.1720 | 0.1792 | 0.0303 | 0.0155 | 0.0000 |
| 2512-2601-1 | DSC_0034 | 0.1612 | 0.1646 | 0.0055 | 0.0116 | 0.0000 |
| 2512-2601-1 | DSC_0003 | 0.1610 | 0.1707 | 0.0427 | 0.0167 | 0.0000 |
| 2510-11-1 | DSC_0010 | 0.1603 | 0.1769 | 0.1076 | 0.0267 | 0.0030 |
| 2512-2601-1 | DSC_0005 | 0.1549 | 0.1665 | 0.0292 | 0.0199 | 0.0000 |
| 2511-12-1 | DSC_0036 | 0.1523 | 0.1516 | 0.0017 | 0.0157 | 0.0000 |
| 2506-1 | DSC_0032 | 0.1492 | 0.1649 | 0.0246 | 0.0505 | 0.0000 |
| 2512-2601-1 | DSC_0037 | 0.1439 | 0.1458 | 0.0142 | 0.0114 | 0.0000 |
| 2511-12-1 | DSC_0030 | 0.1433 | 0.1465 | 0.0145 | 0.0199 | 0.0000 |
| 2506-1 | DSC_0029 | 0.1375 | 0.1463 | 0.0175 | 0.0638 | 0.0000 |
| 2512-2601-1 | DSC_0023 | 0.1355 | 0.1551 | 0.0422 | 0.0137 | 0.0000 |
| 2512-2601-1 | DSC_0036 | 0.1349 | 0.1329 | 0.0115 | 0.0138 | 0.0000 |
