# Calibration session — inversion

- created: 2026-06-22T12:21:37
- git commit: ad035a1
- rolls: 2506-1, 2510-11-1
- metric: histogram_emd_total (render vs GT-render, picture-vs-picture)
- fit method: coordinate_descent
- method params: epsilon=0.0005, max_iters=6, init_step=per-param/auto, step_min=per-param/auto, step_shrink=0.5
- fitted params: a, l, l
- trial count: 1250
- wall time: 61614.85s (prep 100.9s)
- objective: 0.0556 (init) -> 0.0375 (final)

## Convergence

- method: coordinate_descent  trials: 1250
- cycles run: 3 (of max 6); converged early: True (epsilon 0.0005)
- per-cycle objective (improvement):
  - cycle 1: 0.0401 (−0.0156)
  - cycle 2: 0.0376 (−0.0025)
  - cycle 3: 0.0375 (−0.0001)
- best-so-far improved 73 time(s); curve (trial: objective):
  - 1: 0.0556
  - 2: 0.0552
  - 3: 0.0551
  - 4: 0.0529
  - 12: 0.0529
  - 16: 0.0529
  - 20: 0.0529
  - 24: 0.0516
  - 34: 0.0516
  - 37: 0.0516
  - 40: 0.0506
  - 44: 0.0493
  - 49: 0.0487
  - 52: 0.0487
  - 56: 0.0487
  - 59: 0.0445
  - 60: 0.0440
  - 64: 0.0438
  - 67: 0.0436
  - 70: 0.0436
  - 75: 0.0436
  - 105: 0.0427
  - 116: 0.0418
  - 165: 0.0410
  - 214: 0.0410
  - 218: 0.0409
  - 223: 0.0409
  - 229: 0.0409
  - 288: 0.0406
  - 292: 0.0404
  - 299: 0.0404
  - 329: 0.0404
  - 338: 0.0404
  - 349: 0.0404
  - 353: 0.0404
  - 361: 0.0404
  - 366: 0.0404
  - 370: 0.0404
  - 383: 0.0404
  - 433: 0.0401
  - 439: 0.0401
  - 442: 0.0401
  - 450: 0.0401
  - 457: 0.0397
  - 465: 0.0397
  - 473: 0.0395
  - 479: 0.0393
  - 487: 0.0393
  - 490: 0.0390
  - 494: 0.0387
  - 585: 0.0386
  - 634: 0.0386
  - 635: 0.0386
  - 636: 0.0386
  - 642: 0.0386
  - 645: 0.0386
  - 651: 0.0386
  - 659: 0.0386
  - 732: 0.0386
  - 749: 0.0385
  - 767: 0.0383
  - 805: 0.0379
  - 811: 0.0379
  - 845: 0.0378
  - 848: 0.0376
  - 858: 0.0376
  - 864: 0.0376
  - 875: 0.0376
  - 879: 0.0376
  - 1049: 0.0376
  - 1160: 0.0376
  - 1174: 0.0376
  - 1212: 0.0375

## Main contributors (objective drop credited per param)

Free attribution from coordinate descent's accepted moves — which single params drove convergence.

| param | objective drop | % of total | moves |
|---|---|---|---|
| DMAX_DEFAULT | 0.0057 | 31.6% | 9 |
| OFFSET_DEFAULT | 0.0033 | 18.3% | 7 |
| P_LOW | 0.0028 | 15.3% | 11 |
| P_HIGH | 0.0017 | 9.2% | 7 |
| PATCH_UNIFORMITY_MAX | 0.0009 | 5.1% | 2 |
| HIGHLIGHT_BAND_PCT[1] | 0.0009 | 5.0% | 1 |
| HIGHLIGHT_BAND_PCT[0] | 0.0009 | 4.9% | 1 |
| PRINT_GAMMA | 0.0006 | 3.4% | 3 |
| WB_HIGH_PRIOR[0] | 0.0005 | 2.8% | 3 |
| PRINT_HI_CEIL | 0.0003 | 1.9% | 3 |
| WB_LOW_PRIOR[1] | 0.0002 | 1.1% | 4 |
| WB_LOW_BAND_PCT[0] | 0.0001 | 0.7% | 11 |
| WB_LOW_PRIOR[0] | 0.0001 | 0.5% | 4 |
| PRINT_HI_PCT | 0.0000 | 0.2% | 1 |
| WB_HIGH_PRIOR[1] | 0.0000 | 0.0% | 1 |
| WB_LOW_PRIOR[2] | 0.0000 | 0.0% | 3 |
| WB_LOW_BAND_PCT[1] | 0.0000 | 0.0% | 1 |

## Fitted constants (record-only — adopt by hand)

| constant | init | fitted |
|---|---|---|
| P_LOW | 0.5000 | 1.7305 |
| P_HIGH | 99.7227 | 99.5445 |
| OFFSET_DEFAULT | 0.0063 | 0.0191 |
| DMAX_DEFAULT | 1.7218 | 1.9132 |
| PATCH_WIN_FRAC | 0.0494 | 0.0494 |
| PATCH_STRIDE_DIV | 2.0000 | 2.0000 |
| HIGHLIGHT_BAND_PCT[0] | 73.7500 | 71.8750 |
| HIGHLIGHT_BAND_PCT[1] | 96.7500 | 98.5000 |
| SHADOW_MIN_LUMA | 0.0050 | 0.0050 |
| PATCH_CHROMA_MAX | 0.3500 | 0.3500 |
| PATCH_CHROMA_FLOOR | 0.0200 | 0.0200 |
| PATCH_UNIFORMITY_MAX | 0.4500 | 0.5750 |
| PATCH_LUMA_FLOOR | 0.0200 | 0.0200 |
| MIN_PATCH_DENSITY | 0.0500 | 0.0500 |
| HIGHLIGHT_CLIP_FRAC_MAX | 0.0200 | 0.0200 |
| WB_LOW_BAND_PCT[0] | 0.0000 | 8.4375 |
| WB_LOW_BAND_PCT[1] | 20.0000 | 20.1172 |
| WB_HIGH_BAND_PCT[0] | 70.0000 | 70.0000 |
| WB_HIGH_BAND_PCT[1] | 95.0000 | 95.0000 |
| WB_LOW_DESAT | 0.0000 | 0.0000 |
| WB_HIGH_DESAT | 0.0000 | 0.0000 |
| WB_REGION_MIN_FRAC | 0.0001 | 0.0001 |
| WB_HIGH_PRIOR[0] | 1.7590 | 1.8117 |
| WB_HIGH_PRIOR[1] | 1.3656 | 1.3695 |
| WB_HIGH_PRIOR[2] | 1.0641 | 1.0641 |
| WB_LOW_PRIOR[0] | 1.0891 | 1.0922 |
| WB_LOW_PRIOR[1] | 0.7461 | 0.7383 |
| WB_LOW_PRIOR[2] | 0.6941 | 0.6805 |
| PRINT_HI_PCT | 99.9155 | 99.9193 |
| PRINT_HI_CEIL | 0.9693 | 0.9648 |
| PRINT_CLIP_BUDGET | 0.0030 | 0.0030 |
| PRINT_TUNE_ITERS | 11.0000 | 11.0000 |
| PRINT_GAMMA | 4.5312 | 4.5020 |

## Aggregate (all rolls)

- median_total: 0.0375
- median_luma: 0.0330
- median_color: 0.0183
- median_hi999: 0.0073
- max_clip: 0.0027
- frames_over_clip_budget: 0
- clip_budget: 0.0100
- n_frames: 76

## Per-roll

### 2506-1  (prep 35.49s)
```
{
  "n": 38,
  "median_total": 0.028604664651428822,
  "worst_frames": [
    {
      "stem": "DSC_0029",
      "total": 0.1518142173710808
    },
    {
      "stem": "DSC_0032",
      "total": 0.07884077672964954
    },
    {
      "stem": "DSC_0008",
      "total": 0.07546023877558163
    },
    {
      "stem": "DSC_0021",
      "total": 0.07430792164365084
    },
    {
      "stem": "DSC_0013",
      "total": 0.0654916559964448
    }
  ]
}
```
### 2510-11-1  (prep 65.41s)
```
{
  "n": 38,
  "median_total": 0.05284613529632532,
  "worst_frames": [
    {
      "stem": "DSC_0010",
      "total": 0.16652297510856198
    },
    {
      "stem": "DSC_0005",
      "total": 0.1277660084244082
    },
    {
      "stem": "DSC_0007",
      "total": 0.11165091945649176
    },
    {
      "stem": "DSC_0008",
      "total": 0.0879574838355437
    },
    {
      "stem": "DSC_0019",
      "total": 0.08527032380834133
    }
  ]
}
```

## Worst frames (find where to look)

| roll | stem | total | luma | color | hi999 | clip |
|---|---|---|---|---|---|---|
| 2510-11-1 | DSC_0010 | 0.1665 | 0.1910 | 0.1090 | 0.0426 | 0.0001 |
| 2506-1 | DSC_0029 | 0.1518 | 0.1485 | 0.0132 | 0.0698 | 0.0018 |
| 2510-11-1 | DSC_0005 | 0.1278 | 0.1575 | 0.0692 | 0.0189 | 0.0000 |
| 2510-11-1 | DSC_0007 | 0.1117 | 0.1320 | 0.0491 | 0.0093 | 0.0000 |
| 2510-11-1 | DSC_0008 | 0.0880 | 0.1177 | 0.0546 | 0.0105 | 0.0000 |
| 2510-11-1 | DSC_0019 | 0.0853 | 0.0766 | 0.0256 | 0.0388 | 0.0000 |
| 2510-11-1 | DSC_0004 | 0.0837 | 0.0918 | 0.0136 | 0.0095 | 0.0000 |
| 2510-11-1 | DSC_0025 | 0.0821 | 0.0210 | 0.0834 | 0.0060 | 0.0007 |
| 2510-11-1 | DSC_0020 | 0.0793 | 0.0947 | 0.0366 | 0.0136 | 0.0000 |
| 2506-1 | DSC_0032 | 0.0788 | 0.0869 | 0.0162 | 0.0132 | 0.0000 |
| 2506-1 | DSC_0008 | 0.0755 | 0.0670 | 0.0161 | 0.0056 | 0.0000 |
| 2510-11-1 | DSC_0033 | 0.0745 | 0.0684 | 0.0517 | 0.0177 | 0.0000 |
| 2506-1 | DSC_0021 | 0.0743 | 0.0722 | 0.0183 | 0.0207 | 0.0000 |
| 2510-11-1 | DSC_0035 | 0.0699 | 0.0674 | 0.0448 | 0.0613 | 0.0000 |
| 2510-11-1 | DSC_0016 | 0.0687 | 0.0594 | 0.0181 | 0.0398 | 0.0000 |
