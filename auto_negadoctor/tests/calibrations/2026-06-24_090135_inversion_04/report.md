# Calibration session — inversion

- comment: test running inversion on less params after pca run detected 13 non important params
- created: 2026-06-23T22:48:34
- git commit: 9b6ca96
- rolls: 2506-1, 2510-11-1
- metric: histogram_emd_total (render vs GT-render, picture-vs-picture)
- fit method: coordinate_descent
- method params: epsilon=0.0005, max_iters=4, init_step=per-param/auto, step_min=per-param/auto, step_shrink=0.5
- fitted params: PATCH_WIN_FRAC, PATCH_STRIDE_DIV, PRINT_HI_PCT, HIGHLIGHT_BAND_PCT[1], PRINT_GAMMA, WB_HIGH_PRIOR[2], PRINT_HI_CEIL, OFFSET_DEFAULT, WB_HIGH_PRIOR[0], WB_HIGH_DESAT, HIGHLIGHT_BAND_PCT[0], PATCH_UNIFORMITY_MAX, PRINT_TUNE_ITERS, DMAX_DEFAULT, WB_HIGH_PRIOR[1], P_HIGH, PRINT_CLIP_BUDGET, WB_HIGH_BAND_PCT[0], P_LOW, WB_HIGH_BAND_PCT[1]
- trial count: 1033
- wall time: 36779.67s (prep 55.53s)
- objective: 0.0556 (init) -> 0.0373 (final)

## Convergence

- method: coordinate_descent  trials: 1033
- cycles run: 4 (of max 4); converged early: True (epsilon 0.0005)
- per-cycle objective (improvement):
  - cycle 1: 0.0426 (−0.0131)
  - cycle 2: 0.0395 (−0.0031)
  - cycle 3: 0.0376 (−0.0019)
  - cycle 4: 0.0373 (−0.0003)
- best-so-far improved 72 time(s); curve (trial: objective):
  - 1: 0.0556
  - 2: 0.0551
  - 3: 0.0548
  - 7: 0.0543
  - 12: 0.0519
  - 31: 0.0496
  - 35: 0.0490
  - 39: 0.0486
  - 42: 0.0478
  - 45: 0.0459
  - 64: 0.0449
  - 70: 0.0448
  - 81: 0.0448
  - 84: 0.0446
  - 88: 0.0444
  - 96: 0.0443
  - 99: 0.0442
  - 104: 0.0440
  - 108: 0.0438
  - 116: 0.0438
  - 124: 0.0432
  - 131: 0.0432
  - 135: 0.0432
  - 187: 0.0431
  - 193: 0.0431
  - 201: 0.0428
  - 209: 0.0428
  - 216: 0.0426
  - 220: 0.0426
  - 294: 0.0422
  - 296: 0.0411
  - 301: 0.0403
  - 309: 0.0403
  - 331: 0.0401
  - 334: 0.0400
  - 349: 0.0400
  - 361: 0.0397
  - 374: 0.0397
  - 422: 0.0396
  - 425: 0.0396
  - 446: 0.0396
  - 497: 0.0395
  - 498: 0.0395
  - 499: 0.0395
  - 500: 0.0395
  - 501: 0.0395
  - 506: 0.0395
  - 510: 0.0395
  - 514: 0.0395
  - 533: 0.0386
  - 560: 0.0382
  - 564: 0.0381
  - 594: 0.0380
  - 620: 0.0380
  - 631: 0.0379
  - 644: 0.0379
  - 680: 0.0378
  - 702: 0.0377
  - 725: 0.0376
  - 728: 0.0376
  - 732: 0.0376
  - 763: 0.0376
  - 767: 0.0376
  - 844: 0.0376
  - 858: 0.0376
  - 932: 0.0375
  - 952: 0.0374
  - 975: 0.0374
  - 978: 0.0374
  - 1005: 0.0373
  - 1009: 0.0373
  - 1017: 0.0373

## Main contributors (objective drop credited per param)

Free attribution from coordinate descent's accepted moves — which single params drove convergence.

| param | objective drop | % of total | moves |
|---|---|---|---|
| PRINT_HI_PCT | 0.0069 | 37.6% | 10 |
| PATCH_WIN_FRAC | 0.0046 | 25.3% | 5 |
| HIGHLIGHT_BAND_PCT[1] | 0.0019 | 10.1% | 1 |
| PRINT_GAMMA | 0.0015 | 8.2% | 6 |
| PRINT_HI_CEIL | 0.0008 | 4.3% | 5 |
| WB_HIGH_PRIOR[0] | 0.0006 | 3.3% | 4 |
| OFFSET_DEFAULT | 0.0005 | 3.0% | 5 |
| PRINT_TUNE_ITERS | 0.0004 | 2.0% | 4 |
| P_HIGH | 0.0003 | 1.8% | 7 |
| WB_HIGH_PRIOR[1] | 0.0003 | 1.7% | 2 |
| DMAX_DEFAULT | 0.0002 | 1.2% | 5 |
| P_LOW | 0.0002 | 0.9% | 13 |
| WB_HIGH_PRIOR[2] | 0.0001 | 0.7% | 4 |

## Fitted constants (record-only — adopt by hand)

| constant | init | fitted |
|---|---|---|
| PATCH_WIN_FRAC | 0.0494 | 0.0653 |
| PATCH_STRIDE_DIV | 2.0000 | 2.0000 |
| PRINT_HI_PCT | 99.9155 | 99.7657 |
| HIGHLIGHT_BAND_PCT[1] | 96.7500 | 98.5000 |
| PRINT_GAMMA | 4.5312 | 4.5410 |
| WB_HIGH_PRIOR[2] | 1.0641 | 1.0687 |
| PRINT_HI_CEIL | 0.9693 | 0.9418 |
| OFFSET_DEFAULT | 0.0063 | 0.0273 |
| WB_HIGH_PRIOR[0] | 1.7590 | 1.6828 |
| WB_HIGH_DESAT | 0.0000 | 0.0000 |
| HIGHLIGHT_BAND_PCT[0] | 73.7500 | 73.7500 |
| PATCH_UNIFORMITY_MAX | 0.4500 | 0.4500 |
| PRINT_TUNE_ITERS | 11.0000 | 7.0625 |
| DMAX_DEFAULT | 1.7218 | 1.6944 |
| WB_HIGH_PRIOR[1] | 1.3656 | 1.3305 |
| P_HIGH | 99.7227 | 99.8117 |
| PRINT_CLIP_BUDGET | 0.0030 | 0.0030 |
| WB_HIGH_BAND_PCT[0] | 70.0000 | 70.0000 |
| P_LOW | 0.5000 | 3.6641 |
| WB_HIGH_BAND_PCT[1] | 95.0000 | 95.0000 |

## Aggregate (all rolls)

- median_total: 0.0373
- median_luma: 0.0296
- median_color: 0.0233
- median_hi999: 0.0086
- max_clip: 0.0074
- frames_over_clip_budget: 0
- clip_budget: 0.0100
- n_frames: 76

## Per-roll

### 2506-1  (prep 25.3s)
```
{
  "n": 38,
  "median_total": 0.03274050062164688,
  "worst_frames": [
    {
      "stem": "DSC_0029",
      "total": 0.15637971950325838
    },
    {
      "stem": "DSC_0008",
      "total": 0.08908779854371462
    },
    {
      "stem": "DSC_0021",
      "total": 0.06770499570166877
    },
    {
      "stem": "DSC_0013",
      "total": 0.06093169947031938
    },
    {
      "stem": "DSC_0024",
      "total": 0.056232677339594146
    }
  ]
}
```
### 2510-11-1  (prep 30.23s)
```
{
  "n": 38,
  "median_total": 0.06336260321264686,
  "worst_frames": [
    {
      "stem": "DSC_0010",
      "total": 0.1535002256260363
    },
    {
      "stem": "DSC_0015",
      "total": 0.10603980678295279
    },
    {
      "stem": "DSC_0024",
      "total": 0.10007170303204775
    },
    {
      "stem": "DSC_0007",
      "total": 0.09751028527701633
    },
    {
      "stem": "DSC_0005",
      "total": 0.09390847080612665
    }
  ]
}
```

## Worst frames (find where to look)

| roll | stem | total | luma | color | hi999 | clip |
|---|---|---|---|---|---|---|
| 2506-1 | DSC_0029 | 0.1564 | 0.1682 | 0.0194 | 0.0705 | 0.0000 |
| 2510-11-1 | DSC_0010 | 0.1535 | 0.1650 | 0.1046 | 0.0307 | 0.0000 |
| 2510-11-1 | DSC_0015 | 0.1060 | 0.0322 | 0.1094 | 0.0173 | 0.0010 |
| 2510-11-1 | DSC_0024 | 0.1001 | 0.0451 | 0.0875 | 0.0191 | 0.0001 |
| 2510-11-1 | DSC_0007 | 0.0975 | 0.1178 | 0.0579 | 0.0097 | 0.0000 |
| 2510-11-1 | DSC_0005 | 0.0939 | 0.1175 | 0.0543 | 0.0081 | 0.0000 |
| 2510-11-1 | DSC_0017 | 0.0920 | 0.0237 | 0.0982 | 0.0094 | 0.0004 |
| 2506-1 | DSC_0008 | 0.0891 | 0.0750 | 0.0255 | 0.0010 | 0.0000 |
| 2510-11-1 | DSC_0011 | 0.0865 | 0.0494 | 0.0805 | 0.0170 | 0.0010 |
| 2510-11-1 | DSC_0004 | 0.0854 | 0.0958 | 0.0616 | 0.0080 | 0.0000 |
| 2510-11-1 | DSC_0025 | 0.0825 | 0.0249 | 0.0803 | 0.0083 | 0.0000 |
| 2510-11-1 | DSC_0040 | 0.0793 | 0.0887 | 0.0168 | 0.0089 | 0.0007 |
| 2510-11-1 | DSC_0022 | 0.0780 | 0.0612 | 0.0259 | 0.0130 | 0.0000 |
| 2510-11-1 | DSC_0019 | 0.0738 | 0.0531 | 0.0351 | 0.0322 | 0.0000 |
| 2510-11-1 | DSC_0026 | 0.0733 | 0.0241 | 0.0714 | 0.0121 | 0.0000 |
