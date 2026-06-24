# Calibration session — inversion

- created: 2026-06-22T01:54:16
- git commit: ad035a1
- rolls: 2506-1
- metric: histogram_emd_total (render vs GT-render, picture-vs-picture)
- fit method: coordinate_descent
- method params: epsilon=0.0005, max_iters=6, init_step=per-param/auto, step_min=per-param/auto, step_shrink=0.5
- fitted params: a, l, l
- trial count: 813
- wall time: 12611.44s (prep 26.41s)
- objective: 0.0375 (init) -> 0.0292 (final)

## Convergence

- method: coordinate_descent  trials: 813
- cycles run: 2 (of max 6); converged early: True (epsilon 0.0005)
- per-cycle objective (improvement):
  - cycle 1: 0.0296 (−0.0079)
  - cycle 2: 0.0292 (−0.0004)
- best-so-far improved 38 time(s); curve (trial: objective):
  - 1: 0.0375
  - 13: 0.0375
  - 16: 0.0374
  - 21: 0.0374
  - 24: 0.0344
  - 28: 0.0318
  - 43: 0.0312
  - 46: 0.0308
  - 49: 0.0307
  - 62: 0.0303
  - 115: 0.0301
  - 194: 0.0300
  - 205: 0.0300
  - 234: 0.0300
  - 240: 0.0300
  - 243: 0.0300
  - 272: 0.0296
  - 285: 0.0296
  - 288: 0.0296
  - 308: 0.0296
  - 312: 0.0296
  - 316: 0.0296
  - 328: 0.0296
  - 332: 0.0296
  - 357: 0.0296
  - 410: 0.0296
  - 414: 0.0296
  - 424: 0.0296
  - 610: 0.0296
  - 613: 0.0296
  - 681: 0.0296
  - 714: 0.0296
  - 718: 0.0296
  - 734: 0.0296
  - 747: 0.0296
  - 754: 0.0293
  - 762: 0.0292
  - 775: 0.0292

## Main contributors (objective drop credited per param)

Free attribution from coordinate descent's accepted moves — which single params drove convergence.

| param | objective drop | % of total | moves |
|---|---|---|---|
| OFFSET_DEFAULT | 0.0056 | 67.5% | 2 |
| DMAX_DEFAULT | 0.0011 | 13.5% | 3 |
| WB_HIGH_PRIOR[0] | 0.0004 | 4.8% | 2 |
| PRINT_HI_PCT | 0.0004 | 4.5% | 3 |
| PATCH_WIN_FRAC | 0.0003 | 4.0% | 1 |
| PATCH_CHROMA_MAX | 0.0002 | 2.8% | 1 |
| WB_LOW_BAND_PCT[1] | 0.0001 | 1.1% | 4 |
| P_HIGH | 0.0001 | 0.9% | 3 |
| PRINT_HI_CEIL | 0.0000 | 0.5% | 1 |
| WB_LOW_DESAT | 0.0000 | 0.1% | 3 |
| WB_HIGH_PRIOR[1] | 0.0000 | 0.0% | 2 |
| P_LOW | 0.0000 | 0.0% | 3 |
| WB_LOW_PRIOR[0] | 0.0000 | 0.0% | 5 |
| WB_LOW_PRIOR[1] | 0.0000 | 0.0% | 3 |
| WB_LOW_PRIOR[2] | 0.0000 | 0.0% | 1 |

## Fitted constants (record-only — adopt by hand)

| constant | init | fitted |
|---|---|---|
| P_LOW | 0.5000 | 0.7637 |
| P_HIGH | 99.7227 | 99.7004 |
| OFFSET_DEFAULT | 0.0063 | 0.0250 |
| DMAX_DEFAULT | 1.7218 | 1.7765 |
| PATCH_WIN_FRAC | 0.0494 | 0.0498 |
| PATCH_STRIDE_DIV | 2.0000 | 2.0000 |
| HIGHLIGHT_BAND_PCT[0] | 73.7500 | 73.7500 |
| HIGHLIGHT_BAND_PCT[1] | 96.7500 | 96.7500 |
| SHADOW_MIN_LUMA | 0.0050 | 0.0050 |
| PATCH_CHROMA_MAX | 0.3500 | 0.4125 |
| PATCH_CHROMA_FLOOR | 0.0200 | 0.0200 |
| PATCH_UNIFORMITY_MAX | 0.4500 | 0.4500 |
| PATCH_LUMA_FLOOR | 0.0200 | 0.0200 |
| MIN_PATCH_DENSITY | 0.0500 | 0.0500 |
| HIGHLIGHT_CLIP_FRAC_MAX | 0.0200 | 0.0200 |
| WB_LOW_BAND_PCT[0] | 0.0000 | 0.0000 |
| WB_LOW_BAND_PCT[1] | 20.0000 | 23.7500 |
| WB_HIGH_BAND_PCT[0] | 70.0000 | 70.0000 |
| WB_HIGH_BAND_PCT[1] | 95.0000 | 95.0000 |
| WB_LOW_DESAT | 0.0000 | 0.0273 |
| WB_HIGH_DESAT | 0.0000 | 0.0000 |
| WB_REGION_MIN_FRAC | 0.0001 | 0.0001 |
| WB_HIGH_PRIOR[0] | 1.7590 | 1.7648 |
| WB_HIGH_PRIOR[1] | 1.3656 | 1.3773 |
| WB_HIGH_PRIOR[2] | 1.0641 | 1.0641 |
| WB_LOW_PRIOR[0] | 1.0891 | 1.0703 |
| WB_LOW_PRIOR[1] | 0.7461 | 0.7324 |
| WB_LOW_PRIOR[2] | 0.6941 | 0.6961 |
| PRINT_HI_PCT | 99.9155 | 99.9503 |
| PRINT_HI_CEIL | 0.9693 | 0.9700 |
| PRINT_CLIP_BUDGET | 0.0030 | 0.0030 |
| PRINT_TUNE_ITERS | 11.0000 | 11.0000 |
| PRINT_GAMMA | 4.5312 | 4.5312 |

## Aggregate (all rolls)

- median_total: 0.0292
- median_luma: 0.0281
- median_color: 0.0139
- median_hi999: 0.0053
- max_clip: 0.0001
- frames_over_clip_budget: 0
- clip_budget: 0.0100
- n_frames: 38

## Per-roll

### 2506-1  (prep 26.41s)
```
{
  "n": 38,
  "median_total": 0.029196558951954797,
  "worst_frames": [
    {
      "stem": "DSC_0029",
      "total": 0.15626124158242052
    },
    {
      "stem": "DSC_0032",
      "total": 0.1103171708023692
    },
    {
      "stem": "DSC_0008",
      "total": 0.09027274439296278
    },
    {
      "stem": "DSC_0021",
      "total": 0.08144680455987849
    },
    {
      "stem": "DSC_0013",
      "total": 0.060464488383060976
    }
  ]
}
```

## Worst frames (find where to look)

| roll | stem | total | luma | color | hi999 | clip |
|---|---|---|---|---|---|---|
| 2506-1 | DSC_0029 | 0.1563 | 0.1696 | 0.0190 | 0.0720 | 0.0001 |
| 2506-1 | DSC_0032 | 0.1103 | 0.1276 | 0.0318 | 0.0284 | 0.0000 |
| 2506-1 | DSC_0008 | 0.0903 | 0.0780 | 0.0229 | 0.0053 | 0.0000 |
| 2506-1 | DSC_0021 | 0.0814 | 0.0774 | 0.0098 | 0.0202 | 0.0000 |
| 2506-1 | DSC_0013 | 0.0605 | 0.0589 | 0.0076 | 0.0054 | 0.0000 |
| 2506-1 | DSC_0025 | 0.0587 | 0.0524 | 0.0116 | 0.0269 | 0.0000 |
| 2506-1 | DSC_0024 | 0.0581 | 0.0461 | 0.0221 | 0.0086 | 0.0000 |
| 2506-1 | DSC_0033 | 0.0579 | 0.0599 | 0.0341 | 0.0088 | 0.0000 |
| 2506-1 | DSC_0011 | 0.0498 | 0.0393 | 0.0184 | 0.0013 | 0.0000 |
| 2506-1 | DSC_0003 | 0.0462 | 0.0547 | 0.0326 | 0.0176 | 0.0000 |
| 2506-1 | DSC_0038 | 0.0454 | 0.0335 | 0.0152 | 0.0083 | 0.0000 |
| 2506-1 | DSC_0039 | 0.0446 | 0.0306 | 0.0208 | 0.0010 | 0.0000 |
| 2506-1 | DSC_0028 | 0.0441 | 0.0374 | 0.0169 | 0.0019 | 0.0000 |
| 2506-1 | DSC_0037 | 0.0402 | 0.0315 | 0.0127 | 0.0102 | 0.0000 |
| 2506-1 | DSC_0027 | 0.0397 | 0.0353 | 0.0100 | 0.0107 | 0.0000 |
