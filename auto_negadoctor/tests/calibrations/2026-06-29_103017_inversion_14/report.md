# NOTE: DEPRECATED BY _12

# Calibration session — inversion

- comment: rerun calib of 1 roll after fixing GT defaults usage and also fixing GT. compare with _01
- created: 2026-06-28T21:36:28
- git commit: 91b1d9e
- rolls: 2506-1
- metric: histogram_emd_total (render vs GT-render, picture-vs-picture)
- fit method: coordinate_descent
- method params: epsilon=0.0005, max_iters=6, init_step=per-param/auto, step_min=per-param/auto, step_shrink=0.5
- fitted params: a, l, l
- trial count: 1196
- wall time: 46428.43s (prep 115.18s)
- objective: 0.0402 (init) -> 0.0281 (final)

## Convergence

- method: coordinate_descent  trials: 1196
- cycles run: 3 (of max 6); converged early: True (epsilon 0.0005)
- per-cycle objective (improvement):
  - cycle 1: 0.0297 (−0.0105)
  - cycle 2: 0.0284 (−0.0013)
  - cycle 3: 0.0281 (−0.0003)
- best-so-far improved 57 time(s); curve (trial: objective):
  - 1: 0.0402
  - 9: 0.0379
  - 14: 0.0378
  - 19: 0.0377
  - 24: 0.0364
  - 28: 0.0327
  - 41: 0.0321
  - 45: 0.0316
  - 48: 0.0310
  - 63: 0.0310
  - 116: 0.0303
  - 189: 0.0302
  - 235: 0.0302
  - 240: 0.0302
  - 269: 0.0298
  - 273: 0.0298
  - 287: 0.0298
  - 306: 0.0298
  - 309: 0.0298
  - 314: 0.0298
  - 345: 0.0298
  - 397: 0.0297
  - 400: 0.0297
  - 414: 0.0294
  - 447: 0.0293
  - 475: 0.0291
  - 524: 0.0290
  - 586: 0.0289
  - 591: 0.0289
  - 594: 0.0289
  - 628: 0.0288
  - 663: 0.0288
  - 677: 0.0288
  - 689: 0.0288
  - 711: 0.0288
  - 717: 0.0288
  - 720: 0.0288
  - 732: 0.0288
  - 745: 0.0285
  - 760: 0.0284
  - 809: 0.0284
  - 814: 0.0284
  - 817: 0.0284
  - 830: 0.0283
  - 844: 0.0283
  - 996: 0.0283
  - 1066: 0.0283
  - 1070: 0.0282
  - 1086: 0.0282
  - 1096: 0.0282
  - 1099: 0.0282
  - 1103: 0.0282
  - 1112: 0.0282
  - 1120: 0.0282
  - 1130: 0.0282
  - 1145: 0.0282
  - 1158: 0.0281

## Main contributors (objective drop credited per param)

Free attribution from coordinate descent's accepted moves — which single params drove convergence.

| param | objective drop | % of total | moves |
|---|---|---|---|
| OFFSET_DEFAULT | 0.0052 | 43.0% | 3 |
| P_HIGH | 0.0028 | 23.6% | 7 |
| DMAX_DEFAULT | 0.0017 | 14.3% | 5 |
| PATCH_CHROMA_MAX | 0.0006 | 5.2% | 1 |
| PRINT_HI_PCT | 0.0004 | 3.3% | 3 |
| WB_HIGH_PRIOR[0] | 0.0004 | 2.9% | 3 |
| WB_LOW_BAND_PCT[0] | 0.0003 | 2.6% | 4 |
| HIGHLIGHT_BAND_PCT[0] | 0.0002 | 1.9% | 1 |
| PRINT_HI_CEIL | 0.0002 | 1.4% | 2 |
| PATCH_CHROMA_FLOOR | 0.0001 | 0.7% | 1 |
| PRINT_GAMMA | 0.0001 | 0.5% | 2 |
| PATCH_WIN_FRAC | 0.0000 | 0.2% | 1 |
| WB_LOW_DESAT | 0.0000 | 0.2% | 3 |
| WB_HIGH_PRIOR[2] | 0.0000 | 0.1% | 2 |
| WB_LOW_PRIOR[0] | 0.0000 | 0.0% | 6 |
| WB_HIGH_PRIOR[1] | 0.0000 | 0.0% | 4 |
| WB_LOW_BAND_PCT[1] | 0.0000 | 0.0% | 1 |
| WB_LOW_PRIOR[1] | 0.0000 | 0.0% | 4 |
| WB_LOW_PRIOR[2] | 0.0000 | 0.0% | 3 |

## Fitted constants (record-only — adopt by hand)

| constant | init | fitted |
|---|---|---|
| P_LOW | 0.5000 | 0.5000 |
| P_HIGH | 99.7227 | 99.5223 |
| OFFSET_DEFAULT | 0.0063 | 0.0262 |
| DMAX_DEFAULT | 1.7218 | 1.7687 |
| PATCH_WIN_FRAC | 0.0494 | 0.0498 |
| PATCH_STRIDE_DIV | 2.0000 | 2.0000 |
| HIGHLIGHT_BAND_PCT[0] | 73.7500 | 70.0000 |
| HIGHLIGHT_BAND_PCT[1] | 96.7500 | 96.7500 |
| SHADOW_MIN_LUMA | 0.0050 | 0.0050 |
| PATCH_CHROMA_MAX | 0.3500 | 0.4125 |
| PATCH_CHROMA_FLOOR | 0.0200 | 0.0256 |
| PATCH_UNIFORMITY_MAX | 0.4500 | 0.4500 |
| PATCH_LUMA_FLOOR | 0.0200 | 0.0200 |
| MIN_PATCH_DENSITY | 0.0500 | 0.0500 |
| HIGHLIGHT_CLIP_FRAC_MAX | 0.0200 | 0.0200 |
| WB_LOW_BAND_PCT[0] | 0.0000 | 0.2344 |
| WB_LOW_BAND_PCT[1] | 20.0000 | 20.1172 |
| WB_HIGH_BAND_PCT[0] | 70.0000 | 70.0000 |
| WB_HIGH_BAND_PCT[1] | 95.0000 | 95.0000 |
| WB_LOW_DESAT | 0.0000 | 0.0000 |
| WB_HIGH_DESAT | 0.0000 | 0.0000 |
| WB_REGION_MIN_FRAC | 0.0001 | 0.0001 |
| WB_HIGH_PRIOR[0] | 1.7590 | 1.7707 |
| WB_HIGH_PRIOR[1] | 1.3656 | 1.3344 |
| WB_HIGH_PRIOR[2] | 1.0641 | 1.0594 |
| WB_LOW_PRIOR[0] | 1.0891 | 1.1047 |
| WB_LOW_PRIOR[1] | 0.7461 | 0.7363 |
| WB_LOW_PRIOR[2] | 0.6941 | 0.6238 |
| PRINT_HI_PCT | 99.9155 | 99.9580 |
| PRINT_HI_CEIL | 0.9693 | 0.9707 |
| PRINT_CLIP_BUDGET | 0.0030 | 0.0030 |
| PRINT_TUNE_ITERS | 11.0000 | 11.0000 |
| PRINT_GAMMA | 4.5312 | 4.5117 |

## Aggregate (all rolls)

- median_total: 0.0281
- median_luma: 0.0254
- median_color: 0.0143
- median_hi999: 0.0050
- max_clip: 0.0013
- frames_over_clip_budget: 0
- clip_budget: 0.0100
- n_frames: 38

## Per-roll

### 2506-1  (prep 115.18s)
```
{
  "n": 38,
  "median_total": 0.028114871944035794,
  "worst_frames": [
    {
      "stem": "DSC_0032",
      "total": 0.13299117949851547
    },
    {
      "stem": "DSC_0029",
      "total": 0.09844851622183892
    },
    {
      "stem": "DSC_0021",
      "total": 0.08095682330824418
    },
    {
      "stem": "DSC_0008",
      "total": 0.0796797270748398
    },
    {
      "stem": "DSC_0039",
      "total": 0.07591526219593535
    }
  ]
}
```

## Worst frames (find where to look)

| roll | stem | total | luma | color | hi999 | clip |
|---|---|---|---|---|---|---|
| 2506-1 | DSC_0032 | 0.1330 | 0.1503 | 0.0325 | 0.0357 | 0.0000 |
| 2506-1 | DSC_0029 | 0.0984 | 0.0955 | 0.0136 | 0.0292 | 0.0013 |
| 2506-1 | DSC_0021 | 0.0810 | 0.0772 | 0.0099 | 0.0199 | 0.0000 |
| 2506-1 | DSC_0008 | 0.0797 | 0.0641 | 0.0278 | 0.0032 | 0.0000 |
| 2506-1 | DSC_0039 | 0.0759 | 0.0823 | 0.0520 | 0.0045 | 0.0000 |
| 2506-1 | DSC_0030 | 0.0692 | 0.0789 | 0.0257 | 0.0179 | 0.0000 |
| 2506-1 | DSC_0033 | 0.0579 | 0.0602 | 0.0345 | 0.0090 | 0.0000 |
| 2506-1 | DSC_0025 | 0.0568 | 0.0485 | 0.0141 | 0.0261 | 0.0000 |
| 2506-1 | DSC_0013 | 0.0492 | 0.0450 | 0.0078 | 0.0022 | 0.0000 |
| 2506-1 | DSC_0011 | 0.0489 | 0.0327 | 0.0251 | 0.0002 | 0.0000 |
| 2506-1 | DSC_0038 | 0.0478 | 0.0334 | 0.0175 | 0.0083 | 0.0000 |
| 2506-1 | DSC_0003 | 0.0455 | 0.0542 | 0.0319 | 0.0183 | 0.0000 |
| 2506-1 | DSC_0028 | 0.0453 | 0.0374 | 0.0143 | 0.0021 | 0.0000 |
| 2506-1 | DSC_0027 | 0.0402 | 0.0348 | 0.0126 | 0.0103 | 0.0000 |
| 2506-1 | DSC_0026 | 0.0389 | 0.0357 | 0.0103 | 0.0085 | 0.0000 |
