# Calibration session — inversion

- comment: repeat after changing GT and making GT depend only on GT params, not defaults, and also extracted out of loop for speed, compare with _01.
NOTE: IS NEWER AND MORE CORRECT THAN _14, WHICH WAS STARTED EARLIER BUT FINISHED LATER BECAUSE IT CALCULATED GT ON EVERY TRY
- created: 2026-06-29T00:15:00
- git commit: 91b1d9e
- rolls: 2506-1
- metric: histogram_emd_total (render vs GT-render, picture-vs-picture)
- fit method: coordinate_descent
- method params: epsilon=0.0005, max_iters=6, init_step=per-param/auto, step_min=per-param/auto, step_shrink=0.5
- fitted params: a, l, l
- trial count: 821
- wall time: 30747.65s (prep 68.56s)
- objective: 0.0470 (init) -> 0.0239 (final)

## Convergence

- method: coordinate_descent  trials: 821
- cycles run: 2 (of max 6); converged early: True (epsilon 0.0005)
- per-cycle objective (improvement):
  - cycle 1: 0.0244 (−0.0226)
  - cycle 2: 0.0239 (−0.0005)
- best-so-far improved 46 time(s); curve (trial: objective):
  - 1: 0.0470
  - 9: 0.0437
  - 11: 0.0420
  - 13: 0.0404
  - 23: 0.0404
  - 28: 0.0372
  - 32: 0.0349
  - 42: 0.0349
  - 45: 0.0337
  - 46: 0.0298
  - 47: 0.0292
  - 51: 0.0284
  - 56: 0.0283
  - 62: 0.0283
  - 66: 0.0275
  - 68: 0.0263
  - 73: 0.0251
  - 78: 0.0251
  - 107: 0.0247
  - 240: 0.0246
  - 244: 0.0245
  - 247: 0.0244
  - 253: 0.0244
  - 256: 0.0244
  - 288: 0.0244
  - 302: 0.0244
  - 315: 0.0244
  - 324: 0.0244
  - 335: 0.0244
  - 418: 0.0244
  - 422: 0.0244
  - 427: 0.0244
  - 430: 0.0243
  - 438: 0.0242
  - 601: 0.0242
  - 606: 0.0242
  - 612: 0.0242
  - 615: 0.0242
  - 648: 0.0241
  - 653: 0.0240
  - 659: 0.0240
  - 662: 0.0240
  - 736: 0.0240
  - 742: 0.0240
  - 745: 0.0240
  - 769: 0.0239

## Main contributors (objective drop credited per param)

Free attribution from coordinate descent's accepted moves — which single params drove convergence.

| param | objective drop | % of total | moves |
|---|---|---|---|
| P_HIGH | 0.0067 | 29.0% | 6 |
| DMAX_DEFAULT | 0.0066 | 28.6% | 6 |
| OFFSET_DEFAULT | 0.0056 | 24.2% | 3 |
| PATCH_WIN_FRAC | 0.0032 | 13.7% | 4 |
| WB_LOW_DESAT | 0.0005 | 2.3% | 9 |
| HIGHLIGHT_BAND_PCT[1] | 0.0004 | 1.8% | 1 |
| PRINT_HI_PCT | 0.0001 | 0.3% | 1 |
| WB_LOW_BAND_PCT[0] | 0.0000 | 0.0% | 4 |
| P_LOW | 0.0000 | 0.0% | 3 |
| WB_HIGH_PRIOR[0] | 0.0000 | 0.0% | 1 |
| WB_HIGH_PRIOR[2] | 0.0000 | 0.0% | 1 |
| WB_LOW_PRIOR[0] | 0.0000 | 0.0% | 1 |
| WB_HIGH_PRIOR[1] | 0.0000 | 0.0% | 1 |
| WB_LOW_PRIOR[1] | 0.0000 | 0.0% | 4 |

## Fitted constants (record-only — adopt by hand)

| constant | init | fitted |
|---|---|---|
| P_LOW | 0.5000 | 0.5879 |
| P_HIGH | 99.7227 | 99.2031 |
| OFFSET_DEFAULT | 0.0063 | 0.0238 |
| DMAX_DEFAULT | 1.7218 | 2.0460 |
| PATCH_WIN_FRAC | 0.0494 | 0.0367 |
| PATCH_STRIDE_DIV | 2.0000 | 2.0000 |
| HIGHLIGHT_BAND_PCT[0] | 73.7500 | 73.7500 |
| HIGHLIGHT_BAND_PCT[1] | 96.7500 | 98.5000 |
| SHADOW_MIN_LUMA | 0.0050 | 0.0050 |
| PATCH_CHROMA_MAX | 0.3500 | 0.3500 |
| PATCH_CHROMA_FLOOR | 0.0200 | 0.0200 |
| PATCH_UNIFORMITY_MAX | 0.4500 | 0.4500 |
| PATCH_LUMA_FLOOR | 0.0200 | 0.0200 |
| MIN_PATCH_DENSITY | 0.0500 | 0.0500 |
| HIGHLIGHT_CLIP_FRAC_MAX | 0.0200 | 0.0200 |
| WB_LOW_BAND_PCT[0] | 0.0000 | 3.0469 |
| WB_LOW_BAND_PCT[1] | 20.0000 | 20.0000 |
| WB_HIGH_BAND_PCT[0] | 70.0000 | 70.0000 |
| WB_HIGH_BAND_PCT[1] | 95.0000 | 95.0000 |
| WB_LOW_DESAT | 0.0000 | 0.2422 |
| WB_HIGH_DESAT | 0.0000 | 0.0000 |
| WB_REGION_MIN_FRAC | 0.0001 | 0.0001 |
| WB_HIGH_PRIOR[0] | 1.7590 | 1.7531 |
| WB_HIGH_PRIOR[1] | 1.3656 | 1.3617 |
| WB_HIGH_PRIOR[2] | 1.0641 | 1.0656 |
| WB_LOW_PRIOR[0] | 1.0891 | 1.0953 |
| WB_LOW_PRIOR[1] | 0.7461 | 0.7441 |
| WB_LOW_PRIOR[2] | 0.6941 | 0.6941 |
| PRINT_HI_PCT | 99.9155 | 99.9077 |
| PRINT_HI_CEIL | 0.9693 | 0.9693 |
| PRINT_CLIP_BUDGET | 0.0030 | 0.0030 |
| PRINT_TUNE_ITERS | 11.0000 | 11.0000 |
| PRINT_GAMMA | 4.5312 | 4.5312 |

## Aggregate (all rolls)

- median_total: 0.0239
- median_luma: 0.0246
- median_color: 0.0141
- median_hi999: 0.0076
- max_clip: 0.0022
- frames_over_clip_budget: 0
- clip_budget: 0.0100
- n_frames: 38

## Per-roll

### 2506-1  (prep 68.56s)
```
{
  "n": 38,
  "median_total": 0.0239418750722687,
  "worst_frames": [
    {
      "stem": "DSC_0029",
      "total": 0.16809577330578684
    },
    {
      "stem": "DSC_0013",
      "total": 0.11536817044792651
    },
    {
      "stem": "DSC_0032",
      "total": 0.10115731231888371
    },
    {
      "stem": "DSC_0024",
      "total": 0.08919987616356458
    },
    {
      "stem": "DSC_0008",
      "total": 0.06136316238180304
    }
  ]
}
```

## Worst frames (find where to look)

| roll | stem | total | luma | color | hi999 | clip |
|---|---|---|---|---|---|---|
| 2506-1 | DSC_0029 | 0.1681 | 0.1806 | 0.0192 | 0.1111 | 0.0022 |
| 2506-1 | DSC_0013 | 0.1154 | 0.1156 | 0.0196 | 0.0384 | 0.0000 |
| 2506-1 | DSC_0032 | 0.1012 | 0.1150 | 0.0204 | 0.0115 | 0.0000 |
| 2506-1 | DSC_0024 | 0.0892 | 0.0803 | 0.0145 | 0.0260 | 0.0000 |
| 2506-1 | DSC_0008 | 0.0614 | 0.0525 | 0.0193 | 0.0076 | 0.0000 |
| 2506-1 | DSC_0036 | 0.0594 | 0.0544 | 0.0178 | 0.0356 | 0.0000 |
| 2506-1 | DSC_0021 | 0.0558 | 0.0492 | 0.0118 | 0.0213 | 0.0000 |
| 2506-1 | DSC_0018 | 0.0488 | 0.0552 | 0.0198 | 0.0035 | 0.0000 |
| 2506-1 | DSC_0002 | 0.0478 | 0.0554 | 0.0116 | 0.0387 | 0.0000 |
| 2506-1 | DSC_0026 | 0.0435 | 0.0431 | 0.0122 | 0.0090 | 0.0000 |
| 2506-1 | DSC_0025 | 0.0417 | 0.0355 | 0.0111 | 0.0289 | 0.0000 |
| 2506-1 | DSC_0027 | 0.0370 | 0.0354 | 0.0097 | 0.0111 | 0.0000 |
| 2506-1 | DSC_0006 | 0.0343 | 0.0352 | 0.0138 | 0.0016 | 0.0000 |
| 2506-1 | DSC_0017 | 0.0307 | 0.0288 | 0.0043 | 0.0234 | 0.0000 |
| 2506-1 | DSC_0004 | 0.0300 | 0.0120 | 0.0251 | 0.0074 | 0.0000 |
