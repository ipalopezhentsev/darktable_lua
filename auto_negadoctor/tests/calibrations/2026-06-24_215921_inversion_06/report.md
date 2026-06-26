# Calibration session — inversion

- comment: first cmaes for inversion. small generations for start. compare with _02
- created: 2026-06-24T20:23:18
- git commit: 7bb7153
- rolls: 2506-1, 2510-11-1
- metric: histogram_emd_total (render vs GT-render, picture-vs-picture)
- fit method: cmaes
- method params: sigma=0.3, popsize=16, max_iters=10, seed=0, workers=auto
- fitted params: a, l, l
- trial count: 160
- wall time: 5762.39s (prep 53.81s)
- objective: 0.0556 (init) -> 0.0434 (final)

## Convergence

- method: cmaes  trials: 160
- generations run: 10 (popsize 16, sigma0 0.3, seed 0)
- per-generation best objective (sigma):
  - gen 1: 0.0574 (sigma 0.2777)
  - gen 2: 0.0548 (sigma 0.2604)
  - gen 3: 0.0493 (sigma 0.2536)
  - gen 4: 0.0493 (sigma 0.2501)
  - gen 5: 0.0493 (sigma 0.2448)
  - gen 6: 0.0493 (sigma 0.2431)
  - gen 7: 0.0474 (sigma 0.2414)
  - gen 8: 0.0441 (sigma 0.2386)
  - gen 9: 0.0441 (sigma 0.2385)
  - gen 10: 0.0434 (sigma 0.2367)
- best-so-far improved 7 time(s); curve (trial: objective):
  - 1: 1000000.0950
  - 2: 0.0574
  - 27: 0.0548
  - 45: 0.0493
  - 97: 0.0474
  - 122: 0.0441
  - 160: 0.0434

## Fitted constants (record-only — adopt by hand)

| constant | init | fitted |
|---|---|---|
| P_LOW | 0.5000 | 0.5000 |
| P_HIGH | 99.7227 | 99.7298 |
| OFFSET_DEFAULT | 0.0063 | 0.0293 |
| DMAX_DEFAULT | 1.7218 | 1.5000 |
| PATCH_WIN_FRAC | 0.0494 | 0.0248 |
| PATCH_STRIDE_DIV | 2.0000 | 3.0835 |
| HIGHLIGHT_BAND_PCT[0] | 73.7500 | 80.0000 |
| HIGHLIGHT_BAND_PCT[1] | 96.7500 | 99.0000 |
| SHADOW_MIN_LUMA | 0.0050 | 0.0010 |
| PATCH_CHROMA_MAX | 0.3500 | 0.4189 |
| PATCH_CHROMA_FLOOR | 0.0200 | 0.0178 |
| PATCH_UNIFORMITY_MAX | 0.4500 | 0.3942 |
| PATCH_LUMA_FLOOR | 0.0200 | 0.0050 |
| MIN_PATCH_DENSITY | 0.0500 | 0.0100 |
| HIGHLIGHT_CLIP_FRAC_MAX | 0.0200 | 0.0050 |
| WB_LOW_BAND_PCT[0] | 0.0000 | 0.0000 |
| WB_LOW_BAND_PCT[1] | 20.0000 | 44.0947 |
| WB_HIGH_BAND_PCT[0] | 70.0000 | 56.2194 |
| WB_HIGH_BAND_PCT[1] | 95.0000 | 85.0000 |
| WB_LOW_DESAT | 0.0000 | 0.4980 |
| WB_HIGH_DESAT | 0.0000 | 0.0000 |
| WB_REGION_MIN_FRAC | 0.0001 | 0.0000 |
| WB_HIGH_PRIOR[0] | 1.7590 | 1.8416 |
| WB_HIGH_PRIOR[1] | 1.3656 | 1.6262 |
| WB_HIGH_PRIOR[2] | 1.0641 | 1.1715 |
| WB_LOW_PRIOR[0] | 1.0891 | 1.1054 |
| WB_LOW_PRIOR[1] | 0.7461 | 0.9836 |
| WB_LOW_PRIOR[2] | 0.6941 | 0.5701 |
| PRINT_HI_PCT | 99.9155 | 99.4458 |
| PRINT_HI_CEIL | 0.9693 | 0.9194 |
| PRINT_CLIP_BUDGET | 0.0030 | 0.0000 |
| PRINT_TUNE_ITERS | 11.0000 | 13.6818 |
| PRINT_GAMMA | 4.5312 | 4.6537 |

## Aggregate (all rolls)

- median_total: 0.0434
- median_luma: 0.0360
- median_color: 0.0211
- median_hi999: 0.0105
- max_clip: 0.0000
- frames_over_clip_budget: 0
- clip_budget: 0.0100
- n_frames: 76

## Per-roll

### 2506-1  (prep 24.97s)
```
{
  "n": 38,
  "median_total": 0.027271171097528496,
  "worst_frames": [
    {
      "stem": "DSC_0008",
      "total": 0.10127863469435681
    },
    {
      "stem": "DSC_0029",
      "total": 0.09986238701099326
    },
    {
      "stem": "DSC_0024",
      "total": 0.06899824232602021
    },
    {
      "stem": "DSC_0038",
      "total": 0.06193836761969027
    },
    {
      "stem": "DSC_0021",
      "total": 0.061551935524358704
    }
  ]
}
```
### 2510-11-1  (prep 28.84s)
```
{
  "n": 38,
  "median_total": 0.055766647459367664,
  "worst_frames": [
    {
      "stem": "DSC_0010",
      "total": 0.14273904956595782
    },
    {
      "stem": "DSC_0008",
      "total": 0.12393425052511092
    },
    {
      "stem": "DSC_0005",
      "total": 0.11913402323757089
    },
    {
      "stem": "DSC_0007",
      "total": 0.0991006849648759
    },
    {
      "stem": "DSC_0016",
      "total": 0.09120312510856791
    }
  ]
}
```

## Worst frames (find where to look)

| roll | stem | total | luma | color | hi999 | clip |
|---|---|---|---|---|---|---|
| 2510-11-1 | DSC_0010 | 0.1427 | 0.1664 | 0.0724 | 0.0246 | 0.0000 |
| 2510-11-1 | DSC_0008 | 0.1239 | 0.1390 | 0.0914 | 0.0241 | 0.0000 |
| 2510-11-1 | DSC_0005 | 0.1191 | 0.1438 | 0.0614 | 0.0258 | 0.0000 |
| 2506-1 | DSC_0008 | 0.1013 | 0.1005 | 0.0100 | 0.0001 | 0.0000 |
| 2506-1 | DSC_0029 | 0.0999 | 0.1250 | 0.0502 | 0.0457 | 0.0000 |
| 2510-11-1 | DSC_0007 | 0.0991 | 0.1203 | 0.0582 | 0.0121 | 0.0000 |
| 2510-11-1 | DSC_0016 | 0.0912 | 0.0936 | 0.0192 | 0.0399 | 0.0000 |
| 2510-11-1 | DSC_0040 | 0.0885 | 0.1020 | 0.0354 | 0.0147 | 0.0000 |
| 2510-11-1 | DSC_0019 | 0.0829 | 0.0786 | 0.0154 | 0.0309 | 0.0000 |
| 2510-11-1 | DSC_0022 | 0.0799 | 0.0635 | 0.0283 | 0.0149 | 0.0000 |
| 2510-11-1 | DSC_0024 | 0.0799 | 0.0647 | 0.0439 | 0.0020 | 0.0000 |
| 2510-11-1 | DSC_0031 | 0.0736 | 0.0930 | 0.0436 | 0.0973 | 0.0000 |
| 2510-11-1 | DSC_0020 | 0.0705 | 0.0823 | 0.0275 | 0.0199 | 0.0000 |
| 2506-1 | DSC_0024 | 0.0690 | 0.0613 | 0.0127 | 0.0023 | 0.0000 |
| 2510-11-1 | DSC_0027 | 0.0679 | 0.0581 | 0.0261 | 0.0323 | 0.0000 |
