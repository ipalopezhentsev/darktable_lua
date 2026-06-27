# Calibration session — inversion

- comment: same as _07 but with 2x downsample. compare with it and _02.
- created: 2026-06-27T01:35:20
- git commit: d6f21ed
- rolls: 2506-1, 2510-11-1
- metric: histogram_emd_total (render vs GT-render, picture-vs-picture)
- fit method: cmaes
- downsample (calibration-only): 2x
- method params: sigma=0.3, popsize=16, max_iters=100, seed=1, workers=1
- fitted params: a, l, l
- trial count: 1600
- wall time: 26667.55s (prep 85.54s)
- objective: 0.0545 (init) -> 0.0389 (final)

## Convergence

- method: cmaes  trials: 1600
- generations run: 100 (popsize 16, sigma0 0.3, seed 1)
- per-generation best objective (sigma):
  - gen 1: 0.0480 (sigma 0.2792)
  - gen 2: 0.0480 (sigma 0.2718)
  - gen 3: 0.0480 (sigma 0.2683)
  - gen 4: 0.0480 (sigma 0.2740)
  - gen 5: 0.0480 (sigma 0.2790)
  - gen 6: 0.0480 (sigma 0.2836)
  - gen 7: 0.0480 (sigma 0.2854)
  - gen 8: 0.0480 (sigma 0.2901)
  - gen 9: 0.0480 (sigma 0.2952)
  - gen 10: 0.0480 (sigma 0.2987)
  - gen 11: 0.0442 (sigma 0.3059)
  - gen 12: 0.0442 (sigma 0.3116)
  - gen 13: 0.0442 (sigma 0.3156)
  - gen 14: 0.0442 (sigma 0.3212)
  - gen 15: 0.0442 (sigma 0.3259)
  - gen 16: 0.0426 (sigma 0.3340)
  - gen 17: 0.0426 (sigma 0.3474)
  - gen 18: 0.0426 (sigma 0.3615)
  - gen 19: 0.0426 (sigma 0.3659)
  - gen 20: 0.0421 (sigma 0.3691)
  - gen 21: 0.0421 (sigma 0.3759)
  - gen 22: 0.0421 (sigma 0.3827)
  - gen 23: 0.0421 (sigma 0.3843)
  - gen 24: 0.0416 (sigma 0.3844)
  - gen 25: 0.0416 (sigma 0.3792)
  - gen 26: 0.0416 (sigma 0.3690)
  - gen 27: 0.0416 (sigma 0.3591)
  - gen 28: 0.0416 (sigma 0.3500)
  - gen 29: 0.0412 (sigma 0.3505)
  - gen 30: 0.0412 (sigma 0.3478)
  - gen 31: 0.0412 (sigma 0.3427)
  - gen 32: 0.0412 (sigma 0.3456)
  - gen 33: 0.0412 (sigma 0.3498)
  - gen 34: 0.0412 (sigma 0.3547)
  - gen 35: 0.0412 (sigma 0.3610)
  - gen 36: 0.0408 (sigma 0.3718)
  - gen 37: 0.0406 (sigma 0.3837)
  - gen 38: 0.0406 (sigma 0.3990)
  - gen 39: 0.0406 (sigma 0.4045)
  - gen 40: 0.0403 (sigma 0.4094)
  - gen 41: 0.0403 (sigma 0.4117)
  - gen 42: 0.0403 (sigma 0.4179)
  - gen 43: 0.0403 (sigma 0.4165)
  - gen 44: 0.0403 (sigma 0.4184)
  - gen 45: 0.0403 (sigma 0.4154)
  - gen 46: 0.0403 (sigma 0.4147)
  - gen 47: 0.0403 (sigma 0.4098)
  - gen 48: 0.0403 (sigma 0.4031)
  - gen 49: 0.0396 (sigma 0.3950)
  - gen 50: 0.0392 (sigma 0.3897)
  - gen 51: 0.0392 (sigma 0.3958)
  - gen 52: 0.0392 (sigma 0.3984)
  - gen 53: 0.0392 (sigma 0.4050)
  - gen 54: 0.0392 (sigma 0.4064)
  - gen 55: 0.0392 (sigma 0.4088)
  - gen 56: 0.0392 (sigma 0.4127)
  - gen 57: 0.0392 (sigma 0.4151)
  - gen 58: 0.0392 (sigma 0.4158)
  - gen 59: 0.0392 (sigma 0.4127)
  - gen 60: 0.0392 (sigma 0.4124)
  - gen 61: 0.0392 (sigma 0.4131)
  - gen 62: 0.0392 (sigma 0.4090)
  - gen 63: 0.0392 (sigma 0.4168)
  - gen 64: 0.0392 (sigma 0.4152)
  - gen 65: 0.0392 (sigma 0.4053)
  - gen 66: 0.0392 (sigma 0.3979)
  - gen 67: 0.0392 (sigma 0.3908)
  - gen 68: 0.0392 (sigma 0.3845)
  - gen 69: 0.0392 (sigma 0.3860)
  - gen 70: 0.0392 (sigma 0.3891)
  - gen 71: 0.0392 (sigma 0.3888)
  - gen 72: 0.0392 (sigma 0.3953)
  - gen 73: 0.0392 (sigma 0.3982)
  - gen 74: 0.0392 (sigma 0.3990)
  - gen 75: 0.0392 (sigma 0.3874)
  - gen 76: 0.0392 (sigma 0.3734)
  - gen 77: 0.0392 (sigma 0.3603)
  - gen 78: 0.0392 (sigma 0.3594)
  - gen 79: 0.0392 (sigma 0.3666)
  - gen 80: 0.0392 (sigma 0.3692)
  - gen 81: 0.0392 (sigma 0.3705)
  - gen 82: 0.0392 (sigma 0.3653)
  - gen 83: 0.0392 (sigma 0.3679)
  - gen 84: 0.0392 (sigma 0.3713)
  - gen 85: 0.0392 (sigma 0.3778)
  - gen 86: 0.0392 (sigma 0.3854)
  - gen 87: 0.0391 (sigma 0.3920)
  - gen 88: 0.0391 (sigma 0.3951)
  - gen 89: 0.0391 (sigma 0.3940)
  - gen 90: 0.0391 (sigma 0.3898)
  - gen 91: 0.0391 (sigma 0.3775)
  - gen 92: 0.0391 (sigma 0.3694)
  - gen 93: 0.0391 (sigma 0.3611)
  - gen 94: 0.0391 (sigma 0.3538)
  - gen 95: 0.0391 (sigma 0.3482)
  - gen 96: 0.0391 (sigma 0.3461)
  - gen 97: 0.0389 (sigma 0.3477)
  - gen 98: 0.0389 (sigma 0.3508)
  - gen 99: 0.0389 (sigma 0.3513)
  - gen 100: 0.0389 (sigma 0.3437)
- best-so-far improved 16 time(s); curve (trial: objective):
  - 1: 0.1017
  - 2: 0.0734
  - 4: 0.0713
  - 5: 0.0480
  - 166: 0.0442
  - 245: 0.0426
  - 306: 0.0421
  - 381: 0.0416
  - 449: 0.0412
  - 571: 0.0408
  - 577: 0.0406
  - 636: 0.0403
  - 783: 0.0396
  - 785: 0.0392
  - 1389: 0.0391
  - 1548: 0.0389

## Fitted constants (record-only — adopt by hand)

| constant | init | fitted |
|---|---|---|
| P_LOW | 0.5000 | 0.5000 |
| P_HIGH | 99.7227 | 98.0000 |
| OFFSET_DEFAULT | 0.0063 | 0.0715 |
| DMAX_DEFAULT | 1.7218 | 1.5000 |
| PATCH_WIN_FRAC | 0.0494 | 0.0200 |
| PATCH_STRIDE_DIV | 2.0000 | 4.0000 |
| HIGHLIGHT_BAND_PCT[0] | 73.7500 | 80.0000 |
| HIGHLIGHT_BAND_PCT[1] | 96.7500 | 99.0000 |
| SHADOW_MIN_LUMA | 0.0050 | 0.0167 |
| PATCH_CHROMA_MAX | 0.3500 | 0.6000 |
| PATCH_CHROMA_FLOOR | 0.0200 | 0.0500 |
| PATCH_UNIFORMITY_MAX | 0.4500 | 0.2000 |
| PATCH_LUMA_FLOOR | 0.0200 | 0.0050 |
| MIN_PATCH_DENSITY | 0.0500 | 0.1500 |
| HIGHLIGHT_CLIP_FRAC_MAX | 0.0200 | 0.0050 |
| WB_LOW_BAND_PCT[0] | 0.0000 | 0.0000 |
| WB_LOW_BAND_PCT[1] | 20.0000 | 20.0000 |
| WB_HIGH_BAND_PCT[0] | 70.0000 | 50.0000 |
| WB_HIGH_BAND_PCT[1] | 95.0000 | 97.6748 |
| WB_LOW_DESAT | 0.0000 | 0.7438 |
| WB_HIGH_DESAT | 0.0000 | 0.8203 |
| WB_REGION_MIN_FRAC | 0.0001 | 0.0010 |
| WB_HIGH_PRIOR[0] | 1.7590 | 2.5000 |
| WB_HIGH_PRIOR[1] | 1.3656 | 2.0000 |
| WB_HIGH_PRIOR[2] | 1.0641 | 1.2000 |
| WB_LOW_PRIOR[0] | 1.0891 | 1.2000 |
| WB_LOW_PRIOR[1] | 0.7461 | 1.0000 |
| WB_LOW_PRIOR[2] | 0.6941 | 0.5000 |
| PRINT_HI_PCT | 99.9155 | 99.0000 |
| PRINT_HI_CEIL | 0.9693 | 0.8000 |
| PRINT_CLIP_BUDGET | 0.0030 | 0.0000 |
| PRINT_TUNE_ITERS | 11.0000 | 6.0000 |
| PRINT_GAMMA | 4.5312 | 4.0000 |

## Aggregate (all rolls)

- median_total: 0.0389
- median_luma: 0.0344
- median_color: 0.0161
- median_hi999: 0.0309
- max_clip: 0.0011
- frames_over_clip_budget: 0
- clip_budget: 0.0100
- n_frames: 76

## Per-roll

### 2506-1  (prep 31.76s)
```
{
  "n": 38,
  "median_total": 0.033424050976182174,
  "worst_frames": [
    {
      "stem": "DSC_0029",
      "total": 0.11747102151112254
    },
    {
      "stem": "DSC_0008",
      "total": 0.07681067142216132
    },
    {
      "stem": "DSC_0018",
      "total": 0.0690798876925301
    },
    {
      "stem": "DSC_0013",
      "total": 0.064014456849084
    },
    {
      "stem": "DSC_0032",
      "total": 0.05771319339646386
    }
  ]
}
```
### 2510-11-1  (prep 53.78s)
```
{
  "n": 38,
  "median_total": 0.051229083551846244,
  "worst_frames": [
    {
      "stem": "DSC_0040",
      "total": 0.14959328926515425
    },
    {
      "stem": "DSC_0010",
      "total": 0.13023114623492105
    },
    {
      "stem": "DSC_0005",
      "total": 0.11271172104153844
    },
    {
      "stem": "DSC_0024",
      "total": 0.10609854386085589
    },
    {
      "stem": "DSC_0013",
      "total": 0.08863842920253019
    }
  ]
}
```

## Worst frames (find where to look)

| roll | stem | total | luma | color | hi999 | clip |
|---|---|---|---|---|---|---|
| 2510-11-1 | DSC_0040 | 0.1496 | 0.1734 | 0.0554 | 0.0170 | 0.0000 |
| 2510-11-1 | DSC_0010 | 0.1302 | 0.1510 | 0.0622 | 0.0382 | 0.0000 |
| 2506-1 | DSC_0029 | 0.1175 | 0.1462 | 0.0465 | 0.0449 | 0.0000 |
| 2510-11-1 | DSC_0005 | 0.1127 | 0.1314 | 0.0405 | 0.0486 | 0.0000 |
| 2510-11-1 | DSC_0024 | 0.1061 | 0.1049 | 0.0020 | 0.0146 | 0.0000 |
| 2510-11-1 | DSC_0013 | 0.0886 | 0.0720 | 0.0150 | 0.0255 | 0.0000 |
| 2510-11-1 | DSC_0022 | 0.0799 | 0.0652 | 0.0158 | 0.0187 | 0.0000 |
| 2510-11-1 | DSC_0016 | 0.0770 | 0.0772 | 0.0266 | 0.0156 | 0.0000 |
| 2506-1 | DSC_0008 | 0.0768 | 0.0746 | 0.0031 | 0.0275 | 0.0000 |
| 2510-11-1 | DSC_0026 | 0.0758 | 0.0650 | 0.0262 | 0.0389 | 0.0000 |
| 2510-11-1 | DSC_0030 | 0.0732 | 0.0892 | 0.0455 | 0.0574 | 0.0000 |
| 2510-11-1 | DSC_0035 | 0.0717 | 0.0746 | 0.0450 | 0.0269 | 0.0000 |
| 2510-11-1 | DSC_0002 | 0.0694 | 0.0681 | 0.0044 | 0.0059 | 0.0000 |
| 2506-1 | DSC_0018 | 0.0691 | 0.0787 | 0.0140 | 0.0506 | 0.0000 |
| 2510-11-1 | DSC_0003 | 0.0683 | 0.0819 | 0.0381 | 0.0587 | 0.0000 |
