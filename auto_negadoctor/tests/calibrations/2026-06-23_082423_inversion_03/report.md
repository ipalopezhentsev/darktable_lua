# Calibration session — inversion

- created: 2026-06-22T23:31:52
- git commit: 0a5f65f
- rolls: 2506-1, 2510-11-1
- metric: histogram_emd_total (render vs GT-render, picture-vs-picture)
- fit method: none
- fitted params: a, l, l
- trial count: 1
- wall time: 31949.79s (prep 118.24s)
- objective: 0.0556 (init) -> 0.0556 (final)

## Convergence

- method: none  trials: 1
- best-so-far improved 1 time(s); curve (trial: objective):
  - 1: 0.0556

## Principal components (curvature of the metric at the optimum)

Eigen-decomposition of the finite-difference Hessian in grid-step-normalized units (595 extra evals). Each component is a DIRECTION in param space (loadings show the coupled params); larger |eigenvalue| = the metric changes more sharply along it.

- **PC1** eigenvalue 0.0200 (18% of total curvature): +0.47·PATCH_STRIDE_DIV  +0.37·WB_HIGH_PRIOR[2]  +0.35·HIGHLIGHT_BAND_PCT[1]  +0.35·PRINT_HI_CEIL  +0.30·PRINT_GAMMA  +0.24·PRINT_HI_PCT
- **PC2** eigenvalue -0.0172 (15% of total curvature): -0.51·HIGHLIGHT_BAND_PCT[1]  -0.51·PATCH_STRIDE_DIV  +0.34·PATCH_WIN_FRAC  +0.32·PRINT_HI_PCT  +0.30·WB_HIGH_PRIOR[1]  +0.23·PRINT_GAMMA
- **PC3** eigenvalue -0.0154 (14% of total curvature): -0.76·PATCH_WIN_FRAC  +0.37·PRINT_GAMMA  +0.36·PRINT_HI_PCT  -0.25·HIGHLIGHT_BAND_PCT[1]  +0.14·PATCH_STRIDE_DIV  -0.12·PATCH_UNIFORMITY_MAX
- **PC4** eigenvalue -0.0117 (10% of total curvature): -0.64·PATCH_STRIDE_DIV  +0.61·HIGHLIGHT_BAND_PCT[1]  +0.26·WB_HIGH_PRIOR[2]  -0.25·PATCH_WIN_FRAC  +0.22·PRINT_HI_PCT  -0.14·HIGHLIGHT_BAND_PCT[0]
- **PC5** eigenvalue -0.0073 (7% of total curvature): +0.57·PRINT_HI_PCT  -0.51·PRINT_GAMMA  -0.39·DMAX_DEFAULT  +0.29·P_HIGH  +0.22·PATCH_UNIFORMITY_MAX  +0.20·PRINT_TUNE_ITERS
- **PC6** eigenvalue -0.0055 (5% of total curvature): -0.47·WB_HIGH_PRIOR[2]  -0.39·DMAX_DEFAULT  +0.38·PRINT_GAMMA  +0.27·PRINT_HI_CEIL  -0.24·HIGHLIGHT_BAND_PCT[0]  -0.23·OFFSET_DEFAULT
- **PC7** eigenvalue 0.0051 (5% of total curvature): +0.57·WB_HIGH_PRIOR[2]  -0.47·WB_HIGH_PRIOR[1]  -0.38·PATCH_CHROMA_MAX  -0.28·OFFSET_DEFAULT  +0.24·PATCH_UNIFORMITY_MAX  -0.20·PRINT_CLIP_BUDGET
- **PC8** eigenvalue 0.0047 (4% of total curvature): +0.76·HIGHLIGHT_BAND_PCT[0]  -0.24·OFFSET_DEFAULT  +0.23·PRINT_GAMMA  +0.23·DMAX_DEFAULT  +0.23·HIGHLIGHT_BAND_PCT[1]  -0.21·WB_HIGH_PRIOR[1]

### Per-param sensitivity (freeze signal)

Central-difference gradient + diagonal curvature in grid-step units (one probe = one grid_step per axis), and the largest objective change from a single ±grid_step move. Sorted loudest-first — params near the bottom (max Δobj ≈ 0) are flat at this point and can be FROZEN (drop from `fit.params`). A huge Δobj means a step crosses a clip/containment cliff (the param gates a hard constraint).

| param | grid_step | gradient | curvature | max Δobj |
|---|---|---|---|---|
| PATCH_WIN_FRAC | 0.01 | +0.003427 | -0.011023 | 0.008939 |
| PATCH_STRIDE_DIV | 1 | -0.003584 | -0.005270 | 0.006219 |
| PRINT_HI_PCT | 0.1 | -0.002970 | -0.005949 | 0.005944 |
| HIGHLIGHT_BAND_PCT[1] | 1 | -0.002075 | -0.007034 | 0.005592 |
| PRINT_GAMMA | 0.25 | -0.003130 | -0.004303 | 0.005282 |
| WB_HIGH_PRIOR[2] | 0.05 | -0.003816 | +0.002098 | 0.004865 |
| PRINT_HI_CEIL | 0.01 | +0.002377 | -0.001124 | 0.002939 |
| OFFSET_DEFAULT | 0.01 | -0.001996 | +0.001767 | 0.002879 |
| WB_HIGH_PRIOR[0] | 0.05 | +0.001352 | +0.000793 | 0.001749 |
| WB_HIGH_DESAT | 0.05 | -0.000780 | -0.001559 | 0.001559 |
| HIGHLIGHT_BAND_PCT[0] | 2 | +0.000000 | +0.003049 | 0.001524 |
| PATCH_UNIFORMITY_MAX | 0.05 | -0.000254 | -0.002333 | 0.001420 |
| PRINT_TUNE_ITERS | 2 | +0.000735 | -0.001018 | 0.001245 |
| DMAX_DEFAULT | 0.05 | -0.000539 | -0.001352 | 0.001215 |
| WB_HIGH_PRIOR[1] | 0.05 | -0.000102 | +0.002105 | 0.001154 |
| P_HIGH | 0.1 | +0.000450 | -0.000997 | 0.000949 |
| PRINT_CLIP_BUDGET | 0.002 | -0.000796 | -0.000233 | 0.000912 |
| WB_HIGH_BAND_PCT[0] | 2 | -0.000304 | +0.000674 | 0.000641 |
| P_LOW | 0.5 | -0.000173 | -0.000347 | 0.000347 |
| WB_HIGH_BAND_PCT[1] | 1 | -0.000145 | +0.000354 | 0.000322 |
| WB_LOW_DESAT | 0.05 | -0.000030 | -0.000059 | 0.000059 |
| WB_LOW_BAND_PCT[1] | 2 | +0.000011 | +0.000022 | 0.000022 |
| WB_LOW_PRIOR[0] | 0.05 | +0.000001 | -0.000000 | 0.000001 |
| WB_LOW_PRIOR[2] | 0.05 | -0.000000 | +0.000000 | 0.000000 |
| WB_LOW_PRIOR[1] | 0.05 | -0.000000 | -0.000000 | 0.000000 |
| WB_LOW_BAND_PCT[0] | 1 | -0.000000 | -0.000000 | 0.000000 |
| SHADOW_MIN_LUMA | 0.001 | +0.000000 | +0.000000 | 0.000000 |
| PATCH_CHROMA_MAX | 0.05 | +0.000000 | +0.000000 | 0.000000 |
| PATCH_CHROMA_FLOOR | 0.005 | +0.000000 | +0.000000 | 0.000000 |
| PATCH_LUMA_FLOOR | 0.005 | +0.000000 | +0.000000 | 0.000000 |
| MIN_PATCH_DENSITY | 0.01 | +0.000000 | +0.000000 | 0.000000 |
| HIGHLIGHT_CLIP_FRAC_MAX | 0.005 | +0.000000 | +0.000000 | 0.000000 |
| WB_REGION_MIN_FRAC | 0.0001 | +0.000000 | +0.000000 | 0.000000 |

## Fitted constants (record-only — adopt by hand)

| constant | init | fitted |
|---|---|---|
| P_LOW | 0.5000 | 0.5000 |
| P_HIGH | 99.7227 | 99.7227 |
| OFFSET_DEFAULT | 0.0063 | 0.0063 |
| DMAX_DEFAULT | 1.7218 | 1.7218 |
| PATCH_WIN_FRAC | 0.0494 | 0.0494 |
| PATCH_STRIDE_DIV | 2.0000 | 2.0000 |
| HIGHLIGHT_BAND_PCT[0] | 73.7500 | 73.7500 |
| HIGHLIGHT_BAND_PCT[1] | 96.7500 | 96.7500 |
| SHADOW_MIN_LUMA | 0.0050 | 0.0050 |
| PATCH_CHROMA_MAX | 0.3500 | 0.3500 |
| PATCH_CHROMA_FLOOR | 0.0200 | 0.0200 |
| PATCH_UNIFORMITY_MAX | 0.4500 | 0.4500 |
| PATCH_LUMA_FLOOR | 0.0200 | 0.0200 |
| MIN_PATCH_DENSITY | 0.0500 | 0.0500 |
| HIGHLIGHT_CLIP_FRAC_MAX | 0.0200 | 0.0200 |
| WB_LOW_BAND_PCT[0] | 0.0000 | 0.0000 |
| WB_LOW_BAND_PCT[1] | 20.0000 | 20.0000 |
| WB_HIGH_BAND_PCT[0] | 70.0000 | 70.0000 |
| WB_HIGH_BAND_PCT[1] | 95.0000 | 95.0000 |
| WB_LOW_DESAT | 0.0000 | 0.0000 |
| WB_HIGH_DESAT | 0.0000 | 0.0000 |
| WB_REGION_MIN_FRAC | 0.0001 | 0.0001 |
| WB_HIGH_PRIOR[0] | 1.7590 | 1.7590 |
| WB_HIGH_PRIOR[1] | 1.3656 | 1.3656 |
| WB_HIGH_PRIOR[2] | 1.0641 | 1.0641 |
| WB_LOW_PRIOR[0] | 1.0891 | 1.0891 |
| WB_LOW_PRIOR[1] | 0.7461 | 0.7461 |
| WB_LOW_PRIOR[2] | 0.6941 | 0.6941 |
| PRINT_HI_PCT | 99.9155 | 99.9155 |
| PRINT_HI_CEIL | 0.9693 | 0.9693 |
| PRINT_CLIP_BUDGET | 0.0030 | 0.0030 |
| PRINT_TUNE_ITERS | 11.0000 | 11.0000 |
| PRINT_GAMMA | 4.5312 | 4.5312 |

## Aggregate (all rolls)

- median_total: 0.0556
- median_luma: 0.0416
- median_color: 0.0228
- median_hi999: 0.0091
- max_clip: 0.0021
- frames_over_clip_budget: 0
- clip_budget: 0.0100
- n_frames: 76

## Per-roll

### 2506-1  (prep 57.16s)
```
{
  "n": 38,
  "median_total": 0.037454973548976284,
  "worst_frames": [
    {
      "stem": "DSC_0029",
      "total": 0.1679245452312731
    },
    {
      "stem": "DSC_0008",
      "total": 0.09963872845935766
    },
    {
      "stem": "DSC_0021",
      "total": 0.09486635146714395
    },
    {
      "stem": "DSC_0032",
      "total": 0.08291340119666447
    },
    {
      "stem": "DSC_0013",
      "total": 0.08254123741892964
    }
  ]
}
```
### 2510-11-1  (prep 61.08s)
```
{
  "n": 38,
  "median_total": 0.06534409427451787,
  "worst_frames": [
    {
      "stem": "DSC_0010",
      "total": 0.15698151204551092
    },
    {
      "stem": "DSC_0015",
      "total": 0.11770949030372335
    },
    {
      "stem": "DSC_0005",
      "total": 0.11219006945725163
    },
    {
      "stem": "DSC_0019",
      "total": 0.10204023328304701
    },
    {
      "stem": "DSC_0007",
      "total": 0.10180463181848938
    }
  ]
}
```

## Worst frames (find where to look)

| roll | stem | total | luma | color | hi999 | clip |
|---|---|---|---|---|---|---|
| 2506-1 | DSC_0029 | 0.1679 | 0.1812 | 0.0201 | 0.0744 | 0.0002 |
| 2510-11-1 | DSC_0010 | 0.1570 | 0.1672 | 0.1085 | 0.0213 | 0.0021 |
| 2510-11-1 | DSC_0015 | 0.1177 | 0.0473 | 0.1173 | 0.0125 | 0.0017 |
| 2510-11-1 | DSC_0005 | 0.1122 | 0.1344 | 0.0583 | 0.0157 | 0.0000 |
| 2510-11-1 | DSC_0019 | 0.1020 | 0.0940 | 0.0305 | 0.0408 | 0.0000 |
| 2510-11-1 | DSC_0007 | 0.1018 | 0.1221 | 0.0482 | 0.0115 | 0.0000 |
| 2510-11-1 | DSC_0011 | 0.1004 | 0.0714 | 0.0884 | 0.0224 | 0.0004 |
| 2506-1 | DSC_0008 | 0.0996 | 0.0870 | 0.0246 | 0.0069 | 0.0000 |
| 2510-11-1 | DSC_0020 | 0.0993 | 0.1050 | 0.0706 | 0.0328 | 0.0000 |
| 2506-1 | DSC_0021 | 0.0949 | 0.0930 | 0.0198 | 0.0220 | 0.0000 |
| 2510-11-1 | DSC_0035 | 0.0865 | 0.0864 | 0.0540 | 0.0639 | 0.0000 |
| 2510-11-1 | DSC_0026 | 0.0858 | 0.0181 | 0.0913 | 0.0074 | 0.0019 |
| 2510-11-1 | DSC_0031 | 0.0855 | 0.0797 | 0.0282 | 0.0461 | 0.0000 |
| 2506-1 | DSC_0032 | 0.0829 | 0.1002 | 0.0301 | 0.0154 | 0.0000 |
| 2506-1 | DSC_0013 | 0.0825 | 0.0821 | 0.0204 | 0.0117 | 0.0000 |
