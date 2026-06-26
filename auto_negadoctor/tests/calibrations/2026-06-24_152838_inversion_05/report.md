# Calibration session — inversion

- created: 2026-06-24T10:54:16
- git commit: 7bb7153
- rolls: 2506-1
- metric: histogram_emd_total (render vs GT-render, picture-vs-picture)
- fit method: none
- fitted params: a, l, l
- trial count: 1
- wall time: 16461.18s (prep 29.79s)
- objective: 0.0375 (init) -> 0.0375 (final)

## Convergence

- method: none  trials: 1
- best-so-far improved 1 time(s); curve (trial: objective):
  - 1: 0.0375

## Principal components (curvature of the metric at the optimum)

Eigen-decomposition of the finite-difference Hessian in grid-step-normalized units (595 extra evals). Each component is a DIRECTION in param space (loadings show the coupled params); larger |eigenvalue| = the metric changes more sharply along it.

- **PC1** eigenvalue 0.0178 (24% of total curvature): -0.51·PRINT_GAMMA  -0.46·PRINT_HI_PCT  -0.40·PATCH_WIN_FRAC  -0.29·PATCH_STRIDE_DIV  -0.29·OFFSET_DEFAULT  -0.28·WB_HIGH_PRIOR[2]
- **PC2** eigenvalue -0.0116 (16% of total curvature): -0.64·PRINT_HI_PCT  +0.39·OFFSET_DEFAULT  +0.34·PATCH_STRIDE_DIV  -0.29·PRINT_HI_CEIL  -0.23·WB_HIGH_PRIOR[1]  -0.21·WB_HIGH_PRIOR[0]
- **PC3** eigenvalue -0.0084 (11% of total curvature): +0.61·PATCH_WIN_FRAC  -0.53·PRINT_HI_PCT  -0.46·PATCH_STRIDE_DIV  +0.27·PRINT_GAMMA  -0.19·OFFSET_DEFAULT  +0.12·HIGHLIGHT_BAND_PCT[0]
- **PC4** eigenvalue 0.0072 (10% of total curvature): -0.55·PRINT_HI_CEIL  -0.37·OFFSET_DEFAULT  +0.37·WB_HIGH_PRIOR[2]  +0.36·DMAX_DEFAULT  -0.25·WB_HIGH_PRIOR[1]  +0.23·HIGHLIGHT_BAND_PCT[0]
- **PC5** eigenvalue -0.0060 (8% of total curvature): +0.60·PATCH_STRIDE_DIV  -0.57·OFFSET_DEFAULT  +0.36·WB_HIGH_PRIOR[0]  +0.24·PRINT_GAMMA  -0.18·PRINT_HI_PCT  +0.16·PRINT_HI_CEIL
- **PC6** eigenvalue 0.0058 (8% of total curvature): -0.51·WB_HIGH_PRIOR[0]  +0.43·PRINT_GAMMA  -0.35·WB_HIGH_PRIOR[1]  -0.32·OFFSET_DEFAULT  +0.29·PRINT_HI_CEIL  -0.25·WB_LOW_PRIOR[2]
- **PC7** eigenvalue -0.0038 (5% of total curvature): +0.67·WB_HIGH_PRIOR[2]  +0.50·PATCH_CHROMA_MAX  +0.33·PRINT_HI_CEIL  -0.20·WB_LOW_PRIOR[2]  +0.20·WB_HIGH_PRIOR[1]  +0.17·OFFSET_DEFAULT
- **PC8** eigenvalue 0.0030 (4% of total curvature): +0.54·DMAX_DEFAULT  -0.48·WB_HIGH_PRIOR[2]  +0.46·PATCH_CHROMA_MAX  -0.31·WB_HIGH_PRIOR[0]  -0.27·P_HIGH  +0.18·HIGHLIGHT_BAND_PCT[0]

### Per-param sensitivity (freeze signal)

Central-difference gradient + diagonal curvature in grid-step units, and the largest objective change from a single ±grid_step move. Sorted loudest-first — params near the bottom (max Δobj ≈ 0) are flat and can be FROZEN.

| param | grid_step | gradient | curvature | max Δobj |
|---|---|---|---|---|
| PRINT_HI_CEIL | 0.01 | +0.006501 | +0.002159 | 0.007581 |
| PRINT_HI_PCT | 0.1 | -0.005401 | -0.003402 | 0.007102 |
| PRINT_GAMMA | 0.25 | -0.004815 | +0.004418 | 0.007024 |
| WB_HIGH_PRIOR[2] | 0.05 | -0.004005 | +0.001257 | 0.004634 |
| OFFSET_DEFAULT | 0.01 | -0.003331 | -0.001201 | 0.003932 |
| WB_HIGH_PRIOR[0] | 0.05 | +0.003114 | +0.001312 | 0.003770 |
| DMAX_DEFAULT | 0.05 | -0.002370 | +0.002194 | 0.003467 |
| PATCH_STRIDE_DIV | 1 | -0.001154 | -0.003682 | 0.002995 |
| PRINT_TUNE_ITERS | 2 | +0.001631 | -0.000601 | 0.001931 |
| HIGHLIGHT_BAND_PCT[0] | 2 | +0.001438 | +0.000490 | 0.001683 |
| PATCH_WIN_FRAC | 0.01 | +0.000814 | -0.000664 | 0.001146 |
| WB_HIGH_PRIOR[1] | 0.05 | +0.000508 | -0.000316 | 0.000666 |
| P_HIGH | 0.1 | +0.000203 | +0.000592 | 0.000498 |
| WB_LOW_PRIOR[0] | 0.05 | -0.000176 | +0.000351 | 0.000351 |
| P_LOW | 0.5 | +0.000071 | +0.000143 | 0.000143 |
| WB_LOW_DESAT | 0.05 | -0.000042 | -0.000084 | 0.000084 |
| WB_LOW_BAND_PCT[1] | 2 | -0.000009 | -0.000018 | 0.000018 |
| WB_LOW_BAND_PCT[0] | 1 | -0.000003 | -0.000006 | 0.000006 |
| WB_LOW_PRIOR[1] | 0.05 | +0.000000 | -0.000000 | 0.000000 |
| WB_LOW_PRIOR[2] | 0.05 | -0.000000 | +0.000000 | 0.000000 |
| HIGHLIGHT_BAND_PCT[1] | 1 | +0.000000 | +0.000000 | 0.000000 |
| SHADOW_MIN_LUMA | 0.001 | +0.000000 | +0.000000 | 0.000000 |
| PATCH_CHROMA_MAX | 0.05 | +0.000000 | +0.000000 | 0.000000 |
| PATCH_CHROMA_FLOOR | 0.005 | +0.000000 | +0.000000 | 0.000000 |
| PATCH_UNIFORMITY_MAX | 0.05 | +0.000000 | +0.000000 | 0.000000 |
| PATCH_LUMA_FLOOR | 0.005 | +0.000000 | +0.000000 | 0.000000 |
| MIN_PATCH_DENSITY | 0.01 | +0.000000 | +0.000000 | 0.000000 |
| HIGHLIGHT_CLIP_FRAC_MAX | 0.005 | +0.000000 | +0.000000 | 0.000000 |
| WB_HIGH_BAND_PCT[0] | 2 | +0.000000 | +0.000000 | 0.000000 |
| WB_HIGH_BAND_PCT[1] | 1 | +0.000000 | +0.000000 | 0.000000 |
| WB_HIGH_DESAT | 0.05 | +0.000000 | +0.000000 | 0.000000 |
| WB_REGION_MIN_FRAC | 0.0001 | +0.000000 | +0.000000 | 0.000000 |
| PRINT_CLIP_BUDGET | 0.002 | +0.000000 | +0.000000 | 0.000000 |

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

- median_total: 0.0375
- median_luma: 0.0316
- median_color: 0.0182
- median_hi999: 0.0059
- max_clip: 0.0002
- frames_over_clip_budget: 0
- clip_budget: 0.0100
- n_frames: 38

## Per-roll

### 2506-1  (prep 29.79s)
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

## Worst frames (find where to look)

| roll | stem | total | luma | color | hi999 | clip |
|---|---|---|---|---|---|---|
| 2506-1 | DSC_0029 | 0.1679 | 0.1812 | 0.0201 | 0.0744 | 0.0002 |
| 2506-1 | DSC_0008 | 0.0996 | 0.0870 | 0.0246 | 0.0069 | 0.0000 |
| 2506-1 | DSC_0021 | 0.0949 | 0.0930 | 0.0198 | 0.0220 | 0.0000 |
| 2506-1 | DSC_0032 | 0.0829 | 0.1002 | 0.0301 | 0.0154 | 0.0000 |
| 2506-1 | DSC_0013 | 0.0825 | 0.0821 | 0.0204 | 0.0117 | 0.0000 |
| 2506-1 | DSC_0024 | 0.0731 | 0.0610 | 0.0205 | 0.0102 | 0.0000 |
| 2506-1 | DSC_0025 | 0.0689 | 0.0633 | 0.0165 | 0.0289 | 0.0000 |
| 2506-1 | DSC_0011 | 0.0651 | 0.0566 | 0.0227 | 0.0028 | 0.0000 |
| 2506-1 | DSC_0033 | 0.0619 | 0.0623 | 0.0353 | 0.0092 | 0.0000 |
| 2506-1 | DSC_0038 | 0.0610 | 0.0515 | 0.0194 | 0.0074 | 0.0000 |
| 2506-1 | DSC_0003 | 0.0583 | 0.0612 | 0.0446 | 0.0151 | 0.0000 |
| 2506-1 | DSC_0037 | 0.0579 | 0.0501 | 0.0146 | 0.0118 | 0.0000 |
| 2506-1 | DSC_0005 | 0.0547 | 0.0024 | 0.0613 | 0.0026 | 0.0000 |
| 2506-1 | DSC_0028 | 0.0495 | 0.0433 | 0.0119 | 0.0022 | 0.0000 |
| 2506-1 | DSC_0039 | 0.0481 | 0.0334 | 0.0227 | 0.0004 | 0.0000 |
