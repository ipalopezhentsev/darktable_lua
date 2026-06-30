# NOTE - DEPRECATED BY _15

# Calibration session — inversion

- comment: pca on 4 rolls. NOTE: MUST BE RERUN, I CHANGED GT AND METRICS TO IT
- created: 2026-06-28T12:11:35
- git commit: 91b1d9e
- rolls: 2506-1, 2510-11-1, 2511-12-1, 2512-2601-1
- metric: histogram_emd_total (render vs GT-render, picture-vs-picture)
- fit method: none
- fitted params: a, l, l
- trial count: 1
- wall time: 74181.47s (prep 177.64s)
- objective: 0.0634 (init) -> 0.0634 (final)

## Convergence

- method: none  trials: 1
- best-so-far improved 1 time(s); curve (trial: objective):
  - 1: 0.0634

## Principal components (curvature of the metric at the optimum)

Eigen-decomposition of the finite-difference Hessian in grid-step-normalized units (595 extra evals). Each component is a DIRECTION in param space (loadings show the coupled params); larger |eigenvalue| = the metric changes more sharply along it.

- **PC1** eigenvalue 0.0127 (18% of total curvature): -0.48·PRINT_HI_PCT  -0.40·WB_HIGH_PRIOR[2]  -0.39·PRINT_GAMMA  -0.34·WB_HIGH_PRIOR[0]  -0.31·PRINT_HI_CEIL  +0.28·WB_HIGH_PRIOR[1]
- **PC2** eigenvalue -0.0115 (16% of total curvature): -0.72·PRINT_HI_PCT  +0.41·PRINT_HI_CEIL  +0.41·PATCH_WIN_FRAC  +0.25·WB_HIGH_PRIOR[0]  +0.13·OFFSET_DEFAULT  +0.12·PATCH_UNIFORMITY_MAX
- **PC3** eigenvalue -0.0090 (13% of total curvature): -0.71·PRINT_GAMMA  +0.38·HIGHLIGHT_BAND_PCT[1]  +0.37·PATCH_WIN_FRAC  -0.28·PATCH_STRIDE_DIV  -0.19·WB_HIGH_PRIOR[1]  -0.15·OFFSET_DEFAULT
- **PC4** eigenvalue -0.0065 (9% of total curvature): -0.37·DMAX_DEFAULT  -0.35·WB_HIGH_PRIOR[1]  -0.35·OFFSET_DEFAULT  -0.35·PATCH_UNIFORMITY_MAX  -0.30·PATCH_STRIDE_DIV  +0.30·PRINT_HI_CEIL
- **PC5** eigenvalue -0.0056 (8% of total curvature): -0.57·WB_HIGH_PRIOR[2]  -0.45·HIGHLIGHT_BAND_PCT[1]  +0.39·PRINT_HI_CEIL  +0.34·WB_HIGH_PRIOR[0]  +0.24·HIGHLIGHT_BAND_PCT[0]  -0.22·PRINT_GAMMA
- **PC6** eigenvalue 0.0047 (7% of total curvature): -0.46·OFFSET_DEFAULT  -0.40·HIGHLIGHT_BAND_PCT[0]  +0.39·P_HIGH  -0.32·PRINT_GAMMA  +0.30·WB_HIGH_PRIOR[1]  +0.26·WB_HIGH_PRIOR[0]
- **PC7** eigenvalue -0.0035 (5% of total curvature): -0.60·WB_HIGH_PRIOR[2]  +0.60·HIGHLIGHT_BAND_PCT[1]  -0.29·WB_HIGH_PRIOR[1]  +0.23·PRINT_GAMMA  -0.20·PATCH_WIN_FRAC  +0.14·PATCH_CHROMA_MAX
- **PC8** eigenvalue -0.0028 (4% of total curvature): +0.60·DMAX_DEFAULT  -0.41·PATCH_STRIDE_DIV  -0.32·PATCH_WIN_FRAC  +0.31·P_HIGH  -0.30·PRINT_TUNE_ITERS  +0.25·OFFSET_DEFAULT

### Per-param sensitivity (freeze signal)

Central-difference gradient + diagonal curvature in grid-step units, and the largest objective change from a single ±grid_step move. Sorted loudest-first — params near the bottom (max Δobj ≈ 0) are flat and can be FROZEN.

| param | grid_step | gradient | curvature | max Δobj |
|---|---|---|---|---|
| PRINT_HI_PCT | 0.1 | -0.002852 | -0.003817 | 0.004760 |
| PATCH_WIN_FRAC | 0.01 | +0.002631 | -0.003339 | 0.004301 |
| WB_HIGH_PRIOR[2] | 0.05 | -0.003081 | -0.001309 | 0.003736 |
| HIGHLIGHT_BAND_PCT[1] | 1 | -0.000266 | -0.003444 | 0.001988 |
| PRINT_HI_CEIL | 0.01 | +0.000769 | -0.002076 | 0.001808 |
| PATCH_STRIDE_DIV | 1 | -0.000809 | -0.001655 | 0.001636 |
| WB_HIGH_PRIOR[1] | 0.05 | +0.000889 | +0.001122 | 0.001450 |
| PRINT_GAMMA | 0.25 | -0.000079 | -0.002663 | 0.001410 |
| HIGHLIGHT_BAND_PCT[0] | 2 | -0.000121 | +0.001809 | 0.001025 |
| DMAX_DEFAULT | 0.05 | +0.000157 | -0.001563 | 0.000939 |
| PATCH_UNIFORMITY_MAX | 0.05 | -0.000860 | -0.000153 | 0.000937 |
| OFFSET_DEFAULT | 0.01 | +0.000561 | +0.000557 | 0.000839 |
| WB_HIGH_PRIOR[0] | 0.05 | +0.000462 | +0.000644 | 0.000785 |
| PATCH_CHROMA_MAX | 0.05 | -0.000392 | +0.000783 | 0.000783 |
| P_HIGH | 0.1 | -0.000335 | +0.000195 | 0.000433 |
| PRINT_TUNE_ITERS | 2 | -0.000024 | -0.000189 | 0.000119 |
| P_LOW | 0.5 | -0.000050 | -0.000101 | 0.000101 |
| WB_LOW_BAND_PCT[1] | 2 | +0.000027 | +0.000055 | 0.000055 |
| WB_LOW_DESAT | 0.05 | +0.000014 | +0.000028 | 0.000028 |
| WB_LOW_BAND_PCT[0] | 1 | +0.000006 | +0.000013 | 0.000013 |
| WB_LOW_PRIOR[1] | 0.05 | +0.000000 | +0.000000 | 0.000000 |
| WB_LOW_PRIOR[2] | 0.05 | -0.000000 | +0.000000 | 0.000000 |
| WB_LOW_PRIOR[0] | 0.05 | -0.000000 | +0.000000 | 0.000000 |
| SHADOW_MIN_LUMA | 0.001 | +0.000000 | +0.000000 | 0.000000 |
| PATCH_CHROMA_FLOOR | 0.005 | +0.000000 | +0.000000 | 0.000000 |
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

- median_total: 0.0634
- median_luma: 0.0509
- median_color: 0.0202
- median_hi999: 0.0098
- max_clip: 0.0021
- frames_over_clip_budget: 0
- clip_budget: 0.0100
- n_frames: 150

## Per-roll

### 2506-1  (prep 42.23s)
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
### 2510-11-1  (prep 44.51s)
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
### 2511-12-1  (prep 46.06s)
```
{
  "n": 37,
  "median_total": 0.05429023938164764,
  "worst_frames": [
    {
      "stem": "DSC_0020",
      "total": 0.2813524604098393
    },
    {
      "stem": "DSC_0036",
      "total": 0.1314681934832516
    },
    {
      "stem": "DSC_0030",
      "total": 0.12102352870208953
    },
    {
      "stem": "DSC_0026",
      "total": 0.10611956586100775
    },
    {
      "stem": "DSC_0015",
      "total": 0.10539459138569303
    }
  ]
}
```
### 2512-2601-1  (prep 44.84s)
```
{
  "n": 37,
  "median_total": 0.08929957776835405,
  "worst_frames": [
    {
      "stem": "DSC_0008",
      "total": 0.19713064225309349
    },
    {
      "stem": "DSC_0014",
      "total": 0.15948403394913932
    },
    {
      "stem": "DSC_0029",
      "total": 0.1560304620629065
    },
    {
      "stem": "DSC_0003",
      "total": 0.1510084698623975
    },
    {
      "stem": "DSC_0013",
      "total": 0.14960578887241424
    }
  ]
}
```

## Worst frames (find where to look)

| roll | stem | total | luma | color | hi999 | clip |
|---|---|---|---|---|---|---|
| 2511-12-1 | DSC_0020 | 0.2814 | 0.2245 | 0.2366 | 0.1822 | 0.0000 |
| 2512-2601-1 | DSC_0008 | 0.1971 | 0.2036 | 0.0167 | 0.0130 | 0.0000 |
| 2506-1 | DSC_0029 | 0.1679 | 0.1812 | 0.0201 | 0.0744 | 0.0002 |
| 2512-2601-1 | DSC_0014 | 0.1595 | 0.1645 | 0.0126 | 0.0119 | 0.0000 |
| 2510-11-1 | DSC_0010 | 0.1570 | 0.1672 | 0.1085 | 0.0213 | 0.0021 |
| 2512-2601-1 | DSC_0029 | 0.1560 | 0.1609 | 0.0076 | 0.0094 | 0.0000 |
| 2512-2601-1 | DSC_0003 | 0.1510 | 0.1595 | 0.0338 | 0.0137 | 0.0000 |
| 2512-2601-1 | DSC_0013 | 0.1496 | 0.1558 | 0.0156 | 0.0129 | 0.0000 |
| 2512-2601-1 | DSC_0005 | 0.1388 | 0.1518 | 0.0274 | 0.0163 | 0.0000 |
| 2511-12-1 | DSC_0036 | 0.1315 | 0.1296 | 0.0126 | 0.0110 | 0.0000 |
| 2512-2601-1 | DSC_0034 | 0.1263 | 0.1228 | 0.0219 | 0.0067 | 0.0000 |
| 2512-2601-1 | DSC_0037 | 0.1260 | 0.1275 | 0.0176 | 0.0095 | 0.0000 |
| 2512-2601-1 | DSC_0010 | 0.1227 | 0.1332 | 0.0326 | 0.0147 | 0.0000 |
| 2511-12-1 | DSC_0030 | 0.1210 | 0.1243 | 0.0109 | 0.0167 | 0.0000 |
| 2510-11-1 | DSC_0015 | 0.1177 | 0.0473 | 0.1173 | 0.0125 | 0.0017 |
