# Calibration session — inversion

- comment: pca on 4 rolls AFTER ALL CORRECT CHANGES. Compare with _05. Discard _13
- created: 2026-06-29T09:06:02
- git commit: 644b3a2
- rolls: 2506-1, 2510-11-1, 2511-12-1, 2512-2601-1
- metric: histogram_emd_total (render vs GT-render, picture-vs-picture)
- fit method: none
- fitted params: a, l, l
- trial count: 1
- wall time: 70203.1s (prep 194.31s)
- objective: 0.0602 (init) -> 0.0602 (final)

## Convergence

- method: none  trials: 1
- best-so-far improved 1 time(s); curve (trial: objective):
  - 1: 0.0602

## Principal components (curvature of the metric at the optimum)

Eigen-decomposition of the finite-difference Hessian in grid-step-normalized units (595 extra evals). Each component is a DIRECTION in param space (loadings show the coupled params); larger |eigenvalue| = the metric changes more sharply along it.

- **PC1** eigenvalue 0.0153 (17% of total curvature): -0.43·OFFSET_DEFAULT  -0.41·DMAX_DEFAULT  -0.40·WB_HIGH_PRIOR[2]  -0.35·PRINT_GAMMA  -0.34·HIGHLIGHT_BAND_PCT[1]  -0.27·PRINT_HI_CEIL
- **PC2** eigenvalue -0.0133 (15% of total curvature): +0.53·HIGHLIGHT_BAND_PCT[1]  -0.44·PATCH_WIN_FRAC  -0.35·PRINT_HI_PCT  -0.33·WB_HIGH_PRIOR[1]  -0.26·WB_HIGH_PRIOR[0]  -0.24·PRINT_HI_CEIL
- **PC3** eigenvalue -0.0114 (13% of total curvature): +0.46·PRINT_HI_CEIL  -0.45·OFFSET_DEFAULT  +0.45·PRINT_HI_PCT  +0.43·HIGHLIGHT_BAND_PCT[1]  -0.32·DMAX_DEFAULT  +0.14·PATCH_STRIDE_DIV
- **PC4** eigenvalue -0.0096 (11% of total curvature): -0.64·PRINT_GAMMA  -0.40·OFFSET_DEFAULT  +0.33·DMAX_DEFAULT  +0.30·WB_HIGH_PRIOR[2]  +0.30·HIGHLIGHT_BAND_PCT[1]  +0.21·WB_HIGH_PRIOR[0]
- **PC5** eigenvalue 0.0096 (11% of total curvature): -0.53·PATCH_STRIDE_DIV  +0.40·WB_HIGH_PRIOR[2]  +0.38·PRINT_HI_PCT  +0.34·PATCH_CHROMA_MAX  -0.33·WB_HIGH_PRIOR[1]  +0.20·PATCH_UNIFORMITY_MAX
- **PC6** eigenvalue -0.0058 (6% of total curvature): +0.56·OFFSET_DEFAULT  -0.48·PRINT_GAMMA  +0.35·PATCH_WIN_FRAC  -0.34·WB_HIGH_PRIOR[0]  +0.29·HIGHLIGHT_BAND_PCT[1]  +0.19·PRINT_HI_CEIL
- **PC7** eigenvalue 0.0047 (5% of total curvature): +0.59·P_HIGH  +0.39·PATCH_UNIFORMITY_MAX  -0.35·PRINT_HI_CEIL  -0.33·PATCH_WIN_FRAC  +0.30·WB_HIGH_PRIOR[0]  +0.22·P_LOW
- **PC8** eigenvalue -0.0040 (4% of total curvature): +0.47·PATCH_STRIDE_DIV  -0.38·PRINT_HI_CEIL  +0.35·PATCH_UNIFORMITY_MAX  -0.32·P_LOW  +0.29·PRINT_HI_PCT  -0.25·WB_HIGH_PRIOR[2]

### Per-param sensitivity (freeze signal)

Central-difference gradient + diagonal curvature in grid-step units, and the largest objective change from a single ±grid_step move. Sorted loudest-first — params near the bottom (max Δobj ≈ 0) are flat and can be FROZEN.

| param | grid_step | gradient | curvature | max Δobj |
|---|---|---|---|---|
| WB_HIGH_PRIOR[2] | 0.05 | -0.006746 | +0.002036 | 0.007764 |
| PRINT_HI_PCT | 0.1 | -0.005349 | -0.002741 | 0.006720 |
| PRINT_GAMMA | 0.25 | -0.004036 | -0.003257 | 0.005665 |
| PATCH_WIN_FRAC | 0.01 | +0.003879 | -0.001841 | 0.004800 |
| PRINT_HI_CEIL | 0.01 | +0.003778 | -0.001828 | 0.004693 |
| HIGHLIGHT_BAND_PCT[1] | 1 | -0.001882 | -0.005323 | 0.004544 |
| WB_HIGH_PRIOR[0] | 0.05 | +0.003355 | -0.001574 | 0.004142 |
| OFFSET_DEFAULT | 0.01 | -0.002265 | -0.002704 | 0.003617 |
| PATCH_STRIDE_DIV | 1 | -0.002217 | +0.001961 | 0.003197 |
| DMAX_DEFAULT | 0.05 | -0.002983 | -0.000171 | 0.003069 |
| PRINT_TUNE_ITERS | 2 | +0.001672 | -0.001117 | 0.002231 |
| P_HIGH | 0.1 | +0.000551 | +0.001328 | 0.001215 |
| WB_HIGH_PRIOR[1] | 0.05 | +0.000783 | +0.000159 | 0.000863 |
| HIGHLIGHT_BAND_PCT[0] | 2 | +0.000431 | +0.000862 | 0.000862 |
| PATCH_CHROMA_MAX | 0.05 | -0.000821 | +0.000083 | 0.000862 |
| PATCH_UNIFORMITY_MAX | 0.05 | -0.000821 | +0.000083 | 0.000862 |
| WB_LOW_DESAT | 0.05 | -0.000009 | -0.000018 | 0.000018 |
| P_LOW | 0.5 | -0.000009 | -0.000017 | 0.000017 |
| WB_LOW_BAND_PCT[0] | 1 | +0.000007 | +0.000013 | 0.000013 |
| WB_LOW_BAND_PCT[1] | 2 | +0.000003 | +0.000007 | 0.000007 |
| WB_LOW_PRIOR[0] | 0.05 | +0.000000 | -0.000000 | 0.000000 |
| WB_LOW_PRIOR[1] | 0.05 | -0.000000 | -0.000000 | 0.000000 |
| WB_LOW_PRIOR[2] | 0.05 | -0.000000 | +0.000000 | 0.000000 |
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

- median_total: 0.0602
- median_luma: 0.0450
- median_color: 0.0232
- median_hi999: 0.0105
- max_clip: 0.0021
- frames_over_clip_budget: 0
- clip_budget: 0.0100
- n_frames: 150

## Per-roll

### 2506-1  (prep 43.04s)
```
{
  "n": 38,
  "median_total": 0.04695953605068618,
  "worst_frames": [
    {
      "stem": "DSC_0029",
      "total": 0.20132154772979557
    },
    {
      "stem": "DSC_0013",
      "total": 0.1443296857516044
    },
    {
      "stem": "DSC_0024",
      "total": 0.13493797962378154
    },
    {
      "stem": "DSC_0008",
      "total": 0.09963872845935766
    },
    {
      "stem": "DSC_0032",
      "total": 0.09586343455422884
    }
  ]
}
```
### 2510-11-1  (prep 46.15s)
```
{
  "n": 38,
  "median_total": 0.07375734105159695,
  "worst_frames": [
    {
      "stem": "DSC_0010",
      "total": 0.1528357857863721
    },
    {
      "stem": "DSC_0019",
      "total": 0.1500123122645739
    },
    {
      "stem": "DSC_0031",
      "total": 0.13864561230559078
    },
    {
      "stem": "DSC_0016",
      "total": 0.10633335747263806
    },
    {
      "stem": "DSC_0022",
      "total": 0.10626640481988836
    }
  ]
}
```
### 2511-12-1  (prep 45.92s)
```
{
  "n": 37,
  "median_total": 0.04912466643227419,
  "worst_frames": [
    {
      "stem": "DSC_0020",
      "total": 0.24125823955424633
    },
    {
      "stem": "DSC_0017",
      "total": 0.10251338503997214
    },
    {
      "stem": "DSC_0016",
      "total": 0.09621012701447029
    },
    {
      "stem": "DSC_0012",
      "total": 0.08939565644524201
    },
    {
      "stem": "DSC_0027",
      "total": 0.08857462020963547
    }
  ]
}
```
### 2512-2601-1  (prep 59.2s)
```
{
  "n": 37,
  "median_total": 0.06825021739543273,
  "worst_frames": [
    {
      "stem": "DSC_0008",
      "total": 0.12485444914975034
    },
    {
      "stem": "DSC_0002",
      "total": 0.12317291333239049
    },
    {
      "stem": "DSC_0015",
      "total": 0.09799128889750276
    },
    {
      "stem": "DSC_0031",
      "total": 0.09585164531171975
    },
    {
      "stem": "DSC_0013",
      "total": 0.09191855312203577
    }
  ]
}
```

## Worst frames (find where to look)

| roll | stem | total | luma | color | hi999 | clip |
|---|---|---|---|---|---|---|
| 2511-12-1 | DSC_0020 | 0.2413 | 0.1382 | 0.2245 | 0.1533 | 0.0000 |
| 2506-1 | DSC_0029 | 0.2013 | 0.2159 | 0.0289 | 0.1106 | 0.0002 |
| 2510-11-1 | DSC_0010 | 0.1528 | 0.1567 | 0.1111 | 0.0221 | 0.0021 |
| 2510-11-1 | DSC_0019 | 0.1500 | 0.1441 | 0.0453 | 0.0505 | 0.0000 |
| 2506-1 | DSC_0013 | 0.1443 | 0.1463 | 0.0349 | 0.0383 | 0.0000 |
| 2510-11-1 | DSC_0031 | 0.1386 | 0.1366 | 0.0378 | 0.0758 | 0.0000 |
| 2506-1 | DSC_0024 | 0.1349 | 0.1220 | 0.0200 | 0.0250 | 0.0000 |
| 2512-2601-1 | DSC_0008 | 0.1249 | 0.1376 | 0.0276 | 0.0035 | 0.0000 |
| 2512-2601-1 | DSC_0002 | 0.1232 | 0.0267 | 0.1111 | 0.0242 | 0.0017 |
| 2510-11-1 | DSC_0016 | 0.1063 | 0.0909 | 0.0352 | 0.0342 | 0.0000 |
| 2510-11-1 | DSC_0022 | 0.1063 | 0.0869 | 0.0285 | 0.0328 | 0.0000 |
| 2511-12-1 | DSC_0017 | 0.1025 | 0.0937 | 0.0136 | 0.0307 | 0.0000 |
| 2506-1 | DSC_0008 | 0.0996 | 0.0870 | 0.0246 | 0.0069 | 0.0000 |
| 2512-2601-1 | DSC_0015 | 0.0980 | 0.0887 | 0.0217 | 0.0457 | 0.0000 |
| 2511-12-1 | DSC_0016 | 0.0962 | 0.0893 | 0.0159 | 0.0287 | 0.0000 |
