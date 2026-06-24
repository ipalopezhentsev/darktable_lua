# Calibration session — crop

- created: 2026-06-23T20:36:11
- git commit: 9b6ca96
- rolls: 2506-1, 2510-11-1, 2511-12-1, 2512-2601-1
- metric: crop_overtrim_area_fraction (containment HARD)
- fit method: none
- fitted params: a, l, l
- trial count: 1
- wall time: 432.24s (prep 151.96s)
- objective: 40000001.5915 (init) -> 40000001.5915 (final)

## Convergence

- method: none  trials: 1
- best-so-far improved 1 time(s); curve (trial: objective):
  - 1: 40000001.5915

## Principal components (curvature of the metric at the optimum)

Eigen-decomposition of the finite-difference Hessian in grid-step-normalized units (190 extra evals). Each component is a DIRECTION in param space (loadings show the coupled params); larger |eigenvalue| = the metric changes more sharply along it.

- **PC1** eigenvalue 18627997.4546 (42% of total curvature): -0.94·CROP_SHADOW_CORE_FRAC  +0.29·CROP_SHADOW_REL  -0.10·HOLDER_LUMA_THR  -0.09·CROP_PAD_FRAC  -0.07·CROP_GAP_TOL_FRAC  -0.06·CROP_SHADOW_MAX_FRAC
- **PC2** eigenvalue 9702429.4483 (22% of total curvature): +0.95·CROP_PAD_FRAC  +0.22·CROP_SHADOW_REL  +0.21·CROP_SHADOW_MAX_FRAC  +0.06·HOLDER_LUMA_THR
- **PC3** eigenvalue 5379183.5118 (12% of total curvature): +0.85·CROP_GAP_TOL_FRAC  +0.40·CROP_SHADOW_REL  -0.23·HOLDER_LUMA_THR  -0.21·CROP_REBATE_MARGIN_D  +0.11·CROP_SHADOW_CORE_FRAC  -0.07·CROP_PAD_FRAC
- **PC4** eigenvalue 4442608.6943 (10% of total curvature): +0.67·CROP_SHADOW_REL  -0.48·CROP_GAP_TOL_FRAC  -0.47·HOLDER_LUMA_THR  +0.29·CROP_SHADOW_CORE_FRAC  -0.12·CROP_PAD_FRAC
- **PC5** eigenvalue -1488848.5530 (3% of total curvature): +0.89·CROP_REBATE_LINE_FRAC  -0.44·CROP_SHADOW_MAX_FRAC  +0.09·CROP_PAD_FRAC
- **PC6** eigenvalue 1128846.9799 (3% of total curvature): +0.73·CROP_SHADOW_MAX_FRAC  -0.47·HOLDER_LUMA_THR  +0.34·CROP_REBATE_LINE_FRAC  -0.29·CROP_SHADOW_REL  +0.12·CROP_REBATE_MARGIN_D  -0.10·CROP_SHADOW_CORE_FRAC
- **PC7** eigenvalue 999999.9967 (2% of total curvature): +1.00·CROP_REBATE_BAND_HUE_TOL
- **PC8** eigenvalue 999999.9344 (2% of total curvature): +1.00·CROP_LEAK_MARGIN_D

### Per-param sensitivity (freeze signal)

Central-difference gradient + diagonal curvature in grid-step units, and the largest objective change from a single ±grid_step move. Sorted loudest-first — params near the bottom (max Δobj ≈ 0) are flat and can be FROZEN.

| param | grid_step | gradient | curvature | max Δobj |
|---|---|---|---|---|
| CROP_PAD_FRAC | 0.002 | -30499999.081058 | +9000000.774363 | 34999999.468240 |
| CROP_SHADOW_CORE_FRAC | 0.004 | -12499999.644606 | +17000000.345014 | 20999999.817114 |
| CROP_GAP_TOL_FRAC | 0.005 | -1499999.964726 | +4999999.931610 | 3999999.930531 |
| CROP_SHADOW_MAX_FRAC | 0.005 | -3499999.876525 | +1000000.098027 | 3999999.925538 |
| CROP_SHADOW_REL | 0.05 | -1499999.900632 | +4999999.994383 | 3999999.897824 |
| HOLDER_LUMA_THR | 0.005 | -1999999.982601 | +2000000.002299 | 2999999.983751 |
| CROP_REBATE_MARGIN_D | 0.01 | +1000000.042167 | +0.016869 | 1000000.050601 |
| CROP_REBATE_BAND_HUE_TOL | 0.001 | -499999.998362 | +999999.996725 | 999999.996725 |
| CROP_REBATE_LINE_FRAC | 0.02 | -500000.003580 | -999999.922283 | 999999.964722 |
| CROP_LEAK_MARGIN_D | 0.01 | +499999.967179 | +999999.934357 | 999999.934357 |
| CROP_JUNK_LINE_FRAC | 0.01 | -0.043350 | -0.037148 | 0.061924 |
| CROP_REBATE_MAX_FRAC | 0.005 | +0.008890 | +0.017780 | 0.017780 |
| BORDER_MAX_FRAC | 0.01 | +0.004694 | +0.009388 | 0.009388 |
| CROP_REBATE_HUE_TOL | 0.005 | +0.002709 | +0.005418 | 0.005418 |
| CROP_REBATE_WIDE_LINE_FRAC | 0.05 | -0.002002 | +0.004004 | 0.004004 |
| CROP_REBATE_TERM_FRAC | 0.005 | +0.000000 | +0.000000 | 0.000000 |
| CROP_REBATE_WIDE_MAX_D | 0.025 | +0.000000 | +0.000000 | 0.000000 |
| CROP_REBATE_WIDE_FRAC | 0.05 | +0.000000 | +0.000000 | 0.000000 |

## Fitted constants (record-only — adopt by hand)

| constant | init | fitted |
|---|---|---|
| CROP_JUNK_LINE_FRAC | 0.0800 | 0.0800 |
| CROP_LEAK_MARGIN_D | 0.0300 | 0.0300 |
| CROP_REBATE_MARGIN_D | 0.1456 | 0.1456 |
| CROP_REBATE_LINE_FRAC | 0.1827 | 0.1827 |
| CROP_REBATE_TERM_FRAC | 0.0400 | 0.0400 |
| CROP_REBATE_MAX_FRAC | 0.0269 | 0.0269 |
| CROP_REBATE_HUE_TOL | 0.0200 | 0.0200 |
| CROP_REBATE_WIDE_MAX_D | 0.3871 | 0.3871 |
| CROP_REBATE_WIDE_LINE_FRAC | 0.8000 | 0.8000 |
| CROP_REBATE_BAND_HUE_TOL | 0.0094 | 0.0094 |
| CROP_REBATE_WIDE_FRAC | 0.5500 | 0.5500 |
| CROP_PAD_FRAC | 0.0050 | 0.0050 |
| CROP_SHADOW_REL | 0.7375 | 0.7375 |
| CROP_SHADOW_MAX_FRAC | 0.0309 | 0.0309 |
| CROP_SHADOW_CORE_FRAC | 0.0167 | 0.0167 |
| CROP_GAP_TOL_FRAC | 0.0153 | 0.0153 |
| HOLDER_LUMA_THR | 0.0241 | 0.0241 |
| BORDER_MAX_FRAC | 0.0800 | 0.0800 |

## Aggregate (all rolls)

- total_overtrim_area: 1.5915
- median_overtrim_area: 0.0088
- max_overtrim_area: 0.0848
- total_undertrim_area: 0.1449
- max_undertrim_area: 0.0123
- containment_violations: 40
- containment_weight: None
- n_frames: 150

## Per-roll

### 2506-1  (prep 24.33s)
```
{
  "n": 38,
  "median_overtrim_area": 0.004387059214903526,
  "worst_frames": [
    {
      "stem": "DSC_0020",
      "overtrim_area": 0.05093866321411231
    },
    {
      "stem": "DSC_0022",
      "overtrim_area": 0.019281257305209402
    },
    {
      "stem": "DSC_0011",
      "overtrim_area": 0.016393264522007036
    },
    {
      "stem": "DSC_0004",
      "overtrim_area": 0.009645498792205378
    },
    {
      "stem": "DSC_0031",
      "overtrim_area": 0.009431587275898653
    }
  ]
}
```
### 2510-11-1  (prep 29.64s)
```
{
  "n": 38,
  "median_overtrim_area": 0.011241406076735419,
  "worst_frames": [
    {
      "stem": "DSC_0014",
      "overtrim_area": 0.02984421547295799
    },
    {
      "stem": "DSC_0003",
      "overtrim_area": 0.02814266362170554
    },
    {
      "stem": "DSC_0032",
      "overtrim_area": 0.02565402228575881
    },
    {
      "stem": "DSC_0018",
      "overtrim_area": 0.02493736251221281
    },
    {
      "stem": "DSC_0040",
      "overtrim_area": 0.023867055678432923
    }
  ]
}
```
### 2511-12-1  (prep 46.11s)
```
{
  "n": 37,
  "median_overtrim_area": 0.009719300138461815,
  "worst_frames": [
    {
      "stem": "DSC_0033",
      "overtrim_area": 0.08483033932135728
    },
    {
      "stem": "DSC_0014",
      "overtrim_area": 0.02129674584764405
    },
    {
      "stem": "DSC_0038",
      "overtrim_area": 0.01736841632051213
    },
    {
      "stem": "DSC_0027",
      "overtrim_area": 0.017030128931326535
    },
    {
      "stem": "DSC_0006",
      "overtrim_area": 0.016704578830327332
    }
  ]
}
```
### 2512-2601-1  (prep 51.88s)
```
{
  "n": 37,
  "median_overtrim_area": 0.00898802994611378,
  "worst_frames": [
    {
      "stem": "DSC_0024",
      "overtrim_area": 0.029383050715386046
    },
    {
      "stem": "DSC_0019",
      "overtrim_area": 0.02392811973650297
    },
    {
      "stem": "DSC_0011",
      "overtrim_area": 0.023604442766119414
    },
    {
      "stem": "DSC_0035",
      "overtrim_area": 0.018639148130166094
    },
    {
      "stem": "DSC_0012",
      "overtrim_area": 0.016227679775583967
    }
  ]
}
```

## Worst frames (find where to look)

| contained | overtrim_area | undertrim_area | overtrim_edges | roll | stem |
|---|---|---|---|---|---|
| False | 0.0848 | 0.0068 | {'left': -0.0025025025025025025, 'top': 0.0, 'right': -0.0055055055055055055, 'bottom': 0.08982035928143713} | 2511-12-1 | DSC_0033 |
| False | 0.0509 | 0.0056 | {'left': -0.0045045045045045045, 'top': 0.0007485029940119761, 'right': 0.053553553553553554, 'bottom': -0.0014970059880239522} | 2506-1 | DSC_0020 |
| True | 0.0298 | 0.0000 | {'left': 0.0025025025025025025, 'top': 0.025449101796407185, 'right': 0.0005005005005005005, 'bottom': 0.0029940119760479044} | 2510-11-1 | DSC_0014 |
| True | 0.0294 | 0.0000 | {'left': 0.021021021021021023, 'top': 0.004491017964071856, 'right': 0.002002002002002002, 'bottom': 0.0037425149700598802} | 2512-2601-1 | DSC_0024 |
| True | 0.0281 | 0.0000 | {'left': 0.001001001001001001, 'top': 0.02470059880239521, 'right': 0.001001001001001001, 'bottom': 0.0029940119760479044} | 2510-11-1 | DSC_0003 |
| True | 0.0257 | 0.0000 | {'left': 0.014014014014014014, 'top': 0.006736526946107785, 'right': 0.002002002002002002, 'bottom': 0.004491017964071856} | 2510-11-1 | DSC_0032 |
| True | 0.0249 | 0.0000 | {'left': 0.003003003003003003, 'top': 0.01122754491017964, 'right': 0.00850850850850851, 'bottom': 0.0037425149700598802} | 2510-11-1 | DSC_0018 |
| True | 0.0239 | 0.0000 | {'left': 0.0015015015015015015, 'top': 0.0029940119760479044, 'right': 0.018018018018018018, 'bottom': 0.0029940119760479044} | 2512-2601-1 | DSC_0019 |
| True | 0.0239 | 0.0000 | {'left': 0.001001001001001001, 'top': 0.006736526946107785, 'right': 0.016516516516516516, 'bottom': 0.0014970059880239522} | 2510-11-1 | DSC_0040 |
| True | 0.0236 | 0.0000 | {'left': 0.003003003003003003, 'top': 0.0187125748502994, 'right': 0.0025025025025025025, 'bottom': 0.0007485029940119761} | 2512-2601-1 | DSC_0011 |
| True | 0.0221 | 0.0000 | {'left': 0.002002002002002002, 'top': 0.006736526946107785, 'right': 0.002002002002002002, 'bottom': 0.012724550898203593} | 2510-11-1 | DSC_0017 |
| True | 0.0213 | 0.0000 | {'left': 0.0015015015015015015, 'top': 0.017964071856287425, 'right': 0.0015015015015015015, 'bottom': 0.0014970059880239522} | 2511-12-1 | DSC_0014 |
| True | 0.0193 | 0.0000 | {'left': 0.0005005005005005005, 'top': 0.004491017964071856, 'right': 0.015515515515515516, 'bottom': 0.0} | 2506-1 | DSC_0022 |
| True | 0.0186 | 0.0000 | {'left': 0.0015015015015015015, 'top': 0.004491017964071856, 'right': 0.0025025025025025025, 'bottom': 0.01122754491017964} | 2512-2601-1 | DSC_0035 |
| False | 0.0174 | 0.0014 | {'left': 0.012512512512512513, 'top': 0.0029940119760479044, 'right': -0.0015015015015015015, 'bottom': 0.0029940119760479044} | 2511-12-1 | DSC_0038 |
