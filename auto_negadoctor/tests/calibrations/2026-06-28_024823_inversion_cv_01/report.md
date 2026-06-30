# Cross-validation (leave-one-roll-out) — inversion

- comment: Leave-one-roll-out CROSS-VALIDATION of a re-tune of the print-tune tone levers (PRINT_GAMMA / PRINT_HI_CEIL / PRINT_HI_PCT / PRINT_CLIP_BUDGET) -- the constants spec 03/04 found carry the SYSTEMATIC brightness/contrast bias vs the GT (exposure pinned too high, gamma off). These are all FAST-path (re-tune on cached frames, no whole-pipeline re-run), so a 4-fold CV is tractable without downsampling. Each fold fits CMA-ES on the OTHER rolls and scores on the held-out roll vs the current preset -> VERDICT (ADOPT / DO NOT ADOPT) + per-constant cross-fold stability. The final all-rolls fit is OFF here (cv_final_fit:false) so this is a VERDICT-FIRST run; re-run with --cv-final-fit (or set it true) to also produce the adoptable fitted_preset.json -- adopt only if the verdict is ADOPT. To extend the re-tune to the wb/offset constants (WB_*_DESAT, OFFSET_DEFAULT, the wb_high family) add them to params, but those are FULL-path (whole pipeline per trial) so add --downsample 2 to keep the K-fold cost reasonable.
- created: 2026-06-27T19:27:57
- git commit: 91b1d9e
- rolls (4 folds): 2506-1, 2510-11-1, 2511-12-1, 2512-2601-1
- metric: histogram_emd_total (render vs GT-render, picture-vs-picture) (lower = better)
- inner fit: cmaes(sigma=0.3, popsize=12, max_iters=30, seed=1, workers=1)
- fitted params: PRINT_GAMMA, PRINT_HI_CEIL, PRINT_HI_PCT, PRINT_CLIP_BUDGET
- downsample (calibration-only): 4x
- wall time: 26424.39s

## Verdict — does the re-tune generalise?

Mean held-out objective over 4 folds — each roll scored by constants fitted on the OTHER rolls only (it never saw itself):

- current preset (baseline): **0.0652**
- re-tuned (candidate):      **0.0812**
- delta: **0.0160** (REGRESSION)

### DO NOT ADOPT

The re-tune does NOT beat the current preset on held-out rolls — it is overfitting the calibration rolls. **Keep the current preset.**

## Per fold

| held-out roll | baseline | candidate | delta | train objective | trials |
|---|---|---|---|---|---|
| 2506-1 | 0.0419 | 0.0663 | 0.0244 | 0.0607 | 288 |
| 2510-11-1 | 0.0677 | 0.0647 | -0.0030 | 0.0533 | 360 |
| 2511-12-1 | 0.0636 | 0.0710 | 0.0074 | 0.0551 | 360 |
| 2512-2601-1 | 0.0876 | 0.1228 | 0.0352 | 0.0458 | 360 |

## Per-constant stability across folds

How much each fitted constant moves depending on which roll is held out. SMALL spread = a universal constant (safe to bake into the preset); LARGE spread = roll-specific taste (belongs in a per-frame / scene layer, not the global preset).

| constant | init | fold min | fold max | spread (% of range) | verdict |
|---|---|---|---|---|---|
| PRINT_GAMMA | 4.5312 | 4.0660 | 4.6795 | 25% | VARIABLE |
| PRINT_HI_CEIL | 0.9693 | 0.8813 | 0.9900 | 57% | VARIABLE |
| PRINT_HI_PCT | 99.9155 | 99.5347 | 99.9900 | 46% | VARIABLE |
| PRINT_CLIP_BUDGET | 0.0030 | 0.0000 | 0.0113 | 57% | VARIABLE |

## Final fit (all rolls)

Skipped — the final all-rolls fit is OFF by default. This run measured generalisation only (verdict + stability above). Re-run with `--cv-final-fit` to also produce the adoptable `fitted_preset.json`.

