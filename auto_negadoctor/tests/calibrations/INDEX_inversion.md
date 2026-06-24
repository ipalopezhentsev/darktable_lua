# Calibration history — inversion

Comparable rows (one metric per kind; see calibrations/README.md).

| session | comment | rolls | method | objective | headline | wall_s | commit |
|---|---|---|---|---|---|---|---|
| [2026-06-22_052428_inversion_01](2026-06-22_052428_inversion_01/report.md) | | 2506-1 | coordinate_descent(epsilon=0.0005, max_iters=6, init_step=per-param/auto, step_min=per-param/auto, step_shrink=0.5) | 0.0292 | EMD 0.0292 clip 0 | 12611.44 | ad035a1 |
| [2026-06-23_052834_inversion_02](2026-06-23_052834_inversion_02/report.md) | | 2506-1,2510-11-1 | coordinate_descent(epsilon=0.0005, max_iters=6, init_step=per-param/auto, step_min=per-param/auto, step_shrink=0.5) | 0.0375 | EMD 0.0375 clip 0 | 61614.85 | ad035a1 |
| [2026-06-23_082423_inversion_03](2026-06-23_082423_inversion_03/report.md) | | 2506-1,2510-11-1 | none | 0.0556 | EMD 0.0556 clip 0 | 31949.79 | 0a5f65f |
| [2026-06-24_090135_inversion_04](2026-06-24_090135_inversion_04/report.md) | test running inversion on less params after pca run detected 13 non important params | 2506-1,2510-11-1 | coordinate_descent(epsilon=0.0005, max_iters=4, init_step=per-param/auto, step_min=per-param/auto, step_shrink=0.5) | 0.0373 | EMD 0.0373 clip 0 | 36779.67 | 9b6ca96 |
