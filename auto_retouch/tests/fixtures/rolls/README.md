# fixtures/rolls — per-roll calibration ground truth (auto_retouch)

Same layout as `auto_negadoctor/tests/fixtures/rolls/`. Each annotated roll has
ONE folder here, named by its roll id, holding **everything** for that roll:

```
fixtures/rolls/<roll_id>/
    DSC_0001.jpg ... DSC_00NN.jpg     (gitignored) the exported scans detection runs on
    <session>/                                     one or more annotation sessions
        DSC_0001_annotations.json     (tracked)    user corrections = the ground truth
        ...
```

The roll id is the folder name, so a stem that collides across rolls (every roll
has a `DSC_0013`) never cross-contaminates — `run_calibration.discover_rolls()`
hands each roll only its own fixtures (gathered with `rglob`, so a flat set of
`*_annotations.json` directly in the roll folder also works).

## The ground truth = the user's annotations

Calibration scores DETECTION (re-run per trial under a candidate preset) against
each frame's `{stem}_annotations.json`:

- `false_positives` — detections the user marked NOT-dust → a trial that still
  produces them is penalised (PRECISION; weighted highest).
- `missed_dust` / `missed_strokes` — defects the user wants found → a trial that
  leaves them uncovered is penalised (RECALL).

A frame whose annotations are all empty contributes no signal and is skipped, so a
session only re-detects the frames that actually constrain the fit.

## Images are NOT committed

The JPGs are regenerable (re-export the roll from darktable: AutoRetouch Debug /
keep-temp), so `*.jpg`/`*.jpeg` under `fixtures/rolls/` is `.gitignore`d — only the
annotations are tracked. Detection is resolution-independent (every size constant is
a fraction of the frame), so any export size works.

## Adding a new ground-truth roll

1. Export the roll's scans to JPG.
2. Annotate in the dust debug UI (mark false positives, add missed dust/strokes).
3. Drop the JPGs into `fixtures/rolls/<roll_id>/` and the `*_annotations.json` into
   a session subfolder (e.g. `correct-dust/`). Run `run_calibration.py --kind dust
   --method none` to confirm the roll is discovered and scored.

`baseline-2026-06/` is the adapted original `baseline_session` (its annotations are
empty — the baseline was approved as-is — so it carries no fit signal yet; it
establishes the layout).
