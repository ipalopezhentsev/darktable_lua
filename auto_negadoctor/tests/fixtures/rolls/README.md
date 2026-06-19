# fixtures/rolls — per-roll annotation + image fixtures

Each annotated film roll has ONE folder here, named by its roll id
(e.g. `2512-2601-1/`, `2510-11-1/`). The folder holds **everything** for that
roll together:

```
fixtures/rolls/<roll_id>/
    exif_params.txt                 (tracked)  per-frame EXIF for exposure comp
    scene_labels.json               (tracked)  frozen LLM scene labels (AI gate)
    DSC_0001.tif ... DSC_00NN.tif   (gitignored) linear-Rec2020 float exports
    <session>/                                 one or more annotation sessions
        DSC_0001_annotations.json   (tracked)
        ...
```

The roll id is the folder name, so there is **no `roll.txt`** anymore — a stem
that collides across rolls (every roll has a `DSC_0013`) never cross-contaminates
because `run_quality_tests.discover_rolls()` hands each roll only its own
fixtures. Annotation sessions may be dated subfolders (multiple per stem, e.g.
`2026-06-11_taste/`, `2026-06-13_wb_print_roll/`) or a flat set of
`*_annotations.json` directly in the roll folder — both are gathered with
`rglob`.

## Images are NOT committed

The linear-Rec2020 float TIFFs are heavy (~1.2 GB/roll at 2000px; float photo
data does not compress) and are **regenerable from the source raws**, so
`*.tif` under `fixtures/rolls/` is `.gitignore`d. Only the annotations,
`exif_params.txt`, and `scene_labels.json` are tracked.

Detection is **resolution-independent** (every size-dependent constant is a
fraction of the frame; annotation coords are normalized fractions), so any
export width works — 1000px, 2000px, etc.

## Repopulating a roll's images after a fresh checkout

1. In darktable, select the roll (the uninverted scans).
2. Run **AutoNegadoctor in-place (keep temp folder)** (or the Debug action).
3. Copy the resulting `*.tif` files **and** `exif_params.txt` from the temp
   folder (`%TEMP%/darktable_autonegadoctor_<timestamp>/`) into the roll's
   folder here (`fixtures/rolls/<roll_id>/`).

Until a roll has local TIFFs, `run_quality_tests.py` simply skips it (it only
runs rolls that have images). With no rolls populated at all, the suite SKIPs
with a message pointing here, and `calibrate_scene_tuning.py` SKIPs.

## Adding a new annotated roll

Make a new `fixtures/rolls/<roll_id>/` folder, drop the roll's
`*_annotations.json` (optionally under a dated session subfolder) and
`exif_params.txt` in it, and repopulate its TIFFs as above. The regression /
tuning utilities pick it up automatically — they iterate over every roll folder
that has local images.

## These annotations are the calibration ground truth

The annotations here are the fixed ground truth that the **recorded calibration
sessions** (`tests/calibrations/`, spec 05) tune the algorithm toward. Run them
with `tests/run_calibration.py`; each session records its inputs, its
algorithm-independent closeness metric, and its results in a dated folder. See
[../../calibrations/README.md](../../calibrations/README.md). All three
calibration kinds (crop, vignette, inversion) derive what they need from the
roll's local TIFFs — there are no extra hand-produced fixture files.
