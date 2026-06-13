# images_tif — detection / regression fixtures

The reference roll's analysis exports (32-bit float linear-Rec2020 TIFFs) live
here **locally** but are **NOT committed** — the full roll is ~1.2 GB at 2000px
and float photo data does not compress. They are `.gitignore`d
(`auto_negadoctor/tests/images_tif/*.tif`); only `exif_params.txt` is tracked.

The detection code and the crop annotations are **resolution-independent**
(size-dependent constants are fractions of the frame; annotation coords are
stored as normalized fractions), so any export width works — 1000px, 2000px, etc.

## Repopulating after a fresh checkout

1. In darktable, select the roll (the uninverted scans).
2. Run **AutoNegadoctor in-place (keep temp folder)** (or the Debug action).
3. Copy the resulting `*.tif` files **and** `exif_params.txt` from the temp
   folder (`%TEMP%/darktable_autonegadoctor_<timestamp>/`) into this directory.
4. Set `roll.txt` here to the roll's id (e.g. `2512-2601-1` for the reference
   roll, `2510-11-1` for the second roll). This MUST match — annotation stems
   collide across rolls (every roll has a `DSC_0013`), so
   `run_quality_tests.py` only checks the `fixtures/annotations/<session>/`
   folders whose own `roll.txt` matches this one. Loading roll A's TIFFs while
   `roll.txt` still names roll B would compare B's hand-drawn crops against A's
   images.

Until then, `run_quality_tests.py` SKIPs with a message pointing here, and
`test_calibration.py` SKIPs (it uses the separate sRGB JPEG fixtures in
`tests/images/`).
