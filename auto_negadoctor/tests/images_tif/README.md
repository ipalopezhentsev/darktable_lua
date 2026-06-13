# images_tif — detection / regression fixtures

The reference roll's analysis exports (32-bit float linear-Rec2020 TIFFs) live
here **locally** but are **NOT committed** — the full roll is ~1.2 GB at 2000px
and float photo data does not compress. They are `.gitignore`d
(`auto_negadoctor/tests/images_tif/*.tif`); only `exif_params.txt` is tracked.

The detection code and the crop annotations are **resolution-independent**
(size-dependent constants are fractions of the frame; annotation coords are
stored as normalized fractions), so any export width works — 1000px, 2000px, etc.

## Repopulating after a fresh checkout

1. In darktable, select the reference roll (the uninverted scans).
2. Run **AutoNegadoctor in-place (keep temp folder)** (or the Debug action).
3. Copy the resulting `*.tif` files **and** `exif_params.txt` from the temp
   folder (`%TEMP%/darktable_autonegadoctor_<timestamp>/`) into this directory.

Until then, `run_quality_tests.py` SKIPs with a message pointing here, and
`test_calibration.py` SKIPs (it uses the separate sRGB JPEG fixtures in
`tests/images/`).
