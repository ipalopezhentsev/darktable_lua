### Input specs for features are in `specs`

### Data Flow

1. `auto_retouch/auto_retouch.lua` exports selected images as full-resolution JPEGs to a temp folder (`%TEMP%/darktable_autoretouch_<timestamp>/`)
2. Lua reads each image's XMP sidecar to extract flip/crop transform params, writes `transform_params.txt`
3. Lua calls `auto_retouch/detect_dust.py` via `conda run -n autocrop` with file paths as arguments
4. Python detects bright dust spots using local background subtraction (OpenCV), generates darktable retouch module binary data (brush masks, group mask, retouch params, blendop params), writes `dust_results.txt`
5. Python applies inverse coordinate transforms (undo crop, undo flip) so mask coords are in darktable's original image space
6. Lua parses results, injects retouch history entry + mask entries into each source image's XMP sidecar
7. Lua calls `image:apply_sidecar(xmp_path)` to reload the modified XMP into darktable

### Canonical Data Principle (auto_retouch)

The spot dicts produced by `detect_spots()` are the **single source of truth** for all detected data (location, radius, etc.). Any algorithm change — whether to detection logic, brush sizing, coordinate transforms, whatever — must be reflected in the spot dict fields themselves. All four consumers read from the same dict and must stay in sync automatically:

1. **`debug_spots.json`** — serializes spot dicts wholesale; new fields appear for free
2. **`_dust_overlay.jpg`** — `save_visualization()` draws from spot dict fields
3. **Debug UI** — `debug_ui.py` draws circles and shows info from spot dict fields
4. **XMP output** — `generate_xmp_data_for_spots()` reads spot dict fields

**Consequence:** never put a tunable parameter only inside `generate_xmp_data_for_spots()` — the debug pipeline never calls it, so the change is invisible during testing. Compute everything meaningful at `spots.append()` time in `detect_spots()`.

Current key fields: `cx`, `cy`, `radius_px` (raw detected, for algorithm internals), `brush_radius_px` (scaled effective size, for display and XMP brush).

### Registered Actions

- **AutoRetouch_Debug** (`export_and_detect_dust_debug`) - export and detect dust spots only, saves visualization overlays.
- **AutoRetouch_InPlace** (`export_detect_and_apply_retouch_inplace`) - full pipeline: export, detect dust, apply heal retouch to source image's XMP.

## Known Bugs / TODOs

For auto_retouch feature:
- [x] Detection quality: the python script may detect image features instead of actual dust spots, and miss some real dust
- [ ] Add ability to heal sensor dust (larger, common between selected frames)
- [ ] Add ability to heal thread(fiber)-like dust
- [ ] Check all logic for consistency given different input sizes, i.e. does its constants contain relative metrics instead of absolute - absolute ones won't detect the same stuff on differently sized input, say if the same film frame were shot with a camera with higher megapixels.
- [ ] add params now hardcoded in py to DT UI, pass along with crops?
- [ ] I see it writes just one flip to transform_params, i.e. ignores that flip can be different in two axes
- [ ] on some heavily dusted images, e.g. DSC_0012, running second debug pass after applying first correction helps detect even more dust not picked up by the first pass. But currently, second application to xmp does not add new shapes but replaces previous ones. Consider adding second retouch instance. 
