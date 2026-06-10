### Input specs for features are in `specs`

### Data Flow

1. `auto_retouch/auto_retouch.lua` exports selected images as full-resolution JPEGs to a temp folder (`%TEMP%/darktable_autoretouch_<timestamp>/`)
2. Lua reads each image's XMP sidecar to extract flip/crop transform params, writes `transform_params.txt`
3. Lua calls `auto_retouch/detect_dust.py` via `conda run -n autocrop` with file paths as arguments
4. Python detects bright dust spots using local background subtraction (OpenCV), generates darktable retouch module binary data (brush masks, group mask, retouch params, blendop params), writes `dust_results.txt`
5. Python applies inverse coordinate transforms (undo crop, undo flip, undo ashift when the
 "rotate and perspective" module is active) so mask coords are in darktable's raw image space.
 NOTE: lens correction is not yet inverted (small constant offset, present on all frames).
6. Lua parses results, injects retouch history entry + mask entries into each source image's XMP sidecar
7. Lua calls `image:apply_sidecar(xmp_path)` to reload the modified XMP into darktable

### Testing approach

The `tests` folder holds a saved baseline of 'reasonably good detection' plus checks that run
against it. **Two obligations, not one** — running the suite is necessary but not sufficient:

1. **RUN it after any change to `detect_dust.py`.** `conda run -n autocrop python
   tests/run_quality_tests.py` — confirm no large regression vs the baseline. Also run the
   focused tests: `tests/test_source_invariants.py`, `tests/test_ashift_transform.py`.

2. **EXTEND it whenever you add a feature.** The suite only checks what someone taught it to
   check — a passing run on an un-extended suite is a false sense of safety. This is exactly how
   stroke healing shipped unverified: `run_quality_tests` only diffed *dots* against the baseline,
   so stroke sources/paths/radii were invisible to it. When you add anything, ask "what new data
   or property did I just create, and what here would catch it regressing?" and add that:
   - **New detected data** (new spot kind / field): make sure the baseline diff covers it, OR add
     a check. Regenerate the baseline (`generate_baseline.py`) ONLY after the user approves the
     new detections — the baseline is ground truth, don't bake in unreviewed output.
   - **New correctness property / invariant** (e.g. "healing source must not overlap the defect",
     "a coordinate transform must round-trip"): add it to `check_spot_invariants()` in
     run_quality_tests, or a dedicated `tests/test_*.py`. Invariants are better than baseline diffs
     where possible — they need no baseline and state the actual requirement. Give a non-trivial
     check its own self-test (a "test for the test") so it can't silently degrade into a no-op.

What the suite currently covers: dot match/missing/FP + source/radius mismatch vs baseline;
stroke match/missing/new counts vs baseline; per-spot source-clearance + geometry invariants
(`check_spot_invariants`, baseline-independent); ashift inverse transform (round-trip + bbox).

### Canonical Data Principle (auto_retouch)

The spot dicts produced by `detect_spots()` are the **single source of truth** for all detected data (location, radius, etc.). Any algorithm change — whether to detection logic, brush sizing, coordinate transforms, whatever — must be reflected in the spot dict fields themselves. All four consumers read from the same dict and must stay in sync automatically:

1. **`{stem}_debug_spots.json`** — serializes spot dicts wholesale per image; new fields appear for free
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
- [x] Add ability to heal thread(fiber)-like dust — stroke detection (spec 06): elongated
      bright threads detected from the threshold binary, healed with multi-node brush strokes
      (`kind="stroke"` spots). Faint film-scratch ridge pass implemented but disabled by
      default (noise-floor FPs); human adds faint scratches via the debug UI.
- [ ] Check all logic for consistency given different input sizes, i.e. does its constants contain relative metrics instead of absolute - absolute ones won't detect the same stuff on differently sized input, say if the same film frame were shot with a camera with higher megapixels.
- [ ] add params now hardcoded in py to DT UI, pass along with crops?
- [ ] on some heavily dusted images, e.g. DSC_0012, running second debug pass after applying first correction helps detect even more dust not picked up by the first pass.
- [ ] add full negadoctor automation
- [ ] generalize debug ui - viewer is common part, other features are like modules to it, add other detectors to UI, not only retouching
- [ ] speed up retouching (GPU?)