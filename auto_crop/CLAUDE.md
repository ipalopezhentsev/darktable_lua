### Input specs for features are in `specs`

### Data Flow

1. `auto_crop/auto_crop.lua` exports selected images as downscaled JPEGs to a temp folder (`%TEMP%/darktable_autocrop_<timestamp>/`)
2. Lua calls `auto_crop/auto_crop.py` via `conda run -n autocrop` with file paths as arguments
3. Python detects margins using brightness profile analysis (OpenCV), writes results to `crop_results.txt` in a simple line format (`OK|filename|L=x|T=x|R=x|B=x`)
4. Lua parses results, modifies each source image's XMP sidecar to inject a crop history entry with the detected parameters, and updates `change_timestamp`/`history_current_hash` to force preview regeneration
5. Lua calls `image:apply_sidecar(xmp_path)` to reload the modified XMP into darktable

### Key Design Decisions

- Crop params are written directly as binary hex into XMP (`darktable:params` field) - 4 little-endian floats for L/T/R/B + 8 zero bytes
- Python outputs crop as percentages from each edge; Lua converts to fractions where R/B are edge positions (1 - margin)
- `darktable:change_timestamp` (microseconds since 0001-01-01) and `darktable:history_current_hash` (random hex) are updated in XMP to force darktable to regenerate previews

### Registered Actions

- **AutoCrop_Debug** (`export_and_find_edges_debug`) - export and detect only, no crop application. For testing edge detection.
- **AutoCrop_InPlace** (`export_detect_and_apply_inplace`) - full pipeline: export, detect, apply crop directly to source image's XMP (no virtual copies). Updates `darktable:change_timestamp` and `darktable:history_current_hash` to force preview regeneration.

## Known Bugs / TODOs

For auto_crop feature:
- [ ] Check all logic for consistency given different input sizes, i.e. does its constants contain relative metrics instead of absolute - absolute ones won't detect the same stuff on differently sized input, say if the same film frame were shot with a camera with higher megapixels.
- [ ] Introduce parallelizm/progress reporting in auto_crop like in auto_retouch
- [ ] in auto_crop.lua, implement the same bugfix as in auto_retouch where it erroneously could delete previous history if there were some disabled steps in it. (check commit 7b444a3f77be05cf064097cef4e4eb4823ce7664)
- [ ] Create spec/current algo description like for auto_retouch feature
- [ ] create baseline and tests - any improvement should base on them and not deviate
