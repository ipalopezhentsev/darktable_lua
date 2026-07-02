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

### Debug UI

`auto_crop.py --debug-ui <images...>` writes per-image `{stem}_debug_crop.json` (crop %, per-edge confidence, detected_region, constants) and opens `auto_crop/debug_ui.py` (`CropDebugUI`, subclass of `common/debug_ui_base.py`). The user marks wrong edges: select an edge (click its line or keys 1/2/3/4), then Ctrl+Click the correct position (scroll = 1 px nudge, Shift = 10 px; C clears). Corrections auto-save to `{stem}_annotations.json`; closing writes `debug_report.txt` with detected vs corrected per edge (% and px) for tuning. A correction IS the wrong-edge marker. The annotation JSONs are designed to seed the future baseline/tests (see TODO below).

UI chrome (2026-06-24): the marker legend + the full mouse/key reference moved OFF the left panel onto the **Help menu** (`Marker legend…` popup from `_LEGEND_ENTRIES`, `Mouse & keyboard shortcuts…` dialog), matching auto_negadoctor / auto_retouch. The View / Navigate / Help menu bar, the top toolbar's common ◀ ▶ / － ＋ / Fit navigation+zoom buttons, and the `P` display-colour-management toggle all come from the shared base skeleton in `common/debug_ui_base.py` (crop adds no toolbar widgets of its own and keeps its small bottom button column).

### Registered Actions

ONE unified continuous-edit action (2026-07-02; the old Debug / InPlace /
InPlace_KeepTemp trio collapsed into it — the "debug vs apply vs keep-temp"
choice is now the close dialog):

- **AutoCrop** (`edit_crop`) — export + detect + open the crop debug UI
  **BLOCKING** (`export_and_detect(images, true, true)` → `auto_crop.py --debug-ui`
  runs the detect, then launches `debug_ui.py --apply` foreground). On close the
  shared finish dialog (see root CLAUDE.md / `common/debug_ui_base.py`) asks (1)
  apply the crop in-place, (2) delete the temp folder. `CropDebugUI.write_apply_results`
  (re)writes `crop_results.txt` (per-edge `corrected_pct` where the user marked an
  edge, else the detected `crop%`; same `OK|stem|L=..|T=..|R=..|B=..` format the
  Lua `parse_crop_results` already reads); Lua then reads `close_choices.txt` and,
  if apply, runs the existing `apply_crop_in_place` loop (updates
  `darktable:change_timestamp` + `history_current_hash` to force preview
  regeneration), and deletes the temp dir if chosen.

## Known Bugs / TODOs

For auto_crop feature:
- [ ] Check all logic for consistency given different input sizes, i.e. does its constants contain relative metrics instead of absolute - absolute ones won't detect the same stuff on differently sized input, say if the same film frame were shot with a camera with higher megapixels.
- [ ] Introduce parallelizm/progress reporting in auto_crop like in auto_retouch
- [ ] in auto_crop.lua, implement the same bugfix as in auto_retouch where it erroneously could delete previous history if there were some disabled steps in it. (check commit 7b444a3f77be05cf064097cef4e4eb4823ce7664)
- [ ] Create spec/current algo description like for auto_retouch feature
- [ ] create baseline and tests - any improvement should base on them and not deviate
- [ ] showing diff vs baseline in debug UI
- [ ] fine tune via debug ui marking like I did with threaded dust