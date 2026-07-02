# CLAUDE.md

## What This Is

Darktable Lua plugins organized in three subdirectories:
- **`auto_crop/`** — automatic cropping of DSLR-scanned film frames. Detects film holder edges and applies crop parameters to remove them.
- **`auto_retouch/`** — automatic dust detection on DSLR-scanned film frames. Detects dust particles and applies healing brushes via darktable's retouch module. See `auto_retouch/dust_detection_spec.md` for the detection algorithm spec.
- **`auto_negadoctor/`** — automatic color-negative inversion. Finds the roll-wide film base (exposure-compensated across frames via EXIF), derives full negadoctor module parameters per frame from a single linear-Rec2020 TIFF export (Python re-implements negadoctor's forward model and picker formulas), writes them into the XMPs; plus a remove-negadoctor action for clean re-runs. An opt-in vision-LLM layer (`scene_tuner.py`, moondream via Ollama; spec 03) can nudge the final params per scene as an alternate variant. See `auto_negadoctor/specs/01_auto_negadoctor_spec.md` and `auto_negadoctor/CLAUDE.md`.

## Architecture

Two-component system per plugin: **Lua plugin** (runs inside darktable) calls a **Python script** (runs externally) for image analysis. Each plugin lives in its own subdirectory with its Lua + Python files together.

Shared Python code lives in **`common/`** — notably `common/debug_ui_base.py`, the tkinter annotation-viewer base class (image display, zoom/pan, navigation, sortable item table, annotation save/load lifecycle, report shell, the centered canvas progress overlay, sRGB→monitor display color management + its `P` toggle, a generic **View / Navigate / Help menu-bar skeleton** every UI inherits, a shared **toolbar skeleton** with the common navigation + zoom buttons (◀ ▶ / － ＋ / Fit, via `build_toolbar` + the `toolbar_button` / `toolbar_separator` helpers, plus `make_readonly_combobox` — a readonly ttk.Combobox that stays readable when a value (e.g. a long preset name) overflows: entry sized to the longest value but capped, drop-down list widened past that cap, and a full-value hover tooltip; used by negadoctor's + retouch's preset selectors), a shared **RGB histogram + pipette panel** at the top-left (above the thumbnails — `_build_histogram_panel` / `_refresh_histogram` / the `<Motion>` pipette readout, fed by the `_display_rgb_array` / `_histogram_pixels` hooks; a feature may set `_clip_stats` for clip spikes), the shared **non-modal `_show_legend` / `_show_shortcuts` Help popups** driven by each UI's `_LEGEND_ENTRIES` / `_SHORTCUTS_TEXT` class attrs, and a shared **`--choose-dir` native folder picker** in `run_main` (`_prompt_session_dir` via `filedialog.askdirectory`; echoes `CHOSEN_DIR|<path>` to stdout) used by the negadoctor + retouch **apply-from-folder** actions to point the apply-mode UI at a saved ground-truth folder instead of a temp dir), and the shared **close/finish dialog** (2026-07-02): when a subclass sets `CLOSE_DIALOG = True`, `_on_close` pops a modal with two checkboxes — **apply annotations back to darktable** / **delete the temp folder** — writes the decision to `close_choices.txt` (`apply=0|1` + `delete_temp=0|1`) for the Lua caller to read after the (blocking) UI exits, and calls the `write_apply_results()` hook only when apply is chosen (each feature overrides it to write its results file: `applied_results.txt` / `dust_results.txt` / `crop_results.txt`); `CLOSE_APPLY_DEFAULT` / `CLOSE_DELETE_TEMP_DEFAULT` / `CLOSE_DELETE_TEMP_ENABLED` (False greys delete-temp for apply-from-folder) tune it, and `CLOSE_DIALOG_AUTOCONFIRM` skips the modal for the smoke tests. This is what lets each plugin expose ONE unified continuous-edit action (open the UI, decide apply/keep-temp on close) instead of the old Debug/InPlace/KeepTemp action trios. Each feature's `debug_ui.py` subclasses it and implements only feature-specific hooks (overlays, hit-testing, annotation schema, report content, and the chrome hooks `extend_view_menu` / `build_feature_menus` / `extend_help_menu` / `build_feature_toolbar` that add cascades + toolbar widgets on top of the shared skeletons). The three UIs are uniformly menu/toolbar-driven: negadoctor + retouch set `SHOW_BOTTOM_BUTTONS = False` (edit actions on feature menu cascades, not a lower-left button column); crop still keeps its small bottom button column. Subclasses: `auto_retouch/debug_ui.py` (`DustDebugUI`, serves both film-dust and sensor-dust sessions), `auto_crop/debug_ui.py` (`CropDebugUI`) and `auto_negadoctor/debug_ui.py` (`NegadoctorDebugUI`).

The same base/subclass pattern backs **calibration**: `common/calibration/` (`schema.TuningSchema` preset I/O + the immutable `Tuning` namedtuple; `registry.Registry` the fittable-constant catalog + per-trial cfg builder; `runner.run_main` the optimizers + recorded-session lifecycle + CLI) is the feature-agnostic engine. Both `auto_negadoctor` and `auto_retouch` externalize their tunable constants to `presets/*.json` (schema + docs in their `tuning.py`), enumerate a fittable catalog in `tests/calibration_registry.py`, and supply per-kind EVALUATORS in `tests/run_calibration.py` via a `runner.CalibrationAdapter`. Constants reach the analysis via an explicit `cfg` argument (no global mutation → parallel trials). See each feature's CLAUDE.md "Calibration" section.

On the **Lua** side, `common/dt_utils.lua` holds the stateless XMP/binary helpers that were copy-pasted across all three plugins (`generate_random_hex`, `generate_darktable_timestamp`, `find_max_history_num`, `float_to_le_hex` / `le_hex_to_float`, `write_source_paths` — the `stem|source-path` manifest, used by negadoctor + retouch). Each feature adds `common/` to `package.path` (`package.path = package.path .. ";" .. script_dir .. "../common/?.lua"`) and `local dtu = require("dt_utils")`, then aliases the names (`local generate_random_hex = dtu.generate_random_hex`) so call sites are unchanged. Feature-specific Lua (module-XML builders, the export/apply pipelines, action registration) stays in each plugin. Lua has no automated tests here — verify Lua changes by reloading in darktable.

## Python rules
When writing Python scripts, follow rules in @docs/python_rules.md

## Lua rules
When writing Lua scripts, follow rules in @docs/lua_rules.md

## Testing

1. Save Lua script
2. Reload in darktable via Lua console (toggle on/off)
3. Select images in lighttable, run via GUI action or shortcut
4. Check log: `%USERPROFILE%\Documents\Darktable\darktable-log.txt` (Windows)
5. Check exported files in the temp directory

## Dependencies

- darktable 4.x+ (API 7.0.0+)
- `lib/dtutils`, `lib/dtutils.file`, `lib/dtutils.system`, `lib/dtutils.log`, `lib/dtutils.debug` (darktable utility libs, not in this repo)
- Python 3.11 via conda `autocrop` environment

## General TODO

(auto negadoctor implemented — remaining feature TODOs live in `auto_negadoctor/CLAUDE.md`)
