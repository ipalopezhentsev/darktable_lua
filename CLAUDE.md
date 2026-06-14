# CLAUDE.md

## What This Is

Darktable Lua plugins organized in three subdirectories:
- **`auto_crop/`** — automatic cropping of DSLR-scanned film frames. Detects film holder edges and applies crop parameters to remove them.
- **`auto_retouch/`** — automatic dust detection on DSLR-scanned film frames. Detects dust particles and applies healing brushes via darktable's retouch module. See `auto_retouch/dust_detection_spec.md` for the detection algorithm spec.
- **`auto_negadoctor/`** — automatic color-negative inversion. Finds the roll-wide film base (exposure-compensated across frames via EXIF), derives full negadoctor module parameters per frame from a single linear-Rec2020 TIFF export (Python re-implements negadoctor's forward model and picker formulas), writes them into the XMPs; plus a remove-negadoctor action for clean re-runs. An opt-in vision-LLM layer (`scene_tuner.py`, moondream via Ollama; spec 03) can nudge the final params per scene as an alternate variant. See `auto_negadoctor/specs/01_auto_negadoctor_spec.md` and `auto_negadoctor/CLAUDE.md`.

## Architecture

Two-component system per plugin: **Lua plugin** (runs inside darktable) calls a **Python script** (runs externally) for image analysis. Each plugin lives in its own subdirectory with its Lua + Python files together.

Shared Python code lives in **`common/`** — notably `common/debug_ui_base.py`, the tkinter annotation-viewer base class (image display, zoom/pan, navigation, sortable item table, annotation save/load lifecycle, report shell). Each feature's `debug_ui.py` subclasses it and implements only feature-specific hooks (overlays, hit-testing, annotation schema, report content). Subclasses: `auto_retouch/debug_ui.py` (`DustDebugUI`, serves both film-dust and sensor-dust sessions), `auto_crop/debug_ui.py` (`CropDebugUI`) and `auto_negadoctor/debug_ui.py` (`NegadoctorDebugUI`).

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
