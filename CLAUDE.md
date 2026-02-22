# CLAUDE.md

## What This Is

Darktable Lua plugins organized in two subdirectories:
- **`auto_crop/`** — automatic cropping of DSLR-scanned film frames. Detects film holder edges and applies crop parameters to remove them.
- **`auto_retouch/`** — automatic dust detection on DSLR-scanned film frames. Detects dust particles and applies healing brushes via darktable's retouch module. See `auto_retouch/dust_detection_spec.md` for the detection algorithm spec.

## Architecture

Two-component system per plugin: **Lua plugin** (runs inside darktable) calls a **Python script** (runs externally) for image analysis. Each plugin lives in its own subdirectory with its Lua + Python files together.

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

