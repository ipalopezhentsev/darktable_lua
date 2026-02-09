# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Darktable Lua plugin for automatic cropping of DSLR-scanned film frames. Detects film holder edges and applies crop parameters to remove them.

## Architecture

Two-component system: **Lua plugin** (runs inside darktable) calls a **Python script** (runs externally) for image analysis.

### Data Flow

1. `auto_crop.lua` exports selected images as downscaled JPEGs to a temp folder (`%TEMP%/darktable_autocrop_<timestamp>/`)
2. Lua calls `process_images.py` via `conda run -n autocrop` with file paths as arguments
3. Python detects margins using brightness profile analysis (OpenCV), writes results to `crop_results.json` in the same temp folder
4. Lua reads JSON results (via bundled `dkjson.lua`), modifies each source image's XMP sidecar to inject a crop history entry with the detected parameters, and updates `change_timestamp`/`history_current_hash` to force preview regeneration
5. Lua calls `image:apply_sidecar(xmp_path)` to reload the modified XMP into darktable

### Key Design Decisions

- Crop params are written directly as binary hex into XMP (`darktable:params` field) - 4 little-endian floats for L/T/R/B + 8 zero bytes
- Python outputs crop as percentages from each edge; Lua converts to fractions where R/B are edge positions (1 - margin)
- `darktable:change_timestamp` (microseconds since 0001-01-01) and `darktable:history_current_hash` (random hex) are updated in XMP to force darktable to regenerate previews

### Registered Actions

- **AutoCrop_Debug** (`export_and_find_edges_debug`) - export and detect only, no crop application. For testing edge detection.
- **AutoCrop_InPlace** (`export_detect_and_apply_inplace`) - full pipeline: export, detect, apply crop directly to source image's XMP (no virtual copies). Updates `darktable:change_timestamp` and `darktable:history_current_hash` to force preview regeneration.

## Python Environment

- Uses conda environment named `autocrop` (defined in `environment.yml`)
- Setup: `conda env create -f environment.yml`
- Update: `conda env update -f environment.yml --prune`
- Dependencies: Python 3.11, OpenCV, NumPy, Pillow
- Standalone test: `conda run -n autocrop python process_images.py <image.jpg>`

## Python Usage Rules

- Darktable Lua API is very limited; use Python with OpenCV for any actual image analysis
- New Python dependencies must be added via `environment.yml`

## Darktable Lua API

Docs: https://docs.darktable.org/lua/stable/

### Logging

- `dt.print()` for UI messages (user-facing info only)
- For debug/log file output, use `dlog.msg(level, context, message)` — NOT `dt.print_log()` directly
  - Example: `dlog.msg(dlog.info, "export_detect_and_apply", "About to call apply_crop_to_image")`
- Log level must be re-set in event handlers: `dlog.log_level(dlog.info)` at the top of each handler

### Image Export Pattern

```lua
local images = dt.gui.selection()
local format = dt.new_format("jpeg")
format.max_width = width
format.max_height = height
local success = format:write_image(image, filename, false)  -- 3rd param: allow_upscale
```

### Plugin Registration Pattern

Register GUI action + keyboard shortcut; clean up both in `destroy()`:
```lua
dt.gui.libs.image.register_action("Name", _("Description"), function() ... end, _("Tooltip"))
dt.register_event("Name", "shortcut", function(event, shortcut) ... end, "Name")

local function destroy()
    dt.gui.libs.image.destroy_action("Name")
    dt.destroy_event("Name", "shortcut")
end
script_data.destroy = destroy
```

## Common Gotchas

### Never use `_` as a loop variable
The `_()` function is gettext localization. Using `_` as a discard variable in `for` loops shadows it and causes runtime errors. Use `i` instead:
```lua
for i, image in ipairs(images) do  -- NOT: for _, image in ipairs(images) do
```

### Directory creation
Use `df.mkdir()` directly, never wrapped in `df.check_if_bin_exists()`.

### Filename handling
Always strip extension and sanitize before export:
```lua
local base_name = image.filename:match("(.+)%..+$") or image.filename
local safe_name = df.sanitize_filename(base_name)
```

## Testing

1. Save Lua script
2. Reload in darktable via Lua console (toggle on/off)
3. Select images in lighttable, run via GUI action or shortcut
4. Check log: `%USERPROFILE%\Documents\Darktable\darktable-log.txt` (Windows)
5. Check exported files and `crop_results.json` in the temp directory

## Known Bugs / TODOs

- [ ] Remove Windows-specific invocations (`cmd /c conda run ...`), use platform-independent methods
- [ ] Remove temp dir after creating in-place copies
- [ ] Try to replace external `dkjson.lua` with some simpler in-place variant.
- [ ] Now rate of "slight errors" (where one edge is detected "significantly" wrong compared to human choice) is about 25%, try to improve

## Dependencies

- darktable 4.x+ (API 7.0.0+)
- `lib/dtutils`, `lib/dtutils.file`, `lib/dtutils.system`, `lib/dtutils.log`, `lib/dtutils.debug` (darktable utility libs, not in this repo)
- `dkjson.lua` (bundled JSON parser)
- Python 3.11 via conda `autocrop` environment

