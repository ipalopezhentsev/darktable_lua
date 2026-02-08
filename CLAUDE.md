# Darktable Lua Automation Scripts

This directory contains custom Lua scripts for darktable automation.

## Project Structure

- `auto_crop.lua` - Main darktable plugin for exporting and processing selected images
- `process_images.py` - Python script for processing exported images (planned integration)

## Current Functionality

### AutoCrop Plugin (`auto_crop.lua`)

Exports selected images from darktable to a temporary folder as JPEG files at 10% of original size, then optionally processes them with a Python script.

**Key Features:**
- Exports selected images to temp directory: `%TEMP%\darktable_autocrop_[timestamp]`
- Reduces image size to 10% of original dimensions
- Exports as JPEG format
- GUI action and keyboard shortcut support
- Progress reporting in darktable console

## Darktable Lua API Patterns

### Image Export Workflow

```lua
-- 1. Get selected images
local images = dt.gui.selection()

-- 2. Create export format
local format = dt.new_format("jpeg")

-- 3. Set dimensions
format.max_width = width
format.max_height = height

-- 4. Export image
local success = format:write_image(image, filename, false)
```

**API Documentation:**
- `dt.new_format()`: https://docs.darktable.org/lua/stable/lua.api.manual/darktable/darktable.new_format/
- `dt_imageio_module_format_t`: https://docs.darktable.org/lua/stable/lua.api.manual/types/dt_imageio_module_format_t/

### Important Details

- `write_image(image, filename, allow_upscale)` - third parameter controls upscaling
- Image dimensions available via `image.width` and `image.height` properties
- Format object properties (`max_width`, `max_height`) control output size
- Use `dt.print()` for console output messages

## Common Gotchas

### Variable Shadowing with `_`

**Problem:** The `_()` function is commonly used for gettext localization. Using `_` as a loop variable will shadow this function and cause runtime errors.

```lua
-- WRONG - shadows the _() gettext function
for _, image in ipairs(images) do
  dt.print(_("Some message"))  -- ERROR: attempt to call a number
end

-- CORRECT - use a different variable name
for i, image in ipairs(images) do
  dt.print(_("Some message"))  -- Works correctly
end
```

### Directory Creation

Use `df.mkdir()` directly, not wrapped in `df.check_if_bin_exists()`:

```lua
local df = require "lib/dtutils.file"

-- CORRECT
if not df.mkdir(export_dir) then
  dt.print("Failed to create directory")
  return
end
```

### Filename Handling

Strip file extensions and sanitize names before export:

```lua
local base_name = image.filename:match("(.+)%..+$") or image.filename
local safe_name = df.sanitize_filename(base_name)
local filename = export_dir .. "/" .. safe_name .. ".jpg"
```

## Plugin Registration

### GUI Action Registration

```lua
dt.gui.libs.image.register_action(
    "ActionName",
    _("Display description"),
    function() your_function() end,
    _("Tooltip text")
)
```

### Keyboard Shortcut Registration

```lua
dt.register_event(
    "EventName",
    "shortcut",
    function(event, shortcut) your_function() end,
    "ShortcutName"
)
```

### Cleanup on Unload

```lua
local function destroy()
    dt.gui.libs.image.destroy_action("ActionName")
    dt.destroy_event("EventName", "shortcut")
end

script_data.destroy = destroy
```

## TODOs

- [ ] Integrate Python script execution after export
- [ ] Add error handling for Python script failures
- [ ] Make export size configurable (currently hardcoded to 10%)
- [ ] Add quality settings for JPEG export
- [ ] Consider adding other export formats (PNG, TIFF)
- [ ] Test with large batch exports

## Testing

To test changes:
1. Save the Lua script
2. Restart darktable or reload scripts via Lua console
3. Select images in lighttable
4. Run via GUI action or keyboard shortcut
5. Check console output for progress/errors
6. Verify exported files in temp directory

## Dependencies

- darktable 4.x or later (API version 7.0.0+)
- `lib/dtutils` - darktable utility library
- `lib/dtutils.file` - file operations
- `lib/dtutils.system` - system command execution
- Python 3.x (for future processing integration)
