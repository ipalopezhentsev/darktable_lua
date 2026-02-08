# Darktable Lua Automation Scripts

This directory contains custom Lua scripts for darktable automation.

## Project Structure

- `auto_crop.lua` - Performs automatic cropping out of film holder edges at DSLR scanned film frames.

## Current Functionality

### AutoCrop Plugin (`auto_crop.lua`)

Exports selected images from darktable to a temporary folder as JPEG files at smaller size, then finds film holeder edges in them with a Python script.
The script outputs location of these edges as crop percentages. The calling lua script reads these parameters back and applies them to a virtual copy of each
image, as crop parameters. It performs the last step as creating or editing an XMP file for a virtual copy and then instructs darktable to refresh its database
from the modified xmp file.

## Python usage rules

- As darktable Lua API has very limited functionality, for tasks requiring actual image analysis we use Python and libs OpenCV.
- Python is used from a conda virtual env.
- New Python dependencies should be added via `environment.yml` file.

## Darktable Lua API Patterns

### Darktable Lua API
Is described here: https://docs.darktable.org/lua/stable/

### Logging
- Darktable has two 'channels' for printing logs messages: UI (via `dt.print()`) and log file (`dt.print_log()`)
- For logging to UI, use `dt.print()`. UI should have just important user-level info
- All debugging info should go to log file, not UI. And for this, do not use `dt.print_log()` directly. Instead, there is higher-level api for logging where we can control base logging level and filter. Example of invocation: `dlog.msg(dlog.info, "export_detect_and_apply", "About to call apply_crop_to_image")`. Here first argument means level, second - context, third - actual msg

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
- [ ] Fix bug with unreliable crop parameters application. Sometimes parent image breaks (i.e. its history is cleared, it seems that we modify wrong image)
- [ ] Create another action for working directly on source image rather than on virtual copies
- [ ] Ensure no Windows-specific things are present. We should use platform independent invocation methods

## Testing

To test changes:
1. Save the Lua script
2. Reload scripts via Lua console (press on/off)
3. Select images in lighttable
4. Run via GUI action or keyboard shortcut
5. Check UI/log output for progress/errors. In Windows, log is in %USERPROFILE%\Documents\Darktable\darktable-log.txt 
6. Verify exported files in temp directory

## Dependencies

- darktable 4.x or later (API version 7.0.0+)
- `lib/dtutils` - darktable utility library
- `lib/dtutils.file` - file operations
- `lib/dtutils.system` - system command execution
- Python 3.x

