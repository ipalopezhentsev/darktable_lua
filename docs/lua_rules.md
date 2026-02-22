## Lua usage
- Darktable has Lua 5.4. We do not care about Lua features unsupported by previous Lua versions
- Darktable has helper commands for platform independence. We should favor using them, to make the script platform independent
- In case we see we require slightly different calls for e.g. Windows vs Mac, we create if branches, if unable to find one common approach
- Darktable Lua API is very limited; use Python with OpenCV for any actual image analysis

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

### Locale-safe float formatting
`string.format("%.6f", val)` uses the system locale's decimal separator (comma on European systems). When writing floats for cross-language interchange (e.g. Lua → Python), always force dots:
```lua
string.format("%.6f", val):gsub(",", ".")
```

### Filename handling
Always strip extension and sanitize before export:
```lua
local base_name = image.filename:match("(.+)%..+$") or image.filename
local safe_name = df.sanitize_filename(base_name)
```
