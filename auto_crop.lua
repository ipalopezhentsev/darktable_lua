--[[
Export Selected Images Plugin for Darktable

  Export selected images to a temporary folder at 10% of original size
  and call a Python script to process them.
  
  darktable is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
]]

local dt = require "darktable"
local du = require "lib/dtutils"
local df = require "lib/dtutils.file"
local dsys = require "lib/dtutils.system"
local gettext = dt.gettext.gettext

-- Check API version
du.check_min_api_version("7.0.0", "AutoCrop")

local function _(msgid)
    return gettext(msgid)
end

-- Return data structure for script_manager
local script_data = {}

script_data.metadata = {
  name = _("AutoCrop"),
  purpose = _("Export selected images at 10% size to temp folder and process with Python script"),
  author = "Ilya Palopezhentsev",
  help = "https://github.com/ipalopezhentsev/darktable_lua"
}

script_data.destroy = nil
script_data.destroy_method = nil
script_data.restart = nil
script_data.show = nil

-- Main export and process function
local function export_and_process()
  local images = dt.gui.selection()

  if #images == 0 then
    dt.print(_("No images selected"))
    return
  end

  -- Create temp folder
  local temp_dir = os.getenv("TEMP") or os.getenv("TMP") or "/tmp"
  local export_dir = temp_dir .. "/darktable_autocrop_" .. os.time()

  if not df.mkdir(export_dir) then
    dt.print(_("Failed to create temp directory: " .. export_dir))
    return
  end

  dt.print(string.format(_("Exporting %d images to %s"), #images, export_dir))

  -- Create JPEG format
  local format = dt.new_format("jpeg")

  -- Export each selected image
  local exported_files = {}
  local target_height = 1000
  for i, image in ipairs(images) do
    local scale = target_height / image.height
    local width = math.floor(image.width * scale)
    local height = target_height

    format.max_width = width
    format.max_height = height

    -- Generate filename (remove extension and sanitize)
    local base_name = image.filename:match("(.+)%..+$") or image.filename
    local safe_name = df.sanitize_filename(base_name)
    local filename = export_dir .. "/" .. safe_name .. ".jpg"

    -- Export the image with progress indicator
    dt.print(string.format(_("Exporting (%d/%d): %s (%dx%d)"), i, #images, image.filename, width, height))
    local success = format:write_image(image, filename, false)

    if success then
      table.insert(exported_files, filename)
      dt.print(string.format(_("  Exported: %s"), filename))
    else
      dt.print(string.format(_("  Failed to export: %s"), image.filename))
    end
  end

  dt.print(string.format(_("Export complete: %d of %d images exported"), #exported_files, #images))

  -- Call Python script with all exported files at once
  if #exported_files > 0 then
    -- Get the script directory (where this Lua file is located)
    local script_dir = debug.getinfo(1).source:match("@?(.*[/\\])")
    local python_script = script_dir .. "process_images.py"

    -- Check if Python script exists
    if df.check_if_file_exists(python_script) then
      dt.print(string.format(_("Processing %d exported image(s) with Python script..."), #exported_files))

      -- Build command with all file paths
      local file_args = ""
      for _, image_file in ipairs(exported_files) do
        file_args = file_args .. ' "' .. image_file .. '"'
      end

      -- Set up log file path
      local log_file = export_dir .. "/processing.log"

      -- Call Python script once with all files
      -- Using 'conda run' to execute within the autocrop environment
      local command = string.format('cmd /c conda run -n autocrop python "%s"%s > "%s" 2>&1',
                                     python_script, file_args, log_file)

      local result = dsys.external_command(command)

      if result == 0 then
        dt.print(string.format(_("Python processing completed successfully. Log: %s"), log_file))
      else
        dt.print(string.format(_("Python processing failed with code: %d. Check log: %s"), result, log_file))
      end
    else
      dt.print(string.format(_("Python script not found: %s"), python_script))
    end
  end
end

local function destroy()
    dt.gui.libs.image.destroy_action("AutoCrop")
    dt.destroy_event("AutoCrop", "shortcut")
end

-- GUI registration
dt.gui.libs.image.register_action(
    "AutoCrop",
    _("Auto crop selected images"),
    function() export_and_process() end,
    _("export selected images at 10% size to temp and process with Python")
)

-- Shortcut registration
dt.register_event(
    "AutoCrop",
    "shortcut",
    function(event, shortcut) export_and_process() end,
    "AutoCrop"
)

script_data.destroy = destroy

return script_data
