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
local dlog = require "lib/dtutils.log"
local dd = require "lib/dtutils.debug"
local gettext = dt.gettext.gettext

-- Load dkjson from the same directory as this script
local dkjson_dir = debug.getinfo(1).source:match("@?(.*[/\\])")
package.path = package.path .. ";" .. dkjson_dir .. "?.lua"
local dkjson = require "dkjson"

-- Set up logging
--NOTE in event handlers it won't apply and needs to be set up again!
dlog.log_level(dlog.info)  -- Enable info level and above

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

-- Parse JSON crop results file
local function parse_crop_results(json_file_path)
  if not df.check_if_file_exists(json_file_path) then
    dlog.msg(dlog.error, _(string.format("JSON results file not found: %s", json_file_path)))
    return nil
  end

  -- Read JSON file
  local file = io.open(json_file_path, "r")
  if not file then
    dlog.msg(dlog.error, _(string.format("Failed to open JSON file: %s", json_file_path)))
    return nil
  end

  local json_content = file:read("*all")
  file:close()

  -- Parse JSON
  local success, results = pcall(dkjson.decode, json_content)
  if not success then
    dlog.msg(dlog.error, _(string.format("Failed to parse JSON: %s", results)))
    return nil
  end

  -- Validate structure
  if not results or not results.results then
    dlog.msg(dlog.error, _("Invalid JSON structure: missing results array"))
    return nil
  end

  return results.results
end

-- Get or create virtual copy for an image
local function get_or_create_virtual_copy(image)
  -- For now, always create a new virtual copy
  -- TODO: Check for existing virtual copies using correct API
  dlog.msg(dlog.info, _("  Creating virtual copy..."))

  -- Log original image details before duplication
  dlog.msg(dlog.info, "get_or_create_virtual_copy", string.format("Original image: %s, id=%d", image.filename, image.id))

  --we don't need history as we'll create our own xmp from source because if we call duplicate_with_history then dt will create xmp in background and there are random race conditions with our edits
  local virtual_copy = image:duplicate()

  -- Log virtual copy details after duplication
  dlog.msg(dlog.info, "get_or_create_virtual_copy", string.format("Virtual copy created: id=%d", virtual_copy.id))

  return virtual_copy, false  -- return image and flag indicating it was created
end

-- Helper: Encode a 32-bit float as little-endian hex string
local function float_to_le_hex(value)
    local packed = string.pack("<f", value)
    return (packed:gsub(".", function(c) return string.format("%02x", string.byte(c)) end))
end

-- Helper: Find the highest history item num in XMP content
-- This is more reliable than using history_end attribute
local function find_max_history_num(xmp_content)
  local max_num = -1

  -- Scan through all darktable:num attributes in history items
  for num_str in xmp_content:gmatch('darktable:num="(%d+)"') do
    local num = tonumber(num_str)
    if num and num > max_num then
      max_num = num
    end
  end

  dlog.msg(dlog.info, "find_max_history_num", string.format("Found max history num=%d", max_num))
  return max_num
end

-- Helper: Create crop module XML entry
local function create_crop_module_xml(num, params_hex)
  -- Template based on real darktable XMP structure
  return string.format([[     <rdf:li
      darktable:num="%d"
      darktable:operation="crop"
      darktable:enabled="1"
      darktable:modversion="3"
      darktable:params="%s"
      darktable:multi_name=""
      darktable:multi_name_hand_edited="0"
      darktable:multi_priority="0"
      darktable:blendop_version="14"
      darktable:blendop_params="gz11eJxjYIAACQYYOOHEgAZY0QWAgBGLGANDgz0Ej1Q+dcF/IADRAGpyHQU="/>]],
    num, params_hex)
end

-- Apply crop to an image by modifying its XMP sidecar file
local function apply_crop_to_image(image, crop_data, source_image)
  dlog.msg(dlog.info, "apply_crop_to_image", string.format("Called with L=%.2f T=%.2f R=%.2f B=%.2f",
    crop_data.left, crop_data.top, crop_data.right, crop_data.bottom))

  local success, error_msg = pcall(function()
    -- Get the XMP sidecar file path
    local image_path = tostring(image)  -- Full path to image file
    local xmp_path = image.sidecar

    dlog.msg(dlog.info, "apply_crop_to_image", string.format("Image path: %s", image_path))
    dlog.msg(dlog.info, "apply_crop_to_image", string.format("Path of XMP copy: %s", xmp_path))
    dlog.msg(dlog.info, "apply_crop_to_image", string.format("XMP copy exists: %s", df.check_if_file_exists(xmp_path) and "yes" or "no"))

    -- Convert crop percentages to fractions
    local left_fraction = crop_data.left / 100.0
    local top_fraction = crop_data.top / 100.0
    local right_edge = (100.0 - crop_data.right) / 100.0  -- Edge position, not margin
    local bottom_edge = (100.0 - crop_data.bottom) / 100.0  -- Edge position, not margin

    dlog.msg(dlog.info, "apply_crop_to_image", string.format("Fractions: L=%.6f T=%.6f R=%.6f B=%.6f",
      left_fraction, top_fraction, right_edge, bottom_edge))

    -- Encode as hex params (4 floats + 8 bytes zeros)
    local params_hex = float_to_le_hex(left_fraction) ..
                       float_to_le_hex(top_fraction) ..
                       float_to_le_hex(right_edge) ..
                       float_to_le_hex(bottom_edge) ..
                       "0000000000000000"  -- 8 bytes of zeros

    dlog.msg(dlog.info, "apply_crop_to_image", string.format("Crop params hex: %s", params_hex))

    -- Check if the XMP file of the copy currently exists (before we modify it)
    local xmp_existed_before = df.check_if_file_exists(xmp_path)

    -- Use darktable API to find source XMP for original file
    local xmp_content
    local source_xmp_path = source_image.sidecar

    dlog.msg(dlog.info, "apply_crop_to_image", string.format("Image info: duplicate_index=%d, path=%s, filename=%s",
      image.duplicate_index, image.path, image.filename))

    -- Read source XMP if found via sidecar reference or search
    if source_xmp_path then
      local file = io.open(source_xmp_path, "r")
      if not file then
        error("Failed to open source XMP file for reading: " .. source_xmp_path)
      end
      xmp_content = file:read("*all")
      file:close()
      dlog.msg(dlog.info, "apply_crop_to_image", "Read source XMP to preserve history")
    else
      error("Cannot find XMP of base image: " .. (source_xmp_path or "Empty source xmp path!!!"))
    end

    -- Find the highest history item number in the XMP
    local max_history_num = find_max_history_num(xmp_content)
    local new_history_num = max_history_num + 1
    local new_history_end = new_history_num + 1

    dlog.msg(dlog.info, "apply_crop_to_image", string.format("Max history num: %d, adding crop at num=%d", max_history_num, new_history_num))

    -- Create the crop module entry
    local crop_module_xml = create_crop_module_xml(new_history_num, params_hex)

    -- Find the end of the history sequence (before </rdf:Seq>)
    local before_seq_end = xmp_content:find("</rdf:Seq>%s*</darktable:history>")

    if not before_seq_end then
      error("Could not find history sequence end tag in XMP")
    end

    -- Insert crop module before </rdf:Seq>
    local new_xmp = xmp_content:sub(1, before_seq_end - 1) .. "\n" ..
                    crop_module_xml .. "\n" ..
                    xmp_content:sub(before_seq_end)

    -- Update history_end attribute
    new_xmp = new_xmp:gsub('darktable:history_end="%d+"',
                           string.format('darktable:history_end="%d"', new_history_end))

    -- Write XMP file
    local file = io.open(xmp_path, "w")
    if not file then
      error("Failed to open XMP file for writing: " .. xmp_path)
    end
    file:write(new_xmp)
    file:close()

    -- if xmp_existed_before then
    --   dlog.msg(dlog.info, "apply_crop_to_image", "Modified existing XMP file")
    -- else
      dlog.msg(dlog.info, "apply_crop_to_image", "Created new XMP file with crop")
    -- end

    dlog.msg(dlog.info, string.format(_("  Applied crop via XMP: L=%.2f%% T=%.2f%% R=%.2f%% B=%.2f%%"),
      crop_data.left, crop_data.top, crop_data.right, crop_data.bottom))

    --image:drop_cache()
    --dlog.msg(dlog.info, "apply_crop_to_image", "Dropped cache to regenerate thumbnail")

    -- Reload the XMP sidecar to apply changes immediately
    dlog.msg(dlog.info, "apply_crop_to_image", string.format("Calling image:apply_sidecar(%s) to reload XMP", xmp_path))
    image:apply_sidecar(xmp_path)
    dlog.msg(dlog.info, "apply_crop_to_image", "XMP reloaded successfully")
    
    dt.print(_("  Crop applied and loaded - check darkroom view to see the result"))
  end)

  if not success then
    dlog.msg(dlog.error, "apply_crop_to_image", string.format("Failed: %s", tostring(error_msg)))
    return false, error_msg or "Unknown error"
  end

  return true, nil
end

-- Main export and process function (debug version - no crop application)
local function export_and_find_edges_debug()
  dlog.log_level(dlog.info)
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

-- Export, detect, and apply crops to virtual copies
local function export_detect_and_apply()
  --by default log level will be warn in event handlers...  
  dlog.log_level(dlog.info)  -- Enable info level and above
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

  -- Export each selected image and build filename mapping
  local exported_files = {}
  local filename_to_image = {}  -- Map exported filename to original image object
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
      filename_to_image[safe_name] = image  -- Store mapping
      dt.print(string.format(_("  Exported: %s"), filename))
    else
      dt.print(string.format(_("  Failed to export: %s"), image.filename))
    end
  end

  dt.print(string.format(_("Export complete: %d of %d images exported"), #exported_files, #images))

  -- Call Python script with all exported files at once
  if #exported_files == 0 then
    dt.print(_("No files exported, aborting"))
    return
  end

  -- Get the script directory (where this Lua file is located)
  local script_dir = debug.getinfo(1).source:match("@?(.*[/\\])")
  local python_script = script_dir .. "process_images.py"

  -- Check if Python script exists
  if not df.check_if_file_exists(python_script) then
    dt.print(string.format(_("Python script not found: %s"), python_script))
    return
  end

  dt.print(string.format(_("Processing %d exported image(s) with Python script..."), #exported_files))

  -- Build command with all file paths
  local file_args = ""
  for _, image_file in ipairs(exported_files) do
    file_args = file_args .. ' "' .. image_file .. '"'
  end

  -- Set up log file path
  local log_file = export_dir .. "/processing.log"

  -- Call Python script once with all files
  local command = string.format('cmd /c conda run -n autocrop python "%s"%s > "%s" 2>&1',
                                 python_script, file_args, log_file)

  local result = dsys.external_command(command)

  if result ~= 0 then
    dt.print(string.format(_("Python processing failed with code: %d. Check log: %s"), result, log_file))
    return
  end

  dt.print(string.format(_("Python processing completed successfully. Log: %s"), log_file))

  -- Parse JSON results
  local json_file = export_dir .. "/crop_results.json"
  dt.print(_("Parsing crop results..."))

  local crop_results = parse_crop_results(json_file)
  if not crop_results then
    dt.print(_("Failed to parse crop results, aborting crop application"))
    return
  end

  -- Apply crops to virtual copies
  dlog.msg(dlog.info, "export_detect_and_apply", string.format(_("Applying crops to %d images (creating virtual copies)..."), #crop_results))
  dlog.msg(dlog.info, "export_detect_and_apply", string.format("crop_results table has %d entries", #crop_results))

  local stats = {
    applied = 0,
    failed = 0
  }

  for idx, result_data in ipairs(crop_results) do
    dlog.msg(dlog.info, "export_detect_and_apply", string.format("Processing result %d, status=%s", idx, result_data.status or "nil"))
    if result_data.status == "success" then
      dlog.msg(dlog.info, "export_detect_and_apply", string.format("Looking up filename: %s", result_data.filename or "nil"))
      local original_image = filename_to_image[result_data.filename]
      dlog.msg(dlog.info, "export_detect_and_apply", string.format("Found image: %s", original_image and "yes" or "no"))

      if original_image then
        dt.print(string.format(_("Processing %s..."), original_image.filename))
        dlog.msg(dlog.info, "export_detect_and_apply", "Sidecar of original image: " .. (original_image.sidecar or "EMPTY!!!"))

        -- Get or create virtual copy
        local virtual_copy, was_reused = get_or_create_virtual_copy(original_image)

        -- Apply crop to virtual copy
        dlog.msg(dlog.info, "export_detect_and_apply", "About to call apply_crop_to_image")
        local success, error_msg = apply_crop_to_image(virtual_copy, result_data.crop, original_image)
        dlog.msg(dlog.info, "export_detect_and_apply", string.format("apply_crop_to_image returned: success=%s, error=%s",
          tostring(success), tostring(error_msg or "none")))

        if success then
          stats.applied = stats.applied + 1
          dlog.msg(dlog.info, "export_detect_and_apply", string.format(_("  Applied crop: L=%.2f%% T=%.2f%% R=%.2f%% B=%.2f%%"),
            result_data.crop.left, result_data.crop.top,
            result_data.crop.right, result_data.crop.bottom))
        else
          stats.failed = stats.failed + 1
          dt.print(string.format(_("  *** FAILED to apply crop: %s ***"), error_msg or "Unknown error"))
        end
      else
        stats.failed = stats.failed + 1
        dt.print(string.format(_("Warning: Could not find original image for %s"), result_data.filename))
      end
    else
      stats.failed = stats.failed + 1
      dt.print(string.format(_("Skipped %s: %s"), result_data.filename, result_data.error or "Unknown error"))
    end
  end

  -- Display summary
  dt.print(string.format(_("Auto Crop Complete: %d applied, %d failed"),
    stats.applied, stats.failed))
end

local function destroy()
    dt.gui.libs.image.destroy_action("AutoCrop_Debug")
    dt.gui.libs.image.destroy_action("AutoCrop")
    dt.destroy_event("AutoCrop_Debug", "shortcut")
    dt.destroy_event("AutoCrop", "shortcut")
end

-- GUI registration for debug action (export and detect only, no crop application)
dt.gui.libs.image.register_action(
    "AutoCrop_Debug",
    _("Auto crop debug (no apply)"),
    function() export_and_find_edges_debug() end,
    _("Export and detect margins only - for debugging")
)

-- GUI registration for apply action (export, detect, and apply crops to virtual copies)
dt.gui.libs.image.register_action(
    "AutoCrop",
    _("Auto crop and apply"),
    function() export_detect_and_apply() end,
    _("Export, detect, and apply crop margins to virtual copies")
)

-- Shortcut registration for debug action
dt.register_event(
    "AutoCrop_Debug",
    "shortcut",
    function(event, shortcut) export_and_find_edges_debug() end,
    "AutoCrop_Debug"
)

-- Shortcut registration for apply action
dt.register_event(
    "AutoCrop",
    "shortcut",
    function(event, shortcut) export_detect_and_apply() end,
    "AutoCrop"
)

script_data.destroy = destroy

return script_data
