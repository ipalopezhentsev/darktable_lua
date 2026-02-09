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

-- Script directory (for finding process_images.py)
local script_dir = debug.getinfo(1).source:match("@?(.*[/\\])")

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

-- Parse crop results file (simple line format: OK|filename|L=x|T=x|R=x|B=x or ERR|filename|message)
local function parse_crop_results(results_file_path)
  local file = io.open(results_file_path, "r")
  if not file then
    dlog.msg(dlog.error, "parse_crop_results", "Failed to open results file: " .. results_file_path)
    return nil
  end

  local results = {}
  for line in file:lines() do
    local status, rest = line:match("^(%u+)|(.+)$")
    if status == "OK" then
      local filename, l, t, r, b = rest:match("^([^|]+)|L=([%d%.]+)|T=([%d%.]+)|R=([%d%.]+)|B=([%d%.]+)$")
      if filename then
        results[#results + 1] = {
          status = "success",
          filename = filename,
          crop = { left = tonumber(l), top = tonumber(t), right = tonumber(r), bottom = tonumber(b) }
        }
      end
    elseif status == "ERR" then
      local filename, error_msg = rest:match("^([^|]+)|(.+)$")
      if filename then
        results[#results + 1] = { status = "error", filename = filename, error = error_msg }
      end
    end
  end

  file:close()
  return results
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

-- Helper: Generate a random hex string of given length
local function generate_random_hex(length)
  local hex = ""
  for i = 1, length do
    hex = hex .. string.format("%x", math.random(0, 15))
  end
  return hex
end

-- Helper: Generate darktable-format timestamp (microseconds since 0001-01-01)
local function generate_darktable_timestamp()
  return (os.time() + 62135596800) * 1000000
end

-- Apply crop to source image in-place by modifying its own XMP sidecar file
-- Also updates change_timestamp and history_current_hash to force preview regeneration
local function apply_crop_in_place(image, crop_data)
  dlog.msg(dlog.info, "apply_crop_in_place", string.format("Called with L=%.2f T=%.2f R=%.2f B=%.2f",
    crop_data.left, crop_data.top, crop_data.right, crop_data.bottom))

  local success, error_msg = pcall(function()
    local xmp_path = image.sidecar

    dlog.msg(dlog.info, "apply_crop_in_place", string.format("Image: %s, id=%d, sidecar: %s",
      image.filename, image.id, xmp_path or "nil"))

    if not xmp_path then
      error("No sidecar path for image: " .. image.filename)
    end

    -- Convert crop percentages to fractions
    local left_fraction = crop_data.left / 100.0
    local top_fraction = crop_data.top / 100.0
    local right_edge = (100.0 - crop_data.right) / 100.0
    local bottom_edge = (100.0 - crop_data.bottom) / 100.0

    dlog.msg(dlog.info, "apply_crop_in_place", string.format("Fractions: L=%.6f T=%.6f R=%.6f B=%.6f",
      left_fraction, top_fraction, right_edge, bottom_edge))

    -- Encode as hex params (4 floats + 8 bytes zeros)
    local params_hex = float_to_le_hex(left_fraction) ..
                       float_to_le_hex(top_fraction) ..
                       float_to_le_hex(right_edge) ..
                       float_to_le_hex(bottom_edge) ..
                       "0000000000000000"

    -- Read existing XMP
    local file = io.open(xmp_path, "r")
    if not file then
      error("Failed to open XMP file for reading: " .. xmp_path)
    end
    local xmp_content = file:read("*all")
    file:close()

    -- Find max history num and compute new values
    local max_history_num = find_max_history_num(xmp_content)
    local new_history_num = max_history_num + 1
    local new_history_end = new_history_num + 1

    dlog.msg(dlog.info, "apply_crop_in_place", string.format("Max history num: %d, adding crop at num=%d", max_history_num, new_history_num))

    -- Create and insert crop module entry
    local crop_module_xml = create_crop_module_xml(new_history_num, params_hex)

    local before_seq_end = xmp_content:find("</rdf:Seq>%s*</darktable:history>")
    if not before_seq_end then
      error("Could not find history sequence end tag in XMP")
    end

    local new_xmp = xmp_content:sub(1, before_seq_end - 1) .. "\n" ..
                    crop_module_xml .. "\n" ..
                    xmp_content:sub(before_seq_end)

    -- Update history_end
    new_xmp = new_xmp:gsub('darktable:history_end="%d+"',
                           string.format('darktable:history_end="%d"', new_history_end))

    -- Update change_timestamp to force darktable to detect a change
    local timestamp = generate_darktable_timestamp()
    local new_xmp_ts, count_ts = new_xmp:gsub(
      'darktable:change_timestamp="%-?%d+"',
      string.format('darktable:change_timestamp="%d"', timestamp))
    if count_ts == 0 then
      dlog.msg(dlog.warn, "apply_crop_in_place", "darktable:change_timestamp not found in XMP, skipping update")
    else
      new_xmp = new_xmp_ts
    end

    -- Update history_current_hash with random value to force preview regeneration
    local new_hash = generate_random_hex(32)
    local new_xmp_hash, count_hash = new_xmp:gsub(
      'darktable:history_current_hash="[%x]+"',
      string.format('darktable:history_current_hash="%s"', new_hash))
    if count_hash == 0 then
      dlog.msg(dlog.warn, "apply_crop_in_place", "darktable:history_current_hash not found in XMP, skipping update")
    else
      new_xmp = new_xmp_hash
    end

    -- Write modified XMP back to the same file
    file = io.open(xmp_path, "w")
    if not file then
      error("Failed to open XMP file for writing: " .. xmp_path)
    end
    file:write(new_xmp)
    file:close()

    dlog.msg(dlog.info, "apply_crop_in_place", string.format("Written XMP with crop, timestamp=%d, hash=%s", timestamp, new_hash))

    -- Reload the XMP sidecar to apply changes immediately
    dlog.msg(dlog.info, "apply_crop_in_place", string.format("Calling image:apply_sidecar(%s) to reload XMP", xmp_path))
    image:apply_sidecar(xmp_path)
    dlog.msg(dlog.info, "apply_crop_in_place", "XMP reloaded successfully")

    dt.print(_("  Crop applied in-place - check darkroom view to see the result"))
  end)

  if not success then
    dlog.msg(dlog.error, "apply_crop_in_place", string.format("Failed: %s", tostring(error_msg)))
    return false, error_msg or "Unknown error"
  end

  return true, nil
end

-- Shared helper: export images and run Python edge detection
-- Returns: crop_results, filename_to_image, export_dir (or nil on failure)
local function export_and_detect(images)
  -- Create temp folder
  local temp_dir = os.getenv("TEMP") or os.getenv("TMP") or "/tmp"
  local export_dir = temp_dir .. "/darktable_autocrop_" .. os.time()

  if not df.mkdir(export_dir) then
    dt.print(_("Failed to create temp directory: " .. export_dir))
    return nil, nil, nil
  end

  dt.print(string.format(_("Exporting %d images to %s"), #images, export_dir))

  -- Create JPEG format
  local format = dt.new_format("jpeg")

  -- Export each selected image and build filename mapping
  local exported_files = {}
  local filename_to_image = {}
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
      filename_to_image[safe_name] = image
      dt.print(string.format(_("  Exported: %s"), filename))
    else
      dt.print(string.format(_("  Failed to export: %s"), image.filename))
    end
  end

  dt.print(string.format(_("Export complete: %d of %d images exported"), #exported_files, #images))

  if #exported_files == 0 then
    dt.print(_("No files exported, aborting"))
    return nil, nil, nil
  end

  -- Call Python script with all exported files at once
  local python_script = script_dir .. "process_images.py"

  if not df.check_if_file_exists(python_script) then
    dt.print(string.format(_("Python script not found: %s"), python_script))
    return nil, nil, nil
  end

  dt.print(string.format(_("Processing %d exported image(s) with Python script..."), #exported_files))

  local file_args = ""
  for i, image_file in ipairs(exported_files) do
    file_args = file_args .. ' "' .. image_file .. '"'
  end

  local log_file = export_dir .. "/processing.log"

  local command = string.format('cmd /c conda run -n autocrop python "%s"%s > "%s" 2>&1',
                                 python_script, file_args, log_file)

  local result = dsys.external_command(command)

  if result ~= 0 then
    dt.print(string.format(_("Python processing failed with code: %d. Check log: %s"), result, log_file))
    return nil, nil, nil
  end

  dt.print(string.format(_("Python processing completed successfully. Log: %s"), log_file))

  -- Parse crop results
  local results_file = export_dir .. "/crop_results.txt"
  dt.print(_("Parsing crop results..."))

  local crop_results = parse_crop_results(results_file)
  if not crop_results then
    dt.print(_("Failed to parse crop results, aborting"))
    return nil, nil, nil
  end

  return crop_results, filename_to_image, export_dir
end

-- Main export and process function (debug version - no crop application)
local function export_and_find_edges_debug()
  dlog.log_level(dlog.info)
  local images = dt.gui.selection()

  if #images == 0 then
    dt.print(_("No images selected"))
    return
  end

  export_and_detect(images)
  -- Debug mode: just export and detect, results are in the temp folder
end

-- Export, detect, and apply crops directly to source images (no virtual copies)
local function export_detect_and_apply_inplace()
  dlog.log_level(dlog.info)
  math.randomseed(os.time() + os.clock() * 1000)

  local images = dt.gui.selection()

  if #images == 0 then
    dt.print(_("No images selected"))
    return
  end

  local crop_results, filename_to_image, export_dir = export_and_detect(images)
  if not crop_results then
    return
  end

  -- Apply crops in-place (no virtual copies)
  dlog.msg(dlog.info, "export_detect_and_apply_inplace",
    string.format("Applying crops to %d images (in-place, no virtual copies)...", #crop_results))

  local stats = {
    applied = 0,
    failed = 0
  }

  for idx, result_data in ipairs(crop_results) do
    dlog.msg(dlog.info, "export_detect_and_apply_inplace",
      string.format("Processing result %d, status=%s", idx, result_data.status or "nil"))

    if result_data.status == "success" then
      local original_image = filename_to_image[result_data.filename]

      if original_image then
        dt.print(string.format(_("Processing %s (in-place)..."), original_image.filename))

        local success, error_msg = apply_crop_in_place(original_image, result_data.crop)

        if success then
          stats.applied = stats.applied + 1
          dlog.msg(dlog.info, "export_detect_and_apply_inplace",
            string.format("Applied in-place crop: L=%.2f%% T=%.2f%% R=%.2f%% B=%.2f%%",
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

  dt.print(string.format(_("Auto Crop In-Place Complete: %d applied, %d failed"),
    stats.applied, stats.failed))
end

local function destroy()
    dt.gui.libs.image.destroy_action("AutoCrop_Debug")
    dt.gui.libs.image.destroy_action("AutoCrop_InPlace")
    dt.destroy_event("AutoCrop_Debug", "shortcut")
    dt.destroy_event("AutoCrop_InPlace", "shortcut")
end

-- GUI registration for debug action (export and detect only, no crop application)
dt.gui.libs.image.register_action(
    "AutoCrop_Debug",
    _("Auto crop debug (no apply)"),
    function() export_and_find_edges_debug() end,
    _("Export and detect margins only - for debugging")
)

-- Shortcut registration for debug action
dt.register_event(
    "AutoCrop_Debug",
    "shortcut",
    function(event, shortcut) export_and_find_edges_debug() end,
    "AutoCrop_Debug"
)

-- GUI registration for in-place action (no virtual copies)
dt.gui.libs.image.register_action(
    "AutoCrop_InPlace",
    _("Auto crop in-place (no virtual copy)"),
    function() export_detect_and_apply_inplace() end,
    _("Export, detect, and apply crop margins directly to selected images")
)

-- Shortcut registration for in-place action
dt.register_event(
    "AutoCrop_InPlace",
    "shortcut",
    function(event, shortcut) export_detect_and_apply_inplace() end,
    "AutoCrop_InPlace"
)

script_data.destroy = destroy

return script_data
