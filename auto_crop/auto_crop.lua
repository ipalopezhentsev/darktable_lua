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

-- Script directory (for finding auto_crop.py)
local script_dir = debug.getinfo(1).source:match("@?(.*[/\\])")

-- Shared darktable-Lua utilities (common/dt_utils.lua) — stateless XMP/binary
-- helpers shared by all three auto_* plugins.
package.path = package.path .. ";" .. script_dir .. "../common/?.lua"
local dtu = require("dt_utils")

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

-- Parse close_choices.txt written by the debug UI's finish dialog:
-- `apply=0|1` + `delete_temp=0|1`. Returns { apply=bool, delete_temp=bool },
-- or nil when the file is missing (UI closed without the dialog / cancelled).
local function parse_close_choices(export_dir)
  local file = io.open(export_dir .. "/close_choices.txt", "r")
  if not file then return nil end
  local choices = { apply = false, delete_temp = false }
  for line in file:lines() do
    local k, v = line:match("^([%w_]+)=(%d+)$")
    if k == "apply" then choices.apply = (v == "1")
    elseif k == "delete_temp" then choices.delete_temp = (v == "1") end
  end
  file:close()
  return choices
end

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
local float_to_le_hex = dtu.float_to_le_hex

-- Helper: Find the highest history item num in XMP content
-- This is more reliable than using history_end attribute
local find_max_history_num = dtu.find_max_history_num

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
local generate_random_hex = dtu.generate_random_hex

-- Helper: Generate darktable-format timestamp (microseconds since 0001-01-01)
local generate_darktable_timestamp = dtu.generate_darktable_timestamp

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
-- debug_ui_mode: if true, Python saves crop-overlay visualizations, writes the
--   debug session ({stem}_debug_crop.json) and opens debug_ui.py; the Python
--   process is launched detached so darktable isn't blocked while the UI is
--   open, and this function returns nil (no results to apply).
-- Returns: crop_results, filename_to_image, export_dir (or nil on failure /
--   detached debug launch)
-- debug_blocking: run the debug UI in the FOREGROUND (blocking) instead of the
-- detached launch — the unified edit action needs this so it can read
-- crop_results.txt + close_choices.txt after the UI closes.
local function export_and_detect(images, debug_ui_mode, debug_blocking)
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
  local python_script = script_dir .. "auto_crop.py"

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

  local vis_flag = debug_ui_mode and "" or " --no-vis"
  local debug_flag = debug_ui_mode and " --debug-ui" or ""
  local command = string.format('conda run -n autocrop python "%s"%s%s%s',
                                 python_script, debug_flag, vis_flag, file_args)

  -- In debug UI mode, launch Python detached so darktable isn't blocked while the
  -- UI is open — UNLESS debug_blocking (the unified edit action wants to read
  -- crop_results.txt + close_choices.txt after the UI closes, so it falls through
  -- to the foreground path below).
  if debug_ui_mode and not debug_blocking then
    dt.print(_("Running crop detection in background..."))
    if dt.configuration.running_os == "windows" then
      local bat_file = export_dir .. "/run_debug.bat"
      local f = io.open(bat_file, "w")
      if not f then
        dt.print(_("Failed to write batch file for debug launch"))
        return nil, nil, nil
      end
      f:write(command .. ' > "' .. log_file .. '" 2>&1\n')
      f:close()
      -- VBScript launcher: window style 0 = hidden, False = don't wait (fire and forget)
      local vbs_file = export_dir .. "/run_debug.vbs"
      local fv = io.open(vbs_file, "w")
      if not fv then
        dt.print(_("Failed to write vbs launcher for debug launch"))
        return nil, nil, nil
      end
      fv:write('Set WshShell = CreateObject("WScript.Shell")\n')
      fv:write('WshShell.Run Chr(34) & "' .. bat_file .. '" & Chr(34), 0, False\n')
      fv:close()
      os.execute('wscript "' .. vbs_file .. '"')
    else
      os.execute(command .. ' > "' .. log_file .. '" 2>&1 &')
    end
    dt.print(string.format(_("Debug detection started — UI will open when detection finishes. Log: %s"), log_file))
    return nil, nil, nil
  end

  command = command .. ' > "' .. log_file .. '" 2>&1'
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

-- Unified continuous-edit action: export + detect + open the crop debug UI
-- (blocking). On close a finish dialog decides whether to apply the crop (the
-- user's edge corrections, else the auto detection) in-place and whether to
-- delete the temp folder.
local function edit_crop()
  dlog.log_level(dlog.info)
  math.randomseed(os.time() + os.clock() * 1000)

  local images = dt.gui.selection()

  if #images == 0 then
    dt.print(_("No images selected"))
    return
  end

  -- Blocking debug UI: auto_crop.py detects, opens the UI foreground in --apply
  -- mode, and on close the UI (re)writes crop_results.txt + close_choices.txt.
  local crop_results, filename_to_image, export_dir = export_and_detect(images, true, true)
  if not export_dir then
    return
  end

  local choices = parse_close_choices(export_dir)
  local do_apply = choices and choices.apply
  local do_delete = choices and choices.delete_temp

  if not do_apply then
    dt.print(_("Debug UI closed without applying - nothing written to darktable"))
  elseif not crop_results then
    dt.print(_("Apply was chosen but the UI produced no results - nothing applied"))
  else
    dlog.msg(dlog.info, "edit_crop",
      string.format("Applying crops to %d images (in-place, no virtual copies)...", #crop_results))

    local stats = { applied = 0, failed = 0 }

    for idx, result_data in ipairs(crop_results) do
      if result_data.status == "success" then
        local original_image = filename_to_image[result_data.filename]
        if original_image then
          dt.print(string.format(_("Processing %s (in-place)..."), original_image.filename))
          local success, error_msg = apply_crop_in_place(original_image, result_data.crop)
          if success then
            stats.applied = stats.applied + 1
            dlog.msg(dlog.info, "edit_crop",
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

    dt.print(string.format(_("Auto Crop Complete: %d applied, %d failed"),
      stats.applied, stats.failed))
  end

  if do_delete then
    df.rmdir(export_dir)
    dlog.msg(dlog.info, "edit_crop", "Removed temp dir: " .. export_dir)
    dt.print(string.format(_("Removed temp folder: %s"), export_dir))
  else
    dt.print(string.format(_("Temp folder kept: %s"), export_dir))
  end
end

local function destroy()
    dt.gui.libs.image.destroy_action("AutoCrop")
    -- pcall: darktable throws if the event is already gone (e.g. double-destroy on reload)
    pcall(dt.destroy_event, "AutoCrop", "shortcut")
end

-- Unified continuous-edit action: export + detect + open the crop debug UI
-- (blocking). On close a finish dialog decides whether to apply the crop
-- (the user's edge corrections, else the auto detection) and whether to delete
-- the temp folder.
dt.gui.libs.image.register_action(
    "AutoCrop",
    _("Auto crop (edit in debug UI, apply on close)"),
    function() edit_crop() end,
    _("Export, detect margins, open the crop debug UI; on close choose whether to apply the crop in-place and whether to delete the temp folder")
)

dt.register_event(
    "AutoCrop",
    "shortcut",
    function(event, shortcut) edit_crop() end,
    "AutoCrop"
)

script_data.destroy = destroy

return script_data
