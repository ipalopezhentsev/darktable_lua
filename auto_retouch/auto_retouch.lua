--[[
Auto Retouch Plugin for Darktable

  Detect dust spots on DSLR-scanned film negatives and apply
  heal retouch entries to source images' XMP sidecar files.

  darktable is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
]]

local dt = require "darktable"
local du = require "lib/dtutils"
local df = require "lib/dtutils.file"
local dlog = require "lib/dtutils.log"
local dd = require "lib/dtutils.debug"
local gettext = dt.gettext.gettext

-- Script directory (for finding detect_dust.py)
local script_dir = debug.getinfo(1).source:match("@?(.*[/\\])")

-- Set up logging
--NOTE in event handlers it won't apply and needs to be set up again!
dlog.log_level(dlog.info)

-- Check API version
du.check_min_api_version("7.0.0", "AutoRetouch")

local function _(msgid)
    return gettext(msgid)
end

-- Return data structure for script_manager
local script_data = {}

script_data.metadata = {
  name = _("AutoRetouch"),
  purpose = _("Detect dust spots on film scans and apply heal retouch"),
  author = "Ilya Palopezhentsev",
  help = "https://github.com/ipalopezhentsev/darktable_lua"
}

script_data.destroy = nil
script_data.destroy_method = nil
script_data.restart = nil
script_data.show = nil

-- ===================================================================
-- Helper functions
-- ===================================================================

-- Find the highest history item num in XMP content
local function find_max_history_num(xmp_content)
  local max_num = -1
  for num_str in xmp_content:gmatch('darktable:num="(%d+)"') do
    local num = tonumber(num_str)
    if num and num > max_num then
      max_num = num
    end
  end
  dlog.msg(dlog.info, "find_max_history_num", string.format("Found max history num=%d", max_num))
  return max_num
end

-- Generate a random hex string of given length
local function generate_random_hex(length)
  local hex = ""
  for i = 1, length do
    hex = hex .. string.format("%x", math.random(0, 15))
  end
  return hex
end

-- Generate darktable-format timestamp (microseconds since 0001-01-01)
local function generate_darktable_timestamp()
  return (os.time() + 62135596800) * 1000000
end

-- Decode 8-char little-endian hex string to float
local function le_hex_to_float(hex_str)
  local bytes = {}
  for i = 1, #hex_str, 2 do
    bytes[#bytes + 1] = string.char(tonumber(hex_str:sub(i, i + 1), 16))
  end
  return string.unpack("<f", table.concat(bytes))
end

-- Find the last enabled history entry for a given operation.
-- Returns the params hex string, or nil if not found.
-- Note: XMP entries span multiple lines, so we collapse newlines first.
local function find_last_enabled_params(xmp_content, history_end, operation)
  -- Collapse newlines so '.' matches everything within an entry
  local flat = xmp_content:gsub("\n", " ")
  local best_num = -1
  local best_params = nil

  for entry in flat:gmatch('<rdf:li%s.-/>') do
    local num = tonumber(entry:match('darktable:num="(%d+)"'))
    local op = entry:match('darktable:operation="([^"]+)"')
    local enabled = entry:match('darktable:enabled="([^"]+)"')
    local params = entry:match('darktable:params="([^"]+)"')

    if op == operation and enabled == "1" and num and num <= history_end and num > best_num then
      best_num = num
      best_params = params
    end
  end

  return best_params
end

local function parse_flip_param(xmp_content, history_end)
  local params_hex = find_last_enabled_params(xmp_content, history_end, "flip")
  if not params_hex or #params_hex < 8 then
    return 0
  end

  -- Decode params: 4-byte LE int32
  local flip_val = string.unpack("<i4",
    string.char(tonumber(params_hex:sub(1, 2), 16)) ..
    string.char(tonumber(params_hex:sub(3, 4), 16)) ..
    string.char(tonumber(params_hex:sub(5, 6), 16)) ..
    string.char(tonumber(params_hex:sub(7, 8), 16)))

  -- Auto (-1) treated as 0 for now
  if flip_val < 0 then
    return 0
  end
  return flip_val
end

-- Parse the effective crop parameters from XMP content.
-- Returns L, T, R, B floats. Default 0,0,1,1 if no crop.
local function parse_crop_params(xmp_content, history_end)
  local params_hex = find_last_enabled_params(xmp_content, history_end, "crop")
  if not params_hex or #params_hex < 32 then
    return 0.0, 0.0, 1.0, 1.0
  end

  local left = le_hex_to_float(params_hex:sub(1, 8))
  local top = le_hex_to_float(params_hex:sub(9, 16))
  local right = le_hex_to_float(params_hex:sub(17, 24))
  local bottom = le_hex_to_float(params_hex:sub(25, 32))

  return left, top, right, bottom
end

-- ===================================================================
-- Parse dust detection results
-- ===================================================================

local function parse_dust_results(results_file_path)
  local file = io.open(results_file_path, "r")
  if not file then
    dlog.msg(dlog.error, "parse_dust_results", "Cannot open results file: " .. results_file_path)
    return nil
  end

  -- results[filename] = { count=N, brushes={...}, group={...}, params={...} }
  -- or { count=0, error="..." }
  local results = {}

  for line in file:lines() do
    local line_type = line:match("^(%u+)|")

    if line_type == "OK" then
      local filename, count_str = line:match("^OK|([^|]+)|N=(%d+)$")
      if filename then
        results[filename] = {
          count = tonumber(count_str),
          brushes = {},
          group = nil,
          params = nil
        }
      end

    elseif line_type == "BRUSH" then
      local filename, idx_str, mask_id_str, points_hex, src_hex, nb_str =
        line:match("^BRUSH|([^|]+)|(%d+)|mask_id=(%d+)|mask_points=([^|]+)|mask_src=([^|]+)|mask_nb=(%d+)$")
      if filename and results[filename] then
        table.insert(results[filename].brushes, {
          mask_id = mask_id_str,
          mask_points = points_hex,
          mask_src = src_hex,
          mask_nb = nb_str
        })
      end

    elseif line_type == "GROUP" then
      local filename, mask_id_str, points_hex, src_hex, nb_str =
        line:match("^GROUP|([^|]+)|mask_id=(%d+)|mask_points=([^|]+)|mask_src=([^|]+)|mask_nb=(%d+)$")
      if filename and results[filename] then
        results[filename].group = {
          mask_id = mask_id_str,
          mask_points = points_hex,
          mask_src = src_hex,
          mask_nb = nb_str
        }
      end

    elseif line_type == "PARAMS" then
      local filename, retouch_params, blendop_params =
        line:match("^PARAMS|([^|]+)|retouch_params=([^|]+)|blendop_params=(.+)$")
      if filename and results[filename] then
        results[filename].params = {
          retouch = retouch_params,
          blendop = blendop_params
        }
      end

    elseif line_type == "ERR" then
      local filename, error_msg = line:match("^ERR|([^|]+)|(.+)$")
      if filename then
        results[filename] = { count = 0, error = error_msg }
      end
    end
  end

  file:close()
  return results
end

-- ===================================================================
-- XMP manipulation
-- ===================================================================

-- Create mask XML entries for all brushes and the group
local function create_mask_entries_xml(mask_num, brushes, group)
  local entries = {}

  -- Brush mask entries
  for i, brush in ipairs(brushes) do
    table.insert(entries, string.format(
      [[     <rdf:li
      darktable:mask_num="%s"
      darktable:mask_id="%s"
      darktable:mask_type="72"
      darktable:mask_name="brush #%d"
      darktable:mask_version="6"
      darktable:mask_points="%s"
      darktable:mask_nb="%s"
      darktable:mask_src="%s"/>]],
      mask_num, brush.mask_id, i, brush.mask_points, brush.mask_nb, brush.mask_src))
  end

  -- Group mask entry
  table.insert(entries, string.format(
    [[     <rdf:li
      darktable:mask_num="%s"
      darktable:mask_id="%s"
      darktable:mask_type="12"
      darktable:mask_name="group `retouch'"
      darktable:mask_version="6"
      darktable:mask_points="%s"
      darktable:mask_nb="%s"
      darktable:mask_src="%s"/>]],
    mask_num, group.mask_id, group.mask_points, group.mask_nb, group.mask_src))

  return table.concat(entries, "\n")
end

-- Create retouch module history XML entry
local function create_retouch_module_xml(num, retouch_params, blendop_params)
  return string.format([[     <rdf:li
      darktable:num="%d"
      darktable:operation="retouch"
      darktable:enabled="1"
      darktable:modversion="3"
      darktable:params="%s"
      darktable:multi_name=""
      darktable:multi_name_hand_edited="0"
      darktable:multi_priority="0"
      darktable:blendop_version="14"
      darktable:blendop_params="%s"/>]],
    num, retouch_params, blendop_params)
end

-- Apply retouch to source image in-place by modifying its XMP sidecar
local function apply_retouch_in_place(image, dust_data)
  dlog.msg(dlog.info, "apply_retouch_in_place",
    string.format("Called for %s with %d spots", image.filename, dust_data.count))

  local success, error_msg = pcall(function()
    local xmp_path = image.sidecar

    if not xmp_path then
      error("No sidecar path for image: " .. image.filename)
    end

    -- Read existing XMP
    local file = io.open(xmp_path, "r")
    if not file then
      error("Failed to open XMP file for reading: " .. xmp_path)
    end
    local xmp_content = file:read("*all")
    file:close()

    -- Determine the new history entry number.
    -- Use the max of: (a) the highest darktable:num found by scan, and
    -- (b) history_end - 1 (the last active entry index).
    -- This protects against find_max_history_num missing entries (e.g. due to
    -- darktable internally compacting/renaming entries), which would otherwise
    -- set history_end too low and deactivate existing history steps.
    local current_end_str = xmp_content:match('darktable:history_end="(%d+)"')
    local current_history_end = tonumber(current_end_str) or 0
    local scanned_max = find_max_history_num(xmp_content)
    local max_history_num = math.max(scanned_max, current_history_end - 1)
    local new_history_num = max_history_num + 1
    local new_history_end = new_history_num + 1
    local mask_num = tostring(new_history_num)

    dlog.msg(dlog.info, "apply_retouch_in_place",
      string.format("Max history num: %d, adding retouch at num=%d", max_history_num, new_history_num))

    -- Step 1: Insert mask entries into masks_history
    local mask_xml = create_mask_entries_xml(mask_num, dust_data.brushes, dust_data.group)

    local has_masks_history = xmp_content:find("<darktable:masks_history>")
    if has_masks_history then
      -- Non-empty masks_history: has explicit </rdf:Seq> closing tag
      local insert_pos = xmp_content:find("</rdf:Seq>%s*</darktable:masks_history>")
      if insert_pos then
        xmp_content = xmp_content:sub(1, insert_pos - 1) .. "\n" ..
                      mask_xml .. "\n" ..
                      xmp_content:sub(insert_pos)
      else
        -- Empty masks_history uses self-closing <rdf:Seq/> — expand it to hold our masks
        local seq_pos = xmp_content:find("<rdf:Seq/>", has_masks_history, true)
        if not seq_pos then
          error("Found masks_history but could not find its content sequence")
        end
        xmp_content = xmp_content:sub(1, seq_pos - 1) ..
                      "<rdf:Seq>\n" .. mask_xml .. "\n    </rdf:Seq>" ..
                      xmp_content:sub(seq_pos + 10)  -- skip past "<rdf:Seq/>"
      end
    else
      -- Create masks_history section before <darktable:history>
      local history_pos = xmp_content:find("<darktable:history>")
      if not history_pos then
        error("Could not find <darktable:history> in XMP")
      end
      local masks_section = "   <darktable:masks_history>\n" ..
                           "    <rdf:Seq>\n" ..
                           mask_xml .. "\n" ..
                           "    </rdf:Seq>\n" ..
                           "   </darktable:masks_history>\n   "
      xmp_content = xmp_content:sub(1, history_pos - 1) ..
                    masks_section ..
                    xmp_content:sub(history_pos)
    end

    -- Step 2: Insert retouch history entry
    local retouch_xml = create_retouch_module_xml(
      new_history_num, dust_data.params.retouch, dust_data.params.blendop)

    local before_seq_end = xmp_content:find("</rdf:Seq>%s*</darktable:history>")
    if not before_seq_end then
      error("Could not find history sequence end tag in XMP")
    end

    xmp_content = xmp_content:sub(1, before_seq_end - 1) .. "\n" ..
                  retouch_xml .. "\n" ..
                  xmp_content:sub(before_seq_end)

    -- Step 3: Update history_end
    xmp_content = xmp_content:gsub('darktable:history_end="%d+"',
      string.format('darktable:history_end="%d"', new_history_end))

    -- Step 4: Update change_timestamp
    local timestamp = generate_darktable_timestamp()
    local new_xmp_ts, count_ts = xmp_content:gsub(
      'darktable:change_timestamp="%-?%d+"',
      string.format('darktable:change_timestamp="%d"', timestamp))
    if count_ts > 0 then
      xmp_content = new_xmp_ts
    end

    -- Step 5: Update history_current_hash
    local new_hash = generate_random_hex(32)
    local new_xmp_hash, count_hash = xmp_content:gsub(
      'darktable:history_current_hash="[%x]+"',
      string.format('darktable:history_current_hash="%s"', new_hash))
    if count_hash > 0 then
      xmp_content = new_xmp_hash
    end

    -- Write modified XMP
    file = io.open(xmp_path, "w")
    if not file then
      error("Failed to open XMP file for writing: " .. xmp_path)
    end
    file:write(xmp_content)
    file:close()

    dlog.msg(dlog.info, "apply_retouch_in_place",
      string.format("Written XMP with retouch, timestamp=%d, hash=%s", timestamp, new_hash))

    -- Reload the XMP sidecar
    image:apply_sidecar(xmp_path)
    dlog.msg(dlog.info, "apply_retouch_in_place", "XMP reloaded successfully")

    dt.print(string.format(_("  Retouch applied: %d heal spots"), dust_data.count))
  end)

  if not success then
    dlog.msg(dlog.error, "apply_retouch_in_place", string.format("Failed: %s", tostring(error_msg)))
    return false, error_msg or "Unknown error"
  end

  return true, nil
end

-- ===================================================================
-- Export and detection pipeline
-- ===================================================================

-- Export images at full resolution and run Python dust detection
local function export_and_detect(images, save_visualization, debug_ui_mode)
  -- Create temp folder
  local temp_dir = os.getenv("TEMP") or os.getenv("TMP") or "/tmp"
  local export_dir = temp_dir .. "/darktable_autoretouch_" .. os.time()

  if not df.mkdir(export_dir) then
    dt.print(_("Failed to create temp directory: " .. export_dir))
    return nil, nil, nil
  end

  dt.print(string.format(_("Exporting %d images to %s (full resolution)"), #images, export_dir))

  -- Create JPEG format
  local format = dt.new_format("jpeg")

  -- Export each selected image at full resolution
  local exported_files = {}
  local filename_to_image = {}

  for i, image in ipairs(images) do
    format.max_width = image.width
    format.max_height = image.height

    -- Generate filename (remove extension and sanitize)
    local base_name = image.filename:match("(.+)%..+$") or image.filename
    local safe_name = df.sanitize_filename(base_name)
    local filename = export_dir .. "/" .. safe_name .. ".jpg"

    dt.print(string.format(_("Exporting (%d/%d): %s (full res %dx%d)"),
      i, #images, image.filename, image.width, image.height))
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

  -- Write per-image transform params (flip/crop) for Python
  local transform_file = export_dir .. "/transform_params.txt"
  local tf = io.open(transform_file, "w")
  if tf then
    for safe_name, image in pairs(filename_to_image) do
      local xmp_path = image.sidecar
      local flip_val = 0
      local cl, ct, cr, cb = 0.0, 0.0, 1.0, 1.0

      if xmp_path then
        local xf = io.open(xmp_path, "r")
        if xf then
          local xmp = xf:read("*all")
          xf:close()
          local history_end = tonumber(xmp:match('darktable:history_end="(%d+)"')) or 999
          flip_val = parse_flip_param(xmp, history_end)
          cl, ct, cr, cb = parse_crop_params(xmp, history_end)
          dlog.msg(dlog.info, "export_and_detect",
            string.format("Transform for %s: flip=%d crop=%.4f,%.4f,%.4f,%.4f",
              safe_name, flip_val, cl, ct, cr, cb))
        end
      end

      -- Force dots as decimal separators regardless of system locale
      local function fmt_float(val)
        return (string.format("%.6f", val):gsub(",", "."))
      end
      tf:write(string.format("%s|flip=%d|crop=%s,%s,%s,%s\n",
        safe_name, flip_val, fmt_float(cl), fmt_float(ct), fmt_float(cr), fmt_float(cb)))
    end
    tf:close()
  end

  -- Call Python script
  local python_script = script_dir .. "detect_dust.py"

  if not df.check_if_file_exists(python_script) then
    dt.print(string.format(_("Python script not found: %s"), python_script))
    return nil, nil, nil
  end

  dt.print(string.format(_("Detecting dust spots on %d image(s) with Python..."), #exported_files))

  local file_args = ""
  for i, image_file in ipairs(exported_files) do
    file_args = file_args .. ' "' .. image_file .. '"'
  end

  local log_file = export_dir .. "/processing.log"
  local vis_flag = save_visualization and "" or " --no-vis"
  local debug_flag = debug_ui_mode and " --debug-ui" or ""
  local command = string.format('conda run --no-capture-output -n autocrop python -u "%s"%s%s%s',
                                 python_script, vis_flag, debug_flag, file_args)

  -- In debug UI mode, launch Python detached so darktable isn't blocked while the UI is open
  if debug_ui_mode then
    dt.print(_("Running detection in background (conda env + detection can take ~30s)..."))
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

  local log_handle = io.open(log_file, "w")
  local pipe = io.popen(command .. " 2>&1")
  if not pipe then
    if log_handle then log_handle:close() end
    dt.print(_("Failed to launch Python process"))
    return nil, nil, nil
  end
  for line in pipe:lines() do
    if log_handle then log_handle:write(line .. "\n"); log_handle:flush() end
    local done, total = line:match("^PROGRESS|(%d+)|(%d+)")
    if done then
      dt.print(string.format(_("Dust detection: %s / %s images done..."), done, total))
    end
  end
  if log_handle then log_handle:close() end
  local pipe_status = {pipe:close()}
  local ok   = pipe_status[1]
  local code = pipe_status[3]
  local exit_code = code or 0
  local result = (ok or exit_code == 0) and 0 or exit_code

  if result ~= 0 then
    dt.print(string.format(_("Dust detection failed with code: %d. Check log: %s"), result, log_file))
    return nil, nil, nil
  end

  dt.print(string.format(_("Dust detection completed. Log: %s"), log_file))

  -- Parse results
  local results_file = export_dir .. "/dust_results.txt"
  local dust_results = parse_dust_results(results_file)
  if not dust_results then
    dt.print(_("Failed to parse dust results, aborting"))
    return nil, nil, nil
  end

  return dust_results, filename_to_image, export_dir
end

-- ===================================================================
-- Entry points
-- ===================================================================

-- Debug mode: export and detect only, no retouch application
local function export_and_detect_dust_debug()
  dlog.log_level(dlog.info)
  local images = dt.gui.selection()

  if #images == 0 then
    dt.print(_("No images selected"))
    return
  end

  export_and_detect(images, true, true)
end

-- Full pipeline: export, detect, apply retouch to source images
local function export_detect_and_apply_retouch_inplace()
  dlog.log_level(dlog.info)
  math.randomseed(os.time() + os.clock() * 1000)

  local images = dt.gui.selection()

  if #images == 0 then
    dt.print(_("No images selected"))
    return
  end

  local dust_results, filename_to_image, export_dir = export_and_detect(images, false, false)
  if not dust_results then
    return
  end

  dlog.msg(dlog.info, "export_detect_and_apply_retouch_inplace", "Applying retouch to images...")

  local stats = {
    applied = 0,
    skipped = 0,
    failed = 0
  }

  for filename, dust_data in pairs(dust_results) do
    if dust_data.error then
      stats.failed = stats.failed + 1
      dt.print(string.format(_("Skipped %s: %s"), filename, dust_data.error))
    elseif dust_data.count == 0 then
      stats.skipped = stats.skipped + 1
      dt.print(string.format(_("No dust spots found in %s"), filename))
    else
      local original_image = filename_to_image[filename]
      if original_image then
        dt.print(string.format(_("Applying retouch to %s (%d spots)..."),
          original_image.filename, dust_data.count))

        local success, error_msg = apply_retouch_in_place(original_image, dust_data)

        if success then
          stats.applied = stats.applied + 1
        else
          stats.failed = stats.failed + 1
          dt.print(string.format(_("  *** FAILED to apply retouch: %s ***"), error_msg or "Unknown error"))
        end
      else
        stats.failed = stats.failed + 1
        dt.print(string.format(_("Warning: Could not find original image for %s"), filename))
      end
    end
  end

  dt.print(string.format(_("Auto Retouch Complete: %d applied, %d no dust, %d failed"),
    stats.applied, stats.skipped, stats.failed))

  -- Clean up temp dir if no errors
  if stats.failed == 0 then
    df.rmdir(export_dir)
    dlog.msg(dlog.info, "export_detect_and_apply_retouch_inplace", "Removed temp dir: " .. export_dir)
  else
    dlog.msg(dlog.info, "export_detect_and_apply_retouch_inplace",
      "Keeping temp dir for inspection due to errors: " .. export_dir)
  end
end

-- ===================================================================
-- Plugin registration
-- ===================================================================

local function destroy()
    dt.gui.libs.image.destroy_action("AutoRetouch_Debug")
    dt.gui.libs.image.destroy_action("AutoRetouch_InPlace")
    dt.destroy_event("AutoRetouch_Debug", "shortcut")
    dt.destroy_event("AutoRetouch_InPlace", "shortcut")
end

-- Debug action
dt.gui.libs.image.register_action(
    "AutoRetouch_Debug",
    _("Auto retouch debug (no apply)"),
    function() export_and_detect_dust_debug() end,
    _("Export and detect dust spots only - for debugging")
)

dt.register_event(
    "AutoRetouch_Debug",
    "shortcut",
    function(event, shortcut) export_and_detect_dust_debug() end,
    "AutoRetouch_Debug"
)

-- In-place action
dt.gui.libs.image.register_action(
    "AutoRetouch_InPlace",
    _("Auto retouch in-place (heal dust)"),
    function() export_detect_and_apply_retouch_inplace() end,
    _("Detect dust spots and apply heal retouch to selected images")
)

dt.register_event(
    "AutoRetouch_InPlace",
    "shortcut",
    function(event, shortcut) export_detect_and_apply_retouch_inplace() end,
    "AutoRetouch_InPlace"
)

script_data.destroy = destroy

return script_data
