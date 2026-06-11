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

-- Find the highest multi_priority among existing instances of an operation.
-- Returns -1 if the operation has no entries yet (so caller can start at 0).
-- Note: XMP entries span multiple lines, so we collapse newlines first.
local function find_max_multi_priority(xmp_content, operation)
  local flat = xmp_content:gsub("\n", " ")
  local max_priority = -1

  for entry in flat:gmatch('<rdf:li%s.-/>') do
    local op = entry:match('darktable:operation="([^"]+)"')
    if op == operation then
      local priority = tonumber(entry:match('darktable:multi_priority="(%d+)"')) or 0
      if priority > max_priority then
        max_priority = priority
      end
    end
  end

  dlog.msg(dlog.info, "find_max_multi_priority",
    string.format("Max multi_priority for %s=%d", operation, max_priority))
  return max_priority
end

-- Canonical built-in module pipe order, keyed by darktable:iop_order_version.
-- Images using a built-in order store only the version number (no explicit
-- iop_order_list), but a second module instance cannot be placed without an
-- explicit list. These templates were captured verbatim from darktable-written
-- sidecars; each lists every module once at priority 0. The duplicate is added
-- by add_instance_to_iop_order. If darktable ever writes a new version number,
-- add its template here (capture by manually duplicating a module in darkroom).
local IOP_ORDER_TEMPLATES = {
  ["4"] = "rawprepare,0,invert,0,temperature,0,rasterfile,0,highlights,0,cacorrect,0,hotpixels,0,rawdenoise,0,demosaic,0,denoiseprofile,0,bilateral,0,rotatepixels,0,scalepixels,0,lens,0,cacorrectrgb,0,hazeremoval,0,ashift,0,flip,0,enlargecanvas,0,overlay,0,clipping,0,liquify,0,spots,0,retouch,0,exposure,0,mask_manager,0,tonemap,0,toneequal,0,crop,0,graduatednd,0,profile_gamma,0,equalizer,0,colorin,0,channelmixerrgb,0,diffuse,0,censorize,0,negadoctor,0,blurs,0,primaries,0,nlmeans,0,colorchecker,0,defringe,0,atrous,0,lowpass,0,highpass,0,sharpen,0,colortransfer,0,colormapping,0,channelmixer,0,basicadj,0,colorbalance,0,colorequal,0,colorbalancergb,0,rgbcurve,0,rgblevels,0,basecurve,0,filmic,0,sigmoid,0,agx,0,filmicrgb,0,lut3d,0,colisa,0,tonecurve,0,levels,0,shadhi,0,zonesystem,0,globaltonemap,0,relight,0,bilat,0,colorcorrection,0,colorcontrast,0,velvia,0,vibrance,0,colorzones,0,bloom,0,colorize,0,lowlight,0,monochrome,0,grain,0,soften,0,splittoning,0,vignette,0,colorreconstruct,0,finalscale,0,colorout,0,clahe,0,overexposed,0,rawoverexposed,0,dither,0,borders,0,watermark,0,gamma,0",
}

-- Register a new module instance in darktable's pipe-order list.
-- darktable stores module order in the attribute
--   darktable:iop_order_list="op,priority,op,priority,..."
-- A new instance (multi_priority > 0) MUST appear here or darktable silently
-- drops its history entry and the instance never shows up. The base instance
-- (priority already present, typically 0) needs no change.
-- When the image uses a built-in order (no iop_order_list), we synthesize the
-- full list from IOP_ORDER_TEMPLATES so the new instance can be placed.
-- Returns the (possibly unchanged) xmp_content and a boolean "modified".
local function add_instance_to_iop_order(xmp_content, operation, multi_priority)
  local list = xmp_content:match('darktable:iop_order_list="([^"]*)"')
  local need_inject = false

  if not list then
    -- Built-in order: expand the version template into an explicit list.
    local version = xmp_content:match('darktable:iop_order_version="(%d+)"')
    local template = version and IOP_ORDER_TEMPLATES[version]
    if not template then
      dlog.msg(dlog.error, "add_instance_to_iop_order",
        string.format("No iop_order_list and no template for iop_order_version=%s; "
          .. "cannot add instance (capture this version's order from a manual "
          .. "duplicate and add it to IOP_ORDER_TEMPLATES)", tostring(version)))
      return xmp_content, false
    end
    list = template
    need_inject = true
  end

  -- Tokenize into a flat array: name, priority, name, priority, ...
  local tokens = {}
  for tok in (list .. ","):gmatch("([^,]*),") do
    tokens[#tokens + 1] = tok
  end

  -- Find the last entry for this operation; note if our exact pair already exists.
  local last_idx = nil
  local already = false
  local i = 1
  while i < #tokens do
    if tokens[i] == operation then
      last_idx = i
      if tonumber(tokens[i + 1]) == multi_priority then
        already = true
      end
    end
    i = i + 2
  end

  if already then
    return xmp_content, false  -- e.g. base priority 0 already listed
  end
  if not last_idx then
    dlog.msg(dlog.info, "add_instance_to_iop_order",
      string.format("Operation %s absent from iop_order_list; not inserting", operation))
    return xmp_content, false
  end

  -- Insert the new instance immediately after the last existing one (adjacent
  -- in pipe order). last_idx is the name token; +2/+3 land just past its pair.
  table.insert(tokens, last_idx + 2, operation)
  table.insert(tokens, last_idx + 3, tostring(multi_priority))

  local new_list = table.concat(tokens, ",")
  local safe_list = new_list:gsub("%%", "%%%%")  -- guard gsub replacement metachars
  local new_content
  if need_inject then
    -- No attribute existed: add one right after iop_order_version, matching the
    -- 3-space indentation darktable uses for sibling attributes. %1 keeps the
    -- captured version number.
    local repl = 'darktable:iop_order_version="%1"\n'
      .. '   darktable:iop_order_list="' .. safe_list .. '"'
    new_content = xmp_content:gsub('darktable:iop_order_version="(%d+)"', repl, 1)
  else
    local repl = 'darktable:iop_order_list="' .. safe_list .. '"'
    new_content = xmp_content:gsub('darktable:iop_order_list="[^"]*"', repl, 1)
  end

  dlog.msg(dlog.info, "add_instance_to_iop_order",
    string.format("%s %s,%d in iop_order_list",
      need_inject and "Created list with" or "Inserted", operation, multi_priority))
  return new_content, true
end

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

-- Extract the ashift gz16 base64 blob from XMP content.
-- Returns the base64 string (without the "gz16" prefix), or nil if absent/disabled.
local function parse_ashift_params(xmp_content, history_end)
  local params_str = find_last_enabled_params(xmp_content, history_end, "ashift")
  if not params_str then
    return nil
  end
  -- darktable encodes compressed blobs as "gz16<base64>"; strip the prefix
  local b64 = params_str:match("^gz16(.+)$")
  return b64  -- nil if prefix not found (unexpected format)
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

-- Build XML for every form in the previous (latest) masks snapshot, re-stamped
-- to new_mask_num. darktable's masks_history is CUMULATIVE: each history step
-- must list ALL forms active at that point, not just newly added ones. Without
-- carrying the prior forms forward, adding a new retouch instance erases the
-- earlier instances' forms at the new history position (they show up empty),
-- and darktable may roll back the whole step on reload.
-- Returns the concatenated XML, or nil if there are no prior forms to carry.
local function carry_forward_masks_xml(xmp_content, new_mask_num)
  -- The forms to preserve belong to existing retouch instances; their latest
  -- cumulative snapshot sits at the mask_num equal to the highest existing
  -- retouch history entry. We deliberately tie carry-forward to an actual
  -- retouch entry so leftover/orphan masks (e.g. mask_num=0 cruft from removed
  -- history) are NOT dragged into the new instance.
  local prev_num = -1
  for entry in xmp_content:gmatch('<rdf:li.-/>') do
    if entry:match('darktable:operation="retouch"') then
      local n = tonumber(entry:match('darktable:num="(%d+)"'))
      if n and n > prev_num then prev_num = n end
    end
  end
  if prev_num < 0 then return nil end

  -- Re-emit each form from that snapshot under the new mask_num. (Lua patterns'
  -- '.' matches newlines, so '<rdf:li.-/>' spans a whole multi-line entry;
  -- history entries lack mask_num and are filtered out.)
  local entries = {}
  for entry in xmp_content:gmatch('<rdf:li.-/>') do
    local n = entry:match('darktable:mask_num="(%d+)"')
    if n and tonumber(n) == prev_num then
      entries[#entries + 1] = (entry:gsub('darktable:mask_num="%d+"',
        string.format('darktable:mask_num="%d"', new_mask_num)))
    end
  end

  if #entries == 0 then return nil end
  dlog.msg(dlog.info, "carry_forward_masks_xml",
    string.format("Carried %d forms from snapshot mask_num=%d to %d",
      #entries, prev_num, new_mask_num))
  return table.concat(entries, "\n")
end

-- Create retouch module history XML entry.
-- multi_priority and multi_name make this a distinct, named module instance so
-- repeated runs add new retouch instances instead of overwriting earlier ones.
local function create_retouch_module_xml(num, retouch_params, blendop_params, multi_priority, multi_name)
  -- multi_name_hand_edited="1" tells darktable the label is user-set and must be preserved
  local hand_edited = (multi_name ~= nil and multi_name ~= "") and "1" or "0"
  return string.format([[     <rdf:li
      darktable:num="%d"
      darktable:operation="retouch"
      darktable:enabled="1"
      darktable:modversion="3"
      darktable:params="%s"
      darktable:multi_name="%s"
      darktable:multi_name_hand_edited="%s"
      darktable:multi_priority="%d"
      darktable:blendop_version="14"
      darktable:blendop_params="%s"/>]],
    num, retouch_params, multi_name or "", hand_edited, multi_priority or 0, blendop_params)
end

-- Apply retouch to source image in-place by modifying its XMP sidecar.
-- multi_name is the label shown in darktable for this retouch instance (e.g.
-- "film dust" / "sensor dust"); each call adds a NEW instance, leaving any
-- existing retouch instances intact.
local function apply_retouch_in_place(image, dust_data, multi_name)
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

    -- Each run becomes its own retouch instance: pick a multi_priority above any
    -- existing retouch instance so darktable does not overwrite earlier ones.
    local new_multi_priority = find_max_multi_priority(xmp_content, "retouch") + 1

    dlog.msg(dlog.info, "apply_retouch_in_place",
      string.format("Max history num: %d, adding retouch at num=%d, multi_priority=%d, label=%s",
        max_history_num, new_history_num, new_multi_priority, multi_name or ""))

    -- Step 1: Insert mask entries into masks_history. The new snapshot must be
    -- cumulative: carry forward all forms from the previous snapshot (so earlier
    -- retouch instances keep their brushes) and append this run's new forms.
    local new_masks_xml = create_mask_entries_xml(mask_num, dust_data.brushes, dust_data.group)
    local carried_xml = carry_forward_masks_xml(xmp_content, new_history_num)
    local mask_xml = carried_xml and (carried_xml .. "\n" .. new_masks_xml) or new_masks_xml

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
      new_history_num, dust_data.params.retouch, dust_data.params.blendop,
      new_multi_priority, multi_name)

    local before_seq_end = xmp_content:find("</rdf:Seq>%s*</darktable:history>")
    if not before_seq_end then
      error("Could not find history sequence end tag in XMP")
    end

    xmp_content = xmp_content:sub(1, before_seq_end - 1) .. "\n" ..
                  retouch_xml .. "\n" ..
                  xmp_content:sub(before_seq_end)

    -- Step 2b: Register this instance in the module pipe-order list, otherwise
    -- darktable ignores the history entry for any non-base instance.
    local iop_modified
    xmp_content, iop_modified = add_instance_to_iop_order(xmp_content, "retouch", new_multi_priority)
    if not iop_modified and new_multi_priority > 0 then
      dlog.msg(dlog.error, "apply_retouch_in_place",
        string.format("Could not register retouch instance %d in iop_order_list; "
          .. "darktable may drop it", new_multi_priority))
    end

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
local function export_and_detect(images, debug_ui_mode, sensor_dust_mode)
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

      local ashift_b64 = nil
      if xmp_path then
        local xf = io.open(xmp_path, "r")
        if xf then
          local xmp = xf:read("*all")
          xf:close()
          local history_end = tonumber(xmp:match('darktable:history_end="(%d+)"')) or 999
          flip_val = parse_flip_param(xmp, history_end)
          cl, ct, cr, cb = parse_crop_params(xmp, history_end)
          ashift_b64 = parse_ashift_params(xmp, history_end)
          dlog.msg(dlog.info, "export_and_detect",
            string.format("Transform for %s: flip=%d crop=%.4f,%.4f,%.4f,%.4f ashift=%s",
              safe_name, flip_val, cl, ct, cr, cb, ashift_b64 and "yes" or "no"))
        end
      end

      -- Force dots as decimal separators regardless of system locale
      local function fmt_float(val)
        return (string.format("%.6f", val):gsub(",", "."))
      end
      local line = string.format("%s|flip=%d|crop=%s,%s,%s,%s",
        safe_name, flip_val, fmt_float(cl), fmt_float(ct), fmt_float(cr), fmt_float(cb))
      if ashift_b64 then
        line = line .. "|ashift=" .. ashift_b64
      end
      tf:write(line .. "\n")
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
  local debug_flag = debug_ui_mode and " --debug-ui" or ""
  local sensor_dust_flag = sensor_dust_mode and " --sensor-dust" or ""
  local ml_model_path = script_dir .. "dust_ml_model.pkl"
  local ml_flag = df.check_if_file_exists(ml_model_path) and (' --ml-model "' .. ml_model_path .. '"') or ""
  local command = string.format('conda run --no-capture-output -n autocrop python -u "%s"%s%s%s%s',
                                 python_script, debug_flag, sensor_dust_flag, ml_flag, file_args)

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

  export_and_detect(images, true)
end

-- Sensor dust debug mode: export, detect and open the debug UI - no apply
local function export_and_detect_sensor_dust_debug()
  dlog.log_level(dlog.info)
  local images = dt.gui.selection()

  if #images < 2 then
    dt.print(_("Select 2 or more images from the same scanning session for sensor dust removal"))
    return
  end

  export_and_detect(images, true, true)
end

-- Full pipeline: export, detect, apply retouch to source images
-- keep_temp: if true, the temp folder is kept after a successful run for analysis
local function export_detect_and_apply_retouch_inplace(keep_temp)
  dlog.log_level(dlog.info)
  math.randomseed(os.time() + os.clock() * 1000)

  local images = dt.gui.selection()

  if #images == 0 then
    dt.print(_("No images selected"))
    return
  end

  local dust_results, filename_to_image, export_dir = export_and_detect(images, false)
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

        local success, error_msg = apply_retouch_in_place(original_image, dust_data, _("film dust"))

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

  -- Clean up temp dir if no errors (keep on failure or in keep-temp mode)
  if stats.failed == 0 and not keep_temp then
    df.rmdir(export_dir)
    dlog.msg(dlog.info, "export_detect_and_apply_retouch_inplace", "Removed temp dir: " .. export_dir)
  else
    dt.print(string.format(_("Temp folder kept for analysis: %s"), export_dir))
    dlog.msg(dlog.info, "export_detect_and_apply_retouch_inplace",
      "Keeping temp dir for inspection: " .. export_dir)
  end
end

-- Sensor dust pipeline: detect dust common across all selected frames, apply to each image
local function export_detect_and_apply_sensor_dust(keep_temp)
  dlog.log_level(dlog.info)
  math.randomseed(os.time() + os.clock() * 1000)

  local images = dt.gui.selection()

  if #images < 2 then
    dt.print(_("Select 2 or more images from the same scanning session for sensor dust removal"))
    return
  end

  local dust_results, filename_to_image, export_dir = export_and_detect(images, false, true)
  if not dust_results then
    return
  end

  dlog.msg(dlog.info, "export_detect_and_apply_sensor_dust", "Applying sensor dust retouch to images...")

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
      dt.print(string.format(_("No sensor dust in crop of %s"), filename))
    else
      local original_image = filename_to_image[filename]
      if original_image then
        dt.print(string.format(_("Applying sensor dust retouch to %s (%d spot(s))..."),
          original_image.filename, dust_data.count))

        local success, error_msg = apply_retouch_in_place(original_image, dust_data, _("sensor dust"))

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

  dt.print(string.format(_("Sensor Dust Removal: %d applied, %d no dust in crop, %d failed"),
    stats.applied, stats.skipped, stats.failed))

  -- Clean up temp dir on clean success (skipped in debug mode)
  if stats.failed == 0 and stats.applied > 0 and not keep_temp then
    df.rmdir(export_dir)
    dlog.msg(dlog.info, "export_detect_and_apply_sensor_dust", "Removed temp dir: " .. export_dir)
  else
    dt.print(string.format(_("Log kept for inspection: %s"), export_dir .. "/processing.log"))
    dlog.msg(dlog.info, "export_detect_and_apply_sensor_dust",
      "Keeping temp dir for inspection: " .. export_dir)
  end
end

-- ===================================================================
-- Plugin registration
-- ===================================================================

local function destroy()
    dt.gui.libs.image.destroy_action("AutoRetouch_Debug")
    dt.gui.libs.image.destroy_action("AutoRetouch_InPlace")
    dt.gui.libs.image.destroy_action("AutoRetouch_InPlace_KeepTemp")
    dt.gui.libs.image.destroy_action("AutoRetouch_SensorDust")
    dt.gui.libs.image.destroy_action("AutoRetouch_SensorDust_KeepTemp")
    dt.gui.libs.image.destroy_action("AutoRetouch_SensorDust_Debug")
    -- pcall: darktable throws if the event is already gone (e.g. double-destroy on reload)
    pcall(dt.destroy_event, "AutoRetouch_Debug", "shortcut")
    pcall(dt.destroy_event, "AutoRetouch_InPlace", "shortcut")
    pcall(dt.destroy_event, "AutoRetouch_InPlace_KeepTemp", "shortcut")
    pcall(dt.destroy_event, "AutoRetouch_SensorDust", "shortcut")
    pcall(dt.destroy_event, "AutoRetouch_SensorDust_KeepTemp", "shortcut")
    pcall(dt.destroy_event, "AutoRetouch_SensorDust_Debug", "shortcut")
end

-- ============ Film dust ============

-- Mode 1: debug UI (detect + annotate, no apply)
dt.gui.libs.image.register_action(
    "AutoRetouch_Debug",
    _("Auto retouch debug (open debug UI, no apply)"),
    function() export_and_detect_dust_debug() end,
    _("Export, detect dust spots and open the debug UI - nothing applied")
)

dt.register_event(
    "AutoRetouch_Debug",
    "shortcut",
    function(event, shortcut) export_and_detect_dust_debug() end,
    "AutoRetouch_Debug"
)

-- Mode 2: fully automatic, temp folder removed on success
dt.gui.libs.image.register_action(
    "AutoRetouch_InPlace",
    _("Auto retouch in-place (heal dust)"),
    function() export_detect_and_apply_retouch_inplace(false) end,
    _("Detect dust spots and apply heal retouch to selected images")
)

dt.register_event(
    "AutoRetouch_InPlace",
    "shortcut",
    function(event, shortcut) export_detect_and_apply_retouch_inplace(false) end,
    "AutoRetouch_InPlace"
)

-- Mode 3: fully automatic, temp folder kept for analysis
dt.gui.libs.image.register_action(
    "AutoRetouch_InPlace_KeepTemp",
    _("Auto retouch in-place (keep temp folder)"),
    function() export_detect_and_apply_retouch_inplace(true) end,
    _("Same as in-place, but keeps the temp folder and log for analysis")
)

dt.register_event(
    "AutoRetouch_InPlace_KeepTemp",
    "shortcut",
    function(event, shortcut) export_detect_and_apply_retouch_inplace(true) end,
    "AutoRetouch_InPlace_KeepTemp"
)

-- ============ Sensor dust (cross-frame: select all frames from one scanning session) ============

-- Mode 1: debug UI (detect + annotate, no apply)
dt.gui.libs.image.register_action(
    "AutoRetouch_SensorDust_Debug",
    _("Auto retouch sensor dust debug (open debug UI, no apply)"),
    function() export_and_detect_sensor_dust_debug() end,
    _("Detect sensor dust across selected frames and open the debug UI - nothing applied")
)

dt.register_event(
    "AutoRetouch_SensorDust_Debug",
    "shortcut",
    function(event, shortcut) export_and_detect_sensor_dust_debug() end,
    "AutoRetouch_SensorDust_Debug"
)

-- Mode 2: fully automatic, temp folder removed on success
dt.gui.libs.image.register_action(
    "AutoRetouch_SensorDust",
    _("Auto retouch sensor dust (heal sensor dust)"),
    function() export_detect_and_apply_sensor_dust(false) end,
    _("Detect and heal DSLR sensor dust common across all selected frames")
)

dt.register_event(
    "AutoRetouch_SensorDust",
    "shortcut",
    function(event, shortcut) export_detect_and_apply_sensor_dust(false) end,
    "AutoRetouch_SensorDust"
)

-- Mode 3: fully automatic, temp folder kept for analysis
dt.gui.libs.image.register_action(
    "AutoRetouch_SensorDust_KeepTemp",
    _("Auto retouch sensor dust (keep temp folder)"),
    function() export_detect_and_apply_sensor_dust(true) end,
    _("Same as sensor dust heal, but keeps the temp folder and log for analysis")
)

dt.register_event(
    "AutoRetouch_SensorDust_KeepTemp",
    "shortcut",
    function(event, shortcut) export_detect_and_apply_sensor_dust(true) end,
    "AutoRetouch_SensorDust_KeepTemp"
)

script_data.destroy = destroy

return script_data
