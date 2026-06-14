--[[
Auto Negadoctor Plugin for Darktable

  Automates color-negative inversion for DSLR-scanned film rolls:
  exports the selected (uninverted) frames, runs auto_negadoctor.py which
  finds the roll-wide film base (exposure-compensated via EXIF) and derives
  full negadoctor parameters per frame, then writes a negadoctor history
  entry into each frame's XMP sidecar.

  The export is 16-bit TIFF in linear Rec2020: the orange film base is out
  of the sRGB gamut, so an sRGB export would clip its red channel and ruin
  the film-base color measurement. The export color profile preferences are
  temporarily overridden and restored afterwards.

  darktable is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
]]

local dt = require "darktable"
local du = require "lib/dtutils"
local df = require "lib/dtutils.file"
local dlog = require "lib/dtutils.log"
local gettext = dt.gettext.gettext

-- Script directory (for finding auto_negadoctor.py)
local script_dir = debug.getinfo(1).source:match("@?(.*[/\\])")

-- Set up logging
--NOTE in event handlers it won't apply and needs to be set up again!
dlog.log_level(dlog.info)

du.check_min_api_version("7.0.0", "AutoNegadoctor")

local function _(msgid)
    return gettext(msgid)
end

local script_data = {}

script_data.metadata = {
  name = _("AutoNegadoctor"),
  purpose = _("Automatic color negative inversion via the negadoctor module"),
  author = "Ilya Palopezhentsev",
  help = "https://github.com/ipalopezhentsev/darktable_lua"
}

script_data.destroy = nil
script_data.destroy_method = nil
script_data.restart = nil
script_data.show = nil

-- Analysis export size. auto_negadoctor.py is resolution-invariant (its
-- size-dependent constants are fractions of the frame), so this can be
-- changed freely without retuning anything.
local EXPORT_MAX_WIDTH = 2000

-- darktable export color profile codes (dt_colorspaces_color_profile_type_t)
local DT_COLORSPACE_LIN_REC2020 = 4

-- Tone mappers that distort the export and must be disabled on the scans
local TONE_MAPPER_OPS = { "agx", "filmicrgb", "sigmoid", "basecurve" }

-- ===================================================================
-- Results parsing
-- ===================================================================

-- Parse negadoctor results (OK|stem|params=hex / ERR|stem|msg; DETAIL
-- ignored). A roll-level VIGNETTE line carries the ready lens-module params
-- (gz04 blob) with the fitted manual vignette correction.
-- Returns results list, vignette table (or nil).
local function parse_nega_results(results_file_path)
  local file = io.open(results_file_path, "r")
  if not file then
    dlog.msg(dlog.error, "parse_nega_results", "Failed to open results file: " .. results_file_path)
    return nil
  end

  local results = {}
  local vignette = nil
  for line in file:lines() do
    local status, rest = line:match("^(%u+)|(.+)$")
    if status == "OK" then
      local filename, params_hex = rest:match("^([^|]+)|params=([0-9a-fA-F]+)$")
      if filename and #params_hex == 152 then
        results[#results + 1] = {
          status = "success",
          filename = filename,
          params_hex = params_hex,
        }
      else
        dlog.msg(dlog.warn, "parse_nega_results", "Malformed OK line: " .. line)
      end
    elseif status == "ERR" then
      local filename, error_msg = rest:match("^([^|]+)|(.+)$")
      if filename then
        results[#results + 1] = { status = "error", filename = filename, error = error_msg }
      end
    elseif status == "VIGNETTE" then
      local lens_params = rest:match("|params=([%w%+/=]+)$")
      local strength = rest:match("strength=([%d%.%-]+)")
      if lens_params then
        vignette = { params = lens_params, strength = strength }
        dlog.msg(dlog.info, "parse_nega_results",
          "Vignette correction found, strength=" .. tostring(strength))
      end
    end
    -- DETAIL lines are for humans/logs only
  end

  file:close()
  return results, vignette
end

-- ===================================================================
-- XMP helpers
-- ===================================================================

-- Helper: Find the highest history item num in XMP content
local function find_max_history_num(xmp_content)
  local max_num = -1
  for num_str in xmp_content:gmatch('darktable:num="(%d+)"') do
    local num = tonumber(num_str)
    if num and num > max_num then
      max_num = num
    end
  end
  return max_num
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

-- Iterate all history <rdf:li .../> entries in an XMP. Returns a list of
-- { s = start_pos, e = end_pos, text, num, operation, enabled }.
local function scan_history_entries(xmp_content)
  local entries = {}
  local init = 1
  while true do
    local s, e = xmp_content:find("<rdf:li.-/>", init)
    if not s then break end
    local text = xmp_content:sub(s, e)
    local op = text:match('darktable:operation="([^"]+)"')
    if op then  -- masks_history entries have no operation attribute
      entries[#entries + 1] = {
        s = s, e = e, text = text,
        num = tonumber(text:match('darktable:num="(%d+)"')) or -1,
        operation = op,
        enabled = text:match('darktable:enabled="(%d)"') == "1",
      }
    end
    init = e + 1
  end
  return entries
end

-- Check whether an operation has an ACTIVE entry (enabled and inside
-- history_end) in the XMP.
local function has_enabled_module(xmp_content, opname)
  local history_end = tonumber(xmp_content:match('darktable:history_end="(%d+)"')) or 999
  for i, entry in ipairs(scan_history_entries(xmp_content)) do
    if entry.operation == opname and entry.enabled and entry.num < history_end then
      return true
    end
  end
  return false
end

-- Read an image's XMP sidecar content, or nil when the file doesn't exist.
local function read_xmp(image)
  local xmp_path = image.sidecar
  if not xmp_path then return nil, nil end
  local f = io.open(xmp_path, "r")
  if not f then return nil, xmp_path end
  local content = f:read("*all")
  f:close()
  return content, xmp_path
end

-- Pre-flight check over the selection. Returns ok, plus prints warnings.
-- Aborts (returns false) when any frame has an ACTIVE negadoctor entry:
-- the analysis export would already be inverted. Warns about tone mappers
-- (agx/filmic/sigmoid/basecurve) and missing XMPs (in-place apply needs one).
local function preflight_check(images)
  local inverted = {}
  local tone_mapped = {}
  local missing_xmp = 0

  for i, image in ipairs(images) do
    local xmp = read_xmp(image)
    if not xmp then
      missing_xmp = missing_xmp + 1
    else
      if has_enabled_module(xmp, "negadoctor") then
        inverted[#inverted + 1] = image.filename
      end
      for j, op in ipairs(TONE_MAPPER_OPS) do
        if has_enabled_module(xmp, op) then
          tone_mapped[#tone_mapped + 1] = image.filename .. " (" .. op .. ")"
          break
        end
      end
    end
  end

  if #inverted > 0 then
    dt.print(string.format(
      _("ABORT: %d frame(s) already have negadoctor enabled (e.g. %s). Run AutoNegadoctor_Remove first."),
      #inverted, inverted[1]))
    dlog.msg(dlog.error, "preflight_check",
      "Aborting: negadoctor already enabled on " .. table.concat(inverted, ", "))
    return false
  end

  if #tone_mapped > 0 then
    dt.print(string.format(
      _("WARNING: tone mapper enabled on %d frame(s) (e.g. %s) - this distorts the analysis; disable it for negative scans"),
      #tone_mapped, tone_mapped[1]))
    dlog.msg(dlog.warn, "preflight_check",
      "Tone mapper active on: " .. table.concat(tone_mapped, ", "))
  end

  if missing_xmp > 0 then
    dt.print(string.format(
      _("Note: %d frame(s) have no XMP sidecar yet - applying will fail for them (open them once in darkroom first)"),
      missing_xmp))
  end

  return true
end

-- ===================================================================
-- Apply / remove negadoctor in XMP
-- ===================================================================

-- Helper: Create negadoctor module XML entry (modversion 2, single instance)
local function create_negadoctor_module_xml(num, params_hex)
  return string.format([[     <rdf:li
      darktable:num="%d"
      darktable:operation="negadoctor"
      darktable:enabled="1"
      darktable:modversion="2"
      darktable:params="%s"
      darktable:multi_name=""
      darktable:multi_name_hand_edited="0"
      darktable:multi_priority="0"
      darktable:blendop_version="14"
      darktable:blendop_params="gz11eJxjYIAACQYYOOHEgAZY0QWAgBGLGANDgz0Ej1Q+dcF/IADRAGpyHQU="/>]],
    num, params_hex)
end

-- Update change_timestamp + history_current_hash, write the XMP and reload it.
local function finalize_xmp(image, xmp_path, xmp_content, context)
  local timestamp = generate_darktable_timestamp()
  local with_ts, count_ts = xmp_content:gsub(
    'darktable:change_timestamp="%-?%d+"',
    string.format('darktable:change_timestamp="%d"', timestamp))
  if count_ts > 0 then
    xmp_content = with_ts
  end

  local new_hash = generate_random_hex(32)
  local with_hash, count_hash = xmp_content:gsub(
    'darktable:history_current_hash="[%x]+"',
    string.format('darktable:history_current_hash="%s"', new_hash))
  if count_hash > 0 then
    xmp_content = with_hash
  end

  local file = io.open(xmp_path, "w")
  if not file then
    error("Failed to open XMP file for writing: " .. xmp_path)
  end
  file:write(xmp_content)
  file:close()

  dlog.msg(dlog.info, context,
    string.format("Written XMP, timestamp=%d, hash=%s", timestamp, new_hash))

  image:apply_sidecar(xmp_path)
  dlog.msg(dlog.info, context, "XMP reloaded successfully")
end

-- Apply negadoctor params to a source image's XMP. If a negadoctor entry
-- already exists (only a DISABLED one can get here - pre-flight aborts on
-- enabled ones), its params are replaced in place and it is re-enabled;
-- otherwise a new history entry is inserted.
local function apply_negadoctor_in_place(image, params_hex)
  dlog.msg(dlog.info, "apply_negadoctor_in_place", "Called for " .. image.filename)

  local success, error_msg = pcall(function()
    local xmp_content, xmp_path = read_xmp(image)
    if not xmp_path then
      error("No sidecar path for image: " .. image.filename)
    end
    if not xmp_content then
      error("XMP sidecar does not exist yet for " .. image.filename ..
            " - open the image once in darkroom so darktable creates it")
    end

    -- Find an existing negadoctor entry (take the last one by position)
    local existing = nil
    for i, entry in ipairs(scan_history_entries(xmp_content)) do
      if entry.operation == "negadoctor" then
        existing = entry
      end
    end

    local current_history_end = tonumber(xmp_content:match('darktable:history_end="(%d+)"')) or 0

    if existing then
      -- Replace params in place and force enabled; never grows the history
      local new_entry = existing.text
        :gsub('darktable:params="[^"]*"',
              'darktable:params="' .. params_hex .. '"')
        :gsub('darktable:enabled="%d"', 'darktable:enabled="1"')
      xmp_content = xmp_content:sub(1, existing.s - 1) .. new_entry ..
                    xmp_content:sub(existing.e + 1)
      if existing.num >= current_history_end then
        -- The entry sat above history_end (inactive tail) - extend to cover it
        xmp_content = xmp_content:gsub('darktable:history_end="%d+"',
          string.format('darktable:history_end="%d"', existing.num + 1))
      end
      dlog.msg(dlog.info, "apply_negadoctor_in_place",
        string.format("Replaced existing negadoctor entry num=%d", existing.num))
    else
      -- Insert a new entry. history_end protection: use the max of the
      -- scanned num and history_end-1 so disabled trailing steps are not
      -- deactivated (the auto_retouch fix).
      local scanned_max = find_max_history_num(xmp_content)
      local new_history_num = math.max(scanned_max, current_history_end - 1) + 1
      local new_history_end = new_history_num + 1

      local module_xml = create_negadoctor_module_xml(new_history_num, params_hex)
      local before_seq_end = xmp_content:find("</rdf:Seq>%s*</darktable:history>")
      if not before_seq_end then
        error("Could not find history sequence end tag in XMP")
      end
      xmp_content = xmp_content:sub(1, before_seq_end - 1) .. "\n" ..
                    module_xml .. "\n" ..
                    xmp_content:sub(before_seq_end)
      xmp_content = xmp_content:gsub('darktable:history_end="%d+"',
        string.format('darktable:history_end="%d"', new_history_end))
      dlog.msg(dlog.info, "apply_negadoctor_in_place",
        string.format("Inserted negadoctor at num=%d, history_end=%d",
          new_history_num, new_history_end))
      -- No iop_order_list edit needed: negadoctor at multi_priority 0 is part
      -- of every built-in module order.
    end

    finalize_xmp(image, xmp_path, xmp_content, "apply_negadoctor_in_place")
    dt.print(_("  Negadoctor applied - check darkroom view"))
  end)

  if not success then
    dlog.msg(dlog.error, "apply_negadoctor_in_place", string.format("Failed: %s", tostring(error_msg)))
    return false, error_msg or "Unknown error"
  end

  return true, nil
end

-- Helper: Create lens module XML entry (modversion 10; params blob built by
-- Python: lensfun template for the user's rig with the fitted manual
-- vignette and the lensfun vignetting flag cleared)
local function create_lens_module_xml(num, params_gz)
  return string.format([[     <rdf:li
      darktable:num="%d"
      darktable:operation="lens"
      darktable:enabled="1"
      darktable:modversion="10"
      darktable:params="%s"
      darktable:multi_name=""
      darktable:multi_name_hand_edited="0"
      darktable:multi_priority="0"
      darktable:blendop_version="14"
      darktable:blendop_params="gz11eJxjYIAACQYYOOHEgAZY0QWAgBGLGANDgz0Ej1Q+dcF/IADRAGpyHQU="/>]],
    num, params_gz)
end

-- Apply the roll's lens-module entry (vignette correction) to an image's
-- XMP. If an ENABLED lens entry exists it is left untouched (user-managed -
-- the export was then already corrected and the Python-side estimate is
-- near zero, so no double correction). A disabled entry gets its params
-- replaced and re-enabled; otherwise a new entry is inserted.
-- Returns "applied", "kept" or false, error_msg.
local function apply_lens_in_place(image, params_gz)
  local outcome = nil
  local success, error_msg = pcall(function()
    local xmp_content, xmp_path = read_xmp(image)
    if not xmp_path then
      error("No sidecar path for image: " .. image.filename)
    end
    if not xmp_content then
      error("XMP sidecar does not exist yet for " .. image.filename ..
            " - open the image once in darkroom so darktable creates it")
    end

    local existing = nil
    for i, entry in ipairs(scan_history_entries(xmp_content)) do
      if entry.operation == "lens" then
        existing = entry
      end
    end

    local current_history_end = tonumber(xmp_content:match('darktable:history_end="(%d+)"')) or 0

    if existing and existing.enabled and existing.num < current_history_end then
      dlog.msg(dlog.info, "apply_lens_in_place",
        "Enabled lens entry already present - keeping the user's correction")
      outcome = "kept"
      return
    end

    if existing then
      local new_entry = existing.text
        :gsub('darktable:params="[^"]*"',
              'darktable:params="' .. params_gz .. '"')
        :gsub('darktable:enabled="%d"', 'darktable:enabled="1"')
      xmp_content = xmp_content:sub(1, existing.s - 1) .. new_entry ..
                    xmp_content:sub(existing.e + 1)
      if existing.num >= current_history_end then
        xmp_content = xmp_content:gsub('darktable:history_end="%d+"',
          string.format('darktable:history_end="%d"', existing.num + 1))
      end
      dlog.msg(dlog.info, "apply_lens_in_place",
        string.format("Replaced existing lens entry num=%d", existing.num))
    else
      local scanned_max = find_max_history_num(xmp_content)
      local new_history_num = math.max(scanned_max, current_history_end - 1) + 1
      local new_history_end = new_history_num + 1
      local module_xml = create_lens_module_xml(new_history_num, params_gz)
      local before_seq_end = xmp_content:find("</rdf:Seq>%s*</darktable:history>")
      if not before_seq_end then
        error("Could not find history sequence end tag in XMP")
      end
      xmp_content = xmp_content:sub(1, before_seq_end - 1) .. "\n" ..
                    module_xml .. "\n" ..
                    xmp_content:sub(before_seq_end)
      xmp_content = xmp_content:gsub('darktable:history_end="%d+"',
        string.format('darktable:history_end="%d"', new_history_end))
      dlog.msg(dlog.info, "apply_lens_in_place",
        string.format("Inserted lens entry at num=%d", new_history_num))
    end

    finalize_xmp(image, xmp_path, xmp_content, "apply_lens_in_place")
    outcome = "applied"
  end)

  if not success then
    dlog.msg(dlog.error, "apply_lens_in_place", string.format("Failed: %s", tostring(error_msg)))
    return false, error_msg or "Unknown error"
  end

  return outcome, nil
end

-- Remove all negadoctor entries from an image's history (for clean re-runs).
-- Renumbers the surviving entries and fixes history_end accordingly.
-- Returns "removed", "none" or false, error_msg.
local function remove_negadoctor_in_place(image)
  dlog.msg(dlog.info, "remove_negadoctor_in_place", "Called for " .. image.filename)

  local outcome = nil
  local success, error_msg = pcall(function()
    local xmp_content, xmp_path = read_xmp(image)
    if not xmp_path or not xmp_content then
      outcome = "none"   -- no XMP -> nothing to remove
      return
    end

    -- History entries live between <darktable:history> and </darktable:history>;
    -- scan_history_entries only returns entries with an operation attribute,
    -- so masks_history items are never touched.
    local entries = scan_history_entries(xmp_content)
    local removed, survivors = {}, {}
    for i, entry in ipairs(entries) do
      if entry.operation == "negadoctor" then
        removed[#removed + 1] = entry
      else
        survivors[#survivors + 1] = entry
      end
    end

    if #removed == 0 then
      outcome = "none"
      return
    end

    local old_end = tonumber(xmp_content:match('darktable:history_end="(%d+)"')) or 0
    local removed_below_end = 0
    for i, entry in ipairs(removed) do
      if entry.num < old_end then
        removed_below_end = removed_below_end + 1
      end
    end

    -- Renumber survivors sequentially in their original num order
    table.sort(survivors, function(a, b) return a.num < b.num end)
    local rebuilt = {}
    for new_num, entry in ipairs(survivors) do
      rebuilt[#rebuilt + 1] = entry.text:gsub('darktable:num="%d+"',
        string.format('darktable:num="%d"', new_num - 1), 1)
    end

    -- Replace the whole history <rdf:Seq> content with the rebuilt entries
    local hist_start = xmp_content:find("<darktable:history>")
    local hist_end_tag = xmp_content:find("</darktable:history>")
    if not hist_start or not hist_end_tag then
      error("Could not find <darktable:history> section in XMP")
    end
    local seq_open_s, seq_open_e = xmp_content:find("<rdf:Seq>", hist_start, true)
    local seq_close_s = xmp_content:find("</rdf:Seq>", seq_open_e or hist_start, true)
    if not seq_open_e or not seq_close_s or seq_close_s > hist_end_tag then
      error("Could not find history rdf:Seq in XMP")
    end

    local body = #rebuilt > 0 and ("\n" .. table.concat(rebuilt, "\n") .. "\n    ") or "\n    "
    xmp_content = xmp_content:sub(1, seq_open_e) .. body ..
                  xmp_content:sub(seq_close_s)

    local new_end = math.max(old_end - removed_below_end, 0)
    xmp_content = xmp_content:gsub('darktable:history_end="%d+"',
      string.format('darktable:history_end="%d"', new_end))

    dlog.msg(dlog.info, "remove_negadoctor_in_place",
      string.format("Removed %d negadoctor entr%s, history_end %d -> %d",
        #removed, #removed == 1 and "y" or "ies", old_end, new_end))

    finalize_xmp(image, xmp_path, xmp_content, "remove_negadoctor_in_place")
    outcome = "removed"
  end)

  if not success then
    dlog.msg(dlog.error, "remove_negadoctor_in_place", string.format("Failed: %s", tostring(error_msg)))
    return false, error_msg or "Unknown error"
  end

  return outcome, nil
end

-- ===================================================================
-- Export and detection pipeline
-- ===================================================================

-- Export selected images as 16-bit linear-Rec2020 TIFFs and run the Python
-- analysis. The export ICC preferences are temporarily overridden (the
-- film-base color is out of sRGB gamut) and restored afterwards.
-- Returns: nega_results, filename_to_image, export_dir (or nil on failure /
--   detached debug launch)
local function export_and_detect(images, debug_ui_mode, ai_tune)
  if not preflight_check(images) then
    return nil, nil, nil
  end

  local temp_dir = os.getenv("TEMP") or os.getenv("TMP") or "/tmp"
  local export_dir = temp_dir .. "/darktable_autonegadoctor_" .. os.time()

  if not df.mkdir(export_dir) then
    dt.print(_("Failed to create temp directory: " .. export_dir))
    return nil, nil, nil
  end

  dt.print(string.format(_("Exporting %d images to %s (16-bit linear Rec2020 TIFF)"),
    #images, export_dir))

  -- Temporarily force the export profile to linear Rec2020
  local saved_icctype = dt.preferences.read("darktable", "plugins/lighttable/export/icctype", "integer")
  local saved_iccprofile = dt.preferences.read("darktable", "plugins/lighttable/export/iccprofile", "string")
  dt.preferences.write("darktable", "plugins/lighttable/export/icctype", "integer", DT_COLORSPACE_LIN_REC2020)
  dt.preferences.write("darktable", "plugins/lighttable/export/iccprofile", "string", "")

  -- 32-bit float TIFF: a 16-bit integer TIFF clamps the pipeline at 1.0 and
  -- the film base's red channel exceeds 1.0 with current import defaults,
  -- which would clip the Dmin measurement (found on the first live run)
  local format = dt.new_format("tiff")
  format.bpp = 32
  if format.bpp ~= 32 then
    dt.print(string.format(
      _("WARNING: TIFF bpp=32 rejected (got %s) - film base will be clipped at 1.0"),
      tostring(format.bpp)))
  end
  dlog.msg(dlog.info, "export_and_detect", "TIFF export bpp=" .. tostring(format.bpp))

  local exported_files = {}
  local filename_to_image = {}

  local export_ok, export_err = pcall(function()
    for i, image in ipairs(images) do
      local scale = EXPORT_MAX_WIDTH / image.width
      format.max_width = EXPORT_MAX_WIDTH
      format.max_height = math.floor(image.height * scale)

      local base_name = image.filename:match("(.+)%..+$") or image.filename
      local safe_name = df.sanitize_filename(base_name)
      local filename = export_dir .. "/" .. safe_name .. ".tif"

      dt.print(string.format(_("Exporting (%d/%d): %s"), i, #images, image.filename))
      local success = format:write_image(image, filename, false)

      if success then
        table.insert(exported_files, filename)
        filename_to_image[safe_name] = image
      else
        dt.print(string.format(_("  Failed to export: %s"), image.filename))
      end
    end
  end)

  -- Always restore the export profile preferences
  dt.preferences.write("darktable", "plugins/lighttable/export/icctype", "integer", saved_icctype)
  dt.preferences.write("darktable", "plugins/lighttable/export/iccprofile", "string", saved_iccprofile or "")

  if not export_ok then
    dt.print(string.format(_("Export failed: %s"), tostring(export_err)))
    return nil, nil, nil
  end

  dt.print(string.format(_("Export complete: %d of %d images exported"), #exported_files, #images))

  if #exported_files == 0 then
    dt.print(_("No files exported, aborting"))
    return nil, nil, nil
  end

  -- Write per-image EXIF exposure params for cross-frame compensation
  local exif_file = export_dir .. "/exif_params.txt"
  local ef = io.open(exif_file, "w")
  if ef then
    -- Force dots as decimal separators regardless of system locale
    local function fmt_float(val)
      return (string.format("%.6f", val or 0):gsub(",", "."))
    end
    for safe_name, image in pairs(filename_to_image) do
      ef:write(string.format("%s|exposure=%s|aperture=%s|iso=%s\n",
        safe_name,
        fmt_float(image.exif_exposure),
        fmt_float(image.exif_aperture),
        fmt_float(image.exif_iso)))
    end
    ef:close()
  else
    dlog.msg(dlog.warn, "export_and_detect",
      "Could not write exif_params.txt; Python will read EXIF from the exports")
  end

  -- Call Python script
  local python_script = script_dir .. "auto_negadoctor.py"

  if not df.check_if_file_exists(python_script) then
    dt.print(string.format(_("Python script not found: %s"), python_script))
    return nil, nil, nil
  end

  dt.print(string.format(_("Analyzing %d frame(s) with Python..."), #exported_files))

  local file_args = ""
  for i, image_file in ipairs(exported_files) do
    file_args = file_args .. ' "' .. image_file .. '"'
  end

  local log_file = export_dir .. "/processing.log"
  local debug_flag = debug_ui_mode and " --debug-ui" or ""
  local ai_flag = ai_tune and " --ai-tune" or ""
  local command = string.format('conda run --no-capture-output -n autocrop python -u "%s"%s%s%s',
                                 python_script, debug_flag, ai_flag, file_args)

  -- In debug UI mode, launch Python detached so darktable isn't blocked while the UI is open
  if debug_ui_mode then
    dt.print(_("Running negadoctor analysis in background..."))
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
    dt.print(string.format(_("Analysis started - debug UI will open when it finishes. Log: %s"), log_file))
    return nil, nil, nil
  end

  -- Foreground run with PROGRESS streaming
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
      dt.print(string.format(_("Negadoctor analysis: %s / %s steps done..."), done, total))
    end
  end
  if log_handle then log_handle:close() end
  local pipe_status = {pipe:close()}
  local ok   = pipe_status[1]
  local code = pipe_status[3]
  local exit_code = code or 0
  local result = (ok or exit_code == 0) and 0 or exit_code

  if result ~= 0 then
    dt.print(string.format(_("Negadoctor analysis failed with code: %d. Check log: %s"), result, log_file))
    return nil, nil, nil
  end

  dt.print(string.format(_("Negadoctor analysis completed. Log: %s"), log_file))

  local results_file = export_dir .. "/negadoctor_results.txt"
  local nega_results, vignette = parse_nega_results(results_file)
  if not nega_results then
    dt.print(_("Failed to parse negadoctor results, aborting"))
    return nil, nil, nil, nil
  end

  return nega_results, filename_to_image, export_dir, vignette
end

-- ===================================================================
-- Entry points
-- ===================================================================

-- Mode 1: debug UI (analyze + annotate, no apply). ai_tune adds the vision-LLM
-- alternate variant to the session (switchable in the debug UI with key A).
local function export_and_invert_debug(ai_tune)
  dlog.log_level(dlog.info)
  local images = dt.gui.selection()

  if #images == 0 then
    dt.print(_("No images selected"))
    return
  end

  export_and_detect(images, true, ai_tune)
  -- Detached launch: analysis runs in background and opens debug_ui.py when done
end

-- Modes 2/3: export, analyze, and write negadoctor params into the XMPs.
-- ai_tune writes the vision-LLM nudged params (spec 03) instead of the
-- analytical ones; the analytical pipeline still runs first, unchanged.
local function export_invert_and_apply(keep_temp, ai_tune)
  dlog.log_level(dlog.info)
  math.randomseed(os.time() + os.clock() * 1000)

  local images = dt.gui.selection()

  if #images == 0 then
    dt.print(_("No images selected"))
    return
  end

  local nega_results, filename_to_image, export_dir, vignette = export_and_detect(images, false, ai_tune)
  if not nega_results then
    return
  end

  local stats = { applied = 0, failed = 0, lens_applied = 0, lens_kept = 0 }

  for idx, result_data in ipairs(nega_results) do
    if result_data.status == "success" then
      local original_image = filename_to_image[result_data.filename]
      if original_image then
        dt.print(string.format(_("Applying negadoctor to %s..."), original_image.filename))

        -- vignette correction first (lens module sits before negadoctor in
        -- the pipe; the negadoctor params assume vignette-corrected input)
        if vignette and vignette.params then
          local lens_outcome, lens_err = apply_lens_in_place(original_image, vignette.params)
          if lens_outcome == "applied" then
            stats.lens_applied = stats.lens_applied + 1
          elseif lens_outcome == "kept" then
            stats.lens_kept = stats.lens_kept + 1
          else
            dt.print(string.format(_("  Warning: lens/vignette apply failed: %s"),
              lens_err or "Unknown error"))
          end
        end

        local success, error_msg = apply_negadoctor_in_place(original_image, result_data.params_hex)
        if success then
          stats.applied = stats.applied + 1
        else
          stats.failed = stats.failed + 1
          dt.print(string.format(_("  *** FAILED to apply: %s ***"), error_msg or "Unknown error"))
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

  dt.print(string.format(_("Auto Negadoctor Complete: %d applied, %d failed; lens/vignette: %d applied, %d kept"),
    stats.applied, stats.failed, stats.lens_applied, stats.lens_kept))

  if stats.failed == 0 and not keep_temp then
    df.rmdir(export_dir)
    dlog.msg(dlog.info, "export_invert_and_apply", "Removed temp dir: " .. export_dir)
  else
    dt.print(string.format(_("Temp folder kept for analysis: %s"), export_dir))
  end
end

-- Mode 4: remove negadoctor from the history of all selected frames
local function remove_negadoctor_selected()
  dlog.log_level(dlog.info)
  math.randomseed(os.time() + os.clock() * 1000)

  local images = dt.gui.selection()

  if #images == 0 then
    dt.print(_("No images selected"))
    return
  end

  local stats = { removed = 0, none = 0, failed = 0 }
  for i, image in ipairs(images) do
    local outcome, error_msg = remove_negadoctor_in_place(image)
    if outcome == "removed" then
      stats.removed = stats.removed + 1
    elseif outcome == "none" then
      stats.none = stats.none + 1
    else
      stats.failed = stats.failed + 1
      dt.print(string.format(_("  *** FAILED to remove from %s: %s ***"),
        image.filename, error_msg or "Unknown error"))
    end
  end

  dt.print(string.format(_("Remove Negadoctor: %d removed, %d had none, %d failed"),
    stats.removed, stats.none, stats.failed))
end

-- ===================================================================
-- Registration
-- ===================================================================

local function destroy()
    dt.gui.libs.image.destroy_action("AutoNegadoctor_Debug")
    dt.gui.libs.image.destroy_action("AutoNegadoctor_InPlace")
    dt.gui.libs.image.destroy_action("AutoNegadoctor_InPlace_KeepTemp")
    dt.gui.libs.image.destroy_action("AutoNegadoctor_AI_Debug")
    dt.gui.libs.image.destroy_action("AutoNegadoctor_AI_InPlace")
    dt.gui.libs.image.destroy_action("AutoNegadoctor_Remove")
    -- pcall: darktable throws if the event is already gone (e.g. double-destroy on reload)
    pcall(dt.destroy_event, "AutoNegadoctor_Debug", "shortcut")
    pcall(dt.destroy_event, "AutoNegadoctor_InPlace", "shortcut")
    pcall(dt.destroy_event, "AutoNegadoctor_InPlace_KeepTemp", "shortcut")
    pcall(dt.destroy_event, "AutoNegadoctor_AI_Debug", "shortcut")
    pcall(dt.destroy_event, "AutoNegadoctor_AI_InPlace", "shortcut")
    pcall(dt.destroy_event, "AutoNegadoctor_Remove", "shortcut")
end

-- Mode 1: debug UI (analyze + annotate, no apply)
dt.gui.libs.image.register_action(
    "AutoNegadoctor_Debug",
    _("Auto negadoctor debug (open debug UI, no apply)"),
    function() export_and_invert_debug() end,
    _("Export, analyze the roll and open the negadoctor debug UI - nothing applied")
)

dt.register_event(
    "AutoNegadoctor_Debug",
    "shortcut",
    function(event, shortcut) export_and_invert_debug() end,
    "AutoNegadoctor_Debug"
)

-- Mode 2: fully automatic, temp folder removed on success
dt.gui.libs.image.register_action(
    "AutoNegadoctor_InPlace",
    _("Auto negadoctor in-place"),
    function() export_invert_and_apply(false) end,
    _("Export, analyze the roll, and write negadoctor params into the selected images' XMPs")
)

dt.register_event(
    "AutoNegadoctor_InPlace",
    "shortcut",
    function(event, shortcut) export_invert_and_apply(false) end,
    "AutoNegadoctor_InPlace"
)

-- Mode 3: fully automatic, temp folder kept for analysis
dt.gui.libs.image.register_action(
    "AutoNegadoctor_InPlace_KeepTemp",
    _("Auto negadoctor in-place (keep temp folder)"),
    function() export_invert_and_apply(true) end,
    _("Same as in-place, but keeps the temp folder and log for analysis")
)

dt.register_event(
    "AutoNegadoctor_InPlace_KeepTemp",
    "shortcut",
    function(event, shortcut) export_invert_and_apply(true) end,
    "AutoNegadoctor_InPlace_KeepTemp"
)

-- Spec 03: AI debug — same as Debug, plus the vision-LLM (gemma3/Ollama)
-- alternate variant in the session; switch Analytical<->AI in the UI with key A.
dt.gui.libs.image.register_action(
    "AutoNegadoctor_AI_Debug",
    _("Auto negadoctor AI debug (vision-LLM variant, open debug UI)"),
    function() export_and_invert_debug(true) end,
    _("Like the debug action, but also computes the vision-LLM per-scene variant for A/B comparison (slower)")
)

dt.register_event(
    "AutoNegadoctor_AI_Debug",
    "shortcut",
    function(event, shortcut) export_and_invert_debug(true) end,
    "AutoNegadoctor_AI_Debug"
)

-- Spec 03: AI in-place — writes the vision-LLM nudged params into the XMPs.
-- The full analytical pipeline still runs first; AI only nudges the result.
dt.gui.libs.image.register_action(
    "AutoNegadoctor_AI_InPlace",
    _("Auto negadoctor AI in-place (vision-LLM variant)"),
    function() export_invert_and_apply(false, true) end,
    _("Export, analyze, then apply the vision-LLM per-scene tuned params (gemma3/Ollama) to the XMPs")
)

dt.register_event(
    "AutoNegadoctor_AI_InPlace",
    "shortcut",
    function(event, shortcut) export_invert_and_apply(false, true) end,
    "AutoNegadoctor_AI_InPlace"
)

-- Mode 4: remove negadoctor from history (clean slate for re-runs)
dt.gui.libs.image.register_action(
    "AutoNegadoctor_Remove",
    _("Remove negadoctor from history"),
    function() remove_negadoctor_selected() end,
    _("Strip negadoctor entries from the selected images' history for a fresh run")
)

dt.register_event(
    "AutoNegadoctor_Remove",
    "shortcut",
    function(event, shortcut) remove_negadoctor_selected() end,
    "AutoNegadoctor_Remove"
)

script_data.destroy = destroy

return script_data
