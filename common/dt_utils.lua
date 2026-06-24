--[[
Shared darktable-Lua utilities for the auto_* plugins (auto_negadoctor,
auto_retouch, auto_crop).

These stateless XMP / binary-blob helpers were copy-pasted (byte-identical, bar a
stray log line) across all three feature scripts. They are pure Lua
(string.pack/unpack, Lua 5.3+) with NO darktable API or feature state, so this
module is safe to `require` from any plugin and shared once via require's cache.

Usage (each feature, right after its `script_dir` is computed):
    package.path = package.path .. ";" .. script_dir .. "../common/?.lua"
    local dtu = require("dt_utils")
    ... dtu.generate_random_hex(8) ...

Mirrors how the Python side shares code in common/ — keep feature-specific logic
(module XML builders, the export/apply pipelines, registration) in each plugin.
]]

local M = {}

-- A random lowercase-hex string of `length` nibbles (darktable mask / module ids).
function M.generate_random_hex(length)
  local hex = ""
  for i = 1, length do
    hex = hex .. string.format("%x", math.random(0, 15))
  end
  return hex
end

-- darktable stores timestamps as microseconds since 0001-01-01 (proleptic).
function M.generate_darktable_timestamp()
  return (os.time() + 62135596800) * 1000000
end

-- Highest darktable:num="N" across an XMP's history items (-1 if none).
function M.find_max_history_num(xmp_content)
  local max_num = -1
  for num_str in xmp_content:gmatch('darktable:num="(%d+)"') do
    local num = tonumber(num_str)
    if num and num > max_num then
      max_num = num
    end
  end
  return max_num
end

-- 32-bit float -> little-endian hex (darktable params blobs).
function M.float_to_le_hex(value)
  local packed = string.pack("<f", value)
  return (packed:gsub(".", function(c) return string.format("%02x", string.byte(c)) end))
end

-- Inverse of float_to_le_hex: little-endian hex -> 32-bit float.
function M.le_hex_to_float(hex_str)
  local bytes = {}
  for i = 1, #hex_str, 2 do
    bytes[#bytes + 1] = string.char(tonumber(hex_str:sub(i, i + 1), 16))
  end
  return string.unpack("<f", table.concat(bytes))
end

-- Write `source_paths.txt` (a `stem|/full/source/path` manifest) into `export_dir`,
-- so a temp export folder records where its frames came from and downstream tools
-- (e.g. calibration GT) can key on the full source path instead of the collision-
-- prone stem. `source_paths` is a {safe_name = src_path} table. Returns true on
-- success, false if the file couldn't be opened (caller logs — this stays
-- dependency-free of the feature's dlog).
function M.write_source_paths(export_dir, source_paths)
  local sf = io.open(export_dir .. "/source_paths.txt", "w")
  if not sf then
    return false
  end
  for safe_name, src_path in pairs(source_paths) do
    sf:write(string.format("%s|%s\n", safe_name, src_path))
  end
  sf:close()
  return true
end

return M
