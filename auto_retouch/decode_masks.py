import struct
import base64
import zlib

def parse_brush_point(data, offset):
    corner_x, corner_y = struct.unpack_from('<ff', data, offset); offset += 8
    ctrl1_x,  ctrl1_y  = struct.unpack_from('<ff', data, offset); offset += 8
    ctrl2_x,  ctrl2_y  = struct.unpack_from('<ff', data, offset); offset += 8
    border_x, border_y = struct.unpack_from('<ff', data, offset); offset += 8
    hardness            = struct.unpack_from('<f',  data, offset)[0]; offset += 4
    density             = struct.unpack_from('<f',  data, offset)[0]; offset += 4
    state               = struct.unpack_from('<i',  data, offset)[0]; offset += 4
    return dict(corner_x=corner_x, corner_y=corner_y,
                ctrl1_x=ctrl1_x,   ctrl1_y=ctrl1_y,
                ctrl2_x=ctrl2_x,   ctrl2_y=ctrl2_y,
                border_x=border_x, border_y=border_y,
                hardness=hardness,  density=density, state=state)

def parse_brush(hex_pts, hex_src, label):
    pts_data = bytes.fromhex(hex_pts)
    src_data = bytes.fromhex(hex_src)
    assert len(pts_data) == 88, f"{label}: expected 88 bytes, got {len(pts_data)}"
    assert len(src_data) == 8,  f"{label}: expected 8 bytes src, got {len(src_data)}"
    p0 = parse_brush_point(pts_data, 0)
    p1 = parse_brush_point(pts_data, 44)
    src_x, src_y = struct.unpack('<ff', src_data)
    return p0, p1, src_x, src_y

def parse_group_child(data, offset):
    child_id  = struct.unpack_from('<i', data, offset)[0]; offset += 4
    group_id  = struct.unpack_from('<i', data, offset)[0]; offset += 4
    flags     = struct.unpack_from('<i', data, offset)[0]; offset += 4
    opacity   = struct.unpack_from('<f', data, offset)[0]; offset += 4
    return child_id, group_id, flags, opacity

# ── GROUP 1 brushes ──────────────────────────────────────────────────────────
group1_brushes = [
    ("G1-B1", "1771417258",
     "841f1b3f10f3d13e521f1b3facf2d13eb71f1b3f75f3d13e4320173b4320173b0000803fc3f5283f010000002c201b3f60f4d13efa1f1b3ffbf3d13e5f201b3fc5f4d13e4320173b4320173b0000803fc3f5283f01000000",
     "e1ae1d3f58d4cc3e"),
    ("G1-B2", "1771417259",
     "80b4dc3dedbb583feeb2dc3dbbbb583f13b6dc3d1fbc583f4320173b4320173b0000803fc3f5283f01000000beb9dc3d95bc583f2cb8dc3d62bc583f51bbdc3dc7bc583f4320173b4320173b0000803fc3f5283f01000000",
     "612ff13d912c563f"),
    ("G1-B3", "1771417260",
     "84bc7d3df79f683f5fb97d3dc59f683fa9bf7d3d2aa0683f4320173b4320173b0000803fc3f5283f0100000000c77d3d9fa0683fdbc37d3d6da0683f26ca7d3dd1a0683f4320173b4320173b0000803fc3f5283f01000000",
     "2359933d9b10663f"),
    ("G1-B4", "1771417261",
     "bbcb103f5c72683f89cb103f2a72683fedcb103f8e72683f4320173b4320173b0000803fc3f5283f0100000063cc103f0473683f30cc103fd172683f95cc103f3673683f4320173b4320173b0000803fc3f5283f01000000",
     "175b133f00e3653f"),
    ("G1-B5", "1771417262",
     "80956c3eec4d623fb6946c3eba4d623f49966c3e1e4e623f4320173b4320173b0000803fc3f5283f010000001f986c3e944e623f55976c3e624e623fe8986c3ec64e623f4320173b4320173b0000803fc3f5283f01000000",
     "f0d2763e90be5f3f"),
    ("G1-B6", "1771417263",
     "57b46a3ecc8d443f8eb36a3e9a8d443f20b56a3eff8d443f4320173b4320173b0000803fc3f5283f01000000f6b66a3e748e443f2db66a3e428e443fbfb76a3ea78e443f4320173b4320173b0000803fc3f5283f01000000",
     "c8f1743e70fe413f"),
    ("G1-B7", "1771417264",
     "53278c3e8fc2273fee268c3e5dc2273fb8278c3ec1c2273f4320173b4320173b0000803fc3f5283f01000000a3288c3e37c3273f3e288c3e04c3273f07298c3e69c3273f4320173b4320173b0000803fc3f5283f01000000",
     "0b46913e3333253f"),
    ("G1-B8", "1771417265",
     "36bc1e3e1d9a803d6cbb1e3e8b98803dffbc1e3eb09b803d4320173b4320173b0000803fc3f5283f01000000d5be1e3e5c9f803d0bbe1e3ec99d803d9ebf1e3eeea0803d4320173b4320173b0000803fc3f5283f01000000",
     "a6f9283e783e583d"),
]

# ── GROUP 2 brushes ──────────────────────────────────────────────────────────
group2_brushes = [
    ("G2-B1", "1772233061",
     "a209413f740b383e6f09413fab0a383ed409413f3e0c383eeba0653aeba0653a0000803fc3f5283f01000000490a413f130e383e170a413f4a0d383e7c0a413fdd0e383eeba0653aeba0653a0000803fc3f5283f01000000",
     "be6d403f59b6393e"),
    ("G2-B2", "1772233062",
     "7356453f9595133e4056453fcb94133ea556453f5e96133eeba0653aeba0653a0000803fc3f5283f010000001b57453f3498133ee856453f6a97133e4d57453ffd98133eeba0653aeba0653a0000803fc3f5283f01000000",
     "ef64453f13bd0b3e"),
]

all_results = {}

print("=" * 110)
print(f"{'Name':<8} {'form_id':<12} {'corner_x':>10} {'corner_y':>10} {'src_x':>10} {'src_y':>10} {'radius':>10} {'hard':>6} {'dens':>6}  pt0->pt1 delta")
print("=" * 110)

for brushes in [group1_brushes, group2_brushes]:
    for (label, form_id, hex_pts, hex_src) in brushes:
        p0, p1, src_x, src_y = parse_brush(hex_pts, hex_src, label)
        radius = p0['border_x']
        dx = p1['corner_x'] - p0['corner_x']
        dy = p1['corner_y'] - p0['corner_y']
        delta_str = f"dx={dx:+.6f} dy={dy:+.6f}" if (abs(dx) > 1e-7 or abs(dy) > 1e-7) else "(identical)"
        print(f"{label:<8} {form_id:<12} {p0['corner_x']:>10.6f} {p0['corner_y']:>10.6f} {src_x:>10.6f} {src_y:>10.6f} {radius:>10.6f} {p0['hardness']:>6.4f} {p0['density']:>6.4f}  {delta_str}")
        all_results[int(form_id)] = (label, p0, src_x, src_y)
    print("-" * 110)

print("\nNotes:")
print("  G1 brushes: radius=0.002306 (larger spot), G2 brushes: radius=0.000876 (smaller spot)")
print("  All brushes: hardness=1.0, density=0.66")
print("  All pt0->pt1 delta: dx=+1e-5, dy=+1e-5 (near-zero = dot click, not a stroke)")
print("  G1 src offset from corner: +0.010 in x, -0.010 in y (heal source 10% away diagonally)")
print("  G2 src offset from corner: tiny, ~0.001-0.003 (very close source)")

# ── GROUP 1 compressed children ──────────────────────────────────────────────
print("\n=== GROUP 1 children (compressed, id=1771417266) ===")
gz03_b64 = "eJxbtW5q5iYgZmYAgQb71VA+N5S/Bo2/Fo2/Do2/Ho2/AY2/EY0PAIjtLAU="
raw = base64.b64decode(gz03_b64)
decompressed = zlib.decompress(raw)
n_children = len(decompressed) // 16
print(f"Decompressed {len(raw)} -> {len(decompressed)} bytes = {n_children} children x 16 bytes")
print(f"  {'#':<3} {'child_id':<14} {'group_id':<14} {'flags':<8} {'opacity':<8}  label")
for i in range(n_children):
    child_id, group_id, flags, opacity = parse_group_child(decompressed, i * 16)
    name = all_results.get(child_id, (f"unknown",))[0]
    print(f"  {i:<3} {child_id:<14} {group_id:<14} {flags:<8} {opacity:<8.4f}  {name}")
print("  flags 3 = first child (head); flags 11 = subsequent children")

# ── GROUP 2 raw hex children ──────────────────────────────────────────────────
print("\n=== GROUP 2 children (raw hex, id=1772233063) ===")
raw_hex = "6521a2696721a269030000000000803f6621a2696721a2690b0000000000803f"
data = bytes.fromhex(raw_hex)
n_children = len(data) // 16
print(f"Raw {len(data)} bytes = {n_children} children x 16 bytes")
print(f"  {'#':<3} {'child_id':<14} {'group_id':<14} {'flags':<8} {'opacity':<8}  label")
for i in range(n_children):
    child_id, group_id, flags, opacity = parse_group_child(data, i * 16)
    name = all_results.get(child_id, (f"unknown",))[0]
    print(f"  {i:<3} {child_id:<14} {group_id:<14} {flags:<8} {opacity:<8.4f}  {name}")
print("  flags 3 = first child (head); flags 11 = subsequent children")

# ── RETOUCH PARAMS ────────────────────────────────────────────────────────────
# Confirmed layout from raw scan:
#   - Blob starts directly with forms array (no leading num_forms field)
#   - Form stride = 44 bytes (G2-B1 at offset 0, G2-B2 at offset 44)
#   - RETOUCH_NO_FORMS = 300 -> 300 * 44 = 13200 bytes for forms array
#   - Total decompressed = 13260 -> tail = 60 bytes for global params
#
# dt_iop_retouch_form_data_t (44 bytes):
#   int32  formid             4    offset 0
#   int32  scale              4    offset 4
#   int32  algorithm          4    offset 8  (1=heal,2=clone,3=fill,4=blur)
#   int32  blur_type          4    offset 12
#   float  blur_radius        4    offset 16
#   int32  fill_mode          4    offset 20
#   float  fill_color[3]     12    offset 24
#   float  fill_brightness    4    offset 36
#   int32  distort_mode       4    offset 40
#   = 44 bytes total
#
# Tail (60 bytes) at offset 13200:
#   int32  algorithm (global)    4
#   int32  num_scales            4
#   int32  curr_scale            4
#   int32  merge_from_scale      4
#   float  preview_levels[3]    12
#   int32  blur_type             4
#   float  blur_radius           4
#   int32  fill_mode             4
#   float  fill_color[3]        12
#   float  fill_brightness       4
#   int32  max_heal_iter         4
#   = 60 bytes total  CHECK

RETOUCH_NO_FORMS = 300
FORM_SIZE = 44
algo_names = {0: "none", 1: "heal", 2: "clone", 3: "fill", 4: "blur"}

print("\n=== RETOUCH PARAMS (history num=17) ===")
rt_b64 = "eJzt27ENgDAMBEArDWvAJmGYILEWk2QERgIEXSIkWnTXfOH/EVymbY1TinfXffnQBQAAAAAAAAAAAAAAgD9qf2dyfTLfOc693T5EHFBKBeY="
raw_rt = base64.b64decode(rt_b64)
dec_rt  = zlib.decompress(raw_rt)
print(f"Decompressed {len(raw_rt)} -> {len(dec_rt)} bytes")
print(f"Forms array: {RETOUCH_NO_FORMS} slots x {FORM_SIZE} bytes = {RETOUCH_NO_FORMS*FORM_SIZE} bytes")
print(f"Tail: {len(dec_rt) - RETOUCH_NO_FORMS*FORM_SIZE} bytes")

print(f"\n  {'#':<4} {'form_id':<14} {'scale':<6} {'algorithm':<8} {'blur_type':<10} {'blur_r':<8} {'fill_mode':<10} {'distort':<8}  label")
active_forms = []
for i in range(RETOUCH_NO_FORMS):
    base = i * FORM_SIZE
    form_id      = struct.unpack_from('<i', dec_rt, base +  0)[0]
    scale        = struct.unpack_from('<i', dec_rt, base +  4)[0]
    algo         = struct.unpack_from('<i', dec_rt, base +  8)[0]
    blur_type    = struct.unpack_from('<i', dec_rt, base + 12)[0]
    blur_radius  = struct.unpack_from('<f', dec_rt, base + 16)[0]
    fill_mode    = struct.unpack_from('<i', dec_rt, base + 20)[0]
    fill_r       = struct.unpack_from('<f', dec_rt, base + 24)[0]
    fill_g       = struct.unpack_from('<f', dec_rt, base + 28)[0]
    fill_b       = struct.unpack_from('<f', dec_rt, base + 32)[0]
    fill_bright  = struct.unpack_from('<f', dec_rt, base + 36)[0]
    distort_mode = struct.unpack_from('<i', dec_rt, base + 40)[0]
    if form_id != 0:
        name = all_results.get(form_id, ("unknown",))[0]
        print(f"  {i:<4} {form_id:<14} {scale:<6} {algo_names.get(algo,str(algo)):<8} {blur_type:<10} {blur_radius:<8.3f} {fill_mode:<10} {distort_mode:<8}  {name}")
        active_forms.append((form_id, algo_names.get(algo, str(algo)), scale, name))

print(f"\n  Active forms: {len(active_forms)}")

# Parse global tail params
tail_off = RETOUCH_NO_FORMS * FORM_SIZE
t = tail_off
algo_global    = struct.unpack_from('<i', dec_rt, t+ 0)[0]
num_scales     = struct.unpack_from('<i', dec_rt, t+ 4)[0]
curr_scale     = struct.unpack_from('<i', dec_rt, t+ 8)[0]
merge_from     = struct.unpack_from('<i', dec_rt, t+12)[0]
prev_lev       = struct.unpack_from('<fff', dec_rt, t+16)
blur_type_g    = struct.unpack_from('<i', dec_rt, t+28)[0]
blur_radius_g  = struct.unpack_from('<f', dec_rt, t+32)[0]
fill_mode_g    = struct.unpack_from('<i', dec_rt, t+36)[0]
fill_color_g   = struct.unpack_from('<fff', dec_rt, t+40)
fill_bright_g  = struct.unpack_from('<f', dec_rt, t+52)[0]
max_heal_iter  = struct.unpack_from('<i', dec_rt, t+56)[0]

print(f"\n  Global params (tail at offset {tail_off}):")
print(f"    algorithm (default): {algo_names.get(algo_global, algo_global)}")
print(f"    num_scales:          {num_scales}")
print(f"    curr_scale:          {curr_scale}")
print(f"    merge_from_scale:    {merge_from}")
print(f"    preview_levels:      {prev_lev}")
print(f"    blur_type:           {blur_type_g}")
print(f"    blur_radius:         {blur_radius_g:.3f}")
print(f"    fill_mode:           {fill_mode_g}")
print(f"    fill_color:          ({fill_color_g[0]:.3f}, {fill_color_g[1]:.3f}, {fill_color_g[2]:.3f})")
print(f"    fill_brightness:     {fill_bright_g:.3f}")
print(f"    max_heal_iter:       {max_heal_iter}")

print("\n=== SUMMARY ===")
print(f"  Group 1 (id=1771417266): 8 brush masks, radius=0.002306, hardness=1.0, density=0.66")
for label, form_id, hex_pts, hex_src in group1_brushes:
    p0, p1, sx, sy = parse_brush(hex_pts, hex_src, label)
    print(f"    {label}: corner=({p0['corner_x']:.6f}, {p0['corner_y']:.6f})  src=({sx:.6f}, {sy:.6f})")

print(f"\n  Group 2 (id=1772233063): 2 brush masks, radius=0.000876, hardness=1.0, density=0.66")
for label, form_id, hex_pts, hex_src in group2_brushes:
    p0, p1, sx, sy = parse_brush(hex_pts, hex_src, label)
    print(f"    {label}: corner=({p0['corner_x']:.6f}, {p0['corner_y']:.6f})  src=({sx:.6f}, {sy:.6f})")

print(f"\n  Retouch params: {len(active_forms)} active forms (only G2 brushes in this history entry)")
for fid, algo, scale, name in active_forms:
    print(f"    {name} (id={fid}): algo={algo}, scale={scale}")
