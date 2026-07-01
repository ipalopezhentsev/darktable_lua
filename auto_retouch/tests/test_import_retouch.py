"""Round-trip test for import_retouch.decode_xmp_masks.

Encode a known spot set with detect_dust.generate_xmp_data_for_spots, synthesize
the minimal XMP darktable would write (a retouch history entry + brush mask
entries, plus a SENSOR-dust instance that must be EXCLUDED), then decode it back
and assert positions / radii / sources / kinds survive across flip + crop (+
ashift) — i.e. the importer is a faithful inverse of the exporter.

Run: conda run -n autocrop python tests/test_import_retouch.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import detect_dust
import import_retouch

SENSOR_LABEL = "sensor dust"


def _mask_entries_xml(xmp_data, mask_num, mask_type=72):
    """Mimic the Lua create_mask_entries_xml brush+group entries."""
    parts = []
    for i, b in enumerate(xmp_data["brushes"]):
        parts.append(
            f'     <rdf:li darktable:mask_num="{mask_num}" '
            f'darktable:mask_id="{b["mask_id"]}" darktable:mask_type="{mask_type}" '
            f'darktable:mask_name="brush #{i}" darktable:mask_version="6" '
            f'darktable:mask_points="{b["mask_points"]}" '
            f'darktable:mask_nb="{b["mask_nb"]}" '
            f'darktable:mask_src="{b["mask_src"]}"/>')
    g = xmp_data["group"]
    parts.append(
        f'     <rdf:li darktable:mask_num="{mask_num}" '
        f'darktable:mask_id="{g["mask_id"]}" darktable:mask_type="12" '
        f'darktable:mask_name="group" darktable:mask_version="6" '
        f'darktable:mask_points="{g["mask_points"]}" '
        f'darktable:mask_nb="{g["mask_nb"]}" '
        f'darktable:mask_src="{g["mask_src"]}"/>')
    return "\n".join(parts)


def _history_entry(num, params, blendop, multi_priority, multi_name, enabled=1):
    return (
        f'     <rdf:li darktable:num="{num}" darktable:operation="retouch" '
        f'darktable:enabled="{enabled}" darktable:modversion="3" '
        f'darktable:params="{params}" darktable:multi_name="{multi_name}" '
        f'darktable:multi_name_hand_edited="1" '
        f'darktable:multi_priority="{multi_priority}" '
        f'darktable:blendop_version="14" darktable:blendop_params="{blendop}"/>')


def _build_xmp(dust_xmp, sensor_xmp, history_end=9):
    masks = _mask_entries_xml(dust_xmp, mask_num=2)
    masks += "\n" + _mask_entries_xml(sensor_xmp, mask_num=3)
    history = "\n".join([
        _history_entry(2, dust_xmp["retouch_params"], dust_xmp["blendop_params"],
                       0, "film dust"),
        _history_entry(3, sensor_xmp["retouch_params"], sensor_xmp["blendop_params"],
                       1, SENSOR_LABEL),
    ])
    return (
        '<?xml version="1.0"?>\n<rdf:RDF>\n'
        f'<rdf:Description darktable:history_end="{history_end}">\n'
        '<darktable:masks_history><rdf:Seq>\n' + masks + '\n</rdf:Seq></darktable:masks_history>\n'
        '<darktable:history><rdf:Seq>\n' + history + '\n</rdf:Seq></darktable:history>\n'
        '</rdf:Description>\n</rdf:RDF>\n')


def _match(spots, target, tol):
    """Find a spot near target=(cx,cy) within tol; return it or None."""
    best, bd = None, tol
    for s in spots:
        if s.get("kind") != "dot":
            continue
        d = max(abs(s["cx"] - target[0]), abs(s["cy"] - target[1]))
        if d <= bd:
            best, bd = s, d
    return best


def run_case(flip, crop, ashift, label):
    W, H = 4000, 3000
    dust_spots = [
        {"kind": "dot", "cx": 1000.0, "cy": 800.0, "brush_radius_px": 14.0,
         "src_cx": 1080.0, "src_cy": 760.0},
        {"kind": "dot", "cx": 2500.0, "cy": 1900.0, "brush_radius_px": 9.0,
         "src_cx": 2560.0, "src_cy": 1850.0},
        {"kind": "stroke", "brush_radius_px": 7.0,
         "path": [[1500.0, 1000.0], [1600.0, 1100.0], [1720.0, 1180.0]],
         "src_cx": 1450.0, "src_cy": 950.0},
    ]
    sensor_spots = [
        {"kind": "dot", "cx": 300.0, "cy": 300.0, "brush_radius_px": 6.0},
    ]
    import copy
    dust_xmp = detect_dust.generate_xmp_data_for_spots(
        copy.deepcopy(dust_spots), W, H, flip=flip, crop=crop, ashift_params=ashift)
    sensor_xmp = detect_dust.generate_xmp_data_for_spots(
        copy.deepcopy(sensor_spots), W, H, flip=flip, crop=crop, ashift_params=ashift)
    xmp = _build_xmp(dust_xmp, sensor_xmp)

    transform = {"flip": flip, "crop": crop, "ashift": ashift}
    decoded = import_retouch.decode_xmp_masks(
        xmp, transform, W, H, sensor_label=SENSOR_LABEL)

    fails = []
    dots = [s for s in decoded if s.get("kind") == "dot"]
    strokes = [s for s in decoded if s.get("kind") == "stroke"]
    if len(dots) != 2:
        fails.append(f"expected 2 dots, got {len(dots)}")
    if len(strokes) != 1:
        fails.append(f"expected 1 stroke, got {len(strokes)}")

    tol = 1.5  # px
    for sp in dust_spots:
        if sp["kind"] != "dot":
            continue
        m = _match(dots, (sp["cx"], sp["cy"]), tol)
        if m is None:
            fails.append(f"dot near ({sp['cx']},{sp['cy']}) not recovered")
            continue
        if abs(m["brush_radius_px"] - sp["brush_radius_px"]) > 0.5:
            fails.append(f"dot radius {m['brush_radius_px']:.2f} != {sp['brush_radius_px']}")
        if "src_cx" not in m or abs(m["src_cx"] - sp["src_cx"]) > tol:
            fails.append(f"dot src x {m.get('src_cx')} != {sp['src_cx']}")
        if "src_cy" not in m or abs(m["src_cy"] - sp["src_cy"]) > tol:
            fails.append(f"dot src y {m.get('src_cy')} != {sp['src_cy']}")

    if strokes:
        st = strokes[0]
        orig = dust_spots[2]["path"]
        if len(st.get("path") or []) != len(orig):
            fails.append(f"stroke node count {len(st.get('path') or [])} != {len(orig)}")
        else:
            for (px, py), (ox, oy) in zip(st["path"], orig):
                if abs(px - ox) > tol or abs(py - oy) > tol:
                    fails.append(f"stroke node ({px:.1f},{py:.1f}) != ({ox},{oy})")
                    break
        if abs(st["brush_radius_px"] - 7.0) > 0.5:
            fails.append(f"stroke radius {st['brush_radius_px']:.2f} != 7.0")

    # the sensor-dust dot must NOT be imported
    if _match(dots, (300.0, 300.0), 5.0) is not None:
        fails.append("sensor-dust form was imported (should be excluded)")

    status = "OK" if not fails else "FAIL"
    print(f"[{status}] {label}")
    for f in fails:
        print(f"        - {f}")
    return not fails


def test_change_detection():
    base = [{"kind": "dot", "cx": 100.0, "cy": 100.0, "brush_radius_px": 10.0},
            {"kind": "dot", "cx": 500.0, "cy": 400.0, "brush_radius_px": 8.0}]
    same = [dict(s) for s in base]
    fails = []
    if import_retouch.spots_differ(base, same, 3000):
        fails.append("identical sets reported as differing")
    moved = [dict(s) for s in base]
    moved[0] = dict(moved[0], cx=900.0)
    if not import_retouch.spots_differ(base, moved, 3000):
        fails.append("moved spot not detected as a change")
    added = same + [{"kind": "dot", "cx": 700.0, "cy": 700.0, "brush_radius_px": 5.0}]
    if not import_retouch.spots_differ(base, added, 3000):
        fails.append("added spot not detected as a change")
    status = "OK" if not fails else "FAIL"
    print(f"[{status}] change detection (spots_differ)")
    for f in fails:
        print(f"        - {f}")
    return not fails


def test_seed_annotations():
    """End-to-end seeding: synthesize an export dir (JPG + transform_params +
    source_xmp + a source XMP) and assert seed_import_annotations writes the
    expected missed_dust / missed_strokes annotation files + import_baseline."""
    import copy
    import json
    import tempfile
    from PIL import Image

    fails = []
    W, H = 800, 600
    dust_spots = [
        {"kind": "dot", "cx": 200.0, "cy": 150.0, "brush_radius_px": 6.0,
         "src_cx": 220.0, "src_cy": 140.0},
        {"kind": "dot", "cx": 500.0, "cy": 400.0, "brush_radius_px": 5.0},
        {"kind": "stroke", "brush_radius_px": 4.0,
         "path": [[300.0, 200.0], [340.0, 240.0], [380.0, 300.0]]},
    ]
    sensor_spots = [{"kind": "dot", "cx": 60.0, "cy": 60.0, "brush_radius_px": 3.0}]

    dust_xmp = detect_dust.generate_xmp_data_for_spots(copy.deepcopy(dust_spots), W, H)
    sensor_xmp = detect_dust.generate_xmp_data_for_spots(copy.deepcopy(sensor_spots), W, H)
    xmp_text = _build_xmp(dust_xmp, sensor_xmp)

    with tempfile.TemporaryDirectory() as d:
        stem = "DSC_TEST"
        Image.new("RGB", (W, H), (128, 128, 128)).save(os.path.join(d, f"{stem}.jpg"))
        with open(os.path.join(d, f"{stem}.xmp"), "w", encoding="utf-8") as f:
            f.write(xmp_text)
        with open(os.path.join(d, "transform_params.txt"), "w") as f:
            f.write(f"{stem}|flip=0|crop=0.000000,0.000000,1.000000,1.000000\n")
        with open(os.path.join(d, "source_xmp.txt"), "w", encoding="utf-8") as f:
            f.write(f"SENSOR_LABEL|{SENSOR_LABEL}\n")
            f.write("DUST_LABEL|film dust\n")
            f.write(f"{stem}|{os.path.join(d, f'{stem}.xmp')}\n")

        baseline = import_retouch.seed_import_annotations(d)

        ann_path = os.path.join(d, f"{stem}_annotations.json")
        if not os.path.isfile(ann_path):
            fails.append("seed annotation file not written")
        else:
            with open(ann_path) as f:
                ann = json.load(f)
            if len(ann.get("missed_dust") or []) != 2:
                fails.append(f"expected 2 missed_dust, got {len(ann.get('missed_dust') or [])}")
            if len(ann.get("missed_strokes") or []) != 1:
                fails.append(f"expected 1 missed_stroke, got {len(ann.get('missed_strokes') or [])}")
            # the sensor-dust spot at (60,60) must not be seeded
            for md in ann.get("missed_dust") or []:
                if abs(md["cx"] - 60.0) < 5 and abs(md["cy"] - 60.0) < 5:
                    fails.append("sensor-dust seeded as missed_dust (should be excluded)")
        if stem not in baseline or len(baseline[stem]) != 3:
            fails.append(f"import_baseline missing/short: {len(baseline.get(stem, []))} spots")
        if not os.path.isfile(os.path.join(d, "import_baseline.json")):
            fails.append("import_baseline.json not written")

    status = "OK" if not fails else "FAIL"
    print(f"[{status}] seed_import_annotations end-to-end")
    for f in fails:
        print(f"        - {f}")
    return not fails


def main():
    ashift = {"rotation": 0.4, "lensshift_v": 0.0, "lensshift_h": 0.0, "shear": 0.0,
              "f_length_kb": 24.0, "orthocorr": 0.0, "aspect": 1.0,
              "cl": 0.01, "cr": 0.99, "ct": 0.01, "cb": 0.99}
    ok = True
    ok &= run_case(0, (0.0, 0.0, 1.0, 1.0), None, "identity")
    ok &= run_case(0, (0.05, 0.04, 0.95, 0.96), None, "crop only")
    ok &= run_case(5, (0.02, 0.03, 0.97, 0.98), None, "flip=5 (CCW90) + crop")
    ok &= run_case(6, (0.0, 0.0, 1.0, 1.0), None, "flip=6 (CW90)")
    ok &= run_case(0, (0.02, 0.02, 0.98, 0.98), ashift, "ashift + crop")
    ok &= test_change_detection()
    ok &= test_seed_annotations()
    print("\n" + ("ALL PASSED" if ok else "SOME FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
