"""
Analyze and compare dust detection debug reports.

Usage:
    analyze_debug_report.py <debug_dir>
        Print summary of annotations (FPs, missed) and filter statistics.

    analyze_debug_report.py <debug_dir> --find <cx> <cy> [--image <stem>] [--radius <px>]
        Find rejection reason for a specific coordinate in the rejected candidates.

    analyze_debug_report.py <debug_dir1> <debug_dir2>
        Compare two runs: show what changed (new/fixed FPs, new/fixed misses).

The debug_dir must contain:
    debug_report.txt  — annotations (FPs and missed spots) per image
    debug_spots.json  — full rejected candidates list with reasons
"""

import sys
import os
import json
import math
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def load_json(debug_dir):
    path = Path(debug_dir) / "debug_spots.json"
    if not path.exists():
        print(f"[warn] No debug_spots.json in {debug_dir}")
        return None
    with open(path) as f:
        return json.load(f)


def load_report(debug_dir):
    """Parse debug_report.txt.

    Returns dict: stem -> {"detected": int, "fps": list, "missed": list}
    and a "constants" dict.
    """
    path = Path(debug_dir) / "debug_report.txt"
    if not path.exists():
        print(f"[warn] No debug_report.txt in {debug_dir}")
        return {}, {}

    images = {}
    constants = {}
    current = None
    mode = None  # "fp", "missed", "rejected", "constants"

    with open(path) as f:
        for line in f:
            line = line.rstrip()

            if line.strip().startswith("Detection constants"):
                mode = "constants"
                continue
            if re.match(r"^=+$", line.strip()):
                current = None
                mode = None
                continue

            if mode == "constants":
                m = re.match(r"\s+(\w+)\s*=\s*(-?[\d.]+)", line)
                if m:
                    try:
                        val = float(m.group(2))
                        constants[m.group(1)] = val
                    except ValueError:
                        pass
                elif line.strip() == "":
                    mode = None
                continue

            m = re.match(r"IMAGE: (\S+)\s+\((\d+) x (\d+) px\)", line.strip())
            if m:
                stem = m.group(1)
                current = {"stem": stem, "width": int(m.group(2)), "height": int(m.group(3)),
                           "detected": 0, "fps": [], "missed": []}
                images[stem] = current
                mode = None
                continue

            if current is None:
                continue

            m = re.match(r"\s+Detected:\s*(\d+)", line)
            if m:
                current["detected"] = int(m.group(1))
                continue

            if "FALSE POSITIVES" in line and "none" not in line.lower():
                mode = "fp"
                continue
            if "FALSE POSITIVES: none" in line:
                mode = None
                continue
            if "MISSED DUST" in line and "none" not in line.lower():
                mode = "missed"
                continue
            if "MISSED DUST: none" in line:
                mode = None
                continue
            if "REJECTED CANDIDATES" in line:
                mode = "rejected"
                continue

            if mode == "fp":
                m = re.match(r"\s+cx=([\d.]+)\s+cy=([\d.]+)", line)
                if m:
                    current["fps"].append({"cx": float(m.group(1)), "cy": float(m.group(2))})
                elif line.strip() == "":
                    mode = None
            elif mode == "missed":
                m = re.match(r"\s+cx=(\d+)\s+cy=(\d+)", line)
                if m:
                    current["missed"].append({"cx": int(m.group(1)), "cy": int(m.group(2))})
                elif line.strip() == "":
                    mode = None

    return images, constants


# ---------------------------------------------------------------------------
# Find command
# ---------------------------------------------------------------------------

def find_spot(debug_dir, cx, cy, stem_filter=None, radius=200):
    """Search debug_spots.json for rejected candidates near (cx, cy)."""
    data = load_json(debug_dir)
    if data is None:
        return

    for img in data["images"]:
        if stem_filter and stem_filter not in img["stem"]:
            continue

        nearby = [
            r for r in img["rejected"]
            if math.hypot(r["cx"] - cx, r["cy"] - cy) <= radius
        ]
        if not nearby:
            continue

        nearby.sort(key=lambda r: math.hypot(r["cx"] - cx, r["cy"] - cy))
        print(f"\n{img['stem']} — {len(nearby)} candidate(s) within {radius}px of ({cx},{cy}):")
        for r in nearby[:20]:
            dist = math.hypot(r["cx"] - cx, r["cy"] - cy)
            print(f"  dist={dist:.0f}px  cx={r['cx']:.0f} cy={r['cy']:.0f}  "
                  f"area={r['area']}  contrast={r['contrast']:.0f}  "
                  f"reason={r['reason']}  detail={r['detail']}")
        if len(nearby) > 20:
            print(f"  ... and {len(nearby) - 20} more")

    # Also check detected (accepted) spots
    for img in data["images"]:
        if stem_filter and stem_filter not in img["stem"]:
            continue
        nearby = [
            s for s in img["detected"]
            if math.hypot(s["cx"] - cx, s["cy"] - cy) <= radius
        ]
        if nearby:
            print(f"\n{img['stem']} — ACCEPTED spots near ({cx},{cy}):")
            for s in nearby:
                dist = math.hypot(s["cx"] - cx, s["cy"] - cy)
                extras = ""
                if "spot_sat" in s:
                    extras += f"  spot_sat={s['spot_sat']:.0f}"
                if "context_texture" in s:
                    extras += f"  ctx_tex={s['context_texture']:.1f}"
                print(f"  dist={dist:.0f}px  cx={s['cx']:.0f} cy={s['cy']:.0f}  "
                      f"area={s['area']}  contrast={s['contrast']:.0f}  "
                      f"texture={s['texture']:.1f}  excess_sat={s['excess_sat']:.1f}"
                      f"{extras}")


# ---------------------------------------------------------------------------
# Summary command
# ---------------------------------------------------------------------------

def _filter_stats(data):
    """Aggregate rejection reasons across all images."""
    counts = {}
    for img in data["images"]:
        for r in img["rejected"]:
            counts[r["reason"]] = counts.get(r["reason"], 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def summarize(debug_dir):
    images, constants = load_report(debug_dir)
    data = load_json(debug_dir)

    print(f"Debug dir: {debug_dir}")
    if constants:
        print(f"\nConstants:")
        for k, v in constants.items():
            print(f"  {k} = {v}")

    total_fps = 0
    total_missed = 0
    print(f"\n{'Image':<15}  {'Detected':>8}  {'FP':>4}  {'Missed':>6}")
    print("-" * 40)
    for stem, img in images.items():
        fp_count = len(img["fps"])
        miss_count = len(img["missed"])
        total_fps += fp_count
        total_missed += miss_count
        print(f"{stem:<15}  {img['detected']:>8}  {fp_count:>4}  {miss_count:>6}")
    print("-" * 40)
    print(f"{'TOTAL':<15}  {'':>8}  {total_fps:>4}  {total_missed:>6}")

    if data:
        stats = _filter_stats(data)
        total_rejected = sum(stats.values())
        print(f"\nRejection filter breakdown (total {total_rejected}):")
        for reason, count in stats.items():
            bar = "#" * min(40, count * 40 // max(total_rejected, 1))
            print(f"  {reason:<12} {count:>6}  {bar}")

    # Annotated FP details
    for stem, img in images.items():
        if img["fps"]:
            print(f"\n{stem} false positives (detected but should be ignored):")
            for fp in img["fps"]:
                # Find in detected list
                info = ""
                if data:
                    for dimg in data["images"]:
                        if dimg["stem"] == stem:
                            match = min(dimg["detected"],
                                        key=lambda s: math.hypot(s["cx"]-fp["cx"], s["cy"]-fp["cy"]),
                                        default=None)
                            if match and math.hypot(match["cx"]-fp["cx"], match["cy"]-fp["cy"]) < 10:
                                extras = ""
                                if "spot_sat" in match:
                                    extras += f"  spot_sat={match['spot_sat']:.0f}"
                                if "context_texture" in match:
                                    extras += f"  ctx_tex={match['context_texture']:.1f}"
                                info = (f"  contrast={match['contrast']:.0f}  "
                                        f"texture={match['texture']:.1f}  "
                                        f"excess_sat={match['excess_sat']:.1f}"
                                        f"{extras}")
                print(f"  cx={fp['cx']:.0f} cy={fp['cy']:.0f}{info}")

        if img["missed"]:
            print(f"\n{stem} missed dust (not detected, should have been):")
            for miss in img["missed"]:
                # Find nearest reject reason
                reason = "not in reject list"
                detail = ""
                if data:
                    for dimg in data["images"]:
                        if dimg["stem"] == stem:
                            nearby = [r for r in dimg["rejected"]
                                      if math.hypot(r["cx"]-miss["cx"], r["cy"]-miss["cy"]) < 50]
                            if nearby:
                                best = min(nearby, key=lambda r: math.hypot(r["cx"]-miss["cx"], r["cy"]-miss["cy"]))
                                dist = math.hypot(best["cx"]-miss["cx"], best["cy"]-miss["cy"])
                                reason = best["reason"]
                                detail = f"  [{best['detail']}]  dist={dist:.0f}px"
                print(f"  cx={miss['cx']} cy={miss['cy']}  → {reason}{detail}")


# ---------------------------------------------------------------------------
# Compare command
# ---------------------------------------------------------------------------

def compare(dir1, dir2):
    imgs1, consts1 = load_report(dir1)
    imgs2, consts2 = load_report(dir2)
    data2 = load_json(dir2)

    # Show changed constants
    changed_consts = {k: (consts1.get(k), consts2.get(k))
                      for k in set(consts1) | set(consts2)
                      if consts1.get(k) != consts2.get(k)}
    if changed_consts:
        print("Changed constants:")
        for k, (v1, v2) in changed_consts.items():
            print(f"  {k}: {v1} → {v2}")
    else:
        print("Constants unchanged.")

    all_stems = sorted(set(imgs1) | set(imgs2))
    total_fp1 = total_fp2 = total_miss1 = total_miss2 = 0

    for stem in all_stems:
        i1 = imgs1.get(stem, {"detected": 0, "fps": [], "missed": []})
        i2 = imgs2.get(stem, {"detected": 0, "fps": [], "missed": []})
        fp1, fp2 = len(i1["fps"]), len(i2["fps"])
        m1, m2 = len(i1["missed"]), len(i2["missed"])
        d1, d2 = i1["detected"], i2["detected"]
        total_fp1 += fp1; total_fp2 += fp2
        total_miss1 += m1; total_miss2 += m2

        fp_delta = fp2 - fp1
        miss_delta = m2 - m1
        print(f"\n{stem}:")
        print(f"  detected: {d1} → {d2}  ({d2-d1:+d})")
        print(f"  FP:       {fp1} → {fp2}  ({fp_delta:+d})", end="")
        if fp_delta < 0:
            print("  ✓ fewer false positives", end="")
        elif fp_delta > 0:
            print("  ✗ more false positives", end="")
        print()
        print(f"  missed:   {m1} → {m2}  ({miss_delta:+d})", end="")
        if miss_delta < 0:
            print("  ✓ fewer missed", end="")
        elif miss_delta > 0:
            print("  ✗ more missed", end="")
        print()

        # Show new misses with their rejection reason from dir2
        new_misses = [s for s in i2["missed"]
                      if not any(math.hypot(s["cx"]-o["cx"], s["cy"]-o["cy"]) < 30
                                 for o in i1["missed"])]
        if new_misses and data2:
            print(f"  New misses in run2:")
            for miss in new_misses:
                reason = "not in reject list"
                detail = ""
                for dimg in data2["images"]:
                    if dimg["stem"] == stem:
                        nearby = [r for r in dimg["rejected"]
                                  if math.hypot(r["cx"]-miss["cx"], r["cy"]-miss["cy"]) < 50]
                        if nearby:
                            best = min(nearby, key=lambda r: math.hypot(r["cx"]-miss["cx"], r["cy"]-miss["cy"]))
                            reason = best["reason"]
                            detail = f" [{best['detail']}]"
                print(f"    cx={miss['cx']} cy={miss['cy']}  → {reason}{detail}")

        fixed_fps = [s for s in i1["fps"]
                     if not any(math.hypot(s["cx"]-o["cx"], s["cy"]-o["cy"]) < 30
                                for o in i2["fps"])]
        if fixed_fps:
            coords = ", ".join(f"({s['cx']:.0f},{s['cy']:.0f})" for s in fixed_fps)
            print(f"  Fixed FPs: {coords}")

    print(f"\n{'='*40}")
    print(f"TOTAL  FP: {total_fp1} → {total_fp2}  ({total_fp2-total_fp1:+d})")
    print(f"TOTAL  missed: {total_miss1} → {total_miss2}  ({total_miss2-total_miss1:+d})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    # Compare mode: two positional dirs
    if len(args) >= 2 and not args[1].startswith("--") and os.path.isdir(args[1]):
        compare(args[0], args[1])
        return

    debug_dir = args[0]
    rest = args[1:]

    if "--find" in rest:
        idx = rest.index("--find")
        try:
            cx = float(rest[idx + 1])
            cy = float(rest[idx + 2])
        except (IndexError, ValueError):
            print("Usage: --find <cx> <cy> [--image <stem>] [--radius <px>]")
            sys.exit(1)
        stem_filter = None
        radius = 200
        if "--image" in rest:
            stem_filter = rest[rest.index("--image") + 1]
        if "--radius" in rest:
            radius = float(rest[rest.index("--radius") + 1])
        find_spot(debug_dir, cx, cy, stem_filter, radius)
    else:
        summarize(debug_dir)


if __name__ == "__main__":
    main()
