"""Generate tests/baseline_session/<roll_id>/ from a pipeline run over the
first annotated roll under tests/fixtures/rolls/.

ONLY run this after the user has reviewed and approved the detections in the
debug UI — the baseline is ground truth, don't bake in unreviewed output.

Writes only the {stem}_debug_nega.json files (no preview JPEGs) into
baseline_session/<roll_id>/, replacing whatever was there.

Run: conda run -n autocrop python auto_negadoctor/tests/generate_baseline.py
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

TESTS_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(TESTS_DIR.parent))
import auto_negadoctor as an

sys.path.insert(0, str(TESTS_DIR))
from run_quality_tests import discover_rolls

BASELINE_DIR = TESTS_DIR / "baseline_session"


def main():
    rolls = discover_rolls()
    if not rolls:
        print("No annotated rolls with local TIFFs found (see "
              "fixtures/rolls/README.md)")
        return 1
    # Baselines are per-roll (stems collide across rolls); regenerate the first.
    roll_info = rolls[0]
    images, exif = roll_info["images"], roll_info["exif"]
    out_dir = BASELINE_DIR / roll_info["id"]

    print(f"Roll {roll_info['id']}: running pipeline on {len(images)} frame(s)...")
    frames, roll = an.process_roll(images, exif)
    errors = [f for f in frames if f.get("error")]
    for fr in errors:
        print(f"  ERR {fr['stem']}: {fr['error']}")
    if errors:
        print("Refusing to write a baseline with errors.")
        return 1

    # Write full sessions to a temp dir, then keep only the JSONs
    with tempfile.TemporaryDirectory(prefix="nega_baseline_") as tmp:
        an.write_debug_sessions(frames, roll, tmp)
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("*_debug_nega.json"):
            old.unlink()
        n = 0
        for f in sorted(Path(tmp).glob("*_debug_nega.json")):
            shutil.copy(f, out_dir / f.name)
            n += 1

    print(f"Baseline written: {n} file(s) in {out_dir}")
    print("Commit baseline_session/ only after user approval of the detections.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
