"""Generate tests/baseline_session/ from a pipeline run over tests/images.

ONLY run this after the user has reviewed and approved the detections in the
debug UI — the baseline is ground truth, don't bake in unreviewed output.

Writes only the {stem}_debug_nega.json files (no preview JPEGs) into
baseline_session/, replacing whatever was there.

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
from run_quality_tests import list_test_images

BASELINE_DIR = TESTS_DIR / "baseline_session"


def main():
    images, exif = list_test_images()
    if not images:
        print("No test images found")
        return 1

    print(f"Running pipeline on {len(images)} frame(s)...")
    frames, roll = an.process_roll(images, exif)
    errors = [f for f in frames if f.get("error")]
    for fr in errors:
        print(f"  ERR {fr['stem']}: {fr['error']}")
    if errors:
        print("Refusing to write a baseline with errors.")
        return 1

    # Write full sessions to a temp dir, then keep only the JSONs
    with tempfile.TemporaryDirectory(prefix="nega_baseline_") as tmp:
        an.write_debug_sessions(frames, roll, tmp, save_vis=False)
        BASELINE_DIR.mkdir(exist_ok=True)
        for old in BASELINE_DIR.glob("*_debug_nega.json"):
            old.unlink()
        n = 0
        for f in sorted(Path(tmp).glob("*_debug_nega.json")):
            shutil.copy(f, BASELINE_DIR / f.name)
            n += 1

    print(f"Baseline written: {n} file(s) in {BASELINE_DIR}")
    print("Commit baseline_session/ only after user approval of the detections.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
