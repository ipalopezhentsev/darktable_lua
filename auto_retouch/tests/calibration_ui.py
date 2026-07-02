"""Launch the calibration MANAGER UI for auto_retouch.

A thin launcher (same sys.path preamble as run_calibration.py): it hands the shared,
feature-agnostic manager (common/calibration/calib_ui.py) this feature's
`CalibrationAdapter` + the path to run_calibration.py, which the manager spawns per
queued job. It never runs a calibration itself — the console runner stays the engine.

  conda run -n autocrop python auto_retouch/tests/calibration_ui.py
"""
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent.parent))   # repo root -> common
sys.path.insert(0, str(TESTS_DIR.parent))
sys.path.insert(0, str(TESTS_DIR))

import run_calibration as rc              # noqa: E402
from common.calibration import calib_ui   # noqa: E402

if __name__ == "__main__":
    calib_ui.run(rc.ADAPTER, script_path=rc.__file__,
                 configs_dir=rc.CALIB_DIR / "configs")
