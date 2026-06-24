"""The catalog of every detection constant a calibration may fit, per KIND.

Mirrors auto_negadoctor's calibration_registry: a session's config picks a SUBSET
under `fit.params`. Built on the shared `common.calibration.registry.Registry`
(same machinery), so the runner gets `fittable` / `current` / `snapshot` / `restore`
/ `to_tuning` for free.

KIND = which annotation metric judges the constant (the kind it is calibrated under),
taken from its top-level group in `tuning.GROUPS`:
  dust    circular-spot ("dot") thresholds (judged by false-positive + missed-dust).
  stroke  thread / scratch / streak / radon thresholds (judged by missed-strokes + FP).
  sensor  multi-frame sensor-dust consensus (no calibration kind wired yet).

The REGISTRY is GENERATED from the tuning schema: every numeric (float/int) field
becomes fittable with a default-relative range; bool master-switches and the tuple
field (STROKE_RIDGE_SIGMAS) are NOT fittable (you don't line-search a feature toggle)
and are omitted. Tune the ranges in a session's config when a constant needs a wider
or tighter search than the heuristic default.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import detect_dust as dd          # noqa: E402
import tuning                     # noqa: E402
from common.calibration.registry import Registry, P  # noqa: E402

_MODULES = {"detect_dust": dd}

# Per-name range OVERRIDES where the default-relative heuristic is wrong (a bounded
# fraction, a small-integer count, a percentile, …). Everything else uses the
# heuristic below. (lo, hi, step[, int]).
_OVERRIDES = {
    # probabilities / fractions that must stay in (0, 1]
    "MIN_ASPECT_RATIO": (0.1, 0.6, 0.05), "MIN_COMPACTNESS": (0.1, 0.5, 0.05),
    "MIN_SOLIDITY": (0.3, 0.8, 0.05), "MIN_CIRCULARITY": (0.05, 0.4, 0.025),
    "DOT_MIN_CIRCLE_FILL": (0.2, 0.7, 0.05), "MAX_BG_GRADIENT_RATIO": (0.04, 0.2, 0.01),
    "MIN_BRIGHTNESS_FRAC_SMALL": (0.3, 0.8, 0.05),
    "MIN_BRIGHTNESS_FRAC_LARGE": (0.5, 0.95, 0.05),
    "MIN_LOCAL_BG_FRACTION": (0.3, 0.8, 0.05), "MIN_SURROUND_BG_RATIO": (0.5, 0.9, 0.05),
    "STROKE_MIN_CRISPNESS": (0.2, 0.7, 0.05), "STROKE_MAX_SIDE_ASYMMETRY": (0.05, 0.4, 0.025),
    "STROKE_MAX_FILL_RATIO": (0.3, 0.7, 0.05), "STREAK_RADON_MIN_COV": (0.2, 0.6, 0.05),
    "STROKE_HYST_LOW_FACTOR": (0.3, 0.8, 0.05),
    # small-integer counts / votes
    "MIN_DUST_VOTES": (1, 3, 1, True), "MAX_NEARBY_ACCEPTED": (1, 8, 1, True),
    "STROKE_FIELD_MAX_NEIGHBORS": (1, 6, 1, True),
    "SENSOR_DUST_MIN_FRAMES": (2, 5, 1, True),
    # percentiles (bounded)
    "STROKE_COVERAGE_PCTL": (80, 99, 1, True), "STROKE_CLIP_LEVEL": (240, 254, 1, True),
    # odd kernel sizes — leave to explicit config tuning; modest band
    "LOCAL_BG_KERNEL": (101, 301, 50, True), "TEXTURE_KERNEL": (15, 51, 2, True),
}


def _spec_for(name):
    """Range spec for a numeric field: an explicit override, else a default-relative
    band (clamped to >=0; a [0,1]-looking default stays within ~[0, 1])."""
    if name in _OVERRIDES:
        o = _OVERRIDES[name]
        return P(o[0], o[1], o[2], integer=(len(o) > 3 and o[3]), module="detect_dust")
    is_int = name in tuning.INT_FIELDS
    v = float(getattr(dd, name))
    lo, hi = v * 0.5, v * 1.5
    if v < 0:
        lo, hi = v * 1.5, v * 0.5
    if 0.0 < v <= 1.0:                       # looks like a fraction — keep sane band
        hi = min(hi, 1.0)
    span = hi - lo or max(abs(v), 1.0)
    step = max(1, round(span / 10)) if is_int else round(span / 10, 6)
    return P(round(lo, 6), round(hi, 6), step, integer=is_int, module="detect_dust")


def _build_registry():
    reg = {}
    for kind, subs in tuning.GROUPS.items():
        entries = {}
        for names in subs.values():
            for n in names:
                if n in tuning.BOOL_FIELDS or n in tuning.TUPLE_FIELDS:
                    continue   # master switches + the sigma tuple are not line-searched
                entries[n] = _spec_for(n)
        reg[kind] = entries
    return reg


REGISTRY = _build_registry()

# Bind the shared Registry operations at module level (so `reg.fittable`,
# `reg.to_tuning`, … match the auto_negadoctor surface the runner expects).
_reg = Registry(REGISTRY, _MODULES, tuning, "detect_dust")
fittable = _reg.fittable
kind_of = _reg.kind_of
current = _reg.current
snapshot = _reg.snapshot
apply = _reg.apply
restore = _reg.restore
to_tuning = _reg.to_tuning
_find = _reg._find
_parse = _reg._parse
