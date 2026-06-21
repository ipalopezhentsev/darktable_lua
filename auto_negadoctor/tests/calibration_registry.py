"""The COMPLETE catalog of every algorithm tuning constant a calibration may fit.

A session's config picks a SUBSET of these under `fit.params` (fitting all of a
kind at once is infeasible — coordinate descent over dozens of constants with a
pipeline re-run per trial — so you compose the set you care about per session).
This file is the menu; the config is the order.

`run_calibration.py` builds an immutable `an.Tuning` per trial via `to_tuning()`
(DEFAULT_TUNING + the chosen overrides) and PASSES it into the analysis functions
(process_roll / detect_content_crop / estimate_vignette all take `cfg`). Because
no shared module state is mutated, independent trials (random_search) can run in
parallel threads. (`apply()`/`snapshot()`/`restore()` — the old global-setattr
path — remain as utilities but the runner no longer needs them.)

RECORD-ONLY: nothing here is written back to the source. A session records its
best values in `fitted_params.json`; the user adopts the good ones by hand.

Grouped by KIND = which metric/evaluator judges the constant (i.e. which output
it moves), NOT where it lives in the code:
  crop       constants `detect_content_crop` reads (judged by crop over-trim).
  inversion  everything that shapes the inverted PICTURE — film base, pickers,
             patch search, white balance, print tune (judged by histogram EMD).
  vignette   the roll-wide vignette estimator (judged by fit validity+residual).

Tuple-valued constants (wb priors, the percentile bands) are exposed PER ELEMENT
as `NAME[i]` (e.g. `WB_HIGH_PRIOR[0]`). Integer constants carry `"int": True`.

Fast vs full path (the runner dispatches automatically):
  - inversion: `PRINT_TUNE_PARAMS` only affect the print tune, so they re-tune on
    cached frames (fast). Anything else re-runs the whole pipeline per trial.
  - vignette: `VIG_FIT_PARAMS` only affect `fit_vignette_profile`, so the roll's
    radial envelope is captured from the TIFFs ONCE and only the profile fit
    re-runs (fast). The envelope-ACCUMULATION constants reshape the envelope, so
    they re-run the whole `estimate_vignette` on the TIFFs per trial (full).
  - crop: every listed constant is read by `detect_content_crop`, so the crop
    evaluator always re-runs just that (fast) on the cached per-frame buffer.
"""

import auto_negadoctor as an
import nega_model as nm

_MODULES = {"auto_negadoctor": an, "nega_model": nm}

# Inversion constants that affect ONLY tune_print_params (fast path eligible).
PRINT_TUNE_PARAMS = ("PRINT_GAMMA", "PRINT_HI_CEIL", "PRINT_HI_PCT",
                     "PRINT_CLIP_BUDGET", "PRINT_TUNE_ITERS")
# The film-base search constants read by process_roll's trial-INVARIANT PREFIX
# (stage A → the global base / Dmin). They are UPSTREAM of the inversion look and
# are NOT in the inversion REGISTRY (see the inversion block) — the prefix is
# computed ONCE per session and reused across trials. This tuple is the guard
# `_make_inversion_full` asserts the fit never touches, so the invariant fails
# loud if one ever leaks back into the registry.
BASE_PREFIX_PARAMS = ("BASE_WIN_FRAC", "MIN_WIN_FRAC", "BASE_SCAN_STRIDE_FRAC",
                      "BASE_MASK_SOLID_FRAC", "BASE_AREA_MIN_FRAC",
                      "CLIP_FRAC_MAX", "BASE_MIN_LUMA",
                      "BASE_UNIFORMITY_MAX", "BASE_MIN_RG_RATIO", "BASE_GB_TOL")
# Vignette constants that affect ONLY fit_vignette_profile (envelope-fixture
# fast path; the rest reshape the envelope and need estimate_vignette on TIFFs).
VIG_FIT_PARAMS = ("VIG_MIN_STRENGTH", "VIG_PEAK_CENTER_FRAC", "VIG_TAIL_CUT_REL")


def P(lo, hi, step, integer=False, module="auto_negadoctor"):
    e = {"module": module, "range": [lo, hi], "grid_step": step}
    if integer:
        e["int"] = True
    return e


REGISTRY = {
    # ====================== CROP (detect_content_crop) ======================
    "crop": {
        "CROP_JUNK_LINE_FRAC":   P(0.02, 0.10, 0.01),
        "CROP_LEAK_MARGIN_D":    P(0.03, 0.12, 0.01),
        "CROP_REBATE_MARGIN_D":  P(0.08, 0.18, 0.01),
        "CROP_REBATE_LINE_FRAC": P(0.08, 0.30, 0.02),
        "CROP_REBATE_TERM_FRAC": P(0.02, 0.08, 0.005),
        "CROP_REBATE_MAX_FRAC":  P(0.02, 0.08, 0.005),
        # WIDE confident-rebate path — for frames where the user scrolled film to
        # scan extra rebate so it fills ~half the frame. Per-pixel signatures gate
        # WHERE the band runs (hue match, low density, solid columns); the
        # band-mean hue is the decisive accept/reject; wide frac is the depth cap.
        "CROP_REBATE_HUE_TOL":        P(0.02, 0.06, 0.005),
        "CROP_REBATE_WIDE_MAX_D":     P(0.25, 0.50, 0.025),
        "CROP_REBATE_WIDE_LINE_FRAC": P(0.40, 0.80, 0.05),
        "CROP_REBATE_BAND_HUE_TOL":   P(0.004, 0.015, 0.001),
        "CROP_REBATE_WIDE_FRAC":      P(0.10, 0.60, 0.05),
        "CROP_PAD_FRAC":         P(0.004, 0.020, 0.002),
        "CROP_SHADOW_REL":       P(0.55, 0.85, 0.05),
        "CROP_SHADOW_MAX_FRAC":  P(0.020, 0.050, 0.005),
        "CROP_SHADOW_CORE_FRAC": P(0.008, 0.030, 0.004),
        "CROP_GAP_TOL_FRAC":     P(0.01, 0.04, 0.005),
        "HOLDER_LUMA_THR":       P(0.01, 0.08, 0.005),
        # The conservative per-edge cap stays tight — the wide rebate path uses
        # its OWN CROP_REBATE_WIDE_FRAC cap, so this need not be loosened.
        "BORDER_MAX_FRAC":       P(0.08, 0.15, 0.01),
    },

    # ====================== INVERSION (the picture) =========================
    # The roll-wide VIGNETTE and the FILM-BASE search (→ the global base / Dmin)
    # are UPSTREAM of the inversion look: process_roll computes them ONCE per
    # session (its trial-invariant PREFIX) and they are NOT calibrated by an
    # inversion session. They are therefore DELIBERATELY ABSENT here — the
    # film-base constants (BASE_*/CLIP_FRAC_MAX/MIN_WIN_FRAC) and the vignette
    # constants (VIG_*/BORDER_*) live under their own concerns (vignette is its
    # own kind; tune crop + vignette FIRST, then inversion). Fitting them here
    # would move the Dmin/global base out from under every render AND force the
    # prefix — vignette included — to recompute on every trial. (BORDER_* are not
    # inversion params at all: BORDER_DARK_THR / BORDER_PAD_FRAC only feed the
    # vignette mask, BORDER_MAX_FRAC is a crop param.)
    "inversion": {
        # ---- density-range pickers / defaults ----
        "P_LOW":                P(0.5, 5.0, 0.5),
        "P_HIGH":               P(98.0, 99.9, 0.1),
        "OFFSET_DEFAULT":       P(-0.2, 0.1, 0.01),
        "DMAX_DEFAULT":         P(1.5, 2.5, 0.05),
        # ---- neutral-patch search (wb_high source) ----
        "PATCH_WIN_FRAC":         P(0.02, 0.08, 0.01),
        "PATCH_STRIDE_DIV":       P(1, 4, 1, integer=True),
        "HIGHLIGHT_BAND_PCT[0]":  P(50.0, 80.0, 2.0),
        "HIGHLIGHT_BAND_PCT[1]":  P(85.0, 99.0, 1.0),
        "SHADOW_MIN_LUMA":        P(0.001, 0.02, 0.001),
        "PATCH_CHROMA_MAX":       P(0.10, 0.60, 0.05),
        "PATCH_CHROMA_FLOOR":     P(0.005, 0.05, 0.005),
        "PATCH_UNIFORMITY_MAX":   P(0.20, 0.70, 0.05),
        "PATCH_LUMA_FLOOR":       P(0.005, 0.05, 0.005),
        "MIN_PATCH_DENSITY":      P(0.01, 0.15, 0.01),
        "HIGHLIGHT_CLIP_FRAC_MAX": P(0.005, 0.05, 0.005),
        # ---- COLOR / white balance (the wb the user picks on the wheels) ----
        "WB_LOW_BAND_PCT[0]":   P(0.0, 20.0, 1.0),
        "WB_LOW_BAND_PCT[1]":   P(20.0, 50.0, 2.0),
        "WB_HIGH_BAND_PCT[0]":  P(50.0, 80.0, 2.0),
        "WB_HIGH_BAND_PCT[1]":  P(85.0, 99.0, 1.0),
        "WB_LOW_DESAT":         P(0.0, 1.0, 0.05),
        "WB_HIGH_DESAT":        P(0.0, 1.0, 0.05),
        "WB_REGION_MIN_FRAC":   P(1e-5, 1e-3, 1e-4),
        "WB_HIGH_PRIOR[0]":     P(1.0, 2.5, 0.05),
        "WB_HIGH_PRIOR[1]":     P(1.0, 2.0, 0.05),
        "WB_HIGH_PRIOR[2]":     P(0.8, 1.2, 0.05),
        "WB_LOW_PRIOR[0]":      P(0.8, 1.2, 0.05),
        "WB_LOW_PRIOR[1]":      P(0.5, 1.0, 0.05),
        "WB_LOW_PRIOR[2]":      P(0.5, 1.0, 0.05),
        # ---- print tune / brightness (LUMA) — fast path ----
        "PRINT_HI_PCT":         P(99.0, 99.99, 0.1),
        "PRINT_HI_CEIL":        P(0.80, 0.99, 0.01),
        "PRINT_CLIP_BUDGET":    P(0.0, 0.02, 0.002),
        "PRINT_TUNE_ITERS":     P(6, 24, 2, integer=True),
        "PRINT_GAMMA":          P(4.0, 6.5, 0.25),
    },

    # ====================== VIGNETTE (roll-wide estimate) ===================
    "vignette": {
        # holder mask (detect_dark_border) — defines the valid region the
        # envelope accumulates over; the ONLY remaining use of these two now
        # that the film-base search runs full-frame (FULL path).
        "BORDER_DARK_THR":      P(0.005, 0.05, 0.005),
        "BORDER_PAD_FRAC":      P(0.0, 0.02, 0.002),
        # envelope accumulation (FULL path: estimate_vignette on TIFFs)
        "VIG_DOWNSAMPLE_FRAC":  P(0.002, 0.008, 0.001),
        "VIG_INSET_FRAC":       P(0.004, 0.03, 0.002),
        "VIG_BINS":             P(16, 48, 4, integer=True),
        "VIG_PROFILE_PCT":      P(80.0, 99.0, 1.0),
        "VIG_MIN_BIN_SAMPLES":  P(10, 50, 5, integer=True),
        # profile fit (FAST path: fit_vignette_profile on the captured envelope)
        "VIG_MIN_STRENGTH":     P(0.005, 0.06, 0.005),
        "VIG_PEAK_CENTER_FRAC": P(0.05, 0.30, 0.05),
        "VIG_TAIL_CUT_REL":     P(0.90, 0.995, 0.005),
    },
}


def fittable(kind):
    """The {name: spec} the given kind may fit (copy; safe to mutate)."""
    return {n: dict(s) for n, s in REGISTRY.get(kind, {}).items()}


def _find(name):
    for kind in REGISTRY.values():
        if name in kind:
            return kind[name]
    raise KeyError(f"{name!r} is not a registered fittable constant")


def kind_of(name):
    for kind, params in REGISTRY.items():
        if name in params:
            return kind
    raise KeyError(name)


def _parse(name):
    """('WB_HIGH_PRIOR[0]') -> ('WB_HIGH_PRIOR', 0); ('P_LOW') -> ('P_LOW', None)."""
    if name.endswith("]") and "[" in name:
        base, idx = name[:-1].split("[", 1)
        return base, int(idx)
    return name, None


def _coerce(val, spec):
    return int(round(val)) if spec.get("int") else float(val)


def current(name):
    """The constant's LIVE value on its module (the search start / init).
    For an indexed name, the addressed tuple element."""
    spec = _find(name)
    base, idx = _parse(name)
    val = getattr(_MODULES[spec["module"]], base)
    return float(val[idx]) if idx is not None else float(val)


def snapshot(names):
    """Capture the live value of each name's BASE attribute (whole tuple for
    indexed names) so it can be restored intact. Keyed by (module, base)."""
    snap = {}
    for name in names:
        spec = _find(name)
        base, _ = _parse(name)
        key = (spec["module"], base)
        if key not in snap:
            snap[key] = getattr(_MODULES[spec["module"]], base)
    return snap


def apply(overrides):
    """Set each name on its module. Indexed names of the same tuple are gathered
    and the tuple is rebuilt once (tuples are immutable)."""
    tuples = {}   # (module, base) -> {idx: value}
    for name, val in overrides.items():
        spec = _find(name)
        base, idx = _parse(name)
        if idx is None:
            setattr(_MODULES[spec["module"]], base, _coerce(val, spec))
        else:
            tuples.setdefault((spec["module"], base), {})[idx] = _coerce(val, spec)
    for (modname, base), idxvals in tuples.items():
        cur = list(getattr(_MODULES[modname], base))
        for i, v in idxvals.items():
            cur[i] = v
        setattr(_MODULES[modname], base, tuple(cur))


def restore(snap):
    """Inverse of apply(): write the snapshotted base values back."""
    for (modname, base), val in snap.items():
        setattr(_MODULES[modname], base, val)


def to_tuning(overrides, base=None):
    """Build an immutable `an.Tuning` = `base` (default an.DEFAULT_TUNING) with
    `overrides` applied — the THREAD-SAFE alternative to apply()/restore(): the
    runner gives each trial its OWN cfg to pass into the analysis functions
    instead of mutating shared module globals, so independent trials can run in
    parallel. Indexed names (NAME[i]) patch the addressed tuple element; ints are
    rounded. Every fittable constant lives in auto_negadoctor (asserted), so it
    is a Tuning field."""
    base = base if base is not None else an.DEFAULT_TUNING
    fields = base._asdict()
    tuples = {}   # base name -> {idx: value}
    for name, val in overrides.items():
        spec = _find(name)
        if spec["module"] != "auto_negadoctor":
            raise ValueError(f"{name!r} is not in auto_negadoctor; cfg only "
                             "covers auto_negadoctor constants")
        b, idx = _parse(name)
        if idx is None:
            fields[b] = _coerce(val, spec)
        else:
            tuples.setdefault(b, {})[idx] = _coerce(val, spec)
    for b, idxvals in tuples.items():
        cur = list(fields[b])
        for i, v in idxvals.items():
            cur[i] = v
        fields[b] = tuple(cur)
    return an.Tuning(**fields)
