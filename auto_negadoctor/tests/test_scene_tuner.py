"""Unit tests for scene_tuner.py (pure math + mocked LLM, NO network).

Covers the param-nudging guarantees that keep the AI variant physical and
clip-safe, plus the graceful-degradation contract of categorize_scene.

Run: conda run -n autocrop python auto_negadoctor/tests/test_scene_tuner.py
Exit code 0 = all pass.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import auto_negadoctor as an
import nega_model as nm
import scene_tuner as st

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}{('  ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Synthetic frame + analytical params (no real TIFF needed)
# ---------------------------------------------------------------------------

def _synthetic_frame():
    """A linear 'negative': lightest (=film base) at one edge, denser scene
    toward the other, with a couple of bright scene patches. Returns (lin,
    border, dmin, analytical_params)."""
    H, W = 80, 120
    base = np.array([0.80, 0.35, 0.18], dtype=np.float32)
    ramp = np.linspace(1.0, 3.5, W, dtype=np.float32)        # density multiplier
    lin = base[None, None, :] * ramp[None, :, None] * np.ones((H, W, 1), np.float32)
    # a few very dense (bright-scene) pixels to exercise the clip guard
    lin[10:20, 100:115, :] *= 1.4
    border = (4, 4, 4, 4)
    dmin = [float(v) for v in base]
    lin32 = lin.astype(np.float32)
    pmin, pmax = an.frame_percentiles(lin32, np.zeros_like(lin32), border, dmin)
    d_max = nm.compute_dmax(dmin, pmin)
    params = an.make_params(dmin, d_max, an.OFFSET_DEFAULT,
                            [1.0, 0.8, 0.72], [1.7, 1.3, 1.0], pmin, pmax)
    params, _ = an.tune_print_params(lin, params, border, dmin)
    return lin, border, dmin, params


def _clip_frac(lin, params, border):
    l, t, r, b = border
    h, w = lin.shape[:2]
    region = lin[t:h - b, l:w - r].reshape(-1, 3)
    out = nm.render_negadoctor(region, params)
    return float(np.mean((out >= 0.999).any(axis=1)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_scene_is_identity():
    lin, border, dmin, params = _synthetic_frame()
    out, info = st.apply_scene_tuning(params, None, lin, border)
    check("scene=None returns analytical params unchanged",
          out == params and not info.get("tuned"))
    check("scene=None returns a COPY (not the same object)", out is not params)


def test_gamma_bounds():
    lin, border, dmin, params = _synthetic_frame()
    lo = max(st.GAMMA_GT_RANGE[0], nm.GAMMA_RANGE[0])
    hi = min(st.GAMMA_GT_RANGE[1], nm.GAMMA_RANGE[1])
    for contrast in st.CONTRASTS:
        scene = {"scene": "other", "mood": "neutral", "warmth": "neutral",
                 "contrast": contrast, "confidence": 0.9, "rationale": ""}
        out, _ = st.apply_scene_tuning(params, scene, lin, border)
        check(f"gamma in [{lo},{hi}] for contrast={contrast}",
              lo - 1e-9 <= out["gamma"] <= hi + 1e-9, f"got {out['gamma']}")
    # the SCENE label now drives the paper grade (the contrast label is noise on
    # the GT roll, so CONTRAST_GAMMA is disabled): a flat/daylight scene prints a
    # lower grade than a tungsten/night scene.
    low = st.apply_scene_tuning(params, {"scene": "daylight", "mood": "neutral",
            "warmth": "neutral", "contrast": "normal", "rationale": ""},
            lin, border)[0]["gamma"]
    high = st.apply_scene_tuning(params, {"scene": "night", "mood": "neutral",
            "warmth": "neutral", "contrast": "normal", "rationale": ""},
            lin, border)[0]["gamma"]
    check("daylight gamma <= night gamma", low <= high, f"{low} vs {high}")


def test_wb_normalization_preserved():
    lin, border, dmin, params = _synthetic_frame()
    # the warmth label does not perturb wb on the current roll (WARMTH_SHIFT
    # disabled), but apply_scene_tuning must keep wb normalized regardless.
    for warmth in ("warmer", "cooler"):
        scene = {"scene": "indoor_tungsten", "mood": "dark", "warmth": warmth,
                 "contrast": "normal", "confidence": 0.9, "rationale": ""}
        out, _ = st.apply_scene_tuning(params, scene, lin, border)
        check(f"max(wb_low)==1 after warmth={warmth}",
              abs(max(out["wb_low"]) - 1.0) < 1e-3, str(out["wb_low"]))
        check(f"min(wb_high)==1 after warmth={warmth}",
              abs(min(out["wb_high"]) - 1.0) < 1e-3, str(out["wb_high"]))
    # the _warm_wb lever itself (kept for re-enabling on a roll where the warmth
    # label is reliable) preserves normalization and the warmer-raises-R/B
    # direction.
    warm = st._warm_wb(params["wb_low"], "shadows", +0.08, an._normalize_wb)
    cool = st._warm_wb(params["wb_low"], "shadows", -0.06, an._normalize_wb)
    check("max(wb_low)==1 after warm shift", abs(max(warm) - 1.0) < 1e-3, str(warm))
    rb_warm = warm[0] / max(warm[2], 1e-6)
    rb_cool = cool[0] / max(cool[2], 1e-6)
    check("warmer R/B > cooler R/B (helper)", rb_warm > rb_cool,
          f"{rb_warm} vs {rb_cool}")


def test_clip_safety():
    """The AI variant must never clip more than the analytical budget — the
    binding constraint (re-applied unconditionally inside tune_print_params)."""
    lin, border, dmin, params = _synthetic_frame()
    base_clip = _clip_frac(lin, params, border)
    for mood in st.MOODS:
        for contrast in st.CONTRASTS:
            scene = {"scene": "snow", "mood": mood, "warmth": "warmer",
                     "contrast": contrast, "confidence": 0.9, "rationale": ""}
            out, _ = st.apply_scene_tuning(params, scene, lin, border)
            clip = _clip_frac(lin, out, border)
            check(f"clip<=budget for mood={mood},contrast={contrast}",
                  clip <= an.PRINT_CLIP_BUDGET + 1e-6,
                  f"clip {clip:.4f} > budget {an.PRINT_CLIP_BUDGET}")
    check("analytical baseline within budget too",
          base_clip <= an.PRINT_CLIP_BUDGET + 1e-6, f"{base_clip}")


def test_mood_brightness_ordering():
    """A 'dark/moody' scene should not end up BRIGHTER than a 'bright' scene
    (lower highlight ceiling -> lower exposure pin)."""
    lin, border, dmin, params = _synthetic_frame()
    def midtone(scene):
        out, _ = st.apply_scene_tuning(params, scene, lin, border)
        l, t, r, b = border
        h, w = lin.shape[:2]
        region = lin[t:h - b, l:w - r].reshape(-1, 3)
        rgb = nm.render_negadoctor(region, out)
        return float(np.percentile(rgb.mean(axis=1), 50))
    base = {"scene": "daylight", "warmth": "neutral", "contrast": "normal",
            "confidence": 0.9, "rationale": ""}
    bright = midtone(dict(base, mood="bright"))
    moody = midtone(dict(base, mood="moody"))
    check("moody midtone <= bright midtone", moody <= bright + 1e-6,
          f"moody {moody:.4f} vs bright {bright:.4f}")


def test_validate():
    good = {"scene": "fog", "mood": "dark", "warmth": "warmer",
            "contrast": "soft", "rationale": "x"}
    check("valid dict passes through", st._validate(good) is not None)
    check("non-dict -> None", st._validate("nope") is None)
    coerced = st._validate({"scene": "bogus", "mood": "??", "warmth": "x",
                            "contrast": "y"})
    check("unknown labels coerced to safe defaults",
          coerced and coerced["scene"] == "other"
          and coerced["mood"] == "neutral" and coerced["contrast"] == "normal")


def test_categorize_offline_returns_none(monkeypatch_host="http://127.0.0.1:9"):
    """Unreachable endpoint -> None (caller keeps analytical params)."""
    orig = st.OLLAMA_HOST
    st.OLLAMA_HOST = monkeypatch_host
    try:
        img = (np.random.default_rng(0).random((40, 60, 3)) * 255).astype(np.uint8)
        st.OLLAMA_TIMEOUT_S_orig = st.OLLAMA_TIMEOUT_S
        st.OLLAMA_TIMEOUT_S = 1.0
        res = st.categorize_scene(img)
        check("unreachable Ollama -> None", res is None)
    finally:
        st.OLLAMA_HOST = orig
        st.OLLAMA_TIMEOUT_S = st.OLLAMA_TIMEOUT_S_orig


def test_cache_hit_skips_network():
    """A pre-populated cache must be used without any network call."""
    img = (np.zeros((30, 40, 3))).astype(np.uint8)
    _, raw = st._encode_image(img)
    import hashlib
    digest = hashlib.sha1(raw).hexdigest()[:12]
    key = f"DSC_X:{st.OLLAMA_MODEL}:{digest}"
    cached = {"scene": "forest", "mood": "neutral", "warmth": "neutral",
              "contrast": "punchy", "confidence": 0.9, "rationale": "cached"}
    with tempfile.TemporaryDirectory() as d:
        cache_path = os.path.join(d, "scene_cache.json")
        with open(cache_path, "w") as f:
            json.dump({key: cached}, f)
        # break the network so a miss would fail loudly -> proves cache hit
        orig = st._query_ollama
        st._query_ollama = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no net"))
        try:
            res = st.categorize_scene(img, cache_path=cache_path, cache_key="DSC_X")
        finally:
            st._query_ollama = orig
    check("cache hit returns cached value without network",
          res is not None and res["scene"] == "forest")


def main():
    test_no_scene_is_identity()
    test_gamma_bounds()
    test_wb_normalization_preserved()
    test_clip_safety()
    test_mood_brightness_ordering()
    test_validate()
    test_categorize_offline_returns_none()
    test_cache_hit_skips_network()
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
