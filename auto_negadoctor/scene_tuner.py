"""Vision-LLM scene categorization + per-scene param nudging (spec 03).

An ADDITIVE layer on top of the analytical pipeline. `process_roll()` produces
the full analytical params (roll-wide vignette estimation, content-crop
detection, global film-base finding, picker percentiles, wb, print tune — ALL
unchanged). This module then:

  1. `categorize_scene()` reads the scene with a local vision LLM (gemma3 via
     Ollama) from the already-rendered inverted preview, returning a small
     structured label dict.
  2. `apply_scene_tuning()` nudges the FINAL gamma / wb-warmth / brightness
     toward the user's per-scene aesthetic (the irreducible "taste" residual
     the analytical math can't reach — fog→soft, forest→punch, museum→warm/
     dark; see auto_negadoctor CLAUDE.md), then re-runs the print tuner so the
     hard-clip guard is unconditionally re-applied.

It NEVER touches cropping, vignetting or film-base detection, and degrades
gracefully to the analytical params (categorize_scene -> None) whenever Ollama
is unavailable. The whole AI path is opt-in (--ai-tune flag / separate Lua
action); with it off the pipeline is byte-for-byte identical to before.

The category->offset mappings below are interpretable, hand-auditable constants
(NOT a learned model) — calibrated against the 37-frame ground truth by
tests/calibrate_scene_tuning.py.
"""

import base64
import hashlib
import json
import os
import urllib.request

import numpy as np

import nega_model as nm

# --- Ollama endpoint -------------------------------------------------------
OLLAMA_HOST = os.environ.get("NEGA_OLLAMA_HOST", "http://localhost:11434")
# moondream: ~2.4s/frame on a 6GB laptop GPU vs ~15s for gemma3:4b (whose
# vision encoder prefills on CPU at ~30 tok/s). Override with NEGA_OLLAMA_MODEL
# (gemma3:4b gives richer labels but ~6x slower; llava-phi3 is fast but its
# labels were repetitive in benchmarking).
OLLAMA_MODEL = os.environ.get("NEGA_OLLAMA_MODEL", "moondream")
OLLAMA_TIMEOUT_S = float(os.environ.get("NEGA_OLLAMA_TIMEOUT", "300"))
# Small context window: we send ONE small image + a tiny prompt. The model's
# default 128k context allocates a KV-cache that spills the model off a 6GB GPU
# (~48% CPU -> ~3x slower); 4096 keeps the whole 4b model GPU-resident.
OLLAMA_NUM_CTX = int(os.environ.get("NEGA_OLLAMA_NUM_CTX", "4096"))
LLM_MAX_SIDE = 512               # downscale the preview before sending
LLM_JPEG_QUALITY = 85

# --- label vocabulary (kept small + fixed; enforced via the JSON schema) ---
SCENES = ["snow", "forest", "fog", "indoor_tungsten", "daylight",
          "night", "sunset", "other"]
MOODS = ["bright", "neutral", "dark", "moody"]
WARMTHS = ["cooler", "neutral", "warmer"]
CONTRASTS = ["soft", "normal", "punchy"]

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {"type": "string", "enum": SCENES},
        "mood": {"type": "string", "enum": MOODS},
        "warmth": {"type": "string", "enum": WARMTHS},
        "contrast": {"type": "string", "enum": CONTRASTS},
        "rationale": {"type": "string"},
    },
    "required": ["scene", "mood", "warmth", "contrast", "rationale"],
}

PROMPT = (
    "You are a film-photography colorist judging a scanned, already-inverted "
    "color photo to guide its print tonality. Classify the SCENE, the intended "
    "MOOD (how bright/dark it should feel), the WARMTH the print wants "
    "(warmer = more orange, for tungsten interiors / sunsets; cooler = bluer, "
    "for overcast / snow), and the CONTRAST the scene calls for (soft for "
    "fog/haze/flat light, punchy for sunlit foliage/landscapes). Judge intent, "
    "not the current rendering. Keep `rationale` to a SHORT phrase (max ~6 "
    "words). Respond ONLY with the JSON object."
)

# ---------------------------------------------------------------------------
# Category -> param-nudge mappings (interpretable; calibrated against the
# 2026-06-13 GT roll by tests/calibrate_scene_tuning.py). Bounds are enforced
# after every nudge so the AI variant stays physical.
#
# Calibration finding (37-frame roll): the BIG, systematic gap between the
# analytical params and the user's hand-printed GT is NOT per-scene taste — it
# is a global re-target of the print tuner. The analytical tuner pushes
# brightness to the clip boundary (PRINT_HI_CEIL 0.99); the user prints LOWER
# and with lifted blacks. So the AI variant re-targets the print tune from
# within scene_tuner (the analytical path stays byte-for-byte):
#   * ceil 0.99 -> AI_HI_CEIL 0.72  halves the exposure delta (0.276 -> 0.149),
#   * +AI_BLACK_LIFT 0.10 to the tuned black  halves the black delta
#     (0.159 -> 0.068),
# both clip-safe (worst hard-clip ~0.27%, under PRINT_CLIP_BUDGET). The LLM
# labels add only the MODEST residual on top (per-scene gamma + mood ceiling
# modulation); the warmth label did not track the GT wb cast on this roll so it
# does not perturb wb (see WARMTH_SHIFT below).
# ---------------------------------------------------------------------------

# AI-variant print-tune re-target (global; analytical PRINT_HI_CEIL untouched).
AI_HI_CEIL = 0.72        # AI-variant base highlight pin (vs analytical 0.99)
AI_BLACK_LIFT = 0.10     # less-negative black to match the user's lifted shadows

# gamma offset (paper grade), keyed on SCENE — the `contrast` label is degenerate
# (moondream tagged 36/37 frames "soft"), so it carries no grade signal; the
# scene does. Per-scene GT-median gamma offset vs PRINT_GAMMA (6.1). Robust
# groups: forest(n=18)~0, snow(n=6) -0.1, fog(n=4)~0, night(n=3) +0.7;
# daylight/sunset/indoor_tungsten are small-n but real photographic tendencies
# (tungsten interiors print punchy, flat daylight soft). Clamped to the GT span.
SCENE_GAMMA = {"night": +0.7, "indoor_tungsten": +1.4, "snow": -0.1,
               "daylight": -0.75, "sunset": -0.55, "fog": 0.0,
               "forest": 0.0, "other": 0.0}
# contrast-label grade nudge: DISABLED. moondream tagged 36/37 frames "soft" and
# the lone "punchy" frame actually wanted a LOWER grade than the median, so the
# label carries no usable grade signal on this roll — any nonzero value hurt more
# frames than it helped (gamma median 0.10 -> 0.175). Kept (zeroed) so the
# physical-prior lever (punchy -> higher grade, soft -> lower) can be re-enabled
# on a roll where the contrast label is more discriminating. Added to SCENE_GAMMA.
CONTRAST_GAMMA = {"soft": 0.0, "normal": 0.0, "punchy": 0.0}
GAMMA_GT_RANGE = (4.5, 7.8)      # per-frame GT gamma span (auto_negadoctor CLAUDE.md)

# warmth: DISABLED. A chroma shift (R up / B down) keyed on the warmth label,
# re-normalized through darktable's picker. On the GT roll the LLM warmth label
# did NOT track the user's wb cast (R/B even ran counter-intuitive: "warmer"
# frames had LOWER wb_low R/B than "cooler"), so nudging wb through it hurt as
# many frames as it helped. wb is left to the analytical region-cast finder; its
# residual (wb_low B ~0.16, wb_high R ~0.08) is partly taste and needs a more
# reliable signal than this label. Kept (zeroed) so the lever can be re-enabled.
WARMTH_SHIFT = {"cooler": 0.0, "neutral": 0.0, "warmer": 0.0}

# mood: bias on the print tuner's highlight ceiling around AI_HI_CEIL. Dark/moody
# scenes pin the highlight lower so they print darker; bright keeps the AI base.
MOOD_CEIL_BIAS = {"bright": 0.0, "neutral": 0.0, "dark": -0.06, "moody": -0.10}
CEIL_RANGE = (0.45, 0.99)


# ---------------------------------------------------------------------------
# LLM call + cache
# ---------------------------------------------------------------------------

def _encode_image(rgb_or_path):
    """RGB uint8 array OR an image path -> base64 JPEG (downscaled). Imports
    cv2 lazily so the pure-math callers / tests don't need it."""
    import cv2
    if isinstance(rgb_or_path, (str, os.PathLike)):
        bgr = cv2.imread(str(rgb_or_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"could not read image: {rgb_or_path}")
    else:
        bgr = cv2.cvtColor(np.asarray(rgb_or_path, dtype=np.uint8),
                           cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    scale = LLM_MAX_SIDE / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, LLM_JPEG_QUALITY])
    if not ok:
        raise ValueError("jpeg encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii"), bytes(buf)


def _query_ollama(image_b64):
    """One Ollama vision call. Returns the parsed dict, or raises on any error."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": PROMPT,
        "images": [image_b64],
        "stream": False,
        "format": RESPONSE_SCHEMA,
        "options": {"temperature": 0.1, "num_ctx": OLLAMA_NUM_CTX},
    }
    req = urllib.request.Request(
        OLLAMA_HOST.rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_S) as r:
        resp = json.loads(r.read())
    return json.loads(resp["response"])


def _validate(scene):
    """Coerce / sanity-check the LLM dict; return None if unusable."""
    if not isinstance(scene, dict):
        return None
    out = {
        "scene": scene.get("scene") if scene.get("scene") in SCENES else "other",
        "mood": scene.get("mood") if scene.get("mood") in MOODS else "neutral",
        "warmth": scene.get("warmth") if scene.get("warmth") in WARMTHS else "neutral",
        "contrast": scene.get("contrast") if scene.get("contrast") in CONTRASTS else "normal",
        "rationale": str(scene.get("rationale", "")),
    }
    return out


def _load_cache(cache_path):
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def categorize_scene(rgb_or_path, cache_path=None, cache_key=None):
    """Classify the (already-inverted) preview with the local vision LLM.

    rgb_or_path : an RGB uint8 array OR an image path (the inverted preview).
    cache_path  : optional JSON file persisting results across runs/renders.
    cache_key   : stable key (e.g. the stem); the model name + image content
                  hash are folded in so a re-rendered preview OR a model change
                  re-queries.

    Returns the validated label dict, or None on ANY failure / low confidence
    (the caller then keeps the analytical params).
    """
    try:
        image_b64, raw = _encode_image(rgb_or_path)
    except Exception:
        return None

    key = None
    cache = {}
    if cache_path:
        digest = hashlib.sha1(raw).hexdigest()[:12]
        key = f"{cache_key or 'frame'}:{OLLAMA_MODEL}:{digest}"
        cache = _load_cache(cache_path)
        if key in cache:
            return _validate(cache[key])

    try:
        scene = _query_ollama(image_b64)
    except Exception:
        return None
    validated = _validate(scene)

    if cache_path and key is not None:
        # store the raw (pre-validation) dict so a vocabulary change re-validates
        cache[key] = scene
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
        except Exception:
            pass
    return validated


# ---------------------------------------------------------------------------
# Param nudging
# ---------------------------------------------------------------------------

def _warm_wb(wb, kind, shift, normalize):
    """Apply a warmth shift to a wb gain (shift>0 warmer: R up, B down), then
    restore darktable's picker normalization via `normalize`."""
    out = [wb[0] * (1.0 + shift), wb[1], wb[2] * (1.0 - shift)]
    return normalize(out, kind)


def apply_scene_tuning(params, scene, lin, border, hi_ceil_bias=0.0):
    """Nudge analytical params toward the scene's aesthetic. Pure given the
    inputs except for the print re-tune (which renders a subsampled frame).

    Returns (params_ai, info). With scene=None returns a copy of params
    unchanged so callers can treat the AI variant uniformly. The hard-clip
    guard inside tune_print_params is ALWAYS re-applied, so the AI variant can
    never clip more than the analytical one for the same scene.
    """
    import auto_negadoctor as an   # lazy: avoids the import cycle

    p = dict(params)
    if not scene:
        return p, {"tuned": False, "reason": "no scene"}

    info = {"tuned": True, "scene": dict(scene)}

    # --- gamma (paper grade) from the SCENE label + a small contrast nudge ---
    g = (p["gamma"] + SCENE_GAMMA.get(scene.get("scene"), 0.0)
         + CONTRAST_GAMMA.get(scene.get("contrast"), 0.0))
    lo = max(GAMMA_GT_RANGE[0], nm.GAMMA_RANGE[0])
    hi = min(GAMMA_GT_RANGE[1], nm.GAMMA_RANGE[1])
    p["gamma"] = nm.clamp(g, (lo, hi))

    # --- warmth: shift both wb gains, then re-normalize (disabled by default) ---
    shift = WARMTH_SHIFT.get(scene.get("warmth"), 0.0)
    if shift and p.get("wb_low") and p.get("wb_high"):
        p["wb_low"] = _warm_wb(p["wb_low"], "shadows", shift, an._normalize_wb)
        p["wb_high"] = _warm_wb(p["wb_high"], "highlights", shift, an._normalize_wb)

    # --- print tune: re-target to the user's print point, mood modulates it ---
    # Base ceiling is the AI re-target (AI_HI_CEIL), lowered further on dark/moody
    # scenes; re-tune exposure/black at that ceiling (analytical path untouched).
    ceil = nm.clamp(AI_HI_CEIL
                    + MOOD_CEIL_BIAS.get(scene.get("mood"), 0.0)
                    + float(hi_ceil_bias), CEIL_RANGE)
    info["hi_ceil"] = ceil
    try:
        p, tinfo = an.tune_print_params(lin, p, border, p["Dmin"], hi_ceil=ceil)
        info["print_tuning"] = tinfo
        # lift the black toward the user's printed shadows, then re-assert the
        # hard-clip guard (a less-negative black brightens, so it can re-clip).
        if AI_BLACK_LIFT:
            p["black"] = nm.clamp(p["black"] + AI_BLACK_LIFT, nm.BLACK_RANGE)
            info["black_lift"] = AI_BLACK_LIFT
            _clip_guard(p, lin, border)
    except Exception as e:
        info["print_error"] = str(e)
    return p, info


def _clip_guard(p, lin, border):
    """Back exposure off (in place) until the hard-clip fraction is within
    an.PRINT_CLIP_BUDGET — the same final guard tune_print_params applies, re-run
    here because the post-tune black lift can push highlights back over clip."""
    import auto_negadoctor as an
    l, t, r, b = border
    h, w = lin.shape[:2]
    s = max(1, int(round(w * an.PRINT_TUNE_SUBSAMPLE_FRAC)))
    region = lin[t:h - b:s, l:w - r:s].reshape(-1, 3)
    if region.size == 0:
        return
    for _ in range(8):
        out = nm.render_negadoctor(region, p)
        clip = float(np.mean((out >= 0.999).any(axis=1)))
        if clip <= an.PRINT_CLIP_BUDGET:
            break
        p["exposure"] = nm.clamp(p["exposure"] * 0.97, nm.EXPOSURE_RANGE)
