"""Spec-06 — wire the LLM critic and measure it vs the oracle / analytical.

The offline de-risking (calibrate_ai_critic.py) proved: the levers reach GT
(oracle +62-72%) and CORRECT directional signs alone, through the annealing loop,
recover +40-47%. The ONE remaining unknown is how accurate a real LLM's signs
are. This wires it: render the analytical inversion CROPPED to content (no holder
garbage), color-managed to sRGB, downscaled; ask a local vision LLM which way the
picture is wrong ('too magenta / too dark / too flat'); feed those SIGNS through
the same annealing loop (re-critiqued each pass); and compare the LLM-corrected
median histogram EMD-to-GT against the oracle (GT-derived signs) and analytical.

Needs Ollama. Model via --model (default moondream). The interpreter + oracle are
imported from calibrate_ai_critic so the metric matches exactly.

Run: conda run -n autocrop python auto_negadoctor/tests/critic_llm_eval.py \
        --model moondream --rolls 2506-1
"""
import argparse
import base64
import hashlib
import json
import os
import statistics
import sys
import urllib.request
from pathlib import Path

import numpy as np

TESTS_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(TESTS_DIR.parent))

import auto_negadoctor as an
import calibrate_ai_critic as cac
import critic_corrections as cc
import nega_model as nm
import run_quality_tests as rqt

_render_crop_rows = rqt._render_crop_rows
gt_params_for_frame = rqt.gt_params_for_frame
HIST_BINS = rqt.HIST_BINS

OLLAMA_HOST = os.environ.get("NEGA_OLLAMA_HOST", "http://localhost:11434")
OLLAMA_NUM_CTX = int(os.environ.get("NEGA_OLLAMA_NUM_CTX", "4096"))
OLLAMA_TIMEOUT_S = float(os.environ.get("NEGA_OLLAMA_TIMEOUT", "300"))
LLM_MAX_SIDE = 512
LLM_JPEG_QUALITY = 85

# Annealing loop (same shape as the open-loop proxy; capped passes for LLM cost).
LOOP_START = 0.8
LOOP_DECAY = 0.6
LOOP_MIN = 0.04
MAX_PASSES = 4

_CAST = ["ok", "warmer", "cooler", "greener", "magentaer"]
_TONE = {"midtones": ["darker", "ok", "lighter"],
         "contrast": ["flatter", "ok", "punchier"],
         "highlights": ["tame", "ok", "lift"]}

CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        "midtones": {"type": "string", "enum": _TONE["midtones"]},
        "contrast": {"type": "string", "enum": _TONE["contrast"]},
        "highlights": {"type": "string", "enum": _TONE["highlights"]},
        "hi_cast": {"type": "string", "enum": _CAST},
        "shadow_cast": {"type": "string", "enum": _CAST},
        "looks_natural": {"type": "boolean"},
    },
    "required": ["rationale", "midtones", "contrast", "highlights", "hi_cast",
                 "shadow_cast", "looks_natural"],
}

# v2: default-OK framing (the v1 prompt invited a generic 'make it pop' critique
# on every frame — moondream emitted an IDENTICAL verdict for all 152 renders).
CRITIC_PROMPT = (
    "You are checking a photo (inverted from a color negative) for OBVIOUS "
    "processing errors. MOST of these photos are ALREADY correctly processed — so "
    "your default answer for every axis is 'ok'. Flag an axis ONLY if there is a "
    "CLEAR, strong problem an ordinary viewer would immediately notice; when in "
    "doubt, answer 'ok'. It is normal for ALL axes to be 'ok'.\n"
    "Natural reference: a daylight scene has NO strong color cast; a night/indoor "
    "scene is ALLOWED to be dark (do not brighten it); contrast should suit the "
    "scene.\n"
    "First write a one-sentence rationale describing what you actually see, THEN "
    "answer each axis (default 'ok'):\n"
    "- midtones: 'lighter' ONLY if clearly too dark/muddy; 'darker' ONLY if clearly "
    "washed out.\n"
    "- contrast: 'punchier' ONLY if clearly flat/hazy when it shouldn't be; "
    "'flatter' ONLY if clearly harsh.\n"
    "- highlights: 'tame' ONLY if the brights are clearly blown; 'lift' ONLY if "
    "clearly dull.\n"
    "- hi_cast / shadow_cast: name the tint to REMOVE ONLY if the bright / dark "
    "areas have a clear UNNATURAL color cast, else 'ok'.\n"
    "Set looks_natural=true when every axis is 'ok'. Respond ONLY with the JSON."
)
PROMPT_VERSION = "v2"


def _median(vals):
    return statistics.median(vals) if vals else float("nan")


def _encode_image(rgb_u8):
    import cv2
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, LLM_JPEG_QUALITY])
    if not ok:
        raise ValueError("jpeg encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii"), bytes(buf)


def render_crop_u8(lin, border, params, max_side=LLM_MAX_SIDE):
    """The analytical inversion CROPPED to content, color-managed sRGB, downscaled
    — exactly what the critic should judge (no holder/rebate)."""
    import cv2
    l, t, r, b = border
    h, w = lin.shape[:2]
    crop = lin[t:h - b, l:w - r]
    ch, cw = crop.shape[:2]
    if ch == 0 or cw == 0:
        return None
    scale = max_side / max(ch, cw)
    if scale < 1.0:
        crop = cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))),
                          interpolation=cv2.INTER_AREA)
    hh, ww = crop.shape[:2]
    srgb = nm.working_to_srgb(nm.render_negadoctor(crop.reshape(-1, 3), params))
    return np.clip(srgb.reshape(hh, ww, 3) * 255.0, 0, 255).astype(np.uint8)


def _validate(d):
    if not isinstance(d, dict):
        return None
    out = {"looks_natural": bool(d.get("looks_natural", False))}
    for k, allowed in (("midtones", _TONE["midtones"]), ("contrast", _TONE["contrast"]),
                       ("highlights", _TONE["highlights"]),
                       ("hi_cast", _CAST), ("shadow_cast", _CAST)):
        v = d.get(k)
        out[k] = v if v in allowed else "ok"
    out["rationale"] = str(d.get("rationale", ""))
    return out


def _query(image_b64, model):
    payload = {"model": model, "prompt": CRITIC_PROMPT, "images": [image_b64],
               "stream": False, "format": CRITIC_SCHEMA,
               "options": {"temperature": 0.0, "num_ctx": OLLAMA_NUM_CTX}}
    req = urllib.request.Request(
        OLLAMA_HOST.rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_S) as r:
        resp = json.loads(r.read())
    return json.loads(resp["response"])


def critic_directions(rgb_u8, model, cache):
    """One LLM critique of a rendered crop -> validated directions (cached by
    model+image hash, so a converged/repeated render never re-queries)."""
    image_b64, raw = _encode_image(rgb_u8)
    key = f"{model}:{PROMPT_VERSION}:{hashlib.sha1(raw).hexdigest()[:12]}"
    if key in cache:
        return _validate(cache[key]), key
    try:
        d = _query(image_b64, model)
    except Exception as e:
        return None, key
    cache[key] = d
    return _validate(d), key


def llm_correct_frame(ana_params, lin, border, model, cache, max_passes=MAX_PASSES):
    """Annealing loop driven by the LLM's signs (re-critiqued each pass)."""
    p = dict(ana_params)
    queries = 0
    for k in range(max_passes):
        s = LOOP_START * (LOOP_DECAY ** k)
        if s < LOOP_MIN:
            break
        img = render_crop_u8(lin, border, p)
        if img is None:
            break
        dirs, _ = critic_directions(img, model, cache)
        queries += 1
        if dirs is None:
            break
        if dirs.get("looks_natural"):
            break
        active = {key: dirs[key] for key in cac._LEVERS if dirs.get(key, "ok") != "ok"}
        if not active:
            break
        p = cc.apply_corrections(p, active, lin, border, scale=s)
    return p, queries


def run_roll(roll_info, model, cache, max_passes=MAX_PASSES):
    images, exif = roll_info["images"], roll_info["exif"]
    fixtures = cac._gt_fixtures(roll_info)
    gt_by_stem = rqt._load_ground_truth(fixtures)
    if not gt_by_stem:
        print(f"\n=== Roll {roll_info['id']}: (no GT) ===")
        return None
    print(f"\n=== Roll {roll_info['id']} [{model}]: {len(images)} frame(s), "
          f"GT {len(gt_by_stem)} ===")
    frames, roll = an.process_roll(images, exif)
    by_stem = {fr["stem"]: fr for fr in frames if not fr.get("error")}

    ana_e, ora_e, llm_e = [], [], []
    llm_better = 0
    print(f"  {'stem':10s} {'ana':>8s} {'oracle':>8s} {'llm':>8s} {'delta':>8s} "
          f"{'q':>3s}")
    for stem, gt in sorted(gt_by_stem.items()):
        fr = by_stem.get(stem)
        if not fr or fr.get("error") or "params" not in fr or not fr.get("border"):
            continue
        try:
            enc_f, lin = an.load_frame(fr["path"], fr.get("vignette"))
        except Exception as e:
            print(f"  {stem}: load failed ({e})")
            continue
        gt_rows = _render_crop_rows(lin, fr["border"], gt_params_for_frame(fr, gt))
        if gt_rows is None:
            continue
        ora_p, _ = cac.correct_frame(fr["params"], lin, fr["border"], gt_rows)
        llm_p, q = llm_correct_frame(fr["params"], lin, fr["border"], model, cache,
                                     max_passes)

        def emd(params):
            return nm.histogram_distance(
                _render_crop_rows(lin, fr["border"], params), gt_rows,
                bins=HIST_BINS)["total"]

        a, o, l = emd(fr["params"]), emd(ora_p), emd(llm_p)
        ana_e.append(a); ora_e.append(o); llm_e.append(l)
        if l < a - 1e-5:
            llm_better += 1
        print(f"  {stem:10s} {a:8.4f} {o:8.4f} {l:8.4f} {l - a:+8.4f} {q:3d}")

    n = len(ana_e)
    if not n:
        return None
    ma, mo, ml = _median(ana_e), _median(ora_e), _median(llm_e)
    print(f"\n  median EMD-to-GT: analytical {ma:.4f}  oracle {mo:.4f} "
          f"({(ma - mo) / ma * 100:+.0f}%)  LLM {ml:.4f} "
          f"({(ma - ml) / ma * 100:+.0f}%)")
    print(f"  LLM improved {llm_better}/{n} frame(s)")
    return {"roll": roll_info["id"], "n": n, "ana": ma, "oracle": mo, "llm": ml,
            "llm_better": llm_better}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="moondream")
    ap.add_argument("--rolls", nargs="+")
    ap.add_argument("--max-passes", type=int, default=MAX_PASSES)
    args = ap.parse_args()

    rolls = rqt.discover_rolls()
    if args.rolls:
        rolls = [r for r in rolls if r["id"] in args.rolls]
    if not rolls:
        print("SKIP: no annotated rolls with local TIFFs.")
        return 0

    cache_path = TESTS_DIR / f"critic_cache_{args.model.replace(':', '_')}.json"
    cache = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}

    results = []
    for ri in rolls:
        r = run_roll(ri, args.model, cache, args.max_passes)
        cache_path.write_text(json.dumps(cache, indent=2))   # persist progress
        if r:
            results.append(r)
    if not results:
        return 0

    print("\n" + "=" * 70)
    print(f"LLM CRITIC [{args.model}] — median total EMD-to-GT, per roll:")
    print(f"  {'roll':14s} {'n':>3s} {'analytic':>9s} {'oracle':>9s} {'LLM':>9s} "
          f"{'LLM drop':>9s} {'better':>8s}")
    for r in results:
        print(f"  {r['roll']:14s} {r['n']:3d} {r['ana']:9.4f} {r['oracle']:9.4f} "
              f"{r['llm']:9.4f} {(r['ana'] - r['llm']) / r['ana'] * 100:+8.0f}% "
              f"{r['llm_better']:>5d}/{r['n']}")
    print("\nReading: LLM drop near the oracle => its signs are good, ship it. "
          "Near 0\nor negative => its signs are too noisy; the model is the "
          "bottleneck (try a\nstronger one). Compare to the +40-47% the open-loop "
          "got with PERFECT signs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
