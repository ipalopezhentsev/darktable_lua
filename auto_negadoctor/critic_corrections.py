"""Spec-06 (AI critic) — the DETERMINISTIC interpreter.

The reframe (user, 2026-06-27): stop asking a model to PREDICT params (intent is
unknowable, params are ambiguous — spec 04). Instead show it the analytical
render (cropped to content) and have it emit a DIRECTION to move the picture
toward "natural" — e.g. "lighten midtones, tame highlights", "highlights look
magenta -> push yellower". This module is the half that owns the param
ambiguity + clip safety: it maps each qualitative direction to a bounded move on
the one lever that produces that histogram change, and keeps the result
clip-safe. The model (or, for validation, a GT-derived ORACLE) only supplies the
signs; this never asks it for a number.

Levers reused from production:
  - midtones lighter/darker  -> paper BLACK lift, holding the highlight level
    (the debug-UI brighten/darken combo: black moves midtones, exposure re-pins
    P99.9 so the move is clip-neutral).
  - contrast punchier/flatter -> paper GAMMA (grade), then re-pin the highlight.
  - highlights tame/lift     -> lower/raise the highlight target, re-solve EXPOSURE.
  - hi_cast / shadow_cast     -> nudge wb_high / wb_low in channel space, renormalized.
Every path ends in the production hard-clip guard (back exposure off until the
clip fraction is within PRINT_CLIP_BUDGET).

DIRECTIONS schema (all keys optional; "ok"/missing = leave that lever alone):
  {
    "midtones":    "darker" | "ok" | "lighter",
    "contrast":    "flatter" | "ok" | "punchier",
    "highlights":  "tame" | "ok" | "lift",
    "hi_cast":     "ok" | "warmer" | "cooler" | "greener" | "magentaer",
    "shadow_cast": "ok" | "warmer" | "cooler" | "greener" | "magentaer",
  }

`scale` (default 1.0) multiplies every step, so a search/oracle can take smaller
moves; the steps below are the unit (scale=1) magnitudes.
"""
import numpy as np

import auto_negadoctor as an
import nega_model as nm

# Unit step sizes (scaled by `scale`). Picked to match the debug-UI manual moves.
BLACK_STEP = 0.05          # paper-black lift per midtone step (on [-0.5, 0.5])
GAMMA_STEP = 0.5           # paper-grade step (on [1, 8])
HI_CEIL_STEP = 0.05        # highlight-target step (tone units, on the P99.9 pin)
WB_STEP = 0.06             # per-channel wb multiplier nudge, then renormalized

_CAST_DIRS = ("warmer", "cooler", "greener", "magentaer")


def _content(lin, border):
    """Subsampled (N,3) LINEAR content rows — the same region/stride the print
    tuner uses, so the solving here matches production."""
    l, t, r, b = border
    h, w = lin.shape[:2]
    s = max(1, int(round(w * an.PRINT_TUNE_SUBSAMPLE_FRAC)))
    return lin[t:h - b:s, l:w - r:s].reshape(-1, 3).astype(np.float64)


def _high(params, content, cfg):
    """Rendered highlight level (P99.9 of the per-pixel mean) — the statistic
    tune_print_params pins; measured in WORKING space, like production."""
    out = nm.render_negadoctor(content, params)
    return float(np.percentile(out.mean(axis=1), cfg.PRINT_HI_PCT))


def _exposure_for_high(params, content, target, cfg):
    """Exposure whose rendered high percentile == target (monotone in exposure ->
    bisect). Transcribed from debug_ui._exposure_for_high_pct."""
    lo, hi = nm.EXPOSURE_RANGE
    work = dict(params)

    def hp(e):
        work["exposure"] = e
        return _high(work, content, cfg)

    if hp(lo) >= target:
        return lo
    if hp(hi) <= target:
        return hi
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        if hp(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _clip_guard(p, content, cfg):
    """Production final guard: back exposure off until hard-clip <= budget."""
    ec = nm.EXPOSURE_RANGE
    for _ in range(6):
        out = nm.render_negadoctor(content, p)
        if float(np.mean((out >= 0.999).any(axis=1))) <= cfg.PRINT_CLIP_BUDGET:
            break
        p["exposure"] = nm.clamp(p["exposure"] * 0.97, ec)
    return p


def _nudge_wb(wb, direction, kind, step):
    """Move a wb vector in channel space then renormalize (low: max==1, high:
    min==1), clamped to WB_RANGE. Warm = +R/-B, cool = -R/+B, green = +G,
    magenta = -G."""
    w = [float(v) for v in wb]
    if direction == "warmer":
        w[0] *= 1.0 + step; w[2] *= 1.0 - step
    elif direction == "cooler":
        w[0] *= 1.0 - step; w[2] *= 1.0 + step
    elif direction == "greener":
        w[1] *= 1.0 + step
    elif direction == "magentaer":
        w[1] *= 1.0 - step
    else:
        return [nm.clamp(v, nm.WB_RANGE) for v in w]
    m = max(w) if kind == "low" else min(w)
    return [nm.clamp(v / m, nm.WB_RANGE) for v in w]


def apply_corrections(params, directions, lin, border, cfg=an.DEFAULT_TUNING,
                      scale=1.0):
    """Apply one pass of the critic's DIRECTIONS to `params`, returning new,
    clip-safe params. Pure (does not mutate `params`).

    `scale` is either a float (same for every lever) or a per-lever dict keyed by
    direction name (midtones/contrast/highlights/hi_cast/shadow_cast), so a
    closed loop can shrink one lever's step independently (line search)."""
    def _sc(name):
        return scale.get(name, 1.0) if isinstance(scale, dict) else scale

    p = dict(params)
    d = directions or {}
    content = _content(lin, border)
    if content.size == 0:
        return p

    # --- color casts (independent of tone) ---
    if d.get("hi_cast") in _CAST_DIRS:
        p["wb_high"] = _nudge_wb(p["wb_high"], d["hi_cast"], "high",
                                 WB_STEP * _sc("hi_cast"))
    if d.get("shadow_cast") in _CAST_DIRS:
        p["wb_low"] = _nudge_wb(p["wb_low"], d["shadow_cast"], "low",
                                WB_STEP * _sc("shadow_cast"))

    # --- contrast (gamma), holding the highlight level so it is a pure grade move ---
    if d.get("contrast") in ("flatter", "punchier"):
        keep = _high(p, content, cfg)
        step = GAMMA_STEP * _sc("contrast") * (1 if d["contrast"] == "punchier" else -1)
        p["gamma"] = nm.clamp(p["gamma"] + step, nm.GAMMA_RANGE)
        p["exposure"] = _exposure_for_high(p, content, keep, cfg)

    # --- midtones (paper black), holding the highlight level (brighten combo) ---
    if d.get("midtones") in ("darker", "lighter"):
        keep = _high(p, content, cfg)
        step = BLACK_STEP * _sc("midtones") * (1 if d["midtones"] == "lighter" else -1)
        p["black"] = nm.clamp(p["black"] + step, nm.BLACK_RANGE)
        p["exposure"] = _exposure_for_high(p, content, keep, cfg)

    # --- highlights: set the top explicitly (re-solve exposure to the new target) ---
    if d.get("highlights") in ("tame", "lift"):
        cur = _high(p, content, cfg)
        tgt = cur + HI_CEIL_STEP * _sc("highlights") * (
            1 if d["highlights"] == "lift" else -1)
        tgt = max(0.05, min(0.999, tgt))
        p["exposure"] = _exposure_for_high(p, content, tgt, cfg)

    return _clip_guard(p, content, cfg)
