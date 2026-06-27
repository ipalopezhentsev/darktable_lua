"""Spec-06 ORACLE validation — go/no-go for the AI-critic architecture.

Question: can the critic's lever set (midtones/contrast/highlights + hi/shadow
cast), driven by CORRECT directions, actually move the analytical render toward
the user's GT picture? This is the cheap de-risking step BEFORE any LLM: derive
the "perfect" directions from the GT itself (the sign of each picture-stat gap
between the analytical render and the GT render), feed them to the deterministic
interpreter (critic_corrections.apply_corrections), iterate, and measure the
median histogram EMD-to-GT before vs after.

  - If the corrected EMD drops well below analytical, the MECHANISM works and the
    only remaining risk is whether an LLM emits decent signs (a far easier ask
    than param prediction) -> proceed to wire the model.
  - If it barely moves, the lever set cannot represent the GT picture and no
    amount of model quality helps -> stop. (Same spirit as the analytical-ridge
    gravestone.)

This is the OPTIMISTIC ceiling: the oracle knows GT and accepts a step only if it
reduces EMD. The LLM path will do worse; if even the ceiling is low, abandon.

Fully OFFLINE (no Ollama, no scene labels) -- needs each roll's local TIFFs.
Run: conda run -n autocrop python auto_negadoctor/tests/calibrate_ai_critic.py
"""
import os
import statistics
import sys
from pathlib import Path

import numpy as np

TESTS_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(TESTS_DIR.parent))

import auto_negadoctor as an
import critic_corrections as cc
import nega_model as nm
import run_quality_tests as rqt

gt_params_for_frame = rqt.gt_params_for_frame
_render_crop_rows = rqt._render_crop_rows
HIST_BINS = rqt.HIST_BINS

# Dead-bands: a picture-stat gap smaller than this is "ok" (don't chase noise).
THR_MID = 0.010      # median luma
THR_SPREAD = 0.012   # P90-P10 luma (contrast proxy)
THR_HI = 0.006       # P99.9 luma (highlight level)
THR_CAST = 0.006     # chroma-cast component (warm-cool / green-magenta)
MAX_ITERS = 8


def _median(vals):
    return statistics.median(vals) if vals else float("nan")


def _stats(rows):
    """Picture stats of (N,3) sRGB-float rows: midtone/contrast/highlight luma +
    the chroma (per-channel deviation from the pixel mean) of two tonal BANDS.

    Cast is measured over two ADJACENT broad bands that cover the WHOLE frame
    (no hole), so wb catches a midtone / overall cast (the 'daylight looks
    magenta' case) regardless of where the frame's mass sits. An earlier version
    left a P25..P50 gap, so DARK frames (bulk in that gap) saw no cast in either
    band and sat at 0 steps with a large residual. wb_high (the gain wheel) is
    targeted by everything above the lower third (excluding top-0.5% speculars);
    wb_low (the shadow wheel) by the lower third."""
    lum = rows @ nm._LUMA_W
    chroma = rows - rows.mean(axis=1, keepdims=True)
    split = np.percentile(lum, 33)
    p995 = np.percentile(lum, 99.5)

    def cmean(mask):
        sel = chroma[mask]
        return sel.mean(axis=0) if len(sel) else np.zeros(3)

    return {
        "mid": float(np.median(lum)),
        "spread": float(np.percentile(lum, 90) - np.percentile(lum, 10)),
        "hi": float(np.percentile(lum, 99.9)),
        "hi_chroma": cmean((lum > split) & (lum <= p995)),   # -> wb_high
        "sh_chroma": cmean(lum <= split),                    # -> wb_low
    }


def _cast_dir(dc):
    """Pick the dominant chroma-cast correction from a (gt - ana) chroma delta."""
    warm_cool = dc[0] - dc[2]                    # +ve => GT warmer => go warmer
    grn_mag = dc[1] - 0.5 * (dc[0] + dc[2])      # +ve => GT greener => go greener
    if max(abs(warm_cool), abs(grn_mag)) < THR_CAST:
        return "ok"
    if abs(warm_cool) >= abs(grn_mag):
        return "warmer" if warm_cool > 0 else "cooler"
    return "greener" if grn_mag > 0 else "magentaer"


def oracle_directions(ana_rows, gt_rows):
    """The 'perfect' directions: sign of each picture-stat gap (GT - analytical)."""
    a, g = _stats(ana_rows), _stats(gt_rows)
    d = {}
    dmid = g["mid"] - a["mid"]
    d["midtones"] = ("lighter" if dmid > THR_MID
                     else "darker" if dmid < -THR_MID else "ok")
    dsp = g["spread"] - a["spread"]
    d["contrast"] = ("punchier" if dsp > THR_SPREAD
                     else "flatter" if dsp < -THR_SPREAD else "ok")
    dhi = g["hi"] - a["hi"]
    d["highlights"] = ("lift" if dhi > THR_HI
                       else "tame" if dhi < -THR_HI else "ok")
    d["hi_cast"] = _cast_dir(g["hi_chroma"] - a["hi_chroma"])
    d["shadow_cast"] = _cast_dir(g["sh_chroma"] - a["sh_chroma"])
    return d


def _all_ok(d):
    return all(v == "ok" for v in d.values())


# Open-loop proxy: re-critique each pass and step in the directions with a step
# that DECAYS every iteration (annealing), so total movement is bounded and the
# loop provably converges WITHOUT any EMD/GT feedback — driven by SIGNS only.
# (A shrink-only-on-reversal line search drifts: a lever whose sign never flips
# marches unbounded; annealing fixes that.) This is the realistic LLM-in-a-loop.
OPENLOOP_START = 0.8
OPENLOOP_DECAY = 0.7
OPENLOOP_MIN = 0.03
OPENLOOP_ITERS = 16
_LEVERS = ("midtones", "contrast", "highlights", "hi_cast", "shadow_cast")


def correct_frame(ana_params, lin, border, gt_rows):
    """Greedy oracle (the optimistic CEILING): step in the GT-derived directions,
    accept only if EMD-to-GT drops, shrink the step on rejection. The oracle
    knows GT — an LLM will do worse. Returns (best_params, emd_curve)."""
    best_p = dict(ana_params)
    best_rows = _render_crop_rows(lin, border, best_p)
    best_emd = nm.histogram_distance(best_rows, gt_rows, bins=HIST_BINS)["total"]
    curve = [best_emd]
    scale = 1.0
    for _ in range(MAX_ITERS):
        dirs = oracle_directions(best_rows, gt_rows)
        if _all_ok(dirs):
            break
        cand = cc.apply_corrections(best_p, dirs, lin, border, scale=scale)
        cand_rows = _render_crop_rows(lin, border, cand)
        if cand_rows is None:
            break
        e = nm.histogram_distance(cand_rows, gt_rows, bins=HIST_BINS)["total"]
        if e < best_emd - 1e-5:
            best_p, best_rows, best_emd = cand, cand_rows, e
            curve.append(e)
        else:
            scale *= 0.5
            if scale < 0.125:
                break
    return best_p, curve


def correct_frame_openloop(ana_params, lin, border, gt_rows):
    """Open-loop PROXY (the realistic LLM-in-a-loop): re-critique each pass and
    step in the directions with an ANNEALED (decaying) step, NO EMD/GT feedback.
    Stops at 'natural', when the step is tiny, or MAX iters. The gap to the
    greedy ceiling shows how much the GT accept/reject crutch was really doing —
    i.e. whether correct SIGNS alone (no quality measure) suffice."""
    p = dict(ana_params)
    for k in range(OPENLOOP_ITERS):
        s = OPENLOOP_START * (OPENLOOP_DECAY ** k)
        if s < OPENLOOP_MIN:
            break
        rows = _render_crop_rows(lin, border, p)
        if rows is None:
            break
        dirs = oracle_directions(rows, gt_rows)
        active = {key: dirs[key] for key in _LEVERS if dirs.get(key, "ok") != "ok"}
        if not active:
            break
        cand = cc.apply_corrections(p, active, lin, border, scale=s)
        if cand is None:
            break
        p = cand
    return p


def _gt_fixtures(roll_info):
    """The annotation files to use as INVERSION ground truth. Prefer a curated
    'correct-inversion' session when present (the user's vetted GT for this roll,
    per 2026-06-27) so unrelated sessions (e.g. correct-crops) can't bleed into
    the wb/print targets; else fall back to every session under the roll."""
    ci = roll_info["dir"] / "correct-inversion"
    if ci.is_dir():
        fx = sorted(ci.glob("*_annotations.json"))
        if fx:
            return fx
    return roll_info["fixtures"]


def run_roll(roll_info):
    images, exif = roll_info["images"], roll_info["exif"]
    fixtures = _gt_fixtures(roll_info)
    print(f"\n=== Roll {roll_info['id']}: {len(images)} frame(s) "
          f"(GT: {len(fixtures)} annotation file(s)) ===")
    gt_by_stem = rqt._load_ground_truth(fixtures)
    if not gt_by_stem:
        print("  (no wb/print ground-truth annotations)")
        return None
    frames, roll = an.process_roll(images, exif)
    by_stem = {fr["stem"]: fr for fr in frames if not fr.get("error")}

    ana_emds, cor_emds, ol_emds = [], [], []
    cor_lumas, cor_colors = [], []
    improved = ol_improved = 0
    print(f"  {'stem':10s} {'ana_tot':>8s} {'orac':>8s} {'openloop':>8s} "
          f"{'cor_lum':>8s} {'cor_col':>8s} {'steps':>6s}")
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
        best_p, curve = correct_frame(fr["params"], lin, fr["border"], gt_rows)
        ol_p = correct_frame_openloop(fr["params"], lin, fr["border"], gt_rows)
        da = nm.histogram_distance(
            _render_crop_rows(lin, fr["border"], fr["params"]), gt_rows,
            bins=HIST_BINS)
        dc = nm.histogram_distance(_render_crop_rows(lin, fr["border"], best_p),
                                   gt_rows, bins=HIST_BINS)
        do = nm.histogram_distance(_render_crop_rows(lin, fr["border"], ol_p),
                                   gt_rows, bins=HIST_BINS)
        ana_e, cor_e, ol_e = da["total"], dc["total"], do["total"]
        ana_emds.append(ana_e); cor_emds.append(cor_e); ol_emds.append(ol_e)
        cor_lumas.append(dc["luma"]); cor_colors.append(dc["color"])
        if cor_e < ana_e - 1e-5:
            improved += 1
        if ol_e < ana_e - 1e-5:
            ol_improved += 1
        print(f"  {stem:10s} {ana_e:8.4f} {cor_e:8.4f} {ol_e:8.4f} "
              f"{dc['luma']:8.4f} {dc['color']:8.4f} {len(curve) - 1:6d}")

    n = len(ana_emds)
    if not n:
        print("  (no renderable GT frames)")
        return None
    ma, mc, mo = _median(ana_emds), _median(cor_emds), _median(ol_emds)
    print(f"\n  median EMD-to-GT: analytical {ma:.4f}  ->  oracle(ceiling) "
          f"{mc:.4f} ({(ma - mc) / ma * 100:+.0f}%)  ->  open-loop(proxy) "
          f"{mo:.4f} ({(ma - mo) / ma * 100:+.0f}%)")
    print(f"  oracle residual (median): luma {_median(cor_lumas):.4f}  "
          f"color {_median(cor_colors):.4f}")
    print(f"  improved vs analytical: oracle {improved}/{n}  "
          f"open-loop {ol_improved}/{n}")
    return {"roll": roll_info["id"], "n": n, "ana": ma, "cor": mc, "ol": mo,
            "improved": improved, "ol_improved": ol_improved}


def main():
    rolls = rqt.discover_rolls()
    if not rolls:
        print("SKIP: no annotated rolls with local TIFFs under", rqt.ROLLS_DIR)
        return 0
    results = [r for r in (run_roll(ri) for ri in rolls) if r]
    if not results:
        return 0
    print("\n" + "=" * 70)
    print("CRITIC ORACLE — median total EMD-to-GT, per roll:")
    print(f"  {'roll':14s} {'n':>3s} {'analytic':>9s} {'oracle':>9s} "
          f"{'openloop':>9s} {'orac drop':>9s} {'ol drop':>8s}")
    ana_all, cor_all, ol_all = [], [], []
    for r in results:
        print(f"  {r['roll']:14s} {r['n']:3d} {r['ana']:9.4f} {r['cor']:9.4f} "
              f"{r['ol']:9.4f} "
              f"{(r['ana'] - r['cor']) / r['ana'] * 100:+8.0f}% "
              f"{(r['ana'] - r['ol']) / r['ana'] * 100:+7.0f}%")
        ana_all.append(r["ana"]); cor_all.append(r["cor"]); ol_all.append(r["ol"])
    print(f"\n  mean-of-roll-medians: analytical {_median(ana_all):.4f}  ->  "
          f"oracle {_median(cor_all):.4f}  ->  open-loop {_median(ol_all):.4f}")
    print("\nReading:")
    print("  oracle (knows GT, accept-if-better) = the OPTIMISTIC ceiling.")
    print("  open-loop (correct signs, fixed step, no EMD feedback) = the proxy")
    print("  for an LLM-in-a-loop. If open-loop keeps most of the oracle's drop,")
    print("  a decent model's signs should pay off; if it collapses to ~analytic,")
    print("  the win needed the GT crutch and the LLM path is unlikely to deliver.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
