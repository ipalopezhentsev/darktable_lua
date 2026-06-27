"""Reproducible, RECORDED calibration sessions (spec 05) — auto_negadoctor.

The shared engine (optimizers, recorded-session lifecycle, CLI) lives in
`common/calibration/runner.py`; this module supplies only the negadoctor-specific
pieces: the per-kind EVALUATORS (the metric), roll discovery, the parallel map, the
per-kind cosmetics, and the `--review` debug-UI hook — wired through a
`runner.CalibrationAdapter`.

Three KINDS, each with its own metric and history index (INDEX_<kind>.md):

  crop       total OVER-TRIM (fraction of frame area of content the detected crop
             removed inside the user's hand-drawn rect); containment is a HARD
             constraint.
  inversion  median histogram EMD between the algorithm's RENDER and the user's
             GT-param render over the content crop; rendered hard-clip is a HARD
             constraint.
  vignette   per roll: did fit_vignette_profile produce a valid fit? + its residual.

TUNING ORDER: tune `crop` and `vignette` FIRST, then `inversion`. RECORD-ONLY: the
runner never edits auto_negadoctor.py; the user adopts fitted values by hand.

Run (evaluate the current algorithm, no search):
  conda run -n autocrop python auto_negadoctor/tests/run_calibration.py \
      --config .../configs/inversion_default.json --method none
Run a fit:           ... --method coordinate_descent
Review a session:    ... --review <session_dir> [roll_id]
"""

import contextlib
import json
import shutil
import subprocess
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import cv2

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent.parent))   # repo root -> common
sys.path.insert(0, str(TESTS_DIR.parent))
sys.path.insert(0, str(TESTS_DIR))

import auto_negadoctor as an          # noqa: E402
import nega_model as nm               # noqa: E402
import run_quality_tests as rqt       # noqa: E402
import calibration_registry as reg    # noqa: E402
from common.calibration import runner  # noqa: E402

CALIB_DIR = TESTS_DIR / "calibrations"
ROLLS_DIR = rqt.ROLLS_DIR
BIG_PENALTY = runner.BIG_PENALTY

# Thin aliases so the evaluator bodies (below) read exactly as before, while the
# generic machinery lives in the shared runner.
_median = runner._median
_proc_cb = runner._proc_cb
_per_roll_summary = runner.per_roll_summary


def _eval_frames(items, fn):
    return runner.eval_frames(ADAPTER, items, fn)


def _map_frames_prep(roll_id, label, fn, items):
    return runner.map_frames_prep(ADAPTER, roll_id, label, fn, items)


METRIC_NAME = {
    "crop": "crop_overtrim_area_fraction (containment HARD unless "
            "containment_weight set -> soft overtrim + W*undertrim)",
    "inversion": "histogram_emd_total (render vs GT-render, picture-vs-picture)",
    "vignette": "vignette_fit_rejected_count + median_residual",
}


# ---------------------------------------------------------------------------
# Roll discovery
# ---------------------------------------------------------------------------

def discover_image_rolls(roll_ids=None):
    """Rolls with local TIFFs (every kind derives what it needs from them — crop &
    inversion render, vignette captures its envelope). Optionally filtered."""
    rolls = rqt.discover_rolls()
    if roll_ids:
        want = set(roll_ids)
        rolls = [r for r in rolls if r["id"] in want]
    return rolls


# ---------------------------------------------------------------------------
# Per-kind evaluators. Each returns an `evaluate(overrides) -> result` closure plus
# the rolls it loaded; result = {objective, per_frame, per_roll, aggregate}.
# ---------------------------------------------------------------------------

def _crop_containment_weight(tolerances):
    """How hard the crop objective punishes leaving film border outside the user
    rect. None (the key absent) keeps the original HARD rule (+BIG_PENALTY per
    violating frame). A NUMBER switches to a symmetric soft penalty: objective =
    overtrim_total + W * undertrim_total (both frame-area fractions)."""
    w = tolerances.get("containment_weight")
    return None if w is None else float(w)


def make_crop_evaluator(rolls, tolerances, fit_params=()):
    """Crop detection runs on the per-frame buffer + the global-base dmin, neither of
    which depends on the crop constants. We prep those ONCE and also precompute
    detect_content_crop's heavy trial-INVARIANT pixel math (`_crop_fields`) so per
    trial only the cheap `_crop_decide` runs (reads every fittable crop constant)."""
    prepped, prep_secs = [], {}
    nrolls = len(rolls)
    for i, roll in enumerate(rolls, 1):
        t0 = time.perf_counter()
        print(f"[prep {i}/{nrolls}] roll {roll['id']}: analysing "
              f"{len(roll['images'])} frame(s)…", flush=True)
        frames, _ = an.process_roll(roll["images"], roll["exif"],
                                    progress=_proc_cb(roll["id"]))
        by_stem = {fr["stem"]: fr for fr in frames if not fr.get("error")}
        crop_fix = [f for f in roll["fixtures"]
                    if (json.loads(f.read_text()).get("crop_correction") or {})
                    .get("corrected")]
        prep_items = []
        for f in crop_fix:
            data = json.loads(f.read_text())
            crop = data["crop_correction"]
            stem = data.get("stem") or f.name.replace("_annotations.json", "")
            fr = by_stem.get(stem)
            if fr and "border" in fr:
                prep_items.append((stem, fr, crop))

        def _prep_one(item):
            stem, fr, crop = item
            try:
                enc_f, lin = an.load_frame(fr["path"], fr.get("vignette"))
            except Exception:
                return None
            return stem, {"fields": an._crop_fields(lin, fr["dmin"]),
                          "width": fr["width"], "height": fr["height"],
                          "crop": crop["corrected"]}

        done = _map_frames_prep(roll["id"], "crop", _prep_one, prep_items)
        cache = {stem: c for stem, c in done}
        prepped.append({"id": roll["id"], "cache": cache})
        prep_secs[roll["id"]] = round(time.perf_counter() - t0, 2)
        print(f"[prep {i}/{nrolls}] roll {roll['id']}: {len(cache)} crop frame(s) "
              f"ready in {prep_secs[roll['id']]}s", flush=True)

    items = [(roll["id"], stem, c) for roll in prepped
             for stem, c in roll["cache"].items()]
    cw = _crop_containment_weight(tolerances)

    def evaluate(overrides):
        cfg = reg.to_tuning(overrides)   # immutable per-trial config (no globals)

        def _one(it):
            rid, stem, c = it
            border = an._crop_decide(c["fields"], cfg)
            rec = rqt.crop_overtrim(border, c["crop"], c["width"], c["height"])
            return dict(rec, roll=rid, stem=stem)

        per_frame = _eval_frames(items, _one)
        viols = [r for r in per_frame if not r["contained"]]
        overtrim = [r["overtrim_area"] for r in per_frame]
        undertrim = [r["undertrim_area"] for r in per_frame]
        if cw is None:                       # HARD containment (original rule)
            objective = sum(overtrim) + BIG_PENALTY * len(viols)
        else:                                # soft, area-weighted containment
            objective = sum(overtrim) + cw * sum(undertrim)
        aggregate = {
            "total_overtrim_area": sum(overtrim),
            "median_overtrim_area": _median(overtrim),
            "max_overtrim_area": max(overtrim) if overtrim else 0.0,
            "total_undertrim_area": sum(undertrim),
            "max_undertrim_area": max(undertrim) if undertrim else 0.0,
            "containment_violations": len(viols),
            "containment_weight": cw,
            "n_frames": len(per_frame),
        }
        per_roll = _per_roll_summary(per_frame, "overtrim_area")
        return {"objective": objective, "per_frame": per_frame,
                "per_roll": per_roll, "aggregate": aggregate}

    return evaluate, prep_secs


# ---------------------------------------------------------------------------
# OPTIONAL calibration downsampling (inversion only). The per-trial param
# re-derivation (stage B1/B2 in the full path; tune_print_params in the fast
# path) is the slow part and is memory-bandwidth bound, so running it on a
# buffer downsampled Nx makes each trial ~Nx cheaper. Vignette + film base stay
# at full resolution (computed once in the prefix; their params are fractions /
# a colour, resolution-independent), and the EMD that scores each trial is still
# measured on the FULL-res buffer — only the params are derived on fewer pixels.
# Measured effect on roll 2506-1 at 2x: per-trial ~2.7x faster, params drift
# within GT tolerances (exposure ~0.01, wb ~0.01), median per-frame EMD shift
# ~4%. ALWAYS re-validate the fitted preset at full res before adopting it.
# OFF (factor None/1) => byte-identical to the previous behavior.
# ---------------------------------------------------------------------------

def _downsample_lin(arr, factor):
    """Area-resample a linear working buffer by `factor` (the faithful analog of a
    smaller darktable export). factor in (None, 1) -> the array unchanged."""
    if not factor or factor == 1:
        return arr
    h, w = arr.shape[:2]
    nw, nh = max(1, int(round(w / factor))), max(1, int(round(h / factor)))
    return cv2.resize(np.asarray(arr, np.float32), (nw, nh),
                      interpolation=cv2.INTER_AREA)


def _scale_border(border, w_src, w_dst):
    """Scale a (l, t, r, b) crop from a w_src-wide grid to a w_dst-wide grid."""
    s = w_dst / w_src
    return tuple(int(round(v * s)) for v in border)


def _down_prefix(prefix, factor):
    """A copy of a process_roll_prefix result with every frame's decoded buffer +
    film-base rect downsampled by `factor`; the roll-wide vignette + global film
    base (winner_rgb/winner_factor — a colour + a scalar) stay at full res. The
    per-trial process_roll(prep=...) then runs stage B1/B2 on the small buffers.
    factor None/1 -> the prefix unchanged (no copy)."""
    if not factor or factor == 1:
        return prefix
    new = dict(prefix)
    frames = []
    for fr in prefix["frames"]:
        g = dict(fr)
        loaded = fr.get("_loaded")
        if not fr.get("error") and loaded is not None:
            enc_f, lin = loaded
            lin_d = _downsample_lin(lin, factor)
            enc_d = (_downsample_lin(enc_f, factor) if enc_f.any()
                     else np.zeros_like(lin_d))
            g["_loaded"] = (enc_d, lin_d)
            g["width"], g["height"] = lin_d.shape[1], lin_d.shape[0]
            if fr.get("base") and fr["base"].get("rect"):
                x, y, w, h = fr["base"]["rect"]
                g["base"] = dict(fr["base"], rect=[
                    int(round(x / factor)), int(round(y / factor)),
                    max(1, int(round(w / factor))), max(1, int(round(h / factor)))])
        frames.append(g)
    new["frames"] = frames
    return new


def _inversion_result(per_frame, clip_budget):
    """Common aggregate/objective for both inversion paths (median histogram EMD; a
    frame clipping past the budget adds BIG_PENALTY)."""
    totals = [r["total"] for r in per_frame]
    n_clip = sum(1 for r in per_frame if r["clip"] > clip_budget)
    objective = (_median(totals) if totals else BIG_PENALTY) \
        + BIG_PENALTY * n_clip
    aggregate = {
        "median_total": _median(totals),
        "median_luma": _median([r["luma"] for r in per_frame]),
        "median_color": _median([r["color"] for r in per_frame]),
        "median_hi999": _median([r["hi999"] for r in per_frame]),
        "max_clip": max((r["clip"] for r in per_frame), default=0.0),
        "frames_over_clip_budget": n_clip,
        "clip_budget": clip_budget,
        "n_frames": len(per_frame),
    }
    return {"objective": objective, "per_frame": per_frame,
            "per_roll": _per_roll_summary(per_frame, "total"),
            "aggregate": aggregate}


def _clip_frac(lin, border, params):
    """Hard-clip fraction of the rendered production params over the content crop."""
    l, t, r, b = border
    h, w = lin.shape[:2]
    s = max(1, int(round(w * an.PRINT_TUNE_SUBSAMPLE_FRAC)))
    region = lin[t:h - b:s, l:w - r:s].reshape(-1, 3)
    if region.size == 0:
        return 0.0
    out = nm.render_negadoctor(region, params)
    return float(np.mean((out >= rqt.CLIP_OUT_THR).any(axis=1)))


def make_inversion_evaluator(rolls, tolerances, fit_params=()):
    """Picture-vs-picture histogram EMD between the algorithm's render and the user's
    GT-param render. Dispatches FAST (print-tune-only) vs FULL (any wb/picker param).

    tolerances["downsample"] (optional int): re-derive params per trial on a buffer
    downsampled this many times — ~Nx faster trials, params drift slightly, EMD
    still measured full-res (see _downsample_lin and the section comment)."""
    clip_budget = float(tolerances.get("clip_max_frac", rqt.CLIP_MAX_FRAC))
    downsample = int(tolerances.get("downsample") or 1)
    fast = set(fit_params) <= set(reg.PRINT_TUNE_PARAMS)
    if fast:
        return _make_inversion_fast(rolls, clip_budget, downsample)
    return _make_inversion_full(rolls, clip_budget, fit_params, downsample)


def _make_inversion_fast(rolls, clip_budget, downsample=1):
    prepped, prep_secs = [], {}
    nrolls = len(rolls)
    for i, roll in enumerate(rolls, 1):
        t0 = time.perf_counter()
        print(f"[prep {i}/{nrolls}] roll {roll['id']}: analysing "
              f"{len(roll['images'])} frame(s)…", flush=True)
        frames, _ = an.process_roll(roll["images"], roll["exif"],
                                    progress=_proc_cb(roll["id"]))
        gt = rqt._load_ground_truth(roll["fixtures"])
        by_stem = {fr["stem"]: fr for fr in frames if not fr.get("error")}
        prep_items = [(stem, by_stem[stem], g) for stem, g in gt.items()
                      if by_stem.get(stem) and "params" in by_stem[stem]
                      and by_stem[stem].get("border")]

        def _prep_one(item):
            stem, fr, g = item
            try:
                enc_f, lin = an.load_frame(fr["path"], fr.get("vignette"))
            except Exception:
                return None
            gt_f = rqt._render_crop_rows(lin, fr["border"],
                                         rqt.gt_params_for_frame(fr, g))
            if gt_f is None:
                return None
            # tune_print_params runs per trial; downsample the buffer it sees
            # (the wb/pickers are full-res-derived above and unchanged). The EMD
            # render below still uses the full-res `lin`/`border`.
            lin_tune = _downsample_lin(lin, downsample)
            border_tune = (_scale_border(fr["border"], lin.shape[1],
                                         lin_tune.shape[1])
                           if downsample > 1 else fr["border"])
            return stem, {
                "lin": lin, "border": fr["border"], "dmin": fr["dmin"],
                "d_max": fr["d_max"], "offset": fr["offset"],
                "wb_low": fr["wb_low"], "wb_high": fr["wb_high"],
                "picked_min": fr["picked_min"], "picked_max": fr["picked_max"],
                "gt_f": gt_f, "lin_tune": lin_tune, "border_tune": border_tune,
            }

        done = _map_frames_prep(roll["id"], "GT", _prep_one, prep_items)
        cache = {stem: c for stem, c in done}
        prepped.append({"id": roll["id"], "cache": cache})
        prep_secs[roll["id"]] = round(time.perf_counter() - t0, 2)
        print(f"[prep {i}/{nrolls}] roll {roll['id']}: {len(cache)} GT frame(s) "
              f"ready in {prep_secs[roll['id']]}s", flush=True)

    items = [(roll["id"], stem, c) for roll in prepped
             for stem, c in roll["cache"].items()]

    def evaluate(overrides):
        cfg = reg.to_tuning(overrides)   # immutable per-trial config (no globals)

        def _one(it):
            rid, stem, c = it
            p = an.make_params(c["dmin"], c["d_max"], c["offset"],
                               c["wb_low"], c["wb_high"],
                               c["picked_min"], c["picked_max"], cfg=cfg)
            tuned, info = an.tune_print_params(c["lin_tune"], p, c["border_tune"],
                                               c["dmin"], cfg=cfg)
            prod_f = rqt._render_crop_rows(c["lin"], c["border"], tuned)
            if prod_f is None:
                return None
            d = nm.histogram_distance(prod_f, c["gt_f"], bins=rqt.HIST_BINS)
            return {"roll": rid, "stem": stem,
                    "total": d["total"], "luma": d["luma"],
                    "color": d["color"], "hi999": abs(d["hi999"]),
                    "clip": float(info.get("clip") or 0.0)}

        return _inversion_result(_eval_frames(items, _one), clip_budget)

    return evaluate, prep_secs


def _make_inversion_full(rolls, clip_budget, fit_params=(), downsample=1):
    """Stage-B1 params (wb / pickers) reshape the analysis, so the pipeline is re-run
    per trial. The roll-wide vignette + film-base PREFIX is computed ONCE and reused.

    When `downsample` > 1 the per-trial process_roll runs on a downsampled copy of
    the prefix (stage B1/B2 on ~Nx fewer pixels), while the prefix itself
    (vignette + film base) and the EMD scoring buffers stay full-res."""
    leaked = set(fit_params) & set(reg.BASE_PREFIX_PARAMS)
    assert not leaked, (
        f"inversion cannot fit the vignette/film-base PREFIX constants {sorted(leaked)} "
        "— they are computed once per session, never calibrated by inversion")
    prepped, prep_secs = [], {}
    nrolls = len(rolls)
    for i, roll in enumerate(rolls, 1):
        t0 = time.perf_counter()
        print(f"[prep {i}/{nrolls}] roll {roll['id']} (full path): analysing "
              f"{len(roll['images'])} frame(s)…", flush=True)
        prefix = an.process_roll_prefix(roll["images"], roll["exif"],
                                        progress=_proc_cb(roll["id"]),
                                        cfg=an.DEFAULT_TUNING)
        frames, _ = an.process_roll(roll["images"], roll["exif"],
                                    cfg=an.DEFAULT_TUNING, prep=prefix)
        gt = rqt._load_ground_truth(roll["fixtures"])
        by_stem = {fr["stem"]: fr for fr in frames if not fr.get("error")}
        prep_items = [(stem, by_stem[stem], g) for stem, g in gt.items()
                      if by_stem.get(stem) and "params" in by_stem[stem]
                      and by_stem[stem].get("border")]

        def _prep_one(item):
            stem, fr, g = item
            try:
                enc_f, lin = an.load_frame(fr["path"], fr.get("vignette"))
            except Exception:
                return None
            gt_f = rqt._render_crop_rows(lin, fr["border"],
                                         rqt.gt_params_for_frame(fr, g))
            if gt_f is None:
                return None
            return stem, {"lin": lin, "border": fr["border"], "gt_f": gt_f}

        done = _map_frames_prep(roll["id"], "GT", _prep_one, prep_items)
        cache = {stem: c for stem, c in done}
        # the per-trial process_roll reuses a (optionally downsampled) prefix; the
        # full-res prefix is no longer needed once its copy is built (it is GC'd).
        prepped.append({"id": roll["id"], "images": roll["images"],
                        "exif": roll["exif"], "cache": cache,
                        "prefix": _down_prefix(prefix, downsample)})
        prep_secs[roll["id"]] = round(time.perf_counter() - t0, 2)
        ds_note = (f" (B1/B2 buffers downsampled {downsample}x)"
                   if downsample > 1 else "")
        print(f"[prep {i}/{nrolls}] roll {roll['id']}: {len(cache)} GT frame(s) "
              f"ready in {prep_secs[roll['id']]}s. NOTE: full path re-runs only "
              f"stage B1/B2 per trial — vignette + film base computed ONCE{ds_note}.",
              flush=True)

    def _one(it):
        rid, stem, c, params = it
        prod_f = rqt._render_crop_rows(c["lin"], c["border"], params)
        if prod_f is None:
            return None
        d = nm.histogram_distance(prod_f, c["gt_f"], bins=rqt.HIST_BINS)
        return {"roll": rid, "stem": stem,
                "total": d["total"], "luma": d["luma"],
                "color": d["color"], "hi999": abs(d["hi999"]),
                "clip": _clip_frac(c["lin"], c["border"], params)}

    def evaluate(overrides):
        cfg = reg.to_tuning(overrides)   # immutable per-trial config (no globals)
        per_frame = []
        # When this trial is ALREADY running in a parallel optimizer worker
        # (cmaes/random/spsa), process_roll must not spawn its own per-frame pool
        # on top — that nests P trials * 8 workers and thrashes the memory bus
        # (_eval_frames already serializes itself the same way via _trial_local).
        ctx = (an.serial_frames() if getattr(runner._trial_local, "serial", False)
               else contextlib.nullcontext())
        with ctx:
            for roll in prepped:
                frames, _ = an.process_roll(roll["images"], roll["exif"], cfg=cfg,
                                            prep=roll["prefix"])
                by_stem = {fr["stem"]: fr for fr in frames if not fr.get("error")}
                items = [(roll["id"], stem, c, by_stem[stem]["params"])
                         for stem, c in roll["cache"].items()
                         if by_stem.get(stem) and "params" in by_stem[stem]]
                per_frame += _eval_frames(items, _one)
        return _inversion_result(per_frame, clip_budget)

    return evaluate, prep_secs


def _vignette_result(per_frame):
    residuals = [r["residual"] for r in per_frame
                 if not r["rejected"] and r["residual"] is not None]
    n_rej = sum(1 for r in per_frame if r["rejected"])
    objective = BIG_PENALTY * n_rej + (_median(residuals) if residuals else 0.0)
    aggregate = {"rejected_rolls": n_rej, "median_residual": _median(residuals),
                 "max_residual": max(residuals) if residuals else 0.0,
                 "n_frames": len(per_frame)}
    per_roll = {r["roll"]: {"n": 1, "rejected": r["rejected"],
                            "residual": r["residual"],
                            "corner_falloff": r["corner_falloff"]}
                for r in per_frame}
    return {"objective": objective, "per_frame": per_frame,
            "per_roll": per_roll, "aggregate": aggregate}


def _vig_record(roll_id, params, info):
    return {"roll": roll_id, "stem": "(roll)", "rejected": params is None,
            "residual": info.get("residual"),
            "corner_falloff": info.get("corner_falloff"),
            "reason": info.get("reason"), "strength": (params or {}).get("strength")}


def _envelope_record(roll_id, env, cfg=None):
    """A vignette per-roll record from a (possibly empty) captured envelope."""
    if env["r"] is None:
        return _vig_record(roll_id, None, {"reason": env["reason"],
                                           "frames": env["used"]})
    cfg = cfg if cfg is not None else an.DEFAULT_TUNING
    return _vig_record(roll_id,
                       *an.fit_vignette_profile(env["r"], env["e"], env["used"],
                                                cfg))


def make_vignette_evaluator(rolls, tolerances, fit_params=()):
    """A roll whose fit_vignette_profile returns None (rejected) dominates the
    objective. Dispatches FAST (profile-fit only) vs FULL (re-fold the envelope)."""
    fast = set(fit_params) <= set(reg.VIG_FIT_PARAMS)
    prep_secs = {}

    if fast:
        env_rolls, nrolls = [], len(rolls)
        for i, roll in enumerate(rolls, 1):
            t0 = time.perf_counter()
            print(f"[prep {i}/{nrolls}] roll {roll['id']}: capturing vignette "
                  f"envelope from {len(roll['images'])} frame(s)…", flush=True)
            env = an.vignette_envelope(roll["images"], roll["exif"])
            env_rolls.append({"id": roll["id"], "env": env})
            prep_secs[roll["id"]] = round(time.perf_counter() - t0, 2)
            ok = "captured" if env["r"] else f"UNAVAILABLE ({env['reason']})"
            print(f"[prep {i}/{nrolls}] roll {roll['id']}: envelope {ok} in "
                  f"{prep_secs[roll['id']]}s", flush=True)

        def evaluate(overrides):
            cfg = reg.to_tuning(overrides)
            return _vignette_result([_envelope_record(er["id"], er["env"], cfg)
                                     for er in env_rolls])
    else:
        cached, nrolls = [], len(rolls)
        for i, roll in enumerate(rolls, 1):
            t0 = time.perf_counter()
            print(f"[prep {i}/{nrolls}] roll {roll['id']}: decoding "
                  f"{len(roll['images'])} frame(s) for the vignette cache…",
                  flush=True)
            fc = an.vignette_frame_cache(roll["images"], roll["exif"])
            cached.append({"id": roll["id"], "images": roll["images"],
                           "exif": roll["exif"], "frame_cache": fc})
            prep_secs[roll["id"]] = round(time.perf_counter() - t0, 2)
            ndec = sum(1 for c in fc if c is not None)
            print(f"[prep {i}/{nrolls}] roll {roll['id']}: {ndec}/{len(fc)} "
                  f"frame(s) decoded + cached in {prep_secs[roll['id']]}s "
                  "(trials re-fold from RAM — no more disk reads)", flush=True)

        def evaluate(overrides):
            cfg = reg.to_tuning(overrides)
            # See _make_inversion_full: keep the per-frame envelope re-fold serial
            # when this trial is itself a parallel optimizer worker.
            ctx = (an.serial_frames()
                   if getattr(runner._trial_local, "serial", False)
                   else contextlib.nullcontext())
            with ctx:
                return _vignette_result([
                    _vig_record(roll["id"],
                                *an.estimate_vignette(roll["images"], roll["exif"],
                                                      frame_cache=roll["frame_cache"],
                                                      cfg=cfg))
                    for roll in cached])

    return evaluate, prep_secs


EVALUATORS = {
    "crop": make_crop_evaluator,
    "inversion": make_inversion_evaluator,
    "vignette": make_vignette_evaluator,
}


# ---------------------------------------------------------------------------
# Per-kind cosmetics + review (the adapter hooks)
# ---------------------------------------------------------------------------

def _worst_key(kind):
    return {"crop": ("overtrim_area", True),
            "inversion": ("total", True),
            "vignette": ("residual", True)}[kind]


def _headline(kind, agg):
    if kind == "crop":
        cw = agg.get("containment_weight")
        tail = (f"viol {agg['containment_violations']} (HARD)" if cw is None
                else f"undertrim {runner._fmt(agg.get('total_undertrim_area', 0.0))} "
                     f"(W={runner._fmtv(cw)}, viol {agg['containment_violations']})")
        return f"overtrim {runner._fmt(agg['total_overtrim_area'])} " + tail
    if kind == "inversion":
        return (f"EMD {runner._fmt(agg['median_total'])} "
                f"clip {agg['frames_over_clip_budget']}")
    return (f"rejected {agg['rejected_rolls']} "
            f"resid {runner._fmt(agg['median_residual'])}")


def _review_payload(kind, fr, roll_meta):
    """The kind-specific result the debug UI's R toggle swaps (live vs fitted)."""
    if kind == "crop":
        return {"border": list(fr["border"])}
    if kind == "vignette":
        return {"vignette": (roll_meta or {}).get("vignette")}
    return {"params": fr["params"], "params_hex": fr["params_hex"]}  # inversion


def _review_run(kind, roll, cfg):
    """Run the pipeline once with `cfg` and return {stem: payload} + roll_meta."""
    frames, roll_meta = an.process_roll(roll["images"], roll["exif"], cfg=cfg)
    out = {}
    for fr in frames:
        if fr.get("error") or "params" not in fr or not fr.get("border"):
            continue
        out[fr["stem"]] = _review_payload(kind, fr, roll_meta)
    return out, frames, roll_meta


def _review_gt_payloads(kind, roll, base_frames):
    """The GROUND-TRUTH payload per stem, built from the user's annotations (the
    calibration target), so the debug UI's R cycle can show GT as a third source.
    A frame with NO annotation for this kind gets no entry (the UI cycle then
    skips GT for it). `base_frames` supply per-frame context (production params
    for the un-annotated fields, frame dims) — the LIVE run's frames."""
    by_stem = {fr["stem"]: fr for fr in base_frames if not fr.get("error")}
    out = {}
    if kind == "inversion":
        gt = rqt._load_ground_truth(roll["fixtures"])
        for stem, g in gt.items():
            fr = by_stem.get(stem)
            if not fr or "params" not in fr:
                continue
            p = rqt.gt_params_for_frame(fr, g)
            out[stem] = {"params": p, "params_hex": nm.encode_negadoctor_params(p)}
    elif kind == "crop":
        for f in roll["fixtures"]:
            data = json.loads(f.read_text())
            crop = data.get("crop_correction")
            if not crop or not crop.get("corrected"):
                continue
            stem = data.get("stem") or f.name.replace("_annotations.json", "")
            fr = by_stem.get(stem)
            if fr is None or "width" not in fr or "height" not in fr:
                continue
            x, y, cw, ch = rqt._rect_to_px(crop["corrected"],
                                           fr["width"], fr["height"])
            out[stem] = {"border": [x, y, fr["width"] - x - cw,
                                    fr["height"] - y - ch]}
    # vignette: no per-frame GT annotation -> no GT source
    return out


def review_session(session_dir, roll_id=None):
    """Open the debug UI on a finished session showing its FITTED result, with the R
    toggle flipping to the LIVE (current source-code) result. N toggles vignette."""
    session = Path(session_dir)
    config = json.loads((session / "config.json").read_text())
    results = json.loads((session / "results.json").read_text())
    kind = config["kind"]
    fitted = results.get("fitted") or {}
    roll_ids = [roll_id] if roll_id else config["rolls"]
    rolls = discover_image_rolls(roll_ids)
    if not rolls:
        print("No local TIFFs for the session's rolls — repopulate them "
              "(fixtures/rolls/README.md) to review.")
        return None
    roll = rolls[0]

    review_dir = Path(tempfile.mkdtemp(prefix="nega_review_"))
    live_payloads, live_frames, _ = _review_run(kind, roll, an.DEFAULT_TUNING)
    fit_payloads, frames, roll_meta = _review_run(kind, roll,
                                                  reg.to_tuning(fitted))
    gt_payloads = _review_gt_payloads(kind, roll, live_frames)

    for fr in frames:
        if fr.get("error") or fr["stem"] not in fit_payloads:
            continue
        fr["review_kind"] = kind
        review = {"fitted": fit_payloads[fr["stem"]],
                  "live": live_payloads.get(fr["stem"],
                                            fit_payloads[fr["stem"]])}
        if fr["stem"] in gt_payloads:        # only frames with an annotation
            review["gt"] = gt_payloads[fr["stem"]]
        fr["review"] = review
    an.write_debug_sessions(frames, roll_meta, review_dir)

    ui = TESTS_DIR.parent / "debug_ui.py"
    print(f"Opening debug UI for {session.name} (close the window to return)")
    print(f"  R: FITTED ({kind} from this session) -> GT (your annotation) -> "
          "live (current preset)   N: vignette on/off")
    try:
        subprocess.run([sys.executable, str(ui), str(review_dir)])
    finally:
        shutil.rmtree(review_dir, ignore_errors=True)
    return session


# ---------------------------------------------------------------------------
# Adapter — the negadoctor surface the shared runner drives
# ---------------------------------------------------------------------------

class NegaAdapter(runner.CalibrationAdapter):
    registry = reg
    schema = an._tuning
    evaluators = EVALUATORS     # same dict object — in-place test patches are seen
    metric_name = METRIC_NAME
    description = "Recorded calibration sessions (spec 05)"

    # Dynamic so tests that rebind the module globals (CALIB_DIR /
    # discover_image_rolls) are still honoured by the shared runner.
    @property
    def calib_dir(self):
        return CALIB_DIR

    def worst_key(self, kind):
        return _worst_key(kind)

    def headline(self, kind, agg):
        return _headline(kind, agg)

    def discover_rolls(self, roll_ids=None):
        return discover_image_rolls(roll_ids)

    def proc_workers(self, n):
        return an._proc_workers(n)

    def map_frames(self, fn, items, workers, on_done=None):
        return an._map_frames(fn, items, workers, on_done=on_done)

    def review(self, session_dir, roll_id=None):
        return review_session(session_dir, roll_id)


ADAPTER = NegaAdapter()


# Back-compat shims so existing callers / self-tests keep their old entry points
# even though the machinery now lives in common.calibration.runner.
optimize = runner.optimize
principal_components = runner.principal_components


def build_spec(kind, fit_params):
    return runner.build_spec(reg, kind, fit_params)


def run_session(config):
    return runner.run_session(ADAPTER, config)


if __name__ == "__main__":
    sys.exit(runner.run_main(ADAPTER))
