### Input specs for features are in `specs`

### Data Flow

1. `auto_retouch/auto_retouch.lua` exports selected images as full-resolution JPEGs to a temp folder (`%TEMP%/darktable_autoretouch_<timestamp>/`)
2. Lua reads each image's XMP sidecar to extract flip/crop transform params, writes `transform_params.txt`
3. Lua calls `auto_retouch/detect_dust.py` via `conda run -n autocrop` with file paths as arguments
4. Python detects bright dust spots using local background subtraction (OpenCV), generates darktable retouch module binary data (brush masks, group mask, retouch params, blendop params), writes `dust_results.txt`
5. Python applies inverse coordinate transforms (undo crop, undo flip, undo ashift when the
 "rotate and perspective" module is active) so mask coords are in darktable's raw image space.
 NOTE: lens correction is not yet inverted (small constant offset, present on all frames).
6. Lua parses results, injects retouch history entry + mask entries into each source image's XMP sidecar
7. Lua calls `image:apply_sidecar(xmp_path)` to reload the modified XMP into darktable

### Testing approach

The `tests` folder holds a saved baseline of 'reasonably good detection' plus checks that run
against it. **Two obligations, not one** — running the suite is necessary but not sufficient:

1. **RUN it after any change to `detect_dust.py`.** `conda run -n autocrop python
   tests/run_quality_tests.py` — confirm no large regression vs the baseline. Also run the
   focused tests: `tests/test_source_invariants.py`, `tests/test_ashift_transform.py`.

2. **EXTEND it whenever you add a feature.** The suite only checks what someone taught it to
   check — a passing run on an un-extended suite is a false sense of safety. This is exactly how
   stroke healing shipped unverified: `run_quality_tests` only diffed *dots* against the baseline,
   so stroke sources/paths/radii were invisible to it. When you add anything, ask "what new data
   or property did I just create, and what here would catch it regressing?" and add that:
   - **New detected data** (new spot kind / field): make sure the baseline diff covers it, OR add
     a check. Regenerate the baseline (`generate_baseline.py`) ONLY after the user approves the
     new detections — the baseline is ground truth, don't bake in unreviewed output.
   - **New correctness property / invariant** (e.g. "healing source must not overlap the defect",
     "a coordinate transform must round-trip"): add it to `check_spot_invariants()` in
     run_quality_tests, or a dedicated `tests/test_*.py`. Invariants are better than baseline diffs
     where possible — they need no baseline and state the actual requirement. Give a non-trivial
     check its own self-test (a "test for the test") so it can't silently degrade into a no-op.

What the suite currently covers: dot match/missing/FP + source/radius mismatch vs baseline;
stroke match/missing/new counts vs baseline; per-spot source-clearance + geometry invariants
(`check_spot_invariants`, baseline-independent); ashift inverse transform (round-trip + bbox).

### Calibration (presets + recorded sessions) — shares auto_negadoctor's engine

The detection thresholds are no longer hardcoded: they live as VALUES in
`presets/*.json` and as a documented SCHEMA in `tuning.py` (126 fittable
constants, grouped by KIND → pipeline sub-stage: `dust`/`stroke`/`sensor`).
`detect_dust.py` loads `DEFAULT_TUNING = tuning.load($RETOUCH_PRESET or "default")`,
mirrors it onto module globals (back-compat) AND threads an immutable `cfg`
(a `tuning.Tuning`) through the detection call tree — every detection function
takes `cfg=None` (default `DEFAULT_TUNING`) and reads `cfg.NAME`, so a calibration
trial passes its own cfg with NO global mutation (thread-safe parallel trials).
`presets/default.json` is **byte-identical** to the old hardcoded constants
(verified value-for-value); format/structural constants (brush/mask byte layout,
the blendop template, mask versions, ML feature-name lists) stay inline.

This is the SAME architecture as auto_negadoctor — the feature-agnostic machinery
is the shared base `common/calibration/` (`schema.TuningSchema`, `registry.Registry`,
`runner.run_main`/optimizers/recorded-session lifecycle). Each feature supplies only
its schema (`tuning.py`), its fittable catalog (`tests/calibration_registry.py`,
generated from the schema with default-relative ranges; bools + the sigma tuple are
not line-searched) and its per-kind EVALUATORS + roll discovery
(`tests/run_calibration.py`).

- **Ground truth** lives EXACTLY as in negadoctor: `tests/fixtures/rolls/<roll_id>/`
  holds the roll's JPGs (gitignored — regenerable) + annotation-session subfolders
  of `{stem}_annotations.json`. The user's annotations ARE the GT:
  `false_positives` (detections marked NOT-dust), `missed_dust` / `missed_strokes`
  (defects to find). `baseline-2026-06/` is the adapted original `baseline_session`
  (empty annotations → no fit signal yet; it establishes the layout).
- **Two kinds.** `dust` objective = `W_FP·(detections reproducing a false_positive)
  + W_MISS·(missed_dust uncovered)`; `stroke` = the analogous over `missed_strokes`
  + stroke-FPs. PRECISION-weighted (`W_FP > W_MISS`, default 3:1; in
  `fit.tolerances`). The per-frame SCORE helpers (`dust_score_per_frame` /
  `stroke_score_per_frame`) are SHARED with the gate in `run_quality_tests.py`
  (reuse `_match_spots`/`_match_strokes`), self-tested in `test_calibration_runner.py`.
- **Cost:** detection is re-run full-res per frame per trial (no cacheable invariant
  like negadoctor's prefix), so keep `fit.params` small, calibrate on the
  signal-carrying frames only (empty-annotation frames are skipped), and/or export
  smaller. `RETOUCH_CALIB_WORKERS` (default 3) caps the per-frame fan-out.
- **Adopt** a result by hand: `cp <session>/fitted_preset.json presets/<name>.json`
  then `RETOUCH_PRESET=<name>` — never edit `detect_dust.py`.
- Run: `python tests/run_calibration.py --config tests/calibrations/configs/dust_default.json
  --method none|coordinate_descent|random_search|cmaes|spsa`; review:
  `--review <session_dir>`. Self-test (fast, image-free): `tests/test_calibration_runner.py`.
- **Dust debug UI** (`DustDebugUI`) gained a `Detect with:` **preset combo** (toolbar):
  selecting a preset re-runs `detect()` on the CURRENT frame under that preset's
  `Tuning` on a background thread and swaps its spots.
- **Boost-region tool (2026-07-02):** the params are tuned for precision (few FPs),
  so they miss some genuine dust. The toolbar `⊕ Boost region` button (or **G**,
  or the Annotate-menu item) arms a modal tool: **drag a rectangle** over dust the
  detector missed and it re-detects THAT RECTANGLE at RAISED sensitivity, appending
  the new spots to `img["detected"]` tagged `boosted=True` + `boost_factor`. The
  **× entry** next to the button is the adjustable sensitivity multiplier (1–10,
  default 2). Sensitivity = a COPY of the effective cfg with the acceptance GATES
  relaxed — dust-dot (`BOOST_RELAX_DIV` divided, `BOOST_RELAX_MUL` caps multiplied)
  AND thread/stroke (`BOOST_STROKE_DIV` / `BOOST_STROKE_MUL`; a boost must reach
  stroke gates too or a missed thread can never surface), with caps/floors
  (`BOOST_CAP` `MAX_SPOTS`≤300 / `STROKE_MAX_FILL_RATIO`<1; `BOOST_FLOOR`
  `STROKE_MIN_ELONGATION`≥1.8 so a thread stays a LINE) — never persisted to the
  frame's params. **The detection THRESHOLD is deliberately NOT relaxed**
  (`NOISE_THRESHOLD_MULTIPLIER` / `MIN_ABSOLUTE_THRESHOLD` stay at base): lowering
  it floods the binary → components merge into noise → a thread stops being an
  elongated component and dots fragment, so ×10 found FEWER than ×3 (non-monotonic).
  Dust a human can see almost always already crosses the threshold and was dropped
  by a precision gate, so gate-only relaxation makes yield rise MONOTONICALLY with ×
  for both dots and strokes (verified: a dusty region 1→2→3→4 dots over ×1→5,
  recovering all 4 user-marked missed dots; a marked edge-thread t0 surfaces at
  higher ×). Genuinely sub-threshold dust stays a hand Ctrl+click / Draw-thread (T).
  - **AUTO-boost — a ONE-CLICK ⚡ Auto toolbar button** (also an Annotate-menu item;
    `_toggle_boost_region(auto=True)` sets `_boost_auto`): the user draws the
    rectangle because they can SEE dust there, so auto-mode ESCALATES through
    `BOOST_AUTO_LEVELS` (2→3→…→32→…→256) and STOPS at the first level that
    yields a new in-region spot — the minimal boost (fewest FPs). No need to type
    anything (a blank / "auto" × field, `_boost_is_auto`, is a secondary trigger).
    ⊕ Boost region (G) and ⚡ Auto are two flavours of the SAME armed tool: clicking
    the same one again disarms, clicking the other switches while staying armed
    (only the active one lights up; Esc force-disarms either). The whole sweep runs
    on the boost worker thread (`_boost_detect_region` builds `levels`; `has_new`
    dedups each level vs the on-screen spots via `_spot_rep_point`); the queue item
    carries the level used so `_poll_boost_detect` tags + reports it ("Auto-boost
    stopped at ×N" / "nothing new up to ×256 — add by hand"). The × field (for ⊕) is
    UNCAPPED now (was clamped to 10; `_boost_factor` only floors at 1).
  - **Gate attribution:** selecting a boosted spot lists WHICH relaxed gates let it
    in, each NAMING the tuning.py constant (so it maps onto the right-panel params
    table) — `_boost_responsible_params` compares the spot's stored features to the
    base (un-boosted) cfg and reports the gates it would FAIL at base (dot:
    MIN_CONTRAST_TEXTURE_RATIO, MIN_SPOT_AREA_FRAC, LARGE_SPOT_MIN_CONTRAST,
    MAX_LOCAL_TEXTURE_*; stroke: STROKE_MIN_ELONGATION/_LENGTH_FRAC, STROKE_MAX_WIDTH_
    FRAC, STROKE_MIN_CRISPNESS, STROKE_MAX_BAND/CONTEXT_TEXTURE, STROKE_MAX_EXCESS_SAT).
    When the deciding relaxation isn't a recorded feature the panel names the likely
    candidates instead (dot: ML_POSTFILTER_THRESHOLD / MIN_BRIGHTNESS_FRAC_* /
    MAX_NEARBY_ACCEPTED; stroke: STROKE_MIN_BRIGHTNESS_FRAC / STROKE_MIN_RIDGE_DROP).
    Shown on ALL selection paths (spot body, stroke node, healing source) via the
    shared `_boost_note_for`. Guarded by `tests/test_boost_region.py`.
  - **Rejected-candidate reason → param:** selecting a rejected candidate now also
    prints the tuning.py constant(s) behind its `reason` (`_reject_params_for` +
    the `_REJECT_PARAMS` / `_REJECT_SHAPE_PARAMS` maps; "shape"/"dark_bg" refined by
    the detail string) — e.g. `reason=ratio` → `MIN_CONTRAST_TEXTURE_RATIO`,
    `reason=texture` → `MAX_LOCAL_TEXTURE_SMALL/LARGE` — so the user knows which
    right-panel param to relax to recover it. A test asserts EVERY reason emitted by
    `detect_dust.log_reject` (+ the counter-only too_small/too_large/isolation) has
    a mapping, so a newly-added reason can't silently miss one.
  - **Region-only detection (`detect_dust.detect_region`) — the key to speed.** A
    full-frame boosted detect was unusable (~25s; raised sensitivity explodes the
    per-candidate reject loop across the whole frame). `detect_region(image_path,
    roi, …)` instead analyses only a CROP = rectangle + context margin (~1s, **~25×
    faster** on the 5726×3783 fixture) while staying FAITHFUL to a full detect:
    `detect`/`detect_dust_spots[_ml]` gained injection params `image` (a pre-loaded
    crop), `frac_min_dim` (the FULL frame's min_dim, so every `*_FRAC` threshold
    resolves as if on the whole image — NOT the crop's smaller min_dim), and
    `global_stats=(noise_std, bright_ref)` measured on the FULL frame (so a locally
    bright/quiet rectangle can't move the detection threshold / brightness floor —
    "the region is not 100% of the image"). The margin ≥ the background-blur /
    context-texture / isolation radii, so every in-rect candidate sees the same
    neighbourhood a full detect would; `detect_region` offsets crop-local coords
    back to full-frame (`_offset_spot_coords`) and keeps only spots whose centre is
    inside `roi`. The non-region path (all injection args None) is byte-identical.
    Note the region detect MAY surface a few EXTRA spots vs full-frame (a tighter
    crop sees fewer isolation neighbours) — desirable here; the user keeps or FPs
    them. Guarded by `tests/test_region_detect.py` (fixture-gated, SLOW: one full
    detect — asserts every full-detect dot inside the rect is reproduced by the
    region detect + in-rect/full-frame coords + determinism).
  - Boosted spots render **teal** (`#20d0ff`, vs green), get a `boosted` table tag +
    legend entry, and a `⚡ BOOSTED …` note in the info panel; they are otherwise
    normal detected spots (healable, Mark-FP, scroll-resize). The tool is modal via
    `handle_press/drag/release_override` gated on `_boost_drag_active` and CLAIMS
    only a PLAIN drag, so **Ctrl+drag still zooms** and plain rubber-band select is
    unaffected (Esc/G exits). Background detect + merge = `_boost_detect_region` /
    `_poll_boost_detect` / `_merge_boost_spots` (dedups against spots already
    present). The `boosted` field flows through the canonical spot dict →
    `debug_spots.json` + `_final_spots_for_apply` for free. UI-logic guarded by
    `tests/test_boost_region.py` (image-free: cfg relaxation + merge).
- **UI chrome (2026-06-24) — unified with auto_negadoctor:** the dust UI is now
  fully menu/toolbar-driven, no lower-left button column (`SHOW_BOTTOM_BUTTONS =
  False`). The edit actions (Rejected→Missed, Mark/Clear FP, Remove missed, Draw
  thread, Clear selection) live on an **`Annotate` menu cascade**
  (`build_feature_menus`); the formerly-stateful `Remove missed` button's
  enable/disable is driven through a tiny `_MenuEntryState` proxy bound to
  `self.remove_missed_btn`, so the ~12 existing `remove_missed_btn.config(state=…)`
  call sites are unchanged. The **FP/Missed count label** moved to the toolbar.
  The marker legend + full mouse/key reference are NOT in the left panel — they're
  **non-modal popups** on the **Help menu** (`_show_legend` / `_show_shortcuts`,
  both now SHARED in `common/debug_ui_base.py`, driven by each UI's
  `_LEGEND_ENTRIES` / `_SHORTCUTS_TEXT` class attrs). The **top toolbar** is the
  shared base skeleton too: the common ◀ ▶ / － ＋ / Fit buttons come from the base
  `build_toolbar`, and the dust UI adds its widgets via `build_feature_toolbar`
  (FP/Missed count label left; the review fitted/live toggle + 'Detect with:'
  preset combo right). The **`Show rejected candidates` toggle moved from a
  left-panel checkbox to a View-menu checkbutton** (its BooleanVar is created in
  `init_selection_state` so the menu can bind it). (The former `Show source brush`
  toggle is gone — the source patch circle is now ALWAYS drawn, uniform for
  detected and hand-added spots.)
  and the **per-frame status (dims / detected / rejected / timings) moved from the
  hidden left status label into the bottom "Selected" box** via `default_info_text`
  (refreshed in `update_counts` so run-mode streaming counts stay live). The View /
  Navigate / Help bar + `P` colour-management toggle come from the base skeleton;
  the dust UI adds `extend_view_menu` (toggles + Review source) + `extend_help_menu`
  + `build_feature_menus` + `build_feature_toolbar`. The left panel keeps only the
  image list.
- **UI-FIRST + streaming (run mode, 2026-06-23):** the darktable **Debug** action no
  longer detects the whole batch and *then* opens the UI (which left the window blank
  for minutes). `detect_dust.py --debug-ui` now **launches the UI immediately** on the
  export dir (JPGs only, no `*_debug_spots.json`) and exits; the UI's `load_session`
  sees no precomputed spots → builds img_dicts from the JPGs (`_run_mode`) and
  `_start_background_detect` **detects every frame on a small pool**
  (`RETOUCH_UI_WORKERS`, default 3 — full-res is memory-heavy), **streaming each
  frame's spots in as it finalizes** (`_poll_bg_detect`) with the **SHARED centered
  canvas progress overlay** (`_show_canvas_message` etc., now in
  `common/debug_ui_base.py` — negadoctor's machinery). The first frame is visible at
  once; spots appear progressively; the overlay shows `detecting X/N`. The
  ad-hoc preset combo can also load a `(fitted — review)` entry from
  `RETOUCH_REVIEW_PRESET` (`RETOUCH_ML_MODEL` carries the model path). The
  **InPlace/apply** flow and **sensor-dust** debug path are unchanged (they keep the
  batch — sensor is multi-frame consensus, can't stream per-frame).
- **Calibration review: FITTED → GT → LIVE cycle (negadoctor-style, 2026-06-24;
  3-way since 2026-06-26):** `run_calibration.py --review <session>` **precomputes
  the fitted (session result) and live (current source-code) detection** for every
  roll frame (`review_session` detects twice via `ADAPTER.map_frames`) PLUS a third
  **GT** payload from the user's annotation (`_gt_review_payload`: the
  user-corrected output = fitted detections minus the ones reproducing an annotated
  `false_positive`, plus the hand-added `missed_dust`/`missed_strokes` as spots via
  `missed_dust_to_spot`/`missed_stroke_to_spot`; the dropped FPs become GT
  `rejected`). It writes a throwaway session dir of `*_debug_spots.json` whose
  frames carry `detected`/`rejected` = fitted plus a `review={"fitted":…, "gt"?:…,
  "live":…}` payload + `review_kind`, and opens the dust UI. The UI detects
  `review_mode` in `init_selection_state` and key **R** CYCLES the active source IN
  PLACE FITTED → GT → live → FITTED (`_apply_review_source` swaps each frame's
  detected/rejected lists; instant, no re-detection; sources a frame lacks — e.g.
  GT on an un-annotated frame — are skipped). A RIGHT-aligned toolbar button
  (`Src: FITTED`/`Src: GT`/`Src: live`) + a View-menu item show the CURRENT source.
  **R is a dispatcher** (`_on_r_key`): review session → source cycle; otherwise the
  normal "rejected → missed" annotation action. Toggling rebuilds the (empty)
  per-frame annotation state because spot indices differ between sources. **The
  `live` source TRACKS the 'Detect with:' dropdown** (default = `(live default)` =
  default.json): selecting a preset redetects under it (`_on_review_preset_changed`
  → `_redetect_current`, which in review mode stores into `review['live']` tagged
  with `live_preset` rather than overwriting the base frame); other frames refresh
  their live LAZILY on navigation (`_ensure_live_for_current`, detection is
  full-res + slow), and `(live default)` restores the precomputed `live_default`.
  **The 'Detect with:' combo is DISABLED unless `live` is the source on screen**
  (`_update_preset_combo_state`, called from `_refresh_review_display` / toolbar
  build / `_install_roll`): GT and fitted are unmovable, so the preset has no
  effect there (mirrors auto_negadoctor; smoke-tested in `review_cycle`). GT here
  is ALREADY unmovable by the preset — selecting one only redetects into
  `review['live']`, never GT.

### Canonical Data Principle (auto_retouch)

The spot dicts produced by `detect_spots()` are the **single source of truth** for all detected data (location, radius, etc.). Any algorithm change — whether to detection logic, brush sizing, coordinate transforms, whatever — must be reflected in the spot dict fields themselves. All four consumers read from the same dict and must stay in sync automatically:

1. **`{stem}_debug_spots.json`** — serializes spot dicts wholesale per image; new fields appear for free
2. **`_dust_overlay.jpg`** — `save_visualization()` draws from spot dict fields
3. **Debug UI** — `debug_ui.py` draws circles and shows info from spot dict fields
4. **XMP output** — `generate_xmp_data_for_spots()` reads spot dict fields

**Consequence:** never put a tunable parameter only inside `generate_xmp_data_for_spots()` — the debug pipeline never calls it, so the change is invisible during testing. Compute everything meaningful at `spots.append()` time in `detect_spots()`.

Current key fields: `cx`, `cy`, `radius_px` (raw detected, for algorithm internals), `brush_radius_px` (scaled effective size, for display and XMP brush).

### Registered Actions

CONSOLIDATED (2026-07-02): the per-mode Debug / InPlace / InPlace_KeepTemp trios
were removed. Each mode now has ONE **unified continuous-edit action** that always
opens the debug UI **BLOCKING**; on close the shared finish dialog (see root
CLAUDE.md / `common/debug_ui_base.py` — two checkboxes: apply annotations back /
delete temp folder) decides what happens, and Lua reads `close_choices.txt` after
the UI exits. Surviving actions: **AutoRetouch** (film), **AutoRetouch_SensorDust**,
**AutoRetouch_Apply_From_Folder**.

Film dust:
- **AutoRetouch** (`export_import_and_edit`) — the unified film-dust action (the
  former Edit_Existing flow, which now subsumes plain Debug/InPlace: on a fresh
  frame the disable-existing-retouch step is a no-op). Full flow below. On close it
  reads `close_choices.txt`; the apply step (disable prior dust + add edited set for
  the CHANGED frames) runs only if "apply" is checked, and the temp/GT folder is
  deleted only if "delete temp" is checked (default keep — it's calibration GT).
  Detailed continuous-edit / import-GT description (still current) — **import
  existing retouch as ground truth** (2026-07-01; the auto_retouch analog
  of negadoctor's continuous edit). Run it on frames that already carry retouch
  (drawn by hand, or applied by us with the temp folder since deleted) to keep
  editing AND to rebuild the temp GT folder for calibration. Flow
  (foreground/blocking, like Apply_From_Folder):
  1. `disable_all_retouch_for_export` temporarily forces `enabled="0"` on EVERY
     retouch history entry of each frame and reloads, so the export is the CLEAN,
     un-healed scan (mirrors negadoctor's `disable_modules_for_clean_export`;
     crop/flip/ashift stay ON so the export geometry matches `transform_params`).
  2. `export_frames` (the export + `transform_params.txt` + `source_paths.txt`
     block factored out of `export_and_detect`) exports the clean JPEGs; the flow
     also writes `source_xmp.txt` (`SENSOR_LABEL|…` / `DUST_LABEL|…` header lines +
     `stem|<original sidecar path>`).
  3. `restore_xmps` puts the user's retouch back immediately (the clean export is
     already on disk; Python then reads the REAL masks from the restored XMP).
  4. Launches `debug_ui.py <dir> --import-gt --apply` (run mode). On launch
     `DustDebugUI.load_session` calls `import_retouch.seed_import_annotations(dir)`,
     which decodes each frame's ACTIVE film-dust retouch out of its source XMP and
     writes seed `{stem}_annotations.json` (the imported dots → `missed_dust`,
     strokes → `missed_strokes`) + `import_baseline.json`. Run-mode detection then
     streams in and `_poll_bg_detect` → `_load_existing_annotations_for` picks up
     the seeds, so the user sees a **fresh detection with their existing shapes
     pre-marked as missed** (one combined editable set).
  5. On close the finish dialog fires; if "apply" is checked the base hook
     `write_apply_results` (was `_write_apply_results`; now gated by the dialog,
     not the raw `apply_mode` flag) writes `dust_results.txt` (final set =
     detected − FP + missed) AND `import_changed.txt` (stems whose committed set
     differs from `import_baseline.json`, via `import_retouch.spots_differ`).
  6. Lua reads `close_choices.txt`; if apply, it applies ONLY the changed frames:
     `apply_retouch_in_place(…, dust_label,
     sensor_label)` disables every existing **non-sensor** retouch instance
     (`disable_retouch_entries`, keeping their history entries) and adds the edited
     set as a new instance; an emptied frame goes through `disable_prior_dust_in_place`
     (disable, add nothing). **Sensor-dust instances are never touched** (excluded
     from both the import decode and the apply-back — identified by their
     `multi_name` label, passed to Python via `source_xmp.txt`). The temp folder is
     **KEPT by default** (it becomes calibration ground truth) — deleted only if the
     user ticks "delete temp" in the finish dialog.
  - **The decoder** (`import_retouch.decode_xmp_masks`) is the inverse of
    `generate_xmp_data_for_spots`: it unions the active form ids of every enabled
    non-sensor retouch instance (clone/heal algos 1/2; consecutive applications
    stack as several instances), reads their brush/circle masks from the latest
    cumulative `mask_num` snapshot, and maps raw-buffer coords → export pixels with
    `detect_dust._original_to_export` (extended 2026-07-01 with a forward-ashift
    branch `_do_ashift`, the exact inverse of `_undo_ashift`). Brush dots/strokes
    + circles are decoded; ellipse/path/gradient masks are skipped with a notice
    (add by hand in the UI). Guarded by `tests/test_import_retouch.py` (encode →
    synthesize XMP → decode round-trip across flip/crop/ashift + sensor exclusion;
    end-to-end `seed_import_annotations`; `spots_differ` self-test).
- **AutoRetouch_Apply_From_Folder** (`apply_retouch_from_folder`) - apply SAVED
  annotations from a user-picked ground-truth folder (NO export/detect). Launches
  `debug_ui.py --choose-dir --apply` foreground: the base pops a native
  `filedialog.askdirectory()` + echoes `CHOSEN_DIR|<path>`; `DustDebugUI.apply_mode`
  loads the folder's `*_debug_spots.json` + `*_annotations.json`, the user reviews,
  and on close `_write_apply_results` writes `dust_results.txt` from the **FINAL
  spot set** per frame = detected − `false_positives` + `missed_dust` +
  `missed_strokes` (via `generate_xmp_data_for_spots`, reading the folder's
  `transform_params.txt`). Lua reads `dust_results.txt` and heals via the EXISTING
  `apply_retouch_in_place` XMP-inject path (same as InPlace), matched by sanitized
  stem. `load_session` resolves each frame's JPG from the session dir / roll root
  (one level up) when the stored temp path is dead.
  **Sessions are now self-contained:** the UI-first **streaming run mode** had
  stopped writing `*_debug_spots.json` (it detected live and never persisted),
  so re-opening a saved session re-detected from scratch AND `_poll_bg_detect`
  wiped each frame's annotations to a fresh state after detecting it — apply-from-
  folder then showed an empty re-analysis. FIXED (2026-06-26): when a frame's
  detection lands, `_poll_bg_detect` re-loads its `{stem}_annotations.json`
  (`_load_existing_annotations_for`, re-matching FP/overrides against the FRESH
  detection by coords/index) instead of wiping it, and on run-mode completion
  `_persist_debug_spots` writes `{stem}_debug_spots.json` for every frame. So a
  session created by Debug now reloads with NO re-detection and intact
  annotations. `_persist_debug_spots` ONLY runs on the run-mode completion path,
  so a folder that already has debug_spots (a committed fixture being reviewed) is
  never re-detected or overwritten. (A session saved BEFORE this fix has no
  debug_spots yet → it re-detects ONCE on first reopen, now keeping annotations
  and writing debug_spots so the next open is instant.)
  **Missed dust is now a first-class healable spot** (the feature that made apply
  possible): a hand-added `missed_dust` ({cx,cy}) is seeded with a heal
  `brush_radius_px` (median of the frame's detected spots, frame-fraction fallback)
  + an auto-recommended healing `src_cx/src_cy` (`find_healing_source` over
  `prepare_source_buffers`); **scroll** resizes it and you **drag its green source
  square** to move the healing source (a press on the square is CLAIMED via
  `handle_press_override`/`_drag`/`_release` → `_missed_source_at` /
  `_missed_src_drag`, so it doesn't fall through to the base's Ctrl+drag
  zoom-to-rectangle; Ctrl+click still works as an alternative). It renders as a
  circle + dashed source line
  (`selected_missed_source`, `_seed_missed_dust`, `_source_buffers_for`,
  `missed_dust_to_spot`/`missed_stroke_to_spot` in `detect_dust.py`). The UI preview
  and the apply writer share `missed_dust_to_spot`, so legacy {cx,cy}-only
  annotations still heal and edited ones honor the user's radius/source. Guarded by
  `tests/test_missed_spots.py`.
  (The former `AutoRetouch_Debug` / `AutoRetouch_InPlace` / `_InPlace_KeepTemp`
  film-dust actions are gone — this unified action + close dialog covers them.)

Sensor dust (select >= 2 frames from one scanning session):
- **AutoRetouch_SensorDust** (`edit_sensor_dust`) — the unified sensor action
  (replaces the old SensorDust_Debug / SensorDust / _KeepTemp trio). Runs the
  cross-frame consensus, then opens the debug UI **BLOCKING** in apply mode via
  `export_and_detect(images, true, true, true)` → `detect_dust.py --debug-ui
  --sensor-dust` → `run_sensor_dust_mode` now `subprocess.run`s `debug_ui.py --apply`
  foreground (was a detached `Popen`). On close Lua reads `close_choices.txt` and,
  if apply, heals the final set into every frame with the `"sensor dust"` label
  (`apply_retouch_in_place`); the temp folder is deleted only if chosen. (Sensor is
  multi-frame consensus, so it keeps the batch-then-launch path — no per-frame
  streaming.)

## Known Bugs / TODOs

For auto_retouch feature:
- [x] Add ability to heal thread(fiber)-like dust — stroke detection (spec 06): elongated
      bright threads detected from the threshold binary, healed with multi-node brush strokes
      (`kind="stroke"` spots). Faint film-scratch ridge pass implemented but disabled by
      default (noise-floor FPs); human adds faint scratches via the debug UI.
- [x] Check all logic for consistency given different input sizes (2026-07-01): EVERY
      size-dependent detection constant is now a fraction of the frame — lengths/radii/
      kernels as `*_FRAC` (resolved `min_dim * FRAC`, kernels odd-ified), areas as
      `*_AREA_FRAC`/`*_FRAC` (resolved `min_dim**2 * FRAC`). No absolute-pixel constant or
      pixel↔fraction reference remains in `detect_dust.py` (`MIN_BRUSH_PX`→`MIN_BRUSH_FRAC`;
      the `200`/`20` context/dedup/sensor literals → named `*_FRAC` module constants; two new
      preset fields `CONTEXT_TEXTURE_RADIUS_FRAC` + `TEXTURE_TIER_AREA_FRAC`). Values in
      `presets/default.json` were converted ONCE offline (anchored at the fixtures' ~3780px
      min_dim) so detection at today's resolution is unchanged. Guarded by
      `tests/test_resolution_invariance.py` (detect at W vs an exact 2x copy → outputs scale
      ~2x; asserts no `*_PX` module constant leaks back in).
- [ ] add params now hardcoded in py to DT UI, pass along with crops?
- [ ] on some heavily dusted images, e.g. DSC_0012, running second debug pass after applying first correction helps detect even more dust not picked up by the first pass.
- [ ] add full negadoctor automation
- [x] generalize debug ui - viewer is common part, other features are like modules to it, add other detectors to UI, not only retouching — done: `common/debug_ui_base.py` is the shared viewer base; `debug_ui.py` here is `DustDebugUI` (also serves sensor sessions via the `mode="sensor"` marker written by `--sensor-dust --debug-ui`); `auto_crop/debug_ui.py` is `CropDebugUI`
- [ ] speed up retouching (GPU?)
- [ ] fine tune sensor dust via debug ui marking like I did with threaded dust
- [ ] add sensor dust to regression
- [ ] ensure debug ui can visualize new regressions vs baseline