# auto_negadoctor

### Input specs for features are in `specs`

### Data Flow

1. `auto_negadoctor.lua` pre-flight checks the selection's XMPs: ABORTS if any
   frame has an enabled negadoctor entry (the export would already be
   inverted — run `AutoNegadoctor_Remove` first), WARNS if a tone mapper
   (agx/filmicrgb/sigmoid/basecurve) is enabled or an XMP is missing.
2. Lua exports the selected frames as **32-bit float TIFF in linear Rec2020**
   at `EXPORT_MAX_WIDTH` (currently 2000px) to
   `%TEMP%/darktable_autonegadoctor_<timestamp>/`. The analysis is
   resolution-invariant: every size-dependent constant in `auto_negadoctor.py`
   is a fraction of the frame dimension (a `*_FRAC`, applied as
   `int(round(w * FRAC))`) — no reference resolution. The
   export ICC preferences (`plugins/lighttable/export/icctype` = 4 =
   LIN_REC2020) are temporarily overridden and restored — see "Why TIFF"
   below. Float (not 16-bit int) because the integer formats clamp the
   pipeline at 1.0 and the film base's R channel exceeds 1.0 under the
   current import defaults (found on the first live run).
3. Lua writes `exif_params.txt` (`stem|exposure=…|aperture=…|iso=…` per
   frame, locale-safe floats) for cross-frame exposure compensation.
4. Lua calls `auto_negadoctor.py` via `conda run -n autocrop`. Python does
   the WHOLE analysis in one pass — no darktable round-trip:
   - roll-wide **vignette estimation** (stage 0): lens + backlight/holder
     vignetting darkens corners of the negative (inverted: corners too
     bright, levels skewed). Nothing ON film is lighter than unexposed
     base, so the per-pixel MAX over exposure-normalized frames is the
     vignette envelope; its radial profile (center-bin-normalized, with the
     non-monotone tail cut — corner bright leak breaks the ceiling there)
     is fitted with darktable's own lens-module "manual vignette" model
     (v_strength/v_radius/v_steepness, exact lens.cc tanh-spline transcribed
     in nega_model). All level/color analysis then runs on corrected data
     (`load_frame(path, vignette)`), EXCEPT crop detection, which runs on
     uncorrected data (junk geometry is correction-invariant and the
     hard-truth crop fixtures were drawn uncorrected — corrected detection
     trimmed 1-3px less and broke containment). On the reference roll:
     strength 0.35 / radius 0.075 / steepness 0.60, fit residual 0.004,
     corner falloff ~30% (the user's hand value is stronger — 0.751/0.097/
     0.5 ON TOP of lensfun — they may over-correct visually; informational
     comparison in test_calibration).
   - per frame: trim dark holder borders, find the lightest uniform orange
     window = local film-base candidate (white-clip + R/G-ratio guards
     against backlight holes)
   - roll-wide: physical lightness = mean(rgb)/exposure_factor where
     `factor = shutter_s * iso / aperture²`; the lightest wins as the GLOBAL
     film base; each frame gets `Dmin = winner_rgb * factor_frame/factor_winner`
   - per frame: **content crop rectangle** (`detect_content_crop`) — the
     holder is a RECTANGULAR frame around the film frame, so the analysis
     area is the largest inscribed rectangle containing only film, and
     EVERYTHING inside it counts; NO per-pixel masks (user decision after
     auditing earlier pixel-mask approaches, which first flattened prints
     by letting holder junk dominate the dense percentiles, then ate
     edge-adjacent scene shadows). A scan line is junk when it carries
     holder-dark pixels, bright leak (nothing ON film is lighter than the
     unexposed base — catches light leaking past the holder, e.g. the
     no-dark-pixels 15px top band the user cropped on DSC_0001), or is
     mostly FILM REBATE (base-like in ALL channels — max-channel density <
     CROP_REBATE_MARGIN_D; deep scene shadows stay denser in at least one
     channel), or holder-edge SHADOW (line mean luma < CROP_SHADOW_REL of
     the interior reference — the penumbra ramp is too bright for the
     absolute dark threshold). Shadow is validated per edge: a run
     terminating within CROP_SHADOW_MAX_PX is real penumbra (full credit);
     a band continuing deeper is DENSE SCENE (bright sky is dark in
     negative space) and only the CROP_SHADOW_CORE_PX penumbra core counts.
     Junk must form an EDGE-ANCHORED run (CROP_GAP_TOL): detached interior
     base-like lines are scene. A rebate-extended trim is only valid if the
     base-like band TERMINATES within CROP_REBATE_TERM_WINDOW lines —
     unexposed dark scene at the frame edge is base-like too but continues
     indefinitely (DSC_0021's museum ceiling). NOTE: do NOT use a film
     aspect-ratio (3:2) constraint — future rolls won't all be 36x24 (user).
     **HARD RULE (user, 2026-06-12): the detected crop must NEVER extend
     outside the user's hand-drawn crop annotations** — gated by
     check_crop_containment() in run_quality_tests against the
     2026-06-12_crop_roll fixtures (15 frames, third annotation round;
     current detector: 0 violations, over-trim medians 0/+6/0/0 px per
     edge, worst +19px). The film-base SEARCH is separate and may
     legitimately sample the rejected ring (base lives in gaps/rebate).
   - per frame: picker percentiles over everything inside the crop (P_LOW=2.0
     keeps the dense anchor robust to junk slivers), D_max from them, offset
     FIXED at -0.05 (darktable default; auto-offset degenerates on
     uncropped scans and the user's manual rolls never change it); render
     ONE patch-search preview
     with the **taste-prior wb** (close to the final rendition, so plain
     chroma = "looks gray in the print") and find a dark-gray patch
     (→ wb_low) and light-neutral patch (→ wb_high) using the patches'
     NEGATIVE-space colors. (Earlier approaches failed: chroma on an
     uncorrected render rejects everything; gray-world normalization breaks
     on half-lit frames — the user's gray wall measured as the MOST
     chromatic window.) Chroma has a denominator floor (near-black windows),
     uniformity has a luma floor (grain noise), and EVERY channel must be >=
     MIN_PATCH_DENSITY above base (one near-base channel explodes the wb
     ratios). Frames without usable patches fall back to roll-median wb
     (`categorize_frame()` is the future LLM hook)
   - per frame: the patch-derived wb is blended toward the USER-TASTE roll
     prior (WB_HIGH_PRIOR/WB_LOW_PRIOR = the user's corrected wb on the
     reference frame, WB_PRIOR_WEIGHT=0.5): pure patch neutralization kills
     scene-light character — the user corrected the outdoor frame WARMER
     and the indoor frame COOLER, and blending toward a common prior fixes
     both directions. gamma = PRINT_GAMMA (6.5: user signals 5.25/7.1/6.15
     across sessions, manual rolls 6.9). P_LOW=2.0: the user's crop
     annotation proved the densest 0.5-2% were edge junk, so the dense-side
     anchor uses a junk-robust percentile (and true speculars soft-clip,
     matching the punch taste); on the annotated frame this reproduces the
     user-crop D_max (0.56 vs 0.55) without needing a crop
   - per frame: black + exposure start from darktable's auto formulas, then
     `tune_print_params()` closes the loop on the actual rendered output
     (spec: "normal brightness, maximise speculars without heavy clipping"):
     exposure drives content P99.5 luma to PRINT_TARGET_HI, black pulls an
     out-of-band median into PRINT_MID_BAND
   - emits `negadoctor_results.txt` with a ready 152-char hex params blob
     per frame (`OK|stem|params=<hex>`; `DETAIL|…` lines are for humans)
5. Lua writes per frame: a **lens-module entry** (modversion 10) carrying
   the fitted vignette correction, then a negadoctor entry (modversion 2,
   multi_priority 0) — replace-in-place if a disabled entry exists, insert
   otherwise (history_end protection fix from auto_retouch) — and reloads
   via `image:apply_sidecar()`. The lens params blob is built by Python
   (`encode_lens_params`): the user's lensfun template (Nikon D750 + Micro
   Nikkor 60mm, LENS_TEMPLATE_HEX — re-dump if the rig changes) with v_*
   patched and the lensfun VIGNETTING modify-flag cleared, so the manual
   correction (fitted to our TOTAL estimate) is the sole vignette handler
   while lensfun keeps distortion/TCA (manual vignette applies regardless of
   method/flags per lens.cc). An already-ENABLED lens entry is kept
   untouched (user-managed; the export was then pre-corrected and the
   estimator finds ~0 — self-consistent, no double correction; same on
   re-runs after Remove, which strips negadoctor only). No iop_order_list
   edits (both are base modules).

### Why TIFF / linear Rec2020 export (NOT JPEG)

The orange film base is **out of the sRGB gamut** (manual Dmin
(0.804, 0.335, 0.171) in Rec2020 → sRGB-linear R = 1.125). An sRGB export
clips the R channel over the whole upper range of the frame, destroying both
the film-base color (Dmin ~11% low) and the per-channel density ranges that
wb estimation needs. JPEG/PNG inputs still work as an approximate fallback
(used by the committed tests) — `load_frame()` dispatches on extension.

### negadoctor params blob (modversion 2, 76 bytes LE, plain hex)

`struct.pack("<i18f", film_stock, Dmin[4], wb_high[4], wb_low[4], D_max,
offset, black, gamma, soft_clip, exposure)` — encoder/decoder in
`nega_model.py`. Verified byte-for-byte against the user's manually tuned
XMPs. All the picker/auto-tuner formulas in `nega_model.py` are verbatim
transcriptions from darktable `src/iop/negadoctor.c`; the forward model uses
`log10(pix/Dmin)` (sign matters — it makes apply_auto_black/exposure
self-consistent: lightest area prints at 0.1 pre-gamma, densest at 0.96).

### Registered Actions

- **AutoNegadoctor_Debug** (`export_and_invert_debug`) — mode 1: export +
  analyze detached (hidden bat/vbs launch), opens the debug UI; no apply.
- **AutoNegadoctor_InPlace** (`export_invert_and_apply(false)`) — mode 2:
  full pipeline, params written to XMPs, temp removed on success.
- **AutoNegadoctor_InPlace_KeepTemp** (`export_invert_and_apply(true)`) —
  mode 3: same, temp folder kept for analysis.
- **AutoNegadoctor_Remove** (`remove_negadoctor_selected`) — strip all
  negadoctor entries from the selected frames' history (renumbers entries,
  fixes history_end) for a clean re-run after algorithm changes.

### Debug UI

`NegadoctorDebugUI` (subclass of `common/debug_ui_base.py`) shows each frame
ALREADY INVERTED ({stem}_inverted.jpg rendered by Python). Markers: local
film base (orange), GLOBAL winner (gold, double box; other frames carry a
badge naming the winner), shadows patch (cyan), highlights patch (white),
corrections (dashed green). **Crop correction (key 8)**: drag a rubber-band
rectangle around the TRUE photo content when crop detection got it wrong,
or **grab an individual edge** of the rect (within ~10px, works in ANY mode
without selecting crop first) and drag just it;
scroll grows/shrinks all sides, C clears — the live re-render then
recomputes the picker percentiles, D_max, black/exposure and the print tune
INSIDE the crop (full production chain), the hide-rejected view and
histogram use it as authoritative, and the report/quality suite show
user-crop vs auto border rect as the crop-tuning signal. **M cycles the
analysis-crop view**: normal → red tint on the rejected outside-crop area
(from `{stem}_analysis_mask.png`) → frame with the rejected area blanked
out — for auditing the content-crop detection; the same tint goes onto
`{stem}_nega_overlay.jpg`. The mask view PERSISTS across image navigation
(roll review workflow), and while it is active any rubber-band drag defines
the crop even without selecting "crop" first (an earlier session lost the
user's crop because the drag was silently ignored in this mode). A small **RGB histogram** of
the displayed converted image sits top-right (T toggles); in hide-rejected
mode it is computed over photo content only, so the inverted holder can't
fake a clipped-whites spike. Keys: 1/2/3 select patch kind, Ctrl+Click
re-places the selected patch, **scroll resizes it** (first scroll on a
detected patch seeds a correction from the detected rect, so adjusted sizes
land in the annotations), C clears, V toggles inverted/negative view, G
flags a bad inversion (whole frame). **Print-page params are adjustable
too**: 4/5/6/7 select paper black / paper grade (gamma) / paper gloss
(soft_clip) / print exposure, scroll adjusts the value (Shift = big step)
with live preview, stored as `print_overrides` in the annotations and the
report. Any correction/override **live re-renders the inversion**
(debounced): corrected film base re-derives Dmin/D_max, corrected
shadows/highlights re-derive wb_low/wb_high, print overrides replace their
values (otherwise black/exposure keep their tuned values); **X toggles
between the corrected render and the algorithm's default** for A/B
comparison (badges mark which is shown). The info panel shows what wb the
corrected patch produces vs applied. Corrections auto-save to
`{stem}_annotations.json`; closing writes `debug_report.txt` (patch size
changes + print overrides included) for tuning.

### Testing approach

Same two obligations as auto_retouch: RUN after changes, EXTEND for new
behavior (and self-test non-trivial checkers).

- `tests/test_forward_model.py` — pure math: tuner-formula round-trips
  (film base → black, densest → 0.96 target, white patch exactly
  neutralized), hex encode/decode incl. a real manual XMP blob.
- `tests/test_calibration.py` — runs the pipeline on `tests/images/` (the
  ORIGINAL sRGB JPEG exports) and compares against the user's manual
  inversions in `tests/fixtures/manual_xmps/` (same roll). Gates Dmin ONLY;
  wb and D_max informational (wb: gamut-clipped JPEGs + different upstream
  WB; D_max: our holder-free content-percentile definition diverges from
  darktable's whole-area pick by design, and the print tune compensates).
  NOTE: the JPEGs were exported under the same upstream pipeline state (WB
  etc.) as the manual sessions — the newer TIFF fixtures were exported from
  FRESH imports with different WB defaults, so manual-XMP comparison is only
  valid on the JPEGs. Don't "upgrade" this test to the TIFFs.
- `tests/images_tif/` — linear-Rec2020 float TIFF exports: the
  detection/regression fixture set. **NOT committed** (`*.tif` is
  `.gitignore`d; ~1.2GB at 2000px, float doesn't compress) — kept locally and
  regenerated from a darktable export (see `images_tif/README.md`); only
  `exif_params.txt` is tracked. Detection is resolution-independent (size
  constants are fractions of the frame), so any export width works; the local
  set is currently the 2026-06-13 2000px roll.
- `tests/fixtures/annotations/<session>/` — committed debug-UI annotation
  files from the user's review sessions, organized in dated subfolders so
  multiple sessions per stem coexist (2026-06-11_taste: patch/print
  corrections; 2026-06-12_crop_roll: 15 hand-drawn content crops, third
  annotation round — the HARD-TRUTH set for crop containment). Crop/patch
  rects are stored as **NORMALIZED fractions** of the frame (resolution-
  independent); debug_ui.py normalizes on save / denormalizes on load (px
  internally), and run_quality_tests denormalizes via `_rect_to_px`. The
  containment check allows a 1px slack (`CROP_CONTAINMENT_ROUND_TOL`) for
  cross-resolution denorm rounding — real over-includes are many px.
  run_quality_tests reports detector status, user-rect wb vs applied wb and
  user crop vs detected crop on those frames.
- `tests/run_quality_tests.py` — prefers `images_tif/` (SKIPs with a
  repopulate-from-darktable message when no local TIFFs); invariants (param
  ranges, wb normalization max(wb_low)=1 / min(wb_high)=1, Dmin orange
  ordering, exposure-ordering consistency Dmin_i/Dmin_j ≈ factor_i/factor_j,
  patches in bounds/outside border, hex round-trip; checker has a self-test)
  + annotated-frames report + baseline diff vs `tests/baseline_session/`
  (params, film-base location, shadows/highlights patch positions AND sizes).
- `tests/generate_baseline.py` — regenerate baseline ONLY after the user
  approves a debug-UI review.
- `tests/smoke_debug_ui.py` — builds a 3-frame session in %TEMP% and drives
  the UI programmatically (selection, relocation, notes, view toggles).
- `tests/test_resolution_invariance.py` — guardrail against reintroducing
  absolute-pixel constants: runs the detectors on a synthetic frame at W and an
  exact 2x copy and asserts outputs scale ~2x (fixture-free, always runs).
  Checks scaling not absolute values, so threshold tuning is safe.

## Known Bugs / TODOs

- [x] TIFF fixtures repo weight — RESOLVED by decommitting: `tests/images_tif/
      *.tif` is now `.gitignore`d (was ~284MB at 1000px, ~1.2GB at 2000px) and
      regenerated locally from a darktable export (`images_tif/README.md`).
- [x] Resolution independence (2026-06-13): export width is a free knob (Lua
      `EXPORT_MAX_WIDTH`, set to 2000); every size-dependent detection constant
      is a fraction of the frame dimension (`*_FRAC`, applied as
      `int(round(w * FRAC))`) — no reference resolution (REF_WIDTH/_px were
      dismantled per user); annotation crop/patch rects are stored normalized.
      Guarded by `tests/test_resolution_invariance.py` (2x-scaling + asserts no
      REF_WIDTH/_px leftover). The 2000px re-export tripped one real
      over-include (DSC_0037 right rebate at ~0.14 max-density), fixed by
      CROP_REBATE_MARGIN_D 0.12->0.13 (no over-trim regression on the 16 crop
      fixtures).
- [x] Patch detection rescue (first live-run feedback): wb_high bootstrap +
      gray-world chroma + luma-floored uniformity + per-channel density
      guard; print auto-tune for normal brightness / max speculars.
- [ ] Baseline still not generated — needs a user-approved debug-UI review
      first (`generate_baseline.py`, runs on images_tif now).
- [ ] Per-frame wb vs roll-wide wb: the user's manual practice is one synced
      wb per roll + per-frame black/soft_clip. Roll-median fallback exists;
      consider a full roll-consensus mode after more debug-UI feedback.
- [x] gamma: PRINT_GAMMA=5.25 from the user's explicit "more punchy"
      override (2026-06-11, third session); soft_clip stays at 0.75.
      wb taste prior encoded the "warmer" direction the same session —
      verified: applied wb_high on the reference frame moved from
      (1.34,1.16,1.0) to (1.58,1.23,1.0) vs user target (1.77,1.34,1.0),
      and tuned black/exposure now land within ~0.01 of the user's values.
      Raise WB_PRIOR_WEIGHT if the user still wants warmer.
- [ ] LLM scene categorization hook (`categorize_frame()`) for wb fallback
      on gray-less frames (all-green foliage etc.).
- [ ] Live darktable re-verify after the bpp=32 switch: Debug mode, then
      InPlace KeepTemp, check darkroom result, Remove, re-run; confirm
      history doesn't grow on re-apply.
- [ ] DSC_0002-class indoor frames: shadows patch falls back; the user's
      second-session shadow rect IS valid ((1.0, 0.748→) wb_low works) — the
      roll-median fallback (B-suppressing) disagrees with it noticeably.
      Tuning direction from fixtures/annotations: user prefers stronger R in
      wb_high (DSC_0001: 1.72 vs applied 1.34) and milder B suppression in
      wb_low. Revisit patch scoring/fallback with the next annotation batch.
- [x] Float export confirmed live (8MB .tif, Dmin R 1.0367 unclipped) after
      a darktable restart; guards remain for regressions (Lua reads back
      format.bpp and warns; Python warns on 16-bit TIFF input and records
      source_16bit_tiff in the session).
