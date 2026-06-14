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
   the WHOLE analysis in one pass — no darktable round-trip. **The per-frame
   analytical stages run FRAME-PARALLEL** (`_map_frames`): vignette
   accumulation, stage A (film-base search), stage B1 (crop/percentiles/wb),
   and stage B2 (print tune) each fan out over `_proc_workers(n)` threads
   (default `min(cpu, 8)`, override `NEGA_PROC_WORKERS`); the serial reductions
   between them (global film base, roll-median wb) are cheap. **The per-frame
   work is MEMORY-BANDWIDTH bound** (each frame streams several full-frame
   float64 buffers — the prior-wb render, crop scan, patch searches), so
   throughput plateaus ~8 threads and more workers just thrash the memory bus
   (measured 20-core knee: 8w≈20s, 16w≈24s WORSE, 20w≈19s; CPU sits ~40% because
   cores stall on RAM, not for lack of work). float32 was rejected: only ~1.2x
   and it changed 20/37 frames' params. Real further speedup needs FEWER
   full-frame passes, not more cores. Worker threads are
   NAMED with `nega_model._POOL_PREFIX`, so the per-pixel render parallelism
   (`render_negadoctor`/`linear_to_srgb`, which checks that prefix) runs INLINE
   inside them — coarse frame-level parallelism, no nested thread oversubscription.
   Output is bit-identical to serial (verified: params_hex match; ~3.7x on a
   37-frame roll). Each frame's decoded `(enc_f, lin)` is cached on its dict by
   `_get_loaded` so stages A/B1/B2 (+AI) share ONE TIFF decode instead of
   re-reading it per stage; `process_roll` strips `_loaded` before the dicts are
   serialized. The buffers stay resident from stage A to run end (~tens of MB ×
   frames), so `NEGA_FRAME_CACHE=0` disables it for a smaller peak RSS on
   big/high-res rolls. `_vignette_field_cached` is lock-guarded so workers don't
   all rebuild the same field. The opt-in **AI/vision-LLM stage stays serial**
   (`NEGA_AI_WORKERS`, default 1): it's GPU-bound (one Ollama model) and its
   `scene_cache.json` is a whole-file read-modify-write, so concurrency would
   only queue on the GPU and race the cache.
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
     indefinitely (DSC_0021's museum ceiling) — AND the band is NARROWER than
     CROP_REBATE_MAX_FRAC (real rebate is a thin strip ≤58px; bright SCENE
     content — sky/snow/highlights — is base-like too, in DIFFUSE bands
     hundreds of px deep, and its base-like fraction 0.15-0.34 OVERLAPS legit
     rebate's 0.3-0.55 so the line threshold can't separate them, but the
     width can). NOTE: do NOT use a film aspect-ratio (3:2) constraint —
     future rolls won't all be 36x24 (user). Roll 2510-11-1 (2026-06-13)
     drove three over-trim modes, all fixed without touching old-roll
     containment: (1) bright-scene "rebate" rode 99-288px into content →
     CROP_REBATE_MAX_FRAC width cap (0.04); (2) gently-darker top scene formed
     77-80px "shadow" runs that got full credit → CROP_SHADOW_MAX_FRAC
     0.04→0.030 (0.025 breaks old-roll containment, the floor); (3) bright sky
     with ~4% specular "leak" pixels/line let the gap-tolerant hard run ride
     276px → CROP_JUNK_LINE_FRAC 0.04→0.05. After: new-roll sum|over-trim|
     1197→77px, max 276→18; a ~8-15px rebate sliver remains under-trimmed on 4
     gradient edges (DSC_0014 R/B, 0035 B, 0040 R/T) where true rebate merges
     into adjacent bright scene — within the picker's ≤2% (P_LOW) junk
     robustness, far better than the prior scene-eating over-trims.
     **HARD RULE (user, 2026-06-12): the detected crop must NEVER extend
     outside the user's hand-drawn crop annotations** — gated by
     check_crop_containment() in run_quality_tests against the
     2026-06-12_crop_roll fixtures (15 frames, third annotation round;
     current detector: 0 violations, over-trim medians 0/+12/+1/+1 px per
     edge, worst +37px). The film-base SEARCH is separate and may
     legitimately sample the rejected ring (base lives in gaps/rebate).
     **Annotation fixtures are ROLL-SCOPED BY DIRECTORY** (stems collide across
     rolls — every roll has a DSC_0013): each roll lives in its own
     `tests/fixtures/rolls/<roll_id>/` folder holding its images
     (gitignored TIFFs + `exif_params.txt`), its `scene_labels.json`, and its
     annotation sessions, so `run_quality_tests.discover_rolls()` hands each
     roll only its own fixtures (no `roll.txt` anymore — the roll id is the
     folder name). Reference roll = `2512-2601-1`; second roll = `2510-11-1`
     (crop-correction fixtures, dormant until that roll's TIFFs are repopulated
     into its folder).
   - per frame: picker percentiles over everything inside the crop (P_LOW=2.0
     keeps the dense anchor robust to junk slivers), D_max from them, offset
     FIXED at -0.05 (darktable default; auto-offset degenerates on
     uncropped scans and the user's manual rolls never change it); render
     ONE preview with the prior wb (close to the final rendition, so the
     print-luma bands fall on real shadows/highlights).
   - per frame **wb — region-cast + gentle neutralization** (2026-06-13 GT
     tuning; replaced the old taste-prior blend): the user's wheel picks are a
     per-frame neutralization of the scene's color cast, but consistently
     MILDER than full gray-world (e.g. the tungsten-lit DSC_0002 inverts warm
     and the user only cools it partway — wb_low is the INVERSE-of-cast gain).
     `estimate_region_wb()` reads the cast as the mean NEGATIVE-space color of a
     dark/bright print-luma band (robust gray-world — most frames have NO clean
     neutral grey window, so the old single-window shadow search collapsed to a
     near-constant over-warm value), neutralizes via darktable's exact picker
     formula, then `desaturate_wb()` pulls it toward neutral by WB_LOW_DESAT
     (0.45). **wb_low** uses this region estimate; **wb_high** prefers the
     reliable light-neutral PATCH (`find_neutral_patch`) and falls back to the
     bright-region estimate only when no clean patch exists (tungsten frames).
     No more roll-prior blend (the patch wb_high already lands near the GT
     median; for wb_low the old prior over-suppressed G/B). wb's INTENSITY
     (spread) is a tonal lever handled jointly downstream (see print tune).
     gamma = PRINT_GAMMA (6.1 = GT median; per-frame GT gamma 4.55–7.8 is
     aesthetic intent — fog→soft/low, forest→punch/high — NOT image-derivable,
     needs the `categorize_frame()` scene hook). P_LOW=2.0: the user's crop
     annotation proved the densest 0.5-2% were edge junk, so the dense-side
     anchor uses a junk-robust percentile.
   - per frame: black + exposure via `tune_print_params()`, the user's process
     (2026-06-13): **push brightness to the CLIP BOUNDARY — "as bright as
     possible without the highlights clipping" — which auto-preserves MOOD with
     no scene labels.** Each iteration exposure pins the content P99.9 to
     PRINT_HI_CEIL (0.98, just below clip); when exposure SATURATES at its range
     end and the highlight is still off the ceiling, BLACK takes over to keep
     brightening (a less-negative black lifts the whole curve on dense snow/sky
     once exposure can't go higher — without this, snow/sky came out far too
     dark, P99.5~0.5, exposure maxed while black stayed too negative). A final
     guard backs exposure off until the hard-clip fraction ≤ PRINT_CLIP_BUDGET
     (0.3%). Bright scenes use the headroom → bright; genuinely dark scenes
     (museum) clip early → stay dark (verified: DSC_0021 midtone 0.01 vs snow
     0.6+). Clip-free on all 37 (rendering the GT params also clips ~0%).
     **NO HARD CLIPPING is the binding constraint and the param gate does NOT
     measure it** — an earlier 0.95-highlight-target / shadow-point-anchor tuner
     matched the param gate better but blew out 3-60% of pixels (the user
     reported it unusable on re-export); a fixed low 0.86 ceiling was clip-safe
     but left snow/sky too dark. Clipping is gated separately
     (`check_no_clipping`).
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
- **AutoNegadoctor_AI_Debug** (`export_and_invert_debug(true)`) — like Debug,
  but also computes the spec-03 vision-LLM ALTERNATE variant (`--ai-tune`); the
  debug UI switches Analytical↔AI with key **A**.
- **AutoNegadoctor_AI_InPlace** (`export_invert_and_apply(false, true)`) — like
  InPlace, but the params written to the XMPs are the vision-LLM nudged variant.
  The full analytical pipeline still runs first, unchanged.
- **AutoNegadoctor_Remove** (`remove_negadoctor_selected`) — strip all
  negadoctor entries from the selected frames' history (renumbers entries,
  fixes history_end) for a clean re-run after algorithm changes.

### AI scene tuning (spec 03 — `scene_tuner.py`)

An OPT-IN additive layer (`--ai-tune` flag / the two AI actions above) that
nudges the FINAL analytical params toward the user's per-scene aesthetic — the
"irreducible taste" residual (fog→soft gamma, forest→punch, museum→warm/dark)
that pixel math can't reach. **It never touches cropping / vignette / film-base
detection**; the analytical pipeline runs first and unchanged, and with AI off
the output is byte-for-byte identical.

- `categorize_scene()` POSTs the already-inverted preview (downscaled to ≤512px
  JPEG, base64) to **Ollama** (`/api/generate`, default model `moondream`,
  `format`=JSON schema) via stdlib urllib — NO new dependency. Returns
  `{scene, mood, warmth, contrast, rationale}` over a fixed small label
  vocabulary, or `None` on ANY failure (caller keeps analytical). Model/host/
  context via `NEGA_OLLAMA_MODEL` / `NEGA_OLLAMA_HOST` / `NEGA_OLLAMA_NUM_CTX`.
  Responses cached per model+stem+image-hash in `scene_cache.json` (cache key
  includes the model, so switching models re-queries). **Speed (benchmarked on a
  6GB RTX 3060 laptop)**: the bottleneck is the vision-encoder PREFILL per
  distinct image, NOT generation. `gemma3:4b` prefills the SigLIP tower on CPU
  (~14s/frame, ~30 tok/s); `moondream` does ~2.4s/frame (~6x) with comparably
  varied labels, so it's the default. `llava-phi3` is also ~2.4s but its labels
  were repetitive in benchmarking. `num_ctx=4096` (vs the 128k default) keeps the
  model 100% GPU. NO `confidence` field — small models emit a stock value (not
  real calibration), so it was dropped. The LLM preview is rendered from a
  downscaled frame (`_downscale_for_llm`), not the full 2000px (image is fixed at
  256 vision tokens regardless of input size, so smaller input doesn't help).
- `apply_scene_tuning()` maps the labels via interpretable, hand-auditable
  constants (NOT a learned model), **calibrated against the 2026-06-13 GT roll
  (2026-06-14)**. The big GT gap turned out to be SYSTEMATIC, not per-scene
  taste: analytical exposure ~0.25 too high and black ~0.16 too negative on
  nearly every frame (the clip-boundary tuner vs the user's lower, lifted-black
  print). So the AI variant **re-targets the print tune from inside scene_tuner**
  (analytical path untouched): `AI_HI_CEIL` 0.72 (vs analytical `PRINT_HI_CEIL`
  0.99) halves the exposure delta, `AI_BLACK_LIFT` +0.10 halves the black delta,
  both clip-safe. On top, the LLM labels add only the residual: `SCENE_GAMMA`
  (gamma keyed on the **scene** label — night/indoor higher grade, daylight/snow
  lower; the `contrast` label was degenerate, 36/37 `soft`) and `MOOD_CEIL_BIAS`
  (lowers the ceiling further on dark/moody). `CONTRAST_GAMMA` and `WARMTH_SHIFT`
  are **kept but zeroed** — those labels carried no usable per-frame signal on
  this roll (warmth even ran counter-intuitive to the GT wb cast), so any nonzero
  value regressed the gate; re-enable on a roll where they discriminate. The
  hard-clip guard inside `tune_print_params` (takes `hi_ceil`) plus a re-run
  `_clip_guard` after the black lift keep **the AI variant clip-safe** (worst
  ~0.23%) — gated by `tests/test_scene_tuner.py` and `check_ai_variant`.
- `categorize_frame()` is the wired-up hook (was the `None` stub). The
  `--ai-tune` pass (`run_scene_tuning`) stores `params_ai`/`params_ai_hex`/
  `scene` alongside the untouched analytical `params`; `write_results` applies
  the AI blob in the `OK|` line + a `DETAIL_AI|` line; the debug session JSON
  carries both variants for the key-A A/B switch.
- Calibrate the label→offset constants with
  `tests/calibrate_scene_tuning.py` (offline, not a gate; needs Ollama): groups
  the GT params by the LLM's labels, prints per-category medians (gamma by
  **scene** now), and reports AI-vs-analytical GT-delta. The committed gate uses
  each roll's own `fixtures/rolls/<roll_id>/scene_labels.json` (the LLM's labels
  from the user's `--ai-tune` run, stem-keyed) so `check_ai_variant` reproduces
  offline WITHOUT Ollama — re-capture it from a fresh run's `scene_cache.json`
  when the prompt/model/vocabulary changes.
- **Deferred (Phase 2, after 2 more annotated rolls) — learned taste-residual**:
  the 2026-06-14 label-enrichment experiment (spec 03) established that the
  remaining per-frame gamma/exposure error is IRREDUCIBLE darkroom taste, not a
  labeling gap — even gemma3:12b, which labels the scene ACCURATELY (weather/
  season/scene; moondream:1b hallucinates the same enriched prompt and is the
  bottleneck), cannot predict the spread WITHIN ordinary scenes, and the LLM's
  numeric self-ratings are stock values. So Phase 2 is a small regularized
  model (ridge / GBT, sklearn) predicting the per-frame RESIDUAL
  `GT − analytical` from gemma3:12b CATEGORICAL labels + analytical image
  metrics, with leave-one-ROLL-out validation, deployed as a fast function (a
  third "learned" variant). Full plan + milestones in spec 03; this roll's
  gemma3:12b features are saved as
  `fixtures/rolls/2512-2601-1/scene_labels_gemma3_12b_enriched.json`.
  RL/from-scratch nets
  stay infeasible at ~110 frames.

### Debug UI

`NegadoctorDebugUI` (subclass of `common/debug_ui_base.py`) shows each frame
ALREADY INVERTED, rendered **live** from the source negative + the session's
analytical/AI params (`_invert_pil`/`_load_display_pil`; the base class calls
the `_load_display_pil`/`_load_thumb_pil` hooks instead of opening a file).
There are **no baked preview/mask/overlay images** — `write_debug_sessions`
emits only `{stem}_debug_nega.json`, and the UI renders the inversion and the
analysis-crop mask on the fly, so what you see is always the honest output of
the current algorithm. Markers: local
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
(mask computed live from the detected `border` via
`auto_negadoctor.build_analysis_mask`) → frame with the rejected area blanked
out — for auditing the content-crop detection. The mask view PERSISTS across image navigation
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
corrected patch produces vs applied. **Shadows/highlights color wheels**
(spec 02, right panel below the item-table sash): two `ColorWheel` widgets
set `wb_low` / `wb_high`
**directly** with live preview — drag the marker, the inverted preview
re-renders (WB only; black/exposure unchanged unless a crop is also
present). Each wheel's marker starts at the frame's **auto-found wb** (the
applied `wb_low`/`wb_high`, or a saved override), so the user nudges from the
algorithm's choice; the chosen value is stored as a direct abstract
`wb_overrides` entry (`{"shadows"/"highlights": [r,g,b]}`) — the clean ground
truth for later tuning of the production wb finder (loss = distance between
the algorithm's wb and the user's chosen wb). Precedence in the live render:
`wb_override > patch correction > detected patch`. Selecting a wheel (click
it or its `wb_lo`/`wb_hi` item-table row) lets `C` revert it to the auto
value and the note box attach to it. A fixed **gold pin** marks the
algorithm's auto wb (the draggable black/white marker is the chosen value),
so the divergence is visible; the wheels **grow to fill the footer pane**
(drag the right-panel/sash wider for finer precision — `ColorWheel.resize`
re-renders the disk and re-places marker+pin from the remembered wb). The wb↔wheel mapping is pure math in
`nega_model.py` (`wb_to_wheel` / `wheel_to_wb`, a log-space zero-sum chroma
projection). Corrections auto-save to
`{stem}_annotations.json`; closing writes `debug_report.txt` (patch size
changes + print overrides + wb wheel overrides included) for tuning.

### Testing approach

Same two obligations as auto_retouch: RUN after changes, EXTEND for new
behavior (and self-test non-trivial checkers).

- `tests/test_forward_model.py` — pure math: tuner-formula round-trips
  (film base → black, densest → 0.96 target, white patch exactly
  neutralized), hex encode/decode incl. a real manual XMP blob, and the
  color-wheel ↔ wb mapping (`test_wheel_mapping`: neutral→center,
  wb→wheel→wb round-trip, max(wb_low)=1 / min(wb_high)=1 + range clamp over
  the whole disk).
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
- `tests/fixtures/rolls/<roll_id>/` — ONE folder per annotated roll, holding
  EVERYTHING for that roll together (see `fixtures/rolls/README.md`): its
  linear-Rec2020 float TIFF exports (**NOT committed** — `*.tif` is
  `.gitignore`d, ~1.2GB/roll at 2000px, regenerated from the source raws), its
  tracked `exif_params.txt` and `scene_labels.json`, and its annotation
  session subfolders. The roll id is the folder name — **no `roll.txt`**.
  Detection is resolution-independent (size constants are fractions of the
  frame), so any export width works. Annotation sessions are dated subfolders
  so multiple sessions per stem coexist (2026-06-11_taste: patch/print
  corrections; 2026-06-12_crop_roll: 15 hand-drawn content crops, third
  annotation round — the HARD-TRUTH set for crop containment;
  2026-06-13_wb_print_roll: the full 37-frame roll tuned with the shadows/
  highlights color wheels + print sliders — wheel-picked `wb_overrides`
  (wb_low/wb_high, normalized) and `print_overrides` (black/exposure/gamma),
  the HARD-TRUTH set for the wb/print ground-truth gate; `run_quality_tests`
  also reports user-chosen vs applied wb as the tuning target) or a flat set of
  `*_annotations.json` (roll `2510-11-1`). Crop/patch rects are stored as
  **NORMALIZED fractions** of the frame (resolution-independent); debug_ui.py
  normalizes on save / denormalizes on load (px internally), and
  run_quality_tests denormalizes via `_rect_to_px`. The containment check
  allows a 1px slack (`CROP_CONTAINMENT_ROUND_TOL`) for cross-resolution denorm
  rounding — real over-includes are many px.
- `tests/run_quality_tests.py` — `discover_rolls()` ITERATES over every
  `fixtures/rolls/<roll_id>/` that has local TIFFs (SKIPs with a
  repopulate-from-darktable message when none); per roll: invariants (param
  ranges, wb normalization max(wb_low)=1 / min(wb_high)=1, Dmin orange
  ordering, exposure-ordering consistency Dmin_i/Dmin_j ≈ factor_i/factor_j,
  patches in bounds/outside border, hex round-trip; checker has a self-test)
  + annotated-frames report + **ground-truth HARD gate** (`check_ground_truth`,
  self-tested: production wb_low/wb_high/black/exposure/gamma must match the
  wheel/print annotations within strict tolerances `TOL_GT_*`; per-stem
  field-level last-writer-wins across sessions; prints a per-param aggregate
  median/max-delta dashboard — FAILS by design: the strict per-frame gate is
  irreducible taste, NOT a tuning bug, so this stays RED) + **AI-variant HARD
  gate** (`check_ai_variant`: builds the spec-03 scene_tuner variant from the
  roll's committed `scene_labels.json` and FAILs only if it REGRESSES any
  param's aggregate median vs analytical or clips beyond `CLIP_MAX_FRAC` — a
  net-improvement guard that is currently GREEN; prints the analytical→AI
  dashboard) + baseline diff vs
  `tests/baseline_session/<roll_id>/` (per-roll; params, film-base location,
  shadows/highlights patch positions AND sizes).
- `tests/generate_baseline.py` — regenerate baseline ONLY after the user
  approves a debug-UI review.
- `tests/smoke_debug_ui.py` — builds a 3-frame session in %TEMP% and drives
  the UI programmatically (selection, relocation, notes, view toggles).
- `tests/test_vignette.py` — pure-math regression for the roll-wide vignette
  estimator (`fit_vignette_profile`), fixture-free (embeds two rolls' captured
  radial envelopes, no TIFFs): roll **2512-2601-1** (centre-brightest reference
  roll, 0.525/0.05/0.375 — must stay unchanged) and roll **2511-12-1** (envelope
  PEAKS off-centre, innermost bins slightly dimmer). The old monotone-tail cut
  started at bin 0, mistook that central dip for the corner-leak tail, cut after
  2 bins and rejected the roll ("profile not vignette-like") → no correction →
  inverted corners too bright. Fix (2026-06-13): anchor the cut at the envelope
  peak (`argmin(target)`) and run the monotone/tail cut OUTWARD from it; the
  leading bins are the flat centre plateau, not a corner leak. Roll 2511-12-1
  now fits 0.25/0.125/0.825 (~29% falloff). Plus a synthetic corner-leak case
  proving the OUTER tail is still cut.
- `tests/test_scene_tuner.py` — pure-math + MOCKED-LLM (no network) gate for
  the spec-03 AI layer: `apply_scene_tuning` keeps gamma in the GT range,
  re-normalizes wb, mood ordering (moody ≤ bright midtone), and **the AI
  variant never clips more than `PRINT_CLIP_BUDGET`** across all mood×contrast;
  plus `categorize_scene` returns None when Ollama is unreachable and honours a
  pre-populated cache without a network call.
- `tests/calibrate_scene_tuning.py` — OFFLINE calibration helper (NOT a gate;
  needs local TIFFs + Ollama): groups the GT params by the LLM's labels, prints
  per-category medians to set `scene_tuner` constants, and reports AI-vs-
  analytical GT-delta. Runs the FIRST roll under `fixtures/rolls/` and caches in
  that roll's `scene_cache.json`.
- `tests/test_resolution_invariance.py` — guardrail against reintroducing
  absolute-pixel constants: runs the detectors on a synthetic frame at W and an
  exact 2x copy and asserts outputs scale ~2x (fixture-free, always runs).
  Checks scaling not absolute values, so threshold tuning is safe.

## Known Bugs / TODOs

- [x] TIFF fixtures repo weight — RESOLVED by decommitting:
      `fixtures/rolls/**/*.tif` is `.gitignore`d (was ~284MB at 1000px, ~1.2GB
      at 2000px) and regenerated locally from a darktable export
      (`fixtures/rolls/README.md`). (Per-roll layout since 2026-06-14, replacing
      the single `images_tif/` slot.)
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
      first (`generate_baseline.py`, writes per-roll `baseline_session/<roll_id>/`).
- [~] Converge to the 2026-06-13 wb/print ground truth — IN PROGRESS. Rebuilt
      wb (region-cast + gentle neutralization) + print tune (clip-boundary
      brightness push) + gamma 6.1. NOTE: the param gate (`check_ground_truth`)
      is a PROXY that can diverge from render quality — the clip-boundary tuner
      raised the param-gate count (~166→178) while HALVING the rendered-midtone
      error vs GT (0.088→0.050) and fixing the user's "snow/sky too dark"
      complaint. Optimize the RENDER (brightness + no clip), not the gate number.
      The residual is dominated by IRREDUCIBLE per-frame aesthetic taste, established
      empirically: rendering the GT params shows the user's outputs are NOT
      tonally invariant (median brightness 0.03–0.87 by design — mood is
      preserved), GT exposure⊥black are a -0.95 manifold the user picks a point
      on, and per-frame gamma (fog→4.55 soft, forest→7.8 punch) has ~0 corr with
      any contrast metric. Closing the residual needs SCENE UNDERSTANDING, not
      more pixel math → the `categorize_frame()` LLM hook (day/night, fog,
      foliage, indoor-tungsten) feeding wb-fallback + gamma + brightness intent.
      Remaining derivable headroom is small (wb_low B channel, warm-frame
      wb_high R — both partly taste). The gate stays RED by design; do NOT relax
      `TOL_GT_*`.
      **CLIPPING IS A SEPARATE, HARDER CONSTRAINT THE PARAM GATE DOES NOT
      MEASURE.** A tuner can match the gate's black/exposure values yet blow out
      3-60% of pixels (a shadow-anchor attempt did, on normal scenes — the user
      reported it unusable on re-export). The production tuner pins highlights
      below clip as the binding rule and is verified 0/37 frames hard-clipping;
      ALWAYS re-check rendered hard-clip fraction (out>=0.999) when touching the
      print tune, not just the gate deltas.
- [x] Clip-fraction HARD gate added to run_quality_tests (`check_no_clipping`):
      renders each frame's production params and FAILs if hard-clip fraction
      (any channel >= CLIP_OUT_THR=0.999) exceeds CLIP_MAX_FRAC=1% — so the
      suite can't go green on a clipping tuner. Production tune passes 0/37.
- [ ] Per-frame wb vs roll-wide wb: the user's manual practice is one synced
      wb per roll + per-frame black/soft_clip. Roll-median fallback exists;
      consider a full roll-consensus mode after more debug-UI feedback.
- [x] gamma: PRINT_GAMMA now 6.1 (= 2026-06-13 GT median; supersedes the
      earlier 5.25/6.5 single-frame signals). Per-frame GT gamma 4.55–7.8 is
      aesthetic intent, not derivable — see the convergence TODO above.
      soft_clip stays at 0.75. (The old WB_PRIOR_WEIGHT taste-prior blend was
      removed in the 2026-06-13 wb rebuild; wb is now region-cast based.)
- [x] LLM scene categorization hook (`categorize_frame()`) — IMPLEMENTED
      (spec 03, `scene_tuner.py`, moondream via Ollama). Wired as an opt-in
      ALTERNATE variant (`--ai-tune` / AI_Debug / AI_InPlace actions, debug-UI
      key A), nudging gamma/wb-warmth/brightness per scene. Analytical path
      unchanged.
- [x] Calibrate the scene_tuner label→offset table against the 2026-06-13 GT
      roll (2026-06-14, spec 03 "Calibration pass"). Finding: the LLM labels are
      degenerate (36/37 `soft`, 25/37 `forest`), so the big GT gap is the
      SYSTEMATIC exposure/black bias, not per-scene taste → AI variant re-targets
      the print tune (`AI_HI_CEIL` 0.72, `AI_BLACK_LIFT` +0.10) halving both
      deltas, clip-safe; gamma moved to `SCENE_GAMMA` (scene-keyed); degenerate
      `CONTRAST_GAMMA`/`WARMTH_SHIFT` zeroed. Guarded by `check_ai_variant`
      (net-improvement gate, green) using the roll's `scene_labels.json`. The
      strict per-frame `check_ground_truth` stays RED by design. FOLLOW-UP for
      better per-frame match: a stronger vision model / enriched prompt+vocab.
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
