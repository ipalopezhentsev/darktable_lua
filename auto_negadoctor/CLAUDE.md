# auto_negadoctor

### Input specs for features are in `specs`

### Data Flow

1. `auto_negadoctor.lua` pre-flight checks the selection's XMPs: ABORTS if any
   frame has an enabled negadoctor entry (the export would already be
   inverted — run `AutoNegadoctor_Remove` first), WARNS if a tone mapper
   (agx/filmicrgb/sigmoid/basecurve) is enabled or an XMP is missing.
   **EXCEPTION — continuous edit (annotate-apply only, 2026-06-22):** the
   `AutoNegadoctor_Annotate_Apply` flow does NOT abort on an already-inverted
   frame. Instead, `export_and_detect` (when `annotate_apply`) calls
   `disable_modules_for_clean_export` to temporarily force `enabled="0"` on every
   active negadoctor/crop/lens history entry of those frames (saving the original
   XMP bytes), reloads via `apply_sidecar`, exports the now-CLEAN negative for
   re-analysis, then `restore_xmps` rewrites the original XMP bytes and reloads —
   always, including the export-failure path. On apply the corrections land as a
   NEW negadoctor (+crop) history entry on top of the prior ones (which stay
   enabled and scrollable), so repeated edits stack as history steps. All three
   ops are disabled together (not just negadoctor) so the film-base search sees
   the rebate, crop detection sees the full frame, and the vignette estimator
   isn't fed already-corrected pixels — i.e. exactly the first-run conditions.
   **Annotations carry over across re-edits (the continuous part):** the debug
   UI saves per-frame annotations into the throwaway temp `session_dir`, so to
   make a re-edit START FROM the last edit (not fresh auto-detection),
   `seed_session_annotations` (Python `main()`, annotate-apply path) writes the
   session's `{stem}_annotations.json` BEFORE the UI launches, with this
   precedence per frame:
   1. **Durable sidecar** next to the source raw —
      `<source_dir>/<stem>.negadoctor_annotations.json` (`DURABLE_ANN_SUFFIX`),
      keyed by the full source path from `source_paths.txt` so same-named stems
      across rolls never collide. Richest source (carries patch rects + notes +
      only the params the user actually changed). After the UI closes,
      `persist_durable_annotations` copies the UI's output back out so the next
      run finds it.
   2. **Reconstruct from the applied XMP** (`_annotation_from_applied`) when no
      sidecar survives — the XMP is the real source of truth and outlives any
      temp/sidecar file (per the user, 2026-06-22: "don't rely on the temp
      folder; take the values from the XMP"). The Lua side writes
      `applied_state.txt` (`stem|negadoctor=<hex>|crop=<hex>`,
      `active_module_params` reads the effective entry) for every re-edited
      frame; Python decodes the negadoctor blob into wb_overrides
      (wb_low/wb_high) + print_overrides (D_max/offset/black/gamma/soft_clip/
      exposure) and the crop blob into a `crop_correction` rect. This pins the
      whole current look (the XMP can't say which values were user-edited).
      Dmin/film-base is NOT reconstructed (an annotation carries a patch rect,
      not a Dmin vector) — it's left to the fresh auto re-derivation, reliably
      ~identical for the roll.
   Rects are normalized [0,1] (export-resolution-independent). Reconstructed/
   loaded annotations are OVERRIDES applied on top of the fresh clean-negative
   auto analysis, so a re-edit = "continue from the current look."
2. Lua exports the selected frames as **32-bit float TIFF** at
   `EXPORT_MAX_WIDTH` (currently 2000px) to
   `%TEMP%/darktable_autonegadoctor_<timestamp>/`. The analysis is
   resolution-invariant: every size-dependent constant in `auto_negadoctor.py`
   is a fraction of the frame dimension (a `*_FRAC`, applied as
   `int(round(w * FRAC))`) — no reference resolution. Float (not 16-bit int)
   because the integer formats clamp the pipeline at 1.0 and the film base's R
   channel exceeds 1.0. **COLOR SPACE (critical — 2026-06-15):** the Lua sets
   the export ICC pref (`plugins/lighttable/export/icctype` = 4 = LIN_REC2020),
   but **that override can SILENTLY FAIL** — darktable then embeds an **sRGB**
   profile (sRGB primaries, D50, gamma TRC). So the export is NOT reliably
   linear Rec2020. negadoctor (`default_colorspace == IOP_CS_RGB`) consumes the
   **working** profile (linear Rec2020 D65) BEFORE `colorout`, so the FIX lives
   in `load_frame`: it reads the TIFF's **embedded ICC** and color-manages every
   pixel into linear Rec2020 D65 = negadoctor's actual input (sRGB → EOTF decode
   + device→XYZ(D50) matrix + `chad` un-adapt to D65 + XYZ→Rec2020). A genuine
   linear-Rec2020 export round-trips to ~identity, so it's correct either way.
   See "Why TIFF" below. (This was THE root cause of the long
   debug-UI-looks-great-but-darktable-washes-out saga: the whole analysis ran on
   sRGB-as-linear data, fabricating a tiny D_max and an over-bright exposure that
   blew out when darktable applied the params to its true-linear working data.)
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
     in nega_model). All level/color analysis — INCLUDING crop detection (since
     2026-06-19) — runs on corrected data (`load_frame(path, vignette)`). Crop was
     historically run on UNcorrected data, but that was a bug on heavily-vignetted
     rolls: the film base is sampled on CORRECTED data, so comparing it against
     UNcorrected edge pixels (which strong vignette — e.g. 55% corner falloff —
     darkens far below the bright central base) made the rebate detectors mistake
     a per-frame edge REBATE for image and leave it uncropped (user-reported:
     foggy roll 2512-2601-1 DSC_0007, right rebate left in; ran fine ALONE only
     because then the frame is its own global-base winner). Vignette correction is
     a per-pixel scalar, so it does NOT move geometry — the hand-drawn crop
     fixtures stay valid; only the value-based junk/rebate classification changes.
     On the reference roll:
     strength 0.35 / radius 0.075 / steepness 0.60, fit residual 0.004,
     corner falloff ~30% (the user's hand value is stronger — 0.751/0.097/
     0.5 ON TOP of lensfun — they may over-correct visually; informational
     comparison in test_calibration).
   - per frame: find the lightest uniform orange window = local film-base
     candidate. The search runs on the **FULL UNCROPPED frame** (NO border/crop
     gate) — the base is the lightest orange patch anywhere, usually the
     unexposed rebate OUTSIDE the content crop; the white-clip (stray backlight)
     + R/G-ratio + min-luma + uniformity guards are what exclude holder/holes.
     Hence `BORDER_*` constants are NOT inversion params: `BORDER_DARK_THR`/
     `BORDER_PAD_FRAC` only feed `detect_dark_border`'s **vignette** mask now,
     and `BORDER_MAX_FRAC` is a **crop** cap (`_crop_decide`).
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
     **WIDE deliberate-rebate path (2026-06-19):** when the user scrolls film in
     the holder to scan extra rebate, the unexposed rebate band can fill ~HALF a
     frame — far past CROP_REBATE_MAX_FRAC (and past BORDER_MAX_FRAC), so the
     conservative path leaves it INSIDE the content (real failure: roll 2511-12-1
     DSC_0002, left ~41% rebate kept). A separate wide tier trims it, gated so it
     NEVER eats ordinary scene: in a negative, lots of smooth daytime scene
     (sky/ground/walls) also inverts to light orange and matches the base by
     hue+density+texture, so per-pixel signatures alone caused ~13 catastrophic
     false-positive over-trims across the 4 rolls (whole daytime frames trimmed to
     a sliver — they pass containment because they over-trim INWARD, which that
     gate doesn't catch). The decisive guard is **band-mean hue**: a real rebate
     band IS the film base so its WHOLE-BAND mean color matches base chromaticity
     to ~0.002, while a scene band that speckles the per-pixel mask still averages
     0.012-0.044 off-base. Five `CROP_REBATE_WIDE_*`/`_HUE_*` constants: per-pixel
     mask = tight hue (HUE_TOL 0.03) + low density (WIDE_MAX_D 0.35, looser than
     base_like so vignette-darkened near-holder rebate still counts) + brighter
     than holder; columns must be SOLID (WIDE_LINE_FRAC 0.5); the contiguous
     edge-anchored run (skip leading holder, do NOT bridge through the dark scene
     that follows) must TERMINATE before WIDE_FRAC (0.55) AND its band mean must
     be within BAND_HUE_TOL (0.008) of base. Result: 0 changes on the two daytime
     rolls, only DSC_0002 fixed (left→956, contained) + two small contained rebate
     slivers on 2506-1. The wide cap is INDEPENDENT of BORDER_MAX_FRAC so ordinary
     edges keep their tight 0.1 conservative trim. `_crop_fields` now also carries
     `hue_dev`, `base_hue` and per-line `col_rgb`/`row_rgb` (the band-mean source).
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
     keeps the dense anchor robust to junk slivers), D_max from them, offset =
     `OFFSET_DEFAULT` (a preset constant, NOT per-frame auto — the auto formula
     degenerates to ~0 on uncropped scans where the lightest area is the film
     base; but a legitimate CALIBRATION target, either sign). NOTE: offset is the
     ONLY route wb_low (shadows) reaches the image
     (`offset_c = wb_high·offset·wb_low`), so sign(offset) = the shadows cast
     direction and |offset| its strength — see the shadows-wheel offset-sign flip
     in the Debug UI section and the `OFFSET_DEFAULT` doc in `tuning.py`. Then
     render ONE preview with the prior wb (close to the final rendition, so the
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
     gamma = PRINT_GAMMA (**5.0 since spec 04**, was 6.1). 6.1 was the GT *param*
     median, but spec 04 tunes the RENDERED PICTURE not the params: rendering the
     GT annotations and minimizing the histogram (luma-EMD) distance to the
     algorithm's render over ALL annotated rolls puts the joint minimum near 5.0
     (the steep 6.1 curve, under the near-clip highlight pin, crushed midtones
     DARKER than the GT picture — median signed luma +0.084 too dark on roll
     2510-11-1). With 4 rolls the aggregate optimum drifted to ~5.5 but
     5.0–5.75 are within run-to-run noise on total EMD; 5.0 is kept because it
     best serves the worst-case dark roll 2510-11-1. The
     param-gate gamma delta gets WORSE by design (we optimize the picture, the
     param proxy is non-unique — the user reaches the same brighter picture via
     low exposure + strongly POSITIVE black + high gamma). Per-frame GT gamma
     4.55–7.8 is irreducible aesthetic intent — fog→soft/low, forest→punch/high —
     NOT image-derivable, needs the `categorize_frame()` scene hook. Re-derive
     PRINT_GAMMA with `tests/calibrate_histogram_match.py` if the rolls/export
     change. P_LOW=2.0: the user's crop
     annotation proved the densest 0.5-2% were edge junk, so the dense-side
     anchor uses a junk-robust percentile.
   - per frame: black + exposure via `tune_print_params()`, the user's process
     (2026-06-13): **push brightness to the CLIP BOUNDARY — "as bright as
     possible without the highlights clipping" — which auto-preserves MOOD with
     no scene labels.** Each iteration exposure pins the content P99.9 to
     PRINT_HI_CEIL (**0.97 since spec-04 pass 2** — the high-res histogram metric
     showed 0.99 over-pushed the top toward white vs the GT; see the convergence
     TODO); when exposure SATURATES at its range
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

### Why TIFF + float (NOT 8-bit JPEG); and why color management is mandatory

The orange film base is **out of the sRGB gamut** (manual Dmin
(0.804, 0.335, 0.171) in Rec2020 → sRGB-linear R = 1.125). An **8-bit** sRGB
export clips the R channel over the whole upper range of the frame, destroying
the film-base color and the per-channel density ranges wb needs. The **float
TIFF** keeps those values unclamped (R > 1), which is why TIFF/float is required
regardless of profile. JPEG/PNG inputs still work as an approximate fallback
(committed tests) — `load_frame()` dispatches on extension.

**The TIFF is NOT guaranteed to be in linear Rec2020** — the `icctype` override
can fail, leaving an sRGB-encoded float TIFF (sRGB primaries, D50, gamma TRC,
values still > 1 because it's float). Treating that as linear Rec2020 is wrong
on all three axes and was the root-cause bug. `load_frame` therefore **always
color-manages the TIFF to linear Rec2020 D65 via its embedded ICC** (see Data
Flow §2 and `lin_rec2020_from_export`). The embedded-ICC reader is dependency-free
(`_read_icc_from_tiff` parses TIFF tag 34675). NOTE: the committed roll fixture
TIFFs are these sRGB exports, so the analysis now decodes them — old
baselines/GT were captured on the un-color-managed (sRGB-as-linear) data and
must be regenerated.

### D_max is a preset constant (not per-frame auto-derived)

`DMAX_DEFAULT` (darktable negadoctor `init()` default 2.046; this roll's
calibrated value 1.7218) — not auto-derived per frame. We previously auto-derived
it via `apply_auto_Dmax`'s formula but fed it the P_LOW percentile over the whole
content crop (vs darktable's picker, which takes the MIN of the small densest
region you drag over), producing a fabricated ~0.7 that skewed everything
downstream (all of offset/wb/black/exposure divide by D_max). It's a preset
constant now, a legitimate calibration target (like `OFFSET_DEFAULT`), AND
hand-adjustable/annotatable per frame in the debug UI (key 0, in `PRINT_PARAMS`).
`compute_dmax` is retained for the forward-model tests only.

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
- **AutoNegadoctor_Annotate_Apply** (`export_annotate_and_apply`) — the
  annotate-and-apply flow: export + analyze (like InPlace), open the debug UI
  **BLOCKING** (foreground, `--annotate-apply` → Python `subprocess.run`s the UI
  and waits), and when the user CLOSES the UI, write their corrections (auto
  where none) into the XMPs. The debug UI in `apply_mode` writes
  `applied_results.txt` on close (`OK|stem|params=<hex>|crop=L,T,R,B|flag=ok`,
  params = `_corrected_params` over the auto analysis, crop = the user's crop
  annotation else the auto content border, as normalized [0,1] edge positions).
  Lua then applies per frame: vignette (`apply_lens_in_place`), crop
  (`apply_crop_new_item`, crop module modversion 3, same 4-float+8-zero encoding
  as auto_crop — positions are post-orientation so orientation is preserved, the
  flip entry is never touched) and negadoctor (`apply_negadoctor_new_item` —
  ALWAYS a NEW history item, per the user's request, not an in-place replace).
  **Continuous edit:** this flow may be re-run on already-inverted frames (no
  abort — see Data Flow §1); the re-analysis exports the clean negative and the
  new corrections stack as fresh negadoctor/crop history entries on top.
  Both crop and negadoctor insert via the shared `insert_history_entry`
  (history_end-protected). The temp folder is KEPT (it holds the annotations).
  Guarded by `tests/smoke_debug_ui.py` (sets `apply_mode`, verifies
  `applied_results.txt` shape on close).
- **AutoNegadoctor_Remove** (`remove_negadoctor_selected`) — strip ALL the
  modules the apply flow writes — **negadoctor + crop + lens/vignette** — from
  the selected frames' history (renumbers entries, fixes history_end) for a
  clean re-run after algorithm changes. Generalized over `remove_modules_in_place
  (image, opset)` with `REMOVE_OPS = {negadoctor, crop, lens}`; reports per-op
  counts. NOTE: this also removes a manually-set crop or lens distortion/TCA
  correction (no reliable signal distinguishes ours from the user's), so it's
  destructive of those — re-apply afterwards if needed.

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
the current algorithm. **DISPLAY COLOR (critical — 2026-06-21):** `render_negadoctor`
emits in the WORKING profile (linear Rec2020), exactly like darktable's
negadoctor. The display path MUST then run darktable's colorout —
`nega_model.working_to_srgb` = Rec2020→sRGB primaries matrix (`REC2020_TO_SRGB`)
+ display-gamut clip + sRGB OETF — used by `render_negadoctor_srgb8`, the
negative-view, and the LLM preview (`render_preview_srgb`). Feeding Rec2020
straight into `linear_to_srgb` (the old bug) desaturates saturated oranges/reds
toward YELLOW — the "leaves look yellow in the debug UI but orange in darktable"
mismatch (user-reported; was R −8/B +8 vs a real darktable sRGB export, now
<0.3/255 per channel). This is SEPARATE from the earlier sRGB-as-linear INPUT
saga (Data Flow §2). The histogram/clip-meter read the displayed pixels so they
follow automatically; `tune_print_params`/`_high_pct`/`_clip_fraction` and the
spec-04 histogram gate stay in working space ON PURPOSE (negadoctor clips at 1.0
there; the gate is a relative algo-vs-GT match, same transform both sides).
**MONITOR COLOR MANAGEMENT (also 2026-06-21):** with values now matching the
sRGB export to <0.3/255, a residual "debug UI looks MORE saturated than
darktable" remained — it was the DISPLAY, not the pixels. darktable color-manages
its darkroom view to the monitor ICC profile; Tkinter blits raw sRGB bytes, which
OVER-saturate on a wide-gamut panel (the user's monitor is Display P3). The base
UI (`common/debug_ui_base.py`) now transforms the displayed bitmap sRGB→monitor
profile at blit time (`_color_manage` in `_redraw` + thumbnails;
`detect_display_icc` via GetICMProfile on Windows, `build_srgb_to_display_transform`
= PIL ImageCms relative-colorimetric+BPC). Applied to the DISPLAYED bitmap ONLY —
analysis/histogram stay on true sRGB (= the export). Override/disable via
`NEGA_DISPLAY_ICC=<path>|off`; toggle in-app with **P** (View menu). This
composes correctly because the user's ground truth IS the sRGB export
(working→sRGB→monitor = a color-managed viewer showing that export). Benefits the
crop/dust debug UIs too (same wide-gamut over-saturation).
Markers: local
film base (orange), GLOBAL winner (gold, double box; other frames carry a
badge naming the winner), highlights patch (white),
corrections (dashed green). (There is NO shadows patch — wb_low is region-
based; the shadows color wheel is its manual control.) **Crop correction (key 8)**: drag a rubber-band
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
fake a clipped-whites spike. **Clipping indication** (added 2026-06-15): the
displayed 8-bit sRGB render is checked per pixel for blown highlights (any
channel == 255 → linear >= 1.0) and crushed shadows (any channel == 0); a
VU-style **clip meter** sits top-right under the histogram (always on: an H bar
for highlights with a gold tick at the clip budget `PRINT_CLIP_BUDGET`=0.3% and
an S bar for shadows, each full at `CLIP_METER_FULL_PCT`=2%, with the exact %),
red/blue **spikes** on the histogram edges show the same fractions, and **L**
toggles an **on-image overlay** (default OFF) tinting clipped pixels red
(highlights) / blue (shadows). Clip fractions use the SAME pixel set as the
histogram (content-only in hide-rejected mode); the overlay masks are captured
from the true render BEFORE mask-blanking so a blanked holder can't read as
shadow clipping. Driven by `_clip_stats` (in `_refresh_histogram`),
`_draw_clip_meter`, the clip branch of `_decorate`; smoke-tested
(`clipping` step). Keys: 1/3 select patch kind (film base / highlights — no
shadows patch), **drag a patch rect to MOVE it** (`_patch_at` +
handle_press/drag/release; non-Ctrl press inside a patch — film base / shadows /
highlights — seeds a correction at the moved spot, a press with no movement just
selects; smoke-tested `drag_patch`), Ctrl+Click
re-places the selected patch, **scroll resizes it** (first scroll on a
detected patch seeds a correction from the detected rect, so adjusted sizes
land in the annotations), C clears, V toggles inverted/negative view, G
flags a bad inversion (whole frame), **N toggles the vignette correction
on/off** (before/after; reloads the negative via the `_neg_cache_key`), **R
flips the calibration-review source fitted↔live** (only in a `--review`
session). **Print-page params are adjustable
too**: 4/5/6/7 select paper black / paper grade (gamma) / paper gloss
(soft_clip) / print exposure, **9 selects scan exposure bias (offset)** and **0
selects D max (dynamic range)** — offset and D_max are darktable's FILM-properties
params, surfaced in the same adjustable/annotatable `PRINT_PARAMS` table because
offset is the sole lever for the shadows cast (see the shadows-wheel offset note)
and D_max divides every downstream picker. Scroll adjusts the value (Shift = big
step) with live preview, stored as `print_overrides` in the annotations and the
report; all these overrides flow into `_corrected_params`/`applied_results` and
into the calibration ground truth (`gt_params_for_frame` / `_load_ground_truth` /
`check_ground_truth` now carry offset + D_max, `TOL_GT_OFFSET` / `TOL_GT_DMAX`).
All params display in **darktable units** (`PRINT_DISPLAY`): offset/black/gloss as
signed/unsigned percent, D_max + gamma plain, print exposure in EV (offset =
signed %, e.g. +0.62%; D_max = plain density, e.g. 1.72). **Brighter/darker combo (keys ] / [, or the buttons)** automate the
user's repeated manual move — raise paper black to lift midtones (which clips
the highlights), then drop print exposure until the highlights stop clipping.
`_brighten()` nudges black by `BRIGHTEN_BLACK_STEP` (0.05 on the [-0.5, 0.5]
black param, a "5%" move), then re-solves exposure to **hold the highlight level
constant** — `_exposure_for_high_pct` bisects exposure (the P99.9 render
percentile is monotonic in exposure) to restore the P99.9 it measured BEFORE the
black change. Pinning that level — instead of "lower exposure until clip is
gone" — makes the result a deterministic function of black, so brighter-then-
darker returns BOTH params exactly (the highlight level is the op's invariant;
this fixed the original "exposure stays new" non-reversibility), and dark scenes
aren't over-brightened to the clip ceiling. Both land as black + exposure
`print_overrides`, so X compares with default and C clears. Smoke step `brighten`
asserts the highlight hold AND the round-trip reversibility. **Copy/paste params
(Ctrl+C / Ctrl+V, or Adjust menu)** — `_copy_params` captures the current frame's
EFFECTIVE tunable look (the 6 `PRINT_PARAMS` + the two wb wheels) into a
session clipboard `params_clipboard`, taking the ANNOTATED (corrected) value
where the source frame has one, else the auto value; `_paste_params` applies them
onto another frame as `print_overrides`/`wb_overrides` (so they flow through the
same live re-render + `applied_results`). The film base/Dmin (exposure-compensated
per frame — that's the global film-base override) and crop (geometry) are NOT
copied. The Ctrl combos beat the plain `<c>`/`<v>` bindings in Tk's
most-specific-wins dispatch, so Ctrl+C never also clears a correction. Smoke step
`copy_paste_params` asserts the annotated-not-auto copy and the cross-frame paste.
Any correction/override **live re-renders the inversion**
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
projection). **SHADOWS-WHEEL OFFSET-SIGN FLIP (2026-06-21):** `wb_low` reaches
the image ONLY through `offset_c = wb_high·offset·wb_low`, so the SIGN of
`offset` is the sign of the shadows wheel's image cast (and `|offset|` its
strength). With a POSITIVE `OFFSET_DEFAULT` (the roll's calibrated value can be
either sign — darktable allows it; this roll's is +0.006) the wheel inverted —
dragging toward magenta greened the shadows. Fix: `ColorWheel.invert` rotates the
wheel angle 180° vs the wb vector, set per frame in `_sync_wheels` from
`sign(offset)` (only the `low`/shadows wheel; `wb_high` also drives the
main gain term so highlights never flip). The STORED `wb_low` is always the true
negadoctor vector — only the UI interaction direction flips, so annotations/GT
are unchanged. NOTE: near offset≈0 the shadows wheel is correct-direction but
weak (wb_low authority ∝ |offset|). **`offset` is EDITABLE in the UI (key 9 /
slider) to EITHER sign**, so `_sync_wheels` reads the EFFECTIVE offset
(`_effective_print_value("offset")` — a `print_overrides` value wins over the
auto `params.offset`), and editing offset across zero re-syncs the wheel live
(`_adjust_print_param` + `_apply_print_value` call `_sync_wheels` for `offset`).
Originally the flag was read from `params.offset` only and never re-synced on
edit, so a user-set NEGATIVE offset left the wheel stuck inverted (user-reported,
2026-06-22). Guarded by `smoke_debug_ui` step `offset_flips_shadows_wheel`. Corrections auto-save to
`{stem}_annotations.json`; closing writes `debug_report.txt` (patch size
changes + print overrides + wb wheel overrides included) for tuning.

**UI chrome (2026-06-20):** actions live on a **window menu bar** (View /
Select / Adjust / Navigate / Help, with the key shown as each item's
accelerator) plus a **top toolbar** of view/navigation toggles; the lower-left
panel no longer carries the big always-on key list (it's in Help → Shortcuts…).
The transient on-image text badges (NEGATIVE/DEFAULT/RE-RENDERED, global-base,
BAD INVERSION, analysis-crop hints) moved OFF the image into a **status strip**
under the toolbar (`_update_canvas_status`); the gold GLOBAL box + bad-inversion
red border stay on the image. The per-frame detail (exif/factor, Dmax/blk/exp/g,
scene, vignette, timings) moved from the cramped left status label into the
bottom "Selected" panel (shown via `default_info_text` when nothing is selected);
the left label is now a 3-4 line glance. The base (`common/debug_ui_base.py`)
gained optional `build_menus`/`build_toolbar` hooks (no-op for crop/dust) and an
`ITEM_PANEL_WIDTH` knob (negadoctor 360, for bigger color wheels).
**Global film-base OVERRIDE (Adjust → "Set global film base from this
frame"):** the auto winner (`choose_global_base`) can be overridden — the
CURRENT frame's effective film base becomes the roll-wide base and every frame's
Dmin is transferred from it via `dmin_for_frame` (exposure-factor ratio, the
same math the pipeline uses). Stored as a snapshot `{winner_rgb, winner_factor}`
in the SOURCE frame's annotation (`global_base`; one source at a time);
`_corrected_params` applies the transferred Dmin to every frame (a per-frame
film-base rect correction still wins locally) and re-derives wb, so it flows
into the live render AND `applied_results.txt`. Clear via Adjust → "Clear global
film-base override". Smoke-tested (`global_base_override` step).

### Testing approach

Same two obligations as auto_retouch: RUN after changes, EXTEND for new
behavior (and self-test non-trivial checkers).

- `tests/test_forward_model.py` — pure math: tuner-formula round-trips
  (film base → black, densest → 0.96 target, white patch exactly
  neutralized), hex encode/decode incl. a real manual XMP blob, and the
  color-wheel ↔ wb mapping (`test_wheel_mapping`: neutral→center,
  wb→wheel→wb round-trip, max(wb_low)=1 / min(wb_high)=1 + range clamp over
  the whole disk), and the **histogram-distance loss** (`test_histogram_distance`,
  spec 04: identity→0, a uniform brightness lift→luma EMD ≈ Δ/256 with the color
  term ~0, a pure chroma cast→color term >0 with luma_signed tiny, EMD monotone
  in shift, clip-mass detection).
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
  dashboard) + **histogram-match gate** (`check_histogram_match`, spec 04,
  self-tested: renders the production params and the GT-params (annotation
  `corrected` over production Dmin/D_max/offset/soft_clip) over the content crop
  **as FLOAT at 14-bit bins with a denser sample** (HIST_BINS=16384,
  HIST_SUBSAMPLE_FRAC=0.001) and reports median EMD (luma/color) PLUS the
  top-highlight percentile gap (|ΔP99.9|, |ΔP99.99|, hl-color); informational
  dashboard + a REGRESSION guard vs the committed per-roll
  `histogram_baseline.json` — FAILs if median total EMD grows beyond
  `HIST_REGRESS_EPS` OR median highlight |hi999| beyond `HIST_HL_REGRESS_EPS`.
  Inert until the baseline exists (regenerate when bins/subsample change — the
  baseline records both). This is the param-INVARIANT picture-match signal that
  drove the spec-04 PRINT_GAMMA + PRINT_HI_CEIL retunes; the strict
  per-frame match stays the red-by-design `check_ground_truth`. **COLOR SPACE OF
  THE LOSS (2026-06-21):** `_render_crop_rows` renders through
  `nega_model.working_to_srgb` (linear Rec2020 working profile → display sRGB),
  so the picture-match EMD is measured in the SAME proper-sRGB space the user
  evaluated and darktable EXPORTS — NOT the old bare-OETF-on-Rec2020 space (the
  display-bug colorspace), which skewed the luma/chroma terms toward yellow and
  tuned the 'look' against colors no one ever saw. sRGB (device-independent) is
  deliberate — the monitor ICC profile (the debug-UI display CM) is a per-display
  detail kept OUT of the loss so calibration stays reproducible. The GT itself is
  stored as PARAMS (colorspace-free), so what the user saw is captured regardless;
  this fix only aligns the algorithm-tuning loss with it. PRINT_GAMMA/
  PRINT_HI_CEIL were derived under the OLD metric — re-derive via
  `calibrate_histogram_match.py` before arming a `histogram_baseline.json`) +
  baseline diff vs
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
- `tests/calibrate_histogram_match.py` — OFFLINE spec-04 instrument (NOT a gate;
  needs local TIFFs, no Ollama). Per GT frame over EVERY roll: renders production
  vs GT params over the content crop and prints the histogram EMD (total/luma/
  signed-luma/color + near-white mass), aggregate medians, a brightness-vs-color
  gap verdict, and a `PRINT_HI_CEIL` sweep. `--write-baseline` writes each roll's
  committed `histogram_baseline.json` (the `check_histogram_match` reference;
  total/luma/color medians + bins + subsample_frac + print_gamma). Use it to
  re-derive `PRINT_GAMMA` (a gamma sweep over all rolls put the joint luma/total-
  EMD minimum near 5.0; 5.0–5.75 are within run-to-run noise as rolls accrue).
  Shared render/GT-reconstruction
  helpers (`gt_params_for_frame`, `_render_crop_rows`) live in
  `run_quality_tests.py` so the instrument and the gate use ONE metric.
- `tests/test_resolution_invariance.py` — guardrail against reintroducing
  absolute-pixel constants: runs the detectors on a synthetic frame at W and an
  exact 2x copy and asserts outputs scale ~2x (fixture-free, always runs).
  Checks scaling not absolute values, so threshold tuning is safe.
- `tests/test_calibration_runner.py` — self-tests for the spec-05 calibration
  runner: the optimizers find a synthetic minimum, the crop over-trim metric is
  correct (delegates to `run_quality_tests.selftest_crop_overtrim`), the
  registry's apply/restore leaves the module globals untouched, `build_spec`
  rejects unknown params, the vignette objective is dominated by a rejected
  roll, and an end-to-end `vignette` session (method `none`, an INLINE envelope +
  a fake evaluator, NO TIFFs) writes the full session schema + records wall
  time + restores globals.

### Calibration sessions (spec 05) — `tests/run_calibration.py`

Reworks tuning from lost stdout into **recorded, reproducible sessions** under
`tests/calibrations/<YYYY-MM-DD_HHMMSS>_<kind>_<NN>/` (`config.json` inputs +
`results.json`/`report.md` outputs + record-only `fitted_params.json`; per-kind
`INDEX_<kind>.md` leaderboards; wall time recorded). The hand annotations in
`fixtures/rolls/` are the unchanged ground truth. See
`tests/calibrations/README.md`. RECORD-ONLY: the runner never edits
`auto_negadoctor.py`; the user adopts good fitted values by hand.

**Tuning order: crop + vignette FIRST, then inversion** — inversion depends on
both (its pickers/wb/print tune run INSIDE the content crop and on
vignette-corrected data), so tune it only once crop and vignette are settled, and
re-run it whenever either is re-tuned.

Three independent KINDS, each with its OWN algorithm-INDEPENDENT metric and its
own history index (never mixed):
- **crop** — objective = total **over-trim** (fraction of frame AREA of content
  the detected crop removed inside the user's hand-drawn rect; per-edge as a
  fraction of the edge — never pixels), with **containment** (never extend
  outside the user rect) a hard `+BIG_PENALTY` constraint. Fits `CROP_*_FRAC`.
  Fast path: prep the per-frame vignette-CORRECTED buffer + global-base `dmin` once;
  `detect_content_crop` is split into `_crop_fields` (heavy trial-INVARIANT pixel
  math — full-res log10 density extremes + line means, precomputed once per frame)
  + `_crop_decide` (cheap; reads the fitted CROP_* / HOLDER_LUMA_THR /
  BORDER_MAX_FRAC at call time), so each trial re-runs only `_crop_decide`
  (~16× cheaper/eval, bit-identical — guarded by
  `test_calibration_runner.test_crop_fields_split_identical`).
- **inversion** — objective = median **histogram EMD** (`nega_model.
  histogram_distance`) between the algorithm's render and the user's GT-param
  render over the content crop (picture-vs-picture; the GT render is FIXED per
  trial), with rendered hard-clip > `clip_max_frac` a hard constraint. Fits ANY
  constant that shapes the picture (film base/Dmin, P_LOW/P_HIGH pickers, patch
  search, **white balance** — desats/bands/priors, print tune). Dispatch: a
  print-tune-only fit (`PRINT_TUNE_PARAMS`) re-tunes on cached frames (fast,
  spec-04 path); any wb/picker/base/patch param re-runs the WHOLE pipeline per
  trial (`_make_inversion_full`) — correct, slower.
- **vignette** — objective = `(#rolls rejected)·BIG + median residual`. The
  radial envelope is derived from the roll's TIFFs by `vignette_envelope()` (the
  accumulation half split out of `estimate_vignette`) — NO committed fixture. A
  profile-fit-only fit (`VIG_FIT_PARAMS`: `VIG_MIN_STRENGTH` + the promoted
  tail-cut constants `VIG_PEAK_CENTER_FRAC` / `VIG_TAIL_CUT_REL`, was hardcoded
  0.15 / 0.97) captures each roll's envelope ONCE, then re-runs only
  `fit_vignette_profile` per trial; fitting an accumulation constant
  (`VIG_BINS`/`VIG_DOWNSAMPLE_FRAC`/…) re-folds the envelope per trial but the
  TIFF DECODE is trial-invariant, so `vignette_envelope` was split into
  `_vig_decode_frame` (heavy, invariant: luma/clip/border) + `_vig_frame_envelope`
  (cheap, reads the VIG_* constants) and `vignette_frame_cache` decodes every
  frame ONCE into RAM; the runner's full path re-folds from that cache per trial
  (~40× cheaper/eval than re-decoding, bit-identical — guarded by
  `test_vignette.test_frame_cache_envelope_identical`). `estimate_vignette` /
  `vignette_envelope` take an optional `frame_cache=`. The "a new roll fails to
  auto-find vignette" guard: add the roll and
  re-run; the optimizer searches the `VIG_*` to keep ALL rolls non-rejected, or
  the report names which still fails. (residual = the vignette-model fit error;
  see below.)

Fitting methods (all native, recorded in `config.json` with their
hyperparameters): `none` (evaluate once), `grid` (per-param `grid_step`),
`coordinate_descent` (`epsilon`/`max_iters` + per-param step controls),
`random_search` (`n_trials`/`seed`). Every method hyperparameter has a CLI flag
that overrides the config (`--epsilon`/`--max-iters`/`--step-shrink`/
`--init-step`/`--step-min`/`--n-trials`/`--seed`, plus `--method`/`--rolls`/
`--pca`; per-param ranges stay in the config) — precedence CLI > config >
optimizer default. The EFFECTIVE method + its params (defaults filled in) are
echoed to the console banner, the `report.md` "method params" line, and the
`INDEX_<kind>.md` method column, so a one-off override is still fully recorded. `tests/calibration_registry.py` is the
COMPLETE catalog of every tuning constant per kind (tuple constants exposed per
element as `NAME[i]`, ints flagged); a config fits a SUBSET. **Tuning config object (dependency injection).** The 58 fittable constants are
the immutable `auto_negadoctor.Tuning` namedtuple. **Values are EXTERNAL** —
`tuning.py` holds the schema + per-field docs (JSON can't carry comments) and
`presets/*.json` hold the values (nested by calibration kind → pipeline
sub-stage via `tuning.GROUPS`; load flattens + validates, and still accepts a
flat preset); `DEFAULT_TUNING = tuning.load($NEGA_PRESET or
"default")` and is mirrored back onto the module globals (so the registry's
getattr/setattr utilities still resolve). `presets/default.json` is byte-identical
to the old hardcoded constants. Select a preset with `--preset NAME|PATH` /
`NEGA_PRESET`; the source holds NO hardcoded tuning values (only structural
correctness facts like `CLIP_SRGB_THR` / byte offsets stay inline). Adopting a
calibration result = drop in its `fitted_preset.json` (no editing `auto_negadoctor.py`).
See `presets/README.md`. The analysis functions (`process_roll`,
`detect_content_crop`/`_crop_decide`, `estimate_vignette`/`vignette_envelope`/
`fit_vignette_profile`, the pickers, `make_params`, `tune_print_params`,
`detect_dark_border`) take `cfg=DEFAULT_TUNING` and read constants as `cfg.NAME`.
`calibration_registry.to_tuning(overrides)` builds a per-trial cfg (handles
`NAME[i]` tuple elements + int coercion); the runner passes it down instead of
mutating globals, and writes both `fitted_params.json` (just what moved) and a
complete `fitted_preset.json` (drop-in). `apply`/`snapshot`/`restore` (global
setattr) remain as utilities but are no longer on the hot path.
PARALLELISM: (0) PREP — the trial-invariant precompute (decode + `_crop_fields` /
the fixed GT render) runs frame-parallel per roll via `_map_frames_prep` (was a
serial per-frame loop that idled the cores after process_roll); (1) per-frame
within a trial (`_eval_frames` → `an._map_frames`; vignette refolds via the same
pool) — all methods, ~4–5×; (2) per-trial for `random_search` (`_random` runs
trials in a thread pool, `--workers`/`fit.workers`; points sampled up front +
recorded in order → bit-identical to serial). Worker count = `an._proc_workers`
(`min(cpu,8)` default = memory-bandwidth knee; override `NEGA_PROC_WORKERS`). Enabled
by the cfg refactor (no shared mutation). Scaling is GIL-bound: render-heavy
inversion scales, crop's pure-Python `_crop_decide` less so. `coordinate_descent`
is a dependent chain — per-frame parallel only. Guards:
`test_calibration_runner.test_to_tuning_builds_cfg` /
`test_random_search_parallel_equivalence`, and byte-identical default confirmed
by the full quality gate (crop containment numbers unchanged).
The metric helpers are
SHARED with the gates: `run_quality_tests.histogram_per_frame` (extracted from
`_histogram_match_medians`) and `crop_overtrim` / `crop_overtrim_per_frame`.
`--review <session>` works for **all three kinds**: it runs the pipeline twice
(once under the session's FITTED constants, once under the LIVE source-code
constants), attaches per-frame `review={fitted,live}` payloads (kind-specific:
inversion params / crop border / roll vignette) via `write_debug_sessions`, and
opens the debug UI. There key **R** flips fitted↔live (swapping the payload into
`img_dict` so every render path uses it) and key **N** toggles the vignette
correction on/off in the preview (before/after). `_review_payload` /
`_review_run` build the payloads; the UI side is `_apply_review_source` /
`_toggle_review_source` / `_toggle_vignette` (smoke-tested).

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
- [~] Converge to the wb/print ground truth — IN PROGRESS.
      **Spec 04 (`specs/04_tune_algo_params_via_histograms.md`, 2026-06-14)
      reframed the loss: the ground truth is the rendered PICTURE (its
      HISTOGRAM), not the (non-unique, exposure↔wb↔black↔gamma-interdependent)
      params.** New offline instrument `tests/calibrate_histogram_match.py` +
      `nega_model.histogram_distance` (per-channel EMD, decomposed into a luma /
      brightness term, a color / chroma-cast term, and **top-highlight percentile
      deltas hi999/hi9999 + a highlight-color term**) measure, per GT frame, the
      EMD between the algorithm's render and the GT-params render over the content
      crop. **The metric compares the FLOAT render at 14-bit bins (16384) with a
      denser sample** — pass 1 used 8-bit/64-bin which was BLIND to the highlight
      shoulder (user: "you lack resolution"). FINDING 1: the algorithm rendered
      too DARK (NOT too bright as the earlier param-based AI calibration concluded
      — the proxy trap); the lever is GAMMA → `PRINT_GAMMA` 6.1→5.0 (joint EMD
      minimum; cut median total EMD roll 2510-11-1 0.1005→0.0771, 2512-2601-1
      0.0659→0.0589). FINDING 2 (after the resolution upgrade): the **highlight
      ceiling** is the dominant top-end lever — pass 1's 8-bit metric had it look
      flat/optimal-at-0.99, but the tuner over-pushed P99.9 to 0.99 vs the GT's
      lower top. `PRINT_HI_CEIL` 0.99→**0.97** cut median |ΔP99.9| ~33%
      (0.0162→0.0108) and total EMD ~16% (0.0599→0.0505) over all 4 rolls, centred
      brightness (signed −0.029→−0.002). Both clip-safe (<0.3%), AI gate still
      green. `soft_clip` is NOT a lever (cancels between the two renders). The
      residual is irreducible per-frame taste. Guarded by `check_histogram_match`
      (regression guard on total EMD + highlight |hi999| vs committed
      `histogram_baseline.json`). Rebuilt
      wb (region-cast + gentle neutralization) + print tune (clip-boundary
      brightness push). NOTE: the param gate (`check_ground_truth`)
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
- [x] gamma: PRINT_GAMMA = **5.0 (spec 04, histogram-matched over all annotated
      rolls)**, was 6.1 (the GT *param* median). The picture, not the param, is
      ground truth — see the convergence TODO and the spec-04 line above.
      Per-frame GT gamma 4.55–7.8 is aesthetic intent, not derivable. soft_clip
      stays at 0.75. (The old WB_PRIOR_WEIGHT taste-prior blend was removed in the
      2026-06-13 wb rebuild; wb is now region-cast based.)
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
- [ ] DSC_0002-class indoor frames: wb_low's region estimate can fall back to
      the roll-median (B-suppressing), which disagrees noticeably with the
      user's manual pick — the second-session shadow rect ((1.0, 0.748→) wb_low)
      worked. (The shadows PATCH was removed 2026-06-20 — wb_low is region-based
      / shadows-wheel-driven; this is now about the region estimate + roll-median
      fallback, not patch scoring.) Tuning direction from fixtures/annotations:
      user prefers stronger R in wb_high (DSC_0001: 1.72 vs applied 1.34) and
      milder B suppression in wb_low. Revisit region-wb scoring/fallback with the
      next annotation batch.
- [x] Float export confirmed live (8MB .tif, Dmin R 1.0367 unclipped) after
      a darktable restart; guards remain for regressions (Lua reads back
      format.bpp and warns; Python warns on 16-bit TIFF input and records
      source_16bit_tiff in the session).
