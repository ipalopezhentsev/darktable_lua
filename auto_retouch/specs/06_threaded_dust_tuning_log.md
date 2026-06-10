# Spec 06 — Threaded / Scratch ("stroke") dust: implementation & tuning log

> Source of truth for the thread/scratch detection + healing work (spec `06_threaded_dust_spec.md`).
> Lives in the repo (committed, portable) rather than Claude's home-dir memory.
> Append a new "Tuning pass N" section for each change; keep the running result line
> (`dots X PASS/Y WARN/Z FAIL; strokes N`) so regressions are obvious.

## Implementation state (as of 2026-06-06)

Implemented thread/scratch ("stroke") detection in `auto_retouch/detect_dust.py`,
`auto_retouch/debug_ui.py`, `auto_retouch/tests/run_quality_tests.py`,
`environment.yml` (+scikit-image). User prioritises **precision over recall** — a missed
thread is fine (fixed by hand in darktable), a false positive is not.

### How it works
- Strokes are darktable brush forms with N nodes along the centerline (same format as
  dot brushes; `mask_nb`=node count, `mask_type=72`). `make_stroke_mask_points()` emits
  Catmull-Rom bezier nodes. No Lua change needed.
- Canonical spot dict gains `kind` ("dot"|"stroke"), and for strokes: `path`,
  `stroke_width_px`, `length_px`. `cx,cy`=midpoint, `brush_radius_px`=half-width*scale.
- Detection: producer (a) = elongated components from the threshold binary, pre-filtered
  by FILL ratio (NOT bbox aspect — curvy threads have ~square bbox). Shared extractor:
  `skeletonize` -> longest path (double-BFS) -> `approxPolyDP` -> width via distance
  transform. Gates: ridge-vs-edge symmetry (`_stroke_side_brightness`), smooth-area
  texture (ring offset past TEXTURE_KERNEL to avoid self-contamination), achromatic,
  length/elongation.
- Weighing (spec 1.2a): elongated accepted dots convert to strokes when a circle would
  heal much more clean area (`STROKE_PREFER_RATIO`).
- Source: `find_stroke_healing_source` finds a translation onto smooth bg (see pass 22).
- Debug UI: strokes render as polyline + draggable key-point handles + translated source
  polyline; click a handle to select, Ctrl+Click to move; persisted as `path_overrides`.

### Key decisions / limitations
- **Producer (b) ridge pass (faint scratches) is DISABLED by default**
  (`STROKE_RIDGE_ENABLE=False`). Evidence: the only ground-truth faint scratch (DSC_0026)
  sits at the noise floor; any sensitivity that catches it floods FPs (500-865 stronger
  ridge structures per frame). Code kept for future use with real scratch labels. Faint
  scratches are added by the human in the debug UI.
- Ground truth: both `specs/feather_dust.png` and `specs/film_scratch.png` are crops from
  **DSC_0026** (template-matched). The feather thread is detected correctly.
- Regression: dots unregressed (9 PASS/27 WARN/0 FAIL, same as before). ~20 strokes across
  the 36-image set show as "new" because the committed baseline predates strokes.

### FP fix from user annotation (DSC_0026)
User marked 2 stroke FPs in the debug UI (deck mark @1900,3063; rigging wire @3655,1263);
left the feather @411,252 as TP. Added a **wide-context texture gate**
(`STROKE_MAX_CONTEXT_TEXTURE=7.0`, `_stroke_context_texture`, ~190px band): feather sky
ctx=3.4 vs wire=9.3, deck=18.8. Both FPs removed, feather + all real threads kept; dots
still 9 PASS/27 WARN/0 FAIL. (Straightness was useless — the deck FP was curvier than the
feather.) Set count dropped 20→17 strokes.

### Debug UI: draw missed thread
Press **T** to enter draw mode, then **click points OR drag freehand** along the thread,
**Enter** to commit (>=2 pts), **Esc/T** to cancel. Points drop on mouse-DOWN
(`_thread_add_point` from `_on_left_press`); dragging samples points every ~8 canvas px
(`_on_left_drag`). (Original bug: points were only added on a clean click-release, so a
drag was treated as rubber-band selection and nothing appeared. Rendering itself was always
correct — verified by simulating the full event flow.) Reload the UI to get code changes;
an already-open window runs old code. Stored in `{stem}_annotations.json` as `missed_strokes`
(`[{"path":[[x,y]...],"stroke_width_px":w}]`, default width `DEFAULT_MISSED_STROKE_WIDTH=8`).
Rendered as cyan polyline + square handles; click one then **Del** to remove. This is an
ANNOTATION (like missed_dust) — NOT auto-healed by AutoRetouch_InPlace and NOT yet wired
into run_quality_tests as expected ground truth (both are possible follow-ups).

### Tuning pass 2 (2026-06-07): catch SHORT threads, kill new FP classes
User annotated DSC_0004 (session 1780817993) with 2 clearly-visible missed threads on
textured green foliage. Root causes + fixes (all in detect_dust.py constants):
- **Too short**: threads were 21px & 36px; `STROKE_MIN_LENGTH_FRAC` 0.012 (~45px) → **0.005**
  (~19px). Sweep showed graceful growth, not a flood — the ridge/elongation/texture gates
  are the real FP guard, not length.
- **Fill pre-filter dropped thread1**: it has fill=0.40 > old `STROKE_MAX_FILL_RATIO`=0.35,
  so Producer (a) skipped it BEFORE skeletonization. Raised to **0.50** (cost gate only; a
  straightish 4px thread in a 10px bbox fills ~0.4; a round dot ~0.78).
- **Texture gates too tight for foliage**: threads sit at band/ctx≈7-8 (NOT smooth sky).
  But blanket-raising the gates flooded DSC_0029 with **78 backlit reed-stalk FPs**. Fix:
  **contrast-tiered texture** — faint marks (contrast < `STROKE_HICONTRAST_MIN`=45) keep the
  tight gate (band 6 / ctx 7); high-contrast fibers get the relaxed gate (band 10 / ctx 11).
  Measured separator: real threads contrast 48-79 vs reed flood median 21. Reeds 78→6
  (remaining 6 are isolated bright stalks, locally indistinguishable from fibers — accept).
- **Printed text/letters FP class**: DSC_0002 had 16 letter/number strokes ("5","6","W"...)
  — high-contrast so the relaxation let them in, but WIDE (10-19px). A fiber is thin; width
  histogram is bimodal with a gap at ~9px. `STROKE_MAX_WIDTH_FRAC` 0.006 (~22px) → **0.0025**
  (~9.5px). DSC_0002 16→0.
- **Result**: dots unregressed (9 PASS/27 WARN/0 FAIL). Total strokes 47 across 36 frames.
  DSC_0004 now finds all 3 real threads (2 annotated + 1 bonus). Worst FP case DSC_0029=6
  (sunset reed field, adversarial). Texture is now a 2-tier system keyed on contrast.

### Tuning pass 3 (2026-06-07): "crispiness" gate + thin-fiber area floor
User annotated DSC_0007 (session 1780821100): 2 FP "thick wires" + 1 missed crisp dust.
Key user insight: **dust is CRISP (sharp edges, clear white body); scene wires/hairs are
SOFT/fuzzy** (slightly out of focus — dust sits ON the negative in the scanner's focal
plane, scene elements don't). Two fixes:
- **Crispiness gate** (`stroke_edge_crispness` + `STROKE_MIN_CRISPNESS`=0.40): median of
  (max perpendicular edge-gradient / bump amplitude) sampled along the centerline.
  ~1.0 for in-focus dust, ~0.3 for a defocused wire; normalised by amplitude so it is
  contrast-independent. Measured: FP wires 0.19/0.28 vs real dust 0.48-0.61 (threads,
  feather, the missed fiber) — clean gap. Stored as spot `crispness`. Runs last in
  build_stroke_spot (most expensive). NOTE: edge width is in px, so at very different
  scan resolutions the 0.40 cutoff may need revisiting (both annotated frames ~3770px).
- **Thin-fiber area floor**: the missed dust (2px×25px, area 43) was dropped by the old
  absolute `STROKE_WEIGH_MIN_AREA=60` producer-(a) floor. Replaced with a resolution-
  relative `min_cand_area = STROKE_MIN_WIDTH_PX * STROKE_MIN_LENGTH_FRAC * min_dim` (~28),
  and deleted the now-unused STROKE_WEIGH_MIN_AREA constant.
- **Result**: DSC_0007 both wire FPs gone, missed dust found (only stroke there). Dots
  unregressed (9/27/0). Total strokes 47→39 (crispness cut soft wires/reeds; area floor
  added thin crisp dust). All prior TPs survive (feather 0.45 — margin only 0.05 above
  the 0.40 cutoff, watch this).

### Tuning pass 4 (2026-06-07): tail hysteresis (brush tracks thread to its true end)
User (session 1780822351) confirmed the DSC_0007 dust is now found but the brush stopped
short — the hair fades GRADUALLY at its lower end (core diff 71 → tail diff 40-57, just
under the global threshold 57.8 → tail clipped from the binary component). Fix = local
hysteresis (Canny-style edge linking):
- `hysteresis_extend_component()`: re-threshold a padded window around the seed bbox at
  `STROKE_HYST_LOW_FACTOR`=0.55 × main threshold, keep only low-thr components CONNECTED
  to the seed. `STROKE_HYST_PAD_FRAC`=0.008×min_dim window pad.
- `_extend_stroke_tail()`: applied AFTER build_stroke_spot accepts the stroke — it only
  lengthens the healing path; validation (crispness/color/ridge) stays on the confident
  seed. KEY reason: the faded tail is often colour-fringed (chromatic aberration) and
  fails the saturation gate — so judging the thread by its tail wrongly rejects it. Keeps
  seed width; skips extension if it would exceed 2.5× seed length (runaway guard).
- Wired into Producer (a) only. Weighing's stroke_area now uses `spot["length_px"]` (post-
  extension). **Zero detection-count change** (39 strokes, dots 9/27/0) — purely lengthens
  paths. DSC_0007 hair path now y1070→1104 (was →1094; user's end ~1107, within brush
  border). Canonical: XMP/overlay/UI all read spot["path"], so the longer brush is automatic.

### Tuning pass 5 (2026-06-07): ridge-crest core sampling (white-on-white threads)
User (session 1780823530, DSC_0008) marked 2 missed threads on a near-WHITE bright
background (local_bg ~227):
- **Thread B (crisp white, curvy)** formed a clean component but was rejected "edge":
  `_stroke_side_brightness` sampled the core as the single-pixel centerline MEDIAN (236),
  which underestimates the true ridge crest (peak 253) because a thin curvy skeleton sits
  ~1px off the brightest line. On a bright bg that 17-level underestimate dropped
  ridge_drop to 7 (< STROKE_MIN_RIDGE_DROP=8). FIX: sample the core as the perpendicular
  CREST (max across ±(width/2+1) px) → core 253, ridge_drop 24. Still rejects step-edges
  (their bright side ≈ crest). Dots unregressed (9/27/0); total strokes 39→45 (recovers
  thin ridges previously under-measured on bright/varied backgrounds). Thread B now found
  (crisp 0.73).
- **Thread A (faint long horizontal)** still MISSED and left so deliberately: it is ~2x
  the noise floor per-pixel (diff 11-25 vs threshold 22, gray ~240 vs bg 227) and dips
  BELOW even the hysteresis low threshold (0.55*thr≈12) at several points, so it neither
  seeds nor bridges. Experiment (extend-then-validate, seed area→5): flooded the test set
  39→100 strokes AND still missed A. Reliably catching A needs a line-INTEGRATING ridge
  detector (SNR grows with sqrt(length)); the sato producer (b) is exactly that and is
  disabled because it floods. Conclusion unchanged: truly sub-threshold faint threads are
  a human-adds-in-UI case, OR a future controlled ridge pass gated by the now-stronger
  crispness/crest-ridge/context-texture filters.

### Tuning pass 6 (2026-06-07): tighten ridge-vs-edge (crane sky/structure outline FP)
User (session 1780825567, DSC_0008) marked a step-edge FP: the bright winter-sky vs dark
crane-boom OUTLINE traced as a "thread" (width 5.6, contrast 29, crisp 0.70). Perpendicular
profiles were monotonic dark→bright (a STEP), not dark→bright→dark (a RIDGE). It slipped the
ridge gate because (a) pass-5's crest(max) core sampling sits ~10 above the noisy sky-side
MEDIAN, manufacturing a fake ridge_drop, and (b) asym 0.25 squeaked under the old 0.30.
Measured separation is wide: real threads asym 0.01-0.05 / ridge_drop 21-153 vs crane
asym 0.25 / ridge_drop 10. FIX: `STROKE_MAX_SIDE_ASYMMETRY` 0.30→**0.15** (primary — a
floating dust ridge has near-equal sides; a step-edge is inherently one-sided),
`STROKE_MIN_RIDGE_DROP` 8→**12** (second guard). Crane gone, all TPs kept (threadB
ridge_drop 21.5, asym 0.01). Dots 9/27/0; total strokes 45→42. User's mental model to
remember: **dust "floats" OVER the photo (bright on both sides); an object outline is just
one region meeting another (bright one side, dark the other) — no white thread there.**

### Tuning pass 7 (2026-06-07): texture is a backstop + FIELD-ISOLATION gate
User (session 1780826390, DSC_0009 harbor scene) marked 4 missed crisp WHITE threads (a
hook, a tadpole, a long curve) on grain/water. They were rejected by the contrast-tiered
texture gate: crisp (0.61-0.73) but only MODERATE contrast (18-43), so they fell in the
"faint→smooth-bg-only" tier despite band/ctx being just 6-8. Key realisation: **dust floats
over the WHOLE photo, textured areas included — texture must not by itself reject a thread.**
Two changes:
- **Dropped contrast-tiering**; texture is now a single loose BACKSTOP
  (`STROKE_MAX_BAND_TEXTURE`=`STROKE_MAX_CONTEXT_TEXTURE`=9.0). Removed STROKE_HICONTRAST_MIN
  and the *_HI consts. All known FPs are caught by crisp(<0.40)/asym(>0.15)/ridge_drop(<12),
  NOT texture (deck crisp 0.24, reed 0.28-0.31, wire rd 3, crane asym 0.21). NOTE: cannot
  raise the crisp floor past 0.44 — feather=0.45, DSC_0004 thread1=0.48 sit just above it.
- **Field-isolation gate** (NEW): relaxing texture re-flooded DSC_0029 (50 reeds) because a
  crisp bright reed stalk IS a valid-looking crisp symmetric ridge — only its DENSITY tells
  it apart. Reject a stroke with > `STROKE_FIELD_MAX_NEIGHBORS`(1) other strokes within
  `STROKE_FIELD_RADIUS_FRAC`(0.12)*min_dim (~450px). Measured: isolated real threads 0-1
  neighbours; reed-field stalks 2-11 → DSC_0029 50→4, DSC_0009/0012/0033 (isolated) all kept.
  Implemented as a post-filter on stroke_spots in detect_dust_spots; a density-rejected
  dot→stroke CONVERSION restores its dot (tag `_conv_label`, recompute labels_to_drop after
  filtering). Texture-vs-reed sweep proved NO single texture threshold separates DSC_0009
  threads (need ≥8.3) from the DSC_0029 field (≥27 reeds at 8.3) — isolation was required.
- **Result**: dots 9/27/0; total strokes 42→62 (texture relax catches more real grain/foliage
  threads), max 7/image, no floods. All 4 DSC_0009 threads + every prior TP found; crane/
  reed/wire FPs rejected. Mental model: a thread "floats ALONE"; reeds/rigging come in fields.

### Tuning pass 8 (2026-06-07): line-candidate field gate (yacht masts/rigging)
User (session 1780829254, DSC_0011 marina) marked 4 FPs: yacht masts + rigging + a deck
"meteostation" on grey sky. These are thin, crisp (0.42-0.78), achromatic, symmetric ridges
on smooth sky — LOCALLY identical to dust. Straightness fails (short real threads are also
straight 1.0); accepted-stroke density fails (rigging fragments are sparse, 0-1 neighbours).
The real signal: rigging/masts are embedded in a FIELD of many thin line-structures even
when few survive as accepted strokes. Added a **line-candidate field gate**: count ALL
elongated bright components (area≥12, longdim≥10, fill<0.45, elong≥2.0) in a 60..0.15*min_dim
annulus around each stroke; reject if ≥ `STROKE_FIELD_MAX_LINE_CANDS`=24. Measured: isolated
real threads 0-2 (one in grain reached 18) vs rigging 17-40. Kept the older accepted-stroke
neighbour gate too (≥2 within 0.12*min_dim) as a 2nd signal — together: reeds 50→2, yacht
5→2, harbor structures trimmed. Both reuse the connected-components already computed; a
density-rejected dot→stroke conversion still restores its dot.
- **Result**: dots 9/27/0; total strokes 62→49 (FPs trimmed across reeds/rigging/harbor),
  all real threads kept (DSC_0009 A at line-density 18 < 24 survives). 3 of 4 yacht FPs gone.
- **KNOWN RESIDUAL**: FP3 (a lone short crisp straight line on open sky, line-density 17 <
  threshold, 0 neighbours) is genuinely INDISTINGUISHABLE from a real dust fiber by every
  available feature — it stays detected. Pushing the gate to catch it would reject real
  isolated threads (real thread A sits at 18, just above FP3's 17). This is the limit of
  local+regional discrimination; the marina context needs scene-level understanding. Such
  lone fragments are a mark-in-UI case. Do NOT lower the line-candidate threshold below ~20.

### Tuning pass 9 (2026-06-07): circle-fill dot gate; KEPT line-candidate gate (precision)
User (session 1780837365, DSC_0012) flagged two things:
1. **A clear S-curve thread on locally-clean background was MISSED** by pass-8's line-candidate
   field gate (DSC_0012 is heavily dusted, 606 elongated marks image-wide → 38 line-candidates
   in the thread's annulus, over the ≥24 reject bar). I first reverted the gate to recover the
   thread, but the user OVERRULED that: they prioritise PRECISION over recall — the gate
   REDUCES yacht-rigging FPs, and missing this thread is an acceptable false negative (they fix
   it by hand in darktable). So the line-candidate field gate is KEPT (both signals:
   line-candidate density + accepted-stroke neighbours). Net effect favours fewer FPs (yacht
   5→2, reeds 50→2) at the cost of some thread recall on busy/dusty frames — the desired trade.
2. **A twisted rope/cord fragment was detected as a circular DOT with a 62px brush** ("not a
   dot by any measure"). The rope spawns ~12 elongated fragments (aspect 0.38-0.56, circ
   0.19-0.34) that passed the dot shape gates (MIN_ASPECT_RATIO=0.3, MIN_CIRCULARITY=0.15 too
   loose) and got enc_r from their LONG axis → huge over-covering brush (3*enc_r). Added a
   **circle-fill dot gate**: reject when enc_r > `DOT_IRREGULAR_RADIUS_FRAC`(0.005)*min_dim
   (~18px) AND area/(pi*enc_r^2) < `DOT_MIN_CIRCLE_FILL`(0.40). Catches all 4 rope frags;
   removes only ~8/984 baseline dots (themselves large irregular blobs). Small irregular
   specks (small brush) and round dots (fill ~0.9) are untouched. Placed right after enc_r is
   computed in the dot loop. Dots verdict unchanged 9 PASS/27 WARN/0 FAIL.

### Tuning pass 10 (2026-06-07): field gate must not kill SMALL clusters of real threads
User (session 1780840455, DSC_0031) flagged 2 very prominent crisp WHITE threads on BLUE SKY
as missed ("sky doesn't have such threads" — i.e. unambiguous dust, must detect). Cause: the
accepted-stroke neighbour gate. The two threads + a fragment = 3 thread-pieces clustered in
the sky, each seeing 2 neighbours within 0.12*min_dim, and the gate rejected anything with
>1 neighbour — treating a 3-thread cluster like a 50-reed field. The line-candidate gate
correctly KEPT them (only 1-2 line-candidates: blue sky is empty apart from the threads). FIX:
`STROKE_FIELD_MAX_NEIGHBORS` 1 → **2** (allows a cluster of up to 3 mutually-close real
threads; dense fields have many more neighbours each). Sweep confirmed: at 2, DSC_0031 sky
threads 2/2 detected while DSC_0029 reeds stay 2 and DSC_0011 yacht stays 2. The
line-candidate gate is the PRIMARY field detector; the neighbour gate is only a backstop and
must stay loose enough not to reject obvious clustered dust. Dots 9/27/0; total strokes 49→52
(just the 2 sky threads), no floods. (3rd DSC_0031 thread T3 is on a TEXTURED bg — build=
texture, 35 line-candidates — left as an acceptable miss per precision-over-recall: catching
it needs weakening the field gate, which readmits rigging FPs.)

### Tuning pass 11 (2026-06-07): per-stroke brush width from the brightness profile (healing)
User switched to testing HEALING (brush application in darktable). Thick threads left a
whitish leftover because the brush was too thin. Cause: brush_radius came from the
distance-transform width, which measures only the above-threshold CORE — a twisted thread
measured 4px but is visibly ~14px wide (peak diff 125, off-centre skeleton). FIX: new
`stroke_coverage_halfwidth(path, diff)` samples the perpendicular diff profile along the
path, finds the local peak, walks OUTWARD contiguously (so a neighbouring thread's separate
bump isn't included) until diff < max(STROKE_COVERAGE_MIN_DIFF=10, STROKE_COVERAGE_FRAC=0.15
*peak), and returns the 80th-pct half-width FROM THE CENTERLINE (handles off-centre). Brush:
`brush_radius_px = max(STROKE_MIN_BORDER_PX, cover_hw+STROKE_COVERAGE_MARGIN_PX=2.0,
width/2*STROKE_BORDER_SCALE)`, capped at STROKE_MAX_BORDER_FRAC(0.006)*min_dim (~22px). Result
on DSC_0004: thick thread #3 brush 4.0→6.5 (covers 8→13px), thin thread #2 stays ~4.5 —
adaptive per stroke. Detection/dots unaffected (brush size only; strokes aren't in baseline).
POSSIBLE FOLLOW-UP: darktable brush nodes each carry their OWN border, so a variable-width
thread could get a PER-NODE width (thick at the twist, thin at the ends) instead of one
uniform radius — make_stroke_mask_points currently uses a single border for all nodes.

### Tuning pass 12 (2026-06-07): clipped-white threads on bright sky (seed + crisp bypass)
User (session 1780851408, DSC_0020) flagged 2 prominent white threads on light-blue sky as
missed. Root cause: both are CLIPPED to gray=255 but the sky background is bright (217-233),
so diff is only 22-38 — below the global threshold (49, high because noise=16.3). They never
form a diff-threshold component. AND one (T2) read crisp=0.38 (<0.40) because the clipped flat
255 plateau on a noisy bright bg gives tiny amplitude — the crispness metric is unreliable
there. Both fixed:
- **Producer (c)**: seed strokes directly from the near-clipped binary `gray >= STROKE_CLIP_LEVEL`
  (252). build_stroke_spot still gates them; overlaps with existing strokes skipped; tails
  extended. (Separate from producer (a) so the dot/weighing label space is untouched; dots
  never see the clip binary, so blown highlights aren't detected as dots.)
- **Crispness bypass**: if the core crest >= STROKE_CLIP_LEVEL, skip the crispness gate — a soft
  scene wire is NEVER blown to 255, so clipping itself certifies hard bright dust. ridge_drop/
  asym/texture/field still apply.
- Result: DSC_0020 T1+T2 (+a 3rd real thread) detected; dots 9/27/0; total strokes 52→64,
  distributed ~1-2/image (clipped-white dust previously missed on bright backgrounds), no
  floods. Visual check of DSC_0020/0011/0009 new strokes = genuine white threads, not blown-
  highlight FPs. Sky tops out at p99.9~251 so 252 is a clean signal. Aligns with precision
  goal: clipped-white-on-non-white is unambiguous dust.

### Tuning pass 13 (2026-06-07): faint axis-aligned scratch producer (film transport)
User wants long, thin, nearly-straight FILM-TRANSPORT scratches detected (35mm travels
horizontally → horizontal scratches; but a frame may be rotated in darktable, so the export
can have them vertical). DSC_0018 (session 1780858058) had 2 missed horizontal streaks. They
are too faint to cross the global threshold (diff ~27-41 vs thr 35.6) and not clipped, so no
existing producer seeds them. NEW **Producer (d)** = `detect_axis_streaks(diff, min_dim)`:
for each axis (horizontal, then vertical via transpose) build a ridge response
(diff - max(row +VGAP, row -VGAP) — bright vs BOTH vertical neighbours, excludes step-edges),
integrate ALONG the line with `cv2.blur((STREAK_INTEG_LEN=41,1))` to lift the faint line
above noise, threshold at `max(6, noise*STREAK_LEVEL_MULT=1.6)`, keep long(>0.04*min_dim) thin
(<=6px) strongly-elongated(>=8) components. Each candidate is then validated by
build_stroke_spot — which is what makes it precise: scene lines on busy/coloured backgrounds
are rejected, only scratches on quiet achromatic areas survive. KEY measured result: at
level_mult 1.6 the WHOLE test set yields just 3 scratch survivors (2 real DSC_0018 + 1
DSC_0012), i.e. ~0 FPs; the raw response had 24/16/15 horizontal candidates on some images,
all filtered by build_stroke_spot. Dots 9/27/0, total strokes 64→67.
- streak2 (the fainter DSC_0018 one, diff mostly 8-26 at the noise floor) is NOT caught:
  catching it needs level_mult 1.2 which brings scene-line FPs — left as an acceptable miss
  per precision-over-recall. Constants: STREAK_* block; master switch STREAK_DETECT.
- SLOPE LIMITATION: bh<=6 over the full width means only ~axis-aligned; a strongly perspective-
  sloped scratch (bh grows with width) would be rejected. Both axes cover 0°/90° rotation; small
  slopes within the ridge response's ~10° tolerance survive if the component stays thin.

### Tuning pass 14 (2026-06-07): scratch tail-extension reaches the real end (gap bridging)
User: DSC_0018 scratch #23 ended too early on the right (x=1029) — it continues into the sky
to ~x=1140. The faint scratch is BRIGHT again at the tail (diff 38-41 at x=1130-1140) but has
NOISE GAPS (diff dips to 13-17 at x≈1050 and x≈1110-1120) that broke the diff-hysteresis
connectivity. Two fixes to the tail extension (`_extend_stroke_tail` / `hysteresis_extend_component`):
1. Window pad now scales with seed length (`STROKE_HYST_PAD_LEN_FRAC`=0.30) — a long scratch's
   tail is far past its seed end (the fixed ~30px pad couldn't reach it).
2. Bridge noise gaps ALONG the stroke's dominant axis with a directional morphological close
   (`STROKE_HYST_BRIDGE_PX`=21, kernel 21x1 horizontal or 1x21 vertical depending on the path's
   dominant axis — 1px across the axis so it does NOT widen the brush). Axis taken from
   spot["path"] endpoints. Result: #23 now x=258..1140 (len 727→887), reaching where the scratch
   fades (diff 41→9 past x=1140). Dots 9/27/0; total strokes 67→66 (a minor merge). The 2.5x-
   seed-length runaway guard still applies, so the bridge can't make a stroke balloon.

### Tuning pass 15 (2026-06-08): RADON full-width faint scratch detector (Producer e)
User: DSC_0026 has a faint horizontal streak across the WHOLE frame, undetected. It is at the
noise floor (diff ~10 ≈ 1.3*noise), fragmented with 600px+ gaps, fails the per-point ridge gate
(ridge_drop ~10<12), and slightly sloped — the per-segment producer (d) cannot link it. The
ONLY signal is collinearity across the frame. User chose (AskUserQuestion) "build it,
conservative". Implemented **Producer (e)** = `detect_radon_streaks(diff, local_std)`:
- Smooth-region-restricted horizontal ridge response; accumulate the MEAN response along
  candidate near-horizontal lines (slopes ±0.02) — collinear fragments reinforce. CRITICAL BUG
  fixed: cv2.warpAffine shear mis-sampled (picked a phantom line at the wrong y/slope); replaced
  with a DIRECT vectorized line accumulation (group columns by round(slope*x) offset, roll+sum).
- Accept only strong full-width lines: `STREAK_RADON_MIN_RESP`=3.0, `STREAK_RADON_MIN_COV`=0.35.
  Measured: real-streak frames 3.2-5.2, clean frames <=2.5 → catches the recurring transport
  scratch on DSC_0018/0025/0026/0027 (same scratch across consecutive roll frames!), 0 clean FPs.
- The straight Radon line over-tilts to the strongest segment, so REFINE: track the actual
  response peak in a ±16px band per x (median-smoothed, keep line estimate across gaps) so the
  brush sits ON the (slightly curved) scratch. DSC_0026 path now left y65→right y51 (matches the
  user's annotation). Spot built DIRECTLY via `_make_radon_streak_spot` (bypasses build gates —
  the Radon accumulation IS the validation). Brush half-width capped at STREAK_RADON_MAX_HALFWIDTH
  =5 (the faint coverage measure over-estimated to 16 → a 32px brush; now ~9px).
- HORIZONTAL ONLY for now (user's case); a transposed vertical pass would handle 90°-rotated
  frames (follow-up). Dots 9/27/0; total strokes 66→70 (the 4 real scratch frames).
- Endpoint extension (user feedback: line stopped too early on the right in 0025/0026): after
  the `present`(>1.0) extent, walk x0/x1 outward while the band still shows faint signal
  (`STREAK_RADON_EXT_FACTOR`*PRESENT=0.35) with `STREAK_RADON_EXT_GAP`=120px gap tolerance,
  backing off the trailing gap. DSC_0026 right end 5340→5580 (near the 5617 edge, y48≈user's 47).

### Tuning pass 16 (2026-06-08): stroke healing source clearance (darktable artifacts)
User (testing healing on DSC_0022 scratch): the heal source was too close to the defect line,
so darktable's clone produced correction artifacts (it dislikes a source touching/intersecting
the defect). `find_stroke_healing_source` used `d_min = max(2*brush_radius, 2*width, 6)` (~9px)
AND its score prefers the NEAREST smooth option, so on uniform sky it picked the minimum — the
source brush ended up edge-to-edge with the defect brush (zero gap). FIX: require a real gap:
`d_min = max(STROKE_SOURCE_OFFSET_FACTOR(3.0)*brush_radius, 2*brush_radius+STROKE_SOURCE_MIN_GAP_PX(10),
2*width, 6)`. Now offset ≥ 2*brush_radius + ~10px → a ≥10px gap between the source and defect
brush edges. DSC_0022: offset 9→19px (gap 10). Stroke-only (dots use find_healing_source, NCC),
so the dot regression (9/27/0) is unaffected. (Superseded by pass 22 for curved defects.)

### Tuning pass 17 (2026-06-08): fix bridge over-extending diagonal threads (regression)
User (DSC_0025 brush #63): a diagonal thread was detected as a wrong 152px HORIZONTAL line.
Root cause = pass-14's tail-extension gap-bridge, which `_extend_stroke_tail` applied to ALL
strokes. The seed was correct (diagonal, 67px, [[3058,1770],[3112,1742]]) but its dx>dy, so the
directional close picked a 21px HORIZONTAL kernel; closing the low-threshold binary bridged the
thread to spurious horizontal pixels → re-extraction returned a horizontal centerline (152px).
FIX: made the bridge OPT-IN (`_extend_stroke_tail(..., bridge=False)` default). Only producer
(d) (faint axis-aligned scratches, which genuinely need gap-bridging along their true axis)
passes bridge=True. Threads (producers a/c) extend via plain hysteresis (no morphological
close), so a merely-more-horizontal-than-vertical diagonal thread is not bent to the axis.
Result: #63 now correct (diagonal 75px x3050-3111 y1742-1769); DSC_0018 scratch tail still
reaches x1140 (bridge preserved for scratches). Dots 9/27/0; total strokes 70 (unchanged).

### Tuning pass 18 (2026-06-08): heal-split — brush pauses over busy regions
User: the DSC_0026 full-width scratch heals badly where it crosses the ship FUNNELS/rigging —
darktable's clone grabs bad patches from the busy area and makes artifacts (and the defect is
nearly invisible there anyway). FIX: `split_stroke_at_busy(spot, local_std, min_dim)` resamples
the path (8px), marks points where local_std > `HEAL_MAX_TEXTURE`=12 as busy (+`HEAL_BUSY_MARGIN_PX`
=25 around them), and emits one sub-spot per contiguous SMOOTH run >= `HEAL_MIN_SEGMENT_FRAC`
(0.01)*min_dim. Each sub-stroke is a separate brush form with its own source, so darktable
"pauses" over the busy part. Measured along the scratch: sky texture 3.6-6.2 vs funnel/rigging
14-54 — clean gap at 12. DSC_0026 scratch 1→2 subs (x0-854 sky | SKIP x854-1581 funnels |
x1581-5580 sky). APPLIED ONLY TO source=="scratch": a first attempt ran it on all strokes and
dropped 70→31 (threads on grain have local_std spikes >12 → marked busy → dropped); short threads
sit on locally-uniform bg and must pass through whole. Dots 9/27/0; total strokes 70→76 (scratches
splitting). Wired after the field gate, before source-finding (so each sub-stroke gets a source).

### Tuning pass 19 (2026-06-09): bolder Radon scratch brush (full darktable coverage)
User: on DSC_0026/0027 the Radon scratch heals incompletely where the line is BOLDER — the
uniform 9px brush (brush_r 4.5) covers the thin pieces but not the thicker ones. Measured
half-width along the scratch: 2.5-5px (fairly uniform, max ~5 → 10px wide). FIX: made the
Radon scratch brush deliberately BOLD — `brush_radius_px = max(STREAK_RADON_MIN_BRUSH_R=8,
cover_hw + STREAK_RADON_BRUSH_MARGIN=4)`, cap STREAK_RADON_MAX_HALFWIDTH 5→9. Now 16px wide
(brush_r 8), covering the thickest pieces + margin. Safe to over-heal: the extra width is on
smooth sky and heal-split (pass 18) already excludes busy crossings. Source clearance (pass 16)
scales with brush_radius so the source stays clear (offset now ~26px). Dots 9/27/0; strokes 76
(unchanged — width only). NOTE: make_stroke_mask_points packs ONE border per node; if a scratch
ever has pieces wider than ~16px, the proper fix is PER-NODE widths (measure half-width per node,
pass a border list) — flagged but not needed here (uniform thickness).

### Tuning pass 20 (2026-06-09): brush covers whole component (hooks) + final dedup
User (DSC_0025 stroke #41): a checkmark/hook dust blob was only partly covered — the skeleton
LONGEST-PATH centerline traverses only 2 of the hook's 3 branches, so the down-left tail (~15px
off the centerline) sat outside the 5px brush. FIX 1: `component_cover_radius(path, comp, bx, by)`
= 99.5th-pct distance from each component pixel to the densified path; `_widen_brush_to_cover`
grows brush_radius to cover it (capped at STROKE_MAX_BORDER_FRAC*min_dim). For a thin thread this
is ~its half-width (no change); for a hook/comma it reaches the off-axis branch. #41 brush 5→15.5
(covers the tail); thin threads unchanged. Called in producers (a) and (c) after build+extend.
FIX 2 (found while debugging): the white hook was detected TWICE — producer (a) (diff) AND the
clip-seed producer (c) — because `_stroke_overlaps_existing` runs on the PRE-extension seed
midpoint, so the post-extension coincidence is missed. Added a FINAL dedup on finalised
midpoints (after all producers, before the field gate): greedily keep the larger-brush version
of any pair within min_sep with length within 50%. Dropping a converted-from-dot stroke here
restores its dot (labels_to_drop recomputed after). Dots 9/27/0; total strokes 76→75 (1 dup
removed). NOTE: still single-radius per stroke; true per-node widths remain the deeper fix.

### Tuning pass 21 (2026-06-09): cover bushy/feathered thread spots (brush boldness)
User (DSC_0026 shape #23 = the feather, brush 7.9): heal artifact where the thread is BOLDER.
Measured: feather half-width median 2.0px but MAX 10.2px (a bushy spot) — the wide spot is
faint (below detection threshold so not in the component_cover) AND a small fraction (so the
80th-pct coverage missed it). FIX: `STROKE_COVERAGE_PCTL` 80→92 and `STROKE_COVERAGE_MARGIN_PX`
2→4, so the brush covers the thread's widest visible spot. Feather brush 7.9→10.6 (covers 10.2).
Side effect: branched/forked/wide threads now get bold brushes (DSC_0009 fork len190→brush 20;
a faint blurry one len61→21.5) — verified these are REAL branched/wide threads that the bold
brush correctly covers (component_cover from pass 20 + this), and the user prioritises full
coverage over minimal over-heal. Dots 9/27/0; strokes 75→73. THE PERMANENT FIX for the
recurring "varies in thickness" requests (scratch pass19, feather pass21) is PER-NODE widths:
darktable brush nodes each carry their own border, so measure half-width per node and pass a
border list to make_stroke_mask_points (currently one border per stroke). Offered; do it if the
uniform bold brush ever over-heals visibly on a textured background.

### Tuning pass 22 (2026-06-10): stroke heal source must CLEAR the curved defect (radial search)
User (DSC_0031 brush #19, the curved sky hair): the heal source intersected the hair → white
smear in darktable. Pass-16 only enforced a *slide distance* (`d_min`) perpendicular to the
straight chord p0→p_last, and scored the band by MEDIAN texture. Both fail for a CURVED hair:
the chord-perpendicular slide bends the rigid copy back toward another section of the same hair
(true gap << d_min), and a thin crossing barely moves the median so it passes. Rewrote
`find_stroke_healing_source`:
- **Densify** the defect centerline (8px) so curvature is represented in the gap test.
- **True clearance gate**: reject any offset whose translated path comes within
  `clearance = 2*brush_radius + STROKE_SOURCE_MIN_GAP_PX` of ANY defect point (point-to-point
  min, not slide distance). This is the real fix — guarantees the source brush never touches
  the defect brush regardless of curve.
- **Radial search** (24 directions × distances) instead of ±chord-perpendicular: a curved hair
  has no single "away" direction; an oblique translation is often the only one that carries the
  whole copy clear. For a straight stroke the perpendicular still wins (nearest clean offset).
- **Texture**: median → 90th-pct of `local_std` along the band (a thin crossing structure hides
  in the median).
- **Clearance-guaranteeing fallback**: if no clean-texture offset exists, take the NEAREST
  clearance-passing one (over-heal on slightly textured bg is recoverable; an overlapping source
  smears — never return that). Only if NOTHING clears in range (tiny image/huge defect) fall back
  to the old chord-perpendicular d_min slide.
- **Result**: DSC_0031 all 3 sky strokes now gap≥required (was #19 gap 5.5 < 30.9 → 31.0; src
  moved y1000→953, "a bit higher" as asked). Dots 9 PASS/27 WARN/0 FAIL, strokes 73, 0 missing —
  zero detection change (source positions only; strokes aren't in the numeric baseline). NOTE:
  `run_quality_tests` does NOT compare stroke source positions (only dot sources via NCC), so
  stroke-source changes must be verified ad-hoc (per-stroke min gap vs 2*br+10).

### Debug UI fix (2026-06-07): drawn missed threads now show in the table
`_finish_thread` did not refresh the spot table, and `_populate_spots_list` only iterated
`missed_dust`. Added a `missed_strokes` loop (rows `t0,t1,…` at path midpoint, kind
`missed_stroke`) + `_populate_spots_list()` call in `_finish_thread`; wired the new kind into
`_on_spot_tree_select` / `_center_on_selected_spot`. (Drawing itself: click points OR drag
freehand, Enter to commit — see the draw-missed-thread section above.)

### Regression suite extension (2026-06-10): baseline-independent source/geometry invariants
After pass 22 it became clear `run_quality_tests` diffs only DOTS against the baseline (which
has no strokes), so stroke sources/radii/paths were entirely unguarded — a regression like the
pass-22 overlap would pass silently. Added `check_spot_invariants(spots, w, h)`: per-spot
property checks that hold on EVERY run with no baseline needed. ERROR (fails the suite, exit 1):
healing **source brush overlaps the defect brush** (dots: centre-to-centre `sep < 2*br`;
strokes: min distance between the translated path and the densified defect path `< 2*br` — same
geometry the heal-source finder enforces), source out of image bounds, non-finite/missing coords,
`brush_radius_px<=0`, or a stroke `path` with < 2 points. WARN: runaway `brush_radius_px`
(> 0.02*min_dim), or a stroke source tighter than `STROKE_SOURCE_MIN_GAP_PX`. Wired into the main
loop (all images, baseline or not), a summary section, and the exit code. Current set: all 36
images clean (9/27/0 dots, 73 strokes, 0 invariant errors). The checker has its OWN self-test,
`tests/test_source_invariants.py` (synthetic bad/good spots — a "test for the test"), so the
guard can't silently degrade into a no-op. PRINCIPLE: extend `check_spot_invariants` whenever a
new feature adds geometry that must satisfy a property (don't rely on the dot-only baseline diff).

## TODO / not yet done
- Regenerate `tests/baseline_session/` (generate_baseline.py) to capture strokes — ONLY
  after the user approves the remaining stroke detections. (Until then strokes have no
  numeric baseline; the invariant check above is what guards them.)
- Optional: wire `missed_strokes` into run_quality_tests (expected ground truth) and/or
  into healing (apply pipeline reading annotations).
- Manual verification in darktable (apply AutoRetouch_InPlace on a thread frame, confirm
  the stroke heals using the parallel source) and a visual pass of the debug UI stroke
  editing.
