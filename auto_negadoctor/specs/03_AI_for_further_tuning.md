I annotated whole roll auto_negadoctor\tests\fixtures\annotations\2026-06-13_wb_print_roll with what I think white balance and exposure parameters should be and gave it to you to fine tune the algo so that it reproduces values closer to my preferred markings. You tried to do this, improved it a bit, but still there are some errors and in general the task stays on - I want futher improvement of the algo so it's closer to my manual markings.

Probably we reached some dead wall of purely analytical methods. Where general method works 'fine' but there are outliers of unclear nature with no easy way to detect them with conventional analysis.

Therefore, maybe it's time to bring some AI? My PC has quite powerful GPU so why not try?

Let's consider some approaches. For example:
1) Can we use reinforcement learning? I.e. we have my markings as the goal and we have tunable params - can we use this AI method to come up with a 'black box' algo what would be better than the current analytical one? (it shouldn't replace the current one, but be another option for tuning. Debug UI can even compute both ways and allow to switch between several variants)
2) Another kind of AI, local LLM with visual inputs which is able to categorize an image, detect if it's e.g. snowy or supposed to be dark as it's showing e.g. dark museum (and that would also turn white balance more 'orangier' as my preference). And these comments would then be interpreted by some tuner making changes to initially found params.
3) Some other local model, not LLM, but something for image categorization? I don't know this field and need guidance.

So, ideally let's explore them all, tune them to reach my annotations, and provide ability to switch in debug UI between them so I can see the diff.

---

## Implementation status (2026-06-14)

Approach **#2 (local vision LLM)** is built — it's the documented direction for
the residual (per-frame gamma/mood/warmth have ~0 correlation with pixel metrics
→ needs scene understanding, not more math). It's an OPT-IN alternate variant;
the analytical cropping/vignette/film-base/wb/print stages are untouched and run
first.

- `scene_tuner.py`: `categorize_scene()` (moondream via Ollama, cached; ~2.4s/
  frame vs ~15s for gemma3:4b whose vision encoder prefills on CPU) +
  `apply_scene_tuning()` (interpretable, GT-calibrated label→offset table for
  gamma/warmth/brightness, with the hard-clip guard always re-applied).
- Trigger: `--ai-tune` flag and two Lua actions (**AutoNegadoctor_AI_Debug**,
  **AutoNegadoctor_AI_InPlace**). Debug UI: key **A** switches Analytical↔AI to
  see the diff (plus the LLM's scene/rationale in the info panel).
- Tests: `tests/test_scene_tuner.py` (clip-safety, bounds, offline fallback);
  `tests/calibrate_scene_tuning.py` (offline label→offset calibration vs GT).

**Approaches #1 (RL/black-box) and #3 (image categorization model) are
DEFERRED.** On the "two more rolls → enough for RL?" question: ~110 frames is
still too few for a from-scratch policy net (overfit). What becomes viable at
that size is (a) CMA-ES black-box optimization of the analytical CONSTANTS and
(b) a regularized learned wb-residual (sklearn), both with cross-roll holdout —
the right "black-box" reading of #1/#3. The gamma/mood residual stays the LLM's
domain regardless of data volume. Revisit as Phase 2 once 2 more rolls are
annotated; expose as a third selectable variant.

## Calibration pass against the 2026-06-13 GT roll (2026-06-14)

Calibrated `scene_tuner` against the full 37-frame ground truth (the user's
`--ai-tune` run; LLM labels frozen as `tests/fixtures/scene_labels.json` so the
gate reproduces offline without Ollama). Key findings:

- **moondream's labels are degenerate on this roll**: 25/37 `forest`, 36/37
  `soft`, 32/37 `warmer`, ~31/37 `bright`. So the `contrast` and `warmth` axes
  carry almost no per-frame signal here — only the `scene` axis does (modestly).
- **The big GT gap is SYSTEMATIC, not per-scene taste.** Analytical exposure is
  ~0.25 too high and black ~0.16 too negative on *almost every* frame — the
  analytical print tuner sits at the clip boundary (`PRINT_HI_CEIL` 0.99) while
  the user prints lower with lifted blacks. The AI variant re-targets the print
  tune from inside scene_tuner (analytical path untouched): `AI_HI_CEIL` 0.72
  halves the exposure delta (0.274→0.142), `AI_BLACK_LIFT` +0.10 halves the
  black delta (0.159→0.068), both clip-safe (worst ~0.23%).
- **gamma** is now keyed on `scene` (`SCENE_GAMMA`), not the degenerate
  `contrast` label: night/indoor print a higher grade, daylight/snow lower
  (median delta 0.10→0.075). `CONTRAST_GAMMA` and `WARMTH_SHIFT` are kept but
  **disabled** (zeroed) — any nonzero value hurt more frames than it helped here;
  re-enable on a roll where those labels discriminate.
- **wb** is left to the analytical region-cast finder; the warmth label ran
  counter-intuitive to the GT cast, so it does not perturb wb. The residual
  (wb_low B ~0.16, wb_high R ~0.08) is partly taste and needs a better signal.

**The strict per-frame GT gate (`check_ground_truth`) stays RED by design** — a
discrete-label LLM cannot place 18 same-labelled frames within ±0.10 of 18
different gamma targets (CLAUDE.md: irreducible taste). The new HARD gate
`check_ai_variant` instead guards that the AI variant **never regresses** any
param's aggregate median vs analytical and stays clip-safe (currently green).

## Label-enrichment experiment (2026-06-14)

Tested the user's idea — **add more categories/labels** (season, time-of-day,
weather, free-text scene + numeric contrast/brightness/warmth) — re-querying all
37 GT-roll inverted previews. Two firm conclusions:

1. **Model capability is the bottleneck, NOT the vocabulary.** `gemma3:12b` with
   the enriched prompt produces ACCURATE, varied labels (autumn/overcast forest,
   winter/snow street, indoor museum, fog→soft, golden reeds at dusk).
   `moondream:1b` given the SAME prompt HALLUCINATES — it tagged ~all 37 frames
   `winter / foggy / dusk / "snowy street"` (incl. the museum and an indoor
   desk), numeric fields pinned at stock values (brightness=8 for ALL). Its
   enriched output has ZERO GT correlation (contrast↔gamma ρ 0.25,
   brightness↔exposure ρ 0.00). Richer categories only pay off with a capable
   model; on moondream they are noise — and MORE fields made it WORSE (its
   original 4-field scene label was marginally useful only because it caught the
   obvious snow/museum). **Do not re-attempt category enrichment on moondream.**
2. **Even gemma3:12b's accurate labels do not move the gate.** The model's
   NUMERIC self-ratings are stock values (contrast=5 on 30/37) — using them HURTS
   gamma (0.075→0.15). The CATEGORICAL `weather` label discriminates the GT
   EXTREMES (fog→gamma 4.85, indoor→6.90 / exposure 0.80, snow→exposure 1.74),
   but the scene-gamma table already captures those; the residual is the spread
   WITHIN ordinary overcast/clear scenes (gamma 4.85–7.8), which a 12B model
   seeing the scene correctly STILL cannot predict. This **empirically confirms
   the residual is irreducible per-frame darkroom taste, not a labeling gap.**

Decision (user): keep `moondream` as the live opt-in labeler (fast; the rich
categories can't be used live anyway), and pursue the residual as a learned model
(below). The accurate gemma3:12b labels for this roll are saved as
`tests/fixtures/scene_labels_gemma3_12b_enriched.json` — offline FEATURES for it.

## Phase-2 plan: learned per-frame taste-residual

Learn the user's darkroom "hand" — the per-frame residual between the analytical
params and the GT picks (gamma / exposure / black; wb later) — which the LLM and
pixel math cannot reach. Key design choices:

- **Target = residual** `GT_param − analytical_param` per param (centered near 0;
  predict it, ADD to analytical → a third "learned" variant; falls back to
  analytical when uncertain). NOT absolute params.
- **Features (computed OFFLINE, may use slow/strong models):**
  (a) gemma3:12b CATEGORICAL labels — weather / season / time_of_day / scene-type
  (one-hot); the numeric LLM fields are excluded (stock values).
  (b) analytical image metrics — log-luma std / percentile spread, mean luma,
  color cast, Dmin, the analytical params themselves.
- **Model:** small + regularized + interpretable — ridge or gradient-boosted
  trees (sklearn), NOT a neural net (~110 frames after 2 more rolls is far too
  few). Heavy regularization; bounded output.
- **Validation:** leave-one-ROLL-out (frames within a roll correlate — random CV
  would leak). Promote only if it beats analytical on HELD-OUT rolls.
- **Deployment without the slow LLM, 3 options to compare:** (b-only) analytical
  features only — the honest baseline; if image metrics alone capture ~nothing
  (likely — CLAUDE.md found ~0 corr of gamma with contrast metrics) that itself
  proves semantic labels are required; (a) accept the slow LLM for label
  features; (c) distill — strong LLM labels offline, train a fast image→label
  model, deploy that.
- **Guardrails:** re-apply the hard-clip guard (as the AI variant does); bound
  the residual; selectable in the debug UI alongside analytical / AI.

**Milestones:** (1) after 2 more annotated rolls assemble the feature/target
table (this roll's gemma3:12b labels already saved); (2) analytical-only ridge,
leave-one-roll-out → establishes the floor; (3) add gemma3:12b semantic labels,
measure lift on held-out rolls; (4) if real, wire a third variant + a holdout
gate and decide live deployment (slow LLM vs distilled labeler). RL / from-scratch
nets stay infeasible at this data volume.