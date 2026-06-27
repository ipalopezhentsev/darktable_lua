# Spec 06 — AI as a perceptual CRITIC (directional corrections)

## Motivation

Spec 03 added a vision-LLM that **predicted** per-scene params (intent → gamma/
wb/brightness). It failed (user, 2026-06-27: "worked like shit"), for reasons we
now understand precisely:

1. **It asked the model for intent it cannot know.** "Why gamma 7.8 here, 4.85
   there?" is darkroom taste; even gemma3:12b labeling scenes accurately could
   not predict the within-scene spread (spec 03 label-enrichment experiment).
2. **It was judged by the wrong ruler.** scene_tuner was calibrated AND gated on
   PARAMETER deltas. Spec 04 then established the param proxy is non-unique
   (exposure↔wb↔black↔gamma interdependent) — the ground truth is the rendered
   PICTURE (its histogram). The AI's flagship move (`AI_HI_CEIL` 0.72 → darker)
   even pointed the WRONG way: spec 04's histogram analysis found the algorithm
   rendered too DARK, not too bright, and fixed it analytically (PRINT_GAMMA
   6.1→5.0, PRINT_HI_CEIL 0.99→0.97). So the AI had nothing real left to add.

Meanwhile global-constant calibration (spec 05; CMA-ES / coordinate descent) is
at its floor — median histogram EMD-to-GT ≈ 0.0375 (roll 2506-1), ~0.052–0.065
(harder roll 2510-11-1) — because one constant set cannot give same-looking
frames different per-frame looks. The remaining gap is per-FRAME.

## The reframe (user, 2026-06-27)

Stop predicting. Make the model a **critic in a feedback loop**:

- Show it the analytical render, **cropped to content** (no holder/rebate
  garbage to distract it).
- It judges what it SEES against a shared prior of "natural" — *daylight
  shouldn't look magenta; a night street can't be flat*. That prior is world
  knowledge the model already has, NOT the user's private taste. **That is why
  this can work where intent-prediction could not.**
- It emits a **DIRECTION to move the picture**, never a param value: "lift
  midtones, tame highlights", "highlights look magenta → push yellower".
- A **deterministic solver** turns each direction into the param move that
  achieves it and owns the param ambiguity + clip safety. The model never
  supplies a number.

Labor split: **AI says WHICH WAY the picture is wrong; analytics figures out HOW
to fix it safely.** Param ambiguity stops mattering because we never ask the AI
for params. "Natural" = the user's GT (the user tunes for natural), so the metric
stays histogram EMD-to-GT — the critic is just a new MECHANISM to drive it down.

This is an OPT-IN alternate variant; the analytical pipeline is untouched and
runs first (same contract as spec 03).

## Architecture

### Directions schema (categorical; the model emits NO numbers)

```
{
  "looks_natural": bool,
  "midtones":    "darker" | "ok" | "lighter",   # -> paper black (brighten combo)
  "contrast":    "flatter" | "ok" | "punchier", # -> paper gamma (grade)
  "highlights":  "tame" | "ok" | "lift",        # -> exposure (highlight target)
  "hi_cast":     "ok" | "warmer" | "cooler" | "greener" | "magentaer",  # -> wb_high
  "shadow_cast": "ok" | "warmer" | "cooler" | "greener" | "magentaer",  # -> wb_low
  "rationale":   "<short>"
}
```

### The interpreter — `critic_corrections.py` (built, deterministic)

`apply_corrections(params, directions, lin, border, cfg, scale)` maps each
non-"ok" direction to a bounded move on the one lever that produces that
histogram change, reusing production solver patterns, then runs the production
hard-clip guard. Pure; the model never sees params.

| direction | lever | how (reused machinery) |
|---|---|---|
| midtones lighter/darker | paper BLACK, holding the highlight level | the debug-UI brighten/darken combo: black moves midtones, exposure re-pins P99.9 → clip-neutral |
| contrast punchier/flatter | paper GAMMA, then re-pin highlight | pure grade move |
| highlights tame/lift | EXPOSURE to a new highlight target | bisect (`_exposure_for_high`) |
| hi_cast / shadow_cast | nudge wb_high / wb_low in channel space, renormalized | warm=+R/−B, cool=−R/+B, green=+G, magenta=−G |

`scale` multiplies every unit step, so a loop/search can take smaller moves.
Every path ends in the clip guard (back exposure off until clip ≤
`PRINT_CLIP_BUDGET`) — the binding constraint stays enforced.

**Dmin is NOT a lever.** `gt_params_for_frame` keeps Dmin = production (Dmin is
not annotated), so the GT render and the analytical render share the same Dmin —
it is not part of the ana-vs-GT gap. A Dmin lever could only "help" by cheating
the histogram toward a target that does not actually use that Dmin.

## Oracle validation (de-risk BEFORE any LLM) — `tests/calibrate_ai_critic.py`

The cheap go/no-go, fully offline (no Ollama, no labels; needs local TIFFs):
derive the "perfect" directions from the GT itself (sign of each picture-stat gap
between the analytical render and the GT render), drive the interpreter greedily
(accept a step only if EMD-to-GT drops; shrink on rejection), and measure median
EMD-to-GT before vs after. This is the OPTIMISTIC ceiling — the oracle knows GT;
the LLM will do worse. If even the ceiling is low, the lever set cannot represent
GT and we abandon before wiring any model.

Picture stats driving the oracle directions: midtone luma (median), contrast
(P90−P10), highlight luma (P99.9), and the chroma of two ADJACENT broad tonal
bands covering the whole frame (lower third → `wb_low`, the rest minus top-0.5%
speculars → `wb_high`).

### Result (uses roll 2506-1's curated `correct-inversion` GT)

| stage | 2506-1 | 2510-11-1 | mean-of-medians |
|---|---|---|---|
| global-constant floor (spec 05) | 0.0375 | 0.0653 | 0.0514 |
| oracle, narrow tail bands | 0.0191 | 0.0376 | 0.0283 |
| oracle, broad bands | 0.0123 | 0.0243 | 0.0183 |
| **oracle, gap-free adjacent bands** | **0.0104 (−72%)** | **0.0231 (−65%)** | **0.0167** |

35/38 and 37/38 frames improved. **GO**: given correct directions, the levers
recover ~two-thirds of the gap the global floor cannot touch — the architecture
is sound. Two measurement lessons mattered: (1) cast must be read over BROAD
bands, not narrow tails, or a midtone/overall cast never triggers a wb move;
(2) the bands must be ADJACENT with no hole, or DARK frames (mass in the gap) sit
at 0 steps with a large residual.

### Open-loop proxy — does it work WITHOUT the GT crutch?

The greedy oracle knows GT (accept-if-EMD-drops). The realistic LLM-in-a-loop has
NO EMD feedback — it re-critiques and steps in the directions. To bracket it,
`correct_frame_openloop` applies the (correct, GT-derived) signs with a step that
DECAYS every iteration (annealing → bounded movement, provable convergence),
re-critiquing each pass, with NO accept/reject. (A shrink-only-on-reversal line
search FAILS: a lever whose sign never flips marches unbounded — gamma pinned at
8.0 — so it scored −17%. Annealing is the fix.)

| roll | analytical | oracle (feedback) | open-loop (signs only) |
|---|---|---|---|
| 2506-1 | 0.0375 | 0.0104 (+72%) | 0.0202 (**+46%**) |
| 2510-11-1 | 0.0653 | 0.0231 (+65%) | 0.0395 (**+40%**) |
| 2511-12-1 | 0.0543 | 0.0206 (+62%) | 0.0290 (**+47%**) |

**Key finding: correct directional signs ALONE — no quality measure — recover
~two-thirds of the oracle ceiling (+40–47% over analytical), stably (≈29/37
frames improved).** So the deterministic annealing loop turns the LLM's job into
the EASY perceptual call ("too magenta / too dark / too flat?"); its richer
"better/worse/natural" judgment only adds on top, toward the oracle ceiling. The
realistic LLM bracket is therefore ~[+40%, +65%] IF its signs match the
GT-derived ones — and that sign accuracy is the ONE remaining unknown, only
resolvable by wiring the model. Offline de-risking is complete and POSITIVE.

### Irreducible residual (a handful of frames, low ROI to chase)

- DSC_0010 (2510): `cor_lum` ~0.12 after 4 steps — luma grossly off, the tonal
  levers physically cannot reach it (GT outlier / forward-model range). Same
  irreducible tail identified across spec 03/04.
- DSC_0024 (2510): 0 steps yet a real histogram-SHAPE gap the 3 luma summary
  anchors don't see. Would need finer levers (offset/soft_clip) or finer
  guidance.

Vocabulary judged SUFFICIENT here; stop widening.

## LLM critic — TESTED, the model is the bottleneck (`tests/critic_llm_eval.py`)

Wired the real critic: cropped color-managed sRGB preview → local vision LLM emits
the directions JSON → the annealing loop (re-critiqued each pass) → EMD vs oracle
vs analytical, on roll 2506-1. Two models × two prompt regimes:

| model | invite-critique prompt (v1) | default-ok prompt (v2) |
|---|---|---|
| moondream | constant "make it pop" on ALL 38, **−58%** | all-natural no-op, **+0%** |
| gemma3:12b | biased over-correction, **−83%** | all-natural no-op, **+0%** |

**The models do NOT perceive the per-frame errors — the prompt's default sets the
output.** moondream emitted a BYTE-IDENTICAL verdict (`lighter/punchier/tame`) for
all 152 v1 renders (zero image dependence); both flip to "everything natural" the
moment the prompt leans that way (v2 → q=1, looks_natural=true on every frame).
gemma showed slightly more image dependence than moondream but its casts were
systematically biased (cooler 146/6) and DISAGREED with moondream's (warmer) —
unreliable perception, not signal. So there is no prompt operating point where the
model correctly says "this frame is fine, that one is too magenta": it either
over-corrects everything (harm) or accepts everything (no-op).

**Conclusion: the architecture is validated and ready, but no available LOCAL VLM
(moondream, gemma3:12b) perceives these inverted-film residuals well enough to
drive it.** The residuals are real (oracle +72%) and a human catches them; the
models are below that bar. This is the same capability ceiling spec 03 hit, now
confirmed for the easier critic framing too.

## Status

- [x] Interpreter built (`critic_corrections.py`).
- [x] Oracle validation passed (`tests/calibrate_ai_critic.py`). GO (+62–72%).
- [x] Open-loop proxy — annealed signs, no feedback: +40–47%, stable. Correct
      signs alone suffice; the loop handles magnitude.
- [x] LLM critic wired + tested (`tests/critic_llm_eval.py`). **Negative:** both
      local VLMs fail to perceive (−83%…+0%). Model is the bottleneck.
- [ ] SHELVED pending a better perceiver. Re-open when one of:
      - a stronger local VLM (gemma3:27b on this box, or a future model) clears
        the perception bar on a balanced prompt;
      - a **comparative** critic (show the model A/B renders, "which looks more
        natural?" — VLMs are better at A/B than absolute judgment) is tried;
      - a cloud VLM is acceptable for an offline pass.
- [x] FALLBACK (works today): the directions map 1:1 onto the existing debug-UI
      manual controls (brighten/darken `]`/`[` = midtones, gamma slider =
      contrast, exposure = highlights, the two wheels = casts). The user applies
      them by eye — the validated interpreter is exactly what the UI already does.
