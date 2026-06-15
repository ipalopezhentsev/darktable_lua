Currently the negative inversion feature works and was tuned by you via my manual annotations here: auto_negadoctor\tests\fixtures\annotations\2026-06-13_wb_print_roll
But it still cannot reach my manual corrections.
You tried many times but hit some wall.

And I think I start understanding why.
The output params that we seek via algorithm and to which I make my manual annotations correcting the errors are essentially two groups:
1) exposure
2) white balance for shadows/highlights. 

WB is not just about hue, it also encodes 'power' of this color. changing wb changes exposure and necessitates nudging exposure afterwards. Or vice versa I can make exposure darker but then compensate it with choosing lighter colors for WB patches.

And here comes the wall you hit I think. So you have my manual params and try to reach them via your algorithm. But due to the aforementioned duality, there can be inconsistencies to my edits, I mean the picture itself that I obtain via my edits is good and ground truth, but not particular combinations of exposure+wb because as we've seen they are interrelated.

That got me thinking how to express my ground truth in 'independent' way not allowing for inconsistencies.
And I think I have it: my params are not golden truth per se, but the inverted PICTURE that they produce IS.
In getting this picture I strive mostly for:
- middtones being sufficiently bright
- but not at the expense of clipping
- inverted picture having natural color balance in shadows/highlights

And what does characterize all of that params? HISTOGRAM!
That is the param-INVARIANT source of ground truth!

So why don't you tune the algo not for blindly matching my PARAMS, but for matching the HISTOGRAM that is obtained by APPLYING my params? I.e. your goal should be to bring respective histograms closer to my ground truth.

I've now have several folders of human annotations:
- auto_negadoctor\tests\fixtures\rolls\2510-11-1
- auto_negadoctor\tests\fixtures\rolls\2512-2601-1\2026-06-13_wb_print_roll
- auto_negadoctor\tests\fixtures\rolls\2511-12-1 

These are ground truths. You need to tune algorithm/hardcoded params so that they lead to histograms which are as close to histograms produced by processing with my ground truth annotations.

---

## Implementation (2026-06-14)

Ground-truth rolls used (all auto-discovered by `run_quality_tests.discover_rolls`):
`2506-1`, `2510-11-1`, `2511-12-1`, and `2512-2601-1/2026-06-13_wb_print_roll`
(2511-12-1 and 2506-1 were added during this work; 148 wb/print GT frames total).

**Histogram = the loss.** `nega_model.histogram_distance(a, b)` computes the
per-channel 1D Wasserstein/EMD between two rendered sRGB outputs over the content
crop, decomposed into a **luma** term (brightness/clip — goals 1+2) and a
**color** term (chroma cast beyond the common brightness shift — goal 3, natural
shadow/highlight balance), plus a signed-luma direction, **top-highlight
percentile deltas (hi999/hi9999) + a highlight-color term**, and a near-white
mass. Pure math, unit-tested in `test_forward_model.py::test_histogram_distance`.

**RESOLUTION (2nd pass, user feedback "you lack resolution / especially top
highlights matter").** The first pass compared the **8-bit** sRGB render at 64
bins — capped at 256 levels, structurally BLIND to the highlight shoulder. The
metric now compares the **FLOAT render** (before display quantization) at
**14-bit bins (16384)** with a **denser sample** (`HIST_SUBSAMPLE_FRAC` 0.001 vs
the tuner's 0.003), so it resolves fine highlight structure (a unit test proves
it sees a sub-8-bit shift the 8-bit metric misses). `check_histogram_match` also
reports + guards the top-highlight percentile gap (`hi999`).

**GT render reconstruction.** `run_quality_tests.gt_params_for_frame` copies the
production params and overrides ONLY the annotated fields (wb_low/wb_high/black/
exposure/gamma); annotations are partial, so missing fields stay = production
(Dmin/D_max/offset/soft_clip are never annotated). This renders the picture the
user actually saw/approved.

**Instrument.** `tests/calibrate_histogram_match.py` (offline, no gate, no Ollama)
prints, per GT frame and aggregate, the prod-vs-GT histogram EMD + a verdict and a
`PRINT_HI_CEIL` sweep. `--write-baseline` writes each roll's committed
`histogram_baseline.json`.

**Finding 1 — gamma (1st pass, vindicates the spec's thesis).** The histogram
showed the algorithm rendered too DARK — the OPPOSITE of the earlier param-based
AI calibration, which compared param *numbers* and pushed exposure DOWN (the
proxy trap this spec warns about). The lever was **gamma**: the steep
`PRINT_GAMMA=6.1` (the GT *param* median) crushed midtones below the near-clip
highlight pin. `PRINT_GAMMA` 6.1→**5.0** (joint EMD minimum; 5.0–5.75 are within
run-to-run noise, 5.0 best serves the worst-case dark roll 2510-11-1). The user
reaches the same picture via a different param combo (low exposure + positive
black + high gamma), so the param gate's gamma delta gets WORSE by design.

**Finding 2 — highlight ceiling (2nd pass, after the resolution upgrade).** At
8-bit the ceiling looked flat/unhelpful; the high-res highlight-aware metric
revealed it is the DOMINANT lever for the top end. The tuner pinned P99.9 to 0.99,
over-pushing the very top toward white where the GT sits LOWER. `PRINT_HI_CEIL`
0.99→**0.97** cut the top-highlight gap (median |ΔP99.9|) ~33% (0.0162→0.0108)
and total EMD ~16% (0.0599→0.0505) across all 4 rolls, centred overall brightness
(signed luma −0.029→−0.002), clip-safe (<0.3%). `soft_clip` is NOT a lever — it
cancels between the algorithm and GT renders (which share it, GT never annotates
it). Both findings are clip-safe (`check_no_clipping` green) and leave the AI gate
green. Re-derive with `calibrate_histogram_match.py` as rolls accrue.

**Guard.** `run_quality_tests.check_histogram_match` is informational + a
regression guard vs `histogram_baseline.json` (FAILs only on a median-total-EMD
regression). The residual (~0.06 total EMD) is irreducible per-frame darkroom
taste — the three rolls' individual gamma optima span ~4.5–5.9. The strict
per-frame `check_ground_truth` stays RED by design.