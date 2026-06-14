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
`2510-11-1`, `2511-12-1` (added during this work — 36 wb / 26 print GT frames),
and `2512-2601-1/2026-06-13_wb_print_roll`.

**Histogram = the loss.** `nega_model.histogram_distance(srgb_a, srgb_b)` computes
the per-channel 1D Wasserstein/EMD between two rendered sRGB outputs over the
content crop, decomposed into a **luma** term (brightness/clip — goals 1+2) and a
**color** term (chroma cast beyond the common brightness shift — goal 3, natural
shadow/highlight balance), plus a signed-luma direction and a near-white mass.
Pure math, unit-tested in `test_forward_model.py::test_histogram_distance`.

**GT render reconstruction.** `run_quality_tests.gt_params_for_frame` copies the
production params and overrides ONLY the annotated fields (wb_low/wb_high/black/
exposure/gamma); annotations are partial, so missing fields stay = production
(Dmin/D_max/offset/soft_clip are never annotated). This renders the picture the
user actually saw/approved.

**Instrument.** `tests/calibrate_histogram_match.py` (offline, no gate, no Ollama)
prints, per GT frame and aggregate, the prod-vs-GT histogram EMD + a verdict and a
`PRINT_HI_CEIL` sweep. `--write-baseline` writes each roll's committed
`histogram_baseline.json`.

**Finding (vindicates the spec's thesis).** The histogram showed the algorithm
rendered too DARK — the OPPOSITE of the earlier param-based AI calibration, which
compared param *numbers* and pushed exposure DOWN (the proxy trap this spec warns
about). The brightness lever was NOT the highlight ceiling (`PRINT_HI_CEIL`
already optimal at 0.99; `PRINT_HI_PCT` flat) but **gamma**: the steep
`PRINT_GAMMA=6.1` (the GT *param* median) crushed midtones below the near-clip
highlight pin. Lowering `PRINT_GAMMA` 6.1→**5.0** is the joint luma/total-EMD
minimum across all three rolls (110 GT frames), clip-safe (<0.3% hard clip,
`check_no_clipping` green), AI gate still green. The user reaches the same brighter
picture via a different param combo (low exposure + strongly POSITIVE black + high
gamma), so the param gate's gamma delta gets WORSE by design.

**Guard.** `run_quality_tests.check_histogram_match` is informational + a
regression guard vs `histogram_baseline.json` (FAILs only on a median-total-EMD
regression). The residual (~0.06 total EMD) is irreducible per-frame darkroom
taste — the three rolls' individual gamma optima span ~4.5–5.9. The strict
per-frame `check_ground_truth` stays RED by design.