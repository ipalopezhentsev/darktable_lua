"""Tuning configuration for auto_retouch dust/stroke/sensor detection.

The SCHEMA + per-field docs live here (JSON can't carry comments); the VALUES live
externally in `presets/*.json`. Adopting a calibration-fitted preset is then "drop in
a new preset file", never editing detect_dust.py. Built on the shared
`common/calibration/schema.TuningSchema` (same machinery auto_negadoctor uses).

Runtime selection (detect_dust.py): `RETOUCH_PRESET` env var (NAME or path),
defaulting to the bundled `default` preset, which is byte-identical to the constants
that used to be hardcoded at the top of detect_dust.py.

Grouped by calibration KIND -> pipeline sub-stage:
  dust    circular-spot ("dot") detection thresholds (judged by FP + missed-dust).
  stroke  thread / scratch / streak / radon detection (judged by missed-strokes + FP).
  sensor  multi-frame sensor-dust consensus (no calibration kind yet; here so its
          constants live in a preset too).

Structural/format constants (brush/mask byte layout, the blendop template, mask
versions, ML feature-name lists) are NOT here — they are correctness facts, not taste
knobs — so they stay inline in detect_dust.py.
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.calibration.schema import TuningSchema, _F  # noqa: E402

# name -> _F(doc, kind in {float,int,tuple,bool}); order = the GROUPS traversal.
FIELDS = collections.OrderedDict([
    # ===================== DUST =====================
    # --- threshold ---
    ("LOCAL_BG_KERNEL_FRAC", _F("Size of the Gaussian blur used to estimate the local background that is "
        "subtracted from the image before spot detection. Larger values smooth over bigger structures and "
        "isolate finer dust. Unit: fraction of min_dim (rounded to an odd pixel count at use).")),
    ("NOISE_THRESHOLD_MULTIPLIER", _F("A pixel must rise this many robust standard deviations (MAD-based, "
        "so it ignores the dust outliers themselves) above the local background to become a spot candidate. "
        "Higher = fewer, stronger candidates. Unit: multiplier of the noise estimate.")),
    ("MIN_ABSOLUTE_THRESHOLD", _F("Absolute floor on the above-background brightness a candidate needs, "
        "regardless of the per-frame noise estimate; stops very clean frames from flagging faint grain. "
        "Unit: gray levels (0-255).")),
    ("REJECT_LOG_CONTRAST_MIN", _F("Minimum contrast a REJECTED candidate needs before it is written to the "
        "debug reject list. Purely cosmetic (limits log volume); does not affect detection. Unit: gray levels.", kind="int")),
    # --- size_shape ---
    ("MIN_SPOT_AREA_FRAC", _F("Smallest connected-component area accepted as dust. Rejects subpixel / "
        "imperceptible specks. Unit: fraction of min_dim**2 (an area fraction).")),
    ("MAX_SPOT_AREA_FRAC", _F("Largest area accepted as a dust dot; anything bigger is treated as image "
        "content, not dust (elongated marks go to the stroke path instead). Unit: fraction of min_dim**2.")),
    ("MIN_ASPECT_RATIO", _F("Minimum bounding-box short/long side ratio. Rejects strongly elongated blobs "
        "(fibers/threads are handled separately as strokes). Unit: ratio 0-1.")),
    ("MIN_COMPACTNESS", _F("Minimum area / bounding-box-area. Rejects sparse, ragged shapes a compact dust "
        "dot would never form. Unit: ratio 0-1.")),
    ("MIN_SOLIDITY", _F("Minimum area / convex-hull-area. Rejects non-convex shapes such as letters or twigs. "
        "Unit: ratio 0-1.")),
    ("MIN_CIRCULARITY", _F("Minimum 4*pi*area / perimeter**2 (1.0 is a perfect circle). Rejects spiky / "
        "complex outlines like symbols and text. Unit: ratio 0-1.")),
    ("DOT_MIN_CIRCLE_FILL", _F("Minimum area / (pi*enc_r**2) — how completely the spot fills its minimum "
        "enclosing circle. Below this the spot is treated as non-compact. Unit: ratio 0-1.")),
    ("DOT_IRREGULAR_RADIUS_FRAC", _F("The DOT_MIN_CIRCLE_FILL non-compact rejection is applied ONLY to spots "
        "whose enclosing-circle radius exceeds this size; smaller spots are exempt. Unit: fraction of min_dim.")),
    ("SHAPE_CHECK_MIN_AREA_FRAC", _F("Solidity/circularity checks run only for spots larger than this area; "
        "tiny spots have too few pixels for reliable shape metrics. Unit: fraction of min_dim**2.")),
    # --- texture_contrast ---
    ("TEXTURE_KERNEL_FRAC", _F("Neighborhood size of the local-texture (rolling standard-deviation) map used "
        "to tell smooth, dust-friendly areas from busy image content. Unit: fraction of min_dim (odd pixel "
        "count at use).")),
    ("MAX_LOCAL_TEXTURE_SMALL", _F("Maximum local texture allowed in the ring around a TINY spot; small specks "
        "may sit in slightly busier surroundings. The limit tightens toward MAX_LOCAL_TEXTURE_LARGE as area "
        "grows. Unit: gray-level standard deviation.")),
    ("MAX_LOCAL_TEXTURE_LARGE", _F("Maximum local texture allowed around a LARGE spot (area >= "
        "TEXTURE_TIER_AREA_FRAC); big blobs must sit on genuinely smooth backgrounds to be believed as dust. "
        "Unit: gray-level standard deviation.")),
    ("MAX_DARK_BG_TEXTURE", _F("Separate, tighter texture limit for the ring around a spot on DARK backgrounds, "
        "where grain reads as texture differently. Unit: gray-level standard deviation.")),
    ("MIN_CONTRAST_TEXTURE_RATIO", _F("Minimum contrast / local-texture. A real dust dot stands out far above "
        "its surrounding grain, so a low ratio means it is hidden in texture. Unit: ratio.")),
    ("MAX_BG_GRADIENT_RATIO", _F("Maximum background-gradient / contrast. A high value means the 'spot' is "
        "really the halo/edge of a brightness ramp (from the background blur), not dust. Unit: ratio.")),
    ("MAX_CONTEXT_TEXTURE", _F("Maximum median texture measured over a WIDE region (radius = "
        "CONTEXT_TEXTURE_RADIUS_FRAC) around the spot. Catches false positives that sit on a locally-smooth "
        "patch inside otherwise busy content. Unit: gray-level standard deviation.")),
    ("CONTEXT_TEXTURE_RADIUS_FRAC", _F("Radius over which MAX_CONTEXT_TEXTURE samples the surrounding "
        "busy-ness. Unit: fraction of min_dim.")),
    ("TEXTURE_TIER_AREA_FRAC", _F("Spot area at which the texture limit finishes interpolating from "
        "MAX_LOCAL_TEXTURE_SMALL down to MAX_LOCAL_TEXTURE_LARGE. Unit: fraction of min_dim**2.")),
    ("LARGE_SPOT_AREA_THRESHOLD_FRAC", _F("Spots larger than this must clear the higher LARGE_SPOT_MIN_CONTRAST "
        "bar; big, pale blobs need strong contrast to be accepted. Unit: fraction of min_dim**2.")),
    ("LARGE_SPOT_MIN_CONTRAST", _F("Minimum contrast a large spot (see LARGE_SPOT_AREA_THRESHOLD_FRAC) must "
        "have; rejects big, pale, foggy blobs. Unit: gray levels.", kind="int")),
    # --- color ---
    ("MAX_EXCESS_SATURATION", _F("Maximum (spot saturation - surrounding saturation). Dust is achromatic and "
        "picks up the local color cast, so it should not be MORE saturated than its surroundings. "
        "Unit: saturation 0-255.", kind="int")),
    ("MAX_SPOT_SATURATION", _F("Lower bound of the compound color check: a spot both above this saturation AND "
        "with positive excess saturation is rejected as a colored feature. Unit: saturation 0-255.", kind="int")),
    ("EMULSION_EXCESS_SAT_THRESHOLD", _F("When a spot is very saturated (above MAX_SPOT_SATURATION), an excess "
        "saturation above this marks it as a colored emulsion artifact rather than dust. "
        "Unit: saturation 0-255.", kind="int")),
    # --- voting ---
    ("SOFT_CONTEXT_VOTE_THRESHOLD", _F("Context texture below this casts one 'is dust' soft vote (clear skies "
        "and plain walls score low). Unit: gray-level standard deviation.")),
    ("SOFT_TEXTURE_VOTE_THRESHOLD", _F("Local (ring) texture below this casts one 'is dust' soft vote. "
        "Unit: gray-level standard deviation.")),
    ("SOFT_RATIO_VOTE_THRESHOLD", _F("Contrast/texture ratio above this casts one 'is dust' soft vote (roughly "
        "1.5x the hard MIN_CONTRAST_TEXTURE_RATIO). Unit: ratio.")),
    ("MIN_DUST_VOTES", _F("How many of the 3 soft votes (context, texture, ratio) a borderline spot must earn "
        "to be accepted. Unit: count (0-3).", kind="int")),
    # --- brightness ---
    ("MIN_BRIGHTNESS_FRAC_SMALL", _F("Brightness floor for the SMALLEST spots, relative to the frame's bright "
        "reference (its 95th-percentile brightness). Interpolated up to MIN_BRIGHTNESS_FRAC_LARGE as area "
        "grows. Unit: fraction of the bright reference.")),
    ("MIN_BRIGHTNESS_FRAC_LARGE", _F("Brightness floor for LARGE spots, relative to the frame's bright "
        "reference; larger dust must be brighter to be accepted. Unit: fraction of the bright reference.")),
    ("MIN_LOCAL_BG_FRACTION", _F("The local background under a spot must be at least this fraction of the "
        "frame's 95th-percentile brightness; rejects 'dust' in dark regions where it is implausible. "
        "Unit: fraction.")),
    ("MIN_SURROUND_BG_RATIO", _F("The immediate surround must be at least this fraction of the local "
        "background; rejects spots embedded in a locally darker patch (an image feature, not dust). "
        "Unit: ratio 0-1.")),
    # --- isolation ---
    ("ISOLATION_RADIUS_FRAC", _F("Radius within which already-accepted spots are counted, to reject dust-like "
        "CLUSTERS that are really texture. Unit: fraction of min_dim.")),
    ("MAX_NEARBY_ACCEPTED", _F("Reject a spot if more than this many already-accepted spots lie within "
        "ISOLATION_RADIUS_FRAC; lone dust survives, dense clusters are dropped. Unit: count.", kind="int")),
    ("MAX_SPOTS", _F("Hard cap on spots kept per frame (highest-contrast first). darktable allows at most 300 "
        "mask forms, so this must stay below that. Unit: count.", kind="int")),
    # --- brush_source ---
    ("ENC_RADIUS_SCALE", _F("Healing brush radius = the spot's minimum-enclosing-circle radius x this. Kept "
        ">1 so darktable's brush fully covers the dust. Unit: multiplier.", kind="int")),
    ("SOURCE_SEARCH_INNER_FACTOR", _F("Inner radius of the healing-source search ring = spot radius x this; "
        "keeps the clone source from overlapping the dust itself. Unit: multiplier.")),
    ("SOURCE_SEARCH_MAX_RADIUS_FRAC", _F("Outer cap on how far the healing-source search looks for clean "
        "background. Unit: fraction of min_dim.")),
    ("SOURCE_SEARCH_MIN_RADIUS_FRAC", _F("Minimum outer search radius, so even tiny spots search a usable "
        "area for a healing source. Unit: fraction of min_dim.")),
    ("SOURCE_GRID_STEP_FRAC", _F("Spacing of the grid of candidate healing-source positions. Unit: fraction "
        "of min_dim (floored to at least one sample). NOTE: reserved — not currently read by the detector.")),
    # --- ml ---
    ("ML_RECOVERY_THRESHOLD_MULT", _F("For the optional ML recovery pass, multiply the base detection "
        "threshold by this (< 1) to surface extra missed-dust candidates for the classifier to judge. "
        "Unit: multiplier.")),
    ("ML_POSTFILTER_THRESHOLD", _F("Minimum classifier probability for a normally-detected spot to be kept "
        "when the ML model is active. Unit: probability 0-1.")),
    ("ML_RECOVERY_THRESHOLD", _F("Higher classifier probability required to accept a spot that ONLY the "
        "recovery pass found. Unit: probability 0-1.")),
    # ===================== STROKE =====================
    # --- detect ---
    ("DETECT_STROKES", _F("Master on/off for thread / scratch (elongated) dust detection. On by default. "
        "Unit: boolean.", kind="bool")),
    # --- geometry ---
    ("STROKE_MIN_LENGTH_FRAC", _F("Minimum centerline length of a stroke; shorter marks are ignored or treated "
        "as dots. Unit: fraction of min_dim.")),
    ("STROKE_MIN_ELONGATION", _F("Minimum length/width; separates true strokes (long and thin) from compact "
        "blobs. Unit: ratio.")),
    ("STROKE_MAX_WIDTH_FRAC", _F("Maximum stroke width; wider marks are not thread/scratch dust. Unit: fraction "
        "of min_dim.")),
    ("STROKE_MIN_WIDTH_FRAC", _F("Floor on the measured stroke width, to avoid divide-by-zero on hairline "
        "single-pixel ridges. Unit: fraction of min_dim.")),
    ("STROKE_DP_EPS_FRAC", _F("Douglas-Peucker tolerance for simplifying the stroke centerline into a few "
        "nodes; larger = coarser path. Unit: fraction of min_dim.")),
    ("STROKE_MAX_FILL_RATIO", _F("Pre-filter on area / bounding-box-area: a genuine thin thread fills only a "
        "small part of its bounding box, so a high fill ratio means a blob, not a stroke. Unit: ratio 0-1.")),
    ("STROKE_PREFER_RATIO", _F("Convert a stroke back into a circular dot when a healing circle would cover "
        "more area than the stroke brush by more than this factor (a near-round mark heals better as a dot). "
        "Unit: ratio.")),
    ("STROKE_MAX_KEYPOINTS", _F("Cap on the number of centerline nodes stored per stroke (darktable brushes "
        "tolerate many). Unit: count.", kind="int")),
    # --- brush ---
    ("STROKE_COVERAGE_FRAC", _F("The brush edge is placed where the brightness difference falls to this "
        "fraction of the stroke's local peak; smaller = wider brush. Unit: fraction of the peak.")),
    ("STROKE_COVERAGE_MIN_DIFF", _F("...but the edge is never taken below this absolute brightness difference, "
        "so the brush does not chase noise out into the background. Unit: gray levels.")),
    ("STROKE_COVERAGE_PCTL", _F("Which percentile of the per-sample half-widths becomes the brush half-width; a "
        "thread can be locally fatter, so a high percentile covers the wide parts. Unit: percentile 0-100.", kind="int")),
    ("STROKE_COVERAGE_MARGIN_FRAC", _F("Extra margin added outside the measured edge so the feathered brush "
        "boundary fully covers the stroke. Unit: fraction of min_dim.")),
    ("STROKE_BORDER_SCALE", _F("If the brightness profile yields no usable edge, fall back to this multiple of "
        "the core half-width for the brush border. Unit: multiplier.")),
    ("STROKE_MIN_BORDER_FRAC", _F("Minimum per-node brush border (darktable effectiveness floor). Unit: "
        "fraction of min_dim.")),
    ("STROKE_MAX_BORDER_FRAC", _F("Upper cap on the brush border to prevent runaway-wide brushes. Unit: "
        "fraction of min_dim.")),
    # --- ridge_pass ---
    ("STROKE_RIDGE_ENABLE", _F("Master on/off for the faint-scratch Hessian (sato) ridge pass. Off by default "
        "because it produces noise-floor false positives; add faint scratches by hand in the debug UI instead. "
        "Unit: boolean.", kind="bool")),
    ("STROKE_RIDGE_SIGMA_FRACS", _F("The ridge-filter scales, from thin to thicker scratches/threads, each as "
        "a fraction of min_dim. Unit: tuple of fractions of min_dim.", kind="tuple")),
    ("STROKE_RIDGE_Z", _F("Keep ridge responses above median + this x MAD (a robust z-score); higher = only "
        "stronger ridges survive. Unit: robust z-score multiplier.")),
    ("STROKE_RIDGE_MIN_CONTRAST", _F("Minimum mean brightness difference (gray minus local background) along a "
        "ridge for it to be accepted. Unit: gray levels.")),
    ("STROKE_RIDGE_DOWNSCALE", _F("Run the ridge filter at this fraction of full resolution for speed (full-res "
        "is too slow); the resulting mask is scaled back up. Unit: fraction (0-1).")),
    ("STROKE_RIDGE_MAX_CANDIDATES", _F("Cap on the number of ridge components processed (largest first) to "
        "bound cost. Unit: count.", kind="int")),
    # --- gating ---
    ("STROKE_MAX_BAND_TEXTURE", _F("Maximum median texture in the narrow band along the stroke centerline; a "
        "stroke lying on grainy content is rejected. Unit: gray-level standard deviation.")),
    ("STROKE_MAX_CONTEXT_TEXTURE", _F("Maximum median texture in a WIDE band around the stroke (not "
        "contrast-tiered); rejects strokes crossing busy scene structure. Unit: gray-level standard "
        "deviation.")),
    ("STROKE_CONTEXT_RADIUS_FRAC", _F("Outer radius of that wide context band. Unit: fraction of min_dim.")),
    ("STROKE_MAX_EXCESS_SAT", _F("Maximum (stroke saturation - surround saturation); dust and scratches are "
        "achromatic. Unit: saturation 0-255.", kind="int")),
    ("STROKE_MIN_BRIGHTNESS_FRAC", _F("On bright backgrounds only, the stroke's mean brightness must be at "
        "least this fraction of the frame's bright reference. Unit: fraction of the bright reference.")),
    ("STROKE_MIN_RIDGE_DROP", _F("The stroke crest must exceed the BRIGHTER of its two sides by at least this, "
        "proving it is a ridge and not one flank of an edge. Unit: gray levels.")),
    ("STROKE_MAX_SIDE_ASYMMETRY", _F("|left background - right background| / core brightness; a high value "
        "means a one-sided (edge) profile, not a symmetric stroke. Unit: ratio.")),
    ("STROKE_SIDE_OFFSET_FACTOR", _F("How far to sample perpendicular to the stroke to read its two sides = "
        "stroke width x this (plus a small fixed margin). Unit: multiplier of width.")),
    ("STROKE_MIN_CRISPNESS", _F("Reject strokes softer (more out-of-focus) than this; a sharp edge = real "
        "surface dust, a blurry one = an in-scene wire/branch. Unit: ratio 0-1.")),
    ("STROKE_CLIP_LEVEL", _F("A stroke crest at or above this brightness counts as clipped-white dust and "
        "skips some gating. Unit: gray level (0-255).", kind="int")),
    # --- source ---
    ("STROKE_SOURCE_OFFSET_FACTOR", _F("Minimum healing-source offset = brush radius x this, so the clone "
        "source sits about one brush-width from the defect. Unit: multiplier of brush radius.")),
    ("STROKE_SOURCE_MIN_GAP_FRAC", _F("...and at least this gap beyond BOTH brush edges, guaranteeing the "
        "source never overlaps the stroke. Unit: fraction of min_dim.")),
    # --- heal_split ---
    ("HEAL_SPLIT_BUSY", _F("Split a stroke into sub-strokes so the healing brush skips the busy (high-texture) "
        "runs it crosses. Unit: boolean.", kind="bool")),
    ("HEAL_MAX_TEXTURE", _F("Texture along the path above this marks a point as 'busy' and it is skipped. "
        "Unit: gray-level standard deviation.")),
    ("HEAL_BUSY_MARGIN_FRAC", _F("Also skip this far on either side of each busy run so the clone sources stay "
        "on clean background. Unit: fraction of min_dim.")),
    ("HEAL_SAMPLE_STEP_FRAC", _F("Spacing at which the path is resampled to test texture for splitting. Unit: "
        "fraction of min_dim.")),
    ("HEAL_MIN_SEGMENT_FRAC", _F("Keep a smooth sub-run only if it is at least this long; shorter smooth gaps "
        "are not worth a separate brush. Unit: fraction of min_dim.")),
    # --- field_isolation ---
    ("STROKE_FIELD_RADIUS_FRAC", _F("Outer radius of the annulus searched for parallel line-candidates around "
        "a stroke; many nearby lines mean scene structure (rigging, fences), not a scratch. Unit: fraction of "
        "min_dim.")),
    ("STROKE_FIELD_INNER_FRAC", _F("Inner radius of that annulus, excluding the stroke's own fragments. Unit: "
        "fraction of min_dim.")),
    ("STROKE_FIELD_MAX_LINE_CANDS", _F("Reject a stroke if this many or more line-candidates are found in its "
        "annulus. Unit: count.", kind="int")),
    ("STROKE_FIELD_CAND_MIN_AREA_FRAC", _F("A line-candidate component must be at least this large to count. "
        "Unit: fraction of min_dim**2.")),
    ("STROKE_FIELD_CAND_MIN_DIM_FRAC", _F("...and at least this long on its major axis. Unit: fraction of "
        "min_dim.")),
    ("STROKE_FIELD_CAND_MAX_FILL", _F("...and thin: area / bounding-box-area below this. Unit: ratio 0-1.")),
    ("STROKE_FIELD_CAND_MIN_ELONG", _F("...and elongated: bounding-box long/short ratio at least this. Unit: "
        "ratio.")),
    ("STROKE_FIELD_NBR_RADIUS_FRAC", _F("Radius within which OTHER accepted strokes are counted, as a second "
        "clustering guard. Unit: fraction of min_dim.")),
    ("STROKE_FIELD_MAX_NEIGHBORS", _F("Reject a stroke if more than this many accepted strokes lie within "
        "STROKE_FIELD_NBR_RADIUS_FRAC. Unit: count.", kind="int")),
    # --- streak ---
    ("STREAK_DETECT", _F("Master on/off for the faint axis-aligned (horizontal/vertical) scratch producer. "
        "Unit: boolean.", kind="bool")),
    ("STREAK_RIDGE_VGAP_FRAC", _F("How far above and below each row is compared to build a ridge (not edge) "
        "response — a true streak is brighter than both neighbors. Unit: fraction of min_dim.")),
    ("STREAK_INTEG_LEN_FRAC", _F("Length over which the response is integrated along the streak axis, boosting "
        "a faint collinear line above the noise. Unit: fraction of min_dim.")),
    ("STREAK_LEVEL_MULT", _F("Response threshold = max(6, frame noise x this); higher = only stronger streaks. "
        "Unit: multiplier of the noise estimate.")),
    ("STREAK_MIN_LEN_FRAC", _F("Minimum streak length. Unit: fraction of min_dim.")),
    ("STREAK_MAX_THICKNESS_FRAC", _F("Maximum streak component thickness; keeps it thin and near-axis. Unit: "
        "fraction of min_dim.")),
    ("STREAK_MIN_ELONG", _F("Minimum length/thickness ratio; a streak must be strongly elongated. Unit: "
        "ratio.", kind="int")),
    # --- radon ---
    ("STREAK_RADON_DETECT", _F("Master on/off for the full-width Radon scratch detector (ultra-faint, "
        "fragmented, near-horizontal lines). Unit: boolean.", kind="bool")),
    ("STREAK_RADON_MIN_RESP", _F("Minimum mean ridge response along the best-fitting line for it to be a "
        "scratch. Unit: gray levels.")),
    ("STREAK_RADON_MIN_COV", _F("Minimum fraction of the frame width along that line that actually carries "
        "response (present, not gap). Unit: fraction 0-1.")),
    ("STREAK_RADON_MAX_SLOPE", _F("Search line slopes in [-this, +this] (about +/-1.1 degrees) — 'almost' "
        "axis-aligned. Unit: slope (rise/run).")),
    ("STREAK_RADON_SLOPES", _F("Number of discrete slope steps tested across that range. Unit: count.", kind="int")),
    ("STREAK_RADON_PRESENT", _F("Response above this counts a given column as 'scratch present' on the line. "
        "Unit: gray levels.")),
    ("STREAK_RADON_EXT_FACTOR", _F("Endpoint-extension threshold = STREAK_RADON_PRESENT x this; a lower bar "
        "lets the line follow a fading scratch outward. Unit: multiplier.")),
    ("STREAK_RADON_EXT_GAP_FRAC", _F("Stop extending the line after this distance with no faint signal. Unit: "
        "fraction of min_dim.")),
    ("STREAK_RADON_MAX_HALFWIDTH_FRAC", _F("Cap on the measured brush half-width for a faint Radon scratch, so "
        "noise cannot inflate it. Unit: fraction of min_dim.")),
    ("STREAK_RADON_MIN_BRUSH_R_FRAC", _F("Minimum brush radius (half-width) for a Radon scratch, so the bold "
        "brush still clears it. Unit: fraction of min_dim.")),
    ("STREAK_RADON_BRUSH_MARGIN_FRAC", _F("Extra margin added to the measured half-width. Unit: fraction of "
        "min_dim.")),
    # --- hysteresis ---
    ("STROKE_HYST_LOW_FACTOR", _F("Tail/extension threshold = the main detection threshold x this; a lower "
        "value lets a stroke grow along its faint ends. Unit: multiplier.")),
    ("STROKE_HYST_PAD_FRAC", _F("Padding added around the seed bounding box when searching for a stroke's "
        "faint extension. Unit: fraction of min_dim.")),
    ("STROKE_HYST_PAD_LEN_FRAC", _F("...or this fraction of the seed's own length, whichever is larger (long "
        "scratches reach further). Unit: fraction of the seed length.")),
    ("STROKE_HYST_BRIDGE_FRAC", _F("For axis-aligned scratches only, bridge noise gaps up to this long ALONG "
        "the stroke axis. Unit: fraction of min_dim.")),
    # ===================== SENSOR =====================
    # --- dog ---
    ("SENSOR_SIGMA_INNER_FRAC", _F("Inner Gaussian sigma of the Difference-of-Gaussians blob detector. Unit: "
        "fraction of min(w,h).")),
    ("SENSOR_SIGMA_OUTER_FRAC", _F("Outer Gaussian sigma of the DoG — large enough to see the background "
        "around a blob. Unit: fraction of min(w,h).")),
    ("SENSOR_DOG_MIN_CONTRAST", _F("Minimum DoG peak value for a blob; low enough to catch faint sensor dust. "
        "Unit: DoG response level.")),
    ("SENSOR_MIN_RADIUS_FRAC", _F("Minimum accepted blob radius. Unit: fraction of min(w,h).")),
    ("SENSOR_MAX_BLOB_RADIUS_FRAC", _F("Maximum accepted blob radius; sensor dust is small. Unit: fraction of "
        "min(w,h).")),
    # --- consensus ---
    ("SENSOR_CLUSTER_RADIUS_NORM", _F("Radius within which per-frame candidates are clustered into one "
        "cross-frame dust location. Unit: normalized full-frame coordinate (0-1).")),
    ("SENSOR_DUST_MIN_FRAMES", _F("A cluster must appear in at least this many frames to be confirmed as sensor "
        "dust (real sensor dust sits in the same place across the session). Unit: count.", kind="int")),
    ("SENSOR_MAX_CANDIDATE_TEXTURE", _F("Pre-filter: drop candidates in busy areas (texture above this) before "
        "cross-frame consensus. Unit: gray-level standard deviation.")),
    ("SENSOR_MAX_CANDIDATES_FOR_CONSENSUS", _F("Per-frame cap on candidates entering the union-find; frames "
        "with more are skipped for consensus, to bound cost. Unit: count.", kind="int")),
    # --- correction ---
    ("SENSOR_BRUSH_SCALE", _F("Sensor-dust brush radius = blob radius x this (floored to the minimum brush "
        "size). Unit: multiplier of blob radius.")),
    ("SENSOR_MAX_CORRECTION_TEXTURE", _F("Skip correcting a confirmed dust spot on any frame where it lands on "
        "a busy area (texture above this). Unit: gray-level standard deviation.")),
    ("SENSOR_MAX_SOURCE_TEXTURE", _F("Skip correction when no clean healing source (texture below this) is "
        "found in any direction. Unit: gray-level standard deviation.")),
])

PRESETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")

# Nested layout a preset JSON is written in (kind -> sub-stage). load() flattens it.
GROUPS = collections.OrderedDict([
    ("dust", collections.OrderedDict([
        ("threshold", ["LOCAL_BG_KERNEL_FRAC", "NOISE_THRESHOLD_MULTIPLIER", "MIN_ABSOLUTE_THRESHOLD", "REJECT_LOG_CONTRAST_MIN"]),
        ("size_shape", ["MIN_SPOT_AREA_FRAC", "MAX_SPOT_AREA_FRAC", "MIN_ASPECT_RATIO", "MIN_COMPACTNESS", "MIN_SOLIDITY", "MIN_CIRCULARITY", "DOT_MIN_CIRCLE_FILL", "DOT_IRREGULAR_RADIUS_FRAC", "SHAPE_CHECK_MIN_AREA_FRAC"]),
        ("texture_contrast", ["TEXTURE_KERNEL_FRAC", "MAX_LOCAL_TEXTURE_SMALL", "MAX_LOCAL_TEXTURE_LARGE", "MAX_DARK_BG_TEXTURE", "MIN_CONTRAST_TEXTURE_RATIO", "MAX_BG_GRADIENT_RATIO", "MAX_CONTEXT_TEXTURE", "CONTEXT_TEXTURE_RADIUS_FRAC", "TEXTURE_TIER_AREA_FRAC", "LARGE_SPOT_AREA_THRESHOLD_FRAC", "LARGE_SPOT_MIN_CONTRAST"]),
        ("color", ["MAX_EXCESS_SATURATION", "MAX_SPOT_SATURATION", "EMULSION_EXCESS_SAT_THRESHOLD"]),
        ("voting", ["SOFT_CONTEXT_VOTE_THRESHOLD", "SOFT_TEXTURE_VOTE_THRESHOLD", "SOFT_RATIO_VOTE_THRESHOLD", "MIN_DUST_VOTES"]),
        ("brightness", ["MIN_BRIGHTNESS_FRAC_SMALL", "MIN_BRIGHTNESS_FRAC_LARGE", "MIN_LOCAL_BG_FRACTION", "MIN_SURROUND_BG_RATIO"]),
        ("isolation", ["ISOLATION_RADIUS_FRAC", "MAX_NEARBY_ACCEPTED", "MAX_SPOTS"]),
        ("brush_source", ["ENC_RADIUS_SCALE", "SOURCE_SEARCH_INNER_FACTOR", "SOURCE_SEARCH_MAX_RADIUS_FRAC", "SOURCE_SEARCH_MIN_RADIUS_FRAC", "SOURCE_GRID_STEP_FRAC"]),
        ("ml", ["ML_RECOVERY_THRESHOLD_MULT", "ML_POSTFILTER_THRESHOLD", "ML_RECOVERY_THRESHOLD"]),
    ])),
    ("stroke", collections.OrderedDict([
        ("detect", ["DETECT_STROKES"]),
        ("geometry", ["STROKE_MIN_LENGTH_FRAC", "STROKE_MIN_ELONGATION", "STROKE_MAX_WIDTH_FRAC", "STROKE_MIN_WIDTH_FRAC", "STROKE_DP_EPS_FRAC", "STROKE_MAX_FILL_RATIO", "STROKE_PREFER_RATIO", "STROKE_MAX_KEYPOINTS"]),
        ("brush", ["STROKE_COVERAGE_FRAC", "STROKE_COVERAGE_MIN_DIFF", "STROKE_COVERAGE_PCTL", "STROKE_COVERAGE_MARGIN_FRAC", "STROKE_BORDER_SCALE", "STROKE_MIN_BORDER_FRAC", "STROKE_MAX_BORDER_FRAC"]),
        ("ridge_pass", ["STROKE_RIDGE_ENABLE", "STROKE_RIDGE_SIGMA_FRACS", "STROKE_RIDGE_Z", "STROKE_RIDGE_MIN_CONTRAST", "STROKE_RIDGE_DOWNSCALE", "STROKE_RIDGE_MAX_CANDIDATES"]),
        ("gating", ["STROKE_MAX_BAND_TEXTURE", "STROKE_MAX_CONTEXT_TEXTURE", "STROKE_CONTEXT_RADIUS_FRAC", "STROKE_MAX_EXCESS_SAT", "STROKE_MIN_BRIGHTNESS_FRAC", "STROKE_MIN_RIDGE_DROP", "STROKE_MAX_SIDE_ASYMMETRY", "STROKE_SIDE_OFFSET_FACTOR", "STROKE_MIN_CRISPNESS", "STROKE_CLIP_LEVEL"]),
        ("source", ["STROKE_SOURCE_OFFSET_FACTOR", "STROKE_SOURCE_MIN_GAP_FRAC"]),
        ("heal_split", ["HEAL_SPLIT_BUSY", "HEAL_MAX_TEXTURE", "HEAL_BUSY_MARGIN_FRAC", "HEAL_SAMPLE_STEP_FRAC", "HEAL_MIN_SEGMENT_FRAC"]),
        ("field_isolation", ["STROKE_FIELD_RADIUS_FRAC", "STROKE_FIELD_INNER_FRAC", "STROKE_FIELD_MAX_LINE_CANDS", "STROKE_FIELD_CAND_MIN_AREA_FRAC", "STROKE_FIELD_CAND_MIN_DIM_FRAC", "STROKE_FIELD_CAND_MAX_FILL", "STROKE_FIELD_CAND_MIN_ELONG", "STROKE_FIELD_NBR_RADIUS_FRAC", "STROKE_FIELD_MAX_NEIGHBORS"]),
        ("streak", ["STREAK_DETECT", "STREAK_RIDGE_VGAP_FRAC", "STREAK_INTEG_LEN_FRAC", "STREAK_LEVEL_MULT", "STREAK_MIN_LEN_FRAC", "STREAK_MAX_THICKNESS_FRAC", "STREAK_MIN_ELONG"]),
        ("radon", ["STREAK_RADON_DETECT", "STREAK_RADON_MIN_RESP", "STREAK_RADON_MIN_COV", "STREAK_RADON_MAX_SLOPE", "STREAK_RADON_SLOPES", "STREAK_RADON_PRESENT", "STREAK_RADON_EXT_FACTOR", "STREAK_RADON_EXT_GAP_FRAC", "STREAK_RADON_MAX_HALFWIDTH_FRAC", "STREAK_RADON_MIN_BRUSH_R_FRAC", "STREAK_RADON_BRUSH_MARGIN_FRAC"]),
        ("hysteresis", ["STROKE_HYST_LOW_FACTOR", "STROKE_HYST_PAD_FRAC", "STROKE_HYST_PAD_LEN_FRAC", "STROKE_HYST_BRIDGE_FRAC"]),
    ])),
    ("sensor", collections.OrderedDict([
        ("dog", ["SENSOR_SIGMA_INNER_FRAC", "SENSOR_SIGMA_OUTER_FRAC", "SENSOR_DOG_MIN_CONTRAST", "SENSOR_MIN_RADIUS_FRAC", "SENSOR_MAX_BLOB_RADIUS_FRAC"]),
        ("consensus", ["SENSOR_CLUSTER_RADIUS_NORM", "SENSOR_DUST_MIN_FRAMES", "SENSOR_MAX_CANDIDATE_TEXTURE", "SENSOR_MAX_CANDIDATES_FOR_CONSENSUS"]),
        ("correction", ["SENSOR_BRUSH_SCALE", "SENSOR_MAX_CORRECTION_TEXTURE", "SENSOR_MAX_SOURCE_TEXTURE"]),
    ])),
])

# Build the shared schema (validates GROUPS partitions FIELDS) + expose its API at
# module level so detect_dust.py can do `tuning.load(...)`, `tuning.Tuning`, etc.
_schema = TuningSchema(FIELDS, GROUPS, PRESETS_DIR)
Tuning = _schema.Tuning
INT_FIELDS = _schema.INT_FIELDS
TUPLE_FIELDS = _schema.TUPLE_FIELDS
BOOL_FIELDS = _schema.BOOL_FIELDS
preset_path = _schema.preset_path
from_mapping = _schema.from_mapping
load = _schema.load
to_mapping = _schema.to_mapping
to_nested = _schema.to_nested
dump = _schema.dump
