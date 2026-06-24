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
    ("LOCAL_BG_KERNEL", _F("Gaussian blur kernel for local background", kind="int")),
    ("NOISE_THRESHOLD_MULTIPLIER", _F("spots must be this many std devs above background")),
    ("MIN_ABSOLUTE_THRESHOLD", _F("minimum brightness difference regardless of noise")),
    ("REJECT_LOG_CONTRAST_MIN", _F("minimum contrast to include in debug reject candidate list", kind="int")),
    # --- size_shape ---
    ("MIN_SPOT_AREA", _F("minimum pixels (reject subpixel/imperceptible dust)", kind="int")),
    ("MAX_SPOT_AREA", _F("maximum pixels (~16px radius at full res)", kind="int")),
    ("MIN_ASPECT_RATIO", _F("bounding box aspect ratio (reject elongated fibers)")),
    ("MIN_COMPACTNESS", _F("area / bbox_area (reject irregular shapes)")),
    ("MIN_SOLIDITY", _F("area / convex_hull_area (reject non-convex shapes like letters)")),
    ("MIN_CIRCULARITY", _F("4*pi*area/perimeter^2 (reject complex shapes like symbols)")),
    ("DOT_MIN_CIRCLE_FILL", _F("area / (pi*enc_r^2) below this counts as non-compact")),
    ("DOT_IRREGULAR_RADIUS_FRAC", _F("...and only when enc_r exceeds this * min_dim (~18px @ 3745)")),
    ("SHAPE_CHECK_MIN_AREA", _F("only check solidity/circularity for spots larger than this", kind="int")),
    # --- texture_contrast ---
    ("TEXTURE_KERNEL", _F("neighborhood size for local texture measurement", kind="int")),
    ("MAX_LOCAL_TEXTURE_SMALL", _F("max texture for tiny spots (area near MIN_SPOT_AREA)")),
    ("MAX_LOCAL_TEXTURE_LARGE", _F("max texture for large spots (area >= 200px)")),
    ("MAX_DARK_BG_TEXTURE", _F("max texture in ring around spot on dark backgrounds (separate from above)")),
    ("MIN_CONTRAST_TEXTURE_RATIO", _F("contrast/texture — reject spots hidden in grain")),
    ("MAX_BG_GRADIENT_RATIO", _F("max bg_gradient/contrast — reject edge halo artifacts")),
    ("MAX_CONTEXT_TEXTURE", _F("max median local_std across a 200px radius from spot center.")),
    ("LARGE_SPOT_AREA_THRESHOLD", _F("spots larger than this require higher contrast", kind="int")),
    ("LARGE_SPOT_MIN_CONTRAST", _F("min contrast for large spots — avoids pale foggy blobs", kind="int")),
    # --- color ---
    ("MAX_EXCESS_SATURATION", _F("max (spot_sat - surround_sat) — dust matches local color cast", kind="int")),
    ("MAX_SPOT_SATURATION", _F("compound check lower bound: spot_sat above this + positive excess_sat", kind="int")),
    ("EMULSION_EXCESS_SAT_THRESHOLD", _F("excess_sat above this (when spot_sat > 230) = emulsion artifact:", kind="int")),
    # --- voting ---
    ("SOFT_CONTEXT_VOTE_THRESHOLD", _F("context_texture < this votes \"dust\" (clear sky/walls: 2-6)")),
    ("SOFT_TEXTURE_VOTE_THRESHOLD", _F("local_texture < this votes \"dust\"")),
    ("SOFT_RATIO_VOTE_THRESHOLD", _F("contrast/texture > this votes \"dust\" (1.5× hard minimum)")),
    ("MIN_DUST_VOTES", _F("require at least this many out of 3 soft votes to accept", kind="int")),
    # --- brightness ---
    ("MIN_BRIGHTNESS_FRAC_SMALL", _F("brightness floor for tiny spots (area ~10)")),
    ("MIN_BRIGHTNESS_FRAC_LARGE", _F("brightness floor for large spots (area >= 100)")),
    ("MIN_LOCAL_BG_FRACTION", _F("local background must be >= this fraction of 95th pct")),
    ("MIN_SURROUND_BG_RATIO", _F("immediate surround must be >= this fraction of local bg")),
    # --- isolation ---
    ("ISOLATION_RADIUS", _F("pixel radius for neighbor density check", kind="int")),
    ("MAX_NEARBY_ACCEPTED", _F("reject if more than this many accepted spots within ISOLATION_RADIUS", kind="int")),
    ("MAX_SPOTS", _F("cap: sort by contrast, take the most obvious ones", kind="int")),
    # --- brush_source ---
    ("ENC_RADIUS_SCALE", _F("multiplicative scale factor for enclosing circle radius (helps darktable", kind="int")),
    ("SOURCE_SEARCH_INNER_FACTOR", _F("inner exclusion ring = radius * this (avoid the spot itself)")),
    ("SOURCE_SEARCH_MAX_RADIUS", _F("cap search radius in pixels", kind="int")),
    ("SOURCE_SEARCH_MIN_RADIUS", _F("minimum search radius for tiny spots", kind="int")),
    ("SOURCE_GRID_STEP", _F("grid step for candidate sampling (pixels)", kind="int")),
    # --- ml ---
    ("ML_RECOVERY_THRESHOLD_MULT", _F("lower threshold for recovery pass (find missed dust)")),
    ("ML_POSTFILTER_THRESHOLD", _F("min ML probability to keep a spot (post-filter)")),
    ("ML_RECOVERY_THRESHOLD", _F("higher bar for accepting recovery candidates")),
    # ===================== STROKE =====================
    # --- detect ---
    ("DETECT_STROKES", _F("master switch for thread/scratch stroke detection (on by default)", kind="bool")),
    # --- geometry ---
    ("STROKE_MIN_LENGTH_FRAC", _F("min centerline length as fraction of min_dim (~19px @ 3786).")),
    ("STROKE_MIN_ELONGATION", _F("min length/width — separates strokes from blobs")),
    ("STROKE_MAX_WIDTH_FRAC", _F("max stroke width as fraction of min_dim (~9.5px @ 3786).")),
    ("STROKE_MIN_WIDTH_PX", _F("floor for measured width (avoid div-by-zero on 1px ridges)")),
    ("STROKE_DP_EPS_FRAC", _F("Douglas-Peucker epsilon as fraction of min_dim (path simplify)")),
    ("STROKE_MAX_FILL_RATIO", _F("area/bbox_area pre-filter: a thin thread fills little of its")),
    ("STROKE_PREFER_RATIO", _F("convert if circle_healed_area / stroke_healed_area > this")),
    ("STROKE_MAX_KEYPOINTS", _F("cap nodes per stroke (darktable form is fine with many)", kind="int")),
    # --- brush ---
    ("STROKE_COVERAGE_FRAC", _F("edge = where diff falls below this fraction of the local peak")),
    ("STROKE_COVERAGE_MIN_DIFF", _F("...but never below this absolute diff (avoid chasing noise)")),
    ("STROKE_COVERAGE_PCTL", _F("use this percentile of per-sample half-widths — a thread can be", kind="int")),
    ("STROKE_COVERAGE_MARGIN_PX", _F("add this margin so the feathered edge is fully covered")),
    ("STROKE_BORDER_SCALE", _F("fallback multiplier on the core half-width if the profile")),
    ("STROKE_MIN_BORDER_PX", _F("minimum per-node brush border in pixels (darktable floor)")),
    ("STROKE_MAX_BORDER_FRAC", _F("cap brush border at this * min_dim (~22px) — no runaway brushes")),
    # --- ridge_pass ---
    ("STROKE_RIDGE_ENABLE", _F("master switch for the faint-scratch Hessian ridge pass (disabled by default — noise-floor false positives)", kind="bool")),
    ("STROKE_RIDGE_SIGMAS", _F("ridge scales in px (thin scratches/threads)", kind="tuple")),
    ("STROKE_RIDGE_Z", _F("keep ridge response above median + this*MAD (robust z-score).")),
    ("STROKE_RIDGE_MIN_CONTRAST", _F("min mean diff (gray-local_bg) along the ridge to accept")),
    ("STROKE_RIDGE_DOWNSCALE", _F("analyze ridge filter at this scale for speed (full-res too slow);")),
    ("STROKE_RIDGE_MAX_CANDIDATES", _F("cap ridge components processed (largest first) to bound cost", kind="int")),
    # --- gating ---
    ("STROKE_MAX_BAND_TEXTURE", _F("max median local_std in the band around the centerline.")),
    ("STROKE_MAX_CONTEXT_TEXTURE", _F("max median local_std in a WIDE band. NOT contrast-tiered")),
    ("STROKE_CONTEXT_RADIUS_FRAC", _F("wide-context band outer radius as fraction of min_dim (~190px)")),
    ("STROKE_MAX_EXCESS_SAT", _F("max (stroke_sat - surround_sat) — dust/scratch is achromatic", kind="int")),
    ("STROKE_MIN_BRIGHTNESS_FRAC", _F("mean stroke brightness >= this * bright_ref (bright bg only)")),
    ("STROKE_MIN_RIDGE_DROP", _F("core crest must beat the BRIGHTER side by this (gray levels)")),
    ("STROKE_MAX_SIDE_ASYMMETRY", _F("|left_bg - right_bg| / core — reject one-sided (edge) profiles")),
    ("STROKE_SIDE_OFFSET_FACTOR", _F("perpendicular sample offset = width * this + a few px")),
    ("STROKE_MIN_CRISPNESS", _F("reject strokes softer than this (out-of-focus scene wires)")),
    ("STROKE_CLIP_LEVEL", _F("core crest >= this (8-bit) counts as clipped-white dust", kind="int")),
    # --- source ---
    ("STROKE_SOURCE_OFFSET_FACTOR", _F("min source offset = this * brush_radius (gap ~= brush_radius)")),
    ("STROKE_SOURCE_MIN_GAP_PX", _F("...and at least this absolute gap beyond the two brush edges")),
    # --- heal_split ---
    ("HEAL_SPLIT_BUSY", _F("split strokes so the brush skips busy (high-texture) runs", kind="bool")),
    ("HEAL_MAX_TEXTURE", _F("local_std above this along the path counts as \"busy\" (skip)")),
    ("HEAL_BUSY_MARGIN_PX", _F("also skip this far around each busy run (sources stay clean)")),
    ("HEAL_SAMPLE_STEP_PX", _F("resample the path at this spacing to test texture")),
    ("HEAL_MIN_SEGMENT_FRAC", _F("keep a smooth run only if at least this * min_dim long (~38px)")),
    # --- field_isolation ---
    ("STROKE_FIELD_RADIUS_FRAC", _F("line-candidate outer annulus radius, fraction of min_dim (~560px)")),
    ("STROKE_FIELD_INNER_PX", _F("inner annulus radius (excludes the stroke's own fragments)", kind="int")),
    ("STROKE_FIELD_MAX_LINE_CANDS", _F("reject a stroke with >= this many line-candidates near it", kind="int")),
    ("STROKE_FIELD_CAND_MIN_AREA", _F("a line-candidate component must be at least this large", kind="int")),
    ("STROKE_FIELD_CAND_MIN_DIM", _F("...and at least this long on its major axis", kind="int")),
    ("STROKE_FIELD_CAND_MAX_FILL", _F("...and thin (area/bbox below this)")),
    ("STROKE_FIELD_CAND_MIN_ELONG", _F("...and elongated (bbox long/short ratio at least this)")),
    ("STROKE_FIELD_NBR_RADIUS_FRAC", _F("accepted-stroke neighbour radius, fraction of min_dim (~450px)")),
    ("STROKE_FIELD_MAX_NEIGHBORS", _F("reject if more than this many accepted strokes are that near.", kind="int")),
    # --- streak ---
    ("STREAK_DETECT", _F("master switch for the faint axis-aligned scratch producer", kind="bool")),
    ("STREAK_RIDGE_VGAP", _F("px above/below to compare against (ridge, not edge)", kind="int")),
    ("STREAK_INTEG_LEN", _F("horizontal integration length (px) — boosts faint line SNR", kind="int")),
    ("STREAK_LEVEL_MULT", _F("response threshold = max(6, noise * this)")),
    ("STREAK_MIN_LEN_FRAC", _F("min streak length as fraction of min_dim (~150px)")),
    ("STREAK_MAX_THICKNESS", _F("max component thickness (px) — keeps it thin & near-axis", kind="int")),
    ("STREAK_MIN_ELONG", _F("min length/thickness ratio (strongly elongated)", kind="int")),
    # --- radon ---
    ("STREAK_RADON_DETECT", _F("master switch for the full-width Radon scratch detector", kind="bool")),
    ("STREAK_RADON_MIN_RESP", _F("min mean ridge response along the best line (gray levels)")),
    ("STREAK_RADON_MIN_COV", _F("min fraction of the width with response present on that line")),
    ("STREAK_RADON_MAX_SLOPE", _F("search slopes in [-this, this] (~1.1 deg) — \"almost\" axis")),
    ("STREAK_RADON_SLOPES", _F("number of slope steps to test", kind="int")),
    ("STREAK_RADON_PRESENT", _F("response above this counts as \"scratch present\" at a column")),
    ("STREAK_RADON_EXT_FACTOR", _F("endpoint-extension threshold = PRESENT * this (follow fading)")),
    ("STREAK_RADON_EXT_GAP", _F("stop extending after this many px with no faint signal", kind="int")),
    ("STREAK_RADON_MAX_HALFWIDTH", _F("cap the measured brush half-width for a faint scratch (px)")),
    ("STREAK_RADON_MIN_BRUSH_R", _F("minimum brush radius (half-width) for a Radon scratch (px)")),
    ("STREAK_RADON_BRUSH_MARGIN", _F("added to the measured half-width")),
    # --- hysteresis ---
    ("STROKE_HYST_LOW_FACTOR", _F("tail/extension threshold = main threshold * this")),
    ("STROKE_HYST_PAD_FRAC", _F("search window pad around the seed bbox, fraction of min_dim")),
    ("STROKE_HYST_PAD_LEN_FRAC", _F("...or this fraction of the seed length, whichever is larger")),
    ("STROKE_HYST_BRIDGE_PX", _F("bridge noise gaps up to this long ALONG the stroke axis", kind="int")),
    # ===================== SENSOR =====================
    # --- dog ---
    ("SENSOR_SIGMA_INNER_FRAC", _F("DoG inner Gaussian sigma as fraction of min(w,h)")),
    ("SENSOR_SIGMA_OUTER_FRAC", _F("DoG outer Gaussian sigma (large enough to see around blobs,")),
    ("SENSOR_DOG_MIN_CONTRAST", _F("minimum DoG peak value; low enough to catch faint dust")),
    ("SENSOR_MIN_RADIUS_FRAC", _F("min blob radius as fraction of min(w,h)")),
    ("SENSOR_MAX_BLOB_RADIUS_FRAC", _F("maximum accepted blob radius (% of min_dim); sensor dust is small")),
    # --- consensus ---
    ("SENSOR_CLUSTER_RADIUS_NORM", _F("cluster radius in normalized full-frame coords")),
    ("SENSOR_DUST_MIN_FRAMES", _F("min frames a cluster must appear in to confirm sensor dust", kind="int")),
    ("SENSOR_MAX_CANDIDATE_TEXTURE", _F("pre-filter: reject candidates in busy areas before consensus")),
    ("SENSOR_MAX_CANDIDATES_FOR_CONSENSUS", _F("per-frame cap before union-find; frames with more candidates", kind="int")),
    # --- correction ---
    ("SENSOR_BRUSH_SCALE", _F("brush_radius_px = max(MIN_BRUSH_PX, radius_px * this)")),
    ("SENSOR_MAX_CORRECTION_TEXTURE", _F("skip correction if dust spot lands on a busy area in this frame")),
    ("SENSOR_MAX_SOURCE_TEXTURE", _F("skip correction if no clean healing source found in any direction")),
])

PRESETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")

# Nested layout a preset JSON is written in (kind -> sub-stage). load() flattens it.
GROUPS = collections.OrderedDict([
    ("dust", collections.OrderedDict([
        ("threshold", ["LOCAL_BG_KERNEL", "NOISE_THRESHOLD_MULTIPLIER", "MIN_ABSOLUTE_THRESHOLD", "REJECT_LOG_CONTRAST_MIN"]),
        ("size_shape", ["MIN_SPOT_AREA", "MAX_SPOT_AREA", "MIN_ASPECT_RATIO", "MIN_COMPACTNESS", "MIN_SOLIDITY", "MIN_CIRCULARITY", "DOT_MIN_CIRCLE_FILL", "DOT_IRREGULAR_RADIUS_FRAC", "SHAPE_CHECK_MIN_AREA"]),
        ("texture_contrast", ["TEXTURE_KERNEL", "MAX_LOCAL_TEXTURE_SMALL", "MAX_LOCAL_TEXTURE_LARGE", "MAX_DARK_BG_TEXTURE", "MIN_CONTRAST_TEXTURE_RATIO", "MAX_BG_GRADIENT_RATIO", "MAX_CONTEXT_TEXTURE", "LARGE_SPOT_AREA_THRESHOLD", "LARGE_SPOT_MIN_CONTRAST"]),
        ("color", ["MAX_EXCESS_SATURATION", "MAX_SPOT_SATURATION", "EMULSION_EXCESS_SAT_THRESHOLD"]),
        ("voting", ["SOFT_CONTEXT_VOTE_THRESHOLD", "SOFT_TEXTURE_VOTE_THRESHOLD", "SOFT_RATIO_VOTE_THRESHOLD", "MIN_DUST_VOTES"]),
        ("brightness", ["MIN_BRIGHTNESS_FRAC_SMALL", "MIN_BRIGHTNESS_FRAC_LARGE", "MIN_LOCAL_BG_FRACTION", "MIN_SURROUND_BG_RATIO"]),
        ("isolation", ["ISOLATION_RADIUS", "MAX_NEARBY_ACCEPTED", "MAX_SPOTS"]),
        ("brush_source", ["ENC_RADIUS_SCALE", "SOURCE_SEARCH_INNER_FACTOR", "SOURCE_SEARCH_MAX_RADIUS", "SOURCE_SEARCH_MIN_RADIUS", "SOURCE_GRID_STEP"]),
        ("ml", ["ML_RECOVERY_THRESHOLD_MULT", "ML_POSTFILTER_THRESHOLD", "ML_RECOVERY_THRESHOLD"]),
    ])),
    ("stroke", collections.OrderedDict([
        ("detect", ["DETECT_STROKES"]),
        ("geometry", ["STROKE_MIN_LENGTH_FRAC", "STROKE_MIN_ELONGATION", "STROKE_MAX_WIDTH_FRAC", "STROKE_MIN_WIDTH_PX", "STROKE_DP_EPS_FRAC", "STROKE_MAX_FILL_RATIO", "STROKE_PREFER_RATIO", "STROKE_MAX_KEYPOINTS"]),
        ("brush", ["STROKE_COVERAGE_FRAC", "STROKE_COVERAGE_MIN_DIFF", "STROKE_COVERAGE_PCTL", "STROKE_COVERAGE_MARGIN_PX", "STROKE_BORDER_SCALE", "STROKE_MIN_BORDER_PX", "STROKE_MAX_BORDER_FRAC"]),
        ("ridge_pass", ["STROKE_RIDGE_ENABLE", "STROKE_RIDGE_SIGMAS", "STROKE_RIDGE_Z", "STROKE_RIDGE_MIN_CONTRAST", "STROKE_RIDGE_DOWNSCALE", "STROKE_RIDGE_MAX_CANDIDATES"]),
        ("gating", ["STROKE_MAX_BAND_TEXTURE", "STROKE_MAX_CONTEXT_TEXTURE", "STROKE_CONTEXT_RADIUS_FRAC", "STROKE_MAX_EXCESS_SAT", "STROKE_MIN_BRIGHTNESS_FRAC", "STROKE_MIN_RIDGE_DROP", "STROKE_MAX_SIDE_ASYMMETRY", "STROKE_SIDE_OFFSET_FACTOR", "STROKE_MIN_CRISPNESS", "STROKE_CLIP_LEVEL"]),
        ("source", ["STROKE_SOURCE_OFFSET_FACTOR", "STROKE_SOURCE_MIN_GAP_PX"]),
        ("heal_split", ["HEAL_SPLIT_BUSY", "HEAL_MAX_TEXTURE", "HEAL_BUSY_MARGIN_PX", "HEAL_SAMPLE_STEP_PX", "HEAL_MIN_SEGMENT_FRAC"]),
        ("field_isolation", ["STROKE_FIELD_RADIUS_FRAC", "STROKE_FIELD_INNER_PX", "STROKE_FIELD_MAX_LINE_CANDS", "STROKE_FIELD_CAND_MIN_AREA", "STROKE_FIELD_CAND_MIN_DIM", "STROKE_FIELD_CAND_MAX_FILL", "STROKE_FIELD_CAND_MIN_ELONG", "STROKE_FIELD_NBR_RADIUS_FRAC", "STROKE_FIELD_MAX_NEIGHBORS"]),
        ("streak", ["STREAK_DETECT", "STREAK_RIDGE_VGAP", "STREAK_INTEG_LEN", "STREAK_LEVEL_MULT", "STREAK_MIN_LEN_FRAC", "STREAK_MAX_THICKNESS", "STREAK_MIN_ELONG"]),
        ("radon", ["STREAK_RADON_DETECT", "STREAK_RADON_MIN_RESP", "STREAK_RADON_MIN_COV", "STREAK_RADON_MAX_SLOPE", "STREAK_RADON_SLOPES", "STREAK_RADON_PRESENT", "STREAK_RADON_EXT_FACTOR", "STREAK_RADON_EXT_GAP", "STREAK_RADON_MAX_HALFWIDTH", "STREAK_RADON_MIN_BRUSH_R", "STREAK_RADON_BRUSH_MARGIN"]),
        ("hysteresis", ["STROKE_HYST_LOW_FACTOR", "STROKE_HYST_PAD_FRAC", "STROKE_HYST_PAD_LEN_FRAC", "STROKE_HYST_BRIDGE_PX"]),
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
