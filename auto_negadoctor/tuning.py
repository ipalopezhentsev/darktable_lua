"""Tuning configuration — the SCHEMA + documentation for every fittable
constant, with the VALUES held externally in `presets/*.json`.

Why split this way: a preset is just a flat ``name -> value`` JSON map, so
adopting calibration-fitted values is "drop in a new preset file", never editing
Python. But JSON can't carry comments, and the per-field RATIONALE (often
paragraphs distilled from tuning sessions) is the valuable part — so it lives
HERE, in `FIELDS`, the canonical commented schema the JSON is validated against
and parsed into. The result is an immutable `Tuning` namedtuple whose docstring
is built from these docs, so the loaded object stays "equally well-commented".

Runtime selection (auto_negadoctor.py): `--preset NAME|PATH` or `NEGA_PRESET`,
defaulting to the bundled `default` preset. `load()` reads + validates a preset;
`dump()` writes one (used by the calibration runner to emit a fitted preset).

Structural/format constants (XMP byte offsets, the lens/negadoctor blob
templates, CLIP_SRGB_THR, RESULTS_FILENAME) are NOT here — they are correctness
facts, not taste knobs (a "different" value means a corrupt sidecar, not a
different look), so they stay as plain constants in auto_negadoctor.py.
"""
import collections
import json
import os

_Field = collections.namedtuple("_Field", "kind doc")


def _F(doc, kind="float"):
    """A schema entry. `kind` drives load-time coercion so a JSON `2` stays an
    int and a JSON `[5, 30]` becomes the `(5.0, 30.0)` tuple the code expects."""
    return _Field(kind, doc)


# Order matches the historical _TUNABLE_NAMES so Tuning._fields is unchanged.
# Grouped by the pipeline stage that reads each constant.
FIELDS = collections.OrderedDict([
    # --- film-holder border trim (shared by crop + vignette) -----------------
    ("BORDER_DARK_THR", _F("linear luminance below this = holder plastic")),
    ("BORDER_MAX_FRAC", _F("never trim more than this per edge (frac of dim)")),
    ("BORDER_PAD_FRAC", _F("safety pad after the detected border (frac of width)")),

    # --- rectangular content crop --------------------------------------------
    # The holder is a RECTANGULAR frame around the film frame, so the analysis
    # area is the largest inscribed rectangle of pure film; everything outside
    # is disregarded in ALL level computations (no per-pixel masks). A scan line
    # is junk when it carries holder-dark pixels OR bright leak (nothing ON film
    # is lighter than the unexposed base, so "brighter than base by a margin"
    # cleanly catches light leaking past the holder edge).
    ("HOLDER_LUMA_THR", _F("linear luma below this is holder-dark")),
    ("CROP_JUNK_LINE_FRAC", _F("dark/leak fraction above which a line is trimmed")),
    ("CROP_LEAK_MARGIN_D", _F("brighter than base by this log10 density = leak")),
    ("CROP_REBATE_MARGIN_D", _F(
        "Film REBATE (unexposed strip at the frame edge) is base-colored: "
        "neither dark nor brighter-than-base. Signature: ALL channels within a "
        "small density margin of base. max-channel density below this = base-like.")),
    ("CROP_REBATE_LINE_FRAC", _F("base-like fraction above which a line is rebate")),
    ("CROP_REBATE_TERM_FRAC", _F(
        "A true rebate band TERMINATES (film content follows within tens of px); "
        "unexposed dark scene at the edge is base-like too but continues. Rebate "
        "trims are only valid if the base-like band ends soon after: lines past "
        "the trim that must NOT stay base-like (frac of width).")),
    ("CROP_REBATE_MAX_FRAC", _F(
        "A real rebate is a NARROW strip; bright SCENE content (sky/snow) is "
        "base-like too but in diffuse bands hundreds of px deep. The line "
        "threshold can't separate them, but band WIDTH can: max rebate-extension "
        "depth past the hard run (frac of width); a wider base-like run is scene.")),
    ("CROP_REBATE_HUE_TOL", _F(
        "WIDE confident-rebate path, signature 1 of 3 (HUE). When the user "
        "deliberately scrolls film in the holder to scan extra rebate, the rebate "
        "band can fill HALF the frame — far past CROP_REBATE_MAX_FRAC. Width can't "
        "tell that from a diffuse bright-scene band, but hue can: unexposed base "
        "matches the strongly-orange base chromaticity almost exactly (L1 dev "
        "~0.01) while any exposed scene — even blue sky, which inverts to a DENSE "
        "off-orange, or bright neutral snow — sits far off (~0.07-0.2). A pixel "
        "within this L1 chromaticity distance of the detected film base counts as "
        "hue-matched. Keep TIGHT (~0.03): looser admits warm scene.")),
    ("CROP_REBATE_WIDE_MAX_D", _F(
        "WIDE rebate signature 2 of 3 (DENSITY). The rebate is the lightest, "
        "UNexposed region; exposed scene (incl. orange-in-negative sky) is denser. "
        "max-channel density below this = light enough to be rebate. LOOSER than "
        "CROP_REBATE_MARGIN_D so the vignette-darkened rebate near the holder edge "
        "still qualifies, but well below typical scene density.")),
    ("CROP_REBATE_WIDE_LINE_FRAC", _F(
        "WIDE rebate signature 3 of 3 (SOLIDITY). Fraction of a line that must be "
        "confident-rebate pixels for the line to count toward the wide trim. HIGH "
        "(~0.5, vs the conservative CROP_REBATE_LINE_FRAC ~0.15): a true wide "
        "rebate fills almost the whole column, whereas warm scene only speckles it "
        "(~0.3-0.5).")),
    ("CROP_REBATE_BAND_HUE_TOL", _F(
        "WIDE rebate FINAL guard, and the decisive one. Once a candidate band is "
        "found, its whole-band MEAN color must be within this L1 chromaticity "
        "distance of the film base. A genuine rebate band IS the base, so it "
        "averages to it almost exactly (~0.002); smooth orange DAYTIME SCENE (sky/ "
        "ground/walls inverting to light orange) that passes the per-pixel mask "
        "still averages well off-base (~0.012-0.044) because the off-base pixels "
        "between the rebate-hued ones drag the mean. Keep TIGHT (~0.008): this is "
        "what stops the wide path from eating ordinary negative scene content.")),
    ("CROP_REBATE_WIDE_FRAC", _F(
        "Max depth the WIDE confident-rebate trim may reach (frac of dim) — much "
        "larger than BORDER_MAX_FRAC, since this path only fires on a SOLID band "
        "that hue-matches the base, is low-density, AND terminates into scene "
        "before this cap. Independent of the conservative BORDER_MAX_FRAC cap, so "
        "ordinary edges keep their tight trim.")),
    ("CROP_PAD_FRAC", _F("safety pad past the last junk line (frac of width)")),
    ("CROP_SHADOW_REL", _F(
        "Holder-edge SHADOW/penumbra: a darkened ramp too bright for "
        "HOLDER_LUMA_THR. Trim when a line's mean luma is below this fraction of "
        "the frame interior (found via the user's left-edge containment violations).")),
    ("CROP_SHADOW_MAX_FRAC", _F("a shadow run terminating within this depth is penumbra (frac)")),
    ("CROP_SHADOW_CORE_FRAC", _F("when the depressed band continues past the shadow max, its core depth (frac)")),
    ("CROP_GAP_TOL_FRAC", _F("junk must form an EDGE-ANCHORED run; tolerate gaps up to this (frac)")),

    # --- film-base rectangle search (on the negative) ------------------------
    ("BASE_GB_TOL", _F("film base is orange: require G*tol >= B")),
    ("BASE_MIN_LUMA", _F("linear; reject base windows darker than this")),
    ("BASE_MIN_RG_RATIO", _F("film base is strongly orange: R >= ratio*G")),
    ("BASE_UNIFORMITY_MAX", _F("luma std/mean above this = textured (e.g. foliage), reject; the lifeless unexposed base is far below it")),
    ("BASE_WIN_FRAC", _F("base uniformity window side as fraction of image width; sized ALONE (NOT floored by MIN_WIN_FRAC) so it can fit inside a thin base strip")),
    ("CLIP_FRAC_MAX", _F("max fraction of white-clipped px in a base window")),
    ("BASE_SCAN_STRIDE_FRAC", _F(
        "coarse-grid cell side (frac of width) for the largest-base-rectangle "
        "search; a cell counts as base when >= BASE_MASK_SOLID_FRAC of its "
        "pixels are base-like")),
    ("BASE_MASK_SOLID_FRAC", _F(
        "fraction of a coarse cell's pixels that must be base-like for the cell "
        "to count toward the rectangle")),
    ("BASE_AREA_MIN_FRAC", _F(
        "min base-rectangle area (frac of frame area) for a frame to QUALIFY as "
        "the roll-base source; among qualifiers the BRIGHTEST wins (area is a "
        "confidence gate, never outranks brightness). If none qualify, all do.")),

    # --- neutral-patch search (on the rendered inverted preview) -------------
    ("MIN_PATCH_DENSITY", _F("patch must be >= this much denser (log10 D) than film base, else the wb formulas degenerate")),
    ("PATCH_CHROMA_FLOOR", _F("chroma denominator floor (near-black windows)")),
    ("PATCH_CHROMA_MAX", _F("chroma of window means on the prior-wb preview")),
    ("PATCH_LUMA_FLOOR", _F("uniformity denominator floor (dark windows would otherwise explode std/mean into grain noise)")),
    ("PATCH_STRIDE_DIV", _F("patch search stride = window / this", kind="int")),
    ("PATCH_UNIFORMITY_MAX", _F("preview luma std/mean guard (scenes are textured)")),
    ("PATCH_WIN_FRAC", _F("neutral-patch search window side (frac of width)")),
    ("HIGHLIGHT_CLIP_FRAC_MAX", _F("negative-space clipped fraction guard")),
    ("SHADOW_MIN_LUMA", _F("preview luma floor (skip film-base gaps)")),
    ("MIN_WIN_FRAC", _F("floor for the search-window side (frac of width)")),

    # --- density-range pickers / levels --------------------------------------
    ("P_LOW", _F(
        "per-channel percentile for 'densest' values. 2.0 (not 0.5): the "
        "dense-side anchor must be robust to small holder-junk residues that "
        "survive the mask; the densest 0.5-2% proved to be edge junk, not scene. "
        "Side effect: the true top speculars soft-clip, matching the user's punch.")),
    ("P_HIGH", _F("per-channel percentile for 'lightest' values")),
    ("DMAX_DEFAULT", _F(
        "D_max (film dynamic range): darktable's negadoctor init() default, kept "
        "FIXED — the user leaves it at the default in practice. Auto-deriving it "
        "fabricated a D_max that exists nowhere in the real workflow and, since "
        "everything downstream divides by it, skewed offset/wb/black/exposure.")),
    ("OFFSET_DEFAULT", _F(
        "Scan exposure bias: darktable's default, kept fixed. The auto formula "
        "needs the lightest PHOTO content, but on uncropped scans the lightest "
        "area is the film base itself, which degenerates the formula to ~0.")),
    ("HIGHLIGHT_BAND_PCT", _F("preview-luma percentile band for the highlight patch", kind="tuple")),

    # --- white balance (the colors the user picks on the wheels) -------------
    # Per-frame wb estimates the cast HUE then applies a gentler-than-full
    # correction: the user's wheel picks are consistently MILDER than full
    # gray-world. wb = mean negative-space color of a dark/bright print-luma
    # BAND (robust region gray-world), neutralized by darktable's picker
    # formula, then pulled toward neutral by WB_*_DESAT. Intensity is a tonal
    # lever handled jointly — tune_print_params renders WITH the wb and adapts
    # black/exposure on that render, so a milder wb is compensated downstream.
    ("WB_LOW_DESAT", _F("pull shadow-cast gain toward neutral (0=full correction)")),
    ("WB_HIGH_DESAT", _F("pull highlight-cast gain toward neutral (the light-neutral patch already lands well, so 0)")),
    ("WB_LOW_BAND_PCT", _F("print-luma band whose mean = shadow cast", kind="tuple")),
    ("WB_HIGH_BAND_PCT", _F("print-luma band whose mean = highlight cast", kind="tuple")),
    ("WB_LOW_PRIOR", _F("preview wb for shadow band SELECTION only (close to the final rendition); no longer a result prior", kind="tuple")),
    ("WB_HIGH_PRIOR", _F("preview wb for highlight band SELECTION only", kind="tuple")),
    ("WB_REGION_MIN_FRAC", _F("min band sample fraction of the cropped area")),

    # --- print auto-tuning / look --------------------------------------------
    # The user's process: push brightness to the CLIP BOUNDARY ("maximise
    # brightness without the highlights clipping"), using black to keep
    # brightening once exposure is maxed, and backing exposure off if it clips.
    # This auto-preserves MOOD without scene labels: a bright scene has headroom
    # so it goes bright; a dark scene clips early so it stays dark. Each iter
    # exposure pins the high percentile to PRINT_HI_CEIL; when exposure
    # saturates and the highlight is still low, BLACK takes over; a final guard
    # lowers exposure until hard-clip frac <= PRINT_CLIP_BUDGET.
    ("PRINT_GAMMA", _F(
        "Paper grade. Tuned to the GT PICTURE, not GT params (spec 04): the "
        "exposure/wb<->picture map is many-to-one, so matching the GT picture "
        "needs a FLATTER curve than the GT gamma number. 5.0 is the joint "
        "luma-EMD minimum across the rolls. The print tune adapts black/exposure "
        "around whatever gamma is set here. Re-derive with calibrate_histogram_match.py.")),
    ("PRINT_HI_CEIL", _F(
        "pin the high percentile (PRINT_HI_PCT) just below the GT's top. Tuned "
        "under the high-resolution (14-bit) histogram metric: 0.97 cut the "
        "top-highlight gap vs the old 0.99 while staying clip-safe (<0.3%).")),
    ("PRINT_HI_PCT", _F("the actual top percentile controlled (not P99.5, which lets the very top run away)")),
    ("PRINT_CLIP_BUDGET", _F("final guard: lower exposure until hard-clip frac (any channel >= 0.999) is at most this")),
    ("PRINT_TUNE_ITERS", _F("print auto-tune iteration count", kind="int")),

    # --- roll-wide vignette estimation ---------------------------------------
    # Lens + backlight vignetting darkens negative corners -> after inversion
    # corners come out brighter and skew level analysis. Physics: nothing ON
    # film is lighter than base, so the per-pixel MAX over exposure-normalized
    # frames approaches V(x,y)*const — the spatial envelope IS the vignette
    # field. Fitted in darktable's lens-module manual-vignette space.
    ("VIG_DOWNSAMPLE_FRAC", _F("accumulate the envelope on a /N grid; stride = frac of width (constant grid size vs resolution)")),
    ("VIG_INSET_FRAC", _F("extra inset past the dark-border trim (frac of width)")),
    ("VIG_BINS", _F("radial bins for the envelope profile", kind="int")),
    ("VIG_PROFILE_PCT", _F("per-bin envelope percentile")),
    ("VIG_MIN_BIN_SAMPLES", _F("bins with fewer samples are skipped", kind="int")),
    ("VIG_MIN_STRENGTH", _F("below this the roll counts as vignette-free")),
    ("VIG_PEAK_CENTER_FRAC", _F("bins below this radius are the centre plateau; the reference envelope is their max (the peak ring isn't always dead centre)")),
    ("VIG_TAIL_CUT_REL", _F("outward from the envelope peak, cut the falling corner-leak tail once a bin drops below this fraction of the running max")),
])

INT_FIELDS = frozenset(n for n, f in FIELDS.items() if f.kind == "int")
TUPLE_FIELDS = frozenset(n for n, f in FIELDS.items() if f.kind == "tuple")

# Nested layout a preset JSON is written in: top level = the three calibration
# KINDS (so the file mirrors how you run/tune), each split into pipeline
# sub-stages. Purely cosmetic grouping — load() flattens it back to the flat
# field set (and still accepts an old flat preset). BORDER_DARK_THR/PAD live
# under vignette (detect_dark_border's holder mask — its only use now the film
# base is found full-frame); BORDER_MAX_FRAC is the content-crop cap, under crop.
GROUPS = collections.OrderedDict([
    ("crop", collections.OrderedDict([
        ("border", ["BORDER_MAX_FRAC"]),
        ("content", ["HOLDER_LUMA_THR", "CROP_JUNK_LINE_FRAC", "CROP_LEAK_MARGIN_D",
                     "CROP_REBATE_MARGIN_D", "CROP_REBATE_LINE_FRAC",
                     "CROP_REBATE_TERM_FRAC", "CROP_REBATE_MAX_FRAC",
                     "CROP_REBATE_HUE_TOL", "CROP_REBATE_WIDE_MAX_D",
                     "CROP_REBATE_WIDE_LINE_FRAC", "CROP_REBATE_BAND_HUE_TOL",
                     "CROP_REBATE_WIDE_FRAC", "CROP_PAD_FRAC",
                     "CROP_SHADOW_REL", "CROP_SHADOW_MAX_FRAC", "CROP_SHADOW_CORE_FRAC",
                     "CROP_GAP_TOL_FRAC"]),
    ])),
    ("inversion", collections.OrderedDict([
        ("film_base", ["BASE_GB_TOL", "BASE_MIN_LUMA", "BASE_MIN_RG_RATIO",
                       "BASE_UNIFORMITY_MAX", "BASE_WIN_FRAC", "CLIP_FRAC_MAX",
                       "BASE_SCAN_STRIDE_FRAC", "BASE_MASK_SOLID_FRAC",
                       "BASE_AREA_MIN_FRAC"]),
        ("neutral_patch", ["MIN_PATCH_DENSITY", "PATCH_CHROMA_FLOOR", "PATCH_CHROMA_MAX",
                           "PATCH_LUMA_FLOOR", "PATCH_STRIDE_DIV", "PATCH_UNIFORMITY_MAX",
                           "PATCH_WIN_FRAC", "HIGHLIGHT_CLIP_FRAC_MAX", "SHADOW_MIN_LUMA",
                           "MIN_WIN_FRAC"]),
        ("levels", ["P_LOW", "P_HIGH", "DMAX_DEFAULT", "OFFSET_DEFAULT",
                    "HIGHLIGHT_BAND_PCT"]),
        ("white_balance", ["WB_LOW_DESAT", "WB_HIGH_DESAT", "WB_LOW_BAND_PCT",
                           "WB_HIGH_BAND_PCT", "WB_LOW_PRIOR", "WB_HIGH_PRIOR",
                           "WB_REGION_MIN_FRAC"]),
        ("print", ["PRINT_GAMMA", "PRINT_HI_CEIL", "PRINT_HI_PCT", "PRINT_CLIP_BUDGET",
                   "PRINT_TUNE_ITERS"]),
    ])),
    ("vignette", collections.OrderedDict([
        # detect_dark_border's holder mask (its only remaining use now that the
        # film-base search is full-frame): bounds the envelope's valid region.
        ("holder_mask", ["BORDER_DARK_THR", "BORDER_PAD_FRAC"]),
        ("envelope", ["VIG_DOWNSAMPLE_FRAC", "VIG_INSET_FRAC", "VIG_BINS",
                      "VIG_PROFILE_PCT", "VIG_MIN_BIN_SAMPLES"]),
        ("profile", ["VIG_MIN_STRENGTH", "VIG_PEAK_CENTER_FRAC", "VIG_TAIL_CUT_REL"]),
    ])),
])

# Guard: the grouping must cover EXACTLY the field set, once each (so a new
# constant can't be silently dropped from presets).
_grouped = [n for kind in GROUPS.values() for grp in kind.values() for n in grp]
assert sorted(_grouped) == sorted(FIELDS) and len(_grouped) == len(FIELDS), (
    "GROUPS must partition FIELDS exactly: "
    f"missing {sorted(set(FIELDS) - set(_grouped))}, "
    f"extra/dup {sorted([n for n in _grouped if _grouped.count(n) > 1 or n not in FIELDS])}")

Tuning = collections.namedtuple("Tuning", list(FIELDS))
Tuning.__doc__ = (
    "Immutable tuning configuration (one field per fittable constant; values "
    "loaded from a preset JSON). Field docs:\n\n"
    + "\n".join(f"  {n}: {f.doc}" for n, f in FIELDS.items()))

PRESETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")


def preset_path(preset):
    """Resolve a preset selector to a JSON path. A bare NAME -> presets/NAME.json;
    a value containing a path separator or ending in .json is used verbatim."""
    if os.sep in preset or (os.altsep and os.altsep in preset) or preset.endswith(".json"):
        return preset
    return os.path.join(PRESETS_DIR, preset + ".json")


def _coerce(name, val):
    if name in INT_FIELDS:
        return int(round(val)) if isinstance(val, float) else int(val)
    if name in TUPLE_FIELDS:
        return tuple(float(x) for x in val)
    return float(val)


def from_mapping(raw, source="<mapping>"):
    """Build a Tuning from a flat name->value mapping, validating it covers
    EXACTLY the schema (no missing or unknown keys) and coercing types."""
    missing = [n for n in FIELDS if n not in raw]
    unknown = [n for n in raw if n not in FIELDS]
    if missing or unknown:
        raise ValueError(
            f"preset {source}: missing {missing or '[]'}, unknown {unknown or '[]'}")
    return Tuning(**{n: _coerce(n, raw[n]) for n in FIELDS})


def _flatten(tree, source, _flat=None):
    """Collapse a (possibly nested) preset mapping to a flat name->value dict.
    Any key that is a field name is a leaf; any other key must map to a nested
    dict (a group) and is recursed into. Accepts the OLD flat layout unchanged
    (every key is a leaf). Raises on a stray/unknown key."""
    flat = {} if _flat is None else _flat
    for k, v in tree.items():
        if k in FIELDS:
            if k in flat:
                raise ValueError(f"preset {source}: field {k!r} appears twice")
            flat[k] = v
        elif isinstance(v, dict):
            _flatten(v, source, flat)
        else:
            raise ValueError(
                f"preset {source}: {k!r} is neither a tuning field nor a group")
    return flat


def load(preset="default"):
    """Load + validate a preset (NAME or path) into an immutable Tuning. The
    JSON may be nested by GROUPS or a flat name->value map (both accepted)."""
    path = preset_path(preset)
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return from_mapping(_flatten(raw, path), source=path)


def to_mapping(cfg):
    """Tuning -> a FLAT JSON-ready dict (tuples become lists)."""
    out = collections.OrderedDict()
    for n in FIELDS:
        v = getattr(cfg, n)
        out[n] = list(v) if isinstance(v, tuple) else v
    return out


def to_nested(cfg):
    """Tuning -> a NESTED JSON-ready dict grouped by GROUPS (tuples -> lists)."""
    flat = to_mapping(cfg)
    out = collections.OrderedDict()
    for kind, subs in GROUPS.items():
        out[kind] = collections.OrderedDict(
            (sub, collections.OrderedDict((n, flat[n]) for n in names))
            for sub, names in subs.items())
    return out


def dump(cfg, path):
    """Write a Tuning as a NESTED preset JSON (values only; docs live in this
    module). Used by the calibration runner to emit a fitted preset directly."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_nested(cfg), fh, indent=2)
        fh.write("\n")
