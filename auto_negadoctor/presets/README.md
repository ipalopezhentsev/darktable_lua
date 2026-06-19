# Tuning presets

Each `*.json` here is a **preset**: the values of the 58 fittable tuning
constants, grouped into a nested tree by the three calibration **kinds** and
their pipeline sub-stages:

```jsonc
{
  "crop":      { "border": {…}, "content": {…} },
  "inversion": { "film_base": {…}, "neutral_patch": {…}, "levels": {…},
                 "white_balance": {…}, "print": {…} },
  "vignette":  { "envelope": {…}, "profile": {…} }
}
```

It is the *values* half of the tuning configuration; the *documentation* half —
every field's rationale — lives in [`../tuning.py`](../tuning.py)'s `FIELDS`
schema, because JSON can't carry comments. The grouping (`GROUPS` in `tuning.py`)
is purely cosmetic: on load the tree is flattened back to the field set and
validated (it must cover **exactly** the 58 constants — no missing, no unknown,
each once) before being parsed into the immutable `Tuning` object the analysis
code reads via its `cfg` argument. A bare **flat** `CONSTANT: value` map is also
still accepted (older presets keep loading).

Structural/format constants (XMP byte offsets, the lens/negadoctor blob
templates, `CLIP_SRGB_THR`, `RESULTS_FILENAME`) are **not** here — a different
value there means a corrupt sidecar, not a different look, so there is nothing
to tune and they stay as plain constants in `auto_negadoctor.py`.

## Selecting a preset

`default.json` is used unless you choose another, by name (→ `presets/<name>.json`)
or by explicit path:

```sh
python auto_negadoctor.py img.tif --preset punchy
python auto_negadoctor.py img.tif --preset /path/to/experiment.json
NEGA_PRESET=punchy python auto_negadoctor.py img.tif      # env var, same effect
```

`--preset` overrides `$NEGA_PRESET`, which overrides the bundled `default`.

## Adopting calibration-fitted values (the whole point)

A calibration session writes a complete **drop-in preset** at
`tests/calibrations/<session>/fitted_preset.json` (= the default preset with the
session's fitted overrides applied). To adopt it there is **no editing of
`auto_negadoctor.py`** — either point at it directly, or copy it in as a named
preset:

```sh
# try it without committing to it
python auto_negadoctor.py img.tif --preset tests/calibrations/<session>/fitted_preset.json

# or promote it to a named preset
cp tests/calibrations/<session>/fitted_preset.json auto_negadoctor/presets/punchy.json
```

`fitted_params.json` in the same session folder remains the minimal record of
*just what moved* (constant → init → fitted); `fitted_preset.json` is the full,
runnable configuration.

## Editing a preset by hand

It's plain JSON — change a value and run. The schema validation will reject a
typo'd or missing key on load and tell you which. For what each constant does
and its sane range, read `FIELDS` in `tuning.py` (the field docs) and the
per-kind ranges in `tests/calibration_registry.py`.
