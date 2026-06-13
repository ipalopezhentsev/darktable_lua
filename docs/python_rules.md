## Python Environment

- Uses conda environment named `autocrop` (defined in `environment.yml`)
- Setup: `conda env create -f environment.yml`
- Update: `conda env update -f environment.yml --prune`
- Dependencies: Python 3.11, OpenCV, NumPy, Pillow

## Python Usage Rules

- New Python dependencies must be added via `environment.yml`

## Resolution independence (image-analysis code)

The analysis scripts run on darktable exports whose pixel size is a free knob
(e.g. `EXPORT_MAX_WIDTH` in the Lua side). Detection MUST behave the same
regardless of that size — a constant tied to the absolute export resolution is
a bug (it silently changes detection when the export width changes).

- Express every size-dependent threshold as a **fraction of the frame
  dimension** — a `*_FRAC` constant turned into pixels at the use site with
  `int(round(w * SOME_FRAC))` (add a `max(1, …)` floor where a count of zero
  would be degenerate, e.g. a stride). `auto_negadoctor.py` follows this for
  all of them; there is NO reference resolution. A raw pixel literal compared
  against image coordinates is a bug.
- Coordinates persisted for later comparison (annotations, fixtures) must be
  **normalized** ([0,1] fractions of the frame), not pixels — so they stay
  valid across export resolutions. Convert at the I/O boundary.
- auto_negadoctor guards this with `tests/test_resolution_invariance.py`: it
  runs the detectors at width W and an exact 2x copy and asserts the outputs
  scale ~2x. Run it (and extend the analogous pattern) when adding detection
  logic. NOTE: `auto_crop` and `auto_retouch` predate this convention and may
  still hold absolute-pixel constants — apply the same rule when touching them.
