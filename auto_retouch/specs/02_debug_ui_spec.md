# Debug UI for Dust Detection — Spec

## Purpose

When tuning the detection algorithm, the human needs to mark exact false
positives and missed dust spots without resorting to screenshot-guessing.
The debug UI lets the human annotate detected/rejected spots directly on the
exported image and produces a structured text file (`debug_report.txt`) that
Claude Code can read to understand what errors in detection need fixing.

Two kinds of errors:
1. **False positives** — detected by the script, but should be ignored.
2. **Missed dust** — not detected, but visible to the human.

---

## Architecture

- `detect_dust.py` is called with `--debug-ui` flag by the `AutoRetouch_Debug`
  action in `auto_retouch.lua`.
- After detection finishes, `detect_dust.py` writes `debug_spots.json` to the
  export dir, then launches `debug_ui.py <export_dir>` as a non-blocking
  subprocess (`Popen`) and exits. The UI runs independently.
- The UI reads `debug_spots.json` and on close writes `debug_report.txt`.

---

## Files Written at Runtime

| File | When | Contents |
|------|------|----------|
| `debug_spots.json` | detect_dust.py with `--debug-ui` | detected + rejected spots per image, detection constants |
| `{stem}_annotations.json` | debug_ui.py on each annotation change | per-image false_positives + missed_dust (auto-saved) |
| `debug_report.txt` | debug_ui.py on window close | human-readable summary for Claude Code |

---

## UI Layout

```
root (Tk, "Dust Detection Debug UI", 1400×900)
└── PanedWindow (horizontal, resizable)
    ├── LEFT PANEL (220px)
    │   ├── "Images:" label
    │   ├── Scrollable canvas thumbnail list (image thumbnail + filename per row)
    │   ├── "Show rejected candidates" checkbox
    │   ├── Status label (N detected / N rejected)
    │   ├── Legend (color → meaning)
    │   └── Hints (mouse + keyboard shortcuts)
    └── RIGHT PANEL
        ├── Image canvas (fill/expand) + H/V scrollbars
        └── Bottom panel (190px)
            ├── Scrollable info Text (selected spot details)
            └── Button panel (right side):
                  Rejected → Missed  (R)
                  Mark False Positive  (M)
                  Clear FP Mark  (C)
                  Remove Missed  (Del)
                  Clear Selection
                  Fit to Window  (F)
                  Hide Markers  (H)
                  FP: N | Missed: N  (count label)
```

---

## Marker Colors

| Marker | Color | Meaning |
|--------|-------|---------|
| Circle outline | Green `#00cc44` | Detected spot (accepted) |
| Circle outline | Yellow `#ffff00` | Detected spot (selected) |
| X through circle | Red `#ff3333` | False positive (marked by user) |
| Circle outline | Orange `#ff8800` | Rejected candidate (visible when toggled) |
| Circle outline | Yellow `#ffcc44` | Rejected candidate (selected) |
| + crosshair | Cyan `#00ffff` | Missed dust (added by user) |
| + crosshair | White `#ffffff` | Missed dust (selected) |
| Dashed rectangle | White | Rubber-band selection |
| Dashed rectangle | Blue `#4488ff` | Ctrl+drag zoom-to-rect |

---

## Mouse Interactions

| Action | Effect |
|--------|--------|
| Scroll wheel | Zoom in/out centered on cursor |
| Middle-button drag | Pan |
| Left-click on marker | Select that marker (deselects others) |
| Left-drag (> 5px) | Rubber-band multi-select (detected + missed + rejected if visible) |
| Shift+click on marker | Toggle marker in/out of selection |
| Shift+drag | Add markers in rect to existing selection |
| Ctrl+click on empty area | Immediately add a missed dust marker |
| Ctrl+click on existing marker | Select that marker (same as normal click) |
| Ctrl+drag (> 5px) | Zoom view to fit the drawn rectangle |

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `R` | Convert selected rejected candidates → missed dust |
| `M` | Mark selected detected spots as false positives |
| `C` | Clear FP mark from selected detected spots |
| `Del` / `Backspace` | Remove selected missed dust markers |
| `H` | Toggle hide/show all markers |
| `+` / `=` | Zoom ×2 centered on canvas center |
| `-` | Zoom ÷2 centered on canvas center |
| `←↑→↓` | Pan 80px |
| `Shift+←↑→↓` | Pan 300px |
| `F` | Fit image to window |

---

## Selection Model

All three marker types use `set` selection:
- `selected_detected` — set of int (detected spot indices)
- `selected_rejected` — set of int (rejected candidate indices)
- `selected_missed` — set of int (missed dust indices)

These are independent; clicking a marker of one type clears the other two.
Shift+click and rubber-band preserve existing selection of the same type.

---

## Annotation Actions

**Mark False Positive (M):** adds `selected_detected` indices to `false_positives` set. Visually shown as red X through the green circle.

**Clear FP Mark (C):** removes `selected_detected` from `false_positives`.

**Rejected → Missed (R):** copies cx/cy of all `selected_rejected` into `missed_dust` list and clears the rejected selection.

**Remove Missed (Del):** deletes all `selected_missed` indices from `missed_dust` (reverse-order deletion to keep indices stable).

**Ctrl+click:** appends a new `{"cx": ix, "cy": iy}` entry to `missed_dust` immediately and selects it.

---

## Auto-save

On every annotation change and on image switch, writes `{stem}_annotations.json`:
```json
{
  "stem": "DSC_0042",
  "false_positives": [{"cx": 1234.5, "cy": 678.2, "radius_px": 3.1, ...}],
  "missed_dust": [{"cx": 3011.0, "cy": 1450.0}]
}
```
Existing annotation files are loaded on startup (session resume).

---

## debug_report.txt Format (for Claude Code)

```
DUST DETECTION DEBUG REPORT
Generated: 2026-02-18 14:32:07
Export dir: C:\...\darktable_autoretouch_...

Detection constants at time of run:
  NOISE_THRESHOLD_MULTIPLIER = 3.0
  ...

================================================
IMAGE: DSC_0042  (5504 x 3672 px)
  Detected: 14 | Rejected shown: 8 | FP marked: 3 | Missed added: 2

  FALSE POSITIVES (detected, should be ignored):
    cx=1234.5  cy=678.2  radius=3.1px  area=31  contrast=47.3  texture=2.1  excess_sat=-0.4
    ...

  MISSED DUST (not detected, should have been):
    cx=3011.0  cy=1450.0
    ...

  REJECTED CANDIDATES (for context):
    cx=892.0  cy=1102.0  area=22  contrast=38.7  reason=texture  detail=texture=13.2>9.8
    ...

================================================
SUMMARY
  Total false positives across all images: 3
  Total missed dust across all images: 2
  Images with annotations: 1 / 2
```

---

## Zoom / Pan Details

- Zoom range: 0.05× – 20×
- `at_fit_zoom` flag: if true, canvas resize automatically re-fits the image
- Visible-region cropping: only the canvas-visible portion of the PIL image is
  uploaded as a PhotoImage (avoids 5500×3800 texture on every redraw)
- Resampling: LANCZOS when zoom ≥ 0.2, NEAREST below (speed)
