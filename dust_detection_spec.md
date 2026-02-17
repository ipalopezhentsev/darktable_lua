# Dust Detection Specification

## What We're Detecting

Dust and debris particles on DSLR-scanned film negatives. After the negative is inverted to positive, dust appears as **bright white spots** on the image because dust blocked light during scanning.

This is **film dust** (varies per frame, random positions) — NOT sensor dust (fixed position across all frames).

## Core Assumptions About Dust

1. **Bright**: Dust is near film-base brightness (the brightest possible value on inverted negatives). On bright backgrounds, dust is at or near the 95th percentile brightness. On dark backgrounds, dust is much brighter than its surroundings but not bright in absolute terms.

2. **Small and round-ish**: Most dust is 6–800 pixels (1–16px radius at full D7200 resolution 5688x3779). Dust particles are roughly circular blobs; fibers are elongated (rejected).

3. **Neutral/unsaturated**: Dust has no color — it matches whatever color cast the film base has. It is never more saturated than its surroundings.

4. **On smooth backgrounds**: Dust is most visible (and most worth fixing) on uniform areas — sky, water, walls, studio backgrounds. On busy/textured areas, dust is either invisible or indistinguishable from image features.

5. **Isolated bright feature**: Dust sits ON the background. It doesn't have dark rings, halos, or frames around it. A bright spot inside a dark frame (like a window reflection or porthole) is NOT dust.

6. **Size vs. brightness rule**: Very small dust (6–10px) can appear gray due to subpixel mixing with the background. But larger dust particles (30+ px) must be white — they physically block light across multiple pixels and always appear at film-base brightness.

## Algorithm Overview

### Stage 1: Global Image Analysis

Computed once per image:

| Map | Method | Purpose |
|-----|--------|---------|
| `gray` | Convert to grayscale | Working image |
| `local_bg` | Gaussian blur (k=201) | Background estimate — smooths dust but preserves gradual brightness changes (vignetting, subjects) |
| `diff` | `gray - local_bg` | Brightness above background — positive = candidate |
| `bg_gradient` | Sobel on `local_bg` | Detects edges in the background where Gaussian halos create false positives |
| `local_std` | Box filter std (k=31) | Local texture/busyness map — smooth areas have low std |
| `saturation` | HSV S channel | Color saturation for each pixel |

### Stage 2: Thresholding

- **Noise estimation**: `noise_std = median(|diff|) * 1.4826` (MAD estimator, robust to outliers)
- **Threshold**: `max(15.0, noise_std * 3.0)` — spots must be at least 3σ above background
- **Binary mask**: `diff > threshold`
- **Connected components**: 8-connectivity, extracts area/bbox/centroid for each blob

### Stage 3: Per-Candidate Filter Pipeline

Each connected component is passed through filters in order. Rejection is immediate — once a filter rejects, no further checks. The order matters for both performance (cheap checks first) and correctness (some checks only apply on bright backgrounds).

#### Filter 1: Size
- Reject if `area < 6` (imperceptible, subpixel)
- Reject if `area > 800` (too large to be dust)

**Rationale**: Very small components are noise. Very large ones are image features, not dust.

#### Filter 2: Aspect Ratio
- `aspect = min(w,h) / max(w,h)` of the bounding box
- Reject if `aspect < 0.3`

**Rationale**: Dust is roughly circular. Elongated shapes are fibers, scratches, or text.

**Relaxed from 0.5**: Large dust can appear as crescents/arcs when the Gaussian blur partially absorbs the center.

#### Filter 3: Compactness (small spots only)
- `compactness = area / (w * h)`
- Reject if `area < 30 AND compactness < 0.25`

**Rationale**: Small spots should fill most of their bounding box. Does NOT apply to large spots because Gaussian absorption can make them hollow/crescent-shaped.

#### Filter 4: Contrast
- `contrast = max(diff)` within the component pixels
- Reject if `contrast < threshold * 0.8`

**Rationale**: Component must clearly exceed the noise floor.

#### Filter 5: Shape — Solidity & Circularity (large spots only, area >= 30)
- `solidity = area / convex_hull_area` — reject if < 0.5
- `circularity = 4π·area / perimeter²` — reject if < 0.15

**Rationale**: Large dust is blob-shaped. Letters, symbols, and complex features have low solidity (concavities) and low circularity (complex perimeters).

**Relaxed thresholds**: Large dust absorbed by Gaussian creates fragmented/crescent shapes, so thresholds are generous.

---

*From here, `is_dark_bg` is computed: `local_bg[centroid] < bright_ref * 0.5`. The pipeline branches based on this.*

---

#### Filter 6: Dark Background Gate
Only for spots where `is_dark_bg = True`:

- **Texture check**: Measure `local_std` in a ring around the spot (inner = max(2×radius, 17px), outer = inner + 15px). Ring avoids the spot itself inflating the measurement in dark areas.
  - Reject if ring texture > 8.0 (the area is busy — grass, stairs, metal grids)
- **Contrast floor**: Reject if `contrast < threshold * 4` (= 12× noise std)

**Rationale**: On dark backgrounds, noise creates many false bright peaks. Only very high-contrast spots on smooth dark surfaces (night sky, dark hallways) are trustworthy. The texture ring avoids the dust pixel itself artificially inflating the std (one bright pixel on a dark background = huge variance).

**Known limitation**: The texture measurement via `local_std` (31×31 kernel) is contaminated by the dust spot within ~15px. The ring starts outside this zone but this makes the check less sensitive for very small spots.

#### Filter 7: Brightness / Dim (bright backgrounds only)
Only for spots where `is_dark_bg = False` and `area >= 10`:

- Scale brightness requirement linearly from 50% of 95th pct (area=10) to 80% (area>=100)
- Reject if spot's mean brightness < required

**Rationale**: On bright backgrounds, dust should be near film-base white. Small spots can be gray (subpixel mixing), but large dust is always white. This catches gray image features like text ("Canon C" letter) that pass shape checks.

**Skipped on dark backgrounds**: Dust on a dark background (bg=18) has absolute brightness ~100–200, which is way below the global 95th percentile but still clearly dust relative to its background.

#### Filter 8: Edge Gradient
- `grad_ratio = bg_gradient[centroid] / contrast`
- Reject if `grad_ratio > 0.08`

**Rationale**: Near edges in the blurred background, the Gaussian kernel creates halo artifacts — false bright/dark fringes. If the background gradient is a significant fraction of the spot's contrast, the "spot" is probably an edge artifact.

**Ratio-based**: Absolute gradient thresholds rejected real dust near moderate gradients. Using the ratio ensures high-contrast dust passes even with some gradient.

#### Filter 9: Embedded Bright Spot (bright backgrounds only)
Only for spots where `is_dark_bg = False`:

- In a neighborhood of radius `max(3 × radius_px, 8)`, compute 5th percentile brightness
- `surround_ratio = 5th_percentile / local_bg_brightness`
- Reject if `surround_ratio < 0.7`

**Rationale**: Dust sits ON a uniform background — the area around it is equally bright. A window reflection or porthole has a dark frame immediately around it, creating a very low 5th percentile. Ship model windows, portholes, and similar features get caught by this.

**Skipped on dark backgrounds**: Noise-level pixel variation on dark surfaces (5th percentile naturally low from grain) triggers false rejects.

#### Filter 10: Texture (all spots)
- Measure `local_std` in a ring from `1.5 × radius` to `3 × radius + TEXTURE_KERNEL` around centroid (using median)
- **Area-scaled threshold**: linearly from 16.0 (area=6) to 8.0 (area>=200)
- Reject if `texture > scaled_threshold`

**Rationale**: Dust is on smooth/uniform areas. Image features (Christmas lights, bokeh, leaves) are surrounded by texture. Larger features need smoother surroundings to be credible dust — a tiny speck in moderate texture could be dust, but a 200px blob in moderate texture is almost certainly an image feature.

#### Filter 11: Contrast-to-Texture Ratio
- Reject if `contrast / texture < 6.0`

**Rationale**: Dust must stand out clearly above the local grain/noise. A faint spot in a moderately noisy area isn't worth fixing.

#### Filter 12: Color / Excess Saturation
- `excess_sat = spot_saturation - surround_saturation` (using the same ring as texture)
- Reject if `excess_sat > 7`

**Rationale**: Dust is neutral — it matches the local film color cast. Colored image features (green leaves, red objects, yellow ornaments) are more saturated than their surroundings. Only checks if the spot is MORE saturated — dust being LESS saturated than surroundings (negative excess_sat) is normal and expected.

### Stage 4: Final Selection

- Sort accepted spots by contrast (descending)
- Take top 200 (darktable's MAX_FORMS limit is 300, but leave headroom)

## Two Regimes: Bright vs. Dark Backgrounds

The algorithm handles these fundamentally differently:

| Property | Bright background | Dark background |
|----------|------------------|-----------------|
| Examples | Sky, water, walls, studio bg | Night sky, dark hallways, shadows |
| `is_dark_bg` | `local_bg >= 50% of 95th pct` | `local_bg < 50% of 95th pct` |
| Brightness check | Yes — must be near film-base white | No — absolute brightness is meaningless |
| Contrast floor | `threshold * 0.8` (3σ noise) | `threshold * 4` (12σ noise) |
| Embedded check | Yes | No — noise makes low percentile unreliable |
| Texture sampling | Ring from 1.5× to 3× radius | Ring outside `local_std` kernel influence zone |

## Known Limitations & Possible Improvements

1. **Gaussian absorption of large dust**: The 201px Gaussian kernel partially absorbs large dust into the background estimate, creating crescent/ring shapes in the diff. This means large dust has reduced contrast, odd shapes, and creates gradients in the blurred background. Shape thresholds are relaxed to compensate, but this makes the system more permissive for image features too.

2. **Texture kernel contamination**: The 31×31 `local_std` kernel means a bright dust pixel contaminates texture readings up to ~15px away. The ring-based sampling works around this but adds complexity and can miss very small spots.

3. **No multi-scale approach**: The single Gaussian kernel size is a compromise. Small dust is well-separated from background with k=201, but very large dust (200+ px) gets absorbed. A multi-scale approach (multiple kernel sizes) could handle both better.

4. **No spatial clustering**: Dust is random; image features often cluster (row of windows, pattern of bolts). Currently no check for whether detections form suspicious patterns.

5. **Dark background texture is noisy**: On very dark areas (bg < 30), even `local_std` in a ring is unreliable because the absolute noise is a huge fraction of the signal. The 12σ contrast floor compensates but is a blunt instrument.

## Test Images

| Image | Content | Key challenge |
|-------|---------|---------------|
| DSC_0025 | Boats on water, sky | Main benchmark — many real dust spots on sky/water |
| DSC_0002 | Keyboard | Avoid detecting key symbols as dust |
| DSC_0030 | Landscape: sky + dark ground | Ground false positives must be rejected |
| DSC_0033 | Canon printer text | "C" letter detected as dust (rejected by brightness filter) |
| DSC_0007 | Christmas lights (off) | Lights look like dust — rejected by area-scaled texture |
| DSC_0005 | Green foliage | Bright leaves on dark bg — rejected by texture + color |
| DSC_0021 | Ship model in dark hallway | Window reflections (embedded filter) + dark-bg dust detection |
| DSC_0006 | Yellow leaf | Colored object — rejected by excess saturation filter |
| DSC_0016 | Christmas ornaments | Bright colored objects — rejected by brightness filter |
