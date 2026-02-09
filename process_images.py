#!/usr/bin/env python3
"""
Auto Crop Image Processing Script
Processes one or more images exported from darktable to detect white/light margins
and calculate crop percentages.

Usage: process_images.py <image_path> [image_path2 ...]
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np

# Margin tightening constants
# Base tightening applied to all margins
BASE_TIGHTENING_PERCENT = 0.3
# Additional tightening applied based on low confidence (0.0 = confident, 1.0 = not confident)
# Final tightening = BASE + (1.0 - confidence) * ADDITIONAL
ADDITIONAL_TIGHTENING_PERCENT = 1.5

def detect_content_bounds(image_path, save_visualization=False):
    """
    Detect the actual content boundaries by finding where margins end.
    Uses brightness profile analysis across full edges to find rectangular margins.
    Returns crop percentages: (left%, top%, right%, bottom%)

    Args:
        image_path: Path to the image file
        save_visualization: If True, saves a copy with boundary overlay
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        return None, "Failed to load image"

    height, width = img.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply slight blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    def scan_edge_profile(axis, direction, scan_depth_percent=20):
        """
        Scan edge using brightness profile analysis.
        Averages across the full perpendicular axis to exploit rectangular margin shape.

        axis: 'horizontal' or 'vertical'
        direction: 'left', 'right', 'top', or 'bottom'
        scan_depth_percent: how far to scan (percentage of image dimension)

        Returns: (boundary_position, confidence_score)
            confidence_score: 0.0 (low confidence) to 1.0 (high confidence)
        """

        def calculate_uniformity(margin_region):
            """
            Calculate uniformity score for a margin region.
            Measures how consistent the brightness is along the perpendicular axis.

            Args:
                margin_region: 2D numpy array of the margin region

            Returns:
                uniformity_score: 0.0-1.0 where 1.0 = very uniform (true margin)
            """
            if margin_region.size == 0:
                return 1.0  # No margin detected, assume uniform

            # Calculate standard deviation of brightness across the margin
            std_dev = np.std(margin_region)

            # Normalize: low std_dev = uniform = high score
            # Typical margin std_dev: 5-20
            # Content/texture std_dev: 30-80+
            # Map std_dev to 0-1 score (lower std_dev = higher score)
            if std_dev < 10:
                uniformity = 1.0
            elif std_dev < 30:
                # Linear interpolation between 1.0 and 0.5
                uniformity = 1.0 - (std_dev - 10) / 40.0
            else:
                # Exponential decay for high variance
                uniformity = 0.5 * np.exp(-(std_dev - 30) / 30.0)

            return max(0.0, min(1.0, uniformity))

        def calculate_confidence(margin_diff, agreement_dist, max_agreement, sanity_applied, uniformity_score):
            """Calculate confidence score based on detection quality metrics."""
            # Base confidence from brightness difference (8-120 range normalized)
            brightness_confidence = min(1.0, (margin_diff - 8) / 112.0)

            # Agreement confidence (good agreement = high confidence)
            agreement_confidence = 1.0 - (agreement_dist / max_agreement)
            agreement_confidence = max(0.0, agreement_confidence)

            # Uniformity confidence (low variance = uniform margin = high confidence)
            # uniformity_score is already 0.0-1.0 where 1.0 = very uniform
            uniformity_confidence = uniformity_score

            # Combine factors (uniformity gets highest weight as it's most reliable)
            confidence = (brightness_confidence * 0.3) + (agreement_confidence * 0.2) + (uniformity_confidence * 0.5)

            # Penalize if sanity check was applied
            if sanity_applied:
                confidence *= 0.5

            return max(0.0, min(1.0, confidence))
        if axis == 'horizontal':
            scan_depth = int(width * scan_depth_percent / 100)

            if direction == 'left':
                # Create brightness profile by averaging across full height
                profile = []
                for x in range(scan_depth):
                    col_mean = blurred[:, x].mean()
                    profile.append(col_mean)

                # Get edge and interior reference brightness
                edge_brightness = np.mean(profile[:15])
                start_x = int(width * 0.2)
                interior_brightness = blurred[:, start_x:start_x+30].mean()

                # Calculate brightness difference between margin and content
                margin_content_diff = abs(edge_brightness - interior_brightness)

                # If there's no significant brightness difference, no margin exists
                if margin_content_diff < 8:
                    return 0, 1.0  # High confidence - no margin detected

                # Use a threshold based on the actual brightness difference
                # Look for where brightness becomes similar to content
                threshold = margin_content_diff * 0.6

                # Scan from interior outward
                outward_result = 0
                for x in range(start_x, -1, -1):
                    col_mean = blurred[:, x].mean()
                    margin_similarity = abs(col_mean - edge_brightness)

                    if margin_similarity < threshold:
                        outward_result = x + 1
                        break

                # Also scan from edge inward for verification
                inward_result = 0
                for x in range(len(profile)):
                    content_similarity = abs(profile[x] - interior_brightness)
                    margin_similarity = abs(profile[x] - edge_brightness)

                    # When closer to content than to margin
                    if content_similarity < margin_similarity:
                        inward_result = x
                        break

                # Check agreement
                agreement = abs(outward_result - inward_result)
                max_agreement = max(20, width * 0.01)
                if agreement < max_agreement:
                    result = outward_result
                else:
                    # When scans disagree, prefer the larger (more conservative crop) if it's reasonable
                    if inward_result > 0:
                        larger_result = max(outward_result, inward_result)
                        larger_pct = (larger_result / width) * 100
                        # Use larger result if it's < 5% (reasonable margin size)
                        if larger_pct < 5.0:
                            result = larger_result
                        else:
                            result = min(outward_result, inward_result)
                    else:
                        result = outward_result

                # Extra sanity check: if margin >3%, require larger brightness difference
                sanity_applied = False
                result_pct = (result / width) * 100
                if result_pct > 3.0 and margin_content_diff < 60:
                    result = min(result, int(width * 0.03))
                    sanity_applied = True

                # Calculate uniformity of detected margin region
                if result > 0:
                    margin_region = blurred[:, 0:result]
                    uniformity = calculate_uniformity(margin_region)
                else:
                    uniformity = 1.0  # No margin = perfectly uniform

                # Calculate confidence and return
                confidence = calculate_confidence(margin_content_diff, agreement, max_agreement, sanity_applied, uniformity)
                return result, confidence

            else:  # right
                # Create brightness profile
                profile = []
                for x in range(width - 1, width - scan_depth - 1, -1):
                    col_mean = blurred[:, x].mean()
                    profile.append(col_mean)

                edge_brightness = np.mean(profile[:15])
                start_x = int(width * 0.8)
                interior_brightness = blurred[:, start_x-30:start_x].mean()

                # Calculate brightness difference
                margin_content_diff = abs(edge_brightness - interior_brightness)

                if margin_content_diff < 8:
                    return width - 1, 1.0  # High confidence - no margin detected

                threshold = margin_content_diff * 0.6

                # Scan from interior outward
                outward_result = width - 1
                for x in range(start_x, width):
                    col_mean = blurred[:, x].mean()
                    margin_similarity = abs(col_mean - edge_brightness)

                    if margin_similarity < threshold:
                        outward_result = x - 1
                        break

                # Scan from edge inward for verification
                inward_result = width - 1
                for i, brightness in enumerate(profile):
                    content_similarity = abs(brightness - interior_brightness)
                    margin_similarity = abs(brightness - edge_brightness)

                    if content_similarity < margin_similarity:
                        inward_result = width - 1 - i
                        break

                # Check agreement
                agreement = abs((width - outward_result) - (width - inward_result))
                max_agreement = max(20, width * 0.01)
                if agreement < max_agreement:
                    result = outward_result
                else:
                    # When scans disagree, prefer the larger margin if reasonable
                    if inward_result < width - 1:
                        smaller_result = min(outward_result, inward_result)
                        smaller_pct = ((width - smaller_result) / width) * 100
                        # Use smaller x-value (larger margin) if margin size is < 5%
                        if smaller_pct < 5.0:
                            result = smaller_result
                        else:
                            result = max(outward_result, inward_result)
                    else:
                        result = outward_result

                # Extra sanity check
                sanity_applied = False
                result_pct = ((width - result) / width) * 100
                if result_pct > 3.0 and margin_content_diff < 60:
                    result = max(result, width - int(width * 0.03))
                    sanity_applied = True

                # Calculate uniformity of detected margin region
                if result < width - 1:
                    margin_region = blurred[:, result:width]
                    uniformity = calculate_uniformity(margin_region)
                else:
                    uniformity = 1.0  # No margin = perfectly uniform

                # Calculate confidence and return
                confidence = calculate_confidence(margin_content_diff, agreement, max_agreement, sanity_applied, uniformity)
                return result, confidence

        else:  # vertical
            scan_depth = int(height * scan_depth_percent / 100)

            if direction == 'top':
                # Create brightness profile by averaging across full width
                profile = []
                for y in range(scan_depth):
                    row_mean = blurred[y, :].mean()
                    profile.append(row_mean)

                edge_brightness = np.mean(profile[:15])
                start_y = int(height * 0.2)
                interior_brightness = blurred[start_y:start_y+30, :].mean()

                # Calculate brightness difference
                margin_content_diff = abs(edge_brightness - interior_brightness)

                if margin_content_diff < 8:
                    return 0, 1.0  # High confidence - no margin detected

                threshold = margin_content_diff * 0.6

                # Scan from interior outward
                outward_result = 0
                for y in range(start_y, -1, -1):
                    row_mean = blurred[y, :].mean()
                    margin_similarity = abs(row_mean - edge_brightness)

                    if margin_similarity < threshold:
                        outward_result = y + 1
                        break

                # Scan from edge inward for verification
                inward_result = 0
                for y in range(len(profile)):
                    content_similarity = abs(profile[y] - interior_brightness)
                    margin_similarity = abs(profile[y] - edge_brightness)

                    if content_similarity < margin_similarity:
                        inward_result = y
                        break

                # Check agreement
                agreement = abs(outward_result - inward_result)
                max_agreement = max(20, height * 0.01)
                if agreement < max_agreement:
                    result = outward_result
                else:
                    # When scans disagree, prefer the larger margin if reasonable
                    if inward_result > 0:
                        larger_result = max(outward_result, inward_result)
                        larger_pct = (larger_result / height) * 100
                        if larger_pct < 5.0:
                            result = larger_result
                        else:
                            result = min(outward_result, inward_result)
                    else:
                        result = outward_result

                # Extra sanity check: if margin >3%, require larger brightness difference
                sanity_applied = False
                result_pct = (result / height) * 100
                if result_pct > 3.0 and margin_content_diff < 60:
                    # Bright content mistaken for margin, use inward scan or cap at 3%
                    result = min(result, int(height * 0.03))
                    sanity_applied = True

                # Calculate uniformity of detected margin region
                if result > 0:
                    margin_region = blurred[0:result, :]
                    uniformity = calculate_uniformity(margin_region)
                else:
                    uniformity = 1.0  # No margin = perfectly uniform

                # Calculate confidence and return
                confidence = calculate_confidence(margin_content_diff, agreement, max_agreement, sanity_applied, uniformity)
                return result, confidence

            else:  # bottom
                # Create brightness profile
                profile = []
                for y in range(height - 1, height - scan_depth - 1, -1):
                    row_mean = blurred[y, :].mean()
                    profile.append(row_mean)

                edge_brightness = np.mean(profile[:15])
                start_y = int(height * 0.8)
                interior_brightness = blurred[start_y-30:start_y, :].mean()

                # Calculate brightness difference
                margin_content_diff = abs(edge_brightness - interior_brightness)

                if margin_content_diff < 8:
                    return height - 1, 1.0  # High confidence - no margin detected

                threshold = margin_content_diff * 0.6

                # Scan from interior outward
                outward_result = height - 1
                for y in range(start_y, height):
                    row_mean = blurred[y, :].mean()
                    margin_similarity = abs(row_mean - edge_brightness)

                    if margin_similarity < threshold:
                        outward_result = y - 1
                        break

                # Scan from edge inward for verification
                inward_result = height - 1
                for i, brightness in enumerate(profile):
                    content_similarity = abs(brightness - interior_brightness)
                    margin_similarity = abs(brightness - edge_brightness)

                    if content_similarity < margin_similarity:
                        inward_result = height - 1 - i
                        break

                # Check agreement
                agreement = abs((height - outward_result) - (height - inward_result))
                max_agreement = max(20, height * 0.01)
                if agreement < max_agreement:
                    result = outward_result
                else:
                    # When scans disagree, prefer the larger margin if reasonable
                    if inward_result < height - 1:
                        smaller_result = min(outward_result, inward_result)
                        smaller_pct = ((height - smaller_result) / height) * 100
                        if smaller_pct < 5.0:
                            result = smaller_result
                        else:
                            result = max(outward_result, inward_result)
                    else:
                        result = outward_result

                # Extra sanity check
                sanity_applied = False
                result_pct = ((height - result) / height) * 100
                if result_pct > 3.0 and margin_content_diff < 60:
                    result = max(result, height - int(height * 0.03))
                    sanity_applied = True

                # Calculate uniformity of detected margin region
                if result < height - 1:
                    margin_region = blurred[result:height, :]
                    uniformity = calculate_uniformity(margin_region)
                else:
                    uniformity = 1.0  # No margin = perfectly uniform

                # Calculate confidence and return
                confidence = calculate_confidence(margin_content_diff, agreement, max_agreement, sanity_applied, uniformity)
                return result, confidence

    # Scan from each edge to find content boundaries
    x, confidence_left = scan_edge_profile('horizontal', 'left')
    x_right, confidence_right = scan_edge_profile('horizontal', 'right')
    y, confidence_top = scan_edge_profile('vertical', 'top')
    y_bottom, confidence_bottom = scan_edge_profile('vertical', 'bottom')

    # Calculate width and height of content region
    w = x_right - x
    h = y_bottom - y

    # Calculate crop percentages
    crop_left = (x / width) * 100
    crop_top = (y / height) * 100
    crop_right = ((width - (x + w)) / width) * 100
    crop_bottom = ((height - (y + h)) / height) * 100

    # Apply confidence-based tightening to make crops more conservative
    # Lower confidence = more tightening
    tightening_left = BASE_TIGHTENING_PERCENT + (1.0 - confidence_left) * ADDITIONAL_TIGHTENING_PERCENT
    tightening_top = BASE_TIGHTENING_PERCENT + (1.0 - confidence_top) * ADDITIONAL_TIGHTENING_PERCENT
    tightening_right = BASE_TIGHTENING_PERCENT + (1.0 - confidence_right) * ADDITIONAL_TIGHTENING_PERCENT
    tightening_bottom = BASE_TIGHTENING_PERCENT + (1.0 - confidence_bottom) * ADDITIONAL_TIGHTENING_PERCENT

    crop_left += tightening_left
    crop_top += tightening_top
    crop_right += tightening_right
    crop_bottom += tightening_bottom

    # Save visualization if requested
    visualization_path = None
    if save_visualization:
        # Create a copy of the original image
        vis_img = img.copy()

        # Draw thin red rectangle (BGR format: red = (0, 0, 255))
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Generate output filename
        img_path = Path(image_path)
        vis_filename = img_path.stem + "_boundary" + img_path.suffix
        visualization_path = img_path.parent / vis_filename

        # Save the visualization
        cv2.imwrite(str(visualization_path), vis_img)

    return {
        'crop_left': crop_left,
        'crop_top': crop_top,
        'crop_right': crop_right,
        'crop_bottom': crop_bottom,
        'confidence_left': confidence_left,
        'confidence_top': confidence_top,
        'confidence_right': confidence_right,
        'confidence_bottom': confidence_bottom,
        'detected_region': (x, y, w, h),
        'image_size': (width, height),
        'visualization_path': visualization_path
    }, None

def write_crop_results(results, output_path):
    """
    Write crop detection results in simple line format for Lua consumption.

    Format: OK|filename|L=1.23|T=2.34|R=3.45|B=4.56
            ERR|filename|error message
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for r in results:
                if r["status"] == "success":
                    c = r["crop"]
                    f.write(f'OK|{r["filename"]}|L={c["left"]:.2f}|T={c["top"]:.2f}|R={c["right"]:.2f}|B={c["bottom"]:.2f}\n')
                else:
                    f.write(f'ERR|{r["filename"]}|{r.get("error", "Unknown error")}\n')
        print(f"\nCrop results written to: {output_path}")
    except Exception as e:
        print(f"Error writing results file: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: process_images.py <image_path> [image_path2 ...]")
        print("\nThis script processes one or more images exported by the darktable AutoCrop plugin.")
        sys.exit(1)

    # Check for --no-vis flag
    save_visualization = "--no-vis" not in sys.argv
    image_paths = [a for a in sys.argv[1:] if a != "--no-vis"]
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}

    # Validate all files first
    image_files = []
    for image_path in image_paths:
        img_file = Path(image_path)

        if not img_file.exists():
            print(f"Error: File not found - {image_path}")
            sys.exit(1)

        if img_file.suffix.lower() not in image_extensions:
            print(f"Error: Not a supported image file - {image_path}")
            sys.exit(1)

        image_files.append(img_file)

    # Process header
    print(f"Auto Crop Processing - Margin Detection")
    print("=" * 60)
    if len(image_files) > 1:
        print(f"Processing {len(image_files)} images")
    print("-" * 60)

    # Collect results for JSON output
    all_results = []
    success_count = 0
    error_count = 0

    # Determine output directory from first image path
    output_dir = image_files[0].parent if image_files else Path.cwd()

    for idx, img_file in enumerate(image_files, 1):
        if len(image_files) > 1:
            print(f"\n[{idx}/{len(image_files)}]")

        print(f"Processing: {img_file.name}")
        print(f"Path: {img_file}")
        print(f"Size: {img_file.stat().st_size:,} bytes")

        # Detect margins and calculate crop percentages
        result, error = detect_content_bounds(img_file, save_visualization=save_visualization)

        if error:
            print(f"Error: {error}")
            error_count += 1

            # Add error result to JSON
            all_results.append({
                "filename": img_file.name,
                "status": "error",
                "error": error
            })
        else:
            print(f"Image dimensions: {result['image_size'][0]}x{result['image_size'][1]} px")
            print(f"Content region: x={result['detected_region'][0]}, y={result['detected_region'][1]}, "
                  f"w={result['detected_region'][2]}, h={result['detected_region'][3]}")
            print(f"\nCrop percentages:")
            print(f"  Crop left:   {result['crop_left']:6.2f}%  (confidence: {result['confidence_left']:.2f})")
            print(f"  Crop top:    {result['crop_top']:6.2f}%  (confidence: {result['confidence_top']:.2f})")
            print(f"  Crop right:  {result['crop_right']:6.2f}%  (confidence: {result['confidence_right']:.2f})")
            print(f"  Crop bottom: {result['crop_bottom']:6.2f}%  (confidence: {result['confidence_bottom']:.2f})")

            if result['visualization_path']:
                print(f"\nVisualization saved: {result['visualization_path'].name}")

            success_count += 1

            # Add success result to JSON (filename without extension for Lua matching)
            filename_base = img_file.stem  # filename without extension
            all_results.append({
                "filename": filename_base,
                "status": "success",
                "crop": {
                    "left": round(result['crop_left'], 2),
                    "top": round(result['crop_top'], 2),
                    "right": round(result['crop_right'], 2),
                    "bottom": round(result['crop_bottom'], 2)
                },
                "confidence": {
                    "left": round(result['confidence_left'], 2),
                    "top": round(result['confidence_top'], 2),
                    "right": round(result['confidence_right'], 2),
                    "bottom": round(result['confidence_bottom'], 2)
                }
            })

    # Summary
    print("\n" + "=" * 60)
    if len(image_files) > 1:
        print(f"Processing complete: {success_count} succeeded, {error_count} failed")
    else:
        print("Processing complete")

    # Write crop results file
    results_output_path = output_dir / "crop_results.txt"
    write_crop_results(all_results, results_output_path)

    # Exit with error code if any files failed
    if error_count > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
