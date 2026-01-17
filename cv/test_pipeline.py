#!/usr/bin/env python3
"""
CV Pipeline Validation Script

Tests the computer vision pipeline on sample sudoku images and generates
debug visualizations to help identify failure modes.

Usage:
    python test_pipeline.py [--images-dir PATH] [--output-dir PATH]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

# Import CV modules
from preprocess import preprocess_for_grid_detection, grayscale, blur, threshold
from grid import find_contours, find_grid_contour, warp_perspective, order_points
from extract import extract_cells, is_cell_empty


class PipelineResult:
    """Results from running the pipeline on a single image."""

    def __init__(self, image_path: Path):
        self.image_path = image_path
        self.image_name = image_path.stem
        self.success = False
        self.error: str | None = None

        # Pipeline outputs
        self.original: NDArray | None = None
        self.preprocessed: NDArray | None = None
        self.contours: list | None = None
        self.grid_corners: NDArray | None = None
        self.warped: NDArray | None = None
        self.cells: list[NDArray] | None = None
        self.empty_mask: list[bool] | None = None

    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else f"FAILED: {self.error}"
        return f"PipelineResult({self.image_name}: {status})"


def draw_contours_debug(
    image: NDArray,
    contours: list,
    grid_corners: NDArray | None,
) -> NDArray:
    """Draw contours and detected grid on image for debugging."""
    debug = image.copy()
    if len(debug.shape) == 2:
        debug = cv2.cvtColor(debug, cv2.COLOR_GRAY2BGR)

    # Draw all contours in blue
    cv2.drawContours(debug, contours, -1, (255, 100, 0), 1)

    # Draw top 5 largest contours in different colors
    colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 128, 255)]
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for i, cnt in enumerate(sorted_contours):
        cv2.drawContours(debug, [cnt], -1, colors[i % len(colors)], 2)
        area = cv2.contourArea(cnt)
        # Label with area
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(debug, f"#{i+1}: {area:.0f}", (cx-30, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i % len(colors)], 1)

    # Highlight detected grid in red
    if grid_corners is not None:
        pts = grid_corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(debug, [pts], True, (0, 0, 255), 3)
        # Draw corner points
        for i, pt in enumerate(grid_corners):
            cv2.circle(debug, tuple(pt.astype(int)), 8, (0, 0, 255), -1)
            cv2.putText(debug, str(i), tuple(pt.astype(int) + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return debug


def draw_cells_grid(cells: list[NDArray], empty_mask: list[bool]) -> NDArray:
    """Create a grid visualization of extracted cells."""
    cell_size = 28
    gap = 2
    grid_size = cell_size * 9 + gap * 10

    grid = np.ones((grid_size, grid_size), dtype=np.uint8) * 128

    for i, (cell, is_empty) in enumerate(zip(cells, empty_mask)):
        row, col = i // 9, i % 9
        y = gap + row * (cell_size + gap)
        x = gap + col * (cell_size + gap)

        # Resize cell if needed
        if cell.shape != (cell_size, cell_size):
            cell = cv2.resize(cell, (cell_size, cell_size))

        grid[y:y+cell_size, x:x+cell_size] = cell

        # Mark empty cells with a small dot
        if is_empty:
            cv2.circle(grid, (x + cell_size//2, y + cell_size//2), 2, 200, -1)

    return grid


def run_pipeline(image_path: Path) -> PipelineResult:
    """Run the full CV pipeline on a single image."""
    result = PipelineResult(image_path)

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        result.error = "Failed to load image"
        return result
    result.original = image

    # Preprocessing
    try:
        result.preprocessed = preprocess_for_grid_detection(image)
    except Exception as e:
        result.error = f"Preprocessing failed: {e}"
        return result

    # Find contours
    try:
        result.contours = find_contours(result.preprocessed)
        if not result.contours:
            result.error = "No contours found"
            return result
    except Exception as e:
        result.error = f"Contour detection failed: {e}"
        return result

    # Find grid
    try:
        result.grid_corners = find_grid_contour(result.preprocessed)
        if result.grid_corners is None:
            result.error = "No quadrilateral grid found"
            return result
    except Exception as e:
        result.error = f"Grid detection failed: {e}"
        return result

    # Warp perspective
    try:
        result.warped = warp_perspective(image, result.grid_corners)
    except Exception as e:
        result.error = f"Perspective warp failed: {e}"
        return result

    # Extract cells
    try:
        result.cells = extract_cells(result.warped)
        if len(result.cells) != 81:
            result.error = f"Expected 81 cells, got {len(result.cells)}"
            return result
    except Exception as e:
        result.error = f"Cell extraction failed: {e}"
        return result

    # Check empty cells
    try:
        result.empty_mask = [is_cell_empty(cell) for cell in result.cells]
    except Exception as e:
        result.error = f"Empty cell detection failed: {e}"
        return result

    result.success = True
    return result


def save_debug_output(result: PipelineResult, output_dir: Path) -> None:
    """Save debug visualizations for a pipeline result."""
    img_dir = output_dir / result.image_name
    img_dir.mkdir(parents=True, exist_ok=True)

    # Save preprocessed
    if result.preprocessed is not None:
        cv2.imwrite(str(img_dir / "1_preprocessed.png"), result.preprocessed)

    # Save contours visualization
    if result.contours is not None and result.original is not None:
        contours_debug = draw_contours_debug(
            result.original, result.contours, result.grid_corners
        )
        cv2.imwrite(str(img_dir / "2_contours.png"), contours_debug)

    # Save grid corners visualization
    if result.grid_corners is not None and result.original is not None:
        grid_debug = result.original.copy()
        pts = result.grid_corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(grid_debug, [pts], True, (0, 255, 0), 3)
        cv2.imwrite(str(img_dir / "3_grid.png"), grid_debug)

    # Save warped
    if result.warped is not None:
        cv2.imwrite(str(img_dir / "4_warped.png"), result.warped)

    # Save cells grid
    if result.cells is not None and result.empty_mask is not None:
        cells_grid = draw_cells_grid(result.cells, result.empty_mask)
        cv2.imwrite(str(img_dir / "5_cells.png"), cells_grid)

        # Also save individual cells
        cells_dir = img_dir / "cells"
        cells_dir.mkdir(exist_ok=True)
        for i, cell in enumerate(result.cells):
            row, col = i // 9, i % 9
            cv2.imwrite(str(cells_dir / f"cell_{row}_{col}.png"), cell)


def print_report(results: list[PipelineResult]) -> None:
    """Print a summary report of pipeline results."""
    print("\n" + "=" * 60)
    print("CV PIPELINE VALIDATION REPORT")
    print("=" * 60)

    total = len(results)
    successes = sum(1 for r in results if r.success)

    print(f"\nOverall: {successes}/{total} images processed successfully")
    print(f"Success rate: {100 * successes / total:.1f}%")

    print("\n" + "-" * 60)
    print("Per-Image Results:")
    print("-" * 60)

    for result in results:
        status = "OK" if result.success else "FAIL"
        print(f"\n  [{status}] {result.image_name}")

        if result.success:
            empty_count = sum(result.empty_mask)
            filled_count = 81 - empty_count
            print(f"       Grid detected: Yes")
            print(f"       Cells extracted: 81")
            print(f"       Empty cells: {empty_count}")
            print(f"       Filled cells: {filled_count}")
        else:
            print(f"       Error: {result.error}")

    # Failure analysis
    failures = [r for r in results if not r.success]
    if failures:
        print("\n" + "-" * 60)
        print("Failure Analysis:")
        print("-" * 60)

        error_types: dict[str, list[str]] = {}
        for r in failures:
            error_key = r.error.split(":")[0] if r.error else "Unknown"
            if error_key not in error_types:
                error_types[error_key] = []
            error_types[error_key].append(r.image_name)

        for error, images in error_types.items():
            print(f"\n  {error}:")
            for img in images:
                print(f"    - {img}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="CV Pipeline Validation")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "test_images",
        help="Directory containing test images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "debug_output",
        help="Directory to save debug output",
    )
    args = parser.parse_args()

    # Find test images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted([
        p for p in args.images_dir.iterdir()
        if p.suffix.lower() in image_extensions and not p.name.endswith("_warped.jpg")
    ])

    if not images:
        print(f"No images found in {args.images_dir}")
        sys.exit(1)

    print(f"Found {len(images)} test images in {args.images_dir}")
    print(f"Debug output will be saved to {args.output_dir}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run pipeline on each image
    results = []
    for image_path in images:
        print(f"\nProcessing {image_path.name}...")
        result = run_pipeline(image_path)
        results.append(result)

        # Save debug output regardless of success
        save_debug_output(result, args.output_dir)

        if result.success:
            print(f"  Success - extracted {81 - sum(result.empty_mask)} filled cells")
        else:
            print(f"  Failed: {result.error}")

    # Print report
    print_report(results)

    # Exit with error if any failures
    if any(not r.success for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
