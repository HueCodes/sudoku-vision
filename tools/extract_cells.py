#!/usr/bin/env python3
"""
Batch cell extraction tool for sudoku images.

Processes a directory of sudoku photos and extracts all 81 cells per image
using the existing CV pipeline.

Usage:
    python tools/extract_cells.py <input_dir> [--output-dir <output_dir>]
    python tools/extract_cells.py data/test_images --output-dir data/raw
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cv.preprocess import preprocess_for_grid_detection
from cv.grid import find_grid_contour, warp_perspective
from cv.extract import extract_cells, is_cell_empty


def process_image(
    image_path: Path,
    output_dir: Path,
    cell_size: int = 28,
) -> dict:
    """Process a single sudoku image and extract cells.

    Args:
        image_path: Path to sudoku image
        output_dir: Directory to save extracted cells
        cell_size: Size of output cell images

    Returns:
        dict with extraction results and metadata
    """
    result = {
        "source_image": str(image_path),
        "timestamp": datetime.now().isoformat(),
        "success": False,
        "cells_extracted": 0,
        "empty_cells": 0,
        "error": None,
        "cells": [],
    }

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        result["error"] = f"Could not load image: {image_path}"
        return result

    # Store original dimensions
    result["image_height"] = image.shape[0]
    result["image_width"] = image.shape[1]

    # Preprocess for grid detection
    try:
        binary = preprocess_for_grid_detection(image)
    except Exception as e:
        result["error"] = f"Preprocessing failed: {e}"
        return result

    # Find grid
    corners = find_grid_contour(binary)
    if corners is None:
        result["error"] = "Could not detect grid in image"
        return result

    result["grid_corners"] = corners.tolist()

    # Warp perspective
    try:
        warped = warp_perspective(image, corners, output_size=450)
    except Exception as e:
        result["error"] = f"Perspective warp failed: {e}"
        return result

    # Extract cells
    try:
        cells = extract_cells(warped, cell_size=cell_size)
    except Exception as e:
        result["error"] = f"Cell extraction failed: {e}"
        return result

    # Create output directory for this image
    image_name = image_path.stem
    image_output_dir = output_dir / image_name
    image_output_dir.mkdir(parents=True, exist_ok=True)

    # Save warped grid for reference
    warped_path = image_output_dir / "warped_grid.png"
    cv2.imwrite(str(warped_path), warped)

    # Save each cell
    empty_count = 0
    for i, cell in enumerate(cells):
        row, col = i // 9, i % 9
        cell_filename = f"cell_{row}_{col}.png"
        cell_path = image_output_dir / cell_filename

        cv2.imwrite(str(cell_path), cell)

        is_empty = is_cell_empty(cell)
        if is_empty:
            empty_count += 1

        cell_info = {
            "filename": cell_filename,
            "row": row,
            "col": col,
            "index": i,
            "is_empty_guess": is_empty,
            "label": None,  # To be filled by labeling tool
        }
        result["cells"].append(cell_info)

    result["success"] = True
    result["cells_extracted"] = len(cells)
    result["empty_cells"] = empty_count
    result["output_dir"] = str(image_output_dir)

    return result


def process_directory(
    input_dir: Path,
    output_dir: Path,
    resume: bool = True,
) -> dict:
    """Process all images in a directory.

    Args:
        input_dir: Directory containing sudoku images
        output_dir: Directory to save extracted cells
        resume: Skip already processed images

    Returns:
        dict with overall extraction results
    """
    # Find all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"No image files found in {input_dir}")
        return {"error": "No images found", "processed": 0}

    print(f"Found {len(image_files)} images to process")

    # Load existing manifest if resuming
    manifest_path = output_dir / "manifest.json"
    if resume and manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        processed_sources = {r["source_image"] for r in manifest.get("results", [])}
    else:
        manifest = {
            "created": datetime.now().isoformat(),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "results": [],
        }
        processed_sources = set()

    # Process each image
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, image_path in enumerate(sorted(image_files)):
        # Skip if already processed
        if str(image_path) in processed_sources:
            print(f"[{i+1}/{len(image_files)}] Skipping (already processed): {image_path.name}")
            skip_count += 1
            continue

        print(f"[{i+1}/{len(image_files)}] Processing: {image_path.name}...", end=" ")

        result = process_image(image_path, output_dir)
        manifest["results"].append(result)

        if result["success"]:
            print(f"OK ({result['cells_extracted']} cells, {result['empty_cells']} empty)")
            success_count += 1
        else:
            print(f"FAILED: {result['error']}")
            fail_count += 1

        # Save manifest after each image (for resume capability)
        manifest["last_updated"] = datetime.now().isoformat()
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    # Summary
    print("\n" + "=" * 50)
    print("Extraction Summary:")
    print(f"  Total images: {len(image_files)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Manifest saved to: {manifest_path}")

    return manifest


def create_labeling_manifest(manifest_path: Path) -> Path:
    """Create a simplified manifest for the labeling tool.

    Args:
        manifest_path: Path to extraction manifest

    Returns:
        Path to labeling manifest
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    output_dir = Path(manifest["output_dir"])
    labeling_manifest = {
        "created": datetime.now().isoformat(),
        "source_manifest": str(manifest_path),
        "cells_to_label": [],
    }

    for result in manifest["results"]:
        if not result["success"]:
            continue

        image_dir = Path(result["output_dir"])
        for cell in result["cells"]:
            cell_path = image_dir / cell["filename"]
            if cell_path.exists():
                labeling_manifest["cells_to_label"].append({
                    "path": str(cell_path),
                    "source_image": result["source_image"],
                    "row": cell["row"],
                    "col": cell["col"],
                    "is_empty_guess": cell["is_empty_guess"],
                    "label": None,
                })

    labeling_path = output_dir / "to_label.json"
    with open(labeling_path, "w") as f:
        json.dump(labeling_manifest, f, indent=2)

    print(f"\nLabeling manifest created: {labeling_path}")
    print(f"  Total cells to label: {len(labeling_manifest['cells_to_label'])}")

    return labeling_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract cells from sudoku images for training data"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing sudoku images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to save extracted cells (default: data/raw)",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=28,
        help="Size of extracted cell images (default: 28)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't skip already processed images",
    )
    parser.add_argument(
        "--create-labeling-manifest",
        action="store_true",
        help="Create manifest for labeling tool after extraction",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    manifest = process_directory(
        args.input_dir,
        args.output_dir,
        resume=not args.no_resume,
    )

    if args.create_labeling_manifest and "error" not in manifest:
        manifest_path = args.output_dir / "manifest.json"
        create_labeling_manifest(manifest_path)


if __name__ == "__main__":
    main()
