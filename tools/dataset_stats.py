#!/usr/bin/env python3
"""
Dataset statistics tool.

Analyzes the current state of training data, showing class distribution,
identifying issues, and generating reports.

Usage:
    python tools/dataset_stats.py data/labeled
    python tools/dataset_stats.py data/raw --manifest manifest.json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def analyze_directory_structure(data_dir: Path) -> dict:
    """Analyze a labeled data directory organized by class.

    Expected structure:
        data_dir/
            0/  (empty cells)
            1/
            ...
            9/
    """
    stats = {
        "type": "class_directory",
        "path": str(data_dir),
        "total_images": 0,
        "class_distribution": {},
        "issues": [],
    }

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        try:
            class_label = int(class_dir.name)
            if class_label < 0 or class_label > 9:
                stats["issues"].append(f"Invalid class directory: {class_dir.name}")
                continue
        except ValueError:
            if class_dir.name not in {"train", "val", "test"}:
                stats["issues"].append(f"Non-numeric directory: {class_dir.name}")
            continue

        # Count images in this class
        image_count = 0
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                image_count += 1

        stats["class_distribution"][str(class_label)] = image_count
        stats["total_images"] += image_count

    return stats


def analyze_manifest(manifest_path: Path) -> dict:
    """Analyze extraction manifest for labeling status."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    stats = {
        "type": "manifest",
        "path": str(manifest_path),
        "total_images_processed": 0,
        "successful_extractions": 0,
        "failed_extractions": 0,
        "total_cells": 0,
        "labeled_cells": 0,
        "unlabeled_cells": 0,
        "class_distribution": Counter(),
        "empty_guesses": 0,
        "errors": [],
    }

    for result in manifest.get("results", []):
        stats["total_images_processed"] += 1

        if result.get("success"):
            stats["successful_extractions"] += 1
            stats["total_cells"] += result.get("cells_extracted", 0)

            for cell in result.get("cells", []):
                if cell.get("is_empty_guess"):
                    stats["empty_guesses"] += 1

                label = cell.get("label")
                if label is not None:
                    stats["labeled_cells"] += 1
                    stats["class_distribution"][str(label)] += 1
                else:
                    stats["unlabeled_cells"] += 1
        else:
            stats["failed_extractions"] += 1
            stats["errors"].append({
                "image": result.get("source_image"),
                "error": result.get("error"),
            })

    stats["class_distribution"] = dict(stats["class_distribution"])
    return stats


def analyze_csv_labels(label_dir: Path) -> dict:
    """Analyze CSV label files in a directory."""
    stats = {
        "type": "csv_labels",
        "path": str(label_dir),
        "csv_files": [],
        "total_labeled": 0,
        "class_distribution": Counter(),
    }

    for csv_file in label_dir.glob("*.csv"):
        file_stats = {
            "name": csv_file.name,
            "rows": 0,
            "distribution": Counter(),
        }

        with open(csv_file) as f:
            # Skip header if present
            first_line = f.readline().strip()
            if not first_line[0].isdigit() and "," in first_line:
                pass  # Was header, continue
            else:
                # Not a header, process this line
                parts = first_line.split(",")
                if len(parts) >= 2:
                    try:
                        label = int(parts[1].strip())
                        file_stats["distribution"][label] += 1
                        file_stats["rows"] += 1
                    except ValueError:
                        pass

            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    try:
                        label = int(parts[1].strip())
                        file_stats["distribution"][label] += 1
                        file_stats["rows"] += 1
                    except ValueError:
                        continue

        stats["csv_files"].append({
            "name": file_stats["name"],
            "rows": file_stats["rows"],
            "distribution": dict(file_stats["distribution"]),
        })
        stats["total_labeled"] += file_stats["rows"]
        stats["class_distribution"] += file_stats["distribution"]

    stats["class_distribution"] = dict(stats["class_distribution"])
    return stats


def analyze_split_structure(data_dir: Path) -> dict:
    """Analyze train/val/test split structure."""
    stats = {
        "type": "split_structure",
        "path": str(data_dir),
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if split_dir.exists():
            split_stats = analyze_directory_structure(split_dir)
            stats["splits"][split] = {
                "total": split_stats["total_images"],
                "distribution": split_stats["class_distribution"],
            }

    return stats


def check_image_quality(data_dir: Path, sample_size: int = 100) -> dict:
    """Check quality of sample images."""
    stats = {
        "sample_size": 0,
        "avg_brightness": 0,
        "avg_contrast": 0,
        "size_issues": [],
        "corrupt_files": [],
    }

    image_files = list(data_dir.rglob("*.png")) + list(data_dir.rglob("*.jpg"))
    if not image_files:
        return stats

    # Sample images
    import random
    sample = random.sample(image_files, min(sample_size, len(image_files)))

    brightness_values = []
    contrast_values = []

    for img_path in sample:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            stats["corrupt_files"].append(str(img_path))
            continue

        stats["sample_size"] += 1

        # Check size
        if img.shape != (28, 28):
            stats["size_issues"].append({
                "path": str(img_path),
                "size": img.shape,
            })

        # Brightness (mean pixel value)
        brightness_values.append(np.mean(img))

        # Contrast (standard deviation)
        contrast_values.append(np.std(img))

    if brightness_values:
        stats["avg_brightness"] = float(np.mean(brightness_values))
        stats["avg_contrast"] = float(np.mean(contrast_values))

    return stats


def find_duplicates(data_dir: Path, threshold: float = 0.99) -> list:
    """Find potential duplicate images using hash comparison."""
    from hashlib import md5

    hashes = defaultdict(list)

    for img_path in data_dir.rglob("*.png"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Simple hash based on resized image
        small = cv2.resize(img, (8, 8))
        img_hash = md5(small.tobytes()).hexdigest()
        hashes[img_hash].append(str(img_path))

    # Find groups with duplicates
    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    return duplicates


def print_distribution_bar(distribution: dict, total: int, width: int = 40):
    """Print a horizontal bar chart of class distribution."""
    if not distribution or total == 0:
        print("  No data")
        return

    max_count = max(distribution.values()) if distribution.values() else 1

    for label in sorted(distribution.keys(), key=lambda x: int(x)):
        count = distribution[label]
        pct = count / total * 100
        bar_len = int(count / max_count * width)
        bar = "█" * bar_len

        label_name = "empty" if label == "0" else f"  {label}  "
        print(f"  {label_name}: {bar} {count:4d} ({pct:5.1f}%)")


def print_report(stats: dict, verbose: bool = False):
    """Print formatted statistics report."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS REPORT")
    print("=" * 60)

    if stats.get("type") == "class_directory":
        print(f"\nDirectory: {stats['path']}")
        print(f"Total images: {stats['total_images']}")
        print("\nClass Distribution:")
        print_distribution_bar(stats["class_distribution"], stats["total_images"])

        if stats.get("issues"):
            print("\nIssues:")
            for issue in stats["issues"]:
                print(f"  - {issue}")

    elif stats.get("type") == "manifest":
        print(f"\nManifest: {stats['path']}")
        print(f"Images processed: {stats['total_images_processed']}")
        print(f"  Successful: {stats['successful_extractions']}")
        print(f"  Failed: {stats['failed_extractions']}")
        print(f"\nTotal cells: {stats['total_cells']}")
        print(f"  Labeled: {stats['labeled_cells']}")
        print(f"  Unlabeled: {stats['unlabeled_cells']}")
        print(f"  Empty (guessed): {stats['empty_guesses']}")

        if stats["class_distribution"]:
            print("\nLabeled Class Distribution:")
            print_distribution_bar(stats["class_distribution"], stats["labeled_cells"])

        if verbose and stats.get("errors"):
            print("\nExtraction Errors:")
            for err in stats["errors"][:10]:
                print(f"  - {err['image']}: {err['error']}")
            if len(stats["errors"]) > 10:
                print(f"  ... and {len(stats['errors']) - 10} more")

    elif stats.get("type") == "csv_labels":
        print(f"\nLabel directory: {stats['path']}")
        print(f"Total labeled: {stats['total_labeled']}")

        for csv_info in stats["csv_files"]:
            print(f"\n  {csv_info['name']}: {csv_info['rows']} labels")

        if stats["class_distribution"]:
            print("\nOverall Class Distribution:")
            print_distribution_bar(stats["class_distribution"], stats["total_labeled"])

    elif stats.get("type") == "split_structure":
        print(f"\nDataset: {stats['path']}")

        for split_name, split_stats in stats["splits"].items():
            print(f"\n{split_name.upper()} ({split_stats['total']} images):")
            print_distribution_bar(split_stats["distribution"], split_stats["total"])

    # Check for class imbalance
    if "class_distribution" in stats and stats["class_distribution"]:
        dist = stats["class_distribution"]
        values = list(dist.values())
        if values:
            min_count = min(values)
            max_count = max(values)
            if max_count > 3 * min_count:
                print("\n⚠️  WARNING: Significant class imbalance detected!")
                print(f"   Min class has {min_count}, max has {max_count}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze dataset statistics"
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Data directory to analyze",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Analyze specific manifest file",
    )
    parser.add_argument(
        "--check-quality",
        action="store_true",
        help="Check image quality (samples images)",
    )
    parser.add_argument(
        "--find-duplicates",
        action="store_true",
        help="Find potential duplicate images",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed information",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Error: Directory does not exist: {args.data_dir}")
        sys.exit(1)

    all_stats = {}

    # Determine analysis type
    if args.manifest:
        stats = analyze_manifest(args.manifest)
        all_stats["manifest"] = stats
    elif (args.data_dir / "manifest.json").exists():
        stats = analyze_manifest(args.data_dir / "manifest.json")
        all_stats["manifest"] = stats
    elif (args.data_dir / "train").exists():
        stats = analyze_split_structure(args.data_dir)
        all_stats["splits"] = stats
    elif any(args.data_dir.glob("*.csv")):
        stats = analyze_csv_labels(args.data_dir)
        all_stats["csv_labels"] = stats
    else:
        stats = analyze_directory_structure(args.data_dir)
        all_stats["directory"] = stats

    # Optional quality check
    if args.check_quality:
        quality_stats = check_image_quality(args.data_dir)
        all_stats["quality"] = quality_stats

    # Optional duplicate check
    if args.find_duplicates:
        duplicates = find_duplicates(args.data_dir)
        all_stats["duplicates"] = duplicates

    # Output
    if args.json:
        print(json.dumps(all_stats, indent=2))
    else:
        for key, stat in all_stats.items():
            if key == "quality":
                print("\nImage Quality Analysis:")
                print(f"  Sample size: {stat['sample_size']}")
                print(f"  Avg brightness: {stat['avg_brightness']:.1f}")
                print(f"  Avg contrast: {stat['avg_contrast']:.1f}")
                if stat["size_issues"]:
                    print(f"  Size issues: {len(stat['size_issues'])}")
                if stat["corrupt_files"]:
                    print(f"  Corrupt files: {len(stat['corrupt_files'])}")
            elif key == "duplicates":
                if stat:
                    print(f"\nPotential Duplicates Found: {len(stat)} groups")
                    for i, group in enumerate(stat[:5]):
                        print(f"  Group {i+1}: {len(group)} files")
                        for path in group[:3]:
                            print(f"    - {path}")
                else:
                    print("\nNo duplicates found")
            else:
                print_report(stat, verbose=args.verbose)


if __name__ == "__main__":
    main()
