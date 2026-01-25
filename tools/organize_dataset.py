#!/usr/bin/env python3
"""
Dataset organization script.

Consolidates labeled data and creates proper train/val/test splits
with stratified sampling to maintain class balance.

Usage:
    python tools/organize_dataset.py data/raw/labels.csv --output data/labeled
    python tools/organize_dataset.py data/real --output data/labeled
"""

import argparse
import csv
import json
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2


def load_labels_from_csv(csv_path: Path) -> list[dict]:
    """Load labels from a CSV file."""
    labels = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append({
                "path": row["path"],
                "label": int(row["label"]),
                "source": row.get("source_image", ""),
            })
    return labels


def load_labels_from_directory(data_dir: Path) -> list[dict]:
    """Load labels from existing class directory structure."""
    labels = []

    for class_dir in data_dir.iterdir():
        if not class_dir.is_dir():
            continue

        try:
            label = int(class_dir.name)
        except ValueError:
            continue

        for img_file in class_dir.glob("*.png"):
            labels.append({
                "path": str(img_file),
                "label": label,
                "source": "",
            })

    return labels


def load_all_labels(sources: list[Path]) -> list[dict]:
    """Load labels from multiple sources (CSVs or directories)."""
    all_labels = []

    for source in sources:
        if source.is_file() and source.suffix == ".csv":
            labels = load_labels_from_csv(source)
            print(f"  Loaded {len(labels)} labels from {source}")
            all_labels.extend(labels)
        elif source.is_dir():
            # Check for CSV files in directory
            csv_files = list(source.glob("*.csv"))
            if csv_files:
                for csv_file in csv_files:
                    labels = load_labels_from_csv(csv_file)
                    print(f"  Loaded {len(labels)} labels from {csv_file}")
                    all_labels.extend(labels)
            else:
                # Try class directory structure
                labels = load_labels_from_directory(source)
                print(f"  Loaded {len(labels)} labels from {source}")
                all_labels.extend(labels)

    return all_labels


def stratified_split(
    labels: list[dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split labels into train/val/test with stratification by class.

    Args:
        labels: List of label dicts
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for test
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_labels, val_labels, test_labels)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01

    random.seed(seed)

    # Group by class
    by_class = defaultdict(list)
    for item in labels:
        by_class[item["label"]].append(item)

    train = []
    val = []
    test = []

    for class_label, items in by_class.items():
        random.shuffle(items)

        n = len(items)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        # Rest goes to test

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    return train, val, test


def copy_files_to_split(
    labels: list[dict],
    output_dir: Path,
    split_name: str,
) -> dict:
    """Copy files to split directory organized by class.

    Args:
        labels: List of label dicts
        output_dir: Base output directory
        split_name: Split name (train/val/test)

    Returns:
        dict with copy statistics
    """
    split_dir = output_dir / split_name
    stats = {
        "copied": 0,
        "failed": 0,
        "by_class": defaultdict(int),
    }

    for item in labels:
        src_path = Path(item["path"])
        if not src_path.exists():
            print(f"  Warning: File not found: {src_path}")
            stats["failed"] += 1
            continue

        # Destination: output_dir/split/class/filename.png
        class_dir = split_dir / str(item["label"])
        class_dir.mkdir(parents=True, exist_ok=True)

        # Create unique filename
        dst_name = f"{src_path.parent.name}_{src_path.name}"
        dst_path = class_dir / dst_name

        # Handle duplicates
        counter = 1
        while dst_path.exists():
            dst_name = f"{src_path.parent.name}_{counter}_{src_path.name}"
            dst_path = class_dir / dst_name
            counter += 1

        shutil.copy2(src_path, dst_path)
        stats["copied"] += 1
        stats["by_class"][item["label"]] += 1

    return stats


def create_split_manifests(
    train: list[dict],
    val: list[dict],
    test: list[dict],
    output_dir: Path,
) -> None:
    """Create CSV manifests for each split."""
    for split_name, labels in [("train", train), ("val", val), ("test", test)]:
        manifest_path = output_dir / f"{split_name}.csv"
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "label", "original_path"])
            writer.writeheader()

            for item in labels:
                src_path = Path(item["path"])
                dst_name = f"{src_path.parent.name}_{src_path.name}"
                writer.writerow({
                    "filename": dst_name,
                    "label": item["label"],
                    "original_path": item["path"],
                })


def print_split_summary(
    train: list[dict],
    val: list[dict],
    test: list[dict],
) -> None:
    """Print summary of split distribution."""
    print("\n" + "=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)

    for split_name, labels in [("Train", train), ("Val", val), ("Test", test)]:
        by_class = defaultdict(int)
        for item in labels:
            by_class[item["label"]] += 1

        print(f"\n{split_name}: {len(labels)} samples")
        for digit in range(10):
            count = by_class.get(digit, 0)
            pct = count / len(labels) * 100 if labels else 0
            label = "empty" if digit == 0 else f"  {digit}  "
            bar = "█" * min(count // 2, 20)
            print(f"  {label}: {bar} {count:4d} ({pct:5.1f}%)")


def verify_no_leakage(
    train: list[dict],
    val: list[dict],
    test: list[dict],
) -> bool:
    """Verify no data leakage between splits."""
    train_paths = {item["path"] for item in train}
    val_paths = {item["path"] for item in val}
    test_paths = {item["path"] for item in test}

    train_val = train_paths & val_paths
    train_test = train_paths & test_paths
    val_test = val_paths & test_paths

    if train_val or train_test or val_test:
        print("\n⚠️  WARNING: Data leakage detected!")
        if train_val:
            print(f"  Train-Val overlap: {len(train_val)} samples")
        if train_test:
            print(f"  Train-Test overlap: {len(train_test)} samples")
        if val_test:
            print(f"  Val-Test overlap: {len(val_test)} samples")
        return False

    print("\n✓ No data leakage detected")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Organize labeled data into train/val/test splits"
    )
    parser.add_argument(
        "sources",
        type=Path,
        nargs="+",
        help="CSV file(s) or directory(s) containing labels",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/labeled"),
        help="Output directory for organized dataset",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Fraction for training set (default: 0.70)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction for validation set (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction for test set (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without copying files",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing output directory before organizing",
    )

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"Error: Ratios must sum to 1.0 (got {total_ratio})")
        sys.exit(1)

    # Check sources exist
    for source in args.sources:
        if not source.exists():
            print(f"Error: Source does not exist: {source}")
            sys.exit(1)

    print("Loading labels...")
    all_labels = load_all_labels(args.sources)

    if not all_labels:
        print("Error: No labels found")
        sys.exit(1)

    print(f"\nTotal labels loaded: {len(all_labels)}")

    # Remove duplicates by path
    seen_paths = set()
    unique_labels = []
    for item in all_labels:
        if item["path"] not in seen_paths:
            seen_paths.add(item["path"])
            unique_labels.append(item)

    if len(unique_labels) < len(all_labels):
        print(f"Removed {len(all_labels) - len(unique_labels)} duplicates")
        all_labels = unique_labels

    # Split
    print("\nCreating stratified splits...")
    train, val, test = stratified_split(
        all_labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # Verify no leakage
    verify_no_leakage(train, val, test)

    # Print summary
    print_split_summary(train, val, test)

    if args.dry_run:
        print("\n[DRY RUN - no files copied]")
        return

    # Clean output directory if requested
    if args.clean and args.output.exists():
        print(f"\nRemoving existing output directory: {args.output}")
        shutil.rmtree(args.output)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Copy files
    print("\nCopying files...")
    for split_name, labels in [("train", train), ("val", val), ("test", test)]:
        print(f"\n  {split_name}:")
        stats = copy_files_to_split(labels, args.output, split_name)
        print(f"    Copied: {stats['copied']}")
        if stats["failed"]:
            print(f"    Failed: {stats['failed']}")

    # Create manifests
    print("\nCreating manifests...")
    create_split_manifests(train, val, test, args.output)

    # Save metadata
    metadata = {
        "created": datetime.now().isoformat(),
        "sources": [str(s) for s in args.sources],
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "total_samples": len(all_labels),
        "train_samples": len(train),
        "val_samples": len(val),
        "test_samples": len(test),
    }

    with open(args.output / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("ORGANIZATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {args.output}")
    print(f"  train/: {len(train)} samples")
    print(f"  val/: {len(val)} samples")
    print(f"  test/: {len(test)} samples")
    print(f"  train.csv, val.csv, test.csv: manifests")
    print(f"  metadata.json: split configuration")


if __name__ == "__main__":
    main()
