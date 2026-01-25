#!/usr/bin/env python3
"""
Prepare real labeled data for training.

Reads labeled_cells.json and organizes images into class directories.
"""

import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    # Load labels
    labels_path = PROJECT_ROOT / "data" / "real" / "labeled_cells.json"
    with open(labels_path) as f:
        data = json.load(f)

    print(f"Processing {data['labeled_cells']} labeled cells...")

    # Create output directories
    output_dir = PROJECT_ROOT / "data" / "real" / "organized"
    for i in range(10):
        (output_dir / str(i)).mkdir(parents=True, exist_ok=True)

    # Copy images to class directories
    counts = {i: 0 for i in range(10)}
    for cell in data["cells"]:
        if cell["label"] is None:
            continue

        label = cell["label"]
        src = PROJECT_ROOT / cell["path"]

        if not src.exists():
            print(f"  Warning: {src} not found")
            continue

        # Create unique filename
        source_name = cell["source_image"].split("/")[-1].replace(".jpg", "")
        dst_name = f"{source_name}_r{cell['row']}_c{cell['col']}.png"
        dst = output_dir / str(label) / dst_name

        shutil.copy(src, dst)
        counts[label] += 1

    print("\nOrganized into class directories:")
    print("-" * 30)
    for i in range(10):
        class_name = "empty" if i == 0 else str(i)
        print(f"  {class_name:>5}: {counts[i]:>4} images")
    print("-" * 30)
    print(f"  Total: {sum(counts.values()):>4} images")
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
