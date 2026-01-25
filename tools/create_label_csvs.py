#!/usr/bin/env python3
"""
Convert labeled_cells.json to CSV format for RealDataset.
"""

import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    # Load labels
    labels_path = PROJECT_ROOT / "data" / "real" / "labeled_cells.json"
    with open(labels_path) as f:
        data = json.load(f)

    # Group by source image
    by_source = defaultdict(list)
    for cell in data["cells"]:
        if cell["label"] is None:
            continue

        # Extract source name (e.g., "new_test" from "data/test_images/new_test.jpg")
        source = cell["source_image"].split("/")[-1].replace(".jpg", "")
        filename = f"cell_{cell['row']}_{cell['col']}.png"
        by_source[source].append((filename, cell["label"]))

    print("Creating CSV files...")
    print("-" * 40)

    real_dir = PROJECT_ROOT / "data" / "real"

    for source, cells in by_source.items():
        csv_path = real_dir / f"labels_{source}.csv"

        # Check if the corresponding image folder exists in raw
        raw_dir = PROJECT_ROOT / "data" / "raw" / source

        if not raw_dir.exists():
            print(f"  Skipping {source}: no raw folder at {raw_dir}")
            continue

        # Create symlink if needed from real/ to raw/
        target_dir = real_dir / source
        if not target_dir.exists():
            target_dir.symlink_to(raw_dir)
            print(f"  Created symlink: {source} -> {raw_dir}")

        # Write CSV
        with open(csv_path, "w") as f:
            f.write("filename,label\n")
            for filename, label in sorted(cells):
                f.write(f"{filename},{label}\n")

        print(f"  {source}: {len(cells)} cells -> {csv_path.name}")

    print("-" * 40)
    print("Done! Ready for training with --dataset real")


if __name__ == "__main__":
    main()
