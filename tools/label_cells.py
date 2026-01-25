#!/usr/bin/env python3
"""
Interactive cell labeling tool.

Provides a CLI interface for rapidly labeling extracted sudoku cells.
Supports keyboard input, progress tracking, and incremental saves.

Usage:
    python tools/label_cells.py data/raw/to_label.json
    python tools/label_cells.py data/raw --auto-find
"""

import argparse
import csv
import json
import os
import sys
import termios
import tty
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def get_single_char() -> str:
    """Read a single character from stdin without echo."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def display_cell_ascii(cell_path: Path, width: int = 28) -> None:
    """Display cell image as ASCII art in terminal."""
    img = cv2.imread(str(cell_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("  [Could not load image]")
        return

    # Resize for display (half height because terminal chars are taller than wide)
    display_height = 14
    display_width = width
    img = cv2.resize(img, (display_width, display_height))

    # ASCII characters from dark to light
    chars = " .:-=+*#%@"

    for row in img:
        line = ""
        for pixel in row:
            # Map pixel value to character
            idx = int(pixel / 255 * (len(chars) - 1))
            line += chars[idx] * 2  # Double width for aspect ratio
        print(f"  {line}")


def display_cell_iterm2(cell_path: Path) -> bool:
    """Display cell image inline using iTerm2 protocol."""
    try:
        import base64

        with open(cell_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()

        # iTerm2 inline image protocol
        print(f"\033]1337;File=inline=1;width=10;preserveAspectRatio=1:{data}\a")
        return True
    except Exception:
        return False


def display_cell_sixel(cell_path: Path) -> bool:
    """Display cell image using Sixel graphics (for supported terminals)."""
    try:
        # Try to use img2sixel if available
        import subprocess
        result = subprocess.run(
            ["img2sixel", "-w", "100", str(cell_path)],
            capture_output=True,
            timeout=2,
        )
        if result.returncode == 0:
            print(result.stdout.decode())
            return True
    except Exception:
        pass
    return False


def display_cell(cell_path: Path, method: str = "auto") -> None:
    """Display cell image using best available method."""
    if method == "auto":
        # Try iTerm2 first, then sixel, fall back to ASCII
        if os.environ.get("TERM_PROGRAM") == "iTerm.app":
            if display_cell_iterm2(cell_path):
                return
        if display_cell_sixel(cell_path):
            return
        display_cell_ascii(cell_path)
    elif method == "iterm2":
        if not display_cell_iterm2(cell_path):
            display_cell_ascii(cell_path)
    elif method == "sixel":
        if not display_cell_sixel(cell_path):
            display_cell_ascii(cell_path)
    else:
        display_cell_ascii(cell_path)


class LabelingSession:
    """Manages an interactive labeling session."""

    def __init__(
        self,
        cells: list[dict],
        output_path: Path,
        display_method: str = "auto",
    ):
        self.cells = cells
        self.output_path = output_path
        self.display_method = display_method
        self.current_index = 0
        self.labels = {}  # path -> label
        self.history = []  # For undo

        # Load existing progress
        self._load_progress()

    def _load_progress(self):
        """Load existing labels from output file."""
        if self.output_path.exists():
            with open(self.output_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.labels[row["path"]] = {
                        "label": int(row["label"]),
                        "timestamp": row.get("timestamp", ""),
                    }

            # Find first unlabeled cell
            for i, cell in enumerate(self.cells):
                if cell["path"] not in self.labels:
                    self.current_index = i
                    break
            else:
                self.current_index = len(self.cells)

    def _save_progress(self):
        """Save labels to CSV file."""
        with open(self.output_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["path", "label", "source_image", "row", "col", "timestamp"]
            )
            writer.writeheader()

            for cell in self.cells:
                if cell["path"] in self.labels:
                    label_info = self.labels[cell["path"]]
                    writer.writerow({
                        "path": cell["path"],
                        "label": label_info["label"],
                        "source_image": cell.get("source_image", ""),
                        "row": cell.get("row", ""),
                        "col": cell.get("col", ""),
                        "timestamp": label_info["timestamp"],
                    })

    def _get_stats(self) -> dict:
        """Get current labeling statistics."""
        labeled_count = len(self.labels)
        total_count = len(self.cells)

        distribution = {}
        for label_info in self.labels.values():
            label = str(label_info["label"])
            distribution[label] = distribution.get(label, 0) + 1

        return {
            "labeled": labeled_count,
            "total": total_count,
            "remaining": total_count - labeled_count,
            "progress_pct": labeled_count / total_count * 100 if total_count > 0 else 0,
            "distribution": distribution,
        }

    def _display_current(self):
        """Display current cell and prompt."""
        os.system("clear")

        cell = self.cells[self.current_index]
        stats = self._get_stats()

        # Header
        print("=" * 60)
        print("SUDOKU CELL LABELING TOOL")
        print("=" * 60)
        print(f"Progress: {stats['labeled']}/{stats['total']} ({stats['progress_pct']:.1f}%)")
        print(f"Remaining: {stats['remaining']}")
        print("-" * 60)

        # Distribution
        if stats["distribution"]:
            print("Labels so far:", end=" ")
            for digit in range(10):
                count = stats["distribution"].get(str(digit), 0)
                label = "E" if digit == 0 else str(digit)
                print(f"{label}:{count}", end=" ")
            print()
            print("-" * 60)

        # Cell info
        print(f"\nCell {self.current_index + 1} of {stats['total']}")
        print(f"Source: {Path(cell.get('source_image', 'unknown')).name}")
        print(f"Position: row {cell.get('row', '?')}, col {cell.get('col', '?')}")

        if cell.get("is_empty_guess"):
            print("(System guessed: EMPTY)")

        print()

        # Display image
        display_cell(Path(cell["path"]), self.display_method)

        print()
        print("-" * 60)
        print("Commands:")
        print("  0-9  = Label as digit (0 = empty cell)")
        print("  e    = Empty cell (same as 0)")
        print("  s    = Skip (don't label)")
        print("  b    = Back (previous cell)")
        print("  u    = Undo last label")
        print("  j    = Jump to cell number")
        print("  q    = Quit and save")
        print("-" * 60)
        print("Enter label: ", end="", flush=True)

    def _handle_input(self, key: str) -> Optional[str]:
        """Handle keyboard input. Returns 'quit' to exit, None to continue."""
        cell = self.cells[self.current_index]

        if key in "0123456789":
            # Label the cell
            label = int(key)
            self.labels[cell["path"]] = {
                "label": label,
                "timestamp": datetime.now().isoformat(),
            }
            self.history.append(("label", cell["path"], label))
            self._save_progress()
            self.current_index += 1

        elif key == "e":
            # Empty cell (same as 0)
            self.labels[cell["path"]] = {
                "label": 0,
                "timestamp": datetime.now().isoformat(),
            }
            self.history.append(("label", cell["path"], 0))
            self._save_progress()
            self.current_index += 1

        elif key == "s":
            # Skip
            self.current_index += 1

        elif key == "b":
            # Back
            if self.current_index > 0:
                self.current_index -= 1

        elif key == "u":
            # Undo
            if self.history:
                action, path, _ = self.history.pop()
                if action == "label" and path in self.labels:
                    del self.labels[path]
                    self._save_progress()
                    # Find index of this cell
                    for i, c in enumerate(self.cells):
                        if c["path"] == path:
                            self.current_index = i
                            break

        elif key == "j":
            # Jump
            print("\nEnter cell number: ", end="", flush=True)
            num_str = ""
            while True:
                ch = get_single_char()
                if ch == "\r" or ch == "\n":
                    break
                if ch.isdigit():
                    num_str += ch
                    print(ch, end="", flush=True)
                elif ch == "\x7f":  # Backspace
                    if num_str:
                        num_str = num_str[:-1]
                        print("\b \b", end="", flush=True)

            if num_str:
                try:
                    new_index = int(num_str) - 1
                    if 0 <= new_index < len(self.cells):
                        self.current_index = new_index
                except ValueError:
                    pass

        elif key == "q" or key == "\x03":  # q or Ctrl+C
            return "quit"

        return None

    def run(self):
        """Run the interactive labeling session."""
        print("Starting labeling session...")
        print(f"Output file: {self.output_path}")
        print(f"Total cells: {len(self.cells)}")
        print(f"Already labeled: {len(self.labels)}")
        print("\nPress any key to begin...")
        get_single_char()

        while self.current_index < len(self.cells):
            self._display_current()
            key = get_single_char()
            result = self._handle_input(key)

            if result == "quit":
                break

        # Final save and summary
        self._save_progress()
        os.system("clear")

        stats = self._get_stats()
        print("=" * 60)
        print("LABELING SESSION COMPLETE")
        print("=" * 60)
        print(f"Total labeled: {stats['labeled']}/{stats['total']}")
        print(f"Saved to: {self.output_path}")
        print("\nFinal distribution:")
        for digit in range(10):
            count = stats["distribution"].get(str(digit), 0)
            label = "empty" if digit == 0 else f"  {digit}  "
            bar = "█" * min(count // 2, 30)
            print(f"  {label}: {bar} {count}")


def load_cells_from_manifest(manifest_path: Path) -> list[dict]:
    """Load cells from a labeling manifest."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    return manifest.get("cells_to_label", [])


def load_cells_from_directory(data_dir: Path) -> list[dict]:
    """Load cells from a raw extraction directory."""
    cells = []

    for image_dir in sorted(data_dir.iterdir()):
        if not image_dir.is_dir():
            continue

        for cell_file in sorted(image_dir.glob("cell_*.png")):
            # Parse row/col from filename
            parts = cell_file.stem.split("_")
            row = int(parts[1]) if len(parts) > 1 else 0
            col = int(parts[2]) if len(parts) > 2 else 0

            cells.append({
                "path": str(cell_file),
                "source_image": image_dir.name,
                "row": row,
                "col": col,
            })

    return cells


def main():
    parser = argparse.ArgumentParser(
        description="Interactive cell labeling tool"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Manifest JSON file or directory with extracted cells",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output CSV file for labels (default: labels.csv in input dir)",
    )
    parser.add_argument(
        "--display",
        choices=["auto", "ascii", "iterm2", "sixel"],
        default="auto",
        help="Image display method",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomize cell order (helps avoid bias)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input does not exist: {args.input}")
        sys.exit(1)

    # Load cells
    if args.input.is_file() and args.input.suffix == ".json":
        cells = load_cells_from_manifest(args.input)
        default_output = args.input.parent / "labels.csv"
    else:
        cells = load_cells_from_directory(args.input)
        default_output = args.input / "labels.csv"

    if not cells:
        print("Error: No cells found to label")
        sys.exit(1)

    # Shuffle if requested
    if args.shuffle:
        import random
        random.shuffle(cells)

    # Output path
    output_path = args.output or default_output

    # Run labeling session
    session = LabelingSession(cells, output_path, args.display)
    session.run()


if __name__ == "__main__":
    main()
