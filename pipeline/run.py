#!/usr/bin/env python3
"""
Sudoku Vision Pipeline - Main Entry Point

Takes a sudoku puzzle image and outputs the solved puzzle.

Usage:
    python run.py <image_path> [--output <path>] [--debug] [--no-display]

Examples:
    python run.py ../data/test_images/sample_4.jpg
    python run.py ../data/test_images/sample_4.jpg --output solved.png
    python run.py ../data/test_images/sample_4.jpg --debug
"""

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "cv"))
sys.path.insert(0, str(PROJECT_ROOT / "ml"))

from preprocess import preprocess_for_grid_detection
from grid import find_grid_contour, warp_perspective
from extract import extract_cells
from model import DigitCNN


@dataclass
class CellPrediction:
    """Prediction result for a single cell."""
    row: int
    col: int
    digit: int  # 0 = empty, 1-9 = digit
    confidence: float
    is_original: bool = True  # True if from image, False if solved


@dataclass
class PipelineResult:
    """Result of running the full pipeline."""
    success: bool
    error: str | None = None

    # Timing (in seconds)
    time_cv: float = 0.0
    time_ml: float = 0.0
    time_solver: float = 0.0
    time_total: float = 0.0

    # Results
    original_image: np.ndarray | None = None
    warped_grid: np.ndarray | None = None
    cells: list[np.ndarray] = field(default_factory=list)
    predictions: list[CellPrediction] = field(default_factory=list)
    recognized_grid: list[list[int]] = field(default_factory=list)
    solution: list[list[int]] = field(default_factory=list)

    # Diagnostics
    low_confidence_cells: list[tuple[int, int, float]] = field(default_factory=list)
    constraint_violations: list[str] = field(default_factory=list)


def preprocess_cell(cell: np.ndarray) -> np.ndarray:
    """Preprocess a cell image for ML inference.

    Applies CLAHE and adaptive thresholding to enhance faint digits.
    """
    # Ensure grayscale
    if len(cell.shape) == 3:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    # Resize to 28x28
    if cell.shape != (28, 28):
        cell = cv2.resize(cell, (28, 28))

    # CLAHE: Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    cell = clahe.apply(cell)

    # Adaptive thresholding for cleaner binary image
    cell = cv2.adaptiveThreshold(
        cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return cell


def load_model() -> tuple[DigitCNN, torch.device]:
    """Load the trained digit recognition model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitCNN().to(device)

    model_path = PROJECT_ROOT / "ml" / "digit_cnn_v2.pt"
    if not model_path.exists():
        # Fallback to original model
        model_path = PROJECT_ROOT / "ml" / "digit_cnn.pt"

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    return model, device


def predict_cells(
    cells: list[np.ndarray],
    model: DigitCNN,
    device: torch.device,
) -> list[CellPrediction]:
    """Run digit recognition on all cells."""
    predictions = []

    for i, cell in enumerate(cells):
        row, col = i // 9, i % 9

        # Preprocess
        processed = preprocess_cell(cell)

        # Invert (white digit on black background)
        processed = 255 - processed

        # To tensor
        tensor = torch.from_numpy(processed).float().unsqueeze(0).unsqueeze(0) / 255.0

        # Normalize
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.to(device)

        # Inference
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()
            conf = probs[0, pred].item()

        predictions.append(CellPrediction(
            row=row,
            col=col,
            digit=pred,
            confidence=conf,
        ))

    return predictions


def build_grid(predictions: list[CellPrediction]) -> list[list[int]]:
    """Build 9x9 grid from predictions."""
    grid = [[0] * 9 for _ in range(9)]
    for pred in predictions:
        grid[pred.row][pred.col] = pred.digit
    return grid


def run_solver(grid: list[list[int]]) -> tuple[bool, list[list[int]]]:
    """Run the C solver on the grid."""
    solver_path = PROJECT_ROOT / "solver" / "sudoku_solver"

    if not solver_path.exists():
        raise FileNotFoundError(f"Solver not found at {solver_path}")

    # Create input string
    input_str = ""
    for row in grid:
        input_str += "".join(str(c) for c in row) + "\n"

    # Write to temp files
    temp_input = Path("/tmp/sudoku_pipeline_input.txt")
    temp_output = Path("/tmp/sudoku_pipeline_output.txt")
    temp_input.write_text(input_str)

    # Run solver
    result = subprocess.run(
        [str(solver_path), str(temp_input), "-o", str(temp_output)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return False, grid

    # Read solution
    if temp_output.exists():
        output = temp_output.read_text().strip()
        solution = []
        for line in output.split("\n"):
            if line:
                row = [int(c) for c in line if c.isdigit()]
                if len(row) == 9:
                    solution.append(row)
        if len(solution) == 9:
            return True, solution

    return False, grid


def check_constraints(grid: list[list[int]]) -> list[str]:
    """Check if grid violates sudoku constraints."""
    violations = []

    # Check rows
    for r in range(9):
        seen = {}
        for c in range(9):
            val = grid[r][c]
            if val > 0:
                if val in seen:
                    violations.append(f"Row {r+1}: duplicate {val} at columns {seen[val]+1} and {c+1}")
                seen[val] = c

    # Check columns
    for c in range(9):
        seen = {}
        for r in range(9):
            val = grid[r][c]
            if val > 0:
                if val in seen:
                    violations.append(f"Column {c+1}: duplicate {val} at rows {seen[val]+1} and {r+1}")
                seen[val] = r

    # Check 3x3 boxes
    for box_r in range(3):
        for box_c in range(3):
            seen = {}
            for r in range(box_r * 3, box_r * 3 + 3):
                for c in range(box_c * 3, box_c * 3 + 3):
                    val = grid[r][c]
                    if val > 0:
                        if val in seen:
                            violations.append(f"Box ({box_r+1},{box_c+1}): duplicate {val}")
                        seen[val] = (r, c)

    return violations


def run_pipeline(image_path: Path, debug: bool = False) -> PipelineResult:
    """Run the full pipeline on an image."""
    result = PipelineResult(success=False)
    start_total = time.time()

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        result.error = f"Failed to load image: {image_path}"
        return result
    result.original_image = image

    # === CV Pipeline ===
    start_cv = time.time()

    # Preprocessing
    try:
        binary = preprocess_for_grid_detection(image)
    except Exception as e:
        result.error = f"Preprocessing failed: {e}"
        return result

    # Grid detection
    try:
        corners = find_grid_contour(binary)
        if corners is None:
            result.error = "Grid detection failed: no quadrilateral found"
            return result
    except Exception as e:
        result.error = f"Grid detection failed: {e}"
        return result

    # Perspective warp
    try:
        warped = warp_perspective(image, corners)
        result.warped_grid = warped
    except Exception as e:
        result.error = f"Perspective warp failed: {e}"
        return result

    # Cell extraction
    try:
        cells = extract_cells(warped)
        if len(cells) != 81:
            result.error = f"Cell extraction failed: expected 81 cells, got {len(cells)}"
            return result
        result.cells = cells
    except Exception as e:
        result.error = f"Cell extraction failed: {e}"
        return result

    result.time_cv = time.time() - start_cv

    # === ML Pipeline ===
    start_ml = time.time()

    try:
        model, device = load_model()
        predictions = predict_cells(cells, model, device)
        result.predictions = predictions

        # Build grid
        grid = build_grid(predictions)
        result.recognized_grid = grid

        # Find low confidence cells
        for pred in predictions:
            if pred.digit > 0 and pred.confidence < 0.7:
                result.low_confidence_cells.append((pred.row, pred.col, pred.confidence))

    except Exception as e:
        result.error = f"ML inference failed: {e}"
        return result

    result.time_ml = time.time() - start_ml

    # === Validation ===
    violations = check_constraints(grid)
    result.constraint_violations = violations

    if violations and debug:
        print(f"Warning: {len(violations)} constraint violations detected")
        for v in violations[:5]:
            print(f"  - {v}")

    # === Solver ===
    start_solver = time.time()

    try:
        success, solution = run_solver(grid)
        if success:
            result.solution = solution

            # Mark solution cells
            for pred in result.predictions:
                if pred.digit == 0:
                    pred.digit = solution[pred.row][pred.col]
                    pred.is_original = False
        else:
            result.error = "Solver failed: puzzle may be invalid or have recognition errors"
            # Still return partial result
            result.solution = grid

    except Exception as e:
        result.error = f"Solver error: {e}"
        return result

    result.time_solver = time.time() - start_solver
    result.time_total = time.time() - start_total

    result.success = success
    return result


def print_grid(grid: list[list[int]], title: str = "Grid"):
    """Pretty print a 9x9 grid."""
    print(f"\n{title}:")
    print("+" + "-" * 7 + "+" + "-" * 7 + "+" + "-" * 7 + "+")
    for i, row in enumerate(grid):
        line = "|"
        for j, val in enumerate(row):
            line += f" {val if val > 0 else '.'}"
            if (j + 1) % 3 == 0:
                line += " |"
        print(line)
        if (i + 1) % 3 == 0:
            print("+" + "-" * 7 + "+" + "-" * 7 + "+" + "-" * 7 + "+")


def main():
    parser = argparse.ArgumentParser(description="Sudoku Vision Pipeline")
    parser.add_argument("image", type=Path, help="Path to sudoku puzzle image")
    parser.add_argument("--output", "-o", type=Path, help="Save result to file")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug output")
    parser.add_argument("--no-display", action="store_true", help="Don't display result window")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Error: Image not found: {args.image}")
        return 1

    print(f"Processing: {args.image}")

    # Run pipeline
    result = run_pipeline(args.image, debug=args.debug)

    # Print results
    if result.success:
        print(f"\n✓ Success!")
        print(f"  CV time:     {result.time_cv*1000:.0f}ms")
        print(f"  ML time:     {result.time_ml*1000:.0f}ms")
        print(f"  Solver time: {result.time_solver*1000:.0f}ms")
        print(f"  Total time:  {result.time_total*1000:.0f}ms")

        if result.low_confidence_cells:
            print(f"\n  Low confidence cells ({len(result.low_confidence_cells)}):")
            for r, c, conf in result.low_confidence_cells[:5]:
                print(f"    ({r+1},{c+1}): {conf:.2f}")

        print_grid(result.recognized_grid, "Recognized")
        print_grid(result.solution, "Solution")

    else:
        print(f"\n✗ Failed: {result.error}")

        if result.recognized_grid:
            print_grid(result.recognized_grid, "Recognized (incomplete)")

        if result.constraint_violations:
            print(f"\nConstraint violations:")
            for v in result.constraint_violations[:5]:
                print(f"  - {v}")

        return 1

    # Visualization
    if result.success and result.original_image is not None:
        from overlay import create_solution_overlay

        overlay = create_solution_overlay(
            result.original_image,
            result.warped_grid,
            result.predictions,
            result.solution,
        )

        if args.output:
            cv2.imwrite(str(args.output), overlay)
            print(f"\nSaved to: {args.output}")

        if not args.no_display:
            cv2.imshow("Sudoku Solution", overlay)
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
