#!/usr/bin/env python3
"""
Smart Sudoku Vision Pipeline (v2).

Improved pipeline with constraint-based validation and error correction.

Features:
- Multi-strategy grid detection
- Confidence-aware predictions with alternatives
- Constraint validation before solving
- Automatic error correction using beam search
- Solver with feedback loop

Usage:
    python run_v2.py <image_path> [--debug]
"""

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "cv"))
sys.path.insert(0, str(PROJECT_ROOT / "ml"))
sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))

from preprocess_v2 import preprocess_for_grid_detection, preprocess_multi_strategy
from grid_v2 import detect_grid, warp_perspective
from grid_quality import assess_grid_quality, get_user_feedback
from extract import extract_cells

from validator import CellInfo, validate_predictions, ValidationResult
from constraint_resolver import resolve_with_constraints
from conflict_resolver import resolve_conflicts, ResolutionResult


@dataclass
class PipelineConfig:
    """Configuration for pipeline."""
    confidence_threshold: float = 0.7
    min_alternative_confidence: float = 0.05
    max_corrections: int = 3
    beam_width: int = 5
    require_quality_check: bool = True
    min_quality_score: float = 40.0


@dataclass
class PipelineResult:
    """Result of running the smart pipeline."""
    success: bool
    status: str  # 'solved', 'unsolvable', 'invalid', 'quality_failed', 'detection_failed'
    error: Optional[str] = None

    # Timing
    time_cv: float = 0.0
    time_ml: float = 0.0
    time_validation: float = 0.0
    time_solver: float = 0.0
    time_total: float = 0.0

    # CV results
    original_image: Optional[np.ndarray] = None
    warped_grid: Optional[np.ndarray] = None
    quality_score: float = 0.0
    quality_feedback: str = ""
    detection_method: str = ""

    # ML results
    cells: List[CellInfo] = field(default_factory=list)
    recognized_grid: List[List[int]] = field(default_factory=list)

    # Validation results
    validation: Optional[ValidationResult] = None
    corrections_made: List = field(default_factory=list)

    # Solution
    solution: List[List[int]] = field(default_factory=list)

    # Confidence map for UI
    confidence_map: List[List[float]] = field(default_factory=list)
    low_confidence_cells: List[Tuple[int, int]] = field(default_factory=list)


def load_model(version: str = "v3"):
    """Load the digit recognition model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if version == "v3":
        try:
            from model_v3 import DigitCNNv3
            model = DigitCNNv3().to(device)
            # Prefer final model (real + synthetic data including golden)
            model_path = PROJECT_ROOT / "models" / "digit_cnn_v3_final.pt"
            if not model_path.exists():
                model_path = PROJECT_ROOT / "models" / "digit_cnn_v3_combined.pt"
            if not model_path.exists():
                model_path = PROJECT_ROOT / "models" / "digit_cnn_v3_synthetic.pt"
            if not model_path.exists():
                model_path = PROJECT_ROOT / "models" / "digit_cnn_v3.pt"
        except ImportError:
            from model import DigitCNN
            model = DigitCNN().to(device)
            model_path = PROJECT_ROOT / "ml" / "digit_cnn_v2.pt"
    else:
        from model import DigitCNN
        model = DigitCNN().to(device)
        model_path = PROJECT_ROOT / "ml" / "digit_cnn_v2.pt"

    if not model_path.exists():
        # Fallback
        model_path = PROJECT_ROOT / "ml" / "digit_cnn.pt"

    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    model.eval()
    return model, device


def preprocess_cell(cell: np.ndarray) -> np.ndarray:
    """Preprocess a cell for ML inference."""
    if len(cell.shape) == 3:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    if cell.shape != (28, 28):
        cell = cv2.resize(cell, (28, 28))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    cell = clahe.apply(cell)

    cell = cv2.adaptiveThreshold(
        cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return 255 - cell  # Invert


def predict_cells_with_alternatives(
    cells: List[np.ndarray],
    model: torch.nn.Module,
    device: torch.device,
    top_k: int = 3,
) -> List[CellInfo]:
    """Predict digits with top-k alternatives."""
    predictions = []

    for i, cell in enumerate(cells):
        row, col = i // 9, i % 9

        processed = preprocess_cell(cell)
        tensor = torch.from_numpy(processed).float().unsqueeze(0).unsqueeze(0) / 255.0
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.to(device)

        with torch.no_grad():
            output = model(tensor)
            probs = F.softmax(output, dim=1)[0]

            # Get top-k predictions
            top_probs, top_indices = probs.topk(top_k)

            best_digit = top_indices[0].item()
            best_conf = top_probs[0].item()

            # Get alternatives (excluding best)
            alternatives = [
                (top_indices[j].item(), top_probs[j].item())
                for j in range(1, top_k)
            ]

        predictions.append(CellInfo(
            row=row,
            col=col,
            digit=best_digit,
            confidence=best_conf,
            alternatives=alternatives,
        ))

    return predictions


def build_grid(cells: List[CellInfo]) -> List[List[int]]:
    """Build 9x9 grid from cells."""
    grid = [[0] * 9 for _ in range(9)]
    for cell in cells:
        grid[cell.row][cell.col] = cell.digit
    return grid


def build_confidence_map(cells: List[CellInfo]) -> List[List[float]]:
    """Build confidence map."""
    conf_map = [[0.0] * 9 for _ in range(9)]
    for cell in cells:
        conf_map[cell.row][cell.col] = cell.confidence
    return conf_map


def run_solver(grid: List[List[int]]) -> Tuple[bool, List[List[int]]]:
    """Run the C solver."""
    solver_path = PROJECT_ROOT / "solver" / "sudoku_solver"

    if not solver_path.exists():
        return False, grid

    input_str = ""
    for row in grid:
        input_str += "".join(str(c) for c in row) + "\n"

    temp_input = Path("/tmp/sudoku_pipeline_input.txt")
    temp_output = Path("/tmp/sudoku_pipeline_output.txt")
    temp_input.write_text(input_str)

    try:
        result = subprocess.run(
            [str(solver_path), str(temp_input), "-o", str(temp_output)],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            return False, grid

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

    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass

    return False, grid


def run_pipeline(
    image_path: Path,
    config: Optional[PipelineConfig] = None,
    debug: bool = False,
) -> PipelineResult:
    """Run the smart pipeline."""
    if config is None:
        config = PipelineConfig()

    result = PipelineResult(success=False, status='detection_failed')
    start_total = time.time()

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        result.error = f"Failed to load image: {image_path}"
        return result
    result.original_image = image

    # === CV Pipeline ===
    start_cv = time.time()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhanced preprocessing
    preprocess_result = preprocess_multi_strategy(image)
    binary = preprocess_result.binary

    if debug:
        print(f"Preprocessing: method={preprocess_result.method_used}, "
              f"shadow={preprocess_result.has_shadow}, glare={preprocess_result.has_glare}")

    # Grid detection
    detection = detect_grid(binary, gray)

    if detection.corners is None:
        result.error = "Grid detection failed"
        result.status = 'detection_failed'
        return result

    result.detection_method = detection.method

    if debug:
        print(f"Grid detection: method={detection.method}, confidence={detection.confidence:.2f}")

    # Quality check
    if config.require_quality_check:
        quality = assess_grid_quality(image, binary, detection.corners)
        result.quality_score = quality.overall
        result.quality_feedback = get_user_feedback(quality)

        if quality.overall < config.min_quality_score:
            result.status = 'quality_failed'
            result.error = result.quality_feedback
            return result

        if debug:
            print(f"Quality: {quality.overall:.1f}/100 - {result.quality_feedback}")

    # Perspective warp
    warped = warp_perspective(image, detection.corners)
    result.warped_grid = warped

    # Cell extraction
    cell_images = extract_cells(warped)
    if len(cell_images) != 81:
        result.error = f"Cell extraction failed: got {len(cell_images)} cells"
        result.status = 'detection_failed'
        return result

    result.time_cv = time.time() - start_cv

    # === ML Pipeline ===
    start_ml = time.time()

    model, device = load_model("v3")
    cells = predict_cells_with_alternatives(cell_images, model, device, top_k=3)

    result.cells = cells
    result.recognized_grid = build_grid(cells)
    result.confidence_map = build_confidence_map(cells)

    # Find low confidence cells
    result.low_confidence_cells = [
        (c.row, c.col) for c in cells
        if c.digit > 0 and c.confidence < config.confidence_threshold
    ]

    result.time_ml = time.time() - start_ml

    # === Validation & Correction ===
    start_validation = time.time()

    validation = validate_predictions(cells)
    result.validation = validation

    if debug:
        print(f"Initial validation: valid={validation.is_valid}, conflicts={validation.num_conflicts}")

    # If conflicts, try to resolve
    if not validation.is_valid:
        resolution = resolve_conflicts(
            cells,
            beam_width=config.beam_width,
            max_corrections=config.max_corrections,
        )

        if debug:
            print(f"Conflict resolution: success={resolution.success}, "
                  f"corrections={len(resolution.corrections_made)}")

        if resolution.success or resolution.validation_result.num_conflicts < validation.num_conflicts:
            cells = resolution.cells
            result.cells = cells
            result.recognized_grid = build_grid(cells)
            result.corrections_made = resolution.corrections_made
            validation = resolution.validation_result
            result.validation = validation

    # Constraint propagation
    confidences = [[c.confidence for c in cells[r*9:(r+1)*9]] for r in range(9)]
    propagation = resolve_with_constraints(result.recognized_grid, confidences)

    if propagation.cells_resolved:
        if debug:
            print(f"Constraint propagation: resolved {len(propagation.cells_resolved)} cells")

        # Update grid
        result.recognized_grid = propagation.grid

    if not propagation.is_valid:
        result.status = 'invalid'
        result.error = f"Puzzle has contradictions at {propagation.contradiction_cell}"
        result.time_validation = time.time() - start_validation
        result.time_total = time.time() - start_total
        return result

    result.time_validation = time.time() - start_validation

    # === Solver ===
    start_solver = time.time()

    success, solution = run_solver(result.recognized_grid)

    if success:
        result.solution = solution
        result.success = True
        result.status = 'solved'
    else:
        result.status = 'unsolvable'
        result.error = "Puzzle could not be solved - likely recognition errors remain"
        result.solution = result.recognized_grid

    result.time_solver = time.time() - start_solver
    result.time_total = time.time() - start_total

    return result


def print_grid(grid: List[List[int]], title: str = "Grid"):
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
    parser = argparse.ArgumentParser(description="Smart Sudoku Vision Pipeline")
    parser.add_argument("image", type=Path, help="Path to sudoku puzzle image")
    parser.add_argument("--output", "-o", type=Path, help="Save result to file")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug output")
    parser.add_argument("--no-quality-check", action="store_true", help="Skip quality check")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Error: Image not found: {args.image}")
        return 1

    print(f"Processing: {args.image}")

    config = PipelineConfig(
        require_quality_check=not args.no_quality_check,
    )

    result = run_pipeline(args.image, config, debug=args.debug)

    # Print results
    print(f"\nStatus: {result.status}")
    print(f"Timing:")
    print(f"  CV:         {result.time_cv*1000:.0f}ms")
    print(f"  ML:         {result.time_ml*1000:.0f}ms")
    print(f"  Validation: {result.time_validation*1000:.0f}ms")
    print(f"  Solver:     {result.time_solver*1000:.0f}ms")
    print(f"  Total:      {result.time_total*1000:.0f}ms")

    if result.quality_score > 0:
        print(f"\nQuality: {result.quality_score:.1f}/100")
        print(f"  {result.quality_feedback}")

    if result.low_confidence_cells:
        print(f"\nLow confidence cells: {len(result.low_confidence_cells)}")
        for r, c in result.low_confidence_cells[:5]:
            conf = result.confidence_map[r][c]
            digit = result.recognized_grid[r][c]
            print(f"  ({r+1},{c+1}): digit={digit}, conf={conf:.2f}")

    if result.corrections_made:
        print(f"\nCorrections applied: {len(result.corrections_made)}")
        for corr in result.corrections_made:
            print(f"  ({corr.row+1},{corr.col+1}): {corr.original_digit} -> {corr.new_digit}")

    if result.validation and not result.validation.is_valid:
        print(f"\nRemaining conflicts: {result.validation.num_conflicts}")

    if result.success:
        print_grid(result.recognized_grid, "Recognized")
        print_grid(result.solution, "Solution")
    else:
        print(f"\nError: {result.error}")
        if result.recognized_grid:
            print_grid(result.recognized_grid, "Recognized (incomplete)")

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
