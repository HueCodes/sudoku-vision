"""
Solution Overlay Visualization

Creates visual overlays showing the solved puzzle on the original image.
"""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class CellPrediction:
    """Prediction result for a single cell."""
    row: int
    col: int
    digit: int
    confidence: float
    is_original: bool = True


def create_solution_overlay(
    original_image: np.ndarray,
    warped_grid: np.ndarray,
    predictions: list[CellPrediction],
    solution: list[list[int]],
    cell_size: int = 50,
) -> np.ndarray:
    """Create an overlay showing the solution on a clean grid.

    Returns a composite image with:
    - Original image (left)
    - Solution overlay (right)
    """
    # Create solution grid image
    grid_size = cell_size * 9 + 20  # 9 cells + padding
    solution_img = np.ones((grid_size, grid_size, 3), dtype=np.uint8) * 255

    # Draw grid lines
    for i in range(10):
        thickness = 3 if i % 3 == 0 else 1
        pos = 10 + i * cell_size
        # Horizontal
        cv2.line(solution_img, (10, pos), (grid_size - 10, pos), (0, 0, 0), thickness)
        # Vertical
        cv2.line(solution_img, (pos, 10), (pos, grid_size - 10), (0, 0, 0), thickness)

    # Draw digits
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = cell_size / 50  # Scale with cell size

    for pred in predictions:
        r, c = pred.row, pred.col
        digit = solution[r][c] if solution else pred.digit

        if digit == 0:
            continue

        # Position (center of cell)
        x = 10 + c * cell_size + cell_size // 2 - 10
        y = 10 + r * cell_size + cell_size // 2 + 10

        # Color: blue for solved cells, black for original
        if pred.is_original:
            color = (0, 0, 0)  # Black
        else:
            color = (255, 100, 0)  # Blue (BGR)

        # Confidence indicator for original cells
        if pred.is_original and pred.confidence < 0.7:
            color = (0, 0, 255)  # Red for low confidence

        cv2.putText(solution_img, str(digit), (x, y), font, font_scale, color, 2)

    # Resize original and warped to match
    target_height = grid_size
    aspect = original_image.shape[1] / original_image.shape[0]
    target_width = int(target_height * aspect)

    original_resized = cv2.resize(original_image, (target_width, target_height))

    if warped_grid is not None:
        warped_resized = cv2.resize(warped_grid, (grid_size, grid_size))
    else:
        warped_resized = None

    # Composite: original | warped | solution
    if warped_resized is not None:
        composite = np.hstack([original_resized, warped_resized, solution_img])
    else:
        composite = np.hstack([original_resized, solution_img])

    # Add labels
    cv2.putText(composite, "Original", (10, 30), font, 0.7, (0, 0, 0), 2)
    if warped_resized is not None:
        cv2.putText(composite, "Detected Grid", (target_width + 10, 30), font, 0.7, (0, 0, 0), 2)
        cv2.putText(composite, "Solution", (target_width + grid_size + 10, 30), font, 0.7, (0, 0, 0), 2)
    else:
        cv2.putText(composite, "Solution", (target_width + 10, 30), font, 0.7, (0, 0, 0), 2)

    return composite


def create_debug_overlay(
    warped_grid: np.ndarray,
    cells: list[np.ndarray],
    predictions: list,
) -> np.ndarray:
    """Create a debug visualization showing cell extractions and predictions."""
    cell_display_size = 56  # 2x the 28x28 size
    gap = 4
    grid_size = cell_display_size * 9 + gap * 10

    debug_img = np.ones((grid_size, grid_size, 3), dtype=np.uint8) * 200

    for i, (cell, pred) in enumerate(zip(cells, predictions)):
        row, col = i // 9, i % 9
        y = gap + row * (cell_display_size + gap)
        x = gap + col * (cell_display_size + gap)

        # Resize cell for display
        if len(cell.shape) == 2:
            cell_rgb = cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)
        else:
            cell_rgb = cell
        cell_resized = cv2.resize(cell_rgb, (cell_display_size, cell_display_size))

        debug_img[y:y+cell_display_size, x:x+cell_display_size] = cell_resized

        # Color border by confidence
        if pred.digit > 0:
            if pred.confidence > 0.9:
                border_color = (0, 255, 0)  # Green
            elif pred.confidence > 0.7:
                border_color = (0, 255, 255)  # Yellow
            else:
                border_color = (0, 0, 255)  # Red
            cv2.rectangle(debug_img, (x-1, y-1),
                         (x+cell_display_size, y+cell_display_size),
                         border_color, 2)

        # Show prediction
        label = str(pred.digit) if pred.digit > 0 else ""
        cv2.putText(debug_img, label, (x+2, y+cell_display_size-4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    return debug_img
