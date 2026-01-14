"""
Cell extraction from warped grid image.

Takes a perspective-corrected 9x9 grid and extracts 81 individual
cell images for digit recognition.
"""

import cv2
import numpy as np
from numpy.typing import NDArray


def extract_cells(
    grid_image: NDArray[np.uint8],
    cell_size: int = 28,
    margin_ratio: float = 0.1,
) -> list[NDArray[np.uint8]]:
    """Extract 81 cell images from warped grid.

    Args:
        grid_image: Warped square grid image
        cell_size: Output size for each cell (28 for MNIST compatibility)
        margin_ratio: Margin to crop from each cell edge (avoids grid lines)

    Returns:
        List of 81 cell images in row-major order (top-left to bottom-right)
    """
    h, w = grid_image.shape[:2]
    cell_h, cell_w = h // 9, w // 9

    margin_h = int(cell_h * margin_ratio)
    margin_w = int(cell_w * margin_ratio)

    cells = []

    for row in range(9):
        for col in range(9):
            # Cell bounds
            y1 = row * cell_h + margin_h
            y2 = (row + 1) * cell_h - margin_h
            x1 = col * cell_w + margin_w
            x2 = (col + 1) * cell_w - margin_w

            # Extract and resize
            cell = grid_image[y1:y2, x1:x2]

            # Convert to grayscale if needed
            if len(cell.shape) == 3:
                cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

            # Resize to target size
            cell = cv2.resize(cell, (cell_size, cell_size))

            cells.append(cell)

    return cells


def is_cell_empty(
    cell: NDArray[np.uint8],
    threshold: float = 0.02,
) -> bool:
    """Check if cell is empty (no digit).

    Args:
        cell: Grayscale cell image
        threshold: Ratio of non-white pixels to consider empty

    Returns:
        True if cell appears empty
    """
    # Threshold to binary
    _, binary = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Count non-zero pixels
    non_zero = cv2.countNonZero(binary)
    total = cell.shape[0] * cell.shape[1]

    return (non_zero / total) < threshold


def preprocess_cell_for_model(cell: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Preprocess cell image for CNN input.

    Returns normalized float32 array with shape (1, 28, 28).
    """
    # Ensure grayscale
    if len(cell.shape) == 3:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    # Resize to 28x28 if needed
    if cell.shape != (28, 28):
        cell = cv2.resize(cell, (28, 28))

    # Normalize to [0, 1]
    normalized = cell.astype(np.float32) / 255.0

    # Add channel dimension
    return normalized.reshape(1, 28, 28)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python extract.py <warped_grid_image>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Error: could not load {sys.argv[1]}")
        sys.exit(1)

    cells = extract_cells(img)
    print(f"Extracted {len(cells)} cells")

    # Save cells
    out_dir = Path("cells")
    out_dir.mkdir(exist_ok=True)

    empty_count = 0
    for i, cell in enumerate(cells):
        row, col = i // 9, i % 9
        cv2.imwrite(str(out_dir / f"cell_{row}_{col}.png"), cell)
        if is_cell_empty(cell):
            empty_count += 1

    print(f"Saved to {out_dir}/")
    print(f"Empty cells: {empty_count}/81")
