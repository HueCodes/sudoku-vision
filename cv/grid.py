"""
Grid detection and perspective correction.

Steps:
    1. Find contours in preprocessed image
    2. Filter for largest quadrilateral (the sudoku grid)
    3. Order corner points consistently
    4. Apply perspective transform to get flat 9x9 grid
"""

import cv2
import numpy as np
from numpy.typing import NDArray


def find_contours(binary: NDArray[np.uint8]) -> list[NDArray]:
    """Find all contours in binary image."""
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def approximate_polygon(contour: NDArray, epsilon_ratio: float = 0.02) -> NDArray:
    """Approximate contour with polygon.

    Args:
        contour: Input contour points
        epsilon_ratio: Approximation accuracy as ratio of perimeter
    """
    perimeter = cv2.arcLength(contour, closed=True)
    epsilon = epsilon_ratio * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    return approx


def find_grid_contour(
    binary: NDArray[np.uint8],
    min_area_ratio: float = 0.1,
) -> NDArray | None:
    """Find the sudoku grid contour (largest quadrilateral).

    Args:
        binary: Preprocessed binary image
        min_area_ratio: Minimum area as ratio of image area

    Returns:
        4 corner points of the grid, or None if not found
    """
    contours = find_contours(binary)
    if not contours:
        return None

    image_area = binary.shape[0] * binary.shape[1]
    min_area = min_area_ratio * image_area

    # Sort by area, largest first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            break

        approx = approximate_polygon(contour)

        # Looking for quadrilateral
        if len(approx) == 4:
            return approx.reshape(4, 2)

    return None


def order_points(pts: NDArray) -> NDArray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left.

    This consistent ordering is required for perspective transform.
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    # Sum: top-left has smallest, bottom-right has largest
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    # Diff: top-right has smallest, bottom-left has largest
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]  # top-right
    rect[3] = pts[np.argmax(d)]  # bottom-left

    return rect


def warp_perspective(
    image: NDArray[np.uint8],
    corners: NDArray,
    output_size: int = 450,
) -> NDArray[np.uint8]:
    """Warp grid region to square image.

    Args:
        image: Original image
        corners: 4 corner points of grid
        output_size: Size of output square image (pixels)

    Returns:
        Warped square image of the grid
    """
    ordered = order_points(corners.astype(np.float32))

    dst = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1],
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, matrix, (output_size, output_size))

    return warped


if __name__ == "__main__":
    import sys
    from preprocess import preprocess_for_grid_detection

    if len(sys.argv) < 2:
        print("Usage: python grid.py <image_path>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Error: could not load {sys.argv[1]}")
        sys.exit(1)

    binary = preprocess_for_grid_detection(img)
    corners = find_grid_contour(binary)

    if corners is None:
        print("Error: could not find grid")
        sys.exit(1)

    print(f"Found grid corners: {corners}")

    warped = warp_perspective(img, corners)
    cv2.imwrite("warped.png", warped)
    print(f"Saved warped.png ({warped.shape})")
