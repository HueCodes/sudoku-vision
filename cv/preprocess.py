"""
Image preprocessing functions for sudoku grid detection.

Steps:
    1. Grayscale conversion
    2. Gaussian blur to reduce noise
    3. Adaptive thresholding for varying lighting
"""

import cv2
import numpy as np
from numpy.typing import NDArray


def grayscale(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert BGR image to grayscale."""
    if len(image.shape) == 2:
        return image  # Already grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def blur(image: NDArray[np.uint8], ksize: int = 5) -> NDArray[np.uint8]:
    """Apply Gaussian blur to reduce noise.

    Args:
        image: Grayscale image
        ksize: Kernel size (must be odd)
    """
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def threshold(
    image: NDArray[np.uint8],
    block_size: int = 11,
    c: int = 2,
) -> NDArray[np.uint8]:
    """Apply adaptive thresholding.

    Uses Gaussian-weighted mean for local thresholds, which handles
    varying lighting conditions better than global thresholding.

    Args:
        image: Grayscale image
        block_size: Size of pixel neighborhood for threshold (must be odd)
        c: Constant subtracted from mean
    """
    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c,
    )


def preprocess_for_grid_detection(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Full preprocessing pipeline for grid detection.

    Returns binary image with grid lines as white on black.
    """
    gray = grayscale(image)
    blurred = blur(gray, ksize=5)
    binary = threshold(blurred, block_size=11, c=2)
    return binary


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <image_path>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Error: could not load {sys.argv[1]}")
        sys.exit(1)

    result = preprocess_for_grid_detection(img)
    cv2.imwrite("preprocessed.png", result)
    print(f"Saved preprocessed.png ({result.shape})")
