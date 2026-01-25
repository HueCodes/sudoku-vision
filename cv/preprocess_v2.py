"""
Enhanced preprocessing for robust grid detection.

Improvements over v1:
- Multiple thresholding strategies
- Shadow and glare detection/removal
- Illumination normalization
- Morphological cleanup

These preprocessing steps help handle challenging real-world conditions
like uneven lighting, shadows, and reflections.
"""

import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class PreprocessResult:
    """Result of preprocessing with multiple outputs."""
    binary: NDArray[np.uint8]
    gray: NDArray[np.uint8]
    enhanced: NDArray[np.uint8]
    illumination_normalized: Optional[NDArray[np.uint8]] = None
    has_glare: bool = False
    has_shadow: bool = False
    method_used: str = "adaptive"


def grayscale(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert BGR image to grayscale."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_illumination(gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Normalize uneven illumination using morphological operations.

    Uses a large morphological closing to estimate the background
    illumination, then divides to normalize.
    """
    # Estimate background illumination with large kernel
    kernel_size = max(gray.shape) // 10
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(kernel_size, 51)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # Avoid division by zero
    background = np.maximum(background, 1).astype(np.float32)

    # Normalize
    normalized = (gray.astype(np.float32) / background * 255).clip(0, 255)
    return normalized.astype(np.uint8)


def detect_glare(gray: NDArray[np.uint8], threshold: int = 250) -> Tuple[bool, NDArray[np.uint8]]:
    """Detect and create mask for glare regions.

    Args:
        gray: Grayscale image
        threshold: Pixel value above which is considered glare

    Returns:
        Tuple of (has_glare, glare_mask)
    """
    glare_mask = gray > threshold
    glare_ratio = np.mean(glare_mask)

    # Consider it glare if more than 1% of image is very bright
    has_glare = glare_ratio > 0.01

    return has_glare, glare_mask.astype(np.uint8) * 255


def detect_shadow(gray: NDArray[np.uint8]) -> Tuple[bool, NDArray[np.uint8]]:
    """Detect shadow regions using local contrast analysis.

    Returns:
        Tuple of (has_shadow, shadow_mask)
    """
    # Compute local mean with large kernel
    kernel_size = max(gray.shape) // 20
    if kernel_size % 2 == 0:
        kernel_size += 1

    local_mean = cv2.blur(gray, (kernel_size, kernel_size))

    # Shadow regions have lower values than local mean by significant margin
    shadow_mask = (gray.astype(np.int32) - local_mean.astype(np.int32)) < -30

    # Need connected regions of significant size
    shadow_ratio = np.mean(shadow_mask)
    has_shadow = shadow_ratio > 0.05 and shadow_ratio < 0.5

    return has_shadow, shadow_mask.astype(np.uint8) * 255


def remove_shadow(gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Remove shadows using division-based normalization."""
    # Dilate to get background estimate
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(gray, kernel)

    # Large blur to smooth
    background = cv2.GaussianBlur(dilated, (21, 21), 0)

    # Avoid division by zero
    background = np.maximum(background, 1).astype(np.float32)

    # Divide to normalize
    normalized = (gray.astype(np.float32) / background * 255).clip(0, 255)
    return normalized.astype(np.uint8)


def apply_clahe(
    gray: NDArray[np.uint8],
    clip_limit: float = 2.0,
    tile_size: int = 8,
) -> NDArray[np.uint8]:
    """Apply Contrast Limited Adaptive Histogram Equalization."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray)


def threshold_adaptive(
    gray: NDArray[np.uint8],
    block_size: int = 11,
    c: int = 2,
) -> NDArray[np.uint8]:
    """Apply adaptive thresholding (Gaussian-weighted)."""
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, c,
    )


def threshold_otsu(gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Apply Otsu's thresholding."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def threshold_sauvola(
    gray: NDArray[np.uint8],
    window_size: int = 25,
    k: float = 0.2,
) -> NDArray[np.uint8]:
    """Apply Sauvola thresholding (better for varying illumination).

    Sauvola's method uses local mean and standard deviation:
    T(x,y) = mean(x,y) * (1 + k * (std(x,y) / R - 1))
    where R is the dynamic range of standard deviation.
    """
    # Compute local mean and std
    mean = cv2.blur(gray.astype(np.float32), (window_size, window_size))

    # Compute local std using E[X^2] - E[X]^2
    sqr_mean = cv2.blur((gray.astype(np.float32) ** 2), (window_size, window_size))
    std = np.sqrt(np.maximum(sqr_mean - mean ** 2, 0))

    # Sauvola threshold
    R = 128  # Dynamic range
    threshold = mean * (1 + k * (std / R - 1))

    binary = (gray < threshold).astype(np.uint8) * 255
    return binary


def morphological_cleanup(
    binary: NDArray[np.uint8],
    close_size: int = 3,
    open_size: int = 2,
) -> NDArray[np.uint8]:
    """Clean up binary image using morphological operations.

    - Closing: fills small gaps in lines
    - Opening: removes small noise
    """
    # Close small gaps
    if close_size > 0:
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_RECT, (close_size, close_size)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    # Remove noise
    if open_size > 0:
        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_RECT, (open_size, open_size)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    return binary


def preprocess_for_grid_detection(
    image: NDArray[np.uint8],
    use_illumination_norm: bool = True,
    use_shadow_removal: bool = True,
) -> NDArray[np.uint8]:
    """Full preprocessing pipeline for grid detection.

    This is the main entry point for preprocessing. It automatically
    detects and handles challenging conditions.

    Returns binary image with grid lines as white on black.
    """
    gray = grayscale(image)

    # Detect challenging conditions
    has_glare, glare_mask = detect_glare(gray)
    has_shadow, shadow_mask = detect_shadow(gray)

    # Apply appropriate preprocessing
    enhanced = gray.copy()

    if has_shadow and use_shadow_removal:
        enhanced = remove_shadow(enhanced)

    if use_illumination_norm:
        enhanced = normalize_illumination(enhanced)

    # Apply CLAHE for contrast enhancement
    enhanced = apply_clahe(enhanced, clip_limit=2.0, tile_size=8)

    # Light blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Adaptive threshold
    binary = threshold_adaptive(blurred, block_size=11, c=2)

    # Morphological cleanup
    binary = morphological_cleanup(binary, close_size=3, open_size=2)

    return binary


def preprocess_multi_strategy(
    image: NDArray[np.uint8],
) -> PreprocessResult:
    """Apply multiple preprocessing strategies and return all results.

    Useful when the best strategy isn't known in advance - the grid
    detector can try each result.
    """
    gray = grayscale(image)

    # Detect conditions
    has_glare, _ = detect_glare(gray)
    has_shadow, _ = detect_shadow(gray)

    # Prepare enhanced version
    enhanced = gray.copy()
    if has_shadow:
        enhanced = remove_shadow(enhanced)
    enhanced = normalize_illumination(enhanced)
    enhanced = apply_clahe(enhanced, clip_limit=2.0)

    # Try multiple thresholding strategies
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Strategy 1: Adaptive threshold
    binary_adaptive = threshold_adaptive(blurred, block_size=11, c=2)
    binary_adaptive = morphological_cleanup(binary_adaptive)

    # Strategy 2: Otsu
    binary_otsu = threshold_otsu(blurred)
    binary_otsu = morphological_cleanup(binary_otsu)

    # Strategy 3: Sauvola (good for varying illumination)
    binary_sauvola = threshold_sauvola(blurred, window_size=25, k=0.2)
    binary_sauvola = morphological_cleanup(binary_sauvola)

    # Choose best: use the one with most structure (grid lines)
    # Heuristic: good grid detection has moderate white pixel ratio
    def score_binary(b):
        ratio = np.mean(b) / 255
        # Ideal ratio is around 5-15% for grid lines
        if ratio < 0.02 or ratio > 0.3:
            return 0
        return 1 - abs(ratio - 0.1) / 0.1

    scores = [
        (score_binary(binary_adaptive), binary_adaptive, "adaptive"),
        (score_binary(binary_otsu), binary_otsu, "otsu"),
        (score_binary(binary_sauvola), binary_sauvola, "sauvola"),
    ]

    best_score, best_binary, method = max(scores, key=lambda x: x[0])

    return PreprocessResult(
        binary=best_binary,
        gray=gray,
        enhanced=enhanced,
        illumination_normalized=normalize_illumination(gray),
        has_glare=has_glare,
        has_shadow=has_shadow,
        method_used=method,
    )


def preprocess_cell(
    cell: NDArray[np.uint8],
    clip_limit: float = 2.0,
    tile_size: int = 4,
) -> NDArray[np.uint8]:
    """Preprocess a single cell for digit recognition.

    Same preprocessing used during training must be applied during inference.
    """
    # Ensure grayscale
    if len(cell.shape) == 3:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(cell)

    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2,
    )

    # Invert to get white digit on black background
    return 255 - binary


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python preprocess_v2.py <image_path>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Error: could not load {sys.argv[1]}")
        sys.exit(1)

    print("Processing image...")

    # Basic preprocessing
    binary = preprocess_for_grid_detection(img)
    cv2.imwrite("preprocessed_v2.png", binary)
    print(f"Saved preprocessed_v2.png")

    # Multi-strategy
    result = preprocess_multi_strategy(img)
    cv2.imwrite("preprocessed_multi.png", result.binary)
    print(f"Method used: {result.method_used}")
    print(f"Has glare: {result.has_glare}")
    print(f"Has shadow: {result.has_shadow}")
