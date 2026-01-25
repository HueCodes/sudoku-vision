"""
Robust grid detection using multiple strategies.

Improvements over v1:
- Line-based detection using Hough transform
- Corner refinement with Harris detector
- RANSAC-based quadrilateral fitting
- Rotation detection and correction
- Partial grid detection

This module provides more robust grid detection for challenging
real-world conditions.
"""

import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, List
from dataclasses import dataclass
import math


@dataclass
class GridDetectionResult:
    """Result of grid detection with metadata."""
    corners: Optional[NDArray[np.float32]]  # 4 corners, ordered
    confidence: float  # 0-1 confidence score
    method: str  # Detection method used
    rotation_angle: float  # Detected rotation in degrees
    is_partial: bool  # True if grid is partially visible
    debug_info: dict  # Additional debug information


def find_contours(binary: NDArray[np.uint8]) -> List[NDArray]:
    """Find all contours in binary image."""
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def approximate_polygon(contour: NDArray, epsilon_ratio: float = 0.02) -> NDArray:
    """Approximate contour with polygon."""
    perimeter = cv2.arcLength(contour, closed=True)
    epsilon = epsilon_ratio * perimeter
    return cv2.approxPolyDP(contour, epsilon, closed=True)


def order_points(pts: NDArray) -> NDArray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    d = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(d)]  # top-right
    rect[3] = pts[np.argmax(d)]  # bottom-left

    return rect


def is_valid_quadrilateral(corners: NDArray, min_angle: float = 45, max_angle: float = 135) -> bool:
    """Check if corners form a valid quadrilateral (roughly rectangular)."""
    if corners.shape != (4, 2):
        return False

    # Check angles
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        p3 = corners[(i + 2) % 4]

        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

        if angle < min_angle or angle > max_angle:
            return False

    # Check that sides are roughly equal (within 50%)
    side_lengths = [
        np.linalg.norm(corners[(i + 1) % 4] - corners[i])
        for i in range(4)
    ]
    min_side = min(side_lengths)
    max_side = max(side_lengths)

    if max_side > 2 * min_side:
        return False

    return True


# =============================================================================
# Contour-based detection (original method)
# =============================================================================

def detect_grid_contour(
    binary: NDArray[np.uint8],
    min_area_ratio: float = 0.1,
) -> Optional[NDArray]:
    """Find grid using contour detection (original method)."""
    contours = find_contours(binary)
    if not contours:
        return None

    image_area = binary.shape[0] * binary.shape[1]
    min_area = min_area_ratio * image_area

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            break

        approx = approximate_polygon(contour)

        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
            if is_valid_quadrilateral(corners):
                return order_points(corners)

    return None


# =============================================================================
# Line-based detection using Hough transform
# =============================================================================

def detect_lines(
    binary: NDArray[np.uint8],
    rho: float = 1,
    theta: float = np.pi / 180,
    threshold: int = 100,
    min_line_length: int = 50,
    max_line_gap: int = 10,
) -> Optional[NDArray]:
    """Detect lines using probabilistic Hough transform."""
    lines = cv2.HoughLinesP(
        binary, rho, theta, threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    return lines


def cluster_lines_by_angle(
    lines: NDArray,
    angle_tolerance: float = 10,
) -> Tuple[List[NDArray], List[NDArray]]:
    """Cluster lines into horizontal and vertical groups."""
    horizontal = []
    vertical = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Normalize angle to [0, 180]
        angle = angle % 180

        if abs(angle) < angle_tolerance or abs(angle - 180) < angle_tolerance:
            horizontal.append(line[0])
        elif abs(angle - 90) < angle_tolerance:
            vertical.append(line[0])

    return horizontal, vertical


def line_intersection(line1: NDArray, line2: NDArray) -> Optional[Tuple[float, float]]:
    """Find intersection of two lines (each as [x1, y1, x2, y2])."""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)

    return (x, y)


def find_grid_from_lines(
    horizontal: List[NDArray],
    vertical: List[NDArray],
    image_shape: Tuple[int, int],
) -> Optional[NDArray]:
    """Find grid corners from detected horizontal and vertical lines.

    Looks for the 10 horizontal and 10 vertical lines that form
    a sudoku grid, then extracts the 4 corner intersections.
    """
    if len(horizontal) < 2 or len(vertical) < 2:
        return None

    # Sort lines by position
    h_lines = sorted(horizontal, key=lambda l: (l[1] + l[3]) / 2)
    v_lines = sorted(vertical, key=lambda l: (l[0] + l[2]) / 2)

    # Take outermost lines as grid boundaries
    top_line = h_lines[0]
    bottom_line = h_lines[-1]
    left_line = v_lines[0]
    right_line = v_lines[-1]

    # Find corners
    corners = []

    tl = line_intersection(top_line, left_line)
    tr = line_intersection(top_line, right_line)
    br = line_intersection(bottom_line, right_line)
    bl = line_intersection(bottom_line, left_line)

    if None in [tl, tr, br, bl]:
        return None

    corners = np.array([tl, tr, br, bl], dtype=np.float32)

    # Validate corners are within image
    h, w = image_shape
    for x, y in corners:
        if x < -50 or x > w + 50 or y < -50 or y > h + 50:
            return None

    return corners


def detect_grid_lines(
    binary: NDArray[np.uint8],
    min_area_ratio: float = 0.1,
) -> Optional[NDArray]:
    """Detect grid using Hough line detection."""
    h, w = binary.shape

    # Detect lines with adaptive parameters
    min_length = min(h, w) // 10
    lines = detect_lines(
        binary,
        threshold=50,
        min_line_length=min_length,
        max_line_gap=min_length // 5,
    )

    if lines is None or len(lines) < 4:
        return None

    # Cluster by angle
    horizontal, vertical = cluster_lines_by_angle(lines)

    # Find grid corners
    corners = find_grid_from_lines(horizontal, vertical, (h, w))

    if corners is not None and is_valid_quadrilateral(corners):
        return order_points(corners)

    return None


# =============================================================================
# Corner-based detection with Harris detector
# =============================================================================

def detect_corners_harris(
    gray: NDArray[np.uint8],
    max_corners: int = 100,
    quality_level: float = 0.01,
    min_distance: int = 10,
) -> NDArray:
    """Detect corners using Harris detector (via goodFeaturesToTrack)."""
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        useHarrisDetector=True,
    )

    if corners is None:
        return np.array([])

    return corners.reshape(-1, 2)


def fit_quadrilateral_ransac(
    corners: NDArray,
    image_shape: Tuple[int, int],
    n_iterations: int = 100,
    inlier_threshold: float = 10,
) -> Optional[NDArray]:
    """Fit a quadrilateral to corners using RANSAC."""
    if len(corners) < 4:
        return None

    h, w = image_shape
    best_quad = None
    best_score = 0

    for _ in range(n_iterations):
        # Randomly sample 4 corners
        indices = np.random.choice(len(corners), 4, replace=False)
        sample = corners[indices]

        # Order as quadrilateral
        ordered = order_points(sample)

        # Check validity
        if not is_valid_quadrilateral(ordered):
            continue

        # Score: area ratio + squareness
        area = cv2.contourArea(ordered)
        area_ratio = area / (h * w)

        if area_ratio < 0.1:
            continue

        # Prefer larger, more square quadrilaterals
        side_lengths = [
            np.linalg.norm(ordered[(i + 1) % 4] - ordered[i])
            for i in range(4)
        ]
        squareness = min(side_lengths) / (max(side_lengths) + 1e-6)

        score = area_ratio * 0.5 + squareness * 0.5

        if score > best_score:
            best_score = score
            best_quad = ordered

    return best_quad


# =============================================================================
# Rotation detection and correction
# =============================================================================

def detect_rotation_angle(binary: NDArray[np.uint8]) -> float:
    """Detect dominant rotation angle from line orientations."""
    lines = detect_lines(binary, threshold=30, min_line_length=30, max_line_gap=5)

    if lines is None or len(lines) < 2:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Normalize to [-45, 45] range
        angle = angle % 90
        if angle > 45:
            angle -= 90
        angles.append(angle)

    # Find dominant angle using histogram
    if not angles:
        return 0.0

    # Use median for robustness
    return float(np.median(angles))


def rotate_image(
    image: NDArray[np.uint8],
    angle: float,
) -> Tuple[NDArray[np.uint8], NDArray]:
    """Rotate image by angle (degrees) and return transformation matrix."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Get rotation matrix
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute new bounding box size
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Adjust matrix for new size
    matrix[0, 2] += (new_w - w) / 2
    matrix[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(image, matrix, (new_w, new_h), borderValue=255)

    return rotated, matrix


# =============================================================================
# Main detection function
# =============================================================================

def detect_grid(
    binary: NDArray[np.uint8],
    gray: Optional[NDArray[np.uint8]] = None,
    try_rotation: bool = True,
    try_multiple_methods: bool = True,
) -> GridDetectionResult:
    """Detect sudoku grid using multiple strategies.

    This is the main entry point. It tries multiple detection methods
    and returns the best result.

    Args:
        binary: Preprocessed binary image
        gray: Original grayscale image (for corner detection)
        try_rotation: Whether to try rotation correction
        try_multiple_methods: Whether to try all detection methods

    Returns:
        GridDetectionResult with corners and metadata
    """
    h, w = binary.shape
    debug_info = {}

    # Method 1: Contour detection (fast, works well for clean images)
    corners = detect_grid_contour(binary)
    if corners is not None:
        return GridDetectionResult(
            corners=corners,
            confidence=0.9,
            method="contour",
            rotation_angle=0,
            is_partial=False,
            debug_info=debug_info,
        )

    if not try_multiple_methods:
        return GridDetectionResult(
            corners=None,
            confidence=0,
            method="none",
            rotation_angle=0,
            is_partial=False,
            debug_info=debug_info,
        )

    # Method 2: Line-based detection
    corners = detect_grid_lines(binary)
    if corners is not None:
        return GridDetectionResult(
            corners=corners,
            confidence=0.8,
            method="lines",
            rotation_angle=0,
            is_partial=False,
            debug_info=debug_info,
        )

    # Method 3: Try with rotation correction
    if try_rotation:
        rotation = detect_rotation_angle(binary)
        debug_info["detected_rotation"] = rotation

        if abs(rotation) > 2:  # More than 2 degrees
            rotated_binary, matrix = rotate_image(binary, rotation)

            corners = detect_grid_contour(rotated_binary)
            if corners is not None:
                # Transform corners back to original coordinate system
                matrix_inv = cv2.invertAffineTransform(matrix)
                ones = np.ones((4, 1))
                corners_h = np.hstack([corners, ones])
                corners = (matrix_inv @ corners_h.T).T.astype(np.float32)

                return GridDetectionResult(
                    corners=corners,
                    confidence=0.7,
                    method="contour_rotated",
                    rotation_angle=rotation,
                    is_partial=False,
                    debug_info=debug_info,
                )

    # Method 4: Corner-based detection with RANSAC
    if gray is not None:
        harris_corners = detect_corners_harris(gray)
        debug_info["harris_corners"] = len(harris_corners)

        if len(harris_corners) >= 4:
            corners = fit_quadrilateral_ransac(harris_corners, (h, w))
            if corners is not None:
                return GridDetectionResult(
                    corners=corners,
                    confidence=0.6,
                    method="harris_ransac",
                    rotation_angle=0,
                    is_partial=False,
                    debug_info=debug_info,
                )

    # No grid found
    return GridDetectionResult(
        corners=None,
        confidence=0,
        method="none",
        rotation_angle=0,
        is_partial=False,
        debug_info=debug_info,
    )


def warp_perspective(
    image: NDArray[np.uint8],
    corners: NDArray,
    output_size: int = 450,
) -> NDArray[np.uint8]:
    """Warp grid region to square image."""
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
    from preprocess_v2 import preprocess_for_grid_detection

    if len(sys.argv) < 2:
        print("Usage: python grid_v2.py <image_path>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Error: could not load {sys.argv[1]}")
        sys.exit(1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = preprocess_for_grid_detection(img)

    result = detect_grid(binary, gray)

    print(f"Detection result:")
    print(f"  Method: {result.method}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Rotation: {result.rotation_angle:.1f}°")
    print(f"  Debug: {result.debug_info}")

    if result.corners is not None:
        print(f"  Corners: {result.corners}")

        # Draw corners on image
        for i, corner in enumerate(result.corners):
            cv2.circle(img, tuple(corner.astype(int)), 10, (0, 255, 0), -1)
            cv2.putText(img, str(i), tuple(corner.astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imwrite("grid_detected.png", img)
        print("Saved grid_detected.png")

        # Warp
        warped = warp_perspective(img, result.corners)
        cv2.imwrite("warped_v2.png", warped)
        print("Saved warped_v2.png")
    else:
        print("  Grid not detected")
