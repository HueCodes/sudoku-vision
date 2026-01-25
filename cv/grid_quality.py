"""
Grid quality scoring module.

Evaluates detected grids on multiple quality metrics to help
decide whether to proceed with recognition or request a better image.

Metrics:
- Sharpness: How focused/clear the image is
- Contrast: Dynamic range of the image
- Completeness: All grid lines visible
- Geometry: How square/regular the grid is
- Size: Resolution of extracted cells
"""

import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, List
from dataclasses import dataclass, field


@dataclass
class QualityScore:
    """Quality assessment result."""
    overall: float  # 0-100 overall quality score
    sharpness: float  # 0-100
    contrast: float  # 0-100
    completeness: float  # 0-100
    geometry: float  # 0-100
    size: float  # 0-100

    # Detailed feedback
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    @property
    def is_acceptable(self) -> bool:
        """Check if quality is good enough for recognition."""
        return self.overall >= 50

    @property
    def is_good(self) -> bool:
        """Check if quality is good."""
        return self.overall >= 70


def compute_sharpness(gray: NDArray[np.uint8]) -> float:
    """Compute image sharpness using Laplacian variance.

    Higher variance = sharper image.
    Returns score 0-100.
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()

    # Map variance to 0-100 score
    # Typical good images have variance > 500
    # Very blurry images have variance < 100
    score = min(100, variance / 10)

    return score


def compute_contrast(gray: NDArray[np.uint8]) -> float:
    """Compute image contrast using histogram spread.

    Returns score 0-100.
    """
    # Compute histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # Find the range containing 95% of pixels
    total = gray.size
    cumsum = np.cumsum(hist)

    low_idx = np.searchsorted(cumsum, total * 0.025)
    high_idx = np.searchsorted(cumsum, total * 0.975)

    dynamic_range = high_idx - low_idx

    # Map to 0-100 score
    # Good contrast has range > 150
    score = min(100, dynamic_range / 2)

    return score


def compute_completeness(
    binary: NDArray[np.uint8],
    corners: NDArray[np.float32],
) -> float:
    """Check if all grid lines are visible.

    Samples along expected grid line positions and checks
    for continuous lines.

    Returns score 0-100.
    """
    # Create warped view for easier analysis
    size = 450
    dst = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1],
    ], dtype=np.float32)

    ordered = _order_points(corners)
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(binary, matrix, (size, size))

    # Check 10 horizontal and 10 vertical lines
    cell_size = size // 9
    line_scores = []

    for i in range(10):
        # Horizontal line at row i
        y = i * cell_size
        if y >= size:
            y = size - 1
        h_line = warped[max(0, y-2):min(size, y+3), :]
        h_coverage = np.mean(h_line > 0)
        line_scores.append(h_coverage)

        # Vertical line at column i
        x = i * cell_size
        if x >= size:
            x = size - 1
        v_line = warped[:, max(0, x-2):min(size, x+3)]
        v_coverage = np.mean(v_line > 0)
        line_scores.append(v_coverage)

    # Average line coverage
    avg_coverage = np.mean(line_scores)

    # Map to 0-100
    # Good grids have coverage > 0.3 (lines aren't continuous due to gaps)
    score = min(100, avg_coverage / 0.5 * 100)

    return score


def compute_geometry(corners: NDArray[np.float32]) -> float:
    """Evaluate how square/regular the detected grid is.

    Returns score 0-100.
    """
    ordered = _order_points(corners)

    # Compute side lengths
    sides = []
    for i in range(4):
        p1 = ordered[i]
        p2 = ordered[(i + 1) % 4]
        length = np.linalg.norm(p2 - p1)
        sides.append(length)

    # Check side length consistency
    mean_side = np.mean(sides)
    side_variation = np.std(sides) / mean_side if mean_side > 0 else 1

    # Check angles (should be ~90 degrees)
    angles = []
    for i in range(4):
        p1 = ordered[i]
        p2 = ordered[(i + 1) % 4]
        p3 = ordered[(i + 2) % 4]

        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        angles.append(abs(angle - 90))

    angle_deviation = np.mean(angles)

    # Combine into score
    # Good geometry: variation < 0.1, angle deviation < 10
    side_score = max(0, 100 - side_variation * 200)
    angle_score = max(0, 100 - angle_deviation * 5)

    return (side_score + angle_score) / 2


def compute_size_score(corners: NDArray[np.float32], image_shape: Tuple[int, int]) -> float:
    """Evaluate if grid is large enough for good recognition.

    Returns score 0-100.
    """
    ordered = _order_points(corners)

    # Compute approximate cell size
    side_lengths = [
        np.linalg.norm(ordered[(i + 1) % 4] - ordered[i])
        for i in range(4)
    ]
    avg_side = np.mean(side_lengths)
    cell_size = avg_side / 9

    # Minimum cell size for good recognition is ~20 pixels
    # Ideal is 50+ pixels
    if cell_size < 15:
        score = cell_size / 15 * 30  # Low scores for very small
    elif cell_size < 30:
        score = 30 + (cell_size - 15) / 15 * 40
    else:
        score = min(100, 70 + (cell_size - 30) / 20 * 30)

    return score


def _order_points(pts: NDArray) -> NDArray:
    """Order points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    d = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    return rect


def assess_grid_quality(
    image: NDArray[np.uint8],
    binary: NDArray[np.uint8],
    corners: NDArray[np.float32],
) -> QualityScore:
    """Comprehensive grid quality assessment.

    Args:
        image: Original BGR or grayscale image
        binary: Preprocessed binary image
        corners: Detected grid corners (4 points)

    Returns:
        QualityScore with detailed metrics
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Compute individual scores
    sharpness = compute_sharpness(gray)
    contrast = compute_contrast(gray)
    completeness = compute_completeness(binary, corners)
    geometry = compute_geometry(corners)
    size_score = compute_size_score(corners, gray.shape)

    # Weighted overall score
    weights = {
        'sharpness': 0.25,
        'contrast': 0.15,
        'completeness': 0.25,
        'geometry': 0.20,
        'size': 0.15,
    }

    overall = (
        weights['sharpness'] * sharpness +
        weights['contrast'] * contrast +
        weights['completeness'] * completeness +
        weights['geometry'] * geometry +
        weights['size'] * size_score
    )

    # Generate feedback
    issues = []
    recommendations = []

    if sharpness < 40:
        issues.append("Image is blurry")
        recommendations.append("Hold camera steady or improve focus")

    if contrast < 40:
        issues.append("Low contrast")
        recommendations.append("Improve lighting conditions")

    if completeness < 40:
        issues.append("Grid lines not fully visible")
        recommendations.append("Ensure entire puzzle is in frame")

    if geometry < 50:
        issues.append("Grid is distorted")
        recommendations.append("Hold camera more perpendicular to puzzle")

    if size_score < 40:
        issues.append("Puzzle appears too small")
        recommendations.append("Move camera closer to puzzle")

    return QualityScore(
        overall=overall,
        sharpness=sharpness,
        contrast=contrast,
        completeness=completeness,
        geometry=geometry,
        size=size_score,
        issues=issues,
        recommendations=recommendations,
    )


def get_user_feedback(quality: QualityScore) -> str:
    """Generate user-friendly feedback message."""
    if quality.is_good:
        return "Image quality is good. Processing..."

    if quality.is_acceptable:
        msg = "Image quality is acceptable but could be better."
        if quality.recommendations:
            msg += f" Tip: {quality.recommendations[0]}"
        return msg

    # Poor quality
    if quality.issues:
        return f"Please retake photo: {quality.issues[0]}. {quality.recommendations[0] if quality.recommendations else ''}"

    return "Image quality is too low. Please retake the photo."


if __name__ == "__main__":
    import sys
    from preprocess_v2 import preprocess_for_grid_detection
    from grid_v2 import detect_grid

    if len(sys.argv) < 2:
        print("Usage: python grid_quality.py <image_path>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Error: could not load {sys.argv[1]}")
        sys.exit(1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = preprocess_for_grid_detection(img)
    result = detect_grid(binary, gray)

    if result.corners is None:
        print("Grid not detected")
        sys.exit(1)

    quality = assess_grid_quality(img, binary, result.corners)

    print("\nGrid Quality Assessment")
    print("=" * 40)
    print(f"Overall:      {quality.overall:5.1f}/100")
    print(f"Sharpness:    {quality.sharpness:5.1f}/100")
    print(f"Contrast:     {quality.contrast:5.1f}/100")
    print(f"Completeness: {quality.completeness:5.1f}/100")
    print(f"Geometry:     {quality.geometry:5.1f}/100")
    print(f"Size:         {quality.size:5.1f}/100")
    print()
    print(f"Acceptable: {quality.is_acceptable}")
    print(f"Good:       {quality.is_good}")

    if quality.issues:
        print("\nIssues:")
        for issue in quality.issues:
            print(f"  - {issue}")

    if quality.recommendations:
        print("\nRecommendations:")
        for rec in quality.recommendations:
            print(f"  - {rec}")

    print(f"\nUser feedback: {get_user_feedback(quality)}")
