"""
Multi-frame stabilizer for video/camera input.

Provides temporal smoothing of grid detection to reduce jitter
and improve reliability when processing live camera feeds.

Features:
- Corner position averaging over N frames
- Detection consistency checking
- Kalman filtering for smooth tracking
- Outlier rejection
"""

import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class StabilizedResult:
    """Result from stabilizer with averaged corners."""
    corners: Optional[NDArray[np.float32]]
    is_stable: bool  # True if detection is stable over time
    frames_detected: int  # Number of recent frames with detection
    confidence: float  # Stability confidence 0-1

    # Original detection for comparison
    raw_corners: Optional[NDArray[np.float32]] = None


class GridStabilizer:
    """Stabilizes grid detection over multiple frames."""

    def __init__(
        self,
        buffer_size: int = 5,
        min_detections: int = 3,
        max_movement: float = 50.0,
        use_kalman: bool = True,
    ):
        """
        Args:
            buffer_size: Number of frames to average over
            min_detections: Minimum detections needed for stable output
            max_movement: Maximum allowed corner movement between frames
            use_kalman: Whether to use Kalman filter for smoothing
        """
        self.buffer_size = buffer_size
        self.min_detections = min_detections
        self.max_movement = max_movement
        self.use_kalman = use_kalman

        # History buffer
        self.corner_history: deque = deque(maxlen=buffer_size)
        self.last_stable_corners: Optional[NDArray[np.float32]] = None

        # Kalman filter (one for each corner point, x and y)
        if use_kalman:
            self.kalman_filters = [self._create_kalman_filter() for _ in range(8)]
        else:
            self.kalman_filters = None

    def _create_kalman_filter(self) -> cv2.KalmanFilter:
        """Create a Kalman filter for tracking a single coordinate."""
        kf = cv2.KalmanFilter(4, 2)  # State: [x, y, vx, vy], Measurement: [x, y]

        # State transition matrix (constant velocity model)
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # Measurement matrix
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # Process noise
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01

        # Measurement noise
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0

        # Initial error covariance
        kf.errorCovPost = np.eye(4, dtype=np.float32)

        return kf

    def _is_valid_transition(
        self,
        new_corners: NDArray[np.float32],
        old_corners: NDArray[np.float32],
    ) -> bool:
        """Check if corner movement is within acceptable range."""
        if old_corners is None:
            return True

        # Check maximum movement for each corner
        for new_pt, old_pt in zip(new_corners, old_corners):
            dist = np.linalg.norm(new_pt - old_pt)
            if dist > self.max_movement:
                return False

        return True

    def _average_corners(self, history: List[NDArray[np.float32]]) -> NDArray[np.float32]:
        """Compute weighted average of corner positions."""
        if not history:
            return None

        # More recent frames get higher weight
        weights = np.linspace(0.5, 1.0, len(history))
        weights = weights / weights.sum()

        averaged = np.zeros((4, 2), dtype=np.float32)
        for weight, corners in zip(weights, history):
            averaged += weight * corners

        return averaged

    def _kalman_predict(self) -> Optional[NDArray[np.float32]]:
        """Get Kalman filter prediction for next frame."""
        if self.kalman_filters is None:
            return None

        corners = np.zeros((4, 2), dtype=np.float32)
        for i in range(4):
            pred_x = self.kalman_filters[i * 2].predict()
            pred_y = self.kalman_filters[i * 2 + 1].predict()
            corners[i] = [pred_x[0, 0], pred_y[0, 0]]

        return corners

    def _kalman_update(self, corners: NDArray[np.float32]) -> NDArray[np.float32]:
        """Update Kalman filters with new measurement."""
        if self.kalman_filters is None:
            return corners

        smoothed = np.zeros((4, 2), dtype=np.float32)
        for i in range(4):
            x, y = corners[i]

            # Update x filter
            self.kalman_filters[i * 2].correct(np.array([[x], [0]], dtype=np.float32))
            smoothed_x = self.kalman_filters[i * 2].statePost[0, 0]

            # Update y filter
            self.kalman_filters[i * 2 + 1].correct(np.array([[y], [0]], dtype=np.float32))
            smoothed_y = self.kalman_filters[i * 2 + 1].statePost[0, 0]

            smoothed[i] = [smoothed_x, smoothed_y]

        return smoothed

    def update(self, corners: Optional[NDArray[np.float32]]) -> StabilizedResult:
        """Update stabilizer with new detection.

        Args:
            corners: Detected corners from current frame, or None if not detected

        Returns:
            StabilizedResult with stabilized corners and confidence
        """
        raw_corners = corners

        # No detection this frame
        if corners is None:
            # Clear some history to indicate instability
            if len(self.corner_history) > 0:
                self.corner_history.popleft()

            # Use last stable corners if available
            if self.last_stable_corners is not None and len(self.corner_history) >= self.min_detections // 2:
                return StabilizedResult(
                    corners=self.last_stable_corners,
                    is_stable=False,
                    frames_detected=len(self.corner_history),
                    confidence=len(self.corner_history) / self.buffer_size * 0.5,
                    raw_corners=raw_corners,
                )

            return StabilizedResult(
                corners=None,
                is_stable=False,
                frames_detected=len(self.corner_history),
                confidence=0,
                raw_corners=raw_corners,
            )

        # Check for valid transition
        last_corners = self.corner_history[-1] if self.corner_history else None
        if not self._is_valid_transition(corners, last_corners):
            # Reject this detection as outlier
            return StabilizedResult(
                corners=self.last_stable_corners,
                is_stable=False,
                frames_detected=len(self.corner_history),
                confidence=len(self.corner_history) / self.buffer_size * 0.7,
                raw_corners=raw_corners,
            )

        # Add to history
        self.corner_history.append(corners.copy())

        # Apply Kalman filter
        if self.use_kalman:
            corners = self._kalman_update(corners)

        # Check if we have enough detections for stability
        if len(self.corner_history) >= self.min_detections:
            # Compute averaged corners
            averaged = self._average_corners(list(self.corner_history))

            # Update last stable corners
            self.last_stable_corners = averaged

            confidence = min(1.0, len(self.corner_history) / self.buffer_size)

            return StabilizedResult(
                corners=averaged,
                is_stable=True,
                frames_detected=len(self.corner_history),
                confidence=confidence,
                raw_corners=raw_corners,
            )

        # Not enough detections yet
        return StabilizedResult(
            corners=corners,  # Return current detection
            is_stable=False,
            frames_detected=len(self.corner_history),
            confidence=len(self.corner_history) / self.buffer_size * 0.5,
            raw_corners=raw_corners,
        )

    def reset(self):
        """Reset stabilizer state."""
        self.corner_history.clear()
        self.last_stable_corners = None

        if self.use_kalman:
            self.kalman_filters = [self._create_kalman_filter() for _ in range(8)]


class MotionDetector:
    """Detects significant motion to pause processing."""

    def __init__(self, threshold: float = 30.0, min_area_ratio: float = 0.01):
        """
        Args:
            threshold: Pixel difference threshold for motion
            min_area_ratio: Minimum ratio of changed pixels for motion
        """
        self.threshold = threshold
        self.min_area_ratio = min_area_ratio
        self.prev_frame: Optional[NDArray[np.uint8]] = None

    def update(self, frame: NDArray[np.uint8]) -> bool:
        """Check for motion in new frame.

        Args:
            frame: Grayscale frame

        Returns:
            True if significant motion detected
        """
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize for faster processing
        small = cv2.resize(frame, (160, 120))

        if self.prev_frame is None:
            self.prev_frame = small
            return False

        # Compute difference
        diff = cv2.absdiff(small, self.prev_frame)
        motion_mask = diff > self.threshold

        motion_ratio = np.mean(motion_mask)

        self.prev_frame = small

        return motion_ratio > self.min_area_ratio


if __name__ == "__main__":
    import sys
    from preprocess_v2 import preprocess_for_grid_detection
    from grid_v2 import detect_grid

    # Simulate video processing with sample images
    if len(sys.argv) < 2:
        print("Usage: python stabilizer.py <image1> [image2] [image3] ...")
        print("Simulates video processing with multiple images")
        sys.exit(1)

    stabilizer = GridStabilizer(buffer_size=5, min_detections=3)
    motion_detector = MotionDetector()

    for i, image_path in enumerate(sys.argv[1:]):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load {image_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = preprocess_for_grid_detection(img)

        # Check motion
        has_motion = motion_detector.update(gray)
        if has_motion:
            print(f"Frame {i+1}: Motion detected, skipping")
            continue

        # Detect grid
        result = detect_grid(binary, gray)
        corners = result.corners

        # Stabilize
        stabilized = stabilizer.update(corners)

        status = "STABLE" if stabilized.is_stable else "unstable"
        detected = "detected" if corners is not None else "no detection"

        print(f"Frame {i+1}: {detected}, {status}, "
              f"confidence={stabilized.confidence:.2f}, "
              f"frames={stabilized.frames_detected}/{stabilizer.buffer_size}")

        if stabilized.corners is not None:
            # Draw corners
            for j, corner in enumerate(stabilized.corners):
                color = (0, 255, 0) if stabilized.is_stable else (0, 165, 255)
                cv2.circle(img, tuple(corner.astype(int)), 8, color, -1)

            cv2.imwrite(f"stabilized_{i+1}.png", img)

    print("\nDone. Output saved to stabilized_*.png")
