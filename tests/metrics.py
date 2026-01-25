"""
Metrics and reporting utilities for E2E testing.

Provides functions for computing, tracking, and visualizing
performance metrics across test runs.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    # Recognition metrics
    grid_detection_rate: float = 0.0
    cell_accuracy: float = 0.0
    digit_accuracy: float = 0.0  # Accuracy on non-empty cells only
    empty_accuracy: float = 0.0  # Accuracy on empty cells

    # End-to-end metrics
    solution_rate: float = 0.0
    partial_solution_rate: float = 0.0  # Got some correct

    # Error analysis
    avg_cell_errors: float = 0.0
    common_confusions: List[Tuple[int, int, int]] = field(default_factory=list)  # (true, pred, count)

    # Timing
    avg_time_ms: float = 0.0
    p50_time_ms: float = 0.0
    p95_time_ms: float = 0.0

    # Per-digit accuracy
    per_digit_accuracy: Dict[int, float] = field(default_factory=dict)

    # Correction effectiveness
    correction_success_rate: float = 0.0
    avg_corrections_per_puzzle: float = 0.0


def compute_confusion_matrix(results: List[dict]) -> np.ndarray:
    """Compute 10x10 confusion matrix from test results."""
    matrix = np.zeros((10, 10), dtype=int)

    for result in results:
        for error in result.get("cell_errors", []):
            row, col, expected, got = error
            matrix[expected][got] += 1

        # Also count correct predictions
        puzzle = result.get("expected_puzzle", [])
        recognized = result.get("recognized_puzzle", [])

        if puzzle and recognized:
            for r in range(9):
                for c in range(9):
                    exp = puzzle[r][c]
                    rec = recognized[r][c] if len(recognized) > r and len(recognized[r]) > c else 0
                    if exp == rec:
                        matrix[exp][rec] += 1

    return matrix


def compute_per_digit_accuracy(confusion_matrix: np.ndarray) -> Dict[int, float]:
    """Compute accuracy for each digit class."""
    per_digit = {}

    for digit in range(10):
        total = confusion_matrix[digit].sum()
        correct = confusion_matrix[digit][digit]
        per_digit[digit] = correct / total if total > 0 else 0.0

    return per_digit


def find_common_confusions(confusion_matrix: np.ndarray, top_k: int = 10) -> List[Tuple[int, int, int]]:
    """Find most common confusion pairs."""
    confusions = []

    for true_digit in range(10):
        for pred_digit in range(10):
            if true_digit != pred_digit:
                count = confusion_matrix[true_digit][pred_digit]
                if count > 0:
                    confusions.append((true_digit, pred_digit, int(count)))

    confusions.sort(key=lambda x: -x[2])
    return confusions[:top_k]


def compute_comprehensive_metrics(results: List[dict]) -> PerformanceMetrics:
    """Compute all metrics from test results."""
    metrics = PerformanceMetrics()

    if not results:
        return metrics

    n_total = len(results)

    # Grid detection
    n_grids = sum(1 for r in results if r.get("grid_detected", False))
    metrics.grid_detection_rate = n_grids / n_total

    # Solution rate
    n_solved = sum(1 for r in results if r.get("solution_correct", False))
    metrics.solution_rate = n_solved / n_total

    # Cell accuracy
    total_cells = sum(r.get("cells_total", 0) for r in results)
    correct_cells = sum(r.get("cells_correct", 0) for r in results)
    metrics.cell_accuracy = correct_cells / total_cells if total_cells > 0 else 0

    # Average cell errors
    total_errors = sum(len(r.get("cell_errors", [])) for r in results)
    metrics.avg_cell_errors = total_errors / n_total

    # Confusion matrix and derived metrics
    confusion = compute_confusion_matrix(results)
    metrics.per_digit_accuracy = compute_per_digit_accuracy(confusion)
    metrics.common_confusions = find_common_confusions(confusion)

    # Separate empty vs digit accuracy
    if 0 in metrics.per_digit_accuracy:
        metrics.empty_accuracy = metrics.per_digit_accuracy[0]
        digit_accs = [v for k, v in metrics.per_digit_accuracy.items() if k > 0]
        metrics.digit_accuracy = np.mean(digit_accs) if digit_accs else 0

    # Timing
    times = [r.get("time_ms", 0) for r in results if r.get("time_ms", 0) > 0]
    if times:
        metrics.avg_time_ms = np.mean(times)
        metrics.p50_time_ms = np.percentile(times, 50)
        metrics.p95_time_ms = np.percentile(times, 95)

    return metrics


def format_metrics_report(metrics: PerformanceMetrics) -> str:
    """Format metrics as human-readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append("PERFORMANCE METRICS REPORT")
    lines.append("=" * 60)

    lines.append("\n## Recognition")
    lines.append(f"Grid detection rate:  {metrics.grid_detection_rate:.1%}")
    lines.append(f"Cell accuracy:        {metrics.cell_accuracy:.1%}")
    lines.append(f"Empty cell accuracy:  {metrics.empty_accuracy:.1%}")
    lines.append(f"Digit accuracy:       {metrics.digit_accuracy:.1%}")

    lines.append("\n## End-to-End")
    lines.append(f"Solution rate:        {metrics.solution_rate:.1%}")
    lines.append(f"Avg cell errors:      {metrics.avg_cell_errors:.2f}")

    lines.append("\n## Timing")
    lines.append(f"Average:              {metrics.avg_time_ms:.0f}ms")
    lines.append(f"P50:                  {metrics.p50_time_ms:.0f}ms")
    lines.append(f"P95:                  {metrics.p95_time_ms:.0f}ms")

    lines.append("\n## Per-Digit Accuracy")
    for digit in range(10):
        acc = metrics.per_digit_accuracy.get(digit, 0)
        label = "empty" if digit == 0 else str(digit)
        bar = "█" * int(acc * 20)
        lines.append(f"  {label:5}: {bar:20} {acc:.1%}")

    if metrics.common_confusions:
        lines.append("\n## Common Confusions")
        for true_d, pred_d, count in metrics.common_confusions[:5]:
            true_label = "empty" if true_d == 0 else str(true_d)
            pred_label = "empty" if pred_d == 0 else str(pred_d)
            lines.append(f"  {true_label} -> {pred_label}: {count} times")

    lines.append("=" * 60)

    return "\n".join(lines)


def compare_metrics(
    current: PerformanceMetrics,
    baseline: PerformanceMetrics,
) -> str:
    """Compare current metrics to baseline and format as report."""
    lines = []
    lines.append("=" * 60)
    lines.append("METRICS COMPARISON (current vs baseline)")
    lines.append("=" * 60)

    def fmt_delta(current_val, baseline_val):
        delta = current_val - baseline_val
        sign = "+" if delta >= 0 else ""
        return f"{current_val:.1%} ({sign}{delta:.1%})"

    lines.append("\n## Recognition")
    lines.append(f"Grid detection:  {fmt_delta(current.grid_detection_rate, baseline.grid_detection_rate)}")
    lines.append(f"Cell accuracy:   {fmt_delta(current.cell_accuracy, baseline.cell_accuracy)}")
    lines.append(f"Digit accuracy:  {fmt_delta(current.digit_accuracy, baseline.digit_accuracy)}")

    lines.append("\n## End-to-End")
    lines.append(f"Solution rate:   {fmt_delta(current.solution_rate, baseline.solution_rate)}")

    time_delta = current.avg_time_ms - baseline.avg_time_ms
    sign = "+" if time_delta >= 0 else ""
    lines.append(f"\nTiming:          {current.avg_time_ms:.0f}ms ({sign}{time_delta:.0f}ms)")

    lines.append("=" * 60)

    return "\n".join(lines)


def load_metrics_history(history_path: Path) -> List[Dict]:
    """Load metrics history from file."""
    if not history_path.exists():
        return []

    with open(history_path) as f:
        return json.load(f)


def save_metrics_to_history(
    metrics: PerformanceMetrics,
    history_path: Path,
    run_info: Optional[Dict] = None,
):
    """Append metrics to history file."""
    history = load_metrics_history(history_path)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "grid_detection_rate": metrics.grid_detection_rate,
        "cell_accuracy": metrics.cell_accuracy,
        "digit_accuracy": metrics.digit_accuracy,
        "solution_rate": metrics.solution_rate,
        "avg_time_ms": metrics.avg_time_ms,
        "per_digit_accuracy": metrics.per_digit_accuracy,
    }

    if run_info:
        entry.update(run_info)

    history.append(entry)

    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


def detect_regression(
    current: PerformanceMetrics,
    baseline: PerformanceMetrics,
    threshold: float = 0.05,
) -> Tuple[bool, List[str]]:
    """Detect if current metrics regressed from baseline.

    Returns:
        Tuple of (has_regression, list of regression descriptions)
    """
    regressions = []

    if current.solution_rate < baseline.solution_rate - threshold:
        regressions.append(
            f"Solution rate dropped: {baseline.solution_rate:.1%} -> {current.solution_rate:.1%}"
        )

    if current.cell_accuracy < baseline.cell_accuracy - threshold:
        regressions.append(
            f"Cell accuracy dropped: {baseline.cell_accuracy:.1%} -> {current.cell_accuracy:.1%}"
        )

    if current.grid_detection_rate < baseline.grid_detection_rate - threshold:
        regressions.append(
            f"Grid detection dropped: {baseline.grid_detection_rate:.1%} -> {current.grid_detection_rate:.1%}"
        )

    return len(regressions) > 0, regressions


if __name__ == "__main__":
    # Demo with sample data
    sample_results = [
        {
            "grid_detected": True,
            "cells_total": 30,
            "cells_correct": 28,
            "solution_correct": True,
            "time_ms": 450,
            "cell_errors": [(0, 1, 5, 3), (2, 3, 8, 6)],
        },
        {
            "grid_detected": True,
            "cells_total": 25,
            "cells_correct": 20,
            "solution_correct": False,
            "time_ms": 520,
            "cell_errors": [(0, 0, 1, 7), (1, 1, 9, 6), (2, 2, 4, 9), (3, 3, 2, 8), (4, 4, 6, 3)],
        },
    ]

    metrics = compute_comprehensive_metrics(sample_results)
    print(format_metrics_report(metrics))
