"""
Failure analysis tools for E2E testing.

Analyzes failed test cases to identify patterns and root causes
of recognition errors.
"""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set

import cv2
import numpy as np


@dataclass
class FailurePattern:
    """A pattern of failures."""
    name: str
    description: str
    count: int
    examples: List[str]  # Paths to example images
    suggested_fix: str


@dataclass
class FailureAnalysis:
    """Complete failure analysis."""
    total_failures: int
    patterns: List[FailurePattern]
    by_category: Dict[str, int]
    by_status: Dict[str, int]

    # Cell-level analysis
    most_confused_digits: List[Tuple[int, int, int]]  # (true, pred, count)
    error_positions: Dict[Tuple[int, int], int]  # Position -> count

    # Recommendations
    recommendations: List[str]


def analyze_failures(failure_reports: List[Dict]) -> FailureAnalysis:
    """Analyze a collection of failure reports."""
    analysis = FailureAnalysis(
        total_failures=len(failure_reports),
        patterns=[],
        by_category={},
        by_status={},
        most_confused_digits=[],
        error_positions={},
        recommendations=[],
    )

    if not failure_reports:
        return analysis

    # Count by status
    status_counts = Counter(r.get("status", "unknown") for r in failure_reports)
    analysis.by_status = dict(status_counts)

    # Count by category (from metadata)
    category_counts = Counter()
    for r in failure_reports:
        meta = r.get("metadata", {})
        category = meta.get("category", "unknown")
        category_counts[category] += 1
    analysis.by_category = dict(category_counts)

    # Analyze cell errors
    confusion_counts = Counter()
    position_counts = Counter()

    for report in failure_reports:
        for error in report.get("cell_errors", []):
            row, col, expected, got = error
            confusion_counts[(expected, got)] += 1
            position_counts[(row, col)] += 1

    analysis.most_confused_digits = [
        (true, pred, count)
        for (true, pred), count in confusion_counts.most_common(10)
    ]
    analysis.error_positions = dict(position_counts)

    # Identify patterns
    patterns = []

    # Pattern: Grid detection failures
    grid_failures = [r for r in failure_reports if r.get("status") == "detection_failed"]
    if grid_failures:
        patterns.append(FailurePattern(
            name="Grid Detection Failure",
            description="Failed to detect sudoku grid in image",
            count=len(grid_failures),
            examples=[r.get("image_path", "") for r in grid_failures[:3]],
            suggested_fix="Check lighting, angle, and grid visibility. Try enhanced preprocessing.",
        ))

    # Pattern: Quality failures
    quality_failures = [r for r in failure_reports if r.get("status") == "quality_failed"]
    if quality_failures:
        patterns.append(FailurePattern(
            name="Image Quality Issues",
            description="Image quality too low for reliable recognition",
            count=len(quality_failures),
            examples=[r.get("image_path", "") for r in quality_failures[:3]],
            suggested_fix="Improve camera focus, lighting, or capture angle.",
        ))

    # Pattern: Digit confusion
    if analysis.most_confused_digits:
        top_confusion = analysis.most_confused_digits[0]
        true_d, pred_d, count = top_confusion
        if count >= 3:
            patterns.append(FailurePattern(
                name=f"Digit Confusion: {true_d} vs {pred_d}",
                description=f"Frequently confuses digit {true_d} with {pred_d}",
                count=count,
                examples=[],
                suggested_fix=f"Add more training examples for digits {true_d} and {pred_d}.",
            ))

    # Pattern: Edge/corner errors
    edge_errors = sum(
        count for (r, c), count in position_counts.items()
        if r in [0, 8] or c in [0, 8]
    )
    total_errors = sum(position_counts.values())
    if edge_errors > total_errors * 0.4:
        patterns.append(FailurePattern(
            name="Edge Cell Errors",
            description="Disproportionate errors at grid edges",
            count=edge_errors,
            examples=[],
            suggested_fix="Check cell extraction margins and grid boundary detection.",
        ))

    # Pattern: Empty cell detection issues
    empty_errors = sum(count for (true, pred), count in confusion_counts.items() if true == 0 or pred == 0)
    if empty_errors > total_errors * 0.3:
        patterns.append(FailurePattern(
            name="Empty Cell Detection",
            description="Issues distinguishing empty cells from digits",
            count=empty_errors,
            examples=[],
            suggested_fix="Adjust empty cell detection threshold or add training data for empty cells.",
        ))

    analysis.patterns = patterns

    # Generate recommendations
    recommendations = []

    if "detection_failed" in analysis.by_status and analysis.by_status["detection_failed"] > len(failure_reports) * 0.2:
        recommendations.append("Improve grid detection with line-based fallback method")

    if confusion_counts:
        most_common = confusion_counts.most_common(1)[0]
        (true, pred), count = most_common
        if count > 5:
            recommendations.append(f"Collect more training data for digits {true} and {pred}")

    if edge_errors > total_errors * 0.3:
        recommendations.append("Review cell extraction to reduce edge artifacts")

    if "unsolvable" in analysis.by_status and analysis.by_status["unsolvable"] > len(failure_reports) * 0.3:
        recommendations.append("Increase beam search width or max corrections in conflict resolver")

    analysis.recommendations = recommendations

    return analysis


def format_failure_report(analysis: FailureAnalysis) -> str:
    """Format failure analysis as human-readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append("FAILURE ANALYSIS REPORT")
    lines.append("=" * 60)

    lines.append(f"\nTotal Failures: {analysis.total_failures}")

    if analysis.by_status:
        lines.append("\n## By Status")
        for status, count in sorted(analysis.by_status.items(), key=lambda x: -x[1]):
            pct = count / analysis.total_failures * 100
            lines.append(f"  {status:20}: {count:4} ({pct:.1f}%)")

    if analysis.by_category:
        lines.append("\n## By Category")
        for category, count in sorted(analysis.by_category.items(), key=lambda x: -x[1]):
            pct = count / analysis.total_failures * 100
            lines.append(f"  {category:20}: {count:4} ({pct:.1f}%)")

    if analysis.patterns:
        lines.append("\n## Identified Patterns")
        for i, pattern in enumerate(analysis.patterns, 1):
            lines.append(f"\n  {i}. {pattern.name}")
            lines.append(f"     {pattern.description}")
            lines.append(f"     Count: {pattern.count}")
            lines.append(f"     Fix: {pattern.suggested_fix}")

    if analysis.most_confused_digits:
        lines.append("\n## Most Confused Digit Pairs")
        for true_d, pred_d, count in analysis.most_confused_digits[:5]:
            true_label = "empty" if true_d == 0 else str(true_d)
            pred_label = "empty" if pred_d == 0 else str(pred_d)
            lines.append(f"  {true_label} -> {pred_label}: {count} times")

    if analysis.error_positions:
        lines.append("\n## Error Position Heatmap")
        # Create simple ASCII heatmap
        max_count = max(analysis.error_positions.values())
        for r in range(9):
            row_str = "  "
            for c in range(9):
                count = analysis.error_positions.get((r, c), 0)
                if count == 0:
                    row_str += "· "
                elif count < max_count / 3:
                    row_str += "░ "
                elif count < max_count * 2 / 3:
                    row_str += "▒ "
                else:
                    row_str += "█ "
                if (c + 1) % 3 == 0:
                    row_str += " "
            lines.append(row_str)
            if (r + 1) % 3 == 0:
                lines.append("")

    if analysis.recommendations:
        lines.append("\n## Recommendations")
        for i, rec in enumerate(analysis.recommendations, 1):
            lines.append(f"  {i}. {rec}")

    lines.append("=" * 60)

    return "\n".join(lines)


def create_error_visualization(
    image_path: Path,
    recognized: List[List[int]],
    expected: List[List[int]],
    output_path: Path,
):
    """Create visualization showing recognition errors."""
    img = cv2.imread(str(image_path))
    if img is None:
        return

    h, w = img.shape[:2]
    cell_w = w // 9
    cell_h = h // 9

    # Draw grid
    for i in range(10):
        thickness = 2 if i % 3 == 0 else 1
        cv2.line(img, (i * cell_w, 0), (i * cell_w, h), (0, 0, 0), thickness)
        cv2.line(img, (0, i * cell_h), (w, i * cell_h), (0, 0, 0), thickness)

    # Mark errors
    for r in range(9):
        for c in range(9):
            exp = expected[r][c] if len(expected) > r and len(expected[r]) > c else 0
            rec = recognized[r][c] if len(recognized) > r and len(recognized[r]) > c else 0

            x = c * cell_w + cell_w // 2
            y = r * cell_h + cell_h // 2

            if exp > 0 and exp != rec:
                # Draw red X for error
                cv2.line(img, (x - 10, y - 10), (x + 10, y + 10), (0, 0, 255), 2)
                cv2.line(img, (x - 10, y + 10), (x + 10, y - 10), (0, 0, 255), 2)

                # Write expected/got
                cv2.putText(
                    img, f"{exp}->{rec}",
                    (x - 15, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1
                )

    cv2.imwrite(str(output_path), img)


def analyze_failures_from_dir(failures_dir: Path) -> FailureAnalysis:
    """Load and analyze all failures from a directory."""
    failure_reports = []

    for json_file in failures_dir.glob("failure_*.json"):
        with open(json_file) as f:
            report = json.load(f)
            failure_reports.append(report)

    return analyze_failures(failure_reports)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python failure_analysis.py <failures_directory>")
        sys.exit(1)

    failures_dir = Path(sys.argv[1])
    if not failures_dir.exists():
        print(f"Directory not found: {failures_dir}")
        sys.exit(1)

    analysis = analyze_failures_from_dir(failures_dir)
    print(format_failure_report(analysis))
