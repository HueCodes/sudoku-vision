#!/usr/bin/env python3
"""
End-to-End Test Runner for Sudoku Vision.

Runs the full pipeline on test images with ground truth and
measures success rates across different conditions.

Usage:
    python tests/e2e_runner.py
    python tests/e2e_runner.py --output-dir results/ --save-failures
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))


@dataclass
class TestCase:
    """A single test case."""
    image_path: str
    solution: List[List[int]]  # Correct solution
    puzzle: Optional[List[List[int]]] = None  # Original puzzle (0=empty), optional
    metadata: Dict = field(default_factory=dict)  # Source, difficulty, etc.


@dataclass
class TestResult:
    """Result of running a single test."""
    test_case: TestCase
    success: bool
    status: str

    # Metrics
    grid_detected: bool = False
    cells_correct: int = 0
    cells_total: int = 81
    solution_correct: bool = False

    # Timing
    time_ms: float = 0.0

    # Details
    recognized_puzzle: Optional[List[List[int]]] = None
    produced_solution: Optional[List[List[int]]] = None
    errors: List[str] = field(default_factory=list)
    cell_errors: List[Tuple[int, int, int, int]] = field(default_factory=list)  # (row, col, expected, got)


@dataclass
class TestSuiteResult:
    """Aggregate results for a test suite."""
    total: int = 0
    passed: int = 0
    failed: int = 0

    # Detailed metrics
    grid_detection_rate: float = 0.0
    cell_accuracy: float = 0.0
    solution_rate: float = 0.0

    # By category
    by_category: Dict[str, Dict] = field(default_factory=dict)

    # Timing
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0

    # Individual results
    results: List[TestResult] = field(default_factory=list)


def load_ground_truth(ground_truth_path: Path) -> List[TestCase]:
    """Load ground truth from JSON file."""
    with open(ground_truth_path) as f:
        data = json.load(f)

    test_cases = []
    for item in data.get("test_cases", []):
        test_cases.append(TestCase(
            image_path=item["image_path"],
            solution=item["solution"],
            puzzle=item.get("puzzle"),
            metadata=item.get("metadata", {}),
        ))

    return test_cases


def compare_grids(
    expected: List[List[int]],
    actual: List[List[int]],
) -> Tuple[int, List[Tuple[int, int, int, int]]]:
    """Compare two grids and return (correct_count, errors).

    Only compares non-empty cells in expected.
    """
    correct = 0
    errors = []

    for r in range(9):
        for c in range(9):
            exp = expected[r][c]
            act = actual[r][c] if actual and len(actual) > r and len(actual[r]) > c else 0

            if exp > 0:  # Only check non-empty expected cells
                if exp == act:
                    correct += 1
                else:
                    errors.append((r, c, exp, act))

    return correct, errors


def run_single_test(test_case: TestCase) -> TestResult:
    """Run pipeline on a single test case."""
    from run_v2 import run_pipeline, PipelineConfig

    result = TestResult(
        test_case=test_case,
        success=False,
        status='not_run',
    )

    image_path = Path(test_case.image_path)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    if not image_path.exists():
        result.status = 'image_not_found'
        result.errors.append(f"Image not found: {image_path}")
        return result

    # Run pipeline
    config = PipelineConfig(require_quality_check=False)
    start = time.time()

    try:
        pipeline_result = run_pipeline(image_path, config)
        result.time_ms = (time.time() - start) * 1000

        result.grid_detected = pipeline_result.warped_grid is not None
        result.status = pipeline_result.status

        if pipeline_result.recognized_grid:
            result.recognized_puzzle = pipeline_result.recognized_grid

            # Count correct cells in recognition (only if puzzle ground truth available)
            if test_case.puzzle:
                non_empty = sum(1 for r in test_case.puzzle for c in r if c > 0)
                result.cells_total = non_empty

                correct, errors = compare_grids(test_case.puzzle, pipeline_result.recognized_grid)
                result.cells_correct = correct
                result.cell_errors = errors
            else:
                result.cells_total = 81  # Assume full grid

        if pipeline_result.success and pipeline_result.solution:
            result.produced_solution = pipeline_result.solution

            # Check if solution matches
            if pipeline_result.solution == test_case.solution:
                result.solution_correct = True
                result.success = True
            else:
                result.errors.append("Solution doesn't match ground truth")

        if pipeline_result.error:
            result.errors.append(pipeline_result.error)

    except Exception as e:
        result.time_ms = (time.time() - start) * 1000
        result.status = 'exception'
        result.errors.append(str(e))

    return result


def run_test_suite(
    test_cases: List[TestCase],
    verbose: bool = False,
) -> TestSuiteResult:
    """Run all test cases and aggregate results."""
    suite_result = TestSuiteResult()
    suite_result.total = len(test_cases)

    category_stats = {}

    for i, test_case in enumerate(test_cases):
        if verbose:
            print(f"[{i+1}/{len(test_cases)}] Testing {test_case.image_path}...", end=" ")

        result = run_single_test(test_case)
        suite_result.results.append(result)

        if result.success:
            suite_result.passed += 1
            if verbose:
                print(f"PASS ({result.time_ms:.0f}ms)")
        else:
            suite_result.failed += 1
            if verbose:
                print(f"FAIL: {result.status}")
                for err in result.errors[:2]:
                    print(f"       {err}")

        suite_result.total_time_ms += result.time_ms

        # Track by category
        category = test_case.metadata.get("category", "default")
        if category not in category_stats:
            category_stats[category] = {
                "total": 0,
                "passed": 0,
                "grid_detected": 0,
                "cells_correct": 0,
                "cells_total": 0,
            }

        stats = category_stats[category]
        stats["total"] += 1
        if result.success:
            stats["passed"] += 1
        if result.grid_detected:
            stats["grid_detected"] += 1
        stats["cells_correct"] += result.cells_correct
        stats["cells_total"] += result.cells_total

    # Compute aggregate metrics
    if suite_result.total > 0:
        suite_result.avg_time_ms = suite_result.total_time_ms / suite_result.total

        grids_detected = sum(1 for r in suite_result.results if r.grid_detected)
        suite_result.grid_detection_rate = grids_detected / suite_result.total

        total_cells = sum(r.cells_total for r in suite_result.results)
        correct_cells = sum(r.cells_correct for r in suite_result.results)
        suite_result.cell_accuracy = correct_cells / total_cells if total_cells > 0 else 0

        suite_result.solution_rate = suite_result.passed / suite_result.total

    # Finalize category stats
    for category, stats in category_stats.items():
        if stats["total"] > 0:
            suite_result.by_category[category] = {
                "total": stats["total"],
                "passed": stats["passed"],
                "pass_rate": stats["passed"] / stats["total"],
                "grid_detection_rate": stats["grid_detected"] / stats["total"],
                "cell_accuracy": stats["cells_correct"] / stats["cells_total"] if stats["cells_total"] > 0 else 0,
            }

    return suite_result


def print_summary(suite_result: TestSuiteResult):
    """Print summary of test results."""
    print("\n" + "=" * 60)
    print("END-TO-END TEST RESULTS")
    print("=" * 60)

    print(f"\nOverall:")
    print(f"  Total tests:        {suite_result.total}")
    print(f"  Passed:             {suite_result.passed}")
    print(f"  Failed:             {suite_result.failed}")
    print(f"  Pass rate:          {suite_result.solution_rate:.1%}")

    print(f"\nMetrics:")
    print(f"  Grid detection:     {suite_result.grid_detection_rate:.1%}")
    print(f"  Cell accuracy:      {suite_result.cell_accuracy:.1%}")
    print(f"  Avg time:           {suite_result.avg_time_ms:.0f}ms")

    if suite_result.by_category:
        print(f"\nBy Category:")
        for category, stats in suite_result.by_category.items():
            print(f"  {category}:")
            print(f"    Pass rate:        {stats['pass_rate']:.1%} ({stats['passed']}/{stats['total']})")
            print(f"    Cell accuracy:    {stats['cell_accuracy']:.1%}")

    print("=" * 60)


def save_failures(
    suite_result: TestSuiteResult,
    output_dir: Path,
):
    """Save failure cases for analysis."""
    failures_dir = output_dir / "failures"
    failures_dir.mkdir(parents=True, exist_ok=True)

    for i, result in enumerate(suite_result.results):
        if not result.success:
            # Create failure report
            report = {
                "image_path": result.test_case.image_path,
                "status": result.status,
                "errors": result.errors,
                "cell_errors": result.cell_errors,
                "expected_puzzle": result.test_case.puzzle,
                "recognized_puzzle": result.recognized_puzzle,
                "expected_solution": result.test_case.solution,
                "produced_solution": result.produced_solution,
            }

            # Save report
            report_path = failures_dir / f"failure_{i:03d}.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            # Copy/save annotated image if available
            image_path = Path(result.test_case.image_path)
            if not image_path.is_absolute():
                image_path = PROJECT_ROOT / image_path

            if image_path.exists():
                img = cv2.imread(str(image_path))
                if img is not None:
                    # Annotate with errors
                    cv2.putText(
                        img, f"Status: {result.status}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                    )
                    cv2.imwrite(str(failures_dir / f"failure_{i:03d}.jpg"), img)

    print(f"Saved {suite_result.failed} failure cases to {failures_dir}")


def create_sample_ground_truth(output_path: Path):
    """Create a sample ground truth file structure."""
    sample = {
        "created": datetime.now().isoformat(),
        "description": "Sample ground truth file - replace with real data",
        "test_cases": [
            {
                "image_path": "data/test_images/sample_1.jpg",
                "puzzle": [
                    [5, 3, 0, 0, 7, 0, 0, 0, 0],
                    [6, 0, 0, 1, 9, 5, 0, 0, 0],
                    [0, 9, 8, 0, 0, 0, 0, 6, 0],
                    [8, 0, 0, 0, 6, 0, 0, 0, 3],
                    [4, 0, 0, 8, 0, 3, 0, 0, 1],
                    [7, 0, 0, 0, 2, 0, 0, 0, 6],
                    [0, 6, 0, 0, 0, 0, 2, 8, 0],
                    [0, 0, 0, 4, 1, 9, 0, 0, 5],
                    [0, 0, 0, 0, 8, 0, 0, 7, 9],
                ],
                "solution": [
                    [5, 3, 4, 6, 7, 8, 9, 1, 2],
                    [6, 7, 2, 1, 9, 5, 3, 4, 8],
                    [1, 9, 8, 3, 4, 2, 5, 6, 7],
                    [8, 5, 9, 7, 6, 1, 4, 2, 3],
                    [4, 2, 6, 8, 5, 3, 7, 9, 1],
                    [7, 1, 3, 9, 2, 4, 8, 5, 6],
                    [9, 6, 1, 5, 3, 7, 2, 8, 4],
                    [2, 8, 7, 4, 1, 9, 6, 3, 5],
                    [3, 4, 5, 2, 8, 6, 1, 7, 9],
                ],
                "metadata": {
                    "source": "sample",
                    "category": "clean",
                    "difficulty": "medium"
                }
            }
        ]
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(sample, f, indent=2)

    print(f"Created sample ground truth at {output_path}")
    print("Edit this file to add your actual test cases.")


def main():
    parser = argparse.ArgumentParser(description="E2E Test Runner for Sudoku Vision")
    parser.add_argument(
        "--ground-truth", "-g",
        type=Path,
        default=PROJECT_ROOT / "data" / "test_e2e" / "ground_truth.json",
        help="Path to ground truth JSON file",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=PROJECT_ROOT / "test_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--save-failures",
        action="store_true",
        help="Save failure cases for analysis",
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create sample ground truth file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    if args.create_sample:
        create_sample_ground_truth(args.ground_truth)
        return 0

    if not args.ground_truth.exists():
        print(f"Ground truth file not found: {args.ground_truth}")
        print("Run with --create-sample to create a template.")
        return 1

    # Load test cases
    print(f"Loading ground truth from {args.ground_truth}")
    test_cases = load_ground_truth(args.ground_truth)
    print(f"Found {len(test_cases)} test cases")

    if not test_cases:
        print("No test cases found.")
        return 1

    # Run tests
    print("\nRunning tests...")
    suite_result = run_test_suite(test_cases, verbose=args.verbose)

    # Print summary
    print_summary(suite_result)

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results_path = args.output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        # Convert to serializable format
        output = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": suite_result.total,
                "passed": suite_result.passed,
                "failed": suite_result.failed,
                "pass_rate": suite_result.solution_rate,
                "grid_detection_rate": suite_result.grid_detection_rate,
                "cell_accuracy": suite_result.cell_accuracy,
                "avg_time_ms": suite_result.avg_time_ms,
            },
            "by_category": suite_result.by_category,
        }
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {results_path}")

    if args.save_failures:
        save_failures(suite_result, args.output_dir)

    return 0 if suite_result.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
