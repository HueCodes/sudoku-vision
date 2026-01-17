#!/usr/bin/env python3
"""
Pipeline Benchmark

Tests the full pipeline on all available test images and reports metrics.

Usage:
    python benchmark.py [--images-dir PATH]
"""

import argparse
import sys
from pathlib import Path

from run import run_pipeline, print_grid


def main():
    parser = argparse.ArgumentParser(description="Pipeline Benchmark")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "test_images",
        help="Directory containing test images",
    )
    args = parser.parse_args()

    # Find test images
    image_extensions = {".jpg", ".jpeg", ".png"}
    images = sorted([
        p for p in args.images_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ])

    if not images:
        print(f"No images found in {args.images_dir}")
        return 1

    print("=" * 60)
    print("SUDOKU VISION PIPELINE BENCHMARK")
    print("=" * 60)
    print(f"\nTesting {len(images)} images from {args.images_dir}\n")

    # Metrics
    results = []
    total_cv_time = 0
    total_ml_time = 0
    total_solver_time = 0
    total_time = 0

    for image_path in images:
        print(f"\n--- {image_path.name} ---")

        result = run_pipeline(image_path)
        results.append((image_path.name, result))

        if result.success:
            print(f"  ✓ SUCCESS")
            print(f"    CV: {result.time_cv*1000:.0f}ms, ML: {result.time_ml*1000:.0f}ms, "
                  f"Solver: {result.time_solver*1000:.0f}ms, Total: {result.time_total*1000:.0f}ms")

            total_cv_time += result.time_cv
            total_ml_time += result.time_ml
            total_solver_time += result.time_solver
            total_time += result.time_total

            # Count recognized digits
            recognized = sum(1 for p in result.predictions if p.digit > 0)
            print(f"    Recognized: {recognized}/81 cells")

            if result.low_confidence_cells:
                print(f"    Low confidence: {len(result.low_confidence_cells)} cells")

        else:
            print(f"  ✗ FAILED: {result.error}")

            if result.constraint_violations:
                print(f"    Violations: {len(result.constraint_violations)}")
                for v in result.constraint_violations[:3]:
                    print(f"      - {v}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successes = sum(1 for _, r in results if r.success)
    failures = len(results) - successes

    print(f"\nResults: {successes}/{len(results)} successful ({100*successes/len(results):.0f}%)")
    print(f"  Successes: {successes}")
    print(f"  Failures:  {failures}")

    if successes > 0:
        print(f"\nAverage timing (successful runs):")
        print(f"  CV time:     {total_cv_time/successes*1000:.0f}ms")
        print(f"  ML time:     {total_ml_time/successes*1000:.0f}ms")
        print(f"  Solver time: {total_solver_time/successes*1000:.0f}ms")
        print(f"  Total time:  {total_time/successes*1000:.0f}ms")

    # Failure analysis
    if failures > 0:
        print(f"\nFailure analysis:")
        for name, result in results:
            if not result.success:
                print(f"  {name}: {result.error}")

    print("\n" + "=" * 60)

    return 0 if successes > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
