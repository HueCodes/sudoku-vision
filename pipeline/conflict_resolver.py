"""
Conflict resolution using beam search.

When recognition conflicts are detected, this module attempts to
correct them by exploring alternative predictions.

Strategy:
1. Identify cells involved in conflicts
2. Rank by (num_conflicts, lowest_confidence)
3. Try alternative predictions for most suspicious cells
4. Use beam search to explore multiple correction paths
5. Validate each path and keep valid solutions
"""

from typing import List, Tuple, Set, Optional, Dict
from dataclasses import dataclass, field
import heapq
import copy

from validator import CellInfo, ValidationResult, validate_predictions
from constraint_resolver import ConstraintResolver


@dataclass
class CorrectionCandidate:
    """A candidate correction for a cell."""
    row: int
    col: int
    original_digit: int
    new_digit: int
    original_confidence: float
    alternative_confidence: float


@dataclass
class CorrectionPath:
    """A path of corrections being explored."""
    corrections: List[CorrectionCandidate]
    cells: List[CellInfo]
    score: float  # Lower is better

    def __lt__(self, other):
        return self.score < other.score


@dataclass
class ResolutionResult:
    """Result of conflict resolution."""
    success: bool
    cells: List[CellInfo]
    grid: List[List[int]]
    corrections_made: List[CorrectionCandidate]
    paths_explored: int
    validation_result: ValidationResult
    score: float = 0.0


class ConflictResolver:
    """Resolves recognition conflicts using beam search."""

    def __init__(
        self,
        beam_width: int = 5,
        max_corrections: int = 3,
        min_alternative_confidence: float = 0.1,
    ):
        """
        Args:
            beam_width: Number of paths to explore in parallel
            max_corrections: Maximum number of cells to correct
            min_alternative_confidence: Minimum confidence for alternative to be considered
        """
        self.beam_width = beam_width
        self.max_corrections = max_corrections
        self.min_alternative_confidence = min_alternative_confidence

    def resolve(self, cells: List[CellInfo]) -> ResolutionResult:
        """Attempt to resolve conflicts in predictions.

        Args:
            cells: List of 81 CellInfo with predictions and alternatives

        Returns:
            ResolutionResult with corrected cells if successful
        """
        # Initial validation
        validation = validate_predictions(cells)

        if validation.is_valid:
            return ResolutionResult(
                success=True,
                cells=cells,
                grid=self._build_grid(cells),
                corrections_made=[],
                paths_explored=1,
                validation_result=validation,
            )

        # Initialize beam with original state
        initial_path = CorrectionPath(
            corrections=[],
            cells=[copy.copy(c) for c in cells],
            score=self._score_path(cells, validation),
        )

        beam = [initial_path]
        paths_explored = 1
        best_valid_result = None

        # Beam search
        for depth in range(self.max_corrections):
            new_beam = []

            for path in beam:
                # Generate candidates for correction
                candidates = self._get_correction_candidates(path.cells)

                for candidate in candidates:
                    # Apply correction
                    new_cells = self._apply_correction(path.cells, candidate)
                    new_validation = validate_predictions(new_cells)
                    paths_explored += 1

                    new_path = CorrectionPath(
                        corrections=path.corrections + [candidate],
                        cells=new_cells,
                        score=self._score_path(new_cells, new_validation),
                    )

                    if new_validation.is_valid:
                        # Found valid solution
                        if best_valid_result is None or new_path.score < best_valid_result.score:
                            best_valid_result = ResolutionResult(
                                success=True,
                                cells=new_cells,
                                grid=self._build_grid(new_cells),
                                corrections_made=new_path.corrections,
                                paths_explored=paths_explored,
                                validation_result=new_validation,
                                score=new_path.score,
                            )
                    else:
                        new_beam.append(new_path)

            # If we found a valid solution, we can stop
            if best_valid_result is not None:
                return best_valid_result

            # Keep top beam_width paths
            beam = heapq.nsmallest(self.beam_width, new_beam)

            if not beam:
                break

        # No valid solution found
        # Return best attempt (fewest conflicts)
        if beam:
            best_path = min(beam, key=lambda p: p.score)
            validation = validate_predictions(best_path.cells)

            return ResolutionResult(
                success=False,
                cells=best_path.cells,
                grid=self._build_grid(best_path.cells),
                corrections_made=best_path.corrections,
                paths_explored=paths_explored,
                validation_result=validation,
            )

        # Return original if nothing worked
        return ResolutionResult(
            success=False,
            cells=cells,
            grid=self._build_grid(cells),
            corrections_made=[],
            paths_explored=paths_explored,
            validation_result=validation,
        )

    def _get_correction_candidates(self, cells: List[CellInfo]) -> List[CorrectionCandidate]:
        """Get candidates for correction ordered by likelihood."""
        validation = validate_predictions(cells)

        if validation.is_valid:
            return []

        # Build cell map
        cell_map = {(c.row, c.col): c for c in cells}

        # Count conflicts per cell
        conflict_count = {}
        for conflict in validation.conflicts:
            for r, c in conflict.cells:
                conflict_count[(r, c)] = conflict_count.get((r, c), 0) + 1

        candidates = []

        # For each cell in conflict, generate alternatives
        for (r, c), count in sorted(conflict_count.items(), key=lambda x: (-x[1], cell_map[x[0]].confidence)):
            cell = cell_map[(r, c)]

            if not cell.alternatives:
                continue

            for alt_digit, alt_conf in cell.alternatives:
                if alt_digit != cell.digit and alt_conf >= self.min_alternative_confidence:
                    candidates.append(CorrectionCandidate(
                        row=r,
                        col=c,
                        original_digit=cell.digit,
                        new_digit=alt_digit,
                        original_confidence=cell.confidence,
                        alternative_confidence=alt_conf,
                    ))

        # Sort by: more conflicts first, then lower confidence, then higher alternative confidence
        candidates.sort(key=lambda c: (
            -conflict_count.get((c.row, c.col), 0),
            c.original_confidence,
            -c.alternative_confidence
        ))

        return candidates[:10]  # Limit to top 10 candidates

    def _apply_correction(self, cells: List[CellInfo], candidate: CorrectionCandidate) -> List[CellInfo]:
        """Apply a correction to cells."""
        new_cells = []

        for cell in cells:
            if cell.row == candidate.row and cell.col == candidate.col:
                new_cell = CellInfo(
                    row=cell.row,
                    col=cell.col,
                    digit=candidate.new_digit,
                    confidence=candidate.alternative_confidence,
                    alternatives=[(candidate.original_digit, cell.confidence)] + [
                        a for a in cell.alternatives if a[0] != candidate.new_digit
                    ],
                )
                new_cells.append(new_cell)
            else:
                new_cells.append(copy.copy(cell))

        return new_cells

    def _score_path(self, cells: List[CellInfo], validation: ValidationResult) -> float:
        """Score a path (lower is better)."""
        # Penalize conflicts heavily
        conflict_penalty = validation.num_conflicts * 100

        # Penalize low average confidence
        total_conf = sum(c.confidence for c in cells if c.digit > 0)
        num_filled = sum(1 for c in cells if c.digit > 0)
        avg_conf = total_conf / num_filled if num_filled > 0 else 0
        conf_penalty = (1 - avg_conf) * 10

        return conflict_penalty + conf_penalty

    def _build_grid(self, cells: List[CellInfo]) -> List[List[int]]:
        """Build 9x9 grid from cells."""
        grid = [[0] * 9 for _ in range(9)]
        for cell in cells:
            grid[cell.row][cell.col] = cell.digit
        return grid


def resolve_conflicts(
    cells: List[CellInfo],
    beam_width: int = 5,
    max_corrections: int = 3,
) -> ResolutionResult:
    """Convenience function to resolve conflicts.

    Args:
        cells: List of 81 CellInfo with predictions
        beam_width: Beam search width
        max_corrections: Maximum corrections to attempt

    Returns:
        ResolutionResult with corrected cells
    """
    resolver = ConflictResolver(
        beam_width=beam_width,
        max_corrections=max_corrections,
    )
    return resolver.resolve(cells)


if __name__ == "__main__":
    # Test conflict resolution
    print("Testing conflict resolver...")

    # Create cells with a conflict and alternatives
    cells = []
    for i in range(81):
        r, c = i // 9, i % 9
        digit = 0
        conf = 0.9
        alts = []

        if r == 0:
            if c == 0:
                digit = 5
                conf = 0.95
                alts = [(3, 0.03), (6, 0.02)]
            elif c == 1:
                digit = 3
                conf = 0.88
                alts = [(8, 0.05), (2, 0.04)]
            elif c == 3:
                # This will create a conflict with c=0
                digit = 5
                conf = 0.6  # Lower confidence - should be corrected
                alts = [(8, 0.25), (9, 0.10)]  # Alternatives

        cells.append(CellInfo(
            row=r,
            col=c,
            digit=digit,
            confidence=conf,
            alternatives=alts,
        ))

    result = resolve_conflicts(cells)

    print(f"Success: {result.success}")
    print(f"Paths explored: {result.paths_explored}")
    print(f"Corrections made: {len(result.corrections_made)}")

    for corr in result.corrections_made:
        print(f"  ({corr.row},{corr.col}): {corr.original_digit} -> {corr.new_digit} "
              f"(conf: {corr.original_confidence:.2f} -> {corr.alternative_confidence:.2f})")

    print(f"Remaining conflicts: {result.validation_result.num_conflicts}")
