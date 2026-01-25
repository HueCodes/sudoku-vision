"""
Constraint propagation for sudoku solving.

Implements classical sudoku solving techniques to narrow down
possibilities and potentially correct recognition errors.

Techniques:
- Naked Singles: Cell with only one possible value
- Hidden Singles: Value that can only go in one cell in a unit
- Candidate Elimination: Remove impossible candidates

These techniques can resolve uncertain cells without guessing.
"""

from typing import List, Set, Dict, Tuple, Optional
from dataclasses import dataclass, field
import copy


@dataclass
class Cell:
    """Cell with candidates."""
    row: int
    col: int
    value: int  # 0 if not set
    candidates: Set[int] = field(default_factory=lambda: set(range(1, 10)))
    confidence: float = 1.0
    is_fixed: bool = False  # True if high confidence or user-verified

    def __hash__(self):
        return hash((self.row, self.col))


@dataclass
class PropagationResult:
    """Result of constraint propagation."""
    grid: List[List[int]]
    cells: List[Cell]
    cells_resolved: List[Tuple[int, int, int]]  # (row, col, value) for cells filled
    iterations: int
    is_valid: bool  # False if contradiction found
    contradiction_cell: Optional[Tuple[int, int]] = None


class ConstraintResolver:
    """Constraint propagation engine for sudoku."""

    def __init__(self, grid: List[List[int]], confidences: Optional[List[List[float]]] = None):
        """
        Args:
            grid: 9x9 grid with 0 for empty cells
            confidences: Optional 9x9 confidence values (0-1)
        """
        self.original_grid = [row[:] for row in grid]
        self.cells: List[List[Cell]] = []

        # Initialize cells
        for r in range(9):
            row = []
            for c in range(9):
                value = grid[r][c]
                conf = confidences[r][c] if confidences else 1.0

                cell = Cell(
                    row=r,
                    col=c,
                    value=value,
                    confidence=conf,
                    is_fixed=value > 0 and conf > 0.9,
                )

                if value > 0:
                    cell.candidates = {value}

                row.append(cell)
            self.cells.append(row)

        # Initialize candidates based on constraints
        self._initialize_candidates()

    def _initialize_candidates(self):
        """Remove candidates that violate constraints."""
        for r in range(9):
            for c in range(9):
                if self.cells[r][c].value > 0:
                    self._eliminate_from_peers(r, c, self.cells[r][c].value)

    def _eliminate_from_peers(self, row: int, col: int, value: int):
        """Remove value from all peers of cell."""
        # Row peers
        for c in range(9):
            if c != col:
                self.cells[row][c].candidates.discard(value)

        # Column peers
        for r in range(9):
            if r != row:
                self.cells[r][col].candidates.discard(value)

        # Box peers
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if r != row or c != col:
                    self.cells[r][c].candidates.discard(value)

    def _set_cell(self, row: int, col: int, value: int) -> bool:
        """Set a cell value and propagate constraints.

        Returns:
            True if valid, False if contradiction
        """
        cell = self.cells[row][col]

        if cell.value > 0:
            return cell.value == value

        if value not in cell.candidates:
            return False

        cell.value = value
        cell.candidates = {value}

        # Eliminate from peers
        self._eliminate_from_peers(row, col, value)

        return True

    def find_naked_singles(self) -> List[Tuple[int, int, int]]:
        """Find cells with only one candidate (naked singles)."""
        singles = []

        for r in range(9):
            for c in range(9):
                cell = self.cells[r][c]
                if cell.value == 0 and len(cell.candidates) == 1:
                    value = next(iter(cell.candidates))
                    singles.append((r, c, value))

        return singles

    def find_hidden_singles(self) -> List[Tuple[int, int, int]]:
        """Find hidden singles (value that can only go in one cell in a unit)."""
        singles = []

        # Check rows
        for r in range(9):
            for digit in range(1, 10):
                positions = []
                for c in range(9):
                    cell = self.cells[r][c]
                    if cell.value == digit:
                        break
                    if cell.value == 0 and digit in cell.candidates:
                        positions.append(c)
                else:
                    if len(positions) == 1:
                        singles.append((r, positions[0], digit))

        # Check columns
        for c in range(9):
            for digit in range(1, 10):
                positions = []
                for r in range(9):
                    cell = self.cells[r][c]
                    if cell.value == digit:
                        break
                    if cell.value == 0 and digit in cell.candidates:
                        positions.append(r)
                else:
                    if len(positions) == 1:
                        singles.append((positions[0], c, digit))

        # Check boxes
        for box in range(9):
            box_row = (box // 3) * 3
            box_col = (box % 3) * 3

            for digit in range(1, 10):
                positions = []
                found = False

                for r in range(box_row, box_row + 3):
                    for c in range(box_col, box_col + 3):
                        cell = self.cells[r][c]
                        if cell.value == digit:
                            found = True
                            break
                        if cell.value == 0 and digit in cell.candidates:
                            positions.append((r, c))
                    if found:
                        break

                if not found and len(positions) == 1:
                    r, c = positions[0]
                    singles.append((r, c, digit))

        # Remove duplicates
        return list(set(singles))

    def propagate(self, max_iterations: int = 100) -> PropagationResult:
        """Run constraint propagation until no more progress."""
        cells_resolved = []
        iterations = 0

        while iterations < max_iterations:
            iterations += 1
            made_progress = False

            # Naked singles
            naked = self.find_naked_singles()
            for r, c, value in naked:
                if self.cells[r][c].value == 0:
                    if not self._set_cell(r, c, value):
                        # Contradiction
                        return PropagationResult(
                            grid=self._get_grid(),
                            cells=self._flatten_cells(),
                            cells_resolved=cells_resolved,
                            iterations=iterations,
                            is_valid=False,
                            contradiction_cell=(r, c),
                        )
                    cells_resolved.append((r, c, value))
                    made_progress = True

            # Hidden singles
            hidden = self.find_hidden_singles()
            for r, c, value in hidden:
                if self.cells[r][c].value == 0:
                    if not self._set_cell(r, c, value):
                        return PropagationResult(
                            grid=self._get_grid(),
                            cells=self._flatten_cells(),
                            cells_resolved=cells_resolved,
                            iterations=iterations,
                            is_valid=False,
                            contradiction_cell=(r, c),
                        )
                    cells_resolved.append((r, c, value))
                    made_progress = True

            # Check for cells with no candidates (contradiction)
            for r in range(9):
                for c in range(9):
                    cell = self.cells[r][c]
                    if cell.value == 0 and len(cell.candidates) == 0:
                        return PropagationResult(
                            grid=self._get_grid(),
                            cells=self._flatten_cells(),
                            cells_resolved=cells_resolved,
                            iterations=iterations,
                            is_valid=False,
                            contradiction_cell=(r, c),
                        )

            if not made_progress:
                break

        return PropagationResult(
            grid=self._get_grid(),
            cells=self._flatten_cells(),
            cells_resolved=cells_resolved,
            iterations=iterations,
            is_valid=True,
        )

    def _get_grid(self) -> List[List[int]]:
        """Convert cells back to grid."""
        return [[self.cells[r][c].value for c in range(9)] for r in range(9)]

    def _flatten_cells(self) -> List[Cell]:
        """Flatten cells to list."""
        return [self.cells[r][c] for r in range(9) for c in range(9)]

    def get_candidates(self, row: int, col: int) -> Set[int]:
        """Get candidates for a cell."""
        return self.cells[row][col].candidates.copy()

    def try_value(self, row: int, col: int, value: int) -> bool:
        """Try setting a value and check if it leads to contradiction.

        Non-destructive: uses copy of state.
        """
        # Create copy
        resolver_copy = ConstraintResolver.__new__(ConstraintResolver)
        resolver_copy.original_grid = self.original_grid
        resolver_copy.cells = [
            [copy.copy(cell) for cell in row]
            for row in self.cells
        ]
        for r in range(9):
            for c in range(9):
                resolver_copy.cells[r][c].candidates = self.cells[r][c].candidates.copy()

        # Try setting value
        if not resolver_copy._set_cell(row, col, value):
            return False

        # Propagate
        result = resolver_copy.propagate()
        return result.is_valid


def resolve_with_constraints(
    grid: List[List[int]],
    confidences: Optional[List[List[float]]] = None,
) -> PropagationResult:
    """Convenience function to run constraint propagation.

    Args:
        grid: 9x9 sudoku grid (0 for empty)
        confidences: Optional 9x9 confidence values

    Returns:
        PropagationResult with resolved grid
    """
    resolver = ConstraintResolver(grid, confidences)
    return resolver.propagate()


if __name__ == "__main__":
    # Test constraint propagation
    print("Testing constraint resolver...")

    # Easy puzzle that can be solved with just constraint propagation
    puzzle = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]

    result = resolve_with_constraints(puzzle)

    print(f"Valid: {result.is_valid}")
    print(f"Iterations: {result.iterations}")
    print(f"Cells resolved: {len(result.cells_resolved)}")

    print("\nResolved grid:")
    for row in result.grid:
        print(" ".join(str(v) if v > 0 else "." for v in row))

    # Count remaining empty cells
    empty = sum(1 for r in result.grid for c in r if c == 0)
    print(f"\nEmpty cells remaining: {empty}")
